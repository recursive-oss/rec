//go:build darwin || linux

package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// serve re-executes its own binary as the supervisor; in tests that is the test binary.
func TestMain(m *testing.M) {
	if len(os.Args) > 1 && os.Args[1] == "_supervise" {
		if err := supervise(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

type lockedBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.b.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.b.String()
}

func TestParseHostAndClientEndpoint(t *testing.T) {
	for _, tc := range []struct {
		host     string
		endpoint string
	}{
		{"127.0.0.1", "http://127.0.0.1:11500/v1"},
		{"0.0.0.0", "http://127.0.0.1:11500/v1"},
		{"::", "http://[::1]:11500/v1"},
		{"192.168.1.5", "http://192.168.1.5:11500/v1"},
		{"fe80::1", "http://[fe80::1]:11500/v1"},
	} {
		if got := clientEndpoint(tc.host, 11500); got != tc.endpoint {
			t.Fatalf("clientEndpoint(%q, 11500) = %q, want %q", tc.host, got, tc.endpoint)
		}
	}
	for host, canonical := range map[string]string{"0:0::0": "::", "::ffff:127.0.0.1": "127.0.0.1"} {
		if addr, err := parseHost(host); err != nil || addr.String() != canonical {
			t.Fatalf("parseHost(%q) = %v, %v; want %s", host, addr, err, canonical)
		}
	}
	for _, bad := range []string{"localhost", "", "example.com", "fe80::1%en0"} {
		if _, err := parseHost(bad); err == nil {
			t.Fatalf("parseHost(%q) accepted a non-IP address", bad)
		}
	}
	for _, good := range []string{"127.0.0.1", "0.0.0.0", "::", "192.168.1.5", "fe80::1"} {
		if _, err := parseHost(good); err != nil {
			t.Fatalf("parseHost(%q) rejected a valid IP: %v", good, err)
		}
	}
}

func TestServeLifecycle(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("serving requires Apple Silicon")
	}
	// Unix socket paths are limited to 104 bytes on macOS; t.TempDir is too long.
	root, err := os.MkdirTemp("/tmp", "rec-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(root)
	p := dataPaths(filepath.Join(root, "data"))
	for _, dir := range []string{p.Data, p.State, p.Config} {
		if err := os.Mkdir(dir, 0700); err != nil {
			t.Fatal(err)
		}
	}
	config := []byte(`{"model_type": "test"}`)
	weights := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write(config) }))
	defer weights.Close()
	server, err := os.ReadFile("testdata/fake-rec-mlx")
	if err != nil {
		t.Fatal(err)
	}
	runtimeDir := filepath.Join(root, "runtime")
	if err := os.Mkdir(runtimeDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runtimeDir, runtimeServer), server, 0700); err != nil {
		t.Fatal(err)
	}
	m := Manifest{Engine: "mlx", Schema: 1, Model: "test", Revision: "test", Development: true, OS: "darwin", Arch: "arm64", Backend: "metal",
		Context: 2048, MinRAM: 1, Entry: "config.json",
		Files:      []File{{Name: "config.json", URL: weights.URL + "/config.json", Size: int64(len(config)), SHA256: digest(config)}},
		RuntimeSHA: digest([]byte("runtime")), RuntimeFiles: []File{{Name: runtimeServer, Size: int64(len(server)), SHA256: digest(server), Executable: true}}}
	if _, err := ensureModelWithClient(context.Background(), p, m, io.Discard, weights.Client()); err != nil {
		t.Fatal(err)
	}
	weights.Close()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()

	var out lockedBuffer
	done := make(chan error, 1)
	go func() { done <- serve(context.Background(), p, m, runtimeDir, "127.0.0.1", port, &out) }()
	deadline := time.Now().Add(2 * time.Minute)
	var s Status
	for !s.Ready {
		select {
		case err := <-done:
			t.Fatalf("serve exited before readiness: %v\n%s", err, out.String())
		default:
		}
		if time.Now().After(deadline) {
			t.Fatalf("not ready:\n%s", out.String())
		}
		time.Sleep(100 * time.Millisecond)
		s, _ = control(p, "status")
	}
	if !strings.Contains(out.String(), "test is ready.") || s.Endpoint != fmt.Sprintf("http://127.0.0.1:%d/v1", port) || s.Engine != "mlx" {
		t.Fatalf("wrong readiness report %+v:\n%s", s, out.String())
	}
	if s.Listen != fmt.Sprintf("127.0.0.1:%d", port) {
		t.Fatalf("wrong listen address: %q", s.Listen)
	}

	var again bytes.Buffer
	if err := serve(context.Background(), p, m, runtimeDir, "127.0.0.1", port, &again); err != nil || !strings.Contains(again.String(), "test is already running") {
		t.Fatalf("identical run did not reuse the server: %v %s", err, again.String())
	}
	changed := m
	changed.Context, changed.MaxContext = 4096, 4096
	if err := serve(context.Background(), p, changed, runtimeDir, "127.0.0.1", port, io.Discard); err == nil {
		t.Fatal("a different context reused the running server")
	}

	key, err := token(p, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, auth := range []string{"", "Bearer " + key} {
		req, _ := http.NewRequest("POST", s.Endpoint+"/chat/completions", strings.NewReader(`{"model":"test","messages":[]}`))
		if auth != "" {
			req.Header.Set("Authorization", auth)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if (resp.StatusCode == 200) != (auth != "") {
			t.Fatalf("authorization %q returned HTTP %d", auth, resp.StatusCode)
		}
	}

	if _, err := control(p, "stop"); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("serve did not stop")
	}
	if _, err := control(p, "status"); err == nil {
		t.Fatal("control socket outlived the server")
	}
	if conn, err := net.Dial("tcp", fmt.Sprintf("127.0.0.1:%d", port)); err == nil {
		conn.Close()
		t.Fatal("engine outlived the server")
	}
}
