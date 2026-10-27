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

func TestServeLifecycle(t *testing.T) {
	if (runtime.GOOS != "darwin" || runtime.GOARCH != "arm64") &&
		(runtime.GOOS != "linux" || runtime.GOARCH != "amd64") {
		t.Skip("serving lifecycle target unavailable")
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
	engine, backend, serverName := "mlx", "metal", runtimeServer
	if runtime.GOOS == "linux" {
		engine, backend, serverName = "flint-native", "hip", "flint-serve"
	}
	if err := os.WriteFile(filepath.Join(runtimeDir, serverName), server, 0700); err != nil {
		t.Fatal(err)
	}
	m := Manifest{Engine: engine, Schema: 1, Model: "test", Revision: "test", Development: true, OS: runtime.GOOS, Arch: runtime.GOARCH, Backend: backend,
		Context: 2048, MinRAM: 1, Entry: "config.json",
		Files:      []File{{Name: "config.json", URL: weights.URL + "/config.json", Size: int64(len(config)), SHA256: digest(config)}},
		RuntimeSHA: digest([]byte("runtime")), RuntimeFiles: []File{{Name: serverName, Size: int64(len(server)), SHA256: digest(server), Executable: true}}}
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
	go func() { done <- serve(context.Background(), p, m, runtimeDir, "", port, &out) }()
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
	if !strings.Contains(out.String(), "test is ready.") || s.Endpoint != fmt.Sprintf("http://127.0.0.1:%d/v1", port) || s.Engine != engine {
		t.Fatalf("wrong readiness report %+v:\n%s", s, out.String())
	}

	var again bytes.Buffer
	if err := serve(context.Background(), p, m, runtimeDir, "", port, &again); err != nil || !strings.Contains(again.String(), "test is already running") {
		t.Fatalf("identical run did not reuse the server: %v %s", err, again.String())
	}
	changed := m
	changed.Context, changed.MaxContext = 4096, 4096
	if err := serve(context.Background(), p, changed, runtimeDir, "", port, io.Discard); err == nil {
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
