package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestProgressFailureAndActivityCleanup(t *testing.T) {
	var output bytes.Buffer
	progress := newTransfer(&output, "Verifying Flint", "Verified Flint", 0, 5)
	_, _ = progress.Write([]byte("wrong"))
	progress.fail("SHA-256 mismatch: model.safetensors")
	if !strings.Contains(output.String(), "✗ Verifying Flint failed  SHA-256 mismatch") || strings.Contains(output.String(), "✓") {
		t.Fatalf("checksum failure reported as success: %s", output.String())
	}
	output.Reset()
	p := startActivity(&output, "Loading Flint", "Loaded Flint")
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := range 100 {
			p.phase(fmt.Sprint("Checking generation ", i))
		}
	}()
	p.cancel()
	wg.Wait()
	before := output.String()
	p.phase("Late callback")
	if output.String() != before || !strings.HasSuffix(before, "– Loading Flint cancelled\n") || strings.ContainsAny(before, "\r\x1b%") {
		t.Fatalf("activity leaked output after cancellation or fabricated progress: %q", output.String())
	}
}

func TestCachedModelVerificationShowsOneProgressBar(t *testing.T) {
	config, weights := []byte("{}"), bytes.Repeat([]byte("w"), 2_000_000)
	bodies := map[string][]byte{"/config.json": config, "/weights": weights}
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write(bodies[r.URL.Path]) }))
	defer s.Close()
	m := Manifest{Model: "test", Entry: "config.json", Files: []File{
		{Name: "config.json", URL: s.URL + "/config.json", Size: int64(len(config)), SHA256: digest(config)},
		{Name: "weights", URL: s.URL + "/weights", Size: int64(len(weights)), SHA256: digest(weights)},
	}}
	p := Paths{Data: t.TempDir()}
	if _, err := ensureModelWithClient(context.Background(), p, m, io.Discard, s.Client()); err != nil {
		t.Fatal(err)
	}
	offline := &http.Client{Transport: &http.Transport{Proxy: func(*http.Request) (*url.URL, error) { return nil, errors.New("offline") }}}
	var output bytes.Buffer
	if _, err := ensureModelWithClient(context.Background(), p, m, &output, offline); err != nil {
		t.Fatalf("verified cache required network: %v", err)
	}
	if output.Len() != 0 {
		t.Fatalf("verified cache was hashed again:\n%s", output.String())
	}

	root := filepath.Join(p.Data, "models", modelDigest(m))
	path := filepath.Join(root, "weights")
	if err := os.Remove(filepath.Join(root, "verified.json")); err != nil {
		t.Fatal(err)
	}
	if _, err := ensureModelWithClient(context.Background(), p, m, &output, offline); err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSuffix(output.String(), "\n"), "\n")
	if len(lines) != 2 || lines[0] != "  Verifying Test    0%  0.0 MB / 2.0 MB" || !strings.HasPrefix(lines[1], "  ✓ Verified Test  2.0 MB in ") {
		t.Fatalf("unverified cache output:\n%s", output.String())
	}

	if err := os.WriteFile(path, bytes.Repeat([]byte("x"), len(weights)), 0600); err != nil {
		t.Fatal(err)
	}
	output.Reset()
	if _, err := ensureModelWithClient(context.Background(), p, m, &output, s.Client()); err != nil {
		t.Fatal(err)
	}
	got := output.String()
	if !strings.Contains(got, "  ✗ Verifying Test failed  SHA-256 mismatch: weights; downloading again\n") || !strings.Contains(got, "  ✓ Downloaded Test  2.0 MB in ") {
		t.Fatalf("corrupt cache output:\n%s", output.String())
	}
	if err := verifyFile(path, m.Files[1]); err != nil {
		t.Fatal(err)
	}
}

func TestDoctorRehashesVerifiedModel(t *testing.T) {
	weights := bytes.Repeat([]byte("w"), 1000)
	m := Manifest{Model: "test", Entry: "weights", Files: []File{{Name: "weights", Size: int64(len(weights)), SHA256: digest(weights)}}}
	p := Paths{Data: t.TempDir()}
	root := filepath.Join(p.Data, "models", modelDigest(m))
	path := filepath.Join(root, "weights")
	if err := os.MkdirAll(root, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, weights, 0600); err != nil {
		t.Fatal(err)
	}
	if id, err := verifyModel(p, m, io.Discard); err != nil || id != modelDigest(m) {
		t.Fatalf("verifyModel = %q, %v", id, err)
	}
	stamp, err := statStamp(path, m.Files[0].SHA256)
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteAt([]byte("x"), 0); err != nil {
		t.Fatal(err)
	}
	file.Close()
	verified, err := loadVerified(root)
	if err != nil {
		t.Fatal(err)
	}
	verified.stamps["weights"] = stamp
	if err := verified.save(); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyModel(p, m, io.Discard); err == nil || !strings.Contains(err.Error(), "SHA-256 mismatch") {
		t.Fatalf("doctor trusted a corrupt file: %v", err)
	}
	if verified, err = loadVerified(root); err != nil || len(verified.stamps) != 0 {
		t.Fatalf("corrupt file kept its stamp: %v %v", verified.stamps, err)
	}
}

func TestEmbeddedManifest(t *testing.T) {
	b, err := assets.ReadFile("assets/manifest.json")
	if os.IsNotExist(err) {
		t.Skip("unpackaged development build")
	}
	m, err := decodeManifest(b)
	if err != nil {
		t.Fatal(err)
	}
	data, err := assets.ReadFile("assets/runtime.tar.gz")
	if err != nil {
		t.Fatal(err)
	}
	if err := unpack(data, t.TempDir(), m); err != nil {
		t.Fatal(err)
	}
}

func TestDownloadResume(t *testing.T) {
	body := []byte("a complete immutable model")
	rangeSeen := false
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") == "bytes=5-" {
			rangeSeen = true
			w.Header().Set("Content-Range", "bytes 5-25/26")
			w.WriteHeader(206)
			w.Write(body[5:])
			return
		}
		w.Write(body)
	}))
	defer server.Close()
	f := File{Name: "model.safetensors", URL: server.URL, Size: int64(len(body)), SHA256: digest(body)}
	dest := filepath.Join(t.TempDir(), f.Name)
	if e := os.WriteFile(dest+".partial", body[:5], 0600); e != nil {
		t.Fatal(e)
	}
	if e := download(context.Background(), server.Client(), f, dest, discardProgress(), 0); e != nil {
		t.Fatal(e)
	}
	if !rangeSeen {
		t.Fatal("did not resume")
	}
	if e := verifyFile(dest, f); e != nil {
		t.Fatal(e)
	}
}

func TestDownloadIgnoredRange(t *testing.T) {
	body := []byte("model body")
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write(body) }))
	defer s.Close()
	f := File{Name: "model.safetensors", URL: s.URL, Size: int64(len(body)), SHA256: digest(body)}
	dest := filepath.Join(t.TempDir(), f.Name)
	if err := os.WriteFile(dest+".partial", body[:2], 0600); err != nil {
		t.Fatal(err)
	}
	if e := download(context.Background(), s.Client(), f, dest, discardProgress(), 0); e != nil {
		t.Fatal(e)
	}
	if e := verifyFile(dest, f); e != nil {
		t.Fatal(e)
	}
}

func TestDownloadRejectsCorruption(t *testing.T) {
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("wrong")) }))
	defer s.Close()
	f := File{Name: "model.safetensors", URL: s.URL, Size: 5, SHA256: digest([]byte("right"))}
	dest := filepath.Join(t.TempDir(), f.Name)
	if e := download(context.Background(), s.Client(), f, dest, discardProgress(), 0); e == nil || !strings.Contains(e.Error(), "SHA-256") {
		t.Fatalf("expected hash failure: %v", e)
	}
	if _, e := os.Stat(dest); !os.IsNotExist(e) {
		t.Fatal("corrupt artifact was promoted")
	}
}

func TestDownloadRejectsBadRange(t *testing.T) {
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Range", "bytes 0-4/5")
		w.WriteHeader(206)
		w.Write([]byte("right"))
	}))
	defer s.Close()
	f := File{Name: "model.safetensors", URL: s.URL, Size: 5, SHA256: digest([]byte("right"))}
	dest := filepath.Join(t.TempDir(), f.Name)
	if err := os.WriteFile(dest+".partial", []byte("ri"), 0600); err != nil {
		t.Fatal(err)
	}
	if e := download(context.Background(), s.Client(), f, dest, discardProgress(), 0); e == nil {
		t.Fatal("accepted wrong range offset")
	}
}

func TestRuntimeRejectsTraversalAndSymlinks(t *testing.T) {
	for _, tc := range []struct {
		name string
		kind byte
	}{{"../escape", tar.TypeReg}, {runtimeServer, tar.TypeSymlink}} {
		t.Run(tc.name, func(t *testing.T) {
			var b bytes.Buffer
			gz := gzip.NewWriter(&b)
			tw := tar.NewWriter(gz)
			h := &tar.Header{Name: tc.name, Typeflag: tc.kind, Mode: 0700, Linkname: "/tmp/escape"}
			if tc.kind == tar.TypeReg {
				h.Size = 1
			}
			if err := tw.WriteHeader(h); err != nil {
				t.Fatal(err)
			}
			if h.Size > 0 {
				if _, err := tw.Write([]byte("x")); err != nil {
					t.Fatal(err)
				}
			}
			tw.Close()
			gz.Close()
			m := Manifest{RuntimeSHA: digest(b.Bytes()), RuntimeFiles: []File{{Name: runtimeServer, Size: 1, SHA256: digest([]byte("x")), Executable: true}}}
			if e := unpack(b.Bytes(), t.TempDir(), m); e == nil {
				t.Fatal("unsafe archive accepted")
			}
		})
	}
}

func TestKeyAndLock(t *testing.T) {
	p := Paths{Data: t.TempDir(), State: t.TempDir(), Config: t.TempDir()}
	a, e := token(p, true)
	if e != nil {
		t.Fatal(e)
	}
	b, e := token(p, false)
	if e != nil || a != b {
		t.Fatal("key was not persisted")
	}
	st, e := os.Stat(filepath.Join(p.Config, "api-key"))
	if e != nil || st.Mode().Perm() != 0600 {
		t.Fatal("key permissions")
	}
	name := filepath.Join(p.State, "instance.lock")
	l, e := lock(name)
	if e != nil {
		t.Fatal(e)
	}
	if l2, e := lock(name); e == nil {
		unlock(l2)
		t.Fatal("lock allowed second owner")
	}
	unlock(l)
	l, e = lock(name)
	if e != nil {
		t.Fatal(e)
	}
	unlock(l)
}

func TestProbeRequiresGeneration(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			w.WriteHeader(401)
			return
		}
		switch r.URL.Path {
		case "/health":
			io.WriteString(w, `{"status":"ok"}`)
		case "/v1/models":
			io.WriteString(w, `{"data":[{"id":"test"}]}`)
		default:
			io.WriteString(w, `{"choices":[]}`)
		}
	}))
	defer s.Close()
	if e := probe(context.Background(), s.URL, "test-key", "test"); e == nil {
		t.Fatal("empty generation passed readiness")
	}
}

func TestManifestSchemaAndPaths(t *testing.T) {
	valid := func() Manifest {
		return Manifest{Engine: "mlx", Schema: 1, Model: "test", Revision: "sha", Development: true, OS: "darwin", Arch: "arm64", Backend: "metal", Context: 2048, MinRAM: 1, Files: []File{{Name: "config.json", Size: 1, SHA256: digest([]byte("x")), URL: "https://example.com/config.json"}}, Entry: "config.json", RuntimeSHA: digest([]byte("r")), RuntimeFiles: []File{{Name: runtimeServer, Size: 1, SHA256: digest([]byte("r")), Executable: true}}}
	}
	if _, e := decodeManifest(mustJSON(t, valid())); e != nil {
		t.Fatal(e)
	}
	for name, change := range map[string]func(*Manifest){
		"other engine":       func(m *Manifest) { m.Engine = "llama.cpp" },
		"other OS":           func(m *Manifest) { m.OS = "linux" },
		"other backend":      func(m *Manifest) { m.Backend = "cuda" },
		"unsafe filename":    func(m *Manifest) { m.Files[0].Name = "../config.json" },
		"plain HTTP":         func(m *Manifest) { m.Files[0].URL = "http://example.com/config.json" },
		"missing entry":      func(m *Manifest) { m.Entry = "model.safetensors" },
		"no server":          func(m *Manifest) { m.RuntimeFiles[0].Name = "server" },
		"unqualified model":  func(m *Manifest) { m.Development = false },
		"context over limit": func(m *Manifest) { m.MaxContext = 1024 },
	} {
		t.Run(name, func(t *testing.T) {
			m := valid()
			change(&m)
			if _, e := decodeManifest(mustJSON(t, m)); e == nil {
				t.Fatal("invalid manifest accepted")
			}
		})
	}
	var extra map[string]any
	if err := json.Unmarshal(mustJSON(t, valid()), &extra); err != nil {
		t.Fatal(err)
	}
	extra["gpu_families"] = []string{"NVIDIA GB10"}
	if _, e := decodeManifest(mustJSON(t, extra)); e == nil {
		t.Fatal("unknown field accepted")
	}
}

func TestAppleChipAllowlist(t *testing.T) {
	for chip, ok := range map[string]bool{"Apple M3 Max": true, "Apple M3 Pro": false, "Apple M4 Pro": true, "Apple M5 Pro": true, "Apple M5 Max": true, "Apple M4 Max": false, "Apple M4": false, "Apple M1 Pro": false} {
		if (checkAppleChip(chip) == nil) != ok {
			t.Fatalf("%s: want supported=%t", chip, ok)
		}
	}
}

func discardProgress() *transfer { return newTransfer(io.Discard, "Downloading", "Downloaded", 0, 0) }

func TestStampRejectsRestoredMtime(t *testing.T) {
	weights := bytes.Repeat([]byte("w"), 1000)
	f := File{Name: "weights", Size: int64(len(weights)), SHA256: digest(weights)}
	root := t.TempDir()
	path := filepath.Join(root, f.Name)
	if err := os.WriteFile(path, weights, 0600); err != nil {
		t.Fatal(err)
	}
	verified, err := loadVerified(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := verified.verify(f, io.Discard); err != nil || !verified.trusted(f) {
		t.Fatalf("verified file is not trusted: %v", err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(20 * time.Millisecond)
	if err := os.WriteFile(path, bytes.Repeat([]byte("x"), len(weights)), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, st.ModTime(), st.ModTime()); err != nil {
		t.Fatal(err)
	}
	if verified.trusted(f) {
		t.Fatal("stamp trusted a rewritten file with a restored mtime")
	}
}
