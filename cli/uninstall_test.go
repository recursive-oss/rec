//go:build darwin || linux

package main

import (
	"bytes"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func installedHome(t *testing.T) (string, Paths) {
	// Unix socket paths are limited to 104 bytes on macOS; t.TempDir is too long.
	home, err := os.MkdirTemp("/tmp", "rec-test-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(home) })
	t.Setenv("HOME", home)
	p := dataPaths(filepath.Join(home, ".local/share/rec"))
	release := filepath.Join(p.Data, "releases/0.1.0")
	for _, dir := range []string{p.State, p.Config, release, filepath.Join(p.Data, "models/m"), filepath.Join(home, ".local/bin")} {
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
	}
	for name, body := range map[string]string{"releases/0.1.0/rec": "rec", "models/m/weights": strings.Repeat("w", 1000)} {
		if err := os.WriteFile(filepath.Join(p.Data, name), []byte(body), 0700); err != nil {
			t.Fatal(err)
		}
	}
	return home, p
}

func link(t *testing.T, target, name string) {
	if err := os.Symlink(target, name); err != nil {
		t.Fatal(err)
	}
}

func TestUninstallRemovesManagedFiles(t *testing.T) {
	home, p := installedHome(t)
	bin := filepath.Join(home, ".local/bin")
	link(t, filepath.Join(p.Data, "releases/0.1.0/rec"), filepath.Join(bin, "rec"))

	var out bytes.Buffer
	if err := uninstall(p, nil, strings.NewReader("n\n"), &out); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{filepath.Join(bin, "rec"), p.Data, "Nothing removed."} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output lacks %q:\n%s", want, out.String())
		}
	}
	if _, err := os.Stat(p.Data); err != nil {
		t.Fatal("declining removed data")
	}

	if err := uninstall(p, nil, strings.NewReader("y\n"), &out); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{filepath.Join(bin, "rec"), p.Data} {
		if _, err := os.Lstat(path); !os.IsNotExist(err) {
			t.Fatalf("%s was not removed", path)
		}
	}
}

func TestUninstallKeepsOtherPrograms(t *testing.T) {
	home, p := installedHome(t)
	bin := filepath.Join(home, ".local/bin")
	if err := os.WriteFile(filepath.Join(bin, "rec"), []byte("other"), 0700); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := uninstall(p, []string{"-y"}, strings.NewReader(""), &out); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out.String(), bin) {
		t.Fatalf("listed another program:\n%s", out.String())
	}
	if _, err := os.Lstat(filepath.Join(bin, "rec")); err != nil {
		t.Fatal("removed another program's rec")
	}
	if _, err := os.Stat(p.Data); !os.IsNotExist(err) {
		t.Fatal("data was not removed")
	}
}

func TestUninstallLeavesHomebrewCommandToBrew(t *testing.T) {
	_, p := installedHome(t)
	installWithHomebrew(t)
	var out bytes.Buffer
	if err := uninstall(p, []string{"-y"}, strings.NewReader(""), &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "brew uninstall rec") || strings.Contains(out.String(), "Recursive CLI PATH") {
		t.Fatalf("output:\n%s", out.String())
	}
	if _, err := os.Stat(p.Data); !os.IsNotExist(err) {
		t.Fatal("data was not removed")
	}
}

func TestUninstallStopsRunningServer(t *testing.T) {
	_, p := installedHome(t)
	l, err := lock(filepath.Join(p.State, "instance.lock"))
	if err != nil {
		t.Fatal(err)
	}
	listener, err := net.Listen("unix", filepath.Join(p.State, "control.sock"))
	if err != nil {
		t.Fatal(err)
	}
	stopped := make(chan struct{})
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "POST" && r.URL.Path == "/stop" {
			w.Write([]byte("{}"))
			unlock(l)
			close(stopped)
		}
	})}
	go func() { _ = srv.Serve(listener) }()
	defer srv.Close()
	var out bytes.Buffer
	if err := uninstall(p, []string{"-y"}, strings.NewReader(""), &out); err != nil {
		t.Fatal(err)
	}
	select {
	case <-stopped:
	default:
		t.Fatal("server was not asked to stop")
	}
	if !strings.Contains(out.String(), "Stopping the running model") {
		t.Fatalf("output:\n%s", out.String())
	}
	if _, err := os.Stat(p.Data); !os.IsNotExist(err) {
		t.Fatal("data was not removed")
	}
}
