//go:build darwin || linux

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestPiConfig(t *testing.T) {
	for _, tc := range []struct {
		context, output int
	}{
		{262144, 262144},
		{32768, 32768},
		{4096, 4096},
	} {
		t.Run(fmt.Sprint(tc.context), func(t *testing.T) {
			extension, config, err := piConfig(Paths{Data: t.TempDir()}, Status{Ready: true, Model: "flint", Engine: "mlx", Revision: "test", Context: tc.context, Endpoint: "http://127.0.0.1:11501/v1"})
			if err != nil {
				t.Fatal(err)
			}
			var c map[string]any
			if err := json.Unmarshal([]byte(config), &c); err != nil {
				t.Fatal(err)
			}
			if c["baseUrl"] != "http://127.0.0.1:11501/v1" || c["contextWindow"] != float64(tc.context) || c["maxTokens"] != float64(tc.output) {
				t.Fatal("wrong live server settings")
			}
			if _, ok := c["sampling"]; ok {
				t.Fatal("sampling belongs to the engine")
			}
			info, err := os.Stat(extension)
			if err != nil || info.Mode().Perm() != 0600 {
				t.Fatalf("extension permissions: %v", err)
			}
			if strings.Contains(config, "apiKey") {
				t.Fatal("credential belongs in the process environment")
			}
		})
	}
}

func TestClientsRejectInvalidStatus(t *testing.T) {
	for _, s := range []Status{
		{Ready: true, Model: "flint", Context: 8192, Endpoint: "https://example.com/v1"},
		{Ready: true, Model: "other", Context: 8192, Endpoint: "http://127.0.0.1:11500/v1"},
		{Ready: false, Model: "flint", Context: 8192, Endpoint: "http://127.0.0.1:11500/v1"},
		{Ready: true, Model: "flint", Context: 1024, Endpoint: "http://127.0.0.1:11500/v1"},
		{Ready: true, Model: "flint", Context: 8192, Endpoint: "http://localhost:11500/v1"},
		{Ready: true, Model: "flint", Context: 8192, Endpoint: "http://127.0.0.1/v1"},
	} {
		p := Paths{Data: t.TempDir()}
		if _, _, err := piConfig(p, s); err == nil {
			t.Fatal("Pi accepted invalid status")
		}
		if _, err := openCodeConfig(p, s); err == nil {
			t.Fatal("OpenCode accepted invalid status")
		}
	}
}

func TestPiEnvUsesRecUserDirectory(t *testing.T) {
	p := Paths{Data: t.TempDir()}
	h, err := harnessCommand(p, "pi", Status{Ready: true, Model: "flint", Context: 8192, Endpoint: "http://127.0.0.1:11500/v1"}, nil, "secret", nil)
	if err != nil {
		t.Fatal(err)
	}
	h.set[3][1] = "config"
	got := h.environ([]string{"PATH=/bin", "REC_API_KEY=old", "REC_PI_CONFIG=old", "PI_SKIP_VERSION_CHECK=0", "PI_CODING_AGENT_DIR=/custom", "PI_CODING_AGENT_SESSION_DIR=/sessions"})
	want := []string{"PATH=/bin", "PI_CODING_AGENT_DIR=" + filepath.Join(p.Data, "clients", "pi", "user"), "PI_SKIP_VERSION_CHECK=1", "REC_API_KEY=secret", "REC_PI_CONFIG=config"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("incorrect environment merge: %q", got)
	}
}

func TestPiPinIsLocked(t *testing.T) {
	var pkg struct{ Dependencies map[string]string }
	var lock struct {
		Packages map[string]struct {
			Version      string
			Dependencies map[string]string
		}
	}
	if err := json.Unmarshal(piPackage, &pkg); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(piLock, &lock); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(lock.Packages[""].Dependencies, pkg.Dependencies) {
		t.Fatal("package-lock.json is stale; run npm install --package-lock-only in cli/clients/pi")
	}
	for name, version := range pkg.Dependencies {
		if lock.Packages["node_modules/"+name].Version != version {
			t.Fatalf("%s must be pinned to an exact locked version", name)
		}
	}
}

func TestInstallPi(t *testing.T) {
	bin := t.TempDir()
	calls := filepath.Join(t.TempDir(), "calls")
	script := "#!/bin/sh\necho \"$*\" >> " + calls + "\n" +
		"cmp -s package.json \"$PI_PACKAGE\" && cmp -s package-lock.json \"$PI_LOCK\" || exit 3\n" +
		"[ -z \"$FAIL\" ] || exit 4\n" +
		"mkdir -p node_modules/.bin && touch node_modules/.bin/pi\n"
	if err := os.WriteFile(filepath.Join(bin, "npm"), []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	fixtures := t.TempDir()
	for name, data := range map[string][]byte{"PI_PACKAGE": piPackage, "PI_LOCK": piLock} {
		path := filepath.Join(fixtures, name)
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
		t.Setenv(name, path)
	}
	t.Setenv("PATH", bin+":/usr/bin:/bin")
	p := dataPaths(t.TempDir())
	if err := os.MkdirAll(p.State, 0700); err != nil {
		t.Fatal(err)
	}
	parent := filepath.Join(p.Data, "clients", "pi")
	earlier := filepath.Join(parent, "npm-0000000000000000")
	interrupted := filepath.Join(parent, ".install-interrupted")
	for _, dir := range []string{earlier, interrupted} {
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
	}

	t.Setenv("FAIL", "1")
	if _, err := installPi(p, io.Discard); err == nil {
		t.Fatal("failed npm ci was accepted")
	}
	entries, err := os.ReadDir(parent)
	if err != nil || len(entries) != 1 || entries[0].Name() != filepath.Base(earlier) {
		t.Fatalf("want only the earlier install after a failed install, got %v %v", entries, err)
	}

	t.Setenv("FAIL", "")
	first, err := installPi(p, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	second, err := installPi(p, io.Discard)
	if err != nil || second != first {
		t.Fatalf("reinstall: %v %s %s", err, first, second)
	}
	if _, err := os.Stat(earlier); err != nil {
		t.Fatal("an earlier install may still be running and must be kept")
	}
	data, err := os.ReadFile(calls)
	if err != nil || string(data) != "ci --no-audit --no-fund --loglevel=error\nci --no-audit --no-fund --loglevel=error\n" {
		t.Fatalf("want one failed and one successful npm ci, got %q", data)
	}
}
