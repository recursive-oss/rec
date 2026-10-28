//go:build darwin || linux

package main

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestOmpConfig(t *testing.T) {
	p := Paths{Data: t.TempDir()}
	user := filepath.Join(p.Data, "clients", "omp", "user")
	s := Status{Ready: true, Model: "flint", Context: 8192, Endpoint: "http://127.0.0.1:11500/v1"}
	path, err := ompConfig(p, s, user)
	if err != nil {
		t.Fatal(err)
	}
	if path != ompConfigPath(p) {
		t.Fatalf("wrong overlay path: %s", path)
	}
	settings, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	models, err := os.ReadFile(filepath.Join(user, "models.yml"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"default: rec/flint",
		"modelFallback: false",
		"enabledProviders:\n  - rec",
	} {
		if !strings.Contains(string(settings), want) {
			t.Fatalf("omp settings missing %q:\n%s", want, settings)
		}
	}
	for _, want := range []string{
		"baseUrl: http://127.0.0.1:11500/v1",
		"apiKey: REC_API_KEY",
		"api: openai-completions",
		"contextWindow: 8192",
		"maxTokens: 8192",
		"maxTokensField: max_tokens",
	} {
		if !strings.Contains(string(models), want) {
			t.Fatalf("omp models.yml missing %q:\n%s", want, models)
		}
	}
	if strings.Contains(string(settings)+string(models), "openrouter") {
		t.Fatal("omp config must not reference outside providers")
	}
}

func TestOmpEnvUsesRecUserDirectory(t *testing.T) {
	p := Paths{Data: t.TempDir()}
	h, err := harnessCommand(p, "omp", Status{Ready: true, Model: "flint", Context: 8192, Endpoint: "http://127.0.0.1:11500/v1"}, nil, "secret", []string{"-p", "hi"})
	if err != nil {
		t.Fatal(err)
	}
	if got := h.argv[0]; got != ompExecutable(p) {
		t.Fatalf("wrong omp executable: %s", got)
	}
	if !reflect.DeepEqual(h.argv[1:4], []string{"--config", ompConfigPath(p), "--model"}) {
		t.Fatalf("wrong omp argv prefix: %q", h.argv)
	}
	if h.argv[4] != "rec/flint" || !reflect.DeepEqual(h.argv[5:], []string{"-p", "hi"}) {
		t.Fatalf("wrong omp model/args: %q", h.argv)
	}
	got := h.environ([]string{"PATH=/bin", "REC_API_KEY=old", "PI_CODING_AGENT_DIR=/custom", "PI_CODING_AGENT_SESSION_DIR=/sessions"})
	want := []string{"PATH=/bin", "PI_CODING_AGENT_DIR=" + filepath.Join(p.Data, "clients", "omp", "user"), "REC_API_KEY=secret"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("incorrect environment merge: %q", got)
	}
}

func ompConfigPath(p Paths) string {
	return filepath.Join(p.Data, "clients", "omp", Version, "rec-omp.yml")
}

func TestOmpPinIsLocked(t *testing.T) {
	var pkg struct{ Dependencies map[string]string }
	var lock struct {
		Packages map[string]struct {
			Version      string
			Dependencies map[string]string
		}
	}
	if err := json.Unmarshal(ompPackage, &pkg); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(ompLock, &lock); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(lock.Packages[""].Dependencies, pkg.Dependencies) {
		t.Fatal("package-lock.json is stale; run npm install --package-lock-only in cli/clients/omp")
	}
	for name, version := range pkg.Dependencies {
		if lock.Packages["node_modules/"+name].Version != version {
			t.Fatalf("%s must be pinned to an exact locked version", name)
		}
	}
}

func TestInstallOmp(t *testing.T) {
	bin := t.TempDir()
	calls := filepath.Join(t.TempDir(), "calls")
	script := "#!/bin/sh\necho \"$*\" >> " + calls + "\n" +
		"cmp -s package.json \"$OMP_PACKAGE\" && cmp -s package-lock.json \"$OMP_LOCK\" || exit 3\n" +
		"[ -z \"$FAIL\" ] || exit 4\n" +
		"mkdir -p node_modules/.bin && touch node_modules/.bin/omp\n"
	for _, name := range []string{"npm", "bun"} {
		if err := os.WriteFile(filepath.Join(bin, name), []byte(script), 0700); err != nil {
			t.Fatal(err)
		}
	}
	fixtures := t.TempDir()
	for name, data := range map[string][]byte{"OMP_PACKAGE": ompPackage, "OMP_LOCK": ompLock} {
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
	parent := filepath.Join(p.Data, "clients", "omp")
	earlier := filepath.Join(parent, "npm-0000000000000000")
	interrupted := filepath.Join(parent, ".install-interrupted")
	for _, dir := range []string{earlier, interrupted} {
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
	}

	t.Setenv("FAIL", "1")
	if _, err := installOmp(p, io.Discard); err == nil {
		t.Fatal("failed npm ci was accepted")
	}
	entries, err := os.ReadDir(parent)
	if err != nil || len(entries) != 1 || entries[0].Name() != filepath.Base(earlier) {
		t.Fatalf("want only the earlier install after a failed install, got %v %v", entries, err)
	}

	t.Setenv("FAIL", "")
	first, err := installOmp(p, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	second, err := installOmp(p, io.Discard)
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

func TestInstallOmpRequiresBun(t *testing.T) {
	t.Setenv("PATH", t.TempDir()+":/usr/bin:/bin")
	p := dataPaths(t.TempDir())
	if err := os.MkdirAll(p.State, 0700); err != nil {
		t.Fatal(err)
	}
	if _, err := installOmp(p, io.Discard); err == nil || !strings.Contains(err.Error(), "Bun") {
		t.Fatalf("want a Bun requirement error, got %v", err)
	}
}
