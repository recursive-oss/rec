//go:build darwin || linux

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestOpenCodeV2Config(t *testing.T) {
	for _, tc := range []struct {
		context, output int
	}{
		{262144, 262144},
		{32768, 32768},
		{4096, 4096},
	} {
		t.Run(fmt.Sprint(tc.context), func(t *testing.T) {
			p := Paths{Data: t.TempDir()}
			config, err := openCodeConfig(p, Status{Ready: true, Model: "flint", Engine: "mlx", Revision: "test", Context: tc.context, Endpoint: "http://127.0.0.1:11500/v1"})
			if err != nil {
				t.Fatal(err)
			}
			var c map[string]any
			if err := json.Unmarshal([]byte(config), &c); err != nil {
				t.Fatal(err)
			}
			for _, legacy := range []string{"provider", "plugin", "small_model", "enabled_providers"} {
				if _, ok := c[legacy]; ok {
					t.Fatalf("legacy v1 field %s", legacy)
				}
			}
			r := c["providers"].(map[string]any)["rec"].(map[string]any)
			if r["package"] != "@opencode/ai/providers/openai-compatible" || r["settings"].(map[string]any)["apiKey"] != "{env:REC_API_KEY}" {
				t.Fatal("wrong provider package or credential reference")
			}
			m := r["models"].(map[string]any)["flint"].(map[string]any)
			want := map[string]any{"max_completion_tokens": float64(tc.output)}
			if !reflect.DeepEqual(m["body"], want) {
				t.Fatalf("body: got %v, want %v", m["body"], want)
			}
			limits := m["limit"].(map[string]any)
			if limits["context"] != float64(tc.context) || limits["output"] != float64(tc.output) || c["compaction"] != nil {
				t.Fatal("lost server context/output budget")
			}
			if m["capabilities"].(map[string]any)["tools"] != true || c["agents"].(map[string]any)["title"].(map[string]any)["model"] != "rec/flint" {
				t.Fatal("tools or auxiliary model missing")
			}
			title := c["agents"].(map[string]any)["title"].(map[string]any)
			if title["disabled"] != true {
				t.Fatal("competing title generation must be disabled")
			}
			instructions := c["instructions"].([]any)[0].(string)
			data, err := os.ReadFile(instructions)
			if err != nil || strings.Contains(string(data), "filePath") {
				t.Fatalf("stale v1 tool guidance: %v", err)
			}
			if _, err := os.Stat(filepath.Join(filepath.Dir(instructions), "flint-tools.mjs")); !os.IsNotExist(err) {
				t.Fatal("v1 tool adapter should not be installed")
			}
		})
	}
}

func TestOpenCodeV2Args(t *testing.T) {
	for _, tc := range []struct{ in, want []string }{
		{nil, []string{"--standalone"}},
		{[]string{"--version"}, []string{"--standalone", "--version"}},
		{[]string{"/tmp/my project"}, []string{"--standalone", "/tmp/my project"}},
		{[]string{"run", "--format", "json", "fix it"}, []string{"run", "--standalone", "--model", "rec/flint", "--format", "json", "fix it"}},
	} {
		if got := openCodeArgs(tc.in); !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("%v: got %v, want %v", tc.in, got, tc.want)
		}
	}
}

func TestOpenCodeEnvUsesRecUserDirectory(t *testing.T) {
	user := []string{"PATH=/bin", "XDG_DATA_HOME=/xdg", "REC_API_KEY=old", "OPENCODE_CONFIG=old.json", "OPENCODE_CONFIG_CONTENT=old", "OPENCODE_CONFIG_DIR=/custom",
		"OPENCODE_DB=/system.db", "REC_OPENCODE_HOME=/other", "OPENCODE_DISABLE_AUTOUPDATE=0", `OPENCODE_CLI_CONFIG_CONTENT={"animations":false}`}
	p := Paths{Data: t.TempDir()}
	h, err := harnessCommand(p, "opencode", Status{Ready: true, Model: "flint", Context: 8192, Endpoint: "http://127.0.0.1:11500/v1"}, user, "secret", nil)
	if err != nil {
		t.Fatal(err)
	}
	logo := filepath.Join(p.Data, "clients", "opencode", Version, "recursive-logo")
	if h.set[4][1] != `{"animations":false,"plugins":["`+logo+`"]}` {
		t.Fatalf("logo plugin not merged into user TUI settings: %s", h.set[4][1])
	}
	h.set[3][1] = "config"
	h.set[4][1] = "tui"
	got := h.environ(user)
	want := []string{"PATH=/bin", "XDG_DATA_HOME=/xdg", "REC_OPENCODE_HOME=" + filepath.Join(p.Data, "clients", "opencode", "user"), "OPENCODE_DISABLE_AUTOUPDATE=1",
		"REC_API_KEY=secret", "OPENCODE_CONFIG_CONTENT=config", "OPENCODE_CLI_CONFIG_CONTENT=tui"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("incorrect environment merge: %q", got)
	}
}

func TestHarnessShellCommand(t *testing.T) {
	h := harness{
		argv:  []string{"/data/opencode", "--standalone"},
		unset: []string{"OPENCODE_CONFIG"},
		set:   [][2]string{{"REC_API_KEY", "secret"}, {"OPENCODE_CONFIG_CONTENT", `{"name":"it's"}`}, {"OPENCODE_DISABLE_AUTOUPDATE", "1"}},
	}
	want := "env \\\n  -u OPENCODE_CONFIG \\\n  REC_API_KEY=\"$(rec auth token)\" \\\n" +
		"  OPENCODE_CONFIG_CONTENT='{\"name\":\"it'\\''s\"}' \\\n  OPENCODE_DISABLE_AUTOUPDATE=1 \\\n" +
		"  /data/opencode --standalone"
	if got := h.shell(); got != want {
		t.Fatalf("got:\n%s\nwant:\n%s", got, want)
	}
	h.argv = []string{"sh", "-c", `printf '%s|' "$REC_API_KEY" "$OPENCODE_CONFIG_CONTENT" "${OPENCODE_CONFIG-unset}" "$0"`, "it's"}
	cmd := exec.Command("sh", "-c", "rec() { echo key; }; "+h.shell())
	cmd.Env = []string{"PATH=/usr/bin:/bin", "OPENCODE_CONFIG=old.json"}
	out, err := cmd.CombinedOutput()
	if err != nil || string(out) != `key|{"name":"it's"}|unset|it's|` {
		t.Fatalf("shell command does not reproduce the environment: %v %q", err, out)
	}
}

func TestValidateClientStatusEndpoint(t *testing.T) {
	for _, tc := range []struct {
		name     string
		endpoint string
		ok       bool
	}{
		{"loopback", "http://127.0.0.1:11500/v1", true},
		{"lan", "http://192.168.1.5:11500/v1", true},
		{"ipv6 loopback", "http://[::1]:11500/v1", true},
		{"https", "https://127.0.0.1:11500/v1", false},
		{"hostname", "http://localhost:11500/v1", false},
		{"missing port", "http://127.0.0.1/v1", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := Status{Ready: true, Model: "flint", Context: 8192, Endpoint: tc.endpoint}
			err := validateClientStatus(s)
			if (err == nil) != tc.ok {
				t.Fatalf("validateClientStatus(%q) error = %v, want ok = %v", tc.endpoint, err, tc.ok)
			}
		})
	}
}
