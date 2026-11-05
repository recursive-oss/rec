package main

import (
	"encoding/json"
	"os"
	"testing"
)

func TestContextBoundsAndLegacyProfiles(t *testing.T) {
	for _, tc := range []struct {
		name      string
		m         Manifest
		requested int
		ok        bool
	}{
		{"full model", Manifest{Context: 4096, MaxContext: 65536}, 65536, true},
		{"smaller allocation", Manifest{Context: 65536, MaxContext: 65536}, 16384, true},
		{"above model", Manifest{Context: 65536, MaxContext: 65536}, 262144, false},
		{"zero", Manifest{Context: 65536}, 0, false},
		{"negative", Manifest{Context: 65536}, -1, false},
		{"too small", Manifest{Context: 65536}, 255, false},
		{"legacy default", Manifest{Context: 4096}, 4096, true},
		{"legacy larger", Manifest{Context: 4096}, 8192, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, err := withContext(tc.m, tc.requested)
			if (err == nil) != tc.ok {
				t.Fatalf("unexpected result: %v", err)
			}
			if err == nil && (m.Context != tc.requested || m.contextLimit() != tc.m.contextLimit()) {
				t.Fatalf("lost allocation or capacity: %+v", m)
			}
		})
	}
}

func TestReleaseManifestContext(t *testing.T) {
	m := releaseManifest(t)
	if m.Context != 262144 || m.MaxContext != 262144 {
		t.Fatalf("wrong Flint context: %+v", m)
	}
	for _, limit := range []int{262144, 65536, 4096, -1} {
		m.MaxContext = limit
		b, _ := json.Marshal(m)
		_, err := decodeManifest(b)
		if (err == nil) != (limit >= m.Context) {
			t.Fatalf("maximum %d: %v", limit, err)
		}
	}
}

func TestExistingInstanceUsesEffectiveProfile(t *testing.T) {
	m := Manifest{Model: "flint", Revision: "test", RuntimeSHA: "runtime", Context: 4096}
	s := Status{Ready: true, Model: m.Model, Revision: m.Revision, Runtime: m.RuntimeSHA,
		Context: m.Context, Instance: instanceKey(m), Listen: "127.0.0.1:11500", Endpoint: "http://127.0.0.1:11500/v1"}
	if !sameInstance(s, m, "127.0.0.1", 11500) {
		t.Fatal("identical instance rejected")
	}
	explicit, _ := withContext(m, 4096)
	if !sameInstance(s, explicit, "127.0.0.1", 11500) {
		t.Fatal("explicit default rejected")
	}
	changed := m
	changed.Context = 16384
	if sameInstance(s, changed, "127.0.0.1", 11500) {
		t.Fatal("reused a smaller context")
	}
	changed = m
	changed.MinRAM++
	if sameInstance(s, changed, "127.0.0.1", 11500) {
		t.Fatal("reused different serving settings")
	}
	s.Instance = ""
	if sameInstance(s, m, "127.0.0.1", 11500) {
		t.Fatal("legacy server without profile must restart")
	}
}

func TestSameInstanceRequiresSameHost(t *testing.T) {
	m := Manifest{Model: "flint", Revision: "test", RuntimeSHA: "runtime", Context: 4096}
	wildcard := Status{Ready: true, Model: m.Model, Revision: m.Revision, Runtime: m.RuntimeSHA,
		Context: m.Context, Instance: instanceKey(m), Listen: "0.0.0.0:11500", Endpoint: "http://127.0.0.1:11500/v1"}
	if sameInstance(wildcard, m, "127.0.0.1", 11500) {
		t.Fatal("reused a server bound to a different host")
	}
	if !sameInstance(wildcard, m, "0.0.0.0", 11500) {
		t.Fatal("a server bound to all interfaces was rejected")
	}
	specific := Status{Ready: true, Model: m.Model, Revision: m.Revision, Runtime: m.RuntimeSHA,
		Context: m.Context, Instance: instanceKey(m), Listen: "192.168.1.5:11500", Endpoint: "http://192.168.1.5:11500/v1"}
	if sameInstance(specific, m, "127.0.0.1", 11500) {
		t.Fatal("reused a server bound to another interface")
	}
	if !sameInstance(specific, m, "192.168.1.5", 11500) {
		t.Fatal("an instance on a specific interface was rejected")
	}
	legacy := Status{Ready: true, Model: m.Model, Revision: m.Revision, Runtime: m.RuntimeSHA,
		Context: m.Context, Instance: instanceKey(m), Endpoint: "http://127.0.0.1:11500/v1"}
	if sameInstance(legacy, m, "127.0.0.1", 11500) {
		t.Fatal("legacy server without listen address must restart")
	}
}

func TestHostAndContextReachesEngine(t *testing.T) {
	m := releaseManifest(t)
	_, args, _ := launchCommand(Paths{}, m, "/runtime", "/model/flint-int4.json", "127.0.0.1", 11500)
	hostArg := ""
	for i, arg := range args {
		if arg == "--host" && i+1 < len(args) {
			hostArg = args[i+1]
		}
	}
	if hostArg != "127.0.0.1" {
		t.Fatalf("the requested host did not reach the engine: %q", hostArg)
	}
	contextLimit := ""
	for i, arg := range args {
		if arg == "--context-limit" && i+1 < len(args) {
			contextLimit = args[i+1]
		}
	}
	if contextLimit != "262144" {
		t.Fatalf("default 256K allocation did not reach the engine: %q", contextLimit)
	}
	if _, err := withContext(m, 524288); err == nil {
		t.Fatal("512K exceeds the engine limit")
	}
	reduced, err := withContext(m, 65536)
	if err != nil || reduced.Context != 65536 || reduced.contextLimit() != 262144 {
		t.Fatal("lower memory allocation lost the supported cap", err)
	}
}

func releaseManifest(t *testing.T) Manifest {
	t.Helper()
	b, err := os.ReadFile("../packaging/flint-int4-mlx.json")
	if err != nil {
		t.Fatal(err)
	}
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	m.RuntimeSHA = digest([]byte("runtime"))
	m.RuntimeFiles = []File{{Name: runtimeServer, Size: 1, SHA256: digest([]byte("r")), Executable: true}}
	if _, err := decodeManifest(mustJSON(t, m)); err != nil {
		t.Fatal(err)
	}
	return m
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestContextSummaryShowsLimitOnlyWhenReduced(t *testing.T) {
	m := releaseManifest(t)
	if got := m.contextSummary(); got != "262,144 tokens, input + output" {
		t.Fatalf("default context: %q", got)
	}
	reduced, err := withContext(m, 32768)
	if err != nil {
		t.Fatal(err)
	}
	if got := reduced.contextSummary(); got != "32,768 tokens, input + output (up to 262,144 with --context)" {
		t.Fatalf("reduced context: %q", got)
	}
}
