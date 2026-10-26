package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestCheckROCmPlatform(t *testing.T) {
	good := func(arch string) rocmProbe {
		p := rocmPlatforms[arch]
		return rocmProbe{Torch: p.Torch, HIP: p.HIP, Devices: 1, Arch: arch, Memory: p.MinMemory + 1}
	}
	for _, arch := range []string{"gfx1201", "gfx1151"} {
		if err := checkROCmPlatform(good(arch)); err != nil {
			t.Errorf("%s rejected: %v", arch, err)
		}
	}
	cases := map[string]func(*rocmProbe){
		"no GPU":             func(p *rocmProbe) { p.Devices = 0 },
		"two GPUs":           func(p *rocmProbe) { p.Devices = 2 },
		"unknown arch":       func(p *rocmProbe) { p.Arch = "gfx1100" },
		"wrong torch":        func(p *rocmProbe) { p.Torch += ".x" },
		"wrong HIP":          func(p *rocmProbe) { p.HIP += ".x" },
		"memory at floor":    func(p *rocmProbe) { p.Memory = rocmPlatforms[p.Arch].MinMemory },
		"other arch's torch": func(p *rocmProbe) { p.Torch = rocmPlatforms["gfx1201"].Torch; p.Arch = "gfx1151" },
	}
	for name, mutate := range cases {
		p := good("gfx1151")
		mutate(&p)
		if err := checkROCmPlatform(p); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
}

func TestROCmPlatformsMatchPython(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 is unavailable")
	}
	cmd := exec.Command("python3", "-c", "import json; from rec_rocm.platforms import PLATFORMS; "+
		"print(json.dumps({k: [v.torch, v.hip, v.min_memory] for k, v in PLATFORMS.items()}))")
	cmd.Env = append(os.Environ(), "PYTHONPATH=../engine/src")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("cannot read the Python platform table: %v", err)
	}
	var python map[string][]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(out))), &python); err != nil {
		t.Fatal(err)
	}
	if len(python) != len(rocmPlatforms) {
		t.Fatalf("Python has %d platforms, Go has %d", len(python), len(rocmPlatforms))
	}
	for arch, want := range rocmPlatforms {
		got, ok := python[arch]
		if !ok || got[0] != want.Torch || got[1] != want.HIP || int64(got[2].(float64)) != want.MinMemory {
			t.Errorf("%s differs: Python %v, Go %+v", arch, got, want)
		}
	}
}

func TestROCmProbeScriptCompiles(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 is unavailable")
	}
	if err := exec.Command("python3", "-c", "import ast,sys; ast.parse(sys.argv[1])", rocmProbeScript).Run(); err != nil {
		t.Fatalf("probe script is not valid Python: %v", err)
	}
}
