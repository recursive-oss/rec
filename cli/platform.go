package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

func command(name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	b, e := exec.CommandContext(ctx, name, args...).Output()
	return strings.TrimSpace(string(b)), e
}

type rocmPlatform struct {
	Torch, HIP string
	MinMemory  int64
}

// Mirrors engine/src/rec_rocm/platforms.py; TestROCmPlatformsMatchPython keeps them in sync.
var rocmPlatforms = map[string]rocmPlatform{
	"gfx1201": {"2.14.0+rocm10.1.0", "7.16.26385", 30 << 30},
	"gfx1151": {"2.9.1+rocm7.2.1.gitff65f5bc", "7.2.53211-e1a6bc5663", 100 << 30},
}

const rocmProbeScript = "import json,torch;n=torch.cuda.device_count();" +
	"p=torch.cuda.get_device_properties(0) if n else None;" +
	"print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,'devices':n," +
	"'arch':p.gcnArchName.split(':')[0] if p else '','memory':p.total_memory if p else 0}))"

type rocmProbe struct {
	Torch   string `json:"torch"`
	HIP     string `json:"hip"`
	Devices int    `json:"devices"`
	Arch    string `json:"arch"`
	Memory  int64  `json:"memory"`
}

func rocmArchitectures() string {
	names := make([]string, 0, len(rocmPlatforms))
	for name := range rocmPlatforms {
		names = append(names, name)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}

func checkROCmPlatform(p rocmProbe) error {
	if p.Devices != 1 {
		return fmt.Errorf("ROCm requires exactly one visible GPU; found %d", p.Devices)
	}
	want, ok := rocmPlatforms[p.Arch]
	if !ok {
		return fmt.Errorf("ROCm is qualified only on %s; found %q", rocmArchitectures(), p.Arch)
	}
	if p.Torch != want.Torch || p.HIP != want.HIP {
		return fmt.Errorf("%s requires torch %s with HIP %s; found torch %s with HIP %s",
			p.Arch, want.Torch, want.HIP, p.Torch, p.HIP)
	}
	if p.Memory <= want.MinMemory {
		return fmt.Errorf("%s needs more than %d GiB of GPU-visible memory; found %d GiB",
			p.Arch, want.MinMemory>>30, p.Memory>>30)
	}
	return nil
}

func preflight(m Manifest) error {
	if m.Engine == "rocm" {
		if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
			return fmt.Errorf("the ROCm runtime requires Linux amd64")
		}
		if _, err := os.Stat("/dev/kfd"); err != nil {
			return fmt.Errorf("ROCm device /dev/kfd is unavailable: %w", err)
		}
		out, err := command("python3", "-c", rocmProbeScript)
		if err != nil {
			return fmt.Errorf("ROCm requires PyTorch for one of the qualified GPUs (%s): %w", rocmArchitectures(), err)
		}
		var probe rocmProbe
		if err := json.Unmarshal([]byte(out), &probe); err != nil {
			return fmt.Errorf("unreadable PyTorch probe output: %w", err)
		}
		return checkROCmPlatform(probe)
	}
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		return fmt.Errorf("rec requires an Apple Silicon Mac")
	}
	s, err := command("/usr/sbin/sysctl", "-n", "hw.memsize")
	if err != nil {
		return err
	}
	ram, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return err
	}
	if ram < m.MinRAM {
		return fmt.Errorf("this model requires %d GiB of memory; detected %d GiB", m.MinRAM>>30, ram>>30)
	}
	if m.Model != "flint" {
		return nil
	}
	chip, err := command("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string")
	if err != nil {
		return err
	}
	return checkAppleChip(chip)
}

func platformDescription(m Manifest) string {
	if m.Engine == "rocm" {
		return "This AMD ROCm platform"
	}
	return "This Mac"
}
