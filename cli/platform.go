package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
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

func preflight(m Manifest) error {
	if m.Engine == "rocm" {
		if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
			return fmt.Errorf("the ROCm runtime requires Linux amd64")
		}
		if _, err := os.Stat("/dev/kfd"); err != nil {
			return fmt.Errorf("ROCm device /dev/kfd is unavailable: %w", err)
		}
		_, err := command("python3", "-c", "import torch; assert torch.__version__=='2.14.0+rocm10.1.0'; assert torch.version.hip=='7.16.26385'; assert torch.cuda.device_count()==1; p=torch.cuda.get_device_properties(0); assert p.gcnArchName.split(':')[0]=='gfx1201'; assert p.total_memory>30*1024**3")
		if err != nil {
			return fmt.Errorf("ROCm requires the qualified ROCm 10.1 stack and one visible 32 GB gfx1201 GPU: %w", err)
		}
		return nil
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
