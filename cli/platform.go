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
	if m.Engine == "flint-native" {
		if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
			return fmt.Errorf("native Flint requires Linux/amd64")
		}
		data, err := os.ReadFile("/proc/meminfo")
		if err != nil {
			return err
		}
		fields := strings.Fields(string(data))
		if len(fields) < 2 || fields[0] != "MemTotal:" {
			return fmt.Errorf("cannot determine host memory")
		}
		kib, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil || kib <= 0 || kib > (1<<63-1)/1024 {
			return fmt.Errorf("cannot determine host memory")
		}
		ram := kib * 1024
		if ram < m.MinRAM {
			return fmt.Errorf("this model requires %d GiB of memory; detected %d GiB", m.MinRAM>>30, ram>>30)
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
