package main

import (
	"context"
	"fmt"
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
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		return fmt.Errorf("rec requires an Apple Silicon Mac")
	}
	ram, err := physicalMemory()
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

func physicalMemory() (int64, error) {
	s, err := command("/usr/sbin/sysctl", "-n", "hw.memsize")
	if err != nil {
		return 0, err
	}
	return strconv.ParseInt(s, 10, 64)
}
