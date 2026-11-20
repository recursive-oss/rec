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
	if m.WeightFormat == "bf16" && chip != "Apple M5 Ultra" {
		return fmt.Errorf("the BF16 Flint checkpoint requires an Apple M5 Ultra; detected %s", chip)
	}
	return checkAppleChip(chip)
}
