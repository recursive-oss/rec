package main

import (
	"fmt"
	"path/filepath"
	"strconv"
)

func launchCommand(p Paths, m Manifest, runtimeDir, model string, host string, port int, temp *float64, presence *float64, repeat *float64) (string, []string, []string) {
	args := []string{
		"serve", m.Model, "--model-dir", filepath.Dir(model), "--manifest", model,
		"--port", strconv.Itoa(port), "--context-limit", strconv.Itoa(m.Context),
		"--api-key-file", filepath.Join(p.Config, "api-key"),
		"--host", host,
	}
	if temp != nil {
		args = append(args, "--temperature", fmt.Sprintf("%g", *temp))
	}
	if presence != nil {
		args = append(args, "--presence-penalty", fmt.Sprintf("%g", *presence))
	}
	if repeat != nil {
		args = append(args, "--repeat-penalty", fmt.Sprintf("%g", *repeat))
	}
	return filepath.Join(runtimeDir, runtimeServer), args, []string{"HF_HUB_OFFLINE=1", "TOKENIZERS_PARALLELISM=false"}
}

func checkAppleChip(name string) error {
	for _, chip := range []string{"Apple M4 Pro", "Apple M5 Pro", "Apple M5 Max"} {
		if name == chip {
			return nil
		}
	}
	return fmt.Errorf("Flint requires an Apple M4 Pro, M5 Pro or M5 Max; detected %s", name)
}
