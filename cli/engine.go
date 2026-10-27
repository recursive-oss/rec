package main

import (
	"fmt"
	"path/filepath"
	"strconv"
)

func launchCommand(p Paths, m Manifest, runtimeDir, model string, port int) (string, []string, []string) {
	return filepath.Join(runtimeDir, runtimeServer), []string{
		"serve", m.Model, "--model-dir", filepath.Dir(model), "--manifest", model,
		"--port", strconv.Itoa(port), "--context-limit", strconv.Itoa(m.Context),
		"--api-key-file", filepath.Join(p.Config, "api-key"),
	}, []string{"HF_HUB_OFFLINE=1", "TOKENIZERS_PARALLELISM=false"}
}

func checkAppleChip(name string) error {
	for _, chip := range []string{"Apple M4 Pro", "Apple M4 Max", "Apple M5 Pro", "Apple M5 Max"} {
		if name == chip {
			return nil
		}
	}
	return fmt.Errorf("Flint requires an Apple M4 Pro, M4 Max, M5 Pro or M5 Max; detected %s", name)
}
