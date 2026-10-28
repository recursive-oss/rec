//go:build darwin || linux

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
)

func ompInstallDir(p Paths) string {
	sum := sha256.Sum256(append(append([]byte{}, ompPackage...), ompLock...))
	return filepath.Join(p.Data, "clients", "omp", "npm-"+hex.EncodeToString(sum[:8]))
}

func ompExecutable(p Paths) string {
	return filepath.Join(ompInstallDir(p), "node_modules", ".bin", "omp")
}

func installOmp(p Paths, out io.Writer) (string, error) {
	dir := ompInstallDir(p)
	parent := filepath.Dir(dir)
	executable := ompExecutable(p)
	if _, err := os.Stat(executable); err == nil {
		return executable, nil
	}
	if _, err := exec.LookPath("bun"); err != nil {
		return "", fmt.Errorf("rec launch omp requires Bun; install Bun, then retry")
	}
	npm, err := exec.LookPath("npm")
	if err != nil {
		return "", fmt.Errorf("rec launch omp requires Node.js/npm; install Node.js, then retry")
	}
	if err := os.MkdirAll(parent, 0700); err != nil {
		return "", err
	}
	l, err := lock(filepath.Join(p.State, "omp-install.lock"))
	if err != nil {
		return "", fmt.Errorf("another launcher is installing omp; retry when it finishes")
	}
	defer unlock(l)
	if _, err := os.Stat(executable); err == nil {
		return executable, nil
	}
	// Remove staging left by an interrupted install. Earlier pins are kept,
	// since a running omp may still load modules from them.
	leftovers, err := filepath.Glob(filepath.Join(parent, ".install-*"))
	if err != nil {
		return "", err
	}
	for _, path := range leftovers {
		if err := os.RemoveAll(path); err != nil {
			return "", err
		}
	}
	staging, err := os.MkdirTemp(parent, ".install-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(staging)
	if err := os.WriteFile(filepath.Join(staging, "package.json"), ompPackage, 0600); err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(staging, "package-lock.json"), ompLock, 0600); err != nil {
		return "", err
	}
	installing := startActivity(out, "Installing omp", "Installed omp")
	cmd := exec.Command(npm, "ci", "--no-audit", "--no-fund", "--loglevel=error")
	cmd.Dir = staging
	if output, err := cmd.CombinedOutput(); err != nil {
		installing.fail()
		_, _ = out.Write(output)
		return "", fmt.Errorf("installing omp with npm ci: %w", err)
	}
	installing.succeed()
	if err := os.Rename(staging, dir); err != nil {
		return "", err
	}
	return executable, nil
}

func ompConfig(p Paths, s Status, user string) (string, error) {
	if err := validateClientStatus(s); err != nil {
		return "", err
	}
	dir := filepath.Join(p.Data, "clients", "omp", Version)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", err
	}
	if err := os.MkdirAll(user, 0700); err != nil {
		return "", err
	}
	models := fmt.Sprintf(`providers:
  rec:
    baseUrl: %s
    apiKey: REC_API_KEY
    api: openai-completions
    models:
      - id: flint
        name: Flint
        reasoning: false
        input: [text]
        contextWindow: %d
        maxTokens: %d
        compat:
          supportsStore: false
          supportsDeveloperRole: false
          supportsReasoningEffort: false
          maxTokensField: max_tokens
`, s.Endpoint, s.Context, s.Context)
	if err := atomicWrite(filepath.Join(user, "models.yml"), []byte(models), 0600); err != nil {
		return "", err
	}
	settings := `modelRoles:
  default: rec/flint
  smol: rec/flint
  slow: rec/flint
  plan: rec/flint
  task: rec/flint
  advisor: rec/flint
  tiny: rec/flint
retry:
  modelFallback: false
  fallbackChains:
    default: []
    smol: []
    slow: []
    plan: []
    task: []
    advisor: []
    tiny: []
enabledProviders:
  - rec
`
	path := filepath.Join(dir, "rec-omp.yml")
	if err := atomicWrite(path, []byte(settings), 0600); err != nil {
		return "", err
	}
	return path, nil
}

func ompUser(user string) harness {
	return harness{
		unset: []string{"PI_CODING_AGENT_SESSION_DIR"},
		set:   [][2]string{{"PI_CODING_AGENT_DIR", user}},
	}
}
