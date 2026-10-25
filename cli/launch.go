//go:build darwin || linux

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"
)

func parseLaunchArgs(args []string) (printOnly bool, quant string, manifest string, temp *float64, presence *float64, repeat *float64, harness string, rest []string, err error) {
	quant = QuantInt4
	beforeDash := args
	afterDash := []string{}
	for i, a := range args {
		if a == "--" {
			beforeDash = args[:i]
			afterDash = args[i+1:]
			break
		}
	}
	var positional []string
	for i := 0; i < len(beforeDash); i++ {
		arg := beforeDash[i]
		switch {
		case arg == "--args":
			printOnly = true
		case arg == "--quant":
			if i+1 >= len(beforeDash) {
				return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("flag needs an argument: --quant")
			}
			i++
			quant = beforeDash[i]
		case strings.HasPrefix(arg, "--quant="):
			quant = strings.TrimPrefix(arg, "--quant=")
		case arg == "--manifest":
			if i+1 >= len(beforeDash) {
				return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("flag needs an argument: --manifest")
			}
			i++
			manifest = beforeDash[i]
		case strings.HasPrefix(arg, "--manifest="):
			manifest = strings.TrimPrefix(arg, "--manifest=")
		case arg == "--temperature":
			if i+1 >= len(beforeDash) {
				return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("flag needs an argument: --temperature")
			}
			i++
			val, err := strconv.ParseFloat(beforeDash[i], 64)
			if err != nil || val < 0 || val > 2 {
				return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("temperature must be between 0 and 2")
			}
			temp = &val
		case strings.HasPrefix(arg, "--temperature="):
			valStr := strings.TrimPrefix(arg, "--temperature=")
			val, err := strconv.ParseFloat(valStr, 64)
			if err != nil || val < 0 || val > 2 {
				return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("temperature must be between 0 and 2")
			}
			temp = &val
		case arg == "--presence-penalty":
			if i+1 >= len(beforeDash) {
				return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("flag needs an argument: --presence-penalty")
			}
			i++
			val, err := strconv.ParseFloat(beforeDash[i], 64)
			if err != nil || val < -2 || val > 2 {
				return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("presence-penalty must be between -2 and 2")
			}
			presence = &val
		case strings.HasPrefix(arg, "--presence-penalty="):
			valStr := strings.TrimPrefix(arg, "--presence-penalty=")
			val, err := strconv.ParseFloat(valStr, 64)
			if err != nil || val < -2 || val > 2 {
				return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("presence-penalty must be between -2 and 2")
			}
			presence = &val
		case arg == "--repeat-penalty" || arg == "--repeat_penalty" || arg == "--repetition-penalty":
			if i+1 >= len(beforeDash) {
				return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("flag needs an argument: %s", arg)
			}
			i++
			val, err := strconv.ParseFloat(beforeDash[i], 64)
			if err != nil || val <= 0 || val > 5 {
				return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("repeat-penalty must be greater than 0 and at most 5")
			}
			repeat = &val
		case strings.HasPrefix(arg, "--repeat-penalty="):
			valStr := strings.TrimPrefix(arg, "--repeat-penalty=")
			val, err := strconv.ParseFloat(valStr, 64)
			if err != nil || val <= 0 || val > 5 {
				return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("repeat-penalty must be greater than 0 and at most 5")
			}
			repeat = &val
		case strings.HasPrefix(arg, "--repeat_penalty="):
			valStr := strings.TrimPrefix(arg, "--repeat_penalty=")
			val, err := strconv.ParseFloat(valStr, 64)
			if err != nil || val <= 0 || val > 5 {
				return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("repeat-penalty must be greater than 0 and at most 5")
			}
			repeat = &val
		case strings.HasPrefix(arg, "--repetition-penalty="):
			valStr := strings.TrimPrefix(arg, "--repetition-penalty=")
			val, err := strconv.ParseFloat(valStr, 64)
			if err != nil || val <= 0 || val > 5 {
				return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("repeat-penalty must be greater than 0 and at most 5")
			}
			repeat = &val
		case strings.HasPrefix(arg, "-"):
			return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("flag provided but not defined: %s", arg)
		default:
			positional = append(positional, arg)
		}
	}
	if len(positional) == 0 {
		return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("missing harness; use rec launch pi or rec launch opencode")
	}
	harness = positional[0]
	rest = append(positional[1:], afterDash...)
	quant, err = validateQuant(quant)
	if err != nil {
		return false, "", "", nil, nil, nil, "", nil, err
	}
	if harness != "opencode" && harness != "pi" {
		return false, "", "", nil, nil, nil, "", nil, fmt.Errorf("unsupported harness %q; use rec launch pi or rec launch opencode", harness)
	}
	return printOnly, quant, manifest, temp, presence, repeat, harness, rest, nil
}

func launch(p Paths, args []string) error {
	if len(args) == 0 || args[0] == "--help" || args[0] == "-h" || args[0] == "help" {
		printUsage(os.Stdout, "launch")
		fmt.Println("\nArguments after -- go to the harness. Reuses a running server, which keeps")
		fmt.Println("running after the harness exits; run rec stop to stop it. Pi requires Node.js")
		fmt.Println("and npm. The first launch installs the pinned harness.")
		return nil
	}
	printOnly, quant, manifest, temp, presence, repeat, targetHarness, rest, err := parseLaunchArgs(args)
	if err != nil {
		return err
	}
	if printOnly {
		return printHarness(p, targetHarness, rest)
	}
	if _, err := installClient(p, targetHarness); err != nil {
		return err
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM, syscall.SIGHUP)
	folder := ""
	if m, err := loadManifest(manifest, quant); err == nil && len(m.Files) > 0 {
		folder = modelDigest(m)
	}
	s, err := launchReady(ctx, p, os.Stdout, quant, manifest, folder, temp, presence, repeat)
	cancel()
	if err != nil {
		return err
	}
	key, err := token(p, false)
	if err != nil {
		return err
	}
	h, err := harnessCommand(p, targetHarness, s, os.Environ(), key, rest)
	if err != nil {
		return err
	}
	openingClient(map[string]string{"pi": "Pi", "opencode": "OpenCode"}[targetHarness], s, quant, folder)
	return syscall.Exec(h.argv[0], h.argv, h.environ(os.Environ()))
}

func printHarness(p Paths, name string, rest []string) error {
	s, err := control(p, "status")
	if err != nil {
		return fmt.Errorf("Flint is not running; start it with rec serve: %w", err)
	}
	h, err := harnessCommand(p, name, s, os.Environ(), "", rest)
	if err != nil {
		return err
	}
	fmt.Println(h.shell())
	return nil
}

type harness struct {
	argv  []string
	unset []string
	set   [][2]string
}

func harnessCommand(p Paths, name string, s Status, env []string, key string, rest []string) (harness, error) {
	user, err := clientUserDir(p, name)
	if err != nil {
		return harness{}, err
	}
	if name == "pi" {
		extension, config, err := piConfig(p, s)
		if err != nil {
			return harness{}, err
		}
		h := piUser(user)
		h.argv = append([]string{piExecutable(p), "--extension", extension, "--provider", "rec", "--model", "flint"}, rest...)
		h.set = append(h.set, [2]string{"REC_API_KEY", key}, [2]string{"REC_PI_CONFIG", config})
		return h, nil
	}
	pin, err := loadOpenCodePin()
	if err != nil {
		return harness{}, err
	}
	config, err := openCodeConfig(p, s)
	if err != nil {
		return harness{}, err
	}
	plugin, err := openCodePlugin(p)
	if err != nil {
		return harness{}, err
	}
	tui := ""
	for _, value := range env {
		if setting, ok := strings.CutPrefix(value, "OPENCODE_CLI_CONFIG_CONTENT="); ok {
			tui = setting
		}
	}
	tui, err = openCodeTUIConfig(tui, plugin)
	h := openCodeUser(user)
	h.argv = append([]string{openCodeBinary(p, pin)}, openCodeArgs(rest)...)
	h.unset = append(h.unset, "OPENCODE_CONFIG")
	h.set = append(h.set, [2]string{"REC_API_KEY", key}, [2]string{"OPENCODE_CONFIG_CONTENT", config}, [2]string{"OPENCODE_CLI_CONFIG_CONTENT", tui})
	return h, err
}

func (h harness) environ(env []string) []string {
	names := slices.Clone(h.unset)
	for _, setting := range h.set {
		names = append(names, setting[0])
	}
	result := withoutEnv(env, names...)
	for _, setting := range h.set {
		result = append(result, setting[0]+"="+setting[1])
	}
	return result
}

func (h harness) shell() string {
	words := []string{"env"}
	for _, name := range h.unset {
		words = append(words, "-u "+name)
	}
	for _, setting := range h.set {
		if setting[0] == "REC_API_KEY" {
			words = append(words, `REC_API_KEY="$(rec auth token)"`)
		} else {
			words = append(words, setting[0]+"="+shellQuote(setting[1]))
		}
	}
	argv := make([]string, len(h.argv))
	for i, arg := range h.argv {
		argv[i] = shellQuote(arg)
	}
	return strings.Join(append(words, strings.Join(argv, " ")), " \\\n  ")
}

var shellSafe = regexp.MustCompile(`^[A-Za-z0-9_@%+=:,./-]+$`)

func shellQuote(value string) string {
	if shellSafe.MatchString(value) {
		return value
	}
	return "'" + strings.ReplaceAll(value, "'", `'\''`) + "'"
}

func openingClient(name string, s Status, quant, folder string) {
	detail := fmt.Sprintf("Flint (%s) at %s, %s tokens", quant, s.Endpoint, formatTokens(s.Context))
	if folder != "" {
		detail = fmt.Sprintf("Flint (%s) (%s) at %s, %s tokens", quant, folder, s.Endpoint, formatTokens(s.Context))
	}
	newScreen(os.Stdout).step(accentStyle.Render("→"), "Opening "+name, detail)
}

func openCodeArgs(args []string) []string {
	// V2's TUI selects its model from config; --model is a run-only flag.
	// A private OpenCode server receives this process's Rec config and API key,
	// instead of reusing a background service started with unrelated settings.
	if len(args) > 0 && args[0] == "run" {
		return append([]string{"run", "--standalone", "--model", "rec/flint"}, args[1:]...)
	}
	return append([]string{"--standalone"}, args...)
}

func launchReady(ctx context.Context, p Paths, out io.Writer, quant, manifest, folder string, temp *float64, presence *float64, repeat *float64) (Status, error) {
	l, err := lock(filepath.Join(p.State, "launch.lock"))
	if err != nil {
		return Status{}, fmt.Errorf("another launcher is starting Flint; retry when it is ready")
	}
	defer unlock(l)
	s, err := control(p, "status")
	if err == nil && s.Model != "flint" {
		return s, fmt.Errorf("running model is %q, expected flint; use rec stop first", s.Model)
	}
	expectedRevision := "flint-" + quant
	if err == nil && s.Ready {
		if s.Revision != "" && s.Revision != expectedRevision {
			return s, fmt.Errorf("running server uses %s, requested %s; use rec stop first", s.Revision, expectedRevision)
		}
		if temp != nil && s.Temperature != nil && *s.Temperature != *temp {
			return s, fmt.Errorf("running server uses temperature %g, requested %g; use rec stop first", *s.Temperature, *temp)
		}
		if presence != nil && s.PresencePenalty != nil && *s.PresencePenalty != *presence {
			return s, fmt.Errorf("running server uses presence penalty %g, requested %g; use rec stop first", *s.PresencePenalty, *presence)
		}
		if repeat != nil && s.RepeatPenalty != nil && *s.RepeatPenalty != *repeat {
			return s, fmt.Errorf("running server uses repeat penalty %g, requested %g; use rec stop first", *s.RepeatPenalty, *repeat)
		}
		readyMsg := fmt.Sprintf("Flint (%s) is ready", quant)
		if folder != "" {
			readyMsg = fmt.Sprintf("Flint (%s) (%s) is ready", quant, folder)
		}
		newScreen(out).succeeded(readyMsg, "")
		return s, nil
	}
	var child *exec.Cmd
	var done chan error
	var log *os.File
	owned := false
	success := false
	defer func() {
		if log != nil {
			log.Close()
		}
		if owned && !success {
			// The unready server is being discarded; its exit status is irrelevant.
			_ = child.Process.Signal(syscall.SIGTERM)
			select {
			case <-done: //nolint:errcheck
			case <-time.After(10 * time.Second):
				_ = child.Process.Kill()
				<-done //nolint:errcheck
			}
		}
	}()
	if err != nil {
		// Complete first-time downloads while we own the terminal, before
		// detaching the server. Serving still verifies every file.
		m, e := loadManifest(manifest, quant)
		if e != nil {
			return s, e
		}
		if !modelFilesPresent(p, m) {
			if e = preflight(m); e != nil {
				return s, e
			}
			if _, e = ensureModel(ctx, p, m, out); e != nil {
				return s, e
			}
		}
		self, e := os.Executable()
		if e != nil {
			return s, e
		}
		logPath := filepath.Join(p.State, "launch-server.log")
		log, e = os.OpenFile(logPath, os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0600)
		if e != nil {
			return s, e
		}
		if e = log.Chmod(0600); e != nil {
			return s, e
		}
		cmdArgs := []string{"_background"}
		if quant != QuantInt4 {
			cmdArgs = append(cmdArgs, "--quant", quant)
		}
		if manifest != "" {
			cmdArgs = append(cmdArgs, "--manifest", manifest)
		}
		if temp != nil {
			cmdArgs = append(cmdArgs, "--temperature", fmt.Sprintf("%g", *temp))
		}
		if presence != nil {
			cmdArgs = append(cmdArgs, "--presence-penalty", fmt.Sprintf("%g", *presence))
		}
		if repeat != nil {
			cmdArgs = append(cmdArgs, "--repeat-penalty", fmt.Sprintf("%g", *repeat))
		}
		child = exec.Command(self, cmdArgs...)
		child.Stdout, child.Stderr = log, log
		child.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
		if e = child.Start(); e != nil {
			return s, e
		}
		owned = true
		done = make(chan error, 1)
		go func() { done <- child.Wait() }()
	}
	label := fmt.Sprintf("Starting Flint (%s)", quant)
	doneLabel := fmt.Sprintf("Flint (%s) is ready", quant)
	if folder != "" {
		label = fmt.Sprintf("Starting Flint (%s) (%s)", quant, folder)
		doneLabel = fmt.Sprintf("Flint (%s) (%s) is ready", quant, folder)
	}
	starting := startActivity(out, label, doneLabel)
	// ReadAt does not share the child's output offset. Show the server's latest
	// lifecycle line as the phase and keep the whole log for diagnosis.
	var server bytes.Buffer
	relay := func() {
		if log == nil {
			return
		}
		buffer := make([]byte, 32768)
		for {
			n, e := log.ReadAt(buffer, int64(server.Len()))
			server.Write(buffer[:n])
			if e != nil {
				break
			}
		}
		starting.phase(serverPhase(server.String()))
	}
	failed := func(err error) (Status, error) {
		starting.fail()
		_, _ = out.Write(server.Bytes())
		return s, err
	}
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	for {
		relay()
		s, err = control(p, "status")
		if err == nil && s.Ready {
			if s.Model != "flint" {
				return failed(fmt.Errorf("running model is %q, expected flint", s.Model))
			}
			starting.succeed()
			success = true
			return s, nil
		}
		select {
		case <-ctx.Done():
			starting.cancel()
			return s, ctx.Err()
		case e := <-done:
			owned = false
			relay()
			return failed(fmt.Errorf("Flint exited before becoming ready (%v); run rec doctor", e))
		case <-ticker.C:
			if child == nil && err != nil {
				return failed(fmt.Errorf("Flint stopped before becoming ready; run rec doctor"))
			}
		}
	}
}

var elapsedSuffix = regexp.MustCompile(`\s*\(\d+(m \d+)?s\)$`)

func serverPhase(log string) string {
	lines := strings.Split(strings.TrimSpace(log), "\n")
	line := strings.TrimSpace(lines[len(lines)-1])
	return elapsedSuffix.ReplaceAllString(strings.TrimPrefix(line, "✓ "), "")
}

func validateClientStatus(s Status) error {
	endpoint, err := url.Parse(s.Endpoint)
	if err != nil || endpoint.Scheme != "http" || endpoint.Hostname() != "127.0.0.1" || endpoint.Port() == "" || endpoint.Path != "/v1" || endpoint.User != nil || endpoint.RawQuery != "" || endpoint.Fragment != "" {
		return fmt.Errorf("invalid local Rec endpoint")
	}
	if !s.Ready || s.Model != "flint" || s.Context < 2048 {
		return fmt.Errorf("Flint is not ready with a usable context")
	}
	return nil
}

func openCodeConfig(p Paths, s Status) (string, error) {
	if err := validateClientStatus(s); err != nil {
		return "", err
	}
	output := s.Context
	dir := filepath.Join(p.Data, "clients", "opencode", Version)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", err
	}
	for _, name := range []string{"instructions.md"} {
		data, err := clientAssets.ReadFile("clients/opencode/" + name)
		if err != nil {
			return "", err
		}
		if err := atomicWrite(filepath.Join(dir, name), data, 0600); err != nil {
			return "", err
		}
	}
	// Apply the context ceiling to every request, including titles. The engine
	// fits generation to the space left by the tokenized prompt.
	body := map[string]any{"max_completion_tokens": output}
	// The engine serves one request at a time; titles would delay foreground turns.
	title := map[string]any{"model": "rec/flint", "disabled": true}
	config := map[string]any{
		"model":  "rec/flint",
		"agents": map[string]any{"title": title},
		"policies": []map[string]string{
			{"action": "provider.use", "resource": "*", "effect": "deny"},
			{"action": "provider.use", "resource": "rec", "effect": "allow"},
		},
		"providers": map[string]any{"rec": map[string]any{
			"package": "@opencode/ai/providers/openai-compatible", "name": "Recursive local",
			"settings": map[string]any{"baseURL": s.Endpoint, "apiKey": "{env:REC_API_KEY}"},
			"models": map[string]any{"flint": map[string]any{
				"name": "Flint", "limit": map[string]int{"context": s.Context, "output": output},
				"body":         body,
				"capabilities": map[string]any{"tools": true, "input": []string{"text"}, "output": []string{"text"}},
			}},
		}},
		"instructions": []string{filepath.Join(dir, "instructions.md")},
	}
	data, err := json.Marshal(config)
	return string(data), err
}

func openCodeUser(user string) harness {
	return harness{
		unset: []string{"OPENCODE_DB", "OPENCODE_CONFIG_DIR"},
		set:   [][2]string{{"REC_OPENCODE_HOME", user}, {"OPENCODE_DISABLE_AUTOUPDATE", "1"}},
	}
}

func withoutEnv(env []string, names ...string) []string {
	result := make([]string, 0, len(env))
	for _, value := range env {
		if name, _, _ := strings.Cut(value, "="); !slices.Contains(names, name) {
			result = append(result, value)
		}
	}
	return result
}

func clientUserDir(p Paths, name string) (string, error) {
	dir := filepath.Join(p.Data, "clients", name, "user")
	return dir, os.MkdirAll(dir, 0700)
}

func installClient(p Paths, name string) (string, error) {
	if name == "pi" {
		return installPi(p, os.Stderr)
	}
	pin, err := loadOpenCodePin()
	if err != nil {
		return "", err
	}
	return openCodeClient(p, pin, downloadClient(), os.Stderr)
}
