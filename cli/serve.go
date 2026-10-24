//go:build darwin || linux

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
)

type Status struct {
	Engine     string `json:"engine"`
	Ready      bool   `json:"ready"`
	Version    string `json:"version"`
	Model      string `json:"model"`
	Revision   string `json:"revision"`
	Runtime    string `json:"runtime_sha256"`
	Endpoint   string `json:"endpoint"`
	Context    int    `json:"context"`
	MaxContext int    `json:"max_context"`
	Instance   string `json:"instance_sha256"`
}

func instanceKey(m Manifest) string {
	m.MaxContext = m.contextLimit()
	b, _ := json.Marshal(m)
	return digest(b)
}

func sameInstance(s Status, m Manifest, port int) bool {
	return s.Ready && s.Model == m.Model && s.Runtime == m.RuntimeSHA &&
		s.Revision == m.Revision && s.Context == m.Context &&
		s.Instance == instanceKey(m) && s.Endpoint == fmt.Sprintf("http://127.0.0.1:%d/v1", port)
}

func control(p Paths, action string) (Status, error) {
	var s Status
	t := &http.Transport{DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, "unix", filepath.Join(p.State, "control.sock"))
	}}
	defer t.CloseIdleConnections()
	c := &http.Client{Transport: t, Timeout: 5 * time.Second}
	method := "GET"
	if action == "stop" {
		method = "POST"
	}
	req, _ := http.NewRequest(method, "http://rec/"+action, nil)
	resp, e := c.Do(req)
	if e != nil {
		return s, fmt.Errorf("no managed rec instance is reachable")
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return s, fmt.Errorf("control request failed")
	}
	e = json.NewDecoder(io.LimitReader(resp.Body, 8192)).Decode(&s)
	return s, e
}

func cleanEnv(extra []string) []string {
	env := []string{}
	for _, v := range os.Environ() {
		// The bundled Python must not see the user's Python environment or libraries.
		if strings.HasPrefix(v, "PYTHON") || strings.HasPrefix(v, "VIRTUAL_ENV=") || strings.HasPrefix(v, "DYLD_LIBRARY_PATH=") {
			continue
		}
		env = append(env, v)
	}
	return append(env, extra...)
}

// A separate watchdog owns the server. The parent pipe closes even after SIGKILL.
func supervise(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("missing server path")
	}
	pipe := os.NewFile(3, "parent-liveness")
	if pipe == nil {
		return fmt.Errorf("missing parent pipe")
	}
	defer pipe.Close()
	if _, e := pipe.Stat(); e != nil {
		return fmt.Errorf("missing parent pipe")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM, syscall.SIGHUP)
	defer stop()
	gone := make(chan struct{})
	go func() { _, _ = io.Copy(io.Discard, pipe); close(gone) }()
	child := exec.Command(args[0], args[1:]...)
	child.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	child.Stdout = os.Stdout
	child.Stderr = os.Stderr
	if err := child.Start(); err != nil {
		return err
	}
	done := make(chan error, 1)
	go func() { done <- child.Wait() }()
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
	case <-gone:
	}
	// The server is being stopped; its exit status is irrelevant.
	_ = syscall.Kill(-child.Process.Pid, syscall.SIGTERM)
	select {
	case <-done: //nolint:errcheck
		return nil
	case <-time.After(5 * time.Second):
		_ = syscall.Kill(-child.Process.Pid, syscall.SIGKILL)
		<-done //nolint:errcheck
		return nil
	}
}

func probe(ctx context.Context, base, key, model string, phase ...func(string)) error {
	c := &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{Proxy: nil}}
	defer c.CloseIdleConnections()
	request := func(method, path string, b []byte) ([]byte, error) {
		req, e := http.NewRequestWithContext(ctx, method, base+path, bytes.NewReader(b))
		if e != nil {
			return nil, e
		}
		req.Header.Set("Authorization", "Bearer "+key)
		req.Header.Set("Content-Type", "application/json")
		r, e := c.Do(req)
		if e != nil {
			return nil, e
		}
		defer r.Body.Close()
		data, e := io.ReadAll(io.LimitReader(r.Body, 1<<20))
		if e != nil {
			return nil, e
		}
		if r.StatusCode != 200 {
			return nil, fmt.Errorf("%s returned HTTP %d", path, r.StatusCode)
		}
		return data, nil
	}
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	for {
		if _, e := request("GET", "/health", nil); e == nil {
			break
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("server never became healthy: %w", ctx.Err())
		case <-ticker.C:
		}
	}
	for _, update := range phase {
		update("Checking model identity")
	}
	b, e := request("GET", "/v1/models", nil)
	if e != nil {
		return e
	}
	var listing struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if e = json.Unmarshal(b, &listing); e != nil {
		return e
	}
	found := false
	for _, v := range listing.Data {
		if v.ID == model {
			found = true
		}
	}
	if !found {
		return fmt.Errorf("server does not advertise %s", model)
	}
	c.Timeout = 2 * time.Minute
	for _, update := range phase {
		update("Checking generation")
	}
	payload, _ := json.Marshal(map[string]any{"model": model, "messages": []map[string]string{{"role": "user", "content": "Say hello."}}, "max_tokens": 8, "temperature": 0, "stream": false})
	b, e = request("POST", "/v1/chat/completions", payload)
	if e != nil {
		return e
	}
	var completion struct {
		Choices []struct {
			Message struct {
				Role string `json:"role"`
			} `json:"message"`
		} `json:"choices"`
	}
	if e = json.Unmarshal(b, &completion); e != nil {
		return e
	}
	if len(completion.Choices) == 0 || completion.Choices[0].Message.Role != "assistant" {
		return fmt.Errorf("generation probe returned no choices")
	}
	return nil
}

func serve(parent context.Context, p Paths, m Manifest, override string, port int, out io.Writer, background ...bool) error {
	l, e := lock(filepath.Join(p.State, "instance.lock"))
	if e != nil {
		if s, err := control(p, "status"); err == nil {
			if sameInstance(s, m, port) {
				newScreen(out).succeeded(m.Model+" is already running", s.Endpoint)
				return nil
			}
			return fmt.Errorf("another instance is starting or uses a different profile (running context %d, requested %d); use rec status or rec stop", s.Context, m.Context)
		}
		return e
	}
	defer unlock(l)
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	address := fmt.Sprintf("127.0.0.1:%d", port)
	check, e := net.Listen("tcp", address)
	if e != nil {
		return fmt.Errorf("port %d is unavailable; use --port with a free port", port)
	}
	check.Close()
	status := Status{Engine: m.Engine, Version: Version, Model: m.Model, Revision: m.Revision, Runtime: m.RuntimeSHA, Endpoint: "http://" + address + "/v1", Context: m.Context, MaxContext: m.contextLimit(), Instance: instanceKey(m)}
	var mu sync.Mutex
	socket := filepath.Join(p.State, "control.sock")
	os.Remove(socket)
	listener, e := net.Listen("unix", socket)
	if e != nil {
		return e
	}
	defer os.Remove(socket)
	defer listener.Close()
	if e = os.Chmod(socket, 0600); e != nil {
		return e
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /status", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		_ = json.NewEncoder(w).Encode(status)
	})
	mux.HandleFunc("POST /stop", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		status.Ready = false
		_ = json.NewEncoder(w).Encode(status)
		mu.Unlock()
		cancel()
	})
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	defer srv.Close()
	go func() { _ = srv.Serve(listener) }()
	ui := newScreen(out)
	if e = preflight(m); e != nil {
		ui.failed(platformDescription(m)+" is not supported", "")
		return e
	}
	ui.succeeded(platformDescription(m)+" is supported", "")
	runtimeProgress := startActivity(out, "Preparing runtime", "Runtime ready")
	runtimeDir, e := ensureRuntime(p, m, override)
	if e != nil {
		runtimeProgress.fail()
		return e
	}
	runtimeProgress.succeed()
	model, e := ensureModel(ctx, p, m, out)
	if e != nil {
		return e
	}
	if e = preflight(m); e != nil {
		return e
	}
	key, e := token(p, true)
	if e != nil {
		return e
	}
	executable, args, extra := launchCommand(p, m, runtimeDir, model, port)
	self, e := os.Executable()
	if e != nil {
		return e
	}
	read, write, e := os.Pipe()
	if e != nil {
		return e
	}
	defer write.Close()
	child := exec.Command(self, append([]string{"_supervise", executable}, args...)...)
	child.ExtraFiles = []*os.File{read}
	child.Env = cleanEnv(extra)
	child.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	// Keep engine diagnostics visible only in foreground serving. Background
	// lifecycle logs must not retain potentially sensitive request bodies.
	child.Stdout = io.Discard
	child.Stderr = io.Discard
	if len(background) <= 0 || !background[0] {
		child.Stderr = os.Stderr
	}
	if e = child.Start(); e != nil {
		read.Close()
		return e
	}
	read.Close()
	done := make(chan error, 1)
	go func() { done <- child.Wait() }()
	defer func() {
		write.Close()
		select {
		case <-done: //nolint:errcheck // Cleanup on return; the exit status no longer matters.
		case <-time.After(7 * time.Second):
			_ = child.Process.Kill()
		}
	}()
	loading := startActivity(out, "Loading "+displayName(m.Model), "Loaded "+displayName(m.Model))
	probeCtx, stopProbe := context.WithTimeout(ctx, 10*time.Minute)
	defer stopProbe()
	ready := make(chan error, 1)
	go func() { ready <- probe(probeCtx, "http://"+address, key, m.Model, loading.phase) }()
	select {
	case e = <-done:
		loading.fail()
		done <- e
		return fmt.Errorf("inference runtime exited before readiness (%v); run rec doctor", e)
	case <-ctx.Done():
		loading.cancel()
		return nil
	case e = <-ready:
		if e != nil {
			loading.fail()
			return fmt.Errorf("readiness failed: %w", e)
		}
	}
	loading.succeed()
	// Do not publish an endpoint from a child that died during the probe.
	select {
	case e = <-done:
		done <- e
		return fmt.Errorf("runtime exited: %v", e)
	default:
	}
	mu.Lock()
	status.Ready = true
	mu.Unlock()
	printReady(ui, m, status.Endpoint, len(background) > 0 && background[0])
	select {
	case <-ctx.Done():
		return nil
	case e = <-done:
		done <- e
		return fmt.Errorf("inference runtime exited: %v", e)
	}
}

func printReady(ui *screen, m Manifest, endpoint string, background bool) {
	var rows strings.Builder
	for i, row := range [][2]string{
		{"Base URL", endpoint},
		{"Model", m.Model},
		{"Checkpoint", m.Revision},
		{"Context", m.contextSummary()},
		{"API key", accentStyle.Render("rec auth token")},
	} {
		if i > 0 {
			rows.WriteString("\n")
		}
		rows.WriteString(faintStyle.Render(fmt.Sprintf("%-10s", row[0])) + "  " + row[1])
	}
	ui.println("")
	ui.println(panelStyle.Render(successStyle.Render("✓ ") + boldStyle.Render(m.Model+" is ready.") + "\n\n" + rows.String()))
	if background {
		ui.println(faintStyle.Render("Running in the background. Use `rec stop` to stop."))
	} else {
		ui.println(faintStyle.Render("Keep this terminal open. Press Ctrl+C to stop."))
	}
}
