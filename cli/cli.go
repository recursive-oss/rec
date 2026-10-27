package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
)

func validateModelSource(command string, m Manifest, modelDir string) error {
	if m.Engine != "flint-native" {
		return nil
	}
	if command == "pull" {
		return fmt.Errorf("native Flint checkpoints are local-only; rec pull is unavailable")
	}
	if command == "serve" && modelDir == "" {
		return fmt.Errorf("native Flint requires --model-dir DIR")
	}
	return nil
}

func run(args []string) error {
	background := len(args) > 0 && args[0] == "_background"
	if background {
		args = append([]string{"serve"}, args[1:]...)
	}
	if len(args) > 0 && args[0] == "_supervise" {
		return supervise(args[1:])
	}
	if len(args) == 0 || args[0] == "help" || args[0] == "--help" || args[0] == "-h" {
		printHelp(os.Stdout)
		return nil
	}
	if args[0] == "--version" {
		fmt.Println("rec " + Version)
		return nil
	}
	p, e := paths()
	if e != nil {
		return e
	}
	switch args[0] {
	case "launch":
		return launch(p, args[1:])
	case "exec":
		return execClient(p, args[1:])
	case "auth":
		if wantsHelp(args[1:]) {
			printUsage(os.Stdout, "auth")
			return nil
		}
		if len(args) != 2 || args[1] != "token" {
			return fmt.Errorf("usage: rec auth token")
		}
		t, e := token(p, false)
		if e == nil {
			fmt.Println(t)
		}
		return e
	case "status", "stop":
		if wantsHelp(args[1:]) {
			printUsage(os.Stdout, args[0])
			return nil
		}
		if len(args) != 1 {
			return fmt.Errorf("usage: rec %s", args[0])
		}
		s, e := control(p, args[0])
		if e != nil {
			return e
		}
		ui := newScreen(os.Stdout)
		if !ui.live {
			return json.NewEncoder(os.Stdout).Encode(s)
		}
		printStatus(ui, args[0], s)
		return nil
	case "update":
		return update(args[1:], os.Stdin, os.Stdout)
	case "uninstall":
		return uninstall(p, args[1:], os.Stdin, os.Stdout)
	case "serve", "pull", "doctor":
	default:
		return fmt.Errorf("unknown command %q; use rec --help", args[0])
	}
	command := args[0]
	rest := args[1:]
	if wantsHelp(rest) {
		printUsage(os.Stdout, command)
		return nil
	}
	name := ""
	if command != "doctor" && len(rest) > 0 && !strings.HasPrefix(rest[0], "-") {
		name = rest[0]
		rest = rest[1:]
	}
	flags := flag.NewFlagSet(command, flag.ContinueOnError)
	manifest := flags.String("manifest", "", "")
	runtimeDir := flags.String("runtime-dir", "", "")
	var contextTokens int
	var modelDir string
	port := 11500
	if command == "serve" {
		flags.IntVar(&contextTokens, "context", 0, "")
		flags.IntVar(&port, "port", port, "")
		flags.StringVar(&modelDir, "model-dir", "", "")
	}
	flags.SetOutput(io.Discard)
	if e := flags.Parse(rest); e != nil {
		return fmt.Errorf("%w; see rec %s --help", e, command)
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected arguments; see rec %s --help", command)
	}
	if port < 1 || port > 65535 {
		return fmt.Errorf("port must be between 1 and 65535")
	}
	m, e := loadManifest(*manifest)
	if e == nil && command == "serve" {
		flags.Visit(func(f *flag.Flag) {
			if f.Name == "context" {
				m, e = withContext(m, contextTokens)
			}
		})
	}
	if command == "doctor" {
		return doctor(p, m, e)
	}
	if e != nil {
		return e
	}
	if name != "" && m.Model != name {
		return fmt.Errorf("this bundle contains %q, not %q", m.Model, name)
	}
	if *runtimeDir != "" && *manifest == "" {
		return fmt.Errorf("--runtime-dir requires --manifest")
	}
	if modelDir != "" && m.Engine != "flint-native" {
		return fmt.Errorf("--model-dir is supported only by the native Flint engine")
	}
	if e := validateModelSource(command, m, modelDir); e != nil {
		return e
	}
	if m.Development {
		newScreen(os.Stderr).step(warningStyle.Render("!"), "Development bundle: this is not a qualified Flint release.", "")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM, syscall.SIGHUP)
	defer stop()
	if command == "pull" {
		path, e := ensureModel(ctx, p, m, os.Stderr)
		if e == nil {
			fmt.Println(path)
		}
		return e
	}
	return serve(ctx, p, m, *runtimeDir, modelDir, port, os.Stderr, background)
}

func doctor(p Paths, m Manifest, manifestErr error) error {
	ui := newScreen(os.Stdout)
	ui.println(boldStyle.Render("rec " + Version))
	ui.println("")
	rows := [][2]string{{"Platform", runtime.GOOS + "/" + runtime.GOARCH}, {"Data", p.Data}}
	if update := updateSummary(); update != "" {
		rows = append(rows, [2]string{"Update", update})
	}
	if manifestErr == nil {
		rows = append(rows,
			[2]string{"Model", m.Model + faintStyle.Render(" ("+m.Revision+")")},
			[2]string{"Engine", m.Engine + faintStyle.Render(" ("+m.Backend+")")},
			[2]string{"Context", m.contextSummary()})
		if m.Development {
			rows = append(rows, [2]string{"Bundle", warningStyle.Render("development, not a qualified release")})
		}
	}
	ui.table(rows)
	ui.println("")
	if manifestErr != nil {
		ui.failed("No release bundle", "")
		return manifestErr
	}
	if err := preflight(m); err != nil {
		ui.failed("This system does not meet the requirements", "")
		return err
	}
	ui.succeeded("This system meets the requirements", "")
	if !modelFilesPresent(p, m) {
		ui.step(warningStyle.Render("!"), displayName(m.Model)+" is not downloaded", "run rec pull")
		return nil
	}
	id, err := verifyModel(p, m, os.Stdout)
	if err != nil {
		return err
	}
	ui.println(faintStyle.Render("    SHA-256 " + id))
	return nil
}

func printStatus(ui *screen, action string, s Status) {
	switch {
	case action == "stop":
		ui.step(accentStyle.Render("■"), "Stopping "+s.Model, "")
	case s.Ready:
		ui.step(successStyle.Render("●"), boldStyle.Render(s.Model)+" is running", s.Endpoint)
	default:
		ui.step(warningStyle.Render("●"), boldStyle.Render(s.Model)+" is starting", "")
	}
	if action == "status" {
		ui.println(faintStyle.Render(fmt.Sprintf("    Context %s tokens · rec %s", formatTokens(s.Context), s.Version)))
	}
}
