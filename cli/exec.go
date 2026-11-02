//go:build darwin || linux

package main

import (
	"fmt"
	"os"
	"syscall"
)

func execClient(p Paths, args []string) error {
	if len(args) == 0 || args[0] == "--help" || args[0] == "-h" || args[0] == "help" {
		printUsage(os.Stdout, "exec")
		fmt.Println("\nRuns the harness rec launches, with its settings and sessions but without")
		fmt.Println("the server or the Flint provider. Arguments go to the harness unchanged.")
		return nil
	}
	if args[0] != "opencode" && args[0] != "pi" && args[0] != "omp" {
		return fmt.Errorf("unsupported harness %q; use rec exec pi, rec exec omp, or rec exec opencode", args[0])
	}
	rest := args[1:]
	if args[0] == "opencode" && len(rest) > 0 && (rest[0] == "upgrade" || rest[0] == "uninstall") {
		return fmt.Errorf("rec manages its OpenCode client; run rec update to update it")
	}
	client, err := installClient(p, args[0])
	if err != nil {
		return err
	}
	user, err := clientUserDir(p, args[0])
	if err != nil {
		return err
	}
	h := piUser(user)
	if args[0] == "omp" {
		h = ompUser(user)
	}
	if args[0] == "opencode" {
		h = openCodeUser(user)
	}
	return syscall.Exec(client, append([]string{client}, rest...), h.environ(os.Environ()))
}
