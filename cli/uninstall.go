//go:build darwin || linux

package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Removes the managed ~/.local/bin/rec link and the data directory. A link that
// points elsewhere belongs to another program and is left alone.
func uninstall(p Paths, args []string, in io.Reader, out io.Writer) error {
	if wantsHelp(args) {
		printUsage(out, "uninstall")
		return nil
	}
	flags := flag.NewFlagSet("uninstall", flag.ContinueOnError)
	yes := flags.Bool("y", false, "remove without asking")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("usage: rec uninstall [-y]")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	bin := filepath.Join(home, ".local/bin")
	var links []string
	if managedLink(filepath.Join(bin, "rec"), func(target string) bool {
		ok, _ := filepath.Match(filepath.Join(p.Data, "releases", "*", "rec"), target)
		return ok
	}) {
		links = append(links, filepath.Join(bin, "rec"))
	}
	size, err := dirSize(p.Data)
	if err != nil {
		return err
	}
	homebrew := installedWithHomebrew()
	removes := "rec, its downloaded models"
	if homebrew {
		removes = "rec's downloaded models"
	}
	ui := newScreen(out)
	ui.println("This removes " + removes + ", its settings and its pi and OpenCode sessions:")
	for _, link := range links {
		ui.println("  " + link)
	}
	ui.println("  " + p.Data + faintStyle.Render(" ("+formatBytes(size)+")"))
	if !*yes {
		_, _ = fmt.Fprint(ui.out, boldStyle.Render("Remove?")+faintStyle.Render(" [y/N] "))
		answer, _ := bufio.NewReader(in).ReadString('\n')
		if a := strings.ToLower(strings.TrimSpace(answer)); a != "y" && a != "yes" {
			ui.println("Nothing removed.")
			return nil
		}
	}
	if err := stopServer(p, out); err != nil {
		return err
	}
	for _, link := range links {
		if err := os.Remove(link); err != nil {
			return err
		}
	}
	if err := os.RemoveAll(p.Data); err != nil {
		return err
	}
	if homebrew {
		ui.succeeded("Removed rec's models and settings", "")
		ui.println(faintStyle.Render("Run brew uninstall rec to remove the rec command."))
		return nil
	}
	ui.succeeded("Removed rec", "")
	ui.println(faintStyle.Render("Your shell configuration still adds ~/.local/bin to PATH after \"# Recursive CLI PATH\"."))
	return nil
}

func managedLink(path string, owned func(target string) bool) bool {
	target, err := os.Readlink(path)
	return err == nil && owned(target)
}

func dirSize(dir string) (int64, error) {
	var size int64
	err := filepath.WalkDir(dir, func(_ string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.Type().IsRegular() {
			info, err := d.Info()
			if err != nil {
				return err
			}
			size += info.Size()
		}
		return nil
	})
	return size, err
}

func stopOlderServer(p Paths, out io.Writer) error {
	s, err := control(p, "status")
	if err != nil || s.Version == Version {
		return nil
	}
	return stopServer(p, out)
}

func stopServer(p Paths, out io.Writer) error {
	name := filepath.Join(p.State, "instance.lock")
	deadline := time.Now().Add(30 * time.Second)
	for announced, requested := false, false; ; {
		if l, err := lock(name); err == nil {
			unlock(l)
			return nil
		}
		if !announced {
			newScreen(out).step(accentStyle.Render("■"), "Stopping the running model", "")
			announced = true
		}
		// A starting server holds the lock briefly before its socket accepts requests.
		if !requested {
			_, err := control(p, "stop")
			requested = err == nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("the running model did not stop; run rec stop and retry")
		}
		time.Sleep(100 * time.Millisecond)
	}
}
