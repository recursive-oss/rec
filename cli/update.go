//go:build darwin || linux

package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"time"
)

func update(args []string, in io.Reader, out io.Writer) error {
	if wantsHelp(args) {
		printUsage(out, "update")
		return nil
	}
	flags := flag.NewFlagSet("update", flag.ContinueOnError)
	yes := flags.Bool("y", false, "update without asking")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("usage: rec update [-y]")
	}
	ui := newScreen(out)
	latest, err := latestRelease()
	if err != nil {
		return err
	}
	if latest == Version {
		ui.succeeded("rec "+Version+" is up to date", "")
		return nil
	}
	ui.println("rec " + latest + " is available. This is rec " + Version + ".")
	if installedWithHomebrew() {
		ui.println("Homebrew installed this rec. Run " + boldStyle.Render("brew upgrade rec") + " to update it.")
		return nil
	}
	if !*yes {
		_, _ = fmt.Fprint(ui.out, boldStyle.Render("Install rec "+latest+"?")+faintStyle.Render(" [y/N] "))
		answer, _ := bufio.NewReader(in).ReadString('\n')
		if a := strings.ToLower(strings.TrimSpace(answer)); a != "y" && a != "yes" {
			ui.println("Nothing changed.")
			return nil
		}
	}
	installer, err := releaseInstaller(latest)
	if err != nil {
		return err
	}
	ui.println("")
	cmd := exec.Command("bash", "-s")
	cmd.Stdin = bytes.NewReader(installer)
	cmd.Stdout = out
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(), "REC_VERSION="+latest)
	return cmd.Run()
}

// updateSummary describes a newer release for rec doctor, or returns "" when rec is current.
func updateSummary() string {
	latest, err := latestRelease()
	if err != nil {
		return warningStyle.Render("could not check for a newer release")
	}
	if latest == Version {
		return ""
	}
	return warningStyle.Render("rec "+latest+" is available") + faintStyle.Render(" · run "+updateCommand())
}

var executable = os.Executable

func installedWithHomebrew() bool {
	self, err := executable()
	if err != nil {
		return false
	}
	if resolved, err := filepath.EvalSymlinks(self); err == nil {
		self = resolved
	}
	return strings.Contains(self, "/Cellar/rec/")
}

func updateCommand() string {
	if installedWithHomebrew() {
		return "brew upgrade rec"
	}
	return "rec update"
}

var (
	releasesURL = "https://github.com/recursive-oss/rec/releases"
	sourceURL   = "https://raw.githubusercontent.com/recursive-oss/rec"
)

// GitHub redirects releases/latest to the newest full release, so prereleases such as the OpenCode client never match.
func latestRelease() (string, error) {
	client := &http.Client{Timeout: 15 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	resp, err := client.Get(releasesURL + "/latest")
	if err != nil {
		return "", fmt.Errorf("release lookup failed: %w", err)
	}
	resp.Body.Close()
	location := resp.Header.Get("Location")
	version, ok := strings.CutPrefix(path.Base(location), "v")
	if resp.StatusCode != http.StatusFound || !ok || !strings.Contains(location, "/releases/tag/") || !identifier.MatchString(version) {
		return "", fmt.Errorf("no published release found")
	}
	return version, nil
}

func releaseInstaller(version string) ([]byte, error) {
	client := &http.Client{Timeout: time.Minute}
	resp, err := client.Get(sourceURL + "/v" + version + "/install.sh")
	if err != nil {
		return nil, fmt.Errorf("installer download failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("installer download failed: HTTP %d", resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}
