//go:build darwin || linux

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExecRejectsBeforeInstallingClient(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	for _, tc := range []struct {
		args []string
		want string
	}{
		{[]string{"codex"}, "unsupported harness"},
		{[]string{"opencode", "upgrade"}, "rec update"},
		{[]string{"opencode", "uninstall"}, "rec update"},
	} {
		p := dataPaths(t.TempDir())
		if err := execClient(p, tc.args); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("%v: %v, want %q", tc.args, err, tc.want)
		}
		if _, err := os.Stat(filepath.Join(p.Data, "clients")); !os.IsNotExist(err) {
			t.Fatalf("%v: installed a client: %v", tc.args, err)
		}
	}
}

func TestExecPointsHomebrewInstallToBrew(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	installWithHomebrew(t)
	if err := execClient(dataPaths(t.TempDir()), []string{"opencode", "upgrade"}); err == nil || !strings.Contains(err.Error(), "brew upgrade rec") {
		t.Fatalf("error = %v", err)
	}
}

func TestClientUserDirIsPrivate(t *testing.T) {
	p := dataPaths(t.TempDir())
	dir, err := clientUserDir(p, "pi")
	if err != nil || dir != filepath.Join(p.Data, "clients", "pi", "user") {
		t.Fatalf("user dir %q: %v", dir, err)
	}
	if info, err := os.Stat(dir); err != nil || info.Mode().Perm() != 0700 {
		t.Fatalf("user dir mode: %v %v", info, err)
	}
}
