//go:build darwin || linux

package main

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fakeReleases(t *testing.T) string {
	dir := t.TempDir()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/releases/latest":
			http.Redirect(w, r, "/releases/tag/v0.3.0", http.StatusFound)
		case "/source/v0.3.0/install.sh":
			fmt.Fprintf(w, "echo \"$REC_VERSION\" > %q\n", filepath.Join(dir, "installed"))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(s.Close)
	setURLs(t, s.URL+"/releases", s.URL+"/source")
	return dir
}

func setURLs(t *testing.T, releases, source string) {
	previousReleases, previousSource := releasesURL, sourceURL
	releasesURL, sourceURL = releases, source
	t.Cleanup(func() { releasesURL, sourceURL = previousReleases, previousSource })
}

func setVersion(t *testing.T, version string) {
	previous := Version
	Version = version
	t.Cleanup(func() { Version = previous })
}

func TestUpdateInstallsLatestRelease(t *testing.T) {
	dir := fakeReleases(t)
	setVersion(t, "0.2.0")
	var out bytes.Buffer
	if err := update(nil, strings.NewReader("n\n"), &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "rec 0.3.0 is available") || !strings.Contains(out.String(), "Nothing changed.") {
		t.Fatalf("output:\n%s", out.String())
	}
	if _, err := os.Stat(filepath.Join(dir, "installed")); !os.IsNotExist(err) {
		t.Fatal("declining ran the installer")
	}

	if err := update([]string{"-y"}, strings.NewReader(""), &out); err != nil {
		t.Fatal(err)
	}
	if got, err := os.ReadFile(filepath.Join(dir, "installed")); err != nil || string(got) != "0.3.0\n" {
		t.Fatalf("installed = %q, %v", got, err)
	}
}

func TestUpdateKeepsLatestRelease(t *testing.T) {
	dir := fakeReleases(t)
	setVersion(t, "0.3.0")
	var out bytes.Buffer
	if err := update([]string{"-y"}, strings.NewReader(""), &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "rec 0.3.0 is up to date") {
		t.Fatalf("output:\n%s", out.String())
	}
	if _, err := os.Stat(filepath.Join(dir, "installed")); !os.IsNotExist(err) {
		t.Fatal("ran the installer for the installed version")
	}
}

func TestDoctorReportsNewerRelease(t *testing.T) {
	fakeReleases(t)
	setVersion(t, "0.2.0")
	if got := updateSummary(); !strings.Contains(got, "rec 0.3.0 is available") || !strings.Contains(got, "run rec update") {
		t.Fatalf("summary = %q", got)
	}
	setVersion(t, "0.3.0")
	if got := updateSummary(); got != "" {
		t.Fatalf("summary for latest release = %q", got)
	}
}

func TestDoctorSurvivesFailedReleaseLookup(t *testing.T) {
	s := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(s.Close)
	setURLs(t, s.URL+"/releases", s.URL+"/source")
	if got := updateSummary(); !strings.Contains(got, "could not check") {
		t.Fatalf("summary = %q", got)
	}
}

func installWithHomebrew(t *testing.T) {
	prefix := t.TempDir()
	cellar := filepath.Join(prefix, "Cellar/rec/0.2.0/bin")
	for _, dir := range []string{cellar, filepath.Join(prefix, "bin")} {
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(cellar, "rec"), []byte("rec"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("../Cellar/rec/0.2.0/bin/rec", filepath.Join(prefix, "bin/rec")); err != nil {
		t.Fatal(err)
	}
	previous := executable
	executable = func() (string, error) { return filepath.Join(prefix, "bin/rec"), nil }
	t.Cleanup(func() { executable = previous })
}

func TestUpdateLeavesHomebrewInstallToBrew(t *testing.T) {
	dir := fakeReleases(t)
	setVersion(t, "0.2.0")
	installWithHomebrew(t)
	var out bytes.Buffer
	if err := update([]string{"-y"}, strings.NewReader(""), &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "rec 0.3.0 is available") || !strings.Contains(out.String(), "brew upgrade rec") {
		t.Fatalf("output:\n%s", out.String())
	}
	if _, err := os.Stat(filepath.Join(dir, "installed")); !os.IsNotExist(err) {
		t.Fatal("ran the installer for a Homebrew install")
	}
	if got := updateSummary(); !strings.Contains(got, "run brew upgrade rec") {
		t.Fatalf("summary = %q", got)
	}
}
