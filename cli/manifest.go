package main

import (
	"bytes"
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

//go:embed assets/*
var assets embed.FS

var Version = "dev"
var identifier = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9._-]*$`)

type File struct {
	Name       string `json:"name"`
	URL        string `json:"url,omitempty"`
	Size       int64  `json:"size"`
	SHA256     string `json:"sha256"`
	Executable bool   `json:"executable,omitempty"`
}

type Manifest struct {
	Engine              string `json:"engine"`
	Schema              int    `json:"schema"`
	Model               string `json:"model"`
	Revision            string `json:"revision"`
	Development         bool   `json:"development"`
	OS                  string `json:"os"`
	Arch                string `json:"arch"`
	Backend             string `json:"backend"`
	Context             int    `json:"context"`
	MaxContext          int    `json:"max_context,omitempty"`
	MinRAM              int64  `json:"min_ram_bytes"`
	Files               []File `json:"files"`
	Entry               string `json:"entry"`
	RuntimeSHA          string `json:"runtime_sha256"`
	RuntimeFiles        []File `json:"runtime_files"`
	RuntimeBinarySHA    string `json:"runtime_binary_sha256,omitempty"`
	RuntimeSourceCommit string `json:"runtime_source_commit,omitempty"`
	RuntimeSourceTree   string `json:"runtime_source_tree,omitempty"`
}

func validHash(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == 32 && s == fmt.Sprintf("%x", b)
}

func safeName(s string) bool { return identifier.MatchString(s) && s != "." && s != ".." }

func safeRuntimeName(s string) bool {
	if s == "" || strings.Contains(s, "\\") || strings.HasPrefix(s, "/") {
		return false
	}
	for _, part := range strings.Split(s, "/") {
		if part == "" || part == "." || part == ".." {
			return false
		}
	}
	return filepath.ToSlash(filepath.Clean(s)) == s
}

const runtimeServer = "rec-mlx"
const nativeFlintRevision = "3b93f9053b530488405ae9fca51d4900d6ce21f0"
const nativeFlintMaxContext = 262144

func runtimeServerFor(m Manifest) (string, error) {
	switch {
	case m.Engine == "mlx" && m.OS == "darwin" && m.Arch == "arm64" && m.Backend == "metal":
		return runtimeServer, nil
	case m.Engine == "flint-native" && m.OS == "linux" && m.Arch == "amd64" && m.Backend == "hip":
		return "flint-serve", nil
	default:
		return "", fmt.Errorf("unsupported engine or target")
	}
}

func decodeManifest(data []byte) (Manifest, error) {
	var m Manifest
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&m); err != nil {
		return m, err
	}
	var extra any
	if d.Decode(&extra) != io.EOF {
		return m, fmt.Errorf("trailing manifest data")
	}
	// Each engine is one closed target tuple.
	serverName, err := runtimeServerFor(m)
	if err != nil {
		return m, err
	}
	if m.MaxContext < 0 || (m.MaxContext > 0 && m.Context > m.MaxContext) {
		return m, fmt.Errorf("invalid maximum context")
	}
	if m.Engine == "flint-native" &&
		(m.Revision != nativeFlintRevision || m.MaxContext != nativeFlintMaxContext || m.Context > nativeFlintMaxContext) {
		return m, fmt.Errorf("native Flint revision or context does not match the runtime contract")
	}
	if m.Engine == "flint-native" &&
		(!validHash(m.RuntimeBinarySHA) || !regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(m.RuntimeSourceCommit) ||
			!regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(m.RuntimeSourceTree)) {
		return m, fmt.Errorf("native Flint trusted runtime pins are missing")
	}
	if m.Schema != 1 || !safeName(m.Model) || m.Revision == "" || m.Context < 256 || m.MinRAM <= 0 || len(m.Files) == 0 {
		return m, fmt.Errorf("incomplete model descriptor")
	}
	if !m.Development && m.Model != "flint" {
		return m, fmt.Errorf("production packages must identify Flint")
	}
	found := false
	for groupIndex, group := range [][]File{m.Files, m.RuntimeFiles} {
		seen := map[string]bool{}
		for _, f := range group {
			validFile := safeRuntimeName(f.Name) && f.Size >= 0
			if groupIndex == 0 {
				validFile = safeName(f.Name) && f.Size > 0
			}
			if !validFile || seen[f.Name] || !validHash(f.SHA256) {
				return m, fmt.Errorf("invalid file descriptor: %q", f.Name)
			}
			seen[f.Name] = true
		}
	}
	for _, f := range m.Files {
		u, err := url.Parse(f.URL)
		if err != nil || u.Scheme != "https" || u.Host == "" || u.User != nil {
			return m, fmt.Errorf("model files require HTTPS URLs without credentials")
		}
		if f.Name == m.Entry {
			found = true
		}
	}
	if !found {
		return m, fmt.Errorf("entry must name a model file")
	}
	if !validHash(m.RuntimeSHA) || len(m.RuntimeFiles) == 0 {
		return m, fmt.Errorf("missing embedded runtime descriptor")
	}
	server := false
	for _, f := range m.RuntimeFiles {
		if f.Name == serverName && f.Executable {
			server = true
		}
	}
	if !server {
		return m, fmt.Errorf("runtime requires executable %s", serverName)
	}
	return m, nil
}

func loadManifest(path string) (Manifest, error) {
	var data []byte
	var err error
	if path != "" {
		data, err = os.ReadFile(path)
	} else {
		data, err = assets.ReadFile("assets/manifest.json")
		if os.IsNotExist(err) {
			return Manifest{}, fmt.Errorf("this CLI has no release bundle; build with packaging/package.py or supply --manifest and --runtime-dir for development")
		}
	}
	if err != nil {
		return Manifest{}, err
	}
	m, err := decodeManifest(data)
	if err == nil && (m.OS != runtime.GOOS || m.Arch != runtime.GOARCH) {
		err = fmt.Errorf("bundle targets %s/%s, this machine is %s/%s", m.OS, m.Arch, runtime.GOOS, runtime.GOARCH)
	}
	return m, err
}

func digest(b []byte) string        { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func modelDigest(m Manifest) string { b, _ := json.Marshal(m.Files); return digest(b) }

func verifyFile(path string, f File) error {
	return verifyFileInto(path, f, io.Discard)
}

// cachedSize reports whether path is a regular file of f's size.
func cachedSize(path string, f File) bool {
	st, err := os.Lstat(path)
	return err == nil && st.Mode().IsRegular() && st.Size() == f.Size
}

// verifyFileInto checks f's size and SHA-256, copying the hashed bytes to progress.
func verifyFileInto(path string, f File, progress io.Writer) error {
	st, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !st.Mode().IsRegular() || st.Size() != f.Size {
		return fmt.Errorf("size/type mismatch: %s", filepath.Base(path))
	}
	in, err := os.Open(path)
	if err != nil {
		return err
	}
	defer in.Close()
	h := sha256.New()
	if _, err = io.Copy(io.MultiWriter(h, progress), in); err != nil {
		return err
	}
	if hex.EncodeToString(h.Sum(nil)) != f.SHA256 {
		return fmt.Errorf("SHA-256 mismatch: %s", filepath.Base(path))
	}
	return nil
}
