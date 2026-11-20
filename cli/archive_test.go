package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestUnpackStreamIntegrity(t *testing.T) {
	body := []byte("a pinned runtime executable")
	var encoded bytes.Buffer
	gz := gzip.NewWriter(&encoded)
	tarWriter := tar.NewWriter(gz)
	if err := tarWriter.WriteHeader(&tar.Header{Name: "server", Size: int64(len(body)), Mode: 0700, Typeflag: tar.TypeReg}); err != nil {
		t.Fatal(err)
	}
	if _, err := tarWriter.Write(body); err != nil {
		t.Fatal(err)
	}
	if err := tarWriter.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	m := Manifest{}
	m.RuntimeSHA = digest(encoded.Bytes())
	m.RuntimeFiles = []File{{Name: "server", Size: int64(len(body)), SHA256: digest(body), Executable: true}}
	dest := t.TempDir()
	if err := unpackStream(bytes.NewReader(encoded.Bytes()), dest, m); err != nil {
		t.Fatal(err)
	}
	actual, err := os.ReadFile(filepath.Join(dest, "server"))
	if err != nil || !bytes.Equal(actual, body) {
		t.Fatal("wrong extracted data", err)
	}
	for _, kind := range []string{"hash", "gzip-crc", "truncated", "trailing"} {
		t.Run(kind, func(t *testing.T) {
			data := append([]byte(nil), encoded.Bytes()...)
			manifest := m
			switch kind {
			case "hash":
				manifest.RuntimeSHA = strings.Repeat("0", 64)
			case "gzip-crc":
				data[len(data)-8] ^= 1
				manifest.RuntimeSHA = digest(data)
			case "truncated":
				data = data[:len(data)-1]
				manifest.RuntimeSHA = digest(data)
			case "trailing":
				data = append(data, []byte("not another gzip member")...)
				manifest.RuntimeSHA = digest(data)
			}
			if err := unpackStream(bytes.NewReader(data), t.TempDir(), manifest); err == nil {
				t.Fatal("accepted damaged archive")
			}
		})
	}
}

func TestUnpackStreamReaderFailure(t *testing.T) {
	if err := unpackStream(io.MultiReader(bytes.NewReader([]byte{0x1f, 0x8b}), failingReader{}), t.TempDir(), Manifest{}); err == nil {
		t.Fatal("discarded reader failure")
	}
}

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, fmt.Errorf("controlled read failure") }
