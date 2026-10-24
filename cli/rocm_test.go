package main

import (
	"encoding/json"
	"path/filepath"
	"testing"
)

func TestROCmManifestSelectsOwnRuntime(t *testing.T) {
	m := Manifest{Schema: 1, Engine: "rocm", Model: "flint", Revision: "pinned", Development: true,
		OS: "linux", Arch: "amd64", Backend: "hip", Context: 8192, MaxContext: 65536, MinRAM: 24 << 30,
		Files: []File{{Name: "flint-int4.json", URL: "https://example.com/flint-int4.json", Size: 1, SHA256: digest([]byte("x"))}},
		Entry: "flint-int4.json", RuntimeSHA: digest([]byte("runtime")),
		RuntimeFiles: []File{{Name: "rec-rocm", Size: 1, SHA256: digest([]byte("r")), Executable: true}}}
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := decodeManifest(data)
	if err != nil {
		t.Fatal(err)
	}
	server, _, _ := launchCommand(Paths{}, decoded, "/runtime", "/model/flint-int4.json", 11500)
	if server != filepath.Join("/runtime", "rec-rocm") {
		t.Fatalf("ROCm selected wrong runtime: %s", server)
	}
	m.RuntimeFiles[0].Name = "rec-mlx"
	data, _ = json.Marshal(m)
	if _, err := decodeManifest(data); err == nil {
		t.Fatal("ROCm target accepted the Apple runtime")
	}
}
