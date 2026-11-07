package main

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestManifestForMemory(t *testing.T) {
	primary := Manifest{Revision: "int4", Context: 262144, Entry: "flint-int4.json"}
	low := Manifest{Revision: "mixed", Context: 131072, Entry: "config.json"}
	for _, gib := range []int64{16, 24, 32, 36, 47, 48, 64, 128} {
		want := primary
		if gib < 48 {
			want = low
		}
		got := manifestForMemory(primary, low, gib<<30)
		if got.Revision != want.Revision || got.Context != want.Context || got.Entry != want.Entry {
			t.Fatalf("%d GiB: got %+v, want %+v", gib, got, want)
		}
	}
}

func TestEmbeddedLowMemoryManifest(t *testing.T) {
	data, err := assets.ReadFile("assets/manifest-low-memory.json")
	if os.IsNotExist(err) {
		t.Skip("no alternate profile packaged")
	}
	if err != nil {
		t.Fatal(err)
	}
	low, err := decodeManifest(data)
	if err != nil {
		t.Fatal(err)
	}
	data, err = assets.ReadFile("assets/manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	primary, err := decodeManifest(data)
	if err != nil {
		t.Fatal(err)
	}
	t.Run("explicit profile", func(t *testing.T) {
		if runtime.GOOS != primary.OS || runtime.GOARCH != primary.Arch {
			t.Skip("explicit profiles require a matching host target")
		}
		// An explicit manifest must retain the requested profile on a 32 GiB Mac.
		path := filepath.Join(t.TempDir(), "manifest.json")
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
		explicit, err := loadManifest(path)
		if err != nil {
			t.Fatal(err)
		}
		if explicit.Revision != primary.Revision {
			t.Fatal("explicit profile was replaced")
		}
	})
	if low.RuntimeSHA != primary.RuntimeSHA || low.MinRAM != 32<<30 || low.Context != 131072 || low.Entry != "config.json" {
		t.Fatalf("invalid low-memory profile: %+v", low)
	}
}
