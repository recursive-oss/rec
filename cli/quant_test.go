package main

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestQuantValidation(t *testing.T) {
	for _, tc := range []struct {
		input    string
		expected string
		ok       bool
	}{
		{"", "int4", true},
		{"int4", "int4", true},
		{"INT4", "int4", true},
		{" int4 ", "int4", true},
		{"int6", "int6", true},
		{"INT6", "int6", true},
		{" int6 ", "int6", true},
		{"int8", "", false},
		{"fp16", "", false},
		{"invalid", "", false},
	} {
		got, err := validateQuant(tc.input)
		if (err == nil) != tc.ok {
			t.Errorf("validateQuant(%q): got error %v, expected ok=%v", tc.input, err, tc.ok)
		}
		if tc.ok && got != tc.expected {
			t.Errorf("validateQuant(%q) = %q, expected %q", tc.input, got, tc.expected)
		}
	}
}

func TestParseLaunchArgs(t *testing.T) {
	printOnly, quant, manifest, temp, presence, repeat, harness, rest, err := parseLaunchArgs([]string{"pi"})
	if err != nil || printOnly || quant != "int4" || manifest != "" || temp != nil || presence != nil || repeat != nil || harness != "pi" || len(rest) != 0 {
		t.Fatalf("unexpected defaults: printOnly=%v quant=%s temp=%v presence=%v repeat=%v harness=%s rest=%v err=%v", printOnly, quant, temp, presence, repeat, harness, rest, err)
	}

	printOnly, quant, manifest, temp, presence, repeat, harness, rest, err = parseLaunchArgs([]string{"--args", "--quant", "int6", "--temperature", "0.6", "--presence-penalty", "0.2", "--repeat-penalty", "1.1", "opencode", "--", "arg1", "arg2"})
	if err != nil || !printOnly || quant != "int6" || manifest != "" || temp == nil || *temp != 0.6 || presence == nil || *presence != 0.2 || repeat == nil || *repeat != 1.1 || harness != "opencode" || len(rest) != 2 || rest[0] != "arg1" || rest[1] != "arg2" {
		t.Fatalf("unexpected parsed args: printOnly=%v quant=%s temp=%v presence=%v repeat=%v harness=%s rest=%v err=%v", printOnly, quant, temp, presence, repeat, harness, rest, err)
	}

	// Flag with equals and after harness name
	_, quant, _, temp, presence, repeat, harness, _, err = parseLaunchArgs([]string{"pi", "--quant=int6", "--temperature=0.7", "--presence-penalty=0.3", "--repeat-penalty=1.2"})
	if err != nil || quant != "int6" || temp == nil || *temp != 0.7 || presence == nil || *presence != 0.3 || repeat == nil || *repeat != 1.2 || harness != "pi" {
		t.Fatalf("unexpected trailing flag parse: quant=%s temp=%v presence=%v repeat=%v harness=%s err=%v", quant, temp, presence, repeat, harness, err)
	}

	// Flag aliases: repeat_penalty and repetition-penalty
	_, _, _, _, _, repeat, _, _, err = parseLaunchArgs([]string{"pi", "--repeat_penalty=1.3"})
	if err != nil || repeat == nil || *repeat != 1.3 {
		t.Fatalf("unexpected repeat_penalty parse: repeat=%v err=%v", repeat, err)
	}
	_, _, _, _, _, repeat, _, _, err = parseLaunchArgs([]string{"pi", "--repetition-penalty", "1.4"})
	if err != nil || repeat == nil || *repeat != 1.4 {
		t.Fatalf("unexpected repetition-penalty parse: repeat=%v err=%v", repeat, err)
	}

	// Invalid temperature
	_, _, _, _, _, _, _, _, err = parseLaunchArgs([]string{"--temperature", "2.5", "pi"})
	if err == nil || !strings.Contains(err.Error(), "temperature must be between 0 and 2") {
		t.Fatalf("expected error for invalid temperature, got %v", err)
	}

	// Invalid presence penalty
	_, _, _, _, _, _, _, _, err = parseLaunchArgs([]string{"--presence-penalty", "3.0", "pi"})
	if err == nil || !strings.Contains(err.Error(), "presence-penalty must be between -2 and 2") {
		t.Fatalf("expected error for invalid presence-penalty, got %v", err)
	}

	// Invalid repeat penalty
	_, _, _, _, _, _, _, _, err = parseLaunchArgs([]string{"--repeat-penalty", "0.0", "pi"})
	if err == nil || !strings.Contains(err.Error(), "repeat-penalty must be greater than 0 and at most 5") {
		t.Fatalf("expected error for invalid repeat-penalty, got %v", err)
	}
	_, _, _, _, _, _, _, _, err = parseLaunchArgs([]string{"--repeat-penalty", "6.0", "pi"})
	if err == nil || !strings.Contains(err.Error(), "repeat-penalty must be greater than 0 and at most 5") {
		t.Fatalf("expected error for invalid repeat-penalty, got %v", err)
	}

	// Invalid quant
	_, _, _, _, _, _, _, _, err = parseLaunchArgs([]string{"--quant", "int8", "pi"})
	if err == nil || !strings.Contains(err.Error(), "invalid quantization") {
		t.Fatalf("expected error for invalid quant, got %v", err)
	}

	// Missing harness
	_, _, _, _, _, _, _, _, err = parseLaunchArgs([]string{"--quant", "int6"})
	if err == nil || !strings.Contains(err.Error(), "missing harness") {
		t.Fatalf("expected missing harness error, got %v", err)
	}
}

func TestReleaseManifestInt6(t *testing.T) {
	b, err := os.ReadFile("../packaging/flint-int6-mlx.json")
	if err != nil {
		t.Fatal(err)
	}
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	m.RuntimeSHA = digest([]byte("runtime"))
	m.RuntimeFiles = []File{{Name: runtimeServer, Size: 1, SHA256: digest([]byte("r")), Executable: true}}
	decoded, err := decodeManifest(mustJSON(t, m))
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Revision != "flint-int6" {
		t.Fatalf("expected revision flint-int6, got %s", decoded.Revision)
	}
	expectedMinRAM := int64(64 * 1024 * 1024 * 1024)
	if decoded.MinRAM != expectedMinRAM {
		t.Fatalf("expected 64 GiB min RAM (%d), got %d", expectedMinRAM, decoded.MinRAM)
	}
	if decoded.Context != 262144 || decoded.MaxContext != 262144 {
		t.Fatalf("expected context 262144, got %d / %d", decoded.Context, decoded.MaxContext)
	}
}

func TestLoadManifestWithQuant(t *testing.T) {
	// Custom manifest path should load regardless of quant
	_, err := loadManifest("../packaging/flint-int6-mlx.json", "int6")
	if err == nil {
		// Needs runtime descriptor
		t.Fatal("expected failure without runtime descriptor")
	}

	// Invalid quant should fail early
	_, err = loadManifest("", "int9")
	if err == nil || !strings.Contains(err.Error(), "invalid quantization") {
		t.Fatalf("expected invalid quant error, got %v", err)
	}
}

func TestLaunchQuantLabels(t *testing.T) {
	for _, quant := range []string{"int4", "int6"} {
		folder := "ccd8b361c654d3544e686c53a057f266135fa3851d18da32efaf5bddfb92b25a"
		label := "Starting Flint (" + quant + ") (" + folder + ")"
		doneLabel := "Flint (" + quant + ") (" + folder + ") is ready"
		if !strings.Contains(label, quant) || !strings.Contains(label, folder) || !strings.Contains(doneLabel, quant) || !strings.Contains(doneLabel, folder) {
			t.Fatalf("labels do not contain quant or folder: %s / %s", label, doneLabel)
		}
	}
}

func TestLaunchCommandSamplingParameters(t *testing.T) {
	temp := 0.6
	presence := 0.2
	repeat := 1.1
	m := Manifest{Model: "flint", Revision: "flint-int6", Context: 262144}
	_, args, _ := launchCommand(Paths{}, m, "/runtime", "/model/flint-int6.json", "127.0.0.1", 11500, &temp, &presence, &repeat)

	var foundTemp, foundPresence, foundRepeat bool
	for i, a := range args {
		if a == "--temperature" && i+1 < len(args) && args[i+1] == "0.6" {
			foundTemp = true
		}
		if a == "--presence-penalty" && i+1 < len(args) && args[i+1] == "0.2" {
			foundPresence = true
		}
		if a == "--repeat-penalty" && i+1 < len(args) && args[i+1] == "1.1" {
			foundRepeat = true
		}
	}
	if !foundTemp {
		t.Fatalf("expected --temperature 0.6 in launchCommand args, got %v", args)
	}
	if !foundPresence {
		t.Fatalf("expected --presence-penalty 0.2 in launchCommand args, got %v", args)
	}
	if !foundRepeat {
		t.Fatalf("expected --repeat-penalty 1.1 in launchCommand args, got %v", args)
	}
}

func TestSameInstanceSamplingParameters(t *testing.T) {
	temp1 := 0.6
	temp2 := 0.8
	presence1 := 0.2
	repeat1 := 1.1
	repeat2 := 1.3
	m := Manifest{Model: "flint", Revision: "flint-int6", RuntimeSHA: "runtime", Context: 262144}
	s := Status{
		Ready:           true,
		Model:           m.Model,
		Revision:        m.Revision,
		Runtime:         m.RuntimeSHA,
		Context:         m.Context,
		Instance:        instanceKey(m),
		Endpoint:        "http://127.0.0.1:11500/v1",
		Temperature:     &temp1,
		PresencePenalty: &presence1,
		RepeatPenalty:   &repeat1,
	}

	if !sameInstance(s, m, "127.0.0.1", 11500, &temp1, &presence1, &repeat1) {
		t.Fatal("expected matching instance to be accepted")
	}
	if sameInstance(s, m, "127.0.0.1", 11500, &temp2, &presence1, &repeat1) {
		t.Fatal("expected different temperature to be rejected")
	}
	if sameInstance(s, m, "127.0.0.1", 11500, &temp1, &presence1, &repeat2) {
		t.Fatal("expected different repeat penalty to be rejected")
	}
}
