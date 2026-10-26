package main

import (
	"fmt"
	"io"
	"slices"
	"strings"

	"charm.land/lipgloss/v2"
)

type commandHelp struct {
	name, args, summary string
	options             [][2]string
}

var quantOption = [2]string{"--quant FORMAT", "Model quantization: int4 (default, 48 GiB+ RAM) or int6 (64 GiB+ RAM)"}
var temperatureOption = [2]string{"--temperature NUM", "Sampling temperature between 0 and 2 (default: 1.0)"}
var presenceOption = [2]string{"--presence-penalty NUM", "Presence penalty between -2 and 2 (default: 0.0)"}
var repeatOption = [2]string{"--repeat-penalty NUM", "Repetition penalty greater than 0 (default: 1.0)"}

var developmentOptions = [][2]string{
	{"--manifest FILE", "Use this manifest instead of the bundled one"},
	{"--runtime-dir DIR", "Use this runtime instead of the bundled one; needs --manifest"},
}

var commands = []commandHelp{
	{"launch", "pi|opencode", "Start the server if needed and open a harness", [][2]string{
		{"--args", "Print the harness command instead of running it; needs a running server"},
		temperatureOption,
		presenceOption,
		repeatOption,
		quantOption,
	}},
	{"exec", "pi|opencode ARGS", "Run a harness command, such as installing a plugin", nil},
	{"serve", "[MODEL]", "Start the model server in this terminal", append([][2]string{
		{"--context TOKENS", "Total input and output tokens (default: set by the release)"},
		{"--port PORT", "Local port (default: 11500)"},
		{"--host HOST", "Network interface to bind the server to (default: 127.0.0.1)"},
		temperatureOption,
		presenceOption,
		repeatOption,
		quantOption,
	}, developmentOptions...)},
	{"status", "", "Show whether the server is running", nil},
	{"stop", "", "Stop the server", nil},
	{"auth token", "", "Print the local API key", nil},
	{"pull", "[MODEL]", "Download the model without starting the server", append([][2]string{quantOption}, developmentOptions...)},
	{"doctor", "", "Check your hardware and installation", append([][2]string{quantOption}, developmentOptions...)},
	{"update", "[-y]", "Update rec to the latest release", [][2]string{{"-y", "Do not ask for confirmation"}}},
	{"uninstall", "[-y]", "Remove rec, its models and its settings", [][2]string{{"-y", "Do not ask for confirmation"}}},
	{"--version", "", "Print the version", nil},
}

func (c commandHelp) usage() string { return strings.TrimSpace("rec " + c.name + " " + c.args) }

func helpRow(name, summary string) string {
	return fmt.Sprintf("  %s %s\n", accentStyle.Render(fmt.Sprintf("%-24s", name)), summary)
}

func printHelp(out io.Writer) {
	_, _ = lipgloss.Fprint(out, "rec runs Recursive's language models locally.\n\n"+boldStyle.Render("Usage:")+"\n")
	for _, c := range commands {
		_, _ = lipgloss.Fprint(out, helpRow(c.usage(), c.summary))
	}
	_, _ = lipgloss.Fprint(out, "\n")
	for _, line := range []string{"MODEL defaults to the model in this release.", "Run rec COMMAND --help for a command's options."} {
		_, _ = lipgloss.Fprint(out, faintStyle.Render(line)+"\n")
	}
}

func printUsage(out io.Writer, name string) {
	var options [][2]string
	_, _ = lipgloss.Fprint(out, boldStyle.Render("Usage:")+"\n")
	for _, c := range commands {
		if c.name == name || strings.HasPrefix(c.name, name+" ") {
			_, _ = lipgloss.Fprint(out, helpRow(c.usage(), c.summary))
			options = append(options, c.options...)
		}
	}
	if len(options) > 0 {
		_, _ = lipgloss.Fprint(out, "\n"+boldStyle.Render("Options:")+"\n")
		for _, o := range options {
			_, _ = lipgloss.Fprint(out, helpRow(o[0], o[1]))
		}
	}
}

func wantsHelp(args []string) bool {
	return slices.ContainsFunc(args, func(a string) bool { return a == "-h" || a == "--help" || a == "-help" })
}
