<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"
      srcset=".github/assets/recursive_logo_light.svg">
    <img src=".github/assets/recursive_logo_dark.svg" alt="Recursive"
      width="400">
  </picture>
</p>

`rec` is our custom inference engine for running [Recursive's language
models](https://www.recursive.com/updates/introducing-recursive-flint)
locally on your machine. It provides lightning-fast inference for
single-session usage. Compared to llama.cpp, rec delivers **1.4–2.6×** the
token throughput across the context lengths shown below on M4 Pro. Use it
with pi, OpenCode, or connect your own client through the local API.

<p align="center">
  <img src=".github/assets/throughput.svg"
    alt="Flint decode throughput: rec vs. llama.cpp at context lengths from 1K
    to 252K tokens" width="640">
</p>

<p align="center"><sub>Flint on Apple M4 Pro (48 GB), 4-bit weights,
1K–252K-token prompts and 1,024 output tokens. Five runs per context; prefill
excluded.</sub></p>

## Requirements

- Apple Silicon Mac with an M4 Pro, M4 Max, M5 Pro, or M5 Max
- At least 48 GB unified memory
- About 30 GB free disk space
- [Node.js 22](https://nodejs.org/en/download) and npm for pi


## Install

Install using:

```bash
curl -fsSL https://raw.githubusercontent.com/recursive-oss/rec/main/install.sh | bash
```

This installs `rec` in `~/.local/bin` and adds it to your PATH for new
terminals. All other data lives under `~/.local/share/rec`.


## Usage

We find the best performance from our models when paired with the
[pi](https://pi.dev/) harness. For convenience, rec provides pre-configured
recipes for running pi and [OpenCode](https://opencode.ai) harnesses. Run
them using:

```bash
rec launch pi  # preferred
rec launch opencode
```

⚠️ rec is optimized for single-session usage. To run multiple agents at a
time, see our [documentation](https://docs.recursive.com/getting-started) for
other ways to run our models.

This starts our inference server and connects to it automatically. The first
run downloads the model and harness. The model server continues to run after
you close the terminal. Run `rec stop` to stop it.

These harnesses are kept separate from anything on your system. Use
`rec exec pi|opencode ARGS` to run their own commands, for example:

```bash
rec exec pi install npm:PACKAGE
```


### Use your own harness

`rec` exposes OpenAI and Anthropic-compatible chat-completion APIs.
Start the server:

```bash
rec serve
```

Keep this terminal open. When `flint is ready.` appears, connect a client from
another terminal.

To expose the server on your LAN, bind to all interfaces:

```bash
rec serve --host 0.0.0.0
```

From another machine, connect to `http://<this-mac's-IP>:11500/v1` with the
`rec auth token` key. Traffic is unencrypted, so only use this on trusted
networks.

To use your own install of pi or OpenCode, run:

```bash
rec launch --args pi
rec launch --args opencode
```

Each command prints the environment and command line that `rec launch` uses,
instead of running it. Replace the harness path with the path to your own
harness. The command also points the harness at rec's own settings and
sessions; remove those variables to keep yours.

To send a request directly to the OpenAI-compatible API:

```bash
curl -fsS http://127.0.0.1:11500/v1/chat/completions \
  -H "Authorization: Bearer $(rec auth token)" \
  -H 'Content-Type: application/json' \
  -d '{"model":"flint","messages":[{"role":"user","content":"Say hello."}],"max_tokens":128}'
```

The response contains generated text in `choices`. For other clients, use
base URL `http://127.0.0.1:11500/v1`, model `flint`, and the API key from
`rec auth token`.

The Anthropic-compatible API is available at
`http://127.0.0.1:11500/v1/messages`. Point `ANTHROPIC_BASE_URL` at
`http://127.0.0.1:11500` and send the token from `rec auth token` as a bearer
token or as `x-api-key`. The model is `flint`.

<details>
<summary>Connect pi or OpenCode on another machine</summary>

On the Mac running rec, collect three values:

- **Base URL**: `http://<this-mac's-IP>:11500/v1`.
- **API key**: the output of `rec auth token`.
- **Context**: shown by `rec status` (262,144 tokens by default). Use it as both
  the context and output limit below. rec fits each reply to the space left.

On the other machine, make the key available to the harness:

```bash
export REC_API_KEY='<key from rec auth token>'
```

**pi**: add the provider to `~/.pi/agent/models.json`. If the file exists, merge
`rec` into its `providers`.

```json
{
  "providers": {
    "rec": {
      "baseUrl": "http://<server-mac's-IP>:11500/v1",
      "api": "openai-completions",
      "apiKey": "REC_API_KEY",
      "models": [{
        "id": "flint", "name": "Flint", "reasoning": false, "input": ["text"],
        "contextWindow": 262144, "maxTokens": 262144,
        "compat": {
          "supportsStore": false, "supportsDeveloperRole": false,
          "supportsReasoningEffort": false, "supportsStrictMode": false,
          "maxTokensField": "max_tokens"
        }
      }]
    }
  }
}
```

Select it with `/model`, or start pi with `pi --provider rec --model flint`.

**OpenCode 2.x**: add the provider to `~/.config/opencode/opencode.json`. If the
file exists, merge `rec` into its `providers`.

```json
{
  "model": "rec/flint",
  "providers": {
    "rec": {
      "package": "@opencode/ai/providers/openai-compatible",
      "name": "Recursive",
      "settings": { "baseURL": "http://<server-mac's-IP>:11500/v1", "apiKey": "{env:REC_API_KEY}" },
      "models": {
        "flint": {
          "name": "Flint",
          "limit": { "context": 262144, "output": 65536 },
          "body": { "max_completion_tokens": 262144 },
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"] }
        }
      }
    }
  }
}
```

`model` makes Flint the default; remove it to keep yours.

</details>


## Commands

| Command | What it does |
|---|---|
| `rec launch pi\|opencode` | Start the server if needed and open a harness |
| `rec exec pi\|opencode ARGS` | Run a harness command, such as installing a plugin |
| `rec serve` | Start the model server in this terminal |
| `rec status` | Show whether the server is running |
| `rec stop` | Stop the server |
| `rec auth token` | Print the local API key |
| `rec pull` | Download the model without starting the server |
| `rec doctor` | Check your hardware and installation |
| `rec update` | Update rec to the latest release |
| `rec uninstall` | Remove rec, its models and its settings |

Run `rec COMMAND --help` for a command's options.


## Help and contributing

- Questions, bugs, and feature requests:
  [GitHub Issues](https://github.com/recursive-oss/rec/issues).
- Contribution guide and local checks: [CONTRIBUTING.md](CONTRIBUTING.md).
- Community guidelines: [Code of conduct](CODE_OF_CONDUCT.md).
- Security vulnerabilities: [Private reporting instructions](SECURITY.md).
- License: [Apache 2.0](LICENSE).
