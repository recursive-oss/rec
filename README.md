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

- Apple Silicon Mac with an M4 Pro, M4 Max, M5 Pro, M5 Max, or M6
- At least 32 GB unified memory
- About 30 GB free disk space
- [Node.js 22](https://nodejs.org/en/download) and npm for pi

On Macs with 32 GB (or less than 48 GB), packaged builds select the community
[Flint mixed 3/4-bit checkpoint](https://huggingface.co/mlx-community/flint-mixed-3-4bit)
(17.8 GB), with a 131,072-token default context. Macs with 48 GB or more use the
official INT4 checkpoint. The mixed checkpoint is an unofficial community conversion.
Use `rec doctor` to see the selected revision and context.

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

After each prompt finishes processing, the server logs prefill throughput:

```text
Prefill: prompt=74051 common_prefix=62433 reused=62433 new=11618 200 t/s
```

The rate is newly processed prompt tokens divided by prefill time. Fully cached
prompts report `n/a t/s`. This measures prefill, rather than generated-token speed.

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

## Runtime build

The bundled runtime uses CPython 3.14.10, nanobind 3.1.0, and MLX built from
GitHub commit `fc344c986c4169d99f76b5d95ae6b4a58a359a63` (HEAD when upgraded).
