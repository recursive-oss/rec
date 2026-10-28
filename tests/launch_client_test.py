#!/usr/bin/env python3

import http.server
import json
import os
import pathlib
import socketserver
import subprocess
import sys
import tempfile
import threading
import time

binary = str(pathlib.Path(sys.argv[1]).resolve())
clients = sys.argv[2:] or ["opencode", "pi", "omp"]
assert all(client in ("opencode", "pi", "omp") for client in clients)


class Quiet(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, data, content="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", content)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


profiles = [
    (client, context, (None, None, None, context))
    for client in clients
    for context in (262144, 32768)
]

for client, context, want in profiles:
    with tempfile.TemporaryDirectory(prefix="rec-wire-", dir="/tmp") as td:
        root = pathlib.Path(td)
        requests = []
        generation_lock = threading.Lock()
        busy_responses = []
        key = "a" * 64

        class API(Quiet):
            def do_POST(self):
                if not generation_lock.acquire(blocking=False):
                    busy_responses.append(self.path)
                    self.send_error(429, "Model busy")
                    return
                try:
                    time.sleep(2)
                    self.complete()
                finally:
                    generation_lock.release()

            def complete(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, self.headers.get("Authorization"), body))
                if body.get("stream"):
                    chunks = [
                        {
                            "id": "test",
                            "object": "chat.completion.chunk",
                            "model": "flint",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": "42"},
                                    "finish_reason": None,
                                }
                            ],
                        },
                        {
                            "id": "test",
                            "object": "chat.completion.chunk",
                            "model": "flint",
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            "usage": {
                                "prompt_tokens": 20,
                                "completion_tokens": 1,
                                "total_tokens": 21,
                            },
                        },
                    ]
                    self.send(
                        (
                            "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                            + "data: [DONE]\n\n"
                        ).encode(),
                        "text/event-stream",
                    )
                else:
                    self.send(
                        json.dumps(
                            {
                                "id": "test",
                                "object": "chat.completion",
                                "model": "flint",
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "42",
                                        },
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 20,
                                    "completion_tokens": 1,
                                    "total_tokens": 21,
                                },
                            }
                        ).encode()
                    )

        api = http.server.ThreadingHTTPServer(("127.0.0.1", 0), API)

        control_requests = []

        class Control(Quiet):
            def do_GET(self):
                control_requests.append(self.path)
                self.send(
                    json.dumps(
                        {
                            "ready": True,
                            "model": "flint",
                            "engine": "mlx",
                            "revision": "test",
                            "context": context,
                            "endpoint": f"http://127.0.0.1:{api.server_port}/v1",
                        }
                    ).encode()
                )

        state = root / "data/rec/state"
        state.mkdir(parents=True)
        config = root / "data/rec/config"
        config.mkdir(parents=True)
        (config / "api-key").write_text(key)
        control = socketserver.UnixStreamServer(str(state / "control.sock"), Control)
        for srv in (api, control):
            threading.Thread(target=srv.serve_forever, daemon=True).start()
        env = {
            k: v for k, v in os.environ.items() if not k.startswith(("OPENCODE_", "XDG_", "REC_"))
        }
        env.update(
            XDG_DATA_HOME=str(root / "data"),
            XDG_CACHE_HOME=str(root / "cache"),
            OPENCODE_TEST_HOME=str(root / "home"),
            PI_CODING_AGENT_DIR=str(root / "pi"),
            PWD=str(root),
        )
        for n in (
            "OPENCODE_DISABLE_PROJECT_CONFIG",
            "OPENCODE_DISABLE_CLAUDE_CODE",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS",
            "OPENCODE_DISABLE_AUTOUPDATE",
            "OPENCODE_DISABLE_MODELS_FETCH",
            "OPENCODE_DISABLE_LSP_DOWNLOAD",
        ):
            env[n] = "true"
        pi_dir = root / "pi"
        pi_dir.mkdir()
        auth = pi_dir / "auth.json"
        auth.write_text(json.dumps({"rec": {"type": "api_key", "key": "stale"}}))
        models = pi_dir / "models.json"
        models.write_text('{"providers": {}}\n')
        original_auth, original_models = auth.read_bytes(), models.read_bytes()
        extensions = pi_dir / "extensions"
        extensions.mkdir()
        (extensions / "legacy.mjs").write_text(
            "export default function(pi) { pi.on('before_provider_request', (e) => ({...e.payload, temperature:0.1, top_p:0.95, top_k:20, max_tokens:128})); }\n"
        )
        if client == "opencode":
            client_args = ["run", "--format", "json", "Answer with 42."]
        elif client == "omp":
            client_args = ["--no-skills", "--mode", "json", "-p", "Answer with 42."]
        else:
            client_args = [
                "--no-skills",
                "--no-context-files",
                "--no-prompt-templates",
                "--offline",
                "--no-session",
                "--mode",
                "json",
                "-p",
                "Answer with 42.",
            ]
        try:
            r = subprocess.run(
                [binary, "launch", client, "--", *client_args],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
            )
            assert r.returncode == 0, (r.returncode, r.stderr[-2000:], r.stdout[-2000:])
            assert requests, "No requests sent"
            assert not busy_responses, "Concurrent requests triggered busy retries"
            assert len(requests) == 1, "A simple turn must not trigger auxiliary generation"
            printed = subprocess.run(
                [binary, "launch", "--args", client, "--", *client_args],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout
            assert key not in printed, "rec launch --args must not print the API key"
            bin_dir = root / "bin"
            bin_dir.mkdir()
            (bin_dir / "rec").symlink_to(binary)
            r = subprocess.run(
                ["sh", "-c", printed],
                cwd=root,
                env={**env, "PATH": f"{bin_dir}:{env['PATH']}"},
                capture_output=True,
                text=True,
                timeout=90,
            )
            assert r.returncode == 0, (r.returncode, r.stderr[-2000:], r.stdout[-2000:])
            assert len(requests) == 2, "rec launch --args must reproduce the launched turn"
            if client == "pi":
                assert auth.read_bytes() == original_auth, "Pi auth.json was modified"
                assert models.read_bytes() == original_models, "Pi models.json was modified"
            assert (root / "data/rec/clients" / client / "user").is_dir(), "No rec user directory"
            assert not (root / "data/opencode").exists(), "OpenCode used the system data directory"
            for path, auth, body in requests:
                assert path == "/v1/chat/completions", path
                assert auth == "Bearer " + key, "wrong local API key"
                assert body["model"] == "flint", body["model"]
                actual = tuple(body.get(k) for k in ("temperature", "top_p", "top_k")) + (
                    body.get("max_tokens", body.get("max_completion_tokens")),
                )
                if client == "omp":
                    # omp reserves headroom for its system prompt and tools.
                    assert actual[:3] == want[:3], (actual, want)
                    assert 0 < actual[3] <= want[3], (actual, want)
                else:
                    assert actual == want, (actual, want)
            print(
                json.dumps(
                    {
                        "client": client,
                        "context": context,
                        "requests": len(requests),
                        "sampling_and_output": want,
                        "auth": "passed",
                    }
                ),
                flush=True,
            )
        finally:
            for srv in (api, control):
                srv.shutdown()
                srv.server_close()

for client in clients:
    with tempfile.TemporaryDirectory(prefix="rec-exec-", dir="/tmp") as td:
        root = pathlib.Path(td)
        env = {
            k: v for k, v in os.environ.items() if not k.startswith(("OPENCODE_", "XDG_", "REC_"))
        }
        env.update(XDG_DATA_HOME=str(root / "data"), PI_CODING_AGENT_DIR=str(root / "pi"))
        r = subprocess.run(
            [binary, "exec", client, "--version"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, (r.returncode, r.stderr[-2000:], r.stdout[-2000:])
        assert (r.stdout + r.stderr).strip(), "No version printed"
        assert not (root / "data/rec/state/control.sock").exists(), "rec exec started the server"
        assert not (root / "pi").exists(), "rec exec used the user's Pi directory"
        assert not (root / "data/opencode").exists(), "rec exec used the system data directory"
        print(json.dumps({"client": client, "exec": "passed"}), flush=True)
