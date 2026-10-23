"""Development stand-in for the Flint engine: serves a small stock MLX model.

Accepts the Flint server's command line and HTTP surface so that `rec serve smoke`
exercises rec's real path on any Apple Silicon Mac: runtime, download,
supervision, authentication, readiness and generation on Metal. It does not
exercise the Flint kernels. tests/build_smoke_runtime.py installs it.
"""

import argparse
import http.server
import json
import time
from pathlib import Path

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("command", choices=("serve",))
parser.add_argument("model")
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--context-limit", type=int, required=True)
parser.add_argument("--api-key-file", type=Path, required=True)
args = parser.parse_args()
key = args.api_key_file.read_text().strip()
model, tokenizer = load(str(args.model_dir))


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def authorized(self):
        if self.headers.get("Authorization") == "Bearer " + key:
            return True
        self.send_error(401)
        return False

    def send_json(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.authorized():
            return
        if self.path == "/health":
            self.send_json({"status": "ok"})
        elif self.path == "/v1/models":
            self.send_json({"object": "list", "data": [{"id": args.model, "object": "model"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.authorized():
            return
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = tokenizer.apply_chat_template(
            request["messages"], add_generation_prompt=True, tokenize=False
        )
        prompt_tokens = len(tokenizer.encode(prompt))
        limit = args.context_limit - prompt_tokens
        if limit < 1:
            self.send_error(400, "prompt exceeds the context limit")
            return
        max_tokens = min(
            request.get("max_tokens") or request.get("max_completion_tokens") or limit, limit
        )
        sampler = make_sampler(
            temp=request.get("temperature", 1.0),
            top_p=request.get("top_p", 1.0),
            top_k=request.get("top_k", 0),
        )
        chunk = {"id": "smoke", "created": int(time.time()), "model": args.model}
        stream = request.get("stream", False)
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
        text, finish, completion_tokens = "", "length", 0
        for response in stream_generate(
            model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
        ):
            completion_tokens += 1
            text += response.text
            if response.finish_reason:
                finish = response.finish_reason
            if stream and response.text:
                delta = {"role": "assistant", "content": response.text}
                event = dict(
                    chunk,
                    object="chat.completion.chunk",
                    choices=[{"index": 0, "delta": delta, "finish_reason": None}],
                )
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if stream:
            event = dict(
                chunk,
                object="chat.completion.chunk",
                choices=[{"index": 0, "delta": {}, "finish_reason": finish}],
                usage=usage,
            )
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\ndata: [DONE]\n\n")
            return
        message = {"role": "assistant", "content": text}
        self.send_json(
            dict(
                chunk,
                object="chat.completion",
                choices=[{"index": 0, "message": message, "finish_reason": finish}],
                usage=usage,
            )
        )


print(f"Listening on http://{args.host}:{args.port}/v1", flush=True)
http.server.HTTPServer((args.host, args.port), Handler).serve_forever()
