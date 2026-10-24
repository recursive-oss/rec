"""Backend-neutral rec HTTP protocol on a serialized ROCm engine."""

import hmac
import json
import logging
import re
import select
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from rec_mlx.anthropic import MessageEventStream, anthropic_error, chat_to_message, messages_to_chat
from rec_mlx.chat import seeded_tool_parser
from rec_mlx.engine import Sampling


def parsed_output(raw, tools, parser):
    reasoning = ""
    content = raw
    if raw.startswith("<think>"):
        if "</think>" in raw:
            reasoning, content = raw[7:].split("</think>", 1)
        else:
            reasoning, content = raw[7:], ""
    calls = []
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.S):
        call = parser(match[1], tools)
        calls.append(
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(
                        call["arguments"], ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
        )
    if "<tool_call>" in content:
        content = content.split("<tool_call>", 1)[0]
    for marker in ("<|end_of_text|>", "<|user|>", "<|observation|>"):
        content = content.split(marker, 1)[0]
    result = {"role": "assistant", "content": content}
    if reasoning:
        result["reasoning_content"] = reasoning
    if calls:
        result["tool_calls"] = calls
    return result


def make_server(
    engine, address, metrics_path=None, *, defaults=Sampling(), api_key=None, grammar_mode="flint"
):
    stopping, busy = threading.Event(), threading.Event()
    lock = threading.Lock()
    admission = threading.BoundedSemaphore(9)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def reply(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def reject(self, status, message):
            self.reply(
                status,
                anthropic_error(status, message)
                if self.path.split("?", 1)[0] == "/v1/messages"
                else {"error": message},
            )

        def authenticated(self):
            if api_key is None:
                return True
            bearer = hmac.compare_digest(
                self.headers.get("Authorization", "").encode(), ("Bearer " + api_key).encode()
            )
            anthropic = self.path.split("?", 1)[0] == "/v1/messages" and hmac.compare_digest(
                self.headers.get("x-api-key", "").encode(), api_key.encode()
            )
            if bearer or anthropic:
                return True
            self.reject(401, "Invalid or missing bearer token")
            self.close_connection = True
            return False

        def is_disconnected(self):
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                return bool(readable) and not self.connection.recv(1, socket.MSG_PEEK)
            except (OSError, ValueError):
                return True

        def do_GET(self):
            if not self.authenticated():
                return
            if self.path == "/health":
                self.reply(
                    503 if stopping.is_set() else 200,
                    {"status": "stopping" if stopping.is_set() else "ready", "busy": busy.is_set()},
                )
            elif self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": "flint", "object": "model"}]})
            else:
                self.reject(404, "Unknown endpoint")

        def prepare(self, body):
            if not isinstance(body, dict):
                raise ValueError("Request must be an object")
            if body.get("model") != "flint":
                raise ValueError("The served model is flint")
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a nonempty list")
            for message in messages:
                if not isinstance(message, dict) or message.get("role") not in (
                    "system",
                    "developer",
                    "user",
                    "assistant",
                    "tool",
                ):
                    raise ValueError("Unsupported message role")
                if "reasoning" in message:
                    if "reasoning_content" in message:
                        raise ValueError("Use one reasoning field per message")
                    message["reasoning_content"] = message.pop("reasoning")
                if isinstance(message.get("content"), list):
                    blocks = message["content"]
                    if any(
                        b.get("type") != "text" or not isinstance(b.get("text"), str)
                        for b in blocks
                    ):
                        raise ValueError("Text input only")
                    message["content"] = "\n\n".join(b["text"] for b in blocks)
            for field, neutral in [
                ("frequency_penalty", 0),
                ("min_p", 0),
                ("repetition_penalty", 1),
            ]:
                if body.get(field, neutral) != neutral:
                    raise ValueError(f"Unsupported non-neutral {field}")
            if body.get("store", False) is not False:
                raise ValueError("Stored-completion retrieval is unsupported")
            if body.get("tool_choice", "auto") not in ("auto", "none"):
                raise ValueError("tool_choice supports auto and none")
            if body.get("parallel_tool_calls", True) is not True:
                raise ValueError("parallel_tool_calls=false is unsupported")
            if type(body.get("stream", False)) is not bool:
                raise ValueError("stream must be a boolean")
            options = body.get("stream_options")
            if options is not None and options != {"include_usage": True}:
                raise ValueError("stream_options must be {include_usage: true}")
            if body.get("stream") and body.get("logprobs"):
                raise ValueError("Token logprobs require stream=false")
            mode = body.get("grammar", grammar_mode)
            if mode not in ("flint", "none"):
                raise ValueError("grammar must be flint or none")
            stop = body.get("stop") or []
            if isinstance(stop, str):
                stop = [stop]
            if not isinstance(stop, list) or any(not isinstance(s, str) or not s for s in stop):
                raise ValueError("Invalid stop strings")
            if mode == "flint" and stop:
                raise ValueError("Flint grammar cannot be combined with custom stop strings")
            tools = body.get("tools") if body.get("tool_choice", "auto") != "none" else None
            ids = engine.prompt(messages, tools=tools)
            remaining = engine.context_limit - len(ids)
            if remaining < 1:
                raise ValueError("The prompt fills the configured context limit")
            maximum = body.get("max_completion_tokens", body.get("max_tokens", remaining))
            if type(maximum) is not int or maximum < 1:
                raise ValueError("max_tokens must be positive integer")
            sampling = Sampling(
                temperature=body.get("temperature", defaults.temperature),
                top_p=body.get("top_p", defaults.top_p),
                top_k=body.get("top_k", defaults.top_k),
                presence_penalty=body.get("presence_penalty", 0),
                max_tokens=min(maximum, remaining),
                seed=body.get("seed", defaults.seed),
            )
            return ids, sampling, tools, mode, stop

        def do_POST(self):
            if not self.authenticated():
                return
            route = self.path.split("?", 1)[0]
            if route not in ("/v1/chat/completions", "/v1/messages"):
                self.reject(404, "Unknown endpoint")
                return
            if stopping.is_set():
                self.reject(503, "The server is stopping")
                return
            if not admission.acquire(False):
                self.reject(429, "The request queue is full")
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 8 * 1024**2:
                    raise ValueError("Request body must be 1 byte to 8 MiB")
                body = json.loads(self.rfile.read(length))
                thinking = True
                if route == "/v1/messages":
                    body, _, thinking = messages_to_chat(body)
                ids, sampling, tools, mode, stops = self.prepare(body)
                with lock:
                    if stopping.is_set():
                        self.reject(503, "The server is stopping")
                        return
                    if self.is_disconnected():
                        return
                    busy.set()
                    try:
                        self.complete(body, ids, sampling, tools, mode, stops, route, thinking)
                    finally:
                        busy.clear()
            except (ValueError, TypeError, KeyError) as error:
                if getattr(self, "started_stream", False):
                    self.close_connection = True
                    logging.exception("Stream failed")
                else:
                    self.reject(400, str(error))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                logging.exception("ROCm generation failed")
                if not getattr(self, "started_stream", False):
                    self.reject(500, "Generation failed; inspect server log")
            finally:
                admission.release()

        def complete(self, body, ids, sampling, tools, mode, stops, route, thinking):
            constraint = engine.constraints.tools(tools) if mode == "flint" else None
            stream = body.get("stream", False)
            identifier = "chatcmpl-" + uuid.uuid4().hex
            created = int(time.time())
            raw = ""
            prior = {"content": "", "reasoning_content": "", "tool_calls": []}
            metrics = {}
            logs = []
            finish = "length"
            generated_count = 0
            writer = self.wfile
            self.started_stream = False

            def emit(delta, finish_reason=None, usage=None):
                value = {
                    "id": identifier,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "flint",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                }
                if usage is not None:
                    value["usage"] = usage
                writer.write(("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode())
                writer.flush()

            events = engine.generate_tokens(
                ids,
                sampling,
                grammar=constraint,
                return_logprobs=bool(body.get("logprobs")),
                should_stop=lambda: stopping.is_set() or self.is_disconnected(),
            )
            try:
                for event in events:
                    if "metrics" in event:
                        metrics = event["metrics"]
                        continue
                    generated_count += 1
                    if stream and not self.started_stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.close_connection = True
                        if route == "/v1/messages":
                            writer = MessageEventStream(
                                self.wfile,
                                input_tokens=len(ids),
                                status=lambda: 200,
                                stop_sequences=tuple(stops),
                                include_thinking=thinking,
                            )
                        self.started_stream = True
                        emit({"role": "assistant", "content": ""})
                    raw += event["text"]
                    finish = event.get("finish_reason") or finish
                    matched = next((s for s in stops if s in raw), None)
                    if matched:
                        raw = raw.split(matched, 1)[0]
                        finish = "stop"
                    message = parsed_output(raw, tools, seeded_tool_parser(sampling, ids))
                    if "logprob" in event:
                        logs.append(
                            {
                                "id": event["token"],
                                "token": engine.tokenizer.decode([event["token"]]),
                                "logprob": event["logprob"],
                                "bytes": None,
                                "top_logprobs": [],
                            }
                        )
                    if stream:
                        delta = {}
                        for field in ("content", "reasoning_content"):
                            now = message.get(field, "")
                            before = prior.get(field, "")
                            if now.startswith(before) and now[len(before) :]:
                                delta["reasoning" if field == "reasoning_content" else field] = now[
                                    len(before) :
                                ]
                        new_calls = message.get("tool_calls", [])[
                            len(prior.get("tool_calls", [])) :
                        ]
                        if new_calls:
                            delta["tool_calls"] = [
                                {"index": len(prior.get("tool_calls", [])) + i, **c}
                                for i, c in enumerate(new_calls)
                            ]
                        if delta:
                            emit(delta)
                    prior = message
                    if matched:
                        break
            finally:
                events.close()
            message = parsed_output(raw, tools, seeded_tool_parser(sampling, ids))
            if message.get("tool_calls"):
                finish = "tool_calls"
            total_generated = metrics.get("generated_tokens", generated_count)
            usage = {
                "prompt_tokens": len(ids),
                "completion_tokens": total_generated,
                "total_tokens": len(ids) + total_generated,
            }
            if stream:
                emit({}, finish)
                if route == "/v1/messages" or body.get("stream_options") == {"include_usage": True}:
                    writer.write(
                        (
                            "data: "
                            + json.dumps(
                                {
                                    "id": identifier,
                                    "object": "chat.completion.chunk",
                                    "model": "flint",
                                    "choices": [],
                                    "usage": usage,
                                }
                            )
                            + "\n\n"
                        ).encode()
                    )
                writer.write(b"data: [DONE]\n\n")
                writer.flush()
            else:
                choice = {"index": 0, "message": message, "finish_reason": finish}
                if body.get("logprobs"):
                    choice["logprobs"] = {"content": logs}
                response = {
                    "id": identifier,
                    "object": "chat.completion",
                    "created": created,
                    "model": "flint",
                    "choices": [choice],
                    "usage": usage,
                    "timings": metrics,
                }
                if route == "/v1/messages":
                    response = chat_to_message(
                        response, stop_sequences=tuple(stops), include_thinking=thinking
                    )
                self.reply(200, response)
            if metrics_path is not None:
                with Path(metrics_path).open("a") as f:
                    f.write(json.dumps({**metrics, "request_id": identifier}) + "\n")

    from pathlib import Path

    server = ThreadingHTTPServer(address, Handler)
    server.daemon_threads = True
    server.generator = SimpleNamespace(
        cancel=stopping.set, join=lambda: None, stopping=stopping, busy=busy
    )
    return server
