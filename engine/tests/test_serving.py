import http.client
import json
import socket
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rec_mlx.server import make_server


class Tokenizer:
    chat_template = "test"

    def get_vocab(self):
        return {}

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def convert_ids_to_tokens(self, token):
        return chr(token)

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


class Engine:
    tokenizer = Tokenizer()
    eos = {0}
    context_limit = 32
    identity = {"model_id": "test"}
    startup = {"kv_format": "test"}

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.failure = None
        self.constraint = SimpleNamespace(stops=(0,))
        self.constraints = SimpleNamespace(tools=Mock(return_value=self.constraint))
        self.grammars = []

    def prepare_prefix(self, *, should_stop):
        return None

    def prompt(self, messages, tools=None):
        return [ord(c) for c in messages[-1]["content"]]

    def generate_tokens(self, ids, sampling, *, should_stop, return_logprobs=False, grammar=None):
        self.calls.append((chr(ids[0]), sampling.max_tokens))
        self.grammars.append(grammar)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        generated = 0
        try:
            if ids[0] == ord("A"):
                self.started.set()
                while not self.release.wait(0.01):
                    if should_stop():
                        break
            if self.failure is not None:
                raise self.failure
            for index in range(sampling.max_tokens):
                if should_stop():
                    self.cancelled.set()
                    break
                generated += 1
                yield {
                    "token": ord("x"),
                    "text": "x",
                    "finish_reason": "length" if index + 1 == sampling.max_tokens else None,
                    **({"logprob": -0.7} if return_logprobs else {}),
                }
                time.sleep(0.002)
            yield {
                "metrics": {
                    "prompt_tokens": len(ids),
                    "cached_prompt_tokens": 1,
                    "generated_tokens": generated,
                }
            }
        finally:
            self.active -= 1


class ServingTest(unittest.TestCase):
    def test_idle_preparation_follows_response_and_yields_to_queued_request(self):
        preparing = threading.Event()
        proceed = threading.Event()
        interrupted = []

        def prepare(*, should_stop):
            if preparing.is_set():
                return
            self.assertEqual(self.engine.active, 0)
            preparing.set()
            self.assertTrue(proceed.wait(3))
            interrupted.append(should_stop())

        with patch.object(self.engine, "prepare_prefix", side_effect=prepare):
            try:
                first = self.pool.submit(self.request, "B", max_tokens=1)
                self.assertTrue(preparing.wait(3))
                self.assertEqual(first.result(timeout=1)[0], 200)
                second = self.pool.submit(self.request, "C", max_tokens=1)
                self.wait_queued()
                proceed.set()
                self.assertEqual(second.result()[0], 200)
                self.assertEqual(interrupted, [True])
                self.assertEqual(self.engine.max_active, 1)
            finally:
                proceed.set()

    def test_streaming_logprobs_are_rejected_before_generation(self):
        status, body = self.request("B", stream=True, logprobs=True)
        self.assertEqual(status, 400)
        self.assertIn("logprobs require stream=false", body)
        self.assertEqual(self.engine.calls, [])

    def test_flint_grammar_is_default_and_can_be_disabled(self):
        self.assertEqual(self.request("B", max_tokens=1)[0], 200)
        self.assertIs(self.engine.grammars[-1], self.engine.constraint)
        self.engine.constraints.tools.assert_called_once()
        self.assertEqual(self.request("B", max_tokens=1, grammar="none")[0], 200)
        self.assertIsNone(self.engine.grammars[-1])
        self.engine.constraints.tools.assert_called_once()

    def test_invalid_grammar_is_rejected(self):
        self.assertEqual(self.request("B", grammar="json")[0], 400)
        self.assertEqual(self.engine.calls, [])

    def test_unknown_fields_are_ignored_with_one_warning_each(self):
        fields = {
            "chat_template": "other",
            "chat_template_kwargs": {},
            "input_ids": [ord("C")],
            "reasoning_effort": "high",
            "top_logprobs": 3,
            "xtc_probability": 0,
            "logit_bias": "x",
            "adapters": 5,
        }
        with self.assertLogs(level="WARNING") as logs:
            self.assertEqual(self.request("B", max_tokens=1, **fields)[0], 200)
            self.assertEqual(self.request("B", max_tokens=1, **fields)[0], 200)
        self.assertEqual(self.engine.calls, [("B", 1), ("B", 1)])
        self.assertEqual(
            logs.output,
            [
                f"WARNING:root:Ignoring unsupported request field {name!r}"
                for name in sorted(fields)
            ],
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = Engine()
        self.server = make_server(self.engine, ("127.0.0.1", 0), Path(self.temp.name) / "metrics")
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.pool = ThreadPoolExecutor(max_workers=12)

    def tearDown(self):
        self.server.generator.cancel()
        self.engine.release.set()
        self.server.shutdown()
        self.server.generator.join()
        self.server.server_close()
        self.thread.join()
        self.pool.shutdown()
        self.temp.cleanup()

    def request(self, message, **options):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(
                    {
                        "model": "flint",
                        "messages": [{"role": "user", "content": message}],
                        **options,
                    }
                ),
                {"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            body = response.read().decode()
            return response.status, body if options.get("stream") else json.loads(body)
        finally:
            conn.close()

    def raw_request(self, message, **options):
        client = socket.create_connection(self.server.server_address, timeout=5)
        body = json.dumps(
            {"model": "flint", "messages": [{"role": "user", "content": message}], **options}
        ).encode()
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )
        return client

    def wait_queued(self, count=1):
        deadline = time.monotonic() + 3
        while self.server.generator.requests.qsize() < count:
            if time.monotonic() > deadline:
                self.fail("request did not enter the queue")
            time.sleep(0.01)

    def test_disconnect_preserves_queued_request(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.engine.started.clear()
                self.engine.release.clear()
                self.engine.cancelled.clear()
                active = self.raw_request("A", max_tokens=30, stream=stream)
                self.assertTrue(self.engine.started.wait(3))
                sibling = self.pool.submit(self.request, "B", max_tokens=2)
                self.wait_queued()
                if stream:
                    self.engine.release.set()
                    data = b""
                    while b"data: " not in data:
                        chunk = active.recv(4096)
                        self.assertTrue(chunk, "Connection closed before the first event")
                        data += chunk
                active.shutdown(socket.SHUT_RDWR)
                active.close()
                self.assertEqual(sibling.result()[0], 200)
                self.assertTrue(self.engine.cancelled.is_set())
                self.assertEqual(self.engine.max_active, 1)
                if stream:
                    rows = [
                        json.loads(line)
                        for line in (Path(self.temp.name) / "metrics").read_text().splitlines()
                    ]
                    self.assertGreater(rows[-2]["generated_tokens"], 0)

    def test_shutdown_wakes_pending_and_cancels_active(self):
        first = self.pool.submit(self.request, "A", max_tokens=2)
        self.assertTrue(self.engine.started.wait(3))
        second = self.pool.submit(self.request, "B", max_tokens=2)
        self.wait_queued()
        self.server.generator.cancel()
        self.server.generator.join()
        self.assertEqual(first.result()[0], 200)
        self.assertEqual(second.result()[0], 503)
        self.assertEqual(self.engine.calls, [("A", 2)])
        self.assertTrue(self.engine.cancelled.is_set())

    def test_worker_failure_wakes_pending_and_marks_unhealthy(self):
        first = self.pool.submit(self.request, "A", max_tokens=2)
        self.assertTrue(self.engine.started.wait(3))
        second = self.pool.submit(self.request, "B", max_tokens=2)
        self.wait_queued()
        self.engine.failure = RuntimeError("decoder failure")
        errors = []
        with patch.object(threading, "excepthook", side_effect=errors.append):
            self.engine.release.set()
            self.server.generator.join()
        with self.assertRaises(http.client.RemoteDisconnected):
            first.result()
        self.assertEqual(second.result()[0], 503)
        self.assertIs(errors[0].exc_value, self.engine.failure)
        self.assertIsNotNone(errors[0].exc_traceback)
        self.assertEqual(self.engine.calls, [("A", 2)])
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            self.assertEqual(response.status, 503)
            self.assertEqual(json.loads(response.read())["status"], "stopping")
        finally:
            conn.close()
        self.assertEqual(self.request("C")[0], 503)

    def test_listen_backlog_holds_a_burst_of_connections(self):
        # Connections beyond the backlog are reset (macOS) or dropped (Linux) before a handler runs.
        self.server.shutdown()
        clients = [
            socket.create_connection(self.server.server_address, timeout=5) for _ in range(16)
        ]
        for client in clients:
            client.close()

    def test_queue_capacity_rejects_excess_and_preserves_admitted_requests(self):
        first = self.pool.submit(self.request, "A", max_tokens=1)
        self.assertTrue(self.engine.started.wait(3))
        queued = [self.pool.submit(self.request, "B", max_tokens=1) for _ in range(8)]
        self.wait_queued(8)
        status, body = self.request("C", max_tokens=1)
        self.assertEqual(status, 429)
        self.assertIn("queue is full", body["error"])
        self.engine.release.set()
        self.assertEqual(first.result()[0], 200)
        self.assertEqual([request.result()[0] for request in queued], [200] * 8)
        self.assertEqual(self.engine.max_active, 1)

    def test_stream_reports_length_at_context_exhaustion(self):
        status, data = self.request("B" * 30, stream=True, stream_options={"include_usage": True})
        self.assertEqual(status, 200)
        events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: {")]
        self.assertEqual(events[-2]["choices"][0]["finish_reason"], "length")
        self.assertEqual(events[-1]["usage"]["completion_tokens"], 2)
        self.assertEqual(events[-1]["usage"]["prompt_tokens_details"]["cached_tokens"], 1)
        self.assertTrue(data.endswith("data: [DONE]\n\n"))

    def test_remaining_context_and_explicit_smaller_limit(self):
        for options, maximum in [
            ({}, 29),
            ({"max_tokens": 1000}, 29),
            ({"max_tokens": 2}, 2),
            ({"max_completion_tokens": 3, "max_tokens": 2}, 3),
        ]:
            with self.subTest(options=options):
                with patch.object(self.engine, "prompt", wraps=self.engine.prompt) as prompt:
                    status, body = self.request("BBB", **options)
                    prompt.assert_called_once()
                self.assertEqual(status, 200)
                self.assertEqual(self.engine.calls[-1], ("B", maximum))
                self.assertEqual(body["choices"][0]["finish_reason"], "length")
                self.assertEqual(body["usage"]["completion_tokens"], maximum)
                self.assertEqual(body["usage"]["prompt_tokens_details"]["cached_tokens"], 1)
        self.assertEqual(self.request("B" * 32)[0], 400)
        self.assertEqual(self.request("B", max_tokens=0)[0], 400)
        self.assertEqual(self.request("B", seed=2**64 - 10)[0], 400)
        self.assertEqual(self.request("B", max_tokens=1)[0], 200)

    def test_presence_penalty_reaches_decoder(self):
        with patch.object(
            self.engine, "generate_tokens", wraps=self.engine.generate_tokens
        ) as generate:
            self.assertEqual(self.request("B", max_tokens=1, presence_penalty=1.5)[0], 200)
            self.assertEqual(generate.call_args.args[1].presence_penalty, 1.5)

    def messages(self, body, headers=None):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        try:
            conn.request(
                "POST",
                "/v1/messages",
                json.dumps(body),
                {"Content-Type": "application/json", **(headers or {})},
            )
            response = conn.getresponse()
            raw = response.read().decode()
            return response.status, raw if body.get("stream") else json.loads(raw)
        finally:
            conn.close()

    def test_messages_translates_onto_the_chat_path(self):
        seen = {}

        def prompt(messages, tools=None):
            seen["messages"] = messages
            seen["tools"] = tools
            return [ord("B")]

        body = {
            "model": "flint",
            "max_tokens": 1,
            "system": "Be brief",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "B"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": "sunny",
                        },
                    ],
                }
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
            "service_tier": "auto",
        }
        with (
            patch.object(self.engine, "prompt", side_effect=prompt),
            self.assertLogs(level="WARNING") as logs,
        ):
            status, message = self.messages(body)
            self.assertEqual(self.messages(body)[0], 200)
        self.assertEqual(status, 200)
        self.assertEqual(seen["messages"][0], {"role": "system", "content": "Be brief"})
        self.assertEqual(seen["messages"][1]["content"], "B")
        self.assertEqual(seen["messages"][2]["role"], "tool")
        self.assertEqual(seen["tools"][0]["function"]["name"], "get_weather")
        self.assertEqual(message["type"], "message")
        self.assertEqual(message["content"], [{"type": "text", "text": "x"}])
        self.assertEqual(message["stop_reason"], "max_tokens")
        self.assertEqual(message["usage"]["output_tokens"], 1)
        self.assertEqual(message["usage"]["input_tokens"], 1)
        self.assertEqual(
            logs.output, ["WARNING:root:Ignoring unsupported request field 'service_tier'"]
        )
        self.assertEqual(self.messages({**body, "tool_choice": {"type": "any"}})[0], 400)
        self.assertEqual(self.engine.calls[-1][0], "B")

    def test_messages_stream_uses_the_event_sequence(self):
        status, raw = self.messages(
            {
                "model": "flint",
                "max_tokens": 2,
                "messages": [{"role": "user", "content": "B"}],
                "stream": True,
            }
        )
        self.assertEqual(status, 200)
        names = [
            line.removeprefix("event: ") for line in raw.splitlines() if line.startswith("event: ")
        ]
        self.assertEqual(names[0], "message_start")
        self.assertIn("content_block_delta", names)
        self.assertEqual(names[-2:], ["message_delta", "message_stop"])
        delta = json.loads(raw.split("event: message_delta\ndata: ", 1)[1].split("\n", 1)[0])
        self.assertEqual(delta["delta"]["stop_reason"], "max_tokens")
        self.assertEqual(delta["usage"]["output_tokens"], 2)

    def test_messages_translates_errors_raised_before_generation(self):
        first = self.pool.submit(self.request, "A", max_tokens=2)
        self.assertTrue(self.engine.started.wait(3))
        body = {"model": "flint", "max_tokens": 2, "messages": [{"role": "user", "content": "B"}]}
        second = self.pool.submit(self.messages, body)
        self.wait_queued()
        self.server.generator.cancel()
        self.server.generator.join()
        self.assertEqual(first.result()[0], 200)
        status, payload = second.result()
        self.assertEqual(status, 503)
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error"]["type"], "overloaded_error")

    def test_messages_accepts_bearer_token_or_api_key(self):
        server = make_server(
            self.engine, ("127.0.0.1", 0), Path(self.temp.name) / "metrics-auth", api_key="secret"
        )
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        body = json.dumps(
            {"model": "flint", "max_tokens": 1, "messages": [{"role": "user", "content": "B"}]}
        )

        def post(path, headers):
            conn = http.client.HTTPConnection(*server.server_address, timeout=5)
            try:
                conn.request("POST", path, body, {"Content-Type": "application/json", **headers})
                response = conn.getresponse()
                payload = json.loads(response.read())
                return response.status, payload
            finally:
                conn.close()

        try:
            status, payload = post("/v1/messages", {})
            self.assertEqual(status, 401)
            self.assertEqual(payload["error"]["type"], "authentication_error")
            self.assertEqual(post("/v1/messages", {"x-api-key": "secret"})[0], 200)
            # The Anthropic SDK and Claude Code add a query string.
            self.assertEqual(post("/v1/messages?beta=true", {"x-api-key": "secret"})[0], 200)
            self.assertEqual(post("/v1/messages", {"Authorization": "Bearer secret"})[0], 200)
            self.assertEqual(post("/v1/chat/completions", {"x-api-key": "secret"})[0], 401)
            self.assertEqual(
                post("/v1/chat/completions", {"Authorization": "Bearer secret"})[0], 200
            )
        finally:
            server.shutdown()
            server.generator.cancel()
            server.generator.join()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
