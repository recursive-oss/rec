import json
import threading
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace

from rec_mlx.chat import seeded_tool_parser
from rec_mlx.engine import Sampling
from rec_rocm.server import make_server, parsed_output


class Protocol(unittest.TestCase):
    def setUp(self):
        class Engine:
            context_limit = 1024
            tokenizer = SimpleNamespace(decode=lambda ids: "x")

            def prompt(self, messages, tools=None):
                return list(range(8))

            def generate_tokens(self, ids, sampling, **kwargs):
                for i, text in enumerate(["<think>", "reason", "</think>", "answer"]):
                    yield {"text": text, "token": i, "finish_reason": "stop" if i == 3 else None}
                yield {"metrics": {"generated_tokens": 4, "prompt_tokens": 8}}

        self.server = make_server(
            Engine(), ("127.0.0.1", 0), api_key="test-only", grammar_mode="none"
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path, body=None, auth=True):
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = "Bearer test-only"
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers,
        )
        return urllib.request.urlopen(req, timeout=5)

    def body(self, **kw):
        return {"model": "flint", "messages": [{"role": "user", "content": "Hello"}], **kw}

    def test_auth_and_models(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.request("/v1/models", auth=False)
        self.assertEqual(ctx.exception.code, 401)
        with self.request("/v1/models") as r:
            self.assertEqual(json.load(r)["data"][0]["id"], "flint")

    def test_chat_and_stream(self):
        with self.request("/v1/chat/completions", self.body()) as r:
            body = json.load(r)
        self.assertEqual(
            body["choices"][0]["message"],
            {"role": "assistant", "content": "answer", "reasoning_content": "reason"},
        )
        self.assertEqual(body["usage"]["completion_tokens"], 4)
        with self.request(
            "/v1/chat/completions", self.body(stream=True, stream_options={"include_usage": True})
        ) as r:
            text = r.read().decode()
        self.assertIn("[DONE]", text)
        self.assertIn('"reasoning": "reason"', text)

    def test_anthropic_stream_and_body(self):
        payload = {
            "model": "flint",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 20,
        }
        with self.request("/v1/messages", payload) as r:
            body = json.load(r)
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["usage"]["output_tokens"], 4)
        with self.request("/v1/messages?beta=true", {**payload, "stream": True}) as r:
            text = r.read().decode()
        self.assertIn("event: message_stop", text)
        self.assertIn("thinking_delta", text)

    def test_invalid_sampling_and_model(self):
        for change in (
            {"model": "other"},
            {"temperature": -1},
            {"max_tokens": True},
            {"tool_choice": "required"},
            {"stream_options": {}},
        ):
            with self.subTest(change=change):
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    self.request("/v1/chat/completions", self.body(**change))
                self.assertEqual(ctx.exception.code, 400)

    def test_tool_parser_is_deterministic(self):
        raw = "<think>reason</think><tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]

        def parse():
            return parsed_output(raw, tools, seeded_tool_parser(Sampling(), [1, 2, 3]))

        self.assertEqual(parse(), parse())
        self.assertEqual(
            json.loads(parse()["tool_calls"][0]["function"]["arguments"]), {"city": "Paris"}
        )


if __name__ == "__main__":
    unittest.main()
