"""thinking_budget over HTTP: request validation, early 400s, /v1/messages and usage fields."""

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from test_serving import Engine as BaseEngine

from rec_mlx.server import make_server

THINK_ID = 7
BUDGET = 3
OK, BAD, ERROR = 200, 400, 500
JSON_HEADERS = {"Content-Type": "application/json"}
# Usage keys before the thinking budget existed (origin/main); mlx-lm omits
# prompt_tokens_details when no metrics event arrives.
BASE_USAGE = {"prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details"}
THINKING_USAGE = {"reasoning_tokens", "thinking_forced_close"}


class ThinkEngine(BaseEngine):
    """BaseEngine that records extra generate_tokens kwargs and reports thinking metrics."""

    def __init__(self):
        super().__init__()
        self.extra = []
        self.reasoning, self.forced = 7, True
        self.drop_metrics = False
        self.drop_key = None
        self.think_ids = (THINK_ID, 8)
        self.prompt_ids = None
        self.validated = []

    def prompt(self, messages, tools=None):
        return self.prompt_ids or super().prompt(messages, tools)

    def validate_thinking_budget(self, ids, budget):
        self.validated.append((list(ids), budget))
        if self.think_ids is None:
            raise ValueError("thinking_budget requires <think> and </think> tokenizer tokens")
        if ids and ids[-1] == self.think_ids[0]:
            raise ValueError("thinking_budget requires a prompt that does not already open <think>")

    def generate_tokens(self, ids, sampling, *, should_stop, grammar=None, **extra):
        self.calls.append((chr(ids[0]), sampling.max_tokens))
        self.grammars.append(grammar)
        self.extra.append(extra)
        for index in range(sampling.max_tokens):
            yield {
                "token": ord("x"),
                "text": "x",
                "finish_reason": "length" if index + 1 == sampling.max_tokens else None,
            }
        if not self.drop_metrics:
            metrics = {
                "prompt_tokens": len(ids),
                "cached_prompt_tokens": 1,
                "generated_tokens": sampling.max_tokens,
                "computed_tokens": sampling.max_tokens,
                "reasoning_tokens": self.reasoning,
                "thinking_forced_close": self.forced,
            }
            metrics.pop(self.drop_key, None)
            yield {"metrics": metrics}


class ServerCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = ThinkEngine()
        self.server = make_server(self.engine, ("127.0.0.1", 0), Path(self.temp.name) / "metrics")
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        self.thread.start()

    def tearDown(self):
        self.server.generator.cancel()
        self.server.shutdown()
        self.server.generator.join()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def post(self, path, body):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        try:
            conn.request("POST", path, json.dumps(body), JSON_HEADERS)
            response = conn.getresponse()
            return response.status, response.read().decode()
        finally:
            conn.close()

    def chat(self, **options):
        body = {"model": "flint", "messages": [{"role": "user", "content": "BB"}], **options}
        return self.post("/v1/chat/completions", body)

    def usage(self, **options):
        """The usage object of a non-stream response, or of the final chunk of a stream."""
        status, text = self.chat(max_tokens=2, **options)
        self.assertEqual(status, OK, text)
        return json.loads(text)["usage"]

    def stream_usage(self, **options):
        options["stream_options"] = {"include_usage": True}
        status, text = self.chat(max_tokens=2, stream=True, **options)
        self.assertEqual(status, OK, text)
        chunks = [json.loads(x[6:]) for x in text.splitlines() if x.startswith("data: {")]
        self.assertTrue(all("usage" not in chunk for chunk in chunks[:-1]))
        return chunks[-1]["usage"]


class RequestTest(ServerCase):
    def test_invalid_budget_values_are_400_and_never_reach_the_engine(self):
        for value in (-1, 1.5, "5", True, False, None, [], {}):
            with self.subTest(value=value):
                status, text = self.chat(max_tokens=1, thinking_budget=value)
                self.assertEqual(status, BAD, text)
        self.assertEqual(self.engine.calls, [])

    def test_valid_budgets_reach_the_engine_and_are_not_dropped(self):
        with self.assertNoLogs(level="WARNING"):
            for value in (0, 1, 4096):
                self.assertEqual(self.chat(max_tokens=1, thinking_budget=value)[0], OK)
                self.assertEqual(self.engine.extra[-1]["thinking_budget"], value)

    def test_budget_needs_the_flint_grammar(self):
        status, text = self.chat(max_tokens=1, grammar="none", thinking_budget=2)
        self.assertEqual(status, BAD, text)
        self.assertEqual(self.engine.calls, [])

    def test_unusable_prompt_or_tokenizer_is_400_before_any_header(self):
        self.engine.prompt_ids = [ord("B"), THINK_ID]  # the prompt already opens <think>
        for stream in (False, True):
            status, text = self.chat(max_tokens=2, thinking_budget=BUDGET, stream=stream)
            self.assertEqual((status, "think" in text), (BAD, True), text)
        self.engine.prompt_ids, self.engine.think_ids = None, None
        for stream in (False, True):
            self.assertEqual(self.chat(max_tokens=2, thinking_budget=BUDGET, stream=stream)[0], BAD)
        self.assertEqual(self.engine.calls, [], "generation must not start")

    def test_validation_gets_the_prompt_and_only_runs_with_a_budget(self):
        self.chat(max_tokens=1, thinking_budget=BUDGET)
        self.assertEqual(self.engine.validated, [([ord("B"), ord("B")], BUDGET)])
        self.engine.think_ids = None
        status, text = self.chat(max_tokens=1)
        self.assertEqual(status, OK, text)
        self.assertEqual(len(self.engine.validated), 1)
        self.assertNotIn("thinking_budget", self.engine.extra[-1])


class MessagesTest(ServerCase):
    def messages(self, **extra):
        body = {"model": "flint", "max_tokens": 4, "messages": [{"role": "user", "content": "BB"}]}
        return self.post("/v1/messages", {**body, **extra})

    def test_top_level_thinking_budget_is_rejected_not_dropped(self):
        status, text = self.messages(thinking_budget=BUDGET)
        self.assertEqual(status, BAD, text)
        self.assertIn("thinking_budget", text)
        self.assertEqual(self.engine.calls, [])

    def test_anthropic_budget_tokens_is_accepted_and_not_forwarded(self):
        status, text = self.messages(thinking={"type": "enabled", "budget_tokens": 1024})
        self.assertEqual(status, OK, text)
        self.assertEqual(self.engine.extra, [{}])


class UsageTest(ServerCase):
    def test_usage_reports_reasoning_only_when_a_budget_is_set(self):
        cases = [  # (options, engine reasoning, engine forced)
            ({"thinking_budget": 64}, 7, True),
            ({"thinking_budget": 64}, 3, False),
            ({"thinking_budget": 0}, 0, False),
        ]
        for options, reasoning, forced in cases:
            self.engine.reasoning, self.engine.forced = reasoning, forced
            for read in (self.usage, self.stream_usage):
                with self.subTest(read=read.__name__, reasoning=reasoning, forced=forced):
                    usage = read(**options)
                    self.assertEqual(set(usage), BASE_USAGE | THINKING_USAGE)
                    self.assertEqual(usage["reasoning_tokens"], reasoning)
                    self.assertIs(usage["thinking_forced_close"], forced)
                    self.assertEqual(usage["completion_tokens"], 2)
                    self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 1)
        for read in (self.usage, self.stream_usage):
            self.assertEqual(set(read()), BASE_USAGE)

    def test_missing_metrics_fail_cleanly_only_with_a_budget(self):
        self.engine.drop_metrics = True
        status, text = self.chat(max_tokens=2, thinking_budget=1)
        self.assertEqual(status, ERROR, text)
        self.assertIn("metrics", text.lower())
        self.assertNotIn("NoneType", text)
        status, text = self.chat(
            max_tokens=2, stream=True, thinking_budget=1, stream_options={"include_usage": True}
        )
        self.assertIn('"error"', text)  # the header is sent; the stream carries the failure
        self.assertIn("metrics", text.lower())
        self.assertEqual(set(self.usage()), BASE_USAGE - {"prompt_tokens_details"})
        self.engine.drop_metrics = False  # the server keeps serving after the error
        self.assertEqual(set(self.usage(thinking_budget=1)), BASE_USAGE | THINKING_USAGE)

    def test_a_metrics_event_missing_a_thinking_key_is_a_500_naming_the_key(self):
        for key in sorted(THINKING_USAGE):
            self.engine.drop_key = key
            with self.subTest(key=key):
                status, text = self.chat(max_tokens=2, thinking_budget=1)
                self.assertEqual(status, ERROR, text)
                self.assertIn("usage is unavailable", text)
                self.assertIn(key, text)
            self.assertEqual(set(self.usage()), BASE_USAGE)  # without a budget it still serves


if __name__ == "__main__":
    unittest.main()
