import io
import json
import unittest

from rec_mlx.anthropic import (
    ErrorTranslator,
    MessageEventStream,
    chat_to_message,
    messages_to_chat,
)


def chat(**overrides):
    body = {
        "model": "flint",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    body.update(overrides)
    converted, ignored, thinking = messages_to_chat(body)
    return converted, ignored, thinking


class MessagesRequestTests(unittest.TestCase):
    def test_maps_system_sampling_and_stop_sequences(self):
        converted, ignored, thinking = chat(
            system=[{"type": "text", "text": "Be brief", "cache_control": {"type": "ephemeral"}}],
            temperature=0.2,
            top_p=0.9,
            top_k=8,
            stop_sequences=["END"],
            metadata={"user_id": "a"},
            thinking={"type": "enabled", "budget_tokens": 100},
        )
        self.assertEqual(ignored, set())
        self.assertTrue(thinking)
        self.assertEqual(converted["messages"][0], {"role": "system", "content": "Be brief"})
        self.assertEqual(converted["temperature"], 0.2)
        self.assertEqual(converted["top_p"], 0.9)
        self.assertEqual(converted["top_k"], 8)
        self.assertEqual(converted["stop"], ["END"])
        self.assertNotIn("stream", converted)

    def test_maps_tools_tool_use_and_tool_results(self):
        converted, ignored, _ = chat(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Look it up", "signature": "sig"},
                        {"type": "text", "text": "Checking"},
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call-1", "content": "sunny"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-2",
                            "content": [{"type": "text", "text": "12"}],
                            "is_error": True,
                        },
                        {"type": "text", "text": "Thanks"},
                    ],
                },
            ],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
            tool_choice={"type": "auto"},
        )
        self.assertEqual(ignored, set())
        self.assertEqual(
            converted["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        )
        self.assertEqual(converted["tool_choice"], "auto")
        assistant, result, failed, thanks = converted["messages"]
        self.assertEqual(assistant["reasoning_content"], "Look it up")
        self.assertEqual(assistant["content"], "Checking")
        self.assertEqual(assistant["tool_calls"][0]["id"], "call-1")
        self.assertEqual(
            json.loads(assistant["tool_calls"][0]["function"]["arguments"]), {"city": "Paris"}
        )
        self.assertEqual(result, {"role": "tool", "tool_call_id": "call-1", "content": "sunny"})
        self.assertEqual(failed["content"], "Error: 12")
        self.assertEqual(thanks, {"role": "user", "content": "Thanks"})

    def test_streaming_requests_usage_and_can_hide_thinking(self):
        converted, _, thinking = chat(stream=True, thinking={"type": "disabled"})
        self.assertFalse(thinking)
        self.assertEqual(converted["stream_options"], {"include_usage": True})

    def test_unknown_fields_are_reported_and_structural_errors_fail(self):
        _, ignored, _ = chat(service_tier="auto")
        self.assertEqual(ignored, {"service_tier"})
        for body, text in (
            ({"model": "flint", "messages": [{"role": "user", "content": "Hi"}]}, "max_tokens"),
            ({"model": "flint", "max_tokens": 4, "messages": []}, "nonempty"),
            (
                {
                    "model": "flint",
                    "max_tokens": 4,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "tool_choice": {"type": "tool", "name": "get_weather"},
                },
                "forced calls",
            ),
            (
                {
                    "model": "flint",
                    "max_tokens": 4,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
                },
                "disable_parallel_tool_use=true is not supported",
            ),
            (
                {
                    "model": "flint",
                    "max_tokens": 4,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "thinking": {"type": "sometimes"},
                },
                "adaptive",
            ),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, text):
                messages_to_chat(body)

    def test_accepts_current_claude_client_fields(self):
        converted, ignored, thinking = chat(
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "system", "content": [{"type": "text", "text": "Now be brief"}]},
            ],
            thinking={"type": "adaptive"},
            tool_choice={"type": "auto", "disable_parallel_tool_use": False},
            disable_parallel_tool_calls=True,
        )
        self.assertTrue(thinking)
        self.assertEqual(converted["messages"][1], {"role": "system", "content": "Now be brief"})
        self.assertFalse(chat(thinking={"type": "adaptive", "display": "omitted"})[2])
        # Not an Anthropic field, so it is ignored like any other unknown field.
        self.assertEqual(ignored, {"disable_parallel_tool_calls"})

    def test_text_blocks_keep_a_separator(self):
        converted, _, _ = chat(
            system=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "c"}, {"type": "text", "text": "d"}],
                }
            ],
        )
        self.assertEqual(converted["messages"][0]["content"], "a\n\nb")
        self.assertEqual(converted["messages"][1]["content"], "c\n\nd")

    def test_images_and_documents_become_notes_so_sessions_continue(self):
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/png"}}
        with self.assertLogs(level="WARNING"):
            converted, _, _ = chat(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is this?"},
                            image,
                            {
                                "type": "document",
                                "source": {
                                    "type": "text",
                                    "media_type": "text/plain",
                                    "data": "notes",
                                },
                            },
                            {"type": "document", "source": {"type": "base64", "data": "JVBE"}},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "call-1", "name": "screenshot", "input": {}}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call-1",
                                "content": [{"type": "text", "text": "Captured"}, image],
                            }
                        ],
                    },
                ]
            )
        user, _, result = converted["messages"]
        self.assertEqual(
            user["content"],
            "What is this?\n\n[image omitted: this model reads text only]\n\nnotes"
            "\n\n[document omitted: this model reads text only]",
        )
        self.assertEqual(
            result["content"], "Captured\n\n[image omitted: this model reads text only]"
        )


class MessagesResponseTests(unittest.TestCase):
    def test_maps_text_thinking_tool_use_and_stop_reasons(self):
        message = chat_to_message(
            {
                "id": "chatcmpl-abc",
                "model": "flint",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": "Done",
                            "reasoning": "Because",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5},
            }
        )
        self.assertEqual(message["id"], "msg_abc")
        self.assertEqual(message["stop_reason"], "tool_use")
        self.assertEqual(
            message["usage"],
            {
                "input_tokens": 3,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )
        self.assertEqual(
            message["content"][0], {"type": "thinking", "thinking": "Because", "signature": ""}
        )
        self.assertEqual(message["content"][1], {"type": "text", "text": "Done"})
        self.assertEqual(message["content"][2]["input"], {"city": "Paris"})
        hidden = chat_to_message(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Stop here", "reasoning": "hidden"},
                    }
                ]
            },
            stop_sequences=["here"],
            include_thinking=False,
        )
        self.assertEqual(hidden["content"], [{"type": "text", "text": "Stop here"}])
        self.assertEqual(hidden["stop_reason"], "stop_sequence")
        self.assertEqual(hidden["stop_sequence"], "here")

    def test_cache_usage_preserves_reuse_without_double_counting(self):
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 80},
        }
        message = chat_to_message({"choices": [{"message": {"content": "OK"}}], "usage": usage})
        expected = {
            "input_tokens": 20,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 0,
            "output_tokens": 6,
        }
        self.assertEqual(message["usage"], expected)
        raw = io.BytesIO()
        stream = MessageEventStream(raw, input_tokens=100)
        for chunk in [
            {"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": usage},
        ]:
            stream.write(f"data: {json.dumps(chunk)}\n\n".encode())
        stream.write(b"data: [DONE]\n\n")
        events = [
            json.loads(line[6:])
            for line in raw.getvalue().decode().splitlines()
            if line.startswith("data: ")
        ]
        final = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(final["usage"], expected)
        self.assertEqual(
            final["usage"]["input_tokens"] + final["usage"]["cache_read_input_tokens"], 100
        )

    def test_stream_emits_the_messages_event_sequence(self):
        raw = io.BytesIO()
        stream = MessageEventStream(raw, input_tokens=4, stop_sequences=["END"])
        chunks = [
            {
                "id": "chatcmpl-abc",
                "model": "flint",
                "choices": [{"delta": {"reasoning": "Hmm"}, "finish_reason": None}],
            },
            {
                "choices": [{"delta": {"content": "Hello END"}, "finish_reason": None}],
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
            {"choices": [], "usage": {"completion_tokens": 6}},
        ]
        stream.write(b": keepalive 1/2\n\n")
        for chunk in chunks:
            stream.write(f"data: {json.dumps(chunk)}\n\n".encode())
        stream.write(b"data: [DONE]\n\n")
        events = []
        for block in raw.getvalue().split(b"\n\n"):
            if not block or block.startswith(b":"):
                continue
            name, data = block.decode().split("\n", 1)
            events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
        names = [name for name, _ in events]
        self.assertEqual(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        )
        self.assertEqual(events[0][1]["message"]["usage"]["input_tokens"], 4)
        self.assertEqual(events[2][1]["delta"], {"type": "thinking_delta", "thinking": "Hmm"})
        self.assertEqual(events[5][1]["delta"]["text"], "Hello END")
        self.assertEqual(events[7][1]["content_block"]["name"], "get_weather")
        self.assertEqual(events[8][1]["delta"]["partial_json"], '{"city":"Paris"}')
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "tool_use")
        self.assertEqual(events[-2][1]["usage"]["output_tokens"], 6)
        self.assertTrue(raw.getvalue().startswith(b": keepalive 1/2\n\n"))

    def test_http_headers_are_written_before_events(self):
        raw = io.BytesIO()
        stream = MessageEventStream(raw, input_tokens=1)
        stream.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
        stream.write(
            b'data: {"id":"chatcmpl-a","model":"flint","choices":'
            b'[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
        )
        stream.write(b"data: [DONE]\n\n")
        body = raw.getvalue()
        self.assertLess(body.index(b"HTTP/1.1 200 OK"), body.index(b"event: message_start"))
        self.assertIn(b"event: content_block_delta", body)

    def test_errors_take_their_type_from_the_status(self):
        for status, kind in (
            (503, "overloaded_error"),
            (404, "not_found_error"),
            (500, "api_error"),
        ):
            with self.subTest(status=status):
                raw = io.BytesIO()
                ErrorTranslator(raw, lambda: status).write(b'{"error": "No worker"}')
                self.assertEqual(
                    json.loads(raw.getvalue()),
                    {"type": "error", "error": {"type": kind, "message": "No worker"}},
                )
        raw = io.BytesIO()
        already = b'{"type": "error", "error": {"type": "invalid_request_error", "message": "x"}}'
        ErrorTranslator(raw, lambda: 400).write(already)
        self.assertEqual(raw.getvalue(), already)
        raw = io.BytesIO()
        MessageEventStream(raw, input_tokens=1, status=lambda: 503).write(b'{"error": "Stopping"}')
        self.assertEqual(json.loads(raw.getvalue())["error"]["type"], "overloaded_error")


if __name__ == "__main__":
    unittest.main()
