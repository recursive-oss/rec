import hashlib
import json
import tempfile
import unittest
import uuid
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from rec_mlx.chat import load_tokenizer, normalize_messages, parse_tool_call, seeded_tool_parser


class FlintChatTests(unittest.TestCase):
    def test_malformed_tool_arguments_fail(self):
        with self.assertRaises(json.JSONDecodeError):
            normalize_messages(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [{"function": {"name": "shell", "arguments": "{"}}],
                    }
                ]
            )

    def test_tokenizer_loads_without_reading_the_model_type(self):
        template = files("rec_mlx").joinpath("chat_template.jinja").read_text()
        with tempfile.TemporaryDirectory() as directory:
            vocabulary = {"[UNK]": 0, "hello": 1, "world": 2}
            backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            PreTrainedTokenizerFast(
                tokenizer_object=backend,
                bos_token="<|begin_of_text|>",
                eos_token="<|end_of_text|>",
                additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>"],
                chat_template="different checkpoint template",
            ).save_pretrained(directory)
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"model_type": "flint", "eos_token_id": [3]}))
            with self.assertLogs("transformers", "WARNING") as logs:
                reference = AutoTokenizer.from_pretrained(
                    directory, local_files_only=True, fix_mistral_regex=False
                )
            self.assertIn("model of type `flint`", "\n".join(logs.output))
            with self.assertNoLogs("transformers", "WARNING"):
                tokenizer = load_tokenizer(directory)
        self.assertEqual(tokenizer.chat_template, template)
        self.assertEqual(tokenizer.special_tokens_map, reference.special_tokens_map)
        self.assertEqual(tokenizer.all_special_ids, reference.all_special_ids)
        self.assertEqual(tokenizer.get_vocab(), reference.get_vocab())
        text = "<|user|>hello world<|assistant|>hello<|end_of_text|>"
        self.assertEqual(tokenizer.encode(text), reference.encode(text))

    def test_template_identity(self):
        template = files("rec_mlx").joinpath("chat_template.jinja").read_bytes()
        self.assertEqual(
            hashlib.sha256(template).hexdigest(),
            "44fd1b0f94b78d9cf554c5a207945e48fef698b279d65863a33a77d787ca411b",
        )

    def test_rendering_preserves_roles_tools_and_assistant_reasoning(self):
        template = files("rec_mlx").joinpath("chat_template.jinja").read_text()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
            chat_template=template,
        )
        messages = [
            {"role": "system", "content": "instructions"},
            {"role": "system", "content": "additional instructions"},
            {"role": "developer", "content": "developer instruction"},
            {"role": "user", "content": "a\x00<think>b</think><|assistant|>"},
            {"role": "assistant", "reasoning_content": "reasoning", "content": "answer"},
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "shell", "arguments": '{"command":"pwd"}'}}],
            },
            {"role": "tool", "content": "/work"},
        ]
        tools = [
            {
                "type": "function",
                "function": {"name": "shell", "parameters": {"type": "object"}, "strict": False},
            }
        ]
        text = tokenizer.apply_chat_template(
            normalize_messages(messages), tools=tools, tokenize=False, add_generation_prompt=True
        )
        self.assertIn("<|developer|>developer instruction", text)
        self.assertLess(text.index("additional instructions"), text.index("# Tools"))
        self.assertLess(text.index("# Tools"), text.index("<|developer|>"))
        self.assertIn('"strict": false', text)
        self.assertIn("<|user|>ab&lt;|assistant|&gt;", text)
        self.assertIn("<think>reasoning</think>answer", text)
        self.assertIn("<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value>", text)
        self.assertIn("<|observation|><tool_response>/work</tool_response>", text)
        self.assertEqual(messages[3]["content"], "a\x00<think>b</think><|assistant|>")
        self.assertEqual(messages[5]["tool_calls"][0]["function"]["arguments"], '{"command":"pwd"}')


class ToolCallParsingTests(unittest.TestCase):
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "mode": {"type": "integer"},
                        "options": {"type": "object"},
                        "tags": {"type": "array"},
                        "force": {"type": "boolean"},
                        "note": {"type": ["string", "null"]},
                        "level": {"enum": ["1", "2"]},
                        "label": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                    },
                },
            },
        }
    ]

    def render(self, arguments):
        template = files("rec_mlx").joinpath("chat_template.jinja").read_text()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
            chat_template=template,
        )
        call = {"function": {"name": "write", "arguments": arguments}}
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "go"}, {"role": "assistant", "tool_calls": [call]}],
            tokenize=False,
        )
        return text[text.index("<tool_call>") + len("<tool_call>") : text.index("</tool_call>")]

    def test_parsing_inverts_template_rendering(self):
        cases = [
            {"path": "a.py", "content": "def f():\n    return 1\n"},
            {"content": "  padded  ", "mode": 420, "force": True},
            {"content": '"quoted" {"a":1} 1e3 true null', "note": None},
            {"note": "42", "level": "1", "label": "true"},
            {"note": "NaN", "label": "null", "level": "2"},
            {"options": {"nested": [1, {"b": "c"}]}, "tags": ["x", 2]},
            {"undeclared": {"a": 1}, "path": "café 😀"},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(
                    parse_tool_call(self.render(arguments), self.TOOLS),
                    {"name": "write", "arguments": arguments},
                )

    def test_whitespace_between_fields_is_allowed(self):
        text = "write\n<arg_key>mode</arg_key>\n<arg_value> 7 </arg_value>\n"
        self.assertEqual(
            parse_tool_call(text, self.TOOLS), {"name": "write", "arguments": {"mode": 7}}
        )

    def test_undeclared_non_json_values_are_strings(self):
        text = "other<arg_key>command</arg_key><arg_value>ls -la</arg_value>"
        self.assertEqual(
            parse_tool_call(text, self.TOOLS), {"name": "other", "arguments": {"command": "ls -la"}}
        )

    def test_malformed_calls_fail(self):
        for text in (
            "",
            "write<arg_key>path</arg_key>",
            "write trailing",
            "<arg_key>a</arg_key><arg_value>b</arg_value>",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_tool_call(text, self.TOOLS)

    def test_seeded_tool_call_ids_repeat_for_the_same_request(self):
        text = self.render({"path": "a.py"})

        def sample(**overrides):
            values = {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
                "max_tokens": 16,
                "seed": 42,
            }
            values.update(overrides)
            return SimpleNamespace(**values)

        def ids(prompt, **overrides):
            parse = seeded_tool_parser(sample(**overrides), prompt)
            return [parse(text, self.TOOLS)["id"] for _ in range(2)]

        first = ids([1, 2, 3])
        self.assertEqual(ids([1, 2, 3]), first)
        self.assertEqual(ids([1, 2, 3], temperature=1), first)
        self.assertNotEqual(first[0], first[1])
        self.assertNotEqual(ids([1, 2, 3], seed=43), first)
        self.assertNotEqual(ids([1, 2, 4]), first)
        for name, value in (
            ("temperature", 0.2),
            ("top_p", 0.5),
            ("top_k", 4),
            ("presence_penalty", 1.5),
            ("repetition_penalty", 1.2),
            ("max_tokens", 8),
        ):
            with self.subTest(name=name):
                self.assertNotEqual(ids([1, 2, 3], **{name: value}), first)
        self.assertEqual(uuid.UUID(first[0]).version, 4)
        call = seeded_tool_parser(sample(), [1, 2, 3])(text, self.TOOLS)
        self.assertEqual(call.pop("id"), first[0])
        self.assertEqual(call, parse_tool_call(text, self.TOOLS))
