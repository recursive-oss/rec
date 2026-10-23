import unittest
from copy import deepcopy
from functools import partial

import mlx.core as mx
import numpy as np
import xgrammar as xgr
from mlx_lm.sample_utils import make_sampler
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from rec_mlx import _grammar_native as native
from rec_mlx.chat import parse_tool_call
from rec_mlx.grammar import Constraint, ConstraintCompiler
from rec_mlx.grammar_builder import _PROTOCOL_MARKERS, build_flint_glm_grammar

VOCAB = 150144
WORDS = VOCAB // 32


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def packet(logits, mask, handle, token, count, temperature, penalty, presence, eos):
    words = mx.arange(VOCAB, dtype=mx.uint32)
    sampler = make_sampler(temp=temperature, top_p=0.95, top_k=20)
    tokens = []
    keys = []
    actives = []
    masks = []
    active = mx.array(True)
    for index in range(count):
        mask = native.handoff(mask, logits[index])
        masks.append(mask[:WORDS])
        allowed = ((mask[words >> 5] >> (words & 31)) & 1) != 0
        active = active & (mask[-1] != 0)
        flat = mx.where(allowed, logits[index] - penalty * presence, -float("inf"))
        flat = mx.where(active, flat, mx.zeros_like(flat))
        selected = sampler(flat - mx.logsumexp(flat)).astype(mx.int32)
        token = mx.where(active, selected, token)
        presence = presence.at[token].maximum(active.astype(mx.float32))
        mask = native.mask(token, active, handle, WORDS)
        tokens.append(token)
        keys.append(mx.random.state[0])
        actives.append(active)
        active = active & ~mx.any(token == eos)
    return mx.stack(tokens), mask, mx.stack(masks), presence, mx.stack(keys), mx.stack(actives)


class GrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        tokenizer = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        cls.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, eos_token="<|end_of_text|>"
        )
        cls.tokenizer.add_tokens([x for x in _PROTOCOL_MARKERS if x != "<|end_of_text|>"])
        cls.compiler = ConstraintCompiler(cls.tokenizer)

    def ebnf(self, source):
        compiled = self.compiler.compiler.compile_grammar(source)
        return Constraint(compiled, (self.tokenizer.eos_token_id,))

    def check_fixture(self, constraint, text, temperature=1, penalty=0):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids or ids[-1] not in constraint.stops:
            ids.append(self.tokenizer.eos_token_id)
        reference = xgr.GrammarMatcher(constraint.compiled)
        bitmask = xgr.allocate_token_bitmask(1, VOCAB)
        masks = []
        invalid = []
        for token in ids:
            reference.fill_next_token_bitmask(bitmask)
            bits = bitmask.numpy().view(np.uint32).reshape(-1).copy()
            allowed = np.unpackbits(bits.view(np.uint8), bitorder="little").astype(bool)
            self.assertTrue(allowed[token])
            masks.append(bits)
            invalid.append(int(np.flatnonzero(~allowed)[0]))
            self.assertTrue(reference.accept_token(token))
        presence = mx.zeros(VOCAB)
        token = mx.array(0, mx.int32)
        actual = []
        mx.random.seed(42)
        with constraint.session() as session:
            mask = session.initial_mask
            for begin in range(0, len(ids), 8):
                count = min(8, len(ids) - begin)
                logits = mx.full((count, VOCAB), -20.0)
                for i in range(count):
                    logits[i, ids[begin + i]] = 30.0
                    logits[i, invalid[begin + i]] = 50.0
                values = packet(
                    logits,
                    mask,
                    session.handle,
                    token,
                    count,
                    temperature,
                    penalty,
                    presence,
                    self.tokenizer.eos_token_id,
                )
                mx.eval(*values)
                session.drain()
                tokens, mask, actual_masks, presence, keys, active = values
                self.assertTrue(
                    np.array_equal(np.array(actual_masks), np.stack(masks[begin : begin + count]))
                )
                actual.extend(tokens.tolist())
                token = tokens[-1]
            self.assertEqual(actual, ids)
            self.assertEqual(session.accepted, len(ids))
        self.check_step_fixture(constraint, ids)

    def check_step_fixture(self, constraint, ids):
        reference = xgr.GrammarMatcher(constraint.compiled)
        bitmask = xgr.allocate_token_bitmask(1, VOCAB)
        state = {}
        with constraint.session() as session:
            initial = np.array(session.first_logits(mx.zeros(VOCAB)))
            for index, token in enumerate(ids):
                reference.fill_next_token_bitmask(bitmask)
                bits = bitmask.numpy().view(np.uint32).reshape(-1)
                allowed = np.unpackbits(bits.view(np.uint8), bitorder="little").astype(bool)
                self.assertTrue(allowed[token])
                invalid = int(np.flatnonzero(~allowed)[0])
                logits = mx.full(VOCAB, -20.0)
                logits[token] = 30.0
                logits[invalid] = 50.0
                masked = session.step_logits(logits, state)
                np.testing.assert_array_equal(
                    np.array(masked), np.where(allowed, np.array(logits), -np.inf)
                )
                self.assertEqual(int(mx.argmax(masked).item()), token)
                if index == 0:
                    self.assertNotIn("grammar_mask", state)
                self.assertTrue(reference.accept_token(token))
                session.accept_first(token, state)
                self.assertEqual(session.accepted, index + 1)
            self.assertTrue(session.stopped)
            np.testing.assert_array_equal(np.array(session.first_logits(mx.zeros(VOCAB))), initial)
            # A stopped session is unconstrained and ignores further accepts.
            logits = mx.arange(VOCAB, dtype=mx.float32)
            np.testing.assert_array_equal(
                np.array(session.step_logits(logits, state)), np.array(logits)
            )
            fills = len(session.mask_seconds)
            session.accept_first(-1, state)
            self.assertEqual(session.accepted, len(ids))
            self.assertEqual(len(session.mask_seconds), fills)
            np.testing.assert_array_equal(
                np.array(session.step_logits(logits, state)), np.array(logits)
            )

    def test_step_logits_tracks_each_token_and_stop(self):
        ids = self.tokenizer.encode("abc", add_special_tokens=False)
        ids.append(self.tokenizer.eos_token_id)
        self.check_step_fixture(self.ebnf('root ::= "abc"\n'), ids)

    def test_step_logits_rejects_illegal_token_without_advancing(self):
        constraint = self.ebnf('root ::= "ab"\n')
        a, b = self.tokenizer.encode("ab", add_special_tokens=False)
        state = {}
        reference = xgr.GrammarMatcher(constraint.compiled)
        bitmask = xgr.allocate_token_bitmask(1, VOCAB)
        with constraint.session() as session:
            for invalid, valid in ((b, a), (a, b), (a, self.tokenizer.eos_token_id)):
                mask = state.get("grammar_mask")
                accepted = session.accepted
                fills = len(session.mask_seconds)
                self.assertFalse(reference.accept_token(invalid))
                with self.assertRaisesRegex(
                    RuntimeError, f"Grammar rejected sampled token {invalid}"
                ):
                    session.accept_first(invalid, state)
                self.assertIs(state.get("grammar_mask"), mask)
                self.assertEqual(session.accepted, accepted)
                self.assertEqual(len(session.mask_seconds), fills)
                self.assertFalse(session.stopped)
                reference.fill_next_token_bitmask(bitmask)
                allowed = (
                    np.unpackbits(bitmask.numpy().view(np.uint8), bitorder="little")
                    .reshape(-1)
                    .astype(bool)
                )
                np.testing.assert_array_equal(
                    np.isfinite(np.array(session.step_logits(mx.zeros(VOCAB), state))), allowed
                )
                self.assertTrue(reference.accept_token(valid))
                session.accept_first(valid, state)
            self.assertTrue(session.stopped)
            self.assertEqual(session.accepted, 3)

    def test_recursive_json_with_split_utf8_and_escapes(self):
        import json

        value = {"text": "café 😀", "escaped": 'quote" and backslash\\'}
        for depth in range(24):
            value = {"depth": depth, "children": [value]}
        self.check_fixture(
            Constraint(
                self.compiler.compiler.compile_json_schema(
                    {"type": "object", "additionalProperties": True}, strict_mode=False
                ),
                (self.tokenizer.eos_token_id,),
            ),
            json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        )

    def test_ambiguous_left_recursive_grammar(self):
        constraint = self.ebnf('root ::= root "+" root | "(" root ")" | [0-9]+\n')
        self.check_fixture(constraint, "(1+(2+3))+4", temperature=0.6, penalty=1.5)

    def test_tool_argument_order_is_preserved(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {"z": {"type": "string"}, "a": {"type": "integer"}},
                        "required": ["z", "a"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ]
        constraint = self.compiler.tools(tools)
        self.check_fixture(
            constraint,
            "<tool_call>write<arg_key>z</arg_key><arg_value>first</arg_value><arg_key>a</arg_key><arg_value>2</arg_value></tool_call><|observation|>",
        )

    def shell_tools(self, strict):
        function = {
            "name": "shell",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        }
        if strict is not None:
            function["strict"] = strict
        return [{"type": "function", "function": function}]

    def accepts(self, constraint, text):
        matcher = xgr.GrammarMatcher(constraint.compiled)
        return all(
            matcher.accept_token(token)
            for token in self.tokenizer.encode(text, add_special_tokens=False)
        )

    VALID_CALL = "<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call><|observation|>"
    SCHEMA_VIOLATIONS = (
        "<tool_call>shell</tool_call><|observation|>",
        VALID_CALL.replace(
            "</tool_call>", "<arg_key>timeout</arg_key><arg_value>soon</arg_value></tool_call>"
        ),
        VALID_CALL.replace(
            "</tool_call>", "<arg_key>unexpected</arg_key><arg_value>value</arg_value></tool_call>"
        ),
    )
    FRAMING_VIOLATIONS = (
        VALID_CALL.replace(">shell<", ">unknown<"),
        "<think>a</think>b</think><|user|>",
        "text<arg_key>command</arg_key><|user|>",
    )

    def test_non_strict_tools_keep_framing_without_argument_schemas(self):
        tools = self.shell_tools(False)
        before = deepcopy(tools)
        constraint = self.compiler.tools(tools)
        self.assertEqual(tools, before)
        self.assertTrue(self.accepts(constraint, self.VALID_CALL))
        for text in self.SCHEMA_VIOLATIONS:
            with self.subTest(text=text):
                self.assertTrue(self.accepts(constraint, text))
        for text in self.FRAMING_VIOLATIONS:
            with self.subTest(text=text):
                self.assertFalse(self.accepts(constraint, text))
        body = "shell<arg_key>timeout</arg_key><arg_value>12</arg_value><arg_key>extra</arg_key><arg_value>café</arg_value>"
        self.check_fixture(
            constraint, f"<think>plan</think><tool_call>{body}</tool_call><|observation|>"
        )
        self.assertEqual(
            parse_tool_call(body, tools),
            {"name": "shell", "arguments": {"timeout": 12, "extra": "café"}},
        )

    def test_strict_and_omitted_tools_enforce_schemas(self):
        for strict in (True, None):
            constraint = self.compiler.tools(self.shell_tools(strict))
            self.assertTrue(self.accepts(constraint, self.VALID_CALL))
            for text in self.SCHEMA_VIOLATIONS + self.FRAMING_VIOLATIONS:
                with self.subTest(strict=strict, text=text):
                    self.assertFalse(self.accepts(constraint, text))

    def test_enforced_schemas_ignore_the_strict_flag(self):
        enforcing = ConstraintCompiler(self.tokenizer, enforce_tool_schemas=True)
        tools = self.shell_tools(False)
        before = deepcopy(tools)
        expected = build_flint_glm_grammar(
            self.tokenizer, tools, enforce_tool_schemas=True
        ).model_dump()
        self.assertEqual(tools, before)
        for strict in (True, None):
            self.assertEqual(
                build_flint_glm_grammar(
                    self.tokenizer, self.shell_tools(strict), enforce_tool_schemas=True
                ).model_dump(),
                expected,
            )
        constraint = enforcing.tools(tools)
        self.check_fixture(constraint, self.VALID_CALL)
        for text in self.SCHEMA_VIOLATIONS + self.FRAMING_VIOLATIONS:
            with self.subTest(text=text):
                self.assertFalse(self.accepts(constraint, text))

    def test_tools_require_object_schemas(self):
        for enforce in (False, True):
            compiler = ConstraintCompiler(self.tokenizer, enforce_tool_schemas=enforce)
            for strict in (False, None):
                for parameters in (None, {"type": "array"}):
                    tools = self.shell_tools(strict)
                    tools[0]["function"]["parameters"] = parameters
                    with (
                        self.subTest(enforce=enforce, strict=strict, parameters=parameters),
                        self.assertRaisesRegex(ValueError, "requires object parameter schemas"),
                    ):
                        compiler.tools(tools)

    def test_callback_error_propagates_and_next_session_survives(self):
        def rejected(token, active):
            raise ValueError("injected parser failure")

        identifier = native.create(rejected)
        try:
            mask = native.mask(
                mx.array(0, mx.int32), mx.array(True), mx.array(identifier, mx.uint64), WORDS
            )
            mx.eval(mask)
            with self.assertRaisesRegex(ValueError, "injected parser failure"):
                native.drain(identifier)
            self.assertEqual(int(mask[-1].item()), 0)
        finally:
            native.close(identifier)
        self.check_fixture(self.ebnf('root ::= "ok"\n'), "ok")


if __name__ == "__main__":
    unittest.main()
