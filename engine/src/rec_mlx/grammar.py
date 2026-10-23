import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import xgrammar as xgr

from . import _grammar_native as native
from .grammar_builder import build_flint_glm_grammar

VOCAB = 150144
WORDS = VOCAB // 32


@dataclass(frozen=True)
class Constraint:
    compiled: object
    stops: tuple[int, ...]

    def session(self):
        return ConstraintSession(self)


class ConstraintCompiler:
    def __init__(self, tokenizer, *, enforce_tool_schemas=False):
        self.tokenizer = tokenizer
        self.enforce_tool_schemas = enforce_tool_schemas
        info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=VOCAB, stop_token_ids=[tokenizer.eos_token_id]
        )
        self.compiler = xgr.GrammarCompiler(info, cache_limit_bytes=128 * 1024**2)

    def tools(self, tools=None):
        tag = build_flint_glm_grammar(
            self.tokenizer, tools, enforce_tool_schemas=self.enforce_tool_schemas
        )
        compiled = self.compiler.compile_structural_tag(tag)
        stops = tuple(
            self.tokenizer.convert_tokens_to_ids(x)
            for x in ("<|end_of_text|>", "<|user|>", "<|observation|>")
        )
        return Constraint(compiled, stops)


class ConstraintSession:
    def __init__(self, constraint):
        self.constraint = constraint
        self.stops = constraint.stops
        self.matcher = xgr.GrammarMatcher(constraint.compiled)
        self.bitmask = xgr.allocate_token_bitmask(1, VOCAB)
        self.numpy_mask = self.bitmask.numpy().view(np.uint32).reshape(-1)
        self.full_mask = np.full(WORDS, 0xFFFFFFFF, np.uint32)
        self.stopped = False
        self.mask_seconds = []
        self.accepted = 0
        first = self._fill()
        self.initial_mask = mx.array(np.concatenate([first, np.array([1], np.uint32)]))
        self.identifier = native.create(self._advance)
        self.handle = mx.array(self.identifier, mx.uint64)
        mx.eval(self.initial_mask, self.handle)

    def _fill(self):
        start = time.perf_counter()
        self.matcher.fill_next_token_bitmask(self.bitmask)
        self.mask_seconds.append(time.perf_counter() - start)
        return self.numpy_mask

    def _advance(self, token, active):
        if not active or self.stopped:
            return self.full_mask
        if not self.matcher.accept_token(token):
            raise RuntimeError(f"Grammar rejected sampled token {token}")
        self.accepted += 1
        if token in self.stops:
            self.stopped = True
            return self.full_mask
        return self._fill()

    def _mask_logits(self, logits, bits):
        words = mx.arange(VOCAB, dtype=mx.uint32)
        allowed = ((bits[words >> 5] >> (words & 31)) & 1) != 0
        return mx.where(allowed, logits, -float("inf"))

    def first_logits(self, logits):
        return self._mask_logits(logits, self.initial_mask)

    def step_logits(self, logits, state):
        return self._mask_logits(logits, state.get("grammar_mask", self.initial_mask))

    def accept_first(self, token, state):
        mask = self._advance(token, True)
        state["grammar_mask"] = mx.array(np.concatenate([mask, np.array([1], np.uint32)]))

    def drain(self):
        native.drain(self.identifier)

    def metrics(self):
        return {
            "grammar_accepted_tokens": self.accepted,
            "grammar_mask_seconds": sum(self.mask_seconds),
            "grammar_max_mask_seconds": max(self.mask_seconds),
        }

    def __enter__(self):
        return self

    def __exit__(self, *error):
        native.close(self.identifier)
