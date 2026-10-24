"""CPU XGrammar matcher with an explicit ROCm vocabulary mask."""

import torch
import xgrammar as xgr

from rec_mlx.grammar_builder import build_flint_glm_grammar


class ConstraintCompiler:
    def __init__(self, tokenizer, *, enforce_tool_schemas=False):
        self.tokenizer = tokenizer
        self.enforce_tool_schemas = enforce_tool_schemas
        info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=150144, stop_token_ids=[tokenizer.eos_token_id]
        )
        self.compiler = xgr.GrammarCompiler(info, cache_limit_bytes=128 * 1024**2)

    def tools(self, tools=None):
        tag = build_flint_glm_grammar(
            self.tokenizer, tools, enforce_tool_schemas=self.enforce_tool_schemas
        )
        return Constraint(
            self.compiler.compile_structural_tag(tag),
            tuple(
                self.tokenizer.convert_tokens_to_ids(x)
                for x in ("<|end_of_text|>", "<|user|>", "<|observation|>")
            ),
        )


class Constraint:
    def __init__(self, compiled, stops):
        self.compiled, self.stops = compiled, stops

    def session(self):
        return ConstraintSession(self)


class ConstraintSession:
    def __init__(self, constraint):
        self.stops = constraint.stops
        self.matcher = xgr.GrammarMatcher(constraint.compiled)
        self.bitmask = xgr.allocate_token_bitmask(1, 150144)
        self.words = torch.empty(self.bitmask.shape, device="cuda", dtype=torch.int32)
        ids = torch.arange(150144, device="cuda")
        self.word_indices = ids >> 5
        self.shifts = ids & 31
        self.accepted = 0

    def mask(self, logits):
        self.matcher.fill_next_token_bitmask(self.bitmask)
        self.words.copy_(self.bitmask)
        allowed = ((self.words.flatten()[self.word_indices] >> self.shifts) & 1).bool()
        return logits.masked_fill(~allowed, -float("inf"))

    def accept(self, token):
        if not self.matcher.accept_token(token):
            raise RuntimeError(f"Grammar rejected token {token}")
        self.accepted += 1
