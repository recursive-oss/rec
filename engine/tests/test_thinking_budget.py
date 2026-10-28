"""ThinkBudget state machine, think-token resolution and the budget-0 first-token mask."""

import numpy as np
import pytest

from rec_mlx import thinking
from rec_mlx.token_ids import dedicated_token_id

THINK_ID = 7
CLOSE_ID = 8
VOCAB = 12
TEXT = 3


def tb(budget):
    return thinking.ThinkBudget(budget, THINK_ID, CLOSE_ID)


def feed(state, *tokens):
    for token in tokens:
        state.accept(token)


class DedicatedTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def convert_ids_to_tokens(self, idx):
        return next((k for k, v in self.vocab.items() if v == idx), None)

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[text]] if text in self.vocab else [101, 102]


def test_resolve_think_ids_needs_two_dedicated_tokens():
    vocab = {"<think>": THINK_ID, "</think>": CLOSE_ID}
    assert thinking.resolve_think_ids(DedicatedTokenizer(vocab)) == (THINK_ID, CLOSE_ID)
    for missing in vocab:  # a marker that encodes to several ids is not a vocabulary token
        partial = {k: v for k, v in vocab.items() if k != missing}
        with pytest.raises(ValueError):
            thinking.resolve_think_ids(DedicatedTokenizer(partial))


@pytest.mark.parametrize("n", [1, 5])
def test_close_is_required_after_exactly_n_reasoning_tokens(n):
    s = tb(n)
    feed(s, THINK_ID)  # <think> does not count
    for i in range(n):
        assert s.forced_token() is None, f"forced early after {i} tokens"
        s.accept(TEXT)
    assert s.forced_token() == CLOSE_ID
    assert s.forced_token() == CLOSE_ID  # a query, not a state change
    s.accept(CLOSE_ID)  # </think> does not count
    assert (s.reasoning_tokens, s.forced_close, s.inside_think) == (n, True, False)
    assert s.forced_token() is None


def test_tokens_outside_think_are_neither_counted_nor_forced():
    cases = {  # tokens -> (reasoning_tokens, forced_close)
        "natural close": ((THINK_ID, TEXT, 4, CLOSE_ID), (2, False)),
        "empty reasoning": ((THINK_ID, CLOSE_ID), (0, False)),
        "after close": ((THINK_ID, TEXT, 4, CLOSE_ID, *[TEXT] * 50), (2, True)),
        "no think": ((TEXT, 4, 5, 6), (0, False)),
    }
    for name, (tokens, expected) in cases.items():
        s = tb(2 if name == "after close" else 5)
        feed(s, *tokens)
        assert (s.reasoning_tokens, s.forced_close) == expected, name
        assert s.forced_token() is None, name


def test_absent_budget_counts_but_never_forces():
    s = tb(None)
    feed(s, THINK_ID, *[TEXT] * 10_000)
    assert s.forced_token() is None and s.reasoning_tokens == 10_000
    s.accept(CLOSE_ID)
    assert s.forced_close is False


def test_remaining_raises_when_the_budget_is_overshot():
    s = tb(2)
    feed(s, THINK_ID, TEXT, TEXT)
    assert s.remaining() == 0
    s.accept(TEXT)  # a third reasoning token without a forced close
    with pytest.raises(RuntimeError):
        s.remaining()


def test_invalid_budget_values_are_rejected():
    for value in (-1, 1.5, "5", True, False, []):
        with pytest.raises(ValueError):
            tb(value)


def test_only_budget_zero_masks_think_at_the_first_token():
    logits = np.linspace(-1.0, 1.0, VOCAB).astype(np.float32)
    logits[THINK_ID] = 50.0  # the model would pick <think>
    before = logits.copy()
    out = np.asarray(tb(0).mask_first_logits(logits))
    assert out[THINK_ID] == -np.inf and int(np.argmax(out)) != THINK_ID
    keep = [i for i in range(VOCAB) if i != THINK_ID]
    np.testing.assert_array_equal(out[keep], before[keep])
    np.testing.assert_array_equal(logits, before)
    for budget in (None, 1, 64):
        np.testing.assert_array_equal(np.asarray(tb(budget).mask_first_logits(logits)), before)


class BosTokenizer(DedicatedTokenizer):
    """encode() prepends a BOS id unless add_special_tokens is False."""

    def encode(self, text, add_special_tokens=False):
        return ([100] if add_special_tokens else []) + super().encode(text)


@pytest.mark.parametrize("tokenizer", [BosTokenizer, DedicatedTokenizer])
def test_dedicated_token_id_returns_the_vocabulary_id(tokenizer):
    assert dedicated_token_id(tokenizer({"<think>": THINK_ID}), "<think>") == THINK_ID


class NonIntTokenizer(DedicatedTokenizer):
    def convert_tokens_to_ids(self, token):
        return str(THINK_ID)


class WrongRoundTripTokenizer(DedicatedTokenizer):
    def convert_ids_to_tokens(self, idx):
        return "<other>"


class SplitEncodeTokenizer(DedicatedTokenizer):
    def encode(self, text, add_special_tokens=False):
        return [THINK_ID, 1]


@pytest.mark.parametrize(
    "tokenizer",
    [NonIntTokenizer, WrongRoundTripTokenizer, SplitEncodeTokenizer, DedicatedTokenizer],
)
def test_dedicated_token_id_rejects_markers_that_are_not_one_vocabulary_token(tokenizer):
    vocab = {} if tokenizer is DedicatedTokenizer else {"<think>": THINK_ID}  # {} -> unknown id
    with pytest.raises(ValueError, match="dedicated vocabulary token"):
        dedicated_token_id(tokenizer(vocab), "<think>")
