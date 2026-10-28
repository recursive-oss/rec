"""End-to-end thinking budget tests on the real Flint weights and an Apple GPU.

They skip, with a reason, when the weights or the GPU are unavailable or free memory is low:

    REC_E2E_MODEL_DIR=/path/to/flint-int4 python -m pytest tests/e2e

The tests call Engine.generate_tokens directly, so no server or port is involved.
"""

from __future__ import annotations

import numpy as np
import pytest
from e2e_config import SKIP_REASON

PARITY_BUDGET = 100_000  # larger than any generation here: the model must close on its own
PARITY_MAX_TOKENS = 384
FORCED_BUDGET = 20
PROBE_MAX_TOKENS = 1024  # baseline run proving the prompt reasons past FORCED_BUDGET
TAIL = 24  # tokens generated after the forced close
FOLLOW_UP_TOKENS = 24
PLAIN_PROMPTS = [
    "What is 17 times 23? Answer with the number.",
    "In one sentence, what does the Unix command `ls -la` do?",
]
REASONING_PROMPT = (
    "A farmer has 17 sheep. All but 9 run away, then he buys twice as many as remain. "
    "How many sheep does he have? Think carefully and show the arithmetic."
)

pytestmark = pytest.mark.skipif(SKIP_REASON is not None, reason=SKIP_REASON or "")


def _user(text):
    return [{"role": "user", "content": text}]


def _run(engine, text, *, max_tokens, **options):
    """Greedy generation through Engine.generate_tokens; returns (prompt ids, token ids, metrics)."""
    from rec_mlx.engine import Sampling

    ids = engine.prompt(_user(text))
    events = list(
        engine.generate_tokens(
            ids,
            Sampling(temperature=0.0, top_p=1.0, top_k=0, max_tokens=max_tokens, seed=0),
            grammar=engine.constraints.tools(None),
            **options,
        )
    )
    metrics = next(e["metrics"] for e in events if "metrics" in e)
    return ids, list(metrics["token_ids"]), metrics


def _cold_logits(engine, ids):
    """Last-position logits of a cold prefill (no prefix) as float32 numpy."""
    mx = engine.mx
    with mx.stream(mx.new_stream(mx.gpu)):
        logits = engine.runtime.make_session(max_new_tokens=1).prefill(ids, prefix=None)
        mx.eval(logits)
        out = np.array(logits.reshape(-1).astype(mx.float32))
        mx.synchronize()
    return out


def test_large_budget_is_token_identical_to_no_budget(engine):
    for prompt in PLAIN_PROMPTS:
        _, base, _ = _run(engine, prompt, max_tokens=PARITY_MAX_TOKENS)
        _, budgeted, metrics = _run(
            engine, prompt, max_tokens=PARITY_MAX_TOKENS, thinking_budget=PARITY_BUDGET
        )
        assert budgeted == base
        assert metrics["thinking_forced_close"] is False


def test_forced_close_continues_exactly_as_a_cold_model_would(engine, think):
    think_id, close_id = think
    _, base, _ = _run(engine, REASONING_PROMPT, max_tokens=PROBE_MAX_TOKENS)
    assert base[0] == think_id
    assert close_id not in base[: 2 + FORCED_BUDGET], "prompt must reason past the budget"
    ids, tokens, metrics = _run(
        engine, REASONING_PROMPT, max_tokens=1 + FORCED_BUDGET + 1 + TAIL,
        thinking_budget=FORCED_BUDGET,
    )  # fmt: skip
    assert metrics["reasoning_tokens"] == FORCED_BUDGET
    assert metrics["thinking_forced_close"] is True
    assert metrics["computed_tokens"] == metrics["generated_tokens"]
    close_at = tokens.index(close_id)
    assert close_at == 1 + FORCED_BUDGET and tokens[close_at + 1 :]
    # The KV written around the forced close must predict what a cold prefill predicts.
    for end in range(close_at, len(tokens) - 1):
        logits = _cold_logits(engine, ids + tokens[: end + 1])
        assert int(np.argmax(logits)) == tokens[end + 1], f"divergence after position {end}"


def test_budget_zero_has_no_think_token(engine, think):
    _, tokens, metrics = _run(engine, REASONING_PROMPT, max_tokens=64, thinking_budget=0)
    assert think[0] not in tokens
    assert metrics["reasoning_tokens"] == 0 and metrics["thinking_forced_close"] is False


def test_follow_up_reusing_the_prefix_equals_a_cold_run(engine, think):
    """A follow-up turn that reuses the KV of a forced-close answer decodes as a cold run does."""
    from rec_mlx.engine import Sampling

    ids, tokens, metrics = _run(
        engine, REASONING_PROMPT, max_tokens=1 + FORCED_BUDGET + 1 + TAIL,
        thinking_budget=FORCED_BUDGET,
    )  # fmt: skip
    assert metrics["thinking_forced_close"] is True, "prompt must reason past the budget"
    engine.prepare_prefix()
    sampling = Sampling(temperature=0.0, top_p=1.0, top_k=0, max_tokens=FOLLOW_UP_TOKENS, seed=0)

    def follow():
        events = list(
            engine.generate_tokens(ids + tokens, sampling, grammar=engine.constraints.tools(None))
        )
        return events[-1]["metrics"]

    warm = follow()
    assert warm["cached_prompt_tokens"] > 0, "the follow-up must reuse the prefix"
    engine._prompt_cache = None
    engine._prefix_preparation = None
    cold = follow()
    assert cold["cached_prompt_tokens"] == 0
    assert warm["token_ids"] == cold["token_ids"]
