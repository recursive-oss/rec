"""Exercise Engine's native generation loop without loading model weights."""

import math
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from rec_mlx.engine import Engine, Sampling
from rec_mlx.prefix import PrefillCancelled


class FakeTokenizer:
    def decode(self, tokens, **kwargs):
        return "".join(f"<{token}>" for token in tokens)


class FakeText:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.emitted = ""

    def add_token(self, token):
        text = self.tokenizer.decode([token])
        self.emitted += text
        return text

    def finish(self, tokens):
        return self.tokenizer.decode(tokens)[len(self.emitted) :]


class FakeSession:
    def __init__(self, runtime, maximum):
        self.runtime = runtime
        self.maximum = maximum
        self.calls = []
        self.cached_tokens = 0
        self.common_prefix_tokens = 0
        self._state = {
            "kv": mx.zeros((32,), dtype=mx.float32),
            "capacity": 32,
            "growth_count": 0,
            "growth_seconds": 0.0,
        }

    def prefill(self, ids, *, prefix=None, should_stop=lambda: False):
        self.prompt_cache = SimpleNamespace(
            tokens=list(ids),
            layers=[mx.zeros((index + 1,), dtype=mx.float32) for index in range(49)],
            checkpoint_bytes=16,
        )
        self.prefill_ids = list(ids)
        self.prefix = prefix
        if should_stop():
            raise PrefillCancelled()
        return self.runtime.logits[0]

    def forward_token(self, token):
        self.calls.append(token)
        index = min(len(self.calls), len(self.runtime.logits) - 1)
        return self.runtime.logits[index]


class FakeRuntime:
    prefill_chunk_size = 512

    def __init__(self, logits):
        self.logits = [mx.array(row, dtype=mx.float32).reshape(1, 1, -1) for row in logits]
        # Match the native contract: exactly 49 cache layers, plus an unused row.
        self.metadata = [[0, int(index % 2 == 0)] for index in range(50)]
        self.sessions = []

    def make_session(self, *, max_new_tokens):
        session = FakeSession(self, max_new_tokens)
        self.sessions.append(session)
        return session


class FakeGrammar:
    stops = (1,)

    def __init__(self):
        self.steps = []
        self.accepted = []
        self.entered = False
        self.closed = False

    @contextmanager
    def session(self):
        self.entered = True
        try:
            yield self
        finally:
            self.closed = True

    def step_logits(self, logits, state):
        mask = state.get("grammar_mask", mx.array([2], dtype=mx.int32))
        allowed_token = int(mask.item())
        self.steps.append((len(self.accepted), allowed_token))
        return mx.where(mx.arange(logits.size) == allowed_token, logits, -float("inf"))

    def accept_first(self, token, state):
        self.accepted.append(token)
        next_token = {2: 0, 0: 1, 1: 1}[token]
        state["grammar_mask"] = mx.array([next_token], dtype=mx.int32)

    def metrics(self):
        return {"grammar_accepted_tokens": len(self.accepted)}


@pytest.fixture(autouse=True)
def fake_streaming_text():
    with patch("rec_mlx.engine.StreamingText", FakeText):
        yield


def build_engine(logits):
    engine = Engine.__new__(Engine)
    engine.mx = mx
    engine.runtime = FakeRuntime(logits)
    engine.resident_sampler = None
    engine.context_limit = 128
    engine.eos = {9}
    engine.tokenizer = FakeTokenizer()
    engine._prompt_cache = None
    engine._prefix_preparation = None
    return engine


def events_for(engine, sampling, **kwargs):
    return list(engine.generate_tokens([1, 2], sampling, **kwargs))


def token_events(events):
    return [event for event in events if "token" in event]


def token_ids(events):
    return [event["token"] for event in token_events(events)]


def test_native_greedy_output_and_single_step_decode():
    engine = build_engine([[0, 3, 1], [4, 2, 0], [0, 1, 5]])
    events = events_for(engine, Sampling(temperature=0, max_tokens=3))
    assert token_ids(events) == [1, 0, 2]
    assert "".join(event["text"] for event in token_events(events)) == "<1><0><2>"
    assert [event["finish_reason"] for event in token_events(events)] == [None, None, "length"]
    assert events[-1]["finish_reason"] == "length"
    session = engine.runtime.sessions[0]
    assert session.maximum == 3
    assert session.prefill_ids == [1, 2]
    assert session.calls == [1, 0]
    assert "token_logprobs" not in events[-1]["metrics"]
    assert all("logprob" not in event for event in token_events(events))


@pytest.mark.parametrize(
    ("temperature", "top_p", "top_k"),
    [(1.0, 1.0, 0), (0.8, 1.0, 0), (0.7, 0.9, 3)],
)
def test_native_seed_repeats_sampling_sequence(temperature, top_p, top_k):
    logits = [[0.1, 1.2, 0.4, 0.7]]
    settings = Sampling(temperature=temperature, top_p=top_p, top_k=top_k, seed=71, max_tokens=12)
    first = events_for(build_engine(logits), settings)
    # Perturb the global RNG between requests; each request must reset it.
    mx.random.seed(991)
    mx.eval(mx.random.uniform(shape=(17,)))
    second = events_for(build_engine(logits), settings)
    assert token_ids(first) == token_ids(second)
    assert len(token_ids(first)) == 12


@pytest.mark.parametrize(("top_p", "top_k"), [(1.0, 1), (0.6, 0), (0.9, 2)])
def test_native_sampler_filters_without_double_temperature_scaling(top_p, top_k):
    from mlx_lm.sample_utils import make_sampler

    calls = []
    inputs = []

    def sampler_factory(**kwargs):
        calls.append(kwargs)
        sampler = make_sampler(**kwargs)

        def record_and_sample(logprobs):
            inputs.append(logprobs.tolist())
            return sampler(logprobs)

        return record_and_sample

    raw_logits = mx.array([0.0, 0.2, 1.5, 2.0], dtype=mx.float32)
    normalized = raw_logits - mx.logsumexp(raw_logits)
    settings = Sampling(temperature=0.7, top_p=top_p, top_k=top_k, seed=3, max_tokens=6)
    # The real sampler owns temperature scaling. Reference and engine must receive
    # the same unscaled normalized logits and consume the same seeded RNG stream.
    reference = make_sampler(temp=settings.temperature, top_p=top_p, top_k=top_k)
    mx.random.seed(settings.seed)
    expected_tokens = [int(reference(normalized).item()) for _ in range(settings.max_tokens)]
    engine = build_engine([raw_logits.tolist()])
    with patch("mlx_lm.sample_utils.make_sampler", side_effect=sampler_factory):
        events = events_for(engine, settings)
    assert calls == [{"temp": 0.7, "top_p": top_p, "top_k": top_k}]
    for received in inputs:
        assert received == pytest.approx(normalized.tolist())
    assert len(inputs) == settings.max_tokens
    assert token_ids(events) == expected_tokens
    if top_k == 1:
        assert token_ids(events) == [3] * settings.max_tokens
    else:
        assert set(token_ids(events)) <= {2, 3}


@pytest.mark.parametrize(("penalty", "expected"), [(0.0, [1, 1, 1]), (2.0, [1, 0, 1])])
def test_presence_penalty_tracks_generated_tokens_only(penalty, expected):
    engine = build_engine([[0, 5, 0], [4, 5.5, 0], [4, 5.5, 0]])
    events = events_for(engine, Sampling(temperature=0, presence_penalty=penalty, max_tokens=3))
    assert token_ids(events) == expected
    if penalty:
        # Token 1 occurs in the prompt, but is not penalized until generated;
        # repeated generations apply presence once, not a frequency penalty.
        assert engine.runtime.sessions[0]._state["presence_mask"].tolist() == [1.0, 1.0, 0.0]


@pytest.mark.parametrize("temperature", [0.0, 0.6, 1.0])
def test_native_logprobs_match_temperature_and_presence_adjusted_logits(temperature):
    rows = [[0.0, 2.0, 1.0], [0.0, 1.0, 3.0], [1.0, 2.0, 0.0]]
    penalty = 1.5
    engine = build_engine(rows)
    events = events_for(
        engine,
        Sampling(
            temperature=temperature,
            top_p=1,
            top_k=0,
            presence_penalty=penalty,
            max_tokens=3,
        ),
        return_logprobs=True,
    )
    expected = []
    seen = set()
    for row, event in zip(rows, token_events(events), strict=True):
        report = [
            (value - (penalty if token in seen else 0)) / (temperature or 1.0)
            for token, value in enumerate(row)
        ]
        maximum = max(report)
        normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in report))
        expected.append(report[event["token"]] - normalizer)
        seen.add(event["token"])
    reported = [event["logprob"] for event in token_events(events)]
    assert reported == pytest.approx(expected, abs=1e-6)
    assert events[-1]["metrics"]["token_logprobs"] == reported


def test_native_metrics_account_for_only_sliding_layers():
    engine = build_engine([[0, 3, 1]])
    events = events_for(engine, Sampling(temperature=0, max_tokens=3))
    metrics = events[-1]["metrics"]
    sliding_bytes = sum(4 * (index + 1) for index in range(49) if index % 2)
    assert metrics["decode_strategy"] == "native_mlx"
    assert metrics["prompt_tokens"] == metrics["prefilled_tokens"] == 2
    assert metrics["cached_prompt_tokens"] == metrics["common_prefix_tokens"] == 0
    assert metrics["generated_tokens"] == metrics["computed_tokens"] == 3
    assert metrics["token_ids"] == [1, 1, 1]
    assert metrics["native_sliding_cache_bytes"] == sliding_bytes
    assert metrics["shared_kv_bytes"] == 128
    assert metrics["prefix_checkpoint_bytes"] == 16
    assert metrics["prompt_cache_bytes"] == sliding_bytes + 128 + 16
    assert metrics["decode_kv_capacity_tokens"] == 32
    assert metrics["decode_kv_growth_count"] == 0
    assert metrics["decode_kv_growth_seconds"] == 0.0
    assert metrics["ttft_seconds"] >= 0
    for name in (
        "prefill_seconds",
        "decode_seconds",
        "request_seconds",
        "decode_tokens_per_second",
    ):
        assert metrics[name] >= 0


def test_native_cancelled_prefill_emits_no_tokens():
    engine = build_engine([[0, 4]])
    assert events_for(engine, Sampling(temperature=0, max_tokens=3), should_stop=lambda: True) == []
    assert engine.runtime.sessions[0].calls == []


def test_native_cancellation_between_tokens_stops_before_next_forward():
    engine = build_engine([[0, 4]])
    cancelled = False
    stream = engine.generate_tokens(
        [1, 2], Sampling(temperature=0, max_tokens=3), should_stop=lambda: cancelled
    )
    first = next(stream)
    assert first["token"] == 1
    cancelled = True
    remaining = list(stream)
    assert len(remaining) == 1
    assert remaining[0]["finish_reason"] == "stop"
    assert remaining[0]["metrics"]["generated_tokens"] == 1
    assert remaining[0]["metrics"]["computed_tokens"] == 1
    assert engine.runtime.sessions[0].calls == []


def test_native_eos_flushes_without_rendering_stop_token():
    engine = build_engine([[0, 8] + [0] * 8, [0] * 9 + [8]])
    events = events_for(engine, Sampling(temperature=0, max_tokens=4), return_logprobs=True)
    assert token_ids(events) == [1, 9]
    assert [event["text"] for event in token_events(events)] == ["<1>", ""]
    assert token_events(events)[-1]["finish_reason"] == "stop"
    assert events[-1]["finish_reason"] == "stop"
    assert events[-1]["metrics"]["computed_tokens"] == 2
    assert len(events[-1]["metrics"]["token_logprobs"]) == 2
    assert engine.runtime.sessions[0].calls == [1]


def test_grammar_uses_updated_mask_and_accepts_each_native_token():
    grammar = FakeGrammar()
    engine = build_engine([[5, 3, 1]])
    events = events_for(
        engine, Sampling(temperature=0, max_tokens=5), grammar=grammar, return_logprobs=True
    )
    assert token_ids(events) == [2, 0, 1]
    assert grammar.accepted == [2, 0, 1]
    assert grammar.steps == [(0, 2), (1, 0), (2, 1)]
    assert grammar.entered and grammar.closed
    assert engine.runtime.sessions[0].calls == [2, 0]
    assert events[-1]["finish_reason"] == "stop"
    assert events[-1]["metrics"]["grammar_accepted_tokens"] == 3
    assert events[-1]["metrics"]["token_logprobs"] == [0.0, 0.0, 0.0]
