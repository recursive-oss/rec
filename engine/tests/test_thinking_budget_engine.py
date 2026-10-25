"""Thinking budget in the decode loop, driven through a stub session, sampler and grammar."""

import json
import subprocess
import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_thinking_budget import CLOSE_ID, THINK_ID, DedicatedTokenizer
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from rec_mlx import constants
from rec_mlx.engine import Engine, Sampling
from rec_mlx.grammar import ConstraintCompiler
from rec_mlx.grammar_builder import _PROTOCOL_MARKERS
from rec_mlx.kernels.sampling import WIDTH, ResidentSampler

VOCAB = 16
EOS = 0
TEXT = 3
FIRST_TEXT = 4
REASON = 5
AFTER = 9
SEED = 100
PROMPT = [1, 2]
LOGPROB = -0.5
PACKET = 8  # the decode packet size, a literal in Engine._generate_tokens


class RecordingSampler:
    """Scripted tokens with the real block() feed semantics, recording every call.

    block(state, token, count) feeds `token` at state["position"], returns the packet, feeds all
    but its last token and advances the position. feed_only feeds `token` and returns nothing.
    """

    def __init__(self, stream):
        self.stream = list(stream)
        self.calls = []
        self.fed = []

    def block(self, state, token, count, seed, *, logprobs=None, feed_only=False, **options):
        self.calls.append(
            {"last": token, "count": count, "seed": seed, "feed_only": feed_only, **options}
        )
        position = state["position"]
        if feed_only:
            self.fed.append((position, token))
            state["position"] = position + 1
            return []
        out = [self.stream.pop(0) if self.stream else TEXT for _ in range(count)]
        self.fed.append((position, token))
        self.fed.extend((position + 1 + i, t) for i, t in enumerate(out[:-1]))
        state["position"] = position + len(out)
        if logprobs is not None:
            logprobs.extend([LOGPROB] * len(out))
        return out


def stub_session(first):
    logits = [0.0] * VOCAB
    logits[first] = 10.0
    state = {"kv": SimpleNamespace(nbytes=0), "capacity": 0, "growth_count": 0}
    state |= {"growth_seconds": 0.0, "position": len(PROMPT)}
    return SimpleNamespace(
        prefill=lambda ids, prefix=None, should_stop=None: mx.array([logits], mx.float32),
        cached_tokens=0,
        common_prefix_tokens=0,
        prompt_cache=SimpleNamespace(layers=[], checkpoint_bytes=0, tokens=[]),
        _state=state,
    )


class StubGrammar:
    stops = (EOS,)

    def __init__(self):
        self.accepted = []

    def first_logits(self, logits):
        return logits

    def accept_first(self, token, state):
        self.accepted.append(token)

    def metrics(self):
        return {}


def build(monkeypatch, stream, first=FIRST_TEXT):
    monkeypatch.setattr(
        "rec_mlx.engine.StreamingText",
        lambda tokenizer: SimpleNamespace(add_token=lambda t: "", finish=lambda toks: ""),
    )
    engine = Engine.__new__(Engine)
    engine.mx = mx
    engine.eos = {EOS}
    engine.context_limit = 10_000
    engine.think_ids = (THINK_ID, CLOSE_ID)
    engine.tokenizer = DedicatedTokenizer({"<think>": THINK_ID, "</think>": CLOSE_ID})
    engine.sampler = engine.resident_sampler = RecordingSampler(stream)
    engine.session = stub_session(first)
    engine.runtime = SimpleNamespace(
        make_session=lambda max_new_tokens: engine.session, prefill_chunk_size=10**9, metadata=[]
    )
    engine._prompt_cache = engine.session.prompt_cache
    engine._prefix_preparation = None
    return engine


def run(
    engine, max_tokens, *, budget=None, grammar=True, ids=PROMPT, logprobs=False, seed=SEED, **kw
):
    if grammar is True:
        grammar = StubGrammar() if budget is not None else None  # a budget requires a grammar
    options = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, **kw}
    extra = {} if budget is None else {"thinking_budget": budget}
    events = list(
        engine._generate_tokens(
            ids,
            Sampling(max_tokens=max_tokens, seed=seed, **options),
            should_stop=lambda: False,
            grammar=grammar,
            return_logprobs=logprobs,
            **extra,
        )
    )
    return [e["token"] for e in events[:-1]], events[:-1], events[-1]["metrics"]


def assert_fed_once_in_order(engine, tokens):
    """Every generated token except the last is written to the KV exactly once, in order."""
    fed = engine.sampler.fed
    assert [p for p, _ in fed] == list(range(len(PROMPT), len(PROMPT) + len(fed)))
    assert [t for _, t in fed] == tokens[: len(fed)]
    assert len(fed) == len(tokens) - 1


def test_unbudgeted_run_is_unchanged(monkeypatch):
    stream = [5, 6, 7, 5, 6, 7, 5, 6, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    engine = build(monkeypatch, stream)
    tokens, events, metrics = run(engine, 20, logprobs=True)
    assert tokens == [FIRST_TEXT, *stream]
    calls = engine.sampler.calls
    assert [c["count"] for c in calls] == [PACKET, PACKET, 3]
    assert [c["seed"] for c in calls] == [SEED + 1, SEED + 9, SEED + 17]
    assert [c["last"] for c in calls] == [FIRST_TEXT, stream[7], stream[15]]
    assert events[-1]["finish_reason"] == "length"
    assert metrics["generated_tokens"] == metrics["computed_tokens"] == 20
    assert metrics["token_ids"] == tokens
    assert metrics["thinking_forced_close"] is False
    assert_fed_once_in_order(engine, tokens)


def construct(monkeypatch, tmp_path, tokenizer):
    """The real Engine.__init__ with hardware, weights and decoder replaced by stubs."""
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": [EOS]}))
    tokenizer.chat_template = "template"
    runtime = SimpleNamespace(
        model=SimpleNamespace(native=SimpleNamespace(parameters=lambda: {})),
        prefill_chunk_size=10**9,
        metadata=[],
    )
    patches = {
        "rec_mlx.engine.platform.system": lambda: "Darwin",
        "rec_mlx.engine.platform.machine": lambda: "arm64",
        "rec_mlx.engine.subprocess.check_output": lambda *a, **k: str(64 * 1024**3),
        "rec_mlx.engine.profile_for_device": lambda device: SimpleNamespace(chip="x"),
        "rec_mlx.engine.validate": lambda *a: {
            "max_context": 4096,
            "weight_format": "w",
            "kv_format": "k",
        },
        "rec_mlx.engine.load_tokenizer": lambda directory: tokenizer,
        "rec_mlx.kernels.build_decoder": lambda model, optimized: (runtime, None),
    }
    for target, value in patches.items():
        monkeypatch.setattr(target, value)
    monkeypatch.setattr(
        "rec_mlx.quantized.QATModel",
        type("QAT", (), {"context_limit": 4096, "__init__": lambda s, d: None}),
    )
    return Engine(tmp_path, tmp_path / "m.json", context_limit=4096)


def test_engine_resolves_think_ids_and_tolerates_a_tokenizer_without_them(monkeypatch, tmp_path):
    vocab = {"<think>": THINK_ID, "</think>": CLOSE_ID}
    assert construct(monkeypatch, tmp_path, DedicatedTokenizer(vocab)).think_ids == (
        THINK_ID,
        CLOSE_ID,
    )
    engine = construct(monkeypatch, tmp_path, DedicatedTokenizer({}))
    assert engine.think_ids is None  # such an engine still serves requests without a budget


def test_unbudgeted_run_needs_no_think_tokens(monkeypatch):
    engine = build(monkeypatch, [])
    engine.think_ids = None
    assert len(run(engine, 4)[0]) == 4


@pytest.mark.parametrize("budget", [3, 2 * PACKET + 1])
def test_forced_close_after_n_reasoning_tokens(monkeypatch, budget):
    reasoning = [100 + i for i in range(budget)]  # distinct, so a wrong t_k is visible
    engine = build(monkeypatch, reasoning + [AFTER] * 30, first=THINK_ID)
    grammar = StubGrammar()
    tokens, _, metrics = run(engine, 1 + budget + 1 + 12, budget=budget, grammar=grammar)
    assert tokens[: 2 + budget] == [THINK_ID, *reasoning, CLOSE_ID]
    assert tokens[2 + budget :] == [AFTER] * 12
    assert grammar.accepted.count(CLOSE_ID) == 1
    assert metrics["reasoning_tokens"] == budget and metrics["thinking_forced_close"] is True
    assert metrics["computed_tokens"] == metrics["generated_tokens"] == len(tokens)
    # t_k, the last reasoning token, is fed alone before </think>; no sampling past the close.
    (feed,) = [c for c in engine.sampler.calls if c["feed_only"]]
    assert (feed["count"], feed["last"], feed["seed"]) == (1, reasoning[-1], SEED + 1 + budget)
    assert_fed_once_in_order(engine, tokens)


def test_forced_close_logprobs_stay_aligned_with_tokens(monkeypatch):
    engine = build(monkeypatch, [REASON, REASON] + [AFTER] * 6, first=THINK_ID)
    tokens, events, metrics = run(engine, 6, budget=2, logprobs=True)
    assert tokens == [THINK_ID, REASON, REASON, CLOSE_ID, AFTER, AFTER]
    reported = [e["logprob"] for e in events]
    assert reported[1:4] == [LOGPROB, LOGPROB, constants.FORCED_TOKEN_LOGPROB]
    assert metrics["token_logprobs"] == reported


def test_forced_close_marks_presence_state_and_keeps_penalty(monkeypatch):
    engine = build(monkeypatch, [REASON, REASON, AFTER, AFTER], first=THINK_ID)
    run(engine, 5, budget=2, presence_penalty=0.5)
    mask = engine.session._state["presence_mask"]
    assert float(mask[CLOSE_ID]) == 1.0 and float(mask[THINK_ID]) == 1.0
    assert all(call["presence_penalty"] == 0.5 for call in engine.sampler.calls)


@pytest.mark.parametrize(("max_tokens", "forced"), [(4, True), (3, False)])
def test_max_tokens_decides_whether_the_close_is_reached(monkeypatch, max_tokens, forced):
    engine = build(monkeypatch, [REASON, REASON, AFTER], first=THINK_ID)
    tokens, _, metrics = run(engine, max_tokens, budget=2)
    assert tokens == [THINK_ID, REASON, REASON, CLOSE_ID][:max_tokens]
    assert metrics["thinking_forced_close"] is forced
    assert_fed_once_in_order(engine, tokens)


def test_natural_close_before_budget_is_not_forced(monkeypatch):
    engine = build(monkeypatch, [REASON, CLOSE_ID] + [AFTER] * 10, first=THINK_ID)
    tokens, _, metrics = run(engine, 6, budget=4)
    assert tokens[:3] == [THINK_ID, REASON, CLOSE_ID]
    assert metrics["thinking_forced_close"] is False and metrics["reasoning_tokens"] == 1
    assert_fed_once_in_order(engine, tokens)


def test_budget_zero_never_opens_think(monkeypatch):
    for sampling in ({"temperature": 0.0}, {"temperature": 0.7, "top_p": 0.9, "top_k": 5}):
        for seed in range(5):
            engine = build(monkeypatch, [REASON] * 4, first=THINK_ID)  # the model wants <think>
            tokens, _, metrics = run(engine, 2, budget=0, seed=seed, **sampling)
            assert tokens[0] != THINK_ID, f"seed {seed}"
            assert metrics["reasoning_tokens"] == 0


def test_budget_without_a_grammar_raises_before_any_decode(monkeypatch):
    for budget in (0, 2):
        engine = build(monkeypatch, [])
        with pytest.raises(ValueError, match="grammar"):
            run(engine, 20, budget=budget, grammar=None)
        assert engine.sampler.calls == []


@pytest.mark.parametrize(
    ("budget", "max_tokens", "counts"),
    [(3, 100, [3]), (PACKET + 2, 100, [PACKET, 2]), (1000, 100, [PACKET]), (1000, 4, [3])],
)
def test_packets_never_sample_past_the_forced_close(monkeypatch, budget, max_tokens, counts):
    engine = build(monkeypatch, [], first=THINK_ID)
    run(engine, max_tokens, budget=budget)
    assert [c["count"] for c in engine.sampler.calls][: len(counts)] == counts


def test_validate_rejects_an_unclosed_think_in_the_prompt():
    engine = SimpleNamespace(think_ids=(THINK_ID, CLOSE_ID))
    for ids in ([1, THINK_ID], [1, THINK_ID, 5, 6], [1, THINK_ID, 5, CLOSE_ID, THINK_ID, 6]):
        with pytest.raises(ValueError, match="think"):
            Engine.validate_thinking_budget(engine, ids, 3)
    # An empty closed block (</think> right after <think>) is closed.
    for ids in ([1, THINK_ID, 5, CLOSE_ID, 6], [1, THINK_ID, CLOSE_ID, 6], [1, 2]):
        Engine.validate_thinking_budget(engine, ids, 3)


class FakeKernelEngine:
    """Records every call; any engine call other than forward fails the feed_only tests."""

    def __init__(self):
        self.calls = []

    def forward(self, state, token):
        self.calls.append(("forward", state, token))

    def __getattr__(self, name):
        raise AssertionError(f"feed_only must not touch engine.{name}")


def resident_sampler():
    sampler = object.__new__(ResidentSampler)  # skips the Metal kernel compilation
    sampler.module, sampler.engine = None, FakeKernelEngine()
    return sampler


def test_resident_feed_only_runs_one_forward_pass_and_samples_nothing():
    sampler, state = resident_sampler(), {"position": 3}
    assert sampler.block(state, REASON, 1, SEED, feed_only=True, grammar=object()) == []
    assert sampler.engine.calls == [("forward", state, REASON)]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"feed_only": True, "count": 2}, "exactly one"),
        ({"feed_only": True, "count": 0}, "exactly one"),
        ({"count": WIDTH + 1}, "between one and eight"),  # feed_only defaults to False
        ({"feed_only": False, "count": 0}, "between one and eight"),
    ],
)
def test_resident_block_rejects_bad_counts_before_touching_the_engine(options, message):
    sampler = resident_sampler()
    with pytest.raises(ValueError, match=message):
        sampler.block({}, REASON, options.pop("count"), SEED, **options)
    assert sampler.engine.calls == []


def test_validate_rejects_a_tokenizer_without_think_tokens():
    with pytest.raises(ValueError, match="think"):
        Engine.validate_thinking_budget(SimpleNamespace(think_ids=None), [1, 2], 3)


def test_decode_backstop_rejects_an_open_think_before_sampling(monkeypatch):
    engine = build(monkeypatch, [])
    with pytest.raises(ValueError, match="think"):
        run(engine, 4, budget=2, ids=[*PROMPT, THINK_ID])
    assert engine.sampler.calls == []


@pytest.fixture(scope="module")
def tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    model = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    model.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    model.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=model, eos_token="<|end_of_text|>")
    tok.add_tokens([x for x in _PROTOCOL_MARKERS if x != "<|end_of_text|>"])
    return tok


def test_flint_grammar_accepts_a_forced_close_after_reasoning(tokenizer):
    think, close = (tokenizer.convert_tokens_to_ids(x) for x in ("<think>", "</think>"))
    text = tokenizer.encode("abc", add_special_tokens=False)
    with ConstraintCompiler(tokenizer).tools().session() as session:
        state = {}
        for token in [think, *text, close, *text, tokenizer.eos_token_id]:
            session.accept_first(token, state)


def test_budget_modules_import_without_mlx():
    probe = """
import importlib, json, sys
for name in ("rec_mlx.engine", "rec_mlx.thinking", "rec_mlx.grammar_builder"):
    importlib.import_module(name)
b = sys.modules["rec_mlx.thinking"].ThinkBudget(1, 7, 8)
for token in (7, 3, 8):
    b.accept(token)
print(json.dumps(sorted(m for m in sys.modules if m == "mlx" or m.startswith("mlx."))))
"""
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"
