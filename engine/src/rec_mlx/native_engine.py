import time
from types import SimpleNamespace
import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

from .cache import KVStorage
from .kernels.device import profile_for_device
from .quantized import QATSession


class NativeEngine:
    def __init__(self, model):
        self.model = model
        device = mx.device_info()
        profile = profile_for_device(device)
        self.device_profile = profile
        self.prefill_chunk_size = profile.prefill_chunk_size
        self.workers = profile.workers

        # Metadata marks global attention (is_global == 1) vs sliding attention (0)
        self.metadata = [[0] * 64 for _ in range(50)]
        for i, layer in enumerate(model.layers):
            self.metadata[i][1] = int(layer.attention.is_global)

    def prefill_state(self, capacity, *, limit=None, storage=None):
        if storage is None:
            storage = KVStorage(self.metadata, capacity)
        else:
            storage.grow(capacity)
        return {
            "usable": True,
            "kv": storage,
            "metadata": storage.metadata,
            "position": 0,
            "capacity": storage.capacity,
            "limit": capacity if limit is None else limit,
            "growth_count": 0,
            "growth_seconds": 0.0,
        }

    def ensure_capacity(self, state, count):
        if not state["usable"]:
            raise RuntimeError("Session was invalidated by a GPU synchronization failure")
        required = state["position"] + count
        if required > state["limit"]:
            raise ValueError("Decode context limit exhausted")
        if required > state["capacity"]:
            started = time.perf_counter()
            state["kv"].grow(required)
            state["capacity"] = state["kv"].capacity
            state["growth_count"] += 1
            state["growth_seconds"] += time.perf_counter() - started

    def make_session(self, max_new_tokens):
        return QATSession(self, max_new_tokens)

    def forward(self, state, token):
        self.ensure_capacity(state, 1)
        inputs = mx.array([[token]], dtype=mx.int32)
        cache = state.get("cache")
        hidden = self.model.native.model(inputs, cache=cache)
        logits = self.model.native.lm_head(hidden[:, -1:])
        state["position"] += 1
        return logits, None


class NativeResidentSampler:
    def __init__(self, engine):
        self.engine = engine

    def block(
        self,
        state,
        token,
        count,
        seed,
        *,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        grammar=None,
        logprobs=None,
    ):
        packet = []
        unfiltered_t1 = (temperature, top_p, top_k) == (1.0, 1.0, 0)
        sampler = (
            make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
            if not unfiltered_t1 and temperature != 0
            else None
        )
        eos_tokens = set(self.engine.model.eos_tokens)
        stops = eos_tokens if grammar is None else set(grammar.stops)
        current = token

        for step in range(count):
            logits = self.engine.forward(state, current)[0]
            flat = logits.reshape(-1).astype(mx.float32)

            if repetition_penalty != 1.0 and "recent_tokens" in state and state["recent_tokens"]:
                indices = mx.array(list(dict.fromkeys(state["recent_tokens"][-64:])))
                selected = flat[indices]
                flat[indices] = mx.where(
                    selected < 0,
                    selected * repetition_penalty,
                    selected / repetition_penalty,
                )

            if presence_penalty and "presence_mask" in state:
                flat = flat - presence_penalty * state["presence_mask"]

            if grammar is not None and "grammar_mask" in state:
                gmask = state["grammar_mask"]
                words = mx.arange(flat.size, dtype=mx.uint32)
                allowed = ((gmask[words >> 5] >> (words & 31)) & 1) != 0
                flat = mx.where(allowed, flat, -float("inf"))

            if temperature == 0:
                next_tok = int(mx.argmax(flat).item())
            elif unfiltered_t1:
                next_tok = int(
                    mx.random.categorical(flat, key=mx.random.key(seed + step)).item()
                )
            else:
                next_tok = int(sampler(flat - mx.logsumexp(flat)).item())

            if logprobs is not None:
                report = flat / temperature if temperature else flat
                logprob_val = float((report[next_tok] - mx.logsumexp(report)).item())
                logprobs.append(logprob_val)

            if presence_penalty and "presence_mask" in state:
                state["presence_mask"][next_tok] = 1.0

            if "recent_tokens" in state:
                state["recent_tokens"].append(next_tok)

            if grammar is not None:
                import numpy as np

                mask = grammar._advance(next_tok, True)
                state["grammar_mask"] = mx.array(np.concatenate([mask, np.array([1], np.uint32)]))

            packet.append(next_tok)
            current = next_tok
            if next_tok in stops:
                break

        return packet


def build_native_decoder(model):
    engine = NativeEngine(model)
    sampler = NativeResidentSampler(engine)
    return engine, sampler
