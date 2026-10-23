import time

import mlx.core as mx

from .cache import KVStorage
from .quantized import QATSession


class NativeRuntime:
    def __init__(self, model, prefill_chunk_size=512):
        self.model = model
        self.prefill_chunk_size = prefill_chunk_size
        self.metadata = [[0] * 64 for _ in range(50)]
        for row, layer in zip(self.metadata[:49], model.layers, strict=True):
            row[1] = int(layer.attention.is_global)

    def make_session(self, *, max_new_tokens=512):
        return NativeSession(self, max_new_tokens)

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


class NativeSession(QATSession):
    def prefill(
        self, tokens, *, prefix=None, should_stop=lambda: False, interrupt_between_layers=False
    ):
        logits = super().prefill(
            tokens,
            prefix=prefix,
            should_stop=should_stop,
            interrupt_between_layers=interrupt_between_layers,
        )
        # Global layers append to the shared banks, with independent offsets.
        # Sliding layers must own their rings before decode can overwrite them.
        self.decode_cache = self.prompt_cache.fork().layers
        for row, entry in zip(self.engine.metadata[:49], self.decode_cache, strict=True):
            if not row[1]:
                entry.grow(len(tokens) + self.maximum)
        mx.eval([entry.state for entry in self.decode_cache])
        return logits

    def forward_token(self, token):
        state = self._state
        if not state["usable"]:
            raise RuntimeError("Session was invalidated by a GPU synchronization failure")
        required = state["position"] + 1
        if required > state["limit"]:
            raise ValueError("Decode context limit exhausted")
        if required > state["capacity"]:
            started = time.perf_counter()
            state["kv"].grow(required)
            state["capacity"] = state["kv"].capacity
            state["growth_count"] += 1
            state["growth_seconds"] += time.perf_counter() - started
        logits = self.engine.model(mx.array([[token]], mx.int32), self.decode_cache)
        mx.eval(logits, [entry.state for entry in self.decode_cache])
        state["position"] = required
        return logits
