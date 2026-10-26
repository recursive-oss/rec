from functools import partial

import mlx.core as mx

from .cache import BLOCK_TOKENS, MAX_BLOCKS, FixedKVCache, FixedSlidingKVCache
from .cache import KVStorage as BaseStorage
from .kernels.megakernel import FullModel as BaseModel


class KVStorage(BaseStorage):
    def __init__(self, metadata, capacity, *, block_tokens=BLOCK_TOKENS):
        self.block_tokens = block_tokens
        self.rows = [list(row) for row in metadata]
        global_layers = sum(bool(row[1]) for row in self.rows[:49])
        self.global_elements = global_layers * block_tokens * 1024
        global_index = local_index = 0
        for row in self.rows[:49]:
            if row[1]:
                row[61:63] = [global_index * block_tokens * 1024, block_tokens]
                global_index += 1
            else:
                row[61:63] = [self.global_elements + local_index * 513 * 1024, 513]
                local_index += 1
        self.local_elements = local_index * 513 * 1024
        self.metadata = mx.array(self.rows, mx.uint32)
        mx.eval(self.metadata)
        self.banks = [mx.zeros((1, 1024), mx.bfloat16) for _ in range(MAX_BLOCKS)]
        self.capacity = 0
        self.grow(capacity)

    def grow(self, capacity):
        assert 0 < capacity <= BLOCK_TOKENS * MAX_BLOCKS
        # The shader selects banks at 65536 tokens; compact blocks stay in bank zero.
        assert self.block_tokens == BLOCK_TOKENS or capacity <= self.block_tokens
        count = (capacity + self.block_tokens - 1) // self.block_tokens
        for index in range(self.capacity // self.block_tokens, count):
            size = self.global_elements + (self.local_elements if index == 0 else 0)
            block = mx.zeros((size // 1024, 1024), mx.bfloat16)
            mx.eval(block)
            self.banks[index] = block
            self.capacity = (index + 1) * self.block_tokens

    def reblock(self, block_tokens):
        assert self.block_tokens < block_tokens <= BLOCK_TOKENS
        replacement = KVStorage(self.rows, self.capacity, block_tokens=block_tokens)
        for index, row in enumerate(self.rows[:49]):
            data = self.layer(index, self.capacity if row[1] else 513)
            replacement.write(index, 0, data[0:1], data[1:2])
        self.__dict__.update(replacement.__dict__)


class GlobalCache(FixedKVCache):
    def make_mask(self, n_tokens, window_size=None, return_array=False):
        if n_tokens > 1 and window_size is None and not return_array:
            return "causal"
        return super().make_mask(n_tokens, window_size=window_size, return_array=return_array)


def make_cache(config, layers, max_sequence_tokens, storage):
    dimensions = {
        "max_sequence_tokens": max_sequence_tokens,
        "num_kv_heads": config["num_key_value_heads"],
        "head_dim": config["head_dim"],
    }
    return [
        GlobalCache(storage, index)
        if layer.attention.is_global
        else FixedSlidingKVCache(**dimensions, window_size=config["sliding_window"])
        for index, layer in enumerate(layers)
    ]


class FullModel(BaseModel):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        model.make_cache = partial(make_cache, model.config, model.layers)

    def prefill_state(self, capacity, *, limit=None, storage=None):
        limit = capacity if limit is None else limit
        block_tokens = min(BLOCK_TOKENS, 1 << (limit - 1).bit_length())
        if storage is None:
            storage = KVStorage(self.metadata, capacity, block_tokens=block_tokens)
        else:
            if storage.block_tokens < block_tokens:
                storage.reblock(block_tokens)
            storage.grow(capacity)
        return {
            "usable": True,
            "kv": storage,
            "metadata": storage.metadata,
            "position": 0,
            "capacity": storage.capacity,
            "limit": limit,
            "growth_count": 0,
            "growth_seconds": 0.0,
        }
