from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask

_copy = mx.fast.metal_kernel(
    name="flint_copy_cache",
    input_names=["values", "size"],
    output_names=["snapshot"],
    source="""
        uint i=thread_position_in_grid.x;
        if(i<size[0])snapshot[i]=values[i];
    """,
)


# MLX deepcopy can share mutable Metal inputs; copy into independent storage.
def copy_array(value):
    snapshot = _copy(
        inputs=[value, mx.array([value.size], mx.uint32)],
        grid=(value.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[value.shape],
        output_dtypes=[value.dtype],
    )[0]
    mx.eval(snapshot)
    return snapshot


_write = mx.fast.metal_kernel(
    name="flint_write_kv",
    input_names=["storage", "keys", "values", "layout"],
    output_names=["done"],
    source="""
        uint i=thread_position_in_grid.x, count=layout[3];
        if(i<count*512u) {
            uint head=i/(count*128u), pos=(i/128u)%count, dim=i%128u;
            uint offset=layout[0]+(head*layout[1]+layout[2]+pos)*128u+dim;
            device bfloat16_t* target=(device bfloat16_t*)storage;
            target[offset]=keys[i];
            target[offset+layout[1]*512u]=values[i];
        }
        if(i==0)done[0]=0;
    """,
)

# Four append-only blocks fit alongside the fused decoder's weight/scratch buffers.
BLOCK_TOKENS = 65536
MAX_BLOCKS = 4


class KVStorage:
    def __init__(self, metadata, capacity):
        self.rows = [list(row) for row in metadata]
        global_layers = sum(bool(row[1]) for row in self.rows[:49])
        self.global_elements = global_layers * BLOCK_TOKENS * 1024
        global_index = local_index = 0
        for row in self.rows[:49]:
            if row[1]:
                row[61:63] = [global_index * BLOCK_TOKENS * 1024, BLOCK_TOKENS]
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

    @property
    def nbytes(self):
        return sum(bank.nbytes for bank in self.banks)

    def grow(self, capacity):
        assert 0 < capacity <= BLOCK_TOKENS * MAX_BLOCKS
        count = (capacity + BLOCK_TOKENS - 1) // BLOCK_TOKENS
        for index in range(self.capacity // BLOCK_TOKENS, count):
            size = self.global_elements + (self.local_elements if index == 0 else 0)
            block = mx.zeros((size // 1024, 1024), mx.bfloat16)
            mx.eval(block)
            self.banks[index] = block
            self.capacity = (index + 1) * BLOCK_TOKENS

    def layer(self, index, count=None):
        offset, cap = self.rows[index][61:63]
        global_attention = self.rows[index][1]
        count = (self.capacity if global_attention else cap) if count is None else count
        parts = []
        for block, begin in enumerate(range(0, count, cap)):
            size = min(cap, count - begin)
            part = self.banks[block][offset // 1024 : offset // 1024 + cap]
            parts.append(part.reshape(2, 4, cap, 128)[:, :, :size])
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)

    def write(self, index, offset, keys, values):
        start, cap = self.rows[index][61:63]
        count = keys.shape[-2]
        assert 0 <= offset < offset + count <= (self.capacity if self.rows[index][1] else cap)
        begin = 0
        while begin < count:
            block, position = divmod(offset + begin, cap)
            size = min(count - begin, cap - position)
            result = _write(
                inputs=[
                    self.banks[block],
                    keys[..., begin : begin + size, :],
                    values[..., begin : begin + size, :],
                    mx.array([start, cap, position, size], mx.uint32),
                ],
                grid=(size * 512, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(1,)],
                output_dtypes=[mx.uint32],
            )
            mx.eval(result)
            begin += size


class FixedKVCache:
    def __init__(self, storage, index):
        self.storage, self.index = storage, index
        self.offset = 0

    @property
    def state(self):
        # Evaluating concatenated views here would duplicate every global layer’s KV.
        return self.storage.banks

    def empty(self) -> bool:
        return self.offset == 0

    def size(self) -> int:
        return self.offset

    def is_trimmable(self) -> bool:
        return False

    def make_mask(self, n_tokens: int, window_size=None, return_array: bool = False):
        if n_tokens == 1 and not return_array:
            return None
        return create_causal_mask(n_tokens, offset=self.offset, window_size=window_size)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        stop = self.offset + keys.shape[-2]
        self.storage.write(self.index, self.offset, keys, values)
        self.offset = stop
        data = self.storage.layer(self.index, stop)
        return data[0:1], data[1:2]


class FixedSlidingKVCache:
    def __init__(
        self,
        *,
        max_sequence_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        window_size: int,
        dtype=mx.bfloat16,
    ):
        self.window_size = window_size
        self.capacity = min(window_size, max_sequence_tokens)
        shape = (1, num_kv_heads, self.capacity, head_dim)
        self.key_buffers = (mx.zeros(shape, dtype=dtype), mx.zeros(shape, dtype=dtype))
        self.value_buffers = (
            mx.zeros(shape, dtype=dtype),
            mx.zeros(shape, dtype=dtype),
        )
        self.offset = 0
        self._valid = 0
        self._slot = 0
        self._ring_index = 0

    def grow(self, capacity):
        capacity = min(self.window_size, capacity)
        if capacity > self.capacity:
            padding = [(0, 0), (0, 0), (0, capacity - self.capacity), (0, 0)]
            self.key_buffers = tuple(mx.pad(value, padding) for value in self.key_buffers)
            self.value_buffers = tuple(mx.pad(value, padding) for value in self.value_buffers)
            self.capacity = capacity

    @property
    def state(self):
        return self.key_buffers, self.value_buffers

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in (*self.key_buffers, *self.value_buffers))

    def empty(self) -> bool:
        return self.offset == 0

    def size(self) -> int:
        return self._valid

    def is_trimmable(self) -> bool:
        return False

    def make_mask(self, n_tokens: int, window_size=None, return_array: bool = False):
        window = self.window_size if window_size is None else window_size
        if n_tokens == 1 and not return_array:
            return None
        if self._valid == 0 and n_tokens <= window and not return_array:
            return "causal"
        return create_causal_mask(n_tokens, offset=self._valid, window_size=window)

    def _prior(self):
        keys = self.key_buffers[self._slot]
        values = self.value_buffers[self._slot]
        if self._valid == self.capacity and self._ring_index:
            keys = mx.concatenate(
                (keys[..., self._ring_index :, :], keys[..., : self._ring_index, :]),
                axis=-2,
            )
            values = mx.concatenate(
                (
                    values[..., self._ring_index :, :],
                    values[..., : self._ring_index, :],
                ),
                axis=-2,
            )
            return keys, values
        return keys[..., : self._valid, :], values[..., : self._valid, :]

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if keys.shape[-2] == 1:
            active_keys = self.key_buffers[self._slot]
            active_values = self.value_buffers[self._slot]
            if self._valid < self.capacity:
                index = self._valid
                self._valid += 1
            else:
                index = self._ring_index
                self._ring_index = (self._ring_index + 1) % self.capacity
            active_keys[..., index : index + 1, :] = keys
            active_values[..., index : index + 1, :] = values
            self.offset += 1
            return (
                active_keys[..., : self._valid, :],
                active_values[..., : self._valid, :],
            )

        prior_keys, prior_values = self._prior()
        attention_keys = mx.concatenate((prior_keys, keys), axis=-2)
        attention_values = mx.concatenate((prior_values, values), axis=-2)
        retained = min(self.capacity, attention_keys.shape[-2])
        next_slot = 1 - self._slot
        self.key_buffers[next_slot][..., :retained, :] = attention_keys[..., -retained:, :]
        self.value_buffers[next_slot][..., :retained, :] = attention_values[..., -retained:, :]
        self.key_buffers = tuple(mx.contiguous(value) for value in self.key_buffers)
        self.value_buffers = tuple(mx.contiguous(value) for value in self.value_buffers)
        self._slot = next_slot
        self._valid = retained
        self._ring_index = 0
        self.offset += keys.shape[-2]
        return attention_keys, attention_values
