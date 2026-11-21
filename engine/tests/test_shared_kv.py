import unittest
from types import SimpleNamespace

import mlx.core as mx

from rec_mlx.cache import BLOCK_TOKENS, FixedKVCache, KVStorage
from rec_mlx.kernels.megakernel import FullModel
from rec_mlx.quantized import QATSession


class SharedKVTests(unittest.TestCase):
    def storage(self, capacity=17):
        metadata = [[0] * 64 for _ in range(50)]
        for index, row in enumerate(metadata[:49]):
            row[1] = index % 4 == 0
        return KVStorage(metadata, capacity)

    @unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
    def test_native_views_follow_shared_writes_and_growth(self):
        storage = self.storage()
        cache = FixedKVCache(storage, 0)
        keys = mx.arange(4 * 9 * 128).reshape(1, 4, 9, 128).astype(mx.bfloat16)
        values = -keys
        actual = cache.update_and_fetch(keys, values)
        self.assertTrue(mx.array_equal(actual[0], keys).item())
        self.assertTrue(mx.array_equal(storage.layer(0)[1:2, :, :9], values).item())
        ring = mx.arange(4 * 513 * 128).reshape(1, 4, 513, 128).astype(mx.bfloat16)
        storage.write(1, 0, ring, -ring)
        prior = storage.banks[0]
        storage.grow(BLOCK_TOKENS + 65)
        self.assertIs(storage.banks[0], prior)
        self.assertTrue(mx.array_equal(storage.layer(1)[0:1], ring).item())
        self.assertTrue(mx.array_equal(storage.layer(1)[1:2], -ring).item())
        cache.offset = BLOCK_TOKENS - 2
        actual = cache.update_and_fetch(keys, values)
        self.assertTrue(mx.array_equal(actual[0][:, :, BLOCK_TOKENS - 2 :], keys).item())
        self.assertTrue(mx.array_equal(actual[1][:, :, BLOCK_TOKENS - 2 :], values).item())

    def test_prefill_allocates_only_the_current_chunk(self):
        metadata = [[0] * 64 for _ in range(50)]
        metadata[0][1] = 1
        capacities = []
        engine = SimpleNamespace(metadata=metadata, prefill_chunk_size=BLOCK_TOKENS)

        def native(tokens, cache):
            capacities.append(session._storage.capacity)
            return mx.zeros((1, 1, 1))

        def make_cache(count, storage):
            session._storage = storage
            return []

        engine.model = SimpleNamespace(
            context_limit=4 * BLOCK_TOKENS,
            make_cache=make_cache,
            native=SimpleNamespace(model=native, lm_head=lambda x: x),
        )
        engine.prefill_state = lambda *a, **kw: FullModel.prefill_state(engine, *a, **kw)
        session = QATSession(engine, 1)
        session.prefill([1] * (2 * BLOCK_TOKENS + 1))
        self.assertEqual(capacities, [BLOCK_TOKENS, 2 * BLOCK_TOKENS, 3 * BLOCK_TOKENS])
        self.assertEqual(session._state["capacity"], 3 * BLOCK_TOKENS)

    def test_packet_growth_and_limit(self):
        storage = self.storage()
        engine = SimpleNamespace(metadata=storage.rows)
        state = FullModel.prefill_state(engine, 17, limit=BLOCK_TOKENS + 8, storage=storage)
        state["position"] = BLOCK_TOKENS - 4
        FullModel.ensure_capacity(engine, state, 8)
        self.assertIs(state["kv"], storage)
        self.assertIs(state["metadata"], storage.metadata)
        self.assertEqual(state["capacity"], 2 * BLOCK_TOKENS)
        self.assertEqual(state["growth_count"], 1)
        state["position"] = BLOCK_TOKENS + 7
        with self.assertRaisesRegex(ValueError, "Decode context limit exhausted"):
            FullModel.ensure_capacity(engine, state, 2)
        state["usable"] = False
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            FullModel.ensure_capacity(engine, state, 1)

    @unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
    def test_write_async_batches_writes_and_touch_resident_pages(self):
        from rec_mlx.engine import Engine

        storage = self.storage()
        ring = mx.arange(4 * 513 * 128).reshape(1, 4, 513, 128).astype(mx.bfloat16)
        dones = storage.write_async(1, 0, ring, -ring)
        mx.eval(dones)
        self.assertTrue(mx.array_equal(storage.layer(1)[0:1], ring).item())
        self.assertTrue(mx.array_equal(storage.layer(1)[1:2], -ring).item())

        engine = Engine.__new__(Engine)
        engine.mx = mx
        engine._touch_kernel = None
        engine.runtime = SimpleNamespace(banks=[storage.banks[0]])
        engine._prompt_cache = SimpleNamespace(storage=storage)
        engine.touch_resident_pages()
        self.assertIsNotNone(engine._touch_kernel)


if __name__ == "__main__":
    unittest.main()
