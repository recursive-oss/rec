import threading
import unittest
import weakref
from types import SimpleNamespace

import mlx.core as mx

from rec_mlx.cache import BLOCK_TOKENS, FixedKVCache, KVStorage
from rec_mlx.engine import Engine, Sampling
from rec_mlx.kernels.megakernel import FullModel
from rec_mlx.prefix import PrefillCancelled, PrefixPreparation, PromptCache
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
        first_storage = session.prompt_cache.storage
        first_bank0 = first_storage.banks[0]
        capacities.clear()
        second = QATSession(engine, 1)
        second.prefill([2] * 16, prefix=session.prompt_cache)
        self.assertIs(second.prompt_cache.storage, first_storage)
        self.assertIs(first_storage.banks[0], first_bank0)
        self.assertEqual(capacities, [BLOCK_TOKENS])
        self.assertEqual(second._state["capacity"], BLOCK_TOKENS)
        self.assertEqual(first_storage.banks[1].shape, (1, 1024))
        self.assertEqual(first_storage.banks[2].shape, (1, 1024))

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


class EngineMemoryTests(unittest.TestCase):
    def make_engine(self):
        engine = Engine.__new__(Engine)
        engine.mx = mx
        engine._local = threading.local()
        engine.context_limit = 64
        engine.eos = {0}
        engine._prompt_cache = None
        engine._prefix_preparation = None
        return engine

    def storage(self, capacity=17):
        metadata = [[0] * 64 for _ in range(50)]
        for index, row in enumerate(metadata[:49]):
            row[1] = index % 4 == 0
        return KVStorage(metadata, capacity)

    def test_engine_reuses_thread_local_gpu_stream(self):
        engine = self.make_engine()
        first = engine._gpu_stream()
        self.assertIs(engine._gpu_stream(), first)

        def stub_generate_tokens(
            ids, sampling, *, should_stop, grammar=None, return_logprobs=False
        ):
            return iter([])

        engine._generate_tokens = stub_generate_tokens
        sampling = Sampling(temperature=0, max_tokens=1)
        list(engine.generate_tokens([1, 2, 3], sampling, should_stop=lambda: False))
        list(engine.generate_tokens([4, 5], sampling, should_stop=lambda: False))
        self.assertIs(engine._local.stream, first)

        stream = {}

        def worker():
            stream["value"] = engine._gpu_stream()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.assertIsNot(stream["value"], first)

    def test_engine_drops_stale_prefix_preparation_before_prefill(self):
        engine = self.make_engine()
        engine.tokenizer = SimpleNamespace(
            backend_tokenizer=None, decode=lambda tokens, skip_special_tokens=False: ""
        )
        engine._prompt_cache = PromptCache(
            tokens=[9, 9, 9],
            layers=[SimpleNamespace(nbytes=0) for _ in range(49)],
            logits=None,
            storage=object(),
        )
        weak = weakref.ref(engine._prompt_cache)
        session = SimpleNamespace(
            prompt_cache=None,
            _state={
                "kv": SimpleNamespace(nbytes=0),
                "capacity": 0,
                "growth_count": 0,
                "growth_seconds": 0,
            },
            cached_tokens=0,
            common_prefix_tokens=0,
            calls=[],
        )

        def prefill(
            tokens, *, prefix=None, should_stop=lambda: False, interrupt_between_layers=False
        ):
            had_prefix = prefix is not None
            prefix = None
            self.assertIsNone(weak())
            session.calls.append((list(tokens), had_prefix))
            session.prompt_cache = SimpleNamespace(
                tokens=[9, 9, 9],
                layers=[SimpleNamespace(nbytes=0) for _ in range(49)],
                logits=None,
                checkpoint_bytes=0,
            )
            raise PrefillCancelled()

        session.prefill = prefill
        engine.runtime = SimpleNamespace(
            make_session=lambda **kwargs: session,
            prefill_chunk_size=2,
            metadata=[[0] * 64 for _ in range(50)],
        )
        preparation = engine._prefix_preparation = PrefixPreparation(
            origin=SimpleNamespace(tokens=[1, 2, 3]),
            tokens=[1, 2, 3, 4, 5],
            cache=SimpleNamespace(tokens=[1, 2, 3, 4, 5], checkpoints=[]),
        )
        sampling = Sampling(temperature=0, max_tokens=1)
        self.assertEqual(
            list(engine.generate_tokens([9, 9, 9], sampling, should_stop=lambda: False)), []
        )
        self.assertEqual(session.calls, [([9, 9, 9], True)])
        self.assertIsNone(engine._prefix_preparation)
        self.assertIs(engine._prompt_cache, session.prompt_cache)
        self.assertIsNot(engine._prompt_cache, preparation.origin)


if __name__ == "__main__":
    unittest.main()
