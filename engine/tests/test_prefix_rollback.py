import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from rec_mlx.cache import FixedSlidingKVCache
from rec_mlx.engine import Engine, Sampling
from rec_mlx.prefix import MAX_CHECKPOINTS, PrefillCancelled, PromptCache, select_prefix
from rec_mlx.quantized import QATSession


class PrefixRollbackTests(unittest.TestCase):
    def prefix(self):
        ring = FixedSlidingKVCache(max_sequence_tokens=5, num_kv_heads=1, head_dim=1, window_size=5)
        return PromptCache([], [SimpleNamespace(offset=0), ring], None, object())

    def append(self, prefix, values):
        data = mx.array(values, mx.bfloat16).reshape(1, 1, -1, 1)
        prefix.layers[1].update_and_fetch(data, -data)
        prefix.layers[0].offset += len(values)
        prefix.tokens.extend(values)
        prefix.logits = data[:, :, -1:]
        mx.eval(prefix.layers[1].state)

    def test_mismatch_restores_physical_ring_then_replays_suffix(self):
        prefix = self.prefix()
        self.append(prefix, list(range(7)))
        self.append(prefix, [7])
        expected = [mx.array(x) for x in prefix.layers[1]._prior()]
        prefix.checkpoint()
        self.append(prefix, [8, 9, 10])
        prefix.checkpoint()
        self.append(prefix, [11, 12])
        storage = prefix.storage
        restored, common = select_prefix(prefix, list(range(9)) + [100, 101])
        self.assertIs(restored, prefix)
        self.assertIs(restored.storage, storage)
        self.assertEqual(common, 9)
        self.assertEqual(len(restored.tokens), 8)
        self.assertEqual(restored.layers[0].offset, 8)
        for actual, wanted in zip(restored.layers[1]._prior(), expected, strict=True):
            self.assertTrue(mx.array_equal(actual, wanted).item())
        self.append(restored, [8, 100, 101])
        fresh = self.prefix()
        self.append(fresh, list(range(9)) + [100, 101])
        for actual, wanted in zip(
            restored.layers[1]._prior(), fresh.layers[1]._prior(), strict=True
        ):
            self.assertTrue(mx.array_equal(actual, wanted).item())

    def test_shortening_exact_hit_early_edit_and_snapshot_bound(self):
        prefix = self.prefix()
        for value in range(12):
            prefix.checkpoint()
            self.append(prefix, [value])
        self.assertGreater(prefix.checkpoint_bytes, 0)
        self.assertIs(select_prefix(prefix, list(range(13)))[0], prefix)
        restored, common = select_prefix(prefix, list(range(10)))
        self.assertEqual(common, 10)
        self.assertEqual(len(restored.tokens), 10)
        self.assertEqual(select_prefix(restored, [99])[0], None)

    def test_new_prompt_sharing_only_the_head_reuses_an_early_checkpoint(self):
        prefix = self.prefix()
        for value in range(12):
            prefix.checkpoint()
            self.append(prefix, [value])
        restored, common = select_prefix(prefix, [0, 1, 2, 99])
        self.assertEqual(common, 3)
        self.assertEqual(restored.tokens, [0, 1, 2])
        self.assertEqual(restored.layers[0].offset, 3)

    def test_checkpoints_are_bounded(self):
        prefix = self.prefix()
        for value in range(MAX_CHECKPOINTS + 4):
            prefix.checkpoint()
            self.append(prefix, [value])
            self.assertLessEqual(len(prefix.checkpoints), MAX_CHECKPOINTS)
        self.assertEqual(len(prefix.checkpoints), MAX_CHECKPOINTS)
        self.assertEqual(prefix.checkpoints[0].position, 4)

    def test_cancelled_native_prefill_retains_last_completed_chunk(self):
        engine = SimpleNamespace(prefill_chunk_size=2)
        storage = SimpleNamespace(grow=lambda capacity: None)
        engine.prefill_state = lambda *a, **kw: {"kv": storage}
        engine.model = SimpleNamespace(
            context_limit=32,
            make_cache=lambda *a: [],
            native=SimpleNamespace(
                model=lambda tokens, cache: tokens[..., None], lm_head=lambda x: x
            ),
        )
        session = QATSession(engine, 1)
        checks = iter([False, True])
        with self.assertRaises(PrefillCancelled):
            session.prefill([1, 2, 3, 4], should_stop=lambda: next(checks))
        self.assertEqual(session.prompt_cache.tokens, [1, 2])
        self.assertEqual(session.prompt_cache.logits.item(), 2)
        restored, common = select_prefix(session.prompt_cache, [1, 2, 5])
        self.assertEqual(common, 2)
        self.assertIs(restored, session.prompt_cache)

    @unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
    def test_generation_selects_prefix_once_before_cancelled_prefill(self):
        prefix = self.prefix()
        self.append(prefix, list(range(8)))
        prefix.checkpoint()
        self.append(prefix, [8, 9, 10])
        prefix.checkpoint()
        self.append(prefix, [11, 12])
        runtime = SimpleNamespace(
            prefill_chunk_size=2,
            prefill_state=lambda *args, **kwargs: {"kv": SimpleNamespace(rows=[[0, 1], [0, 0]])},
            model=SimpleNamespace(context_limit=32, native=None),
        )
        session = QATSession(runtime, 1)
        runtime.make_session = lambda **kwargs: session
        engine = Engine.__new__(Engine)
        engine.mx = mx
        engine.runtime = runtime
        engine.context_limit = 32
        engine.eos = {0}
        engine._prompt_cache = prefix
        engine._prefix_preparation = None
        tokens = list(range(9)) + [100, 101]
        settings = Sampling(temperature=0, max_tokens=1)
        with (
            patch("rec_mlx.prefix.select_prefix", wraps=select_prefix) as select,
            patch("rec_mlx.quantized.select_prefix", select),
            self.assertLogs(level="INFO") as logs,
        ):
            self.assertEqual(
                list(engine.generate_tokens(tokens, settings, should_stop=lambda: True)), []
            )
        self.assertEqual(select.call_count, 1)
        self.assertEqual(session.common_prefix_tokens, 9)
        self.assertIs(engine._prompt_cache, prefix)
        self.assertEqual(prefix.tokens, list(range(8)))
        self.assertEqual(
            logs.output,
            ["INFO:root:Prefill: prompt=11 common_prefix=9 reused=8 new=3"],
        )


if __name__ == "__main__":
    unittest.main()
