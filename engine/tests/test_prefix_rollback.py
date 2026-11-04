import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from rec_mlx.cache import FixedKVCache, FixedSlidingKVCache, KVStorage
from rec_mlx.engine import Engine, Sampling
from rec_mlx.prefix import PrefillCancelled, PrefixTreeCache, PromptCache, select_prefix
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
            self.assertLessEqual(len(prefix.checkpoints), 2)
        self.assertGreater(prefix.checkpoint_bytes, 0)
        self.assertIs(select_prefix(prefix, list(range(13)))[0], prefix)
        restored, common = select_prefix(prefix, list(range(10)))
        self.assertEqual(common, 10)
        self.assertEqual(len(restored.tokens), 10)
        self.assertEqual(select_prefix(restored, [99])[0], None)

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

    def test_subchunk_exact_rewind_without_checkpoints(self):
        prefix = self.prefix()
        self.append(prefix, list(range(12)))
        self.assertEqual(prefix.checkpoints, [])
        diverged = list(range(9)) + [100, 101]
        restored, common = select_prefix(prefix, diverged)
        self.assertIs(restored, prefix)
        self.assertEqual(common, 9)
        self.assertEqual(len(restored.tokens), 9)
        self.assertEqual(restored.layers[0].offset, 9)
        self.append(restored, [100, 101])
        fresh = self.prefix()
        self.append(fresh, diverged)
        for actual, wanted in zip(
            restored.layers[1]._prior(), fresh.layers[1]._prior(), strict=True
        ):
            self.assertTrue(mx.array_equal(actual, wanted).item())

    @unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
    def test_multi_branch_prefix_tree_cache_switch_and_restore(self):
        metadata = [[0] * 64 for _ in range(50)]
        metadata[0][1] = 1
        storage = KVStorage(metadata, 32)
        global_layer = FixedKVCache(storage, 0)
        sliding_layer = FixedSlidingKVCache(
            max_sequence_tokens=16, num_kv_heads=4, head_dim=128, window_size=5
        )
        cache = PromptCache([], [global_layer, sliding_layer], None, storage)
        tree = PrefixTreeCache(max_branches=4)

        def run_prompt(tokens):
            nonlocal cache
            selected = tree.select_best(tokens, cache)
            restored, _ = select_prefix(selected, tokens)
            cached = 0 if restored is None else len(restored.tokens)
            suffix = tokens[cached:]
            if suffix:
                k = mx.broadcast_to(
                    mx.array(suffix, dtype=mx.bfloat16).reshape(1, 1, len(suffix), 1),
                    (1, 4, len(suffix), 128),
                )
                k = mx.contiguous(k)
                v = mx.contiguous(-k)
                cache.layers[0].update_and_fetch(k, v)
                cache.layers[1].update_and_fetch(k, v)
                cache.tokens.extend(suffix)
                cache.logits = mx.array([suffix[-1]], dtype=mx.bfloat16)
                mx.eval(cache.logits, [layer.state for layer in cache.layers])
            tree.save(cache)
            return cached

        branch_a = list(range(1, 11))
        branch_b = list(range(1, 7)) + [201, 202, 203]
        self.assertEqual(run_prompt(branch_a), 0)
        expected_global_a = mx.array(storage.layer(0, len(branch_a)))
        expected_sliding_a = [mx.array(x) for x in cache.layers[1]._prior()]
        mx.eval(expected_global_a, expected_sliding_a)

        self.assertEqual(run_prompt(branch_b), 6)
        self.assertEqual(run_prompt(branch_a), len(branch_a))
        self.assertTrue(mx.array_equal(storage.layer(0, len(branch_a)), expected_global_a).item())
        for actual, wanted in zip(cache.layers[1]._prior(), expected_sliding_a, strict=True):
            self.assertTrue(mx.array_equal(actual, wanted).item())


if __name__ == "__main__":
    unittest.main()
