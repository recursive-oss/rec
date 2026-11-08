import unittest
from types import SimpleNamespace

import mlx.core as mx

from rec_mlx.cache import FixedSlidingKVCache
from rec_mlx.engine import Engine, Sampling
from rec_mlx.prefix import PrefillCancelled, PromptCache


@unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
class PreparedPrefixTests(unittest.TestCase):
    def engine(self):
        engine = Engine.__new__(Engine)
        engine.mx = mx
        engine._prompt_cache = PromptCache([10, 11], [], mx.array([0]), object())
        engine._prefix_preparation = None
        calls = []

        class Session:
            def prefill(self, tokens, *, prefix, should_stop, interrupt_between_layers):
                added = tokens[len(prefix.tokens) :]
                calls.append(added)
                prefix.checkpoint()
                prefix.advance(added, mx.array([sum(added)]))
                self.prompt_cache = prefix

        engine.runtime = SimpleNamespace(
            prefill_chunk_size=4, make_session=lambda **kwargs: Session()
        )
        return engine, calls

    def test_full_chunks_only_and_original_cache_untouched(self):
        engine, calls = self.engine()
        original = engine._prompt_cache
        generated = list(range(13))
        engine._schedule_prefix(original.tokens, generated)
        metrics = engine.prepare_prefix()
        prepared = engine._prefix_preparation.cache
        self.assertEqual(calls, [generated[:4], generated[4:8], generated[8:12]])
        self.assertEqual(prepared.tokens, original.tokens + generated[:12])
        self.assertEqual(original.tokens, [10, 11])
        self.assertIs(prepared.storage, original.storage)
        self.assertEqual(metrics["prepared_tokens"], 12)

    def test_short_output_never_schedules_different_batch_arithmetic(self):
        engine, calls = self.engine()
        engine._schedule_prefix(engine._prompt_cache.tokens, [1, 2, 3])
        self.assertIsNone(engine.prepare_prefix())
        self.assertEqual(calls, [])

    def test_preempt_before_start_and_between_chunks_then_resume(self):
        engine, calls = self.engine()
        engine._schedule_prefix(engine._prompt_cache.tokens, list(range(10)))
        self.assertEqual(engine.prepare_prefix(should_stop=lambda: True)["prepared_tokens"], 0)
        self.assertIsNone(engine._prefix_preparation.cache)
        checks = iter([False, True])
        self.assertEqual(
            engine.prepare_prefix(should_stop=lambda: next(checks))["prepared_tokens"], 4
        )
        self.assertEqual(engine._prefix_preparation.cache.tokens, [10, 11, 0, 1, 2, 3])
        self.assertEqual(engine.prepare_prefix()["prepared_tokens"], 4)
        self.assertEqual(calls, [list(range(4)), list(range(4, 8))])

    def test_only_matching_prepared_prefix_can_replace_original(self):
        for changed in [False, True]:
            with self.subTest(changed=changed):
                engine, _ = self.engine()
                original = engine._prompt_cache
                generated = list(range(10))
                engine._schedule_prefix(original.tokens, generated)
                engine.prepare_prefix()
                prepared = engine._prefix_preparation.cache
                follow = original.tokens + generated + [99]
                if changed:
                    follow[3] = -1
                selected = engine._take_prefix(follow)
                self.assertIs(selected, original if changed else prepared)
                self.assertIsNone(engine._prefix_preparation)
                self.assertIsNone(engine._prompt_cache)

    def test_fork_does_not_alias_sliding_buffers(self):
        layer = FixedSlidingKVCache(
            max_sequence_tokens=5, num_kv_heads=1, head_dim=1, window_size=5
        )
        data = mx.arange(7).reshape(1, 1, 7, 1).astype(mx.bfloat16)
        layer.update_and_fetch(data[..., :5, :], -data[..., :5, :])
        layer.update_and_fetch(data[..., 5:6, :], -data[..., 5:6, :])
        original = PromptCache(list(range(6)), [layer], mx.array([0]), object())
        copied = original.fork()
        expected = layer._prior()[0].tolist()
        copied.layers[0].update_and_fetch(data[..., 6:7, :], -data[..., 6:7, :])
        self.assertEqual(layer._prior()[0].tolist(), expected)
        self.assertEqual(copied.layers[0]._ring_index, (layer._ring_index + 1) % 5)

    def test_edited_tail_can_reuse_an_earlier_full_chunk(self):
        engine, _ = self.engine()
        origin = engine._prompt_cache
        generated = list(range(14))
        engine._schedule_prefix(origin.tokens, generated)
        engine.prepare_prefix()
        follow = origin.tokens + generated + [99]
        follow[len(origin.tokens) + 10] = -1
        selected = engine._take_prefix(follow)
        self.assertEqual(selected.tokens, origin.tokens + generated[:8])
        self.assertEqual(origin.tokens, [10, 11])

    def test_partially_prepared_chunk_does_not_replace_committed_prefix(self):
        engine, _ = self.engine()
        origin = engine._prompt_cache
        engine._schedule_prefix(origin.tokens, list(range(10)))
        checks = iter([False, True])
        engine.prepare_prefix(should_stop=lambda: next(checks))
        committed = engine._prefix_preparation.cache
        old_logits = committed.logits.tolist()

        class InterruptedSession:
            def prefill(self, tokens, *, prefix, should_stop, interrupt_between_layers):
                self.prompt_cache = prefix
                prefix.logits = mx.array([-999])
                raise PrefillCancelled()

        engine.runtime.make_session = lambda **kwargs: InterruptedSession()
        self.assertEqual(engine.prepare_prefix()["prepared_tokens"], 0)
        self.assertIs(engine._prefix_preparation.cache, committed)
        self.assertEqual(committed.logits.tolist(), old_logits)
        self.assertEqual(origin.tokens, [10, 11])

    def test_cancelled_prefill_keeps_original_anchor_for_edited_history(self):
        engine, _ = self.engine()
        original = engine._prompt_cache
        generated = list(range(14))
        engine._schedule_prefix(original.tokens, generated)
        engine.prepare_prefix()
        prepared = engine._prefix_preparation.cache

        class CancelledSession:
            common_prefix_tokens = 0
            cached_tokens = 0

            def prefill(self, tokens, *, prefix, should_stop):
                self.prompt_cache = prefix
                raise PrefillCancelled()

        engine.runtime.make_session = lambda **kwargs: CancelledSession()
        engine.context_limit = 64
        engine.eos = {0}
        follow = original.tokens + generated + [99]
        self.assertEqual(list(engine.generate_tokens(follow, Sampling(max_tokens=1))), [])
        self.assertIs(engine._prompt_cache, original)
        self.assertIs(engine._prefix_preparation.cache, prepared)
        follow[3] = -1
        self.assertIs(engine._take_prefix(follow), original)


if __name__ == "__main__":
    unittest.main()
