import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx

from rec_mlx.cache import FixedKVCache, FixedSlidingKVCache, KVStorage
from rec_mlx.native import NativeRuntime, NativeSession
from rec_mlx.prefix import PrefillCancelled
from rec_mlx.quantized import QATModel, QATSession


class TinyModel:
    """No weights: exercise the real BF16 caches with deterministic token values."""

    context_limit = 32
    make_cache = QATModel.make_cache

    def __init__(self):
        self.config = {"num_key_value_heads": 4, "head_dim": 128, "sliding_window": 4}
        self.layers = [
            SimpleNamespace(attention=SimpleNamespace(is_global=index == 0)) for index in range(49)
        ]
        self.calls = []
        self.native = SimpleNamespace(
            model=self.hidden,
            lm_head=lambda hidden: mx.concatenate((hidden, -hidden), axis=-1),
            make_cache=Mock(side_effect=AssertionError("quantized Model.make_cache called")),
        )

    def hidden(self, tokens, cache, *, should_stop=None):
        if should_stop is not None and should_stop():
            raise PrefillCancelled()
        keys = mx.broadcast_to(tokens[:, None, :, None], (1, 4, tokens.shape[1], 128))
        keys = keys.astype(mx.bfloat16)
        score = mx.array(0, mx.float32)
        for index, entry in enumerate(cache):
            _, values = entry.update_and_fetch(keys, keys)
            if index < 2:
                score = score + values[0, 0, :, 0].astype(mx.float32).sum()
        return score.reshape(1, 1, 1)

    def __call__(self, tokens, cache):
        self.calls.append((tokens.tolist(), tokens.dtype, cache))
        return self.native.lm_head(self.hidden(tokens, cache))


@unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
class NativeTests(unittest.TestCase):
    def setUp(self):
        # Preserve real storage layout/writes while reaching bank boundaries cheaply.
        self.blocks = patch("rec_mlx.cache.BLOCK_TOKENS", 8)
        self.blocks.start()
        self.addCleanup(self.blocks.stop)
        self.model = TinyModel()
        self.runtime = NativeRuntime(self.model, prefill_chunk_size=2)
        self.runtime.forward = Mock(side_effect=AssertionError("resident decode called"))

    def session(self, tokens, maximum):
        session = self.runtime.make_session(max_new_tokens=maximum)
        logits = session.prefill(tokens)
        self.assertEqual(logits.shape, (1, 1, 2))
        return session

    def test_runtime_contract_and_shared_prefill_state(self):
        self.assertEqual(NativeRuntime(self.model).prefill_chunk_size, 512)
        self.assertEqual(len(self.runtime.metadata), 50)
        self.assertTrue(all(len(row) == 64 for row in self.runtime.metadata))
        self.assertEqual([row[1] for row in self.runtime.metadata], [1] + [0] * 49)
        self.assertFalse(hasattr(self.runtime, "device_profile"))
        session = self.runtime.make_session()
        self.assertIsInstance(session, NativeSession)
        self.assertIsInstance(session, QATSession)
        self.assertEqual(session.maximum, 512)
        state = self.runtime.prefill_state(2)
        self.assertIsInstance(state["kv"], KVStorage)
        self.assertIs(state["metadata"], state["kv"].metadata)
        self.assertEqual(
            {key: value for key, value in state.items() if key not in ("kv", "metadata")},
            {
                "usable": True,
                "position": 0,
                "capacity": 8,
                "limit": 2,
                "growth_count": 0,
                "growth_seconds": 0.0,
            },
        )
        bank = state["kv"].banks[0]
        reused = self.runtime.prefill_state(9, limit=12, storage=state["kv"])
        self.assertIs(reused["kv"], state["kv"])
        self.assertIs(reused["kv"].banks[0], bank)
        self.assertEqual(reused["capacity"], 16)
        self.assertEqual(reused["limit"], 12)

    def test_multi_token_decode_grows_sliding_without_mutating_prompt(self):
        session = self.session([1, 2], 4)
        prompt = session.prompt_cache
        original_rings = [
            (
                [buffer.tolist() for buffer in layer.key_buffers],
                [buffer.tolist() for buffer in layer.value_buffers],
            )
            for layer in prompt.layers[1:]
        ]
        original_logits = prompt.logits.tolist()
        original_global = prompt.storage.layer(0, 2).tolist()
        original_local = prompt.storage.layer(1).tolist()
        self.assertEqual(session.decode_cache[1].capacity, 4)
        self.assertEqual(prompt.layers[1].capacity, 2)
        self.assertIsInstance(session.decode_cache[0], FixedKVCache)
        self.assertIsInstance(session.decode_cache[1], FixedSlidingKVCache)
        self.assertIs(session.decode_cache[0].storage, prompt.storage)
        self.assertIsNot(session.decode_cache[0], prompt.layers[0])
        self.assertIsNot(session.decode_cache[1], prompt.layers[1])
        for token in range(3, 7):
            logits = session.forward_token(token)
            expected = sum(range(1, token + 1)) + sum(range(max(1, token - 3), token + 1))
            self.assertEqual(logits.tolist(), [[[expected, -expected]]])
        self.assertEqual([call[0] for call in self.model.calls], [[[i]] for i in range(3, 7)])
        self.assertTrue(all(call[1] == mx.int32 for call in self.model.calls))
        self.assertTrue(all(call[2] is session.decode_cache for call in self.model.calls))
        self.assertEqual(session._state["position"], 6)
        self.assertEqual(session._state["initial"], 2)
        self.assertEqual(session._state["growth_count"], 0)
        self.assertEqual(prompt.tokens, [1, 2])
        self.assertEqual(prompt.logits.tolist(), original_logits)
        self.assertTrue(all(layer.offset == 2 for layer in prompt.layers))
        self.assertTrue(all(layer.offset == 6 for layer in session.decode_cache))
        self.assertEqual(prompt.storage.layer(0, 2).tolist(), original_global)
        self.assertEqual(prompt.storage.layer(1).tolist(), original_local)
        for layer, expected in zip(prompt.layers[1:], original_rings, strict=True):
            self.assertEqual([buffer.tolist() for buffer in layer.key_buffers], expected[0])
            self.assertEqual([buffer.tolist() for buffer in layer.value_buffers], expected[1])
            self.assertTrue(all(buffer.dtype == mx.bfloat16 for buffer in layer.key_buffers))
        self.runtime.forward.assert_not_called()
        self.model.native.make_cache.assert_not_called()

    def test_wrapped_sliding_ring_and_exact_prefix_reuse(self):
        session = self.session(list(range(1, 8)), 5)
        prompt = session.prompt_cache
        ring = prompt.layers[1]
        self.assertEqual(ring._ring_index, 1)
        prior = ring._prior()[0].tolist()
        for token in range(8, 13):
            logits = session.forward_token(token)
            expected = sum(range(1, token + 1)) + sum(range(token - 3, token + 1))
            self.assertEqual(logits[0, 0, 0].item(), expected)
            actual = session.decode_cache[1]._prior()[0][0, 0, :, 0].tolist()
            self.assertEqual(actual, list(range(token - 3, token + 1)))
        self.assertEqual(ring._prior()[0].tolist(), prior)
        self.assertEqual(ring._ring_index, 1)
        reused = self.runtime.make_session(max_new_tokens=2)
        logits = reused.prefill(list(range(1, 8)), prefix=prompt)
        self.assertIs(logits, prompt.logits)
        self.assertIs(reused.prompt_cache, prompt)
        self.assertEqual(reused.cached_tokens, 7)
        self.assertEqual(reused.common_prefix_tokens, 7)
        self.assertIs(reused._state["kv"], session._state["kv"])
        self.assertEqual(reused.decode_cache[0].offset, 7)
        self.assertEqual(reused.forward_token(20)[0, 0, 0].item(), 28 + 20 + 5 + 6 + 7 + 20)
        self.assertEqual(ring._prior()[0].tolist(), prior)
        self.assertEqual(prompt.tokens, list(range(1, 8)))

    def test_bank_growth_and_limit_precede_model_call(self):
        session = self.session(list(range(1, 8)), 3)
        bank = session._state["kv"].banks[0]
        session.forward_token(8)
        self.assertEqual(session._state["capacity"], 8)
        self.assertEqual(session._state["growth_count"], 0)
        session.forward_token(9)
        self.assertEqual(session._state["capacity"], 16)
        self.assertEqual(session._state["growth_count"], 1)
        self.assertGreaterEqual(session._state["growth_seconds"], 0.0)
        self.assertIs(session._state["kv"].banks[0], bank)
        self.assertIs(session.decode_cache[0].storage, session._state["kv"])
        self.assertEqual(session._state["kv"].layer(0, 9)[0, 0, :, 0].tolist(), list(range(1, 10)))
        session.forward_token(10)
        with self.assertRaisesRegex(ValueError, "Decode context limit exhausted"):
            session.forward_token(11)
        self.assertEqual(len(self.model.calls), 3)
        self.assertEqual(session._state["position"], 10)
        self.assertEqual(session.decode_cache[0].offset, 10)
        self.assertEqual(session._state["growth_count"], 1)
        session._state["usable"] = False
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            session.forward_token(11)
        self.assertEqual(len(self.model.calls), 3)

    def test_prefill_only_and_cancelled_prefill_keep_qat_contract(self):
        session = self.session([1, 2], 0)
        with self.assertRaisesRegex(ValueError, "Decode context limit exhausted"):
            session.forward_token(3)
        session = self.runtime.make_session(max_new_tokens=1)
        checks = iter([False, True])
        with self.assertRaises(PrefillCancelled):
            session.prefill([1, 2, 3, 4], should_stop=lambda: next(checks))
        self.assertEqual(session.prompt_cache.tokens, [1, 2])
        self.assertFalse(hasattr(session, "decode_cache"))
        session = self.runtime.make_session(max_new_tokens=1)
        session.prefill([1, 2], interrupt_between_layers=True)
        self.assertEqual(session._state["position"], 2)


if __name__ == "__main__":
    unittest.main()
