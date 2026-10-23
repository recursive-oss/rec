import unittest

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask

from rec_mlx.cache import FixedKVCache, KVStorage
from rec_mlx.model import banded_sliding_attention

WINDOW = 513


class BandedSlidingTests(unittest.TestCase):
    @unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
    def test_matches_full_window_mask_bit_for_bit(self):
        mx.random.seed(0)
        sinks = mx.random.normal((32,)).astype(mx.bfloat16)
        for prior, length in ((0, 2048), (100, 1000), (WINDOW, 2048), (WINDOW, 300), (WINDOW, 4096)):
            with self.subTest(prior=prior, length=length):
                q = mx.random.normal((1, 32, length, 128)).astype(mx.bfloat16)
                k = mx.random.normal((1, 4, prior + length, 128)).astype(mx.bfloat16)
                v = mx.random.normal((1, 4, prior + length, 128)).astype(mx.bfloat16)
                mask = create_causal_mask(length, offset=prior, window_size=WINDOW)
                expected = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=128**-0.5, mask=mask, sinks=sinks
                )
                actual = banded_sliding_attention(q, k, v, 128**-0.5, sinks, prior, WINDOW)
                self.assertTrue(mx.array_equal(actual, expected).item())

    @unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
    def test_global_causal_string_matches_array_mask(self):
        metadata = [[0] * 64 for _ in range(50)]
        metadata[0][1] = 1
        cache = FixedKVCache(KVStorage(metadata, 16), 0)
        cache.offset = 5000
        self.assertEqual(cache.make_mask(2048), "causal")
        q = mx.random.normal((1, 32, 2048, 128)).astype(mx.bfloat16)
        k = mx.random.normal((1, 4, 7048, 128)).astype(mx.bfloat16)
        v = mx.random.normal((1, 4, 7048, 128)).astype(mx.bfloat16)
        sinks = mx.random.normal((32,)).astype(mx.bfloat16)
        args = dict(scale=128**-0.5, sinks=sinks)
        fast = mx.fast.scaled_dot_product_attention(q, k, v, mask="causal", **args)
        full = mx.fast.scaled_dot_product_attention(
            q, k, v, mask=create_causal_mask(2048, offset=5000), **args
        )
        self.assertTrue(mx.array_equal(fast, full).item())


if __name__ == "__main__":
    unittest.main()
