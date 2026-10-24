import unittest

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask

from rec_mlx.cache import FixedSlidingKVCache
from rec_mlx.model import banded_sliding_attention

WINDOW = 513


class BandedSlidingTests(unittest.TestCase):
    @unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
    def test_matches_full_window_mask_bit_for_bit(self):
        mx.random.seed(0)
        sinks = mx.random.normal((32,)).astype(mx.bfloat16)
        for prior, length in (
            (0, 2048),
            (100, 1000),
            (WINDOW, 2048),
            (WINDOW, 300),
            (WINDOW, 4096),
            (0, 2053),
            (WINDOW, 2048 + 100),
            (WINDOW - 1, 2049),
            (511, 2048),  # first visible key at 63 mod 64: catches an off-by-one band start
            (0, 5),
            (100, 777),
            *[(WINDOW, length) for length in range(2050, 2057)],
            *[
                (prior, length)
                for prior in (0, 100, WINDOW)
                for length in (2, 8, 9, 257, 260, 264, 265)
            ],
        ):
            with self.subTest(prior=prior, length=length):
                q = mx.random.normal((1, 32, length, 128)).astype(mx.bfloat16)
                k = mx.random.normal((1, 4, prior + length, 128)).astype(mx.bfloat16)
                v = mx.random.normal((1, 4, prior + length, 128)).astype(mx.bfloat16)
                mask = create_causal_mask(length, offset=prior, window_size=WINDOW)
                expected = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=128**-0.5, mask=mask, sinks=sinks
                )
                actual = banded_sliding_attention(q, k, v, 128**-0.5, sinks, mask, WINDOW)
                self.assertTrue(mx.array_equal(actual, expected).item())

    @unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
    def test_matches_cache_mask_after_ring_rotation(self):
        # The model passes prior=cache._valid and the cache's own window; a rotated ring must keep that contract.
        mx.random.seed(1)
        sinks = mx.random.normal((32,)).astype(mx.bfloat16)
        cache = FixedSlidingKVCache(
            max_sequence_tokens=8192, num_kv_heads=4, head_dim=128, window_size=WINDOW
        )
        for length in (600, *[1] * 40, 300, 2053, 9):
            q = mx.random.normal((1, 32, length, 128)).astype(mx.bfloat16)
            k = mx.random.normal((1, 4, length, 128)).astype(mx.bfloat16)
            v = mx.random.normal((1, 4, length, 128)).astype(mx.bfloat16)
            prior, mask = cache._valid, cache.make_mask(length, return_array=True)
            k, v = cache.update_and_fetch(k, v)
            if length == 1:
                continue
            with self.subTest(prior=prior, length=length):
                expected = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=128**-0.5, mask=mask, sinks=sinks
                )
                actual = banded_sliding_attention(q, k, v, 128**-0.5, sinks, mask, WINDOW)
                self.assertTrue(mx.array_equal(actual, expected).item())


if __name__ == "__main__":
    unittest.main()
