import unittest

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask

from rec_mlx.cache import FixedKVCache, KVStorage


class GlobalCausalMaskTests(unittest.TestCase):
    @unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
    def test_global_causal_string_matches_array_mask(self):
        metadata = [[0] * 64 for _ in range(50)]
        metadata[0][1] = 1
        cache = FixedKVCache(KVStorage(metadata, 16), 0)
        sinks = mx.random.normal((32,)).astype(mx.bfloat16)
        args = dict(scale=128**-0.5, sinks=sinks)
        # 2-8 queries leave the full kernel (sdpa_vector or the unfused fallback); 9+ use it.
        for offset in (0, 5000):
            for length in (2, 5, 8, 9, 2048):
                with self.subTest(offset=offset, length=length):
                    cache.offset = offset
                    self.assertEqual(cache.make_mask(length), "causal")
                    q = mx.random.normal((1, 32, length, 128)).astype(mx.bfloat16)
                    k = mx.random.normal((1, 4, offset + length, 128)).astype(mx.bfloat16)
                    v = mx.random.normal((1, 4, offset + length, 128)).astype(mx.bfloat16)
                    fast = mx.fast.scaled_dot_product_attention(q, k, v, mask="causal", **args)
                    full = mx.fast.scaled_dot_product_attention(
                        q, k, v, mask=create_causal_mask(length, offset=offset), **args
                    )
                    self.assertTrue(mx.array_equal(fast, full).item())
        cache.offset = 5000
        self.assertIsNone(cache.make_mask(1))
        self.assertIsInstance(cache.make_mask(2048, return_array=True), mx.array)


if __name__ == "__main__":
    unittest.main()
