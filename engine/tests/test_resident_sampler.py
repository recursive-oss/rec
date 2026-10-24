import unittest

import mlx.core as mx
import numpy as np
from mlx_lm.sample_utils import apply_top_k, apply_top_p, make_sampler

from rec_mlx.kernels.masked_sampling import make_resident_sampler, top_p_top_k

VOCAB = 150144


def vectors():
    rng = np.random.default_rng(7)
    for scale in (0.5, 2.0, 6.0, 12.0):
        yield rng.standard_normal(VOCAB) * scale
    for levels in (2, 8, 64):  # bf16-style ties everywhere, including at rank 20
        yield np.round(rng.standard_normal(VOCAB) * 3 * levels) / levels
    for head in (3, 19, 20, 21, 25):  # peaked head with exactly tied runs around the cut
        x = rng.standard_normal(VOCAB) - 8
        x[rng.choice(VOCAB, head, replace=False)] = 6.0
        yield x
    x = rng.standard_normal(VOCAB) - 8
    x[:40] = np.round(np.linspace(6, 5, 40), 1)
    yield x[rng.permutation(VOCAB)]
    yield np.zeros(VOCAB)


class ResidentSamplerTest(unittest.TestCase):
    def test_matches_make_sampler(self):
        for top_p, top_k in ((0.95, 20), (0.5, 20), (0.99, 1), (0.3, 7), (0.999, 32)):
            for temp in (1.0, 0.6):
                reference = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
                candidate = make_resident_sampler(temp, top_p, top_k)
                for index, x in enumerate(vectors()):
                    x = mx.array(x.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
                    logprobs = x - mx.logsumexp(x)
                    self.assertTrue(
                        mx.array_equal(
                            top_p_top_k(logprobs, top_p, top_k),
                            apply_top_k(apply_top_p(logprobs, top_p), top_k),
                        ).item(),
                        (top_p, top_k, index),
                    )
                    for seed in range(6):
                        mx.random.seed(seed + 100 * index)
                        expected = reference(logprobs).item()
                        mx.random.seed(seed + 100 * index)
                        actual = candidate(logprobs).item()
                        self.assertEqual(actual, expected, (top_p, top_k, temp, index, seed))


if __name__ == "__main__":
    unittest.main()
