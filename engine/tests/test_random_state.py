import unittest
from functools import partial

import mlx.core as mx

from rec_mlx.random_state import restore_random_key


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def compiled_draw():
    return mx.random.uniform(shape=(32,))


class RandomStateTests(unittest.TestCase):
    def test_restore_replays_eager_and_compiled_draws(self):
        for seed in (42, 0xFEDCBA9876543210):
            for draw in (lambda: mx.random.uniform(shape=(32,)), compiled_draw):
                with self.subTest(seed=seed, draw=draw):
                    mx.random.seed(seed)
                    initial = mx.random.state[0]
                    initial_words = initial.tolist()
                    expected = draw()
                    final = mx.random.state[0]
                    final_words = final.tolist()
                    restore_random_key(initial)
                    actual = draw()
                    self.assertTrue(mx.array_equal(actual, expected).item())
                    self.assertEqual(mx.random.state[0].tolist(), final_words)
                    self.assertEqual(initial.tolist(), initial_words)
                    self.assertEqual(final.tolist(), final_words)
