import unittest

import numpy as np
import torch

from rec_rocm import kernels as k
from rec_rocm import reference as r
from rec_rocm.attention import query_tiled_attention
from rec_rocm.model import rope


class Kernels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def check_quant(self, bits, experts=False):
        n, dim = 65, 192
        shape = (7, n, dim * bits // 8) if experts else (n, dim * bits // 8)
        w = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
        s = torch.rand((*shape[:-1], dim // 64), device="cuda", dtype=torch.bfloat16) / 64
        b = -torch.rand_like(s) / 8
        x = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
        expected = r.unpack_affine(
            w.cpu().numpy(), s.float().cpu().numpy(), b.float().cpu().numpy(), bits
        )
        deq = k.dequantize(w, s, b, bits, dtype=torch.float32).cpu().numpy()
        np.testing.assert_allclose(deq, expected, atol=1e-6, rtol=1e-6)
        ids = torch.tensor([1, 6, 0], device="cuda") if experts else None
        actual = k.gemv(x, w, s, b, bits, ids=ids).float().cpu().numpy()
        selected = expected[[1, 6, 0]] if experts else expected[None]
        wanted = selected @ x.float().cpu().numpy()
        self.assertLess(np.linalg.norm(actual - wanted) / np.linalg.norm(wanted), 0.005)

    def test_int4(self):
        self.check_quant(4)

    def test_int8(self):
        self.check_quant(8)

    def test_expert_gemv(self):
        self.check_quant(4, True)

    def test_sparse_expert_dequant_leaves_inactive_slots(self):
        w = torch.randint(0, 256, (8, 65, 96), device="cuda", dtype=torch.uint8)
        s = torch.randn((8, 65, 3), device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(s)
        ids = torch.tensor([1, 3, 7], device="cuda")
        out = torch.zeros((8, 65, 192), device="cuda", dtype=torch.bfloat16)
        k.dequantize_experts(w, s, b, ids, out, sparse=True)
        reference = r.unpack_affine(
            w.cpu().numpy(), s.float().cpu().numpy(), b.float().cpu().numpy()
        )
        expected = torch.from_numpy(reference).to(device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(out[ids], expected[ids], atol=0, rtol=0)
        self.assertTrue(bool((out[[0, 2, 4, 5, 6]] == 0).all().item()))

    def test_grouped_prefill(self):
        n, dim, m, experts = 80, 192, 19, 7
        w = torch.randint(0, 256, (experts, n, dim // 2), device="cuda", dtype=torch.uint8)
        s = torch.rand((experts, n, dim // 64), device="cuda", dtype=torch.bfloat16) / 64
        b = -torch.rand_like(s) / 8
        x = torch.randn((m, dim), device="cuda", dtype=torch.bfloat16)
        ids = torch.randint(experts, (m, 2), device="cuda")
        actual = k.quantized_mm(x, w, s, b, ids=ids).float().cpu().numpy()
        weights = r.unpack_affine(w.cpu().numpy(), s.float().cpu().numpy(), b.float().cpu().numpy())
        expected = np.stack(
            [
                weights[e] @ x.float().cpu().numpy()[i]
                for i, row in enumerate(ids.cpu().numpy())
                for e in row
            ]
        )
        self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 0.01)

    def test_norm(self):
        x = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        y = k.rms_norm(x, w).float().cpu().numpy()
        wanted = r.rms_norm(x.float().cpu().numpy(), w.float().cpu().numpy(), 1e-6)
        self.assertLess(np.linalg.norm(y - wanted) / np.linalg.norm(wanted), 0.005)

    def test_rotary_long_positions_against_complex_cpu_reference(self):
        positions = torch.tensor([0, 1, 512, 513, 8192, 32768], device="cuda")
        x = torch.randn(6, 3, 128, device="cuda", dtype=torch.bfloat16)
        actual = rope(x, positions).float().cpu().numpy()
        source = x.float().cpu().numpy().astype(np.float64)
        frequencies = np.power(10000.0, -np.arange(64, dtype=np.float64) / 64)
        phase = positions.cpu().numpy()[:, None, None] * frequencies[None, None]
        rotated = (source[..., :64] + 1j * source[..., 64:]) * np.exp(1j * phase)
        expected = np.concatenate([rotated.real, rotated.imag], axis=-1)
        self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 0.005)

    def test_sink_attention_and_window(self):
        for size in (1, 512, 513, 514):
            q = torch.randn(3, 4, 32, device="cuda", dtype=torch.bfloat16)
            key = torch.randn(size, 2, 32, device="cuda", dtype=torch.bfloat16)
            value = torch.randn_like(key)
            sink = torch.tensor([-10.0, 0.0, 2.0, 10.0], device="cuda")
            qp = torch.arange(size - 3, size, device="cuda", dtype=torch.int32)
            kp = torch.arange(size, device="cuda", dtype=torch.int32)
            for window in (None, 513):
                with self.subTest(size=size, window=window):
                    actual = k.attention(q, key, value, sink, qp, kp, window).float().cpu().numpy()
                    expected = r.sink_attention(
                        q.float().cpu().numpy(),
                        key.float().cpu().numpy(),
                        value.float().cpu().numpy(),
                        sink.cpu().numpy(),
                        qp.cpu().numpy(),
                        kp.cpu().numpy(),
                        window,
                    )
                    self.assertLess(
                        np.linalg.norm(actual - expected) / np.linalg.norm(expected), 0.01
                    )

    def test_query_tiled_global_attention(self):
        rows, context = 65, 8193
        q = torch.randn(rows, 4, 32, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(context, 2, 32, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        sink = torch.tensor([-10.0, -1.0, 1.0, 10.0], device="cuda")
        qp = torch.arange(context - rows, context, device="cuda")
        kp = torch.arange(context, device="cuda")
        actual = query_tiled_attention(q, key, value, sink, qp, kp).float().cpu().numpy()
        expected = r.sink_attention(
            q.float().cpu().numpy(),
            key.float().cpu().numpy(),
            value.float().cpu().numpy(),
            sink.cpu().numpy(),
            qp.cpu().numpy(),
            kp.cpu().numpy(),
        )
        self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 0.005)


if __name__ == "__main__":
    unittest.main()
