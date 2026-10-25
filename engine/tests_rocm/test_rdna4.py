"""Independent boundary references for opt-in RDNA4 kernels."""

import math
import unittest

import torch

from rec_rocm.model import rope
from rec_rocm.ordered_attention import decode_pv
from rec_rocm.rdna4 import mask_sink, norm_apply, rope_apply, sink_attention


class ExactPointwise(unittest.TestCase):
    def test_norm_apply(self):
        torch.manual_seed(17)
        for shape in ((1, 2048), (256, 2048), (1, 32, 128), (256, 4, 128)):
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
            reference = (
                x.float()
                * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
                * w.float()
            ).bfloat16()
            self.assertTrue(torch.equal(norm_apply(x, w, 1e-5), reference), str(shape))

    def test_rope_apply(self):
        torch.manual_seed(29)
        for count, heads, position in ((1, 4, 0), (1, 32, 32768), (256, 32, 8192)):
            x = torch.randn((count, heads, 128), device="cuda", dtype=torch.bfloat16)
            positions = torch.arange(position, position + count, device="cuda", dtype=torch.int32)
            theta = 1000000.0
            freq = theta ** (-torch.arange(64, device="cuda", dtype=torch.float32) / 64)
            angle = positions.float()[:, None] * freq[None]
            actual = rope_apply(x, (angle.cos()[:, None], angle.sin()[:, None]))
            self.assertTrue(torch.equal(actual, rope(x, positions, theta)))

    def test_mask_sink(self):
        for count, length, window in ((1, 32768, None), (256, 770, 513)):
            scores = torch.randn((32, count, length), device="cuda")
            sinks = torch.randn(32, device="cuda")
            qp = torch.arange(512, 512 + count, device="cuda", dtype=torch.int32)
            kp = torch.arange(length, device="cuda", dtype=torch.int32).roll(29)
            kp[-1] = -1
            allowed = (kp[None] <= qp[:, None]) & (kp[None] >= 0)
            if window:
                allowed &= kp[None] > qp[:, None] - window
            expected = torch.cat(
                (
                    scores.masked_fill(~allowed[None], -float("inf")),
                    sinks[:, None, None].expand(-1, count, 1),
                ),
                -1,
            )
            self.assertTrue(torch.equal(mask_sink(scores, sinks, qp, kp, window), expected))


class Attention(unittest.TestCase):
    def test_ordered_pv_boundaries_and_graph_position(self):
        torch.manual_seed(113)
        for length, position, window in ((1, 0, None), (1025, 777, None), (513, 1027, 513)):
            with self.subTest(length=length, window=window):
                kp = torch.arange(length, device="cuda", dtype=torch.int32)
                if window:
                    kp = (kp + position - length + 1).roll(137)
                scores = torch.randn(32, 1, length + 1, device="cuda")
                scores[..., :length].masked_fill_((kp > position)[None, None], -float("inf"))
                probability = scores.softmax(-1)[..., :length]
                value = torch.randn(length, 4, 128, device="cuda", dtype=torch.bfloat16)
                # Independent CPU simulation: exact products plus FP32-rounded
                # accumulator after every addition, preserving the physical KV order.
                pp = probability[:, 0].cpu().double()
                vv = value.cpu().double().repeat_interleave(8, dim=1)
                acc = torch.zeros(32, 128)
                for k in range(length):
                    acc = (acc.double() + pp[:, k, None] * vv[k]).float()
                expected = acc.bfloat16()[None]
                pos = torch.tensor([position], device="cuda", dtype=torch.int32)
                self.assertTrue(
                    torch.equal(decode_pv(probability, value, pos, window=window).cpu(), expected)
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = decode_pv(probability, value, pos, window=window)
                graph.replay()
                self.assertTrue(torch.equal(actual.cpu(), expected))
                if not window and length > 1:
                    # Future probabilities may be zero, but a Graph must read a
                    # changed device position rather than freeze a Python length.
                    pos.fill_(0)
                    graph.replay()
                    first = (pp[:, 0, None] * vv[0]).float().bfloat16()[None]
                    self.assertTrue(torch.equal(actual.cpu(), first))

    def test_positions_sink_and_ring(self):
        torch.manual_seed(91)
        for length, position, window in (
            (1, 0, None),
            (514, 512, 513),
            (1025, 777, None),
            (8193, 8191, None),
            (32768, 32766, None),
        ):
            with self.subTest(length=length, position=position, window=window):
                q = torch.randn(1, 4, 128, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(length, 1, 128, dtype=torch.bfloat16, device="cuda")
                v = torch.randn_like(k)
                kp = torch.arange(length, device="cuda", dtype=torch.int32)
                if window:
                    kp = kp.roll(137)
                qp = torch.tensor([position], device="cuda", dtype=torch.int32)
                sink = torch.tensor([-10.0, 0.0, 2.0, 10.0], device="cuda")
                actual = sink_attention(q, k, v, sink, qp, kp, window).float().cpu()
                scores = q[0].cpu().double() @ k[:, 0].cpu().double().T / math.sqrt(128)
                allowed = (kp.cpu() >= 0) & (kp.cpu() <= position)
                if window:
                    allowed &= kp.cpu() > position - window
                scores[:, ~allowed] = -float("inf")
                probabilities = torch.cat((scores, sink.cpu().double()[:, None]), -1).softmax(-1)
                expected = (probabilities[:, :-1] @ v[:, 0].cpu().double())[None]
                relative = (actual.double() - expected).norm() / expected.norm()
                self.assertLess(relative.item(), 0.005)

    def test_empty_slots_are_zero(self):
        q = torch.randn(1, 4, 128, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(1024, 1, 128, device="cuda", dtype=torch.bfloat16)
        out = sink_attention(
            q,
            kv,
            kv,
            torch.zeros(4, device="cuda"),
            torch.tensor([0], device="cuda", dtype=torch.int32),
            torch.full((1024,), -1, device="cuda", dtype=torch.int32),
        )
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))


if __name__ == "__main__":
    unittest.main()
