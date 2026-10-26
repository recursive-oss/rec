import unittest

import torch
import torch.nn.functional as F

from rec_rocm import gfx1151 as g
from rec_rocm import kernels as ops
from rec_rocm.gfx1151 import gemv_w4

IS_GFX1151 = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"


def make(n, k, experts=None, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (n, k // 2) if experts is None else (experts, n, k // 2)
    meta = shape[:-1] + (k // 64,)
    w = torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8, generator=g)
    s = (torch.rand(meta, device="cuda", generator=g) * 0.1).bfloat16()
    b = torch.randn(meta, device="cuda", generator=g).bfloat16()
    return w, s, b


@unittest.skipUnless(IS_GFX1151, "fused decode kernels are qualified only on gfx1151")
class FusedGemv(unittest.TestCase):
    def blas_dense(self, x, w, s, b):
        return F.linear(x, ops.dequantize(w, s, b, dtype=x.dtype))

    def blas_experts(self, x, w, s, b, ids, x_expert):
        weight = torch.empty(
            ids.numel(), w.shape[-2], w.shape[-1] * 2, device="cuda", dtype=x.dtype
        )
        ops.dequantize_experts(w, s, b, ids.long(), weight)
        inputs = x if x_expert else x.expand(ids.numel(), -1)
        return torch.bmm(weight, inputs.unsqueeze(-1)).squeeze(-1)

    def agreement(self, got, want):
        rel = ((got.float() - want.float()).norm() / want.float().norm()).item()
        return rel, (got == want).float().mean().item()

    def test_dense_shapes_match_the_blas_path(self):
        # K=192 and N=37 exercise a non-power-of-two group count and row masking.
        for n, k in (
            (512, 2048),
            (2048, 512),
            (4096, 2048),
            (2050, 2048),
            (37, 192),
            (64, 64),
            (6144, 2048),
        ):
            w, s, b = make(n, k)
            x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
            rel, same = self.agreement(gemv_w4(x, w, s, b), self.blas_dense(x, w, s, b))
            with self.subTest(n=n, k=k):
                self.assertLess(rel, 1e-3)
                self.assertGreater(same, 0.995)

    def test_expert_paths_match_the_blas_path(self):
        w, s, b = make(512, 2048, experts=16)
        ids = torch.tensor(
            [3, 0, 15, 7, 7, 9], device="cuda", dtype=torch.int32
        )  # includes a repeated expert
        for x_expert in (False, True):
            x = torch.randn(6 if x_expert else 1, 2048, device="cuda", dtype=torch.bfloat16)
            with self.subTest(x_expert=x_expert):
                rel, same = self.agreement(
                    gemv_w4(x, w, s, b, ids.long(), x_expert),
                    self.blas_experts(x, w, s, b, ids, x_expert),
                )
                self.assertLess(rel, 1e-3)
                self.assertGreater(same, 0.995)

    def test_detector_catches_a_one_ulp_error_in_one_percent_of_scales(self):
        # Scales are BF16, so a multiplicative 1.001 would round back to the same value and perturb nothing.
        # Over 200 random draws a correct kernel agrees with BLAS on >= 0.998 of outputs and this perturbation
        # leaves <= 0.959; the thresholds sit in that gap.
        w, s, b = make(4096, 2048)
        x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
        want = self.blas_dense(x, w, s, b)
        _, clean = self.agreement(gemv_w4(x, w, s, b), want)
        broken = s.clone()
        flat = broken.view(torch.int16).view(-1)
        flat[torch.randperm(flat.numel(), device="cuda")[: flat.numel() // 100]] += 1
        self.assertFalse(torch.equal(broken, s))
        _, wrong = self.agreement(gemv_w4(x, w, broken, b), want)
        self.assertGreater(clean, 0.99)
        self.assertLess(wrong, 0.97)

    def test_router_ties_bias_and_mixture_weights(self):
        # Exact BF16 logits via a one-hot input; ties at 1 and -2, and a bias that lifts expert 7.
        controlled = torch.tensor([-2, -1, 0, 1, 1, 2, 3, -2], dtype=torch.float64)
        weight = torch.zeros(8, 64, device="cuda", dtype=torch.bfloat16)
        weight[:, 0] = controlled.to(weight)
        h = torch.zeros(1, 64, device="cuda", dtype=torch.bfloat16)
        h[0, 0] = 1
        bias = torch.zeros(8, device="cuda")
        bias[7] = 0.16
        ids, scores = g.route(h, weight, bias, 6)
        probabilities = 1 / (1 + torch.exp(-controlled))
        wanted = torch.argsort(-(probabilities + bias.double().cpu()), stable=True)[:6]
        self.assertEqual(ids.cpu().tolist(), wanted.tolist())
        mixture = (probabilities[wanted] / probabilities[wanted].sum()).bfloat16()
        torch.testing.assert_close(scores.cpu(), mixture, atol=0, rtol=0)

    def test_router_matches_eager_router_on_random_logits(self):
        for experts, top_k in ((256, 6), (8, 2)):
            gen = torch.Generator(device="cuda").manual_seed(experts)
            weight = (torch.randn(experts, 2048, device="cuda", generator=gen) * 0.05).bfloat16()
            bias = torch.randn(experts, device="cuda", generator=gen) * 0.01
            for _ in range(16):
                h = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16, generator=gen)
                route = F.linear(h, weight).float().sigmoid()
                eager = torch.argsort(route + bias, dim=-1, descending=True, stable=True)[0, :top_k]
                ids, scores = g.route(h, weight, bias, top_k)
                with self.subTest(experts=experts, top_k=top_k):
                    # Logits come from a different dot-product order than rocBLAS, so they can differ by
                    # one BF16 ulp; selections must still agree on almost every draw.
                    if torch.equal(ids.long(), eager):
                        picked = route[0, eager]
                        want = (picked / (picked.sum() + 1e-20)).to(h.dtype)
                        if top_k == 6:
                            torch.testing.assert_close(scores, want, atol=0, rtol=0)
                        else:
                            torch.testing.assert_close(scores, want)

    def attention_truth(self, q, ak, av, ap, pos, sinks, window):
        keep = (ap >= 0) & (ap <= pos)
        if window:
            keep &= ap > pos - window
        kv_heads, dim = ak.shape[1], ak.shape[2]
        group = q.shape[1] // kv_heads
        k, v = ak[keep].double(), av[keep].double()
        score = torch.einsum("kgd,nkd->kgn", q[0].double().view(kv_heads, group, dim), k) / dim**0.5
        sink = sinks.double().view(kv_heads, group)
        peak = torch.maximum(score.max(-1).values, sink)
        e = torch.exp(score - peak[..., None])
        probs = e / (e.sum(-1) + torch.exp(sink - peak))[..., None]
        return torch.einsum("kgn,nkd->kgd", probs, v).reshape(1, -1, dim)

    def check_attention(self, slots, pos, window, kv_heads=4, heads=32, dim=128, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        ak = torch.randn(slots, kv_heads, dim, device="cuda", generator=gen).bfloat16()
        av = torch.randn(slots, kv_heads, dim, device="cuda", generator=gen).bfloat16()
        ap = torch.full((slots,), -1, device="cuda", dtype=torch.int32)
        if window:
            # Ring buffer: slot = position % slots, only the most recent `slots` positions present.
            for position in range(max(0, pos - slots + 1), pos + 1):
                ap[position % slots] = position
        else:
            ap[: pos + 1] = torch.arange(pos + 1, device="cuda", dtype=torch.int32)
            ap[pos + 1 :] = (
                torch.randint(0, pos + 1, (slots - pos - 1,), device="cuda", generator=gen) * 0 - 1
            )
        q = torch.randn(1, heads, dim, device="cuda", generator=gen).bfloat16()
        sinks = torch.randn(heads, device="cuda", generator=gen).bfloat16()
        positions = torch.tensor([pos], device="cuda", dtype=torch.int32)
        got = g.sink_attention(q, ak, av, positions, ap, sinks, window).double()
        want = self.attention_truth(q, ak, av, ap, pos, sinks.float(), window)
        # The eager path's own error against FP64 is ~1.8e-3, dominated by the BF16 output rounding.
        self.assertLess(((got - want).norm() / want.norm()).item(), 2.5e-3)

    def test_attention_full_and_window_shapes(self):
        cases = [
            (2048, 0, None),  # only the current slot
            (2048, 1, None),
            (2048, 1023, None),
            (2048, 2047, None),  # cache exactly full
            (34816, 33791, None),
            (513, 3, 513),  # ring not yet wrapped
            (513, 512, 513),
            (513, 513, 513),  # first wrap: window boundary
            (513, 5000, 513),
            (64, 100, 64),  # non-Flint sizes take the same paths
        ]
        for slots, pos, window in cases:
            with self.subTest(slots=slots, pos=pos, window=window):
                self.check_attention(slots, pos, window)
        with self.subTest("fixture shape"):
            self.check_attention(256, 77, None, kv_heads=1, heads=2, dim=32)

    def test_attention_reads_the_position_from_device_memory_in_a_graph(self):
        gen = torch.Generator(device="cuda").manual_seed(3)
        slots = 4096
        ak = torch.randn(slots, 4, 128, device="cuda", generator=gen).bfloat16()
        av = torch.randn(slots, 4, 128, device="cuda", generator=gen).bfloat16()
        ap = torch.arange(slots, device="cuda", dtype=torch.int32)
        q = torch.randn(1, 32, 128, device="cuda", generator=gen).bfloat16()
        sinks = torch.randn(32, device="cuda", generator=gen).bfloat16()
        positions = torch.tensor([10], device="cuda", dtype=torch.int32)
        g.sink_attention(q, ak, av, positions, ap, sinks)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = g.sink_attention(q, ak, av, positions, ap, sinks)
        for pos in (10, 700, 4095, 3):
            positions.fill_(pos)
            graph.replay()
            want = self.attention_truth(q, ak, av, ap, pos, sinks.float(), None)
            with self.subTest(pos=pos):
                self.assertLess(((out.double() - want).norm() / want.norm()).item(), 2.5e-3)

    def test_attention_detector_catches_perturbed_keys(self):
        # Error against FP64 averages out over thousands of keys; agreement with the eager
        # composition on identical inputs is the discriminating measure.
        gen = torch.Generator(device="cuda").manual_seed(5)
        ak = torch.randn(4096, 4, 128, device="cuda", generator=gen).bfloat16()
        av = torch.randn(4096, 4, 128, device="cuda", generator=gen).bfloat16()
        q = (torch.randn(1, 32, 128, device="cuda", generator=gen) * 2).bfloat16()
        sinks = torch.randn(32, device="cuda", generator=gen).bfloat16()
        positions = torch.tensor([4095], device="cuda", dtype=torch.int32)
        ap = torch.arange(4096, device="cuda", dtype=torch.int32)
        score = torch.einsum("mkhd,nkd->khmn", q.float().view(1, 4, 8, 128), ak.float()).reshape(
            32, 1, 4096
        )
        score = torch.cat((score / 128**0.5, sinks.float()[:, None, None]), -1)
        probs = torch.softmax(score, -1)[..., :4096]
        eager = (
            torch.einsum("khmn,nkd->mkhd", probs.view(4, 8, 1, 4096), av.float())
            .reshape(1, 32, 128)
            .bfloat16()
        )
        broken = ak.clone()
        flat = broken.view(torch.int16).view(-1)
        flat[torch.randperm(flat.numel(), device="cuda", generator=gen)[: flat.numel() // 100]] += 1
        clean = (g.sink_attention(q, ak, av, positions, ap, sinks) == eager).float().mean().item()
        wrong = (
            (g.sink_attention(q, broken, av, positions, ap, sinks) == eager).float().mean().item()
        )
        self.assertGreater(clean, 0.98)
        self.assertLess(wrong, 0.85)

    def moe_eager(self, h, ids, gate, up, down):
        """The eager BLAS-path arithmetic per (token, slot): BF16 weights, BF16 rounding between ops."""
        rows = []
        for token in range(h.shape[0]):
            for e in ids[token].tolist():
                gw, uw, dw = (
                    ops.dequantize(*(t[e] for t in m), dtype=torch.bfloat16)
                    for m in (gate, up, down)
                )
                x = h[token : token + 1]
                act = F.silu(F.linear(x, gw)) * F.linear(x, uw)
                rows.append(F.linear(act, dw)[0])
        return torch.stack(rows)

    def check_moe(self, ids, experts=16, hidden=256, inner=128, seed=0):
        gate, up, down = (
            make(inner, hidden, experts=experts, seed=seed),
            make(inner, hidden, experts=experts, seed=seed + 1),
            make(hidden, inner, experts=experts, seed=seed + 2),
        )
        gen = torch.Generator(device="cuda").manual_seed(seed + 3)
        h = torch.randn(ids.shape[0], hidden, device="cuda", dtype=torch.bfloat16, generator=gen)
        got = g.moe_prefill(h, gate, up, down, ids)
        want = self.moe_eager(h, ids, gate, up, down)
        rel, same = self.agreement(got, want)
        self.assertLess(rel, 2e-3)
        self.assertGreater(same, 0.95)

    def test_moe_prefill_routing_patterns(self):
        gen = torch.Generator(device="cuda").manual_seed(11)
        tokens = 37  # not a multiple of any tile height
        cases = {
            "random top-2": torch.stack(
                [torch.randperm(16, device="cuda", generator=gen)[:2] for _ in range(tokens)]
            ),
            "one hot expert": torch.stack(
                [torch.tensor([5, 1 + i % 3], device="cuda") for i in range(tokens)]
            ),
            "many empty experts": torch.stack(
                [torch.tensor([0, 15], device="cuda") for _ in range(tokens)]
            ),
            "top-6": torch.stack(
                [torch.randperm(16, device="cuda", generator=gen)[:6] for _ in range(tokens)]
            ),
        }
        for name, ids in cases.items():
            with self.subTest(name):
                self.check_moe(ids.long())

    def test_moe_prefill_detector_catches_perturbed_scales(self):
        ids = torch.stack([torch.randperm(16, device="cuda")[:2] for _ in range(64)]).long()
        gate, up, down = (
            make(128, 256, experts=16),
            make(128, 256, experts=16, seed=1),
            make(256, 128, experts=16, seed=2),
        )
        h = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
        want = self.moe_eager(h, ids, gate, up, down)
        broken = gate[1].clone()
        flat = broken.view(torch.int16).view(-1)
        flat[torch.randperm(flat.numel(), device="cuda")[: flat.numel() // 100]] += 1
        _, clean = self.agreement(g.moe_prefill(h, gate, up, down, ids), want)
        _, wrong = self.agreement(g.moe_prefill(h, (gate[0], broken, gate[2]), up, down, ids), want)
        # Over 200 random draws: clean agreement >= 0.9999, perturbed <= 0.955.
        self.assertGreater(clean, 0.99)
        self.assertLess(wrong, 0.97)


if __name__ == "__main__":
    unittest.main()
