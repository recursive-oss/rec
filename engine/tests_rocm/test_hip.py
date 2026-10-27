import unittest

import torch
import torch.nn.functional as F

from rec_rocm import gfx1151 as g

IS_GFX1151 = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"


def filled(slots, pos, seed=0, scale=1.0, holes=0):
    from rec_rocm.model import LayerCache

    gen = torch.Generator(device="cuda").manual_seed(seed)
    cache = LayerCache(slots, 4, 128, quantized=True)
    keys = (torch.randn(pos + 1, 4, 128, device="cuda", generator=gen) * scale).bfloat16()
    values = torch.randn(pos + 1, 4, 128, device="cuda", generator=gen).bfloat16()
    cache.update(keys, values, torch.arange(pos + 1, device="cuda", dtype=torch.int32))
    if holes:
        cache.positions[torch.randperm(pos + 1, device="cuda", generator=gen)[:holes]] = -1
    q = (torch.randn(1, 32, 128, device="cuda", generator=gen) * 2).bfloat16()
    sinks = torch.randn(32, device="cuda", generator=gen).bfloat16()
    return cache, q, sinks


def eager(q, cache, slots, pos, sinks, dtype):
    """The decode composition on dequantized K/V: FP32 (or FP64) scores, softmax and PV."""
    ak, av = cache.dense(slots, dtype)
    score = torch.einsum("mkhd,nkd->khmn", q.to(dtype).view(1, 4, 8, 128), ak).reshape(32, 1, -1)
    ap = cache.positions[:slots]
    score = score.masked_fill(~((ap >= 0) & (ap <= pos))[None, None], -float("inf"))
    score = torch.cat((score / 128**0.5, sinks.to(dtype)[:, None, None]), -1)
    probs = torch.softmax(score, -1)[..., :slots]
    return torch.einsum("khmn,nkd->mkhd", probs.view(4, 8, 1, -1), av).reshape(1, 32, 128)


def same(a, b):
    return (a == b).float().mean().item()


@unittest.skipUnless(IS_GFX1151, "HIP kernels are built for gfx1151 only")
class HipAttentionK8V4(unittest.TestCase):
    def setUp(self):
        from rec_rocm import hip

        self.hip = hip

    def test_matches_the_eager_composition_and_the_triton_kernel(self):
        # (capacity, extent, pos, key scale, unwritten slots); extent >= pos + 1 as in decode.
        for slots, extent, pos, scale, holes in (
            (2048, 2048, 0, 1, 0),  # only the current slot
            (2048, 2048, 31, 1, 0),
            (2048, 2048, 1023, 1, 40),
            (2048, 2048, 2047, 3, 0),  # cache exactly full
            (4096, 2048, 1500, 1, 0),  # extent below the capacity
            (34816, 34816, 33791, 1, 0),
            (34816, 34816, 34815, 1, 0),
        ):
            with self.subTest(slots=slots, extent=extent, pos=pos):
                cache, q, sinks = filled(slots, pos, scale=scale, holes=holes)
                positions = torch.tensor([pos], device="cuda", dtype=torch.int32)
                got = self.hip.sink_attention_k8v4(q, cache, extent, positions, sinks)
                triton = g.sink_attention_k8v4(q, cache, extent, positions, sinks)
                want = eager(q, cache, extent, pos, sinks, torch.float32).bfloat16()
                truth = eager(q, cache, extent, pos, sinks, torch.float64)
                error = lambda x: ((x.double() - truth).norm() / truth.norm()).item()  # noqa: E731
                self.assertGreaterEqual(same(got, want), 0.995)
                self.assertGreaterEqual(same(got, want), same(triton, want) - 0.001)
                self.assertLessEqual(error(got), error(want) * 1.02)

    def test_reads_the_position_in_a_graph(self):
        cache, q, sinks = filled(4096, 4095, seed=3)
        positions = torch.tensor([10], device="cuda", dtype=torch.int32)
        self.hip.sink_attention_k8v4(q, cache, 4096, positions, sinks)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.hip.sink_attention_k8v4(q, cache, 4096, positions, sinks)
        for pos in (10, 700, 4095, 0, 3):
            positions.fill_(pos)
            graph.replay()
            want = eager(q, cache, 4096, pos, sinks, torch.float32).bfloat16()
            with self.subTest(pos=pos):
                self.assertGreaterEqual(same(out, want), 0.995)

    def test_detector_catches_perturbed_codes_and_scales(self):
        # Measured (seeds 5/7/11): clean 99.85-99.90%; low bit flipped in 1% of key codes 39-42%;
        # 1% of key scales one BF16 ulp up 53-57%. The thresholds sit between the two.
        cache, q, sinks = filled(4096, 4095, seed=5)
        positions = torch.tensor([4095], device="cuda", dtype=torch.int32)
        want = eager(q, cache, 4096, 4095, sinks, torch.float32).bfloat16()
        clean = same(self.hip.sink_attention_k8v4(q, cache, 4096, positions, sinks), want)
        self.assertGreaterEqual(clean, 0.995)
        gen = torch.Generator(device="cuda").manual_seed(6)
        for name, tensor, flip in (
            ("key codes", cache.key, lambda t: t.view(-1).__ixor__(1)),
            ("key scales", cache.key_scale, lambda t: t.view(torch.int16).view(-1).__iadd__(1)),
        ):
            saved = tensor.clone()
            flat = tensor.view(-1)
            pick = torch.randperm(flat.numel(), device="cuda", generator=gen)[: flat.numel() // 100]
            part = flat[pick].clone()
            flip(part)
            flat[pick] = part
            self.assertFalse(torch.equal(saved, tensor))
            wrong = same(self.hip.sink_attention_k8v4(q, cache, 4096, positions, sinks), want)
            tensor.copy_(saved)
            with self.subTest(name):
                self.assertLess(wrong, 0.9)

    def test_rejects_other_head_layouts(self):
        from rec_rocm.model import LayerCache

        cache = LayerCache(64, 1, 128, quantized=True)
        q = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        positions = torch.zeros(1, device="cuda", dtype=torch.int32)
        with self.assertRaises(ValueError):
            self.hip.sink_attention_k8v4(q, cache, 64, positions, torch.zeros(2, device="cuda"))


if __name__ == "__main__":
    unittest.main()


def make(n, k, experts=None, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (n, k // 2) if experts is None else (experts, n, k // 2)
    meta = shape[:-1] + (k // 64,)
    w = torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8, generator=gen)
    s = (torch.rand(meta, device="cuda", generator=gen) * 0.1).bfloat16()
    b = torch.randn(meta, device="cuda", generator=gen).bfloat16()
    return w, s, b


def blas(x, w, s, b):
    from rec_rocm import kernels as ops

    return F.linear(x, ops.dequantize(w, s, b, dtype=x.dtype))


def blas_experts(x, w, s, b, ids):
    from rec_rocm import kernels as ops

    weight = torch.empty(ids.numel(), w.shape[-2], w.shape[-1] * 2, device="cuda", dtype=x.dtype)
    ops.dequantize_experts(w, s, b, ids.long(), weight)
    return torch.bmm(weight, x.expand(ids.numel(), -1).unsqueeze(-1)).squeeze(-1)


@unittest.skipUnless(IS_GFX1151, "HIP kernels are built for gfx1151 only")
class HipGemv(unittest.TestCase):
    def setUp(self):
        from rec_rocm import hip

        self.hip = hip

    def check(self, got, triton, want, truth=None):
        self.assertGreaterEqual(same(got, want), 0.99)
        self.assertGreaterEqual(same(got, want), same(triton, want) - 0.002)
        if truth is not None:
            error = lambda t: ((t.double() - truth).norm() / truth.norm()).item()  # noqa: E731
            self.assertLessEqual(error(got), error(want) * 1.02)

    def test_gemv_shapes_match_the_blas_path(self):
        # N that is not a multiple of the rows per wave, several K, the lm_head-sized persistent grid.
        for seed, (n, k, residual) in enumerate(
            (
                (5120, 2048, False),
                (2048, 4096, True),
                (2050, 2048, False),
                (37, 2048, True),
                (64, 1024, False),
                (2048, 512, False),
                (2048, 6144, True),  # falls back to the Triton kernel
                (70001, 2048, False),
            )
        ):
            with self.subTest(n=n, k=k, residual=residual):
                w, s, b = make(n, k, seed=seed)
                gen = torch.Generator(device="cuda").manual_seed(100 + seed)
                x = torch.randn(1, k, device="cuda", generator=gen).bfloat16()
                r = torch.randn(1, n, device="cuda", generator=gen).bfloat16() if residual else None
                want = blas(x, w, s, b)
                truth = x.double() @ blas(torch.eye(k, device="cuda").bfloat16(), w, s, b).double()
                if residual:
                    want, truth = want + r, truth.bfloat16().double() + r.double()
                got = self.hip.gemv_w4(x, w, s, b, residual=r)
                self.check(got, g.gemv_w4(x, w, s, b, residual=r), want, truth)

    def test_gate_up_dense_and_experts(self):
        gen = torch.Generator(device="cuda").manual_seed(7)
        x = torch.randn(1, 2048, device="cuda", generator=gen).bfloat16()
        for experts, n in ((None, 1024), (None, 6144), (16, 512)):
            with self.subTest(experts=experts, n=n):
                gate, up = make(n, 2048, experts, seed=1), make(n, 2048, experts, seed=2)
                ids = None
                if experts:
                    ids = torch.tensor([3, 0, 15, 7, 7, 9], device="cuda", dtype=torch.int32)
                    want = F.silu(blas_experts(x, *gate, ids)) * blas_experts(x, *up, ids)
                else:
                    want = F.silu(blas(x, *gate)) * blas(x, *up)
                got = self.hip.gate_up_w4(x, *gate, *up, ids=ids)
                self.check(got, g.gate_up_w4(x, *gate, *up, ids=ids), want)

    def test_down_combine_matches_the_eager_composition(self):
        gen = torch.Generator(device="cuda").manual_seed(8)
        down, shared = make(2048, 512, 16, seed=3), make(2048, 1024, seed=4)
        ids = torch.tensor([3, 0, 15, 7, 7, 9], device="cuda", dtype=torch.int32)
        act = torch.randn(6, 512, device="cuda", generator=gen).bfloat16()
        sact = torch.randn(1, 1024, device="cuda", generator=gen).bfloat16()
        scores = torch.rand(6, device="cuda", generator=gen).bfloat16()
        residual = torch.randn(1, 2048, device="cuda", generator=gen).bfloat16()
        weight = torch.empty(6, 2048, 512, device="cuda", dtype=torch.bfloat16)
        from rec_rocm import kernels as ops

        ops.dequantize_experts(*down, ids.long(), weight)
        routed = torch.bmm(weight, act.unsqueeze(-1)).squeeze(-1)[None]
        want = residual + ((routed * scores[None, :, None]).sum(1) + blas(sact, *shared))
        args = (act, *down, ids, scores, sact, *shared, residual)
        self.check(self.hip.down_combine_w4(*args), g.down_combine_w4(*args), want)

    def test_detector_catches_a_one_ulp_error_in_one_percent_of_scales(self):
        w, s, b = make(4096, 2048, seed=9)
        gen = torch.Generator(device="cuda").manual_seed(10)
        x = torch.randn(1, 2048, device="cuda", generator=gen).bfloat16()
        want = blas(x, w, s, b)
        clean = same(self.hip.gemv_w4(x, w, s, b), want)
        broken = s.clone()
        flat = broken.view(torch.int16).view(-1)
        flat[torch.randperm(flat.numel(), device="cuda", generator=gen)[: flat.numel() // 100]] += 1
        self.assertFalse(torch.equal(broken, s))
        wrong = same(self.hip.gemv_w4(x, w, broken, b), want)
        self.assertGreater(clean, 0.99)
        self.assertLess(wrong, 0.97)

    def test_graph_replay(self):
        w, s, b = make(5120, 2048, seed=11)
        x = torch.zeros(1, 2048, device="cuda", dtype=torch.bfloat16)
        self.hip.gemv_w4(x, w, s, b)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.hip.gemv_w4(x, w, s, b)
        for seed in range(3):
            x.copy_(
                torch.randn(
                    1,
                    2048,
                    device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                )
            )
            graph.replay()
            self.assertTrue(torch.equal(out, self.hip.gemv_w4(x, w, s, b)))
