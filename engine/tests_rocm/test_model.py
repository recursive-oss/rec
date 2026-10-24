import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from fixture import make_checkpoint

from rec_rocm.model import FlintModel, GraphDecoder
from rec_rocm.reference import unpack_affine


class SmallFlint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = make_checkpoint(self.tmp.name)
        self.model = FlintModel(path, capacity=1024)
        self.reference = FlintModel(path, capacity=1024, reference=True)

    def tearDown(self):
        self.tmp.cleanup()

    def relative(self, a, b):
        return ((a - b).norm() / b.norm()).item()

    def test_forward_and_teacher_forcing(self):
        tokens = torch.tensor([1, 20, 12, 85, 100, 101, 3, 50, 9], device="cuda")
        a = self.model.forward(tokens, trace=True)
        b = self.reference.forward(tokens, trace=True)
        self.assertLess(self.relative(a, b), 0.01)
        for a, b in zip(self.model.last_trace, self.reference.last_trace, strict=True):
            self.assertLess(self.relative(a, b), 0.01)
        for token in range(16):
            t = torch.tensor([token], device="cuda")
            a, b = self.model.forward(t), self.reference.forward(t)
            self.assertLess(self.relative(a, b), 0.01)

    def test_prefix_snapshot_and_chunk_equivalence(self):
        tokens = torch.arange(32, device="cuda")
        full = self.model.forward(tokens)
        self.model.reset()
        self.model.forward(tokens[:16])
        state = self.model.snapshot()
        chunk = self.model.forward(tokens[16:])
        self.assertLess(self.relative(chunk, full), 0.01)
        self.model.restore(state)
        replay = self.model.forward(tokens[16:])
        torch.testing.assert_close(replay, chunk, atol=0, rtol=0)

    def test_sliding_window_wrap_and_edited_prefix(self):
        tokens = torch.arange(514, device="cuda") % 128
        whole = self.model.forward(tokens)
        self.model.reset()
        for chunk in tokens.split(128):
            result = self.model.forward(chunk)
        self.assertLess(self.relative(result, whole), 0.01)
        saved = self.model.snapshot()
        tail = torch.tensor([17, 20, 21], device="cuda")
        first = self.model.forward(tail)
        self.model.restore(saved)
        again = self.model.forward(tail)
        torch.testing.assert_close(first, again, atol=0, rtol=0)

    def test_bucket_graph_scalar_parity(self):
        self.model.decode_attention_capacity = 512
        self.model.forward(torch.arange(16, device="cuda"))
        state = self.model.snapshot()
        expected = [self.model.forward(torch.tensor([i], device="cuda")).clone() for i in range(8)]
        self.model.restore(state)
        graph = GraphDecoder(self.model)
        for i, wanted in enumerate(expected):
            actual = graph(torch.tensor([i], device="cuda"))
            torch.testing.assert_close(actual, wanted, atol=0, rtol=0)

    def test_blas_grouped_assignment_order(self):
        ids = torch.tensor(
            [[1, 1] if i % 5 == 0 else [i % 7, (i * 3) % 7] for i in range(19)], device="cuda"
        )
        layout = self.model.blas_layout(ids, 8)
        for projection, assigned in [("gate", False), ("down", True)]:
            path = f"model.layers.1.mlp.switch_mlp.{projection}_proj"
            w, s, b = [
                self.model.weights[path + "." + name] for name in ("weight", "scales", "biases")
            ]
            decoded = unpack_affine(
                w.cpu().numpy(), s.float().cpu().numpy(), b.float().cpu().numpy()
            )
            weight = torch.from_numpy(decoded).to(device="cuda", dtype=torch.bfloat16)
            x = torch.randn(
                (ids.numel() if assigned else ids.shape[0], 64), device="cuda", dtype=torch.bfloat16
            )
            actual = self.model.linear(x, path, ids, layout, x_assignments=assigned)
            expected = torch.stack(
                [
                    F.linear(x[j if assigned else j // 2], weight[expert])
                    for j, expert in enumerate(ids.flatten().cpu().tolist())
                ]
            )
            self.assertLess(self.relative(actual, expected), 0.01)

    def test_top6_router_ties_bias_and_unbiased_mixture_weights(self):
        model = self.model
        model.config["num_experts_per_tok"] = 6
        controlled = np.array([-2, -1, 0, 1, 1, 2, 3, -2], dtype=np.float64)
        bias = np.zeros(8, dtype=np.float32)
        bias[7] = 0.16
        gates = []
        for layer in (1, 2):
            prefix = f"model.layers.{layer}.mlp.gate"
            gates.append(model.weights[prefix + ".weight"])
            model.weights[prefix + ".expert_bias"].copy_(torch.from_numpy(bias).cuda())
        original_linear, original_mlp, original_norm = F.linear, model.mlp, model.norm
        residuals, selected = {}, {}

        def linear(x, weight, *args, **kwargs):
            if any(weight is gate for gate in gates):
                return torch.tensor(controlled, device=x.device, dtype=x.dtype).expand(
                    x.shape[0], -1
                )
            return original_linear(x, weight, *args, **kwargs)

        def norm(x, path):
            if path.endswith(".post_attention_layernorm.weight"):
                residuals[int(path.split(".")[2])] = x.float().cpu().numpy()
            return original_norm(x, path)

        def mlp(x, path, ids=None, layout=None):
            if ids is not None:
                selected[int(path.split(".")[2])] = ids.cpu().numpy()
                return F.one_hot(ids.flatten(), num_classes=64).to(x.dtype)
            if path.endswith(".shared_expert"):
                return torch.zeros_like(x)
            return original_mlp(x, path, ids, layout)

        model.norm, model.mlp = norm, mlp
        with patch("torch.nn.functional.linear", linear):
            model.forward(torch.tensor([17], device="cuda"), trace=True)
        probabilities = 1 / (1 + np.exp(-controlled))
        wanted = np.argsort(-(probabilities + bias), kind="stable")[:6]
        mixture = probabilities[wanted] / probabilities[wanted].sum()
        mixture = torch.from_numpy(mixture).bfloat16().float().numpy()
        for layer in (1, 2):
            np.testing.assert_array_equal(selected[layer][0], wanted)
            expected = residuals[layer].copy()
            expected[0, wanted] += mixture
            torch.testing.assert_close(
                model.last_trace[layer].cpu(), torch.from_numpy(expected).bfloat16(), atol=0, rtol=0
            )


if __name__ == "__main__":
    unittest.main()
