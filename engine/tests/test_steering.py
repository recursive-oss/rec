import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

import mlx.core as mx
from mlx.utils import tree_flatten

from rec_mlx.cli import main
from rec_mlx.kernels import megakernel
from rec_mlx.kernels import steering as decode_steering
from rec_mlx.model import Model, ModelArgs
from rec_mlx.steering import Steering, attach, load

SLOTS = 99


def unit(size=2048, seed=0):
    r = mx.random.normal((size,), key=mx.random.key(seed))
    return r / mx.linalg.norm(r)


def steering(direction, **overrides):
    slots = overrides.pop("slots", SLOTS)
    values = {"strength": 1.0, "low": 0.0, "high": 0.0, "shift": 0.0, **overrides}
    return Steering(
        direction=direction,
        **{
            name: tuple(value) if isinstance(value, list) else (value,) * slots
            for name, value in values.items()
        },
    )


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name, **tensors):
        path = self.root / name
        if name.endswith(".npz"):
            mx.savez(str(path), **tensors)
        else:
            mx.save_safetensors(str(path), tensors)
        return path

    def test_direction_alone_projects_at_every_slot(self):
        path = self.save("r.safetensors", direction=unit() * 3)
        loaded = load(path, hidden_size=2048, layers=49)
        self.assertEqual(loaded.slots, SLOTS)
        self.assertAlmostEqual(float(mx.linalg.norm(loaded.direction).item()), 1.0, places=5)
        self.assertEqual(set(loaded.strength), {1.0})
        self.assertEqual((set(loaded.low), set(loaded.high), set(loaded.shift)), ({0.0},) * 3)
        self.assertEqual(loaded.summary()["active_slots"], SLOTS)
        self.assertEqual(len(loaded.sha256), 64)

    def test_per_slot_settings_and_strength_multiplier(self):
        strength = mx.zeros((SLOTS,))
        strength[10] = 0.5
        high = mx.full((SLOTS,), float("inf"))
        path = self.save("r.npz", direction=unit(), strength=strength, low=-high, high=high)
        loaded = load(path, hidden_size=2048, layers=49, strength=2.0)
        self.assertEqual(loaded.strength[10], 1.0)
        self.assertEqual(loaded.summary()["active_slots"], 1)
        self.assertEqual(loaded.low[0], float("-inf"))

    def test_invalid_files_are_rejected(self):
        cases = {
            "missing": ({"strength": mx.ones((SLOTS,))}, "needs a direction"),
            "width": ({"direction": unit(1024)}, "2048 values"),
            "zero": ({"direction": mx.zeros((2048,))}, "nonzero"),
            "slots": ({"direction": unit(), "low": mx.zeros((98,))}, "99 values"),
            "unknown": ({"direction": unit(), "scale": mx.ones((SLOTS,))}, "Unknown"),
            "order": (
                {"direction": unit(), "low": mx.ones((SLOTS,)), "high": mx.zeros((SLOTS,))},
                "low <= high",
            ),
            "strength": (
                {"direction": unit(), "strength": mx.full((SLOTS,), float("nan"))},
                "finite",
            ),
        }
        for name, (tensors, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                load(self.save(f"{name}.safetensors", **tensors), hidden_size=2048, layers=49)


class ReferenceTests(unittest.TestCase):
    def test_modes_move_the_component_toward_its_target(self):
        r = unit()
        h = (mx.random.normal((3, 2048), key=mx.random.key(1)) + 4 * r).astype(mx.bfloat16)
        d = h.astype(mx.float32) @ r
        for name, settings, expected in (
            ("project", {}, mx.zeros_like(d)),
            ("half", {"strength": 0.5}, d / 2),
            ("cap", {"low": float("-inf"), "high": 1.0}, mx.minimum(d, 1.0)),
            ("shift", {"low": float("-inf"), "high": float("inf"), "shift": 2.0}, d + 2.0),
        ):
            plan = steering(r, **settings)
            for path, apply in (("reference", plan.reference), ("call", plan)):
                with self.subTest(name, path=path):
                    out = apply(h, 5)
                    self.assertEqual(out.dtype, mx.bfloat16)
                    actual = out.astype(mx.float32) @ r
                    self.assertTrue(mx.allclose(actual, expected, atol=0.08).item())

    def test_inactive_slot_returns_the_input(self):
        h = mx.random.normal((2, 2048)).astype(mx.bfloat16)
        self.assertIs(steering(unit(), strength=0.0)(h, 3), h)


def tiny_model():
    args = ModelArgs(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=97,
        sliding_window=8,
        mlp_only_layers=[0],
    )
    mx.random.seed(3)
    model = Model(args)
    mx.eval(model.parameters())
    return model


class PrefillTests(unittest.TestCase):
    def test_inactive_steering_is_bit_identical(self):
        model = tiny_model()
        tokens = mx.array([[3, 14, 15, 92, 65]])
        reference = model(tokens)
        attach(model, steering(unit(64), strength=0.0, slots=7))
        self.assertTrue(mx.array_equal(model(tokens), reference).item())

    def test_each_slot_is_applied_once_in_order(self):
        model = tiny_model()
        seen = []

        class Recorder:
            def __call__(self, h, slot):
                seen.append(slot)
                return h

        attach(model, Recorder())
        model(mx.array([[1, 2, 3]]))
        self.assertEqual(seen, list(range(7)))
        names = [name for name, _ in tree_flatten(model.parameters())]
        self.assertFalse([name for name in names if "steering" in name or "slot" in name])

    def test_projection_changes_the_output(self):
        model = tiny_model()
        tokens = mx.array([[5, 6, 7, 8]])
        reference = model(tokens)
        attach(model, steering(unit(64, seed=4), slots=7))
        self.assertFalse(mx.allclose(model(tokens), reference).item())


class DecodeTransformTests(unittest.TestCase):
    def test_every_residual_update_is_steered_once(self):
        header, body = decode_steering.transform(
            megakernel._HEADER, megakernel._BODY, steering(unit())
        )
        self.assertEqual(body.count("steer_residual(h,1u+2u*layer,"), 1)
        self.assertEqual(body.count("steer_traced(h,0u,"), 1)
        self.assertEqual(body.count("steer_traced(h,2u+2u*layer,"), 2)
        self.assertIn("constant float STEER_DIRECTION[2048]", header)
        self.assertIn("-INFINITY", decode_steering.header(steering(unit(), low=float("-inf"))))

    def test_changed_kernel_text_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "Expected 1 anchors"):
            decode_steering.transform(megakernel._HEADER, "", steering(unit()))
        with self.assertRaisesRegex(ValueError, "2048-wide"):
            decode_steering.transform(megakernel._HEADER, megakernel._BODY, steering(unit(64)))


@unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
class KernelTests(unittest.TestCase):
    def run_kernel(self, plan, h, slots):
        kernel = megakernel._metal(
            name="steering_matches_reference",
            input_names=["x", "slots"],
            output_names=["out", "trace"],
            header=decode_steering.header(plan),
            source=r"""
    uint tid=thread_index_in_threadgroup,lane=thread_index_in_simdgroup;
    uint group=threadgroup_position_in_grid.x;
    threadgroup float steer_scratch[8];
    bfloat16_t h[8];
    for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;h[j]=x[group*2048+i];}
    steer_traced(h,slots[group],tid,lane,simdgroup_index_in_threadgroup,steer_scratch,
                 trace+group*2048,0);
    for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;out[group*2048+i]=h[j];}
""",
            compile_options={"math_mode": "safe"},
        )
        rows = h.shape[0]
        out, trace = kernel(
            inputs=[h, mx.array(slots, dtype=mx.uint32)],
            grid=(256 * rows, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[h.shape, h.shape],
            output_dtypes=[mx.bfloat16, mx.bfloat16],
            init_value=0,
        )
        mx.eval(out, trace)
        return out, trace

    def test_decode_matches_prefill_bits_and_the_reference(self):
        r = unit(seed=7)
        strength = [0.0] * SLOTS
        low, high, shift = [0.0] * SLOTS, [0.0] * SLOTS, [0.0] * SLOTS
        strength[1], strength[2], strength[3], strength[4] = 1.0, 0.5, 1.0, 1.0
        low[3], high[3] = float("-inf"), 1.5
        low[4], high[4], shift[4] = float("-inf"), float("inf"), -3.0
        plan = Steering(
            direction=r,
            strength=tuple(strength),
            low=tuple(low),
            high=tuple(high),
            shift=tuple(shift),
        )
        slots = [0, 1, 2, 3, 4]
        h = (mx.random.normal((5, 2048), key=mx.random.key(8)) * 2 + 5 * r).astype(mx.bfloat16)
        out, trace = self.run_kernel(plan, h, slots)
        prefill = mx.stack([plan(h[i], slot) for i, slot in enumerate(slots)])
        self.assertTrue(mx.array_equal(out, prefill).item(), "decode and prefill must agree")
        self.assertTrue(
            mx.array_equal(plan(h, 3), mx.stack([plan(row, 3) for row in h])).item(),
            "rows are steered independently",
        )
        expected = mx.stack([plan.reference(h[i], slot) for i, slot in enumerate(slots)])
        a, b = out.astype(mx.float32), expected.astype(mx.float32)
        # A different summation order may move a value by one BF16 step.
        tolerance = mx.maximum(mx.abs(a), mx.abs(b)) * 2**-7 + 1e-6
        self.assertTrue(mx.all(mx.abs(a - b) <= tolerance).item())
        self.assertLess(mx.mean((a != b).astype(mx.float32)).item(), 0.01)
        self.assertTrue(mx.array_equal(out[0], h[0]).item(), "an inactive slot must not change h")
        self.assertTrue(mx.array_equal(trace[1:], out[1:]).item(), "active slots rewrite the trace")
        self.assertTrue(mx.array_equal(trace[0], mx.zeros_like(h[0])).item())


class BuildTests(unittest.TestCase):
    def test_recovers_a_planted_direction_and_its_band(self):
        from rec_mlx.steering_build import contrast, steering_file

        r = unit(64, seed=11)
        noise = mx.random.normal((2, 20, 7, 64), key=mx.random.key(12)) * 0.1
        planted = mx.zeros((7,)).at[mx.array([3, 4, 5])].add(mx.array([2.0, 3.0, 2.0]))
        positive = noise[0] + planted[None, :, None] * r
        negative = noise[1]
        separation, direction, best = contrast(positive, negative)
        self.assertEqual(best, 4)
        self.assertGreater(abs(float((direction @ r).item())), 0.99)
        tensors, report = steering_file(positive, negative, direction, separation, 2.0, best)
        self.assertEqual(report["band"], [4])
        self.assertEqual(tensors["strength"].tolist(), [0, 0, 0, 0, 1, 0, 0])
        sign = 1 if float((direction @ r).item()) > 0 else -1
        self.assertAlmostEqual(float(tensors["shift"][4].item()), sign * 6.0, delta=0.2)


class ProbeTests(unittest.TestCase):
    def test_summary_reports_layers_and_the_chosen_series(self):
        from rec_mlx.engine import Engine

        engine = Engine.__new__(Engine)
        engine.probe_layer = 1
        summary = engine._probe_summary([[1.0, -2.0, 0.5], [3.0, 0.0, -0.5]])
        self.assertEqual(summary["tokens"], 2)
        self.assertEqual(summary["layer_mean"], [2.0, -1.0, 0.0])
        self.assertEqual(summary["layer_max"], [3.0, 0.0, 0.5])
        self.assertEqual(summary["per_token"], [-2.0, 0.0])
        engine.probe_layer = None
        self.assertNotIn("per_token", engine._probe_summary([[1.0]]))

    def test_probe_layer_requires_a_probe_file(self):
        argv = ["rec-mlx", "chat", "flint", "--model-dir", ".", "--manifest", "m.json"]
        with (
            patch("sys.argv", [*argv, "--probe-layer", "3"]),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            main()


class CommandLineTests(unittest.TestCase):
    def test_flags_reach_the_engine(self):
        engine = Mock(identity={"model_id": "flint"}, startup={})
        base = ["rec-mlx", "chat", "flint", "--model-dir", ".", "--manifest", "m.json"]
        with (
            patch("sys.argv", [*base, "--steering-file", "r.npz", "--steering-strength", "0.5"]),
            patch.dict(os.environ, {}),
            patch("rec_mlx.cli.Engine", return_value=engine) as construct,
            patch("rec_mlx.cli.chat"),
            redirect_stderr(io.StringIO()),
            patch("sys.stdout", io.StringIO()),
        ):
            main()
        self.assertEqual(construct.call_args.kwargs["steering_file"], Path("r.npz"))
        self.assertEqual(construct.call_args.kwargs["steering_strength"], 0.5)
        with (
            patch("sys.argv", [*base, "--steering-strength", "0.5"]),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            main()


if __name__ == "__main__":
    unittest.main()
