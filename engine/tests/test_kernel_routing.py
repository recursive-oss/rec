import unittest
from types import SimpleNamespace

import mlx.core as mx

from rec_mlx.kernels import expert_teams, megakernel, mixed_bits, rank_routing
from rec_mlx.kernels.device import profile_for_device
from rec_mlx.kernels.kv_teams import function


def resident_device():
    try:
        profile_for_device(mx.device_info())
    except ValueError:
        return False
    return True


def selected_weights(probabilities, bias):
    selected = mx.argpartition(-(probabilities + bias), kth=5)[:6]
    chosen = probabilities[selected]
    total = mx.array(0.0, mx.float32)
    for i in range(6):
        total = chosen[i] + total
    return selected, chosen / (total + 1e-20)


@unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
class RoutingTests(unittest.TestCase):
    @unittest.skipUnless(
        mx.metal.is_available() and resident_device(), "requires M3 Max, M4 Pro, M5 Pro or M5 Max"
    )
    def test_scores_survive_attention_scratch_reuse(self):
        probabilities = (mx.arange(256).astype(mx.float32) + 1) / 1024
        partial = mx.arange(256).astype(mx.float32) % 9 / 16
        partial[mx.array([7, 31, 64])] = 4
        partial[mx.array([2, 17, 93, 201])] = 3
        cases = [
            ("unbiased", mx.zeros((256,))),
            ("all_ties_with_bias", 1 - probabilities),
            ("partial_ties_with_bias", partial - probabilities),
        ]
        for expert in range(256):
            bias = (mx.arange(256).astype(mx.float32) % 13) / 128
            bias[expert] = 2
            cases.append((f"boost_{expert}", bias))

        for workers in (32, 64):
            source = """
    uint tid=thread_index_in_threadgroup,lane=thread_index_in_simdgroup;
    uint group=threadgroup_position_in_grid.x,phase=0;
    device uint* ids=routing+group*16;
    device float* scores=statistics+256;
    device atomic_uint* control=(device atomic_uint*)private_control;
    threadgroup uint ready[1];
    #define SYNC() if(!sync_all(control,tid,phase,ready,profile,group)){if(tid==0)diagnostic[group]=phase;return;}
    threadgroup uint d[10];
    const device uchar* bank=(const device uchar*)bias;
    if(tid==0){d[9]=0;diagnostic[group]=0;}
    if(tid<16)ids[tid]=0xffffffffu;
    if(group==0)statistics[tid]=probabilities[tid];
    SYNC();
    ROUTE
    // The production down-projection join publishes every leader's chosen score.
    SYNC();
    if(group==0 && tid<6)statistics[256+tid]=NAN;
    if(tid==0)routing[group*16+14]=123u;
    SYNC();
            float sums[8];
    uint r=lane;
    if(simdgroup_index_in_threadgroup==0 && lane<6)
        actual[group*6+lane]=1.0f*scores[r]+0.0f*scores[0];
""".replace("ROUTE", rank_routing.ROUTE_CALL)
            sync = function(megakernel._HEADER, "sync_all")
            sync = sync.replace("__WORKERS__", str(workers)).replace("__PROFILE__", "0")
            header, source = rank_routing.transform(sync, source, workers)
            kernel = megakernel._metal(
                name="routing_scores_survive_attention_reuse",
                input_names=["bias", "probabilities", "private_control"],
                output_names=["routing", "statistics", "actual", "diagnostic", "profile"],
                header=header,
                source=source,
                compile_options={"math_mode": "safe"},
            )
            for case, bias in cases:
                with self.subTest(mlp_workers=workers, case=case):
                    selected, expected = selected_weights(probabilities, bias)
                    routing, _, actual, diagnostic, _ = kernel(
                        inputs=[bias, probabilities, mx.zeros((1024,), mx.uint32)],
                        output_shapes=[(workers, 16), (262,), (workers, 6), (workers,), (1,)],
                        output_dtypes=[mx.uint32, mx.float32, mx.float32, mx.uint32, mx.uint32],
                        grid=(workers * 256, 1, 1),
                        threadgroup=(256, 1, 1),
                    )
                    self.assertEqual(diagnostic.tolist(), [0] * workers)
                    self.assertTrue(mx.all(mx.isfinite(actual)).item())
                    self.assertLess(float(mx.max(mx.abs(actual - expected)).item()), 1e-7)
                    ids = routing[:, :6].tolist()
                    selected_ids = selected.tolist()
                    for group in range(workers):
                        expected_ids = [0xFFFFFFFF] * 6
                        if group < workers * 3 // 4:
                            role = group // (workers // 8)
                            expected_ids[role] = selected_ids[role]
                        self.assertEqual(ids[group], expected_ids, group)
                    self.assertEqual(routing[:, 14].tolist(), [123] * workers)
                    self.assertEqual(routing[:, 15].tolist(), [0xFFFFFFFF] * workers)

    def test_native_bf16_expert_reduction(self):
        for workers in (32, 64):
            for bits, specialized in ((4, False), (3, False), (3, True)):
                for ranked in (False, True):
                    with self.subTest(
                        workers=workers, bits=bits, specialized=specialized, ranked=ranked
                    ):
                        self.check_reduction(workers, bits, specialized, ranked)

    def check_reduction(self, workers, bits, specialized, ranked):
        model = SimpleNamespace(
            embedding=SimpleNamespace(bits=8),
            head=SimpleNamespace(bits=4),
            config={"mlx_export": {"routed_expert_bits": bits}},
        )
        header = expert_teams.transform(megakernel._HEADER, workers)
        body = megakernel._BODY
        if ranked:
            header, body = rank_routing.transform(header, body, workers)
        _, body = mixed_bits.transform(header, body, model, workers, specialized)
        start = body.index(
            "            float w3_chosen=" if ranked else "            float sums[8];"
        )
        end = body.index("        } else {", start)
        source = """
            uint tid=thread_index_in_threadgroup,lane=thread_index_in_simdgroup;
            uint group=0, layer=0;
            device uint* routing=(device uint*)scores;
            bfloat16_t h[8];
            for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;h[j]=residual[i];}
        """ + body[start:end]
        kernel = megakernel._metal(
            name="qat_native_bf16_expert_mix",
            input_names=["down", "scores", "residual"],
            output_names=["trace"],
            source=source,
            compile_options={"math_mode": "safe"},
        )
        for seed in (19, 31, 47):
            mx.random.seed(seed)
            down = mx.random.normal((7, 2048)).astype(mx.bfloat16)
            residual = mx.random.normal((2048,)).astype(mx.bfloat16)
            probabilities = mx.random.uniform(shape=(256,))
            bias = mx.random.uniform(low=-0.5, high=0.5, shape=(256,))
            selected, weights = selected_weights(probabilities, bias)
            products = down[:6] * weights.astype(mx.bfloat16)[:, None]
            expected = (products.sum(axis=0) + down[6]) + residual
            fp32_mix = (
                products.astype(mx.float32).sum(axis=0).astype(mx.bfloat16) + down[6]
            ) + residual
            self.assertFalse(mx.array_equal(expected, fp32_mix).item())
            scores = (
                mx.concatenate([mx.zeros((8,)), probabilities[selected]])
                if ranked
                else mx.concatenate([weights, mx.zeros((8,))])
            )
            (trace,) = kernel(
                inputs=[down, scores, residual],
                output_shapes=[(4096,)],
                output_dtypes=[mx.bfloat16],
                grid=(256, 1, 1),
                threadgroup=(256, 1, 1),
            )
            self.assertTrue(mx.array_equal(trace[2048:], expected).item(), seed)


if __name__ == "__main__":
    unittest.main()
