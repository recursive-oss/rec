import time

import mlx.core as mx

from .resident_block import transform as resident_transform

VOCAB = 150144
WIDTH = 8


@mx.compile
def sampling_packet(keys, configuration):
    noise = mx.vmap(lambda key: mx.random.gumbel((VOCAB,), key=key))(keys)
    return mx.concatenate([configuration, noise.reshape(-1)])


def project(traces, direction):
    # Each step's trace holds the residual after the embedding and after every layer.
    return (traces.astype(mx.float32) @ direction).tolist()


class ResidentSampler:
    probe = None

    def __init__(self, module, engine, specification):
        self.module, self.engine = module, engine
        assert engine.workers == engine.device_profile.workers
        assert engine.schedule["mlp"] == engine.device_profile.mlp_workers
        assert 1 <= len(engine.model.eos_tokens) <= 4
        source = resident_transform(specification["source"], engine.workers)
        old = "float value=float(logits[word]);"
        assert source.count(old) == 1
        source = source.replace("block_step<8u", "block_step<uint(sampling[0])")
        old = "if(index==0)profile[block_step*2+1]=block_input_token;"
        assert source.count(old) == 1
        source = source.replace(
            old,
            old
            + """
if(index==0)profile[16]=block_step+1u;
bool stop=false;
for(uint j=0;j<uint(sampling[1]);++j)stop=stop||block_input_token==uint(sampling[2+j]);
if(stop)break;
""",
        )
        options = dict(
            specification,
            source=source.replace(
                "float value=float(logits[word]);",
                "float value=float(logits[word])+sampling[6u+block_step*150144u+word];",
            ),
            input_names=[*specification["input_names"][:-1], "sampling"],
            output_names=[
                "block_logits",
                "block_trace",
                *specification["output_names"][2:],
            ],
        )
        # Keep the original temperature-one source unchanged: even a uniform branch
        # can change the persistent kernel's register allocation and throughput.
        self.kernel = module._metal(name="flint_int4_temperature_one_resident8", **options)
        self.greedy_kernel = module._metal(
            name="flint_int4_greedy_resident8", **dict(options, source=source)
        )
        from .masked_sampling import FilteredResidentSampler

        self.filtered = FilteredResidentSampler(module, engine, specification)

    def block(
        self,
        state,
        token,
        count,
        seed,
        *,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        presence_penalty=0.0,
        grammar=None,
        logprobs=None,
        probe_rows=None,
    ):
        module, engine = self.module, self.engine
        if not 1 <= count <= WIDTH:
            raise ValueError("Resident packet must contain between one and eight tokens")
        engine.ensure_capacity(state, count)
        if (
            grammar is not None
            or logprobs is not None
            or presence_penalty
            or (temperature != 0 and (temperature, top_p, top_k) != (1.0, 1.0, 0))
        ):
            options = {} if logprobs is None else {"logprobs": logprobs}
            if probe_rows is not None:
                options["probe_rows"], options["probe"] = probe_rows, self.probe
            return self.filtered.block(
                state,
                token,
                count,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                seed=seed,
                grammar=grammar,
                **options,
            )
        assert 1 <= len(engine.model.eos_tokens) <= 4
        configuration = mx.array(
            [
                count,
                len(engine.model.eos_tokens),
                *engine.model.eos_tokens,
                *[0] * (4 - len(engine.model.eos_tokens)),
            ],
            dtype=mx.float32,
        )
        greedy = temperature == 0
        sampling = (
            configuration
            if greedy
            else sampling_packet(
                mx.array(
                    [[key >> 32, key & 0xFFFFFFFF] for key in range(seed, seed + count)],
                    dtype=mx.uint32,
                ),
                configuration,
            )
        )
        kernel = self.greedy_kernel if greedy else self.kernel
        out = kernel(
            inputs=[
                *engine.banks,
                state["metadata"],
                *state["kv"].banks,
                mx.array([token, state["position"], state["initial"], 49], mx.uint32),
                engine.silu,
                mx.array([0] * 1024, dtype=mx.uint32),
                sampling,
            ],
            grid=(engine.workers * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[
                (WIDTH, VOCAB),
                (WIDTH, 50, 2048),
                (engine.workers,),
                (17,),
                *module._scratch_layout(
                    module._LAYOUT["decode_scratch"],
                    {
                        "workspace": 78848,
                        "partials": 32 * 1024 * 128 // 2,
                        "statistics": 32 * 1024 * 2,
                    },
                    1,
                    state["capacity"],
                ),
            ],
            output_dtypes=[
                mx.bfloat16,
                mx.bfloat16,
                mx.uint32,
                mx.uint32,
                *[getattr(mx, dtype) for _, dtype, _ in module._LAYOUT["decode_scratch"]],
            ],
        )
        mx.eval(*out)
        diagnostic = out[2].tolist()
        if any(diagnostic):
            state["usable"] = False
            raise RuntimeError(f"Resident sampling synchronization failed: {diagnostic}")
        packet = out[3].tolist()
        produced = packet[16]
        assert 1 <= produced <= count
        pairs = packet[: 2 * produced]
        assert pairs[0] == int(token) and pairs[2::2] == pairs[1:-1:2], "Invalid sampled chain"
        if probe_rows is not None:
            probe_rows.extend(project(out[1][:produced], self.probe))
        state["position"] += produced
        return pairs[1::2]


def generate(model, sampler, prompt, maximum, seed):
    prefill_started = time.perf_counter()
    session = sampler.engine.make_session(max_new_tokens=maximum)
    logits = session.prefill(prompt)
    mx.eval(logits)
    prefill_seconds = time.perf_counter() - prefill_started
    tokens = [
        int(
            mx.random.categorical(
                logits.reshape(-1).astype(mx.float32), key=mx.random.key(seed)
            ).item()
        )
    ]
    started = time.perf_counter()
    while len(tokens) < maximum and tokens[-1] not in model.eos_tokens:
        count = min(WIDTH, maximum - len(tokens))
        tokens.extend(sampler.block(session._state, tokens[-1], count, seed + len(tokens)))
    mx.synchronize()
    seconds = time.perf_counter() - started
    return {
        "prompt_tokens": len(prompt),
        "prefill_seconds": prefill_seconds,
        "prefill_tok_s": len(prompt) / prefill_seconds,
        "token_ids": tokens,
        "decode_tokens": len(tokens) - 1,
        "decode_seconds": seconds,
        "decode_tok_s": (len(tokens) - 1) / seconds,
    }
