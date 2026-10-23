from functools import partial

import mlx.core as mx
from mlx_lm.sample_utils import categorical_sampling, make_sampler

from .nucleus import MAX_TOP_K, select


def top_p_top_k(logprobs, top_p, top_k):
    """apply_top_k(apply_top_p(logprobs)) with one sort.

    apply_top_k is an argpartition (a full stable sort in MLX) of the top_p-masked values.
    The k-th largest value comes from the top_p sort instead; ties at that value keep the
    lowest indices, as argpartition does.
    """
    order = mx.argsort(logprobs, axis=-1)
    cumulative = mx.cumsum(mx.take_along_axis(mx.exp(logprobs), order, axis=-1))
    inverse = mx.put_along_axis(
        mx.zeros_like(order), order, mx.arange(order.shape[-1], dtype=order.dtype), axis=-1
    )
    cumulative = mx.take_along_axis(cumulative, inverse, axis=-1)
    masked = mx.where(cumulative > 1 - top_p, logprobs, -float("inf"))
    kth = logprobs[order[-top_k]]
    greater, tied = masked > kth, masked == kth
    slots = top_k - mx.sum(greater)
    keep = greater | (tied & (mx.cumsum(tied.astype(mx.int32)) <= slots))
    return mx.where(keep, masked, -float("inf"))


def nucleus_eligible(temp, top_p, top_k):
    return temp != 0 and 0 < top_p < 1 and 0 < top_k <= MAX_TOP_K


def make_resident_sampler(temp, top_p, top_k, flags=None):
    """make_sampler that returns the same tokens for the same key.

    With a flags list and top_k <= 32 a kernel selects the nucleus candidates instead of sorting
    the vocabulary; each step appends its status, and a nonzero status means the step must be
    redone with flags=None.
    """
    if temp == 0 or not 0 < top_p < 1 or not 0 < top_k < 150144:
        return make_sampler(temp=temp, top_p=top_p, top_k=top_k)
    if flags is None or not nucleus_eligible(temp, top_p, top_k):
        return lambda logprobs: categorical_sampling(top_p_top_k(logprobs, top_p, top_k), temp)

    def sample(logprobs):
        _, status, sampled = select(
            logprobs, mx.random.gumbel(logprobs.shape), top_p, top_k, 1 / temp
        )
        flags.append(status[0])
        return sampled[0]

    return sample


class FilteredResidentSampler:
    def __init__(self, module, engine, specification):
        self.module, self.engine = module, engine
        # After EOS, remaining graph nodes must not write KV or consume positions.
        inactive = f"""
if(!active) {{
    uint index=thread_position_in_grid.x;
    for(uint i=index;i<150144u;i+={engine.workers * 256}u)logits[i]=bfloat16_t(0.0f);
    for(uint i=index;i<50u*2048u;i+={engine.workers * 256}u)trace[i]=bfloat16_t(0.0f);
    if(thread_index_in_threadgroup==0)diagnostic[index/256u]=0;
    return;
}}
"""
        assert "configuration" not in specification["source"]
        self.kernel = module._metal(
            name="flint_filtered_resident_step",
            **dict(
                specification,
                input_names=[*specification["input_names"][:-1], "active"],
                source=inactive + specification["source"],
            ),
        )

        @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
        def run(
            metadata,
            kv,
            token,
            position,
            initial,
            capacity,
            count,
            temperature,
            top_p,
            top_k,
            presence_penalty,
            presence_mask,
            random_keys,
            grammar_handle,
            grammar_mask,
            stops,
            return_logprobs,
            fast,
        ):
            flags, flagged_steps, grammar_inputs = [], [], []
            sample = make_resident_sampler(temperature, top_p, top_k, flags if fast else None)
            active = mx.array(True)
            # Compile-time CSE must not merge independently mutated semaphore buffers.
            controls = mx.zeros((count, 1024), mx.uint32)
            tokens, masks, diagnostics, logits, traces, keys = [], [], [], [], [], []
            selected_logprobs = []
            for step in range(count):
                out = self.kernel(
                    inputs=[
                        *engine.banks,
                        metadata,
                        *kv,
                        mx.stack([token, position + step, initial, mx.array(49, mx.int32)]).astype(
                            mx.uint32
                        ),
                        engine.silu,
                        controls[step],
                        active,
                    ],
                    grid=(engine.workers * 256, 1, 1),
                    threadgroup=(256, 1, 1),
                    output_shapes=[
                        (150144,),
                        (50, 2048),
                        (engine.workers,),
                        (1,),
                        *module._scratch_layout(
                            module._LAYOUT["decode_scratch"],
                            {
                                "workspace": 78848,
                                "partials": 32 * 1024 * 128 // 2,
                                "statistics": 32 * 1024 * 2,
                            },
                            1,
                            capacity,
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
                flat = out[0].astype(mx.float32)
                if presence_penalty:
                    flat = flat - presence_penalty * presence_mask
                if grammar_handle is not None:
                    from .. import _grammar_native as grammar_native

                    # Keep mask filling parallel; order its GPU fence behind the forward.
                    grammar_inputs.append(grammar_mask)
                    ready = grammar_native.handoff(grammar_mask, out[2])
                    words = mx.arange(150144, dtype=mx.uint32)
                    allowed = ((ready[words >> 5] >> (words & 31)) & 1) != 0
                    active = active & (ready[-1] != 0)
                    flat = mx.where(allowed, flat, -float("inf"))
                    flat = mx.where(active, flat, mx.zeros_like(flat))
                # Order keyed noise after the forward when CPU grammar nodes split the graph.
                next_token = (
                    mx.argmax(
                        flat
                        + mx.random.gumbel(
                            flat.shape,
                            key=mx.depends(random_keys[step], flat)
                            if grammar_handle is not None
                            else random_keys[step],
                        )
                    )
                    if random_keys is not None
                    else sample(flat - mx.logsumexp(flat))
                ).astype(mx.int32)
                token = mx.where(active, next_token, token)
                if fast:
                    # A step the kernel cannot prove is dropped like an EOS step: later steps
                    # write nothing, the grammar never sees its token, and the caller redoes it
                    # with the sort-based sampler.
                    flagged = (flags[-1] != 0) & active
                    flagged_steps.append(flagged)
                if return_logprobs:
                    report = flat / temperature if temperature else flat
                    selected_logprobs.append(report[token] - mx.logsumexp(report))
                if grammar_handle is not None:
                    grammar_mask = grammar_native.mask(
                        token, active & ~flagged if fast else active, grammar_handle, 4692
                    )
                if presence_penalty:
                    presence_mask = presence_mask.at[token].maximum(active.astype(mx.float32))
                tokens.append(token)
                masks.append(active)
                diagnostics.append(out[2])
                logits.append(out[0])
                traces.append(out[1])
                keys.append(mx.random.state[0])
                if fast:
                    active = active & ~flagged
                active = active & ~mx.any(token == stops)
            result = tuple(
                mx.stack(rows) for rows in (tokens, masks, diagnostics, logits, traces, keys)
            )
            if presence_penalty:
                result = (*result, presence_mask)
            if return_logprobs:
                result = (*result, mx.stack(selected_logprobs))
            if fast:
                result = (*result, mx.stack(flagged_steps))
                if grammar_handle is not None:
                    result = (*result, mx.stack(grammar_inputs))
            return (*result, grammar_mask) if grammar_handle is not None else result

        self.run = run

    def redo_step(self, state, token, key, out, args, grammar):
        """Keep the tokens before the first unproven step and draw that step with the exact sampler."""
        if mx.any(out[2] != 0).item():
            state["usable"] = False
            raise RuntimeError(
                f"Filtered resident sampling synchronization failed: {out[2].tolist()}"
            )
        if grammar is not None:
            grammar.drain()
        flagged, grammar_inputs = (out[-3], out[-2]) if grammar is not None else (out[-1], None)
        step = int(mx.argmax(flagged).item())
        tokens = out[0][:step].tolist()
        # KV through this step's input token is already written and is rewritten identically.
        redo = list(args)
        redo[2:4] = [
            mx.array(tokens[-1] if tokens else token, mx.int32),
            mx.array(state["position"] + step, mx.int32),
        ]
        redo[6] = 1
        if grammar is not None:
            redo[14] = grammar_inputs[step]
        mx.random.state[0] = out[5][step - 1] if step else key
        again = self.run(*redo, False)
        mx.eval(*again)
        if mx.any(again[2] != 0).item():
            state["usable"] = False
            raise RuntimeError(
                f"Filtered resident sampling synchronization failed: {again[2].tolist()}"
            )
        if grammar is not None:
            grammar.drain()
            state["grammar_mask"] = again[-1]
        mx.random.state[0] = again[5][0]
        state["position"] += step + 1
        return [*tokens, again[0][0].item()]

    def block(
        self,
        state,
        token,
        count,
        *,
        temperature,
        top_p,
        top_k,
        presence_penalty=0.0,
        seed=0,
        grammar=None,
        logprobs=None,
    ):
        keys = None
        if (presence_penalty or grammar is not None or logprobs is not None) and (temperature, top_p, top_k) == (
            1.0,
            1.0,
            0,
        ):
            keys = mx.array(
                [[key >> 32, key & 0xFFFFFFFF] for key in range(seed, seed + count)],
                mx.uint32,
            )
        # The nucleus kernel runs only where a flagged step can be redone from the same state.
        fast = (
            keys is None
            and not presence_penalty
            and logprobs is None
            and nucleus_eligible(temperature, top_p, top_k)
        )
        key = mx.random.state[0]
        args = (
            state["metadata"],
            tuple(state["kv"].banks),
            mx.array(token, mx.int32),
            mx.array(state["position"], mx.int32),
            mx.array(state["initial"], mx.int32),
            state["capacity"],
            count,
            temperature,
            top_p,
            top_k,
            presence_penalty,
            state["presence_mask"] if presence_penalty else None,
            keys,
            grammar.handle if grammar is not None else None,
            state["grammar_mask"] if grammar is not None else None,
            mx.array(grammar.stops if grammar is not None else self.engine.model.eos_tokens),
            logprobs is not None,
        )
        out = self.run(*args, fast)
        mx.eval(*out)
        if fast and mx.any(out[-3 if grammar is not None else -1]).item():
            return self.redo_step(state, token, key, out, args, grammar)
        if grammar is not None:
            grammar.drain()
            state["grammar_mask"] = out[-1]
        if mx.any(out[2] != 0).item():
            state["usable"] = False
            raise RuntimeError(
                f"Filtered resident sampling synchronization failed: {out[2].tolist()}"
            )
        produced = int(mx.sum(out[1]).item())
        assert 1 <= produced <= count
        if logprobs is not None:
            logprobs.extend(out[7 if presence_penalty else 6][:produced].tolist())
        # Restore the exact scalar sampler's key when EOS truncates the packet.
        mx.random.state[0] = out[5][produced - 1]
        if presence_penalty:
            state["presence_mask"] = out[6]
        state["position"] += produced
        return out[0][:produced].tolist()
