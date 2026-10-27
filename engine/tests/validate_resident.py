"""Compare resident decoding with scalar and native MLX decoding on the Flint checkpoint."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from transformers import AutoTokenizer

from rec_mlx.engine import Engine, Sampling
from rec_mlx.kernels import build_decoder, megakernel
from rec_mlx.kernels.masked_sampling import FilteredResidentSampler
from rec_mlx.quantized import QATModel


def validate_stopping(engine, resident, specification, tokenizer, prompt):
    model = engine.model
    original_eos, original_filtered = model.eos_tokens, resident.filtered
    eos_cases, cancellation_cases = [], []
    modes = ((0, 1, 0), (1, 1, 0), (0.6, 0.95, 20))
    try:
        for temperature, top_p, top_k in modes:
            settings = dict(temperature=temperature, top_p=top_p, top_k=top_k)
            sample = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)

            def pick(logits, index):
                flat = logits.reshape(-1).astype(mx.float32)
                if temperature == 0:
                    return int(mx.argmax(flat).item())
                if (temperature, top_p, top_k) == (1, 1, 0):
                    return int(mx.random.categorical(flat, key=mx.random.key(42 + index)).item())
                return int(sample(flat - mx.logsumexp(flat)).item())

            discovery = engine.make_session(max_new_tokens=33)
            logits = discovery.prefill(prompt)
            mx.random.seed(42)
            tokens, keys = [pick(logits, 0)], [mx.random.state[0]]
            for index in range(32):
                tokens.append(pick(discovery.forward_token(tokens[-1]), index + 1))
                keys.append(mx.random.state[0])
            del discovery
            starts = [
                begin
                for begin in range(25)
                if all(tokens[begin + stop] not in tokens[: begin + stop] for stop in (1, 4, 8))
            ]
            assert starts, "Prompt did not produce unique early/middle/final EOS fixtures"
            begin = starts[0]
            for stop in (1, 4, 8):
                reference = engine.make_session(max_new_tokens=33)
                candidate = engine.make_session(max_new_tokens=33)
                reference.prefill(prompt)
                candidate.prefill(prompt)
                for token in tokens[:begin]:
                    reference.forward_token(token)
                    candidate.forward_token(token)
                hi, lo = keys[begin].tolist()
                mx.random.seed((int(hi) << 32) | int(lo))
                expected = [
                    pick(reference.forward_token(tokens[begin + i]), begin + i + 1)
                    for i in range(stop)
                ]
                expected_key = mx.random.state[0]
                model.eos_tokens = [expected[-1]]
                # Each compiled graph must capture this fixture's EOS, not a previous one.
                resident.filtered = FilteredResidentSampler(resident.module, engine, specification)
                hi, lo = keys[begin].tolist()
                mx.random.seed((int(hi) << 32) | int(lo))
                actual = resident.block(candidate._state, tokens[begin], 8, 43 + begin, **settings)
                assert actual == expected and len(actual) == stop, "EOS truncation mismatch"
                assert mx.array_equal(mx.random.state[0], expected_key).item(), "EOS RNG mismatch"
                assert candidate._state["position"] == reference._state["position"]
                assert same_kv(candidate._state["kv"], reference._state["kv"]), (
                    "EOS wrote beyond the retained prefix"
                )
                eos_cases.append(dict(**settings, stop=stop, packet_start=begin))
                del reference, candidate
    finally:
        model.eos_tokens, resident.filtered = original_eos, original_filtered

    api = Engine.__new__(Engine)
    api._prefix_preparation = None
    api.mx, api.runtime, api.resident_sampler = mx, engine, resident
    api.tokenizer, api.context_limit, api.eos = tokenizer, model.context_limit, set(original_eos)
    mx.eval(model.native.parameters())
    for temperature, top_p, top_k in modes:
        api._prompt_cache = None
        sampling = Sampling(
            temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=17, seed=42
        )
        emitted = []
        for event in api.generate_tokens(prompt, sampling, should_stop=lambda: len(emitted) == 2):
            if "token" in event:
                emitted.append(event["token"])
            else:
                cancelled = event
        assert cancelled["finish_reason"] == "stop" and len(emitted) == 2
        assert cancelled["metrics"]["computed_tokens"] == 9, "Cancellation did not discard a packet"
        cached = list(api.generate_tokens(prompt, sampling))[-1]["metrics"]
        api._prompt_cache = None
        cold = list(api.generate_tokens(prompt, sampling))[-1]["metrics"]
        assert cached["cached_prompt_tokens"] == len(prompt) and cold["cached_prompt_tokens"] == 0
        assert cached["token_ids"] == cold["token_ids"], (
            "Cancelled packet contaminated prefix reuse"
        )
        cancellation_cases.append(dict(temperature=temperature, top_p=top_p, top_k=top_k))
    return dict(eos=eos_cases, cancellation=cancellation_cases)


def same_kv(a, b):
    return all(mx.array_equal(x, y).item() for x, y in zip(a.banks, b.banks, strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Flint checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the results")
    args = parser.parse_args()
    mx.set_cache_limit(256 * 1024**2)
    model = QATModel(args.model)
    specification = {}
    metal = megakernel._metal

    def capture_specification(name, **options):
        if name == "flint_decode_fast_activation":
            specification.update(options)
        return metal(name=name, **options)

    megakernel._metal = capture_specification
    engine, resident = build_decoder(model, optimized=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, fix_mistral_regex=False
    )
    text = "Write a complete Python program using openpyxl to create an Excel sales report, with sample data, formulas, formatting and a bar chart. Explain and test the code."
    prompt = tokenizer.apply_chat_template(
        [dict(role="user", content=text)],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    captured = {}
    persistent_kernel, greedy_kernel, filtered_run = (
        resident.kernel,
        resident.greedy_kernel,
        resident.filtered.run,
    )

    def persistent(**kwargs):
        out = persistent_kernel(**kwargs)
        captured["logits"], captured["trace"] = out[:2]
        return out

    def filtered(*a):
        out = filtered_run(*a)
        captured["logits"], captured["trace"] = out[3:5]
        return out

    def greedy(**kwargs):
        out = greedy_kernel(**kwargs)
        captured["logits"], captured["trace"] = out[:2]
        return out

    resident.kernel, resident.greedy_kernel, resident.filtered.run = persistent, greedy, filtered
    settings = [
        (0, 0.95, 20, 0),
        (0.05, 1, 0, 0),
        (0.6, 1, 0, 0),
        (1, 1, 0, 0),
        (1.7, 1, 0, 0),
        (2, 1, 0, 0),
        (1, 0.95, 0, 0),
        (0.6, 0.95, 20, 0),
        (1, 1, 1, 0),
        (2, 0.5, 149999, 0),
        (1, 0.01, 0, 0),
        (0, 1, 0, 1.5),
        (1, 1, 0, 1.5),
        (1, 0.95, 20, 1.5),
        (1, 0.95, 20, -1.5),
    ]
    results = []
    for context in (len(prompt), 513, 8192):
        ids = (prompt * ((context + len(prompt) - 1) // len(prompt)))[:context]
        for temperature, top_p, top_k, penalty in settings if context != 8192 else settings[:1]:
            for seed in (42, 73) if context != 8192 else (42,):
                steps = 520 if temperature == 0 and context == len(prompt) and seed == 42 else 16
                reference = engine.make_session(max_new_tokens=steps + 1)
                candidate = engine.make_session(max_new_tokens=steps + 1)
                a, b = reference.prefill(ids), candidate.prefill(ids)
                assert mx.array_equal(a, b).item(), "Prefill logits differ"
                assert same_kv(reference._state["kv"], candidate._state["kv"])
                sample = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
                unfiltered_t1 = (temperature, top_p, top_k) == (1, 1, 0)

                seen = set()

                def choose(logits, index):
                    flat = logits.reshape(-1).astype(mx.float32)
                    if penalty and seen:
                        flat[mx.array(list(seen), mx.int32)] -= penalty
                    if temperature == 0:
                        return int(mx.argmax(flat).item())
                    if unfiltered_t1:
                        noise = mx.random.gumbel(flat.shape, key=mx.random.key(seed + index))
                        return int(mx.argmax(flat + noise).item())
                    return int(sample(flat - mx.logsumexp(flat)).item())

                mx.random.seed(seed)
                pending = choose(a, 0)
                seen.add(pending)
                if penalty:
                    candidate._state["presence_mask"] = mx.zeros(a.size, mx.float32)
                    candidate._state["presence_mask"][pending] = 1.0
                scalar_key, packet_key = mx.random.state[0], mx.random.state[0]
                # Shared global KV is mutated by decode; use a separate native reference.
                native_session = engine.make_session(max_new_tokens=steps + 1)
                native_session.prefill(ids)
                native_cache = native_session.prompt_cache.layers
                for row, cache in zip(engine.metadata[:49], native_cache, strict=True):
                    if not row[1]:
                        cache.grow(context + steps + 1)
                native_checks = 0
                position, packets = 0, 0
                while position < steps and pending not in model.eos_tokens:
                    count = min((1, 3, 8)[packets % 3], steps - position)
                    packet_input = pending
                    expected, logits, traces = [], [], []
                    hi, lo = scalar_key.tolist()
                    mx.random.seed((int(hi) << 32) | int(lo))
                    for offset in range(count):
                        value, trace = engine.forward(reference._state, pending)
                        if temperature == 0 and position + offset < 8:
                            h = model.native.model.embed_tokens(mx.array([[pending]], mx.int32))
                            native_trace = [h.reshape(-1)]
                            for layer, cache in zip(model.native.layers, native_cache, strict=True):
                                h = layer(h, None, cache)
                                native_trace.append(h.reshape(-1))
                            native_logits = model.native.lm_head(model.native.model.norm(h))
                            assert mx.array_equal(mx.stack(native_trace), trace).item(), (
                                "Native layer mismatch"
                            )
                            assert mx.array_equal(native_logits.reshape(-1), value).item(), (
                                "Native logits mismatch"
                            )
                            native_checks += 1
                        pending = choose(value, position + offset + 1)
                        seen.add(pending)
                        expected.append(pending)
                        logits.append(value)
                        traces.append(trace)
                        if pending in model.eos_tokens:
                            break
                    scalar_key = mx.random.state[0]
                    hi, lo = packet_key.tolist()
                    mx.random.seed((int(hi) << 32) | int(lo))
                    actual = resident.block(
                        candidate._state,
                        packet_input,
                        count,
                        seed + position + 1,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        presence_penalty=penalty,
                    )
                    packet_key = mx.random.state[0]
                    assert actual == expected, (
                        context,
                        temperature,
                        top_p,
                        top_k,
                        position,
                        actual,
                        expected,
                    )
                    assert mx.array_equal(packet_key, scalar_key).item(), "Sampler key mismatch"
                    for offset in range(len(actual)):
                        assert mx.array_equal(captured["logits"][offset], logits[offset]).item(), (
                            "Packet logits mismatch"
                        )
                        assert mx.array_equal(captured["trace"][offset], traces[offset]).item(), (
                            "Packet layer mismatch"
                        )
                    if penalty:
                        mask = mx.zeros(a.size, mx.float32)
                        mask[mx.array(list(seen), mx.int32)] = 1
                        assert mx.array_equal(candidate._state["presence_mask"], mask).item()
                    assert reference._state["position"] == candidate._state["position"]
                    assert same_kv(reference._state["kv"], candidate._state["kv"]), "KV mismatch"
                    position += len(actual)
                    packets += 1
                row = dict(
                    context=context,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    presence_penalty=penalty,
                    seed=seed,
                    positions=position,
                    packets=packets,
                    native_checks=native_checks,
                )
                results.append(row)
                print(json.dumps(row), flush=True)
                del reference, candidate, native_session, native_cache, a, b
                mx.clear_cache()

    stopping = validate_stopping(engine, resident, specification, tokenizer, prompt)
    print(json.dumps(dict(stopping=stopping)), flush=True)
    performance = []
    ids = prompt
    for temperature, top_p, top_k, penalty in (
        settings[0],
        settings[2],
        settings[3],
        settings[7],
        (1, 0.95, 20, 0),
        (1, 0.95, 20, 1.5),
    ):
        sample = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        timings = {"scalar": [], "packet": []}
        for repeat in range(4):
            for variant in ("scalar", "packet") if repeat % 2 == 0 else ("packet", "scalar"):
                session = engine.make_session(max_new_tokens=257)
                logits = session.prefill(ids)
                mx.random.seed(42)
                if temperature == 0:
                    token = int(mx.argmax(logits).item())
                elif (temperature, top_p, top_k) == (1, 1, 0):
                    token = int(
                        mx.random.categorical(
                            logits.astype(mx.float32), key=mx.random.key(42)
                        ).item()
                    )
                else:
                    flat = logits.reshape(-1).astype(mx.float32)
                    token = int(sample(flat - mx.logsumexp(flat)).item())
                seen = {token}
                if penalty:
                    session._state["presence_mask"] = mx.zeros(logits.size, mx.float32)
                    session._state["presence_mask"][token] = 1.0
                count = 0
                mx.synchronize()
                started = time.perf_counter()
                while count < 256 and token not in model.eos_tokens:
                    if variant == "packet":
                        tokens = resident.block(
                            session._state,
                            token,
                            min(8, 256 - count),
                            43 + count,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            presence_penalty=penalty,
                        )
                        count += len(tokens)
                        token = tokens[-1]
                    else:
                        value = session.forward_token(token).reshape(-1).astype(mx.float32)
                        if penalty:
                            value[mx.array(list(seen), mx.int32)] -= penalty
                        if temperature == 0:
                            token = int(mx.argmax(value).item())
                        elif (temperature, top_p, top_k) == (1, 1, 0):
                            token = int(
                                mx.random.categorical(value, key=mx.random.key(43 + count)).item()
                            )
                        else:
                            token = int(sample(value - mx.logsumexp(value)).item())
                        seen.add(token)
                        count += 1
                mx.synchronize()
                speed = count / (time.perf_counter() - started)
                if repeat:
                    timings[variant].append(speed)
                del session
        row = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=penalty,
            **{name: statistics.median(values) for name, values in timings.items()},
        )
        performance.append(row)
        print(json.dumps(dict(performance=row)), flush=True)
    result = dict(
        device=mx.device_info(),
        config_sha256=hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        cases=results,
        stopping=stopping,
        performance=performance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
