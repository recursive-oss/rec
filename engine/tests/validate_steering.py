"""Check steering on the Flint checkpoint: decode/prefill agreement, speed and a coding task."""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx

from rec_mlx.engine import Engine, Sampling

PROMPTS = [
    "Explain how a hash map handles collisions.",
    "Write a short poem about the sea.",
]
CODE = (
    "Write a Python function is_prime(n) that returns True for primes and False otherwise. "
    "Reply with only the code in one ```python block."
)
CODE_TEST = (
    "assert [n for n in range(-3, 30) if is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]\n"
    "assert is_prime(7919) and not is_prime(7917)\n"
)


def resident_tokens(engine, ids, sampling):
    events = list(engine.generate_tokens(ids, sampling))
    return [e["token"] for e in events if "token" in e], events[-1]["metrics"]


def native_steps_match(engine, prompt, steps=8):
    # As in validate_resident.py: each resident step must equal the native layers bit for bit,
    # with the steered residual in the trace.
    runtime, native = engine.runtime, engine.runtime.model.native
    ids = engine.prompt([{"role": "user", "content": prompt}])
    reference = runtime.make_session(max_new_tokens=steps + 1)
    logits = reference.prefill(ids)
    mirror = runtime.make_session(max_new_tokens=steps + 1)
    mirror.prefill(ids)
    caches = mirror.prompt_cache.layers
    for row, cache in zip(runtime.metadata[:49], caches, strict=True):
        if not row[1]:
            cache.grow(len(ids) + steps + 1)
    token, matches = int(mx.argmax(logits.reshape(-1)).item()), 0
    for _ in range(steps):
        value, trace = runtime.forward(reference._state, token)
        h = native.model.embed_tokens(mx.array([[token]], mx.int32))
        if native.model.steering is not None:
            h = native.model.steering(h, 0)
        rows = [h.reshape(-1)]
        for layer, cache in zip(native.layers, caches, strict=True):
            h = layer(h, None, cache)
            rows.append(h.reshape(-1))
        out = native.lm_head(native.model.norm(h)).reshape(-1)
        matches += (
            mx.array_equal(mx.stack(rows), trace).item() and mx.array_equal(out, value).item()
        )
        token = int(mx.argmax(value.reshape(-1)).item())
    return {"prompt": prompt, "bit_identical_steps": matches, "steps": steps}


def probe_matches(engine, prompt, tokens=48):
    # Probing reads the trace the decoder already writes, so tokens must not change. Row i belongs
    # to the step whose input was token i, so its slot-0 value is that token's embedding projection.
    sampler, native = engine.resident_sampler, engine.runtime.model.native
    direction, sampler.probe = sampler.probe, None
    ids = engine.prompt([{"role": "user", "content": prompt}])
    sampling = Sampling(temperature=0.0, top_p=1.0, top_k=0, max_tokens=tokens)
    engine._prompt_cache = None
    plain, _ = resident_tokens(engine, ids, sampling)
    sampler.probe = direction
    engine._prompt_cache = None
    events = list(engine.generate_tokens(ids, sampling))
    probed = [e["token"] for e in events if "token" in e]
    rows = [e["probe"] for e in events if "token" in e][1:]
    embedded = native.model.embed_tokens(mx.array([probed[:-1]], mx.int32))[0]
    expected = (embedded.astype(mx.float32) @ direction).tolist()
    return {
        "tokens_identical": plain == probed,
        "rows": len(rows),
        "row_width": len(rows[0]),
        "slot0_max_error": max(abs(row[0] - e) for row, e in zip(rows, expected, strict=True)),
        "summary": events[-1]["metrics"]["probe"]["layer_mean"][:5],
    }


def reply(engine, prompt, max_tokens):
    sampling = Sampling(temperature=0.0, top_p=1.0, top_k=0, max_tokens=max_tokens)
    messages = [{"role": "user", "content": prompt}]
    return "".join(event.get("text", "") for event in engine.generate(messages, sampling))


def code_passes(answer):
    match = re.search(r"```python\n(.*?)```", answer, re.S)
    if match is None:
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".py") as script:
        script.write(match.group(1) + "\n" + CODE_TEST)
        script.flush()
        return subprocess.run([sys.executable, script.name], capture_output=True).returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Flint checkpoint directory")
    parser.add_argument("--manifest", type=Path, required=True, help="checkpoint manifest")
    parser.add_argument("--steering-file", type=Path, help="steering file (omit for the baseline)")
    parser.add_argument("--steering-strength", type=float, default=1.0)
    parser.add_argument("--probe-file", type=Path, help="also check the projection probe")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the results")
    args = parser.parse_args()
    engine = Engine(
        args.model,
        args.manifest,
        context_limit=8192,
        steering_file=args.steering_file,
        steering_strength=args.steering_strength,
        probe_file=args.probe_file,
    )
    agreement = [native_steps_match(engine, prompt) for prompt in PROMPTS]
    probe = (
        [probe_matches(engine, prompt) for prompt in PROMPTS]
        if args.probe_file is not None
        else None
    )
    ids = engine.prompt(
        [{"role": "user", "content": "Write a long story about a lighthouse keeper."}]
    )
    speeds = []
    for _ in range(3):
        engine._prompt_cache = None
        _, metrics = resident_tokens(
            engine, ids, Sampling(temperature=0.0, top_p=1.0, top_k=0, max_tokens=256)
        )
        speeds.append(metrics["decode_tokens_per_second"])
    started = time.perf_counter()
    code = reply(engine, CODE, 600)
    result = {
        "steering": engine.startup["steering"],
        "agreement": agreement,
        "probe": probe,
        "decode_tokens_per_second": speeds,
        "code_passes": code_passes(code),
        "code_answer_chars": len(code),
        "code_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
