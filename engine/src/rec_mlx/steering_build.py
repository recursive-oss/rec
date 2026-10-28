"""Build a steering file from contrasting chat pairs.

Each JSONL line holds {"positive": messages, "negative": messages} and optional "tools": the same
request with and without the behaviour to steer toward. Both are run through the native layers, and the residual
at every slot is averaged over the last prompt tokens. The direction is the difference of means
at the slot that separates the two sets best. The file adds `scale` times the measured gap along
that direction at that one slot: the residual stream carries the push forward, so adding it at
every later slot would compound it.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx


class Recorder:
    def __init__(self, last):
        self.last, self.rows = last, {}

    def __call__(self, h, slot):
        self.rows[slot] = h[0, -self.last :].astype(mx.float32).mean(axis=0)
        return h


def capture(model, prompts, last):
    from .steering import attach

    recorder = Recorder(last)
    attach(model, recorder)
    samples = []
    try:
        for ids in prompts:
            recorder.rows = {}
            mx.eval(model(mx.array([ids])))
            samples.append(mx.stack([recorder.rows[slot] for slot in sorted(recorder.rows)]))
    finally:
        attach(model, None)
    return mx.stack(samples)


def contrast(positive, negative, min_slot=1):
    gaps = positive.mean(axis=0) - negative.mean(axis=0)
    units = gaps / mx.linalg.norm(gaps, axis=-1, keepdims=True)
    projected_pos = mx.einsum("nsd,sd->ns", positive, units)
    projected_neg = mx.einsum("nsd,sd->ns", negative, units)
    spread = mx.sqrt((projected_pos.var(axis=0) + projected_neg.var(axis=0)) / 2) + 1e-6
    separation = (projected_pos.mean(axis=0) - projected_neg.mean(axis=0)) / spread
    best = int(mx.argmax(separation[min_slot:]).item()) + min_slot
    return separation, units[best], best


def steering_file(positive, negative, direction, separation, scale, slot):
    gap = (positive.mean(axis=0) - negative.mean(axis=0)) @ direction
    pos = mx.einsum("nsd,d->ns", positive, direction)
    neg = mx.einsum("nsd,d->ns", negative, direction)
    spread = mx.sqrt((pos.var(axis=0) + neg.var(axis=0)) / 2) + 1e-6
    along = gap / spread
    slots = gap.size
    band = mx.arange(slots) == slot
    inf = mx.full((slots,), float("inf"))
    return {
        "direction": direction,
        "strength": band.astype(mx.float32),
        "low": -inf,
        "high": inf,
        "shift": mx.where(band, scale * gap, 0.0),
    }, {
        "band": [slot for slot, inside in enumerate(band.tolist()) if inside],
        "gap": [round(float(v), 4) for v in gap.tolist()],
        "separation_along_direction": [round(float(v), 3) for v in along.tolist()],
        "per_slot_best_separation": [round(float(v), 3) for v in separation.tolist()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--pairs", type=Path, required=True, help="JSONL of positive/negative chats"
    )
    parser.add_argument("--output", type=Path, required=True, help="steering .safetensors to write")
    parser.add_argument("--scale", type=float, default=1.0, help="multiple of the measured gap")
    parser.add_argument("--last", type=int, default=4, help="prompt tokens to average at the end")
    parser.add_argument(
        "--min-slot", type=int, default=1, help="first slot the direction may come from"
    )
    args = parser.parse_args()
    from .engine import Engine

    engine = Engine(args.model_dir, args.manifest, context_limit=8192)
    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    model = engine.runtime.model.native
    with mx.stream(mx.new_stream(mx.gpu)):
        positive = capture(
            model, [engine.prompt(p["positive"], tools=p.get("tools")) for p in pairs], args.last
        )
        negative = capture(
            model, [engine.prompt(p["negative"], tools=p.get("tools")) for p in pairs], args.last
        )
        separation, direction, best = contrast(positive, negative, args.min_slot)
        tensors, report = steering_file(positive, negative, direction, separation, args.scale, best)
        mx.save_safetensors(str(args.output), tensors)
    report = {"pairs": len(pairs), "best_slot": best, "scale": args.scale, **report}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("pairs", "best_slot", "band")}))


if __name__ == "__main__":
    main()
