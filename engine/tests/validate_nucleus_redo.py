"""Check that nucleus-kernel packets match the scalar sampler, including steps redone by the caller.

Run once per --margin: the default (steps are almost never flagged), 3e-3 (about one step in
twenty is flagged and redone) and 10 (every step is redone). A 1000-token prompt keeps the
sliding-window ring buffers full, which is what makes a redone step sensitive to stale writes.
"""

import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

from rec_mlx.engine import Engine, Sampling
from rec_mlx.kernels import masked_sampling, nucleus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Flint checkpoint directory")
    parser.add_argument("--manifest", type=Path, required=True, help="checkpoint manifest")
    parser.add_argument("--margin", type=float, default=nucleus.MARGIN)
    args = parser.parse_args()
    nucleus.MARGIN = args.margin
    engine = Engine(args.model, args.manifest, context_limit=131072)
    runtime, resident = engine.runtime, engine.resident_sampler
    source = Path(nucleus.__file__).read_text()
    ids = engine.tokenizer.encode(source * 4, add_special_tokens=False)[:1000]
    for top_p, top_k in ((0.95, 20), (0.5, 3)):
        for seed in (3, 11):
            streams = {}
            for mode in ("packet", "scalar"):
                session = runtime.make_session(max_new_tokens=200)
                logits = session.prefill(ids)
                mx.random.seed(seed)
                sample = make_sampler(temp=1.0, top_p=top_p, top_k=top_k)

                def draw(values):
                    flat = values.reshape(-1).astype(mx.float32)
                    return int(sample(flat - mx.logsumexp(flat)).item())

                tokens = [draw(logits)]
                while len(tokens) < 97:
                    if mode == "packet":
                        tokens += resident.block(
                            session._state,
                            tokens[-1],
                            8,
                            seed,
                            temperature=1.0,
                            top_p=top_p,
                            top_k=top_k,
                        )
                    else:
                        tokens.append(draw(session.forward_token(tokens[-1])))
                streams[mode] = tokens[:97]
            assert streams["packet"] == streams["scalar"], (top_p, top_k, seed, args.margin)
            print("ok", top_p, top_k, seed, args.margin, flush=True)

    # The server always decodes under a grammar: a redone step must not advance it.
    eligible = masked_sampling.nucleus_eligible
    constraint = engine.constraints.tools(None)
    for seed in (3, 11):
        streams = {}
        for mode, flag in (("nucleus", eligible), ("sort", lambda *a: False)):
            masked_sampling.nucleus_eligible = flag
            sampling = Sampling(temperature=1.0, top_p=0.95, top_k=20, max_tokens=96, seed=seed)
            events = engine.generate_tokens(ids, sampling, grammar=constraint)
            streams[mode] = [event["token"] for event in events if "token" in event]
        masked_sampling.nucleus_eligible = eligible
        assert streams["nucleus"] == streams["sort"] and len(streams["sort"]) == 96, (
            seed,
            args.margin,
        )
        print("ok grammar", seed, args.margin, flush=True)


if __name__ == "__main__":
    main()
