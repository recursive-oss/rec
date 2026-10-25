# RDNA4 performance experiments

This branch adds opt-in Triton candidates for the pinned gfx1201 configuration
described in [README.md](README.md). Every option defaults to `off`. This is a
local development branch; the original ROCm qualification record remains a
historical record.

## Candidate preserving the original reductions

```bash
export REC_RDNA4_NORM=apply
export REC_RDNA4_ROPE=fused
export REC_RDNA4_ATTENTION=mask
```

- `NORM=apply` retains the original Torch FP32 square/mean/rsqrt and fuses only
  the two following multiplications and BF16 store.
- `ROPE=fused` shares the original Torch cos/sin across layers and fuses rotation
  with FP32 intermediate operations. FP contraction is disabled.
- `ATTENTION=mask` fuses causal/window masking and insertion of the learned sink.
  QK, softmax and PV retain the original FP32 implementation. The long-prefill
  query-tiled path is unchanged.

No checkpoint format, activation/KV precision, routing, sampling, grammar or
packet size changes. Startup metadata reports all selected options. Request
metrics include graph capture count and device usage at request completion.

## Numerical evidence and limits

On 2026-10-25, the combined candidate and the following frozen-source control
were bitwise identical to the preceding c9c4e03 baseline in a single runtime:
1K, 8K and 32K prompts, Chinese, code, tool JSON text and a held-out arithmetic
prompt, each with 128 fixed teacher steps. Checks cover all 49 layer outputs,
logits, greedy tokens and all top-6 expert IDs. The original and candidate
classes are loaded from separate source files; the baseline is rerun after the
candidate. The small GPU/protocol suite passes 25 tests with the options enabled.

An existing cross-process baseline reproducibility issue remains unresolved.
Unmodified c9c4e03 can produce different trajectories, including after prefill
warmup. Captured differences first appear in layer-0 attention output; preceding
projections and Q/K norm agree. Therefore same-runtime A/B/A qualification does
not establish deterministic equivalence across arbitrary process initializations.
Invalid A/A experiments were retained and excluded from candidate acceptance.
No tolerance was relaxed: hidden/logit relative L2 <=1%, exact greedy and route
agreement on the frozen cases. HTTP and stability qualification are still in
progress; microbenchmarks are not service performance claims.

## Rejected research switches

The following implementations are retained for diagnosis and default to `off`:

| Variable | Values | Full-model screening result |
| --- | --- | --- |
| `REC_RDNA4_EXPERT` | `simd`, `wmma` | Hidden/logit or route gate failed |
| `REC_RDNA4_PREFILL` | `compact`, `grouped` | Hidden/logit or route gate failed |
| `REC_RDNA4_ATTENTION` | `decode`, `all` | Hidden/logit or route gate failed |
| `REC_RDNA4_NORM` | `on` | Hidden/logit or route gate failed |

`REC_RDNA4_ROPE=on` only reuses angles and is an ablation of `fused`. Failed
projection/attention candidates must not be interpreted as qualified serving
paths, even when their isolated operator error is small. Real projection
fixtures and independent CPU FP64 checks show BF16 midpoint sensitivity; local
agreement alone does not establish full-model agreement for this MoE model.

Local artifacts are recorded under run ID `rdna4-opt/20261025-a`, including
source hashes, frozen fixtures, rejected candidates, profiler traces and raw
measurements. They are not embedded in the runtime bundle.
