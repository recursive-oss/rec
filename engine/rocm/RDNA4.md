# RDNA4 performance experiments

This development branch adds opt-in gfx1201 candidates to the experimental
Linux/ROCm Flint backend. Every engine option defaults to `off`. The checkpoint,
BF16 activation/KV formats, router ordering, grammar and sampling contract remain
unchanged. The ordinary setup is described in [README.md](README.md).

## Current experimental configuration

The source/build under qualification are recorded in the workspace
`profile-ordered.json`. The combined implementation at `e12c03e` passed seven
frozen 128-step A/B/A cases and all three Graph cases. Small-model tests then
identified the need to keep the original PV implementation for head dimensions
other than 128. That guard is now explicit; qualification of the final source,
complete HTTP, memory and sustained load are still in progress. Do not treat local kernel timing as an accepted service speedup.

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export REC_DENSE_CACHE_MIB=2048
export REC_RDNA4_NORM=apply
export REC_RDNA4_ROPE=fused
export REC_RDNA4_ATTENTION=mask
export REC_RDNA4_GRAPH=compact
export REC_RDNA4_PREFILL=compact
export REC_RDNA4_PV=ordered
# This must point to the validated table for the exact qualified library build:
export HIPBLASLT_TUNING_OVERRIDE_FILE=/path/to/c9-family-compact.csv
```

| Option | Behavior |
| --- | --- |
| `NORM=apply` | Retain Torch FP32 square/mean/rsqrt; fuse the following two multiplications and BF16 store. |
| `ROPE=fused` | Share original Torch cos/sin across layers; fuse rotation with FP contraction disabled. |
| `ATTENTION=mask` | Fuse causal/window masking and learned-sink insertion; retain QK and softmax. |
| `GRAPH=compact` | Save/restore only slots touched by capture and warmup, including ring wrap; retain the original full snapshot as default. |
| `PREFILL=compact` | Gather active-expert inputs and unpack their weights directly into contiguous reusable workspace. Preserve BF16 BMM and stable assignment order. |
| `PV=ordered` | For single-token decode with head dimension 128, read BF16 V directly and accumulate FP32 FMAs in ascending physical K order. Prefill retains BLAS. |

Full-attention decode has a contiguous valid KV prefix, so ordered PV can omit
future slots. Sliding-window decode visits the complete physical ring. The
position is read from device memory on every Graph replay. This path preserves
the softmax sink denominator and the BF16 attention-output boundary.

Expandable segments and dense-cache size are process settings, not new engine
defaults. The original c9 comparison retains its 3584 MiB cache budget. Startup
metadata reports all selected engine options, including `pv`; request metrics
report Graph capture count and device usage at request completion.

## Numerical and algorithm-selection contract

The recorded platform is Radeon AI PRO R9700/gfx1201, ROCm 10.1, Torch
2.14.0+rocm10.1.0 and the pinned Triton build. The loaded hipBLASLt revision is
`39d8d603`; library SHA-256 is
`a3922eda57c8789b75e234f0cc3e92ca5a07f6e0cb1450cb280b8a321601a5fb`.
The experimental override table SHA-256 is
`16d22a9814fc8d395c39bc5619de43724354d96775b0010ce9c460e096e16735`.
It records/expands known shapes and fixes the observed FP32 QK/PV family to
140222/140206. Solution indices are not portable across library builds.
Unrecorded shapes can fall back to the vendor heuristic.

Unmodified c9 can select different FP32 PV algorithms across processes. Replays
with identical QK and softmax probabilities demonstrated different FP32/BF16 PV
outputs; a valid algorithm override reproduced either result. The reason for
all heuristic variation has not been fully explained. In this library the CSV
requires an exact `Git Version` first line and a header before every data row;
a conventional one-header CSV was silently ineffective in earlier experiments.

Against the fixed family, the candidate at `e12c03e` and subsequent frozen-c9 control
match the preceding c9 result bitwise on 1K/8K/32K, Chinese, code, tool JSON text
and a held-out arithmetic case, each for 128 fixed teacher steps. Checks cover
all 49 hidden outputs, logits, greedy tokens and top-6 expert IDs/order. Teachers
are immutable, including the first valid original-c9 sequences for short cases.
This bounds the claim to the exercised inputs/runtime; it does not establish
arbitrary-input or cross-version deterministic equivalence.

The ordered-PV local probe covered six shapes and eight configurations, all
FP32/BF16 bitwise equal to the verified vendor family. Hot isolated decode PV
chains improved 2.55–4.95x. Both prefill cases regressed, so prefill was excluded
from this path. Independent CPU arithmetic and Graph position-change tests
exercise the integrated implementation. Complete service gains remain pending.

## Prior rejected configurations and research paths

- The initial pointwise/compact-Graph service diverged from historical HTTP
  output at token 67. Same-runtime A/B/A alone did not qualify that service.
- Packed compact prefill at `683113b`, with the fixed BLAS table, passed all
  numerical/Graph gates and three-context 128-token HTTP screens. Complete
  service-lifetime device usage reached 28.588 GiB, exceeding the unchanged
  28 GiB budget; its formal matrix was therefore blocked. Smaller short-screen
  memory figures were not used to override that failure.
- `EXPERT=simd/wmma`, `PREFILL=grouped`, `ATTENTION=decode/all`, and `NORM=on`
  retain research implementations that failed full-model hidden/logit or route
  gates. The projection search tested 102 bounded configurations without finding
  an exact candidate on all six real fixtures. CPU FP64 checks demonstrate
  BF16 midpoint sensitivity; more accurate arithmetic is not automatically
  equivalent to the original rounded path.
- `ROPE=on` is an angle-reuse ablation of `fused`.

Local evidence is under `artifacts/amd/rocm10/rdna4-opt/20261025-a` in the parent
workspace: `profile-ordered.json`, source/build hashes, frozen fixtures,
`paired-ordered-2048`, `graph-ordered-2048`, `pv-probe`, `blas-freeze`, raw service
requests and telemetry. These files and weights are not embedded in the runtime.
The development bundle and wheel still require the external pinned Python/ROCm
stack. All GPU jobs are serialized on the selected physical device.
