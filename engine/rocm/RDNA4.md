# RDNA4 performance experiments

This development branch adds opt-in gfx1201 candidates to the experimental
Linux/ROCm Flint backend. Every engine option defaults to `off`. The checkpoint,
BF16 activation/KV formats, router ordering, grammar and sampling contract remain
unchanged. The ordinary setup is described in [README.md](README.md).

## Implemented configuration and validation

Compute revision `5c15ba0` completed 29 enabled tests and 29 fallback tests,
seven frozen 128-step A/B/A cases, and all three 128-step Graph cases. Measured
hidden/logit relative L2 was zero; greedy tokens and expert IDs/order matched.
All 60 full-output candidate HTTP measurements (forward and reverse) matched
the frozen original references. This is a bounded claim for the pinned runtime
and exercised inputs, not arbitrary-input or cross-version equivalence.

| Context / output | Original forward token/s | Candidate forward token/s | New llama.cpp token/s |
| --- | ---: | ---: | ---: |
| 1024 / 1024 | 25.908 | 36.867 | 75.604 |
| 8192 / 1024 | 15.306 | 29.872 | 70.910 |
| 32768 / 512 | 10.765 | 18.578 | 51.727 |

Each group used two warmups and five full measurements, cold prefix and warm
model/JIT. rec uses default Flint grammar; llama uses none and a different GGUF
representation, so it is an engineering reference, not a numerical oracle.
Candidate reverse medians were 33.156/29.540/18.870 token/s. The forward gain
includes the fixed BLAS table and allocator settings as well as kernel changes.
All valid slow samples are retained.

The candidate completed 60.22 minutes, 324 mixed requests, 14 protocol rounds and
316 Graph captures. Allocated growth over 100 fixed-shape requests was zero.
Observed device peak was 27.5625 GiB; peak reserved was 26.84375 GiB; observed free minimum was
4.296875 GiB. Maximum per-request nonempty-SSE gap p95 was 89.17 ms. Device
sampling is every 10 seconds plus request-end observations, not an instantaneous
bound. Kernel reset logs were unavailable. Built Go-manager readiness, default
grammar chat, extracted-source hashes and clean stop passed; the full performance
matrix used the identical Python source. Temporary manager state was removed.

The original uncontrolled reverse 1K run diverged at the 19th token in all 14 requests.
It remains a failed validation. A separate fixed-BLAS original control matched
all 30 full outputs, but its 1K/8K windows included three observations of another
GPU2 process. Those windows are exploratory and do not qualify exclusive-card
performance. The candidate lifetimes, original forward, new llama and 32K control
had no observed second GPU owner; this does not prove continuous isolation or
exclusive host CPU resources. M1/M2/M3 all remain unmet, and the original reverse
and whole-batch exclusivity gates remain failed. No default optimization is enabled.

The 2048 MiB predecessor exceeded the device budget during mixed requests.
A 64-recapture reproducer isolated retained BLAS workspaces: original warmup
streams grew 5040 MiB, while stream reuse grew 80 MiB initially and zero thereafter.
All 64 outputs in each run matched eager. `GRAPH=compact` now retains one warmup
stream per serial model while preserving synchronization and KV restoration;
the sustained full-model check above validates the integrated fix. Production
code never calls the private workspace-clear diagnostic API.

The 3072 MiB budget holds all 3207069696 bytes of dense weights, equal to the
original's actual cache. Lowering the budget number did not remove dense weights.
The final profile is `profile-stable-stream.json`; the controlled original recovery
is recorded separately in `profile-stable-stream-controlled-reverse.json`.

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export REC_DENSE_CACHE_MIB=3072
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
| `GRAPH=compact` | Save/restore only slots touched by capture and warmup, including ring wrap; reuse one warmup stream per model to bound BLAS workspace retention. The default full-snapshot path is unchanged. |
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

Against the fixed family, the candidate at `5c15ba0` and subsequent frozen-c9 control
have zero measured hidden/logit relative L2 and identical greedy/routes on 1K/8K/32K, Chinese, code, tool JSON text
and a held-out arithmetic case, each for 128 fixed teacher steps. Checks cover
all 49 hidden outputs, logits, greedy tokens and top-6 expert IDs/order. Teachers
are immutable, including the first valid original-c9 sequences for short cases.
This bounds the claim to the exercised inputs/runtime; it does not establish
arbitrary-input or cross-version deterministic equivalence.

The ordered-PV local probe covered six shapes and eight configurations, all
FP32/BF16 bitwise equal to the verified vendor family. Hot isolated decode PV
chains improved 2.55–4.95x. Both prefill cases regressed, so prefill was excluded
from this path. Independent CPU arithmetic and Graph position-change tests
exercise the integrated implementation. The smaller tiles then passed five ABBA rounds against the integrated first
version, improving isolated PV by 1.19x for the window and 1.43–1.47x for full
attention. These ratios must not be multiplied into earlier benchmarks or
quoted as service gains. HTTP outcomes and their limitations are recorded above.

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
- Token-count buckets with vendor BMM reduced estimated padding, but 282 real
  projections showed differences. A controlled 16-step model screen failed
  (max logit/hidden relative L2 3.27%/14.60%, 275 layer/step route mismatches);
  the post-control was exact. This path remains outside the engine.
- The later 2048 MiB forward matrix completed, and fixed-shape 100-request
  allocated growth was zero. Mixed requests then reached 28.80078125 GiB device
  usage (42 Graph captures); the configuration was rejected and stopped cleanly.
  Merely lowering cache to 1024 MiB was superseded by the workspace-retention
  diagnosis. Its seven numerical cases passed; later gates were not run.
- `ROPE=on` is an angle-reuse ablation of `fused`.

Local evidence is under `artifacts/amd/rocm10/rdna4-opt/20261025-a` in the parent
workspace: `profile-stable-stream.json`, source/build hashes, frozen fixtures,
`paired-stable-stream`, `graph-stable-stream`, `http-stable-stream`,
`http-host-profile`, `stream-retention`, `pv-probe`, `blas-freeze`, raw service
requests and telemetry. These files and weights are not embedded in the runtime.
The development bundle and wheel still require the external pinned Python/ROCm
stack. All jobs in this run were serialized on the selected physical device;
the observed external jobs are recorded separately above.
