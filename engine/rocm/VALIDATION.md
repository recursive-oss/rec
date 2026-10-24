# Initial ROCm qualification record

Date: 2026-10-24. Base rec commit:
`0dcf43c765195c44d72e3c59c6849a228ebe638b`.
This summarizes the original port's measurements. Subsequent PR preparation
removes host-specific startup labels and the experimental projection CLI option,
adds environment checks and a repository-local development builder, and formats
the code. Model/kernel arithmetic is unchanged; the small suite is rerun on the
prepared branch. Shared hardware prevents a controlled performance comparison.

## Identity and correctness

- One Radeon AI PRO R9700, 32 GB, `gfx1201`, Linux amd64, ROCm 10.1.0,
  HIP 7.16.26385, Python 3.13.13, Torch 2.14.0+rocm10.1.0,
  Triton 3.8.0+git669b31ac.rocm10.1.0. CPU affinity: eight NUMA-local cores.
- GPU was shared with existing jobs with the operator's permission. Those jobs,
  power policies and clock settings were not modified. No exclusive-GPU result.
- Original Flint checkpoint revision
  `3b93f9053b530488405ae9fca51d4900d6ce21f0`: all 12 files SHA256-verified;
  1,751 stored tensors plus 47 zero router biases supplied by the loader.
- Small GPU suite: 20 tests and 13 subtests passed; independent CPU FP64/NumPy
  references cover small tensors. Pure upstream protocol tests: 22 tests and
  24 subtests passed. Go race, vet, Ruff, wheel isolation/import checks passed.
- Complete 49-layer checkpoint: 1K prompt plus 128 teacher-forced steps against
  an independent eager Torch unpack/KV implementation, forcing query tiling.
  Every checked hidden/logit relative error was zero, and greedy tokens matched.
  This oracle shares model structure and vendor BF16 BLAS; it is not an
  independent full FP32 model or an Apple/MLX equivalence test.
- Real checkpoint: arithmetic, tool call/replay/result continuation, SSE,
  Anthropic stream/non-stream, deterministic seed, cancellation/reconnection,
  context rejection, Chinese text, Python AST and long natural-stop checks.
- Complete 32K prompt memory gate succeeded after query tiling. An earlier
  untiled attempt OOMed; those failed results are excluded from the final run.
  A benchmark client partial-JSONL race was also fixed before the final run.

The compact-expert BMM candidate failed the full-model gate (28.53% maximum
logit relative L2 and one greedy disagreement). Two fused projection candidates
also exceeded 1%. These candidates were rejected without relaxing thresholds.
The selected prefill path retains the original 256-expert BLAS batch shape but
only dequantizes active slots. No successful correctness check of the rejected
full-model Triton projection path is claimed.

## HTTP measurements

All formal requests have cold prefix, exactly the indicated templated input
token count, and 512 actual output tokens. Each length/grammar configuration has
two warmups followed by five formal measurements. The 1K/8K results predate the
long-attention fix and were retained because their GPU arithmetic path did not
change; all 32K results use the tiled implementation.

| Prompt tokens | Grammar | Decode token/s median (min–max) | Prefill token/s median | HTTP TTFT seconds median |
| --- | --- | ---: | ---: | ---: |
| 1,024 | flint | 12.94 (11.20–15.77) | 260.0 | 4.07 |
| 1,024 | none | 16.42 (12.21–17.70) | 330.9 | 3.24 |
| 8,192 | flint | 8.01 (7.88–10.49) | 204.8 | 40.19 |
| 8,192 | none | 9.88 (7.38–12.08) | 200.7 | 41.02 |
| 32,768 | flint | 8.18 (7.12–9.10) | 206.0 | 159.24 |
| 32,768 | none | 6.47 (5.48–7.34) | 191.0 | 171.97 |

Decode is `(actual output count - 1)/(last token time - first token time)`.
TTFT includes HTTP, tokenization, prefill and the first nonempty content/reasoning
increment. Per-request nonempty stream chunk p95 values were below 500 ms.
Five requests are insufficient to claim a reliable request-level p95.
Prompts use repeated background text plus a long hash-table tutorial task to
control token count; this is not a broad model-quality evaluation.

The experiment's default-grammar 20/15/10 token/s targets, 1K/8K prefill target
of 300 token/s, and 80%-of-llama.cpp target were not met. These were local
engineering targets, not upstream merge requirements. The 1K TTFT target of
five seconds was met under the measured shared conditions.

## Independent HIP reference

llama.cpp commit `6b9ba4f2e8c1436511956f0b3c309e2c9e642071`, built Release with
ROCm 10.1/Clang 24 for gfx1201. Official `RecursiveSI/flint-int4-gguf` Q4_1
revision `9c47151036bc2c34331142dda04488bceab2bc9f`; file SHA256
`43c07baccdba42eb7528a9630ee15699b1942c297291dc1b39a061f508ee1a4b`.

Both backends used BF16 KV, total context capacity 33,536, CPU affinity/threads
8, and identical 1K/8K input token IDs after templating. Native HTTP reference
used no grammar, FA on, batch/ubatch 256, two warmups and five measurements,
all 512 outputs with no prompt cache hits. It offloaded 50/50 layers; buffers
were 23,610.61 MiB GPU plus 1,173 MiB CPU-mapped embedding. Therefore not every
weight was GPU-resident. The complete rec and llama models were loaded serially.

| Prompt tokens | llama decode token/s median (min–max) | rec flint / llama | rec none / llama |
| --- | ---: | ---: | ---: |
| 1,024 | 81.98 (34.56–82.22) | 15.8% | 20.0% |
| 8,192 | 31.51 (27.41–34.13) | 25.4% | 31.3% |

Native decode uses `(predicted_n - 1)/(predicted_ms / 1000)`. GGUF preserves
trained INT4 codes but uses FP16 affine metadata and F32 embedding, while this
backend uses the source metadata and BF16 BLAS activations. It is an engineering
reference, not a numerical oracle. Serial, unpaired shared-GPU timings are not a
controlled speedup claim. With identical 1K+128 teacher token inputs, no grammar
or sampling, and per-step GPU synchronization, rec/llama decode medians were
12.61/32.72 token/s (five measurements after two warmups).

## Memory and stability

The final implementation ran continuously for 3,608.07 seconds, processing 133
requests including 17 additional mixed Chinese/English/code/arithmetic requests.
There were zero observed service tracebacks or allocator retry warnings in this
final interval. After warmup, 100 fixed-shape requests had zero net allocated
memory growth. Historical failures above are not counted as a successful run.

- Torch cumulative peak allocated: 27.850 GiB; reserved: 28.090 GiB.
- Whole-device VRAM peak: 31.773 GiB, including unrelated background jobs,
  versus 31.860 GiB device capacity. The local 28 GiB device-usage budget and
  3 GiB free-headroom goal were not met.
- 351 read-only device samples during the continuous interval, no monitor errors;
  maximum edge/junction/memory temperature: 73/89/97 C.
- Service RSS median/max: 2.17/2.35 GiB; CPU affinity stayed on the selected cores.
- Kernel journal/dmesg access was denied. No claim of a complete GPU-reset log
  audit is made; stability covers successful service requests, process state
  and the readable metrics.

Full model timing/trace dumps and host-specific launchers are intentionally
excluded from the source change. Performance remains a follow-up. The dominant
short-context GPU kernel group in the debug trace was vendor BLAS (about 59.6%
of summed kernel time); that sum is not end-to-end latency. Chunk/packet search,
fused online attention, efficient branch rollback and portable releases remain
unimplemented.
