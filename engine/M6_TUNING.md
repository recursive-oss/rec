# M6 schedule experiments

M6 uses a dedicated device profile: 36 attention workers, 32 expert workers, 32 output-head workers, 2048-token prefill, and generic mixed experts. No tested candidate showed a consistent improvement sufficient to replace it.

Apple documents 12 GPU cores versus 10 on base M5, up to 170 GB/s versus 153 GB/s bandwidth, and updated Dynamic Caching: https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/

This motivated occupancy and chunk-size experiments. These scalar Metal kernels do not automatically use the Neural Engine or GPU Neural Accelerators.

Local M6 32 GiB, mixed 3/4-bit Flint, MLX HEAD, temperature 1/top-p .95/top-k 20, 128 generated tokens after 16-token warmup, cold prompt cache. One measured run per candidate; baseline repeated at the end. This is exploratory tuning, not statistically established speedup evidence.

| Configuration (attention/expert/head; chunk; specialized) | 34 tokens | 4095 tokens | 8192 tokens |
|---|---:|---:|---:|
| 36/32/32; 2048; off | 85.1 | 75.4 | 70.6 |
| 48/32/32; 2048; off | 84.2 | 74.4 | 68.0 |
| 36/32/16; 2048; off | 80.6 | 74.9 | 67.6 |
| 36/32/32; 1024; off | 81.9 | 75.5 | 69.9 |
| 32/32/32; 2048; off | 84.2 | 73.8 | 67.1 |
| 40/32/32; 2048; off | 83.3 | 73.0 | 67.3 |
| 32/32/32; 2048; on | 75.2 | 75.0 | 69.8 |
| Baseline repeat | 84.6 | 73.9 | 67.9 |

Throughput is tokens/second. Baseline drift at 8192 tokens (70.6 to 67.9) means small differences are inconclusive. A 24-expert-worker trial was rejected by the existing 32/64-worker constraint.

The earlier 4096-token/M5-specialization experiment passed 61 scalar/packet, 9 EOS and 3 cancellation cases but retained 72 independent native-reference failures. This tuning does not resolve that limitation. Candidate schedules above received generation/performance checks; they are not promoted as numerically validated alternatives.
