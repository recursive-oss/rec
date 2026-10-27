# Experimental Flint backend for Linux/ROCm

For this branch's opt-in performance work and its numerical limitations, see
[the RDNA4 experiment record](RDNA4.md).

This backend runs the original `RecursiveSI/flint-int4` checkpoint through rec's
CLI on one AMD GPU. It is a development implementation using PyTorch, Triton and
HIP Graphs. It does not port the Metal resident megakernel, and its current
performance is substantially below the independent llama.cpp HIP reference.

The existing installer and release workflow do not build or distribute this
backend. The development bundle below depends on an external Python environment.

## Qualified configuration

Only the following configuration has been exercised:

- Linux amd64, one visible Radeon AI PRO R9700, 32 GB VRAM, `gfx1201`.
- ROCm 10.1.0, HIP component `7.16.26385`, Python 3.13.
- `torch==2.14.0+rocm10.1.0` paired with
  `triton==3.8.0+git669b31ac.rocm10.1.0`.
- `transformers==5.16.1`, `tokenizers==0.23.3`, `safetensors==0.8.0`,
  `xgrammar==0.2.1`.
- Flint revision `3b93f9053b530488405ae9fca51d4900d6ce21f0`, unchanged affine
  group-64 INT4 expert/projection weights and INT8 embeddings, BF16 KV.
- Default 8,192 total input/output tokens; maximum 33,536, including output.

ROCm's release number and HIP's component version differ. The Triton distribution
version above includes a build suffix that `triton.__version__` does not expose.
Keep the exact Torch/Triton pair. Other GPUs and framework versions have not been
qualified. The host needs `/dev/kfd`, the GPU render device and enough CPU RAM to
load the checkpoint; the development manifest requires at least 24 GiB host RAM.

## Environment and wheel

Start in a Python 3.13 environment that already provides the exact ROCm 10.1
Torch/Triton pair above. Building or acquiring that GPU foundation is outside the
development builder; it must not silently substitute CPU, CUDA or another ROCm
release. Check it before installing the smaller application dependencies:

```bash
python3 - <<'PY'
import importlib.metadata as m
import torch
assert torch.__version__ == '2.14.0+rocm10.1.0'
assert torch.version.hip == '7.16.26385'
assert m.version('triton') == '3.8.0+git669b31ac.rocm10.1.0'
PY
python3 -m pip install 'setuptools==81.0.0' wheel \
  'transformers==5.16.1' 'tokenizers==0.23.3' 'safetensors==0.8.0' \
  'xgrammar==0.2.1' 'numpy==2.5.4' 'pytest==9.1.1'
python3 -m pip wheel --no-deps --no-build-isolation engine/rocm -w .build/rocm-wheels
python3 -m pip install --no-deps .build/rocm-wheels/rec_rocm-0.1.0-py3-none-any.whl
```

Run these commands inside the environment/container with the ROCm libraries and
GPU device access. This is not a self-contained ROCm installation recipe. The
wheel reuses pure Python modules under `rec_mlx` for the chat template, checkpoint
validation, sampling contract and protocol conversion. No MLX runtime is imported
by the ROCm entry point; do not install this wheel over a separate `rec-mlx`
installation in the same environment because both provide that package namespace.

## Select a GPU and build rec

Select an available device before importing Torch. Prefer its GPU UUID when HIP
ordinal order differs from physical PCI enumeration:

```bash
export ROCR_VISIBLE_DEVICES=GPU-your-selected-uuid
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export REC_ROCM_PYTHON="$(command -v python3)"
export TRITON_CACHE_DIR="$PWD/.build/rocm-cache/triton"
export TORCH_EXTENSIONS_DIR="$PWD/.build/rocm-cache/torch"
export OMP_NUM_THREADS=8
python3 -c 'from rec_rocm.platform import validate_platform; print(validate_platform())'

python3 packaging/build-rocm-runtime.py --output .build/runtime-rocm
python3 packaging/package.py \
  --runtime-dir .build/runtime-rocm \
  --manifest .build/runtime-rocm.manifest.json \
  --version 0.1.0-rocm-dev --output .build/rec-rocm
.build/rec-rocm doctor
.build/rec-rocm serve
```

The build requires the Go version in `go.mod`. Choose a fresh builder output
directory when rebuilding; existing output is rejected. The runtime carries its
Python source but uses `REC_ROCM_PYTHON`, or `python3` on PATH, for external GPU
dependencies. `rec doctor` uses `python3` on PATH as well, so keep the qualified
environment activated. The binary retains the existing rec download/hash checks
for every checkpoint file. Downloads require about 22.6 GB, and working/build
space is additional. The API binds to loopback port 11500, uses the usual
`rec auth token`, and defaults to Flint grammar.

For 8K or 32K *prompt* tests with output allowance, start with
`.build/rec-rocm serve --context 33536`. Do not run a second complete model while
this service is resident. Use `.build/rec-rocm stop` before full-model validation.
The manifest has `development=true`; automatic updates do not provide a ROCm
release. The standard installer should not be used to install this development
binary.

## Opt-in HIP kernels (gfx1151)

`REC_GFX1151_ATTENTION=hip` (with `REC_GFX1151_DECODE=layer` and
`REC_GFX1151_KV=k8v4`) replaces the Triton decode attention of the quantized
full-attention layers with a HIP kernel; the default Triton paths are unchanged.
The kernel source ships with the Python package and is compiled on first use by
`torch.utils.cpp_extension`, which needs `ninja` (`python3 -m pip install ninja`)
and a ROCm tree whose `hipcc --version` reports exactly `torch.version.hip`
(`ROCM_HOME`, `ROCM_PATH`, the tree of the loaded `libamdhip64`, then `/opt/rocm`
are tried in that order). The model compiles it once at startup (about 20 s) and
caches the library under `REC_HIP_CACHE` (default `~/.cache/rec/hip`), keyed by
the source, flags and Torch/HIP versions; a cached library loads without `hipcc`
or `ninja`.

## Validation

In the pinned environment, with exactly one selected qualified GPU:

```bash
make test                 # normal platform checks; macOS engine tests need a Mac
make test-rocm            # small GPU model, independent CPU references and HTTP tests
```

The small suite covers affine INT4/INT8 unpack, projections, norm, routing,
learned-sink GQA attention, RoPE including position 32K, SWA 513/514 boundaries,
chunked prefill, snapshots, and scalar/Graph parity. HTTP tests use a fixture
engine; actual checkpoint API checks were run separately. See
[the qualification record](VALIDATION.md) for the original full-model tests and
shared-GPU measurements. Those historical timings are not benchmark results for
every subsequent commit.

## Implementation and limits

- Original compressed expert weights stay resident. A 3,584 MiB default budget
  caches decoded dense weights; `REC_DENSE_CACHE_MIB` can reduce it. The measured
  configuration used the default, with a fixed reusable expert scratch buffer.
- BF16 vendor BLAS projections/activations, FP32 attention score/softmax, BF16 KV.
  Prefill uses chunks of 256; full attention with more than 8K keys tiles 64 query
  rows to bound temporary memory. Single-token decoding uses HIP Graphs with
  attention-capacity buckets.
- Generation is serialized with bounded admission. Exact prefix extensions reuse
  KV. Editing or shortening safely resets/replays; the upstream branch rollback
  scheduler has not been ported.
- The ROCm HTTP adapter shares pure protocol utilities but currently has a
  separate handler. Consolidating the transport/scheduler interface with the MLX
  backend needs maintainer agreement. API parity is limited to exercised cases.
- This is text-only, single GPU, packet=1. No continuous batching, multi-GPU,
  262K context, other GPU architectures, portable runtime installer or automatic
  update support is claimed.
- Low-level Triton projection functions remain as small-tensor/reference test
  helpers. End-to-end inference uses the BF16 BLAS path. Fused GEMV and compact
  expert BMM candidates failed the full-model numerical gate and are not exposed
  as runtime CLI options.

Current speed and VRAM headroom need further work. A contribution of this backend
is experimental enablement, not a claim of production support or speedup over
llama.cpp. See the qualification record for the exact limitations.
