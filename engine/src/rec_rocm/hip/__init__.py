"""Opt-in HIP decode kernels for gfx1151, compiled on first use by torch.utils.cpp_extension.

The build needs ninja and a ROCm tree whose hipcc reports the HIP version Torch was built
with. Builds are cached under REC_HIP_CACHE (default ~/.cache/rec/hip), keyed by a hash of
the source, flags and toolchain, so a changed source never reuses a stale library.
"""

import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path

import torch

ARCH = "gfx1151"
_FLAGS = ("-O3", f"--offload-arch={ARCH}")
_modules = {}


def _rocm_home():
    """The first candidate ROCm tree whose hipcc matches torch.version.hip."""
    candidates = [os.environ.get("ROCM_HOME"), os.environ.get("ROCM_PATH")]
    with open("/proc/self/maps") as maps:
        for line in maps:
            if "/libamdhip64.so" in line:
                candidates.append(str(Path(line.split()[-1]).parents[1]))
                break
    candidates.append("/opt/rocm")
    for root in filter(None, candidates):
        hipcc = Path(root) / "bin/hipcc"
        if not hipcc.exists():
            continue
        out = subprocess.run([hipcc, "--version"], capture_output=True, text=True).stdout
        if f"HIP version: {torch.version.hip}\n" in out:
            return root
    raise RuntimeError(
        f"REC HIP kernels need hipcc for HIP {torch.version.hip}; set ROCM_HOME to that ROCm"
    )


def load(name):
    """Build (or reuse) and import the extension compiled from `<name>.hip`."""
    if name in _modules:
        return _modules[name]
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != ARCH:
        raise RuntimeError(f"REC HIP kernels are qualified only on {ARCH}; found {arch}")
    source = Path(__file__).with_name(f"{name}.hip").read_bytes()
    toolchain = [" ".join(_FLAGS), torch.__version__, torch.version.hip]
    key = hashlib.sha256(b"\0".join([source, *map(str.encode, toolchain)])).hexdigest()[:16]
    root = Path(os.environ.get("REC_HIP_CACHE") or Path.home() / ".cache/rec/hip")
    directory = root / f"{name}-{key}"
    module_name = f"rec_hip_{name}_{key}"
    built = directory / f"{module_name}.so"
    if built.exists():
        # A finished build for this exact source and toolchain needs neither hipcc nor ninja.
        spec = importlib.util.spec_from_file_location(module_name, built)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _modules[name] = module
        return module
    rocm = _rocm_home()
    directory.mkdir(parents=True, exist_ok=True)
    copy = directory / f"{name}.hip"
    if not copy.exists() or copy.read_bytes() != source:
        copy.write_bytes(source)
    # cpp_extension reads ROCM_HOME at import and PYTORCH_ROCM_ARCH at build time.
    os.environ.setdefault("ROCM_HOME", rocm)
    previous = os.environ.get("PYTORCH_ROCM_ARCH")
    os.environ["PYTORCH_ROCM_ARCH"] = ARCH
    try:
        from torch.utils import cpp_extension

        if cpp_extension.ROCM_HOME != rocm:
            raise RuntimeError(
                f"torch.utils.cpp_extension uses ROCm {cpp_extension.ROCM_HOME}, expected {rocm}"
            )
        module = cpp_extension.load(
            name=module_name,
            sources=[str(copy)],
            build_directory=str(directory),
            extra_cuda_cflags=list(_FLAGS),
            verbose=False,
        )
    finally:
        if previous is None:
            del os.environ["PYTORCH_ROCM_ARCH"]
        else:
            os.environ["PYTORCH_ROCM_ARCH"] = previous
    _modules[name] = module
    return module


def sink_attention_k8v4(q, cache, extent, positions, sinks):
    """HIP version of gfx1151.sink_attention_k8v4 (32 query heads, 4 KV heads, head_dim 128)."""
    if q.shape[1:] != (32, 128) or cache.key.shape[1:] != (4, 128):
        raise ValueError("The HIP k8v4 attention kernel is built for 32/4 heads of dimension 128")
    # One slot range per CU (gfx11 reports WGPs of two CUs); swept 20-128 on 1K-32K contexts.
    splits = 2 * torch.cuda.get_device_properties(q.device).multi_processor_count
    return load("attention").attention_k8v4(
        q,
        cache.key,
        cache.key_scale,
        cache.key_bias,
        cache.value,
        cache.value_scale,
        cache.value_bias,
        cache.positions,
        positions,
        sinks,
        extent,
        splits,
    )
