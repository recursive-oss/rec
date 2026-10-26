"""Qualified ROCm platforms; cli/platform.go mirrors this table for its preflight."""

from dataclasses import dataclass

GIB = 1024**3


@dataclass(frozen=True)
class Platform:
    torch: str
    hip: str
    triton: str
    min_memory: int
    max_context: int


PLATFORMS = {
    "gfx1201": Platform(
        torch="2.14.0+rocm10.1.0",
        hip="7.16.26385",
        triton="3.8.0+git669b31ac.rocm10.1.0",
        min_memory=30 * GIB,
        max_context=33536,
    ),
    # Only a 120 GiB GTT configuration was exercised.
    "gfx1151": Platform(
        torch="2.9.1+rocm7.2.1.gitff65f5bc",
        hip="7.2.53211-e1a6bc5663",
        triton="3.5.1+rocm7.2.1.gita272dfa8",
        min_memory=100 * GIB,
        max_context=34816,
    ),
}


def qualify(arch, torch_version, hip_version, triton_version, device_count, total_memory):
    """Return the Platform for the single visible GPU, or raise RuntimeError."""
    if device_count != 1:
        raise RuntimeError("Select one GPU with ROCR_VISIBLE_DEVICES before starting rec")
    platform = PLATFORMS.get(arch)
    if platform is None:
        raise RuntimeError(
            f"This experimental backend is qualified only on {', '.join(PLATFORMS)}; found {arch}"
        )
    if (torch_version, hip_version, triton_version) != (
        platform.torch,
        platform.hip,
        platform.triton,
    ):
        raise RuntimeError(
            f"Use the pinned {arch} Torch/Triton environment in engine/rocm/README.md"
        )
    if total_memory <= platform.min_memory:
        raise RuntimeError(
            f"{arch} needs more than {platform.min_memory // GIB} GiB of GPU-visible memory"
        )
    return platform
