"""The deliberately narrow platform qualified for this experimental backend."""

from importlib.metadata import version

import torch


def validate_platform():
    if (
        torch.__version__ != "2.14.0+rocm10.1.0"
        or torch.version.hip != "7.16.26385"
        or version("triton") != "3.8.0+git669b31ac.rocm10.1.0"
    ):
        raise RuntimeError(
            "Use the pinned ROCm 10.1 Torch/Triton environment in engine/rocm/README.md"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select one GPU with ROCR_VISIBLE_DEVICES before starting rec")
    device = torch.cuda.get_device_properties(0)
    if device.gcnArchName.split(":")[0] != "gfx1201" or device.total_memory <= 30 * 1024**3:
        raise RuntimeError("This experimental backend has only been qualified on 32 GB gfx1201")
    return device
