"""The deliberately narrow platforms qualified for this experimental backend."""

from importlib.metadata import version

import torch

from .platforms import qualify


def validate_platform():
    """Return the selected device and its qualified Platform."""
    count = torch.cuda.device_count()
    device = torch.cuda.get_device_properties(0) if count else None
    platform = qualify(
        device.gcnArchName.split(":")[0] if device else None,
        torch.__version__,
        torch.version.hip,
        version("triton"),
        count,
        device.total_memory if device else 0,
    )
    return device, platform
