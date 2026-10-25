"""Shared constants and skip rule for the real-weights e2e tests (no test module imports)."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

MODEL_DIR = Path(os.environ["REC_E2E_MODEL_DIR"]) if os.environ.get("REC_E2E_MODEL_DIR") else None
MANIFEST_NAME = "flint-int4.json"
CONTEXT_LIMIT = 16_384
MIN_AVAILABLE_BYTES = 4 * 1024**3  # free + inactive + speculative + purgeable pages
AVAILABLE_PAGE_KINDS = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")


def _vm_available_bytes() -> int:
    out = subprocess.check_output(["vm_stat"], text=True)
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    total = 0
    for kind in AVAILABLE_PAGE_KINDS:
        total += int(re.search(rf"{kind}:\s+(\d+)\.", out).group(1))
    return total * page


def _skip_reason() -> str | None:
    if MODEL_DIR is None:
        return "REC_E2E_MODEL_DIR is not set (flint-int4 weights unavailable)"
    if not (MODEL_DIR / MANIFEST_NAME).exists():
        return f"{MODEL_DIR / MANIFEST_NAME} not found (flint-int4 weights unavailable)"
    try:
        import mlx.core as mx
    except ImportError:
        return "mlx is not installed"
    if not mx.metal.is_available():
        return "no Apple GPU (Metal) available"
    available = _vm_available_bytes()
    if available < MIN_AVAILABLE_BYTES:
        return f"memory preflight: vm_stat available {available} B < {MIN_AVAILABLE_BYTES} B"
    return None


SKIP_REASON = _skip_reason()
