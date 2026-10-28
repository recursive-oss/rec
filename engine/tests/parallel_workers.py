"""Worker count for `pytest -n auto` that leaves the desktop responsive."""

import os
import subprocess
import sys

# Cores left for the OS and apps during `pytest -n auto`.
RESERVED_CORES = 2
# Same variable pytest-xdist reads for its own auto count.
WORKERS_ENV = "PYTEST_XDIST_AUTO_NUM_WORKERS"
MACOS_PERFORMANCE_CORES_KEY = "hw.perflevel0.logicalcpu"
MACOS_LOGICAL_CORES_KEY = "hw.logicalcpu"


def _sysctl(key):
    result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    return int(result.stdout.strip())


def available_cores():
    if sys.platform == "darwin":
        # Efficiency cores are slow for this suite; Intel Macs have no perflevel key.
        cores = _sysctl(MACOS_PERFORMANCE_CORES_KEY)
        if cores is None:
            cores = _sysctl(MACOS_LOGICAL_CORES_KEY)
    elif sys.platform.startswith("linux"):
        cores = len(os.sched_getaffinity(0))
    else:
        cores = os.cpu_count()
    if not cores or cores < 1:
        raise RuntimeError("cannot determine the number of CPU cores")
    return cores


def auto_worker_count(environ, cores):
    override = environ.get(WORKERS_ENV)
    if override is not None:
        count = int(override)
        if count < 1:
            raise ValueError(f"{WORKERS_ENV} must be >= 1, got {override!r}")
        return count
    return max(1, cores - RESERVED_CORES)
