#!/usr/bin/env python3
"""Stage a relocatable native Flint executable for REC packaging."""

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flint-serve", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.flint_serve
    info = source.lstat()
    if source.is_symlink() or not stat.S_ISREG(info.st_mode) or not info.st_mode & 0o111:
        parser.error("--flint-serve must be an executable regular file, not a symlink")
    fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    opened = os.fstat(fd)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_mode, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )
    if (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino):
        os.close(fd)
        parser.error("--flint-serve changed while being opened")
    executable = f"/proc/self/fd/{fd}"
    probe = subprocess.run(
        [executable, "--help"], capture_output=True, text=True, timeout=30,
        pass_fds=(fd,),
    )
    if probe.returncode != 0 or "flint-serve" not in probe.stdout or "--checkpoint-dir" not in probe.stdout:
        parser.error("executable does not satisfy the native Flint server contract")
    contract_probe = subprocess.run(
        [executable, "--runtime-contract"], capture_output=True, text=True, timeout=30,
        pass_fds=(fd,),
    )
    try:
        contract = json.loads(contract_probe.stdout)
    except (json.JSONDecodeError, TypeError):
        contract = None
    fixed = {
        "schema": 1,
        "engine": "flint-native",
        "os": "linux",
        "arch": "amd64",
        "backend": "hip",
        "revision": "3b93f9053b530488405ae9fca51d4900d6ce21f0",
        "max_context": 262144,
        "arena_base_bytes": 25074426720,
        "kv_bytes_per_token": 28672,
        "sampler_bytes": 1951872,
    }
    source_keys = {"source_commit", "source_tree", "source_clean"}
    if (
        contract_probe.returncode != 0
        or not isinstance(contract, dict)
        or set(contract) != set(fixed) | source_keys
        or any(contract.get(key) != value for key, value in fixed.items())
        or contract.get("source_clean") is not True
        or not all(re.fullmatch(r"[0-9a-f]{40}", str(contract.get(key, ""))) for key in ("source_commit", "source_tree"))
    ):
        os.close(fd)
        parser.error("executable runtime contract is missing, dirty, or incompatible")
    if identity(os.fstat(fd)) != identity(opened):
        os.close(fd)
        parser.error("--flint-serve changed while being probed")

    requested_output = args.output
    if requested_output.exists() or requested_output.is_symlink():
        parser.error("--output must not already exist")
    output = requested_output.resolve()
    output.mkdir(parents=True, mode=0o700)
    destination = output / "flint-serve"
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    with destination.open("xb") as staged:
        while chunk := os.read(fd, 1024 * 1024):
            staged.write(chunk)
            digest.update(chunk)
    if identity(os.fstat(fd)) != identity(opened):
        os.close(fd)
        parser.error("--flint-serve changed while being staged")
    os.close(fd)
    destination.chmod(0o755)
    binary_sha256 = digest.hexdigest()
    (output / "build.json").write_text(
        json.dumps(
            {**{key: fixed[key] for key in ("engine", "os", "arch", "backend")},
             "binary_sha256": binary_sha256, "runtime_contract": contract},
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
