#!/usr/bin/env python3
"""Stage a relocatable native Flint executable for REC packaging."""

import argparse
import json
import shutil
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
    probe = subprocess.run(
        [str(source.resolve()), "--help"], capture_output=True, text=True, timeout=30
    )
    if probe.returncode != 0 or "flint-serve" not in probe.stdout or "--checkpoint-dir" not in probe.stdout:
        parser.error("executable does not satisfy the native Flint server contract")

    output = args.output.resolve()
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            parser.error("--output must be a directory, not a symlink")
        shutil.rmtree(output)
    output.mkdir(parents=True, mode=0o700)
    destination = output / "flint-serve"
    shutil.copyfile(source, destination)
    destination.chmod(0o755)
    (output / "build.json").write_text(
        json.dumps(
            {"engine": "flint-native", "os": "linux", "arch": "amd64", "backend": "hip"},
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
