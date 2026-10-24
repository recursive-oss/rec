#!/usr/bin/env python3
"""Build an experimental ROCm development runtime using an external pinned Python."""

import argparse
import json
import shutil
from pathlib import Path


def build(output):
    root = Path(__file__).resolve().parents[1]
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    source = root / "engine/src"
    library = output / "lib"
    # These modules share the upstream template, protocols and sampling contract.
    # MLX imports are lazy and the ROCm entry point does not instantiate its engine.
    for package in ("rec_rocm", "rec_mlx"):
        shutil.copytree(
            source / package,
            library / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    launcher = output / "rec-rocm"
    launcher.write_text(
        "#!/bin/sh\nset -eu\n"
        'rec_runtime_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
        'export PYTHONPATH="$rec_runtime_dir/lib${PYTHONPATH:+:$PYTHONPATH}"\n'
        'exec "${REC_ROCM_PYTHON:-python3}" -m rec_rocm.cli "$@"\n'
    )
    launcher.chmod(0o755)
    target = {"engine": "rocm", "os": "linux", "arch": "amd64", "backend": "hip"}
    (output / "build.json").write_text(json.dumps(target, indent=2) + "\n")
    manifest = json.loads((root / "packaging/flint-int4-mlx.json").read_text())
    manifest.update(
        **target,
        development=True,
        revision="flint-int4-3b93f905-rocm10",
        context=8192,
        max_context=33536,
        min_ram_bytes=24 * 1024**3,
    )
    manifest_path = output.with_name(output.name + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = build(args.output.resolve())
    except ValueError as error:
        parser.error(str(error))
    print(f"Runtime: {args.output}")
    print(f"Model manifest: {manifest}")
    print("Development runtime only; external pinned Python/ROCm environment required.")


if __name__ == "__main__":
    main()
