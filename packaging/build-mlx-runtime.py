#!/usr/bin/env python3
"""Build the relocatable MLX runtime: standalone CPython 3.14.10 with the rec-mlx engine.

Run on an Apple Silicon Mac. Use a standalone CPython such as uv's managed Python,
not a system Python or a virtualenv."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "engine"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--python-root",
        type=Path,
        required=True,
        help="standalone CPython 3.14.10 installation to copy",
    )
    parser.add_argument("--output", type=Path, required=True, help="new directory for the runtime")
    parser.add_argument("--uv", default="uv", help="uv executable (default: %(default)s)")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    source_python = args.python_root / "bin/python3.14"
    information = json.loads(
        subprocess.check_output(
            [
                str(source_python),
                "-I",
                "-c",
                "import json,platform,sys; print(json.dumps([platform.system(),platform.machine(),sys.prefix,sys.base_prefix,list(sys.version_info[:3])]))",
            ],
            text=True,
        )
    )
    if information[:2] != ["Darwin", "arm64"] or information[2] != information[3]:
        parser.error("build on Darwin ARM64 with standalone CPython 3.14.10, not a virtualenv")
    if information[4] != [3, 14, 10]:
        parser.error("runtime requires CPython 3.14.10")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".mlx-build-") as td:
        wheels = Path(td) / "wheels"
        subprocess.run(
            [
                args.uv,
                "build",
                "--wheel",
                "--python",
                str(source_python),
                "--out-dir",
                str(wheels),
                str(ENGINE),
            ],
            check=True,
        )
        (wheel,) = wheels.glob("rec_mlx-*.whl")
        stage = Path(td) / "runtime"
        stage.mkdir()
        # Dereference the distribution's internal links; rec never extracts symlinks.
        shutil.copytree(
            args.python_root,
            stage / "python",
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        python = stage / "python/bin/python3.14"
        subprocess.run(
            [
                args.uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--break-system-packages",
                str(wheel),
            ],
            check=True,
        )
        launcher = stage / "rec-mlx"
        launcher.write_text(
            '#!/bin/sh\nset -eu\nroot=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\nexec "$root/python/bin/python3.14" -I -B -m rec_mlx.cli "$@"\n'
        )
        launcher.chmod(0o755)
        (stage / "build.json").write_text(
            json.dumps(
                {
                    "engine": "mlx",
                    "os": "darwin",
                    "arch": "arm64",
                    "backend": "metal",
                },
                indent=2,
            )
            + "\n"
        )
        relocated = Path(td) / "relocated"
        stage.rename(relocated)
        subprocess.run([str(relocated.resolve() / "rec-mlx"), "--help"], check=True)
        subprocess.run(
            [
                str(relocated.resolve() / "python/bin/python3.14"),
                "-I",
                "-B",
                "-c",
                "import mlx.core, mlx_lm.server, rec_mlx.scheduler",
            ],
            check=True,
        )
        # Bytecode is disposable and may embed build paths or source mtimes.
        # Never package it: rewriting a pinned .pyc breaks runtime verification.
        for cached in relocated.rglob("__pycache__"):
            shutil.rmtree(cached)
        for bytecode in relocated.rglob("*.pyc"):
            bytecode.unlink()
        relocated.rename(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
