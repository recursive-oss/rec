#!/usr/bin/env python3
"""Build the relocatable MLX runtime: standalone CPython 3.12 with the rec-mlx engine.

Run on an Apple Silicon Mac. Use a standalone CPython such as uv's managed Python,
not a system Python or a virtualenv."""

import argparse
import json
import os
import re
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
        help="standalone CPython 3.12 installation to copy",
    )
    parser.add_argument("--output", type=Path, required=True, help="new directory for the runtime")
    parser.add_argument("--uv", default="uv", help="uv executable (default: %(default)s)")
    parser.add_argument(
        "--macos",
        metavar="VERSION",
        help="oldest macOS the runtime supports, such as 15.0 (default: this Mac's version)",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    if args.macos and not re.fullmatch(r"\d+\.\d+", args.macos):
        parser.error("--macos must look like 15.0")
    # MLX publishes a wheel per macOS release; the macOS 26 metallib uses Metal 4 and fails to
    # load on macOS 15. The deployment target selects those wheels and sets the native
    # extension's minimum OS.
    env = dict(os.environ)
    platform = []
    if args.macos:
        env["MACOSX_DEPLOYMENT_TARGET"] = args.macos
        platform = ["--python-platform", "aarch64-apple-darwin"]
    source_python = args.python_root / "bin/python3.12"
    information = json.loads(
        subprocess.check_output(
            [
                str(source_python),
                "-I",
                "-c",
                "import json,platform,sys; print(json.dumps([platform.system(),platform.machine(),sys.prefix,sys.base_prefix]))",
            ],
            text=True,
        )
    )
    if information[:2] != ["Darwin", "arm64"] or information[2] != information[3]:
        parser.error("build on Darwin ARM64 with standalone CPython 3.12, not a virtualenv")
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
            env=env,
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
        python = stage / "python/bin/python3.12"
        subprocess.run(
            [
                args.uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--break-system-packages",
                *platform,
                "--only-binary",
                ":all:",
                str(wheel),
            ],
            env=env,
            check=True,
        )
        launcher = stage / "rec-mlx"
        launcher.write_text(
            '#!/bin/sh\nset -eu\nroot=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\nexec "$root/python/bin/python3.12" -I -B -m rec_mlx.cli "$@"\n'
        )
        launcher.chmod(0o755)
        (stage / "build.json").write_text(
            json.dumps(
                {
                    "engine": "mlx",
                    "os": "darwin",
                    "arch": "arm64",
                    "backend": "metal",
                    **({"macos": args.macos} if args.macos else {}),
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
                str(relocated.resolve() / "python/bin/python3.12"),
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
