#!/usr/bin/env python3
"""Derive the development smoke runtime and manifest from a built MLX runtime.

The smoke runtime is the Flint runtime with tests/data/smoke-mlx-server.py as its
server, serving the small stock model in tests/data/smoke-mlx.json. Package it with:

  packaging/package.py --runtime-dir OUT/runtime --manifest OUT/smoke-mlx.json ...
"""

import argparse
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        required=True,
        help="runtime directory from packaging/build-mlx-runtime.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory for the smoke runtime and manifest",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    runtime = args.output / "runtime"
    shutil.copytree(args.runtime, runtime, symlinks=False)
    shutil.copy2(root / "tests/data/smoke-mlx-server.py", runtime / "smoke_server.py")
    launcher = runtime / "rec-mlx"
    launcher.write_text(
        '#!/bin/sh\nset -eu\nroot=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
        'exec "$root/python/bin/python3.14" -I -B "$root/smoke_server.py" "$@"\n'
    )
    launcher.chmod(0o755)
    shutil.copy2(root / "tests/data/smoke-mlx.json", args.output / "smoke-mlx.json")
    print(args.output)


if __name__ == "__main__":
    main()
