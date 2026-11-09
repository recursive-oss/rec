#!/usr/bin/env python3
"""Build a rec executable with an embedded runtime and model manifest."""

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def payload(runtime_dir, manifest):
    m = json.loads(manifest.read_text())
    metadata = json.loads((runtime_dir / "build.json").read_text())
    for key in ("engine", "os", "arch", "backend"):
        if metadata[key] != m[key]:
            raise ValueError(f"runtime and model disagree on {key}")
    files = []
    buf = io.BytesIO()
    # Fixed gzip timestamp makes equivalent payloads content-identical.
    compressed = gzip.GzipFile(fileobj=buf, mode="wb", mtime=0)
    with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for source in sorted(runtime_dir.rglob("*")):
            if source.is_symlink():
                raise ValueError(f"runtime payload cannot contain symlinks: {source}")
            if source == runtime_dir / "build.json" or source.is_dir():
                continue
            if not source.is_file():
                raise ValueError(f"runtime payload must contain regular files: {source}")
            data = source.read_bytes()
            executable = bool(source.stat().st_mode & 0o111)
            name = source.relative_to(runtime_dir).as_posix()
            files.append(dict(name=name, size=len(data), sha256=sha(data), executable=executable))
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o700 if executable else 0o600
            tar.addfile(info, io.BytesIO(data))
    compressed.close()
    m["runtime_files"] = files
    m["runtime_sha256"] = sha(buf.getvalue())
    return m, buf.getvalue()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runtime-dir",
        type=Path,
        required=True,
        help="runtime directory from packaging/build-mlx-runtime.py",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="model manifest, such as packaging/flint-int4-mlx.json",
    )
    p.add_argument(
        "--low-memory-manifest", type=Path, help="alternate manifest for Macs below 48 GiB"
    )
    p.add_argument(
        "--version",
        default="0.1.0-dev",
        help="version reported by rec --version (default: %(default)s)",
    )
    p.add_argument("--oq3e-manifest", type=Path, help="oQ3e checkpoint descriptor")
    p.add_argument("--output", type=Path, required=True, help="path of the rec executable to write")
    a = p.parse_args()
    # Existing release commands already select the small-memory descriptor.
    # Include its sibling oQ3e profile without requiring workflow changes.
    if a.oq3e_manifest is None and a.low_memory_manifest:
        sibling = a.low_memory_manifest.with_name("flint-oq3e-mlx.json")
        if sibling.is_file():
            a.oq3e_manifest = sibling
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", a.version):
        p.error("invalid version")
    try:
        m, archive = payload(a.runtime_dir, a.manifest)
    except ValueError as error:
        p.error(str(error))
    low = None
    if a.low_memory_manifest:
        low = json.loads(a.low_memory_manifest.read_text())
        for key in ("engine", "os", "arch", "backend", "model"):
            if low[key] != m[key]:
                p.error(f"low-memory manifest disagrees on {key}")
        low["runtime_files"] = m["runtime_files"]
        low["runtime_sha256"] = m["runtime_sha256"]
    root = Path(__file__).resolve().parents[1]
    output = a.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rec-package-") as td:
        build = Path(td)
        shutil.copy2(root / "go.mod", build / "go.mod")
        shutil.copy2(root / "go.sum", build / "go.sum")
        shutil.copytree(root / "cli", build / "cli")
        assets = build / "cli/assets"
        shutil.rmtree(assets)
        assets.mkdir()
        (assets / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
        (assets / "runtime.tar.gz").write_bytes(archive)
        if low is not None:
            (assets / "manifest-low-memory.json").write_text(json.dumps(low, indent=2) + "\n")
        if a.oq3e_manifest:
            oq = json.loads(a.oq3e_manifest.read_text())
            for key in ("engine", "os", "arch", "backend", "model"):
                if oq[key] != m[key]: p.error(f"oQ3e manifest disagrees on {key}")
            oq["runtime_files"] = m["runtime_files"]
            oq["runtime_sha256"] = m["runtime_sha256"]
            (assets / "manifest-oq3e.json").write_text(json.dumps(oq, indent=2)+"\n")
        subprocess.run(["go", "test", "./cli", "-run", "TestEmbedded"], cwd=build, check=True)
        env = dict(os.environ, CGO_ENABLED="0", GOOS=m["os"], GOARCH=m["arch"])
        staged = build / "rec"
        subprocess.run(
            [
                "go",
                "build",
                "-trimpath",
                "-ldflags",
                f"-s -w -X main.Version={a.version}",
                "-o",
                str(staged),
                "./cli",
            ],
            cwd=build,
            env=env,
            check=True,
        )
        # Replace the inode atomically: overwriting a running Mach-O can leave
        # macOS with stale code-signature pages and cause SIGKILL on launch.
        replacement = output.with_name(output.name + ".new")
        shutil.copy2(staged, replacement)
        os.replace(replacement, output)
    output.chmod(0o755)
    data = output.read_bytes()
    output.with_name(output.name + ".meta").write_text(f"{sha(data)} {len(data)}\n")
    output.with_name(output.name + ".manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    if low is not None:
        output.with_name(output.name + ".low-memory.manifest.json").write_text(
            json.dumps(low, indent=2) + "\n"
        )
    print(output)


if __name__ == "__main__":
    main()
