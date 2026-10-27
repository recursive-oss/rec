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


def validate_native_runtime(runtime_dir, manifest, metadata):
    if manifest["engine"] != "flint-native":
        return
    server = runtime_dir / "flint-serve"
    if server.is_symlink() or not server.is_file():
        raise ValueError("native runtime contract executable is missing or unsafe")
    expected_keys = {"engine", "os", "arch", "backend", "binary_sha256", "runtime_contract"}
    if set(metadata) != expected_keys:
        raise ValueError("native runtime contract metadata is incomplete")
    if metadata["binary_sha256"] != sha(server.read_bytes()):
        raise ValueError("native runtime contract binary digest mismatch")
    contract = metadata["runtime_contract"]
    fixed = {
        "schema": 1, "engine": "flint-native", "os": "linux", "arch": "amd64",
        "backend": "hip", "revision": manifest["revision"],
        "max_context": manifest["max_context"], "arena_base_bytes": 25074426720,
        "kv_bytes_per_token": 28672, "sampler_bytes": 1951872,
    }
    source_keys = {"source_commit", "source_tree", "source_clean"}
    if (not isinstance(contract, dict) or set(contract) != set(fixed) | source_keys
            or any(contract.get(key) != value for key, value in fixed.items())
            or contract.get("source_clean") is not True
            or not all(re.fullmatch(r"[0-9a-f]{40}", str(contract.get(key, "")))
                       for key in ("source_commit", "source_tree"))):
        raise ValueError("native runtime contract is incompatible")
    probe = subprocess.run([str(server.resolve()), "--runtime-contract"],
                           capture_output=True, text=True, timeout=30)
    try:
        live = json.loads(probe.stdout)
    except json.JSONDecodeError:
        live = None
    if probe.returncode != 0 or live != contract:
        raise ValueError("native runtime contract probe disagrees with metadata")


def payload(runtime_dir, manifest):
    m = json.loads(manifest.read_text())
    metadata = json.loads((runtime_dir / "build.json").read_text())
    for key in ("engine", "os", "arch", "backend"):
        if metadata[key] != m[key]:
            raise ValueError(f"runtime and model disagree on {key}")
    validate_native_runtime(runtime_dir, m, metadata)
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
        help="prebuilt runtime directory matching the manifest target",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="model manifest, such as packaging/flint-int4-mlx.json",
    )
    p.add_argument(
        "--version",
        default="0.1.0-dev",
        help="version reported by rec --version (default: %(default)s)",
    )
    p.add_argument("--output", type=Path, required=True, help="path of the rec executable to write")
    a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", a.version):
        p.error("invalid version")
    try:
        m, archive = payload(a.runtime_dir, a.manifest)
    except ValueError as error:
        p.error(str(error))
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
        subprocess.run(
            ["go", "test", "./cli", "-run", "TestEmbeddedManifest"], cwd=build, check=True
        )
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
        shutil.copy2(staged, output)
    output.chmod(0o755)
    data = output.read_bytes()
    output.with_name(output.name + ".meta").write_text(f"{sha(data)} {len(data)}\n")
    output.with_name(output.name + ".manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
