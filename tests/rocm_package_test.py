#!/usr/bin/env python3
"""Exercise the ROCm builder, embedded manifest and external-Python launcher."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="rec rocm package ") as td:
    work = Path(td)
    runtime = work / "runtime"
    builder = [
        sys.executable,
        str(root / "packaging/build-rocm-runtime.py"),
        "--output",
        str(runtime),
    ]
    subprocess.run(builder, check=True)
    manifest = runtime.with_name(runtime.name + ".manifest.json")
    descriptor = json.loads(manifest.read_text())
    assert descriptor["engine"] == "rocm" and descriptor["development"]
    assert descriptor["context"] == 8192 and descriptor["max_context"] == 33536
    original = json.loads((root / "packaging/flint-int4-mlx.json").read_text())
    assert descriptor["files"] == original["files"]
    duplicate = subprocess.run(builder, capture_output=True, text=True)
    assert duplicate.returncode != 0 and "already exists" in duplicate.stderr

    # A standalone external interpreter makes the launcher's argv and relocation
    # behavior observable without requiring Torch or a GPU on the CPU CI runner.
    probe = work / "external python"
    probe.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'path': os.environ['PYTHONPATH']}))\n"
    )
    probe.chmod(0o755)
    env = dict(os.environ, REC_ROCM_PYTHON=str(probe), PYTHONPATH="existing-path")
    result = subprocess.check_output(
        [str(runtime / "rec-rocm"), "serve", "flint", "--model-dir", "model with spaces"],
        cwd=work,
        env=env,
        text=True,
    )
    observed = json.loads(result)
    assert observed["argv"] == [
        "-m",
        "rec_rocm.cli",
        "serve",
        "flint",
        "--model-dir",
        "model with spaces",
    ]
    assert observed["path"] == str(runtime / "lib") + ":existing-path"

    output = work / "rec-linux-amd64"
    subprocess.run(
        [
            sys.executable,
            str(root / "packaging/package.py"),
            "--runtime-dir",
            str(runtime),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--version",
            "0.1.0-rocm-test",
        ],
        check=True,
    )
    embedded = json.loads(output.with_name(output.name + ".manifest.json").read_text())
    assert embedded["engine"] == "rocm" and embedded["backend"] == "hip"
    assert any(f["name"] == "rec-rocm" and f["executable"] for f in embedded["runtime_files"])
    assert any(f["name"] == "lib/rec_rocm/model.py" for f in embedded["runtime_files"])
    if sys.platform == "linux":
        assert (
            subprocess.check_output([str(output), "--version"], text=True).strip()
            == "rec 0.1.0-rocm-test"
        )
print("ROCm development runtime, external interpreter and embedded manifest checks passed")
