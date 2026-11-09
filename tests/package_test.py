#!/usr/bin/env python3

import json
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = root / "packaging/flint-int4-mlx.json"
with tempfile.TemporaryDirectory(prefix="rec-package-test-") as td:
    work = Path(td)
    m = json.loads(manifest.read_text())
    runtime = work / "runtime"
    (runtime / "python/bin").mkdir(parents=True)
    (runtime / "build.json").write_text(
        json.dumps({k: m[k] for k in ("engine", "os", "arch", "backend")})
    )
    server = runtime / "rec-mlx"
    server.write_text("#!/bin/sh\nexit 0\n")
    server.chmod(0o755)
    (runtime / "python/bin/README").write_text("nested runtime file\n")
    output = work / "rec-darwin-arm64"
    base = [
        sys.executable,
        str(root / "packaging/package.py"),
        "--manifest",
        str(manifest),
        "--version",
        "0.1.0-test",
        "--output",
        str(output),
    ]
    subprocess.run([*base, "--runtime-dir", str(runtime)], check=True)
    subprocess.run(
        [
            *base,
            "--runtime-dir",
            str(runtime),
            "--low-memory-manifest",
            str(root / "packaging/flint-mixed-3-4bit-mlx.json"),
        ],
        check=True,
    )
    descriptor = json.loads(output.with_name(output.name + ".manifest.json").read_text())
    low = json.loads(output.with_name(output.name + ".low-memory.manifest.json").read_text())
    assert low["entry"] == "config.json" and low["context"] == 262144
    assert low["runtime_sha256"] == descriptor["runtime_sha256"]
    assert low["runtime_files"] == descriptor["runtime_files"]
    assert {f["name"] for f in descriptor["runtime_files"]} == {
        "rec-mlx",
        "python/bin/README",
    }, descriptor["runtime_files"]
    sha, size = output.with_name(output.name + ".meta").read_text().split()
    assert int(size) == output.stat().st_size
    (runtime / "build.json").write_text(
        json.dumps({**{k: m[k] for k in ("os", "arch", "backend")}, "engine": "other"})
    )
    rejected = subprocess.run(
        [*base, "--runtime-dir", str(runtime)], capture_output=True, text=True
    )
    assert rejected.returncode != 0 and "engine" in rejected.stderr, rejected.stderr
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        result = subprocess.run([str(output), "doctor"], text=True, capture_output=True)
        # CI Macs may fail the hardware requirement after the profile is reported.
        assert re.search(r"Engine +mlx \(metal\)", result.stdout), result.stdout + result.stderr
        for quant, revision in (("3/4", "flint-mixed-3-4bit"), ("oq3e", "flint-oq3e")):
            selected = subprocess.run([str(output), "doctor", "--quant", quant], text=True, capture_output=True)
            assert revision in selected.stdout, selected.stdout + selected.stderr
            assert "262,144" in selected.stdout, selected.stdout + selected.stderr
        assert (
            subprocess.check_output([str(output), "--version"], text=True).strip()
            == "rec 0.1.0-test"
        )
    print("Packaging, archive validation and runtime/manifest mismatch rejection passed")
