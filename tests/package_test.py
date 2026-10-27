#!/usr/bin/env python3

import hashlib
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
    descriptor = json.loads(output.with_name(output.name + ".manifest.json").read_text())
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
        assert (
            subprocess.check_output([str(output), "--version"], text=True).strip()
            == "rec 0.1.0-test"
        )
    fake_flint = work / "source-flint-serve"
    fake_flint.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = --help ]; then echo 'flint-serve --checkpoint-dir'; exit 0; fi\n"
        "exit 1\n"
    )
    fake_flint.chmod(0o755)
    rejected = subprocess.run(
        [sys.executable, str(root / "packaging/build-flint-runtime.py"),
         "--flint-serve", str(fake_flint), "--output", str(work / "bad-contract-runtime")],
        capture_output=True, text=True,
    )
    assert rejected.returncode != 0 and "contract" in rejected.stderr, rejected.stderr
    fake_flint.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = --help ]; then echo 'flint-serve --checkpoint-dir'; exit 0; fi\n"
        "if [ \"$1\" = --runtime-contract ]; then echo '{\"schema\":1,\"engine\":\"flint-native\",\"os\":\"linux\",\"arch\":\"amd64\",\"backend\":\"hip\",\"revision\":\"3b93f9053b530488405ae9fca51d4900d6ce21f0\",\"max_context\":262144,\"arena_base_bytes\":25074426720,\"kv_bytes_per_token\":28672,\"sampler_bytes\":1951872,\"source_commit\":\"0123456789abcdef0123456789abcdef01234567\",\"source_tree\":\"89abcdef0123456789abcdef0123456789abcdef\",\"source_clean\":true}'; exit 0; fi\n"
        "exit 1\n"
    )
    hip_runtime = work / "hip-runtime"
    subprocess.run(
        [sys.executable, str(root / "packaging/build-flint-runtime.py"),
         "--flint-serve", str(fake_flint), "--output", str(hip_runtime)],
        check=True,
    )
    identity = json.loads((hip_runtime / "build.json").read_text())
    target = {key: identity[key] for key in ("engine", "os", "arch", "backend")}
    assert target == {"engine": "flint-native", "os": "linux", "arch": "amd64", "backend": "hip"}
    assert identity["binary_sha256"] == hashlib.sha256((hip_runtime / "flint-serve").read_bytes()).hexdigest()
    assert identity["runtime_contract"]["source_clean"] is True
    assert (hip_runtime / "flint-serve").stat().st_mode & 0o111
    sentinel = hip_runtime / "keep-me"
    sentinel.write_text("do not delete\n")
    rejected = subprocess.run(
        [sys.executable, str(root / "packaging/build-flint-runtime.py"),
         "--flint-serve", str(fake_flint), "--output", str(hip_runtime)],
        capture_output=True, text=True,
    )
    assert rejected.returncode != 0 and sentinel.read_text() == "do not delete\n", rejected.stderr
    bad = work / "flint-link"
    bad.symlink_to(fake_flint)
    rejected = subprocess.run(
        [sys.executable, str(root / "packaging/build-flint-runtime.py"),
         "--flint-serve", str(bad), "--output", str(work / "bad-runtime")],
        capture_output=True, text=True,
    )
    assert rejected.returncode != 0 and "symlink" in rejected.stderr, rejected.stderr
    print("Packaging, archive validation and runtime/manifest mismatch rejection passed")
