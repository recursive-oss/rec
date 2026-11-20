#!/usr/bin/env python3
"""Write the rec model manifest for a published MLX checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="local copy of the published checkpoint"
    )
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="checkpoint manifest with the expected file hashes",
    )
    parser.add_argument(
        "--base-url", required=True, help="immutable HTTPS directory that serves every listed file"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="path of the model manifest to write"
    )
    parser.add_argument(
        "--context", type=int, help="default total context tokens (default: the checkpoint maximum)"
    )
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    if "config.json" not in profile["checkpoint_files"]:
        parser.error("profile must include config.json")
    maximum = profile["max_context"]
    if type(maximum) is not int or maximum < 256:
        parser.error("profile must declare an integer max_context of at least 256")
    context = maximum if args.context is None else args.context
    if not 256 <= context <= maximum:
        parser.error(f"context must be between 256 and {maximum}")
    files = []
    embedded_profile = profile.get("embedded_profile", False)
    if not embedded_profile and args.profile.name in profile["checkpoint_files"]:
        parser.error("profile basename collides with a model file")
    for name, expected in profile["checkpoint_files"].items():
        if Path(name).name != name:
            parser.error("checkpoint filenames must be basenames")
        path = args.model_dir / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected:
            parser.error(f"checkpoint hash mismatch: {name}")
        files.append(
            dict(
                name=name,
                url=args.base_url.rstrip("/") + "/" + quote(name),
                size=path.stat().st_size,
                sha256=digest,
            )
        )
    manifest = dict(
        schema=1,
        model="flint",
        revision=profile["model_id"],
        development=False,
        engine="mlx",
        os="darwin",
        arch="arm64",
        backend="metal",
        context=context,
        max_context=maximum,
        min_ram_bytes=profile.get("min_ram_bytes", 48 * 1024**3),
        entry=profile.get("entry", args.profile.name),
        files=files,
    )
    if embedded_profile:
        manifest.update(
            model_id=profile["model_id"],
            weight_format=profile["weight_format"],
            kv_format=profile["kv_format"],
        )
    else:
        files.append(
            dict(
                name=args.profile.name,
                url=args.base_url.rstrip("/") + "/" + quote(args.profile.name),
                size=args.profile.stat().st_size,
                sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
            )
        )
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
