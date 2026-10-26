import json
from pathlib import Path


def validate(directory: Path, manifest_path: Path) -> dict:
    raw = json.loads(manifest_path.read_text())
    config = json.loads((directory / "config.json").read_text())
    if "weight_format" in raw:
        manifest = raw
    else:
        # Standard Hugging Face MLX checkpoint where manifest is config.json
        q = config.get("quantization", {})
        # Check if routed experts specify 6-bit or if base is 6-bit
        bits = None
        for k, v in q.items():
            if isinstance(v, dict) and "bits" in v and "switch_mlp" in k:
                bits = v["bits"]
                break
        if bits is None:
            bits = q.get("bits", 4)
        shards = {p.name: "" for p in directory.glob("model-*.safetensors")}
        manifest = {
            "model_id": f"flint-int{bits}",
            "weight_format": f"mlx-affine-int{bits}",
            "kv_format": "bf16",
            "max_context": config.get("max_position_embeddings", 262144),
            "checkpoint_files": {
                "config.json": "",
                "tokenizer.json": "",
                "tokenizer_config.json": "",
                **shards,
            },
        }
    if manifest["weight_format"] not in ("mlx-affine-int4", "mlx-affine-int6") or manifest["kv_format"] != "bf16":
        raise ValueError("This engine requires affine INT4 or INT6 weights and BF16 KV")
    files = manifest["checkpoint_files"]
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if "special_tokens_map.json" in files:
        required.add("special_tokens_map.json")
    shards = {name for name in files if name.startswith("model-") and name.endswith(".safetensors")}
    if not required.issubset(files) or not shards:
        raise ValueError("Manifest must include configuration, tokenizer and weight shards")
    # The upstream loader discovers shards by glob; an unpinned extra shard must not be loaded.
    if {p.name for p in directory.glob("model-*.safetensors")} != shards:
        raise ValueError("Checkpoint shards do not match the manifest")
    # rec verifies every file's SHA-256 before it starts the engine.
    for name in files:
        if Path(name).name != name:
            raise ValueError(f"Checkpoint filename must be a basename: {name}")
        if not (directory / name).is_file():
            raise ValueError(f"Checkpoint is missing a manifest file: {name}")
    if config["sliding_window"] != 513:
        raise ValueError("Flint requires sliding_window=513, including the current token")
    expected_bits = 6 if manifest["weight_format"] == "mlx-affine-int6" else 4
    if expected_bits == 6:
        # Check if routed experts are 6-bit or mlx_export specifies 6-bit routed experts
        export = config.get("mlx_export")
        if export is not None and export.get("routed_expert_bits") != 6:
            raise ValueError("Expected INT6 routed experts")
        q = config.get("quantization", {})
        has_6bit = any(
            isinstance(v, dict) and v.get("bits") == 6
            for k, v in q.items()
        ) or q.get("bits") == 6
        if not has_6bit:
            raise ValueError("Expected INT6 quantization for routed experts or model")
    else:
        export = config.get("mlx_export", {
            "bits": 4,
            "routed_expert_bits": 4,
            "embedding_bits": 8,
        })
        if (export.get("bits"), export.get("routed_expert_bits"), export.get("embedding_bits")) != (4, 4, 8):
            raise ValueError("Expected INT4 projections/experts and INT8 embeddings")
        if config.get("quantization", {}).get("bits") != 4:
            raise ValueError("Expected group-64 INT4 quantization")
    return manifest
