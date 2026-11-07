import json
from pathlib import Path


def validate(directory: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    config = json.loads((directory / "config.json").read_text())
    # Community checkpoints use a Hugging Face config and shard index rather
    # than Recursive's separate profile. The CLI pins and verifies both files.
    if manifest_path.name == "config.json":
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        manifest = {
            "model_id": "flint-mixed-3-4bit",
            "weight_format": "mlx-affine-int3-int4",
            "kv_format": "bf16",
            "max_context": config["max_position_embeddings"],
            "checkpoint_files": {
                name: None
                for name in {
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "model.safetensors.index.json",
                    *index["weight_map"].values(),
                }
            },
        }
    formats = {"mlx-affine-int4": 4, "mlx-affine-int3-int4": 3}
    if manifest["weight_format"] not in formats or manifest["kv_format"] != "bf16":
        raise ValueError("This engine requires affine INT4 or mixed INT3/INT4 weights and BF16 KV")
    files = manifest["checkpoint_files"]
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
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
    export = config["mlx_export"]
    expert_bits = formats[manifest["weight_format"]]
    if (export["bits"], export["routed_expert_bits"], export["embedding_bits"]) != (
        4,
        expert_bits,
        8,
    ):
        raise ValueError("Expected INT4 projections, INT3 or INT4 experts, and INT8 embeddings")
    expected = {"group_size": 64, "bits": 4}
    if expert_bits == 3:
        expected["model.embed_tokens"] = {"group_size": 64, "bits": 8}
        for layer in range(config["num_hidden_layers"]):
            if layer not in config["mlp_only_layers"]:
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    expected[f"model.layers.{layer}.mlp.switch_mlp.{projection}"] = {
                        "group_size": 64,
                        "bits": 3,
                    }
    if config["quantization"] != expected:
        raise ValueError("Unexpected group-64 affine quantization layout")
    if "quantization_config" in config and config["quantization_config"] != expected:
        raise ValueError("Conflicting quantization configurations")
    return manifest
