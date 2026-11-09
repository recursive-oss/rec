import json
import tempfile
import unittest
from pathlib import Path

from rec_mlx.checkpoint import validate


class CheckpointTests(unittest.TestCase):
    def fixture(self, directory, bits):
        # Minimal metadata and placeholder shards exercise validation without loading model weights.
        quantization = {"group_size": 64, "bits": 4}
        if bits == 3:
            # Mixed exports override embeddings to 8 bits and routed experts to 3 bits.
            quantization["model.embed_tokens"] = {"group_size": 64, "bits": 8}
            for projection in ("gate_proj", "up_proj", "down_proj"):
                quantization[f"model.layers.1.mlp.switch_mlp.{projection}"] = {
                    "group_size": 64,
                    "bits": 3,
                }
        config = {
            "sliding_window": 513,
            "max_position_embeddings": 262144,
            "num_hidden_layers": 2,
            "mlp_only_layers": [0],
            "mlx_export": {"bits": 4, "routed_expert_bits": bits, "embedding_bits": 8},
            "quantization": quantization,
            "quantization_config": quantization,
        }
        (directory / "config.json").write_text(json.dumps(config))
        names = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model-00001.safetensors",
        ]
        for name in names[1:]:
            (directory / name).write_text("{}")
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {"model.weight": "model-00001.safetensors"},
                }
            )
        )
        profile = directory / "flint-int4.json"
        profile.write_text(
            json.dumps(
                {
                    "weight_format": "mlx-affine-int4",
                    "kv_format": "bf16",
                    "checkpoint_files": dict.fromkeys(names),
                    "max_context": 262144,
                }
            )
        )
        return config, profile if bits == 4 else directory / "config.json"

    def test_both_checkpoint_formats(self):
        # Official INT4 uses a manifest; mixed exports describe their layout in config.json.
        for bits, expected in ((4, "mlx-affine-int4"), (3, "mlx-affine-int3-int4")):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                _, profile = self.fixture(directory, bits)
                identity = validate(directory, profile)
                self.assertEqual(identity["weight_format"], expected)
                self.assertEqual(identity["max_context"], 262144)
                if bits == 3:
                    self.assertEqual(identity["model_id"], "flint-mixed-3-4bit")

    def test_mixed_rejects_wrong_layout(self):
        # Reject metadata that disagrees with the quantization layout expected by the kernels.
        for mutation in ("expert_bits", "group_size", "output_bits", "conflicting_config"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                config, profile = self.fixture(directory, 3)
                if mutation == "expert_bits":
                    config["mlx_export"]["routed_expert_bits"] = 2
                elif mutation == "group_size":
                    config["quantization"]["group_size"] = 128
                elif mutation == "output_bits":
                    config["quantization"]["lm_head"] = {"bits": 8, "group_size": 64}
                else:
                    config["quantization_config"] = {"bits": 2, "group_size": 64}
                profile.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    validate(directory, profile)

    def test_oq3e_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config, profile = self.fixture(directory, 3)
            config.pop("mlx_export")
            config["quantization"] = {"group_size": 64, "bits": 3, "mode": "affine",
                "language_model.model.layers.0.self_attn.q_proj": {"bits": 6, "group_size": 64},
                "language_model.model.embed_tokens": {"bits": 8, "group_size": 64}}
            config["quantization_config"] = config["quantization"]
            profile.write_text(json.dumps(config))
            (directory/"special_tokens_map.json").unlink()
            self.assertEqual(validate(directory, profile)["weight_format"], "mlx-affine-oq3e")
            config["quantization"]["language_model.model.embed_tokens"]["bits"] = 7
            profile.write_text(json.dumps(config))
            with self.assertRaises(ValueError): validate(directory, profile)

    def test_extra_shard_rejected(self):
        # An unindexed shard can indicate a stale or accidentally combined checkpoint.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, profile = self.fixture(directory, 3)
            (directory / "model-extra.safetensors").touch()
            with self.assertRaisesRegex(ValueError, "shards"):
                validate(directory, profile)
