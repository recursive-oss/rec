import json
import tempfile
import unittest
from pathlib import Path

from rec_mlx.checkpoint import validate


class CheckpointValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _setup_files(self, weight_format, bits):
        manifest_data = {
            "model_id": f"flint-{weight_format.split('-')[-1]}",
            "weight_format": weight_format,
            "kv_format": "bf16",
            "max_context": 262144,
            "checkpoint_files": {
                "config.json": "dummy",
                "tokenizer.json": "dummy",
                "tokenizer_config.json": "dummy",
                "special_tokens_map.json": "dummy",
                "model-00000.safetensors": "dummy",
            },
        }
        manifest_path = self.directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))

        config_data = {
            "sliding_window": 513,
            "mlx_export": {
                "bits": bits,
                "routed_expert_bits": bits,
                "embedding_bits": 8,
            },
            "quantization": {
                "group_size": 64,
                "bits": bits,
            },
        }
        (self.directory / "config.json").write_text(json.dumps(config_data))
        (self.directory / "tokenizer.json").write_text("{}")
        (self.directory / "tokenizer_config.json").write_text("{}")
        (self.directory / "special_tokens_map.json").write_text("{}")
        (self.directory / "model-00000.safetensors").write_text("")
        return manifest_path

    def test_validate_int4(self):
        manifest_path = self._setup_files("mlx-affine-int4", 4)
        m = validate(self.directory, manifest_path)
        self.assertEqual(m["weight_format"], "mlx-affine-int4")

    def test_validate_int6(self):
        manifest_path = self._setup_files("mlx-affine-int6", 6)
        m = validate(self.directory, manifest_path)
        self.assertEqual(m["weight_format"], "mlx-affine-int6")

    def test_validate_unsupported_format(self):
        manifest_path = self._setup_files("mlx-affine-int8", 8)
        with self.assertRaises(ValueError) as ctx:
            validate(self.directory, manifest_path)
        self.assertIn("requires affine INT4 or INT6 weights", str(ctx.exception))

    def test_validate_mismatched_bits(self):
        manifest_path = self._setup_files("mlx-affine-int6", 4)
        with self.assertRaises(ValueError) as ctx:
            validate(self.directory, manifest_path)
        self.assertIn("Expected INT6 routed experts", str(ctx.exception))

    def test_validate_jang_int6(self):
        config_data = {
            "sliding_window": 513,
            "quantization": {
                "group_size": 64,
                "bits": 8,
                "model.embed_tokens": {"group_size": 64, "bits": 8, "mode": "affine"},
                "model.layers.2.mlp.switch_mlp.gate_proj": {"group_size": 64, "bits": 6, "mode": "affine"},
            },
        }
        config_path = self.directory / "config.json"
        config_path.write_text(json.dumps(config_data))
        (self.directory / "tokenizer.json").write_text("{}")
        (self.directory / "tokenizer_config.json").write_text("{}")
        (self.directory / "model-00000.safetensors").write_text("")
        m = validate(self.directory, config_path)
        self.assertEqual(m["weight_format"], "mlx-affine-int6")
        self.assertEqual(m["model_id"], "flint-int6")


if __name__ == "__main__":
    unittest.main()
