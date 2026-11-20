import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from .cache import FixedKVCache, FixedSlidingKVCache
from .model import Model, ModelArgs
from .quantized import projections


class BF16Model:
    """Load the original BF16 Flint checkpoint for the M5 Ultra native MLX path."""

    context_limit = 262144

    def __init__(self, directory: Path):
        self.config = json.loads((directory / "config.json").read_text())
        if self.config.get("torch_dtype") != "bfloat16":
            raise ValueError("The BF16 engine requires the original BF16 Flint checkpoint")
        self.config["kv_key_bits"] = 8
        self.config["kv_value_bits"] = 4
        self.config["kv_group_size"] = 64
        native = Model(ModelArgs.from_dict(self.config))
        weights = {}
        for path in sorted(directory.glob("model-*.safetensors")):
            shard = mx.load(str(path))
            duplicate = weights.keys() & shard.keys()
            if duplicate:
                raise ValueError(f"Duplicate tensor in {path}: {next(iter(duplicate))}")
            weights.update(shard)
        native.load_weights(list(native.sanitize(weights).items()), strict=True)
        native.eval()
        mx.eval(native.parameters())
        self.native = native
        self.embedding = native.model.embed_tokens
        self.head = native.lm_head
        self.norm = native.model.norm.weight
        self.layers = []
        for index, layer in enumerate(native.layers):
            attention = layer.self_attn
            a = SimpleNamespace(
                q=attention.q_proj,
                k=attention.k_proj,
                v=attention.v_proj,
                o=attention.o_proj,
                q_norm=attention.q_norm.weight,
                k_norm=attention.k_norm.weight,
                sinks=attention.softmax_offset.astype(mx.bfloat16),
                is_global=not attention.use_rope,
            )
            if index in self.config["mlp_only_layers"]:
                mlp = projections(layer.mlp)
            else:
                mlp = projections(layer.mlp.switch_mlp)
                mlp.shared = projections(layer.mlp.shared_expert)
                mlp.router_weight = layer.mlp.gate.weight.astype(mx.float32)
                mlp.expert_bias = layer.mlp.gate.expert_bias.astype(mx.float32)
            self.layers.append(
                SimpleNamespace(
                    attention=a,
                    mlp=mlp,
                    input_norm=layer.input_layernorm.weight,
                    post_norm=layer.post_attention_layernorm.weight,
                )
            )
        self.eos_tokens = self.config["eos_token_id"]
        mx.clear_cache()

    def make_cache(self, max_sequence_tokens, storage):
        dimensions = {
            "max_sequence_tokens": max_sequence_tokens,
            "num_kv_heads": self.config["num_key_value_heads"],
            "head_dim": self.config["head_dim"],
        }
        return [
            FixedKVCache(storage, index)
            if layer.attention.is_global
            else FixedSlidingKVCache(**dimensions, window_size=self.config["sliding_window"])
            for index, layer in enumerate(self.layers)
        ]

    def __call__(self, tokens, cache):
        return self.native(tokens, cache)[:, -1:]
