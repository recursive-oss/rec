"""Small Flint checkpoint, same packed format and hybrid attention semantics."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def make_checkpoint(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    c = {
        "model_type": "flint",
        "quantization": {"bits": 4, "group_size": 64},
        "hidden_size": 64,
        "num_hidden_layers": 3,
        "intermediate_size": 128,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "no_rope_layer_indices": [0],
        "mlp_only_layers": [0],
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention"],
        "sliding_window": 513,
    }
    torch.manual_seed(88)
    w = {}

    def norm(name, shape):
        w[name] = torch.ones(shape, dtype=torch.bfloat16)

    def linear(path, n, k, experts=None, bits=4):
        leading = (experts,) if experts else ()
        w[path + ".weight"] = torch.randint(0, 256, (*leading, n, k * bits // 8), dtype=torch.uint8)
        scale = torch.full((*leading, n, k // 64), 1 / ((2**bits) * k**0.5), dtype=torch.bfloat16)
        w[path + ".scales"] = scale
        w[path + ".biases"] = (-scale * ((2**bits - 1) / 2)).to(torch.bfloat16)

    linear("model.embed_tokens", 128, 64, bits=8)
    linear("lm_head", 128, 64)
    norm("model.norm.weight", (64,))
    for i in range(3):
        p = f"model.layers.{i}"
        for name in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            norm(p + "." + name, (64,))
        for name in ["q_norm.weight", "k_norm.weight"]:
            norm(p + ".self_attn." + name, (32,))
        w[p + ".self_attn.softmax_offset"] = torch.randn(2, dtype=torch.bfloat16)
        for name, n, k in [("q", 64, 64), ("k", 32, 64), ("v", 32, 64), ("o", 64, 64)]:
            linear(p + f".self_attn.{name}_proj", n, k)
        base = p + ".mlp" if i == 0 else p + ".mlp.switch_mlp"
        mid = 128 if i == 0 else 64
        for name, n, k in [("gate", mid, 64), ("up", mid, 64), ("down", 64, mid)]:
            linear(base + f".{name}_proj", n, k, None if i == 0 else 8)
        if i:
            w[p + ".mlp.gate.weight"] = torch.randn(8, 64, dtype=torch.bfloat16) / 8
            for name in ["gate", "up", "down"]:
                linear(p + f".mlp.shared_expert.{name}_proj", 64, 64)
    save_file(w, directory / "model-00000.safetensors")
    (directory / "config.json").write_text(json.dumps(c))
    return directory
