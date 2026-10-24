import math
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, create_causal_mask
from mlx_lm.models.cache import QuantizedKVCache, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchGLU

from .cache import FixedSlidingKVCache
from .prefix import PrefillCancelled


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "flint"
    hidden_size: int = 2048
    intermediate_size: int = 6144
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 1024
    num_hidden_layers: int = 49
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    num_experts: int = 256
    num_experts_per_tok: int = 6
    vocab_size: int = 150144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    sliding_window: int = 513
    layer_types: list = field(default_factory=list)
    no_rope_layer_indices: list = field(default_factory=list)
    mlp_only_layers: list = field(default_factory=lambda: [0, 1])
    tie_word_embeddings: bool = False
    kv_key_bits: int = 8
    kv_value_bits: int = 4
    kv_group_size: int = 64


class MixedQuantizedKVCache(QuantizedKVCache):
    def __init__(self, group_size: int = 64, key_bits: int = 8, value_bits: int = 4):
        super().__init__(group_size=group_size, bits=key_bits)
        self.key_bits, self.value_bits = key_bits, value_bits

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_dim = keys.shape
        v_dim = values.shape[-1]
        prev = self.offset
        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init(dim, bits):
                return (
                    mx.zeros((*shape, dim * bits // 32), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def grow(parts):
                return tuple(
                    mx.concatenate([x, mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)], axis=-2)
                    for x in parts
                )

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = tuple(x[..., :prev, :] for x in self.keys)
                    self.values = tuple(x[..., :prev, :] for x in self.values)
                self.keys, self.values = grow(self.keys), grow(self.values)
            else:
                self.keys, self.values = (
                    init(k_dim, self.key_bits),
                    init(v_dim, self.value_bits),
                )
        self.offset += num_steps
        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.key_bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.value_bits)
        for i in range(3):
            self.keys[i][..., prev : self.offset, :] = q_keys[i]
            self.values[i][..., prev : self.offset, :] = q_values[i]
        return self.keys_and_values()

    @property
    def state(self):
        return (
            self.keys,
            self.values,
            self.offset,
            self.group_size,
            self.key_bits,
            self.value_bits,
        )

    @state.setter
    def state(self, v):
        (
            self.keys,
            self.values,
            self.offset,
            self.group_size,
            self.key_bits,
            self.value_bits,
        ) = v
        self.bits = self.key_bits


def quantized_attention_with_sinks(
    queries, keys, values, scale, mask, sinks, group_size, key_bits, value_bits
):
    B, n_q, L, D = queries.shape
    n_kv = keys[0].shape[-3]
    repeats = n_q // n_kv
    queries = (queries * scale).reshape(B, n_kv, repeats, L, D)
    keys = tuple(mx.expand_dims(x, -3) for x in keys)
    values = tuple(mx.expand_dims(x, -3) for x in values)
    scores = mx.quantized_matmul(
        queries, *keys, transpose=True, group_size=group_size, bits=key_bits
    )
    scores = scores.astype(mx.float32)
    kL = scores.shape[-1]
    if mask is not None:
        if isinstance(mask, str):
            q_idx = mx.arange(kL - L, kL)
            mask = q_idx[:, None] >= mx.arange(kL)[None]
        if mask.ndim > 3:
            mask = mx.expand_dims(mask, -3)
        scores = mx.where(mask, scores, -mx.inf) if mask.dtype == mx.bool_ else scores + mask
    sink = sinks.astype(mx.float32).reshape(1, n_kv, repeats, 1, 1)
    sink = mx.broadcast_to(sink, (*scores.shape[:-1], 1))
    probs = mx.softmax(mx.concatenate([scores, sink], axis=-1), axis=-1)[..., :kL]
    out = mx.quantized_matmul(
        probs.astype(queries.dtype),
        *values,
        transpose=False,
        group_size=group_size,
        bits=value_bits,
    )
    return out.reshape(B, n_q, L, D)


# Key tile starts must stay on a multiple of this so each block reuses the full kernel's tile grid.
# mlx 0.32.0 scaled_dot_product_attention.cpp: key tile bk is 16 for head_dim 128 on non-NAX GPUs (line 199)
# and 32 on NAX GPUs (line 36); query tile bq is 32 / 64 (lines 198, 35). 64 and 256 are multiples of all four.
BAND_ALIGN = 64
BAND_QUERIES = 256
# Queries <= 8 never reach the full kernel: MLX uses sdpa_vector when queries x GQA factor <= 32 and the
# unfused fallback otherwise (use_fallback, lines 632-640).
SDPA_VECTOR_MAX_QUERIES = 8


def banded_sliding_attention(q, k, v, scale, sinks, prior, window):
    """Sliding-window attention over [prior | chunk] keys, one query block at a time.

    Fully masked leading/trailing key tiles add exactly zero to the online softmax, so skipping
    them is bit-identical to the single call with the full window mask.
    """
    L = q.shape[2]
    if L <= SDPA_VECTOR_MAX_QUERIES or 0 < L % BAND_QUERIES <= SDPA_VECTOR_MAX_QUERIES:
        # A tail block this short would leave the full kernel, a different accumulation order.
        return mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            mask=create_causal_mask(L, offset=prior, window_size=window),
            sinks=sinks,
        )
    outs = []
    for s in range(0, L, BAND_QUERIES):
        n = min(BAND_QUERIES, L - s)
        start = max(0, prior + s - window + 1) // BAND_ALIGN * BAND_ALIGN
        stop = prior + s + n
        mask = create_causal_mask(n, offset=prior + s - start, window_size=window)
        outs.append(
            mx.fast.scaled_dot_product_attention(
                q[:, :, s : s + n],
                k[:, :, start:stop],
                v[:, :, start:stop],
                scale=scale,
                mask=mask,
                sinks=sinks,
            )
        )
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer: int):
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim = (
            args.num_attention_heads,
            args.num_key_value_heads,
            args.head_dim,
        )
        self.scale = 1.0 / math.sqrt(args.head_dim)
        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.softmax_offset = mx.zeros((self.n_heads,))
        self.use_rope = layer not in set(args.no_rope_layer_indices)
        self.rope = (
            nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)
            if self.use_rope
            else None
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)).transpose(
            0, 2, 1, 3
        )
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv, self.head_dim)).transpose(
            0, 2, 1, 3
        )
        v = self.v_proj(x).reshape(B, L, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        if self.use_rope:
            offset = cache.offset if cache is not None else 0
            q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        prior = cache._valid if isinstance(cache, FixedSlidingKVCache) else 0
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        if isinstance(cache, FixedSlidingKVCache) and L > 1 and isinstance(mask, mx.array):
            out = banded_sliding_attention(
                q, k, v, self.scale, self.softmax_offset.astype(q.dtype), prior, cache.window_size
            )
        elif isinstance(cache, MixedQuantizedKVCache) and L == 1:
            out = quantized_attention_with_sinks(
                q,
                k,
                v,
                self.scale,
                mask,
                self.softmax_offset,
                cache.group_size,
                cache.key_bits,
                cache.value_bits,
            )
        elif isinstance(cache, MixedQuantizedKVCache):
            # Prefill: dequantize the cached K/V once (the same values) and use the fused, memory-bounded kernel;
            # the unfused path would materialize [heads, chunk, context] FP32 scores.
            k = mx.dequantize(*k, group_size=cache.group_size, bits=cache.key_bits)
            v = mx.dequantize(*v, group_size=cache.group_size, bits=cache.value_bits)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=self.scale,
                mask=mask,
                sinks=self.softmax_offset.astype(q.dtype),
            )
        else:
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=self.scale,
                mask=mask,
                sinks=self.softmax_offset.astype(q.dtype),
            )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.expert_bias = mx.zeros((args.num_experts,))

    def __call__(self, x):
        return x @ self.weight.T


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.gate = Router(args)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)

    def __call__(self, x):
        scores = mx.sigmoid(self.gate(x).astype(mx.float32))
        choice = scores + self.gate.expert_bias.astype(mx.float32)
        indices = mx.stop_gradient(
            mx.argpartition(-choice, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        )
        weights = mx.take_along_axis(scores, indices, axis=-1)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        routed = (self.switch_mlp(x, indices) * weights[..., None].astype(x.dtype)).sum(axis=-2)
        return routed + self.shared_expert(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer: int):
        super().__init__()
        self.self_attn = Attention(args, layer)
        self.mlp = (
            MLP(args.hidden_size, args.intermediate_size)
            if layer in set(args.mlp_only_layers)
            else SparseMoeBlock(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class FlintTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layer_types = args.layer_types or ["full_attention"] * args.num_hidden_layers
        self.full_index = (
            self.layer_types.index("full_attention")
            if "full_attention" in self.layer_types
            else None
        )
        self.swa_index = (
            self.layer_types.index("sliding_attention")
            if "sliding_attention" in self.layer_types
            else None
        )

    def __call__(
        self, inputs, cache=None, input_embeddings: mx.array | None = None, *, should_stop=None
    ):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        full_mask = (
            create_attention_mask(h, cache[self.full_index])
            if self.full_index is not None
            else None
        )
        swa_mask = (
            create_attention_mask(h, cache[self.swa_index], window_size=self.args.sliding_window)
            if self.swa_index is not None
            else None
        )
        for layer, c, kind in zip(self.layers, cache, self.layer_types, strict=False):
            if should_stop is not None and should_stop():
                raise PrefillCancelled()
            h = layer(h, full_mask if kind == "full_attention" else swa_mask, c)
            if should_stop is not None:
                mx.eval(h, c.state)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FlintTransformer(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        return self.lm_head(self.model(inputs, cache, input_embeddings))

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for key, value in weights.items():
            if key.endswith(".self_attn.softmax_offset"):
                value = value.astype(mx.float32)
            elif key.endswith(".weight") and value.dtype == mx.uint8:
                value = value.view(mx.uint32)
            out[key] = value
        for index, layer in enumerate(self.model.layers):
            name = f"model.layers.{index}.mlp.gate.expert_bias"
            if isinstance(layer.mlp, SparseMoeBlock) and name not in out:
                out[name] = mx.zeros((self.args.num_experts,), dtype=mx.float32)
        return out

    def make_cache(self):
        caches = []
        for kind in self.model.layer_types:
            if kind == "full_attention":
                caches.append(
                    MixedQuantizedKVCache(
                        self.args.kv_group_size,
                        self.args.kv_key_bits,
                        self.args.kv_value_bits,
                    )
                )
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window, keep=0))
        return caches
