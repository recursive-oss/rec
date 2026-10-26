"""Original Flint affine-INT4 model on one ROCm device, with BF16 KV."""

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import kernels as ops
from .attention import query_tiled_attention


def reference_dequant(w, scales, offsets, bits=4):
    shifts = torch.arange(8 // bits, device=w.device, dtype=torch.int32) * bits
    codes = ((w.to(torch.int32).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).flatten(-2)
    return codes.float() * scales.float().repeat_interleave(
        64, -1
    ) + offsets.float().repeat_interleave(64, -1)


def rope(x, positions, theta=10000.0, angles=None):
    half = x.shape[-1] // 2
    if angles is None:
        freq = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
        angle = positions.float()[:, None] * freq[None]
        c, s = angle.cos()[:, None], angle.sin()[:, None]
    else:
        c, s = angles
    left, right = x[..., :half].float(), x[..., half:].float()
    return torch.cat((left * c - right * s, right * c + left * s), -1).to(x.dtype)


class LayerCache:
    def __init__(self, capacity, kvheads, dim, window=None, device="cuda"):
        self.window = window
        self.capacity = min(capacity, window) if window else capacity
        self.key = torch.zeros((self.capacity, kvheads, dim), device=device, dtype=torch.bfloat16)
        self.value = torch.zeros_like(self.key)
        self.positions = torch.full((self.capacity,), -1, device=device, dtype=torch.int32)

    def reset(self):
        self.positions.fill_(-1)

    def update(self, key, value, positions, native=True):
        if self.window:
            key, value, positions = (
                key[-self.capacity :],
                value[-self.capacity :],
                positions[-self.capacity :],
            )
        if native:
            ops.write_cache(key, value, positions, self)
            return
        if self.window:
            slots = positions.long() % self.capacity
        else:
            slots = positions.long()
        self.key.index_copy_(0, slots, key)
        self.value.index_copy_(0, slots, value)
        self.positions.index_copy_(0, slots, positions)

    def snapshot(self):
        return tuple(t.clone() for t in (self.key, self.value, self.positions))

    def restore(self, state):
        for dest, source in zip((self.key, self.value, self.positions), state, strict=True):
            dest.copy_(source)


class FlintModel:
    def __init__(
        self, directory, capacity=8192, *, device="cuda", reference=False, implementation="blas"
    ):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "config.json").read_text())
        self.device = torch.device(device)
        self.capacity = capacity
        self.reference = reference
        self.implementation = implementation
        self.rdna4_expert = os.environ.get("REC_RDNA4_EXPERT", "off")
        self.rdna4_attention = os.environ.get("REC_RDNA4_ATTENTION", "off")
        self.rdna4_pv = os.environ.get("REC_RDNA4_PV", "off")
        self.rdna4_prefill = os.environ.get("REC_RDNA4_PREFILL", "off")
        self.rdna4_norm = os.environ.get("REC_RDNA4_NORM", "off")
        self.rdna4_rope = os.environ.get("REC_RDNA4_ROPE", "off")
        self.rdna4_graph = os.environ.get("REC_RDNA4_GRAPH", "off")
        self.gfx1151_decode = os.environ.get("REC_GFX1151_DECODE", "off")
        self.gfx1151_attention = os.environ.get("REC_GFX1151_ATTENTION", "off")
        self.gfx1151_prefill = os.environ.get("REC_GFX1151_PREFILL", "off")
        if self.gfx1151_prefill not in ("off", "grouped"):
            raise ValueError("Unknown REC_GFX1151_PREFILL")
        if self.gfx1151_prefill != "off" and self.rdna4_prefill != "off":
            raise ValueError("REC_GFX1151_PREFILL replaces REC_RDNA4_PREFILL")
        if self.gfx1151_attention not in ("off", "split"):
            raise ValueError("Unknown REC_GFX1151_ATTENTION")
        if self.gfx1151_attention != "off" and self.gfx1151_decode != "layer":
            raise ValueError("REC_GFX1151_ATTENTION requires REC_GFX1151_DECODE=layer")
        if self.gfx1151_decode not in ("off", "fused", "layer"):
            raise ValueError("Unknown REC_GFX1151_DECODE")
        if self.gfx1151_decode == "layer" and (
            self.rdna4_norm,
            self.rdna4_rope,
            self.rdna4_expert,
        ) != ("off", "off", "off"):
            raise ValueError(
                "REC_GFX1151_DECODE=layer replaces the RDNA4 norm, RoPE and expert paths"
            )
        if (
            self.gfx1151_decode != "off"
            and not reference
            and torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx1151"
        ):
            raise ValueError("REC_GFX1151_DECODE is qualified only on gfx1151")
        if self.rdna4_expert not in ("off", "simd", "wmma"):
            raise ValueError("Unknown REC_RDNA4_EXPERT")
        if self.rdna4_attention not in ("off", "decode", "all", "mask"):
            raise ValueError("Unknown REC_RDNA4_ATTENTION")
        if self.rdna4_pv not in ("off", "ordered"):
            raise ValueError("Unknown REC_RDNA4_PV")
        if self.rdna4_prefill not in ("off", "compact", "grouped"):
            raise ValueError("Unknown REC_RDNA4_PREFILL")
        if self.rdna4_norm not in ("off", "on", "apply"):
            raise ValueError("Unknown REC_RDNA4_NORM")
        if self.rdna4_rope not in ("off", "on", "fused"):
            raise ValueError("Unknown REC_RDNA4_ROPE")
        if self.rdna4_graph not in ("off", "compact"):
            raise ValueError("Unknown REC_RDNA4_GRAPH")
        # The qualified BLAS profile uses BF16 activations and weights, with
        # FP32 attention. Original INT4 codes and affine metadata are unchanged.
        self.activation_dtype = (
            torch.bfloat16 if implementation in ("blas", "hybrid") else torch.float32
        )
        self.weights = {}
        started = time.perf_counter()
        for shard in sorted(self.directory.glob("model-*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for name in f.keys():
                    if name in self.weights:
                        raise ValueError(f"Duplicate tensor {name}")
                    self.weights[name] = f.get_tensor(name).to(self.device)
            print(f"loaded {shard.name}", flush=True)
        self._check_weights()
        for name, value in list(self.weights.items()):
            if name.endswith(".mlp.gate.weight") and implementation not in ("blas", "hybrid"):
                self.weights[name] = value.float()
        self.dense_cache = {}
        self.dense_cache_limit = int(os.environ.get("REC_DENSE_CACHE_MIB", "1536")) * 1024**2
        if implementation in ("blas", "hybrid"):
            size = max(
                v.numel()
                * 2
                * (
                    max(1, math.ceil(16 * self.config["num_experts_per_tok"] / v.shape[0]))
                    if v.ndim == 3
                    else 1
                )
                for name, v in self.weights.items()
                if v.dtype == torch.uint8 and "embed_tokens" not in name
            )
            self.dequant_workspace = torch.zeros(
                size, device=self.device, dtype=self.activation_dtype
            )
            cached_bytes = 0
            paths = [
                name.removesuffix(".weight")
                for name, v in self.weights.items()
                if v.dtype == torch.uint8
                and name.endswith(".weight")
                and v.ndim == 2
                and "embed_tokens" not in name
            ]
            paths.sort(key=lambda name: (name != "lm_head", ".self_attn." in name))
            if not reference:
                for path in paths:
                    w, s, b = [self.weights[path + "." + n] for n in ("weight", "scales", "biases")]
                    needed = w.numel() * 4
                    if cached_bytes + needed > self.dense_cache_limit:
                        continue
                    self.dense_cache[path] = ops.dequantize(w, s, b, dtype=self.activation_dtype)
                    cached_bytes += needed
        self.caches = [
            LayerCache(
                capacity,
                self.config["num_key_value_heads"],
                self.config["head_dim"],
                None if kind == "full_attention" else self.config["sliding_window"],
                device,
            )
            for kind in self.config["layer_types"]
        ]
        self.position = 0
        self.decode_attention_capacity = capacity
        self.attention_query_rows = 64
        self.attention_tile_threshold = 8192
        self.graph_mode = False
        self.graph_position = torch.zeros((1,), device=self.device, dtype=torch.int32)
        self._qkv = None
        self.load_seconds = time.perf_counter() - started
        self.last_trace = []

    def _check_weights(self):
        c = self.config
        if c["model_type"] != "flint" or c["quantization"] != {"bits": 4, "group_size": 64}:
            raise ValueError("Expected Flint group-64 affine INT4 checkpoint")
        expected = set()

        def norm(path, shape):
            expected.add(path)
            if path not in self.weights or tuple(self.weights[path].shape) != shape:
                raise ValueError(f"Incorrect/missing tensor {path}, expected {shape}")

        def linear(path, n, k, experts=None, bits=4):
            leading = (experts,) if experts else ()
            for suffix, shape in [
                ("weight", (*leading, n, k * bits // 8)),
                ("scales", (*leading, n, k // 64)),
                ("biases", (*leading, n, k // 64)),
            ]:
                norm(path + "." + suffix, shape)
            if self.weights[path + ".weight"].dtype != torch.uint8:
                raise ValueError(f"Expected byte-packed weight {path}")

        h, hd = c["hidden_size"], c["head_dim"]
        linear("model.embed_tokens", c["vocab_size"], h, bits=8)
        linear("lm_head", c["vocab_size"], h)
        norm("model.norm.weight", (h,))
        for i in range(c["num_hidden_layers"]):
            p = f"model.layers.{i}"
            norm(p + ".input_layernorm.weight", (h,))
            norm(p + ".post_attention_layernorm.weight", (h,))
            norm(p + ".self_attn.q_norm.weight", (hd,))
            norm(p + ".self_attn.k_norm.weight", (hd,))
            norm(p + ".self_attn.softmax_offset", (c["num_attention_heads"],))
            for proj, n, k in [
                ("q", c["num_attention_heads"] * hd, h),
                ("k", c["num_key_value_heads"] * hd, h),
                ("v", c["num_key_value_heads"] * hd, h),
                ("o", h, c["num_attention_heads"] * hd),
            ]:
                linear(p + f".self_attn.{proj}_proj", n, k)
            dense = i in c["mlp_only_layers"]
            m = c["intermediate_size"] if dense else c["moe_intermediate_size"]
            base = p + ".mlp" if dense else p + ".mlp.switch_mlp"
            for name, n, k in [("gate", m, h), ("up", m, h), ("down", h, m)]:
                linear(base + f".{name}_proj", n, k, None if dense else c["num_experts"])
            if not dense:
                norm(p + ".mlp.gate.weight", (c["num_experts"], h))
                bias = p + ".mlp.gate.expert_bias"
                if bias not in self.weights:
                    self.weights[bias] = torch.zeros(
                        c["num_experts"], device=self.device, dtype=torch.float32
                    )
                norm(bias, (c["num_experts"],))
                m = c["shared_expert_intermediate_size"]
                for name, n, k in [("gate", m, h), ("up", m, h), ("down", h, m)]:
                    linear(p + f".mlp.shared_expert.{name}_proj", n, k)
        extra = set(self.weights) - expected
        if extra:
            raise ValueError(f"Unexpected checkpoint tensors: {sorted(extra)}")

    def linear(self, x, path, ids=None, layout=None, x_assignments=False):
        w, s, b = [self.weights[path + "." + name] for name in ("weight", "scales", "biases")]
        if (
            self.gfx1151_decode != "off"
            and not self.reference
            and (x.shape[0] == 1 or (x_assignments and ids is not None and ids.shape[0] == 1))
        ):
            from .gfx1151 import gemv_w4

            return gemv_w4(
                x.contiguous(),
                w,
                s,
                b,
                ids=ids.flatten() if ids is not None else None,
                x_expert=x_assignments,
            )
        if not self.reference and ids is not None:
            if ids.shape[0] == 1 and self.rdna4_expert != "off":
                if self.rdna4_expert == "wmma":
                    return ops.decode_dot(x, w, s, b, ids.flatten(), x_assignments)
                return ops.gemv(
                    x, w, s, b, ids=ids.flatten(), x_expert=x_assignments, round_bf16=True
                )
            if ids.shape[0] > 16 and self.rdna4_prefill == "grouped":
                return ops.quantized_mm(x, w, s, b, ids=ids, x_assignments=x_assignments)
        if self.implementation in ("blas", "hybrid"):
            if (
                self.implementation == "hybrid"
                and not self.reference
                and (x.shape[0] == 1 or (x_assignments and ids is not None and ids.shape[0] == 1))
                and path not in self.dense_cache
            ):
                return ops.decode_dot(
                    x,
                    w,
                    s,
                    b,
                    ids=ids.flatten() if ids is not None else None,
                    x_expert=x_assignments,
                )
            return self.blas_linear(x, path, w, s, b, ids, x_assignments, layout)
        if self.reference:
            if ids is None:
                if w.shape[0] > 16384:
                    pieces = []
                    for begin in range(0, w.shape[0], 4096):
                        weight = reference_dequant(
                            w[begin : begin + 4096],
                            s[begin : begin + 4096],
                            b[begin : begin + 4096],
                        )
                        pieces.append(F.linear(x.float(), weight).to(x.dtype))
                    return torch.cat(pieces, -1)
                weight = reference_dequant(w, s, b)
                return F.linear(x.float(), weight).to(x.dtype)
            flat = ids.flatten().long()
            w, s, b = [t.index_select(0, flat) for t in (w, s, b)]
            weight = reference_dequant(w, s, b)
            inputs = x if x_assignments else x.repeat_interleave(ids.shape[-1], 0)
            return torch.bmm(weight, inputs.float().unsqueeze(-1)).squeeze(-1).to(x.dtype)
        if x.shape[0] == 1 or (x_assignments and ids.shape[0] == 1):
            return ops.gemv(
                x, w, s, b, ids=ids.flatten() if ids is not None else None, x_expert=x_assignments
            )
        return ops.quantized_mm(x, w, s, b, ids=ids, layout=layout, x_assignments=x_assignments)

    def norm(self, x, path):
        w = self.weights[path]
        if self.rdna4_norm == "apply" and not self.reference:
            from .rdna4 import norm_apply

            return norm_apply(x, w, self.config["rms_norm_eps"])
        if self.rdna4_norm == "on" and not self.reference:
            return ops.rms_norm(x, w, self.config["rms_norm_eps"])
        if self.reference or self.implementation in ("blas", "hybrid"):
            return (
                x.float()
                * torch.rsqrt(
                    x.float().square().mean(-1, keepdim=True) + self.config["rms_norm_eps"]
                )
                * w.float()
            ).to(x.dtype)
        return ops.rms_norm(x, w, self.config["rms_norm_eps"])

    def blas_layout(self, ids, experts):
        flat = ids.flatten().long()
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=experts)
        offsets = counts.cumsum(0) - counts
        width = int(counts.max().item())
        active = (
            torch.arange(experts, device=ids.device)
            if self.reference
            else torch.nonzero(counts, as_tuple=False).flatten()
        )
        row = torch.arange(width, device=ids.device)[None, :]
        valid = row < counts[:, None]
        assignment = order[(offsets[:, None] + row).clamp_max(flat.numel() - 1)]
        rank = torch.empty_like(order)
        rank.scatter_(
            0, order, torch.arange(flat.numel(), device=ids.device) - offsets[flat[order]]
        )
        return assignment, valid, flat, rank, active

    def blas_linear(self, x, path, w, s, b, ids, x_assignments, layout=None):
        def unpack(ww, ss, bb):
            if self.reference:
                return reference_dequant(ww, ss, bb).to(x.dtype)
            shape = (*ww.shape[:-1], ww.shape[-1] * 2)
            out = self.dequant_workspace[: math.prod(shape)].view(shape)
            return ops.dequantize(ww, ss, bb, dtype=x.dtype, out=out)

        if ids is None:
            if self.reference and w.shape[0] > 16384:
                return torch.cat(
                    [
                        F.linear(x, unpack(w[i : i + 4096], s[i : i + 4096], b[i : i + 4096]))
                        for i in range(0, w.shape[0], 4096)
                    ],
                    -1,
                )
            weight = self.dense_cache.get(path) if not self.reference else None
            if weight is None:
                weight = unpack(w, s, b)
            return F.linear(x, weight)
        if ids.shape[0] <= 16:
            flat = ids.flatten().long()
            if self.reference:
                ww, ss, bb = [v.index_select(0, flat) for v in (w, s, b)]
                weight = unpack(ww, ss, bb)
            else:
                shape = (flat.numel(), w.shape[-2], w.shape[-1] * 2)
                weight = self.dequant_workspace[: math.prod(shape)].view(shape)
                ops.dequantize_experts(w, s, b, flat, weight)
            inputs = x if x_assignments else x.repeat_interleave(ids.shape[-1], 0)
            return torch.bmm(weight, inputs.unsqueeze(-1)).squeeze(-1)
        assignment, valid, flat, rank, active = (
            layout if layout is not None else self.blas_layout(ids, w.shape[0])
        )
        source = assignment if x_assignments else assignment // ids.shape[-1]
        if self.rdna4_prefill == "compact" and not self.reference:
            # Preserve stable assignment order and the BF16 BMM boundary, but
            # never materialize inactive expert inputs or copy unpacked weights.
            inputs = x[source[active]].masked_fill(~valid[active, :, None], 0)
            shape = (active.numel(), w.shape[-2], w.shape[-1] * 2)
            weight = self.dequant_workspace[: math.prod(shape)].view(shape)
            ops.dequantize_experts(w, s, b, active, weight)
            inverse = torch.zeros(w.shape[0], dtype=torch.long, device=x.device)
            inverse[active] = torch.arange(active.numel(), device=x.device)
            values = torch.bmm(inputs, weight.transpose(-1, -2))
            return values[inverse[flat], rank]
        inputs = x[source].masked_fill(~valid[..., None], 0)
        if self.reference:
            shape = (*w.shape[:-1], w.shape[-1] * 2)
            weight = torch.empty(shape, device=x.device, dtype=x.dtype)
            for begin in range(0, w.shape[0], 16):
                block = slice(begin, begin + 16)
                weight[block].copy_(reference_dequant(w[block], s[block], b[block]).to(x.dtype))
        else:
            shape = (*w.shape[:-1], w.shape[-1] * 2)
            weight = self.dequant_workspace[: math.prod(shape)].view(shape)
            ops.dequantize_experts(w, s, b, active, weight, sparse=True)
        values = torch.bmm(inputs, weight.transpose(-1, -2))
        return values[flat, rank]

    def mlp(self, x, path, ids=None, layout=None):
        if (
            ids is not None
            and ids.shape[0] > 16
            and self.gfx1151_prefill == "grouped"
            and not self.reference
        ):
            from .gfx1151 import moe_prefill

            weights = [
                [self.weights[f"{path}.{n}_proj.{part}"] for part in ("weight", "scales", "biases")]
                for n in ("gate", "up", "down")
            ]
            return moe_prefill(x, *weights, ids)
        gate = self.linear(x, path + ".gate_proj", ids, layout)
        up = self.linear(x, path + ".up_proj", ids, layout)
        act = F.silu(gate) * up
        return self.linear(act, path + ".down_proj", ids, layout, x_assignments=ids is not None)

    @torch.inference_mode()
    def forward(self, tokens, *, trace=False):
        count = tokens.numel()
        if self.position + count > self.capacity:
            raise ValueError("Context capacity exceeded")
        positions = (
            self.graph_position
            if self.graph_mode
            else torch.arange(
                self.position, self.position + count, device=self.device, dtype=torch.int32
            )
        )
        if count == 1 and self.gfx1151_decode == "layer" and not self.reference:
            return self._forward_decode_layer(tokens, positions, trace)
        emb = [self.weights["model.embed_tokens." + n] for n in ("weight", "scales", "biases")]
        if self.reference:
            selected = [t.index_select(0, tokens.long()) for t in emb]
            x = reference_dequant(*selected, bits=8).to(self.activation_dtype)
        else:
            x = ops.embedding(tokens, *emb, dtype=self.activation_dtype)
        self.last_trace = []
        c = self.config
        angles = None
        if self.rdna4_rope != "off" and not self.reference:
            half = c["head_dim"] // 2
            freq = c["rope_theta"] ** (
                -torch.arange(half, device=self.device, dtype=torch.float32) / half
            )
            angle = positions.float()[:, None] * freq[None]
            angles = angle.cos()[:, None], angle.sin()[:, None]
        for i, cache in enumerate(self.caches):
            p = f"model.layers.{i}"
            h = self.norm(x, p + ".input_layernorm.weight")
            q = self.linear(h, p + ".self_attn.q_proj").reshape(
                count, c["num_attention_heads"], c["head_dim"]
            )
            key = self.linear(h, p + ".self_attn.k_proj").reshape(
                count, c["num_key_value_heads"], c["head_dim"]
            )
            value = self.linear(h, p + ".self_attn.v_proj").reshape_as(key)
            q = self.norm(q.contiguous(), p + ".self_attn.q_norm.weight")
            key = self.norm(key.contiguous(), p + ".self_attn.k_norm.weight")
            if i not in c["no_rope_layer_indices"]:
                if self.rdna4_rope == "fused" and not self.reference:
                    from .rdna4 import rope_apply

                    q, key = rope_apply(q, angles), rope_apply(key, angles)
                else:
                    q, key = (
                        rope(q, positions, c["rope_theta"], angles),
                        rope(key, positions, c["rope_theta"], angles),
                    )
            key, value = key.bfloat16(), value.bfloat16()
            if cache.window and count > 1:
                ak = torch.cat((cache.key, key))
                av = torch.cat((cache.value, value))
                ap = torch.cat((cache.positions, positions))
            else:
                cache.update(key, value, positions, native=not self.reference)
                ak, av, ap = cache.key, cache.value, cache.positions
                if not cache.window and count == 1:
                    extent = self.decode_attention_capacity
                    ak, av, ap = ak[:extent], av[:extent], ap[:extent]
                if not cache.window and count > 1:
                    end = self.position + count
                    ak, av, ap = ak[:end], av[:end], ap[:end]
            attention = self._attend(q, ak, av, ap, positions, cache, p, count, x.dtype)
            if cache.window and count > 1:
                cache.update(key, value, positions, native=not self.reference)
            x = x + self.linear(attention.reshape(count, -1), p + ".self_attn.o_proj")
            h = self.norm(x, p + ".post_attention_layernorm.weight")
            if i in c["mlp_only_layers"]:
                y = self.mlp(h, p + ".mlp")
            else:
                route = (
                    F.linear(h, self.weights[p + ".mlp.gate.weight"].to(h.dtype)).float().sigmoid()
                )
                # stable sort makes the checkpoint's tie convention deterministic.
                choice = route + self.weights[p + ".mlp.gate.expert_bias"].float()
                ids = torch.argsort(choice, dim=-1, descending=True, stable=True)[
                    ..., : c["num_experts_per_tok"]
                ].contiguous()
                scores = route.gather(-1, ids)
                scores = (scores / (scores.sum(-1, keepdim=True) + 1e-20)).to(h.dtype)
                grouped = count > 16 and self.gfx1151_prefill == "grouped" and not self.reference
                layout = (
                    self.blas_layout(ids, c["num_experts"])
                    if count > 16 and self.implementation in ("blas", "hybrid") and not grouped
                    else (
                        None
                        if count == 1 or self.reference or self.implementation in ("blas", "hybrid")
                        else ops.routing_layout(ids, c["num_experts"])
                    )
                )
                routed = self.mlp(h, p + ".mlp.switch_mlp", ids, layout).reshape(
                    count, c["num_experts_per_tok"], -1
                )
                y = (routed * scores[..., None]).sum(1) + self.mlp(h, p + ".mlp.shared_expert")
            x = x + y
            if trace:
                self.last_trace.append(x.clone())
        self.position += count
        h = self.norm(x[-1:], "model.norm.weight")
        logits = self.linear(h, "lm_head").float()
        if self.graph_mode:
            self.graph_position.add_(1)
        return logits

    def _forward_decode_layer(self, tokens, positions, trace):
        """Single-token forward with fused gfx1151 kernels; attention scores and PV use _attend."""
        from . import gfx1151 as g

        c = self.config
        eps = c["rms_norm_eps"]
        heads, kv_heads, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        if self._qkv is None:
            self._qkv = [
                [
                    torch.cat(
                        [self.weights[f"model.layers.{i}.self_attn.{n}_proj.{part}"] for n in "qkv"]
                    )
                    for part in ("weight", "scales", "biases")
                ]
                for i in range(c["num_hidden_layers"])
            ]
            half = dim // 2
            self._rope_freq = c["rope_theta"] ** (
                -torch.arange(half, device=self.device, dtype=torch.float32) / half
            )
        angle = positions.float()[:, None] * self._rope_freq[None]
        rotation = angle.cos().flatten(), angle.sin().flatten()
        emb = [self.weights["model.embed_tokens." + n] for n in ("weight", "scales", "biases")]
        x = ops.embedding(tokens, *emb, dtype=self.activation_dtype)
        self.last_trace = []
        for i, cache in enumerate(self.caches):
            p = f"model.layers.{i}"
            w = self.weights
            h = g.rms_norm(x, w[p + ".input_layernorm.weight"], eps)
            qkv = g.gemv_w4(h, *self._qkv[i])
            q = g.qk_post(
                qkv,
                w[p + ".self_attn.q_norm.weight"],
                w[p + ".self_attn.k_norm.weight"],
                None if i in c["no_rope_layer_indices"] else rotation,
                cache,
                positions,
                eps,
                heads,
                kv_heads,
                dim,
            )
            ak, av, ap = cache.key, cache.value, cache.positions
            if not cache.window:
                extent = self.decode_attention_capacity
                ak, av, ap = ak[:extent], av[:extent], ap[:extent]
            if self.gfx1151_attention == "split":
                sinks = self.weights[p + ".self_attn.softmax_offset"]
                attention = g.sink_attention(q, ak, av, positions, ap, sinks, cache.window)
            else:
                attention = self._attend(q, ak, av, ap, positions, cache, p, 1, x.dtype)
            o = [w[p + ".self_attn.o_proj." + n] for n in ("weight", "scales", "biases")]
            x = g.gemv_w4(attention.reshape(1, -1), *o, residual=x)
            h = g.rms_norm(x, w[p + ".post_attention_layernorm.weight"], eps)
            if i in c["mlp_only_layers"]:
                mlp = p + ".mlp"
                gate, up, down = [
                    [w[f"{mlp}.{n}_proj.{part}"] for part in ("weight", "scales", "biases")]
                    for n in ("gate", "up", "down")
                ]
                x = g.gemv_w4(g.gate_up_w4(h, *gate, *up), *down, residual=x)
            else:
                ids, scores = g.route(
                    h,
                    w[p + ".mlp.gate.weight"],
                    w[p + ".mlp.gate.expert_bias"],
                    c["num_experts_per_tok"],
                )
                routed, shared = p + ".mlp.switch_mlp", p + ".mlp.shared_expert"
                rg, ru, rd, sg, su, sd = [
                    [w[f"{base}.{n}_proj.{part}"] for part in ("weight", "scales", "biases")]
                    for base in (routed, shared)
                    for n in ("gate", "up", "down")
                ]
                act = g.gate_up_w4(h, *rg, *ru, ids=ids)
                shared_act = g.gate_up_w4(h, *sg, *su)
                x = g.down_combine_w4(act, *rd, ids, scores, shared_act, *sd, x)
            if trace:
                self.last_trace.append(x.clone())
        self.position += 1
        h = g.rms_norm(x, self.weights["model.norm.weight"], eps)
        logits = g.gemv_w4(
            h, *[self.weights["lm_head." + n] for n in ("weight", "scales", "biases")]
        ).float()
        if self.graph_mode:
            self.graph_position.add_(1)
        return logits

    def _attend(self, q, ak, av, ap, positions, cache, p, count, dtype):
        c = self.config
        sinks = self.weights[p + ".self_attn.softmax_offset"].float()
        if not self.reference and (
            self.rdna4_attention == "all" or (self.rdna4_attention == "decode" and count == 1)
        ):
            from .rdna4 import sink_attention

            attention = sink_attention(q, ak, av, sinks, positions, ap, cache.window)
        elif (
            not self.reference
            and self.implementation in ("blas", "hybrid")
            and not cache.window
            and count > self.attention_query_rows
            and ak.shape[0] > self.attention_tile_threshold
        ):
            attention = query_tiled_attention(
                q, ak, av, sinks, positions, ap, self.attention_query_rows
            )
        elif self.reference or self.implementation in ("blas", "hybrid"):
            repeats = c["num_attention_heads"] // c["num_key_value_heads"]
            score = torch.einsum(
                "mkhd,nkd->khmn",
                q.float().view(count, c["num_key_value_heads"], repeats, c["head_dim"]),
                ak.float(),
            ).reshape(c["num_attention_heads"], count, ak.shape[0]) / math.sqrt(c["head_dim"])
            if self.rdna4_attention == "mask" and not self.reference:
                from .rdna4 import mask_sink

                score = mask_sink(score, sinks, positions, ap, cache.window)
            else:
                allowed = (ap[None] <= positions[:, None]) & (ap[None] >= 0)
                if cache.window:
                    allowed &= ap[None] > positions[:, None] - cache.window
                score = score.masked_fill(~allowed[None], -float("inf"))
                score = torch.cat((score, sinks[:, None, None].expand(-1, count, 1)), -1)
            probs = torch.softmax(score, -1)[..., : ak.shape[0]]
            del score
            if (
                self.rdna4_pv == "ordered"
                and count == 1
                and c["head_dim"] == 128
                and not self.reference
            ):
                from .ordered_attention import decode_pv

                attention = decode_pv(probs, av, positions, window=cache.window)
            else:
                attention = (
                    torch.einsum(
                        "khmn,nkd->mkhd",
                        probs.view(c["num_key_value_heads"], repeats, count, ak.shape[0]),
                        av.float(),
                    )
                    .reshape(count, c["num_attention_heads"], c["head_dim"])
                    .to(dtype)
                )
            del probs
        else:
            attention = ops.attention(
                q.contiguous(),
                ak.contiguous(),
                av.contiguous(),
                sinks,
                positions,
                ap,
                cache.window,
                dtype=dtype,
            )
        return attention

    def reset(self):
        self.position = 0
        self.graph_position.zero_()
        for cache in self.caches:
            cache.reset()

    def snapshot(self):
        return self.position, [c.snapshot() for c in self.caches]

    def restore(self, state):
        self.position, caches = state
        self.graph_position.fill_(self.position)
        for cache, saved in zip(self.caches, caches, strict=True):
            cache.restore(saved)


class GraphDecoder:
    """A single-token HIP Graph with device-side position and stable KV addresses."""

    def __init__(self, model):
        self.model = model
        self.input = torch.zeros((1,), device=model.device, dtype=torch.long)
        initial = model.position
        if initial >= model.capacity:
            raise ValueError("Context capacity exceeded")
        compact = getattr(model, "rdna4_graph", "off") == "compact"
        if compact:
            # Warmup and capture each write exactly this one position per layer.
            # Preserve only those slots; copying the entire 33K KV costs ~1 GiB.
            state = []
            for cache in model.caches:
                slot = initial % cache.capacity if cache.window else initial
                state.append(
                    (
                        slot,
                        cache.key[slot].clone(),
                        cache.value[slot].clone(),
                        cache.positions[slot].clone(),
                    )
                )
        else:
            state = model.snapshot()

        def restore(saved=state):
            if compact:
                model.position = initial
                model.graph_position.fill_(initial)
                for cache, (slot, key, value, position) in zip(model.caches, saved, strict=True):
                    cache.key[slot].copy_(key)
                    cache.value[slot].copy_(value)
                    cache.positions[slot].copy_(position)
            else:
                model.restore(saved)

        model.graph_position.fill_(initial)
        model.graph_mode = True
        try:
            # Torch retains BLAS workspaces per handle/stream. Repeated Graph
            # recapture on fresh streams accumulates live workspace allocations,
            # even after the old Graph pool is released. Serial model requests
            # can share one warmup stream for the model's lifetime.
            warmup = getattr(model, "_graph_warmup_stream", None) if compact else None
            if warmup is None:
                warmup = torch.cuda.Stream(device=model.device)
                if compact:
                    model._graph_warmup_stream = warmup
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                for _ in range(2):
                    model.position = initial
                    model.graph_position.fill_(initial)
                    model.forward(self.input)
            torch.cuda.current_stream().wait_stream(warmup)
            restore()
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = model.forward(self.input)
            restore()
        finally:
            model.graph_mode = False
            del state

    def __call__(self, token):
        if self.model.position >= self.model.capacity:
            raise ValueError("Context capacity exceeded")
        self.input.copy_(token.reshape(1))
        self.graph.replay()
        self.model.position += 1
        return self.output
