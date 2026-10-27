import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort


class PrefillLinear(nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.projection = projection
        self.eval()

    def __call__(self, x):
        if x.size // x.shape[-1] < 1024:
            return self.projection(x)
        projection = self.projection
        weight = mx.dequantize(
            projection.weight,
            projection.scales,
            projection.biases,
            group_size=projection.group_size,
            bits=projection.bits,
            mode=projection.mode,
        )
        return x @ weight.T


class PrefillSwitchGLU(nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.projection = projection
        self.eval()

    def __call__(self, x, indices):
        projection = self.projection
        gate, up = projection.gate_proj, projection.up_proj
        if gate.bits != 4 or indices.size < gate.weight.shape[0] * 4:
            return projection(x, indices)
        from .kernels.m4_max_prefill import gate_up

        expanded = mx.expand_dims(x, (-2, -3))
        sorted_x, sorted_indices, inverse = _gather_sort(expanded, indices)
        gate_values, up_values = gate_up(sorted_x, gate, up, sorted_indices)
        output = projection.down_proj(
            projection.activation(up_values, gate_values), sorted_indices, sorted_indices=True
        )
        return _scatter_unsort(output, inverse, indices.shape).squeeze(-2)


class PrefillAttention(nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.projection = projection
        self.eval()

    def __call__(self, x, mask=None, cache=None):
        batch, count, _ = x.shape
        projection = self.projection
        if count < 1024 or count % 512:
            return projection(x, mask, cache)
        q = projection.q_norm(
            projection.q_proj(x).reshape(batch, count, projection.n_heads, projection.head_dim)
        ).transpose(0, 2, 1, 3)
        k = projection.k_norm(
            projection.k_proj(x).reshape(batch, count, projection.n_kv, projection.head_dim)
        ).transpose(0, 2, 1, 3)
        v = (
            projection.v_proj(x)
            .reshape(batch, count, projection.n_kv, projection.head_dim)
            .transpose(0, 2, 1, 3)
        )
        offset = cache.offset if cache is not None else 0
        q, k = projection.rope(q, offset=offset), projection.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = window_attention(
            q,
            k,
            v,
            scale=projection.scale,
            mask=mask,
            sinks=projection.softmax_offset.astype(q.dtype),
        )
        return projection.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, count, -1))


def window_attention(q, k, v, *, scale, mask, sinks):
    prior = k.shape[-2] - q.shape[-2]
    outputs = []
    for begin in range(0, q.shape[-2], 512):
        end = begin + 512
        # Preserve native key-tile boundaries and accumulation order.
        start = max(0, (prior + begin - 512) // 128 * 128)
        stop = prior + end
        outputs.append(
            mx.fast.scaled_dot_product_attention(
                q[..., begin:end, :],
                k[..., start:stop, :],
                v[..., start:stop, :],
                scale=scale,
                mask=mask[..., begin:end, start:stop],
                sinks=sinks,
            )
        )
    return mx.concatenate(outputs, axis=-2)


def optimize_prefill(model):
    dense_layers = set(model.config["mlp_only_layers"])
    for index, layer in enumerate(model.native.layers):
        for name in ("q_proj", "o_proj"):
            setattr(layer.self_attn, name, PrefillLinear(getattr(layer.self_attn, name)))
        if model.config["layer_types"][index] == "sliding_attention":
            layer.self_attn = PrefillAttention(layer.self_attn)
        dense = index in dense_layers
        mlp = layer.mlp if dense else layer.mlp.shared_expert
        for name in ("gate_proj", "up_proj", "down_proj") if dense else ("down_proj",):
            setattr(mlp, name, PrefillLinear(getattr(mlp, name)))
        if not dense:
            layer.mlp.switch_mlp = PrefillSwitchGLU(layer.mlp.switch_mlp)
