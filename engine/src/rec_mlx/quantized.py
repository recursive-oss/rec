import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from .cache import FixedKVCache, FixedSlidingKVCache
from .model import Model, ModelArgs
from .prefix import PrefillCancelled, PromptCache, select_prefix


def projections(mlp):
    return SimpleNamespace(gate=mlp.gate_proj, up=mlp.up_proj, down=mlp.down_proj)


class QATModel:
    context_limit = 262144

    def __init__(self, directory):
        self.config = json.loads((directory / "config.json").read_text())
        export = self.config["mlx_export"]
        assert export["bits"] == 4 and export["routed_expert_bits"] in (3, 4)
        assert export["embedding_bits"] == 8
        native = Model(ModelArgs.from_dict(self.config))

        def quantization(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            bits = 4
            if path == "model.embed_tokens":
                bits = 8
            elif ".switch_mlp." in path:
                bits = export["routed_expert_bits"]
            return {"group_size": 64, "bits": bits, "mode": "affine"}

        nn.quantize(native, class_predicate=quantization)
        weights = {}
        for path in sorted(directory.glob("model-*.safetensors")):
            shard = mx.load(str(path))
            assert not weights.keys() & shard.keys(), f"Duplicate tensor in {path}"
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
        self.config["quantization"] = dict(self.config["quantization"], mode="affine")
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


class QATSession:
    def __init__(self, engine, max_new_tokens):
        self.engine, self.maximum = engine, max_new_tokens

    def prefill(
        self, tokens, *, prefix=None, should_stop=lambda: False, interrupt_between_layers=False
    ):
        limit = len(tokens) + self.maximum
        assert limit <= self.engine.model.context_limit
        capacity = min(len(tokens), self.engine.prefill_chunk_size)
        prior_storage = None if prefix is None else prefix.storage
        prefix, self.common_prefix_tokens = select_prefix(prefix, tokens)
        self.cached_tokens = 0 if prefix is None else len(prefix.tokens)
        if hasattr(prior_storage, "shrink"):
            prior_storage.shrink(max(capacity, self.cached_tokens))
        if prefix is not None:
            state = self.engine.prefill_state(capacity, limit=limit, storage=prefix.storage)
            cache, logits = prefix.layers, prefix.logits
            for row, entry in zip(state["kv"].rows[:49], cache, strict=True):
                if not row[1]:
                    entry.grow(len(tokens))
                    mx.eval(entry.state)
        else:
            state = self.engine.prefill_state(capacity, limit=limit, storage=prior_storage)
            cache = self.engine.model.make_cache(len(tokens), state["kv"])
        self.prompt_cache = prefix or PromptCache([], cache, None, state["kv"])
        native = self.engine.model.native
        chunk_size = self.engine.prefill_chunk_size
        for begin in range(self.cached_tokens, len(tokens), chunk_size):
            if should_stop():
                raise PrefillCancelled()
            self.prompt_cache.checkpoint()
            end = min(begin + chunk_size, len(tokens))
            state["kv"].grow(end)
            inputs = mx.array([tokens[begin:end]], dtype=mx.int32)
            hidden = (
                native.model(inputs, cache, should_stop=should_stop)
                if interrupt_between_layers
                else native.model(inputs, cache)
            )
            logits = native.lm_head(hidden[:, -1:])
            mx.eval(logits, [entry.state for entry in cache])
            if end > 32768:
                mx.synchronize()
                mx.clear_cache()
            self.prompt_cache.advance(tokens[begin:end], logits)
        if should_stop():
            raise PrefillCancelled()
        # Global attention already uses decode storage. Only the bounded sliding
        # rings need copying; retain their native prefix for subsequent requests.
        for index, entry in enumerate(cache):
            if not state["kv"].rows[index][1]:
                keys, values = entry._prior()
                state["kv"].write(index, 0, keys, values)
        state["capacity"] = state["kv"].capacity
        state["position"] = state["initial"] = len(tokens)
        self._state = state
        return logits

    def forward_token(self, token):
        return self.engine.forward(self._state, token)[0]
