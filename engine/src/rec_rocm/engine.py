"""Serial single-session engine; public token events follow rec's engine contract."""

import gc
import time
from pathlib import Path

import torch

from rec_mlx.chat import load_tokenizer, normalize_messages
from rec_mlx.checkpoint import validate
from rec_mlx.detokenize import StreamingText
from rec_mlx.engine import ContextLimitError

from .grammar import ConstraintCompiler
from .model import FlintModel, GraphDecoder
from .platform import validate_platform


class Engine:
    def __init__(
        self,
        directory,
        manifest_path,
        *,
        context_limit=8192,
        enforce_tool_schemas=False,
        graph=True,
        prefill_chunk=256,
        implementation="blas",
    ):
        directory, manifest_path = Path(directory), Path(manifest_path)
        device, platform = validate_platform()
        self.identity = validate(directory, manifest_path)
        limit = min(platform.max_context, self.identity["max_context"])
        if not 1 <= context_limit <= limit:
            raise ValueError(f"ROCm context limit must be between 1 and {limit}")
        self.context_limit = context_limit
        self.prefill_chunk = prefill_chunk
        self.use_graph = graph
        self.tokenizer = load_tokenizer(directory)
        self.model = FlintModel(directory, context_limit, implementation=implementation)
        self.eos = set(self.model.config["eos_token_id"])
        self.enforce_tool_schemas = enforce_tool_schemas
        self._constraints = None
        self._tokens = []
        self._logits = None
        self._graph = None
        self._graph_captures = 0
        self.startup = {
            "device": device.name,
            "architecture": device.gcnArchName.split(":")[0],
            "logical_device": 0,
            "kv_format": "bf16",
            "weight_format": self.identity["weight_format"],
            "load_seconds": self.model.load_seconds,
            "engine": "rocm",
            "chat_template": "flint",
            "implementation": implementation,
            "rdna4_options": {
                name: getattr(self.model, "rdna4_" + name)
                for name in ("expert", "attention", "prefill", "norm", "rope", "graph", "pv")
            },
            "gfx1151_options": {
                "decode": self.model.gfx1151_decode,
                "attention": self.model.gfx1151_attention,
                "prefill": self.model.gfx1151_prefill,
            },
            "dense_cache_bytes": sum(
                t.numel() * t.element_size() for t in self.model.dense_cache.values()
            ),
            "prefill_attention_query_rows": self.model.attention_query_rows,
            "prefill_attention_tile_threshold": self.model.attention_tile_threshold,
            "activation_profile": "bf16-blas-projections_fp32-attention_bf16-kv"
            if implementation == "blas"
            else "experimental-fp32-triton_bf16-kv",
        }

    @property
    def constraints(self):
        if self._constraints is None:
            self._constraints = ConstraintCompiler(
                self.tokenizer, enforce_tool_schemas=self.enforce_tool_schemas
            )
        return self._constraints

    def prompt(self, messages, *, tools=None):
        return self.tokenizer.apply_chat_template(
            normalize_messages(messages),
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )

    def clear_prefix(self):
        self.model.reset()
        self._tokens = []
        self._logits = None

    def prepare_prefix(self, **kwargs):
        return None

    @torch.inference_mode()
    def generate_tokens(
        self, ids, sampling, *, should_stop=lambda: False, grammar=None, return_logprobs=False
    ):
        ids = list(ids)
        if not ids or len(ids) + sampling.max_tokens > self.context_limit:
            raise ContextLimitError(
                "Prompt plus output allowance exceeds the configured context limit"
            )
        start = time.perf_counter()
        extent = min(
            self.context_limit, max(512, 1 << (len(ids) + sampling.max_tokens - 1).bit_length())
        )
        self.model.decode_attention_capacity = extent
        # Release a superseded graph pool before allocating prefill temporaries.
        if (
            self.use_graph
            and self._graph is not None
            and getattr(self._graph, "attention_capacity", None) != extent
        ):
            self._logits = self._logits.clone()
            self._graph = None
            gc.collect()
            torch.cuda.empty_cache()
        # Extending a prefix shares all KV. A shortened/edited request safely replays.
        cached = len(self._tokens) if ids[: len(self._tokens)] == self._tokens else 0
        if not cached:
            self.clear_prefix()
        for begin in range(cached, len(ids), self.prefill_chunk):
            if should_stop():
                return
            chunk = ids[begin : begin + self.prefill_chunk]
            self._logits = self.model.forward(torch.tensor(chunk, device="cuda"))
            self._tokens.extend(chunk)
        torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        if self.use_graph and self._graph is None:
            self._graph = GraphDecoder(self.model)
            self._graph.attention_capacity = extent
            self._graph_captures += 1
        self.model.graph_position.fill_(self.model.position)
        constraint = grammar.session() if grammar is not None else None
        stops = self.eos if constraint is None else set(constraint.stops)
        text = StreamingText(self.tokenizer)
        generated = []
        stamps = []
        generator = torch.Generator(device="cuda").manual_seed(sampling.seed)
        presence = (
            torch.zeros(self.model.config["vocab_size"], device="cuda")
            if sampling.presence_penalty
            else None
        )
        reason = "length"
        for index in range(sampling.max_tokens):
            if should_stop():
                reason = "stop"
                break
            if index:
                token_tensor = torch.tensor([generated[-1]], device="cuda")
                self._logits = (
                    self._graph(token_tensor)
                    if self._graph is not None
                    else self.model.forward(token_tensor)
                )
                self._tokens.append(generated[-1])
            flat = self._logits.flatten()
            if index % 64 == 0 and not bool(torch.isfinite(flat).all().item()):
                raise FloatingPointError("Inference produced nonfinite logits")
            if constraint is not None:
                flat = constraint.mask(flat)
            if presence is not None:
                flat = flat - presence * sampling.presence_penalty
            if sampling.temperature == 0:
                token = flat.argmax().item()
            else:
                scaled = flat / sampling.temperature
                if sampling.top_k:
                    values, selected = scaled.topk(min(sampling.top_k, scaled.numel()))
                else:
                    values, selected = scaled.sort(descending=True)
                probs = values.softmax(-1)
                if sampling.top_p < 1:
                    probs = probs.masked_fill(probs.cumsum(-1) - probs > sampling.top_p, 0)
                    probs = probs / probs.sum()
                token = selected[torch.multinomial(probs, 1, generator=generator)].item()
            if constraint is not None:
                constraint.accept(token)
            if presence is not None:
                presence[token] = 1
            generated.append(token)
            stamps.append(time.perf_counter())
            details = (
                {
                    "logprob": float(
                        torch.log_softmax(flat / (sampling.temperature or 1), -1)[token].item()
                    )
                }
                if return_logprobs
                else {}
            )
            if token in stops:
                reason = "stop"
                yield {
                    "token": token,
                    "text": text.finish(generated[:-1]),
                    "finish_reason": "stop",
                    **details,
                }
                break
            delta = text.add_token(token)
            if index + 1 == sampling.max_tokens:
                delta += text.finish(generated)
            yield {
                "token": token,
                "text": delta,
                "finish_reason": "length" if index + 1 == sampling.max_tokens else None,
                **details,
            }
        end = time.perf_counter()
        decode = (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        metrics = {
            "device_used_bytes_at_request_end": total_bytes - free_bytes,
            "prompt_tokens": len(ids),
            "generated_tokens": len(generated),
            "token_ids": generated,
            "cached_prompt_tokens": cached,
            "prefilled_tokens": len(ids) - cached,
            "prefill_seconds": prefill_end - start,
            "ttft_seconds": stamps[0] - start if stamps else None,
            "decode_seconds": decode,
            "decode_tokens_per_second": (len(stamps) - 1) / decode if decode else 0,
            "request_seconds": end - start,
            "decode_strategy": "hip_graph" if self._graph is not None else "eager",
            "graph_capture_count": self._graph_captures,
            "logical_device": 0,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "token_times": stamps,
        }
        yield {"metrics": metrics, "finish_reason": reason}

    def generate(self, messages, sampling, *, tools=None, grammar=None):
        yield from self.generate_tokens(
            self.prompt(messages, tools=tools), sampling, grammar=grammar
        )
