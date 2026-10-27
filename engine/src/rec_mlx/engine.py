import hashlib
import json
import platform
import resource
import subprocess
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from .chat import load_tokenizer, normalize_messages
from .checkpoint import validate
from .detokenize import StreamingText
from .kernels.device import profile_for_device


class ContextLimitError(ValueError):
    pass


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 1024
    seed: int = 0
    presence_penalty: float = 0.0

    def __post_init__(self):
        for name in ("temperature", "top_p", "presence_penalty"):
            if type(getattr(self, name)) not in (int, float):
                raise ValueError(f"{name} must be a number")
        for name in ("top_k", "max_tokens", "seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("temperature must be in [0, 2] and top_p in (0, 1]")
        if not -2 <= self.presence_penalty <= 2:
            raise ValueError("presence_penalty must be in [-2, 2]")
        if self.top_k < 0 or self.max_tokens < 1:
            raise ValueError("top_k must be nonnegative and max_tokens positive")
        if not 0 <= self.seed <= 2**64 - self.max_tokens:
            raise ValueError("seed plus max_tokens must fit in unsigned 64-bit random keys")


class Engine:
    def __init__(
        self,
        directory: Path,
        manifest_path: Path,
        *,
        context_limit: int,
        enforce_tool_schemas: bool = False,
    ):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Flint requires an Apple Silicon Mac")
        memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        if memory < 48 * 1024**3:
            raise RuntimeError("This build targets Macs with at least 48 GiB of memory")
        import mlx.core as mx

        device = mx.device_info()
        profile = profile_for_device(device)
        self.identity = validate(directory, manifest_path)
        config = json.loads((directory / "config.json").read_text())
        from .kernels import build_decoder
        from .quantized import QATModel

        if self.identity["max_context"] != QATModel.context_limit:
            raise ValueError("Manifest context limit does not match the engine capacity")
        if not 1 <= context_limit <= QATModel.context_limit:
            raise ValueError(f"Context limit must be between 1 and {QATModel.context_limit} tokens")
        self.context_limit = context_limit
        self._prompt_cache = None
        self._prefix_preparation = None
        self.mx = mx
        self.tokenizer = load_tokenizer(directory)
        self._constraints = None
        self.enforce_tool_schemas = enforce_tool_schemas
        self._constraints_lock = threading.Lock()
        self.eos = set(config["eos_token_id"])
        if not self.eos:
            raise ValueError("Checkpoint has no EOS tokens")
        start = time.perf_counter()
        mx.set_cache_limit(0)
        mx.clear_cache()
        mx.reset_peak_memory()
        self.runtime, self.resident_sampler = build_decoder(QATModel(directory), optimized=True)
        # Packing replaces native weights with lazy views. Materialize them on their creation
        # stream before HTTP threads use the native prefill model.
        mx.eval(self.runtime.model.native.parameters())
        # Retain enough allocator cache to reuse the first 1.78-GiB KV block.
        cache_limit = 2 * 1024**3
        mx.set_cache_limit(cache_limit)
        self.startup = {
            "device": device,
            "chat_template": "flint",
            "chat_template_sha256": hashlib.sha256(
                self.tokenizer.chat_template.encode()
            ).hexdigest(),
            "allocation_cache_limit_bytes": cache_limit,
            "load_seconds": time.perf_counter() - start,
            "load_peak_mlx_bytes": mx.get_peak_memory(),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "physical_memory_bytes": memory,
            "device_profile": profile.chip,
            "weight_format": self.identity["weight_format"],
            "kv_format": self.identity["kv_format"],
        }

    @property
    def constraints(self):
        with self._constraints_lock:
            if self._constraints is None:
                from .grammar import ConstraintCompiler

                self._constraints = ConstraintCompiler(
                    self.tokenizer, enforce_tool_schemas=self.enforce_tool_schemas
                )
            return self._constraints

    def prompt(self, messages: list[dict], *, tools=None) -> list[int]:
        return self.tokenizer.apply_chat_template(
            normalize_messages(messages),
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )

    def generate(self, messages: list[dict], sampling: Sampling, *, grammar=None):
        events = self.generate_tokens(self.prompt(messages), sampling, grammar=grammar)
        try:
            for event in events:
                if "metrics" in event:
                    yield event
                elif event["text"]:
                    yield {"text": event["text"]}
        finally:
            events.close()

    def generate_tokens(
        self,
        ids,
        sampling,
        *,
        should_stop=lambda: False,
        grammar=None,
        return_logprobs=False,
    ):
        # Metal streams are thread-local; HTTP handlers run outside the model-loading thread.
        with self.mx.stream(self.mx.new_stream(self.mx.gpu)):
            from .prefix import PrefillCancelled

            try:
                with grammar.session() if grammar is not None else nullcontext(None) as constraint:
                    yield from self._generate_tokens(
                        ids,
                        sampling,
                        should_stop=should_stop,
                        grammar=constraint,
                        return_logprobs=return_logprobs,
                    )
            except PrefillCancelled:
                self.mx.synchronize()
                self.mx.clear_cache()

    def prepare_prefix(self, *, should_stop=lambda: False):
        from .prefix import PrefillCancelled

        pending = self._prefix_preparation
        if pending is None:
            return None
        mx = self.mx
        chunk = self.runtime.prefill_chunk_size
        origin = len(pending.origin.tokens)
        end = origin + (len(pending.tokens) - origin) // chunk * chunk
        start = time.perf_counter()
        prepared = 0
        with mx.stream(mx.new_stream(mx.gpu)):
            while (origin if pending.cache is None else len(pending.cache.tokens)) < end:
                if should_stop():
                    break
                prefix = pending.origin if pending.cache is None else pending.cache
                target = len(prefix.tokens) + chunk
                session = self.runtime.make_session(max_new_tokens=0)
                try:
                    session.prefill(
                        pending.tokens[:target],
                        prefix=prefix.fork(),
                        should_stop=should_stop,
                        interrupt_between_layers=True,
                    )
                except PrefillCancelled:
                    # A partial chunk wrote past the committed prefix in shared global KV; never reuse it.
                    if len(session.prompt_cache.tokens) != target:
                        break
                pending.cache = session.prompt_cache
                prepared += chunk
            mx.synchronize()
        return {"prepared_tokens": prepared, "preparation_seconds": time.perf_counter() - start}

    def _schedule_prefix(self, ids, generated):
        from .prefix import PrefixPreparation

        if len(generated) >= self.runtime.prefill_chunk_size:
            self._prefix_preparation = PrefixPreparation(self._prompt_cache, ids + generated)

    def _take_prefix(self, ids):
        prefix, self._prompt_cache = self._prompt_cache, None
        pending, self._prefix_preparation = self._prefix_preparation, None
        if pending is not None and pending.cache is not None:
            prepared = pending.cache
            if ids[: len(prepared.tokens)] != prepared.tokens:
                from .prefix import select_prefix

                prepared, _ = select_prefix(prepared, ids)
            if prepared is not None and len(prepared.tokens) > len(pending.origin.tokens):
                prefix = prepared
        return prefix

    def _generate_tokens(self, ids, sampling, *, should_stop, grammar=None, return_logprobs=False):
        if len(ids) + sampling.max_tokens > self.context_limit:
            raise ContextLimitError(
                "Prompt plus output allowance exceeds the configured context limit"
            )
        mx = self.mx
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(
            temp=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
        )
        mx.reset_peak_memory()
        session = self.runtime.make_session(max_new_tokens=sampling.max_tokens)
        unfiltered_t1 = (sampling.temperature, sampling.top_p, sampling.top_k) == (1.0, 1.0, 0)
        mx.random.seed(sampling.seed)
        stops = self.eos if grammar is None else set(grammar.stops)
        generated = []
        token_logprobs = []
        start = time.perf_counter()
        reason = "length"
        try:
            preparation = self._prefix_preparation
            prefix = self._take_prefix(ids)
            from .prefix import PrefillCancelled

            try:
                logits = session.prefill(ids, prefix=prefix, should_stop=should_stop)
            except PrefillCancelled:
                if preparation is not None and prefix is preparation.cache:
                    # Keep the original input anchor until a new request completes prefill.
                    self._prompt_cache = preparation.origin
                    preparation.cache = session.prompt_cache
                    self._prefix_preparation = preparation
                else:
                    self._prompt_cache = session.prompt_cache
                raise
            del prefix, preparation
            self._prompt_cache = session.prompt_cache
            text_stream = StreamingText(self.tokenizer)
            if sampling.presence_penalty:
                session._state["presence_mask"] = mx.zeros(logits.size, mx.float32)
            mx.eval(logits)
            if self.runtime.device_profile.chip == "M4 Max":
                from .kernels.filtered_sampling import make_sampler

                sampler = make_sampler(
                    temp=sampling.temperature,
                    top_p=sampling.top_p,
                    top_k=sampling.top_k,
                )
            prefill_end = time.perf_counter()
            prefill_peak = mx.get_peak_memory()
            first = None
            packet = []
            computed_tokens = 0
            for index in range(sampling.max_tokens):
                if should_stop():
                    reason = "stop"
                    break
                if index:
                    if not packet:
                        count = min(8, sampling.max_tokens - index)
                        options = {"logprobs": token_logprobs} if return_logprobs else {}
                        packet = self.resident_sampler.block(
                            session._state,
                            generated[-1],
                            count,
                            sampling.seed + index,
                            temperature=sampling.temperature,
                            top_p=sampling.top_p,
                            top_k=sampling.top_k,
                            presence_penalty=sampling.presence_penalty,
                            grammar=grammar,
                            **options,
                        )
                        computed_tokens += len(packet)
                    token = packet.pop(0)
                else:
                    if not bool(mx.all(mx.isfinite(logits)).item()):
                        raise FloatingPointError("The inference engine produced nonfinite logits")
                    flat = logits.reshape(-1).astype(mx.float32)
                    if grammar is not None:
                        flat = grammar.first_logits(flat)
                    if sampling.temperature == 0:
                        token = int(mx.argmax(flat).item())
                    elif unfiltered_t1:
                        # Match the upstream first-token random key exactly.
                        token = int(
                            mx.random.categorical(flat, key=mx.random.key(sampling.seed)).item()
                        )
                    else:
                        token = int(sampler(flat - mx.logsumexp(flat)).item())
                    if grammar is not None:
                        grammar.accept_first(token, session._state)
                    if return_logprobs:
                        report = flat / sampling.temperature if sampling.temperature else flat
                        token_logprobs.append(float((report[token] - mx.logsumexp(report)).item()))
                    computed_tokens += 1
                    if sampling.presence_penalty:
                        session._state["presence_mask"][token] = 1.0
                if first is None:
                    first = time.perf_counter()
                generated.append(token)
                details = {"logprob": token_logprobs[index]} if return_logprobs else {}
                if token in stops:
                    reason = "stop"
                    delta = text_stream.finish(generated[:-1])
                    yield {"token": token, "text": delta, "finish_reason": reason, **details}
                    break
                delta = text_stream.add_token(token)
                if index + 1 == sampling.max_tokens:
                    delta += text_stream.finish(generated)
                yield {
                    "token": token,
                    "text": delta,
                    "finish_reason": "length" if index + 1 == sampling.max_tokens else None,
                    **details,
                }
            mx.synchronize()
            end = time.perf_counter()
            decode_seconds = end - first if first is not None else 0.0
            sliding_bytes = sum(
                cache.nbytes
                for row, cache in zip(
                    self.runtime.metadata[:49], self._prompt_cache.layers, strict=True
                )
                if not row[1]
            )
            yield {
                "finish_reason": reason,
                "metrics": {
                    "prompt_tokens": len(ids),
                    "cached_prompt_tokens": session.cached_tokens,
                    "common_prefix_tokens": session.common_prefix_tokens,
                    "prefix_checkpoint_bytes": self._prompt_cache.checkpoint_bytes,
                    "prefilled_tokens": len(ids) - session.cached_tokens,
                    "prompt_cache_bytes": session._state["kv"].nbytes
                    + sliding_bytes
                    + self._prompt_cache.checkpoint_bytes,
                    "native_sliding_cache_bytes": sliding_bytes,
                    "shared_kv_bytes": session._state["kv"].nbytes,
                    "decode_kv_capacity_tokens": session._state["capacity"],
                    "decode_kv_growth_count": session._state["growth_count"],
                    "decode_kv_growth_seconds": session._state["growth_seconds"],
                    "generated_tokens": len(generated),
                    "token_ids": generated,
                    **(
                        {"token_logprobs": token_logprobs[: len(generated)]}
                        if return_logprobs
                        else {}
                    ),
                    "ttft_seconds": first - start if first is not None else None,
                    "prefill_seconds": prefill_end - start,
                    "decode_seconds": decode_seconds,
                    "request_seconds": end - start,
                    "decode_tokens_per_second": max(0, len(generated) - 1) / decode_seconds
                    if decode_seconds
                    else 0.0,
                    "prefill_peak_mlx_bytes": prefill_peak,
                    "peak_mlx_bytes": mx.get_peak_memory(),
                    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "computed_tokens": computed_tokens,
                    "decode_strategy": "resident_sampling",
                    **(grammar.metrics() if grammar is not None else {}),
                },
            }
        finally:
            if generated:
                self._schedule_prefix(ids, generated)
            mx.synchronize()
            del session
            mx.clear_cache()
