"""Native MLX decoder for oQ3e's per-module 3/6/8-bit affine layout.

No weight requantization and no fused INT4 projection assumptions. Prefixes
are retained as isolated snapshots on this experimental path; KV stays BF16.
"""
import logging
import time
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.sample_utils import make_sampler
from .detokenize import StreamingText


def clone_cache(cache, mx):
    """Copy array handles and metadata so mutable cache objects are never shared."""
    result = []
    for entry in cache:
        if entry.empty():
            result.append(RotatingKVCache(entry.max_size, entry.keep) if isinstance(entry, RotatingKVCache) else KVCache())
        else:
            result.append(type(entry).from_state(tuple(mx.array(x) for x in entry.state), entry.meta_state))
    mx.eval([entry.state for entry in result if not entry.empty()])
    return result


def generate(engine, ids, sampling, *, should_stop, grammar, return_logprobs):
    from .engine import ContextLimitError
    if len(ids) + sampling.max_tokens > engine.context_limit:
        raise ContextLimitError("Prompt plus output allowance exceeds the configured context limit")
    mx = engine.mx
    model = engine.runtime.model.native
    cache = [KVCache() if kind == "full_attention" else RotatingKVCache(max_size=513, keep=0)
             for kind in model.model.layer_types]
    reused = 0
    saved = getattr(engine, "_native_prefix", None)
    if saved and len(saved[0]) < len(ids) and ids[:len(saved[0])] == saved[0]:
        reused = len(saved[0])
        cache = clone_cache(saved[1], mx)
    sampler = make_sampler(temp=sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k)
    mx.random.seed(sampling.seed)
    start = time.perf_counter()
    generated, logprobs = [], []
    stream = StreamingText(engine.tokenizer)
    state = {}
    mask = mx.zeros(model.args.vocab_size, mx.float32)
    first = None
    reason = "length"
    try:
        # Stop one token before the end. That stable prefix remains valid when
        # the next request extends this prompt with generated tool calls/results.
        anchor = len(ids) - 1
        for begin in range(reused, anchor, 512):
            if should_stop(): return
            logits = model(mx.array([ids[begin:min(begin+512, anchor)]]), cache)[:, -1, :]
            mx.eval(logits, [entry.state for entry in cache])
        if should_stop(): return
        snapshot = (list(ids[:anchor]), clone_cache(cache, mx)) if anchor else None
        logits = model(mx.array([ids[-1:]]), cache)[:, -1, :]
        mx.eval(logits, [entry.state for entry in cache])
        engine._native_prefix = snapshot
        prefill_end = time.perf_counter()
        logging.info("Prefill: prompt=%d common_prefix=%d reused=%d new=%d %.0f t/s (native oQ3e)",
                     len(ids), reused, reused, len(ids)-reused, (len(ids)-reused)/max(prefill_end-start, 1e-9))
        for index in range(sampling.max_tokens):
            if should_stop(): reason = "stop"; break
            flat = logits.reshape(-1)
            if sampling.presence_penalty: flat = flat - sampling.presence_penalty * mask
            if grammar is not None: flat = grammar.first_logits(flat)
            probabilities = flat - mx.logsumexp(flat)
            token = int(sampler(probabilities[None]).item())
            if first is None: first = time.perf_counter()
            generated.append(token)
            logprobs.append(float(probabilities[token].item()))
            mask[token] = 1
            if grammar is not None:
                grammar.accept_first(token, state)
                grammar.initial_mask = state["grammar_mask"]
            stops = engine.eos if grammar is None else set(grammar.stops)
            if token in stops:
                reason = "stop"
                yield {"token": token, "text": stream.finish(generated[:-1]), "finish_reason": reason}
                break
            delta = stream.add_token(token)
            if index+1 == sampling.max_tokens: delta += stream.finish(generated)
            yield {"token": token, "text": delta, "finish_reason": "length" if index+1 == sampling.max_tokens else None}
            if index+1 < sampling.max_tokens:
                logits = model(mx.array([[token]]), cache)[:, -1, :]
                mx.eval(logits, [entry.state for entry in cache])
        end = time.perf_counter()
        decode = end - (first or end)
        yield {"finish_reason": reason, "metrics": {
            "prompt_tokens": len(ids), "cached_prompt_tokens": reused, "common_prefix_tokens": reused,
            "prefilled_tokens": len(ids)-reused, "generated_tokens": len(generated), "token_ids": generated,
            "prefill_seconds": prefill_end-start, "ttft_seconds": first-start if first else None,
            "request_seconds": end-start, "decode_seconds": decode,
            "decode_tokens_per_second": max(0,len(generated)-1)/decode if decode else 0,
            "decode_strategy": "native_mlx_oq3e", "peak_mlx_bytes": mx.get_peak_memory(),
            **({"token_logprobs":logprobs} if return_logprobs else {}),
            **(grammar.metrics() if grammar is not None else {})}}
    finally:
        mx.synchronize()
        del cache
        mx.clear_cache()
