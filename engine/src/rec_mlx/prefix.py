import copy
import logging
from dataclasses import dataclass, field

import mlx.core as mx

from .cache import MAX_REWIND_TOKENS, FixedSlidingKVCache, copy_array_async


class PrefillCancelled(Exception):
    pass


@dataclass
class Checkpoint:
    position: int
    sliding: list
    logits: mx.array


@dataclass
class BranchSnapshot:
    tokens: list[int]
    global_kv: dict[int, mx.array]
    sliding: list[tuple]
    logits: mx.array | None


class PrefixTreeCache:
    """Multi-branch LRU prefix cache for fast switching across files and sessions."""

    def __init__(self, max_branches: int = 4):
        self.max_branches = max_branches
        self.branches: list[BranchSnapshot] = []
        self.base_branch: BranchSnapshot | None = None

    def clear(self):
        self.branches.clear()
        self.base_branch = None

    @staticmethod
    def _lcp(a: list[int], b: list[int]) -> int:
        common = 0
        for x, y in zip(a, b):
            if x != y:
                break
            common += 1
        return common

    def _snapshot(self, cache: "PromptCache", target_len: int) -> BranchSnapshot | None:
        storage = cache.storage
        if not (hasattr(storage, "layer") and hasattr(storage, "write_async")):
            return None
        global_kv = {}
        sliding = []
        pending = []
        for index, layer in enumerate(cache.layers):
            if isinstance(layer, FixedSlidingKVCache):
                if target_len == len(cache.tokens):
                    k_buf = copy_array_async(layer.key_buffers[layer._slot])
                    v_buf = copy_array_async(layer.value_buffers[layer._slot])
                    valid = layer._valid
                    ring_idx = layer._ring_index
                    hist_k = (
                        copy_array_async(layer._history_k[..., :target_len, :])
                        if layer._history_k is not None
                        else None
                    )
                    hist_v = (
                        copy_array_async(layer._history_v[..., :target_len, :])
                        if layer._history_v is not None
                        else None
                    )
                else:
                    if not layer.can_rewind_to(target_len):
                        return None
                    retained = min(layer.capacity, target_len)
                    if layer.offset <= layer.capacity and layer._ring_index == 0:
                        src_k = layer.key_buffers[layer._slot][..., :retained, :]
                        src_v = layer.value_buffers[layer._slot][..., :retained, :]
                    else:
                        src_k = layer._history_k[..., target_len - retained : target_len, :]
                        src_v = layer._history_v[..., target_len - retained : target_len, :]
                    k_buf = mx.zeros_like(layer.key_buffers[0])
                    v_buf = mx.zeros_like(layer.value_buffers[0])
                    k_buf[..., :retained, :] = src_k
                    v_buf[..., :retained, :] = src_v
                    k_buf = mx.contiguous(k_buf)
                    v_buf = mx.contiguous(v_buf)
                    valid = retained
                    ring_idx = 0
                    hist_k = (
                        copy_array_async(layer._history_k[..., :target_len, :])
                        if layer._history_k is not None
                        else None
                    )
                    hist_v = (
                        copy_array_async(layer._history_v[..., :target_len, :])
                        if layer._history_v is not None
                        else None
                    )
                pending.extend([k_buf, v_buf])
                if hist_k is not None:
                    pending.extend([hist_k, hist_v])
                sliding.append((k_buf, v_buf, valid, ring_idx, hist_k, hist_v))
            else:
                kv = copy_array_async(storage.layer(index, target_len))
                global_kv[index] = kv
                pending.append(kv)
        logits = cache.logits if target_len == len(cache.tokens) else None
        if logits is not None:
            pending.append(logits)
        if pending:
            mx.eval(pending)
        return BranchSnapshot(
            tokens=cache.tokens[:target_len],
            global_kv=global_kv,
            sliding=sliding,
            logits=logits,
        )

    def save(self, cache: "PromptCache"):
        if cache is None or not cache.tokens:
            return
        n_tok = len(cache.tokens)
        if n_tok > MAX_REWIND_TOKENS:
            return
        for i, branch in enumerate(self.branches):
            if branch.tokens == cache.tokens:
                branch.logits = cache.logits
                self.branches.append(self.branches.pop(i))
                return
        snap = self._snapshot(cache, n_tok)
        if snap is None:
            return
        self.branches.append(snap)
        if len(self.branches) > self.max_branches:
            self.branches.pop(0)
        if self.base_branch is None and n_tok >= 256:
            base_len = min(n_tok, 1024)
            self.base_branch = snap if base_len == n_tok else self._snapshot(cache, base_len)

    def select_best(self, tokens: list[int], active_cache: "PromptCache | None"):
        if active_cache is None:
            self.clear()
            return None
        active_common = self._lcp(active_cache.tokens, tokens)
        best_snap = None
        best_common = active_common
        candidates = list(reversed(self.branches))
        if self.base_branch is not None:
            candidates.append(self.base_branch)
        for snap in candidates:
            common = self._lcp(snap.tokens, tokens)
            if common > best_common:
                best_common = common
                best_snap = snap
        if best_snap is None:
            return active_cache
        shared_with_active = self._lcp(active_cache.tokens, best_snap.tokens)
        target_len = len(best_snap.tokens)
        storage = active_cache.storage
        dones = []
        if shared_with_active < target_len:
            for index, kv in best_snap.global_kv.items():
                k_slice = kv[0:1, :, shared_with_active:target_len, :]
                v_slice = kv[1:2, :, shared_with_active:target_len, :]
                dones.extend(storage.write_async(index, shared_with_active, k_slice, v_slice))
        sliding_iter = iter(best_snap.sliding)
        eval_tensors = list(dones)
        for layer in active_cache.layers:
            layer.offset = target_len
            if isinstance(layer, FixedSlidingKVCache):
                k_buf, v_buf, valid, ring_idx, hist_k, hist_v = next(sliding_iter)
                k_copy = copy_array_async(k_buf)
                v_copy = copy_array_async(v_buf)
                layer.capacity = k_copy.shape[-2]
                layer.key_buffers = (k_copy, mx.zeros_like(k_copy))
                layer.value_buffers = (v_copy, mx.zeros_like(v_copy))
                layer._slot = 0
                layer._valid = valid
                layer._ring_index = ring_idx
                layer._history_k = copy_array_async(hist_k) if hist_k is not None else None
                layer._history_v = copy_array_async(hist_v) if hist_v is not None else None
                eval_tensors.extend([layer.key_buffers[0], layer.value_buffers[0]])
                if layer._history_k is not None:
                    eval_tensors.extend([layer._history_k, layer._history_v])
        if eval_tensors:
            mx.eval(eval_tensors)
        active_cache.tokens = best_snap.tokens[:]
        active_cache.logits = best_snap.logits
        active_cache.checkpoints = []
        if best_snap in self.branches:
            self.branches.remove(best_snap)
            self.branches.append(best_snap)
        return active_cache


@dataclass
class PromptCache:
    tokens: list[int]
    layers: list
    logits: mx.array | None
    storage: object
    checkpoints: list[Checkpoint] = field(default_factory=list)

    def fork(self):
        layers = [copy.copy(layer) for layer in self.layers]
        pending = []
        for layer in layers:
            if isinstance(layer, FixedSlidingKVCache):
                keys = copy_array_async(layer.key_buffers[layer._slot])
                values = copy_array_async(layer.value_buffers[layer._slot])
                layer.key_buffers = (keys, mx.zeros_like(keys))
                layer.value_buffers = (values, mx.zeros_like(values))
                layer._slot = 0
                pending.extend([keys, values, layer.key_buffers[1], layer.value_buffers[1]])
                if layer._history_k is not None:
                    layer._history_k = copy_array_async(layer._history_k)
                    layer._history_v = copy_array_async(layer._history_v)
                    pending.extend([layer._history_k, layer._history_v])
        if pending:
            mx.eval(pending)
        return PromptCache(self.tokens[:], layers, self.logits, self.storage, self.checkpoints[:])

    def advance(self, tokens, logits):
        previous = len(self.tokens)
        self.tokens.extend(tokens)
        self.logits = logits
        if len(self.tokens) // 8192 != previous // 8192:
            logging.info("Prefill progress: %d tokens processed", len(self.tokens))

    def checkpoint(self):
        if not self.tokens:
            return
        self.checkpoints = self.checkpoints[-1:]
        sliding = []
        for layer in self.layers:
            if isinstance(layer, FixedSlidingKVCache):
                sliding.append(
                    (
                        copy_array_async(layer.key_buffers[layer._slot]),
                        copy_array_async(layer.value_buffers[layer._slot]),
                        layer._valid,
                        layer._ring_index,
                    )
                )
        mx.eval(sliding)
        self.checkpoints.append(Checkpoint(len(self.tokens), sliding, self.logits))

    @property
    def checkpoint_bytes(self):
        total = 0
        for checkpoint in self.checkpoints:
            total += sum(keys.nbytes + values.nbytes for keys, values, _, _ in checkpoint.sliding)
            total += checkpoint.logits.nbytes
        return total


# Never stage decode KV: its arithmetic differs from batched prefill and changes probabilities.
@dataclass
class PrefixPreparation:
    origin: PromptCache
    tokens: list[int]
    cache: PromptCache | None = None


def select_prefix(prefix, tokens):
    common = 0
    if prefix is not None:
        for previous, current in zip(prefix.tokens, tokens):
            if previous != current:
                break
            common += 1
        if common != len(prefix.tokens):
            checkpoint = next(
                (item for item in reversed(prefix.checkpoints) if item.position <= common), None
            )
            if checkpoint is not None:
                sliding = iter(checkpoint.sliding)
                for layer in prefix.layers:
                    layer.offset = checkpoint.position
                    if isinstance(layer, FixedSlidingKVCache):
                        keys, values, layer._valid, layer._ring_index = next(sliding)
                        layer.capacity = keys.shape[-2]
                        layer.key_buffers = (keys, mx.zeros_like(keys))
                        layer.value_buffers = (values, mx.zeros_like(values))
                        layer._slot = 0
                        if (
                            layer._history_k is not None
                            and checkpoint.position < layer._history_k.shape[-2]
                        ):
                            layer._history_k = layer._history_k[..., : checkpoint.position, :]
                            layer._history_v = layer._history_v[..., : checkpoint.position, :]
                prefix.tokens = prefix.tokens[: checkpoint.position]
                prefix.logits = checkpoint.logits
                prefix.checkpoints = [
                    item for item in prefix.checkpoints if item.position < checkpoint.position
                ]
            elif not prefix.checkpoints:
                target = min(common, len(tokens) - 1)
                sliding_layers = [
                    layer for layer in prefix.layers if isinstance(layer, FixedSlidingKVCache)
                ]
                if (
                    sliding_layers
                    and 0 < target <= MAX_REWIND_TOKENS
                    and all(layer.can_rewind_to(target) for layer in sliding_layers)
                ):
                    for layer in prefix.layers:
                        if isinstance(layer, FixedSlidingKVCache):
                            layer.rewind_to(target)
                        else:
                            layer.offset = target
                    mx.eval([layer.state for layer in sliding_layers])
                    prefix.tokens = prefix.tokens[:target]
                    prefix.logits = None
                else:
                    prefix = None
            else:
                prefix = None
    cached = 0 if prefix is None else len(prefix.tokens)
    logging.info(
        "Prefill: prompt=%d common_prefix=%d reused=%d new=%d",
        len(tokens),
        common,
        cached,
        len(tokens) - cached,
    )
    return prefix, common
