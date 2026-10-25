"""Opt-in gfx1201 candidates; enable only after full-checkpoint qualification."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _norm_apply(X, W, INV, OUT, D: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < N, 0).to(tl.float32)
    weight = tl.load(W + index % D, index < N, 0).to(tl.float32)
    inv = tl.load(INV + index // D, index < N, 0)
    result = (x * inv) * weight
    tl.store(OUT + index, result, index < N)


def norm_apply(x, weight, eps):
    inverse = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    out = torch.empty_like(x)
    _norm_apply[(tr.cdiv(x.numel(), 256),)](
        x, weight, inverse, out, x.shape[-1], x.numel(), 256, enable_fp_fusion=False
    )
    return out


@tr.jit
def _rope_apply(
    X, COS, SIN, OUT, N: tl.constexpr, HEADS: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    half = D // 2
    row = index // half
    dim = index % half
    token = row // HEADS
    valid = index < N // 2
    left = tl.load(X + row * D + dim, valid, 0).to(tl.float32)
    right = tl.load(X + row * D + half + dim, valid, 0).to(tl.float32)
    cosine = tl.load(COS + token * half + dim, valid, 0)
    sine = tl.load(SIN + token * half + dim, valid, 0)
    first = left * cosine - right * sine
    second = right * cosine + left * sine
    tl.store(OUT + row * D + dim, first, valid)
    tl.store(OUT + row * D + half + dim, second, valid)


def rope_apply(x, angles):
    out = torch.empty_like(x)
    _rope_apply[(tr.cdiv(x.numel() // 2, 256),)](
        x,
        angles[0],
        angles[1],
        out,
        x.numel(),
        x.shape[1],
        x.shape[2],
        256,
        enable_fp_fusion=False,
    )
    return out


@tr.jit
def _mask_sink(
    SCORE,
    SINK,
    QP,
    KP,
    OUT,
    H: tl.constexpr,
    M: tl.constexpr,
    T: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = index % (T + 1)
    row = index // (T + 1)
    qrow = row % M
    head = row // M
    valid = row < H * M
    qp = tl.load(QP + qrow, valid, -1)
    kp = tl.load(KP + col, valid & (col < T), -1)
    score = tl.load(SCORE + row * T + col, valid & (col < T), 0)
    allowed = (kp >= 0) & (kp <= qp)
    if WINDOW:
        allowed &= kp > qp - WINDOW
    score = tl.where(allowed, score, -float("inf"))
    sink = tl.load(SINK + head, valid, 0)
    score = tl.where(col == T, sink, score)
    tl.store(OUT + index, score, valid)


def mask_sink(scores, sinks, qpositions, kpositions, window=None):
    heads, count, length = scores.shape
    scores = scores.contiguous()
    out = torch.empty((heads, count, length + 1), device=scores.device, dtype=scores.dtype)
    _mask_sink[(tr.cdiv(out.numel(), 512),)](
        scores, sinks, qpositions, kpositions, out, heads, count, length, window or 0, 512
    )
    return out


@tr.jit
def _decode_attention(
    Q,
    K,
    V,
    S,
    QP,
    KP,
    PART,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    SPLITS: tl.constexpr,
    TILE: tl.constexpr,
    BD: tl.constexpr,
):
    head = tl.program_id(0)
    split = tl.program_id(1)
    kvhead = head // (H // KH)
    dim = tl.arange(0, BD)
    q = tl.load(Q + head * D + dim, dim < D, 0).to(tl.float32)
    pos = tl.load(QP)
    # Full KV is contiguous; sliding KV uses a ring, so positions mask its slots.
    extent = T if WINDOW else tl.minimum(T, pos + 1)
    tiles = tl.cdiv(extent, TILE)
    per_split = tl.cdiv(tiles, SPLITS)
    begin = split * per_split * TILE
    end = tl.minimum((split + 1) * per_split * TILE, extent)
    mx = tl.full((), -float("inf"), tl.float32)
    denom = tl.full((), 0, tl.float32)
    accum = tl.full((BD,), 0, tl.float32)
    for start in range(begin, end, TILE):
        index = start + tl.arange(0, TILE)
        kp = tl.load(KP + index, index < extent, -1)
        valid = (index < extent) & (kp >= 0) & (kp <= pos)
        if WINDOW:
            valid &= kp > pos - WINDOW
        key = tl.load(
            K + index[:, None] * KH * D + kvhead * D + dim[None, :],
            valid[:, None] & (dim[None, :] < D),
            0,
        ).to(tl.float32)
        score = tl.sum(key * q[None, :], 1) * (D**-0.5)
        score = tl.where(valid, score, -float("inf"))
        new_max = tl.maximum(mx, tl.max(score, 0))
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        correction = tl.exp(mx - safe_max)
        prob = tl.exp(score - safe_max)
        value = tl.load(
            V + index[:, None] * KH * D + kvhead * D + dim[None, :],
            valid[:, None] & (dim[None, :] < D),
            0,
        ).to(tl.float32)
        accum = accum * correction + tl.sum(prob[:, None] * value, 0)
        denom = denom * correction + tl.sum(prob, 0)
        mx = new_max
    base = (head * SPLITS + split) * (D + 2)
    tl.store(PART + base + dim, accum, dim < D)
    tl.store(PART + base + D, mx)
    tl.store(PART + base + D + 1, denom)


@tr.jit
def _merge_attention(
    PART, S, OUT, D: tl.constexpr, SPLITS: tl.constexpr, BD: tl.constexpr, BS: tl.constexpr
):
    head = tl.program_id(0)
    split = tl.arange(0, BS)
    dim = tl.arange(0, BD)
    base = (head * SPLITS + split) * (D + 2)
    mx = tl.load(PART + base + D, split < SPLITS, -float("inf"))
    den = tl.load(PART + base + D + 1, split < SPLITS, 0)
    sink = tl.load(S + head).to(tl.float32)
    maximum = tl.maximum(tl.max(mx, 0), sink)
    scales = tl.exp(mx - maximum)
    numerator = tl.load(
        PART + base[:, None] + dim[None, :], (split[:, None] < SPLITS) & (dim[None, :] < D), 0
    )
    total = tl.sum(den * scales, 0) + tl.exp(sink - maximum)
    result = tl.sum(numerator * scales[:, None], 0) / total
    tl.store(OUT + head * D + dim, result, dim < D)


def sink_attention(q, key, value, sinks, qpositions, kpositions, window=None):
    if q.shape[0] != 1:
        from .kernels import attention

        return attention(
            q.contiguous(),
            key.contiguous(),
            value.contiguous(),
            sinks,
            qpositions,
            kpositions,
            window,
            dtype=q.dtype,
        )
    _, heads, dim = q.shape
    splits = 1 if window else min(8, max(1, tr.cdiv(key.shape[0], 1024)))
    partial = torch.empty((heads, splits, dim + 2), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    _decode_attention[(heads, splits)](
        q,
        key,
        value,
        sinks,
        qpositions,
        kpositions,
        partial,
        out,
        key.shape[0],
        heads,
        key.shape[1],
        dim,
        window or 0,
        splits,
        64,
        tr.next_power_of_2(dim),
        num_warps=4,
        enable_fp_fusion=False,
    )
    _merge_attention[(heads,)](
        partial,
        sinks,
        out,
        dim,
        splits,
        tr.next_power_of_2(dim),
        tr.next_power_of_2(splits),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
