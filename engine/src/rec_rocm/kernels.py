"""Native Triton kernels; only Torch's ROCm allocation/dispatch API is used."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _dequant(
    W, S, B, OUT, N: tl.constexpr, K: tl.constexpr, BITS: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // K, i % K
    w = tl.load(W + row * (K * BITS // 8) + col // (8 // BITS), row < N, 0).to(tl.uint32)
    q = (w >> ((col % (8 // BITS)) * BITS)) & ((1 << BITS) - 1)
    g = row * (K // 64) + col // 64
    s, b = tl.load(S + g, row < N, 0).to(tl.float32), tl.load(B + g, row < N, 0).to(tl.float32)
    tl.store(OUT + i, q.to(tl.float32) * s + b, row < N)


def dequantize(w, s, b, bits=4, dtype=torch.bfloat16, out=None):
    k = w.shape[-1] * (8 // bits)
    if out is None:
        out = torch.empty((*w.shape[:-1], k), dtype=dtype, device=w.device)
    n = out.numel() // k
    _dequant[(tr.cdiv(n * k, 1024),)](w, s, b, out, n, k, bits, 1024)
    return out


@tr.jit(do_not_specialize=["COUNT"])
def _expert_dequant(
    W,
    S,
    B,
    IDS,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    COUNT,
    SPARSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // K, i % K
    expert = tl.load(IDS + row // N, row < COUNT * N, 0)
    source_row = expert * N + row % N
    packed = tl.load(W + source_row * (K // 2) + col // 2, row < COUNT * N, 0).to(tl.uint32)
    code = (packed >> ((col % 2) * 4)) & 15
    group = source_row * (K // 64) + col // 64
    scale = tl.load(S + group, row < COUNT * N, 0).to(tl.float32)
    bias = tl.load(B + group, row < COUNT * N, 0).to(tl.float32)
    target = source_row * K + col if SPARSE else i
    tl.store(OUT + target, code.to(tl.float32) * scale + bias, row < COUNT * N)


def dequantize_experts(w, s, b, ids, out, sparse=False):
    n, k = w.shape[-2], w.shape[-1] * 2
    _expert_dequant[(tr.cdiv(ids.numel() * n * k, 1024),)](
        w, s, b, ids, out, n, k, ids.numel(), sparse, 1024
    )
    return out


@tr.jit
def _write_cache(
    K,
    V,
    POS,
    DEST_K,
    DEST_V,
    DEST_POS,
    COUNT: tl.constexpr,
    WIDTH: tl.constexpr,
    CAP: tl.constexpr,
    RING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, dim = i // WIDTH, i % WIDTH
    pos = tl.load(POS + row, row < COUNT, 0)
    slot = pos % CAP if RING else pos
    key = tl.load(K + i, row < COUNT, 0)
    value = tl.load(V + i, row < COUNT, 0)
    tl.store(DEST_K + slot * WIDTH + dim, key, row < COUNT)
    tl.store(DEST_V + slot * WIDTH + dim, value, row < COUNT)
    tl.store(DEST_POS + slot, pos, (row < COUNT) & (dim == 0))


def write_cache(key, value, positions, cache):
    _write_cache[(tr.cdiv(key.numel(), 256),)](
        key,
        value,
        positions,
        cache.key,
        cache.value,
        cache.positions,
        key.shape[0],
        key.shape[1] * key.shape[2],
        cache.capacity,
        bool(cache.window),
        256,
    )


@tr.jit
def _embedding(IDS, W, S, B, OUT, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    token = tl.load(IDS + row)
    col = tl.arange(0, BLOCK)
    q = tl.load(W + token * K + col, col < K, 0).to(tl.float32)
    scale = tl.load(S + token * (K // 64) + col // 64, col < K, 0).to(tl.float32)
    bias = tl.load(B + token * (K // 64) + col // 64, col < K, 0).to(tl.float32)
    tl.store(OUT + row * K + col, q * scale + bias, col < K)


def embedding(ids, w, s, b, dtype=torch.bfloat16):
    out = torch.empty((ids.numel(), w.shape[-1]), device=w.device, dtype=dtype)
    _embedding[(ids.numel(),)](ids, w, s, b, out, w.shape[-1], tr.next_power_of_2(w.shape[-1]))
    return out


@tr.jit
def _gemv(
    X,
    W,
    S,
    B,
    IDS,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    BITS: tl.constexpr,
    EXPERT: tl.constexpr,
    X_EXPERT: tl.constexpr,
    ROUND_BF16: tl.constexpr,
    ROWS: tl.constexpr,
    BK: tl.constexpr,
):
    expert_slot = tl.program_id(1)
    expert = tl.load(IDS + expert_slot) if EXPERT else 0
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BK)
    x = tl.load(X + (expert_slot * K if X_EXPERT else 0) + cols, cols < K, 0).to(tl.float32)
    w = tl.load(
        W + (expert * N + rows[:, None]) * (K * BITS // 8) + cols[None, :] // (8 // BITS),
        (rows[:, None] < N) & (cols[None, :] < K),
        0,
    ).to(tl.uint32)
    codes = (w >> ((cols[None, :] % (8 // BITS)) * BITS)) & ((1 << BITS) - 1)
    g = (expert * N + rows[:, None]) * (K // 64) + cols[None, :] // 64
    valid = (rows[:, None] < N) & (cols[None, :] < K)
    scale = tl.load(S + g, valid, 0).to(tl.float32)
    bias = tl.load(B + g, valid, 0).to(tl.float32)
    weight = codes.to(tl.float32) * scale + bias
    if ROUND_BF16:
        weight = weight.to(tl.bfloat16).to(tl.float32)
    y = tl.sum(weight * x[None, :], axis=1)
    tl.store(OUT + expert_slot * N + rows, y, rows < N)


def gemv(x, w, s, b, bits=4, ids=None, x_expert=False, round_bf16=False):
    n, k = w.shape[-2], w.shape[-1] * (8 // bits)
    count = 1 if ids is None else ids.numel()
    out = torch.empty((count, n), device=x.device, dtype=x.dtype)
    _gemv[(tr.cdiv(n, 4), count)](
        x,
        w,
        s,
        b,
        ids if ids is not None else x,
        out,
        n,
        k,
        bits,
        ids is not None,
        x_expert,
        round_bf16,
        4,
        tr.next_power_of_2(k),
        num_warps=4,
    )
    return out


@tr.jit
def _qmm(
    X,
    W,
    S,
    B,
    ORDER,
    TILE_EXPERT,
    TILE_START,
    COUNTS,
    OFFSETS,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    X_ASSIGNMENTS: tl.constexpr,
    BITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    expert = tl.load(TILE_EXPERT + tile)
    start = tl.load(TILE_START + tile)
    count, offset = tl.load(COUNTS + expert), tl.load(OFFSETS + expert)
    rows = start + tl.arange(0, BM)
    assignment = tl.load(ORDER + offset + rows, rows < count, 0)
    source_row = assignment if X_ASSIGNMENTS else assignment // TOPK
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    accum = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        ks = block * BK + kk
        x = tl.load(
            X + source_row[:, None] * K + ks[None, :],
            (rows[:, None] < count) & (ks[None, :] < K),
            0,
        )
        w = tl.load(
            W + (expert * N + cols[None, :]) * (K * BITS // 8) + ks[:, None] // (8 // BITS),
            (cols[None, :] < N) & (ks[:, None] < K),
            0,
        ).to(tl.uint32)
        code = (w >> ((ks[:, None] % (8 // BITS)) * BITS)) & ((1 << BITS) - 1)
        g = (expert * N + cols[None, :]) * (K // 64) + ks[:, None] // 64
        scale = tl.load(S + g, (cols[None, :] < N) & (ks[:, None] < K), 0).to(tl.float32)
        bias = tl.load(B + g, (cols[None, :] < N) & (ks[:, None] < K), 0).to(tl.float32)
        weight = (code.to(tl.float32) * scale + bias).to(x.dtype)
        accum += tl.dot(x, weight, input_precision="ieee")
    tl.store(
        OUT + assignment[:, None] * N + cols[None, :],
        accum,
        (rows[:, None] < count) & (cols[None, :] < N),
    )


def routing_layout(ids, experts=256):
    flat = ids.flatten().to(torch.int32)
    order = torch.argsort(flat, stable=True).to(torch.int32)
    counts = torch.bincount(flat.long(), minlength=experts).to(torch.int32)
    offsets = torch.cumsum(counts, 0, dtype=torch.int32) - counts
    host_counts = counts.cpu().tolist()
    tiles = [(e, row) for e, count in enumerate(host_counts) for row in range(0, count, 16)]
    te = torch.tensor([e for e, _ in tiles], device=ids.device, dtype=torch.int32)
    ts = torch.tensor([row for _, row in tiles], device=ids.device, dtype=torch.int32)
    return order, te, ts, counts, offsets


_DENSE_LAYOUTS = {}


@tr.jit
def _decode_dot(
    X,
    W,
    S,
    B,
    IDS,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    EXPERT: tl.constexpr,
    X_EXPERT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    slot = tl.program_id(1)
    expert = tl.load(IDS + slot) if EXPERT else 0
    rows = tl.arange(0, 16)
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    accum = tl.full((16, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        ks = block * BK + kk
        x = tl.load(
            X + (slot * K if X_EXPERT else 0) + ks[None, :],
            (rows[:, None] == 0) & (ks[None, :] < K),
            0,
        )
        w = tl.load(
            W + (expert * N + cols[None, :]) * (K // 2) + ks[:, None] // 2,
            (cols[None, :] < N) & (ks[:, None] < K),
            0,
        ).to(tl.uint32)
        code = (w >> ((ks[:, None] % 2) * 4)) & 15
        g = (expert * N + cols[None, :]) * (K // 64) + ks[:, None] // 64
        scale = tl.load(S + g, (cols[None, :] < N) & (ks[:, None] < K), 0).to(tl.float32)
        bias = tl.load(B + g, (cols[None, :] < N) & (ks[:, None] < K), 0).to(tl.float32)
        weight = (code.to(tl.float32) * scale + bias).to(tl.bfloat16)
        accum += tl.dot(x, weight)
    result = tl.sum(accum, axis=0)
    tl.store(OUT + slot * N + cols, result, cols < N)


def decode_dot(x, w, s, b, ids=None, x_expert=False):
    n, k = w.shape[-2], w.shape[-1] * 2
    count = 1 if ids is None else ids.numel()
    out = torch.empty((count, n), device=x.device, dtype=x.dtype)
    _decode_dot[(tr.cdiv(n, 64), count)](
        x,
        w,
        s,
        b,
        ids if ids is not None else x,
        out,
        n,
        k,
        ids is not None,
        x_expert,
        64,
        64,
        num_warps=4,
    )
    return out


def quantized_mm(x, w, s, b, bits=4, ids=None, layout=None, x_assignments=False):
    n, dim = w.shape[-2], w.shape[-1] * (8 // bits)
    if ids is None:
        m = x.shape[0]
        key = (str(x.device), m)
        if key not in _DENSE_LAYOUTS:
            _DENSE_LAYOUTS[key] = routing_layout(
                torch.zeros((m, 1), device=x.device, dtype=torch.int32), 1
            )
        layout = _DENSE_LAYOUTS[key]
        topk, count = 1, m
    else:
        topk, count = ids.shape[-1], ids.numel()
        if layout is None:
            layout = routing_layout(ids, w.shape[0])
    out = torch.empty((count, n), device=x.device, dtype=x.dtype)
    order, te, ts, counts, offsets = layout
    _qmm[(te.numel(), tr.cdiv(n, 64))](
        x,
        w,
        s,
        b,
        order,
        te,
        ts,
        counts,
        offsets,
        out,
        n,
        dim,
        topk,
        x_assignments,
        bits,
        16,
        64,
        64,
        num_warps=4,
    )
    return out


@tr.jit
def _norm(X, W, OUT, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + col, col < D, 0).to(tl.float32)
    w = tl.load(W + col, col < D, 0).to(tl.float32)
    norm = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    tl.store(OUT + row * D + col, x * norm * w, col < D)


def rms_norm(x, w, eps=1e-6):
    out = torch.empty_like(x)
    _norm[(x.numel() // x.shape[-1],)](x, w, out, x.shape[-1], eps, tr.next_power_of_2(x.shape[-1]))
    return out


@tr.jit
def _attn(
    Q,
    K,
    V,
    S,
    OUT,
    QPOS,
    KPOS,
    M: tl.constexpr,
    T: tl.constexpr,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    h = tl.program_id(1)
    kh = h // (QH // KVH)
    qi = tl.program_id(0) * BM + tl.arange(0, BM)
    d = tl.arange(0, BD)
    q = tl.load(
        Q + qi[:, None] * QH * D + h * D + d[None, :], (qi[:, None] < M) & (d[None, :] < D), 0
    )
    pos = tl.load(QPOS + qi, qi < M, -1)
    # Treat the learned sink as a virtual score with a zero value vector.
    mx = tl.full((BM,), 0, tl.float32) + tl.load(S + h).to(tl.float32)
    denom = tl.full((BM,), 1, tl.float32)
    acc = tl.full((BM, BD), 0, tl.float32)
    for start in range(0, tl.cdiv(T, BN)):
        ki = start * BN + tl.arange(0, BN)
        kp = tl.load(KPOS + ki, ki < T, -10000000)
        k = tl.load(
            K + ki[None, :] * KVH * D + kh * D + d[:, None], (ki[None, :] < T) & (d[:, None] < D), 0
        )
        score = tl.dot(q.to(tl.float32), k.to(tl.float32), input_precision="ieee") * (D**-0.5)
        allowed = (ki[None, :] < T) & (kp[None, :] <= pos[:, None]) & (kp[None, :] >= 0)
        if WINDOW:
            allowed &= kp[None, :] > pos[:, None] - WINDOW
        score = tl.where(allowed, score, -float("inf"))
        new_max = tl.maximum(mx, tl.max(score, axis=1))
        correction = tl.exp(mx - new_max)
        p = tl.exp(score - new_max[:, None])
        denom = denom * correction + tl.sum(p, axis=1)
        value = tl.load(
            V + ki[:, None] * KVH * D + kh * D + d[None, :], (ki[:, None] < T) & (d[None, :] < D), 0
        )
        acc = acc * correction[:, None] + tl.dot(p, value.to(tl.float32), input_precision="ieee")
        mx = new_max
    result = acc / denom[:, None]
    tl.store(
        OUT + qi[:, None] * QH * D + h * D + d[None, :],
        result,
        (qi[:, None] < M) & (d[None, :] < D),
    )


def attention(q, k, v, sinks, qpositions, kpositions, window=None, dtype=None):
    m, h, d = q.shape
    out = torch.empty(q.shape, device=q.device, dtype=dtype or q.dtype)
    _attn[(tr.cdiv(m, 16), h)](
        q,
        k,
        v,
        sinks,
        out,
        qpositions,
        kpositions,
        m,
        k.shape[0],
        h,
        k.shape[1],
        d,
        window or 0,
        16,
        64,
        tr.next_power_of_2(d),
        num_warps=4,
    )
    return out
