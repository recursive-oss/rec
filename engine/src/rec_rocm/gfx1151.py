"""Opt-in gfx1151 decode kernels; enable only after full-checkpoint qualification.

Each fused kernel reproduces the rounding of the eager path it replaces: intermediates that
PyTorch stores as BF16 are rounded to BF16, and exp uses libdevice (tl.exp is approximate).
"""

import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice


@tr.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@tr.jit
def _w4_dot(
    W,
    S,
    B,
    base,
    row_ok,
    X,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    GB: tl.constexpr,
    ROUND_BF16: tl.constexpr,
):
    """Dot of packed INT4 rows `base` (row index into [*, K/8] int32 words) with BF16 vector X."""
    groups = tl.arange(0, GB)
    words = tl.arange(0, 8)
    acc = tl.zeros((ROWS, GB, 8), tl.float32)
    for first in range(0, K // 64, GB):
        group = first + groups
        ok_group = row_ok[:, None] & (group[None, :] < K // 64)
        ok = ok_group[:, :, None] & (words[None, None, :] < 8)
        word = group[:, None] * 8 + words[None, :]
        packed = tl.load(W + base[:, None, None] * (K // 8) + word[None, :, :], ok, 0)
        scale = tl.load(S + base[:, None] * (K // 64) + group[None, :], ok_group, 0).to(tl.float32)
        bias = tl.load(B + base[:, None] * (K // 64) + group[None, :], ok_group, 0).to(tl.float32)
        for nibble in tl.static_range(8):
            weight = ((packed >> (4 * nibble)) & 15).to(tl.float32) * scale[:, :, None] + bias[
                :, :, None
            ]
            if ROUND_BF16:
                # Round-to-nearest-even to BF16 on the bit pattern; weights are finite.
                bits = weight.to(tl.uint32, bitcast=True)
                bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
                weight = bits.to(tl.float32, bitcast=True)
            x = tl.load(X + word * 8 + nibble, group[:, None] < K // 64, 0).to(tl.float32)
            acc += weight * x[None, :, :]
    return tl.sum(tl.sum(acc, axis=2), axis=1)


@tr.jit
def _gemv_w4(
    X,
    W,
    S,
    B,
    IDS,
    RES,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    EXPERT: tl.constexpr,
    X_EXPERT: tl.constexpr,
    ROUND_BF16: tl.constexpr,
    HAS_RES: tl.constexpr,
    ROWS: tl.constexpr,
    GB: tl.constexpr,
):
    slot = tl.program_id(1)
    expert = tl.load(IDS + slot) if EXPERT else 0
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    row_ok = rows < N
    x = X + (slot * K if X_EXPERT else 0)
    total = _w4_dot(W, S, B, expert * N + rows, row_ok, x, K, ROWS, GB, ROUND_BF16)
    if HAS_RES:
        total = _bf16(total) + tl.load(RES + rows, row_ok, 0).to(tl.float32)
    tl.store(OUT + slot * N + rows, total.to(OUT.dtype.element_ty), row_ok)


@tr.jit
def _gate_up_w4(
    X,
    WG,
    SG,
    BG,
    WU,
    SU,
    BU,
    IDS,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    EXPERT: tl.constexpr,
    ROWS: tl.constexpr,
    GB: tl.constexpr,
):
    slot = tl.program_id(1)
    expert = tl.load(IDS + slot) if EXPERT else 0
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    row_ok = rows < N
    base = expert * N + rows
    gate = _bf16(_w4_dot(WG, SG, BG, base, row_ok, X, K, ROWS, GB, True))
    up = _bf16(_w4_dot(WU, SU, BU, base, row_ok, X, K, ROWS, GB, True))
    act = _bf16(tl.div_rn(gate, 1.0 + libdevice.exp(-gate)))
    tl.store(OUT + slot * N + rows, (act * up).to(tl.bfloat16), row_ok)


@tr.jit
def _down_combine_w4(
    ACT,
    WD,
    SD,
    BD,
    IDS,
    SCORES,
    SACT,
    WS,
    SS,
    BS,
    RES,
    OUT,
    N: tl.constexpr,
    KE: tl.constexpr,
    KS: tl.constexpr,
    TOPK: tl.constexpr,
    ROWS: tl.constexpr,
    GBE: tl.constexpr,
    GBS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    row_ok = rows < N
    routed = tl.zeros((ROWS,), tl.float32)
    for j in tl.static_range(TOPK):
        expert = tl.load(IDS + j)
        down = _bf16(
            _w4_dot(WD, SD, BD, expert * N + rows, row_ok, ACT + j * KE, KE, ROWS, GBE, True)
        )
        routed += _bf16(down * tl.load(SCORES + j).to(tl.float32))
    shared = _bf16(_w4_dot(WS, SS, BS, rows, row_ok, SACT, KS, ROWS, GBS, True))
    y = _bf16(_bf16(routed) + shared)
    x = tl.load(RES + rows, row_ok, 0).to(tl.float32)
    tl.store(OUT + rows, (x + y).to(tl.bfloat16), row_ok)


@tr.jit
def _rms_norm(X, W, OUT, EPS, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BD)
    ok = cols < D
    x = tl.load(X + row * D + cols, ok, 0).to(tl.float32)
    inverse = libdevice.rsqrt(tl.sum(x * x, 0) / D + EPS)
    y = (x * inverse) * tl.load(W + cols, ok, 0).to(tl.float32)
    tl.store(OUT + row * D + cols, y.to(tl.bfloat16), ok)


@tr.jit
def _qk_post(
    QKV,
    QW,
    KW,
    COS,
    SIN,
    QOUT,
    KC,
    VC,
    PC,
    POS,
    EPS,
    CAP: tl.constexpr,
    RING: tl.constexpr,
    ROPE: tl.constexpr,
    H: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
):
    """One program per head of the fused q|k|v vector: q/k RMSNorm, RoPE, and the KV-cache write."""
    head = tl.program_id(0)
    half = tl.arange(0, D // 2)
    left = tl.load(QKV + head * D + half).to(tl.float32)
    right = tl.load(QKV + head * D + D // 2 + half).to(tl.float32)
    pos = tl.load(POS)
    slot = pos % CAP if RING else pos
    if head < H + KH:
        is_q = (half * 0 + head) < H
        wl = tl.load(QW + half, is_q, 0).to(tl.float32) + tl.load(KW + half, ~is_q, 0).to(
            tl.float32
        )
        wr = tl.load(QW + D // 2 + half, is_q, 0).to(tl.float32) + tl.load(
            KW + D // 2 + half, ~is_q, 0
        ).to(tl.float32)
        inverse = libdevice.rsqrt((tl.sum(left * left, 0) + tl.sum(right * right, 0)) / D + EPS)
        left = _bf16((left * inverse) * wl)
        right = _bf16((right * inverse) * wr)
        if ROPE:
            c = tl.load(COS + half)
            s = tl.load(SIN + half)
            left, right = left * c - right * s, right * c + left * s
        if head < H:
            tl.store(QOUT + head * D + half, left.to(tl.bfloat16))
            tl.store(QOUT + head * D + D // 2 + half, right.to(tl.bfloat16))
        else:
            k = KC + slot * KH * D + (head - H) * D
            tl.store(k + half, left.to(tl.bfloat16))
            tl.store(k + D // 2 + half, right.to(tl.bfloat16))
            if head == H:
                tl.store(PC + slot, pos)
    else:
        v = VC + slot * KH * D + (head - H - KH) * D
        tl.store(v + half, left.to(tl.bfloat16))
        tl.store(v + D // 2 + half, right.to(tl.bfloat16))


@tr.jit
def _router_logits(
    H, GW, OUT, N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr, BK: tl.constexpr
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BK)
    acc = tl.zeros((ROWS, BK), tl.float32)
    for first in range(0, K, BK):
        k = first + cols
        x = tl.load(H + k, k < K, 0).to(tl.float32)
        w = tl.load(GW + rows[:, None] * K + k[None, :], (rows[:, None] < N) & (k[None, :] < K), 0)
        acc += w.to(tl.float32) * x[None, :]
    tl.store(OUT + rows, _bf16(tl.sum(acc, axis=1)), rows < N)


@tr.jit
def _router_topk(
    LOGIT,
    BIAS,
    IDS,
    SCORES,
    E: tl.constexpr,
    BE: tl.constexpr,
    TOPK: tl.constexpr,
    BT: tl.constexpr,
):
    """Top-k of sigmoid(logit) + bias, ties to the lower index as a stable descending argsort."""
    experts = tl.arange(0, BE)
    valid = experts < E
    logit = tl.load(LOGIT + experts, valid, 0)
    route = tl.div_rn(tl.full(logit.shape, 1.0, tl.float32), 1.0 + libdevice.exp(-logit))
    choice = tl.where(
        valid, route + tl.load(BIAS + experts, valid, 0).to(tl.float32), -float("inf")
    )
    lanes = tl.arange(0, BT)
    picked = tl.zeros((BT,), tl.float32)
    for j in tl.static_range(TOPK):
        best = tl.max(choice, 0)
        index = tl.min(tl.where(choice == best, experts, BE), 0)
        tl.store(IDS + j, index)
        picked = tl.where(lanes == j, tl.sum(tl.where(experts == index, route, 0.0), 0), picked)
        choice = tl.where(experts == index, -float("inf"), choice)
    if TOPK == 6:
        s0 = tl.sum(tl.where(lanes == 0, picked, 0.0), 0)
        s1 = tl.sum(tl.where(lanes == 1, picked, 0.0), 0)
        s2 = tl.sum(tl.where(lanes == 2, picked, 0.0), 0)
        s3 = tl.sum(tl.where(lanes == 3, picked, 0.0), 0)
        s4 = tl.sum(tl.where(lanes == 4, picked, 0.0), 0)
        s5 = tl.sum(tl.where(lanes == 5, picked, 0.0), 0)
        # Same association as torch.sum over six float values on gfx1151 (measured bit for bit).
        total = ((s0 + s4) + (s1 + s5)) + (s2 + s3)
    else:
        # Not verified bit for bit against torch.sum for this k.
        total = tl.sum(picked, 0)
    total = total + 1e-20
    tl.store(
        SCORES + lanes,
        tl.div_rn(picked, tl.zeros_like(picked) + total).to(tl.bfloat16),
        lanes < TOPK,
    )


# (N, K) -> (rows, groups, warps), swept per shape on gfx1151 (bench/gemv_bench.py).
_CONFIG = {
    (512, 2048): (8, 16, 8),
    (2048, 512): (8, 4, 2),
    (1024, 2048): (2, 16, 8),
    (4096, 2048): (4, 16, 8),
    (2048, 4096): (4, 16, 8),
    (150144, 2048): (8, 8, 4),
    (5120, 2048): (4, 16, 8),
}


def _default(k):
    return (4, 1 << (min(16, k // 64).bit_length() - 1), 8)


def gemv_w4(x, w, s, b, ids=None, x_expert=False, round_bf16=True, config=None, residual=None):
    """Fused INT4 GEMV: x [1 or count, K] with packed w [..., N, K/2], scales/biases [..., N, K/64]."""
    n, k = w.shape[-2], w.shape[-1] * 2
    rows, groups, warps = config or _CONFIG.get((n, k), _default(k))
    count = 1 if ids is None else ids.numel()
    out = torch.empty((count, n), device=x.device, dtype=x.dtype)
    _gemv_w4[(tr.cdiv(n, rows), count)](
        x,
        w.view(torch.int32),
        s,
        b,
        ids if ids is not None else x,
        residual if residual is not None else x,
        out,
        n,
        k,
        ids is not None,
        x_expert,
        round_bf16,
        residual is not None,
        rows,
        groups,
        num_warps=warps,
    )
    return out


# (N, K, experts) -> (rows, groups, warps), swept over all layers' weights (bench/w2_sweep.py).
_GATE_UP_CONFIG = {(512, 2048, True): (8, 16, 8), (1024, 2048, False): (2, 16, 8)}


def gate_up_w4(x, wg, sg, bg, wu, su, bu, ids=None, config=None):
    """silu(x @ gate^T) * (x @ up^T) for one dense MLP or for each selected expert."""
    n, k = wg.shape[-2], wg.shape[-1] * 2
    count = 1 if ids is None else ids.numel()
    rows, groups, warps = config or _GATE_UP_CONFIG.get((n, k, ids is not None), _default(k))
    out = torch.empty((count, n), device=x.device, dtype=x.dtype)
    _gate_up_w4[(tr.cdiv(n, rows), count)](
        x,
        wg.view(torch.int32),
        sg,
        bg,
        wu.view(torch.int32),
        su,
        bu,
        ids if ids is not None else x,
        out,
        n,
        k,
        ids is not None,
        rows,
        groups,
        num_warps=warps,
    )
    return out


_DOWN_CONFIG = (4, 4, 4, 2)


def down_combine_w4(act, wd, sd, bd, ids, scores, shared_act, ws, ss, bs, residual, config=None):
    """residual + (sum_j score_j * down_j(act_j) + shared_down(shared_act)) with the eager path's rounding."""
    n, ke, ks = wd.shape[-2], wd.shape[-1] * 2, ws.shape[-1] * 2
    rows, gbe, gbs, warps = config or _DOWN_CONFIG
    out = torch.empty_like(residual)
    _down_combine_w4[(tr.cdiv(n, rows),)](
        act,
        wd.view(torch.int32),
        sd,
        bd,
        ids,
        scores,
        shared_act,
        ws.view(torch.int32),
        ss,
        bs,
        residual,
        out,
        n,
        ke,
        ks,
        ids.numel(),
        rows,
        min(gbe, ke // 64),
        min(gbs, ks // 64),
        num_warps=warps,
    )
    return out


def rms_norm(x, weight, eps):
    d = x.shape[-1]
    out = torch.empty_like(x)
    _rms_norm[(x.numel() // d,)](
        x, weight, out, eps, d, tr.next_power_of_2(d), enable_fp_fusion=False
    )
    return out


def qk_post(qkv, q_weight, k_weight, rotation, cache, positions, eps, heads, kv_heads, dim):
    """Normalize and rotate q and k from the fused projection, write k, v and the position to the cache."""
    q = torch.empty((1, heads, dim), device=qkv.device, dtype=qkv.dtype)
    cos, sin = rotation if rotation is not None else (qkv, qkv)
    _qk_post[(heads + 2 * kv_heads,)](
        qkv,
        q_weight,
        k_weight,
        cos,
        sin,
        q,
        cache.key,
        cache.value,
        cache.positions,
        positions,
        eps,
        cache.capacity,
        bool(cache.window),
        rotation is not None,
        heads,
        kv_heads,
        dim,
        enable_fp_fusion=False,
    )
    return q


_ROUTER_CONFIG = (1, 256, 8)


def route(h, weight, bias, top_k, config=None):
    """Expert ids (int32) and BF16 normalized sigmoid scores, as the eager router computes them."""
    experts, k = weight.shape
    rows, bk, warps = config or _ROUTER_CONFIG
    logits = torch.empty(experts, device=h.device, dtype=torch.float32)
    _router_logits[(tr.cdiv(experts, rows),)](
        h, weight, logits, experts, k, rows, min(bk, tr.next_power_of_2(k)), num_warps=warps
    )
    ids = torch.empty(top_k, device=h.device, dtype=torch.int32)
    scores = torch.empty(top_k, device=h.device, dtype=h.dtype)
    _router_topk[(1,)](
        logits,
        bias,
        ids,
        scores,
        experts,
        tr.next_power_of_2(experts),
        top_k,
        tr.next_power_of_2(top_k),
        num_warps=4,
    )
    return ids, scores


@tr.jit
def _attention_split(
    Q,
    K,
    V,
    KP,
    POS,
    PART_M,
    PART_L,
    PART_ACC,
    SCALE,
    T: tl.constexpr,
    KH: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    SPLITS: tl.constexpr,
    TILE: tl.constexpr,
):
    """Partial softmax(q k^T / sqrt(D)) v over one slot range for the G query heads of one KV head."""
    kv = tl.program_id(0)
    split = tl.program_id(1)
    pos = tl.load(POS)
    # A full-attention slot j only ever holds position j or -1, so slots past pos are always masked.
    extent = T if WINDOW else tl.minimum(T, pos + 1)
    # Split the live extent (read on device, so it follows graph replays) evenly across SPLITS programs.
    span = tl.cdiv(tl.cdiv(extent, TILE), SPLITS) * TILE
    begin = split * span
    end = tl.minimum(begin + span, extent)
    g = tl.arange(0, GP)
    d = tl.arange(0, D)
    q = tl.load(Q + (kv * G + g[:, None]) * D + d[None, :], g[:, None] < G, 0)
    m = tl.full((GP,), -float("inf"), tl.float32)
    total = tl.zeros((GP,), tl.float32)
    acc = tl.zeros((GP, D), tl.float32)
    for start in range(begin, end, TILE):
        index = start + tl.arange(0, TILE)
        inside = index < end
        kp = tl.load(KP + index, inside, -1)
        valid = inside & (kp >= 0) & (kp <= pos)
        if WINDOW:
            valid &= kp > pos - WINDOW
        offset = index[:, None] * (KH * D) + kv * D + d[None, :]
        k = tl.load(K + offset, inside[:, None], 0)
        # BF16 x BF16 products are exact in FP32, so the matrix unit loses nothing here.
        score = tl.dot(q, tl.trans(k))
        score = tl.div_rn(score, tl.zeros_like(score) + SCALE)
        score = tl.where(valid[None, :], score, -float("inf"))
        new_m = tl.maximum(m, tl.max(score, 1))
        safe = tl.where(new_m == -float("inf"), 0.0, new_m)
        correction = libdevice.exp(m - safe)
        p = libdevice.exp(score - safe[:, None])
        v = tl.load(V + offset, inside[:, None], 0)
        # p as a sum of two BF16 terms keeps ~16 mantissa bits, far below the BF16 output rounding.
        high = p.to(tl.bfloat16)
        low = (p - high.to(tl.float32)).to(tl.bfloat16)
        acc = acc * correction[:, None] + (tl.dot(high, v) + tl.dot(low, v))
        total = total * correction + tl.sum(p, 1)
        m = new_m
    base = (kv * SPLITS + split) * GP + g
    tl.store(PART_M + base, m)
    tl.store(PART_L + base, total)
    tl.store(PART_ACC + base[:, None] * D + d[None, :], acc)


@tr.jit
def _attention_merge(
    PART_M,
    PART_L,
    PART_ACC,
    SINK,
    OUT,
    G: tl.constexpr,
    GP: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Combine split partials with the learned sink as an extra softmax column that contributes no value."""
    kv = tl.program_id(0)
    g = tl.program_id(1)
    head = kv * G + g
    s = tl.arange(0, CHUNK)
    d = tl.arange(0, D)
    sink = tl.load(SINK + head).to(tl.float32)
    peak = sink
    for first in range(0, SPLITS, CHUNK):
        ok = first + s < SPLITS
        peak = tl.maximum(
            peak, tl.max(tl.load(PART_M + (kv * SPLITS + first + s) * GP + g, ok, -float("inf")), 0)
        )
    total = libdevice.exp(sink - peak)
    acc = tl.zeros((D,), tl.float32)
    for first in range(0, SPLITS, CHUNK):
        ok = first + s < SPLITS
        base = (kv * SPLITS + first + s) * GP + g
        weight = libdevice.exp(tl.load(PART_M + base, ok, -float("inf")) - peak)
        total += tl.sum(tl.load(PART_L + base, ok, 0) * weight, 0)
        part = tl.load(PART_ACC + base[:, None] * D + d[None, :], ok[:, None], 0)
        acc += tl.sum(part * weight[:, None], 0)
    out = tl.div_rn(acc, tl.zeros_like(acc) + total)
    tl.store(OUT + head * D + d, out.to(OUT.dtype.element_ty))


_ATTENTION_CONFIG = (24, 16, 4)  # splits per KV head, slots per tile, warps


def sink_attention(q, key, value, positions, kpositions, sinks, window=None, config=None):
    """Single-token GQA attention with a learned sink: q [1, H, D]; key/value [T, KH, D] BF16 cache views."""
    _, heads, dim = q.shape
    slots, kv_heads, _ = key.shape
    group = heads // kv_heads
    splits, tile, warps = config or _ATTENTION_CONFIG
    gp = max(16, tr.next_power_of_2(group))
    part_m = torch.empty((kv_heads, splits, gp), device=q.device, dtype=torch.float32)
    part_l = torch.empty_like(part_m)
    part_acc = torch.empty((kv_heads, splits, gp, dim), device=q.device, dtype=torch.float32)
    _attention_split[(kv_heads, splits)](
        q,
        key,
        value,
        kpositions,
        positions,
        part_m,
        part_l,
        part_acc,
        dim**0.5,
        slots,
        kv_heads,
        group,
        gp,
        dim,
        window or 0,
        splits,
        tile,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    out = torch.empty_like(q)
    _attention_merge[(kv_heads, group)](
        part_m,
        part_l,
        part_acc,
        sinks,
        out,
        group,
        gp,
        dim,
        splits,
        32,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@tr.jit
def _moe_tile_dot_joined(
    X,
    W,
    S,
    B,
    source,
    row_ok,
    expert,
    cols,
    col_ok,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Same result as _moe_tile_dot, but the eight nibble planes are interleaved back into k order in registers
    so X is read contiguously and each step is a single K=BK dot."""
    acc = tl.zeros((BM, BN), tl.float32)
    words = tl.arange(0, BK // 8)
    ks = tl.arange(0, BK)
    weight_row = expert * N + cols
    for first in range(0, K, BK):
        word = first // 8 + words
        packed = tl.load(W + weight_row[:, None] * (K // 8) + word[None, :], col_ok[:, None], 0)
        group = word // 8
        scale = tl.load(
            S + weight_row[:, None] * (K // 64) + group[None, :], col_ok[:, None], 0
        ).to(tl.float32)
        bias = tl.load(B + weight_row[:, None] * (K // 64) + group[None, :], col_ok[:, None], 0).to(
            tl.float32
        )
        p0 = ((packed & 15).to(tl.float32) * scale + bias).to(tl.bfloat16)
        p1 = (((packed >> 4) & 15).to(tl.float32) * scale + bias).to(tl.bfloat16)
        p2 = (((packed >> 8) & 15).to(tl.float32) * scale + bias).to(tl.bfloat16)
        p3 = (((packed >> 12) & 15).to(tl.float32) * scale + bias).to(tl.bfloat16)
        p4 = (((packed >> 16) & 15).to(tl.float32) * scale + bias).to(tl.bfloat16)
        p5 = (((packed >> 20) & 15).to(tl.float32) * scale + bias).to(tl.bfloat16)
        p6 = (((packed >> 24) & 15).to(tl.float32) * scale + bias).to(tl.bfloat16)
        p7 = (((packed >> 28) & 15).to(tl.float32) * scale + bias).to(tl.bfloat16)
        # join appends a minor axis: ((p0,p1),(p2,p3)),((p4,p5),(p6,p7)) has axes [.., b0, b1, b2]; k = 4*b2 + 2*b1 + b0.
        low = tl.join(tl.join(p0, p1), tl.join(p2, p3))
        high = tl.join(tl.join(p4, p5), tl.join(p6, p7))
        planes = tl.join(low, high)
        weight = tl.reshape(tl.permute(planes, (0, 1, 4, 3, 2)), (BN, BK))
        x = tl.load(X + source[:, None] * K + (first + ks)[None, :], row_ok[:, None], 0)
        acc += tl.dot(x, tl.trans(weight))
    return acc


@tr.jit
def _moe_gate_up(
    X,
    WG,
    SG,
    BG,
    WU,
    SU,
    BU,
    ORDER,
    TILE_EXPERT,
    TILE_START,
    COUNTS,
    OFFSETS,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    expert = tl.load(TILE_EXPERT + tile)
    start = tl.load(TILE_START + tile)
    count = tl.load(COUNTS + expert)
    rows = start + tl.arange(0, BM)
    row_ok = rows < count
    assignment = tl.load(ORDER + tl.load(OFFSETS + expert) + rows, row_ok, 0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    col_ok = cols < N
    source = assignment // TOPK
    gate = _bf16(
        _moe_tile_dot_joined(X, WG, SG, BG, source, row_ok, expert, cols, col_ok, N, K, BM, BN, BK)
    )
    up = _bf16(
        _moe_tile_dot_joined(X, WU, SU, BU, source, row_ok, expert, cols, col_ok, N, K, BM, BN, BK)
    )
    act = _bf16(tl.div_rn(gate, 1.0 + libdevice.exp(-gate)))
    tl.store(
        OUT + assignment[:, None] * N + cols[None, :],
        (act * up).to(tl.bfloat16),
        row_ok[:, None] & col_ok[None, :],
    )


@tr.jit
def _moe_down(
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
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    expert = tl.load(TILE_EXPERT + tile)
    start = tl.load(TILE_START + tile)
    count = tl.load(COUNTS + expert)
    rows = start + tl.arange(0, BM)
    row_ok = rows < count
    assignment = tl.load(ORDER + tl.load(OFFSETS + expert) + rows, row_ok, 0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    col_ok = cols < N
    acc = _moe_tile_dot_joined(
        X, W, S, B, assignment, row_ok, expert, cols, col_ok, N, K, BM, BN, BK
    )
    tl.store(
        OUT + assignment[:, None] * N + cols[None, :],
        acc.to(tl.bfloat16),
        row_ok[:, None] & col_ok[None, :],
    )


def moe_layout(ids, experts, block_m):
    """Sort (token, slot) assignments by expert and cut each expert's run into block_m-row tiles."""
    flat = ids.flatten()
    order = torch.argsort(flat, stable=True).to(torch.int32)
    counts = torch.bincount(flat, minlength=experts)
    offsets = (counts.cumsum(0) - counts).to(torch.int32)
    tiles_per_expert = (counts + block_m - 1) // block_m
    total = int(tiles_per_expert.sum().item())
    tile_expert = torch.repeat_interleave(
        torch.arange(experts, device=ids.device), tiles_per_expert, output_size=total
    )
    first_tile = tiles_per_expert.cumsum(0) - tiles_per_expert
    tile_start = (torch.arange(total, device=ids.device) - first_tile[tile_expert]) * block_m
    return (
        order,
        tile_expert.to(torch.int32),
        tile_start.to(torch.int32),
        counts.to(torch.int32),
        offsets,
    )


# BM, BN, BK, warps; swept on real routed activations (bench/w4_sweep.py).
_MOE_CONFIG = {"gate_up": (16, 128, 128, 8), "down": (16, 128, 128, 8)}


def moe_prefill(h, gate, up, down, ids, config=None):
    """Routed-expert MLP for many tokens: rows of the result follow ids.flatten() (token-major, slot-minor)."""
    config = config or _MOE_CONFIG
    wg, sg, bg = gate
    experts, n, k = wg.shape[0], wg.shape[1], wg.shape[2] * 2
    bm, bn, bk, warps = config["gate_up"]
    layout = moe_layout(ids, experts, bm)
    order, tile_expert, tile_start, counts, offsets = layout
    assignments = ids.numel()
    act = torch.empty((assignments, n), device=h.device, dtype=h.dtype)
    _moe_gate_up[(tile_expert.numel(), tr.cdiv(n, bn))](
        h,
        wg.view(torch.int32),
        sg,
        bg,
        up[0].view(torch.int32),
        up[1],
        up[2],
        order,
        tile_expert,
        tile_start,
        counts,
        offsets,
        act,
        n,
        k,
        ids.shape[-1],
        bm,
        bn,
        min(bk, k),
        num_warps=warps,
    )
    wd, sd, bd = down
    n2, k2 = wd.shape[1], wd.shape[2] * 2
    bm2, bn2, bk2, warps2 = config["down"]
    if bm2 != bm:
        order, tile_expert, tile_start, counts, offsets = moe_layout(ids, experts, bm2)
    out = torch.empty((assignments, n2), device=h.device, dtype=h.dtype)
    _moe_down[(tile_expert.numel(), tr.cdiv(n2, bn2))](
        act,
        wd.view(torch.int32),
        sd,
        bd,
        order,
        tile_expert,
        tile_start,
        counts,
        offsets,
        out,
        n2,
        k2,
        bm2,
        bn2,
        min(bk2, k2),
        num_warps=warps2,
    )
    return out
