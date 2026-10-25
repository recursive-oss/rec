"""Experimental scalar PV matching the qualified FP32 BLAS accumulation order.

Keep QK, masking, sink, and softmax unchanged. Only batch-one decode uses this
path; prefill is faster with BLAS. Qualification is specific to the pinned
hipBLASLt family, rather than an equivalence claim for arbitrary vendor kernels.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _decode_pv(
    P,
    V,
    OUT,
    POSITION,
    T: tl.constexpr,
    KV: tl.constexpr,
    REPEATS: tl.constexpr,
    P_ROW: tl.constexpr,
    TRUNCATE: tl.constexpr,
):
    kv = tl.program_id(2)
    d = tl.program_id(0) * 16 + tl.arange(0, 16)
    head = tl.program_id(1) * 4 + tl.arange(0, 4)
    kk = tl.arange(0, 8)
    count = tl.minimum(tl.load(POSITION) + 1, T) if TRUNCATE else T
    acc = tl.full((16, 4), 0, tl.float32)
    for start in range(tl.cdiv(count, 8)):
        k = start * 8 + kk
        value = tl.load(V + (k[None, :] * KV + kv) * 128 + d[:, None], k[None, :] < count, 0).to(
            tl.float32
        )
        probability = tl.load(
            P + (kv * REPEATS + head[:, None]) * P_ROW + k[None, :],
            (head[:, None] < REPEATS) & (k[None, :] < count),
            0,
        )
        for offset in tl.static_range(8):
            vv = tl.gather(value, tl.full((16, 1), offset, tl.int32), 1).reshape(16)
            pp = tl.gather(probability, tl.full((4, 1), offset, tl.int32), 1).reshape(4)
            acc = tl.fma(vv[:, None], pp[None, :], acc)
    tl.store(OUT + (kv * REPEATS + head[None, :]) * 128 + d[:, None], acc, head[None, :] < REPEATS)


def decode_pv(probability, value, positions, *, window=None):
    """P is [QH, 1, T] with the sink removed; preserve its padded row stride.

    Full-attention KV has a contiguous valid prefix. A sliding-window ring has
    a different physical order, so all ring slots must be visited even at wrap.
    The device position is read at every Graph replay, without a host sync.
    """
    heads, count, length = probability.shape
    assert count == 1 and probability.dtype == torch.float32
    assert value.shape == (length, value.shape[1], 128) and value.is_contiguous()
    assert value.dtype == torch.bfloat16 and probability.stride(-1) == 1
    assert probability.stride(0) == probability.stride(1)
    kvheads = value.shape[1]
    assert heads % kvheads == 0
    repeats = heads // kvheads
    out = torch.empty((1, heads, 128), device=value.device, dtype=value.dtype)
    _decode_pv[(8, tr.cdiv(repeats, 4), kvheads)](
        probability,
        value,
        out,
        positions,
        length,
        kvheads,
        repeats,
        probability.stride(1),
        window is None,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out
