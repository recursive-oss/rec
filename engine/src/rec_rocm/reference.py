"""Independent, deliberately simple CPU numerical oracles for the packed format."""

import numpy as np


def unpack_affine(packed, scales, offsets, bits=4, group_size=64):
    """MLX exports little-endian codes, lowest bits first, affine w = s*q+b."""
    packed = np.asarray(packed, dtype=np.uint8)
    shifts = np.arange(8 // bits, dtype=np.uint8) * bits
    codes = ((packed[..., None] >> shifts) & ((1 << bits) - 1)).reshape(*packed.shape[:-1], -1)
    groups = np.arange(codes.shape[-1]) // group_size
    return (
        codes.astype(np.float64) * np.asarray(scales, dtype=np.float64)[..., groups]
        + np.asarray(offsets, dtype=np.float64)[..., groups]
    )


def rms_norm(x, weight, eps):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def sink_attention(q, k, v, sink, q_positions, k_positions, window=None):
    q, k, v = [np.asarray(x, dtype=np.float64) for x in (q, k, v)]
    repeats = q.shape[1] // k.shape[1]
    k, v = np.repeat(k, repeats, axis=1), np.repeat(v, repeats, axis=1)
    score = np.einsum("mhd,nhd->hmn", q, k) / np.sqrt(q.shape[-1])
    allowed = np.asarray(k_positions)[None, :] <= np.asarray(q_positions)[:, None]
    if window:
        allowed &= np.asarray(k_positions)[None, :] > np.asarray(q_positions)[:, None] - window
    score = np.where(allowed[None], score, -np.inf)
    maximum = np.maximum(np.max(score, axis=-1), np.asarray(sink)[:, None])
    exp = np.exp(score - maximum[..., None])
    probs = exp / (exp.sum(-1) + np.exp(np.asarray(sink)[:, None] - maximum))[..., None]
    return np.einsum("hmn,nhd->mhd", probs, v)
