"""Bound prefill score memory by splitting queries; keys keep their full history."""

import math

import torch


def query_tiled_attention(q, key, value, sinks, qpositions, kpositions, query_rows=64):
    count, heads, dim = q.shape
    kvheads = key.shape[1]
    repeats = heads // kvheads
    # These conversions are shared across query tiles; all arithmetic stays FP32.
    keys, values = key.float(), value.float()
    parts = []
    for begin in range(0, count, query_rows):
        query = q[begin : begin + query_rows]
        positions = qpositions[begin : begin + query_rows]
        rows = query.shape[0]
        scores = torch.einsum(
            "mkhd,nkd->khmn", query.float().view(rows, kvheads, repeats, dim), keys
        ).reshape(heads, rows, key.shape[0]) / math.sqrt(dim)
        allowed = (kpositions[None] <= positions[:, None]) & (kpositions[None] >= 0)
        scores = scores.masked_fill(~allowed[None], -float("inf"))
        scores = torch.cat((scores, sinks[:, None, None].expand(-1, rows, 1)), -1)
        probabilities = torch.softmax(scores, -1)[..., : key.shape[0]]
        del scores
        part = (
            torch.einsum(
                "khmn,nkd->mkhd", probabilities.view(kvheads, repeats, rows, key.shape[0]), values
            )
            .reshape(rows, heads, dim)
            .to(q.dtype)
        )
        del probabilities
        parts.append(part)
    return torch.cat(parts, 0)
