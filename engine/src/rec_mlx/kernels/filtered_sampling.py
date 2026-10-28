import mlx.core as mx
from mlx_lm.sample_utils import categorical_sampling
from mlx_lm.sample_utils import make_sampler as native_sampler


@mx.compile
def filter_top_p_top_k(logprobs, top_p, top_k):
    size = logprobs.shape[-1]
    if not 0 < top_k < size:
        raise ValueError(f"top_k must be between 1 and {size - 1}")
    indices = mx.argsort(logprobs)
    values = logprobs[indices]
    keep_p = mx.cumsum(mx.exp(values)) > 1 - top_p
    threshold = values[-top_k]
    positions = mx.arange(size)
    tie_start = mx.min(mx.where((values == threshold) & keep_p, positions, size))
    greater = mx.sum(values > threshold)
    keep = keep_p & (
        (values > threshold) | ((values == threshold) & (positions < tie_start + top_k - greater))
    )
    mask = mx.zeros(size, mx.bool_).at[indices].add(keep)
    return mx.where(mask, logprobs, -float("inf"))


def make_sampler(temp=0.0, top_p=0.0, top_k=0):
    if temp == 0 or not (0 < top_p < 1 and top_k > 0):
        return native_sampler(temp=temp, top_p=top_p, top_k=top_k)

    def sample(logprobs):
        return categorical_sampling(filter_top_p_top_k(logprobs, top_p, top_k), temp)

    return sample
