import mlx.core as mx


def restore_random_key(key):
    """Restore a saved key through MLX's thread-local public seed API.

    MLX HEAD exposes random.state as a read-only sentinel. A key is exactly
    the two uint32 words of a uint64 seed, so reseeding restores all its bits
    without mutating an array that another saved state may still reference.
    """
    high, low = key.tolist()
    mx.random.seed((high << 32) | low)
