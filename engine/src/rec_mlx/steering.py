"""Directional steering of Flint's residual stream.

A steering file holds one direction r and per-slot settings. There are 1 + 2 * layers slots: slot 0
follows the embedding, slot 1 + 2 * i follows layer i's attention add and slot 2 + 2 * i its MLP add.
At an active slot the residual's component d = h . r moves toward a target:

    t = clamp(d, low, high) + shift
    h <- h + strength * (t - d) * r

With low = high = shift = 0 and strength 1 this removes the component, which equals the weight
edit used for abliteration. Finite bounds cap unusual activity along r, and a nonzero shift adds
along r. On Metal, prefill runs the decode kernel's own steering function on each BF16 2048-wide
row, so the native layers and the resident decoder stay bit-identical. Other devices and widths
use `reference`, the same float32 arithmetic in MLX operations.
The file is .safetensors or .npz: a direction tensor and optional per-slot strength, low,
high and shift arrays.
"""

import hashlib
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import mlx.core as mx

_FIELDS = ("strength", "low", "high", "shift")


@dataclass(frozen=True)
class Steering:
    direction: mx.array
    strength: tuple[float, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    shift: tuple[float, ...]
    sha256: str = ""

    @property
    def slots(self):
        return len(self.strength)

    def active(self, slot):
        return self.strength[slot] != 0.0

    def summary(self):
        return {
            "sha256": self.sha256,
            "active_slots": sum(self.active(slot) for slot in range(self.slots)),
            "slots": self.slots,
        }

    @cached_property
    def _rows(self):
        from .kernels.steering import rows

        return rows(self)

    def __call__(self, h, slot):
        if not self.active(slot):
            return h
        if h.dtype == mx.bfloat16 and h.shape[-1] == 2048 and mx.metal.is_available():
            return self._rows(h, slot)
        return self.reference(h, slot)

    def reference(self, h, slot):
        r = self.direction
        f = h.astype(mx.float32)
        d = f @ r
        target = mx.clip(d, self.low[slot], self.high[slot]) + self.shift[slot]
        delta = self.strength[slot] * (target - d)
        return (f + delta[..., None] * r).astype(h.dtype)


def load(path: Path, *, hidden_size: int, layers: int, strength: float = 1.0) -> Steering:
    data = path.read_bytes()
    tensors = mx.load(str(path))
    if not isinstance(tensors, dict):
        raise ValueError("A steering file must hold named tensors")
    unknown = set(tensors) - {"direction", *_FIELDS}
    if unknown:
        raise ValueError(f"Unknown steering tensors: {', '.join(sorted(unknown))}")
    if "direction" not in tensors:
        raise ValueError("A steering file needs a direction tensor")
    direction = tensors["direction"].astype(mx.float32).reshape(-1)
    if direction.size != hidden_size:
        raise ValueError(f"The steering direction must have {hidden_size} values")
    norm = float(mx.linalg.norm(direction).item())
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("The steering direction must be finite and nonzero")
    slots = 1 + 2 * layers
    # A file with only a direction removes it at every slot: runtime abliteration.
    defaults = {"strength": 1.0, "low": 0.0, "high": 0.0, "shift": 0.0}
    values = {}
    for name in _FIELDS:
        if name in tensors:
            array = tensors[name].astype(mx.float32).reshape(-1)
            if array.size != slots:
                raise ValueError(f"Steering {name} must have {slots} values, one per slot")
            values[name] = [float(v) for v in array.tolist()]
        else:
            values[name] = [defaults[name]] * slots
    if not math.isfinite(strength):
        raise ValueError("Steering strength must be finite")
    values["strength"] = [s * strength for s in values["strength"]]
    for slot in range(slots):
        low, high = values["low"][slot], values["high"][slot]
        if math.isnan(low) or math.isnan(high) or low > high:
            raise ValueError(f"Steering slot {slot} needs low <= high")
        for name in ("strength", "shift"):
            if not math.isfinite(values[name][slot]):
                raise ValueError(f"Steering {name} must be finite")
    direction = direction / norm
    mx.eval(direction)
    return Steering(
        direction=direction,
        sha256=hashlib.sha256(data).hexdigest(),
        **{name: tuple(values[name]) for name in _FIELDS},
    )


def attach(model, steering):
    transformer = model.model
    transformer.steering = steering
    for index, layer in enumerate(transformer.layers):
        layer.steering = steering
        layer.attention_slot, layer.mlp_slot = 1 + 2 * index, 2 + 2 * index
