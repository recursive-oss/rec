from dataclasses import dataclass

GRAMMAR = "flint"
ENFORCE_TOOL_SCHEMAS = False


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 1024
    seed: int = 0
    presence_penalty: float = 0.0

    def __post_init__(self):
        for name in ("temperature", "top_p", "presence_penalty"):
            if type(getattr(self, name)) not in (int, float):
                raise ValueError(f"{name} must be a number")
        for name in ("top_k", "max_tokens", "seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("temperature must be in [0, 2] and top_p in (0, 1]")
        if not -2 <= self.presence_penalty <= 2:
            raise ValueError("presence_penalty must be in [-2, 2]")
        if self.top_k < 0 or self.max_tokens < 1:
            raise ValueError("top_k must be nonnegative and max_tokens positive")
        if not 0 <= self.seed <= 2**64 - self.max_tokens:
            raise ValueError("seed plus max_tokens must fit in unsigned 64-bit random keys")
