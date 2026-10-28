from .constants import METRIC_FORCED_CLOSE, METRIC_REASONING_TOKENS, THINK_CLOSE, THINK_OPEN
from .token_ids import dedicated_token_id


def resolve_think_ids(tokenizer):
    return dedicated_token_id(tokenizer, THINK_OPEN), dedicated_token_id(tokenizer, THINK_CLOSE)


class ThinkBudget:
    """Counts tokens between <think> and </think> and says when </think> must be forced.

    The engine feeds every emitted token (sampled or forced) to accept(). A budget of None
    counts but never forces; a budget of 0 only masks <think> at the first position.
    """

    def __init__(self, budget, think_id, close_id):
        if budget is not None and (type(budget) is not int or budget < 0):
            raise ValueError("thinking budget must be None or a nonnegative integer")
        self.budget = budget
        self.think_id = think_id
        self.close_id = close_id
        self.reasoning_tokens = 0
        self.forced_close = False
        self.inside_think = False
        self._opened = False

    def remaining(self):
        """Tokens that may still be sampled before the close is forced; None = unlimited."""
        if self.budget is None or not self.inside_think:
            return None
        left = self.budget - self.reasoning_tokens
        if left < 0:
            raise RuntimeError("The thinking budget was exceeded without a forced close")
        return left

    def forced_token(self):
        return self.close_id if self.remaining() == 0 else None

    def accept(self, token):
        if self.inside_think:
            if token == self.close_id:
                self.forced_close = self.forced_token() is not None
                self.inside_think = False
            else:
                self.reasoning_tokens += 1
        elif token == self.think_id and not self._opened:
            self._opened = self.inside_think = True

    def report(self):
        return {
            METRIC_REASONING_TOKENS: self.reasoning_tokens,
            METRIC_FORCED_CLOSE: self.forced_close,
        }

    def mask_first_logits(self, logits):
        if self.budget != 0:
            return logits
        import mlx.core as mx

        return mx.where(mx.arange(logits.shape[-1]) == self.think_id, -float("inf"), logits)
