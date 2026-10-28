# Thinking budget.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
THINKING_BUDGET_FIELD = "thinking_budget"
# A forced token has probability 1 in the constrained process.
FORCED_TOKEN_LOGPROB = 0.0

# Engine metric keys read for usage when thinking_budget is set.
METRIC_REASONING_TOKENS = "reasoning_tokens"
METRIC_FORCED_CLOSE = "thinking_forced_close"

# Usage fields added to chat completions that set thinking_budget.
USAGE_REASONING_TOKENS = "reasoning_tokens"
USAGE_FORCED_CLOSE = "thinking_forced_close"
