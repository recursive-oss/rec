#!/usr/bin/env python3
"""Run the real rec-mlx command line with a scripted decoder in place of Flint.

Run with the packaged runtime's Python and rec-mlx's arguments, as
tests/protocol_test.py does. The command line, chat template, tool parser, grammar
compiler and HTTP server are the shipped code; only token selection is scripted.
Each emitted token must satisfy the request's grammar.

The reply depends on the rendered prompt:
  after a tool result         <think>…</think>The tool returned: RESULT
  "Call NAME with K=V, K=V."  <think>…</think><tool_call>NAME<arg_key>K</arg_key>…<|observation|>
  "Answer with X."            <think>…</think>X
  "Repeat forever."           x x x … until the output limit
  "Repeat slowly."            the same, 50 ms per token
"""

import re
import time
from importlib.resources import files

import xgrammar as xgr
from rec_mlx import cli
from rec_mlx.engine import ContextLimitError, Engine
from rec_mlx.grammar import ConstraintCompiler
from rec_mlx.grammar_builder import _PROTOCOL_MARKERS
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def flint_tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<|end_of_text|>")
    wrapped.add_tokens([x for x in _PROTOCOL_MARKERS if x != "<|end_of_text|>"])
    wrapped.chat_template = files("rec_mlx").joinpath("chat_template.jinja").read_text()
    return wrapped


class ScriptedEngine:
    identity = {"model_id": "scripted-flint"}
    startup = {"engine_experiment": "scripted", "kv_format": "none"}
    prompt = Engine.prompt

    def __init__(
        self,
        model_dir,
        manifest,
        *,
        context_limit,
        enforce_tool_schemas,
        steering_file=None,
        steering_strength=1.0,
        probe_file=None,
        probe_layer=None,
    ):
        assert steering_file is None and steering_strength == 1.0
        assert probe_file is None and probe_layer is None
        self.context_limit = context_limit
        self.tokenizer = flint_tokenizer()
        self.constraints = ConstraintCompiler(self.tokenizer)
        self.eos = {
            self.tokenizer.convert_tokens_to_ids(x)
            for x in ("<|end_of_text|>", "<|user|>", "<|observation|>")
        }

    def prepare_prefix(self, *, should_stop):
        return None

    def reply(self, ids):
        prompt = self.tokenizer.decode(ids, skip_special_tokens=False)
        if not prompt.endswith("<|assistant|>"):
            raise ValueError("The prompt must end with the assistant generation marker")
        turn = re.split(r"(<\|user\|>|<\|observation\|>)", prompt)
        role, text = turn[-2], turn[-1].removesuffix("<|assistant|>")
        if role == "<|observation|>":
            result = re.fullmatch(r"<tool_response>(.*)</tool_response>", text, re.S)[1]
            return "<think>Report the result.</think>The tool returned: " + result, 0
        if call := re.fullmatch(r"Call (\S+) with (.+)\.", text, re.S):
            if f'"name": "{call[1]}"' not in prompt:
                raise ValueError(f"Tool {call[1]} is missing from the prompt")
            arguments = "".join(
                f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
                for key, value in (pair.split("=", 1) for pair in call[2].split(", "))
            )
            return f"<think>Call {call[1]}.</think><tool_call>{call[1]}{arguments}</tool_call>", 0
        if answer := re.fullmatch(r"Answer with (.+)\.", text, re.S):
            return "<think>Answer briefly.</think>" + answer[1], 0
        if text == "Repeat forever.":
            return None, 0
        if text == "Repeat slowly.":
            return None, 0.05
        raise ValueError(f"No scripted reply for {text!r}")

    def generate_tokens(
        self, ids, sampling, *, should_stop, on_checkpoint=None, grammar=None, return_logprobs=False
    ):
        if len(ids) + sampling.max_tokens > self.context_limit:
            raise ContextLimitError("Prompt plus output allowance exceeds the context limit")
        text, delay = self.reply(ids)
        if text is None:
            script = self.tokenizer.encode("x ", add_special_tokens=False) * sampling.max_tokens
        else:
            stop = "<|observation|>" if text.endswith("</tool_call>") else "<|end_of_text|>"
            script = self.tokenizer.encode(text + stop, add_special_tokens=False)
        stops = self.eos if grammar is None else set(grammar.stops)
        matcher = None if grammar is None else xgr.GrammarMatcher(grammar.compiled)
        generated, emitted, reason = [], "", "length"
        for index, token in enumerate(script[: sampling.max_tokens]):
            if should_stop():
                reason = "stop"
                break
            if matcher is not None and not matcher.accept_token(token):
                raise RuntimeError(f"Grammar rejected scripted token {token}")
            time.sleep(delay)
            generated.append(token)
            timings = {
                "prompt_tokens": len(ids),
                "accepted_output_tokens": len(generated),
                "decode_tokens": len(generated) - 1,
                "decode_ms": 0,
                "logical_context_tokens": len(ids) + len(generated),
            }
            if on_checkpoint is not None:
                on_checkpoint(timings, True)
            details = {"logprob": 0.0} if return_logprobs else {}
            if token in stops:
                reason = "stop"
                yield {"token": token, "text": "", "finish_reason": reason, **details}
                break
            decoded = self.tokenizer.decode(generated, skip_special_tokens=False)
            delta, emitted = decoded[len(emitted) :], decoded
            final = index + 1 == sampling.max_tokens
            yield {
                "token": token,
                "text": delta,
                "finish_reason": "length" if final else None,
                **details,
            }
        yield {
            "finish_reason": reason,
            "metrics": {
                "prompt_tokens": len(ids),
                "cached_prompt_tokens": 0,
                "generated_tokens": len(generated),
                "token_ids": generated,
            },
        }


if __name__ == "__main__":
    cli.Engine = ScriptedEngine
    cli.main()
