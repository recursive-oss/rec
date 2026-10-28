from typing import Any

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from xgrammar.builtin_structural_tag import get_model_structural_tag
from xgrammar.openai_tool_call_schema import FunctionToolParam
from xgrammar.structural_tag import (
    AnyTextFormat,
    Format,
    OptionalFormat,
    OrFormat,
    SequenceFormat,
    StructuralTag,
    TagFormat,
    TokenFormat,
)

from .token_ids import dedicated_token_id

_PROTOCOL_MARKERS = (
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|system|>",
    "<|developer|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
)
_STOP_MARKERS = ("<|end_of_text|>", "<|user|>", "<|observation|>")


def build_flint_glm_grammar(
    tokenizer: PreTrainedTokenizerBase,
    tools: list[dict[str, Any]] | None = None,
    *,
    enforce_tool_schemas: bool = False,
) -> StructuralTag:
    ids = {
        marker: dedicated_token_id(tokenizer, marker)
        for marker in (*_STOP_MARKERS, "<think>", "</think>")
    }
    if tokenizer.eos_token_id != ids["<|end_of_text|>"]:
        raise ValueError("Flint grammar requires end-of-text as the tokenizer EOS")
    endings: list[Format] = [OptionalFormat(content=TokenFormat(token=ids["<|user|>"]))]
    if tools:
        function_tools = []
        for definition in tools:
            tool = FunctionToolParam.model_validate(definition)
            parameters = tool.function.parameters
            if parameters is None or parameters.get("type") != "object":
                raise ValueError("Flint grammar requires object parameter schemas")
            if enforce_tool_schemas and tool.function.strict is False:
                # XGrammar discards the schema for strict:false. Clear the flag on this
                # compiler-side copy only; requests and prompts keep the client's value.
                tool = tool.model_copy(
                    update={"function": tool.function.model_copy(update={"strict": None})}
                )
            function_tools.append(tool)
        glm = get_model_structural_tag(
            "glm_4_7", tools=[*function_tools], tool_choice="required", reasoning=False
        )
        endings.append(
            SequenceFormat(elements=[glm.format, TokenFormat(token=ids["<|observation|>"])])
        )
    reasoning = TagFormat(
        begin=TokenFormat(token=ids["<think>"]),
        content=AnyTextFormat(excludes=["</think>", *_STOP_MARKERS]),
        end=TokenFormat(token=ids["</think>"]),
    )
    return StructuralTag(
        format=SequenceFormat(
            elements=[
                OptionalFormat(content=reasoning),
                AnyTextFormat(excludes=list(_PROTOCOL_MARKERS)),
                OrFormat(elements=endings),
            ]
        )
    )
