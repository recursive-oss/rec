import hashlib
import json
import re
import uuid
from array import array
from importlib.resources import files
from itertools import count

from transformers import PreTrainedTokenizerFast


def load_tokenizer(directory):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        directory, local_files_only=True, fix_mistral_regex=False
    )
    tokenizer.chat_template = files("rec_mlx").joinpath("chat_template.jinja").read_text()
    return tokenizer


def normalize_messages(messages):
    normalized = []
    for original in messages:
        message = original
        if message.get("role") != "assistant" and isinstance(message.get("content"), str):
            text = (
                message["content"]
                .replace("\x00", "")
                .replace("<think>", "")
                .replace("</think>", "")
            )
            text = re.sub(
                r"<\|(?:begin_of_text|end_of_text|system|developer|user|assistant|observation)\|>",
                lambda match: match.group().replace("<", "&lt;").replace(">", "&gt;"),
                text,
            )
            message = {**message, "content": text}
        calls = message.get("tool_calls")
        if calls:
            normalized_calls = []
            for call in calls:
                function = call.get("function", call)
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    function = {
                        **function,
                        "arguments": json.loads(arguments) if arguments.strip() else {},
                    }
                normalized_calls.append(
                    {**call, "function": function} if "function" in call else function
                )
            message = {**message, "tool_calls": normalized_calls}
        normalized.append(message)
    return normalized


_CALL = re.compile(
    r"\s*([^<\s]+)\s*((?:<arg_key>.*?</arg_key>\s*<arg_value>.*?</arg_value>\s*)*)", re.S
)
_ARGUMENT = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def _accepts(schema, kind):
    types = schema.get("type")
    if types == kind or (isinstance(types, list) and kind in types):
        return True
    if kind == "string" and any(isinstance(value, str) for value in schema.get("enum", ())):
        return True
    return any(
        _accepts(option, kind) for key in ("anyOf", "oneOf") for option in schema.get(key, ())
    )


def parse_tool_call(text, tools):
    """Invert the template's tool-call rendering: string arguments are raw, others are JSON."""
    call = _CALL.fullmatch(text)
    if call is None:
        raise ValueError(f"Malformed Flint tool call: {text!r}")
    name = call[1]
    properties = {}
    for tool in tools or ():
        if tool["function"]["name"] == name:
            properties = tool["function"].get("parameters", {}).get("properties", {})
    arguments = {}
    for key, value in _ARGUMENT.findall(call[2]):
        schema = properties.get(key, {})
        if value == "null" and _accepts(schema, "null"):
            value = None
        elif not _accepts(schema, "string"):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        arguments[key] = value
    return {"name": name, "arguments": arguments}


def seeded_tool_parser(sampling, prompt_tokens):
    """Parse tool calls with IDs derived from the sample, prompt, and call order.

    The sampler already replays a request exactly for a given seed; deriving the IDs the same
    way makes the whole replayed message identical instead of differing in random UUIDs.
    The sample config is part of the hash, so a different temperature or other sampling
    parameter does not reuse IDs from another sample of the same prompt.
    """
    config = json.dumps(
        {
            "temperature": float(sampling.temperature),
            "top_p": float(sampling.top_p),
            "top_k": sampling.top_k,
            "presence_penalty": float(sampling.presence_penalty),
            "repetition_penalty": float(getattr(sampling, "repetition_penalty", 1.0)),
            "max_tokens": sampling.max_tokens,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    request = hashlib.sha256(sampling.seed.to_bytes(8, "little"))
    request.update(config)
    request.update(array("I", prompt_tokens).tobytes())
    indices = count()

    def parse(text, tools):
        call = parse_tool_call(text, tools)
        digest = request.copy()
        digest.update(next(indices).to_bytes(4, "little"))
        call["id"] = str(uuid.UUID(bytes=digest.digest()[:16], version=4))
        return call

    return parse
