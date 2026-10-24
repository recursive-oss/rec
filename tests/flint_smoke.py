#!/usr/bin/env python3
"""Check tool calls against a running Flint server. Never prints the API key."""

import argparse
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="rec", help="rec executable (default: %(default)s)")
    parser.add_argument("--report", type=Path, help="JSON file for the results")
    parser.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="HTTP timeout in seconds (increase for shared GPUs)",
    )
    parser.add_argument(
        "--generation-tokens",
        type=int,
        default=2048,
        help="Long generation budget including reasoning",
    )
    args = parser.parse_args()
    status = json.loads(subprocess.check_output([args.binary, "status"], text=True))
    assert status["ready"] and status["model"] == "flint", status
    key = subprocess.check_output([args.binary, "auth", "token"], text=True).strip()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, payload=None, authenticated=True):
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = "Bearer " + key
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(status["endpoint"] + path, data=body, headers=headers)
        return opener.open(req, timeout=args.timeout)

    def chat(messages, **extra):
        payload = dict(
            model="flint", messages=messages, temperature=0, max_tokens=512, stream=False
        )
        payload.update(extra)
        with request("/chat/completions", payload) as response:
            return json.load(response)

    try:
        request("/models", authenticated=False)
        raise AssertionError("Unauthenticated model listing was accepted")
    except urllib.error.HTTPError as error:
        assert error.code == 401, error.code

    arithmetic = chat([{"role": "user", "content": "What is 2 + 2? Answer with only the number."}])
    assert arithmetic["choices"][0]["message"]["content"].strip() == "4", arithmetic
    assert arithmetic["choices"][0]["finish_reason"] == "stop", arithmetic

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "Use get_weather to get the temperature in Paris."}]
    tool_choice = "auto"
    called = chat(messages, tools=tools, tool_choice=tool_choice)
    assistant = called["choices"][0]["message"]
    calls = assistant.get("tool_calls", [])
    assert len(calls) == 1, called
    assert called["choices"][0]["finish_reason"] == "tool_calls", called
    call = calls[0]
    assert call["function"]["name"] == "get_weather", called
    assert json.loads(call["function"]["arguments"])["city"].lower() == "paris", called
    replayed = chat(messages, tools=tools, tool_choice=tool_choice)
    assert replayed["choices"][0]["message"] == assistant, (called, replayed)
    messages += [
        assistant,
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": '{"city":"Paris","temperature_c":18}',
        },
    ]
    answered = chat(messages, tools=tools, tool_choice="none")
    assert "18" in (answered["choices"][0]["message"]["content"] or ""), answered
    assert answered["choices"][0]["finish_reason"] == "stop", answered

    with request(
        "/chat/completions",
        dict(
            model="flint",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=32,
            temperature=0,
            stream=True,
        ),
    ) as response:
        stream = response.read().decode()
        assert "data:" in stream and "[DONE]" in stream, stream

    generation = chat(
        [{"role": "user", "content": "Explain how hash tables work in about 200 words."}],
        max_tokens=args.generation_tokens,
    )
    assert generation["choices"][0]["message"]["content"], generation
    assert generation["choices"][0]["finish_reason"] == "stop", generation
    report = {
        "status": status,
        "auth_rejected": True,
        "arithmetic": "4",
        "tool_choice": tool_choice,
        "tool_round_trip": answered["choices"][0]["message"]["content"],
        "tool_call_replayed": True,
        "streaming": True,
        "generation": generation["choices"][0]["message"]["content"],
        "usage": generation.get("usage"),
        "timings": generation.get("timings"),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Flint: authentication, arithmetic, tool-call round trip and replay, streaming, and generation"
        " passed"
    )
    if report["timings"]:
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report["timings"].items()
                    if k not in ("token_ids", "token_times")
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
