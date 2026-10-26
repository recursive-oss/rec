"""Run Flint as a terminal chat or a local OpenAI-compatible server."""

import argparse
import json
import logging
import os
import signal
import threading
from dataclasses import asdict
from pathlib import Path

from .engine import ContextLimitError, Engine, Sampling


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("chat", "serve"), help="chat in this terminal or serve the HTTP API"
    )
    parser.add_argument("model", choices=("flint",), help="model to run")
    parser.add_argument("--model-dir", type=Path, required=True, help="checkpoint directory")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="checkpoint manifest",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=8192,
        help="total input and output tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for serve (default: %(default)s)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface for the serve HTTP server to bind (default: %(default)s)",
    )
    parser.add_argument(
        "--grammar",
        choices=("none", "flint"),
        default="flint",
        help="constrain assistant turns to each request's tool schemas (default: %(default)s)",
    )
    parser.add_argument(
        "--enforce-tool-schemas",
        action="store_true",
        help="with Flint grammar, also constrain strict:false tool arguments",
    )
    parser.add_argument(
        "--api-key-file", type=Path, help="file with the bearer token that requests must present"
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        help="append per-request metrics to this JSONL file (default: disabled)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="default sampling temperature (default: %(default)s)",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="default presence penalty (default: %(default)s)",
    )
    parser.add_argument(
        "--repeat-penalty",
        "--repeat_penalty",
        "--repetition-penalty",
        type=float,
        default=1.0,
        dest="repetition_penalty",
        help="default repetition penalty (default: %(default)s)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="default nucleus sampling threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="default top-k sampling limit (default: %(default)s)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="chat output limit; HTTP uses the request limit or the remaining context (default: %(default)s)",
    )
    parser.add_argument(
        "--fast-sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use MLX fast CPU/GPU synchronization (default: enabled)",
    )
    args = parser.parse_args()
    os.environ["MLX_METAL_FAST_SYNCH"] = "1" if args.fast_sync else "0"
    try:
        settings = Sampling(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens,
        )
    except ValueError as error:
        parser.error(str(error))
    api_key = None
    if args.api_key_file is not None:
        api_key = args.api_key_file.read_text().strip()
        if not api_key or any(c.isspace() for c in api_key):
            parser.error("API-key file must contain one nonempty token")
    engine = Engine(
        args.model_dir,
        args.manifest,
        context_limit=args.context_limit,
        enforce_tool_schemas=args.enforce_tool_schemas,
    )
    print(
        json.dumps(
            {
                "event": "ready",
                "checkpoint": engine.identity["model_id"],
                "grammar": args.grammar,
                "enforce_tool_schemas": args.enforce_tool_schemas,
                "fast_sync": args.fast_sync,
                "context_limit": args.context_limit,
                "sampling": asdict(settings),
                **engine.startup,
            }
        ),
        flush=True,
    )
    if args.command == "serve":
        from .server import make_server

        server = make_server(
            engine,
            (args.host, args.port),
            args.metrics,
            defaults=settings,
            api_key=api_key,
            grammar_mode=args.grammar,
        )
        print(f"Listening on http://{args.host}:{server.server_port}/v1", flush=True)

        def stop(signum, frame):
            server.generator.cancel()
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopping the server.")
        finally:
            server.generator.cancel()
            server.generator.join()
            server.server_close()
        return
    chat(engine, settings, args.metrics, grammar_mode=args.grammar)


def chat(engine, settings, metrics_path, *, grammar_mode="flint"):
    constraint = engine.constraints.tools() if grammar_mode == "flint" else None
    messages = []
    while True:
        try:
            prompt = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt == "/quit":
            return
        if prompt == "/reset":
            messages.clear()
            continue
        messages.append({"role": "user", "content": prompt})
        answer = ""
        events = engine.generate(messages, settings, grammar=constraint)
        try:
            for event in events:
                if "text" in event:
                    print(event["text"], end="", flush=True)
                    answer += event["text"]
                elif metrics_path is not None:
                    with metrics_path.open("a") as output:
                        output.write(
                            json.dumps(
                                {
                                    "checkpoint": engine.identity["model_id"],
                                    "kv_format": engine.startup["kv_format"],
                                    "sampling": asdict(settings),
                                    **event,
                                }
                            )
                            + "\n"
                        )
        except ContextLimitError as error:
            messages.pop()
            print(f"{error}. Use /reset or a shorter message.")
            continue
        except KeyboardInterrupt:
            print("\nGeneration interrupted.")
            if answer:
                messages.append({"role": "assistant", "content": answer})
            else:
                messages.pop()
            continue
        finally:
            events.close()
        messages.append({"role": "assistant", "content": answer})
        print()


if __name__ == "__main__":
    main()
