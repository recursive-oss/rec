"""ROCm runtime CLI, accepting rec's bundled-engine launch arguments."""

import argparse
import json
import logging
import os
import signal
import threading
from pathlib import Path

from rec_mlx.engine import Sampling

from .engine import Engine
from .server import make_server


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["serve", "chat"])
    p.add_argument("model", choices=["flint"])
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--context-limit", type=int, default=8192)
    p.add_argument("--port", type=int, default=11500)
    p.add_argument("--api-key-file", type=Path)
    p.add_argument(
        "--metrics",
        type=Path,
        default=Path(os.environ["REC_METRICS_PATH"])
        if os.environ.get("REC_METRICS_PATH")
        else None,
    )
    p.add_argument("--grammar", choices=["flint", "none"], default="flint")
    p.add_argument("--enforce-tool-schemas", action="store_true")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--prefill-chunk", type=int, default=256)
    a = p.parse_args()
    if a.prefill_chunk < 1:
        p.error("--prefill-chunk must be positive")
    engine = Engine(
        a.model_dir,
        a.manifest,
        context_limit=a.context_limit,
        enforce_tool_schemas=a.enforce_tool_schemas,
        graph=not a.no_graph,
        prefill_chunk=a.prefill_chunk,
    )
    print(
        json.dumps(
            {
                "event": "ready",
                "checkpoint": engine.identity["model_id"],
                "context_limit": a.context_limit,
                **engine.startup,
            }
        ),
        flush=True,
    )
    if a.command == "chat":
        from rec_mlx.cli import chat

        chat(engine, Sampling(), a.metrics, grammar_mode=a.grammar)
        return
    key = a.api_key_file.read_text().strip() if a.api_key_file else None
    if key is not None and (not key or any(c.isspace() for c in key)):
        p.error("API key must be one nonempty token")
    server = make_server(
        engine, ("127.0.0.1", a.port), a.metrics, api_key=key, grammar_mode=a.grammar
    )

    def stop(signum, frame):
        server.generator.cancel()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"Listening on http://127.0.0.1:{a.port}/v1", flush=True)
    try:
        server.serve_forever()
    finally:
        server.generator.cancel()
        server.server_close()


if __name__ == "__main__":
    main()
