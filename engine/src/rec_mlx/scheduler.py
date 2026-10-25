import json
import sys
import threading
from dataclasses import asdict
from queue import Empty, Queue
from types import SimpleNamespace

from mlx_lm.server import (
    GenerationContext,
    Response,
    ResponseGenerator,
    _process_control_tokens,
)
from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer, TokenizerWrapper

from .chat import parse_tool_call, seeded_tool_parser
from .engine import Sampling


class ModelProvider:
    is_batchable = False

    def __init__(self, engine, tokenizer, cli_args):
        self.engine = engine
        self.tokenizer = tokenizer
        self.cli_args = cli_args

    def load_default(self):
        return self.engine, self.tokenizer

    def load(self, model, adapter, draft):
        return self.engine, self.tokenizer


class Generator(ResponseGenerator):
    def __init__(self, engine, metrics_path, defaults):
        self.engine = engine
        self.metrics_path = metrics_path
        self.defaults = defaults
        self.tokenizer = TokenizerWrapper(
            engine.tokenizer,
            NaiveStreamingDetokenizer,
            eos_token_ids=engine.eos,
            tool_parser=parse_tool_call,
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
        )
        cli_args = SimpleNamespace(
            allowed_origins=[],
            num_draft_tokens=0,
            max_tokens=defaults.max_tokens,
            temp=defaults.temperature,
            top_p=defaults.top_p,
            top_k=defaults.top_k,
            min_p=0,
        )
        self.stopping = threading.Event()
        self.busy = threading.Event()
        self._request_lock = threading.Lock()
        self.active_context = None
        super().__init__(ModelProvider(engine, self.tokenizer, cli_args), prompt_cache=None)

    def cancel(self):
        with self._request_lock:
            self.stopping.set()
            self._stop = True
            if self.active_context is not None:
                self.active_context.stop()
            while True:
                try:
                    response_queue, _, _ = self.requests.get_nowait()
                except Empty:
                    break
                response_queue.put(RuntimeError("The server is stopping"))

    def _generate(self):
        try:
            super()._generate()
        finally:
            # If the worker fails, wake HTTP clients and preserve its full traceback.
            self.cancel()

    def generate(self, request, generation_args, progress_callback=None):
        response_queue = Queue()
        with self._request_lock:
            if self.stopping.is_set():
                raise RuntimeError("The server is stopping")
            self.requests.put((response_queue, request, generation_args))
        ctx = response_queue.get()
        if isinstance(ctx, Exception):
            raise ctx

        def responses():
            while True:
                response = response_queue.get()
                if response is None:
                    return
                if isinstance(response, Exception):
                    raise response
                yield response

        request.generation_context = ctx
        return ctx, _process_control_tokens(ctx, responses())

    def _serve_single(self, queued_request):
        response_queue, request, args = queued_request
        try:
            if self.stopping.is_set():
                response_queue.put(RuntimeError("The server is stopping"))
                return
            if request.is_disconnected():
                response_queue.put(ConnectionAbortedError("Client disconnected while queued"))
                return
            self.busy.set()
            ctx, responses = self._start_request(request, args)
            response_queue.put(ctx)
            try:
                for response in responses:
                    response_queue.put(response)
            finally:
                responses.close()
            response_queue.put(None)
        finally:
            error = sys.exception()
            if error is not None:
                response_queue.put(error)
            self.active_context = None
            self.busy.clear()
        # Must run on this serialized GPU worker; a queued request interrupts it between layers.
        self.busy.set()
        try:
            self.engine.prepare_prefix(
                should_stop=lambda: self.stopping.is_set() or not self.requests.empty()
            )
        finally:
            self.busy.clear()

    def _start_request(self, request, args):
        rep = getattr(args.logits, "repetition_penalty", None)
        if rep is None:
            rep = self.defaults.repetition_penalty
        sampling = Sampling(
            temperature=args.sampling.temperature,
            top_p=args.sampling.top_p,
            top_k=args.sampling.top_k,
            presence_penalty=args.logits.presence_penalty,
            repetition_penalty=rep,
            max_tokens=args.max_tokens,
            seed=self.defaults.seed if args.seed is None else args.seed,
        )
        ids = request.prompt_tokens
        sm, sequences = ResponseGenerator._make_state_machine(
            self,
            "flint",
            self.tokenizer,
            args.stop_words,
        )
        ctx = GenerationContext(
            has_tool_calling=bool(request.tools),
            has_thinking=self.tokenizer.has_thinking,
            tool_parser=seeded_tool_parser(sampling, ids),
            sequences=sequences,
            prompt=ids,
        )
        self.active_context = ctx

        def cancelled():
            return self.stopping.is_set() or request.is_disconnected()

        options = {}
        if request.constraint is not None:
            options["grammar"] = request.constraint
        if args.logprobs:
            options["return_logprobs"] = True
        raw = self.engine.generate_tokens(
            ids,
            sampling,
            should_stop=lambda: ctx._should_stop or cancelled(),
            **options,
        )
        return ctx, self.responses(raw, ctx, sm, sampling, cancelled, args.logprobs)

    def responses(self, raw, ctx, sm, sampling, cancelled, logprobs=False):
        state = sm.make_state()
        tokens = []
        complete = False
        try:
            if ctx._should_stop or self.stopping.is_set():
                return
            for event in raw:
                if "metrics" in event:
                    ctx.prompt_cache_count = event["metrics"]["cached_prompt_tokens"]
                    self.record(
                        sampling,
                        "cancelled" if cancelled() else "complete",
                        event["metrics"],
                    )
                    complete = True
                    break
                token = event["token"]
                tokens.append(token)
                state, match, current = sm.match(state, token)
                finish = event["finish_reason"]
                if current is None:
                    finish = "stop"
                    ctx.stop()
                yield Response(
                    event["text"],
                    token,
                    current,
                    match,
                    event["logprob"] if logprobs else 0.0,
                    finish,
                    (),
                )
        finally:
            raw.close()
            if not complete:
                self.record(
                    sampling,
                    "cancelled_or_failed",
                    {
                        "prompt_tokens": len(ctx.prompt),
                        "generated_tokens": len(tokens),
                        "token_ids": tokens,
                    },
                )

    def record(self, sampling, status, metrics):
        if self.metrics_path is None:
            return
        row = {
            "checkpoint": self.engine.identity["model_id"],
            "kv_format": self.engine.startup["kv_format"],
            "sampling": asdict(sampling),
            "status": status,
            **metrics,
        }
        with self.metrics_path.open("a") as output:
            output.write(json.dumps(row) + "\n")
