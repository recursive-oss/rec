import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rec_mlx.cli import main


class DefaultsTests(unittest.TestCase):
    def serve(self, *options):
        """Run `rec-mlx serve` with a mocked engine and server and record what happened."""
        engine = Mock(identity={"model_id": "flint"}, startup={"chat_template": "flint"})
        seen = {}

        def construct_engine(*args, **kwargs):
            seen["fast_synch"] = os.environ["MLX_METAL_FAST_SYNCH"]
            return engine

        argv = ["rec-mlx", "serve", "flint", "--model-dir", ".", "--manifest", "model.json"]
        code = None
        with (
            patch("sys.argv", [*argv, *options]),
            # A stale value proves main() sets the variable before the engine loads.
            patch.dict(os.environ, {"MLX_METAL_FAST_SYNCH": "stale"}),
            patch("rec_mlx.cli.Engine", side_effect=construct_engine) as construct,
            patch("rec_mlx.server.make_server", return_value=Mock(server_port=8080)) as make,
            patch("rec_mlx.cli.signal.signal"),
            redirect_stdout(io.StringIO()) as stdout,
            redirect_stderr(io.StringIO()) as stderr,
        ):
            try:
                main()
            except SystemExit as exit:
                code = exit.code
        return SimpleNamespace(
            construct=construct,
            make=make,
            exit=code,
            fast_synch=seen.get("fast_synch"),
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
        )

    def test_flint_grammar_and_fast_sync_defaults_and_opt_outs(self):
        for options, fast_sync, grammar, enforce in (
            ([], True, "flint", False),
            (["--no-fast-sync"], False, "flint", False),
            (["--grammar", "none"], True, "none", False),
            (["--enforce-tool-schemas"], True, "flint", True),
        ):
            with self.subTest(options=options):
                run = self.serve(*options)
                self.assertIsNone(run.exit)
                self.assertEqual(run.fast_synch, "1" if fast_sync else "0")
                self.assertNotIn("chat_template", run.construct.call_args.kwargs)
                self.assertEqual(run.construct.call_args.kwargs["enforce_tool_schemas"], enforce)
                self.assertEqual(run.make.call_args.args[1], ("127.0.0.1", 8080))
                self.assertEqual(run.make.call_args.kwargs["grammar_mode"], grammar)
                ready = json.loads(run.stdout.splitlines()[0])
                self.assertEqual(ready["grammar"], grammar)
                self.assertEqual(ready["fast_sync"], fast_sync)
                self.assertEqual(ready["enforce_tool_schemas"], enforce)
                self.assertEqual(ready["chat_template"], "flint")

    def test_non_loopback_host_without_api_key_is_rejected_before_loading(self):
        for host in ("0.0.0.0", "::", "192.168.1.5"):
            with self.subTest(host=host):
                run = self.serve("--host", host)
                self.assertEqual(run.exit, 2)
                run.construct.assert_not_called()
                run.make.assert_not_called()
                self.assertIn("--api-key-file is required", run.stderr)

    def test_non_loopback_host_with_invalid_api_key_file_is_rejected_before_loading(self):
        for contents in ("", "\n", "  \t\n", "two tokens\n"):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as directory:
                key = Path(directory) / "api-key"
                key.write_text(contents)
                run = self.serve("--host", "0.0.0.0", "--api-key-file", str(key))
                self.assertEqual(run.exit, 2)
                run.construct.assert_not_called()
                run.make.assert_not_called()
                self.assertIn("API-key file must contain one nonempty token", run.stderr)

    def test_host_must_be_an_ip_literal(self):
        for host in ("localhost", "example.com", "", "fe80::1%en0"):
            with self.subTest(host=host):
                run = self.serve("--host", host)
                self.assertEqual(run.exit, 2)
                run.construct.assert_not_called()
                run.make.assert_not_called()
                self.assertIn("--host must be an IP address", run.stderr)

    def test_wildcard_bind_with_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "api-key"
            key.write_text("secret\n")
            run = self.serve("--host", "0.0.0.0", "--api-key-file", str(key))
        self.assertIsNone(run.exit)
        run.construct.assert_called_once()
        self.assertEqual(run.make.call_args.args[1], ("0.0.0.0", 8080))
        self.assertEqual(run.make.call_args.kwargs["api_key"], "secret")

    def test_loopback_hosts_need_no_api_key(self):
        for host, bind in (
            ("127.0.0.1", "127.0.0.1"),
            ("127.0.0.2", "127.0.0.2"),
            ("::1", "::1"),
            ("::ffff:127.0.0.1", "127.0.0.1"),
        ):
            with self.subTest(host=host):
                run = self.serve("--host", host)
                self.assertIsNone(run.exit)
                run.construct.assert_called_once()
                self.assertEqual(run.make.call_args.args[1], (bind, 8080))
                self.assertIsNone(run.make.call_args.kwargs["api_key"])
