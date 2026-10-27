import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from rec_mlx.cli import main


class DefaultsTests(unittest.TestCase):
    def test_flint_grammar_and_fast_sync_defaults_and_opt_outs(self):
        for options, fast_sync, grammar, enforce in (
            ([], True, "flint", False),
            (["--no-fast-sync"], False, "flint", False),
            (["--grammar", "none"], True, "none", False),
            (["--enforce-tool-schemas"], True, "flint", True),
        ):
            with self.subTest(options=options):
                engine = Mock(identity={"model_id": "flint"}, startup={"chat_template": "flint"})
                server = Mock(server_port=8080)
                with (
                    patch(
                        "sys.argv",
                        [
                            "rec-mlx",
                            "serve",
                            "flint",
                            "--model-dir",
                            ".",
                            "--manifest",
                            "model.json",
                            *options,
                        ],
                    ),
                    patch.dict(os.environ, {"MLX_METAL_FAST_SYNCH": "0" if fast_sync else "1"}),
                    patch("rec_mlx.cli.Engine", return_value=engine) as construct,
                    patch("rec_mlx.server.make_server", return_value=server) as make,
                    patch("rec_mlx.cli.signal.signal"),
                    redirect_stdout(io.StringIO()) as stdout,
                ):

                    def check_defaults(*args, _fast_sync=fast_sync, _engine=engine, **kwargs):
                        self.assertEqual(
                            os.environ["MLX_METAL_FAST_SYNCH"], "1" if _fast_sync else "0"
                        )
                        self.assertNotIn("chat_template", kwargs)
                        return _engine

                    construct.side_effect = check_defaults
                    main()
                    self.assertEqual(make.call_args.args[1], ("127.0.0.1", 8080))
                    self.assertEqual(make.call_args.kwargs["grammar_mode"], grammar)
                    self.assertEqual(construct.call_args.kwargs["enforce_tool_schemas"], enforce)
                    ready = json.loads(stdout.getvalue().splitlines()[0])
                    self.assertEqual(ready["grammar"], grammar)
                    self.assertEqual(ready["fast_sync"], fast_sync)
                    self.assertEqual(ready["enforce_tool_schemas"], enforce)
                    self.assertEqual(ready["chat_template"], "flint")

    def test_serve_host_wildcard_binds_all_interfaces(self):
        engine = Mock(identity={"model_id": "flint"}, startup={"chat_template": "flint"})
        server = Mock(server_port=8080)
        with (
            patch(
                "sys.argv",
                [
                    "rec-mlx",
                    "serve",
                    "flint",
                    "--model-dir",
                    ".",
                    "--manifest",
                    "model.json",
                    "--host",
                    "0.0.0.0",
                ],
            ),
            patch.dict(os.environ, {"MLX_METAL_FAST_SYNCH": "1"}),
            patch("rec_mlx.cli.Engine", return_value=engine),
            patch("rec_mlx.server.make_server", return_value=server) as make,
            patch("rec_mlx.cli.signal.signal"),
            redirect_stdout(io.StringIO()),
        ):
            main()
            self.assertEqual(make.call_args.args[1], ("0.0.0.0", 8080))
