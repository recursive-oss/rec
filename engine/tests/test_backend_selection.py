"""Exercise Engine backend selection without loading weights or compiling kernels."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import pytest

from rec_mlx import engine, kernels, native, quantized


@pytest.fixture
def initialization(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text('{"eos_token_id": [0]}')
    manifest = tmp_path / "manifest.json"
    model = Mock()
    native_runtime = SimpleNamespace(model=model)
    resident_runtime = SimpleNamespace(model=model)
    sampler = Mock()
    mocks = SimpleNamespace(
        directory=tmp_path,
        manifest=manifest,
        device_info=Mock(),
        validate=Mock(
            return_value={
                "max_context": quantized.QATModel.context_limit,
                "weight_format": "mlx-affine-int4",
                "kv_format": "bf16",
            }
        ),
        load_tokenizer=Mock(return_value=SimpleNamespace(chat_template="test")),
        model=Mock(context_limit=quantized.QATModel.context_limit, return_value=model),
        native=Mock(return_value=native_runtime),
        resident=Mock(return_value=(resident_runtime, sampler)),
        compiler=Mock(),
        evaluate=Mock(),
    )
    monkeypatch.setattr(engine.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(engine.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(engine.subprocess, "check_output", Mock(return_value=str(64 * 1024**3)))
    monkeypatch.setattr(engine, "validate", mocks.validate)
    monkeypatch.setattr(engine, "load_tokenizer", mocks.load_tokenizer)
    monkeypatch.setattr(quantized, "QATModel", mocks.model)
    monkeypatch.setattr(native, "NativeRuntime", mocks.native)
    monkeypatch.setattr(kernels, "build_decoder", mocks.resident)
    monkeypatch.setattr(mx, "device_info", mocks.device_info)
    monkeypatch.setattr(mx, "eval", mocks.evaluate)
    for name in ("set_cache_limit", "clear_cache", "reset_peak_memory"):
        monkeypatch.setattr(mx, name, Mock())
    monkeypatch.setattr(mx, "get_peak_memory", Mock(return_value=0))
    grammar = ModuleType("rec_mlx.grammar")
    grammar.ConstraintCompiler = mocks.compiler
    monkeypatch.setitem(sys.modules, "rec_mlx.grammar", grammar)
    return mocks


def initialize(mocks, name, architecture):
    mocks.device_info.return_value = {"device_name": name, "architecture": architecture}
    return engine.Engine(mocks.directory, mocks.manifest, context_limit=1024)


def test_exact_m2_max_selects_native_without_resident_sampler(initialization):
    mocks = initialization
    selected = initialize(mocks, "Apple M2 Max", "applegpu_g14g")

    mocks.native.assert_called_once_with(mocks.model.return_value)
    mocks.resident.assert_not_called()
    assert selected.runtime is mocks.native.return_value
    assert selected.resident_sampler is None
    assert selected.startup["device_profile"] == "M2 Max"
    assert selected.startup["decode_strategy"] == "native_mlx"
    mocks.model.assert_called_once_with(mocks.directory)
    mocks.load_tokenizer.assert_called_once_with(mocks.directory)
    mocks.evaluate.assert_called_once_with(mocks.model.return_value.native.parameters.return_value)
    mocks.compiler.assert_not_called()


@pytest.mark.parametrize(
    ("name", "architecture", "profile"),
    [
        ("Apple M4 Pro", "applegpu_g16s", "M4 Pro"),
        ("Apple M5 Pro", "applegpu_g17s", "M5 Pro"),
        ("Apple M5 Max", "applegpu_g17s", "M5 Max"),
    ],
)
def test_supported_devices_keep_resident_backend(initialization, name, architecture, profile):
    mocks = initialization
    selected = initialize(mocks, name, architecture)

    mocks.resident.assert_called_once_with(mocks.model.return_value, optimized=True)
    mocks.native.assert_not_called()
    runtime, sampler = mocks.resident.return_value
    assert selected.runtime is runtime
    assert selected.resident_sampler is sampler
    assert selected.startup["device_profile"] == profile
    assert selected.startup["decode_strategy"] == "resident_sampling"
    mocks.model.assert_called_once_with(mocks.directory)
    mocks.compiler.assert_not_called()


@pytest.mark.parametrize(
    ("name", "architecture"),
    [
        ("Apple M2", "applegpu_g14g"),
        ("Apple M2 Pro", "applegpu_g14g"),
        ("Apple M2 Max Extra", "applegpu_g14g"),
        ("Apple M5 Max", "unsupported"),
    ],
)
def test_unsupported_device_rejected_before_loading(initialization, name, architecture):
    mocks = initialization
    with pytest.raises(ValueError, match="Unsupported resident kernel device"):
        initialize(mocks, name, architecture)

    mocks.validate.assert_not_called()
    mocks.load_tokenizer.assert_not_called()
    mocks.model.assert_not_called()
    mocks.native.assert_not_called()
    mocks.resident.assert_not_called()
    mocks.compiler.assert_not_called()
