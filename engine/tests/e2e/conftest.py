"""Shared session-scoped (one resident engine for all e2e modules) fixtures for the real-weights e2e tests."""

import pytest
from e2e_config import CONTEXT_LIMIT, MANIFEST_NAME, MODEL_DIR

from rec_mlx import thinking


@pytest.fixture(scope="session")
def engine():
    if MODEL_DIR is None:
        pytest.skip("REC_E2E_MODEL_DIR is not set (flint-int4 weights unavailable)")
    from rec_mlx.engine import Engine

    return Engine(MODEL_DIR, MODEL_DIR / MANIFEST_NAME, context_limit=CONTEXT_LIMIT)


@pytest.fixture(scope="session")
def think(engine):
    return thinking.resolve_think_ids(engine.tokenizer)
