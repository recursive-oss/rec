import os

import pytest

from parallel_workers import auto_worker_count, available_cores


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config):
    return auto_worker_count(os.environ, available_cores())
