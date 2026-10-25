import os
import unittest

import pytest

from parallel_workers import RESERVED_CORES, WORKERS_ENV, auto_worker_count, available_cores


class AutoWorkerCountTest(unittest.TestCase):
    def test_leaves_reserved_cores_for_the_desktop(self):
        self.assertEqual(auto_worker_count({}, RESERVED_CORES + 6), 6)

    def test_never_drops_below_one_worker(self):
        for cores in range(1, RESERVED_CORES + 1):
            self.assertEqual(auto_worker_count({}, cores), 1)

    def test_environment_overrides_the_core_count(self):
        self.assertEqual(auto_worker_count({WORKERS_ENV: "3"}, 64), 3)

    def test_invalid_override_fails_instead_of_being_ignored(self):
        for value in ("many", "0", "-2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                auto_worker_count({WORKERS_ENV: value}, 8)

    def test_available_cores_is_positive(self):
        self.assertGreaterEqual(available_cores(), 1)


def test_xdist_auto_uses_this_worker_count(pytestconfig):
    if not pytestconfig.pluginmanager.hasplugin("xdist"):
        pytest.skip("pytest-xdist is not installed")
    result = pytestconfig.hook.pytest_xdist_auto_num_workers(config=pytestconfig)
    assert result == auto_worker_count(os.environ, available_cores())
