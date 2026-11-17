# SPDX-License-Identifier: Apache-2.0
"""Loaded BLAS limits must be effective in spawn children, not only in env."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import threadpoolctl

from newton_solver_suturebot.examples.grasp.parallel import _worker_thread_limits


class WorkerThreadLimitTests(unittest.TestCase):
    def probe(self, threads, failure="none"):
        env = os.environ.copy()
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
            env.pop(name, None)
        if threads is not None:
            env["OPENBLAS_NUM_THREADS"] = str(threads)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("_parallel_blas_probe.py")), failure],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=True,
        )
        result = json.loads(completed.stdout.splitlines()[-1])
        if not result["worker"]["before"]["pools"]:
            self.skipTest("NumPy has no threadpoolctl-supported BLAS library")
        self.assertEqual(result["parent_before"], result["parent_after"])
        worker = result["worker"]
        self.assertEqual(worker["before"], worker["after"])
        for phase, observation in worker["events"].items():
            with self.subTest(phase=phase):
                self.assertTrue(observation["pools"])
                self.assertEqual(set(observation["pools"].values()), {1})
                self.assertEqual(set(observation["environment"].values()), {"1"})
        return result

    def test_unset_environment_limits_already_loaded_pools_in_fresh_spawn(self):
        result = self.probe(None)
        self.assertIsNone(result["worker"]["before"]["environment"]["OPENBLAS_NUM_THREADS"])
        self.assertIn("observe", result["worker"]["events"])

    def test_eight_thread_import_becomes_one_then_restores_after_worker(self):
        result = self.probe(8)
        self.assertEqual(set(result["worker"]["before"]["pools"].values()), {8})
        self.assertIn("scene_close", result["worker"]["events"])

    def test_setup_observation_and_cleanup_failures_restore_limits(self):
        for failure in ("setup", "observe", "cleanup", "connection"):
            with self.subTest(failure=failure):
                result = self.probe(8, failure)
                worker = result["worker"]
                if failure in ("setup", "observe"):
                    self.assertFalse(worker["messages"][-1][0])
                    self.assertIn(
                        f"injected {failure if failure == 'setup' else 'observation'} failure",
                        worker["messages"][-1][1],
                    )
                else:
                    self.assertIn("injected", worker["raised"])
                if failure != "setup":
                    self.assertIn("viewer_close", worker["events"])

    def test_refresh_discovers_late_library_and_context_restores_both(self):
        # Model two distinct libraries, one becoming visible during scene import.
        pools = {"initial": 8}
        contexts = []

        class Limit:
            def __init__(self, *, limits, user_api):
                self.previous = dict(pools)
                self.assertions = (limits, user_api)
                for name in pools:
                    pools[name] = limits
                contexts.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                pools.update(self.previous)

        previous_env = dict(os.environ)
        with mock.patch.object(threadpoolctl, "threadpool_limits", Limit):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with _worker_thread_limits() as refresh:
                    self.assertEqual(pools, {"initial": 1})
                    pools["late"] = 6
                    refresh()
                    self.assertEqual(pools, {"initial": 1, "late": 1})
                    raise RuntimeError("injected")
        self.assertEqual(pools, {"initial": 8, "late": 6})
        self.assertEqual(dict(os.environ), previous_env)
        self.assertEqual([context.assertions for context in contexts], [(1, "blas"), (1, "blas")])


if __name__ == "__main__":
    unittest.main()
