"""CPU lifecycle and CLI regression checks for the standalone grasp viewer."""

import contextlib
import gc
import io
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from newton_solver_suturebot.examples import example_fem_needle_grasp as grasp_example


class _PoolResource:
    """External worker resource; constructor ownership must survive later errors."""

    def __init__(self, *, fail_observe=False):
        self.fail_observe = fail_observe
        self.close_calls = 0

    def observe(self):
        if self.fail_observe:
            raise RuntimeError("worker observation failed")
        return []

    def geometry(self):
        return {}

    def close(self):
        self.close_calls += 1


class _SingleCloseViewer:
    """USD-style close contract: closing a finalized stage again is an error."""

    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        if self.close_calls > 1:
            raise RuntimeError("viewer already finalized")


class GraspExampleLifecycleTests(unittest.TestCase):
    @staticmethod
    def args(**updates):
        defaults = dict(
            device="cpu",
            num_envs=1,
            substeps=4,
            controller="hold",
            viewer="null",
            speed=0.003,
            settle_seconds=0.5,
            spacing=0.45,
        )
        return SimpleNamespace(**(defaults | updates))

    def test_constructor_failure_releases_worker_resource_once(self):
        pool = _PoolResource(fail_observe=True)
        with patch.object(grasp_example, "ParallelSutureScenes", return_value=pool):
            with self.assertRaisesRegex(RuntimeError, "worker observation failed"):
                grasp_example.Example(_SingleCloseViewer(), self.args())
        gc.collect()
        self.assertEqual(pool.close_calls, 1)

    def test_dropped_example_releases_workers_before_browser_recreation(self):
        pool = _PoolResource()
        with (
            patch.object(grasp_example, "ParallelSutureScenes", return_value=pool),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            example = grasp_example.Example(_SingleCloseViewer(), self.args())
        reference = weakref.ref(example)
        del example
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(pool.close_calls, 1)

    def _run_main(self, *, fail=False):
        viewer = _SingleCloseViewer()
        resources = []
        real_parser = grasp_example.Example.create_parser

        class LightweightExample:
            _instances = weakref.WeakSet()
            create_parser = staticmethod(real_parser)

            def __init__(self, viewer, args):
                self.viewer, self.args = viewer, args
                self.pool = _PoolResource()
                resources.append(self.pool)
                self._instances.add(self)

            def close(self):
                self.pool.close()

        def runner(example, args):
            if fail:
                raise RuntimeError("physical worker failed")
            # Both newton.examples.run and the existing record_examples runner
            # own ordinary successful viewer shutdown.
            example.viewer.close()

        with (
            patch.object(grasp_example, "Example", LightweightExample),
            patch.object(
                grasp_example.newton.examples,
                "init",
                return_value=(viewer, self.args()),
            ),
            patch.object(grasp_example.newton.examples, "run", side_effect=runner),
            patch("sys.argv", ["grasp-example", "--device", "cpu", "--viewer", "null"]),
        ):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "physical worker failed"):
                    grasp_example.main()
            else:
                grasp_example.main()
        return viewer, resources

    def test_successful_standard_or_recording_runner_is_not_closed_twice(self):
        viewer, resources = self._run_main()
        self.assertEqual(viewer.close_calls, 1)
        # The real Example additionally owns weakref finalization; this check
        # focuses on preserving the runner's single-close viewer contract.
        self.assertEqual(len(resources), 1)

    def test_runner_error_closes_viewer_and_releases_workers(self):
        viewer, resources = self._run_main(fail=True)
        self.assertEqual(viewer.close_calls, 1)
        self.assertEqual(resources[0].close_calls, 1)

    def final_state(self, **options):
        example = grasp_example.Example.__new__(grasp_example.Example)
        example.args = self.args(**options)
        example.completed, example.sim_time = 360, 6.0
        rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        example.geometry = {
            "tissue_rest_positions": rest,
            "tissue_tetrahedra": np.array([[0, 1, 2, 3]]),
            "fixed_particle_indices": np.array([0]),
        }
        example.observations = [
            {
                "tissue_positions": rest.copy(),
                "rigid_body_poses": np.zeros((1, 7)),
                "thread_positions": np.zeros((2, 3)),
                "diagnostics": {
                    "independent_free_needle": True,
                    "needle_pose_drive": False,
                },
            }
        ]
        example.controller = SimpleNamespace(
            report=lambda: [{"commanded_arc_complete": False, "entry_count": 0, "exit_count": 0}]
        )
        return example

    def test_custom_speed_or_settling_does_not_imply_six_second_completion(self):
        for options in ({"speed": 0.001}, {"settle_seconds": 7.0}):
            with (
                self.subTest(options=options),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.final_state(**options).test_final()

    def test_default_six_second_procedure_still_requires_entry_and_exit(self):
        with self.assertRaisesRegex(AssertionError, "entry and exit"):
            self.final_state().test_final()

    def test_custom_settings_do_not_bypass_physical_integrity(self):
        example = self.final_state(speed=0.001)
        example.observations[0]["tissue_positions"][3, 2] = -1.0
        with self.assertRaisesRegex(AssertionError, "inverted"):
            example.test_final()


if __name__ == "__main__":
    unittest.main()
