# SPDX-License-Identifier: Apache-2.0
"""Accepted thread graphs preserve friction history and lagged physical loads."""

import os
import unittest

import numpy as np
import warp as wp
from test_worker_thread import scene_fixture

from newton_solver_suturebot._src.worker_thread import WorkerThread


class WorkerThreadGraphTests(unittest.TestCase):
    def test_graphs_match_eager_updates_across_moving_buffers_and_timesteps(self):
        if not wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda:
            self.skipTest("Accepted thread graph parity requires CUDA")
        scene, config = scene_fixture()
        reference, captured = WorkerThread(scene, config, capture=False), WorkerThread(scene, config)
        states = (scene.state_0, scene.model.state())
        initial = states[0].body_q.numpy().copy()
        initial_tissue = states[0].particle_q.numpy().copy()
        cases = [(buffer, dt) for dt in (1 / 240, 1 / 480) for buffer in (0, 1)] * 4
        for step, (buffer, dt) in enumerate(cases):
            with self.subTest(step=step):
                scene.state_0 = states[buffer]
                poses = initial.copy()
                poses[0, 0] += (step + 1) * 2e-6
                poses[1, 1] -= (step + 1) * 1e-6
                scene.state_0.body_q.assign(poses)
                tissue = initial_tissue.copy()
                tissue[:, 2] += (step + 1) * 1e-7
                scene.state_0.particle_q.assign(tissue)
                reference.advance(dt)
                captured.advance(dt)
                for expected, actual in zip(reference.state, captured.state, strict=True):
                    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-10)
                np.testing.assert_array_equal(captured.solver.iterations.numpy(), reference.solver.iterations.numpy())
                self.assertEqual(captured.accepted_steps, step + 1)
                self.assertEqual(captured.physical_time_s, reference.physical_time_s)
                self.assertIs(captured.rigid_contact._body_q, scene.state_0.body_q)
                self.assertEqual(captured.rigid_contact._dt, dt)
        self.assertEqual(captured.graph_captures, 4)
        self.assertEqual(captured.graph_launches, len(cases) - 1)
        self.assertEqual(reference.graph_launches, 0)
        self.assertEqual(captured.diagnostics()["graph_launches"], len(cases) - 1)
        # Invalidation warms through exactly one real accepted step, preserving
        # thread history instead of integrating and rolling back a dummy step.
        captured.invalidate_step_graphs()
        self.assertFalse(captured._step_graphs)
        before = captured.accepted_steps
        reference.advance(1 / 240)
        captured.advance(1 / 240)
        self.assertEqual(captured.accepted_steps, before + 1)
        self.assertEqual(captured.graph_launches, len(cases) - 1)
        for expected, actual in zip(reference.state, captured.state, strict=True):
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
