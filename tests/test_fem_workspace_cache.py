# SPDX-License-Identifier: Apache-2.0
"""Adaptive inner tolerances reuse CUDA scratch/graphs without changing FEM."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from test_needle_insertion_example import _example, _prepare


class FEMWorkspaceCacheTests(unittest.TestCase):
    def setUp(self):
        if not wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda:
            self.skipTest("Workspace and graph reuse applies to CUDA FEM")

    def test_revisiting_tolerances_timesteps_and_states_reuses_exact_physics_graphs(self):
        with _example() as example:
            _prepare(example)
            example.start()
            solver, coupler = example.solver, example.coupler
            solver.refresh_tangent(example.state_0.particle_q, example.sim_dt)
            expected = example.model.state()
            rng = np.random.default_rng(891)
            shape = (example.model.particle_count, 3)
            warmstart = rng.normal(0, 1e-8, shape).astype(np.float32)
            cases = [(dt, tol, state) for dt in (1.0, 0.5) for tol in (3e-3, 3e-4, 3e-5) for state in (0, 1)]
            initial_counts = None
            with patch.object(solver, "_make_linear_solver", wraps=solver._make_linear_solver) as allocate:
                for repeat in range(3):
                    for index, (scale, tolerance, buffer) in enumerate(cases):
                        with self.subTest(repeat=repeat, case=index):
                            state = (example.state_0, example.state_1)[buffer]
                            positions = example.selection.vertices.copy()
                            positions[:, 0] += (repeat + 1) * 1e-7 * np.arange(len(positions))
                            state.particle_q.assign(positions)
                            state.particle_qd.assign(rng.normal(0, 1e-5, shape).astype(np.float32))
                            dt = scale * example.sim_dt
                            solver.linear_solver_tolerance = tolerance
                            solver.refresh_tangent(state.particle_q, dt)
                            coupler._forces.assign(rng.normal(0, 1e-5, shape).astype(np.float32))
                            solver.velocity_increment.assign(warmstart)
                            expected.assign(state)
                            solver.step(state, expected, example.control, None, dt, external_forces=coupler._forces)
                            increment = solver.velocity_increment.numpy().copy()
                            solver.velocity_increment.assign(warmstart)
                            coupler._candidate.assign(state)
                            coupler._step_fem(state, example.control, dt)
                            for name in ("particle_q", "particle_qd"):
                                np.testing.assert_allclose(
                                    getattr(coupler._candidate, name).numpy(),
                                    getattr(expected, name).numpy(),
                                    rtol=1e-6,
                                    atol=2e-9,
                                )
                            np.testing.assert_allclose(
                                solver.velocity_increment.numpy(), increment, rtol=1e-6, atol=2e-9
                            )
                            self.assertEqual(solver.linear_solver_tolerance, tolerance)
                    counts = (allocate.call_count, coupler.fem_graph_captures)
                    if initial_counts is None:
                        initial_counts = counts
                        self.assertEqual(counts, (3, len(cases)))
                    else:
                        self.assertEqual(counts, initial_counts)
            self.assertEqual(coupler.fem_graph_launches, 3 * len(cases))

    def test_replaced_workspace_is_not_resurrected_and_cache_memory_is_bounded(self):
        with _example() as example:
            _prepare(example)
            example.start()
            solver, coupler = example.solver, example.coupler
            state, dt = example.state_0, example.sim_dt
            solver.refresh_tangent(state.particle_q, dt)
            for _ in range(2):
                coupler._step_fem(state, example.control, dt)
            previous = solver._lin_solver
            solver._lin_solver = None
            coupler._step_fem(state, example.control, dt)
            self.assertIsNot(solver._lin_solver, previous)
            self.assertFalse(any(workspace is previous for workspace in solver._linear_workspaces.values()))
            previous = solver._lin_solver
            solver._lin_solver = None
            solver.step(state, coupler._candidate, example.control, None, dt, external_forces=coupler._forces)
            self.assertIsNot(solver._lin_solver, previous)
            self.assertFalse(any(workspace is previous for workspace in solver._linear_workspaces.values()))
            # Distinct timesteps and numerical levels retain only bounded
            # scratch and graph history, even for a caller with no steady cycle.
            for index in range(40):
                solver.linear_solver_tolerance = 0.003 / (1 + index)
                trial_dt = dt * (1 + index / 100)
                solver.refresh_tangent(state.particle_q, trial_dt)
                coupler._step_fem(state, example.control, trial_dt)
                coupler._step_fem(state, example.control, trial_dt)
                self.assertLessEqual(len(solver._linear_workspaces), 8)
                self.assertLessEqual(len(coupler._fem_graphs), 32)
            coupler.invalidate_fem_graph()
            self.assertFalse(coupler._fem_graphs)


if __name__ == "__main__":
    unittest.main()
