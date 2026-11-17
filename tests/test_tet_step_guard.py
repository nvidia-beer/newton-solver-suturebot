# SPDX-License-Identifier: Apache-2.0
"""Swept material feasibility and actual coupling transaction rollback."""

import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from test_needle_insertion_example import _example, _prepare
from test_needle_tissue_coupler import _native_state

from newton_solver_suturebot import NeedleStepTooLarge
from newton_solver_suturebot._src.solvers.coupled.tet_step import TetStepGuard


class TetStepGuardTests(unittest.TestCase):
    def fixture(self, start=None, end=None, *, materials=None, poses=None):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        rest = np.vstack((np.zeros(3), np.eye(3))).astype(np.float32)
        start = rest if start is None else np.asarray(start, dtype=np.float32)
        end = start if end is None else np.asarray(end, dtype=np.float32)
        model = SimpleNamespace(
            device=device,
            tet_count=1,
            tet_indices=wp.array(np.array([[0, 1, 2, 3]], dtype=np.int32), dtype=int, device=device),
            tet_poses=wp.array(np.eye(3)[None] if poses is None else poses, dtype=wp.mat33, device=device),
            tet_materials=wp.array([[1.0, 1.0, 0.0]] if materials is None else materials, dtype=float, device=device),
        )
        return (
            TetStepGuard(model),
            SimpleNamespace(particle_q=wp.array(start, dtype=wp.vec3, device=device)),
            SimpleNamespace(particle_q=wp.array(end, dtype=wp.vec3, device=device)),
        )

    def test_positive_endpoints_do_not_hide_interior_orientation_crossings(self):
        rest = np.vstack((np.zeros(3), np.eye(3)))
        for stretch in ((-3, -1, 1), (-4, -2, 2), (-3, -1, 1 + 1e-6), (-1, -1, 1)):
            with self.subTest(stretch=stretch):
                end = rest * np.asarray(stretch)
                self.assertGreater(np.linalg.det(end[1:] - end[0]), 0)
                guard, start, end = self.fixture(end=end)
                before = (start.particle_q.numpy().copy(), end.particle_q.numpy().copy())
                with self.assertRaisesRegex(NeedleStepTooLarge, "nonpositive swept volume"):
                    guard.check(start, end)
                np.testing.assert_array_equal(start.particle_q.numpy(), before[0])
                np.testing.assert_array_equal(end.particle_q.numpy(), before[1])

    def test_positive_compression_shear_translation_and_scale_are_not_projected(self):
        rest = np.vstack((np.zeros(3), np.eye(3)))
        for scale in (1.0, 1e-3, 1e-5):
            for height in (1.0, 0.2, 0.001):
                with self.subTest(scale=scale, height=height):
                    start = scale * rest
                    end = start @ np.array([[1, 0.2, 0], [0, 1, 0], [0, 0, height]])
                    end += scale * np.array([0.3, -0.2, 0.5])
                    guard, state_in, state_out = self.fixture(start, end, poses=(np.eye(3) / scale)[None])
                    actual = state_out.particle_q.numpy().copy()
                    lower_bound = guard.check(state_in, state_out)
                    self.assertGreater(lower_bound, 0)
                    self.assertAlmostEqual(lower_bound, height, delta=2e-7)
                    np.testing.assert_array_equal(state_out.particle_q.numpy(), actual)

    def test_large_finite_coefficients_cannot_overflow_derivative_root_search(self):
        # Geometry and rest inverse are intentionally extreme but finite. The
        # signed-volume coefficients fit float64 while their squares do not.
        rest = 1e30 * np.vstack((np.zeros(3), np.eye(3)))
        end = rest * np.array([-3.0, -1.0, 2.0])
        guard, state_in, state_out = self.fixture(rest, end, poses=(1e30 * np.eye(3))[None])
        self.assertTrue(np.isfinite(state_in.particle_q.numpy()).all())
        self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())
        with self.assertRaisesRegex(NeedleStepTooLarge, "nonpositive swept volume"):
            guard.check(state_in, state_out)

    def test_invalid_endpoint_initial_volume_and_nonfinite_geometry_fail_closed(self):
        rest = np.vstack((np.zeros(3), np.eye(3)))
        for value in (-0.1, 0.0, np.nan, np.inf):
            for endpoint in ("start", "end"):
                with self.subTest(value=value, endpoint=endpoint):
                    modified = rest.copy()
                    modified[3, 2] = value
                    guard, state_in, state_out = self.fixture(**{endpoint: modified})
                    with self.assertRaises(NeedleStepTooLarge):
                        guard.check(state_in, state_out)

    def test_zero_material_reserved_slots_are_skipped_before_touching_geometry(self):
        guard, state_in, state_out = self.fixture(
            np.full((4, 3), np.nan), materials=[[0.0, 0.0, 0.0]], poses=np.zeros((1, 3, 3))
        )
        self.assertEqual(guard.check(state_in, state_out), math.inf)
        # In-place material activation must immediately enable the guard.
        guard.model.tet_materials.assign(np.array([[1, 1, 0]], np.float32))
        with self.assertRaises(NeedleStepTooLarge):
            guard.check(state_in, state_out)

    def test_coupled_rejection_restores_native_and_fem_state_then_retry_commits_once(self):
        with _example() as example:
            _prepare(example)
            example.start()
            coupler, interaction = example.coupler, example.interaction
            example.state_0.body_f.assign(np.array([[0, 0, 1e-5, 0, 0, 0]], np.float32))
            example.state_0.particle_f.fill_(wp.vec3(0.123, 0.234, 0.345))
            example.solver.velocity_increment.fill_(wp.vec3(1e-7, -2e-7, 3e-7))
            names = (
                "particle_q",
                "particle_qd",
                "particle_f",
                "body_q",
                "body_qd",
                "joint_q",
                "joint_qd",
                "body_f",
            )
            before = [
                {name: getattr(state, name).numpy().copy() for name in names}
                for state in (example.state_0, example.state_1)
            ]
            native = _native_state(example.rigid_solver)
            increment = example.solver.velocity_increment.numpy().copy()
            load = coupler._load.numpy().copy()
            events = interaction.events
            history = (interaction.status, interaction.reason, interaction.puncture_count, interaction.track_count)
            solve = coupler._solve_host

            def adversarial_converged_candidate(state_in, control, dt, *args, solve=solve, coupler=coupler):
                converged = solve(state_in, control, dt, *args)
                self.assertTrue(converged)
                if dt == example.sim_dt:
                    q = coupler._candidate.particle_q.numpy()
                    ids = example.model.tet_indices.numpy()[0]
                    # A converged interface residual cannot certify this
                    # deliberately invalid FEM endpoint. Native MuJoCo has
                    # really advanced and must be restored by rejection.
                    q[ids[3]] = q[ids[0]] - 0.1 * (q[ids[3]] - q[ids[0]])
                    coupler._candidate.particle_q.assign(q)
                return converged

            with (
                patch.object(coupler, "_solve_host", side_effect=adversarial_converged_candidate),
                patch.object(interaction, "accept", wraps=interaction.accept) as accept,
            ):
                with self.assertRaisesRegex(NeedleStepTooLarge, "FEM tetrahedron"):
                    coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
                self.assertEqual(accept.call_count, 0)
                self.assertEqual(coupler.tet_step_rejections, 1)
                self.assertEqual(coupler.minimum_accepted_tet_j, math.inf)
                np.testing.assert_array_equal(_native_state(example.rigid_solver), native)
                np.testing.assert_array_equal(example.solver.velocity_increment.numpy(), increment)
                np.testing.assert_array_equal(coupler._load.numpy(), load)
                for state, values in zip((example.state_0, example.state_1), before, strict=True):
                    for name in names:
                        np.testing.assert_array_equal(getattr(state, name).numpy(), values[name], err_msg=name)
                self.assertIs(interaction.events, events)
                self.assertEqual(
                    (interaction.status, interaction.reason, interaction.puncture_count, interaction.track_count),
                    history,
                )
                dt = example.sim_dt / 2
                example.solver.refresh_tangent(example.state_0.particle_q, dt)
                start_time = float(example.rigid_solver.mj_data.time)
                coupler.step(example.state_0, example.state_1, example.control, dt)
                self.assertEqual(accept.call_count, 1)
                self.assertEqual(coupler.tet_step_rejections, 1)
                self.assertGreater(coupler.minimum_accepted_tet_j, 0)
                self.assertTrue(math.isfinite(coupler.minimum_accepted_tet_j))
                self.assertAlmostEqual(float(example.rigid_solver.mj_data.time), start_time + dt, places=12)


if __name__ == "__main__":
    unittest.main()
