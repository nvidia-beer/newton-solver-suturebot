# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared physical-step acceptance and rollback for material tunnel multipliers."""

import unittest
from unittest.mock import patch

import numpy as np
from test_needle_insertion_example import _example
from test_needle_shaft_preconditioner import history, setup
from test_needle_tissue_coupler import _native_state

from newton_solver_suturebot import NeedleTissueCoupler
from newton_solver_suturebot._src.solvers.coupled.interface.tunnel_alm import _TunnelALM

_STEP = 0.0005
_FIELDS = ("particle_q", "particle_qd", "particle_f", "body_q", "body_qd", "body_f")


def _prepare(example, *, friction=0.0):
    # Accepted geometric probes create a preloaded material track. The tested
    # advance itself uses real FEM and native MuJoCo, without prescribed poses.
    setup(example, _STEP, 1500.0)
    interaction = example.interaction
    interaction.shaft_damping_per_m = 0.0
    interaction.friction = friction
    interaction.tunnel_alm = _TunnelALM(
        interaction, penalty_per_m=1500.0, compliance_per_m=1 / 1500.0, clearance=2e-5, tolerance=2e-6
    )
    coupler = NeedleTissueCoupler(
        example.model,
        example.solver,
        example.rigid_solver,
        interaction,
        example.needle_body,
        example.drive.com,
        absolute_tolerance=2e-7,
        relative_tolerance=1e-4,
        max_iterations=160,
        shaft_preconditioner=False,
    )
    return coupler, interaction.tunnel_alm


def _arrays(state):
    return {name: getattr(state, name).numpy().copy() for name in _FIELDS}


class TunnelCouplerTests(unittest.TestCase):
    def test_dual_cycles_share_one_physical_accept_and_do_not_tighten_short_primal_blocks(self):
        with _example() as example:
            # Nonzero friction retains real tangential dual cycles after the
            # compliant normal reaction has been eliminated analytically.
            coupler, alm = _prepare(example, friction=0.02)
            configured = example.solver.linear_solver_tolerance
            requested_tolerances, block_ends = [], []
            original_step, original_update = coupler._step_fem, alm.update_duals

            def structural_step(*args):
                requested_tolerances.append(example.solver.linear_solver_tolerance)
                return original_step(*args)

            def dual_update():
                block_ends.append(coupler.iterations)
                return original_update()

            with (
                patch.object(coupler, "_step_fem", side_effect=structural_step),
                patch.object(alm, "update_duals", side_effect=dual_update),
                patch.object(alm, "commit", wraps=alm.commit) as commit,
                patch.object(example.interaction, "accept", wraps=example.interaction.accept) as accept,
            ):
                coupler.step(example.state_0, example.state_1, example.control, _STEP)
            self.assertEqual(accept.call_count, 1)
            self.assertEqual(commit.call_count, 1)
            self.assertGreater(alm.dual_updates, 1)
            self.assertGreater(np.linalg.norm(alm.committed_tangent.numpy()), 0)
            self.assertEqual(len(requested_tolerances), coupler.iterations)
            self.assertLessEqual(coupler.iterations, coupler.max_iterations)
            self.assertLessEqual(max(np.diff([0, *block_ends, coupler.iterations])), 8)
            self.assertEqual(requested_tolerances, [configured] * coupler.iterations)
            self.assertEqual(example.solver.linear_solver_tolerance, configured)
            self.assertFalse(alm._open)
            self.assertGreater(np.linalg.norm(alm.committed_normal.numpy()), 0)
            self.assertLessEqual(alm.normal_error, alm.tolerance)
            self.assertTrue(alm.converged(coupler.absolute_tolerance, coupler.relative_tolerance))
            self.assertLess(example.interaction.force_balance_error, 1e-8)
            self.assertLess(example.interaction.torque_balance_error, 1e-9)
            self.assertGreater(
                np.linalg.norm(example.state_1.particle_q.numpy() - example.state_0.particle_q.numpy()), 0
            )

    def test_exhausted_shared_budget_discards_duals_and_retry_matches_fresh_step(self):
        outcomes = []
        for reject_first in (True, False):
            with _example() as example:
                coupler, alm = _prepare(example, friction=0.02)
                if reject_first:
                    before, native = _arrays(example.state_0), _native_state(example.rigid_solver)
                    warm = example.solver.velocity_increment.numpy().copy()
                    accepted_history = history(example.interaction)
                    coupler.max_iterations = 12
                    with patch.object(example.interaction, "accept", wraps=example.interaction.accept) as accept:
                        with self.assertRaisesRegex(RuntimeError, "last converged-primal tunnel"):
                            coupler.step(example.state_0, example.state_1, example.control, _STEP)
                        self.assertEqual(accept.call_count, 0)
                    self.assertEqual(coupler.iterations, 12)
                    self.assertGreater(alm.dual_updates, 0)
                    self.assertFalse(alm._open)
                    np.testing.assert_array_equal(alm.committed_normal.numpy(), 0)
                    np.testing.assert_array_equal(alm.normal.numpy(), alm.committed_normal.numpy())
                    np.testing.assert_array_equal(alm.tangent.numpy(), alm.committed_tangent.numpy())
                    np.testing.assert_array_equal(_native_state(example.rigid_solver), native)
                    np.testing.assert_array_equal(example.solver.velocity_increment.numpy(), warm)
                    for name, value in before.items():
                        np.testing.assert_array_equal(getattr(example.state_0, name).numpy(), value)
                    self.assertEqual(history(example.interaction), accepted_history)
                    coupler.max_iterations = 160
                coupler.step(example.state_0, example.state_1, example.control, _STEP)
                outcomes.append(
                    (
                        _arrays(example.state_1),
                        alm.committed_normal.numpy(),
                        coupler.iterations,
                        alm.committed_tangent.numpy(),
                    )
                )
        self.assertEqual(outcomes[0][2], outcomes[1][2])
        for name in _FIELDS:
            np.testing.assert_array_equal(outcomes[0][0][name], outcomes[1][0][name])
        np.testing.assert_array_equal(outcomes[0][1], outcomes[1][1])
        np.testing.assert_array_equal(outcomes[0][3], outcomes[1][3])

    def test_late_resolution_rejection_preserves_nonzero_committed_multipliers(self):
        with _example() as example:
            coupler, alm = _prepare(example)
            coupler.step(example.state_0, example.state_1, example.control, _STEP)
            example.state_0.assign(example.state_1)
            example.solver.refresh_tangent(example.state_0.particle_q, _STEP)
            before, native = _arrays(example.state_0), _native_state(example.rigid_solver)
            normal, tangent = alm.committed_normal.numpy().copy(), alm.committed_tangent.numpy().copy()
            self.assertGreater(np.linalg.norm(normal), 0)
            accepted_history = history(example.interaction)
            with (
                patch.object(
                    example.interaction, "check_step_resolution", side_effect=RuntimeError("resolution guard")
                ),
                patch.object(alm, "commit", wraps=alm.commit) as commit,
                patch.object(example.interaction, "accept", wraps=example.interaction.accept) as accept,
            ):
                with self.assertRaisesRegex(RuntimeError, "resolution guard"):
                    coupler.step(example.state_0, example.state_1, example.control, _STEP)
            self.assertEqual((accept.call_count, commit.call_count), (0, 0))
            self.assertFalse(alm._open)
            np.testing.assert_array_equal(alm.committed_normal.numpy(), normal)
            np.testing.assert_array_equal(alm.committed_tangent.numpy(), tangent)
            np.testing.assert_array_equal(_native_state(example.rigid_solver), native)
            for name, value in before.items():
                np.testing.assert_array_equal(getattr(example.state_0, name).numpy(), value)
            self.assertEqual(history(example.interaction), accepted_history)

    def test_real_normal_only_coupled_step_needs_no_dual_updates(self):
        with _example() as example:
            coupler, alm = _prepare(example)
            coupler.step(example.state_0, example.state_1, example.control, _STEP)
            self.assertEqual(alm.dual_updates, 0)
            self.assertLessEqual(alm.normal_error, alm.tolerance)
            self.assertTrue(alm.converged(coupler.absolute_tolerance, coupler.relative_tolerance))
            self.assertGreater(np.linalg.norm(alm.committed_normal.numpy()), 0)
            self.assertLess(example.interaction.force_balance_error, 1e-8)

    def test_small_hard_wall_preload_is_resolved_by_real_two_way_motion(self):
        # Narrow hard-wall acceptance case at the example's force tolerances.
        # Larger preload and tighter-force-budget failures are separately retained.
        with _example() as example:
            _, alm = _prepare(example)
            h = 0.002
            example.solver.refresh_tangent(example.state_0.particle_q, h)
            example.drive.configure_mujoco(example.rigid_solver, dt=h)
            alm.compliance_per_m = 0.0
            alm.penalty_per_m = 60000.0
            pose = example.plan.pose(0.5 * example.plan.exit_angle)
            pose[2] += 22e-6
            example._initialize_pose(pose)
            example.state_0.clear_forces()
            wrench = example.state_0.body_f.numpy()
            wrench[example.needle_body, 2] = 0.001
            example.state_0.body_f.assign(wrench)
            coupler = NeedleTissueCoupler(
                example.model,
                example.solver,
                example.rigid_solver,
                example.interaction,
                example.needle_body,
                example.drive.com,
                absolute_tolerance=2e-6,
                relative_tolerance=1e-3,
                max_iterations=160,
                shaft_preconditioner=True,
            )
            q0 = example.state_0.particle_q.numpy().copy()
            with (
                patch.object(example.interaction, "accept", wraps=example.interaction.accept) as accept,
                patch.object(alm, "commit", wraps=alm.commit) as commit,
            ):
                coupler.step(example.state_0, example.state_1, example.control, h)
            self.assertEqual((accept.call_count, commit.call_count), (1, 1))
            self.assertFalse(alm._open)
            self.assertGreater(alm.dual_updates, 1)
            self.assertLessEqual(coupler.iterations, 160)
            stats = coupler._statistics.numpy()
            allowed = coupler.absolute_tolerance + coupler.relative_tolerance * max(stats[1], stats[2])
            self.assertLessEqual(coupler.residual, allowed)
            self.assertTrue(alm.converged(coupler.absolute_tolerance, coupler.relative_tolerance))
            self.assertLessEqual(alm.normal_error, alm.tolerance)
            self.assertGreater(np.linalg.norm(alm.committed_normal.numpy()), 0)
            self.assertGreater(np.max(np.linalg.norm(example.state_1.particle_q.numpy() - q0, axis=1)), 0)
            fixed = example.fixed_vertex_indices
            np.testing.assert_array_equal(example.state_1.particle_q.numpy()[fixed], q0[fixed])
            self.assertLess(example.interaction.force_balance_error, 1e-8)
            self.assertLess(example.interaction.torque_balance_error, 1e-9)


if __name__ == "__main__":
    unittest.main()
