# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Inexact structural solve regression with an independently known contact root."""

import unittest
from unittest.mock import patch

import newton
import numpy as np
import warp as wp
from test_needle_insertion_example import _example
from test_needle_tissue_coupler import _native_state

from newton_solver_suturebot import NeedleTissueCoupler, NeedleTissueInteraction, SolverFEM


class InexactTranslation(SolverFEM):
    """One mass mode whose bounded inner residual alternates within tolerance.

    The perturbation is at most eta*||hF|| for the exact mass equation.
    This deliberately exercises an inexact solve floor, not a physical force.
    """

    def __init__(self, model, tolerance):
        self.model = model
        self.mass = 1e-5
        self.linear_solver_tolerance = tolerance
        self.dv = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.seen = []

    def step(self, state_in, state_out, control, contacts, dt, *, external_forces=None):
        self.seen.append(self.linear_solver_tolerance)
        force = external_forces.numpy().sum(axis=0).astype(float)
        exact = dt * force / self.mass
        perturbation = (-1 if len(self.seen) % 2 else 1) * self.linear_solver_tolerance * np.linalg.norm(exact)
        increment = exact.copy()
        increment[2] += perturbation
        self.dv.assign(np.broadcast_to(increment, (self.model.particle_count, 3)).copy())
        velocity = state_in.particle_qd.numpy().astype(float) + increment
        state_out.particle_qd.assign(velocity.astype(np.float32))
        state_out.particle_q.assign(state_in.particle_q.numpy() + dt * velocity)


def prepare(ex, tolerance, budget):
    ex.model.joint_damping.zero_()
    ex.rigid_solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
    radius, stiffness = 0.0005, 20.0
    body, com = ex.needle_body, ex.drive.com
    ex._initialize_pose(np.r_[np.array([0.0, 0.0, radius]) - com, 0.0, 0.0, 0.0, 1.0])
    velocity = ex.state_0.body_qd.numpy()
    velocity[body, 2] = -0.003
    ex.state_0.body_qd.assign(velocity)
    newton.eval_ik(ex.model, ex.state_0, ex.state_0.joint_q, ex.state_0.joint_qd)
    samples = np.asarray([com, com + [0, 0, 0.01]])
    interaction = NeedleTissueInteraction(
        ex.selection.vertices,
        ex.selection.indices,
        ex.selection.exterior_surface_triangles,
        [],
        samples,
        radius,
        samples[0],
        device=ex.device,
        stiffness=stiffness,
        damping=0.0,
        puncture_force=1.0,
        friction=0.0,
    )
    structural = InexactTranslation(ex.model, tolerance)
    coupler = NeedleTissueCoupler(
        ex.model,
        structural,
        ex.rigid_solver,
        interaction,
        body,
        com,
        absolute_tolerance=1e-9,
        relative_tolerance=0.0,
        max_iterations=budget,
        relaxation=0.3,
    )
    wrench = np.zeros((ex.model.body_count, 6), np.float32)
    wrench[body, 2] = -0.0002
    ex.state_0.body_f.assign(wrench)
    nodal = np.zeros((ex.model.particle_count, 3), np.float32)
    nodal[:, 2] = 0.0003 / ex.model.particle_count
    return structural, coupler, interaction, wp.array(nodal, dtype=wp.vec3, device=ex.device)


class AdaptiveInnerTests(unittest.TestCase):
    def test_tightening_matches_analytic_contact_and_fixed_tight_reference(self):
        states = []
        for configured in (0.003, 1e-6):
            with _example() as ex:
                structural, coupler, interaction, nodal = prepare(ex, configured, 40)
                h = 0.0005
                with patch.object(interaction, "accept", wraps=interaction.accept) as accept:
                    coupler.step(ex.state_0, ex.state_1, ex.control, h, external_forces=nodal)
                    self.assertEqual(accept.call_count, 1)
                rigid_mass = float(ex.model.body_mass.numpy()[ex.needle_body])
                free_speed = -0.003 + h * (-0.0002 / rigid_mass - 0.0003 / structural.mass)
                expected = -h * free_speed * 20 / (1 + h * h * 20 * (1 / rigid_mass + 1 / structural.mass))
                self.assertAlmostEqual(interaction.last_tip_force, expected, delta=2e-9)
                self.assertLessEqual(coupler.residual, 1e-9)
                self.assertEqual(structural.linear_solver_tolerance, configured)
                self.assertLessEqual(coupler.iterations, 40)
                if configured == 0.003:
                    self.assertGreater(coupler.iterations, 8)
                    self.assertLess(min(structural.seen), configured)
                    self.assertEqual(structural.seen[:8], [configured] * 8)
                self.assertGreaterEqual(min(structural.seen), min(configured, 1e-6))
                self.assertEqual(
                    (interaction.puncture_count, interaction.entry_count, interaction.track_count), (0, 0, 0)
                )
                states.append(
                    (
                        ex.state_1.particle_q.numpy(),
                        ex.state_1.particle_qd.numpy(),
                        ex.state_1.body_q.numpy(),
                        ex.state_1.body_qd.numpy(),
                    )
                )
        for a, b in zip(*states, strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-7)

    def test_failed_tightened_attempt_restores_tolerance_state_native_and_warm_start(self):
        with _example() as ex:
            structural, coupler, interaction, nodal = prepare(ex, 0.003, 9)
            native = _native_state(ex.rigid_solver)
            names = ("particle_q", "particle_qd", "particle_f", "body_q", "body_qd", "body_f")
            before = {name: getattr(ex.state_0, name).numpy().copy() for name in names}
            warm = structural.velocity_increment.numpy().copy()
            with patch.object(interaction, "accept", wraps=interaction.accept) as accept:
                with self.assertRaisesRegex(RuntimeError, "did not converge"):
                    coupler.step(ex.state_0, ex.state_1, ex.control, 0.0005, external_forces=nodal)
                self.assertEqual(accept.call_count, 0)
            self.assertEqual(structural.linear_solver_tolerance, 0.003)
            self.assertLess(min(structural.seen), 0.003)
            np.testing.assert_array_equal(_native_state(ex.rigid_solver), native)
            np.testing.assert_array_equal(structural.velocity_increment.numpy(), warm)
            for name in names:
                np.testing.assert_array_equal(getattr(ex.state_0, name).numpy(), before[name])
            self.assertEqual((interaction.puncture_count, interaction.entry_count, interaction.track_count), (0, 0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
