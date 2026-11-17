# SPDX-License-Identifier: Apache-2.0
"""Finite bearing compliance carries tissue while allowing physical withdrawal."""

import unittest

import numpy as np
import warp as wp
from test_needle_insertion_example import _example
from test_needle_tunnel_coupler import _prepare

from newton_solver_suturebot import NeedleTissueCoupler
from newton_solver_suturebot.examples.needle.example_fem_needle_insertion import Example


def _axis_distance(interaction, state, body):
    """Independent Euclidean distance to the transformed finite polyline."""
    engaged = interaction._track_engaged.numpy().astype(bool)
    engaged &= interaction._track_lengths.numpy() > 0
    if not np.any(engaged):
        return 0.0
    q = state.particle_q.numpy().astype(float)
    points = np.einsum(
        "ij,ijk->ik", interaction._track_bary.numpy().astype(float), q[interaction._track_nodes.numpy()]
    )[engaged]
    pose = state.body_q.numpy()[body]
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=float).reshape(3, 3)
    samples = interaction._samples.numpy().astype(float) @ rotation.T + pose[:3]
    directions = np.diff(samples, axis=0)
    offsets = points[:, None, :] - samples[None, :-1, :]
    fractions = np.einsum("ijk,jk->ij", offsets, directions) / np.einsum("ij,ij->i", directions, directions)
    distances = np.linalg.norm(offsets - np.clip(fractions, 0, 1)[..., None] * directions, axis=2)
    return float(np.max(np.min(distances, axis=1)))


def _determinants(q, tetrahedra):
    p = np.asarray(q, dtype=float)[tetrahedra]
    return np.linalg.det(np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2))


class NeedleTunnelLiftTests(unittest.TestCase):
    def run_case(self, compliance):
        with _example() as example:
            # Existing geometric probes seed an embedded track. Subsequent
            # motion is entirely real FEM/native MuJoCo under a finite drive;
            # this fixture is not a dynamic-puncture or full-pad benchmark.
            _, alm = _prepare(example, friction=0.15)
            alm.compliance_per_m = compliance
            alm.penalty_per_m = 60000.0
            h = 0.002
            example.solver.linear_solver_tolerance = 1e-5
            example.solver.linear_solver_maxiter = 160
            example.solver.refresh_tangent(example.state_0.particle_q, h)
            example.drive.configure_mujoco(example.rigid_solver, dt=h)
            angle = 0.5 * example.plan.exit_angle
            base = example.plan.pose(angle)
            initial = base.copy()
            initial[2] += 22e-6
            example._initialize_pose(initial)
            example.control.joint_f.zero_()
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
            interaction = example.interaction
            q0 = example.state_0.particle_q.numpy().copy()
            fixed = example.fixed_vertex_indices
            reference_J = _determinants(q0, interaction.tetrahedra)
            maximum_separation = maximum_deformation = maximum_reaction = 0.0
            phases = []
            for height in np.r_[np.linspace(22e-6, 0.0005, 6)[1:], np.full(5, 0.0005), np.linspace(0.0005, 0, 6)[1:]]:
                target = base.copy()
                target[2] += height
                phases.append(("loaded", target))
            count = int(np.ceil(abs(angle - example.plan.initial_angle) * example.geometry.radius / 5e-5))
            phases.extend(
                ("withdraw", example.plan.pose(a))
                for a in np.linspace(angle, example.plan.initial_angle, count + 1)[1:]
            )
            phases.extend(("outside", example.plan.pose(example.plan.initial_angle)) for _ in range(20))
            initial_track_count = interaction.track_count
            for phase, target in phases:
                example.solver.refresh_tangent(example.state_0.particle_q, h)
                example.state_0.clear_forces()
                example.drive.add_forces(example.state_0, target, np.zeros(6))
                coupler.step(example.state_0, example.state_1, example.control, h)
                q = example.state_1.particle_q.numpy()
                self.assertTrue(np.isfinite(q).all())
                self.assertTrue(np.isfinite(example.state_1.body_q.numpy()).all())
                np.testing.assert_array_equal(q[fixed], q0[fixed])
                self.assertGreater(np.min(_determinants(q, interaction.tetrahedra) / reference_J), 0)
                self.assertFalse(alm._open)
                self.assertTrue(alm.converged(coupler.absolute_tolerance, coupler.relative_tolerance))
                statistics = coupler._statistics.numpy()
                allowed = coupler.absolute_tolerance + coupler.relative_tolerance * max(statistics[1:3])
                self.assertLessEqual(coupler.residual, allowed)
                self.assertLess(interaction.force_balance_error, 2e-8)
                self.assertLess(interaction.torque_balance_error, 1e-9)
                if phase == "loaded":
                    self.assertGreater(interaction.engaged_track_length, 0)
                    maximum_separation = max(
                        maximum_separation, _axis_distance(interaction, example.state_1, example.needle_body)
                    )
                    maximum_deformation = max(
                        maximum_deformation, float(np.linalg.norm(q.astype(float) - q0, axis=1).max())
                    )
                    maximum_reaction = max(maximum_reaction, float(np.linalg.norm(interaction.interface_force)))
                elif phase == "outside":
                    self.assertEqual(interaction.engaged_track_length, 0)
                    np.testing.assert_array_equal(interaction.interface_force, 0)
                example.state_0, example.state_1 = example.state_1, example.state_0
            self.assertEqual((interaction.puncture_count, interaction.entry_count, interaction.exit_count), (1, 1, 1))
            self.assertEqual(interaction.track_count, initial_track_count)
            self.assertEqual(interaction.status, "outside")
            return maximum_separation, maximum_deformation, maximum_reaction, float(interaction.radii.max())

    def test_example_bearing_compliance_reduces_separation_without_locking_withdrawal(self):
        selected = Example.create_parser().parse_args([]).tunnel_compliance
        self.assertIsNotNone(selected)
        self.assertGreater(selected, 0)
        self.assertLess(selected, 1 / 60000)
        original = self.run_case(1 / 60000)
        improved = self.run_case(selected)
        self.assertLess(improved[0], 0.5 * original[0])
        # This finite-compliance load case stays close to the needle axis;
        # it is not a universal hard-wall or zero-overlap guarantee.
        self.assertLess(improved[0], improved[3] / 4)
        for separation, deformation, reaction, _ in (original, improved):
            self.assertGreater(separation, 0)
            self.assertGreater(deformation, 1e-6)
            self.assertGreater(reaction, 1e-3)
        self.assertGreater(improved[1], original[1])


if __name__ == "__main__":
    unittest.main()
