# SPDX-License-Identifier: Apache-2.0
"""Independent clearance, compliance, Coulomb and material-history checks."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleCouplingNotConverged, NeedleTissueInteraction, boundary_faces


class NeedleTunnelALMTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.vertices = np.array(((0, 0, 0), (0.01, 0, 0), (0, 0.01, 0), (0, 0, 0.01)))
        self.tets = np.array(((0, 1, 2, 3),), np.int32)
        self.com = np.array((0.0003, -0.0002, -0.0001))

    def build(self, *, fixed=(), compliance=1 / 60000, penalty=60000, clearance=2e-5):
        samples = np.zeros((21, 3))
        samples[:, 2] = np.linspace(0, -0.01, len(samples))
        interaction = NeedleTissueInteraction(
            self.vertices,
            self.tets,
            boundary_faces(self.tets)[0],
            fixed,
            samples,
            0.0005,
            samples[0],
            device=self.device,
            damping=0,
            shaft_damping_per_m=0,
            friction=0.2,
            track_spacing=0.0001,
            tunnel_model="alm",
            tunnel_compliance_per_m=compliance,
            tunnel_penalty_per_m=penalty,
            tunnel_clearance=clearance,
            tunnel_tolerance=1e-9,
        )
        state = SimpleNamespace(
            particle_q=wp.array(self.vertices, dtype=wp.vec3, device=self.device),
            particle_qd=wp.zeros(len(self.vertices), dtype=wp.vec3, device=self.device),
            body_q=wp.array([[0.002, 0.002, -0.0001, 0, 0, 0, 1]], dtype=wp.transform, device=self.device),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
        )
        forces = wp.zeros(len(self.vertices), dtype=wp.vec3, device=self.device)
        return interaction, state, forces

    @staticmethod
    def live(interaction, array):
        """The rows of a per-track array that hold track samples; the rest of the reserve holds zeros."""
        return array.numpy()[: interaction.track_count]

    def pose(self, state, *, z=None, offset=0, speed=0):
        pose = state.body_q.numpy()
        pose[0, 0] = 0.002 + offset
        if z is not None:
            pose[0, 2] = z
        state.body_q.assign(pose)
        state.body_qd.assign(np.array([[0, 0, speed, 0, 0, 0]], np.float32))

    def accepted_probe(self, interaction, state, forces):
        """Prescribed accepted geometry seeds history, not a dynamic insertion."""
        alm = interaction.tunnel_alm
        alm.begin_step()
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, self.com, forces, 0.001, implicit=True)
        self.assertTrue(alm.converged(1e-9, 1e-6))
        interaction.accept(state, 0)
        self.assertNotEqual(interaction.status, "failed", interaction.reason)
        alm.commit()

    def seed(self, interaction, state, forces, end=0.00045):
        self.accepted_probe(interaction, state, forces)
        self.assertEqual(interaction.puncture_count, 1)
        for z in np.arange(-0.00005, end + 1e-8, 0.00005):
            self.pose(state, z=z)
            self.accepted_probe(interaction, state, forces)
        self.assertGreater(interaction.track_count, 1)
        self.assertGreater(interaction.track_length, 0)

    def evaluate(self, interaction, state, forces, dt=0.001):
        state.body_f.zero_()
        forces.zero_()
        interaction._diagnostics.zero_()
        interaction.tunnel_alm.add_forces(state, 0, self.com, forces, dt)

    def equilibrate(self, interaction, state, forces, *, begin=True):
        alm = interaction.tunnel_alm
        if begin:
            alm.begin_step()
        for _ in range(400):
            self.evaluate(interaction, state, forces)
            if alm.converged(1e-9, 1e-6):
                return
            alm.update_duals()
        self.fail("Prescribed compliant tunnel failed to reach its constitutive equilibrium")

    def test_clearance_is_free_and_compliance_equilibrium_is_penalty_independent(self):
        results = []
        for penalty in (6000, 60000, 600000):
            with self.subTest(penalty=penalty):
                interaction, state, forces = self.build(penalty=penalty)
                self.seed(interaction, state, forces)
                alm = interaction.tunnel_alm
                self.pose(state, offset=1e-5)
                self.equilibrate(interaction, state, forces)
                np.testing.assert_array_equal(state.body_f.numpy(), 0)
                np.testing.assert_array_equal(forces.numpy(), 0)
                alm.commit()
                self.pose(state, offset=8e-5)
                self.equilibrate(interaction, state, forces)
                density = self.live(interaction, alm.proposed_normal)
                actual_offset = float(state.body_q.numpy()[0, 0] - np.float32(0.002))
                expected = (actual_offset - alm.clearance) / alm.compliance_per_m
                np.testing.assert_allclose(density[:, 0], expected, rtol=4e-5, atol=2e-5)
                np.testing.assert_allclose(density[:, 1:], 0, atol=2e-5)
                force = state.body_f.numpy()[0, :3]
                self.assertLess(force[0], 0)
                np.testing.assert_allclose(force[0], -expected * interaction.track_length, rtol=4e-5, atol=2e-8)
                results.append(force.copy())
                alm.commit()
                # An accepted previous reaction must also disappear on return
                # inside the clearance; it is not a permanent radial spring.
                self.pose(state, offset=1e-5)
                self.equilibrate(interaction, state, forces)
                np.testing.assert_allclose(state.body_f.numpy(), 0, atol=2e-8)
                alm.commit()
        np.testing.assert_allclose(results, np.broadcast_to(results[0], (3, 3)), rtol=4e-5, atol=2e-8)

    def test_barycentric_force_and_origin_moment_include_fixed_reaction(self):
        interaction, state, forces = self.build(fixed=(3,))
        self.seed(interaction, state, forces)
        self.pose(state, offset=8e-5)
        self.equilibrate(interaction, state, forces)
        q = state.particle_q.numpy().astype(float)
        nodal = forces.numpy().astype(float)
        wrench = state.body_f.numpy()[0].astype(float)
        center = state.body_q.numpy()[0, :3] + self.com
        np.testing.assert_allclose(wrench[:3] + nodal.sum(axis=0), 0, atol=3e-9)
        np.testing.assert_allclose(
            wrench[3:] + np.cross(center, wrench[:3]) + np.cross(q, nodal).sum(axis=0), 0, atol=2e-11
        )
        self.assertGreater(np.linalg.norm(nodal[3]), 0)
        expected = np.zeros_like(nodal)
        for ids, weights, length, density in zip(
            self.live(interaction, interaction._track_nodes),
            self.live(interaction, interaction._track_bary),
            self.live(interaction, interaction._track_lengths),
            self.live(interaction, interaction.tunnel_alm.proposed_normal),
            strict=True,
        ):
            for node, weight in zip(ids, weights, strict=True):
                expected[node] += weight * length * density
        np.testing.assert_allclose(nodal, expected, rtol=2e-6, atol=2e-9)
        interaction.tunnel_alm.discard()

    def test_coulomb_static_traction_slip_and_reversal(self):
        interaction, state, forces = self.build()
        self.seed(interaction, state, forces)
        self.pose(state, offset=8e-5)
        self.equilibrate(interaction, state, forces)
        alm = interaction.tunnel_alm
        alm.commit()
        # A valid accepted sticking traction can persist without velocity.
        # Unlike viscous drag, zero slip alone must not erase this history.
        bound = interaction.friction * np.linalg.norm(self.live(interaction, alm.committed_normal), axis=1)
        alm.committed_tangent.assign(0.4 * bound)
        self.equilibrate(interaction, state, forces)
        np.testing.assert_allclose(self.live(interaction, alm.proposed_tangent), 0.4 * bound, rtol=2e-6, atol=1e-7)
        self.assertGreater(abs(float(state.body_f.numpy()[0, 2])), 0)
        alm.commit()
        for speed in (0.01, -0.01):
            self.pose(state, offset=8e-5, speed=speed)
            self.equilibrate(interaction, state, forces)
            density = self.live(interaction, alm.proposed_tangent)
            bound = interaction.friction * np.linalg.norm(self.live(interaction, alm.proposed_normal), axis=1)
            np.testing.assert_allclose(np.abs(density), bound, rtol=2e-6, atol=1e-7)
            self.assertLess(float(state.body_f.numpy()[0, 2]) * speed, 0)
            alm.commit()

    def test_trial_discard_preserves_duals_and_withdrawal_clears_them(self):
        interaction, state, forces = self.build()
        self.seed(interaction, state, forces)
        self.pose(state, offset=8e-5)
        self.equilibrate(interaction, state, forces)
        alm = interaction.tunnel_alm
        alm.commit()
        committed = alm.committed_normal.numpy().copy()
        history = (interaction.track_count, interaction.events)
        self.pose(state, offset=0.00012)
        alm.begin_step()
        self.evaluate(interaction, state, forces)
        alm.update_duals()
        self.assertGreater(np.linalg.norm(alm.normal.numpy() - committed), 0)
        np.testing.assert_array_equal(alm.committed_normal.numpy(), committed)
        self.assertEqual((interaction.track_count, interaction.events), history)
        alm.discard()
        np.testing.assert_array_equal(alm.normal.numpy(), committed)
        np.testing.assert_array_equal(alm.proposed_normal.numpy(), committed)
        # Return centered before withdrawing, then accept sufficiently small
        # actual geometric intervals through the original opening.
        self.pose(state, offset=0)
        self.equilibrate(interaction, state, forces)
        alm.commit()
        for z in np.arange(0.0004, -0.000151, -0.00005):
            self.pose(state, z=z)
            self.accepted_probe(interaction, state, forces)
        self.assertEqual(interaction.status, "outside")
        self.assertEqual(interaction.engaged_track_length, 0)
        np.testing.assert_array_equal(alm.committed_normal.numpy(), 0)
        np.testing.assert_array_equal(alm.committed_tangent.numpy(), 0)
        count = interaction.track_count
        for z in np.arange(-0.0001, 0.000451, 0.00005):
            self.pose(state, z=z)
            self.accepted_probe(interaction, state, forces)
        self.assertEqual(interaction.track_count, count)
        self.assertEqual(interaction.puncture_count, 1)
        np.testing.assert_array_equal(alm.committed_normal.numpy(), 0)
        np.testing.assert_array_equal(alm.committed_tangent.numpy(), 0)

    def test_capture_at_an_empty_track_needs_a_bound_projection_and_projects_later_samples(self):
        if not self.device.is_cuda:
            self.skipTest("The cooperative projection is bound and captured on CUDA")
        from newton_solver_suturebot._src.solvers.coupled.interface.shaft_projection import projection_binding
        from newton_solver_suturebot._src.utils.capture import ScopedCapture

        interaction, state, forces = self.build()
        interaction.cooperative_shaft_projection = True
        alm = interaction.tunnel_alm
        radius = float(interaction.radii.max())
        interaction.step_time.set(0.001)
        issued = []
        launch = wp.launch

        def record(kernel, *arguments, **options):
            issued.append(kernel.key)
            return launch(kernel, *arguments, **options)

        def capture():
            alm.begin_step()
            try:
                with ScopedCapture(device=self.device) as scope:
                    self.evaluate(interaction, state, forces)
            finally:
                alm.discard()
            return scope.graph

        # The cooperative projection is on and nothing is bound. The graph of this capture would hold
        # the original tunnel kernel, and no later sample captures it again: the capture is refused.
        self.assertEqual(interaction.track_count, 0)
        with self.assertRaisesRegex(RuntimeError, "no projection is bound"):
            capture()
        self.assertFalse(alm._open)
        # Bound outside the capture, the graph of the empty track holds the projected kernel.
        binding = projection_binding(interaction, state, 0, radius)
        self.assertIsNotNone(binding)
        with patch.object(wp, "launch", side_effect=record):
            graph = capture()
        self.assertEqual(issued, ["prepare_queries", "cooperative_projection", "tunnel_forces_projected"])
        np.testing.assert_array_equal(binding.counts.numpy(), [0, 0])
        # Accepted steps of the host path lay the track; they use the same binding.
        self.seed(interaction, state, forces)
        count = interaction.track_count
        self.assertIs(interaction._cooperative_shaft_binding, binding)
        self.assertEqual(interaction._cooperative_shaft_rebinds, 0)
        self.pose(state, offset=3e-5)
        binding.counts.zero_()
        # The graph captured at the empty track projects the samples that are live now ...
        alm.begin_step()
        wp.capture_launch(graph)
        replayed = (
            forces.numpy().copy(),
            state.body_f.numpy().copy(),
            self.live(interaction, alm.proposed_normal).copy(),
            self.live(interaction, alm.proposed_tangent).copy(),
        )
        alm.discard()
        np.testing.assert_array_equal(binding.counts.numpy(), [count, 0])
        # ... and gives the forces and proposals of the original tunnel kernel.
        interaction.cooperative_shaft_projection = False
        alm.begin_step()
        del issued[:]
        with patch.object(wp, "launch", side_effect=record):
            self.evaluate(interaction, state, forces)
        self.assertEqual(issued, ["_tunnel_forces"])
        original = (
            forces.numpy().copy(),
            state.body_f.numpy().copy(),
            self.live(interaction, alm.proposed_normal).copy(),
            self.live(interaction, alm.proposed_tangent).copy(),
        )
        alm.discard()
        self.assertGreater(np.abs(original[2]).max(), 0.1)
        names = ("particle_f", "body_f", "proposed_normal", "proposed_tangent")
        for name, new, old in zip(names, replayed, original, strict=True):
            np.testing.assert_allclose(new, old, rtol=1e-4, atol=1e-4 * np.abs(old).max(), err_msg=name)

    def test_hard_wall_does_not_treat_geometric_overlap_as_converged(self):
        interaction, state, forces = self.build(compliance=0)
        self.seed(interaction, state, forces)
        alm = interaction.tunnel_alm
        self.pose(state, offset=8e-5)
        alm.begin_step()
        for _ in range(3):
            self.evaluate(interaction, state, forces)
            self.assertFalse(alm.converged(1e-9, 1e-6))
            self.assertGreater(alm.normal_error, 5e-5)
            alm.update_duals()
        np.testing.assert_array_equal(alm.committed_normal.numpy(), 0)
        alm.discard()
        self.pose(state, offset=1e-5)
        self.equilibrate(interaction, state, forces)
        np.testing.assert_array_equal(state.body_f.numpy(), 0)
        alm.commit()

    def test_appended_anchor_starts_zero_without_rescaling_existing_density(self):
        interaction, state, forces = self.build()
        self.seed(interaction, state, forces)
        self.pose(state, offset=8e-5)
        self.equilibrate(interaction, state, forces)
        interaction.accept(state, 0)
        self.assertNotEqual(interaction.status, "failed", interaction.reason)
        alm = interaction.tunnel_alm
        alm.commit()
        count = interaction.track_count
        previous = self.live(interaction, alm.committed_normal).copy()
        self.assertGreater(np.linalg.norm(previous), 0)
        self.pose(state, z=0.000525, offset=8e-5)
        self.equilibrate(interaction, state, forces)
        interaction.accept(state, 0)
        self.assertNotEqual(interaction.status, "failed", interaction.reason)
        self.assertGreater(interaction.track_count, count)
        alm.commit()
        np.testing.assert_allclose(alm.committed_normal.numpy()[:count], previous, rtol=2e-6, atol=2e-5)
        np.testing.assert_array_equal(alm.committed_normal.numpy()[count:], 0)
        np.testing.assert_array_equal(alm.committed_tangent.numpy()[count:], 0)

    def test_small_local_residuals_cannot_hide_large_accumulated_reaction_change(self):
        # Finite normal reaction is exact, but tangential ALM still needs a
        # sum over all anchors; testing only its largest entry is insufficient.
        interaction, state, forces = self.build()
        self.seed(interaction, state, forces)
        alm = interaction.tunnel_alm
        self.pose(state, offset=8e-5, speed=0.001)
        alm.begin_step()
        self.evaluate(interaction, state, forces)
        changes = self.live(interaction, interaction._track_lengths) * np.abs(
            self.live(interaction, alm.proposed_tangent)
        )
        threshold = 0.5 * (changes.max() + changes.sum())
        self.assertTrue(np.all(changes < threshold))
        self.assertGreater(changes.sum(), threshold)
        self.assertFalse(alm.converged(float(threshold), 0))
        self.assertLess(alm.normal_error, alm.tolerance)
        self.assertGreater(alm.dual_residual, threshold)
        alm.discard()

    def test_new_passage_has_zero_duals_and_no_track_quadrature_in_gap(self):
        self.vertices = np.vstack((self.vertices, self.vertices + (0, 0, 0.008)))
        self.tets = np.array(((0, 1, 2, 3), (4, 5, 6, 7)), np.int32)
        interaction, state, forces = self.build()
        self.seed(interaction, state, forces)
        for z in np.arange(0.0005, 0.008701, 0.00005):
            self.pose(state, z=z)
            self.accepted_probe(interaction, state, forces)
        self.assertEqual(interaction.puncture_count, 2)
        self.assertEqual(interaction.entry_count, 2)
        self.assertEqual(interaction.exit_count, 1)
        self.assertEqual(interaction.track_section_count, 2)
        points = np.einsum(
            "ij,ijk->ik",
            self.live(interaction, interaction._track_bary),
            self.vertices[self.live(interaction, interaction._track_nodes)],
        )
        self.assertFalse(np.any((points[:, 2] > 0.006001) & (points[:, 2] < 0.007999)))
        sections = interaction.track_section_ids
        for first, second in interaction._track_edges:
            self.assertEqual(sections[first], sections[second])
        # A centered shaft creates no transverse reaction. Newly appended
        # anchors and the separated second passage cannot inherit old values.
        np.testing.assert_array_equal(interaction.tunnel_alm.committed_normal.numpy(), 0)
        np.testing.assert_array_equal(interaction.tunnel_alm.committed_tangent.numpy(), 0)

    def test_rejected_coupled_step_preserves_committed_tunnel_and_output(self):
        from test_needle_insertion_example import _example  # noqa: PLC0415

        with _example(tunnel_model="alm") as example:
            example.select_point_xy("entry", -0.004, 0)
            example.select_point_xy("exit", 0.004, 0)
            example.preview()
            example.start()
            interaction, alm = example.interaction, example.interaction.tunnel_alm
            self.assertIsNotNone(alm)
            final_angle = 0.5 * example.plan.exit_angle
            count = int(np.ceil((final_angle - example.plan.initial_angle) * example.geometry.radius / 0.00005))
            # Public geometric acceptance creates an embedded initial history;
            # this setup is not claimed as a dynamic insertion trajectory.
            for angle in np.linspace(example.plan.initial_angle, final_angle, count + 1):
                example._initialize_pose(example.plan.pose(angle))
                alm.begin_step()
                example.state_0.clear_forces()
                example._external_forces.zero_()
                interaction.add_forces(
                    example.state_0,
                    example.needle_body,
                    example.drive.com,
                    example._external_forces,
                    example.sim_dt,
                    implicit=True,
                )
                self.assertTrue(alm.converged(2e-6, 1e-3))
                interaction.accept(example.state_0, example.needle_body)
                self.assertNotEqual(interaction.status, "failed", interaction.reason)
                alm.commit()
            self.assertGreater(interaction.track_count, 1)
            pose = example.plan.pose(final_angle)
            pose[2] += 0.00008
            example._initialize_pose(pose)
            alm.begin_step()
            for _ in range(100):
                example.state_0.clear_forces()
                example._external_forces.zero_()
                interaction.add_forces(
                    example.state_0,
                    example.needle_body,
                    example.drive.com,
                    example._external_forces,
                    example.sim_dt,
                    implicit=True,
                )
                if alm.converged(1e-9, 1e-6):
                    break
                alm.update_duals()
            else:
                self.fail("Failed to initialize a nonzero committed tunnel reaction")
            interaction.accept(example.state_0, example.needle_body)
            self.assertNotEqual(interaction.status, "failed", interaction.reason)
            alm.commit()
            normal = alm.committed_normal.numpy().copy()
            tangent = alm.committed_tangent.numpy().copy()
            self.assertGreater(np.linalg.norm(normal), 0)
            history = (interaction.track_count, interaction.events, interaction.status)
            names = ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")
            output = {name: getattr(example.state_1, name).numpy().copy() for name in names}
            example.state_0.clear_forces()
            example.coupler.max_iterations = 1
            example.coupler.absolute_tolerance = 1e-20
            example.coupler.relative_tolerance = 0
            with self.assertRaises(NeedleCouplingNotConverged):
                example.coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
            np.testing.assert_array_equal(alm.committed_normal.numpy(), normal)
            np.testing.assert_array_equal(alm.committed_tangent.numpy(), tangent)
            np.testing.assert_array_equal(alm.normal.numpy(), normal)
            np.testing.assert_array_equal(alm.tangent.numpy(), tangent)
            self.assertEqual((interaction.track_count, interaction.events, interaction.status), history)
            for name, expected in output.items():
                np.testing.assert_array_equal(getattr(example.state_1, name).numpy(), expected)
            alm.begin_step()  # rejection closed the transaction
            alm.discard()

    def test_finite_normal_is_exact_in_one_evaluation_independent_of_warm_dual_and_rho(self):
        for penalty in (6000, 60000, 600000):
            for offset in (0.0, 1e-5, 8e-5, -8e-5):
                with self.subTest(penalty=penalty, offset=offset):
                    interaction, state, forces = self.build(penalty=penalty)
                    interaction.friction = 0
                    self.seed(interaction, state, forces)
                    alm = interaction.tunnel_alm
                    old = np.broadcast_to([7.0, -4.0, 0.0], (interaction.track_count, 3)).copy()
                    alm.committed_normal.assign(old)
                    self.pose(state, offset=offset)
                    alm.begin_step()
                    self.evaluate(interaction, state, forces)
                    actual = float(state.body_q.numpy()[0, 0] - np.float32(0.002))
                    expected = np.sign(actual) * max(0, abs(actual) - alm.clearance) / alm.compliance_per_m
                    np.testing.assert_allclose(
                        self.live(interaction, alm.proposed_normal)[:, 0], expected, atol=2e-5, rtol=4e-5
                    )
                    self.assertTrue(alm.converged(1e-9, 1e-6))
                    self.assertEqual(alm.dual_updates, 0)
                    self.assertLessEqual(alm.dual_residual, 1e-9)
                    np.testing.assert_array_equal(self.live(interaction, alm.committed_normal), old)
                    alm.discard()

    def test_hard_normal_first_proposal_remains_the_original_augmented_projection(self):
        interaction, state, forces = self.build(compliance=0)
        interaction.friction = 0
        self.seed(interaction, state, forces)
        alm = interaction.tunnel_alm
        old = np.broadcast_to([2.0, 1.0, 3.0], (interaction.track_count, 3)).copy()
        alm.committed_normal.assign(old)
        self.pose(state, offset=8e-5)
        alm.begin_step()
        self.evaluate(interaction, state, forces)
        actual = float(state.body_q.numpy()[0, 0] - np.float32(0.002))
        z = np.array([2.0 + alm.penalty_per_m * actual, 1.0, 0.0])
        expected = max(0, 1 - alm.penalty_per_m * alm.clearance / np.linalg.norm(z)) * z
        np.testing.assert_allclose(
            self.live(interaction, alm.proposed_normal), np.broadcast_to(expected, old.shape), atol=3e-5, rtol=3e-6
        )
        self.assertFalse(alm.converged(1e-9, 1e-6))
        alm.discard()


if __name__ == "__main__":
    unittest.main()
