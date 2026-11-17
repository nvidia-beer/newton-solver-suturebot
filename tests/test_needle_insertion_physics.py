# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Conservation and discrete event checks for the illustrative insertion law."""

import os
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleTissueInteraction, boundary_faces
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import _shaft_projection


@wp.kernel(enable_backward=False)
def _probe_tangents(points: wp.array[wp.vec3], samples: wp.array[wp.vec3], radius: float, tangents: wp.array[wp.vec3]):
    i = wp.tid()
    closest, tangent, distance, covered = _shaft_projection(points[i], wp.transform_identity(), samples, radius)
    tangents[i] = tangent


class TestNeedleInsertionPhysics(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], dtype=float)
        self.tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
        self.triangles = boundary_faces(self.tets)[0]

    def build(self, *, fixed=(), sample_count=11, **kwargs):
        samples = np.zeros((sample_count, 3))
        samples[:, 2] = np.linspace(0, -0.01, sample_count)
        interaction = NeedleTissueInteraction(
            self.vertices,
            self.tets,
            self.triangles,
            fixed,
            samples,
            0.0005,
            samples[0],
            device=self.device,
            **kwargs,
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

    def pose(self, state, z, *, x=0.002, velocity=(0, 0, 0)):
        state.body_q.assign(np.array([[x, 0.002, z, 0, 0, 0, 1]], np.float32))
        state.body_qd.assign(np.array([[*velocity, 0, 0, 0]], np.float32))

    def forces(self, interaction, state, forces):
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005)

    def accept_to(self, interaction, state, z, *, x=0.002):
        current = state.body_q.numpy()[0, :3].astype(float)
        target = np.array([x, 0.002, z])
        step = min(float(interaction.radii.min()), interaction.track_spacing) / 2
        count = max(1, int(np.ceil(np.linalg.norm(target - current) / step)))
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            position = current + fraction * (target - current)
            self.pose(state, position[2], x=position[0])
            interaction.accept(state, 0)
            if interaction.status in ("failed", "emerged"):
                break

    def insert(self, interaction, state, forces):
        self.forces(interaction, state, forces)
        interaction.accept(state, 0)
        self.assertTrue(interaction.punctured)
        self.assertEqual(interaction.track_count, 0)
        self.accept_to(interaction, state, 0.0003)
        self.assertEqual(interaction.status, "inserted")

    def test_surface_force_and_torque_balance_and_actual_threshold_force(self):
        interaction, state, forces = self.build()
        body_com = np.array([0.001, 0, 0])
        interaction.add_forces(state, 0, body_com, forces, 0.0005)
        body_force = state.body_f.numpy()[0]
        np.testing.assert_allclose(body_force[:3] + forces.numpy().sum(axis=0), 0, atol=1e-8)
        self.assertGreater(np.linalg.norm(body_force[3:]), 1e-6)
        origin_torque = body_force[3:] + np.cross(state.body_q.numpy()[0, :3] + body_com, body_force[:3])
        tissue_torque = np.cross(self.vertices, forces.numpy()).sum(axis=0)
        np.testing.assert_allclose(origin_torque + tissue_torque, 0, atol=1e-10)
        self.assertAlmostEqual(interaction.last_tip_force, 80 * 0.0004, places=7)
        self.assertEqual(interaction.puncture_count, 0)  # force evaluation does not commit history
        self.assertEqual(interaction.track_count, 0)
        interaction.accept(state, 0)
        self.assertEqual(interaction.puncture_count, 1)
        self.assertEqual(interaction.track_count, 0)  # no internal path before the actual tip enters

    def test_below_threshold_stays_contact_and_unapproved_crossing_fails(self):
        interaction, state, forces = self.build(puncture_force=0.1)
        self.forces(interaction, state, forces)
        interaction.accept(state, 0)
        self.assertEqual(interaction.status, "contact")
        self.assertFalse(interaction.punctured)
        self.pose(state, 0.0001)
        interaction.accept(state, 0)
        self.assertEqual(interaction.status, "failed")
        self.assertIn("before puncture", interaction.reason)

    def test_puncture_threshold_is_independent_of_shaft_sampling(self):
        values = []
        for count in (11, 21, 41):
            interaction, state, forces = self.build(sample_count=count)
            self.forces(interaction, state, forces)
            values.append(interaction.last_tip_force)
        np.testing.assert_allclose(values, values[0], atol=1e-9)

    def test_shaft_slides_without_tangential_spring_and_friction_dissipates(self):
        interaction, state, forces = self.build(damping=0, shaft_damping_per_m=0)
        self.insert(interaction, state, forces)
        self.accept_to(interaction, state, 0.0007)
        self.pose(state, 0.0008)
        self.forces(interaction, state, forces)
        np.testing.assert_allclose(state.body_f.numpy(), 0, atol=2e-8)
        # A transverse offset creates compression. Tangential friction opposes
        # sliding, but no axial position or material-coordinate weld is added.
        self.pose(state, 0.0008, x=0.0022, velocity=(0, 0, 0.01))
        self.forces(interaction, state, forces)
        load = state.body_f.numpy()[0]
        self.assertLess(load[0], 0)
        self.assertLess(load[2], 0)
        self.assertLess(float(load[:3] @ np.array([0, 0, 0.01])), 0)
        self.assertLess(interaction.force_balance_error, 1e-8)
        self.assertLess(interaction.torque_balance_error, 1e-10)

    def test_exit_uses_deformed_surface_and_allows_reentry(self):
        interaction, state, forces = self.build()
        self.insert(interaction, state, forces)
        moved = self.vertices.copy()
        moved[3, 2] = 0.012
        state.particle_q.assign(moved)
        self.accept_to(interaction, state, 0.0065)  # outside rest tetrahedron, inside deformed one
        self.assertEqual(interaction.status, "inserted")
        self.accept_to(interaction, state, 0.0075)
        self.assertEqual(interaction.status, "outside")
        self.assertEqual(interaction.exit_count, 1)
        self.assertGreater(interaction.emergence_position[2], 0.0072)
        before = interaction.track_count
        self.accept_to(interaction, state, 0.004)
        self.assertEqual(interaction.status, "inserted")
        self.assertEqual(interaction.entry_count, 2)
        self.assertEqual(interaction.puncture_count, 1)
        self.assertEqual(interaction.track_count, before)

    def test_fixed_support_is_reported_separately_and_rejects_puncture(self):
        interaction, state, forces = self.build(fixed=(0,))
        self.forces(interaction, state, forces)
        self.assertEqual(interaction.status, "contact")
        self.assertGreater(np.linalg.norm(interaction.support_force), 0)
        np.testing.assert_allclose(
            interaction.interface_force + interaction.tissue_force + interaction.support_force, 0, atol=1e-8
        )
        self.assertFalse(interaction.punctured)
        interaction.accept(state, 0)
        self.pose(state, 0.0001)
        self.forces(interaction, state, forces)
        self.assertEqual(interaction.status, "failed")
        self.assertIn("fixed", interaction.reason)

    def test_inactive_surface_cannot_be_punctured(self):
        interaction, state, forces = self.build(
            reference_vertices=self.vertices,
            reference_triangles=self.triangles,
            reference_to_active=np.full(4, -1),
        )
        self.forces(interaction, state, forces)
        self.assertEqual(interaction.status, "contact")
        np.testing.assert_allclose(forces.numpy(), 0, atol=1e-9)
        np.testing.assert_allclose(interaction.interface_force + interaction.support_force, 0, atol=1e-8)
        interaction.accept(state, 0)
        self.assertFalse(interaction.punctured)
        self.pose(state, 0.0001)
        self.forces(interaction, state, forces)
        self.assertEqual(interaction.status, "failed")
        self.assertIn("inactive", interaction.reason)

    def test_track_resolution_preserves_integrated_shaft_load(self):
        loads = []
        for spacing, count in ((0.0004, 5), (0.0002, 9)):
            interaction, state, forces = self.build(
                damping=0,
                shaft_damping_per_m=0,
                track_spacing=spacing,
            )
            self.forces(interaction, state, forces)
            interaction.accept(state, 0)
            self.pose(state, 0.00005)
            interaction.accept(state, 0)
            # Identical physical steps, with exact common first/last stored
            # material endpoints at both quadrature resolutions.
            for index in range(1, 17):
                self.pose(state, 0.00005 + index * 0.000105)
                interaction.accept(state, 0)
            self.assertEqual(interaction.track_count, count)
            self.pose(state, 0.0025, x=0.0022, velocity=(0, 0, 0.01))
            self.forces(interaction, state, forces)
            loads.append(state.body_f.numpy()[0])
        np.testing.assert_allclose(loads[0], loads[1], atol=2e-9, rtol=2e-5)

    def test_backtracking_does_not_duplicate_created_track(self):
        interaction, state, forces = self.build()
        self.insert(interaction, state, forces)
        self.accept_to(interaction, state, 0.0015)
        before = interaction.track_count
        length = interaction.track_length
        self.accept_to(interaction, state, 0.0003)
        self.assertEqual(interaction.status, "inserted")
        self.assertEqual(interaction.track_count, before)
        self.assertAlmostEqual(interaction.track_length, length)
        self.accept_to(interaction, state, 0.0015)
        self.assertEqual(interaction.track_count, before)
        self.assertAlmostEqual(interaction.track_length, length)
        self.pose(state, 0.0015, x=0.0022, velocity=(0, 0, -0.01))
        self.forces(interaction, state, forces)
        self.assertGreater(state.body_f.numpy()[0, 2], 0)
        self.assertLess(interaction.force_balance_error, 1e-8)

    def test_withdrawal_and_reentry_reuse_the_local_opening_and_track(self):
        interaction, state, forces = self.build()
        self.insert(interaction, state, forces)
        self.accept_to(interaction, state, 0.0015)
        count, length = interaction.track_count, interaction.track_length
        self.accept_to(interaction, state, -0.0015)
        self.assertEqual(interaction.status, "outside")
        self.assertEqual(interaction.exit_count, 1)
        self.forces(interaction, state, forces)
        np.testing.assert_allclose(state.body_f.numpy(), 0, atol=1e-8)
        self.accept_to(interaction, state, 0.0015)
        self.assertEqual(interaction.status, "inserted")
        self.assertEqual(interaction.entry_count, 2)
        self.assertEqual(interaction.puncture_count, 1)
        self.assertEqual(interaction.track_count, count)
        self.assertAlmostEqual(interaction.track_length, length)

    def test_continuous_lateral_motion_keeps_shaft_resistance_beyond_capture_radius(self):
        interaction, state, forces = self.build(damping=0, shaft_damping_per_m=0)
        self.insert(interaction, state, forces)
        self.accept_to(interaction, state, 0.0015)
        self.forces(interaction, state, forces)  # engage the finite shaft
        self.accept_to(interaction, state, 0.0015, x=0.0032)
        self.forces(interaction, state, forces)
        self.assertEqual(interaction.status, "inserted")
        self.assertLess(state.body_f.numpy()[0, 0], -0.001)
        self.assertLess(interaction.force_balance_error, 1e-8)
        self.assertLess(interaction.torque_balance_error, 1e-10)

    def test_stationary_tip_pivot_still_checks_motion_of_the_whole_shaft(self):
        interaction, state, forces = self.build()
        self.forces(interaction, state, forces)
        pose = state.body_q.numpy()
        angle = 0.1  # a 1 mm tail motion despite exactly stationary tip
        pose[0, 3:] = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        state.body_q.assign(pose)
        interaction.accept(state, 0)
        self.assertEqual(interaction.status, "failed")
        self.assertIn("shaft displacement", interaction.reason)

    def test_two_material_passages_keep_gap_empty_and_each_new_surface_needs_puncture(self):
        self.vertices = np.vstack((self.vertices, self.vertices + [0, 0, 0.008]))
        self.tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
        self.triangles = boundary_faces(self.tets)[0]
        samples = np.zeros((81, 3))
        samples[:, 2] = np.linspace(0, -0.01, len(samples))
        interaction = NeedleTissueInteraction(
            self.vertices,
            self.tets,
            self.triangles,
            [],
            samples,
            0.0005,
            samples[0],
            device=self.device,
            damping=0,
            shaft_damping_per_m=0,
        )
        state = SimpleNamespace(
            particle_q=wp.array(self.vertices, dtype=wp.vec3, device=self.device),
            particle_qd=wp.zeros(8, dtype=wp.vec3, device=self.device),
            body_q=wp.array([[0.002, 0.002, -0.0001, 0, 0, 0, 1]], dtype=wp.transform, device=self.device),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
        )
        forces = wp.zeros(8, dtype=wp.vec3, device=self.device)
        events = []

        def advance(z, x=0.002):
            start = state.body_q.numpy()[0, :3].astype(float)
            end = np.array([x, 0.002, z])
            count = max(1, int(np.ceil(np.linalg.norm(end - start) / 0.0001)))
            for fraction in np.linspace(0, 1, count + 1)[1:]:
                self.forces(interaction, state, forces)
                point = start + fraction * (end - start)
                self.pose(state, point[2], x=point[0])
                interaction.accept(state, 0)
                self.assertNotEqual(interaction.status, "failed", interaction.reason)
                events.extend(interaction.events)

        advance(0.007)
        self.assertEqual(interaction.status, "outside")
        self.assertEqual((interaction.entry_count, interaction.exit_count, interaction.puncture_count), (1, 1, 1))
        count, length = interaction.track_count, interaction.track_length
        advance(0.007, x=0.0022)
        self.forces(interaction, state, forces)
        self.assertEqual(interaction.last_tip_force, 0)
        self.assertLess(state.body_f.numpy()[0, 0], -1e-5)  # embedded shaft still reacts
        np.testing.assert_allclose(forces.numpy()[4:], 0, atol=1e-8)  # untouched second block is beyond the tip
        self.assertEqual(interaction.track_count, count)
        self.assertAlmostEqual(interaction.track_length, length)
        self.assertLess(interaction.force_balance_error, 1e-8)
        advance(0.007, x=0.002)
        advance(0.0087)
        self.assertEqual(interaction.status, "inserted")
        self.assertEqual((interaction.entry_count, interaction.exit_count, interaction.puncture_count), (2, 1, 2))
        self.assertEqual(interaction.track_section_count, 2)
        self.assertLess(interaction.track_length, 0.007)  # excludes the 2 mm empty gap
        self.assertEqual([event["kind"] for event in events], ["puncture", "entry", "exit", "puncture", "entry"])
        for a, b in interaction._track_edges:
            self.assertEqual(interaction.track_section_ids[a], interaction.track_section_ids[b])
        self.forces(interaction, state, forces)
        self.assertLess(interaction.force_balance_error, 1e-8)
        self.assertLess(interaction.torque_balance_error, 1e-10)

    def test_trial_that_moves_the_tip_beyond_the_step_resolution_is_rejected_before_its_forces(self):
        from newton_solver_suturebot._src.solvers.coupled.interface.insertion import NeedleStepTooLarge

        interaction, state, forces = self.build()
        resolution = interaction.step_resolution
        self.assertEqual(resolution, min(float(interaction.radii.min()), interaction.track_spacing) * 1.01)
        # Without an accepted tip there is nothing to measure a trial against.
        self.pose(state, 0.003)
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True)
        self.pose(state, -0.0001)
        interaction.accept(state, 0)
        accepted = interaction._previous_tip.copy()
        status = interaction.status
        # A trial within the resolution is evaluated; one beyond it is rejected at once
        # and leaves the committed state of the interaction as it was.
        self.pose(state, -0.0001 + 0.99 * resolution)
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True)
        self.pose(state, -0.0001 + 1.01 * resolution)
        with self.assertRaisesRegex(NeedleStepTooLarge, "Unresolved tip displacement in a trial"):
            interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True)
        np.testing.assert_array_equal(interaction._previous_tip, accepted)
        self.assertEqual(interaction.status, status)
        self.assertIsNone(interaction._trial_fields)
        # The rule is for trials: an explicit evaluation still reports through check_step_resolution.
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005)

    def test_trial_forces_do_not_authorize_or_commit_and_final_evaluation_uses_time_n(self):
        interaction, state, forces = self.build()
        start = SimpleNamespace(
            particle_q=wp.clone(state.particle_q),
            body_q=wp.clone(state.body_q),
        )
        self.pose(state, 0.00005)
        loads = []
        for _ in range(3):
            state.body_f.zero_()
            forces.zero_()
            interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True)
            loads.append(state.body_f.numpy().copy())
            self.assertEqual(interaction.status, "approach")
            self.assertFalse(interaction.punctured)
            self.assertIsNone(interaction._pending_opening)
            self.assertIsNone(interaction._step_start_tip)
            self.assertEqual((interaction.track_count, interaction.puncture_count, interaction.entry_count), (0, 0, 0))
            self.assertEqual(interaction.events, ())
            self.assertGreater(interaction.last_tip_force, interaction.puncture_force)
        np.testing.assert_allclose(loads, np.broadcast_to(loads[0], (3, *loads[0].shape)), atol=1e-9)
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, start_state=start)
        np.testing.assert_allclose(interaction._step_start_tip, [0.002, 0.002, -0.0001], atol=1e-10)
        self.assertEqual(interaction.track_count, 0)
        interaction.accept(state, 0)
        self.assertEqual(interaction.status, "inserted")
        self.assertEqual((interaction.track_count, interaction.puncture_count, interaction.entry_count), (1, 1, 1))
        self.assertEqual([event["kind"] for event in interaction.events], ["puncture", "entry"])

    def test_trial_withdrawal_preserves_engagement_and_temporary_penetration_is_not_fatal(self):
        interaction, state, forces = self.build()
        self.insert(interaction, state, forces)
        self.accept_to(interaction, state, 0.0015)
        self.forces(interaction, state, forces)
        engaged = interaction._track_engaged.numpy().copy()
        self.assertGreater(engaged.sum(), 0)
        count, length = interaction.track_count, interaction.track_length

        def nothing_committed():
            self.assertEqual(interaction.status, "inserted")
            np.testing.assert_array_equal(interaction._track_engaged.numpy(), engaged)
            self.assertEqual(interaction.track_count, count)
            self.assertAlmostEqual(interaction.track_length, length)

        # A trial may withdraw the needle as far as one step can move it.
        self.pose(state, 0.0015 - 0.9 * interaction.step_resolution)
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True)
        nothing_committed()
        # A trial that pulls it out altogether is beyond the step resolution: it is rejected.
        self.pose(state, -0.0015)
        with self.assertRaisesRegex(RuntimeError, "Unresolved tip displacement in a trial"):
            interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True)
        nothing_committed()
        blocked, candidate, candidate_forces = self.build(puncture_force=1.0)
        self.pose(candidate, 0.0001)
        blocked.add_forces(candidate, 0, np.zeros(3), candidate_forces, 0.0005, trial=True)
        self.assertEqual(blocked.status, "approach")
        self.assertEqual(blocked.reason, "")
        self.assertIsNone(blocked._pending_opening)
        self.assertGreater(np.linalg.norm(candidate.body_f.numpy()), 0)

    def test_implicit_friction_is_dissipative_and_not_scaled_by_timestep(self):
        interaction, state, forces = self.build(
            damping=0,
            shaft_damping_per_m=0,
            particle_inverse_mass=np.full(4, 1e8),
            needle_mass=0.0003,
            needle_inertia=np.full(3, 1e-8),
        )
        self.insert(interaction, state, forces)
        self.accept_to(interaction, state, 0.0015)
        self.pose(state, 0.0015, x=0.0022, velocity=(0, 0, 0.01))
        loads = []
        for dt in (0.001, 0.0005):
            state.body_f.zero_()
            forces.zero_()
            interaction.add_forces(state, 0, np.zeros(3), forces, dt, implicit=True, trial=True)
            loads.append(state.body_f.numpy()[0].copy())
            self.assertLess(loads[-1][2], -1e-6)
            self.assertLess(float(loads[-1][:3] @ np.array([0, 0, 0.01])), 0)
            self.assertLess(interaction.force_balance_error, 1e-8)
        np.testing.assert_allclose(loads[0], loads[1], rtol=1e-6, atol=1e-10)
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.001, trial=True)
        self.assertLess(abs(state.body_f.numpy()[0, 2]), 0.1 * abs(loads[0][2]))

    def test_surface_damper_is_continuous_at_onset_compressive_and_timestep_independent(self):
        interaction, state, forces = self.build(stiffness=20, damping=0.025, puncture_force=1.0, friction=0)
        radius = float(np.float32(interaction.radii[0]))
        approach_speed = 0.003
        for penetration in (radius, 1e-6, 1e-7, 1e-8, 0.0):
            self.pose(state, -radius + penetration, velocity=(0, 0, approach_speed))
            actual_penetration = max(0.0, radius + float(state.body_q.numpy()[0, 2]))
            values = []
            for implicit in (False, True):
                for dt in (0.001, 0.0005):
                    state.body_f.zero_()
                    forces.zero_()
                    interaction.add_forces(state, 0, np.zeros(3), forces, dt, trial=True, implicit=implicit)
                    values.append(interaction.last_tip_force)
            expected_damper = 0.025 * actual_penetration / radius * approach_speed
            expected = 20 * actual_penetration + expected_damper
            np.testing.assert_allclose(values, expected, rtol=2e-5, atol=1e-10)
            self.pose(state, -radius + penetration, velocity=(0, 0, -approach_speed))
            state.body_f.zero_()
            forces.zero_()
            interaction.add_forces(state, 0, np.zeros(3), forces, 0.001, trial=True, implicit=True)
            np.testing.assert_allclose(interaction.last_tip_force, 20 * actual_penetration, rtol=2e-5, atol=1e-10)
            self.assertLessEqual(-expected_damper * approach_speed, 0)  # viscous work during compression
        self.assertEqual(values, [0.0] * 4)

    def test_leading_tip_continues_across_an_internal_cell_without_surface_force(self):
        self.vertices = np.vstack((self.vertices, [0.01, 0.01, 0.01]))
        self.tets = np.array([[0, 1, 2, 3], [4, 1, 3, 2]], dtype=np.int32)
        self.triangles = boundary_faces(self.tets)[0]
        interaction, state, forces = self.build(damping=0, shaft_damping_per_m=0, friction=0)
        self.insert(interaction, state, forces)
        self.accept_to(interaction, state, 0.0059)
        count = interaction.track_count
        visited = interaction._visited_tets.copy()
        self.assertEqual(len(visited), 1)
        # x+y+z=0.01 is a shared internal cell face, with no physical surface.
        self.pose(state, 0.0061)
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True, trial=True)
        self.assertEqual(interaction.last_tip_force, 0)
        np.testing.assert_allclose(interaction.interface_force, 0, atol=2e-8)
        self.assertEqual(interaction.track_count, count)
        self.assertEqual(interaction._visited_tets, visited)

    def test_candidate_emergence_has_no_artificial_tip_force_or_early_exit_event(self):
        interaction, state, forces = self.build(damping=0, shaft_damping_per_m=0, friction=0)
        self.insert(interaction, state, forces)
        self.accept_to(interaction, state, 0.0059)
        count = interaction.track_count
        self.pose(state, 0.0061)
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True, trial=True)
        self.assertEqual(interaction.last_tip_force, 0)
        np.testing.assert_allclose(interaction.interface_force, 0, atol=2e-8)
        self.assertEqual(interaction.status, "inserted")
        self.assertEqual(interaction.exit_count, 0)
        self.assertEqual(interaction.events, ())
        self.assertEqual(interaction.track_count, count)

    def test_candidate_exit_does_not_authorize_contact_with_the_next_gap_wall(self):
        first = self.vertices.copy()
        first[3, 2] = 0.0001
        self.vertices = np.vstack((first, first + [0, 0, 0.00012]))
        self.tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
        self.triangles = boundary_faces(self.tets)[0]
        interaction, state, forces = self.build(damping=0, shaft_damping_per_m=0, friction=0)
        self.forces(interaction, state, forces)
        interaction.accept(state, 0)
        self.accept_to(interaction, state, 0.00003)
        self.assertEqual(interaction.status, "inserted")
        self.pose(state, 0.00011)  # past the first exit (60 um), before next entry (120 um)
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True, trial=True)
        self.assertGreater(interaction.last_tip_force, 0.01)
        self.assertLess(interaction.interface_force[2], 0)
        self.assertGreater(forces.numpy()[4:, 2].sum(), 0.01)
        self.assertLess(interaction.force_balance_error, 1e-8)
        self.assertEqual((interaction.entry_count, interaction.exit_count, interaction.puncture_count), (1, 0, 1))

    def test_lateral_shaft_motion_across_internal_cells_adds_no_surface_barrier(self):
        self.vertices = np.vstack((self.vertices, [0.01, 0.01, 0.01]))
        self.tets = np.array([[0, 1, 2, 3], [4, 1, 3, 2]], dtype=np.int32)
        self.triangles = boundary_faces(self.tets)[0]
        interaction, state, forces = self.build(
            sample_count=101, damping=0, shaft_damping_per_m=0, shaft_stiffness_per_m=0, friction=0
        )
        self.insert(interaction, state, forces)
        self.accept_to(interaction, state, 0.0059)
        count, length = interaction.track_count, interaction.track_length
        visited = interaction._visited_tets.copy()
        self.pose(state, 0.0059, x=0.0023)
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True, trial=True)
        # Isolate the surface law: transverse shaft springs are disabled here.
        # Sample1 crossed an internal face near the actual traversed track,
        # while no needle portion crossed an exterior material boundary.
        np.testing.assert_allclose(interaction.interface_force, 0, atol=2e-8)
        self.assertEqual(interaction.last_tip_force, 0)
        self.assertEqual(interaction.track_count, count)
        self.assertAlmostEqual(interaction.track_length, length)
        self.assertEqual(interaction._visited_tets, visited)

    def test_retained_shaft_association_has_no_interior_capture_radius_force_jump(self):
        # Only the middle shaft sample lies in tissue, so this isolates interior
        # surface penalties from legitimate contact near the entry/exit rims.
        interaction, state, forces = self.build(
            sample_count=3, damping=0, shaft_damping_per_m=0, shaft_stiffness_per_m=0, friction=0
        )
        self.insert(interaction, state, forces)
        self.accept_to(interaction, state, 0.007)
        count, length = interaction.track_count, interaction.track_length
        for x in (0.002999, 0.003001, 0.0035):
            self.accept_to(interaction, state, 0.007, x=x)
            state.body_f.zero_()
            forces.zero_()
            interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True, trial=True)
            np.testing.assert_allclose(interaction.interface_force, 0, atol=2e-8)
            self.assertEqual(interaction.track_count, count)
            self.assertAlmostEqual(interaction.track_length, length)
        # Restoring the actual transverse law gives a nonzero restoring load;
        # the geometric exemption does not release the engaged shaft spring.
        interaction.shaft_stiffness_per_m = 1500
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True, trial=True)
        self.assertLess(interaction.interface_force[0], -0.001)
        self.assertLess(interaction.force_balance_error, 1e-8)

    def test_physical_radius_tangent_regularizes_authored_joint_and_is_refinement_invariant(self):
        coarse = np.array([[0, 0, 0], [0, 0, -0.002], [0.00014, 0, -0.004]])
        angle = np.arctan(0.00014 / 0.002)
        bisector = -0.002 + 0.0002 * np.tan(angle / 2)
        points = np.array([[0.0002, 0, bisector - 1e-8], [0.0002, 0, bisector + 1e-8]])
        refined = np.vstack((np.linspace(coarse[0], coarse[1], 6)[:-1], np.linspace(coarse[1], coarse[2], 6)))
        results = []
        for samples in (coarse, refined):
            output = wp.zeros(2, dtype=wp.vec3, device=self.device)
            wp.launch(
                _probe_tangents,
                2,
                [
                    wp.array(points, dtype=wp.vec3, device=self.device),
                    wp.array(samples, dtype=wp.vec3, device=self.device),
                    0.0005,
                    output,
                ],
                device=self.device,
            )
            results.append(output.numpy())
        # The incident segments differ by four degrees. A fixed-radius secant
        # removes their large direction jump at the closest-segment switch.
        self.assertLess(np.linalg.norm(results[0][1] - results[0][0]), 0.002)
        np.testing.assert_allclose(results[0], results[1], atol=2e-6)
        np.testing.assert_allclose(np.linalg.norm(results[0], axis=1), 1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
