# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""A local puncture must not turn off contact with nearby untouched material."""

import os
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp
from test_needle_insertion_geometry import _boxes

from newton_solver_suturebot import NeedleTissueInteraction, boundary_faces


class TestNeedleInsertionContactGuards(unittest.TestCase):
    def build(self, *, wall_x, wall_top, shaft_length, initial_x, initial_z):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        first = np.array([[0, 0, 0], [0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])
        second = np.array(
            [
                [wall_x, 0, wall_top],
                [wall_x + 0.001, 0, wall_top],
                [wall_x, 0.001, wall_top],
                [wall_x, 0, wall_top - 0.001],
            ]
        )
        vertices = np.vstack((first, second))
        tetrahedra = np.array([[0, 1, 2, 3], [4, 5, 7, 6]], np.int32)
        samples = np.zeros((17, 3))
        samples[:, 2] = np.linspace(0, -shaft_length, len(samples))
        interaction = NeedleTissueInteraction(
            vertices,
            tetrahedra,
            boundary_faces(tetrahedra)[0],
            [],
            samples,
            0.0002,
            samples[0],
            device=device,
            shaft_stiffness_per_m=0,
            shaft_damping_per_m=0,
            damping=0,
            friction=0,
            track_spacing=0.00025,
        )
        state = SimpleNamespace(
            particle_q=wp.array(vertices, dtype=wp.vec3, device=device),
            particle_qd=wp.zeros(8, dtype=wp.vec3, device=device),
            body_q=wp.array([[initial_x, 0.0002, initial_z, 0, 0, 0, 1]], dtype=wp.transform, device=device),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=device),
        )
        return interaction, state, wp.zeros(8, dtype=wp.vec3, device=device)

    def forces(self, interaction, state, forces):
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005)

    def advance(self, interaction, state, forces, x, z):
        """Stage an interface step, then supply its accepted prescribed probe pose."""
        self.forces(interaction, state, forces)
        self.assertNotEqual(interaction.status, "failed", interaction.reason)
        state.body_q.assign(np.array([[x, 0.0002, z, 0, 0, 0, 1]], np.float32))
        interaction.accept(state, 0)
        self.assertNotEqual(interaction.status, "failed", interaction.reason)

    def assert_untouched_wall_contact(self, interaction, state, forces):
        self.forces(interaction, state, forces)
        reaction = forces.numpy()[4:].sum(axis=0)
        self.assertGreater(reaction[0], 1.0e-8, "Untouched active wall contact was suppressed by a nearby passage")
        self.assertLess(interaction.force_balance_error, 1.0e-8)
        self.assertLess(interaction.torque_balance_error, 1.0e-10)

    def test_track_does_not_shadow_a_separate_active_wall(self):
        interaction, state, forces = self.build(
            wall_x=0.00055, wall_top=-0.0005, shaft_length=0.0008, initial_x=0.0002, initial_z=-0.0001
        )
        self.advance(interaction, state, forces, 0.0002, -0.00005)
        self.assertTrue(interaction.punctured)
        self.advance(interaction, state, forces, 0.0002, 0.00005)
        self.advance(interaction, state, forces, 0.0002, 0.0001)
        self.assertGreater(interaction.track_count, 0)
        self.advance(interaction, state, forces, 0.0003, 0.0001)
        self.advance(interaction, state, forces, 0.0004, 0.0001)
        self.assertEqual(interaction.status, "inserted")
        # The tip remains inside the first tetrahedron. The shaft tail is now
        # 0.15 mm from a different material wall, inside the 0.2 mm proxy radius.
        self.assert_untouched_wall_contact(interaction, state, forces)

    def test_opening_does_not_authorize_a_separate_active_wall(self):
        interaction, state, forces = self.build(
            wall_x=0.00045, wall_top=-0.0001, shaft_length=0.0001, initial_x=0.0001, initial_z=-0.00005
        )
        self.advance(interaction, state, forces, 0.0001, -0.00005)
        self.assertTrue(interaction.punctured)
        self.assertEqual(interaction.track_count, 0)
        self.advance(interaction, state, forces, 0.0002, -0.00005)
        self.advance(interaction, state, forces, 0.0003, -0.00005)
        self.assertEqual(interaction.track_count, 0)
        # Even before the first embedded track sample, an authorized opening
        # applies only to its own material, not an adjacent disconnected wall.
        self.assert_untouched_wall_contact(interaction, state, forces)

    def thin_layers(self, *, second_layer, puncture_force=0.003):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        first = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.0001]])
        vertices = np.vstack((first, first + [0, 0, 0.00012])) if second_layer else first
        tetrahedra = np.array([[0, 1, 2, 3], [4, 5, 6, 7]] if second_layer else [[0, 1, 2, 3]], np.int32)
        samples = np.array([[0, 0, 0], [0, 0, -0.0004]])
        interaction = NeedleTissueInteraction(
            vertices,
            tetrahedra,
            boundary_faces(tetrahedra)[0],
            [],
            samples,
            0.0002,
            samples[0],
            device=device,
            puncture_force=puncture_force,
            track_spacing=0.00025,
        )
        state = SimpleNamespace(
            particle_q=wp.array(vertices, dtype=wp.vec3, device=device),
            particle_qd=wp.zeros(len(vertices), dtype=wp.vec3, device=device),
            body_q=wp.array([[0.002, 0.002, -0.00005, 0, 0, 0, 1]], dtype=wp.transform, device=device),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=device),
        )
        return interaction, state, wp.zeros(len(vertices), dtype=wp.vec3, device=device)

    def test_thin_gap_between_inside_endpoints_is_not_silently_bridged(self):
        interaction, state, forces = self.thin_layers(second_layer=True)
        self.forces(interaction, state, forces)
        state.body_q.assign(np.array([[0.002, 0.002, 0.00003, 0, 0, 0, 1]], np.float32))
        interaction.accept(state, 0)
        self.assertEqual(interaction.status, "inserted")
        before = interaction.track_count
        self.forces(interaction, state, forces)
        # Both endpoints are inside material. This 0.11 mm step is below the
        # 0.2 mm proxy radius, but crosses the first exit and second entry.
        state.body_q.assign(np.array([[0.002, 0.002, 0.00014, 0, 0, 0, 1]], np.float32))
        interaction.accept(state, 0)
        self.assertEqual(interaction.status, "failed")
        self.assertIn("surface crossings", interaction.reason)
        self.assertEqual(interaction.track_count, before)

    def test_thin_material_between_outside_endpoints_cannot_be_skipped(self):
        interaction, state, forces = self.thin_layers(second_layer=False, puncture_force=1.0)
        self.forces(interaction, state, forces)
        # Endpoints are outside on opposite sides. At this XY the material is
        # only 0.06 mm thick, within the permitted 0.15 mm probe displacement.
        state.body_q.assign(np.array([[0.002, 0.002, 0.0001, 0, 0, 0, 1]], np.float32))
        interaction.accept(state, 0)
        self.assertEqual(interaction.status, "failed")
        self.assertIn("surface crossings", interaction.reason)
        self.assertFalse(interaction.punctured)
        self.assertEqual(interaction.track_count, 0)

    def test_large_unresolved_tip_step_reports_failure(self):
        interaction, state, forces = self.thin_layers(second_layer=False)
        self.forces(interaction, state, forces)
        state.body_q.assign(np.array([[0.002, 0.002, 0.003, 0, 0, 0, 1]], np.float32))
        interaction.accept(state, 0)
        self.assertEqual(interaction.status, "failed")
        self.assertIn("Unresolved tip or shaft displacement", interaction.reason)
        self.assertFalse(interaction.punctured)
        self.assertEqual(interaction.track_count, 0)

    def internal_inactive_layer(self, shaft_direction):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        full = _boxes(
            [
                ((0, 0, 0), (0.01, 0.01, 0.0001)),
                ((0, 0, 0.0001), (0.01, 0.01, 0.00012)),
                ((0, 0, 0.00012), (0.01, 0.01, 0.00022)),
            ],
            [1, 2, 1],
        )
        vertices, remap = np.unique(np.round(full.vertices, 12), axis=0, return_inverse=True)
        all_tetrahedra = remap[full.indices].astype(np.int32)
        active = all_tetrahedra[full.tetrahedron_region_ids == 1]
        # The full reference exterior has no face at either internal interface.
        exterior = boundary_faces(all_tetrahedra)[0]
        samples = np.array([[0, 0, 0], [0, 0, shaft_direction * 0.00011]])
        interaction = NeedleTissueInteraction(
            vertices,
            active,
            exterior,
            [],
            samples,
            0.0002,
            samples[0],
            device=device,
            damping=0,
            shaft_stiffness_per_m=0,
            shaft_damping_per_m=0,
            friction=0,
            reference_vertices=vertices,
            reference_triangles=exterior,
            reference_to_active=np.arange(len(vertices)),
        )
        state = SimpleNamespace(
            particle_q=wp.array(vertices, dtype=wp.vec3, device=device),
            particle_qd=wp.zeros(len(vertices), dtype=wp.vec3, device=device),
            body_q=wp.array([[0.002, 0.002, -0.00005, 0, 0, 0, 1]], dtype=wp.transform, device=device),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=device),
        )
        forces = wp.zeros(len(vertices), dtype=wp.vec3, device=device)
        self.forces(interaction, state, forces)
        state.body_q.assign(np.array([[0.002, 0.002, 0.00003, 0, 0, 0, 1]], np.float32))
        interaction.accept(state, 0)
        self.assertEqual(interaction.status, "inserted")
        return interaction, state, forces

    def test_tip_cannot_bridge_an_internal_inactive_layer_hidden_from_reference_exterior(self):
        interaction, state, forces = self.internal_inactive_layer(-1)
        before = interaction.track_count
        state.body_q.assign(np.array([[0.002, 0.002, 0.00014, 0, 0, 0, 1]], np.float32))
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True, trial=True)
        self.assertGreater(interaction.last_tip_force, 0.01)
        self.assertEqual(interaction.track_count, before)
        interaction.accept(state, 0)
        self.assertEqual(interaction.status, "failed")
        self.assertIn("active material surface crossings", interaction.reason)
        self.assertEqual(interaction.track_count, before)

    def test_shaft_track_association_cannot_cross_an_internal_inactive_layer(self):
        interaction, state, forces = self.internal_inactive_layer(1)
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True, trial=True)
        self.assertEqual(interaction.last_tip_force, 0)
        self.assertGreater(interaction.interface_force[2], 0.001)
        upper = interaction.vertices[:, 2] >= 0.00012
        self.assertGreater(np.linalg.norm(forces.numpy()[upper]), 0.001)
        self.assertLess(interaction.force_balance_error, 1e-8)
        self.assertEqual((interaction.entry_count, interaction.exit_count, interaction.puncture_count), (1, 0, 1))


if __name__ == "__main__":
    unittest.main()
