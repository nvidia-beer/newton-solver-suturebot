# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Finite-radius surface contact at creases, mesh seams, and nearby walls."""

import os
import unittest
from itertools import permutations
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from test_needle_insertion_example import _example
from test_needle_tissue_coupler import _TranslatingTissue

from newton_solver_suturebot import NeedleTissueCoupler, NeedleTissueInteraction, boundary_faces


def _block(subdivisions=1, slope=0.12, thickness=0.004):
    """Closed Tet4 block with an exact V-shaped upper surface along x=y."""
    axis = np.linspace(-0.004, 0.004, subdivisions + 1)
    vertices = np.array(
        [
            (x, y, -thickness if layer == 0 else slope * abs(x - y) / np.sqrt(2))
            for x in axis
            for y in axis
            for layer in (0, 1)
        ]
    )
    nodes = np.arange(len(vertices)).reshape(subdivisions + 1, subdivisions + 1, 2)
    tetrahedra = []
    for x in range(subdivisions):
        for y in range(subdivisions):
            cell = nodes[x : x + 2, y : y + 2].ravel()
            for first, second, _ in permutations(range(3)):
                tet = cell[[0, 1 << first, (1 << first) | (1 << second), 7]].copy()
                if np.linalg.det((vertices[tet[1:]] - vertices[tet[:1]]).T) < 0:
                    tet[1], tet[2] = tet[2], tet[1]
                tetrahedra.append(tet)
    return vertices, np.asarray(tetrahedra, dtype=np.int32)


class NeedleSurfacePatchTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))

    def make(self, vertices, tets, *, radius=0.0005, **parameters):
        samples = np.array([[0, 0, 0], [0, 0, 0.01]])
        options = dict(stiffness=20.0, damping=0.0, friction=0.0, puncture_force=1.0)
        options.update(parameters)
        interaction = NeedleTissueInteraction(
            vertices,
            tets,
            boundary_faces(tets)[0],
            [],
            samples,
            radius,
            samples[0],
            device=self.device,
            **options,
        )
        state = SimpleNamespace(
            particle_q=wp.array(vertices, dtype=wp.vec3, device=self.device),
            particle_qd=wp.zeros(len(vertices), dtype=wp.vec3, device=self.device),
            body_q=wp.array([[0, 0, 0.0004, 0, 0, 0, 1]], dtype=wp.transform, device=self.device),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
        )
        forces = wp.zeros(len(vertices), dtype=wp.vec3, device=self.device)
        return interaction, state, forces

    def evaluate(self, interaction, state, forces, position, *, com=(0, 0, 0)):
        state.body_q.assign(np.array([[*position, 0, 0, 0, 1]], dtype=np.float32))
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.asarray(com), forces, 0.0005, implicit=True)
        return state.body_f.numpy()[0].astype(float), forces.numpy().astype(float)

    def test_deep_interior_recovery_requires_an_actual_punctured_passage(self):
        radius = 0.00028
        options = dict(radius=radius, shaft_stiffness_per_m=0, shaft_damping_per_m=0)
        interaction, state, forces = self.make(*_block(slope=0), **options)
        load, nodal = self.evaluate(interaction, state, forces, (0, 0, -0.001))
        # The nearest surface is farther away than the radius. Unauthorized
        # interior recovery still needs that surface and its outward normal.
        self.assertAlmostEqual(load[2], 20 * (radius + 0.001), delta=1e-8)
        np.testing.assert_allclose(load[:3] + nodal.sum(axis=0), 0, atol=1e-8)

        interaction, state, forces = self.make(*_block(slope=0), puncture_force=0.003, **options)
        self.evaluate(interaction, state, forces, (0, 0, 0.0001))
        interaction.accept(state, 0)
        self.assertEqual(interaction.puncture_count, 1)
        for depth in np.linspace(0.0001, -0.001, 13)[1:]:
            state.body_q.assign(np.array([[0, 0, depth, 0, 0, 0, 1]], np.float32))
            interaction.accept(state, 0)
            self.assertNotEqual(interaction.status, "failed", interaction.reason)
        self.assertTrue(interaction._in_material)
        self.assertGreater(interaction.track_count, 0)
        load, nodal = self.evaluate(interaction, state, forces, (0, 0, -0.001))
        # An actual traversed passage has no remote surface barrier; shaft
        # response is disabled here to isolate the surface contact query.
        np.testing.assert_array_equal(load, 0)
        np.testing.assert_array_equal(nodal, 0)

    def test_concave_bisector_has_continuous_force_and_material_load(self):
        interaction, state, forces = self.make(*_block())
        transverse = np.array([1, -1, 0]) / np.sqrt(2)
        differences = []
        for epsilon in (1e-7, 1e-8, 1e-9):
            values = []
            for side in (-1, 1):
                load, nodal = self.evaluate(
                    interaction, state, forces, np.array([0, 0, 0.0004]) + side * epsilon * transverse
                )
                self.assertGreater(load[2], 0.001)
                values.append(np.r_[load, nodal.ravel()])
            differences.append(np.linalg.norm(values[1] - values[0]))
        self.assertLess(differences[-1], 2e-7)
        self.assertLess(differences[-1], differences[0] / 20)

    def test_concave_patch_conserves_force_and_moment_with_friction(self):
        vertices, tets = _block()
        interaction, state, forces = self.make(vertices, tets, damping=0.025, friction=0.15)
        state.body_qd.assign(np.array([[0.003, 0.001, -0.001, 0, 0, 0]], np.float32))
        position, com = np.array([0.00011, 0.00011, 0.0004]), np.array([0.0007, 0.0003, 0])
        load, nodal = self.evaluate(interaction, state, forces, position, com=com)
        np.testing.assert_allclose(load[:3] + nodal.sum(axis=0), 0, atol=1e-8)
        moment = load[3:] + np.cross(position + com, load[:3]) + np.cross(vertices, nodal).sum(axis=0)
        np.testing.assert_allclose(moment, 0, atol=1e-10)
        self.assertGreater(np.linalg.norm(load[3:]), 1e-7)
        self.assertGreater(np.count_nonzero(np.linalg.norm(nodal, axis=1) > 1e-8), 3)

    def test_planar_force_is_preserved_under_surface_subdivision(self):
        loads = []
        for divisions in (1, 2, 4):
            interaction, state, forces = self.make(*_block(divisions, slope=0))
            load, _ = self.evaluate(interaction, state, forces, (0.00013, 0.00027, 0.0004))
            loads.append(load[:3])
        np.testing.assert_allclose(loads, np.tile([0, 0, 0.002], (3, 1)), atol=2e-7, rtol=2e-4)

    def test_nonpenetrating_facet_retains_its_positive_normalization_weight(self):
        # The sample is 0.8 radii above the first slab and 1.1 above the second.
        # Both top surfaces belong to the blending footprint, while only the
        # first penetrates. Omitting the second's weight would increase the
        # reaction to 2 mN and change the defined surface-pressure law.
        first, tets = _block(slope=0, thickness=0.00005)
        second, other_tets = _block(slope=0, thickness=0.00005)
        second[:, 2] -= 0.00015
        vertices = np.vstack([first, second])
        tets = np.vstack([tets, other_tets + len(first)])
        interaction, state, forces = self.make(vertices, tets)
        load, nodal = self.evaluate(interaction, state, forces, (0, 0, 0.0004))
        expected = 0.002 / (1 + (1 - 0.6**2) ** 2)
        self.assertAlmostEqual(load[2], expected, delta=1e-8)
        self.assertLess(load[2], 0.0015)
        np.testing.assert_allclose(nodal[len(first) :], 0, atol=1e-12)

    def test_convex_edge_first_touch_is_continuous(self):
        interaction, state, forces = self.make(*_block(slope=0))
        edge = np.array([0.004, 0.0002, 0])
        outward = np.array([1, 0, 1]) / np.sqrt(2)
        outside, _ = self.evaluate(interaction, state, forces, edge + (0.0005 + 1e-8) * outward)
        first, _ = self.evaluate(interaction, state, forces, edge + (0.0005 - 1e-9) * outward)
        deeper, _ = self.evaluate(interaction, state, forces, edge + (0.0005 - 1e-5) * outward)
        np.testing.assert_allclose(outside, 0, atol=1e-9)
        self.assertLess(np.linalg.norm(first[:3]), 1e-7)
        self.assertGreater(np.linalg.norm(deeper[:3]), 1e-5)
        self.assertGreater(deeper[:3] @ outward, 0)

    def test_opposite_face_of_thin_layer_does_not_cancel_exterior_contact(self):
        loads = []
        for thickness in (0.004, 0.0001):
            interaction, state, forces = self.make(*_block(slope=0, thickness=thickness))
            load, _ = self.evaluate(interaction, state, forces, (0, 0, 0.0004))
            loads.append(load[:3])
        np.testing.assert_allclose(loads, np.tile([0, 0, 0.002], (2, 1)), atol=2e-7, rtol=2e-4)

    def test_local_opening_does_not_hide_another_intact_wall(self):
        first, tets = _block(slope=0)
        first[:, 0] = (first[:, 0] + 0.004) * (0.00425 / 0.008) - 0.004
        second, other_tets = _block(slope=0)
        second[:, 0] = (second[:, 0] + 0.004) * 0.5 + 0.0004
        second[:, 2] += 0.002
        vertices = np.vstack([first, second])
        tets = np.vstack([tets, other_tets + len(first)])
        interaction, state, forces = self.make(vertices, tets, radius=0.0003, puncture_force=0.002)
        self.evaluate(interaction, state, forces, (-0.0001, 0.0002, 0.0001))
        interaction.accept(state, 0)
        self.assertEqual(interaction.puncture_count, 1)
        load, _ = self.evaluate(interaction, state, forces, (0.00015, 0.0002, 0.0001))
        self.assertLess(load[0], -0.0001)
        self.assertLess(abs(load[2]), 1e-7)

    def _first_entry_trial(self, *, second_layer=False, candidate_x=0.0):
        vertices, tets = _block(slope=0, thickness=0.0001)
        if second_layer:
            other, other_tets = _block(slope=0, thickness=0.0001)
            other[:, 2] -= 0.00015
            tets = np.vstack([tets, other_tets + len(vertices)])
            vertices = np.vstack([vertices, other])
        interaction, state, forces = self.make(vertices, tets, radius=0.00028, puncture_force=0.003)
        self.evaluate(interaction, state, forces, (0, 0, 0.0001))
        interaction.accept(state, 0)
        self.assertEqual((interaction.puncture_count, interaction.entry_count, interaction.track_count), (1, 0, 0))
        events = interaction.events
        state.body_q.assign(np.array([[candidate_x, 0, -1e-7, 0, 0, 0, 1]], np.float32))
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True, implicit=True)
        self.assertEqual((interaction.puncture_count, interaction.entry_count, interaction.track_count), (1, 0, 0))
        self.assertIs(interaction.events, events)
        return interaction, state, forces

    def test_authorized_first_entry_trial_has_no_unvisited_backface_barrier(self):
        interaction, state, forces = self._first_entry_trial()
        np.testing.assert_allclose(state.body_f.numpy(), 0, atol=1e-8)
        np.testing.assert_allclose(forces.numpy(), 0, atol=1e-8)
        interaction.accept(state, 0)
        self.assertEqual(interaction.entry_count, 1)
        self.assertGreater(interaction.track_count, 0)

    def test_authorized_first_entry_trial_still_contacts_intact_front_wall(self):
        interaction, state, forces = self._first_entry_trial(second_layer=True)
        load = state.body_f.numpy()[0, :3]
        self.assertGreater(load[2], 0.001)
        np.testing.assert_allclose(load[:2], 0, atol=1e-8)
        np.testing.assert_allclose(load + forces.numpy().sum(axis=0), 0, atol=1e-8)
        self.assertEqual((interaction.entry_count, interaction.track_count), (0, 0))

    def test_opening_does_not_authorize_a_distant_first_entry_trial(self):
        interaction, state, forces = self._first_entry_trial(candidate_x=0.001)
        load = state.body_f.numpy()[0, :3]
        self.assertGreater(np.linalg.norm(load), 0.003)
        np.testing.assert_allclose(load + forces.numpy().sum(axis=0), 0, atol=1e-8)
        self.assertEqual((interaction.entry_count, interaction.track_count), (0, 0))

    def _separate_corner_trial(self, distance):
        """An entered thin layer and a separate sharp wall across empty space."""
        radius, tip_z = 0.00028, -1e-7
        first, tets = _block(slope=0, thickness=0.0001)
        # The closest wall edge lies diagonally from the tip. Both coordinate
        # separations are inside its radius even when their Euclidean norm is
        # outside, so a triangle AABB alone cannot decide physical contact.
        a, b = 0.6 * distance, 0.8 * distance
        profile = [(a, tip_z - b), (a + 0.001, tip_z - b - 0.00002), (a + 0.001, tip_z - b - 0.001)]
        wall = np.array([(x, y, z) for y in (-0.004, 0.004) for x, z in profile])
        wall_tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int32)
        for tet in wall_tets:
            if np.linalg.det((wall[tet[1:]] - wall[tet[:1]]).T) < 0:
                tet[1], tet[2] = tet[2], tet[1]
        vertices = np.vstack([first, wall])
        interaction, state, forces = self.make(
            vertices,
            np.vstack([tets, wall_tets + len(first)]),
            radius=radius,
            puncture_force=0.003,
            shaft_stiffness_per_m=0,
            shaft_damping_per_m=0,
        )
        self.evaluate(interaction, state, forces, (0, 0, 0.0001))
        interaction.accept(state, 0)
        state.body_q.assign(np.array([[0, 0, tip_z, 0, 0, 0, 1]], np.float32))
        interaction.accept(state, 0)
        self.assertEqual((interaction.puncture_count, interaction.entry_count), (1, 1))
        self.assertGreater(interaction.track_count, 0)
        history = interaction.events
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True, implicit=True)
        self.assertIs(interaction.events, history)
        load, nodal = state.body_f.numpy()[0].astype(float), forces.numpy().astype(float)
        np.testing.assert_allclose(load[:3] + nodal.sum(axis=0), 0, atol=1e-8)
        moment = load[3:] + np.cross([0, 0, tip_z], load[:3]) + np.cross(vertices, nodal).sum(axis=0)
        np.testing.assert_allclose(moment, 0, atol=1e-10)
        return load, nodal

    def test_separate_wall_outside_radius_does_not_contact_through_its_aabb(self):
        distance, radius = 0.00032, 0.00028
        self.assertLess(max(0.6 * distance, 0.8 * distance), radius)
        self.assertGreater(distance, radius)
        load, nodal = self._separate_corner_trial(distance)
        np.testing.assert_allclose(load, 0, atol=1e-9)
        np.testing.assert_allclose(nodal, 0, atol=1e-9)

    def test_separate_wall_contact_starts_continuously_at_physical_radius(self):
        radius = 0.00028
        outside, _ = self._separate_corner_trial(radius + 1e-8)
        first, _ = self._separate_corner_trial(radius - 1e-9)
        deeper, _ = self._separate_corner_trial(radius - 1e-5)
        np.testing.assert_allclose(outside, 0, atol=1e-9)
        self.assertLess(np.linalg.norm(first[:3]), 1e-7)
        self.assertGreater(np.linalg.norm(deeper[:3]), 1e-4)
        self.assertLess(np.linalg.norm(deeper[:3]), 2.1e-4)
        self.assertGreater(deeper[2], 0)

    def test_traversed_tip_crosses_adjacent_facet_plane_without_surface_barrier(self):
        slope, radius = 0.12, 0.00028
        interaction, state, forces = self.make(
            *_block(slope=slope),
            radius=radius,
            puncture_force=0.003,
            shaft_stiffness_per_m=0,
            shaft_damping_per_m=0,
        )
        point = np.array([-0.002, 0, slope * 0.002 / np.sqrt(2) + 0.0001])
        self.evaluate(interaction, state, forces, point)
        interaction.accept(state, 0)
        self.assertEqual(interaction.puncture_count, 1)

        def advance(target):
            nonlocal point
            target = np.asarray(target)
            count = max(1, int(np.ceil(np.linalg.norm(target - point) / (0.4 * radius))))
            start = point.copy()
            for fraction in np.linspace(0, 1, count + 1)[1:]:
                point = start + fraction * (target - start)
                state.body_q.assign(np.array([[*point, 0, 0, 0, 1]], np.float32))
                interaction.accept(state, 0)
                self.assertNotEqual(interaction.status, "failed", interaction.reason)

        advance([-0.002, 0, -0.0002])
        advance([-0.00002, 0.00002, -0.0002])
        opposite_plane = -slope * 0.00004 / np.sqrt(2)
        advance([-0.00002, 0.00002, opposite_plane - 2e-7])
        count = interaction.track_count
        self.assertGreater(count, 1)
        self.assertEqual((interaction.entry_count, interaction.exit_count), (1, 0))
        # Both candidates are still about 7 micrometres below the actual
        # surface. Crossing the infinite plane of an adjacent concave facet
        # is not a material boundary or a new intact wall.
        for offset in (-1e-7, 1e-7):
            candidate = np.array([-0.00002, 0.00002, opposite_plane + offset])
            state.body_q.assign(np.array([[*candidate, 0, 0, 0, 1]], np.float32))
            state.body_f.zero_()
            forces.zero_()
            interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True, implicit=True)
            np.testing.assert_allclose(state.body_f.numpy(), 0, atol=1e-8)
            np.testing.assert_allclose(forces.numpy(), 0, atol=1e-8)
            self.assertEqual((interaction.entry_count, interaction.exit_count, interaction.track_count), (1, 0, count))

    def test_dynamic_contact_crosses_concave_crease_without_a_planned_path(self):
        with _example() as example:
            # This friction-free crossing uses initial lateral momentum and
            # an external normal load, without the example's PD controller.
            example.model.joint_damping.zero_()
            example.rigid_solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
            vertices, tets = _block()
            example.state_0.particle_q.assign(vertices)
            example.state_0.particle_qd.zero_()
            com = example.drive.com
            transverse = np.array([1, -1, 0]) / np.sqrt(2)
            center = np.array([0, 0, 0.0004]) - 2e-6 * transverse
            example._initialize_pose(np.r_[center - com, 0, 0, 0, 1])
            velocity = example.state_0.body_qd.numpy()
            velocity[example.needle_body, :3] = 0.001 * transverse
            example.state_0.body_qd.assign(velocity)
            newton.eval_ik(example.model, example.state_0, example.state_0.joint_q, example.state_0.joint_qd)
            samples = np.asarray([com, com + [0, 0, 0.01]])
            interaction = NeedleTissueInteraction(
                vertices,
                tets,
                boundary_faces(tets)[0],
                [],
                samples,
                0.0005,
                samples[0],
                device=example.device,
                stiffness=20,
                damping=0.025,
                friction=0,
                puncture_force=1,
            )
            coupler = NeedleTissueCoupler(
                example.model,
                _TranslatingTissue(example.model, 1.0),
                example.rigid_solver,
                interaction,
                example.needle_body,
                com,
            )
            count = 0
            for _ in range(12):
                example.state_0.clear_forces()
                wrench = np.zeros((example.model.body_count, 6), np.float32)
                wrench[example.needle_body, 2] = -0.002
                example.state_0.body_f.assign(wrench)
                coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
                example.state_0, example.state_1 = example.state_1, example.state_0
                self.assertEqual(interaction.status, "contact")
                self.assertLessEqual(coupler.iterations, 40)
                count += 1
            pose = wp.transform(*example.state_0.body_q.numpy()[example.needle_body])
            final = np.asarray(wp.transform_point(pose, wp.vec3(*com)))
            self.assertGreater(final @ transverse, 1e-6)
            self.assertEqual(count, 12)
            self.assertEqual((interaction.entry_count, interaction.exit_count, interaction.puncture_count), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
