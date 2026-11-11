# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""The optional least surface stiffness of one shaft sample."""

import os
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleTissueInteraction, boundary_faces

RADIUS = 0.0005
STIFFNESS = 80.0
DAMPING = 0.025


class TestNeedleSurfaceSampleStiffness(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        # One tetrahedron above z = 0; the needle meets its lower face from below.
        self.vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], dtype=float)
        self.tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
        self.triangles = boundary_faces(self.tets)[0]

    def build(self, samples, *, fixed=(), **kwargs):
        interaction = NeedleTissueInteraction(
            self.vertices,
            self.tets,
            self.triangles,
            fixed,
            samples,
            RADIUS,
            samples[0],
            device=self.device,
            stiffness=STIFFNESS,
            damping=DAMPING,
            **kwargs,
        )
        state = SimpleNamespace(
            particle_q=wp.array(self.vertices, dtype=wp.vec3, device=self.device),
            particle_qd=wp.zeros(len(self.vertices), dtype=wp.vec3, device=self.device),
            body_q=wp.zeros(1, dtype=wp.transform, device=self.device),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
        )
        forces = wp.zeros(len(self.vertices), dtype=wp.vec3, device=self.device)
        return interaction, state, forces

    def force(self, interaction, state, forces, gap, *, speed=0.0):
        """Normal force on the needle with its nearest samples ``gap`` below the face, closing at ``speed``."""
        state.body_q.assign(np.array([[0.002, 0.002, -gap, 0, 0, 0, 1]], np.float32))
        state.body_qd.assign(np.array([[0, 0, speed, 0, 0, 0]], np.float32))
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005)
        body = state.body_f.numpy()[0]
        np.testing.assert_allclose(body[:3] + forces.numpy().sum(axis=0), 0, atol=1e-8)
        return -float(body[2])

    @staticmethod
    def standing(count=11):
        """Tip far below, tail at the top: only the tail sample is within one radius of the face."""
        samples = np.zeros((count, 3))
        samples[:, 2] = np.linspace(-0.01, 0, count)
        return samples

    @staticmethod
    def lying(count=9):
        """All samples at one height, 0.5 mm apart along x."""
        samples = np.zeros((count, 3))
        samples[:, 0] = np.linspace(0, 0.004, count)
        return samples

    def test_default_keeps_the_shared_shaft_stiffness(self):
        samples = self.standing()
        default, state, forces = self.build(samples)
        explicit = self.build(samples, surface_sample_stiffness=None)[0]
        lengths = np.linalg.norm(np.diff(samples, axis=0), axis=1)
        shares = np.r_[lengths[0] / 2, (lengths[:-1] + lengths[1:]) / 2, lengths[-1] / 2]
        shares[0] = 0.0
        shares /= shares.sum()
        shares[0] = 1.0
        np.testing.assert_array_equal(default._weights.numpy(), shares.astype(np.float32))
        np.testing.assert_array_equal(explicit._weights.numpy(), default._weights.numpy())
        self.assertIsNone(default.surface_sample_stiffness)
        penetration = 0.0003
        # The tail holds half a segment of the nine and a half that share the shaft stiffness.
        self.assertAlmostEqual(shares[-1], 0.5 / 9.5, places=12)
        self.assertAlmostEqual(
            self.force(default, state, forces, RADIUS - penetration), STIFFNESS * shares[-1] * penetration, places=8
        )

    def test_one_end_sample_carries_the_stated_stiffness_and_damping(self):
        sample_stiffness = 30.0
        interaction, state, forces = self.build(self.standing(), surface_sample_stiffness=sample_stiffness)
        self.assertEqual(interaction.surface_sample_stiffness, sample_stiffness)
        penetration = 0.0003
        static = self.force(interaction, state, forces, RADIUS - penetration)
        self.assertAlmostEqual(static, sample_stiffness * penetration, places=7)
        self.assertEqual(interaction.last_tip_force, 0.0)
        # Damping is raised in the same ratio as stiffness and still acts in compression only.
        speed = 0.01
        closing = self.force(interaction, state, forces, RADIUS - penetration, speed=speed)
        ratio = sample_stiffness / STIFFNESS
        self.assertAlmostEqual(closing - static, ratio * DAMPING * penetration / RADIUS * speed, places=7)
        opening = self.force(interaction, state, forces, RADIUS - penetration, speed=-speed)
        self.assertAlmostEqual(opening, static, places=7)

    def test_every_lying_sample_is_at_least_this_stiff_and_a_larger_share_is_kept(self):
        samples = self.lying()
        penetration = 0.0002
        shared = self.build(samples)
        # Shaft shares sum to one and the tip adds its own stiffness.
        self.assertAlmostEqual(
            self.force(shared[0], *shared[1:], RADIUS - penetration), 2 * STIFFNESS * penetration, places=7
        )
        sample_stiffness = 30.0
        raised = self.build(samples, surface_sample_stiffness=sample_stiffness)
        expected = (STIFFNESS + (len(samples) - 1) * sample_stiffness) * penetration
        self.assertAlmostEqual(self.force(raised[0], *raised[1:], RADIUS - penetration), expected, places=6)
        # A floor below every sample's share changes nothing.
        low = self.build(samples, surface_sample_stiffness=1.0)
        np.testing.assert_array_equal(low[0]._weights.numpy(), shared[0]._weights.numpy())

    def test_tip_stiffness_and_puncture_force_are_unchanged(self):
        samples = np.zeros((11, 3))
        samples[:, 2] = np.linspace(0, -0.01, 11)  # tip at the top, as in the insertion tests
        penetration = 0.0004
        tip_forces = []
        for sample_stiffness in (None, 400.0):
            interaction, state, forces = self.build(samples, surface_sample_stiffness=sample_stiffness)
            self.force(interaction, state, forces, RADIUS - penetration)
            tip_forces.append(interaction.last_tip_force)
            self.assertEqual(interaction._weights.numpy()[0], 1.0)
        self.assertAlmostEqual(tip_forces[0], STIFFNESS * penetration, places=7)
        self.assertEqual(tip_forces[0], tip_forces[1])

    def test_fixed_support_carries_one_sample_without_the_failure_flag(self):
        weight = 0.003
        # Set down on one sample without speed, the needle dips to twice its static depth: one radius.
        sample_stiffness = 2 * weight / RADIUS
        interaction, state, forces = self.build(
            self.standing(), fixed=(0, 1, 2, 3), surface_sample_stiffness=sample_stiffness
        )
        carried = self.force(interaction, state, forces, RADIUS / 2)
        self.assertAlmostEqual(carried, weight, places=7)
        self.assertNotEqual(interaction.status, "failed")
        np.testing.assert_allclose(np.linalg.norm(interaction.support_force), weight, rtol=1e-5)

    def test_setter_switches_the_law_in_the_same_device_array(self):
        interaction, state, forces = self.build(self.standing())
        storage = interaction._weights.ptr
        shared = interaction._weights.numpy().copy()
        penetration = 0.0003
        before = self.force(interaction, state, forces, RADIUS - penetration)
        interaction.set_surface_sample_stiffness(30.0)
        self.assertEqual(interaction.surface_sample_stiffness, 30.0)
        self.assertAlmostEqual(
            self.force(interaction, state, forces, RADIUS - penetration), 30.0 * penetration, places=7
        )
        interaction.set_surface_sample_stiffness(None)
        self.assertIsNone(interaction.surface_sample_stiffness)
        self.assertEqual(self.force(interaction, state, forces, RADIUS - penetration), before)
        np.testing.assert_array_equal(interaction._weights.numpy(), shared)
        # Captured trials read this array by its storage; it must not be replaced.
        self.assertEqual(interaction._weights.ptr, storage)

    def test_invalid_values_are_rejected(self):
        for value in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.build(self.standing(), surface_sample_stiffness=value)
            interaction = self.build(self.standing())[0]
            with self.subTest(setter=value), self.assertRaises(ValueError):
                interaction.set_surface_sample_stiffness(value)


if __name__ == "__main__":
    unittest.main()
