# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Geometry-cache parity and ownership across candidate and accepted states."""

import os
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleTissueInteraction, boundary_faces


class NeedleInsertionGeometryCacheTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], dtype=float)
        self.tetrahedra = np.array([[0, 1, 2, 3]], dtype=np.int32)
        # The second tetrahedron includes one active vertex. Its boundary has
        # both mixed active/inactive faces and a completely inactive face.
        self.reference = np.vstack((self.vertices, [[0.02, 0, 0], [0.02, 0.01, 0], [0.02, 0, 0.01]]))
        self.surface = boundary_faces(np.array([[0, 1, 2, 3], [1, 4, 5, 6]], dtype=np.int32))[0]
        self.mapping = np.array([0, 1, 2, 3, -1, -1, -1], dtype=np.int32)

    def build(self, *, inactive=False):
        samples = np.array([[0, 0, 0], [0, 0, -0.01]], dtype=float)
        mapping = np.full(len(self.mapping), -1, dtype=np.int32) if inactive else self.mapping
        return NeedleTissueInteraction(
            self.vertices,
            self.tetrahedra,
            boundary_faces(self.tetrahedra)[0],
            [1],
            samples,
            0.0005,
            samples[0],
            reference_vertices=self.reference,
            reference_triangles=self.surface,
            reference_to_active=mapping,
            device=self.device,
        )

    def assert_full_recompute(self, interaction, q, reference=None):
        # Independent oracle: reconstruct all reference triangles from their
        # original coordinates and the current nodal state, without using any
        # cache or assuming which reference faces can move.
        # Device queries leave the complete CPU cache lazy; materialize it
        # explicitly when checking that cache's ownership and exact contents.
        interaction._refresh_surface(q)
        points = (self.reference if reference is None else reference).copy()
        live = interaction.mapping >= 0
        points[live] = q[interaction.mapping[live]]
        geometry = points[self.surface]
        np.testing.assert_array_equal(interaction.surface_vertices, points)
        np.testing.assert_array_equal(interaction._surface_geometry, geometry)
        np.testing.assert_array_equal(interaction._surface_bounds[0], geometry.min(axis=1))
        np.testing.assert_array_equal(interaction._surface_bounds[1], geometry.max(axis=1))
        tet_geometry = q[self.tetrahedra]
        np.testing.assert_array_equal(interaction._tet_geometry, tet_geometry)
        np.testing.assert_array_equal(interaction._tet_bounds[0], tet_geometry.min(axis=1))
        np.testing.assert_array_equal(interaction._tet_bounds[1], tet_geometry.max(axis=1))
        active_geometry = q[boundary_faces(self.tetrahedra)[0]]
        np.testing.assert_array_equal(interaction._active_bounds[0], active_geometry.min(axis=1))
        np.testing.assert_array_equal(interaction._active_bounds[1], active_geometry.max(axis=1))
        ids, bary = interaction._locate(q[self.tetrahedra[0]].mean(axis=0), q)
        np.testing.assert_array_equal(ids, self.tetrahedra[0])
        np.testing.assert_allclose(bary, np.full(4, 0.25), atol=1e-14, rtol=0)

    def test_full_geometry_parity_for_in_place_deformation_and_rollback(self):
        interaction = self.build()
        # Extra particles belong to another object and are not referenced by
        # this interaction's tetrahedra or reference-surface mapping.
        q = np.vstack((self.vertices, [[0.03, 0.04, 0.05], [0.06, 0.07, 0.08]]))
        original = q.copy()
        for stage in ("initial", "deformed", "unchanged", "extra particles", "rollback"):
            with self.subTest(stage=stage):
                if stage == "deformed":
                    # Even a supported node must use the actual supplied q.
                    # Fixed status must not classify its faces as immutable.
                    q[1] += [0.0003, -0.0002, 0.0001]
                    q[3] += [-0.0001, 0.0002, 0.0004]
                elif stage == "extra particles":
                    q[4:] += [1, 2, 3]
                elif stage == "rollback":
                    q[:] = original
                interaction._refresh_surface(q)
                self.assert_full_recompute(interaction, q)
                self.assertFalse(np.shares_memory(interaction._query_positions, q))
                np.testing.assert_array_equal(interaction._query_positions, q)

    def test_accepted_geometry_stays_owned_during_trials_and_rollback(self):
        interaction = self.build()
        q = np.vstack((self.vertices, [[0.03, 0.04, 0.05]]))
        state = SimpleNamespace(
            particle_q=wp.array(q, dtype=wp.vec3, device=self.device),
            particle_qd=wp.zeros(len(q), dtype=wp.vec3, device=self.device),
            body_q=wp.array([[0.002, 0.002, -0.01, 0, 0, 0, 1]], dtype=wp.transform, device=self.device),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
        )
        forces = wp.zeros(len(q), dtype=wp.vec3, device=self.device)
        interaction._refresh_surface(state.particle_q.numpy().astype(float))
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True)
        interaction.accept(state, 0)
        self.assertNotEqual(interaction.status, "failed", interaction.reason)
        accepted = (
            interaction._previous_q,
            interaction._previous_surface,
            *interaction._previous_surface_bounds,
            *interaction._previous_active_bounds,
        )
        snapshots = tuple(array.copy() for array in accepted)
        status, events = interaction.status, interaction.events
        original = state.particle_q.numpy().copy()
        candidate = original.copy()
        candidate[1] += [0.0003, -0.0002, 0.0001]
        candidate[3] += [-0.0001, 0.0002, 0.0004]
        for positions in (candidate, original):
            state.particle_q.assign(positions)
            state.body_f.zero_()
            forces.zero_()
            interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True, implicit=True)
            self.assert_full_recompute(interaction, state.particle_q.numpy().astype(float))
            current = (
                interaction._previous_q,
                interaction._previous_surface,
                *interaction._previous_surface_bounds,
                *interaction._previous_active_bounds,
            )
            for actual, previous, snapshot in zip(current, accepted, snapshots, strict=True):
                self.assertIs(actual, previous)
                np.testing.assert_array_equal(actual, snapshot)
            for previous, latest in zip(interaction._previous_surface_bounds, interaction._surface_bounds, strict=True):
                self.assertFalse(np.shares_memory(previous, latest))
            self.assertFalse(np.shares_memory(interaction._previous_surface, interaction.surface_vertices))
            self.assertEqual(interaction.status, status)
            self.assertIs(interaction.events, events)
            self.assertEqual((interaction.puncture_count, interaction.entry_count, interaction.track_count), (0, 0, 0))

    def test_inactive_reference_is_owned_and_independent_of_particle_motion(self):
        original_reference = self.reference.copy()
        interaction = self.build(inactive=True)
        self.assertFalse(np.shares_memory(interaction.surface_vertices, self.reference))
        # Construction owns the inactive geometry; modifying the caller's
        # source array does not move the contact surface.
        self.reference += [0.1, 0.2, 0.3]
        q = self.vertices.copy()
        for displacement in ([0.0003, 0.0002, 0.0001], [-0.0001, 0.0002, 0.0004]):
            q += displacement
            interaction._refresh_surface(q)
            self.assert_full_recompute(interaction, q, reference=original_reference)


if __name__ == "__main__":
    unittest.main()
