# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Device broad-phase parity against unchanged double-precision narrow phases."""

import os
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleTissueInteraction, boundary_faces
from newton_solver_suturebot._src.geometry.needle_queries import NeedleQueryGeometry
from newton_solver_suturebot._src.insertion import _point_aabb_candidates, _surface_crossings, _vertex_bounds


class NeedleDeviceQueryTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], np.float32)
        self.tets = np.array([[0, 1, 2, 3]], np.int32)

    def test_point_candidates_include_exact_ulp_boundaries_and_overflow(self):
        q = np.vstack([self.vertices + [i * 0.02, 0, 0] for i in range(12)]).astype(np.float32)
        tets = np.arange(len(q), dtype=np.int32).reshape(-1, 4)
        query = NeedleQueryGeometry(tets, len(q), device=self.device)
        positions = wp.array(q, dtype=wp.vec3, device=self.device)
        query.update(positions)
        low, high = _vertex_bounds(q[tets].astype(float))
        points = np.vstack(
            (q, low - 1e-8, high + 1e-8, np.nextafter(low - 1e-8, -np.inf), np.nextafter(high + 1e-8, np.inf))
        )
        for point, candidate in zip(points, query.point_candidates(points), strict=True):
            self.assertIsNotNone(candidate)
            actual = candidate[_point_aabb_candidates(point, low[candidate], high[candidate])]
            np.testing.assert_array_equal(actual, _point_aabb_candidates(point, low, high))
        repeated = np.repeat(tets[:1], 5, axis=0)
        overflow = NeedleQueryGeometry(repeated, len(q), device=self.device, capacity=2)
        overflow.update(positions)
        self.assertIsNone(overflow.point_candidates([[0.001, 0.001, 0.001]])[0])

    def test_swept_candidates_preserve_exact_moving_surface_crossings(self):
        triangles = boundary_faces(self.tets)[0]
        old = self.vertices.copy()
        new = old + np.array([0, 0, 0.002], np.float32)
        positions = wp.array(old, dtype=wp.vec3, device=self.device)
        query = NeedleQueryGeometry(self.tets, len(old), device=self.device)
        query.update(positions)
        query.commit(positions)
        positions.assign(new)
        query.update(positions)
        mesh = wp.Mesh(positions, wp.array(triangles.reshape(-1), dtype=int, device=self.device))
        start = end = np.array([0.002, 0.002, 0.001])
        expected = _surface_crossings(start, end, old.astype(float), new.astype(float), triangles)
        self.assertEqual(len(expected), 1)
        candidate = query.surface_candidates(mesh, start - 1e-9, end + 1e-9, swept=True)
        self.assertIsNotNone(candidate)
        actual = _surface_crossings(start, end, old.astype(float), new.astype(float), triangles, candidates=candidate)
        self.assertEqual(actual, expected)
        # A discarded trial must not replace the accepted old positions.
        positions.assign(old)
        query.update(positions)
        mesh.refit()
        np.testing.assert_array_equal(query.previous.numpy(), old)
        candidate = query.surface_candidates(mesh, start - 1e-9, end + 1e-9, swept=True)
        self.assertEqual(
            _surface_crossings(start, end, old.astype(float), old.astype(float), triangles, candidates=candidate), []
        )

    def test_batched_embeddings_match_cpu_ties_deformation_and_extra_particles(self):
        # A shared face checks the exact first-tet tie rule; an extra unrelated
        # particle checks that geometry buffers own only the supplied material.
        vertices = np.vstack((self.vertices, [[0, 0, -0.01]])).astype(np.float32)
        tets = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], np.int32)
        samples = np.array([[0, 0, 0], [0, 0, -0.001]])
        interaction = NeedleTissueInteraction(
            vertices, tets, boundary_faces(tets)[0], [], samples, 0.0002, samples[0], device=self.device
        )
        q = np.vstack((vertices, [[100, 100, 100]])).astype(np.float32)
        state = SimpleNamespace(particle_q=wp.array(q, dtype=wp.vec3, device=self.device))
        for offset in (0, 0.0003, 0):
            q[3, 2] = 0.01 + offset
            state.particle_q.assign(q)
            host = state.particle_q.numpy().astype(float)
            points = np.array([[0.001, 0.001, 0.001], [0, 0, 0], [0.01, 0.01, 0.01], [0.002, 0.002, 0.006]])
            expected = [interaction._embedding_from_candidates(point, host, np.arange(len(tets))) for point in points]
            interaction._prepare_device_queries(state, host, points)
            for point, reference in zip(points, expected, strict=True):
                actual = interaction._locate(point, host)
                if reference is None:
                    self.assertIsNone(actual)
                else:
                    np.testing.assert_array_equal(actual[0], reference[0])
                    np.testing.assert_array_equal(actual[1], reference[1])
        interaction._refresh_surface(host)
        np.testing.assert_array_equal(interaction._tet_geometry, host[tets])

    def test_device_and_complete_cpu_queries_preserve_forces_and_handling_events(self):
        if not self.device.is_cuda:
            self.skipTest("The interaction selects device queries on CUDA")
        samples = np.zeros((11, 3))
        samples[:, 2] = np.linspace(0, -0.01, len(samples))

        def build():
            interaction = NeedleTissueInteraction(
                self.vertices,
                self.tets,
                boundary_faces(self.tets)[0],
                [],
                samples,
                0.0005,
                samples[0],
                device=self.device,
            )
            state = SimpleNamespace(
                particle_q=wp.array(self.vertices, dtype=wp.vec3, device=self.device),
                particle_qd=wp.zeros(4, dtype=wp.vec3, device=self.device),
                body_q=wp.array([[0.002, 0.002, -0.0001, 0, 0, 0, 1]], dtype=wp.transform, device=self.device),
                body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
                body_f=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
            )
            return interaction, state, wp.zeros(4, dtype=wp.vec3, device=self.device)

        optimized, reference = build(), build()
        host = reference[0]
        host._prepare_device_queries = lambda state, q, points: host._refresh_surface(q)
        path = np.r_[
            -0.0001, np.linspace(0, 0.0012, 9), np.linspace(0.00105, -0.0012, 16), np.linspace(-0.00105, 0.0012, 16)
        ]
        for step, z in enumerate(path):
            for interaction, state, forces in (optimized, reference):
                start = SimpleNamespace(particle_q=wp.clone(state.particle_q), body_q=wp.clone(state.body_q))
                moved = self.vertices.copy()
                moved[3, 2] += 1e-5 * np.sin(step * 0.1)
                state.particle_q.assign(moved)
                state.body_q.assign([[0.002, 0.002, z, 0, 0, 0, 1]])
                state.body_qd.assign([[0, 0, 0.001 if step < 10 or step >= 26 else -0.001, 0, 0, 0]])
                state.body_f.zero_()
                forces.zero_()
                interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True, start_state=start)
                interaction.accept(state, 0)
                self.assertNotEqual(interaction.status, "failed", interaction.reason)
            a, b = optimized[0], reference[0]
            np.testing.assert_allclose(optimized[2].numpy(), reference[2].numpy(), atol=2e-10, rtol=2e-6)
            np.testing.assert_allclose(optimized[1].body_f.numpy(), reference[1].body_f.numpy(), atol=2e-10, rtol=2e-6)
            self.assertEqual(
                (a.status, a.puncture_count, a.entry_count, a.exit_count, a.track_count),
                (b.status, b.puncture_count, b.entry_count, b.exit_count, b.track_count),
            )
            self.assertEqual([event["kind"] for event in a.events], [event["kind"] for event in b.events])
            np.testing.assert_array_equal(a._contact_flags.numpy(), b._contact_flags.numpy())
        self.assertEqual(optimized[0].entry_count, 2)
        self.assertEqual(optimized[0].puncture_count, 1)
        self.assertEqual(optimized[0].exit_count, 1)


if __name__ == "__main__":
    unittest.main()
