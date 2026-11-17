# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Preserve material decisions while certifying conservative swept-boundary misses."""

import os
import unittest

import numpy as np
import warp as wp

from newton_solver_suturebot import boundary_faces
from newton_solver_suturebot._src.geometry.needle_queries import NeedleQueryGeometry, _conservative_box
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import _surface_crossings
from newton_solver_suturebot._src.solvers.coupled.interface.needle_trial_forces import (
    FALLBACK_REASONS,
    located,
    tip_boundary_certificate,
    tip_location,
    tip_material_certificate,
)


# Preserve the former broadphase-only certificate as a conservative reference.
# Material masks remain identical; proven boundary misses may now avoid fallback.
@wp.kernel
def legacy_tip_certificate(
    q: wp.array[wp.vec3],
    tets: wp.array[wp.vec4i],
    fixed: wp.array[int],
    bvh: wp.uint64,
    bounds_low: wp.array[wp.vec3],
    bounds_high: wp.array[wp.vec3],
    full_mesh: wp.uint64,
    active_mesh: wp.uint64,
    samples: wp.array[wp.vec3d],
    old_tips: wp.array[wp.vec3d],
    motion: wp.array[wp.float64],
    step_flags: wp.array[int],
    spacing: wp.float64,
    capture: wp.float64,
    radius: wp.float64,
    mask: wp.array[int],
    fallback: wp.array[int],
):
    old_tip = old_tips[0]
    in_material, punctured, has_previous = step_flags[0], step_flags[1], step_flags[2]
    tip = samples[0]
    tet, bary = located(tip, q, tets, bvh, bounds_low, bounds_high)
    if has_previous == 0:
        wp.atomic_max(fallback, 0, 1)
    if in_material:
        if punctured == 0 or tet < 0:
            wp.atomic_max(fallback, 0, 1)
        if tet >= 0:
            ids = tets[tet]
            for j in range(4):
                if fixed[ids[j]] != 0 and bary[j] > wp.float64(0.05):
                    wp.atomic_max(fallback, 0, 1)
    elif punctured != 0 or tet >= 0:
        wp.atomic_max(fallback, 0, 1)
    padding = wp.vec3d(motion[0], motion[1], motion[2]) + wp.vec3d(wp.float64(1e-9))
    low, high = _conservative_box(wp.min(old_tip, tip) - padding, wp.max(old_tip, tip) + padding)
    query = wp.mesh_query_aabb(full_mesh, low, high)
    item = int(0)
    while wp.mesh_query_aabb_next(query, item):
        wp.atomic_max(fallback, 0, 1)
    query2 = wp.mesh_query_aabb(active_mesh, low, high)
    while wp.mesh_query_aabb_next(query2, item):
        wp.atomic_max(fallback, 0, 1)
    if not in_material:
        # Outside contact/puncture must retain the complete CPU authorization path.
        lo, hi = _conservative_box(tip - wp.vec3d(radius), tip + wp.vec3d(radius))
        close = wp.mesh_query_aabb(full_mesh, lo, hi)
        while wp.mesh_query_aabb_next(close, item):
            wp.atomic_max(fallback, 0, 1)
    if in_material and (mask[0] == 0 or spacing > capture):
        mask[0] = 2


class NeedleTipCertificateTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
        self.q = wp.array(
            np.array([[x, y, z] for z in (0, 0.001) for y in (0, 0.001) for x in (0, 0.001)]),
            dtype=wp.vec3,
            device=self.device,
        )
        base_cells = np.array(
            [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]], dtype=np.int32
        )
        vertices = self.q.numpy().astype(float)
        for cell in base_cells:
            if np.linalg.det((vertices[cell[1:]] - vertices[cell[0]]).T) < 0:
                cell[1], cell[2] = cell[2], cell[1]
        cells = np.vstack((base_cells, base_cells[0], [0, 0, 0, 0])).astype(np.int32)
        self.query = NeedleQueryGeometry(cells, 8, device=self.device)
        self.query.update(self.q)
        faces, _ = boundary_faces(base_cells)
        self.faces_host = faces
        self.faces = wp.array(faces, dtype=wp.vec3i, device=self.device)
        self.mapping = wp.array(np.arange(8), dtype=int, device=self.device)
        self.reference = wp.array(vertices, dtype=wp.vec3d, device=self.device)
        self.previous_q = wp.clone(self.q)
        self.poses = wp.array([wp.transform_identity()], dtype=wp.transform, device=self.device)
        self.mesh = wp.Mesh(points=self.q, indices=wp.array(faces.ravel(), dtype=int, device=self.device))
        self.fixed = wp.zeros(8, dtype=int, device=self.device)
        self.point = wp.empty(1, dtype=wp.vec3d, device=self.device)
        self.previous = wp.empty_like(self.point)
        self.motion = wp.zeros(3, dtype=wp.float64, device=self.device)
        self.flags = wp.empty(3, dtype=int, device=self.device)
        self.cell = wp.empty(1, dtype=int, device=self.device)
        self.bary = wp.empty(1, dtype=wp.vec4d, device=self.device)
        self.mask = wp.empty(1, dtype=int, device=self.device)
        # The certificates record the reason of a request beside the request itself.
        self.fallback = wp.zeros(len(FALLBACK_REASONS), dtype=int, device=self.device)
        self.retained_opening = wp.zeros(1, dtype=int, device=self.device)

    def compare(
        self,
        point,
        flags,
        expected,
        *,
        old_point=None,
        previous_q=None,
        fixed=False,
        initial_mask=0,
        spacing=0.00025,
        legacy_expected=None,
        has_openings=True,
    ):
        self.point.assign(np.asarray([point], dtype=np.float64))
        self.previous.assign(np.asarray([point if old_point is None else old_point], dtype=np.float64))
        current = self.q.numpy()
        old_q = current if previous_q is None else np.asarray(previous_q, dtype=np.float32)
        self.previous_q.assign(old_q)
        self.motion.assign(np.max(np.abs(current.astype(float) - old_q.astype(float)), axis=0))
        self.fixed.assign(np.asarray([int(fixed), 0, 0, 0, 0, 0, 0, 0], dtype=np.int32))
        self.flags.assign(np.asarray(flags, dtype=np.int32))
        outputs = []
        g = self.query
        for split in (False, True):
            self.mask.fill_(initial_mask)
            self.fallback.zero_()
            if split:
                wp.launch(
                    tip_location,
                    1,
                    [self.point, self.q, g.tetrahedra, g.bvh.id, g.low, g.high, self.cell, self.bary],
                    device=self.device,
                )
                wp.launch(
                    tip_material_certificate,
                    1,
                    [
                        self.cell,
                        self.bary,
                        g.tetrahedra,
                        self.fixed,
                        self.flags,
                        self.retained_opening,
                        spacing,
                        0.00056,
                        self.mask,
                        self.fallback,
                    ],
                    device=self.device,
                )
                wp.launch(
                    tip_boundary_certificate,
                    3,
                    [
                        self.mesh.id,
                        self.mesh.id,
                        self.point,
                        self.previous,
                        self.motion,
                        self.flags,
                        0.00028,
                        int(has_openings),
                        self.faces,
                        self.faces,
                        self.mapping,
                        self.reference,
                        self.previous_q,
                        self.q,
                        self.poses,
                        0,
                        wp.vec3d(*point),
                        self.fallback,
                    ],
                    device=self.device,
                )
            else:
                wp.launch(
                    legacy_tip_certificate,
                    1,
                    [
                        self.q,
                        g.tetrahedra,
                        self.fixed,
                        g.bvh.id,
                        g.low,
                        g.high,
                        self.mesh.id,
                        self.mesh.id,
                        self.point,
                        self.previous,
                        self.motion,
                        self.flags,
                        spacing,
                        0.00056,
                        0.00028,
                        self.mask,
                        self.fallback,
                    ],
                    device=self.device,
                )
            outputs.append((int(self.mask.numpy()[0]), int(self.fallback.numpy()[0])))
        self.assertEqual(outputs[0][0], outputs[1][0])
        self.assertEqual(outputs[0], expected if legacy_expected is None else legacy_expected)
        self.assertEqual(outputs[1], expected)

    def test_material_outside_authorization_and_moving_boundary_cases(self):
        inside = [0.0002, 0.0002, 0.0002]
        outside = [0.002, 0.002, 0.002]
        cases = (
            ("inside", inside, [1, 1, 1], (2, 0), {}),
            ("retained_mask", inside, [1, 1, 1], (4, 0), dict(initial_mask=4)),
            ("large_spacing", inside, [1, 1, 1], (2, 0), dict(initial_mask=4, spacing=0.001)),
            ("outside", outside, [0, 0, 1], (0, 0), {}),
            ("near_wall", [0.0002, 0.0002, -0.0001], [0, 0, 1], (0, 1), {}),
            ("unauthorized_inside", inside, [1, 0, 1], (2, 1), {}),
            ("unexpected_entry", inside, [0, 0, 1], (0, 1), {}),
            ("unexpected_exit", outside, [1, 1, 1], (2, 1), {}),
            ("missing_previous", inside, [1, 1, 0], (2, 1), {}),
            ("fixed_support", inside, [1, 1, 1], (2, 1), dict(fixed=True)),
            ("tip_sweep", inside, [1, 1, 1], (2, 1), dict(old_point=outside)),
        )
        for name, point, flags, expected, options in cases:
            with self.subTest(name=name):
                self.compare(point, flags, expected, **options)

    def test_moving_surface_crossing_requires_fallback(self):
        tip = np.array([0.0002, 0.0002, 0.0002])
        current = self.q.numpy().astype(float)
        previous = (current + [0.001, 0, 0]).astype(np.float32).astype(float)
        self.assertTrue(_surface_crossings(tip, tip, previous, current, self.faces_host))
        self.compare(tip, [1, 1, 1], (2, 1), previous_q=previous)

    def test_virgin_contact_skips_opening_proximity_but_preserves_crossing_guards(self):
        outside = [0.0002, 0.0002, -0.0001]
        # A stationary external tip has no reuse decision when no openings
        # exist. Force-threshold authorization is checked by force_flags.
        self.compare(outside, [0, 0, 1], (0, 0), has_openings=False, legacy_expected=(0, 1))
        inside = [0.0002, 0.0002, 0.0002]
        self.compare(inside, [0, 0, 1], (0, 1), old_point=outside, has_openings=False)

    def test_distant_vertex_motion_broadphase_hit_is_certified_clear(self):
        tip = np.array([0.0002, 0.0002, 0.0002])
        current = self.q.numpy().astype(float)
        previous = current.copy()
        previous[7, 0] += 0.001
        previous = previous.astype(np.float32).astype(float)
        # Motion elsewhere expands the global broadphase box across every face.
        # The actual triangles never sweep across this interior stationary tip.
        self.assertFalse(_surface_crossings(tip, tip, previous, current, self.faces_host))
        self.compare(tip, [1, 1, 1], (2, 0), previous_q=previous, legacy_expected=(2, 1))

    def test_tie_order_and_degenerate_tetrahedra(self):
        # Strictly inside the byte-identical cells 0 and 6: shared faces can
        # produce slightly different minima from different inverse matrices.
        self.compare([0.0007, 0.0004, 0.0002], [1, 1, 1], (2, 0))
        self.assertEqual(int(self.cell.numpy()[0]), 0)
        np.testing.assert_allclose(self.bary.numpy()[0], [0.3, 0.3, 0.2, 0.2], atol=3e-8)
        self.q.zero_()
        self.query.update(self.q)
        self.mesh.refit()
        self.compare([0, 0, 0], [1, 1, 1], (2, 1))
        self.assertEqual(int(self.cell.numpy()[0]), -1)
        np.testing.assert_array_equal(self.bary.numpy(), 0)


if __name__ == "__main__":
    unittest.main()
