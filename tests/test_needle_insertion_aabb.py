# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact geometry broad-phase parity, including floating-point inclusion ties."""

import unittest

import numpy as np

from newton_solver_suturebot._src.solvers.coupled.interface.insertion import (
    _point_aabb_candidates,
    _segment_aabb_candidates,
    _surface_crossings,
    _vertex_bounds,
)


def point_reference(point, low, high, padding=1e-8):
    return np.flatnonzero(np.all((point >= low - padding) & (point <= high + padding), axis=1))


def segment_reference(start, end, low, high):
    return np.flatnonzero(
        np.all((high >= np.minimum(start, end) - 1e-9) & (low <= np.maximum(start, end) + 1e-9), axis=1)
    )


class NeedleInsertionAabbTests(unittest.TestCase):
    def test_vertex_bounds_equal_reductions_for_empty_and_strided_arrays(self):
        generator = np.random.default_rng(8721)
        for dtype in (np.float32, np.float64):
            for vertices in (3, 4):
                for count in (0, 1, 21):
                    source = generator.uniform(-0.05, 0.05, (count * 2, vertices, 3)).astype(dtype)
                    for points in (source, source[::2], source[:, ::-1]):
                        low, high = _vertex_bounds(points)
                        np.testing.assert_array_equal(low, points.min(axis=1))
                        np.testing.assert_array_equal(high, points.max(axis=1))
                        self.assertEqual(low.dtype, points.dtype)
                        self.assertEqual(high.dtype, points.dtype)
                        self.assertFalse(np.shares_memory(points, low))
                        self.assertFalse(np.shares_memory(points, high))

    def test_sparse_dense_and_empty_candidates_preserve_order(self):
        generator = np.random.default_rng(67391)
        for dtype in (np.float32, np.float64):
            for count in (0, 1, 113):
                centers = generator.uniform(-0.05, 0.05, (count, 3)).astype(dtype)
                for width in (0.0005, 0.005, 0.2):
                    low, high = centers - dtype(width), centers + dtype(width)
                    point, end = np.array([[0.03, -0.04, 0.001], [0.031, -0.039, 0.002]], dtype=dtype)
                    for padding in (1e-8, 0.00028, 0.0005):
                        np.testing.assert_array_equal(
                            _point_aabb_candidates(point, low, high, padding),
                            point_reference(point, low, high, padding),
                        )
                    np.testing.assert_array_equal(
                        _segment_aabb_candidates(point, end, low, high), segment_reference(point, end, low, high)
                    )

    def test_candidates_match_original_arithmetic_at_ulp_boundaries(self):
        for dtype in (np.float32, np.float64):
            for coordinate in (0.0, -0.0, 1e-8, -1e-8, 0.03, -0.04, 1.0, -1.0):
                point = np.full(3, coordinate, dtype=dtype)
                for axis in range(3):
                    for epsilon in (1e-8, 1e-9, 0.00028, 0.0005):
                        for lower in (True, False):
                            tie = dtype(point[axis] + dtype(epsilon) if lower else point[axis] - dtype(epsilon))
                            values = [np.nextafter(tie, -np.inf), tie, np.nextafter(tie, np.inf)]
                            low = np.tile(point - dtype(0.1), (3, 1))
                            high = np.tile(point + dtype(0.1), (3, 1))
                            (low if lower else high)[:, axis] = values
                            # Distant padding forces the sparse branch; no padding
                            # exercises the dense branch with the same inclusion ties.
                            for pad in (0, 100):
                                padded_low = np.concatenate((low, np.full((pad, 3), 10, dtype=dtype)))
                                padded_high = np.concatenate((high, np.full((pad, 3), 11, dtype=dtype)))
                                order = np.arange(len(padded_low))[::-1]
                                a, b = padded_low[order], padded_high[order]
                                if epsilon != 1e-9:
                                    actual = _point_aabb_candidates(point, a, b, epsilon)
                                    expected = point_reference(point, a, b, epsilon)
                                else:
                                    actual = _segment_aabb_candidates(point, point, a, b)
                                    expected = segment_reference(point, point, a, b)
                                np.testing.assert_array_equal(actual, expected)
                                self.assertEqual(actual.dtype, expected.dtype)

    def test_swept_crossings_match_cached_and_uncached_bounds(self):
        # The point crosses two parallel faces only 20 micrometres apart.
        # Most faces have disjoint x bounds, so candidate pruning must retain
        # both nearby crossings and their original face indices.
        triangle = np.array([[-0.001, -0.001, 0.0], [0.001, -0.001, 0.0], [0.0, 0.001, 0.0]])
        geometry = np.array([triangle + (index * 0.01, 0, 0) for index in range(20)])
        geometry[7] = triangle + (0, 0, 0.00002)
        old = geometry.reshape(-1, 3)
        new = old + np.array([0.0, 0.0, 0.000001])
        triangles = np.arange(len(old), dtype=np.int32).reshape(-1, 3)
        start, end = np.array([0.0, 0.0, -0.00001]), np.array([0.0, 0.0, 0.00004])
        old_low, old_high = _vertex_bounds(old[triangles])
        new_low, new_high = _vertex_bounds(new[triangles])
        bounds = np.minimum(old_low, new_low), np.maximum(old_high, new_high)
        uncached = _surface_crossings(start, end, old, new, triangles)
        cached = _surface_crossings(start, end, old, new, triangles, bounds=bounds)
        self.assertEqual(cached, uncached)
        self.assertEqual([event[2] for event in cached], [0, 7])
        np.testing.assert_allclose([event[0] for event in cached], [1 / 4.9, 3 / 4.9], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
