# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Bitwise geometry parity with the original serial double-precision predicates."""

import os
import unittest

import numpy as np
import warp as wp

from newton_solver_suturebot._src.geometry.needle_queries import NeedleQueryGeometry, _conservative_box
from newton_solver_suturebot._src.needle_trial_forces import embedding, located, track_geometry, track_segments


@wp.kernel
def serial_track_geometry(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4d],
    q: wp.array[wp.vec3],
    samples: wp.array[wp.vec3d],
    arc: wp.array[wp.float64],
    engaged: wp.array[int],
    points: wp.array[wp.vec3d],
    projected_arc: wp.array[wp.float64],
    associated: wp.array[int],
):
    i = wp.tid()
    point = wp.vec3d(wp.float64(0))
    for j in range(4):
        point += bary[i][j] * wp.vec3d(q[nodes[i][j]])
    points[i] = point
    best = int(0)
    best_fraction = wp.float64(0)
    distance = wp.float64(1e30)
    for j in range(samples.shape[0] - 1):
        direction = samples[j + 1] - samples[j]
        length = arc[j + 1] - arc[j]
        fraction = wp.dot(point - samples[j], direction) / (length * length)
        delta = samples[j] + wp.clamp(fraction, wp.float64(0), wp.float64(1)) * direction - point
        squared = wp.dot(delta, delta)
        if squared < distance:
            best, best_fraction, distance = j, fraction, squared
    projected_arc[i] = arc[best] + wp.clamp(best_fraction, wp.float64(0), wp.float64(1)) * (arc[best + 1] - arc[best])
    covered = not (
        (best == 0 and best_fraction < wp.float64(0))
        or (best == samples.shape[0] - 2 and best_fraction > wp.float64(1))
    )
    associated[i] = int(engaged[i] != 0 and covered)


@wp.func
def rebuilt_located(point: wp.vec3d, q: wp.array[wp.vec3], tets: wp.array[wp.vec4i], bvh: wp.uint64):
    eps = wp.vec3d(wp.float64(1e-8))
    low, high = _conservative_box(point - eps, point + eps)
    query = wp.bvh_query_aabb(bvh, low, high)
    item, best = int(0), int(-1)
    score = wp.float64(-1e30)
    chosen = wp.vec4d(wp.float64(0))
    while wp.bvh_query_next(query, item):
        ids = tets[item]
        lo, hi = wp.vec3d(q[ids[0]]), wp.vec3d(q[ids[0]])
        for j in range(1, 4):
            lo = wp.min(lo, wp.vec3d(q[ids[j]]))
            hi = wp.max(hi, wp.vec3d(q[ids[j]]))
        keep = bool(True)
        for axis in range(3):
            if point[axis] < lo[axis] - wp.float64(1e-8) or point[axis] > hi[axis] + wp.float64(1e-8):
                keep = False
        if keep:
            bary, det = embedding(point, ids, q)
            minimum = wp.min(wp.min(bary[0], bary[1]), wp.min(bary[2], bary[3]))
            if wp.abs(det) > wp.float64(1e-18) and minimum >= wp.float64(-1e-6):
                if minimum > score or (minimum == score and (best < 0 or item < best)):
                    best, score, chosen = item, minimum, bary
    return best, chosen


@wp.kernel
def location_probe(
    points: wp.array[wp.vec3d],
    q: wp.array[wp.vec3],
    tets: wp.array[wp.vec4i],
    bvh: wp.uint64,
    low: wp.array[wp.vec3],
    high: wp.array[wp.vec3],
    ids: wp.array2d[int],
    bary: wp.array2d[wp.vec4d],
):
    i = wp.tid()
    a, b = rebuilt_located(points[i], q, tets, bvh)
    c, d = located(points[i], q, tets, bvh, low, high)
    ids[i, 0] = a
    ids[i, 1] = c
    bary[i, 0] = b
    bary[i, 1] = d


@wp.kernel
def bounds_probe(
    points: wp.array[wp.vec3d],
    q: wp.array[wp.vec3],
    tets: wp.array[wp.vec4i],
    low: wp.array[wp.vec3],
    high: wp.array[wp.vec3],
    result: wp.array2d[int],
):
    i, k = wp.tid()
    ids = tets[k]
    lo, hi = wp.vec3d(q[ids[0]]), wp.vec3d(q[ids[0]])
    for j in range(1, 4):
        lo = wp.min(lo, wp.vec3d(q[ids[j]]))
        hi = wp.max(hi, wp.vec3d(q[ids[j]]))
    cached_lo, cached_hi = wp.vec3d(low[k]), wp.vec3d(high[k])
    keep, cached_keep = bool(True), bool(True)
    for axis in range(3):
        if points[i][axis] < lo[axis] - wp.float64(1e-8) or points[i][axis] > hi[axis] + wp.float64(1e-8):
            keep = False
        if points[i][axis] < cached_lo[axis] - wp.float64(1e-8) or points[i][axis] > cached_hi[axis] + wp.float64(1e-8):
            cached_keep = False
    result[i, k] = int(keep != cached_keep)


class NeedleTrialGeometryTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))

    def arrays(self, points, samples, bary=None):
        points = np.asarray(points, dtype=np.float32)
        count = len(points)
        nodes = np.arange(4 * count, dtype=np.int32).reshape(count, 4)
        if bary is None:
            bary = np.tile([1.0, 0.0, 0.0, 0.0], (count, 1))
        q = np.repeat(points, 4, axis=0)
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1))]
        inputs = [
            wp.array(nodes, dtype=wp.vec4i, device=self.device),
            wp.array(bary, dtype=wp.vec4d, device=self.device),
            wp.array(q, dtype=wp.vec3, device=self.device),
            wp.array(samples, dtype=wp.vec3d, device=self.device),
            wp.array(arc, dtype=wp.float64, device=self.device),
            wp.array(np.arange(count) % 3 != 0, dtype=int, device=self.device),
        ]
        outputs = [
            [wp.empty(count, dtype=dtype, device=self.device) for dtype in (wp.vec3d, wp.float64, int)]
            for _ in range(2)
        ]
        scratch = wp.empty((count, len(samples) - 1), dtype=wp.vec2d, device=self.device)
        return inputs, outputs, scratch

    def launch(self, inputs, outputs, scratch):
        count = len(inputs[0])
        wp.launch(serial_track_geometry, count, inputs + outputs[0], device=self.device)
        wp.launch(track_segments, scratch.shape, inputs[:5] + [outputs[1][0], scratch], device=self.device)
        wp.launch(track_geometry, count, [scratch, inputs[4], inputs[5], *outputs[1][1:]], device=self.device)

    def assert_equal(self, outputs):
        for original, parallel in zip(*outputs, strict=True):
            # The kernels must keep ties/roundoff, not merely similar distances.
            self.assertEqual(original.numpy().tobytes(), parallel.numpy().tobytes())

    def test_parallel_projection_matches_serial_nonuniform_curved_segments(self):
        rng = np.random.default_rng(1296)
        for count, segments, scale in ((1, 1, 1.0), (13, 190, 0.01), (37, 57, 10.0)):
            samples = np.cumsum(rng.normal(size=(segments + 1, 3)), axis=0) * scale
            points = rng.normal(size=(count, 3)) * scale
            bary = rng.dirichlet(np.ones(4), count)
            inputs, outputs, scratch = self.arrays(points, samples, bary)
            q = inputs[2].numpy()
            q += rng.normal(size=q.shape).astype(np.float32) * np.float32(scale * 0.1)
            inputs[2].assign(q)
            self.launch(inputs, outputs, scratch)
            self.assert_equal(outputs)

    def test_first_segment_ties_and_finite_endpoint_coverage(self):
        samples = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        inputs, outputs, scratch = self.arrays(
            [[0.5, 0.5, 0.0], [-0.5, 0.0, 0.0], [-0.5, 1.0, 0.0], [1.0, 0.0, 0.0]], samples
        )
        inputs[5].fill_(1)
        self.launch(inputs, outputs, scratch)
        self.assert_equal(outputs)
        np.testing.assert_array_equal(outputs[1][1].numpy(), [0.5, 0.0, 3.0, 1.0])
        np.testing.assert_array_equal(outputs[1][2].numpy(), [1, 0, 0, 1])

    def test_capture_reuses_scratch_for_changed_geometry_and_engagement(self):
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture")
        samples = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        inputs, outputs, scratch = self.arrays([[0.1, 0.2, 0.0], [0.5, 0.5, 0.0]], samples)
        self.launch(inputs, outputs, scratch)
        pointer = scratch.ptr
        with wp.ScopedCapture(device=self.device) as capture:
            self.launch(inputs, outputs, scratch)
        for delta in (0.0, 0.3, -0.7):
            inputs[2].assign(
                np.repeat(np.array([[0.1 + delta, 0.2, 0.0], [0.5, 0.5 + delta, 0.0]], np.float32), 4, axis=0)
            )
            inputs[5].assign(np.array([int(delta >= 0), int(delta <= 0)], np.int32))
            wp.capture_launch(capture.graph)
            self.assert_equal(outputs)
            self.assertEqual(scratch.ptr, pointer)

    def test_cached_bounds_match_rebuild_at_double_padding_thresholds(self):
        # Repeated tetrahedra also exercise the original lowest-ID tie selection.
        reference = np.array(
            [[0.03, -0.04, 0.001], [0.04, -0.04, 0.001], [0.03, -0.03, 0.001], [0.03, -0.04, 0.011]], np.float32
        )
        tets = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], np.int32)
        q = wp.array(reference, dtype=wp.vec3, device=self.device)
        query = NeedleQueryGeometry(tets, len(reference), device=self.device)
        for deformation in (0.0, 1e-4, -2e-4):
            values = reference.copy()
            values[1, 1] += deformation
            q.assign(values)
            query.update(q)
            low, high = values.min(axis=0).astype(float), values.max(axis=0).astype(float)
            points = [values.mean(axis=0).astype(float), *values.astype(float)]
            for axis in range(3):
                for edge in (low[axis] - 1e-8, high[axis] + 1e-8):
                    for coordinate in (np.nextafter(edge, -np.inf), edge, np.nextafter(edge, np.inf)):
                        point = (low + high) / 2
                        point[axis] = coordinate
                        points.append(point)
            points = wp.array(np.asarray(points), dtype=wp.vec3d, device=self.device)
            result = wp.empty((len(points), len(tets)), dtype=int, device=self.device)
            wp.launch(
                bounds_probe,
                result.shape,
                [points, q, query.tetrahedra, query.low, query.high, result],
                device=self.device,
            )
            np.testing.assert_array_equal(result.numpy(), 0)
            ids = wp.empty((len(points), 2), dtype=int, device=self.device)
            bary = wp.empty((len(points), 2), dtype=wp.vec4d, device=self.device)
            wp.launch(
                location_probe,
                len(points),
                [points, q, query.tetrahedra, query.bvh.id, query.low, query.high, ids, bary],
                device=self.device,
            )
            np.testing.assert_array_equal(ids.numpy()[:, 0], ids.numpy()[:, 1])
            self.assertEqual(bary.numpy()[:, 0].tobytes(), bary.numpy()[:, 1].tobytes())


if __name__ == "__main__":
    unittest.main()
