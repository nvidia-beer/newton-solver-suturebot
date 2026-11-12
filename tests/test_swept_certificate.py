# SPDX-License-Identifier: Apache-2.0
"""Certificates may reject work, but never replace ambiguous swept CCD."""

import os
import unittest

import numpy as np
import warp as wp

from newton_solver_suturebot._src.geometry.swept_certificate import swept_triangle_clear, tip_roundoff_bound
from newton_solver_suturebot._src.insertion import _surface_crossings
from newton_solver_suturebot._src.needle_trial_forces import tip_boundary_certificate


@wp.kernel(enable_backward=False)
def batch(cases: wp.array2d[wp.vec3d], error: wp.float64, clear: wp.array[int]):
    i = wp.tid()
    clear[i] = int(
        swept_triangle_clear(
            cases[i, 0],
            cases[i, 1],
            cases[i, 2],
            cases[i, 3],
            cases[i, 4],
            cases[i, 5],
            cases[i, 6],
            cases[i, 7],
            error,
        )
    )


@wp.kernel(enable_backward=False)
def tip_errors(poses: wp.array[wp.transform], local: wp.array[wp.vec3d], result: wp.array[wp.float64]):
    i = wp.tid()
    result[i] = tip_roundoff_bound(poses[i], local[i])


class SweptCertificateTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.device.__enter__()

    def tearDown(self):
        self.device.__exit__(None, None, None)

    def evaluate(self, cases, *, error=0.0):
        cases = wp.array(np.asarray(cases), dtype=wp.vec3d)
        result = wp.empty(len(cases), dtype=int)
        wp.launch(batch, len(cases), [cases, error, result])
        return result.numpy().astype(bool)

    def test_crossing_tangency_near_real_near_endpoint_and_degenerate_fallback(self):
        flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        cases = []
        cases.append(np.vstack(([0.2, 0.2, -0.1], [0.2, 0.2, 0.1], flat, flat)))
        # This root lies just after1 but inside the oracle's1+1e-7 allowance.
        cases.append(np.vstack(([0.2, 0.2, 0.001], [0.2, 0.2, 5e-11], flat, flat)))
        # Signed volume=(t-.5)^2, then (t-.5)^2+eps². Neither a tangent
        # nor a near-real conjugate root may be certified as safely separated.
        old = np.array([[0, 0, 0], [1, -0.5, 0], [0, 1, -0.5]])
        new = np.array([[0, 0, 0], [1, 0.5, 0], [0, 1, 0.5]])
        for eps in (0, 5e-8):
            point = [1, 0, eps * eps]
            cases.append(np.vstack((point, point, old, new)))
        cases.append(np.vstack(([0.2, 0.2, 0.1], [0.2, 0.2, 0.2], np.zeros((3, 3)), np.zeros((3, 3)))))
        invalid = cases[0].copy()
        invalid[3, 0] = np.nan
        cases.append(invalid)
        np.testing.assert_array_equal(self.evaluate(cases), False)
        self.assertTrue(_surface_crossings(cases[0][0], cases[0][1], flat, flat, np.array([[0, 1, 2]])))
        self.assertTrue(_surface_crossings(cases[1][0], cases[1][1], flat, flat, np.array([[0, 1, 2]])))

    def test_oblique_plane_and_tip_uncertainty_are_conservative(self):
        triangle = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1]], float)
        case = np.vstack(([0.2, 0.2, 0.35], [0.3, 0.2, 0.45], triangle, triangle))
        self.assertTrue(self.evaluate([case])[0])
        self.assertFalse(self.evaluate([case], error=0.1)[0])
        # Disjoint exact bounds are independently sufficient.
        case[:2, 2] = 2
        self.assertTrue(self.evaluate([case])[0])

    def test_componentwise_tip_discrepancy_remains_conservative_after_rotation(self):
        # Rotate the allowed (10nm,10nm,10nm) local discrepancy onto world Z.
        # A scalar 10nm floor would certify clear although the actual tip crosses.
        angle = np.arccos(1 / np.sqrt(3))
        axis = np.array([1, -1, 0]) / np.sqrt(2)
        quaternion = np.r_[axis * np.sin(angle / 2), np.cos(angle / 2)]
        pose = wp.transform(wp.vec3(), wp.quat(*quaternion))
        actual_tip = np.asarray(wp.transform_point(pose, wp.vec3(1e-8, 1e-8, 1e-8)), dtype=float)
        errors = wp.empty(1, dtype=wp.float64)
        wp.launch(tip_errors, 1, [wp.array([pose], dtype=wp.transform), wp.zeros(1, dtype=wp.vec3d), errors])
        bound = float(errors.numpy()[0])
        self.assertGreaterEqual(bound, np.max(np.abs(actual_tip)))
        triangle = np.array([[-1, -1, 1.2e-8], [1, -1, 1.2e-8], [0, 1, 1.2e-8]])
        start, graph_tip = np.array([0, 0, -2e-8]), np.zeros(3)
        self.assertTrue(_surface_crossings(start, actual_tip, triangle, triangle, np.array([[0, 1, 2]])))
        case = np.vstack((start, graph_tip, triangle, triangle))
        self.assertFalse(self.evaluate([case], error=bound)[0])

    def test_nonunit_and_nonfinite_poses_require_host_fallback(self):
        poses = [wp.transform(wp.vec3(), wp.quat(0, 0, 0, w)) for w in (0, 2, np.nan, np.inf)]
        errors = wp.empty(len(poses), dtype=wp.float64)
        wp.launch(
            tip_errors, len(poses), [wp.array(poses, dtype=wp.transform), wp.zeros(len(poses), dtype=wp.vec3d), errors]
        )
        self.assertTrue(np.all(~np.isfinite(errors.numpy())))

    def test_boundary_kernel_rejects_nonfinite_tips_and_nonunit_poses_before_query(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        q = wp.array(points, dtype=wp.vec3)
        mesh = wp.Mesh(points=q, indices=wp.array([0, 1, 2], dtype=int))
        tips = wp.array([[0, 0, 1]], dtype=wp.vec3d)
        old_tips = wp.array([[0, 0, 0.9]], dtype=wp.vec3d)
        poses = wp.array([wp.transform_identity()], dtype=wp.transform)
        fallback = wp.zeros(1, dtype=int)
        faces = wp.array([[0, 1, 2]], dtype=wp.vec3i)
        inputs = [
            mesh.id,
            mesh.id,
            tips,
            old_tips,
            wp.zeros(3, dtype=wp.float64),
            wp.array([1, 1, 1], dtype=int),
            0.1,
            1,
            faces,
            faces,
            wp.array([0, 1, 2], dtype=int),
            wp.array(points, dtype=wp.vec3d),
            q,
            q,
            poses,
            0,
            wp.vec3d(),
            fallback,
        ]
        wp.launch(tip_boundary_certificate, 3, inputs)
        self.assertEqual(int(fallback.numpy()[0]), 0)
        cases = [
            ([np.nan, 0, 0.9], [0, 0, 1], [0, 0, 0, 1]),
            ([np.inf, 0, 0.9], [0, 0, 1], [0, 0, 0, 1]),
            ([0, 0, 0.9], [0, np.nan, 1], [0, 0, 0, 1]),
            ([0, 0, 0.9], [0, np.inf, 1], [0, 0, 0, 1]),
            ([0, 0, 0.9], [0, 0, 1], [0, 0, 0, 2]),
            ([0, 0, 0.9], [0, 0, 1], [0, 0, 0, np.nan]),
        ]
        for old, tip, quaternion in cases:
            with self.subTest(old=old, tip=tip, quaternion=quaternion):
                old_tips.assign(np.asarray([old], dtype=np.float64))
                tips.assign(np.asarray([tip], dtype=np.float64))
                poses.assign(np.asarray([[0, 0, 0, *quaternion]], dtype=np.float32))
                fallback.zero_()
                wp.launch(tip_boundary_certificate, 3, inputs)
                self.assertEqual(int(fallback.numpy()[0]), 1)

    def test_random_linear_triangle_motions_have_no_false_accept_against_cpu_oracle(self):
        rng = np.random.default_rng(9326)
        count = 2400
        cases = rng.normal(size=(count, 8, 3))
        scales = 10 ** rng.uniform(-4, 0, count)
        cases *= scales[:, None, None]
        cases += rng.uniform(-0.2, 0.2, (count, 1, 3))
        # Include nearly stationary and nearly coplanar configurations, too.
        cases[:400, 5:] = cases[:400, 2:5] + rng.normal(scale=1e-10, size=(400, 3, 3))
        certified = self.evaluate(cases)
        self.assertGreater(np.count_nonzero(certified), 100)
        for index in np.flatnonzero(certified):
            x = cases[index]
            events = _surface_crossings(x[0], x[1], x[2:5], x[5:8], np.array([[0, 1, 2]]))
            self.assertFalse(events, f"certificate rejected an oracle event:case{index}")


if __name__ == "__main__":
    unittest.main()
