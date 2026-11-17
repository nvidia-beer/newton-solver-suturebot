"""True residual, rounded-output acceptance and explicit solve-budget contracts."""

import os
import unittest

import numpy as np
import warp as wp
from warp.optim.linear import aslinearoperator
from warp.sparse import bsr_zeros

from newton_solver_suturebot._src.solvers.fem.certified_cg import CertifiedCorrection, assess, raw_residual
from newton_solver_suturebot._src.utils.capture import ScopedCapture


def matrix(values, offsets, columns, device):
    out = bsr_zeros(
        len(offsets) - 1,
        len(offsets) - 1,
        wp.mat33,
        device=device,
        nnz_capacity=len(columns),
    )
    out.values.assign(values)
    out.columns.assign(columns)
    out.offsets.assign(offsets)
    out.row_counts = None
    out.nnz = len(columns)
    return out


class ReliableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))

    def test_true_residual_uses_raw_float32_coefficients_and_float64_products(self):
        values = np.array(
            [
                np.diag([1.1, 2.3, 3.7]),
                np.eye(3) * -0.13,
                np.eye(3) * -0.13,
                np.diag([2.1, 1.3, 4.7]),
            ],
            np.float32,
        )
        A = matrix(values, [0, 2, 4], [0, 1, 0, 1], self.device)
        b_np = np.asarray([[0.3, 0.2, 0.1], [-0.7, 0.2, 0.4]], np.float32)
        x_np = np.asarray([[0.26, 0.07, 0.09], [-0.29, 0.19, 0.08]], np.float32)
        b = wp.array(b_np, dtype=wp.vec3, device=self.device)
        r = wp.empty(2, dtype=wp.vec3d, device=self.device)
        r32 = wp.empty(2, dtype=wp.vec3, device=self.device)
        dense = np.block([[values[0], values[1]], [values[2], values[3]]]).astype(float)
        for dtype in (wp.vec3, wp.vec3d):
            x = wp.array(x_np, dtype=dtype, device=self.device)
            wp.launch(
                raw_residual,
                2,
                [A.offsets, A.columns, A.values, b, x, r, r32],
                device=self.device,
            )
            expected = b_np.astype(float).ravel() - dense @ x_np.astype(float).ravel()
            np.testing.assert_allclose(r.numpy().ravel(), expected, rtol=1e-14, atol=2e-16)
            np.testing.assert_array_equal(r32.numpy(), r.numpy().astype(np.float32))

    def test_certificate_rejects_nonfinite_and_unmet_original_tolerance(self):
        rr = wp.array([4e-12], dtype=wp.float64, device=self.device)
        bb = wp.array([1.0], dtype=wp.float64, device=self.device)
        flag = wp.empty(1, dtype=int, device=self.device)
        status = wp.empty_like(flag)
        wp.launch(assess, 1, [rr, bb, 1e-6, flag, status], device=self.device)
        self.assertEqual(status.numpy()[0], 0)
        self.assertEqual(flag.numpy()[0], 1)
        rr.assign([np.nan])
        wp.launch(assess, 1, [rr, bb, 1e-6, flag, status], device=self.device)
        self.assertEqual(status.numpy()[0], 0)
        self.assertEqual(flag.numpy()[0], 0)

    def test_exact_solution_and_zero_rhs_keep_correction_off_within_budget(self):
        values = np.asarray([np.eye(3)], np.float32)
        A = matrix(values, [0, 1], [0], self.device)
        for rhs in ([[0.25, 0.5, 0.75]], [[0.0, 0.0, 0.0]]):
            b = wp.array(rhs, dtype=wp.vec3, device=self.device)
            x = wp.clone(b)
            solve = CertifiedCorrection(A, aslinearoperator(A), b, x, None, 1e-6)
            solve()
            report = solve.diagnostics()
            self.assertTrue(report["certified"])
            self.assertFalse(report["correction_used"])
            self.assertLessEqual(report["primary_A_actions"], 160)
            np.testing.assert_array_equal(b.numpy(), x.numpy())

    def test_gradient_output_rejected(self):
        A = matrix(np.asarray([np.eye(3)], np.float32), [0, 1], [0], self.device)
        b = wp.zeros(1, dtype=wp.vec3, device=self.device)
        x = wp.zeros_like(b, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "forward-only"):
            CertifiedCorrection(A, aslinearoperator(A), b, x, None, 1e-6)

    def test_unachievable_rounded_output_is_failure_without_extra_budget(self):
        A = matrix(np.asarray([np.eye(3) * 1.1], np.float32), [0, 1], [0], self.device)
        b = wp.array([[0.7, 0.3, 0.2]], dtype=wp.vec3, device=self.device)
        x = wp.zeros_like(b)
        solve = CertifiedCorrection(A, aslinearoperator(A), b, x, None, 1e-10)
        solve()
        report = solve.diagnostics()
        exact_residual = b.numpy().astype(float) - float(np.float32(1.1)) * x.numpy().astype(float)
        self.assertGreater(
            np.linalg.norm(exact_residual),
            1e-10 * np.linalg.norm(b.numpy().astype(float)),
        )
        self.assertFalse(report["certified"])
        self.assertTrue(report["correction_used"])
        self.assertLessEqual(report["primary_A_actions"], 160)

    def test_capture_replay_rechecks_changed_rhs(self):
        if not self.device.is_cuda:
            self.skipTest("CUDA capture replay")
        A = matrix(np.asarray([np.eye(3) * 1.1], np.float32), [0, 1], [0], self.device)
        b = wp.array([[0.7, 0.3, 0.2]], dtype=wp.vec3, device=self.device)
        x = wp.zeros_like(b)
        solve = CertifiedCorrection(A, aslinearoperator(A), b, x, None, 1e-6)
        solve()
        with ScopedCapture(device=self.device) as capture:
            x.zero_()
            solve()
        b.assign([[1.7, -0.3, 0.22]])
        wp.capture_launch(capture.graph)
        report = solve.diagnostics()
        self.assertTrue(report["certified"])
        residual = b.numpy().astype(float) - float(np.float32(1.1)) * x.numpy().astype(float)
        self.assertLessEqual(np.linalg.norm(residual), 1e-6 * np.linalg.norm(b.numpy().astype(float)))


if __name__ == "__main__":
    unittest.main()
