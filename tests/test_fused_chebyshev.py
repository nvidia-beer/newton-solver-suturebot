# SPDX-License-Identifier: Apache-2.0
"""Polynomial, parity, capture and fallback checks for compact packed fusion."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from test_packed_bsr import make_matrix

from newton_solver_suturebot._src.solvers.fem.chebyshev import ChebyshevJacobi
from newton_solver_suturebot._src.solvers.fem.fused_chebyshev import FusedPackedChebyshev, _packed_next_step
from newton_solver_suturebot._src.solvers.fem.packed_bsr import packed_aslinearoperator


class FusedChebyshevTests(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.rng = np.random.default_rng(819)
        self.n = 19
        raw = self.rng.normal(size=(3 * self.n, 3 * self.n))
        dense = (raw.T @ raw + 10 * np.eye(3 * self.n)).astype(np.float32)
        blocks = dense.reshape(self.n, 3, self.n, 3).transpose(0, 2, 1, 3).copy()
        self.matrix = make_matrix(
            np.arange(self.n + 1) * self.n,
            np.tile(np.arange(self.n), self.n),
            blocks.reshape(-1, 3, 3),
            self.n,
            self.device,
        )
        self.inverse = wp.array(
            np.linalg.inv(blocks[np.arange(self.n), np.arange(self.n)]).astype(np.float32),
            dtype=wp.mat33,
            device=self.device,
        )
        self.dense = dense.astype(np.float64)

    def pair(self, degree, compact=True):
        operator = packed_aslinearoperator(self.matrix)
        reference = ChebyshevJacobi(operator, self.inverse, degree=degree, ratio=256.0, device=self.device)
        fused = FusedPackedChebyshev(
            operator, self.inverse, matrix=self.matrix, compact=compact, degree=degree, ratio=256.0, device=self.device
        )
        for obj in (reference, fused):
            obj.refresh(self.matrix)
        np.testing.assert_array_equal(reference.coefficients.numpy(), fused.coefficients.numpy())
        return reference, fused

    def test_even_odd_degrees_match_packed_unfused_and_float64_polynomial(self):
        wp.load_module(module=_packed_next_step.module, device="cpu")
        rhs = self.rng.normal(size=(self.n, 3)).astype(np.float32)
        r = wp.array(rhs, dtype=wp.vec3, device=self.device)
        expected, actual = wp.empty_like(r), wp.empty_like(r)
        for degree in (2, 3, 4, 5, 8):
            reference, fused = self.pair(degree)
            self.assertEqual(fused.can_fuse(r, actual), self.device.is_cuda)
            reference.matvec(r, expected, expected, 1.0, 0.0)
            fused.matvec(r, actual, actual, 1.0, 0.0)
            coefficients = reference.coefficients.numpy().astype(np.float64)
            inverse = self.inverse.numpy().astype(np.float64)
            direction = coefficients[0] * np.einsum("nij,nj->ni", inverse, rhs.astype(np.float64))
            oracle = direction.copy()
            for k in range(1, degree):
                Az = (self.dense @ oracle.ravel()).reshape(self.n, 3)
                direction = coefficients[2 * k] * direction + coefficients[2 * k + 1] * np.einsum(
                    "nij,nj->ni", inverse, rhs - Az
                )
                oracle += direction
            with self.subTest(degree=degree):
                # Same parity tolerances as the original polynomial tests.
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(actual.numpy(), oracle, rtol=2e-5, atol=2e-6)
                self.assertGreater(float(np.sum(rhs.astype(np.float64) * actual.numpy())), 0.0)
            r.zero_()
            fused.matvec(r, actual, actual, 1.0, 0.0)
            np.testing.assert_array_equal(actual.numpy(), 0)
            r.assign(rhs)

    def test_original_affine_and_alias_rejections_are_unchanged(self):
        _, fused = self.pair(4)
        r = wp.ones(self.n, dtype=wp.vec3, device=self.device)
        out = wp.empty_like(r)
        for alpha, beta, z in ((0.5, 0, out), (1, 0.5, out), (1, 0, r)):
            with self.subTest(alpha=alpha, beta=beta), self.assertRaisesRegex(ValueError, "preconditioner form"):
                fused.matvec(r, out, z, alpha, beta)

    def test_cpu_unknown_full_and_gradient_fallbacks(self):
        r = wp.ones(self.n, dtype=wp.vec3, device=self.device)
        out = wp.empty_like(r)
        for compact in (False, True):
            reference, fused = self.pair(4, compact)
            if compact and self.device.is_cuda:
                r.requires_grad = True
            with patch.object(ChebyshevJacobi, "matvec", autospec=True, side_effect=ChebyshevJacobi.matvec) as fallback:
                fused.matvec(r, out, out, 1.0, 0.0)
                fallback.assert_called_once()
            r.requires_grad = False
            expected = wp.empty_like(r)
            reference.matvec(r, expected, expected, 1.0, 0.0)
            np.testing.assert_array_equal(out.numpy(), expected.numpy())
        _, fused = self.pair(4)
        fixed = wp.zeros(self.n, dtype=int, device=self.device)
        fused.refresh(self.matrix, fixed=fixed)
        self.assertFalse(fused.can_fuse(r, out))
        fused.refresh(self.matrix)
        with wp.Tape():
            self.assertFalse(fused.can_fuse(r, out))

    def test_capture_replays_live_rhs_values_and_coefficients(self):
        if not self.device.is_cuda:
            self.skipTest("CUDA graph replay")
        r = wp.array(self.rng.normal(size=(self.n, 3)), dtype=wp.vec3, device=self.device)
        actual, expected = wp.empty_like(r), wp.empty_like(r)
        values, inverse = self.matrix.values.numpy(), self.inverse.numpy()
        for degree in (3, 4):
            reference, fused = self.pair(degree)
            fused.matvec(r, actual, actual, 1.0, 0.0)
            with wp.ScopedCapture(device=self.device) as capture:
                fused.refresh(self.matrix)
                fused.matvec(r, actual, actual, 1.0, 0.0)
            for scale in (1.0, 0.7, 2.0):
                self.matrix.values.assign(scale * values)
                self.inverse.assign(inverse / scale)
                r.assign(self.rng.normal(size=(self.n, 3)).astype(np.float32))
                reference.refresh(self.matrix)
                reference.matvec(r, expected, expected, 1.0, 0.0)
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(reference.coefficients.numpy(), fused.coefficients.numpy())
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
