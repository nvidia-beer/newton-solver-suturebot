# SPDX-License-Identifier: Apache-2.0
"""Both global Chebyshev spectral estimates bound the same projected matrix."""

import os
import unittest

import numpy as np
import warp as wp
from test_chebyshev_preconditioner import grid_system
from warp.optim.linear import aslinearoperator
from warp.sparse import bsr_set_from_triplets, bsr_zeros

from newton_solver_suturebot._src.solvers.fem.chebyshev import ChebyshevJacobi


class ChebyshevSpectralBoundTests(unittest.TestCase):
    def test_captured_refresh_uses_both_live_bounds(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not device.is_cuda:
            self.skipTest("CUDA graph replay requires CUDA")
        rng = np.random.default_rng(44)
        count, matrix, inverse = grid_system(4, rng, device)
        captured = ChebyshevJacobi(aslinearoperator(matrix), inverse, degree=4, ratio=256, device=device)
        reference = ChebyshevJacobi(aslinearoperator(matrix), inverse, degree=4, ratio=256, device=device)
        rhs = wp.array(rng.normal(size=(count, 3)), dtype=wp.vec3, device=device)
        actual, expected = wp.empty_like(rhs), wp.empty_like(rhs)
        captured.refresh(matrix)
        captured.matvec(rhs, actual, actual, 1.0, 0.0)
        addresses = captured.bound.ptr, captured.coefficients.ptr
        with wp.ScopedCapture(device=device) as capture:
            captured.refresh(matrix)
            captured.matvec(rhs, actual, actual, 1.0, 0.0)
        values = matrix.values.numpy()
        for scale in (1.0, 0.5, 3.0):
            matrix.values.assign(scale * values)
            rhs.assign(rng.normal(size=(count, 3)).astype(np.float32))
            reference.refresh(matrix)
            reference.matvec(rhs, expected, expected, 1.0, 0.0)
            wp.capture_launch(capture.graph)
            self.assertEqual(addresses, (captured.bound.ptr, captured.coefficients.ptr))
            np.testing.assert_allclose(captured.bound.numpy(), reference.bound.numpy(), rtol=2e-6)
            np.testing.assert_allclose(captured.coefficients.numpy(), reference.coefficients.numpy(), rtol=2e-6)
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-6, atol=1e-6)

    def test_bound_dominates_exact_spectrum_with_cross_blocks_and_fixed_nodes(self):
        device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
        rng = np.random.default_rng(1104)
        count = 18
        for seed, scale in enumerate((1e-8, 1.0, 1e8)):
            matrix = np.zeros((count, count, 3, 3))
            for i in range(count):
                block = rng.normal(size=(3, 3))
                matrix[i, i] = 0.05 * np.eye(3) + block.T @ block
            for i in range(count - 1):
                block = rng.normal(size=(3, 3))
                block = block.T @ block
                matrix[i, i] += block
                matrix[i + 1, i + 1] += block
                matrix[i, i + 1] = matrix[i + 1, i] = -block
            matrix = (scale * matrix).astype(np.float32)
            rows, columns = np.nonzero(np.any(matrix != 0, axis=(2, 3)))
            sparse = bsr_zeros(count, count, wp.mat33f, device=device)
            bsr_set_from_triplets(
                sparse,
                wp.array(rows.astype(np.int32), dtype=int, device=device),
                wp.array(columns.astype(np.int32), dtype=int, device=device),
                wp.array(matrix[rows, columns], dtype=wp.mat33f, device=device),
            )
            diagonal = np.linalg.inv(matrix[np.arange(count), np.arange(count)].astype(np.float64)).astype(np.float32)
            inverse = wp.array(diagonal, dtype=wp.mat33f, device=device)
            preconditioner = ChebyshevJacobi(aslinearoperator(sparse), inverse, degree=4, ratio=256, device=device)
            for mask_fixed in (False, True):
                fixed = np.zeros(count, np.int32)
                if mask_fixed:
                    fixed[::3] = 1
                preconditioner.refresh(sparse, fixed=wp.array(fixed, dtype=int, device=device))
                bound = float(preconditioner.bound.numpy()[0])
                retained = np.flatnonzero(fixed == 0)
                action = np.einsum("nab,nmbc->nmac", diagonal[retained].astype(float), matrix[retained][:, retained])
                scalar = action.transpose(0, 2, 1, 3).reshape(3 * len(retained), 3 * len(retained))
                radius = np.max(np.abs(np.linalg.eigvals(scalar)))
                scalar_bound = np.abs(scalar).sum(axis=1).max()
                frobenius_bound = np.linalg.norm(action, axis=(2, 3)).sum(axis=1).max()
                with self.subTest(seed=seed, fixed=mask_fixed):
                    self.assertGreaterEqual(bound, radius * (1 - 2e-6))
                    self.assertLessEqual(bound, scalar_bound * (1 + 2e-6))
                    self.assertLessEqual(bound, frobenius_bound * (1 + 2e-6))


if __name__ == "__main__":
    unittest.main()
