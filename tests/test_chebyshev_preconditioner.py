# SPDX-License-Identifier: Apache-2.0
"""Chebyshev block-Jacobi preconditioner: SPD validity and CG behavior."""

import os
import unittest

import numpy as np
import warp as wp
from warp.optim.linear import aslinearoperator, cg
from warp.sparse import bsr_set_from_triplets, bsr_zeros

from newton_solver_suturebot._src.solvers.fem.chebyshev import ChebyshevJacobi


def grid_system(side, rng, device):
    n = side**3
    grid = np.arange(n).reshape(side, side, side)
    rows, cols, values = [], [], []
    degree = np.zeros(n)
    eye = np.eye(3)
    for axis in range(3):
        a = np.moveaxis(grid, axis, 0)[:-1].reshape(-1)
        b = np.moveaxis(grid, axis, 0)[1:].reshape(-1)
        weight = rng.uniform(0.5, 1.5, a.size)
        for i, j, w in zip(a, b, weight, strict=True):
            rows.extend((i, j))
            cols.extend((j, i))
            values.extend((-w * eye, -w * eye))
            degree[i] += w
            degree[j] += w
    diagonal = np.einsum("n,ij->nij", degree + 1e-2, eye)
    for i in range(n):
        rows.append(i)
        cols.append(i)
        values.append(diagonal[i])
    matrix = bsr_zeros(n, n, wp.mat33f, device=device)
    bsr_set_from_triplets(
        matrix,
        wp.array(np.asarray(rows), dtype=wp.int32, device=device),
        wp.array(np.asarray(cols), dtype=wp.int32, device=device),
        wp.array(np.asarray(values), dtype=wp.mat33f, device=device),
    )
    inverse = wp.array(np.linalg.inv(diagonal), dtype=wp.mat33f, device=device)
    return n, matrix, inverse


class ChebyshevTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not self.device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("Chebyshev preconditioning targets the conditional CUDA CG path")

    def test_apply_is_symmetric_positive_definite_and_polynomial_in_operator(self):
        rng = np.random.default_rng(55)
        n, matrix, inverse = grid_system(5, rng, self.device)
        chebyshev = ChebyshevJacobi(aslinearoperator(matrix), inverse, degree=4, ratio=256.0, device=self.device)
        chebyshev.refresh(matrix)
        self.assertGreater(float(chebyshev.bound.numpy()[0]), 0.0)
        probes = rng.normal(0, 1, (6, n, 3)).astype(np.float32)
        images = []
        for probe in probes:
            r = wp.array(probe, dtype=wp.vec3, device=self.device)
            z = wp.empty_like(r)
            chebyshev.matvec(r, z, z, 1.0, 0.0)
            images.append(z.numpy().reshape(-1).astype(np.float64))
        # Symmetry: <u, M v> == <v, M u> for the linear application.
        for i in range(0, 6, 2):
            u, v = probes[i].reshape(-1).astype(np.float64), probes[i + 1].reshape(-1).astype(np.float64)
            self.assertLess(abs(u @ images[i + 1] - v @ images[i]), 2e-4 * max(abs(u @ images[i + 1]), 1.0))
        # Positive definiteness on the probes.
        for probe, image in zip(probes, images, strict=True):
            self.assertGreater(probe.reshape(-1).astype(np.float64) @ image, 0.0)

    def test_cg_with_chebyshev_meets_the_unchanged_stopping_rule_in_fewer_iterations(self):
        rng = np.random.default_rng(7)
        n, matrix, inverse = grid_system(12, rng, self.device)
        b_host = rng.normal(0, 1, (n, 3)).astype(np.float32)
        b = wp.array(b_host, dtype=wp.vec3, device=self.device)
        iterations = {}
        solutions = {}
        for degree in (0, 4):
            x = wp.zeros(n, dtype=wp.vec3, device=self.device)
            if degree:
                chebyshev = ChebyshevJacobi(
                    aslinearoperator(matrix), inverse, degree=degree, ratio=256.0, device=self.device
                )
                chebyshev.refresh(matrix)
                preconditioner = chebyshev.preconditioner
            else:
                from warp.optim.linear import LinearOperator

                def matvec(x, y, z, alpha, beta, inverse=inverse):
                    values, xv = inverse.numpy(), x.numpy()
                    result = alpha * np.einsum("nij,nj->ni", values, xv)
                    if beta:
                        result = result + beta * y.numpy()
                    z.assign(result.astype(np.float32))

                preconditioner = LinearOperator((n, n), wp.vec3, self.device, matvec)
            functor = cg(
                matrix,
                b,
                x,
                tol=1e-4,
                atol=0.0,
                maxiter=600,
                M=preconditioner,
                check_every=0 if degree else 50,
                use_cuda_graph=bool(degree),
                run=False,
            )
            result = functor(b=b, x=x, M=preconditioner)
            iterations[degree] = int(result[0].numpy()[0]) if isinstance(result[0], wp.array) else int(result[0])
            solutions[degree] = x.numpy().copy()
            operator = aslinearoperator(matrix)
            residual = wp.empty_like(b)
            operator.matvec(x, b, residual, alpha=-1.0, beta=1.0)
            self.assertLessEqual(
                np.linalg.norm(residual.numpy().reshape(-1)), 1.02e-4 * np.linalg.norm(b_host.reshape(-1))
            )
        self.assertLess(iterations[4], 0.5 * iterations[0])
        scale = np.abs(solutions[0]).max()
        self.assertLess(np.abs(solutions[4] - solutions[0]).max(), 1e-3 * scale)

    def test_fixed_rows_are_excluded_and_invariant(self):
        rng = np.random.default_rng(11)
        n, matrix, inverse = grid_system(4, rng, self.device)
        fixed_host = (rng.uniform(size=n) < 0.3).astype(np.int32)
        fixed = wp.array(fixed_host, dtype=int, device=self.device)
        chebyshev = ChebyshevJacobi(aslinearoperator(matrix), inverse, degree=3, ratio=64.0, device=self.device)
        chebyshev.refresh(matrix, fixed=fixed)
        bound_masked = float(chebyshev.bound.numpy()[0])
        chebyshev.refresh(matrix)
        bound_full = float(chebyshev.bound.numpy()[0])
        self.assertLess(bound_masked, bound_full + 1e-6)
        # Zero residual rows stay exactly zero through the polynomial.
        r_host = rng.normal(0, 1, (n, 3)).astype(np.float32)
        r_host[fixed_host.astype(bool)] = 0
        # The operator must preserve the invariant subspace like the projected
        # FEM operator does; emulate with an exact zero of fixed rows/columns.
        values = matrix.values.numpy()
        offsets, columns = matrix.offsets.numpy(), matrix.columns.numpy()
        for row in range(n):
            for slot in range(offsets[row], offsets[row + 1]):
                if fixed_host[row] or fixed_host[columns[slot]]:
                    values[slot] = np.eye(3) if columns[slot] == row else np.zeros((3, 3))
        matrix.values.assign(values)
        chebyshev.refresh(matrix, fixed=fixed)
        r = wp.array(r_host, dtype=wp.vec3, device=self.device)
        z = wp.empty_like(r)
        chebyshev.matvec(r, z, z, 1.0, 0.0)
        np.testing.assert_array_equal(z.numpy()[fixed_host.astype(bool)], 0)


if __name__ == "__main__":
    unittest.main()
