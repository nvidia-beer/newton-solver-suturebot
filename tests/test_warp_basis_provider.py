"""Approximate reference columns never replace the current physical CG solve."""

import unittest

import numpy as np
import warp as wp
from test_affine_guess import DEVICE, sparse

from newton_solver_suturebot._src.solvers.fem.warp_basis_provider import WarpBasisProvider


class WarpBasisTests(unittest.TestCase):
    @staticmethod
    def matrix():
        n = 24
        A = np.diag(np.linspace(2.0, 5.0, n).astype(np.float32))
        A += np.diag(np.full(n - 1, -0.2, np.float32), 1)
        A += np.diag(np.full(n - 1, -0.2, np.float32), -1)
        return A, sparse(A)

    def test_append_reorder_and_changed_current_values_preserve_reference(self):
        A, matrix = self.matrix()
        before = matrix.values.numpy().copy()
        provider = WarpBasisProvider()
        try:
            first, meta = provider(matrix, (0, 1, 2))
            engine = provider.engine
            np.testing.assert_array_equal(matrix.values.numpy(), before)
            matrix.values.assign(before * np.float32(1.7))
            result, meta = provider(matrix, (0, 1, 2, 6, 7, 8))
            self.assertIs(engine, provider.engine)
            self.assertEqual(meta["reference_builds"], 1)
            self.assertEqual(meta["new_columns"], 3)
            self.assertEqual(len(engine.records), 6)
            np.testing.assert_array_equal(first.numpy(), result.numpy()[: 3 * len(A)])
            X = result.numpy().reshape(6, len(A)).T.astype(float)
            rhs = np.eye(len(A))[:, (0, 1, 2, 6, 7, 8)]
            residual = np.linalg.norm(A.astype(float) @ X - rhs, axis=0)
            self.assertLess(float(residual.max()), 3.1e-4)
            for record, expected in zip(engine.records, residual, strict=True):
                self.assertAlmostEqual(record["true_residual"], expected, places=11)
            # This deliberately stale guess does not solve the changed system.
            self.assertGreater(float(np.linalg.norm(1.7 * A @ X - rhs)), 0.5)
            reordered, meta = provider(matrix, (8, 0, 6))
            np.testing.assert_array_equal(reordered.numpy().reshape(3, len(A)), X.T[[5, 0, 3]].astype(np.float32))
            self.assertEqual(meta["new_columns"], 0)
            self.assertEqual(provider.solves, 2)
            self.assertEqual(engine.graph is not None, wp.get_device(DEVICE).is_cuda)
        finally:
            provider.close()
        self.assertTrue(engine.closed)

    def test_engine_made_ahead_takes_the_values_of_the_first_request_and_keeps_its_storage(self):
        A, matrix = self.matrix()
        start = matrix.values.numpy().copy()
        support = (0, 1, 2, 6, 7, 8)
        ahead, late, stale = WarpBasisProvider(), WarpBasisProvider(), WarpBasisProvider()
        try:
            # Start-up: the engine exists, with its scratch and its captured column solve; no reference yet.
            ahead.prepare(matrix)
            engine = ahead.engine
            self.assertIsNotNone(engine)
            self.assertEqual((ahead.reference_builds, ahead.solves, engine.records), (0, 0, []))
            self.assertEqual(engine.graph is not None, wp.get_device(DEVICE).is_cuda)
            ahead.prepare(matrix)
            self.assertIs(ahead.engine, engine)
            owned = (
                engine.matrix.values,
                engine.matrix.offsets,
                engine.matrix.columns,
                engine.diagonal,
                engine.inverse,
                engine.chebyshev.bound,
                engine.chebyshev.coefficients,
                engine.b,
                engine.x,
                engine.quality,
            )
            pointers = [array.ptr for array in owned]
            # The tangent moves in place until the first track support asks for columns.
            matrix.values.assign(start * np.float32(1.7))
            columns, meta = ahead(matrix, support)
            self.assertIs(ahead.engine, engine)
            self.assertEqual((meta["reference_builds"], ahead.reference_builds, ahead.solves), (1, 1, 1))
            self.assertEqual([array.ptr for array in owned], pointers)
            # An engine constructed at that request, as without start-up: the same columns, bit for bit.
            expected, _ = late(matrix, support)
            np.testing.assert_array_equal(columns.numpy().view(np.uint32), expected.numpy().view(np.uint32))
            np.testing.assert_array_equal(
                engine.chebyshev.coefficients.numpy().view(np.uint32),
                late.engine.chebyshev.coefficients.numpy().view(np.uint32),
            )
            np.testing.assert_array_equal(
                engine.matrix.values.numpy()[: engine.matrix.nnz].view(np.uint32),
                late.engine.matrix.values.numpy()[: late.engine.matrix.nnz].view(np.uint32),
            )
            self.assertEqual(engine.records, late.engine.records)
            # They are not the columns of the values the engine was made on.
            other, _ = stale(sparse(A), support)
            self.assertGreater(float(np.abs(columns.numpy() - other.numpy()).max()), 1e-3)
            # The reference is taken once: later values do not move it.
            matrix.values.assign(start * np.float32(3.0))
            more, meta = ahead(matrix, (*support, 12, 13, 14))
            self.assertEqual((meta["reference_builds"], meta["new_columns"]), (1, 3))
            np.testing.assert_array_equal(more.numpy()[: columns.size], columns.numpy())
            self.assertEqual([array.ptr for array in owned], pointers)
            # Another matrix is another engine, made ahead or at a request.
            second = sparse(A * 2)
            ahead.prepare(second)
            self.assertTrue(engine.closed)
            self.assertIsNot(ahead.engine, engine)
            self.assertEqual(ahead.support, ())
        finally:
            for provider in (ahead, late, stale):
                provider.close()

    def test_rebase_refuses_a_closed_engine_and_another_size(self):
        A, matrix = self.matrix()
        provider = WarpBasisProvider()
        try:
            provider.prepare(matrix)
            engine = provider.engine
            with self.assertRaisesRegex(ValueError, "own structure"):
                engine.rebase(sparse(A[:21, :21]))
            provider.invalidate()
            self.assertFalse(provider._reference_pending)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                engine.rebase(matrix)
        finally:
            provider.close()

    def test_request_after_an_engine_made_ahead_constructs_no_capture(self):
        if not wp.get_device(DEVICE).is_cuda:
            self.skipTest("The column solve is captured on CUDA")
        from unittest import mock

        from newton_solver_suturebot._src.solvers.fem import warp_basis_provider

        _, matrix = self.matrix()
        provider = WarpBasisProvider()
        try:
            provider.prepare(matrix)
            engine = provider.engine
            # Captured and launched by the constructor: the executable graph exists.
            self.assertIsNotNone(engine.graph.graph_exec)
            with (
                mock.patch.object(
                    warp_basis_provider, "ScopedCapture", side_effect=AssertionError("capture of the column engine")
                ),
                mock.patch.object(wp, "ScopedCapture", side_effect=AssertionError("capture of a Krylov solve")),
            ):
                first, _ = provider(matrix, (0, 1, 2))
                second, _ = provider(matrix, (0, 1, 2, 9, 10, 11))
            self.assertEqual((provider.reference_builds, provider.solves), (1, 2))
            np.testing.assert_array_equal(second.numpy()[: first.size], first.numpy())
        finally:
            provider.close()

    def test_new_storage_and_explicit_reset_invalidate_reference(self):
        A, matrix = self.matrix()
        provider = WarpBasisProvider()
        try:
            provider(matrix, (0, 1, 2))
            first = provider.engine
            second_matrix = sparse(A * 2)
            result, meta = provider(second_matrix, (0, 1, 2))
            self.assertTrue(first.closed)
            self.assertEqual(meta["reference_builds"], 2)
            residual = 2 * A.astype(float) @ result.numpy().reshape(3, len(A)).T - np.eye(len(A))[:, :3]
            self.assertLess(float(np.linalg.norm(residual, axis=0).max()), 3.1e-4)
            provider.invalidate()
            self.assertIsNone(provider.columns)
            self.assertIsNone(provider.engine)
            provider(second_matrix, (0, 1, 2))
            self.assertEqual(provider.reference_builds, 3)
        finally:
            provider.close()

    def test_approximate_budget_is_reported_and_not_silently_certified(self):
        A, matrix = self.matrix()
        provider = WarpBasisProvider(tolerance=1e-9, maxiter=1)
        try:
            result, meta = provider(matrix, (0, 7))
            self.assertTrue(np.isfinite(result.numpy()).all())
            self.assertEqual(meta["basis_maxiter"], 1)
            records = meta["column_diagnostics"]
            self.assertTrue(all(row["hit_budget"] for row in records))
            self.assertTrue(any(row["true_residual"] > row["threshold"] for row in records))
            # Finite capped columns may be numerical guesses only; the caller's
            # current-A residual guard and original CG still decide acceptance.
            X = result.numpy().reshape(2, len(A)).T
            self.assertGreater(np.linalg.norm(A @ X - np.eye(len(A))[:, [0, 7]]), 1e-9)
        finally:
            provider.close()

    def test_invalid_support_and_gradient_storage_fail_before_reference_build(self):
        _, matrix = self.matrix()
        provider = WarpBasisProvider(capacity=3)
        try:
            for support in ((), (0, 0), (-1,), (24,), (0, 1, 2, 3)):
                with self.assertRaises(ValueError):
                    provider(matrix, support)
            self.assertEqual(provider.reference_builds, 0)
            matrix.values.requires_grad = True
            with self.assertRaises(ValueError):
                provider(matrix, (0,))
            self.assertIsNone(provider.engine)
        finally:
            provider.close()


if __name__ == "__main__":
    unittest.main()
