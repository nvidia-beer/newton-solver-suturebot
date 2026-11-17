"""Numerical-guess contract on an actual sparse Warp operator; no GPU required."""

import os
import types
import unittest
import unittest.mock

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator, aslinearoperator, cg
from warp.sparse import bsr_set_from_triplets, bsr_zeros

from newton_solver_suturebot._src.solvers.fem.affine_guess import COUNTERS, DeviceBasisGuess
from newton_solver_suturebot._src.utils.capture import ScopedCapture

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


def sparse(matrix):
    n = matrix.shape[0] // 3
    rows, cols, values = [], [], []
    for i in range(n):
        for j in range(n):
            rows.append(i)
            cols.append(j)
            values.append(matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3])
    result = bsr_zeros(n, n, wp.mat33, device=DEVICE)
    bsr_set_from_triplets(
        result,
        wp.array(rows, dtype=int, device=DEVICE),
        wp.array(cols, dtype=int, device=DEVICE),
        wp.array(np.asarray(values, np.float32), dtype=wp.mat33, device=DEVICE),
    )
    return result


def vec(data):
    return wp.array(np.asarray(data, np.float32).reshape(-1, 3), dtype=wp.vec3, device=DEVICE)


class AffineGuessTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).astype(np.float32)
        self.matrix[0, 3] = self.matrix[3, 0] = 0.2
        self.bsr = sparse(self.matrix)
        self.free = wp.array([1, 3], dtype=int, device=DEVICE)
        self.core = DeviceBasisGuess(aslinearoperator(self.bsr), self.free, capacity=6)
        self.support = (0, 1, 2)
        columns = np.linalg.solve(self.matrix.astype(float), np.eye(6)[:, self.support]).astype(np.float32)
        self.core.load_basis(wp.array(columns.ravel(order="F"), dtype=float, device=DEVICE), self.support)
        self.maximum = wp.array([1.0], dtype=float, device=DEVICE)
        self.zero = vec(np.zeros(6))
        self.full = vec(np.zeros(12))
        self.fake_linear = types.SimpleNamespace(
            _cur_iter_and_condition=wp.array([3, 0], dtype=int, device=DEVICE),
            _tiled_dot=types.SimpleNamespace(col=lambda _: wp.array([0.0], dtype=float, device=DEVICE)),
            _residuals=[None, wp.array([1e-12], dtype=float, device=DEVICE)],
            _maxiter=160,
        )
        self.core.after(self.full, self.zero, self.maximum, self.fake_linear)

    def propose(self, rhs, guess=None):
        full = np.zeros((4, 3), np.float32)
        full[[1, 3]] = np.asarray(rhs).reshape(2, 3)
        original = vec(np.zeros(6) if guess is None else guess)
        self.core.before(vec(full), vec(rhs), original, self.maximum)
        return original.numpy().ravel(), dict(zip(COUNTERS, self.core.counters.numpy().tolist(), strict=True))

    def test_full_response_and_unknown_support_leave_original_system_intact(self):
        before = self.bsr.values.numpy().copy()
        rhs = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6], np.float32)
        value, counts = self.propose(rhs)
        expected = np.linalg.solve(self.matrix, np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], np.float32))
        np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-8)
        self.assertEqual(counts["selected"], 1)
        self.assertGreater(np.linalg.norm(rhs - self.matrix @ value), 0)
        self.assertLess(np.linalg.norm(rhs - self.matrix @ value), np.linalg.norm(rhs))
        np.testing.assert_array_equal(self.bsr.values.numpy(), before)
        # The candidate is only a guess. Solve the unchanged remainder and
        # recover the ordinary full-system solution including unknown support.
        final = value + np.linalg.solve(self.matrix, rhs - self.matrix @ value)
        np.testing.assert_allclose(final, np.linalg.solve(self.matrix, rhs), rtol=1e-6, atol=1e-8)

    def test_original_warp_cg_runs_after_guess_and_solves_unknown_support(self):
        rhs = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6], np.float32)
        proposed, counts = self.propose(rhs)
        self.assertEqual(counts["selected"], 1)
        operator = aslinearoperator(self.bsr)
        if wp.get_device(DEVICE).is_cpu:
            operator = LinearOperator(
                operator.shape,
                operator.dtype,
                operator.device,
                operator.matvec,
                batch_offsets=wp.array([0, 6], dtype=int, device=DEVICE),
                max_batch_length=6,
            )
        outputs = []
        iterations = []
        for guess in (np.zeros(6, np.float32), proposed):
            x = vec(guess)
            result = cg(operator, vec(rhs), x, tol=1e-6, atol=0.0, maxiter=160, check_every=1, use_cuda_graph=False)
            outputs.append(x.numpy().ravel())
            iterations.append(result[0])
            self.assertLessEqual(result[1], result[2])
        np.testing.assert_allclose(outputs[1], outputs[0], rtol=3e-6, atol=1e-7)
        self.assertLessEqual(iterations[1], iterations[0])

    def test_cuda_capture_reads_updated_rhs_and_preserves_storage(self):
        if not wp.get_device(DEVICE).is_cuda:
            self.skipTest("CUDA graph regression")
        rhs = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], np.float32)
        self.propose(rhs)  # warm the sparse kernel and tile reductions
        physical = np.zeros((4, 3), np.float32)
        physical[[1, 3]] = rhs.reshape(2, 3)
        raw, normalized, guess = vec(physical), vec(rhs), vec(np.zeros(6))
        owners = self.core.owners
        with ScopedCapture(device=DEVICE) as capture:
            self.core.before(raw, normalized, guess, self.maximum)
        for multiplier in (1.0, -2.0):
            raw.assign(physical * multiplier)
            normalized.assign(rhs.reshape(2, 3) * multiplier)
            guess.zero_()
            wp.capture_launch(capture.graph)
            expected = np.linalg.solve(self.matrix, rhs * multiplier)
            np.testing.assert_allclose(guess.numpy().ravel(), expected, rtol=1e-6, atol=1e-8)
        self.assertEqual(tuple(a.ptr for a in owners), tuple(a.ptr for a in self.core.owners))

    def test_device_flag_is_an_owner_and_is_written_only_where_it_changes(self):
        core = self.core
        self.assertTrue(any(array is core.enabled for array in core.owners))
        np.testing.assert_array_equal(core.enabled.numpy(), [0])
        with unittest.mock.patch.object(wp.array, "fill_", autospec=True, side_effect=wp.array.fill_) as fill:
            core.set_enabled(False)
            self.assertEqual(fill.call_count, 0)
            core.set_enabled(True)
            core.set_enabled(1)
            self.assertEqual(fill.call_count, 1)
            np.testing.assert_array_equal(core.enabled.numpy(), [1])
            core.set_enabled(False)
            self.assertEqual(fill.call_count, 2)
        np.testing.assert_array_equal(core.enabled.numpy(), [0])
        # The flag belongs to the storage that a captured solve addresses: a reload keeps it.
        pointer = core.enabled.ptr
        columns = np.linalg.solve(self.matrix.astype(float), np.eye(6)[:, self.support]).astype(np.float32)
        core.load_basis(wp.array(columns.ravel(order="F"), dtype=float, device=DEVICE), self.support)
        self.assertEqual(core.enabled.ptr, pointer)

    def test_device_flag_is_not_switched_inside_a_capture(self):
        if not wp.get_device(DEVICE).is_cuda:
            self.skipTest("CUDA graph capture")
        with ScopedCapture(device=DEVICE):
            self.core.norms.zero_()
            self.core.set_enabled(False)
            with self.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                self.core.set_enabled(True)
        np.testing.assert_array_equal(self.core.enabled.numpy(), [0])

    def test_stale_numeric_matrix_uses_current_residual(self):
        changed = self.matrix.copy() * np.float32(1.15)
        self.core.operator = aslinearoperator(sparse(changed))
        rhs = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], np.float32)
        value, counts = self.propose(rhs)
        self.assertEqual(counts["selected"], 1)
        self.assertLess(np.linalg.norm(rhs - changed @ value), np.linalg.norm(rhs))
        # A much larger current tangent makes the same stale correction worse.
        self.core.operator = aslinearoperator(sparse(self.matrix * 10))
        value, counts = self.propose(rhs)
        np.testing.assert_array_equal(value, np.zeros(6))
        self.assertEqual(counts["no_improvement"], 1)

    def test_bad_basis_nonfinite_history_mismatch_and_reset_fail_closed(self):
        rhs = np.ones(6, np.float32)
        value, counts = self.propose(rhs, guess=np.ones(6, np.float32))
        np.testing.assert_array_equal(value, np.ones(6))
        self.assertEqual(counts["history_mismatch"], 1)
        self.core.basis.fill_(float("nan"))
        value, counts = self.propose(rhs)
        np.testing.assert_array_equal(value, np.zeros(6))
        self.assertEqual(counts["nonfinite"], 1)
        self.core.reset()
        value, counts = self.propose(rhs)
        np.testing.assert_array_equal(value, np.zeros(6))
        self.assertEqual(counts["no_history"], 1)

    def test_tiny_rhs_history_is_stored_in_physical_units_and_bypassed(self):
        self.maximum.fill_(1e-15)
        raw = np.zeros((4, 3), np.float32)
        raw[[1, 3]] = 1e-15
        normalized = vec(np.ones(6, np.float32))
        self.core.after(vec(raw), normalized, self.maximum, self.fake_linear)
        np.testing.assert_allclose(self.core.previous_rhs.numpy(), 1e-15, rtol=1e-6)
        np.testing.assert_allclose(self.core.previous_increment.numpy(), 1e-15, rtol=1e-6)
        value, counts = self.propose(np.full(6, 1e-15, np.float32), guess=np.ones(6))
        np.testing.assert_array_equal(value, np.ones(6))
        self.assertEqual(counts["scaled_rhs_fallback"], 1)

    def test_fixed_capacity_reload_preserves_pointer_and_clears_history(self):
        owners = tuple(a.ptr for a in self.core.owners)
        new_support = (3, 4, 5)
        columns = np.linalg.solve(self.matrix.astype(float), np.eye(6)[:, new_support]).astype(np.float32)
        self.core.load_basis(wp.array(columns.ravel(order="F"), dtype=float, device=DEVICE), new_support)
        self.assertEqual(owners, tuple(a.ptr for a in self.core.owners))
        self.assertEqual(self.core.history.numpy()[0], 0)
        with self.assertRaises(ValueError):
            self.core.load_basis(wp.array(columns.ravel(order="F"), dtype=float, device=DEVICE), (0, 0, 1))


if __name__ == "__main__":
    unittest.main()
