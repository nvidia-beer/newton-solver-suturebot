# SPDX-License-Identifier: Apache-2.0
"""CPU fallback tests; PACKED_TEST_DEVICE=cuda:0 also exercises all custom kernels."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from warp.sparse import bsr_zeros

from newton_solver_suturebot._src.solvers.fem import packed_bsr as packed


def make_matrix(offsets, columns, values, ncol, device, counts=None):
    offsets = np.asarray(offsets, dtype=np.int32)
    columns = np.asarray(columns, dtype=np.int32)
    values = np.asarray(values, dtype=np.float32).reshape((-1, 3, 3))
    A = bsr_zeros(len(offsets) - 1, ncol, wp.mat33, device=device, nnz_capacity=len(columns))
    A.offsets.assign(offsets)
    A.columns.assign(columns)
    A.values.assign(values)
    A.row_counts = None if counts is None else wp.array(np.asarray(counts, np.int32), device=device)
    A.nnz = len(columns)
    return A


def reference(offsets, columns, values, x, y, alpha, beta, counts=None):
    """Float64 oracle and an absolute forward-error scale, including cancellation."""
    expected = np.zeros_like(y, dtype=np.float64)
    magnitude = np.zeros_like(expected)
    max_terms = 0
    for row in range(len(offsets) - 1):
        start = offsets[row]
        end = offsets[row + 1] if counts is None else start + counts[row]
        if alpha != 0:
            a = values[start:end].astype(np.float64)
            b = x[columns[start:end]].astype(np.float64)
            expected[row] = alpha * np.einsum("kij,kj->i", a, b)
            magnitude[row] = abs(alpha) * np.einsum("kij,kj->i", abs(a), abs(b))
            max_terms = max(max_terms, 3 * (end - start))
        if beta != 0:
            expected[row] += beta * y[row].astype(np.float64)
            magnitude[row] += abs(beta) * abs(y[row].astype(np.float64))
    # Conservative serial bound covers both reduction trees and separate multiply/add.
    operations = 2 * max_terms + 20
    unit_roundoff = np.finfo(np.float32).eps / 2
    gamma = operations * unit_roundoff / (1 - operations * unit_roundoff)
    return expected, gamma * magnitude + np.finfo(np.float32).tiny


def assert_reference(case, result, expected, bound):
    error = abs(result.astype(np.float64) - expected)
    case.assertTrue(np.all(np.isfinite(result)))
    case.assertTrue(np.all(error <= bound), f"max error {error.max()}, max bound {bound.max()}")


class PackedBsrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = wp.get_device(os.environ.get("PACKED_TEST_DEVICE", "cpu"))

    def test_cpu_dispatch_and_native_stub_compile(self):
        # Compile all specializations on CPU without invoking their unsupported raw entrypoint.
        for group in packed.GROUP_SIZES:
            wp.load_module(module=packed.packed_kernel(group).module, device="cpu")
        A = make_matrix([0, 1], [0], np.eye(3)[None], 1, "cpu")
        x = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3, device="cpu")
        y = wp.empty_like(x)
        with patch.object(packed, "bsr_mv", wraps=packed.bsr_mv) as original:
            self.assertIs(packed.packed_bsr_mv(A, x, y), y)
            original.assert_called_once()
        np.testing.assert_array_equal(y.numpy(), x.numpy())

    def test_tail_empty_long_rows_and_nonzero_beta(self):
        rng = np.random.default_rng(41)
        # 67 rows: a partial block/warp for every specialization, including group4.
        lengths = np.resize([0, 1, 2, 3, 7, 17, 67, 129], 67)
        offsets = np.r_[0, np.cumsum(lengths)].astype(np.int32)
        columns = rng.integers(0, 19, size=offsets[-1], dtype=np.int32)
        values = rng.normal(size=(len(columns), 3, 3)).astype(np.float32)
        x_np = rng.normal(size=(19, 3)).astype(np.float32)
        y_np = rng.normal(size=(67, 3)).astype(np.float32)
        A = make_matrix(offsets, columns, values, 19, self.device)
        x = wp.array(x_np, dtype=wp.vec3, device=self.device)
        for group in packed.GROUP_SIZES:
            for alpha, beta in ((1.0, 0.0), (-0.5, 0.25), (2.0, -1.5)):
                with self.subTest(group=group, alpha=alpha, beta=beta):
                    y = wp.array(y_np, dtype=wp.vec3, device=self.device)
                    self.assertEqual(packed.packed_supported(A, x, y), self.device.is_cuda)
                    packed.packed_bsr_mv(A, x, y, alpha, beta, group_size=group)
                    expected, bound = reference(offsets, columns, values, x_np, y_np, alpha, beta)
                    assert_reference(self, y.numpy(), expected, bound)

    def test_padded_sparse_capacity_uses_row_counts(self):
        offsets = np.array([0, 4, 7, 11], np.int32)
        counts = np.array([1, 0, 2], np.int32)
        columns = np.full(11, -100, np.int32)  # Invalid unused capacity must not be read.
        values = np.full((11, 3, 3), np.nan, np.float32)
        columns[[0, 7, 8]] = [0, 2, 1]
        values[[0, 7, 8]] = [np.eye(3), 2 * np.eye(3), -3 * np.eye(3)]
        x_np = np.arange(9, dtype=np.float32).reshape(3, 3)
        y_np = np.ones((3, 3), np.float32)
        A = make_matrix(offsets, columns, values, 3, self.device, counts)
        x = wp.array(x_np, dtype=wp.vec3, device=self.device)
        for group in packed.GROUP_SIZES:
            y = wp.array(y_np, dtype=wp.vec3, device=self.device)
            packed.packed_bsr_mv(A, x, y, -0.5, 0.25, group_size=group)
            expected, bound = reference(offsets, columns, values, x_np, y_np, -0.5, 0.25, counts)
            assert_reference(self, y.numpy(), expected, bound)

    def test_zero_alpha_and_zero_beta_do_not_read_nan_operands(self):
        A = make_matrix([0, 1], [0], np.eye(3)[None], 1, self.device)
        x = wp.array([[np.nan] * 3], dtype=wp.vec3, device=self.device)
        for group in packed.GROUP_SIZES:
            y = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3, device=self.device)
            packed.packed_bsr_mv(A, x, y, 0.0, -2.0, group_size=group)
            np.testing.assert_array_equal(y.numpy(), [[-2.0, -4.0, -6.0]])
            y.fill_(wp.vec3(np.nan))
            packed.packed_bsr_mv(A, x, y, 0.0, 0.0, group_size=group)
            np.testing.assert_array_equal(y.numpy(), [[0.0] * 3])

    def test_aliased_vectors_use_snapshot(self):
        offsets = np.array([0, 2, 4], np.int32)
        columns = np.array([0, 1, 0, 1], np.int32)
        values = np.arange(36, dtype=np.float32).reshape(4, 3, 3) / 8
        x_np = np.array([[1, -2, 4], [-3, 5, 2]], np.float32)
        A = make_matrix(offsets, columns, values, 2, self.device)
        for group in packed.GROUP_SIZES:
            x = wp.array(x_np, dtype=wp.vec3, device=self.device)
            scratch = wp.empty_like(x)
            packed.packed_bsr_mv(A, x, x, -0.5, 0.25, group_size=group, work_buffer=scratch)
            expected, bound = reference(offsets, columns, values, x_np, x_np, -0.5, 0.25)
            assert_reference(self, x.numpy(), expected, bound)

    def test_gradient_and_tape_paths_dispatch_to_original(self):
        A = make_matrix([0, 1], [0], np.eye(3)[None], 1, self.device)
        x = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3, device=self.device)
        y = wp.empty_like(x)
        with wp.Tape(), patch.object(packed, "bsr_mv", wraps=packed.bsr_mv) as original:
            packed.packed_bsr_mv(A, x, y)
            original.assert_called_once()
        x.requires_grad = True
        with patch.object(packed, "bsr_mv", wraps=packed.bsr_mv) as original:
            packed.packed_bsr_mv(A, x, y)
            original.assert_called_once()

    def test_partial_alias_and_invalid_work_buffer(self):
        if not self.device.is_cuda:
            self.skipTest("CUDA custom-path overlap contract")
        A = make_matrix([0, 1, 2], [1, 0], np.eye(3)[None].repeat(2, axis=0), 2, self.device)
        original = np.arange(9, dtype=np.float32).reshape(3, 3)
        for group in packed.GROUP_SIZES:
            backing = wp.array(original, dtype=wp.vec3, device=self.device)
            x, y = backing[:2], backing[1:]
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                packed.packed_bsr_mv(A, x, y, group_size=group, work_buffer=x)
            packed.packed_bsr_mv(A, x, y, 0.5, -0.25, group_size=group)
            np.testing.assert_array_equal(y.numpy(), 0.5 * original[:2][::-1] - 0.25 * original[1:])

    def test_invalid_group_is_rejected_before_dispatch(self):
        for invalid in (True, 8.0, 0, 2, 64):
            with self.subTest(group=invalid), self.assertRaises(ValueError):
                packed.packed_bsr_mv(None, None, group_size=invalid)

    def test_captured_repeat_with_nonzero_beta_and_alias(self):
        if not self.device.is_cuda:
            self.skipTest("CUDA capture contract")
        A = make_matrix([0, 1, 2], [1, 0], np.eye(3)[None].repeat(2, axis=0), 2, self.device)
        original = np.array([[1, 2, -3], [4, -5, 6]], np.float32)
        x = wp.array(original, dtype=wp.vec3, device=self.device)
        initial = wp.clone(x)
        scratch = wp.empty_like(x)
        for group in packed.GROUP_SIZES:
            packed.packed_bsr_mv(A, x, x, 0.5, -0.25, group_size=group, work_buffer=scratch)
            with wp.ScopedCapture(device=self.device) as capture:
                wp.copy(x, initial)
                packed.packed_bsr_mv(A, x, x, 0.5, -0.25, group_size=group, work_buffer=scratch)
            for _ in range(3):
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(x.numpy(), 0.5 * original[::-1] - 0.25 * original)


if __name__ == "__main__":
    unittest.main()
