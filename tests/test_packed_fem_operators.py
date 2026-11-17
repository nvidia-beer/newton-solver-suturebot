# SPDX-License-Identifier: Apache-2.0
"""Packed operator contracts with independent dense Dirichlet/rigid systems."""

import gc
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp
from test_packed_bsr import assert_reference, make_matrix, reference
from warp.optim.linear import aslinearoperator

from newton_solver_suturebot._src.solvers.coupled.acceleration.shaft_preconditioner import (
    _ShaftPreconditioner as ShaftPreconditioner,
)
from newton_solver_suturebot._src.solvers.fem import packed_bsr
from newton_solver_suturebot._src.solvers.fem.fixed_particles import FixedParticles


class PackedFEMOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))

    def fixture(self, n=7):
        rng = np.random.default_rng(174)
        offsets = np.arange(n + 1, dtype=np.int32) * n
        columns = np.tile(np.arange(n, dtype=np.int32), n)
        scalar = rng.normal(size=(3 * n, 3 * n))
        scalar = (scalar.T @ scalar + 4 * np.eye(3 * n)).astype(np.float32)
        values = scalar.reshape(n, 3, n, 3).transpose(0, 2, 1, 3).reshape(-1, 3, 3).copy()
        A = make_matrix(offsets, columns, values, n, self.device)
        return A, scalar, offsets, columns, values

    def test_three_operand_affine_and_alias_semantics(self):
        A, _, offsets, columns, values = self.fixture()
        rng = np.random.default_rng(42)
        x_np, y_np = rng.normal(size=(2, 7, 3)).astype(np.float32)
        operator = packed_bsr.packed_aslinearoperator(A)
        self.assertEqual(hasattr(operator, "_packed_bsr_group_size"), self.device.is_cuda)
        for alpha, beta in ((1.0, 0.0), (-0.5, 0.25), (0.0, -2.0)):
            for alias in ("separate", "affine", "input"):
                if alias == "input" and not self.device.is_cuda:
                    continue  # The unchanged Warp adapter has no general x/z guarantee.
                with self.subTest(alpha=alpha, beta=beta, alias=alias):
                    x = wp.array(x_np, dtype=wp.vec3, device=self.device)
                    y = wp.array(y_np if beta else np.full_like(y_np, np.nan), dtype=wp.vec3, device=self.device)
                    z = {"input": x, "affine": y}.get(alias, wp.empty_like(x))
                    operator.matvec(x, y, z, alpha, beta)
                    expected, bound = reference(offsets, columns, values, x_np, y_np, alpha, beta)
                    assert_reference(self, z.numpy(), expected, bound)

    def test_fixed_full_and_compact_operators_match_dense_reduction(self):
        A, dense, _, _, _ = self.fixture(n=17)
        model = SimpleNamespace(
            device=self.device, particle_count=17, particle_q=wp.zeros(17, dtype=wp.vec3, device=self.device)
        )
        fixed = FixedParticles(model, A, [0, 5, 16])
        fixed.prepare(A)
        mask = np.repeat(np.isin(np.arange(17), [0, 5, 16]), 3)
        projected = dense.copy()
        projected[mask, :] = 0
        projected[:, mask] = 0
        projected[np.where(mask)[0], np.where(mask)[0]] = 1
        x_np = np.random.default_rng(16).normal(size=(17, 3)).astype(np.float32)
        y_np = -0.25 * x_np
        x = wp.array(x_np, dtype=wp.vec3, device=self.device)
        y = wp.array(y_np, dtype=wp.vec3, device=self.device)
        z = wp.empty_like(x)
        fixed.operator.matvec(x, y, z, -0.5, 0.25)
        np.testing.assert_allclose(
            z.numpy().ravel(), -0.5 * projected @ x_np.ravel() + 0.25 * y_np.ravel(), rtol=3e-6, atol=3e-6
        )
        np.testing.assert_array_equal(z.numpy()[[0, 5, 16]], -0.5 * x_np[[0, 5, 16]] + 0.25 * y_np[[0, 5, 16]])
        if self.device.is_cuda:
            self.assertEqual(fixed.operator._packed_bsr_group_size, 16)
            self.assertEqual(fixed.compact_operator._packed_bsr_group_size, 16)
            free = np.flatnonzero(~np.isin(np.arange(17), [0, 5, 16]))
            cx = wp.array(x_np[free], dtype=wp.vec3, device=self.device)
            cy = wp.array(y_np[free], dtype=wp.vec3, device=self.device)
            cz = wp.empty_like(cx)
            fixed.compact_operator.matvec(cx, cy, cz, -0.5, 0.25)
            np.testing.assert_allclose(
                cz.numpy().ravel(),
                -0.5 * dense[np.ix_(~mask, ~mask)] @ x_np[free].ravel() + 0.25 * y_np[free].ravel(),
                rtol=3e-6,
                atol=3e-6,
            )
        # Projection must never overwrite the physical matrix used by predictor.
        np.testing.assert_array_equal(
            A.values.numpy().reshape(17, 17, 3, 3).transpose(0, 2, 1, 3).reshape(51, 51), dense
        )

    def test_shaft_projected_mass_block_retains_rigid_and_fixed_terms(self):
        A, dense, _, _, _ = self.fixture(n=17)
        model = SimpleNamespace(
            device=self.device, particle_count=17, particle_q=wp.zeros(17, dtype=wp.vec3, device=self.device)
        )
        fixed = FixedParticles(model, A, [0, 5, 16])
        fixed.prepare(A)
        pc = ShaftPreconditioner.__new__(ShaftPreconditioner)
        pc.device, pc.n, pc.count = self.device, 19, 0
        pc.fixed = fixed.mask
        # A track reserve of one row without a sample: the operator is the mass block alone.
        pc.capacity = 1
        track = SimpleNamespace(
            _track_nodes=wp.zeros(1, dtype=wp.vec4i, device=self.device),
            _track_bary=wp.zeros(1, dtype=wp.vec4, device=self.device),
            _live=wp.zeros(3, dtype=int, device=self.device),
        )
        pc.matrices = wp.full(1, wp.mat33(*([np.nan] * 9)), dtype=wp.mat33, device=self.device)
        pc.levers = wp.zeros(1, dtype=wp.vec3, device=self.device)
        pc.shears = wp.zeros(1, dtype=wp.mat33, device=self.device)
        pc.time_step = wp.ones(1, dtype=float, device=self.device)
        pc.coupler = SimpleNamespace(fem_solver=SimpleNamespace(_fixed_particles=fixed, A_bsr=A), interaction=track)
        rigid = np.arange(36, dtype=np.float32).reshape(6, 6) / 64 + np.eye(6, dtype=np.float32)
        pc.rigid_blocks = wp.array(
            np.stack((rigid[:3, :3], rigid[3:, 3:], rigid[:3, 3:], rigid[3:, :3])), dtype=wp.mat33, device=self.device
        )
        pc.work = wp.empty(19, dtype=wp.vec3, device=self.device)
        mask = np.repeat(np.isin(np.arange(17), [0, 5, 16]), 3)
        expected_matrix = np.zeros((57, 57))
        expected_matrix[:51, :51] = dense
        expected_matrix[np.where(mask)[0], :] = 0
        expected_matrix[:, np.where(mask)[0]] = 0
        expected_matrix[np.where(mask)[0], np.where(mask)[0]] = 1
        expected_matrix[51:, 51:] = rigid
        x_np = np.random.default_rng(91).normal(size=(19, 3)).astype(np.float32)
        y_np = 0.25 * x_np
        x, y = (wp.array(a, dtype=wp.vec3, device=self.device) for a in (x_np, y_np))
        z = wp.empty_like(x)
        pc.matvec(x, y, z, -0.5, 0.25)
        np.testing.assert_allclose(
            z.numpy().ravel(), -0.5 * expected_matrix @ x_np.ravel() + 0.25 * y_np.ravel(), rtol=3e-6, atol=3e-6
        )

    def test_capture_owners_keep_old_storage_and_alias_snapshots_alive(self):
        if not self.device.is_cuda:
            self.skipTest("CUDA ownership/capture contract")
        A, _, _, _, _ = self.fixture()
        operator = packed_bsr.packed_aslinearoperator(A)
        old_values = A.values
        x_np = np.random.default_rng(92).normal(size=(7, 3)).astype(np.float32)
        x = wp.array(x_np, dtype=wp.vec3, device=self.device)
        initial = wp.clone(x)
        y = wp.array(0.25 * x_np, dtype=wp.vec3, device=self.device)
        operator.matvec(x, y, x, -0.5, 0.25)  # Warm alias path.
        expected = x.numpy()
        with wp.ScopedCapture(device=self.device) as capture:
            wp.copy(x, initial)
            operator.matvec(x, y, x, -0.5, 0.25)
        A.values = wp.clone(A.values)
        A.values.zero_()
        self.assertTrue(any(owner is old_values for owner in operator._packed_bsr_owners))
        gc.collect()
        for _ in range(3):
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(x.numpy(), expected)

    def test_active_tape_and_gradient_operand_use_original(self):
        A, _, _, _, _ = self.fixture()
        x = wp.ones(7, dtype=wp.vec3, device=self.device)
        y, z = wp.empty_like(x), wp.empty_like(x)
        operator = packed_bsr.packed_aslinearoperator(A)
        if not self.device.is_cuda:
            self.assertFalse(hasattr(operator, "_packed_bsr_owners"))
            return
        original = aslinearoperator(A)
        delegated = SimpleNamespace(matvec=Mock(wraps=original.matvec))
        # Instrument our factory dependency, not Warp's read-only matvec property.
        with patch.object(packed_bsr, "aslinearoperator", return_value=delegated):
            operator = packed_bsr.packed_aslinearoperator(A)
        expected = wp.empty_like(z)
        original.matvec(x, y, expected, 1.0, 0.0)
        z.fill_(wp.vec3(np.nan))
        with wp.Tape():
            operator.matvec(x, y, z, 1.0, 0.0)
        delegated.matvec.assert_called_once()
        np.testing.assert_array_equal(z.numpy(), expected.numpy())
        delegated.matvec.reset_mock()
        y.requires_grad = True
        z.fill_(wp.vec3(np.nan))
        operator.matvec(x, y, z, 1.0, 0.0)
        delegated.matvec.assert_called_once()
        np.testing.assert_array_equal(z.numpy(), expected.numpy())


if __name__ == "__main__":
    unittest.main()
