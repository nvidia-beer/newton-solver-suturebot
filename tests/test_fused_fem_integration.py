# SPDX-License-Identifier: Apache-2.0
"""Only the proven compact FEM operator may use the fused recurrence."""

import os
import unittest
from types import SimpleNamespace

import numpy as np
import test_fixed_particles as fixed_tests
import warp as wp
from test_packed_bsr import make_matrix

from newton_solver_suturebot._src.solvers.fem.chebyshev import ChebyshevJacobi
from newton_solver_suturebot._src.solvers.fem.fixed_particles import FixedParticles
from newton_solver_suturebot._src.solvers.fem.fused_chebyshev import FusedPackedChebyshev


class FusedFEMIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))

    def test_full_indexed_bound_is_unchanged_for_the_proven_compact_mapping(self):
        rng = np.random.default_rng(193)
        n = 19
        values = rng.normal(size=(3 * n, 3 * n))
        dense = (values.T @ values + 5 * np.eye(3 * n)).astype(np.float32)
        blocks = dense.reshape(n, 3, n, 3).transpose(0, 2, 1, 3).copy()
        matrix = make_matrix(np.arange(n + 1) * n, np.tile(np.arange(n), n), blocks.reshape(-1, 3, 3), n, self.device)
        model = SimpleNamespace(
            device=self.device, particle_count=n, particle_q=wp.zeros(n, dtype=wp.vec3, device=self.device)
        )
        fixed = FixedParticles(model, matrix, [0, 5, 18])
        fixed.prepare(matrix)
        if not self.device.is_cuda:
            fixed._prepare_compact(matrix, matrix.offsets.numpy(), matrix.columns.numpy())
            free = [i for i in range(n) if i not in (0, 5, 18)]
            fixed.compact_matrix.values.assign(blocks[np.ix_(free, free)].reshape(-1, 3, 3))
        diagonal = np.linalg.inv(blocks[np.arange(n), np.arange(n)]).astype(np.float32)
        inverse = wp.array(diagonal, dtype=wp.mat33, device=self.device)
        fixed.gather_diagonal(inverse)
        common = dict(degree=4, ratio=256.0, device=self.device)
        original = ChebyshevJacobi(fixed.compact_operator, fixed.compact_diagonal, **common)
        fused = FusedPackedChebyshev(
            fixed.compact_operator,
            fixed.compact_diagonal,
            matrix=fixed.compact_matrix,
            compact=True,
            bound_source=(matrix, fixed.mask, inverse),
            **common,
        )
        rhs = wp.array(rng.normal(size=(16, 3)), dtype=wp.vec3, device=self.device)
        expected, actual = wp.empty_like(rhs), wp.empty_like(rhs)
        for pre in (original, fused):
            pre.refresh(matrix, fixed=fixed.mask, bound_diagonal=inverse)
        np.testing.assert_array_equal(original.coefficients.numpy(), fused.coefficients.numpy())
        self.assertEqual(fused.can_fuse(rhs, actual), self.device.is_cuda)
        original.matvec(rhs, expected, expected, 1.0, 0.0)
        fused.matvec(rhs, actual, actual, 1.0, 0.0)
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-5, atol=2e-6)
        # Identical contents are insufficient to prove another indexing owner.
        fused.refresh(matrix, fixed=wp.clone(fixed.mask), bound_diagonal=inverse)
        self.assertFalse(fused.can_fuse(rhs, actual))

    def test_actual_compact_workspace_selects_fusion_and_full_operator_does_not(self):
        if not self.device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("Conditional CUDA FEM integration")
        fixture = fixed_tests.TestFixedParticles()
        fixture.device = str(self.device)
        model = fixture.model()
        solver = fixture.solver(model)
        solver.linear_solver_chebyshev_degree = 4
        solver.linear_solver_check_every = 0
        solver.linear_solver_use_cuda_graph = True
        current, following = model.state(), model.state()
        forces = wp.array(
            [[0, 0, 0], [0, 0, 0], [0.01, 0.02, 0.03], [-0.02, 0, 0.01]], dtype=wp.vec3, device=self.device
        )
        solver.step(current, following, None, None, fixture.dt, external_forces=forces)
        fixed = solver._fixed_particles
        compact = solver._chebyshev_preconditioner(fixed.compact_operator, fixed.compact_diagonal)
        self.assertIsInstance(compact, FusedPackedChebyshev)
        self.assertTrue(compact.can_fuse(fixed.compact_rhs, fixed.compact_increment))
        original = ChebyshevJacobi(
            fixed.compact_operator, fixed.compact_diagonal, degree=4, ratio=256.0, device=self.device
        )
        original.refresh(solver.A_bsr, fixed=fixed.mask, bound_diagonal=solver._pc_inv_diag)
        np.testing.assert_array_equal(compact.coefficients.numpy(), original.coefficients.numpy())
        full = solver._chebyshev_preconditioner(fixed.operator, solver._pc_inv_diag)
        self.assertIs(type(full), ChebyshevJacobi)
        np.testing.assert_array_equal(following.particle_q.numpy()[:2], model.particle_q.numpy()[:2])
        np.testing.assert_array_equal(following.particle_qd.numpy()[:2], 0)


if __name__ == "__main__":
    unittest.main()
