# SPDX-License-Identifier: Apache-2.0
"""Generic FEM kernels declare the argument types the solver passes: a first launch changes no module hash.

A kernel with an ``Any`` argument gets an overload at the first launch of each argument type, which changes
the hash of its module and loads another build of it. The types that the solver's workspaces pass are declared
where the kernels are defined, so the build that start-up loads by name is the build every later launch uses.
"""

import os
import unittest

import numpy as np
import warp as wp

from newton_solver_suturebot._src.solvers.fem import affine_guess, certified_cg, warp_basis_provider

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


def _hash(kernel):
    return kernel.module.get_module_hash(256)


class DeclaredOverloadTests(unittest.TestCase):
    def test_statistics_of_a_response_guess_take_the_plain_and_the_certified_workspace(self):
        kernel = affine_guess.finish_statistics
        before = _hash(kernel)
        iterations = wp.zeros(2, dtype=int, device=DEVICE)
        history = wp.zeros(1, dtype=int, device=DEVICE)
        counters = wp.zeros(len(affine_guess.COUNTERS), dtype=int, device=DEVICE)
        for scalar in (wp.float32, wp.float64):
            # As the workspaces pass them: a column of the tiled dot and a row of the residual pair.
            recursive = wp.zeros(1, dtype=scalar, device=DEVICE)
            residuals = wp.ones((2, 1), dtype=scalar, device=DEVICE)
            wp.launch(kernel, 1, [iterations, recursive, residuals[1], 160, history, counters], device=DEVICE)
            self.assertEqual(_hash(kernel), before, scalar)
        np.testing.assert_array_equal(history.numpy(), [1])
        self.assertEqual(counters.numpy()[9], 0)

    def test_quality_packet_of_the_column_engine_takes_its_single_precision_solve(self):
        kernel = warp_basis_provider._quality_packet
        before = _hash(kernel)
        sums = wp.array([0.25, 0.5], dtype=wp.float64, device=DEVICE)
        recursive = wp.array([2.0], dtype=wp.float32, device=DEVICE)
        residuals = wp.array([[0.0], [3.0]], dtype=wp.float32, device=DEVICE)
        iterations = wp.array([7, 0], dtype=int, device=DEVICE)
        slot = wp.zeros(1, dtype=int, device=DEVICE)
        quality = wp.zeros((2, 4), dtype=wp.float64, device=DEVICE)
        wp.launch(kernel, 1, [sums, recursive, residuals[1], iterations, slot, quality], device=DEVICE)
        self.assertEqual(_hash(kernel), before)
        np.testing.assert_array_equal(quality.numpy()[0], [0.75, 2.0, 3.0, 7.0])

    def test_certified_solve_takes_its_single_precision_right_hand_side_and_solution(self):
        before = _hash(certified_cg.copy_double)
        source = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3, device=DEVICE)
        target = wp.zeros(1, dtype=wp.vec3d, device=DEVICE)
        wp.launch(certified_cg.copy_double, 1, [source, target], device=DEVICE)
        self.assertEqual(_hash(certified_cg.copy_double), before)
        np.testing.assert_array_equal(target.numpy(), [[1.0, 2.0, 3.0]])
        offsets = wp.array([0, 1], dtype=int, device=DEVICE)
        columns = wp.array([0], dtype=int, device=DEVICE)
        values = wp.array([np.eye(3, dtype=np.float32) * 2], dtype=wp.mat33, device=DEVICE)
        residual = wp.zeros(1, dtype=wp.vec3d, device=DEVICE)
        correction = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        wp.launch(
            certified_cg.raw_residual,
            1,
            [offsets, columns, values, source, source, residual, correction],
            device=DEVICE,
        )
        self.assertEqual(_hash(certified_cg.raw_residual), before)
        np.testing.assert_array_equal(residual.numpy(), [[-1.0, -2.0, -3.0]])


if __name__ == "__main__":
    unittest.main()
