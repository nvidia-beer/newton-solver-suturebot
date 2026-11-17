# SPDX-License-Identifier: Apache-2.0
"""Preserve original branch choices; separately bound fixed-branch FP32 error."""

import os
import unittest
from pathlib import Path

import _cooperative_projection_oracle as oracle
import numpy as np
import warp as wp
from _cooperative_projection_trace import traced

from newton_solver_suturebot._src.solvers.coupled.interface.insertion import _shaft_projection
from newton_solver_suturebot._src.solvers.coupled.interface.shaft_projection import cooperative_projection
from newton_solver_suturebot._src.solvers.coupled.interface.shaft_projection_tables import ShaftProjectionTables

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


@wp.kernel(enable_backward=False)
def original(
    queries: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    radii: wp.array[float],
    samples: wp.array[wp.vec3],
    output: wp.array2d[float],
):
    i = wp.tid()
    closest, tangent, distance, covered = _shaft_projection(queries[i], poses[i], samples, radii[i])
    for axis in range(3):
        output[i, axis] = closest[axis]
        output[i, axis + 3] = tangent[axis]
    output[i, 6] = distance
    output[i, 7] = float(covered)


class CooperativeGeometryTests(unittest.TestCase):
    def compare(self, samples, queries, poses, radii):
        if not wp.get_device(DEVICE).is_cuda:
            self.skipTest("Cooperative CUDA shuffle")

        def array(value, dtype):
            return wp.array(value, dtype=dtype, device=DEVICE)

        points = array(queries, wp.vec3)
        transform = array(poses, wp.transform)
        radius = array(radii, float)
        source = array(samples, wp.vec3)
        table = ShaftProjectionTables(source)
        table.refresh()
        n = len(queries)
        old = wp.empty((n, 8), dtype=float, device=DEVICE)
        actual = wp.empty_like(old)
        trace = wp.empty((n, 11), dtype=float, device=DEVICE)
        branch = wp.empty((n, 6), dtype=int, device=DEVICE)
        reference = wp.empty_like(branch)
        dirty = wp.ones(n, dtype=int, device=DEVICE)
        initialized = wp.zeros_like(dirty)
        generation = wp.zeros_like(dirty)
        counts = wp.zeros(2, dtype=int, device=DEVICE)
        wp.launch(original, n, [points, transform, radius, source, old], device=DEVICE)
        wp.launch(traced, n, [points, transform, radius, source, trace, reference], device=DEVICE)
        wp.launch(
            cooperative_projection,
            ((n * 32 + 255) // 256) * 256,
            [points, transform, radius, source, table.table, actual, branch, dirty, initialized, generation, counts],
            device=DEVICE,
            block_dim=256,
        )
        np.testing.assert_array_equal(trace.numpy()[:, :8].view(np.uint32), old.numpy().view(np.uint32))
        return old.numpy(), actual.numpy(), reference.numpy(), branch.numpy()

    def test_exact_ties_arc_endpoints_and_extrapolation(self):
        samples = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
        queries = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [-1, 0, 0], [1, 2, 0]], np.float32)
        poses = np.tile([0, 0, 0, 0, 0, 0, 1], (len(queries), 1))
        for radius in (0, 0.5, 1, 2, 5):
            old, new, expected, branch = self.compare(samples, queries, poses, np.full(len(queries), radius))
            np.testing.assert_array_equal(branch, expected)
            np.testing.assert_array_equal(new.view(np.uint32), old.view(np.uint32))

    def test_degenerate_and_nonfinite_samples_or_radius_use_original_map(self):
        for samples in (
            [[0, 0, 0]],
            [],
            [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [1e-30, 0, 0]],
            [[0, 0, 0], [np.inf, 0, 0]],
        ):
            old, new, _, branch = self.compare(
                np.asarray(samples).reshape(-1, 3),
                [[0.5, 0.1, 0], [2, 0, 0]],
                [[0, 0, 0, 0, 0, 0, 1]] * 2,
                [0.01, 0.01],
            )
            np.testing.assert_array_equal(new.view(np.uint32), old.view(np.uint32))
            np.testing.assert_array_equal(branch, -1)
        old, new, _, _ = self.compare(
            [[0, 0, 0], [1, 0, 0]], [[0.5, 0.1, 0]] * 3, [[0, 0, 0, 0, 0, 0, 1]] * 3, [-1, np.nan, np.inf]
        )
        np.testing.assert_array_equal(new.view(np.uint32), old.view(np.uint32))

    def test_curved_random_queries_keep_original_branch_and_double_conditioning(self):
        with np.load(Path(__file__).with_name("cooperative_projection_cases.npz")) as fixture:
            samples, queries, poses, radii = (fixture[key] for key in ("samples", "queries", "poses", "radii"))
            old, new, expected, branch = self.compare(samples, queries, poses, radii)
            np.testing.assert_array_equal(branch, expected)
            np.testing.assert_array_equal(new[:, 6:8], old[:, 6:8])
            for i in range(len(queries)):
                exact, _, bound, _ = oracle.evaluate(
                    samples, queries[i], poses[i], radii[i], segment_override=int(expected[i, 0])
                )
                # This is an operation-count/conditioning diagnostic, not a
                # replacement for physical coupled residual acceptance. Global
                # FP64 nearest choices can differ at sub-ULP distance ties.
                self.assertTrue(np.all(np.abs(old[i] - exact[:8]) <= bound[:8]), f"original {i}")
                self.assertTrue(np.all(np.abs(new[i] - exact[:8]) <= bound[:8]), f"candidate {i}")

    def test_double_oracle_analytic_straight_segment(self):
        result, branch, _, _ = oracle.evaluate([[0, 0, 0], [1, 0, 0]], [0.5, 0.1, 0], [0, 0, 0, 0, 0, 0, 1], 0.1)
        np.testing.assert_allclose(result[:6], [0.5, 0, 0, 1, 0, 0], rtol=0, atol=0)
        self.assertAlmostEqual(result[6], 0.01)
        self.assertEqual(branch[0], 0)


if __name__ == "__main__":
    unittest.main()
