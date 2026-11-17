# SPDX-License-Identifier: Apache-2.0
"""Independent sum/error oracles, short tails and capture for mesh-edge means."""

import math
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot._src.geometry import mesh_refit as refit
from newton_solver_suturebot._src.utils.capture import ScopedCapture

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


@wp.kernel(enable_backward=False)
def sum_serial(partial: wp.array[wp.float64], result: wp.array[wp.float64]):
    total = wp.float64(0)
    for i in range(partial.shape[0]):
        total += partial[i]
    result[0] = total


@wp.kernel(enable_backward=False)
def sum_cooperative(partial: wp.array[wp.float64], result: wp.array[wp.float64]):
    lane = wp.tid()
    total = refit._cooperative_total(partial, lane)
    if lane == 0:
        result[0] = total


@wp.func_native("""
mean.data[0] = ((const wp::Mesh*)id)->average_edge_length;
""")
def read_mean_native(id: wp.uint64, mean: wp.array[float]): ...


@wp.kernel(enable_backward=False)
def read_mean(id: wp.uint64, mean: wp.array[float]):
    read_mean_native(id, mean)


def triangle_mesh(device):
    return wp.Mesh(
        wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3, device=device),
        wp.array([0, 1, 2], dtype=int, device=device),
    )


class EdgeMeanTests(unittest.TestCase):
    def test_serial_cpu_path_and_original_reduction_remain_exact(self):
        values = np.array([1, 2**-30, 3, 2**30, 2, 4], dtype=np.float32)
        edges = wp.array(values, dtype=float, device="cpu")
        partial = wp.empty(1, dtype=wp.float64, device="cpu")
        mesh = triangle_mesh("cpu")
        launch, seen = wp.launch, []

        def record(kernel, *args, **kwargs):
            seen.append(kernel)
            return launch(kernel, *args, **kwargs)

        with patch.object(wp, "launch", side_effect=record):
            refit.update_edge_mean(mesh.id, edges, partial, wp.get_device("cpu"))
        self.assertEqual(seen, [refit.reduce_edges, refit.set_mean])
        expected = 0.0
        for value in values:
            expected += float(value)
        np.testing.assert_array_equal(partial.numpy(), [expected])
        mean = wp.empty(1, dtype=float, device="cpu")
        wp.launch(read_mean, 1, [mesh.id, mean], device="cpu")
        np.testing.assert_array_equal(mean.numpy(), np.array([expected / 3], dtype=np.float32))


class CooperativeEdgeCudaTests(unittest.TestCase):
    def setUp(self):
        if not wp.get_device(DEVICE).is_cuda:
            self.skipTest("Cooperative warp reductions require CUDA")

    def reduce(self, values, *, capture=False):
        values = np.asarray(values, dtype=np.float32)
        edges = wp.array(values, dtype=float, device=DEVICE)
        partial = wp.empty((len(values) + 255) // 256, dtype=wp.float64, device=DEVICE)
        total = wp.empty(1, dtype=wp.float64, device=DEVICE)

        def run():
            wp.launch(refit.reduce_edges_cooperative, partial.size * 32, [edges, partial], device=DEVICE, block_dim=32)
            wp.launch(sum_cooperative, 32, [partial, total], device=DEVICE, block_dim=32)

        run()
        if capture:
            with ScopedCapture(device=DEVICE) as graph:
                run()
            edges.assign(values[::-1].copy())
            wp.capture_launch(graph.graph)
            values = values[::-1].copy()
        return values, partial.numpy(), float(total.numpy()[0])

    def test_small_tails_and_actual_scale_match_serial_and_fsum_exactly(self):
        rng = np.random.default_rng(839)
        for size in (1, 2, 31, 32, 33, 255, 256, 257, 511, 512, 513, 8193, 17_500, 111_000):
            with self.subTest(size=size):
                values, partial, total = self.reduce(rng.uniform(1e-5, 0.01, size), capture=size in (1, 257, 17_500))
                expected = [math.fsum(map(float, values[i : i + 256])) for i in range(0, size, 256)]
                np.testing.assert_array_equal(partial, expected)
                self.assertEqual(total, math.fsum(map(float, values)))
                original = wp.empty(len(partial), dtype=wp.float64, device=DEVICE)
                old_total = wp.empty(1, dtype=wp.float64, device=DEVICE)
                wp.launch(
                    refit.reduce_edges,
                    len(partial),
                    [wp.array(values, dtype=float, device=DEVICE), original],
                    device=DEVICE,
                )
                wp.launch(sum_serial, 1, [original, old_total], device=DEVICE)
                np.testing.assert_array_equal(partial, original.numpy())
                self.assertEqual(total, float(old_total.numpy()[0]))

    def test_wide_exponent_range_obeys_fp64_tree_error_bound(self):
        rng = np.random.default_rng(841)
        for size in (257, 8193, 111_000):
            for pattern in (0, 1):
                values = np.exp2(rng.uniform(-100, 100, size)).astype(np.float32)
                if pattern:
                    values.sort()
                    values = values[::-1].copy()
                values, partial, total = self.reduce(values)
                unit = np.finfo(np.float64).eps / 2
                # At most eight serial additions per lane and five shuffle
                # levels in a chunk, then ceil(chunks/32)+five in the mean.
                depth = 13 + math.ceil(len(partial) / 32) + 5
                gamma = depth * unit / (1 - depth * unit)
                exact = math.fsum(map(float, values))
                self.assertLessEqual(abs(total - exact), gamma * exact)
                for start, value in zip(range(0, size, 256), partial, strict=True):
                    local = math.fsum(map(float, values[start : start + 256]))
                    self.assertLessEqual(abs(value - local), (13 * unit / (1 - 13 * unit)) * local)

    def test_set_mean_uses_complete_warp_and_current_device_values(self):
        mesh = triangle_mesh(DEVICE)
        edges = wp.array(np.arange(1, 259, dtype=np.float32), dtype=float, device=DEVICE)
        partial = wp.empty(2, dtype=wp.float64, device=DEVICE)
        mean = wp.empty(1, dtype=float, device=DEVICE)
        refit.update_edge_mean(mesh.id, edges, partial, wp.get_device(DEVICE))
        with ScopedCapture(device=DEVICE) as graph:
            refit.update_edge_mean(mesh.id, edges, partial, wp.get_device(DEVICE))
            wp.launch(read_mean, 1, [mesh.id, mean], device=DEVICE)
        for scale in (1, 2, 0.25):
            values = np.arange(1, 259, dtype=np.float32) * scale
            edges.assign(values)
            wp.capture_launch(graph.graph)
            expected = np.float32(math.fsum(map(float, values)) / 3)
            np.testing.assert_array_equal(mean.numpy(), [expected])


if __name__ == "__main__":
    unittest.main()
