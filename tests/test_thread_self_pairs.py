# SPDX-License-Identifier: Apache-2.0
"""Parity with the original complete ordered search, including stale outputs."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot._src.solvers.thread.kernels import find_self_contacts
from newton_solver_suturebot._src.solvers.thread.self_contact_pairs import PairwiseSelfContacts, select_ordered
from newton_solver_suturebot._src.utils.capture import ScopedCapture

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


@wp.kernel(enable_backward=False)
def custom_contacts(
    q: wp.array[wp.float64],
    nodes: int,
    reach: wp.float64,
    skip: int,
    xi: wp.array[wp.float64],
    point: wp.array[wp.vec3d],
    normal: wp.array[wp.vec3d],
):
    element = wp.tid()
    xi[element] = wp.float64(0.125)
    normal[element] = wp.vec3d()


def coordinates(points):
    result = np.zeros((*points.shape[:-1], 6))
    result[..., :3] = points
    result[..., 3] = 1.0
    return result.reshape(-1)


class PairwiseTests(unittest.TestCase):
    def test_original_query_handles_gradients_custom_query_and_large_batches(self):
        from test_worker_thread import scene_fixture

        from newton_solver_suturebot import SolverThread
        from newton_solver_suturebot._src.solvers.thread import kernels
        from newton_solver_suturebot._src.solvers.thread.worker_thread import WorkerThread

        scene, config = scene_fixture()
        config.thread_iterations = 2
        worker = WorkerThread(scene, config, capture=False)
        solver = worker.solver
        with patch.object(kernels, "find_self_contacts", custom_contacts):
            worker.advance(1 / 240)
        np.testing.assert_array_equal(solver._self_xi.numpy(), 0.125)
        large = SolverThread(
            scene.model, worker.thread_model, n_envs=17, self_contact=True, contact_distance=0.00015, max_iterations=1
        )
        self.assertIsNone(large._self_pairs)
        if solver._self_pairs is not None:
            solver.q = wp.clone(solver.q, requires_grad=True)
            with patch.object(solver._self_pairs, "find", side_effect=AssertionError("gradient shortcut")):
                worker.advance(1 / 240)
            solver.q = wp.clone(solver.q, requires_grad=False)
            with (
                patch.object(solver._self_pairs, "find", side_effect=AssertionError("tape shortcut")),
                wp.Tape(),
            ):
                worker.advance(1 / 240)

    def compare(self, points, reach, skip=2):
        if points.ndim == 2:
            points = points[None]
        worlds, nodes, _ = points.shape
        n = worlds * (nodes - 1)
        q = wp.array(coordinates(points), dtype=wp.float64, device=DEVICE)
        reference = (
            wp.full(n, 0.375, dtype=wp.float64, device=DEVICE),
            wp.full(n, wp.vec3d(2.0, 3.0, 4.0), dtype=wp.vec3d, device=DEVICE),
            wp.full(n, wp.vec3d(5.0, 6.0, 7.0), dtype=wp.vec3d, device=DEVICE),
        )
        actual = tuple(wp.clone(a) for a in reference)
        finder = PairwiseSelfContacts(nodes, worlds, DEVICE)
        wp.launch(find_self_contacts, n, [q, nodes, reach, skip, *reference], device=DEVICE)
        finder.find(q, reach, skip, *actual)
        for expected, array in zip(reference, actual, strict=True):
            np.testing.assert_array_equal(array.numpy(), expected.numpy())
        return q, finder, actual

    def test_random_chains_and_independent_worlds_match_every_output(self):
        rng = np.random.default_rng(381)
        for worlds, nodes in ((1, 3), (1, 91), (2, 38), (8, 17)):
            points = rng.normal(size=(worlds, nodes, 3)).cumsum(axis=1) * 0.002
            points += np.arange(worlds)[:, None, None] * 10.0
            for reach in (0.0006, 0.05):
                with self.subTest(worlds=worlds, nodes=nodes, reach=reach):
                    self.compare(points, reach)

    def test_parallel_and_exact_crossing_segments_preserve_original_rules(self):
        # Nonconsecutive parallel chords and exact intersections exercise the
        # original parallel branch and its intentional distance>0 predicate.
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 1, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0]],
            dtype=float,
        )
        self.compare(points, 2.0)

    def test_no_candidate_retains_old_xi_and_point_but_clears_normal(self):
        points = np.column_stack((np.arange(16), np.zeros((16, 2))))
        _, _, arrays = self.compare(points, 0.001)
        np.testing.assert_array_equal(arrays[0].numpy(), 0.375)
        np.testing.assert_array_equal(arrays[1].numpy(), np.tile((2.0, 3.0, 4.0), (15, 1)))
        np.testing.assert_array_equal(arrays[2].numpy(), 0.0)

    def test_ordered_selection_preserves_first_tie_strict_reach_and_zero(self):
        distances = wp.array([[1.0, 0.5, 0.5, 0.0], [1.0, 1.0, 0.0, 1.0]], dtype=wp.float64, device=DEVICE)
        fractions = wp.array(np.tile(np.arange(4), (2, 1)) / 4, dtype=wp.float64, device=DEVICE)
        points = wp.array(np.arange(24).reshape(2, 4, 3), dtype=wp.vec3d, device=DEVICE)
        normals = wp.array(np.arange(24, 48).reshape(2, 4, 3), dtype=wp.vec3d, device=DEVICE)
        xi = wp.full(2, -1.0, dtype=wp.float64, device=DEVICE)
        point = wp.full(2, wp.vec3d(-1.0), dtype=wp.vec3d, device=DEVICE)
        normal = wp.full(2, wp.vec3d(-1.0), dtype=wp.vec3d, device=DEVICE)
        wp.launch(select_ordered, 2, [1.0, distances, fractions, points, normals, xi, point, normal], device=DEVICE)
        np.testing.assert_array_equal(xi.numpy(), [0.25, -1.0])
        np.testing.assert_array_equal(point.numpy(), [[3, 4, 5], [-1, -1, -1]])
        np.testing.assert_array_equal(normal.numpy(), [[27, 28, 29], [0, 0, 0]])

    def test_second_call_reuses_storage_but_not_candidates(self):
        rng = np.random.default_rng(413)
        points = rng.normal(size=(91, 3)) * 0.001
        q, finder, actual = self.compare(points, 0.01)
        owners = tuple(array.ptr for array in finder.storage)
        q.assign(coordinates(np.column_stack((np.arange(91), np.zeros((91, 2))))))
        expected = tuple(wp.clone(array) for array in actual)
        wp.launch(find_self_contacts, 90, [q, 91, 0.01, 2, *expected], device=DEVICE)
        finder.find(q, 0.01, 2, *actual)
        self.assertEqual(tuple(array.ptr for array in finder.storage), owners)
        for wanted, array in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(array.numpy(), wanted.numpy())

    def test_capacity_and_coordinate_layout_fail_before_launch(self):
        with self.assertRaisesRegex(ValueError, "bound"):
            PairwiseSelfContacts(513, 1, DEVICE)
        finder = PairwiseSelfContacts(3, 1, DEVICE)
        with self.assertRaisesRegex(ValueError, "layout"):
            finder.find(wp.zeros(6, dtype=wp.float64, device=DEVICE), 0.1, 2, None, None, None)

    def test_actual_accepted_thread_retains_all_contacts_and_iterations(self):
        from test_worker_thread import scene_fixture

        from newton_solver_suturebot._src.solvers.thread.worker_thread import WorkerThread

        scene, config = scene_fixture()
        config.thread_iterations = 2
        reference = WorkerThread(scene, config, capture=False)
        actual = WorkerThread(scene, config, capture=False)
        reference.solver._self_pairs = None
        if wp.get_device(DEVICE).is_cuda:
            self.assertIsNotNone(actual.solver._self_pairs)

        poses = scene.state_0.body_q.numpy().copy()
        particles = scene.state_0.particle_q.numpy().copy()
        for step in range(3):
            poses[0, 0] += 2e-6
            poses[1, 1] -= 1e-6
            particles[:, 2] += 1e-7
            scene.state_0.body_q.assign(poses)
            scene.state_0.particle_q.assign(particles)
            reference.advance(1 / 240)
            actual.advance(1 / 240)
            # This includes rigid capsule friction, self-contact state,
            # attachment, tissue reactions and all lagged history. The existing
            # float32 atomic force scatter may differ in summation order on
            # CUDA, so use the established worker-graph parity bound there.
            for expected, array in zip(reference.state, actual.state, strict=True):
                with self.subTest(step=step, shape=array.shape):
                    if wp.get_device(DEVICE).is_cuda:
                        np.testing.assert_allclose(array.numpy(), expected.numpy(), rtol=1e-6, atol=1e-10)
                    else:
                        np.testing.assert_array_equal(array.numpy(), expected.numpy())
            np.testing.assert_array_equal(actual.solver.iterations.numpy(), reference.solver.iterations.numpy())
            self.assertEqual(actual.accepted_steps, reference.accepted_steps)
            self.assertEqual(actual.physical_time_s, reference.physical_time_s)

    @unittest.skipUnless(DEVICE.startswith("cuda"), "capture requires CUDA")
    def test_captured_search_observes_moving_coordinates(self):
        points = np.random.default_rng(711).normal(size=(91, 3)) * 0.002
        q, finder, actual = self.compare(points, 0.005)
        with ScopedCapture(device=DEVICE) as capture:
            finder.find(q, 0.005, 2, *actual)
        for scale in (1.2, 0.8):
            q.assign(coordinates(points * scale))
            expected = tuple(wp.clone(array) for array in actual)
            wp.launch(find_self_contacts, 90, [q, 91, 0.005, 2, *expected], device=DEVICE)
            wp.capture_launch(capture.graph)
            for wanted, array in zip(expected, actual, strict=True):
                np.testing.assert_array_equal(array.numpy(), wanted.numpy())


if __name__ == "__main__":
    unittest.main()
