# SPDX-License-Identifier: Apache-2.0
"""Small independent thread queries preserve complete accepted-step state."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from test_worker_thread import scene_fixture

from newton_solver_suturebot import SolverThread
from newton_solver_suturebot._src.solvers.thread import kernels, tissue
from newton_solver_suturebot._src.solvers.thread.self_contact_pairs import evaluate_pairs, select_ordered
from newton_solver_suturebot._src.solvers.thread.worker_thread import WorkerThread

QUERY_KERNELS = (tissue._contact_planes, tissue._store_anchors, kernels.find_self_contacts)


class ThreadQueryPolicyTests(unittest.TestCase):
    def test_actual_launches_keep_cpu_and_large_batches_at_original_width(self):
        seen = {}
        launch = wp.launch

        def record(kernel, *args, **kwargs):
            if kernel in (*QUERY_KERNELS, evaluate_pairs, select_ordered):
                seen[kernel.key] = kwargs.get("block_dim", 256)
            return launch(kernel, *args, **kwargs)

        for elements in (90, 127, 128, 129):
            with self.subTest(elements=elements):
                scene, config = scene_fixture()
                config.thread_elements = elements
                worker = WorkerThread(scene, config, capture=False)
                seen.clear()

                with patch.object(wp, "launch", side_effect=record):
                    worker._advance_device(scene.state_0, 1 / 240)
                widths = (elements + 1, elements + 1, elements)
                expected = {
                    kernel.key: 1 if worker.device.is_cuda and count <= 128 else 256
                    for kernel, count in zip(QUERY_KERNELS, widths, strict=True)
                }
                if worker.solver._self_pairs is not None:
                    expected.pop(kernels.find_self_contacts.key)
                    expected[evaluate_pairs.key] = 128
                    expected[select_ordered.key] = 32
                self.assertEqual(seen, expected)

        # The bounded pair scratch also covers two independent small chains.
        scene, config = scene_fixture()
        worker = WorkerThread(scene, config, capture=False)
        solver = SolverThread(
            scene.model, worker.thread_model, n_envs=2, self_contact=True, contact_distance=0.00015, max_iterations=1
        )
        seen.clear()
        with patch.object(wp, "launch", side_effect=record):
            solver.step(None, None, None, None, 1 / 240)
        if solver._self_pairs is not None:
            self.assertEqual(seen, {evaluate_pairs.key: 128, select_ordered.key: 32})
        else:
            self.assertEqual(seen[kernels.find_self_contacts.key], 256)

    def test_friction_and_moving_tissue_snapshots_match_original_queries_exactly(self):
        scene, config = scene_fixture()
        worker = WorkerThread(scene, config, capture=False)
        launch = wp.launch
        body = scene.state_0.body_q.numpy().copy()
        particles = scene.state_0.particle_q.numpy().copy()
        checked = (
            *worker.state,
            worker.solver.contact_point,
            worker.solver.contact_normal,
            worker.tissue._face,
            worker.tissue._weights,
            worker.solver._self_xi,
            worker.solver._self_point,
            worker.solver._self_normal,
        )
        atomic_outputs = (worker.tissue.applied_force, worker._nodal_force)

        def original(kernel, *args, **kwargs):
            if kernel in QUERY_KERNELS:
                kwargs["block_dim"] = 256
            return launch(kernel, *args, **kwargs)

        for step in range(4):
            body[0, 0] += 2e-5
            particles[:, 2] += 1e-6
            scene.state_0.body_q.assign(body)
            scene.state_0.particle_q.assign(particles)
            before = [array.numpy().copy() for array in worker.state]
            with patch.object(worker.solver, "_self_pairs", None), patch.object(wp, "launch", side_effect=original):
                worker._advance_device(scene.state_0, 1 / 240)
            reference = [array.numpy().copy() for array in checked]
            # The unchanged float32 atomic scatter is not bitwise deterministic,
            # including when comparing the original width against itself. Each
            # vertex receives at most one weighted contribution per node. Bound
            # two sum orders by 2*gamma_n*sum(abs(force)), plus relaxation rounding.
            unit_roundoff = np.finfo(np.float32).eps / 2
            nodes = worker.thread_model.node_count
            gamma = nodes * unit_roundoff / (1 - nodes * unit_roundoff)
            force_sum = np.abs(worker.solver.contact_force.numpy()).sum(axis=0)
            old_force = next(
                saved for array, saved in zip(worker.state, before, strict=True) if array is atomic_outputs[0]
            )
            force_bound = 2 * gamma * force_sum + 8 * unit_roundoff * (force_sum + np.abs(old_force))
            for array, saved in zip(worker.state, before, strict=True):
                array.assign(saved)
            if worker.device.is_cuda:
                with wp.ScopedCapture(device=worker.device) as capture:
                    worker._advance_device(scene.state_0, 1 / 240)
                wp.capture_launch(capture.graph)
            else:
                worker._advance_device(scene.state_0, 1 / 240)
            for index, (expected, array) in enumerate(zip(reference, checked, strict=True)):
                with self.subTest(step=step, state_array=index):
                    if any(array is item for item in atomic_outputs):
                        self.assertTrue(np.all(np.abs(array.numpy() - expected) <= force_bound))
                    else:
                        np.testing.assert_array_equal(array.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
