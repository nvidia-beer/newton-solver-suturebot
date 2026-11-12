# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Worker thread uses accepted physics and feeds loads back without renderer state."""

import itertools
import json
import os
import unittest
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleGeometry, boundary_faces
from newton_solver_suturebot._src.thread_initialization import loose_thread_coordinates
from newton_solver_suturebot._src.worker_thread import WorkerThread

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


def scene_fixture():
    points = np.array(list(itertools.product((-0.04, 0.04), (-0.04, 0.04), (-0.005, 0))))
    tets = np.array(
        [[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]], dtype=np.int32
    )
    for tet in tets:
        if np.linalg.det((points[tet[1:]] - points[tet[0]]).T) < 0:
            tet[1], tet[2] = tet[2], tet[1]
    builder = newton.ModelBuilder()
    for position in points:
        builder.add_particle(wp.vec3(position), wp.vec3(), mass=0.1)
    for i in range(3):
        builder.add_body(
            xform=wp.transform(wp.vec3(0, 0, 0.015 + 0.03 * i), wp.quat_identity()),
            mass=0.0003 if i == 0 else 0.01,
            inertia=wp.mat33(1e-8, 0, 0, 0, 1e-8, 0, 0, 0, 2e-8),
        )
    model = builder.finalize(device=DEVICE)
    reference = SimpleNamespace(
        vertices=points,
        global_vertex_indices=np.arange(len(points)),
        exterior_surface_triangles=boundary_faces(tets)[0],
    )
    selection = SimpleNamespace(vertices=points, indices=tets, global_vertex_indices=np.arange(len(points)))
    graspers = [
        SimpleNamespace(
            capsule_bodies=(i,),
            capsule_endpoints=np.array([[[0, -0.01, 0], [0, 0.01, 0]]]),
            capsule_radii=np.array([0.001]),
        )
        for i in (1, 2)
    ]
    scene = SimpleNamespace(
        model=model,
        state_0=model.state(),
        reference=reference,
        selection=selection,
        geometry=NeedleGeometry.load(),
        needle_body=0,
        graspers=graspers,
        args=SimpleNamespace(young_modulus=300.0, shaft_friction=0.2),
    )
    config = SimpleNamespace(
        thread_elements=90, thread_length_m=0.18, thread_radius_m=0.00015, thread_modulus_pa=2e6, thread_iterations=5
    )
    return scene, config


class WorkerThreadTests(unittest.TestCase):
    def test_accepted_thread_step_returns_real_loads_and_passive_positions(self):
        scene, config = scene_fixture()
        worker = WorkerThread(scene, config)
        initial = worker.positions()
        self.assertEqual(initial.shape, (181, 3))
        for _ in range(3):
            worker.advance(1 / 240)
        self.assertEqual(worker.accepted_steps, 3)
        self.assertAlmostEqual(worker.physical_time_s, 3 / 240)
        self.assertEqual(worker.solver.max_eyelets, 0)
        self.assertEqual(worker.tissue.track_count, 0)
        self.assertEqual(worker.rigid_contact.groups, 3)
        self.assertGreater(np.linalg.norm(worker.attachment.wrench.numpy()), 0)
        np.testing.assert_array_equal(worker.attachment._support.numpy(), 0)
        actual_q = worker.solver.q.numpy().copy()
        positions = worker.positions()
        self.assertGreater(np.linalg.norm(positions - initial), 1e-7)
        np.testing.assert_array_equal(worker.positions(), positions)
        np.testing.assert_array_equal(worker.solver.q.numpy(), actual_q)
        self.assertEqual(worker.accepted_steps, 3)
        scene.state_0.body_f.zero_()
        external = wp.zeros(scene.model.particle_count, dtype=wp.vec3, device=DEVICE)
        worker.add_forces(scene.state_0.body_f, external)
        expected = worker.rigid_contact.wrench.numpy().copy()
        expected[scene.needle_body] += worker.attachment.wrench.numpy()[0]
        np.testing.assert_allclose(scene.state_0.body_f.numpy(), expected, rtol=1e-6)
        np.testing.assert_array_equal(external.numpy(), worker._nodal_force.numpy())
        json.dumps(worker.diagnostics(), allow_nan=False)

    def test_independent_workers_and_load_reads_never_advance_history(self):
        scene, config = scene_fixture()
        first, second = WorkerThread(scene, config), WorkerThread(scene, config)
        initial_second = second.positions()
        first.advance(1 / 240)
        saved = [a.numpy().copy() for a in first.state]
        for _ in range(2):
            scene.state_0.body_f.zero_()
            external = wp.zeros(scene.model.particle_count, dtype=wp.vec3, device=DEVICE)
            first.add_forces(scene.state_0.body_f, external)
        for value, array in zip(saved, first.state, strict=True):
            np.testing.assert_array_equal(value, array.numpy())
        np.testing.assert_array_equal(second.positions(), initial_second)
        self.assertEqual(second.accepted_steps, 0)

    def test_loose_initialization_keeps_tail_and_rest_length_scale(self):
        pad = np.array([[-0.04, -0.04, -0.005], [0.04, 0.04, 0]])
        head = np.array([0.01, -0.01, 0.01, 1.0, 0, 0])
        q = loose_thread_coordinates(head, pad)
        self.assertEqual(q.shape, (91, 6))
        np.testing.assert_array_equal(q[0], head)
        np.testing.assert_allclose(np.linalg.norm(q[:, 3:], axis=1), 1, atol=1e-12)
        self.assertLessEqual(np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1).max(), 0.0020000001)
        self.assertTrue(np.isfinite(q).all())
        with self.assertRaises(ValueError):
            loose_thread_coordinates(head, pad * 0.001)
        with self.assertRaises(ValueError):
            loose_thread_coordinates(head * 2, pad)


if __name__ == "__main__":
    unittest.main()
