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
from newton_solver_suturebot._src.solvers.thread.thread_initialization import loose_thread_coordinates
from newton_solver_suturebot._src.solvers.thread.worker_thread import WorkerThread

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


class _TrackRows:
    """Rows of a needle track in the fixture's tissue, with the members that the worker's follower reads."""

    def __init__(self, tetrahedra, weights, rest, sections, order):
        self.tetrahedra, self.weights, self.rest, self.sections, self.order = tetrahedra, weights, rest, sections, order
        self.count, self.events = 0, ()

    @property
    def track_count(self):
        return self.count

    def track_rows(self, start=0):
        rows = slice(int(start), self.count)
        return self.tetrahedra[rows], self.weights[rows], self.rest[rows], self.sections[rows]


def track_rows(scene, rows):
    """A stand-in interaction with ``rows`` eyelet-worthy track rows 2 mm apart inside the tissue block, in two sections.

    Every row becomes an eyelet (2 mm is more than the spacing): the first three in section 0, the others in
    section 1. ``order`` is the row of each eyelet slot. The rows are handed over by setting ``count``.
    """
    points = np.asarray(scene.selection.vertices, dtype=float)
    tetrahedra = np.asarray(scene.selection.indices)
    chosen = np.arange(rows) % len(tetrahedra)
    weights = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.3, 0.2]])[np.arange(rows) % 3]
    weights = weights.reshape(rows, 4)
    # Rest positions on a line, 2 mm apart: what the eyelet rule measures (the embedding gives the eyelet's place).
    rest = np.column_stack((0.002 * np.arange(rows), np.zeros(rows), np.full(rows, -0.002)))
    sections = (np.arange(rows) >= 3).astype(np.int32)
    del points
    return _TrackRows(chosen.astype(np.int32), weights, rest, sections, list(range(rows)))


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

    def test_a_reserve_with_nothing_threaded_leaves_the_worker_bit_identical_over_fifty_steps(self):
        """Fails if the eyelet reserve, the tail kernel or eyelets that no tail has passed change an accepted step.

        Bit for bit on the CPU. On a CUDA device two runs of one and the same worker differ in their last
        bits (the reactions are summed by atomic additions, in an order that varies), so there the
        numbers are compared to that rounding and the integers exactly.
        """

        def run(thread_tracks, rows=0):
            scene, config = scene_fixture()
            scene.plan = SimpleNamespace(expected_passages=2)
            scene.interaction = track_rows(scene, rows)
            worker = WorkerThread(scene, config, thread_tracks=thread_tracks)
            poses = scene.state_0.body_q.numpy().copy()
            tissue = scene.state_0.particle_q.numpy().copy()
            for step in range(50):
                moved = poses.copy()
                moved[0, 0] += (step + 1) * 2e-6
                moved[1, 1] -= (step + 1) * 1e-6
                scene.state_0.body_q.assign(moved)
                lifted = tissue.copy()
                lifted[:, 2] += (step + 1) * 1e-7
                scene.state_0.particle_q.assign(lifted)
                if step == 10 and rows:
                    # The rows arrive while the run goes on, as a tip lays them.
                    scene.interaction.count = rows
                worker.advance(1 / 240)
            return scene, worker

        def carried(worker):
            arrays = (
                worker.solver.q,
                worker.solver.qd,
                worker.solver.contact_limit,
                worker.solver.contact_anchor,
                worker.solver.contact_force,
                worker.solver.iterations,
                *worker.tissue.state,
                worker.attachment.wrench,
                worker.rigid_contact.wrench,
                worker._nodal_force,
            )
            return [array.numpy() for array in arrays]

        def assert_same(first, second):
            for actual, expected in zip(first, second, strict=True):
                if wp.get_device(DEVICE).is_cuda and actual.dtype.kind == "f":
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-9)
                else:
                    self.assertEqual(actual.tobytes(), expected.tobytes())

        _scene, plain = run(False)
        # Without the scene's word the worker is the one of before: no eyelet slot, one track, no follower.
        _scene, unasked = run(None)
        for worker in (plain, unasked):
            self.assertEqual((worker.solver.max_eyelets, worker.eyelet_reserve, worker.track), (0, (0, 1), None))
            self.assertEqual(worker.solver.eyelet_stiffness, 0.0)
        assert_same(carried(unasked), carried(plain))
        _scene, reserve = run(True)
        # The reserve: 18 eyelets per bite (floor(22.43 mm / 1.5 mm) + 2 per passage, 2 passages), 2 bites, 4 tracks.
        length = float(np.linalg.norm(np.diff(NeedleGeometry.load().centerline, axis=0), axis=1).sum())
        self.assertEqual(reserve.eyelet_reserve, (2 * (int(length // 0.0015) + 4), 4))
        self.assertEqual(reserve.eyelet_reserve, (36, 4))
        self.assertEqual(reserve.solver.max_eyelets, 36)
        self.assertEqual(reserve.solver.eyelet_stiffness, reserve.contact_stiffness)
        assert_same(carried(reserve), carried(plain))
        self.assertEqual((reserve.tissue.point_count, reserve.diagnostics()["eyelets"]), (0, 0))
        # With eyelets made from track rows inside the tissue, which the needle's tail (above it) never passes.
        scene, defined = run(True, rows=5)
        assert_same(carried(defined), carried(plain))
        report = defined.diagnostics()
        self.assertEqual(
            [report[name] for name in ("eyelets", "eyelet_passages", "eyelets_threaded", "eyelets_refused")],
            [5, [3, 2], 0, 0],
        )
        self.assertEqual((report["thread_inside_tissue_m"], report["eyelet_reserve"]), (0.0, [36, 4]))
        np.testing.assert_array_equal(defined.solver.eyelet_element.numpy(), -1)
        json.dumps(report, allow_nan=False)
        # The eyelets are where the rows are, in the tissue as it now is.
        particles = scene.state_0.particle_q.numpy().astype(float)
        tetrahedra = np.asarray(scene.selection.indices)
        for slot, row in enumerate(scene.interaction.order):
            np.testing.assert_allclose(
                defined.solver.eyelet_target.numpy()[slot],
                scene.interaction.weights[row] @ particles[tetrahedra[scene.interaction.tetrahedra[row]]],
                atol=1e-6,
            )

    def test_the_worker_threads_the_eyelet_that_the_needle_tail_has_passed_and_reports_it(self):
        """Fails if the worker follows another body or point than the needle's tail, or does not report the threading."""
        scene, config = scene_fixture()
        scene.plan = SimpleNamespace(expected_passages=2)
        scene.interaction = track_rows(scene, 5)
        scene.interaction.count = 5
        worker = WorkerThread(scene, config, thread_tracks=True)
        worker.track.advance()
        accepted = scene.state_0
        worker.tissue.update(accepted.particle_q, accepted.body_q)
        np.testing.assert_array_equal(worker.solver.eyelet_element.numpy(), -1)
        self.assertEqual(worker.diagnostics()["eyelets_threaded"], 0)
        # Put the needle so that its tail is 0.3 mm past eyelet 1 along the direction in which the thread
        # leaves it (the tangent at the tail, as the attachment has it), and 0.2 mm beside that line.
        particles = accepted.particle_q.numpy().astype(float)
        tetrahedra = np.asarray(scene.selection.indices)
        row = scene.interaction.order[1]
        eyelet = scene.interaction.weights[row] @ particles[tetrahedra[scene.interaction.tetrahedra[row]]]
        tangent, tail = np.asarray(worker.tangent), np.asarray(worker.tail)
        across = np.cross(tangent, [0.0, 0.0, 1.0])
        across /= np.linalg.norm(across)
        poses = accepted.body_q.numpy().copy()
        # Identity rotation: body position = tail position - tail in the body frame.
        poses[scene.needle_body, :3] = eyelet - 0.0003 * tangent + 0.0002 * across - tail
        accepted.body_q.assign(poses)
        worker.tissue.update(accepted.particle_q, accepted.body_q)
        np.testing.assert_array_equal(np.flatnonzero(worker.solver.eyelet_element.numpy() >= 0), [1])
        report = worker.diagnostics()
        self.assertEqual((report["eyelets"], report["eyelets_threaded"]), (5, 1))

        # The reach across the thread's line is two wire radii of the needle, and only the needle is followed:
        # with the tail 0.3 mm past eyelet 2 but 0.02 mm further beside it than that, and a tool body standing
        # where the needle would have to stand to thread eyelet 3, nothing more is threaded.
        def place(slot, beside):
            point = scene.interaction.weights[slot] @ particles[tetrahedra[scene.interaction.tetrahedra[slot]]]
            return point - 0.0003 * tangent + beside * across - tail

        reach = 2.0 * float(scene.geometry.thickness)
        poses[scene.needle_body, :3] = place(scene.interaction.order[2], reach + 2e-5)
        poses[1, :3] = place(scene.interaction.order[3], 0.0)
        accepted.body_q.assign(poses)
        worker.tissue.update(accepted.particle_q, accepted.body_q)
        np.testing.assert_array_equal(np.flatnonzero(worker.solver.eyelet_element.numpy() >= 0), [1])
        # Just inside that reach eyelet 2 is threaded.
        poses[scene.needle_body, :3] = place(scene.interaction.order[2], reach - 2e-5)
        accepted.body_q.assign(poses)
        worker.tissue.update(accepted.particle_q, accepted.body_q)
        np.testing.assert_array_equal(np.flatnonzero(worker.solver.eyelet_element.numpy() >= 0), [1, 2])
        self.assertEqual(worker.diagnostics()["eyelets_threaded"], 2)

    def test_the_worker_lays_its_spare_thread_on_the_pad_behind_the_planned_stitch_and_clear_of_the_tools(self):
        """Fails if the layout does not get the scene's surface, its stitch, the thread's own turn radius or the capsules' world places."""
        scene, config = scene_fixture()
        entry, exit_point = np.array([0.0, 0.0, 0.0]), np.array([0.0, -0.01, 0.0])
        scene.plan = SimpleNamespace(
            entry=SimpleNamespace(reference_position=entry), exit=SimpleNamespace(reference_position=exit_point)
        )
        # A vertex of the pad that no triangle uses, 5 mm over its top: the pad's box is higher than its surface.
        scene.reference.vertices = np.vstack((scene.reference.vertices, [[0.0, 0.0, 0.005]]))
        scene.reference.global_vertex_indices = np.arange(len(scene.reference.vertices))
        radius, element = config.thread_radius_m, config.thread_length_m / config.thread_elements

        def laid(q):
            # The pad's surface is level at z = 0: the laid nodes are a thread radius above it, not above its box.
            return int(np.flatnonzero(np.abs(q[:, 2] - radius) > 1e-9)[-1]) + 1

        free = WorkerThread(scene, config).solver.q.numpy().reshape(-1, 6)
        first = laid(free)
        self.assertLess(first, 40)
        # Two radii of the needle's circle straight behind the entry.
        behind = entry[:2] + 2.0 * scene.geometry.radius * np.array([0.0, 1.0])
        self.assertLess(np.linalg.norm(free[first, :2] - behind), element)
        # The walk keeps the thread's elasto-gravity length from the pad's edge and turns with it.
        turn = (config.thread_modulus_pa * radius**2 / (4.0 * 1140.0 * 9.81)) ** (1.0 / 3.0)
        self.assertAlmostEqual(np.abs(free[first:, :2]).max(), 0.04 - turn, delta=1e-9)
        a, b, c = free[first:-2, :2], free[first + 1 : -1, :2], free[first + 2 :, :2]
        area = 0.5 * np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
        sides = np.linalg.norm(a - b, axis=1) * np.linalg.norm(b - c, axis=1) * np.linalg.norm(c - a, axis=1)
        bent = area > 1e-12
        self.assertGreaterEqual(bent.sum(), 16)
        self.assertAlmostEqual(np.median(sides[bent] / (4.0 * area[bent])), turn, delta=2e-3 * turn)
        # A tool capsule across that lead, given as the scene has it: in the frame of a turned and moved body.
        # The body stands 6 mm beside the lead; its capsule, along y in the body, reaches the lead only when it
        # is turned to lie along x.
        middle = free[first // 2, :3]
        poses = scene.state_0.body_q.numpy()
        poses[1] = (middle[0] + 0.006, middle[1], middle[2], 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))
        scene.state_0.body_q.assign(poses)

        def gap(q):
            nodes = q[3 : laid(q), :3]
            along = np.clip(nodes[:, 0] - middle[0], -0.004, 0.016)
            nearest = middle + along[:, None] * np.array([1.0, 0.0, 0.0])
            return np.linalg.norm(nodes - nearest, axis=1).min() - 0.001 - radius

        self.assertLess(gap(free), 0.0)
        blocked = WorkerThread(scene, config).solver.q.numpy().reshape(-1, 6)
        self.assertGreaterEqual(gap(blocked), radius)
        np.testing.assert_array_equal(blocked[0], free[0])

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
