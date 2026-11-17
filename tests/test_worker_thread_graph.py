# SPDX-License-Identifier: Apache-2.0
"""Accepted thread graphs preserve friction history and lagged physical loads."""

import os
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp
from test_worker_thread import scene_fixture, track_rows

from newton_solver_suturebot._src.solvers.thread.worker_thread import WorkerThread


class WorkerThreadGraphTests(unittest.TestCase):
    def test_graphs_match_eager_updates_across_moving_buffers_and_timesteps(self):
        if not wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda:
            self.skipTest("Accepted thread graph parity requires CUDA")
        scene, config = scene_fixture()
        reference, captured = WorkerThread(scene, config, capture=False), WorkerThread(scene, config)
        states = (scene.state_0, scene.model.state())
        initial = states[0].body_q.numpy().copy()
        initial_tissue = states[0].particle_q.numpy().copy()
        cases = [(buffer, dt) for dt in (1 / 240, 1 / 480) for buffer in (0, 1)] * 4
        for step, (buffer, dt) in enumerate(cases):
            with self.subTest(step=step):
                scene.state_0 = states[buffer]
                poses = initial.copy()
                poses[0, 0] += (step + 1) * 2e-6
                poses[1, 1] -= (step + 1) * 1e-6
                scene.state_0.body_q.assign(poses)
                tissue = initial_tissue.copy()
                tissue[:, 2] += (step + 1) * 1e-7
                scene.state_0.particle_q.assign(tissue)
                reference.advance(dt)
                captured.advance(dt)
                for expected, actual in zip(reference.state, captured.state, strict=True):
                    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-10)
                np.testing.assert_array_equal(captured.solver.iterations.numpy(), reference.solver.iterations.numpy())
                self.assertEqual(captured.accepted_steps, step + 1)
                self.assertEqual(captured.physical_time_s, reference.physical_time_s)
                self.assertIs(captured.rigid_contact._body_q, scene.state_0.body_q)
                self.assertEqual(captured.rigid_contact._dt, dt)
                # The first accepted step warms eagerly. Each buffer is then captured once,
                # the second one at the halved step, and replays at either step afterwards.
                self.assertEqual(captured.graph_captures, min(step, 2))
        # One graph per accepted-state buffer: the interval is device data, not a key.
        self.assertEqual(captured.graph_captures, 2)
        self.assertEqual(captured.graph_launches, len(cases) - 1)
        self.assertEqual(reference.graph_launches, 0)
        self.assertEqual(captured.diagnostics()["graph_launches"], len(cases) - 1)
        # Invalidation warms through exactly one real accepted step, preserving
        # thread history instead of integrating and rolling back a dummy step.
        captured.invalidate_step_graphs()
        self.assertFalse(captured._step_graphs)
        before = captured.accepted_steps
        reference.advance(1 / 240)
        captured.advance(1 / 240)
        self.assertEqual(captured.accepted_steps, before + 1)
        self.assertEqual(captured.graph_launches, len(cases) - 1)
        for expected, actual in zip(reference.state, captured.state, strict=True):
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-10)

    def test_one_graph_per_buffer_replays_every_interval_without_recapture(self):
        # The scene loop halves the step after a rejected substep, doubles it again and ends a
        # frame with a remainder that is no halving. The kernels of the thread step read the
        # interval from the solver's step time, so the two graphs of the two accepted-state
        # buffers serve every interval; a graph that kept the interval of its capture would
        # differ from the eager worker in the first replay at another one.
        if not wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda:
            self.skipTest("Accepted thread graph parity requires CUDA")
        scene, config = scene_fixture()
        reference, captured = WorkerThread(scene, config, capture=False), WorkerThread(scene, config)
        states = (scene.state_0, scene.model.state())
        initial = states[0].body_q.numpy().copy()
        initial_tissue = states[0].particle_q.numpy().copy()
        base = 1 / 240
        intervals = (base, base, base, base / 2, 0.0020833333333333355, base / 4, base, base / 3, base / 2, base)
        damping, step_time = captured.solver.mass_damping, captured.solver.step_time
        pointers = (step_time.seconds.ptr, step_time.seconds_f64.ptr, captured.solver._inertia_scale().ptr)
        graphs, relaxations = None, {}
        for step, dt in enumerate(intervals):
            with self.subTest(step=step, dt=dt):
                scene.state_0 = states[step % 2]
                poses = initial.copy()
                poses[0, 0] += (step + 1) * 2e-6
                poses[1, 1] -= (step + 1) * 1e-6
                scene.state_0.body_q.assign(poses)
                tissue = initial_tissue.copy()
                tissue[:, 2] += (step + 1) * 1e-7
                scene.state_0.particle_q.assign(tissue)
                reference.advance(dt)
                captured.advance(dt)
                for expected, actual in zip(reference.state, captured.state, strict=True):
                    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-10)
                np.testing.assert_array_equal(captured.solver.iterations.numpy(), reference.solver.iterations.numpy())
                # What depends on the interval alone: the bodies' share of a fresh contact wrench.
                relaxation = captured.rigid_contact.relaxation.numpy().copy()
                np.testing.assert_allclose(relaxation, reference.rigid_contact.relaxation.numpy(), rtol=1e-12)
                relaxations[dt] = relaxation
                # The device holds the values the host formed in double precision.
                self.assertEqual(step_time.value, dt)
                np.testing.assert_array_equal(step_time.seconds_f64.numpy(), [dt])
                np.testing.assert_array_equal(step_time.seconds.numpy(), [np.float32(dt)])
                np.testing.assert_array_equal(
                    captured.solver._inertia_scale().numpy(), [1.0 / (dt * dt) + damping / dt]
                )
                self.assertEqual(
                    pointers, (step_time.seconds.ptr, step_time.seconds_f64.ptr, captured.solver._inertia_scale().ptr)
                )
                # Eager warm-up, one capture per buffer, then replays only.
                self.assertEqual(captured.graph_captures, min(step, 2))
                self.assertEqual(captured.graph_launches, step)
                current = [entry[0] for entry in captured._step_graphs.values()]
                if step == 2:
                    graphs = set(current)
                    self.assertEqual(len(graphs), 2)
                elif step > 2:
                    self.assertEqual(set(current), graphs)
        self.assertEqual(reference.graph_launches, 0)
        # The intervals did differ in their effect, so a stale one would have been seen above.
        self.assertFalse(np.allclose(relaxations[base], relaxations[base / 2], rtol=1e-3, atol=0))
        self.assertFalse(np.allclose(relaxations[base / 2], relaxations[base / 4], rtol=1e-3, atol=0))

    def test_capture_graphs_leaves_two_launched_graphs_and_the_thread_where_it_was(self):
        # The start-up entry runs real accepted steps (one as written, one per buffer into a
        # capture, one launch of each graph) and gives the thread back bit for bit.
        if not wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda:
            self.skipTest("Accepted thread graphs require CUDA")
        scene, config = scene_fixture()
        reference, captured = WorkerThread(scene, config, capture=False), WorkerThread(scene, config)
        states = (scene.state_0, scene.model.state())
        states[1].assign(states[0])
        dt = 1 / 240

        def bits(worker):
            return [array.numpy().tobytes() for array in (*worker.state, worker.solver.iterations)]

        before = bits(captured)
        contact, step_time = captured.rigid_contact, captured.solver.step_time
        notes = (contact._body_q, contact._dt)
        # A worker without capture declines and issues nothing.
        self.assertFalse(reference.capture_graphs(states, dt))
        self.assertIsNone(reference.solver.step_time.value)
        self.assertTrue(captured.capture_graphs(states, dt))
        self.assertEqual(bits(captured), before)
        self.assertEqual((captured.graph_captures, captured.graph_launches), (2, 0))
        self.assertEqual((captured.accepted_steps, captured.physical_time_s), (0, 0.0))
        self.assertEqual((contact._body_q, contact._dt), notes)
        graphs = [entry[0] for entry in captured._step_graphs.values()]
        self.assertEqual(len(graphs), 2)
        # Warp builds the executable graph at the first launch: both were launched.
        self.assertEqual([graph.graph_exec is None for graph in graphs], [False, False])
        # The step time had no interval before; it holds the one the graphs were captured with.
        self.assertEqual(step_time.value, dt)
        # Buffers that have their graph are left alone, and an interval written before is written back.
        self.assertTrue(captured.capture_graphs(states, dt / 2))
        self.assertEqual((captured.graph_captures, captured.graph_launches), (2, 0))
        self.assertEqual(step_time.value, dt)
        np.testing.assert_array_equal(step_time.seconds_f64.numpy(), [dt])
        self.assertEqual(bits(captured), before)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            captured.capture_graphs(states, 0.0)
        # Accepted steps on either buffer only launch, and equal the worker that steps as written.
        initial = states[0].body_q.numpy().copy()
        initial_tissue = states[0].particle_q.numpy().copy()
        for step, (buffer, interval) in enumerate(((0, dt), (1, dt), (0, dt / 2), (1, dt), (0, dt))):
            with self.subTest(step=step):
                scene.state_0 = states[buffer]
                poses = initial.copy()
                poses[0, 0] += (step + 1) * 2e-6
                poses[1, 1] -= (step + 1) * 1e-6
                scene.state_0.body_q.assign(poses)
                tissue = initial_tissue.copy()
                tissue[:, 2] += (step + 1) * 1e-7
                scene.state_0.particle_q.assign(tissue)
                reference.advance(interval)
                captured.advance(interval)
                for expected, actual in zip(reference.state, captured.state, strict=True):
                    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-10)
                np.testing.assert_array_equal(captured.solver.iterations.numpy(), reference.solver.iterations.numpy())
                self.assertEqual((captured.graph_captures, captured.graph_launches), (2, step + 1))
                self.assertEqual(captured.accepted_steps, step + 1)
                self.assertEqual({entry[0] for entry in captured._step_graphs.values()}, set(graphs))
        # An invalidated worker warms and captures again through the entry.
        captured.invalidate_step_graphs()
        self.assertFalse(captured._graph_warmed)
        before = bits(captured)
        self.assertTrue(captured.capture_graphs(states, dt))
        self.assertEqual(bits(captured), before)
        self.assertEqual((captured.graph_captures, len(captured._step_graphs)), (4, 2))

    def test_eyelets_that_appear_after_start_up_are_read_by_the_same_two_graphs_without_a_capture(self):
        # The thread follows the needle's track: eyelets appear as rows written in place while the run
        # goes on, and the tail kernel in the step graph threads one when the needle's tail has passed
        # it. Neither may capture a graph, change a graph key or replace an array.
        if not wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda:
            self.skipTest("Accepted thread graphs require CUDA")
        scene, config = scene_fixture()
        scene.plan = SimpleNamespace(expected_passages=2)
        scene.interaction = track_rows(scene, 5)
        reference = WorkerThread(scene, config, capture=False, thread_tracks=True)
        captured = WorkerThread(scene, config, thread_tracks=True)
        states = (scene.state_0, scene.model.state())
        states[1].assign(states[0])
        dt = 1 / 240
        self.assertTrue(captured.capture_graphs(states, dt))
        self.assertEqual((captured.graph_captures, captured.graph_launches), (2, 0))
        keys = list(captured._step_graphs)
        graphs = {entry[0] for entry in captured._step_graphs.values()}
        owned = (
            captured.solver.eyelet_target,
            captured.solver.eyelet_element,
            captured.solver.eyelet_xi,
            captured.solver.eyelet_force,
            captured.tissue._tetrahedron_device,
            captured.tissue._barycentric_device,
            captured.tissue._group_device,
            captured.tissue._group_size_device,
            *captured.tissue.track_spans,
        )
        pointers = [(array.ptr, array.shape) for array in owned]
        initial = states[0].body_q.numpy().copy()
        initial_tissue = states[0].particle_q.numpy().copy()
        tetrahedra = np.asarray(scene.selection.indices)
        steps = 12
        for step in range(steps):
            with self.subTest(step=step):
                scene.state_0 = states[step % 2]
                poses = initial.copy()
                poses[0, 0] += (step + 1) * 2e-6
                poses[1, 1] -= (step + 1) * 1e-6
                tissue = initial_tissue.copy()
                tissue[:, 2] += (step + 1) * 1e-7
                if step == 3:
                    scene.interaction.count = 3  # the first passage: three eyelets appear
                if step == 7:
                    scene.interaction.count = 5  # the second: two more
                if step == steps - 1:
                    # The needle's tail 0.3 mm past eyelet 1, on the line along which the thread leaves it.
                    row = scene.interaction.order[1]
                    eyelet = scene.interaction.weights[row] @ tissue[tetrahedra[scene.interaction.tetrahedra[row]]]
                    poses[0, :3] = eyelet - 0.0003 * np.asarray(captured.tangent) - np.asarray(captured.tail)
                scene.state_0.body_q.assign(poses)
                scene.state_0.particle_q.assign(tissue)
                reference.advance(dt)
                captured.advance(dt)
                # Only launches of the two graphs of start-up, under the keys they were captured with.
                self.assertEqual((captured.graph_captures, captured.graph_launches), (2, step + 1))
                self.assertEqual(list(sorted(captured._step_graphs, key=keys.index)), keys)
                self.assertEqual({entry[0] for entry in captured._step_graphs.values()}, graphs)
                made = (0, 0, 0, 3, 3, 3, 3, 5, 5, 5, 5, 5)[step]
                self.assertEqual((captured.tissue.point_count, reference.tissue.point_count), (made, made))
                np.testing.assert_array_equal(
                    captured.tissue._tetrahedron_device.numpy(), reference.tissue._tetrahedron_device.numpy()
                )
                np.testing.assert_array_equal(
                    captured.solver.eyelet_target.numpy(), reference.solver.eyelet_target.numpy()
                )
                threaded = np.flatnonzero(captured.solver.eyelet_element.numpy() >= 0).tolist()
                self.assertEqual(threaded, np.flatnonzero(reference.solver.eyelet_element.numpy() >= 0).tolist())
                if step < steps - 1:
                    self.assertEqual(threaded, [])
                    for expected, actual in zip(reference.state, captured.state, strict=True):
                        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-10)
                else:
                    # The graph's own kernel threaded the eyelet that its rows, written after the capture, define.
                    self.assertEqual(threaded, [1])
        self.assertEqual([(array.ptr, array.shape) for array in owned], pointers)
        self.assertEqual(captured.tissue.refused_points, 0)
        report = captured.diagnostics()
        self.assertEqual((report["eyelets"], report["eyelets_threaded"], report["graph_captures"]), (5, 1, 2))


if __name__ == "__main__":
    unittest.main()
