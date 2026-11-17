# SPDX-License-Identifier: Apache-2.0
"""The frame graphs of the phantom examples read the time step from device data.

Both examples capture one frame before the run: tangent refresh, grips, FEM step and, in the
cutting example, surface contact and the volume guard. Their own time step never changes, but
the graphs hold none by value: replayed after the solver's step time was written with another
time step, they give the frames that eager stepping gives at that time step.
"""

import os
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from test_cut_pipeline import _cube
from test_phantom_regions_example import _example as _regions_example

from newton_solver_suturebot import CutVolumeGuard
from newton_solver_suturebot.examples.incision import example_fem_surgical_incision as cutting

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
# A frame remainder of the needle scenes: no halving of a nominal step.
REMAINDER = 0.0020833333333333355


@contextmanager
def _cut_example(*options):
    """The cut example on a cube that lies on the floor, with floor and piece contact."""
    args = cutting.Example.create_parser().parse_args(
        ["--cutter_mode", "cylinder", "--drop_height", "0.0002", "--fragment_capacity", "64", *options]
    )
    with wp.ScopedDevice(DEVICE), patch.object(cutting, "load_body", side_effect=lambda *_: _cube()):
        viewer = ViewerNull()
        try:
            yield cutting.Example(viewer, args)
        finally:
            viewer.close()


def _frame(captured, eager, dt):
    """One frame of both examples at ``dt``: the captured one replays its graph."""
    for example in (captured, eager):
        example.sim_dt = dt
    # Host preparation of the replay, outside capture: the matrix blocks that no graph holds
    # and the step time. The eager frame does the same in its own tangent refresh.
    captured.solver.refresh_tangent(captured.state_0.particle_q, dt)
    captured.step()
    eager.step()


@unittest.skipUnless(DEVICE.startswith("cuda") and wp.is_cuda_available(), "CUDA graph capture")
class PhantomGraphTimestepTests(unittest.TestCase):
    def assert_grip_followed(self, example, before, dt):
        # The grip's target is far away, so it moves by its speed limit times the time step
        # in every substep: the distance shows which time step the launch read.
        moved = np.linalg.norm(example.holds.driven.numpy()[0] - before)
        self.assertAlmostEqual(moved / (example.args.substeps * example.args.grip_speed * dt), 1.0, delta=1.0e-3)

    def test_regions_frame_graph_replays_at_every_time_step(self):
        with _regions_example(substeps=3) as captured, _regions_example(substeps=3) as eager:
            graph = captured.graph
            self.assertIsNotNone(graph)
            eager.graph = None
            self.assertIs(captured.holds.step_time, captured.solver.step_time)
            for example in (captured, eager):
                hit = example.begin_grab((0.003, 0.004, 0.1), (0, 0, -1))
                self.assertIsNotNone(hit)
                example.move_grab(hit + (0.0006, 0, 0.0021))
            base = captured.sim_dt
            for frame, dt in enumerate((base, base, base / 2, base, REMAINDER / 2, base / 3, base)):
                with self.subTest(frame=frame, dt=dt):
                    before = captured.holds.driven.numpy()[0].copy()
                    _frame(captured, eager, dt)
                    self.assertIs(captured.graph, graph)
                    self.assertEqual(captured.solver.step_time.value, dt)
                    self.assert_grip_followed(captured, before, dt)
                    # The tolerances of the example's own comparison of captured and eager frames.
                    np.testing.assert_allclose(
                        captured.state_0.particle_q.numpy(), eager.state_0.particle_q.numpy(), rtol=0, atol=2.0e-8
                    )
                    np.testing.assert_allclose(
                        captured.state_0.particle_qd.numpy() * dt,
                        eager.state_0.particle_qd.numpy() * dt,
                        rtol=0,
                        atol=4.0e-8,
                    )
            captured.test_final()

    def test_cut_frame_graph_replays_at_every_time_step(self):
        # A solver prepared for streamed topology keeps its construction time step; the
        # sorted assembly accepts another one.
        with ExitStack() as stack:
            captured, eager = (stack.enter_context(_cut_example("--no-streaming_topology")) for _ in range(2))
            graph = captured.graph
            self.assertIsNotNone(graph)
            eager.graph = None
            step_time = captured.solver.step_time
            for owner in (captured.holds, captured.contact, captured.volume_guard):
                self.assertIs(owner.step_time, step_time)
            base = captured.sim_dt
            for frame, dt in enumerate((base, base, base, base / 2, base, REMAINDER, base / 3, base)):
                with self.subTest(frame=frame, dt=dt):
                    if frame == 2:
                        for example in (captured, eager):
                            points = example.contact.update_surface(example.state_0.particle_q).numpy()
                            point = points[:3].mean(axis=0)
                            self.assertTrue(
                                example.holds.begin(
                                    example.surface,
                                    example.cutter.geometry,
                                    0,
                                    (1.0 / 3.0,) * 3,
                                    point,
                                    example.state_0.particle_q,
                                )
                            )
                            example.holds.move(point + np.asarray((0.002, 0, 0.004), dtype=np.float32))
                            example.holds.finish()
                    before = captured.holds.driven.numpy()[0].copy()
                    _frame(captured, eager, dt)
                    self.assertIs(captured.graph, graph)
                    self.assertEqual(step_time.value, dt)
                    if frame >= 2:
                        self.assert_grip_followed(captured, before, dt)
                    # Measured on a GB10: positions within 1.9e-8 m and contact forces within
                    # 3e-5 N of the eager frames, which differ among themselves by 1.5e-8 m and
                    # 1.8e-5 N (float32 sums in no fixed order).
                    np.testing.assert_allclose(
                        captured.state_0.particle_q.numpy(), eager.state_0.particle_q.numpy(), rtol=0, atol=1.0e-7
                    )
                    np.testing.assert_allclose(
                        captured.state_0.particle_qd.numpy() * dt,
                        eager.state_0.particle_qd.numpy() * dt,
                        rtol=0,
                        atol=5.0e-8,
                    )
                    force = captured.contact.forces.numpy()
                    np.testing.assert_allclose(force, eager.contact.forces.numpy(), rtol=0, atol=3.0e-4)
                    # The floor carries the cube in every frame.
                    self.assertGreater(np.abs(force).max(), 1.0)
                    self.assertGreater(int(captured.contact.active_rows.numpy().sum()), 0)
            captured.test_final()

    def test_volume_guard_graph_replays_at_every_time_step(self):
        # The guard writes the velocity of its correction, displacement / dt. One projection
        # sweep leaves the squashed cells below the bound, so its step limit acts as well.
        with _cut_example() as example:
            model, device = example.model, example.model.device
            rest = example.state_0.particle_q.numpy().copy()
            low = rest[:, 2].min()
            trial = rest.copy()
            trial[:, 2] = low - 0.01 * (rest[:, 2] - low)  # turned inside out along z
            trial[:, 0] += 0.003 * (rest[:, 2] - low)
            speeds = np.linspace(0.1, -0.4, trial.size, dtype=np.float32).reshape(-1, 3)
            previous = wp.array(rest, dtype=wp.vec3, device=device)
            base = example.sim_dt
            for iterations in (1, 16):
                captured = CutVolumeGuard(model, iterations=iterations, components=example._mechanical_components)
                eager = CutVolumeGuard(model, iterations=iterations, components=example._mechanical_components)
                q = wp.array(trial, dtype=wp.vec3, device=device)
                qd = wp.array(speeds, dtype=wp.vec3, device=device)
                captured.apply(q, qd, base, previous_q=previous)
                q.assign(trial)
                qd.assign(speeds)
                with wp.ScopedCapture(device=device) as capture:
                    captured.apply(q, qd, base, previous_q=previous)
                    # Another time step would have to be written, which a capture refuses.
                    with self.assertRaisesRegex(RuntimeError, "during a CUDA graph capture"):
                        captured.apply(q, qd, base / 2, previous_q=previous)
                largest = {}
                for dt in (base, base / 2, REMAINDER, base / 3, base):
                    with self.subTest(iterations=iterations, dt=dt):
                        q.assign(trial)
                        qd.assign(speeds)
                        captured.limited_steps.zero_()
                        captured.step_time.set(dt)
                        wp.capture_launch(capture.graph)
                        expected_q = wp.array(trial, dtype=wp.vec3, device=device)
                        expected_qd = wp.array(speeds, dtype=wp.vec3, device=device)
                        eager.limited_steps.zero_()
                        eager.apply(expected_q, expected_qd, dt, previous_q=previous)
                        np.testing.assert_allclose(q.numpy(), expected_q.numpy(), rtol=1.0e-6, atol=1.0e-9)
                        np.testing.assert_allclose(qd.numpy(), expected_qd.numpy(), rtol=1.0e-5, atol=1.0e-6)
                        np.testing.assert_array_equal(captured.limited_steps.numpy(), eager.limited_steps.numpy())
                        # The guard did act: it moved nodes, and with one sweep it cut the step.
                        self.assertGreater(np.abs(q.numpy() - trial).max(), 1.0e-3)
                        self.assertEqual(int(captured.limited_steps.numpy()[0]) > 0, iterations == 1)
                        largest[dt] = np.abs(qd.numpy()).max()
                # The same displacement in half the time: twice the velocity.
                self.assertAlmostEqual(largest[base / 2] / largest[base], 2.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
