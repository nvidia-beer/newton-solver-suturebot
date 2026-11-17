# SPDX-License-Identifier: Apache-2.0
"""The needle-on-a-thread example captures its frame and replays it like the eager frame."""

import io
import os
import unittest
from contextlib import contextmanager, redirect_stdout
from unittest.mock import patch

import numpy as np
import warp as wp
from _startup_support import final_line, launch_late_kernel, marked_on_every_device, prime, run_script
from newton.viewer import ViewerNull

from newton_solver_suturebot.examples.needle import _thread_view
from newton_solver_suturebot.examples.needle import example_needle_thread as needle_thread

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


@contextmanager
def _example(*arguments, viewer=None):
    args = needle_thread.Example.create_parser().parse_args(list(arguments))
    with wp.ScopedDevice(DEVICE):
        viewer = ViewerNull() if viewer is None else viewer
        try:
            yield needle_thread.Example(viewer, args)
        finally:
            viewer.close()


@contextmanager
def _self_test(viewer=None):
    """The example as ``--test`` starts it, with its start-up marked on the CPU device as well."""
    with _example() as other:
        prime(other)
    with marked_on_every_device(), _example("--test", viewer=viewer) as example:
        yield example


class NeedleThreadExampleTests(unittest.TestCase):
    def test_frame_graph_is_captured_and_replays_like_the_eager_frame(self):
        """Fails if the example lost its frame graph, or if the graph's frames differ from eager frames.

        The example falls back to eager stepping when its capture raises, with one printed line. The thread
        kernels of that graph read the time step from the thread solver's step time, and a write of it
        during a capture raises: the eager frame before the capture must have written it.
        """
        if not wp.get_device(DEVICE).is_cuda:
            self.skipTest("The frame graph needs CUDA")
        args = needle_thread.Example.create_parser().parse_args([])
        with wp.ScopedDevice(DEVICE):
            viewers = ViewerNull(), ViewerNull()
            for viewer in viewers:
                self.addCleanup(viewer.close)
            captured = needle_thread.Example(viewers[0], args)
            eager = needle_thread.Example(viewers[1], args)
            self.assertIsNotNone(captured.graph)
            eager.graph = None
            step_time = captured.thread.step_time
            self.assertEqual((step_time.value, step_time.writes), (captured.sim_dt, 1))
            start = captured.thread.q.numpy().copy()
            for _ in range(12):
                captured.step()
                eager.step()
            self.assertEqual((step_time.value, step_time.writes), (captured.sim_dt, 1))
            # Positions [m] of the needle and of the thread's nodes; slopes are dimensionless.
            np.testing.assert_allclose(
                captured.state_0.body_q.numpy(), eager.state_0.body_q.numpy(), rtol=0.0, atol=1.0e-6
            )
            np.testing.assert_allclose(captured.thread.q.numpy(), eager.thread.q.numpy(), rtol=0.0, atol=1.0e-6)
            # The frames did advance the thread: it sags under its weight from the straight start.
            self.assertGreater(np.abs(captured.thread.q.numpy() - start).max(), 1.0e-5)
            captured.test_final()


class NeedleThreadStartupTests(unittest.TestCase):
    """Start-up of the example: graph launched, kernel modules loaded, scene drawn, and the check of ``--test``."""

    def test_self_test_plays_its_script_and_finds_nothing_after_start_up(self):
        viewer = ViewerNull(num_frames=100)
        with _self_test(viewer) as example:
            # The script is longer than the default run of 100 frames: the viewer runs all of it.
            self.assertEqual((example.script.frames, viewer.num_frames, viewer.frame_count), (251, 251, 0))
            run_script(example)
            self.assertEqual(
                [label for _, label in example.script.done],
                ["right press on the needle", "right drag", "right release"]
                + ["right press on the thread", "right drag", "right release", "thread stiffness slider"]
                + ["right press on the needle", "right drag", "right release", "Clear holds"]
                + ["thread stiffness slider", "debug view on", "debug view off"],
            )
            output = io.StringIO()
            with redirect_stdout(output):
                example.test_final()
            message = output.getvalue().splitlines()[0]
            # The lift that the example has always measured, and the needle at rest again at the end.
            self.assertRegex(message, r"hold error \d\.\d+ mm after a 50\.0 mm lift, released needle at rest")
            line = final_line(example)
            self.assertEqual((line["presses"], line["script_frames"], line["script_actions"]), (2, [251, 251], 14))
            self.assertEqual((line["capture_census_status"], line["startups"]), ("checked", 1))
            for name in (*needle_thread.Startup.numbers, "render_module_loads_after_startup"):
                self.assertEqual(line[name], 0, name)
            self.assertEqual(example.young_modulus, example.args.young_modulus)
            self.assertEqual(
                (len(example.holds.attachments), example.show_nodes, example.show_collision), (0, False, False)
            )
            if example.device.is_cuda:
                self.assertEqual((line["startup_captures"], line["startup_solver_captures"]), (1, 0))
            # A scripted press that took no hold fails the self-test: the script must do what it says.
            example._presses[0] = False
            with self.assertRaisesRegex(AssertionError, "took no hold"):
                example.test_final()

    def test_default_run_keeps_its_frames_and_has_no_script(self):
        viewer = ViewerNull(num_frames=100)
        with _example(viewer=viewer) as example:
            self.assertIsNone(example.script)
            self.assertEqual((viewer.num_frames, viewer.frame_count), (100, 0))
            example.step()
            output = io.StringIO()
            with redirect_stdout(output):
                example.test_final()
            # The state is checked; captures and loads are the business of --test.
            self.assertEqual(len(output.getvalue().splitlines()), 1)

    def test_thread_view_loads_its_kernel_module_when_it_is_built(self):
        with patch.object(_thread_view, "load_kernel_modules") as load, _example() as example:
            load.assert_called_once_with(
                example.thread.thread.device,
                _thread_view._sample_centerline,
                _thread_view._tube_segments,
                _thread_view._slope_lines,
            )

    def test_start_state_is_drawn_once_and_with_the_debug_view_where_it_can_be_switched_on(self):
        class Window(ViewerNull):
            # As the OpenGL viewer with a window: it ignores its frame limit and has the panel.
            _headless = False

            def register_ui_callback(self, callback, position="side"):
                pass

        drawn = []
        render = needle_thread.ThreadView.render

        def recording_render(view, *, show_nodes=False):
            drawn.append((show_nodes, view.viewer.show_collision))
            render(view, show_nodes=show_nodes)

        with patch.object(needle_thread.ThreadView, "render", recording_render):
            with _example() as example:
                # No panel and no script: the debug view cannot be switched on, and only the plain frame is drawn.
                self.assertEqual(drawn, [(False, False)])
                self.assertEqual(example.viewer.frame_count, 0)
            for arguments, viewer, frames in ((("--test",), ViewerNull(), 0), ((), Window(), 1)):
                drawn.clear()
                with _example(*arguments, viewer=viewer) as example:
                    # The launches of the debug view and those of a plain frame, before the first frame.
                    self.assertEqual(drawn, [(True, True), (False, False)])
                    # A window draws one whole frame; a frame-limited viewer keeps every frame.
                    self.assertEqual(example.viewer.frame_count, frames)
                    self.assertFalse(example.viewer.show_collision)
                    self.assertFalse(example.thread_view._debug_drawn)

    def test_late_kernel_module_load_fails_the_self_test(self):
        with _self_test() as example:
            render = example.thread_view.render

            def wrong_render(**options):
                # A wrong version: the module of the thread view's kernels loads with the fourth frame.
                if example.script.frame == 4:
                    launch_late_kernel(example.device)
                render(**options)

            example.thread_view.render = wrong_render
            run_script(example, frames=6)
            with self.assertRaisesRegex(AssertionError, r"'module_loads_after_startup': 1.*late_startup_test_kernel"):
                example.test_final()

    def test_capture_in_a_frame_fails_the_self_test(self):
        with _self_test() as example:
            if not example.device.is_cuda:
                self.skipTest("A graph needs CUDA")
            run_script(example, frames=3)
            # A wrong version: the frame graph is captured at its first use, inside a frame.
            example._capture_physics()
            run_script(example, frames=3)
            with self.assertRaisesRegex(AssertionError, "'captures_after_startup': 1"):
                example.test_final()

    def test_frame_graph_is_launched_at_start_up_and_the_start_state_is_put_back(self):
        if not wp.get_device(DEVICE).is_cuda:
            self.skipTest("A graph needs CUDA")
        with _example() as reference:
            # The start state of a scene whose start-up frames were put back: taken before any frame.
            start = {name: getattr(reference.mujoco.mjw_data, name).numpy() for name in ("time", "qacc_warmstart")}
            body_q, thread_q = reference.state_0.body_q.numpy(), reference.thread.q.numpy()
        with _self_test() as example:
            self.assertIsNotNone(example.graph.graph_exec)
            census = example.startup.census()
            self.assertEqual((census["startup_captures"], census["startup_unlaunched_graphs"]), (1, 0))
            np.testing.assert_array_equal(example.state_0.body_q.numpy(), body_q)
            np.testing.assert_array_equal(example.thread.q.numpy(), thread_q)
            np.testing.assert_array_equal(example.thread.qd.numpy(), 0.0)
            # MuJoCo's time and warm start are those of a solver that has not stepped.
            np.testing.assert_array_equal(example.mujoco.mjw_data.time.numpy(), 0.0)
            np.testing.assert_array_equal(example.mujoco.mjw_data.qacc_warmstart.numpy(), 0.0)
            for name, value in start.items():
                np.testing.assert_array_equal(getattr(example.mujoco.mjw_data, name).numpy(), value)
        with patch.object(needle_thread, "launch_startup_graph", lambda graph, device: None), _self_test() as example:
            # A wrong version: the graph is captured at start-up and first launched by a frame.
            self.assertIsNone(example.graph.graph_exec)
            run_script(example, frames=3)
            with self.assertRaisesRegex(AssertionError, "'startup_unlaunched_graphs': 1"):
                example.test_final()

    def test_start_up_frames_leave_later_states_bit_identical(self):
        """A frame that runs at start-up and is put back does not change the run.

        On CUDA the start-up runs the frame twice, eager before the capture and as the launched
        graph. Here a scene that ran two such frames is compared with a scene that ran none,
        over the scripted grab and lift.
        """
        if wp.get_device(DEVICE).is_cuda:
            self.skipTest("Two CUDA runs of one scene differ in rounding")
        with _example("--test") as plain, _example("--test") as started:
            for _ in range(2):
                restore = started._frame_snapshot()
                started.simulate()
                restore()
            for name in ("time", "qpos", "qvel", "qacc_warmstart", "xfrc_applied"):
                np.testing.assert_array_equal(
                    getattr(started.mujoco.mjw_data, name).numpy(), getattr(plain.mujoco.mjw_data, name).numpy(), name
                )
            start = plain.state_0.body_q.numpy().copy()
            for _ in range(40):
                plain.step()
                started.step()
                for first, second in (
                    (plain.state_0.body_q, started.state_0.body_q),
                    (plain.state_0.body_qd, started.state_0.body_qd),
                    (plain.thread.q, started.thread.q),
                    (plain.thread.qd, started.thread.qd),
                ):
                    np.testing.assert_array_equal(first.numpy(), second.numpy())
            # The lift of the script moved the needle.
            self.assertGreater(np.abs(plain.state_0.body_q.numpy() - start).max(), 0.01)


if __name__ == "__main__":
    unittest.main()
