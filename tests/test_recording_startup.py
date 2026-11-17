# SPDX-License-Identifier: Apache-2.0
"""Start-up of the recording examples: the camera overlay with the recording, and the check of ``--test``."""

import io
import os
import unittest
from contextlib import contextmanager, redirect_stdout
from unittest.mock import patch

import numpy as np
import warp as wp
from _startup_support import final_line, launch_late_kernel, marked_on_every_device, prime, run_script
from newton.viewer import ViewerNull

from newton_solver_suturebot.examples._startup import Startup
from newton_solver_suturebot.examples.recording import _recording_example
from newton_solver_suturebot.examples.recording.example_fem_dvrk import Example as Dvrk
from newton_solver_suturebot.examples.recording.example_fem_versius import Example as Versius

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


class PanelViewer(ViewerNull):
    """A viewer with a side panel, as the OpenGL viewer: the box of the camera overlays can be ticked."""

    def __init__(self):
        super().__init__()
        self.callbacks = []

    def register_ui_callback(self, callback, position="side"):
        self.callbacks.append((callback, position))


@contextmanager
def _example(cls, *arguments, viewer_cls=ViewerNull):
    with wp.ScopedDevice(DEVICE):
        viewer = viewer_cls()
        example = cls(viewer, cls.create_parser().parse_args(list(arguments)))
        try:
            yield example
        finally:
            example.close()
            viewer.close()


@contextmanager
def _self_test(cls, *arguments, viewer_cls=ViewerNull):
    """The example as ``--test`` starts it, with its start-up marked on the CPU device as well."""
    with _example(cls) as other:
        prime(other)
    with marked_on_every_device(), _example(cls, "--test", *arguments, viewer_cls=viewer_cls) as example:
        yield example


def _renders(example):
    """Renders of the camera overlay so far."""
    return example.overlay._generation


class RecordingStartupTests(unittest.TestCase):
    def test_overlay_is_built_and_rendered_with_the_recording(self):
        with _example(Dvrk) as example:
            self.assertTrue(example.overlays_possible)
            self.assertEqual(example.viewer.frame_count, 0)
            # Built and rendered once before the first frame, for the frame the replay starts at.
            self.assertEqual((_renders(example), example.overlay.frame_index), (1, example.frame_index))
            self.assertEqual(set(example.overlay.status), set(example.recording.camera_order))
            self.assertTrue(all(count > 0 for count in example.overlay.coverage.values()))
            first = example.overlay
            example.render()
            self.assertIs(example.overlay, first)
            self.assertEqual(_renders(example), 2)
            # A recording switch loads the next recording with its own overlay, rendered once.
            example._next_recording()
            example.step()
            self.assertIsNot(example.overlay, first)
            self.assertEqual(
                (_renders(example), example.startup.startups), (1, 0 if example.startup.mark is None else 2)
            )
            # Hidden overlays are not rendered by a frame, and a recording loaded meanwhile still gets its overlay.
            example._show_overlays(False)
            example._next_recording()
            example.step()
            example.render()
            self.assertEqual(_renders(example), 1)
            example._show_overlays(True)
            example.render()
            self.assertEqual(_renders(example), 2)

    def test_scene_is_drawn_once_with_the_recording_and_uses_no_frame_of_a_limited_viewer(self):
        class Window(ViewerNull):
            # As the OpenGL viewer with a window: it ignores its frame limit.
            _headless = False

        with _example(Dvrk, "--no-camera-overlays") as example:
            self.assertEqual(example.viewer.frame_count, 0)
        with _example(Dvrk, "--no-camera-overlays", viewer_cls=Window) as example:
            # One whole frame at construction; a recording switch begins none.
            self.assertEqual(example.viewer.frame_count, 1)
            example._next_recording()
            example.step()
            self.assertEqual(example.viewer.frame_count, 1)
        calls = []

        def drawing(viewer, draw, time, *, frame):
            logged = []
            with patch.object(viewer, "log_state", logged.append):
                draw()
            calls.append((frame, logged))

        with patch.object(_recording_example, "first_draw", drawing), _example(Dvrk, "--no-camera-overlays") as example:
            first = example.state_0
            example._next_recording()
            example.step()
            # The state of the loaded recording is logged once in the constructor and once in every switch.
            self.assertEqual(calls, [(True, [first]), (False, [example.state_0])])
            self.assertIsNot(example.state_0, first)

    def test_run_without_overlays_and_without_a_panel_builds_none(self):
        with _self_test(Dvrk, "--no-camera-overlays") as example:
            self.assertFalse(example.overlays_possible)
            self.assertIsNone(example.overlay)
            run_script(example)
            self.assertIsNone(example.overlay)
            labels = [label for _, label in example.script.done]
            self.assertNotIn("camera overlays on", labels)
            self.assertEqual(labels.count("Recording"), len(example.samples))
            line = final_line(example)
            self.assertEqual((line["capture_census_status"], line["startups"]), ("checked", 1 + len(example.samples)))
            for name in Startup.numbers:
                self.assertEqual(line[name], 0, name)

    def test_run_without_overlays_on_a_viewer_with_the_panel_loads_nothing_when_the_box_is_ticked(self):
        with _self_test(Dvrk, "--no-camera-overlays", viewer_cls=PanelViewer) as example:
            self.assertTrue(example.overlays_possible)
            self.assertFalse(example.show_overlays)
            # The overlay is there, rendered once at the load, and frames leave it alone while it is hidden.
            self.assertEqual(_renders(example), 1)
            run_script(example, frames=3)
            self.assertEqual(_renders(example), 1)
            overlay = example.overlay
            example._show_overlays(True)
            run_script(example, frames=3)
            self.assertIs(example.overlay, overlay)
            self.assertEqual(_renders(example), 4)
            line = final_line(example)
            self.assertEqual(line["capture_census_status"], "checked")
            for name in (*Startup.numbers, "render_module_loads_after_startup"):
                self.assertEqual(line[name], 0, name)

    def test_self_test_plays_its_script_and_finds_nothing_after_start_up(self):
        with _self_test(Dvrk) as example:
            start = example.recording.path
            self.assertEqual(example.script.frames, 100)
            run_script(example)
            labels = [label for _, label in example.script.done]
            switches = len(example.samples)
            self.assertGreater(switches, 1)
            self.assertEqual(labels.count("Recording"), switches)
            self.assertEqual((labels.count("camera overlays off"), labels.count("camera overlays on")), (2, 2))
            for label in ("Frame slider", "Restart recording", "Frame phantom", "Frame scene"):
                self.assertIn(label, labels)
            # Every bundled recording was shown once; the replay is back at the first one, at its last frame.
            self.assertEqual(example.recording.path, start)
            self.assertEqual(example.frame_index, example.recording.last_frame)
            self.assertFalse(example.playing)
            line = final_line(example)
            self.assertEqual((line["script_frames"], line["script_actions"]), ([100, 100], len(labels)))
            self.assertEqual((line["capture_census_status"], line["startups"]), ("checked", 1 + switches))
            for name in (*Startup.numbers, "render_module_loads_after_startup"):
                self.assertEqual(line[name], 0, name)
            self.assertEqual(line["startup_captures"], 0)

    def test_versius_self_test_has_one_recording_and_one_start_up(self):
        with _self_test(Versius) as example:
            self.assertEqual(len(example.samples), 1)
            run_script(example)
            labels = [label for _, label in example.script.done]
            self.assertNotIn("Recording", labels)
            self.assertEqual((labels.count("camera overlays off"), labels.count("camera overlays on")), (1, 1))
            line = final_line(example)
            self.assertEqual(
                (line["script_frames"], line["capture_census_status"], line["startups"]), ([100, 100], "checked", 1)
            )
            for name in (*Startup.numbers, "render_module_loads_after_startup"):
                self.assertEqual(line[name], 0, name)

    def test_default_run_has_no_script_and_test_final_checks_the_poses_alone(self):
        with _example(Versius, "--no-camera-overlays") as example:
            self.assertIsNone(example.script)
            example.step()
            output = io.StringIO()
            with redirect_stdout(output):
                example.test_final()
            self.assertEqual(output.getvalue(), "")
            actual = example.state_0.body_q.numpy()[example.body_indices]
            np.testing.assert_allclose(actual, example.recording.body_poses[example.frame_index], atol=2.0e-7, rtol=0)

    def test_late_kernel_module_load_fails_the_self_test(self):
        with _self_test(Versius) as example:
            render = example._render_overlay

            def wrong_render_overlay():
                # A wrong version: a kernel module of the overlay loads with the fourth frame.
                if example.script.frame == 4:
                    launch_late_kernel(example.device)
                render()

            example._render_overlay = wrong_render_overlay
            run_script(example, frames=6)
            with self.assertRaisesRegex(AssertionError, r"'module_loads_after_startup': 1.*late_startup_test_kernel"):
                example.test_final()


if __name__ == "__main__":
    unittest.main()
