"""Video capture rejects resizing instead of silently corrupting rawvideo, and takes the camera of its recipe."""

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import runpy
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location("record_examples", TOOLS / "record_examples.py")
RECORDER = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(TOOLS))
try:
    SPEC.loader.exec_module(RECORDER)
finally:
    sys.path.pop(0)

CLOSE = {"position": [0.03, -0.04, 0.05], "target": [0.01, 0.0, 0.0], "fov": 30.0}


class TestVideoDimensions(unittest.TestCase):
    def test_resize_is_rejected_before_writing_wrong_frame_bytes(self):
        process = Mock()
        process.wait.return_value = 0
        with patch.object(RECORDER.subprocess, "Popen", return_value=process):
            clip = RECORDER.Clip(Path("unused.mp4"), 60)
            clip.write(np.zeros((12, 16, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "Capture dimensions changed"):
                clip.write(np.zeros((14, 16, 3), dtype=np.uint8))
            clip.close()
        self.assertEqual(clip.count, 1)
        self.assertEqual(process.stdin.write.call_count, 1)
        self.assertEqual(len(process.stdin.write.call_args.args[0]), 12 * 16 * 3)


class TestCameraEntry(unittest.TestCase):
    def test_one_camera_and_timed_cameras_are_checked(self):
        self.assertEqual(
            RECORDER.camera_keys(CLOSE),
            [{"frame": 0, "position": (0.03, -0.04, 0.05), "target": (0.01, 0.0, 0.0), "fov": 30.0}],
        )
        clipped = RECORDER.camera_keys(CLOSE | {"near": 0.001, "far": 5})
        self.assertEqual((clipped[0]["near"], clipped[0]["far"]), (0.001, 5.0))
        timed = RECORDER.camera_keys([CLOSE | {"frame": 10}, CLOSE | {"frame": 40, "fov": 20}])
        self.assertEqual([key["frame"] for key in timed], [10, 40])
        for entry, message in (
            ([], "one camera"),
            ("close", "one camera"),
            (CLOSE | {"up": [0, 0, 1]}, "unknown entries.*no roll"),
            (CLOSE | {"frame": 0}, "unknown entries"),
            ({"position": [0, 0, 1], "target": [0, 0, 0]}, "lacks fov"),
            ([CLOSE], "lacks frame"),
            (CLOSE | {"position": [0, 0]}, "position must be three numbers"),
            (CLOSE | {"target": [0, 0, math.nan]}, "target must be three numbers"),
            (CLOSE | {"target": [0, 0, "0"]}, "target must be three numbers"),
            (CLOSE | {"target": CLOSE["position"]}, "must differ"),
            (CLOSE | {"fov": 0}, "fov must be a positive number"),
            (CLOSE | {"fov": 180}, "below 180"),
            (CLOSE | {"near": 0}, "near must be a positive number"),
            (CLOSE | {"near": 2, "far": 1}, "near must be below far"),
            ([CLOSE | {"frame": 5}, CLOSE | {"frame": 5}], "rising order"),
            ([CLOSE | {"frame": -1}], "rising order"),
            ([CLOSE | {"frame": 1.5}], "rising order"),
            ([CLOSE | {"frame": 0}, CLOSE | {"frame": 9, "near": 0.001}], "same entries"),
        ):
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, message):
                RECORDER.camera_keys(entry)

    def test_a_timed_camera_moves_linearly_and_stands_outside_its_keys(self):
        keys = RECORDER.camera_keys(
            [
                {"frame": 10, "position": [0.0, 0.0, 1.0], "target": [0.0, 0.0, 0.0], "fov": 40},
                {"frame": 30, "position": [2.0, 0.0, 1.0], "target": [1.0, 0.0, 0.0], "fov": 20},
                {"frame": 40, "position": [2.0, 4.0, 1.0], "target": [1.0, 0.0, 0.0], "fov": 20},
            ]
        )
        self.assertEqual(RECORDER.camera_at(keys, 0)["position"], (0.0, 0.0, 1.0))
        self.assertEqual(RECORDER.camera_at(keys, 10)["fov"], 40.0)
        quarter = RECORDER.camera_at(keys, 15)
        self.assertEqual(quarter["position"], (0.5, 0.0, 1.0))
        self.assertEqual(quarter["target"], (0.25, 0.0, 0.0))
        self.assertEqual(quarter["fov"], 35.0)
        self.assertEqual(RECORDER.camera_at(keys, 30)["position"], (2.0, 0.0, 1.0))
        self.assertEqual(RECORDER.camera_at(keys, 35)["position"], (2.0, 2.0, 1.0))
        self.assertEqual(RECORDER.camera_at(keys, 400)["position"], (2.0, 4.0, 1.0))
        # One camera is the same at every frame.
        self.assertEqual(RECORDER.camera_at(RECORDER.camera_keys(CLOSE), 77)["target"], (0.01, 0.0, 0.0))

    def test_stills_are_frame_indices_in_rising_order(self):
        self.assertEqual(RECORDER.still_frames("300,0,900,300"), [0, 300, 900])
        for text in ("", "1,,2", "one", "-1,4", "1.5"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                RECORDER.still_frames(text)


class TestEdgesEntry(unittest.TestCase):
    def test_the_intervals_are_checked_against_the_clip(self):
        self.assertEqual(RECORDER.edge_intervals([[40, 79]], 120), [(40, 79)])
        # One frame, the first and the last frame of the clip, and intervals that touch without overlap.
        self.assertEqual(RECORDER.edge_intervals([[0, 0], [1, 5], [9, 9]], 10), [(0, 0), (1, 5), (9, 9)])
        # No interval: the switch is off in every frame of the clip.
        self.assertEqual(RECORDER.edge_intervals([], 10), [])
        for entry, message in (
            ({"first": 2, "last": 3}, "must be a list of"),
            ("2-3", "must be a list of"),
            ([2, 3], "edges interval 2 must be"),
            ([[2]], r"edges interval \[2\] must be"),
            ([[2, 3, 4]], "two whole numbers"),
            ([[2.0, 3]], "two whole numbers"),
            ([["2", 3]], "two whole numbers"),
            ([[True, 3]], "two whole numbers"),
            ([[-1, 3]], r"\[-1, 3\] must hold 0 <= first_frame <= last_frame"),
            ([[5, 4]], r"\[5, 4\] must hold 0 <= first_frame <= last_frame"),
            ([[5, 10]], r"\[5, 10\] is outside the clip of 10 frames \(0 to 9\)"),
            ([[4, 6], [6, 8]], r"\[6, 8\] begins at or before frame 6.*rising order without overlap"),
            ([[4, 6], [5, 5]], "rising order without overlap"),
            ([[6, 8], [1, 2]], "rising order without overlap"),
        ):
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, message):
                RECORDER.edge_intervals(entry, 10)

    def test_a_frame_has_the_edges_inside_an_interval_both_ends_included(self):
        intervals = RECORDER.edge_intervals([[2, 3], [6, 6]], 9)
        self.assertEqual(
            [RECORDER.edges_at(intervals, frame) for frame in range(9)],
            [False, False, True, True, False, False, True, False, False],
        )
        self.assertEqual([RECORDER.edges_at([], frame) for frame in range(3)], [False, False, False])

    def test_the_switch_is_the_one_of_the_viewers_renderer(self):
        viewer = Mock(spec=["renderer"])
        viewer.renderer = Mock(spec=["draw_edges"])
        viewer.renderer.draw_edges = False
        RECORDER.show_edges(viewer, True)
        self.assertIs(viewer.renderer.draw_edges, True)
        RECORDER.show_edges(viewer, False)
        self.assertIs(viewer.renderer.draw_edges, False)
        # A viewer that cannot draw the edges is named, not given an attribute that nothing reads.
        for other in (
            Mock(spec=[]),
            Mock(spec=["renderer"], renderer=None),
            Mock(spec=["renderer"], renderer=object()),
        ):
            with self.subTest(other=other), self.assertRaisesRegex(ValueError, "viewer has no renderer.draw_edges"):
                RECORDER.show_edges(other, True)

    def test_the_viewer_of_this_environment_has_the_switch(self):
        # The switch is an attribute that Newton's OpenGL renderer gives itself, off, and reads when it
        # draws. A Newton that names it otherwise fails here, not in the middle of a recording.
        import inspect

        from newton._src.viewer.gl.opengl import RendererGL

        source = inspect.getsource(RendererGL)
        self.assertIn("self.draw_edges = False", source)
        self.assertIn("if self.draw_edges:", source)


class TestCropEntry(unittest.TestCase):
    def test_the_rectangle_is_checked_against_the_captured_frame(self):
        self.assertEqual(RECORDER.crop_box([322, 50, 800, 660], 1280, 720), (322, 50, 800, 660))
        # The whole frame, and a part that ends at its lower right corner.
        self.assertEqual(RECORDER.crop_box([0, 0, 64, 60], 64, 60), (0, 0, 64, 60))
        self.assertEqual(RECORDER.crop_box([63, 59, 2, 2], 65, 61), (63, 59, 2, 2))
        for entry, message in (
            ({"x": 0, "y": 0, "width": 2, "height": 2}, "four whole numbers"),
            ([0, 0, 32], "four whole numbers"),
            ([0, 0, 32, 30, 1], "four whole numbers"),
            ([0.0, 0, 32, 30], "four whole numbers"),
            ([0, 0, "32", 30], "four whole numbers"),
            ([False, 0, 32, 30], "four whole numbers"),
            ([-2, 0, 32, 30], "x >= 0 and y >= 0"),
            ([0, -2, 32, 30], "x >= 0 and y >= 0"),
            ([0, 0, 0, 30], "positive width and height"),
            ([0, 0, 32, -30], "positive width and height"),
            ([0, 0, 31, 30], "must be even for the MP4; they are 31 and 30"),
            ([0, 0, 32, 29], "must be even for the MP4; they are 32 and 29"),
            ([34, 0, 32, 30], r"reaches to pixel \(66, 30\), outside the captured frame of 64x60"),
            ([0, 32, 32, 30], r"reaches to pixel \(32, 62\), outside the captured frame of 64x60"),
            ([0, 0, 66, 60], "outside the captured frame of 64x60"),
        ):
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, message):
                RECORDER.crop_box(entry, 64, 60)

    def test_a_frame_is_cut_from_its_top_left_corner(self):
        # Every pixel names its own place: red is its column, green its row.
        frame = np.zeros((60, 64, 3), dtype=np.uint8)
        frame[..., 0] = np.arange(64)[None, :]
        frame[..., 1] = np.arange(60)[:, None]
        part = RECORDER.crop_frame(frame, RECORDER.crop_box([10, 34, 40, 20], 64, 60))
        self.assertEqual(part.shape, (20, 40, 3))
        # x counts columns from the left edge and y rows from the top edge.
        self.assertEqual(part[0, 0].tolist(), [10, 34, 0])
        self.assertEqual(part[-1, -1].tolist(), [49, 53, 0])
        np.testing.assert_array_equal(part, frame[34:54, 10:50])
        np.testing.assert_array_equal(RECORDER.crop_frame(frame, (0, 0, 64, 60)), frame)
        # The frame that is cut is the one the rectangle was checked against.
        with self.assertRaisesRegex(ValueError, "reaches outside the frame of 64x60"):
            RECORDER.crop_frame(frame, (10, 34, 40, 28))
        with self.assertRaisesRegex(ValueError, "reaches outside the frame of 64x60"):
            RECORDER.crop_frame(frame, (26, 34, 40, 20))


class _Viewer:
    """The part of Newton's OpenGL viewer that takes a camera: its camera and its own ``set_camera``."""

    def __init__(self, up_axis, width=960, height=540):
        import newton.viewer
        from newton._src.viewer.camera import Camera

        self.camera = Camera(width=width, height=height, up_axis=up_axis)
        self._set_camera = newton.viewer.ViewerGL.set_camera

    def set_camera(self, pos, pitch=None, yaw=None):
        self._set_camera(self, pos, pitch, yaw)

    def pixel(self, point):
        """Where the viewer's matrices put a world point in the picture: (x, y) from the top left corner."""
        camera = self.camera
        view = np.asarray(camera.get_view_matrix(), dtype=np.float64).reshape((4, 4), order="F")
        projection = np.asarray(camera.get_projection_matrix(), dtype=np.float64).reshape((4, 4), order="F")
        clip = projection @ view @ np.append(np.asarray(point, dtype=np.float64), 1.0)
        x, y = clip[:2] / clip[3]
        return 0.5 * (x + 1.0) * camera.width, 0.5 * (1.0 - y) * camera.height


class TestCameraOnTheViewer(unittest.TestCase):
    def test_the_viewer_looks_at_the_target_also_from_nearer_than_its_look_at_reaches(self):
        for up_axis in (0, 1, 2):
            for position, target in (
                ((0.17, -0.055, 0.1), (0.0, 0.01, 0.0)),
                # 12 mm from the target: Camera.look_at keeps its old direction below 50 mm.
                ((0.004, 0.006, 0.011), (0.0, 0.0, 0.0)),
                ((-0.3, 0.2, -0.1), (0.1, 0.1, 0.25)),
            ):
                with self.subTest(up_axis=up_axis, position=position):
                    viewer = _Viewer(up_axis)
                    camera = {"position": position, "target": target, "fov": 30.0, "near": 0.001, "far": 20.0}
                    RECORDER.place_camera(viewer, camera)
                    direction = np.subtract(target, position)
                    np.testing.assert_allclose(
                        np.asarray(viewer.camera.get_front(), dtype=float),
                        direction / np.linalg.norm(direction),
                        atol=1.0e-12,
                    )
                    np.testing.assert_allclose(np.asarray(viewer.camera.pos, dtype=float), position, atol=0.0)
                    self.assertEqual((viewer.camera.fov, viewer.camera.near, viewer.camera.far), (30.0, 0.001, 20.0))
                    # The target is the middle of the picture.
                    np.testing.assert_allclose(viewer.pixel(target), (480.0, 270.0), atol=0.02)
                    state = RECORDER.camera_state(viewer)
                    np.testing.assert_allclose(state["position"], position, atol=0.0)
                    self.assertEqual((state["pitch"], state["yaw"]), (viewer.camera.pitch, viewer.camera.yaw))

    def test_the_picture_follows_the_written_convention_with_z_up(self):
        # From +X toward the origin, level: yaw 180, pitch 0. +Y is then at the right of the picture and +Z above.
        self.assertEqual(RECORDER.view_angles((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 2), (0.0, 180.0))
        pitch, yaw = RECORDER.view_angles((0.0, 0.0, 1.0), (1.0, 1.0, 0.0), 2)
        self.assertAlmostEqual(yaw, 45.0)
        self.assertAlmostEqual(pitch, -math.degrees(math.atan(1.0 / math.sqrt(2.0))))
        viewer = _Viewer(2)
        distance, fov = 0.2, 30.0
        RECORDER.place_camera(viewer, {"position": (distance, 0.0, 0.0), "target": (0.0, 0.0, 0.0), "fov": fov})
        # The picture is 2 d tan(fov / 2) high at the target and wider by the aspect ratio.
        half_height = distance * math.tan(math.radians(0.5 * fov))
        np.testing.assert_allclose(viewer.pixel((0.0, 0.0, half_height)), (480.0, 0.0), atol=1.0e-3)
        np.testing.assert_allclose(viewer.pixel((0.0, half_height * 960 / 540, 0.0)), (960.0, 270.0), atol=1.0e-3)
        np.testing.assert_allclose(
            viewer.pixel((0.0, 0.01, 0.02)),
            (480.0 + 0.01 / half_height * 270.0, 270.0 - 0.02 / half_height * 270.0),
            atol=1.0e-3,
        )
        # The earlier clip distances stay without the entries.
        self.assertEqual((viewer.camera.near, viewer.camera.far), (0.01, 1000.0))
        # The pivot is the target: the distance to it is the depth reference of transparent surfaces.
        self.assertAlmostEqual(viewer.camera.pivot_distance, distance)

    def test_a_camera_steeper_than_the_viewer_allows_is_refused(self):
        for up_axis in (0, 1, 2):
            above = [0.0, 0.0, 0.0]
            above[up_axis] = 1.0
            with self.subTest(up_axis=up_axis), self.assertRaisesRegex(ValueError, "viewer allows 89"):
                RECORDER.view_angles(above, (0.0, 0.0, 0.0), up_axis)
        # One degree from the vertical is the bound; the picture's up is then the horizontal part of the view.
        tilt = math.tan(math.radians(1.5))
        pitch, yaw = RECORDER.view_angles((0.0, -tilt, 1.0), (0.0, 0.0, 0.0), 2)
        self.assertAlmostEqual(pitch, -88.5)
        self.assertAlmostEqual(yaw, 90.0)
        viewer = _Viewer(2)
        RECORDER.place_camera(viewer, {"position": (0.0, -tilt, 1.0), "target": (0.0, 0.0, 0.0), "fov": 30.0})
        x, y = viewer.pixel((0.0, 0.1, 0.0))
        self.assertAlmostEqual(x, 480.0, places=3)
        self.assertLess(y, 270.0)


def options(**changes):
    values = {
        "frames": None,
        "fps": None,
        "width": 64,
        "height": 60,
        "device": "cpu",
        "profiles": None,
        "stills": [],
        "gif": False,
        "arguments": [],
        "output_dir": None,
    }
    return argparse.Namespace(**(values | changes))


class TestRecipes(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "docs").mkdir()
        (self.root / "docs/videos.json").write_text(json.dumps({"scene": {"frames": 90, "camera": CLOSE}}))
        self.other = self.root / "mine.json"
        self.other.write_text(
            json.dumps(
                {
                    "scene": {"frames": 50},
                    "bad": {"camera": CLOSE | {"up": [0, 0, 1]}},
                    "lines": {"frames": 50, "edges": [[10, 19], [40, 49]], "crop": [4, 30, 60, 30]},
                }
            )
        )
        patcher = patch.object(RECORDER, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_recipes_come_from_the_repository_or_from_the_named_file(self):
        profile, frames, camera = RECORDER.recipe("scene", options())
        self.assertEqual((frames, camera[0]["fov"], profile["camera"]), (90, 30.0, CLOSE))
        # An example without an entry in docs/videos.json is recorded with the defaults, as before.
        self.assertEqual(RECORDER.recipe("absent", options()), ({}, 240, None))
        (self.root / "docs/videos.json").unlink()
        self.assertEqual(RECORDER.recipe("scene", options()), ({}, 240, None))
        profile, frames, camera = RECORDER.recipe("scene", options(profiles=self.other, frames=20))
        self.assertEqual((profile, frames, camera), ({"frames": 50}, 20, None))
        with self.assertRaisesRegex(ValueError, "mine.json has no recipe for absent"):
            RECORDER.recipe("absent", options(profiles=self.other))
        with self.assertRaisesRegex(ValueError, "bad: camera has unknown entries"):
            RECORDER.recipe("bad", options(profiles=self.other))
        with self.assertRaises(OSError):
            RECORDER.recipe("scene", options(profiles=self.root / "missing.json"))

    def test_edges_and_crop_are_checked_with_the_recipe_and_leave_what_it_returns_as_it_was(self):
        entry = {"frames": 50, "edges": [[10, 19], [40, 49]], "crop": [4, 30, 60, 30]}
        # The same three things as for a recipe without the entries: the recipe, the frames, the cameras.
        self.assertEqual(RECORDER.recipe("lines", options(profiles=self.other)), (entry, 50, None))
        # The intervals are frames of the clip that is recorded: fewer frames leave the second one outside.
        with self.assertRaisesRegex(ValueError, r"lines: edges interval \[40, 49\] is outside the clip of 45 frames"):
            RECORDER.recipe("lines", options(profiles=self.other, frames=45))
        # The rectangle is one of the picture that is captured: a smaller picture leaves it outside.
        with self.assertRaisesRegex(ValueError, r"lines: crop \[4, 30, 60, 30\] reaches .* frame of 62x60 \(--width"):
            RECORDER.recipe("lines", options(profiles=self.other, width=62))
        for change, message in (
            ({"edges": [[10, 9]]}, "lines: edges interval"),
            ({"edges": None}, "lines: edges must be a list"),
            ({"crop": [4, 30, 60, 31]}, "lines: crop width and height must be even"),
            ({"crop": None}, "lines: crop must be"),
        ):
            self.other.write_text(json.dumps({"lines": entry | change}))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                RECORDER.recipe("lines", options(profiles=self.other))

    def test_a_still_outside_the_clip_is_refused_before_the_recording(self):
        self.assertEqual(RECORDER.recipe("scene", options(profiles=self.other, stills=[0, 49]))[1], 50)
        with self.assertRaisesRegex(ValueError, "still 50 is outside the clip of 50 frames"):
            RECORDER.recipe("scene", options(profiles=self.other, stills=[0, 50]))


class TestInsertionClipShowsItsMeshOnAnOpaquePad(unittest.TestCase):
    """The repository's own recipe and scenario of fem_needle_insertion hold one interval in two places.

    The recipe's ``edges`` are the frames in which the clip shows the mesh edges; the scenario draws
    the pad opaque in ``INSERTION_OPAQUE_FRAMES``, because through the see-through pad the lines of
    its underside lie over those of its top surface.
    """

    def setUp(self):
        self.recipe = json.loads((TOOLS.parent / "docs/videos.json").read_text())["fem_needle_insertion"]
        # The scenario file as the recorder loads it.
        self.scenario = runpy.run_path(str(TOOLS / "video_scenarios.py"))

    def test_the_scenarios_opaque_frames_are_the_edges_interval_of_the_recipe(self):
        first, last = self.scenario["INSERTION_OPAQUE_FRAMES"]
        self.assertEqual(self.recipe["edges"], [[first, last]])

    def test_the_scripted_user_draws_the_pad_opaque_in_exactly_the_frames_with_the_edges(self):
        # The clip starts with the pad see-through; the scripted user switches that off and on again.
        self.assertIn("--transparent-pad", self.recipe["args"])
        frames = self.recipe["frames"]
        intervals = RECORDER.edge_intervals(self.recipe["edges"], frames)
        example = Mock()
        example.args = argparse.Namespace(transparent_pad=True)
        opaque = []
        for frame in range(frames):
            # As in a recording: the scripted action of a frame comes before the frame is drawn.
            self.scenario["before_frame"]("fem_needle_insertion", example, frame)
            if not example.args.transparent_pad:
                opaque.append(frame)
        self.assertEqual(opaque, [frame for frame in range(frames) if RECORDER.edges_at(intervals, frame)])
        self.assertTrue(example.args.transparent_pad)


EXAMPLE = '''
"""A scene that frames itself at every turn."""
import argparse

import newton.examples
import newton.viewer

OWN = dict(pos=(1.0, 2.0, 3.0), pitch=-10.0, yaw=20.0)


class Example:
    def __init__(self, viewer):
        self.viewer, self.seen = viewer, []
        viewer.set_camera(**OWN)

    def step(self):
        # A reset, a framing helper, a scripted self-test.
        self.viewer.set_camera(**OWN)
        # A display option of its own: this scene draws its edges.
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_edges = True

    def render(self):
        self.viewer.begin_frame(0.0)
        self.viewer.camera.fov = 70.0
        self.viewer.end_frame()


if __name__ == "__main__":
    newton.examples.run(Example(newton.viewer.ViewerGL(headless=True)), argparse.Namespace(test=False))
'''

SCENARIO = """
def before_frame(name, example, frame):
    camera = example.viewer.camera
    example.seen.append((frame, tuple(camera.pos), camera.fov))
    example.viewer.seen = example.seen
"""


class TestRecordingWithACamera(unittest.TestCase):
    """The whole recorder on a scene without physics and a viewer without OpenGL."""

    def setUp(self):
        # Before the encoder is replaced below: the first import of newton.examples starts a process of its own.
        import newton.examples
        import newton.viewer
        from newton._src.viewer.camera import Camera

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        examples = self.root / "src/recorded_scene/examples"
        examples.mkdir(parents=True)
        (examples.parent / "__init__.py").touch()
        (examples / "__init__.py").touch()
        (examples / "example_scene.py").write_text(textwrap.dedent(EXAMPLE))
        (self.root / "tools").mkdir()
        (self.root / "tools/video_scenarios.py").write_text(SCENARIO)
        self.output = self.root / "clips"
        set_camera = newton.viewer.ViewerGL.set_camera
        viewers = self.viewers = []
        encoders = self.encoders = []
        # What the tests below change about the viewer: it has a renderer with the edge switch, its
        # picture is one grey level per frame, and it is as large as the recorder asks for.
        fixture = self.fixture = argparse.Namespace(renderer=True, pattern=False, wider=0)

        class Renderer:
            """The edge switch of Newton's OpenGL renderer, off at first, and how often it was set."""

            def __init__(self):
                self._draw_edges, self.sets = False, 0

            @property
            def draw_edges(self):
                return self._draw_edges

            @draw_edges.setter
            def draw_edges(self, value):
                self._draw_edges, self.sets = value, self.sets + 1

        class Viewer:
            """Newton's camera and ``set_camera`` without a window: a frame is drawn by noting its camera."""

            def __init__(self, width, height, **_):
                self.camera = Camera(width=width, height=height, up_axis="Z")
                self.width, self.height = width, height
                self.drawn, self.calls, self.seen = [], 0, []
                # The edge switch as it stood when each frame was drawn.
                self.edges_drawn = []
                if fixture.renderer:
                    self.renderer = Renderer()
                viewers.append(self)

            def set_camera(self, pos, pitch=None, yaw=None):
                self.calls += 1
                set_camera(self, pos, pitch, yaw)

            def begin_frame(self, time):
                pass

            def end_frame(self):
                self.drawn.append(RECORDER.camera_state(self))
                if fixture.renderer:
                    self.edges_drawn.append(self.renderer.draw_edges)

            def get_frame(self, render_ui=False):
                # Each frame has its own grey level, so a still can be told from its neighbours.
                width = self.width + fixture.wider
                picture = np.full((self.height, width, 3), 20 * len(self.drawn), dtype=np.uint8)
                if fixture.pattern:
                    # Every pixel names its own place: red is its column, green its row.
                    picture[..., 0] = np.arange(width)[None, :]
                    picture[..., 1] = np.arange(self.height)[:, None]
                return Mock(numpy=lambda: picture)

            def close(self):
                pass

        def encoder(command, **_):
            # The clip of the encoder, and no file for any other process.
            self.assertEqual(command[0], "ffmpeg")
            Path(command[-1]).touch()
            process = Mock()
            process.wait.return_value = 0
            process.command = command
            encoders.append(process)
            return process

        def converter(command, **_):
            # ffmpeg on the finished clip writes the file its command ends with: a poster or an animated preview.
            self.assertEqual(command[0], "ffmpeg")
            Path(command[-1]).touch()

        self.converted = Mock(side_effect=converter)
        sys.path.insert(0, str(self.root / "src"))
        self.addCleanup(sys.path.remove, str(self.root / "src"))
        self.addCleanup(
            lambda: [sys.modules.pop(name) for name in list(sys.modules) if name.startswith("recorded_scene")]
        )
        for target, name, value in (
            (RECORDER, "ROOT", self.root),
            (RECORDER, "project", lambda: ({}, "recorded_scene", {"scene": ("A scene.", False, "example_scene")})),
            (RECORDER, "version", lambda name: "0"),
            (RECORDER.subprocess, "Popen", Mock(side_effect=encoder)),
            (RECORDER.subprocess, "run", self.converted),
            (newton.viewer, "ViewerGL", Viewer),
            (sys, "argv", sys.argv),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def record(self, recipe, **changes):
        profiles = self.root / "mine.json"
        profiles.write_text(json.dumps({"scene": recipe}))
        with patch("builtins.print"):
            RECORDER.record("scene", options(profiles=profiles, output_dir=self.output, **changes))
        return self.viewers[-1], json.loads((self.output / "scene.json").read_text())

    def written(self):
        """The frames that the last recording gave to the encoder, and the picture size it named to it."""
        process = self.encoders[-1]
        size = process.command[process.command.index("-video_size") + 1]
        width, height = (int(value) for value in size.split("x"))
        frames = [
            np.frombuffer(call.args[0], dtype=np.uint8).reshape(height, width, 3)
            for call in process.stdin.write.call_args_list
        ]
        return frames, (width, height)

    def test_the_recipes_camera_wins_over_the_examples_own_framing_in_every_frame(self):
        viewer, sidecar = self.record({"frames": 6, "camera": CLOSE | {"near": 0.002}}, stills=[0, 4])
        self.assertEqual(len(viewer.drawn), 6)
        direction = np.subtract(CLOSE["target"], CLOSE["position"])
        for drawn in viewer.drawn:
            self.assertEqual(drawn["position"], CLOSE["position"])
            self.assertEqual((drawn["fov"], drawn["near"], drawn["far"]), (30.0, 0.002, 1000.0))
            np.testing.assert_allclose(drawn["view_direction"], direction / np.linalg.norm(direction), atol=1.0e-12)
        # The scripted action of a frame, which may pick through a pixel, has the recipe's camera too.
        self.assertEqual(viewer.seen, [(frame, tuple(CLOSE["position"]), 30.0) for frame in range(6)])
        self.assertEqual(sidecar["camera"], CLOSE | {"near": 0.002})
        self.assertEqual([view["frame"] for view in sidecar["camera_view"]], [0, 5])
        self.assertEqual(sidecar["camera_view"][0], {"frame": 0} | viewer.drawn[0])
        self.assertEqual(sidecar["frames"], 6)
        # The stills are the frames that went to the encoder, title bar included.
        self.assertEqual(
            sorted(path.name for path in self.output.glob("*still*")),
            ["scene.still-00000.jpg", "scene.still-00004.jpg"],
        )
        from PIL import Image

        for frame, grey in ((0, 20), (4, 100)):
            picture = np.asarray(Image.open(self.output / f"scene.still-{frame:05d}.jpg"))
            self.assertEqual(picture.shape, (60, 64, 3))
            np.testing.assert_allclose(picture[40:], grey, atol=3)
            np.testing.assert_allclose(picture[:4, 40:], np.broadcast_to((25, 28, 34), (4, 24, 3)), atol=3)

    def test_a_timed_camera_is_drawn_frame_by_frame(self):
        far = {"position": [0.23, -0.04, 0.05], "target": [0.01, 0.0, 0.0], "fov": 10.0}
        viewer, sidecar = self.record({"frames": 5, "camera": [CLOSE | {"frame": 1}, far | {"frame": 3}]})
        np.testing.assert_allclose(
            [drawn["position"][0] for drawn in viewer.drawn], [0.03, 0.03, 0.13, 0.23, 0.23], atol=1.0e-12
        )
        np.testing.assert_allclose([drawn["fov"] for drawn in viewer.drawn], [30.0, 30.0, 20.0, 10.0, 10.0])
        self.assertEqual(sidecar["camera"], [CLOSE | {"frame": 1}, far | {"frame": 3}])
        self.assertEqual(sidecar["camera_view"][1]["position"], far["position"])

    def test_without_the_entry_the_viewer_is_left_to_the_example(self):
        viewer, sidecar = self.record({"frames": 4})
        self.assertEqual(len(viewer.drawn), 4)
        for drawn in viewer.drawn:
            self.assertEqual(
                (drawn["position"], drawn["pitch"], drawn["yaw"], drawn["fov"]), ([1.0, 2.0, 3.0], -10.0, 20.0, 70.0)
            )
        # The example's own calls and no other: one in its constructor and one in each step.
        self.assertEqual(viewer.calls, 5)
        self.assertNotIn("camera", sidecar)
        self.assertNotIn("camera_view", sidecar)
        self.assertEqual(list(self.output.glob("*still*")), [])
        # The same for the edge switch and the picture: the example's own four settings and no other,
        # every frame drawn as the example set it, and the frames written as large as they were captured.
        self.assertEqual((viewer.renderer.sets, viewer.edges_drawn), (4, [True] * 4))
        frames, size = self.written()
        self.assertEqual((size, len(frames)), ((64, 60), 4))
        self.assertEqual((sidecar["width"], sidecar["height"]), (64, 60))
        for name in ("edges", "crop", "captured_width", "captured_height"):
            self.assertNotIn(name, sidecar)

    def test_the_edges_are_switched_before_each_frame_is_drawn(self):
        # The example switches its edges on in every step; the recipe's switch is given after it.
        viewer, sidecar = self.record({"frames": 9, "edges": [[2, 3], [6, 6]]})
        self.assertEqual(viewer.edges_drawn, [False, False, True, True, False, False, True, False, False])
        self.assertTrue(all(type(value) is bool for value in viewer.edges_drawn))
        self.assertEqual(sidecar["edges"], [[2, 3], [6, 6]])
        self.assertEqual((sidecar["frames"], sidecar["width"], sidecar["height"]), (9, 64, 60))
        self.assertNotIn("crop", sidecar)
        # From the first frame to the last, and no interval at all: off in every frame.
        viewer, sidecar = self.record({"frames": 3, "edges": [[0, 2]]})
        self.assertEqual(viewer.edges_drawn, [True, True, True])
        viewer, sidecar = self.record({"frames": 3, "edges": []})
        self.assertEqual((viewer.edges_drawn, sidecar["edges"]), ([False, False, False], []))
        # With a camera both are given where the frame is drawn.
        viewer, sidecar = self.record({"frames": 4, "edges": [[1, 2]], "camera": CLOSE})
        self.assertEqual(viewer.edges_drawn, [False, True, True, False])
        self.assertEqual([drawn["position"] for drawn in viewer.drawn], [CLOSE["position"]] * 4)
        self.assertEqual((sidecar["edges"], sidecar["camera"]), ([[1, 2]], CLOSE))

    def test_edges_on_a_viewer_without_the_switch_stop_the_recording_before_its_first_frame(self):
        self.fixture.renderer = False
        with self.assertRaisesRegex(ValueError, "viewer has no renderer.draw_edges"):
            self.record({"frames": 5, "edges": [[1, 2]]})
        self.assertEqual(self.viewers[-1].drawn, [])
        # Without the entry such a viewer records as before.
        viewer, sidecar = self.record({"frames": 2})
        self.assertEqual(len(viewer.drawn), 2)
        self.assertFalse(hasattr(viewer, "renderer"))
        self.assertNotIn("edges", sidecar)

    def test_every_written_picture_is_the_cropped_part_of_the_captured_frame(self):
        from PIL import Image

        self.fixture.pattern = True
        viewer, sidecar = self.record({"frames": 3, "crop": [10, 34, 40, 20]}, stills=[1])
        frames, size = self.written()
        self.assertEqual((size, len(frames)), ((40, 20), 3))
        for index, frame in enumerate(frames):
            # Below the title bar the part is the captured picture: columns 10 to 49, rows 34 to 53.
            np.testing.assert_array_equal(frame[..., 0], np.broadcast_to(np.arange(10, 50)[None, :], (20, 40)))
            np.testing.assert_array_equal(frame[..., 1], np.broadcast_to(np.arange(34, 54)[:, None], (20, 40)))
            np.testing.assert_array_equal(frame[..., 2], 20 * (index + 1))
        # The clip's size is that of the part; the entry and the captured size are recorded beside it.
        self.assertEqual((sidecar["width"], sidecar["height"]), (40, 20))
        self.assertEqual(sidecar["crop"], [10, 34, 40, 20])
        self.assertEqual((sidecar["captured_width"], sidecar["captured_height"]), (64, 60))
        self.assertNotIn("edges", sidecar)
        # The viewer was not asked for another picture, and its switch was left to the example.
        self.assertEqual((viewer.width, viewer.height, viewer.edges_drawn), (64, 60, [True] * 3))
        # The still and the poster (the last frame, without a poster frame) are the same part.
        still = np.asarray(Image.open(self.output / "scene.still-00001.jpg"))
        poster = np.asarray(Image.open(self.output / "scene.jpg"))
        self.assertEqual((still.shape, poster.shape), ((20, 40, 3), (20, 40, 3)))
        np.testing.assert_allclose(still.astype(int), frames[1].astype(int), atol=12)
        np.testing.assert_allclose(poster.astype(int), frames[2].astype(int), atol=12)
        # With a poster frame and an animated preview, both are made from the finished, cropped clip.
        self.record({"frames": 3, "crop": [10, 34, 40, 20], "poster_frame": 1}, gif=True)
        poster, preview = (call.args[0] for call in self.converted.call_args_list)
        self.assertEqual(poster[poster.index("-i") + 1], str(self.output / "scene.mp4"))
        self.assertEqual(preview[preview.index("-i") + 1], str(self.output / "scene.mp4"))
        self.assertEqual(self.written()[1], (40, 20))

    def test_the_crop_counts_from_the_top_left_corner_of_the_frame_with_its_title_bar(self):
        self.fixture.pattern = True
        self.record({"frames": 2, "crop": [40, 0, 24, 40], "edges": [[1, 1]]})
        (first, second), size = self.written()
        self.assertEqual(size, (24, 40))
        # The recorder's title bar is the upper 30 rows of the captured frame: plain above and below its text.
        np.testing.assert_array_equal(first[:4], np.broadcast_to((25, 28, 34), (4, 24, 3)))
        np.testing.assert_array_equal(first[29], np.broadcast_to((25, 28, 34), (24, 3)))
        np.testing.assert_array_equal(first[30:, :, 0], np.broadcast_to(np.arange(40, 64)[None, :], (10, 24)))
        np.testing.assert_array_equal(first[30:, :, 1], np.broadcast_to(np.arange(30, 40)[:, None], (10, 24)))
        self.assertEqual(self.viewers[-1].edges_drawn, [False, True])
        # A part that begins below the bar has none.
        self.record({"frames": 1, "crop": [0, 30, 64, 30]})
        ((only,), size) = self.written()
        self.assertEqual(size, (64, 30))
        np.testing.assert_array_equal(only[..., 1], np.broadcast_to(np.arange(30, 60)[:, None], (30, 64)))

    def test_a_captured_frame_of_another_size_than_the_crop_was_checked_against_is_refused(self):
        self.fixture.wider = 2
        with self.assertRaisesRegex(ValueError, "captured frame is 66x60, not the 64x60 that the crop was checked"):
            self.record({"frames": 2, "crop": [10, 34, 40, 20]})

    def test_a_crop_outside_the_captured_frame_stops_the_recording_before_a_viewer_is_made(self):
        made = len(self.viewers)
        with self.assertRaisesRegex(ValueError, "scene: crop .* outside the captured frame of 64x60"):
            self.record({"frames": 2, "crop": [10, 34, 60, 20]})
        with self.assertRaisesRegex(ValueError, "scene: edges interval .* outside the clip of 2 frames"):
            self.record({"frames": 2, "edges": [[1, 2]]})
        self.assertEqual(len(self.viewers), made)

    def test_a_terminal_only_example_takes_a_crop_and_no_edges(self):
        (self.root / "tools/video_scenarios.py").write_text(
            textwrap.dedent(
                """
                import numpy as np


                def record_terminal(name, device, frames, width, height):
                    for frame in range(frames):
                        picture = np.full((height, width, 3), 20 * (frame + 1), dtype=np.uint8)
                        picture[..., 0] = np.arange(width)[None, :]
                        yield picture
                """
            )
        )
        terminal = {"scene": ("A scene.", True, "example_scene")}
        with patch.object(RECORDER, "project", lambda: ({}, "recorded_scene", terminal)):
            with self.assertRaisesRegex(ValueError, "scene is terminal-only; its recipe cannot hold edges"):
                self.record({"frames": 2, "edges": [[0, 1]]})
            profiles = self.root / "mine.json"
            profiles.write_text(json.dumps({"scene": {"frames": 2, "crop": [10, 34, 40, 20]}}))
            with patch("builtins.print"):
                RECORDER.record("scene", options(profiles=profiles, output_dir=self.output))
        frames, size = self.written()
        self.assertEqual((size, len(frames)), ((40, 20), 2))
        np.testing.assert_array_equal(frames[1][..., 0], np.broadcast_to(np.arange(10, 50)[None, :], (20, 40)))
        np.testing.assert_array_equal(frames[1][..., 2], 40)
        sidecar = json.loads((self.output / "scene.json").read_text())
        self.assertEqual((sidecar["width"], sidecar["height"], sidecar["crop"]), (40, 20, [10, 34, 40, 20]))

    def test_an_animated_preview_is_written_only_when_asked_for(self):
        _, sidecar = self.record({"frames": 3})
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), ["scene.jpg", "scene.json", "scene.mp4"])
        # Without a poster frame the poster is the last captured frame: no second ffmpeg process at all.
        self.converted.assert_not_called()
        # The sidecar names no file of the clip, so none that was not written.
        self.assertNotRegex(json.dumps(sidecar), r"\.(gif|jpg|mp4)")
        _, asked = self.record({"frames": 3, "poster_frame": 1}, gif=True)
        self.assertEqual(
            sorted(path.name for path in self.output.iterdir()), ["scene.gif", "scene.jpg", "scene.json", "scene.mp4"]
        )
        poster, preview = (call.args[0] for call in self.converted.call_args_list)
        self.assertEqual(poster[-1], str(self.output / "scene.jpg"))
        self.assertEqual(preview[-1], str(self.output / "scene.gif"))
        # The preview is made from the finished clip.
        self.assertEqual(preview[preview.index("-i") + 1], str(self.output / "scene.mp4"))
        self.assertEqual(set(asked), set(sidecar))
        self.assertNotRegex(json.dumps(asked), r"\.(gif|jpg|mp4)")

    def test_a_camera_the_viewer_cannot_take_stops_the_recording_before_its_first_frame(self):
        above = {"position": [0.0, 0.0, 0.2], "target": [0.0, 0.0, 0.0], "fov": 30.0}
        with self.assertRaisesRegex(ValueError, "viewer allows 89"):
            self.record({"frames": 30, "camera": [CLOSE | {"frame": 0}, above | {"frame": 20}]})
        self.assertEqual(self.viewers[-1].drawn, [])


class TestCommandLine(unittest.TestCase):
    """What the recorder hands to the process that records, without starting one."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        (self.root / "docs").mkdir()
        (self.root / "docs/videos.json").write_text(json.dumps({"scene": {"frames": 90}}))
        (self.root / "mine.json").write_text(
            json.dumps({"scene": {"frames": 50, "camera": CLOSE}, "bad": {"camera": {}}})
        )
        (self.root / "lines.json").write_text(
            json.dumps({"scene": {"frames": 50, "edges": [[40, 49]], "crop": [322, 50, 800, 660]}})
        )
        self.run = Mock(return_value=Mock(returncode=0))
        examples = {name: ("A scene.", False, f"example_{name}") for name in ("scene", "bad")}
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(self.root)
        for target, name, value in (
            (RECORDER, "ROOT", self.root),
            (RECORDER, "project", lambda: ({}, "recorded_scene", examples)),
            (RECORDER.shutil, "which", lambda name: "/usr/bin/" + name),
            (RECORDER.subprocess, "run", self.run),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def main(self, *arguments):
        with patch.object(sys, "argv", ["record_examples.py", *arguments]):
            RECORDER.main()
        return [call.args[0] for call in self.run.call_args_list]

    def refused(self, *arguments):
        message = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(message):
            self.main(*arguments)
        self.run.assert_not_called()
        return message.getvalue()

    def test_the_recipe_file_and_the_stills_reach_the_recording_from_the_callers_folder(self):
        (command,) = self.main("scene", "--profiles", "mine.json", "--stills", "30,0", "--output-dir", "clips")
        self.assertEqual(command[2:4], ["scene", "--child"])
        self.assertEqual(command[command.index("--profiles") + 1], str(self.root / "mine.json"))
        self.assertEqual(command[command.index("--stills") + 1], "0,30")
        self.assertEqual(command[command.index("--output-dir") + 1], str(self.root / "clips"))
        # The recording itself runs in the repository; the README shows docs/media and is left alone.
        self.assertEqual(self.run.call_args.kwargs["cwd"], self.root)

    def test_without_the_options_the_recording_is_started_as_before_and_the_readme_follows(self):
        command, readme = self.main("scene")
        self.assertEqual(
            command[2:],
            ["scene", "--child", "--device", "cuda:0", "--width", "960", "--height", "540"]
            + ["--output-dir", str(self.root / "docs/media")],
        )
        self.assertEqual(readme, [sys.executable, str(self.root / "tools/build_readme.py")])

    def test_the_request_for_an_animated_preview_reaches_the_recording(self):
        (command,) = self.main("scene", "--gif", "--output-dir", "clips")
        self.assertEqual(command[2:4], ["scene", "--child"])
        self.assertEqual(command[-1], "--gif")

    def test_edges_and_crop_are_entries_of_the_recipe_and_not_options_of_the_recording(self):
        # The rectangle of the recipe lies inside a picture of 1280x720: the recording is started
        # with the same command as for a recipe without the entries.
        (command,) = self.main(
            "scene", "--profiles", "lines.json", "--width", "1280", "--height", "720", "--output-dir", "clips"
        )
        self.assertEqual(
            command[2:],
            ["scene", "--child", "--device", "cuda:0", "--width", "1280", "--height", "720"]
            + ["--output-dir", str(self.root / "clips"), "--profiles", str(self.root / "lines.json")],
        )

    def test_edges_or_a_crop_that_do_not_fit_the_recording_are_refused_before_a_simulation_starts(self):
        # The picture of the default size is smaller than the recipe's rectangle.
        self.assertIn(
            "scene: crop [322, 50, 800, 660] reaches to pixel (1122, 710), outside the captured frame of 960x540",
            self.refused("scene", "--profiles", "lines.json"),
        )
        # A shorter clip ends before the recipe's interval.
        self.assertIn(
            "scene: edges interval [40, 49] is outside the clip of 30 frames (0 to 29)",
            self.refused("scene", "--profiles", "lines.json", "--width", "1280", "--height", "720", "--frames", "30"),
        )

    def test_a_recipe_that_cannot_be_recorded_is_refused_before_a_simulation_starts(self):
        self.assertIn("still 90 is outside the clip of 90 frames", self.refused("scene", "--stills", "0,90"))
        self.assertIn("still 50 is outside", self.refused("scene", "--profiles", "mine.json", "--stills", "50"))
        self.assertIn("whole numbers separated by commas", self.refused("scene", "--stills", "first"))
        self.assertIn("bad: camera lacks position, target, fov", self.refused("bad", "--profiles", "mine.json"))
        (self.root / "mine.json").write_text(json.dumps({"bad": {}}))
        self.assertIn("mine.json has no recipe for scene", self.refused("--all", "--profiles", "mine.json"))
        self.assertIn("No such file", self.refused("scene", "--profiles", "none.json"))


if __name__ == "__main__":
    unittest.main()
