# SPDX-License-Identifier: Apache-2.0
"""Defaults of fem_needle_grasp: the swept tool/tissue contact and a frame time budget of 66.7 ms.

The example turns both on. The parser of the scene and ``SutureSceneConfig`` keep the contact
off, and a budget stays an argument of one call of ``LocalSutureScene.step``, so a caller that
builds scenes through ``SutureSceneConfig.example_arguments`` gets what it got before.

The runs on the device that the defaults rest on (2026-11-17, docs/usage.md) were made with the
options written out: ``--tool-tissue-contact swept --frame-budget 66.7`` and the unit of the
budget that the example takes by itself today (host-trial, then an option). The first test is
therefore that the default namespace of the parser is the namespace of those options; the
second, that the two switches give back the namespace the parser gave before the defaults
changed. The thread tracks (``--thread-tracks``, a default of
the example and of the scene since the same day) are not a setting of this change: both
namespaces have them as the parser has them, and the window runs of the round that measured
the budget, made before the tracks, are today the runs with ``--no-thread-tracks`` (the third
test). No scene is built here and nothing runs on a device.
"""

import contextlib
import dataclasses
import inspect
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import newton.viewer

from newton_solver_suturebot.examples.grasp import example_fem_needle_grasp as grasp_example
from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene
from newton_solver_suturebot.examples.grasp.parallel import LocalSutureScene, ParallelSutureScenes, SutureSceneConfig

# The options of the runs on the device, and the two switches that turn the defaults off.
# NO_TRACKS: the pull of the runs of before the thread tracks (it ends at the needle's entry).
WRITTEN_OUT = ("--tool-tissue-contact", "swept", "--frame-budget", "66.7")
SWITCHED_OFF = ("--no-tool-tissue-contact", "--frame-budget", "0")
# What the parser gave for the two settings before they became defaults (with no arguments).
BEFORE = {"tool_tissue_contact": False, "frame_budget": 0.0}
NO_TRACKS = ("--no-thread-tracks",)


class _Window(newton.viewer.ViewerGL):
    """Newton's OpenGL viewer as the example sees it; no window and no renderer is made."""

    def __init__(self, headless=False):
        self._headless, self.num_frames = headless, 2400


def _null():
    """Newton's null viewer, which ends the run after its number of frames."""
    viewer = object.__new__(newton.viewer.ViewerNull)
    object.__setattr__(viewer, "num_frames", 2400)
    return viewer


def _plan(viewer, args):
    """Budget [s] and unit that the example takes for the arguments, and everything it prints about them."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        budget = grasp_example.Example._frame_budget(viewer, args)
        unit = grasp_example.Example._frame_unit(args, budget)
    return budget, unit, output.getvalue().splitlines()


def _bare(**settings):
    """Arguments of a caller of the class that names neither the contact nor the budget."""
    return SimpleNamespace(**({"num_envs": 1} | settings))


class DefaultNamespaceTests(unittest.TestCase):
    def setUp(self):
        self.parser = grasp_example.Example.create_parser()

    def test_the_default_namespace_is_that_of_the_options_of_the_runs_on_the_device(self):
        default = self.parser.parse_args([])
        written = self.parser.parse_args(list(WRITTEN_OUT))
        self.assertEqual(vars(default), vars(written))
        self.assertEqual((default.tool_tissue_contact, default.frame_budget), ("swept", 66.7))
        # One source per default: the constants of the module.
        self.assertEqual((grasp_example.TOOL_TISSUE_CONTACT, grasp_example.FRAME_BUDGET_MS), ("swept", 66.7))
        for name, constant in (
            ("tool_tissue_contact", grasp_example.TOOL_TISSUE_CONTACT),
            ("frame_budget", grasp_example.FRAME_BUDGET_MS),
        ):
            self.assertEqual(self.parser.get_default(name), constant)
        # The unit of the budget is no setting: the example takes it from where the budget is in use.
        self.assertNotIn("frame_budget_unit", vars(default))
        self.assertFalse(hasattr(grasp_example, "FRAME_BUDGET_UNIT"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parser.parse_args(["--frame-budget-unit", "attempt"])
        # The thread tracks are as the example has them, in both namespaces: on.
        self.assertIs(grasp_example.THREAD_TRACKS, True)
        self.assertIs(default.thread_tracks, True)
        self.assertIs(written.thread_tracks, True)
        self.assertNotIn("--no-thread-tracks", grasp_example.Example._coupling(default))
        # What the scene is handed is the same too, and ends with the swept law.
        coupling = grasp_example.Example._coupling
        self.assertEqual(coupling(default), coupling(written))
        self.assertEqual(coupling(default)[-2:], ("--tool-tissue-contact", "swept"))
        self.assertEqual(GraspScene.create_parser().parse_args(list(coupling(default))).tool_tissue_contact, "swept")

    def test_the_two_switches_give_the_namespace_of_before_the_defaults(self):
        default = vars(self.parser.parse_args([]))
        off = vars(self.parser.parse_args(list(SWITCHED_OFF)))
        # Every other setting is the default one; the two have the values the parser gave before.
        self.assertEqual(off, default | BEFORE)
        # Among the other settings: the thread tracks, as the example has them.
        self.assertIs(off["thread_tracks"], grasp_example.THREAD_TRACKS)
        self.assertEqual(set(default) - set(BEFORE), set(off) - set(BEFORE))
        self.assertIn("thread_tracks", set(default) - set(BEFORE))
        for name, value in BEFORE.items():
            self.assertIs(type(off[name]), type(value), name)
        # Each switch alone changes its own setting only.
        for arguments, name in (
            (["--no-tool-tissue-contact"], "tool_tissue_contact"),
            (["--frame-budget", "0"], "frame_budget"),
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(vars(self.parser.parse_args(arguments)), default | {name: BEFORE[name]})
        # The scene then gets no contact argument, as before; `off` as the value of the option does the same.
        coupling = grasp_example.Example._coupling
        without = coupling(self.parser.parse_args(list(SWITCHED_OFF)))
        self.assertNotIn("--tool-tissue-contact", without)
        self.assertEqual(coupling(self.parser.parse_args(["--tool-tissue-contact", "off"])), without)
        self.assertEqual(coupling(self.parser.parse_args([])), (*without, "--tool-tissue-contact", "swept"))
        self.assertIs(GraspScene.create_parser().parse_args(list(without)).tool_tissue_contact, False)
        # And nothing is planned or printed: whole frames, in the window too.
        self.assertEqual(_plan(_Window(), self.parser.parse_args(list(SWITCHED_OFF))), (None, "attempt", []))

    def test_the_runs_of_before_the_thread_tracks_are_the_defaults_with_no_thread_tracks(self):
        # The round's window runs (docs/usage.md) had the options written out and a pull that
        # ended at the entry. Today that is one more switch, which changes its own setting only.
        default = vars(self.parser.parse_args([]))
        round_runs = vars(self.parser.parse_args([*WRITTEN_OUT, *NO_TRACKS]))
        self.assertEqual(round_runs, default | {"thread_tracks": False})
        self.assertEqual(round_runs, vars(self.parser.parse_args(list(NO_TRACKS))))
        # The example of before both changes: the two switches and that one.
        before = vars(self.parser.parse_args([*SWITCHED_OFF, *NO_TRACKS]))
        self.assertEqual(before, default | BEFORE | {"thread_tracks": False})
        # The scene is told so after the contact: the swept law with the pull to the entry.
        coupling = grasp_example.Example._coupling
        with_tracks = coupling(self.parser.parse_args([]))
        without_tracks = coupling(self.parser.parse_args(list(NO_TRACKS)))
        self.assertEqual(without_tracks, (*with_tracks, "--no-thread-tracks"))
        scene = GraspScene.create_parser()
        self.assertIs(scene.parse_args(list(with_tracks)).thread_tracks, True)
        self.assertIs(scene.parse_args(list(without_tracks)).thread_tracks, False)
        self.assertEqual(scene.parse_args(list(without_tracks)).tool_tissue_contact, "swept")

    def test_the_first_version_law_is_a_law_of_the_scene_and_not_of_the_example(self):
        # With it the default procedure stops at 6.4 s (docs/usage.md): the example offers swept or off.
        for arguments in (["--tool-tissue-contact", "first"], ["--tool-tissue-contact"]):
            with (
                self.subTest(arguments=arguments),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                self.parser.parse_args(arguments)
        for value in (True, "first"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "'swept' or 'off'"):
                grasp_example.Example._coupling(SimpleNamespace(tool_tissue_contact=value))
        # The scene keeps both laws for a caller of SutureSceneConfig.example_arguments.
        scene = GraspScene.create_parser()
        self.assertIs(scene.parse_args(["--tool-tissue-contact"]).tool_tissue_contact, True)
        self.assertEqual(scene.parse_args(["--tool-tissue-contact", "first"]).tool_tissue_contact, "first")

    def test_the_help_names_the_new_defaults_and_the_switches(self):
        text = " ".join(self.parser.format_help().split())
        self.assertIn("default: swept", text)
        self.assertIn("default: 66.7", text)
        self.assertIn("0 runs whole frames", text)
        self.assertIn("--no-tool-tissue-contact", text)
        # The statements of before the defaults changed are gone.
        for stale in ("default: off", "0 runs whole frames (default)", "the tissue (the default)"):
            self.assertNotIn(stale, text)
        # Nor are the options that selected a way no longer in use.
        for gone in ("--frame-budget-unit", "--event-search", "--event-oracle", "--event-budget", "'first'"):
            self.assertNotIn(gone, text)


class SceneKeepsItsDefaultsTests(unittest.TestCase):
    def test_the_scene_parser_and_the_scene_configuration_give_no_contact(self):
        scene_parser = GraspScene.create_parser()
        self.assertIs(scene_parser.get_default("tool_tissue_contact"), False)
        self.assertIs(scene_parser.parse_args([]).tool_tissue_contact, False)
        # The scene has no option for a budget: a deadline is an argument of a step.
        scene_defaults = vars(scene_parser.parse_args([]))
        self.assertFalse([name for name in scene_defaults if "budget_unit" in name or name.startswith("frame_budget")])
        for config in (SutureSceneConfig(device="cpu"), SutureSceneConfig(device="cpu", control_mode="physical_grasp")):
            self.assertEqual(config.example_arguments, ())
            # The arguments a worker makes of the configuration (parallel._make_scene).
            arguments = ["--regions", str(config.region), "--substeps", str(config.substeps), *config.example_arguments]
            self.assertIs(scene_parser.parse_args(arguments).tool_tissue_contact, False)
        # A caller switches the contact on as before.
        on = SutureSceneConfig(device="cpu", example_arguments=("--tool-tissue-contact", "swept"))
        self.assertEqual(scene_parser.parse_args(list(on.example_arguments)).tool_tissue_contact, "swept")

    def test_the_scene_configuration_has_no_setting_for_the_contact_or_a_budget(self):
        fields = {field.name: field for field in dataclasses.fields(SutureSceneConfig)}
        self.assertFalse([name for name in fields if "contact" in name or "budget" in name or "deadline" in name])
        # The contact reaches a scene through its arguments only, and none are passed by default.
        self.assertEqual(fields["example_arguments"].default, ())

    def test_a_step_of_a_scene_is_a_whole_frame_unless_the_call_says_otherwise(self):
        local = inspect.signature(LocalSutureScene.step).parameters
        self.assertIsNone(local["deadline"].default)
        self.assertEqual(local["unit"].default, "attempt")
        self.assertIs(local["observe_native_trials"].default, False)
        # Scenes in workers have no deadline at all.
        workers = inspect.signature(ParallelSutureScenes.step).parameters
        self.assertFalse({"deadline", "unit"} & set(workers))


class BareArgumentsTests(unittest.TestCase):
    """A namespace without the two settings (another driver of the class) behaves like the parser's default."""

    def setUp(self):
        self.parser = grasp_example.Example.create_parser()

    def test_arguments_without_the_settings_get_the_scene_arguments_of_the_default(self):
        coupling = grasp_example.Example._coupling
        self.assertEqual(coupling(SimpleNamespace()), coupling(self.parser.parse_args([])))
        self.assertEqual(coupling(SimpleNamespace())[-2:], ("--tool-tissue-contact", "swept"))
        # A caller of the class switches the contact off with the setting.
        for value in (False, None, "off"):
            self.assertNotIn("--tool-tissue-contact", coupling(SimpleNamespace(tool_tissue_contact=value)))

    def test_arguments_without_the_settings_get_the_budget_and_the_unit_of_the_default(self):
        cases = (
            # viewer, settings of the run, the plan
            (_Window, {}, (1.0e-3 * 66.7, "host-trial")),
            (_Window, {"rigid_backend": "native"}, (1.0e-3 * 66.7, "attempt")),
            (_Window, {"test": True}, (None, "attempt")),
            (_Window, {"num_envs": 2}, (None, "attempt")),
            (lambda: _Window(headless=True), {}, (None, "attempt")),
            (_null, {}, (None, "attempt")),
        )
        for make_viewer, settings, expected in cases:
            parsed = self.parser.parse_args([])
            for name, value in settings.items():
                setattr(parsed, name, value)
            with self.subTest(settings=settings, viewer=make_viewer):
                bare = _plan(make_viewer(), _bare(**settings))
                self.assertEqual(bare, _plan(make_viewer(), parsed))
                self.assertEqual(bare[:2], expected)

    def test_the_example_built_from_arguments_without_the_settings_takes_the_defaults(self):
        pool = SimpleNamespace(
            observe=lambda: [{"diagnostics": {"physical_time_s": 0.0}}],
            geometry=lambda: {},
            finish_startup=lambda: None,
            close=lambda: None,
        )
        made = []

        def scene(config):
            made.append(config)
            return pool

        args = SimpleNamespace(
            device="cpu", num_envs=1, substeps=4, controller="hold", rigid_backend="mujoco-warp", viewer="null",
            speed=0.003, settle_seconds=0.5, spacing=0.45,
        )  # fmt: skip
        with (
            patch.object(grasp_example, "LocalSutureScene", scene),
            # The GPU rigid backend is refused on a CPU device by the scene's configuration, not by the example.
            patch.object(grasp_example, "SutureSceneConfig", lambda **settings: SimpleNamespace(**settings)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            example = grasp_example.Example(_Window(), args)
            example.close()
        self.assertEqual(example.frame_budget, 1.0e-3 * grasp_example.FRAME_BUDGET_MS)
        self.assertEqual(example.frame_unit, "host-trial")
        self.assertEqual(made[0].example_arguments[-2:], ("--tool-tissue-contact", "swept"))


class PrintedLinesTests(unittest.TestCase):
    """What a run says about its budget: one line where it is in use, nothing where nobody asked."""

    def setUp(self):
        self.parser = grasp_example.Example.create_parser()

    def test_a_run_that_asked_for_nothing_and_keeps_whole_frames_prints_nothing(self):
        for arguments, viewer in (
            (["--test"], _Window()),
            (["--test", "--viewer", "null"], _null()),
            (["--viewer", "null"], _null()),
            (["--headless"], _Window(headless=True)),
            (["--num-envs", "2"], _Window()),
            (["--viewer", "null", "--rigid-backend", "native"], _null()),
            # The defaults written out are the defaults.
            (["--viewer", "null", *WRITTEN_OUT], _null()),
            (["--test", *WRITTEN_OUT], _Window()),
            # Switched off: nothing to say either.
            (list(SWITCHED_OFF), _Window()),
            (["--frame-budget", "0"], _Window()),
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(_plan(viewer, self.parser.parse_args(arguments)), (None, "attempt", []))

    def test_the_window_says_in_one_line_that_the_budget_is_in_use_and_how_to_turn_it_off(self):
        budget, unit, lines = _plan(_Window(), self.parser.parse_args([]))
        self.assertAlmostEqual(budget, 0.0667, places=12)
        self.assertEqual(unit, "host-trial")
        named = [line for line in lines if line.startswith("Frame budget 66.7 ms:")]
        self.assertEqual(len(named), 1, lines)
        self.assertIn("--frame-budget 0 runs whole frames", named[0])
        # One more line names the unit; nothing says "not used".
        self.assertEqual(len(lines), 2, lines)
        self.assertTrue(lines[1].startswith("Frame budget unit host-trial:"), lines)
        self.assertFalse([line for line in lines if "not used" in line])

    def test_a_unit_that_cannot_be_used_under_a_budget_in_use_names_its_reason(self):
        for arguments, reason in ((["--rigid-backend", "native"], "native rigid backend"),):
            with self.subTest(arguments=arguments):
                budget, unit, lines = _plan(_Window(), self.parser.parse_args(arguments))
                self.assertAlmostEqual(budget, 0.0667, places=12)
                self.assertEqual(unit, "attempt")
                self.assertEqual(len(lines), 2, lines)
                self.assertIn("--frame-budget 0 runs whole frames", lines[0])
                self.assertIn("unit host-trial not used, the unit stays the attempt", lines[1])
                self.assertIn(reason, lines[1])

    def test_a_budget_other_than_the_default_says_why_it_is_not_used(self):
        for arguments, viewer, reason in (
            (["--frame-budget", "50", "--test"], _Window(), "the self-test runs whole frames"),
            (["--frame-budget", "50", "--viewer", "null"], _null(), "ends the run after a number of calls"),
            (["--frame-budget", "50", "--num-envs", "2"], _Window(), "several scenes"),
        ):
            with self.subTest(arguments=arguments):
                budget, unit, lines = _plan(viewer, self.parser.parse_args(arguments))
                self.assertEqual((budget, unit), (None, "attempt"))
                # One line, about the budget; the unit, which has no budget to be the unit of, adds none.
                self.assertEqual(len(lines), 1, lines)
                self.assertTrue(lines[0].startswith("Frame budget not used:"), lines)
                self.assertIn(reason, lines[0])
                self.assertNotIn("host-trial", lines[0])


if __name__ == "__main__":
    unittest.main()
