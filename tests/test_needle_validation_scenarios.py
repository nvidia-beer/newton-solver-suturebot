# SPDX-License-Identifier: Apache-2.0
"""Repeatable scenario selection and failure evidence without a full-pad run."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import validate_needle_insertion as validation


class NeedleValidationScenarioTests(unittest.TestCase):
    def test_forwarded_parameters_define_baseline_then_refinement_is_applied(self):
        arguments = ["--substeps", "4", "--track-spacing", "0.0004", "--shaft-spacing", "0.0002"]
        base = validation.case_arguments("base", arguments)
        time = validation.case_arguments("timestep", arguments)
        space = validation.case_arguments("spacing", arguments)
        self.assertEqual((base.substeps, time.substeps, space.substeps), (4, 8, 4))
        self.assertEqual(time.track_spacing, base.track_spacing)
        self.assertEqual(space.track_spacing, base.track_spacing / 2)
        self.assertEqual(space.shaft_spacing, base.shaft_spacing / 2)
        self.assertEqual(validation.case_arguments("base").substeps, 32)
        self.assertEqual(validation.case_arguments("timestep").substeps, 64)

    def test_integrity_failure_retains_last_trace_and_does_not_report_success(self):
        parser = validation.Example.create_parser

        class InvalidExample:
            create_parser = staticmethod(parser)

            def __init__(self, *_):
                self.running = True
                self.interaction = SimpleNamespace(status="outside")

            def step(self):
                self.running = False

            def report(self):
                return {"success": True, "outcome": "Reached exit", "physical_time_s": 0.1}

            def test_final(self):
                raise AssertionError("Regional FEM inverted an element")

        saved = {"time_s": 0.1, "tip_m": [1, 2, 3], "status": "outside"}
        with (
            patch.object(validation, "Example", InvalidExample),
            patch.object(validation, "_snapshot", return_value=saved),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = validation.run_case("base", device="cpu")
        self.assertFalse(result["success"])
        self.assertIn("inverted", result["validation_error"])
        self.assertEqual(result["trace"], [saved])
        self.assertFalse(validation.build_report({"base": result})["checks"]["all_reached_exit"])

    def test_constructor_failure_is_a_saved_failed_case_in_both_suites(self):
        with patch.object(validation.Example, "__init__", side_effect=ValueError("Invalid selected surface")):
            base = validation.run_case("base", device="cpu")
            robust = validation.run_robustness_case("region2", 4, device="cpu")
        self.assertFalse(base["success"])
        self.assertFalse(robust["completed"])
        self.assertEqual(robust["selection"]["regions"], [2])
        self.assertIn("Invalid selected surface", robust["validation_error"])
        report = validation.build_report({"base": base, "timestep": base})
        self.assertFalse(all(report["checks"].values()))

    def test_each_frame_is_checked_and_unscheduled_failure_snapshot_is_preserved(self):
        parser = validation.Example.create_parser
        frames = []

        class InvalidOnSecondFrame:
            create_parser = staticmethod(parser)

            def __init__(self, *_):
                self.running = True
                self.interaction = SimpleNamespace(status="outside")

            def step(self):
                frames.append(len(frames))
                self.running = len(frames) < 3

            def report(self):
                return {"success": False, "physical_time_s": len(frames) / 60}

            def test_final(self):
                if len(frames) == 2:
                    raise AssertionError("Intermediate frame inverted")

        with (
            patch.object(validation, "Example", InvalidOnSecondFrame),
            patch.object(validation, "_snapshot", side_effect=lambda _ex, frame, _phase: {"frame": frame}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = validation.run_case("base", device="cpu")
        self.assertEqual(frames, [0, 1])
        self.assertEqual(result["trace"], [{"frame": 0}, {"frame": 1}])
        self.assertIn("Intermediate frame", result["validation_error"])

    def test_cli_selects_exact_counts_preserves_failed_cases_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            calls = []

            def run(scenario, count, **kwargs):
                if calls:
                    partial = json.loads(output.read_text())
                    self.assertFalse(partial["checks"]["all_cases_recorded"])
                    self.assertIn("gap-h4", partial["cases"])
                calls.append((scenario, count, kwargs["arguments"]))
                passed = len(calls) != 1
                return {"completed": passed, "checks": {"completed": passed}, "trace": [{"time_s": 0.1}]}

            argv = [
                "validate_needle_insertion.py",
                "--suite",
                "robustness",
                "--scenario",
                "gap",
                "region2",
                "--substep-counts",
                "4",
                "8",
                "--device",
                "cpu",
                "--output",
                str(output),
                "--",
                "--tunnel-model",
                "alm",
            ]
            with (
                patch("sys.argv", argv),
                patch.object(validation, "run_robustness_case", side_effect=run),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                validation.main()
            self.assertEqual(
                [(name, count) for name, count, _ in calls], [("gap", 4), ("gap", 8), ("region2", 4), ("region2", 8)]
            )
            self.assertTrue(all(arguments == ["--tunnel-model", "alm"] for _, _, arguments in calls))
            report = json.loads(output.read_text())
            self.assertEqual(len(report["cases"]), 4)
            self.assertFalse(report["cases"]["gap-h4"]["completed"])
            self.assertTrue(report["checks"]["all_cases_recorded"])
            self.assertFalse(report["force_law_assessment"]["convergence_established"])

    def test_robustness_rejects_forwarded_geometry_and_count_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            for argument in ("--substeps=4", "--regions=2", "--entry-xy", "--report"):
                with (
                    self.subTest(argument=argument),
                    patch(
                        "sys.argv",
                        [
                            "validate_needle_insertion.py",
                            "--suite",
                            "robustness",
                            "--output",
                            str(Path(directory) / "report.json"),
                            "--",
                            argument,
                        ],
                    ),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    validation.main()

    def test_scheduled_actions_cover_signed_direct_jumps_and_arc_reversal(self):
        for name, height in (("abrupt-up", 0.002), ("abrupt-down", -0.002)):
            phases = validation.handling_phases(name, 0.006)
            self.assertEqual(phases[0][2], height)
            self.assertEqual(phases[1][2], 0)
            self.assertTrue(all(item[-1] for item in phases))
        for name, height in (("lift", 0.001), ("lower", -0.001)):
            phases = validation.handling_phases(name, 0.006)
            self.assertEqual(phases[0][2], height)
            self.assertFalse(any(item[-1] for item in phases))
            by_name = {item[0]: item for item in phases}
            self.assertEqual(by_name["withdraw"][3], 0)
            self.assertEqual(by_name["reenter"][3], 0.006)
            self.assertIn("outside_hold", by_name)


if __name__ == "__main__":
    unittest.main()
