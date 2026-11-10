# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Separate task outcome stability from force-history convergence."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.validate_needle_insertion import build_report, main


def measured_case(error_x=0.0001):
    return {
        "success": True,
        "status": "emerged",
        "physical_time_s": 4.0,
        "exit_error_m": abs(error_x),
        "maximum_interface_force_N": 0.003,
        "maximum_deformation_m": 0.00001,
        "maximum_force_balance_error_N": 1.0e-10,
        "maximum_torque_balance_error_Nm": 1.0e-12,
        "dt_s": 0.0005,
        "trace": [{"status": "emerged", "tip_m": [error_x, 0, 0], "exit_m": [0, 0, 0]}],
    }


class TestNeedleInsertionValidationReport(unittest.TestCase):
    def test_stable_task_does_not_imply_force_history_convergence(self):
        report = build_report({"base": measured_case(), "timestep": measured_case()})
        self.assertTrue(all(report["task_checks"].values()))
        self.assertEqual(report["checks"], report["task_checks"])
        self.assertFalse(report["force_law_assessment"]["convergence_established"])

    def test_different_scenario_is_not_compared_as_a_baseline_refinement(self):
        gap = measured_case(error_x=0.0008)
        gap["maximum_interface_force_N"] = 0.03
        report = build_report({"base": measured_case(), "gap": gap})
        self.assertTrue(report["task_checks"]["all_reached_exit"])
        self.assertNotIn("gap_exit_change_under_0.2mm", report["task_checks"])
        self.assertNotIn("gap", report["numerical_observations"]["refinements"])
        gap["success"] = False
        self.assertFalse(build_report({"base": measured_case(), "gap": gap})["task_checks"]["all_reached_exit"])

    def test_gap_scenario_requires_actual_multiple_visits_and_sections(self):
        gap = measured_case()
        counters = ("entry_count", "exit_count", "puncture_count", "track_section_count")
        gap.update(dict.fromkeys(counters, 1))
        report = build_report({"gap": gap})
        self.assertTrue(report["task_checks"]["all_reached_exit"])
        self.assertTrue(all(not report["task_checks"][f"gap_{name}_at_least_two"] for name in counters))
        gap.update(dict.fromkeys(counters, 2))
        self.assertTrue(all(build_report({"gap": gap})["task_checks"].values()))

    def test_equal_scalar_errors_do_not_hide_opposite_exit_directions(self):
        report = build_report({"base": measured_case(0.00015), "spacing": measured_case(-0.00015)})
        self.assertTrue(report["task_checks"]["spacing_exit_change_under_0.2mm"])
        self.assertFalse(report["task_checks"]["spacing_exit_vector_change_under_0.2mm"])
        self.assertAlmostEqual(
            report["numerical_observations"]["refinements"]["spacing"]["exit_error_vector_change_m"], 0.0003
        )

    def test_controller_success_is_independent_of_physical_outside_status(self):
        cases = {name: measured_case() for name in ("base", "timestep")}
        for case in cases.values():
            case["status"] = "outside"
            case["task_status"] = "success"
            case["trace"][-1]["status"] = "outside"
        report = build_report(cases)
        self.assertTrue(all(report["task_checks"].values()))
        self.assertEqual(report["numerical_observations"]["refinements"]["timestep"]["emergence_time_change_s"], 0)
        self.assertAlmostEqual(
            report["numerical_observations"]["cases"]["base"]["final_exit_error_vector_m"][0], 0.0001
        )

    def test_missing_force_evidence_is_not_marked_as_a_pass(self):
        base, refined = measured_case(), measured_case()
        for result in (base, refined):
            del result["maximum_force_balance_error_N"]
        report = build_report({"base": base, "timestep": refined})
        self.assertIsNone(
            report["numerical_observations"]["refinements"]["timestep"]["post_entry_force_trace_comparison"]
        )
        self.assertIn(None, report["coupling_balance_checks"].values())

    def test_cli_preserves_gap_report_and_fails_on_failed_or_missing_checks(self):
        for failure in (None, "task", "balance", "missing_balance"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                gap = measured_case()
                gap.update(dict.fromkeys(("entry_count", "exit_count", "puncture_count", "track_section_count"), 2))
                if failure == "task":
                    gap["success"] = False
                elif failure == "balance":
                    gap["maximum_force_balance_error_N"] = 2.0e-8
                elif failure == "missing_balance":
                    del gap["maximum_force_balance_error_N"]
                output = Path(directory) / "report.json"
                with (
                    patch("sys.argv", ["validate_needle_insertion.py", "--case", "gap", "--output", str(output)]),
                    patch("tools.validate_needle_insertion.run_case", return_value=gap),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    if failure is None:
                        main()
                    else:
                        with self.assertRaises(SystemExit) as raised:
                            main()
                        self.assertNotEqual(raised.exception.code, 0)
                saved = json.loads(output.read_text())
                self.assertEqual(saved["cases"], {"gap": gap})
                self.assertEqual(saved, build_report({"gap": gap}))

    def test_force_trace_reports_vector_disagreement_despite_equal_magnitudes(self):
        cases = {name: measured_case() for name in ("base", "timestep")}
        for name, sign in (("base", 1), ("timestep", -1)):
            cases[name]["trace"] = [
                {"status": "inserted", "time_s": 1.0, "force_N": [sign * 0.001, 0, 0]},
                {"status": "emerged", "time_s": 3.0, "force_N": [sign * 0.001, 0, 0]},
            ]
        report = build_report(cases)
        comparison = report["numerical_observations"]["refinements"]["timestep"]["post_entry_force_trace_comparison"]
        self.assertAlmostEqual(comparison["vector_difference_rms_N"], 0.002)
        self.assertAlmostEqual(comparison["vector_difference_relative_rms"], 2.0)
        self.assertEqual(comparison["shared_time_interval_s"], [1.0, 3.0])

    def test_force_trace_does_not_shift_event_times_to_hide_disagreement(self):
        cases = {name: measured_case() for name in ("base", "timestep")}
        for name, shift in (("base", 0.0), ("timestep", 1.0)):
            cases[name]["trace"] = [
                {"status": "inserted", "time_s": 1.0 + shift, "force_N": [0, 0, 0]},
                {"status": "emerged", "time_s": 3.0 + shift, "force_N": [0.002, 0, 0]},
            ]
        report = build_report(cases)
        comparison = report["numerical_observations"]["refinements"]["timestep"]["post_entry_force_trace_comparison"]
        self.assertAlmostEqual(comparison["vector_difference_rms_N"], 0.001)
        self.assertEqual(comparison["shared_time_interval_s"], [2.0, 3.0])

    def test_force_trace_excludes_rejected_trial_diagnostics_at_accepted_state_time(self):
        cases = {name: measured_case() for name in ("base", "timestep")}
        for case in cases.values():
            case["trace"] = [
                {
                    "status": "inserted",
                    "time_s": time,
                    "force_N": [0.001, 0, 0],
                    "force_evaluation": "accepted_endpoint",
                }
                for time in (1.0, 2.0, 3.0)
            ]
        cases["timestep"]["success"] = False
        rejected = cases["timestep"]["trace"][-1]
        rejected.update(force_N=[100, 0, 0], force_evaluation="rejected_coupling_trial")
        comparison = build_report(cases)["numerical_observations"]["refinements"]["timestep"][
            "post_entry_force_trace_comparison"
        ]
        self.assertEqual(comparison["refined_saved_sample_count"], 2)
        self.assertEqual(comparison["shared_time_interval_s"], [1.0, 2.0])
        self.assertEqual(comparison["vector_difference_rms_N"], 0)


if __name__ == "__main__":
    unittest.main()
