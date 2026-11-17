# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in default row, closing pull and settled drop; ordinary tests verify the audit.

Run from the repository root, without changing any example/solver arguments::

    NEWTON_TEST_FULL_SUTURE=1 NEWTON_TEST_DEVICE=cuda:0 \
      NEWTON_FULL_SUTURE_REPORT=nsys/results/full_suture.json \
      uv run --locked python -m unittest discover -s tests -p test_suture_full_sequence.py -v

The report is optional, must not already exist, and is updated atomically every
60 frames. A killed or capped run retains ``status=running`` or ``failed``;
neither is completion evidence. NEWTON_FULL_SUTURE_MAX_FRAMES (default 12000)
only limits the test's duration, never changes the simulated workload.

The example is built as its entry point builds it, with its start-up call, and
the test ends with the check of that start-up: on a CUDA device no graph was
captured and no kernel module was loaded in any frame of the sequence.
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _signed_six_volumes(positions, tetrahedra):
    vertices = np.asarray(positions, dtype=np.float64)[tetrahedra]
    edges = vertices[:, 1:] - vertices[:, :1]
    return np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))


class _TissueAudit:
    """Independent host checks, including fixed velocities and signed volume ratios."""

    def __init__(self, rest, tetrahedra, fixed, initial_positions, initial_velocities):
        self.tetrahedra = np.asarray(tetrahedra, dtype=int)
        self.rest_volumes = _signed_six_volumes(rest, self.tetrahedra)
        if not np.isfinite(self.rest_volumes).all() or np.any(self.rest_volumes == 0):
            raise AssertionError("Invalid reference tetrahedron")
        self.fixed = np.asarray(fixed, dtype=int)
        self.fixed_positions = np.asarray(initial_positions)[self.fixed].copy()
        self.fixed_velocities = np.asarray(initial_velocities)[self.fixed].copy()

    def check(self, positions, velocities):
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise AssertionError("Nonfinite tissue state")
        if not np.array_equal(positions[self.fixed], self.fixed_positions):
            raise AssertionError("Fixed tissue positions changed")
        if not np.array_equal(velocities[self.fixed], self.fixed_velocities):
            raise AssertionError("Fixed tissue velocities changed")
        minimum_j = float((_signed_six_volumes(positions, self.tetrahedra) / self.rest_volumes).min())
        if not np.isfinite(minimum_j) or minimum_j <= 0:
            raise AssertionError(f"Nonpositive accepted tetrahedron volume ratio: J={minimum_j:.12g}")
        return minimum_j


def _assert_completion(facts, rest_speed, rest_time):
    """Do not trust task_status: the example historically calls drop failures success."""
    if facts["failure_calls"]:
        raise AssertionError(f"The example called _fail: {facts['failure_calls']}")
    if facts["running"] or facts["phase"] != "done" or facts["task_status"] != "success":
        raise AssertionError("The default sequence did not complete")
    if set(facts["phases_seen"]) != {"stitch", "pull", "drop", "done"}:
        raise AssertionError("The sequence did not traverse the row, closing pull and drop")
    if facts["stitches_seen"] != list(range(facts["stitch_count"])):
        raise AssertionError("The sequence did not visit every stitch")
    if facts["threaded_per_stitch"] != facts["planned_eyelets_per_stitch"]:
        raise AssertionError("The thread did not pass through every planned track point")
    if facts["elapsed_s"] < facts["schedule_duration_s"]:
        raise AssertionError("The needle schedule did not finish")
    if not facts["closing_reason"] or not (0 <= facts["wound_gap_m"] < facts["rest_gap_m"]):
        raise AssertionError("The default wound did not narrow before the drop")
    if (
        facts["drop_started_s"] is None
        or facts["still_duration_s"] < rest_time
        or not np.isfinite(facts["needle_speed_m_s"])
        or facts["needle_speed_m_s"] >= rest_speed
    ):
        raise AssertionError("The dropped needle did not settle; a drop time limit is not successful rest")


def _record_pull_transition(example, phases, transitions):
    """Observe the successful method return: pull can end before the same frame ends."""
    original_begin_pull = example._begin_pull

    def begin_pull(pose):
        result = original_begin_pull(pose)
        phases.add(example.phase)
        transitions.append({"method": "_begin_pull", "phase": example.phase, "time_s": example.elapsed})
        return result

    example._begin_pull = begin_pull


def _write_report(path, report):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, indent=2, allow_nan=False, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        self.velocity = np.zeros_like(self.rest)
        self.audit = _TissueAudit(self.rest, [[0, 1, 2, 3]], [0], self.rest, self.velocity)

    def test_inverted_and_collapsed_accepted_states_are_rejected(self):
        for height in (-0.00347857, 0):
            with self.subTest(height=height):
                positions = self.rest.copy()
                positions[3, 2] = height
                with self.assertRaisesRegex(AssertionError, "Nonpositive accepted"):
                    self.audit.check(positions, self.velocity)
        positions[3, 2] = 0.01
        self.assertAlmostEqual(self.audit.check(positions, self.velocity), 0.01)

    def test_fixed_position_velocity_and_nonfinite_tissue_states_are_rejected(self):
        for name, value in (("positions", 0.001), ("velocities", 0.001), ("velocities", np.nan)):
            with self.subTest(name=name, value=value):
                positions, velocities = self.rest.copy(), self.velocity.copy()
                {"positions": positions, "velocities": velocities}[name][0, 0] = value
                with self.assertRaises(AssertionError):
                    self.audit.check(positions, velocities)

    @staticmethod
    def completed_facts():
        return {
            "failure_calls": [],
            "running": False,
            "phase": "done",
            "task_status": "success",
            "phases_seen": ["stitch", "pull", "drop", "done"],
            "stitches_seen": [0, 1, 2],
            "stitch_count": 3,
            "threaded_per_stitch": [4, 4, 4],
            "planned_eyelets_per_stitch": [4, 4, 4],
            "elapsed_s": 50.0,
            "schedule_duration_s": 40.0,
            "closing_reason": "the thread is as tight as the tissue allows",
            "wound_gap_m": 0.001,
            "rest_gap_m": 0.002,
            "drop_started_s": 49.0,
            "still_duration_s": 0.5,
            "needle_speed_m_s": 0.0009,
        }

    def test_drop_failure_disguised_as_success_cannot_pass(self):
        facts = self.completed_facts()
        _assert_completion(facts, 0.001, 0.5)
        facts["failure_calls"] = [{"phase": "drop", "reason": "inverted tissue"}]
        with self.assertRaisesRegex(AssertionError, "called _fail"):
            _assert_completion(facts, 0.001, 0.5)

    def test_time_limit_and_partial_row_cannot_pass(self):
        for field, value in (
            ("still_duration_s", 0.0),
            ("needle_speed_m_s", 0.002),
            ("phases_seen", ["stitch", "done"]),
            ("stitches_seen", [0, 1]),
            ("threaded_per_stitch", [4, 4, 3]),
            ("wound_gap_m", 0.002),
        ):
            with self.subTest(field=field):
                facts = self.completed_facts()
                facts[field] = value
                with self.assertRaises(AssertionError):
                    _assert_completion(facts, 0.001, 0.5)

    def test_successful_pull_transition_survives_immediate_same_frame_drop(self):
        phases, transitions = {"stitch"}, []
        example = SimpleNamespace(phase="stitch", elapsed=40.0)
        pose = object()

        def begin_pull(received_pose):
            self.assertIs(received_pose, pose)
            example.phase = "pull"
            return "original return value"

        example._begin_pull = begin_pull
        _record_pull_transition(example, phases, transitions)
        self.assertEqual(example._begin_pull(pose), "original return value")
        example.phase = "drop"  # _observe_closing can end the pull before the next frame sample.
        phases.add(example.phase)
        example.phase = "done"
        phases.add(example.phase)
        facts = self.completed_facts()
        facts["phases_seen"] = sorted(phases)
        _assert_completion(facts, 0.001, 0.5)
        self.assertEqual(transitions, [{"method": "_begin_pull", "phase": "pull", "time_s": 40.0}])

    def test_failed_pull_initialization_does_not_record_successful_transition(self):
        phases, transitions = {"stitch"}, []
        example = SimpleNamespace(phase="stitch", elapsed=40.0)

        def begin_pull(_pose):
            example.phase = "pull"
            raise RuntimeError("pull initialization failed")

        example._begin_pull = begin_pull
        _record_pull_transition(example, phases, transitions)
        with self.assertRaisesRegex(RuntimeError, "pull initialization failed"):
            example._begin_pull(None)
        self.assertEqual(phases, {"stitch"})
        self.assertEqual(transitions, [])


@unittest.skipUnless(os.environ.get("NEWTON_TEST_FULL_SUTURE") == "1", "Opt-in full default physical suturing sequence")
class FullDefaultSequenceTests(unittest.TestCase):
    def test_every_default_stitch_closing_pull_and_settled_drop(self):
        import warp as wp
        from newton.viewer import ViewerNull

        from newton_solver_suturebot.examples.needle import example_fem_needle_suture as suture

        device = os.environ.get("NEWTON_TEST_DEVICE", "cuda:0")
        if device.startswith("cuda") and not wp.is_cuda_available():
            self.skipTest(f"Requested CUDA device {device} is unavailable")
        report_path = os.environ.get("NEWTON_FULL_SUTURE_REPORT")
        report_path = Path(report_path) if report_path else None
        if report_path is not None and report_path.exists():
            self.fail(f"Refusing to overwrite an existing report: {report_path}")
        maximum_frames = int(os.environ.get("NEWTON_FULL_SUTURE_MAX_FRAMES", "12000"))
        self.assertGreater(maximum_frames, 0)
        self.enterContext(wp.ScopedDevice(device))
        viewer = ViewerNull()
        self.addCleanup(viewer.close)
        args = suture.Example.create_parser().parse_args([])
        report = {
            "schema_version": 1,
            "status": "running",
            "device": device,
            "example_arguments": vars(args),
            "maximum_frames": maximum_frames,
            "frames_checked": 0,
            "minimum_frame_tet_j": None,
            "failure_calls": [],
            "phase_transitions": [],
            "checkpoints": [],
        }
        started = time.monotonic()
        example = None
        try:
            _write_report(report_path, report)
            example = suture.Example(viewer, args, startup=True)
            self.assertEqual(len(example.stitches), 3, "This regression covers the complete default three-stitch row")
            report.update(
                actual_device=str(example.device),
                device_name=wp.get_device(example.device).name,
                cuda=bool(wp.get_device(example.device).is_cuda),
                warp_version=wp.__version__,
                numpy_version=np.__version__,
                dimensions={
                    "tissue_vertices": len(example.selection.vertices),
                    "tetrahedra": len(example.selection.indices),
                    "fixed_vertices": len(example.fixed_vertex_indices),
                    "thread_nodes": int(example.thread.q.size // 6),
                    "eyelets": int(example.thread.max_eyelets),
                    "stitches": len(example.stitches),
                },
                frame_dt_s=example.frame_dt,
                nominal_substep_dt_s=example.sim_dt,
                initial_gap_m=example.rest_gap,
                maximum_observed_closing_gap_m=example.wound_gap,
                minimum_observed_closing_gap_m=example.wound_gap,
            )
            audit = _TissueAudit(
                example.selection.vertices,
                example.selection.indices,
                example.fixed_vertex_indices,
                example.state_0.particle_q.numpy(),
                example.state_0.particle_qd.numpy(),
            )
            report["minimum_frame_tet_j"] = audit.check(
                example.state_0.particle_q.numpy(), example.state_0.particle_qd.numpy()
            )
            original_fail = example._fail

            def record_failure(reason):
                report["failure_calls"].append(
                    {"phase": example.phase, "time_s": example.elapsed, "reason": str(reason)}
                )
                original_fail(reason)

            example._fail = record_failure
            phases, stitches = {example.phase}, {example.stitch_index}
            _record_pull_transition(example, phases, report["phase_transitions"])
            previous_rejections = 0
            for frame in range(1, maximum_frames + 1):
                previous_time = example.elapsed
                example.step()
                report["frames_checked"] = frame
                phases.add(example.phase)
                stitches.add(example.stitch_index)
                minimum_j = audit.check(example.state_0.particle_q.numpy(), example.state_0.particle_qd.numpy())
                report["minimum_frame_tet_j"] = min(report["minimum_frame_tet_j"] or minimum_j, minimum_j)
                for name, array in (
                    ("needle poses", example.state_0.body_q),
                    ("needle velocities", example.state_0.body_qd),
                    ("thread coordinates", example.thread.q),
                    ("thread velocities", example.thread.qd),
                    ("thread eyelet forces", example.thread.eyelet_force),
                    ("tissue external forces", example._external_forces),
                ):
                    self.assertTrue(np.isfinite(array.numpy()).all(), f"Nonfinite {name} at frame {frame}")
                self.assertGreater(example.elapsed, previous_time, f"No accepted progress at frame {frame}")
                swept_j = float(example.coupler.minimum_accepted_tet_j)
                rejections = int(example.coupler.tet_step_rejections)
                self.assertTrue(np.isfinite(swept_j) and swept_j > 0, f"Invalid accepted swept J={swept_j}")
                self.assertGreaterEqual(rejections, previous_rejections)
                previous_rejections = rejections
                report.update(
                    elapsed_s=example.elapsed,
                    phase=example.phase,
                    stitch_index=example.stitch_index,
                    accepted_physics_steps=example.accepted_physics_steps,
                    step_refinements=example.step_refinements,
                    minimum_accepted_swept_tet_j=swept_j,
                    tet_step_rejections=rejections,
                )
                if example.wound_gap is not None:
                    report["maximum_observed_closing_gap_m"] = max(
                        report["maximum_observed_closing_gap_m"], example.wound_gap
                    )
                    report["minimum_observed_closing_gap_m"] = min(
                        report["minimum_observed_closing_gap_m"], example.wound_gap
                    )
                if frame % 60 == 0 or not example.running:
                    report["checkpoints"].append(
                        {
                            "frame": frame,
                            "time_s": example.elapsed,
                            "phase": example.phase,
                            "stitch": example.stitch_index,
                            "minimum_frame_tet_j": minimum_j,
                        }
                    )
                    report["wall_seconds"] = time.monotonic() - started
                    _write_report(report_path, report)
                    print(
                        f"Full suture: frame={frame} time={example.elapsed:.3f}s phase={example.phase} stitch={example.stitch_index + 1}/3 minimum_J={minimum_j:.6g} rejected={rejections}",
                        flush=True,
                    )
                self.assertFalse(report["failure_calls"], report["failure_calls"])
                if not example.running:
                    break
            facts = {
                "failure_calls": report["failure_calls"],
                "running": example.running,
                "phase": example.phase,
                "task_status": example.task_status,
                "phases_seen": sorted(phases),
                "stitches_seen": sorted(stitches),
                "stitch_count": len(example.stitches),
                "threaded_per_stitch": list(example._threaded),
                "planned_eyelets_per_stitch": [len(stitch.angles) for stitch in example.stitches],
                "elapsed_s": example.elapsed,
                "schedule_duration_s": example.schedule.duration,
                "closing_reason": example.closing,
                "commanded_closing_pull_distance_m": example._pulled,
                "wound_gap_m": example.wound_gap,
                "rest_gap_m": example.rest_gap,
                "drop_started_s": example._drop_start,
                "still_duration_s": example._still,
                "needle_speed_m_s": float(np.linalg.norm(example.state_0.body_qd.numpy()[example.needle_body, :3])),
            }
            report["completion"] = facts
            _assert_completion(facts, suture.REST_SPEED, suture.REST_TIME)
            self.assertLessEqual(example.coupler.tet_step_rejections, example.step_refinements)
            example.test_final()
            # Nothing was captured or loaded after the start-up, through every entry, exit, the pull and the drop.
            report["startup"] = example.startup.check()
            report["storage_growths"] = {
                "track": example.interaction.track_storage_growths,
                "opening": example.interaction.opening_storage_growths,
            }
            self.assertEqual(report["storage_growths"], {"track": 0, "opening": 0})
            report["status"] = "passed"
        except BaseException as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            report["wall_seconds"] = time.monotonic() - started
            if example is not None:
                report.update(
                    outcome=example.outcome,
                    final_phase=example.phase,
                    elapsed_s=example.elapsed,
                    final_observed_closing_gap_m=example.wound_gap,
                    gap_measurement_scope="Example wound_gap is refreshed during the closing pull, not the drop",
                )
            _write_report(report_path, report)
            print("FULL_SUTURE_REPORT " + json.dumps(report, allow_nan=False, default=str), flush=True)


if __name__ == "__main__":
    unittest.main()
