"""Check profiling isolation, frame fidelity and asynchronous timing boundaries."""

import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("nsys_runner", Path(__file__).resolve().parents[1] / "nsys" / "runner.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class Trace:
    def __init__(self):
        self.events = []

    @contextmanager
    def annotate(self, label, **kwargs):
        self.events.append(("begin", label))
        try:
            yield
        finally:
            self.events.append(("end", label))


class TestNsightRunner(unittest.TestCase):
    def test_wrappers_preserve_descriptors_return_values_and_restore(self):
        module = types.ModuleType("profiling_fixture")

        class SparseSolve:
            def solve(self, value, *, scale=2):
                return value * scale

            @staticmethod
            def factor(value):
                return value + 1

            @classmethod
            def refactor(cls, value):
                return cls.__name__, value

        SparseSolve.__module__ = module.__name__
        module.SparseSolve = SparseSolve
        original = dict(vars(module.SparseSolve))
        trace = Trace()
        instrument = RUNNER.Instrumentation(module.__name__, trace)
        with patch.dict(sys.modules, {module.__name__: module}):
            instrument.install()
            instrument.install()  # Re-discovery must not add duplicate wrappers.
            self.assertEqual(module.SparseSolve().solve(3, scale=4), 12)
            self.assertEqual(module.SparseSolve.factor(3), 4)
            self.assertEqual(module.SparseSolve.refactor(5), ("SparseSolve", 5))
            self.assertEqual(len(instrument.labels), 3)
            instrument.restore()
        for name in ("solve", "factor", "refactor"):
            self.assertIs(vars(module.SparseSolve)[name], original[name])
        self.assertIn(("begin", "linear_solve:SparseSolve.solve"), trace.events)

    def test_loop_keeps_video_frame_order_and_only_boundary_syncs(self):
        calls = []
        example = types.SimpleNamespace(
            step=lambda: calls.append("step"),
            test_post_step=lambda: calls.append("test"),
            render=lambda: calls.append("render"),
            test_final=lambda: calls.append("final"),
        )
        result = RUNNER.run_frames(
            example,
            types.SimpleNamespace(test=True),
            "demo",
            3,
            {"before_frame": lambda name, obj, frame: calls.append((name, frame))},
            Trace(),
            lambda: calls.append("sync"),
        )
        self.assertEqual(
            calls,
            [
                "sync",
                *(item for frame in range(3) for item in (("demo", frame), "step", "test", "render")),
                "sync",
                "final",
            ],
        )
        self.assertEqual(result["frames_completed"], 3)
        self.assertEqual([row["frame"] for row in result["frame_host_timings"]], [0, 1, 2])
        self.assertGreaterEqual(result["loop_wall_seconds"], 0.0)

    def test_insertion_queries_have_nested_ranges_and_restore_original_calls(self):
        module = types.ModuleType("profiling_fixture.insertion")

        def _surface_crossings(value):
            return value + 1

        class NeedleTissueInteraction:
            def _locate(self, value):
                return module._surface_crossings(value)

            def _visited_sample_mask(self, value):
                return self._locate(value)

            def _add_forces(self, value):
                return self._visited_sample_mask(value)

            def accept(self, value):
                return value

        _surface_crossings.__module__ = module.__name__
        NeedleTissueInteraction.__module__ = module.__name__
        module._surface_crossings = _surface_crossings
        module.NeedleTissueInteraction = NeedleTissueInteraction
        original = dict(vars(NeedleTissueInteraction))
        trace = Trace()
        instrument = RUNNER.Instrumentation("profiling_fixture", trace)
        with patch.dict(sys.modules, {module.__name__: module}):
            instrument.install()
            interaction = NeedleTissueInteraction()
            self.assertEqual(interaction._add_forces(3), 4)
            self.assertEqual(interaction.accept(4), 4)
            instrument.restore()
        self.assertIs(module._surface_crossings, _surface_crossings)
        for name in ("_locate", "_visited_sample_mask", "_add_forces", "accept"):
            self.assertIs(vars(NeedleTissueInteraction)[name], original[name])
        self.assertEqual(
            [label for event, label in trace.events if event == "begin"],
            [
                "forces:NeedleTissueInteraction._add_forces",
                "track_association:NeedleTissueInteraction._visited_sample_mask",
                "geometry_location:NeedleTissueInteraction._locate",
                "geometry_crossings:profiling_fixture.insertion._surface_crossings",
                "acceptance:NeedleTissueInteraction.accept",
            ],
        )

    def test_needle_records_success_and_distinguishes_idle_from_physics_frames(self):
        example = types.SimpleNamespace(running=False, sim_time=0.0, render=lambda: None)

        def before_frame(name, obj, frame):
            if frame == 1:
                obj.running = True

        def step():
            if example.running:
                example.sim_time += 0.01

        example.step = step
        metadata = {"success": True, "outcome": "Success", "entry_count": 2, "exit_count": 2}
        result = RUNNER.run_frames(
            example,
            types.SimpleNamespace(test=False),
            "fem_needle_insertion",
            3,
            {"before_frame": before_frame, "recording_metadata": lambda *_: metadata},
            Trace(),
            lambda: None,
        )
        self.assertIs(result["scenario"], metadata)
        rows = result["frame_host_timings"]
        self.assertEqual([row["physics_running"] for row in rows], [False, True, True])
        self.assertEqual([row["simulation_time_start_s"] for row in rows], [0, 0, 0.01])
        self.assertEqual([row["simulation_time_end_s"] for row in rows], [0, 0.01, 0.02])

    def test_needle_failure_cannot_be_accepted_as_completed_idle_frames(self):
        example = types.SimpleNamespace(running=False, sim_time=0.5, step=lambda: None, render=lambda: None)
        metadata = {"success": False, "outcome": "Coupling did not converge", "physical_time_s": 0.5}
        result = {}
        with self.assertRaisesRegex(RuntimeError, "Needle profiling scenario did not succeed: Coupling"):
            RUNNER.run_frames(
                example,
                types.SimpleNamespace(test=False),
                "fem_needle_insertion",
                3,
                {"recording_metadata": lambda *_: metadata},
                Trace(),
                lambda: None,
                result=result,
            )
        self.assertEqual(result["frames_completed"], 3)
        self.assertIs(result["scenario"], metadata)
        self.assertGreaterEqual(result["loop_wall_seconds"], 0)

    def test_metadata_is_not_required_during_warmup_or_for_other_examples(self):
        example = types.SimpleNamespace(running=False, sim_time=0.0, step=lambda: None, render=lambda: None)
        for name, final_test in (("fem_needle_insertion", False), ("fem_dvrk", True)):
            with self.subTest(name=name):
                result = RUNNER.run_frames(
                    example,
                    types.SimpleNamespace(test=False),
                    name,
                    1,
                    {"recording_metadata": lambda *_: self.fail("Recorder-only metadata hook must not run")},
                    Trace(),
                    lambda: None,
                    final_test=final_test,
                )
                self.assertNotIn("scenario", result)

    def test_partial_progress_survives_failure(self):
        count = 0

        def step():
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("simulation failed")

        result = {}
        with self.assertRaisesRegex(RuntimeError, "simulation failed"):
            RUNNER.run_frames(
                types.SimpleNamespace(step=step, render=lambda: None),
                types.SimpleNamespace(test=False),
                "demo",
                5,
                {},
                Trace(),
                lambda: None,
                result=result,
            )
        self.assertEqual(result["frames_completed"], 1)
        self.assertEqual(len(result["frame_host_timings"]), 1)

    def test_warmup_does_not_run_final_horizon_assertions(self):
        example = types.SimpleNamespace(
            step=lambda: None,
            render=lambda: None,
            test_final=lambda: self.fail("Final checks belong to the full measured workload"),
        )
        RUNNER.run_frames(
            example,
            types.SimpleNamespace(test=True),
            "demo",
            1,
            {},
            Trace(),
            lambda: None,
            final_test=False,
        )

    def test_terminal_runs_the_optimizer_once_without_video_playback(self):
        original_argv = sys.argv
        calls = []
        with patch.object(
            RUNNER.runpy,
            "run_module",
            side_effect=lambda *args, **kwargs: calls.append((args, kwargs, list(sys.argv))),
        ):
            result = RUNNER.run_terminal(
                "demo.optimize",
                ["--iterations", "20"],
                Trace(),
                lambda: calls.append("sync"),
            )
        self.assertEqual(calls[0], "sync")
        self.assertEqual(
            calls[1],
            (
                ("demo.optimize",),
                {"run_name": "__main__"},
                ["demo.optimize", "--iterations", "20"],
            ),
        )
        self.assertEqual(calls[2], "sync")
        self.assertIs(sys.argv, original_argv)
        self.assertIsNone(result["frames_completed"])


if __name__ == "__main__":
    unittest.main()
