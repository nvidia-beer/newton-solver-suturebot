# SPDX-License-Identifier: Apache-2.0
"""Start-up of fem_needle_suture on the bundled pad: the coupled step and its graphs exist when the row is built."""

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull

from newton_solver_suturebot import NeedleTissueCoupler
from newton_solver_suturebot._src.utils import capture
from newton_solver_suturebot.examples.needle import example_fem_needle_suture as suture

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
CUDA = wp.get_device(DEVICE).is_cuda


def _bits(example):
    """The accepted needle and tissue state and the thread, for bit comparisons."""
    state = example.state_0
    result = {name: getattr(state, name).numpy().copy() for name in ("particle_q", "particle_qd", "body_q", "body_qd")}
    result["thread_q"] = example.thread.q.numpy().copy()
    result["thread_qd"] = example.thread.qd.numpy().copy()
    result["external_forces"] = example._external_forces.numpy().copy()
    return result


def _census_line(example):
    """Run ``test_final`` and return the numbers of its start-up check (the summary of the row follows them)."""
    output = io.StringIO()
    with redirect_stdout(output):
        example.test_final()
    return json.loads(next(line for line in output.getvalue().splitlines() if line.startswith("{")))


class SutureStartupTests(unittest.TestCase):
    """The default row, built once as a caller that asks for nothing builds it and once as the entry point does."""

    @classmethod
    def setUpClass(cls):
        if DEVICE.startswith("cuda") and not wp.is_cuda_available():
            raise unittest.SkipTest("NEWTON_TEST_DEVICE asks for CUDA and there is none")
        cls.enterClassContext(wp.ScopedDevice(DEVICE))
        examples = []
        for options in ({}, {"startup": True}):
            viewer = ViewerNull()
            cls.addClassCleanup(viewer.close)
            # --no-autostart: the row waits for Start, as with the button.
            args = suture.Example.create_parser().parse_args(["--test", "--no-autostart"])
            with redirect_stdout(io.StringIO()):
                examples.append(suture.Example(viewer, args, **options))
        cls.plain, cls.started = examples

    def test_1_row_has_its_coupled_step_before_start_only_where_the_start_up_was_asked_for(self):
        plain, started = self.plain, self.started
        self.assertIsNone(plain.startup)
        self.assertIsNone(plain.script)
        self.assertIsNone(plain.interaction)
        self.assertIsNone(plain._startup_tangent)
        self.assertEqual(len(started.stitches), 3)
        self.assertIs(type(started.coupler), NeedleTissueCoupler)
        self.assertFalse(started.running)
        self.assertEqual(started.script.frames, 0)
        self.assertEqual(started._startup_tangent, (started.solver._tangent_version, started.sim_dt, 0))
        self.assertEqual(started._startup_ready, CUDA)
        self.assertEqual(started._startup_status, "start-up call ran" if CUDA else "CPU device")
        # The history reserve is the one of the whole row, as at Start.
        reserve = started._history_reserve()
        self.assertEqual(started.interaction.track_capacity, reserve["track_capacity"])
        self.assertEqual(started.interaction.opening_capacity, reserve["opening_capacity"])
        if CUDA:
            # The thread graph was launched where it was captured; without a start-up it waits for the first frame.
            self.assertIsNotNone(started._thread_graph.graph_exec)
            self.assertIsNone(plain._thread_graph.graph_exec)
            self.assertEqual(started.startup.census()["startup_unlaunched_graphs"], 0)
        for name, values in _bits(plain).items():
            np.testing.assert_array_equal(_bits(started)[name], values, err_msg=f"before Start: {name}")

    def test_2_start_builds_nothing_and_the_first_frames_are_the_same(self):
        plain, started = self.plain, self.started
        coupler = started.coupler
        captures = capture.census.captures, capture.census.solver_captures
        with patch.object(started, "_build_coupling", side_effect=AssertionError("built at Start")):
            started.toggle_running()
        self.assertTrue(started.running)
        self.assertIs(started.coupler, coupler)
        self.assertEqual((capture.census.captures, capture.census.solver_captures), captures)
        # The scene with the start-up call runs first: the counters belong to the process, and the
        # scene without it captures where its frames need a graph.
        frames = {}
        for example in (started, plain):
            if example is plain:
                plain.toggle_running()
                self.assertIs(type(plain.coupler), NeedleTissueCoupler)
            for frame in range(3):
                with redirect_stdout(io.StringIO()):
                    example.step()
                    example.render()
                self.assertTrue(example.running, example.outcome)
                frames[example, frame] = _bits(example)
            if example is started:
                self.assertIsNone(started._startup_tangent)
                line = _census_line(started)
                self.assertEqual(line["capture_census_status"], "checked" if CUDA else "CPU device")
                self.assertEqual(line["scripted_insertions"], [])
                if CUDA:
                    for name in started.startup.numbers:
                        self.assertEqual(line[name], 0, name)
        for frame in range(3):
            for name, values in frames[plain, frame].items():
                actual = frames[started, frame][name]
                if CUDA:
                    # The thread loads the tissue from the first frame on, and the assembly of a
                    # loaded tissue sums in an order that changes from run to run on the GPU.
                    np.testing.assert_allclose(actual, values, rtol=1e-4, atol=1e-6, err_msg=f"{frame}: {name}")
                else:
                    np.testing.assert_array_equal(actual, values, err_msg=f"frame {frame}: {name}")
        self.assertEqual(started.accepted_physics_steps, plain.accepted_physics_steps)

    def test_3_reset_is_a_start_up_and_the_frames_after_it_capture_nothing(self):
        started = self.started
        coupler = started.coupler
        with redirect_stdout(io.StringIO()):
            started.reset()
        self.assertIsNot(started.coupler, coupler)
        self.assertIs(type(started.coupler), NeedleTissueCoupler)
        self.assertFalse(started.running)
        self.assertEqual(started.elapsed, 0)
        self.assertEqual(started._startup_tangent, (started.solver._tangent_version, started.sim_dt, 0))
        self.assertEqual(started._startup_ready, CUDA)
        if CUDA:
            self.assertEqual(started.startup.status, "checked")
            self.assertIsNotNone(started._thread_graph.graph_exec)
        # The counters of the process from here on (the other scene of this class has stopped stepping).
        captures = capture.census.captures, capture.census.solver_captures
        modules = capture.loaded_modules()
        started.toggle_running()
        with redirect_stdout(io.StringIO()):
            for _ in range(3):
                started.step()
                started.render()
        self.assertTrue(started.running, started.outcome)
        if CUDA:
            self.assertEqual((capture.census.captures, capture.census.solver_captures), captures)
            self.assertEqual(capture.loaded_modules(), modules)


if __name__ == "__main__":
    unittest.main()
