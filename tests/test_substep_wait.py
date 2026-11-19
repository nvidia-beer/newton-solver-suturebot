# SPDX-License-Identifier: Apache-2.0
"""Where a call with a deadline waits for the device: inside the substep that issued the work.

The hook of an accepted step (the thread step) is issued and not waited for, and so is the
tangent of a retry. A call with a deadline that returns after them is followed by the caller's
observation, whose first read would wait for that work. The call waits itself, before it takes
the clock for the substep: the time is then in the bound of the next substep and in the tail
of the attempt, and the observation is as short after a call that leaves its frame unfinished
as after one that completes it. Without a deadline nothing is waited for, as before.
"""

import ast
import inspect
import os
import textwrap
import time
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from test_frame_budget import ACTIONS, PASSED, _local, _rejecting
from test_grasp_scene import _scene

from newton_solver_suturebot._src.utils.frame_budget import HostTrialBudget, SubstepBound
from newton_solver_suturebot.examples.grasp.parallel import _apply_action, _step
from newton_solver_suturebot.examples.needle import example_fem_needle_insertion as insertion

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
# What the test lets the device work take [s]: far above every other time outside the trial loops.
WAIT = 0.05


class _Order:
    """The order in which a scene calls what the tests name."""

    def __init__(self, case, scene, wait=0.0):
        self.events, self.wait = [], wait
        owners = (
            (scene, "_accepted_step", "hook"),
            (scene, "_wait_for_device", "wait"),
            (scene, "actual_tip", "tip"),
            (scene.solver, "refresh_tangent", "tangent"),
            (scene, "_attempt_fits", "question"),
        )
        for owner, name, label in owners:
            self._wrap(case, owner, name, label)
        for owner, name, label in ((SubstepBound, "add", "substep"), (HostTrialBudget, "close", "tail")):
            self._wrap(case, owner, name, label)

    def _wrap(self, case, owner, name, label):
        function = getattr(owner, name)

        def recorded(*arguments, **options):
            self.events.append(label)
            if label == "wait" and self.wait:
                # The device work that was issued takes this long.
                time.sleep(self.wait)
            return function(*arguments, **options)

        patcher = patch.object(owner, name, recorded)
        patcher.start()
        case.addCleanup(patcher.stop)

    def take(self):
        events, self.events = self.events, []
        return events


class SubstepWaitTests(unittest.TestCase):
    def enter(self, **settings):
        manager = _scene(**settings)
        scene, config = manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)
        return scene, config

    def test_the_wait_is_a_synchronize_of_the_scenes_device_and_reads_nothing(self):
        scene, _ = self.enter()
        with (
            patch.object(wp, "synchronize_device") as synchronize,
            patch.object(wp.array, "numpy") as read,
            patch.object(wp, "launch") as launch,
            patch.object(wp, "capture_launch") as graph,
        ):
            self.assertIsNone(scene._wait_for_device())
        synchronize.assert_called_once_with(scene.device)
        # No array crosses to the host beside the synchronize, and nothing is issued.
        read.assert_not_called()
        launch.assert_not_called()
        graph.assert_not_called()

    def test_between_two_substeps_the_call_waits_before_it_measures_the_substep(self):
        scene, config = self.enter()
        order = _Order(self, scene)
        _apply_action(scene, config, ACTIONS[0])
        for _ in range(3):
            _step(scene, PASSED)
            self.assertTrue(scene.frame_pending)
            # One accepted attempt: its hook is issued, the call waits, then the substep is measured
            # and the question for the next attempt is asked.
            events = [event for event in order.take() if event != "tangent"]
            self.assertEqual(events, ["hook", "wait", "substep", "question"])
        # The call that completes the frame: the tip sample reads the accepted state, which waits.
        _step(scene, PASSED)
        self.assertFalse(scene.frame_pending)
        self.assertEqual([event for event in order.take() if event != "tangent"], ["hook", "tip", "substep"])

    def test_a_call_without_a_deadline_does_not_wait(self):
        scene, config = self.enter()
        order = _Order(self, scene)
        _apply_action(scene, config, ACTIONS[0])
        _step(scene)
        events = order.take()
        self.assertEqual(events.count("hook"), 4)
        self.assertNotIn("wait", events)
        self.assertNotIn("substep", events)

    def test_a_deadline_far_away_waits_after_every_substep_that_another_one_follows(self):
        scene, config = self.enter()
        order = _Order(self, scene)
        _apply_action(scene, config, ACTIONS[0])
        _step(scene, time.perf_counter() + 3600.0)
        self.assertFalse(scene.frame_pending)
        events = [event for event in order.take() if event != "tangent"]
        # Four substeps in one call: each of the first three ends with the wait, the last with the tip sample.
        self.assertEqual(events, ["hook", "wait", "substep", "question"] * 3 + ["hook", "tip", "substep"])

    def test_the_wait_is_in_the_time_of_the_substep_that_issued_the_work(self):
        scene, config = self.enter()
        order = _Order(self, scene, wait=WAIT)
        _apply_action(scene, config, ACTIONS[0])
        started = time.perf_counter()
        _step(scene, PASSED)
        spent = time.perf_counter() - started
        self.assertEqual(order.take().count("wait"), 1)
        bound = scene._substep_bound
        # The substep's time outside its trial loops holds the wait, and so does the bound of the next one.
        self.assertEqual(bound.substeps, 1)
        self.assertGreaterEqual(bound.longest_outside, WAIT)
        self.assertLessEqual(bound.longest_outside, spent)
        self.assertGreaterEqual(bound.bound(scene._trial_limit()), WAIT + bound.loop_seconds)

    def test_the_wait_is_in_the_tail_of_the_attempt(self):
        scene, config = self.enter()
        order = _Order(self, scene, wait=WAIT)
        _apply_action(scene, config, ACTIONS[0])
        # The unit host-trial: the first call of a run leaves its attempt before the first
        # host-controlled trial, the next one goes on with it.
        calls = 0
        while scene.accepted_physics_steps < 1:
            _step(scene, PASSED, inside=True)
            calls += 1
            self.assertLess(calls, 8)
        events = [event for event in order.take() if event != "tangent"]
        # The wait comes before the tail of the attempt is closed and before the substep is measured.
        self.assertEqual(events[-5:], ["hook", "wait", "tail", "substep", "question"])
        self.assertGreaterEqual(scene._host_budget.longest_tail, WAIT)
        self.assertGreaterEqual(scene._substep_bound.longest_outside, WAIT)

    def test_a_call_that_leaves_an_attempt_open_waits_first(self):
        scene, config = self.enter()
        order = _Order(self, scene, wait=WAIT)
        _apply_action(scene, config, ACTIONS[0])
        # The first host-controlled trial of a run is refused: nothing is measured yet.
        _step(scene, PASSED, inside=True)
        self.assertTrue(scene.frame_pending)
        self.assertIsNotNone(scene._pending_attempt)
        self.assertEqual(scene.accepted_physics_steps, 0)
        self.assertEqual([event for event in order.take() if event != "tangent"], ["wait"])
        # What the call spent on the substep holds the wait.
        self.assertGreaterEqual(scene._pending_frame[3][0], WAIT)

    def test_before_the_question_for_a_retry_the_call_waits_for_the_tangent(self):
        scene, config = self.enter()
        order = _Order(self, scene, wait=WAIT)
        _apply_action(scene, config, ACTIONS[0])
        with _rejecting(scene, (0,)):
            order.take()
            _step(scene, PASSED)
            self.assertEqual(scene.step_refinements, 1)
            # The rejected attempt: no hook; the tangent of the retry, the wait, then the question.
            self.assertEqual(order.take()[-3:], ["tangent", "wait", "question"])
            spent = scene._pending_frame[3]
            # What the call spent on the substep holds the wait.
            self.assertGreaterEqual(spent[0], WAIT)
            _step(scene, PASSED)
        # The substep is measured as one, with the wait of the rejected attempt and its own.
        self.assertEqual(scene._substep_bound.substeps, 1)
        self.assertGreaterEqual(scene._substep_bound.longest_outside, 2 * WAIT)

    def test_the_observation_after_an_unfinished_frame_does_not_hold_the_wait(self):
        action = np.asarray(ACTIONS[0], dtype=float)[None, :]
        with _local() as scenes:
            scene = scenes._scene
            order = _Order(self, scene, wait=WAIT)
            scenes.step(action, deadline=PASSED)
            while scenes.frame_unfinished:
                scenes.step(None, deadline=PASSED)
            # Three calls left the frame unfinished and waited; the one that completed it did not.
            self.assertEqual(order.take().count("wait"), 3)
            # The wait is the substep's (the scene's bound), not the observation's (the driver's reserve).
            self.assertGreaterEqual(scene._substep_bound.longest_outside, WAIT)
            self.assertLess(scenes._observe_longest, WAIT)


class WaitSourceTests(unittest.TestCase):
    def test_only_calls_with_a_deadline_wait(self):
        # The three places of simulate() at which a call can go on or return with its frame
        # unfinished wait, each only in a call with a deadline (``substep`` is the mark of such a call).
        tree = ast.parse(textwrap.dedent(inspect.getsource(insertion.Example.simulate)))
        waits, guarded = [], []

        def visit(node, inside):
            if isinstance(node, ast.If):
                test = ast.unparse(node.test)
                for child in node.body:
                    visit(child, inside or test in ("deadline is not None", "substep is not None"))
                for child in node.orelse:
                    visit(child, inside)
                return
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "self._wait_for_device":
                waits.append(node.lineno)
                if inside:
                    guarded.append(node.lineno)
            for child in ast.iter_child_nodes(node):
                visit(child, inside)

        visit(tree, False)
        self.assertEqual(len(waits), 3)
        self.assertEqual(waits, guarded)
        # ``substep`` is None exactly in a call without a deadline.
        source = inspect.getsource(insertion.Example.simulate)
        self.assertIn("substep = None if deadline is None else self._substep_mark(spent)", source)


if __name__ == "__main__":
    unittest.main()
