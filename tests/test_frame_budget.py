# SPDX-License-Identifier: Apache-2.0
"""A frame under a time budget: split over several calls, it is the frame of one call.

A call with a deadline runs one attempt of a coupled step and starts another one, the next
substep or the retry of a rejected attempt, only if the bound still fits: the worst case that
the run has measured of a substep, of the observation and of the drawing. What is about the
calls (how many, where they return, what they keep, the bound, the deadline, where the option
applies) runs on every device. The comparisons of two runs bit for bit run on the CPU only:
two runs of the same frames on a CUDA device are not bit-identical.
"""

import contextlib
import io
import math
import os
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import newton.viewer
import numpy as np
import warp as wp
from test_grasp_scene import _native, _scene
from test_needle_insertion_example import _REFERENCE, _selection

from newton_solver_suturebot import NeedleStepTooLarge
from newton_solver_suturebot._src.utils.frame_budget import HostTrialBudget, SubstepBound
from newton_solver_suturebot.examples.grasp import example_fem_needle_grasp as grasp_example
from newton_solver_suturebot.examples.grasp import parallel
from newton_solver_suturebot.examples.grasp.parallel import (
    LocalSutureScene,
    SutureSceneConfig,
    _apply_action,
    _observe,
    _reset_scene,
    _step,
)
from newton_solver_suturebot.examples.needle import example_fem_needle_insertion as insertion
from newton_solver_suturebot.examples.phantom import example_fem_phantom_regions as phantom

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
_CUDA = wp.get_device(DEVICE).is_cuda
_BITS = "bit-for-bit comparison of two runs: CPU only"
# Lower the tool into the pad, then move and turn it: free motion, contact, puncture and entry.
ACTIONS = [[0, 0, -1, 0, 0, 0, 0]] * 12 + [[1, 0, -1, 0, 0, 0.5, 0]] * 8
# A deadline that has passed: every call runs exactly one attempt.
PASSED = 0.0
STATE = ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")
# More calls than any frame of these tests takes: a frame that needs more makes no progress.
CALL_LIMIT = 64


class SubstepBoundTests(unittest.TestCase):
    def test_no_bound_before_the_first_measured_substep(self):
        bound = SubstepBound()
        self.assertEqual(bound.bound(13), math.inf)
        self.assertIsNone(bound.per_trial)
        # A stretch of time without a trial is not a substep.
        bound.add(0, 0, 0.0, 1.0)
        self.assertEqual((bound.substeps, bound.trials, bound.last_loops), (0, 0, 1))
        self.assertEqual(bound.bound(13), math.inf)

    def test_bound_is_the_time_outside_the_trials_plus_the_trial_limit_times_the_time_of_a_trial(self):
        bound = SubstepBound()
        bound.add(1, 1, 0.003, 0.0045)
        bound.add(3, 1, 0.006, 0.0075)
        # Four trials took 9 ms in their loops; two substeps took 3 ms outside them.
        self.assertAlmostEqual(bound.per_trial, 0.00225, places=12)
        self.assertAlmostEqual(bound.longest_outside, 0.0015, places=12)
        self.assertAlmostEqual(bound.bound(13), 0.0015 + 13 * 0.00225, places=12)
        self.assertAlmostEqual(bound.bound(160), 0.0015 + 160 * 0.00225, places=12)

    def test_the_time_per_trial_is_a_mean_over_every_substep_measured(self):
        bound = SubstepBound()
        bound.add(2, 1, 0.004, 0.005)
        bound.add(2, 1, 0.008, 0.009)
        self.assertAlmostEqual(bound.per_trial, 0.003, places=12)
        # A substep with a rejected attempt counts the trials and the loop time of both attempts.
        bound.add(8, 2, 0.024, 0.028)
        self.assertEqual((bound.substeps, bound.trials), (3, 12))
        self.assertAlmostEqual(bound.per_trial, 0.003, places=12)

    def test_the_time_outside_the_trials_is_the_longest_one_measured_not_the_mean(self):
        bound = SubstepBound()
        bound.add(2, 1, 0.004, 0.005)
        # The end of this substep took 8 ms longer than that of the others.
        bound.add(2, 1, 0.004, 0.013)
        bound.add(2, 1, 0.004, 0.005)
        # The mean of the three would be 11 ms / 3.
        self.assertAlmostEqual(bound.longest_outside, 0.009, places=12)
        self.assertAlmostEqual(bound.bound(13), 0.009 + 13 * 0.002, places=12)
        # It is the time outside the loops of one substep, whatever loops the next one is taken to run.
        bound.add(4, 2, 0.008, 0.010)
        self.assertAlmostEqual(bound.longest_outside, 0.009, places=12)
        self.assertAlmostEqual(bound.bound(13), 0.009 + 2 * 13 * 0.002, places=12)

    def test_a_loop_is_taken_as_the_longest_loop_measured_and_as_no_less_than_the_trial_limit_of_mean_trials(self):
        bound = SubstepBound()
        for _ in range(9):
            bound.add(1, 1, 0.003, 0.004)
        # Nine loops of one trial: no loop has used up its trials yet, the limit of mean trials stands for one.
        self.assertAlmostEqual(bound.longest_loop, 0.003, places=12)
        self.assertAlmostEqual(bound.bound(13), 0.001 + 13 * 0.003, places=12)
        # A loop of 13 trials that took longer than 13 mean trials: it is the bound of a loop from then on.
        bound.add(13, 1, 0.046, 0.047)
        self.assertAlmostEqual(bound.per_trial, 0.073 / 22, places=12)
        self.assertLess(13 * bound.per_trial, 0.046)
        self.assertAlmostEqual(bound.longest_loop, 0.046, places=12)
        self.assertAlmostEqual(bound.bound(13), 0.001 + 0.046, places=12)
        # With a larger limit the mean trials are the longer of the two again.
        self.assertAlmostEqual(bound.bound(160), 0.001 + 160 * bound.per_trial, places=12)
        # Of a substep with several loops the time of a loop is their mean, as its trials are counted.
        bound.add(14, 2, 0.060, 0.061)
        self.assertAlmostEqual(bound.longest_loop, 0.046, places=12)
        bound.add(14, 2, 0.100, 0.101)
        self.assertAlmostEqual(bound.longest_loop, 0.050, places=12)
        # Two loops are taken for the next substep, each the longer of the two terms.
        self.assertAlmostEqual(bound.per_trial, 0.233 / 50, places=12)
        self.assertAlmostEqual(bound.bound(13), 0.001 + 2 * 13 * bound.per_trial, places=12)
        self.assertAlmostEqual(bound.bound(8), 0.001 + 2 * 0.050, places=12)

    def test_a_maximum_does_not_come_down(self):
        bound = SubstepBound()
        bound.add(13, 1, 0.046, 0.058)
        worst = bound.bound(13)
        self.assertAlmostEqual(worst, 0.012 + 0.046, places=12)
        # A thousand quick substeps after it: the mean trial comes down, the bound of the worst case stays.
        for _ in range(1000):
            bound.add(1, 1, 0.002, 0.003)
        self.assertLess(13 * bound.per_trial, 0.6 * bound.longest_loop)
        self.assertEqual((bound.longest_loop, bound.longest_outside), (0.046, 0.058 - 0.046))
        self.assertAlmostEqual(bound.bound(13), worst, places=12)
        # A new scene, as after a reset, has measured nothing.
        self.assertEqual((SubstepBound().longest_loop, SubstepBound().longest_outside), (0.0, 0.0))

    def test_the_bound_is_never_below_the_longest_measured_nor_below_the_trial_limit_of_mean_trials(self):
        random, bound = np.random.default_rng(7), SubstepBound()
        for _ in range(400):
            loops = int(random.integers(1, 4))
            trials = int(random.integers(loops, 13 * loops + 1))
            loop_seconds = trials * random.uniform(0.002, 0.004) + loops * random.uniform(0.0, 0.006)
            bound.add(trials, loops, loop_seconds, loop_seconds + random.uniform(0.001, 0.02))
            for limit in (13, 160):
                # Loops of the trial limit of mean trials, which stand for a loop not measured yet.
                means = bound.longest_outside + bound.last_loops * (limit * bound.per_trial)
                self.assertGreaterEqual(bound.bound(limit), means)
                # And never below the longest loop and the longest time outside the loops measured.
                worst = bound.longest_outside + bound.last_loops * bound.longest_loop
                self.assertGreaterEqual(bound.bound(limit), worst)
                self.assertAlmostEqual(bound.bound(limit), max(means, worst), places=12)

    def test_one_slow_loop_enters_the_bound_with_its_own_excess_not_with_the_trial_limit_times_it(self):
        bound = SubstepBound()
        for _ in range(99):
            bound.add(1, 1, 0.003, 0.004)
        bound.add(13, 1, 0.039, 0.040)
        before = bound.bound(13)
        self.assertAlmostEqual(before, 0.001 + 0.039, places=12)
        # One loop of a single trial stalls for 6 ms: three times its usual time per trial.
        bound.add(1, 1, 0.009, 0.010)
        # The trial limit times that time per trial would be 117 ms. The loop is shorter than the longest
        # one, and the mean trial moves by the stall's share of the 113 trials measured.
        self.assertAlmostEqual(bound.longest_loop, 0.039, places=12)
        self.assertAlmostEqual(bound.bound(13) - before, 13 * 0.006 / 113, places=12)
        self.assertLess(bound.bound(13) - before, 0.006)
        # The longest loop stalls for 6 ms: the bound grows by those 6 ms.
        before = bound.bound(13)
        bound.add(13, 1, 0.045, 0.046)
        self.assertAlmostEqual(bound.bound(13) - before, 0.045 - before + 0.001, places=12)
        self.assertLessEqual(bound.bound(13) - before, 0.006)

    def test_the_next_substep_is_taken_to_run_as_many_loops_as_the_last_one(self):
        bound = SubstepBound()
        bound.add(4, 1, 0.008, 0.010)
        one_loop = bound.bound(13)
        self.assertAlmostEqual(one_loop, 0.002 + 13 * 0.002, places=12)
        # A handover to the host-controlled loop, or a rejected attempt: two loops.
        bound.add(4, 2, 0.008, 0.010)
        self.assertEqual(bound.last_loops, 2)
        self.assertAlmostEqual(bound.bound(13), 0.002 + 2 * 13 * 0.002, places=12)
        # An attempt rejected twice and accepted at the third try.
        bound.add(6, 3, 0.012, 0.014)
        self.assertAlmostEqual(bound.bound(13), 0.002 + 3 * 13 * 0.002, places=12)
        # The row has ended: a substep of one loop brings the bound back.
        bound.add(2, 1, 0.004, 0.006)
        self.assertEqual(bound.last_loops, 1)
        self.assertAlmostEqual(bound.bound(13), 0.002 + 13 * 0.002, places=12)


def _snapshot(scene):
    """Everything the accepted frame left behind that the next frame starts from, as host copies."""
    predictor = scene.coupler._phase_load_predictor
    return {
        **{name: getattr(scene.state_0, name).numpy().copy() for name in STATE},
        "native": _native(scene),
        "load": scene.coupler._load.numpy().copy(),
        "joint_target_q": scene.control.joint_target_q.numpy().copy(),
        "arm_targets": np.asarray(scene.arm_targets).copy(),
        "thread": np.concatenate(
            [np.asarray(array.numpy(), dtype=np.float64).ravel() for array in scene.worker_thread.state]
        ),
        "thread_positions": np.asarray(scene.worker_thread.positions()).copy(),
        "frame_positions": scene._frame_particle_q.numpy().copy(),
        "elapsed": scene.elapsed,
        "counters": (
            scene.accepted_physics_steps,
            scene.step_refinements,
            scene.minimum_step_dt,
            scene.coupler.evaluations,
            scene.coupler.maximum_iterations,
            scene.task_status,
            scene.failure_reason,
            len(scene.interaction_events),
            len(scene.actual_tip_history),
            scene.worker_thread.accepted_steps,
            scene.worker_thread.physical_time_s,
            predictor.depth,
            predictor.phase,
            predictor.frame_open,
            scene._ramp_frame,
            scene.frame_pending,
        ),
        # How often the tangent of the tissue solver was written, and the time step it is built for.
        "tangent": (scene.solver._tangent_version, scene.solver.dt),
    }


@contextmanager
def _rejecting(scene, reject):
    """Reject the accepted-step checks with the given numbers (counted from 0 over the run).

    The check runs after the trials of an attempt: the coupler restores the attempt and the
    scene retries it at half the time step, as it does when the tip has moved too far.
    ``reject`` is read at every check: a test may add numbers to it while it runs.
    """
    guard = scene.coupler._tet_step_guard
    check, checks = guard.check, [0]

    def counted(state_in, candidate):
        number = checks[0]
        checks[0] += 1
        if number in reject:
            raise NeedleStepTooLarge("rejected by the test")
        return check(state_in, candidate)

    with patch.object(guard, "check", side_effect=counted):
        yield checks


@contextmanager
def _attempts(scene):
    """Count the attempts of the scene: the calls of its coupler's step, accepted or rejected."""
    coupler, step, count = scene.coupler, scene.coupler.step, [0]

    def counted(*arguments, **options):
        count[0] += 1
        return step(*arguments, **options)

    with patch.object(coupler, "step", counted):
        yield count


def _run(deadline, *, reject=(), observe=False, actions=ACTIONS):
    """Run the frames of ``actions``.

    Returns the snapshot after every frame, the calls of every frame (per call: the attempts it
    ran, the steps it had accepted and the attempts it had rejected, and the unfinished frame
    it left), the number of accepted-step checks, the scene's measured bound and its time step.
    """
    snapshots, calls = [], []
    with _scene() as (scene, config), _rejecting(scene, reject) as checks, _attempts(scene) as attempts:
        for action in actions:
            _apply_action(scene, config, action)
            frame = []
            while True:
                before = (attempts[0], scene.accepted_physics_steps, scene.step_refinements)
                poses = _step(scene) if deadline is None else _step(scene, deadline)
                frame.append(
                    {
                        "attempts": attempts[0] - before[0],
                        "accepted": scene.accepted_physics_steps - before[1],
                        "rejected": scene.step_refinements - before[2],
                        "pending": scene._pending_frame,
                    }
                )
                assert len(frame) <= CALL_LIMIT, "a frame makes no progress"
                if observe:
                    # What a driver does between two calls: read the accepted state.
                    _observe(scene, 0, body_q=poses)
                if not scene.frame_pending:
                    break
                # A frame in progress has ended nothing yet.
                assert poses is None and scene._ramp_frame is not None
                assert scene.coupler._phase_load_predictor.frame_open
            scene.actual_tip_history.clear()
            snapshots.append(_snapshot(scene))
            calls.append(frame)
        bound = scene._substep_bound
        sim_dt = scene.sim_dt
    return snapshots, calls, checks[0], bound, sim_dt


def _flat(calls):
    return [call for frame in calls for call in frame]


class SplitFrameTests(unittest.TestCase):
    def assertSameFrames(self, whole, split):
        self.assertEqual(len(whole), len(split))
        for frame, (expected, actual) in enumerate(zip(whole, split, strict=True)):
            self.assertEqual(expected.keys(), actual.keys())
            for name, value in expected.items():
                with self.subTest(frame=frame, name=name):
                    if isinstance(value, np.ndarray):
                        # Bit for bit: the split moves where a call returns and nothing else.
                        self.assertEqual(value.tobytes(), actual[name].tobytes())
                    else:
                        self.assertEqual(value, actual[name])

    def test_a_passed_deadline_runs_one_attempt_per_call(self):
        split, calls, checks, bound, _ = _run(PASSED, observe=True)
        accepted, refinements = split[-1]["counters"][:2]
        records = _flat(calls)
        # One attempt per call: no call without one and none with two.
        self.assertEqual([record["attempts"] for record in records], [1] * len(records))
        self.assertEqual(len(records), accepted + refinements)
        self.assertLessEqual(checks, len(records))
        # Every frame ended in a call that left nothing unfinished, every other call left the frame.
        for frame in calls:
            self.assertIsNone(frame[-1]["pending"])
            self.assertTrue(all(record["pending"] is not None for record in frame[:-1]))
        # Every substep was measured: its trials, the time of its loops and its time outside them.
        self.assertEqual((bound.substeps, bound.trials), (accepted, split[-1]["counters"][3]))
        self.assertGreater(bound.loop_seconds, 0.0)
        self.assertGreater(bound.longest_outside, 0.0)

    @unittest.skipIf(_CUDA, _BITS)
    def test_frames_split_at_every_attempt_are_the_frames_of_one_call(self):
        whole, whole_calls, whole_checks, untouched, _ = _run(None)
        split, split_calls, split_checks, _, _ = _run(PASSED, observe=True)
        self.assertSameFrames(whole, split)
        self.assertEqual([len(frame) for frame in whole_calls], [1] * len(ACTIONS))
        # Without a deadline nothing is measured and nothing is left unfinished.
        self.assertIsNone(untouched)
        # Four steps per frame, and the scene has seen contact: the frames took many trials.
        accepted = whole[-1]["counters"][0]
        self.assertEqual(whole_checks, split_checks)
        self.assertEqual([len(frame) for frame in split_calls], [4] * len(ACTIONS))
        self.assertGreater(whole[-1]["counters"][3], accepted)
        self.assertGreater(whole[-1]["counters"][7], 0)

    def test_a_rejected_attempt_and_its_retry_run_in_different_calls(self):
        # Checks 13 and 14 are the second step of the fourth frame and its first retry.
        split, calls, checks, bound, sim_dt = _run(PASSED, reject=(13, 14), observe=True, actions=ACTIONS[:5])
        records = _flat(calls)
        self.assertEqual([record["attempts"] for record in records], [1] * len(records))
        rejected = [index for index, record in enumerate(records) if record["rejected"]]
        # Two rejections in a row: the attempt and its first retry, each alone in its call.
        self.assertEqual(rejected, [13, 14])
        first, second, third = records[13:16]
        for record, divisor in ((first, 2), (second, 4)):
            self.assertEqual((record["accepted"], record["rejected"]), (0, 1))
            step_dt, _, quick_steps, spent = record["pending"]
            # The time step of the retry, a new count of quick steps, and the substep so far.
            self.assertEqual((step_dt, quick_steps), (sim_dt / divisor, 0))
            self.assertIsNotNone(spent)
        # A rejection leaves the rest of the frame as it was, and the second call measures on
        # from what the first one spent on the substep.
        self.assertEqual(first["pending"][1], second["pending"][1])
        self.assertEqual(first["pending"][1], records[12]["pending"][1])
        for mine, earlier in zip(second["pending"][3][:3], first["pending"][3][:3], strict=True):
            self.assertGreater(mine, earlier)
        # The second retry is accepted; between two substeps nothing of a substep is kept.
        self.assertEqual((third["accepted"], third["rejected"]), (1, 0))
        self.assertIsNone(third["pending"][3])
        self.assertTrue(all(record["pending"] is None or record["pending"][3] is None for record in records[:13]))
        accepted, refinements, shortest = split[-1]["counters"][:3]
        self.assertEqual((refinements, shortest), (2, sim_dt / 4))
        self.assertEqual(len(records), accepted + 2)
        self.assertEqual(checks, accepted + 2)
        # The frame with the rejections took more calls than a frame of four steps.
        self.assertGreater(len(calls[3]), 6)
        # A substep is measured as one, with the trials of its rejected attempts.
        self.assertEqual((bound.substeps, bound.trials), (accepted, split[-1]["counters"][3]))

    @unittest.skipIf(_CUDA, _BITS)
    def test_frames_with_a_rejected_attempt_and_its_retry_in_different_calls_are_the_frames_of_one_call(self):
        whole, _, whole_checks, _, sim_dt = _run(None, reject=(13, 14))
        split, split_calls, split_checks, _, _ = _run(PASSED, reject=(13, 14), observe=True)
        # With the tangent: written as often, and built for the same time step, as in one call.
        self.assertSameFrames(whole, split)
        accepted, refinements, shortest = whole[-1]["counters"][:3]
        self.assertEqual((refinements, shortest), (2, sim_dt / 4))
        self.assertEqual(whole_checks, split_checks)
        self.assertEqual(whole_checks, accepted + 2)
        self.assertEqual(sum(len(frame) for frame in split_calls), accepted + 2)

    def test_a_substep_whose_attempts_ran_in_two_calls_is_measured_as_one(self):
        with _scene() as (scene, config), _rejecting(scene, (0,)) as checks:
            _apply_action(scene, config, ACTIONS[0])
            coupler = scene.coupler
            started = time.perf_counter()
            _step(scene, PASSED)
            first = time.perf_counter() - started
            # The rejected attempt alone: nothing is accepted and no substep is measured yet.
            self.assertEqual((checks[0], scene.accepted_physics_steps, scene.step_refinements), (1, 0, 1))
            self.assertTrue(scene.frame_pending)
            self.assertIsNone(scene._substep_bound)
            step_dt, remaining, quick_steps, spent = scene._pending_frame
            self.assertEqual((step_dt, remaining, quick_steps), (scene.sim_dt / 2, scene.frame_dt, 0))
            # What the call spent on the substep: its time, its trials, its loops and their time.
            self.assertEqual(tuple(spent[1:3]), (coupler.evaluations, coupler.trial_loops))
            self.assertAlmostEqual(spent[3], coupler.loop_seconds, places=12)
            self.assertGreater(spent[0], spent[3])
            self.assertLessEqual(spent[0], first)
            # The time between the two calls is no time of the substep.
            time.sleep(0.25)
            started = time.perf_counter()
            _step(scene, PASSED)
            second = time.perf_counter() - started
            self.assertEqual((checks[0], scene.accepted_physics_steps, scene.step_refinements), (2, 1, 1))
            self.assertTrue(scene.frame_pending)
            self.assertIsNone(scene._pending_frame[3])
            # One substep: the rejected attempt and the accepted one at half the time step.
            bound = scene._substep_bound
            self.assertEqual((bound.substeps, bound.trials), (1, coupler.evaluations))
            self.assertEqual(bound.last_loops, coupler.trial_loops)
            self.assertGreaterEqual(bound.last_loops, 2)
            self.assertAlmostEqual(bound.loop_seconds, coupler.loop_seconds, places=12)
            # One substep is measured: its time is its loops and the longest time outside them.
            self.assertGreater(bound.longest_outside, 0.0)
            self.assertLessEqual(bound.longest_outside + bound.loop_seconds, first + second)
            # The bound for the next substep covers as many loops, each to the trial limit, and
            # the limit of its mean trials is longer than its loops.
            limit = scene._trial_limit()
            self.assertEqual(bound.longest_loop, bound.loop_seconds / bound.last_loops)
            self.assertAlmostEqual(
                bound.bound(limit), bound.longest_outside + bound.last_loops * limit * bound.per_trial, places=12
            )
            loops = coupler.trial_loops
            _step(scene, PASSED)
            self.assertEqual(bound.last_loops, coupler.trial_loops - loops)
            self.assertEqual(bound.substeps, 2)

    def test_a_deadline_far_away_runs_whole_frames_and_measures_them(self):
        # Check 5 is the second step of the second frame: the bound fits, the retry follows in the same call.
        far, calls, checks, bound, _ = _run(time.perf_counter() + 3600.0, reject=(5,), actions=ACTIONS[:4])
        accepted, refinements = far[-1]["counters"][:2]
        self.assertEqual([len(frame) for frame in calls], [1, 1, 1, 1])
        self.assertEqual(refinements, 1)
        self.assertEqual(sum(record["attempts"] for record in _flat(calls)), accepted + 1)
        self.assertEqual((bound.substeps, bound.trials), (accepted, far[-1]["counters"][3]))

    def test_the_bound_of_a_scene_is_no_less_than_any_substep_it_has_measured(self):
        measured, add = [], SubstepBound.add

        def recorded(bound, trials, loops, loop_seconds, seconds):
            measured.append((trials, loops, loop_seconds, seconds))
            add(bound, trials, loops, loop_seconds, seconds)

        # Check 5 is rejected: one substep of the run has two attempts, and so more than one loop.
        with patch.object(SubstepBound, "add", recorded):
            far, _, _, bound, _ = _run(time.perf_counter() + 3600.0, reject=(5,), actions=ACTIONS[:14])
        self.assertEqual(len(measured), far[-1]["counters"][0])
        self.assertGreater(max(loops for _, loops, _, _ in measured), 1)
        # The two maxima are those of the substeps measured ...
        self.assertEqual(bound.longest_outside, max(seconds - loop for _, _, loop, seconds in measured))
        self.assertEqual(bound.longest_loop, max(loop / loops for _, loops, loop, _ in measured))
        mean_outside = sum(seconds - loop for _, _, loop, seconds in measured) / len(measured)
        self.assertGreater(bound.longest_outside, mean_outside)
        for _trials, loops, _loop_seconds, seconds in measured:
            # ... so no substep took longer than the bound of a substep with its loops, whatever
            # the trial limit: the worst case measured is inside the bound.
            self.assertLessEqual(seconds, bound.longest_outside + loops * bound.longest_loop + 1e-12)
            if loops <= bound.last_loops:
                self.assertLessEqual(seconds, bound.bound(1) + 1e-12)

    @unittest.skipIf(_CUDA, _BITS)
    def test_frames_under_a_deadline_far_away_are_the_frames_without_one(self):
        whole, _, _, _, _ = _run(None, reject=(5,), actions=ACTIONS[:4])
        far, _, _, _, _ = _run(time.perf_counter() + 3600.0, reject=(5,), actions=ACTIONS[:4])
        self.assertSameFrames(whole, far)

    def test_the_next_substep_starts_only_if_its_bound_ends_by_the_deadline(self):
        class Bound:
            """A bound of half a second, whatever is measured."""

            limits = []

            def add(self, trials, loops, loop_seconds, seconds):
                pass

            def bound(self, trial_limit):
                self.limits.append(trial_limit)
                return 0.5

        with _scene() as (scene, config):
            # No trial budget on this scene: a loop may run to the limit of the coupler. With a
            # budget it applies its best load once more.
            self.assertIsNone(scene.coupler.trial_budget)
            self.assertEqual(scene._trial_limit(), scene.coupler.max_iterations)
            scene.coupler.trial_budget = 12
            self.assertEqual(scene._trial_limit(), 13)
            scene.coupler.trial_budget = None
            _apply_action(scene, config, ACTIONS[0])
            scene._substep_bound = Bound()
            # The clock stands still at 100 s. A deadline short of the bound ends the call after
            # one substep; a deadline at the bound lets every further substep start.
            with patch.object(time, "perf_counter", return_value=100.0):
                _step(scene, 100.5 - 1e-6)
                self.assertEqual(scene.accepted_physics_steps, 1)
                self.assertTrue(scene.frame_pending)
                _step(scene, 100.5 - 1e-6)
                self.assertEqual(scene.accepted_physics_steps, 2)
                self.assertTrue(scene.frame_pending)
                _step(scene, 100.5)
                self.assertFalse(scene.frame_pending)
            self.assertEqual(scene.accepted_physics_steps, 4)
            self.assertAlmostEqual(scene.elapsed, config.frame_dt, delta=1e-12)
            # Asked once after every substep that another one could follow.
            self.assertEqual(Bound.limits, [scene.coupler.max_iterations] * 3)

    def test_a_retry_starts_in_the_call_of_its_rejection_only_if_its_bound_ends_by_the_deadline(self):
        class Bound:
            """A bound of half a second, whatever is measured."""

            limits = []

            def add(self, trials, loops, loop_seconds, seconds):
                pass

            def bound(self, trial_limit):
                self.limits.append(trial_limit)
                return 0.5

        reject = {0}
        with _scene() as (scene, config), _rejecting(scene, reject) as checks:
            _apply_action(scene, config, ACTIONS[0])
            scene._substep_bound = Bound()
            # The clock stands still at 100 s.
            with patch.object(time, "perf_counter", return_value=100.0):
                # A deadline short of the bound: the call returns before the retry, and has asked.
                _step(scene, 100.5 - 1e-6)
                self.assertEqual((checks[0], scene.accepted_physics_steps, scene.step_refinements), (1, 0, 1))
                self.assertTrue(scene.frame_pending)
                self.assertIsNotNone(scene._pending_frame[3])
                self.assertEqual(len(Bound.limits), 1)
                # The next call starts with the retry whatever its deadline says, then asks again.
                _step(scene, 100.5 - 1e-6)
                self.assertEqual((checks[0], scene.accepted_physics_steps, scene.step_refinements), (2, 1, 1))
                self.assertTrue(scene.frame_pending)
                self.assertIsNone(scene._pending_frame[3])
                self.assertEqual(len(Bound.limits), 2)
                # A deadline at the bound: the rest of the frame in one call.
                _step(scene, 100.5)
                self.assertFalse(scene.frame_pending)
                self.assertAlmostEqual(scene.elapsed, config.frame_dt, delta=1e-12)
                # And a rejected attempt with its retry in one call: the first step of the next frame.
                reject.add(checks[0])
                _apply_action(scene, config, ACTIONS[1])
                _step(scene, 100.5)
                self.assertFalse(scene.frame_pending)
                self.assertEqual(scene.step_refinements, 2)
                self.assertAlmostEqual(scene.elapsed, 2 * config.frame_dt, delta=1e-12)
            # Asked once after every attempt that another one could follow: after every rejected
            # attempt, and after every accepted step but the last one of each of the two frames.
            self.assertEqual(len(Bound.limits), scene.step_refinements + scene.accepted_physics_steps - 2)
            self.assertEqual(set(Bound.limits), {scene.coupler.max_iterations})

    def test_every_call_runs_an_attempt(self):
        # Three rejections in a row, a deadline that passed long ago, and nothing measured at first.
        with _scene() as (scene, config), _rejecting(scene, (0, 1, 2)), _attempts(scene) as attempts:
            _apply_action(scene, config, ACTIONS[0])
            calls = 0
            while True:
                before = attempts[0]
                _step(scene, -math.inf)
                calls += 1
                self.assertEqual(attempts[0], before + 1)
                if not scene.frame_pending:
                    break
                self.assertLess(calls, CALL_LIMIT)
            self.assertEqual(scene.step_refinements, 3)
            self.assertEqual(scene.minimum_step_dt, scene.sim_dt / 8)
            self.assertEqual(calls, scene.accepted_physics_steps + 3)
            self.assertAlmostEqual(scene.elapsed, config.frame_dt, delta=1e-12)

    def test_a_call_without_a_deadline_finishes_an_unfinished_frame(self):
        # The second attempt of each frame is rejected: the call without a deadline starts with its retry.
        reject = set()
        with _scene() as (scene, config), _rejecting(scene, reject) as checks:
            for frame, action in enumerate(ACTIONS[:2]):
                _apply_action(scene, config, action)
                self.assertIsNone(_step(scene, PASSED))
                reject.add(checks[0])
                self.assertIsNone(_step(scene, PASSED))
                self.assertTrue(scene.frame_pending)
                self.assertIsNotNone(scene._pending_frame[3])
                self.assertIsNotNone(_step(scene))
                self.assertFalse(scene.frame_pending)
                self.assertAlmostEqual(scene.elapsed, (frame + 1) * config.frame_dt, delta=1e-12)
            self.assertEqual(scene.step_refinements, 2)

    @unittest.skipIf(_CUDA, _BITS)
    def test_a_frame_finished_by_a_call_without_a_deadline_is_the_frame_of_one_call(self):
        # Checks 1 and 7 are the second attempts of the two frames.
        whole, _, _, _, _ = _run(None, reject=(1, 7), actions=ACTIONS[:2])
        with _scene() as (scene, config), _rejecting(scene, (1, 7)):
            for action in ACTIONS[:2]:
                _apply_action(scene, config, action)
                _step(scene, PASSED)
                _step(scene, PASSED)
                self.assertIsNotNone(scene._pending_frame[3])
                _step(scene)
            scene.actual_tip_history.clear()
            self.assertSameFrames(whole[-1:], [_snapshot(scene)])

    def after_reset(self, deadline, partial):
        """Three frames after a reset that falls between two frames, between two substeps or before a retry."""
        with _scene() as (scene, config), _rejecting(scene, (9,) if partial == "retry" else ()):
            for action in ACTIONS[:2]:
                _apply_action(scene, config, action)
                _step(scene)
            if partial:
                # Two calls of a third frame, under another action; the second attempt is
                # accepted (a reset between two substeps) or rejected (a reset before its retry).
                _apply_action(scene, config, ACTIONS[-1])
                _step(scene, PASSED)
                _step(scene, PASSED)
                self.assertTrue(scene.frame_pending)
                self.assertIsNotNone(scene._ramp_frame)
                if partial == "retry":
                    self.assertEqual((scene.accepted_physics_steps, scene.step_refinements), (9, 1))
                    self.assertIsNotNone(scene._pending_frame[3])
                    # The tangent of the retry is built: the solver holds the halved time step.
                    self.assertEqual(scene.solver.dt, scene.sim_dt / 2)
                else:
                    self.assertEqual((scene.accepted_physics_steps, scene.step_refinements), (10, 0))
                    self.assertIsNone(scene._pending_frame[3])
            _reset_scene(scene, config, rebuild=True)
            self.assertFalse(scene.frame_pending)
            self.assertIsNone(scene._pending_frame)
            self.assertIsNone(scene._ramp_frame)
            self.assertIsNone(scene._substep_bound)
            self.assertEqual((scene.elapsed, scene.accepted_physics_steps, scene.step_refinements), (0.0, 0, 0))
            snapshots = []
            for action in ACTIONS[:3]:
                _apply_action(scene, config, action)
                _step(scene) if deadline is None else _step(scene, deadline)
                if deadline is not None and not snapshots:
                    # The first call after the reset starts a frame at the whole time step.
                    self.assertEqual(scene._pending_frame[0], scene.sim_dt)
                    self.assertIsNone(scene._pending_frame[3])
                    self.assertEqual(scene.solver.dt, scene.sim_dt)
                while scene.frame_pending:
                    _step(scene, deadline)
                scene.actual_tip_history.clear()
                snapshots.append(_snapshot(scene))
            self.assertEqual(snapshots[-1]["counters"][:3], (12, 0, scene.sim_dt))
            return snapshots

    def test_a_reset_in_an_unfinished_frame_drops_it(self):
        self.after_reset(PASSED, partial="substep")

    def test_a_reset_between_a_rejected_attempt_and_its_retry_drops_the_frame_and_the_retry(self):
        self.after_reset(PASSED, partial="retry")

    def test_a_rebuild_drops_an_unfinished_frame_without_the_close_it_runs(self):
        # A rebuild of this scene runs its close() (it releases the response graphs), and a close
        # drops an unfinished frame too. A scene without a close has the reset of its
        # interaction state alone: that reset must drop the frame by itself.
        with _scene() as (scene, config), _rejecting(scene, (1,)):
            _apply_action(scene, config, ACTIONS[0])
            _step(scene, PASSED)
            _step(scene, PASSED)
            self.assertIsNotNone(scene._pending_frame[3])
            self.assertIsNotNone(scene._substep_bound)
            with patch.object(type(scene), "close", lambda self: None):
                _reset_scene(scene, config, rebuild=True)
            self.assertFalse(scene.frame_pending)
            self.assertIsNone(scene._pending_frame)
            self.assertIsNone(scene._ramp_frame)
            self.assertIsNone(scene._substep_bound)

    @unittest.skipIf(_CUDA, _BITS)
    def test_frames_after_a_reset_in_an_unfinished_frame_are_those_after_any_reset(self):
        whole = self.after_reset(None, partial=None)
        self.assertSameFrames(whole, self.after_reset(None, partial="substep"))
        self.assertSameFrames(whole, self.after_reset(PASSED, partial="substep"))
        self.assertSameFrames(whole, self.after_reset(None, partial="retry"))
        self.assertSameFrames(whole, self.after_reset(PASSED, partial="retry"))

    def test_a_scene_that_stops_in_an_unfinished_frame_does_not_go_on_with_it(self):
        for reject in ((), (1,)):
            with self.subTest(reject=reject), _scene() as (scene, config), _rejecting(scene, reject):
                _apply_action(scene, config, ACTIONS[0])
                _step(scene, PASSED)
                _step(scene, PASSED)
                self.assertTrue(scene.frame_pending)
                # Between two substeps, or between a rejected attempt and its retry.
                self.assertEqual(scene._pending_frame[3] is not None, bool(reject))
                scene._fail("stopped by the test")
                steps = scene.accepted_physics_steps
                _step(scene, PASSED)
                self.assertFalse(scene.frame_pending)
                self.assertEqual(scene.accepted_physics_steps, steps)
                # The control frame and the frame of the arm-target ramp are closed.
                self.assertFalse(scene.coupler._phase_load_predictor.frame_open)
                self.assertIsNone(scene._ramp_frame)


@contextmanager
def _local(**settings):
    with (
        TemporaryDirectory() as directory,
        patch.object(phantom, "list_mesh_regions", return_value=_REFERENCE.regions),
        patch.object(phantom, "load_mesh_regions", side_effect=_selection),
    ):
        config = SutureSceneConfig(
            device=DEVICE,
            asset=str(Path(directory) / "pad.usd"),
            entry_xy=(0, 0.005),
            exit_xy=(0, -0.005),
            control_mode="physical_grasp",
            **settings,
        )
        with LocalSutureScene(config) as scenes:
            yield scenes


# Host times, and what describes the process and not the physics, differ between two runs of the same frames.
_UNSTABLE = ("worker_started_ns", "worker_finished_ns", "pid")


def _same_value(case, expected, actual, path):
    """Compare two observation values: arrays bit for bit, everything else by equality."""
    if isinstance(expected, np.ndarray):
        case.assertEqual(expected.dtype, actual.dtype, path)
        case.assertEqual(expected.tobytes(), actual.tobytes(), path)
    elif isinstance(expected, dict):
        case.assertEqual(expected.keys(), actual.keys(), path)
        for key, value in expected.items():
            if key not in _UNSTABLE:
                _same_value(case, value, actual[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        case.assertEqual(len(expected), len(actual), path)
        for index, value in enumerate(expected):
            _same_value(case, value, actual[index], f"{path}[{index}]")
    elif isinstance(expected, float) and math.isnan(expected):
        case.assertTrue(math.isnan(actual), path)
    else:
        case.assertEqual(expected, actual, path)


class LocalSceneBudgetTests(unittest.TestCase):
    # Checks 5 and 6: the second step of the second frame and its first retry.
    reject = (5, 6)

    def frames(self, deadline, actions):
        """The observation after every complete frame, and the observation after every call."""
        complete, every = [], []
        with _local() as scenes, _rejecting(scenes._scene, self.reject):
            for action in actions:
                result = scenes.step(action) if deadline is None else scenes.step(action, deadline=deadline)
                every.append((result[0], scenes.frame_unfinished))
                while scenes.frame_unfinished:
                    with self.assertRaisesRegex(ValueError, "unfinished"):
                        scenes.step(action)
                    result = scenes.step(None, deadline=deadline)
                    every.append((result[0], scenes.frame_unfinished))
                    self.assertLess(len(every), CALL_LIMIT * len(actions))
                complete.append(result[0])
            with self.assertRaisesRegex(ValueError, r"shape \(1, 7\)"):
                scenes.step(None)
            frame_dt = scenes.config.frame_dt
        return complete, every, frame_dt

    def test_an_unfinished_frame_goes_on_without_an_action_one_attempt_per_call(self):
        actions = np.asarray(ACTIONS[:4], dtype=float)[:, None, :]
        complete, every, frame_dt = self.frames(PASSED, actions)
        last = complete[-1]["diagnostics"]
        self.assertEqual(last["step_refinements"], 2)
        self.assertAlmostEqual(last["physical_time_s"], 4 * frame_dt, delta=1e-12)
        # One attempt per call.
        self.assertEqual(len(every), last["accepted_physics_steps"] + 2)
        previous, frames = None, 0
        for observation, unfinished in every:
            diagnostics, before = observation["diagnostics"], previous
            previous = observation
            if unfinished:
                # The accepted state in the middle of a frame lies before the frame's end.
                self.assertLess(diagnostics["physical_time_s"], (frames + 1) * frame_dt - 1e-9)
            else:
                frames += 1
                self.assertAlmostEqual(diagnostics["physical_time_s"], frames * frame_dt, delta=1e-12)
            if before is not None and diagnostics["step_refinements"] > before["diagnostics"]["step_refinements"]:
                # Between a rejected attempt and its retry the driver sees the accepted state
                # it saw before, and the rejection counted.
                self.assertTrue(unfinished)
                self.assertEqual(diagnostics["physical_time_s"], before["diagnostics"]["physical_time_s"])
                self.assertEqual(diagnostics["accepted_physics_steps"], before["diagnostics"]["accepted_physics_steps"])
                self.assertEqual(observation["needle_pose"].tobytes(), before["needle_pose"].tobytes())

    @unittest.skipIf(_CUDA, _BITS)
    def test_observations_of_complete_frames_are_those_of_whole_frames(self):
        actions = np.asarray(ACTIONS[:14], dtype=float)[:, None, :]
        whole, calls, _ = self.frames(None, actions)
        self.assertEqual(len(calls), len(actions))
        split, _, _ = self.frames(PASSED, actions)
        self.assertEqual(whole[-1]["diagnostics"]["step_refinements"], 2)
        for frame, (expected, actual) in enumerate(zip(whole, split, strict=True)):
            # The whole observation of a complete frame: state, report, thread, events, counters.
            _same_value(self, expected, actual, f"frame {frame}")

    def test_a_reset_between_two_calls_of_a_frame_starts_a_new_frame(self):
        action = np.asarray(ACTIONS[0], dtype=float)[None, :]
        with _local() as scenes:
            for reject in ((), (1,)):
                with self.subTest(reject=reject), _rejecting(scenes._scene, reject):
                    scenes.step(action, deadline=PASSED)
                    scenes.step(None, deadline=PASSED)
                    self.assertTrue(scenes.frame_unfinished)
                    # Between two substeps, or between a rejected attempt and its retry.
                    self.assertEqual(scenes._scene._pending_frame[3] is not None, bool(reject))
                    count = scenes._scene.reset_count
                    reset = scenes.reset()[0]
                    self.assertFalse(scenes.frame_unfinished)
                    self.assertEqual((reset["reset_count"], reset["diagnostics"]["physical_time_s"]), (count + 1, 0.0))
                    self.assertEqual(reset["diagnostics"]["step_refinements"], 0)
                    # The frame after the reset needs an action and is a whole frame without a deadline.
                    with self.assertRaisesRegex(ValueError, r"shape \(1, 7\)"):
                        scenes.step(None)
                    stepped = scenes.step(action)[0]["diagnostics"]
                    self.assertFalse(scenes.frame_unfinished)
                    self.assertEqual((stepped["accepted_physics_steps"], stepped["step_refinements"]), (4, 0))
                    self.assertAlmostEqual(stepped["physical_time_s"], scenes.config.frame_dt, delta=1e-12)

    def close_before_a_retry(self):
        """Close a scene between a rejected attempt and its retry; returns the closed scene and its driver."""
        action = np.asarray(ACTIONS[0], dtype=float)[None, :]
        with _local() as scenes:
            scene = scenes._scene
            with _rejecting(scene, (1,)):
                scenes.step(action, deadline=PASSED)
                result = scenes.step(None, deadline=PASSED)[0]["diagnostics"]
            self.assertTrue(scenes.frame_unfinished)
            self.assertIsNotNone(scene._pending_frame[3])
            self.assertIsNotNone(scene._ramp_frame)
            self.assertEqual((result["accepted_physics_steps"], result["step_refinements"]), (1, 1))
            scenes.close()
            return scene, scenes

    def test_a_close_between_a_rejected_attempt_and_its_retry_keeps_nothing_of_the_frame(self):
        scene, scenes = self.close_before_a_retry()
        self.assertFalse(scenes.frame_unfinished)
        self.assertFalse(scene.frame_pending)
        self.assertIsNone(scene._pending_frame)
        self.assertIsNone(scene._ramp_frame)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            scenes.step(None)

    def test_the_scene_gets_the_deadline_less_the_longest_observation(self):
        calls, clock, observation = [], [100.0], [0.0]
        scene = SimpleNamespace(
            actual_tip_history=[], frame_pending=False, close=lambda: None, viewer=SimpleNamespace(close=lambda: None)
        )

        def step(scene, *deadline):
            calls.append(deadline)

        def observe(scene, env_id, **options):
            # The observation takes the time the test gives it, on the clock the test moves.
            clock[0] += observation[0]
            return {"env_id": env_id}

        with (
            patch.object(parallel, "_make_scene", lambda config, env_id: scene),
            patch.object(parallel, "_observe", observe),
            patch.object(parallel, "_apply_action", lambda scene, config, action: None),
            patch.object(parallel, "_step", step),
            patch.object(parallel, "_reset_scene", lambda scene, config, rebuild: None),
            patch.object(time, "perf_counter", lambda: clock[0]),
            LocalSutureScene() as scenes,
        ):
            action = np.zeros((1, 6))
            # Without a deadline the scene is stepped as before: no second argument.
            scenes.step(action)
            self.assertEqual(calls, [()])
            self.assertEqual(scenes._observe_longest, 0.0)
            # An observation of 250 ms, then one of 62.5 ms: the last one and the longest one.
            observation[0] = 0.25
            scenes.step(action)
            observation[0] = 0.0625
            scenes.step(action, deadline=10.0)
            self.assertEqual(calls[2], (9.75,))
            self.assertEqual(scenes._observe_longest, 0.25)
            # The scene gets the deadline less the longest observation, not less the last one.
            scenes.step(action, deadline=10.0)
            self.assertEqual(calls[3], (9.75,))
            # A longer one counts from the call after it.
            observation[0] = 0.5
            scenes.step(action, deadline=10.0)
            observation[0] = 0.0
            scenes.step(action, deadline=10.0)
            self.assertEqual(calls[4:], [(9.75,), (9.5,)])
            self.assertEqual(scenes._observe_longest, 0.5)
            # A reset rebuilds the scene: it measures anew, and so does the observation after its steps.
            observation[0] = 0.125
            scenes.reset()
            self.assertEqual(scenes._observe_longest, 0.0)
            scenes.step(action, deadline=10.0)
            scenes.step(action, deadline=10.0)
            self.assertEqual(calls[6:], [(10.0,), (9.875,)])
            self.assertFalse(scenes.frame_unfinished)
            scene.frame_pending = True
            self.assertTrue(scenes.frame_unfinished)
            # A scene without the notion of an unfinished frame never has one.
            del scene.frame_pending
            self.assertFalse(scenes.frame_unfinished)


class _Pool:
    """A scene pool that needs the given number of calls for each frame."""

    def __init__(self, calls_per_frame, frame_dt):
        self.plan = list(calls_per_frame)
        self.frame_dt = frame_dt
        self.frame_unfinished = False
        self.left = 0
        self.time = 0.0
        self.steps = []

    def step(self, actions, **options):
        self.steps.append((None if actions is None else np.asarray(actions).copy(), options))
        if not self.frame_unfinished:
            self.left = self.plan.pop(0)
            self.fraction = 1.0 / self.left
        self.left -= 1
        self.time += self.fraction * self.frame_dt
        self.frame_unfinished = self.left > 0
        return [{"diagnostics": {"physical_time_s": self.time}}]


class _Viewer:
    """What the example asks a viewer in step(): whether it is paused."""

    def __init__(self, paused=False):
        self.paused = paused

    def is_paused(self):
        return self.paused


class _GLViewer(newton.viewer.ViewerGL):
    """Newton's OpenGL viewer as the example sees it; no window and no renderer is made."""

    def __init__(self, headless=False, paused=False, num_frames=2400):
        self._headless, self._paused, self.num_frames = headless, paused, num_frames


def _other(viewer_class, **attributes):
    """A viewer of another class of Newton, without its window, server or file."""
    viewer = object.__new__(viewer_class)
    for name, value in attributes.items():
        object.__setattr__(viewer, name, value)
    return viewer


class ExampleBudgetTests(unittest.TestCase):
    def example(self, calls_per_frame, budget):
        example = object.__new__(grasp_example.Example)
        example.args = SimpleNamespace(num_envs=1)
        example.config = SimpleNamespace(frame_dt=1.0 / 60.0, action_dimension=7)
        example.pool = _Pool(calls_per_frame, example.config.frame_dt)
        example.observations = [{"diagnostics": {"physical_time_s": 0.0}}]
        example.halted = example.render_scene = example.last_phases = None
        example.sim_time, example.completed = 0.0, 0
        example.frame_budget = budget
        example.viewer = _Viewer()
        self.controlled, self.phase = [], ["settling"]

        def controller(observations):
            self.controlled.append(observations[0]["diagnostics"]["physical_time_s"])
            return np.full((1, 7), 0.5)

        controller.report = lambda: [{"env_id": 0, "phase": self.phase[0], "reason": None}]
        example.controller = controller
        return example

    def test_the_controller_acts_once_per_frame_and_only_complete_frames_are_counted(self):
        example = self.example([1, 3, 1, 2], 1.0 / 15.0)
        # The longest drawing took 8 ms.
        example._draw_longest = 0.008
        before = time.perf_counter()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for call in range(7):
                if call == 2:
                    # The phase changes while a frame is in progress: reported with the next frame.
                    self.phase[0] = "inserting"
                example.step()
                if call == 3:
                    self.assertEqual(output.getvalue().count("Procedural insertion"), 1)
        after = time.perf_counter()
        frame = example.config.frame_dt
        self.assertEqual(example.completed, 4)
        self.assertEqual(example.unfinished_calls, 3)
        self.assertAlmostEqual(example.sim_time, 4 * frame, places=12)
        # The controller saw the state after each complete frame, never one in the middle of a frame.
        np.testing.assert_allclose(self.controlled, [0.0, frame, 2 * frame, 3 * frame], rtol=0, atol=1e-12)
        self.assertEqual(output.getvalue().count("Procedural insertion"), 2)
        given = [actions is not None for actions, _ in example.pool.steps]
        self.assertEqual(given, [True, True, False, False, True, True, False])
        for _, options in example.pool.steps:
            # The scene may take the budget less what the longest drawing took.
            self.assertGreaterEqual(options["deadline"], before + 1.0 / 15.0 - 0.008)
            self.assertLessEqual(options["deadline"], after + 1.0 / 15.0 - 0.008)

    def test_the_single_step_of_a_paused_viewer_has_no_deadline(self):
        example = self.example([2, 1, 2], 1.0 / 15.0)
        with contextlib.redirect_stdout(io.StringIO()):
            example.step()
            self.assertEqual((example.completed, example.unfinished_calls), (0, 1))
            # Paused in the middle of a frame: the run loop calls step() once per "." or Step.
            example.viewer.paused = True
            example.step()
            example.step()
            example.viewer.paused = False
            example.step()
        deadlines = ["deadline" in options for _, options in example.pool.steps]
        self.assertEqual(deadlines, [True, False, False, True])
        # The single step went on with the frame in progress, under the action that started it.
        self.assertEqual([actions is not None for actions, _ in example.pool.steps], [True, False, True, True])
        for _, options in example.pool.steps[1:3]:
            self.assertEqual(options, {"include_tissue": False, "include_thread_diagnostics": False})
        # A viewer that cannot be paused, or says nothing about it, is never single-stepped.
        self.assertFalse(example._single_step())
        example.viewer = SimpleNamespace()
        self.assertFalse(example._single_step())
        example.viewer = _GLViewer(paused=True)
        self.assertTrue(example._single_step())
        example.viewer = _other(newton.viewer.ViewerNull)
        self.assertFalse(example._single_step())

    def test_without_a_budget_the_pool_is_stepped_as_before(self):
        for paused in (False, True):
            example = self.example([1, 1], None)
            example.viewer.paused = paused
            with contextlib.redirect_stdout(io.StringIO()):
                example.step()
                example.step()
            self.assertEqual(example.completed, 2)
            self.assertEqual(example.unfinished_calls, 0)
            for actions, options in example.pool.steps:
                self.assertIsNotNone(actions)
                self.assertEqual(options, {"include_tissue": False, "include_thread_diagnostics": False})

    def test_a_complete_frame_of_the_wrong_length_is_refused_and_an_unfinished_one_is_not(self):
        example = self.example([2], 1.0 / 15.0)
        with contextlib.redirect_stdout(io.StringIO()):
            example.step()
            self.assertEqual((example.completed, example.unfinished_calls), (0, 1))
            example.pool.time += 0.001
            with self.assertRaisesRegex(RuntimeError, "complete requested frame"):
                example.step()

    def test_the_drawing_is_timed_for_the_next_step(self):
        example = self.example([1], 1.0 / 15.0)
        example.viewer = SimpleNamespace(begin_frame=lambda time: None, end_frame=lambda: time.sleep(0.002))
        self.assertEqual(example._draw_longest, 0.0)
        example.render()
        self.assertGreaterEqual(example._draw_longest, 0.002)
        self.assertLess(example._draw_longest, 0.2)

    def test_a_step_leaves_the_longest_drawing_free_not_the_last_one(self):
        example = self.example([1, 1], 1.0 / 15.0)
        clock, drawing = [50.0], [0.25]

        def end_frame():
            # The drawing takes the time the test gives it, on the clock the test moves.
            clock[0] += drawing[0]

        example.viewer = SimpleNamespace(begin_frame=lambda time: None, end_frame=end_frame, is_paused=lambda: False)
        with patch.object(time, "perf_counter", lambda: clock[0]), contextlib.redirect_stdout(io.StringIO()):
            example.render()
            self.assertEqual(example._draw_longest, 0.25)
            # A shorter drawing: the last one is timed, the longest one stays.
            drawing[0] = 0.125
            example.render()
            self.assertEqual(example._draw_longest, 0.25)
            example.step()
            self.assertEqual(example.pool.steps[-1][1]["deadline"], clock[0] + 1.0 / 15.0 - 0.25)
            # A longer one counts from the step after it.
            drawing[0] = 0.5
            example.render()
            self.assertEqual(example._draw_longest, 0.5)
            example.step()
            self.assertEqual(example.pool.steps[-1][1]["deadline"], clock[0] + 1.0 / 15.0 - 0.5)
        # An example that has not drawn leaves nothing free: nothing is measured.
        self.assertEqual(grasp_example.Example._draw_longest, 0.0)

    def test_the_budget_is_66_7_ms_by_default_and_given_in_milliseconds(self):
        parser = grasp_example.Example.create_parser()
        # 15 pictures per second; the parser takes the constant of the module.
        self.assertEqual(grasp_example.FRAME_BUDGET_MS, 66.7)
        self.assertEqual(parser.parse_args([]).frame_budget, grasp_example.FRAME_BUDGET_MS)
        self.assertEqual(parser.get_default("frame_budget"), grasp_example.FRAME_BUDGET_MS)
        # Zero is the switch for whole frames.
        self.assertEqual(parser.parse_args(["--frame-budget", "0"]).frame_budget, 0.0)
        self.assertEqual(parser.parse_args(["--frame-budget", "50"]).frame_budget, 50.0)
        # An example that has not been constructed has no budget: the constructor decides where one is used.
        self.assertIsNone(grasp_example.Example.frame_budget)
        self.assertEqual(parser.parse_args(["--frame-budget", "66.7"]).frame_budget, 66.7)
        for arguments in (["--frame-budget", "-1"], ["--frame-budget", "nan"], ["--frame-budget", "inf"]):
            with (
                self.subTest(arguments=arguments),
                patch("sys.argv", ["grasp-example", *arguments]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                grasp_example.main()

    def test_the_budget_is_used_in_the_window_of_the_opengl_viewer_only(self):
        window = _GLViewer()

        def budget(viewer, **settings):
            values = {"num_envs": 1, "frame_budget": 66.7, "test": False} | settings
            if values["frame_budget"] is None:
                # Arguments without the option (other drivers of the class).
                del values["frame_budget"]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                return grasp_example.Example._frame_budget(viewer, SimpleNamespace(**values)), output.getvalue()

        self.assertTrue(grasp_example.Example._gl_window(window))
        seconds, text = budget(window)
        self.assertAlmostEqual(seconds, 0.0667, places=12)
        self.assertIn("Frame budget 66.7 ms", text)
        # A paused window has the budget too: its single steps are told apart in step().
        self.assertAlmostEqual(budget(_GLViewer(paused=True))[0], 0.0667, places=12)
        usd = _other(newton.viewer.ViewerUSD, num_frames=2400)
        # The rule is the same for every budget; a budget other than the default says why it is not used.
        self.assertNotEqual(grasp_example.FRAME_BUDGET_MS, 50.0)
        self.assertAlmostEqual(budget(window, frame_budget=50.0)[0], 0.05, places=12)
        for viewer, settings, reason in (
            (window, {"test": True}, "self-test"),
            (window, {"num_envs": 2}, "several scenes"),
            # Viewers that end the run after a number of calls.
            (_other(newton.viewer.ViewerNull, num_frames=2400), {}, "ends the run"),
            (_GLViewer(headless=True), {}, "ends the run"),
            (usd, {}, "ends the run"),
            (SimpleNamespace(num_frames=2400), {}, "ends the run"),
            # Viewers that record or send every call, and the RTX window.
            (_other(newton.viewer.ViewerRerun), {}, "OpenGL viewer; this viewer (ViewerRerun)"),
            (_other(newton.viewer.ViewerViser), {}, "OpenGL viewer; this viewer (ViewerViser)"),
            (_other(newton.viewer.ViewerRTX, _headless=False, num_frames=2400), {}, "this viewer (ViewerRTX)"),
            (_other(newton.viewer.ViewerUSD, num_frames=None), {}, "this viewer (ViewerUSD)"),
            # Anything else that does not count its frames: not the OpenGL window either.
            (SimpleNamespace(num_frames=None), {}, "OpenGL viewer"),
            (SimpleNamespace(num_frames=2400, _headless=False), {}, "OpenGL viewer"),
        ):
            with self.subTest(viewer=type(viewer).__name__, reason=reason):
                seconds, text = budget(viewer, frame_budget=50.0, **settings)
                self.assertIsNone(seconds)
                self.assertIn("Frame budget not used", text)
                self.assertIn(reason, text)
                # The default budget is not used there either and says nothing, written out or not:
                # a run that keeps whole frames did not ask for it.
                self.assertEqual(budget(viewer, **settings), (None, ""))
                self.assertEqual(budget(viewer, frame_budget=None, **settings), (None, ""))
        # With the budget switched off nothing is said, wherever the example runs.
        self.assertEqual(budget(window, frame_budget=0.0), (None, ""))
        self.assertEqual(budget(usd, frame_budget=0.0), (None, ""))
        # Arguments without the option (other drivers) have the example's default: in the window the budget.
        seconds, text = budget(window, frame_budget=None)
        self.assertEqual(seconds, 1.0e-3 * grasp_example.FRAME_BUDGET_MS)
        self.assertEqual(text, budget(window)[1])

    def test_the_window_names_the_budget_in_one_line_with_the_option_that_turns_it_off(self):
        parser = grasp_example.Example.create_parser()
        texts = []
        for args in (parser.parse_args([]), parser.parse_args(["--frame-budget", "66.7"]), SimpleNamespace(num_envs=1)):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                seconds = grasp_example.Example._frame_budget(_GLViewer(), args)
            self.assertAlmostEqual(seconds, 0.0667, places=12)
            texts.append(output.getvalue())
        # The default, the default written out and arguments without the option: the same one line.
        self.assertEqual(len(set(texts)), 1)
        lines = texts[0].splitlines()
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("Frame budget 66.7 ms:"), lines)
        self.assertIn("--frame-budget 0 runs whole frames", lines[0])
        self.assertNotIn("not used", lines[0])
        # A run that asked for nothing and keeps whole frames says nothing about the budget.
        for arguments, viewer in (
            (["--test"], _GLViewer()),
            (["--num-envs", "2"], _GLViewer()),
            (["--viewer", "null"], _other(newton.viewer.ViewerNull, num_frames=2400)),
            (["--headless"], _GLViewer(headless=True)),
        ):
            with self.subTest(arguments=arguments):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertIsNone(grasp_example.Example._frame_budget(viewer, parser.parse_args(arguments)))
                self.assertEqual(output.getvalue(), "")

    def test_the_example_takes_its_budget_from_the_option(self):
        pool = SimpleNamespace(
            observe=lambda: [{"diagnostics": {"physical_time_s": 0.0}}],
            geometry=lambda: {},
            finish_startup=lambda: None,
            close=lambda: None,
        )

        def args(**settings):
            defaults = dict(
                device="cpu", num_envs=1, substeps=4, controller="hold", rigid_backend="native", viewer="null",
                speed=0.003, settle_seconds=0.5, spacing=0.45,
            )  # fmt: skip
            return SimpleNamespace(**(defaults | settings))

        with (
            patch.object(grasp_example, "LocalSutureScene", lambda config: pool),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            # Arguments without the option have the example's default budget.
            default = 1.0e-3 * grasp_example.FRAME_BUDGET_MS
            for settings, expected in (({}, default), ({"frame_budget": 0.0}, None), ({"frame_budget": 50.0}, 0.05)):
                example = grasp_example.Example(_GLViewer(), args(**settings))
                self.assertEqual(example.frame_budget, expected)
                example.close()
            # A viewer that counts its frames, and a viewer that is not the OpenGL window, keep whole frames.
            for viewer in (SimpleNamespace(num_frames=10), SimpleNamespace(), _other(newton.viewer.ViewerRerun)):
                example = grasp_example.Example(viewer, args(frame_budget=50.0))
                self.assertIsNone(example.frame_budget)
                example.close()


class RecordedCallTests(unittest.TestCase):
    """The one interval over 66.7 ms of the window run w_def_1950 (2026-11-17, docs/usage.md): its call 2352.

    The call ran an attempt of 5 trials and asked for one more 14.007 ms after its start. By the plan
    of then (mean times, the last observation and the last drawing; no code of it is left) the
    attempt fitted, by 0.87 ms. It ran 13 trials and ended inside its bound; the observation after
    it took 4.66 ms longer than the last one, and the next call started 67.12 ms after this one.
    Times in seconds of the host clock of that run, the call starting at 0; the measurements are
    those the run had made by then.
    """

    BUDGET = 0.0667
    TRIAL_LIMIT = 13
    QUESTION = 0.014007
    # The substeps the scene had measured: their number, their trials, the time of their loops, their time.
    SUBSTEPS, TRIALS, LOOP_SECONDS, SECONDS = 3632, 8011, 26.552221, 37.471088
    # The longest trial loop (13 trials) and the longest time of a substep outside its loops (the run's first).
    LONGEST_LOOP, LONGEST_OUTSIDE = 0.042900, 0.018495
    LAST_OBSERVATION, LONGEST_OBSERVATION = 0.001168, 0.008381
    LAST_DRAWING, LONGEST_DRAWING = 0.004564, 0.009587
    # The longest host-controlled trial and the longest end of an attempt.
    LONGEST_HOST, LONGEST_TAIL = 0.005680, 0.015915
    # What followed the question in the run: the attempt to its end, the observation, the drawing.
    ATTEMPT, OBSERVATION, DRAWING = 0.042569, 0.005827, 0.004663
    # What the plan of then took one more attempt to be: the mean time of a substep outside its
    # loops and 13 mean trials (46.094 ms).
    MEAN_PLAN = (SECONDS - LOOP_SECONDS) / SUBSTEPS + TRIAL_LIMIT * LOOP_SECONDS / TRIALS

    def measured(self):
        bound = SubstepBound()
        bound.substeps, bound.trials, bound.loop_seconds = self.SUBSTEPS, self.TRIALS, self.LOOP_SECONDS
        bound.longest_loop, bound.longest_outside = self.LONGEST_LOOP, self.LONGEST_OUTSIDE
        return bound

    def scene_deadline(self, start):
        """The deadline the scene gets from a step of the example over a local scene that starts at ``start``."""
        deadlines = []
        scene = SimpleNamespace(
            actual_tip_history=[], frame_pending=True, close=lambda: None, viewer=SimpleNamespace(close=lambda: None)
        )
        observation = {"diagnostics": {"physical_time_s": 0.0}}
        with (
            patch.object(parallel, "_make_scene", lambda config, env_id: scene),
            patch.object(parallel, "_observe", lambda scene, env_id, **options: dict(observation)),
            patch.object(parallel, "_step", lambda scene, deadline: deadlines.append(deadline)),
            LocalSutureScene() as scenes,
        ):
            scenes._observe_longest = self.LONGEST_OBSERVATION
            example = object.__new__(grasp_example.Example)
            example.pool, example.config, example.viewer = scenes, scenes.config, _Viewer()
            example.halted = example.render_scene = example.controller = None
            example.sim_time, example.completed = 0.0, 0
            example.frame_budget, example.frame_unit = self.BUDGET, "attempt"
            example._draw_longest = self.LONGEST_DRAWING
            # The call goes on with a frame in progress, as the recorded one left its frame unfinished.
            example._frame_start = [0.0]
            with patch.object(time, "perf_counter", return_value=start):
                example.step()
            self.assertEqual(example.unfinished_calls, 1)
        (deadline,) = deadlines
        return deadline

    def test_the_worst_case_measured_refuses_the_attempt(self):
        bound, start = self.measured(), 100.0
        # The bound of one more attempt: 18.495 ms outside its loops and a loop of 13 mean trials
        # (43.088 ms, longer than the longest loop of the run so far, 42.900 ms).
        worst = bound.bound(self.TRIAL_LIMIT)
        self.assertAlmostEqual(worst, self.LONGEST_OUTSIDE + self.TRIAL_LIMIT * bound.per_trial, places=12)
        self.assertGreater(self.TRIAL_LIMIT * bound.per_trial, self.LONGEST_LOOP)
        self.assertAlmostEqual(worst, 0.061583, places=6)
        # The scene gets the start of the call plus the budget, less the longest drawing and the
        # longest observation: 48.732 ms after the start.
        deadline = self.scene_deadline(start)
        self.assertAlmostEqual(
            deadline - start, self.BUDGET - self.LONGEST_DRAWING - self.LONGEST_OBSERVATION, places=12
        )
        self.assertAlmostEqual(deadline - start, 0.048732, places=9)
        # The scene's rule on that bound and that deadline: refused, by 26.86 ms.
        asked = SimpleNamespace(_substep_bound=bound, _trial_limit=lambda: self.TRIAL_LIMIT)
        fits = insertion.Example._attempt_fits
        self.assertFalse(fits(asked, start + self.QUESTION, deadline))
        self.assertAlmostEqual(start + self.QUESTION + worst - deadline, 0.026858, places=6)
        # The same question fits where the worst case leaves room for it, to the last second of the clock.
        self.assertTrue(fits(asked, deadline - worst, deadline))
        self.assertFalse(fits(asked, np.nextafter(deadline - worst, np.inf), deadline))
        # Each of the three terms alone refuses the recorded attempt, which the plan of then let
        # start (it fitted by 0.87 ms): the longest observation, the longest drawing, the worst
        # case of the substep.
        mean = self.MEAN_PLAN
        self.assertAlmostEqual(mean, 0.046094, places=6)
        self.assertLess(self.QUESTION + mean, self.BUDGET - self.LAST_DRAWING - self.LAST_OBSERVATION)
        self.assertGreater(self.QUESTION + mean, self.BUDGET - self.LAST_DRAWING - self.LONGEST_OBSERVATION)
        self.assertGreater(self.QUESTION + mean, self.BUDGET - self.LONGEST_DRAWING - self.LAST_OBSERVATION)
        self.assertGreater(self.QUESTION + worst, self.BUDGET - self.LAST_DRAWING - self.LAST_OBSERVATION)
        # And what the run then measured of the attempt was inside the worst case, part by part.
        self.assertLess(self.ATTEMPT, worst)
        self.assertLess(self.OBSERVATION, self.LONGEST_OBSERVATION)
        self.assertLess(self.DRAWING, self.LONGEST_DRAWING)

    def test_a_host_controlled_trial_is_asked_by_the_same_deadline(self):
        start, now = 100.0, [100.0]
        budget = HostTrialBudget(lambda: now[0])
        budget.longest_host, budget.longest_tail = self.LONGEST_HOST, self.LONGEST_TAIL
        deadline = self.scene_deadline(start)
        # A trial asked 27 ms into the call ends by the deadline with the longest end of an attempt
        # after it (48.595 ms), and so, in the worst case measured, with the observation and the
        # drawing after that inside the budget.
        budget.begin_call(deadline)
        now[0] = start + 0.027
        self.assertTrue(budget.admit("host"))
        worst = 0.027 + budget.bound() + self.LONGEST_OBSERVATION + self.LONGEST_DRAWING
        self.assertLessEqual(worst, self.BUDGET)
        # One asked 30 ms into the call is refused. By the last observation and the last drawing it
        # started, and the worst case measured is 2.9 ms over the budget.
        budget.begin_call(deadline)
        now[0] = start + 0.030
        self.assertFalse(budget.admit("host"))
        budget.begin_call(start + self.BUDGET - self.LAST_DRAWING - self.LAST_OBSERVATION)
        self.assertTrue(budget.admit("host"))
        worst = 0.030 + budget.bound() + self.LONGEST_OBSERVATION + self.LONGEST_DRAWING
        self.assertAlmostEqual(worst - self.BUDGET, 0.002863, places=6)


class ExampleSceneBudgetTests(unittest.TestCase):
    """The example over its real scene: the procedure, the report and the observations per frame."""

    def run_example(self, budget, frames, single_steps=False):
        """Run complete frames; with ``single_steps`` every second call is the single step of a paused viewer."""
        with (
            TemporaryDirectory() as directory,
            patch.object(phantom, "list_mesh_regions", return_value=_REFERENCE.regions),
            patch.object(phantom, "load_mesh_regions", side_effect=_selection),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            # The example's own scene settings (its coupling tolerance and trial budget) on the small mesh.
            small = {"asset": str(Path(directory) / "pad.usd"), "entry_xy": (0, 0.005), "exit_xy": (0, -0.005)}
            args = SimpleNamespace(
                device=DEVICE, num_envs=1, substeps=4, controller="procedural", rigid_backend="native", viewer="null",
                speed=0.003, settle_seconds=0.02, spacing=0.45, lift_height=0.0,
            )  # fmt: skip
            with patch.object(
                grasp_example, "SutureSceneConfig", lambda **settings: SutureSceneConfig(**(settings | small))
            ):
                example = grasp_example.Example(_Viewer(), args)
            try:
                example.frame_budget = budget
                record, calls = [], 0
                while example.completed < frames:
                    before = example.completed
                    example.viewer.paused = single_steps and calls % 2 == 1
                    example.step()
                    calls += 1
                    if example.viewer.paused:
                        # The single step of a paused viewer ends on a complete frame.
                        self.assertFalse(example.pool.frame_unfinished)
                        self.assertEqual(example.completed, before + 1)
                    if example.completed > before:
                        record.append((example.observations[0], example.controller.report()[0], example.sim_time))
                    self.assertLess(calls, CALL_LIMIT * frames)
                return record, calls, example.unfinished_calls
            finally:
                example.close()

    def test_calls_of_the_procedure_under_a_budget_run_one_attempt_each(self):
        frames = 4
        # A budget that has always run out: every call runs one attempt.
        split, calls, unfinished = self.run_example(1.0e-9, frames)
        diagnostics = split[-1][0]["diagnostics"]
        self.assertEqual(calls - unfinished, frames)
        self.assertEqual(calls, diagnostics["accepted_physics_steps"] + diagnostics["step_refinements"])
        self.assertAlmostEqual(split[-1][2], frames / 60.0, delta=1e-9)

    def test_the_single_step_of_a_paused_viewer_finishes_its_frame(self):
        frames = 3
        # One call under a budget that has run out, then one single step: two calls per frame.
        record, calls, unfinished = self.run_example(1.0e-9, frames, single_steps=True)
        self.assertEqual((calls, unfinished), (2 * frames, frames))
        self.assertAlmostEqual(record[-1][2], frames / 60.0, delta=1e-9)

    @unittest.skipIf(_CUDA, _BITS)
    def test_frames_of_the_procedure_under_a_budget_are_those_of_whole_frames(self):
        frames = 8
        whole, whole_calls, none = self.run_example(None, frames)
        split, split_calls, unfinished = self.run_example(1.0e-9, frames)
        stepped, _, _ = self.run_example(1.0e-9, frames, single_steps=True)
        self.assertEqual((whole_calls, none), (frames, 0))
        self.assertEqual((split_calls, unfinished), (4 * frames, 3 * frames))
        for other in (split, stepped):
            for frame, (expected, actual) in enumerate(zip(whole, other, strict=True)):
                # The observation the controller acts on, the controller's report and the time.
                _same_value(self, expected[0], actual[0], f"frame {frame} observation")
                _same_value(self, expected[1], actual[1], f"frame {frame} report")
                self.assertEqual(expected[2], actual[2])
        # The procedure moved on: the settling ended within these frames.
        self.assertNotEqual(whole[0][1]["phase"], whole[-1][1]["phase"])


if __name__ == "__main__":
    unittest.main()
