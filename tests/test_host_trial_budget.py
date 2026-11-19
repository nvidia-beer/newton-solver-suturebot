# SPDX-License-Identifier: Apache-2.0
"""The unit ``host-trial`` of the frame budget: an attempt left between two host-controlled trials.

A call with a deadline and the unit ``host-trial`` may return inside an attempt of a coupled
step, before a host-controlled trial that does not fit; the next call goes on with that trial.
What is about the calls (where they return, what they keep, what an observation shows, the
rule, where the unit applies) runs on every device. The comparisons of two runs bit for bit
run on the CPU only: two runs of the same frames on a CUDA device are not bit-identical.

The scene of these tests has the native rigid backend: every trial loop is host-controlled.
"""

import contextlib
import functools
import gc
import io
import random
import time
import unittest
import weakref
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import newton.viewer
import numpy as np
from test_frame_budget import (
    _BITS,
    _CUDA,
    _UNSTABLE,
    ACTIONS,
    CALL_LIMIT,
    DEVICE,
    PASSED,
    _GLViewer,
    _local,
    _other,
    _Pool,
    _rejecting,
    _run,
    _same_value,
    _snapshot,
    _Viewer,
)
from test_grasp_scene import _native, _scene
from test_needle_insertion_example import _REFERENCE, _selection
from test_needle_trial_budget import _DT, _coupled, _record_trials, _residuals, _scale_correction

from newton_solver_suturebot import NeedleStepTooLarge, NeedleTissueCoupler
from newton_solver_suturebot._src.utils.frame_budget import HostTrialBudget
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

# What may differ between the observation of a call that left an attempt open and the observation
# of the call before that attempt began, each with its reason. Everything else is equal: every
# array bit for bit, every report entry.
#
# Counts of work that the attempt in progress has done so far (launches, preparations, attempts
# of helpers, loops that the trial budget ended, the Anderson counters; captures and storage
# growths where a scene makes its graphs at their first use, as the scene of these tests does on
# a CUDA device). They describe work, not the accepted state.
WORK_COUNTS = (
    "authorized_outside_certified_trials",
    "certified_linear_attempts",
    "certified_linear_rejections",
    "contact_preparation_captures",
    "contact_preparation_eager",
    "contact_preparation_launches",
    "cooperative_shaft_rebinds",
    "coupling_budget_keeps",
    "coupling_budget_searches",
    "coupling_divergence_stops",
    "coupling_graph_event_restarts",
    "coupling_graph_event_searches",
    "coupling_graph_fallback_reasons",
    "coupling_graph_captures",
    "coupling_graph_fallbacks",
    "coupling_graph_launches",
    "coupling_graph_verifications",
    "device_anderson_max_active_rows",
    "device_anderson_overflow_packets",
    "device_anderson_packet_capacity",
    "device_anderson_transport_bytes",
    "device_anderson_transport_calls",
    "fem_graph_captures",
    "fem_graph_launches",
    "gpu_coupling_anderson_counters",
    "gpu_rigid_diagnostics",
    "held_fem_start_preparations",
    "host_force_graph_captures",
    "host_force_graph_fallbacks",
    "host_force_graph_launches",
    "host_force_primal_reuses",
    "native_batches",
    "native_general_intervals",
    "native_guard_microsteps",
    "native_guard_rejections",
    "native_host_wrench_reuses",
    "native_transfer_graph_captures",
    "native_transfer_graph_launches",
    "native_transfer_readbacks",
    "native_transfer_repeated_readbacks",
    "native_transfer_uploads",
    "phase_predictor_attempts",
    "phase_predictor_invalidations",
    "response_basis_builds",
    "response_reference_builds",
    "shaft_pc_graph_captures",
    "shaft_pc_storage_growths",
    "surface_pc_applications",
    "surface_pc_graph_captures",
    "surface_pc_graph_launches",
    "surface_pc_native_mass_fallbacks",
    "surface_pc_selection_changes",
    "trial_prepare_graph_captures",
    "trial_prepare_graph_launches",
    "tunnel_alm_dual_updates",
    "woodbury_packet_preparation_captures",
    "woodbury_packet_preparation_eager",
    "woodbury_packet_preparation_launches",
)
# Commands that the preparation of the attempt in progress has written: the arm joint targets
# of the interval it solves, and what its inverse kinematics reported.
ATTEMPT_COMMANDS = ("arm_joint_targets", "ik_diagnostics")
# Commands of the frame, set by the action that started it (the first call of a frame).
FRAME_COMMANDS = ("target_pose", "gripper_target_pose", "jaw_target_rad")
# Read from the rigid data of the native backend, which hold a trial while an attempt is open:
# the contacts between the jaws and the needle (their normal force and their count), the
# native clock, and what the guard of the native interval refused last. The GPU backend reads
# the accepted contact metrics, which only an acceptance writes, and has no native clock.
NATIVE_READS = (
    "jaw_contact_normal_force_n",
    "jaw_needle_contact_count",
    "jaw_contact_count",
    "native_clock_s",
    "native_clock_error_s",
    "native_guard_last_failure",
)
_MAY_DIFFER = frozenset((*WORK_COUNTS, *ATTEMPT_COMMANDS, *FRAME_COMMANDS, *NATIVE_READS, *_UNSTABLE))

# The report entries that describe the last attempt.
ATTEMPT_REPORT = (
    "last_coupling_attempt_iterations",
    "last_coupling_attempt_residual_N",
    "fem_graph_status",
    "coupling_graph_status",
    "shaft_preconditioner_status",
    "last_interface_load_on_fixed_nodes_N",
    "tunnel_alm_last_normal_error_m",
    "tunnel_alm_last_dual_residual_N",
    "status",
)
STATE = ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")


class _Clock:
    """A clock that the test moves."""

    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


class HostTrialBudgetTests(unittest.TestCase):
    def measured(self, host=0.004, tail=0.003):
        """A budget that has measured one host-controlled trial and one tail of an attempt."""
        clock = _Clock()
        budget = HostTrialBudget(clock)
        budget.begin_call(None)
        self.assertTrue(budget.admit("host"))
        clock.now += host
        self.assertTrue(budget.admit("end"))
        clock.now += tail
        budget.close()
        return budget, clock

    def test_without_a_deadline_everything_is_admitted(self):
        budget = HostTrialBudget(_Clock())
        budget.begin_call(None)
        # Nothing is measured yet, and nothing has to be.
        self.assertIsNone(budget.bound())
        for kind in ("host", "host", "end", "host"):
            self.assertTrue(budget.admit(kind))
        budget, clock = self.measured()
        budget.begin_call(None)
        clock.now += 1.0e6
        self.assertTrue(budget.admit("host"))
        self.assertEqual(budget.refused, {})

    def test_a_host_trial_starts_only_if_the_longest_trial_and_the_longest_tail_end_by_the_deadline(self):
        budget, clock = self.measured(host=0.004, tail=0.003)
        self.assertAlmostEqual(budget.longest_host, 0.004, places=12)
        self.assertAlmostEqual(budget.longest_tail, 0.003, places=12)
        bound = budget.bound()
        self.assertEqual(bound, budget.longest_host + budget.longest_tail)
        # The clock stands still. Admitted exactly at now + host + tail == deadline ...
        budget.begin_call(clock.now + bound)
        self.assertTrue(budget.admit("host"))
        # ... and refused one tick of the clock's resolution later.
        budget.begin_call(clock.now + bound)
        clock.now = np.nextafter(clock.now, np.inf)
        self.assertFalse(budget.admit("host"))
        # The tail is part of the bound: a deadline that the trial alone would meet is refused.
        budget.begin_call(clock.now + budget.longest_host)
        self.assertFalse(budget.admit("host"))
        budget.begin_call(clock.now + budget.longest_tail)
        self.assertFalse(budget.admit("host"))

    def test_before_a_trial_and_a_tail_were_measured_a_trial_is_admitted_only_as_the_first_piece_of_a_resumed_call(
        self,
    ):
        clock = _Clock()
        far = clock.now + 1.0e6
        budget = HostTrialBudget(clock)
        # Nothing measured: no bound, whatever the deadline says.
        budget.begin_call(far)
        self.assertIsNone(budget.bound())
        self.assertFalse(budget.admit("host"))
        budget.begin_call(far, resumed=True)
        self.assertTrue(budget.admit("host"))
        clock.now += 0.004
        # A trial is measured, a tail is not: still no bound.
        self.assertFalse(budget.admit("host"))
        self.assertEqual(budget.longest_host, clock.now - 100.0)
        self.assertIsNone(budget.longest_tail)
        self.assertIsNone(budget.bound())
        # A tail alone is no bound either.
        other = HostTrialBudget(clock)
        other.begin_call(far)
        self.assertTrue(other.admit("end"))
        clock.now += 0.003
        other.close()
        self.assertIsNone(other.longest_host)
        self.assertIsNotNone(other.longest_tail)
        other.begin_call(far)
        self.assertFalse(other.admit("host"))
        # With both, a deadline far away admits.
        budget.begin_call(far, resumed=True)
        self.assertTrue(budget.admit("end"))
        clock.now += 0.003
        budget.close()
        budget.begin_call(far)
        self.assertTrue(budget.admit("host"))

    def test_the_first_piece_of_a_resumed_call_is_admitted_whatever_the_deadline_says(self):
        budget, clock = self.measured()
        passed = clock.now - 1.0
        budget.begin_call(passed, resumed=True)
        self.assertTrue(budget.admit("host"))
        # The first piece only: the second question of the call asks the bound.
        self.assertFalse(budget.admit("host"))
        # A call that was not resumed has no such piece.
        budget.begin_call(passed)
        self.assertFalse(budget.admit("host"))
        # A new call that goes on has it again, and a refused question does not use it up.
        budget.begin_call(passed, resumed=True)
        self.assertTrue(budget.admit("host"))
        self.assertEqual(budget.admitted["host"], 3)
        self.assertEqual(budget.refused["host"], 2)

    def test_a_trial_lasts_to_the_next_question_and_a_tail_to_the_close(self):
        clock = _Clock()
        budget = HostTrialBudget(clock)
        budget.begin_call(None)
        # Three trials of 2, 5 and 3 ms: each lasts to the next question.
        for seconds in (0.002, 0.005, 0.003):
            self.assertTrue(budget.admit("host"))
            clock.now += seconds
        self.assertTrue(budget.admit("end"))
        self.assertAlmostEqual(budget.longest_host, 0.005, places=12)
        self.assertIsNone(budget.longest_tail)
        # The tail of the attempt: checks, commit, the work after the acceptance, to the close.
        clock.now += 0.004
        budget.close()
        self.assertAlmostEqual(budget.longest_tail, 0.004, places=12)
        # A close without a piece in progress measures nothing.
        clock.now += 1.0
        budget.close()
        self.assertAlmostEqual(budget.longest_tail, 0.004, places=12)
        # A trial that ends in a rejection lasts to the close after it.
        self.assertTrue(budget.admit("host"))
        clock.now += 0.007
        budget.close()
        self.assertAlmostEqual(budget.longest_host, 0.007, places=12)
        # The end of the frame is part of the tail of its last attempt: the tail lasts to the close.
        self.assertTrue(budget.admit("end"))
        clock.now += 0.006
        budget.close()
        self.assertAlmostEqual(budget.longest_tail, 0.006, places=12)
        # The longest ones are kept: shorter pieces change nothing.
        self.assertTrue(budget.admit("host"))
        clock.now += 0.001
        self.assertTrue(budget.admit("end"))
        clock.now += 0.001
        budget.close()
        self.assertAlmostEqual(budget.longest_host, 0.007, places=12)
        self.assertAlmostEqual(budget.longest_tail, 0.006, places=12)
        self.assertAlmostEqual(budget.bound(), 0.013, places=12)
        # A piece that a new call finds in progress is not measured across the calls.
        self.assertTrue(budget.admit("host"))
        clock.now += 5.0
        budget.begin_call(None)
        budget.close()
        self.assertAlmostEqual(budget.longest_host, 0.007, places=12)

    def test_the_end_of_an_attempt_is_never_refused(self):
        budget, clock = self.measured()
        budget.begin_call(clock.now - 1.0)
        self.assertFalse(budget.admit("host"))
        self.assertTrue(budget.admit("end"))
        # Before anything is measured as well.
        fresh = HostTrialBudget(clock)
        fresh.begin_call(clock.now - 1.0)
        self.assertTrue(fresh.admit("end"))
        self.assertEqual(fresh.refused, {})

    def test_admitted_and_refused_pieces_are_counted(self):
        budget, clock = self.measured()
        self.assertEqual((budget.admitted, budget.refused), ({"host": 1, "end": 1}, {}))
        budget.begin_call(clock.now - 1.0, resumed=True)
        self.assertTrue(budget.admit("host"))
        self.assertFalse(budget.admit("host"))
        self.assertFalse(budget.admit("host"))
        self.assertTrue(budget.admit("end"))
        self.assertEqual((budget.admitted, budget.refused), ({"host": 2, "end": 2}, {"host": 2}))
        self.assertEqual(budget.deadline, clock.now - 1.0)


class _Script:
    """A budget with the interface of :class:`HostTrialBudget` whose answers follow a script, not a clock.

    ``every``   admits only the first piece of a call that goes on with an open attempt: an
                attempt is left before every host-controlled trial and before its end.
    ``trials``  the same for the host-controlled trials; the end of an attempt is never refused.
    ``coin``    the first piece of a call that goes on, and every other piece by a seeded coin.
    ``never``   refuses nothing.
    ``refuse``  refuses everything, also the first piece of a call that goes on.
    """

    def __init__(self, script, seed=0):
        self.script, self.rng = script, random.Random(seed)
        self.asked = []
        self.calls = []
        self.closes = 0
        self.deadline = None
        self._first = False

    def begin_call(self, deadline, resumed=False):
        self.deadline = deadline
        self.calls.append(bool(resumed))
        self._first = bool(resumed)

    def admit(self, kind):
        if self.script == "refuse":
            admitted = False
        elif self._first or self.script == "never" or (self.script == "trials" and kind == "end"):
            admitted = True
        else:
            admitted = self.script == "coin" and self.rng.random() < 0.5
        self._first = False
        self.asked.append((kind, admitted))
        return admitted

    def close(self):
        self.closes += 1

    def count(self, kind, admitted):
        return sum(1 for asked in self.asked if asked == (kind, admitted))


@contextmanager
def _counted(*names):
    """Count the calls of methods of the coupler, on the class: the coupler takes the paths it takes without."""
    counts = dict.fromkeys(names, 0)

    def counted(name, method):
        @functools.wraps(method)
        def wrapper(self, *arguments, **options):
            counts[name] += 1
            return method(self, *arguments, **options)

        return wrapper

    with ExitStack() as stack:
        for name in names:
            stack.enter_context(
                patch.object(NeedleTissueCoupler, name, counted(name, getattr(NeedleTissueCoupler, name)))
            )
        yield counts


def _drive(coupler, pieces, arguments, options=None):
    """One attempt through begin_step() and resume_step(); returns the calls it took."""
    pieces.begin_call(None)
    ended = coupler.begin_step(*arguments, pieces=pieces, **(options or {}))
    calls = 1
    while not ended:
        assert coupler.attempt_open, "an attempt that has not ended is open"
        assert calls < CALL_LIMIT, "an attempt makes no progress"
        pieces.begin_call(None, resumed=True)
        ended = coupler.resume_step()
        calls += 1
    assert not coupler.attempt_open
    return calls


def _arguments(example):
    return example.state_0, example.state_1, example.control, _DT


def _outcome(example, trials):
    """What one attempt of the preloaded shaft left behind, as host copies."""
    coupler, alm = example.coupler, example.interaction.tunnel_alm
    return {
        "counters": (
            coupler.iterations,
            coupler.evaluations,
            coupler.trial_loops,
            coupler.coupling_budget_keeps,
            coupler.residual,
            None if alm is None else alm.total_dual_updates,
        ),
        "residuals": np.asarray(_residuals(coupler, trials)),
        "trial_loads": np.asarray([load for _, load in trials]),
        "trial_load": coupler._trial_load.numpy().copy(),
        "load": coupler._load.numpy().copy(),
        **{name: getattr(example.state_1, name).numpy().copy() for name in STATE},
    }


class ResumableAttemptTests(unittest.TestCase):
    """The coupler: an attempt that begin_step() and resume_step() run in pieces.

    The coupler is the one of the trial-budget tests (a preloaded shaft, host-controlled
    trials: six in a loop without multipliers); the rejection and the abandon use the coupler
    of the grasp scene.
    """

    def assertSameOutcome(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for name, value in expected.items():
            with self.subTest(name=name):
                if isinstance(value, np.ndarray):
                    self.assertEqual(value.shape, actual[name].shape)
                    self.assertEqual(value.tobytes(), actual[name].tobytes())
                else:
                    self.assertEqual(value, actual[name])

    def test_step_without_pieces_never_pauses(self):
        with _coupled("penalty") as example, _counted("_pause", "_open_attempt", "_close_attempt") as counts:
            coupler = example.coupler
            self.assertFalse(coupler.attempt_open)
            self.assertIsNone(coupler.step(*_arguments(example)))
            self.assertFalse(coupler.attempt_open)
            # More than one host-controlled trial, and no question was asked.
            self.assertGreater(coupler.iterations, 1)
            self.assertEqual(counts, {"_pause": 0, "_open_attempt": 1, "_close_attempt": 1})
            self.assertEqual(coupler._paused_seconds, 0.0)
            # A budget that refuses nothing: one call as well, with every piece asked.
            pieces = _Script("never")
            self.assertEqual(_drive(coupler, pieces, _arguments(example)), 1)
            self.assertEqual(pieces.count("host", True), coupler.iterations)
            self.assertEqual(pieces.count("end", True), 1)
            self.assertEqual(counts["_close_attempt"], 2)

    def test_a_step_while_an_attempt_is_open_is_an_error(self):
        with _coupled("penalty") as example:
            coupler, arguments = example.coupler, _arguments(example)
            with self.assertRaisesRegex(AssertionError, "No attempt is open"):
                coupler.resume_step()
            pieces = _Script("every")
            pieces.begin_call(None)
            self.assertFalse(coupler.begin_step(*arguments, pieces=pieces))
            self.assertTrue(coupler.attempt_open)
            for call in (
                lambda: coupler.step(*arguments),
                lambda: coupler.begin_step(*arguments, pieces=pieces),
            ):
                with self.assertRaisesRegex(AssertionError, "An attempt is open"):
                    call()
            # The open attempt is untouched by the refused calls: it goes on and ends.
            self.assertTrue(coupler.attempt_open)
            calls = 1
            while coupler.attempt_open:
                pieces.begin_call(None, resumed=True)
                coupler.resume_step()
                calls += 1
            self.assertEqual(calls, pieces.count("host", True) + 2)
        # In a scene the error is not a failure of the physics: it reaches the caller.
        with _scene() as (scene, config):
            _apply_action(scene, config, ACTIONS[0])
            scene._host_budget = _Script("every")
            _step(scene, None, inside=True)
            self.assertTrue(scene.coupler.attempt_open)
            # A driver that lost track of the open attempt: the step of the needle scene starts a new frame.
            scene._pending_frame = scene._pending_attempt = None
            with self.assertRaisesRegex(AssertionError, "An attempt is open"):
                insertion.Example.step(scene)
            self.assertTrue(scene.running)
            self.assertEqual((scene.task_status, scene.failure_reason), ("running", None))
            scene.coupler.abandon_step()

    @unittest.skipIf(_CUDA, _BITS)
    def test_an_attempt_left_before_every_host_trial_is_the_attempt_of_one_call(self):
        names = ("_step_fem", "_evaluate", "_open_attempt", "_begin_attempt", "_close_attempt")
        with _coupled("penalty") as whole, _counted(*names) as expected:
            trials = _record_trials(whole.coupler)
            whole.coupler.step(*_arguments(whole))
            reference = _outcome(whole, trials)
            expected = dict(expected)
        with _coupled("penalty") as example, _counted(*names) as counts:
            coupler = example.coupler
            trials = _record_trials(coupler)
            pieces = _Script("every")
            calls = _drive(coupler, pieces, _arguments(example))
            # Left before each of its trials and before its end: one piece per call.
            self.assertEqual(calls, coupler.iterations + 2)
            self.assertEqual(pieces.count("host", False), coupler.iterations)
            self.assertEqual(pieces.count("end", False), 1)
            self.assertEqual(dict(counts), expected)
            self.assertEqual(counts["_open_attempt"], 1)
            self.assertGreater(coupler.iterations, 5)
            self.assertSameOutcome(reference, _outcome(example, trials))

    @unittest.skipIf(_CUDA, _BITS)
    def test_a_host_loop_under_the_trial_budget_left_between_trials(self):
        # The host-loop cases of the trial budget: the last trial kept, an earlier best load
        # applied once more, the best trial restarted at a multiplier update, and a trial that
        # only waits for a multiplier update kept without it.
        cases = (
            ("kept", "penalty", 3, None, (3, 1, None)),
            ("best load again", "penalty", 3, (2, -2.0), (4, 1, None)),
            ("multiplier update", "alm", 6, None, (6, 1, 1)),
            ("waits for an update", "alm", 4, None, (4, 1, 0)),
        )
        for name, mode, budget, scaled, (iterations, keeps, updates) in cases:
            outcomes = []
            for script in (None, "every", "trials", "coin"):
                with self.subTest(case=name, script=script), _coupled(mode, trial_budget=budget) as example:
                    coupler = example.coupler
                    if scaled is not None:
                        _scale_correction(coupler, *scaled)
                    trials = _record_trials(coupler)
                    if script is None:
                        coupler.step(*_arguments(example))
                    else:
                        calls = _drive(coupler, _Script(script, seed=3), _arguments(example))
                        if script == "every":
                            self.assertEqual(calls, iterations + 2)
                        elif script == "trials":
                            self.assertEqual(calls, iterations + 1)
                    alm = example.interaction.tunnel_alm
                    self.assertEqual(
                        (coupler.iterations, coupler.coupling_budget_keeps, alm.total_dual_updates if alm else None),
                        (iterations, keeps, updates),
                    )
                    if alm is not None:
                        self.assertFalse(alm._open)
                    outcomes.append(_outcome(example, trials))
            for outcome in outcomes[1:]:
                with self.subTest(case=name):
                    self.assertSameOutcome(outcomes[0], outcome)

    def test_a_rejection_arrives_in_the_call_that_goes_on(self):
        with _scene() as (scene, config), _rejecting(scene, {0}) as checks, _counted("_close_attempt") as counts:
            arguments, options, coupler = _prepared(scene, config)
            before = _attempt_start(scene)
            pieces = _Script("every")
            pieces.begin_call(None)
            self.assertFalse(coupler.begin_step(*arguments, pieces=pieces, **options))
            calls, raised = 1, False
            while not raised:
                self.assertLess(calls, CALL_LIMIT)
                self.assertTrue(coupler.attempt_open)
                self.assertEqual(counts["_close_attempt"], 0)
                pieces.begin_call(None, resumed=True)
                try:
                    # The check of the accepted step is in the last piece: no call ends the attempt before it.
                    self.assertFalse(coupler.resume_step())
                except NeedleStepTooLarge as error:
                    self.assertIn("rejected by the test", str(error))
                    raised = True
                calls += 1
            self.assertEqual(checks[0], 1)
            # The rejection came out of the call that ran the end of the attempt, after its trials:
            # one call began the attempt, one ran each trial, one the end.
            self.assertEqual(calls, pieces.count("host", True) + 2)
            self.assertEqual(pieces.count("end", True), 1)
            self.assertGreaterEqual(pieces.count("host", True), coupler.iterations)
            self.assertFalse(coupler.attempt_open)
            self.assertEqual(counts["_close_attempt"], 1)
            # The state the attempt opened with is restored, bit for bit.
            _same_value(self, before, _attempt_start(scene), "after the rejection")
            self.assertEqual(scene.accepted_physics_steps, 0)

    def test_abandon_ends_an_open_attempt_as_a_rejected_one(self):
        with _scene() as (scene, config), _counted("_close_attempt", "_open_attempt") as counts:
            self.assertFalse(scene.coupler.attempt_open)
            # Nothing is open: nothing is done.
            scene.coupler.abandon_step()
            scene.abandon_attempt()
            self.assertEqual(counts["_close_attempt"], 0)
            before = _left_open(self, scene, config)
            coupler = scene.coupler
            alm, predictor = scene.interaction.tunnel_alm, coupler._phase_load_predictor
            configured = scene.args.fem_tolerance
            # Inside the attempt: the multiplier step and the predictor's transaction are open, a
            # trial has run on the native state, and the candidate load is applied.
            self.assertTrue(alm._open)
            self.assertTrue(predictor.open)
            self.assertGreaterEqual(coupler.iterations, 1)
            self.assertNotEqual(_native(scene).tobytes(), before["native"].tobytes())
            self.assertEqual(counts, {"_close_attempt": 0, "_open_attempt": 1})
            coupler.abandon_step()
            self.assertFalse(coupler.attempt_open)
            self.assertEqual(counts, {"_close_attempt": 1, "_open_attempt": 1})
            self.assertFalse(alm._open)
            self.assertFalse(predictor.open)
            self.assertEqual(scene.solver.linear_solver_tolerance, configured)
            # What a rejected attempt gives back: the forces, the velocity increment, the native state.
            _same_value(self, before, _attempt_start(scene), "after the abandon")
            # The abandon is over: what closes a generator later is not an abandon.
            self.assertFalse(coupler._abandoning)
            # A second abandon does nothing.
            coupler.abandon_step()
            self.assertEqual(counts["_close_attempt"], 1)
            self.assertFalse(coupler._abandoning)

    def test_a_collected_attempt_issues_nothing(self):
        with _coupled("penalty") as example, _counted("_close_attempt") as counts:
            coupler = example.coupler
            pieces = _Script("every")
            pieces.begin_call(None)
            self.assertFalse(coupler.begin_step(*_arguments(example), pieces=pieces))
            pieces.begin_call(None, resumed=True)
            self.assertFalse(coupler.resume_step())
            # The attempt is dropped without abandon_step(): what finalizes it closes nothing.
            attempt, coupler._suspended = coupler._suspended, None
            alive = weakref.ref(attempt)
            del attempt
            gc.collect()
            self.assertIsNone(alive())
            self.assertEqual(counts["_close_attempt"], 0)
            self.assertFalse(coupler._abandoning)

    def test_an_attempt_collected_after_an_abandon_issues_nothing(self):
        with _coupled("penalty") as example, _counted("_close_attempt") as counts:
            coupler = example.coupler
            pieces = _Script("every")
            pieces.begin_call(None)
            self.assertFalse(coupler.begin_step(*_arguments(example), pieces=pieces))
            coupler.abandon_step()
            self.assertEqual(counts["_close_attempt"], 1)
            self.assertFalse(coupler._abandoning)
            # A second attempt is dropped without abandon_step(): the abandon before it is over, and
            # what finalizes this one closes nothing.
            pieces.begin_call(None)
            self.assertFalse(coupler.begin_step(*_arguments(example), pieces=pieces))
            attempt, coupler._suspended = coupler._suspended, None
            alive = weakref.ref(attempt)
            del attempt
            gc.collect()
            self.assertIsNone(alive())
            self.assertEqual(counts["_close_attempt"], 1)

    def test_replaced_loops_are_not_left(self):
        with _coupled("penalty") as example:
            coupler, calls = example.coupler, []
            solve_host = coupler._solve_host

            def replaced(*arguments, **options):
                calls.append(coupler.evaluations)
                return solve_host(*arguments, **options)

            coupler._solve_host = replaced
            # A budget that would leave the attempt before each of its host-controlled trials.
            pieces = _Script("trials")
            self.assertEqual(_drive(coupler, pieces, _arguments(example)), 1)
            self.assertEqual(len(calls), 1)
            self.assertGreater(coupler.iterations, 5)
            # The replaced loop was called as without pieces: no trial was asked for.
            self.assertEqual([kind for kind, _ in pieces.asked], ["end"])

    def test_a_budget_that_refuses_the_first_piece_of_a_call_is_an_error(self):
        with _coupled("penalty") as example, _counted("_close_attempt") as counts:
            coupler = example.coupler
            pieces = _Script("refuse")
            pieces.begin_call(None)
            self.assertFalse(coupler.begin_step(*_arguments(example), pieces=pieces))
            self.assertTrue(coupler.attempt_open)
            pieces.begin_call(None, resumed=True)
            with self.assertRaisesRegex(AssertionError, "must admit the first piece"):
                coupler.resume_step()
            # The call made no progress and did not pretend to: the attempt ended as a rejected one.
            self.assertFalse(coupler.attempt_open)
            self.assertEqual(counts["_close_attempt"], 1)
            self.assertEqual(coupler.evaluations, 0)
            self.assertEqual(len(pieces.asked), 2)

    def test_a_loop_left_between_trials_is_counted_once_without_the_time_between_the_calls(self):
        with _coupled("penalty") as example:
            coupler = example.coupler
            pieces = _Script("trials")
            pieces.begin_call(None)
            started = time.perf_counter()
            self.assertFalse(coupler.begin_step(*_arguments(example), pieces=pieces))
            spent = time.perf_counter() - started
            calls = 1
            while coupler.attempt_open:
                # Left inside the loop: it is not counted before it ends.
                self.assertEqual((coupler.trial_loops, coupler.loop_seconds), (0, 0.0))
                # The time between two calls is no time of the loop.
                time.sleep(0.25 if calls == 2 else 0.0)
                pieces.begin_call(None, resumed=True)
                started = time.perf_counter()
                coupler.resume_step()
                spent += time.perf_counter() - started
                calls += 1
            self.assertGreater(calls, 3)
            self.assertEqual(coupler.trial_loops, 1)
            self.assertGreater(coupler.loop_seconds, 0.0)
            self.assertLess(coupler.loop_seconds, spent)
            self.assertGreaterEqual(coupler._paused_seconds, 0.25)


def _prepared(scene, config, action=ACTIONS[0]):
    """The scene as its simulate() has it where the first attempt of a frame starts.

    Returns the arguments and the options of the coupler's step, and the coupler.
    """
    import warp as wp

    _apply_action(scene, config, action)
    wp.copy(scene._frame_particle_q, scene.state_0.particle_q)
    scene.solver.refresh_tangent(scene._frame_particle_q, scene.sim_dt)
    scene.state_0.clear_forces()
    desired, _ = scene._manual_target_pose()
    scene._prepare_step_loads(desired, np.zeros(6), scene.sim_dt)
    arguments = (scene.state_0, scene.state_1, scene.control, scene.sim_dt)
    return arguments, {"external_forces": scene._external_forces}, scene.coupler


def _attempt_start(scene):
    """What an attempt that does not end accepted gives back, as host copies."""
    return {
        **{name: getattr(scene.state_0, name).numpy().copy() for name in STATE},
        "body_f": scene.state_0.body_f.numpy().copy(),
        "particle_f": scene.state_0.particle_f.numpy().copy(),
        "velocity_increment": scene.solver.velocity_increment.numpy().copy(),
        "native": _native(scene),
        "load": scene.coupler._load.numpy().copy(),
        "tolerance": scene.solver.linear_solver_tolerance,
    }


def _left_open(case, scene, config, action=ACTIONS[0], trials=1):
    """Start a frame and leave its first attempt open after ``trials`` host-controlled trials.

    The scene has its own budget. While this runs, a script answers in its place: every call
    runs one piece. Afterwards the scene's own rule decides again; the open attempt asks the
    same object. Returns what the attempt started from (taken where the scene had prepared it).
    """
    start = {}
    script = _Script("every")
    budget = scene._host_budget = HostTrialBudget()
    begin_step = scene.coupler.begin_step

    def begun(*arguments, **options):
        start.update(_attempt_start(scene))
        return begin_step(*arguments, **options)

    _apply_action(scene, config, action)
    with (
        patch.object(budget, "begin_call", script.begin_call),
        patch.object(budget, "admit", script.admit),
        patch.object(budget, "close", script.close),
    ):
        with patch.object(scene.coupler, "begin_step", begun):
            case.assertIsNone(_step(scene, None, inside=True))
        for _ in range(trials):
            case.assertTrue(scene.coupler.attempt_open)
            evaluations = scene.coupler.evaluations
            case.assertIsNone(_step(scene, None, inside=True))
            case.assertEqual(scene.coupler.evaluations, evaluations + 1)
    case.assertTrue(scene.coupler.attempt_open)
    case.assertIsNotNone(scene._pending_attempt)
    case.assertTrue(scene.frame_pending)
    case.assertIs(scene._host_budget, budget)
    return start


def _run_inside(*, deadline=None, script=None, seed=0, reject=(), observe=False, actions=ACTIONS, far=None):
    """Run the frames of ``actions`` with calls that may return inside an attempt.

    With ``script`` the scene's budget is a scripted one; without it the scene's own
    :class:`HostTrialBudget` decides against ``deadline``. ``far``: per call, by a seeded coin,
    this deadline instead of ``deadline``.

    Returns the snapshot after every frame, the calls of every frame (per call: the attempts it
    began, the host-controlled trials it evaluated and those it was admitted, the steps it
    accepted, the attempts it rejected,
    whether it left an attempt open, and the unfinished frame it left), the number of
    accepted-step checks, the counts of the coupler's opens and closes, and the scene's budget.
    """
    snapshots, calls = [], []
    rng = random.Random(seed)
    with (
        _scene() as (scene, config),
        _rejecting(scene, reject) as checks,
        _counted("_open_attempt", "_close_attempt") as counts,
    ):
        if script is not None:
            scene._host_budget = _Script(script, seed)
        prepare, prepared = scene._prepare_step_loads, [0]

        def counted(*arguments, **options):
            prepared[0] += 1
            return prepare(*arguments, **options)

        scene._prepare_step_loads = counted
        for action in actions:
            _apply_action(scene, config, action)
            frame = []
            while True:
                coupler = scene.coupler
                before = (prepared[0], coupler.evaluations, scene.accepted_physics_steps, scene.step_refinements)
                admitted = _admitted(scene._host_budget)
                limit = far if far is not None and rng.random() < 0.5 else deadline
                poses = _step(scene, limit, inside=True)
                is_open = scene._pending_attempt is not None
                assert is_open == coupler.attempt_open, "the scene and the coupler disagree about an open attempt"
                assert not is_open or scene.frame_pending, "an attempt is open and no frame is pending"
                frame.append(
                    {
                        "began": prepared[0] - before[0],
                        "trials": coupler.evaluations - before[1],
                        "hosts": _admitted(scene._host_budget) - admitted,
                        "accepted": scene.accepted_physics_steps - before[2],
                        "rejected": scene.step_refinements - before[3],
                        "open": is_open,
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
        budget = scene._host_budget
        bound = scene._substep_bound
    return snapshots, calls, checks[0], dict(counts), budget, bound


def _flat(calls):
    return [call for frame in calls for call in frame]


def _admitted(budget):
    """Host-controlled trials that a budget has admitted so far."""
    if budget is None:
        return 0
    return budget.count("host", True) if isinstance(budget, _Script) else budget.admitted.get("host", 0)


class HostTrialSceneTests(unittest.TestCase):
    """The scene through ``_apply_action`` and ``_step``, as a worker drives it."""

    def assertSameFrames(self, whole, split):
        self.assertEqual(len(whole), len(split))
        for frame, (expected, actual) in enumerate(zip(whole, split, strict=True)):
            self.assertEqual(expected.keys(), actual.keys())
            for name, value in expected.items():
                with self.subTest(frame=frame, name=name):
                    if isinstance(value, np.ndarray):
                        # Bit for bit: a call that returns inside an attempt moves where calls end and nothing else.
                        self.assertEqual(value.tobytes(), actual[name].tobytes())
                    else:
                        self.assertEqual(value, actual[name])

    def test_a_passed_deadline_runs_one_piece_per_call(self):
        split, calls, checks, counts, budget, bound = _run_inside(deadline=PASSED, reject=(13, 14), observe=True)
        accepted, refinements, _, trials = split[-1]["counters"][:4]
        records = _flat(calls)
        # Every call ran one piece: it began an attempt, or it ran one host-controlled trial.
        self.assertEqual([record["began"] + record["hosts"] for record in records], [1] * len(records))
        self.assertEqual(refinements, 2)
        attempts = accepted + refinements
        hosts = sum(record["hosts"] for record in records)
        self.assertEqual(sum(record["began"] for record in records), attempts)
        self.assertEqual(sum(record["trials"] for record in records), trials)
        # A trial that was started was evaluated (on a CUDA device a trial of captured forces
        # may end in the fallback to the loop without them, before it is counted).
        self.assertGreaterEqual(hosts, trials)
        if not _CUDA:
            self.assertEqual(hosts, trials)
        # The calls of an attempt: one that begins it, and one per host-controlled trial.
        self.assertEqual(len(records), attempts + hosts)
        self.assertGreater(trials, attempts)
        # A call that began an attempt left it open before its first trial; the end of an attempt
        # is never refused, so the call of its last trial ended it.
        for record in records:
            if record["began"]:
                self.assertTrue(record["open"])
                self.assertEqual((record["accepted"], record["rejected"]), (0, 0))
            else:
                self.assertEqual(record["open"], record["accepted"] + record["rejected"] == 0)
        self.assertEqual(sum(record["accepted"] for record in records), accepted)
        # Every frame ended in a call that left nothing unfinished, every other call left the frame.
        for frame in calls:
            self.assertIsNone(frame[-1]["pending"])
            self.assertFalse(frame[-1]["open"])
            self.assertTrue(all(record["pending"] is not None for record in frame[:-1]))
        # One open and one close per attempt, whatever the number of calls it took.
        self.assertEqual(counts, {"_open_attempt": attempts, "_close_attempt": attempts})
        self.assertEqual(checks, attempts)
        # The budget is the scene's own: it has measured, and it refused every trial it was asked for.
        self.assertIsInstance(budget, HostTrialBudget)
        self.assertEqual(budget.admitted, {"host": hosts, "end": attempts})
        self.assertEqual(budget.refused, {"host": hosts})
        self.assertGreater(budget.longest_host, 0.0)
        self.assertGreater(budget.longest_tail, 0.0)
        # Every substep was measured as one.
        self.assertEqual((bound.substeps, bound.trials), (accepted, trials))

    @unittest.skipIf(_CUDA, _BITS)
    def test_frames_left_between_host_trials_are_the_frames_of_one_call(self):
        whole, whole_calls, whole_checks, _, _ = _run(None, reject=(13, 14))
        split, calls, checks, counts, budget, _ = _run_inside(script="every", reject=(13, 14), observe=True)
        # With the tangent version: nothing of a frame's start or of a rejection runs twice.
        self.assertSameFrames(whole, split)
        self.assertEqual([len(frame) for frame in whole_calls], [1] * len(ACTIONS))
        accepted, refinements, shortest, trials = whole[-1]["counters"][:4]
        self.assertEqual((refinements, shortest), (2, 1.0 / 240.0 / 4))
        self.assertEqual(checks, whole_checks)
        records = _flat(calls)
        # The same trials, each alone in its call, and a pause before the end of every attempt.
        self.assertEqual(sum(record["trials"] for record in records), trials)
        self.assertEqual(max(record["trials"] for record in records), 1)
        self.assertEqual(budget.count("host", False), trials)
        self.assertEqual(budget.count("end", False), accepted + refinements)
        self.assertEqual(counts, {"_open_attempt": accepted + refinements, "_close_attempt": accepted + refinements})
        self.assertGreater(sum(record["open"] for record in records), trials)

    @unittest.skipIf(_CUDA, _BITS)
    def test_frames_under_seeded_pauses_are_the_frames_of_one_call(self):
        whole, _, whole_checks, _, _ = _run(None, reject=(13, 14))
        opened = []
        for seed in (1, 2, 3):
            with self.subTest(seed=seed):
                split, calls, checks, _, budget, _ = _run_inside(script="coin", seed=seed, reject=(13, 14))
                self.assertSameFrames(whole, split)
                self.assertEqual(checks, whole_checks)
                opened.append(tuple(record["open"] for record in _flat(calls)))
                self.assertGreater(sum(opened[-1]), 20)
                # Pauses before a trial and before the end of an attempt both occurred.
                self.assertGreater(budget.count("host", False), 5)
                self.assertGreater(budget.count("end", False), 5)
        # The seeds leave the frames at different places.
        self.assertEqual(len(set(opened)), 3)
        # The scene's own budget under a seeded mix of a passed deadline and one far away.
        split, calls, checks, _, budget, _ = _run_inside(
            deadline=PASSED, far=time.perf_counter() + 3600.0, seed=5, reject=(13, 14), observe=True
        )
        self.assertSameFrames(whole, split)
        self.assertGreater(sum(record["open"] for record in _flat(calls)), 5)
        self.assertGreater(max(record["trials"] for record in _flat(calls)), 1)

    def test_a_call_without_a_deadline_finishes_an_open_attempt(self):
        with _scene() as (scene, config), _counted("_open_attempt", "_close_attempt") as counts:
            for frame, action in enumerate(ACTIONS[:2]):
                _left_open(self, scene, config, action)
                opened = counts["_open_attempt"]
                # Without a deadline, and without the unit: the call goes on with the attempt and ends the frame.
                self.assertIsNotNone(_step(scene))
                self.assertFalse(scene.frame_pending)
                self.assertFalse(scene.coupler.attempt_open)
                self.assertIsNone(scene._pending_attempt)
                self.assertAlmostEqual(scene.elapsed, (frame + 1) * config.frame_dt, delta=1e-12)
                # The attempt that was open is one of the four of the frame: it was not begun again.
                self.assertEqual(counts["_open_attempt"], opened + 3)
                self.assertEqual(counts["_close_attempt"], counts["_open_attempt"])
            self.assertEqual(scene.accepted_physics_steps, 8)

    @unittest.skipIf(_CUDA, _BITS)
    def test_a_frame_finished_by_a_call_without_a_deadline_is_the_frame_of_one_call(self):
        whole, _, _, _, _ = _run(None, actions=ACTIONS[:2])
        with _scene() as (scene, config):
            for action in ACTIONS[:2]:
                _left_open(self, scene, config, action)
                _step(scene)
            scene.actual_tip_history.clear()
            self.assertSameFrames(whole[-1:], [_snapshot(scene)])

    def after_reset(self, partial):
        """Three frames after a reset that falls between two frames or on an attempt left open after a trial."""
        with _scene() as (scene, config), _counted("_close_attempt", "abandon_step") as counts:
            for action in ACTIONS[:2]:
                _apply_action(scene, config, action)
                _step(scene)
            old = scene.coupler
            if partial:
                _left_open(self, scene, config, ACTIONS[-1])
                self.assertIsNotNone(scene._report_before_attempt)
            closes = counts["_close_attempt"]
            alm, predictor = scene.interaction.tunnel_alm, old._phase_load_predictor
            _reset_scene(scene, config, rebuild=True)
            # The attempt ended once, as a rejected one, and nothing of it or of its frame is kept.
            self.assertEqual(counts["_close_attempt"], closes + int(partial))
            self.assertFalse(old.attempt_open)
            self.assertFalse(alm._open)
            self.assertFalse(predictor.open)
            self.assertIsNot(scene.coupler, old)
            self.assertFalse(scene.coupler.attempt_open)
            self.assertFalse(scene.frame_pending)
            for name in ("_pending_frame", "_pending_attempt", "_host_budget", "_report_before_attempt", "_ramp_frame"):
                self.assertIsNone(getattr(scene, name), name)
            self.assertEqual((scene.elapsed, scene.accepted_physics_steps, scene.step_refinements), (0.0, 0, 0))
            snapshots = []
            for action in ACTIONS[:3]:
                _apply_action(scene, config, action)
                _step(scene)
                scene.actual_tip_history.clear()
                snapshots.append(_snapshot(scene))
            return snapshots

    def test_a_reset_with_an_attempt_open_leaves_nothing_open(self):
        self.after_reset(partial=True)

    def test_a_rebuild_ends_an_open_attempt_without_the_close_it_runs(self):
        # A rebuild of this scene runs its close() (it releases the response graphs), and a close
        # ends an open attempt too. A scene without a close has the reset of its interaction
        # state alone: that reset must end the attempt by itself.
        with _scene() as (scene, config), _counted("_close_attempt") as counts:
            _left_open(self, scene, config)
            old = scene.coupler
            with patch.object(type(scene), "close", lambda self: None):
                _reset_scene(scene, config, rebuild=True)
            self.assertEqual(counts["_close_attempt"], 1)
            self.assertFalse(old.attempt_open)
            self.assertFalse(old.interaction.tunnel_alm._open)
            for name in ("_pending_frame", "_pending_attempt", "_host_budget", "_report_before_attempt"):
                self.assertIsNone(getattr(scene, name), name)

    @unittest.skipIf(_CUDA, _BITS)
    def test_frames_after_a_reset_with_an_attempt_open_are_those_after_a_reset_between_frames(self):
        self.assertSameFrames(self.after_reset(partial=False), self.after_reset(partial=True))

    def test_a_scene_that_stops_with_an_attempt_open_abandons_it(self):
        with _scene() as (scene, config), _counted("_close_attempt") as counts:
            _left_open(self, scene, config)
            coupler = scene.coupler
            scene._fail("stopped by the test")
            steps, evaluations = scene.accepted_physics_steps, coupler.evaluations
            _step(scene, PASSED, inside=True)
            self.assertFalse(scene.frame_pending)
            self.assertIsNone(scene._pending_attempt)
            self.assertFalse(coupler.attempt_open)
            # Ended as a rejected attempt, once, without another trial.
            self.assertEqual(counts["_close_attempt"], 1)
            self.assertEqual((scene.accepted_physics_steps, coupler.evaluations), (steps, evaluations))
            self.assertFalse(scene.interaction.tunnel_alm._open)
            # The control frame and the frame of the arm-target ramp are closed.
            self.assertFalse(coupler._phase_load_predictor.frame_open)
            self.assertFalse(coupler._phase_load_predictor.open)
            self.assertIsNone(scene._ramp_frame)

    def test_a_close_with_an_attempt_open_abandons_it_first(self):
        with _scene() as (scene, config):
            _left_open(self, scene, config)
            coupler, order = scene.coupler, []
            close_attempt = coupler._close_attempt
            invalidate = coupler._shaft_preconditioner.invalidate_preparation

            def closed(*arguments, **options):
                order.append("the attempt is closed")
                return close_attempt(*arguments, **options)

            def invalidated(*arguments, **options):
                order.append("the preparation is released")
                return invalidate(*arguments, **options)

            with (
                patch.object(coupler, "_close_attempt", closed),
                patch.object(coupler._shaft_preconditioner, "invalidate_preparation", invalidated),
            ):
                scene.close()
            # The close of the attempt addresses what close() releases: it comes first.
            self.assertEqual(order, ["the attempt is closed", "the preparation is released"])
            self.assertFalse(coupler.attempt_open)
            self.assertFalse(scene.frame_pending)
            self.assertIsNone(scene._pending_attempt)
            self.assertIsNone(scene._ramp_frame)
            self.assertFalse(scene.interaction.tunnel_alm._open)

    def test_the_close_of_a_rebuild_abandons_nothing(self):
        # A scene calls its close() while it is built, before it has a coupler, and a rebuild
        # calls it with the coupler of the scene before: with no attempt open nothing is ended.
        with _scene() as (scene, config), _counted("_close_attempt", "abandon_step") as counts:
            _apply_action(scene, config, ACTIONS[0])
            _step(scene)
            closes = counts["_close_attempt"]
            scene.close()
            _reset_scene(scene, config, rebuild=True)
            self.assertEqual(counts, {"_close_attempt": closes, "abandon_step": 0})
            unbuilt = object.__new__(type(scene))
            unbuilt.close()
            self.assertIsNone(unbuilt._pending_attempt)
            self.assertEqual(counts["abandon_step"], 0)

    def test_a_substep_left_inside_an_attempt_is_measured_as_one(self):
        with _scene() as (scene, config):
            _apply_action(scene, config, ACTIONS[0])
            coupler = scene.coupler
            started = time.perf_counter()
            _step(scene, PASSED, inside=True)
            spent = [time.perf_counter() - started]
            # Left before the first trial of the first attempt: nothing is measured yet.
            self.assertTrue(coupler.attempt_open)
            self.assertIsNone(scene._substep_bound)
            step_dt, remaining, quick_steps, kept = scene._pending_frame
            self.assertEqual((step_dt, remaining, quick_steps), (scene.sim_dt, scene.frame_dt, 0))
            # What the call spent on the substep: its time; no trial and no loop has ended.
            self.assertEqual(tuple(kept[1:]), (0, 0, 0.0))
            self.assertGreater(kept[0], 0.0)
            self.assertLessEqual(kept[0], spent[0])
            while scene.accepted_physics_steps == 0:
                self.assertLess(len(spent), CALL_LIMIT)
                # The time between two calls is no time of the substep.
                time.sleep(0.25 if len(spent) == 1 else 0.0)
                started = time.perf_counter()
                _step(scene, PASSED, inside=True)
                spent.append(time.perf_counter() - started)
                if coupler.attempt_open:
                    # What the calls so far spent on the substep grows, and stays below their time.
                    self.assertGreater(scene._pending_frame[3][0], kept[0])
                    self.assertLessEqual(scene._pending_frame[3][0], sum(spent))
                    kept = scene._pending_frame[3]
            # One substep, measured as one: its trials, its one loop, the loop's time without the pause.
            self.assertTrue(scene.frame_pending)
            self.assertIsNone(scene._pending_frame[3])
            bound = scene._substep_bound
            self.assertEqual((bound.substeps, bound.trials), (1, coupler.evaluations))
            self.assertEqual(bound.last_loops, coupler.trial_loops)
            self.assertGreaterEqual(bound.last_loops, 1)
            self.assertAlmostEqual(bound.loop_seconds, coupler.loop_seconds, places=12)
            self.assertGreater(bound.longest_outside, 0.0)
            self.assertLessEqual(bound.longest_outside + bound.loop_seconds, sum(spent))

    def test_the_preparation_of_an_attempt_runs_once(self):
        _, calls, checks, counts, _, _ = _run_inside(script="every", reject=(1, 2), actions=ACTIONS[:3])
        records = _flat(calls)
        attempts = sum(record["accepted"] + record["rejected"] for record in records)
        self.assertEqual(sum(record["rejected"] for record in records), 2)
        self.assertGreater(attempts, 3 * 4 + 2)
        # Prepared once, opened once and closed once per attempt, whatever the calls it took.
        self.assertEqual(sum(record["began"] for record in records), attempts)
        self.assertEqual(counts, {"_open_attempt": attempts, "_close_attempt": attempts})
        self.assertEqual(checks, attempts)
        self.assertGreater(len(records), 2 * attempts)
        # No call that went on with an attempt prepared it again.
        previous = False
        for record in records:
            if previous and not (record["accepted"] or record["rejected"]):
                self.assertEqual(record["began"], 0)
            previous = record["open"]

    def test_an_abandon_that_raises_still_forgets_the_attempt(self):
        with _scene() as (scene, config):
            _left_open(self, scene, config)
            coupler = scene.coupler
            abandon = coupler.abandon_step

            def failing():
                abandon()
                raise RuntimeError("raised by the test")

            with (
                patch.object(coupler, "abandon_step", failing),
                self.assertRaisesRegex(RuntimeError, "raised by the test"),
            ):
                scene.abandon_attempt()
            # The coupler has no attempt open, and the scene does not go on with one.
            self.assertFalse(coupler.attempt_open)
            self.assertIsNone(scene._pending_attempt)
            # The frame is still pending: the next call starts its attempt anew and finishes the frame.
            self.assertTrue(scene.frame_pending)
            _step(scene)
            self.assertFalse(scene.frame_pending)
            self.assertTrue(scene.running, scene.failure_reason)
            self.assertEqual(scene.accepted_physics_steps, 4)

    def test_a_call_without_inside_that_goes_on_may_leave_the_attempt_again(self):
        with _scene() as (scene, config):
            for action in ACTIONS[:12]:
                _apply_action(scene, config, action)
                _step(scene)
            coupler = scene.coupler
            _apply_action(scene, config, ACTIONS[12])
            # In contact a loop takes several trials. With nothing open only a call with ``inside``
            # leaves an attempt (before its first host-controlled trial). A call WITHOUT ``inside``
            # then goes on with it, runs that trial, and under its passed deadline leaves the
            # attempt again before the next one.
            left_again = 0
            for _ in range(CALL_LIMIT):
                if coupler.attempt_open:
                    evaluations = coupler.evaluations
                    _step(scene, PASSED)
                    # One host-controlled trial (on a CUDA device a trial of captured forces may end in the
                    # fallback to the loop without them, before it is counted).
                    self.assertIn(coupler.evaluations - evaluations, (1,) if not _CUDA else (0, 1))
                    if coupler.attempt_open:
                        self.assertIsNotNone(scene._pending_attempt)
                        self.assertTrue(scene.frame_pending)
                        left_again += 1
                        break
                else:
                    _step(scene, PASSED, inside=True)
                self.assertTrue(scene.frame_pending, "no loop of the frame took two host-controlled trials")
            self.assertEqual(left_again, 1)
            # With nothing open such a call never leaves an attempt.
            _step(scene)
            self.assertFalse(scene.frame_pending)
            _apply_action(scene, config, ACTIONS[13])
            for _ in range(CALL_LIMIT):
                _step(scene, PASSED)
                self.assertFalse(coupler.attempt_open)
                if not scene.frame_pending:
                    break
            self.assertFalse(scene.frame_pending)
            # The docstring says both.
            text = " ".join(insertion.Example.simulate.__doc__.split())
            self.assertIn("a call never leaves an attempt that it started with nothing open", text)
            self.assertIn("A call that goes on with an attempt that the call before left open is the exception", text)
            self.assertIn("given a deadline it may leave that attempt again", text)
            self.assertNotIn("an attempt is never interrupted", text)


class BudgetCloseTests(unittest.TestCase):
    """The three places where ``simulate`` ends a timed piece of the scene's budget, each on a clock that the test moves.

    A frame of four substeps runs in one call that may return inside an attempt and has no
    deadline. The clock stands still except where a test moves it, so every measured host
    trial and every measured tail is zero unless the moved time falls into it.
    """

    @contextmanager
    def frame(self, reject=()):
        with _scene() as (scene, config), _rejecting(scene, reject):
            clock = _Clock()
            budget = scene._host_budget = HostTrialBudget(clock)
            yield scene, config, clock, budget
            _apply_action(scene, config, ACTIONS[0])
            self.assertIsNotNone(_step(scene, None, inside=True))
            self.assertFalse(scene.frame_pending)
            self.assertIs(scene._host_budget, budget)
            # A rejected attempt is repeated as two halves, and the frame goes on at the full step.
            self.assertEqual(scene.accepted_physics_steps, 4 + len(reject))
            # Every attempt asked for its end, and every tail and every host-controlled trial was measured.
            self.assertEqual(budget.admitted["end"], scene.accepted_physics_steps + scene.step_refinements)
            self.assertIsNotNone(budget.longest_tail)
            self.assertIsNotNone(budget.longest_host)

    def test_a_rejected_attempt_ends_its_piece_before_the_retry_is_prepared(self):
        with self.frame(reject=(1,)) as (scene, config, clock, budget):
            refresh, seen = scene.solver.refresh_tangent, [scene.step_refinements]

            def refreshed(*arguments, **options):
                if scene.step_refinements > seen[0]:
                    # The tangent of the retry: after the rejected attempt, before the retry's first question.
                    seen[0] = scene.step_refinements
                    clock.now += 100.0
                return refresh(*arguments, **options)

            scene.solver.refresh_tangent = refreshed
        self.assertEqual(scene.step_refinements, 1)
        self.assertEqual(clock.now, 200.0)
        # The time between the rejection and the retry belongs to no piece.
        self.assertEqual((budget.longest_host, budget.longest_tail), (0.0, 0.0))

    def test_an_accepted_attempt_ends_its_tail_before_the_next_substep_is_prepared(self):
        with self.frame() as (scene, config, clock, budget):
            prepare, prepared = scene._prepare_step_loads, [0]

            def moved(*arguments, **options):
                # The loads of an attempt: after the tail of the attempt before, before its own first question.
                prepared[0] += 1
                clock.now += 100.0
                return prepare(*arguments, **options)

            scene._prepare_step_loads = moved
        self.assertEqual(prepared[0], 4)
        self.assertEqual(clock.now, 500.0)
        # The preparation of a substep belongs to no piece: not to the tail of the substep before it.
        self.assertEqual((budget.longest_host, budget.longest_tail), (0.0, 0.0))

    def test_the_last_attempt_of_a_frame_ends_its_tail_with_the_frame(self):
        with self.frame() as (scene, config, clock, budget):
            check = scene._check_tissue_integrity

            def checked():
                # What ends the frame: in the tail of its last attempt.
                clock.now += 5.0
                return check()

            scene._check_tissue_integrity = checked
        self.assertEqual(clock.now, 105.0)
        # The tail of the last attempt was measured, with the end of the frame in it.
        self.assertEqual(budget.longest_tail, 5.0)
        self.assertEqual(budget.longest_host, 0.0)


class LocalSceneHostTrialTests(unittest.TestCase):
    """``LocalSutureScene``: what a driver sees of a call that returned inside an attempt."""

    # Checks 5 and 6: the second step of the second frame and its first retry.
    reject = (5, 6)

    def frames(self, deadline, actions, unit=None):
        """The observation after every complete frame, and after every call with whether an attempt was open."""
        complete, every = [], []
        options = {} if unit is None else {"unit": unit, "observe_native_trials": True}
        if deadline is not None:
            options["deadline"] = deadline
        with _local() as scenes, _rejecting(scenes._scene, self.reject):
            scene = scenes._scene
            # What the driver saw before the first call: the observation of the scene's start.
            self.start = scenes._last[0]
            for action in actions:
                result = scenes.step(action, **options)
                every.append((result[0], scenes.frame_unfinished, scene._pending_attempt is not None))
                while scenes.frame_unfinished:
                    with self.assertRaisesRegex(ValueError, "unfinished"):
                        scenes.step(action, **options)
                    result = scenes.step(None, **options)
                    every.append((result[0], scenes.frame_unfinished, scene._pending_attempt is not None))
                    self.assertLess(len(every), CALL_LIMIT * len(actions))
                complete.append(result[0])
            frame_dt = scenes.config.frame_dt
        return complete, every, frame_dt

    def differing(self, expected, actual):
        """Names of the entries of two observations that differ; arrays bit for bit."""

        def same(a, b):
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                a, b = np.asarray(a), np.asarray(b)
                return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
            if isinstance(a, dict) and isinstance(b, dict):
                return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
            if isinstance(a, float) and isinstance(b, float):
                return a == b or (a != a and b != b)
            return a == b

        names = []
        # The times of a step are not part of the observation of a scene's start.
        self.assertEqual(expected.keys() - set(_UNSTABLE), actual.keys() - set(_UNSTABLE))
        for name in expected.keys() - set(_UNSTABLE):
            if name in ("diagnostics", "thread_diagnostics"):
                self.assertEqual(expected[name].keys(), actual[name].keys())
                names.extend(key for key in expected[name] if not same(expected[name][key], actual[name][key]))
            elif not same(expected[name], actual[name]):
                names.append(name)
        return names

    def test_a_failure_in_a_call_that_accepted_nothing_is_in_its_observation(self):
        action = np.asarray(ACTIONS[0], dtype=float)[None, :]
        options = {"deadline": PASSED, "unit": "host-trial", "observe_native_trials": True}

        class Always:
            def __contains__(self, number):
                return True

        with _local() as scenes, _rejecting(scenes._scene, Always()):
            scene, raised = scenes._scene, []
            resume_step = scene.coupler.resume_step

            def resumed():
                try:
                    return resume_step()
                except NeedleStepTooLarge:
                    raised.append(scene.step_refinements)
                    raise

            calls = 0
            with patch.object(scene.coupler, "resume_step", resumed):
                result = scenes.step(action, **options)[0]
                while scenes.frame_unfinished:
                    self.assertEqual(result["diagnostics"]["task_status"], "running")
                    self.assertIsNone(result["diagnostics"]["failure_reason"])
                    result = scenes.step(None, **options)[0]
                    calls += 1
                    self.assertLess(calls, 40 * CALL_LIMIT)
            # Every attempt was rejected, down to the smallest time step: the last rejection is the failure.
            diagnostics = result["diagnostics"]
            self.assertEqual(diagnostics["accepted_physics_steps"], 0)
            self.assertGreater(diagnostics["step_refinements"], 5)
            self.assertEqual(len(raised), diagnostics["step_refinements"] + 1)
            # It came out of a call that went on with an open attempt and accepted nothing, and
            # the observation which that call returned shows it.
            self.assertEqual(raised[-1], diagnostics["step_refinements"])
            self.assertEqual(diagnostics["task_status"], "failed")
            self.assertEqual(diagnostics["failure_reason"], "rejected by the test")
            self.assertFalse(scene.running)
            self.assertFalse(scene.coupler.attempt_open)
            self.assertIsNone(scene._pending_attempt)

    def test_an_observation_with_an_attempt_open_is_the_observation_before_the_attempt(self):
        actions = np.asarray(ACTIONS, dtype=float)[:, None, :]
        _, every, _ = self.frames(PASSED, actions, unit="host-trial")
        compared, seen, outside = 0, {}, {}
        previous = self.start
        for observation, _, is_open in every:
            if is_open:
                # A passed deadline: every attempt begins at the start of a call, so the
                # observation before the open one is that of the call before the attempt began.
                for name in ("accepted_physics_steps", "step_refinements", "physical_time_s"):
                    self.assertEqual(observation["diagnostics"][name], previous["diagnostics"][name])
                for name in self.differing(previous, observation):
                    seen[name] = seen.get(name, 0) + 1
                    if name not in _MAY_DIFFER:
                        outside[name] = outside.get(name, 0) + 1
                compared += 1
            else:
                previous = observation
        # Every entry that differs is named above, with its reason; any other one fails here by name.
        self.assertEqual(outside, {}, "entries that differ in an observation with an attempt open")
        self.assertGreater(compared, 100)
        # The comparison is not empty: the work counts, the commands and the native reads did differ.
        for names in (WORK_COUNTS, ATTEMPT_COMMANDS, FRAME_COMMANDS, NATIVE_READS):
            self.assertTrue(any(name in seen for name in names), names)

    def test_the_report_of_an_open_attempt_is_the_report_before_it(self):
        action = np.asarray(ACTIONS[0], dtype=float)[None, :]
        options = {"deadline": PASSED, "unit": "host-trial", "observe_native_trials": True}
        with _local() as scenes:
            scene = scenes._scene
            self.assertEqual(scene._ATTEMPT_REPORT, ATTEMPT_REPORT)
            before = scenes.observe()[0]["diagnostics"]
            live_differs = dict.fromkeys(ATTEMPT_REPORT, False)
            calls, opened = 0, 0
            for frame in range(14):
                result = scenes.step(np.asarray(ACTIONS[frame], dtype=float)[None, :] if frame else action, **options)
                while True:
                    calls += 1
                    diagnostics, live = result[0]["diagnostics"], scene._needle_report()
                    if scene._pending_attempt is not None:
                        opened += 1
                        self.assertEqual(tuple(scene._report_before_attempt), ATTEMPT_REPORT)
                        for name in ATTEMPT_REPORT:
                            # The entry of the report is the one of before the attempt, whatever its trials did.
                            _same_value(self, before[name], diagnostics[name], name)
                            _same_value(self, scene._report_before_attempt[name], diagnostics[name], name)
                            live_differs[name] |= live[name] != diagnostics[name]
                        # What does not describe the last attempt is the live value.
                        self.assertEqual(diagnostics["phase_predictor_attempts"], live["phase_predictor_attempts"])
                    else:
                        # No attempt is open: every entry is live again.
                        for name in ATTEMPT_REPORT:
                            _same_value(self, live[name], diagnostics[name], name)
                        before = diagnostics
                    if not scenes.frame_unfinished:
                        break
                    result = scenes.step(None, **options)
                    self.assertLess(calls, 14 * CALL_LIMIT)
            self.assertGreater(opened, 50)
            # The entries are kept for a reason: inside an attempt the live values are other ones.
            self.assertTrue(live_differs["last_coupling_attempt_iterations"])
            self.assertTrue(live_differs["last_coupling_attempt_residual_N"])

    def test_the_entries_kept_before_an_attempt_are_those_of_the_report_and_read_nothing_from_the_device(self):
        import warp as wp

        def kept(scene):
            """The entries as the scene takes them before an attempt: the whole report is not built, no array is read."""
            reads, numpy = [0], wp.array.numpy

            def counted(array, *arguments, **options):
                reads[0] += 1
                return numpy(array, *arguments, **options)

            whole = AssertionError("the whole report was built for the entries of an attempt")
            with (
                patch.object(type(scene), "_needle_report", side_effect=whole),
                patch.object(wp.array, "numpy", counted),
            ):
                entries = scene._attempt_report()
            self.assertEqual(reads[0], 0)
            return entries

        with _scene() as (scene, config):
            self.assertEqual(scene._ATTEMPT_REPORT, ATTEMPT_REPORT)
            # At the start, in free motion, in contact (the needle in the tissue), and after a rejected attempt.
            with _rejecting(scene, (57,)):
                for frames in (0, 2, 12, 2, 4):
                    for _ in range(frames):
                        _apply_action(scene, config, ACTIONS[min(scene.accepted_physics_steps // 4, len(ACTIONS) - 1)])
                        _step(scene)
                    entries, report = kept(scene), scene._needle_report()
                    self.assertEqual(tuple(entries), ATTEMPT_REPORT)
                    for name in ATTEMPT_REPORT:
                        _same_value(self, report[name], entries[name], name)
            self.assertEqual(scene.step_refinements, 1)
            self.assertGreater(scene.interaction.track_count, 0)
            self.assertGreater(entries["last_coupling_attempt_residual_N"], 0.0)
            # A scene between two builds has no coupler and no interaction: the entries of the report then.
            coupler, interaction = scene.coupler, scene.interaction
            scene.coupler = scene.interaction = None
            try:
                entries, report = kept(scene), scene._needle_report()
            finally:
                scene.coupler, scene.interaction = coupler, interaction
            for name in ATTEMPT_REPORT:
                _same_value(self, report[name], entries[name], name)

    def test_the_report_takes_the_entries_of_the_last_attempt_from_one_place(self):
        with _scene() as (scene, config):
            coupler, interaction = scene.coupler, scene.interaction
            # Into contact, to a frame whose last attempt left another residual than the largest of the run:
            # an entry that showed the wrong one of the two would differ.
            for action in ACTIONS[:16]:
                _apply_action(scene, config, action)
                _step(scene)
                if 0.0 < coupler.residual < coupler.maximum_residual:
                    break
            self.assertGreater(coupler.residual, 0.0)
            self.assertLess(coupler.residual, coupler.maximum_residual)
            # What the nine entries are: values that the last attempt left on the coupler and the interaction.
            direct = {
                "last_coupling_attempt_iterations": coupler.iterations,
                "last_coupling_attempt_residual_N": coupler.residual,
                "fem_graph_status": coupler.fem_graph_status,
                "coupling_graph_status": coupler.coupling_graph_status,
                "shaft_preconditioner_status": coupler.shaft_preconditioner_status,
                "last_interface_load_on_fixed_nodes_N": interaction.support_force.tolist(),
                "tunnel_alm_last_normal_error_m": interaction.tunnel_alm.normal_error,
                "tunnel_alm_last_dual_residual_N": interaction.tunnel_alm.dual_residual,
                "status": interaction.status,
            }
            self.assertEqual(tuple(direct), ATTEMPT_REPORT)
            self.assertGreaterEqual(direct["last_coupling_attempt_iterations"], 1)
            entries = scene._attempt_report()
            for name in ATTEMPT_REPORT:
                _same_value(self, direct[name], entries[name], name)
            # One source: the report has what _attempt_report() returns for these names, whatever that is.
            marks = {name: ("from _attempt_report", name) for name in ATTEMPT_REPORT}
            with patch.object(scene, "_attempt_report", lambda: dict(marks)):
                report = scene._needle_report()
            for name in ATTEMPT_REPORT:
                self.assertEqual(report[name], marks[name], name)

    @unittest.skipIf(_CUDA, _BITS)
    def test_complete_frames_under_the_unit_host_trial_are_those_of_whole_frames(self):
        actions = np.asarray(ACTIONS[:14], dtype=float)[:, None, :]
        whole, calls, _ = self.frames(None, actions)
        self.assertEqual(len(calls), len(actions))
        split, every, _ = self.frames(PASSED, actions, unit="host-trial")
        self.assertEqual(whole[-1]["diagnostics"]["step_refinements"], 2)
        self.assertGreater(sum(is_open for _, _, is_open in every), 50)
        for frame, (expected, actual) in enumerate(zip(whole, split, strict=True)):
            # The whole observation of a complete frame: state, report, thread, events, counters.
            _same_value(self, expected, actual, f"frame {frame}")
        # The unit without a deadline runs whole frames, and they are the same.
        far, calls, _ = self.frames(None, actions, unit="host-trial")
        self.assertEqual(len(calls), len(actions))
        for frame, (expected, actual) in enumerate(zip(whole, far, strict=True)):
            _same_value(self, expected, actual, f"frame {frame} without a deadline")

    def test_the_native_backend_refuses_the_unit_without_the_explicit_argument(self):
        action = np.asarray(ACTIONS[0], dtype=float)[None, :]
        with _local() as scenes, patch.object(parallel, "_apply_action") as applied:
            scene = scenes._scene
            with self.assertRaisesRegex(ValueError, "native rigid backend.*contacts and the clock of a trial"):
                scenes.step(action, deadline=PASSED, unit="host-trial")
            with self.assertRaisesRegex(ValueError, "native rigid backend"):
                scenes.step(action, unit="host-trial")
            for unit in ("trial", "substep", None):
                with self.assertRaisesRegex(ValueError, "unit must be 'attempt' or 'host-trial'"):
                    scenes.step(action, deadline=PASSED, unit=unit, observe_native_trials=True)
            # A refused call has not touched the scene: no action applied, no frame begun.
            applied.assert_not_called()
            self.assertEqual((scene.elapsed, scene.accepted_physics_steps), (0.0, 0))
            self.assertFalse(scenes.frame_unfinished)
            self.assertIsNone(scene._host_budget)
        with _local() as scenes:
            # With the explicit argument the unit is taken, and the default unit needs none.
            scenes.step(action, deadline=PASSED, unit="host-trial", observe_native_trials=True)
            self.assertIsNotNone(scenes._scene._pending_attempt)
            scenes.step(None)
            self.assertFalse(scenes.frame_unfinished)
            scenes.step(action, deadline=PASSED)
            self.assertTrue(scenes.frame_unfinished)
            self.assertIsNone(scenes._scene._pending_attempt)

    def test_the_default_unit_steps_the_scene_as_before(self):
        calls = []
        scene = SimpleNamespace(
            actual_tip_history=[], frame_pending=False, close=lambda: None, viewer=SimpleNamespace(close=lambda: None)
        )

        def step(scene, *arguments, **options):
            calls.append((arguments, options))

        with (
            patch.object(parallel, "_make_scene", lambda config, env_id: scene),
            patch.object(parallel, "_observe", lambda scene, env_id, **options: {"env_id": env_id}),
            patch.object(parallel, "_apply_action", lambda scene, config, action: None),
            patch.object(parallel, "_step", step),
            LocalSutureScene() as scenes,
        ):
            action = np.zeros((1, 6))
            # The default unit: the scene is stepped with the arguments it got before the unit existed.
            scenes.step(action)
            # The longest observation took 250 ms; the ones of this test take next to nothing.
            scenes._observe_longest = 0.25
            scenes.step(action, deadline=10.0)
            scenes.step(action, deadline=10.0, unit="attempt")
            self.assertEqual(calls, [((), {}), ((9.75,), {}), ((9.75,), {})])
            self.assertLess(scenes._observe_seconds, 0.25)
            # The unit host-trial: the deadline less the longest observation, and the permission to return inside.
            scenes.step(action, deadline=10.0, unit="host-trial", observe_native_trials=True)
            scenes.step(action, unit="host-trial", observe_native_trials=True)
            self.assertEqual(calls[3:], [((9.75,), {"inside": True}), ((None,), {"inside": True})])

    def to_an_open_attempt(self, scenes, options, frames):
        """Whole frames, then calls of the next frame until one has left an attempt open after a trial of it."""
        scene = scenes._scene
        for frame in range(frames):
            scenes.step(np.asarray(ACTIONS[frame], dtype=float)[None, :])
        scenes.step(np.asarray(ACTIONS[frames], dtype=float)[None, :], **options)
        calls = 0
        while True:
            was_open, evaluations = scene._pending_attempt is not None, scene.coupler.evaluations
            self.assertTrue(scenes.frame_unfinished, "no attempt of the frame was left open after a trial of it")
            scenes.step(None, **options)
            calls += 1
            self.assertLess(calls, CALL_LIMIT)
            if was_open and scene._pending_attempt is not None and scene.coupler.evaluations > evaluations:
                return

    def test_a_reset_and_a_close_between_two_host_trials_keep_nothing(self):
        options = {"deadline": PASSED, "unit": "host-trial", "observe_native_trials": True}
        with _local() as scenes, _counted("_close_attempt", "abandon_step") as counts:
            scene = scenes._scene
            for how in ("reset", "close"):
                end = getattr(scenes, how)
                with self.subTest(end=how):
                    # In contact a loop takes several trials: an attempt is left between two of them.
                    self.to_an_open_attempt(scenes, options, frames=12)
                    old, closes, abandons = scene.coupler, counts["_close_attempt"], counts["abandon_step"]
                    alm, predictor = old.interaction.tunnel_alm, old._phase_load_predictor
                    self.assertTrue(old.attempt_open and alm._open and predictor.open)
                    self.assertIsNotNone(scene._host_budget)
                    configured = scene.args.fem_tolerance
                    reset_scene, close_scene, found = parallel._reset_scene, scene.close, []

                    def rebuilt(*arguments, found=found, old=old, reset_scene=reset_scene, **settings):
                        found.append(old.attempt_open)
                        return reset_scene(*arguments, **settings)

                    def released(found=found, old=old, close_scene=close_scene):
                        found.append(old.attempt_open)
                        return close_scene()

                    with patch.object(parallel, "_reset_scene", rebuilt), patch.object(scene, "close", released):
                        result = end()
                    # The attempt ended once, as a rejected one, before the scene was rebuilt or released.
                    self.assertTrue(found)
                    self.assertFalse(any(found))
                    self.assertEqual(counts["_close_attempt"], closes + 1)
                    self.assertGreaterEqual(counts["abandon_step"], abandons + 1)
                    self.assertFalse(old.attempt_open)
                    self.assertFalse(alm._open)
                    self.assertFalse(predictor.open)
                    self.assertEqual(old.fem_solver.linear_solver_tolerance, configured)
                    self.assertFalse(scenes.frame_unfinished)
                    for name in ("_pending_frame", "_pending_attempt", "_ramp_frame"):
                        self.assertIsNone(getattr(scene, name), name)
                    if how == "reset":
                        self.assertIsNot(scene.coupler, old)
                        self.assertIsNone(scene._host_budget)
                        self.assertIsNone(scene._report_before_attempt)
                        self.assertEqual(result[0]["diagnostics"]["physical_time_s"], 0.0)
                        # The frame after the reset needs an action.
                        with self.assertRaisesRegex(ValueError, r"shape \(1, 7\)"):
                            scenes.step(None)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                scenes.step(None)

    def test_a_close_whose_abandon_raises_still_closes_the_scene(self):
        with _local() as scenes:
            scene = scenes._scene
            abandon, close_scene, close_viewer = scene.abandon_attempt, scene.close, scene.viewer.close
            order = []

            def failing():
                # The explicit abandon of the pool's close raises; the scene's own close abandons again.
                order.append("abandon")
                if order.count("abandon") == 1:
                    raise RuntimeError("raised by the test")
                return abandon()

            def released():
                order.append("the scene is closed")
                return close_scene()

            def shut():
                order.append("the viewer is closed")
                return close_viewer()

            with (
                patch.object(scene, "abandon_attempt", failing),
                patch.object(scene, "close", released),
                patch.object(scene.viewer, "close", shut),
                self.assertRaisesRegex(RuntimeError, "raised by the test"),
            ):
                scenes.close()
            # The error is the caller's, and the scene and its viewer were closed all the same.
            self.assertEqual(order, ["abandon", "the scene is closed", "abandon", "the viewer is closed"])
            self.assertIsNone(scene._response_basis_provider)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                scenes.step(None)

    @unittest.skipUnless(_CUDA, "the GPU rigid backend needs a CUDA device")
    def test_the_gpu_backend_takes_the_unit_and_goes_on_with_an_attempt_it_left(self):
        action = np.asarray(ACTIONS[0], dtype=float)[None, :]
        with _local(rigid_backend="mujoco-warp") as scenes, _counted("_open_attempt", "_close_attempt") as counts:
            scene = scenes._scene
            # No explicit argument is needed: the observation reads accepted values there.
            before = scenes.observe()[0]
            result = scenes.step(action, deadline=PASSED, unit="host-trial")
            calls, opened = 1, 0
            while scenes.frame_unfinished:
                if scene._pending_attempt is not None:
                    opened += 1
                    # The accepted state of before the attempt, in every state array.
                    for name in ("needle_pose", "needle_velocity", "rigid_body_poses", "rigid_body_velocities"):
                        self.assertEqual(result[0][name].tobytes(), before[name].tobytes(), name)
                else:
                    before = result[0]
                result = scenes.step(None, deadline=PASSED, unit="host-trial")
                calls += 1
                self.assertLess(calls, 4 * CALL_LIMIT)
            # The first attempt of a scene is host-controlled (no accepted tip yet): it was left and went on.
            self.assertGreater(opened, 0)
            diagnostics = result[0]["diagnostics"]
            self.assertEqual(diagnostics["task_status"], "running")
            self.assertAlmostEqual(diagnostics["physical_time_s"], scenes.config.frame_dt, delta=1e-12)
            attempts = diagnostics["accepted_physics_steps"] + diagnostics["step_refinements"]
            self.assertEqual(counts["_open_attempt"], attempts)
            self.assertEqual(counts["_close_attempt"], attempts)
            self.assertFalse(scene.coupler.attempt_open)


def _example(calls_per_frame, budget, unit=None):
    """The example over a pool double that needs the given number of calls for each frame."""
    example = object.__new__(grasp_example.Example)
    example.args = SimpleNamespace(num_envs=1)
    example.config = SimpleNamespace(frame_dt=1.0 / 60.0, action_dimension=7)
    example.pool = _Pool(calls_per_frame, example.config.frame_dt)
    example.observations = [{"diagnostics": {"physical_time_s": 0.0}}]
    example.halted = example.render_scene = example.last_phases = example.controller = None
    example.sim_time, example.completed = 0.0, 0
    example.frame_budget = budget
    if unit is not None:
        example.frame_unit = unit
    example.viewer = _Viewer()
    return example


def _unit(budget=0.0667, **settings):
    """The unit the example takes for the given arguments, and what it prints."""
    args = SimpleNamespace(**({"rigid_backend": "mujoco-warp"} | settings))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        return grasp_example.Example._frame_unit(args, budget), output.getvalue()


class ExampleUnitTests(unittest.TestCase):
    PLAIN = {"include_tissue": False, "include_thread_diagnostics": False}

    def test_the_unit_attempt_gives_the_pool_the_keywords_it_got(self):
        # An example that took no unit (the class, before a constructor decided) has the attempt.
        self.assertEqual(grasp_example.Example.frame_unit, "attempt")
        example = _example([2, 1], 1.0 / 15.0)
        for _ in range(3):
            example.step()
        self.assertEqual(example.frame_unit, "attempt")
        for _, options in example.pool.steps:
            self.assertEqual(set(options), {*self.PLAIN, "deadline"})
        # The unit attempt named explicitly: the same keywords.
        example = _example([2, 1], 1.0 / 15.0, unit="attempt")
        for _ in range(3):
            example.step()
        for _, options in example.pool.steps:
            self.assertEqual(set(options), {*self.PLAIN, "deadline"})

    def test_the_unit_is_passed_with_a_deadline_only(self):
        example = _example([2, 1, 2], 1.0 / 15.0, unit="host-trial")
        example.step()
        # The single step of a paused viewer has neither a deadline nor the unit: it finishes its frame.
        example.viewer.paused = True
        example.step()
        example.step()
        example.viewer.paused = False
        example.step()
        options = [options for _, options in example.pool.steps]
        self.assertEqual(options[0] | {"deadline": 0.0}, {**self.PLAIN, "deadline": 0.0, "unit": "host-trial"})
        self.assertEqual(options[1:3], [self.PLAIN, self.PLAIN])
        self.assertEqual(options[3]["unit"], "host-trial")
        self.assertIn("deadline", options[3])
        # Without a budget the pool is stepped as before, whatever the unit says.
        example = _example([1, 1], None, unit="host-trial")
        example.step()
        example.step()
        self.assertEqual([options for _, options in example.pool.steps], [self.PLAIN, self.PLAIN])
        # The example takes the unit only with a budget in use: not under --test, not with a viewer
        # that is not the OpenGL window, not with several scenes (they all leave no budget). Such a
        # run has the unit as the default and did not ask for it: nothing is said about the unit, and
        # nothing about a budget that is the default one.
        window = _GLViewer()
        for viewer, settings in (
            (window, {"test": True}),
            (window, {"num_envs": 2}),
            (_other(newton.viewer.ViewerNull, num_frames=2400), {}),
            (_GLViewer(headless=True), {}),
            (_other(newton.viewer.ViewerRerun), {}),
            (window, {"frame_budget": 0.0}),
        ):
            arguments = {"num_envs": 1, "frame_budget": 66.7, "test": False}
            args = SimpleNamespace(**(arguments | settings | {"rigid_backend": "mujoco-warp"}))
            output = io.StringIO()
            with self.subTest(viewer=type(viewer).__name__, settings=settings), contextlib.redirect_stdout(output):
                budget = grasp_example.Example._frame_budget(viewer, args)
                self.assertIsNone(budget)
                self.assertEqual(grasp_example.Example._frame_unit(args, budget), "attempt")
                self.assertEqual(output.getvalue(), "")
                # The same without the budget in the arguments: it is the default.
                if args.frame_budget:
                    del args.frame_budget
                budget = grasp_example.Example._frame_budget(viewer, args)
                self.assertIsNone(budget)
                self.assertEqual(grasp_example.Example._frame_unit(args, budget), "attempt")
                self.assertEqual(output.getvalue(), "")
        # A budget other than the default that is not used says so in its own line; the unit adds none.
        args = SimpleNamespace(num_envs=1, frame_budget=50.0, test=True, rigid_backend="mujoco-warp")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            budget = grasp_example.Example._frame_budget(window, args)
            self.assertEqual(grasp_example.Example._frame_unit(args, budget), "attempt")
        self.assertEqual(output.getvalue().count("\n"), 1)
        self.assertIn("Frame budget not used: the self-test", output.getvalue())
        self.assertNotIn("unit", output.getvalue())
        # In the window, with a budget, on the GPU rigid backend, the unit is the host trial.
        args = SimpleNamespace(num_envs=1, frame_budget=66.7, test=False, rigid_backend="mujoco-warp")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            budget = grasp_example.Example._frame_budget(window, args)
            self.assertEqual(grasp_example.Example._frame_unit(args, budget), "host-trial")
        self.assertIn("Frame budget unit host-trial:", output.getvalue())

    def test_the_example_takes_its_unit_from_where_its_budget_is_in_use(self):
        pool = SimpleNamespace(
            observe=lambda: [{"diagnostics": {"physical_time_s": 0.0}}],
            geometry=lambda: {},
            finish_startup=lambda: None,
            close=lambda: None,
        )

        def args(**settings):
            defaults = dict(
                device="cpu", num_envs=1, substeps=4, controller="hold", rigid_backend="mujoco-warp", viewer="null",
                speed=0.003, settle_seconds=0.5, spacing=0.45, frame_budget=50.0,
            )  # fmt: skip
            return SimpleNamespace(**(defaults | settings))

        with (
            patch.object(grasp_example, "LocalSutureScene", lambda config: pool),
            # The GPU rigid backend is refused on a CPU device by the scene's configuration, not by the unit.
            patch.object(grasp_example, "SutureSceneConfig", lambda **settings: SimpleNamespace(**settings)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for viewer, settings, expected in (
                (_GLViewer(), {}, "host-trial"),
                (_GLViewer(), {"frame_budget": 0.0}, "attempt"),
                (_GLViewer(), {"rigid_backend": "native"}, "attempt"),
                (_GLViewer(), {"tool_tissue_contact": "swept"}, "host-trial"),
                (_GLViewer(), {"tool_tissue_contact": "off"}, "host-trial"),
                (_GLViewer(), {"test": True}, "attempt"),
                (SimpleNamespace(num_frames=10), {}, "attempt"),
            ):
                with self.subTest(viewer=type(viewer).__name__, settings=settings):
                    example = grasp_example.Example(viewer, args(**settings))
                    self.assertEqual(example.frame_unit, expected)
                    example.close()

    def test_the_native_backend_keeps_the_attempt(self):
        self.assertEqual(_unit()[0], "host-trial")
        unit, text = _unit(rigid_backend="native")
        self.assertEqual(unit, "attempt")
        self.assertIn("unit host-trial not used, the unit stays the attempt", text)
        self.assertIn("native rigid backend", text)
        self.assertIn("contacts and the clock of a trial", text)
        # Arguments without a backend (other drivers) mean the GPU backend, as for the scene's configuration.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(grasp_example.Example._frame_unit(SimpleNamespace(), 0.0667), "host-trial")

    def test_the_swept_contact_between_the_tools_and_the_tissue_takes_the_unit(self):
        # The swept contact changes its state only in the step after an acceptance (ContactHostTrialTests).
        unit, text = _unit(tool_tissue_contact="swept")
        self.assertEqual(unit, "host-trial")
        self.assertIn("Frame budget unit host-trial:", text)
        self.assertNotIn("not used", text)
        for value in (False, None, "off"):
            with self.subTest(value=value):
                self.assertEqual(_unit(tool_tissue_contact=value)[0], "host-trial")
        # The other rules stay: no budget, the native backend.
        self.assertEqual(_unit(budget=None, tool_tissue_contact="swept")[0], "attempt")
        self.assertEqual(_unit(rigid_backend="native", tool_tissue_contact="swept")[0], "attempt")
        # The example's own parser gives the value that the rule reads.
        parser = grasp_example.Example.create_parser()
        args = parser.parse_args(["--tool-tissue-contact", "swept"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(grasp_example.Example._frame_unit(args, 0.0667), "host-trial")

    def test_the_first_version_contact_is_not_a_contact_of_the_example(self):
        # The first-version law is a law of the scene; the example hands on the swept law or none,
        # and refuses any other value when it makes the arguments of its scene.
        for value in (True, "first", "penalty", "on"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "'swept' or 'off'"):
                grasp_example.Example._coupling(SimpleNamespace(tool_tissue_contact=value))

    def test_the_unit_is_no_option_and_follows_from_the_budget_and_the_rigid_backend(self):
        parser = grasp_example.Example.create_parser()
        # The unit attempt was an option while both units were run; the host trial covers it (a
        # step returns between two attempts as well), so the example has no setting for the unit.
        self.assertNotIn("frame_budget_unit", vars(parser.parse_args([])))
        self.assertFalse(hasattr(grasp_example, "FRAME_BUDGET_UNIT"))
        for value in ("attempt", "host-trial"):
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args(["--frame-budget-unit", value])
        # A setting of that name in the arguments of another driver selects nothing.
        self.assertEqual(_unit(frame_budget_unit="attempt")[0], "host-trial")
        # An example that has not been constructed has the attempt: the constructor decides where the unit applies.
        self.assertEqual(grasp_example.Example.frame_unit, "attempt")
        # Arguments of another driver and the parser's namespace give the same: host-trial with a
        # budget in use, and without one the attempt and no line.
        for budget, expected in ((0.0667, "host-trial"), (None, "attempt")):
            results = []
            for args in (SimpleNamespace(), parser.parse_args([])):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    results.append((grasp_example.Example._frame_unit(args, budget), output.getvalue()))
            with self.subTest(budget=budget):
                self.assertEqual(results[0], results[1])
                self.assertEqual(results[0][0], expected)
                self.assertEqual(results[0][1].count("\n"), 1 if budget else 0)
                if budget:
                    self.assertIn("Frame budget unit host-trial:", results[0][1])


class ExampleSceneUnitTests(unittest.TestCase):
    """The example over its real scene: the procedure, the report and the observations per frame."""

    def run_example(self, budget, frames, unit=None, single_steps=False):
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
                speed=0.003, settle_seconds=0.02, spacing=0.45, lift_height=0.0, tip_lift_height=0.0,
            )  # fmt: skip
            with patch.object(
                grasp_example, "SutureSceneConfig", lambda **settings: SutureSceneConfig(**(settings | small))
            ):
                example = grasp_example.Example(_Viewer(), args)
            try:
                example.frame_budget = budget
                if unit is not None:
                    # The example keeps the attempt on the native backend; the test takes the unit there.
                    example.frame_unit = unit
                    pool_step = example.pool.step

                    def step(actions, **options):
                        if options.get("unit") == "host-trial":
                            options["observe_native_trials"] = True
                        return pool_step(actions, **options)

                    example.pool.step = step
                scene = example.pool._scene
                record, calls, opened = [], 0, 0
                while example.completed < frames:
                    before = example.completed
                    example.viewer.paused = single_steps and calls % 2 == 1
                    example.step()
                    calls += 1
                    opened += scene._pending_attempt is not None
                    if example.viewer.paused:
                        # The single step of a paused viewer ends on a complete frame.
                        self.assertFalse(example.pool.frame_unfinished)
                        self.assertFalse(scene.coupler.attempt_open)
                        self.assertEqual(example.completed, before + 1)
                    if example.completed > before:
                        record.append((example.observations[0], example.controller.report()[0], example.sim_time))
                    self.assertLess(calls, 4 * CALL_LIMIT * frames)
                return record, calls, example.unfinished_calls, opened
            finally:
                example.close()

    def test_calls_of_the_procedure_under_the_unit_host_trial_run_one_piece_each(self):
        frames = 4
        # A budget that has always run out: a call begins an attempt or runs one host-controlled trial.
        split, calls, unfinished, opened = self.run_example(1.0e-9, frames, unit="host-trial")
        diagnostics = split[-1][0]["diagnostics"]
        attempts = diagnostics["accepted_physics_steps"] + diagnostics["step_refinements"]
        self.assertEqual(calls - unfinished, frames)
        self.assertGreaterEqual(opened, attempts)
        self.assertGreater(calls, 2 * attempts - 1)
        self.assertAlmostEqual(split[-1][2], frames / 60.0, delta=1e-9)

    @unittest.skipIf(_CUDA, _BITS)
    def test_frames_of_the_procedure_under_the_unit_host_trial_are_those_of_whole_frames(self):
        frames = 8
        whole, whole_calls, none, _ = self.run_example(None, frames)
        split, split_calls, unfinished, opened = self.run_example(1.0e-9, frames, unit="host-trial")
        stepped, _, _, stepped_open = self.run_example(1.0e-9, frames, unit="host-trial", single_steps=True)
        self.assertEqual((whole_calls, none), (frames, 0))
        # Every attempt was left open at least once: before its first host-controlled trial.
        self.assertGreaterEqual(opened, 4 * frames)
        self.assertEqual(split_calls - unfinished, frames)
        # Every single step of a paused viewer went on with an attempt that the call before had left open.
        self.assertEqual(stepped_open, frames)
        for other in (split, stepped):
            for frame, (expected, actual) in enumerate(zip(whole, other, strict=True)):
                # The observation the controller acts on, the controller's report and the time.
                _same_value(self, expected[0], actual[0], f"frame {frame} observation")
                _same_value(self, expected[1], actual[1], f"frame {frame} report")
                self.assertEqual(expected[2], actual[2])
        # The procedure moved on: the settling ended within these frames.
        self.assertNotEqual(whole[0][1]["phase"], whole[-1][1]["phase"])


SWEPT = ("--tool-tissue-contact", "swept")


def _contact_bits(law, solver):
    """What the swept contact holds, as host copies: its state between substeps and its nodes' blocks in the tissue matrix."""
    holds = law.holds
    return {
        **{f"state {index}": array.numpy().copy() for index, array in enumerate(law.state)},
        "blocks": solver.A_bsr.values.numpy()[holds._slots.numpy()].copy(),
        "inverse blocks": solver._pc_inv_diag.numpy()[holds._nodes_host].copy(),
        "blocks as assembled": holds.pristine.numpy().copy(),
    }


@contextmanager
def _loaded_contact(scene):
    """Load the swept contact by hand where the scene prepares an attempt, and count what the contact is asked.

    The tools of the scene of these tests stay clear of the tissue: the contact's own update
    after an accepted step holds no node and lets every stored force run out. Before the loads
    of every attempt are prepared, one free surface node is therefore held (with its hold in
    the tissue matrix, a push direction, an owner and a balancing force) and one axis piece
    stores a force whose reaction is on its body: what a tool on the tissue leaves behind. The
    update after the next accepted step runs on that state as the law has it.

    Yields the held node and the counts: ``loads`` (attempts that the scene prepared), and the
    calls of the contact's ``update``, ``add_forces``, ``add_balance`` and of the ``apply`` of
    its holds (the load's own ``apply`` is not counted). On a CUDA device the worker's step is
    a captured graph: ``update`` and ``apply`` count captures there, not steps.
    """
    law = scene.worker_thread.swept_tool_tissue
    particles, bodies = scene.model.particle_count, scene.model.body_count
    node, body = int(law.nodes_host[0]), int(law.sample_bodies_host[0])
    held = np.zeros(particles, dtype=np.int32)
    held[node] = 1
    push = np.zeros((particles, 3), dtype=np.float32)
    push[node] = (0.0, 0.0, -1.0)
    owner = np.full(particles, -1, dtype=np.int32)
    owner[node] = body
    balance = np.zeros((particles, 3), dtype=np.float32)
    balance[node] = (0.0, 0.0, 1.0e-3)
    stored = np.zeros((law.sample_count, 3), dtype=np.float32)
    stored[0] = (0.0, 0.0, 1.0e-2)
    wrench = np.zeros((bodies, 6), dtype=np.float32)
    wrench[body, 2] = 1.0e-2
    counts = dict.fromkeys(("loads", "update", "add_forces", "add_balance", "apply"), 0)
    prepare, apply = scene._prepare_step_loads, law.holds.apply

    def counted(name, method):
        def wrapper(*arguments, **options):
            counts[name] += 1
            return method(*arguments, **options)

        return wrapper

    def loaded(*arguments, **options):
        # Only the contact that was loaded: a rebuilt scene has another one.
        if getattr(scene.worker_thread, "swept_tool_tissue", None) is law:
            for array, values in (
                (law.held, held),
                (law.push, push),
                (law.owner, owner),
                (law.balance, balance),
                (law.stored, stored),
                (law.wrench, wrench),
            ):
                array.assign(values)
            apply()
            counts["loads"] += 1
        return prepare(*arguments, **options)

    with (
        patch.object(law, "update", counted("update", law.update)),
        patch.object(law, "add_forces", counted("add_forces", law.add_forces)),
        patch.object(law, "add_balance", counted("add_balance", law.add_balance)),
        patch.object(law.holds, "apply", counted("apply", apply)),
        patch.object(scene, "_prepare_step_loads", loaded),
    ):
        yield node, counts


class ContactHostTrialTests(unittest.TestCase):
    """An attempt left between two host-controlled trials while the swept tool/tissue contact is loaded.

    The contact changes its state in one place: its update in the worker's step after an
    acceptance. An attempt reads it where the scene prepares the attempt (the stored reaction
    on the bodies, the balancing forces of the held nodes) and in the tissue matrix (the holds
    of the held nodes). A call that returns inside an attempt, the call that goes on with it,
    and an abandon ask the contact nothing.
    """

    # Whole frames before the frame of a test: the needle is then at the tissue and a loop takes several trials.
    FRAMES = 12
    ASKED = ("loads", "update", "add_forces", "add_balance", "apply")

    def assertSameBits(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for name, value in expected.items():
            with self.subTest(name=name):
                if isinstance(value, np.ndarray):
                    self.assertEqual(value.shape, actual[name].shape)
                    self.assertEqual(value.tobytes(), actual[name].tobytes())
                else:
                    self.assertEqual(value, actual[name])

    def in_contact(self, stack):
        """The scene with the swept contact after the whole frames, with the contact loaded before every attempt from here on."""
        scene, config = stack.enter_context(_scene(example_arguments=SWEPT))
        for action in ACTIONS[: self.FRAMES]:
            _apply_action(scene, config, action)
            _step(scene)
        self.assertTrue(scene.running, scene.failure_reason)
        law = scene.worker_thread.swept_tool_tissue
        self.assertIsNotNone(law)
        node, asked = stack.enter_context(_loaded_contact(scene))
        return scene, config, law, node, asked

    def assertLoaded(self, law, solver, node):
        """The node is held with its hold in the tissue matrix, and a force is stored."""
        bits = _contact_bits(law, solver)
        index = list(law.holds._nodes_host).index(node)
        self.assertEqual(int(law.held.numpy()[node]), 1)
        self.assertGreater(np.trace(bits["blocks"][index]), 1000.0 * np.trace(bits["blocks as assembled"][index]))
        self.assertGreater(float(np.abs(law.stored.numpy()).sum()), 0.0)
        self.assertGreater(float(np.abs(law.wrench.numpy()).sum()), 0.0)
        return bits

    def between_host_trials(self, scene, config):
        """Start the next frame and leave an attempt of it open between two host-controlled trials.

        Calls with a passed deadline run one piece each under the scene's own budget, which
        never refuses the end of an attempt: a call that went on with an open attempt, ran a
        trial of it and left it open again was refused the next host-controlled trial.
        """
        _apply_action(scene, config, ACTIONS[self.FRAMES])
        for _ in range(CALL_LIMIT):
            was_open, evaluations = scene._pending_attempt is not None, scene.coupler.evaluations
            _step(scene, PASSED, inside=True)
            self.assertTrue(scene.frame_pending, "no attempt of the frame was left between two host-controlled trials")
            if was_open and scene._pending_attempt is not None and scene.coupler.evaluations > evaluations:
                return
        self.fail("the frame makes no progress")

    def test_a_call_that_goes_on_asks_the_contact_nothing_and_changes_nothing_it_holds(self):
        with ExitStack() as stack:
            scene, config, law, node, asked = self.in_contact(stack)
            coupler, worker = scene.coupler, scene.worker_thread
            accepted, refinements = scene.accepted_physics_steps, scene.step_refinements
            _apply_action(scene, config, ACTIONS[self.FRAMES])
            pairs, previous = 0, None
            for _ in range(CALL_LIMIT):
                _step(scene, PASSED, inside=True)
                # What a driver does between two calls: read the accepted state.
                _observe(scene, 0)
                attempt = None
                if scene._pending_attempt is not None:
                    attempt = (scene.accepted_physics_steps, scene.step_refinements)
                    self.assertLoaded(law, scene.solver, node)
                now = (attempt, _contact_bits(law, scene.solver), dict(asked), coupler.evaluations)
                if attempt is not None and previous is not None and previous[0] == attempt:
                    # Two calls that left the same attempt open: the second ran one host-controlled trial
                    # of it (on a CUDA device a trial of captured forces may end in the fallback to the
                    # loop without them, before it is counted). Nothing of the contact was asked, nothing
                    # was prepared, and what the contact holds (its state, the held node, the hold in the
                    # matrix) did not change.
                    pairs += 1
                    self.assertIn(now[3] - previous[3], (1,) if not _CUDA else (0, 1))
                    self.assertEqual(now[2], previous[2])
                    self.assertSameBits(previous[1], now[1])
                previous = now
                if not scene.frame_pending:
                    break
            self.assertFalse(scene.frame_pending)
            self.assertGreater(pairs, 0)
            self.assertTrue(scene.running, scene.failure_reason)
            # Over the frame: the reaction and the balance were added once per attempt, where it was
            # prepared, and the contact was updated once per accepted step.
            steps = scene.accepted_physics_steps - accepted
            attempts = steps + scene.step_refinements - refinements
            self.assertGreaterEqual(steps, 4)
            self.assertEqual((asked["loads"], asked["add_forces"], asked["add_balance"]), (attempts,) * 3)
            self.assertEqual(worker.accepted_steps, scene.accepted_physics_steps)
            if not _CUDA:
                self.assertEqual(asked["update"], steps)

    def frames(self, script):
        """Three frames in contact with the loaded contact: whole, or left where ``script`` says."""
        with ExitStack() as stack:
            scene, config, law, node, asked = self.in_contact(stack)
            if script is not None:
                scene._host_budget = _Script(script)
            snapshots, opened = [], 0
            for action in ACTIONS[self.FRAMES : self.FRAMES + 3]:
                _apply_action(scene, config, action)
                for _ in range(CALL_LIMIT):
                    if script is None:
                        _step(scene)
                    else:
                        _step(scene, None, inside=True)
                        opened += scene._pending_attempt is not None
                        _observe(scene, 0)
                    if not scene.frame_pending:
                        break
                self.assertFalse(scene.frame_pending, "a frame makes no progress")
                scene.actual_tip_history.clear()
                snapshots.append({**_snapshot(scene), **_contact_bits(law, scene.solver)})
            self.assertTrue(scene.running, scene.failure_reason)
            return snapshots, opened, dict(asked)

    @unittest.skipIf(_CUDA, _BITS)
    def test_frames_with_a_loaded_contact_left_before_every_piece_are_the_frames_of_one_call(self):
        whole, _, asked = self.frames(None)
        split, opened, asked_split = self.frames("every")
        self.assertEqual(len(whole), len(split))
        for frame, (expected, actual) in enumerate(zip(whole, split, strict=True)):
            with self.subTest(frame=frame):
                # With the thread state (the contact's seven arrays are part of it) and the hold blocks.
                self.assertSameBits(expected, actual)
        # The contact was asked the same things: once per attempt, once per accepted step.
        self.assertEqual(asked, asked_split)
        self.assertGreater(asked["update"], 0)
        # Every attempt was left before each of its trials and before its end.
        self.assertGreater(opened, 2 * asked["loads"])

    def leave_open(self, scenes, options):
        """Calls of the next frame until one has left an attempt open after a host-controlled trial of it."""
        scene = scenes._scene
        scenes.step(np.asarray(ACTIONS[self.FRAMES], dtype=float)[None, :], **options)
        for _ in range(CALL_LIMIT):
            was_open, evaluations = scene._pending_attempt is not None, scene.coupler.evaluations
            self.assertTrue(scenes.frame_unfinished, "no attempt of the frame was left open after a trial of it")
            scenes.step(None, **options)
            if was_open and scene._pending_attempt is not None and scene.coupler.evaluations > evaluations:
                return
        self.fail("the frame makes no progress")

    def test_a_reset_and_a_close_between_two_host_trials_with_nodes_held(self):
        options = {"deadline": PASSED, "unit": "host-trial", "observe_native_trials": True}
        with _local(example_arguments=SWEPT) as scenes, _counted("_close_attempt") as counts:
            scene = scenes._scene
            for how in ("reset", "close"):
                with self.subTest(end=how), ExitStack() as stack:
                    for frame in range(self.FRAMES):
                        scenes.step(np.asarray(ACTIONS[frame], dtype=float)[None, :])
                    law, solver, old = scene.worker_thread.swept_tool_tissue, scene.solver, scene.coupler
                    node, asked = stack.enter_context(_loaded_contact(scene))
                    self.leave_open(scenes, options)
                    self.assertTrue(old.attempt_open)
                    before = self.assertLoaded(law, solver, node)
                    launched, closes = dict(asked), counts["_close_attempt"]
                    result = getattr(scenes, how)()
                    # The attempt ended once, as a rejected one, and the end asked the contact nothing.
                    self.assertEqual(counts["_close_attempt"], closes + 1)
                    self.assertFalse(old.attempt_open)
                    self.assertFalse(old.interaction.tunnel_alm._open)
                    self.assertEqual(asked, launched)
                    self.assertFalse(scenes.frame_unfinished)
                    for name in ("_pending_frame", "_pending_attempt", "_ramp_frame"):
                        self.assertIsNone(getattr(scene, name), name)
                    if how == "close":
                        # The contact of a closed scene is as the last accepted step left it: the node
                        # held, its hold in the matrix, the force stored.
                        self.assertSameBits(before, _contact_bits(law, solver))
                        continue
                    # A reset builds the scene anew: another contact on another tissue solver, which holds
                    # nothing, stores nothing, and whose matrix has the blocks as assembled.
                    fresh = scene.worker_thread.swept_tool_tissue
                    self.assertIsNot(fresh, law)
                    self.assertIsNot(scene.solver, solver)
                    self.assertIsNot(scene.coupler, old)
                    self.assertIs(scene.solver._node_holds, fresh.holds)
                    for array in fresh.state:
                        np.testing.assert_array_equal(array.numpy(), -1 if array is fresh.owner else 0)
                    bits = _contact_bits(fresh, scene.solver)
                    self.assertEqual(bits["blocks"].tobytes(), bits["blocks as assembled"].tobytes())
                    self.assertIsNone(scene._host_budget)
                    self.assertIsNone(scene._report_before_attempt)
                    self.assertEqual(result[0]["diagnostics"]["physical_time_s"], 0.0)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                scenes.step(None)

    def test_an_abandon_between_two_host_trials_with_nodes_held_gives_the_attempt_back_and_asks_the_contact_nothing(
        self,
    ):
        with ExitStack() as stack:
            scene, config, law, node, asked = self.in_contact(stack)
            counts = stack.enter_context(_counted("_close_attempt"))
            coupler, started = scene.coupler, {}
            begin_step = coupler.begin_step

            def begun(*arguments, **options):
                # Where the scene has prepared an attempt: what a rejected attempt gives back, and the contact.
                started["state"], started["contact"] = _attempt_start(scene), _contact_bits(law, scene.solver)
                return begin_step(*arguments, **options)

            with patch.object(coupler, "begin_step", begun):
                self.between_host_trials(scene, config)
            self.assertTrue(coupler.attempt_open)
            # While the attempt was open the contact held what it held where the attempt was prepared.
            self.assertSameBits(started["contact"], self.assertLoaded(law, scene.solver, node))
            launched, closes = dict(asked), counts["_close_attempt"]
            scene.abandon_attempt()
            self.assertFalse(coupler.attempt_open)
            self.assertIsNone(scene._pending_attempt)
            self.assertEqual(counts["_close_attempt"], closes + 1)
            # What a rejected attempt gives back (the body forces with the reaction of the contact among them).
            _same_value(self, started["state"], _attempt_start(scene), "after the abandon")
            # The abandon asked the contact nothing: its state, the held node and its hold are as they were.
            self.assertEqual(asked, launched)
            self.assertSameBits(started["contact"], _contact_bits(law, scene.solver))
            # The frame goes on: its attempt is prepared again and the frame ends.
            self.assertTrue(scene.frame_pending)
            _step(scene)
            self.assertFalse(scene.frame_pending)
            self.assertTrue(scene.running, scene.failure_reason)
            self.assertGreater(asked["loads"], launched["loads"])
            self.assertEqual(scene.worker_thread.accepted_steps, scene.accepted_physics_steps)


if __name__ == "__main__":
    unittest.main()
