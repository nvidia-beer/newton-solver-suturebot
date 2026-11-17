# SPDX-License-Identifier: Apache-2.0
"""Trial budget of the needle/tissue coupling: the best evaluated trial is kept, on the real transaction.

The host-controlled loop runs on every test device, the captured loop on CUDA only. The
decision kernels of the captured loop are tested branch by branch in
``test_needle_coupling_graph_control``.
"""

import contextlib
import io
import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from test_needle_insertion_example import _example, _prepare
from test_needle_shaft_preconditioner import setup

from newton_solver_suturebot import NeedleCouplingNotConverged, NeedleStepTooLarge, NeedleTissueCoupler
from newton_solver_suturebot._src.solvers.coupled.interface.tunnel_alm import _TunnelALM
from newton_solver_suturebot.examples.needle import example_fem_needle_insertion as insertion

_FIELDS = ("particle_q", "particle_qd", "joint_q", "joint_qd", "body_q", "body_qd")
_DT = 0.0005


def _configure(example, mode, **options):
    """The preloaded shaft of the coupled graph tests, with the coupler options under test."""
    setup(example, _DT, 1500.0)
    example.interaction.friction = 0.02 if mode == "alm" else 0.2
    if mode == "alm":
        example.interaction.shaft_damping_per_m = 0
        example.interaction.tunnel_alm = _TunnelALM(
            example.interaction,
            penalty_per_m=1500,
            compliance_per_m=1 / 1500,
            clearance=2e-5,
            tolerance=2e-6,
        )
        example.interaction.tunnel_alm._resize()
    settings = dict(
        absolute_tolerance=2e-7,
        relative_tolerance=1e-4,
        max_iterations=160,
        shaft_preconditioner=True,
        capture_fem=True,
    )
    settings.update(options)
    example.coupler = NeedleTissueCoupler(
        example.model,
        example.solver,
        example.rigid_solver,
        example.interaction,
        example.needle_body,
        example.drive.com,
        **settings,
    )
    example.control.joint_f.assign(np.array([0, 1e-5, 0, 2e-8, 0, 0], np.float32))


@contextmanager
def _coupled(mode, **options):
    with _example() as example:
        _configure(example, mode, **options)
        yield example


@contextmanager
def _pair(mode, budget):
    """A host-controlled reference and a captured loop with the same trial budget."""
    with (
        _coupled(mode, trial_budget=budget, capture_coupling=False) as reference,
        _coupled(mode, trial_budget=budget, capture_coupling=True) as captured,
    ):
        yield reference, captured


def _record_trials(coupler):
    """Record, at the start of every trial, the residual of the trial before and the load it applies."""
    trials = []
    step_fem = coupler._step_fem

    def recorded(*args, **kwargs):
        trials.append((coupler.residual, coupler._trial_load.numpy().copy()))
        return step_fem(*args, **kwargs)

    coupler._step_fem = recorded
    return trials


def _residuals(coupler, trials):
    return [residual for residual, _ in trials[1:]] + [coupler.residual]


def _scale_correction(coupler, number, factor):
    """Scale one search direction of the host loop: the correction after trial ``number``.

    The correction only proposes the next trial load. Scaling one makes the following
    trial worse than the one before, with every force and state evaluated as usual.
    """
    preconditioner = coupler._shaft_preconditioner
    apply, calls = preconditioner.apply, [0]

    def scaled(state, dt, residual, output):
        result = apply(state, dt, residual, output)
        calls[0] += 1
        if calls[0] == number:
            output.assign(output.numpy() * factor)
        return result

    preconditioner.apply = scaled


@wp.kernel(enable_backward=False)
def _scale_direction_of_trial(state: wp.array[int], number: int, factor: wp.float64, direction: wp.array[wp.float64]):
    if state[0] == number:
        direction[wp.tid()] = factor * direction[wp.tid()]


def _scale_captured_correction(coupler, number, factor):
    """The same scaling inside the captured loop, as a kernel that is part of its graph."""
    preconditioner = coupler._shaft_preconditioner
    apply = preconditioner.apply

    def scaled(state, dt, residual, output):
        result = apply(state, dt, residual, output)
        wp.launch(
            _scale_direction_of_trial,
            output.size,
            [coupler._captured_trials.state, number, factor, output],
            device=output.device,
        )
        return result

    preconditioner.apply = scaled


def _after_trial_forces(coupler, action):
    """Call ``action(number)`` after the force evaluation of every host trial, before its residual."""
    evaluate, calls = coupler._evaluate, [0]

    def evaluated(*args, **kwargs):
        evaluate(*args, **kwargs)
        calls[0] += 1
        action(calls[0])

    coupler._evaluate = evaluated


def _report_failure_in_trial(example, number):
    """The interaction reports a failure in one eager trial, as its force evaluation does."""

    def report(trial):
        if trial == number:
            example.interaction._trial_fields.update(status="failed", reason="Stand-in failure of a trial")

    _after_trial_forces(example.coupler, report)


def _step(example):
    with patch.object(example.interaction, "accept", wraps=example.interaction.accept) as accept:
        example.coupler.step(example.state_0, example.state_1, example.control, _DT)
    return accept.call_count


class _BudgetCase(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))

    def assert_same(self, actual, expected, name):
        """Two runs of the same trials: bit for bit on the CPU, to the graph tests' bounds on CUDA."""
        if self.device.is_cuda:
            np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=5e-8, err_msg=name)
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=name)

    def assert_accepted(self, example, accepts):
        """The kept trial went through the acceptance of a converged trial."""
        coupler = example.coupler
        self.assertEqual(accepts, 1)
        for name in _FIELDS:
            np.testing.assert_array_equal(
                getattr(example.state_1, name).numpy(), getattr(coupler._candidate, name).numpy(), err_msg=name
            )
        # The carried load is the load the kept candidate state was computed with, as after
        # convergence; the force of that state differs from it by the kept residual.
        np.testing.assert_array_equal(coupler._load.numpy(), coupler._trial_load.numpy())
        difference = np.abs(coupler._value.numpy() - coupler._trial_load.numpy()).max()
        self.assertEqual(difference, coupler.residual)
        self.assertEqual(coupler.coupling_budget_residual, coupler.residual)
        self.assertEqual(coupler.maximum_residual, coupler.residual)
        if example.interaction.tunnel_alm is not None:
            self.assertFalse(example.interaction.tunnel_alm._open)

    def assert_matches_host(self, reference, captured):
        # Both loops apply the same corrections from the same start. A kept trial is not
        # converged, so the two agree to the solver tolerances, as two search paths do.
        tolerance = max(captured.coupler.relative_tolerance, captured.solver.linear_solver_tolerance)
        for name in _FIELDS:
            expected = getattr(reference.state_1, name).numpy()
            actual = getattr(captured.state_1, name).numpy()
            self.assertTrue(np.isfinite(actual).all(), name)
            np.testing.assert_allclose(
                actual, expected, rtol=0.0, atol=tolerance * np.abs(expected).max(), err_msg=name
            )
        expected = reference.coupler._trial_load.numpy()
        np.testing.assert_allclose(
            captured.coupler._trial_load.numpy(), expected, rtol=0.0, atol=tolerance * np.abs(expected).max()
        )
        # Residuals below the absolute tolerance of the convergence test are not told apart.
        np.testing.assert_allclose(
            captured.coupler.residual,
            reference.coupler.residual,
            rtol=10 * tolerance,
            atol=captured.coupler.absolute_tolerance,
        )


class HostTrialBudgetTests(_BudgetCase):
    def test_no_budget_and_a_budget_that_is_never_reached_leave_the_coupled_step_unchanged(self):
        for mode in ("penalty", "alm"):
            with self.subTest(mode=mode):
                results = []
                # No argument (the coupler of every existing test), None, and the largest budget.
                for options in ({}, {"trial_budget": None}, {"trial_budget": 159}):
                    with _coupled(mode, **options) as example:
                        coupler = example.coupler
                        self.assertEqual(coupler.trial_budget, options.get("trial_budget"))
                        trials = _record_trials(coupler)
                        self.assertEqual(_step(example), 1)
                        self.assertEqual((coupler.coupling_budget_keeps, coupler.coupling_budget_residual), (0, 0.0))
                        results.append(
                            {
                                "trials": coupler.iterations,
                                "residuals": _residuals(coupler, trials),
                                "load": coupler._load.numpy().copy(),
                                **{name: getattr(example.state_1, name).numpy().copy() for name in _FIELDS},
                            }
                        )
                # More trials than any budget of the tests below: 6 (penalty) and 19 (multipliers).
                self.assertGreater(results[0]["trials"], 5)
                for result in results[1:]:
                    self.assertEqual(result["trials"], results[0]["trials"])
                    for name in ("residuals", "load", *_FIELDS):
                        self.assert_same(result[name], results[0][name], name)

    def test_at_the_budget_the_last_trial_is_kept_when_it_is_the_best(self):
        # The residual of this loop falls from trial to trial (1.0e-3, 7.2e-4, 2.7e-4 N, measured),
        # so the third trial is the best one when a budget of three is used up.
        with _coupled("penalty", max_iterations=3) as reference, _coupled("penalty", trial_budget=3) as example:
            # The unchanged loop, stopped after three trials: its step is rejected.
            reference_trials = _record_trials(reference.coupler)
            with self.assertRaises(NeedleCouplingNotConverged):
                reference.coupler.step(reference.state_0, reference.state_1, reference.control, _DT)
            expected = _residuals(reference.coupler, reference_trials)
            self.assertEqual(reference.coupler.coupling_budget_keeps, 0)
            self.assertGreater(expected[0], expected[1])
            self.assertGreater(expected[1], expected[2])

            coupler = example.coupler
            trials = _record_trials(coupler)
            accepts = _step(example)
            self.assertEqual((coupler.iterations, coupler.coupling_budget_keeps), (3, 1))
            self.assert_same(_residuals(coupler, trials), expected, "residuals")
            # The accepted state is the candidate of that third trial, and no correction followed it.
            for name in _FIELDS:
                self.assert_same(
                    getattr(example.state_1, name).numpy(), getattr(reference.coupler._candidate, name).numpy(), name
                )
            np.testing.assert_array_equal(coupler._trial_load.numpy(), trials[2][1])
            self.assert_accepted(example, accepts)
            # Above the tolerance of the convergence test: 2e-7 N + 1e-4 of the largest entry.
            scale = max(np.abs(coupler._value.numpy()).max(), np.abs(coupler._trial_load.numpy()).max())
            self.assertGreater(coupler.residual, 2e-7 + 1e-4 * scale)

    def test_at_the_budget_an_earlier_best_load_is_applied_once_more_and_that_trial_is_kept(self):
        with _coupled("penalty", trial_budget=3) as example:
            coupler = example.coupler
            # The search direction after the second trial is reversed and doubled, so the third
            # trial is worse than the second (9.2e-4 against 7.2e-4 N, measured).
            _scale_correction(coupler, 2, -2.0)
            trials = _record_trials(coupler)
            accepts = _step(example)
            residuals = _residuals(coupler, trials)
            self.assertEqual((coupler.iterations, coupler.coupling_budget_keeps), (4, 1))
            self.assertGreater(residuals[0], residuals[1])
            self.assertGreater(residuals[2], residuals[1])
            # One more trial, at the load of the second trial bit for bit, with nothing after it.
            np.testing.assert_array_equal(trials[3][1], trials[1][1])
            np.testing.assert_array_equal(coupler._trial_load.numpy(), trials[1][1])
            self.assertFalse(np.array_equal(trials[2][1], trials[1][1]))
            # It is kept whatever its residual. Evaluated from another tissue-solver guess it
            # shows the residual of the second trial again, below that of the third.
            self.assertEqual(coupler.residual, residuals[3])
            self.assertLess(residuals[3], residuals[2])
            np.testing.assert_allclose(residuals[3], residuals[1], rtol=example.solver.linear_solver_tolerance)
            self.assert_accepted(example, accepts)

    def test_of_two_trials_with_equal_residual_the_later_one_is_the_best(self):
        with _coupled("penalty", trial_budget=3) as example:
            coupler = example.coupler
            # No correction after the second trial, and the third trial is given the forces of
            # the second: it applies the same load and shows the same residual, bit for bit.
            _scale_correction(coupler, 2, 0.0)
            saved = {}

            def repeat(trial):
                if trial == 2:
                    saved["forces"] = wp.clone(coupler._forces)
                    saved["wrench"] = wp.clone(coupler._candidate.body_f)
                elif trial == 3:
                    wp.copy(coupler._forces, saved["forces"])
                    wp.copy(coupler._candidate.body_f, saved["wrench"])

            _after_trial_forces(coupler, repeat)
            trials = _record_trials(coupler)
            accepts = _step(example)
            residuals = _residuals(coupler, trials)
            self.assertGreater(residuals[0], residuals[1])
            self.assertEqual(residuals[2], residuals[1])
            np.testing.assert_array_equal(trials[2][1], trials[1][1])
            # The third trial is the best one and is kept as it is: no trial is spent on
            # applying the load of the second once more.
            self.assertEqual((coupler.iterations, coupler.coupling_budget_keeps), (3, 1))
            self.assertEqual(accepts, 1)
            self.assertEqual(coupler.coupling_budget_residual, residuals[2])

    def test_a_host_trial_in_which_the_interaction_reports_a_failure_is_never_kept(self):
        cases = (
            # The trial that uses up the budget is the best one, but reports a failure.
            ("last trial", "penalty", 3, None, 3),
            # The third trial is worse than the second (see the test above); the trial that
            # applies the load of the second once more reports a failure.
            ("best load applied once more", "penalty", 3, -2.0, 4),
            # The fourth trial passes the force test and waits for a multiplier update.
            ("waiting for a multiplier update", "alm", 4, None, 4),
        )
        for name, mode, budget, factor, failing in cases:
            with (
                self.subTest(case=name),
                _coupled(mode) as reference,
                _coupled(mode, trial_budget=budget) as example,
            ):
                for item in (reference, example):
                    if factor is not None:
                        _scale_correction(item.coupler, 2, factor)
                    # The same hook in both, so that both run the same trial implementation.
                    _report_failure_in_trial(item, failing if item is example else 0)
                self.assertEqual(_step(reference), 1)
                coupler = example.coupler
                trials = _record_trials(coupler)
                self.assertEqual(_step(example), 1)
                # The loop went on as without a budget: nothing was kept, and it converged.
                self.assertGreater(coupler.iterations, failing)
                self.assertEqual(coupler.coupling_budget_keeps, 0)
                self.assertEqual((coupler.coupling_budget_residual, coupler.coupling_budget_events), (0.0, {}))
                self.assertNotEqual(example.interaction.status, "failed")
                scale = max(np.abs(coupler._value.numpy()).max(), np.abs(coupler._trial_load.numpy()).max())
                self.assertLessEqual(coupler.residual, 2e-7 + 1e-4 * scale)
                self.assertEqual(len(trials), coupler.iterations)
                if factor is None:
                    # Without a restored load the trials are those of the loop without a budget.
                    self.assertEqual(coupler.iterations, reference.coupler.iterations)
                    for field in _FIELDS:
                        self.assert_same(
                            getattr(example.state_1, field).numpy(), getattr(reference.state_1, field).numpy(), field
                        )

    def test_a_kept_trial_that_the_step_refuses_is_counted_as_a_loop_and_not_in_the_kept_residual(self):
        with _coupled("penalty", trial_budget=3) as example:
            coupler = example.coupler
            load = coupler._load.numpy().copy()
            # The check that follows every loop refuses the kept trial, as it can a converged one.
            with (
                patch.object(example.interaction, "check_step_resolution", side_effect=NeedleStepTooLarge("late")),
                patch.object(example.interaction, "accept", wraps=example.interaction.accept) as accept,
            ):
                with self.assertRaisesRegex(NeedleStepTooLarge, "late"):
                    coupler.step(example.state_0, example.state_1, example.control, _DT)
                self.assertEqual(accept.call_count, 0)
            refused = coupler.residual
            self.assertEqual((coupler.iterations, coupler.coupling_budget_keeps), (3, 1))
            self.assertEqual((coupler.coupling_budget_residual, coupler.maximum_residual), (0.0, 0.0))
            np.testing.assert_array_equal(coupler._load.numpy(), load)
            # The retry at half the step, as the scene makes it, is kept and accepted.
            example.drive.prepare_timestep(_DT / 2)
            example.solver.refresh_tangent(example.state_0.particle_q, _DT / 2)
            with patch.object(example.interaction, "accept", wraps=example.interaction.accept) as accept:
                coupler.step(example.state_0, example.state_1, example.control, _DT / 2)
                self.assertEqual(accept.call_count, 1)
            self.assertEqual(coupler.coupling_budget_keeps, 2)
            self.assertNotEqual(coupler.residual, refused)
            self.assertEqual((coupler.coupling_budget_residual, coupler.maximum_residual), (coupler.residual,) * 2)
            self.assertEqual(coupler.coupling_budget_events, {})

    def test_a_converged_loop_after_a_kept_trial_that_could_not_be_staged_is_not_a_kept_step(self):
        from newton_solver_suturebot._src.solvers.coupled.host_trial_forces import TrialForceFallback

        with _coupled("penalty", trial_budget=3) as example:
            coupler = example.coupler
            host, loops = coupler._solve_host, []

            def staged(*args, **kwargs):
                converged = host(*args, **kwargs)
                loops.append((coupler.iterations, coupler.coupling_budget_keeps))
                if len(loops) == 1:
                    # The kept trial cannot be staged, as captured host forces report it: the
                    # step starts again through the oracle, which converges without a budget.
                    coupler.trial_budget = None
                    raise TrialForceFallback()
                return converged

            coupler._solve_host = staged
            self.assertEqual(_step(example), 1)
            # The first loop is counted as a loop the budget ended. The accepted trial is the
            # converged one of the second loop: it is not a kept step.
            self.assertEqual(loops[0], (3, 1))
            self.assertEqual(len(loops), 2)
            self.assertGreater(loops[1][0], 3)
            self.assertEqual(coupler.coupling_budget_keeps, 1)
            self.assertEqual((coupler.coupling_budget_residual, coupler.coupling_budget_events), (0.0, {}))
            self.assertGreater(coupler.maximum_residual, 0.0)
            scale = max(np.abs(coupler._value.numpy()).max(), np.abs(coupler._trial_load.numpy()).max())
            self.assertLessEqual(coupler.residual, 2e-7 + 1e-4 * scale)

    def test_the_best_trial_restarts_at_a_multiplier_update(self):
        # Measured residuals of this loop [N]: 9.8e-4, 6.9e-4, 6.3e-7, 3.5e-9 (passes the force
        # test of 3.0e-7: multiplier update), then 3.3e-6 and 2.3e-6 in the new epoch.
        with _coupled("alm") as reference, _coupled("alm", trial_budget=6) as example:
            reference_trials = _record_trials(reference.coupler)
            self.assertEqual(_step(reference), 1)
            expected = _residuals(reference.coupler, reference_trials)
            # The epoch before the update holds smaller residuals than the sixth trial.
            self.assertLess(min(expected[:4]), expected[5])
            self.assertLess(expected[5], expected[4])

            coupler, alm = example.coupler, example.interaction.tunnel_alm
            trials = _record_trials(coupler)
            accepts = _step(example)
            # The sixth trial is the best of its own epoch and is kept as it is. A best trial
            # carried over the update would have brought back the load of the fourth trial.
            self.assertEqual((coupler.iterations, coupler.coupling_budget_keeps, alm.total_dual_updates), (6, 1, 1))
            self.assert_same(_residuals(coupler, trials), expected[:6], "residuals")
            np.testing.assert_array_equal(coupler._trial_load.numpy(), trials[5][1])
            self.assert_accepted(example, accepts)

    def test_a_trial_that_only_waits_for_a_multiplier_update_is_kept_at_the_budget_without_the_update(self):
        with _coupled("alm") as reference, _coupled("alm", trial_budget=4) as example:
            reference_trials = _record_trials(reference.coupler)
            self.assertEqual(_step(reference), 1)
            expected = _residuals(reference.coupler, reference_trials)
            self.assertGreater(reference.interaction.tunnel_alm.total_dual_updates, 0)

            coupler, alm = example.coupler, example.interaction.tunnel_alm
            trials = _record_trials(coupler)
            accepts = _step(example)
            # The fourth trial passes the force test; the unchanged loop updates the multipliers
            # there. At the budget it is kept, and the multipliers of that trial are committed.
            scale = max(np.abs(coupler._value.numpy()).max(), np.abs(coupler._trial_load.numpy()).max())
            self.assertLessEqual(coupler.residual, 2e-7 + 1e-4 * scale)
            self.assertEqual((coupler.iterations, coupler.coupling_budget_keeps, alm.total_dual_updates), (4, 1, 0))
            self.assert_same(_residuals(coupler, trials), expected[:4], "residuals")
            self.assertGreater(np.abs(alm.committed_normal.numpy()).max(), 0.0)
            self.assert_accepted(example, accepts)

    def test_a_failed_loop_is_never_kept(self):
        # Each failure happens in the third trial, the one that uses up a budget of three.
        cases = (
            # The scaled correction throws the needle tip beyond the step resolution.
            ("tip moved", {}, 1e8, NeedleStepTooLarge, "Unresolved tip displacement"),
            # Relative tolerance 1e-2: the applied load has grown by more than 100.
            ("divergence", {"relative_tolerance": 1e-2}, 3e5, NeedleCouplingNotConverged, "diverged after 3 trials"),
            ("nonfinite", {}, 1.0, NeedleCouplingNotConverged, "Nonfinite"),
        )
        for reason, options, factor, error, message in cases:
            with self.subTest(reason=reason), _coupled("penalty", trial_budget=3, **options) as example:
                coupler = example.coupler
                _scale_correction(coupler, 2, factor)
                if reason == "nonfinite":
                    evaluate, calls = coupler._evaluate, [0]

                    def poisoned(*args, evaluate=evaluate, calls=calls, coupler=coupler, **kwargs):
                        evaluate(*args, **kwargs)
                        calls[0] += 1
                        if calls[0] == 3:
                            forces = coupler._forces.numpy()
                            forces[0, 0] = np.nan
                            coupler._forces.assign(forces)

                    coupler._evaluate = poisoned
                before = {name: getattr(example.state_1, name).numpy().copy() for name in _FIELDS}
                load = coupler._load.numpy().copy()
                with patch.object(example.interaction, "accept", wraps=example.interaction.accept) as accept:
                    with self.assertRaisesRegex(error, message):
                        coupler.step(example.state_0, example.state_1, example.control, _DT)
                    self.assertEqual(accept.call_count, 0)
                self.assertEqual((coupler.coupling_budget_keeps, coupler.coupling_budget_residual), (0, 0.0))
                self.assertEqual(coupler.coupling_divergence_stops, int(reason == "divergence"))
                for name, values in before.items():
                    np.testing.assert_array_equal(getattr(example.state_1, name).numpy(), values, err_msg=name)
                np.testing.assert_array_equal(coupler._load.numpy(), load)

    def test_device_search_past_a_surface_event_that_the_budget_ends_goes_to_the_host_oracle(self):
        from newton_solver_suturebot._src.solvers.coupled.interface.needle_trial_forces import FALLBACK_REASONS

        # A search past a surface event whose last trial carries a request: ended by the budget
        # (kept flag of the loop set) and converged. Both hand their load to the host oracle.
        for ended_by_budget in (True, False):
            with self.subTest(ended_by_budget=ended_by_budget), _coupled("penalty", trial_budget=3) as example:
                coupler = example.coupler
                seen = {}

                class Search:
                    """Stands for the captured loop and its search past a surface event."""

                    captures = launches = 1

                    def __init__(self, coupler=coupler):
                        flags = np.zeros(len(FALLBACK_REASONS), dtype=np.int32)
                        flags[[0, FALLBACK_REASONS.index("tip sweep not clear of the surface")]] = 1
                        self.seen = wp.array(flags, dtype=int, device=coupler.model.device)
                        self.permissions = SimpleNamespace(fallback=self.seen)

                    def solve(self, state_in, control, dt, coupler=coupler, seen=seen, kept=ended_by_budget):
                        # The load of the last trial of the search: the accepted load, changed.
                        load = coupler._trial_load.numpy()
                        load[-6] += 1e-4
                        coupler._trial_load.assign(load)
                        seen["search"] = load.copy()
                        result = np.zeros(13)
                        # Three trials, residual 5e-2 N; the last trial carries a request, and the
                        # first request came in the second trial.
                        result[0], result[3], result[7], result[10], result[11], result[12] = 3, 1, 5e-2, 1, 2, kept
                        return result

                host = coupler._solve_host

                def started(*args, host=host, coupler=coupler, seen=seen, **kwargs):
                    if "host" not in seen:
                        seen["host"] = coupler._trial_load.numpy().copy()
                        seen["oracle"] = not kwargs.get("capture_forces", True)
                        seen["counters"] = (
                            coupler.coupling_budget_keeps,
                            coupler.coupling_budget_searches,
                            coupler.coupling_graph_event_searches,
                            coupler.coupling_budget_residual,
                        )
                    return host(*args, **kwargs)

                coupler._device_trials = lambda state_in, control, dt: Search()
                coupler._solve_host = started
                accepted = coupler._load.numpy().copy()
                self.assertEqual(_step(example), 1)
                # The device loop accepted nothing: the oracle ran, started from the load of the search.
                self.assertTrue(seen["oracle"])
                np.testing.assert_array_equal(seen["host"], seen["search"])
                self.assertFalse(np.array_equal(seen["search"], accepted))
                # A search the budget ended is counted as such: it is neither a kept trial nor a
                # converged search, and its residual is not that of a kept trial.
                self.assertEqual(seen["counters"], (0, 1, 0, 0.0) if ended_by_budget else (0, 0, 1, 0.0))
                self.assertEqual(coupler.coupling_graph_status, "exact surface-event fallback")
                self.assertEqual(coupler.coupling_graph_event_restarts, 0)
                # The oracle decided the step with its own loop, which the budget of three ended.
                # Only that accepted trial is in the residual of the kept trials.
                self.assertEqual(
                    (coupler.coupling_budget_keeps, coupler.coupling_budget_searches), (1, int(ended_by_budget))
                )
                self.assertLess(coupler.residual, 5e-2)
                self.assertEqual(coupler.coupling_budget_residual, coupler.residual)
                self.assertEqual(coupler.maximum_residual, coupler.residual)


class TrialBudgetArgumentTests(_BudgetCase):
    def test_coupler_accepts_none_or_an_integer_from_two_to_below_the_trial_limit(self):
        with _coupled("penalty") as example:

            def coupler(**options):
                return NeedleTissueCoupler(
                    example.model,
                    example.solver,
                    example.rigid_solver,
                    example.interaction,
                    example.needle_body,
                    example.drive.com,
                    **options,
                )

            self.assertIsNone(example.coupler.trial_budget)
            self.assertIsNone(coupler().trial_budget)
            # The default limit is 40 trials.
            for value, expected in ((None, None), (2, 2), (39, 39), (np.int32(12), 12)):
                made = coupler(trial_budget=value)
                self.assertEqual(made.trial_budget, expected)
                self.assertEqual((made.coupling_budget_keeps, made.coupling_budget_residual), (0, 0.0))
            self.assertIs(type(coupler(trial_budget=np.int32(12)).trial_budget), int)
            self.assertEqual(coupler(trial_budget=159, max_iterations=160).trial_budget, 159)
            for value in (1, 0, -3, 40, 41, 2.0, 12.5, "12", True, float("nan")):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "trial_budget"):
                    coupler(trial_budget=value)
            with self.assertRaisesRegex(ValueError, "trial_budget"):
                coupler(trial_budget=160, max_iterations=160)

    def test_insertion_example_has_no_budget_by_default_and_passes_one_to_the_coupler(self):
        parser = insertion.Example.create_parser()
        self.assertIsNone(parser.parse_args([]).trial_budget)
        self.assertEqual(parser.parse_args([]).coupling_relative_tolerance, 1e-3)
        self.assertEqual(parser.parse_args(["--trial-budget", "12"]).trial_budget, 12)
        for off in ("0", "00", "none", "None"):
            self.assertIsNone(parser.parse_args(["--trial-budget", off]).trial_budget)
        # From 2 up to below the trial limit of the example's coupler.
        self.assertEqual(insertion.COUPLING_TRIAL_LIMIT, 160)
        self.assertEqual(parser.parse_args(["--trial-budget", "2"]).trial_budget, 2)
        self.assertEqual(parser.parse_args(["--trial-budget", "159"]).trial_budget, 159)
        for value in ("1", "-2", "2.5", "twelve", "160", "1000"):
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args(["--trial-budget", value])
        for value, expected in ((None, None), (7, 7)):
            with self.subTest(value=value), _example() as example:
                self.assertIsNone(example.args.trial_budget)
                example.args.trial_budget = value
                _prepare(example)
                example.start()
                self.assertEqual(example.coupler.trial_budget, expected)
                report = example.report()
                self.assertEqual(report["coupling_trial_budget"], expected)
                self.assertEqual(example.coupler.max_iterations, insertion.COUPLING_TRIAL_LIMIT)
                names = ("keeps", "searches", "residual", "events")
                self.assertEqual([report[f"coupling_budget_{name}"] for name in names], [0, 0, 0.0, {}])
                coupler = example.coupler
                coupler.coupling_budget_keeps, coupler.coupling_budget_searches = 3, 1
                coupler.coupling_budget_residual, coupler.coupling_budget_events = 2.5e-4, {"puncture": 1}
                report = example.report()
                self.assertEqual([report[f"coupling_budget_{name}"] for name in names], [3, 1, 2.5e-4, {"puncture": 1}])
                # The report holds its own copy of the event counts.
                self.assertIsNot(report["coupling_budget_events"], coupler.coupling_budget_events)
        # A value that does not come through the parser: at or above the trial limit the coupler refuses.
        with _example() as example:
            example.args.trial_budget = 160
            _prepare(example)
            with self.assertRaisesRegex(ValueError, "trial_budget"):
                example.start()


class CapturedTrialBudgetTests(_BudgetCase):
    """The captured loop keeps the trial the host-controlled loop keeps."""

    def setUp(self):
        super().setUp()
        if not self.device.is_cuda:
            self.skipTest("The captured trial loop requires CUDA")

    def step_pair(self, reference, captured):
        """Step both loops once; returns the trials of the captured loop."""
        accepts = [_step(example) for example in (reference, captured)]
        c = captured.coupler
        self.assertEqual(reference.coupler.coupling_graph_launches, 0)
        self.assertEqual((c.coupling_graph_launches, c.coupling_graph_fallbacks), (1, 0))
        self.assertEqual(c.coupling_graph_status, "replaying")
        self.assertEqual(c.iterations, reference.coupler.iterations)
        self.assertEqual((reference.coupler.coupling_budget_keeps, c.coupling_budget_keeps), (1, 1))
        self.assert_matches_host(reference, captured)
        for example, count in zip((reference, captured), accepts, strict=True):
            self.assert_accepted(example, count)
        return c.iterations

    def test_last_trial_is_kept_at_the_budget_as_by_the_host_loop(self):
        for mode in ("penalty", "alm"):
            with self.subTest(mode=mode), _pair(mode, 3) as (reference, captured):
                self.assertEqual(self.step_pair(reference, captured), 3)

    def test_trial_waiting_for_a_multiplier_update_is_kept_at_the_budget_as_by_the_host_loop(self):
        with _pair("alm", 4) as (reference, captured):
            self.assertEqual(self.step_pair(reference, captured), 4)
            for example in (reference, captured):
                self.assertEqual(example.interaction.tunnel_alm.total_dual_updates, 0)

    def test_best_trial_restarts_at_a_multiplier_update_as_in_the_host_loop(self):
        with _pair("alm", 6) as (reference, captured):
            self.assertEqual(self.step_pair(reference, captured), 6)
            for example in (reference, captured):
                self.assertEqual(example.interaction.tunnel_alm.total_dual_updates, 1)

    def test_earlier_best_load_is_applied_once_more_inside_the_graph(self):
        with _pair("penalty", 3) as (reference, captured):
            # The search direction after the second trial is reversed and doubled in both loops.
            _scale_correction(reference.coupler, 2, -2.0)
            _scale_captured_correction(captured.coupler, 2, -2.0)
            self.assertEqual(self.step_pair(reference, captured), 4)
            # The kept trial ran at the saved load of the best trial, and no correction followed.
            c = captured.coupler
            np.testing.assert_array_equal(c._trial_load.numpy(), c._captured_trials.best_load.numpy())

    def test_budget_that_is_never_reached_converges_as_without_one(self):
        with _pair("alm", 159) as (reference, captured):
            for example in (reference, captured):
                self.assertEqual(_step(example), 1)
                self.assertEqual(
                    (example.coupler.coupling_budget_keeps, example.coupler.coupling_budget_residual), (0, 0.0)
                )
            self.assertEqual(captured.coupler.coupling_graph_launches, 1)
            self.assertEqual(captured.coupler.coupling_graph_fallbacks, 0)
            for name in _FIELDS:
                np.testing.assert_allclose(
                    getattr(captured.state_1, name).numpy(),
                    getattr(reference.state_1, name).numpy(),
                    rtol=3e-6,
                    atol=5e-8,
                    err_msg=name,
                )

    def test_failed_device_loop_is_never_kept(self):
        with _pair("penalty", 3) as (_, captured):
            c = captured.coupler
            # The scaled correction throws the needle in the trial that uses up the budget.
            _scale_captured_correction(c, 2, 1e8)
            before = {name: getattr(captured.state_1, name).numpy().copy() for name in _FIELDS}
            with patch.object(captured.interaction, "accept", wraps=captured.interaction.accept) as accept:
                with self.assertRaises((NeedleStepTooLarge, NeedleCouplingNotConverged)):
                    c.step(captured.state_0, captured.state_1, captured.control, _DT)
                self.assertEqual(accept.call_count, 0)
            self.assertEqual(c.coupling_graph_launches, 1)
            self.assertEqual(c.iterations, 3)
            self.assertEqual((c.coupling_budget_keeps, c.coupling_budget_residual), (0, 0.0))
            for name, values in before.items():
                np.testing.assert_array_equal(getattr(captured.state_1, name).numpy(), values, err_msg=name)


class HostForceGraphTrialBudgetTests(_BudgetCase):
    """Host-controlled trials with captured force evaluation keep the trial the eager host loop keeps."""

    def setUp(self):
        super().setUp()
        if not self.device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("Captured host trial forces require conditional CUDA graphs")

    def test_kept_trial_is_staged_from_the_captured_forces(self):
        # Residuals of these host loops fall from trial to trial (penalty), and the fourth trial
        # of the multiplier loop passes the force test and waits for an update (measured on the
        # CPU: 1.0e-3, 7.2e-4, 1.0e-4 N and 9.8e-4, 6.9e-4, 1.9e-6, 1.2e-8 N).
        for mode, budget in (("penalty", 3), ("alm", 4)):
            with self.subTest(mode=mode), _pair(mode, budget) as (reference, captured):
                for example in (reference, captured):
                    # The host/native trial loop of articulated bodies, as in test_host_trial_forces:
                    # no device trials, and captured forces where capture_coupling is set.
                    example.coupler._rigid_bridge = None
                    example.coupler._shaft_preconditioner = None
                accepts = [_step(example) for example in (reference, captured)]
                c, graph = captured.coupler, captured.coupler._host_trial_forces
                self.assertIsNone(reference.coupler._host_trial_forces)
                # Every trial evaluated its forces in the graph, and the kept trial was staged
                # from them: no trial went to the eager oracle.
                self.assertEqual((graph.launches, graph.fallbacks), (budget, 0))
                self.assertEqual((reference.coupler.coupling_graph_launches, c.coupling_graph_launches), (0, 0))
                self.assertEqual((reference.coupler.iterations, c.iterations), (budget, budget))
                self.assertEqual((reference.coupler.coupling_budget_keeps, c.coupling_budget_keeps), (1, 1))
                self.assert_matches_host(reference, captured)
                for example, count in zip((reference, captured), accepts, strict=True):
                    self.assert_accepted(example, count)
                    if mode == "alm":
                        self.assertEqual(example.interaction.tunnel_alm.total_dual_updates, 0)


if __name__ == "__main__":
    unittest.main()
