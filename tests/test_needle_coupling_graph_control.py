# SPDX-License-Identifier: Apache-2.0
"""Control-flow parity against the currently shipped host coupling rules."""

import os
import unittest

import numpy as np
import warp as wp

from newton_solver_suturebot._src.solvers.coupled import coupling_graph as control


class DeviceControlTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.state = wp.zeros(6, dtype=int, device=self.device)
        self.values = wp.zeros(5, dtype=wp.float64, device=self.device)
        self.statistics = wp.zeros(6, dtype=wp.float64, device=self.device)
        self.alm = wp.zeros(5, dtype=float, device=self.device)
        self.fallback = wp.zeros(1, dtype=int, device=self.device)
        self.searchable = wp.zeros(1, dtype=int, device=self.device)
        self.seen = wp.zeros(1, dtype=int, device=self.device)
        self.running = wp.zeros(1, dtype=int, device=self.device)
        self.correct = wp.zeros(1, dtype=int, device=self.device)
        self.dual = wp.zeros(1, dtype=int, device=self.device)
        self.fresh = wp.zeros(1, dtype=int, device=self.device)
        self.stale = wp.zeros(1, dtype=int, device=self.device)
        self.levels = wp.zeros(5, dtype=int, device=self.device)
        self.initial = 0.3
        # Trial budget: book, smallest force residual of the multiplier epoch, the trial
        # load and the load of the best trial.
        self.budget = wp.zeros(5, dtype=int, device=self.device)
        self.best_residual = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.load = wp.zeros(3, dtype=wp.float64, device=self.device)
        self.best_load = wp.zeros(3, dtype=wp.float64, device=self.device)
        self.result = wp.zeros(13, dtype=wp.float64, device=self.device)

    def initialize(self):
        wp.launch(
            control._initialize,
            1,
            [self.state, self.values, self.running, self.fresh, self.stale, self.initial, self.budget],
            device=self.device,
        )

    def assess(self, *, limit=40, has_alm=1, trial_budget=0):
        wp.launch(
            control._assess_trial,
            1,
            [
                self.state,
                self.values,
                self.statistics,
                self.alm,
                self.fallback,
                self.searchable,
                self.seen,
                self.running,
                self.correct,
                self.dual,
                self.fresh,
                self.stale,
                2e-6,
                1e-3,
                self.initial,
                2e-5,
                has_alm,
                limit,
                trial_budget,
                self.best_residual,
                self.budget,
            ],
            device=self.device,
        )

    def test_tolerance_schedule_aitken_and_dual_restart_match_host(self):
        rng = np.random.default_rng(9218)
        self.initialize()
        host_relaxation, mixing_start, level, dual_count = self.initial, 1, 0, 0
        for iteration in range(1, 41):
            if iteration > mixing_start and (iteration - mixing_start) % 8 == 0 and level < 4:
                level += 1
                mixing_start = iteration
                host_relaxation = self.initial
            wp.launch(
                control._begin_trial, 1, [self.state, self.values, self.levels, 5, self.initial], device=self.device
            )
            primal_converged = iteration in (12, 14, 40)
            stats = np.array(
                [1e-7 if primal_converged else 1e-3, 1e-3, 2e-3, rng.uniform(1e-8, 1), rng.uniform(-2, 2), 0.0]
            )
            alm = np.array([0.0, 0.0, 1e-3 if iteration != 40 else 0.0, 1e-3, 0.0], dtype=np.float32)
            self.statistics.assign(stats)
            self.alm.assign(alm)
            self.assess()
            if primal_converged:
                if iteration != 40:
                    dual_count += 1
                    mixing_start = iteration + 1
                    host_relaxation = self.initial
            else:
                if iteration > mixing_start and stats[3] > 1e-30:
                    host_relaxation = float(np.clip(-host_relaxation * stats[4] / stats[3], 0.001, 2.0))
                wp.launch(
                    control._update_relaxation, 1, [self.state, self.values, self.statistics, 1], device=self.device
                )
            np.testing.assert_array_equal(
                self.state.numpy(), [iteration, mixing_start, level, int(iteration == 40), 0, dual_count]
            )
            np.testing.assert_allclose(self.values.numpy()[:2], [host_relaxation, stats[0]], rtol=1e-14, atol=1e-15)
            self.assertEqual(int(self.correct.numpy()[0]), int(not primal_converged))
            self.assertEqual(int(self.dual.numpy()[0]), int(primal_converged and iteration != 40))
            self.assertEqual(int(self.running.numpy()[0]), int(iteration != 40))
            # A dual-only update repeats the unchanged candidate: the next
            # trial must skip its primal/geometry stage, and only then.
            self.assertEqual(int(self.fresh.numpy()[0]), int(not (primal_converged and iteration != 40)))
            self.assertEqual(int(self.stale.numpy()[0]), int(primal_converged and iteration != 40))
            np.testing.assert_array_equal(self.levels.numpy(), np.arange(5) == level)

    def test_relaxation_bound_follows_the_correction_flag_that_the_free_needle_graph_holds(self):
        # The loop of a rigid adapter without a correction of its own passes the flag of the shaft
        # correction to this kernel as a launch argument, so the value is part of the captured
        # graph: with the correction the Aitken relaxation is bounded by 2, without it by 1, as in
        # the host loop. That is why the flag stays an entry of that adapter's trial signature.
        for flag, upper in ((1, 2.0), (0, 1.0)):
            for ratio in (-0.5, -3.0, -40.0, 5.0):
                with self.subTest(flag=flag, ratio=ratio):
                    self.initialize()
                    for _ in range(2):
                        wp.launch(
                            control._begin_trial,
                            1,
                            [self.state, self.values, self.levels, 5, self.initial],
                            device=self.device,
                        )
                    self.statistics.assign(np.array([1e-3, 1e-3, 2e-3, 1.0, ratio, 0.0]))
                    wp.launch(
                        control._update_relaxation,
                        1,
                        [self.state, self.values, self.statistics, flag],
                        device=self.device,
                    )
                    expected = float(np.clip(-self.initial * ratio, 0.001, upper))
                    self.assertEqual(float(self.values.numpy()[0]), expected)

    def trial(self, residual, load, *, force=1e-3, has_alm=1):
        wp.launch(control._begin_trial, 1, [self.state, self.values, self.levels, 5, self.initial], device=self.device)
        self.statistics.assign(np.array([residual, force, load, 1.0, -1.0, 0.0]))
        self.assess(has_alm=has_alm)
        return int(self.state.numpy()[4]), int(self.running.numpy()[0]), int(self.correct.numpy()[0])

    def test_trials_end_when_the_applied_load_has_grown_by_one_over_the_relative_tolerance(self):
        # Tolerances of assess(): absolute 2e-6, relative 1e-3, so the factor is 1000.
        self.initialize()
        self.assertEqual(self.trial(1e-2, 0.5), (0, 1, 1))
        self.assertEqual(float(self.values.numpy()[4]), 0.5)
        # A large force response does not end the trials; only the applied load does.
        self.assertEqual(self.trial(1e4, 499.0, force=1e6), (0, 1, 1))
        self.assertEqual(self.trial(10.0, 500.0), (0, 1, 1))
        # Past the factor the trials end without a correction and without converging.
        self.assertEqual(self.trial(10.0, 500.5), (2, 0, 0))
        self.assertEqual(int(self.state.numpy()[3]), 0)
        self.assertEqual(float(self.values.numpy()[4]), 0.5)

    def test_divergence_reference_has_the_absolute_tolerance_floor_and_does_not_stop_convergence(self):
        # A loop that starts from no load refers to absolute / relative tolerance = 2e-3 N.
        self.initialize()
        self.assertEqual(self.trial(1e-2, 0.0), (0, 1, 1))
        self.assertAlmostEqual(float(self.values.numpy()[4]), 2e-3, delta=1e-18)
        self.assertEqual(self.trial(1.0, 1.99), (0, 1, 1))
        self.assertEqual(self.trial(1.0, 2.01), (2, 0, 0))
        # The convergence test comes first: a trial that meets its tolerance converges
        # whatever its load. Without tunnel multipliers that ends the trials as converged.
        self.initialize()
        self.assertEqual(self.trial(1e-2, 1e-3), (0, 1, 1))
        self.assertEqual(self.trial(1e-9, 50.0, has_alm=0), (0, 0, 0))
        self.assertEqual(int(self.state.numpy()[3]), 1)

    def test_a_trial_that_moves_the_tip_beyond_the_step_resolution_ends_the_trials_before_its_forces(self):
        resolution = 0.0002525
        tip = wp.vec3(0.0, 0.0, 0.01)
        start = wp.array(
            [[0.03, -0.04, 0.0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]], dtype=wp.transform, device=self.device
        )
        valid = wp.zeros(1, dtype=int, device=self.device)
        unresolved = wp.zeros(1, dtype=int, device=self.device)

        def moved(pose):
            candidate = wp.array([pose, [9, 9, 9, 0, 0, 0, 1]], dtype=wp.transform, device=self.device)
            valid.fill_(1)
            unresolved.zero_()
            self.initialize()
            wp.launch(
                control.unresolved_tip_motion,
                1,
                [start, candidate, 0, tip, resolution, valid, unresolved],
                device=self.device,
            )
            wp.launch(control.reject_nonfinite, 1, [valid, unresolved, self.state, self.running], device=self.device)
            return int(valid.numpy()[0]), int(self.state.numpy()[4]), int(self.running.numpy()[0])

        # Only the needle body counts; the other body of the candidate is far away.
        self.assertEqual(moved([0.03, -0.04, 0.00025, 0, 0, 0, 1]), (1, 0, 1))
        self.assertEqual(moved([0.03, -0.04, 0.00026, 0, 0, 0, 1]), (0, 3, 0))
        # A rotation about the body origin moves the tip, 10 mm away, by 10 mm times the angle.
        for angle, expected in ((0.025, (1, 0, 1)), (0.026, (0, 3, 0))):
            quaternion = np.asarray(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle), dtype=float)
            self.assertEqual(moved([0.03, -0.04, 0.0, *quaternion]), expected)
        # A trial that is invalid for another reason keeps its own code.
        valid.zero_()
        unresolved.zero_()
        self.initialize()
        wp.launch(control.reject_nonfinite, 1, [valid, unresolved, self.state, self.running], device=self.device)
        self.assertEqual((int(self.state.numpy()[4]), int(self.running.numpy()[0])), (1, 0))

    def test_failure_and_geometry_fallback_cannot_commit(self):
        for reason in ("geometry", "nonfinite_force", "nonfinite_alm", "nonfinite_alm_flag"):
            with self.subTest(reason=reason):
                self.initialize()
                self.fallback.zero_()
                self.statistics.zero_()
                self.alm.zero_()
                if reason == "geometry":
                    self.fallback.fill_(1)
                elif reason == "nonfinite_force":
                    self.statistics.assign(np.array([0, 0, 0, 0, 0, 1], dtype=float))
                elif reason == "nonfinite_alm":
                    self.alm.assign(np.array([0, np.nan, 0, 0, 0], dtype=np.float32))
                else:
                    self.alm.assign(np.array([0, 0, 0, 0, 1], dtype=np.float32))
                self.assess()
                self.assertEqual(int(self.state.numpy()[3]), 0)
                self.assertEqual(int(self.running.numpy()[0]), 0)
                self.assertEqual(int(self.correct.numpy()[0]), 0)
                self.assertEqual(int(self.dual.numpy()[0]), 0)
                self.assertEqual(int(self.fresh.numpy()[0]), 1)
                self.assertEqual(int(self.stale.numpy()[0]), 0)

    def test_surface_event_requests_keep_searching_and_other_requests_stop(self):
        from newton_solver_suturebot._src.solvers.coupled.interface.needle_trial_forces import (
            FALLBACK_REASONS,
            SEARCHABLE_REASONS,
        )

        count = len(FALLBACK_REASONS)
        mask = np.array([1] + [int(reason in SEARCHABLE_REASONS) for reason in FALLBACK_REASONS[1:]], dtype=np.int32)
        self.fallback = wp.zeros(count, dtype=int, device=self.device)
        self.searchable = wp.array(mask, dtype=int, device=self.device)
        self.seen = wp.zeros(count, dtype=int, device=self.device)
        event = FALLBACK_REASONS.index("tip sweep not clear of the surface")
        other = FALLBACK_REASONS.index("degenerate visited tetrahedron")
        self.assertEqual((mask[event], mask[other]), (1, 0))
        # residual above and below the tolerance of 2e-6 + 1e-3 * 2e-3
        for residual, converged in ((1e-3, 0), (1e-7, 1)):
            for enabled, reasons, searching in (
                (1, (event,), True),
                (0, (event,), False),
                (1, (other,), False),
                (1, (event, other), False),
                (1, (), False),
            ):
                with self.subTest(residual=residual, enabled=enabled, reasons=reasons):
                    self.initialize()
                    wp.launch(
                        control._begin_trial,
                        1,
                        [self.state, self.values, self.levels, 5, self.initial],
                        device=self.device,
                    )
                    self.seen.zero_()
                    mask[0] = enabled
                    self.searchable.assign(mask)
                    flags = np.zeros(count, dtype=np.int32)
                    flags[[0, *reasons]] = 1
                    self.fallback.assign(flags)
                    self.statistics.assign(np.array([residual, 1e-3, 2e-3, 0, 0, 0], dtype=float))
                    self.alm.zero_()
                    self.assess(has_alm=0)
                    # Every request is recorded, whether or not the trials go on.
                    np.testing.assert_array_equal(self.seen.numpy(), flags)
                    self.assertEqual(int(self.state.numpy()[3]), int(searching and converged))
                    self.assertEqual(int(self.correct.numpy()[0]), int(searching and not converged))
                    self.assertEqual(int(self.running.numpy()[0]), int(searching and not converged))

    # Trial budget. Tolerances of assess(): absolute 2e-6, relative 1e-3; with a force of 1e-3
    # and a load of 2e-3 a trial passes the force test at a residual of 4e-6 or less.

    def budgeted_trial(self, residual, load, *, trial_budget, has_alm=0, limit=40, alm=(0.0, 0.0, 0.0, 0.0, 0.0)):
        """One trial as the captured loop issues it; returns what the loop does next.

        ``load`` is the trial load of this trial. The loop's correction, which would write the
        next trial load, is left to the caller: it runs only when ``correct`` is returned set.
        """
        wp.launch(control._begin_trial, 1, [self.state, self.values, self.levels, 5, self.initial], device=self.device)
        if load is not None:
            self.load.assign(np.asarray(load, dtype=np.float64))
        self.statistics.assign(np.array([residual, 1e-3, 2e-3, 1.0, -1.0, 0.0]))
        self.alm.assign(np.asarray(alm, dtype=np.float32))
        self.assess(has_alm=has_alm, limit=limit, trial_budget=trial_budget)
        wp.launch(
            control._exchange_best_load, self.load.size, [self.budget, self.load, self.best_load], device=self.device
        )
        return {
            "trials": int(self.state.numpy()[0]),
            "converged": int(self.state.numpy()[3]),
            "stopped": int(self.state.numpy()[4]),
            "dual_updates": int(self.state.numpy()[5]),
            "running": int(self.running.numpy()[0]),
            "correct": int(self.correct.numpy()[0]),
            "dual": int(self.dual.numpy()[0]),
            "fresh": int(self.fresh.numpy()[0]),
            "stale": int(self.stale.numpy()[0]),
            "kept": int(self.budget.numpy()[4]),
        }

    def packed_kept(self):
        wp.launch(
            control._pack_result,
            1,
            [self.state, self.values, self.fallback, self.seen, self.result, self.budget],
            device=self.device,
        )
        return int(self.result.numpy()[12])

    def test_below_the_budget_the_trials_go_on_as_without_one_and_the_best_load_is_saved(self):
        residuals = (3e-3, 1e-3, 2e-3, 5e-4)
        loads = [[float(trial), 10.0 * trial, -1.0 / trial] for trial in range(1, 5)]
        decisions = {}
        for trial_budget in (0, 5):
            self.initialize()
            self.best_load.zero_()
            decisions[trial_budget] = []
            for trial, (residual, load) in enumerate(zip(residuals, loads, strict=True), 1):
                outcome = self.budgeted_trial(residual, load, trial_budget=trial_budget)
                self.assertEqual(
                    {name: outcome[name] for name in ("trials", "converged", "running", "correct", "kept")},
                    {"trials": trial, "converged": 0, "running": 1, "correct": 1, "kept": 0},
                )
                decisions[trial_budget].append(
                    (outcome, self.state.numpy().tolist(), self.values.numpy().tolist(), self.load.numpy().tolist())
                )
                if trial_budget:
                    # The load of the trial with the smallest residual so far, bit for bit.
                    best = int(np.argmin(residuals[:trial]))
                    np.testing.assert_array_equal(self.best_load.numpy(), loads[best])
                    self.assertEqual(float(self.best_residual.numpy()[0]), residuals[best])
        # Every decision of the loop and the trial load are those of the loop without a budget.
        self.assertEqual(decisions[5], decisions[0])
        np.testing.assert_array_equal(self.result.numpy(), np.zeros(13))
        self.assertEqual(self.packed_kept(), 0)
        # Without a budget no load is saved.
        self.initialize()
        self.best_load.zero_()
        self.budgeted_trial(3e-3, loads[0], trial_budget=0)
        np.testing.assert_array_equal(self.best_load.numpy(), np.zeros(3))

    def test_at_the_budget_the_last_trial_is_kept_as_it_is_when_it_is_the_best(self):
        self.initialize()
        loads = ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0])
        self.assertEqual(self.budgeted_trial(3e-3, loads[0], trial_budget=3)["running"], 1)
        self.assertEqual(self.budgeted_trial(2e-3, loads[1], trial_budget=3)["running"], 1)
        outcome = self.budgeted_trial(1e-3, loads[2], trial_budget=3)
        self.assertEqual(
            outcome,
            {"trials": 3, "converged": 1, "stopped": 0, "dual_updates": 0, "running": 0, "correct": 0, "dual": 0,
             "fresh": 1, "stale": 0, "kept": 1},
        )  # fmt: skip
        # No correction and no restored load: the kept trial keeps the load it was computed with.
        np.testing.assert_array_equal(self.load.numpy(), loads[2])
        self.assertEqual(self.budget.numpy().tolist()[1:4], [0, 0, 0])
        self.assertEqual(float(self.values.numpy()[1]), 1e-3)
        self.assertEqual(self.packed_kept(), 1)
        # Of two trials with the same residual the later one is the best: no further trial.
        self.initialize()
        self.budgeted_trial(1e-3, loads[0], trial_budget=2)
        self.assertEqual(self.budgeted_trial(1e-3, loads[1], trial_budget=2)["kept"], 1)
        np.testing.assert_array_equal(self.load.numpy(), loads[1])

    def test_at_the_budget_an_earlier_best_load_is_applied_once_more_and_that_trial_is_kept(self):
        loads = ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0])
        # Residual of the trial at the restored load: worse than every earlier trial, equal to
        # the best one, and better than it but still above the tolerance of 4e-6.
        for last in (5e-3, 1e-3, 5e-5):
            with self.subTest(last=last):
                self.initialize()
                self.budgeted_trial(3e-3, loads[0], trial_budget=3)
                self.budgeted_trial(1e-3, loads[1], trial_budget=3)
                outcome = self.budgeted_trial(2e-3, loads[2], trial_budget=3)
                # Not kept, not corrected: one more trial with a full primal evaluation ...
                self.assertEqual(
                    outcome,
                    {"trials": 3, "converged": 0, "stopped": 0, "dual_updates": 0, "running": 1, "correct": 0,
                     "dual": 0, "fresh": 1, "stale": 0, "kept": 0},
                )  # fmt: skip
                # ... at the load of the best trial, bit for bit.
                np.testing.assert_array_equal(self.load.numpy(), loads[1])
                self.assertEqual(self.packed_kept(), 0)
                outcome = self.budgeted_trial(last, None, trial_budget=3)
                self.assertEqual(
                    outcome,
                    {"trials": 4, "converged": 1, "stopped": 0, "dual_updates": 0, "running": 0, "correct": 0,
                     "dual": 0, "fresh": 1, "stale": 0, "kept": 1},
                )  # fmt: skip
                np.testing.assert_array_equal(self.load.numpy(), loads[1])
                self.assertEqual(float(self.values.numpy()[1]), last)
                self.assertEqual(self.packed_kept(), 1)
        # A trial at the restored load that meets the tolerances converged: it is not a kept one.
        self.initialize()
        self.budgeted_trial(3e-3, loads[0], trial_budget=3)
        self.budgeted_trial(1e-3, loads[1], trial_budget=3)
        self.budgeted_trial(2e-3, loads[2], trial_budget=3)
        outcome = self.budgeted_trial(1e-7, None, trial_budget=3)
        self.assertEqual((outcome["converged"], outcome["running"], outcome["kept"]), (1, 0, 0))
        # The trial limit stays the limit: with no trial left the loop ends as it does today.
        self.initialize()
        self.budgeted_trial(3e-3, loads[0], trial_budget=3, limit=3)
        self.budgeted_trial(1e-3, loads[1], trial_budget=3, limit=3)
        outcome = self.budgeted_trial(2e-3, loads[2], trial_budget=3, limit=3)
        self.assertEqual((outcome["converged"], outcome["running"], outcome["correct"], outcome["kept"]), (0, 0, 0, 0))
        np.testing.assert_array_equal(self.load.numpy(), loads[2])

    def test_a_failed_loop_is_never_kept(self):
        from newton_solver_suturebot._src.solvers.coupled.interface.needle_trial_forces import FALLBACK_REASONS

        loads = ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0])
        other = FALLBACK_REASONS.index("degenerate visited tetrahedron")
        count = len(FALLBACK_REASONS)
        self.fallback = wp.zeros(count, dtype=int, device=self.device)
        self.searchable = wp.zeros(count, dtype=int, device=self.device)
        self.seen = wp.zeros(count, dtype=int, device=self.device)

        def fail(reason):
            """The failing trial; returns the stop code the loop reports."""
            if reason in ("invalid", "tip moved"):
                # Nonfinite state, backend overflow and a failed tissue-solver certificate clear
                # the validity flag; the forces of such a trial are not evaluated at all.
                wp.launch(
                    control._begin_trial, 1, [self.state, self.values, self.levels, 5, self.initial], device=self.device
                )
                valid = wp.zeros(1, dtype=int, device=self.device)
                unresolved = wp.array([int(reason == "tip moved")], dtype=int, device=self.device)
                wp.launch(
                    control.reject_nonfinite, 1, [valid, unresolved, self.state, self.running], device=self.device
                )
                return 3 if reason == "tip moved" else 1
            if reason == "nonfinite force":
                wp.launch(
                    control._begin_trial, 1, [self.state, self.values, self.levels, 5, self.initial], device=self.device
                )
                self.statistics.assign(np.array([2e-3, 1e-3, 2e-3, 1.0, -1.0, 1.0]))
                self.assess(has_alm=0, trial_budget=3)
                wp.launch(
                    control._exchange_best_load,
                    self.load.size,
                    [self.budget, self.load, self.best_load],
                    device=self.device,
                )
                return 1
            if reason == "divergence":
                # The applied load, 2e-3 in the first trial, has grown by more than 1 / relative
                # tolerance = 1000; the residual is above the tolerance of this trial (2.5e-3).
                wp.launch(
                    control._begin_trial, 1, [self.state, self.values, self.levels, 5, self.initial], device=self.device
                )
                self.statistics.assign(np.array([1.0, 1e-3, 2.5, 1.0, -1.0, 0.0]))
                self.assess(has_alm=0, trial_budget=3)
                wp.launch(
                    control._exchange_best_load,
                    self.load.size,
                    [self.budget, self.load, self.best_load],
                    device=self.device,
                )
                return 2
            # A request that hands the loop to the host oracle.
            flags = np.zeros(count, dtype=np.int32)
            flags[[0, other]] = 1
            self.fallback.assign(flags)
            self.budgeted_trial(2e-3, None, trial_budget=3)
            return 0

        for reason in ("invalid", "tip moved", "nonfinite force", "divergence", "host oracle"):
            # The failure in the trial that uses up the budget, and in the trial at the restored load.
            for restored in (False, True):
                with self.subTest(reason=reason, restored=restored):
                    self.initialize()
                    self.fallback.zero_()
                    self.budgeted_trial(3e-3, loads[0], trial_budget=3)
                    self.budgeted_trial(1e-3, loads[1], trial_budget=3)
                    if restored:
                        self.assertEqual(self.budgeted_trial(2e-3, loads[2], trial_budget=3)["running"], 1)
                    load = self.load.numpy().copy()
                    stopped = fail(reason)
                    self.assertEqual(int(self.state.numpy()[3]), 0)
                    self.assertEqual(int(self.state.numpy()[4]), stopped)
                    self.assertEqual(int(self.running.numpy()[0]), 0)
                    self.assertEqual(int(self.budget.numpy()[4]), 0)
                    self.assertEqual(self.packed_kept(), 0)
                    # No best load is restored for a failed loop.
                    np.testing.assert_array_equal(self.load.numpy(), load)
                    if reason not in ("invalid", "tip moved"):
                        self.assertEqual((int(self.correct.numpy()[0]), int(self.dual.numpy()[0])), (0, 0))

    def test_a_search_past_a_surface_event_that_uses_up_the_budget_ends_with_the_request_of_its_last_trial(self):
        from newton_solver_suturebot._src.solvers.coupled.interface.needle_trial_forces import (
            FALLBACK_REASONS,
            SEARCHABLE_REASONS,
        )

        count = len(FALLBACK_REASONS)
        mask = np.array([1] + [int(reason in SEARCHABLE_REASONS) for reason in FALLBACK_REASONS[1:]], dtype=np.int32)
        self.fallback = wp.zeros(count, dtype=int, device=self.device)
        self.searchable = wp.array(mask, dtype=int, device=self.device)
        self.seen = wp.zeros(count, dtype=int, device=self.device)
        event = FALLBACK_REASONS.index("tip sweep not clear of the surface")
        request = np.zeros(count, dtype=np.int32)
        request[[0, event]] = 1
        loads = ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0])
        # The request comes in the third trial, which uses up the budget: it is the best trial
        # (kept as it is), or the second is and its load is applied once more.
        for residuals, trials in (((3e-3, 2e-3, 1e-3), 3), ((3e-3, 1e-3, 2e-3), 4)):
            # The trial that ends the search carries the request, or is clear of it.
            for requested in (1, 0):
                with self.subTest(trials=trials, requested=requested):
                    self.initialize()
                    self.seen.zero_()
                    self.fallback.zero_()
                    self.budgeted_trial(residuals[0], loads[0], trial_budget=3)
                    self.budgeted_trial(residuals[1], loads[1], trial_budget=3)
                    self.fallback.assign(request)
                    outcome = self.budgeted_trial(residuals[2], loads[2], trial_budget=3)
                    if trials == 4:
                        # The search goes on for the one trial at the best load, with no correction.
                        self.assertEqual((outcome["running"], outcome["correct"], outcome["kept"]), (1, 0, 0))
                        np.testing.assert_array_equal(self.load.numpy(), loads[1])
                        if not requested:
                            self.fallback.zero_()
                        outcome = self.budgeted_trial(5e-3, None, trial_budget=3)
                    elif not requested:
                        continue
                    # The loop ends as a converged search does: converged and kept flags set, no
                    # stop code, the request of the last trial and the number of the first trial
                    # with a request. With a request the coupler hands that load to the host
                    # oracle and accepts nothing; without one the trial is kept.
                    self.assertEqual((outcome["converged"], outcome["running"], outcome["kept"]), (1, 0, 1))
                    self.assertEqual((outcome["correct"], outcome["dual"], outcome["trials"]), (0, 0, trials))
                    self.assertEqual(self.packed_kept(), 1)
                    np.testing.assert_array_equal(self.result.numpy()[[3, 4, 10, 11]], [1, 0, requested, 3])
                    np.testing.assert_array_equal(self.load.numpy(), loads[2] if trials == 3 else loads[1])

    def test_the_best_trial_restarts_at_a_multiplier_update(self):
        waiting = (0.0, 0.0, 1e-3, 1e-3, 0.0)  # multipliers not converged
        loads = [[float(trial), 0.5 * trial, -2.0 * trial] for trial in range(1, 6)]
        self.initialize()
        self.budgeted_trial(1e-3, loads[0], trial_budget=5, has_alm=1)
        np.testing.assert_array_equal(self.best_load.numpy(), loads[0])
        # The second trial passes the force test and waits for a multiplier update: below the
        # budget the update is made, and the best trial restarts with the changed force map.
        outcome = self.budgeted_trial(1e-7, loads[1], trial_budget=5, has_alm=1, alm=waiting)
        self.assertEqual(
            outcome,
            {"trials": 2, "converged": 0, "stopped": 0, "dual_updates": 1, "running": 1, "correct": 0, "dual": 1,
             "fresh": 0, "stale": 1, "kept": 0},
        )  # fmt: skip
        self.assertEqual(int(self.budget.numpy()[0]), 0)
        np.testing.assert_array_equal(self.best_load.numpy(), loads[0])
        # The repetition at the unchanged load is the first trial of the new epoch, and its
        # best one although the old epoch has smaller residuals (1e-3 and 1e-7).
        self.budgeted_trial(2e-3, None, trial_budget=5, has_alm=1)
        np.testing.assert_array_equal(self.best_load.numpy(), loads[1])
        self.assertEqual(float(self.best_residual.numpy()[0]), 2e-3)
        self.budgeted_trial(4e-3, loads[3], trial_budget=5, has_alm=1)
        outcome = self.budgeted_trial(3e-3, loads[4], trial_budget=5, has_alm=1)
        self.assertEqual((outcome["converged"], outcome["running"], outcome["correct"], outcome["kept"]), (0, 1, 0, 0))
        # The load of the best trial of the current epoch comes back, not that of the old one.
        np.testing.assert_array_equal(self.load.numpy(), loads[1])
        outcome = self.budgeted_trial(2.5e-3, None, trial_budget=5, has_alm=1)
        self.assertEqual(
            (outcome["trials"], outcome["converged"], outcome["dual_updates"], outcome["kept"]), (6, 1, 1, 1)
        )

    def test_a_trial_that_only_waits_for_a_multiplier_update_is_kept_at_the_budget_without_the_update(self):
        waiting = (0.0, 0.0, 1e-3, 1e-3, 0.0)
        loads = ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0])
        # At the budget itself, and in the trial at the restored load.
        for restored in (False, True):
            with self.subTest(restored=restored):
                self.initialize()
                self.budgeted_trial(3e-3, loads[0], trial_budget=3, has_alm=1)
                self.budgeted_trial(1e-3, loads[1], trial_budget=3, has_alm=1)
                if restored:
                    self.budgeted_trial(2e-3, loads[2], trial_budget=3, has_alm=1)
                    load = None
                else:
                    load = loads[2]
                outcome = self.budgeted_trial(1e-7, load, trial_budget=3, has_alm=1, alm=waiting)
                self.assertEqual(
                    outcome,
                    {"trials": 3 + int(restored), "converged": 1, "stopped": 0, "dual_updates": 0, "running": 0,
                     "correct": 0, "dual": 0, "fresh": 1, "stale": 0, "kept": 1},
                )  # fmt: skip
                np.testing.assert_array_equal(self.load.numpy(), loads[1] if restored else loads[2])
        # Converged multipliers at the budget are a converged trial, not a kept one.
        self.initialize()
        self.budgeted_trial(3e-3, loads[0], trial_budget=2, has_alm=1)
        outcome = self.budgeted_trial(1e-7, loads[1], trial_budget=2, has_alm=1)
        self.assertEqual((outcome["converged"], outcome["running"], outcome["kept"]), (1, 0, 0))

    def test_random_trial_sequences_end_with_the_trial_the_rule_names(self):
        # Whole loops of the kernels against the rule written out in Python, for random
        # residuals above the tolerance and random trials that wait for a multiplier update.
        waiting = (0.0, 0.0, 1e-3, 1e-3, 0.0)
        rng = np.random.default_rng(31012)
        kept_as_last, kept_after_restore = 0, 0

        def vector(identity):
            # The trial load after the correction that follows trial `identity`; 0: the starting load.
            return [float(identity), 0.5 * identity, -3.0 * identity]

        for _ in range(120):
            trial_budget = int(rng.integers(2, 9))
            events = [
                None if rng.random() < 0.2 else float(rng.choice([1e-3, 2e-3, 3e-3, 4e-3]) * rng.uniform(0.5, 1.5))
                for _ in range(trial_budget + 1)
            ]
            # The rule: best of the multiplier epoch, the later one of equal trials.
            best, load, loads, expected = None, 0, [], None
            for number, residual in enumerate(events, 1):
                loads.append(load)
                if residual is None:
                    if number >= trial_budget:
                        expected = (number, load)
                        break
                    best = None
                    continue
                is_best = best is None or residual <= best[0]
                if number >= trial_budget:
                    expected = (number, load) if is_best else (number + 1, best[1])
                    break
                if is_best:
                    best = (residual, load)
                load = number
            self.initialize()
            self.load.assign(np.asarray(vector(0), dtype=np.float64))
            trials = 0
            while True:
                residual = events[trials] if trials < len(events) else 9e-3
                trials += 1
                if residual is None:
                    outcome = self.budgeted_trial(1e-7, None, trial_budget=trial_budget, has_alm=1, alm=waiting)
                else:
                    outcome = self.budgeted_trial(residual, None, trial_budget=trial_budget, has_alm=1)
                if outcome["correct"]:
                    # The correction of the loop: a new trial load.
                    self.load.assign(np.asarray(vector(trials), dtype=np.float64))
                if not outcome["running"]:
                    break
            self.assertEqual(
                (outcome["trials"], outcome["converged"], outcome["stopped"], outcome["kept"]), (trials, 1, 0, 1)
            )
            self.assertEqual((trials, self.load.numpy().tolist()), (expected[0], vector(expected[1])), events)
            self.assertEqual(self.packed_kept(), 1)
            kept_as_last += int(trials == trial_budget)
            kept_after_restore += int(trials == trial_budget + 1)
        # Both ways of keeping a trial occur among the cases.
        self.assertGreater(min(kept_as_last, kept_after_restore), 10)

    def test_alm_threshold_preserves_double_host_comparison(self):
        self.initialize()
        threshold = 2e-5
        for value in (
            np.nextafter(np.float32(threshold), np.float32(0)),
            np.nextafter(np.float32(threshold), np.float32(1)),
        ):
            self.initialize()
            self.alm.assign(np.array([value, 0, 0, 0, 0], dtype=np.float32))
            self.assess()
            self.assertEqual(int(self.state.numpy()[3]), int(float(value) <= threshold))


if __name__ == "__main__":
    unittest.main()
