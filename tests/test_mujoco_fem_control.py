# SPDX-License-Identifier: Apache-2.0
"""Joint mechanics acceptance, fixed merit, and bounded device line search."""

import os
import unittest

import numpy as np
import warp as wp

from newton_solver_suturebot._src.solvers.mujoco_fem import control, kernels


class TestMuJoCoFEMControl(unittest.TestCase):
    # Binary-exact thresholds let the tests check equality and the adjacent
    # representable value without introducing decimal conversion ambiguity.
    absolute = 2.0**-20
    relative = 2.0**-10
    normal_tolerance = 2.0**-19
    force_scale = 2.0**-8
    fem_tolerance = 2.0**-10
    dt = 2.0**-6

    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.state = wp.zeros(6, dtype=int, device=self.device)
        self.merit = wp.zeros(6, dtype=wp.float64, device=self.device)
        self.weights = wp.ones(3, dtype=wp.float64, device=self.device)
        self.stats = wp.zeros(6, dtype=wp.float64, device=self.device)
        self.alm = wp.zeros(5, dtype=float, device=self.device)
        self.fem_stats = wp.zeros(3, dtype=wp.float64, device=self.device)
        self.fresh_fem_stats = wp.zeros(3, dtype=wp.float64, device=self.device)
        self.fallback = wp.zeros(1, dtype=int, device=self.device)
        self.running = wp.zeros(1, dtype=int, device=self.device)
        self.searching = wp.zeros(1, dtype=int, device=self.device)
        self.continuing = wp.zeros(1, dtype=int, device=self.device)
        self.step = wp.zeros(1, dtype=float, device=self.device)
        self.output = wp.zeros(11, dtype=wp.float64, device=self.device)
        self.initialize()

    def launch(self, kernel, *args):
        wp.launch(kernel, 1, list(args), device=self.device)

    def initialize(self):
        self.launch(control.initialize, self.state, self.merit, self.running, self.weights)
        self.fallback.zero_()
        self.fem_stats.zero_()
        self.fresh_fem_stats.zero_()

    def measure(self, stats, alm=None):
        values = np.asarray(stats, dtype=np.float64).copy()
        if values[3] == 0 and values[4] == 0:
            # Synthetic single-block controller fixtures put their raw norm
            # in the rigid block; multi-block cases pass explicit block norms.
            values[3] = values[2]
        self.stats.assign(values)
        self.alm.assign(np.asarray(np.zeros(5) if alm is None else alm, dtype=np.float32))
        self.launch(
            control.measure,
            self.stats,
            self.alm,
            self.fem_stats,
            self.fallback,
            self.absolute,
            self.relative,
            self.normal_tolerance,
            self.fem_tolerance,
            self.dt,
            self.weights,
            self.state,
            self.merit,
        )

    def decide(self, limit=159):
        self.launch(control.decide, self.state, self.merit, self.fallback, limit, self.running)

    def begin_search(self):
        self.launch(
            control.begin_search,
            self.state,
            self.merit,
            self.step,
            self.searching,
            self.stats,
            self.alm,
            self.fem_stats,
            self.absolute,
            self.relative,
            self.fem_tolerance,
            self.dt,
            self.weights,
        )

    def assess_search(self, limit=159):
        self.launch(control.assess_search, self.state, self.merit, self.step, self.searching, limit)

    def certify(self):
        self.launch(
            control.certification,
            self.stats,
            self.alm,
            self.fem_stats,
            self.fresh_fem_stats,
            self.fallback,
            self.absolute,
            self.relative,
            self.normal_tolerance,
            self.fem_tolerance,
            self.state,
            self.merit,
        )

    def continue_after_certificate(self, limit):
        self.launch(
            control.continue_after_certificate,
            self.state,
            self.merit,
            self.fallback,
            limit,
            self.continuing,
        )

    def test_each_physical_threshold_is_required_independently_of_raw_merit(self):
        threshold = self.absolute + self.relative * self.force_scale
        stats = np.array([threshold, self.force_scale, 1e6, 0, 0, 0], dtype=np.float64)
        alm = np.array([self.normal_tolerance, self.normal_tolerance, threshold, self.force_scale, 0], dtype=np.float32)
        self.measure(stats, alm)
        self.decide()
        self.assertEqual(int(self.state.numpy()[3]), 1)
        self.assertEqual(float(self.merit.numpy()[0]), 1e6)
        for gate in ("momentum", "normal_change", "normal_violation", "tangent_change"):
            with self.subTest(gate=gate):
                self.initialize()
                proposal_stats, proposal_alm = stats.copy(), alm.copy()
                proposal_stats[2] = 0.0  # A small objective is not physical acceptance.
                if gate == "momentum":
                    proposal_stats[0] = np.nextafter(threshold, np.inf)
                else:
                    index = {"normal_change": 0, "normal_violation": 1, "tangent_change": 2}[gate]
                    proposal_alm[index] = np.nextafter(proposal_alm[index], np.float32(np.inf))
                self.measure(proposal_stats, proposal_alm)
                self.decide()
                self.assertEqual(int(self.state.numpy()[3]), 0)
                self.assertEqual(int(self.running.numpy()[0]), 1)

    def test_line_search_keeps_base_weights_despite_an_improving_force_denominator(self):
        self.measure([1.0, 0.0, 1.0, 0, 0, 0])
        self.begin_search()
        self.measure([1.1, 100.0, 1.21, 0, 0, 0])
        # Normalized momentum fell by orders of magnitude, but raw residual
        # increased. Accepting this step would not be a descent line search.
        self.assertLess(1.1 / (self.absolute + 100 * self.relative), 1.0 / self.absolute)
        self.assess_search()
        self.assertEqual(float(self.step.numpy()[0]), 0.5)
        self.assertEqual(int(self.searching.numpy()[0]), 1)
        self.assertEqual(float(self.merit.numpy()[1]), 1.0 / self.absolute**2)

        self.initialize()
        self.measure([1.0, 100.0, 1.0, 0, 0, 0])
        self.begin_search()
        self.measure([0.9, 0.0, 0.81, 0, 0, 0])
        self.assertGreater(0.9 / self.absolute, 1.0 / (self.absolute + 100 * self.relative))
        self.assess_search()
        self.assertEqual(int(self.searching.numpy()[0]), 0)
        self.assertEqual(int(self.state.numpy()[4]), 0)
        self.assertEqual(float(self.step.numpy()[0]), 1.0)

    def test_fem_descent_can_outweigh_rigid_plateau_without_relaxing_acceptance(self):
        # Reproduce the scale imbalance: FEM is still outside its relative
        # accuracy, while rigid rows are well inside their force threshold.
        # Raw merit grows, but a weighted Newton proposal makes real progress.
        sigma_f = self.absolute / 16
        rhs_squared = (sigma_f * self.dt / self.fem_tolerance) ** 2
        tissue_squared = (2 * sigma_f) ** 2
        rigid_squared = (self.absolute / 2) ** 2
        self.fem_stats.assign(np.array([tissue_squared * self.dt**2, rhs_squared, 0], np.float64))
        self.measure([2 * self.absolute, 0, tissue_squared + rigid_squared, rigid_squared, 0, 0])
        self.begin_search()
        original_weights = self.weights.numpy().copy()
        base_merit = float(self.merit.numpy()[1])
        self.assertAlmostEqual(base_merit, 4.25)
        next_tissue_squared = (1.5 * sigma_f) ** 2
        next_rigid_squared = (0.75 * self.absolute) ** 2
        self.fem_stats.assign(np.array([next_tissue_squared * self.dt**2, rhs_squared, 0], np.float64))
        self.measure(
            [
                2 * self.absolute,
                0,
                next_tissue_squared + next_rigid_squared,
                next_rigid_squared,
                0,
                0,
            ]
        )
        self.assertGreater(next_tissue_squared + next_rigid_squared, tissue_squared + rigid_squared)
        self.assertAlmostEqual(float(self.merit.numpy()[0]), 2.8125)
        self.assess_search()
        self.decide()
        self.assertEqual(float(self.step.numpy()[0]), 1)
        self.assertEqual(int(self.searching.numpy()[0]), 0)
        self.assertEqual(int(self.state.numpy()[3]), 0)  # Physical force tolerance still fails.
        self.assertEqual(int(self.running.numpy()[0]), 1)
        np.testing.assert_array_equal(self.weights.numpy(), original_weights)

    def test_rhs_and_friction_scale_growth_cannot_fake_weighted_descent(self):
        for block in ("fem", "friction"):
            with self.subTest(block=block):
                self.initialize()
                sigma = self.absolute
                rhs_squared = (sigma * self.dt / self.fem_tolerance) ** 2
                self.fem_stats.assign(np.array([4 * sigma**2 * self.dt**2, rhs_squared, 0], np.float64))
                self.measure([2 * sigma, 0, 8 * sigma**2, 0, 4 * sigma**2, 0], [0, 0, 2 * sigma, 0, 0])
                self.begin_search()
                frozen = self.weights.numpy().copy()
                if block == "fem":
                    self.fem_stats.assign(np.array([9 * sigma**2 * self.dt**2, 100 * rhs_squared, 0], np.float64))
                    self.measure([3 * sigma, 0, 13 * sigma**2, 0, 4 * sigma**2, 0], [0, 0, 2 * sigma, 0, 0])
                else:
                    self.measure([2 * sigma, 0, 13 * sigma**2, 0, 9 * sigma**2, 0], [0, 0, 3 * sigma, 100, 0])
                self.assertAlmostEqual(float(self.merit.numpy()[0]), 13)
                self.assertAlmostEqual(float(self.merit.numpy()[1]), 8)
                self.assess_search()
                self.assertEqual(float(self.step.numpy()[0]), 0.5)
                self.assertEqual(int(self.searching.numpy()[0]), 1)
                np.testing.assert_array_equal(self.weights.numpy(), frozen)

    def test_new_search_recomputes_both_weights_and_base_merit(self):
        self.measure([1, 0, 1, 1, 0, 0])
        self.begin_search()
        first = self.weights.numpy().copy()
        self.measure([0.5, 100, 0.25, 0.25, 0, 0])
        self.assess_search()
        self.assertEqual(int(self.searching.numpy()[0]), 0)
        self.begin_search()
        second = self.weights.numpy()
        self.assertNotEqual(first[1], second[1])
        self.assertAlmostEqual(float(self.merit.numpy()[1]), 0.25 / (self.absolute + 100 * self.relative) ** 2)
        self.assertTrue(np.isfinite(second).all())  # The zero-RHS merit scale is finite.

    def test_friction_l2_search_surrogate_does_not_replace_aggregate_l1_gate(self):
        # Sixteen defects, each below the aggregate force tolerance, still
        # violate the physical L1 gate even as their squared norm decreases.
        unit = self.absolute
        self.measure([0, 0, 16 * unit**2, 0, 16 * unit**2, 0], [0, 0, 16 * unit, 0, 0])
        self.begin_search()
        self.measure([0, 0, 4 * unit**2, 0, 4 * unit**2, 0], [0, 0, 8 * unit, 0, 0])
        self.assess_search()
        self.decide()
        self.assertEqual(int(self.searching.numpy()[0]), 0)
        self.assertEqual(int(self.running.numpy()[0]), 1)
        self.assertEqual(int(self.state.numpy()[3]), 0)
        self.certify()
        self.assertEqual(int(self.state.numpy()[3]), 0)

    def test_proposal_may_request_correction_but_configured_fem_tolerance_controls_acceptance(self):
        # A physically converged Newton proposal can request correction even
        # with poor FEM relative accuracy. Only its corrected applied-load
        # certificate may accept the physical step.
        impulse_squared = 2.0**-48
        rhs_squared = 2.0**-24
        for tolerance, accepted in ((2.0**-11, True), (2.0**-12, True), (2.0**-13, False)):
            with self.subTest(tolerance=tolerance):
                self.initialize()
                self.fem_tolerance = tolerance
                self.fem_stats.assign(np.array([impulse_squared, rhs_squared, 0], np.float64))
                self.measure([self.absolute / 4, 0, 1e-20, 0, 0, 0])
                self.decide()
                self.assertEqual(int(self.state.numpy()[3]), 1)
                self.assertEqual(int(self.running.numpy()[0]), 0)
                self.certify()
                self.assertEqual(int(self.state.numpy()[3]), int(accepted))

    def test_only_applied_fem_relative_accuracy_is_required_but_fresh_metrics_must_be_finite(self):
        self.measure([0, 0, 0, 0, 0, 0])
        self.decide()
        self.assertEqual(int(self.state.numpy()[3]), 1)
        self.fresh_fem_stats.assign(np.array([4 * self.fem_tolerance**2, 1, 0], np.float64))
        self.certify()
        self.assertEqual(int(self.state.numpy()[3]), 1)
        self.fem_stats.assign(self.fresh_fem_stats.numpy())
        self.certify()
        self.continue_after_certificate(159)
        self.assertEqual(int(self.state.numpy()[3]), 0)
        self.assertEqual(int(self.continuing.numpy()[0]), 1)
        self.fem_stats.zero_()
        self.fresh_fem_stats.assign(np.array([np.nan, 1, 0], np.float64))
        self.certify()
        self.assertEqual(int(self.state.numpy()[3]), 0)
        self.assertEqual(int(self.merit.numpy()[5]), 0)

    def test_applied_fem_zero_rhs_and_nonfinite_metrics_have_no_hidden_accuracy_floor(self):
        for values, accepted, valid in (
            ([0, 0, 0], True, True),
            ([1e-60, 0, 0], False, True),
            ([np.nan, 1, 0], False, False),
            ([0, np.inf, 0], False, False),
            ([0, 1, 1], False, False),
        ):
            with self.subTest(values=values):
                self.initialize()
                self.fem_stats.assign(np.array(values, np.float64))
                self.measure([0, 0, 0, 0, 0, 0])
                self.decide()
                self.assertEqual(int(self.state.numpy()[3]), int(valid))
                self.assertEqual(bool(self.merit.numpy()[5]), valid)
                self.certify()
                self.assertEqual(int(self.state.numpy()[3]), int(accepted))

    def test_correction_counter_counts_runtime_calls_and_resets_explicitly(self):
        counter = wp.zeros(1, dtype=int, device=self.device)
        for _ in range(3):
            self.launch(control.record_correction, counter)
        self.assertEqual(int(counter.numpy()[0]), 3)
        counter.zero_()
        self.launch(control.record_correction, counter)
        self.assertEqual(int(counter.numpy()[0]), 1)

    def test_physically_converged_proposal_needs_no_further_merit_decrease(self):
        self.measure([1.0, 0.0, 0.01, 0, 0, 0])
        self.begin_search()
        self.measure([self.absolute, 0.0, 1.0, 0, 0, 0])
        self.assess_search()
        self.decide()
        self.assertEqual(int(self.searching.numpy()[0]), 0)
        self.assertEqual(int(self.state.numpy()[3]), 1)

    def test_last_nonlinear_evaluation_reserves_one_certification_evaluation(self):
        budget = 4
        limit = budget - 1  # The physical trial caller passes this reserved limit.
        self.measure([1.0, 0.0, 1.0, 0, 0, 0])
        self.decide(limit)
        self.begin_search()
        self.measure([1.1, 0.0, 1.21, 0, 0, 0])
        self.assess_search(limit)
        self.assertEqual(int(self.searching.numpy()[0]), 1)
        self.measure([0.0, 0.0, 0.0, 0, 0, 0])
        self.assess_search(limit)
        self.decide(limit)
        self.assertEqual(int(self.state.numpy()[0]), limit)
        self.assertEqual(int(self.state.numpy()[3]), 1)
        self.assertEqual(int(self.running.numpy()[0]), 0)
        self.certify()
        self.assertEqual(int(self.state.numpy()[0]), budget)
        self.assertEqual(int(self.state.numpy()[3]), 1)

        self.initialize()
        self.measure([1.0, 0.0, 1.0, 0, 0, 0])
        self.begin_search()
        for _ in range(limit - 1):
            self.measure([1.1, 0.0, 1.21, 0, 0, 0])
            self.assess_search(limit)
        self.decide(limit)
        self.assertEqual(int(self.state.numpy()[0]), limit)
        self.assertEqual(int(self.state.numpy()[4]), 1)
        self.assertEqual(int(self.state.numpy()[3]), 0)
        self.assertEqual(int(self.running.numpy()[0]), 0)
        self.assertEqual(int(self.searching.numpy()[0]), 0)

    def test_failure_flag_prevents_success_even_with_a_converged_measurement(self):
        self.state.assign(np.array([2, 1, 1, 0, 1, 0], dtype=np.int32))
        self.measure([0, 0, 0, 0, 0, 0])
        self.decide()
        self.assertEqual(int(self.state.numpy()[5]), 1)
        self.assertEqual(int(self.state.numpy()[3]), 0)
        self.assertEqual(int(self.running.numpy()[0]), 0)
        self.begin_search()
        self.assertEqual(int(self.searching.numpy()[0]), 0)

    def test_invalid_proposal_backtracks_without_reusing_stale_convergence(self):
        self.measure([0, 0, 0, 0, 0, 0])
        self.begin_search()
        valid = wp.zeros(1, dtype=int, device=self.device)
        self.launch(control.invalid_trial, valid, self.state, self.merit)
        self.assertEqual(int(self.state.numpy()[0]), 2)
        self.assertEqual(int(self.state.numpy()[5]), 1)  # Deliberately stale.
        self.assess_search()
        self.decide()
        self.assertEqual(int(self.state.numpy()[3]), 0)
        self.assertEqual(int(self.searching.numpy()[0]), 1)
        self.assertEqual(float(self.step.numpy()[0]), 0.5)
        self.measure([0, 0, 0, 0, 0, 0])
        self.assess_search()
        self.decide()
        self.assertEqual(int(self.state.numpy()[3]), 1)

        self.initialize()
        self.measure([1, 0, 1, 0, 0, 0])
        self.begin_search()
        for _ in range(24):
            self.launch(control.invalid_trial, valid, self.state, self.merit)
            self.assess_search()
        self.decide()
        self.assertEqual(int(self.state.numpy()[0]), 25)
        self.assertEqual(int(self.state.numpy()[2]), 24)
        self.assertEqual(int(self.state.numpy()[4]), 1)
        self.assertEqual(int(self.state.numpy()[3]), 0)
        self.assertEqual(int(self.searching.numpy()[0]), 0)

    def test_finite_geometry_checks_reject_nonfinite_vectors_and_multipliers(self):
        valid = wp.ones(1, dtype=int, device=self.device)
        vectors = wp.array([[1, 2, 3], [4, 5, 6]], dtype=wp.vec3, device=self.device)
        scalars = wp.array([1, 2], dtype=float, device=self.device)
        for value in (np.nan, np.inf, -np.inf):
            for kind in ("vector", "scalar"):
                with self.subTest(value=value, kind=kind):
                    valid.fill_(1)
                    vectors.assign(np.array([[1, 2, 3], [4, value if kind == "vector" else 5, 6]], np.float32))
                    scalars.assign(np.array([1, value if kind == "scalar" else 2], np.float32))
                    wp.launch(control.finite_vectors, 2, [vectors, valid], device=self.device)
                    wp.launch(control.finite_scalars, 2, [scalars, valid], device=self.device)
                    self.assertEqual(int(valid.numpy()[0]), 0)

    def test_certification_rechecks_original_force_normal_dual_and_fallback_gates(self):
        threshold = self.absolute + self.relative * self.force_scale
        # Certification statistics[2] is the other force magnitude, not the
        # joint solve's raw merit. Either applied or recomputed force can scale
        # the original max-norm residual tolerance.
        for scale_index in (1, 2):
            for failure in (None, "force", "normal", "dual", "force_flag", "alm_flag", "nan", "fallback"):
                with self.subTest(scale_index=scale_index, failure=failure):
                    self.initialize()
                    self.measure([0, 0, 0, 0, 0, 0])
                    self.decide()
                    self.assertEqual(int(self.state.numpy()[3]), 1)
                    stats = np.array([threshold, 0, 0, 0, 0, 0], np.float64)
                    stats[scale_index] = self.force_scale
                    alm = np.array([self.normal_tolerance, 0, threshold, self.force_scale, 0], np.float32)
                    if failure == "force":
                        stats[0] = np.nextafter(threshold, np.inf)
                    elif failure == "normal":
                        alm[1] = np.nextafter(np.float32(self.normal_tolerance), np.float32(np.inf))
                    elif failure == "dual":
                        alm[2] = np.nextafter(np.float32(threshold), np.float32(np.inf))
                    elif failure == "force_flag":
                        stats[5] = 1
                    elif failure == "alm_flag":
                        alm[4] = 1
                    elif failure == "nan":
                        alm[3] = np.nan
                    elif failure == "fallback":
                        self.fallback.fill_(1)
                    self.stats.assign(stats)
                    self.alm.assign(alm)
                    self.certify()
                    self.assertEqual(int(self.state.numpy()[0]), 2)
                    self.assertEqual(int(self.state.numpy()[3]), int(failure is None))
                    self.launch(control.result, self.state, self.merit, self.fallback, self.output)
                    self.assertEqual(float(self.output.numpy()[7]), stats[0])
                    self.assertEqual(float(self.output.numpy()[9]), float(alm[2]))

    def test_failed_finite_certificate_resumes_without_resetting_shared_budget(self):
        budget = 4
        self.measure([0, 0, 0, 0, 0, 0])
        self.decide(budget - 1)
        # The canonical endpoint changes the friction proposal after the
        # simultaneous momentum/projection root passed its predictor gates.
        self.alm.assign(np.array([0, 0, 2 * self.absolute, 0, 0], np.float32))
        self.certify()
        self.continue_after_certificate(budget - 1)
        self.assertEqual(int(self.state.numpy()[0]), 2)
        self.assertEqual(int(self.state.numpy()[3]), 0)
        self.assertEqual(int(self.continuing.numpy()[0]), 1)
        self.measure([0, 0, 0, 0, 0, 0])
        self.decide(budget - 1)
        self.certify()
        self.continue_after_certificate(budget - 1)
        self.assertEqual(int(self.state.numpy()[0]), budget)
        self.assertEqual(int(self.state.numpy()[3]), 1)
        self.assertEqual(int(self.continuing.numpy()[0]), 0)

    def test_certificate_continuation_requires_two_remaining_evaluations_and_valid_geometry(self):
        for reason in (
            "budget",
            "fallback",
            "nonfinite_geometry",
            "nonfinite_force",
            "nonfinite_alm",
            "linear_failure",
        ):
            with self.subTest(reason=reason):
                self.initialize()
                self.measure([0, 0, 0, 0, 0, 0])
                self.decide()
                self.alm.assign(np.array([0, 0, 2 * self.absolute, 0, 0], np.float32))
                if reason == "fallback":
                    self.fallback.fill_(1)
                elif reason == "nonfinite_force":
                    self.stats.assign(np.array([0, 0, 0, 0, 0, 1], np.float64))
                elif reason == "nonfinite_alm":
                    self.alm.assign(np.array([0, 0, np.nan, 0, 0], np.float32))
                if reason == "nonfinite_geometry":
                    valid = wp.zeros(1, dtype=int, device=self.device)
                    self.launch(control.invalid_trial, valid, self.state, self.merit)
                else:
                    self.certify()
                if reason == "linear_failure":
                    state = self.state.numpy()
                    state[4] = 1
                    self.state.assign(state)
                self.continue_after_certificate(2 if reason == "budget" else 159)
                self.assertEqual(int(self.state.numpy()[0]), 2)
                self.assertEqual(int(self.continuing.numpy()[0]), 0)

    def test_recovery_changes_only_rigid_unknowns_and_uses_both_orientation_frames(self):
        def quaternion(axis, angle):
            axis = np.asarray(axis, dtype=np.float64)
            axis /= np.linalg.norm(axis)
            return np.r_[axis * np.sin(angle / 2), np.cos(angle / 2)].astype(np.float32)

        def rotation(q):
            x, y, z, w = np.asarray(q, dtype=np.float64)
            return np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ]
            )

        source_q = np.r_[np.zeros(3), quaternion([1, 2, -1], 0.7)].astype(np.float32)
        endpoint_q = np.r_[np.ones(3), quaternion([-2, 1, 3], -0.4)].astype(np.float32)
        source_v = np.array([0.2, -0.3, 0.4, 1.2, -0.7, 0.3], np.float32)
        endpoint_v = np.array([-0.1, 0.4, 0.2, -0.4, 0.9, 1.3], np.float32)
        before = np.arange(18, dtype=np.float32).reshape(6, 3)
        unknown = wp.array(before, dtype=wp.vec3, device=self.device)
        arrays = [wp.array(v, dtype=float, device=self.device) for v in (source_q, source_v, endpoint_q, endpoint_v)]
        length = 0.01
        self.launch(kernels.recover_rigid_increment, *arrays, 2, length, unknown)
        actual = unknown.numpy()
        r0, r1 = rotation(source_q[3:]), rotation(endpoint_q[3:])
        expected_angular = length * r0 @ (r1.T @ endpoint_v[3:] - r0.T @ source_v[3:])
        np.testing.assert_array_equal(actual[:2], before[:2])
        np.testing.assert_array_equal(actual[4:], before[4:])
        np.testing.assert_array_equal(actual[2], endpoint_v[:3] - source_v[:3])
        np.testing.assert_allclose(actual[3], expected_angular, atol=3e-9, rtol=3e-6)

    def test_captured_acceptance_replays_live_statistics_and_certifies_only_success(self):
        if not self.device.is_cuda:
            self.skipTest("CUDA conditional graph test")
        self.stats.zero_()
        self.alm.zero_()
        # Compile all kernels before capture; replay must still read new values.
        self.measure([0, 0, 0, 0, 0, 0])
        self.decide()
        self.certify()
        self.launch(control.result, self.state, self.merit, self.fallback, self.output)
        with wp.ScopedCapture(device=self.device) as capture:
            self.launch(control.initialize, self.state, self.merit, self.running, self.weights)
            self.launch(
                control.measure,
                self.stats,
                self.alm,
                self.fem_stats,
                self.fallback,
                self.absolute,
                self.relative,
                self.normal_tolerance,
                self.fem_tolerance,
                self.dt,
                self.weights,
                self.state,
                self.merit,
            )
            self.decide(limit=1)
            wp.capture_if(self.state[3:4], self.certify)
            self.launch(control.result, self.state, self.merit, self.fallback, self.output)
        for fallback in (0, 1, 0):
            self.fallback.fill_(fallback)
            wp.capture_launch(capture.graph)
            result = self.output.numpy()
            self.assertEqual(result[0], 2 if fallback == 0 else 1)
            self.assertEqual(result[3], int(fallback == 0))
            self.assertEqual(result[10], fallback)


if __name__ == "__main__":
    unittest.main()
