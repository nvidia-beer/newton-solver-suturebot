# SPDX-License-Identifier: Apache-2.0
"""Multisecant proposals preserve balanced forces and cannot accept a state."""

import unittest

import numpy as np

from newton_solver_suturebot._src.surface_anderson import SurfaceAnderson


class SurfaceAndersonTests(unittest.TestCase):
    def test_stiff_two_mode_contact_reaches_original_fixed_point(self):
        # Two force-balanced modes, with very different implicit stiffnesses.
        basis = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float) / np.sqrt(2)
        stiffness = basis @ np.diag([5.0, 80.0]) @ basis.T
        forcing = basis @ np.array([0.002, -0.003])
        exact = np.linalg.solve(np.eye(4) + stiffness, forcing)
        acceleration = SurfaceAnderson()
        load = np.zeros(4)
        tolerance = 1e-11
        for iteration in range(160):
            residual = forcing - stiffness @ load - load
            if np.max(np.abs(residual)) <= tolerance:
                self.assertLess(iteration, 20)
                break
            proposal = acceleration.propose(load, residual, 0.012)
            load = load + 0.012 * residual if proposal is None else proposal
            self.assertAlmostEqual(load.sum(), 0, delta=1e-15)
        else:
            self.fail("The unchanged exact residual did not converge")
        self.assertGreater(acceleration.proposals, 0)
        np.testing.assert_allclose(load, exact, atol=1e-11, rtol=0)

    def test_worse_accelerated_trial_recovers_saved_aitken_proposal(self):
        acceleration = SurfaceAnderson()
        self.assertIsNone(acceleration.propose(np.array([0.0, 0.0]), np.array([1.0, -1.0]), 0.2))
        load, residual = np.array([0.2, -0.2]), np.array([0.6, -0.6])
        proposal = acceleration.propose(load, residual, 0.2)
        self.assertIsNotNone(proposal)
        restored = acceleration.propose(proposal, np.array([10.0, -10.0]), 0.7)
        np.testing.assert_array_equal(restored, load + 0.2 * residual)
        self.assertEqual(acceleration.rejections, 1)
        self.assertEqual(acceleration.history, [])

    def test_nonlinear_compressive_contact_uses_exact_residual_to_converge(self):
        acceleration = SurfaceAnderson()
        load = np.zeros(2)
        for iteration in range(160):
            force = max(0.0, 0.1 - 8.0 * load[0] - 50.0 * load[0] ** 2)
            residual = np.array([force, -force]) - load
            if np.max(np.abs(residual)) <= 1e-10:
                self.assertLess(iteration, 25)
                break
            proposal = acceleration.propose(load, residual, 0.05)
            load = load + 0.05 * residual if proposal is None else proposal
            self.assertAlmostEqual(load.sum(), 0, delta=1e-15)
        else:
            self.fail("The unchanged nonlinear contact residual did not converge")
        exact = (-9.0 + np.sqrt(81.0 + 20.0)) / 100.0
        self.assertAlmostEqual(load[0], exact, delta=1e-10)

    def test_ill_conditioned_linear_contact_filters_dependent_secants(self):
        # Rotated balanced modes span a stiffness ratio of 10,000. Near
        # convergence, redundant historical force differences lose rank.
        for seed in range(3):
            with self.subTest(seed=seed):
                rotation = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))[0]
                basis = np.vstack((rotation, -rotation)) / np.sqrt(2)
                stiffness = basis @ np.diag([1.0, 100.0, 10000.0]) @ basis.T
                forcing = basis @ (0.001 * np.arange(1, 4))
                exact = np.linalg.solve(np.eye(6) + stiffness, forcing)
                acceleration = SurfaceAnderson()
                load = np.zeros(6)
                relaxation = 0.9 / 10001
                for _iteration in range(160):
                    residual = forcing - stiffness @ load - load
                    if np.max(np.abs(residual)) <= 1e-11:
                        break
                    proposal = acceleration.propose(load, residual, relaxation)
                    load = load + relaxation * residual if proposal is None else proposal
                    self.assertAlmostEqual(load.sum(), 0, delta=1e-14)
                self.assertLess(_iteration, 30)
                self.assertGreater(acceleration.filtered_directions, 0)
                np.testing.assert_allclose(load, exact, atol=1e-11, rtol=0)

    def test_insufficient_actual_reduction_increases_regularization_and_recovers(self):
        acceleration = SurfaceAnderson()
        acceleration.propose(np.array([0.0, 0.0]), np.array([1.0, -1.0]), 0.2)
        load, residual = np.array([0.2, -0.2]), np.array([0.6, -0.6])
        proposal = acceleration.propose(load, residual, 0.2)
        regularization = acceleration.regularization
        # This increase was below the original 2x residual rejection threshold.
        restored = acceleration.propose(proposal, np.array([0.9, -0.9]), 0.7)
        np.testing.assert_array_equal(restored, load + 0.2 * residual)
        self.assertGreater(acceleration.regularization, regularization)
        self.assertLess(acceleration.last_ratio, 0.1)
        self.assertEqual(acceleration.history, [])

    def test_zero_rows_have_same_proposals_and_recovery_as_compact_interface(self):
        compact, padded = SurfaceAnderson(), SurfaceAnderson()
        indices = np.array([3, 99, 20000])
        load = np.zeros(3)
        for _ in range(15):
            residual = np.array([0.1, -0.1, 1e-150]) - np.array([6.0, 9.0, 0.0]) * load
            full_load, full_residual = np.zeros(21621), np.zeros(21621)
            full_load[indices], full_residual[indices] = load, residual
            small = compact.propose(load, residual, 0.05)
            large = padded.propose(full_load, full_residual, 0.05)
            self.assertEqual(small is None, large is None)
            if small is not None:
                np.testing.assert_allclose(large[indices], small, atol=1e-14, rtol=1e-13)
                mask = np.ones(21621, dtype=bool)
                mask[indices] = False
                np.testing.assert_array_equal(large[mask], 0)
            load = load + 0.05 * residual if small is None else small
        self.assertGreater(compact.proposals, 0)
        self.assertEqual(compact.proposals, padded.proposals)
        self.assertGreater(load[2], 0)

    def test_reset_discards_secants_and_pending_recovery(self):
        acceleration = SurfaceAnderson()
        acceleration.propose(np.array([0.0]), np.array([1.0]), 0.2)
        acceleration.propose(np.array([0.2]), np.array([0.6]), 0.2)
        acceleration.reset()
        self.assertEqual(acceleration.history, [])
        self.assertIsNone(acceleration.pending)
        self.assertIsNone(acceleration.propose(np.array([0.1]), np.array([0.5]), 0.2))

    def test_repeated_residual_and_nonfinite_input_do_not_propose(self):
        acceleration = SurfaceAnderson()
        self.assertIsNone(acceleration.propose(np.array([0.0]), np.array([1.0]), 0.2))
        self.assertIsNone(acceleration.propose(np.array([0.2]), np.array([1.0]), 0.2))
        self.assertIsNone(acceleration.propose(np.array([np.nan]), np.array([1.0]), 0.2))
        self.assertEqual(acceleration.history, [])


if __name__ == "__main__":
    unittest.main()
