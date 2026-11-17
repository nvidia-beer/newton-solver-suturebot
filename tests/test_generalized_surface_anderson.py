# SPDX-License-Identifier: Apache-2.0
"""Raw force residual controls mixing while search directions may differ."""

import unittest

import numpy as np

from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_anderson import SurfaceAnderson


def seed(acceleration):
    x0 = np.zeros(4)
    r0 = np.array([1.0, -1.0, 0.0, 0.0])
    d0 = np.array([0.3, -0.3, 0.2, -0.2])
    x1 = np.array([0.2, -0.2, 0.0, 0.0])
    r1 = np.array([0.6, -0.6, 0.0, 0.0])
    d1 = np.array([0.1, -0.1, 0.4, -0.4])
    assert acceleration.propose(x0, r0, 0.25, direction=d0) is None
    proposal = acceleration.propose(x1, r1, 0.25, direction=d1)
    assert proposal is not None
    return x0, r0, d0, x1, r1, d1, proposal


class GeneralizedSurfaceAndersonTests(unittest.TestCase):
    def test_raw_residual_least_squares_mixes_preconditioned_images(self):
        acceleration = SurfaceAnderson()
        x0, r0, d0, x1, r1, d1, proposal = seed(acceleration)
        # Independent scalar ridge solution: no SVD implementation duplicated.
        dr = r0 - r1
        coefficient = -np.dot(dr, r1) / (np.dot(dr, dr) + 1e-8 * np.dot(r1, r1))
        expected = x1 + d1 + coefficient * ((x0 + d0) - (x1 + d1))
        np.testing.assert_allclose(proposal, expected, rtol=0, atol=2e-15)
        predicted = r1 + coefficient * dr
        self.assertAlmostEqual(acceleration.pending[1], np.linalg.norm(predicted), delta=1e-15)
        self.assertAlmostEqual(proposal.sum(), 0, delta=1e-15)
        # Direction-only rows must survive the exact-zero active-row trimming.
        self.assertGreater(abs(proposal[2]), 0.5)

    def test_raw_error_rejects_even_when_preconditioned_direction_is_tiny(self):
        acceleration = SurfaceAnderson()
        *_, x1, _r1, d1, proposal = seed(acceleration)
        recovery = acceleration.propose(proposal, np.array([2.0, -2.0, 0.0, 0.0]), 0.9, direction=np.full(4, 1e-30))
        np.testing.assert_array_equal(recovery, x1 + 0.25 * d1)
        self.assertEqual(acceleration.rejections, 1)
        self.assertEqual(acceleration.history, [])
        self.assertIsNone(acceleration.pending)

    def test_raw_reduction_accepts_despite_large_current_direction(self):
        acceleration = SurfaceAnderson()
        *_, proposal = seed(acceleration)
        acceleration.propose(
            proposal, np.array([0.01, -0.01, 0.0, 0.0]), 0.25, direction=np.array([10.0, -10.0, 5.0, -5.0])
        )
        self.assertEqual(acceleration.rejections, 0)
        self.assertGreater(acceleration.last_ratio, 0.75)
        self.assertEqual(len(acceleration.history), 3)

    def test_varying_preconditioner_preserves_raw_linear_fixed_point(self):
        rotation = np.linalg.qr(np.random.default_rng(21).normal(size=(3, 3)))[0]
        basis = np.vstack((rotation, -rotation)) / np.sqrt(2)
        operator = np.eye(6) + basis @ np.diag([3.0, 60.0, 300.0]) @ basis.T
        rhs = basis @ np.array([0.002, -0.003, 0.004])
        exact = np.linalg.solve(operator, rhs)
        acceleration, load = SurfaceAnderson(), np.zeros(6)
        for iteration in range(160):
            raw = rhs - operator @ load
            if np.max(np.abs(raw)) <= 1e-11:
                break
            # A bounded varying search metric, deliberately not equal to raw.
            scale = 0.8 + 0.15 * np.sin(iteration)
            direction = scale * np.linalg.solve(operator + 2 * np.eye(6), raw)
            proposal = acceleration.propose(load, raw, 0.5, direction=direction)
            load = load + 0.5 * direction if proposal is None else proposal
            self.assertAlmostEqual(load.sum(), 0, delta=1e-15)
        self.assertLess(iteration, 40)
        self.assertGreater(acceleration.proposals, 0)
        np.testing.assert_allclose(load, exact, rtol=0, atol=1e-11)

    def test_nonfinite_inputs_or_direction_shape_clear_pending_and_history(self):
        for invalid in ("load", "residual", "relaxation", "direction", "direction_shape"):
            with self.subTest(invalid=invalid):
                acceleration = SurfaceAnderson()
                *_, proposal = seed(acceleration)
                arguments = dict(load=proposal, residual=np.ones(4), relaxation=0.5, direction=np.ones(4))
                if invalid == "direction_shape":
                    arguments["direction"] = np.ones(3)
                elif invalid == "relaxation":
                    arguments[invalid] = np.inf
                else:
                    arguments[invalid] = np.full(4, np.nan)
                self.assertIsNone(acceleration.propose(**arguments))
                self.assertEqual(acceleration.history, [])
                self.assertIsNone(acceleration.pending)

    def test_invalid_load_shape_raises_after_discarding_stale_history(self):
        acceleration = SurfaceAnderson()
        seed(acceleration)
        with self.assertRaises(ValueError):
            acceleration.propose(np.zeros(3), np.ones(4), 0.5, direction=np.ones(4))
        self.assertEqual(acceleration.history, [])
        self.assertIsNone(acceleration.pending)

    def test_valid_new_dimension_starts_new_history_without_stale_fallback(self):
        acceleration = SurfaceAnderson()
        seed(acceleration)
        self.assertIsNone(acceleration.propose(np.zeros(2), np.ones(2), 0.5, direction=np.array([0.1, -0.1])))
        self.assertEqual(len(acceleration.history), 1)
        self.assertEqual(acceleration.history[0][0].shape, (2,))
        self.assertIsNone(acceleration.pending)


if __name__ == "__main__":
    unittest.main()
