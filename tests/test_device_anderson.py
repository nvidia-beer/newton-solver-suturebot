# SPDX-License-Identifier: Apache-2.0
"""Sequence-level transport parity, including recovery, resets and dense fallback."""

import os
import unittest

import numpy as np
import warp as wp

from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_anderson import SurfaceAnderson
from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_anderson_device import DeviceSurfaceAnderson

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


class DeviceAndersonTests(unittest.TestCase):
    def sequence(self, *, capacity=64, dense=False, resets=()):
        rng = np.random.default_rng(514)
        n = 1537
        load = np.zeros(n)
        support = np.arange(n) if dense else np.array([0, 1, 5, 96, 300, 700, n - 1])
        load[support] = rng.normal(size=len(support)) * 0.001
        device_load = wp.array(load, dtype=wp.float64, device=DEVICE)
        device_residual = wp.empty_like(device_load)
        device_direction = wp.empty_like(device_load)
        actual, expected = DeviceSurfaceAnderson(device_load, capacity=capacity), SurfaceAnderson()
        any_proposal = False
        any_recovery = False
        old_rejections = 0
        for trial in range(35):
            if trial in resets:
                actual.reset()
                expected.reset()
            # A nontrivial sparse force map with occasional contact changes.
            residual = np.zeros(n)
            residual[support] = 0.002 * np.sin(0.8 * np.arange(len(support))) - 0.8 * load[support]
            if trial in (5, 15, 27):
                residual[support] += 0.005 * rng.normal(size=len(support))
            direction = residual.copy()
            direction[support] *= np.linspace(0.4, 1.1, len(support))
            # Varying support and a subnormal must survive exact-zero packing.
            if not dense:
                residual[1200 + trial] = 1e-310
            device_load.assign(load)
            device_residual.assign(residual)
            device_direction.assign(direction)
            proposal = expected.propose(load, residual, 0.35, direction=direction)
            applied = actual.propose_device(device_load, device_residual, 0.35, direction=device_direction)
            self.assertEqual(applied, proposal is not None, trial)
            self.assertEqual(actual.proposals, expected.proposals, trial)
            self.assertEqual(actual.rejections, expected.rejections, trial)
            self.assertEqual(actual.filtered_directions, expected.filtered_directions, trial)
            self.assertEqual(actual.regularization, expected.regularization, trial)
            if expected.last_ratio is not None:
                self.assertEqual(actual.last_ratio, expected.last_ratio, trial)
            self.assertEqual(len(actual._slots), len(expected.history), trial)
            self.assertFalse(actual.history)
            self.assertIsNone(actual.pending)
            if proposal is None:
                np.testing.assert_array_equal(device_load.numpy(), load)
                load += 0.35 * direction
            else:
                any_proposal = True
                np.testing.assert_array_equal(device_load.numpy(), proposal)
                load = proposal.copy()
            any_recovery |= actual.rejections > old_rejections
            old_rejections = actual.rejections
        self.assertTrue(any_proposal)
        self.assertTrue(any_recovery)
        return actual

    def test_sparse_histories_match_original_svd_and_every_safeguard(self):
        actual = self.sequence()
        self.assertEqual(actual.overflows, 0)
        self.assertLess(actual.transport_bytes, 35 * 3 * 1537 * 8)

    def test_resets_clear_history_and_pending_fallback(self):
        actual = self.sequence(resets=(8, 9, 23))
        actual.reset()
        self.assertFalse(actual._slots)
        self.assertIsNone(actual._pending_metadata)

    def test_overflow_grows_and_reuses_transport_without_changing_proposal(self):
        actual = self.sequence(capacity=3)
        self.assertGreater(actual.overflows, 0)
        self.assertEqual(actual.transport_calls, 35 + actual.overflows)
        self.assertLess(actual.overflows, 5)
        self.assertGreaterEqual(actual._capacity, actual.max_active_rows)
        grown = actual._capacity
        actual.reset()
        self.assertEqual(actual._capacity, grown)

    def test_dense_data_uses_the_original_full_vector_math(self):
        actual = self.sequence(capacity=16, dense=True)
        self.assertEqual(actual.overflows, 1)
        self.assertEqual(actual.transport_calls, 36)
        self.assertEqual(actual._capacity, 1537)

    def test_default_capacity_grows_exact_support_and_survives_reset(self):
        n = 1025
        x = wp.zeros(n, dtype=wp.float64, device=DEVICE)
        r = wp.empty_like(x)
        actual, expected = DeviceSurfaceAnderson(x), SurfaceAnderson()
        self.assertEqual(actual._capacity, 128)
        for count, capacity in ((78, 128), (350, 512), (1025, 1025)):
            load = np.zeros(n)
            residual = np.zeros(n)
            residual[:count] = np.sin(np.arange(count) * 0.4) + 0.1
            residual[count - 1] = 1e-310
            x.assign(load)
            r.assign(residual)
            proposal = expected.propose(load, residual, 0.5)
            applied = actual.propose_device(x, r, 0.5)
            self.assertEqual(applied, proposal is not None)
            np.testing.assert_array_equal(x.numpy(), load if proposal is None else proposal)
            self.assertEqual(actual._capacity, capacity)
            self.assertEqual(actual.max_active_rows, count)
            actual.reset()
            expected.reset()
        calls, overflows = actual.transport_calls, actual.overflows
        self.assertEqual(overflows, 2)
        for _ in range(3):
            x.zero_()
            actual.propose_device(x, r, 0.5)
            actual.reset()
        self.assertEqual(actual.overflows, overflows)
        self.assertEqual(actual.transport_calls, calls + 3)

    def test_nonfinite_residual_resets_history_and_leaves_load_untouched(self):
        x = wp.array([1.0, 2.0, 3.0], dtype=wp.float64, device=DEVICE)
        r = wp.array([np.nan, 0.0, 1.0], dtype=wp.float64, device=DEVICE)
        actual = DeviceSurfaceAnderson(x)
        self.assertFalse(actual.propose_device(x, r, 0.5))
        np.testing.assert_array_equal(x.numpy(), [1, 2, 3])
        self.assertFalse(actual._slots)
        self.assertIsNone(actual._pending_metadata)

    def test_gradient_contract_fails_closed_before_transport(self):
        x = wp.array([1.0, 2.0, 3.0], dtype=wp.float64, device=DEVICE)
        r = wp.array([0.1, 0.2, 0.3], dtype=wp.float64, device=DEVICE, requires_grad=True)
        actual = DeviceSurfaceAnderson(x)
        with self.assertRaises(ValueError):
            actual.propose_device(x, r, 0.5)
        self.assertEqual(actual.transport_calls, 0)
        with wp.Tape(), self.assertRaises(RuntimeError):
            actual.propose_device(x, x, 0.5)


class ActualDeviceTransportTests(unittest.TestCase):
    def setUp(self):
        if not wp.get_device(DEVICE).is_cuda:
            self.skipTest("Actual coupler selects device transport only on CUDA")

    def test_real_native_contact_preserves_acceptance_and_endpoint(self):
        from test_surface_anderson_coupling import _contact, _snapshot

        records = []
        for enabled in (False, True):
            with _contact(absolute_tolerance=2e-9, relative_tolerance=1e-6) as (ex, coupler, native, _external):
                coupler._device_surface_anderson_enabled = enabled
                coupler.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
                self.assertAlmostEqual(native.mj_data.time, 3.25 + ex.sim_dt, places=12)
                self.assertEqual(ex.interaction.status, "contact")
                self.assertFalse(ex.interaction.events)
                self.assertEqual(ex.interaction.track_count, 0)
                self.assertEqual(coupler.max_iterations, 160)
                self.assertEqual(ex.solver.linear_solver_tolerance, 1e-6)
                if enabled:
                    transport = coupler._device_surface_anderson_workspace
                    self.assertGreater(transport.transport_calls, 0)
                    self.assertGreater(transport.proposals, 0)
                else:
                    self.assertIsNone(getattr(coupler, "_device_surface_anderson_workspace", None))
                records.append(_snapshot(ex.state_1))
        for field in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd"):
            np.testing.assert_allclose(records[1][field], records[0][field], rtol=2e-5, atol=3e-7, err_msg=field)

    def test_failed_attempt_rolls_back_and_reuses_workspace_with_fresh_history(self):
        from test_surface_anderson_coupling import SurfaceAndersonCouplingTests, _contact

        from newton_solver_suturebot._src.solvers.coupled.coupling import NeedleCouplingNotConverged

        with _contact(iterations=6) as (ex, coupler, native, _external):
            SurfaceAndersonCouplingTests()._assert_rollback(
                ex, coupler, native, NeedleCouplingNotConverged, "did not converge"
            )
            transport = coupler._device_surface_anderson_workspace
            calls = transport.transport_calls
            self.assertGreater(calls, 0)
            self.assertFalse(transport._slots)
            self.assertIsNone(transport._pending_metadata)
            coupler.max_iterations = 160
            coupler.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
            self.assertIs(coupler._device_surface_anderson_workspace, transport)
            self.assertGreater(transport.transport_calls, calls)
            self.assertEqual(ex.interaction.status, "contact")
            self.assertAlmostEqual(native.mj_data.time, 3.25 + ex.sim_dt, places=12)


if __name__ == "__main__":
    unittest.main()
