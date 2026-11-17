"""Aitken mixing preserves the host float64 rule without a second CUDA readback."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot._src.solvers.coupled.coupling import (
    _device_aitken_supported,
    _direction_statistics,
    _relax_interface,
    _relax_preconditioned_interface,
)


class DeviceAitkenTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")

    def launch(self, direction, statistics, initial, has_previous, slot, state, load, previous):
        wp.launch(
            _relax_preconditioned_interface,
            direction.size,
            [direction, statistics, initial, int(has_previous), slot, state, load, previous],
            device=self.device,
        )
        return 1 - slot

    def test_combined_reduction_and_mixing_matches_host_reference(self):
        rng = np.random.default_rng(39)
        count = 21621
        values = rng.normal(size=count) * 0.001
        load = wp.array(values, dtype=wp.float64, device=self.device)
        reference = wp.clone(load)
        previous = wp.array(rng.normal(size=count), dtype=wp.float64, device=self.device)
        reference_previous = wp.clone(previous)
        statistics = wp.zeros(6, dtype=wp.float64, device=self.device)
        state = wp.array([np.nan, np.nan], dtype=wp.float64, device=self.device)
        relaxation, slot = 0.3, 0
        for index in range(8):
            direction = wp.array(rng.normal(size=count) * 0.1**index, dtype=wp.float64, device=self.device)
            statistics.zero_()
            # The reduction is unchanged; feed exactly its values to the host
            # oracle so CUDA atomic order is not mistaken for a formula change.
            wp.launch(_direction_statistics, count, [direction, previous, statistics], device=self.device)
            sums = statistics.numpy()
            if index:
                relaxation = float(np.clip(-relaxation * sums[4] / sums[3], 0.001, 2.0))
            slot = self.launch(direction, statistics, 0.3, index > 0, slot, state, load, previous)
            wp.launch(
                _relax_interface, count, [direction, relaxation, reference, reference_previous], device=self.device
            )
            self.assertEqual(state.numpy()[slot], relaxation)
            np.testing.assert_array_equal(load.numpy(), reference.numpy())
            np.testing.assert_array_equal(previous.numpy(), reference_previous.numpy())

    def test_mixing_resets_discard_nonfinite_and_previous_trial_scalars(self):
        direction = wp.array([1.0, -2.0, 0.0], dtype=wp.float64, device=self.device)
        statistics = wp.array([0, 0, 0, 1, -6, 0], dtype=wp.float64, device=self.device)
        state = wp.array([np.nan, np.nan], dtype=wp.float64, device=self.device)
        load, previous = (wp.zeros(3, dtype=wp.float64, device=self.device) for _ in range(2))
        slot = 0
        expected = np.zeros(3)
        # Start, ordinary mix, tighter FEM map, ordinary mix, ALM map, then
        # fresh host fallback with its read slot starting at zero again.
        for has_previous, restart_slot, wanted in (
            (False, None, 0.3),
            (True, None, 1.8),
            (False, None, 0.3),
            (True, None, 1.8),
            (False, None, 0.3),
            (False, 0, 0.3),
        ):
            if restart_slot is not None:
                slot = restart_slot
            slot = self.launch(direction, statistics, 0.3, has_previous, slot, state, load, previous)
            self.assertAlmostEqual(state.numpy()[slot], wanted)
            expected += wanted * np.array([1.0, -2.0, 0.0])
            np.testing.assert_allclose(load.numpy(), expected, rtol=2e-15, atol=0)

    def test_denominator_guard_clamps_and_nonfinite_values_match_numpy(self):
        direction = wp.array([1.0, -1.0], dtype=wp.float64, device=self.device)
        cases = (
            (0.3, 0, 1),
            (0.3, -1, 1),
            (0.3, 1e-30, -2),
            (0.3, np.nextafter(1e-30, np.inf), -2),
            (0.3, 1, -1),
            (0.3, 1, 1),
            (0.3, 1, -100),
            (0.3, np.nan, -1),
            (0.3, np.inf, 1),
            (0.3, 1, np.inf),
            (0.3, 1, -np.inf),
            (0.3, np.inf, np.inf),
            (np.nan, 1, -1),
            (np.inf, 0, 1),
            (np.inf, 1, 0),
        )
        for old, denominator, numerator in cases:
            with self.subTest(old=old, denominator=denominator, numerator=numerator):
                statistics = wp.array([0, 0, 0, denominator, numerator, 0], dtype=wp.float64, device=self.device)
                state = wp.array([old, 999], dtype=wp.float64, device=self.device)
                load, previous = (wp.zeros(2, dtype=wp.float64, device=self.device) for _ in range(2))
                expected = old
                with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                    if denominator > 1e-30:
                        expected = float(np.clip(-np.float64(old) * numerator / denominator, 0.001, 2.0))
                self.launch(direction, statistics, 0.3, True, 0, state, load, previous)
                np.testing.assert_equal(state.numpy()[1], expected)
                np.testing.assert_equal(load.numpy(), np.array([1, -1]) * expected)
                np.testing.assert_array_equal(previous.numpy(), [1, -1])

    def test_only_nondifferentiable_preconditioned_cuda_uses_device_mixing(self):
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
        solver = SimpleNamespace(differentiable=False)
        arrays = {
            name: SimpleNamespace(requires_grad=False)
            for name in (
                "particle_q",
                "particle_qd",
                "particle_f",
                "joint_q",
                "joint_qd",
                "body_q",
                "body_qd",
                "body_f",
            )
        }
        state = SimpleNamespace(**arrays)
        self.assertTrue(_device_aitken_supported(model, solver, state, True))
        self.assertFalse(_device_aitken_supported(model, solver, state, False))
        model.device.is_cuda = False
        self.assertFalse(_device_aitken_supported(model, solver, state, True))
        model.device.is_cuda = True
        for obj, name in (
            (model, "requires_grad"),
            (solver, "differentiable"),
            *((a, "requires_grad") for a in arrays.values()),
        ):
            with self.subTest(field=name, obj=obj):
                setattr(obj, name, True)
                self.assertFalse(_device_aitken_supported(model, solver, state, True))
                setattr(obj, name, False)
        with patch("newton_solver_suturebot._src.solvers.coupled.coupling.active_tape", return_value=object()):
            self.assertFalse(_device_aitken_supported(model, solver, state, True))
        self.assertTrue(_device_aitken_supported(model, solver, state, True))


if __name__ == "__main__":
    unittest.main()
