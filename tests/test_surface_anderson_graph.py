"""CPU-SVD oracle and complete history/overflow/capture contracts."""

import gc
import os
import unittest

import numpy as np
import warp as wp

from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_anderson import SurfaceAnderson
from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_anderson_graph import CapturedSurfaceAnderson

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


@wp.kernel
def sequence_input(
    load: wp.array[wp.float64],
    residual: wp.array[wp.float64],
    direction: wp.array[wp.float64],
    sequence: wp.array3d[wp.float64],
    state: wp.array[int],
    counter: wp.array[int],
):
    row = wp.tid()
    index = 3 - counter[0]
    load[row] = sequence[index, 0, row]
    residual[row] = sequence[index, 1, row]
    direction[row] = sequence[index, 2, row]
    if row == 0:
        state[0] = index


@wp.kernel
def sequence_output(
    load: wp.array[wp.float64],
    applied: wp.array[int],
    output: wp.array2d[wp.float64],
    flags: wp.array[int],
    counter: wp.array[int],
):
    row = wp.tid()
    index = 3 - counter[0]
    output[index, row] = load[row]
    if row == 0:
        flags[index] = applied[0]


@wp.kernel
def decrement(counter: wp.array[int]):
    counter[0] -= 1


class GraphAndersonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        wp.set_device(DEVICE)

    def create(self, size, capacity=256):
        self.load = wp.zeros(size, dtype=wp.float64)
        self.residual = wp.zeros(size, dtype=wp.float64)
        self.direction = wp.zeros(size, dtype=wp.float64)
        self.values = wp.array([0.3, 0.0, 0.0, 0.0], dtype=wp.float64)
        self.iteration = wp.zeros(6, dtype=int)
        self.aa = CapturedSurfaceAnderson(self.load, capacity=capacity)
        return SurfaceAnderson()

    def visit(self, load, residual, direction, index, epoch=0, tolerance=0):
        self.load.assign(np.asarray(load, dtype=np.float64))
        self.residual.assign(np.asarray(residual, dtype=np.float64))
        self.direction.assign(np.asarray(direction, dtype=np.float64))
        self.iteration.assign(np.array([index, epoch, tolerance, 0, 0, 0], dtype=np.int32))
        self.aa.propose_device(self.load, self.residual, self.values, self.direction, self.iteration)
        applied = int(self.aa.applied.numpy()[0])
        self.assertEqual(int(self.aa.not_applied.numpy()[0]), 1 - applied)
        return self.load.numpy().copy() if applied else None

    def assert_oracle(self, actual, expected):
        self.assertEqual(actual is None, expected is None)
        if expected is not None:
            np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-10)

    def test_affine_well_conditioned_map_against_thin_svd(self):
        oracle = self.create(12)
        rng = np.random.default_rng(15)
        q, _ = np.linalg.qr(rng.normal(size=(12, 5)))
        target = q @ np.array([0.2, -0.3, 0.1, 0.04, -0.12])
        matrix = q @ np.diag([0.4, 0.7, 1.2, 1.5, 1.8]) @ q.T + 0.2 * np.eye(12)
        current = np.zeros(12)
        for index in range(9):
            residual = target - matrix @ current
            direction = 0.4 * residual
            expected = oracle.propose(current, residual, 0.3, direction)
            actual = self.visit(current, residual, direction, index)
            self.assert_oracle(actual, expected)
            current = current + 0.3 * direction if expected is None else expected
        self.assertGreater(int(self.aa.counters.numpy()[1]), 0)

    def test_rank_deficient_and_balanced_affine_proposals(self):
        oracle = self.create(9)
        pattern = np.array([1.0, -2.0, 0.5, -1.0, 2.0, -0.5, 0.0, 0.0, 0.0])
        for index, coefficient in enumerate([0.0, 0.1, 0.2, 0.4, 0.6, 0.8]):
            load = coefficient * pattern
            residual = (1 - coefficient) * pattern
            direction = 0.4 * residual
            expected = oracle.propose(load, residual, 0.3, direction)
            actual = self.visit(load, residual, direction, index)
            self.assert_oracle(actual, expected)
            if actual is not None:
                np.testing.assert_array_equal(actual[:3] + actual[3:6], np.zeros(3))
        self.assertGreater(int(self.aa.counters.numpy()[7]), 0)

    def test_measured_rejection_restores_exact_saved_relaxed_load(self):
        oracle = self.create(4)
        for index, (value, residual) in enumerate([(0.0, 1.0), (0.2, 0.8), (1.0, 10.0)]):
            load = np.array([value, 0.0, 0.0, 0.0])
            r = np.array([residual, 0.0, 0.0, 0.0])
            expected = oracle.propose(load, r, 0.3, r)
            actual = self.visit(load, r, r, index)
            self.assert_oracle(actual, expected)
        np.testing.assert_allclose(actual, [0.44, 0.0, 0.0, 0.0], rtol=0, atol=1e-15)
        self.assertEqual(int(self.aa.state.numpy()[0]), 0)
        self.assertEqual(int(self.aa.counters.numpy()[2]), 1)

    def test_epoch_tolerance_and_iteration_restart_clear_pending(self):
        for change in ("epoch", "tolerance", "iteration"):
            self.create(2)
            self.visit([0.0, 0.0], [1.0, 0.0], [1.0, 0.0], 0)
            self.assertIsNotNone(self.visit([0.2, 0.0], [0.8, 0.0], [0.8, 0.0], 1))
            actual = self.visit(
                [4.0, 0.0],
                [10.0, 0.0],
                [10.0, 0.0],
                0 if change == "iteration" else 2,
                epoch=int(change == "epoch"),
                tolerance=int(change == "tolerance"),
            )
            self.assertIsNone(actual)
            np.testing.assert_array_equal(self.load.numpy(), [4.0, 0.0])
            self.assertEqual(int(self.aa.state.numpy()[0]), 1)

    def test_active_union_overflow_never_truncates_or_replaces_load(self):
        self.create(300)
        original = np.arange(300, dtype=np.float64) + 1
        result = self.visit(original, np.ones(300), np.ones(300), 0)
        self.assertIsNone(result)
        np.testing.assert_array_equal(self.load.numpy(), original)
        self.assertEqual(int(self.aa.state.numpy()[0]), 0)
        self.assertEqual(int(self.aa.counters.numpy()[3]), 1)
        self.assertEqual(int(self.aa.total.numpy()[0]), 300)

    def test_disappearing_rows_still_participate_in_saved_recovery(self):
        oracle = self.create(260)
        for index, (value, res) in enumerate([(0.0, 1.0), (0.2, 0.8), (1.0, 10.0)]):
            load, residual = np.zeros(260), np.zeros(260)
            load[259], residual[259] = value, res
            if index == 2:
                load[259], residual[259] = 0.0, 0.0
                residual[1] = 10.0
            expected = oracle.propose(load, residual, 0.3, residual)
            actual = self.visit(load, residual, residual, index)
            self.assert_oracle(actual, expected)
        self.assertAlmostEqual(actual[259], 0.44)
        self.assertEqual(actual[1], 0.0)

    def test_nonfinite_and_gradient_inputs_fail_closed(self):
        self.create(2)
        self.assertIsNone(self.visit([0.0, 0.0], [float("nan"), 0.0], [1.0, 0.0], 0))
        np.testing.assert_array_equal(self.load.numpy(), [0.0, 0.0])
        self.assertEqual(int(self.aa.state.numpy()[0]), 0)
        with self.assertRaises(ValueError):
            CapturedSurfaceAnderson(wp.zeros(2, dtype=wp.float64, requires_grad=True))
        with wp.Tape(), self.assertRaises(RuntimeError):
            self.aa.propose_device(
                self.load,
                self.residual,
                self.values,
                self.direction,
                self.iteration,
            )

    @unittest.skipUnless(DEVICE.startswith("cuda"), "CUDA nested conditional capture")
    def test_captured_conditional_rejection_and_repeated_reset(self):
        self.create(4)
        inputs = np.zeros((3, 3, 4))
        inputs[:, 0, 0] = [0.0, 0.2, 1.0]
        inputs[:, 1, 0] = [1.0, 0.8, 10.0]
        inputs[:, 2, 0] = [1.0, 0.8, 10.0]
        sequence = wp.array(inputs, dtype=wp.float64)
        output = wp.zeros((3, 4), dtype=wp.float64)
        flags = wp.zeros(3, dtype=int)
        counter = wp.zeros(1, dtype=int)

        def body():
            wp.launch(
                sequence_input,
                4,
                [
                    self.load,
                    self.residual,
                    self.direction,
                    sequence,
                    self.iteration,
                    counter,
                ],
            )
            self.aa.propose_device(self.load, self.residual, self.values, self.direction, self.iteration)
            wp.launch(sequence_output, 4, [self.load, self.aa.applied, output, flags, counter])
            wp.launch(decrement, 1, [counter])

        counter.fill_(3)
        body()  # eager compilation
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            with wp.ScopedCapture() as capture:
                self.aa.reset_device()
                counter.fill_(3)
                wp.capture_while(counter, body)
        finally:
            if gc_was_enabled:
                gc.enable()
        for _ in range(2):
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(flags.numpy(), [0, 1, 1])
            np.testing.assert_allclose(output.numpy()[:, 0], [0.0, 1.0, 0.44], atol=1e-8, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
