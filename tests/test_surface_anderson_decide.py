"""Staged captured-Anderson decision kernels against the frozen single-thread kernel.

Every output of a decision must be bit-identical on CPU and on CUDA: state, history
order, counters, norms, regularisation and reference scalars, columns and coefficients,
and through ``propose_device`` the proposed load itself.
"""

import os
import unittest
from unittest import mock

import numpy as np
import warp as wp
from _surface_anderson_decide_reference import enqueue_reference, populate

from newton_solver_suturebot._src.solvers.coupled.acceleration import surface_anderson_decide
from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_anderson_graph import CapturedSurfaceAnderson

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
OUTPUTS = ("state", "order", "total", "invalid", "columns", "applied", "not_applied", "counters")
FLOATS = ("scalars", "norms", "maxima", "coefficients")


class DecisionEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        wp.set_device(DEVICE)

    def assert_identical(self, old, new, label):
        for name in OUTPUTS:
            np.testing.assert_array_equal(
                getattr(new, name).numpy(), getattr(old, name).numpy(), err_msg=f"{label} {name}"
            )
        for name in FLOATS:
            np.testing.assert_array_equal(
                getattr(new, name).numpy().view(np.uint64),
                getattr(old, name).numpy().view(np.uint64),
                err_msg=f"{label} {name}",
            )

    def decide(self, old, new, values, label, **state):
        populate(old, **state)
        populate(new, **state)
        enqueue_reference(old, values)
        surface_anderson_decide.enqueue_decision(new, values)
        self.assert_identical(old, new, label)
        count_rows = int(old.total.numpy()[0])
        if count_rows <= old.capacity and int(old.invalid.numpy()[0]) == 0:
            np.testing.assert_array_equal(
                new.ordered.numpy()[:count_rows], old.rows.numpy()[:count_rows], err_msg=f"{label} rows"
            )
        state, counters = old.state.numpy(), old.counters.numpy()
        if counters[3] or counters[4]:
            return "invalid"
        if state[3] == 2:
            return "rejected"
        if state[3] == 1:
            return "applied"
        return "deficient" if counters[7] else "declined"

    def test_randomised_histories_are_bitwise_identical(self):
        template = wp.zeros(1000, dtype=wp.float64)
        values = wp.array([0.3, 0.0, 0.0, 0.0], dtype=wp.float64)
        rng = np.random.default_rng(7)
        outcomes = {}
        columns = set()
        for depth in range(1, 7):
            old = CapturedSurfaceAnderson(template, depth=depth)
            new = CapturedSurfaceAnderson(template, depth=depth)
            for rows in (1, 20, 81, 256):
                for count in range(depth + 2):
                    for rank in (0, 1, 3, 6, 12):
                        state = {
                            "count": count,
                            "rows": rows,
                            "rank": rank,
                            "seed": int(rng.integers(1 << 30)),
                            "pending": int(rng.integers(5)),
                            "magnitude": float(rng.choice([1e-170, 1e-20, 1e-3, 1.0, 1e3, 1e160])),
                            "ridge": float(rng.choice([1e-12, 1e-8, 1e-4, 1e2])),
                            "stale": float(rng.choice([1.0, 1e6, np.nan, np.inf])),
                        }
                        outcome = self.decide(old, new, values, f"{depth=} {state}", **state)
                        outcomes[outcome] = outcomes.get(outcome, 0) + 1
                        if outcome == "applied":
                            columns.add(int(new.state.numpy()[9]))
        for outcome in ("applied", "rejected", "declined", "deficient", "invalid"):
            self.assertGreater(outcomes.get(outcome, 0), 10, outcomes)
        self.assertEqual(columns, set(range(1, 7)))

    def test_overflow_invalid_and_zero_histories_are_bitwise_identical(self):
        template = wp.zeros(1000, dtype=wp.float64)
        values = wp.array([0.3, 0.0, 0.0, 0.0], dtype=wp.float64)
        old = CapturedSurfaceAnderson(template)
        new = CapturedSurfaceAnderson(template)
        for count in (0, 3, 7):
            for pending in (0, 1):
                common = {"count": count, "pending": pending, "seed": 11 + count}
                self.decide(old, new, values, f"overflow {common}", rows=300, rank=6, **common)
                self.assertEqual(int(new.counters.numpy()[3]), 1)
                self.decide(old, new, values, f"invalid {common}", rows=81, rank=6, invalid=1, **common)
                self.assertEqual(int(new.counters.numpy()[4]), 1)
                self.decide(old, new, values, f"zero {common}", rows=81, rank=0, **common)
                self.assertEqual(int(new.applied.numpy()[0]), pending)
        small_old = CapturedSurfaceAnderson(template, capacity=20)
        small_new = CapturedSurfaceAnderson(template, capacity=20)
        for rows in (20, 21):
            self.decide(small_old, small_new, values, f"capacity {rows}", count=4, rows=rows, rank=6, seed=5)
            self.assertEqual(int(small_new.counters.numpy()[3]), rows - 20)

    def test_propose_sequences_are_bitwise_identical(self):
        size = 600
        totals = np.zeros(8, dtype=np.int64)
        for seed, rank in ((3, 4), (4, 12), (5, 40)):
            rng = np.random.default_rng(seed)
            arrays = []
            for _ in range(2):
                load = wp.zeros(size, dtype=wp.float64)
                arrays.append(
                    {
                        "load": load,
                        "residual": wp.zeros(size, dtype=wp.float64),
                        "direction": wp.zeros(size, dtype=wp.float64),
                        "values": wp.array([0.3, 0.0, 0.0, 0.0], dtype=wp.float64),
                        "iteration": wp.zeros(6, dtype=int),
                        "aa": CapturedSurfaceAnderson(load),
                    }
                )
            old, new = arrays[0]["aa"], arrays[1]["aa"]
            staged = surface_anderson_decide.enqueue_decision

            def enqueue(aa, values, old=old, staged=staged):
                (enqueue_reference if aa is old else staged)(aa, values)

            active = rng.choice(size, 90, replace=False)
            basis = np.linalg.qr(rng.normal(size=(90, rank)))[0]
            gains = rng.uniform(0.4, 1.8, rank)
            target = rng.normal(size=rank)
            current = np.zeros(size)
            applied, proposals = 0, 0
            for index in range(60):
                # A mildly nonlinear map keeps Anderson proposing for the whole sequence.
                coordinates = basis.T @ current[active]
                residual = np.zeros(size)
                residual[active] = basis @ (target - gains * coordinates - 0.3 * coordinates**3)
                if applied and proposals % 5 == 0:
                    # A residual that grew after a proposal exercises measured rejection.
                    residual *= 50.0
                direction = 0.4 * residual
                epoch = int(index >= 31)
                results = []
                for item in arrays:
                    item["load"].assign(current)
                    item["residual"].assign(residual)
                    item["direction"].assign(direction)
                    item["iteration"].assign(np.array([index, epoch, 0, 0, 0, 0], dtype=np.int32))
                    with mock.patch.object(surface_anderson_decide, "enqueue_decision", enqueue):
                        item["aa"].propose_device(
                            item["load"], item["residual"], item["values"], item["direction"], item["iteration"]
                        )
                    results.append(item["load"].numpy().copy())
                self.assert_identical(old, new, f"{seed=} {index=}")
                np.testing.assert_array_equal(results[1].view(np.uint64), results[0].view(np.uint64))
                applied = int(new.applied.numpy()[0])
                proposals += int(new.state.numpy()[3] == 1)
                current = results[1] if applied else current + 0.3 * direction
            totals += new.counters.numpy()
        # [calls, proposals, measured rejections, overflow, invalid, resets, maximum rows, rank deficit]
        self.assertGreater(int(totals[1]), 30)
        self.assertGreater(int(totals[2]), 3)
        self.assertGreater(int(totals[7]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
