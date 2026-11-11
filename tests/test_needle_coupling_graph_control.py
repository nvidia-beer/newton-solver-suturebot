# SPDX-License-Identifier: Apache-2.0
"""Control-flow parity against the currently shipped host coupling rules."""

import os
import unittest

import numpy as np
import warp as wp

from newton_solver_suturebot._src import coupling_graph as control


class DeviceControlTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.state = wp.zeros(6, dtype=int, device=self.device)
        self.values = wp.zeros(4, dtype=wp.float64, device=self.device)
        self.statistics = wp.zeros(6, dtype=wp.float64, device=self.device)
        self.alm = wp.zeros(5, dtype=float, device=self.device)
        self.fallback = wp.zeros(1, dtype=int, device=self.device)
        self.running = wp.zeros(1, dtype=int, device=self.device)
        self.correct = wp.zeros(1, dtype=int, device=self.device)
        self.dual = wp.zeros(1, dtype=int, device=self.device)
        self.fresh = wp.zeros(1, dtype=int, device=self.device)
        self.stale = wp.zeros(1, dtype=int, device=self.device)
        self.levels = wp.zeros(5, dtype=int, device=self.device)
        self.initial = 0.3

    def initialize(self):
        wp.launch(
            control._initialize,
            1,
            [self.state, self.values, self.running, self.fresh, self.stale, self.initial],
            device=self.device,
        )

    def assess(self, *, limit=40, has_alm=1):
        wp.launch(
            control._assess_trial,
            1,
            [
                self.state,
                self.values,
                self.statistics,
                self.alm,
                self.fallback,
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
