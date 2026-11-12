# SPDX-License-Identifier: Apache-2.0
import os
import unittest

import numpy as np
import warp as wp

from newton_solver_suturebot._src.phase_predictor import PhaseLoadPredictor


class PhasePredictorTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.device.__enter__()
        self.current = wp.zeros(6, dtype=wp.float64)
        self.trial = wp.zeros_like(self.current)
        self.engagement = wp.array([1, 0, 1], dtype=int)
        self.predictor = PhaseLoadPredictor(self.current)
        self.command = np.array([0, 0, 0, 0, 0, 0, 1, 0.0])
        self.vector = np.array([1, -1, 0.2, -0.2, 0.5, -0.5])

    def tearDown(self):
        self.device.__exit__(None, None, None)

    def begin_frame(self, frame, *, reset=0, step_dt=0.01):
        self.predictor.begin_frame(
            frame_dt=0.04,
            step_dt=step_dt,
            elapsed=frame * 0.04,
            command=self.command,
            reset_token=reset,
        )

    def accept(self, frame, phase, *, signature="same"):
        self.predictor.begin(self.current, self.trial, 0.01, signature, self.engagement)
        guess = self.trial.numpy().copy()
        accepted = wp.array((1 + 0.1 * frame + 0.4 * phase) * self.vector, dtype=wp.float64)
        self.predictor.commit(accepted)
        wp.copy(self.current, accepted)
        return guess

    def completed_frame(self, frame):
        self.begin_frame(frame)
        guesses = [self.accept(frame, phase) for phase in range(4)]
        self.predictor.end_frame((frame + 1) * 0.04, True)
        return guesses

    def test_same_phase_history_tracks_four_phase_oscillation(self):
        self.completed_frame(0)
        repeated = self.completed_frame(1)
        for phase, guess in enumerate(repeated):
            np.testing.assert_allclose(guess, (1 + 0.4 * phase) * self.vector)
        predicted = self.completed_frame(2)
        for phase, guess in enumerate(predicted):
            np.testing.assert_allclose(guess, (1.2 + 0.4 * phase) * self.vector, atol=1e-14)
        np.testing.assert_array_equal(self.predictor.counts.numpy(), [0, 4, 4])

    def test_rollback_and_partial_frame_never_enter_history(self):
        self.completed_frame(0)
        self.completed_frame(1)
        self.begin_frame(2)
        before = self.current.numpy().copy()
        self.predictor.begin(self.current, self.trial, 0.01, "same", self.engagement)
        self.predictor.discard()
        np.testing.assert_array_equal(self.current.numpy(), before)
        self.assertEqual(self.predictor.depth, 0)
        self.assertFalse(self.predictor.open)
        self.predictor.end_frame(0.08, False)
        self.assertEqual(self.predictor.depth, 0)

    def test_topology_event_and_force_fallback_clear_completed_history(self):
        self.completed_frame(0)
        self.completed_frame(1)
        self.begin_frame(2)
        self.accept(2, 0)
        before = self.current.numpy().copy()
        guess = self.accept(2, 1, signature="new-track")
        np.testing.assert_array_equal(guess, before)
        self.assertEqual(self.predictor.depth, 0)
        self.predictor.begin(self.current, self.trial, 0.01, "new-track", self.engagement)
        self.predictor.cancel()
        self.predictor.commit(self.current)
        self.predictor.end_frame(0.11, False)
        self.assertFalse(self.predictor.open)
        self.assertEqual(self.predictor.depth, 0)

    def test_discontinuous_command_reset_and_changed_phase_count(self):
        for change in ("command", "reset", "dt"):
            with self.subTest(change=change):
                self.predictor = PhaseLoadPredictor(self.current)
                self.command[:] = [0, 0, 0, 0, 0, 0, 1, 0]
                self.completed_frame(0)
                self.completed_frame(1)
                if change == "command":
                    self.command[0] = 0.01
                self.begin_frame(
                    2,
                    reset=int(change == "reset"),
                    step_dt=0.005 if change == "dt" else 0.01,
                )
                self.assertEqual(self.predictor.depth, 0)
                self.predictor.end_frame(0.08, False)

    def test_nonfinite_or_changed_engagement_retains_current_guess(self):
        for change in ("nonfinite", "engagement"):
            with self.subTest(change=change):
                self.predictor = PhaseLoadPredictor(self.current)
                self.engagement.assign(np.array([1, 0, 1], dtype=np.int32))
                self.completed_frame(0)
                self.completed_frame(1)
                self.begin_frame(2)
                if change == "nonfinite":
                    self.predictor.frames[0].loads[0].fill_(float("nan"))
                else:
                    self.engagement.assign(np.array([0, 0, 1], dtype=np.int32))
                before = self.current.numpy().copy()
                self.predictor.begin(self.current, self.trial, 0.01, "same", self.engagement)
                np.testing.assert_array_equal(self.trial.numpy(), before)
                self.predictor.discard()
                self.predictor.end_frame(0.08, False)

    def test_real_rejected_coupler_transaction_closes_and_discards_predictor(self):
        from test_needle_insertion_example import _example, _prepare
        from test_needle_tissue_coupler import _native_state

        from newton_solver_suturebot import NeedleTissueCoupler

        with _example() as example:
            _prepare(example)
            example.start()
            pose = example.plan.pose(0.0).copy()
            pose[2] += 0.6 * example.geometry.thickness
            example._initialize_pose(pose)
            example.state_0.body_f.assign(np.array([[0, 0, -0.0002, 0, 0, 0]], np.float32))
            native = _native_state(example.rigid_solver)
            coupler = NeedleTissueCoupler(
                example.model,
                example.solver,
                example.rigid_solver,
                example.interaction,
                example.needle_body,
                example.drive.com,
                max_iterations=1,
                absolute_tolerance=1e-14,
                relative_tolerance=1e-12,
            )
            predictor = PhaseLoadPredictor(coupler._load)
            coupler._phase_load_predictor = predictor
            predictor.begin_frame(
                frame_dt=example.sim_dt,
                step_dt=example.sim_dt,
                elapsed=0,
                command=self.command,
                reset_token=0,
            )
            with self.assertRaises(RuntimeError):
                coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
            np.testing.assert_array_equal(_native_state(example.rigid_solver), native)
            self.assertFalse(predictor.open)
            self.assertEqual(predictor.depth, 0)
            self.assertEqual(predictor.commits, 0)
            predictor.end_frame(0, False)


if __name__ == "__main__":
    unittest.main()
