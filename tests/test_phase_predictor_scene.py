# SPDX-License-Identifier: Apache-2.0
"""Worker control-frame boundaries own only complete accepted guess history."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene
from newton_solver_suturebot.examples.needle.example_fem_needle_insertion import Example


class PhasePredictorSceneTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.device.__enter__()
        self.scene = object.__new__(GraspScene)
        self.scene.running = True
        self.scene.worker_config = SimpleNamespace(control_mode="physical_grasp")
        self.scene.rigid_solver = None  # not a GPU rigid backend
        self.scene.coupler = SimpleNamespace(_phase_load_predictor=None, _load=wp.zeros(6, dtype=wp.float64))
        self.scene.frame_dt, self.scene.sim_dt, self.scene.elapsed = 0.04, 0.01, 0.0
        self.scene.cartesian_target = np.array([0, 0, 0, 0, 0, 0, 1.0])
        self.scene.jaw_opening_target = 0.0
        self.scene.reset_count = 0
        self.trial = wp.empty_like(self.scene.coupler._load)
        self.engagement = wp.array([1, 0], dtype=int)
        self.guesses = []

    def tearDown(self):
        self.device.__exit__(None, None, None)

    def advance(self, scene, *, phases=4, stop=False, interrupt=False):
        predictor = scene.coupler._phase_load_predictor
        frame = round(scene.elapsed / scene.frame_dt)
        for phase in range(phases):
            predictor.begin(scene.coupler._load, self.trial, scene.sim_dt, "stable", self.engagement)
            self.guesses.append(self.trial.numpy().copy())
            scene.coupler._load.fill_(1.0 + 0.1 * frame + 0.4 * phase)
            predictor.commit(scene.coupler._load)
            scene.elapsed += scene.sim_dt
        if stop:
            scene.running = False
        if interrupt:
            raise KeyboardInterrupt("interrupted after physical advance")

    def test_worker_frame_reuses_only_completed_same_phase_loads(self):
        with patch.object(Example, "step", lambda scene: self.advance(scene)):
            self.scene.step()
            predictor = self.scene.coupler._phase_load_predictor
            self.assertEqual((predictor.depth, predictor.commits, predictor.attempts), (1, 4, 0))
            self.scene.cartesian_target[0] += 0.00005
            self.scene.step()
            self.assertEqual((predictor.depth, predictor.commits, predictor.attempts), (2, 8, 4))
        for phase, guess in enumerate(self.guesses[4:]):
            np.testing.assert_allclose(guess, 1.0 + 0.4 * phase)
        self.assertFalse(predictor.frame_open)

    def test_only_the_device_anderson_gpu_coupling_selects_the_frame_periodic_increment_guess(self):
        self.scene._gpu_pose_ik_active = True
        with patch.object(Example, "step", lambda scene: self.advance(scene)):
            for gpu, policy, expected in (
                (False, None, False),
                (False, "device-anderson", False),
                (True, None, False),
                (True, "current", False),
                (True, "device-aitken", False),
                (True, "device-anderson", True),
            ):
                with self.subTest(gpu=gpu, policy=policy):
                    self.scene.coupler._phase_load_predictor = None
                    self.scene.elapsed = 0.0
                    if policy is None:
                        vars(self.scene.coupler).pop("_gpu_trial_policy", None)
                    else:
                        self.scene.coupler._gpu_trial_policy = policy
                    with patch("newton_solver_suturebot.examples.grasp._grasp_scene.is_gpu_backend", return_value=gpu):
                        self.scene.step()
                    predictor = self.scene.coupler._phase_load_predictor
                    self.assertEqual(predictor.increments, expected)
                    self.assertEqual((predictor.depth, predictor.commits), (1, 4))

    def test_direct_manual_caller_without_worker_fields_keeps_original_step(self):
        for missing in (
            "worker_config",
            "cartesian_target",
            "jaw_opening_target",
            "reset_count",
        ):
            with self.subTest(missing=missing):
                value = getattr(self.scene, missing)
                delattr(self.scene, missing)
                with patch.object(Example, "step", return_value="ordinary") as original:
                    self.assertEqual(self.scene.step(), "ordinary")
                    original.assert_called_once_with()
                self.assertIsNone(self.scene.coupler._phase_load_predictor)
                setattr(self.scene, missing, value)

    def test_partial_failed_or_interrupted_frame_discards_history(self):
        for phases, stop, interrupt in ((2, True, False), (4, False, True)):
            with self.subTest(phases=phases, stop=stop, interrupt=interrupt):
                self.scene.coupler._phase_load_predictor = None
                self.scene.elapsed = 0.0
                self.scene.running = True
                with patch.object(Example, "step", lambda scene: self.advance(scene)):
                    self.scene.step()

                def incomplete(scene, phases=phases, stop=stop, interrupt=interrupt):
                    self.advance(scene, phases=phases, stop=stop, interrupt=interrupt)

                with patch.object(Example, "step", incomplete):
                    if interrupt:
                        with self.assertRaises(KeyboardInterrupt):
                            self.scene.step()
                    else:
                        self.scene.step()
                predictor = self.scene.coupler._phase_load_predictor
                self.assertEqual(predictor.depth, 0)
                self.assertFalse(predictor.frame_open)
                self.assertFalse(predictor.open)

    def test_reset_restarts_history_without_reusing_previous_episode(self):
        with patch.object(Example, "step", lambda scene: self.advance(scene)):
            self.scene.step()
            self.scene.step()
            predictor = self.scene.coupler._phase_load_predictor
            attempts = predictor.attempts
            self.scene.reset_count += 1
            self.scene.elapsed = 0.0
            self.scene.coupler._load.zero_()
            self.scene.step()
        self.assertEqual(predictor.attempts, attempts)
        self.assertEqual(predictor.depth, 1)
        np.testing.assert_array_equal(self.guesses[-4], np.zeros(6))


if __name__ == "__main__":
    unittest.main()
