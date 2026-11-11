# SPDX-License-Identifier: Apache-2.0
"""Input contracts and opt-in physical worker isolation/reset checks.

The integration check runs the authored Wound 1 mesh, native MuJoCo needle,
and FEM solver. Enable with NEWTON_TEST_PARALLEL=1; NEWTON_TEST_DEVICE selects
the device. It does not substitute recorded motion or a small test mesh.
"""

import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from newton_solver_suturebot.parallel import ParallelSutureScenes, SutureSceneConfig


class ParallelSceneConfigurationTests(unittest.TestCase):
    def test_invalid_time_scales_coordinates_and_counts_fail_before_starting_workers(self):
        for arguments in (
            {"substeps": 0},
            {"region": 0},
            {"frame_dt": 0},
            {"translation_step": np.inf},
            {"rotation_step": -1},
            {"entry_xy": (np.nan, 0)},
            {"reset_position_noise_m": -1},
            {"reset_position_noise_m": np.inf},
            {"seed": -1},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                SutureSceneConfig(**arguments)
        for arguments in ({"num_envs": 0}, {"num_envs": 1.5}, {"timeout": 0}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ParallelSutureScenes(**arguments)


@unittest.skipUnless(os.environ.get("NEWTON_TEST_PARALLEL") == "1", "Opt-in authored multi-process physics workload")
class ParallelScenePhysicsTests(unittest.TestCase):
    def test_seeded_resets_are_reproducible_independent_and_reset_only(self):
        config = SutureSceneConfig(
            device=os.environ.get("NEWTON_TEST_DEVICE", "cpu"), reset_position_noise_m=0.0007, seed=42
        )
        records = []
        for _ in range(2):
            with TemporaryDirectory() as logs, ParallelSutureScenes(config, num_envs=2, log_dir=logs) as scenes:
                initial = scenes.observe()
                canonical = scenes.geometry()["ready_pose"]
                self.assertFalse(np.array_equal(initial[0]["needle_pose"], initial[1]["needle_pose"]))
                for observation in initial:
                    offset = observation["needle_pose"][:3] - canonical[:3]
                    self.assertLessEqual(np.max(np.abs(offset)), config.reset_position_noise_m + 1e-8)
                    self.assertGreaterEqual(offset[2], -1e-8)
                    np.testing.assert_allclose(observation["target_pose"], observation["needle_pose"], atol=1e-8)
                    self.assertEqual(observation["diagnostics"]["accepted_physics_steps"], 0)
                scenes.step(np.zeros((2, 6)))
                unselected = scenes.observe([1])[0]
                reset = scenes.reset([0])[0]
                after = scenes.observe([1])[0]
                np.testing.assert_array_equal(after["needle_pose"], unselected["needle_pose"])
                self.assertEqual(after["reset_count"], 0)
                self.assertEqual(reset["reset_count"], 1)
                self.assertEqual(reset["diagnostics"]["physical_time_s"], 0)
                self.assertEqual(reset["diagnostics"]["entry_count"], 0)
                self.assertFalse(np.array_equal(reset["needle_pose"], initial[0]["needle_pose"]))
                records.append((initial[0]["needle_pose"], initial[1]["needle_pose"], reset["needle_pose"]))
        for first, second in zip(*records, strict=True):
            np.testing.assert_array_equal(first, second)

    def test_actions_affect_only_their_scene_and_selective_reset_preserves_other_history(self):
        config = SutureSceneConfig(device=os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        with TemporaryDirectory() as logs, ParallelSutureScenes(config, num_envs=2, log_dir=logs) as scenes:
            initial = scenes.observe()
            compact = scenes.observe(include_tissue=False)
            for full, small in zip(initial, compact, strict=True):
                self.assertNotIn("tissue_positions", small)
                self.assertNotIn("tissue_velocities", small)
                np.testing.assert_array_equal(small["needle_pose"], full["needle_pose"])
                np.testing.assert_allclose(small["entry_position"], full["entry_position"], rtol=0, atol=1e-8)
                np.testing.assert_allclose(small["exit_position"], full["exit_position"], rtol=0, atol=1e-8)
            geometry = scenes.geometry()
            self.assertNotEqual(initial[0]["pid"], initial[1]["pid"])
            self.assertGreater(len(geometry["tissue_tetrahedra"]), 1000)
            actions = np.zeros((2, 6))
            actions[0, 0] = 0.2
            result = scenes.step(actions)
            for _ in range(2):
                result = scenes.step(np.zeros((2, 6)))
            for value in result:
                self.assertIsNone(value["diagnostics"]["failure_reason"])
                self.assertGreater(value["diagnostics"]["accepted_physics_steps"], 0)
                self.assertTrue(np.isfinite(value["tissue_positions"]).all())
                fixed = geometry["fixed_particle_indices"]
                np.testing.assert_array_equal(value["tissue_positions"][fixed], initial[0]["tissue_positions"][fixed])
            delta0 = result[0]["needle_pose"][0] - initial[0]["needle_pose"][0]
            delta1 = result[1]["needle_pose"][0] - initial[1]["needle_pose"][0]
            self.assertGreater(delta0, 0.5 * config.translation_step * actions[0, 0])
            self.assertLess(abs(delta1), 1.0e-7)
            before_reset = scenes.observe([1])[0]
            reset = scenes.reset([0])[0]
            after_reset = scenes.observe([1])[0]
            self.assertEqual(reset["diagnostics"]["physical_time_s"], 0.0)
            self.assertEqual(reset["diagnostics"]["entry_count"], 0)
            np.testing.assert_array_equal(reset["needle_pose"], initial[0]["needle_pose"])
            np.testing.assert_array_equal(reset["tissue_positions"], initial[0]["tissue_positions"])
            for name in ("needle_pose", "needle_velocity", "tissue_positions", "tissue_velocities", "target_pose"):
                np.testing.assert_array_equal(after_reset[name], before_reset[name], err_msg=name)
            self.assertEqual(after_reset["diagnostics"], before_reset["diagnostics"])
            with self.assertRaises(ValueError):
                scenes.step(np.full((2, 6), np.nan))
            with self.assertRaises(ValueError):
                scenes.reset([0, 0])


if __name__ == "__main__":
    unittest.main()
