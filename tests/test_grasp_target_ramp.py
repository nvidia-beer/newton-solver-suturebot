# SPDX-License-Identifier: Apache-2.0
"""The arm actuator targets of a frame are spread over the frame's intervals."""

import os
import unittest

import numpy as np
import warp as wp
from test_grasp_scene import _scene

from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene, _ramp_arm_targets
from newton_solver_suturebot.examples.grasp.parallel import _apply_action, _reset_scene, _step

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


class RampKernelTests(unittest.TestCase):
    def test_targets_lie_on_the_line_between_two_frames_and_only_the_arm_entries_change(self):
        previous = wp.array([0.1, -0.2, 0.3], dtype=float, device=DEVICE)
        target = wp.array([0.5, -0.6, 0.3], dtype=float, device=DEVICE)
        indices = wp.array([4, 1, 2], dtype=int, device=DEVICE)
        joint_target = wp.array(np.arange(6, dtype=np.float32) + 10, dtype=float, device=DEVICE)
        for fraction in (0.25, 0.5, 1.0):
            wp.launch(_ramp_arm_targets, 3, [previous, target, fraction, indices, joint_target], device=DEVICE)
            result = joint_target.numpy()
            expected = np.float32(previous.numpy()) + np.float32(fraction) * (target.numpy() - previous.numpy())
            np.testing.assert_array_equal(result[[4, 1, 2]], expected)
            np.testing.assert_array_equal(result[[0, 3, 5]], [10, 13, 15])
        # The last interval of a frame ends at the frame's targets themselves.
        np.testing.assert_array_equal(joint_target.numpy()[[4, 1, 2]], target.numpy())


class RampedSceneTests(unittest.TestCase):
    def test_no_ramp_outside_a_frame(self):
        scene = object.__new__(GraspScene)
        scene._ramp_frame = None
        # Nothing of the scene is touched: a call outside a frame leaves the held targets.
        self.assertIsNone(scene._ramp_first_arm(1 / 240))

    def test_each_frame_ramps_from_the_previous_targets_and_a_reset_starts_anew(self):
        with _scene() as (scene, config):
            indices = list(scene.graspers[0].arm_q_indices)
            parked = list(scene.graspers[1].arm_q_indices)
            ramp = scene._ramp_first_arm
            recorded = []

            def recording(dt):
                ramp(dt)
                targets = scene.control.joint_target_q.numpy()
                recorded.append((scene.elapsed, dt, targets[indices].copy(), targets[parked].copy()))

            scene._ramp_first_arm = recording
            _apply_action(scene, config, [1, 0, 0, 0, 0, 0, 0])
            _step(scene)
            held = np.asarray(scene.arm_targets[0], dtype=np.float32)
            parked_targets = np.asarray(scene.arm_targets[1], dtype=np.float32)
            # No frame precedes the first one after a reset: its targets are held.
            self.assertGreaterEqual(len(recorded), config.substeps)
            for _, _, targets, other in recorded:
                np.testing.assert_array_equal(targets, held)
                np.testing.assert_array_equal(other, parked_targets)
            self.assertIsNone(scene._ramp_frame)

            recorded.clear()
            start = scene.elapsed
            _apply_action(scene, config, [1, 0, 0, 0, 0, 0, 0])
            _step(scene)
            solved = np.asarray(scene.arm_targets[0], dtype=np.float32)
            self.assertTrue(np.any(solved != held))
            self.assertAlmostEqual(scene.elapsed - start, config.frame_dt, delta=1e-12)
            self.assertGreaterEqual(len(recorded), config.substeps)
            for elapsed, dt, targets, other in recorded:
                fraction = np.float32((elapsed + dt - start) / config.frame_dt)
                np.testing.assert_allclose(targets, held + fraction * (solved - held), rtol=0, atol=1e-7)
                # The parked tool is not ramped.
                np.testing.assert_array_equal(other, parked_targets)
            fractions = sorted({round((elapsed + dt - start) / config.frame_dt, 9) for elapsed, dt, _, _ in recorded})
            self.assertEqual(fractions[-1], 1.0)
            self.assertLess(fractions[0], 1.0)
            # The frame ends at the targets of the commanded pose, as it did with held targets.
            np.testing.assert_array_equal(scene.control.joint_target_q.numpy()[indices], solved)

            _reset_scene(scene, config, rebuild=True)
            recorded.clear()
            scene._ramp_first_arm = recording
            _apply_action(scene, config, [1, 0, 0, 0, 0, 0, 0])
            _step(scene)
            first = np.asarray(scene.arm_targets[0], dtype=np.float32)
            for _, _, targets, _ in recorded:
                np.testing.assert_array_equal(targets, first)


if __name__ == "__main__":
    unittest.main()
