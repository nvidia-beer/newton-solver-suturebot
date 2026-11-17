"""Scene selection and actuator publication never invoke native kinematics in GPU IK frames."""

import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import warp as wp
from test_grasp_scene import _scene

from newton_solver_suturebot._src.sim.ik.gpu_pose_ik import GPUPoseIK
from newton_solver_suturebot._src.sim.ik.pose_ik import NativePoseIK
from newton_solver_suturebot.examples.grasp import _grasp_scene as module


class DeviceSceneIKTests(unittest.TestCase):
    def test_native_default_and_both_arm_unsupported_fallback(self):
        with _scene() as (scene, _):
            self.assertFalse(scene._gpu_pose_ik_active)
            old = tuple(scene.pose_ik)
            calls = []

            def constructor(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise ValueError("unsupported second arm")
                return GPUPoseIK(*args, **kwargs)

            with (
                patch.object(module, "is_gpu_backend", return_value=True),
                patch(
                    "newton_solver_suturebot._src.sim.ik.gpu_pose_ik.GPUPoseIK",
                    side_effect=constructor,
                ),
            ):
                scene._configure_gpu_pose_ik()
            self.assertEqual(tuple(scene.pose_ik), old)
            self.assertTrue(all(type(ik) is NativePoseIK for ik in scene.pose_ik))
            self.assertFalse(scene._gpu_pose_ik_active)
            self.assertIn("second arm", scene._gpu_pose_ik_fallback)

    def test_real_device_ik_update_uses_no_native_shadow_kinematics_or_control_download(
        self,
    ):
        with _scene() as (scene, _):
            old = scene.state_0.joint_q.numpy().copy()
            with patch.object(module, "is_gpu_backend", return_value=True):
                scene._configure_gpu_pose_ik()
            self.assertTrue(scene._gpu_pose_ik_active)
            desired = scene.cartesian_target.copy()
            # A small reachable command forces fresh IK and actuator publication.
            desired[0] += 0.00001
            original_numpy = wp.array.numpy
            control_pointer = scene.control.joint_target_q.ptr

            def checked_numpy(array, *args, **kwargs):
                if array.ptr == control_pointer:
                    raise AssertionError("full actuator target download")
                return original_numpy(array, *args, **kwargs)

            with (
                patch.object(
                    mujoco,
                    "mj_kinematics",
                    side_effect=AssertionError("native kinematics"),
                ),
                patch.object(mujoco, "mj_forward", side_effect=AssertionError("native forward")),
                patch.object(wp.array, "numpy", checked_numpy),
            ):
                scene._prepare_step_loads(desired, np.zeros(6), scene.sim_dt)
            np.testing.assert_array_equal(scene.state_0.joint_q.numpy(), old)
            for assembly, target in zip(scene.graspers, scene.arm_targets, strict=True):
                np.testing.assert_array_equal(
                    scene.control.joint_target_q.numpy()[list(assembly.arm_q_indices)],
                    np.asarray(target, dtype=np.float32),
                )

    def test_supported_frame_skips_shadow_but_native_fallback_preserves_it(self):
        with _scene() as (scene, _):
            with patch.object(module, "is_gpu_backend", return_value=True):
                scene._configure_gpu_pose_ik()
            # Exercise only frame orchestration, not native physical integration.
            scene.running = False
            with (
                patch.object(module, "is_gpu_backend", return_value=True),
                patch.object(module.Example, "step") as base,
                patch.object(scene.rigid_solver, "sync_native_shadow", create=True) as shadow,
            ):
                scene.step()
                base.assert_called_once()
                shadow.assert_not_called()
                scene._gpu_pose_ik_active = False
                scene.step()
                shadow.assert_called_once_with(forward=True, token=scene.elapsed)
