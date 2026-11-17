# SPDX-License-Identifier: Apache-2.0
"""The complete procedural insertion must fit the fixed-trocar joint limits."""

import os
import unittest

import mujoco
import newton
import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleGeometry
from newton_solver_suturebot._src.sim.ik.pose_ik import NativePoseIK, UnreachablePoseError
from newton_solver_suturebot._src.solvers.mujoco.subcycled_mujoco import SubcycledMuJoCo
from newton_solver_suturebot.examples.grasp._dvrk_grasper import needle_to_gripper_pose
from newton_solver_suturebot.examples.grasp._dvrk_rcm_grasper import add_rcm_grasper


class RCMInsertionIKTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.device.__enter__()
        self.geometry = NeedleGeometry.load()
        self.relative_grasp = needle_to_gripper_pose(self.geometry)
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_usd(
            str(self.geometry.asset_path),
            collapse_fixed_joints=False,
            hide_collision_shapes=True,
        )
        self.tool = add_rcm_grasper(builder, label="PSM1")
        self.model = builder.finalize()
        self.state = self.model.state()
        self.tool.set_anchor(self.model, self.relative_grasp)
        self.tool.initialize(self.model, self.state, self.relative_grasp)
        self.solver = SubcycledMuJoCo(self.model, use_mujoco_cpu=True, kinematic_switchable_roots=False)
        self.solver.reset(self.state, flags=0)
        native = self.solver.mj_model
        joints = [
            mujoco.mj_name2id(
                native,
                mujoco.mjtObj.mjOBJ_JOINT,
                self.model.joint_label[joint].replace("/", "_"),
            )
            for joint in self.tool.arm_joints
        ]
        body = mujoco.mj_name2id(
            native,
            mujoco.mjtObj.mjOBJ_BODY,
            self.model.body_label[self.tool.palm_body].replace("/", "_"),
        )
        self.ik = NativePoseIK(native, joints, body)
        self.ik.reset(self.solver.mj_data.qpos)

    def tearDown(self):
        self.device.__exit__(None, None, None)

    def target(self, angle):
        """Rotate the bundled needle around its arc centre, carrying its grasp."""
        rotation = wp.quat_from_axis_angle(wp.vec3(*self.geometry.arc_normal), float(angle))
        translation = self.geometry.arc_center - np.asarray(
            wp.quat_rotate(rotation, wp.vec3(*self.geometry.arc_center))
        )
        return np.asarray(
            wp.transform_multiply(
                wp.transform(translation, rotation),
                wp.transform(*self.relative_grasp),
            )
        )

    def test_complete_procedural_arc_is_reachable_without_changing_live_physics(self):
        live = self.solver.mj_data
        saved_native = {name: getattr(live, name).copy() for name in ("qpos", "qvel", "xpos", "xquat", "ctrl")}
        saved_time = live.time
        saved_joints = self.state.joint_q.numpy().copy()
        saved_bodies = self.state.body_q.numpy().copy()
        saved_anchors = self.model.joint_X_p.numpy().copy()
        commands = []
        # The bundled default entry/exit pair needs approximately 1.29093 rad.
        # A zero-wrist ready posture used to hit its limit after only 0.89 rad.
        for angle in np.linspace(0.0, 1.291, 41):
            command = self.ik.solve(self.target(angle), live.qpos)
            commands.append(command)
            self.assertTrue(self.ik.last_diagnostics["converged"])
            self.assertLessEqual(
                self.ik.last_diagnostics["position_error_m"],
                self.ik.position_tolerance_m,
            )
            self.assertLessEqual(
                self.ik.last_diagnostics["orientation_error_rad"],
                self.ik.orientation_tolerance_rad,
            )
            self.assertTrue(np.all(command >= self.tool.arm_lower))
            self.assertTrue(np.all(command <= self.tool.arm_upper))
        commands = np.asarray(commands)
        # This is an articulated insertion, including axial shaft travel.
        self.assertGreater(np.ptp(commands[:, 2]), 0.001)
        self.assertGreater(np.ptp(commands[:, 4]), 1.0)
        for name, saved in saved_native.items():
            np.testing.assert_array_equal(getattr(live, name), saved)
        self.assertEqual(live.time, saved_time)
        np.testing.assert_array_equal(self.state.joint_q.numpy(), saved_joints)
        np.testing.assert_array_equal(self.state.body_q.numpy(), saved_bodies)
        np.testing.assert_array_equal(self.model.joint_X_p.numpy(), saved_anchors)

    def test_unreachable_pose_does_not_relax_limits_or_move_physical_state(self):
        live = self.solver.mj_data
        target = self.target(0.0)
        self.ik.solve(target, live.qpos)
        saved_qpos = live.qpos.copy()
        saved_bodies = self.state.body_q.numpy().copy()
        saved_anchors = self.model.joint_X_p.numpy().copy()
        target[:3] += [1.0, 1.0, 1.0]
        with self.assertRaisesRegex(UnreachablePoseError, "unreachable"):
            self.ik.solve(target, live.qpos)
        self.assertFalse(self.ik.last_diagnostics["converged"])
        np.testing.assert_array_equal(live.qpos, saved_qpos)
        np.testing.assert_array_equal(self.state.body_q.numpy(), saved_bodies)
        np.testing.assert_array_equal(self.model.joint_X_p.numpy(), saved_anchors)


if __name__ == "__main__":
    unittest.main()
