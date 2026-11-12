# SPDX-License-Identifier: Apache-2.0
"""Actual native MuJoCo kinematics; no substitution of live joint coordinates."""

import unittest

import mujoco
import numpy as np

from newton_solver_suturebot._src.pose_ik import NativePoseIK, UnreachablePoseError


def _model():
    return mujoco.MjModel.from_xml_string("""
    <mujoco><compiler angle="radian"/><option gravity="0 0 0"/>
    <default><joint limited="true" range="-2 2"/><geom type="sphere" size="0.003" mass="0.01" contype="0" conaffinity="0"/></default>
    <worldbody>
      <body name="needle" pos="0.4 0.2 0.3"><freejoint/><geom/></body>
      <body name="yaw" pos="0.03 -0.04 0.15"><joint name="yaw" axis="0 0 1"/><geom/>
        <body name="pitch"><joint name="pitch" axis="0 1 0"/><geom/>
          <body name="shaft" pos="0 0 -0.1"><joint name="insertion" type="slide" axis="0 0 1" range="-0.08 0.08"/><geom/>
            <body name="roll"><joint name="roll" axis="0 0 1" range="-3 3"/><geom/>
              <body name="wrist_pitch" pos="0 0 -0.01"><joint name="wrist_pitch" axis="1 0 0"/><geom/>
                <body name="palm" pos="0 0 -0.008"><joint name="wrist_yaw" axis="0 1 0"/><geom/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </worldbody></mujoco>
    """)


def _fixture():
    model = _model()
    ids = [model.joint(name).id for name in ("yaw", "pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw")]
    body = model.body("palm").id
    return model, mujoco.MjData(model), ids, body


def _target(model, coordinates, values, body):
    data = mujoco.MjData(model)
    data.qpos[coordinates] = values
    mujoco.mj_kinematics(model, data)
    return np.r_[data.xpos[body], data.xquat[body][[1, 2, 3, 0]]]


class PoseIKTests(unittest.TestCase):
    def test_jacobian_matches_full_pose_finite_differences(self):
        model, live, ids, body = _fixture()
        ik = NativePoseIK(model, ids, body)
        values = np.array([0.3, -0.4, 0.02, 0.5, 0.3, -0.4])
        target = _target(model, ik.coordinates, values + [0.1, 0.2, -0.01, -0.2, 0.15, 0.2], body)
        ik.scratch.qpos[:] = live.qpos
        _, jacobian = ik._evaluate(values, target)
        finite_difference = np.empty((6, 6))
        for column in range(6):
            delta = np.zeros(6)
            delta[column] = 1e-6
            before = ik._evaluate(values - delta, target)[0]
            after = ik._evaluate(values + delta, target)[0]
            finite_difference[:, column] = (after - before) / (2e-6)
        np.testing.assert_allclose(jacobian, finite_difference, atol=1e-9, rtol=1e-6)

    def test_solves_full_pose_without_mutating_live_state_and_preserves_rcm(self):
        model, live, ids, body = _fixture()
        ik = NativePoseIK(model, ids, body)
        coordinates = model.jnt_qposadr[ids]
        live.qpos[coordinates] = [0.1, -0.2, 0.01, 0.3, 0.2, -0.1]
        mujoco.mj_forward(model, live)
        saved_qpos, saved_xpos, saved_time = (
            live.qpos.copy(),
            live.xpos.copy(),
            live.time,
        )
        target = _target(model, coordinates, [0.2, -0.3, 0.015, 0.4, 0.3, -0.2], body)
        result = ik.solve(target, live.qpos, initial=live.qpos[coordinates])
        self.assertTrue(ik.last_diagnostics["converged"])
        self.assertLess(ik.last_diagnostics["position_error_m"], 1e-6)
        self.assertLess(ik.last_diagnostics["orientation_error_rad"], 1e-4)
        self.assertTrue(np.all(result >= ik.lower) and np.all(result <= ik.upper))
        np.testing.assert_array_equal(live.qpos, saved_qpos)
        np.testing.assert_array_equal(live.xpos, saved_xpos)
        self.assertEqual(live.time, saved_time)
        shaft = model.body("shaft").id
        axis = ik.scratch.xmat[shaft].reshape(3, 3)[:, 2]
        pivot = np.array([0.03, -0.04, 0.15])
        self.assertLess(np.linalg.norm(np.cross(pivot - ik.scratch.xpos[shaft], axis)), 1e-12)

    def test_cache_ignores_independent_needle_motion_and_returns_unaliased_targets(
        self,
    ):
        model, live, ids, body = _fixture()
        ik = NativePoseIK(model, ids, body)
        target = _target(model, ik.coordinates, [0.2, -0.3, 0.015, 0.4, 0.3, -0.2], body)
        first = ik.solve(target, live.qpos)
        live.qpos[0] += 0.05
        second = ik.solve(target, live.qpos)
        self.assertTrue(ik.last_diagnostics["cached"])
        self.assertEqual(ik.last_diagnostics["iterations"], 0)
        np.testing.assert_array_equal(first, second)
        second[:] = 100
        np.testing.assert_array_equal(ik.solve(target, live.qpos), first)
        ik.reset(live.qpos)
        ik.solve(target, live.qpos)
        self.assertFalse(ik.last_diagnostics["cached"])

    def test_unreachable_target_reports_error_without_accepting_or_mutating_physics(
        self,
    ):
        model, live, ids, body = _fixture()
        ik = NativePoseIK(model, ids, body)
        target = _target(model, ik.coordinates, [0.2, -0.3, 0.015, 0.4, 0.3, -0.2], body)
        saved = live.qpos.copy()
        target[:3] += [1, 1, 1]
        with self.assertRaisesRegex(UnreachablePoseError, "unreachable"):
            ik.solve(target, live.qpos)
        self.assertFalse(ik.last_diagnostics["converged"])
        np.testing.assert_array_equal(live.qpos, saved)

    def test_rejects_free_coordinate_in_arm_and_invalid_pose(self):
        model, live, ids, body = _fixture()
        with self.assertRaisesRegex(ValueError, "scalar hinge"):
            NativePoseIK(model, [0, *ids[1:]], body)
        ik = NativePoseIK(model, ids, body)
        with self.assertRaisesRegex(ValueError, "nonzero quaternion"):
            ik.solve(np.zeros(7), live.qpos)


if __name__ == "__main__":
    unittest.main()
