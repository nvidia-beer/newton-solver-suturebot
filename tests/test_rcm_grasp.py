# SPDX-License-Identifier: Apache-2.0
"""Fixed-trocar kinematics and native joint drives, without needle pose locks."""

import os
import unittest

import mujoco
import newton
import numpy as np
import warp as wp

from newton_solver_suturebot._src.solvers.mujoco.subcycled_mujoco import SubcycledMuJoCo
from newton_solver_suturebot.examples.grasp._dvrk_grasper import GrasperConfig
from newton_solver_suturebot.examples.grasp._dvrk_rcm_grasper import add_rcm_grasper


class _Rig:
    def __init__(self, **tool_options):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        self.needle = builder.add_link(mass=0.001, inertia=wp.mat33(np.eye(3) * 1e-7), label="needle")
        needle_joint = builder.add_joint_free(child=self.needle, label="needle/free")
        builder.add_articulation([needle_joint], label="needle")
        self.tool = add_rcm_grasper(builder, label="PSM1", **tool_options)
        self.model = builder.finalize()
        self.state, self.following = self.model.state(), self.model.state()
        coordinates = self.state.joint_q.numpy().copy()
        coordinates[:7] = (1, 2, 3, 0, 0, 0, 1)
        self.state.joint_q.assign(coordinates)
        rotation = np.asarray(wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.65), dtype=float)
        self.palm_pose = np.r_[0.03, -0.02, 0.05, rotation]
        self.tool.set_anchor(self.model, self.palm_pose)
        self.tool.initialize(self.model, self.state, self.palm_pose)
        self.control = self.model.control()
        self.tool.set_aperture(
            self.control, self.tool.config.initial_opening_rad / self.tool.config.maximum_opening_rad
        )
        self.tool.set_arm_targets(self.control, self.tool.nominal_arm_positions)

    def native_solver(self):
        return SubcycledMuJoCo(
            self.model,
            use_mujoco_cpu=True,
            rigid_substeps=4,
            integrator="implicitfast",
            iterations=200,
            ls_iterations=50,
            tolerance=1e-12,
            cone="elliptic",
            impratio=10,
            kinematic_switchable_roots=False,
        )

    def rcm(self):
        pose = self.state.body_q.numpy()[self.tool.part_body_indices["roll"]]
        axis = np.asarray(wp.quat_rotate(wp.quat(*pose[3:]), wp.vec3(0, 0, 1)))
        displacement = self.tool.trocar_pose[:3] - pose[:3]
        return np.linalg.norm(np.cross(displacement, axis)), float(displacement @ axis)


class RCMGrasperTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.device.__enter__()

    def tearDown(self):
        self.device.__exit__(None, None, None)

    def test_reset_matches_nominal_palm_and_preserves_independent_needle(self):
        rig = _Rig()
        np.testing.assert_array_equal(rig.state.joint_q.numpy()[:7], [1, 2, 3, 0, 0, 0, 1])
        palm = rig.state.body_q.numpy()[rig.tool.palm_body]
        np.testing.assert_allclose(palm[:3], rig.palm_pose[:3], atol=2e-7)
        self.assertAlmostEqual(abs(np.dot(palm[3:], rig.palm_pose[3:])), 1, delta=2e-7)
        self.assertEqual(rig.model.body_count, 9)
        self.assertEqual(rig.model.joint_coord_count, 15)
        self.assertEqual(rig.model.joint_dof_count, 14)
        np.testing.assert_array_equal(rig.model.joint_type.numpy(), [4, 1, 1, 0, 1, 1, 1, 1, 1])
        self.assertEqual(rig.model.joint_parent.numpy()[rig.tool.arm_joints[0]], -1)
        self.assertLess(rig.rcm()[0], 1e-7)
        self.assertAlmostEqual(rig.rcm()[1], -0.08, delta=1e-7)
        np.testing.assert_array_equal(
            rig.state.joint_q.numpy()[list(rig.tool.arm_q_indices)], np.asarray([0, 0, 0, 0, -0.7, 0], dtype=np.float32)
        )
        before = rig.model.joint_X_p.numpy().copy()
        moved = rig.palm_pose.copy()
        moved[0] += 0.001
        with self.assertRaises(RuntimeError):
            rig.tool.set_anchor(rig.model, moved)
        with self.assertRaises(ValueError):
            rig.tool.initialize(rig.model, rig.state, moved)
        np.testing.assert_array_equal(rig.model.joint_X_p.numpy(), before)
        # Explicit reset authoring may replace a preview before solver creation.
        rig.tool.set_anchor(rig.model, moved, reset=True)
        rig.tool.initialize(rig.model, rig.state)
        np.testing.assert_allclose(rig.state.body_q.numpy()[rig.tool.palm_body, :3], moved[:3], atol=2e-7)

    def test_jointed_fk_keeps_trocar_fixed_and_wrist_articulates_independently(self):
        rig = _Rig()
        q = rig.state.joint_q.numpy().copy()
        q[list(rig.tool.arm_q_indices)] = [0.25, -0.3, 0.025, 0.5, 0.4, -0.3]
        rig.state.joint_q.assign(q)
        newton.eval_fk(rig.model, rig.state.joint_q, rig.state.joint_qd, rig.state)
        error, axial = rig.rcm()
        self.assertLess(error, 1e-7)
        self.assertAlmostEqual(axial, -0.105, delta=1e-7)
        self.assertGreaterEqual(axial, -0.123)
        self.assertLessEqual(axial, -0.003)
        before = rig.state.body_q.numpy().copy()
        q[list(rig.tool.arm_q_indices)[4:]] = [-0.4, 0.3]
        rig.state.joint_q.assign(q)
        newton.eval_fk(rig.model, rig.state.joint_q, rig.state.joint_qd, rig.state)
        after = rig.state.body_q.numpy()
        np.testing.assert_array_equal(
            after[rig.tool.part_body_indices["roll"]], before[rig.tool.part_body_indices["roll"]]
        )
        self.assertGreater(np.linalg.norm(after[rig.tool.palm_body] - before[rig.tool.palm_body]), 0.1)

    def test_collision_capsules_use_the_actual_articulated_link_frames(self):
        rig = _Rig()
        # Every capsule lies on the link of the drawn part that it stands for; a part has one or more capsules.
        links = {"shaft": "roll", "wrist": "pitch", "jaw_1": "jaw1", "jaw_2": "jaw2"}
        self.assertEqual(set(rig.tool.capsule_parts), set(links))
        self.assertEqual(
            rig.tool.capsule_bodies,
            tuple(rig.tool.part_body_indices[links[name]] for name in rig.tool.capsule_parts),
        )
        transforms, scales = (rig.model.shape_transform.numpy(), rig.model.shape_scale.numpy())
        for index, shape in enumerate(rig.tool.capsule_shapes):
            transform = wp.transform(*transforms[shape])
            endpoints = [
                np.asarray(wp.transform_point(transform, wp.vec3(0, 0, sign * scales[shape, 1]))) for sign in (-1, 1)
            ]
            np.testing.assert_allclose(endpoints, rig.tool.capsule_endpoints[index], atol=1e-8)
            self.assertAlmostEqual(scales[shape, 0], rig.tool.capsule_radii[index], delta=1e-9)

    def test_native_actuators_insert_and_retract_without_moving_trocar_or_needle(self):
        rig = _Rig()
        solver = rig.native_solver()
        self.assertEqual(solver.mj_model.neq, 0)
        self.assertEqual(solver.mj_model.nu, 8)
        pivot = rig.tool.trocar_pose.copy()
        needle = rig.state.body_q.numpy()[rig.needle].copy()
        for target in (0.005, -0.005):
            values = rig.tool.nominal_arm_positions.copy()
            values[2] = target
            before = rig.state.joint_q.numpy().copy()
            rig.tool.set_arm_targets(rig.control, values)
            np.testing.assert_array_equal(rig.state.joint_q.numpy(), before)
            for _ in range(120):
                rig.state.clear_forces()
                solver.step(rig.state, rig.following, rig.control, None, 1 / 240)
                rig.state, rig.following = rig.following, rig.state
                self.assertLess(rig.rcm()[0], 1e-6)
            actual = rig.state.joint_q.numpy()[rig.tool.arm_q_indices[2]]
            self.assertAlmostEqual(actual, target, delta=1e-4)
            np.testing.assert_array_equal(rig.tool.trocar_pose, pivot)
            np.testing.assert_allclose(rig.state.body_q.numpy()[rig.needle], needle, atol=1e-7)
            self.assertTrue(np.isfinite(rig.state.body_q.numpy()).all())

    def test_invalid_commands_do_not_mutate_actuators_or_allow_too_short_shafts(self):
        rig = _Rig()
        before = rig.control.joint_target_q.numpy().copy()
        for command in (np.zeros(7), [0, 0, 0.031, 0, 0, 0], [0, 0, 0, np.nan, 0, 0]):
            with self.subTest(command=command), self.assertRaises(ValueError):
                rig.tool.set_arm_targets(rig.control, command)
        np.testing.assert_array_equal(rig.control.joint_target_q.numpy(), before)
        with self.assertRaises(ValueError):
            add_rcm_grasper(newton.ModelBuilder(), config=GrasperConfig(shaft_length_m=0.1))

    def test_explicit_source_scale_trocar_and_reset_ik_coordinates_match_native_fk(self):
        rig = _Rig(
            config=GrasperConfig(shaft_length_m=0.24), nominal_insertion_m=0.18, insertion_limits_m=(-0.05, 0.05)
        )
        self.assertEqual(rig.tool.nominal_insertion_m, 0.18)
        self.assertEqual(rig.tool.insertion_limits_m, (-0.05, 0.05))
        rotation = np.asarray(wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 2.7), dtype=float)
        anchor = np.r_[0.116, -0.037, 0.19, rotation]
        with self.assertRaises(RuntimeError):
            rig.tool.set_trocar_pose(rig.model, anchor)
        rig.tool.set_trocar_pose(rig.model, anchor, reset=True)
        positions = np.asarray([0.2, -0.15, 0.03, 0.25, -0.6, 0.1])
        rig.tool.initialize(rig.model, rig.state, arm_positions=positions)
        rig.tool.set_arm_targets(rig.control, positions)
        np.testing.assert_array_equal(rig.state.joint_q.numpy()[:7], [1, 2, 3, 0, 0, 0, 1])
        np.testing.assert_allclose(rig.tool.trocar_pose, anchor, atol=1e-7)
        fixed_anchor = rig.tool.trocar_pose.copy()
        self.assertLess(rig.rcm()[0], 1e-7)
        self.assertAlmostEqual(rig.rcm()[1], -0.21, delta=1e-7)
        palm = rig.state.body_q.numpy()[rig.tool.palm_body].copy()
        rig.tool.initialize(rig.model, rig.state, palm, arm_positions=positions)
        before = rig.state.joint_q.numpy().copy()
        wrong_palm = palm.copy()
        wrong_palm[0] += 0.001
        with self.assertRaises(ValueError):
            rig.tool.initialize(rig.model, rig.state, wrong_palm, arm_positions=positions)
        for invalid in ([0, 0, 0.051, 0, 0, 0], np.zeros(5), [0, 0, np.nan, 0, 0, 0]):
            with self.subTest(positions=invalid), self.assertRaises(ValueError):
                rig.tool.initialize(rig.model, rig.state, arm_positions=invalid)
        np.testing.assert_array_equal(rig.state.joint_q.numpy(), before)

        # Independently evaluate the compiled native chain at the same joint
        # coordinates; do not assume a posed hand has no internal contacts.
        solver = rig.native_solver()
        native_state = mujoco.MjData(solver.mj_model)
        native_state.qpos[:] = rig.state.joint_q.numpy()
        native_state.qpos[3:7] = (1, 0, 0, 0)  # native free-joint quaternion is wxyz
        mujoco.mj_forward(solver.mj_model, native_state)
        palm_id = mujoco.mj_name2id(solver.mj_model, mujoco.mjtObj.mjOBJ_BODY, "PSM1_wrist_yaw")
        self.assertGreater(palm_id, 0)
        np.testing.assert_allclose(native_state.xpos[palm_id], palm[:3], atol=2e-7)
        native_rotation = native_state.xquat[palm_id][[1, 2, 3, 0]]
        self.assertAlmostEqual(abs(np.dot(native_rotation, palm[3:])), 1, delta=2e-7)
        np.testing.assert_array_equal(rig.tool.trocar_pose, fixed_anchor)
        self.assertLess(rig.rcm()[0], 1e-6)

    def test_custom_insertion_range_keeps_entire_travel_on_physical_shaft(self):
        rig = _Rig(
            config=GrasperConfig(shaft_length_m=0.24), nominal_insertion_m=0.18, insertion_limits_m=(-0.05, 0.04)
        )
        self.assertEqual(rig.tool.arm_lower[2], -0.05)
        self.assertEqual(rig.tool.arm_upper[2], 0.04)
        for insertion in (-0.05, 0.04):
            positions = np.asarray([0.4, -0.35, insertion, 0.7, -0.4, 0.3])
            rig.tool.initialize(rig.model, rig.state, arm_positions=positions)
            error, axial = rig.rcm()
            self.assertLess(error, 1e-7)
            self.assertAlmostEqual(axial, -0.18 - insertion, delta=2e-7)
            self.assertGreaterEqual(axial, -0.243)
            self.assertLess(axial, -0.003)

    def test_invalid_insertion_geometry_is_rejected_before_building_links(self):
        invalid_options = (
            {"nominal_insertion_m": 0},
            {"nominal_insertion_m": -0.1},
            {"nominal_insertion_m": np.nan},
            {"nominal_insertion_m": np.inf},
            {"insertion_limits_m": (0.01, 0.03)},
            {"insertion_limits_m": (0.03, -0.03)},
            {"insertion_limits_m": (0, 0)},
            {"insertion_limits_m": (-0.03, np.inf)},
            {"insertion_limits_m": (-0.03,)},
            {"nominal_insertion_m": 0.032, "insertion_limits_m": (-0.03, 0.03)},
            {"nominal_insertion_m": 0.1, "insertion_limits_m": (-0.03, 0.03)},
        )
        for options in invalid_options:
            with self.subTest(options=options):
                builder = newton.ModelBuilder()
                with self.assertRaises(ValueError):
                    add_rcm_grasper(builder, **options)
                self.assertEqual(builder.body_count, 0)


if __name__ == "__main__":
    unittest.main()
