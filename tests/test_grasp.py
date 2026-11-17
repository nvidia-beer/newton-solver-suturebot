# SPDX-License-Identifier: Apache-2.0
"""Actual native MuJoCo contacts must carry, slip and release a free needle."""

import os
import unittest

import mujoco
import newton
import numpy as np
import warp as wp

from newton_solver_suturebot._src.geometry.needle import NeedleGeometry
from newton_solver_suturebot._src.solvers.mujoco.subcycled_mujoco import SubcycledMuJoCo
from newton_solver_suturebot.examples.grasp._dvrk_capsules import dvrk_tool_capsules
from newton_solver_suturebot.examples.grasp._dvrk_grasper import (
    GrasperConfig,
    add_grasper,
    jaw_opening_on_needle,
    needle_to_gripper_pose,
)
from newton_solver_suturebot.examples.needle._needle_insertion_drive import NeedleDrive


class _Rig:
    def __init__(self, *, gravity=True, second_tool=False, config=None, solver_class=SubcycledMuJoCo, particles=False):
        self.geometry = NeedleGeometry.load()
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_usd(str(self.geometry.asset_path), collapse_fixed_joints=False, hide_collision_shapes=True)
        self.tool = add_grasper(builder, label="PSM1", config=config)
        self.parked = add_grasper(builder, label="PSM2", config=config) if second_tool else None
        if particles:
            builder.add_particle(pos=(0.1, 0.2, 0.3), vel=(0.4, 0.5, 0.6), mass=0.001)
        self.model = builder.finalize()
        self.state = self.model.state()
        self.following = self.model.state()
        self.control = self.model.control()
        self.dt = 1 / 240
        needle_pose = np.r_[0, 0, 0.04, np.asarray(wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2))]
        coordinates = self.state.joint_q.numpy()
        coordinates[:7] = needle_pose
        self.state.joint_q.assign(coordinates)
        relative = needle_to_gripper_pose(self.geometry)
        self.target = np.asarray(
            wp.transform_multiply(wp.transform(*needle_pose), wp.transform(*relative)),
            dtype=float,
        )
        # The jaws start about the needle at the opening at which their capsules touch it.
        needle_in_palm = wp.transform_inverse(wp.transform(*relative))
        self.grasp_opening = jaw_opening_on_needle(
            self.tool,
            [np.asarray(wp.transform_point(needle_in_palm, wp.vec3(*point))) for point in self.geometry.centerline],
            self.geometry.thickness,
        )
        self.tool.initialize(self.model, self.state, self.target, opening_rad=self.grasp_opening)
        if self.parked:
            self.parked_target = self.target.copy()
            self.parked_target[0] += 0.1
            self.parked.initialize(self.model, self.state, self.parked_target)
        self.solver = solver_class(
            self.model,
            use_mujoco_cpu=True,
            integrator="implicitfast",
            iterations=200,
            ls_iterations=50,
            tolerance=1e-12,
            cone="elliptic",
            kinematic_switchable_roots=False,
            impratio=10,
        )
        self.solver.reset(self.state, flags=0)
        self.drive = NeedleDrive(self.model, self.tool.palm_body, stiffness=1000, angular_stiffness=1)
        self.drive.configure_mujoco(self.solver, self.dt)
        self.tool.set_aperture(self.control, 0)
        if self.parked:
            self.parked_drive = NeedleDrive(self.model, self.parked.palm_body, stiffness=1000, angular_stiffness=1)
            self.parked_drive.configure_mujoco(self.solver, self.dt)
            self.parked.set_aperture(self.control, 0.3)
        self.gravity = gravity
        self.external = np.zeros((self.model.body_count, 6), dtype=np.float32)
        if gravity:
            masses = self.model.body_mass.numpy()
            self.external[:, 2] = -9.81 * masses
            for tool in (self.tool, self.parked):
                if tool is not None:
                    # Actuated support balances tool weight only. The needle's
                    # weight must be carried by its contact with the jaws.
                    self.external[tool.palm_body, 2] += 9.81 * masses[[tool.palm_body, *tool.jaw_bodies]].sum()
        self.maximum_effort = 0.0

    def step(self, *, target=None, needle_force=None):
        self.state.clear_forces()
        applied = self.external.copy()
        if needle_force is not None:
            applied[0, :3] += needle_force
        self.state.body_f.assign(applied)
        self.drive.add_forces(self.state, self.target if target is None else target, np.zeros(6))
        if self.parked:
            self.parked_drive.add_forces(self.state, self.parked_target, np.zeros(6))
        self.solver.step(self.state, self.following, self.control, None, self.dt)
        self.state, self.following = self.following, self.state
        self.maximum_effort = max(self.maximum_effort, float(np.abs(self.solver.mj_data.actuator_force).max()))

    def observe(self):
        return self.tool.observe(self.model, self.state, self.solver, 0)


class ContactGraspTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.device.__enter__()

    def tearDown(self):
        self.device.__exit__(None, None, None)

    def test_topology_capsules_and_reset_keep_needle_independent(self):
        rig = _Rig(second_tool=True)
        self.assertEqual(rig.solver.mj_model.neq, 0)
        self.assertEqual(rig.model.body_count, 7)
        np.testing.assert_array_equal(rig.model.joint_type.numpy(), [4, 4, 1, 1, 4, 1, 1])
        self.assertEqual(rig.solver.mj_model.nv, 22)
        self.assertEqual(rig.solver.mj_model.nu, 4)
        before = rig.state.body_q.numpy()[0].copy()
        target = rig.target.copy()
        target[1] += 0.001
        rig.tool.initialize(rig.model, rig.state, target, opening_rad=0.4)
        np.testing.assert_array_equal(rig.state.body_q.numpy()[0], before)
        np.testing.assert_allclose(rig.state.joint_q.numpy()[list(rig.tool.jaw_q_indices)], [0.2, -0.2])
        shape_local = rig.model.shape_transform.numpy()
        shape_scale = rig.model.shape_scale.numpy()
        native_shapes = rig.solver.mjc_geom_to_newton_shape.numpy()[0]
        for row, shape in enumerate(rig.tool.capsule_shapes):
            transform = wp.transform(*shape_local[shape])
            length = shape_scale[shape, 1]
            endpoints = [np.asarray(wp.transform_point(transform, wp.vec3(0, 0, side * length))) for side in (-1, 1)]
            np.testing.assert_allclose(endpoints, rig.tool.capsule_endpoints[row], atol=6e-8)
            self.assertAlmostEqual(shape_scale[shape, 0], rig.tool.capsule_radii[row], delta=1e-9)
            native = np.flatnonzero(native_shapes == shape)
            self.assertEqual(len(native), 1)
            geom = int(native[0])
            self.assertEqual(rig.solver.mj_model.geom_type[geom], mujoco.mjtGeom.mjGEOM_CAPSULE)
            np.testing.assert_allclose(rig.solver.mj_model.geom_size[geom, :2], shape_scale[shape, :2], atol=1e-9)
            center = rig.solver.mj_model.geom_pos[geom]
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, rig.solver.mj_model.geom_quat[geom])
            axis = rotation.reshape(3, 3)[:, 2]
            native_endpoints = [center + side * length * axis for side in (-1, 1)]
            np.testing.assert_allclose(native_endpoints, rig.tool.capsule_endpoints[row], atol=6e-8)
        # The rows are the capsule set of the instrument; shaft and wrist lie on the palm as the CAD assembles them.
        capsules = dvrk_tool_capsules("PSM1", rig.tool.config.shaft_length_m)
        self.assertEqual(len(rig.tool.capsule_shapes), len(capsules.radii))
        np.testing.assert_array_equal(rig.tool.capsule_radii, capsules.radii)
        for row, part in enumerate(capsules.parts):
            expected = capsules.endpoints[row]
            if part in ("roll", "pitch"):
                frame = capsules.frames[part]
                expected = expected @ frame[:3, :3].T + frame[:3, 3]
            np.testing.assert_allclose(rig.tool.capsule_endpoints[row], expected, rtol=0, atol=1e-12)
        self.assertEqual(rig.tool.capsule_parts[0], "shaft")
        # The shaft lies along the palm's -y axis, behind the wrist.
        np.testing.assert_allclose(rig.tool.capsule_endpoints[0][:, [0, 2]], 0, rtol=0, atol=1e-6)
        self.assertLess(rig.tool.capsule_endpoints[0, 0, 1], rig.tool.capsule_endpoints[0, 1, 1])
        self.assertLess(rig.tool.capsule_endpoints[0, 1, 1], 0)
        # Correct q-indexing must not overwrite either free-joint quaternion.
        before = rig.control.joint_target_q.numpy().copy()
        rig.tool.set_aperture(rig.control, 0.5)
        after = rig.control.joint_target_q.numpy()
        mask = np.ones(len(after), dtype=bool)
        mask[list(rig.tool.jaw_q_indices)] = False
        np.testing.assert_array_equal(after[mask], before[mask])
        np.testing.assert_allclose(after[list(rig.tool.jaw_q_indices)], [0.325, -0.325])

    def test_configurable_shaft_preserves_wrist_jaws_and_distal_endpoint(self):
        default = _Rig()
        default_length = default.tool.config.shaft_length_m
        default_shaft = np.linalg.norm(np.diff(default.tool.capsule_endpoints[0], axis=0))
        for length in (0.08, 0.547):
            with self.subTest(length=length):
                rig = _Rig(config=GrasperConfig(shaft_length_m=length))
                # Wrist and jaws do not depend on the length of the shaft, which keeps its radius and its distal end.
                self.assertEqual(rig.tool.capsule_parts, default.tool.capsule_parts)
                np.testing.assert_array_equal(rig.tool.capsule_endpoints[1:], default.tool.capsule_endpoints[1:])
                np.testing.assert_array_equal(rig.tool.capsule_radii[1:], default.tool.capsule_radii[1:])
                self.assertAlmostEqual(rig.tool.capsule_radii[0], default.tool.capsule_radii[0], delta=1e-12)
                np.testing.assert_allclose(
                    rig.tool.capsule_endpoints[0, 1], default.tool.capsule_endpoints[0, 1], rtol=0, atol=1e-7
                )
                shaft = np.linalg.norm(np.diff(rig.tool.capsule_endpoints[0], axis=0))
                self.assertAlmostEqual(shaft - default_shaft, length - default_length, delta=1e-6)
                native_shapes = rig.solver.mjc_geom_to_newton_shape.numpy()[0]
                geom = int(np.flatnonzero(native_shapes == rig.tool.capsule_shapes[0])[0])
                self.assertAlmostEqual(rig.solver.mj_model.geom_size[geom, 0], rig.tool.capsule_radii[0], delta=1e-9)
                self.assertAlmostEqual(rig.solver.mj_model.geom_size[geom, 1], shaft / 2, delta=3e-8)

    def test_contact_grips_under_gravity_from_first_step_and_transports(self):
        rig = _Rig(second_tool=True)
        initial = rig.state.body_q.numpy()[0].copy()
        for _ in range(120):
            rig.step()
        grasp = rig.observe()
        self.assertTrue(np.all(grasp["jaw_contact_normal_force_n"] > 0.1), grasp)
        self.assertTrue(np.all(grasp["jaw_needle_contact_count"] > 0))
        self.assertLess(np.linalg.norm(rig.state.body_q.numpy()[0, :3] - initial[:3]), 0.0003)
        self.assertGreater(abs(grasp["jaw_position_rad"][0]), 0.01)
        self.assertLess(rig.maximum_effort, rig.tool.config.jaw_torque_limit_nm + 1e-7)
        relative = grasp["needle_relative_pose"]
        needle = rig.state.body_q.numpy()[0].copy()
        for index in range(240):
            target = rig.target.copy()
            u = min(index / 120, 1)
            ramp = 3 * u**2 - 2 * u**3
            target[0] += 0.005 * ramp
            target[3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.1 * ramp) * wp.quat(*rig.target[3:]))
            rig.step(target=target)
        final = rig.observe()
        displacement = rig.state.body_q.numpy()[0, :3] - needle[:3]
        self.assertGreater(displacement[0], 0.0045)
        self.assertLess(np.linalg.norm(final["needle_relative_pose"][:3] - relative[:3]), 0.0003)
        before_quaternion = relative[3:] / np.linalg.norm(relative[3:])
        after_quaternion = final["needle_relative_pose"][3:]
        after_quaternion /= np.linalg.norm(after_quaternion)
        relative_angle = 2 * np.arccos(np.clip(abs(np.dot(before_quaternion, after_quaternion)), 0, 1))
        self.assertLess(relative_angle, 0.03)
        self.assertTrue(np.all(np.isfinite(rig.state.body_qd.numpy())))
        # This is a loaded contact, with a measurable finite relative error,
        # rather than an exactly copied/welded needle transform.
        self.assertGreater(np.linalg.norm(final["needle_relative_pose"][:3] - relative[:3]), 1e-8)

    def test_opening_releases_needle_and_gravity_continues_independently(self):
        rig = _Rig()
        for _ in range(120):
            rig.step()
        self.assertTrue(np.all(rig.observe()["jaw_contact_normal_force_n"] > 0.1))
        rig.tool.set_aperture(rig.control, 1)
        before = rig.state.body_q.numpy()[0, :3].copy()
        for _ in range(120):
            rig.step()
        after = rig.observe()
        self.assertGreater(after["jaw_position_rad"][0], 0.6)
        self.assertLess(after["jaw_position_rad"][1], -0.6)
        np.testing.assert_array_equal(after["jaw_contact_normal_force_n"], 0)
        self.assertLess(rig.state.body_q.numpy()[0, 2], before[2] - 0.5)
        self.assertLess(np.linalg.norm(after["gripper_pose"][:3] - rig.target[:3]), 0.0003)

    def test_sufficient_external_load_slips_instead_of_enforcing_grasp(self):
        rig = _Rig(gravity=False)
        for _ in range(120):
            rig.step()
        relative = rig.observe()["needle_relative_pose"][:3].copy()
        direction = np.asarray(wp.quat_rotate(wp.quat(*rig.target[3:]), wp.vec3(0, 0, 1)))
        for _ in range(24):
            rig.step(needle_force=1.5 * direction)
        after = rig.observe()
        self.assertGreater(np.linalg.norm(after["needle_relative_pose"][:3] - relative), 0.005)
        self.assertTrue(np.all(np.isfinite(rig.state.body_q.numpy())))
        self.assertLess(rig.maximum_effort, rig.tool.config.jaw_torque_limit_nm + 1e-7)

    def test_invalid_commands_and_parameters_do_not_mutate_controls(self):
        rig = _Rig(gravity=False)
        before = rig.control.joint_target_q.numpy().copy()
        for invalid in (-0.01, 1.01, np.nan, np.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                rig.tool.set_aperture(rig.control, invalid)
        np.testing.assert_array_equal(rig.control.joint_target_q.numpy(), before)
        for kwargs in (
            {"contact_impedance": 1},
            {"jaw_torque_limit_nm": 0},
            {"initial_opening_rad": 2},
            *({"shaft_length_m": length} for length in (-0.1, 0, 0.548, np.nan, np.inf)),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GrasperConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
