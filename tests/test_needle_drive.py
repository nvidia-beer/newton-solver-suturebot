# SPDX-License-Identifier: Apache-2.0
"""Native damping preserves the PD law while avoiding explicit damping growth."""

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import mujoco
import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo

from newton_solver_suturebot.examples._needle_insertion_drive import NeedleDrive, ramp_distance


@contextmanager
def _body(*, com=(0, 0, 0)):
    with wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu")):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        body = builder.add_body(
            mass=0.0003,
            inertia=wp.mat33(1e-8, 0, 0, 0, 1e-8, 0, 0, 0, 2e-8),
            com=wp.vec3(*com),
            label="needle",
        )
        model = builder.finalize()
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, integrator="implicitfast", update_data_interval=1)
        yield model, solver, NeedleDrive(model, body)


def _state(model, solver, pose, velocity):
    state = model.state()
    state.body_q.assign(np.asarray([pose], dtype=np.float32))
    state.body_qd.assign(np.asarray([velocity], dtype=np.float32))
    newton.eval_ik(model, state, state.joint_q, state.joint_qd)
    solver.reset(state, flags=0)
    return state


def _pose(angle=0.0, axis=0, x=0.0):
    vector = np.eye(3)[axis]
    quat = np.asarray(wp.quat_from_axis_angle(wp.vec3(*vector), angle))
    return np.r_[x, 0, 0, quat]


class NeedleDriveTests(unittest.TestCase):
    def test_implicit_spring_matches_backward_euler_translation_and_principal_axes(self):
        with _body() as (model, solver, _):
            drive = NeedleDrive(model, 0, stiffness=400, angular_stiffness=0.05)
            for axis in (0, 3, 5):
                for dt in (1 / 240, 1 / 60):
                    with self.subTest(axis=axis, dt=dt):
                        angular = axis >= 3
                        coordinate = 0.001 if angular else 1e-5
                        velocity = 0.2 if angular else 0.002
                        target_coordinate = -0.002 if angular else -1e-5
                        target_velocity = 0.1 if angular else 0.001
                        pose = _pose(coordinate, axis=axis - 3) if angular else _pose(x=coordinate)
                        target = _pose(target_coordinate, axis=axis - 3) if angular else _pose(x=target_coordinate)
                        twist, target_twist = np.zeros(6), np.zeros(6)
                        twist[axis], target_twist[axis] = velocity, target_velocity
                        state = _state(model, solver, pose, twist)
                        drive.configure_mujoco(solver, dt)
                        following = model.state()
                        inertia = (
                            model.body_inertia.numpy()[0, axis - 3, axis - 3] if angular else model.body_mass.numpy()[0]
                        )
                        stiffness = drive.angular_stiffness if angular else drive.stiffness
                        damping = drive.angular_damping if angular else drive.damping
                        expected_velocity = (
                            inertia * velocity
                            + dt * (stiffness * (target_coordinate - coordinate) + damping * target_velocity)
                        ) / (inertia + dt * damping + dt * dt * stiffness)
                        expected_coordinate = coordinate + dt * expected_velocity
                        drive.add_forces(state, target, target_twist)
                        solver.step(state, following, model.control(), None, dt)
                        self.assertAlmostEqual(
                            following.body_qd.numpy()[0, axis], expected_velocity, delta=2e-7 if angular else 2e-9
                        )
                        actual = following.body_q.numpy()[0]
                        actual_coordinate = 2 * np.arctan2(actual[axis], actual[6]) if angular else actual[0]
                        self.assertAlmostEqual(actual_coordinate, expected_coordinate, delta=2e-9)
                        expected_wrench = stiffness * (target_coordinate - expected_coordinate)
                        expected_wrench += damping * (target_velocity - expected_velocity)
                        self.assertAlmostEqual(
                            drive.net_wrench(following)[axis], expected_wrench, delta=2e-10 if angular else 2e-9
                        )

    def test_timestep_changes_update_native_spring_once_and_preserve_passive_damping(self):
        with _body() as (model, solver, drive):
            passive = np.array((0.01, 0.02, 0.03, 1e-7, 2e-7, 3e-7), np.float32)
            model.joint_damping.assign(passive)
            dt = 1 / 240
            drive.configure_mujoco(solver, dt)
            with patch.object(solver, "notify_model_changed", wraps=solver.notify_model_changed) as notify:
                drive.prepare_timestep(dt)
                drive.prepare_timestep(dt)
                self.assertEqual(notify.call_count, 0)
                for value in (dt / 2, dt):
                    drive.prepare_timestep(value)
                    drive.prepare_timestep(value)
                    expected = passive + np.array(
                        [drive.damping + value * drive.stiffness] * 3
                        + [drive.angular_damping + value * drive.angular_stiffness] * 3,
                        np.float32,
                    )
                    np.testing.assert_allclose(solver.mj_model.dof_damping, expected, rtol=2e-7, atol=1e-12)
                    self.assertEqual(drive.spring_linearization_dt, value)
                self.assertEqual(notify.call_count, 2)
                previous = solver.mj_model.dof_damping.copy()
                for invalid in (0, -1, np.nan, np.inf, None):
                    with self.assertRaises(ValueError):
                        drive.prepare_timestep(invalid)
                np.testing.assert_array_equal(solver.mj_model.dof_damping, previous)
                self.assertEqual(notify.call_count, 2)

    def test_general_rotation_spring_linearization_converges_under_timestep_refinement(self):
        poses = []
        with _body() as (model, solver, drive):
            axis = np.array((0.3, -0.4, 0.5))
            axis /= np.linalg.norm(axis)
            target = np.r_[0, 0, 0, np.asarray(wp.quat_from_axis_angle(wp.vec3(*axis), 0.5))]
            for subdivisions in (4, 8, 16, 256):
                dt = (1 / 60) / subdivisions
                state = _state(model, solver, _pose(), (0, 0, 0, 0.3, -0.2, 0.1))
                following, control = model.state(), model.control()
                drive.configure_mujoco(solver, dt)
                for _ in range(subdivisions):
                    state.clear_forces()
                    drive.add_forces(state, target, np.zeros(6))
                    solver.step(state, following, control, None, dt)
                    state, following = following, state
                poses.append(state.body_q.numpy()[0, 3:].astype(float))
        # This is a consistency check, not an assertion that the scalar
        # rotational linearization equals the finite-rotation endpoint law.
        errors = [min(np.linalg.norm(q - poses[-1]), np.linalg.norm(q + poses[-1])) for q in poses[:-1]]
        self.assertGreater(errors[0], 1e-5)
        self.assertLess(errors[1], 0.8 * errors[0])
        self.assertLess(errors[2], 0.8 * errors[1])

    def test_split_external_and_native_damping_recover_the_same_pd_wrench(self):
        with _body() as (model, solver, drive):
            state = _state(model, solver, _pose(0.4, axis=1), (0.001, -0.002, 0.003, 0.2, -0.3, 0.1))
            target = _pose(0.43, axis=1, x=0.0001)
            target_velocity = np.array((0.002, 0.001, -0.001, -0.1, 0.2, 0.15))
            drive.add_forces(state, target, target_velocity)
            explicit = drive.wrench.numpy()[0].copy()
            state.clear_forces()
            drive.configure_mujoco(solver)
            drive.add_forces(state, target, target_velocity)
            np.testing.assert_allclose(drive.net_wrench(state), explicit, rtol=3e-6, atol=2e-10)
            expected_passive = -np.array([drive.damping] * 3 + [drive.angular_damping] * 3)
            expected_passive *= state.body_qd.numpy()[0]
            np.testing.assert_allclose(drive.wrench.numpy()[0] + expected_passive, explicit, rtol=3e-6, atol=2e-10)
            self.assertTrue(drive.implicit_damping)
            # Native evaluation after a tiny physical step verifies angular
            # damping uses the rotated body basis, while its world wrench agrees.
            following = model.state()
            solver.step(state, following, model.control(), None, 1e-6)
            mujoco.mj_forward(solver.mj_model, solver.mj_data)
            passive = solver.mj_data.qfrc_passive.copy()
            rotation = wp.quat(*following.body_q.numpy()[0, 3:])
            passive[3:] = np.asarray(wp.quat_rotate(rotation, wp.vec3(*passive[3:])))
            coefficients = np.array([drive.damping] * 3 + [drive.angular_damping] * 3, dtype=np.float32)
            expected_passive = -coefficients * following.body_qd.numpy()[0]
            np.testing.assert_allclose(passive, expected_passive, rtol=3e-6, atol=2e-10)

    def test_single_free_body_matches_implicit_damping_discrete_solution(self):
        with _body() as (model, solver, drive):
            drive.configure_mujoco(solver)
            for axis in (0, 3, 5):
                for dt in (1 / 240, 1 / 120):
                    with self.subTest(axis=axis, dt=dt):
                        angular = axis >= 3
                        coordinate = 0.001 if angular else 1e-5
                        velocity = 0.2 if angular else 0.002
                        target_coordinate = -0.002 if angular else -1e-5
                        target_velocity = 0.1 if angular else 0.001
                        pose = _pose(coordinate, axis=axis - 3) if angular else _pose(x=coordinate)
                        target = _pose(target_coordinate, axis=axis - 3) if angular else _pose(x=target_coordinate)
                        twist, target_twist = np.zeros(6), np.zeros(6)
                        twist[axis], target_twist[axis] = velocity, target_velocity
                        state = _state(model, solver, pose, twist)
                        following = model.state()
                        inertia = (
                            model.body_inertia.numpy()[0, axis - 3, axis - 3] if angular else model.body_mass.numpy()[0]
                        )
                        stiffness = drive.angular_stiffness if angular else drive.stiffness
                        damping = drive.angular_damping if angular else drive.damping
                        expected_velocity = (
                            inertia * velocity
                            + dt * (stiffness * (target_coordinate - coordinate) + damping * target_velocity)
                        ) / (inertia + dt * damping)
                        expected_coordinate = coordinate + dt * expected_velocity
                        drive.add_forces(state, target, target_twist)
                        solver.step(state, following, model.control(), None, dt)
                        self.assertAlmostEqual(
                            following.body_qd.numpy()[0, axis], expected_velocity, delta=2e-7 if angular else 2e-9
                        )
                        actual_pose = following.body_q.numpy()[0]
                        actual_coordinate = (
                            2 * np.arctan2(actual_pose[axis], actual_pose[6]) if angular else actual_pose[0]
                        )
                        self.assertAlmostEqual(actual_coordinate, expected_coordinate, delta=2e-9)

    def test_explicit_damping_injects_energy_at_large_step_while_native_damping_dissipates(self):
        energies = {}
        for implicit in (False, True):
            with _body() as (model, solver, drive):
                if implicit:
                    drive.configure_mujoco(solver)
                state = _state(model, solver, _pose(), (0, 0, 0, 1, 0, 0))
                following = model.state()
                drive.add_forces(state, _pose(), np.zeros(6))
                solver.step(state, following, model.control(), None, 1 / 240)
                rotation = following.body_q.numpy()[0, 3:]
                angle = 2 * np.arctan2(rotation[0], rotation[3])
                inertia = float(model.body_inertia.numpy()[0, 0, 0])
                velocity = float(following.body_qd.numpy()[0, 3])
                energies[implicit] = 0.5 * inertia * velocity**2 + 0.5 * drive.angular_stiffness * angle**2
                initial = 0.5 * inertia
                if implicit:
                    self.assertGreater(velocity, 0)
        self.assertGreater(energies[False], initial)
        self.assertLess(energies[True], initial)

    def test_configuration_preserves_existing_damping_and_survives_reset_and_notification(self):
        with _body() as (model, solver, drive):
            passive = np.array((0.01, 0.02, 0.03, 1e-7, 2e-7, 3e-7), dtype=np.float32)
            model.joint_damping.assign(passive)
            drive.configure_mujoco(solver)
            expected = passive + np.array([drive.damping] * 3 + [drive.angular_damping] * 3, dtype=np.float32)
            np.testing.assert_array_equal(model.joint_damping.numpy(), expected)
            np.testing.assert_array_equal(solver.mj_model.dof_damping, expected)
            drive.configure_mujoco(solver)
            np.testing.assert_array_equal(model.joint_damping.numpy(), expected)
            solver.reset(model.state())
            solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
            np.testing.assert_array_equal(solver.mj_model.dof_damping, expected)

    def test_nonzero_com_is_rejected_before_any_damping_or_mode_change(self):
        with _body(com=(0.001, 0, 0)) as (model, solver, drive):
            before = model.joint_damping.numpy().copy()
            native = solver.mj_model.dof_damping.copy()
            with self.assertRaisesRegex(ValueError, "origin at COM"):
                drive.configure_mujoco(solver)
            self.assertFalse(drive.implicit_damping)
            np.testing.assert_array_equal(model.joint_damping.numpy(), before)
            np.testing.assert_array_equal(solver.mj_model.dof_damping, native)

    def test_explicit_integrator_cannot_silently_enable_implicit_drive_mode(self):
        with _body() as (model, solver, drive):
            solver.mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
            before = model.joint_damping.numpy().copy()
            with self.assertRaisesRegex(ValueError, "implicitfast"):
                drive.configure_mujoco(solver)
            self.assertFalse(drive.implicit_damping)
            np.testing.assert_array_equal(model.joint_damping.numpy(), before)

    def test_ramp_distance_still_supports_independent_vector_axes(self):
        times = np.array((0.02, 0.15))
        speeds = np.array((0.01, 0.003))
        expected = [ramp_distance(t, speed, 0.1) for t, speed in zip(times, speeds, strict=True)]
        np.testing.assert_array_equal(ramp_distance(times, speeds, 0.1), expected)


if __name__ == "__main__":
    unittest.main()
