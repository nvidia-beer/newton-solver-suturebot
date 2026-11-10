# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compact transfers preserve Newton/MuJoCo frames, loads and native history."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo

from newton_solver_suturebot._src.mujoco_free_body import MuJoCoFreeBody


def _native_state(solver):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    value = np.empty(mujoco.mj_stateSize(solver.mj_model, spec))
    mujoco.mj_getState(solver.mj_model, solver.mj_data, value, spec)
    return value


class MuJoCoFreeBodyTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))

    def build(self, *, contacts=False, com=(0, 0, 0), mass=0.0003):
        with wp.ScopedDevice(self.device):
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            rotation = np.asarray(
                wp.quat_to_matrix(wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, 2, -1)), 0.7))
            ).reshape(3, 3)
            inertia = rotation @ np.diag([1e-8, 1.4e-8, 2e-8]) @ rotation.T
            body = builder.add_body(
                mass=mass, inertia=wp.mat33(*inertia.reshape(-1)), com=wp.vec3(*com), lock_inertia=True
            )
            if contacts:
                builder.add_shape_sphere(body, radius=0.005)
                builder.add_ground_plane()
            model = builder.finalize()
            solvers = [
                SolverMuJoCo(model, use_mujoco_cpu=True, integrator="implicitfast", update_data_interval=1)
                for _ in range(2)
            ]
            return model, solvers

    def state(self, model, pose, velocity, wrench):
        state = model.state()
        state.joint_q.assign(np.asarray(pose, np.float32))
        state.joint_qd.assign(np.asarray(velocity, np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        state.body_f.assign(np.asarray([wrench], np.float32))
        return state

    def assert_states_close(self, left, right):
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            np.testing.assert_allclose(
                getattr(left, name).numpy(), getattr(right, name).numpy(), rtol=2e-6, atol=2e-7, err_msg=name
            )

    def test_random_rotated_pose_twist_loads_native_damping_and_rollback(self):
        rng = np.random.default_rng(918)
        for mass in (0.0003, 0.002):
            model, (general, compact_solver) = self.build(mass=mass)
            bridge = MuJoCoFreeBody(model, compact_solver)
            self.assertTrue(MuJoCoFreeBody.supports(model, general))
            self.assertGreater(
                np.max(np.abs(model.body_inertia.numpy()[0] - np.diag(np.diag(model.body_inertia.numpy()[0])))), 1e-10
            )
            for iteration in range(12):
                quat = rng.normal(size=4)
                quat /= np.linalg.norm(quat)
                pose = np.r_[rng.uniform(-0.05, 0.05, 3), quat]
                velocity = rng.normal(size=6) * np.array([0.02] * 3 + [2] * 3)
                wrench = rng.normal(size=6) * np.array([0.002] * 3 + [1e-6] * 3)
                state = self.state(model, pose, velocity, wrench)
                control = model.control()
                joint_force = rng.normal(size=6) * np.array([0.001] * 3 + [5e-7] * 3)
                control.joint_f.assign(joint_force.astype(np.float32))
                damping = rng.uniform(0, 1, 6) * np.array([0.01] * 3 + [1e-7] * 3)
                warmstart = rng.normal(size=6)
                for solver in (general, compact_solver):
                    solver.reset(state, flags=0)
                    solver.mj_model.dof_damping[:] = damping
                    solver.mj_data.qacc_warmstart[:] = warmstart
                    solver.mj_data.time = 0.125
                snapshot = _native_state(general)
                dt = (1 / 720, 1 / 240, 1 / 60)[iteration % 3]
                # Repeated restored trials must not retain a rejected dt/load.
                for trial_dt in (dt, dt / 2, dt):
                    for solver in (general, compact_solver):
                        mujoco.mj_setState(
                            solver.mj_model, solver.mj_data, snapshot, mujoco.mjtState.mjSTATE_INTEGRATION
                        )
                    expected, actual = model.state(), model.state()
                    general.step(state, expected, control, None, trial_dt)
                    bridge.step(state, actual, control, trial_dt)
                    self.assert_states_close(actual, expected)
                    np.testing.assert_allclose(
                        _native_state(compact_solver), _native_state(general), rtol=3e-6, atol=2e-7
                    )
                    np.testing.assert_array_equal(compact_solver.mj_data.xfrc_applied, general.mj_data.xfrc_applied)
                    np.testing.assert_array_equal(compact_solver.mj_data.qfrc_applied, general.mj_data.qfrc_applied)
                    self.assertAlmostEqual(compact_solver.mj_data.time, 0.125 + trial_dt)
                    np.testing.assert_array_equal(state.body_f.numpy()[0], wrench.astype(np.float32))

    def test_none_control_and_native_contacts_remain_native(self):
        model, (general, compact_solver) = self.build(contacts=True)
        bridge = MuJoCoFreeBody(model, compact_solver)
        pose = np.array([0, 0, 0.0049, 0.2, -0.1, 0.3, 0.9])
        pose[3:] /= np.linalg.norm(pose[3:])
        states = [
            self.state(model, pose, [0.01, -0.02, -0.01, 1, -2, 0.5], [0.001, 0, 0, 0, 1e-7, 0]) for _ in range(2)
        ]
        for solver, state in zip((general, compact_solver), states, strict=True):
            solver.reset(state, flags=0)
        for _ in range(6):
            outputs = [model.state(), model.state()]
            general.step(states[0], outputs[0], None, None, 1e-4)
            bridge.step(states[1], outputs[1], None, 1e-4)
            self.assert_states_close(outputs[1], outputs[0])
            self.assertGreater(compact_solver.mj_data.ncon, 0)
            self.assertEqual(compact_solver.mj_data.ncon, general.mj_data.ncon)
            np.testing.assert_allclose(
                compact_solver.mj_data.qfrc_constraint, general.mj_data.qfrc_constraint, rtol=3e-5, atol=1e-8
            )
            states = outputs

    def test_guard_rejects_unsupported_sync_frames_and_custom_solver_step(self):
        model, (solver, _) = self.build()
        self.assertTrue(MuJoCoFreeBody.supports(model, solver))
        for interval in (0, 2):
            solver.update_data_interval = interval
            self.assertFalse(MuJoCoFreeBody.supports(model, solver))
        solver.update_data_interval = 1
        for name, changed in (
            ("body_com", [[0.001, 0, 0]]),
            ("gravity", [[0, 0, -9.81]]),
            ("joint_X_p", [[0.001, 0, 0, 0, 0, 0, 1]]),
            ("joint_X_c", [[0, 0, 0, 0, 0, 0.1, np.sqrt(0.99)]]),
        ):
            array = getattr(model, name)
            original = array.numpy()
            array.assign(np.asarray(changed, np.float32))
            self.assertFalse(MuJoCoFreeBody.supports(model, solver), name)
            array.assign(original)
        solver.mj_model.opt.gravity[:] = [0, 0, -1]
        self.assertFalse(MuJoCoFreeBody.supports(model, solver))
        solver.mj_model.opt.gravity[:] = 0
        with patch.object(solver, "step", lambda *_: None):
            self.assertFalse(MuJoCoFreeBody.supports(model, solver))

    def test_bundled_authored_needle_is_eligible_and_coupler_uses_expected_path(self):
        from test_needle_insertion_example import _example, _prepare

        with _example() as example:
            _prepare(example)
            example.start()
            self.assertTrue(MuJoCoFreeBody.supports(example.model, example.rigid_solver))
            self.assertEqual(example.coupler.rigid_transfer_mode, "compact" if self.device.is_cuda else "general")
            np.testing.assert_array_equal(example.model.body_com.numpy(), 0)
            self.assertEqual(example.rigid_solver.mj_model.nu, 0)

    def test_requested_legacy_observables_delegate_to_general_adapter(self):
        model, (solver, _) = self.build()
        bridge = MuJoCoFreeBody(model, solver)
        state = model.state()
        for name in ("body_qdd", "body_parent_f", "mujoco.qfrc_actuator"):
            following = model.state()
            if name.startswith("mujoco"):
                following.mujoco = SimpleNamespace(qfrc_actuator=wp.zeros(6, device=self.device))
            else:
                setattr(following, name, wp.zeros(1, dtype=wp.spatial_vector, device=self.device))
            with patch.object(solver, "step") as step:
                bridge.step(state, following, None, 0.001)
                step.assert_called_once_with(state, following, None, None, 0.001)

    def test_custom_step_installed_after_bridge_construction_is_respected(self):
        model, (solver, _) = self.build()
        bridge = MuJoCoFreeBody(model, solver)
        state, following = model.state(), model.state()
        with patch.object(solver, "step") as step:
            bridge.step(state, following, None, 0.001)
            step.assert_called_once_with(state, following, None, None, 0.001)


if __name__ == "__main__":
    unittest.main()
