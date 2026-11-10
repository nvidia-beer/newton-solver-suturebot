# SPDX-License-Identifier: Apache-2.0
"""The graphable free-body trial follows native implicitfast without committing it."""

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
from newton_solver_suturebot._src.mujoco_free_body_gpu import MuJoCoFreeBodyGPU

_STATE_ARRAYS = ("joint_q", "joint_qd", "body_q", "body_qd")


def _native_state(solver):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(solver.mj_model, spec))
    mujoco.mj_getState(solver.mj_model, solver.mj_data, result, spec)
    return result


@wp.kernel
def _decrement(counter: wp.array[int]):
    counter[0] -= 1


class MuJoCoFreeBodyGPUTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))

    def build(self, *, ground=False, com=(0, 0, 0)):
        with wp.ScopedDevice(self.device):
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            rotation = np.asarray(
                wp.quat_to_matrix(wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, 2, -1)), 0.7))
            ).reshape(3, 3)
            inertia = rotation @ np.diag([1e-8, 1.4e-8, 2e-8]) @ rotation.T
            body = builder.add_body(
                mass=0.0003, inertia=wp.mat33(*inertia.reshape(-1)), com=wp.vec3(*com), lock_inertia=True
            )
            if ground:
                builder.add_shape_sphere(body, radius=0.005)
                builder.add_ground_plane()
            model = builder.finalize()
            native = SolverMuJoCo(model, use_mujoco_cpu=True, integrator="implicitfast", update_data_interval=1)
            return model, native

    def random_state(self, model, rng, *, angular_speed=2):
        state = model.state()
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        state.joint_q.assign(np.r_[rng.uniform(-0.05, 0.05, 3), quat].astype(np.float32))
        state.joint_qd.assign((rng.normal(size=6) * np.array([0.02] * 3 + [angular_speed] * 3)).astype(np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        state.body_f.assign((rng.normal(size=(1, 6)) * np.array([0.002] * 3 + [1e-6] * 3)).astype(np.float32))
        return state

    def assert_state_close(self, actual, expected, *, message=""):
        for name in _STATE_ARRAYS:
            np.testing.assert_allclose(
                getattr(actual, name).numpy(),
                getattr(expected, name).numpy(),
                rtol=3e-6,
                atol=5e-8,
                err_msg=f"{message} {name}",
            )

    def test_random_native_parity_without_mutating_inputs_or_native_history(self):
        model, native = self.build()
        candidate = MuJoCoFreeBodyGPU(model, native)
        bridge = MuJoCoFreeBody(model, native)
        rng = np.random.default_rng(982)
        control = model.control()
        expected, actual = model.state(), model.state()
        for index in range(24):
            dt = (1 / 60, 1 / 240, 1 / 720, 5e-6)[index % 4]
            source = self.random_state(model, rng, angular_speed=10 if index % 3 == 0 else 2)
            control.joint_f.assign((rng.normal(size=6) * np.array([0.001] * 3 + [5e-7] * 3)).astype(np.float32))
            damping = rng.uniform(0.1, 1, 6) * np.array([0.01] * 3 + [1e-7] * 3)
            if index % 3 == 1:
                damping += [400 * dt] * 3 + [0.05 * dt] * 3
            elif index % 3 == 2:
                damping[:] = 0
            native.mj_model.dof_damping[:] = damping
            native.reset(source, flags=0)
            native.mj_data.time = 0.125
            native.mj_data.qacc_warmstart[:] = rng.normal(size=6)
            before = {name: getattr(source, name).numpy() for name in (*_STATE_ARRAYS, "body_f")}
            bridge.step(source, expected, control, dt)
            native_after_reference = _native_state(native)
            candidate.prepare(dt)
            for _ in range(2):
                candidate.step(source, actual, control, dt)
                self.assert_state_close(actual, expected, message=f"random case {index}")
                np.testing.assert_array_equal(_native_state(native), native_after_reference)
                for name, value in before.items():
                    np.testing.assert_array_equal(getattr(source, name).numpy(), value)

    def test_advancing_trajectory_with_changed_timestep_and_damping(self):
        model, native = self.build()
        candidate = MuJoCoFreeBodyGPU(model, native)
        bridge = MuJoCoFreeBody(model, native)
        rng = np.random.default_rng(917)
        source = self.random_state(model, rng)
        gpu_source, expected, actual = model.state(), model.state(), model.state()
        for name in _STATE_ARRAYS:
            wp.copy(getattr(gpu_source, name), getattr(source, name))
        native.reset(source, flags=0)
        for index in range(96):
            dt = (1 / 240, 1 / 60, 5e-6)[index // 32]
            native.mj_model.dof_damping[:] = [0.005] * 3 + [2e-7] * 3 if index < 48 else 0
            candidate.prepare(dt)
            force = (rng.normal(size=(1, 6)) * np.array([0.0001] * 3 + [1e-7] * 3)).astype(np.float32)
            source.body_f.assign(force)
            gpu_source.body_f.assign(force)
            bridge.step(source, expected, None, dt)
            candidate.step(gpu_source, actual, None, dt)
            self.assert_state_close(actual, expected, message=f"advancing step {index}")
            for name in _STATE_ARRAYS:
                wp.copy(getattr(source, name), getattr(expected, name))
                wp.copy(getattr(gpu_source, name), getattr(actual, name))

    def test_eligibility_rejects_contacts_frames_versions_and_callbacks(self):
        model, native = self.build()
        self.assertTrue(MuJoCoFreeBodyGPU.supports(model, native))
        for kwargs in ({"ground": True}, {"com": (0.001, 0, 0)}):
            other_model, other_native = self.build(**kwargs)
            self.assertFalse(MuJoCoFreeBodyGPU.supports(other_model, other_native))
            with self.assertRaises(ValueError):
                MuJoCoFreeBodyGPU(other_model, other_native)
        with patch.object(mujoco, "__version__", "3.15.1"):
            self.assertFalse(MuJoCoFreeBodyGPU.supports(model, native))
        with patch.object(mujoco, "mj_version", return_value=3016000):
            self.assertFalse(MuJoCoFreeBodyGPU.supports(model, native))
        for callback in ("get_mjcb_passive", "get_mjcb_control", "get_mjcb_sensor", "get_mjcb_contactfilter"):
            with patch.object(mujoco, callback, return_value=lambda *_: None):
                self.assertFalse(MuJoCoFreeBodyGPU.supports(model, native))
        with patch.object(native, "step", lambda *_: None):
            self.assertFalse(MuJoCoFreeBodyGPU.supports(model, native))

    def test_changed_native_settings_rejected_before_output_or_history_mutation(self):
        model, native = self.build()
        candidate = MuJoCoFreeBodyGPU(model, native)
        candidate.prepare(1 / 720)
        settings = (
            (native.mj_model.opt, "integrator", mujoco.mjtIntegrator.mjINT_EULER),
            (native.mj_model.opt, "gravity", [0, 0, -1]),
            (native.mj_model.opt, "viscosity", 1e-5),
            (native.mj_model.opt, "density", 1),
            (native.mj_model.opt, "enableflags", int(mujoco.mjtEnableBit.mjENBL_ENERGY)),
            (native.mj_model.opt, "disableflags", 0),
            (native.mj_model, "dof_armature", np.full(6, 1e-9)),
            (native.mj_model, "dof_frictionloss", np.full(6, 1e-7)),
            (native.mj_model, "jnt_stiffness", np.ones_like(native.mj_model.jnt_stiffness)),
            (native.mj_model, "dof_dampingpoly", np.ones_like(native.mj_model.dof_dampingpoly)),
            (native.mj_model, "body_mass", native.mj_model.body_mass * 2),
            (native.mj_model, "body_inertia", native.mj_model.body_inertia * 2),
            (native.mj_model, "dof_damping", np.full(6, -1)),
        )
        for owner, name, changed in settings:
            with self.subTest(setting=name):
                value = getattr(owner, name)
                saved = value.copy() if isinstance(value, np.ndarray) else value
                history = _native_state(native)
                try:
                    if isinstance(value, np.ndarray):
                        value[:] = changed
                    else:
                        setattr(owner, name, changed)
                    with self.assertRaises(ValueError):
                        candidate.prepare(1 / 240)
                    np.testing.assert_array_equal(_native_state(native), history)
                    self.assertEqual(candidate._dt, 1 / 720)
                finally:
                    if isinstance(value, np.ndarray):
                        value[:] = saved
                    else:
                        setattr(owner, name, saved)
        candidate.prepare(1 / 240)

    def test_prepare_refreshes_buffers_without_newton_readback_and_rejects_bad_use(self):
        model, native = self.build()
        candidate = MuJoCoFreeBodyGPU(model, native)
        source, out = model.state(), model.state()
        with self.assertRaises(ValueError):
            candidate.step(source, out, None)
        pointers = (candidate.mass.ptr, candidate.inertia.ptr, candidate.damping.ptr, candidate.timestep.ptr)
        history = _native_state(native)
        with patch.object(wp.array, "numpy", side_effect=AssertionError("unexpected device readback")):
            candidate.prepare(1 / 60)
            native.mj_model.dof_damping[:] = [0.02] * 3 + [2e-7] * 3
            candidate.prepare(1 / 240)
            candidate.prepare(1 / 240)
        self.assertEqual(
            pointers, (candidate.mass.ptr, candidate.inertia.ptr, candidate.damping.ptr, candidate.timestep.ptr)
        )
        np.testing.assert_array_equal(candidate.damping.numpy(), native.mj_model.dof_damping)
        np.testing.assert_array_equal(_native_state(native), history)
        self.assertEqual(candidate.timestep.numpy()[0], 1 / 240)
        for dt in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                candidate.prepare(dt)
        with self.assertRaises(ValueError):
            candidate.step(source, out, None, 1 / 60)
        for name in ("body_qdd", "body_parent_f", "mujoco.qfrc_actuator"):
            other_out = model.state()
            if name.startswith("mujoco"):
                other_out.mujoco = SimpleNamespace(qfrc_actuator=wp.zeros(6, device=self.device))
            else:
                setattr(other_out, name, wp.zeros(1, dtype=wp.spatial_vector, device=self.device))
            with self.assertRaises(ValueError):
                candidate.step(source, other_out, None)

    def test_captured_trials_keep_input_and_native_state_and_refresh_coefficients(self):
        if not self.device.is_cuda:
            self.skipTest("CUDA graph replay requires CUDA")
        model, native = self.build()
        candidate = MuJoCoFreeBodyGPU(model, native)
        bridge = MuJoCoFreeBody(model, native)
        source = self.random_state(model, np.random.default_rng(203))
        expected, actual = model.state(), model.state()
        candidate.prepare(1 / 60)
        candidate.step(source, actual, None)
        counter = wp.zeros(1, dtype=int, device=self.device)

        def body():
            candidate.step(source, actual, None)
            wp.launch(_decrement, 1, [counter], device=self.device)

        # Compile before capture; each nested trial starts from the same source.
        wp.launch(_decrement, 1, [counter], device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            counter.fill_(3)
            wp.capture_while(counter, body)
        before = {name: getattr(source, name).numpy() for name in _STATE_ARRAYS}
        for dt, damping in ((1 / 60, 0.0), (5e-6, 0.005), (1 / 240, 0.002)):
            native.mj_model.dof_damping[:] = [damping] * 3 + [damping * 1e-5] * 3
            candidate.prepare(dt)
            native.reset(source, flags=0)
            bridge.step(source, expected, None, dt)
            history = _native_state(native)
            wp.capture_launch(capture.graph)
            self.assert_state_close(actual, expected, message=f"captured dt {dt}")
            self.assertEqual(counter.numpy()[0], 0)
            np.testing.assert_array_equal(_native_state(native), history)
            for name, value in before.items():
                np.testing.assert_array_equal(getattr(source, name).numpy(), value)


if __name__ == "__main__":
    unittest.main()
