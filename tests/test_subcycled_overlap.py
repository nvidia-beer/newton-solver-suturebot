# SPDX-License-Identifier: Apache-2.0
"""Independent particle work may overlap native dynamics without changing it."""

import os
import unittest
from functools import partial
from unittest.mock import patch

import mujoco
import numpy as np
import warp as wp
from test_grasp import _Rig

from newton_solver_suturebot._src.solvers.mujoco.subcycled_mujoco import SubcycledMuJoCo


class SubcycledOverlapTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.device.__enter__()

    def tearDown(self):
        self.device.__exit__(None, None, None)

    def test_callback_preserves_contact_trajectory_and_particle_output(self):
        for options in ({}, {"packed_transfer": False}, {"batch_native": False}):
            with self.subTest(options=options):
                constructor = partial(SubcycledMuJoCo, **options)
                overlap = _Rig(second_tool=True, particles=True, solver_class=constructor)
                serial = _Rig(second_tool=True, particles=True, solver_class=constructor)
                callback_count = 0
                for interval in range(8):
                    forces = overlap.external.copy()
                    forces[0, :3] += [0.001 * interval, -0.002, 0.0005]
                    particles = wp.array([[1.0, 2.0, 0.01 * interval]], dtype=wp.vec3)
                    for rig in (serial, overlap):
                        rig.state.body_f.assign(forces)
                        rig.tool.set_aperture(rig.control, 0.01 * interval)

                    def particle_work(overlap=overlap, particles=particles):
                        nonlocal callback_count
                        callback_count += 1
                        wp.copy(overlap.following.particle_q, particles)
                        wp.copy(overlap.following.particle_qd, particles)

                    wp.copy(serial.following.particle_q, particles)
                    wp.copy(serial.following.particle_qd, particles)
                    serial.solver.step(serial.state, serial.following, serial.control, None, serial.dt)
                    overlap.solver.step(
                        overlap.state,
                        overlap.following,
                        overlap.control,
                        None,
                        overlap.dt,
                        before_native_step=particle_work,
                    )
                    for name in ("joint_q", "joint_qd", "body_q", "body_qd", "particle_q", "particle_qd"):
                        np.testing.assert_array_equal(
                            getattr(overlap.following, name).numpy(), getattr(serial.following, name).numpy()
                        )
                    for name in ("qpos", "qvel", "qacc_warmstart", "ctrl", "xfrc_applied"):
                        np.testing.assert_array_equal(
                            getattr(overlap.solver.mj_data, name), getattr(serial.solver.mj_data, name)
                        )
                    for rig in (serial, overlap):
                        rig.state, rig.following = rig.following, rig.state
                self.assertEqual(callback_count, 8)
                self.assertGreater(overlap.solver.mj_data.ncon, 0)

    def test_callback_runs_after_import_and_before_first_native_step(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                rig = _Rig(solver_class=partial(SubcycledMuJoCo, packed_transfer=packed))
                solver = rig.solver
                # Poison the old native coordinate: callback must see this
                # replaced by the current Newton input, before any dynamics.
                coordinate = float(rig.state.joint_q.numpy()[0])
                solver.mj_data.qpos[0] = coordinate + 0.1
                native_time = solver.mj_data.time
                events = []
                native_step = mujoco.mj_step

                def particle_work(events=events, solver=solver, coordinate=coordinate, native_time=native_time):
                    events.append("particle")
                    self.assertEqual(solver.mj_data.qpos[0], coordinate)
                    self.assertEqual(solver.mj_data.time, native_time)

                def recorded_step(*args, events=events, native_step=native_step):
                    events.append("rigid")
                    native_step(*args)

                with patch.object(solver._mujoco, "mj_step", side_effect=recorded_step):
                    solver.step(rig.state, rig.following, rig.control, None, rig.dt, before_native_step=particle_work)
                self.assertEqual(events, ["particle", "rigid", "rigid", "rigid", "rigid"])

    def test_callback_failure_can_be_rolled_back_and_replayed(self):
        for batch_native in (False, True):
            with self.subTest(batch_native=batch_native):
                constructor = partial(SubcycledMuJoCo, batch_native=batch_native)
                rig = _Rig(second_tool=True, solver_class=constructor)
                reference = _Rig(second_tool=True, solver_class=constructor)
                for _ in range(3):
                    rig.step()
                    reference.step()
                native = rig.solver
                spec = mujoco.mjtState.mjSTATE_INTEGRATION
                saved = np.empty(mujoco.mj_stateSize(native.mj_model, spec))
                mujoco.mj_getState(native.mj_model, native.mj_data, saved, spec)
                native_time = native.mj_data.time
                native_steps = native._step

                def failed_particle_work():
                    raise RuntimeError("particle trial rejected")

                with self.assertRaisesRegex(RuntimeError, "particle trial rejected"):
                    native.step(
                        rig.state, rig.following, rig.control, None, rig.dt, before_native_step=failed_particle_work
                    )
                self.assertEqual(native.mj_data.time, native_time)
                self.assertEqual(native._step, native_steps)
                # Exactly the enclosing physical transaction's native restore.
                mujoco.mj_setState(native.mj_model, native.mj_data, saved, spec)
                rig.step()
                reference.step()
                actual = np.empty_like(saved)
                expected = np.empty_like(saved)
                mujoco.mj_getState(native.mj_model, native.mj_data, actual, spec)
                mujoco.mj_getState(reference.solver.mj_model, reference.solver.mj_data, expected, spec)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(rig.state.joint_q.numpy(), reference.state.joint_q.numpy())

    def test_invalid_step_does_not_run_callback(self):
        rig = _Rig()
        for dt in (0, -1, float("nan"), float("inf")):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                rig.solver.step(rig.state, rig.following, rig.control, None, dt, before_native_step=self.fail)
        with self.assertRaises(ValueError):
            rig.solver.step(rig.state, rig.state, rig.control, None, rig.dt, before_native_step=self.fail)


if __name__ == "__main__":
    unittest.main()
