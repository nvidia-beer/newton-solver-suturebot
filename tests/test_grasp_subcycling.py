# SPDX-License-Identifier: Apache-2.0
"""Native subcycling matches explicit held-load steps without touching FEM."""

import os
import unittest
from functools import partial

import mujoco
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo
from test_grasp import _Rig

from newton_solver_suturebot._src.solvers.mujoco.subcycled_mujoco import SubcycledMuJoCo


class GraspSubcyclingTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.device.__enter__()

    def tearDown(self):
        self.device.__exit__(None, None, None)

    def test_matches_native_microsteps_and_preserves_output_particles(self):
        grouped = _Rig(second_tool=True, particles=True)
        explicit = _Rig(second_tool=True, particles=True, solver_class=SolverMuJoCo)
        particle_q = np.asarray([[1.1, 1.2, 1.3]], dtype=np.float32)
        particle_qd = np.asarray([[2.1, 2.2, 2.3]], dtype=np.float32)
        for interval in range(12):
            grouped.following.particle_q.assign(particle_q)
            grouped.following.particle_qd.assign(particle_qd)
            # Include changing loads and zero-load intervals, so stale scratch
            # forces or applying the load only once cannot satisfy equality.
            forces = grouped.external.copy() if interval % 3 else np.zeros_like(grouped.external)
            forces[0, :3] += np.asarray([0.002, -0.001, 0.0005]) * (interval % 2)
            grouped.state.body_f.assign(forces)
            target = grouped.target.copy()
            target[0] += 0.0001
            grouped.drive.add_forces(grouped.state, target, np.zeros(6))
            grouped.parked_drive.add_forces(grouped.state, grouped.parked_target, np.zeros(6))
            held = grouped.state.body_f.numpy().copy()
            input_coordinates = grouped.state.joint_q.numpy().copy()
            grouped.solver.step(grouped.state, grouped.following, grouped.control, None, grouped.dt)
            explicit.state.body_f.assign(held)
            native = explicit.solver
            native._apply_mjc_control(explicit.model, explicit.state, explicit.control, native.mj_data)
            native._update_mjc_data(native.mj_data, explicit.model, explicit.state)
            native.mj_model.opt.timestep = explicit.dt / 4
            for _ in range(4):
                mujoco.mj_step(native.mj_model, native.mj_data)
            native._update_newton_state(explicit.model, explicit.following, native.mj_data, state_prev=explicit.state)
            explicit.state, explicit.following = explicit.following, explicit.state
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                np.testing.assert_array_equal(
                    getattr(grouped.following, name).numpy(), getattr(explicit.state, name).numpy()
                )
            np.testing.assert_array_equal(grouped.following.particle_q.numpy(), particle_q)
            np.testing.assert_array_equal(grouped.following.particle_qd.numpy(), particle_qd)
            np.testing.assert_array_equal(grouped.state.joint_q.numpy(), input_coordinates)
            np.testing.assert_array_equal(grouped.state.body_f.numpy(), held)
            np.testing.assert_array_equal(grouped.solver.mj_data.qpos, explicit.solver.mj_data.qpos)
            np.testing.assert_array_equal(grouped.solver.mj_data.qacc_warmstart, explicit.solver.mj_data.qacc_warmstart)
            grouped.state, grouped.following = grouped.following, grouped.state
        self.assertEqual(grouped.solver.native_batches, 12)
        self.assertEqual(grouped.solver.general_intervals, 0)
        self.assertGreater(grouped.solver.mj_data.ncon, 0)

    def test_general_fallback_matches_original_adapter_bitwise(self):
        grouped = _Rig(second_tool=True, solver_class=partial(SubcycledMuJoCo, batch_native=False))
        explicit = _Rig(second_tool=True, solver_class=SolverMuJoCo)
        for _ in range(5):
            grouped.state.body_f.assign(grouped.external)
            grouped.solver.step(grouped.state, grouped.following, grouped.control, None, grouped.dt)
            for _ in range(4):
                explicit.state.body_f.assign(grouped.external)
                explicit.solver.step(explicit.state, explicit.following, explicit.control, None, explicit.dt / 4)
                explicit.state, explicit.following = explicit.following, explicit.state
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                np.testing.assert_array_equal(
                    getattr(grouped.following, name).numpy(), getattr(explicit.state, name).numpy()
                )
            grouped.state, grouped.following = grouped.following, grouped.state
        self.assertEqual(grouped.solver.general_intervals, 5)
        self.assertEqual(grouped.solver.native_batches, 0)

    def test_native_integration_snapshot_restores_contact_trial_exactly(self):
        rig = _Rig(second_tool=True)
        for _ in range(10):
            rig.step()
        native = rig.solver
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(native.mj_model, spec))
        expected = np.empty_like(before)
        actual = np.empty_like(before)
        mujoco.mj_getState(native.mj_model, native.mj_data, before, spec)
        applied = rig.external.copy()
        applied[0] += [0.002, -0.001, 0.003, 1e-6, -2e-6, 3e-6]
        rig.state.body_f.assign(applied)
        native.step(rig.state, rig.following, rig.control, None, rig.dt)
        mujoco.mj_getState(native.mj_model, native.mj_data, expected, spec)
        endpoint = rig.following.joint_q.numpy().copy()
        # The enclosing FEM transaction owns rollback; no hidden per-batch
        # state may make replay of the same physical trial diverge.
        mujoco.mj_setState(native.mj_model, native.mj_data, before, spec)
        native.step(rig.state, rig.following, rig.control, None, rig.dt)
        mujoco.mj_getState(native.mj_model, native.mj_data, actual, spec)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(rig.following.joint_q.numpy(), endpoint)
        np.testing.assert_array_equal(rig.state.body_f.numpy(), applied)

    def test_final_microstep_observables_match_native_dynamics(self):
        grouped = _Rig(second_tool=True)
        reference = _Rig(second_tool=True, solver_class=SolverMuJoCo)
        kind = SolverMuJoCo.ObservableKind
        kinds = {kind.BODY_QDD, kind.BODY_PARENT_F, kind.QFRC_ACTUATOR, kind.QFRC_CONSTRAINT}
        observed = grouped.solver.observables(kinds)
        expected = reference.solver.observables(kinds)
        grouped.state.body_f.assign(grouped.external)
        grouped.solver.step(grouped.state, grouped.following, grouped.control, None, grouped.dt, observables=observed)
        native = reference.solver
        reference.state.body_f.assign(reference.external)
        native._apply_mjc_control(reference.model, reference.state, reference.control, native.mj_data)
        native._update_mjc_data(native.mj_data, reference.model, reference.state)
        native.mj_model.opt.timestep = reference.dt / 4
        for _ in range(3):
            mujoco.mj_step(native.mj_model, native.mj_data)
        native._step_mujoco_cpu_with_observables(reference.following, expected)
        native._update_newton_state(
            reference.model, reference.following, native.mj_data, state_prev=reference.state, observables=expected
        )
        for name in ("body_qdd", "body_parent_f", "qfrc_actuator", "qfrc_constraint"):
            np.testing.assert_array_equal(getattr(observed, name).numpy(), getattr(expected, name).numpy())
        self.assertGreater(np.abs(observed.body_qdd.numpy()).max(), 0)
        self.assertGreater(np.abs(observed.qfrc_actuator.numpy()).max(), 0)

    def test_invalid_subcycles_and_timestep_fail_without_mutation(self):
        rig = _Rig()
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SubcycledMuJoCo(rig.model, rigid_substeps=value, use_mujoco_cpu=True)
        with self.assertRaises(ValueError):
            SubcycledMuJoCo(rig.model, use_mujoco_cpu=False)
        state = rig.state.joint_q.numpy().copy()
        native = rig.solver.mj_data.qpos.copy()
        for dt in (0, -1, float("nan"), float("inf")):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                rig.solver.step(rig.state, rig.following, rig.control, None, dt)
        np.testing.assert_array_equal(rig.state.joint_q.numpy(), state)
        np.testing.assert_array_equal(rig.solver.mj_data.qpos, native)


if __name__ == "__main__":
    unittest.main()
