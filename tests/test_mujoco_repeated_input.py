# SPDX-License-Identifier: Apache-2.0
"""Repeated native inputs preserve contact trials and failed-step rollback."""

import os
import unittest
from functools import partial
from unittest.mock import patch

import mujoco
import numpy as np
import warp as wp
from test_grasp import _Rig

from newton_solver_suturebot._src.subcycled_mujoco import SubcycledMuJoCo


def _snapshot(solver):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(solver.mj_model, spec))
    mujoco.mj_getState(solver.mj_model, solver.mj_data, result, spec)
    return result


def _restore(solver, snapshot):
    mujoco.mj_setState(solver.mj_model, solver.mj_data, snapshot, mujoco.mjtState.mjSTATE_INTEGRATION)


class RepeatedNativeInputTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.device.__enter__()

    def tearDown(self):
        self.device.__exit__(None, None, None)

    def test_rejected_trials_and_changed_controls_match_full_import_bitwise(self):
        fast, original = _Rig(second_tool=True), _Rig(second_tool=True)
        for transaction in range(3):
            for rig in (fast, original):
                rig.tool.set_aperture(rig.control, 0.01 * transaction)
                rig.parked.set_aperture(rig.control, 0.15 + 0.03 * transaction)
                joint_forces = np.zeros(rig.control.joint_f.shape, dtype=np.float32)
                joint_forces[:6] = [0.003, -0.002, 0.001, 3e-6, -2e-6, 1e-6]
                rig.control.joint_f.assign(joint_forces)
            checkpoints = [_snapshot(rig.solver) for rig in (fast, original)]
            transaction_input = fast.solver.begin_repeated_input(fast.state, fast.following, fast.control, 0)
            self.assertIsNotNone(transaction_input)
            try:
                for trial in range(4):
                    for rig, checkpoint in zip((fast, original), checkpoints, strict=True):
                        _restore(rig.solver, checkpoint)
                        force = rig.external.copy()
                        force[0] += np.asarray([0.001, -0.002, 0.003, 1e-6, -2e-6, 3e-6]) * (trial - 1)
                        rig.state.body_f.assign(force)
                        rig.solver.step(rig.state, rig.following, rig.control, None, rig.dt / (transaction + 1))
                    np.testing.assert_array_equal(_snapshot(fast.solver), _snapshot(original.solver))
                    for name in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied", "qacc_warmstart"):
                        np.testing.assert_array_equal(
                            getattr(fast.solver.mj_data, name), getattr(original.solver.mj_data, name)
                        )
                    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                        np.testing.assert_array_equal(
                            getattr(fast.following, name).numpy(), getattr(original.following, name).numpy()
                        )
            finally:
                transaction_input.end_repeated_input()
            for rig in (fast, original):
                rig.state, rig.following = rig.following, rig.state
        transfer = fast.solver._native_transfer
        self.assertEqual(transfer.readbacks, 12)
        self.assertEqual(transfer.repeated_readbacks, 9)
        self.assertFalse(transfer._repeated_active)
        self.assertEqual(original.solver._native_transfer.repeated_readbacks, 0)

    def test_transaction_end_invalidates_values_even_when_state_storage_is_reused(self):
        rig = _Rig()
        native_input = rig.solver.begin_repeated_input(rig.state, rig.following, rig.control, 0)
        try:
            rig.step()
            with self.assertRaisesRegex(RuntimeError, "cannot be nested"):
                rig.solver.begin_repeated_input(rig.state, rig.following, rig.control, 0)
        finally:
            native_input.end_repeated_input()
        rig.tool.set_aperture(rig.control, 0.8)
        rig.step()
        self.assertFalse(native_input._repeated_active)
        self.assertEqual(native_input.repeated_readbacks, 0)
        self.assertGreater(np.abs(rig.solver.mj_data.ctrl).max(), 0.4)

    def test_unsupported_transfer_preserves_general_adapter(self):
        rig = _Rig(solver_class=partial(SubcycledMuJoCo, packed_transfer=False))
        self.assertIsNone(rig.solver.begin_repeated_input(rig.state, rig.following, rig.control, 0))
        self.assertIsNone(rig.solver._native_transfer)
        rig.step()

    def test_coupler_exception_closes_transaction_and_restores_native_history(self):
        from test_grasp_scene import _native, _scene

        with _scene() as (scene, _):
            before = _native(scene)
            coupler = scene.coupler
            coupler.capture_coupling = False
            with patch.object(coupler, "_evaluate", side_effect=RuntimeError("intentional trial failure")):
                with self.assertRaisesRegex(RuntimeError, "intentional trial failure"):
                    coupler.step(scene.state_0, scene.state_1, scene.control, scene.sim_dt)
            transfer = scene.rigid_solver._native_transfer
            self.assertIsNotNone(transfer)
            self.assertFalse(transfer._repeated_active)
            self.assertFalse(transfer._repeated_input.ready)
            np.testing.assert_array_equal(_native(scene), before)
            scene.step()
            self.assertFalse(transfer._repeated_active)
            self.assertGreater(scene.elapsed, 0)


if __name__ == "__main__":
    unittest.main()
