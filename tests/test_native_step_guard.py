# SPDX-License-Identifier: Apache-2.0
"""Native failure detection, export isolation, warning rollback and replay."""

import os
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import warp as wp
from test_grasp import _Rig

from newton_solver_suturebot._src.solvers.coupled.coupling import NeedleTissueCoupler
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import NeedleStepTooLarge
from newton_solver_suturebot._src.solvers.mujoco.mujoco_native_transfer import NativeMuJoCoTransfer
from newton_solver_suturebot._src.solvers.mujoco.native_step_guard import (
    NativeMicrostepGuard,
    NativeRigidStepRejected,
    restore_native_warnings,
    snapshot_native_warnings,
)
from newton_solver_suturebot._src.solvers.mujoco.subcycled_mujoco import SubcycledMuJoCo


def native():
    model = mujoco.MjModel.from_xml_string("""<mujoco><option timestep=".001" gravity="0 0 0"/>
        <worldbody><body><freejoint/><geom type="sphere" size=".01" mass=".01"/></body></worldbody></mujoco>""")
    return model, mujoco.MjData(model)


class NativeGuardTests(unittest.TestCase):
    def test_bad_position_and_velocity_are_detected(self):
        for name, field in (("QPOS", "qpos"), ("QVEL", "qvel")):
            model, data = native()
            guard = NativeMicrostepGuard(mujoco, data)
            getattr(data, field)[0] = 1e100
            token = guard.begin(model.opt.timestep)
            try:
                mujoco.mj_step(model, data)
                with self.subTest(name=name), self.assertRaisesRegex(NativeRigidStepRejected, name):
                    guard.check(token)
            finally:
                guard.finish(token)

    def test_real_reset_detected_twice_at_zero_with_existing_counter_one(self):
        model, data = native()
        guard = NativeMicrostepGuard(mujoco, data)
        bad = int(mujoco.mjtWarning.mjWARN_BADQACC)
        for attempt in range(2):
            data.time = 0
            data.warning[bad].number = 1
            data.warning[bad].lastinfo = 3
            data.qfrc_applied[0] = 1e100
            before = snapshot_native_warnings(data)
            token = guard.begin(model.opt.timestep)
            try:
                mujoco.mj_step(model, data)
                self.assertTrue(np.isfinite(data.qpos).all())
                self.assertEqual(data.time, model.opt.timestep)
                with self.subTest(attempt=attempt), self.assertRaisesRegex(NativeRigidStepRejected, "QACC"):
                    guard.check(token)
            finally:
                guard.finish(token)
            self.assertEqual(data.warning[bad].number, 2)
            restore_native_warnings(data, before)
            self.assertEqual(snapshot_native_warnings(data), before)

    def test_normal_steps_keep_original_warning_records_and_clock(self):
        model, data = native()
        guard = NativeMicrostepGuard(mujoco, data)
        for index, record in enumerate(data.warning):
            record.number, record.lastinfo = index + 1, 3 * index
        before = snapshot_native_warnings(data)
        data.time = 4.5885
        for _ in range(4):
            token = guard.begin(model.opt.timestep)
            try:
                mujoco.mj_step(model, data)
                guard.check(token)
            finally:
                guard.finish(token)
            self.assertEqual(snapshot_native_warnings(data), before)
        self.assertAlmostEqual(data.time, 4.5925, places=14)

    def test_clock_failure_detected_without_warning(self):
        model, data = native()
        guard = NativeMicrostepGuard(mujoco, data)
        data.time = 4.5885
        token = guard.begin(model.opt.timestep)
        data.time = 0.001
        try:
            with self.assertRaisesRegex(NativeRigidStepRejected, "clock"):
                guard.check(token)
        finally:
            guard.finish(token)
        self.assertTrue(issubclass(NativeRigidStepRejected, NeedleStepTooLarge))

    def test_coupler_native_restore_also_restores_warning_snapshot(self):
        model, data = native()
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(model, spec))
        data.time = 0.4
        data.qpos[0] = 0.03
        data.warning[int(mujoco.mjtWarning.mjWARN_CONTACTFULL)].number = 2
        warnings = snapshot_native_warnings(data)
        mujoco.mj_getState(model, data, before, spec)
        data.qfrc_applied[0] = 1e100
        mujoco.mj_step(model, data)
        owner = object.__new__(NeedleTissueCoupler)
        owner._mujoco, owner._native, owner._state_spec = mujoco, before, spec
        owner._native_warnings = warnings
        owner.rigid_solver = SimpleNamespace(mj_model=model, mj_data=data)
        owner._restore_native()
        after = np.empty_like(before)
        mujoco.mj_getState(model, data, after, spec)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(snapshot_native_warnings(data), warnings)

    def test_batch_general_observable_paths_reject_before_endpoint_export(self):
        with wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu")):
            for options in ({}, {"packed_transfer": False}, {"batch_native": False}):
                for observables_enabled in (False, True):
                    with self.subTest(options=options, observables=observables_enabled):
                        rig = _Rig(second_tool=True, solver_class=partial(SubcycledMuJoCo, **options))
                        native_solver = rig.solver
                        before = {
                            name: getattr(rig.following, name).numpy().copy()
                            for name in ("joint_q", "joint_qd", "body_q", "body_qd")
                        }
                        warning_before = snapshot_native_warnings(native_solver.mj_data)
                        kinds = {native_solver.ObservableKind.BODY_QDD} if observables_enabled else None
                        observables = native_solver.observables(kinds) if kinds else None
                        original_step = mujoco.mj_step
                        original_forward = mujoco.mj_forward

                        def invalid_step(model, data, original_step=original_step):
                            data.qfrc_applied[0] = 1e100
                            original_step(model, data)

                        def invalid_forward(model, data, original_forward=original_forward):
                            data.qfrc_applied[0] = 1e100
                            original_forward(model, data)

                        # Injection uses real native validation/auto-reset, not a mocked warning.
                        method = "mj_forward" if observables_enabled else "mj_step"
                        injected = invalid_forward if observables_enabled else invalid_step
                        with (
                            patch.object(mujoco, method, side_effect=injected),
                            patch.object(
                                native_solver, "_update_newton_state", wraps=native_solver._update_newton_state
                            ) as export,
                            patch.object(native_solver, "_update_body_observables") as body_export,
                            patch.object(NativeMuJoCoTransfer, "pull") as packed_export,
                        ):
                            with self.assertRaises(NativeRigidStepRejected):
                                native_solver.step(
                                    rig.state, rig.following, rig.control, None, rig.dt, observables=observables
                                )
                            # General fallback exports its earlier valid
                            # microsteps to private scratch. The invalid
                            # microstep never exports the requested endpoint.
                            for call in export.call_args_list:
                                self.assertIsNot(call.args[1], rig.following)
                            self.assertEqual(
                                export.call_count,
                                3 if options.get("batch_native") is False and observables_enabled else 0,
                            )
                            packed_export.assert_not_called()
                            body_export.assert_not_called()
                        for name, value in before.items():
                            np.testing.assert_array_equal(getattr(rig.following, name).numpy(), value)
                        self.assertEqual(snapshot_native_warnings(native_solver.mj_data), warning_before)
                        self.assertEqual(native_solver.native_guard_rejections, 1)


if __name__ == "__main__":
    unittest.main()
