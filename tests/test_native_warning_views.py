# SPDX-License-Identifier: Apache-2.0
"""Native warning views retain allocation ownership through reset and rebuild."""

import os
import unittest
from functools import partial

import mujoco
import numpy as np
import warp as wp
from test_grasp import _Rig
from test_native_step_guard import native

from newton_solver_suturebot._src.solvers.mujoco.native_step_guard import (
    NativeMicrostepGuard,
    NativeRigidStepRejected,
    restore_native_warnings,
    snapshot_native_warnings,
)
from newton_solver_suturebot._src.solvers.mujoco.subcycled_mujoco import SubcycledMuJoCo


class NativeWarningViewTests(unittest.TestCase):
    def test_strided_views_match_record_access_and_roundtrip_exactly(self):
        _model, data = native()
        rng = np.random.default_rng(741)
        for _ in range(100):
            for row in data.warning:
                row.number = int(rng.integers(0, 100))
                row.lastinfo = int(rng.integers(-5, 100))
            expected = tuple((int(row.number), int(row.lastinfo)) for row in data.warning)
            self.assertEqual(snapshot_native_warnings(data), expected)
            data.warning.number[:] = 0
            data.warning.lastinfo[:] = 0
            restore_native_warnings(data, expected)
            self.assertEqual(tuple((int(row.number), int(row.lastinfo)) for row in data.warning), expected)
        before = snapshot_native_warnings(data)
        with self.assertRaises(ValueError):
            restore_native_warnings(data, before[:-1])
        self.assertEqual(snapshot_native_warnings(data), before)

    def test_owned_views_remain_live_across_reset_and_integration_restore(self):
        model, data = native()
        guard = NativeMicrostepGuard(mujoco, data)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, state, spec)
        bad = int(mujoco.mjtWarning.mjWARN_BADQACC)
        for _ in range(2):
            mujoco.mj_resetData(model, data)
            mujoco.mj_setState(model, data, state, spec)
            data.warning[bad].number = 1
            data.warning[bad].lastinfo = 3
            data.qfrc_applied[0] = 1e100
            token = guard.begin(model.opt.timestep)
            try:
                mujoco.mj_step(model, data)
                with self.assertRaisesRegex(NativeRigidStepRejected, "QACC"):
                    guard.check(token)
            finally:
                guard.finish(token)
            self.assertEqual(data.warning[bad].number, 2)
        with self.assertRaises(AttributeError):
            guard.data = mujoco.MjData(model)

    def test_replaced_native_data_rebinds_guard_before_validation(self):
        with wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu")):
            rig = _Rig(second_tool=True, solver_class=partial(SubcycledMuJoCo))
            solver = rig.solver
            old_data, old_guard = solver.mj_data, solver._native_step_guard
            old_snapshot = snapshot_native_warnings(old_data)
            replacement = mujoco.MjData(solver.mj_model)
            replacement.time = 2.0
            replacement.qfrc_applied[0] = 1e100
            solver.mj_data = replacement
            with self.assertRaisesRegex(NativeRigidStepRejected, "QACC"):
                solver._checked_microstep(rig.following, None, needs_body=False, needs_contact=False)
            self.assertIsNot(solver._native_step_guard, old_guard)
            self.assertIs(solver._native_step_guard.data, replacement)
            self.assertIs(old_guard.data, old_data)
            self.assertEqual(snapshot_native_warnings(old_data), old_snapshot)
            self.assertEqual(solver.native_guard_rejections, 1)


if __name__ == "__main__":
    unittest.main()
