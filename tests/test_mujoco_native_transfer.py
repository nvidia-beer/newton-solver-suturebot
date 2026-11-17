# SPDX-License-Identifier: Apache-2.0
"""Packed transfers preserve articulated control and native contact endpoints."""

import os
import unittest
from functools import partial
from types import SimpleNamespace

import numpy as np
import warp as wp
from test_grasp import _Rig

from newton_solver_suturebot._src.solvers.mujoco.mujoco_native_transfer import NativeMuJoCoTransfer
from newton_solver_suturebot._src.solvers.mujoco.subcycled_mujoco import SubcycledMuJoCo


class NativeMuJoCoTransferTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.device.__enter__()

    def tearDown(self):
        self.device.__exit__(None, None, None)

    def test_changed_controls_and_loads_match_general_conversion_bitwise(self):
        packed = _Rig(second_tool=True)
        general = _Rig(second_tool=True, solver_class=partial(SubcycledMuJoCo, packed_transfer=False))
        packets = None
        for interval in range(12):
            applied = packed.external.copy() if interval % 3 else np.zeros_like(packed.external)
            applied[0, :3] += np.asarray([0.001, -0.002, 0.003], dtype=np.float32) * (interval % 2)
            for rig in (packed, general):
                rig.tool.set_aperture(rig.control, 0.04 if interval > 5 else 0)
                rig.parked.set_aperture(rig.control, 0.2 + interval * 0.01)
                rig.state.body_f.assign(applied)
                rig.solver.step(rig.state, rig.following, rig.control, None, rig.dt)
                rig.state, rig.following = rig.following, rig.state
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                np.testing.assert_array_equal(getattr(packed.state, name).numpy(), getattr(general.state, name).numpy())
            for name in ("qpos", "qvel", "qacc_warmstart", "ctrl", "qfrc_applied", "xfrc_applied"):
                np.testing.assert_array_equal(
                    getattr(packed.solver.mj_data, name), getattr(general.solver.mj_data, name)
                )
            bridge = packed.solver._native_transfer
            addresses = tuple(
                array.ptr for array in (bridge.input, bridge.output, bridge.host_input, bridge.host_output)
            )
            if packets is None:
                packets = addresses
            self.assertEqual(addresses, packets)
        self.assertEqual((bridge.readbacks, bridge.uploads), (12, 12))
        self.assertIsNone(general.solver._native_transfer)
        if wp.get_device().is_cuda:
            self.assertEqual(bridge.captures, 4)
            self.assertGreater(bridge.launches, bridge.captures)

    def test_extended_state_and_tape_keep_general_conversion(self):
        rig = _Rig()

        def supports():
            return NativeMuJoCoTransfer.supports(rig.solver, rig.state, rig.following, rig.control, None)

        self.assertTrue(supports())
        with wp.Tape():
            self.assertFalse(supports())
        rig.following.mujoco = SimpleNamespace(qfrc_constraint=wp.zeros(rig.solver.mj_model.nv))
        self.assertFalse(supports())
        rig.following.mujoco = None
        rig.state.body_f = wp.clone(rig.state.body_f, requires_grad=True)
        self.assertFalse(supports())


if __name__ == "__main__":
    unittest.main()
