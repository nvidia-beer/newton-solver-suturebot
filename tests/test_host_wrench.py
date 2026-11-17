# SPDX-License-Identifier: Apache-2.0
"""One-use provenance, exact conversion and unchanged native trial imports."""

import os
import threading
import unittest
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from newton._src.solvers.mujoco.kernels import (
    apply_mjc_body_f_kernel,
    apply_mjc_free_joint_f_to_body_f_kernel,
)
from test_grasp import _Rig
from test_mujoco_repeated_input import _restore, _snapshot

from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_anderson_device import DeviceSurfaceAnderson
from newton_solver_suturebot._src.solvers.coupled.coupling import _apply_interface_load
from newton_solver_suturebot._src.solvers.coupled.host_wrench import (
    _FMA,
    HostLoadWitness,
    HostWrenchTransaction,
    convert_wrench,
    stream_id,
    supported_kernels,
)


def bits(array):
    return np.asarray(array, dtype=np.float32).view(np.uint32)


class HostWrenchContracts(unittest.TestCase):
    def test_fma_matches_exact_rational_and_preserves_the_required_contraction(self):
        a, b, c = 1.0 + 2**-27, 1.0 - 2**-27, -1.0
        exact = float(Fraction(a) * Fraction(b) + Fraction(c))
        self.assertEqual(_FMA(a, b, c), exact)
        self.assertNotEqual(a * b + c, exact)
        packet = np.zeros(23, dtype=np.float32)
        packet[3] = c
        load = np.zeros(6, dtype=np.float64)
        load[3] = b
        result = convert_wrench(packet, load, a)
        self.assertEqual(bits(result)[3], bits([exact])[0])

    def test_nonfinite_or_unsupported_rounding_uses_fallback(self):
        packet = np.zeros(23, dtype=np.float32)
        load = np.zeros(6, dtype=np.float64)
        for value in (np.inf, np.nan):
            bad = load.copy()
            bad[1] = value
            self.assertIsNone(convert_wrench(packet, bad, 0.006))
        with patch("newton_solver_suturebot._src.solvers.coupled.host_wrench.supported_math", return_value=False):
            self.assertIsNone(convert_wrench(packet, load, 0.006))
        packet[22] = 1
        np.testing.assert_array_equal(convert_wrench(packet, load, 0.006), np.zeros(6))

    def _transaction(self):
        device = wp.get_device("cpu")
        load = wp.zeros(12, dtype=wp.float64, device=device)
        transfer = SimpleNamespace(device=device, uploads=4, _last_stream=None)
        repeated = SimpleNamespace(transfer=transfer, ready=True)
        tx = object.__new__(HostWrenchTransaction)
        tx.repeated, tx.load, tx.moment_length = repeated, load, 0.006
        tx.thread, tx.stream = threading.current_thread(), None
        tx.pending, tx.closed, tx.generation = None, False, 1
        tx.host_view = np.zeros(23, dtype=np.float32)
        tx._bindings = lambda: (1, 2, 3)
        tx.bindings = tx._bindings()
        return tx

    def _witness(self, tx):
        return HostLoadWitness.from_sparse(tx.load, np.array([2, 6, 8, 11]), np.array([99.0, -2.0, 3.0, 4.0]), tx)

    def test_exact_sparse_rows_and_one_use(self):
        tx = self._transaction()
        witness = self._witness(tx)
        np.testing.assert_array_equal(witness.values, [-2, 0, 3, 0, 0, 4])
        self.assertFalse(witness.values.flags.writeable)
        tx.publish(witness, 7)
        token = tx.offer(7)
        result = tx.consume(token)
        np.testing.assert_array_equal(bits(result), bits([-2, 0, 3, 0, 0, 0.024]))
        self.assertIsNone(tx.consume(token))
        self.assertIsNone(tx.offer(7))

    def test_wrong_iteration_generation_transaction_thread_or_upload_cannot_reuse(self):
        for failure in (
            "iteration",
            "generation",
            "transaction",
            "thread",
            "upload",
            "closed",
            "not_ready",
            "storage",
            "stream",
        ):
            with self.subTest(failure=failure):
                tx = self._transaction()
                witness = self._witness(tx)
                if failure == "transaction":
                    witness.scope = self._transaction()
                if failure == "thread":
                    witness.thread = object()
                tx.publish(witness, 7)
                if failure == "generation":
                    tx.invalidate()
                token = tx.offer(8 if failure == "iteration" else 7)
                if failure == "upload":
                    tx.repeated.transfer.uploads += 1
                if failure == "closed":
                    tx.close()
                if failure == "not_ready":
                    tx.repeated.ready = False
                if failure == "storage":
                    tx.bindings = ()
                if failure == "stream":
                    tx.stream = 9
                self.assertIsNone(tx.consume(token))

    def test_replaced_interface_kernels_disable_new_and_pending_host_imports(self):
        from newton_solver_suturebot._src.solvers.coupled import coupling, host_trial_prepare

        self.assertTrue(supported_kernels())
        for module, name in (
            (coupling, "_apply_interface_load"),
            (coupling, "_apply_rigid_interface_load"),
            (host_trial_prepare, "_apply_interface_load"),
        ):
            with self.subTest(module=module.__name__, kernel=name):
                tx = self._transaction()
                witness = self._witness(tx)
                tx.publish(witness, 7)
                token = tx.offer(7)
                with patch.object(module, name, object()):
                    self.assertFalse(supported_kernels())
                    self.assertIsNone(tx.consume(token))
                self.assertTrue(witness.used)
                self.assertIsNone(tx.consume(token))
                self.assertTrue(supported_kernels())

    def test_no_native_import_changes_without_the_explicit_cuda_option(self):
        with wp.ScopedDevice("cpu"):
            rig = _Rig()
            transfer = rig.solver.begin_repeated_input(rig.state, rig.following, rig.control, 0)
            try:
                load = wp.zeros(6, dtype=wp.float64)
                self.assertIsNone(transfer._repeated_input.enable_host_wrench(rig.state.body_f, load, 0.006))
                rig.step()
                self.assertEqual(transfer.readbacks, 1)
                self.assertEqual(transfer._repeated_input.host_reuses, 0)
            finally:
                transfer.end_repeated_input()

    def test_existing_proposal_override_keeps_its_signature_and_declines_witnesses(self):
        class Override(DeviceSurfaceAnderson):
            def propose_device(self, load, residual, relaxation, direction=None):
                return super().propose_device(load, residual, relaxation, direction=direction)

        load = wp.zeros(6, dtype=wp.float64, device="cpu")
        residual = wp.ones_like(load)
        override = Override(load)
        scope = SimpleNamespace(generation=1)
        self.assertFalse(override.bind_host_wrench_scope(scope))
        override.propose_device(load, residual, 0.5)
        self.assertIsNone(override.take_host_load_witness())
        self.assertIsNone(override._host_wrench_scope)
        ordinary = DeviceSurfaceAnderson(load)
        self.assertTrue(ordinary.bind_host_wrench_scope(scope))
        ordinary.reset()
        self.assertIsNone(ordinary._host_wrench_scope)

    def test_snapshot_keeps_raw_external_and_held_joint_values_before_first_import(self):
        with wp.ScopedDevice("cpu"):
            rig = _Rig()
            raw = rig.external.copy()
            raw[0, 3:] = [1.0e-5, -2.0e-5, 3.0e-5]
            rig.state.body_f.assign(raw)
            joint = np.zeros(rig.control.joint_f.size, dtype=np.float32)
            joint[:6] = np.arange(6, dtype=np.float32) * 1.0e-6
            rig.control.joint_f.assign(joint)
            transfer = rig.solver.begin_repeated_input(rig.state, rig.following, rig.control, 0)
            try:
                tx = HostWrenchTransaction(
                    transfer._repeated_input, rig.state.body_f, wp.zeros(6, dtype=wp.float64), 0.006
                )
                rig.state.body_f.zero_()
                transfer._push_full(rig.state, rig.control)
                np.testing.assert_array_equal(bits(tx.host_view[:6]), bits(raw[0]))
                np.testing.assert_array_equal(bits(tx.host_view[14:20]), bits(joint[:6]))
                self.assertEqual(tx.host_view[21], 1)
                self.assertEqual(tx.host_view[22], 0)
                tx.close()
            finally:
                transfer.end_repeated_input()


class HostWrenchCUDA(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not self.device.is_cuda:
            self.skipTest("Actual CUDA PTX and ordered-stream contracts")
        self.scope = wp.ScopedDevice(self.device)
        self.scope.__enter__()

    def tearDown(self):
        if hasattr(self, "scope"):
            self.scope.__exit__(None, None, None)

    def test_original_kernel_chain_bitwise_random_gravity_joint_and_contracted_torque(self):
        rng = np.random.default_rng(743)
        load = wp.zeros(6, dtype=wp.float64)
        external = wp.zeros(1, dtype=wp.spatial_vector)
        body_f = wp.empty_like(external)
        nodes = wp.empty(0, dtype=wp.vec3)
        mapping = wp.array([[0]], dtype=int)
        flags = wp.array([0], dtype=int)
        mass = wp.zeros(1)
        world = wp.array([1], dtype=int)
        gravity = wp.zeros(2, dtype=wp.vec3)
        gravcomp = wp.zeros((1, 1))
        free_start = wp.array([0], dtype=int)
        joint_f = wp.zeros(6)
        output = wp.zeros((1, 1), dtype=wp.spatial_vector)
        cases = []
        for index in range(128):
            packet = np.zeros(23, dtype=np.float32)
            packet[:6] = rng.normal(size=6) * 10.0 ** rng.uniform(-8, 4)
            packet[6] = 10.0 ** rng.uniform(-7, 2)
            packet[7] = rng.uniform(-0.5, 1.5)
            packet[8:14] = rng.normal(size=6) * 9.81
            packet[14:20] = rng.normal(size=6) * 10.0 ** rng.uniform(-8, 2)
            packet[20:22] = [index % 2, (index // 2) % 2]
            packet[22] = index % 17 == 0
            loads = rng.normal(size=6) * 10.0 ** rng.uniform(-10, 5)
            moment = float(10.0 ** rng.uniform(-6, 1))
            cases.append((packet, loads, moment))
        # Explicit fused and signed-zero/subnormal edge cases.
        packet = np.zeros(23, dtype=np.float32)
        packet[3] = -1
        loads = np.zeros(6, dtype=np.float64)
        loads[3] = 1 - 2**-27
        cases.append((packet, loads, 1 + 2**-27))
        packet = np.zeros(23, dtype=np.float32)
        packet[:6] = [-0.0, np.nextafter(np.float32(0), np.float32(1)), 0, -0.0, 0, 0]
        cases.append((packet, np.array([-0.0, -1e-45, 0, -0.0, 0, 0]), 0.006))
        for index, (packet, loads, moment) in enumerate(cases):
            with self.subTest(case=index):
                load.assign(loads)
                external.assign(packet[:6].reshape(1, 6))
                flags.assign([2 if packet[22] else 0])
                mass.assign(packet[6:7])
                gravcomp.assign(packet[7:8].reshape(1, 1))
                world.assign([1 if packet[20] else -2])
                gravity.assign(np.stack((packet[11:14], packet[8:11])))
                free_start.assign([0 if packet[21] else -1])
                joint_f.assign(packet[14:20])
                wp.launch(_apply_interface_load, 1, [load, nodes, external, 0, moment, nodes, body_f])
                wp.launch(
                    apply_mjc_body_f_kernel,
                    (1, 1),
                    [mapping, flags, body_f, mass, world, gravity, gravcomp],
                    [output],
                )
                wp.launch(
                    apply_mjc_free_joint_f_to_body_f_kernel,
                    (1, 1),
                    [mapping, flags, free_start, joint_f],
                    [output],
                )
                actual = output.numpy().reshape(6)
                expected = convert_wrench(packet, loads, moment)
                np.testing.assert_array_equal(bits(actual), bits(expected))

    def test_native_cached_import_and_endpoint_are_bitwise_equal(self):
        fast, reference = _Rig(second_tool=True), _Rig(second_tool=True)
        rng = np.random.default_rng(193)
        for rig in (fast, reference):
            rig.state.body_f.assign(rig.external)
            values = np.zeros(rig.control.joint_f.size, dtype=np.float32)
            values[:6] = [0.003, -0.002, 0.001, 3e-6, -2e-6, 1e-6]
            rig.control.joint_f.assign(values)
        checkpoints = [_snapshot(rig.solver) for rig in (fast, reference)]
        load = wp.zeros(6, dtype=wp.float64)
        transfer = fast.solver.begin_repeated_input(fast.state, fast.following, fast.control, 0)
        tx = transfer._repeated_input.enable_host_wrench(fast.state.body_f, load, 0.006)
        external = wp.array(fast.external, dtype=wp.spatial_vector)
        nodes = wp.empty(0, dtype=wp.vec3)
        try:
            for trial in range(6):
                loads = rng.normal(size=6) * 0.0002
                load.assign(loads)
                for rig in (fast, reference):
                    wp.launch(
                        _apply_interface_load,
                        rig.model.body_count,
                        [load, nodes, external, 0, 0.006, nodes, rig.state.body_f],
                    )
                if trial:
                    # This read is the same ordered-stream fence provided by
                    # DeviceAnderson's already-required packet.numpy().
                    values = load.numpy()
                    witness = HostLoadWitness.from_sparse(load, np.arange(6), values, tx)
                    tx.publish(witness, trial)
                    transfer._repeated_input.host_token = tx.offer(trial)
                for rig, checkpoint in zip((fast, reference), checkpoints, strict=True):
                    _restore(rig.solver, checkpoint)
                    rig.solver.step(rig.state, rig.following, rig.control, None, rig.dt)
                np.testing.assert_array_equal(_snapshot(fast.solver), _snapshot(reference.solver))
                for field in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied", "qacc_warmstart"):
                    np.testing.assert_array_equal(
                        getattr(fast.solver.mj_data, field), getattr(reference.solver.mj_data, field)
                    )
                for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
                    np.testing.assert_array_equal(
                        getattr(fast.following, field).numpy(), getattr(reference.following, field).numpy()
                    )
                tx.invalidate()
            self.assertEqual(transfer.readbacks, 1)
            self.assertEqual(transfer._repeated_input.host_reuses, 5)
            # No witness: original conversion/readback remains the fallback.
            _restore(fast.solver, checkpoints[0])
            fast.solver.step(fast.state, fast.following, fast.control, None, fast.dt)
            self.assertEqual(transfer.readbacks, 2)
            self.assertEqual(transfer.repeated_readbacks, 1)
            # A different stream cannot consume a witness from the old stream.
            values = load.numpy()
            witness = HostLoadWitness.from_sparse(load, np.arange(6), values, tx)
            tx.publish(witness, 10)
            transfer._repeated_input.host_token = tx.offer(10)
            other = wp.Stream(self.device)
            with wp.ScopedStream(other):
                _restore(fast.solver, checkpoints[0])
                fast.solver.step(fast.state, fast.following, fast.control, None, fast.dt)
            self.assertEqual(transfer.readbacks, 3)
            self.assertEqual(transfer._repeated_input.host_reuses, 5)
        finally:
            transfer.end_repeated_input()
        self.assertTrue(tx.closed)
        self.assertIsNone(transfer._repeated_input.host_token)

    def test_device_proposal_sparse_recovery_reset_and_changed_stream(self):
        load = wp.zeros(24, dtype=wp.float64)
        residual = wp.zeros_like(load)
        aa = DeviceSurfaceAnderson(load)
        tx = SimpleNamespace(generation=4)
        found = 0
        for _iteration in range(12):
            values = load.numpy()
            force = np.zeros(24)
            force[[2, 7, 18, 20, 23]] = 0.2 + values[[2, 7, 18, 20, 23]] * 0.3
            residual.assign(force - values)
            self.assertTrue(aa.bind_host_wrench_scope(tx))
            applied = aa.propose_device(load, residual, 0.5)
            witness = aa.take_host_load_witness()
            if applied:
                found += 1
                self.assertIs(witness.load, load)
                self.assertEqual(witness.stream, stream_id(self.device))
                np.testing.assert_array_equal(witness.values.view(np.uint64), load.numpy()[-6:].view(np.uint64))
                self.assertIsNone(aa.take_host_load_witness())
            else:
                self.assertIsNone(witness)
                load.assign(values + 0.5 * (force - values))
        self.assertGreater(found, 0)
        aa.reset()
        self.assertIsNone(aa.take_host_load_witness())

    def test_real_coupling_late_rejection_restores_start_and_closes_witness_scope(self):
        from test_grasp_scene import _scene
        from test_mujoco_repeated_input import _snapshot as native_snapshot
        from test_surface_anderson_coupling import _snapshot

        from newton_solver_suturebot._src.solvers.coupled.coupling import NeedleTissueCoupler

        with _scene() as (example, _config):
            pose = example.plan.pose(0).copy()
            pose[2] += 0.6 * example.geometry.thickness
            example._initialize_pose(pose)
            example.interaction.puncture_force = 1.0
            example.solver.linear_solver_tolerance = 1e-6
            example.solver.refresh_tangent(example.state_0.particle_q, example.sim_dt)
            native = example.rigid_solver
            coupler = NeedleTissueCoupler(
                example.model,
                example.solver,
                native,
                example.interaction,
                example.needle_body,
                example.model.body_com.numpy()[example.needle_body],
                absolute_tolerance=2e-9,
                relative_tolerance=1e-6,
                max_iterations=160,
                relaxation=1.0,
                compact_rigid=False,
            )
            coupler._host_wrench_enabled = True
            input_before = _snapshot(example.state_0)
            native_before = native_snapshot(native)
            output_before = _snapshot(example.state_1)
            load_before = coupler._load.numpy().copy()
            with patch.object(example.interaction, "check_step_resolution", side_effect=RuntimeError("late rejection")):
                with self.assertRaisesRegex(RuntimeError, "late rejection"):
                    coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
            transfer = native._native_transfer
            self.assertIsNotNone(transfer, "Fixture must use the actual packed native path")
            self.assertGreater(transfer._repeated_input.host_reuses, 0)
            self.assertFalse(transfer._repeated_active)
            self.assertIsNone(transfer._repeated_input.host_transaction)
            self.assertIsNone(coupler._host_wrench_transaction)
            np.testing.assert_array_equal(native_snapshot(native), native_before)
            np.testing.assert_array_equal(coupler._load.numpy(), load_before)
            for field, values in input_before.items():
                np.testing.assert_array_equal(getattr(example.state_0, field).numpy(), values)
            for field, values in output_before.items():
                np.testing.assert_array_equal(getattr(example.state_1, field).numpy(), values)
            coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
            self.assertFalse(transfer._repeated_active)
            self.assertIsNone(transfer._repeated_input.host_transaction)


if __name__ == "__main__":
    unittest.main()
