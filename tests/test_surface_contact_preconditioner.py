"""Independent dense-Jacobian oracles and transactional real-contact checks."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from test_needle_tissue_coupler import _native_state
from test_surface_anderson_coupling import _contact, _snapshot

from newton_solver_suturebot._src.coupling import NeedleCouplingNotConverged
from newton_solver_suturebot._src.surface_contact_tangent import SurfaceTangentAnchors
from newton_solver_suturebot._src.surface_preconditioner import (
    SurfaceContactPreconditioner,
    surface_balanced_correction,
    surface_contact_action,
    surface_contact_diagonal,
    validate_recording,
)


def _skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


class SurfaceMetricOperatorTests(unittest.TestCase):
    def fixture(self):
        device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
        n = 4
        fixed = np.array([0, 1, 0, 0], np.int32)
        a = SurfaceTangentAnchors(4, device)
        a.count.assign(np.array([2], np.int32))
        nodes = np.array([[0, 1, 2], [0, -1, 3]], np.int32)
        bary = np.array([[0.2, 0.3, 0.5], [0.2, 0.4, 0.4]], np.float32)
        levers = np.array([[0.4, -0.7, 0.2], [-0.3, 0.5, 0.8]], np.float32)
        factors = np.array(
            [[[1, 0.2, 0.1], [0.1, 2, 0.4], [0.3, 0.1, 3]], [[2, 0.3, 0.5], [0.2, 1, 0.4], [0.1, 0.2, 2]]], np.float32
        )
        matrices = np.asarray([f.T @ f for f in factors], np.float32)
        for name, value in [("nodes", nodes), ("bary", bary), ("levers", levers), ("matrices", matrices)]:
            full = np.zeros_like(getattr(a, name).numpy())
            full[:2] = value
            getattr(a, name).assign(full)
        fixed_device = wp.array(fixed, dtype=int, device=device)
        valid = wp.ones(1, dtype=int, device=device)
        h = wp.array([0.004], dtype=float, device=device)
        args = [fixed_device, a.count, valid, a.nodes, a.bary, a.matrices, a.levers, h]
        free = []
        full = []
        for ids, b, lever in zip(nodes, bary, levers, strict=True):
            jf = np.zeros((3, 3 * (n + 2)))
            ja = jf.copy()
            jf[:, 3 * n : 3 * n + 3] = np.eye(3)
            ja[:, 3 * n : 3 * n + 3] = np.eye(3)
            jf[:, 3 * n + 3 :] = -_skew(lever)
            ja[:, 3 * n + 3 :] = -_skew(lever)
            for node, weight in zip(ids, b, strict=True):
                if node >= 0:
                    ja[:, 3 * node : 3 * node + 3] -= weight * np.eye(3)
                    if not fixed[node]:
                        jf[:, 3 * node : 3 * node + 3] -= weight * np.eye(3)
            free.append(jf)
            full.append(ja)
        return n, a, valid, args, free, full, matrices

    def test_action_diagonal_and_full_transpose_match_independent_dense_oracle(self):
        n, a, valid, args, free, full, matrices = self.fixture()
        device = a.count.device
        x0 = np.random.default_rng(37).normal(size=(n + 2, 3)).astype(np.float32)
        x = wp.array(x0, dtype=wp.vec3, device=device)
        out = wp.zeros_like(x)
        diagonal = wp.zeros(n + 2, dtype=wp.mat33, device=device)
        wp.launch(surface_contact_action, 4, args + [x, out], device=device)
        wp.launch(surface_contact_diagonal, 4, args + [diagonal], device=device)
        dense = sum(0.004 * j.T @ d @ j for j, d in zip(free, matrices, strict=True))
        np.testing.assert_allclose(out.numpy().reshape(-1), dense @ x0.reshape(-1), rtol=3e-6, atol=2e-8)
        expected = np.asarray([dense[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] for i in range(n + 2)])
        np.testing.assert_allclose(diagonal.numpy(), expected, rtol=3e-6, atol=2e-8)
        correction = wp.zeros(3 * (n + 2), dtype=wp.float64, device=device)
        wp.launch(surface_balanced_correction, 4, args + [x, correction], device=device)
        expected = -sum(0.004 * ja.T @ d @ jf @ x0.reshape(-1) for jf, ja, d in zip(free, full, matrices, strict=True))
        np.testing.assert_allclose(correction.numpy(), expected, rtol=3e-6, atol=2e-8)
        self.assertGreater(np.linalg.norm(correction.numpy()[3:6]), 0)  # fixed reaction retained
        np.testing.assert_array_equal(out.numpy()[1], np.zeros(3))  # fixed primal row excluded

    def test_invalid_or_overflow_recording_cannot_index_uninitialized_tail(self):
        n, a, valid, args, *_ = self.fixture()
        for count, flags in [(0, [0, 0]), (5, [1, 0]), (2, [0, 1])]:
            with self.subTest(count=count, flags=flags):
                a.count.assign(np.asarray([count], np.int32))
                a.flags.assign(np.asarray(flags, np.int32))
                valid.fill_(1)
                wp.launch(
                    validate_recording,
                    4,
                    [a.count, a.flags, a.nodes, a.bary, a.matrices, a.levers, n, valid],
                    device=a.count.device,
                )
                self.assertEqual(valid.numpy()[0], 0)
                x = wp.ones(n + 2, dtype=wp.vec3, device=a.count.device)
                out = wp.zeros_like(x)
                wp.launch(surface_contact_action, 4, args + [x, out], device=a.count.device)
                np.testing.assert_array_equal(out.numpy(), np.zeros((n + 2, 3)))


def enable(coupler, capacity=4096):
    pc = SurfaceContactPreconditioner(coupler, capacity)
    coupler._surface_preconditioner = pc
    coupler._direction = wp.empty_like(coupler._residual)
    return pc


class SurfaceMetricContactTests(unittest.TestCase):
    def test_native_first_contact_matches_same_exact_residual_tolerances(self):
        records = []
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                _contact(absolute_tolerance=2e-9, relative_tolerance=1e-6) as (ex, c, native, external),
            ):
                pc = enable(c) if enabled else None
                c.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
                self.assertEqual(ex.interaction.events, ())
                self.assertEqual(ex.interaction.track_count, 0)
                self.assertEqual(ex.interaction.status, "contact")
                self.assertLess(ex.interaction.force_balance_error, 1e-8)
                self.assertLess(ex.interaction.torque_balance_error, 1e-10)
                self.assertAlmostEqual(native.mj_data.time, 3.25 + ex.sim_dt, places=12)
                self.assertEqual(c.max_iterations, 160)
                self.assertEqual(ex.solver.linear_solver_tolerance, 1e-6)
                if pc is not None:
                    self.assertGreater(pc.applications, 0)
                records.append((_snapshot(ex.state_1), ex.interaction.interface_force.copy()))
        for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd"):
            np.testing.assert_allclose(records[0][0][name], records[1][0][name], rtol=2e-5, atol=3e-7, err_msg=name)
        np.testing.assert_allclose(records[0][1], records[1][1], rtol=2e-5, atol=5e-8)

    def test_overflow_uses_original_anderson_and_preserves_endpoint(self):
        records = []
        for capacity in (None, 0):
            with _contact() as (ex, c, native, external):
                pc = None if capacity is None else enable(c, capacity)
                c.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
                if pc is not None:
                    self.assertEqual(pc.applications, 0)
                    self.assertGreater(c._surface_anderson_proposals, 0)
                records.append(_snapshot(ex.state_1))
        for name in records[0]:
            np.testing.assert_array_equal(records[0][name], records[1][name])

    def test_rejection_restores_physics_and_incoming_cg_guess(self):
        with _contact(iterations=1) as (ex, c, native, external):
            pc = enable(c)
            before_in, before_out = _snapshot(ex.state_0), _snapshot(ex.state_1)
            before_native = _native_state(native)
            increment = ex.solver.velocity_increment.numpy().copy()
            loads = c._load.numpy().copy()
            with self.assertRaises(NeedleCouplingNotConverged):
                c.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
            self.assertGreater(pc.applications, 0)
            for name, value in before_in.items():
                np.testing.assert_array_equal(getattr(ex.state_0, name).numpy(), value)
            for name, value in before_out.items():
                np.testing.assert_array_equal(getattr(ex.state_1, name).numpy(), value)
            np.testing.assert_array_equal(_native_state(native), before_native)
            np.testing.assert_array_equal(ex.solver.velocity_increment.numpy(), increment)
            np.testing.assert_array_equal(c._load.numpy(), loads)
            self.assertIsNone(ex.interaction._surface_tangent_anchors)
            self.assertFalse(pc.available)

    def test_modified_force_method_and_gradient_storage_disable_recording(self):
        with _contact() as (ex, c, native, external):
            pc = enable(c)
            with patch.object(c, "_evaluate", wraps=c._evaluate):
                self.assertFalse(pc.begin_trials(ex.state_0, ex.control))
            ex.state_0.particle_f.requires_grad = True
            self.assertFalse(pc.begin_trials(ex.state_0, ex.control))

    def test_valid_overflow_valid_switches_reset_histories_and_rejection_rolls_back(self):
        from newton_solver_suturebot._src.surface_anderson import SurfaceAnderson

        with _contact(iterations=6) as (ex, c, native, _external):
            pc = enable(c)
            original = pc.pack_statistics
            reset = SurfaceAnderson.reset
            calls = 0
            resets = 0

            def packet(statistics):
                nonlocal calls
                calls += 1
                if calls == 2:
                    pc.anchors.flags.assign(np.array([1, 0], np.int32))
                return original(statistics)

            def reset_history(acceleration):
                nonlocal resets
                resets += 1
                return reset(acceleration)

            before_in, before_out = _snapshot(ex.state_0), _snapshot(ex.state_1)
            before_native = _native_state(native)
            from newton_solver_suturebot._src.insertion import NeedleStepTooLarge

            with (
                patch.object(pc, "pack_statistics", side_effect=packet),
                patch.object(SurfaceAnderson, "reset", new=reset_history),
                patch.object(c._tet_step_guard, "check", side_effect=NeedleStepTooLarge("test late rejection")),
            ):
                with self.assertRaises(NeedleStepTooLarge):
                    c.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
            self.assertGreaterEqual(calls, 3)
            self.assertGreaterEqual(pc.selection_changes, 3)
            self.assertGreaterEqual(resets, pc.selection_changes)
            for name, value in before_in.items():
                np.testing.assert_array_equal(getattr(ex.state_0, name).numpy(), value)
            for name, value in before_out.items():
                np.testing.assert_array_equal(getattr(ex.state_1, name).numpy(), value)
            np.testing.assert_array_equal(_native_state(native), before_native)

    def test_packet_preserves_original_scalars_and_rejects_invalid_recording(self):
        with _contact() as (ex, c, _native, _external):
            pc = enable(c, 0)
            original = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1], np.float64)
            c._statistics.assign(original)
            packet = pc.pack_statistics(c._statistics).numpy()
            np.testing.assert_array_equal(packet[:6], original)
            np.testing.assert_array_equal(packet[6:], [0, 0])
            pc.anchors.count.assign(np.array([3], np.int32))
            pc.anchors.flags.assign(np.array([1, 0], np.int32))
            packet = pc.pack_statistics(c._statistics).numpy()
            np.testing.assert_array_equal(packet[:6], original)
            np.testing.assert_array_equal(packet[6:], [3, 0])

    def test_articulated_contact_metric_keeps_real_grasp_and_native_budget(self):
        from test_grasp_scene import _scene

        from newton_solver_suturebot._src.coupling import NeedleTissueCoupler

        endpoints = []
        for enabled in (False, True):
            with self.subTest(enabled=enabled), _scene() as (scene, _config):
                pose = scene.plan.pose(0).copy()
                pose[2] += 0.6 * scene.geometry.thickness
                scene._initialize_pose(pose)  # Initial condition only; elapsed remains zero.
                scene.interaction.puncture_force = 1.0
                scene.solver.linear_solver_tolerance = 1e-6
                scene.solver.refresh_tangent(scene.state_0.particle_q, scene.sim_dt)
                c = NeedleTissueCoupler(
                    scene.model,
                    scene.solver,
                    scene.rigid_solver,
                    scene.interaction,
                    scene.needle_body,
                    scene.model.body_com.numpy()[scene.needle_body],
                    absolute_tolerance=2e-9,
                    relative_tolerance=1e-6,
                    max_iterations=160,
                    relaxation=1.0,
                    compact_rigid=False,
                    surface_preconditioner=enabled,
                )
                c.step(scene.state_0, scene.state_1, scene.control, scene.sim_dt)
                native = scene.rigid_solver
                self.assertEqual(native.rigid_substeps, 4)
                self.assertEqual(native.mj_model.opt.iterations, 200)
                self.assertEqual(native.mj_model.opt.ls_iterations, 50)
                self.assertEqual(native.mj_model.opt.tolerance, 1e-12)
                self.assertEqual(native.mj_model.neq, 0)
                needle_contacts = [
                    contact
                    for contact in native.mj_data.contact
                    if 1 in native.mj_model.geom_bodyid[[contact.geom1, contact.geom2]]
                ]
                self.assertGreaterEqual(len(needle_contacts), 2)
                if enabled:
                    self.assertGreater(c._surface_preconditioner.applications, 0)
                    self.assertTrue(c._surface_preconditioner._native_mass_ready)
                endpoints.append(_snapshot(scene.state_1))
        if len(endpoints) == 2:
            for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd"):
                np.testing.assert_allclose(endpoints[0][name], endpoints[1][name], rtol=3e-5, atol=5e-7, err_msg=name)


if __name__ == "__main__":
    unittest.main()
