# SPDX-License-Identifier: Apache-2.0
"""The joint solver preserves the physical law and accepted-step transaction."""

import os
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import numpy as np
import test_fem_prepared_step as prepared_tests
import warp as wp
from test_needle_coupling_graph import _DT, _FIELDS, _configure, _example, _snapshot
from test_needle_shaft_preconditioner import history

from newton_solver_suturebot import NeedleCouplingNotConverged, NeedleStepTooLarge, SolverMuJoCoFEM
from newton_solver_suturebot._src.solvers.fem.prepared_step import PreparedFEMStep
from newton_solver_suturebot._src.solvers.mujoco_fem import control, kernels
from newton_solver_suturebot._src.solvers.mujoco_fem.trials import _SimultaneousTrials


@contextmanager
def _pair(mode):
    with ExitStack() as stack:
        reference = stack.enter_context(_example())
        joint = stack.enter_context(_example())
        _configure(reference, mode, False)
        _configure(joint, mode, True)
        previous = joint.coupler
        joint.coupler = SolverMuJoCoFEM(
            joint.model,
            joint.solver,
            joint.rigid_solver,
            joint.interaction,
            joint.needle_body,
            joint.drive.com,
            absolute_tolerance=previous.absolute_tolerance,
            relative_tolerance=previous.relative_tolerance,
            max_iterations=previous.max_iterations,
            relaxation=previous.relaxation,
            moment_length=previous.moment_length,
            capture_fem=True,
            capture_coupling=True,
            shaft_preconditioner=True,
        )
        yield reference, joint


class TissueMomentumCertificationTests(unittest.TestCase):
    def test_newton_and_certificate_measure_the_same_exact_rhs_and_free_rows(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        # The fixed row must not dominate either norm. Several weak free-row
        # defects must accumulate in L2 rather than disappear behind the
        # componentwise coupling force threshold.
        rhs_np = np.array([[1e8, -1e8, 1e8], [1e-5, 2e-5, -1e-5], [-2e-5, 1e-5, 1e-5]], np.float32)
        action_np = rhs_np.copy()
        action_np[0] = 0
        action_np[1:] += np.float32(1e-8)
        action, rhs = (wp.array(x, dtype=wp.vec3, device=device) for x in (action_np, rhs_np))
        fixed = wp.array([1, 0, 0], dtype=int, device=device)
        unknown = wp.zeros(5, dtype=wp.vec3, device=device)
        residual = wp.empty_like(unknown)
        force = wp.zeros(3, dtype=wp.vec3, device=device)
        wrench = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        blocks = wp.zeros(2, dtype=wp.mat33, device=device)
        rigid_rhs = wp.zeros(2, dtype=wp.vec3, device=device)
        stats = wp.zeros(6, dtype=wp.float64, device=device)
        fem = wp.zeros(3, dtype=wp.float64, device=device)
        dt = 0.01
        wp.launch(
            kernels.momentum_residual,
            5,
            [action, rhs, force, wrench, fixed, unknown, blocks, rigid_rhs, dt, 0.01, residual, stats, fem],
            device=device,
        )
        impulse = action_np[1:].astype(float) - rhs_np[1:].astype(float)
        expected = [np.sum(impulse**2), np.sum(rhs_np[1:].astype(float) ** 2), 0]
        np.testing.assert_allclose(fem.numpy(), expected, rtol=1e-13, atol=0)
        np.testing.assert_array_equal(residual.numpy()[[0, 3, 4]], 0)
        np.testing.assert_allclose(residual.numpy()[1:3], impulse / float(np.float32(dt)), rtol=1e-7)
        certificate = wp.zeros_like(fem)
        error = wp.zeros(1, dtype=wp.float64, device=device)
        wp.launch(kernels.momentum_defect, 3, [action, rhs, fixed, dt, error, stats, certificate], device=device)
        np.testing.assert_array_equal(certificate.numpy(), fem.numpy())

    def test_exact_prepared_equation_defect_is_measured_in_force_units_without_changing_loads(self):
        model, state, make_solver = prepared_tests.TestPreparedFEMStep().fixture()
        solver = make_solver()
        for dt in (0.002, 0.000731):
            with self.subTest(dt=dt):
                solver.refresh_tangent(state.particle_q, dt)
                prepared = solver.prepare_step(state, None, None, dt)
                external_np = np.linspace(-0.02, 0.03, model.particle_count * 3).reshape(-1, 3).astype(np.float32)
                external = wp.array(external_np, dtype=wp.vec3, device=model.device)
                increment_np = np.linspace(-0.001, 0.002, model.particle_count * 3).reshape(-1, 3).astype(np.float32)
                fixed = prepared.fixed_mask.numpy().astype(bool)
                increment_np[fixed] = 0
                increment = wp.array(increment_np, dtype=wp.vec3, device=model.device)
                action = wp.empty_like(increment)
                with patch.object(prepared, "solve", side_effect=AssertionError("Certification does not solve FEM")):
                    rhs = prepared.rhs(external)
                    prepared.apply_operator(increment, action)
                    error = wp.zeros(1, dtype=wp.float64, device=model.device)
                    statistics = wp.zeros(6, dtype=wp.float64, device=model.device)
                    fem_statistics = wp.zeros(3, dtype=wp.float64, device=model.device)
                    wp.launch(
                        kernels.momentum_defect,
                        model.particle_count,
                        [action, rhs, prepared.fixed_mask, dt, error, statistics, fem_statistics],
                        device=model.device,
                    )
                matrix = prepared_tests.dense(solver.A_bsr)
                dense_defect = (matrix @ increment_np.astype(np.float64).ravel()).reshape(-1, 3) - rhs.numpy()
                dense_error = np.abs(dense_defect[~fixed]).max() / float(np.float32(dt))
                self.assertAlmostEqual(float(error.numpy()[0]), dense_error, delta=max(1e-9, dense_error * 2e-6))
                self.assertEqual(float(statistics.numpy()[0]), float(error.numpy()[0]))
                self.assertEqual(float(statistics.numpy()[5]), 0)
                impulse = action.numpy().astype(float)[~fixed] - rhs.numpy().astype(float)[~fixed]
                np.testing.assert_allclose(
                    fem_statistics.numpy(),
                    [np.sum(impulse**2), np.sum(rhs.numpy().astype(float)[~fixed] ** 2), 0],
                    rtol=1e-12,
                    atol=0,
                )
                np.testing.assert_array_equal(external.numpy(), external_np)
                np.testing.assert_array_equal(increment.numpy(), increment_np)

    def test_fixed_reactions_are_not_momentum_rows_and_nonfinite_free_rows_fail_closed(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        action = wp.array([[np.nan, np.inf, -np.inf], [0.5, -0.25, 0.0]], dtype=wp.vec3, device=device)
        rhs = wp.zeros(2, dtype=wp.vec3, device=device)
        fixed = wp.array([1, 0], dtype=int, device=device)
        error = wp.zeros(1, dtype=wp.float64, device=device)
        statistics = wp.array([3, 0, 0, 0, 0, 0], dtype=wp.float64, device=device)
        fem_statistics = wp.zeros(3, dtype=wp.float64, device=device)
        wp.launch(
            kernels.momentum_defect, 2, [action, rhs, fixed, 0.25, error, statistics, fem_statistics], device=device
        )
        self.assertEqual(float(error.numpy()[0]), 2)
        self.assertEqual(float(statistics.numpy()[0]), 3)  # Retain the larger interface residual.
        self.assertEqual(float(statistics.numpy()[5]), 0)
        np.testing.assert_array_equal(fem_statistics.numpy(), [0.3125, 0, 0])
        action.assign(np.array([[0, 0, 0], [np.nan, 0, 0]], np.float32))
        wp.launch(
            kernels.momentum_defect, 2, [action, rhs, fixed, 0.25, error, statistics, fem_statistics], device=device
        )
        self.assertEqual(float(statistics.numpy()[5]), 1)
        self.assertEqual(float(fem_statistics.numpy()[2]), 1)


class MuJoCoFEMSolverTests(unittest.TestCase):
    def setUp(self):
        if not wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda:
            self.skipTest("The simultaneous MuJoCo/FEM backend requires CUDA")

    def assert_rollback(self, example, saved):
        current = _snapshot(example)
        for group in ("input", "output"):
            for name, values in saved[group].items():
                np.testing.assert_array_equal(current[group][name], values, err_msg=f"rollback {group}.{name}")
        for name in ("history", "visited", "openings"):
            self.assertEqual(current[name], saved[name], name)
        for name in saved.keys() - {"input", "output", "history", "visited", "openings"}:
            np.testing.assert_array_equal(current[name], saved[name], err_msg=f"rollback {name}")
        if example.interaction.tunnel_alm is not None:
            self.assertFalse(example.interaction.tunnel_alm._open)

    def assert_spatial_velocity_equivalent(self, actual, expected, moment_length, name):
        # Free-body arrays store translation in m/s and rotation in rad/s.
        # ||delta v|| + L ||delta omega|| bounds the velocity difference at
        # any point within distance L of the reference origin, in m/s.
        actual = np.asarray(actual, dtype=float).reshape(-1, 6)
        expected = np.asarray(expected, dtype=float).reshape(-1, 6)
        difference = actual - expected
        point_error = np.linalg.norm(difference[:, :3], axis=1) + moment_length * np.linalg.norm(
            difference[:, 3:], axis=1
        )
        reference_speed = np.linalg.norm(expected[:, :3], axis=1) + moment_length * np.linalg.norm(
            expected[:, 3:], axis=1
        )
        self.assertTrue(
            np.all(point_error <= 2e-5 + 3e-6 * reference_speed),
            f"{name}: point-velocity error {point_error} m/s at radius {moment_length} m",
        )

    def assert_accepted_equivalent(self, reference, joint):
        self.assertEqual(history(reference.interaction), history(joint.interaction))
        for name in _FIELDS:
            actual = getattr(joint.state_1, name).numpy()
            expected = getattr(reference.state_1, name).numpy()
            self.assertTrue(np.isfinite(actual).all(), name)
            # Different nonlinear iterations need not reproduce numerical force
            # guesses bitwise. Require submicrometre positions, close velocities,
            # and force differences below the declared coupled force accuracy.
            if name in ("joint_qd", "body_qd"):
                self.assert_spatial_velocity_equivalent(actual, expected, joint.coupler.moment_length, name)
                continue
            if name.endswith("_f"):
                tolerance = 2 * joint.coupler.absolute_tolerance
            elif name.endswith("_qd"):
                tolerance = 2e-5
            else:
                tolerance = 5e-8
            np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=tolerance, err_msg=name)
        for example in (reference, joint):
            np.testing.assert_array_equal(
                example.state_1.particle_q.numpy()[example.fixed_vertex_indices],
                example.state_0.particle_q.numpy()[example.fixed_vertex_indices],
            )
            applied, law = example.coupler._trial_load.numpy(), example.coupler._value.numpy()
            tolerance = example.coupler.absolute_tolerance + example.coupler.relative_tolerance * max(
                np.abs(applied).max(), np.abs(law).max()
            )
            self.assertLessEqual(float(np.abs(law - applied).max()), tolerance)
            alm = example.interaction.tunnel_alm
            if alm is not None:
                self.assertFalse(alm._open)
                self.assertLessEqual(alm.normal_error, alm.tolerance)
                self.assertLessEqual(
                    alm.dual_residual,
                    example.coupler.absolute_tolerance
                    + example.coupler.relative_tolerance * float(alm._statistics.numpy()[3]),
                )
        np.testing.assert_allclose(
            joint.rigid_solver.mj_data.qpos, reference.rigid_solver.mj_data.qpos, rtol=3e-6, atol=5e-8
        )
        self.assert_spatial_velocity_equivalent(
            joint.rigid_solver.mj_data.qvel,
            reference.rigid_solver.mj_data.qvel,
            joint.coupler.moment_length,
            "native qvel",
        )
        self.assertAlmostEqual(joint.rigid_solver.mj_data.time, reference.rigid_solver.mj_data.time, places=12)
        expected, actual = reference.coupler._value.numpy(), joint.coupler._value.numpy()
        self.assertGreater(np.linalg.norm(expected), 1e-7)
        self.assertLessEqual(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 0.01)

    def step_pair(self, reference, joint, dt):
        for example in (reference, joint):
            native_time = float(example.rigid_solver.mj_data.time)
            with patch.object(example.interaction, "accept", wraps=example.interaction.accept) as accept:
                example.coupler.step(example.state_0, example.state_1, example.control, dt)
            self.assertEqual(accept.call_count, 1)
            self.assertAlmostEqual(example.rigid_solver.mj_data.time, native_time + dt, places=12)
        self.assert_accepted_equivalent(reference, joint)

    def test_same_law_replay_with_changed_load_timestep_and_alternating_buffers(self):
        for mode in ("penalty", "alm"):
            with self.subTest(mode=mode), _pair(mode) as (reference, joint):
                for index, dt in enumerate((_DT, _DT, _DT / 2)):
                    for example in (reference, joint):
                        example.drive.prepare_timestep(dt)
                        example.solver.refresh_tangent(example.state_0.particle_q, dt)
                        example.control.joint_f.assign(np.array([0, (-1) ** index * 1e-5, 0, 2e-8, 0, 0], np.float32))
                    self.step_pair(reference, joint, dt)
                    self.assertEqual(joint.coupler.coupling_graph_fallbacks, 0)
                    if mode == "alm":
                        self.assertEqual(joint.coupler.simultaneous_steps, index + 1)
                    else:
                        self.assertEqual(joint.coupler.simultaneous_steps, 0)
                    for example in (reference, joint):
                        example.state_0, example.state_1 = example.state_1, example.state_0
                self.assertGreaterEqual(joint.coupler.coupling_graph_launches, 3)

    def test_exhausted_budget_leaves_no_accepted_history_and_retry_matches_reference(self):
        with _pair("alm") as (reference, joint):
            saved = _snapshot(joint)
            joint.coupler.max_iterations = 1
            with patch.object(joint.interaction, "accept", wraps=joint.interaction.accept) as accept:
                with self.assertRaises(NeedleCouplingNotConverged):
                    joint.coupler.step(joint.state_0, joint.state_1, joint.control, _DT)
                self.assertEqual(accept.call_count, 0)
            self.assert_rollback(joint, saved)
            self.assertEqual(joint.coupler.simultaneous_steps, 0)
            joint.coupler.max_iterations = 160
            self.step_pair(reference, joint, _DT)
            self.assertEqual(joint.coupler.simultaneous_steps, 1)

    def test_inaccurate_final_affine_solve_rejects_rolls_back_and_clean_retry_matches(self):
        with _pair("alm") as (reference, joint):
            saved = _snapshot(joint)
            actual_solve = PreparedFEMStep.solve
            actual_launch = _SimultaneousTrials._launch
            certificate_metrics = wp.zeros(3, dtype=wp.float64, device=joint.model.device)

            def retain_certificate(graph, kernel, count, args):
                value = actual_launch(graph, kernel, count, args)
                if kernel == control.certification:
                    # Later Newton proposals reuse graph.fem_statistics;
                    # retain the actual final-correction certificate instead.
                    wp.copy(certificate_metrics, args[2])
                return value

            def inaccurate_solve(prepared, rhs, out=None, *, tolerance=None):
                # Execute the real final solve, then deliberately replace its
                # finite result. The outer certificate must independently
                # reject it instead of trusting a returned CG functor.
                value = actual_solve(prepared, rhs, out, tolerance=tolerance)
                value.zero_()
                return value

            joint.coupler.max_iterations = 160
            with (
                patch.object(PreparedFEMStep, "solve", inaccurate_solve),
                patch.object(_SimultaneousTrials, "_launch", retain_certificate),
                patch.object(joint.interaction, "accept", wraps=joint.interaction.accept) as accept,
            ):
                with self.assertRaises(NeedleCouplingNotConverged):
                    joint.coupler.step(joint.state_0, joint.state_1, joint.control, _DT)
                self.assertEqual(accept.call_count, 0)
            graph = joint.coupler._simultaneous_trials
            self.assertGreaterEqual(int(graph.final_corrections.numpy()[0]), 1)
            applied = certificate_metrics.numpy()
            self.assertTrue(np.isfinite(applied).all())
            self.assertGreater(applied[0], joint.solver.linear_solver_tolerance**2 * applied[1])
            self.assertEqual(joint.coupler.coupling_graph_fallbacks, 0)
            self.assert_rollback(joint, saved)
            # Remove the deliberately captured zeroing operation as well as
            # the Python hook before validating an ordinary clean retry.
            joint.coupler.invalidate_fem_graph()
            joint.coupler.max_iterations = 160
            self.step_pair(reference, joint, _DT)

    def test_late_resolution_rejection_restores_native_output_warmstart_and_duals(self):
        with _pair("alm") as (reference, joint):
            saved = _snapshot(joint)
            with (
                patch.object(joint.interaction, "check_step_resolution", side_effect=NeedleStepTooLarge("late guard")),
                patch.object(joint.interaction, "accept", wraps=joint.interaction.accept) as accept,
            ):
                with self.assertRaisesRegex(NeedleStepTooLarge, "late guard"):
                    joint.coupler.step(joint.state_0, joint.state_1, joint.control, _DT)
                self.assertEqual(accept.call_count, 0)
            self.assert_rollback(joint, saved)
            self.step_pair(reference, joint, _DT)
            self.assertGreater(joint.coupler.simultaneous_steps, 0)

    def test_final_affine_correction_preserves_applied_accuracy_and_balanced_loads(self):
        with _pair("alm") as (_, joint):
            joint.coupler.step(joint.state_0, joint.state_1, joint.control, _DT)
            self.assertEqual(joint.coupler.simultaneous_steps, 1)
            self.assertEqual(joint.coupler.coupling_graph_fallbacks, 0)
            graph = joint.coupler._simultaneous_trials
            self.assertGreaterEqual(int(graph.final_corrections.numpy()[0]), 1)
            self.assertLessEqual(int(graph.final_corrections.numpy()[0]), joint.coupler.iterations // 2)
            values = graph.fem_statistics.numpy()
            self.assertEqual(values[2], 0)
            self.assertLessEqual(values[0], joint.solver.linear_solver_tolerance**2 * values[1])
            self.assertTrue(np.isfinite(graph.fresh_fem_statistics.numpy()).all())
            prepared = graph.prepared
            matrix = prepared_tests.dense(prepared.unconstrained_matrix)
            increment = joint.solver.velocity_increment.numpy().astype(np.float64)
            fixed = prepared.fixed_mask.numpy().astype(bool)
            external = joint.coupler._external_nodes.numpy().astype(np.float64)
            applied, law = joint.coupler._trial_load.numpy(), joint.coupler._value.numpy()
            tolerance = joint.coupler.absolute_tolerance + joint.coupler.relative_tolerance * max(
                np.abs(applied).max(), np.abs(law).max()
            )
            action = (matrix @ increment.ravel()).reshape(-1, 3)
            for label, load in (("applied", applied), ("fresh law", law)):
                with self.subTest(load=label):
                    total = (external + load[: 3 * len(fixed)].reshape(-1, 3)).astype(np.float32)
                    rhs = prepared.rhs(wp.array(total, dtype=wp.vec3, device=joint.model.device)).numpy()
                    defect = (action - rhs.astype(np.float64)) / float(np.float32(_DT))
                    # Independent double accumulation of the float32 BSR,
                    # allowing only its sparse-matvec summation roundoff.
                    absolute_action = (np.abs(matrix) @ np.abs(increment.ravel())).reshape(-1, 3)
                    roundoff = 16 * np.finfo(np.float32).eps * float(absolute_action[~fixed].max()) / _DT
                    self.assertLessEqual(float(np.abs(defect[~fixed]).max()), tolerance + roundoff)
            self.assertLessEqual(float(graph.momentum_error.numpy()[0]), tolerance)
            np.testing.assert_array_equal(increment[fixed], 0)
            np.testing.assert_array_equal(
                joint.state_1.particle_q.numpy()[fixed], joint.state_0.particle_q.numpy()[fixed]
            )
            # Applied loads remain the physical two-way scatter, including
            # fixed-node contact forces; no dense residual force is injected.
            nodes = applied[: 3 * len(fixed)].reshape(-1, 3)
            net_force = nodes.sum(axis=0) + applied[3 * len(fixed) : 3 * len(fixed) + 3]
            self.assertLess(np.linalg.norm(net_force), 1e-8)


if __name__ == "__main__":
    unittest.main()
