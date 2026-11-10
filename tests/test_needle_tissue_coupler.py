# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Implicit interface convergence and rollback, independent of task planning."""

import os
import unittest
from unittest.mock import patch

import mujoco
import newton
import numpy as np
import warp as wp
from test_needle_insertion_example import _example, _prepare

from newton_solver_suturebot import NeedleTissueCoupler, NeedleTissueInteraction, boundary_faces


def _native_state(solver):
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(solver.mj_model, specification))
    mujoco.mj_getState(solver.mj_model, solver.mj_data, result, specification)
    return result


class _TranslatingTissue:
    """One analytic tissue translation mode with a known effective mass.

    This substitutes only the structural response for a closed-form contact
    regression. Real surface contact and native MuJoCo remain in the loop.
    """

    def __init__(self, model, mass):
        self.model, self.mass = model, mass

    def step(self, state_in, state_out, control, contacts, dt, *, external_forces=None):
        force = np.zeros(3) if external_forces is None else external_forces.numpy().sum(axis=0)
        velocity = state_in.particle_qd.numpy()[0].astype(float) + dt * force / self.mass
        state_out.particle_qd.assign(np.broadcast_to(velocity, (self.model.particle_count, 3)).copy())
        state_out.particle_q.assign(state_in.particle_q.numpy() + dt * velocity)


class NeedleTissueCouplerTests(unittest.TestCase):
    def _require_cuda(self):
        if not wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda:
            self.skipTest("CUDA graph parity requires a CUDA test device")

    def test_requested_capture_keeps_custom_structural_solver_eager(self):
        with _example() as example:
            _prepare(example)
            example.start()
            structural = _TranslatingTissue(example.model, 1e-5)
            coupler = NeedleTissueCoupler(
                example.model,
                structural,
                example.rigid_solver,
                example.interaction,
                example.needle_body,
                example.drive.com,
                capture_fem=True,
            )
            with patch.object(structural, "step", wraps=structural.step) as step:
                coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
                self.assertEqual(step.call_count, coupler.iterations)
            self.assertEqual((coupler.fem_graph_captures, coupler.fem_graph_launches), (0, 0))

    def test_fem_graph_tracks_tangent_state_storage_numerical_settings_and_timestep(self):
        self._require_cuda()
        with _example() as example:
            _prepare(example)
            example.start()
            coupler, solver = example.coupler, example.solver
            solver.refresh_tangent(example.state_0.particle_q, example.sim_dt)
            expected = example.model.state()
            rng = np.random.default_rng(981)
            force = rng.normal(0, 1e-5, (example.model.particle_count, 3)).astype(np.float32)
            warmstart = rng.normal(0, 1e-8, force.shape).astype(np.float32)
            cases = (
                (0, 1.0, 80, False, False),
                (0, 1.0, 80, False, False),
                (1, 1.0, 80, True, False),
                (0, 1.0, 80, False, False),
                (0, 1.0, 80, False, False),
                (0, 1.0, 40, False, False),
                (0, 1.0, 40, False, False),
                (1, 0.5, 40, False, False),
                (1, 0.5, 40, False, False),
                (1, 0.5, 40, False, True),
                (1, 0.5, 40, False, False),
            )
            for index, (buffer, scale, budget, refresh, invalidate) in enumerate(cases):
                with self.subTest(index=index):
                    state = (example.state_0, example.state_1)[buffer]
                    positions = example.selection.vertices.copy()
                    positions[:, 0] += index * 1e-7 * np.arange(len(positions))
                    state.particle_q.assign(positions)
                    state.particle_qd.assign(rng.normal(0, 1e-5, force.shape).astype(np.float32))
                    solver.linear_solver_maxiter = budget
                    if refresh:
                        solver.refresh_tangent(state.particle_q, example.sim_dt)
                    if index == 3:
                        materials = example.model.tet_materials.numpy()
                        materials[:, :2] *= 1.05
                        example.model.tet_materials.assign(materials)
                        solver.refresh_topology(state.particle_q, example.sim_dt)
                    if invalidate:
                        coupler.invalidate_fem_graph()
                    coupler._forces.assign(force * (index + 1))
                    dt = scale * example.sim_dt
                    solver.velocity_increment.assign(warmstart)
                    expected.assign(state)
                    solver.step(state, expected, example.control, None, dt, external_forces=coupler._forces)
                    expected_increment = solver.velocity_increment.numpy().copy()
                    solver.velocity_increment.assign(warmstart)
                    coupler._candidate.assign(state)
                    coupler._step_fem(state, example.control, dt)
                    for name in ("particle_q", "particle_qd"):
                        np.testing.assert_allclose(
                            getattr(coupler._candidate, name).numpy(),
                            getattr(expected, name).numpy(),
                            rtol=1e-6,
                            atol=2e-9,
                            err_msg=name,
                        )
                    np.testing.assert_allclose(
                        solver.velocity_increment.numpy(), expected_increment, rtol=1e-6, atol=2e-9
                    )
                    self.assertEqual(solver.linear_solver_maxiter, budget)
                    self.assertEqual(solver.linear_solver_check_every, 0)
            self.assertGreaterEqual(coupler.fem_graph_captures, 5)
            self.assertGreater(coupler.fem_graph_launches, coupler.fem_graph_captures)

    def test_requested_capture_leaves_host_residual_checks_eager(self):
        self._require_cuda()
        with _example() as example:
            _prepare(example)
            example.start()
            example.solver.linear_solver_check_every = 1
            with patch.object(wp, "ScopedCapture", side_effect=AssertionError("unexpected capture")):
                for _ in range(2):
                    example.coupler._candidate.assign(example.state_0)
                    example.coupler._step_fem(example.state_0, example.control, example.sim_dt)
            self.assertEqual(example.solver.linear_solver_check_every, 1)
            self.assertEqual((example.coupler.fem_graph_captures, example.coupler.fem_graph_launches), (0, 0))
            self.assertIn("host residual checks", example.coupler.fem_graph_status)

    def _graph_contact_sequence(self, capture, *, check_failure=False):
        records = []
        with _example() as example:
            _prepare(example)
            example.start()
            example.coupler.capture_fem = capture
            pose = example.plan.pose(0.0).copy()
            pose[2] += 0.6 * example.geometry.thickness
            example._initialize_pose(pose)
            example.interaction.puncture_force = 1.0
            external = np.array([[0.00001, 0, -0.0002, 0, 0, 1e-8]], np.float32)
            example.solver.refresh_tangent(example.state_0.particle_q, example.sim_dt)
            names = ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")
            for _ in range(5):
                example.state_0.body_f.assign(external)
                example.coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
                records.append(
                    (
                        {name: getattr(example.state_1, name).numpy().copy() for name in names},
                        example.coupler.iterations,
                        example.interaction.interface_force.copy(),
                        _native_state(example.rigid_solver),
                    )
                )
                np.testing.assert_array_equal(example.state_0.body_f.numpy(), external)
                np.testing.assert_array_equal(example.state_1.body_f.numpy(), external)
                self.assertEqual(example.interaction.events, ())
                example.state_0, example.state_1 = example.state_1, example.state_0
            self.assertEqual(example.solver.linear_solver_maxiter, example.args.solver_iterations)
            if capture:
                self.assertGreater(example.coupler.fem_graph_captures, 0)
                self.assertGreater(example.coupler.fem_graph_launches, example.coupler.fem_graph_captures)
            if check_failure:
                example.coupler.max_iterations = 1
                example.coupler.absolute_tolerance = 1e-14
                example.coupler.relative_tolerance = 1e-12
                external[0, 2] -= 0.002
                example.state_0.body_f.assign(external)
                native = _native_state(example.rigid_solver)
                increment = example.solver.velocity_increment.numpy().copy()
                accepted = {name: getattr(example.state_0, name).numpy().copy() for name in names}
                launches = example.coupler.fem_graph_launches
                with patch.object(example.interaction, "accept", wraps=example.interaction.accept) as accept:
                    with self.assertRaisesRegex(RuntimeError, "did not converge"):
                        example.coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
                    self.assertEqual(accept.call_count, 0)
                self.assertGreater(example.coupler.fem_graph_launches, launches)
                np.testing.assert_array_equal(_native_state(example.rigid_solver), native)
                np.testing.assert_array_equal(example.solver.velocity_increment.numpy(), increment)
                np.testing.assert_array_equal(example.state_0.body_f.numpy(), external)
                for name, values in accepted.items():
                    np.testing.assert_array_equal(getattr(example.state_0, name).numpy(), values)
        return records

    def test_captured_contact_preserves_trajectory_interface_iterations_and_native_state(self):
        self._require_cuda()
        eager = self._graph_contact_sequence(False)
        captured = self._graph_contact_sequence(True)
        for (states_a, iterations_a, force_a, native_a), (states_b, iterations_b, force_b, native_b) in zip(
            eager, captured, strict=True
        ):
            self.assertEqual(iterations_a, iterations_b)
            for name in states_a:
                np.testing.assert_allclose(states_a[name], states_b[name], rtol=1e-6, atol=2e-9, err_msg=name)
            np.testing.assert_allclose(force_a, force_b, rtol=1e-6, atol=2e-9)
            np.testing.assert_allclose(native_a, native_b, rtol=1e-6, atol=2e-9)

    def test_captured_nonconvergence_restores_warmstart_and_native_state(self):
        self._require_cuda()
        self._graph_contact_sequence(True, check_failure=True)

    def test_implicit_normal_contact_matches_closed_form_and_advances_native_time_once(self):
        for initial_gap in (0.0, 1.5e-6):
            with self.subTest(initial_gap=initial_gap):
                self._check_implicit_normal_case(initial_gap)

    def _check_implicit_normal_case(self, initial_gap):
        with _example() as example:
            # This analytic contact case has no PD controller: its rigid
            # mobility is 1/m, without the example's native drive damping.
            example.model.joint_damping.zero_()
            example.rigid_solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
            radius, stiffness, damping, tissue_mass = 0.0005, 20.0, 0.025, 1e-5
            body = example.needle_body
            com = example.drive.com
            pose = np.r_[np.array([0, 0, radius + initial_gap]) - com, 0, 0, 0, 1]
            example._initialize_pose(pose)
            velocity = example.state_0.body_qd.numpy()
            velocity[body, 2] = -0.003
            example.state_0.body_qd.assign(velocity)
            newton.eval_ik(example.model, example.state_0, example.state_0.joint_q, example.state_0.joint_qd)
            samples = np.asarray([com, com + [0, 0, 0.01]])
            interaction = NeedleTissueInteraction(
                example.selection.vertices,
                example.selection.indices,
                example.selection.exterior_surface_triangles,
                [],
                samples,
                radius,
                samples[0],
                device=example.device,
                stiffness=stiffness,
                damping=damping,
                puncture_force=1.0,
                friction=0,
            )
            structural = _TranslatingTissue(example.model, tissue_mass)
            coupler = NeedleTissueCoupler(
                example.model,
                structural,
                example.rigid_solver,
                interaction,
                body,
                com,
                absolute_tolerance=1e-9,
                relative_tolerance=1e-5,
            )
            dt = example.sim_dt
            rigid_mass = float(example.model.body_mass.numpy()[body])
            rigid_drive, tissue_drive = -0.0002, 0.000003
            external_wrench = np.zeros((example.model.body_count, 6), np.float32)
            external_wrench[body, 2] = rigid_drive
            example.state_0.body_f.assign(external_wrench)
            nodal = np.zeros((example.model.particle_count, 3), np.float32)
            nodal[:, 2] = tissue_drive / example.model.particle_count
            external_nodal = wp.array(nodal, dtype=wp.vec3, device=example.device)
            example.rigid_solver.mj_data.time = 7.25
            with patch.object(interaction, "accept", wraps=interaction.accept) as accept:
                coupler.step(example.state_0, example.state_1, example.control, dt, external_forces=external_nodal)
                self.assertEqual(accept.call_count, 1)
            relative_free = -0.003 + dt * (rigid_drive / rigid_mass - tissue_drive / tissue_mass)
            mobility = 1 / rigid_mass + 1 / tissue_mass
            # delta = delta_free - h² W f and approach speed u = -v_free - h W f
            # make f = delta * (k + c*u/r) a positive quadratic contact law.
            free_penetration = -initial_gap - dt * relative_free
            compliance, response = dt * dt * mobility, dt * mobility
            rate = damping / radius
            constant = free_penetration * (stiffness - rate * relative_free)
            linear = 1 + compliance * (stiffness - rate * relative_free) + rate * response * free_penetration
            quadratic = compliance * rate * response
            expected = 2 * constant / (linear + np.sqrt(linear * linear - 4 * quadratic * constant))
            self.assertAlmostEqual(example.rigid_solver.mj_data.time, 7.25 + dt, places=12)
            self.assertGreater(coupler.iterations, 1)
            self.assertLess(coupler.residual, 1e-8)
            self.assertGreaterEqual(coupler.evaluations, coupler.iterations)
            self.assertAlmostEqual(interaction.last_tip_force, expected, delta=5e-8)
            rigid_velocity = example.state_1.body_qd.numpy()[body, 2]
            tissue_velocity = example.state_1.particle_qd.numpy()[0, 2]
            self.assertAlmostEqual(rigid_velocity, -0.003 + dt * (rigid_drive + expected) / rigid_mass, delta=2e-7)
            self.assertAlmostEqual(tissue_velocity, dt * (tissue_drive - expected) / tissue_mass, delta=3e-6)
            self.assertLess(rigid_velocity - tissue_velocity, 0)  # approach damping branch used in the analytic law
            np.testing.assert_array_equal(example.state_0.body_f.numpy(), external_wrench)
            np.testing.assert_array_equal(example.state_1.body_f.numpy(), external_wrench)
            np.testing.assert_array_equal(external_nodal.numpy(), nodal)
            self.assertEqual(interaction.events, ())
            self.assertEqual((interaction.puncture_count, interaction.track_count), (0, 0))

    def test_actual_fem_contact_trials_commit_once_and_preserve_external_drive(self):
        with _example() as example:
            _prepare(example)
            example.start()
            pose = example.plan.pose(0.0).copy()
            pose[2] += 0.6 * example.geometry.thickness
            example._initialize_pose(pose)
            example.interaction.puncture_force = 1.0
            external = np.zeros((example.model.body_count, 6), np.float32)
            external[example.needle_body] = (0.00001, 0, -0.0002, 0, 0, 1e-8)
            example.state_0.body_f.assign(external)
            coupler = NeedleTissueCoupler(
                example.model,
                example.solver,
                example.rigid_solver,
                example.interaction,
                example.needle_body,
                example.drive.com,
            )
            original = example.interaction.add_forces
            trials = []

            def checked_forces(*args, **kwargs):
                original(*args, **kwargs)
                if kwargs.get("trial"):
                    trials.append(example.interaction.interface_force.copy())
                    self.assertEqual((example.interaction.puncture_count, example.interaction.track_count), (0, 0))
                    self.assertEqual(example.interaction.events, ())

            time = float(example.rigid_solver.mj_data.time)
            with (
                patch.object(example.interaction, "add_forces", side_effect=checked_forces),
                patch.object(example.interaction, "accept", wraps=example.interaction.accept) as accept,
            ):
                coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
                self.assertEqual(accept.call_count, 1)
            self.assertGreater(len(trials), 1)
            self.assertEqual(example.interaction.status, "contact")
            self.assertGreater(example.interaction.last_tip_force, 0)
            self.assertLess(example.interaction.force_balance_error, 1e-8)
            self.assertLess(example.interaction.torque_balance_error, 1e-10)
            self.assertAlmostEqual(example.rigid_solver.mj_data.time, time + example.sim_dt, places=12)
            np.testing.assert_array_equal(example.state_0.body_f.numpy(), external)
            np.testing.assert_array_equal(example.state_1.body_f.numpy(), external)
            self.assertGreater(np.linalg.norm(example.state_1.particle_qd.numpy()), 0)
            self.assertGreater(np.linalg.norm(example.state_1.body_qd.numpy()), 0)

    def test_lateral_friction_contact_converges_and_applied_loads_balance(self):
        with _example() as example:
            _prepare(example)
            pose = example.plan.pose(0.0).copy()
            pose[2] += 0.5 * example.geometry.thickness
            example._initialize_pose(pose)
            example.start()
            external = np.array([[0, 0.002, 0, 0, 0, 0.000003]], np.float32)
            example.state_0.body_f.assign(external)
            applied = {}
            fem_step = example.solver.step
            rigid_step = example.rigid_solver.step

            def capture_fem(state_in, state_out, control, contacts, dt, **kwargs):
                applied["nodes"] = kwargs["external_forces"].numpy().astype(float)
                return fem_step(state_in, state_out, control, contacts, dt, **kwargs)

            def capture_rigid(state_in, state_out, control, contacts, dt, **kwargs):
                applied["body"] = state_in.body_f.numpy()[0].astype(float) - external[0]
                return rigid_step(state_in, state_out, control, contacts, dt, **kwargs)

            with (
                patch.object(example.solver, "step", side_effect=capture_fem),
                patch.object(example.rigid_solver, "step", side_effect=capture_rigid),
            ):
                example.coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
            self.assertGreater(example.coupler.iterations, 1)
            self.assertLessEqual(example.coupler.iterations, example.coupler.max_iterations)
            self.assertEqual(example.interaction.status, "contact")
            self.assertGreater(np.linalg.norm(applied["body"][:2]), 1e-5)
            # Check the forces actually passed to both solvers. Re-evaluating
            # the law at the endpoint alone would hide mixing/COM errors.
            pose = wp.transform(*example.state_1.body_q.numpy()[0])
            center = np.asarray(wp.transform_point(pose, wp.vec3(*example.drive.com)), dtype=float)
            positions = example.state_1.particle_q.numpy().astype(float)
            force_balance = applied["body"][:3] + applied["nodes"].sum(axis=0)
            moment_balance = (
                applied["body"][3:]
                + np.cross(center, applied["body"][:3])
                + np.cross(positions, applied["nodes"]).sum(axis=0)
            )
            self.assertLess(np.linalg.norm(force_balance), 1e-8)
            self.assertLess(np.linalg.norm(moment_balance), 1e-9)
            np.testing.assert_array_equal(example.state_0.body_f.numpy(), external)

    def test_nonconvergence_restores_native_state_and_does_not_commit_interaction(self):
        with _example() as example:
            _prepare(example)
            example.start()
            pose = example.plan.pose(0.0).copy()
            pose[2] += 0.6 * example.geometry.thickness
            example._initialize_pose(pose)
            example.state_0.body_f.assign(np.array([[0, 0, -0.0002, 0, 0, 0]], np.float32))
            example.rigid_solver.mj_data.time = 3.25
            native = _native_state(example.rigid_solver)
            increment = np.full_like(example.solver.velocity_increment.numpy(), 0.001)
            example.solver.velocity_increment.assign(increment)
            names = ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd", "body_f")
            initial = {name: getattr(example.state_0, name).numpy().copy() for name in names}
            status = example.interaction.status
            coupler = NeedleTissueCoupler(
                example.model,
                example.solver,
                example.rigid_solver,
                example.interaction,
                example.needle_body,
                example.drive.com,
                max_iterations=1,
                absolute_tolerance=1e-14,
                relative_tolerance=1e-12,
            )
            with patch.object(example.interaction, "accept", wraps=example.interaction.accept) as accept:
                with self.assertRaises(RuntimeError):
                    coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
                self.assertEqual(accept.call_count, 0)
            np.testing.assert_array_equal(_native_state(example.rigid_solver), native)
            np.testing.assert_array_equal(example.solver.velocity_increment.numpy(), increment)
            for name, values in initial.items():
                np.testing.assert_array_equal(getattr(example.state_0, name).numpy(), values, err_msg=name)
            self.assertEqual(example.interaction.status, status)
            self.assertEqual((example.interaction.puncture_count, example.interaction.track_count), (0, 0))
            self.assertEqual(example.interaction.events, ())
            self.assertIsNone(example.interaction._pending_opening)
            self.assertIsNone(example.interaction._step_start_tip)

    def test_sparse_native_state_updates_are_rejected_before_trials(self):
        with _example() as example:
            _prepare(example)
            example.start()
            with patch.object(example.rigid_solver, "update_data_interval", 2):
                with self.assertRaisesRegex(ValueError, "update_data_interval=1"):
                    NeedleTissueCoupler(
                        example.model,
                        example.solver,
                        example.rigid_solver,
                        example.interaction,
                        example.needle_body,
                        example.drive.com,
                    )

    def test_withdrawal_and_offset_reentry_commit_engagement_after_convergence(self):
        with _example() as example:
            # The prescribed initial velocity must cross the old opening
            # ballistically; this interaction-only case never applies the PD.
            example.model.joint_damping.zero_()
            example.rigid_solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
            body, com = example.needle_body, example.drive.com
            samples = np.asarray([com, com + [0, 0, 0.01]])
            interaction = NeedleTissueInteraction(
                example.selection.vertices,
                example.selection.indices,
                example.selection.exterior_surface_triangles,
                [],
                samples,
                0.0005,
                samples[0],
                device=example.device,
                stiffness=80,
                damping=0,
                friction=0.15,
                shaft_damping_per_m=0,
                track_spacing=0.0003,
            )
            nodal = wp.zeros(example.model.particle_count, dtype=wp.vec3, device=example.device)
            current = np.array([0.0, 0.0, 0.0001])

            def pose(position):
                example._initialize_pose(np.r_[position - com, 0, 0, 0, 1])

            def advance(x, z):
                nonlocal current
                target = np.array([x, 0, z])
                count = max(1, int(np.ceil(np.linalg.norm(target - current) / 0.0001)))
                for fraction in np.linspace(0, 1, count + 1)[1:]:
                    example.state_0.body_f.zero_()
                    nodal.zero_()
                    interaction.add_forces(example.state_0, body, com, nodal, example.sim_dt)
                    pose(current + fraction * (target - current))
                    interaction.accept(example.state_0, body)
                    self.assertNotEqual(interaction.status, "failed", interaction.reason)
                current = target

            # Public accepted probes construct a real traversed track, withdraw
            # it completely, and return through its opening at a lateral offset.
            pose(current)
            advance(0, 0.0001)
            advance(0, -0.0012)
            count, length = interaction.track_count, interaction.track_length
            advance(0, 0.0007)
            advance(0.0002, 0.0007)
            advance(0.0002, 1.5e-6)
            self.assertEqual((interaction.entry_count, interaction.exit_count, interaction.puncture_count), (1, 1, 1))
            velocity = example.state_0.body_qd.numpy()
            velocity[body, 2] = -0.003
            example.state_0.body_qd.assign(velocity)
            newton.eval_ik(example.model, example.state_0, example.state_0.joint_q, example.state_0.joint_qd)
            example.state_0.body_f.zero_()
            coupler = NeedleTissueCoupler(
                example.model,
                _TranslatingTissue(example.model, 1e-5),
                example.rigid_solver,
                interaction,
                body,
                com,
                absolute_tolerance=1e-9,
                relative_tolerance=1e-5,
            )
            # The free endpoint just crosses the old first track sample.
            # Candidate-time activation previously alternated zero/full load
            # and failed even after 200 trials. The active set is now accepted
            # history: no spring this step, resumed resistance on the next one.
            coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
            np.testing.assert_allclose(interaction.interface_force, 0, atol=1e-10)
            self.assertEqual(interaction.status, "inserted")
            self.assertEqual((interaction.entry_count, interaction.exit_count, interaction.puncture_count), (2, 1, 1))
            self.assertTrue(next(event for event in interaction.events if event["kind"] == "entry")["reused"])
            self.assertEqual(interaction.track_count, count)
            self.assertAlmostEqual(interaction.track_length, length)
            example.state_0.assign(example.state_1)
            coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
            self.assertLess(interaction.interface_force[0], -1e-5)
            self.assertGreater(interaction.interface_force[2], 0)
            self.assertLess(interaction.force_balance_error, 1e-9)
            self.assertLess(interaction.torque_balance_error, 1e-10)
            self.assertLessEqual(coupler.iterations, coupler.max_iterations)
            # Reverse the actual rigid velocity and cross the same finite-shaft
            # endpoint outward. The last covered step still reacts, then the
            # next accepted step is force-free without a trial-time force jump.
            example.state_0.assign(example.state_1)
            velocity = example.state_0.body_qd.numpy()
            velocity[body, 2] = 0.003
            example.state_0.body_qd.assign(velocity)
            newton.eval_ik(example.model, example.state_0, example.state_0.joint_q, example.state_0.joint_qd)
            coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
            self.assertEqual(interaction.status, "outside")
            self.assertEqual(interaction.exit_count, 2)
            self.assertLess(interaction.interface_force[2], 0)
            example.state_0.assign(example.state_1)
            coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
            np.testing.assert_allclose(interaction.interface_force, 0, atol=1e-10)
            self.assertEqual(interaction.track_count, count)
            self.assertAlmostEqual(interaction.track_length, length)

    def test_leading_tip_crosses_internal_tetrahedron_face_in_coupled_step(self):
        with _example() as example:
            # This is a free constant-velocity crossing, without a PD drive.
            # Remove the example's native drive damping just as in the analytic
            # contact test; otherwise its stronger grasp stops before the face.
            example.model.joint_damping.zero_()
            example.rigid_solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
            vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01], [0.01, 0.01, 0.01]])
            tetrahedra = np.array([[0, 1, 2, 3], [4, 1, 3, 2]], dtype=np.int32)
            # The analytic structural mode needs only the first five particles;
            # the remaining example particles carry no interface loads.
            positions = example.state_0.particle_q.numpy()
            positions[:5] = vertices
            example.state_0.particle_q.assign(positions)
            body, com = example.needle_body, example.drive.com
            samples = np.asarray([com, com + [0, 0, -0.01]])
            interaction = NeedleTissueInteraction(
                vertices,
                tetrahedra,
                boundary_faces(tetrahedra)[0],
                [],
                samples,
                0.0005,
                samples[0],
                device=example.device,
                damping=0,
                shaft_damping_per_m=0,
                friction=0,
            )
            nodal = wp.zeros(example.model.particle_count, dtype=wp.vec3, device=example.device)

            def pose(z):
                example._initialize_pose(np.r_[np.array([0.002, 0.002, z]) - com, 0, 0, 0, 1])

            pose(-0.0001)
            for z in np.linspace(-0.0001, 0.006 - 1e-6, 63):
                example.state_0.body_f.zero_()
                nodal.zero_()
                interaction.add_forces(example.state_0, body, com, nodal, example.sim_dt)
                pose(z)
                interaction.accept(example.state_0, body)
                self.assertNotEqual(interaction.status, "failed", interaction.reason)
            self.assertEqual((interaction.entry_count, interaction.puncture_count), (1, 1))
            velocity = example.state_0.body_qd.numpy()
            velocity[body, 2] = 0.003
            example.state_0.body_qd.assign(velocity)
            newton.eval_ik(example.model, example.state_0, example.state_0.joint_q, example.state_0.joint_qd)
            example.state_0.body_f.zero_()
            coupler = NeedleTissueCoupler(
                example.model,
                _TranslatingTissue(example.model, 1e-5),
                example.rigid_solver,
                interaction,
                body,
                com,
                absolute_tolerance=1e-9,
                relative_tolerance=1e-5,
            )
            coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
            self.assertGreater(interaction.tip_position[2], 0.006)
            self.assertAlmostEqual(interaction.tip_position[2], 0.006 - 1e-6 + 0.003 * example.sim_dt, delta=1e-8)
            self.assertEqual(interaction.status, "inserted")
            self.assertEqual(interaction.last_tip_force, 0)
            np.testing.assert_allclose(interaction.interface_force, 0, atol=2e-8)
            np.testing.assert_allclose(example.state_1.particle_q.numpy(), positions, atol=1e-9)
            self.assertEqual((interaction.entry_count, interaction.exit_count, interaction.puncture_count), (1, 0, 1))


if __name__ == "__main__":
    unittest.main()
