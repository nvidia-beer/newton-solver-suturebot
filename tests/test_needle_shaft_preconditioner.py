# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""A shaft solve accelerator must preserve the coupled physical problem."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np
import warp as wp
from test_needle_insertion_example import _example

from newton_solver_suturebot import NeedleStepTooLarge, NeedleTissueCoupler, NeedleTissueInteraction, boundary_faces
from newton_solver_suturebot._src.solvers.coupled.acceleration.shaft_preconditioner import (
    _ShaftPreconditioner as ShaftPreconditioner,
)
from newton_solver_suturebot._src.solvers.coupled.acceleration.shaft_preconditioner import prepare_anchors
from newton_solver_suturebot.examples.needle._needle_insertion_drive import NeedleDrive


def setup(ex, h, stiffness):
    ex.select_regions([1])
    ex.select_point_xy("entry", -0.004, 0)
    ex.select_point_xy("exit", 0.004, 0)
    ex.preview()
    ex.start()
    ex.running = False
    final = 0.5 * ex.plan.exit_angle
    count = int(np.ceil((final - ex.plan.initial_angle) * ex.geometry.radius / 0.00005))
    for angle in np.linspace(ex.plan.initial_angle, final, count + 1):
        ex.state_0.clear_forces()
        ex._external_forces.zero_()
        ex.interaction.add_forces(ex.state_0, ex.needle_body, ex.drive.com, ex._external_forces, ex.sim_dt)
        ex._initialize_pose(ex.plan.pose(angle))
        ex.interaction.accept(ex.state_0, ex.needle_body)
        if ex.interaction.status == "failed":
            raise RuntimeError(ex.interaction.reason)
    # Public accepted geometric probes initialize a preloaded, embedded condition;
    # no claim is made that this setup itself was a dynamic insertion/lift.
    pose = ex.plan.pose(final)
    for shift in np.linspace(0, 0.0002, 9)[1:]:
        ex.state_0.clear_forces()
        ex._external_forces.zero_()
        ex.interaction.add_forces(ex.state_0, ex.needle_body, ex.drive.com, ex._external_forces, h)
        moved = pose.copy()
        moved[2] += shift
        ex._initialize_pose(moved)
        ex.interaction.accept(ex.state_0, ex.needle_body)
        if ex.interaction.status == "failed":
            raise RuntimeError(ex.interaction.reason)
    ex.interaction.shaft_stiffness_per_m = stiffness
    ex.model.joint_damping.zero_()
    ex.drive = NeedleDrive(ex.model, ex.needle_body, stiffness=100, angular_stiffness=0.01)
    ex.drive.configure_mujoco(ex.rigid_solver, dt=h)
    ex.solver.refresh_tangent(ex.state_0.particle_q, h)
    ex.state_0.clear_forces()
    target = pose.copy()
    target[2] += 0.0001
    ex.drive.add_forces(ex.state_0, target, np.zeros(6))


def _apply(pre, state, dt, residual):
    output = wp.empty_like(residual)
    pre.apply(state, dt, residual, output)
    return output


def _configure(ex, enabled):
    ex.coupler = NeedleTissueCoupler(
        ex.model,
        ex.solver,
        ex.rigid_solver,
        ex.interaction,
        ex.needle_body,
        ex.drive.com,
        moment_length=ex.geometry.radius,
        shaft_preconditioner=enabled,
        absolute_tolerance=2e-8,
        relative_tolerance=2e-5,
        max_iterations=160,
    )


def history(interaction):
    return (
        interaction.status,
        interaction.reason,
        interaction.puncture_count,
        interaction.entry_count,
        interaction.exit_count,
        interaction.track_count,
        interaction.track_length,
        repr(interaction.events),
    )


def append_engaged_sample(example, factor=1.2):
    """Lay one more track sample a track spacing ahead of the newest one, with an edge to it.

    Unlike a sample without an edge it starts engaged and has a quadrature length: it gets an
    anchor in the shaft correction and exerts a force. Returns the new track count.
    """
    x = example.interaction
    count = x.track_count
    rest = np.asarray(x._track_rest)
    a = x._track_anchor if x._track_anchor is not None else count - 1
    neighbours = [i if j == a else j for i, j in x._track_edges if a in (i, j)]
    b = max(neighbours) if neighbours else a - 1
    direction = (rest[a] - rest[b]) / np.linalg.norm(rest[a] - rest[b])
    for scale in (factor, 1.5 * factor, 2.0 * factor):
        target = rest[a] + scale * x.track_spacing * direction
        for nodes in x.tetrahedra:
            points = x.vertices[nodes]
            weights = np.linalg.solve((points[1:] - points[0]).T, target - points[0])
            bary = np.array([1 - weights.sum(), *weights])
            if bary.min() >= -1e-9:
                x._record_track(nodes.copy(), bary)
                break
        if x.track_count == count + 1:
            return x.track_count
    raise AssertionError("No engaged sample was recorded")


def dense_action(pre, action):
    x = wp.zeros(pre.n, dtype=wp.vec3, device=pre.device)
    zero = wp.zeros_like(x)
    out = wp.zeros_like(x)
    columns = []
    for index in range(3 * pre.n):
        basis = np.zeros((pre.n, 3), np.float32)
        basis.flat[index] = 1
        x.assign(basis)
        if action == "operator":
            pre.operator.matvec(x, zero, out, 1.0, 0.0)
        else:
            pre.mass_apply(x, out)
        columns.append(out.numpy().reshape(-1).copy())
    return np.asarray(columns, dtype=np.float64).T


def _full_contact_action(pre, velocity, dt):
    """Independent scatter includes fixed reactions but fixes their velocity."""
    interaction = pre.coupler.interaction
    fixed = np.asarray(interaction.fixed, dtype=bool)
    result = np.zeros_like(velocity)
    # The track arrays and the anchors have the rows of the track reserve; the live ones hold the samples.
    count = interaction.track_count
    for nodes, weights, matrix, lever in zip(
        interaction._track_nodes.numpy()[:count],
        interaction._track_bary.numpy()[:count].astype(float),
        pre.matrices.numpy()[:count].astype(float),
        pre.levers.numpy()[:count].astype(float),
        strict=True,
    ):
        relative = velocity[-2] + np.cross(velocity[-1], lever)
        for node, weight in zip(nodes, weights, strict=True):
            if not fixed[node]:
                relative -= weight * velocity[node]
        force = dt * matrix @ relative
        for node, weight in zip(nodes, weights, strict=True):
            result[node] -= weight * force
        result[-2] += force
        result[-1] += np.cross(lever, force)
    return result


def _balance(load, state, coupler):
    q = state.particle_q.numpy().astype(float)
    pose = state.body_q.numpy()[coupler.needle_body]
    center = np.asarray(wp.transform_point(wp.transform(*pose), wp.vec3(*coupler.needle_com)), dtype=float)
    values = np.asarray(load, dtype=float).reshape(-1, 3)
    force = values[-2]
    torque = values[-1] * coupler.moment_length
    return (
        values[: len(q)].sum(axis=0) + force,
        np.cross(q, values[: len(q)]).sum(axis=0) + np.cross(center, force) + torque,
    )


def _balanced_residual(ex, n):
    q = ex.state_0.particle_q.numpy().astype(float)
    pose = ex.state_0.body_q.numpy()[ex.needle_body]
    center = np.asarray(wp.transform_point(wp.transform(*pose), wp.vec3(*ex.drive.com)), dtype=float)
    result = np.zeros((n, 3))
    rng = np.random.default_rng(19)
    count = ex.interaction.track_count
    for nodes, weights in zip(
        ex.interaction._track_nodes.numpy()[:count],
        ex.interaction._track_bary.numpy()[:count].astype(float),
        strict=True,
    ):
        force = rng.normal(0, 1e-4, 3)
        for node, weight in zip(nodes, weights, strict=True):
            result[node] -= weight * force
        result[-2] += force
        result[-1] += np.cross(weights @ q[nodes] - center, force) / ex.coupler.moment_length
    return result.reshape(-1)


class IndependentShaftPreconditionerTests(unittest.TestCase):
    def test_runtime_friction_tangent_is_continuous_psd_without_changing_coulomb_force(self):
        # A straight embedded shaft has an analytic transverse load and axial
        # friction. Construct its track through public puncture/accept calls.
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
        tets = np.array([[0, 1, 2, 3]], np.int32)
        samples = np.zeros((11, 3))
        samples[:, 2] = np.linspace(0, -0.01, len(samples))
        interaction = NeedleTissueInteraction(
            vertices,
            tets,
            boundary_faces(tets)[0],
            (),
            samples,
            0.0005,
            samples[0],
            damping=0,
            shaft_damping_per_m=0,
            shaft_stiffness_per_m=6000,
            device=device,
        )
        state = SimpleNamespace(
            particle_q=wp.array(vertices, dtype=wp.vec3, device=device),
            particle_qd=wp.zeros(4, dtype=wp.vec3, device=device),
            body_q=wp.array([[0.002, 0.002, -0.0001, 0, 0, 0, 1]], dtype=wp.transform, device=device),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=device),
        )
        forces = wp.zeros(4, dtype=wp.vec3, device=device)
        h = 1 / 240
        interaction.add_forces(state, 0, np.zeros(3), forces, h)
        interaction.accept(state, 0)
        self.assertEqual(interaction.puncture_count, 1)
        for z in np.linspace(0, 0.0015, 16):
            state.body_q.assign(np.array([[0.002, 0.002, z, 0, 0, 0, 1]], np.float32))
            interaction.accept(state, 0)
            self.assertNotEqual(interaction.status, "failed")
        self.assertGreater(interaction.track_count, 2)
        state.body_q.assign(np.array([[0.0022, 0.002, 0.0015, 0, 0, 0, 1]], np.float32))
        count = interaction.track_count
        lengths = interaction._track_lengths.numpy()[:count].astype(float)
        nodes = interaction._track_nodes.numpy()[:count]
        bary = interaction._track_bary.numpy()[:count].astype(float)
        material = np.sum(bary[:, :, None] * state.particle_q.numpy()[nodes], axis=1)
        normal = lengths * interaction.shaft_stiffness_per_m * (float(state.body_q.numpy()[0, 0]) - material[:, 0])
        epsilon, mu = interaction.speed_scale, interaction.friction
        matrices = wp.empty(interaction.track_count, dtype=wp.mat33, device=device)
        levers = wp.empty(interaction.track_count, dtype=wp.vec3, device=device)
        committed = history(interaction)
        tangential = {}
        for factor in (-20.0, -1.0001, -1.0, -0.9999, -0.5, 0.0, 0.5, 0.9999, 1.0, 1.0001, 20.0):
            with self.subTest(slip_ratio=factor):
                state.body_qd.assign(np.array([[0, 0, factor * epsilon, 0, 0, 0]], np.float32))
                state.body_f.zero_()
                forces.zero_()
                interaction.add_forces(state, 0, np.zeros(3), forces, h, trial=True, implicit=True)
                physical, nodal = state.body_f.numpy().copy(), forces.numpy().copy()
                speed = abs(float(state.body_qd.numpy()[0, 2]))
                # Physical friction stays regularized Coulomb, including its
                # unchanged saturation. The smaller PC slope is not this force.
                expected_force = np.array([-normal.sum(), 0, -mu * normal.sum() * np.clip(factor, -1, 1)])
                np.testing.assert_allclose(physical[0, :3], expected_force, rtol=2e-5, atol=2e-8)
                wp.launch(
                    prepare_anchors,
                    interaction.track_count,
                    [
                        interaction._track_nodes,
                        interaction._track_bary,
                        interaction._track_lengths,
                        interaction._track_engaged,
                        # The kernel runs the rows of the track samples: the live counts of the interaction.
                        interaction._live,
                        state.particle_q,
                        state.particle_qd,
                        interaction._samples,
                        state.body_q,
                        state.body_qd,
                        0,
                        wp.vec3(0.0),
                        float(interaction.radii.max()),
                        interaction.shaft_stiffness_per_m,
                        0.0,
                        mu,
                        epsilon,
                        # The kernel reads the time step from a one-element device array.
                        wp.array([h], dtype=float, device=device),
                        0.01,
                        matrices,
                        levers,
                    ],
                    device=device,
                )
                actual = matrices.numpy().astype(float)
                expected = np.zeros_like(actual)
                expected[:, 0, 0] = expected[:, 1, 1] = lengths * h * interaction.shaft_stiffness_per_m
                expected[:, 2, 2] = mu * normal * epsilon / max(epsilon, speed) ** 2
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-8)
                self.assertGreaterEqual(np.linalg.eigvalsh(actual).min(), 0)
                np.testing.assert_array_equal(state.body_f.numpy(), physical)
                np.testing.assert_array_equal(forces.numpy(), nodal)
                self.assertEqual(history(interaction), committed)
                tangential[factor] = actual[:, 2, 2]
                interaction.discard_trial_forces()
        for sign in (-1.0, 1.0):
            np.testing.assert_allclose(tangential[sign * 0.9999], tangential[sign * 1.0001], rtol=3e-4)
            np.testing.assert_allclose(tangential[sign * 20], tangential[sign] / 400, rtol=2e-5)

    def test_rotated_nondiagonal_inertia_and_positive_contact_operator(self):
        for h in (1 / 60, 1 / 120, 1 / 240):
            with self.subTest(dt=h), _example() as ex:
                setup(ex, h, 6000)
                # SPD inertia expressed in the model body frame, with genuine
                # off-diagonal terms; the mass operator must rotate all terms.
                rotation = np.asarray(wp.quat_to_matrix(wp.quat(*ex.state_0.body_q.numpy()[0, 3:]))).reshape(3, 3)
                inertia = np.array([[2.2, 0.35, -0.15], [0.35, 3.1, 0.4], [-0.15, 0.4, 4.0]]) * 1e-8
                ex.model.body_inertia.assign(inertia.astype(np.float32)[None])
                ex.rigid_solver.mj_model.dof_damping[:] = np.array([0.002, 0.003, 0.004, 2e-7, 3e-7, 5e-7])
                pre = ShaftPreconditioner(ex.coupler)
                rng = np.random.default_rng(831)
                residual = rng.normal(0, 1e-5, 3 * pre.n).astype(np.float64)
                actual = _apply(pre, ex.state_0, h, wp.array(residual, dtype=wp.float64, device=ex.device)).numpy()
                operator, mass = dense_action(pre, "operator"), dense_action(pre, "mass")
                contact = operator - mass
                scale = np.linalg.norm(operator, ord=2)
                self.assertLess(np.max(np.abs(operator - operator.T)), scale * 1e-6)
                self.assertGreater(np.linalg.eigvalsh(mass).min(), 0)
                self.assertGreater(np.linalg.eigvalsh(operator).min(), 0)
                self.assertGreaterEqual(np.linalg.eigvalsh((contact + contact.T) / 2).min(), -scale * 2e-7)
                np.testing.assert_allclose(
                    np.diag(operator).reshape(-1, 3),
                    np.diagonal(pre.diagonal.numpy(), axis1=1, axis2=2),
                    atol=2e-7,
                    rtol=2e-6,
                )
                expected_rotation = (
                    rotation
                    @ (inertia + h * np.diag(ex.rigid_solver.mj_model.dof_damping[3:]))
                    @ rotation.T
                    / ex.coupler.moment_length**2
                )
                np.testing.assert_allclose(mass[-3:, -3:], expected_rotation, atol=2e-9, rtol=2e-6)
                z = np.linalg.solve(operator, residual).reshape(-1, 3)
                expected = residual.reshape(-1, 3) - _full_contact_action(pre, z, h)
                self.assertLess(np.linalg.norm(actual - expected.reshape(-1)) / np.linalg.norm(expected), 0.005)

    def test_inexact_direction_keeps_fixed_reactions_and_net_wrench(self):
        with _example() as ex:
            dt = 1 / 240
            setup(ex, dt, 6000)
            pre = ShaftPreconditioner(ex.coupler)
            nodes = ex.interaction._track_nodes.numpy()
            bary = ex.interaction._track_bary.numpy()
            fixed = np.asarray(ex.interaction.fixed, dtype=bool)
            self.assertGreater(float(np.max(np.sum(bary * fixed[nodes], axis=1))), 0.01)
            residual = _balanced_residual(ex, pre.n)
            actual = _apply(pre, ex.state_0, dt, wp.array(residual, dtype=wp.float64, device=ex.device)).numpy()
            before_force, before_moment = _balance(residual, ex.state_0, ex.coupler)
            after_force, after_moment = _balance(actual, ex.state_0, ex.coupler)
            self.assertLess(np.linalg.norm(before_force), 1e-10)
            self.assertLess(np.linalg.norm(before_moment), 1e-12)
            # These bounds concern the correction itself, even though the inner
            # Krylov solve is approximate. Copying fixed rows gives ~8 microN.
            self.assertLess(np.linalg.norm(after_force), 1e-9)
            self.assertLess(np.linalg.norm(after_moment), 1e-10)
            fixed_dofs = np.repeat(fixed, 3)
            self.assertGreater(
                np.linalg.norm(actual[: 3 * len(fixed)][fixed_dofs] - residual[: 3 * len(fixed)][fixed_dofs]), 1e-7
            )
            expected = residual.reshape(-1, 3) - _full_contact_action(pre, pre.z.numpy().astype(float), dt)
            np.testing.assert_allclose(actual, expected.reshape(-1), atol=2e-9, rtol=2e-5)

    def test_zero_contact_is_exact_identity_and_public_flag_has_fallback(self):
        with _example() as ex:
            setup(ex, 1 / 240, 6000)
            _configure(ex, False)
            self.assertIsNone(ex.coupler._shaft_preconditioner)
            self.assertEqual(ex.coupler.shaft_preconditioner_status, "disabled")
            _configure(ex, True)
            pre = ex.coupler._shaft_preconditioner
            self.assertIsNotNone(pre)
            residual = wp.array(np.linspace(-1e-5, 1e-5, 3 * pre.n), dtype=wp.float64, device=ex.device)
            ex.interaction._track_engaged.zero_()
            # Written in place from outside the interaction: the length that an accepted step kept
            # is dropped with it.
            ex.interaction._engaged_length = None
            pre.begin_step()
            actual = _apply(pre, ex.state_0, 1 / 240, residual)
            np.testing.assert_array_equal(actual.numpy(), residual.numpy())
            original_step = ex.solver.step
            ex.solver.step = lambda *args, **kwargs: original_step(*args, **kwargs)
            _configure(ex, True)
            self.assertIsNone(ex.coupler._shaft_preconditioner)
            self.assertEqual(ex.coupler.shaft_preconditioner_status, "unsupported")

    def test_write_in_place_with_the_kept_length_dropped_is_seen(self):
        # After an accepted coupled step the interaction keeps the engaged length that the step read.
        # A write of the engagement in place from outside the interaction is seen by begin_step only
        # if the writer drops that length with it (the rule at ``_engaged_length`` in insertion.py).
        with _example() as ex:
            dt = 1 / 240
            setup(ex, dt, 6000)
            _configure(ex, True)
            pre, interaction = ex.coupler._shaft_preconditioner, ex.interaction
            ex.coupler.step(ex.state_0, ex.state_1, ex.control, dt)
            ex.state_0, ex.state_1 = ex.state_1, ex.state_0
            self.assertTrue(pre.active)
            self.assertIsNotNone(interaction._engaged_length)
            self.assertGreater(interaction.engaged_track_length, 0.0)
            interaction._track_engaged.zero_()
            interaction._engaged_length = None
            pre.begin_step()
            self.assertEqual(interaction.engaged_track_length, 0.0)
            self.assertFalse(pre.active)

    def test_begin_step_takes_the_engaged_length_that_the_last_accepted_step_read(self):
        with _example() as ex:
            dt = 1 / 240
            setup(ex, dt, 6000)
            _configure(ex, True)
            pre, interaction = ex.coupler._shaft_preconditioner, ex.interaction

            def device_length():
                lengths, engaged = interaction._track_lengths.numpy(), interaction._track_engaged.numpy()
                return float(np.sum(lengths[engaged != 0]))

            def step():
                ex.coupler.step(ex.state_0, ex.state_1, ex.control, dt)
                ex.state_0, ex.state_1 = ex.state_1, ex.state_0

            accept, in_accept = interaction.accept, []
            with patch.object(
                interaction, "_read_engaged_track_length", wraps=interaction._read_engaged_track_length
            ) as read:

                def counted_accept(state, body):
                    before = read.call_count
                    accept(state, body)
                    in_accept.append(read.call_count - before)

                with patch.object(interaction, "accept", side_effect=counted_accept):
                    # The explicit probes of the setup kept no length: the first step reads it for
                    # its trials, and its acceptance reads it for the next step.
                    step()
                    self.assertEqual(in_accept, [1])
                    before_trials = read.call_count - 1
                    self.assertGreaterEqual(before_trials, 1)
                    for count in range(2, 5):
                        step()
                        # One read per accepted step, made by its acceptance; none before the trials.
                        self.assertEqual((read.call_count, in_accept), (before_trials + count, [1] * count))
                        self.assertEqual(interaction.engaged_track_length, device_length())
                        self.assertTrue(pre.active)
                    # A rejected step and a restart of the trials leave the engagement as it was.
                    with (
                        patch.object(interaction, "check_step_resolution", side_effect=NeedleStepTooLarge("late")),
                        self.assertRaisesRegex(NeedleStepTooLarge, "late"),
                    ):
                        ex.coupler.step(ex.state_0, ex.state_1, ex.control, dt)
                    pre.begin_step()
                    self.assertEqual((read.call_count, in_accept), (before_trials + 4, [1] * 4))
                    self.assertTrue(pre.active)
                    self.assertEqual(interaction.engaged_track_length, device_length())
                    step()
                    self.assertEqual((read.call_count, in_accept), (before_trials + 5, [1] * 5))
                    self.assertGreater(device_length(), 0)
            # An explicit force evaluation with the needle back at its start releases the track
            # without an accepted step; the length that the last acceptance read no longer holds.
            ex._initialize_pose(ex.plan.pose(ex.plan.initial_angle))
            ex.state_0.clear_forces()
            ex._external_forces.zero_()
            interaction.add_forces(ex.state_0, ex.needle_body, ex.drive.com, ex._external_forces, dt)
            self.assertEqual(device_length(), 0.0)
            pre.begin_step()
            self.assertFalse(pre.active)
            self.assertEqual(interaction.engaged_track_length, 0.0)

    def test_reset_starts_from_a_device_read_not_from_the_length_of_the_run_before(self):
        with _example() as ex:
            dt = 1 / 240
            setup(ex, dt, 6000)
            _configure(ex, True)
            first = ex.interaction
            ex.coupler.step(ex.state_0, ex.state_1, ex.control, dt)
            with patch.object(first, "_read_engaged_track_length", side_effect=AssertionError("not kept")):
                self.assertGreater(first.engaged_track_length, 0)
            # The setup selects its region again: the example's reset, with a new interaction.
            setup(ex, dt, 6000)
            _configure(ex, True)
            second, pre = ex.interaction, ex.coupler._shaft_preconditioner
            self.assertIsNot(second, first)
            with patch.object(second, "_read_engaged_track_length", wraps=second._read_engaged_track_length) as read:
                pre.begin_step()
                # The first reader of the flag in the step reads the device: nothing was kept for this interaction.
                self.assertEqual(read.call_count, 0)
                self.assertTrue(pre.active)
                self.assertEqual(read.call_count, 1)
            lengths, engaged = second._track_lengths.numpy(), second._track_engaged.numpy()
            self.assertEqual(second.engaged_track_length, float(np.sum(lengths[engaged != 0])))

    def test_active_is_evaluated_by_its_first_reader_after_begin_step_and_can_be_assigned(self):
        with _example() as ex:
            dt = 1 / 240
            setup(ex, dt, 6000)
            _configure(ex, True)
            pre, interaction = ex.coupler._shaft_preconditioner, ex.interaction

            def eager():
                # What begin_step evaluated before it left the flag to its first reader.
                lengths, engaged = interaction._track_lengths.numpy(), interaction._track_engaged.numpy()
                law = (
                    interaction.tunnel_alm is not None
                    or interaction.shaft_stiffness_per_m > 0
                    or interaction.shaft_damping_per_m > 0
                )
                return bool(law and float(np.sum(lengths[engaged != 0])) > 0)

            self.assertGreater(interaction.track_count, 1)
            with (
                patch.object(
                    interaction, "_read_engaged_track_length", wraps=interaction._read_engaged_track_length
                ) as read,
                patch.object(
                    interaction, "request_engaged_track_length", wraps=interaction.request_engaged_track_length
                ) as request,
            ):

                def counts():
                    return read.call_count, request.call_count

                # Construction and begin_step evaluate nothing, read nothing and request nothing.
                self.assertIsNone(pre._active)
                pre.begin_step()
                self.assertEqual((pre._active, counts()), (None, (0, 0)))
                # The first reader evaluates: one device read (no accepted step kept a length), and one
                # request that the acceptance of this step read the length for the step after it.
                self.assertTrue(eager())
                self.assertIs(pre.active, True)
                self.assertEqual((counts(), interaction._engaged_length_requested), ((1, 1), True))
                # Every later reader of the step takes the held value.
                self.assertIs(pre.active, True)
                self.assertTrue(pre.prepare(dt))
                self.assertEqual(counts(), (1, 1))
                # An assigned value holds, without an evaluation, until the next begin_step.
                pre.active = False
                self.assertIs(pre.active, False)
                self.assertFalse(pre.prepare(dt))
                pre.active = 1
                self.assertIs(pre.active, True)
                self.assertEqual(counts(), (1, 1))
                pre.begin_step()
                self.assertIsNone(pre._active)
                pre.active = False
                pre.begin_step()
                self.assertIs(pre.active, True)
                self.assertEqual(counts(), (2, 2))
                # What the host knows decides before the engagement is read. An empty track:
                pre.begin_step()
                with patch.object(type(interaction), "track_count", new_callable=PropertyMock, return_value=0):
                    self.assertFalse(pre.prepare(dt))
                    self.assertEqual((pre._active, counts()), (None, (2, 2)))
                    self.assertIs(pre.active, False)
                self.assertEqual(counts(), (2, 3))
                # A track without a shaft law: inactive, with no read and no request.
                stiffness, damping = interaction.shaft_stiffness_per_m, interaction.shaft_damping_per_m
                interaction.shaft_stiffness_per_m = interaction.shaft_damping_per_m = 0.0
                pre.begin_step()
                self.assertFalse(pre.prepare(dt))
                self.assertEqual((pre._active, counts()), (None, (2, 3)))
                self.assertFalse(eager())
                self.assertIs(pre.active, False)
                self.assertEqual(counts(), (2, 3))
                interaction.shaft_stiffness_per_m, interaction.shaft_damping_per_m = stiffness, damping
                # A released track with a law: the read decides, as it did in begin_step.
                engaged = wp.clone(interaction._track_engaged)
                interaction._track_engaged.zero_()
                interaction._engaged_length = None
                pre.begin_step()
                self.assertFalse(eager())
                self.assertIs(pre.active, False)
                self.assertFalse(pre.prepare(dt))
                self.assertEqual(counts(), (3, 4))
                wp.copy(interaction._track_engaged, engaged)
                interaction._engaged_length = None
                pre.begin_step()
                self.assertTrue(eager())
                self.assertIs(pre.active, True)
                self.assertEqual(counts(), (4, 5))
                # Coupled steps: the host trials of each step are its first reader, the acceptance
                # answers the next step, and the flag is the eager value at every step.
                interaction._engaged_length_requested = False
                reads = read.call_count
                solve = ex.coupler._solve_host
                seen, begun = [], []

                def no_device_trials(*args):
                    # The attempt has begun: where device trials are chosen, nothing is evaluated yet.
                    begun.append((pre._active, read.call_count))
                    return None

                def host(*args, **kwargs):
                    seen.append((args[5], eager(), read.call_count))
                    return solve(*args, **kwargs)

                with (
                    patch.object(ex.coupler, "_device_trials", side_effect=no_device_trials),
                    patch.object(ex.coupler, "_solve_host", side_effect=host),
                ):
                    for _ in range(3):
                        ex.coupler.step(ex.state_0, ex.state_1, ex.control, dt)
                        ex.state_0, ex.state_1 = ex.state_1, ex.state_0
                self.assertEqual(begun, [(None, reads), (None, reads + 2), (None, reads + 3)])
                self.assertEqual([item[:2] for item in seen], [(True, True)] * 3)
                # The first step read the device before its trials; the two after it did not.
                self.assertEqual([item[2] for item in seen], [reads + 1, reads + 2, reads + 3])
                self.assertEqual(read.call_count, reads + 4)

    def _steps_handed_to_host_trials(self, evaluates):
        """Coupled steps that device trials hand to the host; ``evaluates``: those trials evaluate the flag."""
        from newton_solver_suturebot._src.solvers.coupled.host_trial_forces import TrialForceFallback

        with _example() as ex:
            dt = 1 / 240
            setup(ex, dt, 6000)
            _configure(ex, True)
            coupler, interaction = ex.coupler, ex.interaction
            pre = coupler._shaft_preconditioner
            self.assertGreater(interaction.track_count, 1)
            self.assertFalse(interaction._engaged_length_requested)
            self.assertIsNone(interaction._engaged_length)
            solve, accept, begin = coupler._solve_host, interaction.accept, pre.begin_step
            in_accept, seen, begun, handed, kept, raising = [], [], [], [], [], []
            # The owner of device trials, as far as a step that it hands over asks it.
            trials = SimpleNamespace(evaluates_shaft_flag=evaluates, invalidate=lambda: None)

            def perturbed_step():
                # A carried load that the first trial does not balance: the trials apply corrections.
                load = coupler._load.numpy().copy()
                load[-6] += 1e-4
                coupler._load.assign(load)
                coupler.step(ex.state_0, ex.state_1, ex.control, dt)
                ex.state_0, ex.state_1 = ex.state_1, ex.state_0

            def hand_over(graph, state_in, control, step_dt, tunnel_alm):
                # What device trials do that hand their step to the host: they evaluated no
                # flag, and the trials are restarted from the accepted load.
                handed.append((pre._active, read.call_count, request.call_count))
                coupler._first_trial_event = coupler._event_load_kept = False
                coupler._restart_trials(state_in, tunnel_alm)

            def host(*args, **kwargs):
                seen.append((args[5], pre._active, read.call_count, request.call_count))
                kept.append(interaction._engaged_length)
                if raising:
                    raise raising.pop()
                return solve(*args, **kwargs)

            def counted_accept(state, body):
                before = read.call_count
                accept(state, body)
                in_accept.append(read.call_count - before)

            def counted_begin():
                begun.append(pre._active)
                begin()

            with (
                patch.object(
                    interaction, "_read_engaged_track_length", wraps=interaction._read_engaged_track_length
                ) as read,
                patch.object(
                    interaction, "request_engaged_track_length", wraps=interaction.request_engaged_track_length
                ) as request,
                patch.object(interaction, "accept", side_effect=counted_accept),
                patch.object(pre, "begin_step", side_effect=counted_begin),
                patch.object(pre, "apply", wraps=pre.apply) as apply,
                patch.object(coupler, "_captured_trials", trials),
                patch.object(coupler, "_device_trials", return_value=trials),
                patch.object(coupler, "_solve_device", side_effect=hand_over),
                patch.object(coupler, "_solve_host", side_effect=host),
            ):
                if evaluates:
                    # Trials that evaluate the flag before every launch: the host trials that
                    # take over ask the acceptance for the length of the next step, as they
                    # do in a step without device trials.
                    perturbed_step()
                    self.assertEqual(handed, [(None, 0, 0)])
                    self.assertEqual(seen, [(True, True, 1, 1)])
                    self.assertEqual((in_accept, read.call_count, request.call_count), ([1], 2, 1))
                    perturbed_step()
                    # Answered from that read; the acceptance reads for the step after it.
                    self.assertEqual(seen[1:], [(True, True, 2, 2)])
                    self.assertEqual((in_accept, read.call_count, request.call_count), ([1, 1], 3, 2))
                    return
                # Two steps in a row that are handed over: each reads once, before its host
                # trials, and neither its acceptance nor a request reads for the next step.
                for count in (1, 2):
                    perturbed_step()
                    self.assertEqual(handed[-1], (None, count - 1, 0))
                    self.assertEqual(seen[-1], (True, True, count, 0))
                    self.assertEqual((in_accept, read.call_count, request.call_count), ([0] * count, count, 0))
                    self.assertFalse(interaction._engaged_length_requested)
                    self.assertIsNone(interaction._engaged_length)
                # The captured host forces refuse the step and its trials are restarted, which
                # begins the step of the correction again. The value of the step holds: the
                # second run of the host trials finds it, and its corrections read nothing.
                del begun[:]
                applied = apply.call_count
                raising.append(TrialForceFallback("refused"))
                perturbed_step()
                self.assertEqual(seen[2:], [(True, True, 3, 0), (True, True, 3, 0)])
                # The correction's step was begun with the attempt (it held the value of the step
                # before), by the hand-over (nothing evaluated yet) and by the restart (evaluated
                # by the host trials in between).
                self.assertEqual(begun, [True, None, True])
                self.assertGreater(apply.call_count, applied)
                self.assertEqual((in_accept, read.call_count, request.call_count), ([0, 0, 0], 3, 0))
                # While the host trials ran, the interaction held the length that was read for
                # them, with the two arrays it was read from; each acceptance dropped it.
                for held in kept:
                    self.assertIs(held[0], interaction._track_engaged)
                    self.assertIs(held[1], interaction._track_lengths)
                    self.assertGreater(held[2], 0)
                # An attempt that is handed over and rejected after its trials, and the same step
                # again: the rejected attempt accepted nothing, so the engagement is the one that
                # was read, and the second attempt is answered from the kept length.
                with (
                    patch.object(interaction, "check_step_resolution", side_effect=NeedleStepTooLarge("late")),
                    self.assertRaisesRegex(NeedleStepTooLarge, "late"),
                ):
                    perturbed_step()
                self.assertEqual(seen[-1], (True, True, 4, 0))
                self.assertEqual((in_accept, read.call_count, request.call_count), ([0, 0, 0], 4, 0))
                self.assertIs(interaction._engaged_length, kept[-1])
                perturbed_step()
                self.assertEqual(seen[-1], (True, True, 4, 0))
                self.assertEqual((in_accept, read.call_count, request.call_count), ([0, 0, 0, 0], 4, 0))
                self.assertIsNone(interaction._engaged_length)
                self.assertNotEqual(interaction.status, "failed")
                # The flag is the value that begin_step evaluated before it left it to its reader.
                lengths, engaged = interaction._track_lengths.numpy(), interaction._track_engaged.numpy()
                self.assertGreater(float(np.sum(lengths[engaged != 0])), 0)

    def test_host_trials_after_trials_that_skip_the_flag_read_the_engagement_once_and_request_nothing(self):
        # Device trials whose graph is the same for both values of the flag do not evaluate it
        # (the articulated trials with a surface metric). Host-controlled trials that take a step
        # over from them are the one reader of that step, and the step after it does not ask:
        # the engaged length is read once, where those trials first run, the acceptance of the
        # step reads nothing, and a restart of the trials inside the step keeps the value. The
        # interaction keeps the length until the engagement is written, so a second attempt of
        # a step whose first attempt was rejected reads nothing.
        self._steps_handed_to_host_trials(False)

    def test_host_trials_after_trials_that_evaluate_the_flag_ask_the_acceptance_for_the_next_step(self):
        # The fixture carries a handful of accepted steps: each case starts from a new one.
        self._steps_handed_to_host_trials(True)

    def test_kept_engaged_length_answers_every_query_until_the_engagement_is_written(self):
        with _example() as ex:
            dt = 1 / 240
            setup(ex, dt, 6000)
            _configure(ex, True)
            interaction = ex.interaction

            def device_length():
                lengths, engaged = interaction._track_lengths.numpy(), interaction._track_engaged.numpy()
                return float(np.sum(lengths[engaged != 0]))

            self.assertIsNone(interaction._engaged_length)
            self.assertGreater(device_length(), 0)
            with patch.object(
                interaction, "_read_engaged_track_length", wraps=interaction._read_engaged_track_length
            ) as read:
                # A query alone reads the device every time and keeps nothing.
                self.assertEqual(interaction.engaged_track_length, device_length())
                self.assertEqual((read.call_count, interaction._engaged_length), (1, None))
                # A kept query reads once; it and the plain query are then answered without a read.
                self.assertEqual(interaction.keep_engaged_track_length(), device_length())
                self.assertEqual(read.call_count, 2)
                for _ in range(2):
                    self.assertEqual(interaction.keep_engaged_track_length(), device_length())
                    self.assertEqual(interaction.engaged_track_length, device_length())
                self.assertEqual(read.call_count, 2)
                # It asks the acceptance for nothing.
                self.assertFalse(interaction._engaged_length_requested)
                # A write of the engagement drops the value: an explicit force evaluation with the
                # needle back at its start releases the track without an accepted step.
                ex._initialize_pose(ex.plan.pose(ex.plan.initial_angle))
                ex.state_0.clear_forces()
                ex._external_forces.zero_()
                interaction.add_forces(ex.state_0, ex.needle_body, ex.drive.com, ex._external_forces, dt)
                self.assertIsNone(interaction._engaged_length)
                self.assertEqual(device_length(), 0.0)
                self.assertEqual(interaction.keep_engaged_track_length(), 0.0)
                self.assertEqual(read.call_count, 3)
                self.assertEqual(interaction.engaged_track_length, 0.0)
                self.assertEqual(read.call_count, 3)
                # A track without samples has no engaged length, and nothing is read or kept.
                interaction._engaged_length = None
                with patch.object(type(interaction), "track_count", new_callable=PropertyMock, return_value=0):
                    self.assertEqual(interaction.keep_engaged_track_length(), 0.0)
                self.assertEqual((read.call_count, interaction._engaged_length), (3, None))

    def test_accepted_applied_loads_are_distinguished_from_endpoint_force_law(self):
        for tolerance in (2e-7, 2e-6):
            with self.subTest(absolute_tolerance=tolerance), _example() as ex:
                dt = 1 / 240
                setup(ex, dt, 6000)
                _configure(ex, True)
                ex.coupler.absolute_tolerance = tolerance
                ex.coupler.relative_tolerance = 2e-4
                ex.coupler.relaxation = 1.0
                ex.coupler.step(ex.state_0, ex.state_1, ex.control, dt)
                # The trial vector is the force actually used to advance solvers, and
                # the load the next step starts its trials from. _value is the force
                # law evaluated at the accepted state.
                applied = ex.coupler._trial_load.numpy()
                physical = ex.coupler._value.numpy()
                np.testing.assert_array_equal(ex.coupler._load.numpy(), applied)
                scale = max(np.max(np.abs(applied)), np.max(np.abs(physical)))
                self.assertLessEqual(np.max(np.abs(applied - physical)), tolerance + 2e-4 * scale)
                applied_force, applied_moment = _balance(applied, ex.state_1, ex.coupler)
                law_force, law_moment = _balance(physical, ex.state_1, ex.coupler)
                self.assertLess(np.linalg.norm(law_force), 2e-9)
                self.assertLess(np.linalg.norm(law_moment), 2e-10)
                self.assertLess(np.linalg.norm(applied_force), 2e-9)
                # Applied iterates were assembled at earlier trial geometries.
                # Their final world moment is bounded by the fresh-law scatter
                # plus the accepted force residual, not by scatter roundoff alone:
                # |M_applied-M_law| <= sum |q_i x r_i| + |COM x r_body| + |r_tau|.
                residual = (physical - applied).reshape(-1, 3)
                q = ex.state_1.particle_q.numpy().astype(float)
                pose = ex.state_1.body_q.numpy()[ex.needle_body]
                center = np.asarray(wp.transform_point(wp.transform(*pose), wp.vec3(*ex.drive.com)), dtype=float)
                residual_moment_bound = (
                    np.linalg.norm(np.cross(q, residual[:-2]), axis=1).sum()
                    + np.linalg.norm(np.cross(center, residual[-2]))
                    + ex.coupler.moment_length * np.linalg.norm(residual[-1])
                )
                self.assertLessEqual(np.linalg.norm(applied_moment), np.linalg.norm(law_moment) + residual_moment_bound)

    def test_direction_application_preserves_force_state_and_committed_history(self):
        with _example() as ex:
            h = 1 / 240
            setup(ex, h, 6000)
            state_names = ("particle_q", "particle_qd", "particle_f", "body_q", "body_qd", "body_f")
            before = {name: getattr(ex.state_0, name).numpy().copy() for name in state_names}
            track_names = ("_track_nodes", "_track_bary", "_track_lengths", "_track_engaged")
            track = {name: getattr(ex.interaction, name).numpy().copy() for name in track_names}
            old_history = history(ex.interaction)
            pre = ShaftPreconditioner(ex.coupler)
            for scale in (1.0, -2.0, 0.0):
                residual = wp.array(np.full(3 * pre.n, scale * 1e-5), dtype=wp.float64, device=ex.device)
                _apply(pre, ex.state_0, h, residual)
            self.assertEqual(history(ex.interaction), old_history)
            for name, value in before.items():
                np.testing.assert_array_equal(getattr(ex.state_0, name).numpy(), value)
            for name, value in track.items():
                np.testing.assert_array_equal(getattr(ex.interaction, name).numpy(), value)

    def test_same_equilibrium_for_coarse_steps_and_identical_accepted_history(self):
        for h in (1 / 60, 1 / 120, 1 / 240):
            results = []
            for enabled in (False, True):
                with self.subTest(dt=h, enabled=enabled), _example() as ex:
                    setup(ex, h, 6000)
                    ex.coupler.absolute_tolerance = 2e-8
                    ex.coupler.relative_tolerance = 2e-5
                    ex.coupler.max_iterations = 160
                    _configure(ex, enabled)
                    ex.coupler.step(ex.state_0, ex.state_1, ex.control, h)
                    results.append(
                        {
                            "state": {
                                name: getattr(ex.state_1, name).numpy().copy()
                                for name in ("particle_q", "particle_qd", "body_q", "body_qd")
                            },
                            "force": ex.interaction.interface_force.copy(),
                            "history": history(ex.interaction),
                            "engaged": ex.interaction._track_engaged.numpy().copy(),
                        }
                    )
            a, b = results
            self.assertEqual(a["history"], b["history"])
            np.testing.assert_array_equal(a["engaged"], b["engaged"])
            for name in ("particle_q", "body_q"):
                np.testing.assert_allclose(a["state"][name], b["state"][name], atol=2e-7, rtol=1e-5)
            np.testing.assert_allclose(a["state"]["particle_qd"], b["state"]["particle_qd"], atol=2e-6, rtol=2e-4)
            # Compare rigid twist in translational units using the same physical
            # moment length as the coupling residual; near-zero angular axes
            # otherwise impose an arbitrary units-dependent tolerance.
            av, bv = a["state"]["body_qd"].copy(), b["state"]["body_qd"].copy()
            av[:, 3:] *= ex.coupler.moment_length
            bv[:, 3:] *= ex.coupler.moment_length
            np.testing.assert_allclose(av, bv, atol=2e-6, rtol=2e-4)
            np.testing.assert_allclose(a["force"], b["force"], atol=1e-7, rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
