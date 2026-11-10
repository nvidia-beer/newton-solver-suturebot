# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""A shaft solve accelerator must preserve the coupled physical problem."""

import os
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp
from test_needle_insertion_example import _example

from newton_solver_suturebot import NeedleTissueCoupler, NeedleTissueInteraction, boundary_faces
from newton_solver_suturebot._src.shaft_preconditioner import _ShaftPreconditioner as ShaftPreconditioner
from newton_solver_suturebot._src.shaft_preconditioner import prepare_anchors
from newton_solver_suturebot.examples._needle_insertion_drive import NeedleDrive


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
    for nodes, weights, matrix, lever in zip(
        interaction._track_nodes.numpy(),
        interaction._track_bary.numpy().astype(float),
        pre.matrices.numpy().astype(float),
        pre.levers.numpy().astype(float),
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
    for nodes, weights in zip(
        ex.interaction._track_nodes.numpy(), ex.interaction._track_bary.numpy().astype(float), strict=True
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
        lengths = interaction._track_lengths.numpy().astype(float)
        nodes = interaction._track_nodes.numpy()
        bary = interaction._track_bary.numpy().astype(float)
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
                        h,
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
            pre.begin_step()
            actual = _apply(pre, ex.state_0, 1 / 240, residual)
            np.testing.assert_array_equal(actual.numpy(), residual.numpy())
            original_step = ex.solver.step
            ex.solver.step = lambda *args, **kwargs: original_step(*args, **kwargs)
            _configure(ex, True)
            self.assertIsNone(ex.coupler._shaft_preconditioner)
            self.assertEqual(ex.coupler.shaft_preconditioner_status, "unsupported")

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
                # _load is already replaced by _value after acceptance. The
                # trial vector is the force actually used to advance solvers.
                applied = ex.coupler._trial_load.numpy()
                physical = ex.coupler._value.numpy()
                np.testing.assert_array_equal(ex.coupler._load.numpy(), physical)
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
