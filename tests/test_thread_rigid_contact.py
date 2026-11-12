# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-sided tool contact, moving friction and reaction balance of physical ANCF thread."""

import json
import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_suturebot import SolverThread, ThreadModel
from newton_solver_suturebot._src.thread_rigid_contact import ThreadRigidCapsuleContact

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
RADIUS = 0.00015
DT = 1 / 240


def make_fixture(two_jaws=False, friction=0.4, skip=None):
    builder = newton.ModelBuilder()
    for _ in range(2 if two_jaws else 1):
        builder.add_body(mass=0.01, inertia=wp.mat33(0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.0001))
    model = builder.finalize(device=DEVICE)
    state = model.state()
    positions = np.zeros((5, 3))
    if two_jaws:
        positions[:, 2] = np.linspace(-0.004, 0.004, 5)
        poses = state.body_q.numpy().copy()
        poses[:, 0] = [-0.0009, 0.0009]
        state.body_q.assign(poses)
        endpoints = np.tile([[[0, -0.004, 0], [0, 0.004, 0]]], (2, 1, 1))
        radii, bodies = [0.0008, 0.0008], [0, 1]
        gravity = (0, 0, 0)
    else:
        positions[:, 0] = np.linspace(-0.004, 0.004, 5)
        positions[:, 2] = 0.0011
        endpoints = [[[0, -0.004, 0], [0, 0.004, 0]]]
        radii, bodies, gravity = [0.001], [0], (0, 0, -9.81)
    thread = ThreadModel.create(positions, young_modulus=2e6, radius=RADIUS, density=1140, device=DEVICE)
    solver = SolverThread(
        model,
        thread,
        gravity=gravity,
        mass_damping=2,
        max_iterations=80,
        tolerance=1e-10,
        contact_stiffness=32,
        contact_smoothing=0.1 * RADIUS,
    )
    contact = ThreadRigidCapsuleContact(
        solver, model, bodies, endpoints, radii, thread_radius=RADIUS, friction=friction, skip_by_body=skip
    )
    return model, state, solver, contact


def step(state, solver, contact):
    contact.update(state.body_q, state.body_qd, DT)
    solver.step(None, None, None, None, DT)
    contact.react(state.body_q)


class TestThreadRigidContact(unittest.TestCase):
    def test_penetrating_thread_resolves_with_solver_not_projection(self):
        _, state, solver, contact = make_fixture()
        original = solver.q.numpy().copy()
        step(state, solver, contact)
        q = solver.q.numpy().reshape(-1, 6)
        self.assertGreater(q[2, 2], 0.00115)
        self.assertGreater(contact.diagnostics()["minimum_contact_gap_m"], 0)
        np.testing.assert_array_equal(q[0], original.reshape(-1, 6)[0])
        self.assertGreater(np.linalg.norm(contact.fresh_wrench.numpy()), 0)
        self.assertTrue(np.isfinite(solver.qd.numpy()).all())
        json.dumps(contact.diagnostics(), allow_nan=False)

    def test_both_jaws_contribute_and_head_exclusion_is_body_specific(self):
        _, state, solver, contact = make_fixture(True, skip={0: 3})
        contact.update(state.body_q, state.body_qd, DT)
        contact.finish_step()
        contact.react(state.body_q)
        force = contact.force.numpy()
        # The central crossing belongs to elements1/2. The left jaw is
        # deliberately excluded there, but the right jaw must still act.
        np.testing.assert_array_equal(force[0, :3], 0)
        self.assertLess(force[1, 1, 0], 0)
        _, state, solver, contact = make_fixture(True)
        contact.update(state.body_q, state.body_qd, DT)
        contact.finish_step()
        contact.react(state.body_q)
        force = contact.force.numpy()
        self.assertGreater(force[0, 1, 0], 0)
        self.assertLess(force[1, 1, 0], 0)
        self.assertEqual(contact.groups, 2)
        np.testing.assert_allclose(force.sum((0, 1)), 0, atol=1e-12)
        self.assertGreater(np.linalg.norm(contact.fresh_wrench.numpy()[0]), 0)
        self.assertGreater(np.linalg.norm(contact.fresh_wrench.numpy()[1]), 0)

    def test_moving_capsules_carry_friction_and_conserve_fresh_wrench(self):
        model, state, solver, contact = make_fixture(True)
        contact.update(state.body_q, state.body_qd, DT)
        contact.finish_step()
        contact.react(state.body_q)
        poses = state.body_q.numpy().copy()
        poses[:, 1] += 0.0002  # tangential movement along the capsule axes
        state.body_q.assign(poses)
        contact.update(state.body_q, state.body_qd, DT)
        contact.finish_step()
        contact.react(state.body_q)
        force, points = contact.force.numpy(), contact.application_point.numpy()
        normal = contact.normal.numpy()
        tangent = force - (force * normal).sum(-1)[..., None] * normal
        self.assertGreater(force[..., 1].sum(), 0)
        np.testing.assert_array_less(np.linalg.norm(tangent, axis=-1), contact.used_limit.numpy() + 1e-12)
        rigid = contact.fresh_wrench.numpy()
        np.testing.assert_allclose(rigid[:, :3].sum(0) + force.sum((0, 1)), 0, atol=1e-9)
        # Compare torques about one common world origin, not each body's COM.
        com = poses[:, :3] + model.body_com.numpy()
        world_torque = rigid[:, 3:].sum(0) + np.cross(com, rigid[:, :3]).sum(0)
        np.testing.assert_allclose(world_torque + np.cross(points, force).sum((0, 1)), 0, atol=1e-11)
        state.body_f.zero_()
        contact.add_forces(state.body_f)
        np.testing.assert_array_equal(state.body_f.numpy(), contact.wrench.numpy())
        self.assertLess(contact.diagnostics()["maximum_friction_limit_excess_N"], 1e-12)

    def test_rotating_surface_transports_existing_friction_anchor(self):
        _, state, solver, contact = make_fixture(True)
        contact.update(state.body_q, state.body_qd, DT)
        contact.finish_step()
        contact.react(state.body_q)
        # Rotate bodies and the entire thread together. This introduces no
        # relative tangential displacement, even though world anchors move.
        angle = 0.3
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        translation = np.array([0.013, -0.004, 0.002])
        q = solver.q.numpy().copy().reshape(-1, 6)
        q[:, :3] = q[:, :3] @ rotation.T + translation
        q[:, 3:] = q[:, 3:] @ rotation.T
        solver.q.assign(q.reshape(-1))
        poses = state.body_q.numpy().copy()
        poses[:, :3] = poses[:, :3] @ rotation.T + translation
        poses[:, 3:] = [0, 0, np.sin(angle / 2), np.cos(angle / 2)]
        state.body_q.assign(poses)
        contact.update(state.body_q, state.body_qd, DT)
        contact.finish_step()
        contact.react(state.body_q)
        force, normal = contact.force.numpy(), contact.normal.numpy()
        tangent = force - (force * normal).sum(-1)[..., None] * normal
        # Pose buffers are float32; 1nm registration rounding at k=32 N/m.
        self.assertLess(np.max(np.linalg.norm(tangent, axis=-1)), 5e-8)
        self.assertGreater(np.max(np.linalg.norm(force, axis=-1)), 1e-4)

    def test_complete_state_restores_friction_and_lagged_reactions(self):
        _, state, solver, contact = make_fixture(True)
        step(state, solver, contact)
        saved = [wp.clone(a) for a in solver.state]
        step(state, solver, contact)
        expected = [a.numpy().copy() for a in solver.state]
        for array, prior in zip(solver.state, saved, strict=True):
            wp.copy(array, prior)
        step(state, solver, contact)
        for value, array in zip(expected, solver.state, strict=True):
            np.testing.assert_allclose(array.numpy(), value, atol=1e-12, rtol=1e-7)
        self.assertTrue(all(any(a is b for b in solver.state) for a in contact.state))

    def test_gradient_matches_line_search_energy_with_friction(self):
        _, state, solver, contact = make_fixture(True)
        contact.update(state.body_q, state.body_qd, DT)
        contact.finish_step()
        contact.react(state.body_q)
        poses = state.body_q.numpy().copy()
        poses[:, 1] += 1e-5
        state.body_q.assign(poses)
        contact.update(state.body_q, state.body_qd, DT)
        solver._status.zero_()
        solver._gradient.zero_()
        solver._hessian.zero_()
        contact.add_gradient(True)
        local_gradient = solver._gradient.numpy()
        gradient = np.zeros((solver.thread.node_count, 6))
        for e in range(solver.thread.element_count):
            gradient[e, :3] += local_gradient[e, 0]
            gradient[e, 3:] += local_gradient[e, 1]
            gradient[e + 1, :3] += local_gradient[e, 2]
            gradient[e + 1, 3:] += local_gradient[e, 3]
        original = solver.q.numpy().copy()

        def energy(q):
            solver._trial_q.assign(np.tile(q, len(solver._scales)))
            solver._trial_energy.zero_()
            contact.add_trial_energy()
            return solver._trial_energy.numpy()[0, 0]

        for index in (6, 7, 12, 13, 15, 16):
            epsilon = 1e-8 if index % 6 < 3 else 1e-5
            plus, minus = original.copy(), original.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numeric = (energy(plus) - energy(minus)) / (2 * epsilon)
            self.assertAlmostEqual(numeric, gradient.reshape(-1)[index], delta=1e-8)

    def test_invalid_contracts_fail_before_registration(self):
        model, _, solver, _ = make_fixture()
        cases = [
            dict(bodies=[99]),
            dict(bodies=[0.0]),
            dict(radii=[0]),
            dict(endpoints=[[[0, 0, 0], [0, 0, 0]]]),
            dict(friction=-1),
            dict(thread_radius=float("nan")),
            dict(skip_by_body={0: -1}),
            dict(damping_by_body={0: [0, float("nan")]}),
        ]
        for overrides in cases:
            args = dict(bodies=[0], endpoints=[[[0, 0, 0], [0, 0.01, 0]]], radii=[0.001], thread_radius=RADIUS)
            args.update(overrides)
            count = len(solver.element_contacts)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                ThreadRigidCapsuleContact(solver, model, **args)
            self.assertEqual(len(solver.element_contacts), count)


if __name__ == "__main__":
    unittest.main()
