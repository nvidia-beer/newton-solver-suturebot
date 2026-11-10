# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent dense reference for the joint velocity/Coulomb equations.

This reference freezes the material gather, shaft tangent and moment arm. It
checks that local model, not the derivative of a changing closest segment,
finite-rotation reconstruction, contact topology, or the FEM constitutive law.
All residual rows have force units; all unknowns have velocity units. The
friction unknown is zeta=tau/(rho*h), where tau is a force per track length.
"""

import os
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton_solver_suturebot._src.solvers.mujoco_fem import kernels
from newton_solver_suturebot._src.tunnel_alm import _TunnelALM


def skew(x):
    return np.array([[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]])


def relative_gather(bary, lever, moment_length):
    """World linear velocity and scaled angular velocity L*omega."""
    return np.hstack([*[-b * np.eye(3) for b in bary], np.eye(3), -skew(lever / moment_length)])


def normal_reference(displacement, tangent, compliance, clearance):
    projection = np.eye(3) - np.outer(tangent, tangent)
    d = projection @ displacement
    distance = np.linalg.norm(d)
    if distance <= clearance:
        return np.zeros(3), np.zeros((3, 3))
    direction = d / distance
    normal = (distance - clearance) * direction / compliance
    derivative = (
        (1.0 - clearance / distance) * projection + clearance / distance * np.outer(direction, direction)
    ) / compliance
    return normal, derivative


def joint_reference(unknown, mass, free_impulse, contacts, *, h, rho, compliance, clearance, friction):
    """Return actual local residual and its independent dense semismooth Jacobian.

    Each contact supplies J, unit tangent, reference transverse displacement,
    and positive material quadrature length. A disabled contact has length=0.
    ``mass`` includes the linearly implicit FEM/native rigid effective blocks.
    """
    size = mass.shape[0]
    velocity, zeta = unknown[:size], unknown[size:]
    residual = np.zeros_like(unknown)
    jacobian = np.zeros((len(unknown), len(unknown)))
    residual[:size] = (mass @ velocity - free_impulse) / h
    jacobian[:size, :size] = mass / h
    branches = []
    for i, contact in enumerate(contacts):
        gather, tangent, initial_d, length = contact
        row = size + i
        if length <= 0.0:
            # Numerical padding/disabled row; excluded from physical norms.
            residual[row], jacobian[row, row] = zeta[i], 1.0
            branches.append("inactive")
            continue
        normal, derivative = normal_reference(initial_d + h * gather @ velocity, tangent, compliance, clearance)
        magnitude = np.linalg.norm(normal)
        normal_direction = normal / magnitude if magnitude else np.zeros(3)
        slip = tangent @ gather @ velocity
        argument = rho * h * (zeta[i] + slip)
        bound = friction * magnitude
        proposed = np.clip(argument, -bound, bound)
        sticking = bool(bound > 0.0 and abs(argument) < bound)
        sign = np.sign(argument) if bound > 0.0 else 0.0
        # A zero Coulomb capacity is an unconstrained, friction-free row.
        branches.append("stick" if sticking else "slip" if bound else "free")
        scale = length * rho * h
        residual[:size] += length * gather.T @ (normal + proposed * tangent)
        residual[row] = scale * (zeta[i] - proposed / (rho * h))
        normal_action = h * derivative @ gather
        if sticking:
            drag_velocity = rho * h * tangent @ gather
            drag_zeta = rho * h
        else:
            drag_velocity = sign * friction * normal_direction @ normal_action
            drag_zeta = 0.0
        jacobian[:size, :size] += length * gather.T @ (normal_action + np.outer(tangent, drag_velocity))
        jacobian[:size, row] = length * drag_zeta * gather.T @ tangent
        jacobian[row, :size] = -length * drag_velocity
        jacobian[row, row] = scale - length * drag_zeta
    return residual, jacobian, branches


def block_inverse(rhs, motion, upper, lower, diagonal):
    """Exact dense Schur solve, used to verify the single-contact approximation."""
    first = np.linalg.solve(motion, rhs[: motion.shape[0]])
    inverse_upper = np.linalg.solve(motion, upper)
    schur = diagonal - lower @ inverse_upper
    dual = np.linalg.solve(schur, rhs[motion.shape[0] :] - lower @ first)
    return np.concatenate([first - inverse_upper @ dual, dual]), schur


class MujocoFEMEquationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(812)
        self.h, self.rho, self.alpha, self.clearance, self.mu = 1 / 60, 60000.0, 1 / 60000, 2e-5, 0.2
        self.bary = np.array([0.1, 0.2, 0.3, 0.4])
        self.lever = np.array([0.004, -0.002, 0.006])
        self.gather = relative_gather(self.bary, self.lever, 0.01)
        self.size = self.gather.shape[1]
        random = self.rng.normal(size=(self.size, self.size))
        self.mass = 1e-5 * (np.eye(self.size) + random.T @ random)
        self.velocity = self.rng.normal(size=self.size) * 2e-4
        self.free_impulse = self.rng.normal(size=self.size) * 1e-7
        self.options = dict(h=self.h, rho=self.rho, compliance=self.alpha, clearance=self.clearance, friction=self.mu)
        self.device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")

    def contact(self, branch, tangent=(1.0, 0.0, 0.0), length=0.00025):
        tangent = np.asarray(tangent, dtype=float)
        tangent /= np.linalg.norm(tangent)
        initial = (np.eye(3) - np.outer(tangent, tangent)) @ np.array([5e-5, 9e-5, 4e-5])
        if branch == "free":
            initial = -self.h * self.gather @ self.velocity
        contact = (self.gather, tangent, initial, length)
        normal, _ = normal_reference(
            initial + self.h * self.gather @ self.velocity, tangent, self.alpha, self.clearance
        )
        factor = {"stick": 0.35, "slip": 1.7, "reverse": -1.7, "free": 2.0}[branch]
        argument = factor * self.mu * np.linalg.norm(normal)
        if branch == "free":
            argument = 1.0
        zeta = argument / (self.rho * self.h) - tangent @ self.gather @ self.velocity
        return contact, zeta

    def test_joint_jacobian_matches_force_residual_differences_in_all_smooth_branches(self):
        for branch in ("stick", "slip", "reverse", "free"):
            with self.subTest(branch=branch):
                contact, zeta = self.contact(branch)
                unknown = np.r_[self.velocity, zeta]
                _, derivative, _ = joint_reference(unknown, self.mass, self.free_impulse, [contact], **self.options)
                columns = []
                for direction in np.eye(len(unknown)):
                    delta = 1e-8 * direction
                    plus = joint_reference(unknown + delta, self.mass, self.free_impulse, [contact], **self.options)[0]
                    minus = joint_reference(unknown - delta, self.mass, self.free_impulse, [contact], **self.options)[0]
                    columns.append((plus - minus) / 2e-8)
                np.testing.assert_allclose(derivative, np.array(columns).T, rtol=2e-7, atol=2e-10)

    def test_slip_keeps_normal_dependent_bound_and_is_nonsymmetric(self):
        contact, zeta = self.contact("slip")
        unknown = np.r_[self.velocity, zeta]
        _, matrix, _ = joint_reference(unknown, self.mass, self.free_impulse, [contact], **self.options)
        # Normal displacement changes saturated drag even when slip stays positive.
        direction = np.zeros_like(unknown)
        direction[-6:-3] = [0, 1, 0]
        action = matrix @ direction
        self.assertGreater(abs(action[-1]), 1e-4)
        self.assertGreater(np.linalg.norm(matrix[: self.size, : self.size] - matrix[: self.size, : self.size].T), 1e-3)
        self.assertEqual(np.linalg.norm(matrix[: self.size, -1]), 0.0)
        self.assertGreater(matrix[-1, -1], 0.0)

    def test_stick_is_a_velocity_constraint_with_positive_mobility_schur(self):
        contact, zeta = self.contact("stick")
        unknown = np.r_[self.velocity, zeta]
        _, matrix, branches = joint_reference(unknown, self.mass, self.free_impulse, [contact], **self.options)
        self.assertEqual(branches, ["stick"])
        self.assertEqual(matrix[-1, -1], 0.0)
        rhs = self.rng.normal(size=len(unknown))
        h0, upper, lower = matrix[:-1, :-1], matrix[:-1, -1:], matrix[-1:, :-1]
        solution, schur = block_inverse(rhs, h0, upper, lower, matrix[-1:, -1:])
        axis = contact[1] @ contact[0]
        scale = contact[3] * self.rho * self.h
        expected_schur = scale**2 * axis @ np.linalg.solve(h0, axis)
        self.assertGreater(expected_schur, 0.0)
        np.testing.assert_allclose(schur[0, 0], expected_schur, rtol=1e-13)
        np.testing.assert_allclose(solution, np.linalg.solve(matrix, rhs), rtol=2e-12, atol=2e-10)

    def test_slip_schur_is_triangular_and_disabled_contact_has_no_constraint(self):
        for branch, length in (("slip", 0.00025), ("free", 0.00025), ("stick", 0.0)):
            with self.subTest(branch=branch, length=length):
                contact, zeta = self.contact(branch, length=length)
                unknown = np.r_[self.velocity, zeta]
                _, matrix, _ = joint_reference(unknown, self.mass, self.free_impulse, [contact], **self.options)
                rhs = self.rng.normal(size=len(unknown))
                solution, schur = block_inverse(
                    rhs, matrix[:-1, :-1], matrix[:-1, -1:], matrix[-1:, :-1], matrix[-1:, -1:]
                )
                np.testing.assert_allclose(schur, matrix[-1:, -1:], rtol=0, atol=0)
                np.testing.assert_allclose(solution, np.linalg.solve(matrix, rhs), rtol=2e-12, atol=2e-10)

    def test_scatter_work_and_world_moment_include_fixed_support_reaction(self):
        positions = self.rng.normal(size=(4, 3)) * 0.01
        material_point = self.bary @ positions
        center = material_point - self.lever
        resultant = np.array([0.003, -0.005, 0.002])
        scattered = -(self.gather.T @ resultant)
        nodes = scattered[:-6].reshape(4, 3)
        rigid_force, scaled_torque = scattered[-6:-3], scattered[-3:]
        np.testing.assert_allclose(nodes.sum(axis=0) + rigid_force, 0.0, atol=1e-18)
        world_moment = np.cross(positions, nodes).sum(axis=0) + np.cross(center, rigid_force) + 0.01 * scaled_torque
        np.testing.assert_allclose(world_moment, 0.0, atol=1e-19)
        velocity = self.velocity.copy()
        velocity[:3] = 0.0  # Supported material node remains in reaction scatter.
        np.testing.assert_allclose(velocity @ scattered, -(self.gather @ velocity) @ resultant, rtol=1e-14)
        free = np.arange(3, self.size)
        np.testing.assert_allclose(velocity[free] @ scattered[free], velocity @ scattered, rtol=2e-15, atol=1e-21)
        self.assertGreater(np.linalg.norm(nodes[0]), 0.0)

    def test_friction_row_has_weighted_force_units_and_no_extra_timestep(self):
        contact, zeta = self.contact("stick")
        unknown = np.r_[self.velocity, zeta]
        residual, _, _ = joint_reference(unknown, self.mass, self.free_impulse, [contact], **self.options)
        tau = self.rho * self.h * zeta
        tangent, length = contact[1], contact[3]
        proposed = tau + self.rho * self.h * tangent @ self.gather @ self.velocity
        self.assertAlmostEqual(residual[-1], length * (tau - proposed), places=17)

    def test_projected_drag_momentum_is_row_equivalent_to_physical_multiplier_momentum(self):
        for branch in ("stick", "slip", "reverse", "free"):
            with self.subTest(branch=branch):
                contact, zeta = self.contact(branch)
                unknown = np.r_[self.velocity, zeta]
                residual, _, _ = joint_reference(unknown, self.mass, self.free_impulse, [contact], **self.options)
                gather, tangent, d0, length = contact
                normal, _ = normal_reference(d0 + self.h * gather @ self.velocity, tangent, self.alpha, self.clearance)
                tau = self.rho * self.h * zeta
                physical_momentum = (self.mass @ self.velocity - self.free_impulse) / self.h
                physical_momentum += length * gather.T @ (normal + tau * tangent)
                # The added projected-force term is an exact residual row
                # transformation. It changes conditioning, not simultaneous roots.
                transformed = residual[:-1] + (gather.T @ tangent) * residual[-1]
                np.testing.assert_allclose(transformed, physical_momentum, rtol=2e-14, atol=1e-18)

    def test_native_implicitfast_angular_increment_and_gyro_jacobian(self):
        rotation, _ = np.linalg.qr(self.rng.normal(size=(3, 3)))
        inertia = rotation @ np.diag([2e-9, 3e-9, 7e-9]) @ rotation.T
        omega = np.array([3.0, -5.0, 8.0])
        damping = np.diag([1e-8, 3e-8, 2e-8])
        torque = np.array([2e-7, -3e-7, 4e-7])
        gyro = skew(omega) @ inertia - skew(inertia @ omega)
        effective = inertia + self.h * (damping + gyro)
        rhs = self.h * (torque - damping @ omega - np.cross(omega, inertia @ omega))
        increment = np.linalg.solve(effective, rhs)
        np.testing.assert_allclose(effective @ increment, rhs, rtol=1e-13, atol=1e-22)
        for direction in np.eye(3):
            delta = 1e-5 * direction
            plus, minus = omega + delta, omega - delta
            finite_difference = (np.cross(plus, inertia @ plus) - np.cross(minus, inertia @ minus)) / 2e-5
            np.testing.assert_allclose(gyro @ direction, finite_difference, rtol=2e-9, atol=1e-18)
        self.assertGreater(np.linalg.norm(effective - effective.T), 0.0)
        moment_length = 0.01
        world_increment = moment_length * rotation @ increment
        world_matrix = rotation @ effective @ rotation.T / moment_length**2
        world_rhs = rotation @ rhs / moment_length
        np.testing.assert_allclose(world_matrix @ world_increment, world_rhs, rtol=1e-13, atol=1e-20)

    def kernel_fixture(self, branch, fixed_node=False):
        contact, zeta = self.contact(branch)
        gather, tangent, d0, length = contact
        gather = gather.copy()
        fixed = np.array([int(fixed_node), 0, 0, 0], dtype=np.int32)
        if fixed_node:
            gather[:, :3] = 0.0
        # Independent block-diagonal mechanics permits a direct dense oracle.
        diagonal = [np.diag([1e-4, 2e-4, 3e-4]) * (1 + i) for i in range(6)]
        mass = np.zeros((18, 18))
        for i, block in enumerate(diagonal):
            mass[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = block
        if fixed_node:
            mass[:3, :3] = self.h * np.eye(3)
        state = np.r_[self.velocity, zeta]
        contact = (gather, tangent, d0, length)
        _, jacobian, branches = joint_reference(state, mass, np.zeros(18), [contact], **self.options)
        normal, derivative = normal_reference(d0 + self.h * gather @ self.velocity, tangent, self.alpha, self.clearance)
        magnitude = np.linalg.norm(normal)
        direction = normal / magnitude if magnitude else np.zeros(3)
        argument = self.rho * self.h * (zeta + tangent @ gather @ self.velocity)
        sticking = float(branches[0] == "stick")
        sliding = self.mu * np.sign(argument) if magnitude > 0.0 and not sticking else 0.0
        coefficients = np.array([sticking, sliding, length * self.rho * self.h])

        def array(value, dtype):
            return wp.array(value, dtype=dtype, device=self.device)

        arrays = [
            array(fixed, int),
            array([[0, 1, 2, 3]], wp.vec4i),
            array([self.bary], wp.vec4),
            array([length * self.h * derivative], wp.mat33),
            array([tangent], wp.vec3),
            array([direction], wp.vec3),
            array([self.lever / 0.01], wp.vec3),
            array([self.lever / 0.01], wp.vec3),
            array([coefficients], wp.vec3),
        ]
        return jacobian, diagonal, arrays, derivative, coefficients, tangent, length

    def test_runtime_contact_action_matches_dense_equations_and_fixed_projection(self):
        for branch in ("stick", "slip", "reverse", "free"):
            for fixed_node in (False, True):
                with self.subTest(branch=branch, fixed_node=fixed_node):
                    jacobian, diagonal, arrays, _, _, _, _ = self.kernel_fixture(branch, fixed_node)
                    packed = self.rng.normal(size=(7, 3)).astype(np.float32)
                    unknown = wp.array(packed, dtype=wp.vec3, device=self.device)
                    fem_action = np.array([diagonal[i] @ packed[i] for i in range(4)])
                    blocks = np.array([diagonal[4] / self.h, diagonal[5] / self.h, diagonal[5] / self.h])
                    out = wp.zeros_like(unknown)
                    wp.launch(
                        kernels.base_action,
                        7,
                        [
                            unknown,
                            wp.array(fem_action, dtype=wp.vec3, device=self.device),
                            wp.array(blocks, dtype=wp.mat33, device=self.device),
                            arrays[0],
                            self.h,
                            out,
                        ],
                        device=self.device,
                    )
                    wp.launch(kernels.contact_action, 1, [unknown, *arrays, out, 0.0], device=self.device)
                    expected = jacobian @ np.r_[packed[:6].ravel(), packed[6, 0]]
                    actual = out.numpy()
                    np.testing.assert_allclose(np.r_[actual[:6].ravel(), actual[6, 0]], expected, rtol=2e-5, atol=2e-7)
                    np.testing.assert_array_equal(actual[6, 1:], packed[6, 1:])
                    if fixed_node:
                        np.testing.assert_array_equal(actual[0], packed[0])

    def test_runtime_block_preconditioner_matches_dense_schur_inverse(self):
        for branch in ("stick", "slip", "free"):
            with self.subTest(branch=branch):
                jacobian, diagonal, arrays, derivative, c, tangent, length = self.kernel_fixture(branch, True)
                contact_diagonal = length * self.h * derivative + c[0] * c[2] * np.outer(tangent, tangent)
                expected_blocks = [block / self.h for block in diagonal]
                for i in range(4):
                    expected_blocks[i] += self.bary[i] ** 2 * contact_diagonal
                expected_blocks[0] = np.eye(3)
                expected_blocks[4] += contact_diagonal
                arm = skew(self.lever / 0.01)
                expected_blocks[5] += arm.T @ contact_diagonal @ arm
                h0 = np.zeros((18, 18))
                for i, block in enumerate(expected_blocks):
                    h0[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = block
                diagonal_array = wp.zeros(6, dtype=wp.mat33, device=self.device)
                blocks = np.array([diagonal[4] / self.h, diagonal[5] / self.h, diagonal[5] / self.h])
                wp.launch(
                    kernels.base_diagonal,
                    6,
                    [
                        wp.array(diagonal[:4], dtype=wp.mat33, device=self.device),
                        arrays[0],
                        wp.array(blocks, dtype=wp.mat33, device=self.device),
                        self.h,
                        diagonal_array,
                    ],
                    device=self.device,
                )
                wp.launch(
                    kernels.contact_diagonal,
                    1,
                    [*arrays[:5], arrays[6], arrays[8], diagonal_array],
                    device=self.device,
                )
                np.testing.assert_allclose(diagonal_array.numpy(), expected_blocks, rtol=2e-6, atol=2e-8)
                inverse = wp.empty_like(diagonal_array)
                wp.launch(kernels.invert, 6, [diagonal_array, inverse], device=self.device)
                packed_rhs = self.rng.normal(size=(7, 3)).astype(np.float32)
                rhs = wp.array(packed_rhs, dtype=wp.vec3, device=self.device)
                motion, correction = wp.zeros_like(rhs), wp.zeros_like(rhs)
                failed = wp.zeros(1, dtype=int, device=self.device)
                wp.launch(kernels.precondition_start, 6, [rhs, inverse, motion], device=self.device)
                wp.launch(
                    kernels.precondition_friction,
                    1,
                    [rhs, motion, inverse, *arrays, correction, failed, 0.0],
                    device=self.device,
                )
                wp.launch(kernels.precondition_finish, 6, [motion, inverse, correction], device=self.device)
                np.testing.assert_array_equal(failed.numpy(), [0])
                dense_rhs = np.r_[packed_rhs[:6].ravel(), packed_rhs[6, 0]]
                expected, _ = block_inverse(dense_rhs, h0, jacobian[:-1, -1:], jacobian[-1:, :-1], jacobian[-1:, -1:])
                actual = motion.numpy()
                np.testing.assert_allclose(np.r_[actual[:6].ravel(), actual[6, 0]], expected, rtol=3e-5, atol=4e-5)
                np.testing.assert_array_equal(actual[6, 1:], packed_rhs[6, 1:])

    def test_runtime_preconditioner_flags_unsupported_stick_mobility_instead_of_dividing(self):
        _, _, arrays, _, _, _, _ = self.kernel_fixture("stick")
        for invalid in (0.0, -1.0, np.nan):
            with self.subTest(invalid=invalid):
                inverse = wp.array([invalid * np.eye(3) for _ in range(6)], dtype=wp.mat33, device=self.device)
                rhs = wp.ones(7, dtype=wp.vec3, device=self.device)
                motion = wp.zeros_like(rhs)
                correction = wp.zeros_like(rhs)
                failed = wp.zeros(1, dtype=int, device=self.device)
                wp.launch(
                    kernels.precondition_friction,
                    1,
                    [rhs, motion, inverse, *arrays, correction, failed, 0.0],
                    device=self.device,
                )
                np.testing.assert_array_equal(failed.numpy(), [1])
                np.testing.assert_array_equal(motion.numpy(), 0.0)
                np.testing.assert_array_equal(correction.numpy(), 0.0)

    def test_search_regularization_only_changes_stick_dual_diagonal_and_matching_schur(self):
        regularization = 0.01
        for branch in ("stick", "slip", "free"):
            with self.subTest(branch=branch):
                jacobian, _, arrays, _, c, _, _ = self.kernel_fixture(branch)
                packed = self.rng.normal(size=(7, 3)).astype(np.float32)
                vector = wp.array(packed, dtype=wp.vec3, device=self.device)
                original, regularized = wp.zeros_like(vector), wp.zeros_like(vector)
                wp.launch(kernels.contact_action, 1, [vector, *arrays, original, 0.0], device=self.device)
                wp.launch(kernels.contact_action, 1, [vector, *arrays, regularized, regularization], device=self.device)
                expected_difference = np.zeros((7, 3))
                expected_difference[6, 0] = regularization * c[0] * c[2] * packed[6, 0]
                np.testing.assert_allclose(
                    regularized.numpy() - original.numpy(), expected_difference, rtol=4e-5, atol=2e-8
                )
                # A separate dense block inverse checks that the same diagonal
                # is used by the search preconditioner, including the Schur term.
                inverse = wp.array([np.eye(3)] * 6, dtype=wp.mat33, device=self.device)
                motion, correction = wp.zeros_like(vector), wp.zeros_like(vector)
                failed = wp.zeros(1, dtype=int, device=self.device)
                wp.launch(kernels.precondition_start, 6, [vector, inverse, motion], device=self.device)
                wp.launch(
                    kernels.precondition_friction,
                    1,
                    [vector, motion, inverse, *arrays, correction, failed, regularization],
                    device=self.device,
                )
                wp.launch(kernels.precondition_finish, 6, [motion, inverse, correction], device=self.device)
                diagonal = jacobian[-1:, -1:] + regularization * c[0] * c[2]
                expected, _ = block_inverse(
                    np.r_[packed[:6].ravel(), packed[6, 0]],
                    np.eye(18),
                    jacobian[:-1, -1:],
                    jacobian[-1:, :-1],
                    diagonal,
                )
                actual = motion.numpy()
                np.testing.assert_array_equal(failed.numpy(), [0])
                np.testing.assert_allclose(np.r_[actual[:6].ravel(), actual[6, 0]], expected, rtol=3e-5, atol=4e-5)
                np.testing.assert_array_equal(actual[6, 1:], packed[6, 1:])

    def test_redundant_sticking_rows_gain_bounded_search_inverse_without_changing_residual(self):
        contact, zeta = self.contact("stick")
        unknown = np.r_[self.velocity, zeta, zeta]
        before, matrix, branches = joint_reference(
            unknown, self.mass, self.free_impulse, [contact, contact], **self.options
        )
        self.assertEqual(branches, ["stick", "stick"])
        h0, upper, lower = matrix[:-2, :-2], matrix[:-2, -2:], matrix[-2:, :-2]
        schur = -lower @ np.linalg.solve(h0, upper)
        eigenvalues = np.linalg.eigvalsh(schur)
        self.assertLess(abs(eigenvalues[0]), 1e-13)
        self.assertGreater(eigenvalues[1], 0.0)
        regularization = 0.01
        scale = contact[3] * self.rho * self.h
        shift = regularization * scale
        regularized_schur = schur + shift * np.eye(2)
        np.testing.assert_allclose(np.linalg.eigvalsh(regularized_schur), eigenvalues + shift, rtol=1e-13)
        # Antisymmetric multiplier error is a null direction of the redundant
        # original constraints. The numerical shift bounds that correction.
        error = np.array([1e-7, -1e-7])
        dual_correction = np.linalg.solve(regularized_schur, error)
        np.testing.assert_allclose(dual_correction, error / shift, rtol=1e-13)
        np.testing.assert_allclose(upper @ dual_correction, 0.0, atol=1e-18)
        after = joint_reference(unknown, self.mass, self.free_impulse, [contact, contact], **self.options)[0]
        np.testing.assert_array_equal(after, before)

    def test_runtime_rigid_system_matches_rotated_offdiagonal_native_equations(self):
        angle = 0.6
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        q = np.array([0.02, -0.04, 0.01, 0, 0, np.sin(angle / 2), np.cos(angle / 2)], dtype=np.float32)
        local_rotation, _ = np.linalg.qr(self.rng.normal(size=(3, 3)))
        inertia = local_rotation @ np.diag([2e-9, 3e-9, 7e-9]) @ local_rotation.T
        local_omega = np.array([3.0, -5.0, 8.0])
        velocity = np.array([0.001, -0.002, 0.003])
        qd = np.r_[velocity, rotation @ local_omega].astype(np.float32)
        damping = np.array([0.03, 0.02, 0.04, 1e-8, 3e-8, 2e-8])
        mass, moment_length = 0.0003, 0.01
        external = np.array([0.003, -0.001, 0.002, 2e-7, -3e-7, 4e-7], dtype=np.float32)
        joint = np.array([0.001, 0.002, -0.001, 1e-7, 2e-7, -1e-7], dtype=np.float32)
        wrench = (external + joint).astype(float)
        D = np.diag(damping[3:])
        gyro = skew(local_omega) @ inertia - skew(inertia @ local_omega)
        angular = inertia + self.h * (D + gyro)
        expected_blocks = np.array(
            [
                np.diag(mass / self.h + damping[:3]),
                rotation @ angular @ rotation.T / (self.h * moment_length**2),
                rotation @ (inertia + self.h * D) @ rotation.T / (self.h * moment_length**2),
            ]
        )
        expected_rhs = np.array(
            [
                wrench[:3] - damping[:3] * velocity,
                (wrench[3:] - rotation @ (D @ local_omega + np.cross(local_omega, inertia @ local_omega)))
                / moment_length,
            ]
        )
        blocks = wp.zeros(3, dtype=wp.mat33, device=self.device)
        rhs = wp.zeros(2, dtype=wp.vec3, device=self.device)
        wp.launch(
            kernels.rigid_system,
            1,
            [
                wp.array(q, dtype=float, device=self.device),
                wp.array(qd, dtype=float, device=self.device),
                wp.array([external], dtype=wp.spatial_vector, device=self.device),
                wp.array(joint, dtype=float, device=self.device),
                wp.array([mass], dtype=wp.float64, device=self.device),
                wp.array([inertia], dtype=wp.mat33d, device=self.device),
                wp.array(damping, dtype=wp.float64, device=self.device),
                self.h,
                moment_length,
                blocks,
                rhs,
            ],
            device=self.device,
        )
        np.testing.assert_allclose(blocks.numpy(), expected_blocks, rtol=3e-6, atol=1e-10)
        np.testing.assert_allclose(rhs.numpy(), expected_rhs, rtol=3e-6, atol=2e-11)

    def test_runtime_anchor_linearization_preserves_radial_derivative_and_branches(self):
        for displacement, slip, density, engaged in (
            ([0, 8e-5, 4e-5], 1e-4, 0.0, 1),
            ([0, 8e-5, 4e-5], 0.01, 0.0, 1),
            ([0, 8e-5, 4e-5], -0.01, 0.0, 1),
            ([0, 1e-5, 0], 0.01, 0.0, 1),
            ([0, 8e-5, 4e-5], 0.01, 0.0, 0),
        ):
            with self.subTest(displacement=displacement, slip=slip, engaged=engaged):
                d = np.asarray(displacement)
                tangent = np.array([1.0, 0.0, 0.0])
                normal, derivative = normal_reference(d, tangent, self.alpha, self.clearance)
                magnitude = np.linalg.norm(normal)
                argument = density + self.rho * self.h * slip
                sticking = float(magnitude > 0.0 and abs(argument) < self.mu * magnitude)
                sliding = self.mu * np.sign(argument) if magnitude > 0 and not sticking else 0.0
                length = 0.00025
                normal_matrix = wp.zeros(1, dtype=wp.mat33, device=self.device)
                vector_arrays = [wp.zeros(1, dtype=wp.vec3, device=self.device) for _ in range(5)]
                wp.launch(
                    kernels.linearize_anchors,
                    1,
                    [
                        wp.array([[0, 1, 2, 3]], dtype=wp.vec4i, device=self.device),
                        wp.array([self.bary], dtype=wp.vec4, device=self.device),
                        wp.array([length], dtype=float, device=self.device),
                        wp.array([engaged], dtype=int, device=self.device),
                        wp.zeros(4, dtype=wp.vec3, device=self.device),
                        wp.zeros(4, dtype=wp.vec3, device=self.device),
                        wp.array([d + [-0.01, 0, 0], d + [0.01, 0, 0]], dtype=wp.vec3, device=self.device),
                        wp.array([wp.transform_identity()], dtype=wp.transform, device=self.device),
                        wp.array([[slip, 0, 0, 0, 0, 0]], dtype=wp.spatial_vector, device=self.device),
                        wp.array([density], dtype=float, device=self.device),
                        0.00028,
                        self.clearance,
                        self.alpha,
                        self.rho,
                        self.mu,
                        self.h,
                        0.01,
                        normal_matrix,
                        *vector_arrays,
                    ],
                    device=self.device,
                )
                actual_tangent, actual_normal, actual_arm, actual_geometry_arm, actual_coefficients = [
                    a.numpy()[0] for a in vector_arrays
                ]
                if not engaged:
                    np.testing.assert_array_equal(normal_matrix.numpy(), 0.0)
                    np.testing.assert_array_equal(actual_coefficients, [0, 0, 1])
                    continue
                np.testing.assert_allclose(normal_matrix.numpy()[0], length * self.h * derivative, rtol=2e-6, atol=2e-8)
                np.testing.assert_allclose(actual_tangent, tangent, atol=1e-7)
                np.testing.assert_allclose(actual_normal, normal / magnitude if magnitude else np.zeros(3), atol=1e-7)
                np.testing.assert_allclose(actual_arm, 0.0, atol=0)
                np.testing.assert_allclose(actual_geometry_arm, d / 0.01, atol=1e-7)
                np.testing.assert_allclose(
                    actual_coefficients, [sticking, sliding, length * self.rho * self.h], atol=1e-7
                )

    def test_runtime_reconstruction_uses_old_frame_increment_and_new_world_angular_velocity(self):
        initial_quaternion = np.array([0.2, -0.1, 0.3, 0.9])
        initial_quaternion /= np.linalg.norm(initial_quaternion)
        vq, wq = initial_quaternion[:3], initial_quaternion[3]
        rotation = np.eye(3) + 2 * wq * skew(vq) + 2 * skew(vq) @ skew(vq)
        position = np.array([0.03, -0.04, 0.002])
        linear = np.array([0.01, -0.02, 0.03])
        angular = np.array([1.2, -2.1, 3.4])
        delta_linear, delta_angular = np.array([0.003, 0.001, -0.002]), np.array([-0.1, 0.2, 0.4])
        moment_length = 0.01
        packed = np.array([delta_linear, moment_length * rotation @ delta_angular])
        q = wp.array(np.r_[position, initial_quaternion], dtype=float, device=self.device)
        qd = wp.array(np.r_[linear, rotation @ angular], dtype=float, device=self.device)
        unknown = wp.array(packed, dtype=wp.vec3, device=self.device)
        q_out, qd_out = wp.empty_like(q), wp.empty_like(qd)
        poses = wp.empty(1, dtype=wp.transform, device=self.device)
        velocities = wp.empty(1, dtype=wp.spatial_vector, device=self.device)
        for h in (1 / 720, 0.017123456789, 4.069010416666667e-6):
            with self.subTest(h=h):
                wp.launch(
                    kernels.reconstruct_rigid,
                    1,
                    [q, qd, unknown, 0, h, moment_length, q_out, qd_out, poses, velocities],
                    device=self.device,
                )
                new_local = angular + delta_angular
                half_angle = h * np.linalg.norm(new_local) / 2
                dv = np.sin(half_angle) * new_local / np.linalg.norm(new_local)
                dw = np.cos(half_angle)
                quaternion = np.r_[wq * dv + dw * vq + np.cross(vq, dv), wq * dw - vq @ dv]
                qv, qw = quaternion[:3], quaternion[3]
                next_rotation = np.eye(3) + 2 * qw * skew(qv) + 2 * skew(qv) @ skew(qv)
                expected_q = np.r_[position + h * (linear + delta_linear), quaternion]
                expected_qd = np.r_[linear + delta_linear, next_rotation @ new_local]
                np.testing.assert_allclose(q_out.numpy(), expected_q, rtol=1e-6, atol=4e-8)
                np.testing.assert_allclose(qd_out.numpy(), expected_qd, rtol=2e-6, atol=5e-7)
                np.testing.assert_array_equal(poses.numpy()[0], q_out.numpy())
                np.testing.assert_array_equal(velocities.numpy()[0], qd_out.numpy())

    def predictor_fixture(self, slip, *, initial_density=0.2, gap=(0.0, 8e-5, 4e-5), engaged=1, friction=0.2):
        interaction = SimpleNamespace(
            device=self.device,
            track_count=1,
            _track_engaged=wp.array([engaged], dtype=int, device=self.device),
            _track_lengths=wp.array([0.00025], dtype=float, device=self.device),
        )
        alm = _TunnelALM(
            interaction,
            penalty_per_m=self.rho,
            compliance_per_m=self.alpha,
            clearance=self.clearance,
            tolerance=2e-6,
        )
        alm._resize()
        alm.committed_tangent.assign(np.array([initial_density], dtype=np.float32))
        alm.committed_normal.assign(np.array([[0, 1.3, -0.7]], dtype=np.float32))
        alm.begin_step()
        histories = (alm.committed_tangent, alm.committed_normal, alm.proposed_tangent, alm.proposed_normal)
        before_histories = [array.numpy().copy() for array in histories]

        # Include tissue motion and a nonzero angular lever so using body
        # translation alone would predict the wrong sliding direction.
        center = np.array([0.001, -0.002, 0.004])
        point = np.array([0.003, 0.001, 0.005])
        nodal_velocity = np.array([[0.001 * (j + 1), -0.003, 0.002] for j in range(4)])
        angular = np.array([0.0, 0.0, 0.7])
        linear = np.array(
            [slip + (self.bary @ nodal_velocity)[0] - np.cross(angular, point - center)[0], 0.002, -0.001]
        )
        local_closest = point - center + np.asarray(gap)
        values = self.rng.normal(size=(7, 3)).astype(np.float32)
        values[6] = [initial_density / (self.rho * self.h), 0, 0]
        unknown = wp.array(values, dtype=wp.vec3, device=self.device)
        wp.launch(
            kernels.predict_tangent,
            1,
            [
                unknown,
                wp.array([[0, 1, 2, 3]], dtype=wp.vec4i, device=self.device),
                wp.array([self.bary], dtype=wp.vec4, device=self.device),
                interaction._track_engaged,
                wp.array([point] * 4, dtype=wp.vec3, device=self.device),
                wp.array(nodal_velocity, dtype=wp.vec3, device=self.device),
                wp.array(
                    [local_closest + [-0.01, 0, 0], local_closest + [0.01, 0, 0]],
                    dtype=wp.vec3,
                    device=self.device,
                ),
                wp.array([wp.transform(center, wp.quat_identity())], dtype=wp.transform, device=self.device),
                wp.array([np.r_[linear, angular]], dtype=wp.spatial_vector, device=self.device),
                0.00028,
                self.clearance,
                self.alpha,
                friction,
                self.rho * self.h,
                alm.tangent,
            ],
            device=self.device,
        )
        bound = friction * max(0.0, np.linalg.norm(gap) - self.clearance) / self.alpha if engaged else 0.0
        expected = np.clip(initial_density, -bound, bound)
        if abs(slip) > 1e-6:
            expected = np.sign(slip) * bound
        actual = alm.tangent.numpy()[0]
        np.testing.assert_allclose(actual, expected, rtol=8e-6, atol=2e-7)
        np.testing.assert_allclose(unknown.numpy()[6, 0] * self.rho * self.h, expected, rtol=8e-6, atol=2e-7)
        np.testing.assert_array_equal(unknown.numpy()[:6], values[:6])
        np.testing.assert_array_equal(unknown.numpy()[6, 1:], 0.0)
        for array, before in zip(histories, before_histories, strict=True):
            np.testing.assert_array_equal(array.numpy(), before)
        alm.discard()
        self.assertFalse(alm._open)
        np.testing.assert_array_equal(alm.tangent.numpy(), before_histories[0])
        np.testing.assert_array_equal(alm.normal.numpy(), before_histories[1])
        return actual, bound

    def test_runtime_tangent_predictor_uses_relative_free_velocity_and_current_bound(self):
        for slip in (-0.002, 0.002):
            with self.subTest(slip=slip):
                predicted, bound = self.predictor_fixture(slip, initial_density=-np.sign(slip) * 0.2)
                self.assertGreater(bound, 0.0)
                self.assertEqual(np.sign(predicted), np.sign(slip))

    def test_runtime_tangent_predictor_preserves_quiet_history_and_clears_zero_capacity(self):
        # The predictor deadband is numerical initialization only. Quiet
        # accepted history survives, clipped if its current capacity shrank.
        cases = (
            (0.0, {}),
            (5e-7, {}),
            (-5e-7, {}),
            (0.0, {"initial_density": 3.0}),
            (0.002, {"gap": (0.0, 1e-5, 0.0)}),
            (0.002, {"friction": 0.0}),
            (0.002, {"engaged": 0}),
        )
        for slip, options in cases:
            with self.subTest(slip=slip, **options):
                predicted, bound = self.predictor_fixture(slip, **options)
                if not options:
                    self.assertAlmostEqual(predicted, 0.2, places=6)
                if bound == 0.0:
                    self.assertEqual(predicted, 0.0)


if __name__ == "__main__":
    unittest.main()
