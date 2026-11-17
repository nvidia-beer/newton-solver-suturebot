"""Force-preserving facet recording and frozen-friction PSD search blocks."""

import ast
import inspect
import os
import unittest
from itertools import permutations
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleTissueInteraction, boundary_faces
from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_contact_tangent import (
    SurfaceTangentAnchors,
    surface_contact_tangent,
)
from newton_solver_suturebot._src.solvers.coupled.interface import insertion
from newton_solver_suturebot._src.solvers.coupled.interface.surface_contact_forces import surface_forces_with_tangent

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


def block(subdivisions=1, slope=0.12, thickness=0.004):
    axis = np.linspace(-0.004, 0.004, subdivisions + 1)
    vertices = np.array(
        [
            (x, y, -thickness if layer == 0 else slope * abs(x - y) / np.sqrt(2))
            for x in axis
            for y in axis
            for layer in (0, 1)
        ]
    )
    nodes = np.arange(len(vertices)).reshape(subdivisions + 1, subdivisions + 1, 2)
    tetrahedra = []
    for x in range(subdivisions):
        for y in range(subdivisions):
            cell = nodes[x : x + 2, y : y + 2].ravel()
            for first, second, _ in permutations(range(3)):
                tet = cell[[0, 1 << first, (1 << first) | (1 << second), 7]].copy()
                if np.linalg.det((vertices[tet[1:]] - vertices[tet[:1]]).T) < 0:
                    tet[1], tet[2] = tet[2], tet[1]
                tetrahedra.append(tet)
    return vertices, np.asarray(tetrahedra, dtype=np.int32)


def fixture(vertices=None, tets=None, fixed=(), unmapped=(), *, single_active_sample=False):
    if vertices is None:
        vertices, tets = block()
    samples = np.array([[0.0, 0.0, 0.0], [0.00005, 0.00005, 0.01 if single_active_sample else 0.0]])
    interaction = NeedleTissueInteraction(
        vertices,
        tets,
        boundary_faces(tets)[0],
        fixed,
        samples,
        0.0005,
        samples[0],
        device=DEVICE,
        stiffness=20.0,
        damping=0.025,
        friction=0.15,
        puncture_force=1.0,
    )
    if unmapped:
        interaction.mapping[list(unmapped)] = -1
        interaction._mapping.assign(interaction.mapping)
    state = SimpleNamespace(
        particle_q=wp.array(vertices, dtype=wp.vec3, device=DEVICE),
        particle_qd=wp.zeros(len(vertices), dtype=wp.vec3, device=DEVICE),
        body_q=wp.array([[0.00011, 0.00011, 0.0004, 0, 0, 0, 1]], dtype=wp.transform, device=DEVICE),
        body_qd=wp.array([[0.003, 0.001, -0.001, 0, 0, 0]], dtype=wp.spatial_vector, device=DEVICE),
        body_f=wp.zeros(1, dtype=wp.spatial_vector, device=DEVICE),
    )
    forces = wp.zeros(len(vertices), dtype=wp.vec3, device=DEVICE)
    return interaction, state, forces, vertices


def evaluate(data, anchors=None, *, implicit=True):
    interaction, state, forces, _ = data
    state.body_f.zero_()
    forces.zero_()
    real_launch = wp.launch
    if anchors is not None:
        anchors.clear()

    def launch(kernel, *args, **kwargs):
        if kernel is insertion._surface_forces and anchors is not None:
            return real_launch(surface_forces_with_tangent, args[0], [*args[1], *anchors.arguments(0.01)], **kwargs)
        return real_launch(kernel, *args, **kwargs)

    with patch.object(wp, "launch", side_effect=launch):
        interaction.add_forces(state, 0, np.array([0.0007, 0.0003, 0]), forces, 1 / 240, implicit=implicit)
    return (
        state.body_f.numpy().copy(),
        forces.numpy().copy(),
        interaction._diagnostics.numpy().copy(),
        interaction._contact_flags.numpy().copy(),
    )


def atomic_roundoff_bounds(data):
    """Two FP32 accumulation orders: 2*gamma_n*sum(abs(contributions)).

    This oracle supports only the initial, identity-oriented, stationary-tissue
    surface fixtures here. Other force paths, nonfinite/capped recordings, or
    one contributing sample retain the original strict comparison. It does not
    compare against a fitted relative tolerance or a measured ULP allowance.
    """
    interaction, state, forces, _ = data
    pose = state.body_q.numpy()
    if (
        type(interaction) is not NeedleTissueInteraction
        or interaction.track_count != 0
        or pose.shape != (1, 7)
        or not np.array_equal(pose[0, 3:], [0, 0, 0, 1])
        or np.any(state.particle_qd.numpy() != 0)
    ):
        return None
    capacity = len(interaction.samples) * len(interaction.surface_triangles)
    anchors = SurfaceTangentAnchors(capacity, device=DEVICE)
    evaluate(data, anchors)
    count = int(anchors.count.numpy()[0])
    if count < 1 or count > capacity or np.any(anchors.flags.numpy() != 0):
        return None
    sources = anchors.sources.numpy()[:count]
    if len(np.unique(sources[:, 0])) < 2 or len(np.unique(sources, axis=0)) != count:
        return None
    ids = anchors.nodes.numpy()[:count]
    bary = anchors.bary.numpy()[:count].astype(np.float64)
    points = anchors.points.numpy()[:count]
    normals = anchors.normals.numpy()[:count].astype(np.float64)
    fn = anchors.scalars.numpy()[:count, 3].astype(np.float64)
    if not all(np.isfinite(x).all() for x in (bary, points, normals, fn)) or np.any(bary < 0) or np.any(fn < 0):
        return None
    # Every recorded fn is the actual nonnegative normal coefficient. The
    # uncapped Coulomb magnitude is at most mu*fn and its normalized direction
    # has component magnitude <=1, up to its FP32 arithmetic. Impulse limiting
    # can only reduce that magnitude. gamma_12 encloses those local operations;
    # no subtraction of nearly cancelling final resultants enters the bound.
    unit = float(np.finfo(np.float32).eps) / 2

    def gamma(n):
        return n * unit / (1 - n * unit)

    envelope = fn[:, None] * (np.abs(normals) + abs(interaction.friction)) * (1 + gamma(12))

    def cross_bound(position, force):
        position = np.abs(position.astype(np.float64))
        return np.stack(
            (
                position[:, 1] * force[:, 2] + position[:, 2] * force[:, 1],
                position[:, 2] * force[:, 0] + position[:, 0] * force[:, 2],
                position[:, 0] * force[:, 1] + position[:, 1] * force[:, 0],
            ),
            axis=1,
        ) * (1 + gamma(3))

    com = (pose[0, :3] + np.array([0.0007, 0.0003, 0], dtype=np.float32)).astype(np.float32)
    lever = (points - com).astype(np.float32)
    body_absolute = np.concatenate((envelope.sum(0), cross_bound(lever, envelope).sum(0)))[None]
    nodal_absolute = np.zeros(forces.shape + (3,), dtype=np.float64)
    nodal_count = np.zeros(forces.shape, dtype=np.int64)
    diagnostic_absolute = np.zeros((7, 3))
    diagnostic_count = np.zeros(7, dtype=np.int64)
    diagnostic_absolute[0], diagnostic_absolute[1] = envelope.sum(0), cross_bound(points, envelope).sum(0)
    diagnostic_count[:2] = count
    surface = interaction._surface.numpy()
    for slot in range(count):
        vertices = interaction.surface_triangles[sources[slot, 1]]
        for node, vertex, weight in zip(ids[slot], vertices, bary[slot], strict=True):
            reaction = weight * envelope[slot] * (1 + unit)
            torque = cross_bound(surface[vertex : vertex + 1], reaction[None])[0]
            if node >= 0:
                nodal_absolute[node] += reaction
                nodal_count[node] += 1
            if node < 0 or interaction.fixed[max(node, 0)] != 0:
                diagnostic_absolute[4] += reaction
                diagnostic_absolute[5] += torque
                diagnostic_count[4:6] += 1
            else:
                diagnostic_absolute[2] += reaction
                diagnostic_count[2] += 1
            diagnostic_absolute[3] += torque
            diagnostic_count[3] += 1
    return (
        2 * gamma(count) * body_absolute,
        2 * gamma(nodal_count)[:, None] * nodal_absolute,
        2 * gamma(diagnostic_count)[:, None] * diagnostic_absolute,
    )


@wp.kernel(enable_backward=False)
def coefficient_probe(velocity: wp.array[wp.vec3], matrices: wp.array[wp.mat33]):
    i = wp.tid()
    u = velocity[i]
    n = wp.normalize(wp.vec3(0.2, -0.3, 0.9))
    vn = wp.dot(n, u)
    p = float(0.00002)
    fn = 0.2 * p * (40.0 - 0.003 / 0.00015 * wp.min(vn, 0.0))
    matrices[i] = surface_contact_tangent(0.2, p, vn, fn, u - vn * n, n, 40.0, 0.003, 0.00015, 0.3, 0.001, 1.0 / 240.0)


class SurfaceTangentTests(unittest.TestCase):
    def test_one_contributing_sample_keeps_strict_force_parity(self):
        data = fixture(single_active_sample=True)
        expected = evaluate(data)
        anchors = SurfaceTangentAnchors(64, device=DEVICE)
        observed = evaluate(data, anchors)
        count = int(anchors.count.numpy()[0])
        self.assertGreater(count, 0)
        self.assertEqual(set(anchors.sources.numpy()[:count, 0]), {0})
        self.assertIsNone(atomic_roundoff_bounds(data))
        self.assert_physical_equal(expected, observed)

    def test_atomic_bound_is_scoped_and_preserves_exact_nonsummed_fields(self):
        data = fixture(fixed=[1], unmapped=[3])
        bounds = atomic_roundoff_bounds(data)
        self.assertIsNotNone(bounds)
        self.assertGreater(float(bounds[0].max()), 0.0)
        np.testing.assert_array_equal(bounds[2][6], 0.0)
        pose = data[1].body_q.numpy()
        pose[0, 3:] = [0, 0, 1, 0]
        data[1].body_q.assign(pose)
        self.assertIsNone(atomic_roundoff_bounds(data))

    def test_recording_preserves_original_physical_kernel_statements(self):
        original = ast.parse(inspect.getsource(insertion._surface_forces.func)).body[0]
        recorded = ast.parse(inspect.getsource(surface_forces_with_tangent.func)).body[0]
        self.assertEqual(
            [argument.arg for argument in recorded.args.args[: len(original.args.args)]],
            [argument.arg for argument in original.args.args],
        )
        recorded.name = original.name
        recorded.args.args = recorded.args.args[: len(original.args.args)]

        class RemoveNumericalRecording(ast.NodeTransformer):
            count = 0

            def visit_If(self, node):
                if isinstance(node.test, ast.Compare) and ast.unparse(node.test.left) == "anchor_count.shape[0]":
                    self.count += 1
                    return None
                return self.generic_visit(node)

        remove = RemoveNumericalRecording()
        recorded = remove.visit(recorded)
        self.assertEqual(remove.count, 1)
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(recorded, include_attributes=False))

    def assert_physical_equal(self, first, second, *, data=None):
        bounds = atomic_roundoff_bounds(data) if data is not None and wp.get_device(DEVICE).is_cuda else None
        for index, (a, b) in enumerate(zip(first, second, strict=True)):
            if bounds is None or index == 3:
                np.testing.assert_array_equal(a, b)
            else:
                self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
                difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
                self.assertTrue(
                    np.all(difference <= bounds[index]),
                    msg=f"Atomic output {index}: difference {difference} exceeds {bounds[index]}",
                )
                if index == 2:
                    # Tip resultant/penetration is written by sample zero,
                    # never atomically combined across samples.
                    np.testing.assert_array_equal(a[6], b[6])

    def test_multifacet_original_force_torque_flags_and_sample_weights_preserved(self):
        data = fixture()
        expected = evaluate(data)
        anchors = SurfaceTangentAnchors(64, device=DEVICE)
        observed = evaluate(data, anchors)
        self.assert_physical_equal(expected, observed, data=data)
        count = int(anchors.count.numpy()[0])
        self.assertGreaterEqual(count, 4)
        np.testing.assert_array_equal(anchors.flags.numpy(), 0)
        sources = anchors.sources.numpy()[:count]
        scalars = anchors.scalars.numpy()[:count]
        normals = anchors.normals.numpy()[:count]
        tangents = anchors.tangents.numpy()[:count]
        points = anchors.points.numpy()[:count]
        self.assertEqual(set(sources[:, 0]), {0, 1})
        self.assertGreater(len(set(sources[:, 1])), 1)
        for sample in (0, 1):
            self.assertAlmostEqual(float(scalars[sources[:, 0] == sample, 5].sum()), 1.0, places=6)
        speed = np.linalg.norm(tangents, axis=1)
        forces = scalars[:, 3, None] * (
            normals - data[0].friction * tangents / np.maximum(speed[:, None], data[0].speed_scale)
        )
        np.testing.assert_allclose(forces.sum(axis=0), observed[0][0, :3], rtol=2e-6, atol=1e-9)
        com = data[1].body_q.numpy()[0, :3] + [0.0007, 0.0003, 0]
        np.testing.assert_allclose(
            np.cross(points - com, forces).sum(axis=0), observed[0][0, 3:], rtol=3e-6, atol=1e-11
        )
        matrices = anchors.matrices.numpy()[:count]
        np.testing.assert_allclose(matrices, matrices.transpose(0, 2, 1), atol=2e-8)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(matrices).min()), -2e-8)

    def test_nonpenetrating_facets_still_normalize_recorded_contact(self):
        first, tets = block(slope=0, thickness=0.00005)
        second, other = block(slope=0, thickness=0.00005)
        second[:, 2] -= 0.00015
        data = fixture(np.vstack((first, second)), np.vstack((tets, other + len(first))))
        expected = evaluate(data)
        anchors = SurfaceTangentAnchors(64, device=DEVICE)
        self.assert_physical_equal(expected, evaluate(data, anchors), data=data)
        count = int(anchors.count.numpy()[0])
        scalars = anchors.scalars.numpy()[:count]
        sources = anchors.sources.numpy()[:count]
        for sample in (0, 1):
            # The second slab's unpenetrated surface contributes normalization,
            # but no force or tangent anchor. Renormalizing anchors would be wrong.
            self.assertLess(float(scalars[sources[:, 0] == sample, 5].sum()), 0.8)
        np.testing.assert_array_equal(expected[1][len(first) :], 0)

    def test_overflow_disabled_recording_and_impulse_cap_cannot_change_physics(self):
        data = fixture()
        expected = evaluate(data)
        full = SurfaceTangentAnchors(64, device=DEVICE)
        evaluate(data, full)
        for anchors in (SurfaceTangentAnchors(0, device=DEVICE), SurfaceTangentAnchors(1, device=DEVICE)):
            self.assert_physical_equal(expected, evaluate(data, anchors), data=data)
            self.assertEqual(int(anchors.flags.numpy()[0]), 1)
            self.assertEqual(int(anchors.count.numpy()[0]), int(full.count.numpy()[0]))
        disabled = SurfaceTangentAnchors(0, device=DEVICE, enabled=False)
        self.assert_physical_equal(expected, evaluate(data, disabled), data=data)
        np.testing.assert_array_equal(disabled.flags.numpy(), 0)
        limited = evaluate(data, implicit=False)
        self.assert_physical_equal(limited, evaluate(data, full, implicit=False), data=data)
        self.assertEqual(int(full.flags.numpy()[1]), 1)

    def test_balanced_transpose_and_positive_operator_include_fixed_and_unmapped_supports(self):
        data = fixture(fixed=[1], unmapped=[3])
        expected = evaluate(data)
        anchors = SurfaceTangentAnchors(64, device=DEVICE)
        observed = evaluate(data, anchors)
        self.assert_physical_equal(expected, observed, data=data)
        np.testing.assert_allclose(observed[2][0] + observed[2][2] + observed[2][4], 0, atol=1e-9)
        np.testing.assert_allclose(observed[2][1] + observed[2][3], 0, atol=1e-11)
        count = int(anchors.count.numpy()[0])
        nodes = anchors.nodes.numpy()[:count]
        bary = anchors.bary.numpy()[:count]
        matrices = anchors.matrices.numpy()[:count].astype(float)
        levers = anchors.levers.numpy()[:count].astype(float)
        points = anchors.points.numpy()[:count].astype(float)
        sources = anchors.sources.numpy()[:count]
        vertices = data[3]
        fixed = np.asarray(data[0].fixed, dtype=bool)
        n = len(vertices)
        size = 3 * n + 6
        x = np.random.default_rng(71).normal(size=size)
        operator = np.eye(size)
        nodal = np.zeros((n, 3))
        body_force = np.zeros(3)
        body_torque = np.zeros(3)
        support_force = np.zeros(3)
        support_torque = np.zeros(3)
        com = data[1].body_q.numpy()[0, :3] + [0.0007, 0.0003, 0]
        for ids, b, D, lever, _point, source in zip(nodes, bary, matrices, levers, points, sources, strict=True):
            skew = np.array([[0, -lever[2], lever[1]], [lever[2], 0, -lever[0]], [-lever[1], lever[0], 0]])
            J = np.zeros((3, size))
            J[:, 3 * n : 3 * n + 3] = np.eye(3)
            J[:, 3 * n + 3 :] = -skew
            for node, weight in zip(ids, b, strict=True):
                if node >= 0 and not fixed[node]:
                    J[:, 3 * node : 3 * node + 3] -= weight * np.eye(3)
            operator += (1 / 240) * J.T @ D @ J
            force = (1 / 240) * D @ (J @ x)
            body_force -= force
            body_torque -= np.cross(lever * 0.01, force)
            face_nodes = data[0].surface_triangles[source[1]]
            for node, vertex, weight in zip(ids, face_nodes, b, strict=True):
                if node >= 0:
                    nodal[node] += weight * force
                else:
                    support_force += weight * force
                    support_torque += np.cross(vertices[vertex], weight * force)
        np.testing.assert_allclose(body_force + nodal.sum(axis=0) + support_force, 0, atol=1e-10)
        np.testing.assert_allclose(
            body_torque + np.cross(com, body_force) + np.cross(vertices, nodal).sum(axis=0) + support_torque,
            0,
            atol=1e-11,
        )
        np.testing.assert_allclose(operator, operator.T, atol=2e-8)
        self.assertGreater(float(np.linalg.eigvalsh(operator).min()), 0.999999)
        self.assertGreater(np.linalg.norm(support_force), 0)
        self.assertGreater(np.linalg.norm(nodal[1]), 0)

    def test_frozen_friction_quadratic_bounds_both_branches_and_slip_reversals(self):
        epsilon = 0.0002
        for coefficient in (0.0, 1e-5, 0.03):

            def potential(speed, coefficient=coefficient):
                return coefficient * (speed * speed / (2 * epsilon) if speed <= epsilon else speed - epsilon / 2)

            for initial in (0.0, 0.1, 0.99, 1.0, 1.01, 5.0):
                initial_speed = initial * epsilon
                curvature = coefficient / max(initial_speed, epsilon)
                for final in (0.0, 0.1, 0.99, 1.0, 1.01, 5.0):
                    for angle in (0.0, 0.4, np.pi / 2, np.pi):
                        velocity = final * epsilon * np.array([np.cos(angle), np.sin(angle)])
                        speed = np.linalg.norm(velocity)
                        bound = potential(initial_speed) + 0.5 * curvature * (speed**2 - initial_speed**2)
                        self.assertGreaterEqual(bound - potential(speed), -1e-18)
                # Tangency and gradient match the unchanged force at the base.
                self.assertEqual(potential(initial_speed), potential(initial_speed) + 0.5 * curvature * 0)
                self.assertAlmostEqual(
                    curvature * initial_speed, coefficient * initial_speed / max(initial_speed, epsilon)
                )

    def test_majorizes_frozen_load_friction_jacobian_and_matches_regularized_branch(self):
        normal = np.array([0.2, -0.3, 0.9])
        normal /= np.linalg.norm(normal)
        direction = np.cross(normal, [0, 1, 0])
        direction /= np.linalg.norm(direction)
        velocities = np.asarray(
            [speed * direction + vn * normal for speed in (0.0002, 0.002) for vn in (-0.005, 0.005)], np.float32
        )
        matrices = wp.empty(4, dtype=wp.mat33, device=DEVICE)
        wp.launch(coefficient_probe, 4, [wp.array(velocities, dtype=wp.vec3, device=DEVICE), matrices], device=DEVICE)
        for velocity, actual in zip(velocities.astype(float), matrices.numpy(), strict=True):
            vn = normal @ velocity
            tangent = velocity - vn * normal
            speed = np.linalg.norm(tangent)
            projection = np.eye(3) - np.outer(normal, normal)
            fn = 0.2 * 0.00002 * (40 - 0.003 / 0.00015 * min(vn, 0))
            dn = 0.2 * ((40 - 0.003 / 0.00015 * min(vn, 0)) / 240 + (0.003 * 0.00002 / 0.00015 if vn < 0 else 0))
            expected = dn * np.outer(normal, normal) + 0.3 * fn / max(speed, 0.001) * projection
            np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=2e-8)

            def friction(v, fn=fn, projection=projection):
                t = projection @ v
                return -0.3 * fn * t / max(np.linalg.norm(t), 0.001)

            jacobian = np.column_stack(
                [
                    (friction(velocity + np.eye(3)[axis] * 1e-8) - friction(velocity - np.eye(3)[axis] * 1e-8)) / 2e-8
                    for axis in range(3)
                ]
            )
            exact = dn * np.outer(normal, normal) - jacobian
            difference = actual - exact
            self.assertGreaterEqual(float(np.linalg.eigvalsh(difference).min()), -2e-8)
            if speed < 0.001:
                np.testing.assert_allclose(actual, exact, rtol=5e-6, atol=2e-8)
            else:
                radial = tangent / speed
                expected_difference = 0.3 * fn / speed * np.outer(radial, radial)
                np.testing.assert_allclose(difference, expected_difference, rtol=5e-6, atol=2e-8)


if __name__ == "__main__":
    unittest.main()
