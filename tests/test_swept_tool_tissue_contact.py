# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Swept-capsule tool/tissue contact: geometry of a row, allowed moves, node holds, both force laws, storage.

Every expected number is a hand computation on a flat slab or on one triangle; the slab tests
run the real tissue solver on the device of ``NEWTON_TEST_DEVICE``.
"""

import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_suturebot import SolverFEM
from newton_solver_suturebot._src.solvers.coupled.interface import swept_tool_tissue_contact as swept
from newton_solver_suturebot._src.solvers.coupled.interface.swept_tool_tissue_contact import (
    INTEGRAL_RATIO,
    MOVE_FRACTION,
    SweptToolTissueContact,
    capsule_pieces,
)
from newton_solver_suturebot._src.solvers.fem.node_holds import HOLD_FACTOR, NodeHolds

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
MM = 1.0e-3
DT = 1.0 / 240.0
RADIUS = 0.5 * MM
# One effective mass of the tool for every piece and direction: K_c = m / h^2.
STIFFNESS = 1280.0
MASS = STIFFNESS * DT**2
# One capsule of 3 mm between its end points on body 1, along the body's x axis: seven probes 0.5 mm apart.
ENDPOINTS = np.array([[[-1.5 * MM, 0.0, 0.0], [1.5 * MM, 0.0, 0.0]]])
COM = np.array([0.4 * MM, -0.3 * MM, 0.2 * MM])


@wp.kernel(enable_backward=False)
def _closest_pairs(
    p0: wp.array[wp.vec3],
    p1: wp.array[wp.vec3],
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    s: wp.array[float],
    point: wp.array[wp.vec3],
    distance: wp.array[float],
    crossing: wp.array[int],
):
    index = wp.tid()
    pair = swept._segment_triangle(p0[index], p1[index], a, b, c)
    s[index] = pair.s
    point[index] = pair.point
    distance[index] = pair.distance
    crossing[index] = pair.crossing


@wp.kernel(enable_backward=False)
def _rows(
    a0: wp.array[wp.vec3],
    b0: wp.array[wp.vec3],
    a1: wp.array[wp.vec3],
    b1: wp.array[wp.vec3],
    radius: float,
    inward: wp.array[int],
    outward: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    depth: wp.array[float],
    push: wp.array[wp.vec3],
):
    index = wp.tid()
    row = swept._part_row(a0[index], b0[index], a1[index], b1[index], radius, inward[index], outward, a, b, c)
    depth[index] = row.depth
    push[index] = row.push


@wp.kernel(enable_backward=False)
def _predict(
    pose: wp.array[wp.transform],
    velocity: wp.array[wp.spatial_vector],
    com: wp.vec3,
    seconds: float,
    out: wp.array[wp.transform],
):
    out[wp.tid()] = swept._predicted_pose(pose[wp.tid()], velocity[wp.tid()], com, seconds)


def _vec3(values):
    return wp.array(np.asarray(values, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3, device=DEVICE)


def _slab_mesh(nx, ny, nz, spacing):
    """Nodes of a box grid with its top at z = 0, its cubes cut into six tetrahedra each, and its closed surface."""
    grid = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), axis=-1).reshape(-1, 3)
    points = grid.astype(float) * spacing
    points[:, 2] -= (nz - 1) * spacing

    def node(i, j, k):
        return (i * ny + j) * nz + k

    tetrahedra = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # Kuhn's cut: the six orders of the three axis steps from one corner to the opposite one.
                for order in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
                    corner = [i, j, k]
                    path = [node(*corner)]
                    for axis in order:
                        corner[axis] += 1
                        path.append(node(*corner))
                    tetrahedra.append(path)
    tetrahedra = np.asarray(tetrahedra)
    edges = points[tetrahedra[:, 1:]] - points[tetrahedra[:, :1]]
    flipped = np.linalg.det(edges) < 0
    tetrahedra[flipped] = tetrahedra[flipped][:, [0, 2, 1, 3]]
    faces = {}
    for tet in tetrahedra:
        for opposite in range(4):
            face = [tet[index] for index in range(4) if index != opposite]
            normal = np.cross(points[face[1]] - points[face[0]], points[face[2]] - points[face[0]])
            if normal @ (points[tet[opposite]] - points[face[0]]) > 0:
                face = [face[0], face[2], face[1]]
            faces.setdefault(tuple(sorted(face)), []).append(face)
    surface = np.asarray([found[0] for found in faces.values() if len(found) == 1])
    return points, tetrahedra, surface


class _Slab:
    """A slab of 8 x 8 x 2 mm with its bottom fixed, two rigid bodies and the capsule on body 1.

    The surface vertices with y above 2.5 mm are given as outside the simulated region, so the
    capsule finds tissue that yields at y = 1 mm and tissue that is held at y = 6 mm.
    """

    def __init__(self, *, outside=True, inverse_mass=None):
        points, tetrahedra, surface = _slab_mesh(9, 9, 3, MM)
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for point in points:
            builder.add_particle(pos=tuple(point), vel=(0, 0, 0), mass=1.0e-6)
        for tet in tetrahedra:
            builder.add_tetrahedron(*(int(index) for index in tet), k_mu=2.0e4, k_lambda=8.0e4, k_damp=0.0)
        for _ in range(2):
            builder.add_body(mass=0.01, inertia=wp.mat33(1e-8, 0, 0, 0, 1e-8, 0, 0, 0, 1e-8))
        self.model = builder.finalize(device=DEVICE)
        com = self.model.body_com.numpy()
        com[1] = COM
        self.model.body_com.assign(com)
        self.rest = points
        self.tetrahedra = self.model.tet_indices.numpy()
        self.fixed = np.flatnonzero(points[:, 2] < -1.5 * MM)
        self.solver = SolverFEM(
            self.model,
            dt=DT,
            fixed_particle_indices=self.fixed,
            preconditioner_type="block_diag",
            solver_type="cg",
            linear_solver_maxiter=200,
            min_stretch=0.2,
            max_stretch=5.0,
        )
        self.solver.linear_solver_tolerance = 1.0e-6
        self.state, self.next = self.model.state(), self.model.state()
        self.solver.refresh_tangent(self.state.particle_q, DT)
        self.surface_points = wp.array(points.astype(np.float32), dtype=wp.vec3, device=DEVICE)
        self.triangles = wp.array(surface.reshape(-1).astype(np.int32), dtype=int, device=DEVICE)
        self.mesh = wp.Mesh(self.surface_points, self.triangles)
        particle = np.arange(len(points), dtype=np.int32)
        if outside:
            particle[points[:, 1] > 2.5 * MM] = -1
        self.point_particle = particle
        self.step_seconds = wp.array([DT], dtype=wp.float64, device=DEVICE)
        self.contact = SweptToolTissueContact(
            self.model,
            self.solver,
            self.mesh,
            self.triangles,
            wp.array(particle, dtype=int, device=DEVICE),
            self.fixed,
            [1],
            ENDPOINTS,
            [RADIUS],
            [RADIUS],
            [0],
            step_seconds=self.step_seconds,
            full_step=DT,
            inverse_mass=MASS if inverse_mass is None else inverse_mass,
            reach=0.01,
        )
        self.body_q = wp.array(
            np.array([[0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 1]], dtype=np.float32),
            dtype=wp.transform,
            device=DEVICE,
        )
        self.body_qd = wp.zeros(2, dtype=wp.spatial_vector, device=DEVICE)

    def place(self, x, y, height, velocity=(0.0, 0.0, 0.0)):
        """The capsule axis along world x through (x, y) at ``height`` above z = 0, moving with ``velocity``."""
        poses = self.body_q.numpy()
        poses[1] = (x, y, height, 0, 0, 0, 1)
        self.body_q.assign(poses)
        rates = np.zeros((2, 6), dtype=np.float32)
        rates[1, :3] = velocity
        self.body_qd.assign(rates)

    def refit(self, state):
        self.surface_points.assign(state.particle_q.numpy())
        self.mesh.refit()

    def update(self, state=None):
        state = self.state if state is None else state
        self.refit(state)
        self.contact.update(self.body_q, self.body_qd, state.particle_q, state.particle_qd)

    def volumes(self, positions):
        vertices = positions[self.tetrahedra].astype(np.float64)
        return np.linalg.det(vertices[:, 1:] - vertices[:, :1])

    def under(self, y=1.0):
        """Free surface nodes on the line at ``y`` [mm] whose parts lie under the capsule axis from x = 2.5 to 5.5 mm."""
        rest = self.rest
        return np.flatnonzero(
            (np.abs(rest[:, 2]) < 1e-9)
            & (np.abs(rest[:, 1] - y * MM) < 1e-9)
            & (rest[:, 0] > 1.5 * MM)
            & (rest[:, 0] < 6.5 * MM)
        )


class RowGeometryTests(unittest.TestCase):
    """One axis piece against the triangle (0, 0, 0), (1, 0, 0), (0, 1, 0) whose normal is +z."""

    a, b, c = wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)

    def test_pieces_cut_the_axis_between_its_probes_into_lengths_of_at_most_one_radius(self):
        bodies, probes, starts, ends, radii, capsule, firsts, counts = capsule_pieces(
            [3, 6],
            [[[0, 0, -0.243], [0, 0, -0.003]], [[-0.00075, 0.0015, 0], [-0.00075, 0.009, 0]]],
            [0.0043, 0.00065],
            [0.0042, 0.00095],
        )
        self.assertEqual((firsts.tolist(), counts.tolist()), ([0, 59], [59, 9]))
        self.assertEqual(capsule.tolist(), [0] * 59 + [1] * 9)
        for first, count, radius in zip(firsts, counts, (0.0042, 0.00095), strict=True):
            piece = slice(first, first + count)
            np.testing.assert_array_equal(starts[first], probes[first])
            np.testing.assert_array_equal(ends[first + count - 1], probes[first + count - 1])
            np.testing.assert_allclose(ends[piece][:-1], starts[piece][1:], atol=1e-15)
            np.testing.assert_allclose(ends[piece][:-1], 0.5 * (probes[piece][:-1] + probes[piece][1:]), atol=1e-15)
            self.assertLessEqual(np.linalg.norm(ends[piece] - starts[piece], axis=1).max(), radius * (1 + 1e-12))
            np.testing.assert_array_equal(radii[piece], radius)
        # A capsule shorter than one sphere: one probe, a piece of no length.
        _, probes, starts, ends, *_ = capsule_pieces([0], [[[0, 0, 0], [0.001, 0, 0]]], [0.0005], [0.002])
        np.testing.assert_array_equal(starts, probes)
        np.testing.assert_array_equal(ends, probes)

    def test_closest_pair_of_a_segment_and_a_triangle(self):
        cases = (
            # start, end, parameter, point, distance, crossing
            ((0.2, 0.2, 0.5), (0.4, 0.2, 0.5), 0.0, (0.2, 0.2, 0.0), 0.5, 0),  # parallel above: its first end
            ((0.2, 0.2, 0.5), (0.2, 0.2, -0.5), 0.5, (0.2, 0.2, 0.0), 0.0, 1),  # through the triangle
            ((2.0, 0.0, 1.0), (3.0, 0.0, 1.0), 0.0, (1.0, 0.0, 0.0), np.sqrt(2.0), 0),  # beyond a vertex
            ((0.5, -1.0, 0.3), (0.5, -0.2, 0.3), 1.0, (0.5, 0.0, 0.0), np.sqrt(0.13), 0),  # its end beside an edge
            (
                (2.0, 2.0, 0.5),
                (2.0, 2.0, -0.5),
                0.5,
                (0.5, 0.5, 0.0),
                1.5 * np.sqrt(2.0),
                0,
            ),  # through the plane outside
            ((0.25, 0.25, 0.3), (0.25, 0.25, 0.3), 0.0, (0.25, 0.25, 0.0), 0.3, 0),  # a piece of no length
        )
        count = len(cases)
        outputs = [
            wp.zeros(count, dtype=float, device=DEVICE),
            wp.zeros(count, dtype=wp.vec3, device=DEVICE),
            wp.zeros(count, dtype=float, device=DEVICE),
            wp.zeros(count, dtype=int, device=DEVICE),
        ]
        wp.launch(
            _closest_pairs,
            count,
            inputs=[_vec3([case[0] for case in cases]), _vec3([case[1] for case in cases]), self.a, self.b, self.c],
            outputs=outputs,
            device=DEVICE,
        )
        s, point, distance, crossing = (array.numpy() for array in outputs)
        np.testing.assert_allclose(s, [case[2] for case in cases], atol=1e-6)
        np.testing.assert_allclose(point, [case[3] for case in cases], atol=1e-6)
        np.testing.assert_allclose(distance, [case[4] for case in cases], atol=1e-6)
        np.testing.assert_array_equal(crossing, [case[5] for case in cases])

    def test_row_depth_and_push_direction(self):
        radius = 0.5
        cases = (
            # a0, b0, a1, b1, inward, depth, push
            # Outside, parallel 0.3 above the part: R - distance, pushed straight down.
            ((0.2, 0.2, 0.3), (0.4, 0.2, 0.3), (0.2, 0.2, 0.3), (0.4, 0.2, 0.3), 0, 0.2, (0, 0, -1)),
            # Outside with a clearance: negative depth.
            ((0.2, 0.2, 0.8), (0.4, 0.2, 0.8), (0.2, 0.2, 0.8), (0.4, 0.2, 0.8), 0, -0.3, (0, 0, -1)),
            # Outside beside the edge y = 0, at its height: pushed sideways into the part.
            ((0.5, -0.3, 0.0), (0.5, -0.3, 0.0), (0.5, -0.3, 0.0), (0.5, -0.3, 0.0), 0, 0.2, (0, 1, 0)),
            # Probe under the surface by 0.1, under the lattice point (0.25, 0.25): 0.1 + R along the outward direction.
            ((0.25, 0.25, -0.1), (0.25, 0.25, -0.1), (0.25, 0.25, -0.1), (0.25, 0.25, -0.1), 1, 0.6, (0, 0, -1)),
            # Swept through the surface from 0.4 above to 0.4 below: the part must go under the last pose, 0.4 + R.
            ((0.25, 0.25, 0.4), (0.25, 0.25, 0.4), (0.25, 0.25, -0.4), (0.25, 0.25, -0.4), 1, 0.9, (0, 0, -1)),
            # Swept from below to above with the probe counted inward: the deepest pose is the first, 0.3 + R.
            ((0.25, 0.25, -0.3), (0.25, 0.25, -0.3), (0.25, 0.25, 0.2), (0.25, 0.25, 0.2), 1, 0.8, (0, 0, -1)),
            # Probe outside, the piece through the part from 0.2 above to 0.2 below: along the part's normal, 0.2 + R.
            ((0.25, 0.25, 0.2), (0.25, 0.25, -0.2), (0.25, 0.25, 0.2), (0.25, 0.25, -0.2), 0, 0.7, (0, 0, -1)),
        )
        count = len(cases)
        depth = wp.zeros(count, dtype=float, device=DEVICE)
        push = wp.zeros(count, dtype=wp.vec3, device=DEVICE)
        wp.launch(
            _rows,
            count,
            inputs=[
                *(_vec3([case[index] for case in cases]) for index in range(4)),
                radius,
                wp.array([case[4] for case in cases], dtype=int, device=DEVICE),
                wp.vec3(0.0, 0.0, 1.0),
                self.a,
                self.b,
                self.c,
            ],
            outputs=[depth, push],
            device=DEVICE,
        )
        np.testing.assert_allclose(depth.numpy(), [case[5] for case in cases], atol=2e-6)
        np.testing.assert_allclose(push.numpy(), [case[6] for case in cases], atol=2e-6)

    def test_envelope_has_no_row_for_a_part_beside_the_capsule(self):
        depth = wp.zeros(1, dtype=float, device=DEVICE)
        push = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        point = _vec3([(3.0, 3.0, -0.1)])
        wp.launch(
            _rows,
            1,
            inputs=[
                point,
                point,
                point,
                point,
                0.5,
                wp.array([1], dtype=int, device=DEVICE),
                wp.vec3(0.0, 0.0, 1.0),
                self.a,
                self.b,
                self.c,
            ],
            outputs=[depth, push],
            device=DEVICE,
        )
        self.assertEqual(depth.numpy()[0], -np.inf)

    def test_predicted_pose_moves_the_centre_of_mass_and_turns_about_it(self):
        com = wp.vec3(0.01, 0.0, 0.0)
        quarter = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi)
        poses = wp.array(
            np.array([[1, 2, 3, 0, 0, 0, 1], [1, 2, 3, *quarter], [0, 0, 0, 0, 0, 0, 1]], dtype=np.float32),
            dtype=wp.transform,
            device=DEVICE,
        )
        rates = wp.array(
            np.array([[0.3, -0.2, 0.1, 0, 0, 0], [0.3, -0.2, 0.1, 0, 0, 0], [0, 0, 0, 0, 0, 2.0]], dtype=np.float32),
            dtype=wp.spatial_vector,
            device=DEVICE,
        )
        out = wp.zeros(3, dtype=wp.transform, device=DEVICE)
        wp.launch(_predict, 3, inputs=[poses, rates, com, 0.01], outputs=[out], device=DEVICE)
        result = out.numpy()
        np.testing.assert_allclose(result[0], [1.003, 1.998, 3.001, 0, 0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(result[1], [1.003, 1.998, 3.001, *quarter], atol=1e-6)
        # A turn of 0.02 rad about z through the centre of mass at (0.01, 0, 0): the origin goes round it.
        angle = 0.02
        np.testing.assert_allclose(result[2, 3:], [0, 0, np.sin(0.5 * angle), np.cos(0.5 * angle)], atol=1e-6)
        np.testing.assert_allclose(result[2, :3], [0.01 * (1 - np.cos(angle)), -0.01 * np.sin(angle), 0], atol=1e-7)


class AllowedMoveTests(unittest.TestCase):
    def test_allowed_move_is_an_eighth_of_the_stretch_margin_times_the_rest_altitude(self):
        size = MM
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for point in ((0, 0, 0), (size, 0, 0), (0, size, 0), (0, 0, size)):
            builder.add_particle(pos=point, vel=(0, 0, 0), mass=1e-6)
        builder.add_tetrahedron(0, 1, 2, 3, k_mu=1e4, k_lambda=4e4, k_damp=0)
        model = builder.finalize(device=DEVICE)
        # Rest altitudes: the corner nodes stand one edge above their opposite faces, node 0 size / sqrt(3).
        altitude = np.array([size / np.sqrt(3.0), size, size, size])
        for squeeze in (1.0, 0.5, 0.2, 0.1, -0.5):
            positions = model.particle_q.numpy().copy()
            positions[3, 2] = squeeze * size
            allowed = wp.full(4, np.inf, dtype=float, device=DEVICE)
            back = wp.full(4, np.inf, dtype=float, device=DEVICE)
            wp.launch(
                swept._allowed_moves,
                1,
                inputs=[wp.array(positions, dtype=wp.vec3, device=DEVICE), model.tet_indices, model.tet_poses, 0.2],
                outputs=[allowed, back],
                device=DEVICE,
            )
            # The smallest principal stretch is the squeeze along z (negative for the inverted element).
            np.testing.assert_allclose(
                allowed.numpy(), MOVE_FRACTION * (squeeze - 0.2) * altitude, rtol=2e-5, atol=1e-11
            )
            np.testing.assert_allclose(back.numpy(), MOVE_FRACTION * squeeze * altitude, rtol=2e-5, atol=1e-11)

    def test_four_nodes_that_use_their_allowed_moves_leave_the_element_above_the_stretch_limit(self):
        rng = np.random.default_rng(3)
        size = MM
        rest = np.array(((0, 0, 0), (size, 0, 0), (0, size, 0), (0, 0, size)), dtype=float)
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for point in rest:
            builder.add_particle(pos=tuple(point), vel=(0, 0, 0), mass=1e-6)
        builder.add_tetrahedron(0, 1, 2, 3, k_mu=1e4, k_lambda=4e4, k_damp=0)
        model = builder.finalize(device=DEVICE)
        inverse = np.linalg.inv((rest[1:] - rest[0]).T)
        for _ in range(40):
            positions = rest * rng.uniform(0.3, 1.2, size=3) + rng.normal(0.0, 0.03 * size, size=(4, 3))
            allowed = wp.full(4, np.inf, dtype=float, device=DEVICE)
            back = wp.full(4, np.inf, dtype=float, device=DEVICE)
            wp.launch(
                swept._allowed_moves,
                1,
                inputs=[
                    wp.array(positions.astype(np.float32), dtype=wp.vec3, device=DEVICE),
                    model.tet_indices,
                    model.tet_poses,
                    0.2,
                ],
                outputs=[allowed, back],
                device=DEVICE,
            )
            before = np.linalg.svd((positions[1:] - positions[0]).T @ inverse, compute_uv=False).min()
            # Every node moves by its whole allowed move in a random direction: Weyl's bound, half of the margin.
            directions = rng.normal(size=(4, 3))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            moved = positions + np.maximum(allowed.numpy(), 0.0)[:, None] * directions
            after = np.linalg.svd((moved[1:] - moved[0]).T @ inverse, compute_uv=False).min()
            self.assertGreaterEqual(after, min(before, 0.2 + 0.5 * (before - 0.2)) - 1e-5)
            self.assertGreater(np.linalg.det((moved[1:] - moved[0]).T @ inverse), 0.0)


# A node of the slabs' meshes (Kuhn's cut of a cube grid) has at most 24 tetrahedra: the largest number of terms
# that the device adds, by atomics, into one entry of a nodal force or of a matrix block.
ATOMIC_TERMS = 24
# A float32 sum of N terms depends on their order by at most 2 (N - 1) u sum|terms| (u = 2^-24, the unit
# roundoff), and sum|terms| is at most N times the largest term. The terms of the sums here (one tetrahedron's
# force on a node, its share of a matrix block) are of the size of the largest entry of the summed array.
ATOMIC_ORDER = 2.0 * (ATOMIC_TERMS - 1) * ATOMIC_TERMS * 2.0**-24


def _assert_same(actual, desired, summed=True):
    """Equal bits on the CPU, and on a CUDA device for what no atomic sums (``summed`` False).

    On a CUDA device sums by atomics are not ordered: arrays that hold such sums agree within
    :data:`ATOMIC_ORDER` of the largest finite entry of the array (not of each entry: a small entry of a block
    or of a force is a difference of large terms). The positions after tissue steps of two solvers are compared
    with the same bound: their solves end within the solver's tolerance of each other (1e-7 of moves of
    micrometres there), far below it. Entries that are not finite (a row depth of minus infinity: no row) must
    be the same entries.
    """
    actual, desired = np.asarray(actual), np.asarray(desired)
    if not (summed and wp.get_device(DEVICE).is_cuda) or not np.issubdtype(desired.dtype, np.floating):
        np.testing.assert_array_equal(actual, desired)
        return
    finite = np.isfinite(desired)
    np.testing.assert_array_equal(np.isfinite(actual), finite)
    np.testing.assert_array_equal(actual[~finite], desired[~finite])
    if finite.any():
        scale = float(np.abs(desired[finite]).max())
        np.testing.assert_allclose(actual[finite], desired[finite], rtol=0.0, atol=ATOMIC_ORDER * scale)


def _dense(matrix):
    offsets, columns, values = matrix.offsets.numpy(), matrix.columns.numpy(), matrix.values.numpy()
    result = np.zeros((3 * matrix.nrow, 3 * matrix.ncol))
    for row in range(matrix.nrow):
        for block in range(offsets[row], offsets[row + 1]):
            column = columns[block]
            result[3 * row : 3 * row + 3, 3 * column : 3 * column + 3] += values[block]
    return result


class NodeHoldTests(unittest.TestCase):
    """A slab with one node moved and held: the hold in every copy of the matrix, and bit identity without one."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _slab(self):
        points, tetrahedra, _ = _slab_mesh(5, 5, 3, MM)
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for point in points:
            builder.add_particle(pos=tuple(point), vel=(0, 0, 0), mass=1.0e-6)
        for tet in tetrahedra:
            builder.add_tetrahedron(*(int(index) for index in tet), k_mu=2.0e4, k_lambda=8.0e4, k_damp=0.0)
        model = builder.finalize(device=DEVICE)
        fixed = np.flatnonzero(points[:, 2] < -1.5 * MM)
        solver = SolverFEM(
            model,
            dt=DT,
            fixed_particle_indices=fixed,
            preconditioner_type="block_diag",
            solver_type="cg",
            linear_solver_maxiter=300,
            min_stretch=0.2,
            max_stretch=5.0,
        )
        solver.linear_solver_tolerance = 1.0e-7
        return model, solver, points, fixed

    def test_refuses_fixed_nodes_and_a_solver_without_its_final_structure(self):
        model, solver, _, fixed = self._slab()
        held = wp.zeros(model.particle_count, dtype=int, device=DEVICE)
        top = np.flatnonzero(np.abs(model.particle_q.numpy()[:, 2]) < 1e-9)
        with self.assertRaisesRegex(RuntimeError, "refresh_tangent"):
            NodeHolds(solver, top, held)
        solver.refresh_tangent(model.particle_q, DT)
        with self.assertRaisesRegex(ValueError, "fixed particle"):
            NodeHolds(solver, [int(fixed[0])], held)
        with self.assertRaisesRegex(ValueError, "distinct"):
            NodeHolds(solver, [int(top[0]), int(top[0])], held)
        # None of the refused sets joined the solver; the first that is accepted does, and a second is refused:
        # the solver refreshes one set after it has rewritten its matrix.
        self.assertIsNone(getattr(solver, "_node_holds", None))
        holds = NodeHolds(solver, top, held)
        self.assertIs(solver._node_holds, holds)
        with self.assertRaisesRegex(ValueError, "already has node holds"):
            NodeHolds(solver, top[:1], held)
        self.assertIs(solver._node_holds, holds)

    def test_hold_kernel_writes_the_block_and_its_inverse_to_every_copy(self):
        # Five separate arrays, as on a CUDA device (on the CPU the solver has one matrix and one inverse
        # diagonal, and the kernel is handed each of them twice): node 2 of 4 particles is held, node 0 is not.
        nodes = wp.array([0, 2], dtype=int, device=DEVICE)
        held = wp.array([0, 0, 1, 0], dtype=int, device=DEVICE)
        blocks = np.array([np.diag([2.0, 3.0, 4.0]), [[4.0, 1.0, 0.0], [1.0, 5.0, 0.0], [0.0, 0.0, 6.0]]])
        pristine = wp.array(blocks, dtype=wp.mat33f, device=DEVICE)
        slots = wp.array([1, 3], dtype=int, device=DEVICE)
        compact_slots = wp.array([0, 2], dtype=int, device=DEVICE)
        compact_rows = wp.array([0, 1], dtype=int, device=DEVICE)
        values, projected = (wp.zeros(5, dtype=wp.mat33f, device=DEVICE) for _ in range(2))
        inverse = wp.zeros(4, dtype=wp.mat33f, device=DEVICE)
        compact_values = wp.zeros(3, dtype=wp.mat33f, device=DEVICE)
        compact_inverse = wp.zeros(2, dtype=wp.mat33f, device=DEVICE)
        from newton_solver_suturebot._src.solvers.fem import node_holds

        wp.launch(
            node_holds._apply_holds,
            2,
            inputs=[nodes, held, pristine, HOLD_FACTOR, slots],
            outputs=[values, inverse, projected, compact_slots, compact_values, compact_rows, compact_inverse],
            device=DEVICE,
        )
        # The held block: its own block plus HOLD_FACTOR times the mean of its diagonal (5) on the diagonal.
        expected = [blocks[0], blocks[1] + HOLD_FACTOR * 5.0 * np.eye(3)]
        for index, (slot, node, compact_slot, row) in enumerate(((1, 0, 0, 0), (3, 2, 2, 1))):
            for array, entry in ((values, slot), (projected, slot), (compact_values, compact_slot)):
                np.testing.assert_allclose(array.numpy()[entry], expected[index], rtol=1e-6)
            for array, entry in ((inverse, node), (compact_inverse, row)):
                np.testing.assert_allclose(array.numpy()[entry], np.linalg.inv(expected[index]), rtol=1e-5)
        # Nothing else is written.
        for array, written in ((values, (1, 3)), (projected, (1, 3)), (compact_values, (0, 2)), (inverse, (0, 2))):
            others = np.setdiff1d(np.arange(array.shape[0]), written)
            np.testing.assert_array_equal(array.numpy()[others], 0.0)

    def test_without_a_held_node_the_matrix_the_inverse_blocks_and_a_step_keep_their_bits(self):
        results = []
        for with_holds in (False, True):
            model, solver, points, _ = self._slab()
            state, following = model.state(), model.state()
            solver.refresh_tangent(state.particle_q, DT)
            top = np.flatnonzero(np.abs(points[:, 2]) < 1e-9)
            if with_holds:
                holds = NodeHolds(solver, top, wp.zeros(model.particle_count, dtype=int, device=DEVICE))
            positions = state.particle_q.numpy()
            positions[top[len(top) // 2], 2] -= 0.05 * MM
            state.particle_q.assign(positions)
            for _ in range(3):
                solver.refresh_tangent(state.particle_q, DT)
                if with_holds:
                    holds.apply()
                solver.step(state, following, None, None, DT)
                state, following = following, state
            results.append(
                (
                    state.particle_q.numpy().copy(),
                    solver.A_bsr.values.numpy().copy(),
                    solver._pc_inv_diag.numpy().copy(),
                )
            )
        for without, with_ in zip(*results, strict=True):
            _assert_same(with_, without)

    def test_a_moved_and_held_node_stays_while_the_others_find_their_equilibrium(self):
        model, solver, points, fixed = self._slab()
        state, following = model.state(), model.state()
        solver.refresh_tangent(state.particle_q, DT)
        top = np.flatnonzero(np.abs(points[:, 2]) < 1e-9)
        node = int(top[len(top) // 2])
        held = wp.zeros(model.particle_count, dtype=int, device=DEVICE)
        holds = NodeHolds(solver, top, held)
        pristine = _dense(solver.A_bsr)
        # Move the node 0.05 mm into the slab and hold it there.
        positions = state.particle_q.numpy()
        positions[node, 2] -= 0.05 * MM
        state.particle_q.assign(positions)
        flags = np.zeros(model.particle_count, dtype=np.int32)
        flags[node] = 1
        held.assign(flags)
        holds.apply()
        # The hold is HOLD_FACTOR times the mean of the node's own diagonal, on its diagonal block only.
        matrix = _dense(solver.A_bsr)
        expected = pristine.copy()
        block = slice(3 * node, 3 * node + 3)
        expected[block, block] += HOLD_FACTOR * np.trace(pristine[block, block]) / 3.0 * np.eye(3)
        np.testing.assert_allclose(matrix, expected, rtol=1e-6, atol=0.0)
        np.testing.assert_allclose(solver._pc_inv_diag.numpy()[node], np.linalg.inv(expected[block, block]), rtol=1e-4)
        free = np.setdiff1d(np.arange(model.particle_count), fixed)
        dofs = (3 * free[:, None] + np.arange(3)).ravel()
        for _ in range(2):
            elastic = solver.eval_tetrahedral_forces(model, model.control(), state).numpy().copy()
            velocity = state.particle_qd.numpy().astype(float).ravel()
            mass = np.repeat(model.particle_mass.numpy(), 3)
            rhs = DT * elastic.ravel() - (matrix - np.diag(mass) - (matrix - pristine)) @ velocity
            increment = np.zeros(3 * model.particle_count)
            increment[dofs] = np.linalg.solve(matrix[np.ix_(dofs, dofs)], rhs[dofs])
            before = state.particle_q.numpy().copy()
            solver.step(state, following, None, None, DT)
            # The solve is that of the matrix with the hold: compared with a dense solve of the same system.
            scale = np.abs(increment).max()
            np.testing.assert_allclose(solver.velocity_increment.numpy().ravel(), increment, atol=2e-3 * scale)
            moved = following.particle_q.numpy() - before
            # The held node moves by about 1 / HOLD_FACTOR of what its neighbours pull it with (its row of the
            # matrix sums a dozen neighbours); the others move.
            self.assertLess(np.abs(moved[node]).max(), 12.0 / HOLD_FACTOR * np.abs(moved[top]).max())
            self.assertGreater(np.abs(moved[top]).max(), 1e-8)
            state, following = following, state
            velocities = state.particle_qd.numpy()
            velocities[node] = 0.0
            state.particle_qd.assign(velocities)
        # A tangent refresh rewrites the matrix; the hold is written again on the new block.
        solver.refresh_tangent(state.particle_q, DT)
        refreshed = _dense(solver.A_bsr)
        np.testing.assert_allclose(
            np.trace(refreshed[block, block]),
            (1.0 + HOLD_FACTOR) * np.trace(holds.pristine.numpy()[top.tolist().index(node)]),
            rtol=1e-5,
        )
        # Released: every copy holds the assembled block again, bit for bit.
        held.zero_()
        holds.apply()
        released = solver.A_bsr.values.numpy().copy()
        inverse = solver._pc_inv_diag.numpy().copy()
        solver.refresh_tangent(state.particle_q, DT)
        _assert_same(solver.A_bsr.values.numpy(), released)
        _assert_same(solver._pc_inv_diag.numpy(), inverse)
        fixed_particles = solver._fixed_particles
        if fixed_particles.compact_matrix is not None:
            # CUDA: the projected and the compact copy hold the same blocks as the assembled matrix.
            solver.step(state, following, None, None, DT)
            held.assign(flags)
            holds.apply()
            values = solver.A_bsr.values.numpy()
            slot = holds._slots.numpy()[top.tolist().index(node)]
            np.testing.assert_array_equal(fixed_particles._projected.values.numpy()[slot], values[slot])
            compact_slot = holds._compact_slots.numpy()[top.tolist().index(node)]
            np.testing.assert_array_equal(fixed_particles.compact_matrix.values.numpy()[compact_slot], values[slot])
            row = holds._compact_rows.numpy()[top.tolist().index(node)]
            np.testing.assert_array_equal(
                fixed_particles.compact_diagonal.numpy()[row], solver._pc_inv_diag.numpy()[node]
            )
            np.testing.assert_allclose(
                np.trace(values[slot]),
                (1.0 + HOLD_FACTOR) * np.trace(holds.pristine.numpy()[top.tolist().index(node)]),
                rtol=1e-5,
            )


class HeldTissueForceTests(unittest.TestCase):
    """The capsule over surface vertices that cannot move: one stored force per axis piece, by hand."""

    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.slab = _Slab()

    def setUp(self):
        self.slab.contact.reset()
        self.slab.step_seconds.assign([DT])

    def _force(self):
        return self.slab.contact.wrench.numpy()[1]

    def test_proportional_and_integral_part_of_seven_touching_pieces(self):
        slab, contact = self.slab, self.slab.contact
        depth = 0.02 * MM
        rho = INTEGRAL_RATIO * STIFFNESS
        slab.place(4 * MM, 6 * MM, RADIUS - depth)
        slab.update()
        # Flat held tissue under a parallel axis: every piece is `depth` deep, pushed straight up, n = 7.
        np.testing.assert_allclose(contact._held_depth.numpy(), depth, rtol=2e-3)
        self.assertEqual(contact._tool_count.numpy().tolist(), [7])
        stored = contact.stored.numpy()
        np.testing.assert_allclose(stored[:, 2], rho * depth / 7, rtol=2e-3)
        np.testing.assert_allclose(stored[:, :2], 0.0, atol=1e-9)
        wrench = self._force()
        np.testing.assert_allclose(wrench[:3], [0, 0, (rho + STIFFNESS) * depth], rtol=2e-3, atol=1e-8)
        # Applied at the probes: their middle is the body origin, the torque is about the centre of mass.
        np.testing.assert_allclose(wrench[3:], np.cross(-COM, wrench[:3]), rtol=2e-3, atol=1e-9)
        self.assertEqual(int(contact.held.numpy().sum()), 0)
        # The same pose again: the stored part has doubled.
        slab.update()
        np.testing.assert_allclose(self._force()[2], (2 * rho + STIFFNESS) * depth, rtol=2e-3)
        # A clearance c of 0.001 mm: no row is limited, so the stored force runs out at the proportional gain,
        # by K_c c / n (its stand-off of 2 depth / 16 = 0.0025 mm closes by the clearance), and the applied
        # force is K_c c / n less again.
        clearance = 0.001 * MM
        slab.place(4 * MM, 6 * MM, RADIUS + clearance)
        slab.update()
        np.testing.assert_allclose(contact._held_depth.numpy(), -clearance, rtol=1e-2)
        left = (2 * rho * depth - STIFFNESS * clearance) / 7
        self.assertGreater(left, 0.0)
        np.testing.assert_allclose(contact.stored.numpy()[:, 2], left, rtol=1e-2)
        self.assertEqual(contact._tool_count.numpy().tolist(), [7])
        wrench = self._force()
        np.testing.assert_allclose(wrench[2], 7 * (left - STIFFNESS * clearance / 7), rtol=2e-2)
        np.testing.assert_allclose(wrench[3:], np.cross(-COM, wrench[:3]), rtol=2e-3, atol=1e-10)
        # The same clearance again: the rest of the stand-off (0.0015 mm) closes and the tool is not pushed.
        slab.update()
        slab.update()
        np.testing.assert_array_equal(contact.stored.numpy(), 0.0)
        np.testing.assert_array_equal(self._force(), 0.0)
        # Far above the tissue the clearance counts as one radius and a stored force is gone at once.
        slab.place(4 * MM, 6 * MM, RADIUS - depth)
        slab.update()
        self.assertGreater(self._force()[2], 0.0)
        slab.place(4 * MM, 6 * MM, 5 * MM)
        slab.update()
        np.testing.assert_array_equal(contact.stored.numpy(), 0.0)
        np.testing.assert_array_equal(self._force(), 0.0)
        slab.update()
        self.assertEqual(contact._tool_count.numpy().tolist(), [0])

    def test_stored_force_is_an_integral_over_time(self):
        slab, contact = self.slab, self.slab.contact
        depth = 0.02 * MM
        slab.step_seconds.assign([0.25 * DT])
        slab.place(4 * MM, 6 * MM, RADIUS - depth)
        slab.update()
        rho = INTEGRAL_RATIO * STIFFNESS
        np.testing.assert_allclose(contact.stored.numpy()[:, 2], 0.25 * rho * depth / 7, rtol=2e-3)
        np.testing.assert_allclose(self._force()[2], (0.25 * rho + STIFFNESS) * depth, rtol=2e-3)
        # It runs out over time as well: a quarter step with a clearance takes a quarter of K_c c / n.
        clearance = 0.001 * MM
        slab.place(4 * MM, 6 * MM, RADIUS + clearance)
        slab.update()
        left = 0.25 * (rho * depth - STIFFNESS * clearance) / 7
        self.assertGreater(left, 0.0)
        np.testing.assert_allclose(contact.stored.numpy()[:, 2], left, rtol=3e-2)

    def test_sweep_places_the_capsule_where_it_will_be(self):
        slab, contact = self.slab, self.slab.contact
        # 0.1 mm above the tissue and moving down by 0.12 mm in one full step: 0.02 mm deep at the predicted pose.
        slab.place(4 * MM, 6 * MM, RADIUS + 0.1 * MM, velocity=(0.0, 0.0, -0.12 * MM / DT))
        slab.update()
        np.testing.assert_allclose(contact._held_depth.numpy(), 0.02 * MM, rtol=5e-3)
        np.testing.assert_array_equal(contact._inward.numpy(), 0)
        self.assertGreater(self._force()[2], 0.0)

    def test_one_end_of_a_tilted_capsule_touches_with_one_piece(self):
        slab, contact = self.slab, self.slab.contact
        # Tilted by 0.2 rad about y: the +x end is lowest; its cap is 0.02 mm deep.
        tilt = 0.2
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt)
        height = RADIUS - 0.02 * MM + 1.5 * MM * np.sin(tilt)
        poses = slab.body_q.numpy()
        poses[1] = (4 * MM, 6 * MM, height, *rotation)
        slab.body_q.assign(poses)
        slab.body_qd.zero_()
        slab.update()
        held = contact._held_depth.numpy()
        np.testing.assert_allclose(held[-1], 0.02 * MM, rtol=5e-3)
        self.assertTrue(np.all(held[:-1] < 0))
        self.assertEqual(contact._tool_count.numpy().tolist(), [1])
        np.testing.assert_allclose(
            self._force()[:3], [0, 0, (INTEGRAL_RATIO + 1) * STIFFNESS * 0.02 * MM], rtol=5e-3, atol=1e-8
        )

    def test_what_pieces_without_a_held_depth_lose_goes_to_the_piece_that_still_stops_the_tool(self):
        slab, contact = self.slab, self.slab.contact
        # Tilted by 0.2 rad about y as above: the cap of the +x end is 0.02 mm deep in held tissue, the six other
        # pieces are clear of it. Every piece stores 10 mN along +z from before.
        tilt = 0.2
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt)
        poses = slab.body_q.numpy()
        poses[1] = (4 * MM, 6 * MM, RADIUS - 0.02 * MM + 1.5 * MM * np.sin(tilt), *rotation)
        slab.body_q.assign(poses)
        slab.body_qd.zero_()
        before = np.zeros((7, 3), dtype=np.float32)
        before[:, 2] = 0.01
        contact.stored.assign(before)
        slab.update()
        held = contact._held_depth.numpy()
        np.testing.assert_allclose(held[-1], 0.02 * MM, rtol=5e-3)
        # The lowest point of piece i is its +x end, (2.75 - 0.5 i) mm from the lowest end of the axis; a
        # clearance counts as one radius at most.
        clear = np.minimum(RADIUS, (2.75 - 0.5 * np.arange(6)) * MM * np.sin(tilt) - 0.02 * MM)
        np.testing.assert_allclose(-held[:-1], clear, rtol=5e-3)
        self.assertEqual(contact._tool_count.numpy().tolist(), [7])
        # Each clear piece loses K_c c / n, at most what it has: 5.4 mN the piece next to the end, all 10 mN the
        # five others. The end piece, the only one with a held depth, takes all of it on top of its own growth.
        lost = np.minimum(0.01, STIFFNESS * clear / 7)
        self.assertAlmostEqual(lost.sum(), 0.05 + STIFFNESS * clear[5] / 7, delta=1e-9)
        np.testing.assert_allclose(contact._tool_released.numpy(), [lost.sum()], rtol=5e-3)
        np.testing.assert_allclose(contact._tool_held.numpy(), [held[-1]], rtol=1e-6)
        stored = contact.stored.numpy().copy()
        np.testing.assert_allclose(stored[:-1, 2], 0.01 - lost, rtol=5e-3, atol=1e-7)
        grown = INTEGRAL_RATIO * STIFFNESS * held[-1] / 7
        np.testing.assert_allclose(stored[-1, 2], 0.01 + grown + lost.sum(), rtol=1e-3)
        np.testing.assert_allclose(stored[:, :2], 0.0, atol=1e-9)
        # Nothing is lost: the tool is still stopped, with what it stored and what the end piece added.
        np.testing.assert_allclose(stored[:, 2].sum(), 0.07 + grown, rtol=1e-4)
        # The clear pieces are pushed by nothing (the proportional part answers their clearance); the end piece
        # by its stored force and its proportional part.
        np.testing.assert_allclose(self._force()[2], stored[-1, 2] + STIFFNESS * held[-1] / 7, rtol=1e-3)
        # Lifted clear of the tissue by 0.001 mm at its lowest point: no piece has a held depth, and what the
        # pieces lose now is lost.
        poses[1, 2] += 0.021 * MM
        slab.body_q.assign(poses)
        slab.update()
        np.testing.assert_array_equal(contact._tool_held.numpy(), [0.0])
        after = contact.stored.numpy()
        np.testing.assert_allclose(after[-1, 2], stored[-1, 2] - STIFFNESS * 0.001 * MM / 2, rtol=1e-3)
        self.assertLess(after[:, 2].sum(), stored[:, 2].sum())

    def test_stiffness_follows_the_effective_mass_of_the_tool_along_the_push_direction(self):
        # In the body frame the tool is four times as heavy along z as along x and y; the body is turned by
        # a quarter about its x axis (the capsule axis), so the push direction, world z, is body y or body z.
        heavy = np.tile(np.diag([1.0 / MASS, 1.0 / MASS, 0.25 / MASS]), (7, 1, 1))
        depth = 0.02 * MM
        for angle, factor in ((0.0, 4.0), (0.5 * np.pi, 1.0)):
            slab = _Slab(inverse_mass=heavy)
            rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)
            poses = slab.body_q.numpy()
            poses[1] = (4 * MM, 6 * MM, RADIUS - depth, *rotation)
            slab.body_q.assign(poses)
            slab.update()
            force = slab.contact.wrench.numpy()[1, :3]
            np.testing.assert_allclose(
                force, [0, 0, factor * (INTEGRAL_RATIO + 1) * STIFFNESS * depth], rtol=3e-3, atol=1e-8
            )
            self.assertEqual(slab.contact.parameters()["held_stiffness_heaviest_direction_N_m"], 4 * STIFFNESS)
        for wrong in (np.zeros((7, 3, 3)), np.eye(3), -1.0, np.full((7, 3, 3), np.nan)):
            with self.assertRaises(ValueError):
                _Slab(inverse_mass=wrong)

    def test_stored_force_on_tissue_that_follows_again_runs_out_without_a_proportional_offset(self):
        # Over free tissue: the nodes under the capsule can move 0.07 mm further than the 0.02 mm they are
        # pressed, so the held depth is negative while the deepest row is not. The stored force does not
        # outlive its cause: it loses K_c * slack / n per step (not rho * slack / n) and is applied as it is,
        # without a proportional part for the slack.
        slab = _Slab()
        contact = slab.contact
        stored = np.zeros((7, 3), dtype=np.float32)
        stored[:, 2] = 0.02
        contact.stored.assign(stored)
        slab.place(4 * MM, 1 * MM, RADIUS - 0.02 * MM)
        slab.update()
        held = contact._held_depth.numpy()
        self.assertTrue(np.all(held < -0.03 * MM))
        self.assertTrue(np.all(contact._row_gap.numpy() > 0.019 * MM))
        self.assertEqual(contact._tool_count.numpy().tolist(), [7])
        left = 0.02 + STIFFNESS * held / 7
        self.assertTrue(np.all(left > 0.005))
        np.testing.assert_allclose(contact.stored.numpy()[:, 2], left, rtol=1e-5)
        np.testing.assert_allclose(contact.wrench.numpy()[1, 2], left.sum(), rtol=1e-5)
        # Two more steps over the same tissue and nothing is left: the tool is not carried over tissue that
        # can yield. (At rho it would have taken 16 times as long.)
        for _ in range(2):
            slab.place(4 * MM, 1 * MM, RADIUS - 0.02 * MM)
            slab.update()
        np.testing.assert_array_equal(contact.stored.numpy(), 0.0)
        self.assertEqual(contact._tool_count.numpy().tolist(), [0])

    def test_fixed_surface_particles_have_no_allowed_move(self):
        # Under the slab: its bottom nodes are particles of the model that the solver holds fixed. The capsule
        # touches them from below, 0.02 mm deep: every piece has that held depth (a free node at rest could
        # move 0.07 mm and would give none), and the tool is pushed down and away.
        slab, contact = self.slab, self.slab.contact
        depth = 0.02 * MM
        slab.place(4 * MM, 1 * MM, -2 * MM - (RADIUS - depth))
        slab.update()
        np.testing.assert_allclose(contact._held_depth.numpy(), depth, rtol=5e-3)
        self.assertEqual(contact._tool_count.numpy().tolist(), [7])
        force = (INTEGRAL_RATIO + 1) * STIFFNESS * depth
        np.testing.assert_allclose(self._force()[:3], [0, 0, -force], rtol=5e-3, atol=1e-8)
        self.assertEqual(int(contact.held.numpy().sum()), 0)

    def test_inverse_mass_can_be_replaced_at_a_reset(self):
        slab = _Slab()
        contact = slab.contact
        depth = 0.02 * MM
        pointer = contact._inverse_mass.ptr
        # Twice the effective mass for every piece: twice K_c, in the same array.
        contact.set_inverse_mass(2.0 * MASS)
        self.assertEqual(contact._inverse_mass.ptr, pointer)
        parameters = contact.parameters()
        np.testing.assert_allclose(parameters["held_stiffness_lightest_direction_N_m"], 2 * STIFFNESS, rtol=1e-12)
        slab.place(4 * MM, 6 * MM, RADIUS - depth)
        slab.update()
        np.testing.assert_allclose(
            contact.wrench.numpy()[1, 2], 2 * (INTEGRAL_RATIO + 1) * STIFFNESS * depth, rtol=2e-3
        )
        # Per piece: the first piece four times as heavy as the others.
        matrices = np.tile(np.eye(3) / MASS, (7, 1, 1))
        matrices[0] *= 0.25
        contact.reset()
        contact.set_inverse_mass(matrices)
        slab.update()
        np.testing.assert_allclose(
            contact.stored.numpy()[:, 2] * 7 / (INTEGRAL_RATIO * STIFFNESS * depth), [4, 1, 1, 1, 1, 1, 1], rtol=2e-3
        )
        np.testing.assert_allclose(
            contact.parameters()["held_stiffness_heaviest_direction_N_m"], 4 * STIFFNESS, rtol=1e-12
        )
        for wrong in (np.zeros((7, 3, 3)), np.eye(3), -1.0, np.full((7, 3, 3), np.nan)):
            with self.assertRaises(ValueError):
                contact.set_inverse_mass(wrong)

    def test_parameters_report_the_constants_without_a_read(self):
        parameters = self.slab.contact.parameters()
        np.testing.assert_allclose(parameters["held_stiffness_lightest_direction_N_m"], STIFFNESS, rtol=1e-12)
        np.testing.assert_allclose(parameters["held_stiffness_heaviest_direction_N_m"], STIFFNESS, rtol=1e-12)
        self.assertEqual((parameters["integral_ratio"], INTEGRAL_RATIO), (0.0625, 0.0625))
        self.assertEqual(
            (parameters["move_fraction"], parameters["min_stretch"], parameters["hold_factor"]), (0.125, 0.2, 4096.0)
        )
        self.assertEqual((parameters["axis_pieces"], parameters["sweep_horizon_s"]), (7, DT))


class YieldingTissueTests(unittest.TestCase):
    """The capsule over free tissue: nodes placed on it and held, their reaction one step late, their release."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _pressed(self, depth=0.02 * MM):
        slab = _Slab()
        slab.place(4 * MM, 1 * MM, RADIUS - depth)
        slab.update()
        return slab

    def test_nodes_under_the_capsule_are_placed_on_it_and_held(self):
        depth = 0.02 * MM
        slab = self._pressed(depth)
        contact = slab.contact
        under = slab.under()
        self.assertEqual(len(under), 5)
        held = np.flatnonzero(contact.held.numpy())
        np.testing.assert_array_equal(held, under)
        moved = slab.state.particle_q.numpy() - slab.rest
        np.testing.assert_allclose(moved[under], np.tile((0.0, 0.0, -depth), (5, 1)), rtol=3e-3, atol=2e-9)
        others = np.setdiff1d(np.arange(len(slab.rest)), under)
        np.testing.assert_array_equal(slab.state.particle_q.numpy()[others], slab.rest[others].astype(np.float32))
        np.testing.assert_allclose(contact.push.numpy()[under], np.tile((0.0, 0.0, -1.0), (5, 1)), atol=1e-5)
        np.testing.assert_array_equal(contact.owner.numpy()[under], 1)
        np.testing.assert_array_equal(contact.owner.numpy()[others], -1)
        # Free tissue that follows has no held depth: nothing is stored and no piece touches.
        self.assertTrue(np.all(contact._held_depth.numpy() < 0))
        np.testing.assert_array_equal(contact.stored.numpy(), 0.0)
        np.testing.assert_array_equal(contact.wrench.numpy(), 0.0)
        # The move is far below the allowed move of a node at rest (an eighth of 0.8 rest altitudes).
        self.assertGreater(contact._allowed.numpy()[under].min(), 2 * depth)
        diagnostics = contact.diagnostics()
        self.assertEqual(
            (diagnostics["held_nodes"], diagnostics["limited_nodes"], diagnostics["inward_probes"]), (5, 0, 0)
        )

    def test_a_held_node_is_placed_with_zero_velocity_and_the_others_keep_theirs(self):
        slab = _Slab()
        # Every node moves when the capsule arrives; the hold keeps a node where it is put only if it rests.
        velocity = np.tile(np.array([0.001, -0.002, 0.003], dtype=np.float32), (len(slab.rest), 1))
        slab.state.particle_qd.assign(velocity)
        slab.place(4 * MM, 1 * MM, RADIUS - 0.02 * MM)
        slab.update()
        under = slab.under()
        np.testing.assert_array_equal(np.flatnonzero(slab.contact.held.numpy()), under)
        after = slab.state.particle_qd.numpy()
        np.testing.assert_array_equal(after[under], 0.0)
        others = np.setdiff1d(np.arange(len(slab.rest)), under)
        np.testing.assert_array_equal(after[others], velocity[others])

    def test_a_node_follows_a_withdrawing_tool_by_its_allowed_move_back_only(self):
        slab = self._pressed()
        contact, solver = slab.contact, slab.solver
        under = slab.under()
        solver.refresh_tangent(slab.state.particle_q, DT)
        solver.step(slab.state, slab.next, None, None, DT)
        # The tool withdraws by 0.3 mm in one step, further than a node may follow: one eighth of the smallest
        # principal stretch of its tetrahedra (about 1 here) times its rest altitude. The smallest rest
        # altitude at a node of this mesh is 1 mm / sqrt(2), so the move back is under 0.0884 mm.
        slab.place(4 * MM, 1 * MM, RADIUS - 0.02 * MM + 0.3 * MM)
        before = slab.next.particle_q.numpy().copy()
        slab.update(slab.next)
        moved = slab.next.particle_q.numpy() - before
        back = contact._allowed_back.numpy()[under]
        self.assertTrue(np.all(back > 0.9 * MOVE_FRACTION * MM / np.sqrt(2.0)))
        self.assertTrue(np.all(back <= MOVE_FRACTION * MM / np.sqrt(2.0) * (1 + 1e-2)))
        np.testing.assert_array_equal(contact.compressive.numpy()[under], 1)
        np.testing.assert_allclose(contact._row_depth.numpy()[under], -0.3 * MM, rtol=2e-2)
        np.testing.assert_allclose(moved[under, 2], back, rtol=1e-3)
        # Along the push direction of its row: straight up but for the slope of the dimple beside the node.
        self.assertLess(np.abs(moved[under, :2]).max(), 0.03 * back.max())
        # They are still held, where they were put.
        np.testing.assert_array_equal(np.flatnonzero(contact.held.numpy()), under)
        self.assertGreater(slab.volumes(slab.next.particle_q.numpy()).min(), 0.0)

    def test_reaction_of_the_held_nodes_reaches_the_tool_one_step_late_along_the_push_direction(self):
        slab = self._pressed()
        contact, solver, model = slab.contact, slab.solver, slab.model
        under = slab.under()
        placed = slab.state.particle_q.numpy().copy()
        external = wp.zeros(model.particle_count, dtype=wp.vec3, device=DEVICE)
        contact.add_balance(external)
        elastic = solver.eval_tetrahedral_forces(model, model.control(), slab.state).numpy().copy()
        # The balancing force is minus the elastic force of the accepted state, on the held nodes only: nothing
        # here but the rounding of a slab at rest, since the nodes were moved after that state.
        self.assertLess(np.abs(external.numpy()).max(), 1e-6)
        np.testing.assert_array_equal(external.numpy()[np.setdiff1d(np.arange(model.particle_count), under)], 0.0)
        solver.refresh_tangent(slab.state.particle_q, DT)
        solver.step(slab.state, slab.next, None, None, DT, external_forces=external)
        after = slab.next.particle_q.numpy()
        # Held as points: they stay where they were put (within a thousandth of their move of 0.02 mm) while
        # their neighbours move.
        self.assertLess(np.abs(after[under] - placed[under]).max(), 2e-8)
        self.assertGreater(np.abs(after - placed).max(), 1e-7)
        elastic = solver.eval_tetrahedral_forces(model, model.control(), slab.next).numpy().copy()
        slab.update(slab.next)
        # Every held node is pushed up by the tissue: its force on the tool is the part along the push direction.
        self.assertTrue(np.all(elastic[under, 2] > 0))
        np.testing.assert_array_equal(contact.compressive.numpy()[under], 1)
        wrench = contact.wrench.numpy()[1]
        force = np.zeros((len(under), 3))
        force[:, 2] = elastic[under, 2]
        np.testing.assert_allclose(wrench[:3], force.sum(axis=0), rtol=1e-4, atol=1e-9)
        centre = np.array([4 * MM, 1 * MM, RADIUS - 0.02 * MM]) + COM
        np.testing.assert_allclose(
            wrench[3:], np.cross(after[under] - centre, force).sum(axis=0), rtol=1e-3, atol=1e-10
        )
        np.testing.assert_array_equal(contact.wrench.numpy()[0], 0.0)
        # They stay held, and the next tissue solve gets the force that balances their elastic force.
        np.testing.assert_array_equal(np.flatnonzero(contact.held.numpy()), under)
        external.zero_()
        contact.add_balance(external)
        expected = np.zeros_like(elastic)
        expected[under] = -elastic[under]
        np.testing.assert_allclose(external.numpy(), expected, rtol=1e-5, atol=1e-12)
        body_f = wp.zeros(2, dtype=wp.spatial_vector, device=DEVICE)
        contact.add_forces(body_f)
        contact.add_forces(body_f)
        np.testing.assert_allclose(body_f.numpy(), 2 * contact.wrench.numpy(), rtol=1e-6)

    def test_held_nodes_follow_the_tool_back_and_are_released_when_it_has_gone(self):
        slab = self._pressed()
        contact, solver = slab.contact, slab.solver
        under = slab.under()
        solver.refresh_tangent(slab.state.particle_q, DT)
        solver.step(slab.state, slab.next, None, None, DT)
        # The tool withdraws by 0.005 mm: the nodes that push back follow it by their clearance.
        slab.place(4 * MM, 1 * MM, RADIUS - 0.015 * MM)
        before = slab.next.particle_q.numpy().copy()
        slab.update(slab.next)
        moved = slab.next.particle_q.numpy() - before
        np.testing.assert_allclose(moved[under, 2], 0.005 * MM, rtol=2e-2)
        np.testing.assert_array_equal(np.flatnonzero(contact.held.numpy()), under)
        values = solver.A_bsr.values.numpy().copy()
        # The tool is gone: no row, every node is released and the matrix has its own blocks again.
        slab.place(4 * MM, 1 * MM, 5 * MM)
        slab.update(slab.next)
        self.assertEqual(int(contact.held.numpy().sum()), 0)
        np.testing.assert_array_equal(contact.push.numpy(), 0.0)
        np.testing.assert_array_equal(contact.balance.numpy(), 0.0)
        slots = contact.holds._slots.numpy()
        np.testing.assert_array_equal(solver.A_bsr.values.numpy()[slots], contact.holds.pristine.numpy())
        self.assertFalse(np.array_equal(values[slots], contact.holds.pristine.numpy()))

    def test_a_node_that_is_pulled_is_released(self):
        slab = self._pressed()
        contact = slab.contact
        under = slab.under()
        # The tissue at rest pulls a node that was moved outwards: lift the held nodes above the surface.
        positions = slab.state.particle_q.numpy()
        positions[under, 2] = 0.03 * MM
        slab.state.particle_q.assign(positions)
        slab.place(4 * MM, 1 * MM, RADIUS + 0.03 * MM + 0.001 * MM)
        slab.update()
        np.testing.assert_array_equal(contact.compressive.numpy()[under], 0)
        self.assertEqual(int(contact.held.numpy().sum()), 0)
        np.testing.assert_array_equal(contact.wrench.numpy(), 0.0)

    def test_a_capsule_that_went_through_the_surface_pushes_it_ahead_by_the_allowed_move_only(self):
        slab = _Slab()
        contact = slab.contact
        # From 1.2 radii above to 0.8 radii below the surface in one step: two radii of travel.
        slab.place(4 * MM, 1 * MM, 1.2 * RADIUS, velocity=(0.0, 0.0, -2.0 * RADIUS / DT))
        slab.update()
        np.testing.assert_array_equal(contact._inward.numpy(), 1)
        held = np.flatnonzero(contact.held.numpy())
        self.assertGreaterEqual(len(held), 5)
        moved = slab.state.particle_q.numpy() - slab.rest
        allowed = contact._allowed.numpy()
        depth = contact._row_depth.numpy()
        # Under the axis the surface must go 0.8 + 1 radii down; each node goes down, by its allowed move at most,
        # and not further than the capsule pulled back by the held depth of its piece reaches.
        np.testing.assert_allclose(depth[slab.under()].max(), 1.8 * RADIUS, rtol=2e-3)
        np.testing.assert_allclose(contact.push.numpy()[held], np.tile((0.0, 0.0, -1.0), (len(held), 1)), atol=1e-5)
        stuck = np.maximum(contact._held_depth.numpy()[contact._row_owner.numpy()[held]], 0.0)
        expected = np.clip(depth[held] - stuck, 0.0, allowed[held])
        np.testing.assert_allclose(-moved[held, 2], expected, rtol=1e-3, atol=2e-9)
        under = slab.under()
        np.testing.assert_allclose(-moved[under, 2], allowed[under], rtol=1e-3)
        self.assertLess(-moved[held, 2].min(), 0.2 * MM)
        self.assertGreater(slab.volumes(slab.state.particle_q.numpy()).min(), 0.0)
        # What the tissue cannot follow in this step is held depth: the tool is pushed back.
        self.assertGreater(contact._held_depth.numpy().max(), RADIUS)
        self.assertGreater(contact.wrench.numpy()[1, 2], 0.0)
        self.assertGreater(contact.diagnostics()["limited_nodes"], 0)

    def test_update_is_deterministic_and_keeps_its_storage(self):
        slab = self._pressed(0.03 * MM)
        contact = slab.contact
        slab.solver.refresh_tangent(slab.state.particle_q, DT)
        slab.solver.step(slab.state, slab.next, None, None, DT)
        # The same accepted state and the same contact state, twice. (Two tissue steps would not do: on a CUDA
        # device their solves end within the solver's tolerance of each other, not within rounding.)
        inputs = (*contact.state, slab.next.particle_q, slab.next.particle_qd)
        saved = [array.numpy().copy() for array in inputs]
        names = ("stored", "wrench", "held", "push", "owner", "compressive", "balance")
        arrays = (*contact.state, contact._row_depth, contact._row_push, contact._row_owner, contact._held_depth)
        pointers = [array.ptr for array in arrays]
        results = []
        for _ in range(2):
            for array, values in zip(inputs, saved, strict=True):
                array.assign(values)
            slab.update(slab.next)
            self.assertEqual([array.ptr for array in arrays], pointers)
            results.append([array.numpy().copy() for array in arrays] + [slab.next.particle_q.numpy().copy()])
        # The update did something: nodes are held, they push back, and their reaction is on the tool.
        self.assertEqual(int(results[0][2].sum()), 5)
        self.assertGreater(results[0][1][1, 2], 0.0)
        self.assertTrue(np.all(np.isneginf(results[0][7][slab.fixed])))
        for index, (first, second) in enumerate(zip(*results, strict=True)):
            # Only the reaction on the bodies and the balancing force are sums by atomics (the forces of the
            # tetrahedra on a node, the forces of the nodes on a body); every other array has one writer per entry.
            _assert_same(second, first, summed=index < len(names) and names[index] in ("wrench", "balance"))

    def test_reset_releases_every_node_and_clears_every_force(self):
        slab = self._pressed()
        contact = slab.contact
        self.assertGreater(int(contact.held.numpy().sum()), 0)
        contact.reset()
        for array in (contact.stored, contact.wrench, contact.held, contact.push, contact.compressive, contact.balance):
            np.testing.assert_array_equal(array.numpy(), 0)
        np.testing.assert_array_equal(contact.owner.numpy(), -1)
        slots = contact.holds._slots.numpy()
        np.testing.assert_array_equal(slab.solver.A_bsr.values.numpy()[slots], contact.holds.pristine.numpy())


if __name__ == "__main__":
    unittest.main()
