# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ThreadTissueCoupling on a small tetrahedral block: contact planes, reactions, eyelet tracks, friction, state.

No tissue solver runs. The coupling only reads particle positions and adds
particle forces, so the tests place the block's particles themselves.
"""

import math
import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_suturebot import SolverThread, ThreadModel, ThreadTissueCoupling

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")

# The suture of fem_needle_suture: 0.18 m of thread in 90 elements, one step per frame.
THREAD_RADIUS = 1.5e-4  # [m]
THREAD_DENSITY = 1140.0  # [kg/m³]
THREAD_MODULUS = 2.0e6  # [Pa]
ELEMENT = 0.002  # [m]
GRAVITY = 9.81  # [m/s²]
DT = 1.0 / 60.0  # [s]
FRICTION = 0.2
SMOOTHING = 0.1 * THREAD_RADIUS  # [m]
NODE_WEIGHT = THREAD_DENSITY * math.pi * THREAD_RADIUS**2 * ELEMENT * GRAVITY  # [N], weight of one element
AXIAL_STIFFNESS = THREAD_MODULUS * math.pi * THREAD_RADIUS**2  # [N]

# The block: cubes of 4 mm of the examples' 25 kPa tissue. The examples take k = E h for the contact and the
# eyelet spring and K = E h / 4 for the stiffness of a tissue node.
EDGE = 0.004  # [m]
SPRING = 25000.0 * EDGE  # [N/m]
TISSUE_STIFFNESS = 0.25 * SPRING  # [N/m]
MARGIN = 0.003  # [m], three quarters of a cube
ORIGIN = np.array([0.0, -0.008, -0.008])  # [m]; the block's top is z = 0

# Largest accepted residual of a node [N]: a thousandth of its weight, a hundred times tighter than the examples.
SOLVER_TOLERANCE = 1.0e-3 * NODE_WEIGHT
# Most Newton updates per step; the method notes give the friction of a dragged thread to 0.4 % with eight.
UPDATES = 8
MASS_DAMPING = 2.0  # [1/s], as in the examples
# The surface is stored in float32. One ulp at 0.05 m is 4e-9 m and a point interpolated on a triangle carries
# a few of them.
SURFACE_ROUNDING = 1.0e-7  # [m]

# The six tetrahedra around the diagonal from corner 0 to corner 7 of a cube, whose corner c sits at
# (c & 1, c >> 1 & 1, c >> 2 & 1). Every cube is split alike, so neighbouring cubes share their face diagonals.
_CUBE_TETRAHEDRA = ((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7))


def setUpModule():
    if DEVICE.startswith("cuda") and not wp.is_cuda_available():
        raise unittest.SkipTest("NEWTON_TEST_DEVICE asks for CUDA and there is none")


def _block(cells):
    """Vertices [m] and tetrahedra of a box of ``cells`` cubes with edge EDGE, six tetrahedra per cube."""
    counts = np.asarray(cells) + 1
    grid = np.stack(np.meshgrid(*(np.arange(count) for count in counts), indexing="ij"), axis=-1).reshape(-1, 3)
    tetrahedra = []
    for cube in np.ndindex(*cells):
        corners = [np.ravel_multi_index(tuple(np.add(cube, (c & 1, c >> 1 & 1, c >> 2 & 1))), counts) for c in range(8)]
        tetrahedra.extend([corners[c] for c in tetrahedron] for tetrahedron in _CUBE_TETRAHEDRA)
    return ORIGIN + EDGE * grid, np.array(tetrahedra, dtype=np.int32)


def _outward_triangles(vertices, tetrahedra):
    """The faces that belong to one tetrahedron only, wound so that their normals point out of it."""
    owners = {}
    for tetrahedron in tetrahedra:
        for skipped in range(4):
            face = [int(vertex) for corner, vertex in enumerate(tetrahedron) if corner != skipped]
            owners.setdefault(tuple(sorted(face)), []).append((face, int(tetrahedron[skipped])))
    triangles = []
    for found in owners.values():
        if len(found) == 1:
            (a, b, c), opposite = found[0]
            normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
            triangles.append((a, c, b) if normal @ (vertices[opposite] - vertices[a]) > 0.0 else (a, b, c))
    return np.array(triangles, dtype=np.int32)


def _material_point(vertices, tetrahedra, point):
    """Tetrahedron that holds ``point`` and the point's barycentric weights in it."""
    corners = vertices[tetrahedra]
    inverse = np.linalg.inv((corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1))
    inner = np.einsum("nij,nj->ni", inverse, point - corners[:, 0])
    weights = np.column_stack((1.0 - inner.sum(axis=1), inner))
    best = int(np.argmax(weights.min(axis=1)))
    if weights[best].min() < 0.0:
        raise ValueError("the point lies outside the block")
    return best, weights[best]


def _rest_height(load):
    """Height of a node centre above its plane at which the contact penalty carries ``load`` [N].

    With the depth d = r - height and p = (d + sqrt(d² + s²)) / 2 the penalty pushes with k p dp/dd.
    """

    def push(height):
        depth = THREAD_RADIUS - height
        root = math.hypot(depth, SMOOTHING)
        return SPRING * 0.25 * (depth + root) * (1.0 + depth / root)

    # The push falls with the height: at 0 it is k r, ten thousand node weights, at 100 r it has died away.
    low, high = 0.0, 100.0 * THREAD_RADIUS
    for _ in range(100):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if push(middle) > load else (low, middle)
    return 0.5 * (low + high)


def _line(start, count, direction=(1.0, 0.0, 0.0)):
    """Coordinates of ``count`` nodes one element apart on a straight line: positions and unit slopes."""
    direction = np.asarray(direction, dtype=np.float64)
    positions = np.asarray(start, dtype=np.float64) + ELEMENT * np.arange(count)[:, None] * direction
    return np.column_stack((positions, np.tile(direction, (count, 1))))


def _at(positions):
    """Coordinates of nodes put at arbitrary ``positions`` for a look at their contact planes, without a step."""
    positions = np.asarray(positions, dtype=np.float64)
    return np.column_stack((positions, np.tile((1.0, 0.0, 0.0), (len(positions), 1))))


class _Scene:
    """One thread, its coupling to a block of tissue and the block's particle arrays.

    Args:
        coordinates: Nodal coordinates the thread starts with, shape (count, 6).
        cells: Cubes of the block along x, y and z.
        tissue_from: First layer of cubes along x that is tissue; the layers before it are surface only, and
            the tissue's particles are then not the surface's first vertices. ``None`` makes the whole block
            tissue.
    """

    def __init__(
        self,
        coordinates,
        cells=(12, 4, 2),
        *,
        tissue_from=None,
        friction=FRICTION,
        gravity=GRAVITY,
        eyelets=0,
        eyelet_spring=SPRING,
        tissue_stiffness=TISSUE_STIFFNESS,
        max_tracks=2,
    ):
        coordinates = np.asarray(coordinates, dtype=np.float64)
        count = len(coordinates)
        self.vertices, tetrahedra = _block(cells)
        self.triangles = _outward_triangles(self.vertices, tetrahedra)
        if tissue_from is None:
            self.particles, self.tetrahedra, particle_points = np.arange(len(self.vertices)), tetrahedra, None
        else:
            self.particles = np.flatnonzero(self.vertices[:, 0] >= (tissue_from - 0.5) * EDGE)
            local = np.full(len(self.vertices), -1)
            local[self.particles] = np.arange(len(self.particles))
            self.tetrahedra = local[tetrahedra[np.all(local[tetrahedra] >= 0, axis=1)]]
            particle_points = self.particles
        self.rest = self.vertices[self.particles]
        straight = ELEMENT * np.arange(count)[:, None] * np.array([1.0, 0.0, 0.0])
        chain = ThreadModel.create(
            straight, young_modulus=THREAD_MODULUS, radius=THREAD_RADIUS, density=THREAD_DENSITY, device=DEVICE
        )
        self.thread = SolverThread(
            newton.ModelBuilder().finalize(device=DEVICE),
            chain,
            gravity=(0.0, 0.0, -gravity),
            mass_damping=MASS_DAMPING,
            tolerance=SOLVER_TOLERANCE,
            max_iterations=UPDATES,
            contact_distance=THREAD_RADIUS,
            contact_stiffness=SPRING,
            contact_smoothing=SMOOTHING,
            friction=friction,
            max_eyelets=eyelets,
            eyelet_stiffness=eyelet_spring if eyelets else 0.0,
        )
        self.thread.q.assign(coordinates.reshape(-1))
        self.thread.boundary_q.assign(self.thread.q)
        self._boundary = coordinates.reshape(-1).copy()
        self.coupling = ThreadTissueCoupling(
            self.thread,
            self.vertices,
            self.triangles,
            self.tetrahedra,
            particle_points,
            tissue_stiffness=tissue_stiffness,
            contact_margin=MARGIN,
            max_tracks=max_tracks,
        )
        self.particle_q = wp.array(self.rest, dtype=wp.vec3, device=DEVICE)
        self.particle_f = wp.zeros(len(self.rest), dtype=wp.vec3, device=DEVICE)

    def place(self, positions):
        """Put the tissue particles at ``positions`` [m]."""
        self.particle_q.assign(np.asarray(positions, dtype=np.float32))

    def hold_start(self, position):
        """Prescribe the position of node 0 [m]."""
        self._boundary[:3] = position
        self.thread.boundary_q.assign(self._boundary)

    def advance(self):
        """One coupled step, as the examples make it: planes and eyelets, the thread, then its forces."""
        self.coupling.update(self.particle_q)
        self.thread.step(None, None, None, None, DT)
        self.particle_f.zero_()
        self.coupling.add_forces(self.particle_f)

    def step(self, count=1):
        for _ in range(count):
            self.advance()

    def nodes(self):
        return self.thread.q.numpy().reshape(-1, 6)

    def forces(self):
        return self.particle_f.numpy().astype(np.float64)

    def tracked(self):
        """Chain coordinate of every eyelet, in elements."""
        return self.thread.eyelet_element.numpy() + self.thread.eyelet_xi.numpy()

    def tension(self, element):
        """Axial force of a straight element from the stretch of its chord [N]: E A ε λ with ε = (λ² - 1) / 2."""
        nodes = self.nodes()
        stretch = np.linalg.norm(nodes[element + 1, :3] - nodes[element, :3]) / ELEMENT
        return AXIAL_STIFFNESS * 0.5 * (stretch**2 - 1.0) * stretch


# Elements of the thread that rests on the block.
RESTING = 20


def _resting(x=0.0031, y=0.0013, **options):
    """A thread laid along x from (x, y) on the top of the long block, level at the height where it is carried."""
    return _Scene(_line((x, y, _rest_height(NODE_WEIGHT)), RESTING + 1), **options)


class BlockFixtureTests(unittest.TestCase):
    def test_block_is_filled_closed_and_faces_outwards(self):
        """Fails if the fixture's tetrahedra are inverted or its surface is open or wound inwards."""
        vertices, tetrahedra = _block((12, 4, 2))
        corners = vertices[tetrahedra]
        volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
        size = EDGE * np.array([12, 4, 2])
        self.assertGreater(volumes.min(), 0.0)
        self.assertAlmostEqual(volumes.sum(), size.prod(), delta=1.0e-12 * size.prod())
        triangles = _outward_triangles(vertices, tetrahedra)
        faces = vertices[triangles]
        normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        middle = ORIGIN + 0.5 * size
        self.assertTrue(np.all(np.einsum("ij,ij->i", normals, faces.mean(axis=1) - middle) > 0.0))
        area = 2.0 * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0])
        self.assertAlmostEqual(0.5 * np.linalg.norm(normals, axis=1).sum(), area, delta=1.0e-12 * area)
        # Closed and consistently wound: every edge runs once in each direction.
        edges = np.concatenate([triangles[:, [a, b]] for a, b in ((0, 1), (1, 2), (2, 0))])
        directed = {tuple(edge) for edge in edges.tolist()}
        self.assertEqual(len(directed), len(edges))
        self.assertTrue(all((b, a) in directed for a, b in directed))


# Nodes around the long block, which spans 0..48 mm, -8..8 mm and -8..0 mm: the position [m], the nearest
# surface point [m] and the outward normal there, or None where the surface is farther than the margin.
_AROUND = (
    ((0.0101, 0.0013, 0.5 * MARGIN), (0.0101, 0.0013, 0.0), (0.0, 0.0, 1.0)),
    ((0.0121, 0.0013, 0.5 * MARGIN), (0.0121, 0.0013, 0.0), (0.0, 0.0, 1.0)),
    ((0.0141, 0.0013, 0.9 * MARGIN), (0.0141, 0.0013, 0.0), (0.0, 0.0, 1.0)),
    ((0.0161, 0.0013, 1.1 * MARGIN), None, None),
    ((0.0181, 0.0013, 2.0 * MARGIN), None, None),
    ((0.0201, -0.0031, -0.008 - 0.4 * MARGIN), (0.0201, -0.0031, -0.008), (0.0, 0.0, -1.0)),
    ((0.048 + 0.6 * MARGIN, 0.0021, -0.0037), (0.048, 0.0021, -0.0037), (1.0, 0.0, 0.0)),
    ((-0.3 * MARGIN, 0.0021, -0.0037), (0.0, 0.0021, -0.0037), (-1.0, 0.0, 0.0)),
    ((0.0222, 0.008 + 0.7 * MARGIN, -0.0047), (0.0222, 0.008, -0.0047), (0.0, 1.0, 0.0)),
    ((0.0222, -0.008 - 0.2 * MARGIN, -0.0047), (0.0222, -0.008, -0.0047), (0.0, -1.0, 0.0)),
    ((0.0222, -0.008 - 1.2 * MARGIN, -0.0047), None, None),
    # Inside the block, 1.1 mm under its top: the plane that pushes the node back out.
    ((0.0222, 0.0003, -0.0011), (0.0222, 0.0003, 0.0), (0.0, 0.0, 1.0)),
)


class ContactPlaneTests(unittest.TestCase):
    def _planes(self, scene):
        scene.coupling.update(scene.particle_q)
        return scene.thread.contact_point.numpy(), scene.thread.contact_normal.numpy()

    def test_nodes_within_the_margin_take_the_nearest_surface_point_and_its_outward_normal(self):
        """Fails if a plane is not at the nearest surface point or its normal does not point out of the block."""
        scene = _Scene(_at([position for position, _, _ in _AROUND]))
        points, normals = self._planes(scene)
        for node, (_, point, normal) in enumerate(_AROUND):
            if point is not None:
                with self.subTest(node=node):
                    np.testing.assert_allclose(points[node], point, rtol=0.0, atol=SURFACE_ROUNDING)
                    # A float32 unit normal.
                    np.testing.assert_allclose(normals[node], normal, rtol=0.0, atol=1.0e-6)

    def test_nodes_beyond_the_margin_take_no_plane(self):
        """Fails if a node farther than the margin keeps a plane, such as the solver's floor that it starts with."""
        scene = _Scene(_at([position for position, _, _ in _AROUND]))
        np.testing.assert_array_equal(scene.thread.contact_normal.numpy(), np.tile((0.0, 0.0, 1.0), (len(_AROUND), 1)))
        _, normals = self._planes(scene)
        beyond = np.array([point is None for _, point, _ in _AROUND])
        np.testing.assert_array_equal(normals[beyond], 0.0)
        np.testing.assert_allclose(np.linalg.norm(normals[~beyond], axis=1), 1.0, rtol=0.0, atol=1.0e-6)

    def test_planes_follow_the_moved_particles(self):
        """Fails if the planes stay with the surface as it was built: particles not copied or the tree not refitted."""
        scene = _Scene(_at([position for position, _, _ in _AROUND]))
        above = [0, 1, 2, 3, 4]  # over the top at 0.5, 0.5, 0.9, 1.1 and 2 margins
        below = 5  # 0.4 margins under the bottom
        self._planes(scene)

        # Raised by 2 mm the top comes within the margin of node 3 and the bottom leaves node 5 behind.
        scene.place(scene.rest + np.array([0.0, 0.0, 0.002]))
        points, normals = self._planes(scene)
        np.testing.assert_allclose(points[above[:4], 2], 0.002, rtol=0.0, atol=SURFACE_ROUNDING)
        np.testing.assert_allclose(normals[above[:4]], np.tile((0.0, 0.0, 1.0), (4, 1)), rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(normals[[above[4], below]], 0.0)

        # Lowered by 10 mm the top is 0.8 mm under node 5 and out of reach of every other node.
        scene.place(scene.rest - np.array([0.0, 0.0, 0.010]))
        points, normals = self._planes(scene)
        np.testing.assert_allclose(points[below], (0.0201, -0.0031, -0.010), rtol=0.0, atol=SURFACE_ROUNDING)
        np.testing.assert_allclose(normals[below], (0.0, 0.0, 1.0), rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(np.delete(normals, below, axis=0), 0.0)

        # Sheared so that the top rises by a quarter of the distance along x from x = 12 mm, which leaves it
        # within 2.2 mm of nodes 0 to 2: the plane of such a node is the foot of its perpendicular on the tilted
        # top, with the top's normal.
        slope, pivot = 0.25, 0.012
        tilted = scene.rest.copy()
        tilted[:, 2] += slope * (tilted[:, 0] - pivot)
        scene.place(tilted)
        points, normals = self._planes(scene)
        normal = np.array([-slope, 0.0, 1.0]) / math.hypot(slope, 1.0)
        for node in above[:3]:
            position = np.array(_AROUND[node][0])
            height = (position - np.array([pivot, 0.0, 0.0])) @ normal
            np.testing.assert_allclose(points[node], position - height * normal, rtol=0.0, atol=SURFACE_ROUNDING)
            np.testing.assert_allclose(normals[node], normal, rtol=0.0, atol=1.0e-6)

        # Back where it was built, every node has its first plane again.
        scene.place(scene.rest)
        points, normals = self._planes(scene)
        for node, (_, point, normal) in enumerate(_AROUND):
            np.testing.assert_allclose(normals[node], (0.0, 0.0, 0.0) if normal is None else normal, atol=1.0e-6)
            if point is not None:
                np.testing.assert_allclose(points[node], point, rtol=0.0, atol=SURFACE_ROUNDING)

    def test_surface_beyond_the_tissue_stays_where_it_was_given(self):
        """Fails if surface vertices that are no tissue particles move, or the tissue's ones do not."""
        tissue_from, lift = 6, 0.001
        scene = _Scene(_line((0.0031, 0.0013, 0.0005), RESTING + 1), tissue_from=tissue_from)
        self.assertFalse(np.array_equal(scene.particles, np.arange(len(scene.particles))))
        scene.place(scene.rest + np.array([0.0, 0.0, lift]))
        scene.coupling.update(scene.particle_q)
        points, normals = scene.thread.contact_point.numpy(), scene.thread.contact_normal.numpy()
        # The tissue's top is raised by the lift, the rest of the top is not, and the layer of cubes between
        # the last other vertices and the first tissue vertices forms a ramp.
        edge = tissue_from * EDGE
        height = lift * np.clip((points[:, 0] - (edge - EDGE)) / EDGE, 0.0, 1.0)
        np.testing.assert_allclose(points[:, 2], height, rtol=0.0, atol=SURFACE_ROUNDING)
        on_tissue, on_rest = points[:, 0] > edge, points[:, 0] < edge - EDGE
        on_ramp = ~on_tissue & ~on_rest
        self.assertGreaterEqual(on_tissue.sum(), 5)
        self.assertGreaterEqual(on_ramp.sum(), 1)
        self.assertGreaterEqual(on_rest.sum(), 5)
        level = np.array([0.0, 0.0, 1.0])
        ramp = np.array([-lift, 0.0, EDGE]) / math.hypot(lift, EDGE)
        for chosen, normal in ((on_tissue, level), (on_ramp, ramp), (on_rest, level)):
            np.testing.assert_allclose(normals[chosen], np.tile(normal, (chosen.sum(), 1)), rtol=0.0, atol=1.0e-6)


class ReactionTests(unittest.TestCase):
    def test_resting_thread_loads_the_particles_with_its_weight(self):
        """Fails if the particle forces are not the opposite of the contact forces, spread by the triangle weights."""
        scene = _resting()
        steps = 60
        scene.step(steps)
        nodes, forces = scene.nodes(), scene.forces()
        contact = scene.thread.contact_force.numpy()
        points, normals = scene.thread.contact_point.numpy(), scene.thread.contact_normal.numpy()

        # The thread lies on the top: every node's plane is the top under it, facing up.
        np.testing.assert_allclose(normals, np.tile((0.0, 0.0, 1.0), (len(nodes), 1)), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(points[:, :2], nodes[:, :2], rtol=0.0, atol=SURFACE_ROUNDING)
        np.testing.assert_allclose(points[:, 2], 0.0, rtol=0.0, atol=SURFACE_ROUNDING)

        # Node 0 has no contact: its support carries its half element. The thread is level there, so the
        # support carries nothing else, and the block carries the other 19.5 elements. Every node is in balance
        # to the solver's tolerance, and the relaxation has (1 - K / (K + k))^steps of the load still to go.
        carried = (RESTING - 0.5) * NODE_WEIGHT
        left = (1.0 - TISSUE_STIFFNESS / (TISSUE_STIFFNESS + SPRING)) ** steps
        bound = len(nodes) * SOLVER_TOLERANCE + left * carried  # a thousandth of the load
        self.assertAlmostEqual(forces[:, 2].sum(), -carried, delta=bound)  # measured: equal to 1.4e-6 of the load
        # Friction can hold the settled thread with no more than the Coulomb limit.
        self.assertLessEqual(np.linalg.norm(forces[:, :2].sum(axis=0)), FRICTION * carried)

        # Action and reaction. A thread may still creep within the solver's tolerance while the applied force
        # trails it by a few steps, hence the same bound; measured: equal to 5e-11 N.
        np.testing.assert_allclose(forces.sum(axis=0), -contact.sum(axis=0), rtol=0.0, atol=bound)
        # The weights of a triangle put each reaction at its node's surface point: the first moments agree,
        # with the block's length of 48 mm as the bound's lever. Weights in another order move a reaction by a
        # millimetre or more.
        np.testing.assert_allclose(
            (scene.rest[:, :2] * forces[:, 2:]).sum(axis=0),
            -(points[:, :2] * contact[:, 2:]).sum(axis=0),
            rtol=0.0,
            atol=bound * 12.0 * EDGE,
        )
        # Only the top's particles are loaded.
        np.testing.assert_array_equal(forces[scene.rest[:, 2] < -0.5 * EDGE], 0.0)

    def test_applied_force_approaches_the_contact_force_by_the_stiffness_ratio_per_step(self):
        """Fails if a step moves the applied force by another fraction than K / (K + k), k the stiffer spring."""
        cases = (
            # tissue stiffness K, eyelet spring, expected fraction
            (TISSUE_STIFFNESS, None, 0.2),  # K = k / 4, as in the examples
            (SPRING, None, 0.5),
            (SPRING, 3.0 * SPRING, 0.25),  # the eyelet spring is the stiffer one
        )
        for tissue_stiffness, eyelet_spring, fraction in cases:
            with self.subTest(tissue_stiffness=tissue_stiffness, eyelet_spring=eyelet_spring):
                options = {} if eyelet_spring is None else {"eyelets": 1, "eyelet_spring": eyelet_spring}
                scene = _resting(tissue_stiffness=tissue_stiffness, **options)
                scene.step(20)  # laid at its rest height, the thread is at rest: the contact forces stay
                scene.coupling.applied_force.zero_()
                for step in range(1, 6):
                    scene.advance()
                    load = scene.thread.contact_force.numpy()[:, 2].sum()
                    self.assertAlmostEqual(
                        scene.forces()[:, 2].sum(),
                        -(1.0 - (1.0 - fraction) ** step) * load,
                        delta=(RESTING + 1) * SOLVER_TOLERANCE,
                    )

    def test_forces_are_added_to_what_the_array_holds(self):
        """Fails if add_forces overwrites the particle forces instead of adding the applied force to them."""
        scene = _resting()
        scene.step(10)
        # A load of the size of the thread's forces, so that float32 keeps both.
        present = np.tile((1.0e-6, -2.0e-6, 3.0e-6), (len(scene.rest), 1))
        scene.particle_f.assign(present.astype(np.float32))
        scene.coupling.add_forces(scene.particle_f)
        applied = scene.coupling.applied_force.numpy().astype(np.float64)
        self.assertGreater(np.abs(applied[:, 2]).max(), 0.01 * NODE_WEIGHT)
        # float32 rounding of sums of a few micronewtons
        np.testing.assert_allclose(scene.forces(), present + applied, rtol=0.0, atol=1.0e-11)

    def test_only_the_tissue_particles_of_a_triangle_take_the_reaction(self):
        """Fails if a surface vertex that is no tissue particle passes its share of a reaction on to the tissue."""
        tissue_from, steps = 6, 60
        scene = _resting(tissue_from=tissue_from)
        scene.step(steps)
        nodes, forces = scene.nodes(), scene.forces()
        contact = scene.thread.contact_force.numpy()
        # A triangle's weights interpolate x. Over the tissue all three vertices are particles; in the layer of
        # cubes between the last other vertices and the first tissue vertices, the particles' share rises
        # from zero to one.
        edge = tissue_from * EDGE
        share = np.clip((nodes[:, 0] - (edge - EDGE)) / EDGE, 0.0, 1.0)
        self.assertTrue(np.any((share > 0.0) & (share < 1.0)))
        self.assertTrue(np.any(share == 0.0))
        self.assertAlmostEqual(
            forces[:, 2].sum(), -(share * contact[:, 2]).sum(), delta=len(nodes) * SOLVER_TOLERANCE
        )  # the bound is two thousandths of the load; measured: equal to 1.5e-6 of it
        # The tissue is under half of the thread and carries about half of it.
        self.assertLess(forces[:, 2].sum(), -0.4 * RESTING * NODE_WEIGHT)
        self.assertGreater(forces[:, 2].sum(), -0.6 * RESTING * NODE_WEIGHT)


# The thread through the short block: 20 elements along x at 2.1 mm under the top. The block spans 0..16 mm
# and the nodes straddle its two faces by half an element, so that no node is at a mouth of the track.
THROUGH = 20
THROUGH_START = np.array([-0.013, 0.0013, -0.0021])
# Track points [m along x]: the first and the last 0.2 mm inside the faces, as a stitch's track ends just
# inside the surface.
TRACK = np.array([0.0002, 0.0055, 0.0105, 0.0158])


class _Through(_Scene):
    """A thread through the short block and one track of four eyelets, all threaded where the thread is."""

    def __init__(self, **options):
        super().__init__(_line(THROUGH_START, THROUGH + 1), cells=(4, 4, 2), gravity=0.0, eyelets=4, **options)
        self.points = np.column_stack((TRACK, np.full(len(TRACK), THROUGH_START[1]), np.full(len(TRACK), -0.0021)))
        found = [_material_point(self.vertices, self.tetrahedra, point) for point in self.points]
        self.track_tetrahedra = np.array([tetrahedron for tetrahedron, _ in found])
        self.track_weights = np.array([weights for _, weights in found])
        self.slots = self.coupling.add_track(self.track_tetrahedra, self.track_weights)
        self.stations = (TRACK - THROUGH_START[0]) / ELEMENT
        for slot, station in zip(self.slots, self.stations, strict=True):
            self.coupling.thread_eyelet(slot, int(station), station - int(station))

    def draw(self, distance, steps):
        """Move node 0 by ``distance`` away from the block, along the thread, in equal parts over ``steps``."""
        start = self._boundary[:3].copy()
        for step in range(1, steps + 1):
            self.hold_start(start - np.array([distance * step / steps, 0.0, 0.0]))
            self.advance()


class EyeletTrackTests(unittest.TestCase):
    def test_tracks_take_their_slots_in_order_and_refuse_what_does_not_fit(self):
        """Fails if add_track hands out other slots than the next free ones or accepts more than there is room for."""
        scene = _Scene(_line((0.0031, 0.0013, 0.001), RESTING + 1), eyelets=5, max_tracks=2)
        quarter = np.full(4, 0.25)
        self.assertEqual(scene.coupling.track_count, 0)
        with self.assertRaises(ValueError):
            scene.coupling.add_track([0, 1], [quarter])  # one tetrahedron without weights
        with self.assertRaises(ValueError):
            scene.coupling.add_track([], np.zeros((0, 4)))
        np.testing.assert_array_equal(scene.coupling.add_track([0, 1, 2], [quarter] * 3), [0, 1, 2])
        with self.assertRaises(ValueError):
            scene.coupling.add_track([3, 4, 5], [quarter] * 3)  # two slots are left
        np.testing.assert_array_equal(scene.coupling.add_track([3, 4], [quarter] * 2), [3, 4])
        self.assertEqual(scene.coupling.track_count, 2)
        with self.assertRaises(ValueError):
            scene.coupling.add_track([5], [quarter])  # no track is left
        np.testing.assert_array_equal(scene.thread.eyelet_element.numpy(), -1)  # made, not threaded

    def test_coupling_refuses_a_batch_of_chains(self):
        """Fails if a solver with more than one chain is accepted."""
        vertices, tetrahedra = _block((2, 2, 1))
        chain = ThreadModel.create(
            _line((0.0, 0.0, 0.001), 4)[:, :3],
            young_modulus=THREAD_MODULUS,
            radius=THREAD_RADIUS,
            density=THREAD_DENSITY,
            device=DEVICE,
        )
        thread = SolverThread(
            newton.ModelBuilder().finalize(device=DEVICE),
            chain,
            n_envs=2,
            contact_distance=THREAD_RADIUS,
            contact_stiffness=SPRING,
        )
        with self.assertRaises(ValueError):
            ThreadTissueCoupling(
                thread,
                vertices,
                _outward_triangles(vertices, tetrahedra),
                tetrahedra,
                tissue_stiffness=TISSUE_STIFFNESS,
                contact_margin=MARGIN,
            )

    def test_eyelets_sit_at_their_material_points_of_the_deformed_tissue(self):
        """Fails if an eyelet's position is not its tetrahedron's corners weighted with its barycentric weights."""
        scene = _Through()
        self.assertGreater(scene.track_weights.max() - scene.track_weights.min(), 0.3)  # unequal weights
        # A material point follows an affine map of the tissue.
        stretch = np.array([[1.05, 0.10, 0.00], [0.00, 0.90, 0.20], [0.05, 0.00, 1.10]])
        shift = np.array([0.001, -0.002, 0.003])
        scene.place(scene.rest @ stretch.T + shift)
        scene.coupling.update(scene.particle_q)
        np.testing.assert_allclose(
            scene.thread.eyelet_target.numpy()[scene.slots],
            scene.points @ stretch.T + shift,
            rtol=0.0,
            atol=SURFACE_ROUNDING,
        )

    def test_threaded_track_holds_the_thread_at_its_material_points(self):
        """Fails if the thread leaves the track's material points when the tissue carries them away."""
        scene = _Through()
        scene.step(5)
        shift, steps = np.array([0.0, 0.002, 0.001]), 60
        for step in range(1, steps + 1):
            scene.place(scene.rest + shift * step / steps)
            scene.advance()
        scene.step(60)
        targets = scene.thread.eyelet_target.numpy()[scene.slots]
        np.testing.assert_allclose(targets, scene.points + shift, rtol=0.0, atol=SURFACE_ROUNDING)
        # Node 0 stays 13 mm outside the block, so the thread bends on its way in. The force that bends it, of
        # the order E I |shift| / (13 mm)³ = 1e-6 N, stretches an eyelet's spring by 1e-8 m; measured: up to
        # 1.8e-7 m and 16 µN. The bound is a thousandth of the way the track went.
        gaps = np.linalg.norm(scene.thread.eyelet_point.numpy()[scene.slots] - targets, axis=1)
        self.assertLess(gaps.max(), 1.0e-3 * np.linalg.norm(shift))
        # The thread's forces on the eyelets reach the tissue with their sum and their moment: each is spread
        # over its tetrahedron with the material point's weights. The nodes beside the mouths add the last
        # 1e-9 N of their contact with the block's faces. The thread still creeps within the solver's
        # tolerance, and the applied force trails it by a few steps: the bound is that tolerance over all
        # nodes, 3e-8 N against pulls of 1.6e-5 N, with the block's length of 16 mm as the lever. Measured:
        # 1.5e-9 N and 6e-12 N m.
        pulls = scene.thread.eyelet_force.numpy()[scene.slots]
        pushes, planes = -scene.thread.contact_force.numpy(), scene.thread.contact_point.numpy()
        forces, positions = scene.forces(), scene.rest + shift
        self.assertGreater(np.abs(pulls).max(), NODE_WEIGHT)
        bound = (THROUGH + 1) * SOLVER_TOLERANCE
        np.testing.assert_allclose(forces.sum(axis=0), pulls.sum(axis=0) + pushes.sum(axis=0), rtol=0.0, atol=bound)
        np.testing.assert_allclose(
            np.cross(positions, forces).sum(axis=0),
            np.cross(targets, pulls).sum(axis=0) + np.cross(planes, pushes).sum(axis=0),
            rtol=0.0,
            atol=bound * 4.0 * EDGE,
        )

    def test_thread_slides_through_its_eyelets(self):
        """Fails if the eyelets hold the thread along its length or lose the point of the chain they track."""
        scene = _Through()
        scene.step(5)
        # An eyelet finds its point of the chain to a millionth of an element; the bound is ten times that.
        np.testing.assert_allclose(scene.tracked()[scene.slots], scene.stations, rtol=0.0, atol=1.0e-5)
        # A fifth of an element: the nodes beside the mouths stay 0.6 mm from the block's faces.
        slide = 0.2 * ELEMENT
        end = scene.nodes()[-1, :3].copy()
        scene.draw(slide, 30)
        scene.step(30)
        # Nothing holds the thread back, so it follows as a whole, to the stretch of its mass damping's drag:
        # measured 1e-8 m, with eyelet forces below 1e-9 N. Tied to the block it would be stretched by the whole
        # slide, with 1.4 mN.
        np.testing.assert_allclose(scene.nodes()[-1, :3], end - np.array([slide, 0.0, 0.0]), atol=1.0e-3 * slide)
        np.testing.assert_allclose(scene.tracked()[scene.slots], scene.stations + slide / ELEMENT, atol=1.0e-3)
        tied = AXIAL_STIFFNESS * slide / (THROUGH * ELEMENT)
        self.assertLess(np.abs(scene.thread.eyelet_force.numpy()).max(), 1.0e-3 * tied)
        self.assertLess(np.abs(scene.forces()).max(), 1.0e-3 * tied)

    def test_held_eyelet_ties_the_thread(self):
        """Fails if a held eyelet lets the thread slide or pulls its tetrahedron with another force than the tension."""
        scene = _Through()
        scene.step(5)
        knot = scene.slots[-1]
        scene.coupling.hold_eyelet(knot)
        element, xi = scene.thread.eyelet_element.numpy()[knot], scene.thread.eyelet_xi.numpy()[knot]
        pull = 1.0e-4  # [m], a twentieth of an element
        scene.draw(pull, 30)
        scene.step(60)
        self.assertEqual(scene.thread.eyelet_element.numpy()[knot], element)
        self.assertEqual(scene.thread.eyelet_xi.numpy()[knot], xi)

        # The thread is taut from node 0 to the knot, 14.4 elements on, and slack beyond it. Halfway, the
        # elements are free of the ends: node 0's slope is prescribed with unit length, which keeps the strain
        # out of part of the first element and 0.05 % of it out of the fourth.
        taut, slack = 7, THROUGH - 2
        tension = scene.tension(taut)
        free = (element + xi) * ELEMENT
        # The tension therefore lies between that of the whole free length in series with the knot's spring and
        # that of the free length less one element; the Green strain adds 1.5 pull / free = 0.5 % to both.
        # Measured: 475.9 µN between 467.9 µN and 506.1 µN.
        self.assertGreater(tension, pull / (free / AXIAL_STIFFNESS + 1.0 / SPRING))
        self.assertLess(tension, 1.01 * pull / ((free - ELEMENT) / AXIAL_STIFFNESS + 1.0 / SPRING))
        self.assertLess(abs(scene.tension(slack)), 1.0e-3 * tension)

        # The knot takes the whole tension, towards node 0, through its spring in every direction; the sliding
        # eyelets take none. Measured: the knot's force equals the tension to 1e-5 of it, the others are
        # below 1e-9 N.
        pulls = scene.thread.eyelet_force.numpy()
        np.testing.assert_allclose(pulls[knot], (-tension, 0.0, 0.0), rtol=0.0, atol=1.0e-3 * tension)
        gap = scene.thread.eyelet_point.numpy()[knot] - scene.thread.eyelet_target.numpy()[knot]
        np.testing.assert_allclose(pulls[knot], SPRING * gap, rtol=1.0e-6, atol=1.0e-6 * tension)
        self.assertLess(np.abs(pulls[scene.slots[:-1]]).max(), 1.0e-3 * tension)

        # The tissue takes that force on the four corners of the knot's tetrahedron, by the knot's weights.
        expected = np.zeros((len(scene.rest), 3))
        expected[scene.tetrahedra[scene.track_tetrahedra[-1]]] = np.outer(scene.track_weights[-1], pulls[knot])
        np.testing.assert_allclose(scene.forces(), expected, rtol=0.0, atol=1.0e-3 * tension)

        # Let go, the thread slides on through the eyelet until it is slack.
        scene.coupling.hold_eyelet(knot, False)
        scene.step(120)
        self.assertGreater(scene.tracked()[knot], element + xi + 0.5 * pull / ELEMENT)
        self.assertLess(abs(scene.tension(taut)), 1.0e-3 * tension)

    def test_an_eyelet_without_the_thread_is_not_tied(self):
        """Fails if a tie of an eyelet that the thread does not pass through is stored, or a threaded one refused."""
        scene = _Scene(_line((0.0031, 0.0013, 0.001), RESTING + 1), eyelets=5, max_tracks=2)
        quarter = np.full(4, 0.25)
        slots = scene.coupling.add_track([0, 1, 2], [quarter] * 3)
        scene.coupling.hold_eyelet(slots[1])
        np.testing.assert_array_equal(scene.thread.eyelet_held.numpy(), 0)
        # Threaded later, the eyelet is an eyelet: the refused tie has left nothing behind.
        scene.coupling.thread_eyelet(slots[1], 3, 0.25)
        np.testing.assert_array_equal(scene.thread.eyelet_held.numpy(), 0)
        scene.coupling.hold_eyelet(slots[1])
        np.testing.assert_array_equal(scene.thread.eyelet_held.numpy(), [0, 1, 0, 0, 0])
        # Letting go is never refused, with or without the thread.
        scene.coupling.hold_eyelet(slots[0], False)
        scene.coupling.hold_eyelet(slots[1], False)
        np.testing.assert_array_equal(scene.thread.eyelet_held.numpy(), 0)

    def test_tension_of_a_tied_thread_follows_the_travel_of_its_end_by_the_green_strain(self):
        """Fails if a tied thread whose first node is drawn on carries another tension than its stretch gives.

        The tightening of a stitch: the thread is tied at an eyelet and the needle, which carries the
        thread's first node, moves on. The thread between the node and the knot is stretched by the
        travel less what the knot's spring gives, and its tension follows the Green strain of that
        stretch, ``E A (s² - 1) s / 2`` for the stretch ``s``; at the 8 % of a tightened stitch that is
        12 % more than a tension proportional to the elongation.
        """
        scene = _Through()
        scene.step(5)
        knot = scene.slots[-1]
        scene.coupling.hold_eyelet(knot)
        free = float(scene.tracked()[knot]) * ELEMENT

        def tension_of(travel, length):
            """Tension [N] of ``length`` of thread in series with the knot's spring, its end moved by ``travel``."""
            low, high = 1.0, 1.0 + travel / length
            for _ in range(100):
                stretch = 0.5 * (low + high)
                tension = AXIAL_STIFFNESS * 0.5 * (stretch**2 - 1.0) * stretch
                low, high = (stretch, high) if length * (stretch - 1.0) + tension / SPRING < travel else (low, stretch)
            return AXIAL_STIFFNESS * 0.5 * (low**2 - 1.0) * low

        taut, travelled, proportional = 7, 0.0, []
        for travel in (0.0005, 0.001, 0.002, 0.003):
            scene.draw(travel - travelled, 30)
            scene.step(90)
            travelled = travel
            tension = scene.tension(taut)
            # Node 0's slope is prescribed with unit length, which keeps the strain out of part of the first
            # element: the stretched length lies between the free length and one element less. Measured at
            # 3 mm of travel: 16.29 mN between 16.07 and 17.35 mN; a proportional law gives 15.03 mN at most.
            self.assertGreater(tension, tension_of(travel, free))
            self.assertLess(tension, tension_of(travel, free - ELEMENT))
            proportional.append(travel / ((free - ELEMENT) / AXIAL_STIFFNESS + 1.0 / SPRING))
            # The knot takes that tension, and the eyelets that the thread slides through take none. The
            # tension is that of one element's chord: measured within 0.6 % of the knot's pull.
            pulls = np.linalg.norm(scene.thread.eyelet_force.numpy(), axis=1)
            self.assertAlmostEqual(pulls[knot], tension, delta=1.0e-2 * tension)
            self.assertLess(pulls[scene.slots[:-1]].max(), 1.0e-3 * tension)
        # At 3 mm, 10 % of stretch, the tension is above what a proportional law gives for the shorter of
        # the two lengths: measured 1.084 times that.
        self.assertGreater(tension, 1.05 * proportional[-1])
        self.assertEqual(scene.thread.eyelet_held.numpy()[knot], 1)

    def test_thread_is_drawn_into_a_track_through_its_mouth(self):
        """Fails if a node that reaches the open mouth of a track is stopped by the surface around it."""
        draw, followed = 3.0 * ELEMENT, {}
        for mouths in (False, True):
            scene = _Through(friction=0.0)
            if mouths:
                scene.coupling.open_mouths(LATERAL, ALONG)
            scene.step(5)
            end = scene.nodes()[-1, 0]
            scene.draw(draw, 90)
            scene.step(30)
            followed[mouths] = end - scene.nodes()[-1, 0]
        # Without friction nothing but its inertia and mass damping holds the thread back: with the mouths
        # open it follows as a whole.
        self.assertAlmostEqual(followed[True], draw, delta=0.05 * draw)
        # Without them a node that reaches the mouth is not yet between the track's eyelets, so it still has
        # the plane of the face around the mouth, which stops it with up to k (r + depth of the first eyelet),
        # 35 mN. Measured: the far end follows 1.12 mm of the 6 mm and the thread carries 29 mN.
        self.assertLess(followed[False], 0.5 * draw)

    def test_a_node_over_the_mouth_of_a_threaded_track_takes_no_plane(self):
        """Fails if a node over the hole of a track keeps the plane of the face around it, or one beside the hole loses it."""
        scene = _Through()

        def without():
            # Of the nodes within the contact margin of the block (6 to 15), those without a plane.
            scene.coupling.update(scene.particle_q)
            free = np.flatnonzero(np.all(scene.thread.contact_normal.numpy() == 0.0, axis=1))
            return free[(free >= 6) & (free <= 15)].tolist()

        # The track is threaded from 6.6 to 14.4 elements: the nodes 7 to 14 are inside. Node 6 is 1 mm in
        # front of the face through which the thread goes in and node 15 as far behind the face through
        # which it comes out, each on the line of the track; both have the plane of that face.
        self.assertEqual(without(), list(range(7, 15)))
        normals = scene.thread.contact_normal.numpy()
        np.testing.assert_allclose(normals[[6, 15]], [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)], atol=1.0e-6)
        # With the mouths open: their nearest surface point is over the first and the last eyelet, which lie
        # 0.2 mm under those faces.
        scene.coupling.open_mouths(LATERAL, ALONG)
        self.assertEqual(without(), list(range(6, 16)))
        # An eyelet deeper under the face than the given depth is not at a mouth.
        scene.coupling.open_mouths(LATERAL, 0.0001)
        self.assertEqual(without(), list(range(7, 15)))
        # The tissue carries the track 1 mm aside, more than the radius of the hole: the nodes are beside it.
        scene.coupling.open_mouths(LATERAL, ALONG)
        scene.place(scene.rest + np.array([0.0, 0.001, 0.0]))
        self.assertEqual(without(), list(range(7, 15)))
        # Half that, less than the radius: over the hole again.
        scene.place(scene.rest + np.array([0.0, 0.0005, 0.0]))
        self.assertEqual(without(), list(range(6, 16)))
        for radius, depth in ((0.0, ALONG), (LATERAL, 0.0), (-LATERAL, ALONG), (np.nan, ALONG), (LATERAL, np.inf)):
            with self.assertRaisesRegex(ValueError, "positive radius and a positive depth"):
                scene.coupling.open_mouths(radius, depth)

    def test_only_the_far_end_of_a_track_that_is_being_threaded_is_a_mouth(self):
        """Fails if the eyelet that the tail has passed last counts as a mouth while its track is only partly threaded."""
        # Every node is 1 mm over the top; three eyelets lie 0.2 mm under the top, under the nodes 15, 16 and 17.
        scene = _Scene(_line((0.0031, 0.0013, 0.001), RESTING + 1), eyelets=5, max_tracks=2)
        found = [
            _material_point(scene.vertices, scene.tetrahedra, np.array([x, 0.0013, -0.0002]))
            for x in (0.0331, 0.0351, 0.0371)
        ]
        slots = scene.coupling.add_track([tetrahedron for tetrahedron, _ in found], [weights for _, weights in found])
        scene.coupling.open_mouths(LATERAL, ALONG)

        def without():
            scene.coupling.update(scene.particle_q)
            return np.flatnonzero(np.all(scene.thread.contact_normal.numpy() == 0.0, axis=1)).tolist()

        self.assertEqual(without(), [])
        # Two of the three are threaded, at 3 and at 5 elements along the thread: the thread from its start to
        # 5 elements is inside the tissue. The eyelet at 5 elements is the hole through which the thread went
        # in, and the node over it (17) has no plane. The eyelet at 3 elements is where the tail is now, in
        # the tissue: the node over it (15) keeps its plane.
        scene.coupling.thread_eyelet(slots[0], 3, 0.0)
        scene.coupling.thread_eyelet(slots[2], 5, 0.0)
        self.assertEqual(without(), [0, 1, 2, 3, 4, 5, 17])
        # The whole track is threaded: both ends are mouths, and the node over the middle eyelet has its plane.
        scene.coupling.thread_eyelet(slots[1], 4, 0.0)
        self.assertEqual(without(), [3, 4, 5, 15, 17])

    def test_nodes_inside_a_track_take_no_plane(self):
        """Fails if a node between a track's first and last threaded eyelet, or in a half-made track, has contact."""
        # Every node is 1 mm over the top, well inside the margin.
        scene = _Scene(_line((0.0031, 0.0013, 0.001), RESTING + 1), eyelets=5, max_tracks=2)
        quarter = np.full(4, 0.25)
        earlier = scene.coupling.add_track([0, 1, 2], [quarter] * 3)
        later = scene.coupling.add_track([3, 4], [quarter] * 2)

        def inside():
            scene.coupling.update(scene.particle_q)
            normals = scene.thread.contact_normal.numpy()
            without = np.all(normals == 0.0, axis=1)
            np.testing.assert_allclose(normals[~without], np.tile((0.0, 0.0, 1.0), ((~without).sum(), 1)), atol=1.0e-6)
            return np.flatnonzero(without).tolist()

        self.assertEqual(inside(), [])
        # A half-made track: the thread from its first node to the last threaded eyelet is in the tissue.
        scene.coupling.thread_eyelet(earlier[1], 14, 0.0)
        self.assertEqual(inside(), list(range(15)))
        # The whole track: only the nodes from its first to its last eyelet, here at 12.5 and 16.25 elements.
        scene.coupling.thread_eyelet(earlier[0], 16, 0.25)
        scene.coupling.thread_eyelet(earlier[2], 12, 0.5)
        self.assertEqual(inside(), [13, 14, 15, 16])
        # The next stitch is nearer the thread's start. Half made, it takes the thread from node 0 to its eyelet.
        scene.coupling.thread_eyelet(later[0], 5, 0.0)
        self.assertEqual(inside(), [0, 1, 2, 3, 4, 5, 13, 14, 15, 16])
        # Made, it takes the nodes from 2.5 to 5 elements, and the thread between the two tracks lies on the top.
        scene.coupling.thread_eyelet(later[1], 2, 0.5)
        self.assertEqual(inside(), [3, 4, 5, 13, 14, 15, 16])


class FrictionTests(unittest.TestCase):
    def test_anchors_of_nodes_with_grip_move_with_the_surface(self):
        """Fails if update leaves the friction anchors behind when the surface under them moves."""
        scene = _resting()
        scene.step(20)
        before = scene.thread.contact_anchor.numpy().copy()
        grip = scene.thread.contact_limit.numpy() > 0.0
        np.testing.assert_array_equal(grip, np.arange(RESTING + 1) > 0)  # node 0 has no contact of its own
        # A shift across the thread that grows along it: an anchor moves as its own point of the surface does.
        moved = scene.rest.copy()
        moved[:, 1] += 0.0004 + 0.02 * moved[:, 0]
        scene.place(moved)
        scene.coupling.update(scene.particle_q)
        carried = scene.thread.contact_anchor.numpy() - before
        expected = np.zeros_like(before)
        expected[:, 1] = 0.0004 + 0.02 * before[:, 0]
        np.testing.assert_allclose(carried[grip, :2], expected[grip, :2], rtol=0.0, atol=SURFACE_ROUNDING)
        np.testing.assert_array_equal(carried[~grip], 0.0)

    def test_anchors_without_friction_are_left_alone(self):
        """Fails if update moves the anchor of a node that has no grip."""
        scene = _resting(friction=0.0)
        scene.step(20)
        before = scene.thread.contact_anchor.numpy().copy()
        np.testing.assert_array_equal(scene.thread.contact_limit.numpy(), 0.0)
        scene.place(scene.rest + np.array([0.0, 0.0004, 0.0]))
        scene.coupling.update(scene.particle_q)
        np.testing.assert_array_equal(scene.thread.contact_anchor.numpy(), before)

    def test_sticking_thread_rides_on_the_tissue_and_a_frictionless_one_stays(self):
        """Fails if a thread with friction is left behind by the moving surface, or one without is taken along."""
        # The block goes 0.5 mm across the thread in one second. Its start from rest in one step asks for
        # 0.5 mm/s / dt = 0.03 m/s², far below the µ g = 2 m/s² that friction can give a node.
        shift, steps = np.array([0.0, 0.0005, 0.0]), 60
        moved = {}
        for friction in (FRICTION, 0.0):
            scene = _resting(y=-0.0023, friction=friction)
            scene.step(20)
            before = scene.nodes()[:, :3].copy()
            for step in range(1, steps + 1):
                scene.place(scene.rest + shift * step / steps)
                scene.advance()
            scene.step(30)
            moved[friction] = scene.nodes()[:, :3] - before
        # Node 0 is held in space, so the thread bends between it and the part that rides. The bend reaches
        # about (12 E I d / (µ ρ A g))^(1/4) = 13 mm; the last five nodes are 30 mm and more from node 0.
        # Measured: they ride 0.9997 to 1.0001 of the way; the bound is a hundredth of it.
        np.testing.assert_allclose(moved[FRICTION][-5:, 1], shift[1], rtol=0.0, atol=0.01 * shift[1])
        np.testing.assert_array_equal(moved[FRICTION][0], 0.0)
        # Without friction no force acts along the level top; measured: no node moves more than 3e-11 m.
        self.assertLess(np.abs(moved[0.0][:, :2]).max(), 1.0e-6 * shift[1])

    def test_sliding_thread_drags_the_tissue_with_its_friction(self):
        """Fails if the friction of a sliding thread does not reach the particles, opposite and in full."""
        scene = _resting(x=0.0071)
        scene.step(20)
        start = scene.nodes()[0, :3].copy()
        # Node 0 draws the thread along itself at 4 mm/s for 1.5 s, which leaves every node on the top; the
        # thread slides as a whole after the first steps.
        steps = 90
        for step in range(1, steps + 1):
            scene.hold_start(start - np.array([0.004 * DT * step, 0.0, 0.0]))
            scene.advance()
        forces, contact = scene.forces(), scene.thread.contact_force.numpy()
        self.assertLess(scene.nodes()[-1, 0] - (start[0] + RESTING * ELEMENT), -0.9 * 0.004 * DT * steps)
        # The thread is held back by µ times its normal forces, so the block is drawn after it with as much.
        # The method notes give 0.4 % for this with eight updates a step; measured: equal to 2e-8. The bound
        # is 1 %.
        pressing = -forces[:, 2].sum()
        self.assertAlmostEqual(pressing, (RESTING - 0.5) * NODE_WEIGHT, delta=0.01 * RESTING * NODE_WEIGHT)
        self.assertAlmostEqual(forces[:, 0].sum(), -FRICTION * pressing, delta=0.01 * FRICTION * pressing)
        # Action and reaction in every direction, to the solver's tolerance over all nodes.
        np.testing.assert_allclose(
            forces.sum(axis=0), -contact.sum(axis=0), rtol=0.0, atol=(RESTING + 1) * SOLVER_TOLERANCE
        )


# Elements of the thread that hang over the end of the long block.
HANGING = 6


def _over_the_edge():
    """A thread along x on the top of the long block whose last elements hang over the block's end face.

    Node RESTING - 1 is the last on the top, 1 mm from the edge; the next one is a thread radius off the end
    face, one element from it, and the rest hang straight down.
    """
    edge = 12 * EDGE
    top = _line((edge - 0.001 - ELEMENT * (RESTING - 1), 0.0013, _rest_height(NODE_WEIGHT)), RESTING)
    beside = edge + THREAD_RADIUS
    drop = math.sqrt(ELEMENT**2 - (beside - top[-1, 0]) ** 2)
    down = _line((beside, 0.0013, top[-1, 2] - drop), HANGING, direction=(0.0, 0.0, -1.0))
    return _Scene(np.vstack((top, down)))


class EdgeTests(unittest.TestCase):
    # Defect: only nodes have contact, each with the plane of the face of its nearest surface point. A node
    # that is drawn back to an edge from beyond it is stopped by the plane of the face beside the edge.
    @unittest.expectedFailure
    def test_thread_that_hangs_over_an_edge_is_drawn_back_over_it(self):
        """Fails while the node beside the end face is held by that face's plane when the thread is drawn back."""
        scene = _over_the_edge()
        scene.hold_start(scene.nodes()[0, :3])
        scene.step(60)
        start = scene.nodes()[0, :3].copy()
        steps = 120
        for step in range(1, steps + 1):
            scene.hold_start(start - np.array([ELEMENT * step / steps, 0.0, 0.0]))
            scene.advance()
        scene.step(60)
        # A thread slides over the edge: what it carries on the top is the weight that hangs, raised by the
        # friction round a quarter turn, and the friction of what lies on the top. Twice that is allowed.
        hanging = HANGING * NODE_WEIGHT
        carried = hanging * math.exp(0.5 * math.pi * FRICTION) + FRICTION * RESTING * NODE_WEIGHT
        # Measured: 4.2 mN, 440 times the hanging weight, after the thread is drawn back by one element.
        # Node RESTING sits 0.11 mm off the end face and 0.14 mm below the top; its plane is the end face's
        # (normal +x), which holds it with 4.1 mN against the pull and with 0.6 mN of friction against the
        # rise. The pull has no part along that plane until the node is level with its neighbour on the
        # top, and the element between the two passes through the block's corner. Without friction the
        # node rises above the top and is still held: its nearest surface point is then on the edge, and
        # the plane is still the end face's.
        self.assertLess(scene.tension(5), 2.0 * carried)


def _dragged():
    """A thread on the block whose node 0 is drawn along it while the block moves across: it rides and slides."""
    scene = _resting(x=0.0071, y=-0.0023, eyelets=2)
    start = scene.nodes()[0, :3].copy()

    def inputs(step):
        # The block at 0.5 mm/s across the thread, node 0 at 4 mm/s along it, away from the thread.
        scene.place(scene.rest + np.array([0.0, 0.0005 * DT * step, 0.0]))
        scene.hold_start(start - np.array([0.004 * DT * step, 0.0, 0.0]))

    return scene, inputs


def _uneven(index):
    """Four barycentric weights that differ from eyelet to eyelet and from corner to corner."""
    weights = np.array([0.1, 0.2, 0.3, 0.4]) + 0.01 * index * np.array([1.0, -1.0, 2.0, -2.0])
    return weights / weights.sum()


class TrackPointTests(unittest.TestCase):
    HOST = ("_tetrahedron", "_barycentric", "_group", "_group_size")
    DEVICE_ARRAYS = ("_tetrahedron_device", "_barycentric_device", "_group_device", "_group_size_device")

    def test_a_track_built_point_by_point_equals_add_track_in_every_host_and_device_array(self):
        """Fails if add_track_point writes another row, another array or another value than add_track does."""
        whole = _Scene(_line((0.0031, 0.0013, 0.001), RESTING + 1), eyelets=7, max_tracks=3)
        grown = _Scene(_line((0.0031, 0.0013, 0.001), RESTING + 1), eyelets=7, max_tracks=3)
        tetrahedra = [[11, 3, 40], [7, 25]]
        weights = [[_uneven(0), _uneven(1), _uneven(2)], [_uneven(3), _uneven(4)]]
        slots = [
            whole.coupling.add_track(group, group_weights)
            for group, group_weights in zip(tetrahedra, weights, strict=True)
        ]
        made = []
        for track, (group, group_weights) in enumerate(zip(tetrahedra, weights, strict=True)):
            for tetrahedron, weight in zip(group, group_weights, strict=True):
                made.append(grown.coupling.add_track_point(track, tetrahedron, weight))
                # After every point: the track count, and the eyelets so far, are those of the host copies.
                self.assertEqual(grown.coupling.track_count, track + 1)
                self.assertEqual(grown.coupling.point_count, len(made))
        self.assertEqual(made, [int(slot) for group in slots for slot in group])
        self.assertEqual((grown.coupling.track_count, grown.coupling.point_count), (2, 5))
        self.assertEqual((whole.coupling.track_count, whole.coupling.point_count), (2, 5))
        for name in self.HOST:
            np.testing.assert_array_equal(getattr(grown.coupling, name), getattr(whole.coupling, name), err_msg=name)
        for name in self.DEVICE_ARRAYS:
            np.testing.assert_array_equal(
                getattr(grown.coupling, name).numpy(), getattr(whole.coupling, name).numpy(), err_msg=name
            )
        # The device holds the host copies, and no array was replaced: the rows were written in place.
        for host, device in zip(self.HOST, self.DEVICE_ARRAYS, strict=True):
            np.testing.assert_array_equal(getattr(grown.coupling, device).numpy(), getattr(grown.coupling, host))
        fresh = _Scene(_line((0.0031, 0.0013, 0.001), RESTING + 1), eyelets=7, max_tracks=3)
        pointers = [getattr(fresh.coupling, name).ptr for name in self.DEVICE_ARRAYS]
        fresh.coupling.add_track_point(0, 5, _uneven(0))
        fresh.coupling.add_track_point(1, 6, _uneven(1))
        self.assertEqual([getattr(fresh.coupling, name).ptr for name in self.DEVICE_ARRAYS], pointers)
        # Made, not threaded; the eyelets sit at the same material points either way.
        np.testing.assert_array_equal(grown.thread.eyelet_element.numpy(), -1)
        whole.coupling.update(whole.particle_q)
        grown.coupling.update(grown.particle_q)
        np.testing.assert_array_equal(grown.thread.eyelet_target.numpy(), whole.thread.eyelet_target.numpy())
        # A point may be added to an earlier track after a later one was begun: it takes the next free slot.
        self.assertEqual(grown.coupling.add_track_point(0, 9, _uneven(5)), 5)
        np.testing.assert_array_equal(grown.coupling._group_device.numpy()[:6], [0, 0, 0, 1, 1, 0])
        np.testing.assert_array_equal(grown.coupling._group_size_device.numpy(), [4, 2, 0])

    def test_a_used_up_reserve_refuses_a_point_and_counts_it(self):
        """Fails if a point beyond the eyelet slots or the tracks is made, grows an array, or goes uncounted."""
        scene = _Scene(_line((0.0031, 0.0013, 0.001), RESTING + 1), eyelets=3, max_tracks=2)
        coupling = scene.coupling
        self.assertEqual(
            [coupling.add_track_point(0, 1, _uneven(0)), coupling.add_track_point(0, 2, _uneven(1))], [0, 1]
        )
        self.assertEqual(coupling.refused_points, 0)
        # A point can only join a track that exists or open the next one.
        with self.assertRaises(ValueError):
            coupling.add_track_point(2, 3, _uneven(2))
        with self.assertRaises(ValueError):
            coupling.add_track_point(-1, 3, _uneven(2))
        with self.assertRaises(ValueError):
            coupling.add_track_point(0, len(scene.tetrahedra), _uneven(2))  # no such tetrahedron
        with self.assertRaises(ValueError):
            coupling.add_track_point(0, 3, [0.5, 0.5])
        self.assertEqual((coupling.refused_points, coupling.point_count), (0, 2))
        self.assertEqual(coupling.add_track_point(1, 3, _uneven(2)), 2)
        before = [getattr(coupling, name).numpy().copy() for name in self.DEVICE_ARRAYS]
        host = [getattr(coupling, name).copy() for name in self.HOST]
        # No slot is left: refused and counted, twice.
        self.assertEqual(coupling.add_track_point(1, 4, _uneven(3)), -1)
        self.assertEqual(coupling.add_track_point(0, 4, _uneven(3)), -1)
        self.assertEqual((coupling.refused_points, coupling.point_count, coupling.track_count), (2, 3, 2))
        for name, values in zip(self.DEVICE_ARRAYS, before, strict=True):
            np.testing.assert_array_equal(getattr(coupling, name).numpy(), values, err_msg=name)
            self.assertEqual(len(getattr(coupling, name)), len(values))
        for name, values in zip(self.HOST, host, strict=True):
            np.testing.assert_array_equal(getattr(coupling, name), values, err_msg=name)
        # No track is left, though slots are: refused and counted as well.
        roomy = _Scene(_line((0.0031, 0.0013, 0.001), RESTING + 1), eyelets=6, max_tracks=2).coupling
        self.assertEqual([roomy.add_track_point(0, 1, _uneven(0)), roomy.add_track_point(1, 2, _uneven(1))], [0, 1])
        self.assertEqual(roomy.add_track_point(2, 3, _uneven(2)), -1)
        self.assertEqual((roomy.refused_points, roomy.point_count, roomy.track_count), (1, 2, 2))
        np.testing.assert_array_equal(roomy._group_size_device.numpy(), [1, 1])


# The tail of the tests: a point of a rigid body that moves along x through a row of eyelets inside the
# block, with the thread leaving it backwards. Reach of the rule along and across, as the worker thread sets them.
ALONG = 0.0015  # [m], one eyelet spacing
LATERAL = 0.00056  # [m], two wire radii of the needle
TAIL_Y, TAIL_Z = 0.0013, -0.0021  # [m]
TAIL_EYELETS = np.array([0.0102, 0.0117, 0.0132, 0.0147])  # [m] along x, one spacing apart


class _Tailed(_Scene):
    """A thread over the long block, a track of four unthreaded eyelets inside it and one rigid body to follow."""

    def __init__(self, *, point=(0.0, 0.0, 0.0), outward=(-1.0, 0.0, 0.0), follow=True):
        super().__init__(_line((0.0031, 0.0013, 0.001), RESTING + 1), eyelets=6, max_tracks=2)
        self.points = np.column_stack(
            (TAIL_EYELETS, np.full(len(TAIL_EYELETS), TAIL_Y), np.full(len(TAIL_EYELETS), TAIL_Z))
        )
        self.slots = []
        for point_ in self.points:
            tetrahedron, weights = _material_point(self.vertices, self.tetrahedra, point_)
            self.slots.append(self.coupling.add_track_point(0, tetrahedron, weights))
        self.body_q = wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=DEVICE)
        if follow:
            self.coupling.follow_tail(1, point, outward, ALONG, LATERAL)

    def tail_at(self, position, rotation=None):
        """Put the followed body (index 1) at ``position`` [m] with an optional rotation; body 0 stays elsewhere."""
        rotation = wp.quat_identity() if rotation is None else rotation
        self.body_q.assign(
            [wp.transform(wp.vec3(0.5, 0.5, 0.5), wp.quat_identity()), wp.transform(wp.vec3(*position), rotation)]
        )
        self.coupling.update(self.particle_q, self.body_q)
        return [slot for slot in self.slots if self.thread.eyelet_element.numpy()[slot] >= 0]


class TailTests(unittest.TestCase):
    def test_the_tail_threads_an_eyelet_when_it_has_passed_it_and_not_before_in_track_order(self):
        """Fails if an eyelet ahead of the tail is threaded, one behind it is not, or the order is not the track's."""
        scene = _Tailed()
        order, when = [], {}
        # The tail comes along the track from before its first eyelet, a quarter millimetre per step.
        for x in np.arange(0.0090, 0.0160, 0.00025):
            threaded = scene.tail_at((x, TAIL_Y, TAIL_Z))
            for slot in threaded:
                if slot not in order:
                    order.append(slot)
                    when[slot] = x
            # Nothing ahead of the tail is threaded: every threaded eyelet lies behind it.
            for slot in threaded:
                self.assertLessEqual(scene.points[slot][0], x + 1e-7)
            # Everything the tail has passed is threaded.
            for slot in scene.slots:
                if scene.points[slot][0] <= x - 1e-6:
                    self.assertIn(slot, threaded)
        self.assertEqual(order, scene.slots)
        # Each in the first step in which the tail was at or past it: within one step of 0.25 mm.
        for slot in scene.slots:
            self.assertGreaterEqual(when[slot], scene.points[slot][0] - 1e-7)
            self.assertLess(when[slot], scene.points[slot][0] + 0.00025 + 1e-7)
        # Threaded at the thread's first node.
        np.testing.assert_array_equal(scene.thread.eyelet_element.numpy()[scene.slots], 0)
        np.testing.assert_array_equal(scene.thread.eyelet_xi.numpy()[scene.slots], 0.0)
        # The slots without a material point stay unthreaded, also when the tail goes past where their
        # unwritten position is (the origin) as it would go past an eyelet.
        np.testing.assert_array_equal(scene.thread.eyelet_target.numpy()[4:], 0.0)
        scene.tail_at((0.0005, 0.0, 0.0))
        np.testing.assert_array_equal(scene.thread.eyelet_element.numpy()[4:], -1)

    def test_the_tail_threads_nothing_from_ahead_of_an_eyelet_from_a_millimetre_beside_it_or_from_far_past_it(self):
        """Fails if the rule reaches further than one spacing back or two wire radii across."""
        # Ahead of every eyelet by a hair: nothing.
        scene = _Tailed()
        self.assertEqual(scene.tail_at((TAIL_EYELETS[0] - 2e-6, TAIL_Y, TAIL_Z)), [])
        # The whole way a millimetre beside the track (across it, and below it): nothing.
        for offset in ((0.0, 0.001, 0.0), (0.0, 0.0, -0.001), (0.0, -0.0007, 0.0007)):
            scene = _Tailed()
            for x in np.arange(0.0090, 0.0165, 0.00025):
                self.assertEqual(scene.tail_at(np.array([x, TAIL_Y, TAIL_Z]) + offset), [], msg=str(offset))
        # Just inside and just outside the reach across (0.56 mm), with the tail level with the first eyelet.
        inside, outside = _Tailed(), _Tailed()
        self.assertEqual(inside.tail_at((TAIL_EYELETS[0] + 1e-5, TAIL_Y + LATERAL - 2e-5, TAIL_Z)), inside.slots[:1])
        self.assertEqual(outside.tail_at((TAIL_EYELETS[0] + 1e-5, TAIL_Y + LATERAL + 2e-5, TAIL_Z)), [])
        # Far past: a tail that appears 1.52 mm behind the last eyelet threads nothing; 1.48 mm behind, that one.
        far, near = _Tailed(), _Tailed()
        self.assertEqual(far.tail_at((TAIL_EYELETS[-1] + ALONG + 2e-5, TAIL_Y, TAIL_Z)), [])
        self.assertEqual(near.tail_at((TAIL_EYELETS[-1] + ALONG - 2e-5, TAIL_Y, TAIL_Z)), near.slots[-1:])
        # A threaded eyelet stays as the thread solver has it: the kernel does not thread it again.
        near.coupling.thread_eyelet(near.slots[-1], 7, 0.25)
        near.tail_at((TAIL_EYELETS[-1] + 1e-5, TAIL_Y, TAIL_Z))
        self.assertEqual(float(near.tracked()[near.slots[-1]]), 7.25)

    def test_the_followed_point_and_its_direction_turn_and_move_with_the_body(self):
        """Fails if the point or the outward direction are taken in world axes instead of the body's."""
        # The body is turned a quarter turn about z: its local y is the world's -x. The followed point is
        # 2 mm along local y, and the thread leaves it along local y as well (given with a length of 3).
        turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * math.pi)
        scene = _Tailed(point=(0.0, 0.002, 0.0), outward=(0.0, 3.0, 0.0))
        # World point = body + R (0, 2 mm, 0) = body + (-2 mm, 0, 0); outward = R (0, 1, 0) = (-1, 0, 0).
        # The point is 1 mm past the second eyelet (further than the reach across, nearer than the reach
        # along) and 2.5 mm past the first: the second only, and only with the direction turned as the body is.
        body = np.array([TAIL_EYELETS[1] + 0.001 + 0.002, TAIL_Y, TAIL_Z])
        self.assertEqual(scene.tail_at(body, turn), scene.slots[1:2])
        # Unturned, the same body has its point elsewhere and its thread leaving along +y: nothing.
        scene = _Tailed(point=(0.0, 0.002, 0.0), outward=(0.0, 3.0, 0.0))
        self.assertEqual(scene.tail_at(body), [])

    def test_following_needs_the_body_poses_and_a_coupling_that_follows_nothing_threads_nothing(self):
        """Fails if update() threads without follow_tail, or follows without being given the bodies."""
        scene = _Tailed(follow=False)
        for x in np.arange(0.0090, 0.0165, 0.0005):
            self.assertEqual(scene.tail_at((x, TAIL_Y, TAIL_Z)), [])
        scene.coupling.update(scene.particle_q)  # as before: one argument
        scene = _Tailed()
        with self.assertRaisesRegex(ValueError, "body poses"):
            scene.coupling.update(scene.particle_q)
        for arguments in (
            (1, (0.0, 0.0), (1.0, 0.0, 0.0), ALONG, LATERAL),
            (1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), ALONG, LATERAL),
            (1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0, LATERAL),
            (1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), ALONG, -1.0),
            (1, (0.0, 0.0, 0.0), (1.0, 0.0, float("nan")), ALONG, LATERAL),
        ):
            with self.assertRaises(ValueError):
                scene.coupling.follow_tail(*arguments)
        # The state of the coupling is what it was: the eyelet arrays are written from the host only.
        self.assertEqual(len(scene.coupling.state), 3)

    def test_a_thread_whose_tail_goes_along_a_curved_track_is_left_on_it(self):
        """Fails if the eyelets that the tail kernel threads do not hold the thread on the track behind the tail."""
        # Deep in the block, more than the contact margin from every face, so that no node has a contact plane
        # and only the eyelets act. The track is an arc of a needle's radius, 9.45 mm, in the plane z = -4 mm,
        # with five eyelets a spacing apart. The tail (the thread's first node, with the body) moves along
        # that arc, 0.125 mm per step, and turns with it; the thread starts straight behind it.
        radius, depth = 0.00945, -0.004
        centre = np.array([0.016, 0.0022 - radius, depth])

        def on_arc(length):
            angle = length / radius
            point = centre + radius * np.array([math.sin(angle), math.cos(angle), 0.0])
            return point, np.array([math.cos(angle), -math.sin(angle), 0.0]), angle

        scene = _Scene(
            _line(on_arc(0.0)[0], 7, direction=(-1.0, 0.0, 0.0)),
            gravity=0.0,
            eyelets=6,
            max_tracks=2,
        )
        stations = 0.001 + ALONG * np.arange(5)
        points = np.array([on_arc(station)[0] for station in stations])
        slots = [
            scene.coupling.add_track_point(0, *_material_point(scene.vertices, scene.tetrahedra, point))
            for point in points
        ]
        scene.coupling.follow_tail(0, (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), ALONG, LATERAL)
        body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device=DEVICE)
        order, travel = [], 0.0
        while travel < 0.00945:
            travel += 0.000125
            point, tangent, angle = on_arc(travel)
            # The thread leaves its first node against the tail's direction of travel.
            scene._boundary[:3], scene._boundary[3:6] = point, -tangent
            scene.thread.boundary_q.assign(scene._boundary)
            turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -angle)
            body_q.assign([wp.transform(wp.vec3(*point), turn)])
            scene.coupling.update(scene.particle_q, body_q)
            scene.thread.step(None, None, None, None, DT)
            for slot in slots:
                if scene.thread.eyelet_element.numpy()[slot] >= 0 and slot not in order:
                    order.append(slot)
        # No node had a contact plane: the block's faces are beyond the margin.
        np.testing.assert_array_equal(scene.thread.contact_normal.numpy(), 0.0)
        self.assertEqual(order, slots)
        tracked = scene.tracked()[slots]
        # The first-passed eyelet is furthest along the thread, neighbours one spacing (0.75 element) apart,
        # and the last one as far behind the tail as the tail has gone past it.
        np.testing.assert_allclose(np.diff(tracked), -ALONG / ELEMENT, atol=0.05)
        self.assertAlmostEqual(tracked[-1], (travel - stations[-1]) / ELEMENT, delta=0.1)
        # The thread passes within a thread radius of every eyelet ...
        offsets = np.linalg.norm(scene.thread.eyelet_point.numpy()[slots] - points, axis=1)
        self.assertLess(offsets.max(), THREAD_RADIUS)
        # ... and its nodes between the tail and the first eyelet lie on the arc, where a straight thread
        # from the tail to that eyelet would be 0.9 mm inside it (chord of 8.4 mm on the 9.45 mm circle).
        nodes = scene.nodes()[: int(tracked[0]) + 1, :3]
        from_arc = np.abs(np.linalg.norm(nodes - centre, axis=1) - radius)
        self.assertLess(from_arc.max(), THREAD_RADIUS)
        self.assertGreaterEqual(len(nodes), 5)


# How closely a repeated CUDA run agrees: positions to a thousandth of the contact's smoothing width, below
# which no node changes between pressing and free, and forces to ten times the solver's tolerance.
REPEATED_POSITION = 1.0e-3 * SMOOTHING  # [m]
REPEATED_FORCE = 10.0 * SOLVER_TOLERANCE  # [N]


class StateTests(unittest.TestCase):
    def test_state_holds_what_a_restored_run_needs(self):
        """Fails if the coupling's state leaves out an array that the next step reads from the last one."""
        scene, inputs = _dragged()
        for step in range(1, 31):
            inputs(step)
            scene.advance()
        # The applied force is documented as part of the state.
        self.assertTrue(any(array is scene.coupling.applied_force for array in scene.coupling.state))
        state = (*scene.thread.state, *scene.coupling.state)
        saved = [wp.clone(array) for array in state]
        faces = [array.numpy().copy() for array in scene.coupling.state if array.dtype == wp.int32]

        def run():
            record = []
            for step in range(31, 61):
                inputs(step)
                scene.advance()
                record.append((scene.nodes()[:, :3].copy(), scene.forces()))
            return record

        first = run()
        # The run crosses what the state has to carry: the applied force changes and the anchors change triangles.
        later = [array.numpy() for array in scene.coupling.state if array.dtype == wp.int32]
        self.assertGreater(sum(int((new != old).sum()) for new, old in zip(later, faces, strict=True)), RESTING // 2)
        for array, values in zip(state, saved, strict=True):
            wp.copy(array, values)
        second = run()
        # A CUDA run repeats to rounding: positions agree to a thousandth of the contact's smoothing width and
        # forces to ten times the solver's tolerance. Measured: both runs agree bit for bit on cpu and on
        # cuda:0. Left at its later value, the applied force changes the forces by 8e-7 N and either anchor
        # array changes positions by 27 µm or more.
        for (positions, forces), (again, forces_again) in zip(first, second, strict=True):
            np.testing.assert_allclose(again, positions, rtol=0.0, atol=REPEATED_POSITION)
            np.testing.assert_allclose(forces_again, forces, rtol=0.0, atol=REPEATED_FORCE)

    def test_coupled_step_replays_in_a_cuda_graph(self):
        """Fails if update, the thread's step or add_forces do anything a captured CUDA graph cannot replay."""
        if not wp.get_device(DEVICE).is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("conditional CUDA graphs are unavailable on the selected device")
        eager, eager_inputs = _dragged()
        replayed, replayed_inputs = _dragged()
        end = replayed.nodes()[-1, :2].copy()
        state = (*replayed.thread.state, *replayed.coupling.state)
        saved = [wp.clone(array) for array in state]
        # The first step loads the kernels and makes the step's allocations; neither may happen in a capture.
        replayed.advance()
        with wp.ScopedCapture(device=DEVICE) as capture:
            replayed.advance()
        for array, values in zip(state, saved, strict=True):
            wp.copy(array, values)
        for step in range(1, 61):
            eager_inputs(step)
            eager.advance()
            replayed_inputs(step)
            wp.capture_launch(capture.graph)
        # Measured: bit for bit the same.
        np.testing.assert_allclose(replayed.nodes()[:, :3], eager.nodes()[:, :3], rtol=0.0, atol=REPEATED_POSITION)
        np.testing.assert_allclose(replayed.forces(), eager.forces(), rtol=0.0, atol=REPEATED_FORCE)
        # The replayed thread did slide and ride; measured: 4.0 mm along and 0.59 mm across in the 60 steps.
        travelled = replayed.nodes()[-1, :2] - end
        self.assertLess(travelled[0], -0.003)
        self.assertGreater(travelled[1], 0.0003)


if __name__ == "__main__":
    unittest.main()
