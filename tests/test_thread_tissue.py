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

    # Defect: the surface contact stops a node at a track's mouth with up to k (r + depth of the first eyelet), 35 mN.
    @unittest.expectedFailure
    def test_thread_is_drawn_into_a_track_through_its_mouth(self):
        """Fails while nodes that reach the mouth of a track are stopped by the contact with the surface around it."""
        scene = _Through(friction=0.0)
        scene.step(5)
        end = scene.nodes()[-1, 0]
        draw = 3.0 * ELEMENT
        scene.draw(draw, 90)
        scene.step(30)
        # Without friction nothing but its inertia and mass damping holds the thread back: it follows as a
        # whole. A node that reaches the mouth is not yet between the track's eyelets, so it still has the
        # plane of the face around the mouth. Measured: the far end follows 1.12 mm of the 6 mm and the thread
        # carries 29 mN; with the thread's contact switched off it follows the 6 mm and carries none.
        self.assertAlmostEqual(end - scene.nodes()[-1, 0], draw, delta=0.05 * draw)

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


def _dragged():
    """A thread on the block whose node 0 is drawn along it while the block moves across: it rides and slides."""
    scene = _resting(x=0.0071, y=-0.0023, eyelets=2)
    start = scene.nodes()[0, :3].copy()

    def inputs(step):
        # The block at 0.5 mm/s across the thread, node 0 at 4 mm/s along it, away from the thread.
        scene.place(scene.rest + np.array([0.0, 0.0005 * DT * step, 0.0]))
        scene.hold_start(start - np.array([0.004 * DT * step, 0.0, 0.0]))

    return scene, inputs


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
