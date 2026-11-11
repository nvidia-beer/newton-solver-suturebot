# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The suture thread: its reference model, the solver's own stepping, and its attachment to a rigid body."""

import math
import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_suturebot import SolverThread, ThreadAttachment, ThreadModel

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")

# A 0.3 mm nylon suture at the examples' soft modulus, in 2 mm elements.
RADIUS = 1.5e-4  # [m]
DENSITY = 1140.0  # [kg/m³]
MODULUS = 2.0e6  # [Pa]
ELEMENT = 0.002  # [m]
GRAVITY = 9.81  # [m/s²]
LINE_WEIGHT = DENSITY * math.pi * RADIUS**2 * GRAVITY  # [N/m]
NODE_WEIGHT = LINE_WEIGHT * ELEMENT  # [N], the weight of one element, which an inner node carries
BENDING = MODULUS * math.pi * RADIUS**4 / 4.0  # [N m²]
DT = 1.0 / 60.0  # [s]
DAMPING = 2.0  # [1/s], the examples' mass damping

# Width of the smooth contact onset [m], a tenth of the radius as in the examples. At the contact distance the
# penalty pushes with k s / 4 (penetration s / 2, slope 1 / 2). With this stiffness that is one inner node's
# weight, so a level thread rests exactly at the contact distance.
SMOOTHING = 0.1 * RADIUS
FLOOR = {
    "contact_distance": RADIUS,
    "contact_stiffness": 4.0 * NODE_WEIGHT / SMOOTHING,
    "contact_smoothing": SMOOTHING,
}

# The bundled needle's 0.3 g, with radii of gyration of 6 to 8 mm [kg, kg m²].
BODY_MASS = 3.0e-4
BODY_INERTIA = np.array([1.0e-8, 1.5e-8, 2.0e-8])

GAUSS_POINTS, GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(5)


def setUpModule():
    if DEVICE.startswith("cuda") and not wp.is_cuda_available():
        raise unittest.SkipTest(f"NEWTON_TEST_DEVICE={DEVICE} needs CUDA")


def _line(elements, start=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0)):
    """Node positions [m] of a straight chain of ``elements`` elements."""
    return np.asarray(start) + ELEMENT * np.arange(elements + 1)[:, None] * np.asarray(direction)


def _solver(positions, *, device=DEVICE, modulus=MODULUS, model=None, **options):
    """A thread solver for a chain that is stress-free at ``positions``, under gravity along -z by default."""
    thread = ThreadModel.create(positions, young_modulus=modulus, radius=RADIUS, density=DENSITY, device=device)
    options.setdefault("gravity", (0.0, 0.0, -GRAVITY))
    return SolverThread(newton.ModelBuilder().finalize(device=device) if model is None else model, thread, **options)


def _advance(solver, steps, dt=DT):
    for _ in range(steps):
        solver.step(None, None, None, None, dt)


def _nodes(array):
    """A coordinate array on the host, one row of position and slope per node; on a CPU a view, not a copy."""
    return array.numpy().reshape(-1, 6)


def _rounding(steps, updates, dt=DT, modulus=MODULUS):
    """Largest difference [m] that rounding leaves between two solves of the same ``steps`` steps.

    The chain solve is direct, so two solves differ by rounding, which the tangent's condition number amplifies.
    With the slopes scaled by the element length l, its eigenvalues run from a node's inertia m / dt² to about
    five axial element stiffnesses E A / l (measured 5.8e-4 to 338 N/m), a ratio of 5 E dt² / (rho l²). An update
    is at most g dt² long. Its error stays in the position and, through the velocity, moves it again in every
    later step.
    """
    condition = 5.0 * modulus * dt**2 / (DENSITY * ELEMENT**2)
    update_error = condition * np.finfo(float).eps * GRAVITY * dt**2
    return 0.5 * steps * (steps + 1) * updates * update_error


def _shape_derivatives(xi, lengths):
    """First and second derivatives of the cubic Hermite shape functions along the reference length.

    Rows follow [node a, slope a, node b, slope b]; columns the elements, at ``xi`` in [0, 1].
    """
    one = np.ones_like(lengths)
    first = np.array(
        [
            6.0 * (xi * xi - xi) / lengths,
            (1.0 - 4.0 * xi + 3.0 * xi * xi) * one,
            6.0 * (xi - xi * xi) / lengths,
            (3.0 * xi * xi - 2.0 * xi) * one,
        ]
    )
    second = np.array(
        [
            (12.0 * xi - 6.0) / lengths**2,
            (6.0 * xi - 4.0) / lengths,
            (6.0 - 12.0 * xi) / lengths**2,
            (6.0 * xi - 2.0) / lengths,
        ]
    )
    return first, second


def _centerline_derivatives(nodes, lengths, xi):
    """r' and r'' of every element at ``xi`` from nodal coordinates of shape (nodes, 6)."""
    first, second = _shape_derivatives(xi, lengths)
    vectors = (nodes[:-1, :3], nodes[:-1, 3:], nodes[1:, :3], nodes[1:, 3:])
    tangent = sum(weight[:, None] * vector for weight, vector in zip(first, vectors, strict=True))
    curvature = sum(weight[:, None] * vector for weight, vector in zip(second, vectors, strict=True))
    return tangent, curvature


def _dot(a, b):
    """Row-wise product without conjugation, so that it stays analytic for complex arguments."""
    return (a * b).sum(axis=-1)


def _mass_matrix(thread):
    """The chain's consistent mass over (position, slope) of every node; it acts on the three axes alike."""
    diagonal, coupling = thread.mass_diagonal.numpy(), thread.mass_coupling.numpy()
    mass = np.zeros((2 * thread.node_count, 2 * thread.node_count))
    for node, block in enumerate(diagonal):
        mass[2 * node : 2 * node + 2, 2 * node : 2 * node + 2] = block
    for node, block in enumerate(coupling):
        mass[2 * node : 2 * node + 2, 2 * node + 2 : 2 * node + 4] = block
        mass[2 * node + 2 : 2 * node + 4, 2 * node : 2 * node + 2] = block.T
    return mass


def _step_potential(q, thread, q_start, qd_start, dt, damping, gravity, floor=None):
    """The step potential of docs/method.md at nodal coordinates ``q`` of shape (nodes, 6), real or complex."""
    rest = _nodes(thread.rest_q)
    lengths = thread.lengths.numpy()
    axial, bending = thread.axial_stiffness.numpy(), thread.bending_stiffness.numpy()
    mass = _mass_matrix(thread)
    total = 0.0
    for axis in range(3):
        columns = [axis, axis + 3]
        value, start = q[:, columns].reshape(-1), q_start[:, columns].reshape(-1)
        inertial = value - (start + dt * qd_start[:, columns].reshape(-1))
        moved = value - start
        load = np.zeros(len(value))
        load[0::2] = gravity[axis]
        total += inertial @ mass @ inertial / (2.0 * dt * dt) + damping / (2.0 * dt) * (moved @ mass @ moved)
        total -= value @ mass @ load
    for point, weight in zip(GAUSS_POINTS, GAUSS_WEIGHTS, strict=True):
        xi = 0.5 * (point + 1.0)
        tangent, curvature = _centerline_derivatives(q, lengths, xi)
        rest_tangent, rest_curvature = _centerline_derivatives(rest, lengths, xi)
        rest_metric = _dot(rest_tangent, rest_tangent)
        strain = 0.5 * (_dot(tangent, tangent) / rest_metric - 1.0)
        cross, rest_cross = np.cross(tangent, curvature), np.cross(rest_tangent, rest_curvature)
        kappa = np.sqrt(_dot(cross, cross) / _dot(tangent, tangent) ** 3)
        rest_kappa = np.sqrt(_dot(rest_cross, rest_cross) / rest_metric**3)
        density = 0.5 * axial * strain**2 + 0.5 * bending * (kappa - rest_kappa) ** 2
        total += (0.5 * weight * lengths * np.sqrt(rest_metric) * density).sum()
    if floor is not None:
        depth = floor["contact_distance"] - q[1:, 2]
        penetration = 0.5 * (depth + np.sqrt(depth * depth + floor["contact_smoothing"] ** 2))
        total += (0.5 * floor["contact_stiffness"] * penetration**2).sum()
    return total


def _gradient(function, point):
    """Gradient of an analytic ``function`` at a real ``point`` by complex steps.

    The derivative along a coordinate is the imaginary part of f(x + i h e) / h. No difference is formed, so a
    tiny h makes it exact to rounding.
    """
    step = 1.0e-30
    result = np.zeros(point.shape)
    for index in np.ndindex(point.shape):
        probe = point.astype(complex)
        probe[index] += 1j * step
        result[index] = function(probe).imag / step
    return result


def _quat(vector):
    """Quaternion (x, y, z, w) of a rotation vector."""
    angle = np.linalg.norm(vector)
    if angle == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.append(math.sin(0.5 * angle) / angle * np.asarray(vector), math.cos(0.5 * angle))


def _quat_multiply(a, b):
    return np.append(a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - a[:3] @ b[:3])


def _rotate(quat, vector):
    return vector + 2.0 * np.cross(quat[:3], np.cross(quat[:3], vector) + quat[3] * vector)


def _rotation_vector(quat):
    norm = np.linalg.norm(quat[:3])
    return np.zeros(3) if norm == 0.0 else 2.0 * math.atan2(norm, quat[3]) / norm * quat[:3]


def _turn(vector, target):
    """``target`` turned by a rotation vector that may be complex."""
    angle = np.sqrt(vector @ vector)
    axis = vector / angle
    return (
        target * np.cos(angle) + np.cross(axis, target) * np.sin(angle) + axis * (axis @ target) * (1.0 - np.cos(angle))
    )


class _Body:
    """A rigid body stepped by hand with semi-implicit Euler; its frame is at its centre, on its principal axes.

    A support, if given, is a spring and a damper between the body and the pose it starts in, integrated
    implicitly. Within a step a wrench W then moves the body by (M / dt² + c / dt + k)^-1 W, and a free body by
    dt² M^-1 W.
    """

    def __init__(self, rotation=(0.0, 0.0, 0.0), damping=(0.0, 0.0), stiffness=(0.0, 0.0)):
        self.position, self.velocity, self.spin = np.zeros(3), np.zeros(3), np.zeros(3)
        self.rotation = self.rest = _quat(rotation)
        self.damping, self.stiffness = damping, stiffness

    def world_inertia(self):
        turn = np.column_stack([_rotate(self.rotation, axis) for axis in np.eye(3)])
        return turn @ np.diag(BODY_INERTIA) @ turn.T

    def point(self, local):
        return self.position + _rotate(self.rotation, np.asarray(local))

    def pose(self):
        """The body's transform as Newton stores it, in single precision."""
        return np.concatenate((self.position, self.rotation)).astype(np.float32)[None]

    def after(self, force, torque, dt):
        """The body one step on under a world force [N] and a world torque about its centre [N m]."""
        (linear_damping, angular_damping), (linear_stiffness, angular_stiffness) = self.damping, self.stiffness
        result = _Body(damping=self.damping, stiffness=self.stiffness)
        result.rest = self.rest
        result.velocity = (BODY_MASS * self.velocity + dt * (force - linear_stiffness * self.position)) / (
            BODY_MASS + dt * linear_damping + dt * dt * linear_stiffness
        )
        inertia = self.world_inertia()
        deflection = _rotation_vector(_quat_multiply(self.rotation, self.rest * (-1.0, -1.0, -1.0, 1.0)))
        result.spin = np.linalg.solve(
            inertia + (dt * angular_damping + dt * dt * angular_stiffness) * np.eye(3),
            inertia @ self.spin + dt * (torque - angular_stiffness * deflection),
        )
        result.position = self.position + dt * result.velocity
        result.rotation = _quat_multiply(_quat(dt * result.spin), self.rotation)
        return result


def _body_model():
    """A Newton model with the one free body that :class:`_Body` steps."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body(mass=BODY_MASS, inertia=wp.mat33(*np.diag(BODY_INERTIA).reshape(-1)))
    return builder.finalize(device=DEVICE)


def _dragged_thread():
    """A scene in which every array of ``SolverThread.state`` carries the motion.

    A rigid head draws a thread over a floor with friction, through an eyelet that rubs on it, against a knot at
    its far end, while a pin lifts one node. Returns the solver and a function that makes step ``index``.
    """
    elements = 20
    speed = 0.004  # [m/s], a slow hand
    solver = _solver(
        _line(elements, start=(0.0, 0.0, RADIUS)),
        mass_damping=DAMPING,
        tolerance=0.1 * NODE_WEIGHT,
        # The examples' budget: the steps do not converge, so the replay depends on every first iterate.
        max_iterations=2,
        friction=0.2,
        max_eyelets=2,
        # As stiff as one element of the thread along its axis.
        eyelet_stiffness=MODULUS * math.pi * RADIUS**2 / ELEMENT,
        **FLOOR,
    )
    nodes = _nodes(solver.q).copy()
    # Eyelet 0 lies half an element beside the middle of element 10 and lets the thread slide; eyelet 1 is a knot
    # on the last node.
    beside = 0.5 * (nodes[10, :3] + nodes[11, :3]) + (0.0, 0.5 * ELEMENT, 0.0)
    solver.eyelet_target.assign(np.array([beside, nodes[-1, :3]]))
    solver.eyelet_element.assign(np.array([10, elements - 1], dtype=np.int32))
    solver.eyelet_xi.assign(np.array([0.5, 1.0]))
    solver.eyelet_held.assign(np.array([0, 1], dtype=np.int32))
    # The head has the needle's inertia and its centre 3 mm behind node 0; its predicted pose retreats.
    arm = np.array([0.003, 0.0, 0.0])
    center = nodes[0, :3] - arm
    solver.rigid_head = True
    solver.head_arm.assign(arm[None])
    solver.head_slope.assign(nodes[0, 3:][None])
    solver.head_inertia.assign((np.diag(np.repeat((BODY_MASS, BODY_INERTIA[0]), 3)) / DT**2)[None])
    # A pin on the height of node 5.
    pin = 6 * 5 + 2
    solver.set_prescribed([pin])

    def step(index):
        solver.head_center.assign((center - (speed * DT * (index + 1), 0.0, 0.0))[None])
        # The pin rises from wherever it is, as a dragged pin does, so boundary_q belongs to the state.
        boundary = solver.boundary_q.numpy()
        boundary[pin] += speed * DT
        solver.boundary_q.assign(boundary)
        solver.step(None, None, None, None, DT)

    return solver, step


def _lifted_chains(device, chains, steps, head=False, **options):
    """Chains on a floor, far ends pinned, whose first nodes follow different paths.

    Node 0 is prescribed or, with ``head``, carried by a rigid head whose predicted pose follows the path.
    Returns the coordinates after every step, shape (steps, chains, nodes, 6), and the updates of every step.
    """
    elements = 20
    line = _line(elements, start=(0.0, 0.0, RADIUS))
    solver = _solver(
        line,
        device=device,
        n_envs=len(chains),
        fixed_dofs=[6 * elements + axis for axis in range(3)],
        mass_damping=DAMPING,
        tolerance=0.1 * NODE_WEIGHT,
        **FLOOR,
        **options,
    )
    # Within half a second node 0 of chain 0 rises by 12 mm and closes in by 6 mm, node 0 of chain 1 moves 9 mm
    # over the floor, and chain 2 is left alone [m].
    paths = np.array([[0.006, 0.0, 0.012], [0.004, 0.008, 0.0], [0.0, 0.0, 0.0]])[list(chains)]
    boundary = _nodes(solver.boundary_q).copy()
    # A head has the needle's inertia and its centre 3 mm behind node 0.
    arm = np.array([0.003, 0.0, 0.0])
    if head:
        solver.rigid_head = True
        solver.head_arm.assign(np.tile(arm, (len(chains), 1)))
        solver.head_slope.assign(np.tile(boundary[0, 3:], (len(chains), 1)))
        inertia = np.diag(np.repeat((BODY_MASS, BODY_INERTIA[0]), 3)) / DT**2
        solver.head_inertia.assign(np.tile(inertia, (len(chains), 1, 1)))
    history, updates = [], []
    for step in range(steps):
        moved = line[0] + min(1.0, (step + 1) / 30.0) * paths
        if head:
            solver.head_center.assign(moved - arm)
        else:
            boundary[:: elements + 1, :3] = moved
            solver.boundary_q.assign(boundary.reshape(-1))
        solver.step(None, None, None, None, DT)
        # On a CPU numpy() is a view of the solver's array, which the next step overwrites.
        history.append(solver.q.numpy().reshape(len(chains), elements + 1, 6).copy())
        updates.append(solver.iterations.numpy().copy())
    return np.array(history), np.array(updates)


class ThreadModelTests(unittest.TestCase):
    def test_straight_chain_has_exact_lengths_stiffnesses_and_mass(self):
        """A straight chain with unequal elements has the textbook section stiffnesses and Hermite mass.

        Fails if a length, a slope, E A, E I or an entry of a mass block is wrong.
        """
        direction = np.array([2.0, -1.0, 2.0]) / 3.0
        lengths = np.array([0.001, 0.002, 0.004])
        positions = np.array([0.01, 0.02, 0.03]) + np.concatenate(([0.0], np.cumsum(lengths)))[:, None] * direction
        thread = ThreadModel.create(positions, young_modulus=MODULUS, radius=RADIUS, density=DENSITY, device=DEVICE)

        self.assertEqual((thread.node_count, thread.element_count, thread.dof_count), (4, 3, 24))
        self.assertEqual(thread.device, wp.get_device(DEVICE))
        self.assertEqual(thread.rest_q.device, thread.device)
        # Everything below is exact arithmetic up to rounding: five Gauss points integrate the degree-six
        # products of cubic shape functions exactly.
        rounding = 1.0e-12
        np.testing.assert_allclose(thread.lengths.numpy(), lengths, rtol=rounding)
        np.testing.assert_allclose(
            _nodes(thread.rest_q), np.column_stack((positions, np.tile(direction, (4, 1)))), rtol=rounding
        )
        area = math.pi * RADIUS**2
        np.testing.assert_allclose(thread.axial_stiffness.numpy(), np.full(3, MODULUS * area), rtol=rounding)
        np.testing.assert_allclose(
            thread.bending_stiffness.numpy(), np.full(3, MODULUS * math.pi * RADIUS**4 / 4.0), rtol=rounding
        )
        diagonal, coupling = np.zeros((4, 2, 2)), np.zeros((3, 2, 2))
        for element, length in enumerate(lengths):
            # Consistent mass of a cubic Hermite beam element over [node a, slope a, node b, slope b].
            block = (DENSITY * area * length / 420.0) * np.array(
                [
                    [156.0, 22.0 * length, 54.0, -13.0 * length],
                    [22.0 * length, 4.0 * length**2, 13.0 * length, -3.0 * length**2],
                    [54.0, 13.0 * length, 156.0, -22.0 * length],
                    [-13.0 * length, -3.0 * length**2, -22.0 * length, 4.0 * length**2],
                ]
            )
            diagonal[element] += block[:2, :2]
            diagonal[element + 1] += block[2:, 2:]
            coupling[element] = block[:2, 2:]
        np.testing.assert_allclose(thread.mass_diagonal.numpy(), diagonal, rtol=rounding)
        np.testing.assert_allclose(thread.mass_coupling.numpy(), coupling, rtol=rounding)

    def test_curved_chain_has_bisecting_slopes_and_mass_along_its_curve(self):
        """On an arc the inner slopes are the arc's tangents, and the chain weighs as much as its reference curve.

        Fails if an inner slope is not the unit bisector of its chords, an end slope is not its chord, or the mass
        is integrated along the chords.
        """
        elements, radius = 6, 0.01
        angles = np.linspace(0.0, 0.5 * math.pi, elements + 1)
        positions = radius * np.column_stack((np.sin(angles), 1.0 - np.cos(angles), np.zeros(elements + 1)))
        thread = ThreadModel.create(positions, young_modulus=MODULUS, radius=RADIUS, density=DENSITY, device=DEVICE)
        rest = _nodes(thread.rest_q)

        # Equal chords of a circle: their bisector at a node is the circle's tangent there.
        tangents = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(elements + 1)))
        np.testing.assert_allclose(rest[1:-1, 3:], tangents[1:-1], atol=1.0e-12)
        chords = np.diff(positions, axis=0)
        chord = np.linalg.norm(chords[0])
        np.testing.assert_allclose(rest[[0, -1], 3:], chords[[0, -1]] / chord, atol=1.0e-12)

        # The position shape functions sum to one, so the position entries of the mass sum to the mass of the
        # reference curve. Its length is integrated here with 20 Gauss points per element; the model's five
        # agree to 1e-9 because the speed along the curve is nearly a polynomial (measured 9e-11).
        points, weights = np.polynomial.legendre.leggauss(20)
        length = 0.0
        for point, weight in zip(points, weights, strict=True):
            tangent, _ = _centerline_derivatives(rest, thread.lengths.numpy(), 0.5 * (point + 1.0))
            length += 0.5 * weight * chord * np.linalg.norm(tangent, axis=1).sum()
        mass = _mass_matrix(thread)[0::2, 0::2].sum()
        line_density = DENSITY * math.pi * RADIUS**2
        self.assertAlmostEqual(mass / (line_density * length), 1.0, delta=1.0e-9)
        # An arc is longer than its chord by the fraction phi² / 24 for the angle phi it turns, here 2.9e-3. The
        # two end elements, whose outer slope lies along the chord, show less of it (measured 2.3e-3 in the
        # mean); the mass must show at least half.
        turn = 0.5 * math.pi / elements
        self.assertGreater(mass / (line_density * elements * chord), 1.0 + turn**2 / 48.0)

    def test_create_rejects_bad_input(self):
        """Fails if a chain that cannot be built, or a nonpositive or non-finite material value, is accepted."""
        good = _line(3)
        chains = {
            "two nodes": good[:2],
            "points in a plane": good[:, :2],
            "not finite": np.where(np.eye(4, 3) > 0, np.nan, good),
            "coincident neighbours": good[[0, 1, 1, 2]],
            "reversal at a node": good[[0, 1, 0, 1]],
        }
        for name, positions in chains.items():
            with self.subTest(name), self.assertRaises(ValueError):
                ThreadModel.create(positions, young_modulus=MODULUS, radius=RADIUS, density=DENSITY, device=DEVICE)
        for name in ("young_modulus", "radius", "density"):
            for value in (0.0, -1.0, math.inf, math.nan):
                material = {"young_modulus": MODULUS, "radius": RADIUS, "density": DENSITY, name: value}
                with self.subTest(name, value=value), self.assertRaises(ValueError):
                    ThreadModel.create(good, device=DEVICE, **material)


class SolverThreadTests(unittest.TestCase):
    def test_solver_rejects_bad_options(self):
        """Fails if the solver accepts an option outside its documented range."""
        elements = 4
        bad = {
            "no chain": {"n_envs": 0},
            "no update": {"max_iterations": 0},
            "zero tolerance": {"tolerance": 0.0},
            "negative damping": {"mass_damping": -1.0},
            "negative contact stiffness": {"contact_stiffness": -1.0},
            "zero contact smoothing": {"contact_smoothing": 0.0},
            "negative friction": {"friction": -0.1},
            "negative eyelet count": {"max_eyelets": -1},
            "negative eyelet stiffness": {"eyelet_stiffness": -1.0},
            "coordinate beyond the chain": {"fixed_dofs": [6 * (elements + 1)]},
            "gravity with two components": {"gravity": (0.0, -GRAVITY)},
            "self-contact without a contact distance": {"self_contact": True},
            "zero obstacle distance": {"obstacle_distance": 0.0},
            "multiples without one": {"multiples": (2.0, 0.5, 0.0)},
            "multiples without zero": {"multiples": (2.0, 1.0, 0.5)},
            "negative multiple": {"multiples": (1.0, 0.0, -0.5)},
        }
        for name, options in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                _solver(_line(elements), **options)

    def test_hanging_chain_comes_to_rest_on_its_support(self):
        """A level chain let go, held at node 0, comes to rest and its support then carries its weight.

        Fails if the chain keeps moving, or the residual of node 0 is not the chain's weight, upwards.
        """
        elements = 20
        length = elements * ELEMENT
        # Critical damping of the hanging chain's slowest swing, 1.2 sqrt(g / L).
        damping = 2.0 * 1.2 * math.sqrt(GRAVITY / length)
        tolerance = 1.0e-4 * NODE_WEIGHT
        solver = _solver(_line(elements), mass_damping=damping, tolerance=tolerance)
        # The swing decays as exp(-1.2 sqrt(g / L) t): 1.5 s leave 6e-13 of it, and the solver's tolerance
        # decides what speed is left.
        _advance(solver, 90)

        # At rest: the drag on a node, c m v, is below the solver's force tolerance, 2.6e-5 m/s. Measured
        # 1.7e-6 m/s.
        speed = np.abs(_nodes(solver.qd)[:, :3]).max()
        self.assertLess(speed, tolerance / (damping * NODE_WEIGHT / GRAVITY))
        # The elastic forces cancel in the sum over the chain, so the support's force is the weight less the
        # residuals of the free nodes, each below the tolerance, and less what is left of drag and inertia,
        # which the rest speed puts below the same bound. Measured: 0.03 of this bound.
        support = solver.residual.numpy()[0, :3]
        weight = LINE_WEIGHT * length
        np.testing.assert_allclose(support, (0.0, 0.0, weight), rtol=0.0, atol=2.0 * elements * tolerance)
        # It hangs. The thread leaves its support level and turns down within about one bending length
        # sqrt(E I / W) = 5 mm under its own weight W, so its tip is nearly its length below (measured 0.92 L).
        self.assertLess(_nodes(solver.q)[-1, 2], -(length - 2.0 * math.sqrt(BENDING / weight)))

    def _check_gradient(self, positions, head):
        """Make one update from a strained state and compare the residual with the gradient of the step potential."""
        elements, damping, gravity = len(positions) - 1, 2.0, (0.3, -0.2, -GRAVITY)
        generator = np.random.default_rng(7)
        pinned = 6 * elements + 1
        # One update from a strained start leaves a residual far from zero.
        solver = _solver(
            positions, gravity=gravity, mass_damping=damping, max_iterations=1, fixed_dofs=[pinned], **FLOOR
        )
        rest = _nodes(solver.thread.rest_q)
        # Strains of a tenth, heights within the contact's onset, and nodes that move within the step.
        start = rest + generator.normal(size=rest.shape) * np.repeat((0.1 * ELEMENT, 0.1), 3)
        start[:, 2] = RADIUS + 3.0 * SMOOTHING * generator.normal(size=elements + 1)
        rate = generator.normal(size=rest.shape) * np.repeat((0.01, 5.0), 3)
        boundary = start + generator.normal(size=rest.shape) * np.repeat((0.05 * ELEMENT, 0.05), 3)
        solver.q.assign(start.reshape(-1))
        solver.qd.assign(rate.reshape(-1))
        solver.boundary_q.assign(boundary.reshape(-1))
        if head:
            # A head with little rotary inertia, 3 mm from node 0, that starts the step turned by a third of a
            # radian: the thread then leaves it turned, and the rotation's tangent operator counts.
            center = start[0, :3] + (0.001, 0.002, -0.002)
            arm, slope = start[0, :3] - center, start[0, 3:]
            inertia = np.diag(np.repeat((BODY_MASS, 1.0e-3 * BODY_INERTIA[0]), 3)) / DT**2
            solver.rigid_head = True
            solver.head_center.assign(center[None])
            solver.head_arm.assign(arm[None])
            solver.head_slope.assign(slope[None])
            solver.head_inertia.assign(inertia[None])
            solver.head_motion.assign(np.concatenate((1.0e-4 * generator.normal(size=3), (0.2, -0.2, 0.15)))[None])
        solver.step(None, None, None, None, DT)
        q, residual = _nodes(solver.q), solver.residual.numpy()
        self.assertEqual(solver.iterations.numpy()[0], 1)

        def potential(coordinates):
            return _step_potential(coordinates, solver.thread, start, rate, DT, damping, gravity, FLOOR)

        expected = _gradient(potential, q)
        # Both sides are sums in double precision whose largest terms are the axial forces of the strained
        # elements; they agree to 1e-14 of the largest entry (measured). A wrong or missing term is a whole entry.
        tolerance = 1.0e-10 * np.abs(expected).max()
        # The prescribed coordinate has no residual although the potential has a slope along it.
        self.assertEqual(residual[-1, 1], 0.0)
        self.assertGreater(abs(expected[-1, 1]), 1.0e3 * tolerance)
        expected[-1, 1] = 0.0
        np.testing.assert_allclose(residual[1:], expected[1:], rtol=0.0, atol=tolerance)
        if not head:
            # Node 0 follows boundary_q; its rows are the force of its support.
            np.testing.assert_array_equal(q[0], boundary[0])
            np.testing.assert_allclose(residual[0], expected[0], rtol=0.0, atol=tolerance)
            return

        motion = solver.head_motion.numpy()[0]
        self.assertGreater(np.linalg.norm(motion[3:]), 0.05)

        def head_node(moved):
            return np.concatenate((center + moved[:3] + _turn(moved[3:], arm), _turn(moved[3:], slope)))

        def head_potential(moved):
            coordinates = q.astype(complex)
            coordinates[0] = head_node(moved)
            return potential(coordinates) + 0.5 * moved @ inertia @ moved

        # Node 0 sits on the head, to the rounding of a slope of length one.
        np.testing.assert_allclose(q[0], head_node(motion), rtol=0.0, atol=1.0e-14)
        # The head's rows are a force and a torque, the derivatives by its translation and its rotation vector.
        np.testing.assert_allclose(residual[0], _gradient(head_potential, motion), rtol=0.0, atol=tolerance)

    def test_residual_is_gradient_of_step_potential(self):
        """The residual of an unconverged step is the gradient of the documented step potential.

        Fails if a force term, the mass, the damping, the weight, the contact or a rigid head's rows differ from
        the derivative of the potential, or a prescribed coordinate has a residual.
        """
        elements = 6
        # A straight reference and one bent through 1.2 rad, which has the (kappa - kappa_0)² bending energy.
        arc = np.linspace(0.0, 1.2, elements + 1)
        bend = elements * ELEMENT / arc[-1]
        chains = {
            "straight": _line(elements, start=(0.0, 0.0, RADIUS)),
            "curved": np.column_stack((bend * np.sin(arc), bend * (1.0 - np.cos(arc)), np.full(elements + 1, RADIUS))),
        }
        for name, positions in chains.items():
            for head in (False, True):
                with self.subTest(name, rigid_head=head):
                    self._check_gradient(positions, head)

    def test_level_slack_thread_starts_to_fall(self):
        """A level slack thread held at one end falls with the default multiples and freezes without the small ones.

        Fails if the default line search keeps the thread level, or if multiples down to a quarter already move
        it, which would mean this scene no longer needs the small ones.
        """
        elements, dt, steps = 20, 1.0 / 20.0, 10
        length = elements * ELEMENT
        # One backward Euler step of free fall is g dt² = 25 mm, most of the thread's 40 mm. The Newton update of
        # that load across the thread stretches it so much that a quarter of it still raises the potential.
        drop = {}
        for name, options in (("default", {}), ("coarse", {"multiples": (4.0, 2.0, 1.0, 0.5, 0.25, 0.0)})):
            solver = _solver(_line(elements), **options)
            _advance(solver, steps, dt)
            drop[name] = -_nodes(solver.q)[-1, 2]
        # Frozen: only the zero multiple does not raise the potential, and it leaves every coordinate as it is.
        self.assertEqual(drop["coarse"], 0.0)
        # A free fall covers 1.2 m in this half second, so the thread hangs by then: its tip is its length down
        # but for the turn at the support, about one bending length sqrt(E I / W) = 5 mm (measured 0.92 L).
        self.assertGreater(drop["default"], length - 2.0 * math.sqrt(BENDING / (LINE_WEIGHT * length)))

    def test_set_prescribed_holds_and_frees_coordinates(self):
        """A prescribed coordinate follows boundary_q against the thread; freed, it obeys the forces and keeps moving.

        Fails if a prescribed coordinate leaves its boundary value, a freed one stays held or loses its velocity,
        the list of prescribed coordinates is wrong, or node 0 can be freed.
        """
        elements = 10
        length = elements * ELEMENT
        end = [6 * elements + axis for axis in range(3)]
        solver = _solver(_line(elements), gravity=(0.0, 0.0, 0.0))
        self.assertEqual(solver.fixed_dofs.tolist(), list(range(6)))
        solver.set_prescribed(end)
        self.assertEqual(solver.fixed_dofs.tolist(), list(range(6)) + end)
        for dofs in ([0], [5, 6], [6 * (elements + 1)]):
            with self.subTest(dofs=dofs), self.assertRaises(ValueError):
                solver.set_prescribed(dofs)

        # The end is pulled out by 1 % of the chain's length and held there.
        pull = 0.01 * length
        boundary = _nodes(solver.boundary_q).copy()
        boundary[-1, 0] += pull
        solver.boundary_q.assign(boundary.reshape(-1))
        _advance(solver, 10)
        # The solver copies boundary_q into a prescribed coordinate and gives it a zero update: equal bit for bit.
        np.testing.assert_array_equal(_nodes(solver.q)[-1, :3], boundary[-1, :3])
        # Freed, the end is drawn back by the stretched chain. An axial wave crosses the chain in
        # L sqrt(rho / E) = 0.5 ms, far below the step, so backward Euler damps the recoil within a few steps
        # (measured: 2e-6 of the pull left after 10 steps).
        solver.set_prescribed(end, False)
        self.assertEqual(solver.fixed_dofs.tolist(), list(range(6)))
        _advance(solver, 10)
        self.assertLess(abs(_nodes(solver.q)[-1, 0] - length), 1.0e-3 * pull)

        # A freed coordinate keeps its velocity. The whole chain drifts across itself at one speed, its end
        # prescribed to do the same; freed, the end must drift on. Had it lost its velocity it would be
        # speed * dt = 0.5 mm behind after the next step. No force acts, so the drift is exact to rounding.
        speed = 0.03  # [m/s]
        solver = _solver(_line(elements), gravity=(0.0, 0.0, 0.0))
        solver.set_prescribed(end)
        rest = _nodes(solver.q).copy()
        rate = np.zeros_like(rest)
        rate[:, 1] = speed
        solver.qd.assign(rate.reshape(-1))
        for step in (1, 2, 3):
            if step == 3:
                solver.set_prescribed(end, False)
            drifted = rest.copy()
            drifted[:, 1] += step * DT * speed
            solver.boundary_q.assign(drifted.reshape(-1))
            solver.step(None, None, None, None, DT)
            np.testing.assert_allclose(_nodes(solver.q)[:, :3], drifted[:, :3], rtol=0.0, atol=1.0e-6 * speed * DT)

    def test_state_restores_a_step(self):
        """Steps made from a restored copy of ``state`` repeat the steps made from the original.

        Fails if an array that carries the solver from step to step is missing from ``state``.
        """
        solver, step = _dragged_thread()
        outputs = (*solver.state, solver.contact_force, solver.eyelet_force)

        def replay():
            for index in range(30, 35):
                step(index)
            return [array.numpy().copy() for array in outputs]

        for index in range(30):
            step(index)
        saved = [wp.clone(array) for array in solver.state]
        expected = replay()
        for array, copy in zip(solver.state, saved, strict=True):
            wp.copy(array, copy)
        replayed = replay()

        # The scene is live: the thread has slid through eyelet 0, which started at the middle of its element
        # (measured 0.14 elements), and has drawn the head back by more than a tenth of its 2.3 mm retreat
        # (measured 2.2 mm).
        self.assertGreater(solver.eyelet_xi.numpy()[0], 0.55)
        self.assertGreater(np.linalg.norm(solver.head_motion.numpy()[0, :3]), 2.3e-4)
        if not wp.get_device(DEVICE).is_cuda:
            # A CPU kernel runs in one thread: the replay is the same arithmetic, bit for bit.
            for first, second in zip(expected, replayed, strict=True):
                np.testing.assert_array_equal(first, second)
        else:
            # A GPU adds a potential's terms in no fixed order, so a replay may differ by rounding.
            scale = np.tile(np.repeat((1.0, ELEMENT), 3), solver.thread.node_count)
            np.testing.assert_allclose(expected[0] * scale, replayed[0] * scale, rtol=0.0, atol=_rounding(5, 2))

    def test_chains_of_a_batch_step_independently(self):
        """Each chain of a batch moves as it does alone, with its own number of updates.

        Fails if a chain's motion or update count depends on the other chains of its batch, with node 0
        prescribed or on a rigid head.
        """
        steps = 30
        scale = np.repeat((1.0, ELEMENT), 3)
        for head in (False, True):
            batch, updates = _lifted_chains(DEVICE, (0, 1, 2), steps, head=head, max_iterations=2)
            if not head:
                # The chains differ: the lifted ones use their budget while the one at rest, which a head
                # would let sink a little, never needs an update and stays where it lies.
                self.assertEqual(updates[:, 0].max(), 2)
                self.assertGreater(updates[:, 1].sum(), 0)
                self.assertEqual(updates[:, 2].max(), 0)
                np.testing.assert_array_equal(batch[-1, 2, :, :3], _line(20, start=(0.0, 0.0, RADIUS)))
            for chain in range(3):
                with self.subTest(rigid_head=head, chain=chain):
                    alone, alone_updates = _lifted_chains(DEVICE, (chain,), steps, head=head, max_iterations=2)
                    np.testing.assert_array_equal(updates[:, chain], alone_updates[:, 0])
                    # Bit for bit on a CPU; on a GPU up to the rounding of sums added in no fixed order.
                    tolerance = _rounding(steps, 2) if wp.get_device(DEVICE).is_cuda else 0.0
                    np.testing.assert_allclose(batch[:, chain] * scale, alone[:, 0] * scale, rtol=0.0, atol=tolerance)

    @unittest.skipUnless(DEVICE.startswith("cuda") and wp.is_cuda_available(), "Set NEWTON_TEST_DEVICE=cuda:0 for CUDA")
    def test_cuda_and_cpu_agree(self):
        """The cyclic reduction on CUDA and the serial elimination on the CPU give the same motion.

        Fails if the two chain solves differ by more than rounding in any update of a short run, with node 0
        prescribed or on a rigid head.
        """
        steps = 30
        # Positions [m], and slopes scaled by the element length.
        scale = np.repeat((1.0, ELEMENT), 3)
        for head in (False, True):
            with self.subTest(rigid_head=head):
                # Two updates per step do not converge, so every update's solve shows in the coordinates.
                cuda, cuda_updates = _lifted_chains(DEVICE, (0, 1, 2), steps, head=head, max_iterations=2)
                cpu, cpu_updates = _lifted_chains("cpu", (0, 1, 2), steps, head=head, max_iterations=2)
                np.testing.assert_array_equal(cuda_updates, cpu_updates)
                # Measured 9e-15 m.
                np.testing.assert_allclose(cuda * scale, cpu * scale, rtol=0.0, atol=_rounding(steps, 2))

    def test_floor_carries_the_threads_weight(self):
        """A thread let down on a level floor comes to rest with its weight on the floor.

        Fails if the floor's forces and the support at node 0 do not add up to the thread's weight, a node away
        from the ends does not carry one element's weight at the contact distance, or node 0 has a contact.
        """
        elements = 20
        tolerance = 1.0e-4 * NODE_WEIGHT
        # The thread starts level, one smoothing width above the height at which it rests.
        solver = _solver(
            _line(elements, start=(0.0, 0.0, RADIUS + SMOOTHING)), mass_damping=DAMPING, tolerance=tolerance, **FLOOR
        )
        boundary = _nodes(solver.boundary_q).copy()
        boundary[0, 2] = RADIUS
        solver.boundary_q.assign(boundary.reshape(-1))
        _advance(solver, 60)

        # At rest as in the hanging chain: the drag on a node is below the force tolerance.
        speed = np.abs(_nodes(solver.qd)[:, :3]).max()
        self.assertLess(speed, tolerance / (DAMPING * NODE_WEIGHT / GRAVITY))
        floor = solver.contact_force.numpy()
        support = solver.residual.numpy()[0, 2]
        # The floor is frictionless and node 0 is carried by its support alone.
        np.testing.assert_array_equal(floor[:, :2], 0.0)
        np.testing.assert_array_equal(floor[0], 0.0)
        # Vertical balance: the free nodes' residuals, each below the tolerance, are all that is missing.
        weight = LINE_WEIGHT * elements * ELEMENT
        self.assertAlmostEqual(floor[:, 2].sum() + support, weight, delta=elements * tolerance)
        # Node 0 is held at the height of the others, so its support carries its own half element and the floor
        # the other 19.5 elements.
        self.assertAlmostEqual(support, 0.5 * NODE_WEIGHT, delta=elements * tolerance)
        # The last node carries half an element and floats higher; its lift halves from node to node (measured)
        # and is below the tolerance in the first quarter of the thread. There every node carries one element,
        # at the contact distance: the penalty's stiffness there is k / 2.
        inner = slice(1, elements // 4 + 1)
        np.testing.assert_allclose(floor[inner, 2], NODE_WEIGHT, rtol=0.0, atol=2.0 * tolerance)
        give = 2.0 * tolerance / (0.5 * FLOOR["contact_stiffness"])
        np.testing.assert_allclose(_nodes(solver.q)[inner, 2], RADIUS, rtol=0.0, atol=give)


class ThreadAttachmentTests(unittest.TestCase):
    def test_attachment_rejects_bad_input(self):
        """Fails if an attachment accepts a wrong body list, a zero tangent or a negative support."""
        model = _body_model()
        solver = _solver(_line(4), model=model)
        for name, bodies, tangent in (
            ("two bodies for one chain", [0, 0], (1.0, 0.0, 0.0)),
            ("no such body", [1], (1.0, 0.0, 0.0)),
            ("negative body", [-1], (1.0, 0.0, 0.0)),
            ("zero tangent", [0], (0.0, 0.0, 0.0)),
        ):
            with self.subTest(name), self.assertRaises(ValueError):
                ThreadAttachment(solver, model, bodies, (0.0, 0.0, 0.0), tangent)
        attachment = ThreadAttachment(solver, model, [0], (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        for name, support in (
            ("negative damping", ((-1.0, 0.0), (0.0, 0.0))),
            ("infinite stiffness", ((0.0, 0.0), (math.inf, 0.0))),
        ):
            with self.subTest(name), self.assertRaises(ValueError):
                attachment.set_support(*support)

    def test_classic_mode_keeps_node_0_on_the_body(self):
        """A free body let go on a straight, pinned thread swings to rest with node 0 on its attachment point.

        Fails if node 0 or its slope leaves the body's point and direction, the wrench is not added to the body's
        other loads, the thread feeds energy to the body, or the wrench at rest is not the body's weight.
        """
        elements, dt, steps = 20, 1.0 / 240.0, 360
        length = elements * ELEMENT
        # A thread a hundred times stiffer than the soft default: its axial stiffness times dt² is 20 times the
        # body's mass, the case in which the body cannot take the thread's force one step late.
        modulus = 2.0e8
        self.assertGreater(modulus * math.pi * RADIUS**2 / length * dt**2, 10.0 * BODY_MASS)
        point, tangent = np.array([0.0, 0.001, 0.003]), np.array([1.0, 0.0, 0.0])
        # The body starts turned, so its inertia in world axes is not diagonal.
        body = _Body(rotation=(0.3, -0.2, 0.4))
        model = _body_model()
        line = body.point(point) + ELEMENT * np.arange(elements + 1)[:, None] * _rotate(body.rotation, tangent)
        tolerance = 1.0e-3 * NODE_WEIGHT
        solver = _solver(
            line,
            modulus=modulus,
            model=model,
            mass_damping=DAMPING,
            tolerance=tolerance,
            fixed_dofs=[6 * elements + axis for axis in range(3)],
        )
        attachment = ThreadAttachment(solver, model, [0], point, tangent)
        self.assertTrue(solver.rigid_head)
        state = model.state()
        weight = np.array([0.0, 0.0, -BODY_MASS * GRAVITY])
        # Critical drag for the pendulum that the body makes on the thread, sqrt(g / L). It is taken at the start
        # of the step, a load like any other that the body's own solver knows before the thread.
        drag = 2.0 * math.sqrt(GRAVITY / length)
        lever = np.linalg.norm(point)
        start_height = body.position[2]
        # The pose reaches the attachment in single precision, where the head's place is formed from it and the
        # turned lever: a few units in the last place of the body's distance from the origin, at most the
        # thread's length plus the lever (measured: one).
        rounding = 4.0 * float(np.spacing(np.float32(length + lever)))
        # Spread of the principal inertias: how far a turn of the body changes its inertia in world axes.
        spread = BODY_INERTIA.max() / BODY_INERTIA.min() - 1.0
        for step in range(steps):
            force = weight - drag * BODY_MASS * body.velocity
            torque = -drag * (body.world_inertia() @ body.spin)
            predicted = body.after(force, torque, dt)
            state.body_q.assign(predicted.pose())
            attachment.attach(state, dt)
            solver.step(None, None, None, None, dt)
            loads = np.concatenate((force, torque)).astype(np.float32)
            state.body_f.assign(loads[None])
            attachment.apply(state)
            wrench = attachment.wrench.numpy()[0]
            np.testing.assert_array_equal(state.body_f.numpy()[0], loads + wrench)
            body = body.after(force + wrench[:3], torque + wrench[3:], dt)

            # The head and the body both add the thread's turn theta to the step's own turn phi = dt omega,
            # but differ in second order: the two turns do not commute (theta phi / 2), and the body's solver
            # takes the inertia at the start of the step, not at the predicted pose (theta phi times the
            # spread of the principal inertias).
            turns = np.linalg.norm(solver.head_motion.numpy()[0, 3:]) * dt * np.linalg.norm(predicted.spin)
            second_order = (0.5 + spread) * turns
            node = _nodes(solver.q)[0]
            gap = np.linalg.norm(node[:3] - body.point(point))
            self.assertLess(gap, second_order * lever + rounding, msg=f"step {step}")
            # The slope is a unit vector: single precision rounds it to 1.2e-7.
            slope_gap = np.linalg.norm(node[3:] - _rotate(body.rotation, tangent))
            self.assertLess(slope_gap, second_order + 4.0 * float(np.finfo(np.float32).eps), msg=f"step {step}")
            # The implicit thread only takes energy out: the body is never faster than its fall allows. The
            # thread's own weight, a hundredth of the body's, can add what it loses in falling its length.
            kinetic = (
                0.5 * BODY_MASS * body.velocity @ body.velocity + 0.5 * body.spin @ body.world_inertia() @ body.spin
            )
            lost = BODY_MASS * GRAVITY * (start_height - body.position[2])
            self.assertLess(kinetic, lost + LINE_WEIGHT * length * length, msg=f"step {step}")

        # At rest the thread carries the body. The rounding of the pose moves the head, whose inertia m / dt²
        # answers with a force; the solve itself stops at its tolerance on every node. The body is at rest when
        # what is left of its speed changes its momentum by less than that force within a step.
        unbalanced = BODY_MASS / dt**2 * rounding + elements * tolerance
        self.assertLess(np.linalg.norm(body.velocity), unbalanced * dt / BODY_MASS)
        np.testing.assert_allclose(wrench[:3], -weight, rtol=0.0, atol=unbalanced)

    @staticmethod
    def _support(frequency):
        """Damping and stiffness, linear and angular, of a critically damped spring with sqrt(k / m) = ``frequency``."""
        inertia = BODY_INERTIA.max()
        return (2.0 * frequency * BODY_MASS, 2.0 * frequency * inertia), (
            frequency**2 * BODY_MASS,
            frequency**2 * inertia,
        )

    def _lagged(self, solver, point, tangent, frequency, steps, before_step=None):
        """Step a body on a spring and its thread in lagged mode; returns the body and the last two wrenches.

        In every step the head must have the impedance Z of the body on its support, and node 0 must be off the
        body's point by Z^-1 times the difference between the wrench the head solved and the one applied.
        """
        damping, stiffness = self._support(frequency)
        body = _Body(damping=damping, stiffness=stiffness)
        attachment = ThreadAttachment(solver, solver.model, [0], point, tangent, lagged=True)
        attachment.set_support(damping, stiffness)
        state = solver.model.state()
        linear = BODY_MASS / DT**2 + damping[0] / DT + stiffness[0]
        lever = np.linalg.norm(point)
        for step in range(steps):
            state.body_f.zero_()
            attachment.add_forces(state.body_f)
            applied = state.body_f.numpy()[0].astype(np.float64)
            body = body.after(applied[:3], applied[3:], DT)
            state.body_q.assign(body.pose())
            if before_step is not None:
                before_step(step)
            attachment.attach(state, DT)
            solver.step(None, None, None, None, DT)
            attachment.update()
            wrench = attachment.wrench.numpy()[0].astype(np.float64)

            angular = body.world_inertia() / DT**2 + (damping[1] / DT + stiffness[1]) * np.eye(3)
            impedance = np.zeros((6, 6))
            impedance[:3, :3], impedance[3:, 3:] = linear * np.eye(3), angular
            head = solver.head_inertia.numpy()[0]
            # The body's mass, inertia and pose reach the attachment in single precision, 1.2e-7, and take a
            # few operations there (measured 7e-8).
            np.testing.assert_allclose(
                head, impedance, rtol=1.0e-6, atol=1.0e-6 * angular.max(), err_msg=f"step {step}"
            )
            solved = head @ solver.head_motion.numpy()[0]
            turned, turned_back = np.linalg.solve(angular, solved[3:]), np.linalg.solve(angular, applied[3:])
            arm = _rotate(body.rotation, point)
            expected = (solved[:3] - applied[:3]) / linear + np.cross(turned - turned_back, arm)
            offset = _nodes(solver.q)[0, :3] - body.point(point)
            # The head turns back and then on again; composed, not added, the two turns leave their product.
            # Its place is formed in single precision from the pose, the wrench taken back and the turned
            # lever: a few units in the last place of the point's largest coordinate (measured: two).
            rounding = 8.0 * float(np.spacing(np.float32(np.abs(body.point(point)).max())))
            bound = (np.linalg.norm(turned) + np.linalg.norm(turned_back)) ** 2 * lever + rounding
            self.assertLess(np.linalg.norm(offset - expected), bound, msg=f"step {step}")
        return body, wrench, applied

    def test_lagged_mode_settles_at_the_weight_of_a_hanging_thread(self):
        """A thread hanging from a body on a spring pulls on it with its weight, at its centre of mass.

        Fails if the wrench does not settle, the head's impedance is not Z from set_support, node 0 is off the
        body's point other than by Z^-1 times the wrench still missing, or the settled wrench is not the weight's.
        """
        elements = 20
        length = elements * ELEMENT
        point, tangent = np.array([0.0, 0.001, 0.003]), np.array([1.0, 0.0, 0.0])
        tolerance = 1.0e-4 * NODE_WEIGHT
        # Critical damping of the hanging thread's slowest swing, as in the solver's own test.
        solver = _solver(
            _line(elements, start=point),
            model=_body_model(),
            mass_damping=2.0 * 1.2 * math.sqrt(GRAVITY / length),
            tolerance=tolerance,
        )
        # The body's spring has sqrt(k / m) dt = 2; two seconds bring the thread to rest.
        body, wrench, applied = self._lagged(solver, point, tangent, 2.0 / DT, 120)

        # The weight acts on every node through the consistent mass, on its position and on its slope.
        load = _mass_matrix(solver.thread)[:, 0::2].sum(axis=1)[:, None] * (0.0, 0.0, -GRAVITY)
        nodes = _nodes(solver.q)
        force = load[0::2].sum(axis=0)
        torque = (np.cross(nodes[:, :3] - body.position, load[0::2]) + np.cross(nodes[:, 3:], load[1::2])).sum(axis=0)
        # Each node's residual is below the tolerance; in the torque it acts at up to the thread's length.
        np.testing.assert_allclose(wrench[:3], force, rtol=0.0, atol=elements * tolerance)
        np.testing.assert_allclose(wrench[3:], torque, rtol=0.0, atol=elements * tolerance * length)
        np.testing.assert_allclose(wrench[:3], applied[:3], rtol=0.0, atol=elements * tolerance)
        np.testing.assert_allclose(wrench[3:], applied[3:], rtol=0.0, atol=elements * tolerance * length)
        # The weight is not small against these bounds, and it turns the body about two axes.
        self.assertGreater(np.linalg.norm(force), 1.0e3 * elements * tolerance)
        self.assertGreater(np.abs(torque[:2]).min(), 1.0e2 * elements * tolerance * length)

    def _taut(self, frequency):
        """Draw a pinned thread taut on a body whose spring has sqrt(k / m) = ``frequency``; the wrench must settle."""
        elements = 20
        length = elements * ELEMENT
        # A hundred times the soft default, so that the thread is stiffer along itself than the body's support
        # is within a step: a force passed on one step late without the head's inertia Z would then diverge.
        modulus = 2.0e8
        axial = modulus * math.pi * RADIUS**2
        damping, stiffness = self._support(frequency)
        self.assertGreater(axial / length, 10.0 * (BODY_MASS / DT**2 + damping[0] / DT + stiffness[0]))
        # The thread leaves the body along the line through its centre.
        point, tangent = np.array([0.003, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        tolerance = 1.0e-4 * NODE_WEIGHT
        end = 6 * elements
        solver = _solver(
            _line(elements, start=point),
            modulus=modulus,
            model=_body_model(),
            gravity=(0.0, 0.0, 0.0),
            tolerance=tolerance,
            fixed_dofs=[end, end + 1, end + 2],
        )
        # The pin draws the far end out by 1 % of the thread's length within half a second.
        pull, ramp = 0.01 * length, 30
        boundary = solver.boundary_q.numpy().copy()

        def draw(step):
            boundary[end] = point[0] + length + pull * min(1.0, (step + 1) / ramp)
            solver.boundary_q.assign(boundary)

        # On the stiff spring a linear model of this scene leaves 0.77 of the wrench's change from step to step
        # once the ramp is over: 120 more steps bring it far below the solver's tolerance.
        body, wrench, applied = self._lagged(solver, point, tangent, frequency, ramp + 120, before_step=draw)

        # Settled: the wrench no longer changes beyond what the solver's tolerance on every node allows.
        np.testing.assert_allclose(wrench, applied, rtol=0.0, atol=elements * tolerance)
        # Green strain of the last element, far from node 0 whose slope keeps its reference length; the tension
        # is the derivative of E A e² / 2 per reference length by the stretch.
        nodes = _nodes(solver.q)
        stretch = np.linalg.norm(nodes[-1, :3] - nodes[-2, :3]) / ELEMENT
        tension = axial * 0.5 * (stretch**2 - 1.0) * stretch
        self.assertGreater(tension, 1.0e3 * elements * tolerance)
        np.testing.assert_allclose(wrench[:3], (tension, 0.0, 0.0), rtol=0.0, atol=elements * tolerance)
        # The thread pulls through the body's centre: no torque beyond the tolerance at the lever.
        np.testing.assert_allclose(wrench[3:], 0.0, rtol=0.0, atol=elements * tolerance * np.linalg.norm(point))
        # The body's spring holds the same force: the body gave way by tension / k.
        self.assertAlmostEqual(body.position[0], tension / stiffness[0], delta=elements * tolerance / stiffness[0])

    def test_lagged_mode_settles_at_the_tension_of_a_taut_thread(self):
        """A pinned thread drawn taut pulls a body on a stiff spring with the tension of its stretch.

        Fails if the wrench does not settle on a thread 36 times stiffer than the body's support, or settles at
        a force other than the thread's tension, along the thread.
        """
        # The spring's own frequency is two per step: its stiffness and damping are four times the inertia m / dt².
        self._taut(2.0 / DT)

    # Defect: on a spring slower than about sqrt(k / m) dt = 0.83 a stiff, taut thread makes the lagged wrench diverge.
    @unittest.expectedFailure
    def test_lagged_mode_settles_on_a_slow_spring(self):
        """The same taut thread must settle on a critically damped spring with sqrt(k / m) dt = 0.75 as well.

        Fails, as expected for now: the wrench oscillates about the tension and grows by 2.6 % per step.
        """
        self._taut(0.75 / DT)


if __name__ == "__main__":
    unittest.main()
