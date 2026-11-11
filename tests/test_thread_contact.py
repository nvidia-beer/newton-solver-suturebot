# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Friction, eyelets, self-contact and capsule contact of the suture thread solver.

Every scene steps a 0.3 mm thread of 2 mm elements 60 times per second and
solves each step to a ten-thousandth of a node's weight. A sum of forces over
a chain is then known to that tolerance times its number of nodes, which is
the tolerance of the force balances below. The other tolerances are derived
where they are used.
"""

import functools
import math
import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_suturebot import SolverThread, ThreadCapsuleContact, ThreadModel

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")

# The thread of the suture examples: 0.3 mm thick, nylon, the soft default modulus.
RADIUS = 1.5e-4  # [m]
DENSITY = 1140.0  # [kg/m³]
YOUNG = 2.0e6  # [Pa]
GRAVITY = 9.81  # [m/s²]
ELEMENT = 0.002  # element length [m]
DT = 1.0 / 60.0  # [s]
DAMPING = 2.0  # mass damping rate of the examples [1/s]
# Contact and eyelet springs [N/m] and the width of the contact onset [m], as in the scenes behind the numbers of
# docs/method.md. The onset carries STIFFNESS * SMOOTHING / 4 = 120 µN at the contact distance, 76 node weights,
# so a thread that only carries its weight rests above that distance.
STIFFNESS = 32.0
SMOOTHING = 0.1 * RADIUS

AREA = math.pi * RADIUS**2
AXIAL = YOUNG * AREA  # E A [N]
BENDING = YOUNG * 0.25 * math.pi * RADIUS**4  # E I [N m²]
NODE_MASS = DENSITY * AREA * ELEMENT  # mass of one element, the share of an inner node [kg]
NODE_WEIGHT = NODE_MASS * GRAVITY  # 1.58 µN
# Newton stops a step once every free residual row is below this force [N]. Every step of every scene gets there,
# the slowest in 54 updates. A hundred times lower, a step that follows a stop stalls just above the tolerance.
TOLERANCE = 1.0e-4 * NODE_WEIGHT
UPDATES = 100
# A residual at the tolerance changes a node's velocity by this much in a step [m/s]; a thread at rest is slower.
REST_SPEED = TOLERANCE * DT / NODE_MASS
# Rounding of float64 coordinates of a few centimetres is 1e-17 m; a thousand times that [m].
ROUNDING = 1.0e-14


def setUpModule():
    if DEVICE.startswith("cuda") and not wp.is_cuda_available():
        raise unittest.SkipTest("NEWTON_TEST_DEVICE names a CUDA device and there is none")


def _straight(elements, start=(0.0, 0.0, 0.0)):
    """Node positions of a straight chain along x."""
    return np.linspace(0.0, elements * ELEMENT, elements + 1)[:, None] * np.array([1.0, 0.0, 0.0]) + np.asarray(start)


def _along(path, elements):
    """Node positions of a chain of ``elements`` elements laid along a polyline from its first point."""
    stations = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
    if stations[-1] < elements * ELEMENT:
        raise ValueError("the path is shorter than the chain")
    at = ELEMENT * np.arange(elements + 1)
    return np.column_stack([np.interp(at, stations, path[:, axis]) for axis in range(3)])


def _coordinates(positions):
    """Positions and unit slopes of a chain through ``positions``, flat: what ``SolverThread.q`` holds."""
    chain = ThreadModel.create(positions, young_modulus=YOUNG, radius=RADIUS, density=DENSITY, device="cpu")
    return chain.rest_q.numpy().copy()


def _solver(model, rest_positions, **options):
    """Thread solver of a chain that is stress-free through ``rest_positions``, with this file's settings."""
    thread = ThreadModel.create(rest_positions, young_modulus=YOUNG, radius=RADIUS, density=DENSITY, device=DEVICE)
    settings = dict(mass_damping=DAMPING, tolerance=TOLERANCE, max_iterations=UPDATES)
    settings.update(options)
    return SolverThread(model, thread, **settings)


def _floor():
    """Solver options of a level floor at z = 0 with this file's contact law."""
    return dict(contact_distance=RADIUS, contact_stiffness=STIFFNESS, contact_smoothing=SMOOTHING)


def _read(array):
    """Host copy of a device array; on a CPU ``numpy()`` alone is a view that later steps overwrite."""
    return array.numpy().copy()


def _nodes(solver):
    """Node coordinates and velocities, each of shape (chains, nodes, 6): positions [m] and slopes."""
    shape = (solver.n_envs, solver.thread.node_count, 6)
    return _read(solver.q).reshape(shape), _read(solver.qd).reshape(shape)


def _residual(solver):
    """Step residual of every node [N, N m], shape (chains, nodes, 6); a chain's node 0 holds its support's force."""
    return _read(solver.residual).reshape(solver.n_envs, solver.thread.node_count, 6)


def _unbalanced(solver):
    """Largest free residual of the last step in the solver's norm [N]: slope rows over the element length."""
    rows = np.abs(_residual(solver)[:, 1:])
    return float(max(rows[..., :3].max(), rows[..., 3:].max() / ELEMENT))


def _speed(solver):
    """Largest speed of a node [m/s]."""
    return float(np.linalg.norm(_nodes(solver)[1][..., :3], axis=-1).max())


def _weight(solver):
    """Weight of one chain [N] from its consistent mass: the position blocks of all nodes and couplings."""
    thread = solver.thread
    mass = thread.mass_diagonal.numpy()[:, 0, 0].sum() + 2.0 * thread.mass_coupling.numpy()[:, 0, 0].sum()
    return float(mass) * GRAVITY


def _centerline(coordinates, element, xi):
    """Point and unit tangent of the cubic Hermite centreline of one chain at ``xi`` in [0, 1] of ``element``."""
    a, b = coordinates[element], coordinates[element + 1]
    x2, x3 = xi * xi, xi**3
    point = (
        (1.0 - 3.0 * x2 + 2.0 * x3) * a[:3]
        + ELEMENT * (xi - 2.0 * x2 + x3) * a[3:]
        + (3.0 * x2 - 2.0 * x3) * b[:3]
        + ELEMENT * (x3 - x2) * b[3:]
    )
    tangent = (
        (6.0 * xi - 6.0 * x2) / ELEMENT * (b[:3] - a[:3])
        + (1.0 - 4.0 * xi + 3.0 * x2) * a[3:]
        + (3.0 * x2 - 2.0 * xi) * b[3:]
    )
    return point, tangent / np.linalg.norm(tangent)


def _segment_distance(a0, a1, b0, b1):
    """Least distance between the segments a0-a1 and b0-b1; arrays of shape (..., 3) that broadcast."""
    d1, d2, r = a1 - a0, b1 - b0, a0 - b0
    a, e, b = np.sum(d1 * d1, axis=-1), np.sum(d2 * d2, axis=-1), np.sum(d1 * d2, axis=-1)
    c, f = np.sum(d1 * r, axis=-1), np.sum(d2 * r, axis=-1)
    denominator = a * e - b * b
    skew = denominator > 1.0e-12 * a * e
    s = np.where(skew, np.clip((b * f - c * e) / np.where(skew, denominator, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(a0 + s[..., None] * d1 - b0 - t[..., None] * d2, axis=-1)


def _contact_push(gap):
    """Force of the contact penalty on a point that is ``gap`` beyond its contact distance [N].

    With the depth d = -gap and the smoothing s the penetration is p = (d + sqrt(d² + s²)) / 2 and the potential
    k p² / 2 (docs/method.md), so the force is k p (1 + d / sqrt(d² + s²)) / 2.
    """
    root = math.hypot(gap, SMOOTHING)
    return STIFFNESS * 0.5 * (root - gap) * 0.5 * (1.0 - gap / root)


def _standoff(force):
    """Gap beyond the contact distance at which the penalty carries ``force`` [m]; the push falls as the gap grows."""
    low, high = 0.0, 10.0 * SMOOTHING
    if not _contact_push(high) < force < _contact_push(low):
        raise ValueError("the force is outside the onset of the contact penalty")
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if _contact_push(middle) > force else (low, middle)
    return 0.5 * (low + high)


# A contact that carries one node's weight stands this far off its contact distance: 38 µm.
NODE_STANDOFF = _standoff(NODE_WEIGHT)
# Least distances between centrelines are taken between points sampled every 31 or 63 µm. A sample is at most
# half that from the nearest point of its curve, which lengthens a distance of 0.3 mm or more by less than 1 µm.
SAMPLING = 1.0e-6


class _ThreadCase(unittest.TestCase):
    def assert_solved(self, state):
        """The step behind ``state`` ended below the tolerance, so its forces balance to that tolerance per node."""
        self.assertLessEqual(state["unbalanced"], TOLERANCE)


# Drag on a plane: 20 elements lie along x on the floor, node 0 at the contact distance, 38 µm below where the
# floor carries the other nodes. Node 0 draws the thread along its axis and stops.
PLANE_ELEMENTS = 20
PLANE_MASS = PLANE_ELEMENTS * NODE_MASS
PLANE_BALANCE = PLANE_ELEMENTS * TOLERANCE  # of a sum of forces over the 20 free nodes [N]
DRAG_SPEED = 0.004  # [m/s]
SETTLE_STEPS = 60  # 1 s: the nodes rise from the contact distance to where the floor's onset carries them
DRAG_STEPS = 60  # 1 s, two element lengths
REST_STEPS = 60


@functools.cache
def _drag_on_plane(friction):
    """Settle, drag and stop the thread on the floor; forces at the end of the drag and at two times after it."""
    solver = _solver(
        newton.ModelBuilder().finalize(DEVICE),
        _straight(PLANE_ELEMENTS, (0.0, 0.0, RADIUS)),
        gravity=(0.0, 0.0, -GRAVITY),
        friction=friction,
        **_floor(),
    )

    def measure():
        contact = _read(solver.contact_force)
        return dict(
            support=_residual(solver)[0, 0, :3],
            normal=float(contact[:, 2].sum()),
            friction=contact[:, :2].sum(axis=0),
            height=_nodes(solver)[0][0, :, 2],
            speed=_speed(solver),
            unbalanced=_unbalanced(solver),
        )

    boundary = _read(solver.boundary_q)
    for _ in range(SETTLE_STEPS):
        solver.step(None, None, None, None, DT)
    for _ in range(DRAG_STEPS):
        boundary[0] -= DRAG_SPEED * DT
        solver.boundary_q.assign(boundary)
        solver.step(None, None, None, None, DT)
    result = dict(dragged=measure())
    for name in ("stopped", "rest"):
        for _ in range(REST_STEPS // 2):
            solver.step(None, None, None, None, DT)
        result[name] = measure()
    return result


# Moving surface: the scene of the drag turned as a whole, 30° about y and then 70° about z, so that neither the
# contact plane nor the thread lies along an axis. Node 0 is held and the surface moves under the thread along the
# thread's axis instead: before every step the nodes' friction anchors are carried along with it.
TURN_Y, TURN_Z = math.radians(30.0), math.radians(70.0)
TURN = np.array(
    [[math.cos(TURN_Z), -math.sin(TURN_Z), 0.0], [math.sin(TURN_Z), math.cos(TURN_Z), 0.0], [0.0, 0.0, 1.0]]
) @ np.array([[math.cos(TURN_Y), 0.0, math.sin(TURN_Y)], [0.0, 1.0, 0.0], [-math.sin(TURN_Y), 0.0, math.cos(TURN_Y)]])


@functools.cache
def _held_on_moving_plane():
    """Settle the thread on the turned plane and move the surface under it; forces in the frame of the level scene."""
    solver = _solver(
        newton.ModelBuilder().finalize(DEVICE),
        _straight(PLANE_ELEMENTS, (0.0, 0.0, RADIUS)) @ TURN.T,
        gravity=tuple(TURN @ [0.0, 0.0, -GRAVITY]),
        friction=0.2,
        **_floor(),
    )
    nodes = PLANE_ELEMENTS + 1
    solver.contact_normal.assign(np.tile(TURN @ [0.0, 0.0, 1.0], (nodes, 1)))
    solver.contact_point.assign(np.zeros((nodes, 3)))
    for _ in range(SETTLE_STEPS):
        solver.step(None, None, None, None, DT)
    travel = DRAG_SPEED * DT * (TURN @ [1.0, 0.0, 0.0])
    for _ in range(DRAG_STEPS):
        solver.contact_anchor.assign(_read(solver.contact_anchor) + travel)
        solver.step(None, None, None, None, DT)
    return dict(
        support=_residual(solver)[0, 0, :3] @ TURN,
        contact=_read(solver.contact_force).sum(axis=0) @ TURN,
        speed=_speed(solver),
        unbalanced=_unbalanced(solver),
    )


class ThreadPlaneFrictionTests(_ThreadCase):
    """A thread on a level floor is drawn along its axis at 4 mm/s for a second and then held."""

    def test_drag_is_held_back_by_friction_times_normal_forces(self):
        """Fails if a sliding node's friction is not the coefficient times its normal force, against its motion."""
        friction = 0.2
        dragged = _drag_on_plane(friction)["dragged"]
        self.assert_solved(dragged)
        # The floor and the support of node 0 carry the weight. The support holds node 0 below the others and
        # pulls the thread down, so the normal forces sum to more than the weight: 38.6 µN against 31.6 µN.
        weight = PLANE_ELEMENTS * NODE_WEIGHT
        self.assertAlmostEqual(dragged["normal"] + dragged["support"][2], weight, delta=PLANE_BALANCE)
        self.assertGreater(dragged["normal"], weight + NODE_WEIGHT)
        # A free inner node rests where the floor's onset carries its weight. Node 0, held 38 µm lower, bends
        # the thread over the length (4 E I / k')^(1/4) = 2.7 mm of a beam on the onset, whose stiffness per
        # length is k' = 57 N/m² there. The middle node is 7.3 such lengths away, where that has decayed to
        # 0.03 µm; a hundredth of the standoff, 0.4 µm, is the tolerance.
        middle = dragged["height"][PLANE_ELEMENTS // 2]
        self.assertAlmostEqual(middle, RADIUS + NODE_STANDOFF, delta=0.01 * NODE_STANDOFF)
        # The friction on the nodes points against the motion along -x. A node's limit is the coefficient times
        # its normal force one step earlier, hence twice the tolerance of one balance.
        self.assertAlmostEqual(dragged["friction"][0], friction * dragged["normal"], delta=2.0 * PLANE_BALANCE)
        self.assertAlmostEqual(dragged["friction"][1], 0.0, delta=PLANE_BALANCE)
        # The support draws against that friction and against the mass damping c M v of the moving thread.
        damping = DAMPING * PLANE_MASS * DRAG_SPEED
        hold_back = -dragged["support"][0]
        self.assertAlmostEqual(hold_back, friction * dragged["normal"] + damping, delta=2.0 * PLANE_BALANCE)

    def test_thread_keeps_its_tension_after_the_drag(self):
        """Fails if the thread creeps on once node 0 has stopped, or the nodes do not stick and hold its tension."""
        friction = 0.2
        run = _drag_on_plane(friction)
        sliding = friction * run["dragged"]["normal"]
        # The first step after the stop takes the thread's momentum M v out of the support's pull, and the
        # damping force c M v with it.
        momentum = PLANE_MASS * DRAG_SPEED / DT
        damping = DAMPING * PLANE_MASS * DRAG_SPEED
        # Then the nodes stick and settle back on the springs to their anchors. Element 0, of stiffness
        # E A / l = 71 N/m, pulls node 1 against that node's anchor spring k, so it wins back at most the share
        # (E A / l) / (E A / l + k) = 0.69 of the loss. Nodes that kept sliding would give all of it back.
        won_back = (AXIAL / ELEMENT) / (AXIAL / ELEMENT + STIFFNESS)
        for name in ("stopped", "rest"):
            with self.subTest(state=name):
                state = run[name]
                self.assert_solved(state)
                self.assertLessEqual(state["speed"], REST_SPEED)
                held = -state["support"][0]
                self.assertGreaterEqual(held, sliding - momentum - damping - PLANE_BALANCE)
                self.assertLessEqual(held, sliding - (1.0 - won_back) * momentum + PLANE_BALANCE)
                self.assertAlmostEqual(state["friction"][0], held, delta=PLANE_BALANCE)
        # Half a second later the tension is the same: both states are solved to the tolerance and no node's
        # anchor has moved between them.
        change = run["rest"]["support"][0] - run["stopped"]["support"][0]
        self.assertAlmostEqual(change, 0.0, delta=2.0 * PLANE_BALANCE)

    def test_moving_surface_drags_a_held_thread(self):
        """Fails if friction ignores a node's own contact plane, or anchors that are carried along with a surface."""
        friction = 0.2
        state = _held_on_moving_plane()
        self.assert_solved(state)
        self.assertLessEqual(state["speed"], REST_SPEED)
        # In the frame of the level scene the plane's forces are normal forces along z and friction along x,
        # the way the surface moves; the support holds the thread against that and against its weight.
        normal = state["contact"][2]
        weight = PLANE_ELEMENTS * NODE_WEIGHT
        np.testing.assert_allclose(state["contact"][:2], [friction * normal, 0.0], rtol=0.0, atol=2.0 * PLANE_BALANCE)
        np.testing.assert_allclose(
            state["contact"] + state["support"], [0.0, 0.0, weight], rtol=0.0, atol=PLANE_BALANCE
        )

    def test_no_friction_no_hold_back(self):
        """Fails if a floor with friction 0 holds the dragged thread back or keeps tension in it after the stop."""
        run = _drag_on_plane(0.0)
        for state in run.values():
            self.assert_solved(state)
            np.testing.assert_array_equal(state["friction"], 0.0)
        # Only the mass damping c M v of the moving thread is left of the hold-back: 0.026 µN against 7.75 µN.
        damping = DAMPING * PLANE_MASS * DRAG_SPEED
        self.assertAlmostEqual(-run["dragged"]["support"][0], damping, delta=PLANE_BALANCE)
        for name in ("stopped", "rest"):
            self.assertLessEqual(run[name]["speed"], REST_SPEED)
            self.assertAlmostEqual(run[name]["support"][0], 0.0, delta=PLANE_BALANCE)


# Capstan: a taut thread without weight turns a quarter circle over seven eyelets, 15° apart on a radius of 20 mm.
# Node 0 draws it along a straight lead; the far end is tied to a held eyelet that is fed in at the same speed, so
# that eyelet's spring shows the tension behind the arc.
ARC_RADIUS = 0.020  # [m]
EYELET_ANGLES = np.radians(np.arange(0.0, 90.1, 15.0))
LEAD, TAIL = 0.006, 0.020  # straight thread before and after the arc at the start [m]
CAPSTAN_ELEMENTS = round((LEAD + 0.5 * math.pi * ARC_RADIUS + TAIL) / ELEMENT)  # 29
CAPSTAN_BALANCE = CAPSTAN_ELEMENTS * TOLERANCE  # of a sum of forces over the free nodes [N]
PRETENSION = 0.00015  # stretch of the 57 mm thread before the draw [m]: a strain of 0.26 %, a tension of 0.3 mN
DRAW_SPEED = 0.004  # [m/s]
TRAVEL = DRAW_SPEED * DT / ELEMENT  # of the thread through an eyelet in one step, in elements: 1/30
DRAW_STEPS = 180  # 3 s: 12 mm, six elements through every eyelet
FEED_STEPS = 90  # 1.5 s back: the lead stays 12 mm long, eight bending lengths sqrt(E I / T) = 1.5 mm
SLIDING = len(EYELET_ANGLES)  # eyelets 0 to 6 slide, eyelet 7 is the held far end, slot 8 stays unthreaded
SPARE = np.array([0.05, 0.05, 0.05])  # position of the unthreaded eyelet [m]
# The straight lead and tail are tangent to the circle, and a chord between two eyelets leans 7.5° to the tangent
# at either end: the thread turns by 7.5° at the first and the last eyelet and by 15° at the five between.
TURNS = np.radians([7.5, 15.0, 15.0, 15.0, 15.0, 15.0, 7.5])
# Within a step an eyelet holds the chain point that it tracked when the step began, across the thread's direction
# there. The draw carries that point on by the step's travel, to where the thread has turned further round the
# arc, so the eyelet's spring leans against the draw by the angle the thread turns over that travel and adds to
# the friction angle. Bending spreads the turn at an eyelet over 2 sqrt(E I / T) = 3 mm; the turn over one
# element, 2 mm, bounds the thread's turning rate there. That gives 0.0087 rad; the scenes measure up to 0.0068.
LAG = DRAW_SPEED * DT * TURNS.max() / ELEMENT
# An eyelet's friction limit is the coefficient times its force across the thread one step earlier. That force
# changes while nodes pass the eyelet, by up to 0.34 % from step to step in these scenes; three times that.
STEP_CHANGE = 0.01


def _coulomb_ratio(friction):
    """Tension before over tension behind the eyelets when the thread slides through all of them.

    An eyelet that turns the thread by the angle a presses on it with (T1 + T2) sin(a / 2) across the bisector
    and holds it back with the friction coefficient times that; along the bisector (T1 - T2) cos(a / 2) balances
    it, so T1 / T2 = (1 + mu tan(a / 2)) / (1 - mu tan(a / 2)).
    """
    grip = friction * np.tan(0.5 * TURNS)
    return float(np.prod((1.0 + grip) / (1.0 - grip)))


def _lagging(friction):
    """Friction coefficient whose friction angle is that of ``friction`` plus the lean of a step's travel."""
    return math.tan(math.atan(friction) + LAG)


def _bending_share(state):
    """Part of a tension that the thread's bending can shift between the two ends of the arc.

    Coulomb's ratio is that of a string. Bent to the arc's radius the thread carries a force E I / radius² = 2 µN
    across itself, which the turn exchanges with the tension; it is taken against the smaller tension.
    """
    return BENDING / (ARC_RADIUS**2 * min(state["tension_in"], state["tension_out"]))


@functools.cache
def _capstan(friction):
    """Pretension, draw, stop and feed back the thread over the eyelets; forces and tracked points of each phase."""
    eyelets = ARC_RADIUS * np.column_stack((np.cos(EYELET_ANGLES), 0.0 * EYELET_ANGLES, np.sin(EYELET_ANGLES)))
    arc = np.radians(np.linspace(90.0, 0.0, 181))
    path = np.vstack(
        (
            [-LEAD, 0.0, ARC_RADIUS],
            ARC_RADIUS * np.column_stack((np.cos(arc), 0.0 * arc, np.sin(arc))),
            [ARC_RADIUS, 0.0, -TAIL - ELEMENT],
        )
    )
    laid = _coordinates(_along(path, CAPSTAN_ELEMENTS))
    solver = _solver(
        newton.ModelBuilder().finalize(DEVICE),
        _straight(CAPSTAN_ELEMENTS),
        gravity=(0.0, 0.0, 0.0),
        friction=friction,
        max_eyelets=SLIDING + 2,
        eyelet_stiffness=STIFFNESS,
    )
    solver.q.assign(laid)
    solver.boundary_q.assign(laid)
    # Each sliding eyelet starts on the chain point that lies on it; the held one is tied to the last node.
    stations = LEAD + ARC_RADIUS * (0.5 * math.pi - EYELET_ANGLES)
    solver.eyelet_element.assign(np.r_[(stations // ELEMENT).astype(np.int32), CAPSTAN_ELEMENTS - 1, -1])
    solver.eyelet_xi.assign(np.r_[(stations % ELEMENT) / ELEMENT, 1.0, 0.0])
    solver.eyelet_held.assign(np.r_[np.zeros(SLIDING, dtype=np.int32), 1, 0])
    target = np.vstack((eyelets, laid[-6:-3], SPARE))
    boundary = laid.copy()

    def run(steps, draw, feed):
        for _ in range(steps):
            boundary[0] -= draw * DT
            target[SLIDING, 2] += feed * DT
            solver.boundary_q.assign(boundary)
            solver.eyelet_target.assign(target)
            solver.step(None, None, None, None, DT)
        force, point = _read(solver.eyelet_force), _read(solver.eyelet_point)
        element, xi = _read(solver.eyelet_element), _read(solver.eyelet_xi)
        coordinates = _nodes(solver)[0][0]
        # The chain's point at each sliding eyelet, and the eyelet's force along and across the thread there.
        on_chain, tangent = map(
            np.array, zip(*(_centerline(coordinates, element[i], xi[i]) for i in range(SLIDING)), strict=True)
        )
        along = np.sum(force[:SLIDING] * tangent, axis=1)
        across = force[:SLIDING] - along[:, None] * tangent
        offset = point[:SLIDING] - target[:SLIDING]
        support = _residual(solver)[0, 0, :3]
        return dict(
            tension_in=float(np.linalg.norm(support)),
            tension_out=float(np.linalg.norm(force[SLIDING])),
            support=support,
            force=force,
            point=point,
            target=target.copy(),
            tracked=(element + xi)[:SLIDING],
            held=(int(element[SLIDING]), float(xi[SLIDING])),
            last_node=coordinates[-1, :3],
            on_chain=on_chain,
            along=along,
            across=np.linalg.norm(across, axis=1),
            across_spring=STIFFNESS * (offset - np.sum(offset * tangent, axis=1)[:, None] * tangent) - across,
            speed=_speed(solver),
            unbalanced=_unbalanced(solver),
        )

    run(30, PRETENSION / (30 * DT), 0.0)
    result = dict(taut=run(30, 0.0, 0.0))
    result["drawn"] = run(DRAW_STEPS, DRAW_SPEED, DRAW_SPEED)
    result["stopped"] = run(30, 0.0, 0.0)
    result["rest"] = run(30, 0.0, 0.0)
    result["fed"] = run(FEED_STEPS, -DRAW_SPEED, -DRAW_SPEED)
    return result


# Knot: two straight chains without weight side by side, each through one frictionless eyelet 12.8 mm from
# node 0, inside element 6. The eyelet of chain 0 is held. Node 0 of both chains is drawn back along the thread.
KNOT_ELEMENTS = 20
KNOT_ELEMENT, KNOT_XI = 6, 0.4
KNOT_PULL = 2.0e-5  # [m]: a strain of 0.16 % of the 12.8 mm between node 0 and the eyelet
KNOT_OFFSET = 0.005  # distance between the two chains [m]
KNOT_DRAW_STEPS, KNOT_STEPS = 12, 60  # the draw takes 0.2 s; the chains are left for 0.8 s more


@functools.cache
def _held_and_sliding():
    """Draw two chains through one eyelet each, the first held; the eyelets' state when the chains are at rest."""
    solver = _solver(
        newton.ModelBuilder().finalize(DEVICE),
        _straight(KNOT_ELEMENTS),
        n_envs=2,
        gravity=(0.0, 0.0, 0.0),
        max_eyelets=2,
        eyelet_stiffness=STIFFNESS,
    )
    nodes = KNOT_ELEMENTS + 1
    boundary = _read(solver.q)
    boundary.reshape(2, nodes, 6)[1, :, 1] += KNOT_OFFSET
    solver.q.assign(boundary)
    station = (KNOT_ELEMENT + KNOT_XI) * ELEMENT
    target = np.array([[station, 0.0, 0.0], [station, KNOT_OFFSET, 0.0]])
    solver.eyelet_target.assign(target)
    # An eyelet names its element among the elements of the whole batch.
    solver.eyelet_element.assign(np.array([KNOT_ELEMENT, KNOT_ELEMENTS + KNOT_ELEMENT], dtype=np.int32))
    solver.eyelet_xi.assign(np.array([KNOT_XI, KNOT_XI]))
    solver.eyelet_held.assign(np.array([1, 0], dtype=np.int32))
    for step in range(KNOT_STEPS):
        if step < KNOT_DRAW_STEPS:
            boundary[0] -= KNOT_PULL / KNOT_DRAW_STEPS
            boundary[6 * nodes] -= KNOT_PULL / KNOT_DRAW_STEPS
        solver.boundary_q.assign(boundary)
        solver.step(None, None, None, None, DT)
    coordinates = _nodes(solver)[0]
    element, xi = _read(solver.eyelet_element), _read(solver.eyelet_xi)
    return dict(
        force=_read(solver.eyelet_force),
        point=_read(solver.eyelet_point),
        target=target,
        element=element,
        xi=xi,
        on_chain=np.array([_centerline(coordinates[i], element[i] - i * KNOT_ELEMENTS, xi[i])[0] for i in range(2)]),
        support=_residual(solver)[:, 0, :3],
        speed=_speed(solver),
        unbalanced=_unbalanced(solver),
    )


class ThreadEyeletTests(_ThreadCase):
    """Eyelets: points in space that the thread passes through, sliding against friction or held as a knot."""

    def test_draw_over_eyelets_needs_the_coulomb_factor(self):
        """Fails if an eyelet's friction is not the coefficient times its force across the thread."""
        for friction in (0.2, 0.5, 0.0):
            with self.subTest(friction=friction):
                drawn = _capstan(friction)["drawn"]
                self.assert_solved(drawn)
                # Coulomb's ratio at the least, and at the most that of the friction angle plus the lag's lean.
                ratio = drawn["tension_in"] / drawn["tension_out"]
                self.assertGreaterEqual(ratio, _coulomb_ratio(friction) * (1.0 - _bending_share(drawn)))
                self.assertLessEqual(ratio, _coulomb_ratio(_lagging(friction)) * (1.0 + _bending_share(drawn)))
                # The thread runs towards node 0, against the chain's direction, and drags every eyelet that
                # way: the eyelet's force leans from across the thread by the friction angle and the lag's lean.
                lean = np.arctan2(-drawn["along"], drawn["across"])
                self.assertGreaterEqual(lean.min(), math.atan(friction * (1.0 - STEP_CHANGE)))
                self.assertLessEqual(lean.max(), math.atan(friction) + LAG)

    def test_feeding_back_needs_the_inverse_factor(self):
        """Fails if eyelet friction does not turn round with the direction in which the thread slides."""
        for friction in (0.2, 0.5, 0.0):
            with self.subTest(friction=friction):
                fed = _capstan(friction)["fed"]
                self.assert_solved(fed)
                ratio = fed["tension_out"] / fed["tension_in"]
                self.assertGreaterEqual(ratio, _coulomb_ratio(friction) * (1.0 - _bending_share(fed)))
                self.assertLessEqual(ratio, _coulomb_ratio(_lagging(friction)) * (1.0 + _bending_share(fed)))
                lean = np.arctan2(fed["along"], fed["across"])
                self.assertGreaterEqual(lean.min(), math.atan(friction * (1.0 - STEP_CHANGE)))
                self.assertLessEqual(lean.max(), math.atan(friction) + LAG)

    def test_stopped_thread_keeps_its_tension_difference(self):
        """Fails if the eyelets let the tension difference go, or the thread creep, once the draw has stopped."""
        for friction in (0.2, 0.5):
            with self.subTest(friction=friction):
                run = _capstan(friction)
                stopped, rest = run["stopped"], run["rest"]
                # The first step after the stop takes the thread's momentum M v out of the tensions, at most
                # M v / h from the one before the arc and into the one behind it.
                momentum = 2.0 * CAPSTAN_ELEMENTS * NODE_MASS * DRAW_SPEED / (DT * rest["tension_out"])
                for state in (stopped, rest):
                    self.assert_solved(state)
                    self.assertLessEqual(state["speed"], REST_SPEED)
                    ratio = state["tension_in"] / state["tension_out"]
                    share = _bending_share(state)
                    self.assertGreaterEqual(ratio, _coulomb_ratio(friction) * (1.0 - share - momentum))
                    self.assertLessEqual(ratio, _coulomb_ratio(friction) * (1.0 + share))
                    # No eyelet holds more along the thread than the coefficient times its force across it.
                    self.assertLessEqual((np.abs(state["along"]) - friction * state["across"]).max(), CAPSTAN_BALANCE)
                # Half a second later both tensions are the same: both states are solved to the tolerance and
                # no eyelet's anchored point has moved between them.
                for end in ("tension_in", "tension_out"):
                    self.assertAlmostEqual(rest[end], stopped[end], delta=2.0 * CAPSTAN_BALANCE)

    def test_frictionless_eyelets_hold_no_tension_difference(self):
        """Fails if eyelets with friction 0 hold the thread back along itself."""
        run = _capstan(0.0)
        for name in ("taut", "stopped", "rest"):
            with self.subTest(state=name):
                state = run[name]
                self.assert_solved(state)
                self.assertLessEqual(state["speed"], REST_SPEED)
                self.assertAlmostEqual(state["tension_in"] / state["tension_out"], 1.0, delta=_bending_share(state))
                # At rest an eyelet tracks the chain point nearest to it and pulls straight across the thread.
                self.assertLessEqual(np.abs(state["along"]).max(), CAPSTAN_BALANCE)

    def test_tracked_point_follows_the_thread_through_nodes(self):
        """Fails if an eyelet's tracked point stays behind the thread that slides through it, as at a node."""
        for friction in (0.2, 0.5, 0.0):
            with self.subTest(friction=friction):
                run = _capstan(friction)
                # The tension changes between the phases and shifts the chain's points by at most its strain
                # times the thread's length: 0.09 elements at the largest tension, 0.44 mN.
                stretch = CAPSTAN_ELEMENTS * max(state["tension_in"] for state in run.values()) / AXIAL
                # An eyelet moves its tracked point at the start of a step: it is one step's travel behind.
                drawn = run["drawn"]["tracked"] - run["taut"]["tracked"]
                np.testing.assert_allclose(drawn, (DRAW_STEPS - 1) * TRAVEL, rtol=0.0, atol=stretch)
                self.assertGreaterEqual((np.floor(run["drawn"]["tracked"]) - np.floor(run["taut"]["tracked"])).min(), 5)
                fed = run["fed"]["tracked"] - run["rest"]["tracked"]
                np.testing.assert_allclose(fed, -(FEED_STEPS - 1) * TRAVEL, rtol=0.0, atol=stretch)

    def test_eyelet_point_is_the_threads_point_at_the_eyelet(self):
        """Fails if ``eyelet_point`` is not the chain point an eyelet tracks, or not its own place when unthreaded."""
        run = _capstan(0.2)
        for name, state in run.items():
            with self.subTest(state=name):
                np.testing.assert_allclose(state["point"][:SLIDING], state["on_chain"], rtol=0.0, atol=ROUNDING)
                # The held eyelet keeps the last node, where it was tied.
                self.assertEqual(state["held"], (CAPSTAN_ELEMENTS - 1, 1.0))
                np.testing.assert_allclose(state["point"][SLIDING], state["last_node"], rtol=0.0, atol=ROUNDING)
                # The unthreaded slot reports its own position and no force.
                np.testing.assert_array_equal(state["point"][SLIDING + 1], SPARE)
                np.testing.assert_array_equal(state["force"][SLIDING + 1], 0.0)
        for name in ("taut", "rest"):
            with self.subTest(state=name):
                state = run[name]
                # At rest the point is off its eyelet by the give of the eyelet's spring across the thread.
                np.testing.assert_allclose(state["across_spring"], 0.0, rtol=0.0, atol=CAPSTAN_BALANCE)
                give = np.linalg.norm(state["point"][:SLIDING] - state["target"][:SLIDING], axis=1)
                np.testing.assert_allclose(
                    give, state["across"] / STIFFNESS, rtol=0.0, atol=CAPSTAN_BALANCE / STIFFNESS
                )

    def test_eyelet_force_is_the_threads_force_on_the_eyelets(self):
        """Fails if ``eyelet_force`` has the wrong sign or size: with the support it must balance the thread."""
        for friction in (0.2, 0.0):
            for name in ("taut", "rest"):
                with self.subTest(friction=friction, state=name):
                    state = _capstan(friction)[name]
                    self.assert_solved(state)
                    # The thread has no weight: what its support applies to node 0 it passes on to the eyelets.
                    np.testing.assert_allclose(
                        state["force"].sum(axis=0), state["support"], rtol=0.0, atol=CAPSTAN_BALANCE
                    )
                    # The held eyelet is a spring between its target and the last node.
                    spring = STIFFNESS * (state["point"][SLIDING] - state["target"][SLIDING])
                    np.testing.assert_allclose(state["force"][SLIDING], spring, rtol=0.0, atol=STIFFNESS * ROUNDING)

    def test_held_eyelet_keeps_its_point_and_holds_a_pull(self):
        """Fails if a held eyelet lets the thread slide through it, or an eyelet that is not held holds the pull."""
        knot = _held_and_sliding()
        self.assert_solved(knot)
        self.assertLessEqual(knot["speed"], REST_SPEED)
        balance = KNOT_ELEMENTS * TOLERANCE
        np.testing.assert_allclose(knot["point"], knot["on_chain"], rtol=0.0, atol=ROUNDING)

        # Chain 0: the held eyelet still tracks the point it was given and holds all of the support's pull.
        self.assertEqual((int(knot["element"][0]), float(knot["xi"][0])), (KNOT_ELEMENT, KNOT_XI))
        spring = STIFFNESS * (knot["point"][0] - knot["target"][0])
        np.testing.assert_allclose(knot["force"][0], spring, rtol=0.0, atol=STIFFNESS * ROUNDING)
        np.testing.assert_allclose(knot["force"][0], knot["support"][0], rtol=0.0, atol=balance)
        # The pull stretches the eyelet's spring and the thread between node 0 and the knot in series. Node 0
        # prescribes an unstrained slope, so the first element stretches less than the others: the tension is
        # between that of a uniform stretch and that of a thread whose first element does not stretch at all.
        length = (KNOT_ELEMENT + KNOT_XI) * ELEMENT
        tension = -knot["force"][0, 0]
        self.assertGreaterEqual(tension, KNOT_PULL / (length / AXIAL + 1.0 / STIFFNESS))
        self.assertLessEqual(tension, KNOT_PULL / ((length - ELEMENT) / AXIAL + 1.0 / STIFFNESS))
        np.testing.assert_allclose(knot["force"][0, 1:], 0.0, rtol=0.0, atol=balance)

        # Chain 1: the thread slid through by the pull and nothing holds it. Its eyelet is the second of the
        # batch's elements with that number.
        self.assertEqual(int(knot["element"][1]), KNOT_ELEMENTS + KNOT_ELEMENT)
        # The tension left in a solved chain is below the balance tolerance, so each of the seven elements up
        # to the eyelet is strained by less than that over E A.
        strain = balance / AXIAL
        slid = float(knot["xi"][1]) - KNOT_XI
        self.assertAlmostEqual(slid, KNOT_PULL / ELEMENT, delta=(KNOT_ELEMENT + 1) * strain)
        np.testing.assert_allclose(knot["force"][1], 0.0, rtol=0.0, atol=balance)
        np.testing.assert_allclose(knot["support"][1], 0.0, rtol=0.0, atol=balance)


# Crossing: a thread that is stress-free as a flat loop on the floor. It runs 40 mm along x, turns back through a
# half circle and a quarter circle of 6 mm radius, and its last stretch crosses the first at a right angle, inside
# an element of each. The crossing stretch starts lifted over the first one, within reach of the self-contact.
CROSS_AT = 0.021  # where the last stretch crosses the first [m]: the middle of element 10
CROSS_LIFT = 4.0e-4  # height of the last stretch over the first at the start [m]: a diameter and a third
CROSS_SPAN = 0.008  # half length of the lifted part [m]
CROSS_STEPS = 90  # 1.5 s; the lifted stretch has settled after 0.7 s


@functools.cache
def _crossing(self_contact):
    """Let the lifted stretch of the loop settle; least distance between two stretches and the vertical forces."""
    turn = 0.006
    rest_height = RADIUS + NODE_STANDOFF
    half = np.radians(np.linspace(-90.0, 90.0, 61))
    quarter = np.radians(np.linspace(90.0, 180.0, 31))
    plan = np.vstack(
        (
            [0.0, 0.0],
            [0.040, 0.0],
            np.column_stack((0.040 + turn * np.cos(half), turn + turn * np.sin(half)))[1:],
            [CROSS_AT + turn, 2.0 * turn],
            np.column_stack((CROSS_AT + turn + turn * np.cos(quarter), turn + turn * np.sin(quarter)))[1:],
            [CROSS_AT, -0.016],
        )
    )
    # Eight points per segment, so that the lift below is smooth along the path.
    legs = zip(plan[:-1], plan[1:], strict=True)
    plan = np.vstack([np.linspace(a, b, 8, endpoint=False) for a, b in legs] + [plan[-1:]])
    second_pass = np.arange(len(plan)) > len(plan) // 2
    near = second_pass & (np.abs(plan[:, 1]) < CROSS_SPAN)
    lift = np.where(near, CROSS_LIFT * np.cos(0.5 * np.pi * plan[:, 1] / CROSS_SPAN) ** 2, 0.0)
    elements = int(np.linalg.norm(np.diff(plan, axis=0), axis=1).sum() / ELEMENT) - 1
    solver = _solver(
        newton.ModelBuilder().finalize(DEVICE),
        _along(np.column_stack((plan, np.full(len(plan), rest_height))), elements),
        gravity=(0.0, 0.0, -GRAVITY),
        friction=0.2,
        self_contact=self_contact,
        **_floor(),
    )
    solver.q.assign(_coordinates(_along(np.column_stack((plan, rest_height + lift)), elements)))
    for _ in range(CROSS_STEPS):
        solver.step(None, None, None, None, DT)

    # The two elements, three or more apart along the chain, whose chords are nearest, and the least distance
    # between their centrelines, sampled every 31 µm.
    coordinates = _nodes(solver)[0][0]
    nodes = coordinates[:, :3]
    first, second = np.triu_indices(elements, k=3)
    chords = _segment_distance(nodes[first], nodes[first + 1], nodes[second], nodes[second + 1])
    lower, upper = int(first[np.argmin(chords)]), int(second[np.argmin(chords)])
    samples = np.linspace(0.0, 1.0, 65)
    under = np.array([_centerline(coordinates, lower, xi)[0] for xi in samples])
    over = np.array([_centerline(coordinates, upper, xi)[0] for xi in samples])
    between = np.linalg.norm(under[:, None] - over[None], axis=-1)
    nearest = np.unravel_index(np.argmin(between), between.shape)
    return dict(
        elements=elements,
        pair=(lower, upper),
        chords=float(chords.min()),
        distance=float(between.min()),
        rise=float(over[nearest[1], 2] - under[nearest[0], 2]),
        floor=float(_read(solver.contact_force)[:, 2].sum()),
        support=float(_residual(solver)[0, 0, 2]),
        weight=_weight(solver),
        speed=_speed(solver),
        unbalanced=_unbalanced(solver),
    )


class ThreadSelfContactTests(_ThreadCase):
    """A stretch of a thread that starts just above another stretch of it settles under its weight."""

    def test_self_contact_keeps_crossing_stretches_a_diameter_apart(self):
        """Fails if a stretch sinks into the stretch it lies on, or is held further off than the contact's onset."""
        crossing = _crossing(True)
        self.assert_solved(crossing)
        self.assertLessEqual(crossing["speed"], REST_SPEED)
        # The two elements belong to the first and the last stretch of the loop.
        lower, upper = crossing["pair"]
        self.assertLess(lower, crossing["elements"] // 2)
        self.assertGreater(upper, crossing["elements"] // 2)
        # A contact carries 120 µN at its contact distance, more than the thread's whole weight of 79 µN, so
        # each stretch stays a radius or more from the plane between them: the centres keep a diameter.
        self.assertGreaterEqual(crossing["distance"], 2.0 * RADIUS)
        # The crossing stretch is off the floor over several elements and presses on the other with more than a
        # node's weight, which a contact carries 38 µm off its contact distance; so does the stretch below.
        self.assertLessEqual(crossing["distance"], 2.0 * (RADIUS + NODE_STANDOFF))
        # It lies on top: the stretches cross level and at a right angle, so their distance is the rise from
        # one to the other.
        self.assertAlmostEqual(crossing["rise"], crossing["distance"], delta=SAMPLING)

    def test_without_self_contact_the_stretches_pass_through_each_other(self):
        """Fails if the scene above needs no self-contact: without it the lifted stretch must sink into the other."""
        crossing = _crossing(False)
        self.assert_solved(crossing)
        self.assertLessEqual(crossing["speed"], REST_SPEED)
        # Both centrelines lie at the same height on the floor and cut each other: closer than one radius.
        self.assertLess(crossing["distance"], RADIUS)
        self.assertAlmostEqual(crossing["rise"], 0.0, delta=SAMPLING)
        # The floor and the support of node 0 carry the thread's weight.
        balance = crossing["elements"] * TOLERANCE
        self.assertAlmostEqual(crossing["floor"] + crossing["support"], crossing["weight"], delta=balance)

    # Defect: the self-contact forces on the two stretches are not equal and opposite, so the floor carries 12 % more.
    @unittest.expectedFailure
    def test_thread_resting_on_itself_adds_no_weight(self):
        """Fails if the forces of self-contact on the two stretches do not cancel: the floor then carries more."""
        crossing = _crossing(True)
        self.assert_solved(crossing)
        balance = crossing["elements"] * TOLERANCE
        self.assertAlmostEqual(crossing["floor"] + crossing["support"], crossing["weight"], delta=balance)


# Bar: a rigid body 3 mm above the floor carries a capsule of 0.28 mm radius along its x axis. The body is turned
# a quarter about z, so the bar lies along the world's y, and its centre of mass is off its origin. A straight
# thread of 20 elements starts level just above the bar, crosses it at a right angle 4 mm from the body's origin,
# is held at node 0 and hangs down to the floor behind the bar. The body does not move.
BAR_HEIGHT = 0.003
BAR_RADIUS = 2.8e-4
BAR_AXIS = np.array([[-0.010, 0.0, 0.0], [0.0, 0.0, 0.0], [0.010, 0.0, 0.0]])  # in the body frame [m]
BAR_COM = np.array([0.002, 0.001, 0.0005])  # in the body frame [m]
BAR_MASS = 3.0e-4  # [kg]
BAR_INERTIA = 1.0e-8  # about every axis [kg m²]
# Damper of the support that holds the bar [N s/m, N m s]. With it the body resists a force within a step with
# K = m / h² + c / h = 2.3 N/m and a torque with K_r = I / h² + c_r / h = 1.2e-3 N m; without it with 1.1 N/m and
# 3.6e-5 N m.
BAR_DAMPING = (0.02, 2.0e-5)
BAR_CROSSING = 0.004  # the thread's y [m]
BAR_ELEMENTS = 20
BAR_STEPS = 240  # 4 s; the thread is at rest after 1 s
BAR_EARLY_STEP = 30  # 0.5 s: the bar already carries the thread and the wrench is still an eighth short of that
BAR_REACH = RADIUS + BAR_RADIUS  # between the thread's centre and the bar's axis when they touch [m]


def _relaxation(damping):
    """Fraction by which one reaction moves the body's wrench towards the thread's current one (capsules.py).

    The smaller of K / (K + k) and K_r / (K_r + k a²), with the contact stiffness k and the largest distance a of
    the capsule axis from the centre of mass, 12 mm. With the damper the force's fraction is the smaller, 0.067
    against 0.21; without it the torque's, 0.0077 against 0.033.
    """
    force = BAR_MASS / DT**2 + damping[0] / DT
    torque = BAR_INERTIA / DT**2 + damping[1] / DT
    reach = np.linalg.norm(BAR_AXIS - BAR_COM, axis=1).max()
    return min(force / (force + STIFFNESS), torque / (torque + STIFFNESS * reach**2))


@functools.cache
def _thread_over_bar():
    """Lay the thread over the bar; distances, forces on the thread and the wrench on the body at rest."""
    builder = newton.ModelBuilder()
    quarter_turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * math.pi)
    body = builder.add_body(
        xform=wp.transform((0.0, 0.0, BAR_HEIGHT), quarter_turn),
        mass=BAR_MASS,
        inertia=wp.mat33(*(BAR_INERTIA * np.eye(3)).ravel()),
        com=wp.vec3(*BAR_COM),
    )
    model = builder.finalize(DEVICE)
    state = model.state()
    # The thread starts a third of its diameter clear of the bar.
    start = (-0.5 * BAR_ELEMENTS * ELEMENT, BAR_CROSSING, BAR_HEIGHT + BAR_REACH + 1.0e-4)
    solver = _solver(
        model,
        _straight(BAR_ELEMENTS, start),
        gravity=(0.0, 0.0, -GRAVITY),
        friction=0.2,
        obstacle_distance=BAR_REACH,
        **_floor(),
    )
    contact = ThreadCapsuleContact(solver, model, body, BAR_AXIS, dt=DT, damping=BAR_DAMPING, skip=0)
    pose = _read(state.body_q)[body]
    com = np.array(wp.transform_point(wp.transform(*pose), wp.vec3(*model.body_com.numpy()[body])))

    def opposite():
        """The opposite of the bar's forces on the thread: a force and a torque about the body's centre of mass."""
        force, point = _read(solver.obstacle_force), _read(solver.obstacle_point)
        return np.r_[-force.sum(axis=0), np.cross(point - com, -force).sum(axis=0)]

    early = None
    for step in range(BAR_STEPS):
        before = _read(contact.wrench)[0]
        contact.update(state.body_q)
        solver.step(None, None, None, None, DT)
        contact.react(state.body_q)
        if step == BAR_EARLY_STEP:
            early = dict(before=before, wrench=_read(contact.wrench)[0], opposite=opposite())
    body_f = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=DEVICE)
    contact.add_forces(body_f)
    contact.add_forces(body_f)
    # A second contact on the same thread and body, without the damper, starts without a wrench and takes
    # one reaction.
    undamped = ThreadCapsuleContact(solver, model, body, BAR_AXIS, dt=DT, skip=0)
    undamped.react(state.body_q)

    # The thread's centreline, sampled every 63 µm, against the bar's axis, which runs in the world along y
    # through x = 0 at the bar's height.
    coordinates = _nodes(solver)[0][0]
    samples = np.linspace(0.0, 1.0, 33)
    centerline = np.array([_centerline(coordinates, e, xi)[0] for e in range(BAR_ELEMENTS) for xi in samples])
    ends = np.array([[0.0, BAR_AXIS[0, 0], BAR_HEIGHT], [0.0, BAR_AXIS[-1, 0], BAR_HEIGHT]])
    on_axis = np.clip(centerline[:, 1], ends[0, 1], ends[1, 1])[:, None] * [0.0, 1.0, 0.0] + [0.0, 0.0, BAR_HEIGHT]
    distance = np.linalg.norm(centerline - on_axis, axis=1)
    nearest = int(np.argmin(distance))
    nodes = coordinates[:, :3]
    return dict(
        distance=float(distance[nearest]),
        chords=float(_segment_distance(nodes[:-1], nodes[1:], ends[0], ends[1]).min()),
        above=float(centerline[nearest, 2] - BAR_HEIGHT),
        floor=_read(solver.contact_force).sum(axis=0),
        bar=_read(solver.obstacle_force).sum(axis=0),
        support=_residual(solver)[0, 0, :3],
        opposite=opposite(),
        wrench=_read(contact.wrench)[0],
        early=early,
        body_f=_read(body_f)[body],
        undamped=_read(undamped.wrench)[0],
        speed=_speed(solver),
        unbalanced=_unbalanced(solver),
    )


class ThreadCapsuleContactTests(_ThreadCase):
    """A thread laid over a bar on a rigid body, held at one end and reaching the floor with the other."""

    def test_thread_stays_outside_the_bar(self):
        """Fails if the thread sinks into the bar's capsule or falls through it to the floor."""
        scene = _thread_over_bar()
        self.assert_solved(scene)
        self.assertLessEqual(scene["speed"], REST_SPEED)
        # The contact carries 120 µN at the two radii, more than the thread's weight of 32 µN, so the thread's
        # centre stays that far or further from the bar's axis.
        self.assertGreaterEqual(scene["distance"], BAR_REACH)
        # The bar carries more than a node's weight, which a contact carries 38 µm off its contact distance.
        self.assertLessEqual(scene["distance"], BAR_REACH + NODE_STANDOFF)
        # It lies on the bar: its nearest point is above the top of the bar.
        self.assertGreaterEqual(scene["above"], BAR_RADIUS)

    def test_floor_bar_and_support_carry_the_weight(self):
        """Fails if the bar's forces on the thread do not make up for what the floor and the support do not carry."""
        scene = _thread_over_bar()
        self.assert_solved(scene)
        balance = BAR_ELEMENTS * TOLERANCE
        carried = scene["floor"] + scene["bar"] + scene["support"]
        np.testing.assert_allclose(carried, [0.0, 0.0, BAR_ELEMENTS * NODE_WEIGHT], rtol=0.0, atol=balance)
        # Each of the three carries a part: a node's weight or more.
        for part in ("floor", "bar", "support"):
            self.assertGreater(scene[part][2], NODE_WEIGHT)

    def assert_wrench(self, wrench, expected, relative):
        """A wrench equals ``expected`` to the fraction ``relative`` of its largest force and of its largest torque."""
        for part in (slice(0, 3), slice(3, 6)):
            tolerance = relative * np.abs(expected[part]).max()
            np.testing.assert_allclose(wrench[part], expected[part], rtol=0.0, atol=tolerance)

    def test_wrench_on_the_body_is_the_opposite_of_the_forces_on_the_thread(self):
        """Fails if the body's wrench has the sign of the thread's forces, a wrong lever or a wrong relaxation."""
        scene = _thread_over_bar()
        opposite = scene["opposite"]
        # The wrench is float32, 1.2e-7 per operation; a few operations make it.
        single = 1.0e-6
        # One reaction moves the wrench by the relaxation's fraction of the way to the opposite of the forces
        # on the thread: the force's fraction with the support's damper, the torque's without it.
        fraction = _relaxation(BAR_DAMPING)
        early = scene["early"]
        self.assertGreater(np.abs(early["opposite"] - early["before"])[2], NODE_WEIGHT)
        self.assert_wrench(early["wrench"], early["before"] + fraction * (early["opposite"] - early["before"]), single)
        self.assert_wrench(scene["undamped"], _relaxation((0.0, 0.0)) * opposite, single)
        # At the end the wrench is that opposite. Over the 209 steps since the early one it has closed all but
        # (1 - fraction)^209 of its distance; it lags the little that the thread's forces still changed in
        # that time, and it stops short where a step's share of the distance falls below its rounding.
        left = (1.0 - fraction) ** (BAR_STEPS - 1 - BAR_EARLY_STEP) + float(np.finfo(np.float32).eps) / fraction
        settling = max(
            np.abs(early["opposite"] - opposite)[part].max() / np.abs(opposite[part]).max()
            for part in (slice(0, 3), slice(3, 6))
        )
        self.assert_wrench(scene["wrench"], opposite, left + settling + single)
        # The thread presses the bar down, 2 mm beside the body's centre of mass along y: a torque about -x.
        self.assertLess(scene["wrench"][2], -NODE_WEIGHT)
        self.assertLess(scene["wrench"][3], 0.0)
        # Each call of add_forces adds the wrench to the body's forces.
        self.assert_wrench(scene["body_f"], 2.0 * scene["wrench"], single)


if __name__ == "__main__":
    unittest.main()
