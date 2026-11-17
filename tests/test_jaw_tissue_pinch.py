# SPDX-License-Identifier: Apache-2.0
"""The pinch of the tool/tissue exchange: tissue between the two jaws of a tool (``JawTissuePinch``).

The fixtures are small: a wall of tissue between two bodies with one capsule each (``_Fold``),
moved by writing their poses and velocities, and the exchange built from a ``ToolTissueSetup``
as a scene builds it. The tests run on the CPU, or on the device of ``NEWTON_TEST_DEVICE``.
"""

import math
import os
import unittest

import newton
import numpy as np
import warp as wp
from test_swept_tool_tissue_contact import _slab_mesh

from newton_solver_suturebot import SolverFEM
from newton_solver_suturebot._src.solvers.coupled.interface import jaw_tissue_pinch
from newton_solver_suturebot._src.solvers.coupled.interface import swept_tool_tissue_contact as swept
from newton_solver_suturebot._src.solvers.coupled.interface.jaw_tissue_pinch import (
    NOT_CLAMPING,
    STICK,
    JawTissuePinch,
    free_capsule_ends,
)
from newton_solver_suturebot._src.solvers.coupled.interface.tool_tissue_exchange import (
    ToolTissueExchange,
    ToolTissueSetup,
)
from newton_solver_suturebot._src.solvers.coupled.tet_step import _minimum_tet_motion
from newton_solver_suturebot._src.solvers.fem.kernels import eval_tetrahedra
from newton_solver_suturebot._src.solvers.fem.node_holds import HOLD_FACTOR

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
# Whether two runs of one scene are the same to the bit. On the CPU they are, and a run with the pinch is held
# against a run without it. On a CUDA device the sums of a kernel are taken in the order in which its threads
# arrive, which is not fixed, so that comparison is not made there; what the pinch writes is read inside one run
# instead (``_Fold.watch_pinch``), on every device.
_RUNS_REPEAT_TO_THE_BIT = not DEVICE.startswith("cuda")
MM = 1.0e-3
DT = 1.0 / 240.0
RADIUS = 0.5 * MM
# Held-tissue stiffness K_c of a tool [N/m] (the stiffest tool of the law's notes) and its effective mass at its
# jaws [kg]: K_c = MASS / DT^2.
STIFFNESS = 576.0
MASS = STIFFNESS * DT**2
# Friction between jaws and tissue of the fixtures.
FRICTION = 0.83
# Travel of a jaw per interval while it closes [m]: 3 mm/s, the speed of the examples.
CLOSE = 0.003 * DT


@wp.kernel(enable_backward=False)
def _predict(
    pose: wp.array[wp.transform],
    velocity: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    seconds: float,
    out: wp.array[wp.transform],
):
    out[wp.tid()] = swept._predicted_pose(pose[wp.tid()], velocity[wp.tid()], com[wp.tid()], seconds)


def _transform_points(pose, points):
    """Points of a body frame in the world, for a pose (xyz, xyzw), on the host."""
    vector, scalar = np.asarray(pose[3:6], dtype=float), float(pose[6])
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return (
        np.asarray(pose[:3], dtype=float) + points + 2.0 * np.cross(vector, np.cross(vector, points) + scalar * points)
    )


def _feet(points, start, end):
    """Closest points of a segment to points, and their places along it."""
    axis = end - start
    place = np.clip((points - start) @ axis / (axis @ axis), 0.0, 1.0)
    return start + place[:, None] * axis, place


class _Fold:
    """A wall of tissue between two jaw bodies with capsules along it; a third body stands apart.

    The wall is 3 mm thick (x from 0 to 3 mm), 8 mm long (y) and 6 mm high (z from -6 mm to
    0), its lowest layer fixed. Jaw A (body 0) stands on the side x < 0 and jaw B (body 1) on
    the side x > 3 mm; each carries the capsules of ``endpoints`` (in its frame, which is the
    world's but for the position of the body). Body 2 carries no capsule. The exchange is
    built from a ``ToolTissueSetup``, as a scene builds it.

    Args:
        friction: Friction of the pair with the tissue, or ``None`` for a tool without a pinch. A tuple
            gives one tool per entry: tool ``t`` has the jaw bodies ``2 t`` and ``2 t + 1`` with the same
            capsules, the tools after the first stand half a metre apart from the wall, and the
            body without a capsule is the last one.
        endpoints: End points of the capsule axes of one jaw in its frame [m], shape (capsules, 2, 3).
        mesh: Nodes, tetrahedra and surface triangles of the tissue; ``None`` is the wall.
        fixed: Fixed nodes of ``mesh``.
        heights: Heights of the two jaws [m]; ``None`` puts both at ``HEIGHT``.
    """

    HEIGHT = -2.0 * MM
    # The axis ends between two rows of nodes: no node lies exactly at an end.
    ENDPOINTS = np.array([[[0.0, 1.6 * MM, 0.0], [0.0, 6.4 * MM, 0.0]]])

    def __init__(self, friction=FRICTION, endpoints=None, mesh=None, fixed=None, radius=RADIUS, heights=None):
        if mesh is None:
            points, tetrahedra, surface = _slab_mesh(4, 9, 7, MM)
            fixed = np.flatnonzero(points[:, 2] < -5.5 * MM)
        else:
            points, tetrahedra, surface = mesh
        endpoints = self.ENDPOINTS if endpoints is None else np.asarray(endpoints, dtype=float)
        frictions = friction if isinstance(friction, tuple) else (friction,)
        tools = len(frictions)
        self.bodies = 2 * tools + 1
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for point in points:
            builder.add_particle(pos=tuple(point), vel=(0, 0, 0), mass=1.0e-6)
        for tet in tetrahedra:
            builder.add_tetrahedron(*(int(index) for index in tet), k_mu=2.0e4, k_lambda=8.0e4, k_damp=0.0)
        for _ in range(self.bodies):
            builder.add_body(mass=0.01, inertia=wp.mat33(1e-8, 0, 0, 0, 1e-8, 0, 0, 0, 1e-8))
        self.model = builder.finalize(device=DEVICE)
        self.rest = points
        self.tetrahedra = self.model.tet_indices.numpy()
        self.fixed = fixed
        self.solver = SolverFEM(
            self.model,
            dt=DT,
            fixed_particle_indices=fixed,
            preconditioner_type="block_diag",
            solver_type="cg",
            linear_solver_maxiter=400,
            min_stretch=0.2,
            max_stretch=5.0,
        )
        self.solver.linear_solver_tolerance = 1.0e-7
        self.state, self.next = self.model.state(), self.model.state()
        self.surface_points = wp.array(points.astype(np.float32), dtype=wp.vec3, device=DEVICE)
        self.triangles = wp.array(surface.reshape(-1).astype(np.int32), dtype=int, device=DEVICE)
        self.mesh = wp.Mesh(self.surface_points, self.triangles)
        self.step_seconds = wp.array([DT], dtype=wp.float64, device=DEVICE)
        count = len(endpoints)
        self.endpoints, self.radius = endpoints, radius
        self.setup = ToolTissueSetup(
            law="swept",
            capsule_bodies=np.repeat(np.arange(2 * tools), count),
            capsule_endpoints=np.concatenate([endpoints] * (2 * tools)),
            capsule_radii=np.full(2 * tools * count, radius),
            contact_radii=np.full(2 * tools * count, radius),
            capsule_tools=np.repeat(np.arange(tools, dtype=np.int64), 2 * count),
            jaw_bodies=tuple((2 * tool, 2 * tool + 1) for tool in range(tools)),
            reach=0.01,
            effective_mass=np.full(tools, MASS),
            fixed_particles=np.asarray(fixed),
            full_step=DT,
            tissue_friction=frictions,
        )
        self.exchange = ToolTissueExchange(
            self.model,
            self.solver,
            (self.mesh, self.triangles, wp.array(np.arange(len(points), dtype=np.int32), dtype=int, device=DEVICE)),
            self.setup,
            self.step_seconds,
            particle_q=self.state.particle_q,
        )
        self.law, self.pinch = self.exchange.law, self.exchange.pinch
        self.body_q = wp.zeros(self.bodies, dtype=wp.transform, device=DEVICE)
        self.body_qd = wp.zeros(self.bodies, dtype=wp.spatial_vector, device=DEVICE)
        self.body_f = wp.zeros(self.bodies, dtype=wp.spatial_vector, device=DEVICE)
        self.external = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=DEVICE)
        # Positions of the two jaws [m]: clear of the wall by a tenth of a radius, and nothing of body 2 near.
        low, high = (self.HEIGHT, self.HEIGHT) if heights is None else heights
        pair = np.array([[-1.1 * radius, 0.0, low], [3.0 * MM + 1.1 * radius, 0.0, high]])
        self.positions = np.concatenate([pair + (0.0, 0.0, 0.5 * tool) for tool in range(tools)] + [[[0.0, 0.0, 1.0]]])
        self.move(np.zeros((self.bodies, 3)))

    def move(self, travel):
        """Give the bodies the velocity of ``travel`` [m] per interval from where they stand (shape (bodies, 3) or (3,))."""
        travel = np.broadcast_to(np.asarray(travel, dtype=float), (self.bodies, 3))
        poses = np.zeros((self.bodies, 7), dtype=np.float32)
        poses[:, :3], poses[:, 6] = self.positions, 1.0
        rates = np.zeros((self.bodies, 6), dtype=np.float32)
        rates[:, :3] = travel / DT
        self.body_q.assign(poses)
        self.body_qd.assign(rates)
        self._travel = travel.copy()

    def update(self):
        """The update of the exchange at the accepted state, with the surface refit to it."""
        self.surface_points.assign(self.state.particle_q.numpy())
        self.mesh.refit()
        self.exchange.update(self.body_q, self.body_qd, self.state.particle_q, self.state.particle_qd)

    def step(self, travel=None, update=True):
        """One interval: the loads of the exchange, the tissue step, the bodies at their predicted poses, the update.

        ``travel`` [m] is what the bodies travel in the interval after this one (their velocity
        at the accepted state); ``None`` keeps the travel of the last call. ``update=False``
        stops at the accepted state, before the update of the exchange.
        """
        self.body_f.zero_()
        self.external.zero_()
        self.exchange.add_body_loads(self.body_f)
        self.exchange.add_nodal_loads(self.external)
        self.solver.step(self.state, self.next, None, None, DT, external_forces=self.external)
        self.state, self.next = self.next, self.state
        # The bodies are where the law predicted them.
        self.positions = self.positions + self._travel
        self.move(self._travel if travel is None else travel)
        if update:
            self.update()

    def close(self, depth, hold=8):
        """Close both jaws on the wall until each has pressed ``depth`` [m] into it, then hold for ``hold`` intervals."""
        gap = 0.1 * self.radius
        steps = int(math.ceil((gap + depth) / CLOSE))
        travel = np.array([[CLOSE, 0.0, 0.0], [-CLOSE, 0.0, 0.0], [0.0, 0.0, 0.0]]) * ((gap + depth) / (steps * CLOSE))
        self.move(travel)
        self.update()
        for index in range(steps):
            self.step(None if index < steps - 1 else np.zeros((3, 3)))
        for _ in range(hold):
            self.step()

    def lift(self, intervals, travel=None):
        """Move both jaws together for ``intervals`` intervals: straight up at the closing speed, or by ``travel`` [m]."""
        travel = np.array([0.0, 0.0, CLOSE]) if travel is None else np.asarray(travel, dtype=float)
        both = np.array([travel, travel, np.zeros(3)])
        self.move(both)
        self.update()
        for index in range(intervals):
            self.step(None if index < intervals - 1 else np.zeros((3, 3)))

    def open(self, intervals):
        """Move the jaws apart at the closing speed for ``intervals`` intervals, then stop."""
        apart = np.array([[-CLOSE, 0.0, 0.0], [CLOSE, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.move(apart)
        self.update()
        for index in range(intervals):
            self.step(None if index < intervals - 1 else np.zeros((3, 3)))

    def split_update(self):
        """The update of the exchange in its parts; returns the node positions after the law and after the pinch."""
        self.surface_points.assign(self.state.particle_q.numpy())
        self.mesh.refit()
        self.law.update(self.body_q, self.body_qd, self.state.particle_q, self.state.particle_qd)
        after_law = self.state.particle_q.numpy().astype(float)
        self.pinch.update(self.body_q, self.state.particle_q, self.state.particle_qd)
        self.law.holds.apply()
        return after_law, self.state.particle_q.numpy().astype(float)

    def guard(self):
        """The swept volume bound of the tetrahedra over the last tissue step (the guard of the coupled scenes)."""
        check = wp.array([np.inf, 0.0, 0.0], dtype=wp.float64, device=DEVICE)
        model = self.model
        wp.launch(
            _minimum_tet_motion,
            model.tet_count,
            [
                self.next.particle_q,
                self.state.particle_q,
                model.tet_indices,
                model.tet_poses,
                model.tet_materials,
                check,
            ],
            device=DEVICE,
        )
        return float(check.numpy()[0])

    def predicted(self):
        """The poses that the law predicts for the bodies at the end of the next interval (its own rule)."""
        out = wp.zeros(self.bodies, dtype=wp.transform, device=DEVICE)
        wp.launch(
            _predict,
            self.bodies,
            inputs=[self.body_q, self.body_qd, self.model.body_com, DT],
            outputs=[out],
            device=DEVICE,
        )
        return out.numpy().astype(float)

    def closing(self, particles, jaw):
        """By the definition, on the host: closing direction of nodes of ``jaw``, the foot places and the face gap."""
        poses = self.predicted()
        x = self.state.particle_q.numpy()[particles].astype(float)
        own = [_transform_points(poses[jaw], end)[0] for end in self.endpoints[0]]
        far = [_transform_points(poses[1 - jaw], end)[0] for end in self.endpoints[0]]
        foot_own, _ = _feet(x, *own)
        foot_far, place = _feet(x, *far)
        distance = np.linalg.norm(foot_far - foot_own, axis=1)
        return (foot_far - foot_own) / distance[:, None], place, distance - 2.0 * self.radius

    def craft(self, positions, nodes, forces=None, bound=1.0, ahead=None):
        """Write what the law publishes for chosen nodes, and run the pinch alone on it (no tissue step).

        ``positions`` are those of the bodies [m], at rest; ``nodes`` are tuples (particle,
        position [m], body that the law holds it on or -1 for a node that it has let go, push
        direction, pushed back). Every other node is as the law leaves a node without a row.
        ``forces`` gives elastic forces [N] by particle and ``bound`` the law's bound of a move
        back of every node [m]; ``ahead`` is what the law predicts the bodies to travel in the
        next interval [m] (nothing: the predicted poses are the accepted ones). The law's
        wrench is left as it is.
        """
        law = self.law
        self.positions = np.asarray(positions, dtype=float)
        self.move(np.zeros((self.bodies, 3)))
        predicted = self.body_q.numpy().copy()
        if ahead is not None:
            predicted[:, :3] += np.broadcast_to(np.asarray(ahead, dtype=float), (self.bodies, 3)).astype(np.float32)
        law._predicted.assign(predicted)
        q = self.state.particle_q.numpy()
        count = len(q)
        held, owner = np.zeros(count, dtype=np.int32), np.full(count, -1, dtype=np.int32)
        push, pushing = np.zeros((count, 3), dtype=np.float32), np.zeros(count, dtype=np.int32)
        for particle, position, body, direction, back in nodes:
            if particle not in law.nodes_host:
                raise ValueError("A crafted node must be a free surface node")
            q[particle] = position
            if body >= 0:
                held[particle], owner[particle], pushing[particle] = 1, body, int(back)
                push[particle] = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        self.state.particle_q.assign(q)
        for array, values in ((law.held, held), (law.owner, owner), (law.push, push), (law.compressive, pushing)):
            array.assign(values)
        law.balance.zero_()
        elastic = np.zeros((count, 3), dtype=np.float32)
        for particle, force in (forces or {}).items():
            elastic[particle] = force
        law._elastic.assign(elastic)
        law._allowed_back.fill_(bound)
        self.pinch.update(self.body_q, self.state.particle_q, self.state.particle_qd)
        law.holds.apply()

    def watch_pinch(self):
        """From now on, note :meth:`arrays` at every update as the law's own update left them.

        Returns the list that every update of the exchange appends to, just before the kernels
        of the pinch run: held against :meth:`arrays` after the update (the pinch, and the
        holds applied once more), it shows what the pinch wrote in that update, on any device.
        """
        noted, update = [], self.pinch.update

        def watched(*arrays):
            noted.append(self.arrays())
            return update(*arrays)

        self.pinch.update = watched
        return noted

    def arrays(self):
        """Every array that an update may write: particles, the law's state, the tissue matrix and its inverse blocks.

        Of the matrix, the blocks that it uses: its value storage is longer than its blocks,
        and what lies beyond them is memory that nothing has written (two runs can differ there).
        """
        state = self.state
        matrix = self.solver.A_bsr
        used = int(matrix.offsets.numpy()[matrix.nrow])
        arrays = [array.numpy().copy() for array in (state.particle_q, state.particle_qd, *self.law.state)]
        return [*arrays, matrix.values.numpy()[:used].copy(), self.solver._pc_inv_diag.numpy().copy()]


class CatchTests(unittest.TestCase):
    """P1 and P2: carried and clamped nodes, squeeze and pull; the catch writes nothing of the law or of the tissue."""

    def test_inner_face_nodes_are_carried_and_those_inside_the_cone_clamped(self):
        fold = _Fold()
        fold.close(0.3 * MM)
        law, pinch = fold.law, fold.pinch
        held, owner = law.held.numpy() != 0, law.owner.numpy()
        push, pushing = law.push.numpy().astype(float), law.compressive.numpy() != 0
        jaw, clamped = pinch.jaw.numpy(), pinch.clamped.numpy() != 0
        cone = 1.0 / math.sqrt(1.0 + FRICTION**2)
        report = pinch.diagnostics()
        for body in (0, 1):
            mine = np.flatnonzero(held & (owner == body))
            self.assertGreaterEqual(len(mine), 5)
            c, place, gap = fold.closing(mine, body)
            cosine = np.einsum("ij,ij->i", push[mine], c)
            # Carried: held on the half of the jaw that faces the other one.
            carried = mine[cosine > 0.0]
            np.testing.assert_array_equal(np.flatnonzero(jaw == body), carried)
            self.assertEqual(report["carried_nodes"][0][body], len(carried))
            # Clamped: of those, the ones that push back inside the friction cone, opposite the other jaw, faces apart.
            inside = (cosine >= cone) & pushing[mine] & (place > 0.0) & (place < 1.0) & (gap > 0.0)
            self.assertGreaterEqual(int(inside.sum()), 3)
            self.assertLess(int(inside.sum()), len(carried))
            np.testing.assert_array_equal(np.flatnonzero(clamped & (jaw == body)), mine[inside])
            self.assertEqual(report["clamped_nodes"][0][body], int(inside.sum()))
        self.assertEqual(report["state"], ["stick"])
        self.assertEqual(int(pinch.pair_state.numpy()[0]), STICK)

    def test_squeeze_is_the_law_wrench_along_the_closing_direction(self):
        fold = _Fold()
        fold.close(0.3 * MM)
        law, pinch = fold.law, fold.pinch
        wrench = law.wrench.numpy().astype(float)
        jaw = pinch.jaw.numpy()
        report = pinch.diagnostics()
        normals = []
        for body in (0, 1):
            c, _, _ = fold.closing(np.flatnonzero(jaw == body), body)
            mean = c.sum(axis=0) / np.linalg.norm(c.sum(axis=0))
            np.testing.assert_allclose(report["jaw_closing_direction"][0][body], mean, atol=1e-5)
            # The closing direction of A points at B: along +x for A, -x for B.
            self.assertGreater(mean[0] * (1 - 2 * body), 0.99)
            normals.append(max(0.0, -float(wrench[body, :3] @ mean)))
            self.assertGreater(normals[-1], 0.0)
        np.testing.assert_allclose(report["jaw_normal_N"][0], normals, rtol=1e-5)
        np.testing.assert_allclose(report["squeeze_N"][0], min(normals), rtol=1e-5)
        np.testing.assert_allclose(report["limit_N"][0], 2.0 * FRICTION * min(normals), rtol=1e-5)
        # The pull: the elastic forces of the carried nodes across the closing direction of their jaw.
        elastic = law._elastic.numpy().astype(float)
        total = np.zeros(3)
        for body in (0, 1):
            mean = np.asarray(report["jaw_closing_direction"][0][body])
            force = elastic[np.flatnonzero(jaw == body)]
            total += (force - np.outer(force @ mean, mean)).sum(axis=0)
        np.testing.assert_allclose(report["pull_N"][0], np.linalg.norm(total), rtol=1e-4, atol=1e-7)
        self.assertLessEqual(report["pull_N"][0], report["limit_N"][0])

    def test_the_friction_cone_decides_the_clamp(self):
        # A node on the inner face of jaw A, 30 degrees off the line to jaw B, pushing back, the faces apart: it is
        # clamped with a friction coefficient above the tangent of that angle and only carried with one below.
        angle = math.radians(30.0)
        for friction, expected in ((1.05 * math.tan(angle), 1), (0.95 * math.tan(angle), 0)):
            fold = _Fold(friction=friction)
            node = self._groove_node(fold, 2.5 * RADIUS, angle)
            self.assertEqual((int(fold.pinch.jaw.numpy()[node]), int(fold.pinch.clamped.numpy()[node])), (0, expected))
        # The fold between two jaws sticks; set_friction writes the device value that the next update reads: without
        # friction the limit 2 mu S is zero and the same pair slips under any pull.
        fold = _Fold()
        fold.close(0.3 * MM, hold=0)
        # While the jaws close nothing slips: an interval in which both jaws have a clamped node but the squeeze
        # is what rounding makes of none does not clamp (on any device). In the hold that follows the pair sticks
        # in every interval.
        slipped = fold.pinch.diagnostics()["slipping_intervals"][0]
        self.assertEqual(slipped, 0)
        for _ in range(8):
            fold.step()
            report = fold.pinch.diagnostics()
            self.assertEqual((report["state"], report["slipping_intervals"]), (["stick"], [slipped]))
        fold.pinch.set_friction(0.0)
        np.testing.assert_array_equal(fold.pinch.friction.numpy(), [0.0])
        fold.step()
        report = fold.pinch.diagnostics()
        self.assertEqual(
            (report["state"], report["slipping_intervals"], report["limit_N"]), (["slip"], [slipped + 1], [0.0])
        )
        self.assertGreater(report["squeeze_N"][0], 0.0)
        fold.pinch.set_friction(FRICTION)
        np.testing.assert_allclose(fold.pinch.friction.numpy(), [FRICTION], rtol=1e-6)
        fold.step()
        self.assertEqual(fold.pinch.diagnostics()["state"], ["stick"])
        for values in ([-1.0], [0.5, 0.5], [float("nan")]):
            with self.assertRaises(ValueError):
                fold.pinch.set_friction(values)

    def _groove_node(self, fold, separation, angle):
        """A node on the lower inner quarter of jaw A, ``angle`` [rad] below the line to jaw B, pushing back.

        The jaws stand side by side over the wall, their axes ``separation`` [m] apart; the
        node lies on the surface of A's capsule and is pushed along its normal there.
        """
        node = int(
            fold.law.nodes_host[np.argmin(np.linalg.norm(fold.rest[fold.law.nodes_host] - (1 * MM, 4 * MM, 0), axis=1))]
        )
        positions = np.array([[0.0, 0.0, 1 * MM], [separation, 0.0, 1 * MM], [0.0, 0.0, 1.0]])
        normal = np.array([math.cos(angle), 0.0, -math.sin(angle)])
        fold.craft(positions, [(node, positions[0] + (0.0, 4 * MM, 0.0) + RADIUS * normal, 0, normal, True)])
        return node

    def test_closed_jaws_never_clamp(self):
        # Closed jaws: their capsules reach across the plane in which they meet (here by a fifth of a radius).
        # Tissue in the groove under them lies on the lower inner quarter of a jaw, 30 degrees below the line to
        # the other jaw: inside the friction cone (39.7 degrees at 0.83), pushing back, the other jaw opposite.
        cone = math.atan(FRICTION)
        self.assertGreater(cone, math.radians(30.0))
        fold = _Fold()
        node = self._groove_node(fold, 1.8 * RADIUS, math.radians(30.0))
        self.assertEqual(int(fold.pinch.jaw.numpy()[node]), 0)
        self.assertEqual(int(fold.pinch.clamped.numpy()[node]), 0)
        # The same node between faces that are apart by a twentieth of a radius is clamped: only the faces decide.
        fold = _Fold()
        node = self._groove_node(fold, 2.05 * RADIUS, math.radians(30.0))
        self.assertEqual((int(fold.pinch.jaw.numpy()[node]), int(fold.pinch.clamped.numpy()[node])), (0, 1))
        # And steeper than the friction angle it is carried, not clamped, also between faces that are apart.
        fold = _Fold()
        node = self._groove_node(fold, 2.05 * RADIUS, cone + math.radians(2.0))
        self.assertEqual((int(fold.pinch.jaw.numpy()[node]), int(fold.pinch.clamped.numpy()[node])), (0, 0))
        # On the outer half of the jaw it is not carried at all.
        fold = _Fold()
        node = self._groove_node(fold, 2.05 * RADIUS, math.radians(100.0))
        self.assertEqual(int(fold.pinch.jaw.numpy()[node]), -1)

    def test_jaws_on_a_spacer_do_not_clamp_tissue_beside_under_or_thinner(self):
        # A rigid spacer of the needle's diameter between the jaws: their faces are 0.56 mm apart.
        spacer = 0.56 * MM
        separation = 2.0 * RADIUS + spacer
        fold = _Fold()
        nodes = fold.law.nodes_host
        top = nodes[np.abs(fold.rest[nodes, 2]) < 1e-9]
        a, b, c, d = (int(node) for node in top[:4])
        positions = np.array([[0.0, 0.0, 1 * MM], [separation, 0.0, 1 * MM], [0.0, 0.0, 1.0]])
        axis_a, axis_b = positions[0] + (0.0, 4 * MM, 0.0), positions[1] + (0.0, 4 * MM, 0.0)
        tip = positions[0] + (0.0, 6.4 * MM, 0.0)
        down, out, along = np.array([0.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        inward = np.array([1.0, 0.0, 0.0])
        fold.craft(
            positions,
            [
                # Against the outer side of jaw A, and of jaw B.
                (a, axis_a + RADIUS * out, 0, out, True),
                (b, axis_b - RADIUS * out, 1, -out, True),
                # Under jaw A: pushed straight down.
                (c, axis_a + RADIUS * down, 0, down, True),
                # Beyond the tip of jaw A, pushed along the jaw.
                (d, tip + RADIUS * along, 0, along, True),
            ],
        )
        report = fold.pinch.diagnostics()
        self.assertEqual((report["carried_nodes"], report["clamped_nodes"]), ([[0, 0]], [[0, 0]]))
        self.assertEqual(report["state"], ["not clamping"])
        # Tissue in the gap that is thinner than the spacer: it lies on the inner face of jaw A alone. That node is
        # clamped as far as jaw A can tell, but jaw B has none and the contact presses nothing on it: no clamp.
        fold = _Fold()
        wrench = np.zeros((3, 6), dtype=np.float32)
        wrench[0, 0] = -0.1
        fold.law.wrench.assign(wrench)
        fold.craft(positions, [(a, axis_a + RADIUS * inward, 0, inward, True)])
        report = fold.pinch.diagnostics()
        self.assertEqual((report["carried_nodes"], report["clamped_nodes"]), ([[1, 0]], [[1, 0]]))
        np.testing.assert_allclose(report["jaw_normal_N"][0], [0.1, 0.0], rtol=1e-6)
        self.assertEqual(
            (report["state"], report["squeeze_N"], report["clamping_intervals"]), (["not clamping"], [0.0], [0])
        )
        # The jaws on the spacer pressed into flat tissue from above, one capsule radius deep: the tissue of the
        # groove under them is pushed down and carried, not clamped, and no bit differs from the run without
        # the pinch.
        mesh = _slab_mesh(7, 9, 4, MM)
        fixed = np.flatnonzero(mesh[0][:, 2] < -2.5 * MM)
        folds = [_Fold(friction=friction, mesh=mesh, fixed=fixed) for friction in (FRICTION, None)]
        depth = RADIUS
        steps = int(math.ceil((0.2 * RADIUS + depth) / CLOSE))
        down = np.array([0.0, 0.0, -(0.2 * RADIUS + depth) / steps])
        for fold in folds:
            fold.positions = np.array(
                [
                    [3.0 * MM - 0.5 * separation, 0.0, 1.2 * RADIUS],
                    [3.0 * MM + 0.5 * separation, 0.0, 1.2 * RADIUS],
                    [0.0, 0.0, 1.0],
                ]
            )
            noted = None if fold.pinch is None else fold.watch_pinch()
            fold.move(down)
            fold.update()
            for index in range(steps + 8):
                fold.step(None if index < steps - 1 else np.zeros(3))
                if fold.pinch is not None:
                    self.assertEqual(fold.pinch.diagnostics()["state"], ["not clamping"])
                    # In this update the pinch wrote nothing: every array is as the law's own update left it.
                    for first, second in zip(noted[-1], fold.arrays(), strict=True):
                        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertGreater(int(folds[0].law.held.numpy().sum()), 4)
        report = folds[0].pinch.diagnostics()
        self.assertGreater(min(report["carried_nodes"][0]), 0)
        self.assertEqual((report["clamped_nodes"], report["clamping_intervals"]), ([[0, 0]], [0]))
        if _RUNS_REPEAT_TO_THE_BIT:
            for first, second in zip(folds[0].arrays(), folds[1].arrays(), strict=True):
                self.assertEqual(first.tobytes(), second.tobytes())

    def test_opposite_knows_the_ends_of_a_capsule_set(self):
        # Two cells in a row along the wall: they meet at y = 4 mm, where each end lies inside the other capsule.
        cells = np.array([[[0.0, 1.6 * MM, 0.0], [0.0, 4.0 * MM, 0.0]], [[0.0, 4.0 * MM, 0.0], [0.0, 6.4 * MM, 0.0]]])
        free = free_capsule_ends([0, 0, 1, 1], np.concatenate((cells, cells)), np.full(4, RADIUS))
        np.testing.assert_array_equal(free, [[1, 0], [0, 1], [1, 0], [0, 1]])
        # A third capsule on another body at the same place frees nothing and takes nothing.
        np.testing.assert_array_equal(free_capsule_ends([0, 1], cells, [RADIUS, RADIUS]), [[1, 1], [1, 1]])
        # Two rows of three cells side by side, a radius apart: only the first and the last end of each row are free.
        rows = np.array(
            [[[z, y, 0.0], [z, y + 2.0 * MM, 0.0]] for z in (0.0, RADIUS) for y in (0.0, 2.0 * MM, 4.0 * MM)]
        )
        np.testing.assert_array_equal(
            free_capsule_ends(np.zeros(6, dtype=int), rows, np.full(6, 0.8 * RADIUS)),
            [[1, 0], [0, 0], [0, 1], [1, 0], [0, 0], [0, 1]],
        )
        fold = _Fold(endpoints=cells)
        fold.close(0.3 * MM)
        jaw, clamped = fold.pinch.jaw.numpy(), fold.pinch.clamped.numpy() != 0
        rest = fold.rest
        for body, face in ((0, 0.0), (1, 3.0 * MM)):
            row = (np.abs(rest[:, 0] - face) < 1e-9) & (np.abs(rest[:, 2] - fold.HEIGHT) < 1e-9)
            # Across the meeting point of the two cells the node counts as opposite the other jaw.
            meeting = np.flatnonzero(row & (np.abs(rest[:, 1] - 4.0 * MM) < 1e-9))
            self.assertEqual(len(meeting), 1)
            self.assertTrue(clamped[meeting[0]] and jaw[meeting[0]] == body)
            # Beyond the free ends of the set (y = 1.6 mm and 6.4 mm) a node is carried, never clamped.
            beyond = np.flatnonzero(row & ((rest[:, 1] < 1.5 * MM) | (rest[:, 1] > 6.5 * MM)))
            self.assertGreater(int((jaw[beyond] == body).sum()), 0)
            self.assertFalse(clamped[beyond].any())
            inside = np.flatnonzero(row & (rest[:, 1] > 1.7 * MM) & (rest[:, 1] < 6.3 * MM))
            self.assertEqual(len(inside), 5)
            self.assertTrue(clamped[inside].all())

    def test_contract_with_the_swept_law(self):
        fold = _Fold()
        fold.close(0.3 * MM)
        law, pinch, model = fold.law, fold.pinch, fold.model
        particles, bodies = model.particle_count, model.body_count
        # What the pinch reads of the law: its published state and two scratch arrays of the same update.
        for name, dtype, count in (
            ("held", wp.int32, particles),
            ("owner", wp.int32, particles),
            ("push", wp.vec3, particles),
            ("compressive", wp.int32, particles),
            ("balance", wp.vec3, particles),
            ("wrench", wp.spatial_vector, bodies),
            ("_elastic", wp.vec3, particles),
            ("_predicted", wp.transform, bodies),
            ("_step_seconds", wp.float64, 1),
        ):
            array = getattr(law, name)
            self.assertEqual((array.dtype, array.shape), (dtype, (count,)), name)
        # The blocks of the tissue matrix without a hold, in the order of the law's nodes, and the two intervals:
        # the full one (a host number) and that of the accepted step (device data of the tissue solver).
        self.assertEqual((law.holds.pristine.dtype, law.holds.pristine.shape), (wp.mat33f, (law.node_count,)))
        np.testing.assert_array_equal(law.holds._nodes_host, law.nodes_host)
        self.assertEqual(law.full_step, DT)
        self.assertEqual(float(law._step_seconds.numpy()[0]), DT)
        # The tetrahedra at each of those nodes, as the pinch lists them: four times the tetrahedron plus the corner.
        offsets, codes = pinch._node_tet_offset.numpy(), pinch._node_tets.numpy()
        self.assertEqual((len(offsets), int(offsets[-1])), (law.node_count + 1, len(codes)))
        for index in (0, law.node_count // 2, law.node_count - 1):
            listed = codes[offsets[index] : offsets[index + 1]]
            node = int(law.nodes_host[index])
            np.testing.assert_array_equal(fold.tetrahedra[listed // 4, listed % 4], node)
            self.assertEqual(len(listed), int((fold.tetrahedra == node).sum()))
        self.assertEqual(len(law.nodes_host), law.node_count)
        self.assertIs(law.holds.held, law.held)
        # The predicted poses are those of the law's own rule for the poses and velocities of the update.
        np.testing.assert_allclose(law._predicted.numpy(), fold.predicted(), atol=1e-7)
        # The elastic forces and the bounds of a move are those of the accepted state as the update got it,
        # before the law placed nodes: evaluated again at the positions before an update.
        fold.step(update=False)
        before = wp.clone(fold.state.particle_q)
        fold.update()
        elastic = wp.zeros(particles, dtype=wp.vec3, device=DEVICE)
        wp.launch(
            eval_tetrahedra,
            model.tet_count,
            inputs=[
                before,
                fold.state.particle_qd,
                model.tet_indices,
                model.tet_poses,
                model.tet_materials,
                wp.float32(fold.solver.min_stretch),
                wp.float32(fold.solver.max_stretch),
            ],
            outputs=[elastic],
            device=DEVICE,
        )
        tolerance = 1e-9
        if not _RUNS_REPEAT_TO_THE_BIT:
            # The two evaluations add the tetrahedra of a node in different orders. Two float32 sums of n terms
            # in different orders differ by at most (n - 1) eps times the size of the terms; n is the largest
            # number of tetrahedra at a node, and the largest nodal force stands for the size of the terms.
            terms = int(np.bincount(model.tet_indices.numpy().reshape(-1)).max())
            tolerance = (terms - 1) * float(np.finfo(np.float32).eps) * float(np.abs(elastic.numpy()).max())
        np.testing.assert_allclose(law._elastic.numpy(), elastic.numpy(), rtol=1e-5, atol=tolerance)
        # A node without a row leaves the law's placement with nothing: its flags, push, owner and balance cleared.
        clear = np.setdiff1d(law.nodes_host, np.flatnonzero(law._row_depth.numpy() > -np.inf))
        self.assertGreater(len(clear), 0)
        self.assertFalse(law.held.numpy()[clear].any())
        np.testing.assert_array_equal(law.push.numpy()[clear], 0.0)
        np.testing.assert_array_equal(law.owner.numpy()[clear], -1)
        np.testing.assert_array_equal(law.balance.numpy()[clear], 0.0)
        # The grip is the law's: at rest in a squeeze its force on each jaw along the closing direction is positive.
        self.assertGreater(min(pinch.diagnostics()["jaw_normal_N"][0]), 0.0)
        # The exchange lists the pinch's arrays with the law's and loads its module by name.
        state = [array.ptr for array in fold.exchange.state]
        self.assertEqual(state[: len(law.state)], [array.ptr for array in law.state])
        self.assertEqual(state[len(law.state) :], [array.ptr for array in pinch.state])
        self.assertIn(jaw_tissue_pinch.__name__, [module for module, _ in fold.exchange.modules()])
        # A reset of the exchange leaves no node carried and nothing counted.
        fold.exchange.reset()
        report = pinch.diagnostics()
        self.assertEqual((report["carried_nodes"], report["clamping_intervals"]), ([[0, 0]], [0]))
        self.assertTrue((pinch.jaw.numpy() == -1).all())

    def test_pinch_storage_is_fixed_and_update_reads_nothing(self):
        fold = _Fold()
        fold.close(0.3 * MM, hold=0)
        pinch = fold.pinch
        names = [name for name, value in vars(pinch).items() if isinstance(value, wp.array)]
        self.assertGreater(len(names), 20)
        before = {name: getattr(pinch, name).ptr for name in names}
        read = wp.array.numpy

        def refused(self, *args, **kwargs):
            raise AssertionError("an update of the exchange read an array back")

        for _ in range(50):
            fold.surface_points.assign(fold.state.particle_q.numpy())
            fold.mesh.refit()
            wp.array.numpy = refused
            try:
                fold.exchange.update(fold.body_q, fold.body_qd, fold.state.particle_q, fold.state.particle_qd)
                fold.exchange.add_body_loads(fold.body_f)
                fold.exchange.add_nodal_loads(fold.external)
            finally:
                wp.array.numpy = read
        self.assertEqual({name: getattr(pinch, name).ptr for name in names}, before)
        self.assertEqual({name for name, value in vars(pinch).items() if isinstance(value, wp.array)}, set(names))

    def test_a_tool_without_jaws_or_friction_has_no_pinch(self):
        self.assertIsNone(_Fold(friction=None).exchange.pinch)
        fold = _Fold()
        fixed = (fold.model, fold.law, [0, 1], fold.setup.capsule_endpoints, [RADIUS] * 2)
        with self.assertRaises(ValueError):
            JawTissuePinch(*fixed, ((0, 0),), [0.5], [STIFFNESS])
        with self.assertRaises(ValueError):
            JawTissuePinch(*fixed, ((0, 2),), [0.5], [STIFFNESS])
        with self.assertRaises(ValueError):
            JawTissuePinch(*fixed, ((0, 1),), [-0.5], [STIFFNESS])
        # The held-tissue stiffness of every tool is finite and positive, one per tool.
        for stiffness in ([0.0], [float("nan")], [STIFFNESS, STIFFNESS]):
            with self.assertRaises(ValueError):
                JawTissuePinch(*fixed, ((0, 1),), [0.5], stiffness)
        # A second tool without jaws: its pair is empty, its state stays "not clamping".
        pinch = JawTissuePinch(*fixed, ((0, 1), ()), [0.5, 0.0], [STIFFNESS, STIFFNESS])
        self.assertEqual(pinch.diagnostics()["carried_nodes"], [[0, 0], []])
        self.assertEqual(int(pinch.pair_state.numpy()[1]), NOT_CLAMPING)


def _reaction(fold, read, positions):
    """P6 on the host, from what the law published: per jaw the force and torque of its carried nodes, and the
    parts of the force along the closing direction that the nodes held by the law give."""
    law, pinch = fold.law, fold.pinch
    jaw, owner = pinch.jaw.numpy(), law.owner.numpy()
    elastic, pushing = law._elastic.numpy().astype(float), law.compressive.numpy() != 0
    closing = pinch.body_closing.numpy().astype(float)
    wrench, along_parts = np.zeros((2, 6)), np.zeros((2, 3))
    for body in (0, 1):
        mine = np.flatnonzero(jaw == body)
        force = elastic[mine]
        returned = np.where(pushing[mine][:, None], np.einsum("ij,ij->i", force, read[mine])[:, None] * read[mine], 0.0)
        rest = force - returned
        along = np.outer(rest @ closing[body], closing[body])
        by_law = owner[mine] >= 0
        applied = (rest - along) + np.where(by_law[:, None], along, 0.0)
        wrench[body, :3] = applied.sum(axis=0)
        wrench[body, 3:] = np.cross(positions[mine] - fold.positions[body], applied).sum(axis=0)
        along_parts[body] = along[by_law].sum(axis=0)
    return wrench, along_parts


class CarryTests(unittest.TestCase):
    """P3, P4 and P6: carried tissue goes with the jaws, slips above the friction limit, and loads the tool."""

    UP = np.array([[0.0, 0.0, CLOSE], [0.0, 0.0, CLOSE], [0.0, 0.0, 0.0]])

    def test_carried_nodes_follow_their_jaw_across_the_closing_direction(self):
        fold = _Fold()
        fold.close(0.3 * MM)
        carried = np.flatnonzero(fold.pinch.jaw.numpy() >= 0)
        self.assertEqual(len(carried), 14)
        # The closing has slipped nothing (a squeeze that rounding makes of none does not clamp, on any device).
        closed = fold.pinch.diagnostics()
        slipped, length = closed["slipping_intervals"][0], closed["slip_length_m"][0]
        self.assertEqual((slipped, length), (0, 0.0))
        before = fold.state.particle_q.numpy().astype(float)
        # Up and along the jaws at once: both across the closing direction.
        travel = np.array([0.0, 0.4 * CLOSE, CLOSE])
        both = np.array([travel, travel, np.zeros(3)])
        fold.move(both)
        fold.update()
        jaw = fold.pinch.jaw.numpy()
        for index in range(34):
            fold.step(both if index < 29 else np.zeros((3, 3)), update=False)
            after_law, after_pinch = fold.split_update()
            # The pinch moves no node: every position is as the law's update left it, to the bit.
            self.assertEqual(after_pinch.tobytes(), after_law.tobytes())
            # A carried node gets the velocity of its way across the closing direction of its jaw: along that
            # direction it stays where the law put it.
            way = fold.state.particle_qd.numpy().astype(float) * DT
            directions = fold.pinch.body_closing.numpy().astype(float)
            for body in (0, 1):
                mine = carried[jaw[carried] == body]
                np.testing.assert_allclose(way[mine] @ directions[body], 0.0, atol=2e-9)
                if index < 28:
                    self.assertGreater(np.linalg.norm(way[mine], axis=1).min(), 0.5 * CLOSE)
        report = fold.pinch.diagnostics()
        # The pair stuck through the whole lift: no slipping interval and no slip length were added.
        self.assertEqual(
            (report["state"], report["slipping_intervals"], report["slip_length_m"]), (["stick"], [slipped], [length])
        )
        np.testing.assert_array_equal(np.flatnonzero(fold.pinch.jaw.numpy() >= 0), carried)
        after = fold.state.particle_q.numpy().astype(float)
        moved = after[carried] - before[carried]
        # Across the closing direction (the wall's y and z) the nodes went where their jaws went.
        np.testing.assert_allclose(moved[:, 1:], np.tile(30 * travel[1:], (len(carried), 1)), atol=2e-6)
        # The rest of the wall was pulled along: its top rose with the pinch, by less than the jaws.
        top = np.flatnonzero(np.abs(fold.rest[:, 2]) < 1e-9)
        rise = (after[top, 2] - fold.rest[top, 2]).mean()
        self.assertGreater(rise, 0.5 * 30 * CLOSE)

    def test_a_held_node_that_does_not_push_back_is_carried(self):
        # A node that the contact holds on the inner face of a jaw without a force against it (a collapsed
        # element behind it): carried, not clamped; clamped from the interval in which it pushes back.
        fold = _Fold()
        node = int(fold.law.nodes_host[0])
        positions = np.array([[0.0, 0.0, 1 * MM], [2.5 * RADIUS, 0.0, 1 * MM], [0.0, 0.0, 1.0]])
        inward = np.array([1.0, 0.0, 0.0])
        place = positions[0] + (0.0, 4 * MM, 0.0) + RADIUS * inward
        fold.craft(positions, [(node, place, 0, inward, False)])
        self.assertEqual((int(fold.pinch.jaw.numpy()[node]), int(fold.pinch.clamped.numpy()[node])), (0, 0))
        fold.craft(positions, [(node, place, 0, inward, True)])
        self.assertEqual((int(fold.pinch.jaw.numpy()[node]), int(fold.pinch.clamped.numpy()[node])), (0, 1))

    def test_a_node_the_law_lets_go_is_kept_in_place_along_the_closing_direction(self):
        fold = _Fold()
        fold.close(0.3 * MM)
        law, pinch = fold.law, fold.pinch
        fold.move(self.UP)
        fold.update()
        kept = []
        for _ in range(140):
            fold.step(self.UP, update=False)
            after_law, after_pinch = fold.split_update()
            report = pinch.diagnostics()
            kept = np.asarray(report["kept_particles"], dtype=int)
            if len(kept) and report["state"] == ["stick"]:
                break
        self.assertGreater(len(kept), 0)
        jaw = pinch.jaw.numpy()
        # Kept: carried, let go by the law (no owner, no push), held by the pinch with the balancing force.
        self.assertTrue((jaw[kept] >= 0).all())
        np.testing.assert_array_equal(law.held.numpy()[kept], 1)
        np.testing.assert_array_equal(law.owner.numpy()[kept], -1)
        np.testing.assert_array_equal(law.push.numpy()[kept], 0.0)
        np.testing.assert_array_equal(law.balance.numpy()[kept], -law._elastic.numpy()[kept])
        # The pinch moved none of them, and their velocity lies across the closing direction of their jaw: along
        # it they stay where they are.
        self.assertEqual(after_pinch.tobytes(), after_law.tobytes())
        way = fold.state.particle_qd.numpy().astype(float) * DT
        directions = pinch.body_closing.numpy().astype(float)
        for body in (0, 1):
            mine = kept[jaw[kept] == body]
            if len(mine):
                np.testing.assert_allclose(way[mine] @ directions[body], 0.0, atol=2e-9)
                self.assertGreater(np.linalg.norm(way[mine], axis=1).min(), 0.5 * CLOSE)
        # The hold block of each kept node is in the tissue matrix and in its inverse diagonal.
        holds = law.holds
        values, inverse = fold.solver.A_bsr.values.numpy(), fold.solver._pc_inv_diag.numpy()
        pristine, slots = holds.pristine.numpy(), holds._slots.numpy()
        for node in kept:
            index = int(np.flatnonzero(holds._nodes_host == node)[0])
            block = pristine[index].astype(np.float64)
            block = block + HOLD_FACTOR * np.trace(block) / 3.0 * np.eye(3)
            np.testing.assert_allclose(values[slots[index]], block, rtol=1e-6)
            np.testing.assert_allclose(inverse[node], np.linalg.inv(block), rtol=1e-3)

    def test_reaction_reaches_the_jaws_one_substep_late(self):
        fold = _Fold()
        fold.close(0.3 * MM)
        law, pinch = fold.law, fold.pinch
        fold.move(self.UP)
        fold.update()
        for _ in range(45):
            fold.step(self.UP)
        # One interval in its parts. Its loads are those of the update before: the pinch's wrench of that update.
        earlier = pinch.wrench.numpy().copy()
        fold.body_f.zero_()
        fold.exchange.add_body_loads(fold.body_f)
        np.testing.assert_allclose(fold.body_f.numpy(), law.wrench.numpy() + earlier, rtol=1e-6, atol=1e-12)
        self.assertGreater(np.abs(earlier[:2, :3]).max(), 1e-3)
        fold.step(self.UP, update=False)
        read = pinch.push_read.numpy().astype(float)
        _, positions = fold.split_update()
        self.assertEqual(pinch.diagnostics()["state"], ["stick"])
        expected, _ = _reaction(fold, read, positions)
        wrench = pinch.wrench.numpy().astype(float)
        np.testing.assert_allclose(wrench[:2, :3], expected[:, :3], rtol=2e-4, atol=2e-7)
        np.testing.assert_allclose(wrench[:2, 3:], expected[:, 3:], rtol=2e-4, atol=2e-9)
        np.testing.assert_array_equal(wrench[2], 0.0)
        # The tissue pulls the jaws down, and it pulls on both.
        self.assertLess(wrench[0, 2], -1e-3)
        self.assertLess(wrench[1, 2], -1e-3)
        # A node that the law holds gives the jaws its whole elastic force, the law's part and the pinch's
        # together; summed over both jaws with the kept nodes' parts across the closing direction, the pull
        # on the tool along the lift is the elastic force of the carried nodes along it.
        jaw = pinch.jaw.numpy()
        carried = np.flatnonzero(jaw >= 0)
        elastic = law._elastic.numpy().astype(float)
        self.assertEqual(pinch.diagnostics()["kept_nodes"], 0)
        pushing = law.compressive.numpy()[carried] != 0
        returned = np.where(
            pushing[:, None], np.einsum("ij,ij->i", elastic[carried], read[carried])[:, None] * read[carried], 0.0
        )
        np.testing.assert_allclose(
            wrench[:2, :3].sum(axis=0) + returned.sum(axis=0), elastic[carried].sum(axis=0), rtol=2e-4, atol=2e-7
        )

    def test_a_pull_above_two_mu_s_slips(self):
        fold = _Fold()
        fold.close(0.3 * MM)
        pinch = fold.pinch
        fold.move(self.UP)
        fold.update()
        for _ in range(30):
            fold.step(self.UP)
        report = pinch.diagnostics()
        self.assertEqual(report["state"], ["stick"])
        self.assertGreater(report["pull_N"][0], 0.3 * report["limit_N"][0])
        # The closing and the lift have slipped nothing so far, on any device.
        slipped, length = report["slipping_intervals"][0], report["slip_length_m"][0]
        self.assertEqual((slipped, length), (0, 0.0))
        # A tenth of the friction: the limit 2 mu S falls below the pull of this lift.
        pinch.set_friction(0.1 * FRICTION)
        fold.step(self.UP, update=False)
        read = pinch.push_read.numpy().astype(float)
        after_law, after_pinch = fold.split_update()
        report = pinch.diagnostics()
        self.assertEqual((report["state"], report["slipping_intervals"]), (["slip"], [slipped + 1]))
        self.assertGreater(report["pull_N"][0], report["limit_N"][0])
        # No node is moved by the pinch, and in an interval in which the pair slips none is carried: the nodes
        # that the law holds keep the zero velocity that it gave them.
        np.testing.assert_array_equal(after_pinch, after_law)
        marked = np.flatnonzero((pinch.jaw.numpy() >= 0) & (fold.law.owner.numpy() >= 0))
        self.assertGreater(len(marked), 4)
        np.testing.assert_array_equal(fold.state.particle_qd.numpy()[marked], 0.0)
        # Across the closing direction the jaws get the share 2 mu S / T of what the carried nodes would give
        # them in stick, the friction limit over the pull; along it the nodes that the law holds give their part.
        expected, along = _reaction(fold, read, after_pinch)
        wrench = pinch.wrench.numpy().astype(float)
        share = report["limit_N"][0] / report["pull_N"][0]
        self.assertLess(share, 0.5)
        np.testing.assert_allclose(wrench[:2, :3], along + share * (expected[:, :3] - along), rtol=2e-3, atol=2e-7)
        # The jaws went on by their travel of the interval: that is the slip.
        np.testing.assert_allclose(report["slip_length_m"][0] - length, CLOSE, rtol=0.05)

    def test_the_pair_keeps_and_drops_its_nodes_together(self):
        fold = _Fold()
        fold.close(0.3 * MM)
        law, pinch = fold.law, fold.pinch
        # Lifted until the law has let go of nodes that pull on their faces and the pair sticks: it keeps them.
        fold.move(self.UP)
        fold.update()
        for _ in range(140):
            fold.step(self.UP)
            report = pinch.diagnostics()
            if report["kept_nodes"] and report["state"] == ["stick"]:
                break
        self.assertEqual(report["state"], ["stick"])
        self.assertGreater(report["kept_nodes"], 0)
        # The jaws open at the top: as long as the pair sticks it keeps its nodes; from the interval in which it
        # does not, it keeps none. What is still marked then is held by the law.
        apart = np.array([[-CLOSE, 0.0, 0.0], [CLOSE, 0.0, 0.0], [0.0, 0.0, 0.0]])
        fold.move(apart)
        fold.update()
        states = []
        for _ in range(60):
            fold.step(apart)
            report = pinch.diagnostics()
            states.append(report["state"][0])
            if report["state"] != ["stick"]:
                self.assertEqual(report["kept_nodes"], 0)
                marked = np.flatnonzero(pinch.jaw.numpy() >= 0)
                self.assertTrue((law.owner.numpy()[marked] >= 0).all())
        self.assertIn("slip", states)
        self.assertEqual(states[-1], "not clamping")
        self.assertEqual(pinch.diagnostics()["carried_nodes"], [[0, 0]])

    def test_a_pair_that_never_clamps_changes_no_bit(self):
        # Jaws propped 0.56 mm apart press flat tissue with their inner lower quarters, one radius deep, and lift
        # off again under load: nodes are carried and released, the pair never clamps, and the run is the run
        # without a pinch in every bit: particles, the law's state, the tissue matrix and the forces on the bodies.
        separation = 2.0 * RADIUS + 0.56 * MM
        mesh = _slab_mesh(7, 9, 4, MM)
        fixed = np.flatnonzero(mesh[0][:, 2] < -2.5 * MM)
        folds = [_Fold(friction=friction, mesh=mesh, fixed=fixed) for friction in (FRICTION, None)]
        steps = int(math.ceil((0.2 * RADIUS + RADIUS) / CLOSE))
        down = np.array([0.0, 0.0, -(0.2 * RADIUS + RADIUS) / steps])
        seen, loads = 0, []
        for fold in folds:
            fold.positions = np.array(
                [
                    [3.0 * MM - 0.5 * separation, 0.0, 1.2 * RADIUS],
                    [3.0 * MM + 0.5 * separation, 0.0, 1.2 * RADIUS],
                    [0.0, 0.0, 1.0],
                ]
            )
            noted = None if fold.pinch is None else fold.watch_pinch()
            fold.move(down)
            fold.update()
            for index in range(2 * steps + 20):
                travel = down if index < steps - 1 else (np.zeros(3) if index < steps + 8 else -down)
                fold.step(travel)
                if fold.pinch is not None:
                    report = fold.pinch.diagnostics()
                    seen = max(seen, sum(report["carried_nodes"][0]))
                    self.assertEqual(
                        (report["state"], report["kept_nodes"], report["walking_nodes"]), (["not clamping"], 0, 0)
                    )
                    # In this update the pinch wrote nothing: every array is as the law's own update left it,
                    # and the loads of the exchange on the bodies are the law's alone.
                    for first, second in zip(noted[-1], fold.arrays(), strict=True):
                        self.assertEqual(first.tobytes(), second.tobytes())
                    fold.body_f.zero_()
                    fold.exchange.add_body_loads(fold.body_f)
                    of_law = wp.zeros_like(fold.body_f)
                    fold.law.add_forces(of_law)
                    self.assertEqual(fold.body_f.numpy().tobytes(), of_law.numpy().tobytes())
            fold.body_f.zero_()
            fold.exchange.add_body_loads(fold.body_f)
            loads.append(fold.body_f.numpy().copy())
        self.assertGreater(seen, 0)
        report = folds[0].pinch.diagnostics()
        self.assertEqual((report["clamping_intervals"], report["carried_nodes"]), ([0], [[0, 0]]))
        np.testing.assert_array_equal(folds[0].pinch.wrench.numpy(), 0.0)
        if _RUNS_REPEAT_TO_THE_BIT:
            self.assertEqual(loads[0].tobytes(), loads[1].tobytes())
            for first, second in zip(folds[0].arrays(), folds[1].arrays(), strict=True):
                self.assertEqual(first.tobytes(), second.tobytes())

    def _clamped_pair(self, fold, height=0.0, bound=1.0, forces=None, release=(), pulling=(), squeeze=0.1, ahead=None):
        """Two nodes on the inner faces of the two jaws, pushing back, and the contact pressing on both jaws.

        The jaws stand ``height`` [m] above where the nodes were caught and are predicted to
        travel ``ahead`` [m] in the next interval; a node of ``release`` is given as let go by
        the law, one of ``pulling`` as held by the law without a push back; the contact
        presses with ``squeeze`` [N] on each jaw. Returns the two particles.
        """
        nodes = [int(node) for node in fold.law.nodes_host[:2]]
        separation = 2.5 * RADIUS
        base = np.array([[0.0, 0.0, 1 * MM], [separation, 0.0, 1 * MM], [0.0, 0.0, 1.0]])
        inward = np.array([1.0, 0.0, 0.0])
        places = [base[0] + (0.0, 4 * MM, 0.0) + RADIUS * inward, base[1] + (0.0, 4 * MM, 0.0) - RADIUS * inward]
        wrench = np.zeros((3, 6), dtype=np.float32)
        wrench[0, 0], wrench[1, 0] = -squeeze, squeeze
        fold.law.wrench.assign(wrench)
        q = fold.state.particle_q.numpy().astype(float)
        crafted = []
        for index, (node, place, push) in enumerate(zip(nodes, places, (inward, -inward), strict=True)):
            position = q[node] if height else place
            crafted.append((node, position, -1 if node in release else index, push, node not in pulling))
        positions = base + np.array([[0.0, 0.0, height], [0.0, 0.0, height], [0.0, 0.0, 0.0]])
        fold.craft(positions, crafted, forces=forces, bound=bound, ahead=ahead)
        return nodes

    def test_the_carry_reports_how_far_tissue_is_from_its_places_and_how_far_they_travel(self):
        # P8: the two reports of the carry, over the carried nodes of a sticking pair that were carried while it
        # clamped before.
        fold = _Fold()
        pinch = fold.pinch
        self._clamped_pair(fold, ahead=(0.0, 0.0, 4e-6))
        report = pinch.diagnostics()
        # The interval in which the nodes are caught reports nothing, although the jaws are predicted to travel:
        # the places of the nodes are taken in it, at the predicted pose.
        self.assertEqual(
            (report["state"], report["carry_residual_m"], report["place_travel_m"]), (["stick"], [0.0], [0.0])
        )
        # The jaws stand 10 um above those places (14 um above where the nodes were caught) and are predicted to
        # rise 4 um more in the next interval: the nodes are 10 um from their places, across the closing
        # direction, and those travel 4 um.
        self._clamped_pair(fold, height=14e-6, ahead=(0.0, 0.0, 4e-6))
        report = pinch.diagnostics()
        self.assertEqual(report["state"], ["stick"])
        np.testing.assert_allclose(report["carry_residual_m"], [10e-6], atol=1e-7)
        np.testing.assert_allclose(report["place_travel_m"], [4e-6], atol=1e-7)
        reported = (report["carry_residual_m"], report["place_travel_m"])
        # A distance along the closing direction is not one from the place: that is the contact's side.
        self._clamped_pair(fold, height=14e-6)
        before = fold.state.particle_q.numpy().copy()
        base = np.array([[0.0, 0.0, 1 * MM], [2.5 * RADIUS, 0.0, 1 * MM], [0.0, 0.0, 1.0]])
        a, b = (int(node) for node in fold.law.nodes_host[:2])
        inward = np.array([1.0, 0.0, 0.0])
        wrench = np.zeros((3, 6), dtype=np.float32)
        wrench[0, 0], wrench[1, 0] = -0.1, 0.1
        fold.law.wrench.assign(wrench)
        fold.craft(
            base + (0.0, 0.0, 14e-6),
            [(a, before[a] + 50e-6 * inward, 0, inward, True), (b, before[b] + 50e-6 * inward, 1, -inward, True)],
        )
        report = pinch.diagnostics()
        self.assertEqual(
            (report["state"], report["carry_residual_m"], report["place_travel_m"]), (["stick"], *reported)
        )
        # A pair that slips or does not clamp reports nothing, however far its jaws have gone.
        pinch.set_friction(0.0)
        self._clamped_pair(fold, height=40e-6, ahead=(0.0, 0.0, 30e-6), forces={a: (0.0, 0.0, -0.01)})
        report = pinch.diagnostics()
        self.assertEqual((report["state"], report["carry_residual_m"], report["place_travel_m"]), (["slip"], *reported))
        pinch.set_friction(FRICTION)
        self._clamped_pair(fold, height=70e-6, ahead=(0.0, 0.0, 30e-6), squeeze=0.0)
        report = pinch.diagnostics()
        self.assertEqual(
            (report["state"], report["carry_residual_m"], report["place_travel_m"]), (["not clamping"], *reported)
        )
        # The reports are running maxima of the object's state: a measurement starts them anew between two
        # steps (nothing else of the pinch changes), and a reset clears them.
        self.assertIn(pinch.carry_residual.ptr, [array.ptr for array in pinch.state])
        self.assertIn(pinch.place_travel.ptr, [array.ptr for array in pinch.state])
        marks = [array.numpy().copy() for array in pinch.state]
        pinch.clear_reports()
        report = pinch.diagnostics()
        self.assertEqual((report["carry_residual_m"], report["place_travel_m"]), ([0.0], [0.0]))
        for array, before in zip(pinch.state, marks, strict=True):
            if array.ptr not in (pinch.carry_residual.ptr, pinch.place_travel.ptr):
                self.assertEqual(array.numpy().tobytes(), before.tobytes())
        # The interval without a clamp took the places again at the pose it predicted, 100 um up; with the jaws
        # at 80 um and the pair sticking, the nodes are 20 um from them.
        self._clamped_pair(fold, height=80e-6)
        report = pinch.diagnostics()
        self.assertEqual(report["state"], ["stick"])
        np.testing.assert_allclose(report["carry_residual_m"], [20e-6], atol=1e-7)
        fold.exchange.reset()
        report = pinch.diagnostics()
        self.assertEqual((report["carry_residual_m"], report["place_travel_m"]), ([0.0], [0.0]))

    def test_a_kept_node_gives_its_force_along_the_closing_direction_half_to_each_jaw(self):
        # P6: a carried node that the law has let go (kept by the pinch) gives its jaw its elastic force across
        # the closing direction of the jaw; its part along that direction goes half to each of the two jaws,
        # at the node.
        fold = _Fold()
        a, _ = self._clamped_pair(fold)
        force = np.array([0.02, 0.0, -0.03])
        self._clamped_pair(fold, forces={a: force}, release=(a,))
        report = fold.pinch.diagnostics()
        self.assertEqual((report["state"], report["kept_particles"]), (["stick"], [a]))
        closing = np.asarray(report["jaw_closing_direction"][0][0])
        np.testing.assert_allclose(closing, (1.0, 0.0, 0.0), atol=1e-6)
        wrench = fold.pinch.wrench.numpy().astype(float)
        np.testing.assert_allclose(wrench[0, :3], (0.01, 0.0, -0.03), rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(wrench[1, :3], (0.01, 0.0, 0.0), rtol=1e-6, atol=1e-9)
        # Torques about the centres of mass of the jaws, with the force at the node.
        x = fold.state.particle_q.numpy()[a].astype(float)
        for body in (0, 1):
            np.testing.assert_allclose(
                wrench[body, 3:], np.cross(x - fold.positions[body], wrench[body, :3]), rtol=1e-4, atol=1e-9
            )
        np.testing.assert_array_equal(wrench[2], 0.0)
        # The two jaws together get the whole force of the node.
        np.testing.assert_allclose(wrench[:2, :3].sum(axis=0), force, rtol=1e-6, atol=1e-9)
        # Held by the law, without a push back (the law returned nothing of it), the same node gives its whole force.
        self._clamped_pair(fold, forces={a: force}, pulling=(a,))
        self.assertEqual(fold.pinch.diagnostics()["kept_particles"], [])
        np.testing.assert_allclose(fold.pinch.wrench.numpy()[0, :3], force, rtol=1e-6, atol=1e-9)

    def test_a_held_node_that_pulls_gives_its_whole_force(self):
        # P6: the pinch leaves out what the law returned of a node's force, and the law returns the part along the
        # push direction only for a node that pushed back. A node that the law holds and that pulls on its face
        # (its force has a part along the push direction of the update before, +x here) gives the jaw all of it.
        fold = _Fold()
        a, _ = self._clamped_pair(fold)
        np.testing.assert_allclose(fold.pinch.push_read.numpy()[a], (1.0, 0.0, 0.0), atol=1e-7)
        force = np.array([0.02, 0.0, -0.03])
        self._clamped_pair(fold, forces={a: force}, pulling=(a,))
        self.assertEqual(fold.pinch.diagnostics()["state"], ["stick"])
        self.assertEqual((int(fold.law.held.numpy()[a]), int(fold.law.compressive.numpy()[a])), (1, 0))
        np.testing.assert_allclose(fold.pinch.wrench.numpy()[0, :3], force, rtol=1e-6, atol=1e-9)
        # The same node pushing back: the law has returned its part along the push direction, the pinch gives the rest.
        back = np.array([-0.02, 0.0, -0.03])
        self._clamped_pair(fold, forces={a: back})
        np.testing.assert_allclose(fold.pinch.wrench.numpy()[0, :3], (0.0, 0.0, -0.03), rtol=1e-6, atol=1e-9)

    def test_each_tool_slips_at_its_own_friction(self):
        # P2: two tools on one tissue with the same squeeze (0.1 N) and the same pull (30 mN). The friction of the
        # first (0.83) puts its limit 2 mu S at 166 mN, above the pull; that of the second (0.11, a number of this
        # test) at 22 mN, below it: the first sticks, the second slips, and its jaws feel their own limit.
        frictions = (FRICTION, 0.11)
        fold = _Fold(friction=frictions)
        nodes = [int(node) for node in fold.law.nodes_host[:4]]
        separation = 2.5 * RADIUS
        inward = np.array([1.0, 0.0, 0.0])
        positions = np.array(
            [
                [0.0, 0.0, 1 * MM],
                [separation, 0.0, 1 * MM],
                [0.0, 0.0, 3 * MM],
                [separation, 0.0, 3 * MM],
                [0.0, 0.0, 1.0],
            ]
        )
        wrench = np.zeros((5, 6), dtype=np.float32)
        wrench[[0, 2], 0], wrench[[1, 3], 0] = -0.1, 0.1
        fold.law.wrench.assign(wrench)
        pull = np.array([0.0, 0.0, -0.03])
        crafted = [
            (node, positions[body] + (0.0, 4 * MM, 0.0) + (1 - 2 * (body % 2)) * RADIUS * inward, body, push, True)
            for body, (node, push) in enumerate(zip(nodes, (inward, -inward, inward, -inward), strict=True))
        ]
        fold.craft(positions, crafted, forces={nodes[0]: pull, nodes[2]: pull})
        report = fold.pinch.diagnostics()
        self.assertEqual(report["state"], ["stick", "slip"])
        np.testing.assert_allclose(report["squeeze_N"], [0.1, 0.1], rtol=1e-6)
        np.testing.assert_allclose(report["pull_N"], [0.03, 0.03], rtol=1e-6)
        np.testing.assert_allclose(report["limit_N"], [2.0 * mu * 0.1 for mu in frictions], rtol=1e-6)
        self.assertEqual((report["sticking_intervals"], report["slipping_intervals"]), ([1, 0], [0, 1]))
        # The sticking jaw carries the pull, the slipping one its friction limit.
        reaction = fold.pinch.wrench.numpy().astype(float)
        np.testing.assert_allclose(reaction[0, :3], pull, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(reaction[2, :3], (2.0 * 0.11 * 0.1 / 0.03) * pull, rtol=1e-5, atol=1e-9)

    def _squeeze_floor(self, fold, carried, stiffness=STIFFNESS):
        """The smallest squeeze that clamps, on the host: u (K_c + n K_node), from the rest state of the fixture."""
        spacing = float(np.spacing(np.float32(np.abs(fold.rest).max())))
        x = fold.rest[fold.tetrahedra].astype(float)
        edges = np.stack([x[:, corner] - x[:, 0] for corner in (1, 2, 3)], axis=2)
        inverse = np.linalg.inv(edges)
        gradients = np.concatenate((-inverse.sum(axis=1, keepdims=True), inverse), axis=1)
        volume = np.abs(np.linalg.det(edges)) / 6.0
        # The Lame parameters of the fixture: k_lambda + 2 k_mu.
        modulus = 8.0e4 + 2.0 * 2.0e4
        node = np.zeros(len(fold.rest))
        np.add.at(node, fold.tetrahedra, modulus * volume[:, None] * (gradients**2).sum(axis=2))
        return spacing * (stiffness + carried * node[fold.law.nodes_host].max())

    def test_a_squeeze_that_rounding_can_make_of_none_does_not_clamp(self):
        fold = _Fold()
        pinch = fold.pinch
        parameters = pinch.parameters()
        floor = self._squeeze_floor(fold, 1)
        np.testing.assert_allclose(
            parameters["squeeze_floor_N"][0] + parameters["squeeze_floor_per_node_N"], floor, rtol=1e-5
        )
        # The floor is far below a squeeze that holds anything (0.1 N here) and above rounding (a spacing of the
        # coordinates against the stiffness of the tool alone).
        self.assertLess(floor, 1.0e-4)
        self.assertGreater(floor, parameters["position_spacing_m"] * STIFFNESS)
        # One carried node on each jaw: just below the floor the pair does not clamp, just above it sticks.
        for squeeze, state, counted in ((0.99 * floor, "not clamping", 0), (1.01 * floor, "stick", 1)):
            fold.exchange.reset()
            self._clamped_pair(fold, squeeze=squeeze)
            report = pinch.diagnostics()
            self.assertEqual((report["clamped_nodes"], report["state"]), ([[1, 1]], [state]))
            self.assertEqual(report["clamping_intervals"], [counted])
            np.testing.assert_allclose(report["squeeze_N"], [squeeze], rtol=1e-5)
        # Every carried node adds its stiffness: with a second node on jaw A the squeeze that clamped one node on
        # each jaw is below the floor, and the floor of two nodes decides.
        second = int(fold.law.nodes_host[2])
        inward = np.array([1.0, 0.0, 0.0])
        positions = np.array([[0.0, 0.0, 1 * MM], [2.5 * RADIUS, 0.0, 1 * MM], [0.0, 0.0, 1.0]])
        two = self._squeeze_floor(fold, 2)
        self.assertGreater(0.99 * two, 1.01 * floor)
        for squeeze, state in ((1.01 * floor, "not clamping"), (0.99 * two, "not clamping"), (1.01 * two, "stick")):
            fold.exchange.reset()
            a, b = (int(node) for node in fold.law.nodes_host[:2])
            wrench = np.zeros((3, 6), dtype=np.float32)
            wrench[0, 0], wrench[1, 0] = -squeeze, squeeze
            fold.law.wrench.assign(wrench)
            fold.craft(
                positions,
                [
                    (a, positions[0] + (0.0, 4 * MM, 0.0) + RADIUS * inward, 0, inward, True),
                    (second, positions[0] + (0.0, 5 * MM, 0.0) + RADIUS * inward, 0, inward, True),
                    (b, positions[1] + (0.0, 4 * MM, 0.0) - RADIUS * inward, 1, -inward, True),
                ],
            )
            report = pinch.diagnostics()
            self.assertEqual((report["carried_nodes"], report["state"]), ([[2, 1]], [state]))
        # The floor follows the stiffness of the tool: four times the effective mass, four times K_c.
        fold.exchange.set_effective_mass(np.array([4.0 * MASS]))
        stiffer = self._squeeze_floor(fold, 1, stiffness=4.0 * STIFFNESS)
        np.testing.assert_allclose(
            pinch.squeeze_floor.numpy()[0], parameters["position_spacing_m"] * 4.0 * STIFFNESS, rtol=1e-6
        )
        self.assertGreater(0.99 * stiffer, 1.01 * floor)
        for squeeze, state in (
            (1.01 * floor, "not clamping"),
            (0.99 * stiffer, "not clamping"),
            (1.01 * stiffer, "stick"),
        ):
            fold.exchange.reset()
            self._clamped_pair(fold, squeeze=squeeze)
            self.assertEqual(pinch.diagnostics()["state"], [state])

    def test_the_pull_is_taken_across_the_closing_direction_of_the_jaw(self):
        # P2: jaw B stands further along the wall than jaw A, so that a node of A beside the end of B has its own
        # closing direction 60 degrees off the line between the axes; with a second node of A opposite B the
        # closing direction of the jaw lies 30 degrees from that of either node.
        fold = _Fold()
        a, second, b = (int(node) for node in fold.law.nodes_host[:3])
        separation = 2.5 * RADIUS
        along = separation * math.tan(math.radians(60.0))
        # The first node 0.4 mm from the end of A's axis; B's axis begins ``along`` further.
        positions = np.array([[0.0, 0.0, 1 * MM], [separation, 0.4 * MM + along, 1 * MM], [0.0, 0.0, 1.0]])
        inward = np.array([1.0, 0.0, 0.0])
        wrench = np.zeros((3, 6), dtype=np.float32)
        wrench[0, 0], wrench[1, 0] = -0.1, 0.1
        own = np.array([math.cos(math.radians(60.0)), math.sin(math.radians(60.0)), 0.0])
        mean = np.array([math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 0.0])
        for direction, expected in ((own, 0.5), (mean, 0.0)):
            fold.exchange.reset()
            fold.law.wrench.assign(wrench)
            fold.craft(
                positions,
                [
                    (a, positions[0] + (0.0, 2.0 * MM, 0.0) + RADIUS * inward, 0, inward, True),
                    (second, positions[0] + (0.0, 5.0 * MM, 0.0) + RADIUS * inward, 0, inward, True),
                    (b, (separation, 5.0 * MM, 1 * MM) - RADIUS * inward, 1, -inward, True),
                ],
                forces={a: 0.05 * direction},
            )
            report = fold.pinch.diagnostics()
            self.assertEqual((report["carried_nodes"], report["clamped_nodes"]), ([[2, 1]], [[1, 1]]))
            np.testing.assert_allclose(fold.pinch._closing.numpy()[a], own, atol=1e-5)
            np.testing.assert_allclose(report["jaw_closing_direction"][0][0], mean, atol=1e-5)
            # Along the node's own closing direction the force has half of its size across that of its jaw;
            # along the jaw's it has nothing across it.
            np.testing.assert_allclose(report["pull_N"][0], expected * 0.05, atol=1e-7)
            self.assertEqual(report["state"], ["stick"])

    def test_stick_is_a_velocity_of_the_held_node_and_no_move(self):
        fold = _Fold()
        pinch = fold.pinch
        a, b = self._clamped_pair(fold)
        self.assertEqual(pinch.diagnostics()["state"], ["stick"])
        # Caught at their places: there is no way to go.
        np.testing.assert_array_equal(fold.state.particle_qd.numpy()[[a, b]], 0.0)
        before = fold.state.particle_q.numpy().copy()
        # The jaws stand 10 um above the places of the nodes and are predicted to rise 2.5 um more in the next
        # interval: each node gets the velocity that takes it the whole way in one full interval, whatever the
        # law's bound of a move back is (2 um here), and no position is written.
        self._clamped_pair(fold, height=10e-6, ahead=(0.0, 0.0, 2.5e-6), bound=2e-6)
        self.assertEqual(pinch.diagnostics()["state"], ["stick"])
        np.testing.assert_allclose(
            fold.state.particle_qd.numpy()[[a, b]] * DT, [[0.0, 0.0, 12.5e-6]] * 2, rtol=1e-4, atol=1e-9
        )
        self.assertEqual(fold.state.particle_q.numpy().tobytes(), before.tobytes())
        # The anchor is the place that the node is on its way to: with the jaws where they were predicted and
        # the node not there yet, the way is the same again.
        self._clamped_pair(fold, height=12.5e-6)
        np.testing.assert_allclose(
            fold.state.particle_qd.numpy()[[a, b]] * DT, [[0.0, 0.0, 12.5e-6]] * 2, rtol=1e-4, atol=1e-9
        )
        self.assertEqual(fold.state.particle_q.numpy().tobytes(), before.tobytes())
        self.assertEqual(pinch.diagnostics()["sticking_intervals"], [3])
        # A node that the law has let go is carried the same way, held by the pinch with the balancing force.
        force = np.array([0.0, 0.0, -0.01])
        self._clamped_pair(fold, height=12.5e-6, release=(a,), forces={a: force})
        self.assertEqual(pinch.diagnostics()["kept_particles"], [a])
        np.testing.assert_allclose(fold.state.particle_qd.numpy()[a] * DT, (0.0, 0.0, 12.5e-6), rtol=1e-4, atol=1e-9)
        np.testing.assert_allclose(fold.law.balance.numpy()[a], -force, rtol=1e-6)
        self.assertEqual(fold.state.particle_q.numpy().tobytes(), before.tobytes())

    def test_the_carry_is_over_the_full_interval_also_after_a_shorter_accepted_step(self):
        # The law predicts a full interval ahead after every accepted step, also after one that was halved: the
        # velocity of a carried node is its way over the full interval (a halved attempt then takes it half of the
        # way), not over the interval of the step that was accepted.
        fold = _Fold()
        a, b = self._clamped_pair(fold)
        fold.step_seconds.assign([0.5 * DT])
        self._clamped_pair(fold, height=10e-6, ahead=(0.0, 0.0, 2.5e-6))
        self.assertEqual(fold.pinch.diagnostics()["state"], ["stick"])
        np.testing.assert_allclose(
            fold.state.particle_qd.numpy()[[a, b]] * DT, [[0.0, 0.0, 12.5e-6]] * 2, rtol=1e-4, atol=1e-9
        )

    def test_the_carry_is_taken_across_the_closing_direction_of_the_jaw(self):
        # The jaws of test_the_pull_is_taken_across_the_closing_direction_of_the_jaw: the first node of jaw A has
        # its own closing direction 60 degrees off the line between the axes, its jaw's lies 30 degrees off it.
        fold = _Fold()
        a, second, b = (int(node) for node in fold.law.nodes_host[:3])
        separation = 2.5 * RADIUS
        along = separation * math.tan(math.radians(60.0))
        positions = np.array([[0.0, 0.0, 1 * MM], [separation, 0.4 * MM + along, 1 * MM], [0.0, 0.0, 1.0]])
        inward = np.array([1.0, 0.0, 0.0])
        wrench = np.zeros((3, 6), dtype=np.float32)
        wrench[0, 0], wrench[1, 0] = -0.1, 0.1
        crafted = [
            (a, positions[0] + (0.0, 2.0 * MM, 0.0) + RADIUS * inward, 0, inward, True),
            (second, positions[0] + (0.0, 5.0 * MM, 0.0) + RADIUS * inward, 0, inward, True),
            (b, (separation, 5.0 * MM, 1 * MM) - RADIUS * inward, 1, -inward, True),
        ]
        fold.law.wrench.assign(wrench)
        fold.craft(positions, crafted)
        # Both jaws go 10 um along the wall (y), a direction that neither closing direction is across.
        way = np.array([0.0, 10e-6, 0.0])
        fold.law.wrench.assign(wrench)
        fold.craft(positions + np.array([way, way, np.zeros(3)]), crafted)
        report = fold.pinch.diagnostics()
        self.assertEqual(report["state"], ["stick"])
        # (The 10 um of travel turn the two directions by less than a tenth of a degree.)
        mean = np.asarray(report["jaw_closing_direction"][0][0])
        np.testing.assert_allclose(mean, (math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 0.0), atol=2e-3)
        own = fold.pinch._closing.numpy()[a].astype(float)
        np.testing.assert_allclose(own, (math.cos(math.radians(60.0)), math.sin(math.radians(60.0)), 0.0), atol=4e-3)
        # Its velocity takes the node the way across the closing direction of its jaw: three quarters of the
        # travel along the wall, not the quarter that is across its own closing direction.
        carried = fold.state.particle_qd.numpy()[a].astype(float) * DT
        np.testing.assert_allclose(carried, way - (way @ mean) * mean, atol=2e-8)
        self.assertGreater(np.linalg.norm(carried - (way - (way @ own) * own)), 4e-6)
        np.testing.assert_allclose(carried[1], 7.5e-6, atol=5e-8)
        self.assertLess(abs((way - (way @ own) * own)[1] - 2.5e-6), 1e-7)

    def test_what_both_jaws_share_is_taken_along_the_closing_direction_of_the_jaw(self):
        # The jaws of test_the_carry_is_taken_across_the_closing_direction_of_the_jaw (the first node of jaw A has
        # its own closing direction 60 degrees off the line between the axes, its jaw's lies 30 degrees off it),
        # standing where the nodes were caught and predicted to go 10 um along the wall in the next interval,
        # both alike. What the two share along the closing direction of the JAW is carried with what is across
        # it: the node is given the whole travel. (Taken along its own closing direction it would be given
        # 5 um less along the wall.)
        fold = _Fold()
        a, second, b = (int(node) for node in fold.law.nodes_host[:3])
        separation = 2.5 * RADIUS
        along = separation * math.tan(math.radians(60.0))
        positions = np.array([[0.0, 0.0, 1 * MM], [separation, 0.4 * MM + along, 1 * MM], [0.0, 0.0, 1.0]])
        inward = np.array([1.0, 0.0, 0.0])
        wrench = np.zeros((3, 6), dtype=np.float32)
        wrench[0, 0], wrench[1, 0] = -0.1, 0.1
        crafted = [
            (a, positions[0] + (0.0, 2.0 * MM, 0.0) + RADIUS * inward, 0, inward, True),
            (second, positions[0] + (0.0, 5.0 * MM, 0.0) + RADIUS * inward, 0, inward, True),
            (b, (separation, 5.0 * MM, 1 * MM) - RADIUS * inward, 1, -inward, True),
        ]
        fold.law.wrench.assign(wrench)
        fold.craft(positions, crafted)
        way = np.array([0.0, 10e-6, 0.0])
        fold.law.wrench.assign(wrench)
        fold.craft(positions, crafted, ahead=np.array([way, way, np.zeros(3)]))
        self.assertEqual(fold.pinch.diagnostics()["state"], ["stick"])
        mean = fold.pinch.body_closing.numpy()[0].astype(float)
        own = fold.pinch._closing.numpy()[a].astype(float)
        self.assertGreater(math.degrees(math.acos(float(mean @ own))), 25.0)
        carried = fold.state.particle_qd.numpy()[a].astype(float) * DT
        np.testing.assert_allclose(carried, way, atol=5e-8)
        wrong = way - (way @ mean) * mean + (way @ own) * own
        self.assertGreater(np.linalg.norm(wrong - way), 4e-6)


class SharedTravelTests(unittest.TestCase):
    """P3 along the closing direction: clamped tissue makes the travel that both jaws share, and no other."""

    # Travel of a jaw in the next interval [m]: that of the examples' speed.
    D = CLOSE

    def _ways(self, ahead, release=()):
        """Grip two nodes between resting jaws, then predict the jaws to travel ``ahead`` [m] (per body, or one for
        both): the ways that the two nodes are given for the next interval [m], and the fold."""
        fold = _Fold()
        carry = CarryTests()
        a, b = carry._clamped_pair(fold)
        ahead = np.broadcast_to(np.asarray(ahead, dtype=float), (2, 3))
        carry._clamped_pair(fold, ahead=np.concatenate((ahead, np.zeros((1, 3)))), release=release)
        self.assertEqual(fold.pinch.diagnostics()["state"], ["stick"])
        np.testing.assert_allclose(fold.pinch.body_closing.numpy()[:2], [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], atol=1e-6)
        return fold.state.particle_qd.numpy()[[a, b]].astype(float) * DT, fold

    def test_a_travel_that_both_jaws_make_alike_moves_the_clamped_tissue_whole(self):
        # Both jaws travel alike, 30 degrees off the closing direction (x): each node is given the whole travel,
        # every component of it, on jaw A (which closes along +x) and on jaw B (along -x) alike.
        travel = self.D * np.array([math.cos(math.radians(30.0)), 0.0, math.sin(math.radians(30.0))])
        ways, fold = self._ways(travel)
        np.testing.assert_allclose(ways, [travel, travel], rtol=1e-4, atol=2e-9)
        # The anchors are the places that the nodes are on their way to: with the jaws arrived and the nodes not
        # moved (this fixture steps no tissue), the same way is left.
        positions = np.array([[0.0, 0.0, 1 * MM], [2.5 * RADIUS, 0.0, 1 * MM], [0.0, 0.0, 1.0]])
        positions[:2] += travel
        nodes = [int(node) for node in fold.law.nodes_host[:2]]
        inward = np.array([1.0, 0.0, 0.0])
        q = fold.state.particle_q.numpy().astype(float)
        fold.craft(positions, [(nodes[0], q[nodes[0]], 0, inward, True), (nodes[1], q[nodes[1]], 1, -inward, True)])
        left = fold.state.particle_qd.numpy()[nodes].astype(float) * DT
        np.testing.assert_allclose(left[:, [1, 2]], [travel[[1, 2]]] * 2, rtol=1e-4, atol=2e-9)
        # A node that the law has let go is carried the same way.
        ways, fold = self._ways(travel, release=(int(_Fold().law.nodes_host[0]),))
        self.assertEqual(fold.pinch.diagnostics()["kept_nodes"], 1)
        np.testing.assert_allclose(ways, [travel, travel], rtol=1e-4, atol=2e-9)
        # The other way along the closing direction: shared as well.
        ways, _ = self._ways(-travel)
        np.testing.assert_allclose(ways, [-travel, -travel], rtol=1e-4, atol=2e-9)

    def test_closing_opening_and_one_sided_travel_share_nothing_along_the_closing_direction(self):
        d = self.D
        x, z = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        # Closing and opening jaws travel in opposite senses along the closing direction; one jaw that closes or
        # opens while the other stands shares nothing with it. Along that direction the nodes are the law's: tissue
        # that the law cannot move (an element at the stretch limit) is not pushed on by the pinch.
        for ahead in ([d * x, -d * x], [-d * x, d * x], [d * x, 0.0 * x], [0.0 * x, -d * x], [-d * x, 0.0 * x]):
            ways, _ = self._ways(ahead)
            np.testing.assert_allclose(ways, 0.0, atol=2e-9)
        # With a lift on top of a closing: the lift, which is across the closing direction, and nothing along it.
        ways, _ = self._ways([d * x + d * z, -d * x + d * z])
        np.testing.assert_allclose(ways, [d * z, d * z], rtol=1e-4, atol=2e-9)
        # Both jaws the same way along the closing direction, jaw A three times as far: the nodes make what the
        # two share, the travel of the slower jaw; the rest is a closing of jaw A on the tissue.
        ways, _ = self._ways([3.0 * d * x, d * x])
        np.testing.assert_allclose(ways, [d * x, d * x], rtol=1e-4, atol=2e-9)
        ways, _ = self._ways([-d * x, -3.0 * d * x])
        np.testing.assert_allclose(ways, [-d * x, -d * x], rtol=1e-4, atol=2e-9)


class ReleaseTests(unittest.TestCase):
    """P7: tissue that leaves a pinch under load is let down by its own step; a pinch that never clamped holds nothing."""

    # The two nodes of the fixture's wall that lie on its two faces at the height of the jaws, at rest.
    INWARD = np.array([1.0, 0.0, 0.0])

    def _resting_pair(self, fold, forces=None, release=(), squeeze=0.1):
        """Two nodes at rest on the two faces of the wall, on the inner faces of jaws that just touch it.

        Nothing of the tissue is moved: the tetrahedra at the nodes are those of the wall at
        rest. The contact presses with ``squeeze`` [N] on each jaw; a node of ``release`` is
        given as let go by the law. Returns the two particles.
        """
        rest, height = fold.rest, fold.HEIGHT
        pair = [int(np.argmin(np.linalg.norm(rest - (face, 4.0 * MM, height), axis=1))) for face in (0.0, 3.0 * MM)]
        positions = np.array([[-RADIUS, 0.0, height], [3.0 * MM + RADIUS, 0.0, height], [0.0, 0.0, 1.0]])
        wrench = np.zeros((3, 6), dtype=np.float32)
        wrench[0, 0], wrench[1, 0] = -squeeze, squeeze
        fold.law.wrench.assign(wrench)
        crafted = [
            (node, rest[node], -1 if node in release else body, (1 - 2 * body) * self.INWARD, node not in release)
            for body, node in enumerate(pair)
        ]
        fold.craft(positions, crafted, forces=forces)
        return pair

    @staticmethod
    def _block(fold, node):
        """The block of ``node`` in the tissue matrix without a hold (host)."""
        return fold.law.holds.pristine.numpy()[int(np.flatnonzero(fold.law.nodes_host == node)[0])].astype(float)

    @staticmethod
    def _volume_share(fold, node, step):
        """On the host: the largest share of the volume of a tetrahedron at ``node`` that a move ``step`` of it takes."""
        x = fold.state.particle_q.numpy().astype(float)
        share = -np.inf
        for tet in fold.tetrahedra[(fold.tetrahedra == node).any(axis=1)]:
            edges = x[tet[1:]] - x[tet[0]]
            volume = np.linalg.det(edges)
            moved = x[tet].copy()
            moved[int(np.flatnonzero(tet == node)[0])] += step
            share = max(share, 1.0 - np.linalg.det(moved[1:] - moved[0]) / volume)
        return share

    def _lifted(self, intervals=60):
        """The fold closed and lifted as far as its grip holds without a slip (the pull at four fifths of the limit)."""
        fold = _Fold()
        fold.close(0.3 * MM)
        fold.lift(intervals)
        for _ in range(20):
            fold.step()
        return fold

    def test_a_released_node_is_let_down_by_its_own_step_limited_to_a_share_of_a_volume(self):
        fold = _Fold()
        law, pinch = fold.law, fold.pinch
        a, _ = self._resting_pair(fold)
        self.assertEqual((pinch.diagnostics()["state"], int(pinch.gripped.numpy()[a])), (["stick"], 1))
        block = self._block(fold, a)
        # A step into the wall squeezes the elements at the node: the share of a volume that it takes is linear
        # in its length.
        per_metre = self._volume_share(fold, a, self.INWARD)
        self.assertGreater(per_metre, 0.0)
        np.testing.assert_allclose(self._volume_share(fold, a, 2e-4 * self.INWARD), 2e-4 * per_metre, rtol=1e-9)
        before = fold.state.particle_q.numpy().copy()

        def released(step):
            """Grip the pair, then let the node go under the force whose own step is ``step``, with nothing clamping."""
            fold.exchange.reset()
            fold.state.particle_qd.zero_()
            self._resting_pair(fold)
            force = block @ step / DT**2
            self._resting_pair(fold, forces={a: force}, release=(a,), squeeze=0.0)
            self.assertEqual((pinch.diagnostics()["state"], int(pinch.jaw.numpy()[a])), (["not clamping"], -1))
            # No position is written, whatever happens to the node.
            self.assertEqual(fold.state.particle_q.numpy().tobytes(), before.tobytes())
            return force, fold.state.particle_qd.numpy()[a].astype(float) * DT

        # A step that takes a quarter of a volume is halved, one that takes half of it is quartered: the node is
        # held and moves by the step shortened to an eighth of the volume.
        for share in (0.25, 0.5):
            step = (share / per_metre) * self.INWARD
            force, way = released(step)
            self.assertEqual((int(pinch.walk.numpy()[a]), int(law.held.numpy()[a])), (1, 1))
            self.assertEqual(pinch.diagnostics()["walking_particles"], [a])
            np.testing.assert_allclose(way, (swept.MOVE_FRACTION / share) * step, rtol=2e-3, atol=1e-10)
            np.testing.assert_allclose(law.balance.numpy()[a], -force, rtol=1e-6)
        # In the next interval its step takes less than an eighth: the node is free, nothing is written for it.
        self._resting_pair(
            fold, forces={a: block @ ((0.12 / per_metre) * self.INWARD) / DT**2}, release=(a,), squeeze=0.0
        )
        self.assertEqual((int(pinch.walk.numpy()[a]), int(law.held.numpy()[a])), (0, 0))
        np.testing.assert_array_equal(law.balance.numpy()[a], 0.0)
        # A step within the share from the start: free at once.
        _, way = released((0.12 / per_metre) * self.INWARD)
        self.assertEqual((int(pinch.walk.numpy()[a]), int(law.held.numpy()[a])), (0, 0))
        np.testing.assert_array_equal(way, 0.0)
        # A step out of the wall opens every element at the node: it takes no volume and is not shortened,
        # however long it is (here a whole element).
        outward = -1.0 * MM * self.INWARD
        self.assertLessEqual(self._volume_share(fold, a, outward), 1e-9)
        _, way = released(outward)
        self.assertEqual((int(pinch.walk.numpy()[a]), int(law.held.numpy()[a])), (0, 0))
        np.testing.assert_array_equal(way, 0.0)
        # The law takes a node that is being let down: it is the law's.
        released((0.5 / per_metre) * self.INWARD)
        self.assertEqual(int(pinch.walk.numpy()[a]), 1)
        fold.craft(
            np.array([[-RADIUS, 0.0, fold.HEIGHT], [3.0 * MM + RADIUS, 0.0, fold.HEIGHT + 5 * MM], [0.0, 0.0, 1.0]]),
            [(a, fold.rest[a], 0, -self.INWARD, False)],
            forces={a: block @ ((0.5 / per_metre) * self.INWARD) / DT**2},
        )
        self.assertEqual((int(pinch.walk.numpy()[a]), int(pinch.jaw.numpy()[a])), (0, -1))

    def test_the_step_of_a_let_down_node_is_that_of_the_interval_of_the_accepted_step(self):
        # P7: the node's own step is h^2 A_ii^-1 f with the interval that the tissue matrix holds, the one of the
        # accepted step (the law's step_seconds): after a halved step the same force makes a quarter of the step.
        fold = _Fold()
        law, pinch = fold.law, fold.pinch
        a, _ = self._resting_pair(fold)
        block = self._block(fold, a)
        per_metre = self._volume_share(fold, a, self.INWARD)
        # A force whose step takes 0.4 of a volume with the full interval (held and let down) and 0.1 with half
        # of it (within the share: free).
        force = block @ ((0.4 / per_metre) * self.INWARD) / DT**2
        for seconds, let_down in ((DT, 1), (0.5 * DT, 0)):
            fold.exchange.reset()
            fold.state.particle_qd.zero_()
            fold.step_seconds.assign([DT])
            self._resting_pair(fold)
            fold.step_seconds.assign([seconds])
            self._resting_pair(fold, forces={a: force}, release=(a,), squeeze=0.0)
            self.assertEqual((int(pinch.walk.numpy()[a]), int(law.held.numpy()[a])), (let_down, let_down))

    def test_a_release_under_load_lets_nodes_down_and_they_come_free(self):
        fold = self._lifted()
        law, pinch = fold.law, fold.pinch
        apart = np.array([[-CLOSE, 0.0, 0.0], [CLOSE, 0.0, 0.0], [0.0, 0.0, 0.0]])
        fold.move(apart)
        fold.update()
        walked, last = set(), None
        for index in range(100):
            fold.step(apart if index < 59 else np.zeros((3, 3)), update=False)
            after_law, after_pinch = fold.split_update()
            self.assertEqual(after_pinch.tobytes(), after_law.tobytes())
            walking = np.asarray(pinch.diagnostics()["walking_particles"], dtype=int)
            if len(walking):
                last = index
                walked.update(walking.tolist())
                # A node that is let down: let go by the law, held by the pinch with the balancing force, with
                # the velocity of its own step shortened to an eighth of a volume.
                np.testing.assert_array_equal(law.held.numpy()[walking], 1)
                np.testing.assert_array_equal(law.owner.numpy()[walking], -1)
                np.testing.assert_array_equal(law.balance.numpy()[walking], -law._elastic.numpy()[walking])
                way = fold.state.particle_qd.numpy().astype(float) * DT
                for node in walking:
                    step = DT**2 * np.linalg.solve(self._block(fold, node), law._elastic.numpy()[node].astype(float))
                    share = self._volume_share(fold, node, step)
                    self.assertGreater(share, swept.MOVE_FRACTION)
                    np.testing.assert_allclose(way[node], (swept.MOVE_FRACTION / share) * step, rtol=2e-2, atol=2e-9)
                    self.assertLessEqual(self._volume_share(fold, node, way[node]), 1.02 * swept.MOVE_FRACTION)
        self.assertGreater(len(walked), 0)
        # Every node comes free, and before the jaws have stopped.
        self.assertLess(last, 59)
        self.assertEqual(pinch.diagnostics()["walking_nodes"], 0)
        np.testing.assert_array_equal(law.held.numpy()[sorted(walked)], 0)
        np.testing.assert_array_equal(pinch.walk.numpy(), 0)

    def test_only_a_gripped_node_is_let_down(self):
        fold = _Fold()
        law, pinch = fold.law, fold.pinch
        a, _ = self._resting_pair(fold)
        self.assertEqual(pinch.gripped.numpy()[a], 1)
        # A force whose own step takes a quarter of a volume at the node.
        step = (0.25 / self._volume_share(fold, a, self.INWARD)) * self.INWARD
        pull = self._block(fold, a) @ step / DT**2
        # A pair that clamped, then the law lets one of its nodes go while nothing clamps any more: it is let down.
        self._resting_pair(fold, forces={a: pull}, release=(a,), squeeze=0.0)
        self.assertEqual(pinch.diagnostics()["state"], ["not clamping"])
        self.assertEqual((int(pinch.jaw.numpy()[a]), int(pinch.walk.numpy()[a]), int(law.held.numpy()[a])), (-1, 1, 1))
        np.testing.assert_allclose(law.balance.numpy()[a], -pull, rtol=1e-6)
        # The same node of a pair that never clamped (the contact presses on neither jaw): carried, let go under
        # the same load, and neither held nor given a velocity.
        fold.exchange.reset()
        fold.state.particle_qd.zero_()
        self._resting_pair(fold, squeeze=0.0)
        self.assertEqual((int(pinch.jaw.numpy()[a]), int(pinch.gripped.numpy()[a])), (0, 0))
        self._resting_pair(fold, forces={a: pull}, release=(a,), squeeze=0.0)
        self.assertEqual((int(pinch.jaw.numpy()[a]), int(pinch.walk.numpy()[a]), int(law.held.numpy()[a])), (-1, 0, 0))
        np.testing.assert_array_equal(fold.state.particle_qd.numpy()[a], 0.0)
        np.testing.assert_array_equal(law.balance.numpy()[a], 0.0)

    def test_a_node_that_is_caught_again_while_it_is_let_down_is_not_let_down_later(self):
        fold = _Fold()
        law, pinch = fold.law, fold.pinch
        a, _ = self._resting_pair(fold)
        step = (0.25 / self._volume_share(fold, a, self.INWARD)) * self.INWARD
        pull = self._block(fold, a) @ step / DT**2
        # Gripped, then let go by the law under load while nothing clamps: the node is let down.
        self._resting_pair(fold, forces={a: pull}, release=(a,), squeeze=0.0)
        self.assertEqual((int(pinch.jaw.numpy()[a]), int(pinch.walk.numpy()[a])), (-1, 1))
        # The law takes it again on the inner face of jaw A: it is caught afresh and no longer let down.
        self._resting_pair(fold, squeeze=0.0)
        self.assertEqual(pinch.diagnostics()["state"], ["not clamping"])
        self.assertEqual(
            (int(pinch.jaw.numpy()[a]), int(pinch.walk.numpy()[a]), int(pinch.gripped.numpy()[a])), (0, 0, 0)
        )
        # The law lets it go under the same load; its pair has not clamped since the catch: the node is free,
        # neither held nor given a velocity, as a node that was never gripped.
        velocity = fold.state.particle_qd.numpy().copy()
        self._resting_pair(fold, forces={a: pull}, release=(a,), squeeze=0.0)
        self.assertEqual((int(pinch.jaw.numpy()[a]), int(pinch.walk.numpy()[a])), (-1, 0))
        self.assertEqual(int(law.held.numpy()[a]), 0)
        self.assertEqual(fold.state.particle_qd.numpy().tobytes(), velocity.tobytes())
        np.testing.assert_array_equal(law.balance.numpy()[a], 0.0)

    def test_a_release_under_load_inverts_no_element_and_is_resolved_by_the_step_guard(self):
        fold = self._lifted()
        self.assertEqual(fold.pinch.diagnostics()["state"], ["stick"])
        apart = np.array([[-CLOSE, 0.0, 0.0], [CLOSE, 0.0, 0.0], [0.0, 0.0, 0.0]])
        fold.move(apart)
        fold.update()
        tetrahedra = fold.tetrahedra
        smallest = np.inf
        for index in range(90):
            fold.step(apart if index < 59 else np.zeros((3, 3)))
            # The guard of the coupled scenes over the tissue step that just ran, and the state as it was placed.
            self.assertGreater(fold.guard(), 0.0)
            x = fold.state.particle_q.numpy().astype(np.float64)[tetrahedra]
            volume = np.linalg.det(x[:, 1:] - x[:, :1])
            self.assertGreater(volume.min(), 0.0)
            smallest = min(smallest, fold.guard())
        self.assertTrue(np.isfinite(smallest))
        self.assertEqual(fold.pinch.diagnostics()["carried_nodes"], [[0, 0]])

    def test_after_the_release_nothing_is_held_and_the_tissue_is_at_rest(self):
        fold = self._lifted()
        self.assertGreater(np.linalg.norm(fold.state.particle_q.numpy() - fold.rest, axis=1).max(), 0.7 * MM)
        fold.open(60)
        for _ in range(60):
            fold.step()
        report = fold.pinch.diagnostics()
        self.assertEqual((report["carried_nodes"], report["kept_nodes"], report["walking_nodes"]), ([[0, 0]], 0, 0))
        self.assertEqual(report["state"], ["not clamping"])
        np.testing.assert_array_equal(fold.law.held.numpy(), 0)
        np.testing.assert_array_equal(fold.pinch.wrench.numpy(), 0.0)
        # Within the remainder that the contact itself allows tissue that can yield (20 um).
        self.assertLess(np.linalg.norm(fold.state.particle_q.numpy() - fold.rest, axis=1).max(), 2.0e-5)

    def test_slip_is_reported_with_its_length_and_the_pull_stays(self):
        fold = _Fold()
        fold.close(0.3 * MM)
        pinch = fold.pinch
        # The closing has slipped nothing, on any device: the lift below is counted from it.
        closed = pinch.diagnostics()
        slipped, length = closed["slipping_intervals"][0], closed["slip_length_m"][0]
        self.assertEqual((slipped, length), (0, 0.0))
        up = CarryTests.UP
        fold.move(up)
        fold.update()
        slipping, pulls = 0, []
        for _ in range(100):
            fold.step(up)
            report = pinch.diagnostics()
            if report["state"] == ["slip"]:
                slipping += 1
                # While it slips the pull is above the limit, and the jaws still carry the limit: they are pulled down.
                self.assertGreater(report["pull_N"][0], report["limit_N"][0])
                self.assertLess(sum(force[2] for force in report["reaction_N"][0]), -0.5 * report["limit_N"][0])
                pulls.append(report["pull_N"][0])
        report = pinch.diagnostics()
        self.assertGreater(slipping, 5)
        self.assertEqual(report["slipping_intervals"], [slipped + slipping])
        # Each slipping interval adds the travel of the jaws in it to the slip length.
        np.testing.assert_allclose(report["slip_length_m"][0] - length, slipping * CLOSE, rtol=0.1)
        self.assertLess(report["slip_length_m"][0] - length, 100 * CLOSE)
        # The pull does not collapse when the pair slips: it stays near the limit at which it began.
        self.assertGreater(min(pulls), 0.8 * max(pulls))


if __name__ == "__main__":
    unittest.main()
