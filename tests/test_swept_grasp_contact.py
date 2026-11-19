# SPDX-License-Identifier: Apache-2.0
"""The swept tool/tissue contact in the worker thread and in the grasp scene: its switch, its calls, its start-up."""

import os
import unittest
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from test_swept_tool_tissue_contact import _slab_mesh

from newton_solver_suturebot import NeedleGeometry
from newton_solver_suturebot._src.solvers.coupled.interface.swept_tool_tissue_contact import (
    SweptToolTissueContact,
    capsule_pieces,
)
from newton_solver_suturebot._src.solvers.fem.node_holds import HOLD_FACTOR
from newton_solver_suturebot._src.solvers.fem.solver import SolverFEM
from newton_solver_suturebot._src.solvers.thread.worker_thread import WorkerThread, tool_tissue_law

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
# Grid spacing of the slab [m]: large enough for the pad to carry the loose thread of the worker.
MM = 1.0e-2
DT = 1.0 / 240.0
RADIUS = 0.5 * MM
MASS = 0.02
# Capsule axis of a tool body along its x, three grid spacings long.
ENDPOINTS = np.array([[[-1.5 * MM, 0.0, 0.0], [1.5 * MM, 0.0, 0.0]]])
# Far above the slab: nothing of a capsule that is parked there reaches the tissue.
PARKED = 0.3


def _slab_scene():
    """A slab of 8 x 8 x 2 grid spacings with its bottom fixed under a needle (body 0) and two tool bodies with one capsule each.

    What ``WorkerThread`` reads of a grasp scene, with the tissue solver that the swept contact holds nodes in.
    """
    points, tetrahedra, surface = _slab_mesh(9, 9, 3, MM)
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    for point in points:
        builder.add_particle(pos=tuple(point), vel=(0, 0, 0), mass=1.0e-6)
    for tet in tetrahedra:
        builder.add_tetrahedron(*(int(index) for index in tet), k_mu=2.0e4, k_lambda=8.0e4, k_damp=0.0)
    for index in range(3):
        builder.add_body(
            # The needle 15 mm above the middle of the slab, the tools above it: clear of the tissue.
            xform=wp.transform(wp.vec3(4 * MM, 4 * MM, 0.015 + 0.03 * index), wp.quat_identity()),
            mass=0.0003 if index == 0 else 0.01,
            inertia=wp.mat33(1e-8, 0, 0, 0, 1e-8, 0, 0, 0, 2e-8),
        )
    model = builder.finalize(device=DEVICE)
    fixed = np.flatnonzero(points[:, 2] < -1.5 * MM)
    solver = SolverFEM(
        model,
        dt=DT,
        fixed_particle_indices=fixed,
        preconditioner_type="block_diag",
        solver_type="cg",
        linear_solver_maxiter=200,
        min_stretch=0.2,
        max_stretch=5.0,
    )
    solver.linear_solver_tolerance = 1.0e-6
    reference = SimpleNamespace(
        vertices=points, global_vertex_indices=np.arange(len(points)), exterior_surface_triangles=surface
    )
    selection = SimpleNamespace(
        vertices=points, indices=model.tet_indices.numpy(), global_vertex_indices=np.arange(len(points))
    )
    graspers = [
        SimpleNamespace(capsule_bodies=(body,), capsule_endpoints=ENDPOINTS.copy(), capsule_radii=np.array([RADIUS]))
        for body in (1, 2)
    ]
    scene = SimpleNamespace(
        model=model,
        state_0=model.state(),
        reference=reference,
        selection=selection,
        geometry=NeedleGeometry.load(),
        needle_body=0,
        graspers=graspers,
        args=SimpleNamespace(young_modulus=300.0, shaft_friction=0.2),
        solver=solver,
        fixed_vertex_indices=fixed,
        sim_dt=DT,
    )
    config = SimpleNamespace(
        thread_elements=90, thread_length_m=0.18, thread_radius_m=0.00015, thread_modulus_pa=2e6, thread_iterations=5
    )
    return scene, config


def _press(state, body=1, depth=0.02 * MM, y=1 * MM):
    """Lay the capsule of a tool body on the top of the slab along x through (4, y) spacings, ``depth`` inside it, at rest."""
    poses = state.body_q.numpy()
    poses[body] = [4 * MM, y, RADIUS - depth, 0, 0, 0, 1]
    state.body_q.assign(poses)
    state.body_qd.zero_()


def _under(points, y=1 * MM):
    """Free surface nodes on the line at ``y`` whose parts lie under a capsule axis from x = 2.5 to 5.5 spacings."""
    return np.flatnonzero(
        (np.abs(points[:, 2]) < 1e-9)
        & (np.abs(points[:, 1] - y) < 1e-9)
        & (points[:, 0] > 1.5 * MM)
        & (points[:, 0] < 6.5 * MM)
    )


def _diagonal_blocks(solver, nodes):
    """Diagonal blocks of the assembled tissue matrix at ``nodes``."""
    matrix = solver.A_bsr
    offsets = matrix.offsets.numpy()[: matrix.nrow + 1]
    columns = matrix.columns.numpy()[: offsets[-1]]
    values = matrix.values.numpy()
    blocks = []
    for node in nodes:
        slot = offsets[node] + int(np.flatnonzero(columns[offsets[node] : offsets[node + 1]] == node)[0])
        blocks.append(values[slot])
    return np.asarray(blocks)


def _bits(array):
    return str(array.dtype), tuple(array.shape), array.numpy().tobytes()


class ToolTissueLawTests(unittest.TestCase):
    def test_settings_name_no_contact_the_first_law_or_the_swept_law(self):
        for value in (None, False, "off"):
            self.assertIsNone(tool_tissue_law(value))
        for value in (True, "first"):
            self.assertEqual(tool_tissue_law(value), "first")
        self.assertEqual(tool_tissue_law("swept"), "swept")
        for value in ("on", "sweep", 1.5):
            with self.assertRaises(ValueError):
                tool_tissue_law(value)


class WorkerThreadSweptTests(unittest.TestCase):
    """The swept contact as the worker thread builds and calls it, on a slab with a real tissue solver."""

    def test_constants_follow_from_the_scene_and_the_pieces_are_those_of_the_tool_capsules(self):
        scene, config = _slab_scene()
        worker = WorkerThread(scene, config, tool_tissue="swept", tool_effective_mass=MASS)
        law = worker.swept_tool_tissue
        self.assertIsInstance(law, SweptToolTissueContact)
        self.assertIsNone(worker.tool_tissue)
        # Sweep horizon: the full interval of the scene. K_c = m / h^2 in every direction, for both tools.
        self.assertEqual(law.full_step, DT)
        np.testing.assert_allclose(law.held_stiffness_range, MASS / DT**2, rtol=1e-12)
        self.assertEqual(worker.tool_effective_mass, [MASS, MASS])
        np.testing.assert_allclose(law._inverse_mass.numpy(), np.tile(np.eye(3) / MASS, (14, 1, 1)), rtol=1e-6)
        # Reach: the thickness of the pad. Stretch limit: the tissue solver's.
        self.assertAlmostEqual(law.reach, 2 * MM, delta=1e-12)
        self.assertEqual(law.min_stretch, scene.solver.min_stretch)
        self.assertIs(law.solver, scene.solver)
        self.assertIs(law._step_seconds, worker.solver.step_time.seconds_f64)
        mesh, triangles, point_particle = worker.tissue.surface
        self.assertIs(law.mesh, mesh)
        self.assertIs(law._triangles, triangles)
        self.assertIs(law._point_particle, point_particle)
        # The two tool capsules and not the needle: three spacings at a radius of half a spacing, seven pieces each.
        bodies, _, _, _, radii, _, _, counts = capsule_pieces([1, 2], np.tile(ENDPOINTS, (2, 1, 1)), [RADIUS] * 2)
        self.assertEqual(counts.tolist(), [7, 7])
        np.testing.assert_array_equal(law._sample_body.numpy(), bodies)
        np.testing.assert_allclose(law._sample_radius.numpy(), radii)
        np.testing.assert_array_equal(law._sample_tool.numpy(), [0] * 7 + [1] * 7)
        self.assertEqual(worker.tool_tissue_radii, [RADIUS, RADIUS])
        # Every free node of the top surface can be held; the fixed bottom cannot.
        free_surface = np.setdiff1d(np.unique(scene.reference.exterior_surface_triangles), scene.fixed_vertex_indices)
        np.testing.assert_array_equal(np.sort(law.nodes_host), free_surface)
        # The state of the worker ends with the contact's.
        self.assertEqual([array.ptr for array in worker.state[-len(law.state) :]], [array.ptr for array in law.state])
        # The solver built its tangent for the holds: at the full interval of the scene.
        self.assertIsNotNone(scene.solver._tangent_triplet_slots)
        self.assertIs(scene.solver._node_holds, law.holds)

    def test_radii_of_the_drawn_tool_replace_the_capsule_radii_and_every_piece_has_the_mass_of_its_tool(self):
        scene, config = _slab_scene()
        worker = WorkerThread(
            scene,
            config,
            tool_tissue="swept",
            tool_tissue_radii=[0.8 * MM, 0.6 * MM],
            tool_effective_mass=[MASS, 2 * MASS],
        )
        law = worker.swept_tool_tissue
        self.assertEqual(worker.tool_tissue_radii, [0.8 * MM, 0.6 * MM])
        radii, bodies = law._sample_radius.numpy(), law._sample_body.numpy()
        np.testing.assert_allclose(radii[bodies == 1], 0.8 * MM)
        np.testing.assert_allclose(radii[bodies == 2], 0.6 * MM)
        # K_c = m / h^2 of the tool that a piece belongs to.
        self.assertEqual(worker.tool_effective_mass, [MASS, 2 * MASS])
        inverse = law._inverse_mass.numpy()
        np.testing.assert_allclose(
            inverse[bodies == 1], np.tile(np.eye(3) / MASS, ((bodies == 1).sum(), 1, 1)), rtol=1e-6
        )
        np.testing.assert_allclose(
            inverse[bodies == 2], np.tile(np.eye(3) / (2 * MASS), ((bodies == 2).sum(), 1, 1)), rtol=1e-6
        )
        np.testing.assert_allclose(law.held_stiffness_range, (MASS / DT**2, 2 * MASS / DT**2), rtol=1e-12)

    def test_missing_mass_wrong_radii_and_unknown_laws_are_refused(self):
        scene, config = _slab_scene()
        with self.assertRaises(ValueError):
            WorkerThread(scene, config, tool_tissue="swept")
        with self.assertRaises(ValueError):
            WorkerThread(scene, config, tool_tissue="swept", tool_tissue_radii=[RADIUS], tool_effective_mass=MASS)
        for masses in ([MASS], [MASS, 0.0], [MASS, MASS, MASS]):
            with self.assertRaises(ValueError):
                WorkerThread(scene, config, tool_tissue="swept", tool_effective_mass=masses)
        with self.assertRaises(ValueError):
            WorkerThread(scene, config, tool_tissue="sweep", tool_effective_mass=MASS)

    def test_without_the_option_the_worker_has_no_swept_contact_and_reads_no_tissue_solver(self):
        scene, config = _slab_scene()
        del scene.solver, scene.fixed_vertex_indices, scene.sim_dt
        for options in ({}, {"tool_tissue": False}, {"tool_tissue": "off"}):
            worker = WorkerThread(scene, config, **options)
            self.assertIsNone(worker.swept_tool_tissue)
            self.assertIsNone(worker.tool_tissue)
            expected = (*worker.solver.state, *worker.tissue.state, worker.attachment.wrench, worker._nodal_force)
            self.assertEqual([array.ptr for array in worker.state], [array.ptr for array in expected])
            accepted = scene.state_0
            self.assertEqual(
                [array.ptr for array in worker._accepted_arrays(accepted)],
                [accepted.body_q.ptr, accepted.body_qd.ptr, accepted.particle_q.ptr],
            )

    def test_accepted_step_places_and_holds_the_nodes_and_the_loads_join_the_threads(self):
        scene, config = _slab_scene()
        depth = 0.02 * MM
        _press(scene.state_0, depth=depth)
        worker = WorkerThread(scene, config, tool_tissue="swept", tool_effective_mass=MASS)
        law, solver = worker.swept_tool_tissue, scene.solver
        rest = scene.reference.vertices
        under = _under(rest)
        self.assertEqual(len(under), 5)
        pristine = _diagonal_blocks(solver, under)
        np.testing.assert_array_equal(law.held.numpy(), 0)
        # As in a scene, the tissue has stepped before the thread first advances: the solver has then made
        # the copies of its matrix that its solves read, which the holds are bound to. The step is discarded;
        # at rest it moves a node by the rounding of its elastic forces, far below a micrometre.
        following = scene.model.state()
        solver.step(scene.state_0, following, scene.model.control(), None, DT)
        np.testing.assert_allclose(following.particle_q.numpy(), rest, rtol=0, atol=1e-7)
        worker.advance(DT)
        # The five nodes under the capsule of body 1 are placed on it, at rest, and held; body 2 is far above.
        np.testing.assert_array_equal(np.flatnonzero(law.held.numpy()), under)
        moved = scene.state_0.particle_q.numpy() - rest
        np.testing.assert_allclose(moved[under], np.tile((0.0, 0.0, -depth), (5, 1)), rtol=3e-3, atol=2e-9)
        others = np.setdiff1d(np.arange(len(rest)), under)
        np.testing.assert_array_equal(scene.state_0.particle_q.numpy()[others], rest[others].astype(np.float32))
        np.testing.assert_array_equal(scene.state_0.particle_qd.numpy()[under], 0.0)
        np.testing.assert_array_equal(law.owner.numpy()[under], 1)
        # Their holds are in the tissue matrix: the diagonal block plus HOLD_FACTOR times its mean diagonal.
        blocks = _diagonal_blocks(solver, under)
        scale = HOLD_FACTOR * np.trace(pristine, axis1=1, axis2=2) / 3.0
        np.testing.assert_allclose(blocks, pristine + scale[:, None, None] * np.eye(3), rtol=1e-6)
        # Tissue at rest that follows: no elastic force yet, nothing stored, no reaction on the tools.
        np.testing.assert_array_equal(law.stored.numpy(), 0.0)
        np.testing.assert_array_equal(law.wrench.numpy(), 0.0)
        # The tissue steps with the nodes held where they were placed; the next accepted step reads their
        # elastic force as the reaction on the tool, one step late.
        following.body_q.assign(scene.state_0.body_q.numpy())
        external = wp.zeros(scene.model.particle_count, dtype=wp.vec3, device=DEVICE)
        scene.state_0.body_f.zero_()
        worker.add_forces(scene.state_0.body_f, external)
        solver.step(scene.state_0, following, scene.model.control(), None, DT, external_forces=external)
        placed = scene.state_0.particle_q.numpy()[under]
        # A hold leaves its node within 1 / HOLD_FACTOR of what the free node would move: here its depth.
        np.testing.assert_allclose(following.particle_q.numpy()[under], placed, rtol=0, atol=depth / HOLD_FACTOR)
        self.assertGreater(np.abs(following.particle_q.numpy()[others] - rest[others]).max(), 1e-7)
        scene.state_0 = following
        worker.advance(DT)
        wrench = law.wrench.numpy()
        self.assertGreater(wrench[1, 2], 1e-5)
        np.testing.assert_array_equal(wrench[[0, 2]], 0.0)
        balance = law.balance.numpy()
        np.testing.assert_allclose(balance[under].sum(axis=0)[2], -wrench[1, 2], rtol=1e-4)
        np.testing.assert_array_equal(balance[others], 0.0)
        # The loads of an attempt: the reaction joins the thread's on the bodies, the balancing forces the
        # thread's on the tissue; a rejected attempt adds them again and advances nothing.
        saved = [array.numpy().copy() for array in worker.state]
        for _ in range(2):
            following.body_f.zero_()
            external.zero_()
            worker.add_forces(following.body_f, external)
            expected = worker.rigid_contact.wrench.numpy() + wrench
            expected[scene.needle_body] += worker.attachment.wrench.numpy()[0]
            np.testing.assert_allclose(following.body_f.numpy(), expected, rtol=1e-6, atol=1e-12)
            np.testing.assert_allclose(external.numpy(), worker._nodal_force.numpy() + balance, rtol=1e-6, atol=1e-12)
        for value, array in zip(saved, worker.state, strict=True):
            np.testing.assert_array_equal(value, array.numpy())
        # The thread's reaction that the diagnostics report is the thread's own.
        self.assertAlmostEqual(
            worker.maximum_tissue_reaction(),
            float(np.linalg.norm(worker._nodal_force.numpy(), axis=-1).max(initial=0)),
            delta=1e-12,
        )
        # The tool leaves: the nodes are released, their blocks are those of the matrix again, nothing is stored.
        poses = following.body_q.numpy()
        poses[1, 2] = PARKED
        following.body_q.assign(poses)
        for _ in range(3):
            worker.advance(DT)
        np.testing.assert_array_equal(law.held.numpy(), 0)
        np.testing.assert_array_equal(law.wrench.numpy(), 0.0)
        np.testing.assert_array_equal(law.balance.numpy(), 0.0)
        np.testing.assert_array_equal(_diagonal_blocks(solver, under), pristine)

    def test_a_tangent_refresh_keeps_the_holds_of_the_worker(self):
        scene, config = _slab_scene()
        _press(scene.state_0)
        worker = WorkerThread(scene, config, tool_tissue="swept", tool_effective_mass=MASS)
        solver = scene.solver
        under = _under(scene.reference.vertices)
        worker.advance(DT)
        # As a frame does: the tangent at the accepted state, at the full and at a refined interval.
        for dt in (DT, 0.5 * DT, DT):
            solver.refresh_tangent(scene.state_0.particle_q, dt)
            blocks = _diagonal_blocks(solver, under)
            pristine = worker.swept_tool_tissue.holds.pristine.numpy()
            nodes = worker.swept_tool_tissue.nodes_host
            own = pristine[[int(np.flatnonzero(nodes == node)[0]) for node in under]]
            scale = HOLD_FACTOR * np.trace(own, axis1=1, axis2=2) / 3.0
            np.testing.assert_allclose(blocks, own + scale[:, None, None] * np.eye(3), rtol=1e-6)
            self.assertTrue(np.all(scale > 0))

    def test_start_up_capture_leaves_the_contact_the_tissue_and_the_matrix_as_it_found_them(self):
        if not wp.get_device(DEVICE).is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("The thread's start-up capture needs CUDA with conditional graphs")
        dt = DT

        def pressed():
            scene, config = _slab_scene()
            _press(scene.state_0)
            states = (scene.state_0, scene.model.state())
            states[1].body_q.assign(states[0].body_q.numpy())
            return scene, config, states

        # Two scenes: a tissue solver takes the holds of one contact.
        scene, config, states = pressed()
        reference_scene, _, reference_states = pressed()
        reference = WorkerThread(reference_scene, config, capture=False, tool_tissue="swept", tool_effective_mass=MASS)
        worker = WorkerThread(scene, config, tool_tissue="swept", tool_effective_mass=MASS)
        law, solver = worker.swept_tool_tissue, scene.solver
        under = _under(scene.reference.vertices)
        # The capsule lies in the tissue of both buffers: every pass of the capture places and holds nodes.
        np.testing.assert_array_equal(law.held.numpy(), 0)
        pristine = _diagonal_blocks(solver, under)
        owned = (*worker.state, worker.solver.iterations)
        tissue = [array for state in states for array in (state.particle_q, state.particle_qd)]
        before = [_bits(array) for array in (*owned, *tissue)]
        self.assertTrue(worker.capture_graphs(states, dt))
        self.assertEqual([_bits(array) for array in (*owned, *tissue)], before)
        np.testing.assert_array_equal(_diagonal_blocks(solver, under), pristine)
        self.assertEqual((worker.accepted_steps, worker.graph_launches, len(worker._step_graphs)), (0, 0, 2))
        self.assertEqual(worker.graph_captures, 2)
        # The graphs hold the contact's update: replayed steps on either buffer equal the eager worker.
        for step in range(4):
            scene.state_0, reference_scene.state_0 = states[step % 2], reference_states[step % 2]
            reference.advance(dt)
            worker.advance(dt)
            np.testing.assert_array_equal(law.held.numpy(), reference.swept_tool_tissue.held.numpy())
            np.testing.assert_allclose(
                scene.state_0.particle_q.numpy(), reference_scene.state_0.particle_q.numpy(), rtol=0, atol=1e-9
            )
            np.testing.assert_allclose(
                law.wrench.numpy(), reference.swept_tool_tissue.wrench.numpy(), rtol=1e-4, atol=1e-9
            )
        np.testing.assert_array_equal(np.flatnonzero(law.held.numpy()), under)
        self.assertEqual((worker.graph_captures, worker.graph_launches), (2, 4))
        blocks = _diagonal_blocks(solver, under)
        self.assertTrue(np.all(np.trace(blocks, axis1=1, axis2=2) > 1000.0 * np.trace(pristine, axis1=1, axis2=2)))
        # The holds are bound to the copies of the matrix that the solves read.
        self.assertEqual(law.holds._key(), law.holds._bound_key)


class GraspSceneSweptTests(unittest.TestCase):
    """The switch of the grasp scene with the swept law, on the small real mesh of the scene tests."""

    ARGUMENTS = ("--tool-tissue-contact", "swept")

    def test_scene_option_builds_the_swept_contact_with_the_constants_of_the_scene(self):
        from test_grasp_scene import _geometry, _scene

        from newton_solver_suturebot.examples.grasp._grasp_scene import tool_effective_masses

        with _scene(example_arguments=self.ARGUMENTS) as (scene, config):
            self.assertEqual(scene.args.tool_tissue_contact, "swept")
            worker = scene.worker_thread
            law = worker.swept_tool_tissue
            self.assertIsInstance(law, SweptToolTissueContact)
            self.assertIsNone(worker.tool_tissue)
            # The capsule set of the tools, as their models publish it: the contact has no radii of its own.
            set_radii = np.concatenate([tool.capsule_radii for tool in scene.graspers])
            np.testing.assert_array_equal(worker.tool_tissue_radii, set_radii)
            pieces = capsule_pieces(
                np.concatenate([np.asarray(tool.capsule_bodies) for tool in scene.graspers]),
                np.concatenate([np.asarray(tool.capsule_endpoints, dtype=float) for tool in scene.graspers]),
                set_radii,
            )
            bodies = law._sample_body.numpy()
            np.testing.assert_array_equal(bodies, pieces[0])
            np.testing.assert_allclose(law._sample_radius.numpy(), pieces[4], rtol=0, atol=1e-10)
            for index, tool in enumerate(scene.graspers):
                self.assertTrue(np.all(law._sample_tool.numpy()[np.isin(bodies, tool.capsule_bodies)] == index))
            self.assertEqual(law.sample_count, len(pieces[0]))
            # Sweep horizon, reach, stretch limit and tissue solver of the scene.
            self.assertEqual(law.full_step, scene.sim_dt)
            self.assertAlmostEqual(law.reach, float(np.ptp(scene.reference.vertices[:, 2])), delta=1e-12)
            self.assertEqual(law.min_stretch, scene.solver.min_stretch)
            self.assertIs(law.solver, scene.solver)
            self.assertIs(scene.solver._node_holds, law.holds)
            # K_c of each tool from its smallest effective mass at the reset state: the rule, computed
            # again here on the rigid model at the joint coordinates of the start.
            tools = scene.graspers
            masses = tool_effective_masses(
                scene.rigid_solver,
                scene.model,
                scene.state_0,
                np.concatenate([np.asarray(tool.capsule_bodies) for tool in tools]),
                np.concatenate([np.asarray(tool.capsule_endpoints, dtype=float) for tool in tools]),
                np.concatenate([np.asarray(tool.capsule_radii, dtype=float) for tool in tools]),
                worker.tool_tissue_radii,
                np.repeat(np.arange(len(tools)), [len(tool.capsule_bodies) for tool in tools]),
            )
            np.testing.assert_allclose(worker.tool_effective_mass, masses, rtol=1e-9)
            # Between 12 and 30 g: an end of a jaw capsule of a dVRK tool across its jaw. The parked tool
            # with its jaws open is the lighter one.
            self.assertEqual(masses.shape, (2,))
            self.assertGreater(masses.min(), 0.012)
            self.assertLess(masses.max(), 0.030)
            self.assertLess(masses[1], masses[0])
            np.testing.assert_allclose(
                law.held_stiffness_range, (masses.min() / scene.sim_dt**2, masses.max() / scene.sim_dt**2), rtol=1e-6
            )
            tool_of_piece = law._sample_tool.numpy()
            np.testing.assert_allclose(
                law._inverse_mass.numpy(), np.eye(3)[None] / masses[tool_of_piece][:, None, None], rtol=1e-6
            )
            report = scene.report()["tool_tissue_contact"]
            self.assertEqual(report["axis_pieces"], len(pieces[0]))
            self.assertEqual(report["capsule_contact_radii_m"], worker.tool_tissue_radii)
            self.assertEqual(report["effective_tool_mass_kg"], worker.tool_effective_mass)
            self.assertEqual(len(report["effective_tool_mass_kg"]), 2)
            self.assertEqual(report["sweep_horizon_s"], scene.sim_dt)
            self.assertIn("swept contact of the tool capsules", _geometry(scene, config)["scope"])
            # The scene has the pinch of both tools, by the same exchange; its arrays are part of the state.
            exchange = worker.exchange
            self.assertIsNotNone(exchange.pinch)
            self.assertEqual(
                exchange.pinch.jaw_bodies_host.tolist(), [[int(body) for body in tool.jaw_bodies] for tool in tools]
            )
            np.testing.assert_allclose(exchange.pinch.friction.numpy(), [0.83, 0.83], rtol=1e-6)
            self.assertEqual(len(exchange.state), len(law.state) + len(exchange.pinch.state))
            self.assertEqual(
                len(worker.state), len(worker.solver.state) + len(worker.tissue.state) + 2 + len(exchange.state)
            )

    def test_frames_run_with_it_and_a_reset_leaves_no_force_no_hold_and_no_placed_node(self):
        from test_grasp_scene import _reset_scene, _scene

        with _scene(example_arguments=self.ARGUMENTS) as (scene, config):
            worker = scene.worker_thread
            law = worker.swept_tool_tissue
            for _ in range(2):
                scene.step()
                self.assertTrue(scene.running, scene.failure_reason)
            self.assertEqual(worker.accepted_steps, scene.accepted_physics_steps)
            scene._check_tissue_integrity()
            # The tools are clear of the tissue here: no node is held and nothing is stored.
            np.testing.assert_array_equal(law.held.numpy(), 0)
            np.testing.assert_array_equal(law.stored.numpy(), 0.0)
            # What a contact would leave behind: a stored force, a reaction, a held and moved node with its hold.
            node = int(law.nodes_host[0])
            law.stored.fill_(wp.vec3(0.0, 0.0, 0.01))
            law.wrench.fill_(wp.spatial_vector(0.0, 0.0, 0.01, 0.0, 0.0, 0.0))
            held = np.zeros(scene.model.particle_count, dtype=np.int32)
            held[node] = 1
            law.held.assign(held)
            law.holds.apply()
            positions = scene.state_0.particle_q.numpy()
            positions[node, 2] -= 1.0e-4
            scene.state_0.particle_q.assign(positions)
            self.assertGreater(
                np.trace(_diagonal_blocks(scene.solver, [node])[0]), 1000.0 * np.trace(law.holds.pristine.numpy()[0])
            )
            old_solver = scene.solver
            _reset_scene(scene, config, rebuild=True)
            fresh = scene.worker_thread.swept_tool_tissue
            self.assertIsNot(fresh, law)
            self.assertIsNot(scene.solver, old_solver)
            for array in fresh.state:
                if array is fresh.owner:
                    np.testing.assert_array_equal(array.numpy(), -1)
                else:
                    np.testing.assert_array_equal(array.numpy(), 0)
            # The tissue is at rest and its matrix has the blocks of the assembled tangent.
            np.testing.assert_array_equal(scene.state_0.particle_q.numpy(), scene.selection.vertices.astype(np.float32))
            scene.solver.refresh_tangent(scene.state_0.particle_q, scene.sim_dt)
            np.testing.assert_array_equal(
                _diagonal_blocks(scene.solver, fresh.nodes_host), fresh.holds.pristine.numpy()
            )
            self.assertIs(scene.solver._node_holds, fresh.holds)

    def test_contact_off_keeps_the_scene_of_before(self):
        from test_grasp_scene import _scene

        for arguments in ((), ("--tool-tissue-contact", "off")):
            with _scene(example_arguments=arguments) as (scene, _):
                self.assertIsNone(scene.worker_thread.swept_tool_tissue)
                self.assertIsNone(scene.worker_thread.tool_tissue)
                self.assertNotIn("tool_tissue_contact", scene.report())
                self.assertIsNone(getattr(scene.solver, "_node_holds", None))
                self.assertFalse(hasattr(scene, "_capsule_contact_radii"))

    def test_start_up_call_leaves_the_scene_as_it_found_it_and_frames_capture_and_load_nothing(self):
        device = wp.get_device(DEVICE)
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("The start-up call captures CUDA graphs")
        from test_grasp_scene import (
            _NOTHING_LATE,
            _counted_captures,
            _late,
            _mark_startup,
            _observe,
            _recorded_startup,
            _scene,
        )

        with (
            _recorded_startup() as records,
            _scene(rigid_backend="mujoco-warp", example_arguments=self.ARGUMENTS) as (scene, _),
        ):
            self.assertEqual(len(records), 1)
            (arrays, host, _), (arrays_after, host_after, _), returned = records[0]
            self.assertTrue(returned)
            worker = scene.worker_thread
            law = worker.swept_tool_tissue
            # The record holds every array of the worker's state, the contact's among them, and the tissue.
            self.assertEqual(sum(name.startswith("thread.") for name in arrays), len(worker.state) + 1, sorted(arrays))
            self.assertEqual(sorted(arrays_after), sorted(arrays))
            for name, bits in arrays.items():
                self.assertEqual(arrays_after[name], bits, name)
            self.assertEqual(host_after, host)
            for array in law.state:
                np.testing.assert_array_equal(array.numpy(), -1 if array is law.owner else 0)
            self.assertEqual((worker.graph_captures, worker.graph_launches, len(worker._step_graphs)), (2, 0, 2))
            # The holds are bound to the copies of the tissue matrix that the captured solves read.
            self.assertEqual(law.holds._key(), law.holds._bound_key)
            fixed = scene.solver._fixed_particles
            self.assertIsNotNone(fixed.compact_matrix)
            self.assertIs(law.holds._compact_values, fixed.compact_matrix.values)
            self.assertIs(law.holds._projected_values, fixed._projected.values)
            _observe(scene, 0)
            _mark_startup(scene)
            with _counted_captures() as begun:
                for _ in range(3):
                    scene.step()
                    self.assertTrue(scene.running, scene.failure_reason)
                self.assertEqual(begun, [])
            self.assertEqual((worker.graph_captures, worker.graph_launches), (2, worker.accepted_steps))
            self.assertEqual(worker.accepted_steps, scene.accepted_physics_steps)
            self.assertEqual(law.holds._key(), law.holds._bound_key)
            scene.refresh_capture_census()
            report = scene.report()
            census = report["capture_census"]
            self.assertEqual(census["status"], "checked")
            self.assertEqual(_late(report), _NOTHING_LATE, census)
            self.assertEqual((census["late_sites"], census["late_modules"]), ({}, {}))
            self.assertEqual(census["startup_unlaunched_graphs"], 0)


class ExampleSweptOptionTests(unittest.TestCase):
    def test_example_option_takes_the_law_as_its_value_and_hands_it_to_the_scene(self):
        from newton_solver_suturebot.examples.grasp import example_fem_needle_grasp as grasp_example
        from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene

        parser = grasp_example.Example.create_parser()
        coupling = grasp_example.Example._coupling
        off = coupling(parser.parse_args(["--no-tool-tissue-contact"]))
        self.assertNotIn("--tool-tissue-contact", off)
        self.assertEqual(parser.parse_args(["--tool-tissue-contact", "swept"]).tool_tissue_contact, "swept")
        self.assertEqual(
            coupling(parser.parse_args(["--tool-tissue-contact", "swept"])), (*off, "--tool-tissue-contact", "swept")
        )
        # The first-version law is a law of the scene, not of the example (the wrong values below).
        for arguments in (
            ["--tool-tissue-contact", "off"],
            ["--tool-tissue-contact", "swept", "--no-tool-tissue-contact"],
        ):
            self.assertEqual(coupling(parser.parse_args(arguments)), off)
        # The default of the example is this law, as the parser and a namespace without the setting take it.
        self.assertEqual(grasp_example.TOOL_TISSUE_CONTACT, "swept")
        self.assertEqual(parser.parse_args([]).tool_tissue_contact, "swept")
        self.assertEqual(coupling(parser.parse_args([])), (*off, "--tool-tissue-contact", "swept"))
        self.assertEqual(coupling(SimpleNamespace()), (*off, "--tool-tissue-contact", "swept"))
        # The scene parses what the example hands over.
        scene_parser = GraspScene.create_parser()
        self.assertEqual(scene_parser.parse_args(["--tool-tissue-contact", "swept"]).tool_tissue_contact, "swept")
        self.assertIs(scene_parser.parse_args(["--tool-tissue-contact"]).tool_tissue_contact, True)
        self.assertIs(scene_parser.parse_args([]).tool_tissue_contact, False)
        refused = [
            (candidate, ["--tool-tissue-contact", value])
            for value in ("sweep", "on")
            for candidate in (parser, scene_parser)
        ]
        # The first-version law (the option without a value, or "first") is a law of the scene only.
        refused += [(parser, ["--tool-tissue-contact", *rest]) for rest in ([], ["first"], ["--overview"])]
        for candidate, wrong in refused:
            with self.assertRaises(SystemExit), open(os.devnull, "w") as sink:
                stderr, os.sys.stderr = os.sys.stderr, sink
                try:
                    candidate.parse_args(wrong)
                finally:
                    os.sys.stderr = stderr


if __name__ == "__main__":
    unittest.main()
