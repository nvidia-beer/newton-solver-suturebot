# SPDX-License-Identifier: Apache-2.0
"""The one wiring of the tool/tissue contact: one setup function, one exchange object, two hosts.

``fem_needle_grasp`` (the worker thread's graph) and ``fem_tool_tissue`` (the hooks of its
scene) build the contact by the same function from the same kind of description and issue the
same calls of the same object. The tests run on the CPU, or on the device of
``NEWTON_TEST_DEVICE``.
"""

import ast
import contextlib
import math
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp
from test_swept_grasp_contact import DT, MASS, PARKED, _diagonal_blocks, _press, _slab_scene, _under

import newton_solver_suturebot
from newton_solver_suturebot._src.solvers.coupled.interface.swept_tool_tissue_contact import SweptToolTissueContact
from newton_solver_suturebot._src.solvers.coupled.interface.tool_tissue_contact import ToolTissueContact
from newton_solver_suturebot._src.solvers.coupled.interface.tool_tissue_exchange import (
    LAWS,
    ToolTissueExchange,
    ToolTissueSetup,
    tool_capsules,
)
from newton_solver_suturebot._src.solvers.fem.node_holds import HOLD_FACTOR
from newton_solver_suturebot._src.solvers.thread import worker_thread
from newton_solver_suturebot._src.solvers.thread.worker_thread import WorkerThread
from newton_solver_suturebot.examples.grasp._grasp_scene import (
    placed_tool_effective_masses,
    tool_effective_masses,
    tool_tissue_setup,
)
from newton_solver_suturebot.examples.grasp._tool_tissue_contact import SceneContact

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
# The fields of a setup that are arrays.
ARRAYS = (
    "capsule_bodies",
    "capsule_endpoints",
    "capsule_radii",
    "contact_radii",
    "capsule_tools",
    "effective_mass",
    "fixed_particles",
)


class _RecordedHolds:
    """The holds of a law, with the calls of ``apply`` written down."""

    def __init__(self, holds, calls):
        self._holds, self._calls = holds, calls

    def __getattr__(self, name):
        return getattr(self._holds, name)

    def apply(self):
        self._calls.append(("holds.apply",))
        return self._holds.apply()


class _RecordedLaw:
    """A law that writes down the calls it gets from its exchange and passes them on to the real law."""

    def __init__(self, law, calls):
        self._law, self._calls = law, calls

    def __getattr__(self, name):
        return getattr(self._law, name)

    def add_forces(self, *arrays):
        self._calls.append(("add_forces", *(array.ptr for array in arrays)))
        return self._law.add_forces(*arrays)

    def add_balance(self, *arrays):
        self._calls.append(("add_balance", *(array.ptr for array in arrays)))
        return self._law.add_balance(*arrays)

    def update(self, *arrays):
        self._calls.append(("update", *(array.ptr for array in arrays)))
        return self._law.update(*arrays)

    @property
    def holds(self):
        return _RecordedHolds(self._law.holds, self._calls)


def _tool_scene(contact):
    from test_tool_tissue_example import _scene

    return _scene(contact)


class ToolTissueSetupTests(unittest.TestCase):
    """One function turns a scene with tools into what its exchange is built from."""

    def _assert_setup_of(self, scene, law):
        setup = tool_tissue_setup(scene, law)
        self.assertIsInstance(setup, ToolTissueSetup)
        self.assertEqual(setup.law, law)
        tools = scene.graspers
        bodies = np.concatenate([np.asarray(tool.capsule_bodies) for tool in tools])
        endpoints = np.concatenate([np.asarray(tool.capsule_endpoints, dtype=float) for tool in tools])
        radii = np.concatenate([np.asarray(tool.capsule_radii, dtype=float) for tool in tools])
        of_tool = np.repeat(np.arange(len(tools)), [len(tool.capsule_bodies) for tool in tools])
        # The capsules of the tools in their order, with the tool of each, and the two jaws of each tool.
        np.testing.assert_array_equal(setup.capsule_bodies, bodies)
        np.testing.assert_array_equal(setup.capsule_endpoints, endpoints)
        np.testing.assert_array_equal(setup.capsule_radii, radii)
        np.testing.assert_array_equal(setup.capsule_tools, of_tool)
        self.assertEqual(setup.jaw_bodies, tuple(tuple(int(body) for body in tool.jaw_bodies) for tool in tools))
        for jaws, tool in zip(setup.jaw_bodies, tools, strict=True):
            self.assertEqual(len(jaws), 2)
            self.assertTrue(set(jaws) <= {int(body) for body in tool.capsule_bodies})
        # The friction between jaws and tissue that each tool model gives (the pinch): the one of its configuration.
        self.assertEqual(setup.tissue_friction, tuple(tool.config.tissue_friction for tool in tools))
        self.assertEqual(setup.tissue_friction, (0.83,) * len(tools))
        # No other radii than those of the set.
        np.testing.assert_array_equal(setup.contact_radii, radii)
        self.assertEqual(setup.reach, float(np.ptp(np.asarray(scene.reference.vertices, dtype=float)[:, 2])))
        if law == "swept":
            # The masses of the rule, at the configuration that the scene holds.
            masses = tool_effective_masses(
                scene.rigid_solver, scene.model, scene.state_0, bodies, endpoints, radii, radii, of_tool
            )
            np.testing.assert_array_equal(setup.effective_mass, masses)
            np.testing.assert_array_equal(placed_tool_effective_masses(scene), masses)
            self.assertEqual(setup.effective_mass.shape, (len(tools),))
            np.testing.assert_array_equal(setup.fixed_particles, np.asarray(scene.fixed_vertex_indices))
            self.assertEqual(setup.full_step, scene.sim_dt)
            self.assertIsNone(setup.node_stiffness)
            self.assertIsNone(setup.shear_frequency)
        else:
            args = scene.args
            rest = np.asarray(scene.selection.vertices)
            tetrahedra = np.asarray(scene.selection.indices)
            edge = float(np.linalg.norm(rest[tetrahedra[:, 1:]] - rest[tetrahedra[:, :1]], axis=-1).mean())
            shear = args.young_modulus / (2.0 * (1.0 + args.poisson_ratio))
            self.assertAlmostEqual(setup.node_stiffness, 0.25 * args.young_modulus * edge, delta=1e-12)
            self.assertAlmostEqual(
                setup.shear_frequency,
                0.5 * math.pi * math.sqrt(shear / args.density) / float(np.ptp(rest[:, 2])),
                delta=1e-9,
            )
            self.assertIsNone(setup.effective_mass)
            self.assertIsNone(setup.fixed_particles)
            self.assertIsNone(setup.full_step)
        return setup

    def _assert_same(self, setup, other):
        self.assertEqual((setup.law, setup.jaw_bodies, setup.reach), (other.law, other.jaw_bodies, other.reach))
        self.assertEqual(setup.tissue_friction, other.tissue_friction)
        self.assertEqual(
            (setup.node_stiffness, setup.shear_frequency, setup.full_step),
            (other.node_stiffness, other.shear_frequency, other.full_step),
        )
        for name in ARRAYS:
            mine, theirs = getattr(setup, name), getattr(other, name)
            if mine is None or theirs is None:
                self.assertIs(mine, theirs, name)
            else:
                np.testing.assert_array_equal(mine, theirs, err_msg=name)

    def test_one_setup_function_serves_both_scenes(self):
        from test_grasp_scene import _scene

        self.assertEqual(LAWS, ("first", "swept"))
        # The grasp scene: with the contact off the function gives nothing and the worker thread has no exchange.
        with _scene() as (scene, _):
            for off in (None, False, "off"):
                self.assertIsNone(tool_tissue_setup(scene, off))
            self.assertIsNone(scene.worker_thread.exchange)
            with self.assertRaises(ValueError):
                tool_tissue_setup(scene, "other")
            for law in LAWS:
                self._assert_setup_of(scene, law)
        # With a law its worker thread holds the one exchange, built from what the function gives.
        for law, kind in (("first", ToolTissueContact), ("swept", SweptToolTissueContact)):
            arguments = ("--tool-tissue-contact",) if law == "first" else ("--tool-tissue-contact", "swept")
            with _scene(example_arguments=arguments) as (scene, _):
                exchange = scene.worker_thread.exchange
                self.assertIsInstance(exchange, ToolTissueExchange)
                self.assertIsInstance(exchange.law, kind)
                self.assertEqual(exchange.law_name, law)
                self._assert_same(exchange.setup, self._assert_setup_of(scene, law))
                # The surface is the one that the thread's tissue coupling refits.
                self.assertIs(exchange.law.mesh, scene.worker_thread.tissue.surface[0])
        # The tool/tissue scene: the same function, the same object, under the hooks of the scene.
        scene = _tool_scene("off")
        self.assertIsNone(tool_tissue_setup(scene, "off"))
        self.assertIsInstance(scene.contact, SceneContact)
        self.assertIsNone(scene.contact.exchange)
        for law in LAWS:
            self._assert_setup_of(scene, law)
        for law, kind in (("first", ToolTissueContact), ("swept", SweptToolTissueContact)):
            scene = _tool_scene(law)
            exchange = scene.contact.exchange
            self.assertIsInstance(exchange, ToolTissueExchange)
            self.assertIsInstance(exchange.law, kind)
            self._assert_same(exchange.setup, self._assert_setup_of(scene, law))
            self.assertIs(exchange.law.mesh, scene.surface.mesh)
            self.assertIs(scene.contact.law, exchange.law)

    def test_capsules_of_the_tools_are_gathered_in_one_place(self):
        scene, _ = _slab_scene()
        bodies, endpoints, radii, tools, jaws = tool_capsules(scene.graspers)
        np.testing.assert_array_equal(bodies, [1, 2])
        self.assertEqual(endpoints.shape, (2, 2, 3))
        np.testing.assert_array_equal(radii, [tool.capsule_radii[0] for tool in scene.graspers])
        np.testing.assert_array_equal(tools, [0, 1])
        # A tool model without named jaws has none.
        self.assertEqual(jaws, ((), ()))
        scene.graspers[1].jaw_bodies = (2, 2)
        self.assertEqual(tool_capsules(scene.graspers)[4], ((), (2, 2)))
        with self.assertRaises(ValueError):
            ToolTissueSetup.of_scene(scene, "off")
        scene.graspers.clear()
        with self.assertRaises(ValueError):
            ToolTissueSetup.of_scene(scene, "swept", effective_mass=MASS)

    def test_only_the_exchange_constructs_a_tool_tissue_law(self):
        package = Path(newton_solver_suturebot.__file__).parent
        laws = {"SweptToolTissueContact", "ToolTissueContact"}
        callers = {}
        for source in sorted(package.rglob("*.py")):
            for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                name = function.id if isinstance(function, ast.Name) else getattr(function, "attr", None)
                if name in laws:
                    callers.setdefault(str(source.relative_to(package)), set()).add(name)
        self.assertEqual(callers, {"_src/solvers/coupled/interface/tool_tissue_exchange.py": laws})
        # And the exchange itself is built by the two hosts only: the worker thread and the hooks of the example.
        builders = set()
        for source in sorted(package.rglob("*.py")):
            for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ToolTissueExchange":
                    builders.add(str(source.relative_to(package)))
        self.assertEqual(builders, {"_src/solvers/thread/worker_thread.py", "examples/grasp/_tool_tissue_contact.py"})


class ToolTissueExchangeCallTests(unittest.TestCase):
    """Both hosts issue the calls of the law through the exchange, in the order they had."""

    def test_exchange_issues_the_law_calls_in_their_order(self):
        # Host 1: the worker thread of the grasp scene, on a slab with a pressed capsule.
        scene, config = _slab_scene()
        _press(scene.state_0)
        worker = WorkerThread(scene, config, capture=False, tool_tissue="swept", tool_effective_mass=MASS)
        exchange, calls = worker.exchange, []
        exchange.law = _RecordedLaw(exchange.law, calls)
        external = wp.zeros(scene.model.particle_count, dtype=wp.vec3, device=DEVICE)
        state = scene.state_0
        # The loads of an attempt: the reaction on the bodies, then (after the thread's nodal forces) the balance.
        worker.add_forces(state.body_f, external)
        self.assertEqual(calls, [("add_forces", state.body_f.ptr), ("add_balance", external.ptr)])
        # An accepted step ends with the update, at the accepted state.
        calls.clear()
        worker.advance(DT)
        self.assertEqual(
            calls, [("update", state.body_q.ptr, state.body_qd.ptr, state.particle_q.ptr, state.particle_qd.ptr)]
        )
        self.assertGreater(int(exchange.law.held.numpy().sum()), 0)
        # After a capture has put the flags back, the matrix follows them: the holds are applied once more.
        calls.clear()
        exchange.restore_holds()
        self.assertEqual(calls, [("holds.apply",)])
        self.assertEqual([array.ptr for array in exchange.state], [array.ptr for array in exchange.law.state])
        self.assertTrue(exchange.moves_tissue)
        self.assertEqual(
            [array.ptr for array in worker._accepted_arrays(state)],
            [state.body_q.ptr, state.body_qd.ptr, state.particle_q.ptr, state.particle_qd.ptr],
        )

        # The first law in the same host: the reaction on the bodies; its forces on the tissue go into the
        # array that the thread's forces are in, with the update of the accepted step; it holds nothing.
        scene, config = _slab_scene()
        scene.args.poisson_ratio, scene.args.density = 0.35, 1000.0
        worker = WorkerThread(scene, config, capture=False, tool_tissue=True)
        exchange, calls = worker.exchange, []
        exchange.law = _RecordedLaw(exchange.law, calls)
        state = scene.state_0
        worker.add_forces(state.body_f, external)
        worker.advance(DT)
        exchange.restore_holds()
        self.assertEqual(
            calls, [("add_forces", state.body_f.ptr), ("update", state.body_q.ptr, worker._nodal_force.ptr)]
        )
        self.assertFalse(exchange.moves_tissue)
        self.assertEqual(
            [array.ptr for array in worker._accepted_arrays(state)],
            [state.body_q.ptr, state.body_qd.ptr, state.particle_q.ptr],
        )

        # Host 2: the tool/tissue scene. Every interval of a frame: the reaction before the rigid interval, the
        # balance before the tissue step, the update at the accepted state; its tools have jaws and a friction with
        # the tissue, so the exchange has the pinch, after which the law's holds are applied once more.
        from test_tool_tissue_example import _plan

        scene = _tool_scene("swept")
        _, arm = _plan(scene)
        scene.place(arm[0])
        exchange, calls = scene.contact.exchange, []
        exchange.law = _RecordedLaw(exchange.law, calls)
        scene.set_frame_targets(arm[0], arm[1])
        scene.step()
        self.assertIsNone(scene.failure_reason)
        self.assertIsNotNone(exchange.pinch)
        self.assertEqual(
            [call[0] for call in calls], ["add_forces", "add_balance", "update", "holds.apply"] * scene.args.substeps
        )
        states = (scene.state_0, scene.state_1)
        for interval in range(scene.args.substeps):
            reaction, balance, update = calls[4 * interval : 4 * interval + 3]
            # An interval goes from one buffer into the other: the loads at its start, the update at its end.
            start = next(state for state in states if state.body_f.ptr == reaction[1])
            end = states[1 - states.index(start)]
            self.assertEqual(balance, ("add_balance", scene._external_forces.ptr))
            self.assertEqual(
                update, ("update", end.body_q.ptr, end.body_qd.ptr, end.particle_q.ptr, end.particle_qd.ptr)
            )
        # The first law under the same hooks: its forces on the tissue are in the array of the hooks.
        scene = _tool_scene("first")
        scene.place(arm[0])
        exchange, calls = scene.contact.exchange, []
        exchange.law = _RecordedLaw(exchange.law, calls)
        scene.set_frame_targets(arm[0], arm[1])
        scene.step()
        self.assertIsNone(scene.failure_reason)
        self.assertEqual([call[0] for call in calls], ["add_forces", "update"] * scene.args.substeps)
        self.assertTrue(all(call[2] == scene.contact.nodal_force.ptr for call in calls if call[0] == "update"))
        self.assertEqual(len(scene.contact.state), len(exchange.state) + 1)

    def test_capture_restore_puts_holds_back(self):
        scene, config = _slab_scene()
        _press(scene.state_0)
        worker = WorkerThread(scene, config, capture=False, tool_tissue="swept", tool_effective_mass=MASS)
        exchange, solver, law = worker.exchange, scene.solver, worker.swept_tool_tissue
        under = _under(scene.reference.vertices)
        # As in a scene, the tissue has stepped before the thread first advances: the solver has then made the
        # copies of its matrix that its solves read, which the holds are bound to.
        following = scene.model.state()
        solver.step(scene.state_0, following, scene.model.control(), None, DT)
        pristine = _diagonal_blocks(solver, under)

        def saved_pass(change):
            """What a capture does around a pass: keep the state, run the pass, put the state back."""
            state = scene.state_0
            owned = (*worker.state, worker.solver.iterations, state.particle_q, state.particle_qd, state.body_q)
            saved = [wp.clone(array) for array in owned]
            flags = law.held.numpy().copy()
            before = _diagonal_blocks(solver, under)
            change()
            worker.advance(DT)
            during = _diagonal_blocks(solver, under)
            for array, values in zip(owned, saved, strict=True):
                wp.copy(array, values)
            np.testing.assert_array_equal(law.held.numpy(), flags)
            return before, during

        # A pass that places and holds the nodes under the capsule. With the flags back the matrix still has
        # the holds of the pass; restore_holds() makes it the matrix of the flags.
        before, during = saved_pass(lambda: None)
        np.testing.assert_array_equal(before, pristine)
        scale = HOLD_FACTOR * np.trace(pristine, axis1=1, axis2=2) / 3.0
        np.testing.assert_allclose(during, pristine + scale[:, None, None] * np.eye(3), rtol=1e-6)
        np.testing.assert_array_equal(_diagonal_blocks(solver, under), during)
        exchange.restore_holds()
        np.testing.assert_array_equal(_diagonal_blocks(solver, under), pristine)
        np.testing.assert_array_equal(law.held.numpy(), 0)

        # The other way: nodes are held, and the pass releases them because the tool has left.
        worker.advance(DT)
        np.testing.assert_array_equal(np.flatnonzero(law.held.numpy()), under)
        held = _diagonal_blocks(solver, under)

        def leave():
            poses = scene.state_0.body_q.numpy()
            poses[1, 2] = PARKED
            scene.state_0.body_q.assign(poses)

        before, during = saved_pass(leave)
        np.testing.assert_array_equal(before, held)
        np.testing.assert_array_equal(during, pristine)
        exchange.restore_holds()
        np.testing.assert_array_equal(_diagonal_blocks(solver, under), held)
        np.testing.assert_array_equal(np.flatnonzero(law.held.numpy()), under)

    def test_the_capture_restore_of_the_worker_thread_puts_the_holds_back(self):
        # On a CUDA device the start-up capture of the worker thread is tested as it is
        # (test_swept_grasp_contact). Here its passes and its restore run on the CPU too, where the worker
        # declines to capture: only what needs CUDA is replaced (the refusal, the capture scope, the launch
        # of a graph), so that a pass inside the scope runs as written. Everything else is the worker's own
        # text: the pass as written, one pass per buffer, the restore after each of them and at the end.
        scene, config = _slab_scene()
        _press(scene.state_0)
        worker = WorkerThread(scene, config, tool_tissue="swept", tool_effective_mass=MASS)
        exchange, solver, law = worker.exchange, scene.solver, worker.swept_tool_tissue
        under = _under(scene.reference.vertices)
        # As in a scene, the tissue has stepped before the thread's start-up: the solver has then made the
        # copies of its matrix that its solves read, which the holds are bound to.
        states = (scene.state_0, scene.model.state())
        solver.step(states[0], states[1], scene.model.control(), None, DT)
        states[1].assign(states[0])
        pristine = _diagonal_blocks(solver, under)
        np.testing.assert_array_equal(law.held.numpy(), 0)
        owned = (*worker.state, worker.solver.iterations)
        tissue = [array for state in states for array in (state.particle_q, state.particle_qd)]
        before = [array.numpy().tobytes() for array in (*owned, *tissue)]
        passes, restores, held_in_a_pass = [], [], []
        advance, restore_holds = worker._advance_device, exchange.restore_holds
        cuda = wp.get_device(DEVICE).is_cuda and wp.is_conditional_graph_supported()

        def counted_pass(accepted, dt):
            advance(accepted, dt)
            passes.append(accepted)
            if not cuda:
                # Not on a CUDA device: there a pass is issued into a capture, which nothing may read in.
                held_in_a_pass.append(np.flatnonzero(law.held.numpy()).tolist())

        def counted_restore():
            restores.append(len(passes))
            restore_holds()

        class _Scope:
            """Stands for the capture scope: what is issued inside it runs as written."""

            def __init__(self, device=None):
                self.graph = object()

            def __enter__(self):
                return self

            def __exit__(self, *error):
                return False

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(worker, "_advance_device", counted_pass))
            stack.enter_context(patch.object(exchange, "restore_holds", counted_restore))
            if not cuda:
                stack.enter_context(patch.object(worker, "_capture_refused", lambda arrays: False))
                stack.enter_context(patch.object(worker_thread, "ScopedCapture", _Scope))
                stack.enter_context(patch.object(wp, "capture_launch", lambda graph: None))
            self.assertTrue(worker.capture_graphs(states, DT))
        # The pass as written and one pass per buffer (on a CUDA device each captured pass is launched once more,
        # in its graph); every one of them placed and held the nodes under the capsule.
        self.assertEqual([state is states[index] for state, index in zip(passes, (0, 0, 1), strict=True)], [True] * 3)
        if not cuda:
            self.assertEqual(held_in_a_pass, [under.tolist()] * 3)
        # The exchange's restore_holds() was called after every pass and once more at the end.
        self.assertEqual(restores, [1, 2, 3, 3])
        # So the flags, the tissue and the thread have the bits they had, and the tissue matrix is the matrix of
        # those flags: without the call it would keep the hold blocks of the last pass.
        self.assertEqual([array.numpy().tobytes() for array in (*owned, *tissue)], before)
        np.testing.assert_array_equal(law.held.numpy(), 0)
        np.testing.assert_array_equal(_diagonal_blocks(solver, under), pristine)
        self.assertEqual((worker.accepted_steps, worker.graph_launches, worker.graph_captures), (0, 0, 2))

    def test_effective_masses_are_replaced_per_tool_and_a_reset_clears_the_law(self):
        scene, config = _slab_scene()
        _press(scene.state_0)
        worker = WorkerThread(scene, config, capture=False, tool_tissue="swept", tool_effective_mass=[MASS, 2 * MASS])
        exchange, law = worker.exchange, worker.swept_tool_tissue
        np.testing.assert_array_equal(exchange.effective_mass, [MASS, 2 * MASS])
        self.assertEqual(exchange.contact_radii, worker.tool_tissue_radii)
        tools = law.sample_tools_host
        exchange.set_effective_mass([3 * MASS, MASS])
        np.testing.assert_array_equal(exchange.effective_mass, [3 * MASS, MASS])
        self.assertEqual(worker.tool_effective_mass, [3 * MASS, MASS])
        np.testing.assert_allclose(
            law._inverse_mass.numpy(), np.eye(3)[None] / np.array([3 * MASS, MASS])[tools][:, None, None], rtol=1e-6
        )
        for masses in ([MASS], [MASS, 0.0], [MASS, float("nan")]):
            with self.assertRaises(ValueError):
                exchange.set_effective_mass(masses)
        self.assertEqual(exchange.parameters(), law.parameters())
        self.assertEqual([module for module, _ in exchange.modules()], [module for module, _ in law.modules()])
        worker.advance(DT)
        self.assertGreater(exchange.diagnostics()["held_nodes"], 0)
        exchange.reset()
        self.assertEqual(exchange.diagnostics()["held_nodes"], 0)
        np.testing.assert_array_equal(law.held.numpy(), 0)


if __name__ == "__main__":
    unittest.main()
