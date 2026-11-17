# SPDX-License-Identifier: Apache-2.0
"""Captured trials preserve the public coupled-step transaction and fallback."""

import os
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np
import warp as wp
from test_needle_insertion_example import _example
from test_needle_shaft_preconditioner import append_engaged_sample, history, setup
from test_needle_tissue_coupler import _native_state

from newton_solver_suturebot import NeedleCouplingNotConverged, NeedleStepTooLarge, NeedleTissueCoupler, SolverFEM
from newton_solver_suturebot._src.solvers.coupled.coupling_graph import _CapturedNeedleTrials
from newton_solver_suturebot._src.solvers.coupled.interface.tunnel_alm import _TunnelALM
from newton_solver_suturebot._src.solvers.coupled.mujoco_warp_trials import ArticulatedRigidTrials
from newton_solver_suturebot._src.solvers.mujoco.mujoco_free_body_gpu import MuJoCoFreeBodyGPU

_FIELDS = ("particle_q", "particle_qd", "particle_f", "joint_q", "joint_qd", "body_q", "body_qd", "body_f")
_DT = 0.0005


def _configure(example, mode, capture):
    # Accepted geometric probes seed a preloaded shaft. The subsequent tests
    # advance actual FEM/native rigid dynamics; the setup is not an insertion run.
    setup(example, _DT, 1500.0)
    example.interaction.friction = 0.02 if mode == "alm" else 0.2
    if mode == "alm":
        example.interaction.shaft_damping_per_m = 0
        example.interaction.tunnel_alm = _TunnelALM(
            example.interaction,
            penalty_per_m=1500,
            compliance_per_m=1 / 1500,
            clearance=2e-5,
            tolerance=2e-6,
        )
        example.interaction.tunnel_alm._resize()
    example.coupler = NeedleTissueCoupler(
        example.model,
        example.solver,
        example.rigid_solver,
        example.interaction,
        example.needle_body,
        example.drive.com,
        absolute_tolerance=2e-7,
        relative_tolerance=1e-4,
        max_iterations=160,
        shaft_preconditioner=True,
        capture_fem=True,
        capture_coupling=capture,
    )
    example.control.joint_f.assign(np.array([0, 1e-5, 0, 2e-8, 0, 0], np.float32))


@contextmanager
def _pair(mode):
    with ExitStack() as stack:
        reference = stack.enter_context(_example())
        captured = stack.enter_context(_example())
        _configure(reference, mode, False)
        _configure(captured, mode, True)
        yield reference, captured


def _arrays(state):
    return {name: getattr(state, name).numpy().copy() for name in _FIELDS}


def _snapshot(example):
    interaction = example.interaction
    result = {
        "input": _arrays(example.state_0),
        "output": _arrays(example.state_1),
        "native": _native_state(example.rigid_solver),
        "increment": example.solver.velocity_increment.numpy().copy(),
        "load": example.coupler._load.numpy().copy(),
        "history": history(interaction),
        "visited": tuple(sorted(interaction._visited_tets)),
        "openings": repr(interaction._openings),
    }
    for name in ("_previous_tip", "_previous_q", "_previous_samples"):
        result[name] = getattr(interaction, name).copy()
    for name in ("_track_nodes", "_track_bary", "_track_lengths", "_track_engaged"):
        result[name] = getattr(interaction, name).numpy().copy()
    if interaction.tunnel_alm is not None:
        result["normal"] = interaction.tunnel_alm.committed_normal.numpy().copy()
        result["tangent"] = interaction.tunnel_alm.committed_tangent.numpy().copy()
    return result


class ShaftFlagBranchRuleTests(unittest.TestCase):
    def test_only_the_native_free_needle_without_a_response_basis_branches_on_the_shaft_flag(self):
        # The rule alone, on every device: the trials are stand-ins that hold what the two
        # properties read, with the rigid adapters as their classes define them. The native free
        # needle has no correction of its own; the adapter of the articulated tools on MuJoCo
        # Warp applies the correction itself and keeps the flag as an entry of the signature, or
        # does not evaluate it with a surface metric.
        self.assertFalse(hasattr(MuJoCoFreeBodyGPU, "apply_preconditioner"))
        self.assertTrue(hasattr(ArticulatedRigidTrials, "apply_preconditioner"))

        def trials(adapter, *, surface, basis):
            rigid = object.__new__(adapter)
            if adapter is ArticulatedRigidTrials:
                rigid.surface_preconditioned = surface
            stand_in = object.__new__(_CapturedNeedleTrials)
            stand_in.rigid = rigid
            stand_in.c = SimpleNamespace(fem_solver=SimpleNamespace(_affine_predictor=object() if basis else None))
            return stand_in.evaluates_shaft_flag, stand_in.branches_on_shaft_flag

        self.assertEqual(trials(MuJoCoFreeBodyGPU, surface=False, basis=False), (True, True))
        self.assertEqual(trials(MuJoCoFreeBodyGPU, surface=False, basis=True), (True, False))
        self.assertEqual(trials(ArticulatedRigidTrials, surface=False, basis=False), (True, False))
        self.assertEqual(trials(ArticulatedRigidTrials, surface=False, basis=True), (True, False))
        self.assertEqual(trials(ArticulatedRigidTrials, surface=True, basis=False), (False, False))
        self.assertEqual(trials(ArticulatedRigidTrials, surface=True, basis=True), (False, False))


class NeedleCouplingGraphTests(unittest.TestCase):
    def setUp(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not device.is_cuda:
            self.skipTest("Coupled conditional-graph transaction requires CUDA")

    def assert_rollback(self, example, saved):
        current = _snapshot(example)
        for group in ("input", "output"):
            for name, values in saved[group].items():
                np.testing.assert_array_equal(current[group][name], values, err_msg=f"rollback {group}.{name}")
        for name in ("history", "visited", "openings"):
            self.assertEqual(current[name], saved[name], name)
        for name in saved.keys() - {"input", "output", "history", "visited", "openings"}:
            np.testing.assert_array_equal(current[name], saved[name], err_msg=f"rollback {name}")
        if example.interaction.tunnel_alm is not None:
            self.assertFalse(example.interaction.tunnel_alm._open)

    def assert_equivalent(self, reference, captured):
        self.assertEqual(history(reference.interaction), history(captured.interaction))
        for name in _FIELDS:
            expected = getattr(reference.state_1, name).numpy()
            actual = getattr(captured.state_1, name).numpy()
            self.assertTrue(np.isfinite(actual).all(), name)
            np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=5e-8, err_msg=name)
        np.testing.assert_array_equal(
            captured.state_1.particle_q.numpy()[captured.fixed_vertex_indices],
            captured.state_0.particle_q.numpy()[captured.fixed_vertex_indices],
        )
        np.testing.assert_allclose(
            _native_state(captured.rigid_solver),
            _native_state(reference.rigid_solver),
            rtol=3e-6,
            atol=2e-7,
        )
        expected, actual = reference.coupler._value.numpy(), captured.coupler._value.numpy()
        self.assertLessEqual(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-20), 0.01)
        self.assertGreater(np.linalg.norm(expected), 1e-7)
        for example in (reference, captured):
            if example.interaction.tunnel_alm is not None:
                self.assertFalse(example.interaction.tunnel_alm._open)
        self.assertEqual(reference.coupler.coupling_graph_launches, 0)

    def step_pair(self, reference, captured, dt):
        for example in (reference, captured):
            native_time = float(example.rigid_solver.mj_data.time)
            with patch.object(example.interaction, "accept", wraps=example.interaction.accept) as accept:
                example.coupler.step(example.state_0, example.state_1, example.control, dt)
            self.assertEqual(accept.call_count, 1)
            self.assertAlmostEqual(example.rigid_solver.mj_data.time, native_time + dt, places=12)
        self.assert_equivalent(reference, captured)

    def test_penalty_and_alm_advances_match_with_alternating_buffers_timestep_and_loads(self):
        for mode in ("penalty", "alm"):
            with self.subTest(mode=mode), _pair(mode) as (reference, captured):
                workspaces = None
                for index, dt in enumerate((_DT, _DT, _DT / 2)):
                    for example in (reference, captured):
                        example.drive.prepare_timestep(dt)
                        example.solver.refresh_tangent(example.state_0.particle_q, dt)
                        example.control.joint_f.assign(np.array([0, (-1) ** index * 1e-5, 0, 2e-8, 0, 0], np.float32))
                    self.step_pair(reference, captured, dt)
                    self.assertEqual(captured.coupler.coupling_graph_fallbacks, 0)
                    trials, solver = captured.coupler._captured_trials, captured.solver
                    current = tuple(id(workspace) for _, workspace in trials.linear_solvers)
                    if workspaces is not None:
                        self.assertEqual(current, workspaces)
                    workspaces = current
                    # The graphs bind the workspaces of the solver's registry, one per tolerance
                    # level, and the one a device solve leaves selected is one of the registry.
                    levels = solver.tolerance_levels()
                    self.assertEqual([settings[3] for settings, _ in trials.linear_solvers], levels)
                    registered = list(solver._linear_workspaces.values())
                    for _, workspace in trials.linear_solvers:
                        self.assertTrue(any(workspace is value for value in registered))
                    self.assertEqual(len(solver._pinned_workspaces), len(levels))
                    self.assertTrue(any(solver._lin_solver is value for value in registered))
                    self.assertEqual(trials.signature[-1], solver.linear_binding_epoch())
                    for example in (reference, captured):
                        example.state_0, example.state_1 = example.state_1, example.state_0
                self.assertEqual(captured.coupler.coupling_graph_launches, 3)
                # One capture per state buffer. The third step halves dt, and with it the
                # drive's implicit damping: both are device data, so that step replays the
                # graph of its buffer and still matches the host-controlled reference.
                self.assertEqual(captured.coupler.coupling_graph_captures, 2)

    def test_timestep_changes_replay_both_buffer_graphs_without_recapture(self):
        # dt, dt/2, dt again, then on: each buffer's graph also runs at a time step other
        # than that of its capture. The time step reaches the kernels as device data
        # (StepTime), so no step after the two first captures clears or recaptures a graph,
        # and every step matches the host-controlled trials of a coupler without the graph.
        for mode in ("penalty", "alm"):
            with self.subTest(mode=mode), _pair(mode) as (reference, captured):
                coupler = captured.coupler
                graphs = None
                for index, dt in enumerate((_DT, _DT / 2, _DT, _DT, _DT / 2)):
                    for example in (reference, captured):
                        example.drive.prepare_timestep(dt)
                        example.solver.refresh_tangent(example.state_0.particle_q, dt)
                    self.step_pair(reference, captured, dt)
                    self.assertEqual(coupler.coupling_graph_fallbacks, 0)
                    self.assertEqual(coupler.coupling_graph_launches, index + 1)
                    self.assertEqual(coupler.coupling_graph_captures, min(index + 1, 2))
                    # One object holds the time step of the coupled step for every stage.
                    self.assertEqual(coupler.step_time.value, dt)
                    self.assertIs(captured.solver.step_time, coupler.step_time)
                    self.assertIs(captured.interaction.step_time, coupler.step_time)
                    self.assertIs(coupler._captured_trials.rigid.step_time, coupler.step_time)
                    np.testing.assert_array_equal(coupler.step_time.seconds.numpy(), [np.float32(dt)])
                    np.testing.assert_array_equal(coupler.step_time.seconds_f64.numpy(), [dt])
                    if index == 1:
                        graphs = [entry[0] for entry in coupler._captured_trials.graphs.values()]
                        self.assertEqual(len(graphs), 2)
                    elif index > 1:
                        current = [entry[0] for entry in coupler._captured_trials.graphs.values()]
                        self.assertEqual(len(current), 2)
                        for kept, graph in zip(graphs, current, strict=True):
                            self.assertIs(graph, kept)
                    for example in (reference, captured):
                        example.state_0, example.state_1 = example.state_1, example.state_0

    def test_flag_of_the_shaft_correction_is_device_data_for_a_free_needle_and_its_change_captures_nothing(self):
        # The free-body trials issue the correction of the track behind a device flag: one graph
        # holds the corrected relaxation and the one that hands the residual on, each with the
        # launches that the graph of its value issued. The flag is no entry of the trial
        # signature: a change of it, in either direction, neither drops nor captures a graph, and
        # the steps on either side of the change still match the host-controlled trials of a
        # coupler without the graph.
        for mode in ("penalty", "alm"):
            with self.subTest(mode=mode), _pair(mode) as (reference, captured):
                coupler = captured.coupler
                pre = coupler._shaft_preconditioner

                def step():
                    self.step_pair(reference, captured, _DT)
                    for example in (reference, captured):
                        example.state_0, example.state_1 = example.state_1, example.state_0

                step()
                step()
                self.assertEqual(coupler.coupling_graph_captures, 2)
                trials = coupler._captured_trials
                self.assertFalse(hasattr(trials.rigid, "apply_preconditioner"))
                self.assertFalse(getattr(trials.rigid, "surface_preconditioned", False))
                self.assertTrue(trials.evaluates_shaft_flag)
                self.assertTrue(trials.branches_on_shaft_flag)
                signature = trials.signature
                entry = signature.index(pre.storage_key()) + 1
                self.assertIsNone(signature[entry])
                graphs = [item[0] for item in trials.graphs.values()]
                self.assertEqual(len(graphs), 2)
                self.assertIs(pre.active, True)
                np.testing.assert_array_equal(trials.shaft_active.numpy(), [1])
                # The track reported as released, to both couplers: the flag is False in these steps.
                with patch.object(
                    type(captured.interaction), "engaged_track_length", new_callable=PropertyMock, return_value=0.0
                ):
                    step()
                    self.assertIs(pre.active, False)
                    np.testing.assert_array_equal(trials.shaft_active.numpy(), [0])
                    step()
                    self.assertEqual(trials.signature, signature)
                self.assertEqual(coupler.coupling_graph_captures, 2)
                step()
                self.assertIs(pre.active, True)
                np.testing.assert_array_equal(trials.shaft_active.numpy(), [1])
                step()
                self.assertEqual(trials.signature, signature)
                self.assertEqual(coupler.coupling_graph_captures, 2)
                self.assertEqual(coupler.coupling_graph_launches, 6)
                self.assertEqual(coupler.coupling_graph_fallbacks, 0)
                current = [item[0] for item in trials.graphs.values()]
                self.assertEqual(len(current), 2)
                for kept, graph in zip(graphs, current, strict=True):
                    self.assertIs(graph, kept)

    def test_flag_of_the_shaft_correction_stays_an_entry_where_the_correction_starts_from_a_response_basis(self):
        # A response-basis start guess is bound on the host in every correction: the graph then
        # holds the correction of one value of the flag, which is an entry of its signature.
        with _pair("alm") as (reference, captured):
            coupler = captured.coupler
            self.step_pair(reference, captured, _DT)
            trials = coupler._captured_trials
            self.assertTrue(trials.branches_on_shaft_flag)
            with patch.object(captured.solver, "_affine_predictor", object(), create=True):
                self.assertTrue(trials.evaluates_shaft_flag)
                self.assertFalse(trials.branches_on_shaft_flag)

    def test_new_opening_or_visited_tetrahedron_alone_replays_both_buffer_graphs(self):
        # The registered openings and the visited ids are storage that the interaction made once
        # and writes in place, and their live counts are device data. A new opening or a newly
        # visited tetrahedron therefore changes no entry of the trial signature: no step after
        # the two first captures clears or recaptures a graph, and every step still matches the
        # host-controlled trials of a coupler without the graph.
        with _pair("penalty") as (reference, captured):
            coupler = captured.coupler

            def step():
                self.step_pair(reference, captured, _DT)
                for example in (reference, captured):
                    example.state_0, example.state_1 = example.state_1, example.state_0

            step()
            step()
            self.assertEqual(coupler.coupling_graph_captures, 2)
            graphs = [entry[0] for entry in coupler._captured_trials.graphs.values()]
            self.assertEqual(len(graphs), 2)
            signature = coupler._captured_trials.signature
            helper = coupler._captured_trials.permissions
            storage = [helper.visited, helper.opening_nodes, helper.opening_bary]
            x = captured.interaction
            before = (len(x._visited_tets), x.track_count, len(x._openings))
            # Both far from the needle, where they change no force: only the history grows.
            for example in (reference, captured):
                x = example.interaction
                active = np.flatnonzero(np.all(x.mapping[x.surface_triangles] >= 0, axis=1))
                centers = x.surface_vertices[x.surface_triangles[active]].mean(axis=1)
                face = int(active[np.argmax(np.linalg.norm(centers - x._previous_tip, axis=1))])
                x._register_opening(x.mapping[x.surface_triangles[face]], np.full(3, 1 / 3), face)
            step()
            self.assertEqual(coupler.coupling_graph_captures, 2)
            for example in (reference, captured):
                x = example.interaction
                centers = x._previous_q[x.tetrahedra].mean(axis=1)
                order = np.argsort(-np.linalg.norm(centers - x._previous_tip, axis=1))
                x._visited_tets.add(int(next(tet for tet in order if int(tet) not in x._visited_tets)))
            step()
            step()
            self.assertEqual(coupler.coupling_graph_captures, 2)
            self.assertEqual(coupler.coupling_graph_launches, 5)
            self.assertEqual(coupler.coupling_graph_fallbacks, 0)
            self.assertEqual(coupler._captured_trials.signature, signature)
            current = [entry[0] for entry in coupler._captured_trials.graphs.values()]
            self.assertEqual(len(current), 2)
            for kept, graph in zip(graphs, current, strict=True):
                self.assertIs(graph, kept)
            for kept, array in zip(storage, (helper.visited, helper.opening_nodes, helper.opening_bary), strict=True):
                self.assertIs(array, kept)
            x = captured.interaction
            self.assertEqual(x.opening_storage_growths, 0)
            # The graphs read the grown history: its counts are on the device.
            after = (len(x._visited_tets), x.track_count, len(x._openings))
            np.testing.assert_array_equal(x._live.numpy(), after)
            self.assertEqual(after[2], before[2] + 1)
            self.assertGreater(after[0], before[0])
            # The track did not grow in these steps, so no other entry had a reason to change.
            self.assertEqual(after[1], before[1])

    def test_new_track_sample_alone_replays_both_buffer_graphs(self):
        # The track samples are storage that the interaction made once at its track reserve and
        # writes in place; the helper's scratch per track row, the tunnel multipliers and the
        # anchors of the shaft correction have the rows of that reserve, and the live track count
        # is device data. A new sample therefore changes neither a buffer nor an entry of the
        # trial signature: no step after the two first captures clears or recaptures a graph,
        # and every step still matches the host-controlled trials of a coupler without the graph.
        for mode in ("penalty", "alm"):
            with self.subTest(mode=mode), _pair(mode) as (reference, captured):
                coupler = captured.coupler

                def step():
                    self.step_pair(reference, captured, _DT)
                    for example in (reference, captured):
                        example.state_0, example.state_1 = example.state_1, example.state_0

                step()
                step()
                self.assertEqual(coupler.coupling_graph_captures, 2)
                trials = coupler._captured_trials
                helper, signature = trials.permissions, trials.signature
                helper_signature = helper.signature()
                graphs = [entry[0] for entry in trials.graphs.values()]
                self.assertEqual(len(graphs), 2)
                pre = coupler._shaft_preconditioner
                anchors, storage_key = (pre.matrices, pre.levers, pre.shears), pre.storage_key()
                # The entry of the correction is its storage, which holds no track count.
                self.assertIn(storage_key, signature)
                x = captured.interaction
                names = ("nodes", "bary", "points", "segments", "projected_arc", "associated")
                storage = [getattr(helper, name) for name in names]
                track = [x._track_nodes, x._track_bary, x._track_bary_d, x._track_lengths, x._track_engaged]
                if x.tunnel_alm is not None:
                    track += [getattr(x.tunnel_alm, name) for name in ("normal", "tangent", "proposed_normal")]
                before = (len(x._visited_tets), x.track_count, len(x._openings))
                self.assertGreater(x.track_capacity, before[1] + 1)
                # A sample far from the needle, without an edge to the track and released: it has no
                # quadrature length, exerts no force and permits nothing. Only the history grows.
                for example in (reference, captured):
                    x = example.interaction
                    centers = x._previous_q[x.tetrahedra].mean(axis=1)
                    order = np.argsort(-np.linalg.norm(centers - x._previous_tip, axis=1))
                    tet = int(next(tet for tet in order if int(tet) not in x._visited_tets))
                    anchor, x._track_anchor = x._track_anchor, None
                    x._record_track(x.tetrahedra[tet].copy(), np.full(4, 0.25))
                    x._track_anchor = anchor
                    self.assertEqual(x.track_count, before[1] + 1)
                    released = wp.array([0], dtype=int, device="cpu")
                    wp.copy(x._track_engaged, released, dest_offset=before[1], count=1)
                    x._engaged_length = None
                step()
                step()
                self.assertEqual(coupler.coupling_graph_fallbacks, 0)
                self.assertEqual(coupler.coupling_graph_launches, 4)
                # Nothing is captured again: both graphs are launched with the count on the device.
                self.assertEqual(coupler.coupling_graph_captures, 2)
                self.assertIs(coupler._captured_trials, trials)
                self.assertEqual(trials.signature, signature)
                current = [entry[0] for entry in trials.graphs.values()]
                self.assertEqual(len(current), 2)
                for kept, graph in zip(graphs, current, strict=True):
                    self.assertIs(graph, kept)
                x = captured.interaction
                after = (len(x._visited_tets), x.track_count, len(x._openings))
                # The correction counts the new sample on the host and keeps its anchors and their key.
                self.assertEqual((pre.count, pre.capacity, pre.storage_growths), (after[1], x.track_capacity, 0))
                self.assertEqual(pre.storage_key(), storage_key)
                for kept, array in zip(anchors, (pre.matrices, pre.levers, pre.shears), strict=True):
                    self.assertIs(array, kept)
                self.assertIs(trials.permissions, helper)
                self.assertEqual(helper.signature(), helper_signature)
                for name, kept in zip(names, storage, strict=True):
                    self.assertIs(getattr(helper, name), kept, name)
                current_track = [x._track_nodes, x._track_bary, x._track_bary_d, x._track_lengths, x._track_engaged]
                if x.tunnel_alm is not None:
                    current_track += [getattr(x.tunnel_alm, name) for name in ("normal", "tangent", "proposed_normal")]
                for kept, array in zip(track, current_track, strict=True):
                    self.assertIs(array, kept)
                self.assertEqual(x.track_storage_growths, 0)
                # The graphs read the grown history: its counts are on the device.
                self.assertEqual(after[1], before[1] + 1)
                np.testing.assert_array_equal(x._live.numpy(), after)
                self.assertEqual(int(x._track_engaged.numpy()[before[1]]), 0)
                # One more sample, engaged and with a length this time: it gets an anchor in the shaft
                # correction and exerts a force, so a graph that kept the track count of its capture
                # would leave it out. The graphs captured before it are launched with it: the step
                # matches the host-controlled trials (step_pair), and the correction that the
                # launched graph issues writes the anchor of the new row, as the host-controlled
                # correction of the twin does.
                rows = [append_engaged_sample(example) - 1 for example in (reference, captured)]
                self.assertEqual(rows, [after[1], after[1]])
                step()
                self.assertEqual((coupler.coupling_graph_fallbacks, coupler.coupling_graph_captures), (0, 2))
                self.assertEqual(coupler.coupling_graph_launches, 5)
                for example, row in zip((reference, captured), rows, strict=True):
                    anchors = example.coupler._shaft_preconditioner.matrices.numpy()
                    self.assertTrue(np.any(anchors[row]), "the correction left the new sample out")

    def test_dropped_workspace_registry_changes_only_the_epoch_entry_and_rebinds_the_graphs(self):
        # The trial graphs bind Krylov workspaces of the FEM solver's registry. An explicit
        # invalidation of the solver's workspace drops that registry: the last entry of the trial
        # signature changes with it, alone, the graphs are captured once more with new
        # workspaces, and every step still matches the host-controlled reference.
        with _pair("penalty") as (reference, captured):
            coupler, solver = captured.coupler, captured.solver

            def step():
                self.step_pair(reference, captured, _DT)
                for example in (reference, captured):
                    example.state_0, example.state_1 = example.state_1, example.state_0

            step()
            step()
            self.assertEqual(coupler.coupling_graph_captures, 2)
            trials = coupler._captured_trials
            signature, epoch = trials.signature, solver.linear_binding_epoch()
            self.assertEqual(signature[-1], epoch)
            bound = [workspace for _, workspace in trials.linear_solvers]
            step()
            self.assertEqual((coupler.coupling_graph_captures, trials.signature), (2, signature))
            solver._lin_solver = None
            step()
            step()
            self.assertEqual(coupler.coupling_graph_fallbacks, 0)
            self.assertEqual(coupler.coupling_graph_launches, 5)
            self.assertEqual(coupler.coupling_graph_captures, 4)
            self.assertIs(coupler._captured_trials, trials)
            current = trials.signature
            self.assertEqual(len(current), len(signature))
            changed = [index for index, (old, new) in enumerate(zip(signature, current, strict=True)) if old != new]
            self.assertEqual(changed, [len(signature) - 1])
            self.assertEqual(current[-1], solver.linear_binding_epoch())
            self.assertGreater(current[-1], epoch)
            registered = list(solver._linear_workspaces.values())
            self.assertEqual(len(trials.linear_solvers), len(bound))
            for old, (_, new) in zip(bound, trials.linear_solvers, strict=True):
                self.assertIsNot(new, old)
                self.assertFalse(any(old is value for value in registered))
                self.assertTrue(any(new is value for value in registered))
            self.assertEqual(len(solver._pinned_workspaces), len(bound))

    def test_host_trials_after_device_trials_replay_captured_fem_steps(self):
        # Device trials leave a workspace of the solver's registry selected, and the levels of
        # the trial graph are the workspaces a host-controlled trial selects. A coupled step
        # with host-controlled trials after device trials therefore issues no eager FEM step:
        # each FEM step that Python issues is the capture of a graph, every other one a replay.
        with _pair("penalty") as (reference, captured):
            coupler, solver = captured.coupler, captured.solver

            def step():
                self.step_pair(reference, captured, _DT)
                for example in (reference, captured):
                    example.state_0, example.state_1 = example.state_1, example.state_0

            step()
            step()
            self.assertEqual((coupler.coupling_graph_launches, coupler.coupling_graph_fallbacks), (2, 0))
            self.assertEqual((coupler.fem_graph_captures, coupler.fem_graph_launches), (0, 0))
            trials = coupler._captured_trials
            bound, epoch = [workspace for _, workspace in trials.linear_solvers], solver.linear_binding_epoch()
            original, issued = SolverFEM.step, []

            def counted(self, *args, **kwargs):
                if self is solver:
                    issued.append(bool(self.model.device.is_capturing))
                return original(self, *args, **kwargs)

            coupler.capture_coupling = False
            with patch.object(SolverFEM, "step", counted):
                step()
                step()
                first = (len(issued), coupler.fem_graph_captures, coupler.fem_graph_launches)
                step()
                step()
            self.assertEqual(coupler.coupling_graph_launches, 2)
            self.assertGreater(len(issued), 0)
            self.assertTrue(all(issued))
            self.assertEqual(len(issued), coupler.fem_graph_captures)
            self.assertGreaterEqual(coupler.fem_graph_launches, 4)
            self.assertEqual(coupler.fem_graph_status, "replaying")
            # Returning to each state buffer captures nothing more at the levels it visited before.
            self.assertGreater(coupler.fem_graph_launches, first[2])
            self.assertLessEqual(coupler.fem_graph_captures, 2 * len(solver.tolerance_levels()))
            registered = list(solver._linear_workspaces.values())
            self.assertTrue(any(solver._lin_solver is value for value in registered))
            # Device trials again, with the workspaces they bound before the host-controlled steps.
            coupler.capture_coupling = True
            step()
            self.assertEqual((coupler.coupling_graph_launches, coupler.coupling_graph_fallbacks), (3, 0))
            self.assertIs(coupler._captured_trials, trials)
            for kept, (_, workspace) in zip(bound, trials.linear_solvers, strict=True):
                self.assertIs(workspace, kept)
            self.assertEqual(trials.signature[-1], epoch)
            self.assertEqual(solver.linear_binding_epoch(), epoch)

    def test_exhausted_budget_rolls_back_then_retries_without_committing_history(self):
        for mode in ("penalty", "alm"):
            with self.subTest(mode=mode), _pair(mode) as (reference, captured):
                saved = _snapshot(captured)
                captured.coupler.max_iterations = 1
                with patch.object(captured.interaction, "accept", wraps=captured.interaction.accept) as accept:
                    with self.assertRaises(NeedleCouplingNotConverged):
                        captured.coupler.step(captured.state_0, captured.state_1, captured.control, _DT)
                    self.assertEqual(accept.call_count, 0)
                self.assert_rollback(captured, saved)
                self.assertEqual(captured.coupler.coupling_graph_launches, 1)
                captured.coupler.max_iterations = 160
                self.step_pair(reference, captured, _DT)
                self.assertEqual(captured.coupler.coupling_graph_launches, 2)

    def test_late_resolution_rejection_restores_native_warmstart_output_and_alm_duals(self):
        with _pair("alm") as (reference, captured):
            saved = _snapshot(captured)
            with (
                patch.object(
                    captured.interaction, "check_step_resolution", side_effect=NeedleStepTooLarge("late guard")
                ),
                patch.object(captured.interaction, "accept", wraps=captured.interaction.accept) as accept,
            ):
                with self.assertRaisesRegex(NeedleStepTooLarge, "late guard"):
                    captured.coupler.step(captured.state_0, captured.state_1, captured.control, _DT)
                self.assertEqual(accept.call_count, 0)
            self.assert_rollback(captured, saved)
            self.assertEqual(captured.coupler.coupling_graph_launches, 1)
            self.step_pair(reference, captured, _DT)
            self.assertEqual(captured.coupler.coupling_graph_launches, 2)
            self.assertEqual(captured.coupler.coupling_graph_captures, 1)

    def test_native_damping_change_at_fixed_dt_is_read_by_the_captured_graph(self):
        with _pair("penalty") as (reference, captured):
            self.step_pair(reference, captured, _DT)
            captures = captured.coupler.coupling_graph_captures
            graphs = [entry[0] for entry in captured.coupler._captured_trials.graphs.values()]
            for example in (reference, captured):
                # Preserve input allocation: the same graph binding serves the next step.
                example.state_0.assign(example.state_1)
                example.rigid_solver.mj_model.dof_damping[:] *= 1.5
            # The drive scales this damping with dt, so it changes with every retried
            # interval. The trials read it from device arrays that the host preparation
            # rewrites: the graph is kept, and the step still matches the reference,
            # whose native integration uses the changed damping.
            self.step_pair(reference, captured, _DT)
            self.assertEqual(captured.coupler.coupling_graph_captures, captures)
            self.assertEqual([entry[0] for entry in captured.coupler._captured_trials.graphs.values()], graphs)
            np.testing.assert_array_equal(
                captured.coupler._captured_trials.rigid.damping.numpy(), captured.rigid_solver.mj_model.dof_damping
            )
            self.assertEqual(captured.coupler.coupling_graph_fallbacks, 0)

    def test_failed_preparation_is_not_cached_and_identical_retry_succeeds(self):
        with _pair("alm") as (reference, captured):
            saved = _snapshot(captured)
            original = captured.solver._make_linear_solver
            calls = [0]

            def once(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError("injected preparation failure")
                return original(*args, **kwargs)

            with patch.object(captured.solver, "_make_linear_solver", side_effect=once):
                with self.assertRaisesRegex(RuntimeError, "injected preparation failure"):
                    captured.coupler.step(captured.state_0, captured.state_1, captured.control, _DT)
                self.assert_rollback(captured, saved)
                self.assertEqual(captured.coupler.coupling_graph_launches, 0)
                self.step_pair(reference, captured, _DT)
            self.assertEqual(captured.coupler.coupling_graph_launches, 1)

    def test_near_exit_geometry_uses_original_authorization_and_commits_once(self):
        with _pair("penalty") as (reference, captured):
            for example in (reference, captured):
                middle = example.plan.exit_angle / 2
                poses = []
                for lift in np.linspace(0.0002, 0, 9)[1:]:
                    pose = example.plan.pose(middle)
                    pose[2] += lift
                    poses.append(pose)
                finish = example.plan.exit_angle + 0.01
                count = int(np.ceil((finish - middle) * example.geometry.radius / 0.00005))
                poses.extend(example.plan.pose(angle) for angle in np.linspace(middle, finish, count + 1)[1:])
                for pose in poses:
                    example.state_0.clear_forces()
                    example._external_forces.zero_()
                    example.interaction.add_forces(
                        example.state_0,
                        example.needle_body,
                        example.drive.com,
                        example._external_forces,
                        _DT,
                    )
                    example._initialize_pose(pose)
                    example.interaction.accept(example.state_0, example.needle_body)
                    self.assertNotEqual(example.interaction.status, "failed", example.interaction.reason)
                self.assertGreater(example.interaction.track_count, 0)
                self.assertFalse(example.interaction._in_material)
                example.state_0.clear_forces()
            opening_history = []
            for example in (reference, captured):
                x = example.interaction
                self.assertTrue(x.punctured)
                self.assertIsNotNone(x._opening)
                opening_history.append((tuple(array.copy() for array in x._opening), repr(x.events)))
            # This existing physical comparison retains native clock, exactly
            # one accept, all endpoint fields, fixed boundaries and force checks.
            self.step_pair(reference, captured, _DT)
            for example, (opening, events) in zip((reference, captured), opening_history, strict=True):
                x = example.interaction
                self.assertTrue(x.punctured)
                self.assertFalse(x._in_material)
                self.assertEqual(repr(x.events), events)
                for actual, expected in zip(x._opening, opening, strict=True):
                    np.testing.assert_array_equal(actual, expected)
            self.assertEqual(captured.coupler.coupling_graph_launches, 1)
            self.assertEqual(captured.coupler.coupling_graph_fallbacks, 0)
            self.assertEqual(captured.coupler.coupling_graph_status, "replaying")
            self.assertGreater(captured.interaction._authorized_outside_certified_trials, 0)

    def test_surface_event_fallback_route_and_starting_load(self):
        from newton_solver_suturebot._src.solvers.coupled import coupling_graph

        solve = coupling_graph._CapturedNeedleTrials.solve
        # Captured host forces evaluate the certificates of the device trials. Without the
        # search only an event found after the first device trial can end differently there.
        # With it the device trials go on past the event; the oracle then starts from the
        # load they converged to, or from the accepted load when they did not converge.
        cases = (
            # search, first trial with a request, device converged, captured host forces first, device load kept
            (False, 1, False, False, False),
            (False, 2, False, True, False),
            (True, 1, False, False, False),
            (True, 2, False, True, False),
            (True, 1, True, False, True),
        )
        for search, first, converged, captured_forces, kept in cases:
            with self.subTest(search=search, first=first, converged=converged):
                with _pair("penalty") as (reference, captured):
                    c = captured.coupler
                    c.search_past_surface_events = search
                    seen = {}

                    def event(graph, state_in, control, dt, c=c, seen=seen, first=first, converged=converged):
                        result = solve(graph, state_in, control, dt)
                        seen["device"] = c._trial_load.numpy().copy()
                        result[0], result[3], result[10], result[11] = 3, int(converged), 1, first
                        return result

                    host = c._solve_host

                    def started(*args, c=c, seen=seen, host=host, **kwargs):
                        seen.setdefault("host", c._trial_load.numpy().copy())
                        seen.setdefault("capture_forces", kwargs.get("capture_forces", True))
                        return host(*args, **kwargs)

                    accepted = c._load.numpy().copy()
                    with (
                        patch.object(coupling_graph._CapturedNeedleTrials, "solve", event),
                        patch.object(c, "_solve_host", started),
                    ):
                        if kept:
                            # Another starting load is another search path, and its tissue solve
                            # starts from another guess: the accepted step agrees with the
                            # reference to the solver tolerances, not bit for bit.
                            for example in (reference, captured):
                                example.coupler.step(example.state_0, example.state_1, example.control, _DT)
                            tolerance = max(c.relative_tolerance, captured.solver.linear_solver_tolerance)
                            # particle_f is the tissue solver's impulse scratch, not a state.
                            for name in (field for field in _FIELDS if field != "particle_f"):
                                expected = getattr(reference.state_1, name).numpy()
                                np.testing.assert_allclose(
                                    getattr(captured.state_1, name).numpy(),
                                    expected,
                                    rtol=0.0,
                                    atol=tolerance * np.abs(expected).max(),
                                    err_msg=name,
                                )
                        else:
                            self.step_pair(reference, captured, _DT)
                    self.assertEqual(c.coupling_graph_fallbacks, 1)
                    self.assertEqual(c.coupling_graph_status, "exact surface-event fallback")
                    self.assertEqual(c.coupling_graph_event_searches, int(kept))
                    self.assertEqual(seen["capture_forces"], captured_forces)
                    self.assertFalse(np.array_equal(seen["device"], accepted))
                    np.testing.assert_array_equal(seen["host"], seen["device"] if kept else accepted)

    def test_search_load_the_oracle_cannot_use_restarts_from_the_accepted_load(self):
        from newton_solver_suturebot._src.solvers.coupled import coupling_graph

        solve = coupling_graph._CapturedNeedleTrials.solve
        for failure in ("rejected", "not converged"):
            with self.subTest(failure=failure), _pair("penalty") as (reference, captured):
                c = captured.coupler
                starts = []

                def event(graph, state_in, control, dt):
                    result = solve(graph, state_in, control, dt)
                    result[3], result[10], result[11] = 1, 1, 1
                    return result

                host = c._solve_host

                def started(*args, c=c, starts=starts, host=host, failure=failure, **kwargs):
                    starts.append(c._trial_load.numpy().copy())
                    if len(starts) == 1:
                        if failure == "rejected":
                            raise NeedleStepTooLarge("test")
                        return False
                    return host(*args, **kwargs)

                accepted = c._load.numpy().copy()
                with (
                    patch.object(coupling_graph._CapturedNeedleTrials, "solve", event),
                    patch.object(c, "_solve_host", started),
                ):
                    self.step_pair(reference, captured, _DT)
                self.assertEqual((c.coupling_graph_event_searches, c.coupling_graph_event_restarts), (1, 1))
                self.assertFalse(np.array_equal(starts[0], accepted))
                np.testing.assert_array_equal(starts[1], accepted)

    def test_nonfinite_search_past_an_event_falls_back_instead_of_failing(self):
        from newton_solver_suturebot._src.solvers.coupled import coupling_graph

        solve = coupling_graph._CapturedNeedleTrials.solve
        with _pair("penalty") as (reference, captured):
            c = captured.coupler

            def event(graph, state_in, control, dt):
                result = solve(graph, state_in, control, dt)
                # Trials after an event are a search only; their failure decides nothing.
                result[3], result[4], result[10], result[11] = 0, 1, 0, 1
                return result

            with patch.object(coupling_graph._CapturedNeedleTrials, "solve", event):
                self.step_pair(reference, captured, _DT)
            self.assertEqual(c.coupling_graph_fallbacks, 1)
            self.assertEqual(c.coupling_graph_event_searches, 0)


if __name__ == "__main__":
    unittest.main()
