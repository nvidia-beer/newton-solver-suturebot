# SPDX-License-Identifier: Apache-2.0
"""CUDA force graphs retain native endpoints, event fallback and rollback."""

import os
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import numpy as np
import test_needle_coupling_graph as graph_tests
import warp as wp
from test_needle_coupling_graph import _DT
from test_needle_insertion_example import _example, _prepare

from newton_solver_suturebot import NeedleCouplingNotConverged, NeedleStepTooLarge
from newton_solver_suturebot._src.solvers.coupled.host_trial_forces import TrialForceFallback
from newton_solver_suturebot._src.utils.capture import census

# Every variant of an evaluation: (surface recorder bound, residual of the trial before, new candidate).
_VARIANTS = sorted(
    (surface, previous, fresh) for surface in (False, True) for previous in (False, True) for fresh in (False, True)
)


@contextmanager
def pair(mode):
    with graph_tests._pair(mode) as (reference, captured):
        for example in (reference, captured):
            # Exercise the host/native trial loop used by articulated bodies,
            # independently of the single-free-body device dynamics path.
            example.coupler._rigid_bridge = None
            example.coupler._shaft_preconditioner = None
        yield reference, captured


class HostTrialForcesTests(unittest.TestCase):
    assert_rollback = graph_tests.NeedleCouplingGraphTests.assert_rollback
    assert_equivalent = graph_tests.NeedleCouplingGraphTests.assert_equivalent
    step_pair = graph_tests.NeedleCouplingGraphTests.step_pair

    def setUp(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("Native host-loop force graph requires conditional CUDA graphs")

    def test_penalty_and_alm_match_native_host_oracle_and_reuse_capture(self):
        for mode in ("penalty", "alm"):
            with self.subTest(mode=mode), pair(mode) as (reference, captured):
                for _ in range(3):
                    self.step_pair(reference, captured, _DT)
                    for example in (reference, captured):
                        example.state_0, example.state_1 = example.state_1, example.state_0
                graph = captured.coupler._host_trial_forces
                self.assertGreater(graph.launches, 3)
                self.assertLessEqual(graph.captures, 3)
                if mode == "alm":
                    self.assertGreater(graph.primal_reuses, 0)
                self.assertEqual(graph.fallbacks, 0)
                self.assertEqual(captured.coupler.coupling_graph_launches, 0)

    def test_timestep_changes_replay_the_force_and_fem_graphs_without_recapture(self):
        # A rejected substep is retried with half the time step, later steps return to the base
        # step and a frame ends with a remainder that is no halving. The force kernels and the
        # FEM step of the host-controlled trials read the time step from the coupler's step
        # time, so no graph of these trials is keyed on it: a force graph stands for one
        # (previous residual, fresh candidate) variant, an FEM graph for one state buffer and
        # solver level. Each is captured once and replays at every time step. The reference
        # runs the same trials without any graph.
        for mode in ("penalty", "alm"):
            with self.subTest(mode=mode), pair(mode) as (reference, captured):
                reference.coupler.capture_fem = False
                coupler = captured.coupler
                owner, forces, fem = None, {}, {}
                steps = (_DT, _DT / 2, _DT, _DT, _DT / 2, 0.37 * _DT, _DT)
                for index, dt in enumerate(steps):
                    for example in (reference, captured):
                        example.drive.prepare_timestep(dt)
                        example.solver.refresh_tangent(example.state_0.particle_q, dt)
                    self.step_pair(reference, captured, dt)
                    graph = coupler._host_trial_forces
                    owner = graph if index == 0 else owner
                    self.assertIs(graph, owner)
                    # Nothing was dropped and nothing was captured a second time.
                    self.assertEqual(graph.captures, len(graph.graphs))
                    for key, kept in forces.items():
                        self.assertIs(graph.graphs[key], kept)
                    forces = dict(graph.graphs)
                    current = {key: entry[0] for key, entry in coupler._fem_graphs.items()}
                    self.assertEqual(coupler.fem_graph_captures, len(current))
                    for key, kept in fem.items():
                        self.assertIs(current[key], kept)
                    fem = current
                    self.assertEqual(coupler.step_time.value, dt)
                    np.testing.assert_array_equal(coupler.step_time.seconds.numpy(), [np.float32(dt)])
                    for example in (reference, captured):
                        example.state_0, example.state_1 = example.state_1, example.state_0
                self.assertIsNone(reference.coupler._host_trial_forces)
                self.assertEqual((reference.coupler.fem_graph_captures, reference.coupler.fem_graph_launches), (0, 0))
                self.assertGreater(graph.launches, len(steps))
                self.assertLessEqual(graph.captures, 4)
                self.assertGreater(coupler.fem_graph_launches, len(steps))
                self.assertEqual(graph.fallbacks, 0)
                self.assertEqual(coupler.coupling_graph_launches, 0)

    def test_new_track_sample_replays_the_force_graphs_without_recapture(self):
        # The force graphs address the track through storage that the interaction made once at its
        # track reserve, and read the live track count from the device. A new track sample changes
        # no entry of their signature: every graph captured before it is replayed after it, and
        # the steps still match the reference without a graph.
        for mode in ("penalty", "alm"):
            with self.subTest(mode=mode), pair(mode) as (reference, captured):

                def step():
                    self.step_pair(reference, captured, _DT)
                    for example in (reference, captured):
                        example.state_0, example.state_1 = example.state_1, example.state_0

                for _ in range(3):
                    step()
                graph = captured.coupler._host_trial_forces
                kept, signature = dict(graph.graphs), graph.signature
                helper_signature = graph.permissions.signature()
                self.assertGreater(len(kept), 0)
                self.assertEqual(graph.captures, len(kept))
                launches = graph.launches
                x = captured.interaction
                count = x.track_count
                self.assertGreater(x.track_capacity, count + 1)
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
                    self.assertEqual(x.track_count, count + 1)
                    released = wp.array([0], dtype=int, device="cpu")
                    wp.copy(x._track_engaged, released, dest_offset=count, count=1)
                    x._engaged_length = None
                step()
                step()
                self.assertIs(captured.coupler._host_trial_forces, graph)
                self.assertGreater(graph.launches, launches)
                self.assertEqual(graph.fallbacks, 0)
                self.assertEqual(graph.signature, signature)
                self.assertEqual(graph.permissions.signature(), helper_signature)
                # No graph was captured twice, and the graphs of before are the ones in use.
                self.assertEqual(graph.captures, len(graph.graphs))
                for key, entry in kept.items():
                    self.assertIs(graph.graphs[key], entry, key)
                x = captured.interaction
                self.assertEqual((x.track_count, x.track_storage_growths), (count + 1, 0))
                self.assertEqual(int(x._live.numpy()[1]), count + 1)
                self.assertEqual(captured.coupler.coupling_graph_launches, 0)

    def test_replayed_force_graph_gives_the_forces_of_the_time_step_it_is_launched_with(self):
        # The tunnel drag of the ALM law is penalty * dt * sliding speed. With a candidate needle
        # that slides along its shaft the interface forces depend on the time step, by a few
        # thousandths of their size between dt and dt / 2 here. The steps of the test above
        # hardly slide, so their endpoints barely see the time step of the force kernels; this
        # candidate does. One graph, captured at dt, must give at every other time step the
        # forces of the eager evaluation at that time step.
        with pair("alm") as (_, captured):
            c, alm, body = captured.coupler, captured.interaction.tunnel_alm, captured.coupler.needle_body
            velocity = captured.state_0.body_qd.numpy().copy()
            velocity[body, :3] += 1e-2 * np.array([0.6, 0.0, -0.8])
            velocity[body, 3:] += (0.0, 0.5, 0.0)
            graph, forces = c._host_force_graph(captured.state_0, captured.control, _DT), {}
            for index, dt in enumerate((_DT, _DT, _DT / 2, 0.37 * _DT, _DT)):
                results = []
                for replay in (True, False):
                    c._candidate.assign(captured.state_0)
                    c._candidate.body_qd.assign(velocity)
                    alm.begin_step()
                    try:
                        if replay:
                            self.assertIs(c._host_force_graph(captured.state_0, captured.control, dt), graph)
                            graph.evaluate(dt, False)
                        else:
                            c._evaluate(c._candidate, dt, trial=True)
                        results.append((c._forces.numpy().copy(), c._candidate.body_f.numpy().copy()))
                    finally:
                        alm.discard()
                self.assertEqual((graph.captures, graph.launches, graph.fallbacks), (1, index + 1, 0))
                np.testing.assert_array_equal(c.step_time.seconds.numpy(), [np.float32(dt)])
                for actual, expected, name in zip(*results, ("nodal forces", "body wrenches"), strict=True):
                    scale = np.abs(expected).max()
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5 * scale, err_msg=f"{name} at {dt}")
                forces[dt] = results[0]
            for dt in (_DT / 2, 0.37 * _DT):
                for changed, base in zip(forces[dt], forces[_DT], strict=True):
                    self.assertGreater(np.abs(changed - base).max(), 1e-3 * np.abs(base).max())

    def test_each_recorder_state_keeps_its_own_graphs_and_a_recorder_flip_drops_none(self):
        # A host-controlled trial loop evaluates with the surface recorder bound (first contact,
        # or a correction with history) or not bound (a punctured solve without history). The
        # two have different signatures. Each keeps its own graphs under keys that name it, so
        # going from one to the other and back launches the graph that was captured before.
        # Only a changed signature of one recorder state drops graphs: those of that state.
        from test_surface_pc_capture import contact, prepare_forces

        with contact(capture=True) as (ex, c, _native, pc):
            dt = ex.sim_dt
            graph = prepare_forces(ex, c, pc)
            bound = graph.signature
            graph.evaluate(dt, False)
            self.assertEqual(list(graph.graphs), [(True, False, True)])
            first, released = graph.graphs[(True, False, True)], census.releases
            pc.begin_step()
            graph.prepare(dt)
            unbound = graph.signature
            self.assertNotEqual(unbound, bound)
            self.assertEqual(graph.signatures, {True: bound, False: unbound})
            graph.evaluate(dt, False)
            self.assertEqual(sorted(graph.graphs), [(False, False, True), (True, False, True)])
            second = graph.graphs[(False, False, True)]
            pc.begin_step()
            self.assertTrue(pc.begin_trials(ex.state_0, ex.control))
            graph.prepare(dt)
            self.assertEqual(graph.signature, bound)
            graph.evaluate(dt, False)
            self.assertIs(graph.graphs[(True, False, True)], first)
            self.assertIs(graph.graphs[(False, False, True)], second)
            self.assertEqual((graph.captures, graph.launches, graph.fallbacks), (2, 3, 0))
            self.assertEqual(census.releases, released)
            # Other storage of the recorder: the one graph of the bound state is dropped and counted.
            pc.anchors.matrices = wp.clone(pc.anchors.matrices)
            graph.prepare(dt)
            self.assertNotEqual(graph.signature, bound)
            self.assertEqual(list(graph.graphs), [(False, False, True)])
            self.assertIs(graph.graphs[(False, False, True)], second)
            self.assertEqual(census.releases, released + 1)

    def test_start_up_entry_captures_every_variant_once_and_evaluations_then_only_launch(self):
        # The entry captures the four variants of both recorder states and launches each once,
        # without counting a launch. Evaluations of a trial loop then find their graph,
        # whichever recorder state and variant they are.
        from test_surface_pc_capture import contact, prepare_forces

        with contact(capture=True) as (ex, c, _native, pc):
            dt = ex.sim_dt
            graph = prepare_forces(ex, c, pc)
            graph.capture_graphs(ex.state_0, ex.control, dt)
            self.assertEqual(sorted(graph.graphs), _VARIANTS)
            self.assertEqual((graph.captures, graph.launches, graph.primal_reuses, graph.fallbacks), (8, 0, 0, 0))
            self.assertEqual([entry.graph_exec is None for entry in graph.graphs.values()], [False] * 8)
            # The recorder is as the begin phase of an attempt leaves it.
            self.assertFalse(pc.available)
            self.assertIsNone(ex.interaction._surface_tangent_anchors)
            self.assertEqual(int(pc.valid.numpy()[0]), 0)
            kept, captures, released = dict(graph.graphs), census.captures, census.releases
            # Every variant has its graph: a second call issues nothing.
            with patch.object(wp, "capture_launch", side_effect=AssertionError("launched")):
                graph.capture_graphs(ex.state_0, ex.control, dt)
            for surface in (True, False):
                pc.begin_step()
                if surface:
                    self.assertTrue(pc.begin_trials(ex.state_0, ex.control))
                graph.prepare(dt)
                for previous, fresh in ((False, True), (True, True), (False, False), (True, False)):
                    packet = graph.evaluate(dt, previous, fresh=fresh)
                    self.assertEqual(packet.shape, (8 if surface else 6,))
            self.assertEqual((graph.captures, graph.launches, graph.primal_reuses, graph.fallbacks), (8, 8, 4, 0))
            self.assertEqual((census.captures, census.releases), (captures, released))
            self.assertEqual(sorted(graph.graphs), _VARIANTS)
            for key, entry in kept.items():
                self.assertIs(graph.graphs[key], entry, key)

    def test_empty_track_free_air_trials_match_without_host_force_queries(self):
        with ExitStack() as stack:
            examples = [stack.enter_context(_example()) for _ in range(2)]
            for index, example in enumerate(examples):
                _prepare(example)
                example.start()
                example.coupler.capture_fem = True
                example.coupler.capture_coupling = bool(index)
                example.coupler._rigid_bridge = None
                example.coupler._shaft_preconditioner = None
            for _ in range(4):
                for example in examples:
                    example.coupler.step(example.state_0, example.state_1, example.control, _DT)
                for name in graph_tests._FIELDS:
                    np.testing.assert_allclose(
                        getattr(examples[1].state_1, name).numpy(),
                        getattr(examples[0].state_1, name).numpy(),
                        rtol=3e-6,
                        atol=5e-8,
                    )
                for example in examples:
                    example.state_0, example.state_1 = example.state_1, example.state_0
            graph = examples[1].coupler._host_trial_forces
            self.assertIsNotNone(graph)
            self.assertGreater(graph.launches, 0)
            self.assertEqual(graph.fallbacks, 0)
            self.assertEqual(examples[1].interaction.track_count, 0)

    def test_conservative_boundary_fallback_restarts_original_transaction(self):
        with pair("alm") as (reference, captured):
            # Prepare once, then make the exact certificate request fallback.
            graph = captured.coupler._host_force_graph(captured.state_0, captured.control, _DT)
            with patch.object(graph, "evaluate", side_effect=TrialForceFallback()):
                self.step_pair(reference, captured, _DT)
            self.assertIsNone(captured.interaction._trial_fields)

    def test_rejected_step_restores_every_transaction_field(self):
        for failure in ("budget", "swept_motion"):
            with self.subTest(failure=failure), pair("alm") as (_, captured):
                saved = graph_tests._snapshot(captured)
                if failure == "budget":
                    captured.coupler.max_iterations = 1
                    with self.assertRaises(NeedleCouplingNotConverged):
                        captured.coupler.step(captured.state_0, captured.state_1, captured.control, _DT)
                else:
                    with patch.object(
                        captured.interaction, "check_step_resolution", side_effect=NeedleStepTooLarge("test")
                    ):
                        with self.assertRaises(NeedleStepTooLarge):
                            captured.coupler.step(captured.state_0, captured.state_1, captured.control, _DT)
                self.assert_rollback(captured, saved)
                self.assertGreater(captured.coupler._host_trial_forces.launches, 0)

    def test_late_metadata_fallback_preserves_native_history(self):
        with pair("alm") as (reference, captured):
            with patch.object(captured.interaction, "_stage_certified_trial", return_value=False):
                self.step_pair(reference, captured, _DT)
            self.assertGreater(captured.coupler._host_trial_forces.fallbacks, 0)

    def test_nonfinite_state_rejected_before_refitting_geometry(self):
        with pair("penalty") as (_, captured):
            c = captured.coupler
            graph = c._host_force_graph(captured.state_0, captured.control, _DT)
            c._candidate.assign(captured.state_0)
            values = c._candidate.particle_q.numpy()
            values[0, 0] = np.nan
            c._candidate.particle_q.assign(values)
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                graph.evaluate(_DT, False)


if __name__ == "__main__":
    unittest.main()
