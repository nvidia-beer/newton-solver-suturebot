# SPDX-License-Identifier: Apache-2.0
"""Preparation replay reads current geometry/ALM values and preserves host branches."""

import gc
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_combined_contact_preconditioner as combined_fixture
import test_woodbury_guess as woodbury_fixture
import warp as wp
from test_needle_insertion_example import _example

from newton_solver_suturebot._src.solvers.coupled.acceleration.woodbury_guess import WoodburyInitialGuess
from newton_solver_suturebot._src.solvers.coupled.interface.tunnel_alm import _TunnelALM
from newton_solver_suturebot._src.utils.capture import ScopedCapture, census
from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


class ContactPreparationTests(unittest.TestCase):
    def test_current_geometry_residual_scalar_and_storage_changes_match_eager(self):
        with _example() as ex:
            source = combined_fixture.CombinedContactAlgebraTests()
            h, surface, pc = source.fixture(ex)
            values = np.linspace(-2e-5, 3e-5, 3 * pc.n)
            residual = wp.array(values, dtype=wp.float64, device=ex.device)
            self.assertTrue(pc.prepare(h))
            fields = ("matrices", "levers", "rigid_blocks", "diagonal", "inverse_diagonal", "rhs")
            # What a trial does on CUDA: a new binding is issued as written, its second use is
            # captured and launched, later uses are launched. The time step is device data of
            # the preparation and no part of its binding: trials 3 and 4 change nothing else
            # and replay the graph of trial 1. A scalar of the graph (friction) and replaced
            # storage (levers) are new bindings.
            actions = "eager capture launch launch launch eager capture eager capture launch".split()
            counts, base = [0, 0, 0], h
            for trial, action in enumerate(actions):
                previous = h
                if trial == 3:
                    h = base / 2
                if trial == 4:
                    # The remainder of a frame is no halving of the base step.
                    h = 0.37 * base
                if trial == 5:
                    ex.interaction.friction *= 1.1
                if trial == 7:
                    pc.levers = wp.empty_like(pc.levers)
                if trial == 8:
                    surface.valid.zero_()
                if trial == 9:
                    surface.valid.fill_(1)
                    surface.anchors.matrices.fill_(wp.mat33(0.02))
                residual.assign(values * (trial - 2.5))
                ex.state_0.particle_qd.fill_(wp.vec3(0.001 * trial, -0.0002, 0.0001))
                self.assertTrue(pc.prepare(h))
                before_gc = gc.isenabled()
                pc._prepare_inputs(ex.state_0, h, residual)
                result = {name: getattr(pc, name).numpy().copy() for name in fields}
                self.assertEqual(gc.isenabled(), before_gc)
                replay = pc._input_preparation.replay
                if ex.device.is_cuda:
                    counts[0] += action == "eager"
                    counts[1] += action == "capture"
                    counts[2] += action != "eager"
                    self.assertEqual([replay.eager, replay.captures, replay.launches], counts, f"trial {trial}")
                np.testing.assert_array_equal(pc.time_step.numpy(), [np.float32(h)])
                if h != previous:
                    # The inputs depend on the time step: issued with the previous one they differ,
                    # so a graph that kept the time step of its capture fails the comparison below.
                    pc._enqueue_inputs(ex.state_0, previous, residual)
                    for name in ("matrices", "diagonal"):
                        self.assertFalse(np.allclose(getattr(pc, name).numpy(), result[name], rtol=1e-3, atol=0), name)
                pc._enqueue_inputs(ex.state_0, h, residual)
                for name in fields:
                    np.testing.assert_allclose(
                        getattr(pc, name).numpy(), result[name], rtol=3e-7, atol=2e-8, err_msg=f"{trial}: {name}"
                    )
            if not ex.device.is_cuda:
                self.assertEqual(replay.launches, 0)
                self.assertEqual(replay.eager, len(actions))

    def test_dropped_preparation_graph_is_counted_as_released(self):
        # A new binding retires the graph of the binding before it. The census of the process
        # counts that graph as released; a binding without a graph has nothing to count.
        with _example() as ex:
            h, _surface, pc = combined_fixture.CombinedContactAlgebraTests().fixture(ex)
            residual = wp.zeros(3 * pc.n, dtype=wp.float64, device=ex.device)
            self.assertTrue(pc.prepare(h))
            released = census.releases
            for _ in range(3):
                pc._prepare_inputs(ex.state_0, h, residual)
            replay = pc._input_preparation.replay
            self.assertEqual(census.releases, released)
            ex.interaction.friction *= 1.1
            pc._prepare_inputs(ex.state_0, h, residual)
            # On the CPU every pass runs as written: no graph exists to drop.
            dropped = int(ex.device.is_cuda)
            self.assertEqual(replay.captures, dropped)
            self.assertEqual(census.releases, released + dropped)
            # The new binding ran as written once and has no graph yet.
            self.assertIsNone(replay.graph)
            pc.invalidate_preparation()
            self.assertEqual(census.releases, released + dropped)

    def test_start_up_entry_captures_preparation_and_solve_and_gives_back_the_work_it_counted(self):
        # The entry of the shaft correction issues its contact preparation and its solve twice,
        # with a zero residual: as written for the new binding, then captured and launched. The
        # captures are counted, the launches and the pass as written are not. A correction of a
        # trial loop with the same arrays then launches both graphs and captures nothing.
        with _example() as ex:
            h, _surface, pc = combined_fixture.CombinedContactAlgebraTests().fixture(ex)
            values = np.linspace(-2e-5, 3e-5, 3 * pc.n)
            residual = wp.zeros(3 * pc.n, dtype=wp.float64, device=ex.device)
            out = wp.empty_like(residual)
            if not (ex.device.is_cuda and wp.is_conditional_graph_supported()):
                # No graph is captured here: the entry declines and issues nothing.
                self.assertFalse(pc.capture_host_graphs(ex.state_0, h, residual, out))
                self.assertIsNone(pc._input_preparation)
                self.assertEqual((pc.graph_captures, pc.graph_launches), (0, 0))
                return
            pc._last_dt = None
            self.assertTrue(pc.capture_host_graphs(ex.state_0, h, residual, out))
            replay = pc._input_preparation.replay
            self.assertEqual((replay.eager, replay.captures, replay.launches), (0, 1, 0))
            self.assertEqual((pc.graph_captures, pc.graph_launches, pc.native_mass_fallbacks), (1, 0, 0))
            self.assertEqual([graph.graph_exec is None for graph in (replay.graph, pc._graph)], [False, False])
            self.assertFalse(pc._native_mass_ready)
            self.assertIsNone(pc._woodbury_guess)
            self.assertIsNone(pc._last_dt)
            # A zero residual has a zero correction.
            np.testing.assert_array_equal(out.numpy(), 0.0)
            # Both bindings have their graphs: a second call captures nothing and counts nothing.
            graphs, captures = (replay.graph, pc._graph), (census.captures, census.solver_captures)
            self.assertTrue(pc.capture_host_graphs(ex.state_0, h, residual, out))
            self.assertEqual((replay.eager, replay.captures, replay.launches), (0, 1, 0))
            self.assertEqual((pc.graph_captures, pc.graph_launches), (1, 0))
            # A correction of a trial loop: launched, with the values of the same correction as written.
            residual.assign(values)
            pc.apply(ex.state_0, h, residual, out)
            self.assertEqual((census.captures, census.solver_captures), captures)
            self.assertEqual((replay.eager, replay.captures, replay.launches), (0, 1, 1))
            self.assertEqual((pc.graph_captures, pc.graph_launches), (1, 1))
            self.assertIs(replay.graph, graphs[0])
            self.assertIs(pc._graph, graphs[1])
            fields = ("matrices", "levers", "rigid_blocks", "diagonal", "inverse_diagonal", "rhs")
            launched = {name: getattr(pc, name).numpy().copy() for name in fields}
            correction = out.numpy().copy()
            self.assertGreater(np.abs(correction).max(), 0.0)
            pc._enqueue_inputs(ex.state_0, h, residual)
            for name in fields:
                np.testing.assert_allclose(
                    getattr(pc, name).numpy(), launched[name], rtol=3e-7, atol=2e-8, err_msg=name
                )
            pc._solve(residual, out)
            np.testing.assert_allclose(out.numpy(), correction, rtol=1e-3, atol=1e-3 * np.abs(correction).max())

    def test_current_alm_duals_and_scalar_regime_are_read_without_history_mutation(self):
        with _example() as ex:
            h, _surface, pc = combined_fixture.CombinedContactAlgebraTests().fixture(ex)
            alm = ex.interaction.tunnel_alm = _TunnelALM(
                ex.interaction, penalty_per_m=6000, compliance_per_m=1 / 6000, clearance=1e-5, tolerance=1e-6
            )
            alm.begin_step()
            residual = wp.ones(3 * pc.n, dtype=wp.float64, device=ex.device)
            try:
                self.assertTrue(pc.prepare(h))
                for trial in range(6):
                    alm.normal.fill_(wp.vec3(0, trial * 0.01, -0.03))
                    alm.tangent.fill_(trial * 0.002)
                    if trial == 3:
                        alm.compliance_per_m = 0
                        alm.penalty_per_m *= 1.5
                    normal, tangent = alm.normal.numpy().copy(), alm.tangent.numpy().copy()
                    pc._prepare_inputs(ex.state_0, h, residual)
                    actual = pc.matrices.numpy().copy(), pc.diagonal.numpy().copy()
                    pc._enqueue_inputs(ex.state_0, h, residual)
                    np.testing.assert_array_equal(pc.matrices.numpy(), actual[0])
                    np.testing.assert_allclose(pc.diagonal.numpy(), actual[1], rtol=3e-7, atol=2e-8)
                    np.testing.assert_array_equal(alm.normal.numpy(), normal)
                    np.testing.assert_array_equal(alm.tangent.numpy(), tangent)
                if ex.device.is_cuda:
                    self.assertEqual(pc._input_preparation.replay.captures, 2)
                    self.assertEqual(pc._input_preparation.replay.launches, 4)
            finally:
                alm.discard()

    def test_custom_preparation_and_gradient_buffers_keep_original_eager_path(self):
        with _example() as ex:
            h, _surface, pc = combined_fixture.CombinedContactAlgebraTests().fixture(ex)
            residual = wp.zeros(3 * pc.n, dtype=wp.float64, device=ex.device)
            self.assertTrue(pc.prepare(h))
            pc._prepare_inputs(ex.state_0, h, residual)
            with patch.object(pc, "_enqueue_inputs", wraps=pc._enqueue_inputs) as original:
                pc._prepare_inputs(ex.state_0, h, residual)
                original.assert_called_once()
                self.assertIsNone(pc._input_preparation.replay.key)
            gradient = wp.zeros_like(residual, requires_grad=True)
            pc._prepare_inputs(ex.state_0, h, gradient)
            self.assertIsNone(pc._input_preparation.replay.key)
            pc.capture_preparation = False
            pc._prepare_inputs(ex.state_0, h, residual)
            self.assertIsNone(pc._input_preparation.replay.key)

    def test_begin_keeps_preparation_but_invalidation_reset_and_close_retire_owners(self):
        with _example() as ex:
            h, _surface, pc = combined_fixture.CombinedContactAlgebraTests().fixture(ex)
            ex.coupler._shaft_preconditioner = pc
            residual = wp.ones(3 * pc.n, dtype=wp.float64, device=ex.device)
            self.assertTrue(pc.prepare(h))

            def warm():
                pc._prepare_inputs(ex.state_0, h, residual)
                pc._prepare_inputs(ex.state_0, h, residual)
                return pc._input_preparation.replay

            replay = warm()
            key, graph, owners = replay.key, replay.graph, replay.owners
            pc.begin_step()
            self.assertEqual(replay.key, key)
            self.assertIs(replay.graph, graph)
            self.assertIs(replay.owners, owners)
            ex.coupler.invalidate_fem_graph()
            self.assertIsNone(replay.key)
            self.assertIsNone(replay.graph)
            self.assertEqual(replay.owners, ())
            warm()
            # The public close hook releases preparation before response-basis
            # storage. It also works before a scene has a basis provider.
            GraspScene.close(ex)
            self.assertIsNone(replay.key)
            self.assertEqual(replay.owners, ())
            warm()
            ex._reset_interaction_state()
            self.assertIsNone(replay.key)
            self.assertIsNone(replay.graph)
            self.assertEqual(replay.owners, ())
            self.assertIsNone(ex.coupler)

    def test_cached_preparation_inside_outer_capture_keeps_owners_without_synchronizing(self):
        if not wp.get_device(DEVICE).is_cuda:
            self.skipTest("Nested CUDA capture must keep prior graph owners")
        with _example() as ex:
            h, _surface, pc = combined_fixture.CombinedContactAlgebraTests().fixture(ex)
            residual = wp.ones(3 * pc.n, dtype=wp.float64, device=ex.device)
            self.assertTrue(pc.prepare(h))
            pc._prepare_inputs(ex.state_0, h, residual)
            pc._prepare_inputs(ex.state_0, h, residual)
            replay = pc._input_preparation.replay
            key, graph, owners, launches = replay.key, replay.graph, replay.owners, replay.launches
            self.assertIsNotNone(graph)
            changed = wp.empty_like(residual)
            changed.fill_(2.0)
            pc._enqueue_inputs(ex.state_0, h, changed)
            expected = pc.rhs.numpy().copy()
            pc.rhs.zero_()
            with ScopedCapture(device=ex.device) as outer:
                with patch.object(replay, "invalidate", side_effect=AssertionError("Nested capture retired owners")):
                    pc._prepare_inputs(ex.state_0, h, changed)
            self.assertEqual(replay.key, key)
            self.assertIs(replay.graph, graph)
            self.assertIs(replay.owners, owners)
            self.assertEqual(replay.launches, launches)
            wp.capture_launch(outer.graph)
            np.testing.assert_array_equal(pc.rhs.numpy(), expected)
            # Explicit retirement rejects before mutation instead of poisoning
            # CUDA's active capture with a stream synchronization.
            with ScopedCapture(device=ex.device) as outer:
                with self.assertRaisesRegex(RuntimeError, "outside an enclosing"):
                    replay.invalidate()
                pc._enqueue_inputs(ex.state_0, h, changed)
            wp.capture_launch(outer.graph)
            np.testing.assert_array_equal(pc.rhs.numpy(), expected)
            self.assertIs(replay.graph, graph)
            self.assertIs(replay.owners, owners)
            replay.invalidate()
            self.assertIsNone(replay.graph)
            self.assertEqual(replay.owners, ())

    def test_different_cuda_stream_retires_previous_binding_and_preserves_output(self):
        if not wp.get_device(DEVICE).is_cuda:
            self.skipTest("Cross-stream CUDA graph ownership")
        with _example() as ex:
            h, _surface, pc = combined_fixture.CombinedContactAlgebraTests().fixture(ex)
            residual = wp.ones(3 * pc.n, dtype=wp.float64, device=ex.device)
            self.assertTrue(pc.prepare(h))
            first, second = wp.Stream(ex.device), wp.Stream(ex.device)
            with wp.ScopedStream(first):
                pc._prepare_inputs(ex.state_0, h, residual)
                pc._prepare_inputs(ex.state_0, h, residual)
            wp.synchronize_stream(first)
            expected = pc.rhs.numpy().copy()
            with wp.ScopedStream(second):
                pc._prepare_inputs(ex.state_0, h, residual)
                pc._prepare_inputs(ex.state_0, h, residual)
            wp.synchronize_stream(second)
            np.testing.assert_array_equal(pc.rhs.numpy(), expected)
            self.assertEqual(pc._input_preparation.replay.captures, 2)
            self.assertIs(pc._input_preparation.replay.stream, second)


class PacketPreparationTests(unittest.TestCase):
    def test_packet_replay_reads_changed_values_and_rebinds_support_arrays(self):
        fixture = woodbury_fixture.Tests()
        fixture.setUp()
        p, manager, actual = fixture.pre, fixture.manager, fixture.helper
        p.capture_preparation = True
        p.coupler.model = SimpleNamespace(requires_grad=False)
        p.coupler.fem_solver = SimpleNamespace(differentiable=False)
        reference = WoodburyInitialGuess(p)
        original = p.rhs.numpy().copy()
        packets = []
        for trial in range(8):
            p.rhs.assign(original * (trial - 1.5))
            p.levers.fill_(wp.vec3(0.003 * trial, 0.004, 0.005))
            # The shears of sliding anchors are read by the packet as the tangents and levers are.
            p.shears.fill_(wp.mat33(*([0.01 * trial] * 9)))
            if trial == 3:
                manager.core.support = wp.clone(manager.core.support)
                manager.basis_builds += 1
            if trial == 6:
                # New storage of the shears, as with a new track sample: the graph is not replayed.
                p.shears = wp.clone(p.shears)
            actual.prepare(manager)
            actual_packet = actual.packet.numpy().copy()
            actual_coefficients = actual.coefficients.numpy().copy()
            p.capture_preparation = False
            reference.prepare(manager)
            p.capture_preparation = True
            np.testing.assert_array_equal(actual_packet, reference.packet.numpy())
            np.testing.assert_array_equal(actual_coefficients, reference.coefficients.numpy())
            packets.append(actual_packet[woodbury_fixture._HEADER :].reshape(-1, 20)[: p.count, 8:17].copy())
        # Each trial's shear changed the anchor tangents of the packet.
        for before, after in zip(packets, packets[1:], strict=False):
            self.assertFalse(np.array_equal(before, after))
        replay = actual._packet_preparation.replay
        self.assertTrue(any(owner is p.shears for owner in replay.owners) or not p.device.is_cuda)
        if p.device.is_cuda:
            self.assertEqual(replay.captures, 3)
            self.assertEqual(replay.launches, 5)
        else:
            self.assertEqual(replay.eager, 8)
        manager.active = False
        actual.prepare(manager)
        self.assertEqual(actual.enabled.numpy()[0], 0)
        self.assertEqual(replay.launches, 5 if p.device.is_cuda else 0)
        actual._packet_preparation.replay.invalidate()
        self.assertIsNone(replay.graph)
        self.assertEqual(replay.owners, ())


if __name__ == "__main__":
    unittest.main()
