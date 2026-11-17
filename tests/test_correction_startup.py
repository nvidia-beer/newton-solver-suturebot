# SPDX-License-Identifier: Apache-2.0
"""Start-up entry of the shaft correction once a response basis has storage: the solve from a start guess.

The correction of a host-controlled trial loop starts its solve from a guess as soon as the
response-basis predictor has a core, and prepares the host packet of that guess. Both are
graphs of their own, keyed on the storage of the core. The start-up entry captures them when
that storage exists at start-up; without it they are captured where a run first needs them.
The start-up of the grasp scene makes that storage itself (``AffineGuessManager.prepare``), and
the scene tests check the two graphs it then captures. This test is about storage that appears
after start-up: it keeps the predictor from making it ahead, makes the core by hand, the way
the predictor does at a first track sample, and then calls the entry.
"""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from test_grasp_scene import _host_path_sites, _scene

from newton_solver_suturebot._src.solvers.fem.affine_guess import AffineGuessManager, DeviceBasisGuess


class StartGuessStartupTests(unittest.TestCase):
    def setUp(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("The start-up entry captures CUDA graphs")

    def test_entry_captures_packet_preparation_and_solve_of_a_start_guess_and_gives_its_counters_back(self):
        # A predictor that makes no storage ahead of the first track sample, as for a
        # configuration it does not serve: the start-up of the scene then finds no core.
        with (
            patch.object(AffineGuessManager, "prepare", return_value=False),
            _scene(rigid_backend="mujoco-warp") as (scene, _),
        ):
            coupler, dt = scene.coupler, scene.sim_dt
            pc, manager = coupler._shaft_preconditioner, scene.solver._affine_predictor
            # Start-up found no core: only the solve without a start guess has its graph.
            self.assertIsNone(manager.core)
            self.assertIsNone(pc._guess_storage)
            self.assertEqual((sorted(pc._graphs), pc.graph_captures, pc.graph_launches), ([False], 1, 0))
            plain = pc._graphs[False][1]
            contact = pc._input_preparation.replay
            # Storage of the response basis, as the predictor makes it at the first track sample.
            coupler._refresh_projection()
            fixed = scene.solver._fixed_particles
            manager.core = DeviceBasisGuess(fixed.compact_operator, fixed.free, capacity=manager.capacity)
            manager.structural_key = manager._structure()

            def attempt(issue):
                opened = coupler._open_attempt(scene.state_0, scene.control, dt, None)
                opened.predictor = None
                try:
                    coupler._begin_attempt(opened)
                    coupler._candidate.assign(scene.state_0)
                    return issue()
                finally:
                    coupler._close_attempt(opened, scene.state_0, False)

            self.assertTrue(attempt(lambda: coupler.capture_contact_graphs(dt)))
            guess = pc._guess_storage
            self.assertIsNotNone(guess)
            self.assertIs(guess.core, manager.core)
            # Attached for the passes of the entry only.
            self.assertIsNone(pc._woodbury_guess)
            self.assertEqual(sorted(pc._graphs), [False, True])
            self.assertIs(pc._graphs[False][1], plain)
            started = pc._graphs[True][1]
            packet = guess._packet_preparation.replay
            self.assertEqual([graph.graph_exec is None for graph in (started, packet.graph)], [False, False])
            # The captures are counted; the passes as written and the launches are given back.
            self.assertEqual((pc.graph_captures, pc.graph_launches, pc.native_mass_fallbacks), (2, 0, 0))
            self.assertEqual((packet.captures, packet.launches, packet.eager), (1, 0, 0))
            self.assertEqual((contact.captures, contact.launches, contact.eager), (1, 0, 0))
            self.assertEqual((guess.preparations, guess.ready, guess.fallbacks, guess.fallback_reasons), (0, 0, 0, {}))
            self.assertEqual(guess.last_reason, "unbound")
            np.testing.assert_array_equal(guess.counters.numpy(), 0)
            self.assertFalse(pc._native_mass_ready)

            # A correction of a host-controlled trial loop that starts from the guess: launched.
            def correction():
                rigid = scene.rigid_solver
                rigid.mj_model.opt.timestep = dt / rigid.rigid_substeps
                rigid.sync_native_shadow(forward=True)
                coupler._residual.zero_()
                self.assertTrue(pc.stage(dt, tape=True))
                pc.prepare_rigid_metric(dt)
                pc._prepare_inputs(coupler._candidate, dt, coupler._residual)
                pc._woodbury_guess = guess
                guess.prepare(manager)
                surface = pc.surface
                guess._packet_preparation.run(surface.anchors.count, surface.valid)
                pc._solve_graph(coupler._residual, coupler._direction)

            sites = _host_path_sites()
            attempt(correction)
            self.assertEqual(_host_path_sites(), sites)
            self.assertEqual((pc.graph_captures, pc.graph_launches), (2, 1))
            self.assertEqual((packet.captures, packet.launches, packet.eager), (1, 1, 0))
            self.assertEqual((contact.captures, contact.launches, contact.eager), (1, 1, 0))
            self.assertIs(pc._graphs[True][1], started)
            self.assertIs(pc._graph, started)
            # Every graph of start-up is there: a second call of the entry captures nothing.
            self.assertTrue(attempt(lambda: coupler.capture_contact_graphs(dt)))
            self.assertEqual(_host_path_sites(), sites)
            self.assertEqual((pc.graph_captures, packet.captures, contact.captures), (2, 1, 1))


if __name__ == "__main__":
    unittest.main()
