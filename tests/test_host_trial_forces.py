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
from newton_solver_suturebot._src.host_trial_forces import TrialForceFallback


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
