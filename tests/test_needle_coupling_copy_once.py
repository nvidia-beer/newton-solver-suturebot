# SPDX-License-Identifier: Apache-2.0
"""Moving the candidate reset outside the trial loop preserves state ownership."""

import unittest
from unittest.mock import patch

import numpy as np
import test_needle_coupling_graph as graph_tests

from newton_solver_suturebot import NeedleCouplingNotConverged, NeedleStepTooLarge


def poison_candidate(example):
    for name in graph_tests._FIELDS:
        array = getattr(example.coupler._candidate, name)
        array.assign(np.full_like(array.numpy(), np.nan))


class NeedleCouplingCopyOnceTests(unittest.TestCase):
    setUp = graph_tests.NeedleCouplingGraphTests.setUp
    assert_equivalent = graph_tests.NeedleCouplingGraphTests.assert_equivalent
    assert_rollback = graph_tests.NeedleCouplingGraphTests.assert_rollback

    def step_pair(self, reference, captured, dt):
        poison_candidate(captured)
        graph_tests.NeedleCouplingGraphTests.step_pair(self, reference, captured, dt)
        # In particular, preserve the externally visible FEM force scratch.
        np.testing.assert_allclose(
            captured.state_1.particle_f.numpy(), reference.state_1.particle_f.numpy(), rtol=3e-6, atol=5e-8
        )

    def test_poisoned_scratch_replay_and_swapped_inputs_match_loaded_reference(self):
        for mode in ("penalty", "alm"):
            with self.subTest(mode=mode), graph_tests._pair(mode) as (reference, captured):
                for index, dt in enumerate((graph_tests._DT, graph_tests._DT, graph_tests._DT, graph_tests._DT / 2)):
                    for example in (reference, captured):
                        example.drive.prepare_timestep(dt)
                        example.solver.refresh_tangent(example.state_0.particle_q, dt)
                        example.control.joint_f.assign(np.array([0, (-1) ** index * 1e-5, 0, 2e-8, 0, 0], np.float32))
                        array = example.state_0.particle_f
                        array.assign(np.full_like(array.numpy(), (index + 1) * 1e-5))
                    self.step_pair(reference, captured, dt)
                    self.assertEqual(captured.coupler.coupling_graph_fallbacks, 0)
                    if index == 0:
                        self.assertGreater(captured.coupler.iterations, 1)
                    for example in (reference, captured):
                        if index == 0:
                            example.state_0.assign(example.state_1)
                        else:
                            example.state_0, example.state_1 = example.state_1, example.state_0
                self.assertEqual(captured.coupler.coupling_graph_launches, 4)
                # One capture per state buffer. The halved last step replays the graph of its
                # buffer: the time step and the drive damping scaled by it are device data.
                self.assertEqual(captured.coupler.coupling_graph_captures, 2)

    def test_poisoned_rejected_attempts_do_not_escape_then_retry_commits_once(self):
        with graph_tests._pair("alm") as (reference, captured):
            saved = graph_tests._snapshot(captured)
            captured.coupler.max_iterations = 1
            poison_candidate(captured)
            with patch.object(captured.interaction, "accept", wraps=captured.interaction.accept) as accept:
                with self.assertRaises(NeedleCouplingNotConverged):
                    captured.coupler.step(captured.state_0, captured.state_1, captured.control, graph_tests._DT)
                self.assertEqual(accept.call_count, 0)
            self.assert_rollback(captured, saved)
            captured.coupler.max_iterations = 160
            poison_candidate(captured)
            with (
                patch.object(
                    captured.interaction, "check_step_resolution", side_effect=NeedleStepTooLarge("late guard")
                ),
                patch.object(captured.interaction, "accept", wraps=captured.interaction.accept) as accept,
            ):
                with self.assertRaisesRegex(NeedleStepTooLarge, "late guard"):
                    captured.coupler.step(captured.state_0, captured.state_1, captured.control, graph_tests._DT)
                self.assertEqual(accept.call_count, 0)
            self.assert_rollback(captured, saved)
            self.step_pair(reference, captured, graph_tests._DT)
            self.assertEqual(captured.coupler.coupling_graph_launches, 3)

    def test_poisoned_candidate_near_exit_falls_back_to_original_authorization(self):
        # This existing fixture advances through a real exterior crossing. Our
        # overridden step_pair injects poison before the device attempt, then
        # retains the original strict native/history/force and one-commit checks.
        graph_tests.NeedleCouplingGraphTests.test_near_exit_geometry_uses_original_authorization_and_commits_once(self)


if __name__ == "__main__":
    unittest.main()
