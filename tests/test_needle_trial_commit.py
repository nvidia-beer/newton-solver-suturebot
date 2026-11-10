# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Retaining a converged force trial must match a fresh final evaluation."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleTissueInteraction, boundary_faces


class NeedleTrialCommitTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], np.float32)

    def build(self, *, fixed=()):
        tets = np.array([[0, 1, 2, 3]], np.int32)
        samples = np.zeros((11, 3))
        samples[:, 2] = np.linspace(0, -0.01, 11)
        interaction = NeedleTissueInteraction(
            self.vertices, tets, boundary_faces(tets)[0], fixed, samples, 0.0005, samples[0], device=self.device
        )
        state = SimpleNamespace(
            particle_q=wp.array(self.vertices, dtype=wp.vec3, device=self.device),
            particle_qd=wp.zeros(4, dtype=wp.vec3, device=self.device),
            body_q=wp.array([[0.002, 0.002, -0.0001, 0, 0, 0, 1]], dtype=wp.transform, device=self.device),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
        )
        return interaction, state, wp.zeros(4, dtype=wp.vec3, device=self.device)

    @staticmethod
    def history(interaction):
        return (
            interaction.status,
            interaction.reason,
            interaction.punctured,
            interaction.puncture_count,
            interaction.entry_count,
            interaction.exit_count,
            interaction.track_count,
            interaction.track_length,
        )

    def test_retained_trial_matches_fresh_forces_events_and_track_through_reentry(self):
        retained, fresh = self.build(), self.build()
        path = np.r_[
            -0.0001, np.linspace(0, 0.0012, 9), np.linspace(0.00105, -0.0012, 16), np.linspace(-0.00105, 0.0012, 16)
        ]
        for iteration, z in enumerate(path):
            for use_retained, (interaction, state, forces) in zip((True, False), (retained, fresh), strict=True):
                start = SimpleNamespace(particle_q=wp.clone(state.particle_q), body_q=wp.clone(state.body_q))
                moved = self.vertices.copy()
                moved[3, 2] += 1e-5 * np.sin(iteration * 0.1)
                state.particle_q.assign(moved)
                state.body_q.assign([[0.002, 0.002, z, 0, 0, 0, 1]])
                state.body_qd.assign([[0, 0, 0.001 if iteration < 10 or iteration >= 26 else -0.001, 0, 0, 0]])
                state.body_f.zero_()
                forces.zero_()
                before = self.history(interaction)
                engagement = interaction._track_engaged.numpy().copy()
                interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True, implicit=True)
                self.assertEqual(self.history(interaction), before)
                np.testing.assert_array_equal(interaction._track_engaged.numpy(), engagement)
                interaction.check_step_resolution(start, state, 0)
                if use_retained:
                    with patch.object(
                        interaction, "_add_forces", side_effect=AssertionError("Unexpected force reevaluation")
                    ):
                        interaction.retain_trial_forces(start, 0)
                        interaction.accept(state, 0)
                else:
                    state.body_f.zero_()
                    forces.zero_()
                    interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True, start_state=start)
                    interaction.accept(state, 0)
                self.assertNotEqual(interaction.status, "failed", interaction.reason)
            self.assertEqual(self.history(retained[0]), self.history(fresh[0]))
            np.testing.assert_allclose(retained[2].numpy(), fresh[2].numpy(), rtol=2e-6, atol=2e-10)
            np.testing.assert_allclose(retained[1].body_f.numpy(), fresh[1].body_f.numpy(), rtol=2e-6, atol=2e-10)
            np.testing.assert_array_equal(retained[0]._track_engaged.numpy(), fresh[0]._track_engaged.numpy())
            self.assertEqual([e["kind"] for e in retained[0].events], [e["kind"] for e in fresh[0].events])
        self.assertEqual((retained[0].puncture_count, retained[0].entry_count, retained[0].exit_count), (1, 2, 1))

    def test_discard_and_single_consumption_prevent_stale_trial_reuse(self):
        interaction, state, forces = self.build()
        before = self.history(interaction)
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True, implicit=True)
        interaction.discard_trial_forces()
        self.assertEqual(self.history(interaction), before)
        with self.assertRaisesRegex(RuntimeError, "No implicit"):
            interaction.retain_trial_forces(state, 0)
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True, implicit=False)
        with self.assertRaisesRegex(RuntimeError, "No implicit"):
            interaction.retain_trial_forces(state, 0)
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True, implicit=True)
        interaction.retain_trial_forces(state, 0)
        with self.assertRaisesRegex(RuntimeError, "No implicit"):
            interaction.retain_trial_forces(state, 0)

    def test_retained_failed_candidate_preserves_failure_without_committing_track(self):
        retained, fresh = self.build(fixed=(0, 1, 2)), self.build(fixed=(0, 1, 2))
        for use_retained, (interaction, state, forces) in zip((True, False), (retained, fresh), strict=True):
            start = SimpleNamespace(particle_q=wp.clone(state.particle_q), body_q=wp.clone(state.body_q))
            state.body_q.assign([[0.002, 0.002, 0.0001, 0, 0, 0, 1]])
            interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True, implicit=True)
            self.assertNotEqual(interaction.status, "failed")
            if use_retained:
                interaction.retain_trial_forces(start, 0)
            else:
                state.body_f.zero_()
                forces.zero_()
                interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True, start_state=start)
            interaction.accept(state, 0)
            self.assertEqual(interaction.status, "failed")
            self.assertEqual((interaction.track_count, interaction.puncture_count, interaction.entry_count), (0, 0, 0))
        self.assertEqual(self.history(retained[0]), self.history(fresh[0]))


if __name__ == "__main__":
    unittest.main()
