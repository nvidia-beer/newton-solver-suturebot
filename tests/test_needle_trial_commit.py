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
    def path():
        """Tip heights that enter the material, withdraw through the entry and enter again [m]."""
        return np.r_[
            -0.0001, np.linspace(0, 0.0012, 9), np.linspace(0.00105, -0.0012, 16), np.linspace(-0.00105, 0.0012, 16)
        ]

    def advance(self, interaction, state, forces, iteration, z, *, accept=True):
        """One implicit step of :meth:`path`: trial, resolution check, retained trial and its acceptance."""
        start = SimpleNamespace(particle_q=wp.clone(state.particle_q), body_q=wp.clone(state.body_q))
        moved = self.vertices.copy()
        moved[3, 2] += 1e-5 * np.sin(iteration * 0.1)
        state.particle_q.assign(moved)
        state.body_q.assign([[0.002, 0.002, z, 0, 0, 0, 1]])
        state.body_qd.assign([[0, 0, 0.001 if iteration < 10 or iteration >= 26 else -0.001, 0, 0, 0]])
        state.body_f.zero_()
        forces.zero_()
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True, implicit=True)
        interaction.check_step_resolution(start, state, 0)
        interaction.retain_trial_forces(start, 0)
        if accept:
            interaction.accept(state, 0)
            self.assertNotEqual(interaction.status, "failed", interaction.reason)

    def inserted(self):
        """An interaction after the first ten steps of :meth:`path`, each with a reader's request."""
        interaction, state, forces = self.build()
        for iteration, z in enumerate(self.path()[:10]):
            interaction.request_engaged_track_length()
            self.advance(interaction, state, forces, iteration, z)
        self.assertEqual(interaction.engaged_track_length, self.device_engaged_length(interaction))
        self.assertGreater(interaction.engaged_track_length, 0)
        return interaction, state, forces

    @staticmethod
    def device_engaged_length(interaction):
        """The engaged track length that the two device arrays hold now [m]."""
        lengths, engaged = interaction._track_lengths.numpy(), interaction._track_engaged.numpy()
        return float(np.sum(lengths[engaged != 0]))

    @staticmethod
    def committed(interaction):
        """Everything an acceptance commits, bit for bit."""
        x = interaction
        return (
            NeedleTrialCommitTests.history(x),
            [(event["kind"], event["position"].tobytes(), event["reused"]) for event in x.events],
            (x.tip_inside, x._in_material, x._ever_inside, len(x._openings), sorted(x._visited_tets)),
            x.tip_position.tobytes(),
            None if x._previous_tip is None else (x._previous_tip.tobytes(), x._previous_q.tobytes()),
            tuple(
                getattr(x, name).numpy().tobytes()
                for name in ("_track_nodes", "_track_bary", "_track_lengths", "_track_engaged", "_all_opening_faces")
            ),
        )

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

    def test_requested_engaged_length_is_read_by_accept_and_follows_the_engagement(self):
        interaction, state, forces = self.build()
        seen = []
        with patch.object(
            interaction, "_read_engaged_track_length", wraps=interaction._read_engaged_track_length
        ) as read:
            for iteration, z in enumerate(self.path()):
                reads = read.call_count
                interaction.request_engaged_track_length()
                self.advance(interaction, state, forces, iteration, z)
                # The step's one read is the one accept made; the query is answered from it.
                reads += interaction.track_count > 0
                self.assertEqual(read.call_count, reads)
                seen.append(interaction.engaged_track_length)
                self.assertEqual(read.call_count, reads)
                self.assertEqual(seen[-1], self.device_engaged_length(interaction))
        # The track grows, is released sample by sample, is free after the exit and is taken up
        # again: a length kept from an earlier step differs from the device at every change.
        changes = np.sign(np.diff(seen))
        self.assertGreaterEqual(int(np.sum(changes > 0)), 5)
        self.assertGreaterEqual(int(np.sum(changes < 0)), 3)
        self.assertEqual(seen[25], 0.0)
        self.assertGreater(interaction.track_count, 0)
        self.assertEqual((interaction.puncture_count, interaction.entry_count, interaction.exit_count), (1, 2, 1))

    def test_accept_without_a_request_reads_nothing_and_keeps_no_earlier_length(self):
        interaction, state, forces = self.inserted()
        kept = interaction.engaged_track_length
        lengths = []
        with patch.object(
            interaction, "_read_engaged_track_length", wraps=interaction._read_engaged_track_length
        ) as read:
            for iteration, z in enumerate(self.path()[10:19], start=10):
                self.advance(interaction, state, forces, iteration, z)
                self.assertEqual(read.call_count, len(lengths))
                lengths.append(interaction.engaged_track_length)
                self.assertEqual(read.call_count, len(lengths))
                self.assertEqual(lengths[-1], self.device_engaged_length(interaction))
        # The withdrawal released the whole track; the length of the last request is not served.
        self.assertLess(lengths[1], kept)
        self.assertEqual(lengths[-1], 0.0)
        self.assertGreater(interaction.track_count, 0)

    def test_kept_engaged_length_ends_at_every_other_write_of_the_engagement(self):
        withdrawn = [[0.002, 0.002, -0.0012, 0, 0, 0, 1]]
        with self.subTest("explicit force evaluation updates the engagement"):
            interaction, state, forces = self.inserted()
            state.body_q.assign(withdrawn)
            state.body_f.zero_()
            forces.zero_()
            interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005)
            self.assertEqual(self.device_engaged_length(interaction), 0.0)
            self.assertEqual(interaction.engaged_track_length, 0.0)
        with self.subTest("explicit trial updates the engagement and restores it"):
            interaction, state, forces = self.inserted()
            kept = interaction.engaged_track_length
            state.body_q.assign([[0.002, 0.002, 0.0009, 0, 0, 0, 1]])
            state.body_f.zero_()
            forces.zero_()
            interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True, implicit=False)
            self.assertEqual(interaction.engaged_track_length, self.device_engaged_length(interaction))
            self.assertEqual(interaction.engaged_track_length, kept)
        with self.subTest("acceptance stops after its update of the engagement"):
            interaction, state, forces = self.inserted()
            kept = interaction.engaged_track_length
            self.advance(interaction, state, forces, 10, self.path()[10])
            interaction.request_engaged_track_length()
            self.advance(interaction, state, forces, 11, self.path()[11], accept=False)
            with (
                patch.object(interaction, "_locate", side_effect=RuntimeError("stopped")),
                self.assertRaisesRegex(RuntimeError, "stopped"),
            ):
                interaction.accept(state, 0)
            self.assertLess(self.device_engaged_length(interaction), kept)
            self.assertEqual(interaction.engaged_track_length, self.device_engaged_length(interaction))
        for name in ("_track_engaged", "_track_lengths"):
            with self.subTest("replaced array", array=name):
                interaction, state, forces = self.inserted()
                setattr(interaction, name, wp.zeros_like(getattr(interaction, name)))
                self.assertEqual(interaction.engaged_track_length, 0.0)

    def test_accept_takes_the_checked_crossings_of_its_own_step_and_commits_the_same_history(self):
        reused, repeated = self.build(), self.build()
        crossed = 0
        for iteration, z in enumerate(self.path()):
            for repeat, (interaction, state, forces) in zip((False, True), (reused, repeated), strict=True):
                calls, crossings = [], interaction._resolution_crossings

                def recorded(*arrays, calls=calls, crossings=crossings):
                    calls.append(crossings(*arrays))
                    return calls[-1]

                with patch.object(interaction, "_resolution_crossings", side_effect=recorded):
                    self.advance(interaction, state, forces, iteration, z, accept=False)
                    if repeat:
                        # Without the kept result the acceptance makes the swept tip test itself.
                        interaction._checked_crossings = None
                    interaction.accept(state, 0)
                self.assertNotEqual(interaction.status, "failed", interaction.reason)
                self.assertEqual(len(calls), 2 if repeat else 1)
                # The test of the acceptance is the test of the check: the same crossings.
                self.assertEqual(calls[0], calls[-1])
                crossed += bool(repeat and calls[0][0])
            self.assertEqual(self.committed(reused[0]), self.committed(repeated[0]))
        # Entry, exit and second entry each cross the surface once.
        self.assertEqual(crossed, 3)
        self.assertEqual((reused[0].puncture_count, reused[0].entry_count, reused[0].exit_count), (1, 2, 1))

    def test_accept_repeats_the_swept_tip_test_when_one_of_its_four_arrays_differs(self):
        def moved_node(array):
            nodes = array.numpy()
            nodes[3, 2] += 2e-6
            array.assign(nodes)

        def moved_tip(array):
            pose = array.numpy()
            pose[0, 2] -= 2e-6
            array.assign(pose)

        changes = {
            "none": lambda start, state: None,
            "tip": lambda start, state: moved_tip(state.body_q),
            "nodes": lambda start, state: moved_node(state.particle_q),
            "start tip": lambda start, state: moved_tip(start.body_q),
            "start nodes": lambda start, state: moved_node(start.particle_q),
        }
        for name, change in changes.items():
            with self.subTest(changed=name):
                committed = []
                for check in (True, False):
                    interaction, state, forces = self.inserted()
                    start = SimpleNamespace(particle_q=wp.clone(state.particle_q), body_q=wp.clone(state.body_q))
                    state.body_q.assign([[0.002, 0.002, self.path()[10], 0, 0, 0, 1]])
                    if check:
                        interaction.check_step_resolution(start, state, 0)
                    change(start, state)
                    state.body_f.zero_()
                    forces.zero_()
                    interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True, start_state=start)
                    with patch.object(
                        interaction, "_resolution_crossings", wraps=interaction._resolution_crossings
                    ) as crossings:
                        interaction.accept(state, 0)
                    self.assertNotEqual(interaction.status, "failed", interaction.reason)
                    # Only the step that was checked, with all four arrays as they were, is not tested again.
                    self.assertEqual(crossings.call_count, 0 if check and name == "none" else 1)
                    committed.append(self.committed(interaction))
                self.assertEqual(committed[0], committed[1])


if __name__ == "__main__":
    unittest.main()
