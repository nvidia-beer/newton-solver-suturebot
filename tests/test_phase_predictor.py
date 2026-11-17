# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot._src.solvers.coupled.phase_predictor import PhaseLoadPredictor


class PhasePredictorTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.device.__enter__()
        self.current = wp.zeros(6, dtype=wp.float64)
        self.trial = wp.zeros_like(self.current)
        self.engagement = wp.array([1, 0, 1], dtype=int)
        self.predictor = PhaseLoadPredictor(self.current)
        self.command = np.array([0, 0, 0, 0, 0, 0, 1, 0.0])
        self.vector = np.array([1, -1, 0.2, -0.2, 0.5, -0.5])

    def tearDown(self):
        self.device.__exit__(None, None, None)

    def begin_frame(self, frame, *, reset=0, step_dt=0.01):
        self.predictor.begin_frame(
            frame_dt=0.04,
            step_dt=step_dt,
            elapsed=frame * 0.04,
            command=self.command,
            reset_token=reset,
        )

    def accept(self, frame, phase, *, signature="same"):
        self.predictor.begin(self.current, self.trial, 0.01, signature, self.engagement)
        guess = self.trial.numpy().copy()
        accepted = wp.array((1 + 0.1 * frame + 0.4 * phase) * self.vector, dtype=wp.float64)
        self.predictor.commit(accepted)
        wp.copy(self.current, accepted)
        return guess

    def completed_frame(self, frame):
        self.begin_frame(frame)
        guesses = [self.accept(frame, phase) for phase in range(4)]
        self.predictor.end_frame((frame + 1) * 0.04, True)
        return guesses

    def test_same_phase_history_tracks_four_phase_oscillation(self):
        self.completed_frame(0)
        repeated = self.completed_frame(1)
        for phase, guess in enumerate(repeated):
            np.testing.assert_allclose(guess, (1 + 0.4 * phase) * self.vector)
        predicted = self.completed_frame(2)
        for phase, guess in enumerate(predicted):
            np.testing.assert_allclose(guess, (1.2 + 0.4 * phase) * self.vector, atol=1e-14)
        np.testing.assert_array_equal(self.predictor.counts.numpy(), [0, 4, 4, 0])

    def test_rollback_and_partial_frame_never_enter_history(self):
        self.completed_frame(0)
        self.completed_frame(1)
        self.begin_frame(2)
        before = self.current.numpy().copy()
        self.predictor.begin(self.current, self.trial, 0.01, "same", self.engagement)
        self.predictor.discard()
        np.testing.assert_array_equal(self.current.numpy(), before)
        self.assertEqual(self.predictor.depth, 0)
        self.assertFalse(self.predictor.open)
        self.predictor.end_frame(0.08, False)
        self.assertEqual(self.predictor.depth, 0)

    def test_topology_event_and_force_fallback_clear_completed_history(self):
        self.completed_frame(0)
        self.completed_frame(1)
        self.begin_frame(2)
        self.accept(2, 0)
        before = self.current.numpy().copy()
        guess = self.accept(2, 1, signature="new-track")
        np.testing.assert_array_equal(guess, before)
        self.assertEqual(self.predictor.depth, 0)
        self.predictor.begin(self.current, self.trial, 0.01, "new-track", self.engagement)
        self.predictor.cancel()
        self.predictor.commit(self.current)
        self.predictor.end_frame(0.11, False)
        self.assertFalse(self.predictor.open)
        self.assertEqual(self.predictor.depth, 0)

    def test_discontinuous_command_reset_and_changed_phase_count(self):
        for change in ("command", "reset", "dt"):
            with self.subTest(change=change):
                self.predictor = PhaseLoadPredictor(self.current)
                self.command[:] = [0, 0, 0, 0, 0, 0, 1, 0]
                self.completed_frame(0)
                self.completed_frame(1)
                if change == "command":
                    self.command[0] = 0.01
                self.begin_frame(
                    2,
                    reset=int(change == "reset"),
                    step_dt=0.005 if change == "dt" else 0.01,
                )
                self.assertEqual(self.predictor.depth, 0)
                self.predictor.end_frame(0.08, False)

    def test_nonfinite_or_changed_engagement_retains_current_guess(self):
        for change in ("nonfinite", "engagement"):
            with self.subTest(change=change):
                self.predictor = PhaseLoadPredictor(self.current)
                self.engagement.assign(np.array([1, 0, 1], dtype=np.int32))
                self.completed_frame(0)
                self.completed_frame(1)
                self.begin_frame(2)
                if change == "nonfinite":
                    self.predictor.frames[0].loads[0].fill_(float("nan"))
                else:
                    self.engagement.assign(np.array([0, 0, 1], dtype=np.int32))
                before = self.current.numpy().copy()
                self.predictor.begin(self.current, self.trial, 0.01, "same", self.engagement)
                np.testing.assert_array_equal(self.trial.numpy(), before)
                self.predictor.discard()
                self.predictor.end_frame(0.08, False)

    def test_engagement_in_a_reserve_is_copied_and_compared_over_the_rows_in_use(self):
        # Storage of a fixed size, as the interaction's engagement at its track reserve. The rows in
        # use are copied and compared; the rest of the storage has no effect, whatever it holds.
        reserve = wp.array([1, 0, 1, 0, 0, 0], dtype=int)
        launch, copy = wp.launch, wp.copy

        def accept(frame, phase, count):
            sizes, copied = [], []

            def record(kernel, dim, *arguments, **options):
                if kernel.key == "_engagement":
                    sizes.append(dim)
                return launch(kernel, dim, *arguments, **options)

            def copy_rows(destination, source, *arguments, **options):
                if source is reserve:
                    copied.append(options.get("count"))
                return copy(destination, source, *arguments, **options)

            with patch.object(wp, "launch", side_effect=record), patch.object(wp, "copy", side_effect=copy_rows):
                self.predictor.begin(self.current, self.trial, 0.01, "same", reserve, count)
            guess = self.trial.numpy().copy()
            accepted = wp.array((1 + 0.1 * frame + 0.4 * phase) * self.vector, dtype=wp.float64)
            self.predictor.commit(accepted)
            wp.copy(self.current, accepted)
            return guess, sizes, copied

        def frame(index, count, edit=None):
            self.begin_frame(index)
            results = []
            for phase in range(4):
                if edit is not None:
                    edit(phase)
                results.append(accept(index, phase, count))
            self.predictor.end_frame((index + 1) * 0.04, True)
            return results

        for _, sizes, copied in frame(0, 3):
            self.assertEqual((sizes, copied), ([], [3]))
        snapshots = [snapshot.ptr for snapshot in self.predictor.frames[0].engagement]
        self.assertEqual([snapshot.shape for snapshot in self.predictor.frames[0].engagement], [(6,)] * 4)
        for phase, (guess, sizes, copied) in enumerate(frame(1, 3)):
            np.testing.assert_allclose(guess, (1 + 0.4 * phase) * self.vector)
            self.assertEqual((sizes, copied), ([3], [3]))

        # Rows beyond the count change before every phase: the two-frame guess is still made.
        def beyond(phase):
            reserve.assign(np.array([1, 0, 1, phase + 1, 5, -2], dtype=np.int32))

        for phase, (guess, sizes, copied) in enumerate(frame(2, 3, beyond)):
            np.testing.assert_allclose(guess, (1.2 + 0.4 * phase) * self.vector, atol=1e-14)
            self.assertEqual((sizes, copied), ([3], [3]))
        np.testing.assert_array_equal(self.predictor.counts.numpy(), [0, 4, 4, 0])

        # A row in use changes before the third phase: that phase starts from the current load.
        def within(phase):
            reserve.assign(np.array([1, 0, int(phase != 2), 0, 0, 0], dtype=np.int32))

        before = []
        for phase, (guess, sizes, _) in enumerate(
            frame(3, 3, lambda phase: (within(phase), before.append(self.current.numpy().copy())))
        ):
            self.assertEqual(sizes, [3])
            if phase == 2:
                np.testing.assert_array_equal(guess, before[phase])
            else:
                np.testing.assert_allclose(guess, (1.3 + 0.4 * phase) * self.vector, atol=1e-14)
        np.testing.assert_array_equal(self.predictor.counts.numpy(), [1, 4, 7, 0])
        # The snapshots of a frame are made once and reused, also when another count of rows is in use.
        reserve.assign(np.array([1, 0, 1, 1, 0, 0], dtype=np.int32))
        self.predictor.invalidate()
        results = [frame(index, 4) for index in (4, 5, 6)]
        current = sorted(snapshot.ptr for frame_ in self.predictor.frames for snapshot in frame_.engagement)
        self.assertEqual(len(set(current)), 12)
        self.assertTrue(set(snapshots) <= set(current))
        self.assertEqual([entry[1:] for entry in results[2]], [([4], [4])] * 4)
        for phase, (guess, _, _) in enumerate(results[2]):
            np.testing.assert_allclose(guess, (1.6 + 0.4 * phase) * self.vector, atol=1e-14)
        # Another count of rows in use is another engagement: nothing recorded with four rows is used with
        # three, although signature and storage shape are the same.
        attempts = self.predictor.attempts
        before = []
        for phase, (guess, sizes, copied) in enumerate(
            frame(7, 3, lambda phase: before.append(self.current.numpy().copy()))
        ):
            np.testing.assert_array_equal(guess, before[phase])
            self.assertEqual((sizes, copied), ([], [3]))
        self.assertEqual(self.predictor.attempts, attempts)
        # No row in use: nothing is copied (a count of zero would mean the whole array to wp.copy) and
        # no engagement is compared, while the loads are still predicted.
        self.predictor.invalidate()
        results = [frame(index, 0) for index in (8, 9, 10)]
        for entries in results:
            self.assertEqual([entry[1:] for entry in entries], [([], [])] * 4)
        for phase, (guess, _, _) in enumerate(results[2]):
            np.testing.assert_allclose(guess, (2.0 + 0.4 * phase) * self.vector, atol=1e-14)
        self.begin_frame(11)
        for count in (-1, 7):
            with self.assertRaisesRegex(ValueError, "engagement count"):
                self.predictor.begin(self.current, self.trial, 0.01, "same", reserve, count)
        self.assertFalse(self.predictor.open)
        self.predictor.end_frame(0.48, False)

    def test_real_rejected_coupler_transaction_closes_and_discards_predictor(self):
        from test_needle_insertion_example import _example, _prepare
        from test_needle_tissue_coupler import _native_state

        from newton_solver_suturebot import NeedleTissueCoupler

        with _example() as example:
            _prepare(example)
            example.start()
            pose = example.plan.pose(0.0).copy()
            pose[2] += 0.6 * example.geometry.thickness
            example._initialize_pose(pose)
            example.state_0.body_f.assign(np.array([[0, 0, -0.0002, 0, 0, 0]], np.float32))
            native = _native_state(example.rigid_solver)
            coupler = NeedleTissueCoupler(
                example.model,
                example.solver,
                example.rigid_solver,
                example.interaction,
                example.needle_body,
                example.drive.com,
                max_iterations=1,
                absolute_tolerance=1e-14,
                relative_tolerance=1e-12,
            )
            predictor = PhaseLoadPredictor(coupler._load)
            coupler._phase_load_predictor = predictor
            predictor.begin_frame(
                frame_dt=example.sim_dt,
                step_dt=example.sim_dt,
                elapsed=0,
                command=self.command,
                reset_token=0,
            )
            with self.assertRaises(RuntimeError):
                coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
            np.testing.assert_array_equal(_native_state(example.rigid_solver), native)
            self.assertFalse(predictor.open)
            self.assertEqual(predictor.depth, 0)
            self.assertEqual(predictor.commits, 0)
            predictor.end_frame(0, False)

    def test_restart_of_an_attempt_that_took_no_guess_leaves_a_predictor_that_exists_as_it_is(self):
        # The trials of an attempt are restarted where the exact fallback takes over, which cancels
        # the guess of the predictor. An attempt of start-up takes no predictor, and its restart
        # (NeedleTissueCoupler.capture_trial_graph) says so: a predictor that exists already, as
        # in a second start-up call, keeps its history. The restart of an attempt of a step
        # cancels as before.
        from test_needle_insertion_example import _example, _prepare

        from newton_solver_suturebot import NeedleTissueCoupler

        with _example() as example:
            _prepare(example)
            example.start()
            coupler = NeedleTissueCoupler(
                example.model,
                example.solver,
                example.rigid_solver,
                example.interaction,
                example.needle_body,
                example.drive.com,
            )
            predictor = coupler._phase_load_predictor = PhaseLoadPredictor(coupler._load)
            # As after two completed frames.
            predictor.depth = 2
            state = example.state_0
            attempt = coupler._open_attempt(state, example.control, example.sim_dt, None)
            self.assertIs(attempt.predictor, predictor)
            # As the start-up call opens it.
            attempt.predictor = None
            try:
                coupler._begin_attempt(attempt)
                coupler._restart_trials(state, attempt.tunnel_alm, cancel_guess=attempt.predictor is not None)
                self.assertEqual((predictor.depth, predictor.invalidations), (2, 0))
                coupler._restart_trials(state, attempt.tunnel_alm)
                self.assertEqual((predictor.depth, predictor.invalidations), (0, 1))
            finally:
                coupler._close_attempt(attempt, state, False)
            # The close of an attempt without a predictor discards none.
            self.assertEqual((predictor.invalidations, predictor.rejections, predictor.open), (1, 0, False))


class IncrementPredictorTests(unittest.TestCase):
    """History semantics of the frame-periodic increment guess used by the GPU backend."""

    PATTERN = (0.0, 0.4, 0.1, 0.3)

    def setUp(self):
        self.device = wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.device.__enter__()
        self.vector = np.array([1, -1, 0.2, -0.2, 0.5, -0.5])
        self.command = np.array([0, 0, 0, 0, 0, 0, 1, 0.0])
        self.restart()

    def tearDown(self):
        self.device.__exit__(None, None, None)

    def restart(self):
        self.offset, self.reset_token = 0.0, 0
        # The load accepted just before frame 0, so that every recorded change is one of the model.
        self.current = wp.array(self.load(-1, 3), dtype=wp.float64)
        self.trial = wp.zeros_like(self.current)
        self.engagement = wp.array([1, 0, 1], dtype=int)
        self.predictor = PhaseLoadPredictor(self.current, increments=True)

    def load(self, frame, phase):
        """A frame-periodic pattern on a linear drift, plus any persistent jump."""
        return (1 + 0.05 * (4 * frame + phase) + self.PATTERN[phase] + self.offset) * self.vector

    def begin_frame(self, frame):
        self.predictor.begin_frame(
            frame_dt=0.04, step_dt=0.01, elapsed=frame * 0.04, command=self.command, reset_token=self.reset_token
        )

    def accept(self, frame, phase, *, signature="same"):
        """Return (guess, start) of one accepted transaction."""
        start = self.current.numpy().copy()
        self.predictor.begin(self.current, self.trial, 0.01, signature, self.engagement)
        guess = self.trial.numpy().copy()
        accepted = wp.array(self.load(frame, phase), dtype=wp.float64)
        self.predictor.commit(accepted)
        wp.copy(self.current, accepted)
        return guess, start

    def completed_frame(self, frame, *, signature="same"):
        self.begin_frame(frame)
        result = [self.accept(frame, phase, signature=signature) for phase in range(4)]
        self.predictor.end_frame((frame + 1) * 0.04, True)
        return result

    def assert_plain(self, results):
        for guess, start in results:
            np.testing.assert_array_equal(guess, start)

    def assert_predicted(self, frame, results, first=0):
        for phase, (guess, _) in enumerate(results, first):
            np.testing.assert_allclose(guess, self.load(frame, phase), rtol=0, atol=1e-14)

    def test_guess_adds_the_agreed_change_of_the_phase_to_the_last_accepted_load(self):
        # Two completed frames are needed before the changes of a phase can agree.
        self.assert_plain(self.completed_frame(0))
        self.assert_plain(self.completed_frame(1))
        self.assertEqual(self.predictor.attempts, 0)
        self.assert_predicted(2, self.completed_frame(2))
        self.assert_predicted(3, self.completed_frame(3))
        np.testing.assert_array_equal(self.predictor.counts.numpy(), [0, 0, 0, 8])
        # The guess is a combination of accepted, balanced loads.
        np.testing.assert_allclose(self.trial.numpy()[0::2] + self.trial.numpy()[1::2], 0, atol=1e-14)

    def test_storage_reserved_ahead_of_the_first_frame_is_the_storage_of_the_frames_and_changes_no_guess(self):
        for increments in (True, False):
            with self.subTest(increments=increments):
                # Reference: the frames and the engagement snapshots are made where they are first needed.
                self.restart()
                self.predictor = PhaseLoadPredictor(self.current, increments=increments)
                reference = [self.completed_frame(frame) for frame in range(5)]
                counts = self.predictor.counts.numpy().copy()
                self.assertGreater(self.predictor.attempts, 0)
                self.restart()
                predictor = self.predictor = PhaseLoadPredictor(self.current, increments=increments)
                # A frame that is no whole number of steps has no phases: nothing is made for it.
                self.assertFalse(predictor.reserve(frame_dt=0.04, step_dt=0.03, engagement=self.engagement))
                self.assertIsNone(predictor.frames)
                # Without an engagement array: the loads (and increments) of three frames.
                self.assertTrue(predictor.reserve(frame_dt=0.04, step_dt=0.01))
                self.assertEqual((len(predictor.frames), predictor.phase_count), (3, 4))
                loads = [array.ptr for frame in predictor.frames for array in (*frame.loads, *(frame.increments or ()))]
                self.assertEqual(len(set(loads)), 3 * 4 * (2 if increments else 1))
                self.assertEqual([frame.engagement for frame in predictor.frames], [[None] * 4] * 3)
                # With it: one snapshot per frame and phase, with the storage shape of that array.
                self.assertTrue(predictor.reserve(frame_dt=0.04, step_dt=0.01, engagement=wp.zeros(5, dtype=int)))
                self.assertEqual({a.shape for frame in predictor.frames for a in frame.engagement}, {(5,)})
                self.assertTrue(predictor.reserve(frame_dt=0.04, step_dt=0.01, engagement=self.engagement))

                def storage(predictor=predictor):
                    frames = predictor.frames
                    kept = [a.ptr for frame in frames for a in (*frame.loads, *(frame.increments or ()))]
                    return kept, sorted(a.ptr for frame in frames for a in frame.engagement)

                made = storage()
                self.assertEqual(made[0], loads)
                self.assertEqual(len(set(made[1])), 12)
                for frame in predictor.frames:
                    for snapshot in frame.engagement:
                        self.assertEqual(
                            (snapshot.shape, snapshot.dtype), (self.engagement.shape, self.engagement.dtype)
                        )
                    # No history: a snapshot is compared only where its phase has a signature.
                    self.assertEqual((frame.signatures, frame.engaged_rows), ([None] * 4, [0] * 4))
                self.assertEqual((predictor.depth, predictor.frame_open, predictor.open), (0, False, False))
                self.assertEqual((predictor.attempts, predictor.commits, predictor.invalidations), (0, 0, 0))
                # A second call keeps what exists.
                self.assertTrue(predictor.reserve(frame_dt=0.04, step_dt=0.01, engagement=self.engagement))
                self.assertEqual(storage(), made)
                # The frames allocate nothing, and every guess has the bits of the reference.
                with patch.object(wp, "empty_like", side_effect=AssertionError("allocated in a frame")):
                    reserved = [self.completed_frame(frame) for frame in range(5)]
                for expected, actual in zip(reference, reserved, strict=True):
                    for (guess, start), (same_guess, same_start) in zip(expected, actual, strict=True):
                        np.testing.assert_array_equal(same_guess, guess)
                        np.testing.assert_array_equal(same_start, start)
                np.testing.assert_array_equal(predictor.counts.numpy(), counts)
                self.assertEqual(sorted(storage()[0]), sorted(made[0]))
                self.assertEqual(storage()[1], made[1])
                # Another phase count makes the frames of that count, as a frame does.
                self.assertTrue(predictor.reserve(frame_dt=0.04, step_dt=0.02, engagement=self.engagement))
                self.assertEqual(
                    (predictor.phase_count, [len(frame.engagement) for frame in predictor.frames]), (2, [2] * 3)
                )
                for value in (0.0, -0.01, float("nan")):
                    with self.assertRaises(ValueError):
                        predictor.reserve(frame_dt=0.04, step_dt=value)

    def test_single_jump_is_never_repeated_and_disagreeing_changes_retain_the_accepted_load(self):
        self.completed_frame(0)
        self.completed_frame(1)
        self.begin_frame(2)
        results = [self.accept(2, 0)]
        self.offset = 3.0  # an event shifts every load from phase 1 of frame 2 on
        results += [self.accept(2, phase) for phase in range(1, 4)]
        self.predictor.end_frame(0.12, True)
        # The jump itself was not predicted; the phases after it continue with their agreed changes.
        np.testing.assert_allclose(results[1][0], results[1][1] + 0.45 * self.vector, atol=1e-14)
        self.assert_predicted(2, results[2:], first=2)
        for frame in (3, 4):
            # The jump is the last change of phase 1, then the older one: neither frame repeats it.
            results = self.completed_frame(frame)
            np.testing.assert_array_equal(results[1][0], results[1][1])
            self.assert_predicted(frame, results[:1])
            self.assert_predicted(frame, results[2:], first=2)
        self.assert_predicted(5, self.completed_frame(5))
        np.testing.assert_array_equal(self.predictor.counts.numpy(), [2, 0, 0, 14])

    def test_scene_reset_rejection_fallback_and_invalidation_discard_the_history(self):
        for change in ("reset", "rejection", "fallback", "invalidation"):
            with self.subTest(change=change):
                self.restart()
                self.completed_frame(0)
                self.completed_frame(1)
                self.assert_predicted(2, self.completed_frame(2))
                if change == "invalidation":
                    # What NeedleTissueCoupler.invalidate_fem_graph does between completed steps.
                    self.predictor.invalidate()
                self.reset_token = int(change == "reset")
                self.begin_frame(3)
                if change == "rejection":
                    # The coupler discards a rejected step; the scene then retries the phase.
                    before = self.current.numpy().copy()
                    self.predictor.begin(self.current, self.trial, 0.01, "same", self.engagement)
                    self.predictor.discard()
                    np.testing.assert_array_equal(self.current.numpy(), before)
                elif change == "fallback":
                    # A host fallback cancels inside a transaction that is still accepted.
                    self.predictor.begin(self.current, self.trial, 0.01, "same", self.engagement)
                    self.predictor.cancel()
                    accepted = wp.array(self.load(3, 0), dtype=wp.float64)
                    self.predictor.commit(accepted)
                    wp.copy(self.current, accepted)
                self.assertEqual(self.predictor.depth, 0)
                self.assert_plain([self.accept(3, phase) for phase in range(int(change == "fallback"), 4)])
                self.predictor.end_frame(0.16, True)
                # Nothing recorded before the change is used again: two new complete frames come first.
                frame = 4
                if change in ("rejection", "fallback"):
                    # The interrupted frame itself is not history.
                    self.assertEqual(self.predictor.depth, 0)
                    self.assert_plain(self.completed_frame(frame))
                    frame += 1
                self.assert_plain(self.completed_frame(frame))
                self.assertEqual(self.predictor.depth, 2)
                self.assert_predicted(frame + 1, self.completed_frame(frame + 1))

    def test_track_growth_changed_engagement_and_changed_signature_never_reuse_history(self):
        self.completed_frame(0)
        self.completed_frame(1)
        self.assert_predicted(2, self.completed_frame(2))
        # A changed signature (track growth, puncture or material state, parameters, storage) is another force map.
        self.begin_frame(3)
        self.assert_predicted(3, [self.accept(3, 0)])
        attempts = self.predictor.attempts
        guess, start = self.accept(3, 1, signature="grown track")
        np.testing.assert_array_equal(guess, start)
        self.assertEqual(self.predictor.depth, 0)
        self.assert_plain([self.accept(3, phase, signature="grown track") for phase in (2, 3)])
        self.predictor.end_frame(0.16, True)
        self.assertEqual(self.predictor.depth, 0)
        # No change recorded before the growth is offered after it: two complete frames of the new map come first.
        self.assert_plain(self.completed_frame(4, signature="grown track"))
        self.assert_plain(self.completed_frame(5, signature="grown track"))
        self.assertEqual((self.predictor.depth, self.predictor.attempts), (2, attempts))
        self.assert_predicted(6, self.completed_frame(6, signature="grown track"))
        # Anchor storage of another size has no engagement to compare, whatever the signature says.
        self.engagement = wp.array([1, 0, 1, 1], dtype=int)
        attempts = self.predictor.attempts
        self.assert_plain(self.completed_frame(7, signature="grown track"))
        self.assert_plain(self.completed_frame(8, signature="grown track"))
        self.assertEqual((self.predictor.depth, self.predictor.attempts), (2, attempts))
        self.assert_predicted(9, self.completed_frame(9, signature="grown track"))
        # An existing anchor that changed its engagement in either compared frame declines the guess on the device.
        self.engagement = wp.array([0, 0, 1, 1], dtype=int)
        declined = self.predictor.counts.numpy()[0]
        self.assert_plain(self.completed_frame(10, signature="grown track"))
        self.assert_plain(self.completed_frame(11, signature="grown track"))
        self.assertEqual(self.predictor.counts.numpy()[0], declined + 8)
        self.assert_predicted(12, self.completed_frame(12, signature="grown track"))

    def test_rejection_counter_follows_only_discarded_transactions_that_started_from_an_offered_guess(self):
        self.completed_frame(0)
        self.completed_frame(1)
        self.begin_frame(2)
        # An accepted guess is no rejection.
        self.assert_predicted(2, [self.accept(2, 0)])
        self.assertEqual((self.predictor.attempted, self.predictor.rejections), (False, 0))
        # A rejected transaction that started from the offered guess is counted and leaves no history.
        self.predictor.begin(self.current, self.trial, 0.01, "same", self.engagement)
        np.testing.assert_allclose(self.trial.numpy(), self.load(2, 1), rtol=0, atol=1e-14)
        self.assertTrue(self.predictor.attempted)
        self.predictor.discard()
        self.assertEqual((self.predictor.attempted, self.predictor.rejections, self.predictor.depth), (False, 1, 0))
        # The caller's next attempt starts from the accepted load; its own rejection is not a guess rejection.
        before = self.current.numpy().copy()
        self.predictor.begin(self.current, self.trial, 0.01, "same", self.engagement)
        np.testing.assert_array_equal(self.trial.numpy(), before)
        self.assertFalse(self.predictor.attempted)
        self.predictor.discard()
        self.predictor.end_frame(0.08, False)
        self.assertEqual(self.predictor.rejections, 1)
        # An exact fallback restarts its trials from the accepted load inside the transaction.
        self.restart()
        self.completed_frame(0)
        self.completed_frame(1)
        self.begin_frame(2)
        self.predictor.begin(self.current, self.trial, 0.01, "same", self.engagement)
        self.assertTrue(self.predictor.attempted)
        self.predictor.cancel()
        self.predictor.discard()
        self.predictor.end_frame(0.08, False)
        self.assertEqual(self.predictor.rejections, 0)
        # The same-phase guess of the default mode is not counted.
        self.current.assign(self.load(-1, 3))
        self.predictor = PhaseLoadPredictor(self.current)
        self.completed_frame(0)
        self.completed_frame(1)
        self.begin_frame(2)
        self.predictor.begin(self.current, self.trial, 0.01, "same", self.engagement)
        self.assertEqual(self.predictor.attempts, 5)
        self.predictor.discard()
        self.predictor.end_frame(0.08, False)
        self.assertEqual(self.predictor.rejections, 0)

    def test_modified_start_load_or_nonfinite_history_retains_the_current_load(self):
        for change in ("load", "nonfinite"):
            with self.subTest(change=change):
                self.restart()
                self.completed_frame(0)
                self.completed_frame(1)
                self.begin_frame(2)
                if change == "load":
                    # The start is no longer the load the recorded changes continue.
                    self.current.assign(self.current.numpy() + 1e-3)
                else:
                    self.predictor.frames[1].increments[0].fill_(float("nan"))
                guess, start = self.accept(2, 0)
                np.testing.assert_array_equal(guess, start)
                np.testing.assert_array_equal(self.predictor.counts.numpy(), [1, 0, 0, 0])
                self.predictor.end_frame(0.08, False)

    def test_increment_signature_changes_with_every_track_growth_and_no_history_crosses_a_real_one(self):
        from unittest.mock import PropertyMock, patch

        from test_grasp_scene import _scene

        from newton_solver_suturebot._src.solvers.coupled.phase_predictor import coupler_signature

        with _scene() as (scene, _):
            coupler, x = scene.coupler, scene.coupler.interaction
            x._previous_tip = np.zeros(3)
            with patch.object(type(x), "track_count", new_callable=PropertyMock, return_value=2) as track_count:
                # An engaged track whose tip left the material is supported by the increment guess only.
                x.punctured = x._in_material = False
                self.assertIsNone(coupler_signature(coupler))
                outside = coupler_signature(coupler, increments=True)
                self.assertIsNotNone(outside)
                x.punctured = x._in_material = True
                inside = coupler_signature(coupler, increments=True)
                self.assertNotEqual(inside, outside)
                # Inside the material both guesses compare the same force map.
                self.assertEqual(inside, coupler_signature(coupler))

                def visited_tet():
                    x._visited_tets.add(0)
                    return lambda: x._visited_tets.discard(0)

                def track_edge():
                    x._track_edges.add((0, 1))
                    return lambda: x._track_edges.discard((0, 1))

                def anchor_count():
                    track_count.return_value = 3
                    return lambda: setattr(track_count, "return_value", 2)

                def opening():
                    x._openings.append((np.zeros(3, dtype=np.int32), np.zeros(3)))
                    return x._openings.pop

                def current_opening():
                    x._opening = (np.zeros(3, dtype=np.int32), np.zeros(3))
                    return lambda: setattr(x, "_opening", None)

                def storage(name):
                    def replace():
                        original = getattr(x, name)
                        setattr(x, name, wp.zeros(4, dtype=original.dtype, device=original.device))
                        return lambda: setattr(x, name, original)

                    return replace

                growth = {
                    "visited tetrahedron": visited_tet,
                    "track edge": track_edge,
                    "anchor count": anchor_count,
                    "opening": opening,
                    "current opening": current_opening,
                    **{
                        name: storage(name)
                        for name in ("_track_nodes", "_track_bary", "_track_lengths", "_track_engaged")
                    },
                }
                self.assertIsNone(x._opening)
                for kind, grow in growth.items():
                    with self.subTest(growth=kind):
                        # Every kind of track growth is another force map for the increment guess too.
                        restore = grow()
                        self.assertNotEqual(coupler_signature(coupler, increments=True), inside)
                        restore()
                        self.assertEqual(coupler_signature(coupler, increments=True), inside)

                # Changes recorded with the real signature are not used after the track grew.
                def step(frame, phase):
                    return self.accept(frame, phase, signature=coupler_signature(coupler, increments=True))

                def frame_of_steps(frame, first=0):
                    if first == 0:
                        self.begin_frame(frame)
                    result = [step(frame, phase) for phase in range(first, 4)]
                    self.predictor.end_frame((frame + 1) * 0.04, True)
                    return result

                self.assert_plain(frame_of_steps(0))
                self.assert_plain(frame_of_steps(1))
                self.begin_frame(2)
                self.assert_predicted(2, [step(2, 0)])
                attempts = self.predictor.attempts
                x._visited_tets.add(0)
                x._track_edges.add((0, 1))
                track_count.return_value = 3
                self.assert_plain(frame_of_steps(2, first=1))
                self.assertEqual(self.predictor.depth, 0)
                self.assert_plain(frame_of_steps(3))
                self.assert_plain(frame_of_steps(4))
                self.assertEqual((self.predictor.depth, self.predictor.attempts), (2, attempts))
                self.assert_predicted(5, frame_of_steps(5))
                x._visited_tets.discard(0)
                x._track_edges.discard((0, 1))
                track_count.return_value = 2

                # Parameters, other storage and surface events belong to the force map as well.
                friction = x.friction
                x.friction = friction + 0.125
                self.assertNotEqual(coupler_signature(coupler, increments=True), inside)
                x.friction = friction
                x._samples = wp.clone(x._samples)
                self.assertNotEqual(coupler_signature(coupler, increments=True), inside)
                x._pending_opening = (np.zeros(3, dtype=int), np.zeros(3))
                self.assertIsNone(coupler_signature(coupler, increments=True))
                x._pending_opening = None
            # No engaged track: surface contact alone has no supported history.
            self.assertIsNone(coupler_signature(coupler, increments=True))


if __name__ == "__main__":
    unittest.main()
