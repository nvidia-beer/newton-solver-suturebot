# SPDX-License-Identifier: Apache-2.0
"""The start guess of a captured correction: one graph for every track, by device flags.

A captured trial loop holds the start of the correction's solve once. Whether a launch may use
the bound response basis is the device flag ``available`` of the guess; the padded tile that
factorizes the track samples of a launch is named by the size flags of its device solve. The
host writes both where they change, so neither a new track sample nor a new support of the
basis, nor a basis that one launch may not use, is a new graph.

Checked here:

- a start that may not use the basis (too many track samples, too many support columns, an
  inactive manager) is the zero vector, marked as a discarded guess: no Schur scratch is
  touched, the first residual of the solve is the right-hand side bit for bit, and nothing is
  counted; issued as written on any device, and replayed from a captured graph on CUDA, which
  the same graph follows back to the guess when the basis is usable again;
- one captured correction replayed for every track count of the scratch, with columns that fit
  the compacted tile and with columns that do not, and for another support of the basis: the
  bits of the launches issued one by one with the tile of that count;
- a guess bound ahead of the first track sample (a response core whose storage a start-up made,
  with no column in it): the start is the zero start until the first sample, and one start
  captured in that state serves the first sample and those after it with the bits of the
  launches issued one by one;
- the preparation of the host packet is not keyed on the number of track samples or of support
  columns.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_woodbury_device as fixture
import warp as wp

from newton_solver_suturebot._src.solvers.coupled.acceleration import woodbury_device
from newton_solver_suturebot._src.solvers.coupled.acceleration.contact_preparation import PacketPreparation
from newton_solver_suturebot._src.solvers.coupled.acceleration.woodbury_device import _COMPACT_ROWS
from newton_solver_suturebot._src.solvers.coupled.acceleration.woodbury_guess import (
    _CAPACITY,
    _HEADER,
    _MAX_TRACKS,
    WoodburyInitialGuess,
)
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import LIVE_TRACK
from newton_solver_suturebot._src.utils.capture import ScopedCapture

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
CUDA = DEVICE.startswith("cuda")


def set_tracks(packet, count):
    """Number of track samples on the device (live counts) and on the host."""
    live = packet.pre.coupler.interaction._live
    counts = live.numpy()
    counts[LIVE_TRACK] = count
    live.assign(counts)
    packet.pre.count = count


def solver_of(packet, iterations):
    """A finished Krylov solve of ``iterations`` iterations, as ``after_solve`` reads it."""
    device = packet.pre.device
    packet.pre.linear_solver = SimpleNamespace(
        _cur_iter_and_condition=wp.array([iterations, 0], dtype=int, device=device), _maxiter=80
    )


def poison(helper):
    """Values that no kernel writes in every output of the device coefficients and of the start."""
    fixture._reset(helper)
    helper.coefficients.fill_(np.nan)
    helper.selected.fill_(-1)
    helper.pre.z.fill_(wp.vec3(7.0))
    helper.product.fill_(wp.vec3(np.nan))


def outputs(helper):
    values = fixture._snapshot(helper)
    values["z"] = helper.pre.z.numpy().copy()
    values["selected"] = helper.selected.numpy().copy()
    return values


def same_bits(test, actual, expected, label):
    test.assertEqual(sorted(actual), sorted(expected))
    for name in actual:
        test.assertEqual(actual[name].tobytes(), expected[name].tobytes(), f"{label}: {name}")


def device_start(helper, served):
    """What a correction of a captured trial loop issues for its start, its first residual and its count."""
    pre = helper.pre
    helper.device_coefficients = True
    try:
        helper.initialize()
        if not helper.first_residual(pre.z, pre.rhs, served, -1.0, 1.0):
            raise AssertionError("The start did not serve the first residual")
        helper.after_solve()
    finally:
        helper.device_coefficients = False


# Each makes the bound basis unusable for a launch, and undoes it.
def too_many_tracks(packet):
    set_tracks(packet, _MAX_TRACKS + 1)
    return lambda: set_tracks(packet, 6)


def too_many_columns(packet):
    support = packet.manager.support
    packet.manager.support = support + tuple(range(900, 900 + _CAPACITY))
    return lambda: setattr(packet.manager, "support", support)


def inactive_manager(packet):
    packet.manager.active = False
    return lambda: setattr(packet.manager, "active", True)


UNUSABLE = {
    "one track sample more than the scratch holds": too_many_tracks,
    "more support columns than the scratch holds": too_many_columns,
    "inactive manager": inactive_manager,
}


class ZeroStartTests(unittest.TestCase):
    def packet(self, seed):
        # The live track storage holds one sample more than the scratch of the guess.
        packet = fixture.Packet(seed, _MAX_TRACKS + 1)
        set_tracks(packet, 6)
        solver_of(packet, 4)
        return packet

    def test_start_without_a_usable_basis_is_zero_touches_no_scratch_and_counts_nothing(self):
        for index, (name, make_unusable) in enumerate(UNUSABLE.items()):
            with self.subTest(name):
                packet = self.packet(700 + index)
                helper, pre = packet.helper, packet.pre
                served = wp.zeros_like(pre.rhs)
                key = helper.prepare_device(packet.manager)
                self.assertIsNotNone(key)
                restore = make_unusable(packet)
                self.assertEqual(helper.prepare_device(packet.manager), key)
                self.assertEqual(helper.available.numpy().tolist(), [0])
                poison(helper)
                before = fixture._snapshot(helper)
                counters = helper.counters.numpy().copy()
                # Outside a capture the host knows the flag: no conditional reads it back.
                with patch.object(wp, "capture_if", side_effect=AssertionError("a read of the flag on the host")):
                    helper.device_solve.enqueue_topology()
                    device_start(helper, served)
                same_bits(self, fixture._snapshot(helper), before, name)
                np.testing.assert_array_equal(pre.z.numpy(), 0)
                self.assertEqual(helper.selected.numpy().tolist(), [0])
                self.assertEqual(served.numpy().tobytes(), pre.rhs.numpy().tobytes())
                np.testing.assert_array_equal(helper.counters.numpy(), counters)
                # Usable again: the same key, and the guess of the host solve.
                restore()
                expected = packet.host()
                self.assertEqual(helper.prepare_device(packet.manager), key)
                self.assertEqual(helper.available.numpy().tolist(), [1])
                poison(helper)
                helper.device_solve.enqueue_topology()
                device_start(helper, served)
                self.assertEqual(helper.enabled.numpy().tolist(), [1])
                self.assertEqual(helper.selected.numpy().tolist(), [1])
                fixture._close(helper.coefficients.numpy(), expected, helper)
                self.assertTrue(np.any(pre.z.numpy() != 0))
                self.assertTrue(np.any(served.numpy() != pre.rhs.numpy()))
                change = helper.counters.numpy() - counters
                # One application, selected, with the iterations of its solve.
                self.assertEqual(change.tolist()[:4], [1, 1, 4, 0])

    def test_recorded_packet_counts_nothing_while_its_basis_is_unavailable(self):
        recorded = fixture._recorded()
        helper, pre = recorded.helper, recorded.pre
        pre.linear_solver = SimpleNamespace(
            _cur_iter_and_condition=wp.array([9, 0], dtype=int, device=pre.device), _maxiter=80
        )
        served = wp.zeros_like(pre.rhs)
        key = helper.prepare_device(recorded.manager)
        recorded.manager.active = False
        self.assertEqual(helper.prepare_device(recorded.manager), key)
        for _ in range(3):
            helper.device_solve.enqueue_topology()
            device_start(helper, served)
            self.assertEqual(served.numpy().tobytes(), pre.rhs.numpy().tobytes())
        # As without a guess: no application, no selection, no iteration is counted.
        np.testing.assert_array_equal(helper.counters.numpy(), 0)
        self.assertEqual((helper.preparations, helper.ready, helper.fallbacks), (0, 0, 0))
        recorded.manager.active = True
        self.assertEqual(helper.prepare_device(recorded.manager), key)
        helper.device_solve.enqueue_topology()
        device_start(helper, served)
        self.assertEqual(helper.counters.numpy().tolist()[:4], [1, 1, 9, 0])

    @unittest.skipUnless(CUDA, "Conditional CUDA graph replay contract")
    def test_one_captured_start_follows_the_device_flag_both_ways(self):
        packet = self.packet(711)
        helper, pre = packet.helper, packet.pre
        served = wp.zeros_like(pre.rhs)
        key = helper.prepare_device(packet.manager)
        owners = [array.ptr for array in helper.owners + helper.device_solve.owners]
        remaining = wp.zeros(1, dtype=int, device=pre.device)

        def trial():
            device_start(helper, served)
            wp.launch(fixture.countdown, 1, [remaining], device=pre.device)

        with ScopedCapture(device=pre.device) as capture:
            helper.device_solve.enqueue_topology()
            wp.capture_while(remaining, trial)

        def replay():
            poison(helper)
            served.fill_(wp.vec3(-3.0))
            remaining.fill_(2)
            wp.capture_launch(capture.graph)

        expected = packet.host()
        self.assertEqual(helper.prepare_device(packet.manager), key)
        replay()
        self.assertEqual((helper.enabled.numpy().tolist(), helper.selected.numpy().tolist()), ([1], [1]))
        fixture._close(helper.coefficients.numpy(), expected, helper)
        guess = pre.z.numpy().copy()
        self.assertTrue(np.any(guess != 0))
        counted = helper.counters.numpy().copy()
        self.assertEqual(counted.tolist()[:4], [2, 2, 8, 0])
        for name, make_unusable in UNUSABLE.items():
            with self.subTest(name):
                restore = make_unusable(packet)
                self.assertEqual(helper.prepare_device(packet.manager), key)
                poison(helper)
                before = fixture._snapshot(helper)
                replay()
                # The same graph: nothing of the Schur scratch is written, the start is zero and
                # discarded, its first residual is the right-hand side, and nothing is counted.
                same_bits(self, fixture._snapshot(helper), before, name)
                np.testing.assert_array_equal(pre.z.numpy(), 0)
                self.assertEqual(helper.selected.numpy().tolist(), [0])
                self.assertEqual(served.numpy().tobytes(), pre.rhs.numpy().tobytes())
                np.testing.assert_array_equal(helper.counters.numpy(), counted)
                restore()
                self.assertEqual(helper.prepare_device(packet.manager), key)
                replay()
                self.assertEqual(helper.selected.numpy().tolist(), [1])
                self.assertEqual(pre.z.numpy().tobytes(), guess.tobytes())
                counted = helper.counters.numpy().copy()
        self.assertEqual([array.ptr for array in helper.owners + helper.device_solve.owners], owners)


class CapturedSizeTests(unittest.TestCase):
    @unittest.skipUnless(CUDA, "Conditional CUDA graph replay contract")
    def test_one_captured_correction_serves_every_track_count_with_the_bits_of_its_tile(self):
        packet = fixture.Packet(721, _MAX_TRACKS)
        helper, pre = packet.helper, packet.pre
        solve_of = lambda: helper.device_solve  # noqa: E731
        set_tracks(packet, 1)
        key = helper.prepare_device(packet.manager)
        self.assertIsNotNone(key)
        solve = solve_of()
        remaining = wp.zeros(1, dtype=int, device=pre.device)
        applied = wp.ones(1, dtype=int, device=pre.device)
        served = wp.zeros_like(pre.rhs)
        solver_of(packet, 3)

        def trial():
            # As in the trial loop: the correction is itself a branch of a loop body.
            wp.capture_if(applied, lambda: device_start(helper, served))
            wp.launch(fixture.countdown, 1, [remaining], device=pre.device)

        # Captured once, with one track sample.
        with ScopedCapture(device=pre.device) as capture:
            solve.enqueue_topology()
            wp.capture_while(remaining, trial)

        sticking = np.asarray([fixture._tangent(np.random.default_rng(7), 3) for _ in range(_MAX_TRACKS)])
        tiles, compacted, padded_large = set(), 0, 0

        def compare(count, matrices, label):
            nonlocal compacted, padded_large
            set_tracks(packet, count)
            pre.matrices.assign(matrices)
            self.assertEqual(helper.prepare_device(packet.manager), key, label)
            self.assertIs(solve_of(), solve)
            poison(helper)
            remaining.fill_(2)
            wp.capture_launch(capture.graph)
            replayed = outputs(helper)
            self.assertEqual(replayed["enabled"].tolist(), [1], label)
            # The same launches issued one by one, with the tile of this count and its branch.
            poison(helper)
            solve.enqueue_topology()
            fixture._correction(solve)
            helper.initialize()
            same_bits(self, replayed, outputs(helper), label)
            indexed = 3 * solve.capacity > _COMPACT_ROWS
            self.assertEqual(replayed["compact_fits"].tolist()[0] != -5, indexed, label)
            tiles.add(solve.capacity)
            compacted += int(fixture._compacted(solve))
            padded_large += int(indexed and not fixture._compacted(solve))
            return replayed

        seen = {}
        for count in range(1, _MAX_TRACKS + 1):
            with self.subTest(tracks=count):
                seen[count] = compare(count, packet.matrices, f"{count} samples")["coefficients"].tobytes()
                self.assertEqual(solve.capacity, woodbury_device._padded_samples(count))
        # The replays follow the count: every new sample with a tangent changes the proposal.
        self.assertGreater(len(set(seen.values())), _MAX_TRACKS // 2)
        # More nonzero columns than the compacted tile holds: the padded tiles of 120 and 144 rows.
        for count in (39, 40, 41, _MAX_TRACKS):
            with self.subTest(tracks=count, columns="sticking"):
                compare(count, sticking, f"{count} sticking samples")
        # Back at a small tile, and the first count again: the bits of the first replay.
        with self.subTest("repeat"):
            self.assertEqual(compare(1, packet.matrices, "1 sample again")["coefficients"].tobytes(), seen[1])
            self.assertEqual(compare(17, packet.matrices, "17 samples again")["coefficients"].tobytes(), seen[17])
        self.assertEqual(tiles, set(woodbury_device._TILES))
        self.assertGreater(compacted, 0)
        self.assertEqual(padded_large, 4)
        # Another order of the support columns of the basis: the same graph, the host's proposal.
        with self.subTest("support"):
            set_tracks(packet, 9)
            pre.matrices.assign(packet.matrices)
            support = packet.manager.support[::-1]
            flat = packet.manager.core.basis.numpy()
            flat[: packet.response.size] = packet.response[:, ::-1].ravel(order="F")
            packet.manager.core.basis.assign(flat)
            packet.manager.core.support.assign(np.asarray(support, dtype=np.int32))
            packet.manager.support = support
            packet.manager.basis_builds += 1
            expected = packet.host()
            self.assertEqual(helper.prepare_device(packet.manager), key)
            poison(helper)
            remaining.fill_(1)
            wp.capture_launch(capture.graph)
            self.assertEqual(helper.enabled.numpy().tolist(), [1])
            self.assertEqual(helper.selected.numpy().tolist(), [1])
            fixture._close(helper.coefficients.numpy(), expected, helper)


def ahead_of_the_track(packet):
    """Put ``packet`` into the state that a start-up leaves which makes the storage of the response basis first.

    A core of the current FEM structure without a column, an inactive manager and no track sample.
    Returns the call that brings a track of ``count`` samples with the support of its basis.
    """
    manager = packet.manager
    support = manager.support
    manager.active, manager.support = False, ()
    manager.core.count.fill_(0)
    set_tracks(packet, 0)

    def track(count):
        manager.active, manager.support = True, support
        manager.core.count.fill_(len(support))
        set_tracks(packet, count)

    return track


class AheadOfTheTrackTests(unittest.TestCase):
    def assert_zero_start(self, helper, served, before, counters, label):
        """Nothing of the Schur scratch written, the start zero and discarded, the residual served, nothing counted."""
        same_bits(self, fixture._snapshot(helper), before, label)
        np.testing.assert_array_equal(helper.pre.z.numpy(), 0)
        self.assertEqual(helper.selected.numpy().tolist(), [0], label)
        self.assertEqual(served.numpy().tobytes(), helper.pre.rhs.numpy().tobytes(), label)
        np.testing.assert_array_equal(helper.counters.numpy(), counters)

    def test_guess_bound_ahead_of_the_track_starts_from_zero_until_the_first_sample(self):
        packet = fixture.Packet(741, 12)
        helper, pre = packet.helper, packet.pre
        solver_of(packet, 4)
        served = wp.zeros_like(pre.rhs)
        track = ahead_of_the_track(packet)
        key = helper.prepare_device(packet.manager)
        self.assertIsNotNone(key)
        self.assertEqual(helper.available.numpy().tolist(), [0])
        solve = helper.device_solve
        storage = [array.ptr for array in helper.owners + solve.owners]
        # No padded tile was ever selected: a start issued as written is the zero start alone.
        self.assertIsNone(solve.solve_kernel)
        poison(helper)
        before = fixture._snapshot(helper)
        with patch.object(wp, "capture_if", side_effect=AssertionError("a read of the flag on the host")):
            solve.enqueue_topology()
            device_start(helper, served)
        self.assert_zero_start(helper, served, before, np.zeros(6, dtype=np.int32), "ahead of the track")
        # The first track sample, and more of them: the same key and storage, the host's proposal.
        applications = 0
        for count in (1, 2, 12):
            with self.subTest(tracks=count):
                track(count)
                expected = packet.host()
                self.assertEqual(helper.prepare_device(packet.manager), key)
                self.assertEqual(helper.available.numpy().tolist(), [1])
                self.assertEqual(solve.capacity, woodbury_device._padded_samples(count))
                poison(helper)
                with patch.object(wp, "capture_if", side_effect=AssertionError("a read of the flag on the host")):
                    solve.enqueue_topology()
                    device_start(helper, served)
                self.assertEqual(helper.enabled.numpy().tolist(), [1])
                self.assertEqual(helper.selected.numpy().tolist(), [1])
                fixture._close(helper.coefficients.numpy(), expected, helper)
                self.assertTrue(np.any(pre.z.numpy() != 0))
                applications += 1
                # One application each, selected, with the iterations of its solve.
                self.assertEqual(
                    helper.counters.numpy().tolist()[:4], [applications, applications, 4 * applications, 0]
                )
        self.assertIs(helper.device_solve, solve)
        self.assertEqual([array.ptr for array in helper.owners + solve.owners], storage)

    @unittest.skipUnless(CUDA, "Conditional CUDA graph replay contract")
    def test_one_start_captured_ahead_of_the_track_serves_the_first_sample_and_those_after_it(self):
        # The packet of the captured-size test above, here captured before its first sample.
        packet = fixture.Packet(721, _MAX_TRACKS)
        helper, pre = packet.helper, packet.pre
        solver_of(packet, 3)
        served = wp.zeros_like(pre.rhs)
        track = ahead_of_the_track(packet)
        key = helper.prepare_device(packet.manager)
        self.assertIsNotNone(key)
        self.assertEqual(helper.available.numpy().tolist(), [0])
        solve = helper.device_solve
        storage = [array.ptr for array in helper.owners + solve.owners]
        remaining = wp.zeros(1, dtype=int, device=pre.device)
        applied = wp.ones(1, dtype=int, device=pre.device)

        def trial():
            # As in the trial loop: the correction is itself a branch of a loop body.
            wp.capture_if(applied, lambda: device_start(helper, served))
            wp.launch(fixture.countdown, 1, [remaining], device=pre.device)

        # Captured once, before a basis exists, a support column is mapped or a tile is selected.
        self.assertIsNone(solve.solve_kernel)
        with ScopedCapture(device=pre.device) as capture:
            solve.enqueue_topology()
            wp.capture_while(remaining, trial)

        def replay():
            poison(helper)
            served.fill_(wp.vec3(-3.0))
            remaining.fill_(2)
            before = fixture._snapshot(helper)
            wp.capture_launch(capture.graph)
            return before

        # Launches ahead of the track take the zero start.
        counters = helper.counters.numpy().copy()
        np.testing.assert_array_equal(counters, 0)
        for _ in range(2):
            before = replay()
            self.assert_zero_start(helper, served, before, counters, "ahead of the track")
        # The first track sample and those after it, through every padded tile: the same graph
        # leaves the bits of the launches issued one by one with the tile of that count.
        sticking = np.asarray([fixture._tangent(np.random.default_rng(7), 3) for _ in range(_MAX_TRACKS)])
        cases = [(count, packet.matrices) for count in (1, 2, 4, 5, 9, 17, 38, 39, 40, _MAX_TRACKS)]
        # More nonzero columns than the compacted tile holds: the padded tiles of 120 and 144 rows.
        cases += [(40, sticking), (_MAX_TRACKS, sticking)]
        tiles, padded_large = set(), 0
        for index, (count, matrices) in enumerate(cases):
            with self.subTest(case=index, tracks=count):
                track(count)
                pre.matrices.assign(matrices)
                self.assertEqual(helper.prepare_device(packet.manager), key)
                self.assertEqual(helper.available.numpy().tolist(), [1])
                replay()
                replayed = outputs(helper)
                self.assertEqual(replayed["enabled"].tolist(), [1])
                self.assertTrue(np.any(served.numpy() != -3.0))
                # The same launches issued one by one, with the tile of this count and its branch.
                poison(helper)
                solve.enqueue_topology()
                fixture._correction(solve)
                helper.initialize()
                same_bits(self, replayed, outputs(helper), f"case {index}: {count} samples")
                tiles.add(solve.capacity)
                padded_large += int(3 * solve.capacity > _COMPACT_ROWS and not fixture._compacted(solve))
        self.assertEqual(tiles, set(woodbury_device._TILES))
        self.assertEqual(padded_large, 2)
        # Each of those launches counted its two starts and each comparison its one; none before them.
        self.assertEqual(int(helper.counters.numpy()[0]), 3 * len(cases))
        # Ahead of the track again, as after a reset of the scene: the zero start of the same graph.
        ahead_of_the_track(packet)
        self.assertEqual(helper.prepare_device(packet.manager), key)
        counters = helper.counters.numpy().copy()
        before = replay()
        self.assert_zero_start(helper, served, before, counters, "ahead of the track again")
        self.assertIs(helper.device_solve, solve)
        self.assertEqual([array.ptr for array in helper.owners + solve.owners], storage)


class PacketKeyTests(unittest.TestCase):
    def test_packet_preparation_is_not_keyed_on_the_track_or_the_support_count(self):
        packet = fixture.Packet(731, 12)
        helper, pre = packet.helper, packet.pre
        pre.capture_preparation = True
        pre.coupler.model = SimpleNamespace(requires_grad=False)
        pre.coupler.fem_solver = SimpleNamespace(differentiable=False)
        reference = WoodburyInitialGuess(pre)
        keys = []
        support = packet.manager.support

        def prepare(count, columns):
            set_tracks(packet, count)
            packet.manager.support = columns
            packet.manager.core.count.fill_(len(columns))
            packet.manager.basis_builds += 1
            helper.prepare(packet.manager)
            self.assertIsInstance(helper._packet_preparation, PacketPreparation)
            keys.append(helper._packet_preparation.replay.key)
            actual = helper.packet.numpy().copy()
            # The same packet as written, without a graph.
            pre.capture_preparation = False
            reference.packet.zero_()
            reference.prepare(packet.manager)
            pre.capture_preparation = True
            # Header and residual on the support columns in use, the rigid residual, the records of the track samples.
            used = np.r_[0 : 38 + len(columns), 38 + _CAPACITY : _HEADER + 20 * count]
            np.testing.assert_array_equal(actual[used], reference.packet.numpy()[used])
            self.assertTrue(np.any(actual[_HEADER : _HEADER + 20 * count] != 0))

        # More track samples, then fewer support columns, then both again.
        prepare(12, support)
        prepare(12, support)
        prepare(7, support)
        prepare(7, support[:-3])
        prepare(12, support)
        replay = helper._packet_preparation.replay
        if CUDA:
            self.assertEqual(len(set(keys)), 1)
            self.assertEqual((replay.eager, replay.captures, replay.launches), (1, 1, 4))
        else:
            self.assertEqual(replay.eager, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
