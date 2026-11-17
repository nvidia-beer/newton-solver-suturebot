# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""NeedleTrackEyelets: which rows of a needle's track become eyelets of the thread, by hand.

The rule is host bookkeeping, so most tests hand it rows through a stand-in for the needle/tissue
interaction (its three members that the follower reads) and a stand-in for the coupling that keeps what
it is asked to make. One test uses the real interaction and the real coupling on a small block: the
eyelet that is made from a track row sits where the interaction's own sample is in the deformed tissue.
"""

import unittest

import numpy as np
import warp as wp
from test_thread_tissue import DEVICE, RESTING, _line, _Scene

from newton_solver_suturebot import NeedleTissueInteraction
from newton_solver_suturebot._src.solvers.thread import NeedleTrackEyelets

SPACING = 0.0015  # [m], the eyelet spacing of the worker thread
# Rows of a track are a quarter millimetre apart; 0.26 mm keeps a row off the exact multiples of the spacing.
ROW = 0.00026  # [m]


class Rows:
    """The members of ``NeedleTissueInteraction`` that the follower reads, filled by the test."""

    def __init__(self):
        self.tetrahedra, self.weights, self.rest, self.sections = [], [], [], []
        self.events = ()
        self.reads = 0

    @property
    def track_count(self):
        return len(self.tetrahedra)

    def track_rows(self, start=0):
        self.reads += 1
        rows = slice(int(start), self.track_count)
        return (
            np.asarray(self.tetrahedra[rows], dtype=np.int32),
            np.asarray(self.weights[rows], dtype=float).reshape(-1, 4),
            np.asarray(self.rest[rows], dtype=float).reshape(-1, 3),
            np.asarray(self.sections[rows], dtype=np.int32),
        )

    def lay(self, section, distance, *, start=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0)):
        """Record one row of ``section`` at ``distance`` [m] along a line; its tetrahedron is its row number."""
        row = self.track_count
        self.tetrahedra.append(row)
        self.weights.append(np.array([0.1, 0.2, 0.3, 0.4]) + 0.001 * row)
        self.rest.append(np.asarray(start, dtype=float) + distance * np.asarray(direction, dtype=float))
        self.sections.append(section)
        return row


class Made:
    """The reserve of a ``ThreadTissueCoupling`` as ``add_track_point`` keeps it: slots, tracks, refusals."""

    def __init__(self, eyelets=36, tracks=4):
        self.eyelets, self.tracks = eyelets, tracks
        self.points, self.track_count, self.refused_points = [], 0, 0

    def add_track_point(self, track, tetrahedron, weights):
        if not 0 <= track <= self.track_count:
            raise ValueError("no such track")
        if len(self.points) >= self.eyelets or track >= self.tracks:
            self.refused_points += 1
            return -1
        self.points.append((track, int(tetrahedron), np.asarray(weights, dtype=float).copy()))
        self.track_count = max(self.track_count, track + 1)
        return len(self.points) - 1

    def rows(self, track=None):
        """Tetrahedra of the eyelets made (the row numbers of :meth:`Rows.lay`), of one track or of all."""
        return [tetrahedron for group, tetrahedron, _ in self.points if track is None or group == track]


def follower(eyelets=36, tracks=4):
    interaction, coupling = Rows(), Made(eyelets, tracks)
    scene = type("Scene", (), {})()
    scene.interaction = interaction
    return interaction, coupling, scene, NeedleTrackEyelets(coupling, scene, SPACING)


class EyeletRuleTests(unittest.TestCase):
    def test_the_first_row_and_every_row_a_spacing_on_become_eyelets_as_the_rows_arrive(self):
        """Fails if a row other than the first of its section and the first one a spacing from the last eyelet is made."""
        interaction, coupling, _scene, track = follower()
        made = []
        for row in range(20):
            interaction.lay(0, row * ROW)
            track.advance()
            made.append(coupling.rows())
        # 6 rows are 1.56 mm, the first distance of at least 1.5 mm: rows 0, 6, 12, 18.
        self.assertEqual(made[0], [0])
        self.assertEqual(made[5], [0])
        self.assertEqual(made[6], [0, 6])
        self.assertEqual(made[11], [0, 6])
        self.assertEqual(made[19], [0, 6, 12, 18])
        self.assertEqual((track.passages, track.rows, track.refused), ([[0, 1, 2, 3]], 20, 0))
        # The weights of an eyelet are those of its row, in their order.
        np.testing.assert_array_equal(coupling.points[2][2], interaction.weights[12])
        # The distance is that of the REST positions from the last eyelet, not a count of rows: a row that
        # comes back to within a spacing of the last eyelet is none, however many rows lie between.
        interaction, coupling, _scene, track = follower()
        for distance in (0.0, 0.0014, 0.0029, 0.0016, 0.0031):
            interaction.lay(0, distance)
            track.advance()
        self.assertEqual(coupling.rows(), [0, 2])

    def test_rows_that_arrive_together_give_the_eyelets_of_rows_that_arrive_one_at_a_time(self):
        """Fails if the eyelets depend on how many rows a call takes."""
        one, coupling_one, _scene, track_one = follower()
        many, coupling_many, _scene, track_many = follower()
        for row in range(17):
            one.lay(row // 9, row * ROW)
            many.lay(row // 9, row * ROW)
            track_one.advance()
        track_many.advance()
        self.assertEqual(coupling_many.rows(), coupling_one.rows())
        self.assertEqual(track_many.passages, track_one.passages)
        self.assertEqual(many.reads, 1)

    def test_the_exit_makes_the_last_row_an_eyelet_unless_the_last_eyelet_is_within_half_a_spacing(self):
        """Fails if the hole where the thread comes out gets no eyelet, or one that crowds the last eyelet."""
        # Rows 0 .. 10: eyelets at 0 and 6; row 10 is 1.04 mm from row 6, more than half a spacing.
        interaction, coupling, _scene, track = follower()
        for row in range(11):
            interaction.lay(0, row * ROW)
            track.advance()
        self.assertEqual(coupling.rows(), [0, 6])
        # Without an exit nothing more is made, however often the follower is asked.
        track.advance()
        self.assertEqual(coupling.rows(), [0, 6])
        interaction.events = ({"kind": "entry"}, {"kind": "puncture"})
        track.advance()
        self.assertEqual(coupling.rows(), [0, 6])
        interaction.events = ({"kind": "exit"},)
        track.advance()
        self.assertEqual(coupling.rows(), [0, 6, 10])
        # A row is written once: the same exit seen again makes nothing.
        track.advance()
        self.assertEqual(coupling.rows(), [0, 6, 10])
        self.assertEqual(track.passages, [[0, 1, 2]])
        # Rows 0 .. 8: row 8 is 0.52 mm from the eyelet at row 6, less than half a spacing (0.75 mm): skipped.
        interaction, coupling, _scene, track = follower()
        for row in range(9):
            interaction.lay(0, row * ROW)
        interaction.events = ({"kind": "exit"},)
        track.advance()
        self.assertEqual(coupling.rows(), [0, 6])
        # Exactly at the limit of the rule: 3 rows of 0.26 mm are 0.78 mm, the first distance of at least 0.75 mm.
        interaction, coupling, _scene, track = follower()
        for row in range(10):
            interaction.lay(0, row * ROW)
        interaction.events = ({"kind": "exit"},)
        track.advance()
        self.assertEqual(coupling.rows(), [0, 6, 9])
        # A passage of one row has its one eyelet and no second one at the exit.
        interaction, coupling, _scene, track = follower()
        interaction.lay(0, 0.0)
        interaction.events = ({"kind": "exit"},)
        track.advance()
        self.assertEqual(coupling.rows(), [0])

    def test_two_sections_give_two_groups_in_the_order_the_tip_laid_them(self):
        """Fails if the eyelets of two passages share a track, or a passage's eyelets are out of the tip's order."""
        interaction, coupling, _scene, track = follower()
        for row in range(8):  # the first lip
            interaction.lay(0, row * ROW)
            track.advance()
        interaction.events = ({"kind": "exit"},)
        track.advance()
        interaction.events = ()
        for row in range(8):  # the second lip, 3 mm on
            interaction.lay(1, 0.005 + row * ROW)
            track.advance()
        interaction.events = ({"kind": "exit"},)
        track.advance()
        self.assertEqual(coupling.rows(0), [0, 6])
        self.assertEqual(coupling.rows(1), [8, 14])
        self.assertEqual(track.passages, [[0, 1], [2, 3]])
        self.assertEqual(coupling.track_count, 2)
        # The spacing is counted per section: the first row of the second lip is an eyelet although
        # it is nearer than a spacing to nothing of its own section, and far from the first lip's.
        # A row that the tip lays in the first section afterwards joins the first group.
        interaction.events = ()
        interaction.lay(0, 8 * ROW + 0.0016)
        track.advance()
        self.assertEqual(coupling.rows(0), [0, 6, 16])
        self.assertEqual(track.passages, [[0, 1, 4], [2, 3]])

    def test_a_reserve_that_is_used_up_refuses_and_counts_and_nothing_grows(self):
        """Fails if an eyelet beyond the reserve is made, is not counted, or stops the spacing of the ones after it."""
        interaction, coupling, _scene, track = follower(eyelets=3, tracks=4)
        for row in range(26):  # eyelets at rows 0, 6, 12, 18, 24: the reserve holds three
            interaction.lay(0, row * ROW)
            track.advance()
        self.assertEqual(coupling.rows(), [0, 6, 12])
        self.assertEqual((track.refused, track.passages), (2, [[0, 1, 2]]))
        # No track left: the eyelets of a third section are refused one by one, and the first two stay.
        interaction, coupling, _scene, track = follower(eyelets=36, tracks=2)
        for section in range(3):
            for row in range(8):
                interaction.lay(section, 0.01 * section + row * ROW)
                track.advance()
        self.assertEqual((coupling.rows(0), coupling.rows(1), coupling.track_count), ([0, 6], [8, 14], 2))
        self.assertEqual((track.refused, track.passages), (2, [[0, 1], [2, 3]]))

    def test_nothing_new_costs_no_read_of_the_rows(self):
        """Fails if a call without a new row or event reads the track."""
        interaction, _coupling, scene, track = follower()
        track.advance()
        self.assertEqual(interaction.reads, 0)
        interaction.lay(0, 0.0)
        track.advance()
        track.advance()
        track.advance()
        self.assertEqual(interaction.reads, 1)
        # A scene that has no interaction yet has no rows.
        scene.interaction = None
        fresh = NeedleTrackEyelets(Made(), scene, SPACING)
        fresh.advance()
        self.assertEqual((fresh.rows, fresh.passages), (0, []))

    def test_a_changed_interaction_raises(self):
        """Fails if rows of another interaction are taken for the rows of the one the eyelets came from."""
        interaction, _coupling, scene, track = follower()
        # Before a row was taken the scene may still get its interaction.
        scene.interaction = Rows()
        track.advance()
        scene.interaction = interaction
        interaction.lay(0, 0.0)
        track.advance()
        other = Rows()
        other.lay(0, 0.0)
        other.lay(0, 0.002)
        scene.interaction = other
        with self.assertRaisesRegex(RuntimeError, "interaction changed"):
            track.advance()
        scene.interaction = None
        with self.assertRaisesRegex(RuntimeError, "interaction changed"):
            track.advance()
        with self.assertRaisesRegex(ValueError, "spacing"):
            NeedleTrackEyelets(Made(), scene, 0.0)


class TrackSampleTests(unittest.TestCase):
    def test_an_eyelet_sits_where_the_interactions_own_track_sample_is_in_the_deformed_tissue(self):
        """Fails if tetrahedron or weights of an eyelet are not those of its track row, in the coupling's vertex order."""
        if DEVICE.startswith("cuda") and not wp.is_cuda_available():
            self.skipTest("NEWTON_TEST_DEVICE asks for CUDA and there is none")
        # A thread over the long block with room for eyelets, and an interaction on the same tetrahedra.
        tissue = _Scene(_line((0.0031, 0.0013, 0.001), RESTING + 1), eyelets=6, max_tracks=2)
        interaction = NeedleTissueInteraction(
            tissue.rest,
            tissue.tetrahedra,
            tissue.triangles,
            np.array([0]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.005]]),
            np.full(2, 0.00028),
            np.zeros(3),
            device=DEVICE,
            track_spacing=0.00025,
        )
        scene = type("Scene", (), {})()
        scene.interaction = interaction
        track = NeedleTrackEyelets(tissue.coupling, scene, SPACING)
        # Material points down a slanted line through the block, 0.4 mm apart, as a tip lays them: the track
        # rows of the interaction itself (its own embedding, _record_track), each with uneven weights.
        start, step = np.array([0.0102, 0.0013, -0.0004]), np.array([0.00033, 0.00011, -0.0002])
        points = [start + k * step for k in range(12)]
        for point in points:
            ids, bary = interaction._locate(point, tissue.rest.astype(float))
            interaction._record_track(ids, bary)
            track.advance()
        self.assertEqual(interaction.track_count, len(points))
        tetrahedra, weights, rest, sections = interaction.track_rows()
        np.testing.assert_allclose(rest, points, atol=1e-9)
        np.testing.assert_array_equal(sections, 0)
        # Rows from a later start are the tail of all rows.
        np.testing.assert_array_equal(interaction.track_rows(9)[0], tetrahedra[9:])
        self.assertEqual(len(interaction.track_rows(len(points))[0]), 0)
        # The eyelets: rows 0, 4, 8 (four rows are 1.6 mm).
        self.assertEqual(track.passages, [[0, 1, 2]])
        # Deform the tissue unevenly, so that a wrong vertex order or tetrahedron moves the eyelet.
        rng = np.random.default_rng(3)
        moved = tissue.rest + 0.0004 * rng.standard_normal(tissue.rest.shape)
        tissue.place(moved)
        tissue.coupling.update(tissue.particle_q)
        targets = tissue.thread.eyelet_target.numpy()
        nodes = np.asarray(interaction._track_nodes_host)
        bary = np.asarray(interaction._track_bary_host)
        placed = moved.astype(np.float32).astype(float)
        for slot, row in zip(track.passages[0], (0, 4, 8), strict=True):
            sample = bary[row] @ placed[nodes[row]]
            # The interaction's own sample of that row, at the same deformed tissue: within 1e-6 m.
            np.testing.assert_allclose(targets[slot], sample, atol=1e-6)
            # A wrong order would be off by the deformation, hundreds of micrometres.
            wrong = bary[row][::-1] @ placed[nodes[row]]
            self.assertGreater(np.linalg.norm(wrong - sample), 2e-5)


if __name__ == "__main__":
    unittest.main()
