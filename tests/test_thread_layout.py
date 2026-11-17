# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Where the spare thread of a needle is laid: on the pad's surface, behind the stitch, along the pad's border.

Geometry only: no solver runs. The pad of these tests is a square plate with a mound, given as its top surface.
"""

import math
import unittest

import numpy as np

from newton_solver_suturebot._src.solvers.thread.thread_initialization import (
    border_walk,
    bridge,
    lay_along_path,
    loose_thread_coordinates,
    rounded,
    surface_top,
)

LENGTH, ELEMENTS, RADIUS = 0.18, 90, 0.00015  # [m], the scene's thread
ELEMENT = LENGTH / ELEMENTS
TURN = 0.010  # [m]
NEEDLE = 0.009  # [m], radius of the needle's circle
HALF = 0.05  # [m], half the plate's side
MOUND = 0.004  # [m], height of the mound
ENTRY, EXIT = np.array([0.0, -0.010, 0.0]), np.array([0.0, -0.020, 0.0])
HEAD = np.array([0.0, -0.008, 0.015, 0.0, -1.0, 0.0])  # above the entry, the tail pointing to the exit


def _height(x, y):
    """Top of the plate [m]: level, with a cone of MOUND and 12 mm radius about (0.034, 0), under the walk's second side."""
    return MOUND * np.clip(1.0 - np.hypot(x - 0.034, y) / 0.012, 0.0, None)


def _plate(cells=50, hole=None):
    """Vertices and triangles of the plate's top; ``hole`` (x0, x1, y0, y1) leaves the cells inside it out."""
    ticks = np.linspace(-HALF, HALF, cells + 1)
    x, y = np.meshgrid(ticks, ticks, indexing="ij")
    vertices = np.column_stack((x.ravel(), y.ravel(), _height(x, y).ravel()))
    triangles = []
    for i in range(cells):
        for j in range(cells):
            a, b, c, d = (
                i * (cells + 1) + j,
                (i + 1) * (cells + 1) + j,
                (i + 1) * (cells + 1) + j + 1,
                i * (cells + 1) + j + 1,
            )
            middle = vertices[[a, c], :2].mean(axis=0)
            if hole is not None and hole[0] < middle[0] < hole[1] and hole[2] < middle[1] < hole[3]:
                continue
            triangles.extend(((a, b, c), (a, c, d)))
    return vertices, np.array(triangles, dtype=np.int32)


def _lay(**options):
    vertices, triangles = _plate()
    settings = {"triangles": triangles, "stitch": (ENTRY, EXIT, NEEDLE), "turn_radius": TURN}
    settings.update(options)
    return loose_thread_coordinates(HEAD, vertices, length=LENGTH, elements=ELEMENTS, radius=RADIUS, **settings)


def _laid(q):
    """Index of the first node from which all lie on the pad: a thread radius above the plate's top under them.

    The laid path has a point every half element and the nodes lie on the lines between them: where such a
    line crosses an edge of the cone's facets a node is up to 0.05 mm off the surface. The thread comes down
    on the level plate, where a laid node is at its height to rounding.
    """
    above = np.abs(q[:, 2] - surface_top(q, *_plate()) - RADIUS)
    lead = int(np.flatnonzero(above >= 1.0e-4)[-1]) + 1
    return lead + int(np.flatnonzero(above[lead:] < 1.0e-9)[0])


class SurfaceTopTests(unittest.TestCase):
    def test_top_is_the_highest_surface_over_a_point_and_nothing_beside_the_surface(self):
        """Fails if a point gets a lower sheet of the surface, a height off its triangle, or a height where no surface is."""
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 2],
                [1, 0, 3],
                [0, 1, 2],
                [0.5, 0.5, -1],
                [0.5, 0.5, 9],
            ],
            dtype=float,
        )
        # A level square, a sloping triangle over half of it, and a triangle that stands on edge.
        triangles = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6], [0, 7, 8]])
        points = np.array([[0.25, 0.25], [0.75, 0.75], [0.5, 0.0], [1.5, 0.5], [0.0, 0.0]])
        top = surface_top(points, vertices, triangles)
        np.testing.assert_allclose(top[:3], [2.25, 0.0, 2.5], atol=1e-12)
        self.assertEqual(top[3], -np.inf)
        self.assertAlmostEqual(top[4], 2.0)
        # Positions with a height are read as (x, y).
        np.testing.assert_array_equal(surface_top(np.column_stack((points, np.full(5, 7.0))), vertices, triangles), top)


class BorderWalkTests(unittest.TestCase):
    LOW, HIGH = np.array([-0.03, -0.03]), np.array([0.03, 0.03])

    def test_walk_joins_the_nearest_side_that_is_not_before_its_start_and_follows_it_away(self):
        """Fails if the walk goes to a side before the start, to a farther side, or along the side towards the stitch."""
        away = np.array([0.6, 0.8])
        start = np.array([0.012, -0.02])
        points = border_walk(self.LOW, self.HIGH, start, away, TURN, 0.06)
        # The side at x = 0.03 is 18 mm off, the one at y = -0.03 is nearer but lies before the start.
        np.testing.assert_allclose(points[:3], [start, [0.03, -0.02], [0.03, 0.03]], atol=1e-15)
        np.testing.assert_allclose(points[3], [-0.03, 0.03], atol=1e-15)
        # With room both ways along the side the walk goes the way ``away`` points.
        for away_y, corner_y in ((0.8, 0.03), (-0.8, -0.03)):
            points = border_walk(self.LOW, self.HIGH, np.array([0.012, 0.0]), np.array([0.6, away_y]), TURN, 0.03)
            np.testing.assert_allclose(points[:3], [[0.012, 0.0], [0.03, 0.0], [0.03, corner_y]], atol=1e-15)
        # Straight away from a side the walk takes the longer way along it.
        points = border_walk(self.LOW, self.HIGH, np.array([0.005, 0.01]), np.array([0.0, 1.0]), TURN, 0.03)
        np.testing.assert_allclose(points[:3], [[0.005, 0.01], [0.005, 0.03], [-0.03, 0.03]], atol=1e-15)
        # A corner nearer than two turn radii leaves no room for both arcs: the longer way again.
        points = border_walk(self.LOW, self.HIGH, np.array([0.016, 0.012]), np.array([0.8, 0.6]), TURN, 0.03)
        np.testing.assert_allclose(points[:3], [[0.016, 0.012], [0.03, 0.012], [0.03, -0.03]], atol=1e-15)

    def test_start_nearer_to_its_side_than_a_turn_radius_is_put_on_it(self):
        """Fails if a leg too short for the arc of its corner is kept."""
        points = border_walk(self.LOW, self.HIGH, np.array([0.025, 0.0]), np.array([1.0, 0.0]), TURN, 0.02)
        np.testing.assert_allclose(points[:2], [[0.03, 0.0], [0.03, 0.03]], atol=1e-15)

    def test_walk_longer_than_the_box_goes_round_again_two_turn_radii_inside(self):
        """Fails if a second round runs over the first or the walk stops short of its length."""
        points = border_walk(self.LOW, self.HIGH, np.array([0.03, -0.03]), np.array([0.0, 1.0]), 0.005, 0.30)
        expected = [
            [0.03, -0.03],
            [0.03, 0.03],
            [-0.03, 0.03],
            [-0.03, -0.03],
            [0.02, -0.03],
            [0.02, 0.02],
            [-0.02, 0.02],
        ]
        np.testing.assert_allclose(points[: len(expected)], expected, atol=1e-15)
        self.assertGreaterEqual(np.linalg.norm(np.diff(points, axis=0), axis=1).sum(), 0.30)

    def test_walk_that_runs_out_of_box_is_refused(self):
        """Fails if a thread too long for the pad is laid over itself."""
        # Sides of 60, 60, 60, 40 and 40 mm; the sixth would be 20 mm, two turn radii: no straight piece left.
        points = border_walk(self.LOW, self.HIGH, np.array([0.03, -0.03]), np.array([0.0, 1.0]), TURN, 0.25)
        self.assertEqual(len(points), 6)
        with self.assertRaisesRegex(ValueError, "do not fit"):
            border_walk(self.LOW, self.HIGH, np.array([0.03, -0.03]), np.array([0.0, 1.0]), TURN, 0.27)


class RoundedTests(unittest.TestCase):
    def test_corners_are_arcs_of_the_radius_that_leave_and_meet_the_legs_tangentially(self):
        """Fails if a corner stays sharp, an arc has another radius or bulges to the wrong side, or points are far apart."""
        corners = np.array([[0.0, 0.0], [0.04, 0.0], [0.04, 0.05], [-0.01, 0.05]])
        step = 0.0005
        path = rounded(corners, TURN, step)
        np.testing.assert_allclose(path[[0, -1]], corners[[0, -1]], rtol=0.0, atol=1e-15)
        pieces = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.assertGreater(pieces.min(), 0.0)
        self.assertLessEqual(pieces.max(), step * (1.0 + 1e-9))
        # Two quarter turns to the left: each shortens the walk by (2 - pi / 2) radii.
        # (The arcs are drawn in straight pieces: 3 um short of that.)
        self.assertAlmostEqual(pieces.sum(), 0.14 - 2.0 * (2.0 - 0.5 * math.pi) * TURN, delta=1e-5)
        heading = np.unwrap(np.arctan2(*np.diff(path, axis=0).T[::-1]))
        self.assertTrue(np.all(np.diff(heading) > -1e-9))
        # No piece turns more than its length over the radius allows.
        self.assertLessEqual(np.diff(heading).max(), step / TURN * (1.0 + 1e-6))
        for center in ([0.03, 0.01], [0.03, 0.04]):
            near = np.linalg.norm(path - center, axis=1)
            self.assertAlmostEqual(near.min(), TURN, delta=1e-9)

    def test_corner_without_room_for_its_arc_is_left_as_it_is(self):
        """Fails if an arc is drawn past the ends of its legs."""
        corners = np.array([[0.0, 0.0], [0.005, 0.0], [0.005, 0.05]])
        path = rounded(corners, TURN, 0.001)
        self.assertTrue(np.any(np.all(np.isclose(path, corners[1], atol=1e-15), axis=1)))
        self.assertLessEqual(path[:, 0].max(), 0.005 + 1e-15)


class LooseThreadTests(unittest.TestCase):
    def test_chain_starts_at_the_tail_with_whole_elements_and_unit_tangents(self):
        """Fails if the head is moved, an element is longer than its rest length or a tangent is not a unit vector."""
        q = _lay()
        self.assertEqual(q.shape, (ELEMENTS + 1, 6))
        np.testing.assert_array_equal(q[0], HEAD)
        self.assertTrue(np.isfinite(q).all())
        np.testing.assert_allclose(np.linalg.norm(q[:, 3:], axis=1), 1.0, atol=1e-12)
        chords = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1)
        self.assertLessEqual(chords.max(), ELEMENT * (1.0 + 1e-9))

    def test_no_bend_on_the_pad_is_tighter_than_the_turn_radius(self):
        """Fails if a corner of the walk is sharper than the radius the thread's weight can hold flat."""
        q = _lay()
        laid = _laid(q)
        a, b, c = q[laid:-2, :2], q[laid + 1 : -1, :2], q[laid + 2 :, :2]
        sides = np.linalg.norm(a - b, axis=1) * np.linalg.norm(b - c, axis=1) * np.linalg.norm(c - a, axis=1)
        area = 0.5 * np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
        bent = area > 1e-12
        # The circle through three neighbouring nodes. The thread reaches round two corners of this plate.
        radii = sides[bent] / (4.0 * area[bent])
        self.assertGreaterEqual(bent.sum(), 16)
        # The nodes lie on the chords of an arc drawn in pieces of half an element: 0.12 % inside it.
        self.assertGreaterEqual(radii.min(), TURN * (1.0 - 2.0e-3))
        self.assertLess(np.median(radii), TURN * (1.0 + 2.0e-3))

    def test_laid_nodes_lie_a_thread_radius_above_the_surface_under_them_also_on_the_mound(self):
        """Fails if the thread is laid level with the pad's highest point, as the bounding box would have it."""
        q = _lay()
        laid = _laid(q)
        self.assertLess(laid, 30)
        vertices, triangles = _plate()
        height = surface_top(q[laid:], vertices, triangles)
        level = height == 0.0
        np.testing.assert_allclose(q[laid:, 2][level], RADIUS, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(q[laid:, 2], height + RADIUS, rtol=0.0, atol=1e-4)
        # The walk's second side passes 6 mm from the cone's axis: nodes on the plate and up to 2 mm above it.
        self.assertEqual(height.min(), 0.0)
        self.assertAlmostEqual(height.max(), 0.5 * MOUND, delta=1e-4)
        np.testing.assert_allclose(height, _height(q[laid:, 0], q[laid:, 1]), rtol=0.0, atol=1e-4)
        # Without the surface the pad is its bounding box: everything is level with the top of the mound.
        boxed = loose_thread_coordinates(HEAD, vertices, stitch=(ENTRY, EXIT, NEEDLE), turn_radius=TURN)
        np.testing.assert_allclose(boxed[40:, 2], vertices[:, 2].max() + RADIUS, atol=1e-12)

    def test_thread_comes_down_two_needle_radii_behind_the_entry_and_walks_the_border_away_from_the_stitch(self):
        """Fails if the first laid point is beside or before the stitch, or the walk comes nearer the box than a turn radius."""
        q = _lay()
        laid = _laid(q)
        behind = ENTRY[:2] + 2.0 * NEEDLE * np.array([0.0, 1.0])
        # The node count of the lead is whole: the first laid node is within an element of that point, on the leg.
        self.assertLess(np.linalg.norm(q[laid, :2] - behind), ELEMENT)
        self.assertAlmostEqual(q[laid, 0], behind[0], delta=1e-9)
        # The side behind the stitch is the nearest that is not before that point; then round the box.
        self.assertTrue(np.all(np.abs(q[laid:, :2]) <= HALF - TURN + 1e-9))
        on_side = np.isclose(q[:, 1], HALF - TURN, atol=1e-9)
        self.assertGreaterEqual(on_side.sum(), 8)
        first = int(np.flatnonzero(on_side)[0])
        self.assertTrue(np.all(q[laid:first, 1] >= behind[1] - 1e-9))
        self.assertTrue(np.all(np.diff(q[laid:first, 1]) > 0.0))

    def test_without_a_stitch_the_thread_trails_the_way_the_tail_points(self):
        """Fails if a thread with no planned stitch is laid against its tail's tangent."""
        vertices, triangles = _plate()
        head = np.array([0.0, 0.0, 0.01, 0.6, 0.0, 0.8])
        q = loose_thread_coordinates(head, vertices, triangles=triangles, turn_radius=TURN)
        laid = _laid(q)
        # One turn radius beyond the tail, then on to the side at x = +40 mm.
        self.assertLess(np.linalg.norm(q[laid, :2] - [TURN, 0.0]), ELEMENT)
        self.assertTrue(np.all(np.diff(q[laid : laid + 5, 0]) > 0.0))
        self.assertAlmostEqual(q[:, 0].max(), HALF - TURN, delta=1e-9)
        # A tail that points straight up trails along x.
        up = loose_thread_coordinates(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 1.0]), vertices, triangles=triangles)
        self.assertAlmostEqual(up[:, 0].max(), HALF - 0.01, delta=1e-9)

    def test_lead_keeps_a_thread_radius_clear_of_the_capsules_where_a_reach_does(self):
        """Fails if the lead is laid through a capsule that another reach clears, or changed by a capsule out of its way."""
        free = _lay()
        laid = _laid(free)
        # A capsule of 1 mm radius across the free lead, half way down, its axis 0.6 mm beside the lead.
        tangent = free[laid // 2 + 1, :3] - free[laid // 2 - 1, :3]
        beside = np.cross(tangent, [1.0, 0.0, 0.0])
        middle = free[laid // 2, :3] + 0.0006 * beside / np.linalg.norm(beside)
        capsule = (middle[None] - [[0.004, 0.0, 0.0]], middle[None] + [[0.004, 0.0, 0.0]], np.array([0.001]))

        def gap(q):
            nodes = q[3 : _laid(q), :3]
            along = np.clip((nodes[:, 0] - capsule[0][0, 0]) / 0.008, 0.0, 1.0)
            nearest = capsule[0] + along[:, None] * np.array([0.008, 0.0, 0.0])
            return np.linalg.norm(nodes - nearest, axis=1).min() - 0.001 - RADIUS

        # The free lead passes the axis at 0.6 mm: inside the capsule, though not through its axis.
        self.assertLess(gap(free), -0.0004)
        self.assertGreater(gap(free), -0.0007)
        blocked = _lay(obstacles=capsule)
        self.assertGreater(gap(blocked), RADIUS)
        np.testing.assert_array_equal(blocked[0], HEAD)
        np.testing.assert_array_equal(blocked[-1, 2], free[-1, 2])
        # A capsule that touches nothing changes nothing; none at all is the same as an empty set.
        far = (np.array([[0.04, -0.04, 0.02]]), np.array([[0.045, -0.04, 0.02]]), np.array([0.001]))
        np.testing.assert_array_equal(_lay(obstacles=far), free)
        np.testing.assert_array_equal(_lay(obstacles=(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))), free)

    def test_lead_bends_the_thread_least_and_is_the_shortest_of_equals(self):
        """Fails if a tail that points away from the pad's first point gets the tight lead, or one that points at it a long one."""
        # The scene: the tail points to the exit and the thread must turn back over it. The widest reach bends least.
        turned = _lay()
        chords = np.linalg.norm(np.diff(turned[: _laid(turned), :3], axis=0), axis=1) / ELEMENT
        self.assertGreater(chords.min(), 0.9)
        # A tail that points where the thread goes: the two shorter reaches bend the thread by less than a
        # hundredth of an element's chord (the widest folds over), and the shorter of the two is taken.
        vertices, triangles = _plate()
        first, along = np.array([0.0, 0.008, RADIUS]), np.array([0.0, 1.0, 0.0])
        start = np.array([0.0, -0.03, 0.006])
        head = np.r_[start, (first - start) / np.linalg.norm(first - start)]
        ahead = loose_thread_coordinates(head, vertices, triangles=triangles, stitch=(ENTRY, EXIT, NEEDLE))
        leads, bends = [], []
        for part in (1 / 3, 2 / 3, 1):
            curve = bridge(head[:3] + ELEMENT * head[3:], head[3:], first, along, ELEMENT / 2, part)
            leads.append(lay_along_path(np.vstack((head[:3], curve, first + 0.02 * along)), 21, ELEMENT)[:, :3])
            bends.append(round(float(np.linalg.norm(np.diff(leads[-1], axis=0), axis=1).min() / ELEMENT), 2))
        self.assertEqual(bends[:2], [1.0, 1.0])
        self.assertLess(bends[2], 0.9)
        self.assertGreater(np.abs(leads[1][:19] - leads[0][:19]).max(), 5e-5)
        np.testing.assert_allclose(ahead[:19, :3], leads[0][:19], rtol=0.0, atol=1e-12)
        self.assertLess(_laid(ahead), _laid(turned))

    def test_thread_longer_than_one_round_of_a_small_pad_goes_round_again_inside(self):
        """Fails if a pad that holds the thread in two rounds is refused, or the second round lies on the first."""
        vertices = np.array([[-0.04, -0.04, 0.0], [0.04, 0.04, 0.0]])
        head = np.array([0.0, -0.02, 0.01, 0.0, -1.0, 0.0])
        q = loose_thread_coordinates(head, vertices, length=0.21, elements=105, turn_radius=TURN)
        laid = int(np.flatnonzero(np.abs(q[:, 2] - RADIUS) > 1e-9)[-1]) + 1
        # Down to the side at y = -30 mm, round the 60 mm box, and the fifth side 20 mm inside the first.
        self.assertAlmostEqual(np.abs(q[laid:, :2]).max(), 0.03, delta=1e-9)
        np.testing.assert_allclose(q[-4:, 1], -0.01, rtol=0.0, atol=1e-9)
        self.assertTrue(np.all(np.diff(q[-4:, 0]) > 0.0))
        self.assertLess(q[-1, 0], 0.01)
        # Laid nodes more than three quarters of a turn apart along the thread stay a turn radius apart.
        apart = np.linalg.norm(q[laid:, None, :2] - q[None, laid:, :2], axis=2)
        index = np.arange(len(apart))
        far = np.abs(index[:, None] - index[None]) > 1.5 * math.pi * TURN / ELEMENT
        self.assertGreater(apart[far].min(), TURN)
        # The sixth side would have 20 mm between its corners, no room for two arcs: 223 mm is all this pad
        # holds in turns of 10 mm. A longer thread is laid in turns of 5 mm.
        longer = loose_thread_coordinates(head, vertices, length=0.26, elements=130, turn_radius=TURN)
        self.assertAlmostEqual(np.abs(longer[20:, :2]).max(), 0.035, delta=1e-9)

    def test_walk_over_a_gap_in_the_pad_lies_level_with_the_top_of_its_box(self):
        """Fails if a pad of two blocks cannot be laid on, or the nodes over the gap get a height from nowhere."""
        hole = (0.03, 0.05, 0.01, 0.03)
        vertices, triangles = _plate(hole=hole)
        q = loose_thread_coordinates(
            HEAD, vertices, triangles=triangles, stitch=(ENTRY, EXIT, NEEDLE), turn_radius=TURN
        )
        # The walk is the one of the whole plate: its second side crosses the hole at x = 40 mm.
        self.assertAlmostEqual(q[:, 0].max(), HALF - TURN, delta=1e-9)
        # How far a node is inside the hole (negative: outside it); the laid path has a point every millimetre.
        depth = np.minimum.reduce([q[:, 0] - hole[0], hole[1] - q[:, 0], q[:, 1] - hole[2], hole[3] - q[:, 1]])
        over_gap = depth > 0.0015
        self.assertGreaterEqual(over_gap.sum(), 6)
        np.testing.assert_allclose(q[over_gap, 2], vertices[:, 2].max() + RADIUS, rtol=0.0, atol=1e-12)
        on = (depth < -0.0015) & (np.arange(len(q)) >= _laid(_lay()))
        np.testing.assert_allclose(q[on, 2], surface_top(q[on], vertices, triangles) + RADIUS, rtol=0.0, atol=1e-4)

    def test_pad_too_small_for_the_turn_radius_gets_half_or_a_quarter_of_it(self):
        """Fails if a pad that holds the thread in tighter turns is refused, or tighter turns are taken than it needs."""
        head = np.array([0.0, -0.005, 0.01, 0.0, -1.0, 0.0])
        stitch = (np.array([0.0, 0.0, 0.0]), np.array([0.0, -0.005, 0.0]), NEEDLE)

        def corner_radius(vertices):
            q = loose_thread_coordinates(head, vertices, stitch=stitch, turn_radius=TURN)
            laid = int(np.flatnonzero(np.abs(q[:, 2] - RADIUS) > 1e-9)[-1]) + 1
            a, b, c = q[laid:-2, :2], q[laid + 1 : -1, :2], q[laid + 2 :, :2]
            area = 0.5 * np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
            sides = np.linalg.norm(a - b, axis=1) * np.linalg.norm(b - c, axis=1) * np.linalg.norm(c - a, axis=1)
            inside = np.abs(q[laid:, :2]).max(axis=0)
            # The tightest circle through three neighbouring laid nodes: a corner of the walk.
            return float((sides[area > 1e-12] / (4.0 * area[area > 1e-12])).min()), inside

        # (An arc of 5 mm drawn in pieces of 1 mm: its nodes lie half a percent inside it.)
        # 160 mm by 40 mm: the box inside one turn radius is 20 mm wide, no room for two corners; half fits.
        radius, inside = corner_radius(np.array([[-0.08, -0.02, 0.0], [0.08, 0.02, 0.0]]))
        self.assertAlmostEqual(radius, 0.5 * TURN, delta=1e-2 * radius)
        np.testing.assert_allclose(inside, [0.08 - 0.5 * TURN, 0.02 - 0.5 * TURN], rtol=0.0, atol=1e-9)
        # 60 mm square: one round with the full radius holds 170 mm of walk, with half of it two rounds hold the thread.
        radius, inside = corner_radius(np.array([[-0.03, -0.03, 0.0], [0.03, 0.03, 0.0]]))
        self.assertAlmostEqual(radius, 0.5 * TURN, delta=1e-2 * radius)
        # 100 mm square: the full radius.
        radius, inside = corner_radius(np.array([[-0.05, -0.05, 0.0], [0.05, 0.05, 0.0]]))
        self.assertAlmostEqual(radius, TURN, delta=1e-2 * radius)

    def test_small_pads_and_bad_arguments_are_refused(self):
        """Fails if a pad without room for the thread in any turn, a head that is no unit tangent or a bad size is accepted."""
        vertices, triangles = _plate()
        with self.assertRaisesRegex(ValueError, "do not fit"):
            loose_thread_coordinates(HEAD, vertices * 0.09, triangles=triangles, turn_radius=TURN)
        for options in ({"turn_radius": 0.0}, {"length": -1.0}, {"radius": 0.0}, {"elements": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                loose_thread_coordinates(HEAD, vertices, triangles=triangles, **options)
        with self.assertRaises(ValueError):
            loose_thread_coordinates(HEAD * 2.0, vertices, triangles=triangles)


if __name__ == "__main__":
    unittest.main()
