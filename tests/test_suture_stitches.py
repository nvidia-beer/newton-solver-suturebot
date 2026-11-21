# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stitch planning helpers: thread paths, the needle's schedule, stitches on the bundled pad and one headless run."""

import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull

from newton_solver_suturebot import NeedleGeometry
from newton_solver_suturebot.examples.needle import example_fem_needle_suture as suture
from newton_solver_suturebot.examples.needle._needle_insertion_geometry import InsertionPlan, MaterialPoint, TetLocator
from newton_solver_suturebot.examples.needle._suture_stitches import (
    NeedleSchedule,
    Stitch,
    _Lift,
    _Turn,
    plan_stitches,
    wound_axis,
)
from newton_solver_suturebot.examples.needle._thread_pins import bridge, lay_along_path, serpentine

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")

# Sums and products of a few float64 lengths of some centimetres.
ROUNDING = 1.0e-12  # [m]


def _lengths(points):
    """Lengths of the segments of a polyline [m]."""
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def _on_path(path, points):
    """Distance from the polyline ``path`` and distance along it of each of ``points`` [m]."""
    starts = np.concatenate(([0.0], np.cumsum(_lengths(path))))
    found = []
    for point in points:
        nearest = (np.inf, 0.0)
        for start, a, b in zip(starts[:-1], path[:-1], path[1:], strict=True):
            length = np.linalg.norm(b - a)
            along = np.clip((point - a) @ (b - a) / length, 0.0, length)
            nearest = min(nearest, (np.linalg.norm(a + along * (b - a) / length - point), start + along))
        found.append(nearest)
    return np.array(found).T


class LayAlongPathTests(unittest.TestCase):
    # A path in space whose corners lie between the nodes of a chain with 2 mm elements.
    path = np.array(
        [[0.0, 0.0, 0.0], [0.0071, 0.0, 0.0], [0.0071, 0.0052, 0.001], [0.001, 0.0052, 0.004], [0.001, -0.002, 0.004]]
    )
    spacing = 0.002

    def test_nodes_are_one_spacing_apart_along_the_path(self):
        """Fails if a node is off the path or not at its multiple of the spacing along it."""
        count = int(_lengths(self.path).sum() / self.spacing) + 1
        self.assertGreaterEqual(count, 12)
        nodes = lay_along_path(self.path, count, self.spacing)
        self.assertEqual(nodes.shape, (count, 6))
        off, stations = _on_path(self.path, nodes[:, :3])
        np.testing.assert_allclose(off, 0.0, rtol=0.0, atol=ROUNDING)
        np.testing.assert_allclose(stations, self.spacing * np.arange(count), rtol=0.0, atol=ROUNDING)
        np.testing.assert_array_equal(nodes[0, :3], self.path[0])

    def test_tangents_are_the_unit_directions_of_the_path(self):
        """Fails if a node's slope is not the unit direction of the segment it lies on."""
        count = int(_lengths(self.path).sum() / self.spacing) + 1
        nodes = lay_along_path(self.path, count, self.spacing)
        starts = np.concatenate(([0.0], np.cumsum(_lengths(self.path))))
        segment = np.searchsorted(starts, self.spacing * np.arange(count), side="right") - 1
        self.assertEqual(sorted(set(segment.tolist())), [0, 1, 2, 3])  # every segment holds a node
        directions = np.diff(self.path, axis=0) / _lengths(self.path)[:, None]
        np.testing.assert_allclose(nodes[:, 3:], directions[segment], rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(np.linalg.norm(nodes[:, 3:], axis=1), 1.0, rtol=0.0, atol=1.0e-12)

    def test_node_on_a_corner_takes_the_bisector(self):
        """Fails if a node that lies on a corner of the path does not take the unit bisector of its two directions."""
        # Legs of three elements; the corners' distances are the same float products as the nodes' stations.
        leg = 3 * self.spacing
        path = np.array([[0.0, 0.0, 0.0], [leg, 0.0, 0.0], [leg, leg, 0.0], [0.0, leg, 0.0]])
        nodes = lay_along_path(path, 10, self.spacing)
        diagonal = math.sqrt(0.5)
        np.testing.assert_allclose(nodes[3], (leg, 0.0, 0.0, diagonal, diagonal, 0.0), rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(nodes[6], (leg, leg, 0.0, -diagonal, diagonal, 0.0), rtol=0.0, atol=1.0e-12)
        # The first and the last node take the direction of their only segment.
        np.testing.assert_allclose(nodes[0], (0.0, 0.0, 0.0, 1.0, 0.0, 0.0), rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(nodes[9], (0.0, leg, 0.0, -1.0, 0.0, 0.0), rtol=0.0, atol=1.0e-12)

    # Defect: 5 * 0.0003 rounds to just below 0.0015, so the node is taken as the end of the first segment and
    # gets that segment's direction; 17 % of random corners at a node's station round this way.
    @unittest.expectedFailure
    def test_node_on_a_corner_takes_the_bisector_whatever_the_rounding(self):
        """Fails while the rounding of a node's station decides whether a node on a corner gets the bisector."""
        path = np.array([[0.0, 0.0, 0.0], [0.0015, 0.0, 0.0], [0.0015, 0.0015, 0.0]])
        nodes = lay_along_path(path, 11, 0.0003)
        np.testing.assert_allclose(nodes[5, :3], (0.0015, 0.0, 0.0), rtol=0.0, atol=ROUNDING)
        diagonal = math.sqrt(0.5)
        np.testing.assert_allclose(nodes[5, 3:], (diagonal, diagonal, 0.0), rtol=0.0, atol=1.0e-12)

    def test_path_must_be_as_long_as_the_chain(self):
        """Fails if a chain longer than its path is laid anyway, or one that just fits is refused."""
        leg = 3 * self.spacing
        path = np.array([[0.0, 0.0, 0.0], [leg, 0.0, 0.0], [leg, leg, 0.0], [0.0, leg, 0.0]])
        nodes = lay_along_path(path, 10, self.spacing)  # 9 elements on 18 mm
        np.testing.assert_allclose(nodes[-1, :3], path[-1], rtol=0.0, atol=ROUNDING)
        with self.assertRaisesRegex(ValueError, r"the path is 0\.0180 m long; the chain needs 0\.0200 m"):
            lay_along_path(path, 11, self.spacing)

    def test_repeated_path_points_are_refused(self):
        """Fails if a path with two equal neighbouring points, which has no direction there, is accepted."""
        with self.assertRaisesRegex(ValueError, "distinct"):
            lay_along_path(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]), 3, self.spacing)


class SerpentineTests(unittest.TestCase):
    start = np.array([0.01, 0.02, 0.003])
    along = np.array([0.6, 0.8, 0.0])
    across = np.array([-0.8, 0.6, 0.0])
    row_length, turn_radius, step = 0.03, 0.004, 0.001
    # A row and its turn are 42.5 mm long: four of them fall 2 mm short of this length, so a fifth is needed.
    length = 0.172

    def setUp(self):
        self.points = serpentine(
            self.start, self.along, self.across, self.row_length, self.turn_radius, self.length, self.step
        )
        offsets = self.points - self.start
        self.u, self.v = offsets @ self.along, offsets @ self.across
        # A row is the only segment longer than the step.
        self.rows = np.flatnonzero(_lengths(self.points) > 1.5 * self.step)

    def test_length_is_reached_with_less_than_one_row_and_turn_to_spare(self):
        """Fails if the polyline is shorter than asked, or goes on after it has reached the length."""
        np.testing.assert_array_equal(self.points[0], self.start)
        total = _lengths(self.points).sum()
        self.assertGreaterEqual(total, self.length)
        self.assertLess(total, self.length + self.row_length + math.pi * self.turn_radius)
        # It stays in the plane of the two directions.
        np.testing.assert_allclose((self.points - self.start)[:, 2], 0.0, rtol=0.0, atol=ROUNDING)

    def test_rows_run_back_and_forth_two_turn_radii_apart(self):
        """Fails if a row is not straight along the row direction, not opposite to its neighbour or not 2 r from it."""
        self.assertEqual(len(self.rows), 5)
        for number, first in enumerate(self.rows):
            begin, end = (0.0, self.row_length) if number % 2 == 0 else (self.row_length, 0.0)
            np.testing.assert_allclose(self.u[[first, first + 1]], (begin, end), rtol=0.0, atol=ROUNDING)
            np.testing.assert_allclose(
                self.v[[first, first + 1]], 2.0 * self.turn_radius * number, rtol=0.0, atol=ROUNDING
            )

    def test_turns_are_half_circles_of_the_turn_radius(self):
        """Fails if the points between two rows leave the half circle that joins them or are more than a step apart."""
        ends = [*self.rows[1:], len(self.points) - 1]
        for number, (first, last) in enumerate(zip(self.rows, ends, strict=True)):
            turn = self.points[first + 1 : last + 1]  # from the row's end to the next row's start
            center = self.points[first + 1] + self.turn_radius * self.across
            offsets = turn - center
            np.testing.assert_allclose(np.linalg.norm(offsets, axis=1), self.turn_radius, rtol=0.0, atol=ROUNDING)
            # The turn bulges beyond the end of its row: forwards after an even row, backwards after an odd one.
            forwards = 1.0 if number % 2 == 0 else -1.0
            self.assertGreaterEqual((forwards * offsets @ self.along).min(), -ROUNDING)
            # From the row's end, at angle 0, in equal steps to the next row's start, at angle pi. The arc
            # cosine loses half the digits at its two ends, hence 1e-6 rad.
            angles = np.arccos(np.clip(-offsets @ self.across / self.turn_radius, -1.0, 1.0))
            self.assertAlmostEqual(angles[0], 0.0, delta=1.0e-6)
            self.assertAlmostEqual(angles[-1], math.pi, delta=1.0e-6)
            steps = np.diff(angles)
            np.testing.assert_allclose(steps, math.pi / (len(turn) - 1), rtol=0.0, atol=1.0e-6)
            self.assertLessEqual(_lengths(turn).max(), self.step)


class BridgeTests(unittest.TestCase):
    start = np.array([0.0, 0.0, 0.01])
    start_direction = np.array([0.0, 0.6, -0.8])
    end = np.array([0.02, 0.01, 0.0])
    end_direction = np.array([1.0, 0.0, 0.0])
    step = 0.00025

    def test_curve_joins_the_end_points_along_the_end_directions(self):
        """Fails if the curve does not start and end at the given points along the given directions."""
        points = bridge(self.start, self.start_direction, self.end, self.end_direction, self.step)
        np.testing.assert_array_equal(points[0], self.start)
        np.testing.assert_allclose(points[-1], self.end, rtol=0.0, atol=ROUNDING)
        chords = np.diff(points, axis=0)
        first = chords[0] / np.linalg.norm(chords[0])
        last = chords[-1] / np.linalg.norm(chords[-1])
        # The first chord of a cubic Bezier sampled at the parameter step h is 3 h (P1 - P0) + 3 h² (P2 - 2 P1
        # + P0) to second order. With the handles a third of the chord c, |P1 - P0| = c / 3 and |P2 - 2 P1 + P0|
        # is at most 2 c, so the chord turns from the end direction by at most 6 h. Measured: 0.70° and 0.36°
        # against a bound of 2.86°.
        bound = 6.0 / (len(points) - 1)
        self.assertLess(math.acos(min(1.0, first @ self.start_direction)), bound)
        self.assertLess(math.acos(min(1.0, last @ self.end_direction)), bound)
        # At a quarter of the step both angles halve: the curve itself leaves along the directions.
        finer = bridge(self.start, self.start_direction, self.end, self.end_direction, 0.25 * self.step)
        chord = finer[1] - finer[0]
        self.assertLess(
            math.acos(min(1.0, chord @ self.start_direction / np.linalg.norm(chord))),
            0.3 * math.acos(min(1.0, first @ self.start_direction)),
        )

    def test_neighbouring_points_are_about_a_step_apart(self):
        """Fails if the curve is sampled so coarsely that neighbouring points are farther apart than its bound."""
        points = bridge(self.start, self.start_direction, self.end, self.end_direction, self.step)
        handle = np.linalg.norm(self.end - self.start) / 3.0
        legs = np.array(
            [
                handle,
                np.linalg.norm((self.end - handle * self.end_direction) - (self.start + handle * self.start_direction)),
                handle,
            ]
        )
        # A cubic Bezier is never faster than three times its longest control leg, and the curve has at least
        # as many intervals as its control polygon has steps. Measured: 0.80 to 0.96 steps.
        self.assertLessEqual(_lengths(points).max(), 3.0 * legs.max() / legs.sum() * self.step)
        self.assertGreater(_lengths(points).min(), 0.5 * self.step)

    def test_reach_sets_how_wide_the_curve_turns(self):
        """Fails if the inner control points are not the given fraction of the chord from the ends."""
        chord = 0.01
        for reach in (1.0 / 3.0, 2.0 / 3.0, 1.0):
            with self.subTest(reach=reach):
                # A turn back: out along +x and in along -x, the ends a chord apart along y. The curve is
                # x = 3 reach chord t (1 - t), widest at t = 1/2 with 3/4 reach chord. The samples miss the
                # widest point by at most half a parameter step, which costs 3 reach chord h² / 4.
                points = bridge(np.zeros(3), (1.0, 0.0, 0.0), (0.0, chord, 0.0), (-1.0, 0.0, 0.0), 1.0e-4, reach)
                widest = 0.75 * reach * chord
                self.assertLessEqual(points[:, 0].max(), widest + ROUNDING)
                self.assertGreaterEqual(points[:, 0].max(), widest * (1.0 - 1.0 / (len(points) - 1) ** 2) - ROUNDING)


def _stitch(geometry, entry, exit_point):
    """A stitch between two points of a level surface, as plan_stitch makes it, without a pad to check it on."""
    up = np.array([0.0, 0.0, 1.0])
    ends = [
        MaterialPoint(1, np.zeros(3, dtype=np.int32), np.full(3, 1.0 / 3.0), np.asarray(point, dtype=float), up)
        for point in (entry, exit_point)
    ]
    plan = InsertionPlan.create(geometry, *ends)
    # The tail leaves the tissue one needle span after the plan's finish angle.
    return Stitch(
        plan, np.zeros(0), np.zeros(0, dtype=np.int32), np.zeros((0, 4)), plan.finish_angle + geometry.span, ()
    )


def _straight_above(pose, end):
    """Whether ``pose`` is the pose ``end`` moved straight up or down: the same turn at the same place on the map."""
    turned = min(np.abs(pose[3:] - end[3:]).max(), np.abs(pose[3:] + end[3:]).max())  # either sign is the same turn
    return turned < ROUNDING and np.linalg.norm(pose[:2] - end[:2]) < ROUNDING


def _needle(geometry, pose, points=None):
    """Points of the needle at a body pose [m]; by default every 0.2 mm along its centreline."""
    x, y, z, w = pose[3:]
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )
    points = geometry.sample_centerline(2.0e-4) if points is None else points
    return points @ rotation.T + pose[:3]


class NeedleScheduleTests(unittest.TestCase):
    pass_speed, air_speed = 0.004, 0.012  # [m/s], the defaults of fem_needle_suture
    # A plan's pose carries a float32 quaternion, good to 2e-7 rad: 4e-9 m at the needle's 16 mm, on each of two
    # poses that are compared. Measured: up to 5e-9 m.
    pose_rounding = 5.0e-8  # [m]

    @classmethod
    def setUpClass(cls):
        cls.geometry = NeedleGeometry.load()
        # Three copies of a stitch, 6 mm apart along the wound, as plan_stitches makes a row.
        pitch = np.array([0.006, 0.0, 0.0])
        cls.row = [
            _stitch(cls.geometry, (0.03, -0.040, 0.0) + number * pitch, (0.03, -0.048, 0.0) + number * pitch)
            for number in range(3)
        ]
        # Stitches that point every way, at different heights, as they are picked on the map.
        cls.picked = [
            _stitch(cls.geometry, (0.030, -0.040, 0.000), (0.030, -0.048, 0.000)),
            _stitch(cls.geometry, (0.020, -0.030, 0.002), (0.027, -0.027, 0.003)),
            _stitch(cls.geometry, (0.050, -0.050, 0.001), (0.050, -0.043, 0.000)),
            _stitch(cls.geometry, (0.040, -0.020, 0.012), (0.033, -0.020, 0.012)),
        ]
        # The pad's highest point plus the needle's thickness, as the example sets it. The last stitch lies
        # above it, so its lift has nothing to rise.
        cls.clear_height = 0.004 + cls.geometry.thickness
        cls.turning = NeedleSchedule(cls.row, cls.pass_speed, cls.air_speed)
        cls.lifting = NeedleSchedule(cls.picked, cls.pass_speed, cls.air_speed, cls.clear_height)

    def _gap(self, pose, other):
        """Largest distance between the same points of the needle at two poses [m]."""
        return np.linalg.norm(_needle(self.geometry, pose) - _needle(self.geometry, other), axis=1).max()

    @staticmethod
    def _boundaries(schedule):
        """Times at which a move of the schedule ends and the next begins [s]: stitch, travel, stitch and so on."""
        return schedule._starts[1:-1]

    def test_stitch_takes_the_tip_from_its_approach_angle_to_where_the_tail_has_left(self):
        """Fails if a stitch does not start at the plan's first pose, end at its leave angle or stay on its circle."""
        for schedule, stitches in ((self.turning, self.row), (self.lifting, self.picked)):
            starts = np.concatenate(([0.0], self._boundaries(schedule), [schedule.duration]))
            for number, stitch in enumerate(stitches):
                begin, end = starts[2 * number], starts[2 * number + 1]
                pose, index, angle = schedule.at(begin)
                self.assertEqual(index, number)
                self.assertEqual(angle, stitch.plan.initial_angle)
                self.assertLess(self._gap(pose, stitch.plan.pose(stitch.plan.initial_angle)), self.pose_rounding)
                angles = []
                for time in np.linspace(begin, end, 41)[1:-1]:
                    pose, index, angle = schedule.at(time)
                    self.assertEqual(index, number)
                    self.assertLess(self._gap(pose, stitch.plan.pose(angle)), self.pose_rounding)
                    angles.append(angle)
                self.assertTrue(np.all(np.diff(angles) > 0.0))
                pose, index, angle = schedule.at(end - 1.0e-9)
                self.assertEqual(index, number)
                self.assertAlmostEqual(angle, stitch.leave_angle, delta=1.0e-9)
                self.assertLess(self._gap(pose, stitch.plan.pose(stitch.leave_angle)), self.pose_rounding)

    def test_pose_is_continuous_where_a_stitch_and_a_travel_meet(self):
        """Fails if the needle jumps at the end of a stitch or of the travel after it."""
        for schedule, stitches in ((self.turning, self.row), (self.lifting, self.picked)):
            boundaries = self._boundaries(schedule)
            self.assertEqual(len(boundaries), 2 * (len(stitches) - 1))
            for boundary in boundaries:
                # Every move starts and stops smoothly over 0.25 s: 0.1 ms from its end it has less than
                # air speed * 0.25 s * (0.0004)³ = 2e-13 m to go, so what is left is the rounding of the poses.
                before, after = schedule.at(boundary - 1.0e-4)[0], schedule.at(boundary + 1.0e-4)[0]
                self.assertLess(self._gap(before, after), self.pose_rounding)

    def test_travel_reports_the_stitch_it_left_at_its_leave_angle(self):
        """Fails if the stitch index advances before the travel is over or the tip angle moves during it."""
        for schedule, stitches in ((self.turning, self.row), (self.lifting, self.picked)):
            boundaries = self._boundaries(schedule)
            for number in range(len(stitches) - 1):
                begin, end = boundaries[2 * number], boundaries[2 * number + 1]
                for time in np.linspace(begin, end, 9)[:-1]:
                    _, index, angle = schedule.at(time)
                    self.assertEqual(index, number)
                    self.assertEqual(angle, stitches[number].leave_angle)
                self.assertEqual(schedule.at(end)[1], number + 1)

    def test_turn_and_lift_start_where_the_stitch_left_and_end_at_the_next_first_pose(self):
        """Fails if a travel does not begin at the leave pose of its stitch or end at the next stitch's first pose."""
        travels = [_Turn(a, b) for a, b in zip(self.row[:-1], self.row[1:], strict=True)]
        travels += [_Lift(a, b, self.clear_height) for a, b in zip(self.picked[:-1], self.picked[1:], strict=True)]
        pairs = [*zip(self.row[:-1], self.row[1:], strict=True), *zip(self.picked[:-1], self.picked[1:], strict=True)]
        for travel, (stitch, following) in zip(travels, pairs, strict=True):
            with self.subTest(travel=type(travel).__name__):
                leave = stitch.plan.pose(stitch.leave_angle)
                first = following.plan.pose(following.plan.initial_angle)
                self.assertGreater(self._gap(leave, first), 1.0e-3)  # there is a way to go
                self.assertLess(self._gap(travel.pose(0.0), leave), self.pose_rounding)
                self.assertLess(self._gap(travel.pose(1.0), first), self.pose_rounding)
        # The schedules end their travels there too.
        for schedule, stitches in ((self.turning, self.row), (self.lifting, self.picked)):
            for number, end in enumerate(self._boundaries(schedule)[1::2]):
                first = stitches[number + 1].plan.pose(stitches[number + 1].plan.initial_angle)
                self.assertLess(self._gap(schedule.at(end - 1.0e-9)[0], first), self.pose_rounding)

    def test_turn_keeps_the_needle_on_a_circle_that_moves_to_the_next_stitch(self):
        """Fails if the needle leaves its circle during a turn or the circle does not move steadily to the next one."""
        stitch, following = self.row[0], self.row[1]
        travel = _Turn(stitch, following)
        turn = following.plan.initial_angle + 2.0 * math.pi - stitch.leave_angle  # the rest of the full turn
        self.assertAlmostEqual(travel.length, turn * self.geometry.radius, delta=ROUNDING)
        way = following.plan.center - stitch.plan.center
        self.assertGreater(np.linalg.norm(way), 5.0e-3)
        for fraction in np.linspace(0.0, 1.0, 21):
            # The ends of the needle's capsules lie on its circle, to the 1e-6 m the needle's loader accepts;
            # measured: 2e-9 m.
            points = _needle(self.geometry, travel.pose(fraction), self.geometry.centerline)
            offsets = points - (stitch.plan.center + fraction * way)
            np.testing.assert_allclose(np.linalg.norm(offsets, axis=1), self.geometry.radius, rtol=0.0, atol=2.0e-6)
            np.testing.assert_allclose(offsets @ stitch.plan.axis, 0.0, rtol=0.0, atol=2.0e-6)
            # The tip has gone its share of the turn.
            tip = stitch.plan.tip(np.array(stitch.leave_angle + fraction * turn)) + fraction * way
            np.testing.assert_allclose(points[0], tip, rtol=0.0, atol=2.0e-6)

    def test_lift_moves_across_only_with_the_whole_needle_above_the_clear_height(self):
        """Fails if the needle turns or moves sideways during a lift while a point of it is below the clear height."""
        for stitch, following in zip(self.picked[:-1], self.picked[1:], strict=True):
            travel = _Lift(stitch, following, self.clear_height)
            leave = stitch.plan.pose(stitch.leave_angle)
            first = following.plan.pose(following.plan.initial_angle)
            lowest, level = [], 0
            fractions = np.linspace(0.0, 1.0, 801)
            for fraction in fractions:
                pose = travel.pose(fraction)
                height = _needle(self.geometry, pose)[:, 2].min()
                lowest.append(height)
                # Not straight above where it left the stitch nor above where it enters the next: on its way.
                if not _straight_above(pose, leave) and not _straight_above(pose, first):
                    level += 1
                    self.assertGreaterEqual(height, self.clear_height - self.pose_rounding)
            # Most of the way is level; measured: the lowest point stays 0.87 mm or more above the clear height.
            self.assertGreater(level, len(fractions) // 2)
            # The first three stitches end below the clear height, so the lift did rise.
            self.assertLess(lowest[0], self.clear_height)

    def test_lift_moves_no_point_of_the_needle_farther_than_its_length(self):
        """Fails if a point of the needle goes a longer way than the length the lift's duration is made from."""
        for stitch, following in zip(self.picked[:-1], self.picked[1:], strict=True):
            travel = _Lift(stitch, following, self.clear_height)
            fractions = np.linspace(0.0, 1.0, 2001)
            points = np.array([_needle(self.geometry, travel.pose(fraction)) for fraction in fractions])
            steps = np.linalg.norm(np.diff(points, axis=0), axis=2)
            # In no part of the lift does a point go farther than that part of the length, which also means
            # that the rise, the way across and the descent join without a jump. Measured: while the needle
            # rises or comes down every point goes exactly as far.
            self.assertLessEqual(steps.max(), travel.length / (len(fractions) - 1) + ROUNDING)
            self.assertLessEqual(steps.sum(axis=0).max(), travel.length * (1.0 + 1.0e-9))

    def test_time_past_the_end_keeps_the_last_pose(self):
        """Fails if the schedule runs on, or wraps around, after its duration."""
        last = self.row[-1]
        pose, index, angle = self.turning.at(self.turning.duration + 1.0)
        self.assertEqual(index, len(self.row) - 1)
        self.assertAlmostEqual(angle, last.leave_angle, delta=1.0e-12)
        self.assertLess(self._gap(pose, last.plan.pose(last.leave_angle)), self.pose_rounding)

    def test_tip_is_never_faster_than_the_pass_speed(self):
        """Fails if the tip goes along its circle faster than the pass speed, or the row takes less time than that."""
        sweeps = sum((stitch.leave_angle - stitch.plan.initial_angle) * self.geometry.radius for stitch in self.row)
        travels = sum(_Turn(a, b).length for a, b in zip(self.row[:-1], self.row[1:], strict=True))
        self.assertGreater(self.turning.duration, sweeps / self.pass_speed + travels / self.air_speed)
        # The first stitch in a thousand time steps.
        times = np.linspace(0.0, self._boundaries(self.turning)[0], 1001)
        angles = np.array([self.turning.at(time)[2] for time in times])
        speeds = self.geometry.radius * np.diff(angles) / np.diff(times)
        self.assertLessEqual(speeds.max(), self.pass_speed * (1.0 + 1.0e-9))
        self.assertGreater(speeds.max(), 0.9 * self.pass_speed)

    # Defect: a move's duration allows 0.5 s for its two 0.25 s ramps, which cost 0.25 s together, so the tip
    # cruises at D / (D / v + 0.25 s): 3.89 mm/s for the 4 mm/s asked, and 10.7 mm/s of 12 mm/s in the air.
    @unittest.expectedFailure
    def test_tip_cruises_at_the_pass_speed(self):
        """Fails while the tip's steady speed during a pass is below the pass speed the schedule was given."""
        end = self._boundaries(self.turning)[0]
        step = 1.0e-3
        angles = [self.turning.at(0.5 * end + sign * step)[2] for sign in (-1.0, 1.0)]
        speed = self.geometry.radius * (angles[1] - angles[0]) / (2.0 * step)
        self.assertAlmostEqual(speed, self.pass_speed, delta=0.005 * self.pass_speed)


class ClosingRuleTests(unittest.TestCase):
    """The end of the closing pull (``Example._observe_closing``) on a stand-in for the example: no scene is built."""

    def pulling(self, least_volume, gap=0.002, rule=None):
        """An example in its pull: two eyelets ``gap`` apart across the wound, the kept least volume of the frame."""
        if DEVICE.startswith("cuda") and not wp.is_cuda_available():
            raise unittest.SkipTest("NEWTON_TEST_DEVICE asks for CUDA and there is none")
        device = wp.get_device(DEVICE)
        example = suture.Example.__new__(suture.Example)
        targets = np.zeros((4, 3))
        targets[1] = (gap, 0.0, 0.0)
        forces = np.zeros((4, 3))
        forces[0], forces[3] = (0.0, 0.0, 0.004), (0.003, 0.0, 0.0)
        example.device = device
        example.thread = SimpleNamespace(
            eyelet_target=wp.array(targets, dtype=wp.vec3d, device=device),
            eyelet_force=wp.array(forces, dtype=wp.vec3d, device=device),
            max_eyelets=4,
        )
        example._gap_near = wp.array([0], dtype=int, device=device)
        example._gap_far = wp.array([1], dtype=int, device=device)
        example._closure = wp.zeros(2, dtype=float, device=device)
        example.rest_gap = example.wound_gap = 0.0023
        example.least_volume_ratio = least_volume
        arguments = [] if rule is None else ["--least-volume", str(rule)]
        example.args = suture.Example.create_parser().parse_args(arguments)
        example.closing, example.phase, example.elapsed = "", "pull", 12.0
        example._pulled, example._pull_reach = 0.001, 0.02
        held = []
        example.coupling = SimpleNamespace(hold_eyelet=held.append)
        example._slots = [[0, 1], [2, 3]]
        example.drive = SimpleNamespace(configure_mujoco=lambda solver, spring: None)
        example.rigid_solver = None
        example._support_needle = lambda spring: None
        return example, held

    def test_the_pull_ends_where_the_kept_least_volume_of_the_frame_is_under_the_tightness_rule(self):
        """Fails if the pull does not end by the least tissue volume that the frame's check kept, or reads for it."""
        from newton_solver_suturebot._src.solvers.thread.tissue import TIGHT_VOLUME_RATIO

        # The default of --least-volume is the coupling's one number, which fem_needle_grasp stops by too.
        self.assertEqual(suture.Example.create_parser().parse_args([]).least_volume, TIGHT_VOLUME_RATIO)
        self.assertEqual(TIGHT_VOLUME_RATIO, 0.5)
        # Above the rule the pull goes on.
        example, held = self.pulling(0.51)
        reads = []
        original = wp.array.numpy

        def read(array):
            reads.append(array)
            return original(array)

        with patch.object(wp.array, "numpy", new=read):
            example._observe_closing()
        # One read: the wound's width and the largest pull. The least volume is the number that was kept.
        self.assertEqual(len(reads), 1)
        self.assertIs(reads[0], example._closure)
        self.assertEqual((example.phase, example.closing, held), ("pull", "", []))
        self.assertEqual(example.least_volume, 0.51)
        self.assertAlmostEqual(example.wound_gap, 0.002, delta=1e-9)
        self.assertAlmostEqual(example.largest_pull, 0.004, delta=1e-9)
        self.assertEqual(example.outcome, "Pulling the thread: wound 2.00 mm of 2.30 mm wide")
        # Under it the thread is as tight as the tissue allows: a knot at the last exit, and the needle is let go.
        example, held = self.pulling(0.49)
        example._observe_closing()
        self.assertEqual((example.phase, held), ("drop", [3]))
        self.assertEqual(example.closing, "the thread is as tight as the tissue allows")
        self.assertEqual(example.outcome, "Needle dropped: the thread is as tight as the tissue allows")
        self.assertEqual((example._drop_start, example._still), (12.0, 0.0))
        # The option moves the rule.
        example, held = self.pulling(0.49, rule=0.3)
        example._observe_closing()
        self.assertEqual((example.phase, held), ("pull", []))
        # A wound that is closed to the thread's diameter ends the pull first.
        example, held = self.pulling(0.49, gap=0.00025)
        example._observe_closing()
        self.assertEqual((example.phase, example.closing, held), ("drop", "the wound is closed", [3]))


class BundledPadTests(unittest.TestCase):
    """The planned row on the bundled pad, built once through the example as it runs by default."""

    @classmethod
    def setUpClass(cls):
        if DEVICE.startswith("cuda") and not wp.is_cuda_available():
            raise unittest.SkipTest("NEWTON_TEST_DEVICE asks for CUDA and there is none")
        cls.enterClassContext(wp.ScopedDevice(DEVICE))
        cls.args = suture.Example.create_parser().parse_args([])
        viewer = ViewerNull()
        cls.addClassCleanup(viewer.close)
        cls.example = example = suture.Example(viewer, cls.args)
        cls.scene = (
            example.geometry,
            example.region_map,
            example.reference,
            example.selection,
            example.fixed_vertex_indices,
        )
        cls.first = (cls.args.first_entry_xy, cls.args.first_exit_xy)
        cls.stitches = plan_stitches(
            *cls.scene, *cls.first, cls.args.stitches, cls.args.stitch_pitch, cls.args.eyelet_spacing
        )
        cls.tissue = TetLocator(example.selection.vertices, example.selection.indices)
        cls.pad = TetLocator(example.reference.vertices, example.reference.indices)

    def test_row_has_the_requested_number_of_stitches_the_needle_can_make(self):
        """Fails if the row is short, or a stitch's sweep goes through pad material that is not the tissue."""
        self.assertEqual(len(self.stitches), self.args.stitches)
        geometry = self.example.geometry
        spacing = 1.0e-4 / geometry.radius  # a sample every 0.1 mm of the tip's way
        for stitch in self.stitches:
            plan = stitch.plan
            self.assertLess(plan.exit_angle, geometry.span)
            # Through the pad, from entry to exit, the tip is in the tissue or in the open wound.
            angles = np.arange(0.0, plan.exit_angle, spacing)
            in_tissue = self.tissue.locate(plan.tip(angles))[0] >= 0
            in_pad = self.pad.locate(plan.tip(angles))[0] >= 0
            self.assertFalse(np.any(in_pad & ~in_tissue))
            self.assertGreater(in_tissue.mean(), 0.5)  # the two lips are wider than the wound between them
            # From the plan's finish until the tail has left, the tip is clear of the pad.
            self.assertAlmostEqual(stitch.leave_angle, plan.finish_angle + geometry.span, delta=1.0e-12)
            sweep = np.arange(plan.finish_angle, stitch.leave_angle, spacing)
            self.assertTrue(np.all(self.pad.locate(plan.tip(sweep))[0] < 0))

    def test_row_is_the_first_stitch_and_copies_along_the_wound(self):
        """Fails if the stitches are not the given one and copies moved by whole pitches along the wound."""
        axis = wound_axis(self.example.selection)
        self.assertAlmostEqual(np.linalg.norm(axis), 1.0, delta=1.0e-12)
        # The wound is the long way of the active region.
        top = self.example.selection.vertices[:, :2]
        self.assertGreater(np.ptp(top @ axis), np.ptp(top @ np.array([-axis[1], axis[0]])))
        self.assertGreaterEqual(len(self.stitches), 2)  # there are copies to look at
        entries = np.array([stitch.plan.entry.reference_position[:2] for stitch in self.stitches])
        exits = np.array([stitch.plan.exit.reference_position[:2] for stitch in self.stitches])
        steps = (entries - self.first[0]) @ axis / self.args.stitch_pitch
        # Picking goes through the map's pixel coordinates and back: rounding of positions of some centimetres.
        np.testing.assert_allclose(steps, np.round(steps), rtol=0.0, atol=1.0e-9)
        self.assertIn(0, np.round(steps).astype(int).tolist())
        self.assertTrue(np.all(np.diff(np.round(steps)) >= 1.0))  # in order along the wound, no stitch twice
        offsets = np.outer(np.round(steps) * self.args.stitch_pitch, axis)
        np.testing.assert_allclose(entries, self.first[0] + offsets, rtol=0.0, atol=ROUNDING)
        np.testing.assert_allclose(exits, self.first[1] + offsets, rtol=0.0, atol=ROUNDING)
        # A row of one is the given stitch.
        (single,) = plan_stitches(*self.scene, *self.first, 1, self.args.stitch_pitch, self.args.eyelet_spacing)
        np.testing.assert_allclose(single.plan.entry.reference_position[:2], self.first[0], rtol=0.0, atol=ROUNDING)
        given = self.stitches[int(np.argmin(np.abs(steps)))]
        np.testing.assert_array_equal(single.angles, given.angles)
        np.testing.assert_array_equal(single.tetrahedra, given.tetrahedra)

    def test_track_points_lie_in_the_tissue_on_the_needle_circle(self):
        """Fails if a track point's tetrahedron and weights do not give its point of the stitch's circle."""
        selection = self.example.selection
        for stitch in self.stitches:
            count = len(stitch.angles)
            self.assertGreaterEqual(count, 4)
            self.assertEqual(stitch.tetrahedra.shape, (count,))
            self.assertEqual(stitch.weights.shape, (count, 4))
            self.assertTrue(np.all((stitch.tetrahedra >= 0) & (stitch.tetrahedra < len(selection.indices))))
            # Inside its tetrahedron, to the locator's tolerance of 1e-6.
            self.assertGreaterEqual(stitch.weights.min(), -1.0e-6)
            np.testing.assert_allclose(stitch.weights.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12)
            material = np.einsum("nc,ncd->nd", stitch.weights, selection.vertices[selection.indices[stitch.tetrahedra]])
            np.testing.assert_allclose(material, stitch.points, rtol=0.0, atol=ROUNDING)
            np.testing.assert_allclose(stitch.points, stitch.plan.tip(stitch.angles), rtol=0.0, atol=ROUNDING)
            # In order from the entry to the exit.
            self.assertTrue(np.all(np.diff(stitch.angles) > 0.0))
            self.assertGreater(stitch.angles[0], 0.0)
            self.assertLess(stitch.angles[-1], stitch.plan.exit_angle)

    def test_passages_index_the_track_points_of_each_lip(self):
        """Fails if a passage does not span the track points of one stretch of tissue, first to last."""
        radius = self.example.geometry.radius
        for stitch in self.stitches:
            intervals = stitch.plan.tissue_intervals
            # Across the open wound: in through one lip, out, and in through the other.
            self.assertEqual(len(intervals), 2)
            self.assertEqual(len(stitch.passages), len(intervals))
            # The passages take the track points in order and leave none out.
            self.assertEqual(stitch.passages[0][0], 0)
            self.assertEqual(stitch.passages[-1][1], len(stitch.angles) - 1)
            for (_, last), (first, _) in zip(stitch.passages[:-1], stitch.passages[1:], strict=True):
                self.assertEqual(first, last + 1)
            for (first, last), (start, end) in zip(stitch.passages, intervals, strict=True):
                angles = stitch.angles[first : last + 1]
                self.assertGreaterEqual(len(angles), 2)
                # The ends sit a hundredth of the passage inside the surface, the others evenly between the
                # surface points, no farther apart than the spacing asked for.
                self.assertAlmostEqual(angles[0], start + 0.01 * (end - start), delta=1.0e-12)
                self.assertAlmostEqual(angles[-1], end - 0.01 * (end - start), delta=1.0e-12)
                self.assertLessEqual(np.diff(angles).max() * radius, self.args.eyelet_spacing)
                inner = np.linspace(start, end, len(angles))[1:-1]
                np.testing.assert_allclose(angles[1:-1], inner, rtol=0.0, atol=1.0e-12)
            # Between two passages the circle crosses the open wound: no tissue at the middle of the gap.
            for (_, last), (first, _) in zip(stitch.passages[:-1], stitch.passages[1:], strict=True):
                middle = stitch.plan.tip(np.array([0.5 * (stitch.angles[last] + stitch.angles[first])]))
                self.assertEqual(self.tissue.locate(middle)[0][0], -1)
                self.assertGreater(np.linalg.norm(stitch.points[first] - stitch.points[last]), 1.0e-3)

    def test_rows_the_needle_cannot_make_are_refused(self):
        """Fails if a row is returned although its first stitch or enough of its copies cannot be made."""
        spacing, pitch = self.args.eyelet_spacing, self.args.stitch_pitch
        with self.assertRaisesRegex(ValueError, "The first stitch is not feasible"):
            plan_stitches(*self.scene, (0.5, 0.5), (0.5, 0.492), 1, pitch, spacing)  # off the pad
        with self.assertRaisesRegex(ValueError, "Only 1 of 2 stitches are feasible"):
            plan_stitches(*self.scene, *self.first, 2, pitch, spacing, search=0)  # no copies are tried

    def test_example_runs_thirty_frames_headless(self):
        """Fails if the default run cannot be built without a window, stops or fails within half a second."""
        # The example of the class: the planning tests only read its undeformed pad, so the run does not touch them.
        example = self.example
        self.assertEqual(len(example.stitches), self.args.stitches)
        for stitch, planned in zip(example.stitches, self.stitches, strict=True):
            np.testing.assert_array_equal(stitch.angles, planned.angles)
        self.assertTrue(example.running)  # the row starts at once by default
        start, frames = example.elapsed, 30
        for _ in range(frames):
            example.step()
        self.assertTrue(example.running, example.outcome)
        self.assertNotEqual(example.task_status, "failed")
        self.assertAlmostEqual(example.elapsed - start, frames * example.frame_dt, delta=1.0e-9)
        self.assertTrue(np.isfinite(example.thread.q.numpy()).all())
        example.test_final()

    def test_planned_entries_and_exits_are_drawn_on_their_material_points(self):
        """Fails if the spheres of the planned stitches are not where their surface triangles are in the drawn state."""
        example = self.example
        plans = [stitch.plan for stitch in example.stitches]
        points = [plan.entry for plan in plans] + [plan.exit for plan in plans]
        calls = []

        def log_points(name, points, radii=None, colors=None, hidden=False):
            if name == "/suture/stitches":
                calls.append((points.numpy().copy(), radii.numpy().copy(), colors.numpy().copy(), hidden))

        def draw():
            del calls[:]
            with patch.object(example.viewer, "log_points", side_effect=log_points):
                example.render()
            self.assertEqual(len(calls), 1)
            return calls[0]

        saved = wp.clone(example.state_0.particle_q)
        self.addCleanup(lambda: wp.copy(example.state_0.particle_q, saved))
        before = example.state_0.particle_q.numpy()
        drawn, radii, colors, hidden = draw()
        self.assertFalse(hidden)
        # One float32 sum of three nodes at a few centimetres.
        np.testing.assert_allclose(drawn, [point.position(before) for point in points], rtol=0, atol=1e-8)
        np.testing.assert_array_equal(radii, np.full(len(points), 0.0004, dtype=np.float32))
        np.testing.assert_array_equal(
            colors, np.float32([suture.ENTRY_COLOR] * len(plans) + [suture.EXIT_COLOR] * len(plans))
        )
        # The tissue moves, each node by its own amount: every sphere goes with its triangle.
        moved = before + 0.001 * np.sin(300.0 * before[:, [1, 2, 0]]).astype(np.float32)
        example.state_0.particle_q.assign(moved)
        after = draw()[0]
        np.testing.assert_allclose(after, [point.position(moved) for point in points], rtol=0, atol=1e-8)
        self.assertGreater(float(np.linalg.norm(after - drawn, axis=1).min()), 1e-5)

    def test_thread_graph_serves_every_step_size(self):
        """Fails if the thread graph of the nominal step differs from the eager thread step at another step size.

        A rejected needle and tissue step is repeated with half the step, and a frame can end with a remainder.
        The graph reads the step size from the thread's step time, so it is launched for those steps too. Also
        fails if such a step runs eagerly, or if the step size has no effect on the thread step here.
        """
        example = self.example
        if example._thread_graph is None:
            self.skipTest("The thread graph needs CUDA")
        state = (
            *example.thread.state,
            *example.coupling.state,
            *example.needle_contact.state,
            example._external_forces,
            example.attachment.wrench,
        )
        saved = [wp.clone(array) for array in state]

        def restore():
            for array, values in zip(state, saved, strict=True):
                wp.copy(array, values)

        def results():
            return [array.numpy().copy() for array in state]

        self.addCleanup(restore)
        base, graph, steps = example.sim_dt, example._thread_graph, {}
        for dt in (base, base / 2, 0.0020833333333333355, base / 3, base):
            with self.subTest(dt=dt):
                restore()
                with patch.object(example, "_advance_thread", side_effect=AssertionError("eager thread step")):
                    example._step_thread(dt)
                replayed = results()
                self.assertIs(example._thread_graph, graph)
                self.assertEqual(example.thread.step_time.value, dt)
                np.testing.assert_array_equal(example.thread.step_time.seconds_f64.numpy(), [dt])
                # The same step issued eagerly from the same state and input buffers.
                restore()
                example._advance_thread(dt)
                for name, actual, expected in zip(range(len(state)), replayed, results(), strict=True):
                    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-10, err_msg=f"state[{name}]")
                steps[dt] = replayed
        # The step size did matter: the thread's velocities after the nominal and the halved step differ.
        self.assertFalse(np.allclose(steps[base][1], steps[base / 2][1], rtol=1.0e-3, atol=0.0))


if __name__ == "__main__":
    unittest.main()
