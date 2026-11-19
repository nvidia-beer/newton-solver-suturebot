"""The way of the regrasp from the one grasp to the other: its rules against computations by hand.

The tool lets go of the needle, backs away from it and from the tissue, rises along its shaft, turns over to
the needle tip and comes down on it. Every capsule of the tool (shaft, wrist, jaws) has to stay clear of the
tissue at rest and of the needle on that way. Here the tissue is the plane z = 0 with the tissue below it, so
that the clearance of a capsule is the height of the lowest point of its axis less its radius, and each length
of the way can be written down. Nothing here steps physics; test_grasp_lift runs the whole regrasp.
"""

import contextlib
import io
import unittest
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from test_grasp_procedure import fixture
from test_grasp_tip_clearance import (
    DRAWN_RADIUS,
    TROCAR,
    WIRE_RADIUS,
    controller_for,
    exact_offset,
    flat_tissue,
    lowest_axis_height,
    missed,
    planned,
    pose_from,
    straight_needle,
    straight_pose,
    tool,
)

from newton_solver_suturebot.examples.grasp import _grasp_procedure as procedure
from newton_solver_suturebot.examples.grasp._dvrk_rcm_grasper import WRIST_LENGTH_M, _arm_pose, arm_link_poses
from newton_solver_suturebot.examples.grasp._grasp_procedure import (
    CLEARANCE_REFINEMENT,
    CLEARANCE_SPACING_RADII,
    ROUTE_CAPSULES_PER_STEP,
    TissueSurface,
    action_to,
    jaw_capsules,
    line_to,
    sweep_clearance,
    tool_capsules,
    turning_clearance,
    wire_clearance,
)

# The triangles of the flat tissue of these tests have legs of 1 mm: every point of one is within half its
# longest edge of a corner (0.707 mm) and within the distance of its centre from its far corners of that centre
# (0.745 mm). A rule that takes the surface by such points is short of the exact distance by no more.
TRIANGLE_COVER = np.sqrt(5.0) / 3.0 * 0.001
# A needle far from everything, for the rules that are asked about the tissue alone.
FAR = np.r_[0.3, 0.3, 0.05, 0.0, 0.0, 0.0, 1.0]


def stepped(pose, action, config):
    """The target after one frame of an action, as the scene applies it."""
    result = pose.copy()
    result[:3] += config.translation_step * action[:3]
    rotation = config.rotation_step * np.asarray(action[3:6], dtype=float)
    angle = float(np.linalg.norm(rotation))
    if angle:
        turn = wp.quat_from_axis_angle(wp.vec3(*(rotation / angle)), angle)
        quaternion = np.asarray(turn * wp.quat(*result[3:]), dtype=float)
        result[3:] = quaternion / np.linalg.norm(quaternion)
    return result


def jaws_only():
    """The tool of the fixture without its shaft and wrist capsules: the jaws move straight with the palm."""
    entry = tool()
    for name in ("capsule_endpoints", "capsule_radii", "capsule_contact_radii"):
        entry[name] = entry[name][2:]
    entry["capsule_bodies"] = entry["capsule_bodies"][2:]
    return entry


def scene_state(controller, leg="rise"):
    """A scene of the controller in the regrasp, as the release leaves it."""
    state = procedure._Scene(0, 0.0, 0.0, 0.0)
    state.stage, state.leg = "retreat", leg
    return state


def decided(call):
    """The answer of a question that is asked a number of capsules per call."""
    for _ in range(64):
        answer = call()
        if answer is not None:
            return answer
    raise AssertionError("The question was not decided")


class StraightStepTests(unittest.TestCase):
    def test_a_line_step_keeps_the_target_on_the_segment_and_turns_it_about_one_axis(self):
        config = fixture()[1]
        start = np.r_[0.01, 0.02, 0.03, 0.0, 0.0, 0.0, 1.0]
        axis = np.array([1.0, 2.0, 2.0]) / 3.0
        goal = np.r_[start[:3] + [0.003, -0.001, 0.0005], np.asarray(wp.quat_from_axis_angle(wp.vec3(*axis), 0.8))]
        way = goal[:3] - start[:3]
        pose, frames, turned = start.copy(), 0, []
        while True:
            action = line_to(pose, goal, config)
            if np.max(np.abs(action)) < 1e-3:
                break
            # No component above its rate limit; the largest one of each part at it until the last frame.
            self.assertLessEqual(np.max(np.abs(action)), 1.0 + 1e-9)
            before, pose = pose, stepped(pose, action, config)
            frames += 1
            # On the segment: no part of the position across the way.
            self.assertLess(np.linalg.norm(np.cross(pose[:3] - start[:3], way)) / np.linalg.norm(way), 1e-9)
            step = np.asarray(wp.quat(*pose[3:]) * wp.quat_inverse(wp.quat(*before[3:])), dtype=float)
            if np.linalg.norm(step[:3]) > 1e-6:
                turned.append(step[:3] / np.linalg.norm(step[:3]) * np.sign(step[3]))
            self.assertLess(frames, 200)
        np.testing.assert_allclose(pose[:3], goal[:3], atol=1e-9)
        self.assertAlmostEqual(abs(pose[3:] @ goal[3:]), 1.0, delta=1e-6)
        # 3 mm at 0.1 mm per frame along its largest component; 0.8 rad about (1, 2, 2) / 3 at 0.01 rad per
        # frame of its largest component, 0.8 x 2 / 3 rad.
        self.assertEqual(frames, max(30, int(np.ceil(0.8 * 2.0 / 3.0 / 0.01))))
        np.testing.assert_allclose(turned, np.tile(axis, (len(turned), 1)), atol=2e-5)
        # The same target by components leaves the segment: after ten frames the two smaller components are
        # at their ends, and the position (1, -1, 0.5) mm is 0.698 mm from the line to (3, -1, 0.5) mm.
        bent = start.copy()
        for _ in range(10):
            bent = stepped(bent, action_to(bent, goal, config), config)
        np.testing.assert_allclose(bent[:3] - start[:3], [0.001, -0.001, 0.0005], atol=1e-9)
        off = np.linalg.norm(np.cross(bent[:3] - start[:3], way)) / np.linalg.norm(way)
        self.assertAlmostEqual(off, np.sqrt(5.0e-12) / np.sqrt(10.25e-6), delta=1e-9)


class LinkPoseTests(unittest.TestCase):
    def test_shaft_and_wrist_follow_the_palm_and_the_trocar_as_the_joint_frames_have_them(self):
        nominal = 0.18
        wrist_turn = wp.quat(-0.5, -0.5, -0.5, 0.5)

        def turn(axis, angle):
            return wp.transform((0, 0, 0), wp.quat_from_axis_angle(wp.vec3(axis), float(angle)))

        trocars = (
            np.r_[0.1, -0.03, 0.2, 0.0, 0.0, 0.0, 1.0],
            np.r_[0.116, -0.037, 0.19, np.asarray(wp.quat_from_axis_angle(wp.vec3(0.6, 0.0, 0.8), 2.5))],
        )
        joints = (
            (0.0, 0.0, 0.0, 0.0, -0.7, 0.0),
            (0.3, -0.2, 0.02, 2.0, 0.13, 0.95),
            (-0.25, 0.4, -0.03, 3.1, -0.35, -0.3),
            (0.1, 0.1, 0.04, -1.2, 0.9, -1.0),
        )
        for trocar in trocars:
            for yaw, pitch, slide, roll, wrist_pitch, wrist_yaw in joints:
                with self.subTest(joints=(yaw, pitch, slide, roll, wrist_pitch, wrist_yaw)):
                    # The chain of add_rcm_grasper, link by link.
                    shaft = wp.transform(*trocar) * turn((0, 1, 0), yaw) * turn((1, 0, 0), pitch)
                    shaft = shaft * wp.transform((0, 0, nominal + slide), wp.quat_identity()) * turn((0, 0, 1), roll)
                    wrist = shaft * wp.transform((0, 0, 0), wrist_turn) * turn((0, 0, 1), wrist_pitch)
                    palm = wrist * wp.transform((WRIST_LENGTH_M, 0, 0), wrist_turn) * turn((0, 0, 1), wrist_yaw)
                    np.testing.assert_allclose(
                        np.asarray(palm, dtype=float)[:3],
                        np.asarray(
                            wp.transform(*trocar)
                            * _arm_pose((yaw, pitch, slide, roll, wrist_pitch, wrist_yaw), nominal)
                        )[:3],
                        atol=1e-7,
                    )
                    found = arm_link_poses(trocar, np.asarray(palm, dtype=float))
                    for link, expected in zip(found, (shaft, wrist), strict=True):
                        expected = np.asarray(expected, dtype=float)
                        np.testing.assert_allclose(link[:3], expected[:3], atol=2e-7)
                        self.assertAlmostEqual(abs(link[3:] @ expected[3:]), 1.0, delta=1e-6)


class ToolCapsuleTests(unittest.TestCase):
    def test_every_capsule_has_its_link_its_contact_radius_and_its_bound(self):
        entry = tool()
        rows = tool_capsules(entry)
        self.assertEqual([row[4] for row in rows], ["roll", "pitch", "jaw1", "jaw2"])
        np.testing.assert_allclose([row[2] for row in rows], entry["capsule_contact_radii"])
        self.assertEqual([row[3] for row in rows], [0.0] * 4)
        # The jaws as jaw_capsules has them; the wrist drawn in by its contact radius less its model radius
        # at each end (0.8 mm), the shaft let out by 0.1 mm: their caps reach as far as the model's.
        for row, jaw in zip(rows[2:], jaw_capsules(entry), strict=True):
            np.testing.assert_allclose(row[0], jaw[0], atol=1e-15)
            np.testing.assert_allclose(row[1], jaw[1], atol=1e-15)
        np.testing.assert_allclose(rows[1][0], [0.0008, 0, 0], atol=1e-12)
        np.testing.assert_allclose(rows[1][1], [0.0072, 0, 0], atol=1e-12)
        np.testing.assert_allclose(rows[0][0], [0, 0, -0.2431], atol=1e-12)
        np.testing.assert_allclose(rows[0][1], [0, 0, -0.0029], atol=1e-12)
        bounded = {**entry, "capsule_bounds": np.array([0.0, 0.0006, 0.0001, 0.0002])}
        self.assertEqual([row[3] for row in tool_capsules(bounded)], [0.0, 0.0006, 0.0001, 0.0002])
        with self.assertRaises(ValueError):
            tool_capsules({**entry, "capsule_bounds": np.array([0.0, -0.0006, 0.0001, 0.0002])})
        # A body that the tool does not name cannot be placed: the regrasp of such a tool is refused.
        unnamed = {name: value for name, value in entry.items() if name != "part_body_indices"}
        self.assertEqual([row[4] for row in tool_capsules(unnamed)], [None, None, "jaw1", "jaw2"])
        self.assertFalse(controller_for(straight_needle(), unnamed).routable)
        self.assertTrue(controller_for(straight_needle(), entry).routable)

    def test_the_turning_radius_is_the_furthest_reach_of_a_capsule_from_the_palm_origin(self):
        controller = controller_for(straight_needle(), tool())
        # By hand, for the tool of the fixture. A jaw: the far end of its axis, (0.75, 9 - 0.326) mm from the
        # origin, and its radius 0.976 mm. The wrist: its axis ends 0.8 mm from the wrist joint, which is the
        # wrist length from the palm origin; radius 3 mm. The shaft: the wrist length and its radius 4.2 mm.
        jaw = np.hypot(0.00075, 0.009 - (DRAWN_RADIUS - 0.00065)) + DRAWN_RADIUS
        wrist = WRIST_LENGTH_M - 0.0008 + 0.003
        shaft = WRIST_LENGTH_M + 0.0042
        self.assertAlmostEqual(controller.turning_radius, max(jaw, wrist, shaft), delta=1e-9)
        self.assertAlmostEqual(controller.turning_radius, 0.0133, delta=1e-9)
        self.assertAlmostEqual(controller_for(straight_needle(), jaws_only()).turning_radius, jaw, delta=1e-12)
        # With the bound of a drawn part the capsule counts for more.
        bounded = {**tool(), "capsule_bounds": np.array([0.0, 0.0025, 0.0, 0.0])}
        self.assertAlmostEqual(controller_for(straight_needle(), bounded).turning_radius, wrist + 0.0025, delta=1e-9)
        # The wrist length is that of the tool model, and the wrist link holds the palm origin that far
        # along its own X axis: the radius is not measured at a pose and is the same wherever the trocar is.
        self.assertEqual(controller.wrist_length, WRIST_LENGTH_M)
        elsewhere = {**tool(), "trocar_pose": TROCAR + np.r_[0.02, -0.03, 0.05, np.zeros(4)]}
        self.assertEqual(controller_for(straight_needle(), elsewhere).turning_radius, controller.turning_radius)
        # The links of a posed tool give the same lengths, to the rounding of the pose.
        palm = np.r_[TROCAR[:3] - [0.03, 0.02, 0.1], 0.0, 0.0, 0.0, 1.0]
        roll, pitch = arm_link_poses(TROCAR, palm)
        self.assertAlmostEqual(np.linalg.norm(roll[:3] - palm[:3]), WRIST_LENGTH_M, delta=1e-15)
        ends = np.stack(tool_capsules(tool())[1][:2]) @ rotation_of(pitch).T + pitch[:3] - palm[:3]
        self.assertAlmostEqual(np.linalg.norm(ends, axis=1).max() + 0.003, wrist, delta=1e-8)


class SweepClearanceTests(unittest.TestCase):
    """A capsule over the plane z = 0: its clearance is the least height of its axis less its radius."""

    surface = TissueSurface(*flat_tissue())

    def check(self, start, end, radius, move, exact):
        for decide in (False, True):
            value, point = sweep_clearance(self.surface, start, end, radius, move, decide=decide)
            # A lower bound, short of the exact value by no more than the cover of the samples that decide:
            # a sixteenth of the radius, or a quarter of it where the first samples already decide (a capsule
            # in the tissue, or one that they show clear when no more is asked).
            short = CLEARANCE_SPACING_RADII * radius / (1 if decide or exact < 0 else CLEARANCE_REFINEMENT)
            self.assertLessEqual(value, exact + 1e-9)
            self.assertGreaterEqual(value, exact - short - 1e-9)
            self.assertAlmostEqual(point[2], 0.0, delta=1e-12)
        return value

    def test_a_capsule_is_as_far_from_a_plane_as_the_lowest_point_of_its_sweep_less_its_radius(self):
        radius = 0.001
        level = (np.array([0.002, 0.001, 0.0014]), np.array([0.008, -0.002, 0.0014]))
        tilted = (np.array([0.002, 0.001, 0.0031]), np.array([0.008, -0.002, 0.0016]))
        still, up, along, down = (
            np.zeros(3),
            np.array([0.001, 0.0, 0.004]),
            np.array([0.003, 0.002, 0.0]),
            np.array([0.002, 0.0, -0.0009]),
        )
        self.check(*level, radius, still, 0.0004)
        self.check(*tilted, radius, still, 0.0006)
        # Rising or moving level: the start decides. Coming down: the end.
        self.check(*tilted, radius, up, 0.0006)
        self.check(*level, radius, along, 0.0004)
        self.check(*tilted, radius, down, 0.0006 - 0.0009)
        # A larger capsule on the same axis is nearer by the difference of the radii.
        self.check(*tilted, 0.0015, still, 0.0001)

    def test_a_capsule_that_starts_inside_the_tissue_and_comes_out_is_not_clear(self):
        radius = 0.001
        inside = (np.array([0.002, 0.001, -0.003]), np.array([0.008, -0.002, -0.003]))
        value, point = sweep_clearance(self.surface, *inside, radius, np.array([0.0, 0.0, 0.01]))
        # Its sweep passes through the surface: nearer than its radius by the whole radius.
        self.assertLessEqual(value, -radius + 1e-9)
        self.assertGreaterEqual(value, -radius - CLEARANCE_SPACING_RADII * radius)
        self.assertAlmostEqual(point[2], 0.0, delta=1e-12)

    def test_no_tissue_in_reach_is_clear_without_a_point(self):
        far = (np.array([0.002, 0.001, 0.02]), np.array([0.008, -0.002, 0.02]))
        self.assertEqual(sweep_clearance(self.surface, *far, 0.001, np.array([0.0, 0.0, 0.01])), (np.inf, None))


class TurningClearanceTests(unittest.TestCase):
    surface = TissueSurface(*flat_tissue())

    def test_tissue_and_wire_are_as_far_from_the_palms_way_as_its_lowest_and_nearest_point(self):
        radius = 0.0133
        first, second = np.array([0.002, 0.001, 0.016]), np.array([0.011, -0.004, 0.0145])
        apex = np.array([0.0, 0.0, 0.26])
        # The strip of a tool without a shaft is the segment; with one, the points part of the way to the trocar.
        for fraction in (0.0, 0.2):
            strip = (first, second, second + fraction * (apex - second), first + fraction * (apex - first))
            value, witness = turning_clearance(self.surface, np.zeros((0, 3)), 0.0, strip, radius)
            # By hand: the lower end of the way, 14.5 mm above the plane.
            self.assertLessEqual(value, 0.0145 - radius + 1e-9)
            self.assertGreaterEqual(value, 0.0145 - radius - TRIANGLE_COVER)
            self.assertEqual(witness[0], "tissue")
            self.assertAlmostEqual(witness[1][2], 0.0, delta=1e-12)
        # A way 1 mm lower is within the turning radius of the tissue.
        low = (first - [0, 0, 0.0015], second - [0, 0, 0.0015], second - [0, 0, 0.0015], first - [0, 0, 0.0015])
        self.assertLess(turning_clearance(self.surface, np.zeros((0, 3)), 0.0, low, radius)[0], 0.0)
        # A wire: stations 13.9 mm above the middle of the way count with the wire radius and what a point
        # between two stations can be nearer.
        middle = 0.5 * (first + second)
        wire = middle + np.array([[0.0, -0.001, 0.0139], [0.0, 0.0, 0.0139], [0.0, 0.001, 0.0139]])
        strip = (first, second, second, first)
        along = (second - first) / np.linalg.norm(second - first)
        nearest = np.linalg.norm(np.cross(wire - first, along), axis=1).min()
        value, witness = turning_clearance(self.surface, wire, WIRE_RADIUS + 0.00007, strip, radius)
        self.assertAlmostEqual(value, nearest - WIRE_RADIUS - 0.00007 - radius, delta=1e-9)
        self.assertEqual(witness[0], "wire")
        self.assertLess(value, 0.0145 - radius - TRIANGLE_COVER)
        # Neither in reach: clear, with nothing to name.
        high = tuple(point + [0, 0, 0.03] for point in strip)
        self.assertEqual(turning_clearance(self.surface, np.zeros((0, 3)), 0.0, high, radius), (np.inf, None))


class RouteRuleTests(unittest.TestCase):
    """The rules of the way on the plane z = 0, for the tool of the fixture; the needle far away unless said."""

    half = 0.5 * procedure.ProceduralGraspController.release_opening_rad

    def flat_pose(self, height, x=0.004, y=-0.003):
        """The tool flat over the tissue: its jaws along the world's -X, its closing axis straight down."""
        frame = np.column_stack(([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
        return pose_from(frame, (x, y, height))

    def test_the_tool_backs_away_along_the_open_jaw_that_rises_towards_its_trocar(self):
        controller = controller_for(straight_needle(), tool())
        pose = self.flat_pose(0.0035)
        # The open jaws point along (-cos h, 0, +-sin h): the second one down. Back along it is up.
        np.testing.assert_allclose(
            controller._leave_axis(pose, self.half), [np.cos(self.half), 0.0, np.sin(self.half)], atol=1e-7
        )
        # Turned over, it is the first jaw that points down.
        turned = pose_from(np.column_stack(([0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0])), pose[:3])
        np.testing.assert_allclose(
            controller._leave_axis(turned, self.half), [np.cos(self.half), 0.0, np.sin(self.half)], atol=1e-7
        )
        # The rise is along the shaft, from the wrist joint to the trocar, as far as the slide goes: the shaft
        # is 0.18 - 0.05 m at its shortest.
        wrist = arm_link_poses(TROCAR, pose)[0][:3]
        rise = controller._rise(pose)
        length = np.linalg.norm(TROCAR[:3] - wrist)
        np.testing.assert_allclose(rise, (TROCAR[:3] - wrist) / length * (length - 0.13), atol=1e-12)

    def test_the_way_back_ends_where_the_rise_is_clear_of_the_tissue(self):
        controller = controller_for(straight_needle(), jaws_only())
        radius, half = DRAWN_RADIUS, self.half
        released = self.flat_pose(0.0035)
        back = controller._leave_axis(released, half)
        # By hand: the far end of the lower jaw's axis is (0.75 cos h + 8.674 sin h) mm below the palm origin.
        # Back along that jaw the tool rises by sin h per length: clear of the plane after
        reach = 0.009 - (DRAWN_RADIUS - 0.00065)
        below = 0.00075 * np.cos(half) + reach * np.sin(half)
        exact = (below + radius - 0.0035) / np.sin(half)
        self.assertGreater(exact, 0.002)
        item = {"needle_pose": FAR}

        def blocked(length):
            state = scene_state(controller, leg="rise")
            pose = controller._shifted(released, length * back)
            return decided(lambda: controller._rise_blocked(state, item, pose, half)), state

        self.assertTrue(blocked(0.0)[0])
        answer, state = blocked(exact - 0.0001)
        self.assertTrue(answer)
        # What is in the way is named: a point of the tissue under the lower jaw (the second capsule).
        kind, row, point = state.rise_witness
        self.assertEqual((kind, row), ("tissue", 1))
        self.assertAlmostEqual(point[2], 0.0, delta=1e-12)
        # Clear, by the rule, no further on than what its samples can miss (a sixteenth of the radius).
        slack = CLEARANCE_SPACING_RADII * radius / CLEARANCE_REFINEMENT / np.sin(half)
        answer, state = blocked(exact + slack + 0.0001)
        self.assertFalse(answer)
        self.assertIsNone(state.rise_witness)

    def test_a_wire_over_a_jaw_is_in_the_way_of_the_rise(self):
        controller = controller_for(straight_needle(), jaws_only())
        half = self.half
        pose = self.flat_pose(0.012)
        state = scene_state(controller)
        self.assertFalse(decided(lambda: controller._rise_blocked(state, {"needle_pose": FAR}, pose, half)))
        # A straight wire along the world's Y, 3 mm above the far end of the upper jaw's axis: the rise, which
        # is all but straight up, takes the jaw into it. The same wire 2.5 mm further out, beyond the jaw's
        # radius, the wire's and what a station can miss: the jaw passes it.
        reach = 0.009 - (DRAWN_RADIUS - 0.00065)
        end = pose[:3] + [
            -(reach * np.cos(half) - 0.00075 * np.sin(half)),
            0.0,
            0.00075 * np.cos(half) + reach * np.sin(half),
        ]
        along_y = pose_from(np.column_stack(([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0])), (0.0, 0.0, 0.0))
        for out, expected in ((0.0, True), (0.0025, False)):
            needle = along_y.copy()
            needle[:3] = end + [-out, -0.01, 0.003]
            state = scene_state(controller)
            item = {"needle_pose": needle}
            answer = decided(lambda state=state, item=item: controller._rise_blocked(state, item, pose, half))
            self.assertEqual(answer, expected)
            if expected:
                self.assertEqual(state.rise_witness[:2], ("wire", 0))

    def test_the_tool_can_turn_where_tissue_and_wire_are_a_turning_radius_from_its_palm(self):
        controller = controller_for(straight_needle(), tool())
        radius = controller.turning_radius
        item = {"needle_pose": FAR}

        def blocked(first, second):
            state = scene_state(controller)
            return controller._turn_blocked(state, item, np.asarray(first), np.asarray(second)), state

        # Standing: the palm origin a turning radius above the plane, and the cover of the samples more.
        (answer, looked), state = blocked((0.004, -0.003, radius - 0.0001), (0.004, -0.003, radius - 0.0001))
        self.assertEqual((answer, looked), (True, True))
        self.assertEqual(state.turn_witness[0], "tissue")
        # What was found in the way settles the next question without a look at the tissue.
        lower = (np.array([0.004, -0.003, radius - 0.0002]),) * 2
        self.assertEqual(controller._turn_blocked(state, item, *lower), (True, False))
        (answer, looked), state = blocked((0.004, -0.003, radius + 0.00075), (0.004, -0.003, radius + 0.00075))
        self.assertEqual((answer, looked), (False, True))
        # Asked again where it stands, the tool does not look at the tissue again.
        way = (np.array([0.004, -0.003, radius + 0.00075]),) * 2
        self.assertEqual(controller._turn_blocked(state, item, *way), (False, False))
        # On the way to a lower point it is the lower end that counts.
        (answer, _looked), _state = blocked((0.004, -0.003, radius + 0.003), (0.010, -0.008, radius - 0.0001))
        self.assertTrue(answer)
        (answer, _looked), _state = blocked((0.004, -0.003, radius + 0.003), (0.010, -0.008, radius + 0.00075))
        self.assertFalse(answer)
        # A wire within the turning radius, the wire radius and what a station can miss of the palm is in the
        # way as well: a straight wire along the world's Y at the height of the palm, 13.5 mm beside it; 13.8
        # mm beside it, it is not (13.3 + 0.28 + 0.07 mm).
        along_y = pose_from(np.column_stack(([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0])), (0.0, -0.013, 0.0))
        point = np.array([0.004, -0.003, radius + 0.003])
        for beside, expected in ((0.0135, True), (0.0138, False)):
            needle = along_y.copy()
            needle[0], needle[2] = point[0] - beside, point[2]
            state = scene_state(controller)
            answer, _looked = controller._turn_blocked(state, {"needle_pose": needle}, point, point)
            self.assertEqual(answer, expected)
            if expected:
                self.assertEqual(state.turn_witness[0], "wire")
        # The strip of the way: the part of each end's line to the trocar on which the shaft is not yet
        # within the reach about the trocar that is free of tissue and wire.
        first, second = np.array([0.004, -0.003, 0.02]), np.array([0.010, -0.008, 0.016])
        safe = controller._safe_reach(controller._wire(FAR))
        strip = controller._strip(first, second, safe)
        fraction = max(
            1.0 - (safe - 0.0042) / (np.linalg.norm(point - TROCAR[:3]) + WRIST_LENGTH_M) for point in (first, second)
        )
        self.assertGreater(fraction, 0.001)
        np.testing.assert_allclose(strip[2], second + fraction * (TROCAR[:3] - second), atol=1e-12)
        np.testing.assert_allclose(strip[3], first + fraction * (TROCAR[:3] - first), atol=1e-12)
        # No tissue is nearer the trocar than the nearest corner of the patch, (25, -25, 0) mm, less what a
        # corner stands for: half the longest edge of its triangles.
        nearest = np.linalg.norm(TROCAR[:3] - [0.025, -0.025, 0.0]) - 0.5 * np.sqrt(2.0) * 0.001
        self.assertAlmostEqual(controller.trocar_gap, nearest, delta=1e-9)

    def test_the_way_down_starts_above_the_pregrasp_where_the_tool_can_turn(self):
        controller = controller_for(straight_needle(), jaws_only())
        radius = controller.turning_radius
        clearance = controller.transfer_clearance

        def route(grasp):
            state = scene_state(controller)
            state.tip_grasp, state.plan_pose = grasp, FAR
            plan = controller._plan_route(state)
            for _ in range(400):
                try:
                    next(plan)
                except StopIteration as done:
                    return done.value, state
            raise AssertionError("The way was not planned")

        def grasp(height, down):
            """A grasp with the jaws pointing ``down`` [rad] below the level, closing across them, level."""
            along = np.array([np.cos(down), 0.0, -np.sin(down)])
            frame = np.column_stack(([0.0, 1.0, 0.0], along, np.cross([0.0, 1.0, 0.0], along)))
            return pose_from(frame, (0.004, -0.003, height))

        # The pregrasp of a grasp 9 mm up, jaws at 30 degrees, is 15 mm up: the tool can turn there.
        reason, state = route(grasp(0.009, np.pi / 6))
        self.assertIsNone(reason)
        np.testing.assert_allclose(state.above, controller._backed_off(state.tip_grasp), atol=1e-12)
        # Of a grasp 3 mm up it is 9 mm up, less than the turning radius (9.68 mm): the tool comes to the line
        # of the descent further back, where its palm is a turning radius up, by hand (radius - 3 mm) / sin 30,
        # and no further than the cover of the samples and the end of the search (a 256th of the radius) on.
        reason, state = route(grasp(0.003, np.pi / 6))
        self.assertIsNone(reason)
        exact = (radius - 0.003) / 0.5
        self.assertGreater(exact, clearance)
        length = np.linalg.norm(state.above[:3] - state.tip_grasp[:3])
        np.testing.assert_allclose(state.above, controller._backed_off(state.tip_grasp, length), atol=1e-12)
        self.assertGreaterEqual(length, exact - 1e-9)
        self.assertLessEqual(length, exact + (TRIANGLE_COVER + radius / 256) / 0.5 + 1e-9)
        # Jaws that lie level gain no height: there is no room, and the way is refused with that reason.
        reason, state = route(grasp(0.006, 0.0))
        self.assertIn("no room above the needle tip", reason)
        self.assertIsNone(state.above)
        # Jaws that would come down through the tissue on their way to the pregrasp are refused: a grasp 6 mm
        # under the plane, jaws straight down; its pregrasp is 6 mm up and the point above it 10 mm up.
        reason, _state = route(grasp(-0.006, np.pi / 2))
        self.assertIn("reach into the tissue", reason)

    def test_wrist_and_shaft_are_kept_out_of_the_tissue_on_the_way_down(self):
        controller = controller_for(straight_needle(), tool())
        state = scene_state(controller)
        # Jaws level 20 degrees under the horizon, the grasp 3.5 mm up: the jaws stay above the plane, the
        # wrist capsule (radius 3 mm, about the palm) does not.
        along = np.array([np.cos(0.35), 0.0, -np.sin(0.35)])
        frame = np.column_stack(([0.0, 1.0, 0.0], along, np.cross([0.0, 1.0, 0.0], along)))
        state.tip_grasp, state.plan_pose = pose_from(frame, (0.004, -0.003, 0.0035)), FAR
        self.assertGreater(0.0035 - 0.009 * np.sin(0.35) - DRAWN_RADIUS, -0.0007)
        plan = controller._plan_route(state)
        with self.assertRaises(StopIteration) as done:
            for _ in range(600):
                next(plan)
        self.assertIn("wrist or the shaft", done.exception.value)
        self.assertGreaterEqual(ROUTE_CAPSULES_PER_STEP, 1)


class FittedSetPlanTests(unittest.TestCase):
    """The plan of the grasp at the tip with the capsule set that the scene gives its tools."""

    @classmethod
    def setUpClass(cls):
        from newton_solver_suturebot.examples.grasp._grasp_scene import add_dvrk_tools

        with wp.ScopedDevice("cpu"):
            assembly = add_dvrk_tools(newton.ModelBuilder())[0]
        cls.entry = {
            "jaw_body_indices": list(assembly.jaw_bodies),
            "capsule_bodies": list(assembly.capsule_bodies),
            "capsule_endpoints": assembly.capsule_endpoints,
            "capsule_radii": assembly.capsule_radii,
            "capsule_bounds": assembly.capsule_bounds,
        }

    def test_the_grasp_point_is_the_one_at_which_every_capsule_of_the_fitted_jaws_is_clear(self):
        controller = controller_for(straight_needle(), self.entry)
        # Eight capsules on each jaw, side by side across it and off the plane in which the jaws turn.
        self.assertEqual(len(controller.jaws), 16)
        self.assertEqual(sorted(row[3] for row in controller.jaws), [-1.0] * 8 + [1.0] * 8)
        self.assertGreater(min(abs(row[1][2]) for row in controller.jaws), 0.0002)
        half = 0.5 * controller.release_opening_rad
        for exit_angle, tip_height in ((np.radians(70.0), 0.0048), (np.radians(85.0), 0.0052)):
            with self.subTest(exit_angle=exit_angle):
                needle_pose = straight_pose(exit_angle, tip_height)
                free = tip_height / np.sin(exit_angle)
                result, steps = planned(controller, needle_pose)
                self.assertIsNotNone(result)
                pose, offset = result
                # A point of the search is spread over the caller's steps, a capsule with tissue in reach at a
                # time; asked for whole points, the plan takes fewer steps and comes to the same grasp.
                spread = procedure.PLAN_CAPSULES_PER_STEP
                try:
                    procedure.PLAN_CAPSULES_PER_STEP = len(controller.jaws) + 1
                    whole, fewer = planned(controller, needle_pose)
                finally:
                    procedure.PLAN_CAPSULES_PER_STEP = spread
                self.assertGreater(steps, fewer + 4)
                self.assertEqual(whole[1], offset)
                np.testing.assert_array_equal(whole[0], pose)
                # By hand, over all sixteen capsules: the tissue decides here, not the wire between the jaws.
                expected = exact_offset(controller, needle_pose, free)
                radius = max(capsule[2] for capsule in controller.jaws)
                self.assertLessEqual(offset, expected + 1e-9)
                self.assertGreaterEqual(
                    offset, exact_offset(controller, needle_pose, free, missed(radius)) - radius / 64 - 1e-7
                )
                for capsule in controller.jaws:
                    self.assertGreaterEqual(lowest_axis_height(controller, needle_pose, offset, capsule), capsule[2])
                wire = (straight_needle() @ rotation_of(needle_pose).T + needle_pose[:3] - pose[:3]) @ rotation_of(pose)
                self.assertGreaterEqual(wire_clearance(controller.jaw_cover, wire, WIRE_RADIUS, half), 0.0)
                # One capsule alone would be clear nearer to the tissue: the first of each jaw by 0.3 mm or more.
                first = [next(row for row in controller.jaws if row[3] == sense) for sense in (1.0, -1.0)]
                alone = SimpleNamespace(**{**vars(controller), "jaws": first})
                alone.centerline, alone.arc_axes = controller.centerline, controller.arc_axes
                self.assertGreater(exact_offset_of(alone, controller, needle_pose, free), expected + 0.0003)


def rotation_of(pose):
    return np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=float).reshape(3, 3)


def exact_offset_of(rows, controller, needle_pose, upper):
    """The closed form of ``exact_offset`` for the jaw rows of ``rows`` and the grasp of ``controller``."""

    def clear(offset):
        return min(lowest_axis_height(controller, needle_pose, offset, capsule) - capsule[2] for capsule in rows.jaws)

    low, high = WIRE_RADIUS, upper
    if clear(high) >= 0:
        return high
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if clear(middle) >= 0 else (low, middle)
    return low


class DoneLineTests(unittest.TestCase):
    def test_the_example_names_the_end_of_the_procedure_once(self):
        from newton_solver_suturebot.examples.grasp import example_fem_needle_grasp as module

        example = object.__new__(module.Example)
        example.args = SimpleNamespace(num_envs=1)
        example.config = SimpleNamespace(frame_dt=1.0 / 60.0, action_dimension=7)
        time = [0.0]

        def step(actions, **options):
            time[0] += example.config.frame_dt
            return [{"diagnostics": {"physical_time_s": time[0]}}]

        example.pool = SimpleNamespace(step=step, frame_unfinished=False)
        example.observations = [{"diagnostics": {"physical_time_s": 0.0}}]
        example.halted = example.render_scene = example.last_phases = None
        example.sim_time, example.completed, example.frame_budget = 0.0, 0, None
        example.viewer = SimpleNamespace()
        state = {"stage": "tip_lift", "phase": "lowering"}
        controller = lambda observations: np.zeros((1, 7))  # noqa: E731
        controller.report = lambda: [{"env_id": 0, "reason": None, **state}]
        example.controller = controller
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            example.step()
            # The controller's phase at the end is "holding", as while it waits at the end of the arc.
            state.update(stage="done", phase="holding")
            for _ in range(5):
                example.step()
        lines = output.getvalue().splitlines()
        self.assertEqual(lines, ["Procedural insertion: 0: lowering", "Procedural insertion: 0: done"])
        # Holding at the end of the arc, before the lifts, is still named as the controller names it.
        state.update(stage="arc", phase="holding")
        with contextlib.redirect_stdout(output):
            example.step()
        self.assertEqual(output.getvalue().splitlines()[-1], "Procedural insertion: 0: holding")


if __name__ == "__main__":
    unittest.main()
