"""Lifts after the insertion, and the regrasp at the needle tip between them.

These are controller contracts: what the tool is commanded to do and when.
Tissue deformation, a needle that stays in the tissue and a grasp that holds
still require the coupled physical demonstration; target motion is not that proof.
"""

import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import warp as wp
from test_grasp_procedure import fixture, snapshot
from test_grasp_tip_clearance import TROCAR, flat_tissue, tool

from newton_solver_suturebot.examples.grasp import _grasp_procedure as procedure
from newton_solver_suturebot.examples.grasp._dvrk_grasper import GRASP_BITE_DEPTH_M, needle_to_gripper_pose
from newton_solver_suturebot.examples.grasp._dvrk_rcm_grasper import arm_link_poses
from newton_solver_suturebot.examples.grasp._grasp_procedure import (
    ProceduralGraspController,
    gripper_target,
    jaw_clearance,
    pose_at,
)

SPAN = 2.375
EXIT_ANGLE = 0.9
# Arc angle at which the drive of the lift fixture ends: the needle centred between entry and exit.
DRIVE_ANGLE = 0.5 * EXIT_ANGLE + 0.5 * SPAN
WIRE_RADIUS = 0.00028


def wire_point(plan, behind):
    """Needle-frame point of the wire ``behind`` radians of arc behind the tip."""
    basis = np.asarray(plan["local_basis"], dtype=float)
    return plan["arc_center_local"] + plan["radius"] * (np.cos(behind) * basis[:, 0] - np.sin(behind) * basis[:, 1])


def lift_fixture(*, normal=(0, 0, 1), out=0.008):
    """The procedure fixture with what a lift needs: entry normal, exit angle and the needle arc.

    For the lift at the tip it also has the jaw capsules of the tool and a flat tissue surface: with the
    needle at the end of its drive, the wire comes out of it at a right angle, ``out`` [m] behind the tip.
    """
    geometry, config = fixture()
    plan = geometry["insertion_plan"]
    plan["exit_angle"] = EXIT_ANGLE
    # Where the wire leaves the tissue, and the direction of the wire there towards the tip.
    needle = wp.transform(*pose_at(plan, DRIVE_ANGLE))
    behind = out / plan["radius"]
    basis = np.asarray(plan["local_basis"], dtype=float)
    exit_point = np.asarray(wp.transform_point(needle, wp.vec3(*wire_point(plan, behind))), dtype=float)
    outward = np.asarray(
        wp.transform_vector(needle, wp.vec3(*(np.sin(behind) * basis[:, 0] + np.cos(behind) * basis[:, 1]))),
        dtype=float,
    )
    across = np.cross(outward, [0.0, 0.0, 1.0])
    across /= np.linalg.norm(across)
    vertices, triangles = flat_tissue()
    # The patch is made in the plane z = 0 with the tissue below: its axes go to across, up and outward.
    vertices = vertices @ np.stack((across, np.cross(outward, across), outward)) + exit_point
    geometry.update(
        entry_normal=np.asarray(normal, dtype=float),
        needle_span=SPAN,
        needle_tip_local=wire_point(plan, 0.0),
        # Tip first, as NeedleGeometry.centerline.
        needle_centerline=np.array([wire_point(plan, behind) for behind in np.linspace(0.0, SPAN, 49)]),
        needle_radius=WIRE_RADIUS,
        grippers=[tool()],
        reference_positions=vertices,
        reference_triangles=triangles,
    )
    return geometry, config


def regrasp_fixture(*, out=0.006):
    """The lift fixture for a whole regrasp: a needle that stands as a U in a flat tissue, both ends out.

    The tissue is the plane z = 0 with the tissue below it. At the end of its drive the needle lies in the
    vertical plane y = -0.04 with its middle lowest, and the wire leaves the tissue ``out`` [m] of arc
    behind the tip and as far before the tail: the tip is ``radius (cos(span / 2 - out / radius) -
    cos(span / 2))`` above the tissue (4.5 mm for 6 mm). The trocar of the tool is 0.26 m above the tissue.
    """
    geometry, config = lift_fixture()
    plan = geometry["insertion_plan"]
    radius = plan["radius"]
    # At the drive angle the middle of the needle, half a span behind the tip, points straight down.
    turn = -0.5 * np.pi - DRIVE_ANGLE + 0.5 * SPAN
    plan["radial"] = np.array([np.cos(turn), 0.0, np.sin(turn)])
    plan["tangent"] = np.array([-np.sin(turn), 0.0, np.cos(turn)])
    plan["axis"] = np.array([0.0, -1.0, 0.0])
    plan["center"] = np.array([0.03, -0.04, radius * np.cos(0.5 * SPAN - out / radius)])
    vertices, triangles = flat_tissue(centre=(0.03, -0.04))
    geometry.update(entry_normal=np.array([0.0, 0.0, 1.0]), reference_positions=vertices, reference_triangles=triangles)
    return geometry, config


def held(item, geometry, behind):
    """Put the grasp of a snapshot on the wire ``behind`` radians of arc behind the tip."""
    item["needle_relative_pose"] = np.r_[-wire_point(geometry["insertion_plan"], behind), 0.0, 0.0, 0.0, 1.0]
    item["gripper_target_pose"] = gripper_target(item["needle_pose"], item["needle_relative_pose"])
    return item


def gripped(item, geometry, behind, twist=-1.7):
    """Put the grasp of a snapshot on the wire as the jaws hold it: the wire across them at their bite point.

    The bite point of the jaws, 6 mm along them, is on the wire ``behind`` radians of arc behind the tip;
    the tool is twisted about the wire by ``twist`` [rad] (``needle_to_gripper_pose``). With the regrasp
    fixture and the default twist the tool lies nearly flat over the tissue beside the tail of the needle,
    its jaws across the plane of the needle and one of them below the other.
    """
    plan = geometry["insertion_plan"]
    shape = SimpleNamespace(centerline=geometry["needle_centerline"], arc_normal=np.asarray(plan["local_basis"])[:, 2])
    palm = needle_to_gripper_pose(shape, arc_fraction=behind / SPAN, tangent_roll_rad=twist)
    item["needle_relative_pose"] = np.asarray(wp.transform_inverse(wp.transform(*palm)), dtype=float)
    item["gripper_target_pose"] = gripper_target(item["needle_pose"], item["needle_relative_pose"])
    return item


def tick(controller, item, config):
    """One frame of a tool whose target follows the action exactly."""
    item["diagnostics"]["physical_time_s"] += config.frame_dt
    action = controller([item])[0]
    pose = item["gripper_target_pose"]
    pose[:3] += config.translation_step * action[:3]
    rotation = config.rotation_step * action[3:6].astype(float)
    angle = float(np.linalg.norm(rotation))
    if angle > 0:
        turn = wp.quat_from_axis_angle(wp.vec3(*(rotation / angle)), angle)
        pose[3:] = np.asarray(turn * wp.quat(*pose[3:]), dtype=float)
    opening = item["jaw_target_rad"][0] - item["jaw_target_rad"][1] + config.jaw_step_rad * float(action[6])
    item["jaw_target_rad"] = np.array([opening / 2, -opening / 2])
    return action


def at_drive_angle(*, height=0.0005, tip_height=0.0, behind=2.0, normal=(0, 0, 1), out=0.008):
    """A controller whose commanded arc is complete, with the needle not yet in the tissue."""
    geometry, config = lift_fixture(normal=normal, out=out)
    controller = ProceduralGraspController(
        geometry, config, settle_seconds=0, lift_height=height, tip_lift_height=tip_height
    )
    item = held(snapshot(geometry), geometry, behind)
    controller([item])
    item["diagnostics"]["physical_time_s"] = 30.0
    controller([item])
    return geometry, config, controller, item


def before_regrasp(*, out=0.006, behind=2.0, twist=-1.7):
    """A controller of the regrasp fixture whose first lift has begun, with the tool where it holds the needle.

    The needle is at the end of its drive and the tool target is the grasp of that needle: the anchor of
    the lift, and so the pose at which the jaws will let go, is the tool on the wire.
    """
    geometry, config = regrasp_fixture(out=out)
    controller = ProceduralGraspController(
        geometry, config, settle_seconds=0, lift_height=0.001, tip_lift_height=0.0005
    )
    item = gripped(snapshot(geometry), geometry, behind, twist)
    controller([item])
    item["diagnostics"]["physical_time_s"] = 30.0
    controller([item])
    item["needle_pose"] = pose_at(geometry["insertion_plan"], controller.report()[0]["commanded_angle_rad"])
    item["gripper_target_pose"] = gripper_target(item["needle_pose"], item["needle_relative_pose"])
    item["diagnostics"].update(entry_count=1, physical_time_s=30.1)
    controller([item])
    return geometry, config, controller, item


def started_lift(**options):
    """As ``at_drive_angle``, one call after the first entry: the lift has its anchor."""
    geometry, config, controller, item = at_drive_angle(**options)
    # The actual accepted tool target need not equal the ideal arc target.
    item["gripper_target_pose"][:3] += [0.001, -0.002, 0.0003]
    anchor = item["gripper_target_pose"].copy()
    item["diagnostics"].update(entry_count=1, physical_time_s=30.1)
    controller([item])
    return geometry, config, controller, item, anchor


def world_point(pose, local):
    return np.asarray(wp.transform_point(wp.transform(*pose), wp.vec3(*local)), dtype=float)


def world_axis(pose, local):
    return np.asarray(wp.quat_rotate(wp.quat(*pose[3:]), wp.vec3(*local)), dtype=float)


class GraspLiftTests(unittest.TestCase):
    def assert_orientation_held(self, action, config):
        # action_to uses float32 quaternion multiplication even for identical
        # orientations. Bound the angular error in radians, before action scaling.
        self.assertLessEqual(np.linalg.norm(action[3:6]) * config.rotation_step, 4 * np.finfo(np.float32).eps)

    def test_drive_continues_until_the_needle_is_centred_but_keeps_the_jaws_clear_of_the_entry(self):
        radius = fixture()[0]["insertion_plan"]["radius"]
        finish = fixture()[0]["insertion_plan"]["finish_angle"]
        centred = 0.5 * EXIT_ANGLE + 0.5 * SPAN
        clearance = ProceduralGraspController.entry_clearance / radius
        # (arc between tip and jaws, commanded angle where the drive stops)
        cases = (
            (2.0, centred),  # jaws far back: both needle ends are outside the tissue
            (1.5, 1.5 - clearance),  # jaws would reach the entry first: stop short of it
            (0.5, finish),  # never less than the planned arc
        )
        for behind, expected in cases:
            with self.subTest(behind=behind):
                _geometry, _config, c, _item = at_drive_angle(behind=behind)
                report = c.report()[0]
                # The jaw position comes from a float32 transform inverse.
                self.assertAlmostEqual(report["commanded_angle_rad"], expected, delta=1e-6)
                self.assertEqual(report["phase"], "holding")
                self.assertTrue(report["commanded_arc_complete"])
        # Without a lift the arc ends where the plan ends.
        geometry, config = lift_fixture()
        c = ProceduralGraspController(geometry, config, settle_seconds=0)
        item = held(snapshot(geometry), geometry, 2.0)
        c([item])
        item["diagnostics"]["physical_time_s"] = 30.0
        c([item])
        self.assertEqual(c.report()[0]["commanded_angle_rad"], finish)

    def test_lift_starts_after_the_arc_and_only_with_an_entry_and_a_bilateral_grasp(self):
        _geometry, _config, c, item = at_drive_angle()
        # The commanded arc is complete, but the needle never entered the tissue.
        for time in (30.5, 31.0):
            item["diagnostics"]["physical_time_s"] = time
            c([item])
            self.assertFalse(c.report()[0]["lift_started"])
            self.assertEqual(c.report()[0]["phase"], "holding")
        item["diagnostics"]["entry_count"] = 1
        item["jaw_contact_count"][0] = 0
        item["diagnostics"]["physical_time_s"] = 31.1
        np.testing.assert_array_equal(c([item]), 0)
        self.assertFalse(c.report()[0]["lift_started"])
        item["jaw_contact_count"][0] = 1
        item["diagnostics"]["physical_time_s"] = 31.2
        c([item])
        self.assertTrue(c.report()[0]["lift_started"])
        self.assertEqual(c.report()[0]["stage"], "lift")
        # An entry during the arc does not start the lift before the arc is complete.
        geometry, config = lift_fixture()
        c = ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.0005)
        item = held(snapshot(geometry), geometry, 2.0)
        item["diagnostics"]["entry_count"] = 1
        c([item])
        item["diagnostics"]["physical_time_s"] = 2.0
        c([item])
        self.assertEqual(c.report()[0]["phase"], "inserting")
        self.assertFalse(c.report()[0]["lift_started"])

    def test_normalized_lift_anchors_observed_target_and_reports_smooth_phases(self):
        geometry, config, c, item, anchor = started_lift(normal=(0, 3, 4))
        angle = c.report()[0]["commanded_angle_rad"]
        # Changing caller-owned geometry after construction cannot rotate a lift.
        geometry["entry_normal"][:] = [1, 0, 0]
        self.assertEqual(c.rise_seconds, 0.75)
        for elapsed, phase, height in (
            (0.375, "lifting", 0.00025),
            (1.0, "lift_hold", 0.0005),
            (1.625, "lowering", 0.00025),
        ):
            item["diagnostics"]["physical_time_s"] = 30.1 + elapsed
            action = c([item])[0]
            report = c.report()[0]
            self.assertEqual(report["phase"], phase)
            self.assertAlmostEqual(report["commanded_lift_m"], height, delta=1e-10)
            self.assertEqual(report["commanded_angle_rad"], angle)
            self.assertFalse(report["lift_complete"])
            expected = np.clip(height * np.array([0, 0.6, 0.8]) / config.translation_step, -1, 1)
            np.testing.assert_allclose(action[:3], expected, atol=2e-6)
            self.assert_orientation_held(action, config)
            np.testing.assert_array_equal(item["gripper_target_pose"], anchor)

    def test_rise_time_keeps_a_ten_millimetre_lift_within_the_rate_limit(self):
        _geometry, config, c, item, anchor = started_lift(height=0.01)
        # Peak speed of a smoothstep rise is 1.5 h / T; T puts it at 80 % of the rate limit.
        limit = config.translation_step / config.frame_dt
        self.assertAlmostEqual(c.rise_seconds, 1.5 * 0.01 / (0.8 * limit), delta=1e-12)
        self.assertAlmostEqual(c.lift_duration, 2 * c.rise_seconds + c.lift_hold_seconds, delta=1e-12)
        # The first frame brings the target back from the arc target to the anchor.
        tick(c, item, config)
        peak, fastest = 0.0, 0.0
        for _ in range(int(np.ceil(c.lift_duration / config.frame_dt)) + 2):
            action = tick(c, item, config)
            fastest = max(fastest, float(np.max(np.abs(action[:3]))))
            peak = max(peak, item["gripper_target_pose"][2] - anchor[2])
            self.assert_orientation_held(action, config)
        self.assertLess(fastest, 0.81)
        self.assertAlmostEqual(peak, 0.01, delta=1e-7)
        np.testing.assert_allclose(item["gripper_target_pose"], anchor, atol=1e-7)
        report = c.report()[0]
        self.assertTrue(report["lift_complete"])
        self.assertEqual(report["lifts_complete"], 1)
        self.assertEqual(report["stage"], "done")
        self.assertEqual(report["phase"], "holding")
        self.assertAlmostEqual(report["commanded_lift_m"], 0, delta=1e-12)
        # A needle that stays embedded must not launch a second lift.
        for _ in range(30):
            np.testing.assert_allclose(tick(c, item, config), 0, atol=1e-6)
        self.assertEqual(c.report()[0]["lifts_complete"], 1)

    def test_contact_loss_pauses_lift_clock_and_original_stop_remains(self):
        _geometry, _config, c, item, _anchor = started_lift()
        item["diagnostics"]["physical_time_s"] = 30.3
        c([item])
        before = c.report()[0]
        item["jaw_contact_count"][:] = 0
        item["diagnostics"]["physical_time_s"] = 30.35
        np.testing.assert_array_equal(c([item]), 0)
        self.assertEqual(c.report()[0]["phase"], "grasp_lost")
        item["jaw_contact_count"][:] = 1
        item["diagnostics"]["physical_time_s"] = 30.5
        c([item])
        self.assertEqual(c.report()[0]["commanded_lift_m"], before["commanded_lift_m"])
        item["diagnostics"]["physical_time_s"] = 30.55
        c([item])
        self.assertGreater(c.report()[0]["commanded_lift_m"], before["commanded_lift_m"])
        item["jaw_contact_count"][:] = 0
        item["diagnostics"]["physical_time_s"] = 30.6
        c([item])
        item["diagnostics"]["physical_time_s"] = 30.85
        np.testing.assert_array_equal(c([item]), 0)
        self.assertEqual(c.report()[0]["phase"], "stopped")
        self.assertFalse(c.report()[0]["lift_complete"])
        self.assertEqual(c.report()[0]["lifts_complete"], 0)

    def run_first_lift(self, c, item, config):
        for _ in range(int(np.ceil(c.lift_duration / config.frame_dt)) + 3):
            tick(c, item, config)
        self.assertEqual(c.report()[0]["lifts_complete"], 1)

    def release(self, c, item, config):
        """Run the first lift and the release; returns after the call that starts the retreat."""
        self.run_first_lift(c, item, config)
        self.assertEqual(c.report()[0]["stage"], "release")
        while item["jaw_target_rad"][0] - item["jaw_target_rad"][1] < c.release_opening_rad - 1e-9:
            self.assertNotEqual(c.report()[0]["phase"], "stopped")
            action = tick(c, item, config)
            # Release: the jaws open, the tool does not move.
            np.testing.assert_array_equal(action[:6], 0)
            self.assertGreater(action[6], 0)
            self.assertEqual(c.report()[0]["phase"], "release")
        # Jaws that still touch the needle wait.
        np.testing.assert_array_equal(tick(c, item, config), 0)
        self.assertEqual(c.report()[0]["stage"], "release")
        item["jaw_contact_count"][:] = 0
        item["jaw_contact_normal_force_n"][:] = 0
        tick(c, item, config)
        self.assertEqual(c.report()[0]["stage"], "retreat")

    def ticks(self, c, item, config, going, limit=600):
        """Tick while ``going()`` holds; the procedure must not stop and the loop must end."""
        for _ in range(limit):
            if not going():
                return
            tick(c, item, config)
            self.assertNotEqual(c.report()[0]["phase"], "stopped", c.report()[0]["reason"])
        self.fail("The procedure did not get there")

    def transfer(self, c, item, config, *, until="close", moving=None):
        """Tick to the stage ``until``; per frame the stage, the part of it, the tool target and the jaw opening."""
        frames = []
        for _ in range(3000):
            report, state = c.report()[0], c.scenes[0]
            if report["stage"] == until or report["phase"] == "stopped":
                break
            if moving is not None:
                moving(len(frames), report, state)
            action = tick(c, item, config)
            self.assertEqual(action[6], 0)
            frames.append(
                (
                    c.report()[0]["stage"],
                    c.scenes[0].leg,
                    item["gripper_target_pose"].copy(),
                    float(item["jaw_target_rad"][0] - item["jaw_target_rad"][1]),
                )
            )
        return frames

    @staticmethod
    def capsules(c, geometry, pose, opening):
        """By hand: the axes of the tool's capsules in the world for a palm pose, with their contact radii.

        The jaws are the palm turned about its Z axis by half the opening each way; the wrist and the shaft
        follow from the palm and the trocar (``arm_link_poses``, checked against the joint frames in
        test_grasp_regrasp_path). The axes in the frames of the bodies are the rows of ``tool_capsules``.
        """
        half = 0.5 * opening
        roll, pitch = arm_link_poses(TROCAR, pose)
        frames = {"roll": roll, "pitch": pitch}
        for name, sign in (("jaw1", 1.0), ("jaw2", -1.0)):
            turned = wp.transform(*pose) * wp.transform(
                wp.vec3(0.0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), sign * half)
            )
            frames[name] = np.asarray(turned, dtype=float)
        rows = []
        for (start, end), radius, link in zip(c.tool_ends, c.tool_reach, c.tool_links, strict=True):
            rows.append((world_point(frames[link], start), world_point(frames[link], end), radius))
        return rows

    def clearances(self, c, geometry, needle, pose, opening):
        """By hand: how far the capsules are from the tissue (the plane z = 0) and from the wire, least first [m]."""
        wire = np.array([world_point(needle, point) for point in geometry["needle_centerline"]])
        fine = np.concatenate([np.linspace(a, b, 9)[:-1] for a, b in zip(wire[:-1], wire[1:], strict=True)])
        tissue, gap = np.inf, np.inf
        for start, end, radius in self.capsules(c, geometry, pose, opening):
            tissue = min(tissue, min(start[2], end[2]) - radius)
            axis = np.linspace(start, end, 41)
            gap = min(gap, np.linalg.norm(axis[:, None, :] - fine[None, :, :], axis=2).min() - radius - WIRE_RADIUS)
        return tissue, gap

    def test_regrasp_leaves_the_needle_rises_turns_and_comes_down_on_the_tip_clear_of_the_tissue(self):
        geometry, config, c, item = before_regrasp()
        plan = geometry["insertion_plan"]
        needle = item["needle_pose"].copy()
        # The tool lets go where it holds the needle: beside the tail, its lower jaw in the tissue.
        released = item["gripper_target_pose"].copy()
        self.release(c, item, config)
        np.testing.assert_allclose(item["gripper_target_pose"], released, atol=1e-9)
        tissue, gap = self.clearances(c, geometry, needle, released, c.release_opening_rad)
        self.assertLess(tissue, -0.0005)
        self.assertGreater(gap, 0.0)
        self.assertIsNone(c.report()[0]["tip_grasp_offset_m"])

        frames = self.transfer(c, item, config)
        self.assertNotIn(c.report()[0]["phase"], ("grasp_lost", "stopped"))
        stages = [stage for index, (stage, *_rest) in enumerate(frames) if index == 0 or stage != frames[index - 1][0]]
        self.assertEqual(stages, ["retreat", "approach", "descend", "close"])
        legs = [leg for index, (_s, leg, *_r) in enumerate(frames) if index == 0 or leg != frames[index - 1][1]]
        self.assertEqual(legs, ["leave", "rise", "traverse", "down"])
        self.assertAlmostEqual(frames[-1][3], c.release_opening_rad, delta=1e-9)
        state = c.scenes[0]
        via, turn, above, pregrasp, grasp = state.via, state.turn, state.above, state.pregrasp, state.tip_grasp

        # Leave: straight back along the axis of the open jaw along which the tool rises faster to its trocar.
        half = 0.5 * c.release_opening_rad
        axes = [world_axis(released, (-sign * np.sin(half), np.cos(half), 0.0)) for sign in (1.0, -1.0)]
        wrist = arm_link_poses(TROCAR, released)[0][:3]
        up = (TROCAR[:3] - wrist) / np.linalg.norm(TROCAR[:3] - wrist)
        back = -max(axes, key=lambda axis: float(-axis @ up))
        leave = [pose for _stage, leg, pose, _opening in frames if leg == "leave"]
        for pose in leave:
            self.assertLess(np.linalg.norm(np.cross(pose[:3] - released[:3], back)), 1e-8)
            np.testing.assert_allclose(pose[3:], released[3:], atol=1e-7)
        np.testing.assert_allclose(via, leave[-1], atol=1e-12)
        length = float((via[:3] - released[:3]) @ back)
        self.assertGreater(length, 0.0)
        self.assertLess(length, c.transfer_clearance)
        self.assertAlmostEqual(c.report()[0]["retreat_m"], length, delta=1e-9)
        # It ends where the tool is clear of the tissue and of the wire, and no earlier: one step before,
        # a capsule is within what the rule's samples can miss of the tissue (a sixteenth of its radius),
        # or within a quarter of a wire radius of touching the wire.
        tissue, gap = self.clearances(c, geometry, needle, via, c.release_opening_rad)
        self.assertGreaterEqual(tissue, 0.0)
        self.assertGreaterEqual(gap, 0.0)
        before = leave[-1 - sum(np.allclose(pose, via, atol=1e-12) for pose in leave)]
        tissue, gap = self.clearances(c, geometry, needle, before, c.release_opening_rad)
        self.assertTrue(tissue < c.tool_reach.max() / 16 or gap < 0.25 * WIRE_RADIUS + 2e-5, (tissue, gap))

        # Rise: straight up the shaft from there, to the first height at which tissue and wire are further
        # than the turning radius from the straight way of the palm to the point above the pregrasp.
        wrist = arm_link_poses(TROCAR, via)[0][:3]
        up = (TROCAR[:3] - wrist) / np.linalg.norm(TROCAR[:3] - wrist)
        rise = [pose for _stage, leg, pose, _opening in frames if leg == "rise"]
        for pose in rise:
            self.assertLess(np.linalg.norm(np.cross(pose[:3] - via[:3], up)), 1e-8)
            np.testing.assert_allclose(pose[3:], released[3:], atol=1e-7)
        np.testing.assert_allclose(turn, rise[-1], atol=1e-12)
        self.assertAlmostEqual(c.report()[0]["rise_m"], np.linalg.norm(turn[:3] - via[:3]), delta=1e-9)
        radius = c.turning_radius
        wire = np.array([world_point(needle, point) for point in geometry["needle_centerline"]])
        fine = np.concatenate([np.linspace(a, b, 9)[:-1] for a, b in zip(wire[:-1], wire[1:], strict=True)])

        def room(first, second):
            """By hand: the tissue and the wire from the palm's straight way, less the turning radius [m]."""
            way = np.linspace(first[:3], second[:3], 201)
            return (
                way[:, 2].min() - radius,
                np.linalg.norm(way[:, None, :] - fine[None, :, :], axis=2).min() - WIRE_RADIUS - radius,
            )

        tissue, gap = room(turn, above)
        self.assertGreaterEqual(tissue, 0.0)
        self.assertGreaterEqual(gap, 0.0)
        # One step lower the way is not clear by the rule: within the cover of its samples of the tissue
        # (the triangles of the fixture have legs of 1 mm) or of the wire (a quarter of a wire radius).
        lower = next(pose for pose in reversed(rise) if not np.allclose(pose, turn, atol=1e-12))
        tissue, gap = room(lower, above)
        self.assertTrue(tissue < 0.00075 or gap < 0.25 * WIRE_RADIUS + 2e-5, (tissue, gap))

        # Approach: the palm goes straight to the point above the pregrasp while the tool turns to the grasp,
        # then down the jaws of the grasp to the pregrasp. Here the pregrasp has room to turn: no part "down".
        traverse = [pose for _stage, leg, pose, _opening in frames if leg == "traverse"]
        direction = (above[:3] - turn[:3]) / np.linalg.norm(above[:3] - turn[:3])
        for pose in traverse:
            self.assertLess(np.linalg.norm(np.cross(pose[:3] - turn[:3], direction)), 1e-8)
        np.testing.assert_allclose(
            above[:3], grasp[:3] - c.report()[0]["above_m"] * world_axis(grasp, (0, 1, 0)), atol=1e-9
        )
        self.assertAlmostEqual(c.report()[0]["above_m"], c.transfer_clearance, delta=1e-9)
        np.testing.assert_allclose(
            pregrasp[:3], grasp[:3] - c.transfer_clearance * world_axis(grasp, (0, 1, 0)), atol=1e-9
        )
        # (The target of these tests is turned frame by frame without being made a unit quaternion again.)
        reached = next(pose for stage, _leg, pose, _opening in frames if stage == "descend")
        np.testing.assert_allclose(np.abs(reached[3:] @ grasp[3:]) / np.linalg.norm(reached[3:]), 1, atol=1e-9)
        # Descent: straight along the jaws of the grasp, from the transfer clearance.
        descent = [pose for stage, _leg, pose, _opening in frames if stage == "descend"]
        for pose in descent:
            self.assertLess(np.linalg.norm(np.cross(pose[:3] - grasp[:3], world_axis(grasp, (0, 1, 0)))), 2e-7)
        np.testing.assert_allclose(frames[-1][2][:3], grasp[:3], atol=2e-7)

        # From the end of the way back to the grasp no capsule of the tool is in the tissue, and none touches
        # the wire before the jaws have come down on it: by hand, in every frame.
        clear = [index for index, (_stage, leg, _pose, _opening) in enumerate(frames) if leg != "leave"]
        for index in clear[:: max(1, len(clear) // 150)] + [clear[-1]]:
            stage, _leg, pose, opening = frames[index]
            tissue, gap = self.clearances(c, geometry, needle, pose, opening)
            self.assertGreaterEqual(tissue, 0.0, (index, stage))
            if stage != "descend":
                self.assertGreaterEqual(gap, 0.0, (index, stage))

        # The bite point of the jaws sits on the wire, the planned length behind the tip.
        offset = c.report()[0]["tip_grasp_offset_m"]
        bite = world_point(grasp, (0, 0.006, 0))
        local = np.asarray(wp.transform_point(wp.transform_inverse(wp.transform(*needle)), wp.vec3(*bite)), dtype=float)
        centre = local - plan["arc_center_local"]
        self.assertAlmostEqual(np.linalg.norm(centre), plan["radius"], delta=1e-5)
        self.assertAlmostEqual(np.arctan2(-centre[1], centre[0]), offset / plan["radius"], delta=2e-3)
        # That point is on the wire that sticks out (6 mm here), and it is the nearest to the tissue at which
        # the jaws stay out of it while they come down and close: clear there, not clear a jaw radius nearer.
        self.assertGreater(offset, WIRE_RADIUS)
        self.assertLess(offset, 0.006 - WIRE_RADIUS)
        opening, descent = 0.5 * c.release_opening_rad, c.transfer_clearance
        self.assertGreaterEqual(jaw_clearance(c.tissue, c.jaws, grasp, (0, 0.006, 0), opening, descent), 0.0)
        nearer = c._tip_grasp(needle, offset + c.jaws[0][2])
        self.assertLess(jaw_clearance(c.tissue, c.jaws, nearer, (0, 0.006, 0), opening, descent), 0.0)

        # Close: the jaws close without moving the tool; the lift waits for contact on both.
        while item["jaw_target_rad"][0] - item["jaw_target_rad"][1] > 1e-9:
            action = tick(c, item, config)
            np.testing.assert_array_equal(action[:6], 0)
            self.assertLess(action[6], 0)
        self.assertEqual(c.report()[0]["stage"], "close")
        item["jaw_contact_count"][:] = 1
        item["jaw_contact_normal_force_n"][:] = 0.2
        item["needle_relative_pose"] = np.asarray(
            wp.transform_multiply(wp.transform_inverse(wp.transform(*grasp)), wp.transform(*needle)), dtype=float
        )
        tick(c, item, config)
        self.assertEqual(c.report()[0]["stage"], "tip_lift")

        # The lift by the tip uses its own height and returns to where the jaws closed.
        peak = 0.0
        for _ in range(int(np.ceil(c.lift_duration / config.frame_dt)) + 3):
            tick(c, item, config)
            peak = max(peak, item["gripper_target_pose"][2] - grasp[2])
        self.assertAlmostEqual(peak, 0.0005, delta=1e-7)
        np.testing.assert_allclose(item["gripper_target_pose"][:3], grasp[:3], atol=2e-7)
        report = c.report()[0]
        self.assertEqual((report["stage"], report["lifts_complete"], report["phase"]), ("done", 2, "holding"))
        self.assertEqual(report["tip_grasp_plans"], 1)

    def test_the_grasp_at_the_tip_is_planned_while_the_tool_rises_one_piece_of_work_per_frame(self):
        geometry, config, c, item = before_regrasp()
        self.release(c, item, config)
        state = c.scenes[0]
        # While the tool backs away nothing is planned: the needle may still be moving with the tissue.
        for _ in range(400):
            if state.leg != "leave":
                break
            tick(c, item, config)
            self.assertNotEqual(c.report()[0]["phase"], "stopped")
            if state.leg == "leave":
                self.assertIsNone(state.plan)
                self.assertIsNone(state.plan_pose)
            self.assertIsNone(c.report()[0]["tip_grasp_offset_m"])
        self.assertEqual(state.leg, "rise")
        # Rising, the tool does not wait for the plan: the looks at the surface, the points of the search
        # and its six halvings, and the way down, a piece per frame.
        frames, moved = 0, 0
        while not state.planned:
            before = item["gripper_target_pose"].copy()
            tick(c, item, config)
            moved += bool(np.linalg.norm(item["gripper_target_pose"][:3] - before[:3]) > 1e-6)
            self.assertNotIn(c.report()[0]["phase"], ("grasp_lost", "stopped"))
            frames += 1
            self.assertLess(frames, 120)
        self.assertGreater(frames, 12)
        self.assertEqual(moved, frames)
        self.assertEqual(c.report()[0]["stage"], "retreat")
        self.assertGreater(c.report()[0]["tip_grasp_offset_m"], WIRE_RADIUS)
        np.testing.assert_allclose(state.plan_pose, item["needle_pose"], atol=1e-12)
        # The tool turns towards the tip only with the plan and from where it can turn.
        self.ticks(c, item, config, lambda: c.report()[0]["stage"] == "retreat")
        self.assertEqual(c.report()[0]["stage"], "approach")
        self.assertGreater(np.linalg.norm(state.turn[:3] - state.via[:3]), 0.005)

    def test_a_needle_that_moves_while_the_plan_is_made_is_planned_again_and_stops_the_procedure_if_it_never_rests(
        self,
    ):
        geometry, config, c, item = before_regrasp()
        self.release(c, item, config)
        state = c.scenes[0]
        self.ticks(c, item, config, lambda: state.plan is None)
        # The needle comes up by 0.2 mm during the first plan (more than the finest spacing, 0.061 mm) and
        # then rests: the first plan is dropped and the second one, made for the needle at rest, stands.
        first = state.plan_pose.copy()
        item["needle_pose"][2] += 0.0002
        frames = 0
        while not state.planned:
            tick(c, item, config)
            frames += 1
            self.assertLess(frames, 160)
        self.assertGreater(frames, 24)
        np.testing.assert_allclose(state.plan_pose[:3] - first[:3], [0, 0, 0.0002], atol=1e-12)
        np.testing.assert_allclose(
            state.tip_grasp, c._tip_grasp(item["needle_pose"], c.report()[0]["tip_grasp_offset_m"]), atol=1e-12
        )
        # A plan that stands is given up as well when the needle moves while the tool still rises; that is
        # not a plan made again at the pregrasp.
        if c.report()[0]["stage"] == "retreat":
            item["needle_pose"][2] += 0.0001
            tick(c, item, config)
            self.assertFalse(state.planned)
        self.ticks(c, item, config, lambda: c.report()[0]["stage"] == "retreat")
        self.assertEqual(c.report()[0]["stage"], "approach")
        np.testing.assert_allclose(state.plan_pose[:3] - first[:3], [0, 0, 0.0003], atol=1e-12)
        self.assertEqual(c.report()[0]["tip_grasp_plans"], 1)

        # A needle that keeps moving: no plan stands, and 3 s after the first one was begun the procedure stops.
        geometry, config, c, item = before_regrasp()
        self.release(c, item, config)
        for _ in range(int(5.0 / config.frame_dt)):
            item["needle_pose"][2] += 0.00001
            tick(c, item, config)
            if c.report()[0]["phase"] == "stopped":
                break
        report = c.report()[0]
        self.assertEqual((report["stage"], report["phase"]), ("retreat", "stopped"))
        self.assertIn("did not come to rest", report["reason"])
        self.assertIsNone(report["tip_grasp_offset_m"])

    def test_a_needle_that_has_moved_at_the_pregrasp_is_planned_again_once(self):
        geometry, config, c, item = before_regrasp()
        self.release(c, item, config)
        state = c.scenes[0]
        self.transfer(c, item, config, until="approach")
        first, offset = state.tip_grasp.copy(), c.report()[0]["tip_grasp_offset_m"]
        # On the way over, the needle sinks by 0.1 mm: more than the plan allows (0.061 mm). At the pregrasp
        # the tool does not come down: it plans again where it stands and goes to the new pregrasp.
        item["needle_pose"][2] -= 0.0001
        frames = self.transfer(c, item, config, until="descend")
        self.assertEqual(c.report()[0]["stage"], "descend")
        self.assertEqual(c.report()[0]["tip_grasp_plans"], 2)
        stages = [stage for index, (stage, *_rest) in enumerate(frames) if index == 0 or stage != frames[index - 1][0]]
        self.assertEqual(stages, ["approach", "retreat", "approach", "descend"])
        np.testing.assert_allclose(state.plan_pose, item["needle_pose"], atol=1e-12)
        np.testing.assert_allclose(
            state.tip_grasp, c._tip_grasp(item["needle_pose"], c.report()[0]["tip_grasp_offset_m"]), atol=1e-12
        )
        self.assertLess(c.report()[0]["tip_grasp_offset_m"], offset)
        self.assertGreater(np.linalg.norm(state.tip_grasp[:3] - first[:3]), 2e-5)
        np.testing.assert_allclose(frames[-2][2][:3], state.pregrasp[:3], atol=2e-7)
        # A needle within the tolerance is left alone: the tool comes down.
        item["needle_pose"][2] += 0.00003
        self.transfer(c, item, config)
        self.assertEqual(c.report()[0]["stage"], "close")
        self.assertEqual(c.report()[0]["tip_grasp_plans"], 2)

        # A needle that has moved again at the second pregrasp stops the procedure.
        geometry, config, c, item = before_regrasp()
        self.release(c, item, config)
        self.transfer(c, item, config, until="approach")
        item["needle_pose"][2] -= 0.0001
        self.transfer(c, item, config, until="retreat")
        self.transfer(c, item, config, until="approach")
        item["needle_pose"][2] -= 0.0001
        self.transfer(c, item, config)
        report = c.report()[0]
        self.assertEqual((report["stage"], report["phase"]), ("approach", "stopped"))
        self.assertIn("did not stay at rest", report["reason"])
        np.testing.assert_array_equal(tick(c, item, config), 0)

    def test_a_tip_that_does_not_stick_out_far_enough_stops_the_procedure(self):
        # 3 mm of wire out of the tissue at either end: the tip is 2.6 mm above it. The tool holds the tail
        # further up, where it can leave it.
        geometry, config, c, item = before_regrasp(out=0.003, behind=2.3)
        self.release(c, item, config)
        for _ in range(300):
            action = tick(c, item, config)
            if c.report()[0]["phase"] == "stopped":
                break
        report = c.report()[0]
        self.assertEqual((report["stage"], report["phase"]), ("retreat", "stopped"))
        self.assertIn("does not stick out", report["reason"])
        self.assertIsNone(report["tip_grasp_offset_m"])
        np.testing.assert_array_equal(action, 0)
        np.testing.assert_array_equal(tick(c, item, config), 0)

    def test_a_way_that_cannot_be_kept_clear_stops_the_procedure_with_its_reason(self):
        # The jaws across the tail with the wire deep between them and the tool flat on the tissue: within
        # the transfer clearance no way back along an open jaw frees the rise of tissue and wire.
        geometry, config, c, item = before_regrasp(behind=2.2, twist=-0.5 * np.pi)
        self.release(c, item, config)
        released = item["gripper_target_pose"].copy()
        frames = self.transfer(c, item, config)
        report = c.report()[0]
        self.assertEqual((report["stage"], report["phase"]), ("retreat", "stopped"))
        self.assertIn("cannot leave the needle and the tissue", report["reason"])
        self.assertAlmostEqual(np.linalg.norm(frames[-1][2][:3] - released[:3]), c.transfer_clearance, delta=2e-7)
        np.testing.assert_array_equal(tick(c, item, config), 0)

        # A shaft whose slide ends 3 mm above where the tool has left the needle: the tool cannot rise to
        # where it can turn (the turning radius is 13.3 mm, the palm is 4 mm above the tissue).
        geometry, config, c, item = before_regrasp()
        self.release(c, item, config)
        state = c.scenes[0]
        self.ticks(c, item, config, lambda: state.leg == "leave")
        wrist = arm_link_poses(TROCAR, state.via)[0][:3]
        c.shaft_shortest = float(np.linalg.norm(TROCAR[:3] - wrist)) - 0.003
        state.rise = c._rise(state.via)
        self.assertAlmostEqual(np.linalg.norm(state.rise), 0.003, delta=1e-12)
        frames = self.transfer(c, item, config)
        report = c.report()[0]
        self.assertEqual((report["stage"], report["phase"]), ("retreat", "stopped"))
        self.assertIn("cannot rise far enough", report["reason"])
        self.assertAlmostEqual(np.linalg.norm(frames[-1][2][:3] - state.via[:3]), 0.003, delta=2e-7)

        # A tool that does not say where its shaft and wrist are cannot be kept out of the tissue: it is
        # planned a grasp for as before (test_grasp_tip_clearance), and the regrasp stops before it moves.
        geometry, config = regrasp_fixture()
        del geometry["grippers"][0]["trocar_pose"]
        c = ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001, tip_lift_height=0.0005)
        item = gripped(snapshot(geometry), geometry, 2.0)
        c([item])
        for time in (30.0, 30.1):
            item["diagnostics"].update(entry_count=1, physical_time_s=time)
            c([item])
        self.run_first_lift(c, item, config)
        np.testing.assert_array_equal(tick(c, item, config), 0)
        report = c.report()[0]
        self.assertEqual((report["stage"], report["phase"]), ("release", "stopped"))
        self.assertIn("trocar", report["reason"])

    def test_jaws_that_do_not_close_on_the_tip_stop_the_procedure(self):
        _geometry, config, c, item = before_regrasp()
        self.run_first_lift(c, item, config)
        item["jaw_contact_count"][:] = 0
        item["jaw_contact_normal_force_n"][:] = 0
        for _ in range(3000):
            if c.report()[0]["stage"] == "close":
                break
            tick(c, item, config)
        self.assertEqual(c.report()[0]["stage"], "close")
        for _ in range(int(3.0 / config.frame_dt) + 3):
            tick(c, item, config)
        report = c.report()[0]
        self.assertEqual(report["phase"], "stopped")
        self.assertIn("did not close", report["reason"])
        self.assertEqual(report["lifts_complete"], 1)
        np.testing.assert_array_equal(tick(c, item, config), 0)

    def test_reset_clears_only_its_world_lift_and_allows_a_new_trigger(self):
        geometry, config = lift_fixture()
        c = ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.0005)
        items = [held(snapshot(geometry, env_id=i), geometry, 2.0) for i in range(2)]
        c(items)
        for time in (30.0, 30.1, 30.3):
            for item in items:
                item["diagnostics"].update(entry_count=1, physical_time_s=time)
            c(items)
        self.assertTrue(all(row["lift_started"] for row in c.report()))
        neighbor = deepcopy(c.report()[1])
        items[0] = held(snapshot(geometry, env_id=0, reset=1), geometry, 2.0)
        c(items)
        self.assertFalse(c.report()[0]["lift_started"])
        self.assertFalse(c.report()[0]["lift_complete"])
        self.assertEqual(c.report()[0]["stage"], "arc")
        self.assertEqual(c.report()[1], neighbor)
        for time in (30.0, 30.1):
            items[0]["diagnostics"].update(entry_count=1, physical_time_s=time)
            c(items)
        self.assertTrue(c.report()[0]["lift_started"])
        self.assertEqual(c.report()[1], neighbor)

    def test_zero_default_preserves_actions_and_arc_history_without_a_normal(self):
        geometry, config = fixture()
        original = ProceduralGraspController(geometry, config, settle_seconds=0)
        explicit = ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.0)
        item = snapshot(geometry)
        item["diagnostics"].update(entry_count=1, exit_count=0)
        for time in (0, 0.2, 0.75, 2, 4):
            item["diagnostics"]["physical_time_s"] = time
            np.testing.assert_array_equal(original([item]), explicit([item]))
            self.assertEqual(original.report(), explicit.report())
            self.assertFalse(original.report()[0]["lift_started"])
        self.assertGreater(original.report()[0]["commanded_angle_rad"], geometry["insertion_plan"]["initial_angle"])
        self.assertEqual(original.lift_duration, 0.0)

    def test_every_choice_of_the_procedure_has_one_name_its_value_and_a_row_in_the_table(self):
        # The numbers that are not derived, as they are in use. A change of one is a change of the
        # commanded procedure and needs its own proof; the table at the top of the module says what
        # chose each of them.
        module = {
            "RAMP_SECONDS": 0.5,
            "REACHED_STEPS": 1e-3,
            "CLEARANCE_SPACING_RADII": 0.25,
            "CLEARANCE_REFINEMENT": 4,
            "PLAN_CAPSULES_PER_STEP": 1,
            "ROUTE_CAPSULES_PER_STEP": 4,
            "SEARCH_HALVINGS": 6,
            "ABOVE_SEARCH_RADII": 2.0,
            "WIRE_STATION_RADII": 0.5,
        }
        controller = {
            "lift_rise_seconds": 0.75,
            "lift_hold_seconds": 0.5,
            "lift_height_limit": 0.01,
            "rate_fraction": 0.8,
            "contact_force_n": 1e-5,
            "contact_loss_seconds": 0.25,
            "grasp_wait_seconds": 2.0,
            "stage_wait_seconds": 3.0,
            "entry_clearance": 0.003,
            "transfer_clearance": 0.012,
            "release_opening_rad": 0.6,
            "tip_grasp_roll_rad": -2.618,
            "tip_grasp_bite_rad": 0.7854,
            "tip_grasp_jaws_exchanged": False,
            "pull_grasp_roll_rad": -1.0472,
            "pull_grasp_bite_rad": 0.5236,
            "pull_grasp_jaws_exchanged": True,
            "pull_slack_radii": 4.0,
            "draw_needle_lengths": 1.0,
            "tighten_travel_bound": 0.0075,
        }
        for name, value in module.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(procedure, name), value)
                self.assertIn(f"``{name}``", procedure.__doc__)
        for name, value in controller.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(ProceduralGraspController, name), value)
                self.assertIn(f"``{name}``", procedure.__doc__)
        # The bite point of every grasp lives with the grasp pose and has its row as well.
        self.assertEqual(GRASP_BITE_DEPTH_M, 0.006)
        self.assertIn("``GRASP_BITE_DEPTH_M``", procedure.__doc__)
        # Two numbers that follow from the choices are written as what they follow from.
        self.assertEqual(ProceduralGraspController.pull_wait_seconds, ProceduralGraspController.contact_loss_seconds)
        self.assertEqual(procedure.WIRE_REACH_RADII, 1.5)
        # The least tissue volume at which a tied thread is tight is the thread/tissue coupling's, shared
        # with the suture example; the table names it and the controller takes it from there, unless the
        # scene publishes one for its own tissue (``tight_volume_ratio``), which the table names as well.
        from newton_solver_suturebot._src.solvers.thread.tissue import TIGHT_VOLUME_RATIO

        self.assertEqual(TIGHT_VOLUME_RATIO, 0.5)
        self.assertIn("``TIGHT_VOLUME_RATIO``", procedure.__doc__)
        self.assertIn("``tight_volume_ratio``", procedure.__doc__)
        self.assertIn("``STITCH_TIGHT_VOLUME_RATIO``", procedure.__doc__)
        # The defaults of the arguments are the two choices that the example offers as options.
        geometry, config = fixture()
        default = ProceduralGraspController(geometry, config)
        self.assertEqual((default.speed, default.settle_seconds), (0.003, 0.5))
        # The rounded radians of the two grasps at the tip are the degrees of their scans.
        angles = np.rad2deg([-2.618, 0.7854, -1.0472, 0.5236])
        np.testing.assert_allclose(angles, [-150.0, 45.0, -60.0, 30.0], atol=5e-4)

    def test_height_and_positive_lift_normal_validation(self):
        geometry, config = lift_fixture()
        for height in (-0.0001, 0.010001, float("nan"), float("inf")):
            with self.subTest(height=height), self.assertRaises(ValueError):
                ProceduralGraspController(geometry, config, lift_height=height)
        # The lift at the tip cannot exceed the lift after the insertion.
        for tip in (-0.0001, 0.002001, float("nan")):
            with self.subTest(tip=tip), self.assertRaises(ValueError):
                ProceduralGraspController(geometry, config, lift_height=0.002, tip_lift_height=tip)
        for normal in (None, [0, 0], [0, 0, 0], [0, 0, float("nan")], [0, float("inf"), 1]):
            invalid = deepcopy(geometry)
            if normal is None:
                del invalid["entry_normal"]
            else:
                invalid["entry_normal"] = normal
            with self.subTest(normal=normal), self.assertRaises(ValueError):
                ProceduralGraspController(invalid, config, lift_height=0.0005)
        geometry["entry_normal"] = [0, 0, 2]
        ProceduralGraspController(geometry, config, lift_height=0.01, tip_lift_height=0.01)


if __name__ == "__main__":
    unittest.main()
