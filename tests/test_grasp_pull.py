"""The pull of the needle by its tip: joints, reach, kinematics and end rule against computations by hand.

After the regrasp the tool can turn the needle on about the needle's own centre of curvature. The grasp at the
tip is then one from which the tool reaches that arc, and the plan takes only points whose descent, closing and
pull the tool reaches with every joint inside its limits. Nothing here steps physics: the tool target follows
the action exactly and the needle follows the tool rigidly, so every angle and length can be written down.
"""

import inspect
import re
import unittest

import numpy as np
import warp as wp
from test_grasp_lift import DRIVE_ANGLE, EXIT_ANGLE, SPAN, gripped, regrasp_fixture, wire_point
from test_grasp_lift import tick as lift_tick
from test_grasp_procedure import fixture, snapshot
from test_grasp_regrasp_path import FAR, stepped
from test_grasp_tip_clearance import (
    TROCAR,
    WIRE_RADIUS,
    flat_tissue,
    planned,
    pose_from,
    straight_needle,
    straight_pose,
    tool,
)

from newton_solver_suturebot._src.sim.ik.pose_ik import NativePoseIK
from newton_solver_suturebot.examples.grasp import _grasp_procedure as procedure
from newton_solver_suturebot.examples.grasp._dvrk_rcm_grasper import WRIST_LENGTH_M, _arm_pose, arm_joint_positions
from newton_solver_suturebot.examples.grasp._grasp_procedure import (
    CLEARANCE_REFINEMENT,
    CLEARANCE_SPACING_RADII,
    ProceduralGraspController,
    gripper_target,
    line_targets,
    line_to,
    pose_at,
    pull_targets,
    ramp_distance,
    ramp_speed,
)

NOMINAL = 0.18
# The limits of the dVRK tool of the scene (add_rcm_grasper with the slide of the tool of these tests).
LIMITS = np.array([[-0.6, -0.6, -0.05, -np.pi, -1.1, -1.1], [0.6, 0.6, 0.05, np.pi, 1.1, 1.1]])
# A trocar a quarter of a metre above the needle of the fixtures, with its Z axis pointing down at the tissue:
# the joints of the tool are counted from a shaft along that axis.
DOWN = np.r_[TROCAR[:3], 1.0, 0.0, 0.0, 0.0]
# What the scene publishes beside the limits (GraspScene.grasp_geometry): the tolerances of the pose IK of a tool
# [m, rad] and the radius of the hole that the needle makes [m], two wire radii.
IK_TOLERANCES = {"ik_position_tolerance_m": 1.0e-5, "ik_orientation_tolerance_rad": 1.0e-3}
HOLE = 2.0 * WIRE_RADIUS


def pose(transform):
    return np.asarray(transform, dtype=float)


def times(first, second):
    return pose(wp.transform_multiply(wp.transform(*first), wp.transform(*second)))


def inverse(value):
    return pose(wp.transform_inverse(wp.transform(*value)))


def palm_for(joints, trocar=DOWN):
    """Palm pose of the fixed-trocar tool for its joints, from the joint frames of the tool model."""
    return times(trocar, pose(_arm_pose(joints, NOMINAL)))


def pulling_tool(limits=LIMITS, trocar=DOWN, **published):
    return dict(
        tool(),
        trocar_pose=np.array(trocar, dtype=float),
        arm_joint_limits=np.array(limits, dtype=float),
        **(IK_TOLERANCES | published),
    )


def trocar_behind(grasp, distance=0.19):
    """A trocar ``distance`` [m] behind a tool pose along its jaws, with its Z axis along them towards the tool.

    The tool then stands nearly straight at that pose: the shaft along the jaws, the wrist joints and the
    joints about the trocar near zero, the slide near ``distance`` less the wrist length and the nominal
    insertion (0.19 - 0.0091 - 0.18 = 0.9 mm).
    """
    along = procedure._rotation(grasp)[:, 1]
    turn = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*along))
    return np.r_[grasp[:3] - distance * along, np.asarray(turn, dtype=float)]


def pull_controller(limits=LIMITS, trocar=DOWN, *, thread_element=None, hole=HOLE, **options):
    """A controller that pulls, on the regrasp fixture: the needle as a U in flat tissue, both ends out.

    With ``thread_element`` [m] the scene says that its thread follows the needle's track through the tissue
    and gives the length of a thread element, as ``GraspScene.grasp_geometry`` does; ``hole`` [m] is the
    radius of the needle's hole that the scene publishes.
    """
    geometry, config = regrasp_fixture()
    geometry["grippers"] = [pulling_tool(limits, trocar)]
    geometry["needle_capture_radius"] = hole
    if thread_element is not None:
        geometry.update(thread_tracks=True, thread_element_length=thread_element)
    return (
        geometry,
        config,
        ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001, pull_through=True, **options),
    )


def closed_on_the_tip(
    limits=LIMITS, *, offset=0.002, entry_arc=0.00299, exit_arc=0.012, thread_element=None, hole=HOLE
):
    """The pull fixture at the call in which the jaws have closed on the needle ``offset`` [m] behind its tip.

    The entry marker lies ``entry_arc`` [m] of arc ahead of the tail on the needle's own circle and the exit
    marker ``exit_arc``; the scene of the controller stands in its stage ``close`` with both jaws on the wire.
    The trocar of the tool is behind that grasp (:func:`trocar_behind`).
    """
    geometry, config = regrasp_fixture()
    needle = pose_at(geometry["insertion_plan"], DRIVE_ANGLE)
    grasp = pull_controller()[2]._tip_grasp(needle, offset)
    geometry, config, c = pull_controller(limits, trocar_behind(grasp), thread_element=thread_element, hole=hole)
    plan = geometry["insertion_plan"]
    item = snapshot(geometry)
    item.update(
        needle_pose=needle.copy(),
        needle_relative_pose=times(inverse(grasp), needle),
        gripper_target_pose=grasp.copy(),
        entry_position=pose(
            wp.transform_point(wp.transform(*needle), wp.vec3(*wire_point(plan, SPAN - entry_arc / plan["radius"])))
        ),
        exit_position=pose(
            wp.transform_point(wp.transform(*needle), wp.vec3(*wire_point(plan, SPAN - exit_arc / plan["radius"])))
        ),
    )
    state = procedure._Scene(0, 0.0, 0.0, DRIVE_ANGLE)
    state.grasp, state.stage, state.had_contact = item["needle_relative_pose"].copy(), "close", True
    state.reach_rad = state.reach_m = np.inf
    c.scenes[0] = state
    return geometry, config, c, item, state


def tick(c, item, config, *, follow=True, contact=True):
    """One frame: the tool target follows the action exactly; the needle follows the tool as it lies in the jaws."""
    item["diagnostics"]["physical_time_s"] += config.frame_dt
    item["jaw_contact_count"] = np.full(2, int(contact))
    item["jaw_contact_normal_force_n"] = np.full(2, 0.2 if contact else 0.0)
    action = c([item])[0]
    item["gripper_target_pose"] = stepped(item["gripper_target_pose"], action, config)
    if follow:
        item["needle_pose"] = times(item["gripper_target_pose"], item["needle_relative_pose"])
    return action


class ArmJointTests(unittest.TestCase):
    def test_joints_of_poses_that_can_be_written_down(self):
        trocar = np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        # All joints zero: the wrist link lies along the shaft, so the palm is the nominal insertion and the
        # wrist length along Z from the trocar; the two fixed turns of the wrist take the palm's X axis to
        # Y, its Y axis to Z and its Z axis to X.
        turned = procedure._pose(np.r_[0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(procedure._rotation(turned), [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-12)
        straight = np.r_[0.0, 0.0, NOMINAL + WRIST_LENGTH_M, turned[3:]]
        np.testing.assert_allclose(arm_joint_positions(trocar, straight, NOMINAL), np.zeros(6), atol=1e-12)
        # The slide alone moves the palm along the shaft.
        further = straight.copy()
        further[2] += 0.02
        np.testing.assert_allclose(arm_joint_positions(trocar, further, NOMINAL), [0, 0, 0.02, 0, 0, 0], atol=1e-12)
        # The yaw about the trocar alone swings the straight tool about Y: by 0.3 rad the palm is at
        # (sin 0.3, 0, cos 0.3) times its distance and turned with it.
        swing = np.r_[0.0, np.sin(0.15), 0.0, np.cos(0.15)]
        swung = times(np.r_[0.0, 0.0, 0.0, swing], straight)
        reach = NOMINAL + WRIST_LENGTH_M
        np.testing.assert_allclose(swung[:3], [reach * np.sin(0.3), 0.0, reach * np.cos(0.3)], atol=1e-7)
        np.testing.assert_allclose(arm_joint_positions(trocar, swung, NOMINAL), [0.3, 0, 0, 0, 0, 0], atol=1e-6)
        # The roll alone turns the palm about the shaft and leaves its origin where it is.
        rolled = times(np.r_[0.0, 0.0, 0.0, 0.0, 0.0, np.sin(0.4), np.cos(0.4)], straight)
        np.testing.assert_allclose(rolled[:3], straight[:3], atol=1e-7)
        np.testing.assert_allclose(arm_joint_positions(trocar, rolled, NOMINAL), [0, 0, 0, 0.8, 0, 0], atol=1e-6)

    def test_joints_are_those_of_the_joint_frames_one_pose_or_many(self):
        rng = np.random.default_rng(11)
        joints = LIMITS[0] + (LIMITS[1] - LIMITS[0]) * rng.random((200, 6))
        palms = np.array([palm_for(q) for q in joints])
        # The joint frames multiply in single precision.
        np.testing.assert_allclose(arm_joint_positions(DOWN, palms, NOMINAL), joints, atol=3e-6)
        for q, palm in zip(joints[:20], palms[:20], strict=True):
            np.testing.assert_allclose(arm_joint_positions(DOWN, palm, NOMINAL), q, atol=3e-6)
        with self.assertRaisesRegex(ValueError, "finite"):
            arm_joint_positions(DOWN, np.r_[np.nan, palms[0][1:]], NOMINAL)

    def test_the_other_grasp_with_the_jaws_exchanged_is_the_mirror_solution_of_the_wrist(self):
        # A palm turned half a turn about its jaw axis (Y) through its own origin: the roll joint half a
        # turn on, the wrist pitch and the wrist yaw reversed, the trocar joints and the slide the same.
        half_turn = np.r_[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        for q in ([0.1, -0.2, 0.01, 0.5, 0.4, -0.3], [-0.3, 0.2, -0.02, -2.0, -0.6, 0.7]):
            mirrored = arm_joint_positions(DOWN, times(palm_for(q), half_turn), NOMINAL)
            roll = q[3] + np.pi if q[3] < 0 else q[3] - np.pi
            np.testing.assert_allclose(mirrored, [q[0], q[1], q[2], roll, -q[4], -q[5]], atol=3e-6)


class ReachTests(unittest.TestCase):
    def test_the_margin_follows_from_the_tolerances_the_creep_of_the_needle_and_the_wrist(self):
        _geometry, _config, c = pull_controller(LIMITS)
        # Rest tolerance e: a sixteenth of the smallest jaw contact radius (0.976 mm here).
        rest = CLEARANCE_SPACING_RADII * 0.000976 / CLEARANCE_REFINEMENT
        self.assertAlmostEqual(c._rest_tolerance(), rest, delta=1e-15)
        # Chord of the needle of the fixture: 2 R sin(span / 2).
        chord = 2.0 * 0.00945 * np.sin(0.5 * SPAN)
        turn = 1.0e-3 + 4.0 * rest / chord
        shift = 1.0e-5 + 2.0 * rest
        rotary = (turn + shift / (0.18 - 0.05)) / np.cos(1.1)
        expected = [rotary, rotary, shift + 0.0091 * rotary, rotary, rotary, rotary]
        np.testing.assert_allclose(c.reach_margin, expected, rtol=1e-9)
        # With the numbers of the dVRK tools (smallest jaw radius 0.646 mm, chord 17.5 mm): 0.024 rad, 0.31 mm.
        rest = 0.25 * 0.000646 / 4
        rotary = (1.0e-3 + 4.0 * rest / 0.0175 + (1.0e-5 + 2.0 * rest) / 0.13) / np.cos(1.1)
        self.assertAlmostEqual(rotary, 0.0241, delta=5e-5)
        self.assertAlmostEqual(1.0e-5 + 2.0 * rest + 0.0091 * rotary, 0.00031, delta=5e-6)
        # The tolerances of these fixtures are those that the scene's pose IK has unless it is given others.
        defaults = inspect.signature(NativePoseIK.__init__).parameters
        self.assertEqual(c.ik_position_tolerance, defaults["position_tolerance_m"].default)
        self.assertEqual(c.ik_orientation_tolerance, defaults["orientation_tolerance_rad"].default)

    def test_the_margin_follows_the_tolerances_that_the_scene_publishes_for_the_tool(self):
        """Fails if the controller keeps tolerances of its own instead of those of the scene's pose IK."""
        geometry, config = regrasp_fixture()
        geometry["needle_capture_radius"] = HOLE
        # A scene whose pose IK is three times as tight in position and half as tight in orientation.
        geometry["grippers"] = [pulling_tool(ik_position_tolerance_m=3.0e-6, ik_orientation_tolerance_rad=2.0e-3)]
        c = ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001, pull_through=True)
        self.assertEqual((c.ik_position_tolerance, c.ik_orientation_tolerance), (3.0e-6, 2.0e-3))
        rest = CLEARANCE_SPACING_RADII * 0.000976 / CLEARANCE_REFINEMENT
        chord = 2.0 * 0.00945 * np.sin(0.5 * SPAN)
        rotary = (2.0e-3 + 4.0 * rest / chord + (3.0e-6 + 2.0 * rest) / (0.18 - 0.05)) / np.cos(1.1)
        expected = [rotary, rotary, 3.0e-6 + 2.0 * rest + 0.0091 * rotary, rotary, rotary, rotary]
        np.testing.assert_allclose(c.reach_margin, expected, rtol=1e-9)
        # 1 mrad more of orientation tolerance is 1 mrad / cos(1.1) = 2.2 mrad more of margin.
        usual = pull_controller(LIMITS)[2].reach_margin
        self.assertAlmostEqual(c.reach_margin[0] - usual[0], (1.0e-3 - 7.0e-6 / 0.13) / np.cos(1.1), delta=1e-12)
        # A tool without them, or with a tolerance that is no length or angle, is refused.
        for missing in IK_TOLERANCES:
            entry = pulling_tool()
            del entry[missing]
            geometry["grippers"] = [entry]
            with self.assertRaisesRegex(ValueError, missing):
                ProceduralGraspController(geometry, config, lift_height=0.001, pull_through=True)
        for value in (-1.0e-5, float("nan"), float("inf")):
            geometry["grippers"] = [pulling_tool(ik_position_tolerance_m=value)]
            with self.assertRaisesRegex(ValueError, "tolerances of the pose IK"):
                ProceduralGraspController(geometry, config, lift_height=0.001, pull_through=True)

    def test_a_row_is_reached_only_with_every_joint_inside_by_the_margin_and_by_what_it_moves(self):
        _geometry, _config, c = pull_controller(LIMITS)
        margin = c.reach_margin
        base = np.array([0.1, -0.1, 0.0, 0.5, 0.3, -0.2])

        def row(joint, values):
            joints = np.tile(base, (len(values), 1))
            joints[:, joint] = values
            return np.array([palm_for(q) for q in joints])

        # One pose: inside by more than the margin is reached, by less is not; the joint is named.
        why, rotary, slide = c._reach(row(5, [1.1 - 1.5 * margin[5]]))
        self.assertIsNone(why)
        self.assertAlmostEqual(rotary, 1.5 * margin[5], delta=3e-6)
        self.assertAlmostEqual(slide, 0.05, delta=3e-6)
        why, rotary, _slide = c._reach(row(5, [1.1 - 0.5 * margin[5]]))
        self.assertRegex(why, "^wrist_yaw is .* rad inside its limit")
        self.assertAlmostEqual(rotary, 0.5 * margin[5], delta=3e-6)
        # Beyond the limit is not reached either (the distance is then negative).
        why, rotary, _slide = c._reach(row(4, [-1.1 - 0.01]))
        self.assertRegex(why, "^wrist_pitch is -0.01")
        self.assertAlmostEqual(rotary, -0.01, delta=3e-6)
        # The slide counts in metres with its own margin.
        self.assertIsNone(c._reach(row(2, [0.05 - 1.5 * margin[2]]))[0])
        why, _rotary, slide = c._reach(row(2, [0.05 - 0.5 * margin[2]]))
        self.assertRegex(why, "^insertion is .* m inside its limit")
        self.assertAlmostEqual(slide, 0.5 * margin[2], delta=3e-6)
        # A row: a joint that moves 0.03 rad from pose to pose must be inside by the margin and 0.03 rad
        # at the poses beside that move. Ending 5 mrad further inside than that it is reached, 5 mrad
        # less it is not.
        needed = margin[3] + 0.03
        self.assertIsNone(c._reach(row(3, np.pi - (needed + 0.005) - 0.03 * np.arange(4)[::-1]))[0])
        why, rotary, _slide = c._reach(row(3, np.pi - (needed - 0.005) - 0.03 * np.arange(4)[::-1]))
        self.assertRegex(why, f"^roll is {needed - 0.005:.3g} rad inside its limit where {needed:.3g} rad are needed$")
        self.assertAlmostEqual(rotary, needed - 0.005, delta=3e-6)
        # A roll joint that would have to pass its limit between two poses is not inside: 0.2 rad from the
        # limit on either side of it.
        self.assertRegex(c._reach(row(3, [np.pi - 0.2, -np.pi + 0.2]))[0], "^roll is ")

    def test_the_pull_needs_the_limits_of_the_tool_and_excludes_the_lift_at_the_tip(self):
        geometry, config = regrasp_fixture()
        # A scene without thread tracks says how large the hole of the needle is: the pull ends there.
        for hole in (None, 0.0, -HOLE, float("nan")):
            if hole is not None:
                geometry["needle_capture_radius"] = hole
            with self.assertRaisesRegex(ValueError, "needle_capture_radius"):
                ProceduralGraspController(geometry, config, lift_height=0.001, pull_through=True)
        geometry["needle_capture_radius"] = HOLE
        with self.assertRaisesRegex(ValueError, "arm_joint_limits"):
            ProceduralGraspController(geometry, config, lift_height=0.001, pull_through=True)
        geometry["grippers"] = [pulling_tool(LIMITS)]
        with self.assertRaisesRegex(ValueError, "The pull replaces the lift at the tip"):
            ProceduralGraspController(geometry, config, lift_height=0.001, tip_lift_height=0.0005, pull_through=True)
        geometry["grippers"] = [pulling_tool(LIMITS[:, :5])]
        with self.assertRaisesRegex(ValueError, "six increasing finite joint limits"):
            ProceduralGraspController(geometry, config, lift_height=0.001, pull_through=True)
        geometry["grippers"] = [
            pulling_tool(np.array([[-0.6, -0.6, -0.05, -3.0, -1.6, -1.1], [0.6, 0.6, 0.05, 3.0, 1.6, 1.1]]))
        ]
        with self.assertRaisesRegex(ValueError, "wrist pitch of less than a quarter turn"):
            ProceduralGraspController(geometry, config, lift_height=0.001, pull_through=True)
        # Without the argument nothing of the pull is read or used, and the grasp at the tip is the earlier one.
        earlier = ProceduralGraspController(regrasp_fixture()[0], config, lift_height=0.001, tip_lift_height=0.0005)
        self.assertFalse(earlier.pull_through)
        self.assertIsNone(earlier.joint_limits)
        self.assertEqual(
            (earlier.tip_grasp_roll_rad, earlier.tip_grasp_bite_rad, earlier.tip_grasp_jaws_exchanged),
            (-2.618, 0.7854, False),
        )
        pulling = pull_controller(LIMITS)[2]
        self.assertEqual(
            (pulling.tip_grasp_roll_rad, pulling.tip_grasp_bite_rad, pulling.tip_grasp_jaws_exchanged),
            (-1.0472, 0.5236, True),
        )


class PublishedValueTests(unittest.TestCase):
    """What the controller reads for the pull is what the scene has: on the small real scene of test_grasp_scene."""

    def test_the_scene_publishes_the_tolerances_of_its_pose_ik_and_the_hole_of_its_needle(self):
        """Fails if the geometry does not carry the values of the scene's own pose IK and needle/tissue interaction."""
        from test_grasp_scene import _geometry, _scene

        with _scene() as (scene, config):
            geometry = scene.grasp_geometry()
            self.assertEqual(len(geometry["grippers"]), len(scene.pose_ik))
            for entry, ik in zip(geometry["grippers"], scene.pose_ik, strict=True):
                self.assertEqual(entry["ik_position_tolerance_m"], ik.position_tolerance_m)
                self.assertEqual(entry["ik_orientation_tolerance_rad"], ik.orientation_tolerance_rad)
            # The hole: the radius within which the interaction takes a point for an opening, two wire radii.
            self.assertEqual(geometry["needle_capture_radius"], scene.interaction.capture_radius)
            self.assertEqual(geometry["needle_capture_radius"], 2.0 * scene.geometry.thickness)
            # The controller with the pull reads the tolerances of the tool that it commands.
            c = ProceduralGraspController(_geometry(scene, config), config, lift_height=0.01, pull_through=True)
            self.assertEqual(
                (c.ik_position_tolerance, c.ik_orientation_tolerance),
                (scene.pose_ik[0].position_tolerance_m, scene.pose_ik[0].orientation_tolerance_rad),
            )
            # A scene with other values publishes those, tool by tool, and the controller follows them.
            scene.pose_ik[0].position_tolerance_m, scene.pose_ik[0].orientation_tolerance_rad = 2.0e-5, 5.0e-4
            scene.interaction.radii[:] = 0.0004
            changed = scene.grasp_geometry()
            self.assertEqual([entry["ik_position_tolerance_m"] for entry in changed["grippers"]], [2.0e-5, 1.0e-5])
            self.assertEqual([entry["ik_orientation_tolerance_rad"] for entry in changed["grippers"]], [5.0e-4, 1.0e-3])
            self.assertEqual(changed["needle_capture_radius"], 0.0008)
            other = ProceduralGraspController(_geometry(scene, config), config, lift_height=0.01, pull_through=True)
            self.assertEqual((other.ik_position_tolerance, other.ik_orientation_tolerance), (2.0e-5, 5.0e-4))
            self.assertLess(other.reach_margin[0], c.reach_margin[0])
        # Without thread tracks the pull ends at the hole of the entry: the controller has the scene's radius.
        with _scene(example_arguments=("--no-thread-tracks",)) as (scene, config):
            c = ProceduralGraspController(_geometry(scene, config), config, lift_height=0.01, pull_through=True)
            self.assertFalse(c.thread_tracks)
            self.assertEqual(c.pull_hole, scene.interaction.capture_radius)
            self.assertEqual(c.pull_hole, 2.0 * c.needle_radius)


class PullKinematicsTests(unittest.TestCase):
    def test_targets_turn_the_tool_about_the_centre_of_the_needle(self):
        radius = 0.01
        center, axis = np.array([-radius, 0.0, 0.0]), np.array([0.0, 0.0, 2.0])
        identity = np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        # Needle frame = world, the tool at the tip (the origin) with the axes of the needle: after the
        # angle phi the tool is at c + R (cos phi, sin phi, 0) and turned by phi about Z.
        angles = np.array([0.0, 0.5 * np.pi, 1.0, -0.3])
        targets = pull_targets(identity, identity, center, axis, angles)
        self.assertEqual(targets.shape, (4, 7))
        expected = center + radius * np.stack((np.cos(angles), np.sin(angles), np.zeros(4)), axis=1)
        np.testing.assert_allclose(targets[:, :3], expected, atol=1e-15)
        np.testing.assert_allclose(
            targets[1], [-radius, radius, 0.0, 0.0, 0.0, np.sin(0.25 * np.pi), np.cos(0.25 * np.pi)], atol=1e-15
        )
        np.testing.assert_allclose(
            targets[:, 3:],
            np.stack((0 * angles, 0 * angles, np.sin(0.5 * angles), np.cos(0.5 * angles)), axis=1),
            atol=1e-15,
        )
        # A tool held elsewhere on the needle keeps its distance from the centre and from the plane of the needle.
        grasp = np.r_[0.004, -0.003, 0.002, np.asarray(wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.7), dtype=float)]
        needle = np.r_[0.03, -0.04, 0.01, np.asarray(wp.quat_from_axis_angle(wp.vec3(0.6, 0.0, 0.8), 1.1), dtype=float)]
        angles = np.linspace(0.0, 2.0, 9)
        targets = pull_targets(needle, grasp, center, axis, angles)
        centre_world = procedure._rotation(needle) @ center + needle[:3]
        axis_world = procedure._rotation(needle) @ np.array([0.0, 0.0, 1.0])
        offsets = targets[:, :3] - centre_world
        np.testing.assert_allclose(offsets @ axis_world, np.full(9, offsets[0] @ axis_world), atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(offsets, axis=1), np.full(9, np.linalg.norm(offsets[0])), atol=1e-9)
        # The same as the needle turned about its centre with the tool held to it (single precision transforms).
        for angle, target in zip(angles, targets, strict=True):
            turn = np.asarray(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(angle)), dtype=float)
            about = np.r_[center - procedure._rotation(np.r_[0.0, 0.0, 0.0, turn]) @ center, turn]
            reference = gripper_target(times(needle, about), grasp)
            np.testing.assert_allclose(target[:3], reference[:3], atol=2e-8)
            self.assertAlmostEqual(abs(target[3:] @ reference[3:]), 1.0, delta=1e-6)
            # The needle that the tool then holds has its centre of curvature where it was.
            held = times(target, grasp)
            np.testing.assert_allclose(procedure._rotation(held) @ center + held[:3], centre_world, atol=2e-8)

    def test_line_targets_are_the_targets_of_the_straight_steps(self):
        config = fixture()[1]
        start = np.r_[0.01, 0.02, 0.03, np.asarray(wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.4), dtype=float)]
        axis = np.array([1.0, 2.0, 2.0]) / 3.0
        turn = wp.quat_from_axis_angle(wp.vec3(*axis), 0.8)
        goal = np.r_[start[:3] + [0.003, -0.001, 0.0005], np.asarray(turn * wp.quat(*start[3:]), dtype=float)]
        rows = line_targets(start, goal, config)
        # 3 mm at 0.1 mm per frame; 0.8 rad about (1, 2, 2) / 3 at 0.01 rad per frame of its largest component.
        self.assertEqual(len(rows), 1 + max(30, int(np.ceil(0.8 * 2.0 / 3.0 / 0.01))))
        target = start.copy()
        for row in rows:
            np.testing.assert_allclose(row[:3], target[:3], atol=1e-9)
            self.assertAlmostEqual(abs(row[3:] @ target[3:]), 1.0, delta=1e-6)
            target = stepped(target, line_to(target, goal, config), config)
        np.testing.assert_allclose(rows[-1][:3], goal[:3], atol=1e-12)
        self.assertAlmostEqual(abs(rows[-1][3:] @ goal[3:]), 1.0, delta=1e-7)
        # A way of no length is its one pose.
        self.assertEqual(len(line_targets(start, start, config)), 1)

    def test_tail_arc_past_a_marker_is_measured_on_the_circle_of_the_needle(self):
        geometry, _config, c = pull_controller()
        plan = geometry["insertion_plan"]
        radius = plan["radius"]
        needle = pose_at(plan, DRIVE_ANGLE)

        def marker(behind, out_of_plane=0.0, further=1.0):
            local = wire_point(plan, behind)
            local = plan["arc_center_local"] + further * (local - plan["arc_center_local"]) + [0.0, 0.0, out_of_plane]
            return pose(wp.transform_point(wp.transform(*needle), wp.vec3(*local)))

        # The tail is SPAN of arc behind the tip. A marker 0.4 rad of arc ahead of the tail: the tail has
        # 0.4 R to go. Beside the tail: none. 0.1 rad behind the tail: the tail is 0.1 R past it.
        self.assertAlmostEqual(c._tail_past(needle, marker(SPAN - 0.4)), -0.4 * radius, delta=1e-8)
        self.assertAlmostEqual(c._tail_past(needle, marker(SPAN)), 0.0, delta=1e-8)
        self.assertAlmostEqual(c._tail_past(needle, marker(SPAN + 0.1)), 0.1 * radius, delta=1e-8)
        # Only the angle about the axis counts: not the height above the plane, not the distance from the centre.
        self.assertAlmostEqual(c._tail_past(needle, marker(SPAN - 0.4, 0.002, 1.3)), -0.4 * radius, delta=1e-8)
        # The pull to a marker: that arc and the slack of four wire radii, as an angle.
        self.assertAlmostEqual(c._pull_span(needle, marker(SPAN - 0.4)), 0.4 + 4.0 * WIRE_RADIUS / radius, delta=1e-6)
        self.assertAlmostEqual(c._pull_span(needle, marker(SPAN + 0.1)), 4.0 * WIRE_RADIUS / radius, delta=1e-6)
        # To the hole at a marker, two wire radii of arc before it.
        self.assertAlmostEqual(
            c._pull_span(needle, marker(SPAN - 0.4), 2.0 * WIRE_RADIUS), 0.4 + 2.0 * WIRE_RADIUS / radius, delta=1e-6
        )

    def test_the_speed_towards_a_stop_is_the_ramp_of_the_insertion_run_backwards(self):
        speed, ramp_time = 0.003, 0.5
        # The ramp covers speed x ramp_time / 2 = 0.75 mm: with that much or more left the speed is whole.
        self.assertEqual(ramp_speed(0.00075, speed), speed)
        self.assertEqual(ramp_speed(0.01, speed), speed)
        self.assertEqual(ramp_speed(0.0, speed), 0.0)
        self.assertEqual(ramp_speed(-0.001, speed), 0.0)
        # Half way through its time the ramp has covered speed x ramp_time x (1/8 - 1/32) = 0.140625 mm and
        # runs at half speed (3/4 - 2/8).
        self.assertAlmostEqual(ramp_speed(0.003 * 0.5 * (0.125 - 0.03125), speed), 0.5 * speed, delta=1e-12)
        # At every time of the ramp: the speed with the covered distance left is the ramp's own speed.
        for u in np.linspace(0.05, 1.0, 20):
            covered = float(ramp_distance(u * ramp_time, speed))
            self.assertAlmostEqual(ramp_speed(covered, speed), speed * u * u * (3.0 - 2.0 * u), delta=1e-11)
            # ... which is the slope of the ramp there.
            slope = (ramp_distance(u * ramp_time + 1e-6, speed) - ramp_distance(u * ramp_time - 1e-6, speed)) / 2e-6
            self.assertAlmostEqual(ramp_speed(covered, speed), min(float(slope), speed), delta=2e-8)


class PullStageTests(unittest.TestCase):
    def assert_held(self, action, config):
        """The tool holds its target and keeps its jaws closed: the quaternion arithmetic of a pose
        difference in single precision leaves a rotation of a few float32 rounding errors."""
        np.testing.assert_array_equal(action[:3], np.zeros(3))
        self.assertLessEqual(np.linalg.norm(action[3:6]) * config.rotation_step, 4 * np.finfo(np.float32).eps)
        self.assertEqual(action[6], 0.0)

    def test_the_pull_turns_the_needle_to_the_hole_of_the_entry_at_the_speed_rule_then_holds_and_is_done(self):
        geometry, config, c, item, state = closed_on_the_tip()
        plan = geometry["insertion_plan"]
        radius, center = plan["radius"], plan["arc_center_local"]
        hole = 2.0 * WIRE_RADIUS
        start = item["needle_pose"].copy()
        centre_world = procedure._rotation(start) @ center + start[:3]
        # The call in which the jaws are closed starts the pull and moves nothing.
        action = tick(c, item, config)
        np.testing.assert_array_equal(action, np.zeros(7))
        self.assertEqual((state.stage, c.report()[0]["phase"]), ("pull", "close"))
        self.assertAlmostEqual(state.tail_past_entry, -0.00299, delta=1e-8)
        self.assertAlmostEqual(state.tail_past_exit, -0.012, delta=1e-8)
        # The tail has 2.99 mm less the radius of the hole to go; the command may go the slack further.
        self.assertAlmostEqual(state.pull_bound, (0.00299 - hole + 4.0 * WIRE_RADIUS) / radius, delta=1e-6)
        # Speed rule: the palm origin is this far from the axis of the needle (from the measured grasp).
        palm = inverse(item["needle_relative_pose"])[:3] - center
        arm = np.hypot(palm[0], palm[1])
        speed = min(0.003, 0.8 * 0.0001 * 60.0 * radius / arm, 0.8 * 0.01 * 60.0 * radius)
        self.assertAlmostEqual(state.pull_speed, speed, delta=1e-9)
        # By hand, frame by frame: the ramp of the insertion adds its arc, but no more than the speed of the
        # same ramp towards its stop allows with what the tail has left; the needle follows the tool exactly.
        travelled, frames, steps = 0.0, 0, []
        while 0.00299 - hole - travelled > 0.0:
            frames += 1
            added = float(
                ramp_distance(frames * config.frame_dt, speed) - ramp_distance((frames - 1) * config.frame_dt, speed)
            )
            added = min(added, ramp_speed(0.00299 - hole - travelled, speed) * config.frame_dt)
            steps.append(added)
            travelled += added
            self.assertLess(frames, 400)
        # It starts and ends slowly: the first and the last frame add less than a twentieth of a frame at speed.
        self.assertLess(steps[0], 0.05 * speed * config.frame_dt)
        self.assertLess(steps[-1], 0.05 * speed * config.frame_dt)
        self.assertAlmostEqual(max(steps), speed * config.frame_dt, delta=1e-12)
        # Both ramps take half a second and cover 0.75 mm each at 3 mm/s: 1.5 mm in a second where the whole
        # speed would cover 3 mm. The frames in all: those of the arc at speed and half a second more.
        # (a few frames fewer: a frame of the stop runs at the speed of its beginning.)
        if speed == 0.003:
            self.assertLessEqual(abs(frames - ((0.00299 - hole) / speed * 60.0 + 30.0)), 5.0)
        # The needle of the test follows the tool through single precision transforms, so the arc that the
        # controller reads as left differs from this hand count by nanometres: the angles agree to 5e-6 rad
        # (0.05 um of arc) and the last, nearly still frame of the stop may fall one call earlier or later.
        covered, pulled = 0.0, 0
        while True:
            action = tick(c, item, config)
            if state.stage != "pull":
                break
            pulled += 1
            self.assertEqual(c.report()[0]["phase"], "pulling")
            # Within the rate limits, so that the target is the commanded one, and the jaws stay closed.
            self.assertLess(np.max(np.abs(action[:6])), 0.81)
            self.assertEqual(action[6], 0.0)
            covered += steps[pulled - 1] if pulled <= frames else 0.0
            angle = covered / radius
            self.assertAlmostEqual(c.report()[0]["pull_angle_rad"], angle, delta=5e-6)
            # The needle has turned by the commanded angle about its own centre, in its own plane.
            angle = c.report()[0]["pull_angle_rad"]
            needle = item["needle_pose"]
            np.testing.assert_allclose(procedure._rotation(needle) @ center + needle[:3], centre_world, atol=3e-7)
            turned = np.asarray(wp.quat(*needle[3:]) * wp.quat_inverse(wp.quat(*start[3:])), dtype=float)
            turned *= np.sign(turned[3])
            np.testing.assert_allclose(turned[:3], np.sin(0.5 * angle) * plan["axis"], atol=1e-6)
            self.assertAlmostEqual(2.0 * np.arctan2(np.linalg.norm(turned[:3]), turned[3]), angle, delta=2e-6)
            self.assertLess(pulled, frames + 3)
        self.assertLessEqual(abs(pulled - frames), 1)
        # That call saw the tail at the hole: the pull has ended and the tool holds its target.
        report = c.report()[0]
        self.assertEqual((report["stage"], report["phase"], report["pull_complete"]), ("clear", "holding", True))
        self.assert_held(action, config)
        self.assertAlmostEqual(report["tail_past_entry_m"], travelled - 0.00299, delta=3e-7)
        # The tail is within the radius of the hole of the entry marker, by less than a slow frame's arc.
        self.assertGreaterEqual(report["tail_past_entry_m"], -hole - 3e-7)
        self.assertLess(report["tail_past_entry_m"], -hole + 0.05 * speed * config.frame_dt)
        self.assertAlmostEqual(report["tail_past_exit_m"], travelled - 0.012, delta=3e-7)
        # It holds for the hold time of a lift (0.5 s) and is then done; nothing moves any more.
        held = item["gripper_target_pose"].copy()
        for _ in range(29):
            tick(c, item, config)
            self.assertEqual(c.report()[0]["stage"], "clear")
        tick(c, item, config)
        self.assertEqual((c.report()[0]["stage"], c.report()[0]["phase"]), ("done", "holding"))
        for _ in range(5):
            self.assert_held(tick(c, item, config), config)
        np.testing.assert_allclose(item["gripper_target_pose"], held, atol=1e-6)
        self.assertEqual(
            (c.report()[0]["stage"], c.report()[0]["phase"], c.report()[0]["reason"]), ("done", "holding", None)
        )

    def test_the_pull_to_the_entry_ends_at_the_hole_that_the_scene_publishes(self):
        """Fails if the pull ends at a hole of the controller's own size instead of the scene's."""
        for hole in (HOLE, 1.5 * HOLE):
            with self.subTest(hole=hole):
                _geometry, config, c, item, state = closed_on_the_tip(hole=hole)
                self.assertEqual(c.pull_hole, hole)
                tick(c, item, config)
                # The tail has 2.99 mm less the radius of that hole to go, and the slack of four wire radii beyond.
                self.assertAlmostEqual(state.pull_bound, (0.00299 - hole + 4.0 * WIRE_RADIUS) / c.radius, delta=1e-6)
                for _ in range(400):
                    tick(c, item, config)
                    if state.stage != "pull":
                        break
                report = c.report()[0]
                self.assertEqual((report["stage"], report["pull_complete"]), ("clear", True))
                # Within the radius of the hole of the entry marker, by less than a slow frame's arc.
                self.assertGreaterEqual(report["tail_past_entry_m"], -hole - 3e-7)
                self.assertLess(report["tail_past_entry_m"], -hole + 0.05 * state.pull_speed * config.frame_dt)

    def test_the_report_follows_the_tail_of_the_needle_after_the_procedure_is_done(self):
        """Fails if a done procedure reports the tail where the last frame of the hold had it."""
        # The pull to the entry, and the pull through tissue that holds nothing of the track.
        for fixture_of in (closed_on_the_tip, through_the_tissue):
            with self.subTest(fixture=fixture_of.__name__):
                _geometry, config, c, item, state = fixture_of()
                item["diagnostics"]["engaged_track_length_m"] = 0.0
                for _ in range(600):
                    tick(c, item, config)
                    if state.stage == "done":
                        break
                self.assertEqual((state.stage, state.pull_complete), ("done", True))
                before = c.report()[0]
                # The needle creeps on with the tissue while the tool holds it: 0.1 mm of arc about its own
                # centre, in the sense of the insertion. Its tail is then 0.1 mm further past both markers.
                turn = np.asarray(wp.quat_from_axis_angle(wp.vec3(*c.arc_axes[2]), 0.0001 / c.radius), dtype=float)
                centre = c.arc_center_local
                about = np.r_[centre - procedure._rotation(np.r_[0.0, 0.0, 0.0, turn]) @ centre, turn]
                item["needle_pose"] = times(item["needle_pose"], about)
                action = tick(c, item, config, follow=False)
                after = c.report()[0]
                self.assertEqual((after["stage"], after["phase"]), ("done", "holding"))
                self.assertAlmostEqual(after["tail_past_entry_m"] - before["tail_past_entry_m"], 0.0001, delta=3e-8)
                self.assertAlmostEqual(after["tail_past_exit_m"] - before["tail_past_exit_m"], 0.0001, delta=3e-8)
                # The command is not changed by it: the tool holds its target.
                self.assert_held(action, config)

    def test_without_a_lift_the_tool_lets_go_of_the_centred_needle_as_soon_as_the_arc_is_complete(self):
        geometry, config = regrasp_fixture()
        geometry["grippers"] = [pulling_tool()]
        geometry["needle_capture_radius"] = HOLE
        c = ProceduralGraspController(geometry, config, settle_seconds=0, pull_through=True)
        # The drive goes on to the centred needle, as before a lift, and the entry normal is read for the plan.
        self.assertAlmostEqual(c.drive_angle, 0.5 * EXIT_ANGLE + 0.5 * SPAN, delta=1e-12)
        np.testing.assert_array_equal(c.lift_normal, [0.0, 0.0, 1.0])
        item = gripped(snapshot(geometry), geometry, 2.0)
        item.update(entry_position=np.zeros(3), exit_position=np.zeros(3))
        c([item])
        item["diagnostics"]["physical_time_s"] = 30.0
        c([item])
        report = c.report()[0]
        self.assertEqual((report["stage"], report["phase"]), ("arc", "holding"))
        self.assertAlmostEqual(report["commanded_angle_rad"], DRIVE_ANGLE, delta=1e-6)
        # An arc angle alone does not prove that the needle is in the tissue: nothing follows without an entry.
        item["diagnostics"]["physical_time_s"] = 30.1
        c([item])
        self.assertEqual(c.report()[0]["stage"], "arc")
        item["diagnostics"].update(entry_count=1, physical_time_s=30.2)
        action = c([item])[0]
        self.assertEqual((c.report()[0]["stage"], c.report()[0]["lifts_complete"]), ("release", 0))
        self.assertEqual(action[6], 0.0)
        # The next call is the release: the tool stands and its jaws open.
        item["diagnostics"]["physical_time_s"] = 30.3
        action = c([item])[0]
        np.testing.assert_array_equal(action, [0, 0, 0, 0, 0, 0, 1])
        # Without the pull and without a lift the procedure ends with the arc, as before.
        plain = ProceduralGraspController(regrasp_fixture()[0], config, settle_seconds=0)
        self.assertEqual(plain.drive_angle, plain.finish_angle)

    def test_a_palm_far_from_the_axis_slows_the_pull_to_the_rate_limit(self):
        _geometry, config, c, item, state = closed_on_the_tip()
        radius, center = c.radius, c.arc_center_local
        # The same grasp with the palm 30 mm from the axis of the needle, in the plane of the needle.
        grasp = inverse(item["needle_relative_pose"])
        grasp[:3] = center + [0.03, 0.0, 0.0]
        item["needle_relative_pose"] = inverse(grasp)
        item["gripper_target_pose"] = gripper_target(item["needle_pose"], item["needle_relative_pose"])
        tick(c, item, config)
        # 80 % of 6 mm/s at the palm is 4.8 mm/s x 9.45 / 30 at the needle.
        self.assertAlmostEqual(state.pull_speed, 0.8 * 0.006 * radius / 0.03, delta=1e-8)
        self.assertLess(state.pull_speed, 0.003)
        for _ in range(40):
            self.assertLess(np.max(np.abs(tick(c, item, config)[:6])), 0.81)

    def test_time_runs_only_while_both_jaws_hold_and_a_needle_that_does_not_follow_stops_the_pull(self):
        _geometry, config, c, item, state = closed_on_the_tip()
        tick(c, item, config)
        for _ in range(10):
            tick(c, item, config)
        angle, time = state.pull_angle, state.pull_time
        self.assertAlmostEqual(time, 10 * config.frame_dt, delta=1e-12)
        # A jaw off the needle: the command stands; with both back on it goes on from where it stood.
        for _ in range(3):
            tick(c, item, config, contact=False)
            self.assertEqual(c.report()[0]["phase"], "grasp_lost")
        self.assertEqual((state.pull_angle, state.pull_time), (angle, time))
        tick(c, item, config)
        self.assertEqual((state.pull_angle, state.pull_time), (angle, time))
        tick(c, item, config)
        self.assertAlmostEqual(state.pull_time, time + config.frame_dt, delta=1e-12)
        # A needle that stays where it is: the command stops at its bound, and 0.25 s later the procedure stops.
        geometry, config, c, item, state = closed_on_the_tip()
        bound = (0.00299 - 2.0 * WIRE_RADIUS + 4.0 * WIRE_RADIUS) / c.radius
        frames = next(
            k
            for k in range(1, 900)
            if ramp_distance(k * config.frame_dt, state.pull_speed or 0.003) / c.radius >= bound
        )
        tick(c, item, config, follow=False)
        for _ in range(frames):
            tick(c, item, config, follow=False)
            self.assertEqual(c.report()[0]["phase"], "pulling")
        self.assertAlmostEqual(state.pull_angle, bound, delta=1e-6)
        waited = 0
        while c.report()[0]["phase"] != "stopped":
            tick(c, item, config, follow=False)
            waited += 1
            self.assertLess(waited, 30)
        # The call that reaches the bound is the first of the wait; 0.25 s are 15 frames (the sum of the
        # frames' arcs reaches the bound in the frame the ramp does, or by rounding in the next one).
        self.assertIn(waited, (14, 15))
        self.assertRegex(c.report()[0]["reason"], "did not follow the pull")
        self.assertFalse(c.report()[0]["pull_complete"])
        self.assertAlmostEqual(c.report()[0]["pull_angle_rad"], bound, delta=1e-6)

    def test_a_pull_that_the_tool_cannot_reach_from_where_the_jaws_closed_does_not_start(self):
        # The joints of the tool over the arc of the pull, from the measured grasp, by the joint frames.
        _geometry, _config, c, item, state = closed_on_the_tip()
        span = (0.00299 - 2.0 * WIRE_RADIUS + 4.0 * WIRE_RADIUS) / c.radius
        targets = pull_targets(
            item["needle_pose"],
            item["needle_relative_pose"],
            c.arc_center_local,
            c.arc_axes[2],
            np.linspace(0.0, span, 60),
        )
        joints = arm_joint_positions(c.trocar, targets, NOMINAL)
        moving = int(np.argmax(np.ptp(joints[:, [0, 1, 3, 4, 5]], axis=0)))
        joint = [0, 1, 3, 4, 5][moving]
        self.assertGreater(np.ptp(joints[:, joint]), 0.1)
        # A limit of that joint in the middle of what the arc asks of it: the start is inside, the arc is not.
        limits = LIMITS.copy()
        middle = 0.5 * (joints[:, joint].min() + joints[:, joint].max())
        rising = joints[-1, joint] > joints[0, joint]
        limits[1 if rising else 0, joint] = middle
        geometry, config, c, item, state = closed_on_the_tip(limits)
        target = item["gripper_target_pose"].copy()
        action = tick(c, item, config)
        report = c.report()[0]
        self.assertEqual((report["stage"], report["phase"]), ("close", "stopped"))
        name = ("rcm_yaw", "rcm_pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw")[joint]
        self.assertRegex(report["reason"], f"cannot pull the needle on from where its jaws closed .*: {name} is ")
        np.testing.assert_array_equal(action[:6], np.zeros(6))
        np.testing.assert_array_equal(item["gripper_target_pose"], target)
        # With the limit beyond what the arc asks, by more than the margin and a step, it starts.
        limits[1 if rising else 0, joint] = joints[:, joint].max() + 0.1 if rising else joints[:, joint].min() - 0.1
        geometry, config, c, item, state = closed_on_the_tip(limits)
        tick(c, item, config)
        self.assertEqual(state.stage, "pull")
        self.assertLess(state.reach_rad, np.inf)


# The pull through the tissue of these tests: the thread element of the scenes (0.18 m in 90 elements), the exit
# marker 3 mm of arc ahead of the tail and the entry marker 1 mm, and tissue that holds the needle's track until
# the tail is level with the exit marker.
ELEMENT = 0.002
EXIT_ARC, ENTRY_ARC = 0.003, 0.001


def through_the_tissue(**options):
    geometry, config, c, item, state = closed_on_the_tip(
        entry_arc=ENTRY_ARC, exit_arc=EXIT_ARC, thread_element=ELEMENT, **options
    )
    item["diagnostics"]["engaged_track_length_m"] = 0.004
    return geometry, config, c, item, state


class PullThroughTests(unittest.TestCase):
    def test_with_thread_tracks_the_tail_goes_through_the_tissue_and_one_thread_element_past_the_exit(self):
        """Fails if the pull stops at the entry, ends while the tissue holds the track or before the tail is clear."""
        geometry, config, c, item, state = through_the_tissue()
        self.assertTrue(c.thread_tracks)
        self.assertEqual(c.pull_clearance, ELEMENT)
        radius = geometry["insertion_plan"]["radius"]
        action = tick(c, item, config)
        np.testing.assert_array_equal(action, np.zeros(7))
        self.assertEqual(state.stage, "pull")
        # The command may go to two elements past the exit marker: the arc to it, the clearance and as much again.
        self.assertAlmostEqual(state.pull_bound, (EXIT_ARC + 2.0 * ELEMENT) / radius, delta=1e-6)
        speed = state.pull_speed
        self.assertAlmostEqual(speed, 0.003, delta=1e-9)
        # By hand: the arc to go is that to the exit marker and one element more; the stop ramp of the
        # insertion begins when that much is left, and the tissue lets go of the track when the tail is at the marker.
        need = EXIT_ARC + ELEMENT
        travelled, frames, steps = 0.0, 0, []
        while need - travelled > 0.0:
            frames += 1
            added = float(
                ramp_distance(frames * config.frame_dt, speed) - ramp_distance((frames - 1) * config.frame_dt, speed)
            )
            added = min(added, ramp_speed(need - travelled, speed) * config.frame_dt)
            steps.append(added)
            travelled += added
            self.assertLess(frames, 400)
        self.assertLess(steps[-1], 0.05 * speed * config.frame_dt)
        pulled, past_entry, held_past_marker = 0, False, 0
        while True:
            # The tissue of the test: it holds the track until the tail is level with the exit marker.
            item["diagnostics"]["engaged_track_length_m"] = 0.004 if state.tail_past_exit < 0.0 else 0.0
            before = state.pull_angle
            tick(c, item, config)
            if state.stage != "pull":
                break
            pulled += 1
            report = c.report()[0]
            self.assertEqual(report["phase"], "pulling")
            past_entry = past_entry or report["tail_past_entry_m"] > 0.0005
            if item["diagnostics"]["engaged_track_length_m"] > 0.0:
                held_past_marker += 1
                # While the track is held the stop ramp has not begun: after the start ramp the frame's arc is whole.
                if pulled > 31:
                    self.assertAlmostEqual((state.pull_angle - before) * radius, speed * config.frame_dt, delta=1e-9)
            self.assertLess(pulled, frames + 4)
        # The tail went past the entry marker without stopping there, and the frames are those of the hand count.
        self.assertTrue(past_entry)
        self.assertGreater(held_past_marker, 30)
        self.assertLessEqual(abs(pulled - frames), 1)
        report = c.report()[0]
        self.assertEqual((report["stage"], report["phase"], report["pull_complete"]), ("clear", "holding", True))
        # Clear of the exit by one thread element, by less than a slow frame's arc more.
        self.assertGreaterEqual(report["tail_past_exit_m"], ELEMENT - 3e-7)
        self.assertLess(report["tail_past_exit_m"], ELEMENT + 0.05 * speed * config.frame_dt)
        self.assertAlmostEqual(report["pull_angle_rad"], travelled / radius, delta=5e-6)
        # Then the tool holds the needle for the hold time of a lift and the procedure is done.
        for _ in range(29):
            tick(c, item, config)
            self.assertEqual(c.report()[0]["stage"], "clear")
        tick(c, item, config)
        self.assertEqual((c.report()[0]["stage"], c.report()[0]["reason"]), ("done", None))

    def test_the_pull_does_not_end_while_the_tissue_holds_the_track_and_stops_with_a_reason_at_its_bound(self):
        """Fails if a tail that is past the exit ends the pull although the tissue has not let go of the needle."""
        geometry, config, c, item, state = through_the_tissue()
        radius = geometry["insertion_plan"]["radius"]
        bound = (EXIT_ARC + 2.0 * ELEMENT) / radius
        tick(c, item, config)
        frames = 0
        # The tissue never lets go (engaged_track_length_m stays 4 mm): the needle follows, the tail passes the
        # exit marker and the clearance beyond it, and the pull goes on, at speed, to its bound.
        while state.pull_angle < bound - 1e-9:
            before = state.pull_angle
            tick(c, item, config)
            frames += 1
            self.assertEqual((state.stage, c.report()[0]["phase"]), ("pull", "pulling"))
            if frames > 31 and state.pull_angle < bound - 1e-9:
                self.assertAlmostEqual((state.pull_angle - before) * radius, 0.003 * config.frame_dt, delta=1e-9)
            self.assertLess(frames, 400)
        self.assertGreater(c.report()[0]["tail_past_exit_m"], ELEMENT + 0.0015)
        waited = 0
        while c.report()[0]["phase"] != "stopped":
            tick(c, item, config)
            waited += 1
            self.assertLess(waited, 30)
        self.assertIn(waited, (14, 15))
        self.assertEqual(c.report()[0]["reason"], "The needle did not come free of the tissue")
        self.assertFalse(c.report()[0]["pull_complete"])
        # A needle that does not follow, in tissue that holds nothing: the other reason, at the same bound.
        geometry, config, c, item, state = through_the_tissue()
        item["diagnostics"]["engaged_track_length_m"] = 0.0
        tick(c, item, config, follow=False)
        for _ in range(400):
            tick(c, item, config, follow=False)
            if c.report()[0]["phase"] == "stopped":
                break
        self.assertEqual(c.report()[0]["reason"], "The needle did not follow the pull out of its track")
        self.assertAlmostEqual(c.report()[0]["pull_angle_rad"], bound, delta=1e-6)
        self.assertFalse(c.report()[0]["pull_complete"])

    def test_held_tissue_leaves_at_least_the_clearance_to_go_and_free_tissue_the_arc_to_the_clear_point(self):
        """Fails if the arc that the stop ramp is run over is not the clearance while held and the measured arc when free."""
        geometry, config, c, item, state = through_the_tissue()
        radius = geometry["insertion_plan"]["radius"]
        tick(c, item, config)
        # Bring the needle (and the tool with it) 0.3 mm short of the clear point, at speed.
        state.pull_time = 10.0
        target = pull_targets(
            state.pull_needle, state.grasp, c.arc_center_local, c.arc_axes[2], (EXIT_ARC + ELEMENT - 0.0003) / radius
        )[0]
        item["gripper_target_pose"] = target
        item["needle_pose"] = times(target, item["needle_relative_pose"])
        state.pull_angle = (EXIT_ARC + ELEMENT - 0.0003) / radius
        # Held: the whole arc of a frame at speed. Free: what the stop ramp allows with 0.3 mm left.
        item["diagnostics"]["engaged_track_length_m"] = 0.001
        before = state.pull_angle
        tick(c, item, config, follow=False)
        self.assertAlmostEqual((state.pull_angle - before) * radius, 0.003 * config.frame_dt, delta=1e-9)
        item["diagnostics"]["engaged_track_length_m"] = 0.0
        before = state.pull_angle
        tick(c, item, config, follow=False)
        self.assertAlmostEqual(
            (state.pull_angle - before) * radius, ramp_speed(0.0003, 0.003) * config.frame_dt, delta=2e-9
        )
        self.assertLess(ramp_speed(0.0003, 0.003), 0.003)

    def test_the_reach_of_the_plan_is_asked_to_the_bound_of_the_pull_through_the_tissue(self):
        """Fails if the plan tests the tool's joints over a shorter arc than the pull may command."""
        geometry, config, c, item, _state = through_the_tissue()
        radius = geometry["insertion_plan"]["radius"]
        needle, marker = item["needle_pose"], item["exit_position"]
        self.assertAlmostEqual(c._pull_out_span(needle, marker), (EXIT_ARC + 2.0 * ELEMENT) / radius, delta=1e-6)
        # Without thread tracks: to the exit marker and the slack of four wire radii, as before.
        _geometry, _config, plain, plain_item, _state = closed_on_the_tip(entry_arc=ENTRY_ARC, exit_arc=EXIT_ARC)
        self.assertFalse(plain.thread_tracks)
        self.assertAlmostEqual(
            plain._pull_out_span(plain_item["needle_pose"], plain_item["exit_position"]),
            (EXIT_ARC + 4.0 * WIRE_RADIUS) / radius,
            delta=1e-6,
        )
        # The rows that _grasp_reach asks: the last pose of its pull is the tool at that span.
        grasp = item["gripper_target_pose"]
        asked = []
        reach = c._reach
        c._reach = lambda poses: (asked.append(np.array(poses)), reach(poses))[1]
        c._grasp_reach(needle, grasp, marker)
        end = pull_targets(
            needle, item["needle_relative_pose"], c.arc_center_local, c.arc_axes[2], c._pull_out_span(needle, marker)
        )[0]
        np.testing.assert_allclose(asked[-1][-1], end, atol=1e-6)

    def test_thread_tracks_need_the_length_of_a_thread_element_and_mean_nothing_without_the_pull(self):
        """Fails if a scene that says its thread follows the track can start a pull without saying how far clear is."""
        geometry, config = regrasp_fixture()
        geometry["grippers"] = [pulling_tool()]
        geometry["thread_tracks"] = True
        for length in (None, 0.0, -0.002, float("nan")):
            if length is not None:
                geometry["thread_element_length"] = length
            with self.assertRaisesRegex(ValueError, "thread element"):
                ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001, pull_through=True)
        geometry["thread_element_length"] = ELEMENT
        with_pull = ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001, pull_through=True)
        self.assertEqual((with_pull.thread_tracks, with_pull.pull_clearance), (True, ELEMENT))
        without = ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001)
        self.assertEqual((without.thread_tracks, without.pull_clearance), (False, 0.0))
        # With thread tracks the pull does not end at the hole of the entry, and its radius is not asked for.
        self.assertEqual(with_pull.pull_hole, 0.0)
        geometry["thread_tracks"] = False
        with self.assertRaisesRegex(ValueError, "needle_capture_radius"):
            ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001, pull_through=True)
        geometry["needle_capture_radius"] = HOLE
        off = ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001, pull_through=True)
        self.assertEqual((off.thread_tracks, off.pull_clearance, off.pull_hole), (False, 0.0, HOLE))


class PlanReachTests(unittest.TestCase):
    NEEDLE = straight_pose(np.radians(60.0), 0.012)
    # The exit marker beside the tail of the wire: the pull-out arc of the plan is its slack alone.
    EXIT = None

    def controller(self, limits=LIMITS, trocar=DOWN, **options):
        """A pulling controller for a straight wire that sticks out of flat tissue at 60 degrees."""
        geometry, config = fixture()
        vertices, triangles = flat_tissue()
        centerline = straight_needle()
        geometry.update(
            entry_normal=np.array([0.0, 0.0, 1.0]),
            needle_span=2.375,
            needle_tip_local=centerline[0],
            needle_centerline=centerline,
            needle_radius=WIRE_RADIUS,
            needle_capture_radius=HOLE,
            grippers=[pulling_tool(limits, trocar)],
            reference_positions=vertices,
            reference_triangles=triangles,
        )
        geometry["insertion_plan"]["exit_angle"] = 0.9
        return ProceduralGraspController(
            geometry, config, settle_seconds=0, lift_height=0.001, pull_through=True, **options
        )

    def clear_point(self):
        """The grasp that the jaws alone allow (the plan without the reach), and a trocar behind it."""
        (grasp, clear), _steps = planned(self.controller(), self.NEEDLE)
        return grasp, clear, trocar_behind(grasp)

    def exit_marker(self, c):
        return pose(wp.transform_point(wp.transform(*self.NEEDLE), wp.vec3(*c.centerline[-1])))

    @staticmethod
    def run_plan(c, needle, marker):
        notes = {}
        plan = c._plan_tip_grasp(needle, marker, notes)
        while True:
            try:
                next(plan)
            except StopIteration as done:
                return done.value, notes

    def test_the_reach_of_a_grasp_is_that_of_its_descent_and_its_pull(self):
        grasp, _clear, trocar = self.clear_point()
        c = self.controller(trocar=trocar)
        marker = self.exit_marker(c)
        # The two rows by hand: 12 mm of descent along the jaws in steps of 0.1 mm; the needle turned about
        # its centre by the slack of four wire radii, in steps of no more than 0.01 rad, the tool held to
        # it as at the grasp.
        along = procedure._rotation(grasp)[:, 1]
        descent = np.tile(grasp, (121, 1))
        descent[:, :3] -= np.linspace(0.012, 0.0, 121)[:, None] * along
        span = 4.0 * WIRE_RADIUS / c.radius
        self.assertAlmostEqual(c._pull_span(self.NEEDLE, marker), span, delta=1e-6)
        angles = np.linspace(0.0, c._pull_span(self.NEEDLE, marker), int(np.ceil(span / 0.01)) + 1)
        held = times(inverse(grasp), self.NEEDLE)
        pull = pull_targets(self.NEEDLE, held, c.arc_center_local, c.arc_axes[2], angles)
        rows = [c._reach(row) for row in (descent, pull)]
        why, rotary, slide = c._grasp_reach(self.NEEDLE, grasp, marker)
        self.assertIsNone(why)
        self.assertAlmostEqual(rotary, min(row[1] for row in rows), delta=1e-7)
        self.assertAlmostEqual(slide, min(row[2] for row in rows), delta=1e-9)
        # The tool stands nearly straight at the grasp: the descent pulls the slide back by its 12 mm, to
        # 0.19 - 0.012 - 0.0091 - 0.18 = -11.1 mm at the pregrasp, 38.9 mm from its lower limit.
        self.assertAlmostEqual(rows[0][2], 0.05 - 0.0111, delta=2e-5)
        # A row that is not reached is named: the descent with a slide that cannot go back that far, ...
        limits = LIMITS.copy()
        limits[0, 2] = -0.011
        short = self.controller(limits, trocar)
        self.assertRegex(short._grasp_reach(self.NEEDLE, grasp, marker)[0], "^in the descent insertion is -0.0001")
        # ... and the pull with a joint that the arc takes beyond a limit which the grasp itself is inside.
        joints = arm_joint_positions(trocar, pull, NOMINAL)
        joint = [0, 1, 3, 4, 5][int(np.argmax(np.ptp(joints[:, [0, 1, 3, 4, 5]], axis=0)))]
        rising = joints[-1, joint] > joints[0, joint]
        limits = LIMITS.copy()
        limits[1 if rising else 0, joint] = joints[-1, joint]
        name = ("rcm_yaw", "rcm_pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw")[joint]
        self.assertGreater(abs(joints[-1, joint] - joints[0, joint]), 2.0 * c.reach_margin[joint])
        self.assertRegex(
            self.controller(limits, trocar)._grasp_reach(self.NEEDLE, grasp, marker)[0],
            f"^in the pull {name} is ",
        )

    def test_the_plan_takes_only_points_that_the_tool_reaches_and_says_why_when_there_is_none(self):
        grasp, clear, trocar = self.clear_point()
        c = self.controller(trocar=trocar)
        marker = self.exit_marker(c)
        # With every joint well inside its limits the reach changes nothing: the same point as without it.
        (reached_grasp, reached), notes = self.run_plan(c, self.NEEDLE, marker)
        self.assertEqual(reached, clear)
        np.testing.assert_array_equal(reached_grasp, grasp)
        self.assertNotIn("unreachable", notes)
        # The margins of the point are those of its rows.
        why, rotary, slide = c._grasp_reach(self.NEEDLE, grasp, marker)
        self.assertIsNone(why)
        self.assertEqual((notes["reach_rad"], notes["reach_m"]), (rotary, slide))
        # A slide that cannot go back as far as the descent to any grasp asks (11.1 mm at the clear point,
        # and the other points of the wire lie beside it): no point is reached, and the plan says why.
        limits = LIMITS.copy()
        limits[0, 2] = -0.005
        blocked = self.controller(limits, trocar)
        result, notes = self.run_plan(blocked, self.NEEDLE, marker)
        self.assertIsNone(result)
        self.assertRegex(notes["unreachable"], r"^in the descent insertion is -0\.0\d+ m inside its limit")
        # The procedure stops with that reason.
        state = procedure._Scene(0, 0.0, 0.0, 0.0)
        state.stage, state.leg = "retreat", "rise"
        item = {"needle_pose": self.NEEDLE, "exit_position": marker}
        for _ in range(400):
            blocked._tip_grasp_planned(state, item)
            if state.status == "stopped":
                break
        self.assertEqual(state.status, "stopped")
        self.assertRegex(
            state.reason,
            "^No point of the needle tip is both clear of the tissue for the jaws and within the reach of the tool: "
            "in the descent insertion is ",
        )
        # Without the pull the same plan is not asked about the reach at all.
        self.assertEqual(planned(blocked, self.NEEDLE)[0][1], clear)

    def test_a_limit_between_two_points_moves_the_grasp_to_the_one_that_is_reached(self):
        _grasp, clear, trocar = self.clear_point()
        c = self.controller(trocar=trocar)
        marker = self.exit_marker(c)
        # Along the wire the grasps are one pose moved along it: the shaft turns about the trocar as the
        # grasp moves, by about 10 mm x sin(angle between wire and shaft) / 0.19 m between the clear point
        # and the tip. The joint about the trocar that moves most over the points of the wire:
        offsets = np.linspace(WIRE_RADIUS, clear, 40)
        joints = np.array([arm_joint_positions(trocar, c._tip_grasp(self.NEEDLE, s), NOMINAL) for s in offsets])
        joint = int(np.argmax(np.ptp(joints[:, :2], axis=0)))
        self.assertGreater(np.ptp(joints[:, joint]), 0.03)
        # A limit of that joint which the grasp half way along is inside by the margin: the clear point is
        # nearer the limit than the margin, the points near the tip are further inside.
        rising = joints[-1, joint] > joints[0, joint]
        limits = LIMITS.copy()
        limits[1 if rising else 0, joint] = joints[20, joint] + (1.0 if rising else -1.0) * c.reach_margin[joint]
        limited = self.controller(limits, trocar)

        def reached(offset):
            return limited._grasp_reach(self.NEEDLE, limited._tip_grasp(self.NEEDLE, offset), marker)[0] is None

        self.assertFalse(reached(clear))
        self.assertTrue(reached(WIRE_RADIUS))
        (_pose, found), notes = self.run_plan(limited, self.NEEDLE, marker)
        # The plan returns the point nearest the clear one that is reached: it is reached, and the point a
        # 64th of the step of the search (the smallest jaw radius) further from the tip is not.
        step = min(radius for _start, _end, radius, _sense in limited.jaws)
        self.assertLess(found, clear)
        self.assertGreater(found, WIRE_RADIUS)
        self.assertTrue(reached(found))
        self.assertFalse(reached(found + step / 64.0 + 1e-7))
        name = ("rcm_yaw", "rcm_pitch")[joint]
        self.assertRegex(notes["unreachable"], f"{name} is ")


class WayReachTests(unittest.TestCase):
    """The reach of the tool on the way of the regrasp with the pull: back, up, over and down.

    Each leg against a joint limit that is narrowed to the middle of what the leg asks of the joint, so that
    the leg starts inside the limit and does not end inside it. The tissue is the plane z = 0, the needle is
    far away, and the trocar is behind the tool along its jaws (:func:`trocar_behind`): the tool stands
    straight, and a move along its jaws is a move of its slide alone.
    """

    HALF = 0.5 * ProceduralGraspController.release_opening_rad
    controller = PlanReachTests.controller

    @staticmethod
    def tilted(height, down=np.pi / 6, position=(0.004, -0.003)):
        """A tool pose with its jaws ``down`` [rad] below the level along the world's X, closing across them."""
        along = np.array([np.cos(down), 0.0, -np.sin(down)])
        frame = np.column_stack(([0.0, 1.0, 0.0], along, np.cross([0.0, 1.0, 0.0], along)))
        return pose_from(frame, (*position, height))

    @staticmethod
    def standing(c, stage, leg, pose, opening):
        """The scene of ``c`` in a stage of the regrasp with the tool standing at ``pose``, and its observation."""
        item = {
            "env_id": 0,
            "reset_count": 0,
            "diagnostics": {"physical_time_s": 0.0, "entry_count": 1, "exit_count": 1},
            "needle_pose": FAR.copy(),
            "needle_relative_pose": np.r_[0.0, 0.006, 0.0, 0.0, 0.0, 0.0, 1.0],
            "gripper_target_pose": pose.copy(),
            "jaw_contact_count": np.zeros(2),
            "jaw_contact_normal_force_n": np.zeros(2),
            "jaw_target_rad": np.array([0.5 * opening, -0.5 * opening]),
        }
        state = procedure._Scene(0, 0.0, 0.0, 0.0)
        state.grasp, state.stage, state.leg = item["needle_relative_pose"].copy(), stage, leg
        state.reach_rad = state.reach_m = np.inf
        c.scenes[0] = state
        return item, state

    def assert_slide_short(self, reason, start, c, inside):
        """The reason of a stop names the slide ``inside`` [m] of its limit, to within a frame, and what is needed."""
        self.assertTrue(reason.startswith(start), reason)
        found = re.fullmatch(r"insertion is (\S+) m inside its limit where (\S+) m are needed", reason[len(start) :])
        self.assertIsNotNone(found, reason)
        # The pose that is furthest short counts with what the slide moves beside it: the last pose of a way
        # can be a part of a frame from the one before, which is then named instead.
        frame = c.config.translation_step * np.sqrt(3.0)
        self.assertAlmostEqual(float(found[1]), inside, delta=frame)
        self.assertGreater(float(found[2]), c.reach_margin[2])
        self.assertLessEqual(float(found[2]), c.reach_margin[2] + frame)

    @staticmethod
    def asked_of(c):
        """The rows of poses that the reach test of ``c`` is asked from now on."""
        asked, reach = [], c._reach
        c._reach = lambda poses: (asked.append(np.array(poses)), reach(poses))[1]
        return asked

    def test_the_way_back_is_asked_for_reach_when_the_jaws_have_opened(self):
        """Fails if the tool starts to back away from the needle on a way that takes a joint past its limit."""
        released = self.tilted(0.0035)
        trocar = trocar_behind(released)
        free = self.controller(trocar=trocar)
        end = free._shifted(released, free.transfer_clearance * free._leave_axis(released, self.HALF))
        # By hand: the way back is 12 mm along the axis of an open jaw, 0.3 rad off the jaws: the wrist joint
        # comes 12 cos(0.3) mm nearer the trocar and goes 12 sin(0.3) mm aside of the shaft of 180.9 mm.
        slide = arm_joint_positions(trocar, np.stack((released, end)), NOMINAL)[:, 2]
        shaft = 0.19 - WRIST_LENGTH_M
        shorter = shaft - np.hypot(shaft - 0.012 * np.cos(self.HALF), 0.012 * np.sin(self.HALF))
        self.assertAlmostEqual(slide[0], 0.19 - WRIST_LENGTH_M - NOMINAL, delta=3e-6)
        self.assertAlmostEqual(slide[0] - slide[1], shorter, delta=3e-6)
        for lower, reached in ((LIMITS[0, 2], True), (slide.mean(), False)):
            with self.subTest(lower=lower):
                limits = LIMITS.copy()
                limits[0, 2] = lower
                c = self.controller(limits, trocar)
                item, state = self.standing(c, "release", None, released, 2.0 * self.HALF)
                asked = self.asked_of(c)
                # Three calls with the jaws open and off the needle: the tool stands in each (the first two
                # take the tissue near the way back), and the third would start the retreat.
                for _ in range(3):
                    np.testing.assert_array_equal(lift_tick(c, item, c.config), np.zeros(7))
                # The reach is asked once, for the whole way back as it can be commanded.
                self.assertEqual(len(asked), 1)
                np.testing.assert_array_equal(asked[0][0], released)
                np.testing.assert_allclose(asked[0][-1], end, atol=1e-12)
                steps = np.linalg.norm(np.diff(asked[0][:, :3], axis=0), axis=1)
                self.assertLessEqual(steps.max(), np.sqrt(3.0) * c.config.translation_step + 1e-12)
                report = c.report()[0]
                if reached:
                    self.assertEqual((report["stage"], report["phase"], report["reason"]), ("retreat", "release", None))
                    continue
                self.assertEqual((report["stage"], report["phase"]), ("release", "stopped"))
                # The slide is short of its limit at the end of the way: by half of what the way asks of it.
                start = "The open jaws cannot back away from the needle within the joint limits of the tool: "
                self.assert_slide_short(report["reason"], start, c, -0.5 * shorter)
                np.testing.assert_array_equal(lift_tick(c, item, c.config), np.zeros(7))
                np.testing.assert_array_equal(item["gripper_target_pose"], released)

    def test_each_step_of_the_rise_is_asked_for_reach_and_the_rise_ends_before_the_limit_of_the_slide(self):
        """Fails if the tool is commanded up its shaft to where its slide has less than the reach margin left."""
        via = self.tilted(0.005)
        trocar = trocar_behind(via)
        slide = float(arm_joint_positions(trocar, via, NOMINAL)[2])
        # The tool rises along its shaft, 30 degrees above the level: the largest component of that direction
        # moves at the rate limit, so a frame is 0.1 mm / cos(30 degrees) of rise, all of it slide.
        step = 0.0001 / np.cos(np.pi / 6)
        # A slide with 4 mm of room above where the tool stands; then the slide of the tool itself, whose
        # limit is where the rise that is open to the tool ends (50.9 mm up).
        for lower in (slide - 0.004, LIMITS[0, 2]):
            with self.subTest(lower=lower):
                limits = LIMITS.copy()
                limits[0, 2] = lower
                c = self.controller(limits, trocar)
                item, state = self.standing(c, "retreat", "rise", via, 2.0 * self.HALF)
                state.via, state.rise = via.copy(), c._rise(via)
                self.assertAlmostEqual(np.linalg.norm(state.rise), slide + 0.05, delta=3e-6)
                # A plan stands whose point above the pregrasp is as low as the tool is now, 10 mm beside it:
                # however high the tool rises, the way over to it stays within the turning radius of the tissue.
                state.planned, state.plan_pose = True, FAR.copy()
                state.above = c._shifted(via, np.array([0.0, 0.01, 0.0]))
                frames = 0
                while c.report()[0]["phase"] != "stopped":
                    before = item["gripper_target_pose"].copy()
                    action = lift_tick(c, item, c.config)
                    frames += 1
                    self.assertLess(frames, 800)
                report = c.report()[0]
                self.assertEqual(report["stage"], "retreat")
                self.assertRegex(
                    report["reason"],
                    "^The tool cannot rise far enough to turn towards the needle tip within its joint limits: "
                    "insertion is ",
                )
                # The call that stops commands nothing: the tool stands where its last step took it.
                np.testing.assert_array_equal(action, np.zeros(7))
                np.testing.assert_array_equal(item["gripper_target_pose"], before)
                # It has risen straight up its shaft ...
                risen = item["gripper_target_pose"][:3] - via[:3]
                up = state.rise / np.linalg.norm(state.rise)
                self.assertLess(np.linalg.norm(np.cross(risen, up)), 1e-9)
                # ... as far as every step left the slide inside its limit by the margin and by that step:
                # the next step would leave less, so between one and two steps and the margin are left.
                inside = float(arm_joint_positions(trocar, item["gripper_target_pose"], NOMINAL)[2]) - lower
                self.assertGreaterEqual(inside, c.reach_margin[2] + step - 3e-6)
                self.assertLess(inside, c.reach_margin[2] + 2.0 * step + 3e-6)
                self.assertAlmostEqual(risen @ up, slide - lower - inside, delta=3e-6)
                self.assertEqual(frames, 1 + int(round((risen @ up) / step)))

    def test_the_way_over_is_asked_for_reach_frame_by_frame_before_the_tool_turns(self):
        """Fails if the reach test of the way over to the point above the pregrasp is left out."""
        # The tool stands straight 30 mm above the tissue, where it can turn. The point above the pregrasp
        # is 12 mm on and 3 mm higher, with the tool turned 0.5 rad about the vertical: the wrist has to
        # give most of that turn (its axes are across the shaft).
        start = self.tilted(0.03)
        trocar = trocar_behind(start)
        turn = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5), dtype=float)
        above = np.r_[start[:3] + [0.0, 0.012, 0.003], procedure._quat_product(turn, start[3:])]
        free = self.controller(trocar=trocar)
        way = line_targets(start, above, free.config)
        joints = arm_joint_positions(trocar, way, NOMINAL)
        # 12 mm at 0.1 mm per frame: 120 frames; the turn of 0.5 rad about the vertical ends after 50 of them.
        self.assertEqual(len(way), 121)
        np.testing.assert_allclose(way[50:, 3:], np.tile(above[3:], (71, 1)), atol=1e-12)
        joint = [0, 1, 3, 4, 5][int(np.argmax(np.ptp(joints[:, [0, 1, 3, 4, 5]], axis=0)))]
        self.assertGreater(np.ptp(joints[:, joint]), 0.2)
        rising = joints[-1, joint] > joints[0, joint]
        name = ("rcm_yaw", "rcm_pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw")[joint]
        middle = 0.5 * (joints[0, joint] + joints[-1, joint])
        beyond = joints[:, joint].max() + 0.1 if rising else joints[:, joint].min() - 0.1
        for limit, reached in ((beyond, True), (middle, False)):
            with self.subTest(limit=limit):
                limits = LIMITS.copy()
                limits[1 if rising else 0, joint] = limit
                c = self.controller(limits, trocar)
                item, state = self.standing(c, "retreat", "rise", start, 2.0 * self.HALF)
                state.via, state.rise = start.copy(), c._rise(start)
                state.planned, state.plan_pose, state.above = True, FAR.copy(), above.copy()
                asked = self.asked_of(c)
                action = lift_tick(c, item, c.config)
                # The way is asked once, as it will be commanded: from where the tool stands to the point above.
                self.assertEqual(len(asked), 1)
                np.testing.assert_array_equal(asked[0], way)
                report = c.report()[0]
                if reached:
                    self.assertEqual(
                        (report["stage"], report["phase"], state.leg), ("approach", "approach", "traverse")
                    )
                    np.testing.assert_array_equal(state.turn, start)
                    np.testing.assert_array_equal(action[:6], line_to(start, above, c.config).astype(np.float32))
                    # The margins of the way are kept for the report.
                    inside = np.minimum(joints - limits[0], limits[1] - joints)
                    self.assertAlmostEqual(report["joint_margin_rad"], inside[:, [0, 1, 3, 4, 5]].min(), delta=1e-12)
                    self.assertAlmostEqual(report["joint_margin_m"], inside[:, 2].min(), delta=1e-12)
                    continue
                self.assertEqual((report["stage"], report["phase"]), ("retreat", "stopped"))
                self.assertRegex(
                    report["reason"],
                    f"^The tool cannot turn over to the needle tip within its joint limits: {name} is ",
                )
                np.testing.assert_array_equal(action, np.zeros(7))
                np.testing.assert_array_equal(item["gripper_target_pose"], start)
                self.assertIsNone(state.turn)

    def test_the_way_down_to_the_pregrasp_is_asked_for_reach_by_the_plan_of_the_way(self):
        """Fails if the reach test of the way down from the point above the pregrasp is left out."""
        # A grasp 5 mm above the tissue with the jaws 30 degrees below the level. Its pregrasp, 12 mm back
        # along the jaws, is 11 mm up: less than the turning radius of the tool (13.3 mm) and the cover of
        # the tissue samples, so the tool comes to the line of the descent further back and has a way down.
        grasp = self.tilted(0.005)
        trocar = trocar_behind(grasp)
        free = self.controller(trocar=trocar)
        radius = free.turning_radius
        self.assertAlmostEqual(radius, 0.0133, delta=1e-9)

        def route(limits):
            c = self.controller(limits, trocar)
            state = procedure._Scene(0, 0.0, 0.0, 0.0)
            state.stage, state.leg = "retreat", "rise"
            state.tip_grasp, state.plan_pose, state.pregrasp = grasp.copy(), FAR.copy(), c._backed_off(grasp)
            state.reach_rad = state.reach_m = np.inf
            asked = self.asked_of(c)
            plan = c._plan_route(state)
            for _ in range(2000):
                try:
                    next(plan)
                except StopIteration as done:
                    return done.value, state, asked, c
            raise AssertionError("The way was not planned")

        reason, state, asked, c = route(LIMITS)
        self.assertIsNone(reason)
        # By hand: the palm is a turning radius above the tissue (radius - 5 mm) / sin(30 degrees) back.
        length = float(np.linalg.norm(state.above[:3] - grasp[:3]))
        self.assertGreaterEqual(length, (radius - 0.005) / 0.5 - 1e-9)
        self.assertLess(length, (radius - 0.005) / 0.5 + 0.004)
        self.assertGreater(length, c.transfer_clearance + 0.004)
        # The way down is asked once: from the point above to the pregrasp, 0.1 mm per frame of its largest
        # component (cos 30 degrees of it).
        self.assertEqual(len(asked), 1)
        np.testing.assert_array_equal(asked[0], line_targets(state.above, state.pregrasp, c.config))
        self.assertEqual(len(asked[0]), 1 + int(np.ceil((length - 0.012) * np.cos(np.pi / 6) / 0.0001 - 1e-3)))
        # The tool stands straight: the way down is its slide alone, from 0.9 mm less that length to
        # 0.9 mm less the 12 mm of the pregrasp; the margin of the way is that of its upper end.
        slide = arm_joint_positions(trocar, asked[0], NOMINAL)[:, 2]
        self.assertAlmostEqual(slide[0], 0.19 - WRIST_LENGTH_M - NOMINAL - length, delta=3e-6)
        self.assertAlmostEqual(slide[-1], 0.19 - WRIST_LENGTH_M - NOMINAL - 0.012, delta=3e-6)
        self.assertAlmostEqual(state.reach_m, 0.05 + slide[0], delta=1e-12)
        # A slide whose limit is in the middle of that way: the pregrasp is inside it by more than the
        # margin, the point above is not, and the way is refused with the joint.
        limits = LIMITS.copy()
        limits[0, 2] = slide.mean()
        self.assertGreater(slide[-1] - limits[0, 2], 2.0 * c.reach_margin[2])
        reason, state, asked, c = route(limits)
        start = "The tool cannot come down on the needle tip within its joint limits: "
        self.assert_slide_short(reason, start, c, -0.5 * (slide[-1] - slide[0]))
        self.assertIsNone(state.above)
        self.assertEqual(len(asked), 1)


if __name__ == "__main__":
    unittest.main()
