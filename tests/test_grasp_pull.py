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
# [m, rad].
IK_TOLERANCES = {"ik_position_tolerance_m": 1.0e-5, "ik_orientation_tolerance_rad": 1.0e-3}
# The pull through the tissue of these tests: the thread element of the scenes (0.18 m in 90 elements), the exit
# marker 3 mm of arc ahead of the tail and the entry marker 1 mm.
ELEMENT = 0.002
EXIT_ARC, ENTRY_ARC = 0.003, 0.001
# The weight that a scene with thread tracks applies to its needle [N] (0.3 g, as the dVRK scene's), and the
# passages of its planned arc: what the stitch after the pull through the tissue reads beside the thread element.
WEIGHT = 0.00294
PASSAGES = 2


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


def pull_controller(limits=LIMITS, trocar=DOWN, *, thread_element=ELEMENT, **options):
    """A controller that pulls, on the regrasp fixture: the needle as a U in flat tissue, both ends out.

    The scene says that its thread follows the needle's track through the tissue and gives the length of a
    thread element (``thread_element`` [m]), the weight it applies to the needle and the passages of its
    plan, as ``GraspScene.grasp_geometry`` and the scene's geometry do.
    """
    geometry, config = regrasp_fixture()
    geometry["grippers"] = [pulling_tool(limits, trocar)]
    geometry.update(thread_tracks=True, thread_element_length=thread_element, needle_weight_N=WEIGHT)
    geometry["insertion_plan"] = dict(geometry["insertion_plan"], expected_passages=PASSAGES)
    return (
        geometry,
        config,
        ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001, pull_through=True, **options),
    )


def closed_on_the_tip(limits=LIMITS, *, offset=0.002, entry_arc=ENTRY_ARC, exit_arc=EXIT_ARC, thread_element=ELEMENT):
    """The pull fixture at the call in which the jaws have closed on the needle ``offset`` [m] behind its tip.

    The entry marker lies ``entry_arc`` [m] of arc ahead of the tail on the needle's own circle and the exit
    marker ``exit_arc``; the scene of the controller stands in its stage ``close`` with both jaws on the wire.
    The trocar of the tool is behind that grasp (:func:`trocar_behind`).
    """
    geometry, config = regrasp_fixture()
    needle = pose_at(geometry["insertion_plan"], DRIVE_ANGLE)
    grasp = pull_controller()[2]._tip_grasp(needle, offset)
    geometry, config, c = pull_controller(limits, trocar_behind(grasp), thread_element=thread_element)
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
    """One frame: the tool target follows the action exactly and the palm is at it; the needle follows the tool as it lies in the jaws."""
    item["diagnostics"]["physical_time_s"] += config.frame_dt
    item["jaw_contact_count"] = np.full(2, int(contact))
    item["jaw_contact_normal_force_n"] = np.full(2, 0.2 if contact else 0.0)
    action = c([item])[0]
    item["gripper_target_pose"] = stepped(item["gripper_target_pose"], action, config)
    item["gripper_pose"] = item["gripper_target_pose"].copy()
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
        geometry, config, _usual = pull_controller()
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

    def test_the_pull_needs_thread_tracks_and_the_limits_of_the_tool(self):
        geometry, config = regrasp_fixture()
        # The pull takes the needle through the tissue: a scene whose thread does not follow the track is refused.
        for tracks in (None, False):
            if tracks is not None:
                geometry["thread_tracks"] = tracks
            with self.assertRaisesRegex(ValueError, "thread_tracks"):
                ProceduralGraspController(geometry, config, lift_height=0.001, pull_through=True)
        geometry.update(thread_tracks=True, thread_element_length=ELEMENT, needle_weight_N=WEIGHT)
        geometry["insertion_plan"] = dict(geometry["insertion_plan"], expected_passages=PASSAGES)
        with self.assertRaisesRegex(ValueError, "arm_joint_limits"):
            ProceduralGraspController(geometry, config, lift_height=0.001, pull_through=True)
        geometry["grippers"] = [pulling_tool(LIMITS[:, :5])]
        with self.assertRaisesRegex(ValueError, "six increasing finite joint limits"):
            ProceduralGraspController(geometry, config, lift_height=0.001, pull_through=True)
        geometry["grippers"] = [
            pulling_tool(np.array([[-0.6, -0.6, -0.05, -3.0, -1.6, -1.1], [0.6, 0.6, 0.05, 3.0, 1.6, 1.1]]))
        ]
        with self.assertRaisesRegex(ValueError, "wrist pitch of less than a quarter turn"):
            ProceduralGraspController(geometry, config, lift_height=0.001, pull_through=True)
        # Without the argument nothing of the pull is read or used.
        earlier = ProceduralGraspController(regrasp_fixture()[0], config, lift_height=0.001)
        self.assertFalse(earlier.pull_through)
        self.assertIsNone(earlier.joint_limits)
        self.assertIsNone(earlier.jaws)
        pulling = pull_controller(LIMITS)[2]
        self.assertEqual(
            (pulling.pull_grasp_roll_rad, pulling.pull_grasp_bite_rad, pulling.pull_grasp_jaws_exchanged),
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
        # A scene whose thread stays on the surface says so, and the controller refuses to pull in it.
        with _scene(example_arguments=("--no-thread-tracks",)) as (scene, config):
            geometry = _geometry(scene, config)
            self.assertFalse(geometry["thread_tracks"])
            with self.assertRaisesRegex(ValueError, "thread_tracks"):
                ProceduralGraspController(geometry, config, lift_height=0.01, pull_through=True)
            self.assertFalse(ProceduralGraspController(geometry, config, lift_height=0.01).thread_tracks)


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
        # The pull to a marker: that arc and the slack beyond it, here four wire radii, as an angle.
        slack = 4.0 * WIRE_RADIUS
        self.assertAlmostEqual(c._pull_span(needle, marker(SPAN - 0.4), 0.0, slack), 0.4 + slack / radius, delta=1e-6)
        self.assertAlmostEqual(c._pull_span(needle, marker(SPAN + 0.1), 0.0, slack), slack / radius, delta=1e-6)
        # To two wire radii of arc before a marker, and to one thread element past it.
        self.assertAlmostEqual(
            c._pull_span(needle, marker(SPAN - 0.4), 2.0 * WIRE_RADIUS, slack),
            0.4 + 2.0 * WIRE_RADIUS / radius,
            delta=1e-6,
        )
        self.assertAlmostEqual(
            c._pull_span(needle, marker(SPAN - 0.4), -ELEMENT, ELEMENT), 0.4 + 2.0 * ELEMENT / radius, delta=1e-6
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

    def test_the_way_of_a_stop_from_a_speed_is_the_distance_at_which_the_ramp_runs_at_that_speed(self):
        """Fails if a stop begun at a speed is given another way than the ramp of the insertion needs from it."""
        speed, ramp_way = 0.003, procedure.ramp_way
        # From the full speed or more: speed x ramp_time / 2 = 0.75 mm; from rest: nothing.
        self.assertEqual(ramp_way(speed, speed), 0.00075)
        self.assertEqual(ramp_way(2.0 * speed, speed), 0.00075)
        self.assertEqual(ramp_way(0.0, speed), 0.0)
        self.assertEqual(ramp_way(-0.001, speed), 0.0)
        # From half speed: the ramp runs at it half way through its time, with 0.140625 mm covered.
        self.assertAlmostEqual(ramp_way(0.5 * speed, speed), 0.003 * 0.5 * (0.125 - 0.03125), delta=1e-15)
        # The inverse of ramp_speed at every point of the ramp, and for another ramp time.
        for left in np.linspace(0.00001, 0.00074, 25):
            self.assertAlmostEqual(ramp_way(ramp_speed(left, speed), speed), left, delta=1e-14)
            self.assertAlmostEqual(ramp_way(ramp_speed(left, speed, 0.8), speed, 0.8), left, delta=1e-14)


class PullStageTests(unittest.TestCase):
    def assert_held(self, action, config):
        """The tool holds its target and keeps its jaws closed: the quaternion arithmetic of a pose
        difference in single precision leaves a rotation of a few float32 rounding errors."""
        np.testing.assert_array_equal(action[:3], np.zeros(3))
        self.assertLessEqual(np.linalg.norm(action[3:6]) * config.rotation_step, 4 * np.finfo(np.float32).eps)
        self.assertEqual(action[6], 0.0)

    def test_the_pull_turns_the_needle_at_the_speed_rule_one_thread_element_past_the_exit_then_holds(self):
        geometry, config, c, item, state = closed_on_the_tip()
        plan = geometry["insertion_plan"]
        radius, center = plan["radius"], plan["arc_center_local"]
        # The tissue holds nothing of the track: the tail goes to one thread element past the exit marker.
        way = EXIT_ARC + ELEMENT
        start = item["needle_pose"].copy()
        centre_world = procedure._rotation(start) @ center + start[:3]
        # The call in which the jaws are closed starts the pull and moves nothing.
        action = tick(c, item, config)
        np.testing.assert_array_equal(action, np.zeros(7))
        self.assertEqual((state.stage, c.report()[0]["phase"]), ("pull", "close"))
        self.assertAlmostEqual(state.tail_past_entry, -ENTRY_ARC, delta=1e-8)
        self.assertAlmostEqual(state.tail_past_exit, -EXIT_ARC, delta=1e-8)
        # The tail has 3 mm and the element to go; the command may go one element further.
        self.assertAlmostEqual(state.pull_bound, (way + ELEMENT) / radius, delta=1e-6)
        # Speed rule: the palm origin is this far from the axis of the needle (from the measured grasp).
        palm = inverse(item["needle_relative_pose"])[:3] - center
        arm = np.hypot(palm[0], palm[1])
        speed = min(0.003, 0.8 * 0.0001 * 60.0 * radius / arm, 0.8 * 0.01 * 60.0 * radius)
        self.assertAlmostEqual(state.pull_speed, speed, delta=1e-9)
        # By hand, frame by frame: the ramp of the insertion adds its arc, but no more than the speed of the
        # same ramp towards its stop allows with what the tail has left; the needle follows the tool exactly.
        travelled, frames, steps = 0.0, 0, []
        while way - travelled > 0.0:
            frames += 1
            added = float(
                ramp_distance(frames * config.frame_dt, speed) - ramp_distance((frames - 1) * config.frame_dt, speed)
            )
            added = min(added, ramp_speed(way - travelled, speed) * config.frame_dt)
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
            self.assertLessEqual(abs(frames - ((way) / speed * 60.0 + 30.0)), 5.0)
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
        # That call saw the tail clear of the exit: the pull has ended and the tool holds its target.
        report = c.report()[0]
        self.assertEqual((report["stage"], report["phase"], report["pull_complete"]), ("clear", "holding", True))
        self.assert_held(action, config)
        self.assertAlmostEqual(report["tail_past_entry_m"], travelled - ENTRY_ARC, delta=3e-7)
        # The tail is one thread element past the exit marker, by less than a slow frame's arc more.
        self.assertGreaterEqual(report["tail_past_exit_m"], ELEMENT - 3e-7)
        self.assertLess(report["tail_past_exit_m"], ELEMENT + 0.05 * speed * config.frame_dt)
        self.assertAlmostEqual(report["tail_past_exit_m"], travelled - EXIT_ARC, delta=3e-7)
        # It holds for the hold time of a lift (0.5 s) without moving; the draw of the thread follows (DrawTests).
        held = item["gripper_target_pose"].copy()
        for _ in range(29):
            self.assert_held(tick(c, item, config), config)
            self.assertEqual(c.report()[0]["stage"], "clear")
        np.testing.assert_allclose(item["gripper_target_pose"], held, atol=1e-6)
        tick(c, item, config)
        self.assertEqual((c.report()[0]["stage"], c.report()[0]["reason"]), ("draw", None))

    def test_the_report_follows_the_tail_of_the_needle_after_the_procedure_is_done(self):
        """Fails if a done procedure reports the tail where the last frame of the hold had it."""
        # The pull through tissue that holds nothing of the track.
        for fixture_of in (through_the_tissue,):
            with self.subTest(fixture=fixture_of.__name__):
                _geometry, config, c, item, state = fixture_of()
                item["diagnostics"]["engaged_track_length_m"] = 0.0
                # After the pull through the tissue the stitch follows: the scene of the test ties and its
                # tissue is at the tightness rule after 3.4 mm.
                scene = Stitched(c, item, config) if fixture_of is through_the_tissue else None
                for _ in range(1200):
                    tick(c, item, config) if scene is None else scene.tick()
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
        geometry.update(thread_tracks=True, thread_element_length=ELEMENT, needle_weight_N=WEIGHT)
        geometry["insertion_plan"] = dict(geometry["insertion_plan"], expected_passages=PASSAGES)
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
        self.assertEqual(state.stage, "pull")
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
        bound = (EXIT_ARC + 2.0 * ELEMENT) / c.radius
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
        span = (EXIT_ARC + 2.0 * ELEMENT) / c.radius
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


def through_the_tissue(**options):
    """The pull fixture with tissue that holds the needle's track until the tail is level with the exit marker."""
    geometry, config, c, item, state = closed_on_the_tip(**options)
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
        # Then the tool holds the needle for the hold time of a lift, and the draw of the thread follows
        # (DrawTests), then the stitch (StitchTests).
        for _ in range(29):
            tick(c, item, config)
            self.assertEqual(c.report()[0]["stage"], "clear")
        tick(c, item, config)
        self.assertEqual((c.report()[0]["stage"], c.report()[0]["reason"]), ("draw", None))

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
        # The stitch that follows such a pull ends with a thread that pulls the needle with less than the
        # needle weighs, tied at the exit of the last planned passage: the scene must say both.
        for weight in (None, -0.001, float("nan")):
            if weight is not None:
                geometry["needle_weight_N"] = weight
            with self.assertRaisesRegex(ValueError, "needle_weight_N"):
                ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001, pull_through=True)
        # A scene that applies no weight to its needle says zero.
        geometry["needle_weight_N"] = 0.0
        plan = geometry["insertion_plan"]
        for passages in (None, 0, 1.5, True):
            geometry["insertion_plan"] = plan if passages is None else dict(plan, expected_passages=passages)
            with self.assertRaisesRegex(ValueError, "expected_passages"):
                ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001, pull_through=True)
        geometry["insertion_plan"] = dict(plan, expected_passages=PASSAGES)
        with_pull = ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001, pull_through=True)
        self.assertEqual((with_pull.thread_tracks, with_pull.pull_clearance), (True, ELEMENT))
        self.assertEqual((with_pull.needle_weight, with_pull.passages), (0.0, PASSAGES))
        # The least tissue volume of a tight thread is the coupling's rule unless the scene publishes one for
        # its own tissue under the tie: that may ask for an earlier stop, never for a later one.
        self.assertEqual(with_pull.tight_volume, 0.5)
        for volume in (None, 0.49, 0.0, 1.0, 1.2, float("nan"), float("inf")):
            geometry["tight_volume_ratio"] = volume
            with self.subTest(volume=volume), self.assertRaisesRegex(ValueError, "tight_volume_ratio"):
                ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001, pull_through=True)
        for volume in (0.5, 0.75, 0.99):
            geometry["tight_volume_ratio"] = volume
            own = ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001, pull_through=True)
            self.assertEqual(own.tight_volume, volume)
        del geometry["tight_volume_ratio"]
        # Neither is asked of a scene whose controller does not pull.
        geometry["insertion_plan"] = plan
        del geometry["needle_weight_N"]
        without = ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001)
        self.assertEqual((without.thread_tracks, without.pull_clearance), (False, 0.0))
        # A scene whose thread stays on the surface cannot be pulled in.
        geometry["thread_tracks"] = False
        with self.assertRaisesRegex(ValueError, "thread_tracks"):
            ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001, pull_through=True)


def clear_of_the_tissue(stage="tie", **options):
    """The fixture of the pull through the tissue at the call that begins ``stage``.

    ``draw``: the call that ends the clear hold of the pull. ``tie`` (the default): the call that ends the hold
    after the draw of the thread; the tie is asked for in the next.
    """
    geometry, config, c, item, state = through_the_tissue(**options)
    tick(c, item, config)
    for _ in range(1200):
        # The tissue of the test: it holds the track until the tail is level with the exit marker.
        item["diagnostics"]["engaged_track_length_m"] = 0.004 if state.tail_past_exit < 0.0 else 0.0
        tick(c, item, config)
        if state.stage == stage:
            break
    assert (state.stage, state.pull_complete, c.report()[0]["reason"]) == (stage, True, None), c.report()[0]
    return geometry, config, c, item, state


class DrawTests(unittest.TestCase):
    """The draw of the loose thread through the tissue, between the pull and the first tie. No physics."""

    def test_the_palm_draws_straight_along_the_chord_by_one_length_of_the_needle_then_holds_and_the_tie_follows(self):
        """Fails if the draw turns the tool, leaves the chord, has another length, steps in speed or skips its hold."""
        from newton_solver_suturebot.examples.grasp._dvrk_grasper import GRASP_BITE_DEPTH_M

        _geometry, config, c, item, state = clear_of_the_tissue("draw")
        needle, palm = item["needle_pose"], item["gripper_pose"]
        tail = procedure._rotation(needle) @ c.centerline[-1] + needle[:3]
        bite = procedure._rotation(palm) @ np.array([0.0, GRASP_BITE_DEPTH_M, 0.0]) + palm[:3]
        direction = (bite - tail) / np.linalg.norm(bite - tail)
        # To the rounding that a held target leaves in the poses of the call after it (float32 quaternions).
        np.testing.assert_allclose(state.draw_direction, direction, atol=1e-6)
        direction = state.draw_direction
        # One length of the needle: the chord from its tail to its tip.
        length = float(np.linalg.norm(c.centerline[0] - c.centerline[-1]))
        self.assertEqual(c.draw_needle_lengths, 1.0)
        self.assertAlmostEqual(state.draw_length, length, delta=1e-15)
        report = c.report()[0]
        self.assertEqual(
            (report["draw_travel_m"], report["draw_length_m"], report["draw_complete"]), (0.0, length, False)
        )
        self.assertIsNone(c.commands())
        start = item["gripper_target_pose"].copy()
        speed = c.tighten_speed
        # By hand: the ramp of the insertion at the start, and the same ramp backwards over what is left.
        travelled, frames = 0.0, 0
        while state.stage == "draw":
            action = tick(c, item, config)
            if state.stage != "draw":
                break
            frames += 1
            added = float(
                ramp_distance(frames * config.frame_dt, speed) - ramp_distance((frames - 1) * config.frame_dt, speed)
            )
            left = length - travelled
            travelled += min(added, ramp_speed(left, speed) * config.frame_dt, left)
            moved = item["gripper_target_pose"][:3] - start[:3]
            # Straight along the chord, without turning; the jaws stay closed; nothing is asked of the scene.
            np.testing.assert_allclose(moved, travelled * direction, atol=2e-8)
            self.assertLessEqual(np.linalg.norm(action[3:6]) * config.rotation_step, 4 * np.finfo(np.float32).eps)
            self.assertEqual(action[6], 0.0)
            self.assertIsNone(c.commands())
            self.assertEqual(c.report()[0]["phase"], "drawing")
            self.assertAlmostEqual(c.report()[0]["draw_travel_m"], travelled, delta=2e-9)
            self.assertLess(frames, 800)
        # The whole length, in the time of the travel at 3 mm/s and of one ramp (each of the two ramps takes
        # a quarter of a second longer than its way at full speed), to a tenth of a second.
        self.assertAlmostEqual(state.draw_travel, length, delta=1e-12)
        self.assertLessEqual(abs(frames - round(60 * (length / speed + 0.5))), 6)
        report = c.report()[0]
        self.assertEqual((report["stage"], report["phase"], report["draw_complete"]), ("drawn", "holding", True))
        # Then the tool holds for the hold time of a lift, and the tie follows (StitchTests).
        held = item["gripper_target_pose"].copy()
        for _ in range(29):
            tick(c, item, config)
            self.assertEqual(c.report()[0]["stage"], "drawn")
        tick(c, item, config)
        self.assertEqual((c.report()[0]["stage"], c.report()[0]["reason"]), ("tie", None))
        np.testing.assert_array_equal(item["gripper_target_pose"], held)

    def test_a_thread_that_does_not_slide_stops_the_draw_with_the_reason(self):
        """Fails if the draw goes on while the tissue is pulled to the tightness rule of the thread."""
        _geometry, config, c, item, state = clear_of_the_tissue("draw")
        item["diagnostics"]["least_volume_ratio"] = 0.9
        item["thread_tension"] = {"needle_force_N": 0.004}
        for _ in range(40):
            tick(c, item, config)
        self.assertEqual((state.stage, c.report()[0]["phase"]), ("draw", "drawing"))
        self.assertEqual(c.report()[0]["draw_thread_force_N"], 0.004)
        travel = state.draw_travel
        item["diagnostics"]["least_volume_ratio"] = 0.5
        action = tick(c, item, config)
        report = c.report()[0]
        self.assertEqual((report["phase"], report["stage"], report["draw_complete"]), ("stopped", "draw", False))
        self.assertEqual(
            report["reason"],
            f"The thread did not run through its track: {1e3 * travel:.2f} mm into the draw the "
            "least tissue volume is 0.5 where 0.5 is a tight thread",
        )
        np.testing.assert_array_equal(action[:3], np.zeros(3))

    def test_a_draw_that_the_tool_cannot_reach_with_the_tightening_after_it_stops_before_it_begins(self):
        """Fails if the draw begins on a line that the joints cannot follow to its end and to the tightening's bound."""
        from newton_solver_suturebot.examples.grasp._dvrk_grasper import GRASP_BITE_DEPTH_M

        def at_the_end_of_the_clear_hold():
            _geometry, config, c, item, state = clear_of_the_tissue("clear")
            c.joint_limits = c.joint_limits.copy()
            return config, c, item, state

        def slide(c, item, travel):
            # The slide of the tool with its palm ``travel`` [m] along the chord from its target.
            needle, palm = item["needle_pose"], item["gripper_pose"]
            tail = procedure._rotation(needle) @ c.centerline[-1] + needle[:3]
            bite = procedure._rotation(palm) @ np.array([0.0, GRASP_BITE_DEPTH_M, 0.0]) + palm[:3]
            pose = item["gripper_target_pose"].copy()
            pose[:3] += travel * (bite - tail) / np.linalg.norm(bite - tail)
            return arm_joint_positions(c.trocar, pose[None], c.nominal_insertion)[0][2]

        # The slide of this tool has no room left: any way along the chord takes it out.
        config, c, item, state = at_the_end_of_the_clear_hold()
        here = slide(c, item, 0.0)
        c.joint_limits[:, 2] = here - 0.002, here + 0.002
        for _ in range(31):
            tick(c, item, config)
        report = c.report()[0]
        self.assertEqual((report["phase"], report["stage"]), ("stopped", "clear"))
        self.assertRegex(
            report["reason"],
            "^The tool cannot draw the thread on through the tissue within its joint limits: insertion is ",
        )
        # The slide has room for the draw and not for the tightening after it: its limit lies halfway between
        # where the draw ends and where the tightening's bound is. The same tool with a tightening that needs
        # no room begins the draw.
        for bound, begins in ((None, False), (1.0e-6, True)):
            with self.subTest(bound=bound):
                config, c, item, state = at_the_end_of_the_clear_hold()
                if bound is not None:
                    c.tighten_travel_bound = bound
                length = float(np.linalg.norm(c.centerline[0] - c.centerline[-1]))
                drawn = slide(c, item, length)
                beyond = slide(c, item, length + ProceduralGraspController.tighten_travel_bound)
                # The draw moves the slide, and the tightening moves it on by more than the reach margin twice.
                self.assertGreater(abs(drawn - here), 0.005)
                self.assertGreater(abs(beyond - drawn), 4.0 * c.reach_margin[2])
                limit = 0.5 * (drawn + beyond)
                c.joint_limits[:, 2] = (min(here, limit) - 0.01, limit) if beyond > drawn else (limit, here + 0.01)
                for _ in range(31):
                    tick(c, item, config)
                report = c.report()[0]
                if begins:
                    self.assertEqual((report["phase"], report["stage"], report["reason"]), ("drawing", "draw", None))
                else:
                    self.assertEqual((report["phase"], report["stage"]), ("stopped", "clear"))
                    self.assertRegex(report["reason"], "^The tool cannot draw the thread on through the tissue")


class Stitched:
    """What the scene answers in the stitch: the ties it is asked for, a palm behind its target, a tissue volume, a pull.

    The least tissue volume is ``start`` while the palm has moved less than ``slack`` [m] from where it was
    at the tie, and falls from there by one over ``length`` [m] of palm travel: the linear law that the pull
    of a tied thread measured on the scene's tissue (0.998 - s / 6.77 mm). ``lag`` is the part of the way to
    its target that the palm has still to go after a frame (0: the palm is at its target). ``law``, if given,
    is the least volume as a function of the palm's travel [m] instead of the linear law.

    The thread's force on the needle rises by ``taut`` [N/m] over that travel beyond the slack (5.2 mN per
    mm, measured with the same pull). From the tie at the exit on it is that of the thread between the exit
    and the needle alone: it falls by ``release`` [N/m] over the palm's travel back (12 mN per mm) to
    nothing, and beyond that point the thread is pushed together and pushes back by ``push`` [N/m] (2.2 mN
    per mm): the curve measured on the scene's tissue. ``ties`` False: the scene ties nothing; a number: it
    holds at most that many ties; ``report`` False: its observation has no force of the thread on the needle.
    """

    def __init__(
        self,
        c,
        item,
        config,
        *,
        length=0.00677,
        slack=0.0,
        start=1.0,
        lag=0.0,
        ties=True,
        volume=True,
        taut=5.2,
        release=12.0,
        push=2.2,
        report=True,
        law=None,
    ):
        self.c, self.item, self.config = c, item, config
        self.length, self.slack, self.start, self.lag, self.ties, self.volume = length, slack, start, lag, ties, volume
        self.taut, self.release, self.push, self.report = taut, release, push, report
        self.law = law
        self.asked, self.commands = [], []
        self.palm = item["gripper_target_pose"][:3].copy()
        # Where the palm was when the thread was tied; None before the tie. Then its travel and the thread's
        # force on the needle when the thread was tied at the exit; None before that tie.
        self.origin = self.locked = None
        self.observe()

    @property
    def travel(self):
        """How far the palm is from where it was at the tie [m]; zero before the tie."""
        return 0.0 if self.origin is None else float(np.linalg.norm(self.palm - self.origin))

    @property
    def pull(self):
        """The thread's force on the needle [N] where the palm is: see the class."""
        if self.locked is None:
            return self.taut * max(0.0, self.travel - self.slack)
        travel, pull = self.locked
        back = travel - self.travel
        return pull - self.release * back if self.release * back < pull else self.push * (back - pull / self.release)

    def observe(self):
        """The observation after a frame: the palm, the thread's ties and pull and the least tissue volume."""
        item = self.item
        self.palm += (1.0 - self.lag) * (item["gripper_target_pose"][:3] - self.palm)
        item["gripper_pose"] = np.r_[self.palm, item["gripper_target_pose"][3:]]
        held = len(self.asked) if self.ties is True else min(len(self.asked), int(self.ties))
        item["thread_tension"] = {"ties": held, "ties_asked": len(self.asked), "eyelet_force_N": 0.0}
        if self.report:
            item["thread_tension"]["needle_force_N"] = self.pull
        if self.volume and self.law is not None:
            item["diagnostics"]["least_volume_ratio"] = self.law(self.travel)
        elif self.volume:
            item["diagnostics"]["least_volume_ratio"] = self.start - max(0.0, self.travel - self.slack) / self.length

    def tick(self, **options):
        """One call of the controller and the frame after it; returns the action."""
        action = tick(self.c, self.item, self.config, **options)
        commands = self.c.commands()
        self.commands.append(commands)
        if commands is not None:
            # The scene carries out what came with the action before it steps the frame: the tie at the
            # entry, or the tie at the exit, from which on the thread to the needle is a piece of its own.
            if self.asked:
                self.locked = (self.travel, self.pull)
            else:
                self.origin = self.palm.copy()
            self.asked.extend(commands[0])
        self.observe()
        return action

    def run(self, stage, limit=600):
        """Tick until the stage of the scene is ``stage`` or the procedure has stopped; returns the frames."""
        for frame in range(1, limit + 1):
            self.tick()
            report = self.c.report()[0]
            if report["stage"] == stage or report["phase"] == "stopped":
                return frame
        raise AssertionError(f"The stage {stage} was not reached in {limit} frames: {self.c.report()[0]}")


# What the lock asks of the scene: the tie at the last eyelet of the last planned passage.
LOCK = ("tie_thread", {"passage": PASSAGES - 1, "last": True})


class StitchTests(unittest.TestCase):
    """The stitch after the pull through the tissue: tie, tighten, hold; then :class:`LockTests`. No physics: :class:`Stitched` answers."""

    def tightening(self, **tissue):
        """The fixture in the call that has begun the tightening: the tie is made and nothing has moved."""
        geometry, config, c, item, state = clear_of_the_tissue()
        scene = Stitched(c, item, config, **tissue)
        scene.run("tighten")
        self.assertEqual((state.stage, c.report()[0]["reason"]), ("tighten", None))
        return geometry, config, c, item, state, scene

    def test_the_tie_is_asked_for_once_beside_the_action_and_the_scene_must_report_it(self):
        """Fails if the tie is asked for in another call than the first of the stage, more than once, or not checked."""
        geometry, config, c, item, state = clear_of_the_tissue()
        # No call of the procedure up to here has asked the scene for anything.
        self.assertIsNone(c.commands())
        scene = Stitched(c, item, config)
        held = item["gripper_target_pose"].copy()
        action = scene.tick()
        # The first call of the stage asks for the tie of the thread at its first entry, and holds the tool.
        self.assertEqual(scene.commands[-1], [(("tie_thread", {}),)])
        self.assertEqual((state.stage, c.report()[0]["phase"]), ("tie", "tying"))
        np.testing.assert_array_equal(action[:3], np.zeros(3))
        # The next call sees the tie in the scene's report and begins the tightening; it asks for nothing.
        scene.tick()
        self.assertIsNone(scene.commands[-1])
        self.assertEqual((state.stage, c.report()[0]["ties"]), ("tighten", 1))
        np.testing.assert_allclose(item["gripper_target_pose"], held, atol=1e-6)
        scene.run("lock")
        self.assertEqual(scene.asked, [("tie_thread", {})])
        self.assertEqual([commands for commands in scene.commands if commands is not None], [[(("tie_thread", {}),)]])
        # A scene that does not tie (an eyelet without the thread is refused on the device): the procedure stops
        # with what the scene reports.
        geometry, config, c, item, state = clear_of_the_tissue()
        scene = Stitched(c, item, config, ties=False)
        scene.tick()
        scene.tick()
        report = c.report()[0]
        self.assertEqual((report["phase"], report["stage"], report["stitch_complete"]), ("stopped", "tie", False))
        self.assertEqual(
            report["reason"], "The thread was not tied at its entry: the scene reports 0 ties of 1 asked for"
        )
        # A caller that does not pass the command on: nothing was asked of the scene.
        geometry, config, c, item, state = clear_of_the_tissue()
        scene = Stitched(c, item, config)
        scene.tick()
        del scene.asked[:]
        scene.observe()
        scene.tick()
        self.assertEqual(
            c.report()[0]["reason"], "The thread was not tied at its entry: the scene reports 0 ties of 0 asked for"
        )
        # A scene whose observation has no thread report at all.
        geometry, config, c, item, state = clear_of_the_tissue()
        tick(c, item, config)
        tick(c, item, config)
        self.assertEqual(
            c.report()[0]["reason"],
            "The thread was not tied at its entry: the scene reports None ties of None asked for",
        )

    def test_the_arc_with_a_lift_ends_where_it_ended_and_asks_the_scene_for_nothing(self):
        """Fails if the procedure without the pull goes on after its lift, asks for a tie or reports a stitch."""
        geometry, config = regrasp_fixture()
        c = ProceduralGraspController(geometry, config, settle_seconds=0, lift_height=0.001)
        item = gripped(snapshot(geometry), geometry, 2.0)
        asked = []
        for time in (0.0, 30.0):
            item["diagnostics"]["physical_time_s"] = time
            c([item])
            asked.append(c.commands())
        item["diagnostics"]["entry_count"] = 1
        for _ in range(600):
            lift_tick(c, item, config)
            asked.append(c.commands())
            if c.report()[0]["stage"] == "done":
                break
        report = c.report()[0]
        self.assertEqual(
            (report["stage"], report["lifts_complete"], report["pull_complete"], report["stitch_complete"]),
            ("done", 1, False, False),
        )
        self.assertEqual(set(asked), {None})
        self.assertEqual(
            (report["ties"], report["tighten_travel_bound_m"], report["tighten_stop_volume"]), (None, None, None)
        )

    def test_the_palm_moves_straight_along_the_chord_from_the_tail_to_the_bite_point_at_the_ramp_of_the_insertion(self):
        """Fails if the tightening turns the tool, leaves the chord of the needle or starts with a step in speed."""
        from newton_solver_suturebot.examples.grasp._dvrk_grasper import GRASP_BITE_DEPTH_M

        _geometry, config, c, item, state, scene = self.tightening()
        needle, palm = item["needle_pose"], item["gripper_pose"]
        tail = procedure._rotation(needle) @ c.centerline[-1] + needle[:3]
        bite = procedure._rotation(palm) @ np.array([0.0, GRASP_BITE_DEPTH_M, 0.0]) + palm[:3]
        direction = (bite - tail) / np.linalg.norm(bite - tail)
        # To the rounding that a held target leaves in the poses of the call after it (float32 quaternions).
        np.testing.assert_allclose(state.tighten_direction, direction, atol=1e-6)
        direction = state.tighten_direction
        # The jaws hold the needle 2 mm behind its tip: the chord is longer than the needle's radius.
        self.assertGreater(np.linalg.norm(bite - tail), c.radius)
        start = item["gripper_target_pose"].copy()
        self.assertAlmostEqual(c.tighten_speed, 0.003, delta=1e-15)
        travelled = []
        for frame in range(1, 46):
            action = scene.tick()
            moved = item["gripper_target_pose"][:3] - start[:3]
            travelled.append(float(moved @ direction))
            # Straight along the chord, without turning; the jaws stay closed.
            np.testing.assert_allclose(moved, travelled[-1] * direction, atol=1e-9)
            self.assertLessEqual(np.linalg.norm(action[3:6]) * config.rotation_step, 4 * np.finfo(np.float32).eps)
            self.assertEqual(action[6], 0.0)
            # The travel is that of the ramp of the insertion: 0.75 mm in its 0.5 s, then 3 mm/s.
            self.assertAlmostEqual(travelled[-1], float(ramp_distance(frame * config.frame_dt, 0.003)), delta=2e-9)
            self.assertAlmostEqual(c.report()[0]["tighten_travel_m"], travelled[-1], delta=2e-9)
        self.assertAlmostEqual(travelled[29], 0.00075, delta=2e-9)
        self.assertAlmostEqual(travelled[44] - travelled[43], 0.003 * config.frame_dt, delta=2e-9)
        self.assertEqual((state.stage, c.report()[0]["phase"]), ("tighten", "tightening"))
        # A palm that the rate limit of the actions holds back sets the speed, as in a lift: 0.8 of it.
        config.translation_step = 0.00004
        self.assertAlmostEqual(c.tighten_speed, 0.8 * 0.00004 * 60.0, delta=1e-15)

    def test_the_stop_begins_where_the_measured_slope_predicts_the_rule_at_rest_and_ends_there(self):
        """Fails if the tightening stops where the least volume is at its rule instead of coming to rest there."""
        length, way = 0.00677, 0.00075
        for tissue in ({}, {"lag": 0.6}, {"slack": 0.001}, {"start": 0.9, "slack": 0.0004, "lag": 0.3}):
            with self.subTest(tissue=tissue):
                _geometry, config, c, item, state, scene = self.tightening(**tissue)
                start, slack = tissue.get("start", 1.0), tissue.get("slack", 0.0)
                self.assertEqual(c.tight_volume, 0.5)
                self.assertEqual(c.report()[0]["tighten_start_volume"], start)
                steps, volumes = [], []
                while state.stage == "tighten":
                    before = state.tighten_travel
                    scene.tick()
                    steps.append(state.tighten_travel - before)
                    volumes.append(item["diagnostics"]["least_volume_ratio"])
                    self.assertLess(len(steps), 400)
                report = c.report()[0]
                self.assertEqual((report["stage"], report["phase"], report["reason"]), ("hold", "holding", None))
                # By hand: at 3 mm/s the ramp to rest takes v T / 2 = 0.75 mm, over which the volume falls by
                # 0.75 / 6.77 = 0.111. The stop therefore begins in the first frame in which the command is
                # within 0.75 mm of the travel at which the volume is 0.5, slack + 6.77 mm (start - 0.5): for
                # a volume that is linear in the palm's travel this is so whatever the palm lags.
                rule = slack + length * (start - 0.5)
                travel = report["tighten_travel_m"]
                self.assertGreaterEqual(travel, rule - 1e-9)
                self.assertLess(travel, rule + 0.003 * config.frame_dt)
                self.assertAlmostEqual(report["tighten_slope_per_m"], -1.0 / length, delta=1e-6 / length)
                # What the rule predicted when it began the stop: at most the rule's value, by less than the
                # fall of one frame at speed, 0.05 mm / 6.77 mm = 0.0074.
                predicted = report["tighten_predicted_volume"]
                self.assertLessEqual(predicted, 0.5)
                self.assertGreater(predicted, 0.5 - 0.003 * config.frame_dt / length)
                self.assertAlmostEqual(predicted, start - (travel - slack) / length, delta=1e-9)
                # The way that the palm had left when the stop began: the ramp's 0.75 mm and what it lagged.
                self.assertGreaterEqual(report["tighten_way_to_rest_m"], way - 1e-9)
                # The stop is the ramp of the insertion run backwards: the frame steps fall to nothing.
                stop = len(steps) - int(np.argmax(np.array(steps[::-1]) > 0.003 * config.frame_dt - 1e-9))
                self.assertGreater(len(steps) - stop, 20)
                self.assertTrue(np.all(np.diff(steps[stop:]) <= 1e-12))
                self.assertLess(steps[-2], 0.05 * 0.003 * config.frame_dt)
                self.assertAlmostEqual(sum(steps[stop:]), way, delta=2.0 * 0.003 * config.frame_dt)
                # The tool holds for the hold time of a lift (0.5 s, 30 frames), and the lock follows
                # (LockTests); a palm that lagged has arrived, and the least volume is where the rule
                # predicted it.
                held = item["gripper_target_pose"].copy()
                self.assertEqual(scene.run("lock"), 30)
                np.testing.assert_allclose(item["gripper_target_pose"], held, atol=1e-6)
                report = c.report()[0]
                self.assertEqual((report["stage"], report["stitch_complete"], report["ties"]), ("lock", False, 1))
                self.assertAlmostEqual(report["least_volume_ratio"], predicted, delta=2e-3)
                self.assertLessEqual(
                    abs(report["least_volume_ratio"] - 0.5),
                    -report["tighten_slope_per_m"] * report["tighten_way_to_rest_m"],
                )
                self.assertLess(report["tighten_travel_m"], report["tighten_travel_bound_m"])

    def test_the_slope_of_the_rule_is_the_secant_over_the_last_way_to_rest_of_the_palms_travel(self):
        """Fails if the rule takes its slope over another length of travel than the 0.75 mm it extrapolates over."""
        # Tissue whose least volume falls by 0.05 per mm up to 2 mm of travel and by 0.5 per mm from there
        # on: the stop begins while the window of the slope still reaches back over the kink.
        kink, slow, fast = 0.002, 50.0, 500.0

        def law(travel):
            return 1.0 - slow * travel if travel <= kink else 1.0 - slow * kink - fast * (travel - kink)

        _geometry, config, c, item, state, scene = self.tightening(law=law)
        seen = []
        while c.report()[0]["tighten_predicted_volume"] is None:
            # What the call that follows sees: the palm's travel and the least volume of the frame before.
            seen.append((scene.travel, item["diagnostics"]["least_volume_ratio"]))
            scene.tick()
            self.assertLess(len(seen), 400)
        report = c.report()[0]
        travel, volume = seen[-1]
        window = 0.5 * 0.003 * 0.5
        first = next(sample for sample in seen if sample[0] >= travel - window)
        self.assertLess(first[0], kink)
        self.assertGreater(travel, kink)
        # The slope is the secant from the first sample inside the last 0.75 mm of the palm's travel:
        # at full speed a sample every 0.05 mm, so it reaches back by 0.70 mm at least.
        self.assertGreater(travel - first[0], window - 0.003 * config.frame_dt - 1e-9)
        self.assertLessEqual(travel - first[0], window + 1e-9)
        slope = (volume - first[1]) / (travel - first[0])
        self.assertAlmostEqual(report["tighten_slope_per_m"], slope, delta=1e-6 * abs(slope))
        self.assertAlmostEqual(report["tighten_way_to_rest_m"], window, delta=1e-6)
        # By hand, x the travel beyond the kink in mm, the window 0.75 mm, the palm at its target:
        # J = 0.9 - 0.5 x, J_w = 0.9 + 0.05 (0.75 - x), J_rest = J + (J - J_w) = 0.8625 - 0.95 x <= 0.5
        # from x = 0.3816 mm on. A slope over 1.5 mm would begin the stop 0.12 mm later (x >= 0.5 mm).
        self.assertGreater(travel, kink + 0.0003816 - 2e-6)
        self.assertLess(travel, kink + 0.0003816 + 0.003 * config.frame_dt + 2e-6)
        self.assertAlmostEqual(report["tighten_predicted_volume"], volume + slope * window, delta=1e-4)
        self.assertLessEqual(report["tighten_predicted_volume"], 0.5)

    def test_a_stop_that_begins_in_the_start_ramp_takes_the_way_of_the_speed_it_has_reached(self):
        """Fails if the way to rest is taken for the full speed while the start ramp has not reached it."""
        # Tissue six times as tight as the measured one: the volume is at the rule after 0.56 mm.
        length = 0.00677 / 6.0
        _geometry, config, c, item, state, scene = self.tightening(length=length)
        scene.run("hold")
        report = c.report()[0]
        # From the speed v u² (3 - 2 u) the ramp to rest covers v T (u³ - u⁴ / 2), as much as the start ramp
        # had covered when it reached that speed: the stop begins at half of the travel to the rule.
        way, travel = report["tighten_way_to_rest_m"], report["tighten_travel_m"]
        self.assertLess(way, 0.00075)
        self.assertAlmostEqual(way, 0.5 * travel, delta=1e-9)
        self.assertGreaterEqual(travel, 0.5 * length - 1e-9)
        # Within the travel of two frames at the speed that the ramp had reached there (2.1 mm/s).
        self.assertLess(travel, 0.5 * length + 2.0 * 0.0021 * config.frame_dt)
        self.assertEqual(report["reason"], None)

    def test_a_thread_that_does_not_come_tight_stops_the_procedure_at_the_travel_bound(self):
        """Fails if a tightening whose rule never answers goes on beyond its bound or ends as if it were tight."""
        # The least volume does not fall: the thread slides, or was not tied where it was said to be.
        _geometry, config, c, item, state, scene = self.tightening(slack=1.0)
        bound = c.tighten_travel_bound
        # The travel at which the tissue under the tie collapsed when it was pulled on without a stop (7.7 mm,
        # device), rounded down; the table at the top of the module names it.
        self.assertAlmostEqual(bound, 0.0075, delta=1e-12)
        steps = []
        while c.report()[0]["phase"] != "stopped":
            before = state.tighten_travel
            scene.tick()
            steps.append(state.tighten_travel - before)
            self.assertLess(len(steps), 400)
        report = c.report()[0]
        self.assertEqual((report["stage"], report["stitch_complete"]), ("tighten", False))
        self.assertAlmostEqual(report["tighten_travel_m"], bound, delta=1e-12)
        self.assertRegex(report["reason"], r"^The thread did not come tight: the tool moved the needle on by 7\.50 mm")
        self.assertIsNone(report["tighten_predicted_volume"])
        # The command came to rest at the bound by the ramp of the insertion run backwards.
        self.assertLess(steps[-2], 0.05 * 0.003 * config.frame_dt)
        self.assertAlmostEqual(sum(steps), bound, delta=1e-12)
        # Tissue whose volume falls only inside the ramp to the bound, from 6.75 mm on and to the rule within
        # 0.5 mm: the rule ends the tightening, where that ramp ends: at the bound, with no reason to stop.
        _geometry, config, c, item, state, scene = self.tightening(slack=0.00675, length=0.001)
        scene.run("hold")
        report = c.report()[0]
        self.assertEqual((report["stage"], report["reason"]), ("hold", None))
        self.assertAlmostEqual(report["tighten_travel_m"], bound, delta=1e-12)
        self.assertLessEqual(report["tighten_predicted_volume"], 0.5)

    def test_the_way_of_the_tightening_is_asked_for_reach_to_its_bound_before_it_begins(self):
        """Fails if the tool starts a tightening whose line it cannot follow to the travel bound."""
        geometry, config, c, item, state = clear_of_the_tissue()
        scene = Stitched(c, item, config)
        asked = []
        reach = c._reach
        c._reach = lambda poses: (asked.append(np.array(poses)), reach(poses))[1]
        scene.run("tighten")
        (row,) = asked
        # From the target of the tool, straight along the direction, a pose every translation step of the
        # actions at most, to the bound; the tool does not turn.
        np.testing.assert_allclose(row[0], item["gripper_target_pose"], atol=1e-12)
        np.testing.assert_allclose(
            row[-1, :3] - row[0, :3], c.tighten_travel_bound * state.tighten_direction, atol=1e-12
        )
        self.assertEqual(len(row), int(np.ceil(c.tighten_travel_bound / config.translation_step)) + 1)
        self.assertLessEqual(np.linalg.norm(np.diff(row[:, :3], axis=0), axis=1).max(), config.translation_step + 1e-12)
        np.testing.assert_array_equal(row[:, 3:], np.tile(row[0, 3:], (len(row), 1)))
        margins = (c.report()[0]["joint_margin_rad"], c.report()[0]["joint_margin_m"])
        self.assertTrue(np.isfinite(margins).all())
        # A line that the tool does not reach: the procedure stops before the first target, with the joint.
        geometry, config, c, item, state = clear_of_the_tissue()
        scene = Stitched(c, item, config)
        held = item["gripper_target_pose"].copy()
        c._reach = lambda poses: ("roll is 0.01 rad inside its limit where 0.03 rad are needed", 0.01, 0.02)
        scene.run("tighten")
        report = c.report()[0]
        self.assertEqual((report["phase"], report["stage"]), ("stopped", "tie"))
        self.assertEqual(
            report["reason"],
            "The tool cannot move the needle on to tighten the thread within its joint limits: "
            "roll is 0.01 rad inside its limit where 0.03 rad are needed",
        )
        np.testing.assert_allclose(item["gripper_target_pose"], held, atol=1e-6)

    def test_the_rule_needs_the_least_volume_of_the_scene_and_stops_at_once_where_the_tissue_is_at_it(self):
        """Fails if the tightening begins without the number it stops by, or moves on in tissue that is at the rule."""
        geometry, config, c, item, state = clear_of_the_tissue()
        scene = Stitched(c, item, config, volume=False)
        scene.run("tighten")
        report = c.report()[0]
        self.assertEqual((report["phase"], report["stage"]), ("stopped", "tie"))
        self.assertEqual(
            report["reason"], "The scene does not report the least tissue volume that the tightening stops by"
        )
        # Tissue that is at the rule before the tool has moved: the tool holds where it is. Nothing was
        # tightened by the rule, so the report has no slope and no travel, which the self-test asks for.
        _geometry, config, c, item, state, scene = self.tightening(start=0.5)
        held = item["gripper_target_pose"].copy()
        scene.run("done")
        report = c.report()[0]
        np.testing.assert_allclose(item["gripper_target_pose"], held, atol=1e-6)
        self.assertEqual((report["tighten_travel_m"], report["tighten_slope_per_m"]), (0.0, None))
        self.assertEqual(report["tighten_predicted_volume"], 0.5)

    def test_the_report_has_the_needle_in_the_jaws_and_the_calls_without_both_jaws(self):
        """Fails if the stitch does not report how the needle moved in the jaws or counts time without a grasp."""
        _geometry, config, c, item, state, scene = self.tightening()
        for _ in range(40):
            scene.tick()
        report = c.report()[0]
        self.assertEqual(
            (report["tighten_slip_m"], report["tighten_slip_rad"], report["stitch_calls_without_grasp"]), (0.0, 0.0, 0)
        )
        # The needle slips 0.02 mm along the jaws and turns 20 mrad in them.
        turn = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.02), dtype=float)
        item["needle_relative_pose"] = times(item["needle_relative_pose"], np.r_[0.0, 0.0, 0.0, turn])
        item["needle_relative_pose"][:3] += (0.0, 0.00002, 0.0)
        scene.tick()
        report = c.report()[0]
        self.assertAlmostEqual(report["tighten_slip_m"], 0.00002, delta=1e-9)
        self.assertAlmostEqual(report["tighten_slip_rad"], 0.02, delta=2e-4)
        # The largest values are kept, and the pull's own numbers (against the grasp of the pull) go on.
        item["needle_relative_pose"][:3] -= (0.0, 0.00001, 0.0)
        scene.tick()
        self.assertAlmostEqual(c.report()[0]["tighten_slip_m"], 0.00002, delta=1e-9)
        self.assertAlmostEqual(c.report()[0]["relative_translation_m"], 0.00001, delta=1e-9)
        # Two calls with a jaw off the needle: counted, and the travel does not go on in them.
        travel = state.tighten_travel
        scene.tick(contact=False)
        scene.tick(contact=False)
        report = c.report()[0]
        self.assertEqual((report["stitch_calls_without_grasp"], report["phase"]), (2, "grasp_lost"))
        self.assertEqual(state.tighten_travel, travel)
        # Both jaws again: the tightening goes on from the next call (the time without them does not
        # count); lost for the time of a lost contact, the procedure stops.
        scene.tick()
        self.assertEqual((c.report()[0]["phase"], state.tighten_travel), ("tightening", travel))
        scene.tick()
        self.assertGreater(state.tighten_travel, travel)
        for _ in range(16):
            scene.tick(contact=False)
        report = c.report()[0]
        self.assertEqual(
            (report["phase"], report["reason"]), ("stopped", "Bilateral needle contact lost for 0.25 seconds")
        )
        self.assertEqual(report["stitch_calls_without_grasp"], 18)


class LockTests(unittest.TestCase):
    """The end of the stitch: the tie at the exit, the slackening, the hold. No physics: :class:`Stitched` answers."""

    def locked(self, **tissue):
        """The fixture in the call that has begun the slackening: both ties are made and the tool has not moved back."""
        geometry, config, c, item, state = clear_of_the_tissue()
        scene = Stitched(c, item, config, **tissue)
        scene.run("slacken")
        self.assertEqual((state.stage, c.report()[0]["reason"]), ("slacken", None))
        return geometry, config, c, item, state, scene

    def test_the_tie_at_the_exit_is_asked_for_once_after_the_hold_and_the_scene_must_report_two_ties(self):
        """Fails if the second tie is asked for before the hold has ended, at another eyelet, twice, or not checked."""
        geometry, config, c, item, state = clear_of_the_tissue()
        scene = Stitched(c, item, config)
        scene.run("hold")
        held = item["gripper_target_pose"].copy()
        # The hold of the tightened thread: 0.5 s, 30 frames, with nothing asked of the scene.
        self.assertEqual(scene.run("lock"), 30)
        self.assertEqual(scene.asked, [("tie_thread", {})])
        self.assertEqual(c.passages, PASSAGES)
        # The first call of the stage asks for the tie where the thread leaves the last planned passage,
        # and holds the tool.
        action = scene.tick()
        self.assertEqual(scene.commands[-1], [(LOCK,)])
        self.assertEqual((state.stage, c.report()[0]["phase"]), ("lock", "tying"))
        np.testing.assert_array_equal(action[:3], np.zeros(3))
        # The next call sees both ties in the scene's report and begins the slackening; it asks for nothing
        # and the tool has not moved yet.
        scene.tick()
        self.assertIsNone(scene.commands[-1])
        self.assertEqual((state.stage, c.report()[0]["ties"]), ("slacken", 2))
        np.testing.assert_allclose(item["gripper_target_pose"], held, atol=1e-6)
        scene.run("done")
        self.assertEqual(scene.asked, [("tie_thread", {}), LOCK])
        self.assertEqual(
            [commands for commands in scene.commands if commands is not None], [[(("tie_thread", {}),)], [(LOCK,)]]
        )
        report = c.report()[0]
        self.assertEqual((report["stitch_complete"], report["ties"], report["reason"]), (True, 2, None))
        # A scene that does not make the second tie: the procedure stops with what the scene reports.
        geometry, config, c, item, state = clear_of_the_tissue()
        scene = Stitched(c, item, config, ties=1)
        scene.run("lock")
        scene.tick()
        scene.tick()
        report = c.report()[0]
        self.assertEqual((report["phase"], report["stage"], report["stitch_complete"]), ("stopped", "lock", False))
        self.assertEqual(
            report["reason"], "The thread was not tied at its exit: the scene reports 1 ties of 2 asked for"
        )
        # A scene whose observation does not say how the thread pulls the needle: nothing to slacken by.
        geometry, config, c, item, state = clear_of_the_tissue()
        scene = Stitched(c, item, config, report=False)
        scene.run("lock")
        scene.tick()
        scene.tick()
        report = c.report()[0]
        self.assertEqual((report["phase"], report["stage"]), ("stopped", "lock"))
        self.assertEqual(
            report["reason"], "The scene does not report the thread's pull on the needle that the slackening stops by"
        )

    def test_the_needle_goes_back_along_the_line_of_the_tightening_at_the_ramp_of_the_insertion(self):
        """Fails if the slackening leaves the line of the tightening, turns the tool or starts with a step in speed."""
        # A thread that stays taut for the whole way back: the pull falls by 1 mN per mm from 17.6 mN.
        _geometry, config, c, item, state, scene = self.locked(release=1.0)
        direction, origin = state.tighten_direction, state.tighten_origin
        start = item["gripper_target_pose"].copy()
        # The tool stands where the tightening ended: its travel along the direction from where it began.
        np.testing.assert_allclose(start[:3], origin[:3] + state.tighten_travel * direction, atol=1e-9)
        travelled = []
        for frame in range(1, 46):
            action = scene.tick()
            moved = item["gripper_target_pose"][:3] - start[:3]
            travelled.append(float(-moved @ direction))
            # Straight back along the line, without turning; the jaws stay closed.
            np.testing.assert_allclose(moved, -travelled[-1] * direction, atol=1e-9)
            self.assertLessEqual(np.linalg.norm(action[3:6]) * config.rotation_step, 4 * np.finfo(np.float32).eps)
            self.assertEqual(action[6], 0.0)
            # The travel is that of the ramp of the insertion: 0.75 mm in its 0.5 s, then 3 mm/s.
            self.assertAlmostEqual(travelled[-1], float(ramp_distance(frame * config.frame_dt, 0.003)), delta=2e-9)
            self.assertAlmostEqual(c.report()[0]["slacken_travel_m"], travelled[-1], delta=2e-9)
        self.assertEqual((state.stage, c.report()[0]["phase"]), ("slacken", "slackening"))

    def test_the_stop_begins_under_the_weight_of_the_needle_and_ends_where_the_slope_predicts_a_slack_thread(self):
        """Fails if the slackening runs the way of a stop from full speed on into the thread that it pushes together."""
        way = 0.00075
        for tissue in ({}, {"release": 6.0}, {"release": 24.0, "lag": 0.5}, {"slack": 0.001, "lag": 0.3}):
            with self.subTest(tissue=tissue):
                _geometry, config, c, item, state, scene = self.locked(**tissue)
                release = tissue.get("release", 12.0)
                travel, pull = scene.locked
                self.assertEqual(c.needle_weight, WEIGHT)
                self.assertGreater(pull, 3.0 * WEIGHT)
                # By hand: the thread is slack where the travel back is pull / release (1.5 mm for the 17.6 mN
                # of the hold and 12 mN per mm), and its pull is below the weight 0.245 mm before that.
                slack, under = pull / release, (pull - WEIGHT) / release
                steps, pulls = [], []
                while state.stage == "slacken":
                    before = state.slack_travel
                    pulls.append(item["thread_tension"]["needle_force_N"])
                    scene.tick()
                    steps.append(state.slack_travel - before)
                    self.assertLess(len(steps), 400)
                    # What the slope is taken over: the commanded travel back at the call and the pull of the
                    # observation that the call acted on, not the travel of the palm, which lags.
                    self.assertEqual(state.slack_samples[-1], (before, pulls[-1]))
                report = c.report()[0]
                self.assertEqual((report["stage"], report["phase"], report["reason"]), ("hold", "holding", None))
                # The stop began in the first frame whose observation had the pull below the weight.
                begun = int(np.argmax(np.array(pulls) < WEIGHT))
                self.assertTrue(all(value >= WEIGHT for value in pulls[:begun]))
                self.assertEqual(report["slacken_stop_force_N"], pulls[begun])
                self.assertLess(report["slacken_stop_force_N"], WEIGHT)
                self.assertEqual(report["needle_weight_N"], WEIGHT)
                # The slope that the slackening measured itself is that of the thread, whatever the palm lags.
                # (A palm that lags catches up while the command slows down: it then moves a little more than
                # the command over the same window.)
                self.assertAlmostEqual(report["slacken_slope_N_per_m"], -release, delta=0.05 * release)
                # It ended where the thread is slack, within the travel of one frame at speed (0.05 mm) and
                # what the command leads a palm that lags (lag / (1 - lag) frames): the stop was that of the
                # speed to which the command had slowed down, where a stop from the full speed would have
                # gone 0.75 mm beyond the point at which the pull fell below the weight.
                back = report["slacken_travel_m"]
                lag = tissue.get("lag", 0.0)
                self.assertAlmostEqual(back, slack, delta=(1.0 + lag / (1.0 - lag)) * 0.003 * config.frame_dt)
                self.assertGreater(under + way - slack, 0.0002)
                self.assertLess(back, travel)
                # Until the stop began the command went no faster than the stop ramp allows for the travel
                # that the measured slope leaves to the slack thread, and from there it only slowed down.
                self.assertLess(steps[begun - 1], 0.003 * config.frame_dt)
                self.assertTrue(np.all(np.diff(steps[begun:]) <= 1e-12))
                self.assertLess(steps[-2], 0.1 * 0.003 * config.frame_dt)
                # The tool holds for the hold time of a lift (0.5 s, 30 frames) and the procedure is done; the
                # palm has arrived, and the thread pulls the needle with less than a fifth of the weight.
                held = item["gripper_target_pose"].copy()
                ended = item["diagnostics"]["physical_time_s"]
                self.assertEqual(scene.run("done"), 30)
                np.testing.assert_allclose(item["gripper_target_pose"], held, atol=1e-6)
                scene.tick()
                report = c.report()[0]
                self.assertEqual((report["stage"], report["stitch_complete"], report["ties"]), ("done", True, 2))
                self.assertLess(report["thread_needle_force_N"], 0.2 * WEIGHT)
                self.assertAlmostEqual(report["slacken_end_time_s"], ended, delta=1e-9)

    def test_without_a_slope_the_stop_takes_the_way_of_the_speed_that_the_start_ramp_has_reached(self):
        """Fails if a stop without a measured slope ends at once or runs on to the bound."""
        # A thread whose pull does not change over the way back and then is gone at once: 1.2 mm back
        # nothing holds it any more (its pull and its push are those of a thread that carries nothing).
        _geometry, config, c, item, state, scene = self.locked(release=1e-9, push=0.0)
        pull = scene.locked[1]
        steps = []
        while state.stage == "slacken":
            if state.slack_travel >= 0.0012:
                scene.locked = (scene.locked[0], 0.0)
                scene.observe()
            before = state.slack_travel
            scene.tick()
            steps.append(state.slack_travel - before)
            self.assertLess(len(steps), 400)
        report = c.report()[0]
        self.assertEqual((report["stage"], report["reason"]), ("hold", None))
        # The pull fell from its full value to nothing between two frames, at the full speed: no slope had
        # slowed the command, and the stop is the ramp of the insertion run backwards from there, 0.75 mm.
        self.assertEqual(report["slacken_stop_force_N"], 0.0)
        self.assertLess(report["slacken_slope_N_per_m"], -pull / 0.00076)
        self.assertGreaterEqual(report["slacken_travel_m"], 0.0012 + 0.00075 - 1e-9)
        self.assertLess(report["slacken_travel_m"], 0.0012 + 0.00075 + 0.003 * config.frame_dt)
        stop = len(steps) - int(np.argmax(np.array(steps[::-1]) > 0.003 * config.frame_dt - 1e-9))
        self.assertGreater(len(steps) - stop, 20)
        self.assertTrue(np.all(np.diff(steps[stop:]) <= 1e-12))
        self.assertLess(steps[-2], 0.05 * 0.003 * config.frame_dt)
        # A thread that is slack when it is tied at the exit: the tool does not move at all.
        _geometry, config, c, item, state, scene = self.locked(taut=0.5)
        self.assertLess(scene.locked[1], WEIGHT)
        held = item["gripper_target_pose"].copy()
        scene.run("done")
        report = c.report()[0]
        np.testing.assert_allclose(item["gripper_target_pose"], held, atol=1e-6)
        self.assertEqual((report["slacken_travel_m"], report["slacken_slope_N_per_m"]), (0.0, None))
        self.assertEqual(report["slacken_stop_force_N"], scene.locked[1])
        # A pull that is below the weight in the sixth frame of the start ramp, before a slope can be
        # measured (the command has not gone the travel of one frame at speed): the stop takes the way of
        # the speed that the ramp has reached, about as much again as the ramp has covered.
        _geometry, config, c, item, state, scene = self.locked()
        for _ in range(5):
            scene.tick()
        scene.locked = (scene.locked[0], 0.5 * WEIGHT)
        scene.release = 0.0
        scene.observe()
        covered, rate = state.slack_travel, state.slack_rate
        self.assertLess(covered, 0.003 * config.frame_dt)
        self.assertAlmostEqual(rate, (covered - float(ramp_distance(4 * config.frame_dt, 0.003))) / config.frame_dt)
        scene.run("hold")
        report = c.report()[0]
        self.assertEqual(report["slacken_slope_N_per_m"], None)
        self.assertAlmostEqual(report["slacken_travel_m"], covered + procedure.ramp_way(rate, 0.003), delta=1e-12)
        self.assertGreater(report["slacken_travel_m"], 1.5 * covered)
        self.assertLess(report["slacken_travel_m"], 2.0 * covered)

    def test_the_slackening_ends_at_the_latest_where_the_tightening_began(self):
        """Fails if the tool takes the needle back beyond where the tightening began, or stops the procedure there."""
        # A thread that stays taut over the whole way back (1 mN per mm): the command comes to rest by the
        # ramp where the tightening began, and the procedure goes on to its end; the report says that the
        # stop was not that of the rule.
        _geometry, config, c, item, state, scene = self.locked(release=1.0)
        travel = state.tighten_travel
        steps = []
        while state.stage == "slacken":
            before = state.slack_travel
            scene.tick()
            steps.append(state.slack_travel - before)
            self.assertLess(len(steps), 400)
        report = c.report()[0]
        self.assertEqual((report["stage"], report["reason"]), ("hold", None))
        self.assertAlmostEqual(report["slacken_travel_m"], travel, delta=1e-12)
        np.testing.assert_allclose(item["gripper_target_pose"], state.tighten_origin, atol=1e-9)
        self.assertIsNone(report["slacken_stop_force_N"])
        self.assertLess(steps[-2], 0.05 * 0.003 * config.frame_dt)
        self.assertAlmostEqual(sum(steps), travel, delta=1e-12)
        scene.run("done")
        scene.tick()
        report = c.report()[0]
        self.assertEqual((report["stitch_complete"], report["reason"]), (True, None))
        self.assertGreater(report["thread_needle_force_N"], WEIGHT)
        # A thread that goes slack only inside the ramp to the bound: the stop of the rule ends no later.
        _geometry, config, c, item, state, scene = self.locked(release=5.2)
        scene.run("hold")
        report = c.report()[0]
        self.assertLessEqual(report["slacken_travel_m"], state.tighten_travel)
        self.assertLess(report["slacken_stop_force_N"], WEIGHT)
        # A scene that applies no weight to its needle: no pull is below it, and the way back is the whole.
        geometry, config, c, item, state = clear_of_the_tissue()
        c.needle_weight = 0.0
        scene = Stitched(c, item, config)
        scene.run("slacken")
        scene.run("hold")
        report = c.report()[0]
        self.assertAlmostEqual(report["slacken_travel_m"], report["tighten_travel_m"], delta=1e-12)
        self.assertIsNone(report["slacken_stop_force_N"])

    def test_the_report_follows_the_needle_in_the_jaws_from_the_tie_at_the_exit_and_the_pull_to_the_end(self):
        """Fails if the slip since the lock is that since the tightening, or the report stops following when done."""
        geometry, config, c, item, state = clear_of_the_tissue()
        scene = Stitched(c, item, config)
        scene.run("tighten")
        # The needle slips 0.03 mm along the jaws while the thread is tightened.
        for _ in range(20):
            scene.tick()
        item["needle_relative_pose"][:3] += (0.0, 0.00003, 0.0)
        scene.run("slacken")
        report = c.report()[0]
        self.assertAlmostEqual(report["tighten_slip_m"], 0.00003, delta=1e-9)
        self.assertEqual((report["lock_slip_m"], report["lock_slip_rad"]), (0.0, 0.0))
        # It turns 15 mrad in the jaws while the thread goes slack.
        for _ in range(10):
            scene.tick()
        turn = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.015), dtype=float)
        item["needle_relative_pose"] = times(item["needle_relative_pose"], np.r_[0.0, 0.0, 0.0, turn])
        scene.tick()
        report = c.report()[0]
        self.assertAlmostEqual(report["lock_slip_rad"], 0.015, delta=2e-4)
        self.assertAlmostEqual(report["tighten_slip_rad"], 0.015, delta=2e-4)
        self.assertLess(report["lock_slip_m"], 1e-9)
        # Two calls with a jaw off the needle: counted, and the travel back does not go on in them.
        back = state.slack_travel
        scene.tick(contact=False)
        scene.tick(contact=False)
        report = c.report()[0]
        self.assertEqual((report["stitch_calls_without_grasp"], report["phase"]), (2, "grasp_lost"))
        self.assertEqual(state.slack_travel, back)
        scene.tick()
        scene.run("done")
        # Done: the report goes on with the thread's force on the needle and with the needle in the jaws.
        scene.tick()
        self.assertEqual(c.report()[0]["thread_needle_force_N"], item["thread_tension"]["needle_force_N"])
        scene.locked = (scene.locked[0], scene.locked[1] + 0.004)
        scene.observe()
        item["needle_relative_pose"][:3] += (0.0, 0.00002, 0.0)
        scene.tick()
        report = c.report()[0]
        self.assertEqual(report["thread_needle_force_N"], item["thread_tension"]["needle_force_N"])
        self.assertAlmostEqual(report["lock_slip_m"], 0.00002, delta=1e-9)
        self.assertEqual((report["stage"], report["stitch_calls_without_grasp"]), ("done", 2))


class PlanReachTests(unittest.TestCase):
    NEEDLE = straight_pose(np.radians(60.0), 0.012)
    # The exit marker beside the tail of the wire, and a thread element of two wire radii: the pull-out arc of the
    # plan is the two elements past the marker alone, four wire radii.
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
            thread_tracks=True,
            thread_element_length=2.0 * WIRE_RADIUS,
            needle_weight_N=WEIGHT,
            grippers=[pulling_tool(limits, trocar)],
            reference_positions=vertices,
            reference_triangles=triangles,
        )
        geometry["insertion_plan"]["exit_angle"] = 0.9
        geometry["insertion_plan"]["expected_passages"] = PASSAGES
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
        # its centre by the four wire radii of the two thread elements, in steps of no more than 0.01 rad, the tool held to
        # it as at the grasp.
        along = procedure._rotation(grasp)[:, 1]
        descent = np.tile(grasp, (121, 1))
        descent[:, :3] -= np.linspace(0.012, 0.0, 121)[:, None] * along
        span = 4.0 * WIRE_RADIUS / c.radius
        self.assertAlmostEqual(c._pull_out_span(self.NEEDLE, marker), span, delta=1e-6)
        angles = np.linspace(0.0, c._pull_out_span(self.NEEDLE, marker), int(np.ceil(span / 0.01)) + 1)
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
