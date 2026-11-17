"""Where the regrasp takes the needle by its tip: the rule against computations by hand.

A needle sticks out of a flat tissue surface (the plane z = 0, tissue below). The jaw capsules of the tool
model, the tissue surface and the resting needle decide the grasp point; nothing here steps physics. The
expected values are closed forms: against a plane the clearance of a capsule is the height of the lowest
point of its axis less its radius, and that height is linear along the axis and along the descent and a
sinusoid in the jaw angle, so its smallest value is at one of a few points that can be written down.
"""

import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import warp as wp
from test_grasp_procedure import fixture

from newton_solver_suturebot.examples.grasp._dvrk_grasper import needle_to_gripper_pose
from newton_solver_suturebot.examples.grasp._grasp_procedure import (
    CLEARANCE_REFINEMENT,
    CLEARANCE_SPACING_RADII,
    PLAN_CAPSULES_PER_STEP,
    ProceduralGraspController,
    TissueSurface,
    jaw_capsules,
    jaw_clearance,
)

# The jaw capsules of the dVRK tool model (add_rcm_grasper) and the radius of the drawn jaw about them.
MODEL_RADIUS = 0.00065
DRAWN_RADIUS = 0.000976
WIRE_RADIUS = 0.00028
BITE = np.array([0.0, 0.006, 0.0])


# The trocar of the tool of these tests: a quarter of a metre above the needle of the procedure fixture.
TROCAR = np.array([0.03, -0.04, 0.26, 0.0, 0.0, 0.0, 1.0])


def tool(contact_radius=DRAWN_RADIUS, model_radius=MODEL_RADIUS):
    """One entry of the scene's ``grippers``: shaft, wrist and the two jaw capsules, bodies 4, 5, 7 and 8.

    With what places the shaft and the wrist for a palm pose: the links of the bodies, the trocar and the
    slide of the shaft (0.18 m between the trocar and the wrist at rest, 0.05 m either way).
    """
    result = {
        "capsule_bodies": [4, 5, 7, 8],
        "jaw_body_indices": [7, 8],
        "part_body_indices": {"roll": 4, "pitch": 5, "yaw": 6, "jaw1": 7, "jaw2": 8},
        "trocar_pose": TROCAR.copy(),
        "nominal_insertion_m": 0.18,
        "insertion_limits_m": (-0.05, 0.05),
        "capsule_endpoints": np.array(
            [
                [[0, 0, -0.243], [0, 0, -0.003]],
                [[0, 0, 0], [0.008, 0, 0]],
                [[-0.00075, 0.0015, 0], [-0.00075, 0.009, 0]],
                [[0.00075, 0.0015, 0], [0.00075, 0.009, 0]],
            ]
        ),
        "capsule_radii": np.array([0.0043, 0.0022, model_radius, model_radius]),
    }
    if contact_radius is not None:
        result["capsule_contact_radii"] = np.array([0.0042, 0.003, contact_radius, contact_radius])
    return result


def flat_tissue(size=0.05, step=0.001, centre=(0.0, 0.0)):
    """A square of the plane z = 0 in right triangles with legs ``step``; the tissue is below it."""
    count = int(round(size / step))
    axis = (np.arange(count + 1) - 0.5 * count) * step
    x, y = np.meshgrid(axis + centre[0], axis + centre[1], indexing="ij")
    vertices = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=1)
    index = np.arange((count + 1) ** 2).reshape(count + 1, count + 1)
    a, b, c, d = index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    return vertices, np.concatenate((np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1)))


def straight_needle(length=0.02, count=41):
    """Centre line of a straight wire in the needle frame: the tip at the origin, the tail along +x."""
    return np.stack((np.linspace(0.0, length, count), np.zeros(count), np.zeros(count)), axis=1)


def arc_needle(radius, span=1.2, count=481):
    """Centre line of a circular wire in the needle frame, tip first, as the procedure fixture has it."""
    behind = np.linspace(0.0, span, count)
    return np.stack((radius * (np.cos(behind) - 1.0), -radius * np.sin(behind), np.zeros(count)), axis=1)


def pose_from(rotation, translation):
    quaternion = np.asarray(
        wp.quat_from_matrix(wp.mat33(*np.asarray(rotation, dtype=np.float64).reshape(-1))), dtype=float
    )
    return np.r_[np.asarray(translation, dtype=float), quaternion / np.linalg.norm(quaternion)]


def rotation_of(pose):
    return np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=float).reshape(3, 3)


def straight_pose(exit_angle, tip_height):
    """Pose of the straight needle: the tail goes down into the plane at ``exit_angle`` to it, in the x-z plane."""
    tail = np.array([np.cos(exit_angle), 0.0, -np.sin(exit_angle)])
    normal = np.array([0.0, 1.0, 0.0])
    return pose_from(np.column_stack((tail, np.cross(normal, tail), normal)), (0.0, 0.0, tip_height))


def controller_for(centerline, tool_entry, *, roll=None, bite=None, tissue=None):
    """A controller whose regrasp sees the given needle, tool and flat tissue."""
    geometry, config = fixture()
    vertices, triangles = flat_tissue() if tissue is None else tissue
    geometry.update(
        entry_normal=np.array([0.0, 0.0, 1.0]),
        needle_span=2.375,
        needle_tip_local=centerline[0],
        needle_centerline=centerline,
        needle_radius=WIRE_RADIUS,
        grippers=[tool_entry],
        reference_positions=vertices,
        reference_triangles=triangles,
    )
    geometry["insertion_plan"]["exit_angle"] = 0.9
    controller = ProceduralGraspController(
        geometry, config, settle_seconds=0, lift_height=0.001, tip_lift_height=0.0005
    )
    if roll is not None:
        controller.tip_grasp_roll_rad = roll
    if bite is not None:
        controller.tip_grasp_bite_rad = bite
    return controller


def planned(controller, needle_pose):
    """Run the plan of the grasp at the tip to its end: (pose, offset) or None, and the number of its steps."""
    plan, steps = controller._plan_tip_grasp(needle_pose), 0
    while True:
        steps += 1
        try:
            next(plan)
        except StopIteration as done:
            return done.value, steps


def lowest_axis_height(controller, needle_pose, offset, capsule):
    """Height above z = 0 of the lowest point that the axis of a jaw capsule reaches, by hand.

    A point of the axis at jaw angle t and descent d is ``O + (x cos t - y sin t) X + (x sin t + y cos t) Y +
    z Z - d Y`` with the tool axes X, Y, Z and origin O: linear in the point of the axis and in d, and
    ``A cos t + B sin t + const`` in t. Its smallest height is therefore at an end of the axis, at d = 0 or
    the whole descent (open jaw), and at t = 0, the open angle, or where the sinusoid has its minimum.
    """
    start, end, _radius, sense = capsule
    shape = SimpleNamespace(centerline=controller.centerline, arc_normal=controller.arc_axes[2])
    length = float(np.linalg.norm(np.diff(controller.centerline, axis=0), axis=1).sum())
    relative = needle_to_gripper_pose(
        shape,
        arc_fraction=offset / length,
        tangent_roll_rad=controller.tip_grasp_roll_rad,
        bite_angle_rad=controller.tip_grasp_bite_rad,
    )
    frame = rotation_of(needle_pose) @ rotation_of(relative)
    origin = rotation_of(needle_pose) @ relative[:3] + needle_pose[:3]
    x_up, y_up, z_up = frame[2]
    half = 0.5 * controller.release_opening_rad
    heights = []
    for point in (start, end):
        a = point[0] * x_up + point[1] * y_up
        b = -point[1] * x_up + point[0] * y_up
        # A cos t + B sin t is smallest where (cos t, sin t) points against (A, B).
        angles = [0.0, sense * half]
        stationary = np.arctan2(-b, -a)
        if 0.0 <= sense * stationary <= half:
            angles.append(stationary)
        heights.extend(origin[2] + point[2] * z_up + a * np.cos(t) + b * np.sin(t) for t in angles)
        open_height = origin[2] + point[2] * z_up + a * np.cos(sense * half) + b * np.sin(sense * half)
        heights.append(open_height - controller.transfer_clearance * y_up)
    return min(heights)


def exact_offset(controller, needle_pose, upper, margin=0.0):
    """The largest offset up to ``upper`` at which every jaw capsule is clear of the plane by ``margin``.

    By bisection of the closed form, which falls as the grasp nears the tissue.
    """

    def clear(offset):
        return min(lowest_axis_height(controller, needle_pose, offset, c) - c[2] for c in controller.jaws) - margin

    low, high = WIRE_RADIUS, upper
    if clear(high) >= 0:
        return high
    if clear(low) < 0:
        return None
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if clear(middle) >= 0 else (low, middle)
    return low


def missed(radius):
    """What the sampling of the rule can miss of a clearance [m].

    The fine samples lie at most R / 16 apart, and a fine step turns the jaw so that no point of its axis
    moves further than that, of which half is subtracted.
    """
    return 1.5 * CLEARANCE_SPACING_RADII * radius / CLEARANCE_REFINEMENT


class JawCapsuleTests(unittest.TestCase):
    def test_contact_capsules_keep_the_reach_of_the_model_capsules_and_read_the_contact_radii(self):
        rows = jaw_capsules(tool())
        self.assertEqual(len(rows), 2)
        for (start, end, radius, sense), side in zip(rows, (-1.0, 1.0), strict=True):
            self.assertEqual(radius, DRAWN_RADIUS)
            self.assertEqual(sense, -side)
            np.testing.assert_allclose(start, [side * 0.00075, 0.0015 + DRAWN_RADIUS - MODEL_RADIUS, 0], atol=1e-12)
            np.testing.assert_allclose(end, [side * 0.00075, 0.009 - DRAWN_RADIUS + MODEL_RADIUS, 0], atol=1e-12)
            # Cap to cap the contact capsule is as long as the model capsule.
            self.assertAlmostEqual(end[1] + radius, 0.009 + MODEL_RADIUS, delta=1e-12)

    def test_without_contact_radii_the_model_radii_are_used_and_follow_the_model(self):
        for radius in (MODEL_RADIUS, 0.002):
            rows = jaw_capsules(tool(contact_radius=None, model_radius=radius))
            for start, end, used, _sense in rows:
                self.assertEqual(used, radius)
                np.testing.assert_allclose((start[1], end[1]), (0.0015, 0.009), atol=1e-12)

    def test_a_capsule_shorter_than_its_contact_radius_allows_is_its_middle(self):
        entry = tool(contact_radius=0.005)
        for start, end, radius, _sense in jaw_capsules(entry):
            self.assertEqual(radius, 0.005)
            np.testing.assert_allclose(start, end, atol=1e-15)
            self.assertAlmostEqual(start[1], 0.00525, delta=1e-12)

    def test_invalid_tools_are_refused(self):
        for change in (
            {"jaw_body_indices": [7]},
            {"capsule_radii": np.array([0.0043, 0.0022, 0.0, 0.00065])},
            {"capsule_contact_radii": np.array([0.004, 0.003, np.nan, 0.001])},
            {"jaw_body_indices": [20, 21]},
        ):
            entry = tool()
            entry.update(change)
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                jaw_capsules(entry)


class TissueSurfaceTests(unittest.TestCase):
    def test_every_point_of_a_triangle_is_within_the_cover_of_a_sample(self):
        vertices = np.array(
            [[0, 0, 0], [0.003, 0, 0], [0, 0.001, 0], [0.004, 0.0005, 0.0002], [0.002, 0.002, 0]], dtype=float
        )
        triangles = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
        surface = TissueSurface(vertices, triangles)
        # Right triangle with legs 3 and 1 mm: not acute, half the hypotenuse.
        self.assertAlmostEqual(surface.cover[0], 0.5 * np.hypot(0.003, 0.001), delta=1e-12)
        rng = np.random.default_rng(3)
        for spacing in (0.0005, 0.0001):
            points, cover = surface.samples(np.arange(3), spacing)
            self.assertLessEqual(cover.max(), spacing)
            per_triangle = len(points) // 3
            for index, corners in enumerate(vertices[triangles]):
                weights = rng.dirichlet(np.ones(3), size=400)
                inside = weights @ corners
                own = points[index * per_triangle : (index + 1) * per_triangle]
                nearest = np.linalg.norm(inside[:, None] - own[None], axis=2).min(axis=1)
                self.assertLessEqual(nearest.max(), cover[index * per_triangle] + 1e-12)

    def test_near_finds_the_triangles_within_a_radius_by_their_bounding_spheres(self):
        vertices, triangles = flat_tissue(size=0.01)
        surface = TissueSurface(vertices, triangles)
        found = surface.near(np.array([0.0, 0.0, 0.002]), 0.0025)
        distance = np.linalg.norm(surface.centres - [0, 0, 0.002], axis=1) - surface.radii
        np.testing.assert_array_equal(found, np.flatnonzero(distance <= 0.0025))
        among = np.arange(0, len(triangles), 3)
        np.testing.assert_array_equal(
            surface.near(np.array([0.0, 0.0, 0.002]), 0.0025, among), np.intersect1d(found, among)
        )


class JawClearanceTests(unittest.TestCase):
    """The clearance of the jaws above a plane against the height of their lowest point less the radius."""

    def test_level_jaws_are_clear_by_their_height_less_the_radius(self):
        surface = TissueSurface(*flat_tissue())
        for radius in (DRAWN_RADIUS, 0.002):
            capsules = jaw_capsules(tool(contact_radius=radius))
            for height in (radius + 0.0005, radius + 0.0001, radius - 0.0002):
                pose = np.r_[0.0, 0.0, height, 0.0, 0.0, 0.0, 1.0]
                value = jaw_clearance(surface, capsules, pose, BITE, 0.3, 0.012)
                with self.subTest(radius=radius, height=height):
                    # A lower bound, within what the sampling can miss.
                    self.assertLessEqual(value, height - radius + 1e-12)
                    if height >= radius:
                        self.assertGreaterEqual(value, height - radius - missed(radius))
                    else:
                        self.assertLess(value, 0.0)

    def test_tilted_jaws_are_measured_at_the_lowest_point_of_their_sweep(self):
        surface = TissueSurface(*flat_tissue())
        capsules = jaw_capsules(tool())
        end = capsules[0][1][1]
        for tilt, descent in ((0.5, 0.012), (-0.4, 0.012), (-0.4, 0.0)):
            # The tool turned about its X axis: its Y axis rises (tilt > 0) or falls with the angle. The jaws
            # turn about the tool's Z axis; with the X axis level only the Y coordinate changes the height.
            # Falling jaws are lowest at the far end of a closed jaw, rising ones at the near end of an open
            # jaw at the start of the descent.
            y_open = capsules[0][0][1] * np.cos(0.3) - 0.00075 * np.sin(0.3)
            drop = end * np.sin(tilt) if tilt < 0 else (y_open - descent) * np.sin(tilt)
            for clearance in (0.0003, -0.0002):
                rotation = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
                pose = pose_from(rotation, (0.0, 0.0, DRAWN_RADIUS + clearance - drop))
                value = jaw_clearance(surface, capsules, pose, BITE, 0.3, descent)
                with self.subTest(tilt=tilt, descent=descent, clearance=clearance):
                    self.assertLessEqual(value, clearance + 1e-7)
                    if clearance > 0:
                        self.assertGreaterEqual(value, clearance - missed(DRAWN_RADIUS) - 1e-7)
                    else:
                        self.assertLess(value, 0.0)

    def test_a_surface_between_the_grasp_point_and_a_jaw_is_not_clear(self):
        # The jaws stand 5 mm under the surface, further from it than their radius: not clear, because the
        # surface lies between the grasp point above it and the jaws.
        surface = TissueSurface(*flat_tissue())
        capsules = jaw_capsules(tool())
        pose = np.r_[0.0, 0.0, -0.005, 0.0, 0.0, 0.0, 1.0]
        self.assertGreater(jaw_clearance(surface, capsules, pose, BITE, 0.0, 0.0), 0.003)
        self.assertLess(jaw_clearance(surface, capsules, pose, BITE + [0.0, 0.0, 0.007], 0.0, 0.0), 0.0)

    def test_two_capsules_of_a_jaw_side_by_side_along_its_hinge_axis_are_each_measured_where_they_are(self):
        # Each jaw has two capsules, 1 mm to either side of the plane in which the jaws turn (the hinge axis
        # is the tool's Z axis); the one listed second is the lower one. The tool points down at 60 degrees
        # and is rolled about its jaws by 0.7 rad, so the hinge axis rises by sin(0.7) cos(60 deg) per metre.
        rows = [
            (np.array([x, 0.0015, z]), np.array([x, 0.009, z]), DRAWN_RADIUS, sense)
            for sense, x in ((1.0, -0.00075), (-1.0, 0.00075))
            for z in (0.001, -0.001)
        ]
        along = np.array([0.5, 0.0, -np.sqrt(0.75)])
        level = np.cross(along, [0.0, 1.0, 0.0])
        hinge = np.cos(0.7) * np.array([0.0, 1.0, 0.0]) + np.sin(0.7) * level
        frame = np.column_stack((np.cross(along, hinge), along, hinge))
        self.assertGreater(hinge[2], 0.3)
        half, descent, origin = 0.3, 0.004, np.array([0.0, 0.0, 0.0103])
        surface = TissueSurface(*flat_tissue())

        def lowest(rows):
            """By hand: the least height of an axis point over the closing and the descent, less the radius.

            A point (x, y, z) of an axis at jaw angle t and descent d is at the height ``O + (x cos t - y sin t)
            X + (x sin t + y cos t) Y + z Z - d Y``, taken along the world's Z: linear in the point and in d,
            and ``a cos t + b sin t`` in t. Its minimum is at an end of the axis, at t = 0, the open angle or
            the stationary angle between them, and at d = 0 or the whole descent with the jaw open.
            """
            x_up, y_up, z_up = frame[2]
            result = np.inf
            for start, end, radius, sense in rows:
                for point in (start, end):
                    a = point[0] * x_up + point[1] * y_up
                    b = -point[1] * x_up + point[0] * y_up
                    angles = [0.0, sense * half]
                    stationary = np.arctan2(-b, -a)
                    if 0.0 <= sense * stationary <= half:
                        angles.append(stationary)
                    heights = [origin[2] + point[2] * z_up + a * np.cos(t) + b * np.sin(t) for t in angles]
                    heights.append(heights[1] - descent * y_up)
                    result = min(result, min(heights) - radius)
            return result

        exact = lowest(rows)
        self.assertGreater(exact, 0.0002)
        # The lower capsule of each pair decides, by the height of the hinge axis between the two.
        self.assertAlmostEqual(lowest(rows[0::2]) - lowest(rows[1::2]), 0.002 * hinge[2], delta=1e-12)
        value = jaw_clearance(surface, rows, pose_from(frame, origin), BITE, half, descent)
        self.assertLessEqual(value, exact + 1e-9)
        self.assertGreaterEqual(value, exact - missed(DRAWN_RADIUS))
        # Lowered by that clearance and what the sampling can miss, the jaws are not clear.
        lowered = pose_from(frame, origin - [0.0, 0.0, exact + 1e-5])
        self.assertLess(jaw_clearance(surface, rows, lowered, BITE, half, descent), 0.0)

    def test_no_surface_near_is_clear(self):
        surface = TissueSurface(*flat_tissue(size=0.004, centre=(0.1, 0.1)))
        pose = np.r_[0.0, 0.0, 0.003, 0.0, 0.0, 0.0, 1.0]
        self.assertEqual(jaw_clearance(surface, jaw_capsules(tool()), pose, BITE, 0.3, 0.012), np.inf)


class TipGraspRuleTests(unittest.TestCase):
    """The grasp point of the rule against the closed form, for straight and curved needles."""

    def check(self, controller, needle_pose, *, free_wire):
        result, steps = planned(controller, needle_pose)
        expected = exact_offset(controller, needle_pose, free_wire)
        self.assertIsNotNone(expected)
        self.assertIsNotNone(result)
        pose, offset = result
        radius = max(capsule[2] for capsule in controller.jaws)
        # The rule is never nearer to the tissue than the exact point, and no further from it than the point
        # at which the exact clearance equals what the sampling can miss, less the end of the search.
        self.assertLessEqual(offset, expected + 1e-9)
        self.assertGreaterEqual(
            offset, exact_offset(controller, needle_pose, free_wire, missed(radius)) - radius / 64 - 1e-7
        )
        # The pose is the grasp at that point, and the jaws are clear there by the closed form.
        np.testing.assert_allclose(pose, controller._tip_grasp(needle_pose, offset), atol=1e-12)
        for capsule in controller.jaws:
            self.assertGreaterEqual(lowest_axis_height(controller, needle_pose, offset, capsule), capsule[2])
        # The two looks at the surface, the free wire, the scan towards the tip and six halvings; a point
        # of the search takes a step for each jaw capsule that has tissue in reach.
        points = 1 + int(np.ceil(free_wire / min(c[2] for c in controller.jaws))) + 6
        self.assertLessEqual(steps, 3 + points * len(controller.jaws) / PLAN_CAPSULES_PER_STEP)
        return offset, expected

    def test_straight_needle_with_level_closing_axis_matches_the_formula_at_several_exit_angles(self):
        # Twist 0: the closing axis is level, so the lowest point is the far end of a closed jaw when the
        # jaws point down: clearance(s) = (H - s sin a) + (y_end - 6 mm) cos(a + bite) - R.
        for radius in (DRAWN_RADIUS, 0.002):
            for exit_angle, bite in (
                (np.pi / 2, np.pi / 4),
                (np.radians(70), np.pi / 4),
                (np.radians(50), np.radians(70)),
            ):
                controller = controller_for(straight_needle(), tool(contact_radius=radius), roll=0.0, bite=bite)
                tip_height = 0.008
                needle_pose = straight_pose(exit_angle, tip_height)
                y_end = controller.jaws[0][1][1]
                formula = (tip_height + (y_end - 0.006) * np.cos(exit_angle + bite) - radius) / np.sin(exit_angle)
                with self.subTest(radius=radius, exit_angle=exit_angle):
                    self.assertLess(np.cos(exit_angle + bite), 0.0)
                    free = controller._free_wire(needle_pose, np.arange(len(controller.tissue.cover)))
                    offset, expected = self.check(controller, needle_pose, free_wire=free)
                    # The pose of the test is held in single precision.
                    self.assertAlmostEqual(expected, formula, delta=2e-8)
                    self.assertLess(offset, free)

    def test_straight_needle_with_the_twist_and_bite_of_the_procedure(self):
        # The twist tilts the closing axis: the open lower jaw decides, at an angle between closed and open.
        for radius in (DRAWN_RADIUS, 0.002):
            for exit_angle in (np.pi / 2, np.radians(70), np.radians(55)):
                controller = controller_for(straight_needle(), tool(contact_radius=radius))
                needle_pose = straight_pose(exit_angle, 0.009)
                with self.subTest(radius=radius, exit_angle=exit_angle):
                    free = controller._free_wire(needle_pose, np.arange(len(controller.tissue.cover)))
                    self.check(controller, needle_pose, free_wire=free)

    def test_curved_needle_at_several_exit_angles(self):
        radius_of_needle = 0.00945
        for jaw_radius in (DRAWN_RADIUS, 0.002):
            for exit_angle, out in ((np.radians(85), 0.008), (np.radians(70), 0.009), (np.radians(55), 0.011)):
                controller = controller_for(arc_needle(radius_of_needle), tool(contact_radius=jaw_radius))
                # The needle frame is placed so that the wire meets the plane ``out`` behind the tip, at the exit angle.
                behind = out / radius_of_needle
                point = radius_of_needle * np.array([np.cos(behind) - 1.0, -np.sin(behind), 0.0])
                tail = np.array([-np.sin(behind), -np.cos(behind), 0.0])
                down = np.array([np.cos(exit_angle), 0.0, -np.sin(exit_angle)])
                side = np.array([0.0, 1.0, 0.0])
                # Rotation that takes the local tail direction to ``down`` and the arc normal to ``side``.
                local = np.column_stack((tail, np.cross([0.0, 0.0, 1.0], tail), [0.0, 0.0, 1.0]))
                world = np.column_stack((down, np.cross(side, down), side))
                rotation = world @ local.T
                needle_pose = pose_from(rotation, -rotation @ point)
                with self.subTest(jaw_radius=jaw_radius, exit_angle=exit_angle):
                    tip = rotation_of(needle_pose) @ controller.centerline[0] + needle_pose[:3]
                    self.assertGreater(tip[2], 0.004)
                    free = controller._free_wire(needle_pose, np.arange(len(controller.tissue.cover)))
                    # The wire is clear until it is one wire radius above the plane, give or take the stations.
                    self.assertLess(free, out)
                    self.assertGreater(free, out - 3.0 * WIRE_RADIUS / np.sin(exit_angle))
                    self.check(controller, needle_pose, free_wire=free)

    def test_a_larger_jaw_radius_moves_the_grasp_towards_the_tip(self):
        needle_pose = straight_pose(np.radians(70), 0.009)
        offsets = []
        for radius in (MODEL_RADIUS, DRAWN_RADIUS, 0.0015, 0.002):
            controller = controller_for(straight_needle(), tool(contact_radius=radius))
            offsets.append(planned(controller, needle_pose)[0][1])
        self.assertTrue(np.all(np.diff(offsets) < 0), offsets)

    def test_the_radii_of_the_tool_model_decide_when_the_scene_gives_no_contact_radii(self):
        needle_pose = straight_pose(np.radians(70), 0.009)
        thin = controller_for(straight_needle(), tool(contact_radius=None, model_radius=MODEL_RADIUS))
        thick = controller_for(straight_needle(), tool(contact_radius=None, model_radius=0.002))
        drawn = controller_for(straight_needle(), tool(contact_radius=DRAWN_RADIUS, model_radius=MODEL_RADIUS))
        thin_offset, thick_offset = planned(thin, needle_pose)[0][1], planned(thick, needle_pose)[0][1]
        self.assertLess(thick_offset, thin_offset)
        self.assertLess(planned(drawn, needle_pose)[0][1], thin_offset)

    def test_a_tip_that_does_not_stick_out_far_enough_has_no_grasp_point(self):
        for radius, tip_height in ((DRAWN_RADIUS, 0.0025), (0.002, 0.003)):
            controller = controller_for(straight_needle(), tool(contact_radius=radius))
            needle_pose = straight_pose(np.pi / 2, tip_height)
            with self.subTest(radius=radius):
                # By hand: even at the lowest admissible point, one wire radius behind the tip, a jaw is not clear.
                self.assertIsNone(exact_offset(controller, needle_pose, tip_height - WIRE_RADIUS))
                result, _steps = planned(controller, needle_pose)
                self.assertIsNone(result)
        # A tip under the surface: no free wire at all.
        controller = controller_for(straight_needle(), tool())
        self.assertIsNone(planned(controller, straight_pose(np.pi / 2, -0.001))[0])
        self.assertEqual(
            controller._free_wire(straight_pose(np.pi / 2, 0.0001), np.arange(len(controller.tissue.cover))), 0.0
        )

    def test_without_tissue_near_the_whole_wire_is_free_and_the_jaws_take_its_far_end(self):
        controller = controller_for(straight_needle(), tool(), tissue=flat_tissue(size=0.004, centre=(0.3, 0.3)))
        result, steps = planned(controller, straight_pose(np.pi / 2, 0.008))
        self.assertAlmostEqual(result[1], 0.02, delta=1e-9)
        # The looks at the whole surface, the free wire, the one point of the search, the end.
        self.assertEqual(steps, 4)

    def test_a_lift_at_the_tip_needs_the_tool_the_surface_and_the_wire_radius(self):
        geometry, config = fixture()
        vertices, triangles = flat_tissue(size=0.01)
        geometry.update(
            entry_normal=np.array([0.0, 0.0, 1.0]),
            needle_span=2.375,
            needle_centerline=straight_needle(),
            needle_radius=WIRE_RADIUS,
            grippers=[tool()],
            reference_positions=vertices,
            reference_triangles=triangles,
        )
        geometry["insertion_plan"]["exit_angle"] = 0.9
        ProceduralGraspController(geometry, config, lift_height=0.001, tip_lift_height=0.0005)
        for missing in ("grippers", "reference_positions", "reference_triangles", "needle_radius"):
            incomplete = deepcopy(geometry)
            del incomplete[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                ProceduralGraspController(incomplete, config, lift_height=0.001, tip_lift_height=0.0005)
            # Without a lift at the tip nothing of it is read.
            ProceduralGraspController(incomplete, config, lift_height=0.001)
        invalid = deepcopy(geometry)
        invalid["needle_radius"] = 0.0
        with self.assertRaises(ValueError):
            ProceduralGraspController(invalid, config, lift_height=0.001, tip_lift_height=0.0005)


if __name__ == "__main__":
    unittest.main()
