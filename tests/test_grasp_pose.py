# SPDX-License-Identifier: Apache-2.0
"""Reset grasps can twist or tilt without moving their needle contact point."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton_solver_suturebot._src.geometry.needle import NeedleGeometry
from newton_solver_suturebot.examples.grasp._dvrk_grasper import needle_to_gripper_pose


def _axes(pose):
    return np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=float).reshape(3, 3)


class GraspPoseTests(unittest.TestCase):
    def test_negative_quarter_turn_around_tangent_preserves_the_bite(self):
        geometry = SimpleNamespace(centerline=[[0, 0, 0], [0, 0, 0.01]], arc_normal=[1, 0, 0])
        default = needle_to_gripper_pose(geometry)
        twisted = needle_to_gripper_pose(geometry, tangent_roll_rad=-np.pi / 2)
        np.testing.assert_allclose(default, [0, -0.006, 0.008, 0, 0, 0, 1], atol=1e-8)
        np.testing.assert_allclose(twisted[:3], [-0.006, 0, 0.008], atol=1e-8)
        np.testing.assert_allclose(_axes(twisted), [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=2e-7)
        np.testing.assert_allclose(twisted[:3] + 0.006 * _axes(twisted)[:, 1], [0, 0, 0.008], atol=2e-9)

    def test_source_needle_keeps_contact_location_and_tangent_under_twist(self):
        geometry = NeedleGeometry.load()
        for arc_fraction in (0.2, 0.8):
            default = needle_to_gripper_pose(geometry, arc_fraction=arc_fraction)
            axes = _axes(default)
            bite = default[:3] + 0.006 * axes[:, 1]
            for twist in (-np.pi / 2, 0, 0.35, np.pi):
                with self.subTest(arc_fraction=arc_fraction, twist=twist):
                    pose = needle_to_gripper_pose(geometry, arc_fraction=arc_fraction, tangent_roll_rad=twist)
                    rotated = _axes(pose)
                    np.testing.assert_allclose(pose[:3] + 0.006 * rotated[:, 1], bite, atol=3e-9)
                    np.testing.assert_allclose(rotated[:, 2], axes[:, 2], atol=3e-7)
                    self.assertAlmostEqual(np.linalg.norm(pose[3:]), 1, delta=2e-7)
                    if twist == 0:
                        np.testing.assert_array_equal(pose, default)

    def test_nonfinite_twists_are_rejected(self):
        geometry = SimpleNamespace(centerline=[[0, 0, 0], [0, 0, 0.01]], arc_normal=[1, 0, 0])
        for twist in (np.nan, np.inf, -np.inf):
            with self.subTest(twist=twist), self.assertRaisesRegex(ValueError, "finite"):
                needle_to_gripper_pose(geometry, tangent_roll_rad=twist)

    def test_bite_tilt_keeps_closing_normal_and_the_needle_between_jaws(self):
        geometry = NeedleGeometry.load()
        for arc_fraction in (0.2, 0.5, 0.8):
            for twist in (-np.pi / 2, -5 * np.pi / 12, 0, 0.35, np.pi):
                untilted = needle_to_gripper_pose(geometry, arc_fraction=arc_fraction, tangent_roll_rad=twist)
                original_axes = _axes(untilted)
                bite = untilted[:3] + 0.006 * original_axes[:, 1]
                tangent = original_axes[:, 2]
                for angle in (-np.pi / 3, -np.pi / 12, 0, np.pi / 12, np.pi / 3):
                    with self.subTest(arc_fraction=arc_fraction, twist=twist, angle=angle):
                        pose = needle_to_gripper_pose(
                            geometry,
                            arc_fraction=arc_fraction,
                            tangent_roll_rad=twist,
                            bite_angle_rad=angle,
                        )
                        axes = _axes(pose)
                        # Both the arc location and the tool-local jaw bite
                        # remain fixed, including for the chosen oblique grasp.
                        np.testing.assert_allclose(axes.T @ (bite - pose[:3]), [0, 0.006, 0], atol=1e-8)
                        np.testing.assert_allclose(axes[:, 0], original_axes[:, 0], atol=1e-6)
                        # The tangent stays in the jaw plane and retains a
                        # positive component across the jaws, at the set angle.
                        local_tangent = axes.T @ tangent
                        np.testing.assert_allclose(local_tangent, [0, np.sin(angle), np.cos(angle)], atol=1e-6)
                        self.assertGreater(local_tangent[2], 0)
                        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-6)
                        np.testing.assert_allclose(np.cross(axes[:, 0], axes[:, 1]), axes[:, 2], atol=1e-6)
                        self.assertAlmostEqual(np.linalg.det(axes), 1, delta=1e-6)
                        self.assertAlmostEqual(np.linalg.norm(pose[3:]), 1, delta=2e-7)
                        if angle == 0:
                            np.testing.assert_array_equal(pose, untilted)

    def test_bite_tilt_is_about_the_closing_axis_after_tangent_twist(self):
        geometry = SimpleNamespace(centerline=[[0, 0, 0], [0, 0, 0.01]], arc_normal=[1, 0, 0])
        pose = needle_to_gripper_pose(geometry, tangent_roll_rad=-np.pi / 2, bite_angle_rad=np.pi / 6)
        root_three = np.sqrt(3) / 2
        np.testing.assert_allclose(_axes(pose), [[0, root_three, -0.5], [-1, 0, 0], [0, 0.5, root_three]], atol=3e-7)
        np.testing.assert_allclose(pose[:3], [-0.006 * root_three, 0, 0.005], atol=2e-9)

    def test_exchanged_jaws_turn_the_tool_half_a_turn_about_its_jaw_axis(self):
        geometry = SimpleNamespace(centerline=[[0, 0, 0], [0, 0, 0.01]], arc_normal=[1, 0, 0])
        # Straight wire along Z, closing axis X: the tool stands 6 mm back along -Y of the bite at
        # (0, 0, 8 mm). Exchanged, it stands at the same place with its X and Z axes reversed.
        default = needle_to_gripper_pose(geometry)
        np.testing.assert_array_equal(needle_to_gripper_pose(geometry, jaws_exchanged=False), default)
        exchanged = needle_to_gripper_pose(geometry, jaws_exchanged=True)
        np.testing.assert_allclose(exchanged[:3], [0, -0.006, 0.008], atol=1e-12)
        np.testing.assert_allclose(_axes(exchanged), [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], atol=2e-7)
        # With the twist and the bite of the grasp that pulls (-60 and 30 degrees) on the source needle:
        # the same bite point and the same jaw axis, the closing axis and the third axis reversed, which
        # is the tool turned by pi about its own Y axis through its own origin.
        source = NeedleGeometry.load()
        for arc_fraction in (0.1, 0.5):
            options = dict(arc_fraction=arc_fraction, tangent_roll_rad=-np.pi / 3, bite_angle_rad=np.pi / 6)
            plain = needle_to_gripper_pose(source, **options)
            np.testing.assert_array_equal(needle_to_gripper_pose(source, jaws_exchanged=False, **options), plain)
            turned = needle_to_gripper_pose(source, jaws_exchanged=True, **options)
            axes, original = _axes(turned), _axes(plain)
            np.testing.assert_allclose(turned[:3], plain[:3], atol=1e-12)
            np.testing.assert_allclose(turned[:3] + 0.006 * axes[:, 1], plain[:3] + 0.006 * original[:, 1], atol=3e-9)
            np.testing.assert_allclose(axes[:, 1], original[:, 1], atol=3e-7)
            np.testing.assert_allclose(axes[:, [0, 2]], -original[:, [0, 2]], atol=3e-7)
            half_turn = wp.transform((0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi))
            composed = np.asarray(wp.transform_multiply(wp.transform(*plain), half_turn), dtype=float)
            np.testing.assert_allclose(turned[:3], composed[:3], atol=1e-9)
            self.assertAlmostEqual(abs(turned[3:] @ composed[3:]), 1.0, delta=1e-6)
            self.assertAlmostEqual(np.linalg.det(axes), 1, delta=1e-6)

    def test_invalid_bite_angles_are_rejected(self):
        geometry = SimpleNamespace(centerline=[[0, 0, 0], [0, 0, 0.01]], arc_normal=[1, 0, 0])
        for angle in (np.nan, np.inf, -np.inf, -np.pi / 2, np.pi / 2, -np.pi, np.pi):
            with self.subTest(angle=angle), self.assertRaisesRegex(ValueError, "strictly between -pi/2 and pi/2"):
                needle_to_gripper_pose(geometry, bite_angle_rad=angle)


if __name__ == "__main__":
    unittest.main()
