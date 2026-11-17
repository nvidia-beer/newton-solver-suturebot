"""CPU command contracts for the physical-grasp example, without an RL stack.

These tests exercise control and reset behavior; physical insertion success
still requires a complete coupled-solver rollout.
"""

import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton_solver_suturebot.examples.grasp._grasp_procedure import (
    ProceduralGraspController,
    action_to,
    gripper_target,
    pose_at,
)


def fixture():
    radius = 0.00945
    plan = {
        "center": np.array([0.03, -0.04, 0.01]),
        "radial": np.array([1.0, 0.0, 0.0]),
        "tangent": np.array([0.0, 1.0, 0.0]),
        "axis": np.array([0.0, 0.0, 1.0]),
        "local_basis": np.eye(3),
        "arc_center_local": np.array([-radius, 0.0, 0.0]),
        "radius": radius,
        "initial_angle": -0.2,
        "finish_angle": 1.05,
    }
    config = SimpleNamespace(
        control_mode="physical_grasp", frame_dt=1 / 60, translation_step=0.0001, rotation_step=0.01, jaw_step_rad=0.02
    )
    return {"control_mode": "physical_grasp", "insertion_plan": plan}, config


def snapshot(geometry, *, env_id=0, time=0, reset=0, contact=True):
    relative = np.r_[[0.001, 0.006, -0.002], np.asarray(wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.7))]
    pose = pose_at(geometry["insertion_plan"], geometry["insertion_plan"]["initial_angle"])
    return {
        "env_id": env_id,
        "reset_count": reset,
        "diagnostics": {"physical_time_s": time, "entry_count": 0, "exit_count": 0},
        "needle_pose": pose.copy(),
        "needle_relative_pose": relative,
        "gripper_target_pose": gripper_target(pose, relative),
        "jaw_contact_count": np.full(2, int(contact)),
        "jaw_contact_normal_force_n": np.full(2, 0.2 if contact else 0),
        "jaw_target_rad": np.zeros(2),
    }


class ProceduralGraspTests(unittest.TestCase):
    def test_grasp_transform_solves_full_rigid_pose_with_nontrivial_rotation(self):
        needle = np.r_[[0.03, -0.05, 0.07], np.asarray(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 1.2))]
        grasp = np.r_[[0.004, 0.007, -0.002], np.asarray(wp.quat_from_axis_angle(wp.vec3(1, 0, 0), -0.8))]
        palm = gripper_target(needle, grasp)
        reconstructed = np.asarray(wp.transform_multiply(wp.transform(*palm), wp.transform(*grasp)))
        np.testing.assert_allclose(reconstructed, needle, atol=1e-7)

    def test_settles_closes_jaws_then_bounds_actions_without_mutating_observations(self):
        geometry, config = fixture()
        controller = ProceduralGraspController(geometry, config)
        item = snapshot(geometry)
        item["jaw_target_rad"] = np.array([0.3, -0.3])
        original = deepcopy(item)
        np.testing.assert_array_equal(controller([item]), [[0, 0, 0, 0, 0, 0, -1]])
        for name, value in original.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(item[name], value)
            else:
                self.assertEqual(item[name], value)
        item["diagnostics"]["physical_time_s"] = 0.5
        controller([item])
        item["diagnostics"]["physical_time_s"] = 0.75
        actions = controller([item])
        self.assertEqual(actions.shape, (1, 7))
        self.assertEqual(actions.dtype, np.float32)
        self.assertTrue(np.isfinite(actions).all())
        self.assertLessEqual(np.max(np.abs(actions)), 1)
        self.assertGreater(np.linalg.norm(actions[0, :6]), 0)
        self.assertGreater(controller.report()[0]["commanded_angle_rad"], geometry["insertion_plan"]["initial_angle"])

    def test_grasp_loss_stops_until_only_affected_world_resets(self):
        geometry, config = fixture()
        controller = ProceduralGraspController(geometry, config, settle_seconds=0)
        items = [snapshot(geometry, env_id=index) for index in range(2)]
        controller(items)
        for item in items:
            item["diagnostics"]["physical_time_s"] = 0.1
        controller(items)
        before = controller.report()[0]["commanded_angle_rad"]
        items[0]["jaw_contact_count"][:] = 0
        for item in items:
            item["diagnostics"]["physical_time_s"] = 0.2
        np.testing.assert_array_equal(controller(items)[0], 0)
        self.assertEqual(controller.report()[0]["commanded_angle_rad"], before)
        for item in items:
            item["diagnostics"]["physical_time_s"] = 0.45
        controller(items)
        self.assertEqual(controller.report()[0]["phase"], "stopped")
        self.assertEqual(controller.report()[1]["phase"], "inserting")
        items[0]["jaw_contact_count"][:] = 1
        np.testing.assert_array_equal(controller(items)[0], 0)
        neighbor_angle = controller.report()[1]["commanded_angle_rad"]
        items[0] = snapshot(geometry, env_id=0, reset=1)
        controller(items)
        self.assertEqual(controller.report()[0]["phase"], "inserting")
        self.assertEqual(controller.report()[0]["commanded_angle_rad"], geometry["insertion_plan"]["initial_angle"])
        self.assertEqual(controller.report()[1]["commanded_angle_rad"], neighbor_angle)

    def test_brief_contact_loss_resumes_without_jumping_over_lost_time(self):
        geometry, config = fixture()
        controller = ProceduralGraspController(geometry, config, settle_seconds=0)
        item = snapshot(geometry)
        controller([item])
        item["diagnostics"]["physical_time_s"] = 0.1
        controller([item])
        before = controller.report()[0]["commanded_angle_rad"]
        item["jaw_contact_count"][:] = 0
        item["diagnostics"]["physical_time_s"] = 0.15
        controller([item])
        item["jaw_contact_count"][:] = 1
        item["diagnostics"]["physical_time_s"] = 0.3
        controller([item])
        self.assertEqual(controller.report()[0]["commanded_angle_rad"], before)
        item["diagnostics"]["physical_time_s"] = 0.4
        controller([item])
        self.assertGreater(controller.report()[0]["commanded_angle_rad"], before)

    def test_initial_grasp_timeout_and_solver_failure_stop_commands(self):
        geometry, config = fixture()
        controller = ProceduralGraspController(geometry, config)
        item = snapshot(geometry, contact=False)
        controller([item])
        item["diagnostics"]["physical_time_s"] = 2.5
        np.testing.assert_array_equal(controller([item]), 0)
        self.assertIn("not established", controller.report()[0]["reason"])
        item = snapshot(geometry, reset=1)
        item["diagnostics"]["failure_reason"] = "FEM failure"
        np.testing.assert_array_equal(controller([item]), 0)
        self.assertEqual(controller.report()[0]["reason"], "FEM failure")

    def test_slip_is_diagnostic_and_preserves_fixed_grasp_transform(self):
        geometry, config = fixture()
        controller = ProceduralGraspController(geometry, config, settle_seconds=0)
        item = snapshot(geometry)
        controller([item])
        item["diagnostics"]["physical_time_s"] = 0.2
        expected = controller([item])
        item["needle_relative_pose"][:3] += [0, 0, 0.004]
        item["needle_pose"][:3] += [0, 0, 0.004]
        np.testing.assert_array_equal(controller([item]), expected)
        self.assertAlmostEqual(controller.report()[0]["relative_translation_m"], 0.004)

    def test_commanded_arc_completion_is_separate_from_physical_events(self):
        geometry, config = fixture()
        controller = ProceduralGraspController(geometry, config, settle_seconds=0)
        item = snapshot(geometry)
        controller([item])
        item["diagnostics"]["physical_time_s"] = 30
        controller([item])
        report = controller.report()[0]
        self.assertEqual(report["phase"], "holding")
        self.assertTrue(report["commanded_arc_complete"])
        self.assertEqual(report["entry_count"], 0)
        self.assertEqual(report["exit_count"], 0)
        self.assertNotIn("success", report)
        item["diagnostics"]["entry_count"], item["diagnostics"]["exit_count"] = 1, 1
        controller([item])
        self.assertEqual(controller.report()[0]["entry_count"], 1)
        self.assertEqual(controller.report()[0]["exit_count"], 1)

    def test_rejects_invalid_mode_clock_and_duplicate_worlds(self):
        geometry, config = fixture()
        config.control_mode = "free_needle"
        with self.assertRaisesRegex(ValueError, "physical_grasp"):
            ProceduralGraspController(geometry, config)
        config.control_mode = "physical_grasp"
        controller = ProceduralGraspController(geometry, config)
        item = snapshot(geometry, time=1)
        controller([item])
        item["diagnostics"]["physical_time_s"] = 0
        with self.assertRaisesRegex(ValueError, "backward"):
            controller([item])
        with self.assertRaisesRegex(ValueError, "distinct env_id"):
            ProceduralGraspController(geometry, config)([item, deepcopy(item)])

    def test_rejects_nonfinite_commands_and_invalid_plan_without_mutation(self):
        geometry, config = fixture()
        for name in ("frame_dt", "translation_step", "rotation_step", "jaw_step_rad"):
            invalid = deepcopy(config)
            setattr(invalid, name, float("nan"))
            with self.subTest(scale=name), self.assertRaises(ValueError):
                ProceduralGraspController(geometry, invalid)
        invalid = deepcopy(geometry)
        invalid["insertion_plan"]["radius"] = 0
        with self.assertRaisesRegex(ValueError, "positive radius"):
            ProceduralGraspController(invalid, config)
        for field in ("needle_relative_pose", "gripper_target_pose", "jaw_contact_normal_force_n"):
            item = snapshot(geometry)
            item[field][0] = float("nan")
            with self.subTest(field=field), self.assertRaises(ValueError):
                ProceduralGraspController(geometry, config)([item])

    def test_world_rotation_increment_is_shortest_arc_and_rate_limited(self):
        _, config = fixture()
        old = np.r_[[0.1, 0.2, 0.3], np.asarray(wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.8))]
        turn = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.004)
        target = old.copy()
        target[3:] = np.asarray(turn * wp.quat(*old[3:]))
        target[:3] += [0.00004, -0.00007, 0.0]
        expected = [0.4, -0.7, 0, 0, 0, 0.4]
        np.testing.assert_allclose(action_to(old, target, config), expected, atol=2e-5)
        target[3:] *= -1
        np.testing.assert_allclose(action_to(old, target, config), expected, atol=2e-5)
        target[0] += 1
        self.assertEqual(action_to(old, target, config)[0], 1)
        with self.assertRaisesRegex(ValueError, "rate limits"):
            action_to(old, target, config, strict=True)

    def test_controller_copies_plan_and_keeps_input_order_independent_of_ids(self):
        geometry, config = fixture()
        controller = ProceduralGraspController(geometry, config, settle_seconds=0)
        items = [snapshot(geometry, env_id=7), snapshot(geometry, env_id=2)]
        controller(items)
        for item in items:
            item["diagnostics"]["physical_time_s"] = 0.2
        expected = controller(items)
        geometry["insertion_plan"]["center"][:] = 100
        np.testing.assert_array_equal(controller(items[::-1]), expected[::-1])
        self.assertEqual([row["env_id"] for row in controller.report()], [2, 7])
        self.assertEqual(controller([]).shape, (0, 7))


if __name__ == "__main__":
    unittest.main()
