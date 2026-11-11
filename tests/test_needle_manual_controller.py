# SPDX-License-Identifier: Apache-2.0
"""Direct slider targets move the physical needle through MuJoCo/FEM forces."""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import warp as wp
from test_needle_insertion_example import _example, _prepare


@contextmanager
def _prepared():
    with _example() as example:
        example.args.substeps = 12
        example.sim_dt = example.frame_dt / example.args.substeps
        example.select_regions([1])  # Rebuild the FEM operator at its construction timestep.
        _prepare(example)
        yield example


def _physical_state(example):
    return [
        getattr(state, name).numpy().copy()
        for state in (example.state_0, example.state_1)
        for name in ("body_q", "body_qd", "particle_q", "particle_qd")
    ]


class NeedleManualControllerTests(unittest.TestCase):
    def assertPhysicalEqual(self, example, expected):
        for actual, before in zip(_physical_state(example), expected, strict=True):
            np.testing.assert_array_equal(actual, before)

    def advance(self, example, count=1):
        before = example.elapsed
        for _ in range(count):
            example.step()
            self.assertTrue(example.running, example.outcome)
            self.assertNotEqual(example.task_status, "failed", example.outcome)
        self.assertAlmostEqual(example.elapsed - before, count * example.frame_dt, places=12)

    def test_takeover_seeds_goal_without_changing_actual_state_time_or_desired_pose(self):
        with _prepared() as example:
            example.start()
            self.advance(example)
            physical = _physical_state(example)
            desired = example.desired_pose.copy()
            times = (example.elapsed, example.sim_time, example.rigid_solver.mj_data.time)
            example.use_manual_control()
            self.assertTrue(example.manual_control)
            self.assertPhysicalEqual(example, physical)
            np.testing.assert_array_equal(example.desired_pose, desired)
            np.testing.assert_array_equal(example.manual_target, example.manual_position)
            angle = example.plan.initial_angle + example.manual_position[0] / example.geometry.radius
            np.testing.assert_allclose(example.plan.pose(angle), desired, atol=1e-12)
            self.assertEqual((example.elapsed, example.sim_time, example.rigid_solver.mj_data.time), times)

    def test_pause_freezes_physics_and_resume_retains_the_direct_goal(self):
        with _prepared() as example:
            example.set_manual_target(insertion=0.0001, lift=0.0001)
            self.advance(example)
            example.toggle_running()
            physical = _physical_state(example)
            desired = example.desired_pose.copy()
            position, goal = example.manual_position.copy(), example.manual_target.copy()
            elapsed = example.elapsed
            with patch.object(example, "_manual_target_pose") as target:
                for _ in range(5):
                    example.step()
                target.assert_not_called()
            self.assertPhysicalEqual(example, physical)
            np.testing.assert_array_equal(example.desired_pose, desired)
            np.testing.assert_array_equal(example.manual_position, position)
            np.testing.assert_array_equal(example.manual_target, goal)
            self.assertEqual(example.elapsed, elapsed)
            example.toggle_running()
            self.advance(example)
            np.testing.assert_array_equal(example.manual_position, goal)
            np.testing.assert_array_equal(example.desired_pose, desired)
            self.assertGreater(np.linalg.norm(_physical_state(example)[0] - physical[0]), 0)

    def test_first_accepted_step_uses_full_slider_goal_with_zero_target_velocity(self):
        with _prepared() as example:
            goal = np.array((2e-5, -4e-5))
            physical = _physical_state(example)
            desired = example.desired_pose.copy()
            example.set_manual_target(insertion=goal[0], lift=goal[1])
            self.assertPhysicalEqual(example, physical)
            np.testing.assert_array_equal(example.desired_pose, desired)
            np.testing.assert_array_equal(example.manual_position, (0, 0))
            target = example.plan.pose(example.plan.initial_angle + goal[0] / example.geometry.radius)
            target[2] += goal[1]
            drive_calls, accepted = [], []
            original_drive, original_step = example.drive.add_forces, example.coupler.step

            def add_forces(state, pose, velocity):
                drive_calls.append((pose.copy(), np.asarray(velocity).copy()))
                return original_drive(state, pose, velocity)

            def physical_step(state_in, state_out, *args, **kwargs):
                result = original_step(state_in, state_out, *args, **kwargs)
                accepted.append((example.manual_position.copy(), state_out.body_q.numpy().copy()))
                return result

            with (
                patch.object(example.drive, "add_forces", side_effect=add_forces),
                patch.object(example.coupler, "step", side_effect=physical_step),
            ):
                self.advance(example)
            self.assertEqual(len(accepted), example.args.substeps)
            for pose, velocity in drive_calls:
                np.testing.assert_array_equal(pose, target)
                np.testing.assert_array_equal(velocity, np.zeros(6))
            np.testing.assert_array_equal(accepted[0][0], (0, 0))
            np.testing.assert_array_equal(accepted[1][0], goal)
            np.testing.assert_array_equal(example.manual_position, goal)
            np.testing.assert_array_equal(example.desired_pose, target)
            # The accepted rigid state moves under finite force; it is neither
            # left at its old pose nor copied from the requested target.
            self.assertGreater(np.linalg.norm(accepted[0][1] - physical[0]), 0)
            self.assertGreater(np.linalg.norm(accepted[0][1][0, :3] - target[:3]), 1e-8)
            body = example.state_0.body_q.numpy().copy()
            self.advance(example, 2)
            np.testing.assert_array_equal(example.desired_pose, target)
            np.testing.assert_array_equal(example.manual_position, goal)
            self.assertGreater(np.linalg.norm(example.state_0.body_q.numpy() - body), 0)

    def test_lift_and_lower_change_world_z_without_rotating_the_target(self):
        with _prepared() as example:
            initial = example.desired_pose.copy()
            physical = _physical_state(example)
            example.set_manual_target(lift=0.0002)
            self.assertPhysicalEqual(example, physical)
            np.testing.assert_array_equal(example.desired_pose, initial)
            self.advance(example, 2)
            raised = example.desired_pose.copy()
            self.assertGreater(raised[2], initial[2])
            self.assertAlmostEqual(raised[2] - initial[2], 0.0002, places=12)
            np.testing.assert_array_equal(raised[:2], initial[:2])
            np.testing.assert_array_equal(raised[3:], initial[3:])
            example.set_manual_target(lift=0)
            self.advance(example, 2)
            self.assertLess(example.desired_pose[2], raised[2])
            np.testing.assert_allclose(example.desired_pose, initial, atol=1e-12)
            self.assertAlmostEqual(example.manual_position[1], 0, places=12)

    def test_insert_and_retract_follow_arc_without_teleporting_body(self):
        with _prepared() as example:
            initial = example.desired_pose.copy()
            physical = _physical_state(example)
            example.set_manual_target(insertion=0.0002)
            self.assertPhysicalEqual(example, physical)
            np.testing.assert_array_equal(example.desired_pose, initial)
            self.advance(example, 2)
            advanced = example.manual_position[0]
            self.assertGreater(advanced, 0)
            self.assertEqual(advanced, 0.0002)
            angle = example.plan.initial_angle + advanced / example.geometry.radius
            np.testing.assert_allclose(example.desired_pose, example.plan.pose(angle), atol=1e-12)
            self.assertGreater(np.linalg.norm(example.desired_pose[3:] - initial[3:]), 0)
            physical = _physical_state(example)
            example.set_manual_target(insertion=0)
            self.assertPhysicalEqual(example, physical)
            self.advance(example, 2)
            self.assertLess(example.manual_position[0], advanced)
            self.assertAlmostEqual(example.manual_position[0], 0, places=12)
            np.testing.assert_allclose(example.desired_pose, initial, atol=1e-12)

    def test_abrupt_goals_resolve_actual_motion_without_slowing_the_direct_target(self):
        for goal in ({"insertion": 0.002}, {"lift": 0.001}):
            with self.subTest(goal=goal):
                self._check_abrupt_goal(goal)

    def _check_abrupt_goal(self, goal):
        with _prepared() as example:
            physical = _physical_state(example)
            desired_before = example.desired_pose.copy()
            tip_before = example.actual_tip().copy()
            example.set_manual_target(**goal)
            self.assertPhysicalEqual(example, physical)
            np.testing.assert_array_equal(example.desired_pose, desired_before)
            target = example.plan.pose(example.plan.initial_angle + example.manual_target[0] / example.geometry.radius)
            target[2] += example.manual_target[1]
            fixed = example.fixed_vertex_indices
            fixed_reference = example.selection.vertices[fixed].astype(np.float32)
            bound = 1.01 * min(float(example.interaction.radii.min()), example.interaction.track_spacing)
            accepted_dt, motions, drive_targets = [], [], []
            original_step, original_drive = example.coupler.step, example.drive.add_forces

            def world_samples(state):
                pose = state.body_q.numpy()[example.needle_body]
                rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
                return example.interaction.samples @ rotation.T + pose[:3]

            def physical_step(state_in, state_out, control, dt, **kwargs):
                before = world_samples(state_in)
                result = original_step(state_in, state_out, control, dt, **kwargs)
                accepted_dt.append(dt)
                motions.append(float(np.linalg.norm(world_samples(state_out) - before, axis=1).max()))
                for field in ("body_q", "body_qd", "particle_q", "particle_qd"):
                    self.assertTrue(np.isfinite(getattr(state_out, field).numpy()).all())
                np.testing.assert_array_equal(state_out.particle_q.numpy()[fixed], fixed_reference)
                self.assertNotEqual(example.interaction.status, "failed", example.interaction.reason)
                return result

            def add_forces(state, pose, velocity):
                drive_targets.append(pose.copy())
                return original_drive(state, pose, velocity)

            with (
                patch.object(example.coupler, "step", side_effect=physical_step),
                patch.object(example.drive, "add_forces", side_effect=add_forces),
            ):
                self.advance(example)
            self.assertGreater(len(accepted_dt), 0)
            self.assertAlmostEqual(sum(accepted_dt), example.frame_dt, places=12)
            self.assertLessEqual(max(motions), bound)
            for pose in drive_targets:
                np.testing.assert_array_equal(pose, target)
            np.testing.assert_array_equal(example.desired_pose, target)
            np.testing.assert_array_equal(example.manual_position, example.manual_target)
            self.assertGreater(np.linalg.norm(example.actual_tip() - tip_before), 5e-5)

    def test_reset_clears_manual_goals_history_and_failure_and_restores_automatic_mode(self):
        with _prepared() as example:
            example.set_manual_target(insertion=0.0001, lift=0.0001)
            self.advance(example)
            example._fail("test failure before reset")
            example.reset()
            self.assertFalse(example.manual_control)
            self.assertFalse(example.running)
            self.assertIsNone(example.plan)
            self.assertIsNone(example.desired_pose)
            self.assertIsNone(example.failure_reason)
            self.assertEqual((example.elapsed, example.sim_time), (0, 0))
            for field in ("manual_position", "manual_target"):
                np.testing.assert_array_equal(getattr(example, field), (0, 0))
            _prepare(example)
            example.start()
            self.assertFalse(example.manual_control)
            self.assertEqual(example.report()["control_mode"], "automatic")

    def test_rejected_step_does_not_commit_direct_target_or_advance_clock(self):
        with _prepared() as example:
            example.set_manual_target(insertion=0.0001, lift=0.0001)
            self.advance(example)
            physical = _physical_state(example)
            example.set_manual_target(insertion=0.00015, lift=0.00015)
            fields = ("desired_pose", "manual_position", "manual_target")
            snapshots = {field: getattr(example, field).copy() for field in fields}
            times = (example.elapsed, example.sim_time, example.rigid_solver.mj_data.time)
            with patch.object(example.coupler, "step", side_effect=RuntimeError("test rejected trial")):
                example.step()
            self.assertFalse(example.running)
            self.assertEqual(example.task_status, "failed")
            self.assertEqual(example.failure_reason, "test rejected trial")
            self.assertPhysicalEqual(example, physical)
            for field, expected in snapshots.items():
                np.testing.assert_array_equal(getattr(example, field), expected)
            self.assertEqual((example.elapsed, example.sim_time, example.rigid_solver.mj_data.time), times)
            example._action(example.set_manual_target, lift=0)
            self.assertIn("test rejected trial", example.outcome)
            self.assertEqual(example.report()["failure_reason"], "test rejected trial")
            np.testing.assert_array_equal(example.manual_target, snapshots["manual_target"])

    def test_manual_exit_is_observed_without_finishing_and_deadline_is_ignored(self):
        with _prepared() as example:
            example.set_manual_target(insertion=0, lift=0)
            target = example.exit.position(example.state_0.particle_q)
            example.interaction.events = ({"kind": "exit", "position": target},)
            with patch.object(example, "actual_tip", return_value=target):
                example._observe_interaction()
            self.assertTrue(example.running)
            self.assertEqual(example.last_exit_error, 0)
            self.assertFalse(example.report()["success"])
            self.assertIsNone(example.exit_error)
            self.assertEqual(example.interaction_events[-1]["kind"], "exit")
            # Seed only the controller clock to exercise the automatic deadline
            # policy; the following call still performs twelve real physical steps.
            deadline = (example.plan.finish_angle - example.plan.initial_angle) * example.geometry.radius
            example.elapsed = example.sim_time = deadline / example.args.insertion_speed + 2.0
            self.advance(example)
            self.assertEqual(example.task_status, "running")
            self.assertFalse(example.report()["success"])

    def test_invalid_goals_do_not_start_or_mutate_controller_or_physics(self):
        with _prepared() as example:
            maximum = (example.plan.finish_angle - example.plan.initial_angle) * example.geometry.radius
            commands = (
                {"insertion": np.nan},
                {"insertion": -1e-6},
                {"insertion": maximum + 1e-6},
                {"lift": np.inf},
                {"lift": 0.005001},
                {"lift": -0.005001},
                {"insertion": 1e-4, "lift": np.nan},
            )
            for command in commands:
                with self.subTest(command=command):
                    physical = _physical_state(example)
                    desired = example.desired_pose.copy()
                    with self.assertRaises(ValueError):
                        example.set_manual_target(**command)
                    self.assertFalse(example.manual_control)
                    self.assertFalse(example.running)
                    self.assertEqual(example.elapsed, 0)
                    self.assertIsNone(example.interaction)
                    np.testing.assert_array_equal(example.manual_target, (0, 0))
                    np.testing.assert_array_equal(example.desired_pose, desired)
                    self.assertPhysicalEqual(example, physical)

    def test_goals_require_preview_and_preview_cannot_restart_active_manual_task(self):
        with _example() as example:
            physical = _physical_state(example)
            with self.assertRaisesRegex(ValueError, "Preview"):
                example.set_manual_target(lift=0.0001)
            self.assertPhysicalEqual(example, physical)
            self.assertFalse(example.manual_control)
            self.assertIsNone(example.plan)
        with _prepared() as example:
            example.start()
            physical = _physical_state(example)
            desired = example.desired_pose.copy()
            with self.assertRaisesRegex(ValueError, "Reset"):
                example.preview()
            example.use_manual_control()
            for paused in (False, True):
                if paused:
                    example.toggle_running()
                with self.assertRaisesRegex(ValueError, "Reset"):
                    example.preview()
                self.assertEqual(example.elapsed, 0)
                self.assertEqual(example.running, not paused)
                self.assertTrue(example.manual_control)
                self.assertPhysicalEqual(example, physical)
                np.testing.assert_array_equal(example.desired_pose, desired)


if __name__ == "__main__":
    unittest.main()
