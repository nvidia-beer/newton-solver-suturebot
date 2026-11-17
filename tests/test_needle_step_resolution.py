# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Unresolved physical motion is rejected before any accepted-state commit."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_needle_insertion_contact_guards as guards
import warp as wp
from test_needle_insertion_example import _example, _prepare
from test_needle_tissue_coupler import _native_state

from newton_solver_suturebot import NeedleCouplingNotConverged, NeedleStepTooLarge


def _copy_state(state):
    return SimpleNamespace(**{name: wp.clone(value) for name, value in vars(state).items()})


class NeedleStepResolutionTests(unittest.TestCase):
    def test_resolved_transient_recovers_step_size_and_finishes_exactly_one_frame(self):
        with _example() as example:
            example.args.substeps = 1
            example.sim_dt = example.frame_dt
            example.select_regions([1])
            _prepare(example)
            example.start()
            original = example.coupler.step
            fine_interval = example.frame_dt / 512
            accepted = []
            initial_time = float(example.rigid_solver.mj_data.time)

            def transient(state_in, state_out, control, dt, *, external_forces=None):
                if not accepted and dt > fine_interval:
                    raise NeedleCouplingNotConverged("Initial transient requires a fine interval")
                original(state_in, state_out, control, dt, external_forces=external_forces)
                accepted.append(dt)

            with patch.object(example.coupler, "step", side_effect=transient):
                example.simulate()
            self.assertTrue(example.running, example.outcome)
            self.assertEqual(accepted[0], fine_interval)
            self.assertGreater(max(accepted), 32 * fine_interval)
            self.assertLess(len(accepted), 40)
            self.assertAlmostEqual(sum(accepted), example.frame_dt, places=12)
            self.assertAlmostEqual(example.elapsed, example.frame_dt, places=12)
            self.assertAlmostEqual(example.rigid_solver.mj_data.time - initial_time, example.frame_dt, places=12)

    def test_coarse_normal_step_can_resolve_a_short_transient_without_loosening_tolerance(self):
        with _example() as example:
            example.args.substeps = 1
            example.sim_dt = example.frame_dt
            example.select_regions([1])
            _prepare(example)
            example.start()
            original = example.coupler.step
            thresholds = (example.coupler.absolute_tolerance, example.coupler.relative_tolerance)
            accepted_interval = example.frame_dt / 512
            attempts = []
            initial_time = float(example.rigid_solver.mj_data.time)
            initial_positions = example.state_0.particle_q.numpy().copy()

            def transient(state_in, state_out, control, dt, *, external_forces=None):
                attempts.append(dt)
                if dt > accepted_interval:
                    self.assertEqual(float(example.rigid_solver.mj_data.time), initial_time)
                    np.testing.assert_array_equal(state_in.particle_q.numpy(), initial_positions)
                    raise NeedleCouplingNotConverged("Transient needs a shorter physical interval")
                original(state_in, state_out, control, dt, external_forces=external_forces)
                # End after the first accepted physical step; the test checks
                # the retry floor, while full handling runs cover frame completion.
                example.running = False

            with patch.object(example.coupler, "step", side_effect=transient):
                example.simulate()
            self.assertGreater(len(attempts), 8)
            self.assertEqual(example.accepted_physics_steps, 1)
            self.assertAlmostEqual(example.elapsed, accepted_interval, places=12)
            self.assertAlmostEqual(example.rigid_solver.mj_data.time - initial_time, accepted_interval, places=12)
            self.assertEqual((example.coupler.absolute_tolerance, example.coupler.relative_tolerance), thresholds)
            self.assertNotEqual(example.task_status, "failed")

    def test_frictional_iteration_rejection_can_retry_without_changing_force_law_or_tolerances(self):
        with _example() as example:
            _prepare(example)
            pose = example.plan.pose(0.0).copy()
            pose[2] += 0.6 * example.geometry.thickness
            example._initialize_pose(pose)
            example.start()
            interaction, coupler = example.interaction, example.coupler
            interaction.puncture_force = 1.0
            # Seed a real accepted contact load. A small declared trial budget
            # isolates rejection/retry behavior; the production budget stays40.
            example.state_0.body_f.assign(np.array([[0, 0.002, -0.0002, 0, 0, 0]], np.float32))
            coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
            self.assertGreater(np.linalg.norm(interaction.interface_force[:2]), 1e-7)
            example.state_0.assign(example.state_1)
            example.state_0.body_f.assign(np.array([[0, -0.002, -0.0002, 0, 0, 0]], np.float32))
            coupler.max_iterations = 4
            tolerances = (coupler.absolute_tolerance, coupler.relative_tolerance, coupler.max_iterations)
            native = _native_state(example.rigid_solver)
            increment = example.solver.velocity_increment.numpy().copy()
            names = ("particle_q", "particle_qd", "particle_f", "body_q", "body_qd", "joint_q", "joint_qd", "body_f")
            before = [
                {name: getattr(state, name).numpy().copy() for name in names}
                for state in (example.state_0, example.state_1)
            ]
            events, status = interaction.events, interaction.status
            with patch.object(interaction, "accept", wraps=interaction.accept) as accept:
                with self.assertRaisesRegex(NeedleCouplingNotConverged, "did not converge in 4 trials"):
                    coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
                self.assertEqual(accept.call_count, 0)
                np.testing.assert_array_equal(_native_state(example.rigid_solver), native)
                np.testing.assert_array_equal(example.solver.velocity_increment.numpy(), increment)
                for state, record in zip((example.state_0, example.state_1), before, strict=True):
                    for name, expected in record.items():
                        np.testing.assert_array_equal(getattr(state, name).numpy(), expected, err_msg=name)
                self.assertIs(interaction.events, events)
                self.assertEqual(interaction.status, status)
                dt = example.sim_dt
                time = float(example.rigid_solver.mj_data.time)
                for _ in range(8):
                    dt /= 2
                    example.solver.refresh_tangent(example.state_0.particle_q, dt)
                    try:
                        coupler.step(example.state_0, example.state_1, example.control, dt)
                    except NeedleCouplingNotConverged:
                        self.assertEqual(accept.call_count, 0)
                        np.testing.assert_array_equal(_native_state(example.rigid_solver), native)
                    else:
                        break
                else:
                    self.fail("The unchanged contact solve did not converge within eight timestep halvings")
                self.assertEqual(accept.call_count, 1)
                self.assertAlmostEqual(example.rigid_solver.mj_data.time, time + dt, places=12)
                self.assertEqual(
                    (coupler.absolute_tolerance, coupler.relative_tolerance, coupler.max_iterations), tolerances
                )
                self.assertGreater(np.linalg.norm(interaction.interface_force[:2]), 1e-7)

    def test_converged_oversized_step_rolls_back_and_smaller_retry_commits_once(self):
        with _example() as example:
            _prepare(example)
            example.start()
            interaction, coupler = example.interaction, example.coupler
            # An upward load in free space isolates resolution rejection from
            # contact iteration failure. The force solve itself converges.
            example.state_0.body_f.assign(np.array([[0, 0, 2, 0, 0, 0]], np.float32))
            example.state_0.particle_f.fill_(wp.vec3(0.123, 0.234, 0.345))
            example.solver.velocity_increment.fill_(wp.vec3(1e-7, -2e-7, 3e-7))
            names = ("particle_q", "particle_qd", "particle_f", "body_q", "body_qd", "joint_q", "joint_qd", "body_f")
            before = [
                {name: getattr(state, name).numpy().copy() for name in names}
                for state in (example.state_0, example.state_1)
            ]
            native = _native_state(example.rigid_solver)
            increment = example.solver.velocity_increment.numpy().copy()
            history = (interaction.status, interaction.reason, interaction.puncture_count, interaction.track_count)
            events = interaction.events
            with patch.object(interaction, "accept", wraps=interaction.accept) as accept:
                with self.assertRaisesRegex(NeedleStepTooLarge, "Unresolved tip or shaft displacement"):
                    coupler.step(example.state_0, example.state_1, example.control, example.sim_dt)
                self.assertEqual(accept.call_count, 0)
                self.assertLess(coupler.residual, coupler.absolute_tolerance)
                self.assertEqual(coupler.maximum_iterations, 0)
                for state, record in zip((example.state_0, example.state_1), before, strict=True):
                    for name, expected in record.items():
                        np.testing.assert_array_equal(getattr(state, name).numpy(), expected, err_msg=name)
                np.testing.assert_array_equal(_native_state(example.rigid_solver), native)
                np.testing.assert_array_equal(example.solver.velocity_increment.numpy(), increment)
                self.assertEqual(
                    (interaction.status, interaction.reason, interaction.puncture_count, interaction.track_count),
                    history,
                )
                self.assertIs(interaction.events, events)
                # Refresh the h-dependent structural matrix, not merely the
                # integration argument, before retrying the accepted state.
                dt = example.sim_dt / 8
                example.solver.refresh_tangent(example.state_0.particle_q, dt)
                time = float(example.rigid_solver.mj_data.time)
                coupler.step(example.state_0, example.state_1, example.control, dt)
                self.assertEqual(accept.call_count, 1)
                self.assertAlmostEqual(example.rigid_solver.mj_data.time, time + dt, places=12)
                self.assertGreater(example.state_1.body_q.numpy()[0, 2], before[0]["body_q"][0, 2])
                self.assertNotEqual(interaction.status, "failed")
                self.assertEqual((interaction.puncture_count, interaction.track_count), (0, 0))

    def test_pure_preflight_catches_rotation_with_stationary_tip(self):
        interaction, start, _ = guards.TestNeedleInsertionContactGuards().thin_layers(second_layer=False)
        end = _copy_state(start)
        pose = end.body_q.numpy()
        pose[0, 3:] = (0, np.sin(0.6), 0, np.cos(0.6))
        end.body_q.assign(pose)
        events, status = interaction.events, interaction.status
        with self.assertRaisesRegex(NeedleStepTooLarge, "tip or shaft"):
            interaction.check_step_resolution(start, end, 0)
        self.assertIs(interaction.events, events)
        self.assertEqual(interaction.status, status)
        np.testing.assert_array_equal(start.body_q.numpy()[0, :3], end.body_q.numpy()[0, :3])

    def test_pure_preflight_catches_thin_gap_despite_small_endpoint_displacement(self):
        fixture = guards.TestNeedleInsertionContactGuards()
        interaction, state, forces = fixture.thin_layers(second_layer=True)
        fixture.forces(interaction, state, forces)
        state.body_q.assign(np.array([[0.002, 0.002, 0.00003, 0, 0, 0, 1]], np.float32))
        interaction.accept(state, 0)
        self.assertEqual(interaction.status, "inserted")
        end = _copy_state(state)
        end.body_q.assign(np.array([[0.002, 0.002, 0.00014, 0, 0, 0, 1]], np.float32))
        events = interaction.events
        counts = (interaction.track_count, interaction.entry_count, interaction.exit_count, interaction.puncture_count)
        with self.assertRaisesRegex(NeedleStepTooLarge, "Multiple surface crossings"):
            interaction.check_step_resolution(state, end, 0)
        self.assertIs(interaction.events, events)
        self.assertEqual(interaction.status, "inserted")
        self.assertEqual(
            (interaction.track_count, interaction.entry_count, interaction.exit_count, interaction.puncture_count),
            counts,
        )

    def test_pure_preflight_catches_outside_to_outside_thin_material_crossing(self):
        interaction, start, _ = guards.TestNeedleInsertionContactGuards().thin_layers(second_layer=False)
        end = _copy_state(start)
        end.body_q.assign(np.array([[0.002, 0.002, 0.00008, 0, 0, 0, 1]], np.float32))
        events, status = interaction.events, interaction.status
        with self.assertRaisesRegex(NeedleStepTooLarge, "Multiple surface crossings"):
            interaction.check_step_resolution(start, end, 0)
        self.assertIs(interaction.events, events)
        self.assertEqual(interaction.status, status)
        self.assertEqual((interaction.puncture_count, interaction.track_count), (0, 0))

    def test_accept_tests_its_own_step_after_a_preflight_of_another_candidate(self):
        fixture = guards.TestNeedleInsertionContactGuards()
        interaction, start, forces = fixture.thin_layers(second_layer=False, puncture_force=1.0)
        fixture.forces(interaction, start, forces)
        below = _copy_state(start)
        below.body_q.assign(np.array([[0.002, 0.002, -0.00004, 0, 0, 0, 1]], np.float32))
        # This candidate stays below the layer: the preflight passes without a crossing.
        interaction.check_step_resolution(start, below, 0)
        # The accepted step starts from the same state and ends above the layer. Only its tip
        # differs from the candidate of the preflight, and its two crossings are its own.
        above = _copy_state(start)
        above.body_q.assign(np.array([[0.002, 0.002, 0.0001, 0, 0, 0, 1]], np.float32))
        interaction.accept(above, 0)
        self.assertEqual(interaction.status, "failed")
        self.assertIn("surface crossings", interaction.reason)
        self.assertFalse(interaction.punctured)
        self.assertEqual(interaction.track_count, 0)


if __name__ == "__main__":
    unittest.main()
