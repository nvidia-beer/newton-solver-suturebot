# SPDX-License-Identifier: Apache-2.0
"""Grasp/FEM transactions and physical-thread lifecycle on a small real mesh."""

import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mujoco
import numpy as np
import warp as wp
from test_needle_insertion_example import _REFERENCE, _selection

from newton_solver_suturebot import NeedleStepTooLarge
from newton_solver_suturebot.examples import example_fem_phantom_regions as phantom
from newton_solver_suturebot.parallel import (
    SutureSceneConfig,
    _apply_action,
    _geometry,
    _make_scene,
    _observe,
    _reset_scene,
)


@contextmanager
def _scene():
    with (
        TemporaryDirectory() as directory,
        patch.object(phantom, "list_mesh_regions", return_value=_REFERENCE.regions),
        patch.object(phantom, "load_mesh_regions", side_effect=_selection),
    ):
        config = SutureSceneConfig(
            device=os.environ.get("NEWTON_TEST_DEVICE", "cpu"),
            asset=str(Path(directory) / "pad.usd"),
            entry_xy=(0, 0.005),
            exit_xy=(0, -0.005),
            control_mode="physical_grasp",
        )
        scene = _make_scene(config, 0)
        try:
            yield scene, config
        finally:
            scene.viewer.close()


def _native(scene):
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    values = np.empty(mujoco.mj_stateSize(scene.rigid_solver.mj_model, specification))
    mujoco.mj_getState(scene.rigid_solver.mj_model, scene.rigid_solver.mj_data, values, specification)
    return values


class GraspSceneTests(unittest.TestCase):
    def test_observation_reads_each_rigid_array_once_and_preserves_diagnostics(self):
        with _scene() as (scene, _):
            scene.step()
            expected = scene.observe_grasp()
            arrays = {id(getattr(scene.state_0, name)): name for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
            reads = dict.fromkeys(arrays.values(), 0)
            original = wp.array.numpy

            def read(array):
                name = arrays.get(id(array))
                if name is not None:
                    reads[name] += 1
                return original(array)

            with patch.object(wp.array, "numpy", new=read):
                observed = _observe(scene, 0)
            self.assertEqual(reads, dict.fromkeys(reads, 1))
            for name, expected_value in expected.items():
                if isinstance(expected_value, np.ndarray):
                    np.testing.assert_array_equal(observed[name], expected_value)
                else:
                    self.assertEqual(observed[name], expected_value)
            np.testing.assert_array_equal(observed["needle_pose"], observed["rigid_body_poses"][scene.needle_body])
            np.testing.assert_array_equal(observed["tip_position"], scene.actual_tip())
            # Readbacks are scoped to this observation, never retained across
            # physical advancement or a swap of accepted device-state buffers.
            scene.step()
            current = _observe(scene, 0)
            np.testing.assert_array_equal(current["rigid_body_poses"], scene.state_0.body_q.numpy())
            np.testing.assert_array_equal(
                current["arm_joint_positions"],
                [scene.state_0.joint_q.numpy()[list(tool.arm_q_indices)] for tool in scene.graspers],
            )

    def test_incremental_actions_do_not_reposition_independent_needle_or_thread(self):
        with _scene() as (scene, config):
            initial = _observe(scene, 0)
            rigid = scene.state_0.body_q.numpy().copy()
            thread = scene.worker_thread.positions().copy()
            geometry = _geometry(scene, config)
            self.assertEqual(config.action_dimension, 7)
            self.assertEqual(scene.model.body_count, 17)
            self.assertEqual(scene.rigid_solver.mj_model.neq, 0)
            self.assertNotEqual(geometry["needle_body_index"], geometry["gripper_body_index"])
            self.assertEqual(scene.coupler.rigid_transfer_mode, "general")
            self.assertEqual(len(geometry["grippers"]), 2)
            report = initial["diagnostics"]
            self.assertEqual(report["control_mode"], "physical_grasp")
            self.assertTrue(report["independent_free_needle"])
            self.assertFalse(report["needle_pose_drive"])
            self.assertIn("fixed trocar", report["shaft_constraint"])
            self.assertIn("joint position actuators", report["pose_control"])
            self.assertNotIn("drive_stiffness_N_per_m", report)
            self.assertEqual(initial["arm_joint_positions"].shape, (2, 6))
            for item in initial["rcm_diagnostics"]:
                self.assertLess(item["shaft_to_trocar_distance_m"], 1e-6)
                self.assertTrue(item["trocar_on_shaft"])
            self.assertEqual(report["native_rigid_substeps"], 4)
            self.assertEqual(report["native_contact_cone"], "elliptic")
            self.assertEqual(scene.rigid_solver.mj_model.opt.cone, mujoco.mjtCone.mjCONE_ELLIPTIC)
            self.assertEqual(report["native_solver_iterations"], 200)
            self.assertEqual(report["native_solver_line_search_iterations"], 50)
            self.assertAlmostEqual(report["native_solver_tolerance"], 1e-12, delta=1e-18)
            self.assertAlmostEqual(report["native_rigid_dt_s"], config.frame_dt / (config.substeps * 4))
            _apply_action(scene, config, [0.1, 0, 0, 0, 0, 0, 1])
            self.assertAlmostEqual(scene.jaw_opening_target, config.jaw_step_rad)
            target = scene.cartesian_target.copy()
            _apply_action(scene, config, np.zeros(7))
            self.assertAlmostEqual(scene.jaw_opening_target, config.jaw_step_rad)
            np.testing.assert_array_equal(scene.cartesian_target, target)
            np.testing.assert_array_equal(scene.state_0.body_q.numpy(), rigid)
            np.testing.assert_array_equal(scene.worker_thread.positions(), thread)
            self.assertEqual(scene.worker_thread.accepted_steps, 0)
            with self.assertRaises(ValueError):
                _apply_action(scene, config, np.zeros(6))
            scene.step()
            self.assertEqual(scene.worker_thread.accepted_steps, scene.accepted_physics_steps)
            self.assertAlmostEqual(scene.worker_thread.physical_time_s, scene.elapsed)
            self.assertGreater(scene.elapsed, 0)
            self.assertTrue(np.isfinite(scene.worker_thread.positions()).all())
            self.assertTrue(np.isfinite(scene.state_0.body_q.numpy()).all())
            scene._check_tissue_integrity()
            current = _observe(scene, 0)
            np.testing.assert_array_equal(current["trocar_poses"], initial["trocar_poses"])
            for item in current["rcm_diagnostics"]:
                self.assertLess(item["shaft_to_trocar_distance_m"], 1e-6)
                self.assertTrue(item["trocar_on_shaft"])
            _reset_scene(scene, config, rebuild=True)
            reset = _observe(scene, 0)
            self.assertEqual(reset["reset_count"], initial["reset_count"] + 1)
            self.assertEqual(reset["thread_diagnostics"]["accepted_steps"], 0)
            self.assertEqual(reset["thread_diagnostics"]["physical_time_s"], 0)
            np.testing.assert_array_equal(reset["rigid_body_poses"], rigid)
            np.testing.assert_array_equal(reset["thread_positions"], thread)

    def test_rejected_multi_body_trial_restores_native_and_output_without_advancing_thread(
        self,
    ):
        with _scene() as (scene, _):
            scene.state_0.clear_forces()
            scene._prepare_step_loads(scene.cartesian_target, np.zeros(6), scene.sim_dt)
            scene.solver.refresh_tangent(scene.state_0.particle_q, scene.sim_dt)
            native = _native(scene)
            saved = {
                name: getattr(scene.state_1, name).numpy().copy()
                for name in (
                    "joint_q",
                    "joint_qd",
                    "body_q",
                    "body_qd",
                    "particle_q",
                    "particle_qd",
                )
            }
            forces = scene.state_0.body_f.numpy().copy()
            thread = [array.numpy().copy() for array in scene.worker_thread.state]
            with (
                patch.object(
                    scene.coupler._tet_step_guard,
                    "check",
                    side_effect=NeedleStepTooLarge("test rejection"),
                ),
                self.assertRaises(NeedleStepTooLarge),
            ):
                scene.coupler.step(
                    scene.state_0,
                    scene.state_1,
                    scene.control,
                    scene.sim_dt,
                    external_forces=scene._external_forces,
                )
            np.testing.assert_array_equal(_native(scene), native)
            for name, expected in saved.items():
                np.testing.assert_array_equal(getattr(scene.state_1, name).numpy(), expected)
            np.testing.assert_array_equal(scene.state_0.body_f.numpy(), forces)
            for expected, array in zip(thread, scene.worker_thread.state, strict=True):
                np.testing.assert_array_equal(array.numpy(), expected)
            self.assertEqual(scene.worker_thread.accepted_steps, 0)
            self.assertEqual(scene.elapsed, 0)

    def test_adaptive_retry_advances_thread_only_for_accepted_intervals(self):
        with _scene() as (scene, config):
            check = scene.coupler._tet_step_guard.check
            attempts = 0

            def once(state_in, candidate):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise NeedleStepTooLarge("one unresolved candidate")
                return check(state_in, candidate)

            with patch.object(scene.coupler._tet_step_guard, "check", side_effect=once):
                scene.step()
            self.assertEqual(scene.step_refinements, 1)
            self.assertEqual(scene.coupler.tet_step_rejections, 1)
            self.assertEqual(attempts, scene.accepted_physics_steps + 1)
            self.assertEqual(scene.worker_thread.accepted_steps, scene.accepted_physics_steps)
            self.assertAlmostEqual(scene.elapsed, config.frame_dt)
            self.assertAlmostEqual(scene.worker_thread.physical_time_s, config.frame_dt)
            self.assertTrue(scene.running, scene.failure_reason)


if __name__ == "__main__":
    unittest.main()
