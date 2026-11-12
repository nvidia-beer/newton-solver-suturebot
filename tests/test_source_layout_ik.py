# SPDX-License-Identifier: Apache-2.0
"""Native reachability and inter-tool clearance for source and custom stitches.

These checks exercise the generated articulated chain without changing live
physics. They do not certify coupled FEM dynamics or measured calibration.
"""

import json
import os
import unittest
from pathlib import Path

import mujoco
import newton
import numpy as np
import warp as wp

from newton_solver_suturebot import load_mesh_regions
from newton_solver_suturebot._src.geometry.needle import NeedleGeometry
from newton_solver_suturebot._src.grasp import GrasperConfig, needle_to_gripper_pose
from newton_solver_suturebot._src.pose_ik import NativePoseIK
from newton_solver_suturebot._src.rcm_grasp import add_rcm_grasper
from newton_solver_suturebot._src.subcycled_mujoco import SubcycledMuJoCo
from newton_solver_suturebot._src.suturebot_layout import load_source_layout, load_source_stitch
from newton_solver_suturebot.examples._needle_insertion_geometry import InsertionPlan, MaterialPoint
from newton_solver_suturebot.grasp_scene import _initial_grasp_parameters
from newton_solver_suturebot.parallel import SutureSceneConfig


def source_material_points():
    """Use the recording's original triangles and material weights on region 1."""
    import newton_solver_suturebot

    stitch = load_source_stitch()
    asset = Path(newton_solver_suturebot.__file__).parent / "examples/assets/phantom_pad/phantom_regions.usd"
    numbering = json.loads(asset.with_name("suturebot_region_map.json").read_text())["regions"]
    source_region = next(item["source_region_id"] for item in numbering if item["id"] == stitch["region_id"])
    selection = load_mesh_regions(asset, [source_region])
    points = []
    for label, position in (("insertion", stitch["entry"]), ("exit", stitch["exit"])):
        binding = stitch["provenance"]["material_bindings"][label]
        nodes = np.asarray(binding["reference_vertex_indices"], dtype=int)
        weights = np.asarray(binding["barycentric"], dtype=float)
        triangle = selection.vertices[nodes]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        np.testing.assert_allclose(weights @ triangle, position, atol=2e-8, rtol=0)
        points.append(MaterialPoint(stitch["region_id"], nodes, weights, np.asarray(position), normal))
    return points


class SourceLayoutIKTests(unittest.TestCase):
    def test_native_tools_reach_source_and_custom_arcs_with_clearance_without_moving_physics(self):
        with wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu")):
            layout = load_source_layout()
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            needle = builder.add_link(mass=0.001, inertia=wp.mat33(np.eye(3) * 1e-7), label="needle")
            joint = builder.add_joint_free(child=needle, label="needle/free")
            builder.add_articulation([joint], label="needle")
            tools = [
                add_rcm_grasper(
                    builder,
                    config=GrasperConfig(shaft_length_m=0.24),
                    label=f"PSM{index + 1}",
                    nominal_insertion_m=0.18,
                    insertion_limits_m=(-0.05, 0.05),
                )
                for index in range(2)
            ]
            model = builder.finalize()
            state = model.state()
            for tool, anchor in zip(tools, layout["trocar_poses"], strict=True):
                tool.set_trocar_pose(model, anchor)
                tool.initialize(model, state)
            solver = SubcycledMuJoCo(model, use_mujoco_cpu=True, kinematic_switchable_roots=False)
            solver.reset(state, flags=0)
            native, live = solver.mj_model, solver.mj_data.qpos.copy()
            body_poses, anchors = state.body_q.numpy().copy(), model.joint_X_p.numpy().copy()
            scratch = mujoco.MjData(native)
            scratch.qpos[:] = live
            iks = []
            for index, tool in enumerate(tools):
                joints = [
                    mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_JOINT, model.joint_label[j].replace("/", "_"))
                    for j in tool.arm_joints
                ]
                body = mujoco.mj_name2id(
                    native, mujoco.mjtObj.mjOBJ_BODY, model.body_label[tool.palm_body].replace("/", "_")
                )
                iks.append(NativePoseIK(native, joints, body))
                # Use actual reset jaw targets, including the open parked tool.
                for joint, sign in zip(tool.jaw_joints, (1, -1), strict=True):
                    native_joint = mujoco.mj_name2id(
                        native, mujoco.mjtObj.mjOBJ_JOINT, model.joint_label[joint].replace("/", "_")
                    )
                    scratch.qpos[native.jnt_qposadr[native_joint]] = sign * (0.35 if index else 0.0)
            parked = iks[1].solve(layout["parked_pose"], live)
            scratch.qpos[iks[1].coordinates] = parked
            mapping = solver.mjc_geom_to_newton_shape.numpy()[0]
            geom_ids = [[int(np.flatnonzero(mapping == shape)[0]) for shape in tool.capsule_shapes] for tool in tools]
            geometry = NeedleGeometry.load()
            legacy_entry = MaterialPoint(
                1,
                np.array([49, 34, 48]),
                np.array([0.15632881, 0.08250713, 0.76116407]),
                np.array([0.0, -0.046, -0.00224092375]),
                np.array([-0.12559007, 0.31112399, 0.9420345]),
            )
            legacy_exit = MaterialPoint(
                1,
                np.array([42, 51, 38]),
                np.array([0.23071421, 0.40375338, 0.3655324]),
                np.array([0.0, -0.054, -0.000705309387]),
                np.array([-0.58781991, -0.66304527, 0.46350698]),
            )
            source_entry, source_exit = source_material_points()
            config = SutureSceneConfig(control_mode="physical_grasp")
            np.testing.assert_allclose(config.entry_xy, source_entry.reference_position[:2], atol=1e-8, rtol=0)
            np.testing.assert_allclose(config.exit_xy, source_exit.reference_position[:2], atol=1e-8, rtol=0)
            for name, entry, exit_point in (
                ("custom", legacy_entry, legacy_exit),
                ("source", source_entry, source_exit),
            ):
                with self.subTest(stitch=name):
                    plan = InsertionPlan.create(geometry, entry, exit_point)
                    grasp = _initial_grasp_parameters(plan)
                    self.assertEqual(grasp["arc_fraction"], 0.8)
                    self.assertAlmostEqual(
                        grasp["tangent_roll_rad"], -5 * np.pi / 12 if name == "source" else -np.pi / 2
                    )
                    self.assertAlmostEqual(grasp["bite_angle_rad"], -np.pi / 12 if name == "source" else 0.0)
                    relative = wp.transform(*needle_to_gripper_pose(geometry, **grasp))
                    iks[0].reset(live)
                    solutions, clearance, shaft_clearance = [], float("inf"), float("inf")
                    for angle in np.linspace(plan.initial_angle, plan.finish_angle, 150):
                        target = np.asarray(wp.transform_multiply(wp.transform(*plan.pose(angle)), relative))
                        q = iks[0].solve(target, live)
                        self.assertTrue(iks[0].last_diagnostics["converged"])
                        self.assertTrue(np.all(q >= tools[0].arm_lower))
                        self.assertTrue(np.all(q <= tools[0].arm_upper))
                        solutions.append(q)
                        scratch.qpos[iks[0].coordinates] = q
                        mujoco.mj_forward(native, scratch)
                        for first in geom_ids[0]:
                            for second in geom_ids[1]:
                                distance = mujoco.mj_geomDistance(native, scratch, first, second, 1.0, None)
                                clearance = min(clearance, distance)
                        shaft_clearance = min(
                            shaft_clearance,
                            mujoco.mj_geomDistance(native, scratch, geom_ids[0][0], geom_ids[1][0], 1.0, None),
                        )
                        if name == "source":
                            # PSM1 remains on its recorded side of the parked PSM2.
                            self.assertGreater(scratch.xpos[iks[0].body_id, 0], scratch.xpos[iks[1].body_id, 0] + 0.01)
                    solutions = np.asarray(solutions)
                    self.assertGreater(np.ptp(solutions[:, 3]), 1.0)
                    self.assertGreater(np.ptp(solutions[:, 2]), 0.004)
                    self.assertGreater(clearance, 0.01 if name == "source" else 0.002)
                    self.assertGreater(shaft_clearance, 0.03 if name == "source" else 0.002)
                    self.assertLess(np.abs(solutions[:, 4:]).max(), 0.95 if name == "source" else 0.6)
            np.testing.assert_array_equal(solver.mj_data.qpos, live)
            np.testing.assert_array_equal(state.body_q.numpy(), body_poses)
            np.testing.assert_array_equal(model.joint_X_p.numpy(), anchors)


if __name__ == "__main__":
    unittest.main()
