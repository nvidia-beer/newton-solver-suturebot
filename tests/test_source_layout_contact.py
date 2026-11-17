# SPDX-License-Identifier: Apache-2.0
"""Hold a free needle with the articulated source-layout jaws under gravity."""

import os
import unittest

import mujoco
import newton
import numpy as np
import warp as wp

from newton_solver_suturebot._src.geometry.needle import NeedleGeometry
from newton_solver_suturebot._src.sim.ik.pose_ik import NativePoseIK
from newton_solver_suturebot._src.solvers.mujoco.subcycled_mujoco import SubcycledMuJoCo
from newton_solver_suturebot.examples.grasp._dvrk_grasper import (
    GrasperConfig,
    jaw_opening_on_needle,
    needle_to_gripper_pose,
)
from newton_solver_suturebot.examples.grasp._dvrk_layout import load_source_layout, load_source_stitch
from newton_solver_suturebot.examples.grasp._dvrk_rcm_grasper import add_rcm_grasper
from newton_solver_suturebot.examples.grasp._grasp_scene import _initial_grasp_parameters
from newton_solver_suturebot.examples.needle._needle_insertion_geometry import InsertionPlan, MaterialPoint


class SourceLayoutContactTests(unittest.TestCase):
    def test_adjacent_wrist_proxies_do_not_force_jaws_apart(self):
        with wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu")):
            geometry, layout = NeedleGeometry.load(), load_source_layout()
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            builder.add_usd(str(geometry.asset_path), collapse_fixed_joints=False, hide_collision_shapes=True)
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
            stitch = load_source_stitch()
            points = []
            # Outward normals of the two annotated reference triangles.
            normals = ((0.01005626, 0.52210993, 0.85281891), (-0.03037521, 0.15459991, 0.98751011))
            for label, key, normal in zip(("insertion", "exit"), ("entry", "exit"), normals, strict=True):
                binding = stitch["provenance"]["material_bindings"][label]
                points.append(
                    MaterialPoint(
                        stitch["region_id"],
                        np.asarray(binding["reference_vertex_indices"]),
                        np.asarray(binding["barycentric"]),
                        stitch[key],
                        np.asarray(normal),
                    )
                )
            plan = InsertionPlan.create(geometry, *points)
            needle_pose = plan.pose(plan.initial_angle)
            model = builder.finalize()
            state, following, control = model.state(), model.state(), model.control()
            coordinates = state.joint_q.numpy().copy()
            coordinates[:7] = needle_pose
            state.joint_q.assign(coordinates)
            for tool, anchor in zip(tools, layout["trocar_poses"], strict=True):
                tool.set_trocar_pose(model, anchor)
                tool.initialize(model, state)
            solver = SubcycledMuJoCo(
                model,
                use_mujoco_cpu=True,
                rigid_substeps=4,
                iterations=200,
                ls_iterations=50,
                tolerance=1e-12,
                cone="elliptic",
                integrator="implicitfast",
                impratio=10,
                kinematic_switchable_roots=False,
            )
            solver.reset(state, flags=0)
            relative = needle_to_gripper_pose(geometry, **_initial_grasp_parameters(plan))
            palm_pose = np.asarray(wp.transform_multiply(wp.transform(*needle_pose), wp.transform(*relative)))
            # The jaws start about the needle at the opening at which their capsules touch it.
            needle_in_palm = wp.transform_inverse(wp.transform(*relative))
            grasp_opening = jaw_opening_on_needle(
                tools[0],
                [np.asarray(wp.transform_point(needle_in_palm, wp.vec3(*point))) for point in geometry.centerline],
                geometry.thickness,
            )
            self.assertTrue(all(angle > 0.0 for angle in grasp_opening))
            targets = []
            for index, tool in enumerate(tools):
                native = solver.mj_model
                joints = [
                    mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_JOINT, model.joint_label[j].replace("/", "_"))
                    for j in tool.arm_joints
                ]
                body = mujoco.mj_name2id(
                    native, mujoco.mjtObj.mjOBJ_BODY, model.body_label[tool.palm_body].replace("/", "_")
                )
                ik = NativePoseIK(native, joints, body)
                ik.reset(solver.mj_data.qpos)
                target = ik.solve(palm_pose if index == 0 else layout["parked_pose"], solver.mj_data.qpos)
                targets.append(target)
                opening = grasp_opening if index == 0 else 0.7
                tool.initialize(model, state, arm_positions=target, opening_rad=opening)
                tool.set_arm_targets(control, target)
                # The first tool is commanded closed on the needle; the parked one stays open.
                tool.set_aperture(control, (0 if index == 0 else 0.7) / tool.config.maximum_opening_rad)
            solver.reset(state, flags=0)
            initial = tools[0].observe(model, state, solver, 0)["needle_relative_pose"]
            fixed_anchors = model.joint_X_p.numpy().copy()
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            force[0, 2] = -9.81 * model.body_mass.numpy()[0]
            for _ in range(120):
                state.clear_forces()
                state.body_f.assign(force)
                solver.step(state, following, control, None, 1 / 240)
                state, following = following, state
            observed = tools[0].observe(model, state, solver, 0)
            relative = observed["needle_relative_pose"]
            self.assertLess(np.linalg.norm(relative[:3] - initial[:3]), 0.0001)
            angle = 2 * np.arccos(np.clip(abs(relative[3:] @ initial[3:]), -1, 1))
            self.assertLess(angle, 0.015)
            self.assertTrue(np.all(observed["jaw_needle_contact_count"] > 0))
            self.assertTrue(np.all(observed["jaw_contact_normal_force_n"] > 0.01))
            self.assertTrue(np.all(observed["jaw_contact_normal_force_n"] < 1.0))
            for tool, target in zip(tools, targets, strict=True):
                q = state.joint_q.numpy()[list(tool.jaw_q_indices)]
                self.assertGreaterEqual(q[0], 0)
                self.assertLessEqual(q[1], 0)
                np.testing.assert_allclose(state.joint_q.numpy()[list(tool.arm_q_indices)], target, atol=0.001, rtol=0)
            np.testing.assert_allclose(state.joint_q.numpy()[list(tools[1].jaw_q_indices)], [0.35, -0.35], atol=1e-5)
            np.testing.assert_array_equal(model.joint_X_p.numpy(), fixed_anchors)
            self.assertEqual(solver.mj_model.neq, 0)
            mapping = solver.mjc_geom_to_newton_shape.numpy()[0]
            # Excluded: the pairs among the wrist and the two jaws of one tool, which adjoin and overlap.
            excluded = set()
            for tool in tools:
                distal = [
                    pair for pair in zip(tool.capsule_shapes, tool.capsule_parts, strict=True) if pair[1] != "shaft"
                ]
                excluded |= {frozenset((a, b)) for a, first in distal for b, second in distal if first != second}
            jaws = [
                shape
                for shape, part in zip(tools[0].capsule_shapes, tools[0].capsule_parts, strict=True)
                if part.startswith("jaw")
            ]
            others = set(tools[0].capsule_shapes).difference(jaws).union(tools[1].capsule_shapes)
            negative_pairs = set()
            for contact in solver.mj_data.contact[: solver.mj_data.ncon]:
                pair = frozenset(int(mapping[int(geom)]) for geom in contact.geom)
                self.assertNotIn(pair, excluded)
                if contact.dist < 0:
                    negative_pairs.add(pair)
            # The oblique reset bite must not use an accidental wrist/shaft
            # penetration to hold the otherwise freely jointed needle.
            for pair in negative_pairs:
                self.assertTrue(pair.intersection(jaws))
                self.assertFalse(pair.intersection(others))


if __name__ == "__main__":
    unittest.main()
