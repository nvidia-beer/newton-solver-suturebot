# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Short MuJoCo/FEM integration checks on two detached regional test blocks."""

import os
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from test_needle_insertion_geometry import _boxes

from newton_solver_suturebot import NeedleTissueCoupler, SolverMuJoCoFEM, boundary_faces
from newton_solver_suturebot.examples import example_fem_needle_insertion as insertion
from newton_solver_suturebot.examples import example_fem_phantom_regions as phantom

_REFERENCE = _boxes([((-0.03, -0.02, -0.02), (0.03, 0.02, 0)), ((0.07, -0.02, -0.02), (0.13, 0.02, 0))], [1, 2])


def _selection(_path, ids):
    chosen = np.flatnonzero(np.isin(_REFERENCE.tetrahedron_region_ids, ids))
    tetrahedra = _REFERENCE.indices[chosen]
    vertices = np.unique(tetrahedra)
    local = np.searchsorted(vertices, tetrahedra).astype(np.int32)
    triangles, owners = boundary_faces(local)
    labels = _REFERENCE.tetrahedron_region_ids[chosen]
    return replace(
        _REFERENCE,
        vertices=_REFERENCE.vertices[vertices],
        indices=local,
        surface_triangles=triangles,
        exterior_surface_triangles=triangles,
        tetrahedron_region_ids=labels,
        surface_region_ids=labels[owners],
        exterior_surface_region_ids=labels[owners],
        global_vertex_indices=vertices,
        global_tetrahedron_indices=chosen,
        regions=tuple(region for region in _REFERENCE.regions if region.id in ids),
    )


@contextmanager
def _example(*, region_catalog=None, region_loader=None, tunnel_model="penalty", needle_solver=None):
    with (
        TemporaryDirectory() as directory,
        wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu")),
        patch.object(
            phantom, "list_mesh_regions", return_value=_REFERENCE.regions if region_catalog is None else region_catalog
        ),
        patch.object(phantom, "load_mesh_regions", side_effect=_selection if region_loader is None else region_loader),
    ):
        arguments = ["--asset", str(Path(directory) / "pad.usd"), "--regions", "1", "--substeps", "32"]
        # Existing analytic/explicit-force fixtures exercise the penalty API.
        # None leaves the real example default intact for its own integration test.
        if tunnel_model is not None:
            arguments.extend(("--tunnel-model", tunnel_model))
        if needle_solver is not None:
            arguments.extend(("--needle-solver", needle_solver))
        args = insertion.Example.create_parser().parse_args(arguments)
        viewer = ViewerNull()
        try:
            yield insertion.Example(viewer, args)
        finally:
            viewer.close()


def _prepare(example):
    example.select_point_xy("entry", -0.005, 0)
    example.select_point_xy("exit", 0.005, 0)
    example.preview()


class NeedleInsertionExampleTests(unittest.TestCase):
    def test_solver_selection_constructs_public_class_and_advances_physical_state(self):
        for requested, expected_class in ((None, NeedleTissueCoupler), ("mujoco-fem", SolverMuJoCoFEM)):
            with self.subTest(solver=requested), _example(tunnel_model=None, needle_solver=requested) as example:
                self.assertEqual(example.args.needle_solver, requested or "partitioned")
                self.assertIsNone(example.report()["needle_solver_class"])
                _prepare(example)
                example.start()
                self.assertIs(type(example.coupler), expected_class)
                before = example.state_0.body_q.numpy()[example.needle_body].copy()
                fixed = example.state_0.particle_q.numpy()[example.fixed_vertex_indices].copy()
                example.step()
                self.assertTrue(example.running, example.failure_reason)
                self.assertGreater(example.elapsed, 0)
                self.assertGreater(example.accepted_physics_steps, 0)
                after = example.state_0.body_q.numpy()[example.needle_body]
                self.assertGreater(np.linalg.norm(after[:3] - before[:3]), 0)
                np.testing.assert_array_equal(example.state_0.particle_q.numpy()[example.fixed_vertex_indices], fixed)
                report = example.report()
                self.assertEqual(report["needle_solver_requested"], requested or "partitioned")
                self.assertEqual(report["needle_solver_class"], expected_class.__name__)
                self.assertEqual(report["coupling_graph_status"], example.coupler.coupling_graph_status)
                if example.device.is_cpu:
                    self.assertEqual(report["simultaneous_steps"], 0)
                    self.assertNotEqual(report["coupling_graph_status"], "simultaneous momentum/friction")
                example.test_final()

    def test_default_tunnel_uses_alm_and_advances_the_real_coupled_step(self):
        with _example(tunnel_model=None) as example:
            self.assertEqual(example.args.tunnel_model, "alm")
            _prepare(example)
            example.start()
            alm = example.interaction.tunnel_alm
            self.assertIsNotNone(alm)
            self.assertGreater(alm.compliance_per_m, 0)
            example.step()
            self.assertTrue(example.running, example.outcome)
            self.assertGreater(example.elapsed, 0)
            self.assertGreater(example.accepted_physics_steps, 0)
            self.assertFalse(alm._open)
            self.assertEqual(example.report()["tunnel_model"], "alm")
            example.test_final()

    def test_compact_integrity_checks_detect_inversion_fixed_motion_and_nonfinite_velocity(self):
        with _example() as example:
            original = example.state_0.particle_q.numpy().copy()
            example._check_tissue_integrity()
            free = np.setdiff1d(np.arange(len(original)), example.fixed_vertex_indices)[0]
            positions = original.copy()
            positions[free, 2] = -0.03
            example.state_0.particle_q.assign(positions)
            with self.assertRaisesRegex(AssertionError, "inverted"):
                example._check_tissue_integrity()
            positions = original.copy()
            positions[example.fixed_vertex_indices[0], 0] += 1e-6
            example.state_0.particle_q.assign(positions)
            with self.assertRaisesRegex(AssertionError, "Fixed"):
                example._check_tissue_integrity()
            example.state_0.particle_q.assign(original)
            velocities = example.state_0.particle_qd.numpy()
            velocities[free, 1] = np.inf
            example.state_0.particle_qd.assign(velocities)
            with self.assertRaisesRegex(AssertionError, "nonfinite"):
                example._check_tissue_integrity()

    def test_nonfinite_accepted_state_stops_for_each_position_and_velocity_kind(self):
        cases = (
            ("particle_q", 2, np.nan),
            ("particle_qd", 1, np.inf),
            ("body_q", 6, np.nan),
            ("body_qd", 5, -np.inf),
        )
        for name, component, value in cases:
            with self.subTest(state_array=name), _example() as example:
                _prepare(example)
                example.start()

                def corrupted_step(state_in, state_out, _control, _dt, name=name, component=component, value=value):
                    state_out.assign(state_in)
                    array = getattr(state_out, name)
                    values = array.numpy()
                    values[0, component] = value
                    array.assign(values)

                with patch.object(example.coupler, "step", side_effect=corrupted_step):
                    example.step()
                self.assertFalse(example.running)
                self.assertEqual(example.task_status, "failed")
                self.assertIn("nonfinite", example.failure_reason)
                self.assertEqual(example.accepted_physics_steps, 1)

    def test_current_deformation_and_historical_peak_follow_material_motion(self):
        with _example() as example:
            _prepare(example)
            original = example.state_0.particle_q.numpy().copy()
            free = np.setdiff1d(np.arange(len(original)), example.fixed_vertex_indices)[0]
            desired = example.state_0.body_q.numpy()[example.needle_body].copy()
            for time, offset in ((0.125, (0.0002, -0.0003, 0.0004)), (0.25, (0.0001, 0, 0))):
                positions = original.copy()
                positions[free] += offset
                example.state_0.particle_q.assign(positions)
                example.elapsed = time
                pose_error = example._observe_state(desired)
                expected = float(np.linalg.norm(positions - example.selection.vertices, axis=1).max())
                self.assertAlmostEqual(example.current_deformation, expected, delta=1e-12)
                self.assertEqual(pose_error, 0.0)
                if time == 0.125:
                    peak = expected
                self.assertAlmostEqual(example.maximum_deformation, peak, delta=1e-12)
                self.assertEqual(example.maximum_deformation_time, 0.125)
            np.testing.assert_array_equal(
                example.state_0.particle_q.numpy()[example.fixed_vertex_indices],
                original[example.fixed_vertex_indices],
            )

    def test_render_empty_selection_then_return_to_active_region(self):
        with _example() as example:
            _prepare(example)
            example.select_regions([])
            self.assertIsNone(example.entry)
            self.assertIsNone(example.exit)
            self.assertEqual(example.model.particle_count, 0)
            example.render()
            np.testing.assert_array_equal(example.display_points.numpy(), example.reference.vertices.astype(np.float32))
            example.select_regions([1])
            example.render()
            self.assertGreater(example.model.particle_count, 0)

    def test_compact_render_preserves_triangles_through_motion_opacity_and_region_change(self):
        from test_phantom_regions_example import _CATALOG
        from test_phantom_regions_example import _selection as adjoining_selection

        with _example(region_catalog=_CATALOG, region_loader=adjoining_selection) as example:
            scene = {}
            uploads = []

            def log_mesh(name, points, indices, *_attributes, **material):
                if name.startswith("/phantom/"):
                    scene[name] = (points.numpy().copy(), indices.numpy().copy(), material)
                    uploads.append(name)

            def check_scene(opacity):
                displayed = example.reference.vertices.astype(np.float32).copy()
                selected = np.searchsorted(
                    example.reference.global_vertex_indices, example.selection.global_vertex_indices
                )
                displayed[selected] = example.state_0.particle_q.numpy()
                for region in example.regions:
                    points, indices, material = scene[f"/phantom/region_{region.id:02d}"]
                    triangles = example.reference.exterior_surface_triangles[
                        example.reference.exterior_surface_region_ids == region.id
                    ]
                    np.testing.assert_array_equal(points[indices.reshape(-1, 3)], displayed[triangles])
                    np.testing.assert_array_equal(np.unique(indices), np.arange(len(points)))
                    self.assertEqual(material["opacity"], opacity)
                    self.assertEqual(material["color"], insertion.PAD_COLOR)
                    self.assertFalse(material["backface_culling"])

            # Exercise public persistent-mesh semantics without an OpenGL
            # window; the retained scene must remain complete across frames.
            with (
                patch.object(insertion, "ViewerGL", type(example.viewer)),
                patch.object(example.viewer, "log_mesh", side_effect=log_mesh),
            ):
                inactive_nodes = np.unique(example.reference.indices[example.reference.tetrahedron_region_ids != 1])
                shared = np.flatnonzero(np.isin(example.selection.global_vertex_indices, inactive_nodes))
                self.assertGreater(len(shared), 0)
                self.assertTrue(np.isin(shared, example.fixed_vertex_indices).all())
                example.render()
                check_scene(1.0)
                positions = example.state_0.particle_q.numpy().copy()
                free = np.setdiff1d(np.arange(len(positions)), example.fixed_vertex_indices)
                positions[free] += (0, 0, 0.0004)
                example.state_0.particle_q.assign(positions)
                example.render()
                check_scene(1.0)
                self.assertEqual(uploads.count("/phantom/region_02"), 1)
                example.args.transparent_pad = True
                example.render()
                check_scene(0.55)
                self.assertEqual(uploads.count("/phantom/region_02"), 2)
                example.select_regions([2])
                scene.clear()  # Model replacement may clear the viewer scene.
                example.render()
                check_scene(0.55)
                example.reset()
                scene.clear()
                example.render()
                check_scene(0.55)

    def test_region_switch_clears_material_selections_plan_and_committed_history(self):
        with _example() as example:
            _prepare(example)
            example.start()
            example.step()
            self.assertGreater(example.elapsed, 0)
            self.assertEqual(len(example.actual_tip_history), 1)
            example.select_regions([2])
            self.assertEqual(example.selected_region_ids, (2,))
            self.assertIsNone(example.entry)
            self.assertIsNone(example.exit)
            self.assertIsNone(example.plan)
            self.assertIsNone(example.interaction)
            self.assertEqual(example.actual_tip_history, [])
            self.assertEqual(example.elapsed, 0)
            self.assertFalse(example.running)

    def test_bad_map_clicks_leave_valid_selections_and_plan_unchanged(self):
        with _example() as example:
            _prepare(example)
            entry, exit_point, plan = example.entry, example.exit, example.plan
            with self.assertRaisesRegex(ValueError, "background"):
                example.select_point("entry", 0, 0)
            with self.assertRaisesRegex(ValueError, "coincide"):
                example.select_point_xy("exit", -0.005, 0)
            with self.assertRaisesRegex(ValueError, "selected region"):
                example.select_point_xy("exit", 0.1, 0)
            self.assertIs(example.entry, entry)
            self.assertIs(example.exit, exit_point)
            self.assertIs(example.plan, plan)

    def test_marker_follows_selected_fem_material(self):
        with _example() as example:
            _prepare(example)
            positions = example.state_0.particle_q.numpy()
            positions[example.entry.node_indices] += (0.0002, 0.0003, 0.0004)
            example.state_0.particle_q.assign(positions)
            np.testing.assert_allclose(
                example.entry.position(example.state_0.particle_q),
                example.entry.reference_position + (0.0002, 0.0003, 0.0004),
                atol=2e-9,
            )

    def test_pause_preserves_both_states_and_time_resume_then_reset(self):
        with _example() as example:
            _prepare(example)
            example.start()
            example.step()
            self.assertTrue(example.running, example.outcome)
            example.running = False
            time = example.elapsed
            positions = example.state_0.particle_q.numpy()
            pose = example.state_0.body_q.numpy()
            events = example.interaction.puncture_count
            example.step()
            example.step()
            self.assertEqual(example.elapsed, time)
            self.assertEqual(example.sim_time, time)
            np.testing.assert_array_equal(example.state_0.particle_q.numpy(), positions)
            np.testing.assert_array_equal(example.state_0.body_q.numpy(), pose)
            self.assertEqual(example.interaction.puncture_count, events)
            example.start()
            example.step()
            self.assertGreater(example.elapsed, time)
            example.reset()
            self.assertEqual(example.elapsed, 0)
            self.assertEqual(example.sim_time, 0)
            self.assertIsNone(example.plan)
            self.assertFalse(example.running)
            np.testing.assert_array_equal(
                example.state_0.particle_q.numpy(), example.selection.vertices.astype(np.float32)
            )

    def test_external_force_and_torque_change_actual_motion_under_same_drive(self):
        poses, desired = [], []
        for loaded in (False, True):
            with _example() as example:
                _prepare(example)
                example.start()
                if loaded:
                    original = example.interaction.add_forces

                    def loaded_forces(state, body, com, forces, dt, original=original, **kwargs):
                        original(state, body, com, forces, dt, **kwargs)
                        wrench = state.body_f.numpy()
                        wrench[body] += (0, 0.002, 0, 0, 0, 0.000003)
                        state.body_f.assign(wrench)

                    example.interaction.add_forces = loaded_forces
                initial = example.state_0.body_q.numpy()[0].copy()
                for _ in range(4):
                    example.step()
                self.assertTrue(example.running, example.outcome)
                self.assertGreater(np.linalg.norm(example.state_0.body_q.numpy()[0] - initial), 1e-7)
                poses.append(example.state_0.body_q.numpy()[0].copy())
                desired.append(example.desired_pose.copy())
                example.test_final()
        np.testing.assert_array_equal(desired[0], desired[1])
        self.assertGreater(abs(poses[1][1] - poses[0][1]), 1e-6)
        self.assertGreater(np.linalg.norm(poses[1][3:] - poses[0][3:]), 1e-5)

    def test_controller_requires_actual_exit_and_continues_past_an_intermediate_exit(self):
        with _example() as example:
            _prepare(example)
            example.start()
            target = example.exit.position(example.state_0.particle_q)
            example.interaction.events = ()
            with patch.object(example, "actual_tip", return_value=target):
                example._observe_interaction()
            self.assertTrue(example.running)
            self.assertFalse(example.report()["success"])
            intermediate = target + (0.003, 0, 0)
            example.interaction.events = ({"kind": "exit", "position": intermediate},)
            with patch.object(example, "actual_tip", return_value=intermediate):
                example._observe_interaction()
            self.assertTrue(example.running)
            self.assertIsNone(example.exit_error)
            example.interaction.events = ({"kind": "exit", "position": target},)
            with patch.object(example, "actual_tip", return_value=target):
                example._observe_interaction()
            self.assertFalse(example.running)
            self.assertTrue(example.report()["success"])
            self.assertEqual(example.exit_error, 0)

    def test_generated_tissue_reaction_survives_fem_and_drives_mujoco(self):
        velocities, desired = [], []
        for coupled in (False, True):
            with _example() as example:
                _prepare(example)
                example.start()
                pose = example.plan.pose(0.0).copy()
                # The real spherical tip overlaps the undeformed top surface;
                # its center remains outside. Only initial setup sets a pose.
                pose[2] += 0.6 * example.geometry.thickness
                example._initialize_pose(pose)
                example.state_0.clear_forces()
                example._external_forces.zero_()
                example.drive.add_forces(example.state_0, pose, np.zeros(6))
                desired.append(pose)
                if coupled:
                    example.interaction.add_forces(
                        example.state_0,
                        example.needle_body,
                        example.drive.com,
                        example._external_forces,
                        example.sim_dt,
                    )
                    self.assertEqual(example.interaction.status, "contact")
                wrench = example.state_0.body_f.numpy().copy()
                if coupled:
                    self.assertGreater(np.linalg.norm(wrench[0, :3]), 1e-4)
                    self.assertGreater(np.linalg.norm(wrench[0, 3:]), 1e-8)
                    np.testing.assert_allclose(
                        wrench[0, :3] + example._external_forces.numpy().sum(axis=0),
                        0,
                        atol=1e-8,
                    )
                example.solver.step(
                    example.state_0,
                    example.state_1,
                    example.control,
                    None,
                    example.sim_dt,
                    external_forces=example._external_forces,
                )
                np.testing.assert_array_equal(example.state_0.body_f.numpy(), wrench)
                example.rigid_solver.step(
                    example.state_0,
                    example.state_1,
                    example.control,
                    None,
                    example.sim_dt,
                )
                velocities.append(example.state_1.body_qd.numpy()[0].copy())
        np.testing.assert_array_equal(desired[0], desired[1])
        self.assertGreater(np.linalg.norm(velocities[1][:3] - velocities[0][:3]), 1e-6)
        self.assertGreater(np.linalg.norm(velocities[1][3:] - velocities[0][3:]), 1e-4)


if __name__ == "__main__":
    unittest.main()
