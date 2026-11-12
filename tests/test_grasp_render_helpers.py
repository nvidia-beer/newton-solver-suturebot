# SPDX-License-Identifier: Apache-2.0
"""Standalone physical visuals preserve asset frames and accepted snapshots."""

import copy
import unittest

import newton
import numpy as np

from newton_solver_suturebot import NeedleGeometry
from newton_solver_suturebot.examples._grasp_render import PhysicalSurgicalRenderScene, tube_triangles, tube_vertices
from newton_solver_suturebot.examples._grasp_surface import FullSurfaceGeometry
from newton_solver_suturebot.examples._grasp_view import fit_inspection_camera, framing_points, update_scene
from newton_solver_suturebot.examples._grasp_visual_assets import (
    load_dvrk_tool,
    load_needle_visual,
    physical_tool_parts,
)


def _fixture():
    needle = NeedleGeometry.load()
    vertices = np.array([[-0.025, -0.025, 0], [0.025, -0.025, 0], [0.025, 0.025, 0], [-0.025, 0.025, 0]])
    bodies = np.array([[0.02, 0.01, 0.01, 0, 0, 0, 1], [0, 0, 0.01, 0, 0, 0, 1]])
    geometry = {
        "tissue_rest_positions": vertices,
        "tissue_triangles": np.array([[0, 1, 2], [0, 2, 3]]),
        "needle_visual_asset": str(needle.asset_path),
        "needle_body_index": 1,
        "rigid_body_labels": ["other", "needle"],
        "grippers": [],
        "thread_radius": 0.00015,
        "thread_point_count": 3,
    }
    observation = {
        "tissue_positions": vertices.copy(),
        "needle_pose": bodies[1].copy(),
        "rigid_body_poses": bodies.copy(),
        "thread_positions": np.array([[-0.01, 0.01, 0.005], [0, 0.015, 0.002], [0.015, 0.01, 0.0002]]),
        "thread_diagnostics": {"accepted_steps": 10},
        "diagnostics": {},
    }
    return geometry, [copy.deepcopy(observation) for _ in range(2)]


class GraspRenderHelpersTests(unittest.TestCase):
    def test_full_surface_updates_by_node_identity_without_mutating_inputs(self):
        reference = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.3, 0.3, -0.5]], dtype=float)
        ids = np.array([4, 2, 0])
        geometry = {
            "reference_positions": reference,
            "reference_triangles": np.array([[0, 1, 2], [2, 1, 3]]),
            "active_reference_indices": ids,
            "tissue_rest_positions": reference[ids],
        }
        surface = FullSurfaceGeometry(geometry)
        particles = np.repeat(reference[ids][None], 2, axis=0)
        particles[0] += [0.1, -0.2, 0.3]
        saved = particles.copy()
        rendered = surface.update(particles)
        np.testing.assert_array_equal(rendered[0, [2, 0]], particles[0, [1, 2]])
        np.testing.assert_array_equal(rendered[0, [1, 3]], reference[[1, 3]])
        np.testing.assert_array_equal(rendered[1], reference[:4])
        np.testing.assert_array_equal(particles, saved)

    def test_tool_local_frames_keep_original_materials_and_shortened_shaft(self):
        for instrument in ("PSM1", "PSM2"):
            with self.subTest(instrument=instrument):
                asset = load_dvrk_tool(instrument=instrument, shaft_length_m=0.24)
                mapping = {mesh.name: i for i, mesh in enumerate(asset.meshes)}
                parts = physical_tool_parts(
                    asset, {"instrument": instrument, "part_body_indices": mapping}, body_count=5
                )
                for (body, local), source in zip(parts, asset.meshes, strict=True):
                    transform = asset.component_transforms[source.name]
                    world = local.vertices @ transform[:3, :3].T + transform[:3, 3]
                    np.testing.assert_allclose(world, source.vertices, atol=4e-8, rtol=0)
                    np.testing.assert_array_equal(local.triangles, source.triangles)
                    self.assertEqual(body, mapping[source.name])
                    self.assertEqual(
                        (local.diffuse_rgb, local.roughness, local.metallic),
                        (source.diffuse_rgb, source.roughness, source.metallic),
                    )
                self.assertEqual(asset.metadata["shaft_centerline_length_m"], 0.24)
                self.assertGreater(asset.meshes[0].vertices[:, 1].min(), -0.26)

    def test_needle_preserves_original_taper_and_polished_material(self):
        from pxr import Usd

        geometry = NeedleGeometry.load()
        mesh = load_needle_visual().meshes[0]
        stage = Usd.Stage.Open(str(geometry.asset_path))
        prim = stage.GetPrimAtPath("/Needle/OriginalCAD")
        np.testing.assert_allclose(mesh.vertices, np.asarray(prim.GetAttribute("points").Get()), atol=1e-9)
        np.testing.assert_array_equal(mesh.triangles.reshape(-1), prim.GetAttribute("faceVertexIndices").Get())
        np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1, atol=2e-7)
        self.assertEqual(mesh.metallic, 1)
        self.assertEqual(mesh.roughness, 0.19)

    def test_render_updates_only_disposable_state_and_preserves_world_isolation(self):
        geometry, observations = _fixture()
        builder = newton.ModelBuilder()
        scene = PhysicalSurgicalRenderScene(builder, geometry, 2, "cpu")
        state = builder.finalize(device="cpu").state()
        update_scene(scene, state, observations)
        before = state.body_q.numpy().copy()
        observations[0]["rigid_body_poses"][0, 0] += 0.007
        saved = copy.deepcopy(observations)
        update_scene(scene, state, observations)
        after = state.body_q.numpy()
        np.testing.assert_array_equal(after[scene.body_indices[0, 1]], before[scene.body_indices[0, 1]])
        np.testing.assert_array_equal(after[scene.body_indices[1]], before[scene.body_indices[1]])
        self.assertNotEqual(after[scene.body_indices[0, 0], 0], before[scene.body_indices[0, 0], 0])
        local = state.particle_q.numpy().copy()
        offsets = np.zeros_like(local)
        offsets[len(local) // 2 :, 0] = 0.45
        for _ in range(2):
            update_scene(scene, state, observations, particle_offsets=offsets)
            np.testing.assert_array_equal(state.particle_q.numpy(), local + offsets)
        for actual, original in zip(observations, saved, strict=True):
            for name in ("tissue_positions", "needle_pose", "rigid_body_poses", "thread_positions"):
                np.testing.assert_array_equal(actual[name], original[name])
        points = framing_points(scene, geometry, observations, np.array([[0, 0, 0], [0.45, 0, 0]]))
        eye, target = fit_inspection_camera(points)
        self.assertTrue(np.isfinite([eye, target]).all())
        self.assertGreater(np.linalg.norm(np.subtract(eye, target)), 0)
        observations[0]["rigid_body_poses"][1, 0] += 0.001
        with self.assertRaisesRegex(ValueError, "disagrees"):
            update_scene(scene, state, observations)

    def test_thread_tube_uses_physical_radius_and_rejects_invalid_geometry(self):
        line = np.array([[[0, 0, 0], [0, 0, 0.01], [0, 0, 0.02]]])
        rings = tube_vertices(line, 0.00015).reshape(1, 3, 8, 3)
        np.testing.assert_allclose(np.linalg.norm(rings - line[:, :, None], axis=-1), 0.00015, atol=1e-10)
        triangles = tube_triangles(3)
        self.assertEqual(triangles.shape, (32, 3))
        self.assertLess(triangles.max(), 24)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            tube_vertices(np.zeros((1, 3, 3)), 0.00015)
        with self.assertRaises(ValueError):
            fit_inspection_camera(np.zeros((0, 3)))


if __name__ == "__main__":
    unittest.main()
