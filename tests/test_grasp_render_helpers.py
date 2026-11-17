# SPDX-License-Identifier: Apache-2.0
"""Standalone physical visuals preserve asset frames and accepted snapshots."""

import copy
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleGeometry
from newton_solver_suturebot.examples.grasp._grasp_render import (
    PhysicalSurgicalRenderScene,
    tube_triangles,
    tube_vertices,
)
from newton_solver_suturebot.examples.grasp._grasp_surface import FullSurfaceGeometry
from newton_solver_suturebot.examples.grasp._grasp_view import fit_inspection_camera, framing_points, update_scene
from newton_solver_suturebot.examples.grasp._grasp_visual_assets import (
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
        # Shading data is per triangle corner, so every corner is a vertex of the visual mesh;
        # positions and triangles are those of the CAD.
        indices = np.asarray(prim.GetAttribute("faceVertexIndices").Get())
        points = np.asarray(prim.GetAttribute("points").Get())
        np.testing.assert_allclose(mesh.vertices, points[indices], atol=1e-9)
        np.testing.assert_array_equal(mesh.triangles.reshape(-1), np.arange(len(indices)))
        np.testing.assert_array_equal(mesh.source_vertex_indices, indices)
        np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1, atol=2e-7)
        np.testing.assert_allclose(mesh.normals, np.asarray(prim.GetAttribute("normals").Get()), atol=1e-6)
        corners = mesh.vertices[mesh.triangles]
        face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        face /= np.linalg.norm(face, axis=1)[:, None]
        # a corner normal stays within the 35 degree crease angle of its own face
        cosines = np.einsum("tki,ti->tk", mesh.normals[mesh.triangles], face)
        self.assertGreater(cosines.min(), np.cos(np.radians(35.0)) - 1e-4)
        self.assertEqual(mesh.uvs.shape, (len(indices), 2))
        self.assertTrue(((mesh.uvs[:, 0] >= 0) & (mesh.uvs[:, 0] <= 1)).all())
        # No triangle of the wire spans the seam around it; only faces of the two ends, which
        # cross the wire, cover more than half a turn.
        around = np.ptp(mesh.uvs[:, 1].reshape(-1, 3), axis=1)
        self.assertLess(np.count_nonzero(around >= 0.5), 0.01 * len(around))
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

    def test_collision_capsules_are_hidden_collision_shapes_on_their_bodies_for_the_viewer_option(self):
        geometry, _ = _fixture()
        needle = NeedleGeometry.load()
        geometry["needle_centerline"] = needle.centerline.copy()
        geometry["needle_radius"] = needle.thickness
        # One tool capsule along x on body 0, as a tool publishes them.
        tool = {
            "capsule_bodies": [0],
            "capsule_endpoints": np.array([[[0.0, 0.0, 0.0], [0.008, 0.0, 0.0]]]),
            "capsule_radii": [0.0022],
        }
        from newton_solver_suturebot.examples.grasp._grasp_render import collision_capsules

        capsules = collision_capsules({**geometry, "grippers": [tool]})
        self.assertEqual(len(capsules), 1 + len(needle.centerline) - 1)
        self.assertEqual([body for body, *_ in capsules], [0] + [1] * (len(needle.centerline) - 1))
        np.testing.assert_array_equal(capsules[1][1], needle.centerline[0])
        np.testing.assert_array_equal(capsules[-1][2], needle.centerline[-1])
        self.assertEqual(capsules[1][3], needle.thickness)
        # In the model: collision flag without the visible flag, on the right bodies, with the
        # capsule axis along the segment; the visual meshes keep no collision flag.
        builder = newton.ModelBuilder()
        scene = PhysicalSurgicalRenderScene(builder, geometry, 2, "cpu")
        model = builder.finalize(device="cpu")
        flags = model.shape_flags.numpy()
        collide, visible = int(newton.ShapeFlags.COLLIDE_SHAPES), int(newton.ShapeFlags.VISIBLE)
        count = len(needle.centerline) - 1
        self.assertEqual(len(scene.collision_shapes), 2 * count)
        self.assertTrue(np.all(flags[scene.collision_shapes] & collide))
        self.assertFalse(np.any(flags[scene.collision_shapes] & visible))
        others = np.setdiff1d(np.arange(model.shape_count), scene.collision_shapes)
        self.assertFalse(np.any(flags[others] & collide))
        np.testing.assert_array_equal(
            model.shape_body.numpy()[scene.collision_shapes], np.repeat(scene.body_indices[:, 1], count)
        )
        transforms = model.shape_transform.numpy()[scene.collision_shapes[:count]]
        scale = model.shape_scale.numpy()[scene.collision_shapes[:count]]
        for row, (start, end) in enumerate(zip(needle.centerline[:-1], needle.centerline[1:], strict=True)):
            np.testing.assert_allclose(transforms[row, :3], (start + end) / 2, atol=1e-7)
            axis = np.asarray(wp.quat_rotate(wp.quat(*transforms[row, 3:]), wp.vec3(0.0, 0.0, 1.0)))
            np.testing.assert_allclose(axis, (end - start) / np.linalg.norm(end - start), atol=1e-5)
            self.assertAlmostEqual(scale[row, 0], needle.thickness, delta=1e-9)
            self.assertAlmostEqual(scale[row, 1], np.linalg.norm(end - start) / 2, delta=1e-8)

    def test_rendered_thread_tube_is_the_host_tube_with_display_offsets(self):
        # The per-frame kernel writes the rings of `tube_vertices` into the thread's slots of the
        # render particles, after each world's tissue surface block, and adds the display offsets.
        geometry, observations = _fixture()
        builder = newton.ModelBuilder()
        scene = PhysicalSurgicalRenderScene(builder, geometry, 2, "cpu")
        state = builder.finalize(device="cpu").state()
        count = scene.thread_point_count
        t = np.linspace(0.0, 1.0, count)
        for world, observation in enumerate(observations):
            # A curved, twisting line: the transported frame is not constant along it.
            observation["thread_positions"] = np.column_stack(
                (0.02 * t + 0.01 * world, 0.004 * np.sin(5.0 * t), 0.004 * np.cos(3.0 * t) + 0.01)
            ).astype(np.float32)
        centers = np.asarray([observation["thread_positions"] for observation in observations], dtype=np.float32)
        expected = tube_vertices(centers, scene.thread_radius)
        surface = len(scene.surface.rest_positions)
        block = surface + count * 8
        update_scene(scene, state, observations)
        particles = state.particle_q.numpy().reshape(2, block, 3)
        np.testing.assert_allclose(particles[:, surface:], expected, rtol=0, atol=2e-8)
        tissue = particles[:, :surface].copy()
        offsets = np.zeros((2, block, 3), dtype=np.float32)
        offsets[1] += [0.45, 0.0, 0.0]
        update_scene(scene, state, observations, particle_offsets=offsets.reshape(-1, 3))
        shifted = state.particle_q.numpy().reshape(2, block, 3)
        np.testing.assert_allclose(shifted[:, surface:], expected + offsets[:, surface:], rtol=0, atol=5e-8)
        np.testing.assert_allclose(shifted[:, :surface], tissue + offsets[:, :surface], rtol=0, atol=5e-8)
        # A repeated point leaves no tangent to build the tube on.
        observations[0]["thread_positions"][1] = observations[0]["thread_positions"][0]
        with self.assertRaisesRegex(ValueError, "degenerate"):
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
