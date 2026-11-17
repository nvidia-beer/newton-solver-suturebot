# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepared material labels stay on actual reference triangles and render as spheres on an untextured pad."""

import json
import os
import unittest
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton.sensors import SensorCamera
from newton.viewer import ViewerNull
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade

from newton_solver_suturebot.examples.recording._recording_example import POINT_LABEL_RADIUS, add_point_label_spheres
from newton_solver_suturebot.examples.recording._suturebot_cameras import (
    CameraOverlay,
    camera_rays_cv,
    camera_transform_cv,
)
from newton_solver_suturebot.examples.recording._suturebot_recording import (
    DEFAULT_RECORDING,
    Recording,
    bundled_recordings,
)
from newton_solver_suturebot.examples.recording.example_fem_dvrk import Example


class TestSutureBotPointLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recording = Recording(Path(os.environ.get("SUTUREBOT_RECORDING", DEFAULT_RECORDING)))
        metadata = cls.recording.manifest.get("point_labels")
        if metadata is None:
            raise unittest.SkipTest("This recording has no prepared dataset point labels")
        cls.report = json.loads(cls.recording.asset(metadata["report"]).read_text())
        cls.device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")

    def test_reference_triangle_bindings(self):
        report = self.report
        stage = Usd.Stage.Open(str(self.recording.asset(self.recording.manifest["phantom"]["asset"])))
        self.assertEqual(len(report["points"]), 2)
        self.assertEqual(report["annotation_frame"], self.recording.frame_count - 1)
        self.assertTrue(report["geometry_arrays_unchanged"])
        for row in report["points"]:
            prim = stage.GetPrimAtPath(row["prim_path"])
            points = np.asarray(prim.GetAttribute("newton:referencePoints").Get())
            triangles = np.asarray(prim.GetAttribute("faceVertexIndices").Get()).reshape(-1, 3)
            face = triangles[row["triangle_index"]]
            np.testing.assert_array_equal(face, row["reference_vertex_indices"])
            weights = np.asarray(row["barycentric"])
            self.assertGreaterEqual(weights.min(), 0)
            self.assertAlmostEqual(weights.sum(), 1, places=6)
            transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
            position = (np.r_[weights @ points[face], 1] @ transform)[:3] * UsdGeom.GetStageMetersPerUnit(stage)
            np.testing.assert_allclose(position, row["position_world_m"], atol=2e-8, rtol=0)

    def test_label_spheres_render_display_color_at_material_points(self):
        with wp.ScopedDevice(self.device):
            builder = newton.ModelBuilder()
            builder.add_usd(str(self.recording.asset(self.recording.manifest["phantom"]["asset"])))
            spheres = add_point_label_spheres(builder, self.recording)
            model = builder.finalize(device=self.device)
            self.assertFalse(
                any(
                    source.texture is not None or source.uvs is not None
                    for source in model.shape_source
                    if source is not None
                )
            )
            sensor = SensorCamera(
                model,
                load_textures=True,
                default_render_config=SensorCamera.RenderConfig(
                    enable_textures=True, enable_particles=False, enable_backface_culling=False
                ),
            )
            pose = self.recording.camera_poses[self.report["annotation_frame"], 0]
            points = np.array([row["position_world_m"] for row in self.report["points"]] + [[0.0, 0.0, 0.0]])
            directions = (points - pose[:3, 3]) @ pose[:3, :3]
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            rays, _ = camera_rays_cv(directions[None].astype(np.float32), device=self.device)
            albedo = sensor.allocate_image_albedo(1, 3, 1)
            shape_index = sensor.allocate_image_shape_index(1, 3, 1)
            sensor.update(
                model.state(),
                wp.array([camera_transform_cv(pose)], dtype=wp.transform, device=self.device),
                rays,
                albedo_image=albedo,
                shape_index_image=shape_index,
            )
            colors = SensorCamera.Utils.to_rgba_from_color(albedo).numpy()[0, 0, :, :3]
            hits = shape_index.numpy()[0, 0]
            # Rays toward the material points stop on the label spheres; an
            # unlabelled pad point keeps the plain pad material color.
            source = np.asarray(Image.open(self.recording.asset(self.report["source_mask"])).convert("RGB"))
            for column, row in enumerate(self.report["points"]):
                x, y = np.rint(row["pixel_original"]).astype(int)
                self.assertTrue(np.any(source[y, x] > 0))
                self.assertEqual(hits[column], spheres[row["label"]])
            self.assertEqual(self.report["display_color_rgb"], [0, 255, 0])
            np.testing.assert_allclose(colors[:2], [self.report["display_color_rgb"]] * 2, atol=10, rtol=0)
            self.assertNotIn(hits[2], spheres.values())
            pad_color = 255.0 * model.shape_color.numpy()[hits[2]]
            np.testing.assert_allclose(colors[2], pad_color, atol=2, rtol=0)

    def test_every_labeled_recording_shows_spheres_on_an_untextured_pad(self):
        labeled = 0
        with wp.ScopedDevice(self.device):
            for name, path in bundled_recordings(include_aliases=False).items():
                with self.subTest(recording=name):
                    viewer = ViewerNull()
                    args = Example.create_parser().parse_args(["--sample", str(path), "--no-camera-overlays"])
                    example = Example(viewer, args)
                    try:
                        recording, model = example.recording, example.model
                        stage = Usd.Stage.Open(str(recording.scene_path))
                        for prim in Usd.PrimRange(stage.GetDefaultPrim().GetChild("Phantom")):
                            if prim.GetTypeName() != "Mesh":
                                continue
                            self.assertFalse(prim.HasAttribute("primvars:st"), prim.GetPath())
                            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                            for shader in Usd.PrimRange(material.GetPrim()):
                                self.assertFalse(shader.HasAttribute("inputs:file"), shader.GetPath())
                        spheres = {
                            label: index
                            for index, label in enumerate(model.shape_label)
                            if label.startswith("/PointLabels/")
                        }
                        labels = recording.manifest.get("point_labels")
                        if labels is None:
                            self.assertFalse(spheres)
                            continue
                        labeled += 1
                        report = json.loads(recording.asset(labels["report"]).read_text())
                        self.assertEqual(len(report["points"]), labels["count"])
                        self.assertEqual(set(spheres), {f"/PointLabels/{point['label']}" for point in report["points"]})
                        transforms, scales = model.shape_transform.numpy(), model.shape_scale.numpy()
                        colors, flags, bodies = (
                            model.shape_color.numpy(),
                            model.shape_flags.numpy(),
                            model.shape_body.numpy(),
                        )
                        for point in report["points"]:
                            index = spheres[f"/PointLabels/{point['label']}"]
                            np.testing.assert_allclose(transforms[index][:3], point["position_world_m"], atol=1.0e-7)
                            self.assertAlmostEqual(float(scales[index][0]), POINT_LABEL_RADIUS, places=9)
                            np.testing.assert_allclose(colors[index], np.asarray(labels["display_color_rgb"]) / 255.0)
                            self.assertEqual(bodies[index], -1)
                            self.assertEqual(flags[index], int(newton.ShapeFlags.VISIBLE))
                        # The annotation camera at the annotation frame sees every label sphere.
                        frame, camera = report["annotation_frame"], report["camera"]
                        overlay = CameraOverlay(
                            model, recording.cameras, recording.root, camera_order=recording.camera_order
                        )
                        try:
                            poses, valid = recording.camera_frame(frame)
                            overlay.render(example.state_0, frame, poses, valid)
                            for label, index in spheres.items():
                                self.assertGreater(np.count_nonzero(overlay.shape_indices[camera] == index), 0, label)
                        finally:
                            overlay.close()
                    finally:
                        example.close()
                        viewer.close()
        self.assertEqual(labeled, 3)


if __name__ == "__main__":
    unittest.main()
