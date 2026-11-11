# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The illustrative metal preset reaches Newton without changing replay geometry."""

import unittest

import newton
import numpy as np
import warp as wp
from pxr import Usd, UsdShade

from newton_solver_suturebot.examples._suturebot_recording import Recording, bundled_recordings
from newton_solver_suturebot.examples.example_fem_versius import DEFAULT_RECORDING as VERSIUS_RECORDING


def _material_values(material):
    shader = material.ComputeSurfaceSource()[0]
    return (
        tuple(shader.GetInput("diffuseColor").Get()),
        shader.GetInput("roughness").Get(),
        shader.GetInput("metallic").Get(),
    )


def _import(path):
    builder = newton.ModelBuilder()
    builder.add_usd(str(path), collapse_fixed_joints=False, hide_collision_shapes=True)
    return builder


class DvrkMaterialTests(unittest.TestCase):
    def test_all_recordings_import_versius_metals_without_recoloring_pad_or_black_shaft(self):
        # Read the independent packaged source materials, not exporter output
        # metadata or a second copy of its numerical constants.
        versius = Recording(VERSIUS_RECORDING, catalog={})
        source = Usd.Stage.Open(str(versius.scene_path))
        material_paths = {
            "pitch": "/VersiusPreview/Left/shaft/CAD/Materials/shaft_shaft_metal",
            "yaw": "/VersiusPreview/Left/wrist/CAD/Materials/wrist_trim_metal",
            "jaw1": "/VersiusPreview/Left/jaw_1/CAD/Materials/jaw_1",
            "jaw2": "/VersiusPreview/Left/jaw_2/CAD/Materials/jaw_2",
        }
        expected = {
            f"/SutureBotReplay/{arm}/{part}/CAD": _material_values(UsdShade.Material(source.GetPrimAtPath(path)))
            for arm in ("PSM1", "PSM2")
            for part, path in material_paths.items()
        }
        recordings = bundled_recordings(include_aliases=False)
        self.assertEqual(len(recordings), 4)
        with wp.ScopedDevice("cpu"):
            for name, path in recordings.items():
                with self.subTest(recording=name):
                    recording = Recording(path)
                    baseline = _import(recording.root / "scene_tools.usda")
                    current = _import(recording.scene_path)
                    stage = Usd.Stage.Open(str(recording.scene_path))
                    self.assertEqual(current.shape_label, baseline.shape_label)
                    self.assertEqual(current.body_label, baseline.body_label)
                    for field in ("shape_body", "shape_transform", "body_q", "body_mass", "body_inertia"):
                        np.testing.assert_array_equal(getattr(current, field), getattr(baseline, field), err_msg=field)
                    changed = set()
                    pad_meshes = 0
                    for index, label in enumerate(current.shape_label):
                        old, new = baseline.shape_source[index], current.shape_source[index]
                        if old is None:
                            self.assertIsNone(new)
                            np.testing.assert_array_equal(current.shape_color[index], baseline.shape_color[index])
                            continue
                        for field in ("vertices", "indices", "normals", "uvs"):
                            a, b = getattr(old, field), getattr(new, field)
                            if a is None:
                                self.assertIsNone(b, f"{label}: {field}")
                            else:
                                np.testing.assert_array_equal(a, b, err_msg=f"{label}: {field}")
                        before = (old.color, old.roughness, old.metallic)
                        after = (new.color, new.roughness, new.metallic)
                        if after != before:
                            changed.add(label)
                        if label in expected:
                            material = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(label)).ComputeBoundMaterial()[0]
                            self.assertTrue(material, label)
                            authored = _material_values(material)
                            self.assertEqual(authored, expected[label], label)
                            # UsdPreviewSurface colors are linear; Newton's
                            # viewer mesh stores their sRGB representation.
                            linear = np.asarray(authored[0])
                            srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
                            np.testing.assert_allclose(new.color, srgb, rtol=0, atol=1.0e-7, err_msg=label)
                            self.assertAlmostEqual(new.roughness, authored[1], places=7, msg=label)
                            self.assertEqual(new.metallic, 1.0, label)
                        else:
                            self.assertEqual(after, before, label)
                            self.assertEqual(new.opacity, old.opacity, label)
                            np.testing.assert_array_equal(current.shape_color[index], baseline.shape_color[index])
                        pad_meshes += "/Phantom/" in label
                    self.assertEqual(changed, set(expected))
                    self.assertEqual(pad_meshes, 12)


if __name__ == "__main__":
    unittest.main()
