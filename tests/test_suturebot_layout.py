# SPDX-License-Identifier: Apache-2.0
"""Source-informed static placement uses registered coordinates and fixed ports."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot._src.suturebot_layout import (
    _RECORDING_DIR,
    _rigid_transform,
    load_source_layout,
    load_source_stitch,
    source_layout_asset_paths,
)


def _rotation(pose):
    return np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=float).reshape(3, 3)


class SutureBotLayoutTests(unittest.TestCase):
    def test_fixed_ports_use_native_pad_coordinates_and_converging_source_axes(self):
        layout = load_source_layout()
        poses = layout["trocar_poses"]
        expected = [
            [0.1159662159223707, -0.03676013671229389, 0.19006977365785985],
            [-0.0541527492115652, -0.027688329130538616, 0.16387886429437495],
        ]
        self.assertEqual(poses.shape, (2, 7))
        np.testing.assert_allclose(poses[:, :3], expected, atol=1e-10)
        self.assertAlmostEqual(np.linalg.norm(poses[0, :3] - poses[1, :3]), 0.17236218762921285, delta=1e-10)
        axes = np.asarray([_rotation(pose)[:, 2] for pose in poses])
        np.testing.assert_allclose(
            axes,
            [[-0.32310986, -0.05612748, -0.94469557], [0.40981805, -0.14883881, -0.89994232]],
            atol=3e-7,
        )
        angle = np.degrees(np.arccos(np.clip(axes[0] @ axes[1], -1, 1)))
        self.assertGreater(angle, 40)
        self.assertLess(angle, 50)

    def test_parked_pose_is_a_bounded_posture_near_registered_psm2_yaw(self):
        layout = load_source_layout()
        recording = json.loads((_RECORDING_DIR / "recording.json").read_text())
        native_from_world = np.linalg.inv(recording["phantom"]["T_world_from_native"])
        with np.load(_RECORDING_DIR / "motion.npz", allow_pickle=False) as motion:
            index = list(motion["body_paths"]).index("/SutureBotReplay/PSM2/yaw")
            expected = native_from_world @ motion["body_transforms"][4, index]
            roll_index = list(motion["body_paths"]).index("/SutureBotReplay/PSM1/roll")
            source_roll = native_from_world @ motion["body_transforms"][4, roll_index]
        provenance = layout["provenance"]
        source_pose = np.asarray(provenance["source_parked_pose"])
        np.testing.assert_allclose(source_pose[:3], expected[:3, 3], atol=1e-10)
        np.testing.assert_allclose(_rotation(source_pose), expected[:3, :3], atol=3e-7)
        np.testing.assert_allclose(layout["parked_pose"][:3], [0.02338113, -0.05360977, 0.01208000], atol=1e-7)
        self.assertLess(np.linalg.norm(layout["parked_pose"][:3] - source_pose[:3]), 0.002)
        relative_rotation = _rotation(layout["parked_pose"]).T @ expected[:3, :3]
        angle = np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))
        self.assertAlmostEqual(angle, 0.1745826, delta=2e-6)
        self.assertEqual(provenance["parked_arm_positions"][4], -1.0)
        self.assertEqual(provenance["parked_nominal_insertion_m"], 0.18)
        self.assertEqual(provenance["psm1_roll_zero_offset_rad"], np.pi)
        np.testing.assert_allclose(
            _rotation(layout["trocar_poses"][0]), source_roll[:3, :3] @ np.diag([-1, -1, 1]), atol=3e-7
        )
        self.assertFalse(layout["provenance"]["absolute_metric_accuracy_validated"])
        self.assertFalse(layout["provenance"]["pad_alignment_accepted"])
        self.assertTrue(layout["provenance"]["static_layout_only"])
        json.dumps(layout["provenance"])

    def test_callers_cannot_mutate_subsequent_layouts(self):
        first = load_source_layout()
        expected = first["trocar_poses"].copy()
        first["trocar_poses"][:] = 0
        first["parked_pose"][:] = 0
        first["provenance"]["sample_id"] = "changed"
        first["provenance"]["parked_arm_positions"][:] = [0] * 6
        second = load_source_layout()
        np.testing.assert_array_equal(second["trocar_poses"], expected)
        self.assertGreater(np.linalg.norm(second["parked_pose"][3:]), 0.99)
        self.assertEqual(second["provenance"]["sample_id"], "tissue3_215300")
        self.assertEqual(second["provenance"]["parked_arm_positions"][4], -1.0)

    def test_nonrigid_or_nonfinite_source_transforms_are_rejected(self):
        for diagonal in ([1, 1, -1, 1], [2, 1, 1, 1], [1, np.nan, 1, 1], [1, 1, 1, 0]):
            with self.subTest(diagonal=diagonal), self.assertRaises(ValueError):
                _rigid_transform(np.diag(diagonal))


class SutureBotStitchTests(unittest.TestCase):
    def setUp(self):
        self.recording = json.loads((_RECORDING_DIR / "recording.json").read_text())
        self.labels = json.loads((_RECORDING_DIR / "point_labels.json").read_text())

    def _load_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "recording.json").write_text(json.dumps(self.recording))
            (directory / "point_labels.json").write_text(json.dumps(self.labels))
            with patch("newton_solver_suturebot._src.suturebot_layout._RECORDING_DIR", directory):
                return load_source_stitch()

    def test_targets_match_material_points_on_the_bundled_phantom(self):
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(_RECORDING_DIR / "phantom.usdc"))
        stitch = load_source_stitch()
        self.assertEqual(stitch["region_id"], 1)
        for point, key in zip(self.labels["points"], ("entry", "exit"), strict=True):
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(point["prim_path"]))
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)[point["triangle_index"]]
            np.testing.assert_array_equal(indices, point["reference_vertex_indices"])
            material_position = np.asarray(point["barycentric"]) @ vertices[indices]
            np.testing.assert_allclose(stitch[key], material_position, atol=1e-9, rtol=0)
            self.assertEqual(
                stitch["provenance"]["material_bindings"][point["label"]]["reference_vertex_indices"], indices.tolist()
            )
        self.assertGreater(stitch["entry"][0], stitch["exit"][0])
        self.assertGreater(stitch["entry"][1], stitch["exit"][1])
        self.assertFalse(stitch["provenance"]["absolute_metric_accuracy_validated"])
        self.assertFalse(stitch["provenance"]["pad_alignment_accepted"])
        json.dumps(stitch["provenance"])

    def test_semantic_labels_preserve_entry_exit_when_file_order_changes(self):
        expected = load_source_stitch()
        self.labels["points"].reverse()
        actual = self._load_modified()
        for key in ("entry", "exit"):
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_rigid_registration_is_inverted_including_rotation(self):
        expected = load_source_stitch()
        transform = np.array([[0, -1, 0, 0.1], [1, 0, 0, -0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]], dtype=float)
        self.recording["phantom"]["T_world_from_native"] = transform.tolist()
        for point, key in zip(self.labels["points"], ("entry", "exit"), strict=True):
            point["position_world_m"] = (transform @ np.r_[expected[key], 1])[:3].tolist()
        actual = self._load_modified()
        for key in ("entry", "exit"):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-15, rtol=0)

    def test_invalid_material_labels_are_rejected(self):
        mutations = [
            ("label", "exit"),
            ("region_id", 0),
            ("region_id", 2),
            ("region_id", 1.0),
            ("triangle_index", -1),
            ("triangle_index", 100000),
            ("reference_vertex_indices", [1, 1, 2]),
            ("reference_vertex_indices", [1, 2, 100000]),
            ("reference_vertex_indices", [1.0, 2.0, 3.0]),
            ("barycentric", [1, 1, 0]),
            ("barycentric", [-0.1, 1.1, 0]),
            ("barycentric", [float("nan"), 0, 1]),
            ("position_world_m", [0, 0]),
            ("position_world_m", [0, float("inf"), 0]),
            ("position_world_m", self.labels["points"][1]["position_world_m"]),
        ]
        original = json.dumps(self.labels)
        for key, value in mutations:
            self.labels = json.loads(original)
            self.labels["points"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self._load_modified()

    def test_returned_targets_are_independent_and_labels_affect_evidence_hashing(self):
        expected = load_source_stitch()
        changed = load_source_stitch()
        changed["entry"][:] = 0
        changed["exit"][:] = 0
        changed["provenance"]["material_bindings"]["insertion"]["barycentric"][:] = [0, 0, 0]
        actual = load_source_stitch()
        for key in ("entry", "exit"):
            np.testing.assert_array_equal(actual[key], expected[key])
        self.assertEqual(actual["provenance"], expected["provenance"])
        self.assertIn(_RECORDING_DIR / "point_labels.json", source_layout_asset_paths())


if __name__ == "__main__":
    unittest.main()
