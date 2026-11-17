# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Packaged Versius preview identity, portable geometry and actual pose replay."""

import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

from newton_solver_suturebot.examples.recording import _suturebot_cli as cli
from newton_solver_suturebot.examples.recording._suturebot_recording import Recording
from newton_solver_suturebot.examples.recording.example_fem_versius import (
    DEFAULT_RECORDING,
    Example,
    bundled_recordings,
)


class VersiusRecordingTests(unittest.TestCase):
    def test_two_view_generated_preview_preserves_placeholder_identity_and_clocks(self):
        recording = Recording(DEFAULT_RECORDING, catalog={})
        manifest = recording.manifest
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(recording.camera_order, ("left", "right"))
        self.assertEqual(recording.camera_count, 2)
        self.assertFalse(manifest["motion"]["measured"])
        self.assertEqual(manifest["motion"]["kind"], "scripted_preview")
        self.assertFalse(manifest["alignment"]["accepted"])
        self.assertFalse(manifest["alignment"]["absolute_metric_accuracy_validated"])
        for camera in recording.cameras.values():
            self.assertIs(camera["placeholder"], True)
            self.assertIn("dVRK", camera["label"])
            self.assertIn("not a Versius observation", camera["placeholder_reason"])
            self.assertIn("not_registered", camera["status"])
        with np.load(recording.asset(manifest["motion_npz"]), allow_pickle=False) as data:
            source_time = data["placeholder_source_timestamps_ns"]
            self.assertEqual(source_time.dtype, np.dtype("int64"))
            self.assertEqual(source_time.shape, (recording.frame_count,))
            self.assertTrue(np.all(np.diff(source_time) > 0))
            self.assertFalse(np.array_equal(source_time, recording.timestamps_ns))
            np.testing.assert_array_equal(data["placeholder_source_frame_indices"], np.arange(recording.frame_count))

    def test_catalog_and_cli_do_not_select_dvrk_recordings(self):
        catalog = bundled_recordings()
        self.assertIn("versius_stereo_preview", catalog)
        self.assertEqual(catalog["versius_stereo_preview"], DEFAULT_RECORDING)
        self.assertTrue(
            all(path.resolve().is_relative_to(DEFAULT_RECORDING.parent.parent) for path in catalog.values())
        )
        with patch.object(cli, "bundled_recordings", side_effect=AssertionError("must not consult dVRK samples")):
            parser = Example.create_parser()
            output = io.StringIO()
            with (
                patch("sys.argv", ["fem_versius", "--list-samples"]),
                redirect_stdout(output),
                self.assertRaises(SystemExit) as stopped,
            ):
                cli.configure_sample(
                    parser,
                    catalog=bundled_recordings,
                    default_recording=DEFAULT_RECORDING,
                    title=Example.title,
                )
            self.assertEqual(stopped.exception.code, 0)
            self.assertIn("Versius", output.getvalue())
            self.assertIn("dVRK placeholder", output.getvalue())
            self.assertNotIn("tissue3_215300 [default]", output.getvalue())
            resolved = Recording("versius_stereo_preview", catalog=catalog)
            self.assertEqual(resolved.path, DEFAULT_RECORDING.resolve())
            with self.assertRaises(FileNotFoundError):
                Recording("tissue3_215300", catalog=catalog)

    def test_copied_bundle_closes_all_usd_dependencies_and_retains_the_full_static_pad(self):
        with tempfile.TemporaryDirectory() as directory:
            platform = DEFAULT_RECORDING.parents[2]
            root = Path(directory) / "assets"
            copied_platform = root / "versius"
            shutil.copytree(platform, copied_platform)
            shutil.copytree(platform.parent / "phantom_pad", root / "phantom_pad")
            recording = Recording(copied_platform / DEFAULT_RECORDING.relative_to(platform), catalog={})
            # Resolve the copied package: accidentally retaining a preparation
            # checkout or sibling bundle must fail even when it exists locally.
            layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(recording.scene_path))
            self.assertFalse(unresolved)
            self.assertGreater(len(layers), 1)
            for layer in layers:
                self.assertTrue(Path(layer.realPath).resolve().is_relative_to(root), layer.identifier)
                for reference in layer.GetExternalReferences():
                    self.assertFalse(Path(reference).is_absolute(), reference)
            for asset in assets:
                self.assertTrue(Path(asset).resolve().is_relative_to(root), asset)
            for camera in recording.cameras.values():
                recording.asset(camera["ray_directions_cv_path"])
                recording.asset(camera["ray_valid_path"])
                for frame in (recording.first_frame, recording.last_frame):
                    recording.asset(camera["frame_path_pattern"].format(frame=frame))
            stage = Usd.Stage.Open(str(recording.scene_path))
            pad = stage.GetPrimAtPath("/VersiusPreview/Phantom")
            self.assertTrue(pad)
            meshes = [UsdGeom.Mesh(prim) for prim in Usd.PrimRange(pad) if prim.IsA(UsdGeom.Mesh)]
            self.assertEqual(len(meshes), 12)
            self.assertEqual(sum(len(mesh.GetFaceVertexCountsAttr().Get()) for mesh in meshes), 110956)
            for prim in Usd.PrimRange(pad):
                self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI), str(prim.GetPath()))
            # The complete volume accompanies the display surface, even
            # though this kinematic replay deliberately creates no FEM state.
            volume = Usd.Stage.Open(str(recording.asset(recording.manifest["phantom"]["asset"])))
            tets = [prim for prim in volume.Traverse() if prim.GetTypeName() == "TetMesh"]
            self.assertEqual(len(tets), 12)
            self.assertEqual(sum(len(prim.GetAttribute("tetVertexIndices").Get()) for prim in tets), 413405)
            for layer in volume.GetUsedLayers():
                if layer.realPath:
                    self.assertTrue(Path(layer.realPath).resolve().is_relative_to(root), layer.identifier)

    def test_null_viewer_replays_all_eight_links_and_resets_without_moving_pad(self):
        with wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu")):
            viewer = ViewerNull()
            example = Example(viewer, Example.create_parser().parse_args(["--no-camera-overlays"]))
            try:
                self.assertEqual(example.model.body_count, 8)
                self.assertEqual(example.model.particle_count, 0)
                expected = {
                    f"/VersiusPreview/{arm}/{part}"
                    for arm in ("Left", "Right")
                    for part in ("shaft", "wrist", "jaw_1", "jaw_2")
                }
                self.assertEqual(set(example.recording.body_paths), expected)
                self.assertEqual(set(example.model.body_label), expected)
                shape_body = example.model.shape_body.numpy()
                pad_shapes = [index for index, label in enumerate(example.model.shape_label) if "/Phantom/" in label]
                self.assertEqual(len(pad_shapes), 12)
                np.testing.assert_array_equal(shape_body[pad_shapes], -1)
                pad_transforms = example.model.shape_transform.numpy()[pad_shapes].copy()
                frames = np.unique(
                    np.linspace(example.recording.first_frame, example.recording.last_frame, 7, dtype=int)
                )
                poses = []
                for frame in frames:
                    example.seek(frame)
                    example.test_final()
                    poses.append(example.state_0.body_q.numpy().copy())
                    np.testing.assert_array_equal(example.state_0.body_qd.numpy(), 0)
                    np.testing.assert_array_equal(example.model.shape_transform.numpy()[pad_shapes], pad_transforms)
                self.assertGreater(
                    float(np.max(np.linalg.norm(np.asarray(poses)[:, :, :3] - poses[0][:, :3], axis=-1))), 0.001
                )
                labels = {name: index for index, name in enumerate(example.model.body_label)}
                for arm in ("Left", "Right"):
                    for parent, child in (("shaft", "wrist"), ("wrist", "jaw_1"), ("wrist", "jaw_2")):
                        relative = []
                        for pose in poses:
                            a, b = (pose[labels[f"/VersiusPreview/{arm}/{part}"]] for part in (parent, child))
                            relative.append(
                                np.asarray(
                                    wp.transform_multiply(wp.transform_inverse(wp.transform(*a)), wp.transform(*b))
                                )
                            )
                        self.assertGreater(float(np.max(np.ptp(relative, axis=0))), 0.01, f"{arm}/{child} articulation")
                example.playing = False
                last = example.state_0.body_q.numpy().copy()
                example.step()
                np.testing.assert_array_equal(example.state_0.body_q.numpy(), last)
                example.pending_frame = example.recording.first_frame
                example.step()
                self.assertEqual(example.frame_index, example.recording.first_frame)
                np.testing.assert_array_equal(example.state_0.body_q.numpy(), poses[0])
                example.test_final()
                self.assertIsNone(example.overlay)
            finally:
                example.close()
                viewer.close()


if __name__ == "__main__":
    unittest.main()
