# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Initial thread reference support, oriented data and visual-only USD replay."""

import json
import os
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

import newton
import numpy as np
import warp as wp
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from newton_solver_suturebot.examples._suturebot_needle import NeedleReplay, add_needle_to_builder, pose_matrix
from newton_solver_suturebot.examples._suturebot_recording import CAMERAS, Recording
from newton_solver_suturebot.examples._suturebot_thread import ThreadReference


class ThreadReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="suturebot-thread-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.timestamps = np.array([1745891580486691183, 1745891580516691183, 1745891580556691183], dtype=np.int64)
        poses = np.zeros((3, 1, 7))
        poses[:, 0, 0] = np.arange(3) * 0.001
        poses[..., 6] = 1
        np.savez(
            self.root / "motion.npz",
            timestamps_ns=self.timestamps,
            time_s=(self.timestamps - self.timestamps[0]) * 1.0e-9,
            body_poses=poses,
            body_valid=np.ones(3, bool),
        )
        np.savez(
            self.root / "cameras.npz",
            timestamps_ns=self.timestamps,
            camera_poses_cv=np.broadcast_to(np.eye(4), (3, 4, 4, 4)),
            camera_valid=np.ones((3, 4), bool),
        )
        self.manifest = {
            "schema_version": 1,
            "sample_id": "synthetic_thread_reference",
            "region_id": 1,
            "scene_usd": "scene.usda",
            "motion_npz": "motion.npz",
            "camera_poses_npz": "cameras.npz",
            "body_paths": ["/Rig/Tool"],
            "camera_order": list(CAMERAS),
            "cameras": {name: {"frame_count": 3} for name in CAMERAS},
            "timing": {"nominal_fps": 30, "frame_count": 3},
            "thread": {
                "usd": "thread.usda",
                "report": "thread.json",
                "inputs_npz": "thread_inputs.npz",
                "source_frame": 1,
                "source_timestamp_ns": int(self.timestamps[1]),
                "status": "initial_geometry_estimate",
                "dynamics": False,
                "collision": False,
                "support": "initial_source_frame_only",
            },
        }
        self.report = {
            "schema_version": 1,
            "source_frame": 1,
            "source_timestamp_ns": int(self.timestamps[1]),
            "coordinate_frame": "recording_world",
            "length_unit": "m",
            "material_frame": {"physical_twist_status": "unobserved"},
            "components": [{"id": 0, "prim_path": "/SutureBotThread/Component_000", "array_prefix": "c000"}],
        }
        positions = np.column_stack((np.zeros(4), np.zeros(4), np.linspace(0, 0.03, 4)))
        self.arrays = {
            "source_frame": np.array(1, np.int64),
            "source_timestamp_ns": self.timestamps[1],
            "c000_control_points_world_m": positions.copy(),
            "c000_knots_u": np.array([0, 0, 0, 0, 1, 1, 1, 1], float),
            "c000_parameter_u": np.linspace(0, 1, 4),
            "c000_position_world_m": positions.copy(),
            "c000_reference_arclength_m": np.linspace(0, 0.03, 4),
            "c000_tangent_world": np.tile([0.0, 0.0, 1.0], (4, 1)),
            "c000_orientation_xyzw": np.tile([0.0, 0.0, 0.0, 1.0], (4, 1)),
            "c000_reference_roll_control_rad": np.zeros(4),
            "c000_reference_roll_rad": np.zeros(4),
        }
        stage = Usd.Stage.CreateNew(str(self.root / "scene.usda"))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, "/Rig")
        stage.SetDefaultPrim(root.GetPrim())
        tool = UsdGeom.Xform.Define(stage, "/Rig/Tool")
        UsdPhysics.RigidBodyAPI.Apply(tool.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(tool.GetPrim())
        mass.CreateMassAttr(0.1)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0e-5))
        UsdGeom.Sphere.Define(stage, "/Rig/Tool/Visual").CreateRadiusAttr(0.001)
        stage.GetRootLayer().Save()
        self._write_thread_usd()
        self._save()

    def _write_thread_usd(self, *, collision=False, rigid=False, filename="thread.usda"):
        stage = Usd.Stage.CreateNew(str(self.root / filename))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, "/SutureBotThread")
        stage.SetDefaultPrim(root.GetPrim())
        if rigid:
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
            mass.CreateMassAttr(0.001)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0e-8))
        mesh = UsdGeom.Mesh.Define(stage, "/SutureBotThread/Component_000")
        mesh.CreatePointsAttr([(-0.001, 0, 0), (0.001, 0, 0), (0.001, 0, 0.03), (-0.001, 0, 0.03)])
        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        if collision:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        stage.GetRootLayer().Save()

    def _save(self):
        (self.root / "recording.json").write_text(json.dumps(self.manifest))
        (self.root / "thread.json").write_text(json.dumps(self.report))
        np.savez(self.root / "thread_inputs.npz", **self.arrays)

    def _reference(self):
        return ThreadReference(Recording(self.root / "recording.json"))

    def _add_anchor(self, endpoint="start"):
        frame = 0
        pose = [0.01, 0.02, 0.03, 0, np.sqrt(0.5), 0, np.sqrt(0.5)]
        local = np.array([0.002, 0, 0])
        transform = pose_matrix(pose)
        world = transform[:3, :3] @ local + transform[:3, 3]
        stage = Usd.Stage.CreateNew(str(self.root / "needle.usda"))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, "/Needle")
        stage.SetDefaultPrim(root.GetPrim())
        body = UsdGeom.Xform.Define(stage, "/Needle/Body")
        body.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(0.0003)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-8))
        UsdGeom.Sphere.Define(stage, "/Needle/Body/Visual").CreateRadiusAttr(0.0005)
        stage.GetRootLayer().Save()
        np.savez(
            self.root / "needle_inputs.npz",
            timestamps_ns=self.timestamps,
            jaw_opening_rad=np.tile([-0.1, 0.3], (3, 1)),
            centreline_local_m=np.array([[0.002, 0, 0], [0, 0, 0], [-0.002, 0, 0]]),
        )
        self.manifest["needle"] = {
            "usd": "needle.usda",
            "inputs_npz": "needle_inputs.npz",
            "source": {"initial_frame": frame},
            "body_path": "/Needle/Body",
            "initial_pose_world_xyz_xyzw": pose,
            "initial_held_arm": "psm1",
            "grasp": {
                "close_angle_rad": 0.05,
                "open_angle_rad": 0.18,
                "capture_distance_m": 0.002,
                "tool_body_paths": {arm: "/Rig/Tool" for arm in ("psm1", "psm2")},
                "tool_point_local_m": {arm: [0, 0, 0] for arm in ("psm1", "psm2")},
            },
            "simulation": {"maximum_step_s": 1 / 240, "gravity_m_s2": [0, 0, -9.81]},
        }
        self.manifest["thread"].update(source_frame=frame, source_timestamp_ns=int(self.timestamps[frame]))
        self.report.update(schema_version=2, source_frame=frame, source_timestamp_ns=int(self.timestamps[frame]))
        self.arrays.update(source_frame=np.array(frame, np.int64), source_timestamp_ns=self.timestamps[frame])
        index = 0 if endpoint == "start" else -1
        shift = world - self.arrays["c000_position_world_m"][index]
        for key in ("c000_position_world_m", "c000_control_points_world_m"):
            self.arrays[key] += shift
        attachment = {
            "endpoint": endpoint,
            "source_frame": frame,
            "constraint": "hard_position",
            "orientation_constraint": "free",
            "body_path": "/Needle/Body",
            "body_local_position_m": local.tolist(),
            "world_position_m": world.tolist(),
            "body_world_pose_xyz_xyzw": pose.copy(),
            "needle_usd_sha256": sha256((self.root / "needle.usda").read_bytes()).hexdigest(),
            "needle_inputs_sha256": sha256((self.root / "needle_inputs.npz").read_bytes()).hexdigest(),
            "endpoint_gap_m": 0.0,
        }
        self.report["components"][0]["needle_attachment"] = attachment
        self.arrays.update(
            c000_attachment_endpoint_index=np.array(index, np.int64),
            c000_attachment_source_frame=np.array(frame, np.int64),
            c000_attachment_body_local_position_m=local,
            c000_attachment_world_position_m=world,
            c000_attachment_body_world_pose_xyz_xyzw=np.array(pose),
            c000_attachment_json=np.array(json.dumps(attachment, sort_keys=True, separators=(",", ":"))),
        )
        stage = Usd.Stage.Open(str(self.root / "thread.usda"))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/SutureBotThread/Component_000"))
        # Match the offline tube convention: sampled rings, then two cap centers.
        points = self.arrays["c000_position_world_m"]
        offsets = 0.0001 * np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        vertices = np.vstack([(points[:, None] + offsets).reshape(-1, 3), points[[0, -1]]])
        faces = []
        for ring in range(len(points) - 1):
            for side in range(4):
                a, b = ring * 4 + side, ring * 4 + (side + 1) % 4
                faces.extend([(a, b, b + 4), (a, b + 4, a + 4)])
        for side in range(4):
            following = (side + 1) % 4
            faces.extend([(16, following, side), (17, 12 + side, 12 + following)])
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(np.asarray(faces).ravel().tolist())
        mesh.GetPrim().CreateAttribute("suturebot:needleAttachment", Sdf.ValueTypeNames.String).Set(
            str(self.arrays["c000_attachment_json"])
        )
        stage.GetRootLayer().Save()
        self._save()
        return attachment

    def test_oriented_samples_and_original_source_clock_are_preserved(self):
        reference = self._reference()
        self.assertEqual(reference.source_frame, 1)
        for name in ("position_world_m", "tangent_world", "orientation_xyzw", "reference_roll_control_rad"):
            np.testing.assert_array_equal(reference.components[0][name], self.arrays[f"c000_{name}"])

    def test_legacy_partial_reference_does_not_claim_an_attachment(self):
        reference = self._reference()
        self.assertFalse(reference.has_needle_attachment)
        reference.validate_needle_pose(reference.source_frame, None, None)

    def test_attached_schema_cannot_silently_drop_its_anchor(self):
        self._add_anchor()
        self.report["components"][0].pop("needle_attachment")
        self._save()
        with self.assertRaisesRegex(ValueError, "schema 2 must retain"):
            self._reference()

    def test_alias_sample_id_does_not_invalidate_identical_anchor_assets(self):
        self._add_anchor()
        self.report["sample_id"] = "canonical_recording"
        self.manifest["sample_id"] = "canonical_recording_mesh_preview"
        self._save()
        self.assertTrue(self._reference().has_needle_attachment)

    def test_start_and_end_anchors_use_rotated_body_local_point(self):
        for endpoint in ("start", "end"):
            with self.subTest(endpoint=endpoint):
                attachment = self._add_anchor(endpoint)
                reference = self._reference()
                self.assertTrue(reference.has_needle_attachment)
                reference.validate_needle_pose(0, attachment["body_path"], attachment["body_world_pose_xyz_xyzw"])
                reference.add_to_builder(newton.ModelBuilder())

    def test_detached_control_and_sample_endpoint_is_rejected(self):
        self._add_anchor()
        for key in ("c000_position_world_m", "c000_control_points_world_m"):
            self.arrays[key][0, 0] += 0.0001
        self._save()
        with self.assertRaisesRegex(ValueError, "detached.*declared needle anchor"):
            self._reference()

    def test_stale_initial_pose_and_needle_geometry_are_rejected(self):
        self._add_anchor()
        self.manifest["needle"]["initial_pose_world_xyz_xyzw"][0] += 0.001
        self._save()
        with self.assertRaisesRegex(ValueError, "stale initial needle pose"):
            self._reference()
        self.manifest["needle"]["initial_pose_world_xyz_xyzw"][0] -= 0.001
        self._save()
        with (self.root / "needle.usda").open("a") as stream:
            stream.write("\n# changed needle asset\n")
        with self.assertRaisesRegex(ValueError, "stale needle assets"):
            self._reference()

    def test_attachment_mirrors_cannot_silently_disagree(self):
        self._add_anchor()
        original = self.arrays["c000_attachment_body_local_position_m"].copy()
        self.arrays["c000_attachment_body_local_position_m"][0] += 0.001
        self._save()
        with self.assertRaisesRegex(ValueError, "arrays differ"):
            self._reference()
        self.arrays["c000_attachment_body_local_position_m"] = original
        self._save()
        reference = self._reference()
        stage = Usd.Stage.Open(str(self.root / "thread.usda"))
        stage.GetPrimAtPath("/SutureBotThread/Component_000").GetAttribute("suturebot:needleAttachment").Set("{}")
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "USD needle attachment differs"):
            reference.add_to_builder(newton.ModelBuilder())

    def test_stale_baked_usd_endpoint_is_rejected_even_with_correct_metadata(self):
        self._add_anchor()
        reference = self._reference()
        stage = Usd.Stage.Open(str(self.root / "thread.usda"))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/SutureBotThread/Component_000"))
        points = mesh.GetPointsAttr().Get()
        points[-2] = points[-2] + Gf.Vec3f(0.001, 0, 0)
        mesh.GetPointsAttr().Set(points)
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "USD end-cap center is detached"):
            reference.add_to_builder(newton.ModelBuilder())

    def test_actual_runtime_anchor_rejects_missing_shifted_or_wrong_body(self):
        attachment = self._add_anchor()
        reference = self._reference()
        pose = np.array(attachment["body_world_pose_xyz_xyzw"])
        with self.assertRaisesRegex(ValueError, "requires its needle"):
            reference.validate_needle_pose(0, None, None)
        with self.assertRaisesRegex(ValueError, "exact reference frame"):
            reference.validate_needle_pose(1, attachment["body_path"], pose)
        pose[0] += 0.0001
        with self.assertRaisesRegex(ValueError, "actual runtime needle anchor"):
            reference.validate_needle_pose(0, attachment["body_path"], pose)
        pose[0] -= 0.0001
        with self.assertRaisesRegex(ValueError, "actual runtime needle anchor"):
            reference.validate_needle_pose(0, "/WrongNeedle", pose)

    def test_initial_held_runtime_preserves_anchor_without_grasp_translation(self):
        attachment = self._add_anchor()
        recording = Recording(self.root / "recording.json")
        reference = ThreadReference(recording)
        with wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu")):
            builder = newton.ModelBuilder()
            add_needle_to_builder(builder, recording)
            model = builder.finalize()
            state = model.state()
            needle = NeedleReplay(recording, model, state, backend="cpu")
            actual = needle.seek(0, state)
            reference.validate_needle_pose(0, attachment["body_path"], actual)
            np.testing.assert_allclose(actual[:3], attachment["body_world_pose_xyz_xyzw"][:3], atol=2e-9, rtol=0)
            self.assertEqual([event["event"] for event in needle.events], ["initial_grasp"])
            # A later tool-driven state cannot be substituted for the reference.
            moved = needle.seek(1, state)
            with self.assertRaisesRegex(ValueError, "actual runtime needle anchor"):
                reference.validate_needle_pose(0, attachment["body_path"], moved)

    def test_stale_or_unsupported_source_frame_is_rejected(self):
        self.arrays["source_timestamp_ns"] += np.int64(1)
        self._save()
        with self.assertRaisesRegex(ValueError, "exact source frame"):
            self._reference()
        self.arrays["source_timestamp_ns"] = self.timestamps[1]
        self.manifest["thread"]["source_frame"] = 3
        self._save()
        with self.assertRaisesRegex(ValueError, "source frame must have supported"):
            self._reference()

    def test_invalid_oriented_arrays_are_not_silently_repaired(self):
        for key, value, message in (
            ("c000_position_world_m", np.zeros((4, 3)), "clamped spline endpoints"),
            ("c000_position_world_m", self.arrays["c000_position_world_m"][[0, 0, 2, 3]], "nonzero segments"),
            ("c000_orientation_xyzw", np.tile([0.0, 1.0, 0.0, 0.0], (4, 1)), "local \\+Z"),
            ("c000_tangent_world", np.tile([0.0, 0.0, 2.0], (4, 1)), "unit vectors"),
            ("c000_knots_u", np.zeros(8), "cubic domain"),
            ("c000_reference_roll_control_rad", np.zeros(3), "reference-roll controls"),
            ("c000_position_world_m", np.full((4, 3), np.nan), "finite floating-point"),
        ):
            with self.subTest(key=key):
                original = self.arrays[key]
                self.arrays[key] = value
                self._save()
                with self.assertRaisesRegex(ValueError, message):
                    self._reference()
                self.arrays[key] = original

    def test_quaternion_sign_continuity_preserves_rotation_without_repair(self):
        # Global q -> -q preserves both rotations and the continuous sign choice.
        self.arrays["c000_orientation_xyzw"] *= -1
        self._save()
        reference = self._reference()
        np.testing.assert_array_equal(reference.components[0]["orientation_xyzw"], self.arrays["c000_orientation_xyzw"])
        # A single sign flip also preserves each rotation, but breaks the stored
        # continuous orientation representation needed by future interpolation.
        self.arrays["c000_orientation_xyzw"][2] *= -1
        self._save()
        with self.assertRaisesRegex(ValueError, "quaternion signs must remain continuous"):
            self._reference()

    def test_unclamped_or_partially_sampled_spline_domain_is_rejected(self):
        for key, value, message in (
            ("c000_knots_u", np.array([-0.3, -0.2, -0.1, 0, 1, 1.1, 1.2, 1.3]), "clamped cubic domain"),
            ("c000_parameter_u", np.linspace(0.1, 0.9, 4), "clamped cubic domain"),
            ("c000_control_points_world_m", self.arrays["c000_control_points_world_m"] + 0.001, "spline endpoints"),
        ):
            with self.subTest(key=key):
                original = self.arrays[key]
                self.arrays[key] = value
                self._save()
                with self.assertRaisesRegex(ValueError, message):
                    self._reference()
                self.arrays[key] = original

    def test_thread_usd_cannot_introduce_collision_or_rigid_bodies(self):
        reference = self._reference()
        builder = newton.ModelBuilder()
        reference.add_to_builder(builder)
        self.assertEqual((builder.shape_count, builder.body_count, builder.joint_count), (1, 0, 0))
        self.assertEqual(builder.shape_flags, [int(newton.ShapeFlags.VISIBLE)])
        for options, message in (({"collision": True}, "non-colliding"), ({"rigid": True}, "bodies, joints")):
            with self.subTest(options=options):
                filename = f"{next(iter(options))}_thread.usda"
                self._write_thread_usd(**options, filename=filename)
                reference.usd_path = self.root / filename
                with self.assertRaisesRegex(ValueError, message):
                    reference.add_to_builder(newton.ModelBuilder())


if __name__ == "__main__":
    unittest.main()
