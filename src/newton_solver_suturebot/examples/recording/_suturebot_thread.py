# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read an oriented initial thread reference prepared outside the example.

The USD contains a baked visual mesh. Spline samples and rotation-minimizing
frames are retained for future rod initialization; they do not measure material
twist or define a stress-free configuration. No geometry or forces are generated
here, and the reference is valid at exactly one recorded source frame.
"""

import json
from hashlib import sha256

import newton
import numpy as np


class ThreadReference:
    def __init__(self, recording):
        specification = recording.manifest["thread"]
        if (
            specification.get("support") != "initial_source_frame_only"
            or specification.get("dynamics") is not False
            or specification.get("collision") is not False
        ):
            raise ValueError("Thread reference must declare one source frame without dynamics or collision")
        self.source_frame = specification["source_frame"]
        if (
            type(self.source_frame) is not int
            or not recording.first_frame <= self.source_frame <= recording.last_frame
            or not recording.body_valid[self.source_frame]
            or not recording.camera_valid[self.source_frame].all()
        ):
            raise ValueError("Thread reference source frame must have supported tool and four-camera poses")
        timestamp = specification["source_timestamp_ns"]
        if type(timestamp) is not int or timestamp != int(recording.timestamps_ns[self.source_frame]):
            raise ValueError("Thread reference must use the exact recording source timestamp")
        self.usd_path = recording.asset(specification["usd"])
        self.report = json.loads(recording.asset(specification["report"]).read_text())
        if type(self.report.get("schema_version")) is not int or self.report["schema_version"] not in (1, 2):
            raise ValueError("Unsupported prepared thread schema")
        if (
            self.report.get("source_frame") != self.source_frame
            or self.report.get("source_timestamp_ns") != timestamp
            or self.report.get("coordinate_frame") != "recording_world"
            or self.report.get("length_unit") != "m"
        ):
            raise ValueError("Thread report must describe the same source frame in recording-world metres")
        if self.report.get("material_frame", {}).get("physical_twist_status") != "unobserved":
            raise ValueError("Initial thread reference must identify physical material twist as unobserved")
        components = self.report.get("components", [])
        if not components:
            raise ValueError("Thread reference needs at least one prepared component")
        self.components = []
        with np.load(recording.asset(specification["inputs_npz"]), allow_pickle=False) as data:
            for key, expected in (("source_frame", self.source_frame), ("source_timestamp_ns", timestamp)):
                value = data[key]
                if value.shape != () or value.dtype != np.dtype("int64") or int(value) != expected:
                    raise ValueError("Thread arrays must preserve their exact source frame and int64 timestamp")
            for component in components:
                prefix, prim = component["array_prefix"], component["prim_path"]
                if not isinstance(prefix, str) or not prefix or not isinstance(prim, str) or not prim.startswith("/"):
                    raise ValueError("Thread components need an array prefix and an absolute USD prim path")
                arrays = {
                    name: np.asarray(data[f"{prefix}_{name}"]).copy()
                    for name in (
                        "control_points_world_m",
                        "knots_u",
                        "parameter_u",
                        "position_world_m",
                        "reference_arclength_m",
                        "tangent_world",
                        "orientation_xyzw",
                        "reference_roll_control_rad",
                        "reference_roll_rad",
                    )
                }
                self._validate_component(arrays)
                attachment = component.get("needle_attachment")
                if attachment is not None:
                    self._validate_attachment(recording, arrays, attachment, data, prefix)
                self.components.append(
                    {"prim_path": prim, "array_prefix": prefix, "needle_attachment": attachment, **arrays}
                )
        for key in ("prim_path", "array_prefix"):
            if len({component[key] for component in self.components}) != len(self.components):
                raise ValueError("Thread component paths and array prefixes must be unique")
        self.has_needle_attachment = any(component["needle_attachment"] is not None for component in self.components)
        if self.has_needle_attachment != (self.report["schema_version"] == 2):
            raise ValueError("Thread schema 2 must retain at least one hard needle attachment; schema 1 is unanchored")

    def _validate_attachment(self, recording, arrays, attachment, data, prefix):
        """Check the prepared hard endpoint against its current rigid-needle input."""
        from newton_solver_suturebot.examples.recording._suturebot_needle import pose_matrix

        needle = recording.manifest.get("needle")
        if (
            not isinstance(attachment, dict)
            or needle is None
            or attachment.get("constraint") != "hard_position"
            or attachment.get("orientation_constraint") != "free"
            or attachment.get("endpoint") not in ("start", "end")
            or type(attachment.get("source_frame")) is not int
            or attachment["source_frame"] != self.source_frame
            or needle.get("source", {}).get("initial_frame") != self.source_frame
            or attachment.get("body_path") != needle.get("body_path")
        ):
            raise ValueError(
                "Thread needle attachment must identify its initial frame, rigid body and free orientation"
            )
        endpoint = 0 if attachment["endpoint"] == "start" else -1
        local = self._finite_vector(attachment.get("body_local_position_m"), 3)
        world = self._finite_vector(attachment.get("world_position_m"), 3)
        declared_pose = self._needle_pose(attachment.get("body_world_pose_xyz_xyzw"))
        initial_pose = self._needle_pose(needle.get("initial_pose_world_xyz_xyzw"))
        initial = pose_matrix(initial_pose)
        if not np.allclose(pose_matrix(declared_pose), initial, atol=1.0e-8, rtol=0):
            raise ValueError("Thread needle attachment uses a stale initial needle pose")
        transformed = initial[:3, :3] @ local + initial[:3, 3]
        gap = attachment.get("endpoint_gap_m")
        if (
            type(gap) not in (int, float)
            or not np.isfinite(gap)
            or not 0 <= gap <= 1.0e-9
            or np.linalg.norm(transformed - world) > 1.0e-9
            or np.linalg.norm(arrays["position_world_m"][endpoint] - world) > 1.0e-9
            or np.linalg.norm(arrays["control_points_world_m"][endpoint] - world) > 1.0e-9
        ):
            raise ValueError("Thread needle attachment endpoint is detached from its declared needle anchor")
        for field, asset in (("needle_usd_sha256", "usd"), ("needle_inputs_sha256", "inputs_npz")):
            if attachment.get(field) != sha256(recording.asset(needle[asset]).read_bytes()).hexdigest():
                raise ValueError("Thread needle attachment refers to stale needle assets")
        expected_fields = {
            "attachment_endpoint_index": endpoint,
            "attachment_source_frame": self.source_frame,
            "attachment_body_local_position_m": local,
            "attachment_world_position_m": world,
            "attachment_body_world_pose_xyz_xyzw": declared_pose,
        }
        for name, expected in expected_fields.items():
            value = data.get(f"{prefix}_{name}")
            if value is None or value.shape != np.shape(expected) or not np.array_equal(value, expected):
                raise ValueError("Thread needle attachment arrays differ from the prepared metadata")
            if name in ("attachment_endpoint_index", "attachment_source_frame"):
                if value.dtype != np.dtype("int64"):
                    raise ValueError("Thread needle attachment indices must be exact int64 scalars")
            elif not np.issubdtype(value.dtype, np.floating):
                raise ValueError("Thread needle attachment positions and pose must be floating-point arrays")
        encoded = data.get(f"{prefix}_attachment_json")
        if (
            encoded is None
            or encoded.shape != ()
            or encoded.dtype.kind != "U"
            or json.loads(str(encoded)) != attachment
        ):
            raise ValueError("Thread needle attachment JSON differs from the prepared metadata")

    @staticmethod
    def _finite_vector(value, size):
        vector = np.asarray(value, dtype=float)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise ValueError("Thread needle attachment requires finite position and pose vectors")
        return vector

    @classmethod
    def _needle_pose(cls, value):
        pose = cls._finite_vector(value, 7)
        if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=2.0e-5, rtol=0):
            raise ValueError("Thread needle attachment requires a unit xyzw needle orientation")
        return pose

    def validate_needle_pose(self, frame, body_path, pose):
        """Check the actual post-seek rigid state; the static thread never moves."""
        if not self.has_needle_attachment:
            return
        from newton_solver_suturebot.examples.recording._suturebot_needle import pose_matrix

        if frame != self.source_frame or pose is None:
            raise ValueError("Anchored thread reference requires its needle at the exact reference frame")
        matrix = pose_matrix(self._needle_pose(pose))
        for component in self.components:
            attachment = component["needle_attachment"]
            if attachment is None:
                continue
            local = np.asarray(attachment["body_local_position_m"])
            actual_anchor = matrix[:3, :3] @ local + matrix[:3, 3]
            # Newton rigid state is float32. This tolerance covers its rounding,
            # not a geometric snap or a permitted reconstruction displacement.
            if (
                body_path != attachment["body_path"]
                or np.linalg.norm(actual_anchor - attachment["world_position_m"]) > 2.0e-7
            ):
                raise ValueError("Thread endpoint is detached from the actual runtime needle anchor")

    @staticmethod
    def _validate_component(arrays):
        if not all(np.issubdtype(value.dtype, np.floating) and np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Prepared thread geometry and orientations must be finite floating-point arrays")
        points, controls = arrays["position_world_m"], arrays["control_points_world_m"]
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("Thread component positions must have shape (N, 3), N >= 2")
        if controls.ndim != 2 or controls.shape[1] != 3 or len(controls) < 4:
            raise ValueError("Thread cubic spline controls must have shape (K, 3), K >= 4")
        knots, parameters = arrays["knots_u"], arrays["parameter_u"]
        if (
            knots.shape != (len(controls) + 4,)
            or np.any(np.diff(knots) < 0)
            or knots[3] >= knots[-4]
            or not np.allclose(knots[:4], knots[3], atol=1.0e-12, rtol=0)
            or not np.allclose(knots[-4:], knots[-4], atol=1.0e-12, rtol=0)
            or parameters.shape != (len(points),)
            or np.any(np.diff(parameters) <= 0)
            or not np.allclose(parameters[[0, -1]], knots[[3, -4]], atol=1.0e-12, rtol=0)
        ):
            raise ValueError("Thread spline samples must span the full nonempty clamped cubic domain")
        if not np.allclose(points[[0, -1]], controls[[0, -1]], atol=1.0e-8, rtol=0):
            raise ValueError("Thread clamped spline endpoints must match their endpoint control points")
        arclength = arrays["reference_arclength_m"]
        if (
            arclength.shape != (len(points),)
            or arclength[0] != 0.0
            or np.any(np.diff(arclength) <= 0)
            or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1.0e-10)
        ):
            raise ValueError("Thread samples need increasing component arc distances and nonzero segments")
        tangents, orientations = arrays["tangent_world"], arrays["orientation_xyzw"]
        if (
            tangents.shape != points.shape
            or orientations.shape != (len(points), 4)
            or not np.allclose(np.linalg.norm(tangents, axis=1), 1.0, atol=2.0e-5, rtol=0)
            or not np.allclose(np.linalg.norm(orientations, axis=1), 1.0, atol=2.0e-5, rtol=0)
        ):
            raise ValueError("Thread tangents and xyzw orientation quaternions must be unit vectors")
        if np.any(np.sum(orientations[:-1] * orientations[1:], axis=1) < -1.0e-12):
            raise ValueError("Thread orientation quaternion signs must remain continuous within each component")
        x, y, z, w = orientations.T
        local_z = np.column_stack((2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)))
        if not np.allclose(local_z, tangents, atol=3.0e-5, rtol=0):
            raise ValueError("Thread orientation local +Z must follow its prepared spline tangent")
        if arrays["reference_roll_control_rad"].shape != (len(controls),) or arrays["reference_roll_rad"].shape != (
            len(points),
        ):
            raise ValueError("Thread reference-roll controls and samples must match the prepared spline")

    def add_to_builder(self, builder):
        """Import only world-fixed visual meshes, rejecting accidental physics."""
        if self.has_needle_attachment:
            from pxr import Gf, Usd, UsdGeom

            stage = Usd.Stage.Open(str(self.usd_path))
            if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or not np.isclose(
                UsdGeom.GetStageMetersPerUnit(stage), 1.0, atol=1.0e-12, rtol=0
            ):
                raise ValueError("Anchored thread USD must use recording-world Z-up metres")
            cache = UsdGeom.XformCache()
            for component in self.components:
                attachment = component["needle_attachment"]
                if attachment is not None:
                    prim = stage.GetPrimAtPath(component["prim_path"])
                    value = prim.GetAttribute("suturebot:needleAttachment").Get() if prim else None
                    if not isinstance(value, str) or json.loads(value) != attachment:
                        raise ValueError("Thread USD needle attachment differs from the prepared metadata")
                    points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
                        raise ValueError("Anchored thread USD is missing its exported end-cap centers")
                    # The offline tube exporter appends the two cap centers.
                    index = -2 if attachment["endpoint"] == "start" else -1
                    cap = np.asarray(cache.GetLocalToWorldTransform(prim).Transform(Gf.Vec3d(*points[index])))
                    if not np.isfinite(cap).all() or np.linalg.norm(cap - attachment["world_position_m"]) > 2.0e-7:
                        raise ValueError("Thread USD end-cap center is detached from its needle anchor")
        counts = builder.body_count, builder.joint_count, builder.particle_count
        first_shape = builder.shape_count
        builder.add_usd(str(self.usd_path), collapse_fixed_joints=False)
        if counts != (builder.body_count, builder.joint_count, builder.particle_count):
            raise ValueError("Thread reference USD must not add bodies, joints or particles")
        shapes = range(first_shape, builder.shape_count)
        if not shapes:
            raise ValueError("Thread reference USD contains no importable visual meshes")
        for shape in shapes:
            if (
                builder.shape_body[shape] != -1
                or builder.shape_flags[shape] != int(newton.ShapeFlags.VISIBLE)
                or builder.shape_type[shape] != newton.GeoType.MESH
            ):
                raise ValueError("Thread reference USD must contain only visible non-colliding world-space meshes")
        if set(builder.shape_label[first_shape:]) != {component["prim_path"] for component in self.components}:
            raise ValueError("Thread USD visual mesh paths differ from the prepared components")
