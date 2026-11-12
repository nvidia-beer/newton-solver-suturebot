# SPDX-License-Identifier: Apache-2.0
"""Static tool layout from the bundled SutureBot needle-throw recording.

The registered port positions and one recorded tool pose inform initialization
only. Simulation subsequently moves articulated bodies with joint actuators;
this module does not replay recorded motion. The source alignment is an
estimate, not a measured hand-eye calibration.
"""

import json
from pathlib import Path

import numpy as np
import warp as wp

from .rcm_grasp import _arm_pose

_SAMPLE = "tissue3_215300"
_FRAME = 4
_RECORDING_DIR = Path(__file__).resolve().parents[1] / "examples/assets/dvrk/recordings" / _SAMPLE
_PARKED_NOMINAL_INSERTION_M = 0.18
# The source wrist pitch (-1.17458 rad) exceeds the surrogate's -1.1 limit.
# Preserve its approach and use a nearby, bounded -1.0 rad wrist posture.
_PARKED_ARM_POSITIONS = (
    3.6734510554093775e-6,
    1.4187881924088385e-6,
    -0.012673657162961356,
    3.396699872536789e-6,
    -1.0,
    -0.2649668686629805,
)


def _rigid_transform(value):
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Source layout requires finite rigid transforms with proper rotations")
    return matrix


def _pose(matrix):
    matrix = _rigid_transform(matrix)
    rotation = np.asarray(wp.quat_from_matrix(wp.mat33(matrix[:3, :3])), dtype=float)
    rotation /= np.linalg.norm(rotation)
    return np.r_[matrix[:3, 3], rotation]


def source_layout_asset_paths():
    """Physical initialization inputs, shared with scaling evidence hashing."""
    return tuple(_RECORDING_DIR / name for name in ("recording.json", "motion.npz", "point_labels.json"))


def load_source_stitch():
    """Return the recorded insertion/exit labels in the native FEM frame.

    The labels are material points on the original region's exterior triangles.
    Their world positions include the recording's rigid phantom placement, so
    apply its inverse before using them as simulation targets. Labels are
    resolved by their semantic names; file order must never reverse the stitch.
    This imports neither recorded motion nor a CUDA context. The source camera
    registration is an estimate, not an independent metric calibration.
    """
    recording_path, _, labels_path = source_layout_asset_paths()
    recording = json.loads(recording_path.read_text())
    labels = json.loads(labels_path.read_text())
    native_from_world = np.linalg.inv(_rigid_transform(recording["phantom"]["T_world_from_native"]))
    points = labels.get("points", [])
    if len(points) != 2 or {point.get("label") for point in points} != {"insertion", "exit"}:
        raise ValueError("Source stitch requires exactly one insertion and one exit material label")
    known_regions = {region["id"]: region for region in recording["phantom"]["regions"]}
    native_points = {}
    bindings = {}
    for point in points:
        name = point["label"]
        region = point.get("region_id")
        if type(region) is not int or region <= 0 or region not in known_regions:
            raise ValueError("Source stitch material labels require a recorded positive region ID")
        vertices = np.asarray(point.get("reference_vertex_indices", []))
        triangle = point.get("triangle_index")
        weights = np.asarray(point.get("barycentric", []), dtype=float)
        if (
            type(triangle) is not int
            or triangle < 0
            or triangle >= known_regions[region]["exterior_triangles"]
            or vertices.shape != (3,)
            or not np.issubdtype(vertices.dtype, np.integer)
            or np.any(vertices < 0)
            or np.any(vertices >= known_regions[region]["vertices"])
            or len(np.unique(vertices)) != 3
            or weights.shape != (3,)
            or not np.isfinite(weights).all()
            or np.any(weights < -1e-7)
            or np.any(weights > 1 + 1e-7)
            or not np.isclose(weights.sum(), 1, atol=1e-6, rtol=0)
        ):
            raise ValueError("Source stitch material labels require valid triangle and barycentric bindings")
        position = np.asarray(point.get("position_world_m", []), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Source stitch material labels require finite three-dimensional positions")
        native_points[name] = (native_from_world @ np.r_[position, 1])[:3]
        bindings[name] = {
            "region_id": region,
            "triangle_index": triangle,
            "reference_vertex_indices": vertices.tolist(),
            "barycentric": weights.tolist(),
        }
    if bindings["insertion"]["region_id"] != bindings["exit"]["region_id"]:
        raise ValueError("Source stitch insertion and exit must belong to the same FEM region")
    if np.linalg.norm(native_points["insertion"] - native_points["exit"]) <= 1e-9:
        raise ValueError("Source stitch insertion and exit must be distinct material points")
    return {
        "region_id": bindings["insertion"]["region_id"],
        "entry": native_points["insertion"],
        "exit": native_points["exit"],
        "provenance": {
            "sample_id": _SAMPLE,
            "source_episode": recording["source_episode"],
            "point_source": "point_labels.json",
            "annotation_frame": labels["annotation_frame"],
            "annotation_semantics": labels["annotation_semantics"],
            "coordinate_frame": "native FEM phantom, metres, Z up",
            "material_bindings": bindings,
            "absolute_metric_accuracy_validated": bool(recording["alignment"]["absolute_metric_accuracy_validated"]),
            "pad_alignment_accepted": bool(recording["alignment"].get("accepted", False)),
            "limitations": list(labels.get("limitations", [])),
        },
    }


def load_source_layout():
    """Return independent static poses in the native FEM phantom frame.

    ``trocar_poses`` contains PSM1 then PSM2 fixed world poses (xyz, xyzw).
    Their positions are registered SUJ/fixture ports; their initial axes come
    from each roll body in valid frame 4. PSM1's roll zero is shifted by pi to
    keep the insertion arc within the +/-pi roll limits. ``parked_pose`` is a
    nearby bounded PSM2 posture because its recorded wrist pitch exceeds the
    surrogate's range. Registered ``body_transforms`` are used instead of the
    pre-registration per-arm motion arrays.
    """
    recording_path, motion_path, _ = source_layout_asset_paths()
    recording = json.loads(recording_path.read_text())
    native_from_world = np.linalg.inv(_rigid_transform(recording["phantom"]["T_world_from_native"]))
    fixture = recording["rig_geometry"]["fixture"]
    ports_world = np.asarray(fixture["measured_suj_pivots_world_m"], dtype=float)
    if ports_world.shape != (2, 3) or not np.isfinite(ports_world).all():
        raise ValueError("Source layout requires two finite registered trocar positions")

    with np.load(motion_path, allow_pickle=False) as motion:
        paths = [str(path) for path in motion["body_paths"]]
        transforms = motion["body_transforms"][_FRAME]
        valid = motion["body_valid_per_body"][_FRAME]

    def body_pose(arm, body):
        path = f"/SutureBotReplay/{arm}/{body}"
        index = paths.index(path)
        if not valid[index]:
            raise ValueError(f"Source layout frame {_FRAME} does not support {path}")
        return native_from_world @ _rigid_transform(transforms[index])

    trocar_poses = []
    for index, arm in enumerate(("PSM1", "PSM2")):
        transform = body_pose(arm, "roll")
        if index == 0:
            transform[:3, :3] = transform[:3, :3] @ np.diag([-1, -1, 1])
        transform[:3, 3] = (native_from_world @ np.r_[ports_world[index], 1])[:3]
        trocar_poses.append(_pose(transform))
    parked_pose = np.asarray(
        wp.transform_multiply(
            wp.transform(*trocar_poses[1]),
            _arm_pose(_PARKED_ARM_POSITIONS, _PARKED_NOMINAL_INSERTION_M),
        ),
        dtype=float,
    )
    parked_pose[3:] /= np.linalg.norm(parked_pose[3:])

    return {
        "trocar_poses": np.asarray(trocar_poses),
        "parked_pose": parked_pose,
        "provenance": {
            "status": "source_recording_registered_layout_estimate",
            "sample_id": _SAMPLE,
            "source_episode": recording["source_episode"],
            "reference_frame": _FRAME,
            "coordinate_frame": "native FEM phantom, metres, Z up; poses xyz and quaternion xyzw",
            "port_source": "recording.json:rig_geometry.fixture.measured_suj_pivots_world_m",
            "orientation_source": "motion.npz:body_transforms, PSM1/roll and PSM2/roll",
            "psm1_roll_zero_offset_rad": float(np.pi),
            "parked_pose_source": "bounded surrogate FK near frame 4 PSM2/yaw; wrist pitch adjusted to -1.0 rad",
            "source_parked_pose": _pose(body_pose("PSM2", "yaw")).tolist(),
            "parked_arm_positions": list(_PARKED_ARM_POSITIONS),
            "parked_nominal_insertion_m": _PARKED_NOMINAL_INSERTION_M,
            "absolute_metric_accuracy_validated": bool(recording["alignment"]["absolute_metric_accuracy_validated"]),
            "pad_alignment_accepted": bool(recording["alignment"].get("accepted", False)),
            "port_mid_surface_uncertainty_m": fixture["port_mid_surface_uncertainty_m"],
            "static_layout_only": True,
        },
    }
