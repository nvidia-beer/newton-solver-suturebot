# SPDX-License-Identifier: Apache-2.0
"""Display-only camera framing and accepted-snapshot updates for physical grasp scenes."""

import math

import numpy as np


def fit_inspection_camera(points, *, width=1280, height=720, focal_length=12.0, view_direction=(0.65, -0.4, 1.0)):
    """Frame the bounds, with ``view_direction`` pointing from target to eye."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("Camera framing needs nonempty finite [N,3] points")
    if min(width, height, focal_length) <= 0 or not np.isfinite([width, height, focal_length]).all():
        raise ValueError("Camera dimensions and focal length must be finite and positive")
    target = (points.min(0) + points.max(0)) / 2
    radius = float(np.linalg.norm(points - target, axis=1).max())
    # Convert focal length using a 15.2908 mm vertical camera aperture.
    vertical = math.atan(15.2908 / (2 * focal_length))
    horizontal = math.atan(math.tan(vertical) * width / height)
    distance = max(0.075, 1.15 * radius / math.sin(min(vertical, horizontal)), radius + 0.02)
    direction = np.asarray(view_direction, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError("Camera view direction must contain three finite values and be nonzero")
    direction = direction / np.linalg.norm(direction)
    return tuple(target + distance * direction), tuple(target)


def framing_points(scene, geometry, observations, offsets):
    """Frame the full pad, support, thread and tool CAD in every displayed world."""
    rest = scene.surface.rest_positions
    center = (rest.min(0) + rest.max(0)) / 2
    center[2] = rest[:, 2].min() - 0.002
    support = center + np.array([[-0.2, -0.2, -0.001], [0.2, 0.2, 0.001]])
    meshes = [(scene.needle_body, scene.needle.meshes[0])]
    if hasattr(scene, "tool_parts"):
        # Articulated parts are already in each moving body's local frame.
        for parts in scene.tool_parts:
            meshes.extend(parts)
    else:
        for assembly, asset in zip(geometry["grippers"], scene.assets, strict=True):
            for mesh in asset.meshes:
                body = assembly["palm_body_index"]
                if mesh.name in ("jaw1", "jaw2"):
                    body = assembly["jaw_body_indices"][int(mesh.name[-1]) - 1]
                meshes.append((body, mesh))
    trocar_bounds = [
        np.asarray(assembly["trocar_pose"][:3]) + np.array([[-0.008] * 3, [0.008] * 3])
        for assembly in geometry["grippers"]
        if "trocar_pose" in assembly
    ]
    bounds = []
    for observation, offset in zip(observations, offsets, strict=True):
        points = [
            rest,
            support,
            observation["tissue_positions"],
            observation["thread_positions"],
            *trocar_bounds,
        ]
        poses = np.asarray(observation["rigid_body_poses"])
        for body, mesh in meshes:
            xyz, quat = poses[body, :3], poses[body, 3:]
            cross = 2 * np.cross(quat[:3], mesh.vertices)
            points.append(mesh.vertices + quat[3] * cross + np.cross(quat[:3], cross) + xyz)
        vertices = np.concatenate(points)
        low, high = vertices.min(0) + offset, vertices.max(0) + offset
        # All eight corners, including the two mixed XY corners of the support.
        bounds.extend([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    return np.asarray(bounds)


def update_scene(scene, state, observations, *, particle_offsets=None):
    for index, observation in enumerate(observations):
        diagnostics = observation["diagnostics"]
        if (
            diagnostics.get("failure_reason")
            or diagnostics.get("task_status") == "failed"
            or diagnostics.get("status") == "failed"
        ):
            raise RuntimeError(f"Scene {index} failed: {diagnostics}")
    positions = np.asarray([item["tissue_positions"] for item in observations], dtype=np.float32)
    poses = np.asarray([item["needle_pose"] for item in observations], dtype=np.float32)
    if not np.isfinite(positions).all() or not np.isfinite(poses).all():
        raise ValueError("Cannot display nonfinite physical state")
    # Add soft-mesh world offsets to the existing host display snapshot before
    # its one upload. Reading the uploaded GPU mesh back here stalls every frame.
    scene.update(state, observations, positions, poses, particle_offsets=particle_offsets)
