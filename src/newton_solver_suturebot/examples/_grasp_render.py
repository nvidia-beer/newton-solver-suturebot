# SPDX-License-Identifier: Apache-2.0
"""Original tool/needle CAD driven exclusively by accepted physical snapshots."""

import newton
import numpy as np
import warp as wp

from ._grasp_surface import FullSurfaceGeometry
from ._grasp_visual_assets import load_dvrk_tool, load_needle_visual, physical_tool_parts


def tube_vertices(centerlines, radius, sides=8):
    """Build physical-radius rings with transported, continuous cross sections."""
    points = np.asarray(centerlines, dtype=np.float64)
    if (
        points.ndim != 3
        or points.shape[0] < 1
        or points.shape[1] < 2
        or points.shape[2] != 3
        or not np.isfinite(points).all()
    ):
        raise ValueError("Thread centerlines must be finite [N,K,3], N>=1, K>=2")
    if not np.isscalar(radius) or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Tube radius must be finite and positive")
    if not isinstance(sides, (int, np.integer)) or isinstance(sides, bool) or sides < 3:
        raise ValueError("A tube requires at least three integer sides")
    tangents = np.empty_like(points)
    tangents[:, 1:-1] = points[:, 2:] - points[:, :-2]
    tangents[:, 0] = points[:, 1] - points[:, 0]
    tangents[:, -1] = points[:, -1] - points[:, -2]
    norm = np.linalg.norm(tangents, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm < 1e-12):
        raise ValueError("Thread centerline has a degenerate tangent")
    tangents /= norm
    normal = np.empty_like(tangents)
    axis = np.eye(3)[np.argmin(np.abs(tangents[:, 0]), axis=1)]
    normal[:, 0] = np.cross(tangents[:, 0], axis)
    normal[:, 0] /= np.linalg.norm(normal[:, 0], axis=-1, keepdims=True)
    for index in range(1, points.shape[1]):
        previous, current = tangents[:, index - 1], tangents[:, index]
        previous_normal = normal[:, index - 1]
        denominator = 1 + np.sum(previous * current, axis=1)
        # Minimal rotation from the previous tangent to the next. An exact
        # reversal has no unique transported frame; retaining the previous
        # normal gives a valid half-turn without introducing NaNs.
        factor = np.divide(
            np.sum(previous_normal * current, axis=1),
            denominator,
            out=np.zeros(len(points)),
            where=denominator > 1e-10,
        )
        transported = previous_normal - factor[:, None] * (previous + current)
        transported -= np.sum(transported * current, axis=1)[:, None] * current
        transported /= np.linalg.norm(transported, axis=1, keepdims=True)
        normal[:, index] = transported
    binormal = np.cross(tangents, normal)
    angles = np.arange(sides) * (2 * np.pi / sides)
    rings = points[:, :, None] + radius * (
        normal[:, :, None] * np.cos(angles)[None, None, :, None]
        + binormal[:, :, None] * np.sin(angles)[None, None, :, None]
    )
    return rings.astype(np.float32).reshape(len(points), -1, 3)


def tube_triangles(count, sides=8):
    """Outward-wound side faces, with one open boundary ring at each end."""
    if (
        any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) for value in (count, sides))
        or count < 2
        or sides < 3
    ):
        raise ValueError("Tube topology requires integer count>=2 and sides>=3")
    rings, sectors = np.meshgrid(np.arange(count - 1), np.arange(sides), indexing="ij")
    a = rings * sides + sectors
    b = rings * sides + (sectors + 1) % sides
    return np.stack((np.stack((a, b, b + sides), -1), np.stack((a, b + sides, a + sides), -1)), -2).reshape(-1, 3)


def _trocar_marker_mesh():
    """Small visual annulus in local XY; its opening follows local shaft Z."""
    sides, inner, outer, half_depth = 32, 0.0055, 0.0075, 0.001
    angle = np.arange(sides) * (2 * np.pi / sides)
    vertices = np.concatenate(
        [
            np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.full(sides, z)))
            for z in (-half_depth, half_depth)
            for radius in (inner, outer)
        ]
    ).astype(np.float32)
    triangles = []
    for i in range(sides):
        j = (i + 1) % sides
        # Bottom, top, outside and inside surfaces, all outward facing.
        for a, b, c, d in (
            (i, j, sides + j, sides + i),
            (2 * sides + i, 3 * sides + i, 3 * sides + j, 2 * sides + j),
            (sides + i, sides + j, 3 * sides + j, 3 * sides + i),
            (i, 2 * sides + i, 2 * sides + j, j),
        ):
            triangles.extend(((a, b, c), (a, c, d)))
    return newton.Mesh(
        vertices,
        np.asarray(triangles, dtype=np.int32).reshape(-1),
        color=(0.85, 0.60, 0.15),
        roughness=0.6,
        metallic=0.2,
        compute_inertia=False,
    )


class PhysicalSurgicalRenderScene:
    """No render-time constraints, attachments or thread integration."""

    def __init__(self, builder, geometry, num_envs, device):
        self.num_envs = num_envs
        self.surface = FullSurfaceGeometry(geometry)
        self.thread = None
        self.thread_radius = float(geometry["thread_radius"])
        self.thread_point_count = int(geometry["thread_point_count"])
        self.body_count = len(geometry["rigid_body_labels"])
        self.body_indices = []
        self.needle_body = int(geometry["needle_body_index"])
        self.metal_shapes = []
        self.needle = load_needle_visual(geometry.get("needle_visual_asset"))
        self.assets = [
            load_dvrk_tool(
                instrument=tool["instrument"],
                jaw_opening=0,
                shaft_length_m=tool.get("parameters", {}).get("shaft_length_m"),
            )
            for tool in geometry["grippers"]
        ]
        self.tool_parts = [
            physical_tool_parts(asset, tool, body_count=self.body_count)
            for asset, tool in zip(self.assets, geometry["grippers"], strict=True)
        ]
        self.trocar_poses = [
            np.asarray(tool["trocar_pose"], dtype=float).copy()
            for tool in geometry["grippers"]
            if "trocar_pose" in tool
        ]
        if any(
            pose.shape != (7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-5)
            for pose in self.trocar_poses
        ):
            raise ValueError("Trocar markers need finite xyz/xyzw poses with unit quaternions")
        trocar_mesh = _trocar_marker_mesh() if self.trocar_poses else None
        needle_mesh = self.needle.meshes[0].to_newton_mesh()
        tool_meshes = [[mesh.to_newton_mesh() for _, mesh in parts] for parts in self.tool_parts]
        self.last_thread_diagnostics = []
        cfg = newton.ModelBuilder.ShapeConfig(density=0, has_shape_collision=False, has_particle_collision=False)
        rest = self.surface.rest_positions
        low, high = rest.min(0), rest.max(0)
        drape_center = (low + high) / 2
        drape_center[2] = low[2] - 0.002
        for world in range(num_envs):
            builder.begin_world()
            offset = builder.particle_count
            builder.add_particles(rest, np.zeros_like(rest), np.zeros(len(rest)))
            builder.add_triangles(*(self.surface.triangles + offset).T, color=(0.86, 0.48, 0.40))
            line = np.column_stack(
                (np.linspace(0, 0.18, self.thread_point_count), np.zeros((self.thread_point_count, 2)))
            )
            vertices = tube_vertices(line[None], self.thread_radius)[0]
            offset = builder.particle_count
            builder.add_particles(vertices, np.zeros_like(vertices), np.zeros(len(vertices)))
            builder.add_triangles(*(tube_triangles(self.thread_point_count) + offset).T, color=(0.19, 0.09, 0.29))
            bodies = [
                builder.add_body(is_kinematic=True, label=f"render_{world}_{index}") for index in range(self.body_count)
            ]
            self.body_indices.append(bodies)
            needle_shape = builder.add_shape_mesh(bodies[self.needle_body], mesh=needle_mesh, cfg=cfg)
            self.metal_shapes.append((needle_shape, 0.19))
            for physical_parts, parts in zip(self.tool_parts, tool_meshes, strict=True):
                for (body, mesh), part in zip(physical_parts, parts, strict=True):
                    shape = builder.add_shape_mesh(bodies[body], mesh=part, cfg=cfg)
                    if mesh.metallic > 0.5:
                        self.metal_shapes.append((shape, max(mesh.roughness, 0.22)))
            for index, pose in enumerate(self.trocar_poses):
                builder.add_shape_mesh(
                    -1,
                    xform=wp.transform(*pose),
                    mesh=trocar_mesh,
                    cfg=cfg,
                    label=f"trocar_marker_{world}_{index}",
                )
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(drape_center), wp.quat_identity()),
                hx=0.20,
                hy=0.20,
                hz=0.001,
                cfg=cfg,
                color=(0.16, 0.34, 0.37),
            )
            builder.end_world()
        self.body_indices = np.asarray(self.body_indices)

    def update(self, state, observations, positions, poses, *, particle_offsets=None):
        surface = self.surface.update(positions)
        centers = np.asarray([observation["thread_positions"] for observation in observations], dtype=np.float32)
        bodies = np.asarray([observation["rigid_body_poses"] for observation in observations], dtype=np.float32)
        if centers.shape != (self.num_envs, self.thread_point_count, 3) or bodies.shape != (
            self.num_envs,
            self.body_count,
            7,
        ):
            raise ValueError("Physical thread or rigid body layout changed; recreate the renderer")
        if not np.isfinite(centers).all() or not np.isfinite(bodies).all():
            raise ValueError("Cannot render nonfinite physical thread or body state")
        if not np.allclose(bodies[:, self.needle_body], poses, atol=1e-7, rtol=0):
            raise ValueError("Independent needle pose disagrees with its physical body snapshot")
        surface = np.concatenate((surface, tube_vertices(centers, self.thread_radius)), axis=1)
        particles = np.asarray(surface, dtype=np.float32).reshape(-1, 3)
        if particle_offsets is not None:
            # ViewerGL offsets rigid shapes itself; soft meshes need the same
            # offsets in this disposable render copy. Physical snapshots stay
            # unchanged, and no GPU-to-CPU round trip is needed for display.
            offsets = np.asarray(particle_offsets, dtype=np.float32)
            if offsets.shape != particles.shape:
                raise ValueError("Display offsets must match the rendered particle layout")
            particles += offsets
        state.particle_q.assign(particles)
        render_bodies = np.empty((self.num_envs * self.body_count, 7), dtype=np.float32)
        render_bodies[self.body_indices] = bodies
        state.body_q.assign(render_bodies)
        self.last_thread_diagnostics = [observation.get("thread_diagnostics", {}) for observation in observations]

    def metadata(self):
        return {
            "tissue": "full native exterior; active surface follows accepted FEM state",
            "tissue_surface_vertices": len(self.surface.rest_positions),
            "tissue_surface_triangles": len(self.surface.triangles),
            "tissue_material": "constant phantom diffuse color; no photographic texture",
            "needle": self.needle.metadata,
            "tools": [asset.metadata for asset in self.assets],
            "tool_motion": "accepted MuJoCo component body poses, including articulated wrists when present; no needle-relative render attachment",
            "trocar_markers": "fixed gold pivot rings; visual only, with no contact or added constraint",
            "trocar_marker_count": len(self.trocar_poses),
            "full_robot_arms": False,
            "camera_rig_geometry": False,
            "thread": "accepted worker ANCF state with tissue, needle, gripper jaw and shaft capsule contacts; lagged reaction forces",
            "background": "synthetic teal support surface beneath pad; presentation only",
            "lighting": "Newton viewer lighting; authored material colors, roughness and metallic values",
            "pbr_material_limit": "appearance depends on the viewer; optical properties are uncalibrated",
        }

    def diagnostics(self):
        return {"physical_thread": self.last_thread_diagnostics}
