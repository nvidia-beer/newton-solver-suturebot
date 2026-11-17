# SPDX-License-Identifier: Apache-2.0
"""Bounded ordered facet scratch for one unchanged captured surface trial.

Every fresh trial collects anew. Full overflow selects the unmodified original
kernel before any force or recorder output is written. No inter-trial geometry,
permissions, nearest/sign result or material history is reused.
"""

import warp as wp

from ...fem.differentiation import active_tape
from ..acceleration.surface_contact_tangent import SurfaceTangentAnchors, surface_contact_tangent
from .insertion import (
    _facet_weight,
    _give_force,
    _point_inverse_mass,
    _surface_facet,
    _surface_forces,
    _surface_permitted,
)
from .surface_contact_forces import surface_forces_with_tangent

_ORIGINAL_KERNEL = surface_forces_with_tangent
_ORIGINAL_PHYSICAL_KERNEL = _surface_forces


@wp.kernel(enable_backward=False)
def collect_facets(
    mesh: wp.uint64,
    active_mesh: wp.uint64,
    triangles: wp.array[wp.vec3i],
    points: wp.array[wp.vec3],
    mapping: wp.array[int],
    fixed: wp.array[int],
    qd: wp.array[wp.vec3],
    samples: wp.array[wp.vec3],
    radii: wp.array[float],
    weights: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.vec3,
    body: int,
    stiffness: float,
    damping: float,
    friction: float,
    speed_scale: float,
    dt: wp.array[float],
    limit_impulses: int,
    rigid_inverse_mass: float,
    rigid_inverse_inertia: wp.mat33,
    particle_inverse_mass: wp.array[float],
    openings: wp.array[wp.vec3],
    opening_faces: wp.array2d[int],
    opening_radius: float,
    sample_in_visited_material: wp.array[int],
    tip_exit_point: wp.vec3,
    track: wp.array[wp.vec3],
    live: wp.array[int],
    capture_radius: float,
    particle_f: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    diagnostics: wp.array[wp.vec3],
    contact_flags: wp.array[int],
    moment_length: float,
    anchor_count: wp.array[int],
    anchor_flags: wp.array[int],
    anchor_nodes: wp.array[wp.vec3i],
    anchor_bary: wp.array[wp.vec3],
    anchor_points: wp.array[wp.vec3],
    anchor_levers: wp.array[wp.vec3],
    anchor_normals: wp.array[wp.vec3],
    anchor_tangents: wp.array[wp.vec3],
    anchor_matrices: wp.array[wp.mat33],
    anchor_sources: wp.array[wp.vec2i],
    anchor_scalars: wp.array2d[float],
    counts: wp.array[int],
    face_ids: wp.array2d[int],
    saved_bary: wp.array2d[wp.vec3],
    saved_points: wp.array2d[wp.vec3],
    saved_normals: wp.array2d[wp.vec3],
    saved_geometry: wp.array2d[wp.vec3],
    permissions: wp.array2d[int],
    passages: wp.array[int],
    sides: wp.array[float],
    valid: wp.array[int],
    overflow: wp.array[int],
):
    i = wp.tid()
    counts[i] = -1
    pose = body_q[body]
    sample = wp.transform_point(pose, samples[i])
    hit = wp.mesh_query_point_sign_normal(mesh, sample, 0.02)
    if not hit.result:
        return
    nearest_bary, nearest_point, nearest_normal, nearest_distance, nearest_gap, nearest_area = _surface_facet(
        sample, triangles[hit.face], points
    )
    if hit.sign > 0.0 and nearest_distance >= radii[i]:
        return
    local_passage = _surface_permitted(
        active_mesh,
        i,
        hit.face,
        sample,
        nearest_point,
        nearest_normal,
        nearest_gap,
        nearest_bary,
        triangles,
        points,
        mapping,
        fixed,
        openings,
        opening_faces,
        opening_radius,
        sample_in_visited_material,
        tip_exit_point,
        track,
        live,
        capture_radius,
    )
    width = 0.5 * radii[i]
    extent = wp.vec3(wp.max(radii[i], nearest_distance) + width)
    if local_passage:
        extent = wp.vec3(radii[i] + width)
    passages[i] = int(local_passage)
    sides[i] = hit.sign
    query = wp.mesh_query_aabb(mesh, sample - extent, sample + extent)
    face = int(0)
    count = int(0)
    while wp.mesh_query_aabb_next(query, face):
        if count < face_ids.shape[1]:
            bary, point, normal, distance, gap, area = _surface_facet(sample, triangles[face], points)
            face_ids[i, count] = face
            saved_bary[i, count] = bary
            saved_points[i, count] = point
            saved_normals[i, count] = normal
            saved_geometry[i, count] = wp.vec3(distance, gap, area)
            permissions[i, count] = -1
        count += 1
    counts[i] = count
    if count > face_ids.shape[1]:
        wp.atomic_max(overflow, 0, 1)
        wp.atomic_min(valid, 0, 0)


@wp.kernel(enable_backward=False)
def forces_from_facets(
    mesh: wp.uint64,
    active_mesh: wp.uint64,
    triangles: wp.array[wp.vec3i],
    points: wp.array[wp.vec3],
    mapping: wp.array[int],
    fixed: wp.array[int],
    qd: wp.array[wp.vec3],
    samples: wp.array[wp.vec3],
    radii: wp.array[float],
    weights: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.vec3,
    body: int,
    stiffness: float,
    damping: float,
    friction: float,
    speed_scale: float,
    dt: wp.array[float],
    limit_impulses: int,
    rigid_inverse_mass: float,
    rigid_inverse_inertia: wp.mat33,
    particle_inverse_mass: wp.array[float],
    openings: wp.array[wp.vec3],
    opening_faces: wp.array2d[int],
    opening_radius: float,
    sample_in_visited_material: wp.array[int],
    tip_exit_point: wp.vec3,
    track: wp.array[wp.vec3],
    live: wp.array[int],
    capture_radius: float,
    particle_f: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    diagnostics: wp.array[wp.vec3],
    contact_flags: wp.array[int],
    moment_length: float,
    anchor_count: wp.array[int],
    anchor_flags: wp.array[int],
    anchor_nodes: wp.array[wp.vec3i],
    anchor_bary: wp.array[wp.vec3],
    anchor_points: wp.array[wp.vec3],
    anchor_levers: wp.array[wp.vec3],
    anchor_normals: wp.array[wp.vec3],
    anchor_tangents: wp.array[wp.vec3],
    anchor_matrices: wp.array[wp.mat33],
    anchor_sources: wp.array[wp.vec2i],
    anchor_scalars: wp.array2d[float],
    counts: wp.array[int],
    face_ids: wp.array2d[int],
    saved_bary: wp.array2d[wp.vec3],
    saved_points: wp.array2d[wp.vec3],
    saved_normals: wp.array2d[wp.vec3],
    saved_geometry: wp.array2d[wp.vec3],
    permissions: wp.array2d[int],
    passages: wp.array[int],
    sides: wp.array[float],
    valid: wp.array[int],
    overflow: wp.array[int],
):
    i = wp.tid()
    if counts[i] < 0:
        return
    pose = body_q[body]
    sample = wp.transform_point(pose, samples[i])
    local_passage = passages[i] != 0
    width = 0.5 * radii[i]
    minimum = float(1e20)
    side = sides[i]
    for facet_slot in range(counts[i]):
        face = face_ids[i, facet_slot]
        bary = saved_bary[i, facet_slot]
        point = saved_points[i, facet_slot]
        normal = saved_normals[i, facet_slot]
        distance = saved_geometry[i, facet_slot][0]
        gap = saved_geometry[i, facet_slot][1]
        area = saved_geometry[i, facet_slot][2]
        if not (
            distance < minimum
            and (not local_passage or distance < radii[i])
            and (not local_passage and sides[i] < 0.0 or gap >= 0.0)
        ):
            continue
        if permissions[i, facet_slot] < 0:
            permitted = _surface_permitted(
                active_mesh,
                i,
                face,
                sample,
                point,
                normal,
                gap,
                bary,
                triangles,
                points,
                mapping,
                fixed,
                openings,
                opening_faces,
                opening_radius,
                sample_in_visited_material,
                tip_exit_point,
                track,
                live,
                capture_radius,
            )
            permissions[i, facet_slot] = int(permitted)
        else:
            permitted = permissions[i, facet_slot] != 0
        if not permitted:
            minimum = distance
            side = float(1.0)
            if gap < 0.0 and (not local_passage):
                side = -1.0
    penetration_limit = wp.max(0.0, radii[i] - side * minimum)
    if minimum == 1e20 or penetration_limit <= 0.0:
        return
    normalization = float(0.0)
    for facet_slot in range(counts[i]):
        face = face_ids[i, facet_slot]
        bary = saved_bary[i, facet_slot]
        point = saved_points[i, facet_slot]
        normal = saved_normals[i, facet_slot]
        distance = saved_geometry[i, facet_slot][0]
        gap = saved_geometry[i, facet_slot][1]
        area = saved_geometry[i, facet_slot][2]
        weight = _facet_weight(distance, gap, area, minimum, side, width)
        if local_passage:
            weight *= wp.pow(wp.max(0.0, 1.0 - distance * distance / (radii[i] * radii[i])), 2.0)
        if weight == 0.0:
            continue
        if permissions[i, facet_slot] < 0:
            permitted = _surface_permitted(
                active_mesh,
                i,
                face,
                sample,
                point,
                normal,
                gap,
                bary,
                triangles,
                points,
                mapping,
                fixed,
                openings,
                opening_faces,
                opening_radius,
                sample_in_visited_material,
                tip_exit_point,
                track,
                live,
                capture_radius,
            )
            permissions[i, facet_slot] = int(permitted)
        else:
            permitted = permissions[i, facet_slot] != 0
        if not permitted:
            normalization += weight
    if normalization <= 1e-30:
        return
    normal_resultant = wp.vec3(0.0)
    com = wp.transform_point(pose, body_com)
    for facet_slot in range(counts[i]):
        face = face_ids[i, facet_slot]
        ids = triangles[face]
        bary = saved_bary[i, facet_slot]
        point = saved_points[i, facet_slot]
        normal = saved_normals[i, facet_slot]
        distance = saved_geometry[i, facet_slot][0]
        gap = saved_geometry[i, facet_slot][1]
        area = saved_geometry[i, facet_slot][2]
        weight = _facet_weight(distance, gap, area, minimum, side, width) / normalization
        if local_passage:
            weight *= wp.pow(wp.max(0.0, 1.0 - distance * distance / (radii[i] * radii[i])), 2.0)
        penetration = wp.min(wp.max(0.0, radii[i] - gap), penetration_limit)
        if not (weight > 0.0 and penetration > 0.0):
            continue
        if permissions[i, facet_slot] < 0:
            permitted = _surface_permitted(
                active_mesh,
                i,
                face,
                sample,
                point,
                normal,
                gap,
                bary,
                triangles,
                points,
                mapping,
                fixed,
                openings,
                opening_faces,
                opening_radius,
                sample_in_visited_material,
                tip_exit_point,
                track,
                live,
                capture_radius,
            )
            permissions[i, facet_slot] = int(permitted)
        else:
            permitted = permissions[i, facet_slot] != 0
        if not permitted:
            velocity = wp.spatial_top(body_qd[body]) + wp.cross(wp.spatial_bottom(body_qd[body]), point - com)
            for j in range(3):
                if mapping[ids[j]] >= 0:
                    velocity -= bary[j] * qd[mapping[ids[j]]]
            vn = wp.dot(velocity, normal)
            tangent = velocity - vn * normal
            fn = weights[i] * weight * (stiffness * penetration - damping * penetration / radii[i] * wp.min(vn, 0.0))
            normal_resultant += fn * normal
            tangent_speed = wp.length(tangent)
            tangent_direction = tangent / wp.max(tangent_speed, 1e-12)
            friction_magnitude = friction * fn * tangent_speed / wp.max(tangent_speed, speed_scale)
            inverse_mass = _point_inverse_mass(
                tangent_direction, point, com, pose, rigid_inverse_mass, rigid_inverse_inertia
            )
            for j in range(3):
                particle = mapping[ids[j]]
                if particle >= 0 and fixed[particle] == 0:
                    inverse_mass += bary[j] * bary[j] * particle_inverse_mass[particle]
            if limit_impulses != 0 and inverse_mass > 0.0:
                friction_magnitude = wp.min(friction_magnitude, tangent_speed / (dt[0] * inverse_mass))
            force = fn * normal - friction_magnitude * tangent_direction
            _give_force(force, point, com, body, body_f, diagnostics)
            for j in range(3):
                particle = mapping[ids[j]]
                reaction = -bary[j] * force
                if particle >= 0:
                    wp.atomic_add(particle_f, particle, reaction)
                if particle < 0 or fixed[wp.max(particle, 0)] != 0:
                    wp.atomic_add(diagnostics, 4, reaction)
                    wp.atomic_add(diagnostics, 5, wp.cross(points[ids[j]], reaction))
                    if bary[j] > 0.05 and penetration > radii[i] + 1e-08:
                        wp.atomic_max(contact_flags, 1, 1)
                else:
                    wp.atomic_add(diagnostics, 2, reaction)
                wp.atomic_add(diagnostics, 3, wp.cross(points[ids[j]], reaction))
            wp.atomic_max(contact_flags, 0, 1)
            if anchor_count.shape[0] != 0:
                if limit_impulses != 0:
                    wp.atomic_max(anchor_flags, 1, 1)
                slot = wp.atomic_add(anchor_count, 0, 1)
                if slot < anchor_matrices.shape[0]:
                    anchor_nodes[slot] = wp.vec3i(mapping[ids[0]], mapping[ids[1]], mapping[ids[2]])
                    anchor_bary[slot] = bary
                    anchor_points[slot] = point
                    anchor_levers[slot] = (point - com) / moment_length
                    anchor_normals[slot] = normal
                    anchor_tangents[slot] = tangent
                    anchor_sources[slot] = wp.vec2i(i, face)
                    anchor_scalars[slot, 0] = weights[i] * weight
                    anchor_scalars[slot, 1] = penetration
                    anchor_scalars[slot, 2] = vn
                    anchor_scalars[slot, 3] = fn
                    anchor_scalars[slot, 4] = radii[i]
                    anchor_scalars[slot, 5] = weight
                    anchor_scalars[slot, 6] = penetration_limit
                    anchor_matrices[slot] = surface_contact_tangent(
                        weights[i] * weight,
                        penetration,
                        vn,
                        fn,
                        tangent,
                        normal,
                        stiffness,
                        damping,
                        radii[i],
                        friction,
                        speed_scale,
                        dt[0],
                    )
                else:
                    wp.atomic_max(anchor_flags, 0, 1)
    if i == 0:
        diagnostics[6] = wp.vec3(wp.length(normal_resultant), penetration_limit, 0.0)


class SurfaceFacetScratch:
    """Strong fixed owners; only captured CUDA forward calls use this path."""

    def __init__(self, samples, device, *, capacity=64):
        if samples < 1 or capacity < 0:
            raise ValueError("Positive sample count and nonnegative facet capacity required")
        self.device = wp.get_device(device)
        self.samples, self.capacity = int(samples), int(capacity)
        shape = (self.samples, self.capacity)
        self.counts = wp.empty(self.samples, dtype=int, device=self.device)
        self.faces = wp.empty(shape, dtype=int, device=self.device)
        self.bary = wp.empty(shape, dtype=wp.vec3, device=self.device)
        self.points = wp.empty(shape, dtype=wp.vec3, device=self.device)
        self.normals = wp.empty(shape, dtype=wp.vec3, device=self.device)
        self.geometry = wp.empty(shape, dtype=wp.vec3, device=self.device)
        self.permissions = wp.empty(shape, dtype=int, device=self.device)
        self.passages = wp.empty(self.samples, dtype=int, device=self.device)
        self.sides = wp.empty(self.samples, dtype=float, device=self.device)
        self.valid = wp.ones(1, dtype=int, device=self.device)
        self.overflow = wp.zeros(1, dtype=int, device=self.device)
        self.disabled_recording = SurfaceTangentAnchors(0, device=self.device, enabled=False)
        if self.device.is_cuda:
            wp.load_module(collect_facets.module, device=self.device, block_dim=1)

    @property
    def storage(self):
        return (
            self.counts,
            self.faces,
            self.bary,
            self.points,
            self.normals,
            self.geometry,
            self.permissions,
            self.passages,
            self.sides,
            self.valid,
            self.overflow,
        )

    @property
    def bindings(self):
        return self.storage + tuple(self.disabled_recording.arguments(1.0)[1:])

    def collect(self, arguments, *, block_dim=1):
        self.valid.fill_(1)
        self.overflow.zero_()
        wp.launch(collect_facets, self.samples, [*arguments, *self.storage], device=self.device, block_dim=block_dim)

    def force(self, arguments, *, block_dim=1):
        """Internal second pass: the caller must select valid bounded collection."""
        wp.launch(
            forces_from_facets, self.samples, [*arguments, *self.storage], device=self.device, block_dim=block_dim
        )

    def launch(self, kernel, arguments, *, block_dim=1):
        if (
            kernel not in (_ORIGINAL_KERNEL, _ORIGINAL_PHYSICAL_KERNEL)
            or not self.device.is_cuda
            or not self.device.is_capturing
            or active_tape() is not None
            or any(isinstance(a, wp.array) and (a.requires_grad or a.device != self.device) for a in arguments)
        ):
            return False
        original_arguments = arguments
        if kernel is _ORIGINAL_PHYSICAL_KERNEL:
            arguments = [*arguments, *self.disabled_recording.arguments(1.0)]
        if block_dim != 1 or len(arguments) != 47 or arguments[7].shape != (self.samples,):
            return False
        expected = (
            ((self.samples,), wp.int32),
            ((self.samples, self.capacity), wp.int32),
            ((self.samples, self.capacity), wp.vec3),
            ((self.samples, self.capacity), wp.vec3),
            ((self.samples, self.capacity), wp.vec3),
            ((self.samples, self.capacity), wp.vec3),
            ((self.samples, self.capacity), wp.int32),
            ((self.samples,), wp.int32),
            ((self.samples,), wp.float32),
            ((1,), wp.int32),
            ((1,), wp.int32),
        )
        if any(
            a.shape != shape or a.dtype != dtype or a.device != self.device or a.requires_grad
            for a, (shape, dtype) in zip(self.storage, expected, strict=True)
        ):
            return False
        self.collect(arguments, block_dim=block_dim)
        wp.capture_if(self.valid, lambda: self.force(arguments, block_dim=block_dim))
        wp.capture_if(
            self.overflow,
            lambda: wp.launch(kernel, self.samples, original_arguments, device=self.device, block_dim=block_dim),
        )
        return True
