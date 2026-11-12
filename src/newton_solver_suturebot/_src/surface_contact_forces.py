# SPDX-License-Identifier: Apache-2.0
"""Surface forces with optional numerical contact-tangent recording.

The physical facet pass matches insertion._surface_forces. The extra hook
records its contributing facets for the surface search preconditioner without
another geometry query. The original kernel remains the default path; an AST
regression guards the shared physical statements against drift.
"""

import warp as wp

from newton_solver_suturebot._src.insertion import (
    _facet_weight,
    _give_force,
    _point_inverse_mass,
    _surface_facet,
    _surface_permitted,
)

from .surface_contact_tangent import surface_contact_tangent


@wp.kernel(enable_backward=False)
def surface_forces_with_tangent(
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
    dt: float,
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
    track_count: int,
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
):
    i = wp.tid()
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
        track_count,
        capture_radius,
    )
    width = 0.5 * radii[i]
    # Search the entire physical radius even when the nearest face belongs to
    # an authorized opening, so a second intact wall cannot hide behind it.
    extent = wp.vec3(wp.max(radii[i], nearest_distance) + width)
    if local_passage:
        extent = wp.vec3(radii[i] + width)
    minimum = float(1.0e20)
    side = hit.sign
    query = wp.mesh_query_aabb(mesh, sample - extent, sample + extent)
    face = int(0)
    while wp.mesh_query_aabb_next(query, face):
        bary, point, normal, distance, gap, area = _surface_facet(sample, triangles[face], points)
        if not (
            distance < minimum
            and (not local_passage or distance < radii[i])
            and ((not local_passage and hit.sign < 0.0) or gap >= 0.0)
        ):
            continue
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
            track_count,
            capture_radius,
        )
        if not permitted:
            minimum = distance
            side = float(1.0)
            if gap < 0.0 and not local_passage:
                side = -1.0
    penetration_limit = wp.max(0.0, radii[i] - side * minimum)
    if minimum == 1.0e20 or penetration_limit <= 0.0:
        return
    normalization = float(0.0)
    query = wp.mesh_query_aabb(mesh, sample - extent, sample + extent)
    while wp.mesh_query_aabb_next(query, face):
        bary, point, normal, distance, gap, area = _surface_facet(sample, triangles[face], points)
        weight = _facet_weight(distance, gap, area, minimum, side, width)
        if local_passage:
            weight *= wp.pow(wp.max(0.0, 1.0 - distance * distance / (radii[i] * radii[i])), 2.0)
        if weight == 0.0:
            continue
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
            track_count,
            capture_radius,
        )
        if not permitted:
            normalization += weight
    if normalization <= 1.0e-30:
        return
    normal_resultant = wp.vec3(0.0)
    com = wp.transform_point(pose, body_com)
    query = wp.mesh_query_aabb(mesh, sample - extent, sample + extent)
    while wp.mesh_query_aabb_next(query, face):
        ids = triangles[face]
        bary, point, normal, distance, gap, area = _surface_facet(sample, ids, points)
        weight = _facet_weight(distance, gap, area, minimum, side, width) / normalization
        if local_passage:
            weight *= wp.pow(wp.max(0.0, 1.0 - distance * distance / (radii[i] * radii[i])), 2.0)
        penetration = wp.min(wp.max(0.0, radii[i] - gap), penetration_limit)
        if not (weight > 0.0 and penetration > 0.0):
            continue
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
            track_count,
            capture_radius,
        )
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
            tangent_direction = tangent / wp.max(tangent_speed, 1.0e-12)
            friction_magnitude = friction * fn * tangent_speed / wp.max(tangent_speed, speed_scale)
            inverse_mass = _point_inverse_mass(
                tangent_direction, point, com, pose, rigid_inverse_mass, rigid_inverse_inertia
            )
            for j in range(3):
                particle = mapping[ids[j]]
                if particle >= 0 and fixed[particle] == 0:
                    inverse_mass += bary[j] * bary[j] * particle_inverse_mass[particle]
            if limit_impulses != 0 and inverse_mass > 0.0:
                friction_magnitude = wp.min(friction_magnitude, tangent_speed / (dt * inverse_mass))
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
                    if bary[j] > 0.05 and penetration > radii[i] + 1.0e-8:
                        wp.atomic_max(contact_flags, 1, 1)
                else:
                    wp.atomic_add(diagnostics, 2, reaction)
                wp.atomic_add(diagnostics, 3, wp.cross(points[ids[j]], reaction))
            wp.atomic_max(contact_flags, 0, 1)
            # Numerical scratch only: use this very facet's existing force
            # geometry and normalized weight; never repeat or truncate forces.
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
                        dt,
                    )
                else:
                    wp.atomic_max(anchor_flags, 0, 1)
    if i == 0:
        diagnostics[6] = wp.vec3(wp.length(normal_resultant), penetration_limit, 0.0)
