# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Illustrative needle interaction forces on an unchanged tetrahedral mesh.

A threshold authorizes one local surface puncture. Accepted actual tip positions
create material track samples; they do not remove material or resolve fracture.
Transverse shaft springs allow tangential sliding, with regularized Coulomb
friction. Cutting resistance is omitted. All parameters
are illustrative, not identified tissue properties. CUDA broad-phase queries
feed the existing double-precision host containment and swept-crossing tests;
force evaluation and balanced scattering use Warp.
"""

import warnings

import numpy as np
import warp as wp

from ....geometry import boundary_faces
from ....geometry.needle_queries import NeedleQueryGeometry
from ....utils.step_time import StepTime


class NeedleStepTooLarge(RuntimeError):
    """A candidate must be retried with smaller physical steps before acceptance."""


# Entries of ``NeedleTissueInteraction._live``, the device array of the live history counts.
LIVE_VISITED = 0
LIVE_TRACK = 1
LIVE_OPENINGS = 2


@wp.func
def _give_force(
    force: wp.vec3,
    point: wp.vec3,
    center: wp.vec3,
    body: int,
    body_f: wp.array[wp.spatial_vector],
    diagnostics: wp.array[wp.vec3],
):
    wp.atomic_add(body_f, body, wp.spatial_vector(force, wp.cross(point - center, force)))
    wp.atomic_add(diagnostics, 0, force)
    wp.atomic_add(diagnostics, 1, wp.cross(point, force))


@wp.func
def _point_inverse_mass(
    direction: wp.vec3,
    point: wp.vec3,
    center: wp.vec3,
    pose: wp.transform,
    inverse_mass: float,
    inverse_inertia: wp.mat33,
):
    angular = wp.quat_rotate_inv(wp.transform_get_rotation(pose), wp.cross(point - center, direction))
    return inverse_mass + wp.dot(angular, inverse_inertia * angular)


@wp.kernel(enable_backward=False)
def _surface_positions(q: wp.array[wp.vec3], mapping: wp.array[int], points: wp.array[wp.vec3]):
    i = wp.tid()
    if mapping[i] >= 0:
        points[i] = q[mapping[i]]


@wp.kernel(enable_backward=False)
def _track_positions(
    q: wp.array[wp.vec3], nodes: wp.array[wp.vec4i], bary: wp.array[wp.vec4], points: wp.array[wp.vec3]
):
    i = wp.tid()
    point = wp.vec3(0.0)
    for j in range(4):
        point += bary[i][j] * q[nodes[i][j]]
    points[i] = point


@wp.func
def _triangle_barycentric(a: wp.vec3, b: wp.vec3, c: wp.vec3, point: wp.vec3):
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = wp.dot(ab, ap), wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)
    bp = point - b
    d3, d4 = wp.dot(ab, bp), wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return wp.vec3(1.0 - v, v, 0.0)
    cp = point - c
    d5, d6 = wp.dot(ab, cp), wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return wp.vec3(1.0 - w, 0.0, w)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 >= d3 and d5 >= d6:
        w = (d4 - d3) / (d4 - d3 + d5 - d6)
        return wp.vec3(0.0, 1.0 - w, w)
    inverse = 1.0 / (va + vb + vc)
    v, w = vb * inverse, vc * inverse
    return wp.vec3(1.0 - v - w, v, w)


@wp.func
def _surface_facet(sample: wp.vec3, ids: wp.vec3i, points: wp.array[wp.vec3]):
    a, b, c = points[ids[0]], points[ids[1]], points[ids[2]]
    bary = _triangle_barycentric(a, b, c, sample)
    point = bary[0] * a + bary[1] * b + bary[2] * c
    area_vector = wp.cross(b - a, c - a)
    twice_area = wp.length(area_vector)
    normal = area_vector / wp.max(twice_area, 1.0e-20)
    distance = wp.length(sample - point)
    gap = wp.dot(normal, sample - point)
    return bary, point, normal, distance, gap, 0.5 * twice_area


@wp.func
def _surface_permitted(
    active_mesh: wp.uint64,
    i: int,
    face: int,
    sample: wp.vec3,
    point: wp.vec3,
    normal: wp.vec3,
    gap: float,
    bary: wp.vec3,
    triangles: wp.array[wp.vec3i],
    points: wp.array[wp.vec3],
    mapping: wp.array[int],
    fixed: wp.array[int],
    openings: wp.array[wp.vec3],
    opening_faces: wp.array2d[int],
    opening_radius: float,
    sample_in_visited_material: wp.array[int],
    tip_exit_point: wp.vec3,
    track: wp.array[wp.vec3],
    live: wp.array[int],
    capture_radius: float,
):
    ids = triangles[face]
    permitted = bool(False)
    associated = (i == 0 and sample_in_visited_material[i] == 2) or sample_in_visited_material[i] == 4
    if sample_in_visited_material[i] == 1:
        for j in range(live[LIVE_TRACK]):
            if wp.length(sample - track[j]) <= capture_radius:
                associated = True
    if associated:
        # The plane of a neighboring facet can extend through the interior of
        # a concave surface. Its signed plane gap cannot identify another wall.
        # Permit only facets reached through active material, ignoring the
        # intended endpoint intersection within float32 geometric roundoff.
        # Reconstruct locally: subtracting a rounded world-space barycentric
        # point loses nanometres that a grazing ray can amplify substantially.
        displacement = (
            bary[0] * (points[ids[0]] - sample)
            + bary[1] * (points[ids[1]] - sample)
            + bary[2] * (points[ids[2]] - sample)
        )
        distance = wp.length(displacement)
        boundary = wp.mesh_query_ray(
            active_mesh, sample, displacement / wp.max(distance, 1.0e-12), wp.max(0.0, distance - 1.0e-8)
        )
        permitted = not boundary.result
    if i == 0 and sample_in_visited_material[i] < 0:
        exit_ids = triangles[-sample_in_visited_material[i] - 1]
        connected = bool(False)
        for j in range(3):
            for k in range(3):
                if ids[j] == exit_ids[k]:
                    connected = True
        exit_normal = wp.normalize(
            wp.cross(points[exit_ids[1]] - points[exit_ids[0]], points[exit_ids[2]] - points[exit_ids[0]])
        )
        if (
            connected
            and wp.dot(normal, exit_normal) > 0.95
            and wp.length(point - tip_exit_point) <= opening_radius
            and wp.length(sample - tip_exit_point) <= capture_radius
        ):
            permitted = True
    for opening_index in range(live[LIVE_OPENINGS]):
        if (
            opening_faces[opening_index, face] != 0
            and wp.length(point - openings[opening_index]) <= opening_radius
            and wp.length(sample - openings[opening_index]) <= capture_radius
        ):
            permitted = True
    supported = True
    for j in range(3):
        particle = mapping[ids[j]]
        if bary[j] > 0.01 and (particle < 0 or fixed[wp.max(particle, 0)] != 0):
            supported = False
    return permitted and supported


@wp.func
def _facet_weight(distance: float, gap: float, area: float, minimum: float, side: float, width: float):
    # Compact C1 distance blending, with a smooth orientation factor excluding
    # the opposite side of a thin wall. This is a pressure-force approximation,
    # not the derivative of a claimed contact potential.
    u = wp.max(0.0, distance - minimum) / width
    value = float(0.0)
    if u < 1.0:
        orientation = wp.max(0.0, side * gap / wp.max(distance, 1.0e-12))
        if distance <= 1.0e-8:
            orientation = 1.0
        value = area * wp.pow(1.0 - u * u, 2.0) * orientation * orientation
    return value


@wp.kernel(enable_backward=False)
def _surface_forces(
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
        live,
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
            live,
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
            live,
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
            live,
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
                    if bary[j] > 0.05 and penetration > radii[i] + 1.0e-8:
                        wp.atomic_max(contact_flags, 1, 1)
                else:
                    wp.atomic_add(diagnostics, 2, reaction)
                wp.atomic_add(diagnostics, 3, wp.cross(points[ids[j]], reaction))
            wp.atomic_max(contact_flags, 0, 1)
    if i == 0:
        diagnostics[6] = wp.vec3(wp.length(normal_resultant), penetration_limit, 0.0)


@wp.func
def _shaft_projection(point: wp.vec3, pose: wp.transform, samples: wp.array[wp.vec3], tangent_radius: float):
    local_point = wp.transform_point(wp.transform_inverse(pose), point)
    closest, tangent = wp.vec3(0.0), wp.vec3(0.0)
    best = float(1.0e20)
    fraction_best = float(0.0)
    segment_best = int(0)
    arc_best, arc_length = float(0.0), float(0.0)
    for j in range(samples.shape[0] - 1):
        a, b = samples[j], samples[j + 1]
        direction = b - a
        length = wp.length(direction)
        fraction = wp.dot(local_point - a, direction) / wp.max(wp.dot(direction, direction), 1.0e-20)
        candidate = a + wp.clamp(fraction, 0.0, 1.0) * direction
        distance = wp.length_sq(candidate - local_point)
        if distance < best:
            closest, tangent, best = candidate, wp.normalize(direction), distance
            fraction_best, segment_best = fraction, j
            arc_best = arc_length + wp.clamp(fraction, 0.0, 1.0) * length
        arc_length += length
    # Slip direction is a physical-radius secant of the authored polyline.
    # It varies continuously along the shaft across authored sample joints,
    # instead of jumping between each incident segment's different tangent.
    # The physical support length remains fixed under sampling refinement.
    before = wp.max(0.0, arc_best - tangent_radius)
    after = wp.min(arc_length, arc_best + tangent_radius)
    lower, upper = wp.vec3(0.0), wp.vec3(0.0)
    arc_length = float(0.0)
    for j in range(samples.shape[0] - 1):
        start_point, end_point = samples[j], samples[j + 1]
        segment_length = wp.length(end_point - start_point)
        if arc_length <= before and before <= arc_length + segment_length:
            lower = start_point + (before - arc_length) / segment_length * (end_point - start_point)
        if arc_length <= after and after <= arc_length + segment_length:
            upper = start_point + (after - arc_length) / segment_length * (end_point - start_point)
        arc_length += segment_length
    if wp.length_sq(upper - lower) > 1.0e-20:
        tangent = wp.normalize(upper - lower)
    covered = not (
        (segment_best == 0 and fraction_best < 0.0) or (segment_best == samples.shape[0] - 2 and fraction_best > 1.0)
    )
    return wp.transform_point(pose, closest), wp.quat_rotate(wp.transform_get_rotation(pose), tangent), best, covered


@wp.kernel(enable_backward=False)
def _update_track_engagement(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    q: wp.array[wp.vec3],
    samples: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body: int,
    engagement_radius: float,
    engaged: wp.array[int],
):
    i = wp.tid()
    point = wp.vec3(0.0)
    for j in range(4):
        point += bary[i][j] * q[nodes[i][j]]
    closest, tangent, distance, covered = _shaft_projection(point, body_q[body], samples, 0.5 * engagement_radius)
    if not covered:
        engaged[i] = 0
    elif engaged[i] != 0 or distance <= engagement_radius * engagement_radius:
        engaged[i] = 1


@wp.kernel(enable_backward=False)
def _shaft_forces(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    lengths: wp.array[float],
    engaged: wp.array[int],
    live: wp.array[int],
    update_engagement: int,
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    fixed: wp.array[int],
    samples: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.vec3,
    body: int,
    stiffness: float,
    damping: float,
    friction: float,
    speed_scale: float,
    engagement_radius: float,
    dt: wp.array[float],
    limit_impulses: int,
    rigid_inverse_mass: float,
    rigid_inverse_inertia: wp.mat33,
    particle_inverse_mass: wp.array[float],
    particle_f: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    diagnostics: wp.array[wp.vec3],
):
    i = wp.tid()
    if i >= live[LIVE_TRACK]:
        return
    point, tissue_velocity = wp.vec3(0.0), wp.vec3(0.0)
    for j in range(4):
        point += bary[i][j] * q[nodes[i][j]]
        tissue_velocity += bary[i][j] * qd[nodes[i][j]]
    pose = body_q[body]
    closest, tangent, best, covered = _shaft_projection(point, pose, samples, 0.5 * engagement_radius)
    # A continuously covered material point keeps resisting lateral motion.
    # Once the finite shaft withdraws, it may engage again only on local return.
    # Implicit evaluation holds this discrete history fixed through the solve:
    # switching on a finite transverse/friction load inside a trial can leave
    # the endpoint equation without a root. Acceptance updates the next step.
    if update_engagement != 0:
        if not covered:
            engaged[i] = 0
        elif engaged[i] != 0 or best <= engagement_radius * engagement_radius:
            engaged[i] = 1
    if engaged[i] == 0:
        return
    center = wp.transform_point(pose, body_com)
    velocity = wp.spatial_top(body_qd[body]) + wp.cross(wp.spatial_bottom(body_qd[body]), point - center)
    velocity -= tissue_velocity
    transverse = closest - point
    transverse -= wp.dot(transverse, tangent) * tangent
    sliding = wp.dot(velocity, tangent)
    normal_velocity = velocity - sliding * tangent
    normal_force = -lengths[i] * (stiffness * transverse + damping * normal_velocity)
    friction_magnitude = friction * wp.length(normal_force) * wp.abs(sliding) / wp.max(wp.abs(sliding), speed_scale)
    inverse_mass = _point_inverse_mass(tangent, point, center, pose, rigid_inverse_mass, rigid_inverse_inertia)
    for j in range(4):
        if fixed[nodes[i][j]] == 0:
            inverse_mass += bary[i][j] * bary[i][j] * particle_inverse_mass[nodes[i][j]]
    if limit_impulses != 0 and inverse_mass > 0.0:
        friction_magnitude = wp.min(friction_magnitude, wp.abs(sliding) / (dt[0] * inverse_mass))
    force = normal_force - friction_magnitude * wp.sign(sliding) * tangent
    _give_force(force, point, center, body, body_f, diagnostics)
    for j in range(4):
        reaction = -bary[i][j] * force
        wp.atomic_add(particle_f, nodes[i][j], reaction)
        if fixed[nodes[i][j]] != 0:
            wp.atomic_add(diagnostics, 4, reaction)
            wp.atomic_add(diagnostics, 5, wp.cross(q[nodes[i][j]], reaction))
        else:
            wp.atomic_add(diagnostics, 2, reaction)
        wp.atomic_add(diagnostics, 3, wp.cross(q[nodes[i][j]], reaction))


def _closest_triangle(point, vertices, triangles):
    """Closest surface point and barycentrics; includes triangle edges."""
    tri = vertices[triangles]
    a, ab, ac = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    rel = point - a
    aa, bb, cc = np.sum(ab * ab, 1), np.sum(ab * ac, 1), np.sum(ac * ac, 1)
    rhs1, rhs2 = np.sum(rel * ab, 1), np.sum(rel * ac, 1)
    det = aa * cc - bb * bb
    u = (cc * rhs1 - bb * rhs2) / np.maximum(det, 1.0e-30)
    v = (aa * rhs2 - bb * rhs1) / np.maximum(det, 1.0e-30)
    bary = np.stack((1 - u - v, u, v), axis=1)
    candidates = np.einsum("ij,ijk->ik", bary, tri)
    distance = np.sum((candidates - point) ** 2, axis=1)
    distance[np.any(bary < 0, axis=1)] = np.inf
    for j, k in ((0, 1), (1, 2), (2, 0)):
        direction = tri[:, k] - tri[:, j]
        t = np.clip(np.sum((point - tri[:, j]) * direction, 1) / np.maximum(np.sum(direction**2, 1), 1e-30), 0, 1)
        candidate = tri[:, j] + t[:, None] * direction
        d = np.sum((candidate - point) ** 2, 1)
        update = d < distance
        candidates[update], distance[update] = candidate[update], d[update]
        bary[update] = 0
        bary[update, j], bary[update, k] = 1 - t[update], t[update]
    index = int(np.argmin(distance))
    return index, bary[index], candidates[index], float(np.sqrt(distance[index]))


def _vertex_bounds(points):
    """Coordinate bounds of triangle/tetrahedron vertices without a strided reduction."""
    low = np.minimum(points[:, 0], points[:, 1])
    high = np.maximum(points[:, 0], points[:, 1])
    for vertex in range(2, points.shape[1]):
        np.minimum(low, points[:, vertex], out=low)
        np.maximum(high, points[:, vertex], out=high)
    return low, high


def _point_aabb_candidates(point, low, high, padding=1e-8):
    # Preserve bounds +/- epsilon: moving epsilon onto the point changes ties.
    candidates = np.flatnonzero((point[0] >= low[:, 0] - padding) & (point[0] <= high[:, 0] + padding))
    if len(candidates) > len(low) // 4:
        return np.flatnonzero(np.all((point >= low - padding) & (point <= high + padding), axis=1))
    return candidates[np.all((point >= low[candidates] - padding) & (point <= high[candidates] + padding), axis=1)]


def _segment_aabb_candidates(start, end, low, high):
    query_low, query_high = np.minimum(start, end) - 1e-9, np.maximum(start, end) + 1e-9
    candidates = np.flatnonzero((high[:, 0] >= query_low[0]) & (low[:, 0] <= query_high[0]))
    # Gathering most of the mesh costs more than evaluating the full predicate.
    if len(candidates) > len(low) // 4:
        return np.flatnonzero(np.all((high >= query_low) & (low <= query_high), axis=1))
    return candidates[np.all((high[candidates] >= query_low) & (low[candidates] <= query_high), axis=1)]


def _surface_crossings(start, end, old_points, new_points, triangles, *, bounds=None, candidates=None):
    """Swept point / linearly moving triangle intersections during one step.

    Coplanarity is cubic in normalized time. Broad-phase swept AABBs select
    triangles before exact polynomial roots and barycentric containment tests.
    Return ordered (time, outward/inward, face) crossings, merging shared-edge hits.
    """
    if candidates is not None:
        candidates = np.asarray(candidates, dtype=np.int32)
        old, new = old_points[triangles[candidates]], new_points[triangles[candidates]]
        old_low, old_high = _vertex_bounds(old)
        new_low, new_high = _vertex_bounds(new)
        selected = _segment_aabb_candidates(start, end, np.minimum(old_low, new_low), np.maximum(old_high, new_high))
        candidates, old, new = candidates[selected], old[selected], new[selected]
    else:
        if bounds is None:
            old, new = old_points[triangles], new_points[triangles]
            old_low, old_high = _vertex_bounds(old)
            new_low, new_high = _vertex_bounds(new)
            low, high = np.minimum(old_low, new_low), np.maximum(old_high, new_high)
        else:
            low, high = bounds
        candidates = _segment_aabb_candidates(start, end, low, high)
        if bounds is None:
            old, new = old[candidates], new[candidates]
        else:
            old, new = old_points[triangles[candidates]], new_points[triangles[candidates]]
    crossings = []
    for local_index, index in enumerate(candidates):
        a, b, c = old[local_index]
        da, db, dc = new[local_index] - old[local_index]
        r, dr = start - a, end - start - da
        e, de, f, df = b - a, db - da, c - a, dc - da
        n0, n1, n2 = np.cross(e, f), np.cross(de, f) + np.cross(e, df), np.cross(de, df)
        coefficients = np.array([dr @ n2, r @ n2 + dr @ n1, r @ n1 + dr @ n0, r @ n0])
        scale = max(np.max(np.abs(coefficients)), 1e-30)
        significant = np.flatnonzero(np.abs(coefficients) > 1e-12 * scale)
        if not len(significant) or significant[0] == 3:
            continue
        for root in np.roots(coefficients[significant[0] :]):
            if abs(np.imag(root)) > 1e-7 or not 1e-7 < np.real(root) <= 1 + 1e-7:
                continue
            time = float(np.clip(np.real(root), 0, 1))
            relative, edge1, edge2 = r + time * dr, e + time * de, f + time * df
            gram = np.array([[edge1 @ edge1, edge1 @ edge2], [edge1 @ edge2, edge2 @ edge2]])
            if np.linalg.det(gram) <= 1e-28:
                continue
            uv = np.linalg.solve(gram, [relative @ edge1, relative @ edge2])
            if min(uv[0], uv[1], 1 - uv.sum()) < -1e-6:
                continue
            derivative = (3 * coefficients[0] * time + 2 * coefficients[1]) * time + coefficients[2]
            if abs(derivative) > 1e-12 * scale:
                crossings.append((time, 1 if derivative > 0 else -1, int(index)))
    unique = []
    for event in sorted(crossings):
        if not unique or abs(event[0] - unique[-1][0]) > 1e-5:
            unique.append(event)
    return unique


class NeedleTissueInteraction:
    """Force- and moment-balanced contact with a progressively embedded track.

    Call :meth:`add_forces` once at time n, advance FEM and MuJoCo through the
    same ``dt``, then call :meth:`accept` once with the resulting state. Reset by
    constructing a new interaction. No pose is prescribed by this class.
    ``sample_points_local`` must be an ordered shaft polyline, tip first.
    Stiffness is tip contact N/m and total shaft surface N/m; the tip is a
    separate contact sample and shaft quadrature weights sum to one.
    ``surface_sample_stiffness`` [N/m] is optional: every shaft sample is then
    at least this stiff, with its damping raised in the same ratio, so that a
    single sample can carry a load; the tip keeps ``stiffness``. The default
    ``None`` keeps the shared shaft stiffness alone, and
    :meth:`set_surface_sample_stiffness` changes it between steps.
    Surface normal damping is ``damping * penetration / sample_radius`` during
    compression and zero during separation, so its force is continuous at onset.
    Surface reactions blend nearby blocking facets over half the proxy radius,
    with deformed-area and compact distance/orientation weights. Facet normals
    approximate a finite contact footprint; coplanar facets recover the stated
    stiffness. Puncture uses the norm of their summed tip normal reactions.
    Shaft stiffness/damping are per metre of
    traversed material track. ``reference_to_active`` maps full-surface vertices
    to active particles, using -1 for inactive material.

    The interaction has no target, planned path, or terminal success state.
    ``events`` is replaced by each accepted step and contains actual ``entry``,
    ``exit``, and newly authorized ``puncture`` events. Entry/exit positions are
    the accepted tip position (within one resolved step of the moving surface).
    Old local openings and material track persist during withdrawal and re-entry.

    ``tunnel_model="alm"`` requires :class:`NeedleTissueCoupler`. It solves
    compression outside ``tunnel_clearance`` and projected Coulomb friction.
    ``tunnel_compliance_per_m`` [m^2/N] controls physical radial compliance;
    zero requests a hard wall. Its default is reciprocal shaft stiffness.
    ``tunnel_penalty_per_m`` [N/m^2] controls numerical dual convergence and
    defaults to shaft stiffness. Multipliers commit only with an accepted step.

    ``opening_capacity`` is the number of registered openings that the device storage holds from
    the start (a reserve, not a limit). A registration beyond it doubles the storage, which is
    counted in ``opening_storage_growths``, warned about once per growth, and makes captured
    graphs that read the storage capture once more.

    ``track_capacity`` is the number of track samples that the device storage holds from the
    start: a reserve as well. The default is one needle length of samples at ``track_spacing``
    and eight more, at most ``maximum_track_points``. A sample beyond it doubles the storage, up
    to ``maximum_track_points``, which is counted in ``track_storage_growths`` and warned about
    in the same way. ``maximum_track_points`` stays the only hard limit of the track.

    A swept tip test normally starts from the tissue positions that the last :meth:`accept` kept
    and committed to the device query. A caller that moves nodes of the accepted state afterwards
    (a contact that places nodes) makes the next step start from other positions. Such a test still
    takes the device candidates, for a query box grown by the largest difference of a coordinate
    between the start positions and the committed copy, and runs the exact test on them with the
    actual start positions (:meth:`_shifted_start_growth` states when). Per test of one surface,
    ``shifted_start_candidate_tests`` counts the tests that did, ``shifted_start_host_tests`` the
    tests that kept the complete host test, by reason, and ``shifted_start_largest_growth`` is the
    largest growth of a query box so far [m]. A step that starts from the committed copy changes
    none of them; a step without a committed copy (the first one) counts as a host test.
    """

    # The device arrays that hold one row per track sample, with their data types.
    _TRACK_STORAGE = (
        ("_track_nodes", wp.vec4i),
        ("_track_bary", wp.vec4),
        ("_track_bary_d", wp.vec4d),
        ("_track_lengths", float),
        ("_track_engaged", int),
        ("_track_points", wp.vec3),
    )

    _TRIAL_FIELDS = (
        "status",
        "reason",
        "punctured",
        "_opening",
        "_opening_reused",
        "_pending_opening",
        "_pending_opening_face",
        "_pending_reused",
        "_tip_embedding",
        "_step_start_tip",
        "_step_start_q",
        "_step_start_samples",
    )

    def __init__(
        self,
        vertices,
        tetrahedra,
        exterior_triangles,
        fixed_indices,
        sample_points_local,
        sample_radii,
        tip_local,
        *,
        device=None,
        reference_vertices=None,
        reference_triangles=None,
        reference_to_active=None,
        stiffness=80.0,
        damping=0.025,
        surface_sample_stiffness=None,
        puncture_force=0.003,
        shaft_stiffness_per_m=1500.0,
        shaft_damping_per_m=0.5,
        friction=0.15,
        track_spacing=0.0003,
        speed_scale=0.0002,
        maximum_track_points=4096,
        opening_capacity=16,
        track_capacity=None,
        particle_inverse_mass=None,
        needle_mass=None,
        needle_inertia=None,
        tunnel_model="penalty",
        tunnel_clearance=2e-5,
        tunnel_compliance_per_m=None,
        tunnel_penalty_per_m=None,
        tunnel_tolerance=2e-6,
    ):
        if tunnel_model not in ("penalty", "alm"):
            raise ValueError("Tunnel model must be 'penalty' or 'alm'")
        if (
            tunnel_model == "alm"
            and shaft_stiffness_per_m <= 0
            and (tunnel_compliance_per_m is None or tunnel_penalty_per_m is None)
        ):
            raise ValueError("ALM needs positive shaft stiffness or explicit tunnel compliance and penalty")
        self.device = wp.get_device(device)
        # The time step that the force kernels read as device data. A coupler replaces
        # this object by the one of its step, so all stages of that step read the same arrays.
        self.step_time = StepTime(self.device)
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.tetrahedra = np.asarray(tetrahedra, dtype=np.int32).reshape(-1, 4)
        self.exterior_triangles = np.asarray(exterior_triangles, dtype=np.int32).reshape(-1, 3)
        self.samples = np.asarray(sample_points_local, dtype=np.float64)
        self.radii = np.broadcast_to(np.asarray(sample_radii, dtype=float), (len(self.samples),)).copy()
        self.tip_local = np.asarray(tip_local, dtype=float)
        if self.vertices.ndim != 2 or self.vertices.shape[1:] != (3,) or not np.isfinite(self.vertices).all():
            raise ValueError("Tissue vertices must be finite xyz coordinates")
        if not len(self.tetrahedra) or not len(self.exterior_triangles):
            raise ValueError("Insertion requires active tetrahedra and their exterior surface")
        if self.tetrahedra.min() < 0 or self.tetrahedra.max() >= len(self.vertices):
            raise ValueError("Tetrahedra reference missing particles")
        parameters = (
            stiffness,
            damping,
            puncture_force,
            shaft_stiffness_per_m,
            shaft_damping_per_m,
            friction,
            track_spacing,
            speed_scale,
        )
        if (
            not np.isfinite(parameters).all()
            or min(parameters) < 0
            or min(stiffness, puncture_force, track_spacing, speed_scale) <= 0
        ):
            raise ValueError("Insertion parameters must be finite and nonnegative; thresholds and scales positive")
        if surface_sample_stiffness is not None and not (
            np.isfinite(surface_sample_stiffness) and surface_sample_stiffness > 0
        ):
            raise ValueError("surface_sample_stiffness must be finite and positive, or None")
        if (
            len(self.samples) < 2
            or self.samples.shape[1:] != (3,)
            or not np.isfinite(self.samples).all()
            or not np.isfinite(self.radii).all()
            or np.any(self.radii <= 0)
        ):
            raise ValueError("Needle needs a finite ordered polyline and positive sample radii")
        if self.tip_local.shape != (3,) or not np.allclose(self.tip_local, self.samples[0], atol=1e-8, rtol=0):
            raise ValueError("tip_local must equal the first ordered needle sample")
        self.stiffness, self.damping, self.puncture_force = stiffness, damping, puncture_force
        self.shaft_stiffness_per_m, self.shaft_damping_per_m = shaft_stiffness_per_m, shaft_damping_per_m
        self.friction, self.track_spacing, self.speed_scale = (
            friction,
            track_spacing,
            speed_scale,
        )
        self.maximum_track_points = int(maximum_track_points)
        if self.maximum_track_points < 2:
            raise ValueError("maximum_track_points must be at least two")
        self.opening_capacity = int(opening_capacity)
        if self.opening_capacity < 1:
            raise ValueError("opening_capacity must be at least one")
        self.opening_storage_growths = 0
        self.surface_vertices = (
            self.vertices.copy() if reference_vertices is None else np.asarray(reference_vertices, dtype=float).copy()
        )
        self.surface_triangles = (
            self.exterior_triangles if reference_triangles is None else np.asarray(reference_triangles, dtype=np.int32)
        )
        self.mapping = (
            np.arange(len(self.vertices), dtype=np.int32)
            if reference_to_active is None
            else np.asarray(reference_to_active, dtype=np.int32)
        )
        if len(self.mapping) != len(self.surface_vertices):
            raise ValueError("reference_to_active must map every reference surface vertex")
        if np.any(self.mapping < -1) or np.any(self.mapping >= len(self.vertices)):
            raise ValueError("reference_to_active must contain active particle indices or -1")
        self._live_surface_vertices = np.flatnonzero(self.mapping >= 0)
        self._dynamic_surface_faces = np.flatnonzero(np.any(self.mapping[self.surface_triangles] >= 0, axis=1))
        self._surface_geometry = None
        self._surface_bounds = None
        self.fixed = np.zeros(len(self.vertices), np.int32)
        self.fixed[np.asarray(fixed_indices, dtype=np.int32)] = 1
        lengths = np.linalg.norm(np.diff(self.samples, axis=0), axis=1)
        if np.any(lengths <= 0):
            raise ValueError("Consecutive needle samples must be distinct")
        self._sample_arc = np.r_[0.0, np.cumsum(lengths)]
        weights = np.r_[lengths[0] / 2, (lengths[:-1] + lengths[1:]) / 2, lengths[-1] / 2]
        weights[0] = 0.0
        weights /= weights.sum()
        weights[0] = 1.0
        self._surface_shares = weights
        self.surface_sample_stiffness = surface_sample_stiffness
        self._samples = wp.array(self.samples, dtype=wp.vec3, device=self.device)
        self._radii = wp.array(self.radii, dtype=float, device=self.device)
        self._weights = wp.array(self._surface_weights(), dtype=float, device=self.device)
        self._mapping = wp.array(self.mapping, dtype=int, device=self.device)
        self._opening_faces = np.zeros(len(self.surface_triangles), dtype=np.int32)
        self._openings = []
        self._opening_masks = []
        # Registered openings, one row each, allocated once at the reserve and written in place
        # (_register_opening): face masks, nodes and weights; _opening_points is scratch for their
        # current positions. Rows from the live count on (LIVE_OPENINGS) hold zeros and are not read.
        self._opening_points = wp.zeros(self.opening_capacity, dtype=wp.vec3, device=self.device)
        self._all_opening_faces = wp.zeros(
            (self.opening_capacity, len(self.surface_triangles)), dtype=int, device=self.device
        )
        self._opening_nodes = wp.zeros(self.opening_capacity, dtype=wp.vec3i, device=self.device)
        self._opening_bary_d = wp.zeros(self.opening_capacity, dtype=wp.vec3d, device=self.device)
        self._sample_in_visited_material = wp.zeros(len(self.samples), dtype=int, device=self.device)
        self._tet_indices = {tuple(ids): i for i, ids in enumerate(self.tetrahedra)}
        # The provided render/exterior surface may omit active/inactive cuts.
        # Material continuity must use the complete boundary of active cells.
        self._active_triangles = boundary_faces(self.tetrahedra)[0]
        self._visited_tets = set()
        # Sorted ids of the visited tetrahedra in the first LIVE_VISITED entries (_sync_visited).
        # One entry per tetrahedron: a set of ids cannot outgrow it.
        self._visited = wp.zeros(len(self.tetrahedra), dtype=int, device=self.device)
        self._visited_uploaded = set()
        self._fixed = wp.array(self.fixed, dtype=int, device=self.device)
        inverse_mass = (
            np.zeros(len(self.vertices)) if particle_inverse_mass is None else np.asarray(particle_inverse_mass)
        )
        if (
            inverse_mass.shape != (len(self.vertices),)
            or not np.isfinite(inverse_mass).all()
            or np.any(inverse_mass < 0)
        ):
            raise ValueError("particle_inverse_mass must contain finite nonnegative nodal inverse masses")
        self._particle_inverse_mass = wp.array(inverse_mass, dtype=float, device=self.device)
        self._rigid_inverse_mass, inverse_inertia = 0.0, np.zeros((3, 3))
        if needle_mass is not None or needle_inertia is not None:
            inertia = np.asarray(needle_inertia, dtype=float)
            inertia = np.diag(inertia) if inertia.shape == (3,) else inertia
            if (
                needle_mass is None
                or not np.isfinite(needle_mass)
                or needle_mass <= 0
                or inertia.shape != (3, 3)
                or not np.isfinite(inertia).all()
                or np.min(np.linalg.eigvalsh(inertia)) <= 0
            ):
                raise ValueError("Needle impulse limiting needs positive mass and positive definite local inertia")
            self._rigid_inverse_mass, inverse_inertia = 1 / needle_mass, np.linalg.inv(inertia)
        self._rigid_inverse_inertia = wp.mat33(*inverse_inertia.reshape(-1))
        self._surface = wp.array(self.surface_vertices, dtype=wp.vec3, device=self.device)
        self._triangles = wp.array(self.surface_triangles, dtype=wp.vec3i, device=self.device)
        self.mesh = wp.Mesh(self._surface, wp.array(self.surface_triangles.reshape(-1), dtype=int, device=self.device))
        self._active_surface = wp.array(self.vertices, dtype=wp.vec3, device=self.device)
        self._active_mesh = wp.Mesh(
            self._active_surface, wp.array(self._active_triangles.reshape(-1), dtype=int, device=self.device)
        )
        self._diagnostics = wp.zeros(7, dtype=wp.vec3, device=self.device)
        self._contact_flags = wp.zeros(2, dtype=int, device=self.device)
        self._track_nodes_host, self._track_bary_host, self._track_rest = [], [], []
        self._track_edges = set()
        self._track_sections = []
        self._track_anchor = None
        self._active_section = 0
        self._next_section = 0
        # Track samples, one row each, allocated once at the reserve and written in place
        # (_upload_track): nodes, weights in single and in double precision, quadrature lengths
        # and engagement; _track_points is scratch for their current positions. Rows from the
        # live count on (LIVE_TRACK) hold zeros. No kernel that writes history is launched over
        # them, and a kernel launched over the reserve returns there at its bound or at the zero
        # engagement.
        if track_capacity is None:
            # One needle length of samples (a passage cannot lay more while the tip follows the
            # needle's own arc) and a reserve of eight for the sections of a passage.
            track_capacity = int(np.floor(self._sample_arc[-1] / track_spacing)) + 8
        self.track_capacity = min(self.maximum_track_points, int(track_capacity))
        if self.track_capacity < 1:
            raise ValueError("track_capacity must be at least one")
        self.track_storage_growths = 0
        # Number of samples whose rows are on the device.
        self._track_uploaded = 0
        for name, dtype in self._TRACK_STORAGE:
            setattr(self, name, wp.zeros(self.track_capacity, dtype=dtype, device=self.device))
        # engaged_track_length as accept() read it after its update of the engagement, or as a
        # reader between accepted steps had it kept (keep_engaged_track_length), with the two
        # arrays it was read from: (engaged, lengths, value). None when the engagement was
        # written after that read, or no reader asked for it (request_engaged_track_length).
        # It is served for these two arrays only, and every kernel that writes the engagement
        # sets it to None; a write in place that bypasses this class must do the same.
        self._engaged_length = None
        self._engaged_length_requested = False
        # Live counts [visited tetrahedra, track samples, openings] as device data (LIVE_*): the
        # kernels read their bounds over the history from this array. It is never replaced.
        self._live = wp.zeros(3, dtype=int, device=self.device)
        self._live_written = (0, 0, 0)
        self.status, self.reason = "approach", ""
        self.punctured, self.tip_inside, self.puncture_count = False, False, 0
        self.entry_count = self.exit_count = 0
        self.events = ()
        self._opening = None
        self._opening_reused = False
        self._pending_opening = None
        self._pending_reused = False
        self._tip_embedding = None
        self._previous_tip = None
        self._previous_q = None
        self._previous_samples = None
        self._previous_surface = None
        self._previous_surface_bounds = None
        self._previous_active_bounds = None
        self._step_start_tip = self._step_start_q = None
        self._step_start_samples = None
        # Swept tip crossings of the candidate step that check_step_resolution passed last, with
        # the four arrays they were computed from: (start tip, tip, start nodes, nodes, surface
        # crossings, active crossings). accept() takes them once, for a step with equal arrays.
        self._checked_crossings = None
        # Swept tip tests whose start positions are not the committed copy (class docstring), per
        # test of one surface: with the device candidates of the grown box; with the complete host
        # test, {reason: count}; and the largest growth of a query box so far [m]. Host values.
        self.shifted_start_candidate_tests = 0
        self.shifted_start_host_tests = {}
        self.shifted_start_largest_growth = 0.0
        # The committed copy (_previous_q, the object) from which the device query measured the
        # motion of the positions it describes (_device_query_positions). None while it measured
        # none. accept() replaces _previous_q, so a motion measured before is told by identity.
        self._device_motion_start = None
        self._trial_fields = None
        self._pending_opening_face = None
        self._ever_inside = False
        self._in_material = False
        # The surface face at which the last accepted step held a change of side
        # (_side_change_held), else None. Written by accept() alone, like _in_material: a trial,
        # its retention and a rejected candidate leave it with the accepted state. A reset makes
        # a new interaction.
        self._held_face = None
        self.last_tip_force = 0.0
        self.emergence_position = None
        self.interface_force = np.zeros(3)
        self.interface_torque_origin = np.zeros(3)
        self.interface_torque = np.zeros(3)
        self.tissue_force = np.zeros(3)
        self.support_force = np.zeros(3)
        self.support_torque = np.zeros(3)
        self.tip_position = None
        self.force_balance_error = self.torque_balance_error = 0.0
        self._query_positions = None
        self._device_queries = None
        self._host_sample_permissions = None
        self._host_sample_signature = None
        self._host_sample_owners = None
        self._device_query_positions = None
        self._device_locations = {}
        self._device_query_points = None
        self._refresh_surface(self.vertices)
        active_reference = self.vertices[self._active_triangles]
        self._active_reference_bounds = _vertex_bounds(active_reference)
        self.tunnel_alm = None
        if tunnel_model == "alm":
            from .tunnel_alm import _TunnelALM  # noqa: PLC0415

            self.tunnel_alm = _TunnelALM(
                self,
                penalty_per_m=shaft_stiffness_per_m if tunnel_penalty_per_m is None else tunnel_penalty_per_m,
                compliance_per_m=1.0 / shaft_stiffness_per_m
                if tunnel_compliance_per_m is None
                else tunnel_compliance_per_m,
                clearance=tunnel_clearance,
                tolerance=tunnel_tolerance,
            )

    def _surface_weights(self):
        """Surface weight of every sample: its share, raised to the optional least sample stiffness."""
        weights = self._surface_shares.copy()
        if self.surface_sample_stiffness is not None:
            # The kernel scales stiffness and damping by a sample's weight, so
            # a floor on the weight is a floor on both. The tip keeps its own.
            weights[1:] = np.maximum(weights[1:], self.surface_sample_stiffness / self.stiffness)
        return weights

    def set_surface_sample_stiffness(self, value):
        """Change ``surface_sample_stiffness`` between steps; ``None`` restores the shared stiffness."""
        if value is not None and not (np.isfinite(value) and value > 0):
            raise ValueError("surface_sample_stiffness must be finite and positive, or None")
        self.surface_sample_stiffness = value
        self._weights.assign(self._surface_weights())

    @property
    def track_count(self):
        return len(self._track_nodes_host)

    @property
    def opening_count(self):
        """Number of registered openings."""
        return len(self._openings)

    def _write_live(self):
        """Write the live history counts to the device when one differs from the value written last (host to device).

        The visited count is the number of ids that :meth:`_sync_visited` uploaded and the track
        count the number of rows that :meth:`_upload_track` wrote, so a kernel never reads an
        entry of ``_visited`` or a track row that was not written.
        """
        counts = (len(self._visited_uploaded), self._track_uploaded, len(self._openings))
        if counts != self._live_written:
            self._live.assign(np.asarray(counts, dtype=np.int32))
            self._live_written = counts

    def _sync_visited(self):
        """Upload the sorted ids of the visited tetrahedra when the host set changed, then the live counts.

        Host to device only. ``accept`` calls it after its insert; the preparation of a trial-force
        helper calls it too, which covers a set that was edited without an accepted step.
        """
        if self._visited_tets != self._visited_uploaded:
            ids = np.asarray(sorted(self._visited_tets), dtype=np.int32)
            if len(ids):
                # In place: entries from the live count on keep what they held and are not read.
                wp.copy(self._visited, wp.array(ids, dtype=int, device="cpu", copy=False), count=len(ids))
            self._visited_uploaded = set(self._visited_tets)
        self._write_live()

    @property
    def track_length(self):
        """Total rest-material length of unique track edges; excludes gaps."""
        return float(sum(np.linalg.norm(self._track_rest[a] - self._track_rest[b]) for a, b in self._track_edges))

    @property
    def engaged_track_length(self):
        """Rest-length quadrature currently transmitting shaft interaction [m]."""
        if not self.track_count:
            return 0.0
        stored = self._engaged_length
        if stored is not None and stored[0] is self._track_engaged and stored[1] is self._track_lengths:
            return stored[2]
        return self._read_engaged_track_length()

    def _read_engaged_track_length(self):
        """Read the engaged track length from the device (two reads); the live rows are summed."""
        count = self.track_count
        lengths, engaged = self._track_lengths.numpy()[:count], self._track_engaged.numpy()[:count]
        return float(np.sum(lengths[engaged != 0]))

    def request_engaged_track_length(self):
        """Have the next :meth:`accept` read :attr:`engaged_track_length` for the step after it.

        For a reader that needs the length once per step, before the step's trials. The length
        changes where an accepted step updates the engagement; read there, it waits for nothing
        but that update, and the reader's query in the following step is answered from it. A
        device read at that later query would wait for all device work issued since the step was
        accepted. One request serves one accepted step. Without a request :meth:`accept` reads
        nothing, and the property reads the device whenever it is queried.
        """
        self._engaged_length_requested = True

    def keep_engaged_track_length(self):
        """:attr:`engaged_track_length`, kept from this query until the engagement is next written.

        For a reader between accepted steps that asked :meth:`accept` for nothing. A kept value
        answers it; otherwise the device is read here and that value is kept. A second attempt
        of the same step, after a rejected one, is then answered without a read: a rejected
        attempt accepts nothing, so the engagement is the one that was read. Every write of the
        engagement drops the value, as it drops the one that :meth:`accept` read.
        """
        length = self.engaged_track_length
        if self.track_count:
            self._engaged_length = (self._track_engaged, self._track_lengths, length)
        return length

    @property
    def track_section_ids(self):
        return np.asarray(self._track_sections, dtype=np.int32)

    @property
    def track_section_count(self):
        return len(set(self._track_sections))

    def track_rows(self, start=0):
        """The track samples from row ``start`` on, as the host lists hold them: nothing is read from the device.

        Rows are in the order in which the tip laid them, and a row once recorded does not change.

        Returns:
            Per row: the index of its tetrahedron (in the tetrahedra this interaction was given),
            its four barycentric weights in the order of that tetrahedron's vertices, its rest
            position [m] and its section (one per entry of the tip into material).
        """
        rows = slice(max(int(start), 0), self.track_count)
        tetrahedra = np.asarray([self._tet_indices[tuple(ids)] for ids in self._track_nodes_host[rows]], dtype=np.int32)
        weights = np.asarray(self._track_bary_host[rows], dtype=float).reshape(-1, 4)
        rest = np.asarray(self._track_rest[rows], dtype=float).reshape(-1, 3)
        return tetrahedra, weights, rest, np.asarray(self._track_sections[rows], dtype=np.int32)

    def _device_geometry_matches(self, q):
        return self._device_query_positions is not None and np.array_equal(q, self._device_query_positions)

    def _embedding_from_candidates(self, point, q, candidates):
        if not len(candidates):
            return None
        t = q[self.tetrahedra[candidates]]
        low, high = _vertex_bounds(t)
        selected = _point_aabb_candidates(point, low, high)
        candidates, t = candidates[selected], t[selected]
        if not len(candidates):
            return None
        matrices = np.transpose(t[:, 1:] - t[:, :1], (0, 2, 1))
        determinants = np.linalg.det(matrices)
        valid = np.abs(determinants) > 1e-18
        candidates = candidates[valid]
        if not len(candidates):
            return None
        coordinates = np.linalg.solve(matrices[valid], (point - t[valid, 0])[..., None])[..., 0]
        bary = np.c_[1 - coordinates.sum(axis=1), coordinates]
        inside = np.flatnonzero(np.min(bary, axis=1) >= -1e-6)
        if not len(inside):
            return None
        index = inside[np.argmax(np.min(bary[inside], axis=1))]
        return self.tetrahedra[candidates[index]], bary[index]

    def _force_query_points(self, tip, samples):
        # The device shaft mask needs geometry, not host sample embeddings.
        # Locate only the tip initially; an exact CPU mask fallback prepares
        # the full batch below before it requests any shaft embeddings.
        # Customized mask/query methods keep the complete original batch.
        if type(self) is NeedleTissueInteraction and all(
            getattr(getattr(self, name), "__func__", None) is getattr(NeedleTissueInteraction, name)
            for name in (
                "_visited_sample_mask",
                "_device_visited_sample_mask",
                "_tip_boundary_permission",
                "_prepare_device_queries",
                "_locate",
            )
        ):
            return np.asarray(tip).reshape(1, 3)
        return np.vstack((tip, samples))

    def _prepare_device_queries(self, state, q, points):
        if self._device_queries is None:
            self._device_queries = NeedleQueryGeometry(self.tetrahedra, len(self.vertices), device=self.device)
        if self._query_positions is not None and not np.array_equal(q, self._query_positions):
            self._query_positions = None
        self.surface_vertices[self._live_surface_vertices] = q[self.mapping[self._live_surface_vertices]]
        unchanged = self._device_geometry_matches(q)
        if not unchanged:
            self._device_queries.update(state.particle_q)
            self._device_query_positions = q.copy()
            # update() measured the motion of q from the positions the query holds as committed.
            self._device_motion_start = self._previous_q if self._device_queries.has_previous else None
        if unchanged and self._device_query_points is not None and np.array_equal(points, self._device_query_points):
            return
        self._device_query_points = points.copy()
        candidates = self._device_queries.point_candidates(points)
        candidates = [np.arange(len(self.tetrahedra)) if ids is None else ids for ids in candidates]
        self._device_locations = dict.fromkeys(map(tuple, points))
        sizes = np.array([len(ids) for ids in candidates])
        if not sizes.sum():
            return
        query_ids = np.repeat(np.arange(len(points)), sizes)
        tet_ids = np.concatenate(candidates)
        t = q[self.tetrahedra[tet_ids]]
        low, high = _vertex_bounds(t)
        keep = np.all((points[query_ids] >= low - 1e-8) & (points[query_ids] <= high + 1e-8), axis=1)
        query_ids, tet_ids, t = query_ids[keep], tet_ids[keep], t[keep]
        matrices = np.transpose(t[:, 1:] - t[:, :1], (0, 2, 1))
        valid = np.abs(np.linalg.det(matrices)) > 1e-18
        query_ids, tet_ids = query_ids[valid], tet_ids[valid]
        if not len(query_ids):
            return
        coordinates = np.linalg.solve(matrices[valid], (points[query_ids] - t[valid, 0])[..., None])[..., 0]
        bary = np.c_[1 - coordinates.sum(axis=1), coordinates]
        minimum = np.min(bary, axis=1)
        inside = minimum >= -1e-6
        query_ids, tet_ids, bary, minimum = query_ids[inside], tet_ids[inside], bary[inside], minimum[inside]
        # Candidates are sorted by original tet ID. np.argmax therefore keeps
        # the same first-tet tie break as the complete CPU query.
        for index in np.unique(query_ids):
            group = np.flatnonzero(query_ids == index)
            best = group[np.argmax(minimum[group])]
            self._device_locations[tuple(points[index])] = self.tetrahedra[tet_ids[best]], bary[best]

    def _locate(self, point, q):
        if self._device_geometry_matches(q) and tuple(point) in self._device_locations:
            return self._device_locations[tuple(point)]
        self._refresh_surface(q)
        candidates = _point_aabb_candidates(point, *self._tet_bounds)
        return self._embedding_from_candidates(point, q, candidates)

    def _shifted_start_growth(self, old_q):
        """Growth of the device query box for start positions that are not the committed copy, or why there is none.

        Returns ``(growth, None)`` or ``(None, reason)``. ``growth`` [m] is the largest difference
        of a coordinate between ``old_q`` and the committed copy (``_previous_q``) over the
        particles of the tissue, measured here for this step. The device query
        (``NeedleQueryGeometry``) pads its box, on each axis, by the largest motion from the
        committed copy to the end positions. On every axis a start vertex is within ``growth`` of
        the committed copy, so it is within ``growth`` plus that motion of its end position, and so
        is every position between. A triangle whose swept bounds (start and end, as the exact test
        takes them) reach the tip's box therefore has end bounds that reach the box grown by
        ``growth`` and the motion, which is the box the device tests.

        That holds only while the device measured the motion of the end positions from the very
        committed copy the host holds: ``_device_motion_start`` is that copy. No committed copy, a
        motion measured from an earlier one, another particle count, or a difference that is not a
        finite number give no growth; the caller then keeps the complete host test.
        """
        previous = self._previous_q
        if previous is None:
            return None, "no committed copy"
        if self._device_queries is None or self._device_motion_start is not previous:
            return None, "device motion not measured from the committed copy"
        if old_q.shape != previous.shape:
            return None, "changed particle count"
        # The device measures the motion of these particles (the tissue's), and the test reads no other.
        count = len(self.vertices)
        difference = old_q[:count] - previous[:count]
        growth = float(np.abs(difference, out=difference).max())
        if not np.isfinite(growth):
            return None, "start difference not finite"
        return growth, None

    def _material_crossings(self, start, end, old_q, q, *, active=False):
        triangles = self._active_triangles if active else self.surface_triangles
        if active:
            old_points, points = old_q, q
        else:
            points = self.surface_vertices
            if self._previous_surface is not None and np.array_equal(old_q, self._previous_q):
                old_points = self._previous_surface
            elif old_q is q:
                old_points = points
            else:
                old_points = self.surface_vertices.copy()
                old_points[self._live_surface_vertices] = old_q[self.mapping[self._live_surface_vertices]]
        candidates = None
        bounds = None
        if self._device_geometry_matches(q):
            static = old_q is q
            if static or (self._previous_q is not None and np.array_equal(old_q, self._previous_q)):
                mesh = self._active_mesh if active else self.mesh
                candidates = self._device_queries.surface_candidates(
                    mesh, np.minimum(start, end) - 1e-9, np.maximum(start, end) + 1e-9, swept=not static
                )
            else:
                # The step starts from positions that are not the committed copy. The device
                # candidates of the box grown by the measured difference hold every triangle the
                # exact test needs (_shifted_start_growth); the exact test below reads the actual
                # start positions. One device read: the candidate packet, which decides the
                # triangles of that test and, when it overflowed, the complete host test.
                growth, reason = self._shifted_start_growth(old_q)
                if growth is not None:
                    mesh = self._active_mesh if active else self.mesh
                    candidates = self._device_queries.surface_candidates(
                        mesh,
                        np.minimum(start, end) - 1e-9 - growth,
                        np.maximum(start, end) + 1e-9 + growth,
                        swept=True,
                    )
                    if candidates is None:
                        reason = "candidate overflow"
                    else:
                        self.shifted_start_candidate_tests += 1
                        self.shifted_start_largest_growth = max(self.shifted_start_largest_growth, growth)
                if candidates is None:
                    self.shifted_start_host_tests[reason] = self.shifted_start_host_tests.get(reason, 0) + 1
        else:
            current = self._active_bounds if active else self._surface_bounds
            previous = self._previous_active_bounds if active else self._previous_surface_bounds
            if old_q is q:
                bounds = current
            elif previous is not None and np.array_equal(old_q, self._previous_q):
                bounds = (np.minimum(previous[0], current[0]), np.maximum(previous[1], current[1]))
        return _surface_crossings(start, end, old_points, points, triangles, bounds=bounds, candidates=candidates)

    @property
    def step_resolution(self):
        """Largest needle motion of one accepted step [m]: needle radius and track spacing."""
        return min(float(self.radii.min()), self.track_spacing) * 1.01

    @property
    def capture_radius(self):
        """Radius of an opening of the needle's track [m]: twice the largest wire radius.

        The rule of this class wherever it asks whether a point belongs to an opening (written out
        there); here for a caller that has to know where the hole of the needle ends.
        """
        return 2 * float(self.radii.max())

    def _body_tip(self, state, body, pose=None):
        pose = state.body_q.numpy()[body] if pose is None else pose
        return np.asarray(wp.transform_point(wp.transform(*pose), wp.vec3(*self.tip_local)), dtype=float)

    def _body_samples(self, state, body, pose=None):
        pose = state.body_q.numpy()[body] if pose is None else pose
        rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
        return self.samples @ rotation.T + pose[:3]

    def _opening_position(self, q):
        if self._opening is None:
            return np.zeros(3)
        ids, bary = self._opening
        return bary @ q[ids]

    def _register_opening(self, ids, bary, face):
        # The faces of an opening share a vertex with the punctured face and face its way. Only
        # those faces need a normal: the unit normals of the connected faces are computed with
        # the operations that gave them for every face of the surface, and no other face can
        # enter the mask. A face whose three vertices are tissue nodes has all its connected
        # faces among the faces with a tissue node, which is where they are looked for; both
        # lists are ascending, so the connected faces are the ones a search of every face finds.
        triangles = self.surface_triangles
        if np.all(self.mapping[triangles[face]] >= 0):
            candidates = self._dynamic_surface_faces
            connected = candidates[np.any(np.isin(triangles[candidates], triangles[face]), axis=1)]
        else:
            connected = np.flatnonzero(np.any(np.isin(triangles, triangles[face]), axis=1))
        geometry = self.surface_vertices[triangles[connected]]
        normals = np.cross(geometry[:, 1] - geometry[:, 0], geometry[:, 2] - geometry[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
        mask = np.zeros(len(triangles), dtype=np.int32)
        # The punctured face is one of its own connected faces; the list is sorted.
        mask[connected] = normals @ normals[np.searchsorted(connected, face)] > 0.95
        row = len(self._openings)
        if row == self.opening_capacity:
            self._grow_opening_storage()
        self._openings.append((ids.copy(), bary.copy()))
        self._opening_masks.append(mask)
        # One row of each array, host to device, at the row's own offset: no storage is replaced.
        for storage, values, width in (
            (self._all_opening_faces, mask, len(mask)),
            (self._opening_nodes, np.asarray(ids, dtype=np.int32).reshape(1, 3), 1),
            (self._opening_bary_d, np.asarray(bary, dtype=np.float64).reshape(1, 3), 1),
        ):
            source = wp.array(values, dtype=storage.dtype, device="cpu", copy=False)
            wp.copy(storage, source, dest_offset=row * width, count=width)
        self._write_live()
        return mask

    def _grow_opening_storage(self):
        """Double the opening storage: the reserve was too small. The live rows are copied device to device."""
        live, old = len(self._openings), self.opening_capacity
        self.opening_capacity = 2 * old
        faces = len(self.surface_triangles)
        for name, width in (
            ("_opening_points", 1),
            ("_all_opening_faces", faces),
            ("_opening_nodes", 1),
            ("_opening_bary_d", 1),
        ):
            storage = getattr(self, name)
            grown = wp.zeros((self.opening_capacity, *storage.shape[1:]), dtype=storage.dtype, device=self.device)
            if live:
                wp.copy(grown, storage, count=live * width)
            setattr(self, name, grown)
        self.opening_storage_growths += 1
        warnings.warn(
            f"Needle opening storage grew from {old} to {self.opening_capacity} rows at {live} openings; "
            "pass opening_capacity to reserve it at construction",
            stacklevel=3,
        )

    def _matching_opening(self, tip, face, q):
        radius = 2 * float(self.radii.max())
        for index, ((ids, bary), mask) in enumerate(zip(self._openings, self._opening_masks, strict=True)):
            if mask[face] and np.linalg.norm(tip - bary @ q[ids]) <= radius:
                return index
        return None

    def _refresh_surface(self, q):
        # Local queries share exact endpoint geometry. Reuse only while every
        # nodal position is unchanged, including final evaluation -> accept.
        if self._query_positions is not None and np.array_equal(q, self._query_positions):
            return
        live = self._live_surface_vertices
        self.surface_vertices[live] = q[self.mapping[live]]
        self._query_positions = q.copy()
        self._tet_geometry = q[self.tetrahedra]
        self._tet_bounds = _vertex_bounds(self._tet_geometry)
        if self._surface_geometry is None:
            self._surface_geometry = self.surface_vertices[self.surface_triangles]
            self._surface_bounds = _vertex_bounds(self._surface_geometry)
        else:
            # Reference faces without active vertices never move. Keep their
            # exact geometry and bounds; refresh every face touching an active
            # vertex, including fixed active vertices. Accepted-state bounds
            # remain independent copies, made by accept().
            faces = self._dynamic_surface_faces
            geometry = self.surface_vertices[self.surface_triangles[faces]]
            self._surface_geometry[faces] = geometry
            self._surface_bounds[0][faces], self._surface_bounds[1][faces] = _vertex_bounds(geometry)
        active_surface = q[self._active_triangles]
        self._active_bounds = _vertex_bounds(active_surface)

    def _tip_boundary_permission(self, tip, q):
        if not self.punctured or self._previous_tip is None:
            return 0
        if self._tip_embedding is not None:
            ids, bary = self._tip_embedding
            if np.any(self.fixed[ids] & (bary > 0.05)):
                return 0
        elif not self._in_material:
            return 0
        crossings = self._material_crossings(self._previous_tip, tip, self._previous_q, q)
        active_crossings = self._material_crossings(self._previous_tip, tip, self._previous_q, q, active=True)
        matching = len(crossings) == len(active_crossings) and all(
            a[1] == b[1] and abs(a[0] - b[0]) <= 1e-5 for a, b in zip(crossings, active_crossings, strict=True)
        )
        if self._in_material:
            if not crossings and not active_crossings and self._tip_embedding is not None:
                # Internal tetrahedral faces are not physical barriers, even
                # before acceptance records a newly entered cell.
                return 2
            if matching and len(crossings) == 1 and crossings[0][1] > 0 and self._tip_embedding is None:
                # Provisional release through the actual outgoing face; other
                # nearby gap walls retain their ordinary contact reactions.
                time, _, face = crossings[0]
                self._tip_exit_point = self._previous_tip + time * (tip - self._previous_tip)
                return -int(face) - 1
        elif matching and len(crossings) == 1 and crossings[0][1] < 0 and self._tip_embedding is not None:
            _, _, face = crossings[0]
            if self._matching_opening(tip, face, q) is not None:
                # The leading tip may enter through an already authorized
                # opening during an endpoint trial, before any track exists.
                # This suppresses only material-side facets; a front-facing
                # intact wall across a gap still receives contact. Acceptance
                # alone publishes the entry event and first material sample.
                return 2
        return 0

    # Methods that the device sample mask stands for: with any of them replaced, the host mask is used.
    _HOST_SAMPLE_METHODS = (
        "_visited_sample_mask",
        "_tip_boundary_permission",
        "_material_crossings",
        "_locate",
        "_body_samples",
        "_body_tip",
        "_device_geometry_matches",
        "_prepare_device_queries",
    )

    def _host_sample_methods_original(self):
        return type(self) is NeedleTissueInteraction and not any(
            getattr(getattr(self, name), "__func__", None) is not getattr(NeedleTissueInteraction, name)
            for name in self._HOST_SAMPLE_METHODS
        )

    def _host_sample_helper(self):
        """The device helper of the host sample mask, and whether this call built it.

        One helper serves one binding: the geometry of this interaction, its device meshes and
        its tetrahedron query. It is built at the first call for a binding and kept; a new
        track sample, opening or visited tetrahedron is no new binding.
        """
        # Retain owners as well as IDs: the helper copies immutable geometry,
        # so dropping an owner could otherwise let Python reuse its identity.
        owners = (
            self.samples,
            self._sample_arc,
            self.tetrahedra,
            self._active_triangles,
            self.fixed,
            self.mapping,
            self.vertices,
        )

        def signature():
            query = self._device_queries
            return (
                tuple(id(array) for array in owners),
                tuple(
                    (array.ptr, array.shape)
                    for array in (
                        self._fixed,
                        self._surface,
                        self._active_surface,
                        query.low,
                        query.high,
                    )
                ),
                self.mesh.id,
                self._active_mesh.id,
                query.bvh.id,
                id(query),
            )

        query = self._device_queries
        if (
            self._host_sample_permissions is not None
            and query is not None
            and query.bvh is not None
            and signature() == self._host_sample_signature
        ):
            return self._host_sample_permissions, False
        from .needle_trial_forces import _NeedleTrialForces

        # A helper creates the tetrahedron query of the interaction if there is none yet.
        self._host_sample_permissions = _NeedleTrialForces(self)
        self._host_sample_signature = signature()
        self._host_sample_owners = owners
        return self._host_sample_permissions, True

    def prepare_host_sample_helper(self):
        """Build, ahead of the first step, the device helper that the host sample mask uses with a track.

        For start-up. The mask builds this helper at its first evaluation with a track sample,
        which is in the middle of a run; this call builds it for the same binding, with its
        buffers and with the kernel modules of its launches loaded, so that the evaluation
        finds it. Nothing is evaluated.

        Returns:
            False, with nothing built, on a CPU device and for an interaction whose mask
            methods are replaced: the host mask then never uses a helper.
        """
        if not self.device.is_cuda or not self._host_sample_methods_original():
            return False
        self._host_sample_helper()
        return True

    def _device_visited_sample_mask(self, state, body, q, samples):
        """Use existing shaft geometry on CUDA; retain the exact host tip oracle."""
        if not self.device.is_cuda or not self._visited_tets or not self.track_count:
            return None
        from ...fem.differentiation import active_tape

        if (
            not self._host_sample_methods_original()
            or active_tape() is not None
            or any(
                array.requires_grad
                for array in (state.particle_q, state.particle_qd, state.body_q, state.body_qd, state.body_f)
            )
            or self._device_queries is None
            or not self._device_queries.has_previous
            or not self._device_geometry_matches(q)
        ):
            return None
        # The CPU oracle inverts every visited cell, including those away
        # from current samples. Retain its fallback/error behavior there too.
        visited = q[self.tetrahedra[np.asarray(sorted(self._visited_tets))]]
        determinants = np.linalg.det(np.transpose(visited[:, 1:] - visited[:, :1], (0, 2, 1)))
        if not np.isfinite(determinants).all() or np.any(np.abs(determinants) <= 1e-18):
            return None
        helper, built = self._host_sample_helper()
        if not built:
            helper.prepare(self)
        helper.evaluate(state, body, certify_tip=False)
        if helper.fallback.numpy()[0]:
            return None
        mask = helper.mask.numpy()
        self._tip_exit_point = np.zeros(3)
        capture = 2 * float(self.radii.max())
        # Preserve the original tip permission and outgoing-face location;
        # only lateral shaft association uses the already verified GPU path.
        if mask[0] == 0 or self.track_spacing > capture or not self._in_material:
            mask[0] = self._tip_boundary_permission(samples[0], q)
        return mask

    def _visited_sample_mask(self, state, body, q, *, samples=None):
        mask = np.zeros(len(self.samples), dtype=np.int32)
        self._tip_exit_point = np.zeros(3)
        samples = self._body_samples(state, body) if samples is None else samples
        if not self._visited_tets:
            mask[0] = self._tip_boundary_permission(samples[0], q)
            return mask
        tets = q[self.tetrahedra[np.asarray(sorted(self._visited_tets))]]
        matrices = np.transpose(tets[:, 1:] - tets[:, :1], (0, 2, 1))
        inverse = np.linalg.inv(matrices)
        low, high = _vertex_bounds(tets)
        # Keep sample-major / sorted-tet order and the original three-term
        # contraction, while gathering all AABB candidates in one host call.
        sample_ids, tet_ids = np.nonzero(
            np.all(
                (samples[:, None] >= low[None] - 1e-8) & (samples[:, None] <= high[None] + 1e-8),
                axis=2,
            )
        )
        if len(sample_ids):
            uvw = np.einsum("ijk,ik->ij", inverse[tet_ids], samples[sample_ids] - tets[tet_ids, 0])
            inside = (uvw.min(axis=1) >= -1e-5) & (uvw.sum(axis=1) <= 1 + 1e-5)
            mask[sample_ids[inside]] = 1
        # Lateral shaft motion must not encounter an artificial wall at an
        # internal tetrahedral face either. Extend permission only through a
        # continuous, local material segment from an actual traversed anchor.
        # A nearby disconnected wall or lip has intervening exterior crossings
        # and receives ordinary surface contact even within the capture radius.
        track = np.einsum("ij,ijk->ik", np.asarray(self._track_bary_host), q[np.asarray(self._track_nodes_host)])
        capture = 2 * float(self.radii.max())
        directions = np.diff(samples, axis=0)
        lengths = np.diff(self._sample_arc)
        fractions = np.einsum("ijk,jk->ij", track[:, None] - samples[None, :-1], directions) / lengths**2
        projections = samples[None, :-1] + np.clip(fractions, 0, 1)[..., None] * directions
        segments = np.argmin(np.sum((projections - track[:, None]) ** 2, axis=2), axis=1)
        fractions = fractions[np.arange(self.track_count), segments]
        arc = self._sample_arc[segments] + np.clip(fractions, 0, 1) * lengths[segments]
        covered = ~(((segments == 0) & (fractions < 0)) | ((segments == len(lengths) - 1) & (fractions > 1)))
        engaged = (self._track_engaged.numpy()[: self.track_count] != 0) & covered
        distances_by_sample = np.linalg.norm(track[None] - samples[:, None], axis=2)
        near_by_sample = distances_by_sample <= capture
        # Preserve physical-distance norms, arc distances, and the existing
        # finite-shaft engagement rule. Do not replace norms by squared tests.
        associated_by_sample = engaged[None] & (np.abs(arc[None] - self._sample_arc[:, None]) <= capture)
        needs_association = ~(mask.astype(bool) & np.any(near_by_sample, axis=1))
        needs_association &= np.any(near_by_sample | associated_by_sample, axis=1)
        needs_association[0] = False
        for i in np.flatnonzero(needs_association):
            distances = distances_by_sample[i]
            associated = associated_by_sample[i]
            anchors = np.flatnonzero(near_by_sample[i] | associated)
            embedding = self._locate(samples[i], q)
            if embedding is None:
                continue
            ids, bary = embedding
            if np.any(self.fixed[ids] & (bary > 0.05)):
                continue
            for anchor in anchors[np.argsort(distances[anchors])]:
                if not self._material_crossings(track[anchor], samples[i], q, q, active=True):
                    mask[i] = 4 if associated[anchor] else 1
                    break
        if mask[0] == 0 or self.track_spacing > capture or not self._in_material:
            mask[0] = self._tip_boundary_permission(samples[0], q)
        return mask

    def add_forces(
        self, state, body_index, body_com_local, external_forces, dt, *, trial=False, start_state=None, implicit=False
    ):
        """Accumulate balanced interface loads, preserving other applied forces.

        ``trial=True`` evaluates a candidate state without changing committed
        engagement, authorization, events, status, or accepted-step bookkeeping.
        Derived geometry and force diagnostics remain available to a coupling
        iteration. Temporary candidate penetration cannot cause a committed
        failure. The caller must clear its force buffers before each trial.

        After convergence, evaluate once with ``trial=False`` and
        ``start_state`` set to the actual time-n state, then :meth:`accept` the
        resulting time-(n+1) state exactly once. The force and puncture criterion
        use the supplied candidate state; swept event checks use ``start_state``.
        Omitting both options retains the explicit, single-evaluation interface.
        ``implicit=True`` evaluates the regularized Coulomb friction law without
        the local non-reversal impulse limits used for explicit force exchange.
        Shaft engagement is held at its accepted value throughout an implicit
        step; :meth:`accept` updates finite-shaft coverage and local re-entry
        for the following step. Explicit evaluation updates engagement directly.
        """
        if self.tunnel_alm is not None and not implicit:
            raise ValueError("Tunnel ALM requires implicit coupling; explicit force exchange supports penalty mode")
        self._trial_fields = None
        if trial:
            names = self._TRIAL_FIELDS
            saved = {name: getattr(self, name) for name in names}
            # Implicit force evaluation freezes engagement. Only the explicit
            # kernel can mutate it and needs a trial snapshot/restore.
            engagement = None if implicit else wp.clone(self._track_engaged)
            try:
                self._add_forces(state, body_index, body_com_local, external_forces, dt, implicit=implicit, trial=True)
                if implicit:
                    self._trial_fields = {name: getattr(self, name) for name in names}
            finally:
                for name, value in saved.items():
                    setattr(self, name, value)
                if engagement is not None:
                    wp.copy(self._track_engaged, engagement)
        else:
            self._add_forces(state, body_index, body_com_local, external_forces, dt, implicit=implicit)
            if start_state is not None:
                pose = start_state.body_q.numpy()[body_index]
                self._step_start_tip = self._body_tip(start_state, body_index, pose)
                self._step_start_q = start_state.particle_q.numpy().astype(float)
                self._step_start_samples = self._body_samples(start_state, body_index, pose)

    def retain_trial_forces(self, start_state, body_index):
        """Retain the last implicit trial's force context before accepting it.

        The caller must keep the candidate state unchanged after that trial.
        This avoids evaluating identical endpoint forces again. No material
        history is committed here: :meth:`accept` still performs the swept
        checks and publishes events exactly once. Rejected trials must call
        :meth:`discard_trial_forces`.
        """
        if self._trial_fields is None:
            raise RuntimeError("No implicit needle force trial is available to retain")
        for name, value in self._trial_fields.items():
            setattr(self, name, value)
        self._trial_fields = None
        pose = start_state.body_q.numpy()[body_index]
        self._step_start_tip = self._body_tip(start_state, body_index, pose)
        self._step_start_samples = self._body_samples(start_state, body_index, pose)
        self._step_start_q = start_state.particle_q.numpy().astype(float)

    def discard_trial_forces(self):
        """Discard only the uncommitted endpoint context after a rejected solve."""
        self._trial_fields = None

    def _publish_force_diagnostics(self, diagnostic, pose, body_com_local):
        """Publish the same force diagnostics for eager and certified trials."""
        self.interface_force, self.interface_torque_origin = diagnostic[:2]
        self.tissue_force, self.support_force = diagnostic[2], diagnostic[4]
        com = np.asarray(wp.transform_point(wp.transform(*pose), wp.vec3(*body_com_local)))
        self.interface_torque = self.interface_torque_origin - np.cross(com, self.interface_force)
        self.support_torque = diagnostic[5]
        self.force_balance_error = float(np.linalg.norm(diagnostic[0] + diagnostic[2] + diagnostic[4]))
        self.torque_balance_error = float(np.linalg.norm(diagnostic[1] + diagnostic[3]))
        self.last_tip_force = float(diagnostic[6, 0])

    def _stage_certified_trial(self, state, body_index, body_com_local, sample_mask):
        """Stage metadata after an event-free device trial and exact native audit.

        The caller must have a zero device fallback flag and unchanged nodal
        and bitwise-identical rigid coordinates/velocities. The device owns the
        current refitted geometry and law forces. This method performs no force
        evaluation or authorization; swept checks and acceptance still follow.
        Return False for inconsistent metadata so the caller restores and uses
        the original host-controlled transaction.
        """
        self._trial_fields = None
        if self.status == "failed" or self._device_queries is None or (self._in_material and not self.punctured):
            return False
        flags = self._contact_flags.numpy()
        diagnostic = self._diagnostics.numpy().astype(float)
        if flags[1] or np.any((flags < 0) | (flags > 1)) or not np.isfinite(diagnostic).all():
            return False
        if not self._in_material and diagnostic[6, 0] != 0:
            # Subthreshold elastic contact is certified only when the exact
            # opening-distance audit below also excludes puncture reuse.
            if not 0 < diagnostic[6, 0] < self.puncture_force * (1.0 - 1e-6):
                return False
        q = state.particle_q.numpy().astype(float)
        pose = state.body_q.numpy()[body_index]
        if not np.isfinite(q).all() or not np.isfinite(pose).all():
            return False
        tip = self._body_tip(state, body_index, pose)
        if not self._in_material:
            capture = 2 * float(self.radii.max())
            if self.punctured:
                # No new authorization: retain only the existing outside-tip
                # branch whose current opening is strictly within capture.
                # The GPU separately certifies transform/barycentric rounding
                # and swept clearance. Equality/uncertainty stays eager.
                if self._opening is None:
                    return False
                distance = np.linalg.norm(tip - self._opening_position(q))
                if not np.isfinite(distance) or distance >= capture - 1e-12:
                    return False
            elif any(np.linalg.norm(tip - bary @ q[ids]) <= capture + 1e-12 for ids, bary in self._openings):
                return False
        # Publish the device geometry certificate, then query only the tip for
        # exact CPU embedding/CCD metadata. Shaft association is already proven
        # by the device trial and is not recomputed here.
        self._device_query_positions = q.copy()
        # The device trial refitted the query to these positions (NeedleQueryGeometry.update), which
        # measured their motion from the positions the query holds as committed.
        self._device_motion_start = self._previous_q if self._device_queries.has_previous else None
        self._device_query_points = None
        self._device_locations = {}
        self._prepare_device_queries(state, q, tip[None])
        embedding = self._locate(tip, q)
        if (embedding is not None) != self._in_material:
            return False
        if embedding is not None and np.any(self.fixed[embedding[0]] & (embedding[1] > 0.05)):
            return False
        fields = {name: getattr(self, name) for name in self._TRIAL_FIELDS}
        fields.update(
            _pending_opening=None,
            _pending_reused=False,
            _tip_embedding=embedding,
            _step_start_tip=tip.copy(),
            _step_start_q=q,
            _step_start_samples=self._body_samples(state, body_index, pose),
        )
        if not self.punctured:
            fields["status"] = "contact" if diagnostic[6, 0] > 0 else ("outside" if self._ever_inside else "approach")
        self.tip_position = tip.copy()
        self._tip_exit_point = np.zeros(3)
        wp.copy(self._sample_in_visited_material, sample_mask)
        self._publish_force_diagnostics(diagnostic, pose, body_com_local)
        self._trial_fields = fields
        if self.punctured and not self._in_material:
            # Staged endpoints, not accepted physical steps. A later physical
            # guard may still reject this transaction; no device read is added.
            self._authorized_outside_certified_trials = getattr(self, "_authorized_outside_certified_trials", 0) + 1
        return True

    def _add_forces(self, state, body_index, body_com_local, external_forces, dt, *, implicit=False, trial=False):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.step_time.set(dt)
        if self.status == "failed":
            return
        q = state.particle_q.numpy().astype(float)
        if not np.isfinite(q).all():
            self.status, self.reason = "failed", "Non-finite tissue positions"
            return
        pose = state.body_q.numpy()[body_index]
        tip = self._body_tip(state, body_index, pose)
        if trial and self._previous_tip is not None and np.linalg.norm(tip - self._previous_tip) > self.step_resolution:
            # No step with this motion is accepted (check_step_resolution), and the swept tip
            # tests below visit every triangle along the path: a trial that threw the tip far
            # would spend its time there. Reject it now; the caller retries a shorter interval.
            raise NeedleStepTooLarge(
                "Unresolved tip displacement in a trial; reduce timestep to resolve needle radius and track spacing"
            )
        self.tip_position = tip.copy()
        self._step_start_tip, self._step_start_q = tip.copy(), q.copy()
        self._step_start_samples = self._body_samples(state, body_index, pose)
        if (
            self.punctured
            and not self._in_material
            and np.linalg.norm(tip - self._opening_position(q)) > 2 * float(self.radii.max())
        ):
            self.punctured, self._opening = False, None
        self._diagnostics.zero_()
        self._contact_flags.zero_()
        wp.launch(
            _surface_positions, len(self.mapping), [state.particle_q, self._mapping, self._surface], device=self.device
        )
        self.mesh.refit()
        wp.copy(self._active_surface, state.particle_q, count=len(self.vertices))
        self._active_mesh.refit()
        if self.device.is_cuda:
            self._prepare_device_queries(state, q, self._force_query_points(tip, self._step_start_samples))
        else:
            self._refresh_surface(q)
        self._tip_embedding = self._locate(tip, q)
        if self.track_count:
            wp.launch(
                _track_positions,
                self.track_count,
                [state.particle_q, self._track_nodes, self._track_bary, self._track_points],
                device=self.device,
            )
        if self._openings:
            # The first rows of the scratch; the kernels read up to the live count.
            self._opening_points.assign(np.asarray([bary @ q[ids] for ids, bary in self._openings]))
        capture = 2 * float(self.radii.max())
        sample_mask = self._device_visited_sample_mask(state, body_index, q, self._step_start_samples)
        if sample_mask is None:
            if (
                self.device.is_cuda
                and self._visited_tets
                and self._device_query_points is not None
                and len(self._device_query_points) == 1
            ):
                self._prepare_device_queries(state, q, np.vstack((tip, self._step_start_samples)))
            sample_mask = self._visited_sample_mask(state, body_index, q, samples=self._step_start_samples)
        self._sample_in_visited_material.assign(sample_mask)
        recorder = getattr(self, "_surface_tangent_anchors", None)
        surface_kernel, surface_extra = _surface_forces, []
        if recorder is not None:
            from .surface_contact_forces import surface_forces_with_tangent

            recorder.clear()
            surface_kernel = surface_forces_with_tangent
            surface_extra = recorder.arguments(self._surface_tangent_moment_length)
        wp.launch(
            surface_kernel,
            len(self.samples),
            [
                self.mesh.id,
                self._active_mesh.id,
                self._triangles,
                self._surface,
                self._mapping,
                self._fixed,
                state.particle_qd,
                self._samples,
                self._radii,
                self._weights,
                state.body_q,
                state.body_qd,
                wp.vec3(*body_com_local),
                body_index,
                self.stiffness,
                self.damping,
                self.friction,
                self.speed_scale,
                self.step_time.seconds,
                int(not implicit),
                self._rigid_inverse_mass,
                self._rigid_inverse_inertia,
                self._particle_inverse_mass,
                self._opening_points,
                self._all_opening_faces,
                2 * float(self.radii.max()),
                self._sample_in_visited_material,
                wp.vec3(*self._tip_exit_point),
                self._track_points,
                self._live,
                capture,
                external_forces,
                state.body_f,
                self._diagnostics,
                self._contact_flags,
            ]
            + surface_extra,
            device=self.device,
            block_dim=1,
        )
        if self.tunnel_alm is not None:
            self.tunnel_alm.add_forces(state, body_index, body_com_local, external_forces, dt)
        elif self.track_count:
            from .shaft_projection import prepare_projection

            projection = prepare_projection(
                self, state, body_index, float(self.radii.max()), operands=(external_forces,)
            )
            shaft_kernel, projection_extra = _shaft_forces, []
            if projection is not None:
                from .shaft_projected_kernels import shaft_forces_projected

                shaft_kernel, projection_extra = shaft_forces_projected, [projection.result]
            if not implicit:
                # The explicit law updates the engagement inside this kernel.
                self._engaged_length = None
            wp.launch(
                shaft_kernel,
                self.track_count,
                [
                    self._track_nodes,
                    self._track_bary,
                    self._track_lengths,
                    self._track_engaged,
                    self._live,
                    int(not implicit),
                    state.particle_q,
                    state.particle_qd,
                    self._fixed,
                    self._samples,
                    state.body_q,
                    state.body_qd,
                    wp.vec3(*body_com_local),
                    body_index,
                    self.shaft_stiffness_per_m,
                    self.shaft_damping_per_m,
                    self.friction,
                    self.speed_scale,
                    capture,
                    self.step_time.seconds,
                    int(not implicit),
                    self._rigid_inverse_mass,
                    self._rigid_inverse_inertia,
                    self._particle_inverse_mass,
                    external_forces,
                    state.body_f,
                    self._diagnostics,
                ]
                + projection_extra,
                device=self.device,
            )
        flags = self._contact_flags.numpy()
        if flags[1]:
            self.status, self.reason = "failed", "Needle center penetrated fixed or inactive tissue; reduce timestep"
        diagnostic = self._diagnostics.numpy().astype(float)
        self._publish_force_diagnostics(diagnostic, pose, body_com_local)
        self._pending_opening = None
        self._pending_reused = False
        if self.status == "failed":
            return
        if not self.punctured:
            if self.status != "failed":
                self.status = "contact" if self.last_tip_force > 0 else ("outside" if self._ever_inside else "approach")
            radius = float(self.radii[0])
            nearby = None
            if self._device_geometry_matches(q):
                nearby = self._device_queries.surface_candidates(self.mesh, tip - radius, tip + radius)
            if nearby is None:
                self._refresh_surface(q)
                nearby = _point_aabb_candidates(tip, *self._surface_bounds, padding=radius)
            else:
                low, high = _vertex_bounds(self.surface_vertices[self.surface_triangles[nearby]])
                nearby = nearby[_point_aabb_candidates(tip, low, high, padding=radius)]
            if not len(nearby):
                return
            local_face, bary, point, distance = _closest_triangle(
                tip, self.surface_vertices, self.surface_triangles[nearby]
            )
            face = nearby[local_face]
            ids = self.mapping[self.surface_triangles[face]]
            # The criterion uses the actual localized normal force, including
            # approach damping, evaluated by the same contact kernel.
            reuse = self._matching_opening(tip, face, q)
            if reuse is not None:
                self._pending_opening = self._openings[reuse]
                self._pending_opening_face = int(face)
                self._pending_reused = True
            elif self.last_tip_force >= self.puncture_force and np.all(ids >= 0) and not np.any(self.fixed[ids]):
                self._pending_opening = (ids.copy(), bary.copy())
                self._pending_opening_face = int(face)
            if self._tip_embedding is not None and self._pending_opening is None:
                self.status, self.reason = "failed", "Tip crossed the surface before puncture authorization"

    def _resolution_crossings(self, start, tip, old_q, q):
        """Read cached geometry when it matches; never publish candidate history."""
        if self._device_geometry_matches(q):
            return (
                self._material_crossings(start, tip, old_q, q),
                self._material_crossings(start, tip, old_q, q, active=True),
            )
        current_cached = self._query_positions is not None and np.array_equal(q, self._query_positions)
        committed_start = self._previous_q is not None and np.array_equal(old_q, self._previous_q)
        previous_cached = self._previous_surface_bounds is not None and committed_start
        if not committed_start:
            # Both surfaces get the complete host test from the actual start positions.
            reason = "no device geometry of the end positions"
            self.shifted_start_host_tests[reason] = self.shifted_start_host_tests.get(reason, 0) + 2
        if current_cached:
            points = self.surface_vertices
        else:
            points = self.surface_vertices.copy()
            points[self._live_surface_vertices] = q[self.mapping[self._live_surface_vertices]]
        if previous_cached:
            old_points = self._previous_surface
        else:
            old_points = self.surface_vertices.copy()
            old_points[self._live_surface_vertices] = old_q[self.mapping[self._live_surface_vertices]]
        bounds = active_bounds = None
        if current_cached and previous_cached:
            bounds = (
                np.minimum(self._previous_surface_bounds[0], self._surface_bounds[0]),
                np.maximum(self._previous_surface_bounds[1], self._surface_bounds[1]),
            )
            active_bounds = (
                np.minimum(self._previous_active_bounds[0], self._active_bounds[0]),
                np.maximum(self._previous_active_bounds[1], self._active_bounds[1]),
            )
        crossings = _surface_crossings(start, tip, old_points, points, self.surface_triangles, bounds=bounds)
        active_crossings = _surface_crossings(start, tip, old_q, q, self._active_triangles, bounds=active_bounds)
        return crossings, active_crossings

    def check_step_resolution(self, state_in, state_out, body_index):
        """Reject unresolved candidate motion without changing interaction state.

        Call before non-trial force evaluation and :meth:`accept`. A
        :class:`NeedleStepTooLarge` permits retrying from the same accepted state
        with smaller physical timesteps. The caller must also roll back both
        integrators. This checks every needle sample and swept tip crossings;
        it neither authorizes puncture nor accepts unsupported material entry.
        The crossings of a candidate that passes are kept for its :meth:`accept`.
        """
        self._checked_crossings = None
        old_pose, pose = state_in.body_q.numpy()[body_index], state_out.body_q.numpy()[body_index]
        old_samples = self._body_samples(state_in, body_index, old_pose)
        samples = self._body_samples(state_out, body_index, pose)
        old_q = state_in.particle_q.numpy().astype(float)
        q = state_out.particle_q.numpy().astype(float)
        if not all(np.isfinite(values).all() for values in (old_samples, samples, old_q, q)):
            raise RuntimeError("Non-finite coupled state")
        motion = np.linalg.norm(samples - old_samples, axis=1).max()
        if motion > self.step_resolution:
            raise NeedleStepTooLarge(
                "Unresolved tip or shaft displacement; reduce timestep to resolve needle radius and track spacing"
            )
        start, tip = self._body_tip(state_in, body_index, old_pose), self._body_tip(state_out, body_index, pose)
        crossings, active_crossings = self._resolution_crossings(start, tip, old_q, q)
        if len(crossings) > 1:
            raise NeedleStepTooLarge(
                "Multiple surface crossings within one step; reduce timestep to resolve tissue gaps"
            )
        if len(active_crossings) > 1:
            raise NeedleStepTooLarge("Multiple active material surface crossings; reduce timestep")
        # accept() makes this swept tip test for the step it commits. When that step has these
        # four arrays, value by value, it takes this result instead of repeating the test.
        self._checked_crossings = (start, tip, old_q, q, crossings, active_crossings)

    def _side_change_held(self, tip, start, crossings):
        """Whether the tip keeps its side in a step whose accepted tip is on the other side of the surface.

        An entry or an exit is a passage of the tip. An accepted step that is not converged beside
        an opening can leave the surface on the other side of the tip for that one step, while the
        tip keeps moving the same way; the surface falls back in the next step. Such a change of
        side is held: it is the surface that passed, not the needle.

        ``travel`` is the displacement of the tip in this step, from ``start`` to ``tip``, along the
        outward normal at the accepted state of the face the event would take: the face of the
        swept crossing in the direction of the change (``crossings``, the ones :meth:`accept`
        holds), else the face at which the last accepted step held this change of side
        (``_held_face``), else the face nearest the tip. An entry is held when the tip moved
        outward and an exit when it moved inward, by more than the resolution of the stored tip
        position: 16 units in the last place of its largest ``float32`` coordinate. A change of
        side is held only while the tip is at most one resolved step (:attr:`step_resolution`)
        past that face. Farther than that the surface has not merely passed over the tip, and the
        change is committed. Host arithmetic on arrays :meth:`accept` already has.

        The face of a held change of side is kept for the next accepted step, in which no
        surface crosses the tip: in tissue thinner than twice the tip's depth, or with another
        wall nearer than the one the tip left, the nearest face is not the face of the event,
        and its normal reads the tip's motion with the other sign. The kept face is dropped
        when the tip is back on its side, when the change is committed, and by :meth:`accept`
        when it no longer stands (:meth:`_held_face_stands`): the tip is no longer past its
        plane, or the exit branch would refuse the face. While a change of side is held the
        in-material flag does not change, so a kept face is of the direction of this change.
        """
        kept, self._held_face = self._held_face, None
        if start is None or self.tip_inside == self._in_material:
            return False
        direction = -1 if self.tip_inside else 1  # of the tip through the surface: inward, outward
        crossed = [event[2] for event in crossings if event[1] == direction]
        if not crossed and kept is not None:
            crossed = [kept]
        triangles = self.surface_triangles[crossed[:1]] if crossed else self.surface_triangles
        face, _, point, _ = _closest_triangle(tip, self.surface_vertices, triangles)
        triangle = self.surface_vertices[triangles[face]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= max(float(np.linalg.norm(normal)), 1e-30)
        travel = float((tip - start) @ normal)  # of the tip along the outward normal in this step
        past = direction * float((tip - point) @ normal)  # of the tip beyond the face, on its new side
        deadband = 16.0 * float(np.spacing(np.float32(np.abs(tip).max())))
        held = direction * travel < -deadband and past <= self.step_resolution
        if held:
            self._held_face = int(crossed[0]) if crossed else int(face)
        return held

    def _exit_face(self, tip, face=None, crossed=False):
        """The face of an exit, and why the exit branch of :meth:`accept` refuses it, if it does.

        ``face`` is a surface triangle, or None for the one nearest the tip. ``crossed`` says
        that the tip crossed that face outward in this step. Returns the face, its tissue nodes,
        the barycentric coordinates of its point nearest the tip, and the reason of the refusal
        or None. The branch refuses a face the tip is not over (unless the tip crossed it), a
        face the tip is more than two needle radii from, and a face whose point nearest the tip
        is in inactive or supported tissue. :meth:`_held_face_stands` asks the same of a face
        kept from a held step through this method, so the two readings cannot differ.
        """
        triangles = self.surface_triangles if face is None else self.surface_triangles[[face]]
        nearest, bary, point, distance = _closest_triangle(tip, self.surface_vertices, triangles)
        if face is None:
            face = nearest
        triangle = self.surface_vertices[self.surface_triangles[face]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        outward = crossed or float((tip - point) @ normal) > 0
        ids = self.mapping[self.surface_triangles[face]]
        refusal = None
        if not outward or distance > 2 * float(self.radii.max()):
            refusal = "Tip left active tissue without a resolved outward surface crossing"
        elif np.any(ids < 0) or np.any(self.fixed[ids] & (bary > 0.05)):
            refusal = "Tip center penetrated fixed or inactive tissue; reduce timestep"
        return face, ids, bary, refusal

    def _held_face_stands(self, tip, face):
        """Whether surface triangle ``face``, kept from a held step, is still the face of that change of side.

        For an entry the tip, in the tissue, must be under the plane of the face. For an exit
        the exit branch must not refuse the face (:meth:`_exit_face`): the tip is over it and
        not more than two needle radii from it, and its point nearest the tip is in active
        tissue that is not supported. A kept face stops standing when the tip goes round an
        edge of the surface. The face of an exit also stops standing when the tip slides along
        the surface, more than two radii away from that triangle or to where the triangle's
        nearest point is supported. The change of side is then read at the nearest face again,
        as without a kept face, so a commit at a kept face is never refused by the exit branch.
        """
        if not self.tip_inside:
            return self._exit_face(tip, face)[3] is None
        triangle = self.surface_vertices[self.surface_triangles[face]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        return float((tip - triangle[0]) @ normal) < 0  # under the face

    def accept(self, state, body_index):
        """Commit natural events after a physical step; no task completion.

        Exit/entry event positions are accepted actual tip positions, within
        one resolved step of the moving boundary. Exits permit later handling.
        A change of side that the surface makes against the tip's own motion
        commits nothing (:meth:`_side_change_held`); one that was held and is
        committed in a later step takes the face it was held at, while that
        face stands (:meth:`_held_face_stands`).
        """
        requested, self._engaged_length_requested = self._engaged_length_requested, False
        checked, self._checked_crossings = self._checked_crossings, None
        self.events = ()
        if self.status == "failed":
            return
        events = []
        q = state.particle_q.numpy().astype(float)
        pose = state.body_q.numpy()[body_index]
        tip = self._body_tip(state, body_index, pose)
        samples = self._body_samples(state, body_index, pose)
        self.tip_position = tip.copy()
        if not np.isfinite(samples).all() or not np.isfinite(q).all():
            self.status, self.reason = "failed", "Non-finite coupled state"
            return
        start = self._step_start_tip if self._step_start_tip is not None else self._previous_tip
        old_q = self._step_start_q if self._step_start_q is not None else self._previous_q
        old_samples = self._step_start_samples if self._step_start_samples is not None else self._previous_samples
        self._step_start_tip = self._step_start_q = self._step_start_samples = None
        crossings = []
        if start is not None:
            motion = np.linalg.norm(samples - old_samples, axis=1).max()
            if motion > min(float(self.radii.min()), self.track_spacing) * 1.01:
                self.status, self.reason = (
                    "failed",
                    "Unresolved tip or shaft displacement; reduce timestep to resolve needle radius and track spacing",
                )
                return
            if not self._device_geometry_matches(q):
                self._refresh_surface(q)
            if checked is not None and all(
                np.array_equal(a, b) for a, b in zip(checked[:4], (start, tip, old_q, q), strict=True)
            ):
                # check_step_resolution made the swept tip test of this very step.
                crossings, active_crossings = checked[4:]
            else:
                crossings, active_crossings = self._resolution_crossings(start, tip, old_q, q)
            if len(crossings) > 1:
                self.status, self.reason = (
                    "failed",
                    "Multiple surface crossings within one step; reduce timestep to resolve tissue gaps",
                )
                return
            if len(active_crossings) > 1:
                self.status, self.reason = "failed", "Multiple active material surface crossings; reduce timestep"
                return
            if len(active_crossings) != len(crossings) or any(
                a[1] != b[1] or abs(a[0] - b[0]) > 1e-5 for a, b in zip(active_crossings, crossings, strict=True)
            ):
                self.status, self.reason = "failed", "Tip crossed an inactive material interface; reduce timestep"
                return
        if not self._device_geometry_matches(q):
            self._refresh_surface(q)
        if self.track_count:
            self._engaged_length = None
            wp.launch(
                _update_track_engagement,
                self.track_count,
                [
                    self._track_nodes,
                    self._track_bary,
                    state.particle_q,
                    self._samples,
                    state.body_q,
                    body_index,
                    2 * float(self.radii.max()),
                    self._track_engaged,
                ],
                device=self.device,
            )
        if self._pending_opening is not None and not self.punctured:
            self._opening = self._pending_opening
            self._opening_reused = self._pending_reused
            face = self._pending_opening_face
            if self._pending_reused:
                index = self._matching_opening(tip, face, q)
                if index is None:
                    self._opening = None
                else:
                    self._opening_faces = self._opening_masks[index]
            else:
                ids, bary = self._opening
                self._opening_faces = self._register_opening(ids, bary, face)
                self.puncture_count += 1
                events.append({"kind": "puncture", "position": self._opening_position(q).copy(), "reused": False})
            self.punctured = self._opening is not None
            if self.punctured:
                self.status = "punctured" if not self._pending_reused else "outside"
        embedding = self._locate(tip, q)
        self.tip_inside = embedding is not None
        if self._held_face is not None and not self._held_face_stands(tip, self._held_face):
            self._held_face = None  # the tip went round an edge of it, or the exit branch would refuse it
        kept = self._held_face  # of a change of side held so far; the call below replaces it
        if self._side_change_held(tip, start, crossings):
            # The tip keeps its side: no event, no counter, no bookkeeping and no failure check
            # of the branches below. tip_inside says where the tip is; the next step asks again.
            pass
        elif self.tip_inside:
            ids, bary = embedding
            if not self._in_material:
                inward = [event for event in crossings if event[1] < 0]
                if inward:
                    face = inward[0][2]
                elif kept is not None:
                    face = kept
                else:
                    face = _closest_triangle(tip, self.surface_vertices, self.surface_triangles)[0]
                reuse = self._matching_opening(tip, face, q)
                if not self.punctured and reuse is not None:
                    self._opening = self._openings[reuse]
                    self._opening_faces = self._opening_masks[reuse]
                    self.punctured = True
                    self._opening_reused = True
                if not self.punctured:
                    self.status, self.reason = "failed", "Tip crossed the surface before puncture authorization"
                    return
                if self._opening_faces[face] == 0 or np.linalg.norm(tip - self._opening_position(q)) > 2 * float(
                    self.radii.max()
                ):
                    self.status, self.reason = "failed", "Tip entered outside its authorized material opening"
                    return
                self.entry_count += 1
                events.append({"kind": "entry", "position": tip.copy(), "reused": self._opening_reused})
                self._active_section = self._next_section
                self._next_section += 1
                self._track_anchor = None
            if np.any(self.fixed[ids] & (bary > 0.05)):
                self.status, self.reason = "failed", "Tip center penetrated supported material; reduce timestep"
                return
            self._visited_tets.add(self._tet_indices[tuple(ids)])
            self._sync_visited()
            self._in_material = self._ever_inside = True
            self.status = "inserted"
            self._record_track(ids, bary)
        elif self._in_material:
            outward_events = [event for event in crossings if event[1] > 0]
            # The crossed face, else the face the exit was held at, else the nearest one.
            face, ids, bary, refusal = self._exit_face(
                tip, outward_events[0][2] if outward_events else kept, bool(outward_events)
            )
            if refusal is not None:
                self.status, self.reason = "failed", refusal
                return
            if self._matching_opening(tip, face, q) is None:
                self._register_opening(ids, bary, face)
            self.exit_count += 1
            self.emergence_position = tip.copy()
            events.append({"kind": "exit", "position": tip.copy(), "reused": False})
            self._in_material = self.punctured = False
            self._opening = self._track_anchor = None
            self.status = "outside"
        elif any(event[1] < 0 for event in crossings):
            self.status, self.reason = "failed", "Tip center penetrated inactive tissue; reduce timestep"
            return
        self.events = tuple(events)
        self._previous_tip, self._previous_q, self._previous_samples = tip.copy(), q.copy(), samples.copy()
        self._previous_surface = self.surface_vertices.copy()
        if self._query_positions is not None and np.array_equal(q, self._query_positions):
            self._previous_surface_bounds = tuple(bound.copy() for bound in self._surface_bounds)
            self._previous_active_bounds = tuple(bound.copy() for bound in self._active_bounds)
        else:
            self._previous_surface_bounds = self._previous_active_bounds = None
        if self._device_queries is not None:
            self._device_queries.commit(state.particle_q)
        self._pending_opening = None
        if requested and self.track_count:
            # The engagement was updated above. Implicit trials hold it, so the reader's query
            # of the next step is answered from this read, which waits for that update alone.
            self._engaged_length = (self._track_engaged, self._track_lengths, self._read_engaged_track_length())

    def _rest_segment_clear(self, start, end):
        return not _surface_crossings(
            start,
            end,
            self.vertices,
            self.vertices,
            self._active_triangles,
            bounds=self._active_reference_bounds,
        )

    def _record_track(self, ids, bary):
        rest = bary @ self.vertices[ids]
        if self._track_rest:
            distances = np.linalg.norm(np.asarray(self._track_rest) - rest, axis=1)
            for index in np.argsort(distances):
                if distances[index] >= self.track_spacing:
                    break
                if np.array_equal(ids, self._track_nodes_host[index]) or self._rest_segment_clear(
                    self._track_rest[index], rest
                ):
                    self._track_anchor = int(index)
                    self._active_section = self._track_sections[index]
                    return
        if self.track_count >= self.maximum_track_points:
            self.status, self.reason = "failed", "Insertion track capacity exceeded"
            return
        index = self.track_count
        if self._track_anchor is not None and (
            np.array_equal(ids, self._track_nodes_host[self._track_anchor])
            or self._rest_segment_clear(self._track_rest[self._track_anchor], rest)
        ):
            self._track_edges.add(tuple(sorted((self._track_anchor, index))))
        self._track_nodes_host.append(ids.copy())
        self._track_bary_host.append(bary.copy())
        self._track_rest.append(rest)
        self._track_sections.append(self._active_section)
        self._track_anchor = index
        self._upload_track()

    def _upload_track(self):
        """Write the appended samples, the lengths of the live rows and the live count to the device.

        In place and host to device only: no array is replaced and nothing is read back. An older
        row keeps its nodes, weights and engagement on the device. The quadrature lengths change
        with every new edge and are written for all live rows.
        """
        count, start = self.track_count, self._track_uploaded
        while count > self.track_capacity:
            self._grow_track_storage()
        if count > start:
            rows = count - start
            nodes = np.asarray(self._track_nodes_host[start:count], dtype=np.int32).reshape(rows, 4)
            bary = np.asarray(self._track_bary_host[start:count], dtype=np.float64).reshape(rows, 4)
            for storage, values in (
                (self._track_nodes, nodes),
                (self._track_bary, bary.astype(np.float32)),
                (self._track_bary_d, bary),
                # A newly traversed material sample starts engaged at the actual tip.
                (self._track_engaged, np.ones(rows, dtype=np.int32)),
            ):
                source = wp.array(values, dtype=storage.dtype, device="cpu", copy=False)
                wp.copy(storage, source, dest_offset=start, count=rows)
            self._track_uploaded = count
        lengths = np.zeros(count)
        for a, b in self._track_edges:
            half_length = np.linalg.norm(self._track_rest[a] - self._track_rest[b]) / 2
            lengths[a] += half_length
            lengths[b] += half_length
        if count:
            source = wp.array(lengths.astype(np.float32), dtype=float, device="cpu", copy=False)
            wp.copy(self._track_lengths, source, count=count)
        # Engagement and lengths changed in place: an engaged length read before is not theirs.
        self._engaged_length = None
        self._write_live()

    def _grow_track_storage(self):
        """Double the track storage, at most to ``maximum_track_points``: the reserve was too small.

        The uploaded rows are copied device to device. Scratch that other objects hold per track
        row (trial-force helpers, tunnel multipliers, projection bindings) follows the count of
        growths.
        """
        live, old = self._track_uploaded, self.track_capacity
        self.track_capacity = min(2 * old, self.maximum_track_points)
        if self.track_capacity == old:
            raise RuntimeError("The track storage holds maximum_track_points samples and cannot grow")
        for name, dtype in self._TRACK_STORAGE:
            storage = getattr(self, name)
            grown = wp.zeros(self.track_capacity, dtype=dtype, device=self.device)
            if live:
                wp.copy(grown, storage, count=live)
            setattr(self, name, grown)
        self.track_storage_growths += 1
        warnings.warn(
            f"Needle track storage grew from {old} to {self.track_capacity} rows at {live} samples; "
            "pass track_capacity to reserve it at construction",
            stacklevel=3,
        )
