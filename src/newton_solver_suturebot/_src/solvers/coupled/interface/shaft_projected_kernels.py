# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Projection-specialized copies for isolated validation.

Only the geometric lookup differs from the original kernels. Source-contract
regressions verify all physical force, derivative and scatter statements match.
"""

import warp as wp

from .insertion import LIVE_TRACK, _give_force, _point_inverse_mass
from .shaft_projection import read_projection
from .tunnel_alm import _normal_projection


@wp.kernel(enable_backward=False)
def shaft_forces_projected(
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
    projected: wp.array2d[float],
):
    i = wp.tid()
    if i >= live[LIVE_TRACK]:
        return
    point, tissue_velocity = wp.vec3(0.0), wp.vec3(0.0)
    for j in range(4):
        point += bary[i][j] * q[nodes[i][j]]
        tissue_velocity += bary[i][j] * qd[nodes[i][j]]
    pose = body_q[body]
    closest, tangent, best, covered = read_projection(projected, i)
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


@wp.kernel(enable_backward=False)
def tunnel_forces_projected(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    lengths: wp.array[float],
    engaged: wp.array[int],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    fixed: wp.array[int],
    samples: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.vec3,
    body: int,
    normal: wp.array[wp.vec3],
    tangential: wp.array[float],
    next_normal: wp.array[wp.vec3],
    next_tangential: wp.array[float],
    penalty: float,
    compliance: float,
    clearance: float,
    tangent_radius: float,
    friction: float,
    dt: wp.array[float],
    particle_f: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    diagnostics: wp.array[wp.vec3],
    statistics: wp.array[float],
    projected: wp.array2d[float],
):
    i = wp.tid()
    next_normal[i] = wp.vec3(0.0)
    next_tangential[i] = 0.0
    if engaged[i] == 0 or lengths[i] <= 0.0:
        return
    point, tissue_velocity = wp.vec3(0.0), wp.vec3(0.0)
    for j in range(4):
        point += bary[i][j] * q[nodes[i][j]]
        tissue_velocity += bary[i][j] * qd[nodes[i][j]]
    pose = body_q[body]
    closest, tangent, distance, covered = read_projection(projected, i)
    displacement = closest - point
    displacement -= wp.dot(displacement, tangent) * tangent
    previous_normal = normal[i] - wp.dot(normal[i], tangent) * tangent
    reaction = _normal_projection(displacement, tangent, previous_normal, penalty, compliance, clearance)
    if compliance > 0.0:
        # The normal constitutive equation is already satisfied exactly.
        # Only tangential history still contributes a dual fixed-point error.
        previous_normal = reaction
    center = wp.transform_point(pose, body_com)
    velocity = wp.spatial_top(body_qd[body]) + wp.cross(wp.spatial_bottom(body_qd[body]), point - center)
    sliding = wp.dot(velocity - tissue_velocity, tangent)
    bound = friction * wp.length(reaction)
    drag = wp.clamp(tangential[i] + penalty * dt[0] * sliding, -bound, bound)
    next_normal[i] = reaction
    next_tangential[i] = drag

    # The proximal normal residual enforces the normal cone of the clearance
    # disk. For positive reaction its fixed point is |d|-clearance=alpha*|lambda|.
    normal_error = wp.length(reaction - previous_normal) / penalty
    geometric_error = wp.max(0.0, wp.length(displacement) - clearance - compliance * wp.length(reaction))
    wp.atomic_max(statistics, 0, normal_error)
    wp.atomic_max(statistics, 1, geometric_error)
    # Sum changes before scatter: this bounds the resultant and any barycentric
    # nodal load change, including normal dual error as well as friction.
    wp.atomic_add(statistics, 2, lengths[i] * (wp.length(reaction - previous_normal) + wp.abs(drag - tangential[i])))
    scale = wp.max(wp.length(reaction), wp.length(previous_normal))
    scale = wp.max(scale, wp.max(wp.abs(drag), wp.abs(tangential[i])))
    wp.atomic_add(statistics, 3, lengths[i] * scale)
    if not wp.isfinite(reaction[0] + reaction[1] + reaction[2] + drag):
        wp.atomic_max(statistics, 4, 1.0)

    force = -lengths[i] * (reaction + drag * tangent)
    _give_force(force, point, center, body, body_f, diagnostics)
    for j in range(4):
        nodal = -bary[i][j] * force
        wp.atomic_add(particle_f, nodes[i][j], nodal)
        if fixed[nodes[i][j]] != 0:
            wp.atomic_add(diagnostics, 4, nodal)
            wp.atomic_add(diagnostics, 5, wp.cross(q[nodes[i][j]], nodal))
        else:
            wp.atomic_add(diagnostics, 2, nodal)
        wp.atomic_add(diagnostics, 3, wp.cross(q[nodes[i][j]], nodal))


@wp.kernel(enable_backward=False)
def prepare_anchors_projected(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    lengths: wp.array[float],
    engaged: wp.array[int],
    live: wp.array[int],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    samples: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    body: int,
    com: wp.vec3,
    radius: float,
    stiffness: float,
    damping: float,
    friction: float,
    speed: float,
    h: wp.array[float],
    moment_length: float,
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
    projected: wp.array2d[float],
):
    i = wp.tid()
    if i >= live[LIVE_TRACK]:
        return
    matrix = wp.mat33(0.0)
    lever = wp.vec3(0.0)
    if engaged[i] != 0 and lengths[i] > 0.0:
        point = wp.vec3(0.0)
        tissue_v = wp.vec3(0.0)
        for j in range(4):
            point += bary[i][j] * q[nodes[i][j]]
            tissue_v += bary[i][j] * qd[nodes[i][j]]
        pose = poses[body]
        closest, tangent, distance, covered = read_projection(projected, i)
        center = wp.transform_point(pose, com)
        arm = point - center
        lever = arm / moment_length
        relative = wp.spatial_top(velocities[body]) + wp.cross(wp.spatial_bottom(velocities[body]), arm) - tissue_v
        sliding = wp.dot(relative, tangent)
        normal_velocity = relative - sliding * tangent
        transverse = closest - point
        transverse -= wp.dot(transverse, tangent) * tangent
        normal_force = -lengths[i] * (stiffness * transverse + damping * normal_velocity)
        projection = wp.identity(n=3, dtype=float) - wp.outer(tangent, tangent)
        matrix = lengths[i] * (h[0] * stiffness + damping) * projection
        tangent_coefficient = friction * wp.length(normal_force) / wp.max(speed, wp.abs(sliding))
        if wp.abs(sliding) > speed:
            tangent_coefficient *= wp.max(0.0, speed / wp.abs(sliding))
        matrix += tangent_coefficient * wp.outer(tangent, tangent)
    matrices[i] = matrix
    levers[i] = lever


@wp.kernel(enable_backward=False)
def prepare_alm_anchors_projected(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    lengths: wp.array[float],
    engaged: wp.array[int],
    live: wp.array[int],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    samples: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    body: int,
    com: wp.vec3,
    radius: float,
    normal: wp.array[wp.vec3],
    tangential: wp.array[float],
    penalty: float,
    compliance: float,
    clearance: float,
    friction: float,
    h: wp.array[float],
    moment_length: float,
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
    shears: wp.array[wp.mat33],
    projected: wp.array2d[float],
):
    i = wp.tid()
    if i >= live[LIVE_TRACK]:
        return
    matrix = wp.mat33(0.0)
    lever = wp.vec3(0.0)
    shear = wp.mat33(0.0)
    if engaged[i] != 0 and lengths[i] > 0.0:
        point = wp.vec3(0.0)
        tissue_v = wp.vec3(0.0)
        for j in range(4):
            point += bary[i][j] * q[nodes[i][j]]
            tissue_v += bary[i][j] * qd[nodes[i][j]]
        pose = poses[body]
        closest, tangent, distance, covered = read_projection(projected, i)
        center = wp.transform_point(pose, com)
        arm = point - center
        lever = arm / moment_length
        relative = wp.spatial_top(velocities[body]) + wp.cross(wp.spatial_bottom(velocities[body]), arm) - tissue_v
        slip = wp.dot(relative, tangent)
        projection = wp.identity(n=3, dtype=float) - wp.outer(tangent, tangent)
        z = projection * (normal[i] + penalty * (closest - point))
        threshold = penalty * clearance
        factor = 1.0 / (1.0 + penalty * compliance)
        normal_stiffness = factor * penalty
        if compliance > 0.0:
            z = projection * (closest - point)
            threshold = clearance
            factor = 1.0 / compliance
            normal_stiffness = factor
        magnitude = wp.length(z)
        if magnitude > threshold:
            direction = z / magnitude
            fraction = threshold / magnitude
            # Exact local normal derivative: compliant constitutive law, or
            # the unchanged proximal map for a zero-compliance hard wall.
            derivative = normal_stiffness * ((1.0 - fraction) * projection + fraction * wp.outer(direction, direction))
            matrix = lengths[i] * h[0] * derivative
            normal_magnitude = factor * (magnitude - threshold)
            bound = friction * normal_magnitude
            tangent_trial = tangential[i] + penalty * h[0] * slip
            if wp.abs(tangent_trial) < bound:
                matrix += lengths[i] * penalty * h[0] * wp.outer(tangent, tangent)
            else:
                # Sliding: the drag is the Coulomb bound of this reaction, sign * friction * |reaction|.
                # A change dR of the reaction changes it by friction * sign * (direction . dR) along the shaft.
                shear = friction * wp.sign(tangent_trial) * wp.outer(tangent, direction)
    matrices[i] = matrix
    levers[i] = lever
    shears[i] = shear
