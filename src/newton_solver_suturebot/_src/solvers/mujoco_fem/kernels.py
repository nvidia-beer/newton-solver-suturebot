# SPDX-License-Identifier: Apache-2.0
"""Simultaneous momentum and Coulomb-projection equations.

The unknowns are tissue increments, needle translation, L times its angular
increment in the accepted world frame, and zeta=tau/(rho*h). All active
residual rows have force units. Two unused lanes per friction row are identity
equations. The search matrix freezes shaft features and rotation derivatives;
the nonlinear residual always uses the actual endpoint geometry.
"""

import warp as wp

from ...insertion import _shaft_projection
from ..fem.kernels import inverse_diagonal_block


@wp.kernel(enable_backward=False)
def rigid_system(
    q: wp.array[float],
    qd: wp.array[float],
    external: wp.array[wp.spatial_vector],
    joint: wp.array[float],
    mass: wp.array[wp.float64],
    inertia: wp.array[wp.mat33d],
    damping: wp.array[wp.float64],
    h: wp.float64,
    length: float,
    blocks: wp.array[wp.mat33],
    rhs: wp.array[wp.vec3],
):
    rotation32 = wp.quat(q[3], q[4], q[5], q[6])
    rotation = wp.normalize(wp.quatd(rotation32))
    R = wp.quat_to_matrix(rotation)
    w = wp.vec3d(wp.quat_rotate_inv(rotation32, wp.vec3(qd[3], qd[4], qd[5])))
    v = wp.vec3d(wp.vec3(qd[0], qd[1], qd[2]))
    D = wp.diag(wp.vec3d(damping[3], damping[4], damping[5]))
    momentum = inertia[0] * w
    gyro = wp.skew(w) * inertia[0] - wp.skew(momentum)
    hi = wp.float64(h)
    li = wp.float64(length)
    A = inertia[0] + hi * (D + gyro)
    force32 = external[0] + wp.spatial_vector(joint[0], joint[1], joint[2], joint[3], joint[4], joint[5])
    blocks[0] = wp.mat33(
        wp.diag(wp.vec3d(mass[0] / hi + damping[0], mass[0] / hi + damping[1], mass[0] / hi + damping[2]))
    )
    blocks[1] = wp.mat33(R * A * wp.transpose(R) / (hi * li * li))
    blocks[2] = wp.mat33(R * (inertia[0] + hi * D) * wp.transpose(R) / (hi * li * li))
    rhs[0] = wp.vec3(
        wp.vec3d(wp.spatial_top(force32)) - wp.vec3d(damping[0] * v[0], damping[1] * v[1], damping[2] * v[2])
    )
    rhs[1] = wp.vec3((wp.vec3d(wp.spatial_bottom(force32)) - R * (D * w + wp.cross(w, momentum))) / li)


@wp.kernel(enable_backward=False)
def initialize(
    dv: wp.array[wp.vec3],
    load: wp.array[wp.float64],
    tangent: wp.array[float],
    blocks: wp.array[wp.mat33],
    rigid_rhs: wp.array[wp.vec3],
    rho_h: float,
    count: int,
    unknown: wp.array[wp.vec3],
):
    i = wp.tid()
    if i < count:
        unknown[i] = dv[i]
    elif i < count + 2:
        k = i - count
        f = wp.vec3(float(load[3 * i]), float(load[3 * i + 1]), float(load[3 * i + 2]))
        unknown[i] = inverse_diagonal_block(blocks[k]) * (rigid_rhs[k] + f)
    else:
        unknown[i] = wp.vec3(tangent[i - count - 2] / rho_h, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def reconstruct_rigid(
    source_q: wp.array[float],
    source_qd: wp.array[float],
    unknown: wp.array[wp.vec3],
    count: int,
    h: wp.float64,
    length: float,
    q: wp.array[float],
    qd: wp.array[float],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
):
    rot32 = wp.quat(source_q[3], source_q[4], source_q[5], source_q[6])
    rot = wp.normalize(wp.quatd(rot32))
    local32 = wp.quat_rotate_inv(rot32, wp.vec3(source_qd[3], source_qd[4], source_qd[5]))
    w = wp.vec3d(local32) + wp.quat_rotate_inv(rot, wp.vec3d(unknown[count + 1])) / wp.float64(length)
    v = wp.vec3d(wp.vec3(source_qd[0], source_qd[1], source_qd[2])) + wp.vec3d(unknown[count])
    angle = wp.float64(h) * wp.length(w)
    axis = wp.normalize(w)
    half = wp.float64(0.5) * angle
    delta = wp.quatd(axis[0] * wp.sin(half), axis[1] * wp.sin(half), axis[2] * wp.sin(half), wp.cos(half))
    next_rot = wp.quat(wp.normalize(rot * delta))
    next_pos = wp.vec3(wp.vec3d(wp.vec3(source_q[0], source_q[1], source_q[2])) + wp.float64(h) * v)
    next_v = wp.vec3(v)
    next_w = wp.quat_rotate(next_rot, wp.vec3(w))
    for j in range(3):
        q[j] = next_pos[j]
        q[j + 3] = next_rot[j]
        qd[j] = next_v[j]
        qd[j + 3] = next_w[j]
    q[6] = next_rot[3]
    poses[0] = wp.transform(next_pos, next_rot)
    velocities[0] = wp.spatial_vector(next_v, next_w)


@wp.kernel(enable_backward=False)
def momentum_defect(
    action: wp.array[wp.vec3],
    rhs: wp.array[wp.vec3],
    fixed: wp.array[int],
    h: float,
    error: wp.array[wp.float64],
    statistics: wp.array[wp.float64],
    fem_statistics: wp.array[wp.float64],
):
    node = wp.tid()
    # Constrained rows represent prescribed motion, not momentum equations.
    # Their physical contact loads and support reactions remain untouched.
    if fixed[node] == 0:
        for axis in range(3):
            right = wp.float64(rhs[node][axis])
            defect = wp.float64(action[node][axis]) - right
            value = defect / wp.float64(h)
            if not wp.isfinite(value) or not wp.isfinite(right):
                wp.atomic_max(statistics, 5, wp.float64(1.0))
                wp.atomic_max(fem_statistics, 2, wp.float64(1.0))
            else:
                wp.atomic_max(error, 0, wp.abs(value))
                wp.atomic_max(statistics, 0, wp.abs(value))
                wp.atomic_add(fem_statistics, 0, defect * defect)
                wp.atomic_add(fem_statistics, 1, right * right)


@wp.kernel(enable_backward=False)
def recover_rigid_increment(
    source_q: wp.array[float],
    source_qd: wp.array[float],
    endpoint_q: wp.array[float],
    endpoint_qd: wp.array[float],
    count: int,
    length: float,
    unknown: wp.array[wp.vec3],
):
    # Exported velocities are world-frame, while reconstruct_rigid advances
    # the local angular velocity from the accepted orientation. This is only
    # a numerical restart; final acceptance still integrates canonically.
    old_rotation = wp.quat(source_q[3], source_q[4], source_q[5], source_q[6])
    new_rotation = wp.quat(endpoint_q[3], endpoint_q[4], endpoint_q[5], endpoint_q[6])
    old_local = wp.quat_rotate_inv(old_rotation, wp.vec3(source_qd[3], source_qd[4], source_qd[5]))
    new_local = wp.quat_rotate_inv(new_rotation, wp.vec3(endpoint_qd[3], endpoint_qd[4], endpoint_qd[5]))
    unknown[count] = wp.vec3(endpoint_qd[0], endpoint_qd[1], endpoint_qd[2]) - wp.vec3(
        source_qd[0], source_qd[1], source_qd[2]
    )
    unknown[count + 1] = wp.vec3(
        wp.float64(length)
        * wp.quat_rotate(wp.normalize(wp.quatd(old_rotation)), wp.vec3d(new_local) - wp.vec3d(old_local))
    )


@wp.kernel(enable_backward=False)
def set_tangent(unknown: wp.array[wp.vec3], count: int, rho_h: float, tangent: wp.array[float]):
    i = wp.tid()
    tangent[i] = rho_h * unknown[count + 2 + i][0]


@wp.kernel(enable_backward=False)
def predict_tangent(
    unknown: wp.array[wp.vec3],
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    engaged: wp.array[int],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    samples: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    radius: float,
    clearance: float,
    compliance: float,
    friction: float,
    rho_h: float,
    tangent: wp.array[float],
):
    # A free-motion sliding predictor avoids initializing every new material
    # anchor in an incompatible sticking branch. It is only an initial guess;
    # all subsequent corrections solve the unchanged projection equations.
    i = wp.tid()
    row = q.shape[0] + 2 + i
    bound = float(0.0)
    slip = float(0.0)
    if engaged[i] != 0:
        p = wp.vec3(0.0)
        v = wp.vec3(0.0)
        for j in range(4):
            p += bary[i][j] * q[nodes[i][j]]
            v += bary[i][j] * qd[nodes[i][j]]
        closest, t, distance, covered = _shaft_projection(p, poses[0], samples, radius)
        d = closest - p
        d -= wp.dot(d, t) * t
        bound = friction * wp.max(0.0, wp.length(d) - clearance) / compliance
        arm = p - wp.transform_get_translation(poses[0])
        slip = wp.dot(wp.spatial_top(velocities[0]) + wp.cross(wp.spatial_bottom(velocities[0]), arm) - v, t)
    tau = wp.clamp(rho_h * unknown[row][0], -bound, bound)
    if wp.abs(slip) > 1.0e-6:
        tau = wp.sign(slip) * bound
    unknown[row] = wp.vec3(tau / rho_h, 0.0, 0.0)
    tangent[i] = tau


@wp.kernel(enable_backward=False)
def momentum_residual(
    action: wp.array[wp.vec3],
    rhs: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    wrench: wp.array[wp.spatial_vector],
    fixed: wp.array[int],
    unknown: wp.array[wp.vec3],
    blocks: wp.array[wp.mat33],
    rigid_rhs: wp.array[wp.vec3],
    h: float,
    length: float,
    residual: wp.array[wp.vec3],
    statistics: wp.array[wp.float64],
    fem_statistics: wp.array[wp.float64],
):
    i = wp.tid()
    n = fixed.shape[0]
    r = wp.vec3(0.0)
    accurate = wp.vec3d(0.0)
    scale = wp.vec3(0.0)
    if i < n:
        if fixed[i] == 0:
            # Use the ordinary load-before-predictor RHS. Dividing the
            # zero-load RHS first changes both its floating-point equation
            # and the meaning of the configured FEM residual tolerance.
            for axis in range(3):
                right = wp.float64(rhs[i][axis])
                defect = wp.float64(action[i][axis]) - right
                accurate[axis] = defect / wp.float64(h)
                wp.atomic_add(fem_statistics, 0, defect * defect)
                wp.atomic_add(fem_statistics, 1, right * right)
                if not wp.isfinite(defect) or not wp.isfinite(right):
                    wp.atomic_max(fem_statistics, 2, wp.float64(1.0))
            r = wp.vec3(accurate)
            scale = forces[i]
        else:
            r = unknown[i]
            accurate = wp.vec3d(r)
    elif i == n:
        r = blocks[0] * unknown[i] - rigid_rhs[0] - wp.spatial_top(wrench[0])
        accurate = wp.vec3d(r)
        scale = wp.spatial_top(wrench[0])
    else:
        r = blocks[1] * unknown[i] - rigid_rhs[1] - wp.spatial_bottom(wrench[0]) / length
        accurate = wp.vec3d(r)
        scale = wp.spatial_bottom(wrench[0]) / length
    residual[i] = r
    wp.atomic_add(statistics, 2, wp.dot(accurate, accurate))
    if i >= n:
        wp.atomic_add(statistics, 3, wp.dot(accurate, accurate))
    elif fixed[i] != 0:
        wp.atomic_add(statistics, 3, wp.dot(accurate, accurate))
    for j in range(3):
        wp.atomic_max(statistics, 0, wp.abs(accurate[j]))
        wp.atomic_max(statistics, 1, wp.float64(wp.abs(scale[j])))
        if not wp.isfinite(r[j]):
            wp.atomic_max(statistics, 5, wp.float64(1.0))


@wp.kernel(enable_backward=False)
def friction_residual(
    unknown: wp.array[wp.vec3],
    count: int,
    rho_h: float,
    lengths: wp.array[float],
    engaged: wp.array[int],
    tangent: wp.array[float],
    proposed: wp.array[float],
    residual: wp.array[wp.vec3],
    statistics: wp.array[wp.float64],
):
    i = wp.tid()
    row = count + 2 + i
    r = unknown[row]
    if engaged[i] != 0 and lengths[i] > 0.0:
        r[0] = lengths[i] * (tangent[i] - proposed[i])
    residual[row] = r
    accurate = wp.vec3d(r)
    squared = wp.dot(accurate, accurate)
    wp.atomic_add(statistics, 2, squared)
    wp.atomic_add(statistics, 4, squared)


@wp.kernel(enable_backward=False)
def linearize_anchors(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    lengths: wp.array[float],
    engaged: wp.array[int],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    samples: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    tangent_density: wp.array[float],
    radius: float,
    clearance: float,
    compliance: float,
    rho: float,
    mu: float,
    h: float,
    length: float,
    normal_matrix: wp.array[wp.mat33],
    tangent: wp.array[wp.vec3],
    normal_direction: wp.array[wp.vec3],
    lever: wp.array[wp.vec3],
    geometric_lever: wp.array[wp.vec3],
    coefficients: wp.array[wp.vec3],
):
    i = wp.tid()
    normal_matrix[i] = wp.mat33(0.0)
    tangent[i] = wp.vec3(0.0)
    normal_direction[i] = wp.vec3(0.0)
    lever[i] = wp.vec3(0.0)
    geometric_lever[i] = wp.vec3(0.0)
    coefficients[i] = wp.vec3(0.0, 0.0, 1.0)
    if engaged[i] != 0 and lengths[i] > 0.0:
        p = wp.vec3(0.0)
        v = wp.vec3(0.0)
        for j in range(4):
            p += bary[i][j] * q[nodes[i][j]]
            v += bary[i][j] * qd[nodes[i][j]]
        closest, t, distance, covered = _shaft_projection(p, poses[0], samples, radius)
        center = wp.transform_get_translation(poses[0])
        arm = p - center
        d = closest - p
        d -= wp.dot(d, t) * t
        r = wp.length(d)
        normal = wp.float32(0.0)
        if r > clearance:
            e = d / r
            fraction = clearance / r
            P = wp.identity(n=3, dtype=float) - wp.outer(t, t)
            normal_matrix[i] = (lengths[i] * h / compliance) * ((1.0 - fraction) * P + fraction * wp.outer(e, e))
            normal_direction[i] = e
            normal = (r - clearance) / compliance
        relative = wp.spatial_top(velocities[0]) + wp.cross(wp.spatial_bottom(velocities[0]), arm) - v
        y = tangent_density[i] + rho * h * wp.dot(relative, t)
        bound = mu * normal
        sticking = float(bound > 0.0 and wp.abs(y) < bound)
        sliding = wp.float32(0.0)
        if bound > 0.0 and sticking == 0.0:
            sliding = mu * wp.sign(y)
        coefficients[i] = wp.vec3(sticking, sliding, lengths[i] * rho * h)
        tangent[i] = t
        lever[i] = arm / length
        geometric_lever[i] = (closest - center) / length


@wp.func
def relative_action(x: wp.array[wp.vec3], fixed: wp.array[int], ids: wp.vec4i, weights: wp.vec4, arm: wp.vec3):
    n = fixed.shape[0]
    v = x[n] + wp.cross(x[n + 1], arm)
    for j in range(4):
        if fixed[ids[j]] == 0:
            v -= weights[j] * x[ids[j]]
    return v


@wp.func
def scatter(x: wp.array[wp.vec3], fixed: wp.array[int], ids: wp.vec4i, weights: wp.vec4, arm: wp.vec3, force: wp.vec3):
    n = fixed.shape[0]
    for j in range(4):
        if fixed[ids[j]] == 0:
            wp.atomic_add(x, ids[j], -weights[j] * force)
    wp.atomic_add(x, n, force)
    wp.atomic_add(x, n + 1, wp.cross(arm, force))


@wp.kernel(enable_backward=False)
def base_action(
    x: wp.array[wp.vec3],
    fem_action: wp.array[wp.vec3],
    blocks: wp.array[wp.mat33],
    fixed: wp.array[int],
    h: float,
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    n = fixed.shape[0]
    if i < n:
        out[i] = fem_action[i] / h
        if fixed[i] != 0:
            out[i] = x[i]
    elif i < n + 2:
        out[i] = blocks[i - n] * x[i]
    else:
        out[i] = x[i]


@wp.kernel(enable_backward=False)
def contact_action(
    x: wp.array[wp.vec3],
    fixed: wp.array[int],
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    normal_matrix: wp.array[wp.mat33],
    tangent: wp.array[wp.vec3],
    normal_direction: wp.array[wp.vec3],
    lever: wp.array[wp.vec3],
    geometric_lever: wp.array[wp.vec3],
    coefficients: wp.array[wp.vec3],
    out: wp.array[wp.vec3],
    regularization: float,
):
    i = wp.tid()
    row = fixed.shape[0] + 2 + i
    c = coefficients[i]
    v = relative_action(x, fixed, nodes[i], bary[i], lever[i])
    vg = relative_action(x, fixed, nodes[i], bary[i], geometric_lever[i])
    f = normal_matrix[i] * vg
    drag = c[0] * c[2] * (wp.dot(tangent[i], v) + x[row][0]) + c[1] * wp.dot(normal_direction[i], f)
    scatter(out, fixed, nodes[i], bary[i], lever[i], f + drag * tangent[i])
    out[row] = wp.vec3((1.0 + regularization * c[0]) * c[2] * x[row][0] - drag, x[row][1], x[row][2])


@wp.kernel(enable_backward=False)
def base_diagonal(
    diagonal: wp.array[wp.mat33], fixed: wp.array[int], blocks: wp.array[wp.mat33], h: float, out: wp.array[wp.mat33]
):
    i = wp.tid()
    n = fixed.shape[0]
    if i < n:
        out[i] = diagonal[i] / h
        if fixed[i] != 0:
            out[i] = wp.identity(n=3, dtype=float)
    elif i == n:
        out[i] = blocks[0]
    else:
        out[i] = blocks[2]


@wp.kernel(enable_backward=False)
def contact_diagonal(
    fixed: wp.array[int],
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    normal_matrix: wp.array[wp.mat33],
    tangent: wp.array[wp.vec3],
    lever: wp.array[wp.vec3],
    coefficients: wp.array[wp.vec3],
    out: wp.array[wp.mat33],
):
    i = wp.tid()
    n = fixed.shape[0]
    c = coefficients[i]
    K = normal_matrix[i] + c[0] * c[2] * wp.outer(tangent[i], tangent[i])
    for j in range(4):
        if fixed[nodes[i][j]] == 0:
            wp.atomic_add(out, nodes[i][j], bary[i][j] * bary[i][j] * K)
    wp.atomic_add(out, n, K)
    arm = wp.skew(lever[i])
    wp.atomic_add(out, n + 1, -arm * K * arm)


@wp.kernel(enable_backward=False)
def invert(diagonal: wp.array[wp.mat33], inverse: wp.array[wp.mat33]):
    i = wp.tid()
    inverse[i] = inverse_diagonal_block(diagonal[i])


@wp.kernel(enable_backward=False)
def precondition_start(x: wp.array[wp.vec3], inverse: wp.array[wp.mat33], out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = inverse[i] * x[i]


@wp.kernel(enable_backward=False)
def precondition_friction(
    x: wp.array[wp.vec3],
    motion: wp.array[wp.vec3],
    inverse: wp.array[wp.mat33],
    fixed: wp.array[int],
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    normal_matrix: wp.array[wp.mat33],
    tangent: wp.array[wp.vec3],
    normal_direction: wp.array[wp.vec3],
    lever: wp.array[wp.vec3],
    geometric_lever: wp.array[wp.vec3],
    coefficients: wp.array[wp.vec3],
    correction: wp.array[wp.vec3],
    failure: wp.array[int],
    regularization: float,
):
    i = wp.tid()
    n = fixed.shape[0]
    row = n + 2 + i
    c = coefficients[i]
    t = tangent[i]
    arm = wp.cross(lever[i], t)
    mobility = wp.dot(t, inverse[n] * t) + wp.dot(arm, inverse[n + 1] * arm)
    for j in range(4):
        if fixed[nodes[i][j]] == 0:
            mobility += bary[i][j] * bary[i][j] * wp.dot(t, inverse[nodes[i][j]] * t)
    S = (1.0 - c[0] + regularization * c[0]) * c[2] + c[0] * c[2] * c[2] * mobility
    if not wp.isfinite(S) or S <= 0.0 or (c[0] != 0.0 and (not wp.isfinite(mobility) or mobility <= 0.0)):
        wp.atomic_max(failure, 0, 1)
        motion[row] = wp.vec3(0.0)
        return
    v = relative_action(motion, fixed, nodes[i], bary[i], lever[i])
    vg = relative_action(motion, fixed, nodes[i], bary[i], geometric_lever[i])
    Cw = -c[0] * c[2] * wp.dot(t, v) - c[1] * wp.dot(normal_direction[i], normal_matrix[i] * vg)
    z = (x[row][0] - Cw) / S
    if not wp.isfinite(z):
        wp.atomic_max(failure, 0, 1)
        motion[row] = wp.vec3(0.0)
        return
    motion[row] = wp.vec3(z, x[row][1], x[row][2])
    scatter(correction, fixed, nodes[i], bary[i], lever[i], c[0] * c[2] * z * t)


@wp.kernel(enable_backward=False)
def precondition_finish(motion: wp.array[wp.vec3], inverse: wp.array[wp.mat33], correction: wp.array[wp.vec3]):
    i = wp.tid()
    motion[i] -= inverse[i] * correction[i]


@wp.kernel(enable_backward=False)
def combine(x: wp.array[wp.vec3], y: wp.array[wp.vec3], out: wp.array[wp.vec3], alpha: float, beta: float):
    i = wp.tid()
    v = alpha * x[i]
    if beta != 0.0:
        v += beta * y[i]
    out[i] = v


@wp.kernel(enable_backward=False)
def negative(x: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = -x[i]


@wp.kernel(enable_backward=False)
def proposal(base: wp.array[wp.vec3], direction: wp.array[wp.vec3], step: wp.array[float], out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = base[i] + step[0] * direction[i]
