# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device functions and kernels of the thread solver; all arithmetic is float64.

A chain has ``nodes`` nodes with six coordinates each, the position and the
slope. Arrays over nodes, elements and coordinates hold the chains of a batch
one after the other. A chain whose ``status`` is nonzero has finished its step
and is skipped by every kernel.
"""

import warp as wp

Vec6d = wp.types.vector(length=6, dtype=wp.float64)
Mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
_Vec12d = wp.types.vector(length=12, dtype=wp.float64)
_Mat1212d = wp.types.matrix(shape=(12, 12), dtype=wp.float64)


@wp.func
def _gauss_point(index: int) -> wp.float64:
    """Five-point Gauss-Legendre abscissa on [0, 1]."""
    if index == 0:
        return wp.float64(0.0469100770306680036)
    if index == 1:
        return wp.float64(0.2307653449471584545)
    if index == 2:
        return wp.float64(0.5)
    if index == 3:
        return wp.float64(0.7692346550528415455)
    return wp.float64(0.9530899229693319964)


@wp.func
def _gauss_weight(index: int) -> wp.float64:
    if index == 0 or index == 4:
        return wp.float64(0.1184634425280945438)
    if index == 1 or index == 3:
        return wp.float64(0.2393143352496832340)
    return wp.float64(0.2844444444444444444)


@wp.func
def _shape_derivatives(xi: wp.float64, length: wp.float64):
    """First and second derivatives of the cubic Hermite shape functions along the reference length."""
    x2 = xi * xi
    first = wp.vec4d(
        (wp.float64(6.0) * x2 - wp.float64(6.0) * xi) / length,
        wp.float64(1.0) - wp.float64(4.0) * xi + wp.float64(3.0) * x2,
        (wp.float64(6.0) * xi - wp.float64(6.0) * x2) / length,
        -wp.float64(2.0) * xi + wp.float64(3.0) * x2,
    )
    second = wp.vec4d(
        (wp.float64(12.0) * xi - wp.float64(6.0)) / (length * length),
        (-wp.float64(4.0) + wp.float64(6.0) * xi) / length,
        (wp.float64(6.0) - wp.float64(12.0) * xi) / (length * length),
        (-wp.float64(2.0) + wp.float64(6.0) * xi) / length,
    )
    return first, second


@wp.func
def _node_vector(q: wp.array[wp.float64], node: int, offset: int) -> wp.vec3d:
    base = 6 * node + offset
    return wp.vec3d(q[base], q[base + 1], q[base + 2])


@wp.func
def _shape(xi: wp.float64, length: wp.float64) -> wp.vec4d:
    """Cubic Hermite shape functions of an element at ``xi`` in [0, 1]."""
    x2 = xi * xi
    x3 = x2 * xi
    return wp.vec4d(
        wp.float64(1.0) - wp.float64(3.0) * x2 + wp.float64(2.0) * x3,
        length * (xi - wp.float64(2.0) * x2 + x3),
        wp.float64(3.0) * x2 - wp.float64(2.0) * x3,
        length * (-x2 + x3),
    )


@wp.func
def _centerline_point(q: wp.array[wp.float64], node_a: int, shape: wp.vec4d) -> wp.vec3d:
    """Point of the centerline between ``node_a`` and the next node."""
    return (
        shape[0] * _node_vector(q, node_a, 0)
        + shape[1] * _node_vector(q, node_a, 3)
        + shape[2] * _node_vector(q, node_a + 1, 0)
        + shape[3] * _node_vector(q, node_a + 1, 3)
    )


@wp.func
def _centerline_derivatives(q: wp.array[wp.float64], node_a: int, node_b: int, first: wp.vec4d, second: wp.vec4d):
    """r' and r'' of the centerline between two nodes; relative positions keep translations out."""
    position_delta = wp.vec3d(
        q[6 * node_b] - q[6 * node_a], q[6 * node_b + 1] - q[6 * node_a + 1], q[6 * node_b + 2] - q[6 * node_a + 2]
    )
    slope_a = wp.vec3d(q[6 * node_a + 3], q[6 * node_a + 4], q[6 * node_a + 5])
    slope_b = wp.vec3d(q[6 * node_b + 3], q[6 * node_b + 4], q[6 * node_b + 5])
    tangent = first[2] * position_delta + first[1] * slope_a + first[3] * slope_b
    second_derivative = second[2] * position_delta + second[1] * slope_a + second[3] * slope_b
    return tangent, second_derivative


@wp.func
def _curvature_coefficients(
    tangent: wp.vec3d, second: wp.vec3d, rest_a: wp.vec3d, rest_b: wp.vec3d, length: wp.float64
):
    """Bending energy density per unit E I, and its first and second derivatives by the squared curvature.

    A straight reference keeps the quadratic energy kappa²/2. A curved one
    uses (kappa - kappa_0)²/2 with a small offset under the roots.
    """
    metric = wp.dot(tangent, tangent) + wp.float64(1.0e-24)
    cross = wp.cross(tangent, second)
    curvature_squared = wp.dot(cross, cross) / (metric * metric * metric)
    rest_metric = wp.dot(rest_a, rest_a) + wp.float64(1.0e-24)
    rest_cross = wp.cross(rest_a, rest_b)
    rest_curvature_squared = wp.dot(rest_cross, rest_cross) / (rest_metric * rest_metric * rest_metric)
    energy = wp.float64(0.5) * curvature_squared
    first = wp.float64(0.5)
    second_coefficient = wp.float64(0.0)
    if rest_curvature_squared * length * length > wp.float64(1.0e-20):
        epsilon_squared = wp.float64(1.0e-16) / (length * length)
        curvature = wp.sqrt(curvature_squared + epsilon_squared)
        rest_curvature = wp.sqrt(rest_curvature_squared + epsilon_squared)
        delta = curvature - rest_curvature
        energy = wp.float64(0.5) * delta * delta
        first = wp.float64(0.5) * (wp.float64(1.0) - rest_curvature / curvature)
        second_coefficient = wp.float64(0.25) * rest_curvature / (curvature * curvature * curvature)
    return energy, first, second_coefficient


@wp.func
def contact_terms(distance: wp.float64, height: wp.float64, stiffness: wp.float64, smoothing: wp.float64):
    """Smooth contact potential of one node and its first two derivatives by its height above the contact plane.

    The node centre is kept ``distance`` above the plane. With depth
    d = distance - height the penetration is p = (d + sqrt(d² + s²)) / 2 and
    the potential stiffness p² / 2.
    """
    depth = distance - height
    root = wp.sqrt(depth * depth + smoothing * smoothing)
    penetration = wp.float64(0.5) * (depth + root)
    first = wp.float64(0.5) * (wp.float64(1.0) + depth / root)
    second = wp.float64(0.5) * smoothing * smoothing / (root * root * root)
    return wp.vec3d(
        wp.float64(0.5) * stiffness * penetration * penetration,
        -stiffness * penetration * first,
        stiffness * (first * first + penetration * second),
    )


@wp.func
def _plane_offset(position: wp.vec3d, anchor: wp.vec3d, normal: wp.vec3d) -> wp.vec3d:
    """From a node's friction anchor to the node, along its contact plane."""
    relative = position - anchor
    return relative - wp.dot(relative, normal) * normal


@wp.func
def _friction_energy(slip: wp.float64, stiffness: wp.float64, limit: wp.float64) -> wp.float64:
    """Friction potential of a slip distance: a spring up to the Coulomb limit, the limit force beyond."""
    reach = limit / stiffness
    if slip <= reach:
        return wp.float64(0.5) * stiffness * slip * slip
    return limit * (slip - wp.float64(0.5) * reach)


@wp.func
def _friction_scale(slip: wp.float64, stiffness: wp.float64, limit: wp.float64) -> wp.float64:
    """Friction force per slip distance: the spring's stiffness while sticking, the limit over the slip while sliding.

    It is also the stiffness of the spring that has the same force at this
    slip and whose energy lies above the friction potential everywhere else.
    Used as the tangent, it makes a Newton update a step that cannot raise
    the friction potential, where the potential's own curvature, zero along
    a sliding direction, would let the update overshoot.
    """
    if stiffness * slip <= limit:
        return stiffness
    return limit / slip


@wp.func
def _stack(position: wp.vec3d, slope: wp.vec3d) -> Vec6d:
    return Vec6d(position[0], position[1], position[2], slope[0], slope[1], slope[2])


@wp.func
def _node_values(q: wp.array[wp.float64], node: int) -> Vec6d:
    return _stack(_node_vector(q, node, 0), _node_vector(q, node, 3))


@wp.func
def _blocks(a: wp.mat33d, b: wp.mat33d, c: wp.mat33d, d: wp.mat33d) -> Mat66d:
    """Assemble [[a, b], [c, d]]."""
    result = Mat66d()
    for i in range(3):
        for j in range(3):
            result[i, j] = a[i, j]
            result[i, j + 3] = b[i, j]
            result[i + 3, j] = c[i, j]
            result[i + 3, j + 3] = d[i, j]
    return result


@wp.func
def _add_mass(matrix: Mat66d, mass: wp.mat22d, scale: wp.float64) -> Mat66d:
    """Add scale * (mass kron I3): the Hermite mass couples equal axes only."""
    result = matrix
    for axis in range(3):
        result[axis, axis] += scale * mass[0, 0]
        result[axis, axis + 3] += scale * mass[0, 1]
        result[axis + 3, axis] += scale * mass[1, 0]
        result[axis + 3, axis + 3] += scale * mass[1, 1]
    return result


@wp.func
def _inverse(matrix: Mat66d) -> Mat66d:
    """Gauss-Jordan inverse without pivoting, for the positive definite node blocks."""
    a = matrix
    inverse = wp.identity(n=6, dtype=wp.float64)
    for column in range(6):
        pivot = wp.float64(1.0) / a[column, column]
        for j in range(6):
            a[column, j] = a[column, j] * pivot
            inverse[column, j] = inverse[column, j] * pivot
        for row in range(6):
            if row != column:
                factor = a[row, column]
                for j in range(6):
                    a[row, j] = a[row, j] - factor * a[column, j]
                    inverse[row, j] = inverse[row, j] - factor * inverse[column, j]
    return inverse


@wp.func
def _head_coordinates(center: wp.vec3d, arm: wp.vec3d, slope: wp.vec3d, motion: Vec6d):
    """Attachment point, slope and lever of a head moved by (translation, rotation vector) from its predicted pose."""
    turn = wp.vec3d(motion[3], motion[4], motion[5])
    angle = wp.length(turn)
    lever, tangent = arm, slope
    if angle > wp.float64(0.0):
        rotation = wp.quat_from_axis_angle(turn / angle, angle)
        lever = wp.quat_rotate(rotation, arm)
        tangent = wp.quat_rotate(rotation, slope)
    return center + wp.vec3d(motion[0], motion[1], motion[2]) + lever, tangent, lever


@wp.func
def _head_jacobian(lever: wp.vec3d, tangent: wp.vec3d, motion: Vec6d) -> Mat66d:
    """Node-0 coordinates per unit head motion, exact in the rotation vector.

    A change dtheta of the rotation vector theta spins the head by
    T(theta) dtheta, with the tangent operator of the rotation group

        T = I + (1 - cos a) / a² [theta]x + (a - sin a) / a³ [theta]x²,  a = |theta|,

    so dp = du + (T dtheta) x lever and ds = (T dtheta) x tangent. With T the
    head's residual rows are the gradient of the step potential in the head's
    coordinates, which the line search needs once the head turns within a step.
    """
    turn = wp.vec3d(motion[3], motion[4], motion[5])
    angle = wp.length(turn)
    # Series of the two coefficients below the angle where 1 - cos a loses half of its digits.
    first = wp.float64(0.5) - angle * angle / wp.float64(24.0)
    second = wp.float64(1.0) / wp.float64(6.0) - angle * angle / wp.float64(120.0)
    if angle > wp.float64(1.0e-4):
        first = (wp.float64(1.0) - wp.cos(angle)) / (angle * angle)
        second = (angle - wp.sin(angle)) / (angle * angle * angle)
    skew_turn = wp.skew(turn)
    spin = wp.identity(3, dtype=wp.float64) + first * skew_turn + second * skew_turn * skew_turn
    position_turn, slope_turn = -wp.skew(lever) * spin, -wp.skew(tangent) * spin
    jacobian = Mat66d()
    for i in range(3):
        jacobian[i, i] = wp.float64(1.0)
        for j in range(3):
            jacobian[i, j + 3] = position_turn[i, j]
            jacobian[i + 3, j + 3] = slope_turn[i, j]
    return jacobian


@wp.kernel(enable_backward=False)
def predict(
    q: wp.array[wp.float64],
    qd: wp.array[wp.float64],
    boundary_q: wp.array[wp.float64],
    fixed: wp.array[int],
    dt: wp.float64,
    predicted_q: wp.array[wp.float64],
):
    """Store the inertial target q + dt qd and move q to the first iterate; launch over all coordinates."""
    i = wp.tid()
    predicted_q[i] = q[i] + dt * qd[i]
    if fixed[i % fixed.shape[0]] != 0:
        q[i] = boundary_q[i]
    else:
        q[i] = predicted_q[i]


@wp.kernel(enable_backward=False)
def evaluate_elements(
    q: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    axial_stiffness: wp.array[wp.float64],
    bending_stiffness: wp.array[wp.float64],
    rest_q: wp.array[wp.float64],
    tangent_blocks: bool,
    status: wp.array[int],
    gradient: wp.array2d[wp.vec3d],
    hessian: wp.array3d[wp.mat33d],
):
    """Elastic energy gradient and positive semidefinite tangent as 3-by-3 blocks; launch over all elements.

    Blocks follow the element's nodal vectors [position_a, slope_a, position_b,
    slope_b]. Each Gauss point is evaluated once and differentiated per vector.
    With k = r' x r'' / |r'|³ the curvature vector, the bending tangent keeps
    (dk)^T (dk) and drops k . d²k; the axial tangent drops the geometric term
    in compression. Without ``tangent_blocks`` only the gradient is written.
    Sums are kept in thread-local storage and written once.
    """
    index = wp.tid()
    per_chain = lengths.shape[0]
    if status[index // per_chain] != 0:
        return
    element = index % per_chain
    node_a = (index // per_chain) * (per_chain + 1) + element
    node_b = node_a + 1
    length = lengths[element]
    axial, bending = axial_stiffness[element], bending_stiffness[element]
    zero = wp.float64(0.0)
    two = wp.float64(2.0)
    identity = wp.identity(n=3, dtype=wp.float64)
    force = _Vec12d()
    tangent_sum = _Mat1212d()
    points = int(5)
    for point in range(points):
        first, second = _shape_derivatives(_gauss_point(point), length)
        tangent, curvature = _centerline_derivatives(q, node_a, node_b, first, second)
        rest_a, rest_b = _centerline_derivatives(rest_q, element, element + 1, first, second)
        rest_metric = wp.dot(rest_a, rest_a)
        metric = wp.dot(tangent, tangent) + wp.float64(1.0e-24)
        inv_metric = wp.float64(1.0) / metric
        inv_cube = inv_metric * inv_metric * inv_metric
        strain = wp.float64(0.5) * (wp.dot(tangent, tangent) / rest_metric - wp.float64(1.0))
        cross = wp.cross(tangent, curvature)
        numerator = wp.dot(cross, cross)
        _, bending_first, bending_second = _curvature_coefficients(tangent, curvature, rest_a, rest_b, length)
        weight = length * _gauss_weight(point) * wp.sqrt(rest_metric)
        skew_tangent, skew_curvature = wp.skew(tangent), wp.skew(curvature)
        vector_scale = inv_metric * wp.sqrt(inv_metric)
        tension = wp.max(strain, zero)
        stiffening = wp.max(bending_first, zero)
        # Shared by every nodal vector: d(numerator) and d(k) along r' and along r''.
        numerator_first = two * wp.cross(curvature, cross)
        numerator_second = two * wp.cross(cross, tangent)
        vector_first = vector_scale * (-skew_curvature - wp.float64(3.0) * inv_metric * wp.outer(cross, tangent))
        vector_second = vector_scale * skew_tangent
        for b in range(4):
            # Along coordinate e of vector b: d(r') = first[b] e and d(r'') = second[b] e.
            strain_b = (first[b] / rest_metric) * tangent
            numerator_b = first[b] * numerator_first + second[b] * numerator_second
            curvature_b = inv_cube * (numerator_b - wp.float64(6.0) * numerator * inv_metric * first[b] * tangent)
            value = weight * (axial * strain * strain_b + bending * bending_first * curvature_b)
            for i in range(3):
                force[3 * b + i] += value[i]
            if tangent_blocks:
                vector_b = first[b] * vector_first + second[b] * vector_second
                for c in range(b, 4):
                    strain_c = (first[c] / rest_metric) * tangent
                    numerator_c = first[c] * numerator_first + second[c] * numerator_second
                    curvature_c = inv_cube * (
                        numerator_c - wp.float64(6.0) * numerator * inv_metric * first[c] * tangent
                    )
                    vector_c = first[c] * vector_first + second[c] * vector_second
                    block = weight * (
                        axial
                        * (wp.outer(strain_b, strain_c) + tension * (first[b] * first[c] / rest_metric) * identity)
                        + bending
                        * (
                            bending_second * wp.outer(curvature_b, curvature_c)
                            + two * stiffening * wp.transpose(vector_b) * vector_c
                        )
                    )
                    for i in range(3):
                        for j in range(3):
                            tangent_sum[3 * b + i, 3 * c + j] += block[i, j]
    for b in range(4):
        gradient[index, b] = wp.vec3d(force[3 * b], force[3 * b + 1], force[3 * b + 2])
        if tangent_blocks:
            for c in range(4):
                block = wp.mat33d()
                for i in range(3):
                    for j in range(3):
                        if c >= b:
                            block[i, j] = tangent_sum[3 * b + i, 3 * c + j]
                        else:
                            block[i, j] = tangent_sum[3 * c + j, 3 * b + i]
                hessian[index, b, c] = block


@wp.func
def _kinematics(
    q: wp.array[wp.float64],
    shift: int,
    predicted_q: wp.array[wp.float64],
    qd: wp.array[wp.float64],
    node: int,
    offset: int,
    dt: wp.float64,
):
    """Backward Euler acceleration and velocity of one nodal vector; q is read ``shift`` nodes further on.

    With the inertial target q~ = q_n + dt v_n: a = (q - q~) / dt² and
    v = v_n + dt a = (q - q_n) / dt.
    """
    acceleration = (_node_vector(q, node + shift, offset) - _node_vector(predicted_q, node, offset)) / (dt * dt)
    velocity = _node_vector(qd, node, offset) + dt * acceleration
    return acceleration, velocity


@wp.func
def _node_dynamics(
    q: wp.array[wp.float64],
    shift: int,
    predicted_q: wp.array[wp.float64],
    qd: wp.array[wp.float64],
    mass_diagonal: wp.array[wp.mat22d],
    mass_coupling: wp.array[wp.mat22d],
    gravity: wp.vec3d,
    dt: wp.float64,
    damping: wp.float64,
    node: int,
):
    """Inertia M a, damping c M v and weight on one node [N, N m]; q is read ``shift`` nodes further on."""
    count = mass_diagonal.shape[0]
    local = node % count
    mass = mass_diagonal[local]
    a_p, v_p = _kinematics(q, shift, predicted_q, qd, node, 0, dt)
    a_s, v_s = _kinematics(q, shift, predicted_q, qd, node, 3, dt)
    inertia_p = mass[0, 0] * a_p + mass[0, 1] * a_s
    inertia_s = mass[1, 0] * a_p + mass[1, 1] * a_s
    drag_p = mass[0, 0] * v_p + mass[0, 1] * v_s
    drag_s = mass[1, 0] * v_p + mass[1, 1] * v_s
    weight_p = mass[0, 0] * gravity
    weight_s = mass[1, 0] * gravity
    if local > 0:
        # Rows of this node, columns of the previous one: the transposed coupling block.
        coupling = mass_coupling[local - 1]
        b_p, w_p = _kinematics(q, shift, predicted_q, qd, node - 1, 0, dt)
        b_s, w_s = _kinematics(q, shift, predicted_q, qd, node - 1, 3, dt)
        inertia_p += coupling[0, 0] * b_p + coupling[1, 0] * b_s
        inertia_s += coupling[0, 1] * b_p + coupling[1, 1] * b_s
        drag_p += coupling[0, 0] * w_p + coupling[1, 0] * w_s
        drag_s += coupling[0, 1] * w_p + coupling[1, 1] * w_s
        weight_p += coupling[0, 0] * gravity
        weight_s += coupling[0, 1] * gravity
    if local < count - 1:
        coupling = mass_coupling[local]
        b_p, w_p = _kinematics(q, shift, predicted_q, qd, node + 1, 0, dt)
        b_s, w_s = _kinematics(q, shift, predicted_q, qd, node + 1, 3, dt)
        inertia_p += coupling[0, 0] * b_p + coupling[0, 1] * b_s
        inertia_s += coupling[1, 0] * b_p + coupling[1, 1] * b_s
        drag_p += coupling[0, 0] * w_p + coupling[0, 1] * w_s
        drag_s += coupling[1, 0] * w_p + coupling[1, 1] * w_s
        weight_p += coupling[0, 0] * gravity
        weight_s += coupling[1, 0] * gravity
    return _stack(inertia_p, inertia_s), damping * _stack(drag_p, drag_s), _stack(weight_p, weight_s)


@wp.func
def _node_elastic(element_gradient: wp.array2d[wp.vec3d], count: int, node: int) -> Vec6d:
    """Elastic energy gradient on one node from its two elements [N, N m]."""
    local = node % count
    element = (node // count) * (count - 1) + local
    position, slope = wp.vec3d(wp.float64(0.0)), wp.vec3d(wp.float64(0.0))
    if local > 0:
        position += element_gradient[element - 1, 2]
        slope += element_gradient[element - 1, 3]
    if local < count - 1:
        position += element_gradient[element, 0]
        slope += element_gradient[element, 1]
    return _stack(position, slope)


@wp.kernel(enable_backward=False)
def assemble(
    q: wp.array[wp.float64],
    predicted_q: wp.array[wp.float64],
    qd: wp.array[wp.float64],
    element_gradient: wp.array2d[wp.vec3d],
    element_hessian: wp.array3d[wp.mat33d],
    mass_diagonal: wp.array[wp.mat22d],
    mass_coupling: wp.array[wp.mat22d],
    fixed: wp.array[int],
    status: wp.array[int],
    tangent_blocks: bool,
    gravity: wp.vec3d,
    dt: wp.float64,
    damping: wp.float64,
    inertia_scale: wp.float64,
    contact: bool,
    contact_distance: wp.float64,
    contact_stiffness: wp.float64,
    contact_smoothing: wp.float64,
    contact_point: wp.array[wp.vec3d],
    contact_normal: wp.array[wp.vec3d],
    contact_limit: wp.array[wp.float64],
    contact_anchor: wp.array[wp.vec3d],
    rigid: bool,
    head_center: wp.array[wp.vec3d],
    head_arm: wp.array[wp.vec3d],
    head_slope: wp.array[wp.vec3d],
    head_inertia: wp.array[Mat66d],
    head_motion: wp.array[Vec6d],
    diagonal: wp.array[Mat66d],
    coupling: wp.array[Mat66d],
    residual: wp.array[Vec6d],
):
    """Step residual and, with ``tangent_blocks``, effective tangent blocks of one node; launch over all nodes.

    The residual is M a + c M v + elastic force - weight + contact force. A
    node touches its own contact plane; a zero normal switches its contact
    off, and node 0 has none because its support carries it. A node in
    contact also has friction along its plane: a spring to its anchor whose
    force stops growing at the node's Coulomb limit.
    ``coupling[node]`` has the rows of this node and the columns of the next.
    Fixed coordinates become identity rows, except on a chain's node 0, whose
    rows stay complete because the solve condenses the chain onto them. With a
    rigid head those rows are expressed in the head's translation and rotation
    and include its inertia.
    """
    node = wp.tid()
    count = mass_diagonal.shape[0]
    if status[node // count] != 0:
        return
    local = node % count
    element = (node // count) * (count - 1) + local
    inertia, drag, weight = _node_dynamics(
        q, 0, predicted_q, qd, mass_diagonal, mass_coupling, gravity, dt, damping, node
    )
    internal = _node_elastic(element_gradient, count, node) + drag
    value = Vec6d()
    for i in range(6):
        value[i] = inertia[i] + (internal[i] - weight[i])
    contact_tangent = wp.mat33d()
    if contact and local > 0:
        normal = contact_normal[node]
        if wp.dot(normal, normal) > wp.float64(0.0):
            position = _node_vector(q, node, 0)
            terms = contact_terms(
                contact_distance, wp.dot(normal, position - contact_point[node]), contact_stiffness, contact_smoothing
            )
            for i in range(3):
                value[i] += terms[1] * normal[i]
            contact_tangent = terms[2] * wp.outer(normal, normal)
            limit = contact_limit[node]
            if limit > wp.float64(0.0):
                offset = _plane_offset(position, contact_anchor[node], normal)
                scale = _friction_scale(wp.length(offset), contact_stiffness, limit)
                for i in range(3):
                    value[i] += scale * offset[i]
                contact_tangent += scale * (wp.identity(n=3, dtype=wp.float64) - wp.outer(normal, normal))
    for i in range(6):
        if local > 0 and fixed[6 * local + i] != 0:
            value[i] = wp.float64(0.0)
    chain = node // count
    jacobian = Mat66d()
    if rigid and local == 0:
        _, tangent, lever = _head_coordinates(
            head_center[chain], head_arm[chain], head_slope[chain], head_motion[chain]
        )
        jacobian = _head_jacobian(lever, tangent, head_motion[chain])
        value = wp.transpose(jacobian) * value + head_inertia[chain] * head_motion[chain]
    residual[node] = value
    if not tangent_blocks:
        return
    zero = wp.mat33d(wp.float64(0.0))
    k_pp, k_ps, k_sp, k_ss = zero, zero, zero, zero
    if local > 0:
        k_pp += element_hessian[element - 1, 2, 2]
        k_ps += element_hessian[element - 1, 2, 3]
        k_sp += element_hessian[element - 1, 3, 2]
        k_ss += element_hessian[element - 1, 3, 3]
    next_block = Mat66d()
    if local < count - 1:
        k_pp += element_hessian[element, 0, 0]
        k_ps += element_hessian[element, 0, 1]
        k_sp += element_hessian[element, 1, 0]
        k_ss += element_hessian[element, 1, 1]
        next_block = _blocks(
            element_hessian[element, 0, 2],
            element_hessian[element, 0, 3],
            element_hessian[element, 1, 2],
            element_hessian[element, 1, 3],
        )
        next_block = _add_mass(next_block, mass_coupling[local], inertia_scale)
    block = _add_mass(_blocks(k_pp, k_ps, k_sp, k_ss), mass_diagonal[local], inertia_scale)
    for i in range(3):
        for j in range(3):
            block[i, j] += contact_tangent[i, j]
    for i in range(6):
        if local > 0 and fixed[6 * local + i] != 0:
            for j in range(6):
                block[i, j] = wp.float64(0.0)
                block[j, i] = wp.float64(0.0)
                next_block[i, j] = wp.float64(0.0)
            block[i, i] = wp.float64(1.0)
        if local < count - 1 and fixed[6 * (local + 1) + i] != 0:
            for j in range(6):
                next_block[j, i] = wp.float64(0.0)
    if rigid and local == 0:
        # The Jacobian's own derivative is dropped: the thread's load times the lever is small beside the inertia,
        # and the block stays positive semidefinite.
        block = wp.transpose(jacobian) * block * jacobian + head_inertia[chain]
        next_block = wp.transpose(jacobian) * next_block
    diagonal[node] = block
    coupling[node] = next_block


@wp.kernel(enable_backward=False)
def solve_chains(
    nodes: int,
    rigid: bool,
    status: wp.array[int],
    diagonal: wp.array[Mat66d],
    coupling: wp.array[Mat66d],
    residual: wp.array[Vec6d],
    inverse: wp.array[Mat66d],
    reduced: wp.array[Vec6d],
    update: wp.array[Vec6d],
):
    """Block-tridiagonal elimination from the far end toward node 0; one thread per chain.

    A prescribed node 0 has a zero update. A rigid head is solved with the
    Schur complement that the elimination leaves on node 0, and its update is
    a translation and a rotation.
    """
    chain = wp.tid()
    if status[chain] != 0:
        return
    first = chain * nodes
    last = first + nodes - 1
    block_inverse = _inverse(diagonal[last])
    load = Vec6d() + residual[last]
    inverse[last] = block_inverse
    reduced[last] = load
    for step in range(1, nodes - 1):
        node = last - step
        carry = coupling[node] * block_inverse
        block_inverse = _inverse(diagonal[node] - carry * wp.transpose(coupling[node]))
        load = residual[node] - carry * load
        inverse[node] = block_inverse
        reduced[node] = load
    carry = coupling[first] * block_inverse
    head_change = Vec6d()
    if rigid:
        complement = diagonal[first] - carry * wp.transpose(coupling[first])
        head_change = -(_inverse(complement) * (residual[first] - carry * load))
    update[first] = head_change
    change = -(block_inverse * (load + wp.transpose(coupling[first]) * head_change))
    update[first + 1] = change
    for node in range(first + 2, last + 1):
        change = -(inverse[node] * (reduced[node] + wp.transpose(coupling[node - 1]) * change))
        update[node] = change


# Cyclic reduction: the same solve for a GPU, where one thread per chain is slow because it eliminates the
# chain's blocks one after the other. Every node keeps its six rows of the system as REDUCTION_COLUMNS columns:
# its diagonal block, its coupling block to the next node at the current stride, and its load. One level
# eliminates, in all nodes at once, each node's two neighbours at the stride with their own rows; the node is
# then coupled to the nodes twice as far away. The stride doubles until no neighbour is left in the chain. Every
# level leaves Schur complements of the chain's tangent: they stay symmetric, so one coupling block per node is
# enough, and positive definite, so a diagonal block needs no pivoting. One GPU thread is slow on a float64
# block, so a thread computes one column or one entry of a block.
_COUPLING = 6
_LOAD = 12
REDUCTION_COLUMNS = 13


@wp.func
def _solve_block(rows: wp.array3d[wp.float64], node: int, load: Vec6d) -> Vec6d:
    """Solve a node's diagonal block for one load by Gaussian elimination without pivoting.

    Both triangles of the block are read, as ``_inverse`` does: the block of a
    rigid head is symmetric only up to the rounding of the head's inertia.
    """
    block = Mat66d()
    for i in range(6):
        for j in range(6):
            block[i, j] = rows[node, i, j]
    inverse_pivot = Vec6d()
    result = load
    for k in range(6):
        inverse_pivot[k] = wp.float64(1.0) / block[k, k]
        for i in range(k + 1, 6):
            multiple = block[i, k] * inverse_pivot[k]
            for j in range(k + 1, 6):
                block[i, j] -= multiple * block[k, j]
            result[i] -= multiple * result[k]
    for step in range(6):
        i = 5 - step
        value = result[i]
        for j in range(i + 1, 6):
            value -= block[i, j] * result[j]
        result[i] = value * inverse_pivot[i]
    return result


@wp.kernel(enable_backward=False)
def begin_reduction(
    nodes: int,
    rigid: bool,
    status: wp.array[int],
    diagonal: wp.array[Mat66d],
    coupling: wp.array[Mat66d],
    residual: wp.array[Vec6d],
    rows: wp.array3d[wp.float64],
):
    """Write one node's rows of the chain system for the cyclic reduction; launch over all nodes.

    A prescribed node 0 becomes an identity block without coupling and load:
    its update is zero and the other nodes do not see it. With a rigid head
    node 0 is a node like the others.
    """
    node = wp.tid()
    if status[node // nodes] != 0:
        return
    local = node % nodes
    block = diagonal[node]
    next_block = coupling[node]
    load = residual[node]
    if local == 0 and not rigid:
        block = wp.identity(n=6, dtype=wp.float64)
        next_block = Mat66d()
        load = Vec6d()
    if local == nodes - 1:
        next_block = Mat66d()
    for i in range(6):
        for j in range(6):
            rows[node, i, j] = block[i, j]
            rows[node, i, _COUPLING + j] = next_block[i, j]
        rows[node, i, _LOAD] = load[i]


@wp.kernel(enable_backward=False)
def condense_nodes(
    nodes: int,
    stride: int,
    status: wp.array[int],
    rows: wp.array3d[wp.float64],
    condensed: wp.array3d[wp.float64],
):
    """One column of a node's couplings and load, solved by its diagonal block; launch (all nodes, columns).

    With B the node's diagonal block, columns 0 to 5 are B^-1 C for its
    coupling C to the next node at ``stride``, columns 6 to 11 are B^-1 P^T for
    the coupling P of the previous node to this one, and column 12 is B^-1 r
    for its load r. A coupling beyond either end of the chain is zero.
    """
    node, column = wp.tid()
    if status[node // nodes] != 0:
        return
    load = Vec6d()
    if column < _COUPLING:
        for i in range(6):
            load[i] = rows[node, i, _COUPLING + column]
    elif column == _LOAD:
        for i in range(6):
            load[i] = rows[node, i, _LOAD]
    elif node % nodes >= stride:
        for i in range(6):
            load[i] = rows[node - stride, column - _COUPLING, _COUPLING + i]
    result = _solve_block(rows, node, load)
    for i in range(6):
        condensed[node, i, column] = result[i]


@wp.kernel(enable_backward=False)
def reduce_nodes(
    nodes: int,
    stride: int,
    status: wp.array[int],
    rows: wp.array3d[wp.float64],
    condensed: wp.array3d[wp.float64],
    reduced: wp.array3d[wp.float64],
):
    """One entry of a node's rows after its neighbours at ``stride`` are eliminated; launch (all nodes, 6, columns).

    With the names of ``condense_nodes`` and the indices -, + for the previous
    and the next node at the stride, the node's rows become

        B - P^T (B^-1 C)_- - C (B^-1 P^T)_+,   -C (B^-1 C)_+,   r - P^T (B^-1 r)_- - C (B^-1 r)_+,

    where the new coupling reaches the node two strides on.
    """
    node, row, column = wp.tid()
    if status[node // nodes] != 0:
        return
    local = node % nodes
    value = wp.float64(0.0)
    if column < _COUPLING or column == _LOAD:
        value = rows[node, row, column]
        if local >= stride:
            for k in range(6):
                value -= rows[node - stride, k, _COUPLING + row] * condensed[node - stride, k, column]
        if local + stride < nodes:
            source = column
            if column < _COUPLING:
                source = _COUPLING + column
            for k in range(6):
                value -= rows[node, row, _COUPLING + k] * condensed[node + stride, k, source]
    elif local + stride < nodes:
        for k in range(6):
            value -= rows[node, row, _COUPLING + k] * condensed[node + stride, k, column - _COUPLING]
    reduced[node, row, column] = value


@wp.kernel(enable_backward=False)
def solve_reduced(
    nodes: int,
    rigid: bool,
    status: wp.array[int],
    rows: wp.array3d[wp.float64],
    update: wp.array[Vec6d],
):
    """Newton update -B^-1 r of one node from its uncoupled rows; launch over all nodes.

    A prescribed node 0 has a zero update. The update of a rigid head is a
    translation and a rotation.
    """
    node = wp.tid()
    if status[node // nodes] != 0:
        return
    change = Vec6d()
    if rigid or node % nodes > 0:
        load = Vec6d()
        for i in range(6):
            load[i] = rows[node, i, _LOAD]
        change = -_solve_block(rows, node, load)
    update[node] = change


@wp.kernel(enable_backward=False)
def trial_coordinates(
    q: wp.array[wp.float64],
    update: wp.array[Vec6d],
    scales: wp.array[wp.float64],
    nodes: int,
    status: wp.array[int],
    rigid: bool,
    head_center: wp.array[wp.vec3d],
    head_arm: wp.array[wp.vec3d],
    head_slope: wp.array[wp.vec3d],
    head_motion: wp.array[Vec6d],
    trial_q: wp.array[wp.float64],
):
    """Coordinates of one node at each candidate multiple of the Newton update; launch (candidates, nodes)."""
    candidate, node = wp.tid()
    chain = node // nodes
    if status[chain] != 0:
        return
    base = candidate * q.shape[0] + 6 * node
    if rigid and node % nodes == 0:
        position, tangent, _ = _head_coordinates(
            head_center[chain],
            head_arm[chain],
            head_slope[chain],
            head_motion[chain] + scales[candidate] * update[node],
        )
        for i in range(3):
            trial_q[base + i] = position[i]
            trial_q[base + 3 + i] = tangent[i]
    else:
        for i in range(6):
            trial_q[base + i] = q[6 * node + i] + scales[candidate] * update[node][i]


@wp.kernel(enable_backward=False)
def clear(values: wp.array[wp.float64]):
    values[wp.tid()] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def trial_element_energy(
    trial_q: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    axial_stiffness: wp.array[wp.float64],
    bending_stiffness: wp.array[wp.float64],
    rest_q: wp.array[wp.float64],
    total_nodes: int,
    status: wp.array[int],
    energy: wp.array2d[wp.float64],
):
    """Elastic energy of one element at one candidate; launch (candidates, elements)."""
    candidate, index = wp.tid()
    per_chain = lengths.shape[0]
    element = index % per_chain
    chain = index // per_chain
    if status[chain] != 0:
        return
    node_a = candidate * total_nodes + chain * (per_chain + 1) + element
    length = lengths[element]
    result = wp.float64(0.0)
    for point in range(5):
        first, second = _shape_derivatives(_gauss_point(point), length)
        tangent, curvature = _centerline_derivatives(trial_q, node_a, node_a + 1, first, second)
        rest_a, rest_b = _centerline_derivatives(rest_q, element, element + 1, first, second)
        rest_metric = wp.dot(rest_a, rest_a)
        strain = wp.float64(0.5) * (wp.dot(tangent, tangent) / rest_metric - wp.float64(1.0))
        bending_energy, _, _ = _curvature_coefficients(tangent, curvature, rest_a, rest_b, length)
        weight = length * _gauss_weight(point) * wp.sqrt(rest_metric)
        result += weight * (
            wp.float64(0.5) * axial_stiffness[element] * strain * strain + bending_stiffness[element] * bending_energy
        )
    wp.atomic_add(energy, chain, candidate, result)


@wp.kernel(enable_backward=False)
def trial_node_energy(
    trial_q: wp.array[wp.float64],
    predicted_q: wp.array[wp.float64],
    qd: wp.array[wp.float64],
    mass_diagonal: wp.array[wp.mat22d],
    mass_coupling: wp.array[wp.mat22d],
    gravity: wp.vec3d,
    dt: wp.float64,
    damping: wp.float64,
    contact: bool,
    contact_distance: wp.float64,
    contact_stiffness: wp.float64,
    contact_smoothing: wp.float64,
    contact_point: wp.array[wp.vec3d],
    contact_normal: wp.array[wp.vec3d],
    contact_limit: wp.array[wp.float64],
    contact_anchor: wp.array[wp.vec3d],
    total_nodes: int,
    status: wp.array[int],
    rigid: bool,
    head_inertia: wp.array[Mat66d],
    head_motion: wp.array[Vec6d],
    update: wp.array[Vec6d],
    scales: wp.array[wp.float64],
    energy: wp.array2d[wp.float64],
):
    """Inertial, damping, weight and contact part of the step potential of one node; launch (candidates, nodes).

    Its gradient with the elastic energy is the step residual, so a candidate
    with a lower total is a better iterate.
    """
    candidate, node = wp.tid()
    if status[node // mass_diagonal.shape[0]] != 0:
        return
    shift = candidate * total_nodes
    inertia, drag, weight = _node_dynamics(
        trial_q, shift, predicted_q, qd, mass_diagonal, mass_coupling, gravity, dt, damping, node
    )
    # Damping at the start of the step: the damping potential is quadratic between the two.
    predicted_drag = drag - (damping * dt) * inertia
    position = _node_values(trial_q, node + shift)
    increment = position - _node_values(predicted_q, node)
    value = wp.float64(0.5) * wp.dot(increment, inertia) + (
        wp.float64(0.5) * wp.dot(increment, drag + predicted_drag) - wp.dot(position, weight)
    )
    if contact and node % mass_diagonal.shape[0] > 0:
        normal = contact_normal[node]
        if wp.dot(normal, normal) > wp.float64(0.0):
            point = wp.vec3d(position[0], position[1], position[2])
            height = wp.dot(normal, point - contact_point[node])
            value += contact_terms(contact_distance, height, contact_stiffness, contact_smoothing)[0]
            if contact_limit[node] > wp.float64(0.0):
                slip = wp.length(_plane_offset(point, contact_anchor[node], normal))
                value += _friction_energy(slip, contact_stiffness, contact_limit[node])
    chain = node // mass_diagonal.shape[0]
    if rigid and node % mass_diagonal.shape[0] == 0:
        # Moving the head's inertia away from its predicted pose within the step.
        motion = head_motion[chain] + scales[candidate] * update[node]
        value += wp.float64(0.5) * wp.dot(motion, head_inertia[chain] * motion)
    wp.atomic_add(energy, chain, candidate, value)


@wp.kernel(enable_backward=False)
def apply_best(
    update: wp.array[Vec6d],
    scales: wp.array[wp.float64],
    energy: wp.array2d[wp.float64],
    nodes: int,
    unit: int,
    status: wp.array[int],
    rigid: bool,
    head_center: wp.array[wp.vec3d],
    head_arm: wp.array[wp.vec3d],
    head_slope: wp.array[wp.vec3d],
    head_motion: wp.array[Vec6d],
    q: wp.array[wp.float64],
):
    """Take, per chain, the candidate with the lowest step potential; launch over all nodes.

    The full Newton update, candidate ``unit``, is kept unless another
    candidate is lower by more than the rounding of the potential's sum. Near
    convergence the candidates differ by less than that, and the Newton update
    is then the accurate one. One candidate is a zero step, so the potential
    does not increase beyond rounding.
    """
    node = wp.tid()
    chain = node // nodes
    if status[chain] != 0:
        return
    best = unit
    for candidate in range(scales.shape[0]):
        rounding = wp.float64(1.0e-13) * (wp.abs(energy[chain, candidate]) + wp.abs(energy[chain, best]))
        if energy[chain, candidate] < energy[chain, best] - rounding:
            best = candidate
    if rigid and node % nodes == 0:
        motion = head_motion[chain] + scales[best] * update[node]
        position, tangent, _ = _head_coordinates(head_center[chain], head_arm[chain], head_slope[chain], motion)
        head_motion[chain] = motion
        for i in range(3):
            q[6 * node + i] = position[i]
            q[6 * node + 3 + i] = tangent[i]
    else:
        for i in range(6):
            q[6 * node + i] = q[6 * node + i] + scales[best] * update[node][i]


@wp.kernel(enable_backward=False)
def begin_step(
    nodes: int,
    rigid: bool,
    head_center: wp.array[wp.vec3d],
    head_arm: wp.array[wp.vec3d],
    head_slope: wp.array[wp.vec3d],
    head_motion: wp.array[Vec6d],
    status: wp.array[int],
    iterations: wp.array[int],
    active: wp.array[int],
    passes: wp.array[int],
    q: wp.array[wp.float64],
):
    """Mark every chain as iterating and place its head at the first iterate; launch over chains.

    A rigid head starts from the motion it made in the previous step: a body
    that the chain holds still needs the same motion again.
    """
    chain = wp.tid()
    status[chain] = 0
    iterations[chain] = 0
    active[0] = 1
    passes[0] = 0
    if rigid:
        position, tangent, _ = _head_coordinates(
            head_center[chain], head_arm[chain], head_slope[chain], head_motion[chain]
        )
        for i in range(3):
            q[6 * chain * nodes + i] = position[i]
            q[6 * chain * nodes + 3 + i] = tangent[i]


@wp.kernel(enable_backward=False)
def begin_pass(active: wp.array[int], passes: wp.array[int]):
    active[0] = 0
    passes[0] = passes[0] + 1


@wp.kernel(enable_backward=False)
def check(
    nodes: int,
    rigid: bool,
    residual: wp.array[Vec6d],
    slope_scale: wp.float64,
    tolerance: wp.float64,
    max_iterations: int,
    passes: wp.array[int],
    status: wp.array[int],
    iterations: wp.array[int],
    active: wp.array[int],
):
    """Decide whether a chain needs another Newton update; one thread per chain.

    The norm is the largest free residual, slope rows divided by the element
    length [N]. Status 0 iterates; 1 is finished.
    """
    chain = wp.tid()
    if status[chain] != 0:
        return
    norm = wp.float64(0.0)
    start = chain * nodes + 1
    if rigid:
        # The head's rows are a force and a torque; the torque is scaled like the slope rows.
        start = chain * nodes
    for node in range(start, (chain + 1) * nodes):
        for i in range(3):
            norm = wp.max(norm, wp.abs(residual[node][i]))
            norm = wp.max(norm, wp.abs(residual[node][i + 3]) * slope_scale)
    # NaN compares false, so a non-finite residual stops the chain instead of iterating on it.
    if norm > tolerance and passes[0] <= max_iterations:
        iterations[chain] = iterations[chain] + 1
        wp.atomic_max(active, 0, 1)
    else:
        status[chain] = 1


@wp.kernel(enable_backward=False)
def finish(q: wp.array[wp.float64], predicted_q: wp.array[wp.float64], dt: wp.float64, qd: wp.array[wp.float64]):
    """Accept the step: v = v_n + dt a with a = (q - q~) / dt²; launch over all coordinates."""
    i = wp.tid()
    qd[i] = qd[i] + dt * ((q[i] - predicted_q[i]) / (dt * dt))


# Eyelets: points in space that the thread must pass through but may slide along. Each eyelet tracks
# the point of the chain that is nearest to it, as a global element index and a coordinate xi.


@wp.func
def _eyelet_offset(
    q: wp.array[wp.float64], node_a: int, xi: wp.float64, length: wp.float64, target: wp.vec3d, tangent: wp.vec3d
) -> wp.vec3d:
    """From the eyelet to the chain's tracked point, without the part along the thread."""
    offset = _centerline_point(q, node_a, _shape(xi, length)) - target
    return offset - wp.dot(offset, tangent) * tangent


@wp.func
def _eyelet_drift(
    q: wp.array[wp.float64],
    node_a: int,
    xi: wp.float64,
    length: wp.float64,
    target: wp.vec3d,
    tangent: wp.vec3d,
) -> wp.float64:
    """How far the chain's anchored point has moved past the eyelet along the thread."""
    return wp.dot(tangent, _centerline_point(q, node_a, _shape(xi, length)) - target)


@wp.kernel(enable_backward=False)
def slide_eyelets(
    q: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    nodes: int,
    target: wp.array[wp.vec3d],
    held: wp.array[int],
    stiffness: wp.float64,
    limit: wp.array[wp.float64],
    element: wp.array[int],
    xi: wp.array[wp.float64],
    tangent: wp.array[wp.vec3d],
    anchor_element: wp.array[int],
    anchor_xi: wp.array[wp.float64],
):
    """Move each eyelet's tracked point to the chain point nearest to it; launch over eyelets.

    The search covers the tracked element and its two neighbours, so the
    thread may slide through an eyelet by up to one element per step. The
    unit tangent at the tracked point is stored for the step.

    A held eyelet keeps its point of the chain and stores a zero tangent:
    its spring then acts in every direction, a knot at that point.

    Friction: the eyelet also keeps an anchored point of the chain. It stays
    while the thread sticks and otherwise trails the tracked point by the
    reach of the eyelet's Coulomb limit, measured along the chain.
    """
    eyelet = wp.tid()
    index = element[eyelet]
    if index < 0:
        return
    if held[eyelet] != 0:
        tangent[eyelet] = wp.vec3d()
        return
    per_chain = lengths.shape[0]
    chain = index // per_chain
    local = index % per_chain
    point = target[eyelet]
    samples = int(8)
    best_local = local
    best_xi = xi[eyelet]
    best = wp.float64(1.0e300)
    for candidate in range(wp.max(local - 1, 0), wp.min(local + 2, per_chain)):
        node_a = chain * nodes + candidate
        for sample in range(samples + 1):
            coordinate = wp.float64(sample) / wp.float64(samples)
            offset = _centerline_point(q, node_a, _shape(coordinate, lengths[candidate])) - point
            distance = wp.dot(offset, offset)
            if distance < best:
                best = distance
                best_local = candidate
                best_xi = coordinate
    # Newton's method on the squared distance along the chain, from the best sample. An iterate that leaves
    # its element goes on in the neighbouring one, so that a node can pass through the eyelet.
    node_a = chain * nodes + best_local
    length = lengths[best_local]
    direction = wp.vec3d()
    for _iteration in range(4):
        first, second = _shape_derivatives(best_xi, length)
        direction, curvature = _centerline_derivatives(q, node_a, node_a + 1, first, second)
        offset = _centerline_point(q, node_a, _shape(best_xi, length)) - point
        slope = wp.dot(offset, direction)
        bend = wp.dot(direction, direction) + wp.dot(offset, curvature)
        if bend > wp.float64(0.0):
            moved = best_xi - slope / (bend * length)
            if moved > wp.float64(1.0) and best_local < per_chain - 1:
                moved = (moved - wp.float64(1.0)) * length / lengths[best_local + 1]
                best_local += 1
            elif moved < wp.float64(0.0) and best_local > 0:
                moved = wp.float64(1.0) + moved * length / lengths[best_local - 1]
                best_local -= 1
            node_a = chain * nodes + best_local
            length = lengths[best_local]
            best_xi = wp.clamp(moved, wp.float64(0.0), wp.float64(1.0))
    first, second = _shape_derivatives(best_xi, length)
    direction, curvature = _centerline_derivatives(q, node_a, node_a + 1, first, second)
    element[eyelet] = chain * per_chain + best_local
    xi[eyelet] = best_xi
    tangent[eyelet] = wp.normalize(direction)
    tracked = wp.float64(best_local) + best_xi
    anchored = tracked
    if anchor_element[eyelet] >= 0:
        reach = limit[eyelet] / (stiffness * length)
        anchored = wp.float64(anchor_element[eyelet] % per_chain) + anchor_xi[eyelet]
        anchored = wp.clamp(anchored, tracked - reach, tracked + reach)
    anchored = wp.clamp(anchored, wp.float64(0.0), wp.float64(per_chain))
    cell = wp.min(int(anchored), per_chain - 1)
    anchor_element[eyelet] = chain * per_chain + cell
    anchor_xi[eyelet] = anchored - wp.float64(cell)


@wp.kernel(enable_backward=False)
def add_eyelets(
    q: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    nodes: int,
    target: wp.array[wp.vec3d],
    element: wp.array[int],
    xi: wp.array[wp.float64],
    tangent: wp.array[wp.vec3d],
    stiffness: wp.float64,
    tangent_blocks: bool,
    status: wp.array[int],
    limit: wp.array[wp.float64],
    anchor_element: wp.array[int],
    anchor_xi: wp.array[wp.float64],
    gradient: wp.array2d[wp.vec3d],
    hessian: wp.array3d[wp.mat33d],
):
    """Add each eyelet's spring to its element's gradient and tangent blocks; launch over eyelets.

    The spring k acts between the eyelet and the chain's tracked point, across
    the thread only: its energy is k |P d|² / 2 with d the offset and
    P = I - t t^T for the unit tangent t stored by :func:`slide_eyelets`.

    Along the thread the eyelet has friction: a spring k between the eyelet
    and the chain's anchored point whose force stops growing at the eyelet's
    Coulomb limit. With a zero limit the thread slides through freely.
    """
    eyelet = wp.tid()
    index = element[eyelet]
    if index < 0:
        return
    per_chain = lengths.shape[0]
    chain = index // per_chain
    if status[chain] != 0:
        return
    local = index % per_chain
    shape = _shape(xi[eyelet], lengths[local])
    direction = tangent[eyelet]
    force = stiffness * _eyelet_offset(q, chain * nodes + local, xi[eyelet], lengths[local], target[eyelet], direction)
    projector = wp.identity(n=3, dtype=wp.float64) - wp.outer(direction, direction)
    for b in range(4):
        wp.atomic_add(gradient, index, b, shape[b] * force)
        if tangent_blocks:
            for c in range(4):
                wp.atomic_add(hessian, index, b, c, (stiffness * shape[b] * shape[c]) * projector)
    grip = limit[eyelet]
    anchored = anchor_element[eyelet]
    if grip > wp.float64(0.0) and anchored >= 0 and wp.dot(direction, direction) > wp.float64(0.0):
        anchored_local = anchored % per_chain
        anchored_shape = _shape(anchor_xi[eyelet], lengths[anchored_local])
        drift = _eyelet_drift(
            q, chain * nodes + anchored_local, anchor_xi[eyelet], lengths[anchored_local], target[eyelet], direction
        )
        scale = _friction_scale(wp.abs(drift), stiffness, grip)
        for b in range(4):
            wp.atomic_add(gradient, anchored, b, (anchored_shape[b] * scale * drift) * direction)
            if tangent_blocks:
                for c in range(4):
                    wp.atomic_add(
                        hessian,
                        anchored,
                        b,
                        c,
                        (scale * anchored_shape[b] * anchored_shape[c]) * wp.outer(direction, direction),
                    )


@wp.kernel(enable_backward=False)
def trial_eyelet_energy(
    trial_q: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    nodes: int,
    total_nodes: int,
    target: wp.array[wp.vec3d],
    element: wp.array[int],
    xi: wp.array[wp.float64],
    tangent: wp.array[wp.vec3d],
    stiffness: wp.float64,
    status: wp.array[int],
    limit: wp.array[wp.float64],
    anchor_element: wp.array[int],
    anchor_xi: wp.array[wp.float64],
    energy: wp.array2d[wp.float64],
):
    """Spring and friction energy of one eyelet at one candidate; launch (candidates, eyelets)."""
    candidate, eyelet = wp.tid()
    index = element[eyelet]
    if index < 0:
        return
    per_chain = lengths.shape[0]
    chain = index // per_chain
    if status[chain] != 0:
        return
    local = index % per_chain
    offset = _eyelet_offset(
        trial_q,
        candidate * total_nodes + chain * nodes + local,
        xi[eyelet],
        lengths[local],
        target[eyelet],
        tangent[eyelet],
    )
    value = wp.float64(0.5) * stiffness * wp.dot(offset, offset)
    grip = limit[eyelet]
    anchored = anchor_element[eyelet]
    if grip > wp.float64(0.0) and anchored >= 0 and wp.dot(tangent[eyelet], tangent[eyelet]) > wp.float64(0.0):
        anchored_local = anchored % per_chain
        drift = _eyelet_drift(
            trial_q,
            candidate * total_nodes + chain * nodes + anchored_local,
            anchor_xi[eyelet],
            lengths[anchored_local],
            target[eyelet],
            tangent[eyelet],
        )
        value += _friction_energy(wp.abs(drift), stiffness, grip)
    wp.atomic_add(energy, chain, candidate, value)


@wp.kernel(enable_backward=False)
def eyelet_forces(
    q: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    nodes: int,
    target: wp.array[wp.vec3d],
    element: wp.array[int],
    xi: wp.array[wp.float64],
    tangent: wp.array[wp.vec3d],
    stiffness: wp.float64,
    friction: wp.float64,
    anchor_element: wp.array[int],
    anchor_xi: wp.array[wp.float64],
    limit: wp.array[wp.float64],
    force: wp.array[wp.vec3d],
    point: wp.array[wp.vec3d],
):
    """Force of the thread on each eyelet [N] and the thread's point there [m] at the accepted coordinates.

    Launch over eyelets. An unthreaded eyelet has no force and reports its
    own position. Also sets the eyelet's Coulomb limit for the next step: the
    friction coefficient times the force across the thread now.
    """
    eyelet = wp.tid()
    index = element[eyelet]
    force[eyelet] = wp.vec3d()
    point[eyelet] = target[eyelet]
    if index < 0:
        limit[eyelet] = wp.float64(0.0)
        return
    per_chain = lengths.shape[0]
    local = index % per_chain
    chain = index // per_chain
    node_a = chain * nodes + local
    direction = tangent[eyelet]
    across = stiffness * _eyelet_offset(q, node_a, xi[eyelet], lengths[local], target[eyelet], direction)
    result = across
    grip = limit[eyelet]
    anchored = anchor_element[eyelet]
    if grip > wp.float64(0.0) and anchored >= 0 and wp.dot(direction, direction) > wp.float64(0.0):
        anchored_local = anchored % per_chain
        drift = _eyelet_drift(
            q, chain * nodes + anchored_local, anchor_xi[eyelet], lengths[anchored_local], target[eyelet], direction
        )
        result += (_friction_scale(wp.abs(drift), stiffness, grip) * drift) * direction
    force[eyelet] = result
    point[eyelet] = _centerline_point(q, node_a, _shape(xi[eyelet], lengths[local]))
    limit[eyelet] = wp.float64(0.0)
    if wp.dot(direction, direction) > wp.float64(0.0):
        limit[eyelet] = friction * wp.length(across)


@wp.func
def closest_on_segments(p1: wp.vec3d, q1: wp.vec3d, p2: wp.vec3d, q2: wp.vec3d) -> wp.vec2d:
    """Where two segments of nonzero length are closest: the fractions of the way along the first and the second."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    b = wp.dot(d1, d2)
    c = wp.dot(d1, r)
    f = wp.dot(d2, r)
    s = wp.float64(0.0)
    denominator = a * e - b * b
    # Parallel segments keep s = 0: any pair of closest points serves.
    if denominator > wp.float64(1.0e-12) * a * e:
        s = wp.clamp((b * f - c * e) / denominator, wp.float64(0.0), wp.float64(1.0))
    t = (b * s + f) / e
    if t < wp.float64(0.0):
        t = wp.float64(0.0)
        s = wp.clamp(-c / a, wp.float64(0.0), wp.float64(1.0))
    elif t > wp.float64(1.0):
        t = wp.float64(1.0)
        s = wp.clamp((b - c) / a, wp.float64(0.0), wp.float64(1.0))
    return wp.vec2d(s, t)


@wp.kernel(enable_backward=False)
def find_self_contacts(
    q: wp.array[wp.float64],
    nodes: int,
    reach: wp.float64,
    skip: int,
    xi: wp.array[wp.float64],
    point: wp.array[wp.vec3d],
    normal: wp.array[wp.vec3d],
):
    """Give every element the plane between it and the nearest other stretch of its chain; launch over elements.

    Elements are taken as their chords. The plane lies midway between the
    two closest points, so each of the two elements keeps half the distance
    and neither has to know how the other moves in the step. Elements no
    more than ``skip`` apart along the chain cannot touch. An element with
    nothing within ``reach`` gets a zero normal.
    """
    element = wp.tid()
    per_chain = nodes - 1
    chain = element // per_chain
    local = element % per_chain
    first = chain * nodes + local
    a0 = _node_vector(q, first, 0)
    a1 = _node_vector(q, first + 1, 0)
    nearest = reach
    normal[element] = wp.vec3d()
    for other in range(per_chain):
        if wp.abs(other - local) <= skip:
            continue
        b0 = _node_vector(q, chain * nodes + other, 0)
        b1 = _node_vector(q, chain * nodes + other + 1, 0)
        fractions = closest_on_segments(a0, a1, b0, b1)
        here = a0 + fractions[0] * (a1 - a0)
        there = b0 + fractions[1] * (b1 - b0)
        distance = wp.length(here - there)
        if distance < nearest and distance > wp.float64(0.0):
            nearest = distance
            xi[element] = fractions[0]
            point[element] = wp.float64(0.5) * (here + there)
            normal[element] = (here - there) / distance


@wp.kernel(enable_backward=False)
def add_obstacles(
    q: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    nodes: int,
    xi: wp.array[wp.float64],
    point: wp.array[wp.vec3d],
    normal: wp.array[wp.vec3d],
    distance: wp.float64,
    stiffness: wp.float64,
    smoothing: wp.float64,
    tangent_blocks: bool,
    status: wp.array[int],
    gradient: wp.array2d[wp.vec3d],
    hessian: wp.array3d[wp.mat33d],
):
    """Add each element's obstacle plane to its gradient and tangent blocks; launch over elements.

    The element's point at ``xi`` is kept ``distance`` above the plane by
    the contact penalty of :func:`contact_terms`. A zero normal is no plane.
    """
    element = wp.tid()
    direction = normal[element]
    if wp.dot(direction, direction) == wp.float64(0.0):
        return
    per_chain = lengths.shape[0]
    chain = element // per_chain
    if status[chain] != 0:
        return
    local = element % per_chain
    shape = _shape(xi[element], lengths[local])
    position = _centerline_point(q, chain * nodes + local, shape)
    terms = contact_terms(distance, wp.dot(direction, position - point[element]), stiffness, smoothing)
    for b in range(4):
        wp.atomic_add(gradient, element, b, (shape[b] * terms[1]) * direction)
        if tangent_blocks:
            for c in range(4):
                wp.atomic_add(hessian, element, b, c, (terms[2] * shape[b] * shape[c]) * wp.outer(direction, direction))


@wp.kernel(enable_backward=False)
def trial_obstacle_energy(
    trial_q: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    nodes: int,
    total_nodes: int,
    xi: wp.array[wp.float64],
    point: wp.array[wp.vec3d],
    normal: wp.array[wp.vec3d],
    distance: wp.float64,
    stiffness: wp.float64,
    smoothing: wp.float64,
    status: wp.array[int],
    energy: wp.array2d[wp.float64],
):
    """Contact energy of one element's obstacle plane at one candidate; launch (candidates, elements)."""
    candidate, element = wp.tid()
    direction = normal[element]
    if wp.dot(direction, direction) == wp.float64(0.0):
        return
    per_chain = lengths.shape[0]
    chain = element // per_chain
    if status[chain] != 0:
        return
    local = element % per_chain
    position = _centerline_point(
        trial_q, candidate * total_nodes + chain * nodes + local, _shape(xi[element], lengths[local])
    )
    height = wp.dot(direction, position - point[element])
    wp.atomic_add(energy, chain, candidate, contact_terms(distance, height, stiffness, smoothing)[0])


@wp.kernel(enable_backward=False)
def obstacle_forces(
    q: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    nodes: int,
    xi: wp.array[wp.float64],
    point: wp.array[wp.vec3d],
    normal: wp.array[wp.vec3d],
    distance: wp.float64,
    stiffness: wp.float64,
    smoothing: wp.float64,
    force: wp.array[wp.vec3d],
):
    """Force of its obstacle plane on every element at the accepted coordinates [N]; launch over elements."""
    element = wp.tid()
    force[element] = wp.vec3d()
    direction = normal[element]
    if wp.dot(direction, direction) == wp.float64(0.0):
        return
    per_chain = lengths.shape[0]
    local = element % per_chain
    position = _centerline_point(q, (element // per_chain) * nodes + local, _shape(xi[element], lengths[local]))
    height = wp.dot(direction, position - point[element])
    force[element] = -contact_terms(distance, height, stiffness, smoothing)[1] * direction


@wp.kernel(enable_backward=False)
def contact_forces(
    q: wp.array[wp.float64],
    nodes: int,
    contact_distance: wp.float64,
    contact_stiffness: wp.float64,
    contact_smoothing: wp.float64,
    contact_point: wp.array[wp.vec3d],
    contact_normal: wp.array[wp.vec3d],
    friction: wp.float64,
    limit: wp.array[wp.float64],
    anchor: wp.array[wp.vec3d],
    force: wp.array[wp.vec3d],
):
    """Force of its contact plane on every node at the accepted coordinates [N]; launch over all nodes.

    Also prepares the friction of the next step. A node's Coulomb limit is
    the friction coefficient times its normal force now. Its anchor stays
    while the node sticks, follows the node at the limit's reach while it
    slides, and is the node's own position while it has no grip.
    """
    node = wp.tid()
    force[node] = wp.vec3d()
    normal = contact_normal[node]
    position = _node_vector(q, node, 0)
    if node % nodes == 0 or wp.dot(normal, normal) == wp.float64(0.0):
        limit[node] = wp.float64(0.0)
        anchor[node] = position
        return
    height = wp.dot(normal, position - contact_point[node])
    push = -contact_terms(contact_distance, height, contact_stiffness, contact_smoothing)[1]
    result = push * normal
    held = limit[node]
    if held > wp.float64(0.0):
        offset = _plane_offset(position, anchor[node], normal)
        slip = wp.length(offset)
        result -= _friction_scale(slip, contact_stiffness, held) * offset
        if contact_stiffness * slip > held:
            anchor[node] = position - (held / (contact_stiffness * slip)) * offset
    else:
        anchor[node] = position
    limit[node] = friction * push
    force[node] = result
