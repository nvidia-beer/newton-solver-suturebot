# SPDX-License-Identifier: Apache-2.0
"""Original source arithmetic with diagnostic branch outputs; no physical use."""

import warp as wp

from newton_solver_suturebot._src.solvers.coupled.interface.tunnel_alm import _normal_projection

vec6i = wp.types.vector(length=6, dtype=wp.int32)


@wp.func
def traced_projection(
    point: wp.vec3,
    pose: wp.transform,
    samples: wp.array[wp.vec3],
    tangent_radius: float,
):
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
    lower_index, upper_index = int(-1), int(-1)
    for j in range(samples.shape[0] - 1):
        start_point, end_point = samples[j], samples[j + 1]
        segment_length = wp.length(end_point - start_point)
        if arc_length <= before and before <= arc_length + segment_length:
            lower_index = j
            lower = start_point + (before - arc_length) / segment_length * (end_point - start_point)
        if arc_length <= after and after <= arc_length + segment_length:
            upper_index = j
            upper = start_point + (after - arc_length) / segment_length * (end_point - start_point)
        arc_length += segment_length
    if wp.length_sq(upper - lower) > 1.0e-20:
        tangent = wp.normalize(upper - lower)
    covered = not (
        (segment_best == 0 and fraction_best < 0.0) or (segment_best == samples.shape[0] - 2 and fraction_best > 1.0)
    )
    branch = vec6i(
        segment_best,
        int(fraction_best < 0.0) - int(fraction_best > 1.0),
        lower_index,
        upper_index,
        int(wp.length_sq(upper - lower) > 1.0e-20),
        int(covered),
    )
    return (
        wp.transform_point(pose, closest),
        wp.quat_rotate(wp.transform_get_rotation(pose), tangent),
        best,
        covered,
        branch,
    )


@wp.kernel(enable_backward=False)
def traced(
    queries: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    radii: wp.array[float],
    samples: wp.array[wp.vec3],
    result: wp.array2d[float],
    branches: wp.array2d[int],
):
    i = wp.tid()
    closest, tangent, distance, covered, branch = traced_projection(queries[i], poses[i], samples, radii[i])
    for j in range(3):
        result[i, j] = closest[j]
        result[i, j + 3] = tangent[j]
    result[i, 6] = distance
    result[i, 7] = float(covered)
    displacement = closest - queries[i]
    displacement -= wp.dot(displacement, tangent) * tangent
    reaction = _normal_projection(displacement, tangent, wp.vec3(0.1, -0.2, 0.3), 1000.0, 0.001, 0.00002)
    for j in range(3):
        result[i, j + 8] = reaction[j]
    for j in range(6):
        branches[i, j] = branch[j]
