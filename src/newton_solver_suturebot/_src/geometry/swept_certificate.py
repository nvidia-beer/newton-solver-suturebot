# SPDX-License-Identifier: Apache-2.0
"""One-sided certificates that rule out swept point/triangle intersections.

A false result leaves the exact CPU moving-triangle oracle responsible for all
roots, tangencies, barycentric containment and event publication.
"""

import warp as wp


@wp.func
def _l1(v: wp.vec3d):
    return wp.abs(v[0]) + wp.abs(v[1]) + wp.abs(v[2])


@wp.func
def _finite(v: wp.vec3d):
    return wp.isfinite(v[0]) and wp.isfinite(v[1]) and wp.isfinite(v[2])


@wp.func
def swept_triangle_clear(
    start: wp.vec3d,
    end: wp.vec3d,
    a: wp.vec3d,
    b: wp.vec3d,
    c: wp.vec3d,
    aa: wp.vec3d,
    bb: wp.vec3d,
    cc: wp.vec3d,
    point_error: wp.float64,
):
    finite = (
        _finite(start)
        and _finite(end)
        and _finite(a)
        and _finite(b)
        and _finite(c)
        and _finite(aa)
        and _finite(bb)
        and _finite(cc)
        and wp.isfinite(point_error)
    )
    old_normal, new_normal = wp.cross(b - a, c - a), wp.cross(bb - aa, cc - aa)
    if (
        not finite
        or point_error < wp.float64(0.0)
        or wp.dot(old_normal, old_normal) <= wp.float64(1e-28)
        or wp.dot(new_normal, new_normal) <= wp.float64(1e-28)
    ):
        return False
    # The oracle first intersects the point's segment AABB with the exact
    # union of old/new triangle bounds, padded by1nm. Vertex motion is linear,
    # so the union encloses every intermediate triangle.
    padding = wp.vec3d(wp.float64(1e-9) + point_error)
    low, high = wp.min(start, end) - padding, wp.max(start, end) + padding
    triangle_low = wp.min(wp.min(a, b), wp.min(c, wp.min(wp.min(aa, bb), cc)))
    triangle_high = wp.max(wp.max(a, b), wp.max(c, wp.max(wp.max(aa, bb), cc)))
    separated = bool(False)
    for axis in range(3):
        separated = separated or triangle_high[axis] < low[axis] or triangle_low[axis] > high[axis]
    if separated:
        return True
    # Coplanarity is trilinear in r=point-a, e=b-a and f=c-a, hence cubic
    # in time. The oracle admits roots through1+1e-7; certify that extended
    # interval too. Bernstein controls bound the entire polynomial by their
    # convex hull, without solving roots or relying on endpoint signs alone.
    t = wp.float64(1.0 + 1e-7)
    r0, e0, f0 = start - a, b - a, c - a
    r1 = r0 + t * ((end - start) - (aa - a))
    e1 = e0 + t * ((bb - aa) - e0)
    f1 = f0 + t * ((cc - aa) - f0)
    p0 = wp.dot(r0, wp.cross(e0, f0))
    p1 = (wp.dot(r1, wp.cross(e0, f0)) + wp.dot(r0, wp.cross(e1, f0)) + wp.dot(r0, wp.cross(e0, f1))) / wp.float64(3.0)
    p2 = (wp.dot(r0, wp.cross(e1, f1)) + wp.dot(r1, wp.cross(e0, f1)) + wp.dot(r1, wp.cross(e1, f0))) / wp.float64(3.0)
    p3 = wp.dot(r1, wp.cross(e1, f1))
    edge_bound = wp.max(_l1(e0), _l1(e1)) * wp.max(_l1(f0), _l1(f1))
    magnitude = wp.max(_l1(r0), _l1(r1)) * edge_bound
    # With R,E,F the endpoint L1 bounds, the power-coefficient magnitudes
    # are bounded by (1,6,12,8)*R*E*F. Discarding up to three leading
    # coefficients at 1e-12 of their maximum changes the polynomial on
    # [0,1+1e-7] by <3.61e-11*R*E*F. For a conjugate root pair whose
    # imaginary part is <=1e-7, its real-part residual is bounded by
    # <3.61e-13*R*E*F. A 1e-8 margin additionally leaves over two orders
    # of headroom for FP64 evaluation/root-roundoff around those conventions.
    # The second term encloses the caller's float32 tip-transform uncertainty.
    # Cancellation, tangency, degeneracy and nearzero controls remain ambiguous.
    margin = wp.float64(1e-8) * magnitude + wp.float64(2.0) * point_error * edge_bound + wp.float64(1e-30)
    minimum = wp.min(wp.min(p0, p1), wp.min(p2, p3))
    maximum = wp.max(wp.max(p0, p1), wp.max(p2, p3))
    finite = wp.isfinite(p0) and wp.isfinite(p1) and wp.isfinite(p2) and wp.isfinite(p3) and wp.isfinite(margin)
    return finite and (minimum > margin or maximum < -margin)


@wp.func
def tip_roundoff_bound(pose: wp.transform, local: wp.vec3d):
    # Bound the difference between the host's float32 transform_point and
    # the force graph's float32 rotation matrix times float64 local sample.
    # Translation and rotation magnitudes retain a bound even when world-space
    # cancellation makes the final point small. The allowed 10nm difference
    # between tip_local and the first sample applies to each component; its
    # L1 norm can be 30nm and rotation can mix all three components.
    q = wp.transform_get_rotation(pose)
    q64 = wp.vec4d(wp.float64(q[0]), wp.float64(q[1]), wp.float64(q[2]), wp.float64(q[3]))
    norm2 = wp.dot(q64, q64)
    # Native poses are unit quaternions. Grossly nonunit finite inputs are not
    # eligible for this fast certificate; retain the original host oracle.
    if not wp.isfinite(norm2) or wp.abs(norm2 - wp.float64(1.0)) > wp.float64(1e-4):
        return wp.float64(wp.inf)
    qnorm = wp.abs(wp.float64(q[0])) + wp.abs(wp.float64(q[1])) + wp.abs(wp.float64(q[2])) + wp.abs(wp.float64(q[3]))
    position = wp.vec3d(wp.transform_get_translation(pose))
    rotation_bound = wp.float64(1.0) + wp.float64(2.0) * qnorm * qnorm
    scale = _l1(position) + rotation_bound * _l1(local)
    return wp.float64(64.0 * 1.1920928955078125e-7) * scale + wp.float64(3e-8) * rotation_bound
