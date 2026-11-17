# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact between rigid tool capsules and a tissue surface on the sweep of the capsules.

After every accepted substep each capsule is swept to the pose it will have at the end of
the next one, and the tissue is placed there before the tool arrives:

1. **Sweep.** ``P^ = P + h_full * v`` with the accepted pose and velocity of the body and
   the full substep ``h_full`` of the scene (also when the next attempt is shorter). A
   capsule is its axis and its radius; the axis is cut into pieces of at most one radius.
   Each piece has a probe, a point of the axis: the sign of the surface at both ends of the
   probe's path (ray parity, ``wp.mesh_query_point``) and a ray along the path say whether
   the axis ends inside the tissue or went through the surface (the probe is *inward*).
2. **Rows.** Every surface triangle is split into the three parts that belong to its
   vertices (vertex, two edge midpoints, centroid). A row is one axis piece at its predicted
   pose against one part: the exact closest pair, depth ``R - distance`` (negative: a
   clearance) and the push direction from the axis to the surface. Under an inward probe
   the depth is taken along the probe's outward direction against the capsule over its
   sweep, so that the surface is pushed ahead of the capsule and not aside.
3. **Tissue that can follow** is moved, not loaded. A free surface node takes the deepest
   row of its own parts. It is moved along the push direction by that depth, at most by its
   *allowed move*, its velocity is set to zero and it is held as a point through the next
   substep by a block on the diagonal of the tissue matrix (``fem/node_holds.py``). A node
   that was held, still pushes against the tool and still has a row follows the tool back.
   Where the piece that owns the row is being stopped (5.), the node is placed on the
   capsule pulled back by the held depth of that piece: the tissue beside is not pushed
   further than the stopping tissue lets the capsule be.
   The elastic force on a held node along its push direction, read at the next accepted
   state, is the reaction on the tool, one substep late; only nodes that push back give one
   (a contact cannot pull), and a node that pulls is released.
4. **Allowed move.** With ``F`` the deformation gradient of a tetrahedron, a move ``d`` of
   its node ``m`` changes ``F`` by ``d (x) grad N_m`` with ``|grad N_m| = 1 / h_m`` (``h_m``
   the rest altitude at that node), so by Weyl's inequality the smallest principal stretch
   changes by at most ``|d| / h_m``. A node may move by

       min over its tetrahedra of  MOVE_FRACTION * h_m * (sigma_min - min_stretch)

   per substep into the tissue, not below zero: one eighth of what the element has left
   above the stretch limit of the material law, so that four moving nodes together use at
   most half of it. For an element that is squeezed along that altitude alone this is one
   eighth of what the altitude has left above ``min_stretch`` of its rest value. A node
   that follows the tool back may move by ``MOVE_FRACTION * h_m * sigma_min``, one eighth of
   what the element has left above zero stretch. So no placement can invert an element.
   Fixed nodes and surface vertices outside the simulated region have no allowed move.
5. **Tissue that is held or cannot follow stops the tool.** The *held depth* of an axis
   piece is the largest (row depth - allowed move of the row's vertex) over its rows,
   negative with a clearance. Each piece stores one force ``I`` [N], a vector. With ``u``
   the direction of its deepest held row (from the surface to the axis), ``n`` the number
   of touching pieces of its tool (held depth positive or force stored) and ``pen`` the
   held depth:

       pen >= 0:   I <- I + (r rho pen / n + L pen / P) u    f = I + (K_c pen / n) u
       pen <  0:  |I| <- max(0, |I| + r K_c pen / n)        |f| = max(0, |I| + K_c min(0, gap) / n), along I

   ``gap`` is the deepest row of the piece itself, negative only where the capsule is
   clear of the surface; on held tissue it is ``pen``. ``L`` is what the pieces of the
   same tool with ``pen < 0`` lose of ``|I|`` in this substep [N] and ``P`` the sum of the
   held depths of its pieces with ``pen >= 0`` [m].
   The stored force is a stand-off: ``f = (K_c / n) (s + pen)`` with ``s = |I| n / K_c``
   [m], the distance by which the piece is kept off the tissue under it. While that tissue
   is limited the stand-off grows by ``rho / K_c`` of the held depth per substep. Once no
   row of the piece is limited (``pen < 0``) its tissue can follow by the slack ``-pen`` in
   the next substep (or the piece is clear by it), and the stand-off closes by that slack:
   the stored force runs out at the proportional gain, not at ``rho``. What the pieces of
   a tool lose in this way (``L``) is handed to those pieces of the same tool that have a
   held depth, in proportion to their held depths and along their directions: the tool
   is still being stopped, now there. Only when no piece of the tool has a held depth is
   it lost. So the stored force does not outlive its cause: a tool is not carried over
   tissue that can yield, and not pushed when nothing touches it. On tissue that follows
   again (``pen < 0 <= gap``) the applied force is the stored one, without a jump when the
   slack changes.
   ``f`` acts on the body at the probe. ``I`` is never reset: it runs out through the
   law. Every piece takes every part in its search box, so a contact that
   slides from one piece to the next changes both held depths continuously; ``n`` keeps
   the stiffness of all touching pieces together at ``K_c``.
   ``K_c = m_eff(u) / h_full^2``: half of the bound ``2 m / h^2`` of a spring that is read
   at the predicted pose on a free mass, with ``m_eff(u) = 1 / (u . W u)`` the effective
   mass of the tool at the probe along the push direction and ``W = J M^-1 J^T`` its
   inverse mass matrix there with every actuator free, given per piece in the frame of
   its body. ``rho = K_c / 16`` (``INTEGRAL_RATIO``). ``W`` and the ratio are constants of
   the scene for the full substep and do not follow a refined step. ``r`` is the accepted
   step over the full step: the stored force is an integral over time, so a refined step
   does not raise its gain per second.

``max(0, .)`` is the contact condition, the allowed move is a bound from the geometry of an
element; no force has an upper limit. Storage is fixed: every array is sized by samples,
bodies or particles. Nothing here reads the device; :meth:`diagnostics` is an explicit read.
"""

import math

import numpy as np
import warp as wp

from ...fem.kernels import eval_tetrahedra
from ...fem.node_holds import NodeHolds
from .tool_tissue_contact import _add, capsule_samples

# Share of an element's remaining stretch margin that one node may use in one substep: four
# moving nodes of a tetrahedron together use at most half of it.
MOVE_FRACTION = 0.125
# Integral gain over proportional gain of the held-tissue force, per full substep: 1/16, as the
# design has it. Two bounds, both above it:
# - Stability. With K_c fixed at its full-step value, the sweep horizon at the full step and
#   the stored force integrated over time, the chain model of one dVRK tool (no actuator at
#   its force limit) is stable up to rho / K_c = 0.544 at 1/240 s, 0.73 at 1/480 s, 0.90 at
#   1/960 s and 1.10 at 1/3840 s. (With an actuator at its force limit no gain is stable.)
# - Smoothness. While a piece is pressed in, its stored force grows by rho / K_c of the
#   proportional part of its force per substep, on top of the growth of that part itself;
#   the contact force may not change by more than a quarter per substep above 50 mN. On the
#   stiffest tool model of the design (576 N/m at the jaw tip, driven at 3 mm/s) the
#   proportional part grows by 7 mN per substep at 50 mN and the held depth is 37 um there:
#   rho below 149 N/m = K_c / 8.5.
# A larger ratio shortens what a driven tool stands inside held tissue (speed * step *
# tool stiffness / rho) and the time in which that closes at rest.
INTEGRAL_RATIO = 0.0625
# The lower envelope of a capsule along a probe direction is sampled on the 15 points of a
# part (barycentric fourths) and at axis stations and sweep poses a quarter radius apart.
LATTICE = 4
STATION_RADII = 0.25


@wp.struct
class _Closest:
    """Closest pair between a segment and a triangle."""

    s: float
    point: wp.vec3
    distance: float
    crossing: int


@wp.struct
class _Row:
    """One axis piece against one triangle part: depth [m] and unit push direction (from the axis to the surface)."""

    depth: float
    push: wp.vec3


@wp.func
def _closest_point_triangle(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    """Closest point of the triangle (a, b, c) to p (Ericson, Real-Time Collision Detection, 5.1.5)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denominator = 1.0 / (va + vb + vc)
    return a + ab * (vb * denominator) + ac * (vc * denominator)


@wp.func
def _segment_segment(p1: wp.vec3, q1: wp.vec3, p2: wp.vec3, q2: wp.vec3):
    """Parameters (s on p1-q1, t on p2-q2) of the closest pair of two segments (Ericson, 5.1.9)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)
    s = float(0.0)
    t = float(0.0)
    # Squared lengths below this are points: 1e-12 m is far below any tool or mesh length.
    tiny = 1.0e-24
    if a <= tiny and e <= tiny:
        return wp.vec2(0.0, 0.0)
    if a <= tiny:
        t = wp.clamp(f / e, 0.0, 1.0)
    else:
        c = wp.dot(d1, r)
        if e <= tiny:
            s = wp.clamp(-c / a, 0.0, 1.0)
        else:
            b = wp.dot(d1, d2)
            denominator = a * e - b * b
            if denominator > 0.0:
                s = wp.clamp((b * f - c * e) / denominator, 0.0, 1.0)
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = wp.clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = wp.clamp((b - c) / a, 0.0, 1.0)
    return wp.vec2(s, t)


@wp.func
def _segment_triangle(p0: wp.vec3, p1: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    """Closest pair between the segment p0-p1 and the triangle (a, b, c); the distance is zero where they cross."""
    result = _Closest()
    axis = p1 - p0
    point = _closest_point_triangle(p0, a, b, c)
    best = wp.length_sq(p0 - point)
    s = float(0.0)
    candidate = _closest_point_triangle(p1, a, b, c)
    distance = wp.length_sq(p1 - candidate)
    if distance < best:
        best = distance
        point = candidate
        s = 1.0
    for edge in range(3):
        x = a
        y = b
        if edge == 1:
            x = b
            y = c
        elif edge == 2:
            x = c
            y = a
        st = _segment_segment(p0, p1, x, y)
        candidate = x + st[1] * (y - x)
        distance = wp.length_sq(p0 + st[0] * axis - candidate)
        if distance < best:
            best = distance
            point = candidate
            s = st[0]
    result.crossing = 0
    normal = wp.cross(b - a, c - a)
    h0 = wp.dot(p0 - a, normal)
    h1 = wp.dot(p1 - a, normal)
    if h0 * h1 < 0.0:
        sx = h0 / (h0 - h1)
        x = p0 + sx * axis
        # Inside the triangle: on the inner side of all three edges.
        if (
            wp.dot(wp.cross(b - a, x - a), normal) >= 0.0
            and wp.dot(wp.cross(c - b, x - b), normal) >= 0.0
            and wp.dot(wp.cross(a - c, x - c), normal) >= 0.0
        ):
            result.crossing = 1
            best = 0.0
            point = x
            s = sx
    result.s = s
    result.point = point
    result.distance = wp.sqrt(best)
    return result


@wp.func
def _envelope_depth(
    a0: wp.vec3,
    b0: wp.vec3,
    a1: wp.vec3,
    b1: wp.vec3,
    radius: float,
    normal: wp.vec3,
    pa: wp.vec3,
    pb: wp.vec3,
    pc: wp.vec3,
):
    """How far the part (pa, pb, pc) must move against ``normal`` to lie under the capsule over its sweep.

    The largest of ``(q - x) . n + sqrt(R^2 - |(q - x) x n|^2)`` over the lattice points q of
    the part and the points x of the axis during the sweep from (a0, b0) to (a1, b1) that are
    within one radius of q across ``normal``; minus infinity if there is none.
    """
    # Across ``normal``: no point of the part is nearer to the swept axis than the part's middle is to the
    # predicted piece, less the reach of the part about its middle and less what the sweep moves the axis
    # sideways. Beyond one radius there is no row.
    centre = (pa + pb + pc) / 3.0
    reach = wp.sqrt(wp.max(wp.length_sq(pa - centre), wp.max(wp.length_sq(pb - centre), wp.length_sq(pc - centre))))
    side_a = (a0 - a1) - wp.dot(a0 - a1, normal) * normal
    side_b = (b0 - b1) - wp.dot(b0 - b1, normal) * normal
    sideways = wp.sqrt(wp.max(wp.length_sq(side_a), wp.length_sq(side_b)))
    along_axis = (b1 - a1) - wp.dot(b1 - a1, normal) * normal
    from_start = (centre - a1) - wp.dot(centre - a1, normal) * normal
    length_sq = wp.length_sq(along_axis)
    place = float(0.0)
    if length_sq > 0.0:
        place = wp.clamp(wp.dot(from_start, along_axis) / length_sq, 0.0, 1.0)
    if wp.length(from_start - place * along_axis) > radius + reach + sideways:
        return float(-wp.inf)
    spacing = STATION_RADII * radius
    stations = wp.max(1, int(wp.ceil(wp.length(b1 - a1) / spacing)))
    sweep = wp.max(wp.length(a1 - a0), wp.length(b1 - b0))
    poses = int(0)
    if sweep > spacing:
        poses = int(wp.ceil(sweep / spacing))
    best = float(-wp.inf)
    square = radius * radius
    for pose in range(poses + 1):
        fraction = float(1.0)
        if poses > 0:
            fraction = 1.0 - float(pose) / float(poses)
        a = a0 + fraction * (a1 - a0)
        b = b0 + fraction * (b1 - b0)
        for station in range(stations + 1):
            x = a + (float(station) / float(stations)) * (b - a)
            for i in range(LATTICE + 1):
                for j in range(LATTICE + 1 - i):
                    wa = float(i) / float(LATTICE)
                    wb = float(j) / float(LATTICE)
                    q = wa * pa + wb * pb + (1.0 - wa - wb) * pc
                    offset = q - x
                    along = wp.dot(offset, normal)
                    lateral = wp.dot(offset, offset) - along * along
                    if lateral < square:
                        best = wp.max(best, along + wp.sqrt(square - lateral))
    return best


@wp.func
def _part_row(
    a0: wp.vec3,
    b0: wp.vec3,
    a1: wp.vec3,
    b1: wp.vec3,
    radius: float,
    inward: int,
    outward: wp.vec3,
    pa: wp.vec3,
    pb: wp.vec3,
    pc: wp.vec3,
):
    """The row of one axis piece (a0-b0 now, a1-b1 predicted) against the triangle part (pa, pb, pc).

    Probe outside and the predicted axis clear of the part: the exact closest pair of the
    predicted piece. Probe inward: the depth along its outward direction. Axis through the
    part with the probe outside: the depth along the part's own normal.
    """
    row = _Row()
    closest = _segment_triangle(a1, b1, pa, pb, pc)
    if inward == 0 and closest.crossing == 0 and closest.distance > 0.0:
        row.depth = radius - closest.distance
        row.push = (closest.point - (a1 + closest.s * (b1 - a1))) / closest.distance
        return row
    normal = outward
    if inward == 0:
        normal = wp.normalize(wp.cross(pb - pa, pc - pa))
    row.depth = _envelope_depth(a0, b0, a1, b1, radius, normal, pa, pb, pc)
    row.push = -normal
    return row


@wp.func
def _predicted_pose(pose: wp.transform, velocity: wp.spatial_vector, com: wp.vec3, seconds: float):
    """The pose after ``seconds`` at the velocity of the centre of mass and the angular velocity (world frame)."""
    rotation = wp.transform_get_rotation(pose)
    centre = wp.transform_point(pose, com)
    angular = wp.spatial_bottom(velocity)
    turned = wp.normalize(rotation + 0.5 * seconds * (wp.quat(angular[0], angular[1], angular[2], 0.0) * rotation))
    moved = centre + seconds * wp.spatial_top(velocity)
    return wp.transform(moved - wp.quat_rotate(turned, com), turned)


@wp.func
def _boxes_meet(low_a: wp.vec3, high_a: wp.vec3, low_b: wp.vec3, high_b: wp.vec3):
    return (
        low_a[0] <= high_b[0]
        and low_b[0] <= high_a[0]
        and low_a[1] <= high_b[1]
        and low_b[1] <= high_a[1]
        and low_a[2] <= high_b[2]
        and low_b[2] <= high_a[2]
    )


@wp.kernel(enable_backward=False)
def _begin(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    seconds: float,
    predicted: wp.array[wp.transform],
    wrench: wp.array[wp.spatial_vector],
    tool_count: wp.array[int],
    tool_released: wp.array[float],
    tool_held: wp.array[float],
):
    """Clear the stored wrenches and the sums per tool; predict every body to the end of the next substep."""
    index = wp.tid()
    if index < predicted.shape[0]:
        predicted[index] = _predicted_pose(body_q[index], body_qd[index], body_com[index], seconds)
        wrench[index] = wp.spatial_vector()
    if index < tool_count.shape[0]:
        tool_count[index] = 0
        tool_released[index] = 0.0
        tool_held[index] = 0.0


@wp.kernel(enable_backward=False)
def _clear_particles(
    elastic: wp.array[wp.vec3], allowed: wp.array[float], allowed_back: wp.array[float], row_depth: wp.array[float]
):
    """Per particle: no elastic force yet, no bounds on its move yet, no row."""
    particle = wp.tid()
    elastic[particle] = wp.vec3(0.0)
    allowed[particle] = wp.inf
    allowed_back[particle] = wp.inf
    row_depth[particle] = -wp.inf


@wp.kernel(enable_backward=False)
def _allowed_moves(
    particle_q: wp.array[wp.vec3],
    indices: wp.array2d[int],
    poses: wp.array[wp.mat33],
    min_stretch: float,
    allowed: wp.array[float],
    allowed_back: wp.array[float],
):
    """Per node: the smallest ``MOVE_FRACTION * h_m * (sigma_min - min_stretch)`` over its tetrahedra [m].

    And the smallest ``MOVE_FRACTION * h_m * sigma_min``, the bound of a move back towards the tool.

    ``poses`` holds the inverse rest edge matrix of each tetrahedron: its rows are the
    gradients of the shape functions of nodes 1 to 3, minus their sum that of node 0, and
    the rest altitude at a node is one over the length of its gradient. ``wp.svd3`` returns
    a negative smallest singular value for an inverted element.
    """
    element = wp.tid()
    i = indices[element, 0]
    j = indices[element, 1]
    k = indices[element, 2]
    m = indices[element, 3]
    inverse = poses[element]
    deformation = (
        wp.matrix_from_cols(particle_q[j] - particle_q[i], particle_q[k] - particle_q[i], particle_q[m] - particle_q[i])
        * inverse
    )
    u = wp.mat33()
    sigma = wp.vec3()
    v = wp.mat33()
    wp.svd3(deformation, u, sigma, v)
    smallest = wp.min(sigma[0], wp.min(sigma[1], sigma[2]))
    margin = MOVE_FRACTION * (smallest - min_stretch)
    back = MOVE_FRACTION * smallest
    g1 = wp.vec3(inverse[0, 0], inverse[0, 1], inverse[0, 2])
    g2 = wp.vec3(inverse[1, 0], inverse[1, 1], inverse[1, 2])
    g3 = wp.vec3(inverse[2, 0], inverse[2, 1], inverse[2, 2])
    h0 = 1.0 / wp.length(g1 + g2 + g3)
    h1 = 1.0 / wp.length(g1)
    h2 = 1.0 / wp.length(g2)
    h3 = 1.0 / wp.length(g3)
    wp.atomic_min(allowed, i, margin * h0)
    wp.atomic_min(allowed, j, margin * h1)
    wp.atomic_min(allowed, k, margin * h2)
    wp.atomic_min(allowed, m, margin * h3)
    wp.atomic_min(allowed_back, i, back * h0)
    wp.atomic_min(allowed_back, j, back * h1)
    wp.atomic_min(allowed_back, k, back * h2)
    wp.atomic_min(allowed_back, m, back * h3)


@wp.kernel(enable_backward=False)
def _reaction(
    nodes: wp.array[int],
    particle_q: wp.array[wp.vec3],
    elastic: wp.array[wp.vec3],
    held: wp.array[int],
    push: wp.array[wp.vec3],
    owner: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    compressive: wp.array[int],
    wrench: wp.array[wp.spatial_vector],
):
    """The nodes held through the substep that just ended: which of them push back, and their force on the tool.

    The force on the tool is the elastic force on the node along its push direction, applied at the node.
    """
    particle = nodes[wp.tid()]
    compressive[particle] = 0
    if held[particle] == 0:
        return
    along = wp.dot(elastic[particle], push[particle])
    if along < 0.0:
        compressive[particle] = 1
        force = along * push[particle]
        body = owner[particle]
        lever = particle_q[particle] - wp.transform_point(body_q[body], body_com[body])
        wp.atomic_add(wrench, body, wp.spatial_vector(force, wp.cross(lever, force)))


@wp.kernel(enable_backward=False)
def _probe(
    mesh: wp.uint64,
    body_q: wp.array[wp.transform],
    predicted: wp.array[wp.transform],
    sample_body: wp.array[int],
    probe_local: wp.array[wp.vec3],
    start_local: wp.array[wp.vec3],
    end_local: wp.array[wp.vec3],
    sample_radius: wp.array[float],
    reach: float,
    inward: wp.array[int],
    outward: wp.array[wp.vec3],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    """Per axis piece: its search box, and whether its probe ends inside the tissue or went through the surface.

    The box is that of the piece over its sweep, grown by the radius and by the larger of the
    radius and the sweep: every part within that distance gets a row. A piece without a
    triangle in its box has no row and needs no probe. The sign is that of ray parity. The
    outward direction is the normal of the surface the probe went through, or, for a probe
    that started inside, the direction to the nearest surface point.
    """
    sample = wp.tid()
    body = sample_body[sample]
    now = body_q[body]
    then = predicted[body]
    radius = sample_radius[sample]
    a0 = wp.transform_point(now, start_local[sample])
    b0 = wp.transform_point(now, end_local[sample])
    a1 = wp.transform_point(then, start_local[sample])
    b1 = wp.transform_point(then, end_local[sample])
    margin = radius + wp.max(radius, wp.max(wp.length(a1 - a0), wp.length(b1 - b0)))
    grow = wp.vec3(margin, margin, margin)
    low = wp.min(wp.min(a0, b0), wp.min(a1, b1)) - grow
    high = wp.max(wp.max(a0, b0), wp.max(a1, b1)) + grow
    lower[sample] = low
    upper[sample] = high
    inward[sample] = 0
    outward[sample] = wp.vec3(0.0)
    near = wp.mesh_query_aabb(mesh, low, high)
    triangle = int(0)
    if not wp.mesh_query_aabb_next(near, triangle):
        return
    start = wp.transform_point(now, probe_local[sample])
    end = wp.transform_point(then, probe_local[sample])
    direction = wp.vec3(0.0)
    started = int(0)
    ended = int(0)
    crossed = int(0)
    query = wp.mesh_query_point(mesh, start, reach)
    if query.result and query.sign < 0.0:
        started = 1
        offset = wp.mesh_eval_position(mesh, query.face, query.u, query.v) - start
        length = wp.length(offset)
        direction = wp.mesh_eval_face_normal(mesh, query.face)
        if length > 0.0:
            direction = offset / length
    query = wp.mesh_query_point(mesh, end, reach)
    if query.result and query.sign < 0.0:
        ended = 1
        if started == 0:
            # Ended inside without a crossing found below: out by the nearest surface point of the end.
            offset = wp.mesh_eval_position(mesh, query.face, query.u, query.v) - end
            length = wp.length(offset)
            direction = wp.mesh_eval_face_normal(mesh, query.face)
            if length > 0.0:
                direction = offset / length
    path = end - start
    length = wp.length(path)
    if started == 0 and length > 0.0:
        ray = wp.mesh_query_ray(mesh, start, path / length, length)
        if ray.result:
            crossed = 1
            direction = wp.normalize(ray.normal)
            if wp.dot(direction, path) > 0.0:
                direction = -direction
    if ended != 0 or crossed != 0:
        inward[sample] = 1
    outward[sample] = direction


@wp.kernel(enable_backward=False)
def _capsule_boxes(
    capsule_first: wp.array[int],
    capsule_count: wp.array[int],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    capsule_lower: wp.array[wp.vec3],
    capsule_upper: wp.array[wp.vec3],
):
    """The box of every capsule: that of the search boxes of its pieces."""
    capsule = wp.tid()
    first = capsule_first[capsule]
    low = lower[first]
    high = upper[first]
    for sample in range(first + 1, first + capsule_count[capsule]):
        low = wp.min(low, lower[sample])
        high = wp.max(high, upper[sample])
    capsule_lower[capsule] = low
    capsule_upper[capsule] = high


@wp.kernel(enable_backward=False)
def _node_rows(
    nodes: wp.array[int],
    node_vertex: wp.array[int],
    adjacency_offset: wp.array[int],
    adjacency: wp.array[int],
    points: wp.array[wp.vec3],
    triangles: wp.array[int],
    capsule_first: wp.array[int],
    capsule_count: wp.array[int],
    capsule_lower: wp.array[wp.vec3],
    capsule_upper: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    predicted: wp.array[wp.transform],
    sample_body: wp.array[int],
    start_local: wp.array[wp.vec3],
    end_local: wp.array[wp.vec3],
    sample_radius: wp.array[float],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    inward: wp.array[int],
    outward: wp.array[wp.vec3],
    row_depth: wp.array[float],
    row_push: wp.array[wp.vec3],
    row_owner: wp.array[int],
):
    """Per free surface node: the deepest row of its own triangle parts, with its push direction and its axis piece.

    One thread per node walks its parts, the capsules and their pieces in a fixed order and
    keeps the first deepest row, so equal depths have one answer: the lowest part, then the
    lowest piece.
    """
    index = wp.tid()
    particle = nodes[index]
    x = points[node_vertex[index]]
    best = float(-wp.inf)
    best_push = wp.vec3(0.0)
    best_owner = int(-1)
    for entry in range(adjacency_offset[index], adjacency_offset[index + 1]):
        code = adjacency[entry]
        triangle = code // 3
        corner = code - 3 * triangle
        xj = points[triangles[3 * triangle + (corner + 1) % 3]]
        xl = points[triangles[3 * triangle + (corner + 2) % 3]]
        mid_j = 0.5 * (x + xj)
        mid_l = 0.5 * (x + xl)
        centroid = (x + xj + xl) / 3.0
        low = wp.min(wp.min(x, centroid), wp.min(mid_j, mid_l))
        high = wp.max(wp.max(x, centroid), wp.max(mid_j, mid_l))
        for capsule in range(capsule_first.shape[0]):
            if not _boxes_meet(low, high, capsule_lower[capsule], capsule_upper[capsule]):
                continue
            first = capsule_first[capsule]
            for sample in range(first, first + capsule_count[capsule]):
                if not _boxes_meet(low, high, lower[sample], upper[sample]):
                    continue
                body = sample_body[sample]
                a0 = wp.transform_point(body_q[body], start_local[sample])
                b0 = wp.transform_point(body_q[body], end_local[sample])
                a1 = wp.transform_point(predicted[body], start_local[sample])
                b1 = wp.transform_point(predicted[body], end_local[sample])
                if inward[sample] == 0:
                    # No part of a triangle is nearer to the piece than the triangle is: a triangle that the
                    # axis does not go through and that cannot give a deeper row than the best is left out.
                    whole = _segment_triangle(a1, b1, x, xj, xl)
                    if whole.crossing == 0 and sample_radius[sample] - whole.distance <= best:
                        continue
                for half in range(2):
                    pb = mid_j
                    pc = centroid
                    if half == 1:
                        pb = centroid
                        pc = mid_l
                    row = _part_row(a0, b0, a1, b1, sample_radius[sample], inward[sample], outward[sample], x, pb, pc)
                    if row.depth > best:
                        best = row.depth
                        best_push = row.push
                        best_owner = sample
    row_depth[particle] = best
    row_push[particle] = best_push
    row_owner[particle] = best_owner


@wp.kernel(enable_backward=False)
def _piece_rows(
    mesh: wp.uint64,
    points: wp.array[wp.vec3],
    triangles: wp.array[int],
    point_particle: wp.array[int],
    movable: wp.array[int],
    allowed: wp.array[float],
    body_q: wp.array[wp.transform],
    predicted: wp.array[wp.transform],
    sample_body: wp.array[int],
    start_local: wp.array[wp.vec3],
    end_local: wp.array[wp.vec3],
    sample_radius: wp.array[float],
    sample_tool: wp.array[int],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    inward: wp.array[int],
    outward: wp.array[wp.vec3],
    stored: wp.array[wp.vec3],
    held_depth: wp.array[float],
    held_direction: wp.array[wp.vec3],
    row_gap: wp.array[float],
    tool_count: wp.array[int],
):
    """Per axis piece: its held depth, the push direction of that row, its deepest row, and whether it touches.

    The held depth is the largest (row depth - allowed move of the row's vertex) over the
    parts in the piece's search box; a vertex that is fixed or outside the simulated region
    has no allowed move. The piece takes every part in its box, also where the closest
    point of the capsule lies on a neighbouring piece: its own distance to that part is then
    larger, and a contact that slides along the axis passes from piece to piece without a
    step. Without a row the clearance counts as one radius.
    """
    sample = wp.tid()
    body = sample_body[sample]
    radius = sample_radius[sample]
    a0 = wp.transform_point(body_q[body], start_local[sample])
    b0 = wp.transform_point(body_q[body], end_local[sample])
    a1 = wp.transform_point(predicted[body], start_local[sample])
    b1 = wp.transform_point(predicted[body], end_local[sample])
    best = -radius
    deepest = -radius
    direction = wp.vec3(0.0)
    query = wp.mesh_query_aabb(mesh, lower[sample], upper[sample])
    triangle = int(0)
    while wp.mesh_query_aabb_next(query, triangle):
        if inward[sample] == 0:
            # No part of a triangle is nearer to the piece than the triangle is, and no allowed move is
            # negative: a triangle that the axis does not go through and that cannot give a larger held
            # depth than the best is left out (the deepest row is never below the best held depth).
            whole = _segment_triangle(
                a1,
                b1,
                points[triangles[3 * triangle]],
                points[triangles[3 * triangle + 1]],
                points[triangles[3 * triangle + 2]],
            )
            if whole.crossing == 0 and radius - whole.distance <= best:
                continue
        for corner in range(3):
            vertex = triangles[3 * triangle + corner]
            x = points[vertex]
            xj = points[triangles[3 * triangle + (corner + 1) % 3]]
            xl = points[triangles[3 * triangle + (corner + 2) % 3]]
            mid_j = 0.5 * (x + xj)
            mid_l = 0.5 * (x + xl)
            centroid = (x + xj + xl) / 3.0
            move = float(0.0)
            particle = point_particle[vertex]
            if particle >= 0:
                if movable[particle] != 0:
                    move = wp.max(0.0, allowed[particle])
            for half in range(2):
                pb = mid_j
                pc = centroid
                if half == 1:
                    pb = centroid
                    pc = mid_l
                row = _part_row(a0, b0, a1, b1, radius, inward[sample], outward[sample], x, pb, pc)
                deepest = wp.max(deepest, row.depth)
                if row.depth - move > best:
                    best = row.depth - move
                    direction = row.push
    held_depth[sample] = best
    held_direction[sample] = direction
    row_gap[sample] = deepest
    if best > 0.0 or wp.length_sq(stored[sample]) > 0.0:
        wp.atomic_add(tool_count, sample_tool[sample], 1)


@wp.func
def _held_stiffness(pose: wp.transform, away: wp.vec3, inverse_mass: wp.mat33, full_step: float):
    """``K_c = m_eff(u) / h_full^2`` with the effective mass of the tool at the probe along the unit direction ``away``."""
    local = wp.quat_rotate_inv(wp.transform_get_rotation(pose), away)
    return 1.0 / (full_step * full_step * wp.dot(local, inverse_mass * local))


@wp.kernel(enable_backward=False)
def _piece_release(
    inverse_mass: wp.array[wp.mat33],
    step_seconds: wp.array[wp.float64],
    full_step: float,
    held_depth: wp.array[float],
    row_gap: wp.array[float],
    sample_tool: wp.array[int],
    tool_count: wp.array[int],
    sample_body: wp.array[int],
    probe_local: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    stored: wp.array[wp.vec3],
    wrench: wp.array[wp.spatial_vector],
    tool_released: wp.array[float],
    tool_held: wp.array[float],
):
    """The pieces without a held depth: their stored force runs out; what it loses and the held depths, per tool.

    ``tool_released`` gets the size of the stored force that the pieces of a tool lose [N],
    ``tool_held`` the sum of the held depths of its other pieces [m]: :func:`_piece_forces`
    hands the first to those pieces in proportion to their depths.
    """
    sample = wp.tid()
    depth = held_depth[sample]
    tool = sample_tool[sample]
    if depth >= 0.0:
        wp.atomic_add(tool_held, tool, depth)
        return
    force = stored[sample]
    size = wp.length(force)
    if size == 0.0:
        return
    count = float(tool_count[tool])
    body = sample_body[sample]
    pose = body_q[body]
    stiffness = _held_stiffness(pose, force / size, inverse_mass[sample], full_step)
    # The size runs out at K_c per full substep (the stand-off |I| n / K_c closes by the slack), the direction
    # stays. The proportional part answers the true clearance of the piece only, not the slack of tissue that
    # follows again.
    left = wp.max(0.0, size + stiffness * (float(step_seconds[0]) / full_step) * depth / count)
    wp.atomic_add(tool_released, tool, size - left)
    force = force * (left / size)
    stored[sample] = force
    if left > 0.0:
        applied = force * (wp.max(0.0, left + stiffness * wp.min(row_gap[sample], 0.0) / count) / left)
        if wp.length_sq(applied) > 0.0:
            lever = wp.transform_point(pose, probe_local[sample]) - wp.transform_point(pose, body_com[body])
            wp.atomic_add(wrench, body, wp.spatial_vector(applied, wp.cross(lever, applied)))


@wp.kernel(enable_backward=False)
def _piece_forces(
    inverse_mass: wp.array[wp.mat33],
    integral_ratio: float,
    step_seconds: wp.array[wp.float64],
    full_step: float,
    held_depth: wp.array[float],
    held_direction: wp.array[wp.vec3],
    sample_tool: wp.array[int],
    tool_count: wp.array[int],
    tool_released: wp.array[float],
    tool_held: wp.array[float],
    sample_body: wp.array[int],
    probe_local: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    stored: wp.array[wp.vec3],
    wrench: wp.array[wp.spatial_vector],
):
    """The pieces with a held depth: their stored force grows, takes what the tool's other pieces lost, and acts.

    ``inverse_mass`` holds, per piece and in the frame of its body, the inverse mass matrix
    of the tool at the probe [1/kg]; ``step_seconds`` the accepted step [s]. The force acts
    on the body at the probe.
    """
    sample = wp.tid()
    depth = held_depth[sample]
    force = stored[sample]
    if depth < 0.0 or not (depth > 0.0 or wp.length_sq(force) > 0.0):
        return
    tool = sample_tool[sample]
    count = float(tool_count[tool])
    body = sample_body[sample]
    pose = body_q[body]
    # Out of the tissue, along the row of the held depth.
    away = -held_direction[sample]
    stiffness = _held_stiffness(pose, away, inverse_mass[sample], full_step)
    # The stored force is an integral over time: rho = integral_ratio * K_c per full substep.
    grown = integral_ratio * stiffness * (float(step_seconds[0]) / full_step) * depth / count
    if tool_held[tool] > 0.0:
        grown += tool_released[tool] * depth / tool_held[tool]
    force = force + grown * away
    applied = force + (stiffness * depth / count) * away
    stored[sample] = force
    if wp.length_sq(applied) > 0.0:
        lever = wp.transform_point(pose, probe_local[sample]) - wp.transform_point(pose, body_com[body])
        wp.atomic_add(wrench, body, wp.spatial_vector(applied, wp.cross(lever, applied)))


@wp.kernel(enable_backward=False)
def _place(
    nodes: wp.array[int],
    row_depth: wp.array[float],
    row_push: wp.array[wp.vec3],
    row_owner: wp.array[int],
    sample_body: wp.array[int],
    held_depth: wp.array[float],
    allowed: wp.array[float],
    allowed_back: wp.array[float],
    compressive: wp.array[int],
    elastic: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    held: wp.array[int],
    push: wp.array[wp.vec3],
    owner: wp.array[int],
    balance: wp.array[wp.vec3],
):
    """Per free surface node: whether it is held through the next substep, and its move onto the swept capsule.

    Held: its row is inside the capsule, or it was held, pushed back and still has a row (it
    then follows the tool back by its clearance). A node moves by its depth, taken against
    the capsule pulled back by the held depth of the piece that owns its row, into the
    tissue at most by its allowed move and back at most by its allowed move back. A held
    node gets zero velocity and the force that balances its elastic force in the next
    tissue solve.
    """
    particle = nodes[wp.tid()]
    depth = row_depth[particle]
    keep = int(0)
    if depth > 0.0:
        keep = 1
    elif held[particle] != 0 and compressive[particle] != 0 and depth > -wp.inf:
        keep = 1
    if keep == 0:
        held[particle] = 0
        push[particle] = wp.vec3(0.0)
        owner[particle] = -1
        balance[particle] = wp.vec3(0.0)
        return
    piece = row_owner[particle]
    move = wp.max(depth, -wp.max(0.0, allowed_back[particle]))
    if depth > 0.0:
        move = wp.clamp(depth - wp.max(0.0, held_depth[piece]), 0.0, wp.max(0.0, allowed[particle]))
    particle_q[particle] = particle_q[particle] + move * row_push[particle]
    particle_qd[particle] = wp.vec3(0.0)
    held[particle] = 1
    push[particle] = row_push[particle]
    owner[particle] = sample_body[piece]
    balance[particle] = -elastic[particle]


@wp.kernel(enable_backward=False)
def _add_balance(balance: wp.array[wp.vec3], particle_f: wp.array[wp.vec3]):
    particle = wp.tid()
    particle_f[particle] += balance[particle]


def capsule_pieces(bodies, endpoints, radii, contact_radii=None):
    """Axis pieces of capsules with their probes, in the frames of their bodies.

    The probes are the centres of :func:`capsule_samples` (at most one contact radius apart;
    the first and the last lie where the caps of the drawn capsule reach as far as those of
    the collision capsule). A piece reaches from the midpoint to the probe before it to the
    midpoint to the probe after it; the first piece starts at the first probe and the last
    ends at the last.

    Returns:
        Body of each piece, its probe, its start and its end [m] in the body frame, its
        radius [m], the capsule of each piece, and for each capsule its first piece and its
        number of pieces.
    """
    bodies = np.asarray(bodies)
    sample_bodies, centres, sample_radii = capsule_samples(bodies, endpoints, radii, contact_radii)
    starts, ends, capsule_of, firsts, counts = centres.copy(), centres.copy(), [], [], []
    # capsule_samples lists the samples capsule by capsule; a body may carry several capsules.
    endpoints = np.asarray(endpoints, dtype=float)
    radii = np.asarray(radii, dtype=float)
    contact = radii if contact_radii is None else np.asarray(contact_radii, dtype=float)
    position = 0
    for capsule, ((start, end), radius, reach) in enumerate(zip(endpoints, radii, contact, strict=True)):
        length = float(np.linalg.norm(end - start))
        span = length - 2.0 * (reach - radius)
        count = int(math.ceil(span / reach * (1.0 - 1e-12))) + 1 if span > 0.0 else 1
        firsts.append(position)
        counts.append(count)
        for index in range(count):
            sample = position + index
            if index > 0:
                starts[sample] = 0.5 * (centres[sample - 1] + centres[sample])
            if index < count - 1:
                ends[sample] = 0.5 * (centres[sample] + centres[sample + 1])
            capsule_of.append(capsule)
        position += count
    if position != len(sample_bodies):
        raise RuntimeError("The pieces do not match the capsule samples")
    return (
        sample_bodies,
        centres,
        starts,
        ends,
        sample_radii,
        np.asarray(capsule_of, dtype=np.int32),
        np.asarray(firsts, dtype=np.int32),
        np.asarray(counts, dtype=np.int32),
    )


class SweptToolTissueContact:
    """Swept-capsule contact of rigid tool capsules with the surface of an FEM tissue.

    After an accepted substep: :meth:`update` with the accepted state (it moves the held
    nodes of that state and writes their holds into the tissue matrix). Before every attempt
    of the next substep: :meth:`add_forces` on the body forces and :meth:`add_balance` on the
    external nodal forces of the tissue solve. All of them only launch kernels. The owner of
    ``mesh`` moves its points to the accepted state and refits it before :meth:`update`.

    Args:
        model: Model of the bodies and of the tissue particles and tetrahedra.
        solver: The tissue solver (``SolverFEM``) after its first tangent refresh.
        mesh: Closed triangle surface of the tissue with outward-facing triangles (``wp.Mesh``).
        triangles: Vertex indices of the surface triangles, flat, three per triangle.
        point_particle: Tissue particle of each surface vertex, -1 for a vertex that is not one.
        fixed_particles: Particles that the tissue solver holds fixed.
        capsule_bodies: Body of each capsule, shape (capsules,).
        capsule_endpoints: End points of each capsule axis in the body frame [m], shape (capsules, 2, 3).
        capsule_radii: Radii of the collision capsules [m].
        contact_radii: Radii of the capsules that touch the tissue [m] (the drawn tool).
        capsule_tools: Tool of each capsule, shape (capsules,), numbered from zero.
        step_seconds: One ``float64`` on the device that holds the step [s] of the accepted
            substep when :meth:`update` runs.
        full_step: The full substep of the scene [s]: the sweep horizon.
        inverse_mass: Inverse mass matrix ``J M^-1 J^T`` of the tool at the probe of every axis
            piece [1/kg], in the frame of the piece's body, shape (pieces, 3, 3) in the order of
            :func:`capsule_pieces`; or one effective mass [kg] for every piece and direction.
            :meth:`set_inverse_mass` replaces it at a reset.
        reach: Largest distance from a probe at which the surface is looked for [m].

    Attributes:
        stored: The force that each axis piece stores [N]; with :attr:`wrench`, :attr:`held`,
            :attr:`push`, :attr:`owner`, :attr:`compressive` and :attr:`balance` the state
            between substeps.
        wrench: Stored reaction on every body [N, N m], force first, torque about the centre
            of mass, world frame.
        held: One int per particle: nonzero while the node is held on a capsule.
    """

    def __init__(
        self,
        model,
        solver,
        mesh,
        triangles,
        point_particle,
        fixed_particles,
        capsule_bodies,
        capsule_endpoints,
        capsule_radii,
        contact_radii,
        capsule_tools,
        *,
        step_seconds,
        full_step,
        inverse_mass,
        reach,
    ):
        device = model.device
        if step_seconds.shape != (1,) or step_seconds.dtype != wp.float64 or step_seconds.device != device:
            raise ValueError("step_seconds must be one float64 on the device of the model")
        self._step_seconds = step_seconds
        values = (full_step, reach)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Full step and reach must be finite and positive")
        capsule_tools = np.asarray(capsule_tools, dtype=np.int32)
        bodies, probes, starts, ends, radii, capsule_of, firsts, counts = capsule_pieces(
            capsule_bodies, capsule_endpoints, capsule_radii, contact_radii
        )
        if capsule_tools.shape != firsts.shape or capsule_tools.min() < 0:
            raise ValueError("capsule_tools must name one tool per capsule, numbered from zero")
        if np.any(bodies < 0) or np.any(bodies >= model.body_count):
            raise ValueError("Capsule bodies must exist in the model")
        self.model, self.solver, self.mesh = model, solver, mesh
        self.full_step, self.reach = float(full_step), float(reach)
        self.min_stretch = float(solver.min_stretch)
        self.sample_count, self.capsule_count = len(bodies), len(firsts)
        inverse_mass = self._inverse_mass_matrices(inverse_mass)
        self.tool_count = int(capsule_tools.max()) + 1
        self._triangles, self._point_particle = triangles, point_particle
        point_particle_host = point_particle.numpy()
        triangles_host = triangles.numpy().reshape(-1, 3)
        particles = model.particle_count
        fixed = np.zeros(particles, dtype=bool)
        fixed[np.asarray(fixed_particles, dtype=np.int64)] = True
        # The free surface nodes of the simulated region, with the triangle corners that each of them owns.
        vertices = np.unique(triangles_host)
        vertices = vertices[point_particle_host[vertices] >= 0]
        vertices = vertices[~fixed[point_particle_host[vertices]]]
        if not len(vertices):
            raise ValueError("The surface has no free tissue node")
        slot = np.full(len(point_particle_host), -1, dtype=np.int64)
        slot[vertices] = np.arange(len(vertices))
        corners = slot[triangles_host]
        owned = np.argwhere(corners >= 0)
        order = np.lexsort((owned[:, 1], owned[:, 0], corners[owned[:, 0], owned[:, 1]]))
        owned = owned[order]
        offsets = np.zeros(len(vertices) + 1, dtype=np.int32)
        np.add.at(offsets, corners[owned[:, 0], owned[:, 1]] + 1, 1)
        self.node_count = len(vertices)
        self.nodes_host = point_particle_host[vertices].astype(np.int32)
        movable = np.zeros(particles, dtype=np.int32)
        movable[self.nodes_host] = 1

        def array(values, dtype):
            return wp.array(values, dtype=dtype, device=device)

        self._nodes = array(self.nodes_host, int)
        self._node_vertex = array(vertices.astype(np.int32), int)
        self._adjacency_offset = array(np.cumsum(offsets, dtype=np.int32), int)
        self._adjacency = array((3 * owned[:, 0] + owned[:, 1]).astype(np.int32), int)
        self._movable = array(movable, int)
        self._sample_body = array(bodies, int)
        self._probe_local = array(probes, wp.vec3)
        self._start_local = array(starts, wp.vec3)
        self._end_local = array(ends, wp.vec3)
        self._sample_radius = array(radii, float)
        self._sample_tool = array(capsule_tools[capsule_of], int)
        self._capsule_first = array(firsts, int)
        self._capsule_count = array(counts, int)
        self.sample_bodies_host, self.sample_tools_host = bodies, capsule_tools[capsule_of]
        # Device data: a changed tool is a write of this array (set_inverse_mass), not a new graph.
        self._inverse_mass = array(inverse_mass, wp.mat33)

        def zeros(count, dtype):
            return wp.zeros(count, dtype=dtype, device=device)

        samples, bodies_count = self.sample_count, model.body_count
        # State between substeps.
        self.stored = zeros(samples, wp.vec3)
        self.wrench = zeros(bodies_count, wp.spatial_vector)
        self.held = zeros(particles, int)
        self.push = zeros(particles, wp.vec3)
        self.owner = wp.full(particles, -1, dtype=int, device=device)
        self.compressive = zeros(particles, int)
        self.balance = zeros(particles, wp.vec3)
        # Scratch of one update.
        self._predicted = zeros(bodies_count, wp.transform)
        self._tool_count = zeros(self.tool_count, int)
        self._tool_released, self._tool_held = zeros(self.tool_count, float), zeros(self.tool_count, float)
        self._elastic = zeros(particles, wp.vec3)
        self._allowed = zeros(particles, float)
        self._allowed_back = zeros(particles, float)
        self._row_depth = wp.full(particles, -np.inf, dtype=float, device=device)
        self._row_push = zeros(particles, wp.vec3)
        self._row_owner = wp.full(particles, -1, dtype=int, device=device)
        self._inward = zeros(samples, int)
        self._outward = zeros(samples, wp.vec3)
        self._lower, self._upper = zeros(samples, wp.vec3), zeros(samples, wp.vec3)
        self._capsule_lower, self._capsule_upper = (
            zeros(self.capsule_count, wp.vec3),
            zeros(self.capsule_count, wp.vec3),
        )
        self._held_depth = zeros(samples, float)
        self._row_gap = zeros(samples, float)
        self._held_direction = zeros(samples, wp.vec3)
        self.holds = NodeHolds(solver, self.nodes_host, self.held)
        # As the first contact's surface query: a batch that fits in one default block spreads its
        # independent walks of the bounding volume hierarchy with one query per block.
        self._query_block = 1 if device.is_cuda and samples <= 256 else 256

    def _inverse_mass_matrices(self, inverse_mass):
        """Check ``inverse_mass`` and return it per piece, shape (pieces, 3, 3); notes the range of ``K_c``."""
        inverse_mass = np.asarray(inverse_mass, dtype=float)
        if inverse_mass.ndim == 0:
            if not np.isfinite(inverse_mass) or inverse_mass <= 0:
                raise ValueError("The effective mass must be finite and positive")
            inverse_mass = np.tile(np.eye(3) / float(inverse_mass), (self.sample_count, 1, 1))
        if inverse_mass.shape != (self.sample_count, 3, 3) or not np.isfinite(inverse_mass).all():
            raise ValueError("inverse_mass must be one effective mass or finite [pieces,3,3] matrices")
        eigenvalues = np.linalg.eigvalsh(0.5 * (inverse_mass + inverse_mass.transpose(0, 2, 1)))
        if eigenvalues.min() <= 0:
            raise ValueError("Every inverse mass matrix must be positive definite")
        # K_c along the lightest and the heaviest direction of any piece [N/m].
        self.held_stiffness_range = (
            float(1.0 / (eigenvalues.max() * self.full_step**2)),
            float(1.0 / (eigenvalues.min() * self.full_step**2)),
        )
        return inverse_mass

    def set_inverse_mass(self, inverse_mass):
        """Replace the inverse mass matrices of the tool at the probes (as the constructor takes them).

        A host write of the device array that the force kernel reads, for the reset of a scene
        that places its tools anew: no graph changes. Not for the step.
        """
        self._inverse_mass.assign(self._inverse_mass_matrices(inverse_mass))

    @property
    def state(self) -> tuple:
        """The arrays that carry over from one accepted substep to the next."""
        return (self.stored, self.wrench, self.held, self.push, self.owner, self.compressive, self.balance)

    def update(self, body_q, body_qd, particle_q, particle_qd):
        """At an accepted state: reactions of the last substep, sweep, rows, forces, placement and holds of the next.

        Writes ``particle_q`` and ``particle_qd`` of the nodes that are held through the next
        substep, their holds into the tissue matrix, and the reaction on the bodies for
        :meth:`add_forces`.
        """
        model, device = self.model, self.model.device
        solver = self.solver
        wp.launch(
            _begin,
            max(model.body_count, self.tool_count),
            inputs=[body_q, body_qd, model.body_com, self.full_step],
            outputs=[self._predicted, self.wrench, self._tool_count, self._tool_released, self._tool_held],
            device=device,
        )
        wp.launch(
            _clear_particles,
            model.particle_count,
            outputs=[self._elastic, self._allowed, self._allowed_back, self._row_depth],
            device=device,
        )
        wp.launch(
            eval_tetrahedra,
            model.tet_count,
            inputs=[
                particle_q,
                particle_qd,
                model.tet_indices,
                model.tet_poses,
                model.tet_materials,
                wp.float32(solver.min_stretch),
                wp.float32(solver.max_stretch),
            ],
            outputs=[self._elastic],
            device=device,
        )
        wp.launch(
            _allowed_moves,
            model.tet_count,
            inputs=[particle_q, model.tet_indices, model.tet_poses, self.min_stretch],
            outputs=[self._allowed, self._allowed_back],
            device=device,
        )
        wp.launch(
            _reaction,
            self.node_count,
            inputs=[self._nodes, particle_q, self._elastic, self.held, self.push, self.owner, body_q, model.body_com],
            outputs=[self.compressive, self.wrench],
            device=device,
        )
        wp.launch(
            _probe,
            self.sample_count,
            inputs=[
                self.mesh.id,
                body_q,
                self._predicted,
                self._sample_body,
                self._probe_local,
                self._start_local,
                self._end_local,
                self._sample_radius,
                self.reach,
            ],
            outputs=[self._inward, self._outward, self._lower, self._upper],
            device=device,
            block_dim=self._query_block,
        )
        wp.launch(
            _capsule_boxes,
            self.capsule_count,
            inputs=[self._capsule_first, self._capsule_count, self._lower, self._upper],
            outputs=[self._capsule_lower, self._capsule_upper],
            device=device,
        )
        wp.launch(
            _node_rows,
            self.node_count,
            inputs=[
                self._nodes,
                self._node_vertex,
                self._adjacency_offset,
                self._adjacency,
                self.mesh.points,
                self._triangles,
                self._capsule_first,
                self._capsule_count,
                self._capsule_lower,
                self._capsule_upper,
                body_q,
                self._predicted,
                self._sample_body,
                self._start_local,
                self._end_local,
                self._sample_radius,
                self._lower,
                self._upper,
                self._inward,
                self._outward,
            ],
            outputs=[self._row_depth, self._row_push, self._row_owner],
            device=device,
        )
        wp.launch(
            _piece_rows,
            self.sample_count,
            inputs=[
                self.mesh.id,
                self.mesh.points,
                self._triangles,
                self._point_particle,
                self._movable,
                self._allowed,
                body_q,
                self._predicted,
                self._sample_body,
                self._start_local,
                self._end_local,
                self._sample_radius,
                self._sample_tool,
                self._lower,
                self._upper,
                self._inward,
                self._outward,
                self.stored,
            ],
            outputs=[self._held_depth, self._held_direction, self._row_gap, self._tool_count],
            device=device,
            block_dim=self._query_block,
        )
        wp.launch(
            _piece_release,
            self.sample_count,
            inputs=[
                self._inverse_mass,
                self._step_seconds,
                self.full_step,
                self._held_depth,
                self._row_gap,
                self._sample_tool,
                self._tool_count,
                self._sample_body,
                self._probe_local,
                body_q,
                model.body_com,
            ],
            outputs=[self.stored, self.wrench, self._tool_released, self._tool_held],
            device=device,
        )
        wp.launch(
            _piece_forces,
            self.sample_count,
            inputs=[
                self._inverse_mass,
                INTEGRAL_RATIO,
                self._step_seconds,
                self.full_step,
                self._held_depth,
                self._held_direction,
                self._sample_tool,
                self._tool_count,
                self._tool_released,
                self._tool_held,
                self._sample_body,
                self._probe_local,
                body_q,
                model.body_com,
            ],
            outputs=[self.stored, self.wrench],
            device=device,
        )
        wp.launch(
            _place,
            self.node_count,
            inputs=[
                self._nodes,
                self._row_depth,
                self._row_push,
                self._row_owner,
                self._sample_body,
                self._held_depth,
                self._allowed,
                self._allowed_back,
                self.compressive,
                self._elastic,
            ],
            outputs=[particle_q, particle_qd, self.held, self.push, self.owner, self.balance],
            device=device,
        )
        self.holds.apply()

    def add_forces(self, body_f):
        """Add the stored reaction to the body forces; does not change the contact state."""
        wp.launch(_add, self.model.body_count, inputs=[self.wrench, body_f], device=self.model.device)

    def add_balance(self, particle_f):
        """Add, for every held node, the force that balances its elastic force of the accepted state [N].

        The hold carries that force whatever is added here; with it the entry of a held node
        in the right-hand side of the tissue solve is only the change over one substep, so
        the relative tolerance of that solve is not set by the held nodes.
        """
        wp.launch(_add_balance, self.model.particle_count, inputs=[self.balance, particle_f], device=self.model.device)

    def reset(self):
        """Release every node and clear every stored force; the tissue matrix gets its own blocks back."""
        for array in (self.stored, self.wrench, self.held, self.push, self.compressive, self.balance):
            array.zero_()
        self.owner.fill_(-1)
        self.holds.apply()

    def modules(self):
        """Kernel modules that this object launches, with their block sizes, for a start-up that loads by name."""
        from ...fem import kernels, node_holds
        from . import tool_tissue_contact

        return (
            (__name__, tuple(sorted({256, self._query_block}))),
            (node_holds, (256,)),
            (kernels, (256,)),
            (tool_tissue_contact, (256,)),
        )

    def parameters(self):
        """Host values of the contact, without a device read."""
        return {
            "law": "swept capsules: free surface nodes are placed on the predicted capsule and held through the "
            "next substep; held tissue and tissue that cannot follow stop the tool by one stored force per axis piece",
            "sweep_horizon_s": self.full_step,
            "held_stiffness_rule": "K_c = 1 / (h_full^2 u.W u) with the inverse mass matrix W of the tool at the probe",
            "held_stiffness_lightest_direction_N_m": self.held_stiffness_range[0],
            "held_stiffness_heaviest_direction_N_m": self.held_stiffness_range[1],
            "integral_ratio": INTEGRAL_RATIO,
            "move_fraction": MOVE_FRACTION,
            "min_stretch": self.min_stretch,
            "hold_factor": self.holds.factor,
            "reach_m": self.reach,
            "axis_pieces": self.sample_count,
            "free_surface_nodes": self.node_count,
        }

    def diagnostics(self):
        """Explicit host readback for reports and tests; never called by the step.

        Per body: the deepest row of a free node (``deepest_row_m``), the held depth of its
        pieces (``held_depth_m``) and the smallest allowed move among its held nodes
        (``smallest_allowed_move_m``); the totals of the forces on the tools.
        """
        nodes = self.nodes_host
        held = self.held.numpy()[nodes] != 0
        depth = self._row_depth.numpy()[nodes]
        piece = self._row_owner.numpy()[nodes]
        owner = np.where(piece >= 0, self.sample_bodies_host[np.maximum(piece, 0)], -1)
        allowed = np.maximum(self._allowed.numpy()[nodes], 0.0)
        held_depth = self._held_depth.numpy()
        stored = np.linalg.norm(self.stored.numpy(), axis=1)
        wrench = self.wrench.numpy()
        bodies = self.model.body_count
        deepest = np.full(bodies, -np.inf)
        found = owner >= 0
        np.maximum.at(deepest, owner[found], depth[found])
        by_body = np.full(bodies, -np.inf)
        np.maximum.at(by_body, self.sample_bodies_host, held_depth)
        smallest = np.full(bodies, np.inf)
        mine = held & found
        np.minimum.at(smallest, owner[mine], allowed[mine])
        store = np.zeros(bodies)
        np.add.at(store, self.sample_bodies_host, stored)
        return {
            **self.parameters(),
            "held_nodes": int(held.sum()),
            "pushing_nodes": int((self.compressive.numpy()[nodes] != 0).sum()),
            "limited_nodes": int((held & (depth > allowed)).sum()),
            "largest_held_depth_m": float(held_depth.max()),
            "touching_pieces": self._tool_count.numpy().tolist(),
            "inward_probes": int(self._inward.numpy().sum()),
            "stored_force_N": float(stored.sum()),
            "loaded_pieces": int((stored > 0.0).sum()),
            "stored_force_by_body_N": store.tolist(),
            "deepest_row_m": [float(value) if np.isfinite(value) else None for value in deepest],
            "held_depth_m": [float(value) if np.isfinite(value) else None for value in by_body],
            "smallest_allowed_move_m": [float(value) if np.isfinite(value) else None for value in smallest],
            "applied_body_wrench_N_Nm": wrench.tolist(),
        }
