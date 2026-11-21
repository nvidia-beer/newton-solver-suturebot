# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mapped surface contact for the position-only, cut-tetrahedron example.

Virtual tetrahedron corners can lie outside their remaining material. Only
the actual exterior and incision triangles participate in contact. Contact
forces are mapped back with the transpose of the surface interpolation.

Each contact row is a spring with a damper along its normal, and the Linear FEM
step takes it linearly implicitly, like a grip: the force is that of the row at
the end of the step if its normal velocity stayed as it is, and the row's
tangent, dt² k + dt c along the normal, stands in the solver's matrix during the
solve, where it multiplies the velocity change (``add_matrix`` before the solve,
``remove_matrix`` after it). A body at rest has no velocity change: the reaction
that balances its weight is the plain penalty, k times the penetration, whatever
the timestep. The tangent enters as a block-diagonal bound by row sums, so it
needs no matrix entry between the nodes of two pieces.

A row is in a step when its point is under the other surface at the start of
the step, or would be at its end at its present velocity. Within that step it is
linear: it pushes while the point is under the surface and pulls for what is
left of the step once the point is out, and it is gone in the next. This is
discrete contact, not a converged contact solve or continuous collision
detection.

The floor has Coulomb friction (``ground_friction``), with the law of the suture
thread on its contact planes (``solvers/thread/kernels.py``, docs/method.md,
"Friction on a contact plane"): a floor row that pushes is tied to an anchor on
the floor by a spring of the row's own stiffness k, whose force stops at mu times
the row's push N. The row sticks within the reach mu N / k of its anchor and
slides beyond it, and the anchor of a row that has slid follows it at that reach.
Like the push, the friction is linearly implicit: its force is that of the offset
at the end of the step if the row's velocity stayed as it is, and the spring that
has this force at this offset, k min(1, reach / offset) across the normal, stands
in the solver's matrix with dt² (the same bound by row sums). The anchors and
reaches carry over from step to step (``state``) and follow the surface through a
cut (``update_topology``, ``take_over``). The contact between cut pieces has no
friction.
"""

import math

import warp as wp

from ..solvers.fem.solve_diagonal import SolveDiagonal
from ..utils.step_time import StepTime


@wp.kernel(enable_backward=False)
def _copy_surface_topology(
    source_nodes: wp.array2d[int],
    source_weights: wp.array[wp.vec4],
    source_components: wp.array[int],
    source_kind: wp.array[int],
    source_area: wp.array[float],
    count: int,
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    indices: wp.array[int],
    components: wp.array[int],
    kind: wp.array[int],
    reference_area: wp.array[float],
):
    triangle = wp.tid()
    components[triangle] = -1
    kind[triangle] = 0
    reference_area[triangle] = 0.0
    for corner in range(3):
        vertex = 3 * triangle + corner
        indices[vertex] = 0
        weights[vertex] = wp.vec4(0.0)
        for k in range(4):
            nodes[vertex, k] = 0
        if triangle < count:
            indices[vertex] = vertex
            weights[vertex] = source_weights[vertex]
            for k in range(4):
                nodes[vertex, k] = source_nodes[vertex, k]
    if triangle < count:
        components[triangle] = source_components[triangle]
        kind[triangle] = source_kind[triangle]
        reference_area[triangle] = source_area[triangle]


def _copy_topology(destination, source, triangle_count):
    count = int(triangle_count)
    if count < 0 or count != triangle_count or count > source.components.shape[0]:
        raise ValueError("triangle count must fit the source surface")
    capacity = destination.components.shape[0]
    if count > capacity:
        raise ValueError(f"Surface needs {count} triangles, exceeding reserved capacity {capacity}")
    if destination.vertex_nodes.ptr == source.vertex_nodes.ptr:
        raise ValueError("source and destination surface topology must use separate storage")
    if capacity:
        wp.launch(
            _copy_surface_topology,
            dim=capacity,
            inputs=[
                source.vertex_nodes,
                source.weights,
                source.components,
                source.kind,
                source.reference_area,
                count,
                destination.vertex_nodes,
                destination.weights,
                destination.indices,
                destination.components,
                destination.kind,
                destination.reference_area,
            ],
            device=destination.vertex_nodes.device,
        )
    destination.count.fill_(count)
    destination.error.zero_()


@wp.func
def _closest_barycentric(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = wp.dot(ab, ap), wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)
    bp = p - b
    d3, d4 = wp.dot(ab, bp), wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return wp.vec3(1.0 - v, v, 0.0)
    cp = p - c
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
    return wp.vec3(1.0 - (vb + vc) * inverse, vb * inverse, vc * inverse)


@wp.kernel(enable_backward=False)
def _surface_points(
    q: wp.array[wp.vec3],
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    count: wp.array[int],
    points: wp.array[wp.vec3],
):
    vertex = wp.tid()
    value = wp.vec3(0.0)
    if vertex < 3 * count[0]:
        for k in range(4):
            value += weights[vertex][k] * q[nodes[vertex, k]]
    # A viewer can retain a fixed sequential index buffer across cuts. Unused
    # triangles must collapse even after the active prefix shrinks.
    points[vertex] = value


@wp.kernel(enable_backward=False)
def _surface_bounds(
    points: wp.array[wp.vec3],
    count: wp.array[int],
    components: wp.array[int],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    area: wp.array[float],
    groups: wp.array[int],
    multiple_components: wp.array[int],
):
    triangle = wp.tid()
    lower[triangle] = wp.vec3(1.0e10)
    upper[triangle] = wp.vec3(-1.0e10)
    area[triangle] = 0.0
    groups[triangle] = triangle + 1
    normal[triangle] = wp.vec3(0.0)
    if triangle < count[0]:
        a, b, c = points[3 * triangle], points[3 * triangle + 1], points[3 * triangle + 2]
        n = wp.cross(b - a, c - a)
        size = wp.length(n)
        if size > 1.0e-12:
            lower[triangle] = wp.min(wp.min(a, b), c)
            upper[triangle] = wp.max(wp.max(a, b), c)
            area[triangle] = 0.5 * size
            normal[triangle] = n / size
            groups[triangle] = 0
            if components[triangle] != components[0]:
                wp.atomic_max(multiple_components, 0, 1)


@wp.kernel(enable_backward=False)
def _floor_rows(
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    points: wp.array[wp.vec3],
    areas: wp.array[float],
    count: wp.array[int],
    height: float,
    density: float,
    indices: wp.array2d[int],
    coefficients: wp.array2d[float],
    normals: wp.array[wp.vec3],
    gaps: wp.array[float],
    stiffness: wp.array[float],
):
    row = wp.tid()
    if row < 3 * count[0] and areas[row // 3] > 0.0:
        for k in range(4):
            indices[row, k] = nodes[row, k]
            coefficients[row, k] = weights[row][k]
        normals[row] = wp.vec3(0.0, 0.0, 1.0)
        gaps[row] = points[row][2] - height
        stiffness[row] = density * areas[row // 3] / 3.0


@wp.kernel(enable_backward=False)
def _piece_rows(
    bvh: wp.uint64,
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    components: wp.array[int],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    areas: wp.array[float],
    count: wp.array[int],
    multiple_components: wp.array[int],
    row_offset: int,
    radius: float,
    density: float,
    indices: wp.array2d[int],
    coefficients: wp.array2d[float],
    contact_normals: wp.array[wp.vec3],
    gaps: wp.array[float],
    stiffness: wp.array[float],
):
    source = wp.tid()
    if source < count[0] and multiple_components[0] != 0 and areas[source] > 0.0:
        p = (points[3 * source] + points[3 * source + 1] + points[3 * source + 2]) / 3.0
        query = wp.bvh_query_aabb(bvh, p - wp.vec3(radius), p + wp.vec3(radius))
        candidate, target = int(0), int(-1)
        best = radius * radius
        barycentric = wp.vec3(0.0)
        while wp.bvh_query_next(query, candidate):
            if candidate < count[0] and components[candidate] != components[source] and areas[candidate] > 0.0:
                a, b, c = points[3 * candidate], points[3 * candidate + 1], points[3 * candidate + 2]
                bary = _closest_barycentric(p, a, b, c)
                delta = p - (bary[0] * a + bary[1] * b + bary[2] * c)
                distance = wp.dot(delta, delta)
                if distance < best:
                    target, best, barycentric = candidate, distance, bary
        # Pick the nearest surface before testing normals. Searching only for
        # opposing faces incorrectly contacts the back side of a nearby piece.
        if target >= 0 and wp.dot(normals[source], normals[target]) < -0.1:
            row = row_offset + source
            for corner in range(3):
                for k in range(4):
                    slot = 4 * corner + k
                    indices[row, slot] = nodes[3 * source + corner, k]
                    coefficients[row, slot] = weights[3 * source + corner][k] / 3.0
                    indices[row, 12 + slot] = nodes[3 * target + corner, k]
                    coefficients[row, 12 + slot] = -barycentric[corner] * weights[3 * target + corner][k]
            normal = normals[target]
            closest = (
                barycentric[0] * points[3 * target]
                + barycentric[1] * points[3 * target + 1]
                + barycentric[2] * points[3 * target + 2]
            )
            contact_normals[row] = normal
            gaps[row] = wp.dot(p - closest, normal)
            # Both sides are sampled, so halve their quadrature contribution.
            stiffness[row] = 0.5 * density * areas[source]


@wp.kernel(enable_backward=False)
def _contact_bounds(
    indices: wp.array2d[int],
    coefficients: wp.array2d[float],
    normals: wp.array[wp.vec3],
    gaps: wp.array[float],
    inverse_mass: wp.array[float],
    velocity: wp.array[wp.vec3],
    dt: wp.array[float],
    damping_ratio: float,
    stiffness: wp.array[float],
    damping: wp.array[float],
    normal_velocity: wp.array[float],
    node_tangent: wp.array[wp.mat33],
):
    # dt holds the timestep [s] in its one element (StepTime.seconds).
    row = wp.tid()
    k = stiffness[row]
    damping[row] = 0.0
    if k > 0.0:
        rate, inverse, weight_sum = float(0.0), float(0.0), float(0.0)
        for i in range(24):
            index, w = int(indices[row, i]), float(coefficients[row, i])
            rate += w * wp.dot(normals[row], velocity[index])
            weight_sum += wp.abs(w)
        if wp.min(gaps[row], gaps[row] + dt[0] * rate) >= 0.0:
            stiffness[row] = 0.0
            return
        for i in range(24):
            index, w = int(indices[row, i]), float(coefficients[row, i])
            # Repeated corner indices must be combined for effective mass.
            if w != 0.0:
                same = float(0.0)
                for j in range(24):
                    if indices[row, j] == index:
                        same += coefficients[row, j]
                inverse += w * same * inverse_mass[index]
        if inverse <= 0.0:
            stiffness[row] = 0.0
        else:
            c = 2.0 * damping_ratio * wp.sqrt(k / inverse)
            damping[row] = c
            normal_velocity[row] = rate
            # The row's tangent in the solver's matrix is (dt² k + dt c) (w n)(w n)ᵀ over its
            # coefficients w. Its blocks between different nodes are replaced by their absolute
            # row sums on the diagonal, |w_i| sum_j |w_j| n nᵀ, which is never less than the
            # tangent itself: the step stays stable, and a row between two pieces needs no
            # matrix entry between their nodes. For a surface corner that is a node (w = 1 on
            # one node) the bound is the tangent. The sums include all samples sharing a node.
            tangent = (dt[0] * dt[0] * k + dt[0] * c) * weight_sum * wp.outer(normals[row], normals[row])
            for i in range(24):
                w = wp.abs(coefficients[row, i])
                if w > 0.0:
                    wp.atomic_add(node_tangent, indices[row, i], w * tangent)


@wp.kernel(enable_backward=False)
def _scatter_contacts(
    indices: wp.array2d[int],
    coefficients: wp.array2d[float],
    normals: wp.array[wp.vec3],
    gaps: wp.array[float],
    stiffness: wp.array[float],
    damping: wp.array[float],
    normal_velocity: wp.array[float],
    dt: wp.array[float],
    piece_row_offset: int,
    active_rows: wp.array[int],
    force: wp.array[wp.vec3],
):
    # dt holds the timestep [s] in its one element (StepTime.seconds).
    row = wp.tid()
    if stiffness[row] > 0.0:
        # The row at the end of the step if its normal velocity stayed as it is; the matrix holds
        # what the velocity change adds (_contact_bounds). Not clamped: a row is linear within its
        # step. A row that only pushed would be dropped in every step in which its point moves out,
        # and a body then bounces between two states once dt² k exceeds four times its mass.
        rate = normal_velocity[row]
        magnitude = -stiffness[row] * (gaps[row] + dt[0] * rate) - damping[row] * rate
        wp.atomic_add(active_rows, int(row >= piece_row_offset), 1)
        value = magnitude * normals[row]
        for i in range(24):
            if coefficients[row, i] != 0.0:
                wp.atomic_add(force, indices[row, i], coefficients[row, i] * value)


@wp.func
def _plane_offset(position: wp.vec3, anchor: wp.vec3, normal: wp.vec3) -> wp.vec3:
    """From a row's friction anchor to the row's point, along its contact plane."""
    relative = position - anchor
    return relative - wp.dot(relative, normal) * normal


@wp.func
def _friction_scale(slip: float, stiffness: float, limit: float) -> float:
    """Friction force per slip distance: the spring's stiffness while sticking, the limit over the slip while sliding.

    The thread's function of this name in single precision (``solvers/thread/kernels.py``): also
    the stiffness of the spring that has the same force at this slip, which is the row's tangent.
    """
    if stiffness * slip <= limit:
        return stiffness
    return limit / slip


@wp.kernel(enable_backward=False)
def _floor_friction(
    indices: wp.array2d[int],
    coefficients: wp.array2d[float],
    normals: wp.array[wp.vec3],
    gaps: wp.array[float],
    stiffness: wp.array[float],
    damping: wp.array[float],
    normal_velocity: wp.array[float],
    points: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    dt: wp.array[float],
    friction: float,
    anchor: wp.array[wp.vec3],
    reach: wp.array[float],
    force: wp.array[wp.vec3],
    node_tangent: wp.array[wp.mat33],
):
    # One floor row; launch over the floor rows, after _contact_bounds and _scatter_contacts.
    # dt holds the timestep [s] in its one element (StepTime.seconds).
    row = wp.tid()
    k = stiffness[row]
    held = reach[row]
    # A row that is not in this step, or pulls in it, has no grip: its next anchor is its own point.
    reach[row] = 0.0
    if k > 0.0:
        normal, point = normals[row], points[row]
        if held > 0.0:
            # The anchor of a row that slid in the last step follows it at that step's reach.
            offset = _plane_offset(point, anchor[row], normal)
            slip = wp.length(offset)
            if slip > held:
                anchor[row] = point - (held / slip) * offset
        else:
            anchor[row] = point
        # The row's push, as _scatter_contacts applies it.
        rate = normal_velocity[row]
        push = -k * (gaps[row] + dt[0] * rate) - damping[row] * rate
        if push > 0.0:
            limit = friction * push
            reach[row] = limit / k
            moved, weight_sum = wp.vec3(0.0), float(0.0)
            for i in range(4):
                w = coefficients[row, i]
                moved += w * velocity[indices[row, i]]
                weight_sum += wp.abs(w)
            # The offset at the end of the step if the row's velocity stayed as it is. The matrix
            # holds what the velocity change adds: dt² times the spring that has this force at this
            # offset, across the normal. It is positive semidefinite, and its blocks between
            # different nodes are bounded by row sums on the diagonal like those of the push.
            offset = _plane_offset(point + dt[0] * moved, anchor[row], normal)
            scale = _friction_scale(wp.length(offset), k, limit)
            value = -scale * offset
            tangent = (dt[0] * dt[0] * scale * weight_sum) * (wp.identity(n=3, dtype=float) - wp.outer(normal, normal))
            for i in range(4):
                w = coefficients[row, i]
                if w != 0.0:
                    wp.atomic_add(force, indices[row, i], w * value)
                    wp.atomic_add(node_tangent, indices[row, i], wp.abs(w) * tangent)


@wp.func
def _off_triangle(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> float:
    """Squared distance of a point from a triangle."""
    bary = _closest_barycentric(p, a, b, c)
    delta = p - (bary[0] * a + bary[1] * b + bary[2] * c)
    return wp.dot(delta, delta)


@wp.kernel(enable_backward=False)
def _carry_friction(
    bvh: wp.uint64,
    old_points: wp.array[wp.vec3],
    old_areas: wp.array[float],
    old_normals: wp.array[wp.vec3],
    old_anchor: wp.array[wp.vec3],
    old_reach: wp.array[float],
    points: wp.array[wp.vec3],
    count: wp.array[int],
    radius: float,
    anchor: wp.array[wp.vec3],
    reach: wp.array[float],
):
    # One floor row of the new surface, whose corners are ``points``; launch over its floor rows.
    # ``bvh`` and the old arrays are those of the surface as it was, at the same positions.
    row = wp.tid()
    anchor[row] = wp.vec3(0.0)
    reach[row] = 0.0
    if row < 3 * count[0]:
        p = points[row]
        first = 3 * (row // 3)
        a1, b1, c1 = points[first], points[first + 1], points[first + 2]
        facing = wp.cross(b1 - a1, c1 - a1)
        centre = (a1 + b1 + c1) / 3.0
        query = wp.bvh_query_aabb(bvh, p - wp.vec3(radius), p + wp.vec3(radius))
        candidate, target = int(0), int(-1)
        best = radius * radius
        best_away, best_apart = int(0), float(0.0)
        barycentric = wp.vec3(0.0)
        while wp.bvh_query_next(query, candidate):
            if old_areas[candidate] > 0.0:
                a, b, c = old_points[3 * candidate], old_points[3 * candidate + 1], old_points[3 * candidate + 2]
                bary = _closest_barycentric(p, a, b, c)
                delta = p - (bary[0] * a + bary[1] * b + bary[2] * c)
                distance = wp.dot(delta, delta)
                if distance <= best:
                    # Several old triangles hold a point that is a corner of them all, and each
                    # has its own row there. The row's own triangle is the one that is the same
                    # surface as the new one: it faces the same way, and one of the two holds the
                    # other's middle (a part of a cut triangle lies in the whole, and the reverse).
                    away = int(0)
                    if wp.dot(facing, old_normals[candidate]) < 0.0:
                        away = 1
                    apart = _off_triangle(centre, a, b, c)
                    # A new triangle without area has no surface of its own to hold an old middle.
                    if wp.dot(facing, facing) > 0.0:
                        apart = wp.min(apart, _off_triangle((a + b + c) / 3.0, a1, b1, c1))
                    better = distance < best or target < 0
                    if not better:
                        better = away < best_away or (away == best_away and apart < best_apart)
                    if better:
                        target, best, barycentric = candidate, distance, bary
                        best_away, best_apart = away, apart
        if target >= 0:
            # The anchor and the reach are fields on the surface: a row of the new surface takes
            # them from the triangle of the old one that it lies on. A corner of that triangle
            # without grip is its own anchor and has no reach. A row that is a corner of its old
            # triangle gets that corner's anchor and reach back to the bit.
            carried, foot, grip = wp.vec3(0.0), wp.vec3(0.0), float(0.0)
            for corner in range(3):
                old = 3 * target + corner
                foot += barycentric[corner] * old_points[old]
                if old_reach[old] > 0.0:
                    carried += barycentric[corner] * old_anchor[old]
                    grip += barycentric[corner] * old_reach[old]
                else:
                    carried += barycentric[corner] * old_points[old]
            anchor[row] = carried + (p - foot)
            reach[row] = grip


class CutSurfaceContact:
    """Persistent GPU buffers for area-weighted surface contact.

    Recreate this helper when surface or particle capacity changes. The surface
    count and mappings can change in place between captured steps. ``evaluate``
    returns forces; it does not modify positions or the integrator's state.

    A step with contact is ``evaluate``, :meth:`add_matrix`, the solver's step
    with the returned forces, :meth:`remove_matrix`: the rows of ``evaluate``
    are linearly implicit in that step (see the module's text). The stiffnesses
    are penalties per area [N/m³]: a contact in balance is as deep as its
    pressure over the stiffness.

    ``ground_friction`` is the Coulomb coefficient between the surface and the
    floor; zero, the default, is a floor without friction. With friction the
    contact carries the anchors of the floor rows from one ``evaluate`` to the
    next (:attr:`state`): a new contact starts without grip, ``update_topology``
    carries the anchors over to the surface it publishes, and a contact that
    replaces another takes them over with :meth:`take_over`.

    The kernels read the timestep from ``step_time`` (:class:`StepTime`), which
    ``evaluate`` writes; inside a CUDA graph capture it must already hold its
    ``dt``. Pass the ``step_time`` of the solver that consumes the forces, so
    that the whole step reads one time step. Without it the contact keeps one
    of its own.
    """

    def __init__(
        self,
        surface,
        particle_q: wp.array,
        particle_inv_mass: wp.array,
        *,
        ground_height: float = 0.0,
        ground_stiffness: float = 2.0e7,
        piece_stiffness: float = 2.0e7,
        damping_ratio: float = 0.5,
        search_radius: float = 0.03,
        step_time: StepTime | None = None,
        ground_friction: float = 0.0,
    ):
        for name, value in (
            ("ground_stiffness", ground_stiffness),
            ("piece_stiffness", piece_stiffness),
            ("damping_ratio", damping_ratio),
            ("ground_friction", ground_friction),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(ground_height) or not math.isfinite(search_radius) or search_radius <= 0.0:
            raise ValueError("ground height must be finite and search radius finite and positive")
        self.surface, self.device = surface, particle_q.device
        self.step_time = StepTime(self.device) if step_time is None else step_time
        self.inverse_mass = particle_inv_mass
        self.ground_height, self.ground_stiffness = float(ground_height), float(ground_stiffness)
        self.piece_stiffness, self.damping_ratio = float(piece_stiffness), float(damping_ratio)
        self.search_radius = float(search_radius)
        self.ground_friction = float(ground_friction)
        self.capacity = surface.components.shape[0]
        rows = 4 * self.capacity
        self.points = wp.zeros(3 * self.capacity, dtype=wp.vec3, device=self.device)
        self.normals = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        self.areas = wp.zeros(self.capacity, dtype=float, device=self.device)
        self.lower = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        self.upper = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        # Equal centroids in the reserved empty slots can form a long LBVH
        # chain. Distinct inactive sort groups keep refits bounded; queries
        # still traverse the complete tree and do not filter by these groups.
        self.groups = wp.zeros(self.capacity, dtype=int, device=self.device)
        # Recompute on device so the same graph skips pointless self queries
        # before separation and immediately enables them after a topology edit.
        self.multiple_components = wp.zeros(1, dtype=int, device=self.device)
        self.indices = wp.zeros((rows, 24), dtype=int, device=self.device)
        self.coefficients = wp.zeros((rows, 24), dtype=float, device=self.device)
        self.contact_normals = wp.zeros(rows, dtype=wp.vec3, device=self.device)
        self.gaps = wp.zeros(rows, dtype=float, device=self.device)
        self.stiffness = wp.zeros(rows, dtype=float, device=self.device)
        self.damping = wp.zeros(rows, dtype=float, device=self.device)
        self.normal_velocity = wp.zeros(rows, dtype=float, device=self.device)
        # The rows' tangent of the last evaluate, dt² k + dt c along the normals [kg], as one block
        # per node; add_matrix puts it on the diagonal of the solver's system for one solve.
        self.node_tangent = wp.zeros(particle_q.shape[0], dtype=wp.mat33, device=self.device)
        self._solve_diagonal = SolveDiagonal(particle_q.shape[0], self.device)
        # Friction of the floor rows, from the last evaluate: each row's anchor on the floor [m] and
        # its reach [m], the slip mu N / k at which it slides; a row with reach zero has no grip.
        self.anchor = wp.zeros(3 * self.capacity, dtype=wp.vec3, device=self.device)
        self.reach = wp.zeros(3 * self.capacity, dtype=float, device=self.device)
        # What update_topology carries over to the rows of the surface it publishes, and where
        # those rows are.
        self._carried_anchor = wp.zeros_like(self.anchor)
        self._carried_reach = wp.zeros_like(self.reach)
        self._carried_points = wp.zeros_like(self.points)
        # Device-side diagnostics: ground rows, then inter-piece rows.
        self.active_rows = wp.zeros(2, dtype=int, device=self.device)
        self.forces = wp.zeros_like(particle_q)
        self._update_surface(particle_q)
        self.bvh = wp.Bvh(self.lower, self.upper, groups=self.groups) if self.capacity else None

    def _update_surface(self, q):
        self.multiple_components.zero_()
        if not self.capacity:
            return
        wp.launch(
            _surface_points,
            dim=3 * self.capacity,
            inputs=[q, self.surface.vertex_nodes, self.surface.weights, self.surface.count, self.points],
            device=self.device,
        )
        wp.launch(
            _surface_bounds,
            dim=self.capacity,
            inputs=[
                self.points,
                self.surface.count,
                self.surface.components,
                self.lower,
                self.upper,
                self.normals,
                self.areas,
                self.groups,
                self.multiple_components,
            ],
            device=self.device,
        )

    @staticmethod
    def reserve_surface(source, triangle_count: int, capacity: int):
        """Copy validated surface triangles into persistent reserved storage.

        This allocates once, before graph capture. Insufficient capacity raises
        an error before changing any live topology. No device count is read.
        """
        if int(capacity) != capacity or capacity < triangle_count:
            raise ValueError("reserved surface capacity must fit its active triangles")
        surface = type(source).allocate(int(capacity), source.vertex_nodes.device)
        _copy_topology(surface, source, triangle_count)
        return surface

    @property
    def state(self) -> tuple:
        """The contact's arrays that carry over from one step to the next: the anchors and reaches of the floor rows.

        Whoever runs a step that must leave the scene as it was (a warm-up, a graph capture)
        saves and restores them with the states of the step.
        """
        return (self.anchor, self.reach)

    def _has_friction(self) -> bool:
        return self.ground_friction > 0.0 and self.ground_stiffness > 0.0 and self.capacity > 0

    def _carry(self, previous, particle_q, anchor, reach):
        """Write the friction of ``previous``, on its surface as it was, for the rows of this contact's surface.

        A cut renumbers the surface triangles, and a row is a corner of one. A row's anchor and
        its reach are taken from the triangle of the old surface on which the row's point lies
        (the nearest one within the search radius), at the positions ``particle_q``: a row of a
        triangle that the cut left as it was gets its own anchor and reach back to the bit, and
        so does a row of a triangle that is a part of an old one, or the whole of old parts. A
        row with no old triangle that near starts without grip. The points and the search
        structure of ``previous`` must be those of its surface at ``particle_q``.
        """
        wp.launch(
            _surface_points,
            dim=3 * self.capacity,
            inputs=[
                particle_q,
                self.surface.vertex_nodes,
                self.surface.weights,
                self.surface.count,
                self._carried_points,
            ],
            device=self.device,
        )
        wp.launch(
            _carry_friction,
            dim=3 * self.capacity,
            inputs=[
                previous.bvh.id,
                previous.points,
                previous.areas,
                previous.normals,
                previous.anchor,
                previous.reach,
                self._carried_points,
                self.surface.count,
                self.search_radius,
                anchor,
                reach,
            ],
            device=self.device,
        )

    def take_over(self, previous: "CutSurfaceContact", particle_q: wp.array):
        """Take over the friction anchors of the contact that this one replaces.

        Call once, between steps, before the first ``evaluate`` of this contact: its surface
        is that of ``previous`` after an edit, in other storage. ``previous`` keeps its state.
        """
        if self._has_friction() and previous.capacity:
            previous._update_surface(particle_q)
            previous.bvh.refit()
            self._carry(previous, particle_q, self.anchor, self.reach)

    def update_topology(self, source, triangle_count: int, particle_q: wp.array):
        """Publish validated triangle mappings without invalidating a graph.

        Call between graph replays, after cloning particle coordinates. Capacity
        errors leave the current surface unchanged. Particle inverse masses
        must also be updated in their original buffer by the owning model.
        The friction anchors of the floor rows go over to the rows of the
        published surface: ``particle_q`` must hold the nodes of the surface
        as it was as well as those of the published one.
        """
        friction = self._has_friction()
        if friction:
            # The surface as it was, at these positions: the points and the search structure that
            # _carry reads stay those of the old surface until _update_surface below.
            self._update_surface(particle_q)
            self.bvh.refit()
        _copy_topology(self.surface, source, triangle_count)
        if friction:
            self._carry(self, particle_q, self._carried_anchor, self._carried_reach)
            wp.copy(self.anchor, self._carried_anchor)
            wp.copy(self.reach, self._carried_reach)
        self._update_surface(particle_q)
        if self.bvh is not None:
            # A refit preserves the original leaf order. Newly active triangles
            # need a fresh spatial hierarchy, retaining its allocated storage.
            self.bvh.rebuild()

    def update_surface(self, q: wp.array) -> wp.array:
        """Update and return surface positions [m] for rendering, without contact.

        This reads the live interpolation and active triangle count. The
        returned persistent buffer is also used by :meth:`evaluate`.
        """
        self._update_surface(q)
        return self.points

    def evaluate(self, q: wp.array, qd: wp.array, dt: float, *, piece_contact: bool = True) -> wp.array:
        """Return mapped contact forces [N] without allocating or synchronizing.

        Args:
            q: Basis node positions [m].
            qd: Basis node velocities [m/s].
            dt: Integration timestep [s].
            piece_contact: Enable collisions between distinct component labels.
        """
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("contact timestep must be finite and positive")
        self.step_time.set(dt)
        self.forces.zero_()
        self.active_rows.zero_()
        self.node_tangent.zero_()
        if not self.capacity:
            return self.forces
        self.stiffness.zero_()
        self._update_surface(q)
        if self.ground_stiffness:
            wp.launch(
                _floor_rows,
                dim=3 * self.capacity,
                inputs=[
                    self.surface.vertex_nodes,
                    self.surface.weights,
                    self.points,
                    self.areas,
                    self.surface.count,
                    self.ground_height,
                    self.ground_stiffness,
                    self.indices,
                    self.coefficients,
                    self.contact_normals,
                    self.gaps,
                    self.stiffness,
                ],
                device=self.device,
            )
        if piece_contact and self.piece_stiffness:
            self.bvh.refit()
            wp.launch(
                _piece_rows,
                dim=self.capacity,
                inputs=[
                    self.bvh.id,
                    self.surface.vertex_nodes,
                    self.surface.weights,
                    self.surface.components,
                    self.points,
                    self.normals,
                    self.areas,
                    self.surface.count,
                    self.multiple_components,
                    3 * self.capacity,
                    self.search_radius,
                    self.piece_stiffness,
                    self.indices,
                    self.coefficients,
                    self.contact_normals,
                    self.gaps,
                    self.stiffness,
                ],
                device=self.device,
            )
        wp.launch(
            _contact_bounds,
            dim=4 * self.capacity,
            inputs=[
                self.indices,
                self.coefficients,
                self.contact_normals,
                self.gaps,
                self.inverse_mass,
                qd,
                self.step_time.seconds,
                self.damping_ratio,
                self.stiffness,
                self.damping,
                self.normal_velocity,
                self.node_tangent,
            ],
            device=self.device,
        )
        wp.launch(
            _scatter_contacts,
            dim=4 * self.capacity,
            inputs=[
                self.indices,
                self.coefficients,
                self.contact_normals,
                self.gaps,
                self.stiffness,
                self.damping,
                self.normal_velocity,
                self.step_time.seconds,
                3 * self.capacity,
                self.active_rows,
                self.forces,
            ],
            device=self.device,
        )
        if self._has_friction():
            wp.launch(
                _floor_friction,
                dim=3 * self.capacity,
                inputs=[
                    self.indices,
                    self.coefficients,
                    self.contact_normals,
                    self.gaps,
                    self.stiffness,
                    self.damping,
                    self.normal_velocity,
                    self.points,
                    qd,
                    self.step_time.seconds,
                    self.ground_friction,
                    self.anchor,
                    self.reach,
                    self.forces,
                    self.node_tangent,
                ],
                device=self.device,
            )
        return self.forces

    def add_matrix(self, solver):
        """Put the tangent of the rows of the last :meth:`evaluate` into the solver's system, for one solve.

        Call after ``evaluate`` and after everything else that changes the matrix in this step
        (the tangent refresh, the grips), directly before the solver's step; call
        :meth:`remove_matrix` directly after that step. One launch, without allocation or
        synchronization: the diagonal blocks that the solve reads are saved and replaced here
        and put back there (:class:`SolveDiagonal`).
        """
        self._solve_diagonal.add(solver, self.node_tangent)

    def remove_matrix(self, solver):
        """Take the rows' tangent out of the solver's system again, to the bit."""
        self._solve_diagonal.remove(solver)
