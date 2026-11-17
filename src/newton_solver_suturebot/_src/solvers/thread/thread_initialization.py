# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pure geometry for initializing loose ANCF thread; no dynamics or event claims."""

import math

import numpy as np


def lay_along_path(path, node_count, spacing):
    """Nodal coordinates of a chain laid along a polyline, starting at its first point.

    Args:
        path: Polyline points [m], shape (count, 3); at least ``(node_count - 1) * spacing`` long.
        node_count: Number of nodes.
        spacing: Distance between neighbouring nodes along the path [m], the element length.

    Returns:
        Coordinates, shape (node_count, 6): position and unit tangent of the path at each node.
    """
    path = np.asarray(path, dtype=np.float64)
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("neighbouring path points must be distinct")
    starts = np.concatenate(([0.0], np.cumsum(lengths)))
    stations = spacing * np.arange(node_count)
    if stations[-1] > starts[-1] * (1.0 + 1.0e-9):
        raise ValueError(f"the path is {starts[-1]:.4f} m long; the chain needs {stations[-1]:.4f} m")
    segment = np.clip(np.searchsorted(starts, stations, side="right") - 1, 0, len(segments) - 1)
    directions = segments / lengths[:, None]
    positions = path[segment] + (stations - starts[segment])[:, None] * directions[segment]
    # A node at a corner of the path takes the bisector of the two directions.
    tangents = directions[segment].copy()
    at_corner = np.isclose(stations, starts[segment]) & (segment > 0)
    tangents[at_corner] += directions[segment[at_corner] - 1]
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    return np.column_stack((positions, tangents))


def serpentine(start, along, across, row_length, turn_radius, length, step):
    """Polyline of straight rows laid back and forth and joined by half-circle turns, at least ``length`` long.

    Args:
        start: First point [m].
        along: Unit direction of the first row; every other row runs the opposite way.
        across: Unit direction from one row to the next, perpendicular to ``along``.
        row_length: Length of the straight part of a row [m].
        turn_radius: Radius of the turns [m]; neighbouring rows are two radii apart.
        length: Length the polyline must reach [m].
        step: Largest distance between neighbouring points on a turn [m].
    """
    points = [np.asarray(start, dtype=np.float64)]
    along, across = np.asarray(along, dtype=np.float64), np.asarray(across, dtype=np.float64)
    angles = np.linspace(0.0, np.pi, max(2, int(np.ceil(np.pi * turn_radius / step))) + 1)[1:]
    chord = 2.0 * turn_radius * np.sin(0.5 * angles[0])
    covered, direction = 0.0, 1.0
    while covered < length:
        points.append(points[-1] + direction * row_length * along)
        center = points[-1] + turn_radius * across
        points.extend(
            center + turn_radius * (direction * np.sin(angle) * along - np.cos(angle) * across) for angle in angles
        )
        covered += row_length + len(angles) * chord
        direction = -direction
    return np.array(points)


def bridge(start, start_direction, end, end_direction, step, reach=1.0 / 3.0):
    """Polyline of a smooth curve that leaves ``start`` along one unit direction and reaches ``end`` along another.

    The curve is the cubic Bezier whose two inner control points lie the
    fraction ``reach`` of the chord from its ends; a longer reach makes a
    wider turn. Neighbouring points are at most about ``step`` metres apart.
    """
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    handle = reach * np.linalg.norm(end - start)
    controls = (start, start + handle * np.asarray(start_direction), end - handle * np.asarray(end_direction), end)
    # The control polygon is never shorter than the curve.
    polygon = sum(np.linalg.norm(b - a) for a, b in zip(controls[:-1], controls[1:], strict=True))
    t = np.linspace(0.0, 1.0, int(np.ceil(polygon / step)) + 2)[:, None]
    weights = ((1.0 - t) ** 3, 3.0 * (1.0 - t) ** 2 * t, 3.0 * (1.0 - t) * t**2, t**3)
    return sum(weight * control for weight, control in zip(weights, controls, strict=True))


def surface_top(points, vertices, triangles):
    """Height of the highest point of a triangle surface over each (x, y) [m]; ``-inf`` where the surface is not under it.

    Args:
        points: Positions [m], shape (count, 2) or (count, 3); only x and y are read.
        vertices: Vertices of the surface [m], shape (count, 3).
        triangles: Vertex indices of its triangles, shape (count, 3).
    """
    points = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    corners = vertices[np.asarray(triangles, dtype=np.int64).reshape(-1, 3)]
    low, high = corners[:, :, :2].min(axis=1), corners[:, :, :2].max(axis=1)
    top = np.full(len(points), -np.inf)
    # Points come along a path: a run of them shares the few triangles that reach into the run's own box.
    for begin in range(0, len(points), 16):
        run = points[begin : begin + 16, :2]
        chosen = np.flatnonzero(np.all((low <= run.max(axis=0)) & (high >= run.min(axis=0)), axis=1))
        first, edge1, edge2 = (
            corners[chosen, 0],
            corners[chosen, 1] - corners[chosen, 0],
            corners[chosen, 2] - corners[chosen, 0],
        )
        for index, point in enumerate(run, start=begin):
            near = np.flatnonzero(np.all((low[chosen] <= point) & (high[chosen] >= point), axis=1))
            a, b = edge1[near], edge2[near]
            area = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            offset = point - first[near, :2]
            # A triangle seen edge on has no area in the plan and nothing over it.
            flat = area == 0.0
            scale = np.where(flat, 1.0, area)
            u = (offset[:, 0] * b[:, 1] - offset[:, 1] * b[:, 0]) / scale
            v = (a[:, 0] * offset[:, 1] - a[:, 1] * offset[:, 0]) / scale
            over = ~flat & (u >= -1.0e-9) & (v >= -1.0e-9) & (u + v <= 1.0 + 1.0e-9)
            if over.any():
                top[index] = (first[near, 2] + u * a[:, 2] + v * b[:, 2])[over].max()
    return top


def border_walk(low, high, start, away, turn_radius, length):
    """Corner points of a walk from ``start`` to the nearest side of a box and on along its sides.

    The walk joins the nearest side that does not lie before ``start`` along ``away``, follows it the way
    ``away`` points (towards the farther corner where ``away`` is square to the side or the nearer corner is
    less than two turn radii off), and turns onto the next side at every corner. A side that has been walked
    moves in by two turn radii, so a walk longer than the box goes round again inside itself. A ``start``
    nearer to its side than one turn radius is put on the side.

    Args:
        low: Lower corner of the box (x, y) [m].
        high: Upper corner of the box (x, y) [m].
        start: First point (x, y) [m], inside the box.
        away: Unit direction (x, y) in which the walk leaves ``start``.
        turn_radius: Radius with which the corners are later rounded [m].
        length: Length that the walk must reach along the sides [m]; it has one side at least.

    Returns:
        The corner points, shape (count, 2), the first being ``start`` or its foot on the side.

    Raises:
        ValueError: If the box has no room left for a side between two corners before that length is reached.
    """
    low, high = np.array(low, dtype=np.float64), np.array(high, dtype=np.float64)
    start, away = np.asarray(start, dtype=np.float64), np.asarray(away, dtype=np.float64)
    sides = []
    for across in (0, 1):
        for inward, bound in ((1.0, low), (-1.0, high)):
            foot = start.copy()
            foot[across] = bound[across]
            if (foot - start) @ away >= 0.0:
                sides.append((abs(foot[across] - start[across]), across, inward, foot))
    distance, across, inward, position = min(sides, key=lambda side: side[0])
    points = [start, position] if distance >= turn_radius else [position]
    along = 1 - across
    ahead, back = high[along] - position[along], position[along] - low[along]
    sign = np.sign(away[along])
    # Two corners need a turn radius of the side each: where the side ends sooner the way ``away`` points,
    # and where ``away`` is square to the side, the walk takes the longer way.
    if sign == 0.0 or (ahead if sign > 0.0 else back) < 2.0 * turn_radius:
        sign = 1.0 if ahead >= back else -1.0
    covered = 0.0
    while len(points) < 2 or covered < length:
        corner = position.copy()
        corner[along] = high[along] if sign > 0.0 else low[along]
        side = sign * (corner[along] - position[along])
        # Every corner takes a turn radius of each of its sides; a walk that begins on a side has no corner there.
        if side <= (2.0 if len(points) > 1 else 1.0) * turn_radius:
            raise ValueError("Loose thread rows do not fit the full pad")
        covered += side
        points.append(corner)
        # The side just walked: the next round passes two turn radii inside it.
        if inward < 0.0:
            high[across] -= 2.0 * turn_radius
        else:
            low[across] += 2.0 * turn_radius
        position, along, across, sign, inward = corner, across, along, inward, -sign
    return np.array(points)


def rounded(points, radius, step):
    """Polyline through corner points (x, y) [m] with every corner replaced by an arc of ``radius``.

    Neighbouring points of the result are at most about ``step`` apart; a corner whose two legs are shorter
    than the arc needs is left sharp.
    """
    points = np.asarray(points, dtype=np.float64)
    path = [points[0]]
    for before, corner, after in zip(points[:-2], points[1:-1], points[2:], strict=True):
        leg_in, leg_out = np.linalg.norm(corner - before), np.linalg.norm(after - corner)
        into, out = (corner - before) / leg_in, (after - corner) / leg_out
        turn = math.acos(float(np.clip(into @ out, -1.0, 1.0)))
        back = radius * math.tan(0.5 * turn) if turn < math.pi - 1.0e-6 else math.inf
        if turn < 1.0e-9 or back > min(leg_in, leg_out) * (1.0 + 1.0e-9):
            path.append(corner)
            continue
        side = 1.0 if into[0] * out[1] - into[1] * out[0] > 0.0 else -1.0
        center = corner - back * into + side * radius * np.array([-into[1], into[0]])
        first = math.atan2(*(corner - back * into - center)[::-1])
        count = max(2, math.ceil(radius * turn / step))
        path.extend(
            center + radius * np.array([math.cos(angle), math.sin(angle)])
            for angle in first + side * turn * np.arange(count + 1) / count
        )
    path.append(points[-1])
    dense = [path[0]]
    for point in path[1:]:
        pieces = math.ceil(np.linalg.norm(point - dense[-1]) / step)
        # A corner arc may begin where the leg before it began: no piece, no second point.
        dense.extend(dense[-1] + (point - dense[-1]) * (np.arange(1, pieces + 1) / max(pieces, 1))[:, None])
    return np.array(dense)


def _clearance(points, obstacles, radius):
    """Smallest gap [m] between a thread of ``radius`` through ``points`` and capsules (starts, ends, radii)."""
    starts, ends, radii = (np.asarray(part, dtype=np.float64) for part in obstacles)
    axis = ends - starts
    along = np.einsum("pcj,cj->pc", points[:, None] - starts[None], axis) / np.maximum(
        np.einsum("cj,cj->c", axis, axis), 1.0e-30
    )
    nearest = starts[None] + np.clip(along, 0.0, 1.0)[:, :, None] * axis[None]
    return float((np.linalg.norm(points[:, None] - nearest, axis=2) - radii[None]).min() - radius)


def loose_thread_coordinates(
    head,
    pad,
    *,
    length=0.18,
    elements=90,
    radius=0.00015,
    triangles=None,
    stitch=None,
    turn_radius=0.01,
    obstacles=None,
):
    """Lay the spare thread of a needle on the pad: from the tail down behind the stitch, then along the pad's border.

    What the thread does once it is let go decides where it is laid (measured in the dVRK grasp scene):

    - It lies on the pad's surface, a thread radius above the highest point of the surface under each node,
      which is where the contact carries it. Laid higher it falls, and what falls is thrown about.
    - It trails behind the stitch: its first point on the pad is ``2 R`` straight behind the entry, ``R`` the
      radius of the needle's circle and "behind" the horizontal direction from the exit to the entry. The
      thread leaves its entry hole that way when the needle is pulled through, so a thread that lies there is
      drawn along its own length; one that lies beside or before the stitch must first swing round the entry,
      which throws its slack outward. No point of a needle that turns through the entry on its circle is
      farther from the entry than ``2 R``.
    - From there it goes to the nearest side of the pad's border that is not before that point, and along the
      border away from the stitch (:func:`border_walk`). The walk keeps one turn radius from the pad's
      bounding box, and its corners are arcs of the turn radius.
    - The turn radius is the thread's elasto-gravity length ``(E I / (rho A g))^(1/3)``: a thread that lies
      on the pad holds bends down to that radius, tighter ones spring open. It is also the length over which a
      push at one point bends the thread aside, which is why the walk keeps that distance from the edge. A
      pad that is too small for a walk with that radius gets half of it, then a quarter.
    - The lead from the tail to the first point on the pad is the smooth curve of :func:`bridge`. Of its
      reaches 1/3, 2/3 and 1 the one is taken that stays a thread radius clear of the ``obstacles`` and bends
      the thread least (the shortest chord of an element, to a hundredth; the shorter reach of equals); if
      none is clear, the one that comes nearest to being so. The first two elements are not looked at: they
      lie where the tail and its tangent put them (the thread's own contact leaves them out too).

    Args:
        head: World position [m] followed by the unit tangent of the needle's tail, shape (6,).
        pad: Vertices of the whole pad [m], shape (count, 3).
        length: Length of the thread [m].
        elements: Number of thread elements.
        radius: Radius of the thread [m].
        triangles: Vertex indices of the pad's surface triangles, shape (count, 3). Without them the pad is
            its bounding box: every node is laid a thread radius above the highest vertex. That is also
            where a node is laid under which the box has no surface: the walk follows the box, not the
            outline of a pad that does not fill it.
        stitch: Entry and exit of the planned stitch [m] and the radius of the needle's circle [m]. Without
            a stitch the thread trails the way the tail's tangent points, from one turn radius beyond the tail.
        turn_radius: Radius of the walk's corners and its distance from the pad's bounding box [m], halved
            or quartered on a pad that is too small for it; the default is the elasto-gravity length of a
            thread of 2 MPa, 0.15 mm radius and 1140 kg/m³.
        obstacles: World start points, end points and radii [m] of the rigid capsules that the thread can
            touch, shapes (count, 3), (count, 3) and (count,).

    Returns:
        ANCF coordinates, shape (elements + 1, 6): position and unit tangent of every node. This is where the
        thread is put, not a settled state: the lead hangs in the air and comes down in the first frames.
        It makes no puncture track, eyelet or constraint; the far end is free.

    Raises:
        ValueError: If the pad is too small for the walk, also with a quarter of the turn radius.
    """
    head, pad = np.asarray(head, dtype=float), np.asarray(pad, dtype=float)
    if head.shape != (6,) or not np.isfinite(head).all() or not np.isclose(np.linalg.norm(head[3:]), 1, atol=1e-7):
        raise ValueError("head must contain finite position and unit tangent")
    if pad.ndim != 2 or pad.shape[1] != 3 or not len(pad) or not np.isfinite(pad).all():
        raise ValueError("pad must contain finite full surface vertices [V,3]")
    if isinstance(elements, bool) or not isinstance(elements, (int, np.integer)) or elements < 2:
        raise ValueError("elements must be an integer at least two")
    if not np.isfinite(length) or length <= 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Thread length and radius must be finite positive metres")
    if not np.isfinite(turn_radius) or turn_radius <= 0:
        raise ValueError("The turn radius must be finite positive metres")
    spacing = length / elements
    if stitch is None:
        away = head[3:5]
        first = head[:2]
        behind = None
    else:
        entry, exit_point, needle_radius = stitch
        first = np.asarray(entry, dtype=float)[:2]
        away = first - np.asarray(exit_point, dtype=float)[:2]
        behind = 2.0 * float(needle_radius)
    reach = np.linalg.norm(away)
    # A tail that points straight up or down trails nowhere: the walk then leaves along x.
    away = away / reach if reach > 1.0e-9 else np.array([1.0, 0.0])
    rows = None
    # A pad too small for turns of the thread's own radius gets the widest of half and a quarter of it that fit.
    for turn in (turn_radius, 0.5 * turn_radius, 0.25 * turn_radius):
        low, high = pad.min(0)[:2] + turn, pad.max(0)[:2] - turn
        if np.any(high - low <= 2 * turn):
            continue
        start = np.clip(first + (turn if behind is None else behind) * away, low, high)
        walk = 0.0
        try:
            while True:
                rows = rounded(border_walk(low, high, start, away, turn, walk), turn, spacing / 2)
                # Enough for the chain: the lead is no shorter than the straight line from the tail, in the plan too.
                if np.linalg.norm(rows[0] - head[:2]) + np.linalg.norm(np.diff(rows, axis=0), axis=1).sum() >= length:
                    break
                walk += 2.0 * turn
        except ValueError:
            rows = None
            continue
        break
    if rows is None:
        raise ValueError("Loose thread rows do not fit the full pad")
    top = np.full(len(rows), pad[:, 2].max())
    if triangles is not None:
        # Where the pad's box has no surface under the walk (a gap between two blocks), the box's top stands in.
        under = surface_top(rows, pad, triangles)
        top = np.where(np.isfinite(under), under, top)
    rows = np.column_stack((rows, top + radius))
    along = (rows[1] - rows[0]) / np.linalg.norm(rows[1] - rows[0])
    best = None
    for part in (1 / 3, 2 / 3, 1):
        lead = bridge(head[:3] + spacing * head[3:], head[3:], rows[0], along, spacing / 2, part)
        q = lay_along_path(np.vstack((head[:3], lead, rows[1:])), elements + 1, spacing)
        station = spacing + np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(lead, axis=0), axis=1))]
        # A bent element's chord is shorter than the element: the tightest bend of the lead and its first laid element.
        chords = np.linalg.norm(np.diff(q[: math.ceil(station[-1] / spacing) + 2, :3], axis=0), axis=1)
        bend = round(float(chords.min() / spacing), 2)
        gap = np.inf
        if obstacles is not None and len(obstacles[2]):
            gap = _clearance(lead[station > 2 * spacing], obstacles, radius)
        # Clear of the obstacles by a thread radius, then least bent; of leads that touch, the one that touches least.
        rank = (True, bend) if gap >= radius else (False, gap)
        if best is None or rank > best[0]:
            best = (rank, q)
    q = best[1]
    q[0] = head
    return q
