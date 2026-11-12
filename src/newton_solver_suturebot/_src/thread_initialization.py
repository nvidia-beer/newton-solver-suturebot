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


def loose_thread_coordinates(head, pad, *, length=0.18, elements=90, radius=0.00015):
    """Lay a free tail in loose rows above the full pad at a measured needle tail.

    ``head`` is world position followed by unit tangent, shape [6]; ``pad``
    contains current full phantom vertices [V,3]. Returns ANCF coordinates
    [elements+1,6]. This initialization is not a settled or collision-free
    certificate: physical contact and gravity must subsequently be integrated.
    It does not create puncture tracks, eyelets or constraints at the free end.
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
    low, high = pad.min(0), pad.max(0)
    spacing, candidates = length / elements, []
    inset = 2 * spacing
    if np.any(high[:2] - low[:2] <= 4 * inset):
        raise ValueError("Full pad is too small for loose thread rows")
    height = high[2] + 2 * radius
    for axis in (0, 1):
        other = 1 - axis
        for side in (-1, 1):
            for turn in (0.005, 0.003, 0.0015):
                along, across = np.eye(3)[axis], side * np.eye(3)[other]
                start = np.array([low[0] + inset, low[1] + inset, height])
                if side < 0:
                    start[other] = high[other] - inset
                start[axis] += turn
                row_length = high[axis] - low[axis] - 2 * (inset + turn)
                if row_length <= spacing:
                    continue
                rows = serpentine(start, along, across, row_length, turn, length, spacing / 2)
                lead = bridge(head[:3] + spacing * head[3:], head[3:], start, along, spacing / 2)
                path = np.vstack((head[:3], lead, rows[1:]))
                q = lay_along_path(path, elements + 1, spacing)
                lead_length = np.linalg.norm(np.diff(path[: len(lead) + 1], axis=0), axis=1).sum()
                loose = q[min(elements, math.ceil(lead_length / spacing)) :, :2]
                if np.any(loose < low[:2] + radius) or np.any(loose > high[:2] - radius):
                    continue
                q[0] = head
                candidates.append((lead_length, -turn, q))
    if not candidates:
        raise ValueError("Loose thread rows do not fit the full pad")
    return min(candidates, key=lambda item: item[:2])[2]
