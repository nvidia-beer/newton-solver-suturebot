# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pins on thread nodes for :class:`ScreenPickTool`, and a thread laid along a path.

A pin is a prescribed node position of the thread solver, not a spring: a
node weighs a fraction of a milligram, and an explicit spring that is stable
on it could not hold the thread's tension. The node's slope stays free. The
prescribed position moves to its target in equal parts over the substeps of a
frame.
"""

import math

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _pin_node(q: wp.array[wp.float64], node: int, boundary_q: wp.array[wp.float64]):
    """Start a pin where the node is."""
    for axis in range(3):
        boundary_q[6 * node + axis] = q[6 * node + axis]


@wp.kernel(enable_backward=False)
def _follow_pins(
    pinned: wp.array[int],
    targets: wp.array[wp.vec3d],
    fraction: wp.float64,
    boundary_q: wp.array[wp.float64],
):
    """Move every pin's prescribed position ``fraction`` of the way to its target; launch over nodes."""
    node = wp.tid()
    if pinned[node] == 0:
        return
    for axis in range(3):
        value = boundary_q[6 * node + axis]
        boundary_q[6 * node + axis] = value + fraction * (targets[node][axis] - value)


def nearest_to_ray(viewer, points, allowed, origin, direction, click_radius):
    """Index of the allowed point nearest to a view ray in pixels, among those within ``click_radius`` pixels, or None."""
    offsets = points - origin
    along = offsets @ direction
    distance = np.linalg.norm(offsets - along[:, None] * direction, axis=1)
    camera = viewer.camera
    # World size of one pixel at each point's depth.
    pixel = 2.0 * np.maximum(along, 0.0) * math.tan(0.5 * math.radians(camera.fov)) / max(camera.height, 1)
    inside = allowed & (along > 0.0) & (distance <= click_radius * pixel)
    if not inside.any():
        return None
    return int(np.argmin(np.where(inside, distance / np.maximum(pixel, 1.0e-12), np.inf)))


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


class ThreadPins:
    """Pins on the nodes of a one-chain thread solver, usable as the backend of :class:`ScreenPickTool`.

    Pins are numbered in the order they were made; removing one moves the last
    into its place, as the tool expects.

    Args:
        viewer: Newton viewer; its camera gives the size of a pixel for picking.
        thread: Thread solver with one chain. Nodes that it already prescribes,
            such as node 0 and a fixed end, cannot be pinned.
    """

    # A press within this many pixels of a thread node grabs it.
    click_radius = 12.0

    def __init__(self, viewer, thread):
        if thread.n_envs != 1:
            raise ValueError("ThreadPins pins nodes of a solver with one chain")
        self.viewer, self.thread = viewer, thread
        nodes = thread.thread.node_count
        # Host copies of the device arrays; the device is only ever written from them.
        self._pinned = np.zeros(nodes, dtype=np.int32)
        self._target = np.zeros((nodes, 3), dtype=np.float64)
        self._pinned_device = wp.array(self._pinned, dtype=int, device=thread.device)
        self._target_device = wp.array(self._target, dtype=wp.vec3d, device=thread.device)
        self._prescribed = {int(dof) // 6 for dof in thread.fixed_dofs}
        self.pins = []
        self._dragged = None

    def candidates(self):
        """Positions of all nodes [m] and which of them can be pinned; one readback."""
        nodes = self.thread.q.numpy().reshape(-1, 6)[:, :3]
        free = np.array([node not in self._prescribed and not self._pinned[node] for node in range(len(nodes))])
        return nodes, free

    def pick(self, origin, direction):
        """Node under the view ray that can be pinned, and its position, or None; one readback."""
        nodes, free = self.candidates()
        nearest = nearest_to_ray(self.viewer, nodes, free, origin, direction, self.click_radius)
        return None if nearest is None else (nearest, nodes[nearest])

    def pin(self, node, position=None):
        """Pin ``node`` where it is; returns the node, or None if it cannot be pinned.

        ``position`` is the node's current position if the caller has it; it is read back otherwise.
        """
        node = int(node)
        if not 0 < node < len(self._pinned) or node in self._prescribed or self._pinned[node]:
            return None
        thread = self.thread
        self._target[node] = thread.q.numpy()[6 * node : 6 * node + 3] if position is None else position
        self._pinned[node] = 1
        wp.launch(_pin_node, dim=1, inputs=[thread.q, node], outputs=[thread.boundary_q], device=thread.device)
        thread.set_prescribed([6 * node, 6 * node + 1, 6 * node + 2], True)
        self._upload()
        return node

    def hold(self, node, position=None):
        """Pin ``node`` and keep the pin, as a finished drag would; returns the node or None."""
        node = self.pin(node, position)
        if node is not None:
            self.pins.append(node)
        return node

    def set_target(self, node, target):
        """Move the pin of ``node`` to the world position ``target`` [m]."""
        self._target[node] = target
        self._target_device.assign(self._target)

    def target(self, node):
        return self._target[node].copy()

    def release(self, node):
        """Free ``node``; it keeps its velocity."""
        self._pinned[node] = 0
        self._upload()
        self.thread.set_prescribed([6 * node, 6 * node + 1, 6 * node + 2], False)

    def clear(self):
        """Release every pin, including one that is being dragged."""
        self.cancel()
        for node in self.pins:
            self.release(node)
        self.pins.clear()

    def _upload(self):
        self._pinned_device.assign(self._pinned)
        self._target_device.assign(self._target)

    def follow(self, fraction):
        """Move the prescribed positions ``fraction`` of the way to their targets; call before every thread step.

        With ``fraction = 1 / (substeps left in the frame)`` the positions move
        in equal parts and arrive with the frame's last substep. Only launches
        a kernel, so a CUDA graph can capture it.
        """
        wp.launch(
            _follow_pins,
            dim=len(self._pinned),
            inputs=[self._pinned_device, self._target_device, wp.float64(fraction)],
            outputs=[self.thread.boundary_q],
            device=self.thread.device,
        )

    # ScreenPickTool backend.

    def begin(self, origin, direction):
        self.cancel()
        picked = self.pick(origin, direction)
        if picked is None:
            return None
        self._dragged = self.pin(*picked)
        return None if self._dragged is None else picked[1]

    def move(self, target):
        if self._dragged is not None:
            self.set_target(self._dragged, target)

    def finish(self):
        if self._dragged is not None:
            self.pins.append(self._dragged)
            self._dragged = None

    def cancel(self):
        if self._dragged is not None:
            self.release(self._dragged)
            self._dragged = None

    def remove(self, index):
        """Release pin ``index`` and move the last one into its place."""
        if not 0 <= index < len(self.pins):
            return False
        self.release(self.pins[index])
        self.pins[index] = self.pins[-1]
        self.pins.pop()
        return True
