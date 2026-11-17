# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Draw a thread as a tube along its cubic Hermite centerline, with optional nodes and slopes."""

import newton
import warp as wp
from newton.math import quat_between_vectors_robust

from newton_solver_suturebot.examples._startup import load_kernel_modules


@wp.kernel(enable_backward=False)
def _sample_centerline(
    q: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    samples: int,
    centers: wp.array[wp.vec3],
    nodes: wp.array[wp.vec3],
):
    """Points on the centerline, ``samples`` per element plus the last node; launch over all points."""
    point = wp.tid()
    element = wp.min(point // samples, lengths.shape[0] - 1)
    xi = wp.float64(point - element * samples) / wp.float64(samples)
    xi2, xi3 = xi * xi, xi * xi * xi
    left = 6 * element
    right = left + 6
    p0 = wp.vec3d(q[left], q[left + 1], q[left + 2])
    d0 = wp.vec3d(q[left + 3], q[left + 4], q[left + 5])
    p1 = wp.vec3d(q[right], q[right + 1], q[right + 2])
    d1 = wp.vec3d(q[right + 3], q[right + 4], q[right + 5])
    position = (
        (wp.float64(1.0) - wp.float64(3.0) * xi2 + wp.float64(2.0) * xi3) * p0
        + lengths[element] * (xi - wp.float64(2.0) * xi2 + xi3) * d0
        + (wp.float64(3.0) * xi2 - wp.float64(2.0) * xi3) * p1
        + lengths[element] * (-xi2 + xi3) * d1
    )
    centers[point] = wp.vec3(position)
    if point % samples == 0:
        nodes[point // samples] = wp.vec3(position)


@wp.func
def _joint_overlap(direction: wp.vec3, neighbor: wp.vec3, radius: float):
    cosine = wp.clamp(wp.dot(direction, wp.normalize(neighbor)), -1.0, 1.0)
    # Overlap cylinder ends across a bend so their flat caps leave no visible cracks.
    return radius * wp.min(wp.sqrt((1.0 - cosine) / wp.max(1.0 + cosine, 1.0e-6)), 1.0)


@wp.kernel(enable_backward=False)
def _tube_segments(
    centers: wp.array[wp.vec3],
    radius: float,
    transforms: wp.array[wp.transform],
    scales: wp.array[wp.vec3],
):
    """One cylinder between neighbouring centerline points; launch over all segments."""
    segment = wp.tid()
    start, end = centers[segment], centers[segment + 1]
    delta = end - start
    length = wp.length(delta)
    direction = wp.vec3(0.0, 0.0, 1.0)
    if length > 1.0e-9:
        direction = delta / length
    start_overlap, end_overlap = float(0.0), float(0.0)
    if segment > 0:
        start_overlap = _joint_overlap(direction, start - centers[segment - 1], radius)
    if segment + 2 < centers.shape[0]:
        end_overlap = _joint_overlap(direction, centers[segment + 2] - end, radius)
    center = 0.5 * (start + end) + 0.5 * (end_overlap - start_overlap) * direction
    transforms[segment] = wp.transform(center, quat_between_vectors_robust(wp.vec3(0.0, 0.0, 1.0), direction))
    scales[segment] = wp.vec3(radius, radius, 0.5 * (length + start_overlap + end_overlap))


@wp.kernel(enable_backward=False)
def _slope_lines(q: wp.array[wp.float64], length: wp.float64, starts: wp.array[wp.vec3], ends: wp.array[wp.vec3]):
    """A line from each node along its slope vector, ``length`` per unit slope; launch over nodes."""
    node = wp.tid()
    position = wp.vec3d(q[6 * node], q[6 * node + 1], q[6 * node + 2])
    slope = wp.vec3d(q[6 * node + 3], q[6 * node + 4], q[6 * node + 5])
    starts[node] = wp.vec3(position)
    ends[node] = wp.vec3(position + length * slope)


class ThreadView:
    """Viewer geometry of the first chain of a thread solver, updated on the device.

    Args:
        viewer: Newton viewer.
        solver: Thread solver; its first chain is drawn.
        radius: Thread radius [m].
        samples: Tube segments per element.
        color: Tube colour.
    """

    def __init__(self, viewer, solver, radius, *, samples=4, color=(0.36, 0.2, 0.62)):
        self.viewer, self.solver, self.radius, self.samples = viewer, solver, float(radius), samples
        thread, device = solver.thread, solver.thread.device
        points = thread.element_count * samples + 1
        self._centers = wp.empty(points, dtype=wp.vec3, device=device)
        self._nodes = wp.empty(thread.node_count, dtype=wp.vec3, device=device)
        self._transforms = wp.empty(points - 1, dtype=wp.transform, device=device)
        self._scales = wp.empty(points - 1, dtype=wp.vec3, device=device)
        self._colors = wp.full(points - 1, wp.vec3(*color), dtype=wp.vec3, device=device)
        self._materials = wp.full(points - 1, wp.vec4(0.5, 0.0, 0.0, 0.0), dtype=wp.vec4, device=device)
        self._node_colors = wp.full(thread.node_count, wp.vec3(0.95, 0.9, 0.2), dtype=wp.vec3, device=device)
        self._slope_starts = wp.empty(thread.node_count, dtype=wp.vec3, device=device)
        self._slope_ends = wp.empty_like(self._slope_starts)
        self._slope_colors = wp.full(thread.node_count, wp.vec3(0.95, 0.15, 0.15), dtype=wp.vec3, device=device)
        # Slopes are drawn half an element long.
        self._slope_length = wp.float64(0.5 * float(thread.lengths.numpy().mean()))
        self._debug_drawn = False
        # The first render of a run launches these kernels: their module loads with the view, not then.
        load_kernel_modules(device, _sample_centerline, _tube_segments, _slope_lines)
        viewer.log_geo("thread/cylinder", newton.GeoType.CYLINDER, (1.0, 1.0), 0.0, True, hidden=True)

    def render(self, *, show_nodes=False):
        """Draw the tube and, with ``show_nodes``, the nodes and their slopes; call inside a viewer frame."""
        solver, device = self.solver, self.solver.thread.device
        wp.launch(
            _sample_centerline,
            dim=len(self._centers),
            inputs=[solver.q, solver.thread.lengths, self.samples],
            outputs=[self._centers, self._nodes],
            device=device,
        )
        wp.launch(
            _tube_segments,
            dim=len(self._transforms),
            inputs=[self._centers, self.radius],
            outputs=[self._transforms, self._scales],
            device=device,
        )
        self.viewer.log_instances(
            "thread/tube", "thread/cylinder", self._transforms, self._scales, self._colors, self._materials
        )
        if show_nodes:
            wp.launch(
                _slope_lines,
                dim=len(self._nodes),
                inputs=[solver.q, self._slope_length],
                outputs=[self._slope_starts, self._slope_ends],
                device=device,
            )
            self.viewer.log_points("thread/nodes", self._nodes, radii=2.0 * self.radius, colors=self._node_colors)
            self.viewer.log_lines("thread/slopes", self._slope_starts, self._slope_ends, self._slope_colors)
            self._debug_drawn = True
        elif self._debug_drawn:
            self.viewer.log_points("thread/nodes", None)
            self.viewer.log_lines("thread/slopes", None, None, None)
            self._debug_drawn = False
