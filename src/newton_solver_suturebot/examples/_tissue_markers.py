# SPDX-License-Identifier: Apache-2.0
"""Sphere markers that ride on deforming tissue, for the examples of several folders.

A marker is bound to one triangle of the drawn tissue surface: three node indices into the
particle array the tissue is drawn from, and three weights that add up to one. It is drawn
at that material point of the current state, so it moves with the tissue. A point picked on
the surface brings its triangle and weights with it (:meth:`TissueMarkers.attach`). A
position in space, such as the place where a needle went through the surface, is bound to
the closest point of the drawn surface in the state of that moment
(:meth:`TissueMarkers.attach_point`).

Everything a frame needs exists before the run: the reserve of markers and the search
scratch are allocated when the object is made, the kernel module is loaded there, and
:meth:`TissueMarkers.warm` issues every launch once in the start-up draw. A frame allocates
no array and reads none back: a new marker goes to the device as kernel arguments, and its
triangle is found there.
"""

import numpy as np
import warp as wp

from newton_solver_suturebot.examples._startup import load_kernel_modules

# Squared distance and triangle index that no triangle has: the search scratch between two searches.
_FAR = 3.0e38
_NO_TRIANGLE = 2147483647


@wp.func
def _triangle_weights(point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    """Weights of the corners of triangle (a, b, c) at its point closest to ``point`` (Ericson, 5.1.5)."""
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)
    bp = point - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return wp.vec3(1.0 - v, v, 0.0)
    cp = point - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return wp.vec3(1.0 - w, 0.0, w)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return wp.vec3(0.0, 1.0 - w, w)
    denominator = va + vb + vc
    v = vb / denominator
    w = vc / denominator
    return wp.vec3(1.0 - v - w, v, w)


@wp.kernel(enable_backward=False)
def marker_points(
    nodes: wp.array[wp.vec3i], weights: wp.array[wp.vec3], q: wp.array[wp.vec3], points: wp.array[wp.vec3]
):
    """Where every marker's material point is in the state ``q``; launch over the markers in use."""
    i = wp.tid()
    ids, weight = nodes[i], weights[i]
    points[i] = weight[0] * q[ids[0]] + weight[1] * q[ids[1]] + weight[2] * q[ids[2]]


@wp.kernel(enable_backward=False)
def _triangle_distances(
    point: wp.vec3,
    shift: int,
    triangles: wp.array[wp.vec3i],
    q: wp.array[wp.vec3],
    distance: wp.array[float],
    nearest: wp.array[float],
):
    """Squared distance from ``point`` to every triangle of one world, and the smallest; launch over the triangles.

    ``shift`` is the index of the world's first node: its triangles are those of world 0, moved by it.
    """
    t = wp.tid()
    ids = triangles[t] + wp.vec3i(shift, shift, shift)
    a, b, c = q[ids[0]], q[ids[1]], q[ids[2]]
    weight = _triangle_weights(point, a, b, c)
    offset = weight[0] * a + weight[1] * b + weight[2] * c - point
    squared = wp.dot(offset, offset)
    distance[t] = squared
    wp.atomic_min(nearest, 0, squared)


@wp.kernel(enable_backward=False)
def _nearest_triangle(distance: wp.array[float], nearest: wp.array[float], chosen: wp.array[int]):
    """The lowest index among the triangles at the smallest distance; launch over the triangles."""
    t = wp.tid()
    if distance[t] == nearest[0]:
        wp.atomic_min(chosen, 0, t)


@wp.kernel(enable_backward=False)
def _bind_point(
    point: wp.vec3,
    shift: int,
    slot: int,
    color: wp.vec3,
    triangles: wp.array[wp.vec3i],
    q: wp.array[wp.vec3],
    nearest: wp.array[float],
    chosen: wp.array[int],
    nodes: wp.array[wp.vec3i],
    weights: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
):
    """Bind marker ``slot`` to the chosen triangle and leave the scratch ready for the next search; launch once."""
    ids = triangles[chosen[0]] + wp.vec3i(shift, shift, shift)
    nodes[slot] = ids
    weights[slot] = _triangle_weights(point, q[ids[0]], q[ids[1]], q[ids[2]])
    colors[slot] = color
    nearest[0] = _FAR
    chosen[0] = _NO_TRIANGLE


@wp.kernel(enable_backward=False)
def _bind_triangle(
    slot: int,
    ids: wp.vec3i,
    weight: wp.vec3,
    color: wp.vec3,
    nodes: wp.array[wp.vec3i],
    weights: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
):
    """Bind marker ``slot`` to a triangle that the caller knows; launch once."""
    nodes[slot] = ids
    weights[slot] = weight
    colors[slot] = color


@wp.kernel(enable_backward=False)
def _move_marker(
    source: int, target: int, nodes: wp.array[wp.vec3i], weights: wp.array[wp.vec3], colors: wp.array[wp.vec3]
):
    """Move the binding of marker ``source`` to slot ``target``; launch once."""
    nodes[target] = nodes[source]
    weights[target] = weights[source]
    colors[target] = colors[source]


class TissueMarkers:
    """One batch of sphere markers bound to surface triangles of the drawn tissue.

    The slots in use are the first ``count`` rows of the device arrays, in the order the
    markers were attached. Each world has a reserve of ``per_world`` markers; a marker
    beyond it takes the slot of that world's oldest one (``replaced`` counts them), so the
    newest markers of every world are the ones shown and nothing is allocated for them.

    Args:
        viewer: Newton viewer; the batch is logged with ``log_points``.
        name: Name of the batch in the viewer.
        radius: Radius of the spheres [m].
        device: Device of the drawn particle array.
        per_world: Markers that one world can show at a time.
        worlds: Worlds drawn from the one particle array.
        triangles: Surface triangles of the drawn tissue of world 0, ``[T, 3]`` node indices
            into the drawn particle array. Needed only for :meth:`attach_point`.
        world_nodes: Particles of one world in the drawn array: the nodes of world ``w`` are
            those of world 0 plus ``w * world_nodes``. Needed only with several worlds.
    """

    def __init__(self, viewer, name, *, radius, device, per_world, worlds=1, triangles=None, world_nodes=0):
        self.viewer, self.name = viewer, name
        self.device = wp.get_device(device)
        self.per_world, self.worlds = int(per_world), int(worlds)
        if self.per_world < 1 or self.worlds < 1:
            raise ValueError("A marker batch needs a reserve of at least one marker in at least one world")
        self.capacity = self.per_world * self.worlds
        self.radius = float(radius)
        self.nodes = wp.zeros(self.capacity, dtype=wp.vec3i, device=self.device)
        self.weights = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        self.points = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        self.colors = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        self.radii = wp.full(self.capacity, self.radius, dtype=wp.float32, device=self.device)
        self.world_nodes = int(world_nodes)
        self._triangles = None
        if triangles is not None:
            triangles = np.asarray(triangles, dtype=np.int32)
            if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
                raise ValueError("triangles must be [T, 3] node indices of the surface of world 0")
            if self.worlds > 1 and self.world_nodes <= int(triangles.max()):
                raise ValueError("Several worlds need world_nodes, the particles of one world in the drawn array")
            self._triangles = wp.array(triangles, dtype=wp.vec3i, device=self.device)
            self._distance = wp.zeros(len(triangles), dtype=float, device=self.device)
            self._nearest = wp.full(1, _FAR, dtype=float, device=self.device)
            self._chosen = wp.full(1, _NO_TRIANGLE, dtype=int, device=self.device)
        # World and order of attachment of every slot in use (host values).
        self._worlds, self._order, self._attached = [], [], 0
        self.replaced = 0
        # Views of the rows in use; made when their number changes, not in a frame.
        self._shown = None
        load_kernel_modules(self.device, marker_points)

    @property
    def count(self):
        """Markers shown: the rows in use of ``nodes``, ``weights``, ``points`` and ``colors``."""
        return len(self._worlds)

    def _take_slot(self, world):
        """Row of a new marker of ``world``: the next free one, or the row of the world's oldest marker."""
        world = int(world)
        if not 0 <= world < self.worlds:
            raise ValueError(f"World {world} is not one of the {self.worlds} worlds of these markers")
        own = [slot for slot, owner in enumerate(self._worlds) if owner == world]
        if len(own) < self.per_world:
            slot = len(self._worlds)
            self._worlds.append(world)
            self._order.append(self._attached)
            self._show()
        else:
            slot = min(own, key=self._order.__getitem__)
            self._order[slot] = self._attached
            self.replaced += 1
        self._attached += 1
        return slot

    def _show(self):
        count = self.count
        self._shown = (self.points[:count], self.radii[:count], self.colors[:count]) if count else None

    def attach(self, nodes, weights, color, world=0):
        """Bind a new marker to a known surface triangle; returns its row.

        Args:
            nodes: The three node indices of the triangle in the drawn particle array.
            weights: Weights of the three nodes at the material point; they add up to one.
            color: RGB of the sphere, each in [0, 1].
            world: World of the marker.
        """
        ids, weight = wp.vec3i(*(int(node) for node in nodes)), wp.vec3(*(float(value) for value in weights))
        slot = self._take_slot(world)
        wp.launch(
            _bind_triangle,
            1,
            [slot, ids, weight, wp.vec3(*(float(value) for value in color)), self.nodes, self.weights, self.colors],
            device=self.device,
        )
        return slot

    def attach_point(self, particle_q, position, color, world=0):
        """Bind a new marker to the point of the drawn surface that is closest to ``position``; returns its row.

        The search runs on the device over the triangles of ``world`` in the state
        ``particle_q``: the squared distance to every triangle and the smallest of them, the
        lowest triangle index at that distance, then the weights of the closest point in it.
        The position and the colour are kernel arguments; nothing is read back.

        Args:
            particle_q: Drawn particle positions at the moment of the binding [m].
            position: Position in the frame of ``particle_q`` (with the world's offset) [m].
            color: RGB of the sphere, each in [0, 1].
            world: World whose surface the marker belongs to.
        """
        if self._triangles is None:
            raise ValueError("Binding a position needs the surface triangles of the drawn tissue")
        point = wp.vec3(*(float(value) for value in position))
        slot = self._take_slot(world)
        shift, count = int(world) * self.world_nodes, len(self._distance)
        wp.launch(
            _triangle_distances,
            count,
            [point, shift, self._triangles, particle_q, self._distance, self._nearest],
            device=self.device,
        )
        wp.launch(_nearest_triangle, count, [self._distance, self._nearest, self._chosen], device=self.device)
        wp.launch(
            _bind_point,
            1,
            [
                point,
                shift,
                slot,
                wp.vec3(*(float(value) for value in color)),
                self._triangles,
                particle_q,
                self._nearest,
                self._chosen,
                self.nodes,
                self.weights,
                self.colors,
            ],
            device=self.device,
        )
        return slot

    def clear(self, world=None):
        """Remove every marker, or those of one world (a reset of that scene); the others keep their bindings."""
        if world is None:
            self._worlds, self._order = [], []
        else:
            kept = [slot for slot, owner in enumerate(self._worlds) if owner != int(world)]
            for target, source in enumerate(kept):
                if source != target:
                    # Rows move towards the front in rising order: a source is never overwritten before its move.
                    wp.launch(
                        _move_marker, 1, [source, target, self.nodes, self.weights, self.colors], device=self.device
                    )
            self._worlds, self._order = [self._worlds[slot] for slot in kept], [self._order[slot] for slot in kept]
        self._show()

    def draw(self, particle_q):
        """Log the markers where their material points are in ``particle_q``; in every drawn frame.

        One launch over the markers in use and one ``log_points`` with views of the reserve;
        without a marker the batch is hidden.
        """
        if self._shown is None or particle_q is None:
            self.viewer.log_points(self.name, None, hidden=True)
            return
        points, radii, colors = self._shown
        wp.launch(marker_points, len(points), [self.nodes, self.weights, particle_q, self.points], device=self.device)
        self.viewer.log_points(self.name, points, radii=radii, colors=colors, hidden=False)

    def warm(self, particle_q):
        """Issue every launch of these markers once and log the whole reserve hidden; in a start-up draw.

        A frame of the run then launches nothing for the first time, and a viewer that makes
        room for a batch at its first log makes it here, for the whole reserve. Called with no
        marker attached; it leaves none.
        """
        if self.count:
            raise RuntimeError("The start-up draw of the markers comes before the first marker")
        if particle_q is not None:
            black = (0.0, 0.0, 0.0)
            self.attach((0, 0, 0), (1.0, 0.0, 0.0), black)
            if self._triangles is not None:
                self.clear()
                self.attach_point(particle_q, black, black)
            wp.launch(_move_marker, 1, [0, 0, self.nodes, self.weights, self.colors], device=self.device)
            wp.launch(
                marker_points, self.capacity, [self.nodes, self.weights, particle_q, self.points], device=self.device
            )
            self.clear()
            self._attached = self.replaced = 0
        self.viewer.log_points(self.name, self.points, radii=self.radii, colors=self.colors, hidden=True)
