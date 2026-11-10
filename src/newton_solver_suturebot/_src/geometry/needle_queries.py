# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Conservative device broad phase; needle narrow-phase predicates stay in float64."""

import numpy as np
import warp as wp


@wp.kernel
def _tet_bounds(q: wp.array[wp.vec3], tets: wp.array[wp.vec4i], low: wp.array[wp.vec3], high: wp.array[wp.vec3]):
    i = wp.tid()
    ids = tets[i]
    lo, hi = wp.vec3(q[ids[0]]), wp.vec3(q[ids[0]])
    for j in range(1, 4):
        lo = wp.min(lo, q[ids[j]])
        hi = wp.max(hi, q[ids[j]])
    low[i] = lo
    high[i] = hi


@wp.kernel
def _maximum_motion(q: wp.array[wp.vec3], previous: wp.array[wp.vec3], motion: wp.array[wp.float64]):
    i = wp.tid()
    a, b = wp.vec3d(q[i]), wp.vec3d(previous[i])
    for axis in range(3):
        distance = wp.abs(a[axis] - b[axis])
        if distance > wp.float64(0.0):
            wp.atomic_max(motion, axis, distance)


@wp.func
def _conservative_box(low: wp.vec3d, high: wp.vec3d):
    # Float32 BVH boxes/queries may round inward. Expand by four relative
    # float32 epsilons, then retain the original float64 predicate on the host.
    magnitude = wp.max(wp.abs(low), wp.abs(high))
    margin = magnitude * wp.float64(4.76837158203125e-7) + wp.vec3d(wp.float64(1e-12))
    return wp.vec3(low - margin), wp.vec3(high + margin)


@wp.kernel
def _tet_candidates(bvh: wp.uint64, points: wp.array[wp.vec3d], result: wp.array2d[int]):
    i = wp.tid()
    radius = wp.vec3d(wp.float64(1e-8))
    low, high = _conservative_box(points[i] - radius, points[i] + radius)
    query = wp.bvh_query_aabb(bvh, low, high)
    item, count = int(0), int(0)
    while wp.bvh_query_next(query, item):
        if count + 1 < result.shape[1]:
            result[i, count + 1] = item
        count += 1
    result[i, 0] = count


@wp.kernel
def _surface_candidates(
    mesh: wp.uint64,
    low: wp.vec3d,
    high: wp.vec3d,
    motion: wp.array[wp.float64],
    swept: int,
    result: wp.array[int],
):
    padding = wp.vec3d()
    if swept:
        padding = wp.vec3d(motion[0], motion[1], motion[2])
    lo, hi = _conservative_box(low - padding, high + padding)
    query = wp.mesh_query_aabb(mesh, lo, hi)
    item, count = int(0), int(0)
    while wp.mesh_query_aabb_next(query, item):
        if count + 1 < result.shape[0]:
            result[count + 1] = item
        count += 1
    result[0] = count


class NeedleQueryGeometry:
    """Persistent GPU broad phase, with explicit overflow and accepted-state ownership.

    Return sorted candidate IDs or ``None`` on packet overflow. Callers then
    use their original complete CPU query. No candidate truncation is allowed.
    Existing full/active surface meshes remain owned/refitted by the interaction.
    """

    def __init__(self, tetrahedra, particle_count, *, device, capacity=128):
        self.device = wp.get_device(device)
        self.capacity = int(capacity)
        self.tetrahedra = wp.array(tetrahedra, dtype=wp.vec4i, device=self.device)
        self.low = wp.empty(len(tetrahedra), dtype=wp.vec3, device=self.device)
        self.high = wp.empty_like(self.low)
        self.previous = wp.empty(particle_count, dtype=wp.vec3, device=self.device)
        self.motion = wp.zeros(3, dtype=wp.float64, device=self.device)
        self.bvh = None
        self.has_previous = False
        self.points = None
        self.point_results = None
        self.surface_result = wp.empty(self.capacity + 1, dtype=int, device=self.device)

    def update(self, positions):
        wp.launch(
            _tet_bounds, len(self.tetrahedra), [positions, self.tetrahedra, self.low, self.high], device=self.device
        )
        if self.bvh is None:
            self.bvh = wp.Bvh(self.low, self.high)
        else:
            self.bvh.refit()
        self.motion.zero_()
        if self.has_previous:
            wp.launch(_maximum_motion, len(self.previous), [positions, self.previous, self.motion], device=self.device)

    def commit(self, positions):
        wp.copy(self.previous, positions, count=len(self.previous))
        self.has_previous = True

    def point_candidates(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self.points is None or len(self.points) != len(points):
            self.points = wp.empty(len(points), dtype=wp.vec3d, device=self.device)
            self.point_results = wp.empty((len(points), self.capacity + 1), dtype=int, device=self.device)
        self.points.assign(points)
        wp.launch(_tet_candidates, len(points), [self.bvh.id, self.points, self.point_results], device=self.device)
        output = self.point_results.numpy()
        return [None if row[0] > self.capacity else np.sort(row[1 : row[0] + 1]) for row in output]

    def surface_candidates(self, mesh, low, high, *, swept=False):
        wp.launch(
            _surface_candidates,
            1,
            [mesh.id, wp.vec3d(*low), wp.vec3d(*high), self.motion, int(swept), self.surface_result],
            device=self.device,
        )
        output = self.surface_result.numpy()
        return None if output[0] > self.capacity else np.sort(output[1 : output[0] + 1])
