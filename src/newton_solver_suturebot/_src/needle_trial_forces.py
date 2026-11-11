# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Device trial forces with conservative permission and boundary certificates.

Prepare topology/accepted metadata only between physical steps. Device kernels
recompute candidate material association; possible tip boundary events request
transactional CPU fallback instead of approximating moving-triangle CCD.
"""

import numpy as np
import warp as wp

from newton_solver_suturebot._src.geometry.needle_queries import NeedleQueryGeometry, _conservative_box
from newton_solver_suturebot._src.insertion import _shaft_forces, _surface_forces, _surface_positions, _track_positions


@wp.func
def embedding(point: wp.vec3d, ids: wp.vec4i, q: wp.array[wp.vec3]):
    a = wp.vec3d(q[ids[0]])
    e, f, g = wp.vec3d(q[ids[1]]) - a, wp.vec3d(q[ids[2]]) - a, wp.vec3d(q[ids[3]]) - a
    matrix = wp.matrix_from_cols(e, f, g)
    determinant = wp.determinant(matrix)
    uvw = wp.vec3d(wp.float64(-1e30))
    if wp.abs(determinant) > wp.float64(1e-18):
        uvw = wp.inverse(matrix) * (point - a)
    return wp.vec4d(wp.float64(1.0) - uvw[0] - uvw[1] - uvw[2], uvw[0], uvw[1], uvw[2]), determinant


@wp.func
def located(
    point: wp.vec3d,
    q: wp.array[wp.vec3],
    tets: wp.array[wp.vec4i],
    bvh: wp.uint64,
    bounds_low: wp.array[wp.vec3],
    bounds_high: wp.array[wp.vec3],
):
    eps = wp.vec3d(wp.float64(1e-8))
    low, high = _conservative_box(point - eps, point + eps)
    query = wp.bvh_query_aabb(bvh, low, high)
    item, best = int(0), int(-1)
    score = wp.float64(-1e30)
    chosen = wp.vec4d(wp.float64(0))
    while wp.bvh_query_next(query, item):
        ids = tets[item]
        # The current-trial extrema are selected from float32 vertex values.
        # Casting those extrema to double is exact; retain double padding/tests.
        lo, hi = wp.vec3d(bounds_low[item]), wp.vec3d(bounds_high[item])
        keep = bool(True)
        for axis in range(3):
            if point[axis] < lo[axis] - wp.float64(1e-8) or point[axis] > hi[axis] + wp.float64(1e-8):
                keep = False
        if keep:
            bary, det = embedding(point, ids, q)
            minimum = wp.min(wp.min(bary[0], bary[1]), wp.min(bary[2], bary[3]))
            if wp.abs(det) > wp.float64(1e-18) and minimum >= wp.float64(-1e-6):
                if minimum > score or (minimum == score and (best < 0 or item < best)):
                    best, score, chosen = item, minimum, bary
    return best, chosen


@wp.func
def static_crossing(start: wp.vec3d, end: wp.vec3d, q: wp.array[wp.vec3], faces: wp.array[wp.vec3i], mesh: wp.uint64):
    query_low = wp.min(start, end) - wp.vec3d(wp.float64(1e-9))
    query_high = wp.max(start, end) + wp.vec3d(wp.float64(1e-9))
    low, high = _conservative_box(query_low, query_high)
    query = wp.mesh_query_aabb(mesh, low, high)
    item = int(0)
    hit = bool(False)
    while wp.mesh_query_aabb_next(query, item):
        ids = faces[item]
        a, b, c = wp.vec3d(q[ids[0]]), wp.vec3d(q[ids[1]]), wp.vec3d(q[ids[2]])
        lo, hi = wp.min(a, wp.min(b, c)), wp.max(a, wp.max(b, c))
        keep = bool(True)
        for axis in range(3):
            if hi[axis] < query_low[axis] or lo[axis] > query_high[axis]:
                keep = False
        if keep:
            r, dr, e, f = start - a, end - start, b - a, c - a
            normal = wp.cross(e, f)
            linear, constant = wp.dot(dr, normal), wp.dot(r, normal)
            scale = wp.max(wp.max(wp.abs(linear), wp.abs(constant)), wp.float64(1e-30))
            if wp.abs(linear) > wp.float64(1e-12) * scale:
                time = -constant / linear
                if time > wp.float64(1e-7) and time <= wp.float64(1.0 + 1e-7):
                    time = wp.clamp(time, wp.float64(0), wp.float64(1))
                    relative = r + time * dr
                    ee, ef, ff = wp.dot(e, e), wp.dot(e, f), wp.dot(f, f)
                    determinant = ee * ff - ef * ef
                    if determinant > wp.float64(1e-28):
                        re, rf = wp.dot(relative, e), wp.dot(relative, f)
                        u, v = (ff * re - ef * rf) / determinant, (ee * rf - ef * re) / determinant
                        if wp.min(wp.min(u, v), wp.float64(1) - u - v) >= wp.float64(-1e-6):
                            hit = True
    return hit


@wp.kernel
def sample_positions(local: wp.array[wp.vec3d], body_q: wp.array[wp.transform], body: int, world: wp.array[wp.vec3d]):
    i = wp.tid()
    pose = body_q[body]
    # Match the CPU oracle's float32 quaternion matrix, then double local coordinates.
    rotation = wp.mat33d(wp.quat_to_matrix(wp.transform_get_rotation(pose)))
    world[i] = rotation * local[i] + wp.vec3d(wp.transform_get_translation(pose))


@wp.kernel
def track_segments(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4d],
    q: wp.array[wp.vec3],
    samples: wp.array[wp.vec3d],
    arc: wp.array[wp.float64],
    points: wp.array[wp.vec3d],
    segments: wp.array2d[wp.vec2d],
):
    # Independent projections expose parallelism even with only a few anchors.
    # Repeat the same four-term sum to keep the original double arithmetic.
    i, j = wp.tid()
    point = wp.vec3d(wp.float64(0))
    for k in range(4):
        point += bary[i][k] * wp.vec3d(q[nodes[i][k]])
    if j == 0:
        points[i] = point
    direction = samples[j + 1] - samples[j]
    length = arc[j + 1] - arc[j]
    fraction = wp.dot(point - samples[j], direction) / (length * length)
    delta = samples[j] + wp.clamp(fraction, wp.float64(0), wp.float64(1)) * direction - point
    segments[i, j] = wp.vec2d(wp.dot(delta, delta), fraction)


@wp.kernel
def track_geometry(
    segments: wp.array2d[wp.vec2d],
    arc: wp.array[wp.float64],
    engaged: wp.array[int],
    projected_arc: wp.array[wp.float64],
    associated: wp.array[int],
):
    i = wp.tid()
    best = int(0)
    best_fraction = wp.float64(0)
    distance = wp.float64(1e30)
    # Preserve the ascending, strict first-min rule, including endpoint ties.
    for j in range(segments.shape[1]):
        pair = segments[i, j]
        if pair[0] < distance:
            best, best_fraction, distance = j, pair[1], pair[0]
    projected_arc[i] = arc[best] + wp.clamp(best_fraction, wp.float64(0), wp.float64(1)) * (arc[best + 1] - arc[best])
    covered = not (
        (best == 0 and best_fraction < wp.float64(0))
        or (best == segments.shape[1] - 1 and best_fraction > wp.float64(1))
    )
    associated[i] = int(engaged[i] != 0 and covered)


@wp.kernel
def shaft_mask(
    q: wp.array[wp.vec3],
    tets: wp.array[wp.vec4i],
    visited: wp.array[int],
    fixed: wp.array[int],
    bvh: wp.uint64,
    bounds_low: wp.array[wp.vec3],
    bounds_high: wp.array[wp.vec3],
    active_mesh: wp.uint64,
    active_faces: wp.array[wp.vec3i],
    samples: wp.array[wp.vec3d],
    arc: wp.array[wp.float64],
    track: wp.array[wp.vec3d],
    projected_arc: wp.array[wp.float64],
    associated: wp.array[int],
    capture: wp.float64,
    mask: wp.array[int],
    fallback: wp.array[int],
):
    i = wp.tid()
    point = samples[i]
    value = int(0)
    if visited.shape[0] == 0:
        mask[i] = 0
        return
    for k in range(visited.shape[0]):
        ids = tets[visited[k]]
        lo, hi = wp.vec3d(bounds_low[visited[k]]), wp.vec3d(bounds_high[visited[k]])
        keep = bool(True)
        for axis in range(3):
            if point[axis] < lo[axis] - wp.float64(1e-8) or point[axis] > hi[axis] + wp.float64(1e-8):
                keep = False
        if keep:
            bary, det = embedding(point, ids, q)
            if wp.abs(det) <= wp.float64(1e-18):
                wp.atomic_max(fallback, 0, 1)
            elif wp.min(wp.min(bary[1], bary[2]), bary[3]) >= wp.float64(-1e-5) and bary[1] + bary[2] + bary[
                3
            ] <= wp.float64(1.0 + 1e-5):
                value = 1
    near, possible = bool(False), bool(False)
    for anchor in range(track.shape[0]):
        close = wp.length(track[anchor] - point) <= capture
        linked = associated[anchor] != 0 and wp.abs(projected_arc[anchor] - arc[i]) <= capture
        near = near or close
        possible = possible or close or linked
    if i > 0 and not (value != 0 and near) and possible:
        tet, bary = located(point, q, tets, bvh, bounds_low, bounds_high)
        supported = bool(False)
        if tet >= 0:
            ids = tets[tet]
            for j in range(4):
                if fixed[ids[j]] != 0 and bary[j] > wp.float64(0.05):
                    supported = True
        if tet >= 0 and not supported:
            best = wp.float64(1e30)
            for anchor in range(track.shape[0]):
                distance = wp.length(track[anchor] - point)
                linked = associated[anchor] != 0 and wp.abs(projected_arc[anchor] - arc[i]) <= capture
                if (distance <= capture or linked) and distance < best:
                    if not static_crossing(track[anchor], point, q, active_faces, active_mesh):
                        best = distance
                        value = 1
                        if linked:
                            value = 4
    mask[i] = value


@wp.kernel
def tip_location(
    points: wp.array[wp.vec3d],
    q: wp.array[wp.vec3],
    tets: wp.array[wp.vec4i],
    bvh: wp.uint64,
    low: wp.array[wp.vec3],
    high: wp.array[wp.vec3],
    result: wp.array[int],
    bary: wp.array[wp.vec4d],
):
    cell, weights = located(points[0], q, tets, bvh, low, high)
    result[0] = cell
    bary[0] = weights


@wp.kernel
def tip_material_certificate(
    result: wp.array[int],
    barycentric: wp.array[wp.vec4d],
    tets: wp.array[wp.vec4i],
    fixed: wp.array[int],
    step_flags: wp.array[int],
    spacing: wp.float64,
    capture: wp.float64,
    mask: wp.array[int],
    fallback: wp.array[int],
):
    tet, bary = result[0], barycentric[0]
    in_material, punctured, has_previous = step_flags[0], step_flags[1], step_flags[2]
    if has_previous == 0:
        wp.atomic_max(fallback, 0, 1)
    if in_material:
        if punctured == 0 or tet < 0:
            wp.atomic_max(fallback, 0, 1)
        if tet >= 0:
            ids = tets[tet]
            for j in range(4):
                if fixed[ids[j]] != 0 and bary[j] > wp.float64(0.05):
                    wp.atomic_max(fallback, 0, 1)
    elif punctured != 0 or tet >= 0:
        wp.atomic_max(fallback, 0, 1)
    if in_material and (mask[0] == 0 or spacing > capture):
        mask[0] = 2


@wp.kernel
def tip_boundary_certificate(
    full_mesh: wp.uint64,
    active_mesh: wp.uint64,
    samples: wp.array[wp.vec3d],
    old_tips: wp.array[wp.vec3d],
    motion: wp.array[wp.float64],
    step_flags: wp.array[int],
    radius: wp.float64,
    fallback: wp.array[int],
):
    # The three queries are independent disjunctions into the fallback flag;
    # one lane each overlaps their sequential bounding-volume walks.
    lane = wp.tid()
    old_tip, tip = old_tips[0], samples[0]
    item = int(0)
    if lane == 0:
        padding = wp.vec3d(motion[0], motion[1], motion[2]) + wp.vec3d(wp.float64(1e-9))
        low, high = _conservative_box(wp.min(old_tip, tip) - padding, wp.max(old_tip, tip) + padding)
        query = wp.mesh_query_aabb(full_mesh, low, high)
        while wp.mesh_query_aabb_next(query, item):
            wp.atomic_max(fallback, 0, 1)
    elif lane == 1:
        padding = wp.vec3d(motion[0], motion[1], motion[2]) + wp.vec3d(wp.float64(1e-9))
        low, high = _conservative_box(wp.min(old_tip, tip) - padding, wp.max(old_tip, tip) + padding)
        query2 = wp.mesh_query_aabb(active_mesh, low, high)
        while wp.mesh_query_aabb_next(query2, item):
            wp.atomic_max(fallback, 0, 1)
    elif not step_flags[0]:
        lo, hi = _conservative_box(tip - wp.vec3d(radius), tip + wp.vec3d(radius))
        close = wp.mesh_query_aabb(full_mesh, lo, hi)
        while wp.mesh_query_aabb_next(close, item):
            wp.atomic_max(fallback, 0, 1)


@wp.kernel
def opening_positions(
    nodes: wp.array[wp.vec3i], bary: wp.array[wp.vec3d], q: wp.array[wp.vec3], points: wp.array[wp.vec3]
):
    i = wp.tid()
    point = wp.vec3d(wp.float64(0))
    for j in range(3):
        point += bary[i][j] * wp.vec3d(q[nodes[i][j]])
    points[i] = wp.vec3(point)


@wp.kernel
def force_flags(
    contact: wp.array[int], diagnostics: wp.array[wp.vec3], step_flags: wp.array[int], fallback: wp.array[int]
):
    if contact[1] != 0 or (step_flags[0] == 0 and diagnostics[6][0] > 0.0):
        wp.atomic_max(fallback, 0, 1)
    for i in range(diagnostics.shape[0]):
        if not wp.isfinite(diagnostics[i][0] + diagnostics[i][1] + diagnostics[i][2]):
            wp.atomic_max(fallback, 0, 1)


class _NeedleTrialForces:
    """Prepare accepted metadata on host; evaluate finite trial states on device.

    Geometry/topology belongs to one fixed interaction; rebuild this helper when
    replacing that geometry. Buffer identities remain stable until accepted
    visited/track/opening counts change.
    The caller must exclude nonfinite candidate positions before mesh refits.
    ``fallback`` requests rollback and the complete original CPU step path.
    """

    def __init__(self, interaction, *, query_block_dim=None):
        self.interaction = interaction
        self.device = interaction.device
        self.shaft_block_dim = 64 if query_block_dim is None else query_block_dim
        self.surface_block_dim = 32 if query_block_dim is None else query_block_dim
        self.local = self._array(interaction.samples, wp.vec3d)
        self.arc = self._array(interaction._sample_arc, wp.float64)
        self.tets = self._array(interaction.tetrahedra, wp.vec4i)
        self.faces = self._array(interaction._active_triangles, wp.vec3i)
        self.world = wp.empty(len(interaction.samples), dtype=wp.vec3d, device=self.device)
        self.mask = wp.empty(len(interaction.samples), dtype=int, device=self.device)
        self.fallback = wp.zeros(1, dtype=int, device=self.device)
        self.tip_cell = wp.empty(1, dtype=int, device=self.device)
        self.tip_bary = wp.empty(1, dtype=wp.vec4d, device=self.device)
        self.old_tip = wp.zeros(1, dtype=wp.vec3d, device=self.device)
        self.step_flags = wp.zeros(3, dtype=int, device=self.device)
        self._counts = None
        self.prepare(interaction)
        if self.device.is_cuda:
            from .geometry.mesh_refit import GraphMeshRefit

            # Vertices that trials can move: mapped exterior vertices sourced
            # from free nodes, and free nodes of the active-boundary copy.
            # Fixed nodes are restored to their references before every solve,
            # so faces without any movable vertex keep identical bounds.
            fixed = np.asarray(interaction.fixed).astype(bool)
            mapping = np.asarray(interaction.mapping)
            surface_moving = np.zeros(len(interaction.mesh.points), dtype=bool)
            mapped = mapping >= 0
            surface_moving[mapped] = ~fixed[mapping[mapped]]
            active_moving = np.zeros(len(interaction._active_mesh.points), dtype=bool)
            active_moving[: len(fixed)] = ~fixed
            self._surface_refit = GraphMeshRefit(interaction.mesh, moving_vertex_mask=surface_moving)
            self._active_refit = GraphMeshRefit(interaction._active_mesh, moving_vertex_mask=active_moving)
        else:
            self._surface_refit = interaction.mesh
            self._active_refit = interaction._active_mesh

    def _array(self, values, dtype):
        return wp.array(np.asarray(values), dtype=dtype, device=self.device)

    def prepare(self, interaction):
        """Upload accepted metadata; return whether captured buffer storage changed."""
        if interaction is not self.interaction:
            raise ValueError("Construct new permissions for a different interaction")
        x = interaction
        counts = (len(x._visited_tets), x.track_count, len(x._openings))
        changed = counts != self._counts
        if self._counts is None or counts[0] != self._counts[0]:
            self.visited = self._array(sorted(x._visited_tets), int)
        if self._counts is None or counts[1] != self._counts[1]:
            self.nodes = self._array(x._track_nodes_host, wp.vec4i)
            self.bary = self._array(x._track_bary_host, wp.vec4d)
            self.points = wp.empty(x.track_count, dtype=wp.vec3d, device=self.device)
            self.segments = wp.empty((x.track_count, len(x.samples) - 1), dtype=wp.vec2d, device=self.device)
            self.projected_arc = wp.empty(x.track_count, dtype=wp.float64, device=self.device)
            self.associated = wp.empty(x.track_count, dtype=int, device=self.device)
        if self._counts is None or counts[2] != self._counts[2]:
            self.opening_nodes = self._array([ids for ids, _ in x._openings], wp.vec3i)
            self.opening_bary = self._array([bary for _, bary in x._openings], wp.vec3d)
            if len(x._opening_points) != len(x._openings):
                x._opening_points = wp.empty(len(x._openings), dtype=wp.vec3, device=self.device)
        self._counts = counts
        old_tip = x._previous_tip if x._previous_tip is not None else np.zeros(3)
        self.old_tip.assign(np.asarray([old_tip], dtype=np.float64))
        self.step_flags.assign(np.asarray([x._in_material, x.punctured, x._previous_tip is not None], dtype=np.int32))
        if x._device_queries is None:
            x._device_queries = NeedleQueryGeometry(x.tetrahedra, len(x.vertices), device=self.device)
        query = x._device_queries
        if query.bvh is None or not query.has_previous:
            accepted = x.vertices if x._previous_q is None else x._previous_q
            positions = self._array(accepted, wp.vec3)
            query.commit(positions)
            query.update(positions)
            changed = True
        return changed

    def signature(self):
        """Captured settings/storage; mutable device contents do not invalidate graphs.

        The caller additionally keys candidate/input/output states, body/COM, dt,
        and its solver/preconditioner buffers. Keep those graph input arrays alive.
        Geometry replacement requires a new helper, rather than reusing its copied
        float64 reference samples and material topology.
        """
        x, query = self.interaction, self.interaction._device_queries
        arrays = [
            self.local,
            self.arc,
            self.tets,
            self.faces,
            self.world,
            self.mask,
            self.fallback,
            self.tip_cell,
            self.tip_bary,
            self.old_tip,
            self.step_flags,
            self.visited,
            self.nodes,
            self.bary,
            self.points,
            self.segments,
            self.projected_arc,
            self.associated,
            self.opening_nodes,
            self.opening_bary,
            query.low,
            query.high,
            query.previous,
            query.motion,
        ]
        arrays.extend(
            getattr(x, name)
            for name in (
                "_samples",
                "_radii",
                "_weights",
                "_fixed",
                "_mapping",
                "_triangles",
                "_surface",
                "_active_surface",
                "_opening_points",
                "_all_opening_faces",
                "_track_nodes",
                "_track_bary",
                "_track_lengths",
                "_track_engaged",
                "_track_points",
                "_particle_inverse_mass",
                "_diagnostics",
                "_contact_flags",
            )
        )
        alm = x.tunnel_alm
        alm_settings = None
        if alm is not None:
            alm_settings = (id(alm), alm.penalty_per_m, alm.compliance_per_m, alm.clearance, alm.tolerance)
            arrays.extend((alm.normal, alm.tangent, alm.proposed_normal, alm.proposed_tangent, alm._statistics))
        return (
            self._counts,
            self.shaft_block_dim,
            self.surface_block_dim,
            x.mesh.id,
            x._active_mesh.id,
            query.bvh.id,
            x.stiffness,
            x.damping,
            x.friction,
            x.speed_scale,
            x.shaft_stiffness_per_m,
            x.shaft_damping_per_m,
            x.track_spacing,
            float(x.radii.max()),
            x._rigid_inverse_mass,
            tuple(np.asarray(x._rigid_inverse_inertia).reshape(-1)),
            alm_settings,
            tuple((array.ptr, array.shape, array.dtype) for array in arrays),
        )

    def evaluate(self, state, body, *, certify_tip=True):
        x = self.interaction
        query = x._device_queries
        capture = 2 * float(x.radii.max())
        self.fallback.zero_()
        wp.launch(sample_positions, len(x.samples), [self.local, state.body_q, body, self.world], device=self.device)
        if x.track_count:
            wp.launch(
                track_segments,
                self.segments.shape,
                [
                    self.nodes,
                    self.bary,
                    state.particle_q,
                    self.world,
                    self.arc,
                    self.points,
                    self.segments,
                ],
                device=self.device,
            )
            wp.launch(
                track_geometry,
                x.track_count,
                [self.segments, self.arc, x._track_engaged, self.projected_arc, self.associated],
                device=self.device,
            )
        wp.launch(
            shaft_mask,
            len(x.samples),
            [
                state.particle_q,
                self.tets,
                self.visited,
                x._fixed,
                query.bvh.id,
                query.low,
                query.high,
                x._active_mesh.id,
                self.faces,
                self.world,
                self.arc,
                self.points,
                self.projected_arc,
                self.associated,
                capture,
                self.mask,
                self.fallback,
            ],
            device=self.device,
            block_dim=self.shaft_block_dim,
        )
        if certify_tip:
            # Keep the exact locator, material checks and boundary queries, but
            # avoid one large single-thread kernel for these independent stages.
            wp.launch(
                tip_location,
                1,
                [
                    self.world,
                    state.particle_q,
                    self.tets,
                    query.bvh.id,
                    query.low,
                    query.high,
                    self.tip_cell,
                    self.tip_bary,
                ],
                device=self.device,
            )
            wp.launch(
                tip_material_certificate,
                1,
                [
                    self.tip_cell,
                    self.tip_bary,
                    self.tets,
                    x._fixed,
                    self.step_flags,
                    x.track_spacing,
                    capture,
                    self.mask,
                    self.fallback,
                ],
                device=self.device,
            )
            wp.launch(
                tip_boundary_certificate,
                3,
                [
                    x.mesh.id,
                    x._active_mesh.id,
                    self.world,
                    self.old_tip,
                    query.motion,
                    self.step_flags,
                    float(x.radii[0]),
                    self.fallback,
                ],
                device=self.device,
            )

    def add_forces(self, state, body, body_com_local, external_forces, dt):
        """Graph-ready trial forces. No history/status publishing or host readback.

        Caller clears external_forces/body_f and has ruled out nonfinite states.
        Before returning to the CPU query path, invalidate its cached position key
        (interaction._device_query_positions = None): this method refits device
        geometry without downloading coordinates or publishing host query caches.
        After a zero fallback result and convergence, run the existing CPU final
        metadata/CCD/accept transaction before publishing the physical state.
        """
        self.add_surface_forces(state, body, body_com_local, external_forces, dt)
        self.add_history_forces(state, body, body_com_local, external_forces, dt)

    def add_surface_forces(self, state, body, body_com_local, external_forces, dt):
        """Candidate-state geometry, permissions and surface forces only.

        This stage depends solely on the candidate positions/velocities and
        accepted metadata — not on trial-local tunnel multipliers. A dual-only
        repetition of an unchanged candidate may reuse its outputs verbatim.
        """
        x = self.interaction
        capture = 2 * float(x.radii.max())
        x._diagnostics.zero_()
        x._contact_flags.zero_()
        wp.launch(_surface_positions, len(x.mapping), [state.particle_q, x._mapping, x._surface], device=self.device)
        self._surface_refit.refit()
        wp.copy(x._active_surface, state.particle_q, count=len(x.vertices))
        self._active_refit.refit()
        x._device_queries.update(state.particle_q)
        if x.track_count:
            wp.launch(
                _track_positions,
                x.track_count,
                [state.particle_q, x._track_nodes, x._track_bary, x._track_points],
                device=self.device,
            )
        if self._counts[2]:
            wp.launch(
                opening_positions,
                self._counts[2],
                [self.opening_nodes, self.opening_bary, state.particle_q, x._opening_points],
                device=self.device,
            )
        self.evaluate(state, body)
        wp.launch(
            _surface_forces,
            len(x.samples),
            [
                x.mesh.id,
                x._active_mesh.id,
                x._triangles,
                x._surface,
                x._mapping,
                x._fixed,
                state.particle_qd,
                x._samples,
                x._radii,
                x._weights,
                state.body_q,
                state.body_qd,
                wp.vec3(*body_com_local),
                body,
                x.stiffness,
                x.damping,
                x.friction,
                x.speed_scale,
                dt,
                0,
                x._rigid_inverse_mass,
                x._rigid_inverse_inertia,
                x._particle_inverse_mass,
                x._opening_points,
                x._all_opening_faces,
                capture,
                self.mask,
                wp.vec3(0.0),
                x._track_points,
                x.track_count,
                capture,
                external_forces,
                state.body_f,
                x._diagnostics,
                x._contact_flags,
            ],
            device=self.device,
            block_dim=self.surface_block_dim,
        )

    def add_history_forces(self, state, body, body_com_local, external_forces, dt):
        """Tunnel/shaft track forces and the trial flag check.

        The ALM branch reads the trial-local dual multipliers, so this stage is
        re-evaluated after every dual update even for an unchanged candidate.
        """
        x = self.interaction
        capture = 2 * float(x.radii.max())
        if x.tunnel_alm is not None:
            x.tunnel_alm.add_forces(state, body, body_com_local, external_forces, dt)
        elif x.track_count:
            wp.launch(
                _shaft_forces,
                x.track_count,
                [
                    x._track_nodes,
                    x._track_bary,
                    x._track_lengths,
                    x._track_engaged,
                    0,
                    state.particle_q,
                    state.particle_qd,
                    x._fixed,
                    x._samples,
                    state.body_q,
                    state.body_qd,
                    wp.vec3(*body_com_local),
                    body,
                    x.shaft_stiffness_per_m,
                    x.shaft_damping_per_m,
                    x.friction,
                    x.speed_scale,
                    capture,
                    dt,
                    0,
                    x._rigid_inverse_mass,
                    x._rigid_inverse_inertia,
                    x._particle_inverse_mass,
                    external_forces,
                    state.body_f,
                    x._diagnostics,
                ],
                device=self.device,
            )
        wp.launch(
            force_flags, 1, [x._contact_flags, x._diagnostics, self.step_flags, self.fallback], device=self.device
        )
