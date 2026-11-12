# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simultaneous thread contacts with capsules on independently moving rigid bodies.

This extends the existing SolverThread normal penalty and lagged Coulomb law;
there is no post-step position projection. Each (body, radius) group contributes
one nearest chord/capsule contact per element. Distinct jaws therefore retain
simultaneous contacts. Detection freezes the straight element chord for a step,
while forces act on its cubic ANCF centreline. This is a discrete, compliant
contact approximation, not continuous collision detection or exact exclusion.

Friction anchors move with their owning body. Fresh reaction wrenches exactly
oppose the thread forces at their ANCF application points. The next rigid step
receives a relaxed, one-step-late wrench, as in ThreadCapsuleContact; instantaneous
force balance applies to the fresh pair, not the asynchronously applied loads.
"""

import numpy as np
import warp as wp

from .solvers.thread.kernels import (
    _centerline_point,
    _friction_energy,
    _friction_scale,
    _node_vector,
    _plane_offset,
    _shape,
    closest_on_segments,
    contact_terms,
)


@wp.func
def _rotation(pose: wp.transform) -> wp.mat33d:
    return wp.mat33d(wp.quat_to_matrix(wp.transform_get_rotation(pose)))


@wp.func
def _world(pose: wp.transform, point: wp.vec3d) -> wp.vec3d:
    return wp.vec3d(wp.transform_get_translation(pose)) + _rotation(pose) @ point


@wp.func
def _local(pose: wp.transform, point: wp.vec3d) -> wp.vec3d:
    return wp.transpose(_rotation(pose)) @ (point - wp.vec3d(wp.transform_get_translation(pose)))


@wp.kernel(enable_backward=False)
def _find(
    q: wp.array[wp.float64],
    qd: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    previous_pose: wp.array[wp.transform],
    bodies: wp.array[int],
    offsets: wp.array[int],
    endpoints: wp.array2d[wp.vec3d],
    distance: wp.array[wp.float64],
    skip: wp.array[int],
    smoothing: wp.float64,
    dt: wp.float64,
    xi: wp.array2d[wp.float64],
    point: wp.array2d[wp.vec3d],
    normal: wp.array2d[wp.vec3d],
    limit: wp.array2d[wp.float64],
    local_anchor: wp.array2d[wp.vec3d],
    anchor: wp.array2d[wp.vec3d],
):
    group, element = wp.tid()
    previous_normal = normal[group, element]
    normal[group, element] = wp.vec3d()
    if element < skip[group]:
        limit[group, element] = wp.float64(0.0)
        return
    body = bodies[group]
    pose = body_q[body]
    a0, a1 = _node_vector(q, element, 0), _node_vector(q, element + 1, 0)
    speed = wp.max(wp.length(_node_vector(qd, element, 0)), wp.length(_node_vector(qd, element + 1, 0)))
    # Body poses are already accepted. Their measured velocity only expands
    # the candidate margin; it never supplies a prescribed thread velocity.
    speed += wp.length(wp.vec3d(wp.spatial_top(body_qd[body])))
    nearest = distance[group] + wp.float64(10.0) * smoothing + dt * speed
    chosen_xi = xi[group, element]
    chosen_point, chosen_normal = wp.vec3d(), wp.vec3d()
    for capsule in range(offsets[group], offsets[group + 1]):
        b0, b1 = _world(pose, endpoints[capsule, 0]), _world(pose, endpoints[capsule, 1])
        fractions = closest_on_segments(a0, a1, b0, b1)
        here = a0 + fractions[0] * (a1 - a0)
        there = b0 + fractions[1] * (b1 - b0)
        gap = wp.length(here - there)
        if gap < nearest:
            nearest, chosen_xi, chosen_point = gap, fractions[0], there
            if gap > wp.float64(1.0e-14):
                chosen_normal = (here - there) / gap
            elif wp.length(previous_normal) > wp.float64(0.0):
                chosen_normal = previous_normal
            else:
                axis = wp.normalize(b1 - b0)
                basis = wp.vec3d(1.0, 0.0, 0.0)
                if wp.abs(axis[0]) > wp.float64(0.9):
                    basis = wp.vec3d(0.0, 1.0, 0.0)
                chosen_normal = wp.normalize(wp.cross(axis, basis))
    if wp.length(chosen_normal) == wp.float64(0.0):
        limit[group, element] = wp.float64(0.0)
        return
    position = _centerline_point(q, element, _shape(chosen_xi, lengths[element]))
    if limit[group, element] > wp.float64(0.0):
        old_position = _centerline_point(q, element, _shape(xi[group, element], lengths[element]))
        # Carry the old elastic slip with the rigid surface. If the closest
        # material coordinate changes, transfer the same local slip across
        # this element instead of mistaking that change for tangential motion.
        shift = wp.transpose(_rotation(previous_pose[body])) @ (position - old_position)
        anchor[group, element] = _world(pose, local_anchor[group, element] + shift)
    else:
        anchor[group, element] = position
    xi[group, element] = chosen_xi
    point[group, element] = chosen_point
    normal[group, element] = chosen_normal


@wp.kernel(enable_backward=False)
def _gradient(
    q: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    xi: wp.array2d[wp.float64],
    point: wp.array2d[wp.vec3d],
    normal: wp.array2d[wp.vec3d],
    distance: wp.array[wp.float64],
    anchor: wp.array2d[wp.vec3d],
    limit: wp.array2d[wp.float64],
    stiffness: wp.float64,
    smoothing: wp.float64,
    tangent_blocks: bool,
    status: wp.array[int],
    gradient: wp.array2d[wp.vec3d],
    hessian: wp.array3d[wp.mat33d],
):
    group, element = wp.tid()
    direction = normal[group, element]
    if status[0] != 0 or wp.length(direction) == wp.float64(0.0):
        return
    shape = _shape(xi[group, element], lengths[element])
    position = _centerline_point(q, element, shape)
    terms = contact_terms(distance[group], wp.dot(direction, position - point[group, element]), stiffness, smoothing)
    offset = _plane_offset(position, anchor[group, element], direction)
    scale = wp.float64(0.0)
    if limit[group, element] > wp.float64(0.0):
        scale = _friction_scale(wp.length(offset), stiffness, limit[group, element])
    force = terms[1] * direction + scale * offset
    tangent = terms[2] * wp.outer(direction, direction)
    tangent += scale * (wp.identity(n=3, dtype=wp.float64) - wp.outer(direction, direction))
    for b in range(4):
        wp.atomic_add(gradient, element, b, shape[b] * force)
        if tangent_blocks:
            for c in range(4):
                wp.atomic_add(hessian, element, b, c, (shape[b] * shape[c]) * tangent)


@wp.kernel(enable_backward=False)
def _energy(
    q: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    xi: wp.array2d[wp.float64],
    point: wp.array2d[wp.vec3d],
    normal: wp.array2d[wp.vec3d],
    distance: wp.array[wp.float64],
    anchor: wp.array2d[wp.vec3d],
    limit: wp.array2d[wp.float64],
    stiffness: wp.float64,
    smoothing: wp.float64,
    status: wp.array[int],
    energy: wp.array2d[wp.float64],
):
    candidate, group, element = wp.tid()
    direction = normal[group, element]
    if status[0] != 0 or wp.length(direction) == wp.float64(0.0):
        return
    position = _centerline_point(
        q, candidate * (lengths.shape[0] + 1) + element, _shape(xi[group, element], lengths[element])
    )
    height = wp.dot(direction, position - point[group, element])
    slip = wp.length(_plane_offset(position, anchor[group, element], direction))
    value = contact_terms(distance[group], height, stiffness, smoothing)[0]
    value += _friction_energy(slip, stiffness, limit[group, element])
    wp.atomic_add(energy, 0, candidate, value)


@wp.kernel(enable_backward=False)
def _forces(
    q: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    xi: wp.array2d[wp.float64],
    point: wp.array2d[wp.vec3d],
    normal: wp.array2d[wp.vec3d],
    distance: wp.array[wp.float64],
    anchor: wp.array2d[wp.vec3d],
    limit: wp.array2d[wp.float64],
    stiffness: wp.float64,
    smoothing: wp.float64,
    friction: wp.float64,
    force: wp.array2d[wp.vec3d],
    application: wp.array2d[wp.vec3d],
    used_limit: wp.array2d[wp.float64],
):
    group, element = wp.tid()
    direction = normal[group, element]
    force[group, element] = wp.vec3d()
    used_limit[group, element] = limit[group, element]
    if wp.length(direction) == wp.float64(0.0):
        limit[group, element] = wp.float64(0.0)
        return
    position = _centerline_point(q, element, _shape(xi[group, element], lengths[element]))
    application[group, element] = position
    push = -contact_terms(distance[group], wp.dot(direction, position - point[group, element]), stiffness, smoothing)[1]
    offset = _plane_offset(position, anchor[group, element], direction)
    slip, held = wp.length(offset), limit[group, element]
    force[group, element] = push * direction - _friction_scale(slip, stiffness, held) * offset
    if held == wp.float64(0.0):
        anchor[group, element] = position
    elif stiffness * slip > held:
        anchor[group, element] = position - (held / (stiffness * slip)) * offset
    limit[group, element] = friction * push


@wp.kernel(enable_backward=False)
def _reaction(
    bodies: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    force: wp.array2d[wp.vec3d],
    application: wp.array2d[wp.vec3d],
    normal: wp.array2d[wp.vec3d],
    anchor: wp.array2d[wp.vec3d],
    local_anchor: wp.array2d[wp.vec3d],
    fresh: wp.array[wp.spatial_vector],
    count: wp.array[int],
):
    group, element = wp.tid()
    body = bodies[group]
    if wp.length(normal[group, element]) == wp.float64(0.0):
        return
    local_anchor[group, element] = _local(body_q[body], anchor[group, element])
    push = -force[group, element]
    lever = application[group, element] - _world(body_q[body], wp.vec3d(body_com[body]))
    wp.atomic_add(fresh, body, wp.spatial_vector(wp.vec3(push), wp.vec3(wp.cross(lever, push))))
    wp.atomic_add(count, body, 1)


@wp.kernel(enable_backward=False)
def _relax(
    dt: wp.float64,
    stiffness: wp.float64,
    mass: wp.array[wp.float64],
    inertia: wp.array[wp.float64],
    reach: wp.array[wp.float64],
    damping: wp.array2d[wp.float64],
    count: wp.array[int],
    fresh: wp.array[wp.spatial_vector],
    wrench: wp.array[wp.spatial_vector],
    relaxation: wp.array[wp.float64],
):
    body = wp.tid()
    if mass[body] <= wp.float64(0.0):
        return
    linear = mass[body] / (dt * dt) + damping[body, 0] / dt
    angular = inertia[body] / (dt * dt) + damping[body, 1] / dt
    contact = stiffness * wp.float64(wp.max(count[body], 1))
    fraction = wp.min(linear / (linear + contact), angular / (angular + contact * reach[body] * reach[body]))
    relaxation[body] = fraction
    wrench[body] += float(fraction) * (fresh[body] - wrench[body])


@wp.kernel(enable_backward=False)
def _add(wrench: wp.array[wp.spatial_vector], body_f: wp.array[wp.spatial_vector]):
    body = wp.tid()
    body_f[body] += wrench[body]


class ThreadRigidCapsuleContact:
    """Register simultaneous finite-friction capsule contacts on one SolverThread.

    Capsule endpoints are [C,2,3] in their actual Newton body frames, radii [C]
    in metres and bodies [C] integer model indices. The same arrays should build
    the MuJoCo collision geometry. ``skip_by_body`` excludes attached head
    elements only from that body, never from other tools. Support damping must
    be a known lower bound of a real drive; the default uses body inertia only.

    Before rigid stepping: add_forces(body_f). After an accepted rigid/tissue
    interval: update(body_q,body_qd,dt), thread.step(...,dt), react(body_q).
    No thread state may advance on rejected rigid trials. ``state`` contains
    all mutable buffers, including friction memory and lagged reactions.
    """

    def __init__(
        self,
        thread,
        model,
        bodies,
        endpoints,
        radii,
        *,
        thread_radius,
        friction=0.0,
        skip_by_body=None,
        damping_by_body=None,
    ):
        if thread.n_envs != 1 or model.device != thread.device:
            raise ValueError("Rigid capsule contacts need one thread chain on the rigid model device")
        bodies = np.asarray(bodies)
        endpoints, radii = np.asarray(endpoints, dtype=float), np.asarray(radii, dtype=float)
        if bodies.ndim != 1 or not len(bodies) or not np.issubdtype(bodies.dtype, np.integer):
            raise ValueError("bodies must be a nonempty integer vector")
        if np.any(bodies < 0) or np.any(bodies >= model.body_count):
            raise ValueError("Capsule bodies must exist in the model")
        if endpoints.shape != (len(bodies), 2, 3) or not np.isfinite(endpoints).all():
            raise ValueError("endpoints must be finite [capsules,2,3] local coordinates")
        if np.any(np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1) <= 1e-12):
            raise ValueError("Capsule axes must have nonzero length")
        if radii.shape != bodies.shape or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("Capsule radii must be finite positive metres")
        if not np.isfinite(thread_radius) or thread_radius <= 0 or not np.isfinite(friction) or friction < 0:
            raise ValueError("thread_radius must be positive and friction nonnegative")
        if thread.contact_stiffness <= 0:
            raise ValueError("Rigid capsule contact needs positive thread contact stiffness")
        skips, supports = dict(skip_by_body or {}), dict(damping_by_body or {})
        unique_bodies = set(map(int, bodies))
        if set(skips) - unique_bodies or set(supports) - unique_bodies:
            raise ValueError("Skip/support entries must name capsule bodies")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= v <= thread.thread.element_count
            for v in skips.values()
        ):
            raise ValueError("Skipped element counts must be integers in the thread range")
        mass = np.asarray(model.body_mass.numpy(), dtype=float)
        inertia = np.linalg.eigvalsh(model.body_inertia.numpy()).min(axis=1).astype(float)
        if any(mass[b] <= 0 or inertia[b] <= 0 for b in unique_bodies):
            raise ValueError("Capsule contact bodies need finite positive mass and inertia")
        if not np.isfinite(mass).all() or not np.isfinite(inertia).all():
            raise ValueError("Rigid mass and inertia must be finite")
        damping = np.zeros((model.body_count, 2))
        for body, value in supports.items():
            value = np.asarray(value, dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError("Support damping must be finite nonnegative linear/angular pairs")
            damping[body] = value
        keys = sorted(set(zip(map(int, bodies), map(float, radii), strict=True)))
        order, offsets = [], [0]
        for body, radius in keys:
            order.extend(np.flatnonzero((bodies == body) & (radii == radius)))
            offsets.append(len(order))
        com, reach = model.body_com.numpy(), np.zeros(model.body_count)
        for body in unique_bodies:
            take = bodies == body
            reach[body] = np.max(
                np.linalg.norm(endpoints[take] - com[body], axis=-1) + radii[take, None] + thread_radius
            )
        self.thread, self.model, self.friction = thread, model, float(friction)
        self.groups, self.elements = len(keys), thread.thread.element_count
        self._group_body_host = np.array([k[0] for k in keys], dtype=np.int32)
        device = model.device
        self.bodies = wp.array(self._group_body_host, dtype=int, device=device)
        self._offsets = wp.array(offsets, dtype=int, device=device)
        self._endpoints = wp.array(endpoints[order], dtype=wp.vec3d, device=device)
        self._distance = wp.array([r + thread_radius for _, r in keys], dtype=wp.float64, device=device)
        self._skip = wp.array([skips.get(b, 0) for b, _ in keys], dtype=int, device=device)
        self._mass, self._inertia, self._reach = [
            wp.array(a, dtype=wp.float64, device=device) for a in (mass, inertia, reach)
        ]
        self._damping = wp.array(damping, dtype=wp.float64, device=device)
        shape = (self.groups, self.elements)

        def zeros(dtype=wp.float64):
            return wp.zeros(shape, dtype=dtype, device=device)

        self.xi, self.point, self.normal = zeros(), zeros(wp.vec3d), zeros(wp.vec3d)
        self.anchor, self._local_anchor = zeros(wp.vec3d), zeros(wp.vec3d)
        self.limit, self.used_limit = zeros(), zeros()
        self.force, self.application_point = zeros(wp.vec3d), zeros(wp.vec3d)
        self.fresh_wrench = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
        self.wrench = wp.zeros_like(self.fresh_wrench)
        self._count = wp.zeros(model.body_count, dtype=int, device=device)
        self.relaxation = wp.zeros(model.body_count, dtype=wp.float64, device=device)
        self._previous_pose = wp.full(model.body_count, wp.transform_identity(), dtype=wp.transform, device=device)
        self._body_q = None
        self._dt = None
        thread.element_contacts.append(self)

    @property
    def state(self):
        return (
            self.xi,
            self.point,
            self.normal,
            self.anchor,
            self._local_anchor,
            self.limit,
            self.used_limit,
            self.force,
            self.application_point,
            self.fresh_wrench,
            self.wrench,
            self._count,
            self.relaxation,
            self._previous_pose,
        )

    def update(self, body_q, body_qd, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Rigid capsule contact dt must be finite and positive")
        if body_q.shape != (self.model.body_count,) or body_qd.shape != (self.model.body_count,):
            raise ValueError("Rigid contact poses and velocities must cover every model body")
        self._body_q, self._dt = body_q, float(dt)
        wp.launch(
            _find,
            (self.groups, self.elements),
            inputs=[
                self.thread.q,
                self.thread.qd,
                self.thread.thread.lengths,
                body_q,
                body_qd,
                self._previous_pose,
                self.bodies,
                self._offsets,
                self._endpoints,
                self._distance,
                self._skip,
                self.thread.contact_smoothing,
                dt,
                self.xi,
                self.point,
                self.normal,
                self.limit,
                self._local_anchor,
                self.anchor,
            ],
            device=self.model.device,
        )

    def _common(self):
        return [
            self.thread.thread.lengths,
            self.xi,
            self.point,
            self.normal,
            self._distance,
            self.anchor,
            self.limit,
            self.thread.contact_stiffness,
            self.thread.contact_smoothing,
        ]

    def add_gradient(self, tangent_blocks):
        wp.launch(
            _gradient,
            (self.groups, self.elements),
            inputs=[
                self.thread.q,
                *self._common(),
                tangent_blocks,
                self.thread._status,
                self.thread._gradient,
                self.thread._hessian,
            ],
            device=self.model.device,
        )

    def add_trial_energy(self):
        wp.launch(
            _energy,
            (len(self.thread._scales), self.groups, self.elements),
            inputs=[self.thread._trial_q, *self._common(), self.thread._status, self.thread._trial_energy],
            device=self.model.device,
        )

    def finish_step(self):
        wp.launch(
            _forces,
            (self.groups, self.elements),
            inputs=[self.thread.q, *self._common(), self.friction, self.force, self.application_point, self.used_limit],
            device=self.model.device,
        )

    def react(self, body_q):
        if self._dt is None:
            raise RuntimeError("update must precede the thread step and react")
        self.fresh_wrench.zero_()
        self._count.zero_()
        wp.launch(
            _reaction,
            (self.groups, self.elements),
            inputs=[
                self.bodies,
                body_q,
                self.model.body_com,
                self.force,
                self.application_point,
                self.normal,
                self.anchor,
                self._local_anchor,
                self.fresh_wrench,
                self._count,
            ],
            device=self.model.device,
        )
        wp.launch(
            _relax,
            self.model.body_count,
            inputs=[
                self._dt,
                self.thread.contact_stiffness,
                self._mass,
                self._inertia,
                self._reach,
                self._damping,
                self._count,
                self.fresh_wrench,
                self.wrench,
                self.relaxation,
            ],
            device=self.model.device,
        )
        wp.copy(self._previous_pose, body_q)

    def add_forces(self, body_f):
        wp.launch(_add, self.model.body_count, inputs=[self.wrench, body_f], device=self.model.device)

    def diagnostics(self):
        """Explicit host readback for reports; never called by the step kernels."""
        normal, force = self.normal.numpy(), self.force.numpy()
        active = np.linalg.norm(normal, axis=-1) > 0
        normal_force = np.sum(force * normal, axis=-1)
        tangent = force - normal_force[..., None] * normal
        gap = (
            np.sum(normal * (self.application_point.numpy() - self.point.numpy()), axis=-1)
            - self._distance.numpy()[:, None]
        )
        return {
            "candidate_contacts": int(active.sum()),
            "minimum_contact_gap_m": float(gap[active].min()) if active.any() else None,
            "maximum_normal_force_N": float(normal_force.max(initial=0)),
            "maximum_friction_force_N": float(np.linalg.norm(tangent, axis=-1).max(initial=0)),
            "maximum_friction_limit_excess_N": float(
                (np.linalg.norm(tangent, axis=-1) - self.used_limit.numpy()).max(initial=0)
            ),
            "fresh_body_wrench_N_Nm": self.fresh_wrench.numpy().tolist(),
            "applied_lagged_body_wrench_N_Nm": self.wrench.numpy().tolist(),
            "relaxation": self.relaxation.numpy().tolist(),
            "friction_coefficient": self.friction,
            "groups": self.groups,
            "collision_scope": "simultaneous nearest chord/capsule plane per body-radius group; compliant discrete contact",
        }
