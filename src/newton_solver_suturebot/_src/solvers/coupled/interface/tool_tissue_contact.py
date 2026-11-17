# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frictionless contact between rigid tool capsules and a tissue surface, one accepted step late.

Every capsule carries a chain of sample spheres that are fixed on its body. After an accepted
step each sphere takes the nearest point of the tissue surface at the accepted state and
updates one contact force

    f <- max(0, f - phi (K / n) g)

with g the gap between the sphere and the surface [m] (negative inside the tissue), K a lower
bound of the stiffness with which one tissue node resists a point force [N/m], n the number of
active samples (those that carry a force or are inside the tissue) and

    phi = min(1, (omega h)^2 / 2)

for the step h [s] and the lowest shear frequency omega [rad/s] of the tissue layer. The force
acts along the outward direction of the surface at that point: the normal of the triangle, or,
where the point lies on an edge or a vertex, the direction between the point and the centre.
It is spread with the opposite sign over the tissue particles of the touched triangle by the
barycentric weights of the surface point, and it is applied to the body of the sample at the
surface point. Both loads are stored. The caller applies them through the whole next step, so
the pair is equal and opposite as applied.

The force stops changing when the sphere rests on the surface: nothing of the commanded overlap
is left at rest, whatever the stiffness of the tissue, and a moving tool rides inside the
surface by the gap that the update needs to follow it. ``max(0, .)`` is the contact condition
(a contact cannot pull), not a limit on the force. There is no stiffness, damping or speed
constant.

One update is one projected-gradient iteration of the frictionless contact problem of the
samples against the tissue, with the step K / n. With W the compliance of the tissue between
the samples, it does not overshoot while (K / n) lambda_max(W) <= 1. W is symmetric positive
semi-definite, so lambda_max(W) <= trace(W) <= n / K when no loaded surface point is softer
than K: the bound holds for any number and placement of contacts. phi keeps that margin when a
step is shorter than the response time of the tissue layer.

Nothing here reads the device: the number of active samples stays on it.

Where it holds and where it does not (docs/method.md has the measurements). The force of a
sample is one point load on the surface, at the nearest point, and the bound above is for loads
whose points of application stay where they are. Two cases are outside it:

- A nearest point on a triangle with fixed vertices, or with vertices outside the simulated
  region, cannot yield. The force then grows until the tool gives way, and the free vertices
  of such a triangle take their share of a force that the elements beside the held nodes
  cannot carry.
- Under a point load the surface forms a pit, and the nearest point of the sphere moves from
  the pit to its flank. On the 1.3 mm tetrahedra of the grasp scene a dent of 0.3 mm is
  carried; from a dent of 0.4 to 0.7 mm on (about 10 mN on one sphere) the load alternates
  between two neighbouring triangles from one accepted step to the next, and the alternation
  grows.

In both cases tetrahedra invert and the step control rejects the step. This contact is
therefore usable for light touches only, and the scene keeps it off unless it is asked for.
"""

import math

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _clear(wrench: wp.array[wp.spatial_vector], count: wp.array[int]):
    """Zero the stored body wrenches and the number of active samples; launch over bodies."""
    body = wp.tid()
    wrench[body] = wp.spatial_vector()
    if body == 0:
        count[0] = 0


@wp.kernel(enable_backward=False)
def _find(
    mesh: wp.uint64,
    body_q: wp.array[wp.transform],
    sample_body: wp.array[int],
    sample_local: wp.array[wp.vec3],
    sample_radius: wp.array[float],
    reach: float,
    force: wp.array[float],
    gap: wp.array[float],
    face: wp.array[int],
    weights: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    point: wp.array[wp.vec3],
    count: wp.array[int],
):
    """Nearest surface point, outward direction and gap of every sample; count the active ones.

    A sample without a surface point within ``reach`` gets face -1.
    """
    sample = wp.tid()
    face[sample] = -1
    centre = wp.transform_point(body_q[sample_body[sample]], sample_local[sample])
    query = wp.mesh_query_point_sign_normal(mesh, centre, reach)
    if not query.result:
        return
    nearest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
    offset = centre - nearest
    distance = wp.length(offset)
    # Outward from the tissue. Over the interior of a triangle the offset is along the triangle's
    # normal, on either side of it: the normal is that direction without the rounding of a short
    # offset, and it is the direction of a centre on the surface. Over an edge or a vertex the
    # offset decides.
    direction = wp.mesh_eval_face_normal(mesh, query.face)
    if (query.u <= 0.0 or query.v <= 0.0 or query.u + query.v >= 1.0) and distance > 0.0:
        direction = query.sign * offset / distance
    separation = query.sign * distance - sample_radius[sample]
    face[sample] = query.face
    weights[sample] = wp.vec3(query.u, query.v, 1.0 - query.u - query.v)
    normal[sample] = direction
    point[sample] = nearest
    gap[sample] = separation
    if force[sample] > 0.0 or separation < 0.0:
        wp.atomic_add(count, 0, 1)


@wp.kernel(enable_backward=False)
def _update(
    parameters: wp.array[float],
    step_seconds: wp.array[wp.float64],
    count: wp.array[int],
    sample_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    gap: wp.array[float],
    face: wp.array[int],
    weights: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    point: wp.array[wp.vec3],
    triangles: wp.array[int],
    point_particle: wp.array[int],
    force: wp.array[float],
    particle_f: wp.array[wp.vec3],
    wrench: wp.array[wp.spatial_vector],
):
    """Update the force of every sample and add it to the tissue particles and to the body wrenches.

    ``parameters`` holds the node stiffness [N/m] and the shear frequency [rad/s].
    """
    sample = wp.tid()
    if face[sample] < 0:
        force[sample] = 0.0
        return
    value = force[sample]
    if value > 0.0 or gap[sample] < 0.0:
        # This sample is one of the counted ones: the count is at least one.
        angle = parameters[1] * float(step_seconds[0])
        fraction = wp.min(1.0, 0.5 * angle * angle)
        value = wp.max(0.0, value - fraction * (parameters[0] / float(count[0])) * gap[sample])
        force[sample] = value
    if value <= 0.0:
        return
    push = value * normal[sample]
    for corner in range(3):
        particle = point_particle[triangles[3 * face[sample] + corner]]
        if particle >= 0:
            wp.atomic_add(particle_f, particle, -weights[sample][corner] * push)
    body = sample_body[sample]
    lever = point[sample] - wp.transform_point(body_q[body], body_com[body])
    wp.atomic_add(wrench, body, wp.spatial_vector(push, wp.cross(lever, push)))


@wp.kernel(enable_backward=False)
def _add(wrench: wp.array[wp.spatial_vector], body_f: wp.array[wp.spatial_vector]):
    body = wp.tid()
    body_f[body] += wrench[body]


def capsule_samples(bodies, endpoints, radii, contact_radii=None):
    """Sample spheres that cover capsules, in the frames of their bodies.

    A capsule is given by the two end points of its axis and its radius. Its chain of spheres
    has the contact radius (by default the capsule's own) and reaches, cap to cap, as far along
    the axis as the capsule does: with a contact radius R and a capsule radius r the first and
    the last centre lie R - r inside the end points. The centres are at most R apart, so the
    chain lies within R (1 - sqrt(3) / 2) of the capsule of radius R that it stands for. A
    capsule that is shorter, cap to cap, than one sphere gets one sphere at its middle.

    Args:
        bodies: Body index of each capsule, shape (count,).
        endpoints: End points of each capsule axis in the body frame [m], shape (count, 2, 3).
        radii: Capsule radii [m], shape (count,).
        contact_radii: Radii of the spheres [m], shape (count,); ``None`` takes ``radii``.

    Returns:
        Body index of each sample, its centre in the body frame [m] and its radius [m], with
        shapes (samples,), (samples, 3) and (samples,).
    """
    bodies = np.asarray(bodies)
    endpoints, radii = np.asarray(endpoints, dtype=float), np.asarray(radii, dtype=float)
    contact_radii = radii if contact_radii is None else np.asarray(contact_radii, dtype=float)
    if bodies.ndim != 1 or not len(bodies) or not np.issubdtype(bodies.dtype, np.integer):
        raise ValueError("bodies must be a nonempty integer vector")
    if endpoints.shape != (len(bodies), 2, 3) or not np.isfinite(endpoints).all():
        raise ValueError("endpoints must be finite [capsules,2,3] local coordinates")
    for values in (radii, contact_radii):
        if values.shape != bodies.shape or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Capsule and contact radii must be finite positive metres, one per capsule")
    sample_bodies, centres, sample_radii = [], [], []
    for body, (start, end), radius, contact in zip(bodies, endpoints, radii, contact_radii, strict=True):
        length = float(np.linalg.norm(end - start))
        if length <= 1e-12:
            raise ValueError("Capsule axes must have nonzero length")
        inset = (contact - radius) / length
        span = length - 2.0 * (contact - radius)
        if span > 0.0:
            # A span of a whole number of radii is not made one sphere longer by rounding.
            fractions = np.linspace(inset, 1.0 - inset, int(math.ceil(span / contact * (1.0 - 1e-12))) + 1)
        else:
            fractions = np.array([0.5])
        for fraction in fractions:
            sample_bodies.append(int(body))
            centres.append(start + fraction * (end - start))
            sample_radii.append(float(contact))
    return np.asarray(sample_bodies, dtype=np.int32), np.asarray(centres), np.asarray(sample_radii)


class ToolTissueContact:
    """Normal contact of sample spheres on rigid bodies with a triangle surface of a tissue.

    After an accepted step: :meth:`update` with the accepted body poses and the particle force
    array that the tissue solver reads in its next step. Before every attempt of the next step:
    :meth:`add_forces` on the body forces. Both only launch kernels, so the update can be
    captured in a CUDA graph. The owner of ``mesh`` moves its points and refits it at the
    accepted state before :meth:`update`.

    Args:
        model: Rigid model of the bodies.
        mesh: Triangle surface of the tissue with outward-facing triangles (``wp.Mesh``).
        triangles: Vertex indices of the surface triangles, flat, three per triangle.
        point_particle: Tissue particle of each surface vertex, -1 for a vertex that is not one.
        sample_bodies: Body index of each sample, shape (samples,).
        sample_points: Centre of each sample in its body frame [m], shape (samples, 3).
        sample_radii: Radius of each sample [m], shape (samples,).
        node_stiffness: Lower bound of the stiffness of one tissue node [N/m].
        shear_frequency: Lowest shear frequency of the tissue layer [rad/s].
        step_seconds: One ``float64`` on the device that holds the step [s] of the accepted
            step when :meth:`update` runs.
        reach: Largest distance from a sample centre at which the surface is looked for [m].
        reaction: False leaves the bodies without the reaction; for diagnosis only.

    Attributes:
        force: Contact force of every sample [N]; with :attr:`wrench` the state between steps.
        wrench: Stored reaction on every body [N, N m], force first, torque about the centre
            of mass, world frame.
    """

    def __init__(
        self,
        model,
        mesh,
        triangles,
        point_particle,
        sample_bodies,
        sample_points,
        sample_radii,
        *,
        node_stiffness,
        shear_frequency,
        step_seconds,
        reach,
        reaction=True,
    ):
        device = model.device
        sample_bodies = np.asarray(sample_bodies)
        sample_points = np.asarray(sample_points, dtype=float)
        sample_radii = np.asarray(sample_radii, dtype=float)
        count = len(sample_bodies)
        if sample_bodies.ndim != 1 or not count or not np.issubdtype(sample_bodies.dtype, np.integer):
            raise ValueError("sample_bodies must be a nonempty integer vector")
        if np.any(sample_bodies < 0) or np.any(sample_bodies >= model.body_count):
            raise ValueError("Sample bodies must exist in the model")
        if sample_points.shape != (count, 3) or not np.isfinite(sample_points).all():
            raise ValueError("sample_points must be finite [samples,3] local coordinates")
        if sample_radii.shape != (count,) or not np.isfinite(sample_radii).all() or np.any(sample_radii <= 0):
            raise ValueError("Sample radii must be finite positive metres")
        values = (node_stiffness, shear_frequency, reach)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Node stiffness, shear frequency and reach must be finite and positive")
        if step_seconds.shape != (1,) or step_seconds.dtype != wp.float64 or step_seconds.device != device:
            raise ValueError("step_seconds must be one float64 on the device of the model")
        self.model, self.mesh = model, mesh
        self.node_stiffness, self.shear_frequency = float(node_stiffness), float(shear_frequency)
        self.reach, self.reaction = float(reach), bool(reaction)
        self.sample_count = count
        self._triangles, self._point_particle, self._step_seconds = triangles, point_particle, step_seconds
        self._sample_body = wp.array(sample_bodies, dtype=int, device=device)
        self._sample_local = wp.array(sample_points, dtype=wp.vec3, device=device)
        self._sample_radius = wp.array(sample_radii, dtype=float, device=device)
        # Device data: a changed material is a write of this array, not a new graph.
        self._parameters = wp.array([self.node_stiffness, self.shear_frequency], dtype=float, device=device)
        self.force = wp.zeros(count, dtype=float, device=device)
        self.wrench = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
        self._gap = wp.zeros(count, dtype=float, device=device)
        self._face = wp.full(count, -1, dtype=int, device=device)
        self._weights = wp.zeros(count, dtype=wp.vec3, device=device)
        self._normal = wp.zeros(count, dtype=wp.vec3, device=device)
        self._point = wp.zeros(count, dtype=wp.vec3, device=device)
        self._count = wp.zeros(1, dtype=int, device=device)
        # As the thread's surface queries (ThreadTissueCoupling.update): a batch that fits in one
        # default block spreads its independent walks of the bounding volume hierarchy with one
        # query per block. Measured for the 162 samples of the grasp scene on one GB10: the three
        # kernels of an update take 0.13 ms that way and 0.24 ms with one block of 256.
        self._query_block = 1 if device.is_cuda and count <= 256 else 256

    @property
    def state(self) -> tuple:
        """The arrays that carry over from one accepted step to the next."""
        return (self.force, self.wrench)

    def update(self, body_q, particle_f):
        """Update the contact forces at an accepted state and add those on the tissue to ``particle_f`` [N].

        The step is read from ``step_seconds``, which must hold it. The reaction on the bodies
        is stored for :meth:`add_forces`.
        """
        device = self.model.device
        wp.launch(_clear, self.model.body_count, outputs=[self.wrench, self._count], device=device)
        wp.launch(
            _find,
            self.sample_count,
            inputs=[
                self.mesh.id,
                body_q,
                self._sample_body,
                self._sample_local,
                self._sample_radius,
                self.reach,
                self.force,
            ],
            outputs=[self._gap, self._face, self._weights, self._normal, self._point, self._count],
            device=device,
            block_dim=self._query_block,
        )
        wp.launch(
            _update,
            self.sample_count,
            inputs=[
                self._parameters,
                self._step_seconds,
                self._count,
                self._sample_body,
                body_q,
                self.model.body_com,
                self._gap,
                self._face,
                self._weights,
                self._normal,
                self._point,
                self._triangles,
                self._point_particle,
            ],
            outputs=[self.force, particle_f, self.wrench],
            device=device,
        )

    def add_forces(self, body_f):
        """Add the stored reaction to the body forces; does not change the contact state."""
        if self.reaction:
            wp.launch(_add, self.model.body_count, inputs=[self.wrench, body_f], device=self.model.device)

    def parameters(self):
        """Host values of the contact, without a device read."""
        return {
            "law": "f <- max(0, f - phi (K / n) gap) per sample and accepted step; phi = min(1, (omega h)^2 / 2)",
            "node_stiffness_N_m": self.node_stiffness,
            "shear_frequency_rad_s": self.shear_frequency,
            "reach_m": self.reach,
            "samples": self.sample_count,
            "reaction_on_bodies": self.reaction,
        }

    def diagnostics(self):
        """Explicit host readback for reports and tests; never called by the step."""
        force, gap, face = self.force.numpy(), self._gap.numpy(), self._face.numpy()
        found, loaded = face >= 0, force > 0
        body = self._sample_body.numpy()
        per_body = np.zeros(self.model.body_count)
        np.add.at(per_body, body, force)
        return {
            **self.parameters(),
            "active_samples": int(self._count.numpy()[0]),
            "loaded_samples": int(loaded.sum()),
            "total_force_N": float(force.sum()),
            "maximum_sample_force_N": float(force.max(initial=0)),
            "sample_force_by_body_N": per_body.tolist(),
            "deepest_gap_m": float(gap[found].min()) if found.any() else None,
            "applied_body_wrench_N_Nm": self.wrench.numpy().tolist(),
        }
