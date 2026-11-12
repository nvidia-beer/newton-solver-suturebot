# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-way coupling of one thread to a tetrahedral tissue without host readbacks.

The thread touches the tissue in two ways.

Surface contact. Every thread node takes the nearest point of a triangle
surface and its outward normal as its contact plane for the step. The surface
may reach beyond the tissue: its vertices that are tissue particles move with
them, the others stay where they were given. The node is pushed out along the
normal by the thread solver's contact penalty and held along the surface by
its friction. The friction anchor of a node that has grip is a point of the
surface and moves with it, so a thread that sticks rides on the tissue. The
contact force with the opposite sign, normal and friction, is spread over
those vertices of the surface triangle that are tissue particles.

Eyelets. A material point of the tissue, given by a tetrahedron and its
barycentric weights, can be an eyelet of the thread solver: the thread must
pass through it but may slide along, against the solver's friction. Eyelets are made in groups, each an
ordered track through the tissue such as the path a needle cut. The thread
between the first and the last threaded eyelet of a group is inside the
tissue and has no surface contact; while a group is only partly threaded, the
thread from its first node up to the group's last threaded eyelet is inside.
The force of the thread on an eyelet is spread over the four vertices of its
tetrahedron.

The thread solver treats the tissue as given during its step, and the tissue
receives the thread's forces as external loads in its next step. Passed on
at once, that exchange diverges when the thread's spring is stiffer than the
tissue node it acts on: each side then overcorrects the other. The force on
the tissue is therefore relaxed. With k the stiffer of the contact and eyelet
springs and K a lower bound of the stiffness of a tissue node, each step moves
the applied force by the fraction K / (K + k) towards the thread's current
force. For a spring k in series with any thread stiffness and a tissue node
at least as stiff as K, that iteration contracts monotonically. The tissue
feels a change of the thread's pull after about (K + k) / K steps.
"""

import numpy as np
import warp as wp

from ...geometry.mesh_refit import GraphMeshRefit
from .solver import SolverThread


@wp.kernel(enable_backward=False)
def _move_surface(particle_q: wp.array[wp.vec3], particle_point: wp.array[int], points: wp.array[wp.vec3]):
    """Move the surface vertices that are tissue particles; launch over particles."""
    particle = wp.tid()
    points[particle_point[particle]] = particle_q[particle]


@wp.kernel(enable_backward=False)
def _relax(
    fresh: wp.array[wp.vec3],
    fraction: float,
    applied: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
):
    """Move the applied force towards the fresh one, add it to the particle forces and clear the fresh one."""
    particle = wp.tid()
    applied[particle] += fraction * (fresh[particle] - applied[particle])
    particle_f[particle] += applied[particle]
    fresh[particle] = wp.vec3(0.0)


@wp.kernel(enable_backward=False)
def _thread_eyelet(slot: int, tracked: int, coordinate: wp.float64, element: wp.array[int], xi: wp.array[wp.float64]):
    element[slot] = tracked
    xi[slot] = coordinate


@wp.kernel(enable_backward=False)
def _hold_eyelet(slot: int, value: int, held: wp.array[int]):
    held[slot] = value


@wp.kernel(enable_backward=False)
def _clear_groups(lowest: wp.array[wp.float64], highest: wp.array[wp.float64], count: wp.array[int]):
    group = wp.tid()
    lowest[group] = wp.float64(1.0e300)
    highest[group] = wp.float64(-1.0e300)
    count[group] = 0


@wp.kernel(enable_backward=False)
def _group_ranges(
    element: wp.array[int],
    xi: wp.array[wp.float64],
    group: wp.array[int],
    lowest: wp.array[wp.float64],
    highest: wp.array[wp.float64],
    count: wp.array[int],
):
    """Span of each group's threaded eyelets along the thread, in elements; launch over eyelets."""
    eyelet = wp.tid()
    if element[eyelet] < 0:
        return
    coordinate = wp.float64(element[eyelet]) + xi[eyelet]
    wp.atomic_min(lowest, group[eyelet], coordinate)
    wp.atomic_max(highest, group[eyelet], coordinate)
    wp.atomic_add(count, group[eyelet], 1)


@wp.kernel(enable_backward=False)
def _contact_planes(
    mesh: wp.uint64,
    q: wp.array[wp.float64],
    margin: float,
    group_size: wp.array[int],
    lowest: wp.array[wp.float64],
    highest: wp.array[wp.float64],
    count: wp.array[int],
    contact_point: wp.array[wp.vec3d],
    contact_normal: wp.array[wp.vec3d],
    face: wp.array[int],
    weights: wp.array[wp.vec3],
):
    """Nearest surface point and its outward normal for every thread node; launch over nodes.

    A node farther than ``margin`` from the surface, or inside the tissue
    between threaded eyelets, gets no contact plane.
    """
    node = wp.tid()
    face[node] = -1
    contact_normal[node] = wp.vec3d()
    coordinate = wp.float64(node)
    for group in range(group_size.shape[0]):
        if count[group] > 0:
            first = lowest[group]
            if count[group] < group_size[group]:
                # The track is still being threaded: the thread's start is inside it.
                first = wp.float64(0.0)
            if coordinate >= first and coordinate <= highest[group]:
                return
    position = wp.vec3(float(q[6 * node]), float(q[6 * node + 1]), float(q[6 * node + 2]))
    query = wp.mesh_query_point_no_sign(mesh, position, margin)
    if not query.result:
        return
    contact_point[node] = wp.vec3d(wp.mesh_eval_position(mesh, query.face, query.u, query.v))
    contact_normal[node] = wp.vec3d(wp.mesh_eval_face_normal(mesh, query.face))
    face[node] = query.face
    weights[node] = wp.vec3(query.u, query.v, 1.0 - query.u - query.v)


@wp.kernel(enable_backward=False)
def _eyelet_targets(
    particle_q: wp.array[wp.vec3],
    tetrahedra: wp.array[wp.vec4i],
    tetrahedron: wp.array[int],
    weights: wp.array[wp.vec4],
    target: wp.array[wp.vec3d],
):
    """Current position of every eyelet's material point; launch over eyelets."""
    eyelet = wp.tid()
    index = tetrahedron[eyelet]
    if index < 0:
        return
    vertices = tetrahedra[index]
    position = wp.vec3(0.0)
    for corner in range(4):
        position += weights[eyelet][corner] * particle_q[vertices[corner]]
    target[eyelet] = wp.vec3d(position)


@wp.kernel(enable_backward=False)
def _carry_anchors(mesh: wp.uint64, face: wp.array[int], weights: wp.array[wp.vec3], anchor: wp.array[wp.vec3d]):
    """Move each node's friction anchor with the surface point it was stored on; launch over nodes."""
    node = wp.tid()
    if face[node] >= 0:
        anchor[node] = wp.vec3d(wp.mesh_eval_position(mesh, face[node], weights[node][0], weights[node][1]))


@wp.kernel(enable_backward=False)
def _store_anchors(
    mesh: wp.uint64,
    margin: float,
    limit: wp.array[wp.float64],
    anchor: wp.array[wp.vec3d],
    face: wp.array[int],
    weights: wp.array[wp.vec3],
):
    """Remember the surface point under the friction anchor of each node that has grip; launch over nodes."""
    node = wp.tid()
    face[node] = -1
    if limit[node] <= wp.float64(0.0):
        return
    point = wp.vec3(float(anchor[node][0]), float(anchor[node][1]), float(anchor[node][2]))
    query = wp.mesh_query_point_no_sign(mesh, point, margin)
    if query.result:
        face[node] = query.face
        weights[node] = wp.vec3(query.u, query.v, 1.0 - query.u - query.v)


@wp.kernel(enable_backward=False)
def _contact_reaction(
    contact_force: wp.array[wp.vec3d],
    face: wp.array[int],
    weights: wp.array[wp.vec3],
    triangles: wp.array[int],
    point_particle: wp.array[int],
    particle_f: wp.array[wp.vec3],
):
    """Add the opposite of each node's contact force to the tissue particles of its triangle; launch over nodes."""
    node = wp.tid()
    if face[node] < 0:
        return
    force = -wp.vec3(contact_force[node])
    for corner in range(3):
        particle = point_particle[triangles[3 * face[node] + corner]]
        if particle >= 0:
            wp.atomic_add(particle_f, particle, weights[node][corner] * force)


@wp.kernel(enable_backward=False)
def _eyelet_reaction(
    force: wp.array[wp.vec3d],
    tetrahedra: wp.array[wp.vec4i],
    tetrahedron: wp.array[int],
    weights: wp.array[wp.vec4],
    particle_f: wp.array[wp.vec3],
):
    """Spread the thread's force on each eyelet over its tetrahedron; launch over eyelets."""
    eyelet = wp.tid()
    index = tetrahedron[eyelet]
    if index < 0:
        return
    vertices = tetrahedra[index]
    pull = wp.vec3(force[eyelet])
    for corner in range(4):
        wp.atomic_add(particle_f, vertices[corner], weights[eyelet][corner] * pull)


class ThreadTissueCoupling:
    """Surface contact and eyelets between a one-chain thread solver and a tetrahedral tissue.

    Per step: call :meth:`update` with the tissue's current particle positions,
    step the thread, then call :meth:`add_forces` on the particle force array
    that the tissue solver reads in its next step. All three only launch
    kernels, so the sequence can be captured in a CUDA graph.

    Args:
        thread: Thread solver with one chain, built with a contact distance
            and as many eyelet slots as the tracks need.
        surface_points: Positions of the surface's vertices [m], shape
            (count, 3).
        surface_triangles: Outward-facing triangles of the surface as vertex
            indices, shape (count, 3).
        tetrahedra: Tissue tetrahedra as particle indices, shape (count, 4).
        particle_points: Surface vertex of each tissue particle; ``None`` when
            the particles are the surface's vertices in the same order.
        tissue_stiffness: Lower bound of the stiffness with which a tissue
            node resists a point force [N/m]; about a quarter of the tissue's
            Young's modulus times its edge length.
        contact_margin: Largest distance from the surface at which a thread
            node gets a contact plane [m].
        max_tracks: Largest number of eyelet groups.

    Attributes:
        track_count: Number of tracks made so far.
        applied_force: The relaxed force of the thread on every tissue
            particle [N]; part of the state between steps.
    """

    def __init__(
        self,
        thread: SolverThread,
        surface_points,
        surface_triangles,
        tetrahedra,
        particle_points=None,
        *,
        tissue_stiffness,
        contact_margin,
        max_tracks=8,
    ):
        if thread.n_envs != 1:
            raise ValueError("ThreadTissueCoupling couples a thread solver with one chain")
        device = thread.thread.device
        self.thread, self.contact_margin = thread, float(contact_margin)
        nodes, eyelets = thread.thread.node_count, thread.max_eyelets
        self._tetrahedra = wp.array(
            np.asarray(tetrahedra, dtype=np.int32).reshape(-1, 4), dtype=wp.vec4i, device=device
        )
        self._triangles = wp.array(np.asarray(surface_triangles, dtype=np.int32).reshape(-1), dtype=int, device=device)
        surface_points = np.asarray(surface_points, dtype=np.float32).reshape(-1, 3)
        if particle_points is None:
            particle_points = np.arange(len(surface_points))
        particle_points = np.asarray(particle_points, dtype=np.int32)
        point_particle = np.full(len(surface_points), -1, dtype=np.int32)
        point_particle[particle_points] = np.arange(len(particle_points))
        self._particle_point = wp.array(particle_points, dtype=int, device=device)
        self._point_particle = wp.array(point_particle, dtype=int, device=device)
        self._points = wp.array(surface_points, dtype=wp.vec3, device=device)
        self._mesh = wp.Mesh(self._points, self._triangles)
        # Warp's own refit allocates; the graph-safe refit exists for CUDA only and skips what cannot move.
        self._refit = GraphMeshRefit(self._mesh, moving_vertex_mask=point_particle >= 0) if device.is_cuda else None
        self._face = wp.full(nodes, -1, dtype=int, device=device)
        self._weights = wp.zeros(nodes, dtype=wp.vec3, device=device)
        self._anchor_face = wp.full(nodes, -1, dtype=int, device=device)
        self._anchor_weights = wp.zeros(nodes, dtype=wp.vec3, device=device)
        # Host copies of the eyelets' material points and groups; the device is only written from them.
        self._tetrahedron = np.full(eyelets, -1, dtype=np.int32)
        self._barycentric = np.zeros((eyelets, 4), dtype=np.float32)
        self._group = np.zeros(eyelets, dtype=np.int32)
        self._group_size = np.zeros(max_tracks, dtype=np.int32)
        self._tetrahedron_device = wp.array(self._tetrahedron, dtype=int, device=device)
        self._barycentric_device = wp.array(self._barycentric, dtype=wp.vec4, device=device)
        self._group_device = wp.array(self._group, dtype=int, device=device)
        self._group_size_device = wp.array(self._group_size, dtype=int, device=device)
        self._lowest = wp.zeros(max_tracks, dtype=wp.float64, device=device)
        self._highest = wp.zeros(max_tracks, dtype=wp.float64, device=device)
        self._count = wp.zeros(max_tracks, dtype=int, device=device)
        self.track_count = 0
        self._used = 0
        spring = max(thread.contact_stiffness, thread.eyelet_stiffness)
        self._relaxation = float(tissue_stiffness) / (float(tissue_stiffness) + spring)
        self.applied_force = wp.zeros(len(particle_points), dtype=wp.vec3, device=device)
        self._fresh = wp.zeros(len(particle_points), dtype=wp.vec3, device=device)

    @property
    def state(self) -> tuple:
        """The coupling's arrays that carry over from one step to the next; see :attr:`SolverThread.state`."""
        return (self.applied_force, self._anchor_face, self._anchor_weights)

    def add_track(self, tetrahedra, weights):
        """Make an ordered group of eyelets from material points; returns their slots in the thread solver.

        The eyelets start unthreaded; :meth:`thread_eyelet` passes the thread
        through one.

        Args:
            tetrahedra: Tetrahedron index of each material point.
            weights: Barycentric weights of each material point, shape (count, 4).
        """
        tetrahedra = np.asarray(tetrahedra, dtype=np.int32).reshape(-1)
        weights = np.asarray(weights, dtype=np.float32).reshape(-1, 4)
        if len(tetrahedra) != len(weights) or not len(tetrahedra):
            raise ValueError("A track needs one tetrahedron and four weights per eyelet")
        if self.track_count == len(self._group_size) or self._used + len(tetrahedra) > len(self._tetrahedron):
            raise ValueError("No track or eyelet slots are left")
        slots = np.arange(self._used, self._used + len(tetrahedra))
        self._tetrahedron[slots], self._barycentric[slots], self._group[slots] = tetrahedra, weights, self.track_count
        self._group_size[self.track_count] = len(slots)
        self._used += len(slots)
        self.track_count += 1
        self._tetrahedron_device.assign(self._tetrahedron)
        self._barycentric_device.assign(self._barycentric)
        self._group_device.assign(self._group)
        self._group_size_device.assign(self._group_size)
        return slots

    def thread_eyelet(self, slot, element=0, xi=0.0):
        """Pass the thread through eyelet ``slot`` at a point of the chain, by default at its first node.

        From then on the eyelet follows the thread as it slides. Only launches
        a kernel, so the other eyelets keep the points they track.
        """
        thread = self.thread
        wp.launch(
            _thread_eyelet,
            dim=1,
            inputs=[int(slot), int(element), wp.float64(xi)],
            outputs=[thread.eyelet_element, thread.eyelet_xi],
            device=thread.thread.device,
        )

    def hold_eyelet(self, slot, held=True):
        """Tie the thread to eyelet ``slot`` at the point it passes through now, or let it slide again.

        A held eyelet is a knot: the thread pulls its material point in every
        direction and cannot slide through. Only launches a kernel.
        """
        thread = self.thread
        wp.launch(
            _hold_eyelet,
            dim=1,
            inputs=[int(slot), int(bool(held))],
            outputs=[thread.eyelet_held],
            device=thread.thread.device,
        )

    def update(self, particle_q):
        """Give the thread its contact planes and eyelet positions for the tissue at ``particle_q``."""
        thread, device = self.thread, self.thread.thread.device
        wp.launch(
            _move_surface,
            dim=len(self._particle_point),
            inputs=[particle_q, self._particle_point],
            outputs=[self._points],
            device=device,
        )
        if self._refit is not None:
            self._refit.refit()
        else:
            self._mesh.refit()
        eyelets = thread.max_eyelets
        if eyelets:
            wp.launch(
                _eyelet_targets,
                dim=eyelets,
                inputs=[particle_q, self._tetrahedra, self._tetrahedron_device, self._barycentric_device],
                outputs=[thread.eyelet_target],
                device=device,
            )
            wp.launch(
                _clear_groups,
                dim=len(self._group_size),
                outputs=[self._lowest, self._highest, self._count],
                device=device,
            )
            wp.launch(
                _group_ranges,
                dim=eyelets,
                inputs=[thread.eyelet_element, thread.eyelet_xi, self._group_device],
                outputs=[self._lowest, self._highest, self._count],
                device=device,
            )
        # A node that sticks rides on the tissue: its friction anchor is a point of the surface.
        wp.launch(
            _carry_anchors,
            dim=thread.thread.node_count,
            inputs=[self._mesh.id, self._anchor_face, self._anchor_weights],
            outputs=[thread.contact_anchor],
            device=device,
        )
        # Small chains spread their independent BVH walks across SMs with one query per block.
        wp.launch(
            _contact_planes,
            dim=thread.thread.node_count,
            inputs=[
                self._mesh.id,
                thread.q,
                self.contact_margin,
                self._group_size_device,
                self._lowest,
                self._highest,
                self._count,
            ],
            outputs=[thread.contact_point, thread.contact_normal, self._face, self._weights],
            device=device,
            block_dim=1 if device.is_cuda and thread.thread.node_count <= 128 else 256,
        )

    def add_forces(self, particle_f):
        """Add the thread's relaxed contact and eyelet forces on the tissue to ``particle_f`` [N]; call after the thread step."""
        thread, device = self.thread, self.thread.thread.device
        wp.launch(
            _store_anchors,
            dim=thread.thread.node_count,
            inputs=[self._mesh.id, self.contact_margin, thread.contact_limit, thread.contact_anchor],
            outputs=[self._anchor_face, self._anchor_weights],
            device=device,
            block_dim=1 if device.is_cuda and thread.thread.node_count <= 128 else 256,
        )
        wp.launch(
            _contact_reaction,
            dim=thread.thread.node_count,
            inputs=[thread.contact_force, self._face, self._weights, self._triangles, self._point_particle],
            outputs=[self._fresh],
            device=device,
        )
        if thread.max_eyelets:
            wp.launch(
                _eyelet_reaction,
                dim=thread.max_eyelets,
                inputs=[thread.eyelet_force, self._tetrahedra, self._tetrahedron_device, self._barycentric_device],
                outputs=[self._fresh],
                device=device,
            )
        wp.launch(
            _relax,
            dim=len(self._fresh),
            inputs=[self._fresh, self._relaxation],
            outputs=[self.applied_force, particle_f],
            device=device,
        )
