# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical worker-owned thread with lagged needle, tool and tissue reactions.

The caller advances this object only after accepting a rigid/FEM interval and
adds its stored loads before the next rigid/FEM trial. Rejected trials never
advance the thread. No renderer owns this state and no puncture track or eyelet
is scheduled: where the scene asks for it, the thread follows the track that
the needle has actually cut (``thread_tracks``). The far end remains free.
"""

import math
from collections import OrderedDict

import numpy as np
import warp as wp

from ...utils.capture import ScopedCapture, note_release
from ..coupled.interface.tool_tissue_exchange import ToolTissueExchange, ToolTissueSetup, tool_tissue_law
from ..fem.differentiation import active_tape
from . import NeedleTrackEyelets, SolverThread, ThreadAttachment, ThreadModel, ThreadTissueCoupling
from .kernels import _centerline_point, _shape
from .thread_initialization import loose_thread_coordinates
from .thread_rigid_contact import ThreadRigidCapsuleContact


@wp.kernel(enable_backward=False)
def _add_nodal(source: wp.array[wp.vec3], target: wp.array[wp.vec3]):
    node = wp.tid()
    target[node] += source[node]


@wp.kernel(enable_backward=False)
def _sample(q: wp.array[wp.float64], lengths: wp.array[wp.float64], samples: int, points: wp.array[wp.vec3]):
    index = wp.tid()
    element = wp.min(index // samples, lengths.shape[0] - 1)
    xi = wp.float64(index - element * samples) / wp.float64(samples)
    points[index] = wp.vec3(_centerline_point(q, element, _shape(xi, lengths[element])))


@wp.kernel(enable_backward=False)
def _square_components(vectors: wp.array[wp.vec3], squares: wp.array[wp.vec3]):
    # The squares are stored before they are added: a compiler that fuses a product with the
    # addition that follows it rounds once where NumPy rounds twice.
    index = wp.tid()
    squares[index] = wp.cw_mul(vectors[index], vectors[index])


@wp.kernel(enable_backward=False)
def _largest_norm(squares: wp.array[wp.vec3], largest: wp.array[float]):
    # The float32 operations of np.linalg.norm(vectors, axis=-1).max(initial=0), in its order.
    # NumPy's maximum is NaN when a norm is; entry 1 records that, entry 0 the largest number.
    value = squares[wp.tid()]
    norm = wp.sqrt((value[0] + value[1]) + value[2])
    if wp.isnan(norm):
        largest[1] = 1.0
    else:
        wp.atomic_max(largest, 0, norm)


@wp.kernel(enable_backward=False)
def _thread_in_tracks(
    group_size: wp.array[int],
    lowest: wp.array[wp.float64],
    highest: wp.array[wp.float64],
    count: wp.array[int],
    element_length: float,
    report: wp.array[float],
):
    # Launch over the tracks of the tissue coupling. Entry 2: threaded eyelets; entry 3: rest length of
    # thread inside the tissue [m], by the rule of the coupling's contact planes: between the first and
    # the last threaded eyelet of a track, and from the thread's first node while a track is partly threaded.
    track = wp.tid()
    if count[track] <= 0:
        return
    first = lowest[track]
    if count[track] < group_size[track]:
        first = wp.float64(0.0)
    wp.atomic_add(report, 2, float(count[track]))
    wp.atomic_add(report, 3, float(highest[track] - first) * element_length)


@wp.kernel(enable_backward=False)
def _end_at_budget(budget: int, passes: wp.array[int], active: wp.array[int]):
    if passes[0] >= budget:
        active[0] = 0


class _WorkerSolverThread(SolverThread):
    """Thread solver whose Newton loop ends with the last update that its budget allows.

    :class:`SolverThread` runs one more pass after that update. With the budget used, that pass
    updates no chain: it evaluates the forces at the final coordinates into ``residual`` and marks
    every chain finished in the loop's own flags. The worker reads neither, and the pass costs a
    force evaluation of the whole thread. The coordinates, velocities, friction memory and reactions
    of the step are those of :class:`SolverThread`; ``residual`` holds the forces that the last
    update started from.
    """

    def _newton_pass(self, dt):
        super()._newton_pass(dt)
        wp.launch(
            _end_at_budget,
            dim=1,
            inputs=[self.max_iterations, self._passes],
            outputs=[self._active],
            device=self.thread.device,
        )


class WorkerThread:
    """A single scene's physical ANCF thread; all lengths are metres.

    ``scene`` supplies model, state_0 (accepted state), geometry, needle_body,
    reference, selection, args and grasper assemblies. ``config`` supplies
    thread_elements, thread_length_m, thread_radius_m and thread_modulus_pa.
    Optional thread_iterations defaults to2, matching the source stitch demo.

    Attachment uses the real free needle's mass and inertia with zero assumed
    external drive support. Its lagged compliant head can differ from the
    needle tail during force convergence; diagnostics report that discrepancy.
    This is not a welded gripper/needle constraint or a prescribed needle pose.

    ``tool_tissue`` gives the tool/tissue exchange of the scene: a
    :class:`ToolTissueSetup` (what ``tool_tissue_setup`` makes of a scene with
    tools), from which this object builds the one :class:`ToolTissueExchange`
    with the surface that the thread's tissue coupling refits (:attr:`exchange`).
    Without it (the default) the tool capsules exchange no force with the tissue.
    A setting of :func:`tool_tissue_law` is taken as well and made a setup here
    (``ToolTissueSetup.of_scene``), with ``tool_tissue_radii`` (the radius of the
    contact of each tool capsule [m], in the order of the tools' capsules;
    ``None`` takes the capsule radii) and, for the swept law,
    ``tool_effective_mass`` (one mass per tool [kg], or one for all).

    With the first-version law (``True`` or "first": frictionless contact
    between the tool capsules and the tissue surface, :class:`ToolTissueContact`,
    :attr:`tool_tissue`) the forces are updated in the accepted step of this
    object, at the surface that the thread's tissue coupling has refit, and are
    applied with the thread's loads, the tissue forces and the opposite wrenches
    on the tool bodies alike. ``scene.args`` then also supplies poisson_ratio and
    density.

    With the contact on the sweep of the capsules ("swept":
    :class:`SweptToolTissueContact`, :attr:`swept_tool_tissue`) the update is
    the last launch of the accepted step of this object, in the same graph: at
    the accepted state and the surface that the thread's tissue coupling has
    refit it places the tissue nodes that the capsules reach in the next
    interval, writes their holds into the tissue matrix and stores the reaction
    on the tools. :meth:`add_forces` adds that reaction to the body forces and,
    after the thread's loads, the forces that balance the held nodes to the
    nodal forces of the tissue solve. The scene then also supplies ``solver``
    (the tissue solver), ``fixed_vertex_indices`` and ``sim_dt`` (its full
    interval [s]).

    ``thread_tracks`` makes the thread follow the needle through the tissue: the
    track that the needle/tissue interaction of the scene records
    (``scene.interaction``) becomes eyelets of the tissue coupling, one group
    per passage, while the needle goes in (:class:`NeedleTrackEyelets`,
    :attr:`track`), and the coupling threads an eyelet in the accepted step in
    which the needle's tail has gone past it (``ThreadTissueCoupling.follow_tail``).
    ``None`` (the default) takes ``scene.args.thread_tracks``; a scene whose
    arguments have no such entry gets no tracks, and neither does ``False``:
    the solver then has no eyelet slot and the thread stays on the surface.
    Eyelet slots and tracks are a reserve that is fixed here (:attr:`eyelet_reserve`):
    with ``p = scene.plan.expected_passages`` passages per bite, ``L`` the length
    of the needle and ``s`` the eyelet spacing, a bite makes at most
    ``floor(L / s) + 2 p`` eyelets (a passage has one at each end, and its track
    is no longer than the needle), in ``p`` tracks. An eyelet beyond the reserve
    is not made and is counted (``eyelets_refused`` of :meth:`diagnostics`).
    """

    samples_per_element = 2
    density = 1140.0
    gravity = 9.81
    mass_damping = 2.0
    # Rest distance between two eyelets of one passage of the needle's track [m]: a choice, the
    # default of the suture example (--eyelet-spacing). What bounds it: closer than the mean tissue
    # edge (1.31 mm in the dVRK scene) two eyelets load the same nodes; further apart than
    # sqrt(8 R r), with R the radius of the needle and r that of the thread (3.37 mm there), the
    # chord between two eyelets leaves the needle's circle by more than the thread's radius.
    eyelet_spacing = 0.0015
    # Bites (the needle once in and out of the tissue) whose eyelets and tracks the reserve holds:
    # a choice, room for a second bite after the first.
    track_bites = 2

    def __init__(
        self,
        scene,
        config,
        *,
        capture=True,
        tool_tissue=False,
        tool_tissue_radii=None,
        tool_effective_mass=None,
        thread_tracks=None,
    ):
        self.scene, self.config = scene, config
        model, geometry = scene.model, scene.geometry
        self.device = model.device
        self.radius = float(config.thread_radius_m)
        self.length = float(config.thread_length_m)
        self.elements = config.thread_elements
        self.young_modulus = float(config.thread_modulus_pa)
        self.iteration_limit = int(getattr(config, "thread_iterations", 2))
        if isinstance(self.elements, bool) or not isinstance(self.elements, (int, np.integer)) or self.elements < 2:
            raise ValueError("thread_elements must be an integer at least two")
        if (
            not np.isfinite([self.radius, self.length, self.young_modulus]).all()
            or min(self.radius, self.length, self.young_modulus) <= 0
        ):
            raise ValueError("Thread dimensions and modulus must be finite and positive")
        if self.iteration_limit < 1:
            raise ValueError("thread_iterations must be positive")
        self.tail = np.asarray(geometry.tail, dtype=float)
        tangent = np.asarray(geometry.centerline[-1] - geometry.centerline[-2], dtype=float)
        if self.tail.shape != (3,) or not np.isfinite(self.tail).all() or np.linalg.norm(tangent) <= 1e-12:
            raise ValueError("Needle tail and tangent must be finite and nondegenerate")
        self.tangent = tangent / np.linalg.norm(tangent)
        full = np.asarray(scene.reference.vertices, dtype=float)
        global_ids = np.asarray(scene.reference.global_vertex_indices)
        active_global = np.asarray(scene.selection.global_vertex_indices)
        ids = np.searchsorted(global_ids, active_global)
        if np.any(ids >= len(global_ids)) or not np.array_equal(global_ids[ids], active_global):
            raise ValueError("Every active tissue vertex must have an exact full-pad global identity")
        self.active_reference_indices = ids
        current = scene.state_0.particle_q.numpy()
        if current.shape != (len(ids), 3):
            raise ValueError("Worker thread needs one accepted particle per active FEM vertex")
        pad = full.copy()
        pad[ids] = current
        poses = scene.state_0.body_q.numpy()
        pose = poses[scene.needle_body]
        transform = wp.transform(wp.vec3(pose[:3]), wp.quat(pose[3:]))
        head_point = np.asarray(wp.transform_point(transform, wp.vec3(self.tail)), dtype=float)
        head_tangent = np.asarray(wp.quat_rotate(wp.quat(pose[3:]), wp.vec3(self.tangent)), dtype=float)
        head_tangent /= np.linalg.norm(head_tangent)
        # The capsules that the thread can touch: the needle's centre line, then every tool's.
        centerline = np.asarray(geometry.centerline, dtype=float)
        bodies = [scene.needle_body] * (len(centerline) - 1)
        endpoints = [np.stack((a, b)) for a, b in zip(centerline[:-1], centerline[1:], strict=True)]
        radii = [float(geometry.thickness)] * len(bodies)
        for grasper in scene.graspers:
            bodies.extend(grasper.capsule_bodies)
            endpoints.extend(grasper.capsule_endpoints)
            radii.extend(grasper.capsule_radii)
        # Where the spare thread is laid (loose_thread_coordinates): on the pad's surface, behind the planned
        # stitch, in turns of the thread's elasto-gravity length (E I / (rho A g))^(1/3) with I / A = r² / 4,
        # its lead clear of those capsules where they are now.
        world = np.array(
            [
                [
                    wp.transform_point(wp.transform(wp.vec3(poses[body][:3]), wp.quat(poses[body][3:])), wp.vec3(end))
                    for end in ends
                ]
                for body, ends in zip(bodies, endpoints, strict=True)
            ],
            dtype=float,
        ).reshape(-1, 2, 3)
        plan = getattr(scene, "plan", None)
        stitch = None
        if getattr(plan, "entry", None) is not None and getattr(plan, "exit", None) is not None:
            stitch = (plan.entry.reference_position, plan.exit.reference_position, float(geometry.radius))
        initial = loose_thread_coordinates(
            np.r_[head_point, head_tangent],
            pad,
            length=self.length,
            elements=self.elements,
            radius=self.radius,
            triangles=scene.reference.exterior_surface_triangles,
            stitch=stitch,
            turn_radius=(self.young_modulus * self.radius**2 / (4.0 * self.density * self.gravity)) ** (1.0 / 3.0),
            obstacles=(world[:, 0], world[:, 1], np.asarray(radii, dtype=float)),
        )
        straight = np.zeros((self.elements + 1, 3))
        straight[:, 0] = np.linspace(0, self.length, self.elements + 1)
        self.thread_model = ThreadModel.create(
            straight, young_modulus=self.young_modulus, radius=self.radius, density=self.density, device=self.device
        )
        tetrahedra = np.asarray(scene.selection.indices, dtype=np.int32)
        edges = (
            np.asarray(scene.selection.vertices)[tetrahedra[:, 1:]]
            - np.asarray(scene.selection.vertices)[tetrahedra[:, :1]]
        )
        self.tissue_edge_m = float(np.linalg.norm(edges, axis=-1).mean())
        self.contact_stiffness = float(scene.args.young_modulus) * self.tissue_edge_m
        self.friction = float(scene.args.shaft_friction)
        if not np.isfinite(self.contact_stiffness) or self.contact_stiffness <= 0:
            raise ValueError("Thread contact needs a finite positive tissue modulus/edge scale")
        element_length = self.length / self.elements
        node_weight = self.density * math.pi * self.radius**2 * element_length * self.gravity
        if thread_tracks is None:
            thread_tracks = getattr(scene.args, "thread_tracks", False)
        self.thread_tracks = bool(thread_tracks)
        # Without tracks: no eyelet slot and one (unused) track, as before.
        self.eyelet_reserve, eyelets = (0, 1), {}
        if self.thread_tracks:
            passages = int(scene.plan.expected_passages)
            pieces = np.diff(np.asarray(geometry.centerline, dtype=float), axis=0)
            needle_length = float(np.linalg.norm(pieces, axis=1).sum())
            if passages < 1 or not np.isfinite(needle_length) or needle_length <= 0:
                raise ValueError("Thread tracks need a positive number of planned passages and a needle centre line")
            per_bite = int(np.floor(needle_length / self.eyelet_spacing)) + 2 * passages
            self.eyelet_reserve = (self.track_bites * per_bite, self.track_bites * passages)
            # The eyelet spring is the contact spring: the relaxation of the reaction on the tissue is unchanged.
            eyelets = {"eyelet_stiffness": self.contact_stiffness}
        self.solver = _WorkerSolverThread(
            model,
            self.thread_model,
            gravity=(0, 0, -self.gravity),
            mass_damping=self.mass_damping,
            max_iterations=self.iteration_limit,
            tolerance=0.1 * node_weight,
            contact_distance=self.radius,
            contact_stiffness=self.contact_stiffness,
            contact_smoothing=0.1 * self.radius,
            friction=self.friction,
            self_contact=True,
            max_eyelets=self.eyelet_reserve[0],
            **eyelets,
        )
        self.solver.q.assign(initial.reshape(-1))
        self.solver.boundary_q.assign(self.solver.q)
        self.attachment = ThreadAttachment(
            self.solver, model, [scene.needle_body], self.tail, self.tangent, lagged=True
        )
        # Deliberately no support from the old needle servo: the gripper acts
        # on this independent free body through actual MuJoCo contact alone.
        self.tissue = ThreadTissueCoupling(
            self.solver,
            full,
            scene.reference.exterior_surface_triangles,
            tetrahedra,
            ids,
            tissue_stiffness=0.25 * self.contact_stiffness,
            contact_margin=0.005,
            max_tracks=self.eyelet_reserve[1],
        )
        # The needle's actual track as eyelets of the coupling, threaded where the tail has gone past:
        # behind the tail by at most one eyelet spacing, and beside its line by at most two wire radii,
        # the radius within which the needle/tissue interaction takes a point for the needle's track.
        self.track = None
        if self.thread_tracks:
            self.tissue.follow_tail(
                scene.needle_body, self.tail, self.tangent, self.eyelet_spacing, 2.0 * float(geometry.thickness)
            )
            self.track = NeedleTrackEyelets(self.tissue, scene, self.eyelet_spacing)
        self.rigid_contact = ThreadRigidCapsuleContact(
            self.solver,
            model,
            bodies,
            endpoints,
            radii,
            thread_radius=self.radius,
            friction=self.friction,
            skip_by_body={scene.needle_body: 2},
        )
        # What tools and tissue exchange: one object, from the setup of the scene; None without a contact.
        setup = tool_tissue
        if not isinstance(setup, ToolTissueSetup):
            law = tool_tissue_law(tool_tissue)
            setup = None
            if law is not None:
                setup = ToolTissueSetup.of_scene(
                    scene, law, effective_mass=tool_effective_mass, contact_radii=tool_tissue_radii
                )
        self.exchange = None
        if setup is not None:
            self.exchange = ToolTissueExchange(
                model,
                scene.solver if setup.law == "swept" else None,
                self.tissue.surface,
                setup,
                self.solver.step_time.seconds_f64,
                particle_q=scene.state_0.particle_q,
            )
        self._nodal_force = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self._reaction_squares = wp.empty(model.particle_count, dtype=wp.vec3, device=self.device)
        # Reductions that a report reads in one go: the largest nodal reaction and whether one is
        # undefined (maximum_tissue_reaction), then the threaded eyelets and the thread inside the tissue.
        self._largest_reaction = wp.zeros(4, dtype=float, device=self.device)
        self._positions = wp.empty(self.elements * self.samples_per_element + 1, dtype=wp.vec3, device=self.device)
        self.accepted_steps, self.physical_time_s = 0, 0.0
        self.capture = bool(capture)
        self._step_graphs = OrderedDict()
        self._graph_warmed = False
        self.graph_captures = self.graph_launches = 0

    @property
    def tool_tissue(self):
        """The first-version law (:class:`ToolTissueContact`) of the exchange, or None."""
        exchange = self.exchange
        return exchange.law if exchange is not None and exchange.law_name == "first" else None

    @property
    def swept_tool_tissue(self):
        """The contact on the sweep of the capsules (:class:`SweptToolTissueContact`) of the exchange, or None."""
        exchange = self.exchange
        return exchange.law if exchange is not None and exchange.law_name == "swept" else None

    @property
    def tool_tissue_radii(self):
        """The radius with which each tool capsule touches the tissue [m], or None without a contact."""
        return None if self.exchange is None else list(self.exchange.contact_radii)

    @property
    def tool_effective_mass(self):
        """The effective mass of each tool [kg] that the swept contact holds, or None without it."""
        exchange = self.exchange
        if exchange is None or exchange.effective_mass is None:
            return None
        return [float(mass) for mass in exchange.effective_mass]

    @property
    def state(self):
        """Mutable device arrays for accepted-state snapshots/capture warm-up."""
        arrays = (*self.solver.state, *self.tissue.state, self.attachment.wrench, self._nodal_force)
        if self.exchange is not None:
            arrays = (*arrays, *self.exchange.state)
        return arrays

    def add_forces(self, body_f, external_nodal_forces):
        """Add stored reactions; does not advance thread or friction history."""
        self.attachment.add_forces(body_f)
        self.rigid_contact.add_forces(body_f)
        if self.exchange is not None:
            self.exchange.add_body_loads(body_f)
        wp.launch(
            _add_nodal,
            self.scene.model.particle_count,
            inputs=[self._nodal_force, external_nodal_forces],
            device=self.device,
        )
        if self.exchange is not None:
            # The held nodes' share of the right-hand side of the tissue solve is then only their change.
            self.exchange.add_nodal_loads(external_nodal_forces)

    def invalidate_step_graphs(self):
        """Call after replacing worker storage or changing contact/material settings.

        Accepted scene buffers and Newton stopping settings are tracked
        automatically. The time step is device data that every launch reads, so
        a graph serves every ``dt``. In-place state updates need no invalidation.
        The next real accepted step warms resources without a discarded solve.
        """
        note_release(len(self._step_graphs))
        self._step_graphs.clear()
        self._graph_warmed = False

    def _capture_refused(self, arrays):
        """Whether the accepted step of these accepted-state arrays runs as written, without a graph."""
        return (
            not self.capture
            or not self.device.is_cuda
            or not wp.is_conditional_graph_supported()
            or active_tape() is not None
            or getattr(self.scene.model, "requires_grad", False)
            or any(array.requires_grad for array in (*arrays, *self.solver.state))
        )

    def _accepted_arrays(self, accepted):
        """The arrays of an accepted state that the accepted step reads or writes: a graph is bound to them."""
        arrays = (accepted.body_q, accepted.body_qd, accepted.particle_q)
        if self.exchange is not None and self.exchange.moves_tissue:
            # The swept contact places the held nodes of the accepted state and sets their velocity to zero.
            arrays = (*arrays, accepted.particle_qd)
        return arrays

    def _graph_key(self, arrays):
        # The interval is not part of the key: the graph reads it from device data.
        return (
            self.solver.max_iterations,
            self.solver.tolerance,
            tuple((array.ptr, array.shape, array.strides, array.dtype) for array in arrays),
        )

    def _keep_graph(self, key, graph, arrays):
        self._step_graphs[key] = (graph, arrays)
        if len(self._step_graphs) > 16:
            self._step_graphs.popitem(last=False)
            note_release()
        self.graph_captures += 1

    def _advance_device(self, accepted, dt):
        """Device-only accepted-step operations, shared by eager and captured paths."""
        self.attachment.attach(accepted.body_q, dt)
        self.rigid_contact.update(accepted.body_q, accepted.body_qd, dt)
        self.tissue.update(accepted.particle_q, accepted.body_q)
        self.solver.step(None, None, None, None, dt)
        self.attachment.update()
        self.rigid_contact.react(accepted.body_q)
        self._nodal_force.zero_()
        self.tissue.add_forces(self._nodal_force)
        if self.exchange is not None:
            # At the accepted state: the surface is the one that tissue.update() refit above. Last, after
            # everything that reads the accepted tissue: the swept contact moves the nodes that are held
            # through the next interval. The first law adds its forces on the tissue to the thread's.
            self.exchange.update(
                accepted.body_q,
                accepted.body_qd,
                accepted.particle_q,
                accepted.particle_qd,
                nodal_force=self._nodal_force,
            )

    def advance(self, dt):
        """Advance once after the caller accepts a rigid/FEM interval of dt.

        CUDA replay keeps the conditional thread Newton loop on the device;
        eager execution otherwise polls its active flag from Python. Capture
        records operations without an extra physical step or friction update.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Accepted thread interval must be finite and positive")
        # Host preparation, outside capture: the kernels of the step, eager, captured or
        # replayed, read the interval from the thread solver's step time.
        self.solver.step_time.set(dt)
        if self.track is not None:
            # Rows that the needle laid in the interval just accepted become eyelets: host lists to
            # device rows in place, before the step that reads them, whichever way it runs below.
            self.track.advance()
        accepted = self.scene.state_0
        arrays = self._accepted_arrays(accepted)
        if self._capture_refused(arrays):
            self._advance_device(accepted, dt)
        elif not self._graph_warmed:
            self._advance_device(accepted, dt)
            self._graph_warmed = True
        else:
            key = self._graph_key(arrays)
            entry = self._step_graphs.get(key)
            if entry is None:
                with ScopedCapture(device=self.device) as captured:
                    self._advance_device(accepted, dt)
                entry = (captured.graph, arrays)
                self._keep_graph(key, *entry)
            else:
                self._step_graphs.move_to_end(key)
            # Keep the contact object's lightweight Python metadata consistent
            # when alternating previously captured accepted-state buffers.
            self.rigid_contact._body_q, self.rigid_contact._dt = accepted.body_q, float(dt)
            wp.capture_launch(entry[0])
            self.graph_launches += 1
        self.accepted_steps += 1
        self.physical_time_s += dt

    def capture_graphs(self, states, dt):
        """Leave a captured and launched step graph for each accepted-state buffer; for start-up.

        ``states`` are the buffers that will hold accepted states (the scene alternates two) and
        ``dt`` an interval the steps will use [s]. Every buffer must hold a valid accepted state:
        each pass below is a real accepted step from it. If the worker never advanced, the step
        of the first buffer runs once as written, which creates what a capture may not create.
        Then the step of each buffer without a graph is captured and its graph launched once,
        which is where Warp builds the executable. After every pass the thread state and the
        update counts of the thread solver get back the bits they had before this call, so the
        thread has not advanced: ``accepted_steps``, ``physical_time_s``, ``graph_launches``,
        the interval of the thread solver's step time and the host notes of the rigid contact
        keep their values. With the swept tool/tissue contact the tissue of every buffer (the
        nodes that a pass placed) and the holds in the tissue matrix get back what they had as
        well. ``graph_captures`` counts the captures. :meth:`advance` keeps its own
        rule (first step as written, then one capture per buffer) for buffers not given here.

        Returns:
            False if the steps run as written (the refusals of :meth:`advance`); nothing is
            issued then.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Accepted thread interval must be finite and positive")
        states = tuple(states)
        bindings = [(accepted, self._accepted_arrays(accepted)) for accepted in states]
        if any(self._capture_refused(arrays) for _, arrays in bindings):
            return False
        if self.device.is_capturing:
            raise RuntimeError("Capture the thread step graphs outside CUDA graph capture")
        step_time, contact = self.solver.step_time, self.rigid_contact
        interval, notes = step_time.value, (contact._body_q, contact._dt)
        owned = (*self.state, self.solver.iterations)
        exchange = self.exchange
        if exchange is not None and exchange.moves_tissue:
            # A pass places tissue nodes of its buffer and writes their holds.
            for accepted in states:
                owned = (*owned, accepted.particle_q, accepted.particle_qd)
        saved = [wp.clone(array) for array in owned]

        def restore():
            for array, values in zip(owned, saved, strict=True):
                wp.copy(array, values)
            if exchange is not None:
                # The tissue matrix follows the held flags that were just put back.
                exchange.restore_holds()

        try:
            # Host preparation, outside capture, as in advance().
            step_time.set(dt)
            if not self._graph_warmed and bindings:
                self._advance_device(states[0], dt)
                self._graph_warmed = True
                restore()
            for accepted, arrays in bindings:
                key = self._graph_key(arrays)
                if key in self._step_graphs:
                    continue
                with ScopedCapture(device=self.device) as captured:
                    self._advance_device(accepted, dt)
                self._keep_graph(key, captured.graph, arrays)
                wp.capture_launch(captured.graph)
                restore()
        finally:
            restore()
            contact._body_q, contact._dt = notes
            if interval is not None:
                step_time.set(interval)
        return True

    def positions(self):
        """Actual cubic centreline samples [2*elements+1,3], copied to host."""
        wp.launch(
            _sample,
            len(self._positions),
            inputs=[self.solver.q, self.thread_model.lengths, self.samples_per_element, self._positions],
            device=self.device,
        )
        result = self._positions.numpy().copy()
        if not np.isfinite(result).all():
            raise RuntimeError("Physical worker thread has nonfinite centreline coordinates")
        return result

    def maximum_tissue_reaction(self):
        """Largest nodal reaction on the tissue [N], reduced on the device.

        The value is ``np.linalg.norm(nodal force, axis=-1).max(initial=0)`` bit for bit, read
        back as one number instead of the nodal array. It is the thread's reaction: with the
        tool/tissue contact the stored nodal forces also hold that contact's, so the thread's
        own applied force is reduced then.
        """
        return self._reductions()[0]

    def _reductions(self):
        """The reductions of a report, in one read: largest tissue reaction [N], threaded eyelets, thread inside [m].

        The eyelet numbers are those of the last accepted step (what its tissue update found):
        the count of threaded eyelets, and the rest length of thread that the coupling treats as
        inside the tissue, between the first and the last threaded eyelet of each track and from
        the thread's first node while a track is partly threaded.
        """
        count = self.scene.model.particle_count
        reaction = self._nodal_force if self.tool_tissue is None else self.tissue.applied_force
        self._largest_reaction.zero_()
        wp.launch(_square_components, count, inputs=[reaction, self._reaction_squares], device=self.device)
        wp.launch(_largest_norm, count, inputs=[self._reaction_squares, self._largest_reaction], device=self.device)
        if self.solver.max_eyelets:
            wp.launch(
                _thread_in_tracks,
                self.eyelet_reserve[1],
                inputs=[*self.tissue.track_spans, self.length / self.elements, self._largest_reaction],
                device=self.device,
            )
        largest, undefined, threaded, inside = self._largest_reaction.numpy()
        return (float("nan") if undefined else float(largest)), int(round(float(threaded))), float(inside)

    def diagnostics(self, *, needle_pose=None):
        q, qd = self.solver.q.numpy().reshape(-1, 6), self.solver.qd.numpy().reshape(-1, 6)
        pose = self.scene.state_0.body_q.numpy()[self.scene.needle_body] if needle_pose is None else needle_pose
        actual_tail = np.asarray(
            wp.transform_point(wp.transform(wp.vec3(pose[:3]), wp.quat(pose[3:])), wp.vec3(self.tail))
        )
        reaction, threaded, inside = self._reductions()
        values = {
            "scope": "physical worker ANCF thread; lagged relaxed rigid/FEM forces; "
            + (
                "eyelets along the needle's actual track, threaded where the needle's tail has passed"
                if self.thread_tracks
                else "no scheduled eyelets"
            ),
            "physical_time_s": self.physical_time_s,
            "accepted_steps": self.accepted_steps,
            "graph_captures": self.graph_captures,
            "graph_launches": self.graph_launches,
            "radius_m": self.radius,
            "length_m": self.length,
            "elements": int(self.elements),
            "young_modulus_pa": self.young_modulus,
            "density_kg_m3": self.density,
            "newton_iteration_limit": self.iteration_limit,
            "last_newton_updates": self.solver.iterations.numpy().tolist(),
            "contact_stiffness_N_m": self.contact_stiffness,
            "contact_friction": self.friction,
            "head_to_needle_tail_error_m": float(np.linalg.norm(q[0, :3] - actual_tail)),
            "maximum_speed_m_s": float(np.linalg.norm(qd[:, :3], axis=-1).max()),
            "maximum_chord_stretch_ratio": float(
                np.max(np.linalg.norm(np.diff(q[:, :3], axis=0), axis=-1) / (self.length / self.elements))
            ),
            "needle_attachment_wrench_N_Nm": self.attachment.wrench.numpy()[0].tolist(),
            "maximum_tissue_reaction_N": reaction,
            "needle_drive_support_assumed": False,
            # Eyelets made from the needle's track so far, per passage and in all; those the thread passes
            # through; those that found no room in the reserve (slots, tracks); the rest length of thread
            # inside the tissue [m]. Without thread tracks all are zero.
            "eyelets": self.tissue.point_count,
            "eyelet_passages": [] if self.track is None else [len(slots) for slots in self.track.passages],
            "eyelets_threaded": threaded,
            "eyelets_refused": self.tissue.refused_points,
            "eyelet_reserve": list(self.eyelet_reserve),
            "thread_inside_tissue_m": inside,
            "gripper_capsules": sum(len(g.capsule_bodies) for g in self.scene.graspers),
            "rigid_contact": self.rigid_contact.diagnostics(),
        }
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise RuntimeError("Nonfinite physical thread coordinates or velocities")
        return values
