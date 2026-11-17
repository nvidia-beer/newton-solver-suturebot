# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-budget backward Euler integrator for a batch of suture threads.

A thread is a chain of two-node cubic Hermite cable elements with axial Green
strain and curvature bending (:class:`ThreadModel`). One step reads nothing
back to the host, so a CUDA graph can replay it:

- Time integration is backward Euler. A slack thread buckles, which makes
  Newton's method slow, and a second-order step that stops before it has
  converged passes an acceleration error to its next predictor: such a thread
  vibrates without end. A backward Euler step is the minimum of a step
  potential, and every iterate that lowers that potential is a stable step.
  The iteration can therefore stop at a fixed budget; a step cut short only
  adds numerical damping.
- Newton iterations run until every chain's residual is below a tolerance or
  the budget is used, tested on the device: the loop is a conditional graph
  node, and a chain that has converged does no further work. Iterations start
  from the velocity extrapolation ``q + dt qd``. The kernels read the time step
  from device arrays (``step_time``), so a graph serves every ``dt``; the host
  writes them before the launch, outside the capture. Each chain takes the multiple
  of its Newton update, among a fixed few that include zero, with the lowest
  step potential, so that potential never increases.
- The forces are exact; the iteration matrix is positive definite by
  construction. It keeps the axial geometric stiffness only in tension and the
  Gauss-Newton part of the bending tangent, because a slack thread in
  compression makes the exact tangent indefinite. The converged state is
  unchanged.
- A direct solve. Elements couple only neighbouring nodes, so each chain's
  tangent is block tridiagonal with one 6-by-6 block per node. On a CPU one
  kernel thread per chain eliminates the blocks from the far end toward node
  0, and the chains of a batch are eliminated in parallel. On a GPU cyclic
  reduction eliminates the two neighbours of every node at once, at a stride
  that doubles from level to level.

A node touches the world through its own contact plane: by default the floor
z = 0, or a point and a normal that the caller writes every step, for example
from a deforming surface. Eyelets are points in space that the thread must
pass through but may slide along; each tracks the nearest point of its chain
and pulls it in across the thread with a spring.

Node 0 of every chain is its attachment node. Its position and slope are
either prescribed every step, or carried by a rigid head: a body with a given
inertia that moves, by a translation and a rotation, about a predicted pose.
The head's motion is then an unknown of the same Newton solve, with the
quadratic cost of moving the inertia away from the prediction in the step
potential. A light body on a thread that switches between slack and taut is
solved with the thread instead of being fed its force a step late.
"""

from collections.abc import Sequence

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverBase

from ...utils.step_time import StepTime
from ..fem.differentiation import active_tape
from . import kernels
from .gauss import ParallelGaussElements
from .kernels import Mat66d, Vec6d
from .model import ThreadModel
from .self_contact_pairs import PairwiseSelfContacts

# Largest number of nodes in a batch that a GPU solves by cyclic reduction. Measured on an NVIDIA GB10 with
# chains of 91 nodes: 64 chains take 0.65 ms against 2.6 ms for the serial elimination, 256 chains 3.4 ms
# against 2.7 ms. The reduction pays while the device has idle threads.
CYCLIC_REDUCTION_NODES = 16384
_ORIGINAL_SELF_CONTACTS = kernels.find_self_contacts


class SolverThread(SolverBase):
    """Implicit backward Euler integrator for a batch of identical thread chains, without host readbacks.

    One step can be captured in a CUDA graph; its Newton loop is a conditional
    graph node that runs until every chain has converged or used
    ``max_iterations``. The state lives on the solver. Arrays are flat over
    the batch, chain by chain.

    Args:
        model: Newton model used for the device and, by default, gravity.
        thread: One chain. Node 0 is the attachment node.
        n_envs: Number of identical, independent chains solved as one batch.
        fixed_dofs: Further prescribed coordinates of one chain, for example
            the three position coordinates of its last node. Node 0 is always
            prescribed.
        gravity: Gravity [m/s²]; defaults to the model's.
        mass_damping: Mass-proportional damping rate [1/s].
        max_iterations: Cap on Newton updates per step. A step that stops at
            the cap is stable; it damps the motion more than a converged one.
        tolerance: Largest accepted free residual, slope rows divided by the
            mean element length [N].
        contact_distance: Distance kept between a node centre and its contact
            plane [m], usually the thread radius, or ``None`` for no contact.
            Node 0 is carried by its support and has no contact of its own.
        contact_stiffness: Contact penalty stiffness per node [N/m].
        contact_smoothing: Width of the smooth contact onset [m].
        friction: Coulomb friction coefficient of the contact planes and of
            the eyelets; zero for none. A node in contact is tied to an anchor
            on its plane by a spring of the contact stiffness. The spring's
            force stops at the coefficient times the node's normal force of
            the previous step; beyond that the node slides and its anchor
            follows. An eyelet holds a point of the thread in the same way
            along the thread, with the eyelet stiffness and the force across
            the thread as the normal force.
        max_eyelets: Number of eyelet slots.
        eyelet_stiffness: Spring between an eyelet and the thread [N/m].
        self_contact: Keep stretches of a chain from passing through each
            other. Every element is held half a contact distance on its side
            of the plane midway to the nearest other stretch, with the
            contact stiffness and without friction. The stretches are tested
            as chords, at the start of the step.
        obstacle_distance: Distance at which every element is kept from an
            obstacle point of its own [m], or ``None`` for no obstacles.
        multiples: Multiples of the Newton update that the line search
            tries; each chain takes the one with the lowest step potential.
            The positive semidefinite iteration matrix makes the update too
            short while a slack thread buckles, hence the multiples above
            one. A straight slack thread held at one end has no stiffness
            across itself, so the update of a load across it stretches it;
            only a small multiple then lowers the potential, hence the small
            ones. Zero keeps the iterate.

    Attributes:
        q, qd: Generalized positions and velocities, ``n_envs * dof_count``
            entries.
        boundary_q: Prescribed values, read on fixed coordinates. Write the six
            coordinates of each chain's node 0 before each step.
        fixed_dofs: Prescribed coordinates of one chain; change them between
            steps with :meth:`set_prescribed`.
        rigid_head: When true, node 0 of every chain is carried by a rigid
            head. Before each step write ``head_center``, ``head_arm`` and
            ``head_slope`` (the head's predicted centre, its lever from the
            centre to node 0, and node 0's slope, in world axes), node 0's
            matching coordinates in ``boundary_q``, and ``head_inertia`` (the
            head's 6-by-6 mass and inertia divided by dt², for its translation
            and its rotation about the centre). ``head_motion`` then returns
            the translation and rotation vector the head made from the
            predicted pose. Its previous value is the first iterate of the
            next step.
        contact_distance, contact_stiffness, contact_smoothing: The contact
            parameters given at construction.
        contact_force: Force of its contact plane on every node after the
            step, normal and friction [N].
        contact_anchor: Friction anchor of every node [m]. The solver keeps
            it; write it before a step to carry the anchors along with a
            moving surface.
        contact_limit: Largest friction force of every node in the coming
            step [N]; zero where a node has no grip.
        contact_point, contact_normal: A point of each node's contact plane [m]
            and its unit normal towards the free side, ``n_envs * node_count``
            entries. They start as the floor z = 0; write them before a step
            to follow another surface. A zero normal switches a node's contact
            off.
        eyelet_target: Position of each eyelet [m]; write it before each step.
        eyelet_element, eyelet_xi: The point of the batch's chains that each
            eyelet tracks: a global element index (chain * element_count +
            element), or -1 for an unused slot, and the coordinate in [0, 1]
            along that element. Write both to thread an eyelet; every step
            moves them to the nearest point of the chain, at most one element
            away.
        eyelet_held: Nonzero where the thread may not slide through its
            eyelet: the eyelet then holds the point of the chain it tracks,
            a knot.
        eyelet_force: Force of the thread on each eyelet after the step [N].
        eyelet_point: The thread's point at each eyelet after the step [m];
            the eyelet's own position where it is not threaded.
        obstacle_xi, obstacle_point, obstacle_normal: With an obstacle
            distance, one plane per element: the coordinate in [0, 1] of the
            element's point that is tested, a point of the plane [m] and its
            unit normal towards the free side. Write them before each step; a
            zero normal is no obstacle.
        obstacle_force: Force of its obstacle on every element after the
            step [N], acting at the tested point.
        residual: Step residual of every node at the last tested coordinates.
        iterations: Newton updates each chain took in the last step.
        step_time: The time step of the step as device data (``StepTime``).
            :meth:`step` and the attachment and contact objects of this solver
            write it; to replay a captured step at another ``dt``, call
            ``step_time.set(dt)`` before the launch.
    """

    def __init__(
        self,
        model: newton.Model,
        thread: ThreadModel,
        *,
        n_envs: int = 1,
        fixed_dofs: Sequence[int] = (),
        gravity=None,
        mass_damping: float = 0.0,
        max_iterations: int = 30,
        tolerance: float = 1.0e-8,
        contact_distance: float | None = None,
        contact_stiffness: float = 0.0,
        contact_smoothing: float = 1.0e-5,
        friction: float = 0.0,
        max_eyelets: int = 0,
        eyelet_stiffness: float = 0.0,
        self_contact: bool = False,
        obstacle_distance: float | None = None,
        multiples: Sequence[float] = (4.0, 2.0, 1.0, 0.5, 0.25, 1.0 / 16.0, 1.0 / 64.0, 0.0),
    ):
        super().__init__(model)
        if model.device != thread.device:
            raise ValueError("Newton model and thread must share a device")
        if not isinstance(n_envs, int) or n_envs < 1:
            raise ValueError("n_envs must be a positive integer")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if not np.isfinite(mass_damping) or mass_damping < 0:
            raise ValueError("mass_damping must be finite and nonnegative")
        if not np.isfinite(contact_stiffness) or contact_stiffness < 0:
            raise ValueError("contact_stiffness must be finite and nonnegative")
        if not np.isfinite(friction) or friction < 0:
            raise ValueError("friction must be finite and nonnegative")
        if not isinstance(max_eyelets, int) or max_eyelets < 0:
            raise ValueError("max_eyelets must be a nonnegative integer")
        if not np.isfinite(eyelet_stiffness) or eyelet_stiffness < 0:
            raise ValueError("eyelet_stiffness must be finite and nonnegative")
        if not np.isfinite(contact_smoothing) or contact_smoothing <= 0:
            raise ValueError("contact_smoothing must be finite and positive")
        fixed = np.asarray(fixed_dofs, dtype=np.int64).reshape(-1)
        if np.any(fixed < 0) or np.any(fixed >= thread.dof_count):
            raise ValueError("fixed_dofs is out of range")
        gravity = np.asarray(model.gravity.numpy()[0] if gravity is None else gravity, dtype=np.float64)
        if gravity.shape != (3,) or not np.isfinite(gravity).all():
            raise ValueError("gravity must be a finite three-vector")
        self.thread = thread
        self.n_envs = n_envs
        self.gravity = wp.vec3d(gravity)
        self.mass_damping = float(mass_damping)
        self.max_iterations, self.tolerance = max_iterations, float(tolerance)
        self._slope_scale = 1.0 / float(np.mean(thread.lengths.numpy()))
        self._contact = contact_distance is not None
        self.contact_distance = 0.0 if contact_distance is None else float(contact_distance)
        self.contact_stiffness = float(contact_stiffness)
        self.contact_smoothing = float(contact_smoothing)
        self.friction = float(friction)
        self.max_eyelets, self.eyelet_stiffness = max_eyelets, float(eyelet_stiffness)
        if self_contact and contact_distance is None:
            raise ValueError("self_contact needs the contact distance, stiffness and smoothing")
        if obstacle_distance is not None and (not np.isfinite(obstacle_distance) or obstacle_distance <= 0):
            raise ValueError("obstacle_distance must be finite and positive")
        self.self_contact = bool(self_contact)
        self.obstacle_distance = None if obstacle_distance is None else float(obstacle_distance)

        device = thread.device
        # The time step of the step being solved, read by the kernels from these arrays.
        self.step_time = StepTime(device)
        self._inertia_scale()
        self._mask = np.zeros(thread.dof_count, dtype=np.int32)
        self._mask[:6] = 1
        self._mask[fixed] = 1
        self.fixed_dofs = np.flatnonzero(self._mask).astype(np.int32)
        self._fixed = wp.array(self._mask, dtype=int, device=device)

        total_nodes, total_elements = n_envs * thread.node_count, n_envs * thread.element_count
        total_dofs = 6 * total_nodes

        def zeros(shape, dtype=wp.float64):
            return wp.zeros(shape, dtype=dtype, device=device)

        self.q = wp.array(np.tile(thread.rest_q.numpy(), n_envs), dtype=wp.float64, device=device)
        self.qd = zeros(total_dofs)
        self.boundary_q = wp.clone(self.q)
        self.contact_point = zeros(total_nodes, wp.vec3d)
        self.contact_normal = wp.array(
            np.tile((0.0, 0.0, 1.0 if self._contact else 0.0), (total_nodes, 1)), dtype=wp.vec3d, device=device
        )
        self.contact_force = zeros(total_nodes, wp.vec3d)
        self.contact_limit = zeros(total_nodes)
        self.contact_anchor = zeros(total_nodes, wp.vec3d)
        self.eyelet_target = zeros(max_eyelets, wp.vec3d)
        self.eyelet_element = wp.full(max_eyelets, -1, dtype=int, device=device)
        self.eyelet_xi = zeros(max_eyelets)
        self.eyelet_force = zeros(max_eyelets, wp.vec3d)
        self.eyelet_point = zeros(max_eyelets, wp.vec3d)
        self.eyelet_held = wp.zeros(max_eyelets, dtype=int, device=device)
        total_elements = n_envs * thread.element_count
        own = total_elements if self.self_contact else 0
        self._self_xi, self._self_point, self._self_normal = zeros(own), zeros(own, wp.vec3d), zeros(own, wp.vec3d)
        # Small batches leave the expensive ordered other-element scan almost
        # serial. Compute the same complete ordered pairs in parallel, then
        # select in the original index order. Larger/gradient cases retain the
        # original query and have no quadratic scratch allocation.
        self._self_pairs = (
            PairwiseSelfContacts(thread.node_count, n_envs, device)
            if self.self_contact
            and device.is_cuda
            and not getattr(model, "requires_grad", False)
            and thread.element_count <= 128
            and total_elements * thread.element_count <= 131_072
            else None
        )
        outer = total_elements if self.obstacle_distance is not None else 0
        self.obstacle_xi, self.obstacle_point = zeros(outer), zeros(outer, wp.vec3d)
        self.obstacle_normal, self.obstacle_force = zeros(outer, wp.vec3d), zeros(outer, wp.vec3d)
        self._eyelet_tangent = zeros(max_eyelets, wp.vec3d)
        self._eyelet_limit = zeros(max_eyelets)
        self._eyelet_anchor_element = wp.full(max_eyelets, -1, dtype=int, device=device)
        self._eyelet_anchor_xi = zeros(max_eyelets)
        self.residual = zeros(total_nodes, Vec6d)
        self._predicted_q = zeros(total_dofs)
        self._gradient = zeros((total_elements, 4), wp.vec3d)
        self._hessian = zeros((total_elements, 4, 4), wp.mat33d)
        # Small chains leave most GPU multiprocessors idle when one thread
        # performs all five quadrature points. Keep bounded persistent scratch
        # for independent points, followed by the original fixed-order sum.
        self._gauss_elements = (
            ParallelGaussElements(thread, total_elements)
            if device.is_cuda and total_elements <= 1024 and not getattr(model, "requires_grad", False)
            else None
        )
        self._diagonal, self._coupling = zeros(total_nodes, Mat66d), zeros(total_nodes, Mat66d)
        self._inverse, self._reduced = zeros(total_nodes, Mat66d), zeros(total_nodes, Vec6d)
        self._update = zeros(total_nodes, Vec6d)
        # Cyclic reduction, the chain solve on a GPU: every node's rows before and after a level, and the
        # same rows solved by their diagonal block.
        reduced = device.is_cuda and total_nodes <= CYCLIC_REDUCTION_NODES
        rows = (total_nodes if reduced else 0, 6, kernels.REDUCTION_COLUMNS)
        self._rows, self._reduced_rows, self._condensed = zeros(rows), zeros(rows), zeros(rows)
        self.rigid_head = False
        self.head_center, self.head_arm, self.head_slope = (zeros(n_envs, wp.vec3d) for _ in range(3))
        self.head_inertia, self.head_motion = zeros(n_envs, Mat66d), zeros(n_envs, Vec6d)
        self.iterations = zeros(n_envs, int)
        self._status, self._active, self._passes = zeros(n_envs, int), zeros(1, int), zeros(1, int)
        multiples = tuple(float(value) for value in multiples)
        if 1.0 not in multiples or 0.0 not in multiples or min(multiples) < 0.0:
            raise ValueError("multiples must be nonnegative and contain 1 and 0")
        self._scales = wp.array(multiples, dtype=wp.float64, device=device)
        self._unit = multiples.index(1.0)
        self._trial_q = zeros(len(multiples) * total_dofs)
        self._trial_energy = zeros((n_envs, len(multiples)))
        # Optional element contacts participate in the same Newton solve and
        # line search. The empty list preserves the existing obstacle API.
        self.element_contacts = []

    def _inertia_scale(self):
        """One ``float64`` on the device: the displacement coefficient of the consistent mass.

        Inertia M / dt² and damping c M / dt give 1 / dt² + c / dt, which the host forms
        from the time step and ``mass_damping`` as before; ``step_time`` keeps it current
        when the time step changes. The solver has one such array. A changed
        ``mass_damping`` refills it at the next step, outside capture; a step captured
        earlier holds the old ``mass_damping`` as a launch argument and must be
        captured again, as before.
        """
        damping = self.mass_damping
        return self.step_time.derived(
            "thread inertia scale", lambda dt: 1.0 / (dt * dt) + damping / dt, wp.float64, parameters=damping
        )

    def _newton_pass(self, dt: float) -> None:
        """Test every unfinished chain at its current coordinates and update the ones that have not converged."""
        # A comparison after step(); a different time step inside a capture raises.
        self.step_time.set(dt)
        thread, device = self.thread, self.thread.device
        nodes = self.n_envs * thread.node_count
        elements = self.n_envs * thread.element_count
        candidates = self._scales.shape[0]
        contact = [
            self._contact,
            self.contact_distance,
            self.contact_stiffness,
            self.contact_smoothing,
            self.contact_point,
            self.contact_normal,
            self.contact_limit,
            self.contact_anchor,
        ]
        eyelets = [self.eyelet_target, self.eyelet_element, self.eyelet_xi, self._eyelet_tangent, self.eyelet_stiffness]
        grip = [self._eyelet_limit, self._eyelet_anchor_element, self._eyelet_anchor_xi]
        # Planes on elements: the chain's own stretches and the caller's obstacles. Two stretches whose centres
        # are two contact distances apart touch, so each is kept one contact distance from the plane between them.
        planes = []
        if self.self_contact:
            planes.append([self._self_xi, self._self_point, self._self_normal, self.contact_distance])
        if self.obstacle_distance is not None:
            planes.append([self.obstacle_xi, self.obstacle_point, self.obstacle_normal, self.obstacle_distance])
        penalty = [self.contact_stiffness, self.contact_smoothing]
        masses = [thread.mass_diagonal, thread.mass_coupling]
        material = [thread.lengths, thread.axial_stiffness, thread.bending_stiffness, thread.rest_q]
        # Displacement coefficient of the consistent mass: inertia M / dt² and damping c M / dt.
        inertia_scale = self._inertia_scale()
        seconds = self.step_time.seconds_f64
        pose = [self.head_center, self.head_arm, self.head_slope]

        def linearize(tangent_blocks: bool) -> None:
            if (
                self._gauss_elements is not None
                and active_tape() is None
                and not any(array.requires_grad for array in (self.q, *material, self._gradient, self._hessian))
            ):
                self._gauss_elements.evaluate(self.q, tangent_blocks, self._status, self._gradient, self._hessian)
            else:
                wp.launch(
                    kernels.evaluate_elements,
                    dim=elements,
                    inputs=[self.q, *material, tangent_blocks, self._status],
                    outputs=[self._gradient, self._hessian],
                    device=device,
                )
            if self.max_eyelets:
                wp.launch(
                    kernels.add_eyelets,
                    dim=self.max_eyelets,
                    inputs=[self.q, thread.lengths, thread.node_count, *eyelets, tangent_blocks, self._status, *grip],
                    outputs=[self._gradient, self._hessian],
                    device=device,
                )
            for plane in planes:
                wp.launch(
                    kernels.add_obstacles,
                    dim=elements,
                    inputs=[self.q, thread.lengths, thread.node_count, *plane, *penalty, tangent_blocks, self._status],
                    outputs=[self._gradient, self._hessian],
                    device=device,
                )
            for contact_model in self.element_contacts:
                contact_model.add_gradient(tangent_blocks)
            wp.launch(
                kernels.assemble,
                dim=nodes,
                inputs=[
                    self.q,
                    self._predicted_q,
                    self.qd,
                    self._gradient,
                    self._hessian,
                    *masses,
                    self._fixed,
                    self._status,
                    tangent_blocks,
                    self.gravity,
                    seconds,
                    self.mass_damping,
                    inertia_scale,
                    *contact,
                    self.rigid_head,
                    *pose,
                    self.head_inertia,
                    self.head_motion,
                ],
                outputs=[self._diagonal, self._coupling, self.residual],
                device=device,
            )

        wp.launch(kernels.begin_pass, dim=1, outputs=[self._active, self._passes], device=device)
        # Forces only first: a chain that has converged needs no tangent.
        linearize(False)
        wp.launch(
            kernels.check,
            dim=self.n_envs,
            inputs=[
                thread.node_count,
                self.rigid_head,
                self.residual,
                self._slope_scale,
                self.tolerance,
                self.max_iterations,
                self._passes,
            ],
            outputs=[self._status, self.iterations, self._active],
            device=device,
        )
        linearize(True)
        system = [thread.node_count, self.rigid_head, self._status, self._diagonal, self._coupling, self.residual]
        if device.is_cuda and nodes <= CYCLIC_REDUCTION_NODES:
            # One GPU thread per chain is slow on the serial elimination, so every node is eliminated at once,
            # level by level. Every level costs about twice the arithmetic of the whole serial elimination:
            # the reduction pays while the device has idle threads, which a batch of some hundred chains
            # uses up. A block of one warp, 32 threads, lets the device spread a launch over its
            # multiprocessors; the threads of a full block share the memory and float64 units of one.
            warp = 32
            rows, reduced = self._rows, self._reduced_rows
            wp.launch(
                kernels.begin_reduction,
                dim=nodes,
                inputs=system,
                outputs=[rows],
                device=device,
                block_dim=warp,
            )
            stride = 1
            while stride < thread.node_count:
                wp.launch(
                    kernels.condense_nodes,
                    dim=(nodes, kernels.REDUCTION_COLUMNS),
                    inputs=[thread.node_count, stride, self._status, rows],
                    outputs=[self._condensed],
                    device=device,
                    block_dim=warp,
                )
                wp.launch(
                    kernels.reduce_nodes,
                    dim=(nodes, 6, kernels.REDUCTION_COLUMNS),
                    inputs=[thread.node_count, stride, self._status, rows, self._condensed],
                    outputs=[reduced],
                    device=device,
                    block_dim=warp,
                )
                rows, reduced = reduced, rows
                stride *= 2
            wp.launch(
                kernels.solve_reduced,
                dim=nodes,
                inputs=[thread.node_count, self.rigid_head, self._status, rows],
                outputs=[self._update],
                device=device,
                block_dim=warp,
            )
        else:
            # The serial elimination: the fastest solve on a CPU and for a large batch on a GPU, and the
            # reference of the one above. Blocks of one warp spread a GPU launch over its multiprocessors.
            wp.launch(
                kernels.solve_chains,
                dim=self.n_envs,
                inputs=system,
                outputs=[self._inverse, self._reduced, self._update],
                device=device,
                block_dim=32 if device.is_cuda else 256,
            )
        wp.launch(
            kernels.trial_coordinates,
            dim=(candidates, nodes),
            inputs=[
                self.q,
                self._update,
                self._scales,
                thread.node_count,
                self._status,
                self.rigid_head,
                *pose,
                self.head_motion,
            ],
            outputs=[self._trial_q],
            device=device,
        )
        wp.launch(kernels.clear, dim=self._trial_energy.size, inputs=[self._trial_energy.flatten()], device=device)
        wp.launch(
            kernels.trial_element_energy,
            dim=(candidates, elements),
            inputs=[self._trial_q, *material, nodes, self._status],
            outputs=[self._trial_energy],
            device=device,
        )
        if self.max_eyelets:
            wp.launch(
                kernels.trial_eyelet_energy,
                dim=(candidates, self.max_eyelets),
                inputs=[self._trial_q, thread.lengths, thread.node_count, nodes, *eyelets, self._status, *grip],
                outputs=[self._trial_energy],
                device=device,
            )
        for plane in planes:
            wp.launch(
                kernels.trial_obstacle_energy,
                dim=(candidates, elements),
                inputs=[self._trial_q, thread.lengths, thread.node_count, nodes, *plane, *penalty, self._status],
                outputs=[self._trial_energy],
                device=device,
            )
        for contact_model in self.element_contacts:
            contact_model.add_trial_energy()
        wp.launch(
            kernels.trial_node_energy,
            dim=(candidates, nodes),
            inputs=[
                self._trial_q,
                self._predicted_q,
                self.qd,
                *masses,
                self.gravity,
                seconds,
                self.mass_damping,
                *contact,
                nodes,
                self._status,
                self.rigid_head,
                self.head_inertia,
                self.head_motion,
                self._update,
                self._scales,
            ],
            outputs=[self._trial_energy],
            device=device,
        )
        wp.launch(
            kernels.apply_best,
            dim=nodes,
            inputs=[
                self._update,
                self._scales,
                self._trial_energy,
                thread.node_count,
                self._unit,
                self._status,
                self.rigid_head,
                *pose,
            ],
            outputs=[self.head_motion, self.q],
            device=device,
        )

    def step(
        self,
        state_in: newton.State | None,
        state_out: newton.State | None,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
    ) -> None:
        """Advance the solver's own state by one backward Euler step.

        The Newton state, control and contact arguments are unused; set
        ``boundary_q``, with a rigid head the head arrays, and any contact
        planes and eyelet targets on the solver before the call.

        The kernels read ``dt`` from ``step_time``. This call writes a changed
        ``dt`` there; inside a CUDA graph capture it only checks that the arrays
        hold ``dt``, and raises if they do not.
        """
        self.step_time.set(dt)
        self._inertia_scale()
        thread, device = self.thread, self.thread.device
        nodes = self.n_envs * thread.node_count
        eyelets = [self.eyelet_target, self.eyelet_element, self.eyelet_xi, self._eyelet_tangent]
        if self.max_eyelets:
            wp.launch(
                kernels.slide_eyelets,
                dim=self.max_eyelets,
                inputs=[
                    self.q,
                    thread.lengths,
                    thread.node_count,
                    self.eyelet_target,
                    self.eyelet_held,
                    self.eyelet_stiffness,
                    self._eyelet_limit,
                ],
                outputs=[
                    self.eyelet_element,
                    self.eyelet_xi,
                    self._eyelet_tangent,
                    self._eyelet_anchor_element,
                    self._eyelet_anchor_xi,
                ],
                device=device,
            )
        plane = [
            self.contact_distance,
            self.contact_stiffness,
            self.contact_smoothing,
            self.contact_point,
            self.contact_normal,
        ]
        elements = self.n_envs * thread.element_count
        if self.self_contact:
            # Ten smoothing widths beyond touching, the smooth onset of the
            # penalty has died away. Elements at most two apart are excluded.
            reach = 2.0 * (self.contact_distance + 10.0 * self.contact_smoothing)
            if (
                self._self_pairs is not None
                and kernels.find_self_contacts is _ORIGINAL_SELF_CONTACTS
                and active_tape() is None
                and not any(a.requires_grad for a in (self.q, self._self_xi, self._self_point, self._self_normal))
            ):
                self._self_pairs.find(self.q, reach, 2, self._self_xi, self._self_point, self._self_normal)
            else:
                wp.launch(
                    kernels.find_self_contacts,
                    dim=elements,
                    inputs=[self.q, thread.node_count, reach, 2],
                    outputs=[self._self_xi, self._self_point, self._self_normal],
                    device=device,
                    block_dim=1 if device.is_cuda and elements <= 128 else 256,
                )
        wp.launch(
            kernels.predict,
            dim=6 * nodes,
            inputs=[self.q, self.qd, self.boundary_q, self._fixed, self.step_time.seconds_f64],
            outputs=[self._predicted_q],
            device=device,
        )
        wp.launch(
            kernels.begin_step,
            dim=self.n_envs,
            inputs=[
                thread.node_count,
                self.rigid_head,
                self.head_center,
                self.head_arm,
                self.head_slope,
                self.head_motion,
            ],
            outputs=[self._status, self.iterations, self._active, self._passes, self.q],
            device=device,
        )
        wp.capture_while(self._active, self._newton_pass, dt=dt)
        wp.launch(
            kernels.finish,
            dim=6 * nodes,
            inputs=[self.q, self._predicted_q, self.step_time.seconds_f64],
            outputs=[self.qd],
            device=device,
        )
        if self._contact:
            wp.launch(
                kernels.contact_forces,
                dim=nodes,
                inputs=[self.q, thread.node_count, *plane, self.friction],
                outputs=[self.contact_limit, self.contact_anchor, self.contact_force],
                device=device,
            )
        if self.obstacle_distance is not None:
            wp.launch(
                kernels.obstacle_forces,
                dim=elements,
                inputs=[
                    self.q,
                    thread.lengths,
                    thread.node_count,
                    self.obstacle_xi,
                    self.obstacle_point,
                    self.obstacle_normal,
                    self.obstacle_distance,
                    self.contact_stiffness,
                    self.contact_smoothing,
                ],
                outputs=[self.obstacle_force],
                device=device,
            )
        if self.max_eyelets:
            wp.launch(
                kernels.eyelet_forces,
                dim=self.max_eyelets,
                inputs=[
                    self.q,
                    thread.lengths,
                    thread.node_count,
                    *eyelets,
                    self.eyelet_stiffness,
                    self.friction,
                    self._eyelet_anchor_element,
                    self._eyelet_anchor_xi,
                ],
                outputs=[self._eyelet_limit, self.eyelet_force, self.eyelet_point],
                device=device,
            )
        for contact_model in self.element_contacts:
            contact_model.finish_step()

    @property
    def state(self) -> tuple:
        """The arrays that carry the solver from one step to the next.

        Copy them to save a state and copy them back to restore it, for
        example around the trial steps of a graph capture.
        """
        return (
            self.q,
            self.qd,
            self.boundary_q,
            self.head_motion,
            self.contact_limit,
            self.contact_anchor,
            self.eyelet_element,
            self.eyelet_xi,
            self.eyelet_held,
            self._eyelet_limit,
            self._eyelet_anchor_element,
            self._eyelet_anchor_xi,
        ) + tuple(array for contact in self.element_contacts for array in contact.state)

    def set_prescribed(self, dofs: Sequence[int], prescribed: bool = True) -> None:
        """Prescribe or free coordinates of one chain, in every chain, from the next step on.

        A prescribed coordinate follows ``boundary_q``; write its value there
        before the next step. A freed coordinate keeps its velocity. The six
        coordinates of node 0 stay prescribed. The device reads the mask every
        step, so a captured graph follows the change.
        """
        dofs = np.asarray(dofs, dtype=np.int64).reshape(-1)
        if dofs.size and (dofs.min() < 6 or dofs.max() >= self.thread.dof_count):
            raise ValueError("dofs must be coordinates of one chain beyond node 0")
        self._mask[dofs] = 1 if prescribed else 0
        self.fixed_dofs = np.flatnonzero(self._mask).astype(np.int32)
        self._fixed.assign(self._mask)
