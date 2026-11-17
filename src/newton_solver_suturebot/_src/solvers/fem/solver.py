# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Linearly implicit Backward-Euler soft-body solver base.

Handles tetrahedral and/or hexahedral FEM elements based on what is present in
the model (``model.tet_count`` and ``self.hex_count``), plus springs, triangle
membranes, gravity and an optional analytic ground plane.

Design
======

The entire solver reduces to three ideas:

1. **Scalar energies.** Each elastic element defines a scalar strain-energy
   density ``ψ(x)``. The force is ``f = −∇ψ``; the system matrix uses an
   approximate positive-semidefinite tangent stiffness.

2. **Implicit time step.** Advancing the state is one linear solve of an SPD
   system ``A Δv = h f − h² K v``. ``A`` is mass plus a positive semidefinite
   stiffness approximation and assembled spring damping. This is one
   linearization, not a converged nonlinear Backward-Euler solve.

3. **Inequality constraints.** Contact is enforced either by a penalty
   (a steep added energy) or by projecting positions to satisfy it.

Mathematical overview
=====================

Each substep performs one Newton iteration of Backward Euler::

    A · Δv = h · f_n − h² · H_pot · v_n
    A = M + h² · H_pot + h · C        (PSD)

``M`` is the lumped mass matrix, ``H_pot`` the elastic stiffness approximation,
and ``C`` the assembled spring damping matrix. Material and mass-proportional
damping forces are currently evaluated explicitly. The legacy diagonal-only
triangle stiffness retains its existing integration approximation.

BSR triplet layout (built once at construction, never rebuilt)::

    [0 .. P)                           — per-particle diagonal (mass + spring sum)
    [P .. P+2·S)                       — per-spring (i,j) and (j,i) off-diagonals
    [P+2·S .. P+2·S+3·R)              — per-tri 3 lumped diagonal blocks
    [P+2·S+3·R .. P+2·S+3·R+16·T)    — per-tet 4×4 Hessian (rest-state, F=I)
    [... +64·H)                        — per-hex 8×8 Hessian blocks

where P, S, R, T, H = particle/spring/tri/tet/hex counts.
"""

import math
from collections import OrderedDict

import numpy as np
import warp as wp
from newton import Contacts, Control, Model, ParticleFlags, State
from newton.solvers import SolverBase
from warp.optim.linear import LinearOperator, aslinearoperator, bicgstab, cg, cr, gmres, preconditioner
from warp.sparse import bsr_set_from_triplets, bsr_zeros

from ...utils.capture import PauseGC
from ...utils.step_time import StepTime
from . import differentiation
from .fixed_particles import FixedParticles
from .implicit_predictor import restore_spring_damping_predictor, subtract_elastic_predictor
from .kernels import (
    block_jacobi_mv_kernel,
    build_system_matrix_diagonal_kernel,
    build_system_matrix_diagonal_mass_kernel,
    build_system_matrix_hex_kernel,
    build_system_matrix_sparse_kernel,
    build_system_matrix_tet_kernel,
    build_system_matrix_tri_kernel,
    compute_hex_volume_kernel,
    compute_volume_kernel,
    eval_gravity_from_array,
    eval_hexahedra,
    eval_linear_damping_kernel,
    eval_particle_ground_contacts,
    eval_springs,
    eval_springs_linear_and_torque,
    eval_tet_activations,
    eval_tetrahedra,
    eval_triangles,
    invert_block_diagonal_kernel,
    maximum_rhs_component,
    normalize_linear_rhs,
    restore_linear_increment,
    update_state,
)
from .soft_surface_contacts import SoftBodySurfaceContacts
from .sparse_update import accumulate_triplet_blocks, map_triplet_blocks

PARTICLE_FLAG_ACTIVE = int(ParticleFlags.ACTIVE)


@wp.kernel
def accumulate_scaled_kernel(
    src: wp.array[wp.vec3],
    scale: wp.array[wp.float32],
    dst: wp.array[wp.vec3],
):
    tid = wp.tid()
    dst[tid] = dst[tid] + scale[0] * src[tid]


@wp.kernel(enable_backward=False)
def _weighted_tet_volume(
    positions: wp.array[wp.vec3],
    indices: wp.array2d[int],
    weights: wp.array[float],
    volumes: wp.array[float],
):
    element = wp.tid()
    weight = weights[element]
    volumes[element] = 0.0
    if weight > 0.0:
        origin = positions[indices[element, 0]]
        first = positions[indices[element, 1]] - origin
        second = positions[indices[element, 2]] - origin
        third = positions[indices[element, 3]] - origin
        volumes[element] = weight * wp.abs(wp.dot(first, wp.cross(second, third))) / 6.0


class SolverFEM(SolverBase):
    """Linearly implicit Backward-Euler FEM soft-body solver.

    Handles tetrahedral and/or hexahedral Neo-Hookean FEM, springs, triangle
    membrane FEM, gravity, and an optional analytic ground plane.  Which
    element types are active is determined by ``model.tet_count`` and
    ``self.hex_count``; both may be non-zero simultaneously.

    Args:
        model: The Newton model owning the particles.
        dt: Default timestep [s] used for the initial matrix assembly.
        mass: Unused — kept for API compatibility. Per-particle masses are read
            from ``model.particle_mass`` (set by density during
            :meth:`~newton.ModelBuilder.add_soft_mesh`).
        preconditioner_type: Jacobi preconditioner type (``"diag"``).
        solver_type: Iterative solver — ``"cg"``, ``"bicgstab"``,
            ``"gmres"``, or ``"cr"``.
        linear_solver_maxiter: Maximum solver iterations per substep.
        linear_solver_tolerance: Relative residual tolerance for the fast-path
            linear solve. The differentiable path retains its checked solve.
        linear_solver_use_cuda_graph: Allow device-side early exit at the same
            residual tolerance on CUDA devices supporting conditional graphs.
            Disabling this keeps the fixed-budget, graph-capturable solve.
            CPU, unsupported devices and host residual checks retain their
            existing paths. Does not affect the differentiable solve.
        min_stretch: Lower element principal-stretch clamp (dimensionless).
            Pass a negative value to disable.
        max_stretch: Upper element principal-stretch clamp (dimensionless).
            Pass a negative value to disable.
        ground_plane: ``(nx, ny, nz, d)`` coefficients of an analytic
            ground plane ``n·x + d = 0``. ``None`` disables it.
        ground_ke: Ground normal stiffness [N/m].
        ground_kd: Ground normal damping [N·s/m].
        ground_kf: Ground tangential friction stiffness [N/m].
        ground_mu: Ground Coulomb friction coefficient.
        torque_stiffness: Spring bend-torque stiffness [N·m/rad].
            Active only when ``> 0`` and ``spring_rest_direction`` is set.
        torque_damping: Spring bend-torque damping [N·m·s/rad].
        spring_rest_direction: Per-spring rest-frame direction vectors,
            shape ``[spring_count, 3]``. ``None`` means no torque.
        linear_damping: Mass-proportional Rayleigh damping coefficient [1/s].
        differentiable: Enable owned per-step data and a checked sparse
            transpose-solve VJP on native ``wp.Tape``. The cached matrix is
            fixed during each tape; state positions/velocities, explicit
            external forces and enabled tet activations support gradients.
            Rest geometry, model parameters and analytic contact branches are
            fixed. Particle/surface contact and topology changes are unsupported.
            Uses checked CG regardless of the fast-path ``solver_type``, with
            the configured preconditioner and iteration budget. This is the
            converged linear-solve derivative, not the derivative of truncated CG.
        enable_tet_activations: Add the Newton volumetric contraction force
            from ``control.tet_activations``. Disabled by default to retain the
            original showcase/cutting launch sequence.
        defer_differentiation_checks: Accumulate true linear residual checks
            on device. Call ``check_differentiation()`` after forward/backward
            frame or graph launches; this avoids per-substep synchronization.
        tet_volume_weights: Optional model-device float32 array of nonnegative
            material fractions, shape ``(tet_count,)``, for volume telemetry.
            Clipped elements use their occupied fraction; unused slots use zero.
            Read in place; this option does not change forces or masses.
        fixed_particle_indices: Optional one-dimensional sequence of stationary
            Dirichlet node indices, anchored to construction-time model positions.
            Input/output boundary positions are restored and velocities zeroed.
            The symmetric solve eliminates these degrees of freedom while
            retaining physical masses and the complete elastic predictor.
            Not supported with differentiation, streaming or component scaling.
    """

    # ==================================================================
    # 1. Construction
    # ==================================================================

    def __init__(
        self,
        model: Model,
        dt: float = 1.0 / 60.0,
        mass: float = 1.0,
        preconditioner_type: str = "diag",
        solver_type: str = "bicgstab",
        linear_solver_maxiter: int = 50,
        min_stretch: float = 0.05,
        max_stretch: float = 20.0,
        ground_plane: "tuple[float, float, float, float] | None" = None,
        ground_ke: float = 1.0e5,
        ground_kd: float = 1.0e2,
        ground_kf: float = 1.0e3,
        ground_mu: float = 0.5,
        torque_stiffness: float = 0.0,
        torque_damping: float = 0.0,
        spring_rest_direction: "object | None" = None,
        linear_damping: float = 0.0,
        differentiable: bool = False,
        enable_tet_activations: bool = False,
        defer_differentiation_checks: bool = False,
        tet_volume_weights: wp.array | None = None,
        fixed_particle_indices=None,
        linear_solver_tolerance: float = 1.0e-2,
        linear_solver_use_cuda_graph: bool = True,
    ):
        if not math.isfinite(linear_solver_tolerance) or linear_solver_tolerance <= 0:
            raise ValueError("linear_solver_tolerance must be finite and positive")
        self.linear_solver_tolerance = float(linear_solver_tolerance)
        fixed_indices = np.asarray([] if fixed_particle_indices is None else fixed_particle_indices)
        if fixed_indices.ndim != 1 or (fixed_indices.size and fixed_indices.dtype.kind not in "iu"):
            raise ValueError("fixed_particle_indices must be a one-dimensional integer sequence")
        if fixed_indices.size and (fixed_indices.min() < 0 or fixed_indices.max() >= model.particle_count):
            raise ValueError("fixed_particle_indices contains an out-of-range node")
        fixed_indices = np.unique(fixed_indices).astype(np.int32)
        if fixed_indices.size and differentiable:
            raise ValueError("fixed_particle_indices is not supported with differentiable=True")
        if tet_volume_weights is not None:
            if (
                not isinstance(tet_volume_weights, wp.array)
                or tet_volume_weights.shape != (model.tet_count,)
                or tet_volume_weights.dtype != wp.float32
                or tet_volume_weights.device != model.device
            ):
                raise ValueError("tet_volume_weights must be a model-device float32 array of shape (tet_count,)")
            if any(not math.isfinite(value) or value < 0.0 for value in tet_volume_weights.numpy()):
                raise ValueError("tet_volume_weights must be finite and nonnegative")

        self._tet_volume_weights = tet_volume_weights
        # Tangent-value bookkeeping for projected-operator reuse. The version
        # counts host-visible assembled-value changes; a mutation recorded
        # inside CUDA graph capture cannot be counted on replay, so it
        # permanently disables version trust for this solver.
        self._tangent_version = 0
        self._tangent_refresh_captured = False
        # Opt-in for owners (the needle coupler) that re-project eagerly
        # before every captured-graph launch, allowing captured FEM trials to
        # omit the per-trial projection of unchanged assembled values.
        self._projection_before_graph = False
        super().__init__(model=model)

        self.mass = float(mass)
        self.linear_damping = float(linear_damping)
        self.preconditioner_type = preconditioner_type
        self.solver_type = solver_type
        self.linear_solver_maxiter = linear_solver_maxiter
        self.linear_solver_use_cuda_graph = bool(linear_solver_use_cuda_graph)
        # Chebyshev acceleration of the block-Jacobi preconditioner for the
        # conditional CUDA CG path: degree >= 2 enables it, 0 keeps the plain
        # 3x3 block-Jacobi inverse. The spectral interval is
        # [lambda_max/ratio, lambda_max] from a free-subspace Gershgorin bound.
        self.linear_solver_chebyshev_degree = 0
        self.linear_solver_chebyshev_ratio = 256.0
        self._chebyshev_instances = {}
        # Solve the Krylov system on free nodes only when stationary Dirichlet
        # boundaries exist: gathers/scatters at solve boundaries, identical
        # block values, exact zeros dropped from every inner product.
        self.linear_solver_compact = True
        # 0 = graph-capturable, with device-side early exit when conditional
        # graphs are enabled and supported; otherwise runs the full budget.
        # >0 = host-side residual checks, which are not graph-capturable.
        self.linear_solver_check_every = 0
        self.last_cg_iters: int | None = None
        self.min_stretch = float(min_stretch)
        self.max_stretch = float(max_stretch)
        self.dt = float(dt)
        # The timestep of the step being solved, as device data: the launches of a
        # step read it there, so a captured step is not tied to one timestep.
        self.step_time = StepTime(model.device)
        self.step_time.set(dt)
        self.differentiable = bool(differentiable)
        self.enable_tet_activations = bool(enable_tet_activations)
        self.defer_differentiation_checks = bool(defer_differentiation_checks)
        self._recording_forces = False
        self._differentiation_snapshot = None
        self._differentiation_revision = 0
        self._differentiation_eager_buffers = {}
        self._differentiation_stats = (
            wp.zeros(4, dtype=wp.float64, device=model.device) if self.differentiable else None
        )
        self.last_differentiation_residuals = None

        # ------------------------------------------------------------------
        # Hexahedral element data — read from model (set by ModelBuilder)
        # ------------------------------------------------------------------
        hex_data = getattr(model, "fem", None)
        H = 0 if hex_data is None else hex_data.hex_count
        self.hex_count = H
        if H > 0:
            self.hex_indices = hex_data.hex_indices.reshape((H, 8))
            self.hex_materials = hex_data.hex_materials
            self.hex_inv_J0 = hex_data.hex_inv_J0.reshape((H, 8))
            self.hex_det_J0_w = hex_data.hex_det_J0_w.reshape((H, 8))
        else:
            self.hex_indices = None
            self.hex_materials = None
            self.hex_inv_J0 = None
            self.hex_det_J0_w = None

        # BSR triplet buffers.  Layout (all constant, built once at init):
        #   [0..P)                   diagonal (mass + spring)
        #   [P..P+2S)                spring off-diagonals
        #   [P+2S..P+2S+3R)          tri lumped diagonal  (rest-geometry only)
        #   [P+2S+3R..P+2S+3R+16T)  tet 4×4 Hessian (rest-state F=I)
        #   [... +64H)               hex 8×8 Hessian
        num_blocks = model.particle_count + model.spring_count * 2 + model.tri_count * 3 + model.tet_count * 16 + H * 64
        self.bsr_rows = wp.zeros(num_blocks, dtype=wp.int32, device=model.device)
        self.bsr_cols = wp.zeros(num_blocks, dtype=wp.int32, device=model.device)
        self.bsr_values = wp.zeros(num_blocks, dtype=wp.mat33f, device=model.device)
        self.A_bsr = bsr_zeros(
            rows_of_blocks=model.particle_count,
            cols_of_blocks=model.particle_count,
            block_type=wp.mat33f,
            device=model.device,
            nnz_capacity=num_blocks,
        )
        # bsr_zeros leaves that storage unwritten (wp.empty), and the build writes only the blocks in use.
        self.A_bsr.values.zero_()
        # Build the system matrix once at construction; it is never rebuilt.
        self._build_constant_matrix(model, dt)
        bsr_set_from_triplets(
            dest=self.A_bsr,
            rows=self.bsr_rows,
            columns=self.bsr_cols,
            values=self.bsr_values,
            prune_numerical_zeros=True,
        )
        self._pc_inv_diag = None
        self._streaming = None
        self._fixed_particles = FixedParticles(model, self.A_bsr, fixed_indices) if fixed_indices.size else None
        self.M_bsr = self._build_preconditioner()
        self.dv = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self._normalized_rhs = wp.empty_like(self.dv)
        self._rhs_maximum = wp.zeros(1, dtype=float, device=model.device)
        # Scratch reused across steps so nothing allocates inside a captured graph.
        self._force_bufs: dict[str, wp.array] = {}
        for key in (
            "spring",
            "tet",
            "hex",
            "tri",
            "particle",
            "ground",
            "soft_contact",
            "gravity",
            "damping",
            "surface",
        ):
            self._forces(key)
        self._gravity_arr: wp.array | None = None
        self._lin_solver = None
        self._lin_solver_key = None
        self._linear_workspace_storage = None
        # One registry of Krylov workspaces for every caller of the standard solve. A key
        # in _pinned_workspaces belongs to an owner that bound the workspace into a graph
        # (linear_workspace) and is not evicted; _workspace_epoch counts the times the
        # registry was dropped, so such an owner sees that its bindings are gone.
        self._linear_workspaces = OrderedDict()
        self._pinned_workspaces = set()
        self._workspace_epoch = 0
        self._component_scaling = None
        self._tangent_triplet_slots = None
        self._topology_storage = self._topology_signature()

        # Initial volume: sum contributions from tet and/or hex elements.
        self._initial_volume = 0.0
        if model.tet_count > 0:
            _vols = wp.zeros(model.tet_count, dtype=wp.float32, device=model.device)
            wp.launch(
                kernel=compute_volume_kernel,
                dim=model.tet_count,
                inputs=[model.particle_q, model.tet_indices],
                outputs=[_vols],
                device=model.device,
            )
            self._initial_volume += float(_vols.numpy().sum())
        if H > 0:
            _hvols = wp.zeros(H, dtype=wp.float32, device=model.device)
            wp.launch(
                kernel=compute_hex_volume_kernel,
                dim=H,
                inputs=[self.hex_det_J0_w],
                outputs=[_hvols],
                device=model.device,
            )
            self._initial_volume += float(_hvols.numpy().sum())

        # Spring bend-torque (optional).
        self.torque_stiffness = float(torque_stiffness)
        self.torque_damping = float(torque_damping)
        if spring_rest_direction is None:
            self.spring_rest_direction = wp.zeros(model.spring_count, dtype=wp.vec3, device=model.device)
        else:
            if hasattr(spring_rest_direction, "shape"):
                if spring_rest_direction.shape != (model.spring_count, 3):
                    raise ValueError(
                        f"spring_rest_direction shape {spring_rest_direction.shape} != (spring_count={model.spring_count}, 3)"
                    )
            self.spring_rest_direction = wp.array(spring_rest_direction, dtype=wp.vec3, device=model.device)

        # Optional analytic ground plane  n·x + d = 0.
        self._ground_plane = None
        if ground_plane is not None:
            self._ground_plane = wp.array(ground_plane, dtype=wp.float32, device=model.device)
        self._ground_ke = float(ground_ke)
        self._ground_kd = float(ground_kd)
        self._ground_kf = float(ground_kf)
        self._ground_mu = float(ground_mu)

        self._soft_surface_contacts: SoftBodySurfaceContacts | None = None
        self.tet_volumes = wp.zeros(model.tet_count, dtype=wp.float32, device=model.device) if model.tet_count else None
        if tet_volume_weights is not None:
            self._initial_volume = self._compute_volume(model.particle_q)

    # ==================================================================
    # 2. Step
    # ==================================================================

    def notify_model_changed(self, flags):
        """Refresh global gravity; explicit topology edits use refresh_topology.

        Particle/tet capacity or storage changes require a new solver. Updating
        resident masses/connectivity/materials uses ``refresh_topology`` between
        graph launches, so the current linearization state is unambiguous.
        """
        self._invalidate_differentiation()
        if (predictor := getattr(self, "_affine_predictor", None)) is not None:
            predictor.invalidate()
        from newton import ModelFlags

        if flags & ModelFlags.MODEL_PROPERTIES:
            self._gravity_arr = None
        if flags & ModelFlags.CONSTRAINT_PROPERTIES:
            self.refresh_topology(self.model.particle_q)

    def reset(self, state, world_mask=None, flags=None):
        """Clear reusable numerical workspaces after the caller resets state."""
        super().reset(state, world_mask, flags)
        if (predictor := getattr(self, "_affine_predictor", None)) is not None:
            predictor.invalidate()
        if (held := getattr(self, "_held_start", None)) is not None:
            held.invalidate()
        self.dv.zero_()
        self._normalized_rhs.zero_()
        self._rhs_maximum.zero_()
        for force in self._force_bufs.values():
            force.zero_()

    def prepare_differentiation(self):
        """Own the current sparse matrix revision before recording/capture.

        The recorded solve differentiates positions, velocities, external
        forces and enabled tet activations with this matrix held fixed.
        Model geometry, masses, materials and ground coefficients are fixed.
        """
        if not self.differentiable:
            raise RuntimeError("Construct the solver with differentiable=True")
        differentiation.prepare(self)

    def check_differentiation(self, *, reset=True):
        """Check aggregate forward/transpose residuals after a recorded frame.

        Required after forward and backward graph launches when
        ``defer_differentiation_checks=True``. Returns maximum relative
        forward/backward residuals and raises on an unconverged solve.
        """
        if not self.differentiable:
            raise RuntimeError("Construct the solver with differentiable=True")
        return differentiation.check(self, reset=reset)

    def _invalidate_differentiation(self):
        if self.differentiable and differentiation.active_tape() is not None:
            raise RuntimeError("Reassembly/topology changes inside a tape are unsupported; the cached matrix is fixed")
        self._differentiation_revision += 1
        self._differentiation_snapshot = None

    @property
    def soft_surface_contacts(self):
        """Registered surface contact geometry, or None."""
        return self._soft_surface_contacts

    @property
    def velocity_increment(self):
        """Persistent solved velocity increment, also used for capture snapshots."""
        return self.dv

    def prepare_step(self, state_in, control, contacts, dt, *, external_forces=None):
        """Prepare reusable affine mechanics without integrating physical state.

        Returns a persistent :class:`PreparedFEMStep` with the constrained
        momentum operator, preconditioner, base impulse and graph-safe trial
        reconstruction. The caller owns tangent refresh and must reprepare for
        every new accepted state, control, contact set or matrix revision.
        Only the ordinary non-differentiable assembled path is supported;
        :meth:`step` and its differentiation contract are unchanged.
        """
        from .prepared_step import PreparedFEMStep

        PreparedFEMStep.check_supported(self)
        if self.model.device.is_capturing:
            raise RuntimeError("Prepare FEM mechanics before CUDA graph capture")
        if not hasattr(self, "_prepared_step"):
            self._prepared_step = PreparedFEMStep(self)
        return self._prepared_step.prepare(state_in, control, contacts, dt, external_forces)

    def refresh_preconditioner(self):
        """Refresh the existing inverse blocks without invalidating CUDA graphs."""
        self._refresh_preconditioner()

    def rebuild_element_blocks(self, particle_q, dt, *, update_preconditioner=True):
        """Reassemble the reference deformation-dependent tangent between frames."""
        self._rebuild_element_blocks(particle_q, dt, update_preconditioner=update_preconditioner)

    def set_soft_surface_contacts(
        self,
        particle_ranges: "list[tuple[int, int]]",
        surface_triangles_list: "list[np.ndarray | None]",
        ke: float = 1.0e3,
        kd: float = 1.0e2,
    ) -> None:
        """Register per-body surface meshes for deformed soft-soft (and soft-rigid) contacts.

        Must be called after :meth:`newton.ModelBuilder.finalize` and before the
        first :meth:`step`.  The meshes are built from rest-pose positions and
        refitted every time :meth:`_rebuild_element_blocks` is called.

        Args:
            particle_ranges: List of ``(start, count)`` slices into
                ``model.particle_q`` — one entry per soft body.
            surface_triangles_list: Per-body surface triangle arrays, shape
                ``(T_i, 3)``, indices local to the body (0 .. count_i−1).
                Pass ``None`` for bodies without surface data.
            ke: Surface penalty stiffness [N/m].
            kd: Surface contact damping [N·s/m].
        """
        model = self.model
        self._soft_surface_contacts = SoftBodySurfaceContacts(
            particle_q=model.particle_q,
            particle_ranges=particle_ranges,
            surface_triangles_list=surface_triangles_list,
            ke=ke,
            kd=kd,
            device=model.device,
        )

    def _accumulate_forces(
        self,
        model: Model,
        state_in: State,
        contacts: Contacts,
        control: Control,
        dt: float,
    ) -> None:
        """Write ``state_in.particle_f = h · Σf`` from all force terms."""
        self.step_time.set(dt)
        if self._soft_surface_contacts is not None:
            # update() (BVH refit) must be called by the caller before step()
            # when using CUDA graphs — mesh.refit() goes through the C++ runtime
            # and may not be captured.  _accumulate_forces is called inside the
            # graph, so we only do the pure-GPU force evaluation here.
            surface_contact_forces = self._soft_surface_contacts.eval_forces(
                state_in.particle_q,
                state_in.particle_qd,
                model.particle_radius,
                model.particle_flags,
                model.particle_count,
            )
        else:
            surface_contact_forces = self._forces("surface")

        # Accumulate in place: wp.array arithmetic allocates a fresh array per
        # operand, and any allocation inside a captured CUDA graph is freed after
        # capture while the graph keeps writing to it.
        particle_f = state_in.particle_f
        particle_f.zero_()
        for term in (
            self.eval_spring_forces(model, state_in),
            self.eval_tetrahedral_forces(model, control, state_in),
            self.eval_hexahedral_forces(model, control, state_in),
            self.eval_triangle_forces(model, control, state_in),
            self.eval_particle_ground_contact_forces(model, control, state_in),
            self.eval_gravity_forces(model),
            self.eval_linear_damping_forces(model, state_in),
            surface_contact_forces,
        ):
            wp.launch(
                kernel=accumulate_scaled_kernel,
                dim=model.particle_count,
                inputs=[term, self.step_time.seconds],
                outputs=[particle_f],
                device=model.device,
            )

    def _mark_tangent_values_changed(self) -> None:
        self._tangent_version += 1
        if self.model.device.is_capturing:
            self._tangent_refresh_captured = True

    def _projection_values_version(self):
        """Version token for skipping redundant projected-value refreshes.

        ``None`` requests an unconditional refresh: when a tangent mutation was
        ever captured (replays mutate values without advancing the counter), or
        during capture unless the owner re-projects before every graph launch.
        """
        if self._tangent_refresh_captured:
            return None
        if self.model.device.is_capturing and not self._projection_before_graph:
            return None
        return self._tangent_version

    def _write_element_blocks(self, particle_q: wp.array[wp.vec3], dt: float) -> None:
        # The element kernels read the timestep from the step time. Outside capture this
        # writes a changed one; a captured tangent refresh must find it written.
        self.step_time.set(dt)
        self._mark_tangent_values_changed()
        model = self.model
        if model.tet_count > 0:
            wp.launch(
                kernel=build_system_matrix_tet_kernel,
                dim=model.tet_count,
                inputs=[
                    particle_q,
                    model.tet_indices,
                    model.tet_poses,
                    model.tet_materials,
                    wp.float32(self.min_stretch),
                    wp.float32(self.max_stretch),
                    self.step_time.seconds,
                    wp.int32(self._tet_bsr_offset),
                    self.bsr_rows,
                    self.bsr_cols,
                    self.bsr_values,
                    self._tet_all_dirty,
                ],
                device=model.device,
            )
        if self.hex_count > 0:
            wp.launch(
                kernel=build_system_matrix_hex_kernel,
                dim=self.hex_count * 64,
                inputs=[
                    particle_q,
                    self.hex_indices,
                    self.hex_inv_J0,
                    self.hex_det_J0_w,
                    self.hex_materials,
                    wp.float32(self.min_stretch),
                    wp.float32(self.max_stretch),
                    self.step_time.seconds,
                    wp.int32(self._hex_bsr_offset),
                    self.bsr_rows,
                    self.bsr_cols,
                    self.bsr_values,
                    self._hex_all_dirty,
                ],
                device=model.device,
            )

    def _rebuild_element_blocks(
        self,
        particle_q: wp.array,
        dt: float,
        update_preconditioner: bool = True,
    ) -> None:
        """Recompute tet and/or hex Hessian BSR blocks from the current particle positions.

        Rebuilds every active element type's Hessian from the current deformation
        gradient F.  Must be called outside a CUDA graph: ``bsr_set_from_triplets``
        internally calls CUB ``DeviceRadixSort``, which allocates temporary
        storage and cannot be captured.  Invoke from the frame loop before
        ``wp.capture_launch``.

        When a CUDA graph is active, pass ``update_preconditioner=False``.
        ``bsr_set_from_triplets`` reuses the same ``A_bsr`` buffer (same GPU
        pointer), so the captured graph reads the freshly written values.
        ``preconditioner()`` allocates a *new* inv_diag buffer; reassigning
        ``self.M_bsr`` drops the old object's refcount to zero, Python frees
        it, and the graph accesses freed GPU memory.  Skipping the M update
        keeps the captured pointer alive and valid.

        Also refits the surface-contact BVHs if they were registered via
        :meth:`set_soft_surface_contacts`.

        Args:
            particle_q: Current particle positions [m].
            dt: Substep size [s].
            update_preconditioner: Whether to rebuild ``M_bsr`` after assembly.
        """
        self._invalidate_differentiation()
        # BVH refit for soft-surface contacts
        if self._fixed_particles is not None:
            self._fixed_particles.restore_positions(particle_q)
        if self._soft_surface_contacts is not None:
            self._soft_surface_contacts.update(particle_q)

        if self._streaming is not None:
            self.refresh_tangent(particle_q, dt)
            return

        self._write_element_blocks(particle_q, dt)
        bsr_set_from_triplets(
            dest=self.A_bsr,
            rows=self.bsr_rows,
            columns=self.bsr_cols,
            values=self.bsr_values,
            prune_numerical_zeros=False,
        )
        # block_diag refreshes in place (pointers stable) so it is safe even
        # while a CUDA graph is active; other types reallocate M_bsr.
        if update_preconditioner or self.preconditioner_type == "block_diag":
            self._refresh_preconditioner()

    def _rebuild_tet_blocks(self, *args, **kwargs) -> None:
        """Deprecated alias for :meth:`_rebuild_element_blocks`."""
        return self._rebuild_element_blocks(*args, **kwargs)

    def prepare_tetrahedral_updates(self, original_node_count: int) -> None:
        """Reserve persistent sparse connections for duplicated Tet4 basis nodes.

        Call before capturing a physics graph. Pool node ``i`` must represent
        original node ``i % original_node_count``. Future tetrahedra may choose
        any reserved copy of their original corners; the incision geometry is
        not predefined. Unsupported new connections are rejected before matrix
        publication. Existing states, forces, material laws and timestep remain
        unchanged. Requires tetrahedra only and ``block_diag`` preconditioning.
        Material-point holds are supported.

        Args:
            original_node_count: Number of original nodes in each complete copy
                of the particle pool. Must divide ``model.particle_count``.
        """
        if self._fixed_particles is not None:
            raise ValueError("Streaming topology is not supported with fixed_particle_indices")
        self._invalidate_differentiation()
        if self.model.device.is_capturing or self._lin_solver is not None:
            raise RuntimeError("Prepare tetrahedral updates before capturing or stepping the solver")
        if self._streaming is not None:
            raise RuntimeError("Tetrahedral updates are already prepared")
        if self._component_scaling is not None:
            raise RuntimeError("Prepare tetrahedral updates before component scaling")
        from .streaming import StreamingTetAssembly  # noqa: PLC0415

        self._streaming = StreamingTetAssembly(self, original_node_count)
        self._refresh_preconditioner()

    def prepare_component_scaling(self, node_components: wp.array[int], component_capacity: int) -> None:
        """Balance disconnected pieces with widely differing physical masses.

        Each component's matrix and RHS are divided by its total mass. The
        exact solution and physical particle masses remain unchanged, while
        finite-budget Krylov iterations account for small separated pieces.
        Requires ``block_diag`` preconditioning. Call after preparing sparse
        topology and before stepping or capturing a graph.

        Labels must be uniform across every mechanically coupled component.
        They may change in place after a topology edit; publish the new masses
        and labels, then call :meth:`refresh_component_scaling` before stepping.

        Args:
            node_components: Persistent particle-sized int32 labels on the model
                device. Use ``-1`` for unused rows and labels in
                ``[0, component_capacity)`` for active components.
            component_capacity: Reserved number of component labels.
        """
        if self._fixed_particles is not None:
            raise ValueError("Component scaling is not supported with fixed_particle_indices")
        self._invalidate_differentiation()
        if self.model.device.is_capturing or self._lin_solver is not None:
            raise RuntimeError("Prepare component scaling before capturing or stepping the solver")
        if self._component_scaling is not None:
            raise RuntimeError("Component scaling is already prepared")
        if self.preconditioner_type != "block_diag":
            raise ValueError("Component scaling requires block_diag preconditioning")
        if not isinstance(component_capacity, int) or component_capacity < 1:
            raise ValueError("component_capacity must be a positive integer")
        if (
            node_components.dtype != wp.int32
            or node_components.shape != (self.model.particle_count,)
            or node_components.device != self.model.device
        ):
            raise ValueError("node_components must be a particle-sized int32 array on the model device")
        from .component_scaling import ComponentScaling  # noqa: PLC0415

        matrix = self.A_bsr if self._streaming is None else self._streaming.compact_matrix
        self._component_scaling = ComponentScaling(self.model, matrix, node_components, component_capacity)

    def refresh_component_scaling(self) -> None:
        """Refresh component mass normalization on device after a topology edit."""
        if self._component_scaling is None:
            raise RuntimeError("Prepare component scaling before refreshing it")
        self._component_scaling.refresh()

    def refresh_tangent(self, particle_q: wp.array[wp.vec3], dt: float | None = None) -> None:
        """Refresh element stiffness in place, with CUDA graph support.

        Call once before capture to prepare the fixed sparse structure. Later
        calls update its values without sparse sorting or reallocating matrix
        buffers. Use ``preconditioner_type="block_diag"`` for captured calls.
        Use :meth:`refresh_topology` after connectivity changes within existing
        capacities. Construct a new solver if capacities or array storage change;
        captured graphs remain tied to the original model arrays.

        At unchanged timestep, only tetrahedral and hexahedral stiffness is
        refreshed. A changed timestep also rebuilds the mass, spring and
        triangle blocks. This is supported only for non-differentiable,
        non-streaming solvers outside tapes and capture. The element kernels
        read the timestep from ``step_time``, like the launches of a step, and
        this call writes it there; inside a capture it must already be written.
        A captured tangent refresh and a captured step therefore follow a
        timestep that was changed by a call of this method outside capture,
        without recapture.
        The velocity-increment warm start is cleared when the timestep changes.
        Surface-contact BVHs must still be refitted outside capture by the caller.

        Args:
            particle_q: Current particle positions [m], shape (particle_count,).
            dt: Positive substep duration [s], defaulting to the currently
                assembled timestep. Use the same timestep for subsequent steps.
        """
        dt = self.dt if dt is None else float(dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("refresh_tangent dt must be finite and positive")
        model = self.model
        changed_dt = dt != self.dt
        capturing = model.device.is_capturing
        if changed_dt:
            if self.differentiable:
                raise ValueError("Differentiable tangent refresh requires the construction timestep")
            if differentiation.active_tape() is not None or capturing:
                raise RuntimeError("Change the tangent timestep outside tapes and CUDA graph capture")
            if self._streaming is not None:
                raise ValueError("Streaming tangent refresh requires the construction timestep")
        if particle_q.shape != model.particle_q.shape or particle_q.device != model.device:
            raise ValueError("particle_q must match the model's particle shape and device")
        if capturing and self.preconditioner_type != "block_diag":
            raise RuntimeError("Captured tangent refresh requires the block_diag preconditioner")
        self._invalidate_differentiation()
        if self._fixed_particles is not None:
            self._fixed_particles.restore_positions(particle_q)
        if self._streaming is not None:
            self._streaming.refresh(particle_q, dt)
            self._refresh_preconditioner()
            return
        if changed_dt:
            # Spring damping and all stiffness blocks depend on dt; updating
            # only the deformation-dependent tet/hex blocks would mix timesteps.
            # A guess for another h can dwarf the new RHS and lose its initial
            # residual to cancellation. This is Krylov workspace, not velocity.
            self.dv.zero_()
            # Outside capture here. This also writes the step time, after every check
            # above, so steps at this timestep find it written.
            self._build_constant_matrix(model, dt, particle_q=particle_q)
        if self._tangent_triplet_slots is None:
            if capturing:
                raise RuntimeError("Call refresh_tangent once before CUDA graph capture")
            # Include initially zero blocks: deformation can make them nonzero.
            self._rebuild_element_blocks(particle_q, dt)
            self._tangent_triplet_slots = wp.empty_like(self.bsr_rows)
            wp.launch(
                map_triplet_blocks,
                dim=self.bsr_rows.size,
                inputs=[
                    self.bsr_rows,
                    self.bsr_cols,
                    self.A_bsr.offsets,
                    self.A_bsr.columns,
                    self._tangent_triplet_slots,
                ],
                device=model.device,
            )
            # Sparse preparation may have replaced buffers held by an earlier
            # iterative-solver functor. Warm it again before capturing a step.
            self._lin_solver = None
            self._lin_solver_key = None
            self.dt = dt
            return
        if not changed_dt:
            self._write_element_blocks(particle_q, dt)
        self.A_bsr.values.zero_()
        wp.launch(
            accumulate_triplet_blocks,
            dim=self.bsr_rows.size,
            inputs=[self._tangent_triplet_slots, self.bsr_values, self.A_bsr.values],
            device=model.device,
        )
        self._refresh_preconditioner()
        holds = getattr(self, "_node_holds", None)
        if holds is not None:
            # After the preconditioner: the Chebyshev bound is that of the matrix without holds,
            # which no hold exceeds (node_holds.py).
            holds.refresh()
        self.dt = dt

    def _topology_signature(self):
        model = self.model
        counts = (model.particle_count, model.tet_count, self.hex_count, model.spring_count, model.tri_count)
        arrays = (
            model.particle_mass,
            model.particle_inv_mass,
            model.particle_flags,
            model.tet_indices,
            model.tet_poses,
            model.tet_materials,
        )
        return counts, tuple(None if array is None else (array.ptr, array.shape) for array in arrays)

    def refresh_topology(
        self,
        particle_q: wp.array[wp.vec3],
        dt: float | None = None,
        *,
        element_sources: wp.array[wp.int32] | None = None,
    ) -> None:
        """Refresh changed tetrahedral connectivity and masses between graph launches.

        Particle and element capacities and the model's array pointers must
        remain unchanged. Update their contents in place, using zero material
        coefficients and valid node indices for unused tetrahedral slots. All
        updated rest poses must describe the reference geometry.

        By default this sorts the new sparse structure outside capture. After
        :meth:`prepare_tetrahedral_updates`, it instead applies element deltas
        to persistent GPU entries without sorting. Unchanged cached tangents
        are reused only at identical material, connectivity and positions.
        Matrix, preconditioner, and iterative-solver storage remain valid for
        an existing physics graph. Call only between completed graph
        launches, on the same ordered stream. The caller owns validation and
        transactional publication of new geometry and state.

        Args:
            particle_q: Current particle positions [m], shape (particle_count,).
            dt: Substep duration [s], equal to the currently assembled timestep.
            element_sources: Optional resident int32 array of previous element
                IDs for current slots, or -1 for new elements. This preserves
                cache reuse under fragment compaction. The default matches
                equal slot IDs. Used only for prepared tetrahedral updates.
        """
        self._invalidate_differentiation()
        if (predictor := getattr(self, "_affine_predictor", None)) is not None:
            predictor.invalidate()
        model = self.model
        dt = self.dt if dt is None else float(dt)
        if model.device.is_capturing:
            raise RuntimeError("refresh_topology must run outside CUDA graph capture")
        if self.preconditioner_type != "block_diag":
            raise ValueError("Persistent topology updates require the block_diag preconditioner")
        if dt != self.dt:
            raise ValueError("refresh_topology requires the solver's currently assembled timestep")
        if self._topology_signature() != self._topology_storage:
            raise ValueError("Topology updates must preserve model capacities and array storage")
        if particle_q.shape != model.particle_q.shape or particle_q.device != model.device:
            raise ValueError("particle_q must match the model's particle shape and device")
        if self._fixed_particles is not None:
            self._fixed_particles.restore_positions(particle_q)
        if self._streaming is not None:
            self._streaming.update(particle_q, dt, sources=element_sources)
            self._refresh_preconditioner()
            self.dv.zero_()
            return
        self._build_constant_matrix(model, dt, particle_q=particle_q)
        bsr_set_from_triplets(
            dest=self.A_bsr,
            rows=self.bsr_rows,
            columns=self.bsr_cols,
            values=self.bsr_values,
            prune_numerical_zeros=False,
        )
        if self._tangent_triplet_slots is None:
            self._tangent_triplet_slots = wp.empty_like(self.bsr_rows)
        wp.launch(
            map_triplet_blocks,
            dim=self.bsr_rows.size,
            inputs=[self.bsr_rows, self.bsr_cols, self.A_bsr.offsets, self.A_bsr.columns, self._tangent_triplet_slots],
            device=model.device,
        )
        self._refresh_preconditioner()
        self.dv.zero_()

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
        *,
        external_forces: wp.array[wp.vec3] | None = None,
        observables=None,
    ):
        """Advance one linearly implicit substep.

        The fast path evaluates forces into ``state_in.particle_f = h · Σf``, then
        solves ``A · Δv = particle_f − h² K v``. The system matrix ``A`` is constant
        (built once at construction from rest-pose element Hessians); call
        :meth:`refresh_tangent` before ``step()`` if you need the
        tangent refreshed from the current deformed configuration.

        Pass external nodal forces through ``external_forces``; the incoming
        ``state_in.particle_f`` values are not force inputs. The differentiable
        path owns its force workspace and preserves ``state_in.particle_f``.

        The launches read ``dt`` from ``step_time``. This call writes a changed
        ``dt`` there; inside a CUDA graph capture it only checks that the arrays
        hold ``dt``, and raises if they do not. A captured step therefore uses
        the timestep that ``step_time`` holds when the graph is launched, not
        the one of its capture. Before ``wp.capture_launch`` of a graph of your
        own, call ``solver.step_time.set(dt)`` if anything may have written
        another timestep since, and to replay at another timestep refresh the
        matrix for it first with :meth:`refresh_tangent`, outside capture.

        Args:
            state_in: Incoming positions and velocities.
            state_out: Output state storage.
            control: Optional control object accepted by the solver interface.
            contacts: Rigid contact pairs, or ``None`` when unused.
            dt: Substep duration [s].
            external_forces: Optional additional nodal forces [N], shape
                (particle_count,), dtype ``wp.vec3``, on the model device.
                Persistent input storage is supported by CUDA graph capture.
        """
        self.validate_observables(observables, contacts)
        model = self.model
        if external_forces is not None and (
            external_forces.shape != state_in.particle_f.shape
            or external_forces.dtype != wp.vec3
            or external_forces.device != model.device
        ):
            raise ValueError("external_forces must be a model-sized wp.vec3 array on the model device")
        if self.differentiable:
            # It writes the step time itself, after it has accepted dt: a taped step reads
            # the array again in tape.backward(), so a rejected dt must never be written.
            return differentiation.step(self, state_in, state_out, control, contacts, dt, external_forces)
        if differentiation.active_tape() is not None:
            raise RuntimeError("Construct the solver with differentiable=True before recording a step")
        # Outside capture this writes a changed timestep; inside it must already be written.
        self.step_time.set(dt)
        if control is None:
            control = model.control()

        if self._fixed_particles is not None:
            self._fixed_particles.restore(state_in)
        held = getattr(self, "_held_start", None)
        if held is not None and held.matches(state_in, dt, control):
            held.restore_impulse(state_in.particle_f)
        else:
            self._accumulate_forces(model, state_in, contacts, control, dt)
        if external_forces is not None:
            wp.launch(
                accumulate_scaled_kernel,
                dim=model.particle_count,
                inputs=[external_forces, self.step_time.seconds, state_in.particle_f],
                device=model.device,
            )
        self.implicit_integration(model, state_in, state_out, dt)

        return state_out

    # ==================================================================
    # 3. Internals: system matrix + linear solve
    # ==================================================================

    def _build_constant_matrix(self, model: Model, dt: float, particle_q: wp.array[wp.vec3] | None = None) -> None:
        """Write all BSR triplets once at construction.

        Mass, spring, and tri blocks are state-independent.  The tet/hex
        Hessian blocks are written here for the rest pose (F=I) and then
        refreshed per-step via :meth:`_rebuild_element_blocks`.

        Runs at construction, on a changed timestep and on a topology refresh, never
        inside a CUDA graph capture: the mass, spring and triangle kernels take the
        timestep as a launch argument, which a graph would keep. The element kernels
        read it from ``step_time``, which is written here.
        """
        if model.device.is_capturing:
            raise RuntimeError("Assemble the constant FEM matrix blocks outside CUDA graph capture")
        self.step_time.set(dt)
        self._mark_tangent_values_changed()
        offset = 0
        particle_q = model.particle_q if particle_q is None else particle_q

        if model.spring_count > 0:
            wp.launch(
                kernel=build_system_matrix_diagonal_kernel,
                dim=model.particle_count,
                inputs=[
                    self.bsr_rows,
                    self.bsr_cols,
                    self.bsr_values,
                    model.spring_indices,
                    model.spring_stiffness,
                    model.spring_damping,
                    wp.float32(dt),
                    model.particle_mass,
                    wp.int32(model.spring_count),
                ],
                device=model.device,
            )
        else:
            wp.launch(
                kernel=build_system_matrix_diagonal_mass_kernel,
                dim=model.particle_count,
                inputs=[self.bsr_rows, self.bsr_cols, self.bsr_values, model.particle_mass],
                device=model.device,
            )
        offset += model.particle_count

        if model.spring_count > 0:
            wp.launch(
                kernel=build_system_matrix_sparse_kernel,
                dim=model.spring_count,
                inputs=[
                    self.bsr_rows,
                    self.bsr_cols,
                    self.bsr_values,
                    model.spring_indices,
                    model.spring_stiffness,
                    model.spring_damping,
                    wp.float32(dt),
                    wp.int32(offset),
                ],
                device=model.device,
            )
            offset += model.spring_count * 2

        # Tri lumped diagonal — depends only on rest geometry, not positions.
        if model.tri_count > 0:
            wp.launch(
                kernel=build_system_matrix_tri_kernel,
                dim=model.tri_count,
                inputs=[
                    model.tri_indices,
                    model.tri_poses,
                    model.tri_materials,
                    wp.float32(dt),
                    wp.int32(offset),
                    self.bsr_rows,
                    self.bsr_cols,
                    self.bsr_values,
                ],
                device=model.device,
            )
            offset += model.tri_count * 3

        # Tet 4×4 Hessian — written here at rest (F=I); refreshed per-step
        # from the deformed configuration in _rebuild_element_blocks().
        self._tet_bsr_offset = offset
        if model.tet_count > 0:
            if not hasattr(self, "_tet_all_dirty"):
                self._tet_all_dirty = wp.ones(model.tet_count, dtype=wp.int32, device=model.device)
            else:
                self._tet_all_dirty.fill_(1)
            wp.launch(
                kernel=build_system_matrix_tet_kernel,
                dim=model.tet_count,
                inputs=[
                    particle_q,
                    model.tet_indices,
                    model.tet_poses,
                    model.tet_materials,
                    wp.float32(self.min_stretch),
                    wp.float32(self.max_stretch),
                    self.step_time.seconds,
                    wp.int32(offset),
                    self.bsr_rows,
                    self.bsr_cols,
                    self.bsr_values,
                    self._tet_all_dirty,
                ],
                device=model.device,
            )
            offset += model.tet_count * 16

        # Hex 8×8 Hessian (rest state: dirty-flag all elements)
        self._hex_bsr_offset = offset
        if self.hex_count > 0:
            if not hasattr(self, "_hex_all_dirty"):
                self._hex_all_dirty = wp.ones(self.hex_count, dtype=wp.int32, device=model.device)
            else:
                self._hex_all_dirty.fill_(1)
            wp.launch(
                kernel=build_system_matrix_hex_kernel,
                dim=self.hex_count * 64,
                inputs=[
                    particle_q,
                    self.hex_indices,
                    self.hex_inv_J0,
                    self.hex_det_J0_w,
                    self.hex_materials,
                    wp.float32(self.min_stretch),
                    wp.float32(self.max_stretch),
                    self.step_time.seconds,
                    wp.int32(offset),
                    self.bsr_rows,
                    self.bsr_cols,
                    self.bsr_values,
                    self._hex_all_dirty,
                ],
                device=model.device,
            )

    # ------------------------------------------------------------------
    # Preconditioner
    # ------------------------------------------------------------------

    def _build_preconditioner(self):
        """Construct the preconditioner operator for :attr:`A_bsr`.

        ``"block_diag"`` inverts the full 3x3 diagonal block per particle.
        Warp's built-in ``"diag"`` keeps only the three diagonal scalars of
        that block, discarding the intra-particle x/y/z coupling that
        Neo-Hookean elasticity produces; retaining it markedly reduces the
        CG iteration count.

        The inverse-block buffer is allocated **once** and refreshed in place,
        so the operator's device pointers stay valid for the lifetime of any
        CUDA graph that captures a solve.
        """
        if self.preconditioner_type != "block_diag":
            return preconditioner(self.A_bsr, ptype=self.preconditioner_type)

        n = self.model.particle_count
        device = self.model.device
        self._pc_inv_diag = wp.empty(n, dtype=wp.mat33f, device=device)
        self._refresh_preconditioner()

        inv_diag = self._pc_inv_diag

        def block_jacobi_mv(x, y, z, alpha, beta):
            wp.launch(
                kernel=block_jacobi_mv_kernel,
                dim=n,
                inputs=[inv_diag, x, y, z, wp.float32(alpha), wp.float32(beta)],
                device=device,
            )

        return LinearOperator(shape=(n, n), dtype=wp.vec3, device=device, matvec=block_jacobi_mv)

    def _refresh_preconditioner(self) -> None:
        """Recompute the preconditioner from the current :attr:`A_bsr`.

        For ``"block_diag"`` this writes in place, leaving ``M_bsr`` and its
        device pointers untouched — safe to call every frame even while a CUDA
        graph holds references to them.  Other types reallocate, so callers
        must skip this while a graph is active.
        """
        if self.preconditioner_type != "block_diag":
            self.M_bsr = preconditioner(self.A_bsr, ptype=self.preconditioner_type)
            return

        if self._streaming is not None:
            self._streaming.refresh_preconditioner(self._pc_inv_diag)
            return

        wp.launch(
            kernel=invert_block_diagonal_kernel,
            dim=self.model.particle_count,
            inputs=[
                self.A_bsr.offsets,
                self.A_bsr.columns,
                self.A_bsr.values,
                self._pc_inv_diag,
            ],
            device=self.model.device,
        )
        if self._fixed_particles is not None and self._fixed_particles.compact_matrix is not None:
            self._fixed_particles.gather_diagonal(self._pc_inv_diag)
        self._refresh_chebyshev()

    def _linear_solver_settings(self, tolerance=None):
        """Numerical settings shared by normal and captured solve workspaces."""
        use_cuda_graph = (
            self.linear_solver_use_cuda_graph
            and self.linear_solver_check_every == 0
            and self.model.device.is_cuda
            and wp.is_conditional_graph_supported()
        )
        return (
            self.solver_type,
            self.linear_solver_maxiter,
            self.linear_solver_check_every,
            self.linear_solver_tolerance if tolerance is None else tolerance,
            use_cuda_graph,
            int(self.linear_solver_chebyshev_degree) if use_cuda_graph else 0,
            bool(self.linear_solver_compact and use_cuda_graph),
        )

    def _chebyshev_preconditioner(self, operator, inverse_diagonal):
        """Create or reuse a Chebyshev acceleration of the block-Jacobi inverse."""
        from .chebyshev import ChebyshevJacobi  # noqa: PLC0415

        operator = aslinearoperator(operator)
        key = (
            id(operator),
            inverse_diagonal.ptr,
            int(self.linear_solver_chebyshev_degree),
            float(self.linear_solver_chebyshev_ratio),
        )
        instance = self._chebyshev_instances.get(key)
        if instance is None:
            if len(self._chebyshev_instances) >= 4:
                self._chebyshev_instances.clear()
            factory = ChebyshevJacobi
            options = {}
            fixed = self._fixed_particles
            if (
                fixed is not None
                and fixed.compact_matrix is not None
                and operator is fixed.compact_operator
                and inverse_diagonal is fixed.compact_diagonal
                and getattr(operator, "_packed_bsr_group_size", None) == 16
                and self.model.device.is_cuda
                and not self.differentiable
                and not getattr(self.model, "requires_grad", False)
                and differentiation.active_tape() is None
            ):
                from .fused_chebyshev import FusedPackedChebyshev  # noqa: PLC0415

                factory = FusedPackedChebyshev
                options = {
                    "matrix": fixed.compact_matrix,
                    "compact": True,
                    "bound_source": (self.A_bsr, fixed.mask, self._pc_inv_diag),
                }
            instance = factory(
                operator,
                inverse_diagonal,
                degree=int(self.linear_solver_chebyshev_degree),
                ratio=float(self.linear_solver_chebyshev_ratio),
                device=self.model.device,
                **options,
            )
            self._chebyshev_instances[key] = instance
            self._refresh_chebyshev()
        return instance

    def _refresh_chebyshev(self):
        # The spectral bound always comes from the assembled matrix and its
        # full-indexed inverse diagonal: the free-free blocks are identical in
        # the compact formulation, so one bound serves every instance.
        fixed = None if self._fixed_particles is None else self._fixed_particles.mask
        for instance in self._chebyshev_instances.values():
            instance.refresh(self.A_bsr, fixed=fixed, bound_diagonal=self._pc_inv_diag)

    def _compact_jacobi(self, fixed):
        operator = getattr(fixed, "compact_preconditioner", None)
        if operator is None or operator.shape[0] != fixed.free.size:
            count = fixed.free.size
            diagonal = fixed.compact_diagonal

            def compact_jacobi_mv(x, y, z, alpha, beta):
                wp.launch(
                    kernel=block_jacobi_mv_kernel,
                    dim=count,
                    inputs=[diagonal, x, y, z, wp.float32(alpha), wp.float32(beta)],
                    device=self.model.device,
                )

            operator = LinearOperator((count, count), wp.vec3, self.model.device, compact_jacobi_mv)
            fixed.compact_preconditioner = operator
        return operator

    def _linear_workspace_key(self, operator, inverse, settings):
        """Cache only nondifferentiable CUDA CG scratch bound to stable storage.

        Coupling tightens the inner tolerance during a difficult physical
        interval and restores it afterward. Those levels need distinct Krylov
        functors, but not a fresh allocation every time a level returns.
        Tangent values may change in place; replacing their storage or the
        fixed-particle operator must select a new workspace. During capture,
        retain the workspace explicitly prepared by the graph's owner.
        """
        if (
            not self.model.device.is_cuda
            or self.model.device.is_capturing
            or self.differentiable
            or getattr(self.model, "requires_grad", False)
            or differentiation.active_tape() is not None
            or self.solver_type != "cg"
            or self.linear_solver_check_every != 0
            or self.preconditioner_type != "block_diag"
            or self._streaming is not None
            or self._component_scaling is not None
        ):
            return None

        def storage(array):
            return None if array is None else (array.ptr, array.shape, array.strides, array.dtype)

        matrix, fixed = self.A_bsr, self._fixed_particles
        arrays = [
            matrix.offsets,
            matrix.columns,
            matrix.values,
            self._pc_inv_diag,
            self._normalized_rhs,
            self.dv,
        ]
        if fixed is not None:
            for name in ("compact_rhs", "compact_increment", "compact_diagonal"):
                arrays.append(getattr(fixed, name, None))
            compact = fixed.compact_matrix
            if compact is not None:
                arrays.extend((compact.offsets, compact.columns, compact.values))
        return (
            settings,
            None if getattr(self, "_certified_linear", None) is None else self._certified_linear.key(),
            id(operator),
            id(inverse),
            None if fixed is None else id(fixed.compact_operator),
            matrix.nrow,
            matrix.ncol,
            matrix.nnz,
            float(self.linear_solver_chebyshev_ratio),
            tuple(storage(array) for array in arrays),
        )

    def _reuse_linear_workspace(self):
        """Select an already warmed standard workspace without running a solve.

        Used before outer graph lookup so its workspace identity follows the
        requested tolerance, rather than the preceding interval's tolerance.
        A missing or replaced workspace still warms through a real eager step.
        """
        # Explicit workspace invalidation/replacement must not resurrect an
        # older cached functor, even when numerical settings stayed equal.
        if self._lin_solver is None or all(self._lin_solver is not value for value in self._linear_workspaces.values()):
            self._drop_linear_workspaces()
            return False
        operator = self.A_bsr if self._fixed_particles is None else self._fixed_particles.operator
        settings = self._linear_solver_settings()
        key = self._linear_workspace_key(operator, self.M_bsr, settings)
        workspace = None if key is None else self._linear_workspaces.get(key)
        if workspace is None:
            return False
        self._linear_workspaces.move_to_end(key)
        self._lin_solver, self._lin_solver_key = workspace, settings
        self._linear_workspace_storage = key
        return True

    def tolerance_levels(self, tolerance=None):
        """Inner tolerances that one coupled interval can visit, the loosest first.

        The first level is ``tolerance`` (default: the configured
        ``linear_solver_tolerance``). Each further level is a tenth of the one
        before it, not below ``min(tolerance, 1e-6)``, which is the last level.
        The host-controlled trials tighten the tolerance by the same expression,
        so they reach these floats and no others.
        """
        tolerance = self.linear_solver_tolerance if tolerance is None else tolerance
        tolerances = [tolerance]
        while True:
            value = max(min(tolerance, 1e-6), tolerances[-1] * 0.1)
            if value == tolerances[-1]:
                break
            tolerances.append(value)
        return tolerances

    def _drop_linear_workspaces(self):
        """Forget every registered workspace, the pinned ones included.

        The epoch tells an owner that bound registered workspaces into a graph
        (see :meth:`linear_binding_epoch`) that they are no longer the solver's.
        """
        self._linear_workspaces.clear()
        self._pinned_workspaces.clear()
        self._workspace_epoch += 1

    def linear_binding_epoch(self):
        """Number of times the registry of Krylov workspaces was dropped.

        For an owner that binds workspaces of :meth:`linear_workspace` into a
        captured graph: the bindings are the solver's as long as this number is
        the one read when they were fetched. An explicit invalidation
        (``_lin_solver`` set to ``None``) is honored here, before the number is
        returned, so call this before fetching and not between two fetches.
        """
        if self._lin_solver is None:
            self._drop_linear_workspaces()
        return self._workspace_epoch

    def linear_workspace(self, tolerance=None):
        """Registered Krylov workspace of the standard solve at ``tolerance``.

        Returns ``(settings, workspace)`` for the current operator, as a step at
        this tolerance selects it: the workspace of the registry if it has one,
        otherwise a new one, allocated without a solve and registered. The
        workspace is pinned, that is, kept in the registry until the registry is
        dropped or :meth:`unpin_linear_workspaces` releases it, so a graph that
        binds it and a later eager or captured step at the same tolerance use
        the same object. Workspace keys hold the state of the certificate scope:
        fetch inside the coupling attempt whose steps will use the workspace.

        A configuration without a cacheable key (see
        :meth:`_linear_workspace_key`) gets a new, unregistered workspace on
        every call. Must be called outside CUDA graph capture.
        """
        if self.model.device.is_capturing:
            raise RuntimeError("Fetch FEM Krylov workspaces before CUDA graph capture")
        operator = self.A_bsr if self._fixed_particles is None else self._fixed_particles.operator
        settings = self._linear_solver_settings(tolerance)
        key = self._linear_workspace_key(operator, self.M_bsr, settings)
        workspace = None if key is None else self._linear_workspaces.get(key)
        if workspace is None:
            workspace = self._make_linear_solver(operator, self.M_bsr, settings)
            if key is not None:
                self._linear_workspaces[key] = workspace
        if key is not None:
            self._pinned_workspaces.add(key)
        return settings, workspace

    def unpin_linear_workspaces(self, workspaces):
        """Let the registry evict these workspaces again.

        They stay registered while at most eight workspaces are without a pin; the oldest
        of the others leave here, never the selected workspace (a selection that is not
        registered counts as an explicit invalidation, see :meth:`_reuse_linear_workspace`).
        """
        for key, workspace in self._linear_workspaces.items():
            if any(workspace is released for released in workspaces):
                self._pinned_workspaces.discard(key)
        self._evict_linear_workspaces(keep=self._lin_solver)

    def _evict_linear_workspaces(self, keep=None):
        """At most eight workspaces that no owner pinned stay: the oldest of the others leave, never ``keep``."""
        excess = len(self._linear_workspaces) - len(self._pinned_workspaces) - 8
        if excess > 0:
            unpinned = [
                name
                for name, workspace in self._linear_workspaces.items()
                if name not in self._pinned_workspaces and workspace is not keep
            ]
            for name in unpinned[:excess]:
                del self._linear_workspaces[name]

    def load_kernel_modules(self):
        """Load the kernel modules of this package that a step can launch, by name and with no launch.

        For start-up on a CUDA device: a CUDA graph capture loads the modules of what
        it records, and the first launch of any other kernel loads its module where it
        happens, in a step. The modules are loaded in the build of the default thread
        block (256), which every launch of them uses. Kernels that Warp generates per
        argument type (its sparse and Krylov kernels) have no name to load them by:
        they are loaded by the launches of the first projection, of the workspaces and
        of the first step. Nothing is done on a CPU device.
        """
        device = self.model.device
        if not device.is_cuda:
            return
        from . import (  # noqa: PLC0415
            affine_guess,
            certified_cg,
            certified_scope,
            chebyshev,
            fixed_particles,
            fused_chebyshev,
            held_start,
            kernels,
            prepared_step,
            sparse_update,
            warp_basis_provider,
        )
        from .packed_bsr import packed_kernel  # noqa: PLC0415

        modules = (
            kernels,
            __name__,
            fixed_particles,
            sparse_update,
            prepared_step,
            held_start,
            chebyshev,
            fused_chebyshev,
            certified_cg,
            certified_scope,
            affine_guess,
            warp_basis_provider,
            # The packed matrix-vector product of the compact operator: one module per group size.
            packed_kernel(16).module,
        )
        for module in modules:
            wp.load_module(module, device=device, block_dim=256)

    def _make_linear_solver(self, operator, inverse, settings, *, rhs=None, increment=None):
        """Allocate persistent Krylov scratch without running a discarded solve."""
        kind, maxiter, check_every, tolerance, use_cuda_graph, chebyshev_degree, compact = settings
        standard = (
            kind == "cg"
            and inverse is self.M_bsr
            and self.preconditioner_type == "block_diag"
            and self._pc_inv_diag is not None
            and self._streaming is None
        )
        fixed = self._fixed_particles
        compact_solve = (
            compact
            and standard
            and rhs is None
            and increment is None
            and fixed is not None
            and getattr(fixed, "compact_operator", None) is not None
            and operator is fixed.operator
        )
        if compact_solve:
            fixed.gather_diagonal(self._pc_inv_diag)
            operator = fixed.compact_operator
            rhs, increment = fixed.compact_rhs, fixed.compact_increment
            inverse = self._compact_jacobi(fixed)
            if chebyshev_degree >= 2:
                inverse = self._chebyshev_preconditioner(operator, fixed.compact_diagonal).preconditioner
        elif chebyshev_degree >= 2 and standard:
            # Same matrix, right-hand side and stopping rule; only the SPD
            # preconditioner improves. See chebyshev.py for the construction.
            inverse = self._chebyshev_preconditioner(operator, self._pc_inv_diag).preconditioner
        solve = {"cg": cg, "bicgstab": bicgstab, "gmres": gmres, "cr": cr}.get(kind)
        if solve is None:
            raise ValueError(f"Invalid solver type: {kind}")
        if self.model.device.is_cpu and kind == "cg":
            operator = aslinearoperator(operator)
            if operator.batch_offsets is None:
                # Warp's unbatched CPU tiled dot repeats each tile per lane.
                # One scalar-DOF batch selects its direct CPU reduction.
                scalar_count = 3 * self.model.particle_count
                operator = LinearOperator(
                    operator.shape,
                    operator.dtype,
                    operator.device,
                    operator.matvec,
                    batch_offsets=wp.array([0, scalar_count], dtype=int, device=self.model.device),
                    max_batch_length=scalar_count,
                )
        certificate = getattr(self, "_certified_linear", None)
        if certificate is not None and certificate.active:
            from .certified_cg import CertifiedCorrection

            if not (
                compact_solve
                and kind == "cg"
                and maxiter == 160
                and check_every == 0
                and use_cuda_graph
                and not any(a.requires_grad for a in (fixed.compact_matrix.values, rhs, increment))
            ):
                raise RuntimeError("Certified FEM configuration changed during its attempt")
            functor = CertifiedCorrection(
                fixed.compact_matrix,
                operator,
                rhs,
                increment,
                inverse,
                tolerance,
                status=certificate.status,
            )
        else:
            functor = solve(
                operator,
                self._normalized_rhs if rhs is None else rhs,
                self.dv if increment is None else increment,
                tol=tolerance,
                # Warp otherwise defaults atol to tol. A 0.01 N*s absolute
                # threshold suppresses gravity for small physical masses.
                atol=0.0,
                maxiter=maxiter,
                M=inverse,
                check_every=check_every,
                use_cuda_graph=use_cuda_graph,
                run=False,
            )
        # Callers pass M explicitly on every call; record the preconditioner
        # this workspace was built around so they apply the same one.
        functor.bound_preconditioner = inverse
        functor.compact_solve = compact_solve
        functor.bound_rhs = self._normalized_rhs if rhs is None else rhs
        return functor

    def implicit_integration(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
    ) -> None:
        """Solve ``A · Δv = h*f - h²*K*v`` and integrate ``v``, ``x``.

        ``state_in.particle_f`` must already be pre-multiplied by ``dt``.
        The predictor uses the assembled tet/hex and spring stiffness. The
        legacy triangle-only diagonal approximation has no off-diagonal
        elastic predictor and retains its existing integration behavior.
        """
        self.step_time.set(dt)
        held = getattr(self, "_held_start", None)
        if held is not None and held.matches(state_in, dt):
            held.subtract_predictor(state_in.particle_f, dt)
        elif self._streaming is not None:
            self._streaming.subtract_predictor(state_in.particle_qd, state_in.particle_f)
        elif model.tet_count or self.hex_count or model.spring_count:
            wp.launch(
                subtract_elastic_predictor,
                dim=model.particle_count,
                inputs=[
                    self.A_bsr.offsets,
                    self.A_bsr.columns,
                    self.A_bsr.values,
                    state_in.particle_qd,
                    state_in.particle_f,
                ],
                device=model.device,
            )
            if model.spring_count:
                wp.launch(
                    restore_spring_damping_predictor,
                    dim=model.spring_count,
                    inputs=[
                        model.spring_indices,
                        model.spring_damping,
                        self.step_time.seconds,
                        state_in.particle_qd,
                        state_in.particle_f,
                    ],
                    device=model.device,
                )
        check_every = self.linear_solver_check_every
        rhs = state_in.particle_f
        operator = self.A_bsr if self._streaming is None else self._streaming.operator
        inverse = self.M_bsr
        if self._fixed_particles is not None:
            self._fixed_particles.project_rhs(rhs)
            if self._fixed_particles.prepare(self.A_bsr, values_version=self._projection_values_version()):
                self._lin_solver = None
                self._lin_solver_key = None
            operator = self._fixed_particles.operator
        if self._component_scaling is not None:
            matrix = self.A_bsr if self._streaming is None else self._streaming.compact_matrix
            self._component_scaling.prepare(matrix, self._pc_inv_diag, rhs)
            rhs = self._component_scaling.rhs
            operator, inverse = self._component_scaling.operator, self._component_scaling.preconditioner

        self._rhs_maximum.zero_()
        wp.launch(
            maximum_rhs_component,
            dim=model.particle_count,
            inputs=[rhs, self._rhs_maximum],
            device=model.device,
        )
        wp.launch(
            normalize_linear_rhs,
            dim=model.particle_count,
            inputs=[rhs, self._rhs_maximum, self._normalized_rhs, self.dv],
            device=model.device,
        )
        # Persistent functor: the one-shot solver functions allocate their scratch
        # buffers on every call, which is not safe inside a captured CUDA graph.
        key = self._linear_solver_settings()
        workspace_key = self._linear_workspace_key(operator, inverse, key)
        if self._lin_solver is None:
            # Honor explicit invalidation in direct solver.step callers too,
            # including a freshly rebuilt fixed-particle operator above.
            self._drop_linear_workspaces()
        if (
            self._lin_solver is None
            or self._lin_solver_key != key
            or (workspace_key is not None and workspace_key != self._linear_workspace_storage)
        ):
            workspace = None if workspace_key is None else self._linear_workspaces.get(workspace_key)
            if workspace is None:
                workspace = self._make_linear_solver(operator, inverse, key)
                if workspace_key is not None:
                    self._linear_workspaces[workspace_key] = workspace
                    # Eight tolerance/settings levels cover the adaptive inner
                    # schedule while bounding scratch memory for arbitrary users.
                    # Workspaces that an owner pinned are bound into its graphs
                    # and are not counted: the oldest of the others leave.
                    self._evict_linear_workspaces()
            elif workspace_key is not None:
                self._linear_workspaces.move_to_end(workspace_key)
            self._lin_solver, self._lin_solver_key = workspace, key
            self._linear_workspace_storage = workspace_key
        if getattr(self._lin_solver, "compact_solve", False):
            self._fixed_particles.gather_solve_operands(self._normalized_rhs, self.dv)
        affine = getattr(self, "_affine_predictor", None)
        if affine is not None:
            affine.before_solve(rhs, self._lin_solver)
        with PauseGC():
            result = self._lin_solver(
                b=getattr(self._lin_solver, "bound_rhs", self._normalized_rhs),
                M=getattr(self._lin_solver, "bound_preconditioner", inverse),
            )
        if affine is not None:
            affine.after_solve(rhs, self._lin_solver)
        if getattr(self._lin_solver, "compact_solve", False):
            self._fixed_particles.scatter_increment(self.dv)
        wp.launch(
            restore_linear_increment,
            dim=model.particle_count,
            inputs=[self._rhs_maximum, self.dv],
            device=model.device,
        )

        if check_every > 0 and result is not None:
            # Host-check path returns (iterations, residual, tolerance).
            self.last_cg_iters = int(result[0])

        wp.launch(
            kernel=update_state,
            dim=model.particle_count,
            inputs=[
                self.dv,
                self.step_time.seconds,
                state_in.particle_q,
                state_in.particle_qd,
                state_out.particle_q,
                state_out.particle_qd,
            ],
            device=model.device,
        )
        if self._fixed_particles is not None:
            self._fixed_particles.restore(state_out)

    # ==================================================================
    # 4. Force evaluators
    # ==================================================================

    def eval_tetrahedral_forces(self, model: Model, control: Control, state: State):
        """Tet FEM elastic force from Stable Neo-Hookean [N]."""
        forces = self._forces("tet")
        if model.tet_count:
            wp.launch(
                kernel=eval_tetrahedra,
                dim=model.tet_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    model.tet_indices,
                    model.tet_poses,
                    model.tet_materials,
                    wp.float32(self.min_stretch),
                    wp.float32(self.max_stretch),
                ],
                outputs=[forces],
                device=model.device,
            )
            if self.enable_tet_activations:
                wp.launch(
                    eval_tet_activations,
                    dim=model.tet_count,
                    inputs=[
                        state.particle_q,
                        model.tet_indices,
                        model.tet_poses,
                        model.tet_materials,
                        control.tet_activations,
                    ],
                    outputs=[forces],
                    device=model.device,
                )
        return forces

    def eval_hexahedral_forces(self, model: Model, control: Control, state: State) -> wp.array:
        """Q1 hex FEM elastic force from Stable Neo-Hookean [N].

        Returns zeros when ``hex_count == 0``.
        """
        forces = self._forces("hex")
        if self.hex_count == 0:
            return forces
        wp.launch(
            kernel=eval_hexahedra,
            dim=self.hex_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                self.hex_indices,
                self.hex_inv_J0,
                self.hex_det_J0_w,
                self.hex_materials,
                wp.float32(self.min_stretch),
                wp.float32(self.max_stretch),
            ],
            outputs=[forces],
            device=model.device,
        )
        return forces

    def eval_triangle_forces(self, model: Model, control: Control, state: State):
        """Triangle membrane (in-plane stretch + area-preservation) force [N]."""
        forces = self._forces("tri")
        if model.tri_count:
            wp.launch(
                kernel=eval_triangles,
                dim=model.tri_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    model.tri_indices,
                    model.tri_poses,
                    model.tri_materials,
                ],
                outputs=[forces],
                device=model.device,
            )
        return forces

    def eval_spring_forces(self, model: Model, state: State):
        """Linear spring force [N] with optional bend torque."""
        forces = self._forces("spring")
        if model.spring_count == 0:
            return forces
        if self.torque_stiffness > 0.0:
            wp.launch(
                kernel=eval_springs_linear_and_torque,
                dim=model.spring_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    model.spring_indices,
                    model.spring_rest_length,
                    model.spring_stiffness,
                    model.spring_damping,
                    self.spring_rest_direction,
                    wp.float32(self.torque_stiffness),
                    wp.float32(self.torque_damping),
                ],
                outputs=[forces],
                device=model.device,
            )
        else:
            wp.launch(
                kernel=eval_springs,
                dim=model.spring_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    model.spring_indices,
                    model.spring_rest_length,
                    model.spring_stiffness,
                    model.spring_damping,
                ],
                outputs=[forces],
                device=model.device,
            )
        return forces

    def eval_particle_ground_contact_forces(self, model: Model, control: Control, state: State):
        """Particle vs analytic ground plane (penalty + Coulomb friction) [N].

        Returns zeros when no ``ground_plane`` was configured.
        """
        forces = self._forces("ground")
        if self._ground_plane is None:
            return forces
        wp.launch(
            kernel=eval_particle_ground_contacts,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.particle_radius,
                model.particle_inv_mass,
                model.particle_flags,
                self._ground_ke,
                self._ground_kd,
                self._ground_kf,
                self._ground_mu,
                self._ground_plane,
                self._get_gravity_array(model),
            ],
            outputs=[forces],
            device=model.device,
        )
        return forces

    def eval_gravity_forces(self, model: Model):
        """Per-particle gravity ``f = m · g`` [N]."""
        forces = self._forces("gravity")
        if model.particle_count:
            wp.launch(
                kernel=eval_gravity_from_array,
                dim=model.particle_count,
                inputs=[self._get_gravity_array(model), model.particle_mass, model.particle_flags],
                outputs=[forces],
                device=model.device,
            )
        return forces

    def eval_linear_damping_forces(self, model: Model, state: State) -> wp.array:
        """Mass-proportional Rayleigh damping ``f = −α·m·v`` [N].

        Damps rigid-body drift and rolling without affecting rest-state
        equilibrium.  Zero-cost when ``linear_damping == 0``.
        """
        forces = self._forces("damping")
        if self.linear_damping > 0.0 and model.particle_count:
            wp.launch(
                kernel=eval_linear_damping_kernel,
                dim=model.particle_count,
                inputs=[
                    state.particle_qd,
                    model.particle_mass,
                    model.particle_flags,
                    wp.float32(self.linear_damping),
                ],
                outputs=[forces],
                device=model.device,
            )
        return forces

    # ==================================================================
    # 5. Helpers
    # ==================================================================

    def _get_gravity_array(self, model: Model) -> wp.array:
        """Return ``model.gravity`` as a ``wp.array[wp.vec3]``."""
        g = model.gravity
        if isinstance(g, wp.array):
            return g
        if self._gravity_arr is None:
            self._gravity_arr = wp.array([g], dtype=wp.vec3, device=model.device)
        return self._gravity_arr

    def _forces(self, key: str) -> wp.array:
        """Return the zeroed persistent force scratch buffer for ``key``."""
        buf = self._force_bufs.get(key)
        if buf is None:
            buf = wp.zeros(
                self.model.particle_count, dtype=wp.vec3, device=self.model.device, requires_grad=self._recording_forces
            )
            self._force_bufs[key] = buf
        else:
            buf.zero_()
        return buf

    def compute_volume(self, state: State | None = None) -> float:
        """Current volume [m³] from tet particle positions and/or hex Gauss data.

        For tet elements, ``state`` is required (volumes are computed from
        current particle positions, weighted by ``tet_volume_weights`` when
        provided). For hex elements, ``state`` is not used
        (volume is computed analytically from the reference Gauss data).

        Args:
            state: Current simulation state. Required when tet elements are
                present; ignored (but accepted for API parity) for hex-only.
        """
        return self._compute_volume(None if state is None else state.particle_q)

    def _compute_volume(self, positions: wp.array | None) -> float:
        """Sum actual occupied material, excluding inactive reserved slots."""
        model = self.model
        total = 0.0

        if model.tet_count > 0 and self.tet_volumes is not None and positions is not None:
            kernel = compute_volume_kernel
            inputs = [positions, model.tet_indices]
            if self._tet_volume_weights is not None:
                kernel = _weighted_tet_volume
                inputs.append(self._tet_volume_weights)
            wp.launch(
                kernel=kernel,
                dim=model.tet_count,
                inputs=inputs,
                outputs=[self.tet_volumes],
                device=model.device,
            )
            total += float(self.tet_volumes.numpy().sum())

        if self.hex_count > 0:
            vols = wp.zeros(self.hex_count, dtype=wp.float32, device=model.device)
            wp.launch(
                kernel=compute_hex_volume_kernel,
                dim=self.hex_count,
                inputs=[self.hex_det_J0_w],
                outputs=[vols],
                device=model.device,
            )
            total += float(vols.numpy().sum())

        return total

    def get_initial_volume(self, state: State | None = None) -> float:
        """Reference volume [m³].

        Always available after construction and includes any supplied material
        volume weights. The ``state`` argument is accepted for API parity but
        is not used.

        Args:
            state: Ignored.
        """
        return self._initial_volume if self._initial_volume > 1e-12 else 1.0

    def get_volume_ratio(self, state: State | None = None) -> float:
        """Measured volume divided by reference volume [dimensionless].

        Args:
            state: Current simulation state (required for tet; ignored for hex).
        """
        initial = self.get_initial_volume(state)
        if initial <= 1.0e-12:
            return 1.0
        current = self.compute_volume(state)
        if not math.isfinite(current) or current <= 0.0:
            return 1.0
        ratio = current / initial
        if not math.isfinite(ratio):
            return 1.0
        return ratio
