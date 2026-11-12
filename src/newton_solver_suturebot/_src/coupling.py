# SPDX-License-Identifier: Apache-2.0
"""Converged partitioned coupling of a dynamic needle and FEM material.

The interaction is evaluated at the shared end-of-step state. Every trial
restarts both integrators from the same accepted state; only the converged
trial advances material history. No controller or target geometry enters here.
"""

from collections import OrderedDict

import numpy as np
import warp as wp

from .insertion import NeedleStepTooLarge
from .mujoco_free_body import MuJoCoFreeBody, synchronizes_every_step
from .shaft_preconditioner import _ShaftPreconditioner
from .solvers.fem import SolverFEM
from .solvers.fem.differentiation import active_tape
from .subcycled_mujoco import SubcycledMuJoCo
from .tet_step import TetStepGuard


@wp.kernel(enable_backward=False)
def _apply_interface_load(
    load: wp.array[wp.float64],
    external_nodes: wp.array[wp.vec3],
    external_wrench: wp.array[wp.spatial_vector],
    body: int,
    moment_length: wp.float64,
    particle_f: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    if i < particle_f.shape[0]:
        force = wp.vec3(0.0)
        for axis in range(3):
            force[axis] = float(load[3 * i + axis] + wp.float64(external_nodes[i][axis]))
        particle_f[i] = force
    if i < body_f.shape[0]:
        wrench = external_wrench[i]
        if i == body:
            for axis in range(6):
                scale = wp.float64(1.0)
                if axis >= 3:
                    scale = moment_length
                wrench[axis] = float(wp.float64(wrench[axis]) + scale * load[3 * particle_f.shape[0] + axis])
        body_f[i] = wrench


@wp.kernel(enable_backward=False)
def _apply_rigid_interface_load(
    load: wp.array[wp.float64],
    external_wrench: wp.array[wp.spatial_vector],
    body: int,
    particle_count: int,
    moment_length: wp.float64,
    body_f: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    wrench = external_wrench[i]
    if i == body:
        for axis in range(6):
            scale = wp.float64(1.0)
            if axis >= 3:
                scale = moment_length
            wrench[axis] = float(wp.float64(wrench[axis]) + scale * load[3 * particle_count + axis])
    body_f[i] = wrench


@wp.func
def _endpoint_bits(
    i: int,
    q: wp.array[wp.uint32],
    qd: wp.array[wp.uint32],
    poses: wp.array[wp.uint32],
    velocities: wp.array[wp.uint32],
):
    if i < q.shape[0]:
        return q[i]
    i -= q.shape[0]
    if i < qd.shape[0]:
        return qd[i]
    i -= qd.shape[0]
    if i < poses.shape[0]:
        return poses[i]
    return velocities[i - poses.shape[0]]


@wp.kernel(enable_backward=False)
def _snapshot_rigid_endpoint(
    q: wp.array[wp.uint32],
    qd: wp.array[wp.uint32],
    poses: wp.array[wp.uint32],
    velocities: wp.array[wp.uint32],
    saved: wp.array[wp.uint32],
):
    i = wp.tid()
    saved[i] = _endpoint_bits(i, q, qd, poses, velocities)


@wp.kernel(enable_backward=False)
def _compare_rigid_endpoint(
    q: wp.array[wp.uint32],
    qd: wp.array[wp.uint32],
    poses: wp.array[wp.uint32],
    velocities: wp.array[wp.uint32],
    saved: wp.array[wp.uint32],
    equal: wp.array[int],
):
    i = wp.tid()
    if saved[i] != _endpoint_bits(i, q, qd, poses, velocities):
        wp.atomic_min(equal, 0, 0)


@wp.kernel(enable_backward=False)
def _interface_residual(
    particle_f: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    body: int,
    moment_length: wp.float64,
    load: wp.array[wp.float64],
    previous: wp.array[wp.float64],
    has_previous: int,
    value: wp.array[wp.float64],
    residual: wp.array[wp.float64],
    statistics: wp.array[wp.float64],
):
    i = wp.tid()
    count = 3 * particle_f.shape[0]
    force = wp.float64(0.0)
    if i < count:
        force = wp.float64(particle_f[i // 3][i % 3])
    else:
        axis = i - count
        force = wp.float64(body_f[body][axis])
        if axis >= 3:
            force /= moment_length
    value[i] = force
    difference = force - load[i]
    residual[i] = difference
    if not wp.isfinite(force):
        wp.atomic_max(statistics, 5, wp.float64(1.0))
    # Interface loads are sparse; avoid contending on zero contributions from
    # every unaffected tissue degree of freedom.
    if difference != wp.float64(0.0):
        wp.atomic_max(statistics, 0, wp.abs(difference))
    if force != wp.float64(0.0):
        wp.atomic_max(statistics, 1, wp.abs(force))
    if load[i] != wp.float64(0.0):
        wp.atomic_max(statistics, 2, wp.abs(load[i]))
    if has_previous != 0:
        delta = difference - previous[i]
        if delta != wp.float64(0.0):
            wp.atomic_add(statistics, 3, delta * delta)
            wp.atomic_add(statistics, 4, previous[i] * delta)


@wp.kernel(enable_backward=False)
def _relax_interface(
    residual: wp.array[wp.float64],
    relaxation: wp.float64,
    load: wp.array[wp.float64],
    previous: wp.array[wp.float64],
):
    i = wp.tid()
    load[i] += relaxation * residual[i]
    previous[i] = residual[i]


@wp.kernel(enable_backward=False)
def _direction_statistics(
    direction: wp.array[wp.float64],
    previous: wp.array[wp.float64],
    statistics: wp.array[wp.float64],
):
    i = wp.tid()
    difference = direction[i] - previous[i]
    if difference != wp.float64(0.0):
        wp.atomic_add(statistics, 3, difference * difference)
        wp.atomic_add(statistics, 4, previous[i] * difference)


@wp.kernel(enable_backward=False)
def _relax_preconditioned_interface(
    direction: wp.array[wp.float64],
    statistics: wp.array[wp.float64],
    initial_relaxation: wp.float64,
    has_previous: int,
    read_slot: int,
    relaxation_state: wp.array[wp.float64],
    load: wp.array[wp.float64],
    previous: wp.array[wp.float64],
):
    i = wp.tid()
    relaxation = initial_relaxation
    if has_previous != 0:
        relaxation = relaxation_state[read_slot]
        denominator = statistics[3]
        if denominator > wp.float64(1e-30):
            relaxation = -relaxation * statistics[4] / denominator
            # Comparisons preserve np.clip's NaN propagation. In particular,
            # an infinite numerator/denominator produces NaN, not a clamp.
            if relaxation < wp.float64(0.001):
                relaxation = wp.float64(0.001)
            elif relaxation > wp.float64(2.0):
                relaxation = wp.float64(2.0)
    load[i] += relaxation * direction[i]
    previous[i] = direction[i]
    # A different write slot prevents thread zero racing the other readers.
    # Reset branches never read old state, including after event fallback.
    if i == 0:
        relaxation_state[1 - read_slot] = relaxation


def _device_aitken_supported(model, fem_solver, state_in, preconditioned):
    return (
        preconditioned
        and model.device.is_cuda
        and not getattr(model, "requires_grad", False)
        and not getattr(fem_solver, "differentiable", False)
        and active_tape() is None
        and not any(
            array.requires_grad
            for array in (
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                state_in.joint_q,
                state_in.joint_qd,
                state_in.body_q,
                state_in.body_qd,
                state_in.body_f,
            )
        )
    )


def _array_storage(array):
    return None if array is None else (array.ptr, array.shape, array.strides, array.dtype)


class NeedleCouplingNotConverged(RuntimeError):
    """The interface iteration budget was exhausted without an accepted step."""


class NeedleTissueCoupler:
    """Implicit force exchange using the public FEM and CPU MuJoCo APIs.

    ``state_in.body_f`` and optional ``external_forces`` are external loads,
    kept separate from interface reactions. Torque residuals are divided by
    ``moment_length`` so that the convergence norm has force units. Events
    change the contact active set for the following accepted substep.

    ``NeedleCouplingNotConverged`` and pre-commit ``NeedleStepTooLarge`` rejections
    leave output state and accepted interaction history unchanged, restore the
    native MuJoCo integration state, and restore the FEM velocity warm start
    and incoming particle-force workspace. Physical event failures after
    convergence are retained in the interaction status for the caller to handle.
    This history-dependent coupling does not provide automatic derivatives.

    ``capture_fem=True`` reuses CUDA graphs for fixed-budget, non-differentiable
    Tet4 SolverFEM CG trials with block-diagonal preconditioning, including
    their triangle surface elements. CPU devices and unsupported configurations
    retain eager execution. Each graph executes the same complete FEM step;
    rigid integration, interface iteration and history remain uncaptured.
    Capture assumes fixed model/solver allocations. Call
    :meth:`invalidate_fem_graph` after replacing their array storage or cached
    resources. Supported in-place tangent/topology refreshes need no invalidation.

    ``compact_rigid=True`` batches native MuJoCo transfers for the supported
    single world FREE-body configuration. Other rigid models retain Newton's
    general adapter. Both paths use the same native model, integrator and
    rollback state; no rigid dynamics are replaced. Joint/body frames and
    topology must stay fixed for the lifetime of this coupler.

    ``surface_preconditioner=True`` optionally accelerates native first contact
    with a frozen facet/friction metric. It defaults off; physical residual
    and event acceptance are unchanged. Invalid/overflow recordings fall back
    to the original Anderson/Aitken search within the same iteration budget.

    ``shaft_preconditioner=True`` accounts for transverse shaft bearing and
    regularized friction in the interface correction for supported projected
    FEM/single-free-body models. It changes only the iteration: the physical
    force residual and acceptance tolerances remain authoritative. Unsupported
    models retain unpreconditioned Aitken relaxation.

    ``capture_coupling=True`` additionally keeps eligible established-tunnel
    trials on CUDA, including rigid motion, deformed-geometry queries, ALM and
    convergence decisions. It requires the supported single-body implicitfast
    configuration, conditional CG and the shaft preconditioner. Native rigid
    history and exact endpoint/event acceptance run once after convergence.
    Ambiguous boundary events restore the transaction and use the original
    host-controlled loop. This option does not change physical tolerances.
    """

    def __init__(
        self,
        model,
        fem_solver,
        rigid_solver,
        interaction,
        needle_body,
        needle_com,
        *,
        absolute_tolerance=2.0e-7,
        relative_tolerance=2.0e-4,
        max_iterations=40,
        relaxation=0.3,
        moment_length=0.01,
        capture_fem=False,
        capture_coupling=False,
        compact_rigid=True,
        shaft_preconditioner=False,
        surface_preconditioner=False,
        surface_tangent_capacity=4096,
    ):
        import mujoco

        self._mujoco = mujoco
        if not rigid_solver.use_mujoco_cpu or rigid_solver.mj_data is None:
            raise ValueError("NeedleTissueCoupler requires the CPU MuJoCo backend")
        if not synchronizes_every_step(rigid_solver):
            raise ValueError("NeedleTissueCoupler requires MuJoCo update_data_interval=1 for repeatable trials")
        if not np.isfinite([absolute_tolerance, relative_tolerance, moment_length, relaxation]).all():
            raise ValueError("Coupling parameters must be finite")
        if min(absolute_tolerance, moment_length, relaxation) <= 0 or relative_tolerance < 0:
            raise ValueError("Coupling tolerances, relaxation and length must be positive")
        if max_iterations < 1 or relaxation > 1:
            raise ValueError("Invalid coupling iteration budget")
        self.model, self.fem_solver, self.rigid_solver = model, fem_solver, rigid_solver
        self.interaction, self.needle_body, self.needle_com = interaction, needle_body, needle_com
        self.absolute_tolerance, self.relative_tolerance = absolute_tolerance, relative_tolerance
        self.max_iterations, self.relaxation = max_iterations, relaxation
        self.moment_length = moment_length
        self.capture_fem = bool(capture_fem)
        self.capture_coupling = bool(capture_coupling)
        self._captured_trials = None
        self._host_trial_forces = None
        self._host_trial_preparation = None
        self.coupling_graph_status = "not prepared" if capture_coupling else "disabled"
        self.coupling_graph_captures = self.coupling_graph_launches = self.coupling_graph_fallbacks = 0
        self.coupling_graph_verifications = 0
        self._rigid_bridge = (
            MuJoCoFreeBody(model, rigid_solver)
            if compact_rigid and model.device.is_cuda and MuJoCoFreeBody.supports(model, rigid_solver)
            else None
        )
        self.rigid_transfer_mode = "compact" if self._rigid_bridge is not None else "general"
        self._fem_graph_signature = None
        self._fem_graphs = OrderedDict()
        self.fem_graph_captures = self.fem_graph_launches = 0
        self.fem_graph_status = "not prepared" if capture_fem else "disabled"
        self._forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self._candidate = model.state()
        self._tet_step_guard = TetStepGuard(model) if model.tet_count else None
        self.tet_step_rejections = 0
        self.minimum_accepted_tet_j = np.inf
        self._endpoint_arrays = self._endpoint_snapshot = self._endpoint_equal = None
        self._input_particle_forces = wp.empty_like(self._candidate.particle_f)
        increment = getattr(fem_solver, "velocity_increment", None)
        self._increment = None if increment is None else wp.clone(increment)
        # Keep nodal loads and float64 Aitken arithmetic on the model device.
        # Only six convergence scalars cross to the host in each trial.
        self._load = wp.zeros(model.particle_count * 3 + 6, dtype=wp.float64, device=model.device)
        self._phase_load_predictor = None
        self._trial_load = wp.empty_like(self._load)
        self._value = wp.empty_like(self._load)
        self._residual = wp.empty_like(self._load)
        self._previous_residual = wp.empty_like(self._load)
        self._statistics = wp.zeros(6, dtype=wp.float64, device=model.device)
        self._aitken_relaxation = wp.empty(2, dtype=wp.float64, device=model.device)
        self._external_wrench = wp.empty_like(self._candidate.body_f)
        self._external_nodes = wp.zeros_like(self._forces)
        self._state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._native = np.empty(mujoco.mj_stateSize(rigid_solver.mj_model, self._state_spec))
        self.iterations = self.evaluations = self.maximum_iterations = 0
        self._surface_anderson_proposals = self._surface_anderson_rejections = 0
        self.residual = self.maximum_residual = 0.0
        self._shaft_preconditioner = (
            _ShaftPreconditioner(self) if shaft_preconditioner and _ShaftPreconditioner.supports(self) else None
        )
        if (
            type(fem_solver) is SolverFEM
            and getattr(fem_solver, "_fixed_particles", None) is not None
            and model.device.is_cuda
        ):
            # This coupler re-projects unchanged assembled values eagerly
            # before every captured launch (see _refresh_projection), so its
            # captured FEM trials may omit the per-trial projection.
            fem_solver._projection_before_graph = True
        self.shaft_preconditioner_status = (
            "enabled"
            if self._shaft_preconditioner is not None
            else ("unsupported" if shaft_preconditioner else "disabled")
        )
        self._surface_preconditioner = None
        if surface_preconditioner:
            from .surface_preconditioner import SurfaceContactPreconditioner

            if SurfaceContactPreconditioner.supports(self):
                self._surface_preconditioner = SurfaceContactPreconditioner(self, surface_tangent_capacity)
        self.surface_preconditioner_status = (
            "enabled"
            if self._surface_preconditioner is not None
            else ("unsupported" if surface_preconditioner else "disabled")
        )
        self._direction = (
            wp.empty_like(self._residual)
            if (self._shaft_preconditioner is not None or self._surface_preconditioner is not None)
            else None
        )

    def invalidate_fem_graph(self):
        """Discard captured FEM trials; the next eligible trial runs eagerly.

        Call between completed coupled steps after replacing internal solver
        resources, model array storage, force implementations, or model-change
        notifications that rebuild cached resources. Timestep and numerical
        setting changes select separate bounded cache entries. Supported in-place
        tangent/topology updates need no invalidation. Construct a new coupler
        if model capacities change. This does not change states or warm starts.
        """
        self._fem_graphs.clear()
        if self._phase_load_predictor is not None:
            self._phase_load_predictor.invalidate()
        self._fem_graph_signature = None
        self._host_trial_forces = None
        self._host_trial_preparation = None
        if self._captured_trials is not None:
            self._captured_trials.invalidate()
        self.fem_graph_status = "not prepared" if self.capture_fem else "disabled"

    def _fem_capture_key(self, state_in, control, dt):
        solver, model = self.fem_solver, self.model
        reason = None
        if not self.capture_fem:
            reason = "disabled"
        elif not model.device.is_cuda:
            reason = "CPU execution"
        elif type(solver) is not SolverFEM or getattr(solver.step, "__func__", None) is not SolverFEM.step:
            reason = "custom structural solver or step override"
        elif solver.differentiable or getattr(model, "requires_grad", False) or active_tape() is not None:
            reason = "differentiable configuration"
        elif (
            solver.linear_solver_check_every != 0
            or solver.solver_type != "cg"
            or solver.preconditioner_type != "block_diag"
        ):
            reason = "host residual checks or unsupported linear solver"
        elif model.spring_count or solver.hex_count or solver.enable_tet_activations:
            reason = "additional element or activation system"
        elif solver._streaming is not None or solver._component_scaling is not None:
            reason = "streaming or component-scaled system"
        elif solver.soft_surface_contacts is not None or solver._ground_plane is not None:
            reason = "additional surface or ground contact"
        elif control is None:
            reason = "implicit control allocation"
        elif any(array.requires_grad for array in (state_in.particle_q, state_in.particle_qd, state_in.particle_f)):
            reason = "gradient-enabled state"
        if reason is not None:
            self.invalidate_fem_graph()
            self.fem_graph_status = reason
            return None
        return (
            float(dt),
            solver.linear_solver_maxiter,
            solver.linear_solver_tolerance,
            solver.linear_solver_use_cuda_graph,
            solver.min_stretch,
            solver.max_stretch,
            solver.linear_damping,
            # First tangent preparation and changed Krylov settings may replace
            # this package-owned workspace while matrix pointers stay stable.
            id(solver._lin_solver),
        )

    def _step_fem(self, state_in, control, dt):
        # A difficult interval may visit several CG tolerances, then the next
        # interval starts at the configured tolerance again. Select its warmed
        # scratch before looking up a graph tied to that exact workspace.
        key = self._fem_capture_key(state_in, control, dt)
        if key is not None:
            # A rebuilt projection can replace its bound linear operator. Do
            # this before choosing any workspace or graph that references it.
            self._refresh_projection()
        reusable = key is not None and self.fem_solver._reuse_linear_workspace()
        if reusable:
            key = self._fem_capture_key(state_in, control, dt)
        if key is None or not reusable:
            # A real first trial initializes persistent Krylov/scratch storage;
            # no discarded warmup integration or extra interface iteration.
            self.fem_solver.step(state_in, self._candidate, control, None, dt, external_forces=self._forces)
            self._fem_graph_signature = None if key is None else self._fem_capture_key(state_in, control, dt)
            if key is not None:
                self.fem_graph_status = "warmed"
            return
        state_key = tuple(
            _array_storage(array) for array in (state_in.particle_q, state_in.particle_qd, state_in.particle_f)
        )
        graph_key = (key, state_key)
        entry = self._fem_graphs.get(graph_key)
        if entry is None:
            try:
                with wp.ScopedCapture(device=self.model.device) as capture:
                    self.fem_solver.step(state_in, self._candidate, control, None, dt, external_forces=self._forces)
            except Exception as error:
                self.invalidate_fem_graph()
                raise RuntimeError(
                    "FEM CUDA graph capture failed; disable capture_fem for this configuration"
                ) from error
            # Keep input and Krylov storage alive for every retained graph.
            entry = (
                capture.graph,
                (state_in.particle_q, state_in.particle_qd, state_in.particle_f),
                self.fem_solver._lin_solver,
            )
            self._fem_graphs[graph_key] = entry
            # Retain both alternating states at recurring timestep/tolerance
            # levels. LRU eviction bounds memory without discarding every graph
            # whenever a retry or an inner convergence level changes.
            if len(self._fem_graphs) > 32:
                self._fem_graphs.popitem(last=False)
            self.fem_graph_captures += 1
        else:
            self._fem_graphs.move_to_end(graph_key)
        self._fem_graph_signature = key
        wp.capture_launch(entry[0])
        self.fem_graph_launches += 1
        self.fem_graph_status = "replaying"

    def _restore_native(self):
        self._mujoco.mj_setState(self.rigid_solver.mj_model, self.rigid_solver.mj_data, self._native, self._state_spec)

    def _refresh_projection(self):
        """Re-project assembled values changed since the last captured launch.

        Captured FEM trials omit the projection of invariant values; every
        graph-replay site must therefore refresh the projected operator
        eagerly whenever the tangent values changed. Unchanged versions skip
        the launch; the projected values are identical either way.
        """
        solver = self.fem_solver
        fixed = getattr(solver, "_fixed_particles", None)
        if fixed is None or not getattr(solver, "_projection_before_graph", False):
            return
        if fixed.prepare(solver.A_bsr, values_version=solver._projection_values_version()):
            solver._lin_solver = None
            solver._lin_solver_key = None

    def _evaluate(self, state, dt, *, trial, start_state=None):
        self._forces.zero_()
        state.body_f.zero_()
        self.interaction.add_forces(
            state,
            self.needle_body,
            self.needle_com,
            self._forces,
            dt,
            trial=trial,
            start_state=start_state,
            implicit=True,
        )

    def _device_trials(self, state_in, control, dt):
        """Choose the narrowly supported graph path without weakening fallback."""
        if not self.capture_coupling:
            self.coupling_graph_status = "disabled"
            return None
        if self._fem_capture_key(state_in, control, dt) is None:
            self.coupling_graph_status = self.fem_graph_status
            return None
        if not self.fem_solver.linear_solver_use_cuda_graph or not wp.is_conditional_graph_supported():
            self.coupling_graph_status = "conditional CUDA graphs unavailable"
            return None
        if self._shaft_preconditioner is None or self._rigid_bridge is None:
            self.coupling_graph_status = "unsupported rigid or structural configuration"
            return None
        if any(
            array.requires_grad for array in (state_in.joint_q, state_in.joint_qd, state_in.body_q, state_in.body_qd)
        ):
            self.coupling_graph_status = "gradient-enabled rigid state"
            return None
        from .insertion import NeedleTissueInteraction

        if type(self.interaction) is not NeedleTissueInteraction or any(
            getattr(getattr(self.interaction, name), "__func__", None) is not getattr(NeedleTissueInteraction, name)
            for name in ("add_forces", "_add_forces", "_visited_sample_mask", "_tip_boundary_permission")
        ):
            self.coupling_graph_status = "custom interaction"
            return None
        if any(
            getattr(getattr(self, name), "__func__", None) is not getattr(NeedleTissueCoupler, name)
            for name in ("_step_fem", "_evaluate")
        ):
            self.coupling_graph_status = "custom trial implementation"
            return None
        if self.interaction.track_count == 0 or self.interaction._previous_tip is None:
            self.coupling_graph_status = "surface event preparation"
            return None
        from .coupling_graph import _CapturedNeedleTrials
        from .mujoco_free_body_gpu import MuJoCoFreeBodyGPU

        if self._captured_trials is None:
            if not MuJoCoFreeBodyGPU.supports(self.model, self.rigid_solver):
                self.coupling_graph_status = "unsupported native rigid settings"
                return None
            try:
                self._captured_trials = _CapturedNeedleTrials(self)
            except ValueError as error:
                self.coupling_graph_status = str(error)
                return None
        try:
            self._captured_trials.rigid.prepare(dt)
        except ValueError as error:
            self.coupling_graph_status = str(error)
            return None
        return self._captured_trials

    def _invalidate_query_cache(self):
        # Device trials update the BVHs while the host's position certificate
        # still describes its previous query. Rebuild that ownership at the
        # next exact endpoint evaluation or transactional fallback.
        self.interaction._query_positions = None
        self.interaction._device_query_positions = None
        self.interaction._device_query_points = None
        self.interaction._device_locations = {}

    def _restart_trials(self, state_in, tunnel_alm):
        if self._phase_load_predictor is not None:
            self._phase_load_predictor.cancel()
        self._restore_native()
        wp.copy(state_in.body_f, self._external_wrench)
        wp.copy(state_in.particle_f, self._input_particle_forces)
        wp.copy(self.fem_solver.velocity_increment, self._increment)
        wp.copy(self._trial_load, self._load)
        if self._shaft_preconditioner is not None:
            self._shaft_preconditioner.begin_step()
        self.interaction.discard_trial_forces()
        if tunnel_alm is not None:
            tunnel_alm.discard()
            tunnel_alm.begin_step()

    def _solve_device(self, graph, state_in, control, dt, tunnel_alm):
        """Return None for exact fallback, or the completed GPU solve result."""
        self._invalidate_query_cache()
        try:
            result = graph.solve(state_in, control, dt)
        finally:
            self._invalidate_query_cache()
            self.coupling_graph_captures = graph.captures
            self.coupling_graph_launches = graph.launches
        self.iterations, self.residual = int(result[0]), float(result[7])
        self.evaluations += self.iterations
        if tunnel_alm is not None:
            tunnel_alm.dual_updates = int(result[5])
            tunnel_alm.total_dual_updates += int(result[5])
            tunnel_alm.normal_error, tunnel_alm.dual_residual = float(result[8]), float(result[9])
        if result[4]:
            raise RuntimeError("Nonfinite needle/tissue interface state or force")
        if result[10]:
            self.coupling_graph_status = "exact surface-event fallback"
            self.coupling_graph_fallbacks += 1
            self._restart_trials(state_in, tunnel_alm)
            return None
        self.coupling_graph_status = "replaying"
        if not result[3]:
            return False

        # Advance native history once from the same start and final APPLIED
        # load. Preserve the graph's law forces, and require bitwise-identical
        # joint/body endpoints before retaining its certified force context.
        if self._endpoint_snapshot is None:
            self._endpoint_arrays = tuple(
                getattr(self._candidate, name).view(wp.uint32).flatten()
                for name in ("joint_q", "joint_qd", "body_q", "body_qd")
            )
            self._endpoint_snapshot = wp.empty(
                sum(array.size for array in self._endpoint_arrays), dtype=wp.uint32, device=self.model.device
            )
            self._endpoint_equal = wp.ones(1, dtype=int, device=self.model.device)
        wp.launch(
            _snapshot_rigid_endpoint,
            self._endpoint_snapshot.size,
            [*self._endpoint_arrays, self._endpoint_snapshot],
            device=self.model.device,
        )
        self._restore_native()
        wp.launch(
            _apply_rigid_interface_load,
            self.model.body_count,
            [
                self._trial_load,
                self._external_wrench,
                self.needle_body,
                self.model.particle_count,
                self.moment_length,
                state_in.body_f,
            ],
            device=self.model.device,
        )
        self._rigid_bridge.step(state_in, self._candidate, control, dt)
        self._endpoint_equal.fill_(1)
        wp.launch(
            _compare_rigid_endpoint,
            self._endpoint_snapshot.size,
            [*self._endpoint_arrays, self._endpoint_snapshot, self._endpoint_equal],
            device=self.model.device,
        )
        self.coupling_graph_verifications += 1
        if self._endpoint_equal.numpy()[0] and self.interaction._stage_certified_trial(
            self._candidate, self.needle_body, self.needle_com, graph.permissions.mask
        ):
            return True
        self.coupling_graph_status = "exact endpoint fallback"
        self.coupling_graph_fallbacks += 1
        self._restart_trials(state_in, tunnel_alm)
        return None

    def _host_force_graph(self, state_in, control, dt):
        """Use certified CUDA forces while retaining native articulated dynamics."""
        if not self.capture_coupling or self._fem_capture_key(state_in, control, dt) is None:
            return None
        if not wp.is_conditional_graph_supported():
            return None
        from .insertion import NeedleTissueInteraction

        if type(self.interaction) is not NeedleTissueInteraction or any(
            getattr(getattr(self.interaction, name), "__func__", None) is not getattr(NeedleTissueInteraction, name)
            for name in ("add_forces", "_add_forces", "_visited_sample_mask", "_tip_boundary_permission")
        ):
            return None
        if any(
            getattr(getattr(self, name), "__func__", None) is not getattr(NeedleTissueCoupler, name)
            for name in ("_step_fem", "_evaluate")
        ):
            return None
        if any(
            array.requires_grad for array in (state_in.joint_q, state_in.joint_qd, state_in.body_q, state_in.body_qd)
        ):
            return None
        if self.interaction._previous_tip is None:
            return None
        from .host_trial_forces import CapturedHostTrialForces

        if self._host_trial_forces is None:
            self._host_trial_forces = CapturedHostTrialForces(self)
        self._host_trial_forces.prepare(dt)
        return self._host_trial_forces

    def _solve_host(
        self, state_in, control, dt, configured_tolerance, tunnel_alm, preconditioned, *, capture_forces=True
    ):
        """Original host-controlled trial loop; also the exact event fallback."""
        surface_pc = self._surface_preconditioner
        surface_available = surface_pc is not None and not preconditioned and surface_pc.begin_trials(state_in, control)
        surface_selected = False
        surface_device_aitken = surface_available and _device_aitken_supported(
            self.model, self.fem_solver, state_in, True
        )
        force_graph = self._host_force_graph(state_in, control, dt) if capture_forces else None
        from .host_trial_prepare import CapturedTrialPreparation

        if self._host_trial_preparation is None:
            self._host_trial_preparation = CapturedTrialPreparation(self)
        prepare_trial = self._host_trial_preparation.bind(state_in)
        overlap_fem = (
            self._rigid_bridge is None
            and type(self.rigid_solver) is SubcycledMuJoCo
            and getattr(self.rigid_solver.step, "__func__", None) is SubcycledMuJoCo.step
            and type(self.fem_solver) is SolverFEM
            and getattr(self.fem_solver.step, "__func__", None) is SolverFEM.step
            and getattr(self._step_fem, "__func__", None) is NeedleTissueCoupler._step_fem
        )

        def enqueue_fem():
            self._step_fem(state_in, control, dt)

        relaxation, mixing_start = self.relaxation, 1
        acceleration = None
        if (
            getattr(self, "_surface_anderson_enabled", True)
            and not preconditioned
            and type(self.rigid_solver) is SubcycledMuJoCo
            and not getattr(self.interaction, "punctured", True)
            and configured_tolerance is not None
            and not getattr(self.model, "requires_grad", False)
            and not getattr(self.fem_solver, "differentiable", False)
            and active_tape() is None
            and not any(
                array is not None and array.requires_grad
                for state in (state_in, self._candidate)
                for array in (
                    state.particle_q,
                    state.particle_qd,
                    state.particle_f,
                    state.joint_q,
                    state.joint_qd,
                    state.body_q,
                    state.body_qd,
                    state.body_f,
                )
            )
            and not any(
                array is not None and array.requires_grad
                for array in (
                    getattr(control, "joint_f", None),
                    getattr(control, "joint_target_q", None),
                    getattr(control, "joint_target_qd", None),
                    getattr(control, "tet_activations", None),
                    getattr(getattr(control, "mujoco", None), "ctrl", None),
                )
            )
        ):
            from .surface_anderson import SurfaceAnderson

            acceleration = SurfaceAnderson()
        device_aitken = _device_aitken_supported(self.model, self.fem_solver, state_in, preconditioned)
        relaxation_slot = 0
        converged = False
        reuse_primal = False
        for iteration in range(1, self.max_iterations + 1):
            if configured_tolerance is not None and iteration > mixing_start and (iteration - mixing_start) % 8 == 0:
                tolerance = max(min(configured_tolerance, 1e-6), self.fem_solver.linear_solver_tolerance * 0.1)
                if tolerance < self.fem_solver.linear_solver_tolerance:
                    self.fem_solver.linear_solver_tolerance = tolerance
                    # Aitken differences must belong to the same approximate
                    # force map; tighter inner solves begin a fresh history.
                    relaxation, mixing_start = self.relaxation, iteration
                    if acceleration is not None:
                        acceleration.reset()
            if not reuse_primal:
                self._restore_native()
                prepare_trial()
                # Reuse the preceding trial's numerical CG guess. Physical
                # integration still starts from state_in; rejection restores
                # the incoming guess together with the native rigid state.
                if overlap_fem:
                    # Import native coordinates before enqueueing FEM. CPU
                    # contact integration can then overlap the structural GPU
                    # solve; its endpoint export stays ordered on this stream.
                    self.rigid_solver.step(state_in, self._candidate, control, None, dt, before_native_step=enqueue_fem)
                else:
                    self._step_fem(state_in, control, dt)
                    if self._rigid_bridge is None:
                        self.rigid_solver.step(state_in, self._candidate, control, None, dt)
                    else:
                        self._rigid_bridge.step(state_in, self._candidate, control, dt)
            if force_graph is None:
                self._evaluate(self._candidate, dt, trial=True)
                self._statistics.zero_()
                wp.launch(
                    _interface_residual,
                    self._load.size,
                    [
                        self._forces,
                        self._candidate.body_f,
                        self.needle_body,
                        self.moment_length,
                        self._trial_load,
                        self._previous_residual,
                        int(iteration > mixing_start and not preconditioned),
                        self._value,
                        self._residual,
                        self._statistics,
                    ],
                    device=self.model.device,
                )
                packet = surface_pc.pack_statistics(self._statistics) if surface_available else self._statistics
                statistics = packet.numpy()
            else:
                statistics = force_graph.evaluate(
                    dt, iteration > mixing_start and not preconditioned, fresh=not reuse_primal
                )
            self.evaluations += 1
            self.iterations = iteration
            self.residual = float(statistics[0])
            if statistics[5]:
                raise RuntimeError("Nonfinite needle/tissue interface force")
            tolerance = self.absolute_tolerance + self.relative_tolerance * max(
                float(statistics[1]), float(statistics[2])
            )
            if self.residual <= tolerance:
                if tunnel_alm is None or tunnel_alm.converged(self.absolute_tolerance, self.relative_tolerance):
                    if force_graph is not None:
                        force_graph.stage()
                    converged = True
                    break
                # A dual update changes the proximal force map. Re-solve
                # from the same physical start with a fresh Aitken history;
                # every primal evaluation still consumes the shared budget.
                tunnel_alm.update_duals()
                relaxation, mixing_start = self.relaxation, iteration + 1
                if acceleration is not None:
                    acceleration.reset()
                reuse_primal = force_graph is not None
                continue
            # Vector Aitken relaxation adapts to the combined structural,
            # rigid and regularized-friction response without changing any
            # physical force. Positive relaxation handles contact switches.
            reuse_primal = False
            direction = self._residual
            selected_surface = bool(surface_available and statistics[7] == 1 and statistics[6] > 0)
            if selected_surface:
                surface_pc.active = True
                selected_surface = surface_pc.apply(self._candidate, dt, self._residual, self._direction)
                surface_pc.active = selected_surface
            if selected_surface != surface_selected:
                surface_pc.selection_changes += 1
                # Preconditioned and raw histories are different maps. Never
                # mix differences across empty/overflow/capped/contact switches.
                relaxation, mixing_start, relaxation_slot = self.relaxation, iteration, 0
                self._previous_residual.zero_()
                if acceleration is not None:
                    acceleration.reset()
                surface_selected = selected_surface
            trial_preconditioned = preconditioned or selected_surface
            trial_device_aitken = device_aitken or (selected_surface and surface_device_aitken)
            if trial_preconditioned:
                if preconditioned:
                    self._shaft_preconditioner.apply(self._candidate, dt, self._residual, self._direction)
                direction = self._direction
                if selected_surface:
                    from .surface_preconditioner import clear_direction_statistics

                    wp.launch(clear_direction_statistics, 1, [self._statistics], device=self.model.device)
                if iteration > mixing_start:
                    wp.launch(
                        _direction_statistics,
                        self._load.size,
                        [direction, self._previous_residual, self._statistics],
                        device=self.model.device,
                    )
                    if not trial_device_aitken:
                        statistics = self._statistics.numpy()
            if trial_device_aitken:
                wp.launch(
                    _relax_preconditioned_interface,
                    self._load.size,
                    [
                        direction,
                        self._statistics,
                        self.relaxation,
                        int(iteration > mixing_start),
                        relaxation_slot,
                        self._aitken_relaxation,
                        self._trial_load,
                        self._previous_residual,
                    ],
                    device=self.model.device,
                )
                relaxation_slot = 1 - relaxation_slot
                continue
            if iteration > mixing_start:
                denominator = float(statistics[3])
                if denominator > 1e-30:
                    relaxation = float(
                        np.clip(-relaxation * statistics[4] / denominator, 0.001, 2.0 if trial_preconditioned else 1.0)
                    )
            if (
                not trial_preconditioned
                and acceleration is not None
                and self.fem_solver.linear_solver_tolerance <= min(configured_tolerance, 1e-6)
            ):
                residual = self._residual.numpy()
                proposals, rejections = acceleration.proposals, acceleration.rejections
                proposal = acceleration.propose(self._trial_load.numpy(), residual, relaxation)
                self._surface_anderson_proposals += acceleration.proposals - proposals
                self._surface_anderson_rejections += acceleration.rejections - rejections
                if proposal is not None:
                    self._trial_load.assign(proposal)
                    self._previous_residual.assign(residual)
                    relaxation, mixing_start = self.relaxation, iteration + 1
                    continue
            wp.launch(
                _relax_interface,
                self._load.size,
                [direction, relaxation, self._trial_load, self._previous_residual],
                device=self.model.device,
            )
        return converged

    def step(self, state_in, state_out, control, dt, *, external_forces=None):
        """Advance one shared step, raising if the endpoint force solve fails."""
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("Coupled timestep must be finite and positive")
        wp.copy(self._external_wrench, state_in.body_f)
        if external_forces is None:
            self._external_nodes.zero_()
        else:
            wp.copy(self._external_nodes, external_forces)
        self._mujoco.mj_getState(self.rigid_solver.mj_model, self.rigid_solver.mj_data, self._native, self._state_spec)
        if self._increment is not None:
            wp.copy(self._increment, self.fem_solver.velocity_increment)
        wp.copy(self._input_particle_forces, state_in.particle_f)
        wp.copy(self._trial_load, self._load)
        configured_tolerance = (
            self.fem_solver.linear_solver_tolerance if isinstance(self.fem_solver, SolverFEM) else None
        )
        tunnel_alm = getattr(self.interaction, "tunnel_alm", None)
        converged = False
        committed = False
        predictor = self._phase_load_predictor
        native_input = None
        try:
            if (
                self._rigid_bridge is None
                and type(self.rigid_solver) is SubcycledMuJoCo
                and getattr(self.rigid_solver.step, "__func__", None) is SubcycledMuJoCo.step
            ):
                native_input = self.rigid_solver.begin_repeated_input(
                    state_in, self._candidate, control, self.needle_body
                )
            if predictor is not None:
                from .phase_predictor import coupler_signature

                predictor.begin(
                    self._load, self._trial_load, dt, coupler_signature(self), self.interaction._track_engaged
                )
            if tunnel_alm is not None:
                tunnel_alm.begin_step()
            preconditioned = False
            if self._surface_preconditioner is not None:
                self._surface_preconditioner.begin_step()
            if self._shaft_preconditioner is not None:
                self._shaft_preconditioner.begin_step()
                preconditioned = self._shaft_preconditioner.active
            graph = self._device_trials(state_in, control, dt)
            converged = None if graph is None else self._solve_device(graph, state_in, control, dt, tunnel_alm)
            if converged is None:
                from .host_trial_forces import TrialForceFallback

                try:
                    converged = self._solve_host(
                        state_in, control, dt, configured_tolerance, tunnel_alm, preconditioned
                    )
                except TrialForceFallback:
                    # Device force evaluation never commits history. Restore
                    # native integration, numerical guesses and ALM trial state
                    # before the original swept-event/authorization oracle.
                    self._restart_trials(state_in, tunnel_alm)
                    self._invalidate_query_cache()
                    self.fem_solver.linear_solver_tolerance = configured_tolerance
                    converged = self._solve_host(
                        state_in, control, dt, configured_tolerance, tunnel_alm, preconditioned, capture_forces=False
                    )
            if not converged:
                detail = f"force residual {self.residual:.3g} N"
                if tunnel_alm is not None and tunnel_alm.dual_updates:
                    detail += (
                        f", last converged-primal tunnel normal residual {tunnel_alm.normal_error:.3g} m"
                        f" and dual force residual {tunnel_alm.dual_residual:.3g} N"
                    )
                raise NeedleCouplingNotConverged(
                    f"Needle/tissue coupling did not converge in {self.max_iterations} trials ({detail})"
                )
            # Interface convergence does not make a linearly implicit FEM step
            # inversion-free. Reject unresolved material motion while native
            # state, warm starts, events and output still belong to this trial.
            minimum_tet_j = np.inf
            if self._tet_step_guard is not None:
                try:
                    minimum_tet_j = self._tet_step_guard.check(state_in, self._candidate)
                except NeedleStepTooLarge:
                    self.tet_step_rejections += 1
                    raise
            self.interaction.check_step_resolution(state_in, self._candidate, self.needle_body)
            self.interaction.retain_trial_forces(state_in, self.needle_body)
            self.interaction.accept(self._candidate, self.needle_body)
            if tunnel_alm is not None:
                tunnel_alm.commit()
            wp.copy(self._candidate.body_f, self._external_wrench)
            state_out.assign(self._candidate)
            committed = True
            if predictor is not None:
                predictor.commit(self._value)
            self.minimum_accepted_tet_j = min(self.minimum_accepted_tet_j, minimum_tet_j)
            wp.copy(self._load, self._value)
            self.maximum_iterations = max(self.maximum_iterations, self.iterations)
            self.maximum_residual = max(self.maximum_residual, self.residual)
        finally:
            if self._surface_preconditioner is not None:
                self.interaction._surface_tangent_anchors = None
                self._surface_preconditioner.available = False
            if native_input is not None:
                native_input.end_repeated_input()
            if configured_tolerance is not None:
                self.fem_solver.linear_solver_tolerance = configured_tolerance
            wp.copy(state_in.body_f, self._external_wrench)
            if not committed:
                if predictor is not None:
                    predictor.discard()
                if tunnel_alm is not None:
                    tunnel_alm.discard()
                self.interaction.discard_trial_forces()
                self._restore_native()
                wp.copy(state_in.particle_f, self._input_particle_forces)
                if self._increment is not None:
                    wp.copy(self.fem_solver.velocity_increment, self._increment)
