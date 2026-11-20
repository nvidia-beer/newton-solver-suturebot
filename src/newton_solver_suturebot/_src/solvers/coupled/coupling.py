# SPDX-License-Identifier: Apache-2.0
"""Converged partitioned coupling of a dynamic needle and FEM material.

The interaction is evaluated at the shared end-of-step state. Every trial
restarts both integrators from the same accepted state; only the converged
trial advances material history. No controller or target geometry enters here.
"""

import time
from collections import OrderedDict

import numpy as np
import warp as wp

from ...utils.capture import ScopedCapture, note_release
from ...utils.step_time import StepTime
from ..fem import SolverFEM
from ..fem.differentiation import active_tape
from ..mujoco.experimental_rigid_backend import is_gpu_backend, supported_subcycled
from ..mujoco.mujoco_free_body import MuJoCoFreeBody, synchronizes_every_step
from ..mujoco.native_step_guard import restore_native_warnings, snapshot_native_warnings
from ..mujoco.subcycled_mujoco import SubcycledMuJoCo
from ..mujoco.subcycled_mujoco_warp import WarpIntegrationCheckpoint
from .acceleration.shaft_preconditioner import _ShaftPreconditioner
from .interface.insertion import NeedleStepTooLarge
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


_HOST_WRENCH_INTERFACE_KERNELS = (_apply_interface_load, _apply_rigid_interface_load)


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
    """The interface iteration ended without an accepted step: budget exhausted or diverged."""


class _Attempt:
    """What one attempt of a coupled step holds between its phases (host values only).

    ``_open_attempt`` sets the arguments and the objects that take part in the attempt;
    ``_begin_attempt`` sets ``native_input`` and ``guessed``; ``step`` sets ``preconditioned``
    where host-controlled trials first run; ``_close_attempt`` reads them.
    """

    __slots__ = (
        "affine",
        "certificate",
        "configured_tolerance",
        "control",
        "dt",
        "guessed",
        "held",
        "native_input",
        "preconditioned",
        "predictor",
        "state_in",
        "tunnel_alm",
    )

    def __init__(self, state_in, control, dt, configured_tolerance, tunnel_alm, predictor, affine, held, certificate):
        self.state_in, self.control, self.dt = state_in, control, dt
        # FEM linear tolerance to restore at the end (None: not a SolverFEM).
        self.configured_tolerance = configured_tolerance
        self.tunnel_alm = tunnel_alm  # tunnel multipliers of the interaction, or None
        self.predictor = predictor  # phase load predictor, or None
        self.affine = affine  # affine predictor manager of the FEM solver, or None
        self.held = held  # held start of the FEM solver, or None
        self.certificate = certificate  # certified linear solve of the FEM solver, or None
        self.native_input = None  # repeated native input of the rigid solver, once begun
        self.guessed = False  # the phase predictor wrote a start guess into the trial load
        # The shaft preconditioner is active in this attempt; None until host-controlled trials ask.
        self.preconditioned = None


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

    ``preconditioned_anderson=True`` optionally mixes preconditioned native
    shaft/contact guesses using the original physical residual. It defaults
    off; gradient-enabled inputs and unsupported solvers retain their previous
    iteration. Proposals never bypass physical acceptance or the shared budget.

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

    ``trial_budget`` (default ``None``: off) bounds the trials of one loop. A loop
    that has evaluated that many trials without converging, and has not failed,
    keeps its best evaluated trial instead of going on to ``max_iterations``: the
    trial with the smallest force residual of the current tunnel-multiplier epoch
    (a multiplier update changes the force map, so the best restarts there; a
    trial that passes the force test and only waits for such an update is the best
    of its epoch). The last trial is kept as it is if it is that best one;
    otherwise the best load is applied once more and that trial is kept, whatever
    residual it shows. A kept trial takes the acceptance path of a converged one,
    with no correction and no multiplier update after it; its interface force is
    not converged to the tolerances. The rule bounds no force, load or state. A
    nonfinite or invalid trial, a tip moved beyond the step resolution, the
    divergence stop and a failed tissue-solver certificate are handled as without
    a budget and are never kept.

    Surface events with a budget. A device trial that asks for the host oracle is
    never kept, and a request that ends the device trials ends them as without a
    budget. A device search past a surface event that uses up the budget ends
    there: if its best trial still carries a request, the load of that trial
    starts the host oracle, as the load of a converged search does, and no trial
    of the search is accepted (without a budget a search that does not converge
    hands over nothing). The host oracle is the host-controlled loop and follows
    the rule itself: a trial it keeps is accepted with the surface event of that
    trial (puncture, reuse of an opening, entry, exit), which is then decided at
    a state that is not converged. A host trial in which the interaction reports
    a failure (status ``"failed"``) is never kept; that loop goes on as without a
    budget.

    ``search_past_puncture=True`` (default off) lets the device trials go on past a
    contact force at the puncture threshold, as they go on past a request that concerns
    the path of the tip. Outside the tissue the trial forces of the device are those of
    the host evaluation below and above the threshold: a puncture changes no force
    before the step that follows its acceptance. The host oracle still decides the
    puncture, from the load the search reached; no trial of the search is accepted.

    ``oracle_at_device_trial=True`` (default off) hands a device search that ended with
    a load for the host oracle (converged past a surface event, or ended by the trial
    budget) to the oracle as it stands: the candidate state, the rigid backend state and
    the tunnel multipliers of its last trial are kept, and the first trial of the
    host-controlled loop is the host evaluation of that state. The tissue solve and the
    rigid interval of that trial are not repeated. The oracle decides the surface event
    in that evaluation, with the tolerances of every host trial; if it does not accept
    the trial, the loop corrects the load and goes on with complete host trials. If the
    oracle cannot use the trial, the interval starts again from the accepted load, as
    without this option. The native shadow of the search metric is refreshed from the
    accepted rigid state before the first correction instead of before the first trial.
    ``coupling_oracle_device_trials`` counts these loops and
    ``coupling_oracle_device_accepts`` those that ended in their first trial.

    ``coupling_budget_keeps`` counts the trial loops the budget ended with a kept
    trial (device and host loops). The acceptance checks of the step can still
    refuse such a trial, as they can a converged one; the loop is counted all the
    same. ``coupling_budget_searches`` counts the device searches the budget ended
    whose load started the host oracle. ``coupling_budget_residual`` is the
    largest force residual [N] of a kept trial that became an accepted step, and
    ``coupling_budget_events`` counts the surface events those steps committed,
    by kind.

    Graphs are captured where a step first needs them: the first use of a binding
    runs as written, the next use captures. :meth:`capture_graphs` issues those
    uses ahead of the first step, inside attempts that end like a rejected step:
    the trial preparation, the rigid interval, the held start of the FEM solver,
    the FEM step of the host-controlled trials at every inner tolerance, and the
    trial graph of the device trials.
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
        preconditioned_anderson=False,
        trial_budget=None,
        search_past_puncture=False,
        oracle_at_device_trial=False,
    ):
        import mujoco

        self._mujoco = mujoco
        gpu_rigid = is_gpu_backend(rigid_solver)
        if (not rigid_solver.use_mujoco_cpu and not gpu_rigid) or rigid_solver.mj_data is None:
            raise ValueError("NeedleTissueCoupler requires native MuJoCo or the exact experimental GPU adapter")
        if not synchronizes_every_step(rigid_solver):
            raise ValueError("NeedleTissueCoupler requires MuJoCo update_data_interval=1 for repeatable trials")
        if not np.isfinite([absolute_tolerance, relative_tolerance, moment_length, relaxation]).all():
            raise ValueError("Coupling parameters must be finite")
        if min(absolute_tolerance, moment_length, relaxation) <= 0 or relative_tolerance < 0:
            raise ValueError("Coupling tolerances, relaxation and length must be positive")
        if max_iterations < 1 or relaxation > 1:
            raise ValueError("Invalid coupling iteration budget")
        if trial_budget is not None and (
            isinstance(trial_budget, bool)
            or not isinstance(trial_budget, (int, np.integer))
            or trial_budget < 2
            or trial_budget >= max_iterations
        ):
            raise ValueError("trial_budget must be None or an integer of at least 2 and below max_iterations")
        self.model, self.fem_solver, self.rigid_solver = model, fem_solver, rigid_solver
        self.interaction, self.needle_body, self.needle_com = interaction, needle_body, needle_com
        # The time step of the coupled step as device data. The FEM step, the rigid
        # interval, the interface forces and the correction read these arrays, so no
        # captured graph of the step holds the value of dt. The FEM solver's object is
        # taken, which is the same one for every coupler built on that solver.
        step_time = getattr(fem_solver, "step_time", None)
        if not isinstance(step_time, StepTime):
            step_time = getattr(interaction, "step_time", None)
        self.step_time = step_time if isinstance(step_time, StepTime) else StepTime(model.device)
        self._share_step_time()
        self.absolute_tolerance, self.relative_tolerance = absolute_tolerance, relative_tolerance
        self.max_iterations, self.relaxation = max_iterations, relaxation
        self.trial_budget = None if trial_budget is None else int(trial_budget)
        self.moment_length = moment_length
        self.preconditioned_anderson = bool(preconditioned_anderson)
        self.capture_fem = bool(capture_fem)
        self.capture_coupling = bool(capture_coupling)
        # Capture independent stages of a device trial as concurrent graph branches
        # where the rigid backend allows it. Only the execution order changes.
        self.parallel_trial_branches = True
        self._captured_trials = None
        self._host_trial_forces = None
        self._host_trial_preparation = None
        self.coupling_graph_status = "not prepared" if capture_coupling else "disabled"
        self.coupling_graph_captures = self.coupling_graph_launches = self.coupling_graph_fallbacks = 0
        # Which device certificate asked for the exact host path, by reason text.
        self.coupling_graph_fallback_reasons = {}
        # Device searches that converged past a surface event and started the host oracle.
        self.coupling_graph_event_searches = 0
        # Searches whose load the oracle could not use; the step restarted from the accepted load.
        self.coupling_graph_event_restarts = 0
        # Trial loops ended because their applied load ran away (device and host loops).
        self.coupling_divergence_stops = 0
        self._diverged_after = 0
        # Trial loops the trial budget ended with their best evaluated trial kept (device and
        # host loops), whether or not the checks of the step then accepted that trial.
        self.coupling_budget_keeps = 0
        # Device searches past a surface event that the budget ended and whose best load
        # started the host oracle; no trial of such a search is kept.
        self.coupling_budget_searches = 0
        # Of the kept trials that became an accepted step: the largest force residual [N],
        # and the surface events those steps committed, by kind.
        self.coupling_budget_residual = 0.0
        self.coupling_budget_events = {}
        # The last trial of the current loop is one the budget kept.
        self._budget_kept = False
        # Load of the best trial of a host loop; allocated by the start-up call (capture_graphs),
        # or with the first budgeted host loop of a coupler that had none.
        self._best_load = None
        # Device trial loops ended because a trial moved the tip beyond the step resolution.
        self.coupling_unresolved_trials = 0
        # Let device trials converge past a surface-event request before the host decides it.
        self.search_past_surface_events = True
        # The search also goes on past a contact force at the puncture threshold (see the class text).
        self.search_past_puncture = bool(search_past_puncture)
        # The host oracle evaluates the last trial of a device search as it stands (see the class text).
        self.oracle_at_device_trial = bool(oracle_at_device_trial)
        # Host loops whose first trial was the last trial of a device search, and those of them
        # that the oracle accepted in that trial.
        self.coupling_oracle_device_trials = 0
        self.coupling_oracle_device_accepts = 0
        self._first_trial_event = self._event_load_kept = self._device_trial_stands = False
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
        self._native_warnings = None
        self._gpu_checkpoint = WarpIntegrationCheckpoint(rigid_solver) if gpu_rigid else None
        if gpu_rigid:
            if model.world_count != 1:
                raise ValueError("Experimental coupling currently requires one GPU world")
            self.rigid_transfer_mode = "mujoco-warp-experimental"
        self.iterations = self.evaluations = self.maximum_iterations = 0
        # Trial loops run so far (device and host-controlled loops) and the host-clock time [s]
        # spent in them, with the handling of their result, as ``evaluations`` counts their
        # trials: a caller that budgets its own time takes the differences. No device read.
        self.trial_loops = 0
        self.loop_seconds = 0.0
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
            from .acceleration.surface_preconditioner import SurfaceContactPreconditioner

            if SurfaceContactPreconditioner.supports(self):
                self._surface_preconditioner = SurfaceContactPreconditioner(self, surface_tangent_capacity)
        if self._surface_preconditioner is not None and self._shaft_preconditioner is not None:
            from .acceleration.combined_contact_preconditioner import CombinedContactPreconditioner

            self._shaft_preconditioner = CombinedContactPreconditioner(self, self._surface_preconditioner)
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

    def _share_step_time(self):
        """Make every stage of the coupled step read the arrays of ``self.step_time``.

        Host bookkeeping, outside capture: the interaction's force kernels and the GPU
        rigid interval take this coupler's object in place of their own.
        """
        step_time = self.step_time
        if self.interaction is not None and getattr(self.interaction, "step_time", None) is not step_time:
            self.interaction.step_time = step_time
        adopt = getattr(self.rigid_solver, "adopt_step_time", None)
        if adopt is not None:
            adopt(step_time)

    def invalidate_fem_graph(self):
        """Discard captured FEM trials; the next eligible trial runs eagerly.

        Call between completed coupled steps after replacing internal solver
        resources, model array storage, force implementations, or model-change
        notifications that rebuild cached resources. Numerical setting changes
        select separate bounded cache entries; the time step is device data that
        every captured launch reads, so one entry serves every ``dt``. Supported in-place
        tangent/topology updates need no invalidation. Construct a new coupler
        if model capacities change. This does not change states or warm starts.
        """
        if self._shaft_preconditioner is not None:
            self._shaft_preconditioner.invalidate_preparation()
        self._fem_graphs.clear()
        if self._phase_load_predictor is not None:
            self._phase_load_predictor.invalidate()
        self._fem_graph_signature = None
        if self._host_trial_forces is not None:
            note_release(len(self._host_trial_forces.graphs))
        self._host_trial_forces = None
        if self._host_trial_preparation is not None:
            note_release(len(self._host_trial_preparation.graphs))
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
        # The time step is not an entry: the captured step reads it from device data
        # (StepTime), which _step_fem writes before the launch. What the time step
        # still decides at capture is whether the held start stands for this request
        # (SolverFEM.step and implicit_integration branch on it), so those two
        # decisions are the entry of the held start.
        held = getattr(solver, "_held_start", None)
        held_branches = None if held is None else (held.matches(state_in, dt, control), held.matches(state_in, dt))
        return (
            solver.linear_solver_maxiter,
            solver.linear_solver_tolerance,
            solver.linear_solver_use_cuda_graph,
            solver.min_stretch,
            solver.max_stretch,
            solver.linear_damping,
            # First tangent preparation and changed Krylov settings may replace
            # this package-owned workspace while matrix pointers stay stable.
            id(solver._lin_solver),
            None if getattr(solver, "_certified_linear", None) is None else solver._certified_linear.key(),
            None if getattr(solver, "_affine_predictor", None) is None else solver._affine_predictor.capture_key(),
            None if held is None or not any(held_branches) else (held.capture_key(), held_branches),
        )

    def _step_fem(self, state_in, control, dt):
        # A difficult interval may visit several CG tolerances, then the next
        # interval starts at the configured tolerance again. Select its warmed
        # scratch before looking up a graph tied to that exact workspace.
        # The captured step reads the time step from device data: a replay needs it written.
        self.step_time.set(dt)
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
                with ScopedCapture(device=self.model.device) as capture:
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
                ()
                if getattr(self.fem_solver, "_affine_predictor", None) is None
                else self.fem_solver._affine_predictor.capture_owners,
                () if getattr(self.fem_solver, "_held_start", None) is None else self.fem_solver._held_start.owners,
            )
            self._fem_graphs[graph_key] = entry
            # Retain both alternating states at recurring tolerance levels; a
            # retry at another time step replays the same graphs. LRU eviction
            # bounds memory without discarding every graph whenever an inner
            # convergence level changes.
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
        if getattr(self, "_gpu_checkpoint", None) is not None:
            self._gpu_checkpoint.restore()
            return
        self._mujoco.mj_setState(self.rigid_solver.mj_model, self.rigid_solver.mj_data, self._native, self._state_spec)
        if self._native_warnings is not None:
            restore_native_warnings(self.rigid_solver.mj_data, self._native_warnings)

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

    def _device_trials_supported(self, state_in, control, dt):
        """Whether the configuration admits device trials; a refusal is left in the graph status.

        These are the refusals that do not look at the needle's accepted history. The
        gate on that history is in :meth:`_device_trials`.
        """
        if not self.capture_coupling:
            self.coupling_graph_status = "disabled"
            return False
        gpu_rigid = self._gpu_checkpoint is not None
        if getattr(self.interaction, "cooperative_shaft_projection", False) and not gpu_rigid:
            # This optional packet is owned by the host force/preparation graph
            # caches. The independent all-device coupling graph retains its
            # original projection path until its owner contract is integrated.
            self.coupling_graph_status = "cooperative projection uses host-controlled trials"
            return False
        if self._fem_capture_key(state_in, control, dt) is None:
            self.coupling_graph_status = self.fem_graph_status
            return False
        if not self.fem_solver.linear_solver_use_cuda_graph or not wp.is_conditional_graph_supported():
            self.coupling_graph_status = "conditional CUDA graphs unavailable"
            return False
        if gpu_rigid and getattr(self, "_gpu_trial_policy", "current") not in ("device-aitken", "device-anderson"):
            self.coupling_graph_status = "current Anderson policy uses host-controlled GPU trials"
            return False
        if self._shaft_preconditioner is None or (self._rigid_bridge is None and not gpu_rigid):
            self.coupling_graph_status = "unsupported rigid or structural configuration"
            return False
        if any(
            array.requires_grad for array in (state_in.joint_q, state_in.joint_qd, state_in.body_q, state_in.body_qd)
        ):
            self.coupling_graph_status = "gradient-enabled rigid state"
            return False
        from .interface.insertion import NeedleTissueInteraction

        if type(self.interaction) is not NeedleTissueInteraction or any(
            getattr(getattr(self.interaction, name), "__func__", None) is not getattr(NeedleTissueInteraction, name)
            for name in ("add_forces", "_add_forces", "_visited_sample_mask", "_tip_boundary_permission")
        ):
            self.coupling_graph_status = "custom interaction"
            return False
        if any(
            getattr(getattr(self, name), "__func__", None) is not getattr(NeedleTissueCoupler, name)
            for name in ("_step_fem", "_evaluate")
        ):
            self.coupling_graph_status = "custom trial implementation"
            return False
        return True

    def _device_trials(self, state_in, control, dt):
        """Choose the narrowly supported graph path without weakening fallback."""
        if not self._device_trials_supported(state_in, control, dt):
            return None
        gpu_rigid = self._gpu_checkpoint is not None
        if (self.interaction.track_count == 0 and not gpu_rigid) or self.interaction._previous_tip is None:
            self.coupling_graph_status = "surface event preparation"
            return None
        return self._bound_device_trials(state_in, control, dt)

    def _bound_device_trials(self, state_in, control, dt):
        """The trial-graph owner of a supported configuration, made if missing and prepared for ``dt``.

        The part of :meth:`_device_trials` behind its gate on the accepted history; a
        refusal is left in the graph status. :meth:`capture_graphs` enters here ahead of
        the first step, where that history does not exist yet.
        """
        from ..mujoco.mujoco_free_body_gpu import MuJoCoFreeBodyGPU
        from .coupling_graph import _CapturedNeedleTrials

        gpu_rigid = self._gpu_checkpoint is not None
        if gpu_rigid:
            from .mujoco_warp_trials import CapturedArticulatedTrials

            if self._surface_preconditioner is not None:
                if not self._surface_preconditioner.begin_trials(state_in, control, include_history=True):
                    self.coupling_graph_status = "unsupported surface metric recording"
                    return None
            if self._captured_trials is None:
                self._captured_trials = CapturedArticulatedTrials(self)
            try:
                self._captured_trials.rigid.prepare(dt)
            except ValueError as error:
                self.coupling_graph_status = str(error)
                return None
            return self._captured_trials

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

    def _restart_trials(self, state_in, tunnel_alm, *, keep_trial_load=False, cancel_guess=True):
        host_wrench = getattr(self, "_host_wrench_transaction", None)
        if host_wrench is not None:
            host_wrench.invalidate()
        if (affine := getattr(self.fem_solver, "_affine_predictor", None)) is not None:
            affine.reset()
        if cancel_guess and self._phase_load_predictor is not None:
            self._phase_load_predictor.cancel()
        self._restore_native()
        wp.copy(state_in.body_f, self._external_wrench)
        wp.copy(state_in.particle_f, self._input_particle_forces)
        wp.copy(self.fem_solver.velocity_increment, self._increment)
        if not keep_trial_load:
            # Otherwise the next trials start from the load a device search reached.
            wp.copy(self._trial_load, self._load)
        if self._shaft_preconditioner is not None:
            self._shaft_preconditioner.begin_step()
        self.interaction.discard_trial_forces()
        if tunnel_alm is not None:
            tunnel_alm.discard()
            tunnel_alm.begin_step()

    def _solve_device(self, graph, state_in, control, dt, tunnel_alm):
        """Return None for exact fallback, or the completed GPU solve result."""
        self._first_trial_event = self._event_load_kept = self._device_trial_stands = False
        self._budget_kept = False
        self._invalidate_query_cache()
        try:
            result = graph.solve(state_in, control, dt)
        finally:
            self._invalidate_query_cache()
            self.coupling_graph_captures = graph.captures
            self.coupling_graph_launches = graph.launches
        self.iterations, self.residual = int(result[0]), float(result[7])
        self.evaluations += self.iterations
        # The trial budget ended the device trials at their last trial. The loop reports it
        # with its converged flag set, so it takes the path of a converged trial below, with
        # the load its candidate state was computed with.
        budget_ended = len(result) > 12 and bool(result[12])
        if tunnel_alm is not None:
            tunnel_alm.dual_updates = int(result[5])
            tunnel_alm.total_dual_updates += int(result[5])
            tunnel_alm.normal_error, tunnel_alm.dual_residual = float(result[8]), float(result[9])
        if int(result[4]) == 3:
            # A trial threw the needle tip further than any accepted step moves it.
            self.coupling_unresolved_trials += 1
            self.coupling_graph_status = "device trial moved the tip beyond the step resolution"
            raise NeedleStepTooLarge(
                "Unresolved tip displacement in a device trial; reduce timestep to resolve needle radius "
                "and track spacing"
            )
        if int(result[4]) == 2:
            # The applied load ran away. The host loop would start from the same accepted
            # load; reject the step, so that the caller retries a shorter interval.
            self.coupling_divergence_stops += 1
            self.coupling_graph_status = "device trials diverged"
            raise NeedleCouplingNotConverged(
                f"Needle/tissue coupling did not converge: the trial load diverged after "
                f"{self.iterations} device trials (force residual {self.residual:.3g} N)"
            )
        # Request in the last trial; number of the first trial with a request (0: none).
        # Trial loops that stop at the first request report only the former.
        event = bool(result[10])
        first = int(result[11]) if len(result) > 11 else self.iterations * int(event)
        searched = first != 0
        if result[4] and not searched:
            if self._gpu_checkpoint is not None:
                self.rigid_solver.check_overflow()
            certificate = getattr(self.fem_solver, "_certified_linear", None)
            if certificate is not None and certificate.active:
                certificate.check(certificate.status.numpy()[0])
            if self._gpu_checkpoint is not None:
                self.rigid_solver.reject_unstable_trial()
            # A trial iteration that diverged has not converged: the step is rejected and the
            # caller retries a shorter interval from the accepted state.
            raise NeedleCouplingNotConverged("Nonfinite needle/tissue interface state or force")
        if event or (searched and not result[3]):
            # The host oracle decides the event. A search that converged past it hands over
            # its load, and so does a search the trial budget ended (the load of its best
            # trial); a failed search hands over nothing. No trial of the search is accepted.
            converged = bool(result[3]) and not result[4]
            self.coupling_graph_status = "exact surface-event fallback"
            self.coupling_graph_fallbacks += 1
            if budget_ended:
                self.coupling_budget_searches += 1
            else:
                self.coupling_graph_event_searches += int(converged)
            self._first_trial_event = first == 1
            self._event_load_kept = converged
            self._count_fallback_reasons(getattr(graph, "seen", graph.permissions.fallback))
            if converged and self.oracle_at_device_trial and self._gpu_checkpoint is not None:
                # The last trial of the search stands: its candidate state was computed with the
                # trial load, no correction followed it, and the rigid backend and the tunnel
                # multipliers are those of that trial. The oracle evaluates it as its first trial.
                self._device_trial_stands = True
                if self._phase_load_predictor is not None:
                    self._phase_load_predictor.cancel()
                self.interaction.discard_trial_forces()
            else:
                self._restart_trials(state_in, tunnel_alm, keep_trial_load=converged)
            return None
        self.coupling_graph_status = "replaying"
        if not result[3]:
            return False
        if budget_ended:
            # The last trial is kept. The endpoint certificates below and the checks of the
            # step can still refuse it, as they can a converged trial.
            self._budget_kept = True
            self.coupling_budget_keeps += 1

        if self._gpu_checkpoint is not None:
            # Retain the actual GPU-converged integration state. Never replay
            # native contact integration to manufacture a different endpoint.
            if self.interaction._stage_certified_trial(
                self._candidate, self.needle_body, self.needle_com, graph.permissions.mask
            ):
                return True
            self.coupling_graph_status = "exact GPU endpoint fallback"
            self.coupling_graph_fallbacks += 1
            self._count_fallback_reason("converged endpoint not certified")
            self._restart_trials(state_in, tunnel_alm)
            return None

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

    def _keep_host_trial(self, force_graph):
        """End a host loop under the trial budget with its last trial kept; False if it is not one to keep.

        A trial in which the interaction reported a failure is never kept: an eager trial leaves
        that status in the trial fields of the interaction, which are host values. A kept trial
        is staged as a converged one is.
        """
        fields = getattr(self.interaction, "_trial_fields", None)
        if fields is not None and fields.get("status") == "failed":
            return False
        self._budget_kept = True
        self.coupling_budget_keeps += 1
        if force_graph is not None:
            force_graph.stage()
        return True

    def _count_fallback_reason(self, reason):
        self.coupling_graph_fallback_reasons[reason] = self.coupling_graph_fallback_reasons.get(reason, 0) + 1

    def _count_fallback_reasons(self, flags):
        """Record the certificates behind one fallback; a rare event, read once."""
        from .interface.needle_trial_forces import FALLBACK_REASONS

        raised = flags.numpy()
        for index in np.flatnonzero(raised[1:]) + 1:
            self._count_fallback_reason(FALLBACK_REASONS[index])

    def _host_force_graph_supported(self, state_in, control, dt):
        """Whether the configuration admits captured host-trial forces.

        These are the refusals that do not look at the needle's accepted history. The
        gate on that history is in :meth:`_host_force_graph`.
        """
        if not self.capture_coupling or self._fem_capture_key(state_in, control, dt) is None:
            return False
        if not wp.is_conditional_graph_supported():
            return False
        from .interface.insertion import NeedleTissueInteraction

        if type(self.interaction) is not NeedleTissueInteraction or any(
            getattr(getattr(self.interaction, name), "__func__", None) is not getattr(NeedleTissueInteraction, name)
            for name in ("add_forces", "_add_forces", "_visited_sample_mask", "_tip_boundary_permission")
        ):
            return False
        if any(
            getattr(getattr(self, name), "__func__", None) is not getattr(NeedleTissueCoupler, name)
            for name in ("_step_fem", "_evaluate")
        ):
            return False
        if any(
            array.requires_grad for array in (state_in.joint_q, state_in.joint_qd, state_in.body_q, state_in.body_qd)
        ):
            return False
        return True

    def _host_force_graph(self, state_in, control, dt):
        """Use certified CUDA forces while retaining native articulated dynamics."""
        if not self._host_force_graph_supported(state_in, control, dt):
            return None
        if self.interaction._previous_tip is None:
            return None
        forces = self._host_trial_force_graphs()
        forces.prepare(dt)
        return forces

    def _host_trial_force_graphs(self):
        """The owner of the captured host-trial forces for the current certificate of the FEM solver."""
        from .host_trial_forces import CapturedHostTrialForces

        forces = self._host_trial_forces
        if forces is None or forces.certificate is not getattr(self.fem_solver, "_certified_linear", None):
            if forces is not None:
                note_release(len(forces.graphs))
            forces = self._host_trial_forces = CapturedHostTrialForces(self)
        return forces

    def _solve_host(
        self,
        state_in,
        control,
        dt,
        configured_tolerance,
        tunnel_alm,
        preconditioned,
        *,
        capture_forces=True,
        device_trial=False,
    ):
        """The host-controlled trial loop in one call (see _host_trials)."""
        trials = self._host_trials(
            None,
            state_in,
            control,
            dt,
            configured_tolerance,
            tunnel_alm,
            preconditioned,
            capture_forces=capture_forces,
            device_trial=device_trial,
        )
        try:
            next(trials)
        except StopIteration as finished:
            return finished.value
        raise AssertionError("Host-controlled trials without pieces do not pause")

    def _host_trials(
        self,
        pieces,
        state_in,
        control,
        dt,
        configured_tolerance,
        tunnel_alm,
        preconditioned,
        *,
        capture_forces=True,
        device_trial=False,
    ):
        """Original host-controlled trial loop; also the exact event fallback.

        A generator: with ``pieces`` it pauses before a trial that ``pieces`` does not admit, and goes on
        with that trial when it is resumed. Without ``pieces`` it never pauses.

        ``device_trial=True``: the candidate state is the last trial of a device search, computed
        with the trial load. The first trial evaluates it without repeating its tissue solve and
        rigid interval.
        """
        shadow_pending = False
        if self._gpu_checkpoint is not None:
            # Only the explicitly selected host policy or exact event fallback
            # needs a native shadow for its numerical search metric.
            self.rigid_solver.mj_model.opt.timestep = dt / self.rigid_solver.rigid_substeps
            if device_trial:
                # The backend holds the end of the device trial. The shadow is the accepted
                # state: it is read after the checkpoint is restored, before the first correction.
                shadow_pending = True
            else:
                self.rigid_solver.sync_native_shadow(forward=True)
        surface_pc = self._surface_preconditioner
        surface_available = surface_pc is not None and surface_pc.begin_trials(
            state_in, control, include_history=preconditioned
        )
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

        host_wrench = getattr(self, "_host_wrench_transaction", None)
        if host_wrench is not None:
            host_wrench.invalidate()
        relaxation, mixing_start = self.relaxation, 1
        acceleration = None
        if (
            getattr(self, "_surface_anderson_enabled", True)
            and supported_subcycled(self.rigid_solver)
            and (
                (preconditioned and self.preconditioned_anderson)
                or (not preconditioned and not getattr(self.interaction, "punctured", True))
            )
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
            from .acceleration.surface_anderson import SurfaceAnderson

            if self.model.device.is_cuda and getattr(self, "_device_surface_anderson_enabled", True):
                from .acceleration.surface_anderson_device import DeviceSurfaceAnderson

                acceleration = getattr(self, "_device_surface_anderson_workspace", None)
                if acceleration is None:
                    acceleration = DeviceSurfaceAnderson(self._trial_load)
                    self._device_surface_anderson_workspace = acceleration
                acceleration.reset()
            else:
                acceleration = SurfaceAnderson()
        device_aitken = _device_aitken_supported(self.model, self.fem_solver, state_in, preconditioned)
        relaxation_slot = 0
        converged = False
        # The primal stage of the first trial is the last device trial when one is handed over.
        reuse_primal = bool(device_trial)
        load_reference = 0.0
        self._diverged_after = 0
        self._budget_kept = False
        # Trial budget, the rule of the device loop: smallest force residual of the current
        # multiplier epoch (its load is in _best_load) and whether this trial re-applies it.
        trial_budget = getattr(self, "trial_budget", None)
        best_residual = None
        best_reapplied = False
        if trial_budget is not None and self._best_load is None:
            self._best_load = wp.empty_like(self._trial_load)
        for iteration in range(1, self.max_iterations + 1):
            if pieces is not None:
                yield from self._pause(pieces, "host")
            if configured_tolerance is not None and iteration > mixing_start and (iteration - mixing_start) % 8 == 0:
                tolerance = max(min(configured_tolerance, 1e-6), self.fem_solver.linear_solver_tolerance * 0.1)
                if tolerance < self.fem_solver.linear_solver_tolerance:
                    self.fem_solver.linear_solver_tolerance = tolerance
                    # Aitken differences must belong to the same approximate
                    # force map; tighter inner solves begin a fresh history.
                    relaxation, mixing_start = self.relaxation, iteration
                    if acceleration is not None:
                        acceleration.reset()
                    if host_wrench is not None:
                        host_wrench.invalidate()
            if not reuse_primal:
                self._restore_native()
                prepare_trial()
                if host_wrench is not None:
                    # No load writer is allowed between the preceding successful
                    # proposal and this preparation. All other paths invalidate.
                    host_wrench.repeated.host_token = host_wrench.offer(iteration)
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
                certificate = getattr(self.fem_solver, "_certified_linear", None)
                if certificate is not None and certificate.active:
                    packet = certificate.pack(packet)
                statistics = packet.numpy()
                if certificate is not None and certificate.active:
                    certificate.check(statistics[-1])
                    statistics = statistics[:-1]
            else:
                statistics = force_graph.evaluate(
                    dt, iteration > mixing_start and not preconditioned, fresh=not reuse_primal
                )
            # The pending token is already consumed. Start a new load generation
            # before any ALM reset, relaxation or subsequent proposal can write.
            if host_wrench is not None:
                host_wrench.invalidate()
            self.evaluations += 1
            self.iterations = iteration
            self.residual = float(statistics[0])
            if statistics[5]:
                if self._gpu_checkpoint is not None:
                    # An unstable rigid trial is rejected and retried, as the native guard does.
                    self.rigid_solver.reject_unstable_trial()
                raise NeedleCouplingNotConverged("Nonfinite needle/tissue interface force")
            tolerance = self.absolute_tolerance + self.relative_tolerance * max(
                float(statistics[1]), float(statistics[2])
            )
            if iteration == 1 and self.relative_tolerance > 0:
                # Applied load the trials start from. Below absolute / relative tolerance the
                # absolute tolerance governs, so that load is the smallest reference.
                load_reference = max(float(statistics[2]), self.absolute_tolerance / self.relative_tolerance)
            if self.residual <= tolerance:
                if tunnel_alm is None or tunnel_alm.converged(self.absolute_tolerance, self.relative_tolerance):
                    if force_graph is not None:
                        force_graph.stage()
                    converged = True
                    break
                if trial_budget is not None and iteration >= trial_budget:
                    # The trial budget is used up. This trial passes the force test and only
                    # waits for a multiplier update, so it is the best trial of its epoch, and
                    # it is the last one evaluated: it is kept as it is, without the update.
                    if self._keep_host_trial(force_graph):
                        converged = True
                        break
                    # The interaction reported a failure in it: the loop goes on without a budget.
                    trial_budget = None
                # A dual update changes the proximal force map. Re-solve
                # from the same physical start with a fresh Aitken history;
                # every primal evaluation still consumes the shared budget.
                tunnel_alm.update_duals()
                # The best trial restarts with the changed force map.
                best_residual = None
                relaxation, mixing_start = self.relaxation, iteration + 1
                if acceleration is not None:
                    acceleration.reset()
                reuse_primal = force_graph is not None
                continue
            if load_reference and self.relative_tolerance * float(statistics[2]) > load_reference:
                # The applied load has grown by 1 / relative tolerance over the load the trials
                # started from: the tolerance of this trial alone exceeds that whole load, so
                # the iteration is running away. The loop ends as not converged and the step
                # is rejected as after an exhausted trial budget.
                self.coupling_divergence_stops += 1
                self._diverged_after = iteration
                return False
            if trial_budget is not None:
                # Smallest force residual of the multiplier epoch; of equal trials, the later one.
                best = best_residual is None or self.residual <= best_residual
                if best_reapplied or (best and iteration >= trial_budget):
                    # The trial budget is used up and the loop has neither failed nor diverged.
                    # The last trial is the best one, or it applied the best load once more:
                    # it is kept, through the path of a converged trial, whatever its residual.
                    if self._keep_host_trial(force_graph):
                        converged = True
                        break
                    # The interaction reported a failure in it: the loop goes on without a budget.
                    trial_budget = None
                elif iteration >= trial_budget:
                    # An earlier trial is the best one. Its load is applied once more, with no
                    # correction after this trial, and that trial is kept.
                    wp.copy(self._trial_load, self._best_load)
                    best_reapplied = True
                    reuse_primal = False
                    continue
                elif best:
                    best_residual = self.residual
                    wp.copy(self._best_load, self._trial_load)
            if shadow_pending:
                # The first correction of a loop that started at a device trial. The correction
                # reads the candidate state, not the backend, so the backend can go back to the
                # accepted state here; the next trial restores it again.
                self._restore_native()
                self.rigid_solver.sync_native_shadow(forward=True)
                shadow_pending = False
            # Vector Aitken relaxation adapts to the combined structural,
            # rigid and regularized-friction response without changing any
            # physical force. Positive relaxation handles contact switches.
            reuse_primal = False
            direction = self._residual
            selected_surface = bool(surface_available and statistics[7] == 1 and statistics[6] > 0)
            if selected_surface and not preconditioned:
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
                    from .acceleration.surface_preconditioner import clear_direction_statistics

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
            if preconditioned and acceleration is not None:
                # Preconditioned multisecant proposal. Only the
                # force guess changes; every candidate must pass the original
                # physical and ALM residual checks above. Secant history is
                # reset by the existing tolerance/dual/map-switch branches.
                if hasattr(acceleration, "propose_device"):
                    acceleration.bind_host_wrench_scope(host_wrench)
                    applied = acceleration.propose_device(
                        self._trial_load, self._residual, self.relaxation, direction=direction
                    )
                    if host_wrench is not None and applied:
                        host_wrench.publish(acceleration.take_host_load_witness(), iteration + 1)
                    proposal = None
                else:
                    proposal = acceleration.propose(
                        self._trial_load.numpy(), self._residual.numpy(), self.relaxation, direction=direction.numpy()
                    )
                    applied = proposal is not None
                if not applied:
                    wp.launch(
                        _relax_interface,
                        self._load.size,
                        [direction, self.relaxation, self._trial_load, self._previous_residual],
                        device=self.model.device,
                    )
                elif proposal is not None:
                    self._trial_load.assign(proposal)
                continue
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
                proposals, rejections = acceleration.proposals, acceleration.rejections
                if hasattr(acceleration, "propose_device"):
                    acceleration.bind_host_wrench_scope(host_wrench)
                    applied = acceleration.propose_device(
                        self._trial_load,
                        self._residual,
                        relaxation,
                    )
                    if host_wrench is not None and applied:
                        host_wrench.publish(acceleration.take_host_load_witness(), iteration + 1)
                    proposal = None
                    residual = None
                else:
                    residual = self._residual.numpy()
                    proposal = acceleration.propose(self._trial_load.numpy(), residual, relaxation)
                    applied = proposal is not None
                self._surface_anderson_proposals += acceleration.proposals - proposals
                self._surface_anderson_rejections += acceleration.rejections - rejections
                if applied:
                    if proposal is not None:
                        self._trial_load.assign(proposal)
                        self._previous_residual.assign(residual)
                    else:
                        wp.copy(self._previous_residual, self._residual)
                    relaxation, mixing_start = self.relaxation, iteration + 1
                    continue
            wp.launch(
                _relax_interface,
                self._load.size,
                [direction, relaxation, self._trial_load, self._previous_residual],
                device=self.model.device,
            )
        return converged

    def _open_attempt(self, state_in, control, dt, external_forces):
        """Save what a rejected attempt restores and return the record of the attempt.

        Nothing here is undone by :meth:`_close_attempt` if it raises: the caller runs
        this before the ``try`` whose ``finally`` closes the attempt.
        """
        # Host preparation of the step, outside every capture: all graphs and eager
        # launches of this step read the time step from these arrays.
        self._share_step_time()
        self.step_time.set(dt)
        wp.copy(self._external_wrench, state_in.body_f)
        if external_forces is None:
            self._external_nodes.zero_()
        else:
            wp.copy(self._external_nodes, external_forces)
        if self._gpu_checkpoint is not None:
            self._gpu_checkpoint.save()
        else:
            self._mujoco.mj_getState(
                self.rigid_solver.mj_model, self.rigid_solver.mj_data, self._native, self._state_spec
            )
            self._native_warnings = snapshot_native_warnings(self.rigid_solver.mj_data)
        if self._increment is not None:
            wp.copy(self._increment, self.fem_solver.velocity_increment)
        wp.copy(self._input_particle_forces, state_in.particle_f)
        wp.copy(self._trial_load, self._load)
        configured_tolerance = (
            self.fem_solver.linear_solver_tolerance if isinstance(self.fem_solver, SolverFEM) else None
        )
        tunnel_alm = getattr(self.interaction, "tunnel_alm", None)
        self._budget_kept = False
        predictor = self._phase_load_predictor
        self._host_wrench_transaction = None
        affine = getattr(self.fem_solver, "_affine_predictor", None)
        held = getattr(self.fem_solver, "_held_start", None)
        certificate = getattr(self.fem_solver, "_certified_linear", None)
        return _Attempt(state_in, control, dt, configured_tolerance, tunnel_alm, predictor, affine, held, certificate)

    def _begin_attempt(self, attempt, *, startup=False):
        """Begin the trials of an opened attempt: certificate, start guesses, native input, multipliers.

        The caller runs this inside the ``try`` whose ``finally`` calls :meth:`_close_attempt`,
        so an error raised here is closed like a rejected step. ``startup`` is set by
        :meth:`capture_graphs`: the response-basis predictor then makes its storage and
        its column engine before it begins, and the held start captures the preparation
        of this buffer before it prepares, so that neither does it in an attempt of the run.
        """
        state_in, control, dt = attempt.state_in, attempt.control, attempt.dt
        certificate, affine, held = attempt.certificate, attempt.affine, attempt.held
        predictor, tunnel_alm = attempt.predictor, attempt.tunnel_alm
        if certificate is not None:
            certificate.begin_attempt(self, state_in, control)
        if affine is not None:
            if startup:
                # Its key in the FEM graphs and in the trial graph is then the one of the run.
                affine.prepare(self, state_in, control)
            affine.begin_attempt(self, state_in, control)
        if held is not None:
            if startup:
                held.warm(state_in, control, None, dt)
            held.prepare(state_in, control, None, dt)
        if (
            self._rigid_bridge is None
            and type(self.rigid_solver) is SubcycledMuJoCo
            and getattr(self.rigid_solver.step, "__func__", None) is SubcycledMuJoCo.step
        ):
            # Stored at once: the close ends the repeated input also when the next call raises.
            attempt.native_input = native_input = self.rigid_solver.begin_repeated_input(
                state_in, self._candidate, control, self.needle_body
            )
            if native_input is not None and getattr(self, "_host_wrench_enabled", False):
                self._host_wrench_transaction = native_input._repeated_input.enable_host_wrench(
                    self._external_wrench, self._trial_load, self.moment_length
                )
        guesses = 0 if predictor is None else predictor.attempts
        if predictor is not None:
            from .phase_predictor import coupler_signature

            predictor.begin(
                self._load,
                self._trial_load,
                dt,
                coupler_signature(self, increments=predictor.increments),
                # The engagement has the rows of the track reserve; the live ones are compared.
                self.interaction._track_engaged,
                self.interaction.track_count,
            )
        attempt.guessed = predictor is not None and predictor.attempts != guesses
        if tunnel_alm is not None:
            tunnel_alm.begin_step()
        if self._surface_preconditioner is not None:
            self._surface_preconditioner.begin_step()
        if self._shaft_preconditioner is not None:
            # Its ``active`` flag is not read here: device trials that do not follow it never
            # evaluate it, and host-controlled trials evaluate it where they first run.
            self._shaft_preconditioner.begin_step()

    def _close_attempt(self, attempt, state_in, committed):
        """End an attempt; one that was not committed is restored to the state :meth:`_open_attempt` saved."""
        certificate, affine, held = attempt.certificate, attempt.affine, attempt.held
        predictor, tunnel_alm = attempt.predictor, attempt.tunnel_alm
        native_input, configured_tolerance = attempt.native_input, attempt.configured_tolerance
        if certificate is not None:
            certificate.end_attempt()
        device_anderson = getattr(self, "_device_surface_anderson_workspace", None)
        if device_anderson is not None:
            device_anderson.reset()
        if held is not None:
            held.invalidate()
        if affine is not None:
            affine.end_attempt()
        if self._surface_preconditioner is not None:
            self.interaction._surface_tangent_anchors = None
            self._surface_preconditioner.available = False
        if native_input is not None:
            native_input.end_repeated_input()
        self._host_wrench_transaction = None
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

    # Host values that an attempt writes and the path of a rejected step leaves as written:
    # counters of work, and what the last trial loop reported. A start-up attempt
    # (capture_graphs) gives them back.
    _STARTUP_KEPT = (
        "iterations",
        "evaluations",
        "trial_loops",
        "loop_seconds",
        "residual",
        "maximum_iterations",
        "maximum_residual",
        "coupling_graph_launches",
        "coupling_graph_fallbacks",
        "coupling_graph_fallback_reasons",
        "coupling_graph_event_searches",
        "coupling_graph_event_restarts",
        "coupling_divergence_stops",
        "_diverged_after",
        "coupling_unresolved_trials",
        "coupling_budget_keeps",
        "coupling_budget_searches",
        "coupling_budget_residual",
        "coupling_budget_events",
        "_budget_kept",
        "_first_trial_event",
        "_event_load_kept",
        "fem_graph_launches",
    )

    def _startup_kept(self):
        """Read what :meth:`capture_graphs` gives back: (owner, name, value) of each host value, and device copies."""
        held = getattr(self.fem_solver, "_held_start", None)
        owners = (
            (self, self._STARTUP_KEPT),
            (getattr(self.fem_solver, "_certified_linear", None), ("attempts",)),
            (held, ("preparations", "graph_launches")),
            (
                getattr(self.interaction, "tunnel_alm", None),
                ("dual_updates", "total_dual_updates", "normal_error", "dual_residual"),
            ),
            (self.rigid_solver, ("gpu_graph_launches",)),
            # Set by a reader of the shaft preconditioner's ``active`` flag for the accepted step
            # that follows; no step follows a start-up attempt.
            (self.interaction, ("_engaged_length_requested",)),
        )
        host = []
        for owner, names in owners:
            for name in names:
                if owner is not None and hasattr(owner, name):
                    value = getattr(owner, name)
                    host.append((owner, name, dict(value) if isinstance(value, dict) else value))
        # The held start counts its uses on the device, and every rigid interval writes its
        # microstep to the device copy of the time step (MuJoCo Warp's opt.timestep), whose host
        # copy capture_graph gives back. Copied device to device; nothing is read.
        uses = getattr(held, "uses", None)
        timestep = getattr(getattr(getattr(self.rigid_solver, "mjw_model", None), "opt", None), "timestep", None)
        device = tuple((array, wp.clone(array)) for array in (uses, timestep) if isinstance(array, wp.array))
        # The owner of the trial graph and the tetrahedron query of the interaction may not
        # exist yet; the first start-up attempt then makes them. Their values before it:
        # no launch, and no accepted positions in the query.
        trials = self._captured_trials
        queries = getattr(self.interaction, "_device_queries", None)
        late = (0 if trials is None else trials.launches, False if queries is None else queries.has_previous)
        return host, device, self.step_time.value, late

    def _startup_restore(self, kept):
        """Give back what :meth:`_startup_kept` read, once, after the last start-up attempt."""
        host, device, interval, (launches, has_previous) = kept
        for owner, name, value in host:
            setattr(owner, name, value)
        for array, values in device:
            wp.copy(array, values)
        if interval is not None:
            self.step_time.set(interval)
        if self._captured_trials is not None:
            self._captured_trials.launches = launches
        queries = getattr(self.interaction, "_device_queries", None)
        if queries is not None and getattr(self.interaction, "_previous_tip", None) is None:
            # Set by the first preparation of the trial forces and kept while every buffer is
            # captured: a second commit would report changed storage and drop the graph of
            # the buffer before. Without an accepted tip the first step of a run is
            # host-controlled and reads this flag before its acceptance commits the query, so
            # it finds the flag it would have found. With an accepted tip the flag stays: the
            # committed positions are the accepted ones, which the first device step would
            # commit itself, and a second commit there would drop the graphs captured here.
            queries.has_previous = has_previous

    def warm_fem(self, state_in, control, dt):
        """Capture and launch the FEM step of the host-controlled trials at every inner tolerance.

        For start-up, inside an attempt on ``state_in`` (after its begin phase, so that
        the certificate scope, the response-basis key and the held start are those of a
        trial) and outside capture. A host-controlled interval tightens the inner
        tolerance through ``fem_solver.tolerance_levels()``; for each level the solver's
        registered Krylov workspace is fetched and :meth:`_step_fem` is issued once,
        which captures the graph of (level, ``state_in``) and launches it. A solver
        whose workspace is not selected yet (after its first projection) steps once as
        written first: that step allocates the remaining scratch.

        The steps start from ``state_in`` with no interface load and write the candidate
        state, solver scratch, ``state_in.particle_f`` and the velocity increment; the last
        two and the configured tolerance are restored here, and the history of the
        response-basis predictor and the status of the certified solve are cleared, as
        after the discarded step of the trial-graph preparation. ``fem_graph_launches``
        counts the launches; :meth:`capture_graphs` gives that counter back.

        Returns:
            The number of graphs captured by this call.
        """
        solver = self.fem_solver
        if self.model.device.is_capturing:
            raise RuntimeError("Capture the FEM steps of the host-controlled trials outside CUDA graph capture")
        if self._fem_capture_key(state_in, control, dt) is None:
            return 0
        configured, captures = solver.linear_solver_tolerance, self.fem_graph_captures
        increment, forces = wp.clone(solver.velocity_increment), wp.clone(state_in.particle_f)
        try:
            self._candidate.assign(state_in)
            self._forces.zero_()
            self._refresh_projection()
            if solver._lin_solver is None:
                solver.step(state_in, self._candidate, control, None, dt, external_forces=self._forces)
            for tolerance in solver.tolerance_levels(configured):
                solver.linear_solver_tolerance = tolerance
                # Registered and pinned: the trial graph binds the same workspace at this level,
                # and a host-controlled trial at this tolerance selects it. Selected here, so
                # that the step below looks its graph up under this workspace.
                solver._lin_solver_key, solver._lin_solver = solver.linear_workspace(tolerance)
                self._step_fem(state_in, control, dt)
                if self.fem_graph_status != "replaying":
                    raise RuntimeError(
                        f"The FEM step at inner tolerance {tolerance:g} was not captured: {self.fem_graph_status}"
                    )
        finally:
            solver.linear_solver_tolerance = configured
            wp.copy(solver.velocity_increment, increment)
            wp.copy(state_in.particle_f, forces)
            if (affine := getattr(solver, "_affine_predictor", None)) is not None:
                affine.reset()
            if (certificate := getattr(solver, "_certified_linear", None)) is not None and certificate.active:
                certificate.status.zero_()
        return self.fem_graph_captures - captures

    def capture_trial_graph(self, attempt):
        """Capture the trial graph of an opened and begun attempt and launch it once.

        For start-up. The accepted history is not asked: the interaction has no accepted
        tip yet, so the one trial of the launch requests the host path through its step
        flag and the loop ends there. The result is not interpreted. The attempt is then
        restarted as after a device search that handed over to the host: integration
        state, incoming forces, velocity increment, trial load, multipliers and the
        preconditioner are those the attempt began with.

        Returns:
            False if the configuration has no device trials (``coupling_graph_status``
            names the refusal); nothing is issued then.
        """
        state_in, control, dt = attempt.state_in, attempt.control, attempt.dt
        if not self._device_trials_supported(state_in, control, dt):
            return False
        trials = self._bound_device_trials(state_in, control, dt)
        if trials is None:
            return False
        self._invalidate_query_cache()
        try:
            trials.solve(state_in, control, dt)
        finally:
            self._invalidate_query_cache()
            self.coupling_graph_captures = trials.captures
        # An attempt of start-up takes no load guess (its predictor is None): the restart has none
        # to cancel, and a phase load predictor that exists already keeps what it holds.
        self._restart_trials(state_in, attempt.tunnel_alm, cancel_guess=attempt.predictor is not None)
        if attempt.configured_tolerance is not None:
            self.fem_solver.linear_solver_tolerance = attempt.configured_tolerance
        return True

    def _prepare_native_trials(self):
        """Load and allocate, for start-up, what only eager code of a native free-needle step uses.

        The kernel modules of the compact rigid interval (pack and unpack around the native
        step), of the step guard and of this module's own eager launches, by name and with no
        launch, and the buffers of the endpoint comparison that follows a converged device
        loop, which that loop otherwise allocates at its first use.
        """
        from ..mujoco import mujoco_free_body
        from . import tet_step

        device = self.model.device
        modules = [__name__, mujoco_free_body]
        if self._tet_step_guard is not None:
            modules.append(tet_step)
        for module in modules:
            wp.load_module(module, device=device, block_dim=256)
        if self._endpoint_snapshot is None:
            self._endpoint_arrays = tuple(
                getattr(self._candidate, name).view(wp.uint32).flatten()
                for name in ("joint_q", "joint_qd", "body_q", "body_qd")
            )
            self._endpoint_snapshot = wp.empty(
                sum(array.size for array in self._endpoint_arrays), dtype=wp.uint32, device=device
            )
            self._endpoint_equal = wp.ones(1, dtype=int, device=device)

    def capture_graphs(self, states, control, dt, *, native=False):
        """Capture and launch, ahead of the first step, graphs that host-controlled trials would capture in the run.

        For start-up, at the end of a reset. ``states`` are the buffers that will hold accepted
        states (a scene alternates two), each holding a valid accepted state now; ``control``
        and ``dt`` are those of the steps that follow. For each buffer one attempt is opened
        and begun as :meth:`step` does it, with no external nodal forces and the buffer's own
        ``body_f``, and inside it:

        - the begin phase makes the storage of the response-basis predictor and its column
          engine (the first buffer) and captures and launches the preparation of the held
          start of the buffer (``_begin_attempt(startup=True)``);
        - the trial preparation of the buffer runs once as written (the first buffer) and is
          captured and launched (``CapturedTrialPreparation.capture_graph``);
        - the rigid interval of the buffer runs as written (the first buffer), is captured and
          launched (``capture_graph`` of the rigid solver), each from the integration state
          the attempt saved;
        - the FEM step of the host-controlled trials is captured and launched at every inner
          tolerance (:meth:`warm_fem`);
        - the trial graph of the device trials is captured and launched
          (:meth:`capture_trial_graph`); its launch runs one trial;
        - in the attempt of the first buffer: the device helper of the host sample mask is
          built, and the force graphs of the host-controlled trials are captured and
          launched, one for each variant (:meth:`capture_host_force_graphs`);
        - in the attempt of the last buffer: the graphs of the host-controlled correction are
          captured and launched and the device workspace of the host Anderson mixing is made
          (:meth:`capture_contact_graphs`), and the outputs of the certificate packet of a
          host trial are allocated.

        With a trial budget, the storage of the best load of a host-controlled loop is
        allocated before the first buffer.

        ``native=True`` asks for the same on the native free-needle backend (CPU MuJoCo behind
        the compact bridge), whose rigid solver has no entry of its own. The rigid interval
        is then not issued: it is the native step, which captures nothing, and the attempts
        leave the native integration state as they find it. The trial graph is captured and
        launched with an empty track, a state that the device trials of a run never start
        from (they wait for the first track sample); it holds the correction of both values
        of the shaft flag (``_CapturedNeedleTrials.branches_on_shaft_flag``). The kernel
        modules and buffers of the eager code of such a step are loaded and allocated first
        (:meth:`_prepare_native_trials`). A caller that does not ask is declined as before.

        The attempt then ends as a rejected step ends: nothing is committed, the integration
        state, the incoming forces, the FEM velocity increment, the multipliers and the trial
        context return to what the attempt saved. No state is advanced and no history is
        accepted. The phase load predictor takes no part. After the last buffer the host
        counters of work, the uses of the held start, the device copy of the rigid time step
        (MuJoCo Warp's ``opt.timestep``, which every issued interval writes) and the step time
        get back the values they had before the call; capture counters keep the captures. The
        tetrahedron query of the interaction, which the force helpers build and commit, is
        marked as not committed again if it was not before and the interaction has no accepted
        tip: the first step then starts as it does without this call.

        The solve of the host-controlled correction that starts from a response basis is
        captured here where that basis has its storage, which the begin phase makes for a
        configured predictor. Without that storage it still follows 'first use runs as
        written, the next use captures'.

        Returns:
            False, with nothing issued, on a device without conditional CUDA graphs, without
            ``capture_fem``, for a configuration whose FEM step is not captured
            (``fem_graph_status`` names it) and for a rigid solver without an entry of its own
            (the native backends, which capture where the run first needs it) unless ``native``
            asks for the free-needle backend.
        """
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("Coupled timestep must be finite and positive")
        states = tuple(states)
        device = self.model.device
        rigid_entry = getattr(self.rigid_solver, "capture_graph", None)
        if (
            not states
            or not device.is_cuda
            or not wp.is_conditional_graph_supported()
            or not self.capture_fem
            or (rigid_entry is None and not (native and self._rigid_bridge is not None))
            or any(self._fem_capture_key(state, control, dt) is None for state in states)
        ):
            return False
        if device.is_capturing:
            raise RuntimeError("Capture the graphs of the coupled step outside CUDA graph capture")
        from .host_trial_prepare import CapturedTrialPreparation

        # With the flag of the tetrahedron query: the force helpers commit the query when they are
        # built. It stays committed over all buffers (a second commit would count as changed
        # storage) and is given back after the last one.
        kept = self._startup_kept()
        if getattr(self, "trial_budget", None) is not None and self._best_load is None:
            # Load of the best trial of a budgeted host loop: allocated here, not inside the
            # first such loop of a run. It is written before it is read.
            self._best_load = wp.empty_like(self._trial_load)
        if rigid_entry is None:
            self._prepare_native_trials()
        try:
            for state in states:
                # Before the try, as in step(): an error of the open phase is not closed.
                attempt = self._open_attempt(state, control, dt, None)
                # No load guess: the predictor is in no graph key and is not begun or discarded here.
                attempt.predictor = None
                try:
                    self._begin_attempt(attempt, startup=True)
                    if self._host_trial_preparation is None:
                        self._host_trial_preparation = CapturedTrialPreparation(self)
                    self._host_trial_preparation.capture_graph(state)
                    if rigid_entry is not None:
                        rigid_entry(state, self._candidate, control, dt, before_interval=self._restore_native)
                        self._restore_native()
                    # The FEM step of the host-controlled trials, one graph per inner tolerance.
                    self.warm_fem(state, control, dt)
                    # The trial graph of the device trials, with the workspaces registered above.
                    self.capture_trial_graph(attempt)
                    if state is states[0]:
                        self.capture_host_force_graphs(state, control, dt)
                    if state is states[-1]:
                        self.capture_contact_graphs(dt, native=rigid_entry is None)
                        if attempt.certificate is not None:
                            # Outputs of the certificate packet of a host trial, without and with the
                            # surface metric: allocated here, not by the first host trial of a run.
                            attempt.certificate.reserve(self._statistics.size)
                            if self._surface_preconditioner is not None:
                                attempt.certificate.reserve(self._surface_preconditioner.result.size)
                finally:
                    self._close_attempt(attempt, state, False)
        finally:
            self._startup_restore(kept)
        return True

    def capture_host_force_graphs(self, state_in, control, dt):
        """Build the helper of the host sample mask and capture the force graphs of the host-controlled trials.

        A start-up entry of :meth:`capture_graphs`, inside an attempt on ``state_in`` that ends
        like a rejected step, with a candidate that holds a finite state. The helper that the
        host sample mask uses from the first track sample on is built
        (``prepare_host_sample_helper`` of the interaction). Where the configuration admits
        captured host-trial forces, their owner is made and captures and launches one graph
        for each variant (``CapturedHostTrialForces.capture_graphs``). No launch is read back
        and no work counter changes.

        Returns:
            Whether the force graphs were captured.
        """
        prepare = getattr(self.interaction, "prepare_host_sample_helper", None)
        if prepare is not None:
            prepare()
        if not self._host_force_graph_supported(state_in, control, dt):
            return False
        self._host_trial_force_graphs().capture_graphs(state_in, control, dt)
        return True

    def capture_contact_graphs(self, dt, *, native=False):
        """Capture the graphs of the host-controlled correction and make the workspace of its Anderson mixing.

        A start-up entry of :meth:`capture_graphs` for the MuJoCo Warp rigid backend on CUDA
        with conditional graphs, inside an attempt that ends like a rejected step. In the
        order of a host-controlled trial loop:

        - the device workspace of the surface Anderson mixing is made, with its packet, and
          its kernel module is loaded;
        - the native shadow is brought to the integration state of the attempt, as the trial
          loop does it, for the condensed native metric of the two corrections;
        - with a zero residual, the shaft correction captures and launches its contact
          preparation and its solve (``capture_host_graphs``) and the surface correction its
          solve (``capture_host_graph``), for the candidate, the residual and the direction
          that the trial loop passes to them.

        The passes write scratch of the corrections and the direction only, and each owner
        gives back the work it counted. The time step of the native model and the token of
        the shadow get back their values, and the shadow is brought to the integration state
        again under that time step.

        ``native=True`` asks for the native free-needle backend (CPU MuJoCo behind the compact
        bridge): its host-controlled trials mix without the Anderson workspace and its
        correction reads no native shadow, so only the shaft correction captures and launches
        its contact preparation and its solve, with a zero residual and the same arrays.

        Returns:
            False, with nothing issued, on another rigid backend or device.
        """
        device = self.model.device
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            return False
        if self._gpu_checkpoint is None:
            if not native or self._rigid_bridge is None:
                return False
            if device.is_capturing:
                raise RuntimeError("Capture the graphs of the correction outside CUDA graph capture")
            if self._shaft_preconditioner is not None:
                self._residual.zero_()
                self._shaft_preconditioner.capture_host_graphs(self._candidate, dt, self._residual, self._direction)
            return True
        if device.is_capturing:
            raise RuntimeError("Capture the graphs of the correction outside CUDA graph capture")
        if (
            getattr(self, "_surface_anderson_enabled", True)
            and getattr(self, "_device_surface_anderson_enabled", True)
            and supported_subcycled(self.rigid_solver)
        ):
            from .acceleration import surface_anderson_device

            if getattr(self, "_device_surface_anderson_workspace", None) is None:
                self._device_surface_anderson_workspace = surface_anderson_device.DeviceSurfaceAnderson(
                    self._trial_load
                )
            wp.load_module(surface_anderson_device, device=device, block_dim=256)
        pc, surface_pc = self._shaft_preconditioner, self._surface_preconditioner
        if pc is None and surface_pc is None:
            return True
        rigid = self.rigid_solver
        timestep, token = rigid.mj_model.opt.timestep, rigid._shadow_token
        try:
            rigid.mj_model.opt.timestep = dt / rigid.rigid_substeps
            rigid.sync_native_shadow(forward=True)
            self._residual.zero_()
            if pc is not None:
                pc.capture_host_graphs(self._candidate, dt, self._residual, self._direction)
            if surface_pc is not None:
                surface_pc.capture_host_graph(self._candidate, dt, self._residual, self._direction)
        finally:
            rigid.mj_model.opt.timestep = timestep
            rigid.sync_native_shadow(forward=True)
            rigid._shadow_token = token
        return True

    def _timed_loop(self, solve, *arguments, **options):
        """Run one trial loop of a step; count it and add its host-clock time to ``loop_seconds``."""
        started = time.perf_counter()
        try:
            return solve(*arguments, **options)
        finally:
            self.trial_loops += 1
            self.loop_seconds += time.perf_counter() - started

    # An attempt that begin_step()/resume_step() left open (a generator of _attempt), or None.
    _suspended = None
    _abandoning = False
    # Host-clock time [s] that attempts spent left open between two calls.
    _paused_seconds = 0.0

    def step(self, state_in, state_out, control, dt, *, external_forces=None):
        """Advance one shared step, raising if the endpoint force solve fails."""
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("Coupled timestep must be finite and positive")
        if self._suspended is not None:
            raise AssertionError("An attempt is open: go on with it (resume_step) or abandon it (abandon_step)")
        for _ in self._attempt(state_in, state_out, control, dt, external_forces, None):
            raise AssertionError("An attempt without pieces does not pause")

    @property
    def attempt_open(self):
        """Whether begin_step() or resume_step() returned with the attempt unfinished."""
        return self._suspended is not None

    def begin_step(self, state_in, state_out, control, dt, *, external_forces=None, pieces):
        """Start one attempt of a shared step and run its pieces while ``pieces`` admits them.

        Returns True when the attempt has ended accepted (a rejected attempt raises, as in step()), False when it
        is left open: resume_step() goes on with it, abandon_step() ends it as a rejected attempt.
        """
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("Coupled timestep must be finite and positive")
        if self._suspended is not None:
            raise AssertionError("An attempt is open: go on with it (resume_step) or abandon it (abandon_step)")
        return self._advance(self._attempt(state_in, state_out, control, dt, external_forces, pieces))

    def resume_step(self):
        """Go on with the attempt that was left open; returns as begin_step() does."""
        attempt, self._suspended = self._suspended, None
        if attempt is None:
            raise AssertionError("No attempt is open")
        return self._advance(attempt)

    def abandon_step(self):
        """End an attempt that was left open as a rejected attempt ends; nothing if none is open."""
        attempt, self._suspended = self._suspended, None
        if attempt is not None:
            self._abandoning = True
            try:
                attempt.close()
            finally:
                self._abandoning = False

    def _advance(self, attempt):
        try:
            next(attempt)
        except StopIteration:
            return True
        self._suspended = attempt
        return False

    def _pause(self, pieces, kind):
        """Before a piece of an open attempt: return to the caller once if ``pieces`` does not admit it."""
        if pieces.admit(kind):
            return
        left = time.perf_counter()
        try:
            yield kind
        finally:
            self._paused_seconds += time.perf_counter() - left
        if not pieces.admit(kind):
            raise AssertionError("A budget must admit the first piece of a call that goes on with an attempt")

    def _host_loop(self, pieces, *arguments, **options):
        """One host-controlled loop that may be left between two trials; counted as _timed_loop counts it."""
        started, paused = time.perf_counter(), self._paused_seconds
        try:
            return (yield from self._host_trials(pieces, *arguments, **options))
        finally:
            self.trial_loops += 1
            self.loop_seconds += time.perf_counter() - started - (self._paused_seconds - paused)

    def _attempt(self, state_in, state_out, control, dt, external_forces, pieces):
        """One attempt of a shared step as a generator: the text of step(), which pauses only with ``pieces``."""
        attempt = self._open_attempt(state_in, control, dt, external_forces)
        configured_tolerance, tunnel_alm = attempt.configured_tolerance, attempt.tunnel_alm
        predictor = attempt.predictor
        converged = False
        committed = False
        collected = False
        try:
            self._begin_attempt(attempt)
            graph = self._device_trials(state_in, control, dt)
            converged = None
            if graph is not None:
                converged = self._timed_loop(self._solve_device, graph, state_in, control, dt, tunnel_alm)
            if converged is None:
                from .host_trial_forces import TrialForceFallback

                def host(captured_forces, device_trial=False):
                    pc = self._shaft_preconditioner
                    if attempt.preconditioned is None:
                        # Once per step, where its host-controlled trials first run. The accepted
                        # history that the flag follows is the same for every trial of the step.
                        # Where the trials of every step evaluate the flag, the evaluation asks
                        # the acceptance of this step for the engaged length of the next one.
                        # Device trials that do not evaluate it make these host trials an
                        # exception that the next step does not repeat: the length is read
                        # here, once, and the acceptance of this step reads nothing.
                        trials = self._captured_trials
                        every_step = trials is None or trials.evaluates_shaft_flag
                        attempt.preconditioned = pc is not None and pc.read_active(request=every_step)
                    elif pc is not None:
                        # A restart of the trials began the step of the correction again. The
                        # accepted history is the same, so the value of the step holds: the
                        # corrections of the trials below do not evaluate it a second time.
                        pc.active = attempt.preconditioned
                    # The event oracle is the same loop without captured force evaluation.
                    options = {} if captured_forces else {"capture_forces": False}
                    if device_trial:
                        options["device_trial"] = True
                    own = getattr(self._solve_host, "__func__", None) is NeedleTissueCoupler._solve_host
                    if pieces is None or not own:
                        return self._timed_loop(
                            self._solve_host,
                            state_in,
                            control,
                            dt,
                            configured_tolerance,
                            tunnel_alm,
                            attempt.preconditioned,
                            **options,
                        )
                    return (
                        yield from self._host_loop(
                            pieces,
                            state_in,
                            control,
                            dt,
                            configured_tolerance,
                            tunnel_alm,
                            attempt.preconditioned,
                            **options,
                        )
                    )

                def restart():
                    # Device force evaluation never commits history. Restore
                    # native integration, numerical guesses and ALM trial state
                    # before the original swept-event/authorization oracle.
                    self._restart_trials(state_in, tunnel_alm)
                    self._invalidate_query_cache()
                    self.fem_solver.linear_solver_tolerance = configured_tolerance

                converged = False
                if self._event_load_kept:
                    # A device search converged past a surface event; the oracle decides the
                    # event from that load. The load is a starting point and nothing more: if
                    # the oracle cannot use it, the step starts again from the accepted load.
                    stands = self._device_trial_stands
                    self._device_trial_stands = False
                    self.coupling_oracle_device_trials += int(stands)
                    try:
                        converged = yield from host(False, stands)
                    except NeedleStepTooLarge:
                        converged = False
                    if not converged:
                        self.coupling_graph_event_restarts += 1
                        restart()
                    elif stands and self.iterations == 1:
                        self.coupling_oracle_device_accepts += 1
                if not converged:
                    # Captured host forces evaluate the certificates of the device trials. A
                    # device attempt that failed them in its first trial, started from the
                    # accepted load, is that same evaluation; go to the oracle directly.
                    repeated = graph is not None and self._first_trial_event and not attempt.guessed
                    try:
                        converged = yield from host(not repeated)
                    except TrialForceFallback:
                        restart()
                        converged = yield from host(False)
            if not converged:
                detail = f"force residual {self.residual:.3g} N"
                if tunnel_alm is not None and tunnel_alm.dual_updates:
                    detail += (
                        f", last converged-primal tunnel normal residual {tunnel_alm.normal_error:.3g} m"
                        f" and dual force residual {tunnel_alm.dual_residual:.3g} N"
                    )
                if self._diverged_after:
                    raise NeedleCouplingNotConverged(
                        f"Needle/tissue coupling did not converge: the trial load diverged after "
                        f"{self._diverged_after} trials ({detail})"
                    )
                raise NeedleCouplingNotConverged(
                    f"Needle/tissue coupling did not converge in {self.max_iterations} trials ({detail})"
                )
            if pieces is not None:
                yield from self._pause(pieces, "end")
            # Interface convergence does not make a linearly implicit FEM step
            # inversion-free. Reject unresolved material motion while native
            # state, warm starts, events and output still belong to this trial.
            if self._gpu_checkpoint is not None:
                self.rigid_solver.check_overflow()
            minimum_tet_j = np.inf
            if self._tet_step_guard is not None:
                try:
                    minimum_tet_j = self._tet_step_guard.check(state_in, self._candidate)
                except NeedleStepTooLarge:
                    self.tet_step_rejections += 1
                    raise
            self.interaction.check_step_resolution(state_in, self._candidate, self.needle_body)
            prepared_observations = (
                self.rigid_solver.prepare_observations() if self._gpu_checkpoint is not None else None
            )
            self.interaction.retain_trial_forces(state_in, self.needle_body)
            self.interaction.accept(self._candidate, self.needle_body)
            if tunnel_alm is not None:
                tunnel_alm.commit()
            wp.copy(self._candidate.body_f, self._external_wrench)
            state_out.assign(self._candidate)
            if self._gpu_checkpoint is not None:
                self.rigid_solver.commit_observations(prepared=prepared_observations)
            committed = True
            if predictor is not None:
                predictor.commit(self._trial_load)
            self.minimum_accepted_tet_j = min(self.minimum_accepted_tet_j, minimum_tet_j)
            # The next step starts its trials from the load that was applied in the accepted trial,
            # not from the interface force evaluated at the accepted state. The two differ by the
            # accepted residual, and the evaluated force is the further one from the next solution.
            # The carried load only seeds the next search (and the frame-periodic guess above).
            wp.copy(self._load, self._trial_load)
            self.maximum_iterations = max(self.maximum_iterations, self.iterations)
            self.maximum_residual = max(self.maximum_residual, self.residual)
            if self._budget_kept:
                # The accepted trial is one the trial budget kept. Its residual and the surface
                # events its acceptance committed are host values of this step.
                self.coupling_budget_residual = max(self.coupling_budget_residual, self.residual)
                kinds = [event["kind"] for event in getattr(self.interaction, "events", ())]
                if getattr(self.interaction, "status", None) == "failed":
                    kinds.append("failed")
                for kind in kinds:
                    self.coupling_budget_events[kind] = self.coupling_budget_events.get(kind, 0) + 1
        except GeneratorExit:
            # Closed while it was left open. abandon_step() ends the attempt as a rejected one. A collector
            # that finalizes a dropped attempt must not issue device work at a time of its own.
            collected = not self._abandoning
            raise
        finally:
            if not collected:
                self._close_attempt(attempt, state_in, committed)
