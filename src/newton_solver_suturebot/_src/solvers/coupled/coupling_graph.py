# SPDX-License-Identifier: Apache-2.0
"""Device-controlled trials for the supported dynamic needle/FEM configuration.

The coupler retains the transaction: exact endpoint authorization, native rigid
history, swept motion guards and material-history acceptance remain authoritative.
This module changes execution placement, not force laws or convergence budgets.
"""

import numpy as np
import warp as wp

from ...utils.capture import ScopedCapture
from ...utils.graph_branches import Fork
from ..mujoco.mujoco_free_body_gpu import MuJoCoFreeBodyGPU
from .coupling import _apply_interface_load, _direction_statistics, _interface_residual
from .interface.needle_trial_forces import FALLBACK_REASONS, PUNCTURE_REASON, SEARCHABLE_REASONS, _NeedleTrialForces

# Integer state: iteration, mixing start, FEM tolerance level, converged,
# stopped (1: nonfinite trial, 2: diverging trial load, 3: tip moved beyond the step
# resolution), dual updates.
# Floating state: current relaxation, last residual, two tunnel residuals,
# and the applied load the divergence test refers to.
# Trial budget (optional, see NeedleTissueCoupler): integer book: 0 a best trial of the
# current multiplier epoch exists, 1 save the load of this trial as the best one,
# 2 restore the best load for one more trial, 3 that trial has been asked for,
# 4 the trials ended by the budget and the last trial is kept.
# One floating value: the smallest force residual of the current multiplier epoch.


@wp.kernel(enable_backward=False)
def _initialize(
    state: wp.array[int],
    values: wp.array[wp.float64],
    running: wp.array[int],
    fresh: wp.array[int],
    stale: wp.array[int],
    initial_relaxation: wp.float64,
    budget: wp.array[int],
):
    for i in range(6):
        state[i] = 0
    state[1] = 1
    values[0] = initial_relaxation
    values[1] = wp.float64(0.0)
    values[2] = wp.float64(0.0)
    values[3] = wp.float64(0.0)
    values[4] = wp.float64(0.0)
    running[0] = 1
    fresh[0] = 1
    stale[0] = 0
    for i in range(5):
        budget[i] = 0


@wp.kernel(enable_backward=False)
def _begin_trial(
    state: wp.array[int],
    values: wp.array[wp.float64],
    levels: wp.array[int],
    level_count: int,
    initial_relaxation: wp.float64,
):
    iteration = state[0] + 1
    state[0] = iteration
    if iteration > state[1] and (iteration - state[1]) % 8 == 0 and state[2] + 1 < level_count:
        state[2] += 1
        state[1] = iteration
        values[0] = initial_relaxation
    for i in range(level_count):
        levels[i] = int(state[2] == i)


@wp.kernel(enable_backward=False)
def _assess_trial(
    state: wp.array[int],
    values: wp.array[wp.float64],
    statistics: wp.array[wp.float64],
    alm_statistics: wp.array[float],
    geometry_fallback: wp.array[int],
    searchable: wp.array[int],
    seen: wp.array[int],
    running: wp.array[int],
    correct: wp.array[int],
    update_duals: wp.array[int],
    fresh: wp.array[int],
    stale: wp.array[int],
    absolute_tolerance: wp.float64,
    relative_tolerance: wp.float64,
    initial_relaxation: wp.float64,
    normal_tolerance: wp.float64,
    has_alm: int,
    iteration_limit: int,
    trial_budget: int,
    best_residual: wp.array[wp.float64],
    budget: wp.array[int],
):
    correct[0] = 0
    update_duals[0] = 0
    running[0] = 0
    # A dual-only update keeps the applied load, so the next trial's candidate
    # state, geometry and surface forces are those already evaluated; only the
    # tunnel's constitutive stage must be recomputed.
    fresh[0] = 1
    stale[0] = 0
    values[1] = statistics[0]
    # The trial load is neither saved nor restored unless this trial asks for it below.
    budget[1] = 0
    budget[2] = 0
    if statistics[5] != wp.float64(0.0):
        state[4] = 1
    if state[4] != 0:
        return
    if state[0] == 1 and relative_tolerance > wp.float64(0.0):
        # Applied load the trials start from. Below absolute / relative tolerance the
        # absolute tolerance governs, so that load is the smallest reference.
        values[4] = wp.max(statistics[2], absolute_tolerance / relative_tolerance)
    if geometry_fallback[0] != 0:
        # A request that only concerns a surface event does not end the search: the load
        # it reaches starts the host oracle, which decides the event. Others end it here.
        if seen[0] == 0:
            # Number of the first trial with a request.
            seen[0] = state[0]
        search = searchable[0]
        known = int(0)
        for i in range(1, geometry_fallback.shape[0]):
            if geometry_fallback[i] != 0:
                seen[i] = 1
                known = 1
                if searchable[i] == 0:
                    search = 0
        if search == 0 or known == 0:
            return
    tolerance = absolute_tolerance + relative_tolerance * wp.max(statistics[1], statistics[2])
    if statistics[0] <= tolerance:
        alm_ok = True
        if has_alm != 0:
            for i in range(5):
                if not wp.isfinite(alm_statistics[i]):
                    state[4] = 1
            if alm_statistics[4] != 0.0:
                state[4] = 1
            if state[4] != 0:
                return
            values[2] = wp.float64(wp.max(alm_statistics[0], alm_statistics[1]))
            values[3] = wp.float64(alm_statistics[2])
            # Match the host comparison: float32 statistics promoted to double.
            alm_ok = wp.float64(wp.max(alm_statistics[0], alm_statistics[1])) <= normal_tolerance and wp.float64(
                alm_statistics[2]
            ) <= absolute_tolerance + relative_tolerance * wp.float64(alm_statistics[3])
        if alm_ok:
            state[3] = 1
            return
        if trial_budget > 0 and state[0] >= trial_budget:
            # The trial budget is used up. This trial passes the force test and only waits
            # for a multiplier update, so it is the best trial of its epoch, and it is the
            # last one evaluated: it is kept as it is, without the update.
            state[3] = 1
            budget[4] = 1
            return
        update_duals[0] = 1
        fresh[0] = 0
        stale[0] = 1
        state[5] += 1
        state[1] = state[0] + 1
        values[0] = initial_relaxation
        # The update changes the force map: the best trial restarts with the next one.
        budget[0] = 0
    else:
        if state[0] > 1 and values[4] > wp.float64(0.0) and relative_tolerance * statistics[2] > values[4]:
            # The applied load has grown by 1 / relative tolerance over the load the trials
            # started from: the tolerance of this trial alone exceeds that whole load, so
            # the iteration is running away. End the trials; the step is rejected.
            state[4] = 2
            return
        if trial_budget > 0:
            if budget[3] != 0:
                # This trial applied the best load once more. It is kept whatever its residual.
                state[3] = 1
                budget[4] = 1
                return
            # Smallest force residual of the multiplier epoch; of equal trials, the later one.
            best = int(0)
            if budget[0] == 0:
                best = 1
            elif statistics[0] <= best_residual[0]:
                best = 1
            if state[0] >= trial_budget:
                # The trial budget is used up and the loop has neither failed nor diverged.
                if best != 0:
                    # The last trial is the best one: it is kept as it is.
                    state[3] = 1
                    budget[4] = 1
                    return
                if state[0] < iteration_limit:
                    # An earlier trial is the best one. Its load is applied once more, with no
                    # correction after this trial, and that trial is kept.
                    budget[2] = 1
                    budget[3] = 1
                    running[0] = 1
                return
            if best != 0:
                best_residual[0] = statistics[0]
                budget[0] = 1
                budget[1] = 1
        correct[0] = 1
    running[0] = int(state[0] < iteration_limit)


@wp.kernel(enable_backward=False)
def _exchange_best_load(
    budget: wp.array[int],
    load: wp.array[wp.float64],
    best: wp.array[wp.float64],
):
    # The two requests of the trial budget in one launch, before any correction of the load.
    i = wp.tid()
    if budget[1] != 0:
        best[i] = load[i]
    if budget[2] != 0:
        load[i] = best[i]


@wp.kernel(enable_backward=False)
def _update_relaxation(
    state: wp.array[int],
    values: wp.array[wp.float64],
    statistics: wp.array[wp.float64],
    preconditioned: int,
):
    if state[0] > state[1] and statistics[3] > wp.float64(1.0e-30):
        upper = wp.float64(1.0)
        if preconditioned != 0:
            upper = wp.float64(2.0)
        values[0] = wp.clamp(-values[0] * statistics[4] / statistics[3], wp.float64(0.001), upper)


@wp.kernel(enable_backward=False)
def _relax_load(
    direction: wp.array[wp.float64],
    values: wp.array[wp.float64],
    load: wp.array[wp.float64],
    previous: wp.array[wp.float64],
):
    i = wp.tid()
    load[i] += values[0] * direction[i]
    previous[i] = direction[i]


@wp.kernel(enable_backward=False)
def finite_candidate(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    valid: wp.array[int],
):
    i = wp.tid()
    if i < q.shape[0]:
        for j in range(3):
            if not wp.isfinite(q[i][j]) or not wp.isfinite(qd[i][j]):
                wp.atomic_min(valid, 0, 0)
    if i < poses.shape[0]:
        for j in range(7):
            if not wp.isfinite(poses[i][j]):
                wp.atomic_min(valid, 0, 0)
        for j in range(6):
            if not wp.isfinite(velocities[i][j]):
                wp.atomic_min(valid, 0, 0)


@wp.kernel(enable_backward=False)
def unresolved_tip_motion(
    start: wp.array[wp.transform],
    candidate: wp.array[wp.transform],
    body: int,
    tip: wp.vec3,
    resolution: float,
    valid: wp.array[int],
    unresolved: wp.array[int],
):
    # No step in which the needle moves further than the step resolution is accepted, and the
    # swept queries of such a trial cover the whole path of a thrown needle. Skip its forces.
    moved = wp.transform_point(candidate[body], tip) - wp.transform_point(start[body], tip)
    if wp.length(moved) > resolution:
        valid[0] = 0
        unresolved[0] = 1


@wp.kernel(enable_backward=False)
def reject_nonfinite(valid: wp.array[int], unresolved: wp.array[int], state: wp.array[int], running: wp.array[int]):
    if valid[0] == 0:
        state[4] = 1
        if unresolved[0] != 0:
            state[4] = 3
        running[0] = 0


@wp.kernel(enable_backward=False)
def _pack_result(
    state: wp.array[int],
    values: wp.array[wp.float64],
    fallback: wp.array[int],
    seen: wp.array[int],
    result: wp.array[wp.float64],
    budget: wp.array[int],
):
    for i in range(6):
        result[i] = wp.float64(state[i])
    for i in range(4):
        result[6 + i] = values[i]
    # Request in the last trial, then the number of the first trial with a request.
    result[10] = wp.float64(fallback[0])
    result[11] = wp.float64(seen[0])
    # The trials ended by the trial budget: the last trial is kept, not converged.
    result[12] = wp.float64(budget[4])


class _CapturedNeedleTrials:
    def __init__(self, coupler, *, rigid=None):
        self.c = coupler
        self.rigid = (
            MuJoCoFreeBodyGPU(coupler.model, coupler.rigid_solver, step_time=coupler.step_time)
            if rigid is None
            else rigid
        )
        self.device = coupler.model.device
        self.permissions = _NeedleTrialForces(
            coupler.interaction, query_block_dim=getattr(coupler, "_needle_query_block_dim", None)
        )
        self.state = wp.zeros(6, dtype=int, device=self.device)
        self.values = wp.zeros(5, dtype=wp.float64, device=self.device)
        self.running = wp.zeros(1, dtype=int, device=self.device)
        self.correct = wp.zeros(1, dtype=int, device=self.device)
        self.dual = wp.zeros(1, dtype=int, device=self.device)
        self.empty_alm = wp.zeros(5, dtype=float, device=self.device)
        self.finite = wp.ones(1, dtype=int, device=self.device)
        self.unresolved = wp.zeros(1, dtype=int, device=self.device)
        self.result = wp.zeros(13, dtype=wp.float64, device=self.device)
        # Trial budget: book, smallest force residual of the multiplier epoch and its load.
        self.budget = wp.zeros(5, dtype=int, device=self.device)
        self.best_residual = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.best_load = wp.zeros_like(coupler._trial_load)
        reasons = len(FALLBACK_REASONS)
        self.seen = wp.zeros(reasons, dtype=int, device=self.device)
        self.searchable = wp.zeros(reasons, dtype=int, device=self.device)
        self._search = (False, False)
        # Whether the accepted track transmits a shaft interaction, as the condition of a branch
        # of the graph (see branches_on_shaft_flag): prepare() writes it where the host value
        # changed. ``_shaft_active`` is the host's copy of what the device holds.
        self.shaft_active = wp.zeros(1, dtype=int, device=self.device)
        self._shaft_active = False
        # Dual-only repetitions reuse the previous trial's unchanged candidate:
        # `fresh` gates the primal solve/geometry/surface stage, `stale` its
        # restoration from these snapshots taken after each fresh evaluation.
        self.fresh = wp.ones(1, dtype=int, device=self.device)
        self.stale = wp.zeros(1, dtype=int, device=self.device)
        self.forces_base = wp.empty_like(coupler._forces)
        self.body_f_base = wp.empty_like(coupler._candidate.body_f)
        self.diagnostics_base = wp.empty_like(coupler.interaction._diagnostics)
        self.contact_flags_base = wp.empty_like(coupler.interaction._contact_flags)
        self.graph = None
        self.graphs = {}
        self.signature = None
        self.captures = self.launches = 0
        self.linear_solvers = []
        self._linear_signature = None
        self._fem_prepared = False

    @property
    def evaluates_shaft_flag(self):
        """Whether these trials evaluate the flag of the shaft correction before every launch.

        They do where the correction of a trial follows the flag (see
        :attr:`branches_on_shaft_flag` for how the graph follows it). With a surface metric
        every trial is corrected, by one solve whose track launches read the live count: the
        graph is the same for both values, and the flag, whose evaluation reads the engagement
        from the device, is not evaluated.
        """
        return not getattr(self.rigid, "surface_preconditioned", False)

    @property
    def branches_on_shaft_flag(self):
        """Whether one graph holds the correction of both values of the flag, behind a device branch.

        It does for the native free needle, whose rigid interval has no correction of its own
        (no ``apply_preconditioner``), where the correction of the track is its device part
        alone (no response-basis start guess, whose binding is host work of every correction):
        the graph then issues the corrected and the uncorrected relaxation behind
        ``shaft_active``, which :meth:`prepare` writes, and a flip of the flag between two steps
        neither drops nor captures a graph. Elsewhere the flag, where it is evaluated, is an
        entry of the signature, and a flip captures the graph of the other value: with a
        response-basis start guess, and for every rigid adapter that applies the correction
        itself (the articulated tools on MuJoCo Warp without a surface metric).
        """
        return (
            self.evaluates_shaft_flag
            and not hasattr(self.rigid, "apply_preconditioner")
            and getattr(self.c.fem_solver, "_affine_predictor", None) is None
        )

    def invalidate(self):
        self.graph = None
        self.graphs.clear()
        self.signature = None
        if self.linear_solvers:
            # No graph of this owner binds them any more: the solver's registry may evict them.
            self.c.fem_solver.unpin_linear_workspaces([workspace for _, workspace in self.linear_solvers])
        self.linear_solvers = []
        self._linear_signature = None
        self._fem_prepared = False

    def prepare(self, state_in, control, dt):
        """Prepare scratch once per changed graph shape without advancing history."""
        c, solver = self.c, self.c.fem_solver
        # The time step is device data for every launch of the graph. It is written
        # here, outside capture, on every substep: a changed dt changes no signature
        # entry, so it neither clears nor recaptures the graphs.
        c._share_step_time()
        c.step_time.set(dt)
        search = (
            bool(getattr(c, "search_past_surface_events", False)),
            bool(getattr(c, "search_past_puncture", False)),
        )
        if search != self._search:
            # Device data, read by the assessment kernel: no recapture when it changes.
            reasons = SEARCHABLE_REASONS + ((PUNCTURE_REASON,) if search[1] else ())
            mask = [int(search[0])] + [int(reason in reasons) for reason in FALLBACK_REASONS[1:]]
            self.searchable.assign(np.asarray(mask, dtype=np.int32))
            self._search = search
        changed = self.permissions.prepare(c.interaction)
        self.rigid.prepare(dt)
        if not hasattr(self.rigid, "apply_preconditioner"):
            # The correction reads the native rigid damping from a device array as well.
            # Stage it on every substep: an unchanged signature returns early below. A rigid
            # adapter with a correction of its own has staged both in its prepare() above.
            c._shaft_preconditioner.prepare(dt)
        # Captured trials omit the projection of invariant assembled values;
        # refresh it eagerly whenever the tangent changed since the last launch.
        c._refresh_projection()
        tolerances = solver.tolerance_levels()
        # Neither the time step nor the native rigid damping is an entry: the kernels
        # read both from device arrays that the host preparations above write (StepTime;
        # the damping copies of the free-body trials and of the shaft preconditioner).
        # Nor is the number of track samples: the correction's launches over the track
        # have the size of the track reserve and read the live count on the device, so
        # its entry is the storage that those launches address.
        # Whether the track transmits a shaft interaction is evaluated where the correction
        # follows it (evaluates_shaft_flag). It is device data where one graph holds both
        # corrections (branches_on_shaft_flag), written here when the host value changed, and an
        # entry of the signature where it is evaluated and the graph holds one correction.
        shaft_entry = None
        if self.evaluates_shaft_flag:
            active = bool(c._shaft_preconditioner.active)
            if not self.branches_on_shaft_flag:
                shaft_entry = active
            elif active != self._shaft_active:
                self.shaft_active.fill_(int(active))
                self._shaft_active = active
        signature = (
            (wp.get_stream(self.device).cuda_stream if self.device.is_cuda else None),
            tuple(tolerances),
            solver.linear_solver_maxiter,
            c._shaft_preconditioner.storage_key(),
            shaft_entry,
            solver.A_bsr.values.ptr,
            solver.A_bsr.columns.ptr,
            solver.A_bsr.offsets.ptr,
            solver._pc_inv_diag.ptr,
            id(solver.M_bsr),
            solver.min_stretch,
            solver.max_stretch,
            solver.linear_damping,
            c.absolute_tolerance,
            c.relative_tolerance,
            c.max_iterations,
            getattr(c, "trial_budget", None),
            getattr(c, "parallel_trial_branches", False),
            c.relaxation,
            c.moment_length,
            c.needle_body,
            tuple(c.needle_com),
            self.permissions.signature(),
            None if getattr(solver, "_certified_linear", None) is None else solver._certified_linear.key(),
            None if getattr(solver, "_affine_predictor", None) is None else solver._affine_predictor.capture_key(),
            None if getattr(solver, "_held_start", None) is None else solver._held_start.capture_key(),
            None if getattr(self.rigid, "projection", None) is None else self.rigid.projection.key,
            getattr(self.rigid, "guess_key", None),
            # The graphs bind the Krylov workspaces of the solver's registry, one per
            # tolerance level. This number changes when the registry was dropped, and
            # with it those workspaces; it stays the last entry.
            solver.linear_binding_epoch(),
        )
        if not changed and signature == self.signature:
            return
        self.graph = None
        self.graphs.clear()
        self.signature = None
        self.tolerances = tolerances
        if not hasattr(self, "levels") or len(self.level_views) != len(tolerances):
            self.levels = wp.zeros(len(tolerances), dtype=int, device=self.device)
            self.level_views = [self.levels[i : i + 1] for i in range(len(tolerances))]
        if not self._fem_prepared:
            # One initial execution prepares the structural operator and force
            # buffers. Later dt/history changes only refresh persistent values.
            saved_increment = wp.clone(solver.velocity_increment)
            saved_forces = wp.clone(state_in.particle_f)
            try:
                c._candidate.assign(state_in)
                c._forces.zero_()
                solver.step(state_in, c._candidate, control, None, dt, external_forces=c._forces)
            finally:
                wp.copy(solver.velocity_increment, saved_increment)
                wp.copy(state_in.particle_f, saved_forces)
                if (affine := getattr(solver, "_affine_predictor", None)) is not None:
                    affine.reset()
                if (certificate := getattr(solver, "_certified_linear", None)) is not None and certificate.active:
                    certificate.status.zero_()
            self._fem_prepared = True
        operator = solver.A_bsr
        if solver._fixed_particles is not None:
            # Keep the version recorded by the eager refresh above. Without it the
            # first captured solve sees a stale version and re-projects the whole
            # operator inside the graph on every replay.
            if solver._fixed_particles.prepare(solver.A_bsr, values_version=solver._projection_values_version()):
                solver._lin_solver = solver._lin_solver_key = None
            operator = solver._fixed_particles.operator
        settings = tuple(solver._linear_solver_settings(value) for value in tolerances)
        fixed = solver._fixed_particles
        compact = None if fixed is None or fixed.compact_matrix is None else fixed.compact_matrix
        # Read again: the warm step and the projection above may have dropped the registry.
        epoch = solver.linear_binding_epoch()
        linear_signature = (
            id(operator),
            id(solver.M_bsr),
            solver._normalized_rhs.ptr,
            solver.dv.ptr,
            None if compact is None else (compact.values.ptr, fixed.compact_rhs.ptr, fixed.compact_increment.ptr),
            settings,
            # The registry keys its workspaces by the state of the certificate scope as well.
            None if getattr(solver, "_certified_linear", None) is None else solver._certified_linear.key(),
            epoch,
        )
        if linear_signature != self._linear_signature:
            # One workspace per tolerance level from the solver's registry, which makes a
            # missing one with the public Warp run=False factories (they allocate all Krylov
            # scratch; neither extra tolerance branches nor the PC need a warm solve). The
            # host-controlled trials select the same workspaces at the same tolerances, so
            # a host trial after device trials finds its captured FEM step.
            fetched = [solver.linear_workspace(value) for value in tolerances]
            stale = [w for _, w in self.linear_solvers if all(w is not kept for _, kept in fetched)]
            if stale:
                solver.unpin_linear_workspaces(stale)
            self.linear_solvers = fetched
            self._linear_signature = linear_signature
        solver._lin_solver_key, solver._lin_solver = self.linear_solvers[0]
        self.signature = signature[:-1] + (epoch,)

    def solve(self, state_in, control, dt):
        c, solver, pc = self.c, self.c.fem_solver, self.c._shaft_preconditioner
        self.prepare(state_in, control, dt)
        alm = c.interaction.tunnel_alm
        # One graph for both values of the flag of the track: its correction is a device branch.
        branches = self.branches_on_shaft_flag
        # The surface metric first: with it the flag of the track is not evaluated.
        preconditioned = bool(not self.evaluates_shaft_flag or pc.active)
        # Fixed at capture, as the trial limit is: both are entries of the graph signature.
        trial_budget = getattr(c, "trial_budget", None)

        def fem(level):
            solver.linear_solver_tolerance = self.tolerances[level]
            solver._lin_solver_key, solver._lin_solver = self.linear_solvers[level]
            solver.step(state_in, c._candidate, control, None, dt, external_forces=c._forces)

        def update_duals():
            wp.copy(alm.normal, alm.proposed_normal)
            wp.copy(alm.tangent, alm.proposed_tangent)

        def correct(preconditioned=preconditioned):
            if hasattr(self.rigid, "update_numerical_map"):
                self.rigid.update_numerical_map(self.state, self.values, c.relaxation)
            direction = c._residual
            if preconditioned:
                if hasattr(self.rigid, "apply_preconditioner"):
                    self.rigid.apply_preconditioner(pc, c._candidate, dt, c._residual, c._direction)
                else:
                    pc.apply(c._candidate, dt, c._residual, c._direction)
                direction = c._direction

            def relaxed():
                wp.launch(
                    _relax_load,
                    c._load.size,
                    [direction, self.values, c._trial_load, c._previous_residual],
                    device=self.device,
                )

            acceleration = getattr(self.rigid, "anderson", None)
            if acceleration is not None:
                # Match the host AA-owned branch's constant ordinary fallback;
                # the small-Gram solve is a guarded numerical variant of SVD.
                acceleration.propose_device(c._trial_load, c._residual, self.values, direction, self.state)
                wp.capture_if(acceleration.not_applied, relaxed)
            else:
                wp.launch(
                    _direction_statistics,
                    c._load.size,
                    [direction, c._previous_residual, c._statistics],
                    device=self.device,
                )
                wp.launch(
                    _update_relaxation,
                    1,
                    [self.state, self.values, c._statistics, int(preconditioned)],
                    device=self.device,
                )
                relaxed()

        def correct_by_flag():
            # Both corrections of the relaxation, each as the graph of its value issued it.
            def corrected():
                # pc.apply() decides on the host whether the track is corrected at all (a track
                # sample, an engaged track). Here that is the device flag in front of this branch,
                # so the host gate is lifted while the branch is captured, with or without a
                # track: the launches read the live track count, and prepare() stages the
                # coefficients in every substep whose flag is set.
                gated, pc._device_gated = pc._device_gated, True
                try:
                    correct(True)
                finally:
                    pc._device_gated = gated

            wp.capture_if(self.shaft_active, on_true=corrected, on_false=lambda: correct(False))

        def fresh_surface():
            c._forces.zero_()
            c._candidate.body_f.zero_()
            # The primal stage of this trial already refitted the geometry.
            self.permissions.add_surface_forces(
                c._candidate, c.needle_body, c.needle_com, c._forces, dt, geometry_current=True
            )
            if hasattr(self.rigid, "validate_surface"):
                self.rigid.validate_surface()
            # Snapshot the dual-independent stage for dual-only repetitions.
            wp.copy(self.forces_base, c._forces)
            wp.copy(self.body_f_base, c._candidate.body_f)
            wp.copy(self.diagnostics_base, c.interaction._diagnostics)
            wp.copy(self.contact_flags_base, c.interaction._contact_flags)

        def repeat_surface():
            # The candidate, geometry, permissions and surface forces of a
            # dual-only repetition are those of the previous trial, verbatim.
            wp.copy(c._forces, self.forces_base)
            wp.copy(c._candidate.body_f, self.body_f_base)
            wp.copy(c.interaction._diagnostics, self.diagnostics_base)
            wp.copy(c.interaction._contact_flags, self.contact_flags_base)

        def evaluate():
            wp.capture_if(self.fresh, fresh_surface)
            wp.capture_if(self.stale, repeat_surface)
            self.permissions.add_history_forces(c._candidate, c.needle_body, c.needle_com, c._forces, dt)
            c._statistics.zero_()
            wp.launch(
                _interface_residual,
                c._load.size,
                [
                    c._forces,
                    c._candidate.body_f,
                    c.needle_body,
                    c.moment_length,
                    c._trial_load,
                    c._previous_residual,
                    0,
                    c._value,
                    c._residual,
                    c._statistics,
                ],
                device=self.device,
            )
            wp.launch(
                _assess_trial,
                1,
                [
                    self.state,
                    self.values,
                    c._statistics,
                    self.empty_alm if alm is None else alm._statistics,
                    self.permissions.fallback,
                    self.searchable,
                    self.seen,
                    self.running,
                    self.correct,
                    self.dual,
                    self.fresh,
                    self.stale,
                    c.absolute_tolerance,
                    c.relative_tolerance,
                    c.relaxation,
                    0.0 if alm is None else alm.tolerance,
                    int(alm is not None),
                    c.max_iterations,
                    0 if trial_budget is None else int(trial_budget),
                    self.best_residual,
                    self.budget,
                ],
                device=self.device,
            )
            if trial_budget is not None:
                # Save the load of a new best trial, or restore the best load for the one
                # last trial, before the correction below can change the trial load.
                wp.launch(
                    _exchange_best_load,
                    c._load.size,
                    [self.budget, c._trial_load, self.best_load],
                    device=self.device,
                )
            if alm is not None:
                wp.capture_if(self.dual, update_duals)
            wp.capture_if(self.correct, correct_by_flag if branches else correct)

        def primal():
            wp.launch(
                _apply_interface_load,
                max(c.model.particle_count, c.model.body_count),
                [
                    c._trial_load,
                    c._external_nodes,
                    c._external_wrench,
                    c.needle_body,
                    c.moment_length,
                    c._forces,
                    state_in.body_f,
                ],
                device=self.device,
            )
            # FEM and rigid integration overwrite every evolving coordinate.
            # Preserve the published pre-solve FEM force scratch separately.
            wp.copy(c._candidate.particle_f, state_in.particle_f)

            def tissue():
                for level, condition in enumerate(self.level_views):
                    wp.capture_if(condition, fem, level=level)
                # Surface meshes and tetrahedron queries follow the tissue alone, so
                # their refit belongs to this stage and overlaps the rigid interval.
                self.permissions.refit_geometry(c._candidate)

            fork = None
            if getattr(c, "parallel_trial_branches", False) and getattr(self.rigid, "concurrent_with_tissue", False):
                fork = Fork.begin(self.device)
            if fork is None:
                tissue()
                self.rigid.step(state_in, c._candidate, control, dt)
            else:
                # Both stages start from the applied load. The tissue solve writes
                # particle state and its own scratch; the rigid interval writes
                # rigid state and backend data. Neither reads what the other writes.
                with fork.branch():
                    tissue()
                with fork.branch():
                    self.rigid.step(state_in, c._candidate, control, dt, after_conditional=fork.isolate)
                fork.join()
            self.finite.fill_(1)
            wp.launch(
                finite_candidate,
                max(c.model.particle_count, c.model.body_count),
                [
                    c._candidate.particle_q,
                    c._candidate.particle_qd,
                    c._candidate.body_q,
                    c._candidate.body_qd,
                    self.finite,
                ],
                device=self.device,
            )
            self.unresolved.zero_()
            wp.launch(
                unresolved_tip_motion,
                1,
                [
                    state_in.body_q,
                    c._candidate.body_q,
                    c.needle_body,
                    wp.vec3(*c.interaction.tip_local),
                    float(c.interaction.step_resolution),
                    self.finite,
                    self.unresolved,
                ],
                device=self.device,
            )
            if hasattr(self.rigid, "include_validity"):
                self.rigid.include_validity(self.finite)
            certificate = getattr(c.fem_solver, "_certified_linear", None)
            if certificate is not None and certificate.active:
                from ..fem.certified_scope import require_certificate

                wp.launch(require_certificate, 1, [certificate.status, self.finite], device=self.device)
            wp.launch(reject_nonfinite, 1, [self.finite, self.unresolved, self.state, self.running], device=self.device)

        def trial():
            wp.launch(
                _begin_trial,
                1,
                [self.state, self.values, self.levels, len(self.tolerances), c.relaxation],
                device=self.device,
            )
            # A dual-only repetition applies the identical load to the identical
            # start, so its primal solve, rigid motion, finiteness and geometry
            # are those already computed and checked in the previous trial.
            wp.capture_if(self.fresh, primal)
            wp.capture_if(self.finite, evaluate)

        inputs = (
            state_in.particle_q,
            state_in.particle_qd,
            state_in.particle_f,
            state_in.body_q,
            state_in.body_qd,
            state_in.body_f,
            state_in.joint_q,
            state_in.joint_qd,
            getattr(control, "joint_f", None),
            getattr(control, "joint_target_q", None),
            getattr(control, "joint_target_qd", None),
            getattr(getattr(control, "mujoco", None), "ctrl", None),
            *(
                getattr(c._candidate, name, None)
                for name in (
                    "particle_q",
                    "particle_qd",
                    "particle_f",
                    "body_q",
                    "body_qd",
                    "body_f",
                    "joint_q",
                    "joint_qd",
                    "body_qdd",
                    "body_parent_f",
                )
            ),
            getattr(getattr(c._candidate, "mujoco", None), "qfrc_actuator", None),
        )
        key = tuple(None if array is None else (array.ptr, array.shape, array.strides) for array in inputs)
        entry = self.graphs.get(key)
        if entry is None:
            tolerance, linear_key, linear = solver.linear_solver_tolerance, solver._lin_solver_key, solver._lin_solver
            conditional = pc._conditional
            if branches:
                # The corrected branch is captured whatever the track holds now. Its coefficients
                # are staged here, outside the capture: prepare() stages them for an engaged track.
                pc.stage(dt, tape=True)
            try:
                pc._conditional = False
                with ScopedCapture(device=self.device) as capture:
                    # Refresh untouched/custom state once on every replay.
                    # Trial forces clear body_f before writing their reaction.
                    c._candidate.assign(state_in)
                    if hasattr(self.rigid, "reset_numerical_map"):
                        self.rigid.reset_numerical_map()
                    if hasattr(self.rigid, "begin_trials"):
                        # what the rigid stage of every trial shares is issued once per launch
                        self.rigid.begin_trials(state_in, control, dt)
                    self.seen.zero_()
                    wp.launch(
                        _initialize,
                        1,
                        [self.state, self.values, self.running, self.fresh, self.stale, c.relaxation, self.budget],
                        device=self.device,
                    )
                    wp.capture_while(self.running, trial)
                    wp.launch(
                        _pack_result,
                        1,
                        [self.state, self.values, self.permissions.fallback, self.seen, self.result, self.budget],
                        device=self.device,
                    )
                self.graph = capture.graph
                if len(self.graphs) >= 2:
                    self.graphs.clear()
                self.graphs[key] = (
                    self.graph,
                    inputs
                    + tuple(getattr(self.rigid, "capture_owners", ()))
                    + (
                        getattr(c.fem_solver, "_certified_linear", None),
                        getattr(c.fem_solver, "_affine_predictor", None),
                        getattr(c.fem_solver, "_held_start", None),
                        ()
                        if getattr(c.fem_solver, "_affine_predictor", None) is None
                        else c.fem_solver._affine_predictor.capture_owners,
                        () if getattr(c.fem_solver, "_held_start", None) is None else c.fem_solver._held_start.owners,
                    ),
                )
                self.captures += 1
            finally:
                if hasattr(self.rigid, "end_trials"):
                    self.rigid.end_trials()
                pc._conditional = conditional
                solver.linear_solver_tolerance = tolerance
                solver._lin_solver_key, solver._lin_solver = linear_key, linear
        else:
            self.graph = entry[0]
        wp.capture_launch(self.graph)
        self.launches += 1
        # A single completed-loop readback; no host decisions inside the graph.
        return self.result.numpy()
