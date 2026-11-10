# SPDX-License-Identifier: Apache-2.0
"""Device-controlled trials for the supported dynamic needle/FEM configuration.

The coupler retains the transaction: exact endpoint authorization, native rigid
history, swept motion guards and material-history acceptance remain authoritative.
This module changes execution placement, not force laws or convergence budgets.
"""

import warp as wp

from .coupling import _apply_interface_load, _direction_statistics, _interface_residual
from .mujoco_free_body_gpu import MuJoCoFreeBodyGPU
from .needle_trial_forces import _NeedleTrialForces

# Integer state: iteration, mixing start, FEM tolerance level, converged,
# nonfinite, dual updates. Floating state: current relaxation, last residual.


@wp.kernel(enable_backward=False)
def _initialize(
    state: wp.array[int],
    values: wp.array[wp.float64],
    running: wp.array[int],
    initial_relaxation: wp.float64,
):
    for i in range(6):
        state[i] = 0
    state[1] = 1
    values[0] = initial_relaxation
    values[1] = wp.float64(0.0)
    values[2] = wp.float64(0.0)
    values[3] = wp.float64(0.0)
    running[0] = 1


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
    running: wp.array[int],
    correct: wp.array[int],
    update_duals: wp.array[int],
    absolute_tolerance: wp.float64,
    relative_tolerance: wp.float64,
    initial_relaxation: wp.float64,
    normal_tolerance: wp.float64,
    has_alm: int,
    iteration_limit: int,
):
    correct[0] = 0
    update_duals[0] = 0
    running[0] = 0
    values[1] = statistics[0]
    if statistics[5] != wp.float64(0.0):
        state[4] = 1
    if geometry_fallback[0] != 0 or state[4] != 0:
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
        update_duals[0] = 1
        state[5] += 1
        state[1] = state[0] + 1
        values[0] = initial_relaxation
    else:
        correct[0] = 1
    running[0] = int(state[0] < iteration_limit)


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
def reject_nonfinite(valid: wp.array[int], state: wp.array[int], running: wp.array[int]):
    if valid[0] == 0:
        state[4] = 1
        running[0] = 0


@wp.kernel(enable_backward=False)
def _pack_result(
    state: wp.array[int], values: wp.array[wp.float64], fallback: wp.array[int], result: wp.array[wp.float64]
):
    for i in range(6):
        result[i] = wp.float64(state[i])
    for i in range(4):
        result[6 + i] = values[i]
    result[10] = wp.float64(fallback[0])


class _CapturedNeedleTrials:
    def __init__(self, coupler):
        self.c = coupler
        self.rigid = MuJoCoFreeBodyGPU(coupler.model, coupler.rigid_solver)
        self.device = coupler.model.device
        self.permissions = _NeedleTrialForces(
            coupler.interaction, query_block_dim=getattr(coupler, "_needle_query_block_dim", None)
        )
        self.state = wp.zeros(6, dtype=int, device=self.device)
        self.values = wp.zeros(4, dtype=wp.float64, device=self.device)
        self.running = wp.zeros(1, dtype=int, device=self.device)
        self.correct = wp.zeros(1, dtype=int, device=self.device)
        self.dual = wp.zeros(1, dtype=int, device=self.device)
        self.empty_alm = wp.zeros(5, dtype=float, device=self.device)
        self.finite = wp.ones(1, dtype=int, device=self.device)
        self.result = wp.zeros(11, dtype=wp.float64, device=self.device)
        self.graph = None
        self.graphs = {}
        self.signature = None
        self.captures = self.launches = 0
        self.linear_solvers = []
        self._linear_signature = None
        self._fem_prepared = False

    def invalidate(self):
        self.graph = None
        self.graphs.clear()
        self.signature = None
        self.linear_solvers = []
        self._linear_signature = None
        self._fem_prepared = False

    def prepare(self, state_in, control, dt):
        """Prepare scratch once per changed graph shape without advancing history."""
        c, solver = self.c, self.c.fem_solver
        changed = self.permissions.prepare(c.interaction)
        self.rigid.prepare(dt)
        tolerance = solver.linear_solver_tolerance
        tolerances = [tolerance]
        while True:
            value = max(min(tolerance, 1e-6), tolerances[-1] * 0.1)
            if value == tolerances[-1]:
                break
            tolerances.append(value)
        signature = (
            dt,
            tuple(tolerances),
            solver.linear_solver_maxiter,
            c.interaction.track_count,
            c._shaft_preconditioner.active,
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
            c.relaxation,
            c.moment_length,
            c.needle_body,
            tuple(c.needle_com),
            tuple(c.rigid_solver.mj_model.dof_damping),
            self.permissions.signature(),
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
            self._fem_prepared = True
        operator = solver.A_bsr
        if solver._fixed_particles is not None:
            if solver._fixed_particles.prepare(solver.A_bsr):
                solver._lin_solver = solver._lin_solver_key = None
            operator = solver._fixed_particles.operator
        settings = tuple(solver._linear_solver_settings(value) for value in tolerances)
        linear_signature = (id(operator), id(solver.M_bsr), solver._normalized_rhs.ptr, solver.dv.ptr, settings)
        if linear_signature != self._linear_signature:
            # Public Warp run=False factories allocate all Krylov scratch;
            # neither extra tolerance branches nor the PC need a warm solve.
            prepared = [(key, solver._make_linear_solver(operator, solver.M_bsr, key)) for key in settings]
            self.linear_solvers = prepared
            self._linear_signature = linear_signature
        solver._lin_solver_key, solver._lin_solver = self.linear_solvers[0]
        c._shaft_preconditioner.prepare(dt)
        self.signature = signature

    def solve(self, state_in, control, dt):
        c, solver, pc = self.c, self.c.fem_solver, self.c._shaft_preconditioner
        self.prepare(state_in, control, dt)
        alm = c.interaction.tunnel_alm
        preconditioned = bool(pc.active)

        def fem(level):
            solver.linear_solver_tolerance = self.tolerances[level]
            solver._lin_solver_key, solver._lin_solver = self.linear_solvers[level]
            solver.step(state_in, c._candidate, control, None, dt, external_forces=c._forces)

        def update_duals():
            wp.copy(alm.normal, alm.proposed_normal)
            wp.copy(alm.tangent, alm.proposed_tangent)

        def correct():
            direction = c._residual
            if preconditioned:
                pc.apply(c._candidate, dt, c._residual, c._direction)
                direction = c._direction
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
            wp.launch(
                _relax_load,
                c._load.size,
                [direction, self.values, c._trial_load, c._previous_residual],
                device=self.device,
            )

        def evaluate():
            c._forces.zero_()
            c._candidate.body_f.zero_()
            self.permissions.add_forces(c._candidate, c.needle_body, c.needle_com, c._forces, dt)
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
                    self.running,
                    self.correct,
                    self.dual,
                    c.absolute_tolerance,
                    c.relative_tolerance,
                    c.relaxation,
                    0.0 if alm is None else alm.tolerance,
                    int(alm is not None),
                    c.max_iterations,
                ],
                device=self.device,
            )
            if alm is not None:
                wp.capture_if(self.dual, update_duals)
            wp.capture_if(self.correct, correct)

        def trial():
            wp.launch(
                _begin_trial,
                1,
                [self.state, self.values, self.levels, len(self.tolerances), c.relaxation],
                device=self.device,
            )
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
            for level, condition in enumerate(self.level_views):
                wp.capture_if(condition, fem, level=level)
            self.rigid.step(state_in, c._candidate, control, dt)
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
            wp.launch(reject_nonfinite, 1, [self.finite, self.state, self.running], device=self.device)
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
            control.joint_f,
        )
        key = tuple(None if array is None else (array.ptr, array.shape, array.strides) for array in inputs)
        entry = self.graphs.get(key)
        if entry is None:
            tolerance, linear_key, linear = solver.linear_solver_tolerance, solver._lin_solver_key, solver._lin_solver
            conditional = pc._conditional
            try:
                pc._conditional = False
                with wp.ScopedCapture(device=self.device) as capture:
                    # Refresh untouched/custom state once on every replay.
                    # Trial forces clear body_f before writing their reaction.
                    c._candidate.assign(state_in)
                    wp.launch(
                        _initialize,
                        1,
                        [self.state, self.values, self.running, c.relaxation],
                        device=self.device,
                    )
                    wp.capture_while(self.running, trial)
                    wp.launch(
                        _pack_result,
                        1,
                        [self.state, self.values, self.permissions.fallback, self.result],
                        device=self.device,
                    )
                self.graph = capture.graph
                if len(self.graphs) >= 2:
                    self.graphs.clear()
                self.graphs[key] = (self.graph, inputs)
                self.captures += 1
            finally:
                pc._conditional = conditional
                solver.linear_solver_tolerance = tolerance
                solver._lin_solver_key, solver._lin_solver = linear_key, linear
        else:
            self.graph = entry[0]
        wp.capture_launch(self.graph)
        self.launches += 1
        # A single completed-loop readback; no host decisions inside the graph.
        return self.result.numpy()
