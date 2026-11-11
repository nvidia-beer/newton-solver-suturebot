# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen shaft tangent used only to accelerate the exact interface residual.

For residual r in (nodal force, rigid force, torque/L) coordinates, return
r-h J_full.T D J_free (M+h J_free.T D J_free)^-1 r.
Free rows equal M (M+h J_free.T D J_free)^-1 r for an exact inner solve;
the full transpose also returns fixed-support reactions and preserves internal
load balance when the inner solve is approximate. M is projected FEM A plus
native free-body effective inertia; rigid velocity coordinates are (v, L*omega). For each engaged anchor,
D=ell*(h*k+c)*(I-t*t.T) + mu*|normal_force|*v_eps/max(v_eps,|slip|)^2*(t*t.T).
The positive tangential approximation decreases during saturated sliding.
It is not a quadratic majorizer or a full-contact energy model. Surface and
nonsymmetric cross derivatives are omitted; the unchanged outer physical
residual remains the acceptance test.

For the optional tunnel ALM, D uses the exact radial proximal derivative
(frozen shaft projection), plus the tangential derivative on the sticking
branch. It omits the nonsymmetric change of the Coulomb bound with normal
force. Dual values are read without modification; the physical ALM residual
remains responsible for constraint and complementarity convergence.
"""

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator, cg

from .insertion import _shaft_projection
from .mujoco_free_body import MuJoCoFreeBody
from .solvers.fem.differentiation import active_tape
from .solvers.fem.kernels import inverse_diagonal_block
from .solvers.fem.solver import SolverFEM


@wp.kernel(enable_backward=False)
def prepare_anchors(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    lengths: wp.array[float],
    engaged: wp.array[int],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    samples: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    body: int,
    com: wp.vec3,
    radius: float,
    stiffness: float,
    damping: float,
    friction: float,
    speed: float,
    h: float,
    moment_length: float,
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
):
    i = wp.tid()
    matrix = wp.mat33(0.0)
    lever = wp.vec3(0.0)
    if engaged[i] != 0 and lengths[i] > 0.0:
        point = wp.vec3(0.0)
        tissue_v = wp.vec3(0.0)
        for j in range(4):
            point += bary[i][j] * q[nodes[i][j]]
            tissue_v += bary[i][j] * qd[nodes[i][j]]
        pose = poses[body]
        closest, tangent, distance, covered = _shaft_projection(point, pose, samples, radius)
        center = wp.transform_point(pose, com)
        arm = point - center
        lever = arm / moment_length
        relative = wp.spatial_top(velocities[body]) + wp.cross(wp.spatial_bottom(velocities[body]), arm) - tissue_v
        sliding = wp.dot(relative, tangent)
        normal_velocity = relative - sliding * tangent
        transverse = closest - point
        transverse -= wp.dot(transverse, tangent) * tangent
        normal_force = -lengths[i] * (stiffness * transverse + damping * normal_velocity)
        projection = wp.identity(n=3, dtype=float) - wp.outer(tangent, tangent)
        matrix = lengths[i] * (h * stiffness + damping) * projection
        tangent_coefficient = friction * wp.length(normal_force) / wp.max(speed, wp.abs(sliding))
        if wp.abs(sliding) > speed:
            tangent_coefficient *= wp.max(0.0, speed / wp.abs(sliding))
        matrix += tangent_coefficient * wp.outer(tangent, tangent)
    matrices[i] = matrix
    levers[i] = lever


@wp.kernel(enable_backward=False)
def prepare_alm_anchors(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    lengths: wp.array[float],
    engaged: wp.array[int],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    samples: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    body: int,
    com: wp.vec3,
    radius: float,
    normal: wp.array[wp.vec3],
    tangential: wp.array[float],
    penalty: float,
    compliance: float,
    clearance: float,
    friction: float,
    h: float,
    moment_length: float,
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
):
    i = wp.tid()
    matrix = wp.mat33(0.0)
    lever = wp.vec3(0.0)
    if engaged[i] != 0 and lengths[i] > 0.0:
        point = wp.vec3(0.0)
        tissue_v = wp.vec3(0.0)
        for j in range(4):
            point += bary[i][j] * q[nodes[i][j]]
            tissue_v += bary[i][j] * qd[nodes[i][j]]
        pose = poses[body]
        closest, tangent, distance, covered = _shaft_projection(point, pose, samples, radius)
        center = wp.transform_point(pose, com)
        arm = point - center
        lever = arm / moment_length
        relative = wp.spatial_top(velocities[body]) + wp.cross(wp.spatial_bottom(velocities[body]), arm) - tissue_v
        slip = wp.dot(relative, tangent)
        projection = wp.identity(n=3, dtype=float) - wp.outer(tangent, tangent)
        z = projection * (normal[i] + penalty * (closest - point))
        threshold = penalty * clearance
        factor = 1.0 / (1.0 + penalty * compliance)
        normal_stiffness = factor * penalty
        if compliance > 0.0:
            z = projection * (closest - point)
            threshold = clearance
            factor = 1.0 / compliance
            normal_stiffness = factor
        magnitude = wp.length(z)
        if magnitude > threshold:
            direction = z / magnitude
            fraction = threshold / magnitude
            # Exact local normal derivative: compliant constitutive law, or
            # the unchanged proximal map for a zero-compliance hard wall.
            derivative = normal_stiffness * ((1.0 - fraction) * projection + fraction * wp.outer(direction, direction))
            matrix = lengths[i] * h * derivative
            normal_magnitude = factor * (magnitude - threshold)
            bound = friction * normal_magnitude
            tangent_trial = tangential[i] + penalty * h * slip
            if wp.abs(tangent_trial) < bound:
                matrix += lengths[i] * penalty * h * wp.outer(tangent, tangent)
    matrices[i] = matrix
    levers[i] = lever


@wp.kernel(enable_backward=False)
def mass_action(
    fixed: wp.array[int],
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    rigid_blocks: wp.array[wp.mat33],
    x: wp.array[wp.vec3],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    count = fixed.shape[0]
    result = wp.vec3(0.0)
    if i < count:
        if fixed[i] != 0:
            result = x[i]
        else:
            for k in range(offsets[i], offsets[i + 1]):
                column = columns[k]
                if fixed[column] == 0:
                    result += values[k] * x[column]
    elif i == count:
        result = rigid_blocks[0] * x[i]
    else:
        result = rigid_blocks[1] * x[i]
    out[i] = result


@wp.kernel(enable_backward=False)
def rigid_mass_action(rigid_blocks: wp.array[wp.mat33], x: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    i = wp.tid()
    count = x.shape[0] - 2
    out[count + i] = rigid_blocks[i] * x[count + i]


@wp.kernel(enable_backward=False)
def contact_action(
    fixed: wp.array[int],
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
    h: wp.array[float],
    x: wp.array[wp.vec3],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    count = fixed.shape[0]
    relative = x[count] + wp.cross(x[count + 1], levers[i])
    for j in range(4):
        node = nodes[i][j]
        if fixed[node] == 0:
            relative -= bary[i][j] * x[node]
    force = h[0] * matrices[i] * relative
    for j in range(4):
        node = nodes[i][j]
        if fixed[node] == 0:
            wp.atomic_add(out, node, -bary[i][j] * force)
    wp.atomic_add(out, count, force)
    wp.atomic_add(out, count + 1, wp.cross(levers[i], force))


@wp.kernel(enable_backward=False)
def base_diagonal(
    fixed: wp.array[int],
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    rigid_blocks: wp.array[wp.mat33],
    diagonal: wp.array[wp.mat33],
):
    i = wp.tid()
    count = fixed.shape[0]
    value = wp.mat33(0.0)
    if i < count:
        if fixed[i] != 0:
            value = wp.identity(n=3, dtype=float)
        else:
            for k in range(offsets[i], offsets[i + 1]):
                if columns[k] == i:
                    value = values[k]
    elif i == count:
        value = rigid_blocks[0]
    else:
        value = rigid_blocks[1]
    diagonal[i] = value


@wp.kernel(enable_backward=False)
def contact_diagonal(
    fixed: wp.array[int],
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
    h: wp.array[float],
    diagonal: wp.array[wp.mat33],
):
    i = wp.tid()
    matrix = h[0] * matrices[i]
    for j in range(4):
        node = nodes[i][j]
        if fixed[node] == 0:
            wp.atomic_add(diagonal, node, bary[i][j] * bary[i][j] * matrix)
    count = fixed.shape[0]
    wp.atomic_add(diagonal, count, matrix)
    lever = levers[i]
    skew = wp.mat33(0.0, -lever[2], lever[1], lever[2], 0.0, -lever[0], -lever[1], lever[0], 0.0)
    wp.atomic_add(diagonal, count + 1, -skew * matrix * skew)


@wp.kernel(enable_backward=False)
def diagonal_action(
    diagonal: wp.array[wp.mat33],
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    z: wp.array[wp.vec3],
    alpha: float,
    beta: float,
):
    i = wp.tid()
    result = alpha * (diagonal[i] * x[i])
    if beta != 0.0:
        result += beta * y[i]
    z[i] = result


@wp.kernel(enable_backward=False)
def combine(work: wp.array[wp.vec3], y: wp.array[wp.vec3], z: wp.array[wp.vec3], alpha: float, beta: float):
    i = wp.tid()
    result = alpha * work[i]
    if beta != 0.0:
        result += beta * y[i]
    z[i] = result


@wp.kernel(enable_backward=False)
def unpack(residual: wp.array[wp.float64], rhs: wp.array[wp.vec3]):
    i = wp.tid()
    rhs[i] = wp.vec3(float(residual[3 * i]), float(residual[3 * i + 1]), float(residual[3 * i + 2]))


@wp.kernel(enable_backward=False)
def balanced_correction(
    fixed: wp.array[int],
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
    h: wp.array[float],
    x: wp.array[wp.vec3],
    output: wp.array[wp.float64],
):
    i = wp.tid()
    count = fixed.shape[0]
    relative = x[count] + wp.cross(x[count + 1], levers[i])
    for j in range(4):
        node = nodes[i][j]
        if fixed[node] == 0:
            relative -= bary[i][j] * x[node]
    force = h[0] * matrices[i] * relative
    torque = wp.cross(levers[i], force)
    # Fixed nodes have no trial velocity, but their reaction belongs to the
    # interface load. Scatter the same contact correction to every support.
    for j in range(4):
        node = nodes[i][j]
        for component in range(3):
            wp.atomic_add(output, 3 * node + component, wp.float64(bary[i][j]) * wp.float64(force[component]))
    for component in range(3):
        wp.atomic_add(output, 3 * count + component, -wp.float64(force[component]))
        wp.atomic_add(output, 3 * (count + 1) + component, -wp.float64(torque[component]))


@wp.kernel(enable_backward=False)
def invert_diagonal(diagonal: wp.array[wp.mat33], inverse: wp.array[wp.mat33]):
    i = wp.tid()
    inverse[i] = inverse_diagonal_block(diagonal[i])


@wp.kernel(enable_backward=False)
def prepare_rigid(
    poses: wp.array[wp.transform],
    mass: float,
    inertia: wp.mat33,
    damping: wp.array[float],
    h: wp.array[float],
    length: float,
    blocks: wp.array[wp.mat33],
):
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(poses[0]))
    blocks[0] = wp.diag(wp.vec3(mass + h[0] * damping[0], mass + h[0] * damping[1], mass + h[0] * damping[2]))
    native = inertia + h[0] * wp.diag(wp.vec3(damping[3], damping[4], damping[5]))
    blocks[1] = rotation * native * wp.transpose(rotation) / (length * length)


class _ShaftPreconditioner:
    """Approximate the interface inverse without changing its physical residual.

    The model topology, particle constraints and rigid mass properties stay
    fixed for this object's lifetime. Current pose, native damping, timestep,
    material tangent and accepted track values are read on every application.
    Only its own Krylov buffers and cached CUDA graph are mutated.
    """

    @staticmethod
    def supports(coupler):
        model, solver = coupler.model, coupler.fem_solver
        if (
            type(solver) is not SolverFEM
            or getattr(solver.step, "__func__", None) is not SolverFEM.step
            or solver.model is not model
            or solver.differentiable
            or getattr(model, "requires_grad", False)
            or active_tape() is not None
            or solver._streaming is not None
            or solver._component_scaling is not None
            or solver.A_bsr.nrow != model.particle_count
            or solver.A_bsr.ncol != model.particle_count
            or len(coupler.interaction.fixed) != model.particle_count
            or not MuJoCoFreeBody.supports(model, coupler.rigid_solver)
            or coupler.needle_body != 0
            or np.any(coupler.needle_com)
        ):
            return False
        fixed = np.asarray(coupler.interaction.fixed, dtype=bool)
        projected = solver._fixed_particles
        if projected is None:
            if np.any(fixed):
                return False
        elif not np.array_equal(projected.mask.numpy().astype(bool), fixed):
            return False
        mass = float(model.body_mass.numpy()[0])
        inertia = model.body_inertia.numpy()[0].astype(float)
        damping = np.asarray(coupler.rigid_solver.mj_model.dof_damping)
        return bool(
            np.isfinite(mass)
            and mass > 0
            and np.isfinite(inertia).all()
            and np.allclose(inertia, inertia.T, rtol=1e-6, atol=1e-15)
            and np.linalg.eigvalsh(inertia).min() > 0
            and damping.shape == (6,)
            and np.isfinite(damping).all()
            and np.all(damping >= 0)
        )

    def __init__(self, coupler):
        if not self.supports(coupler):
            raise ValueError("Shaft preconditioning requires projected FEM and one zero-COM native free body")
        self.coupler = coupler
        self.device = coupler.model.device
        self.n = coupler.model.particle_count + 2
        self.fixed = wp.array(coupler.interaction.fixed, dtype=int, device=self.device)
        self.rhs = wp.zeros(self.n, dtype=wp.vec3, device=self.device)
        self.z = wp.zeros_like(self.rhs)
        self.work = wp.zeros_like(self.rhs)
        self.diagonal = wp.zeros(self.n, dtype=wp.mat33, device=self.device)
        self.inverse_diagonal = wp.zeros_like(self.diagonal)
        self.rigid_blocks = wp.zeros(2, dtype=wp.mat33, device=self.device)
        self.time_step = wp.zeros(1, dtype=float, device=self.device)
        self.native_damping = wp.zeros(6, dtype=float, device=self.device)
        self._last_dt = None
        self._last_damping = None
        self.matrices = self.levers = None
        self.count = -1
        self.mass = float(coupler.model.body_mass.numpy()[0])
        self.inertia = wp.mat33(coupler.model.body_inertia.numpy()[0])
        self.operator = LinearOperator((self.n, self.n), wp.vec3, self.device, self.matvec)
        self.inverse = LinearOperator((self.n, self.n), wp.vec3, self.device, self.precondition)
        self._conditional = bool(self.device.is_cuda and wp.is_conditional_graph_supported())
        self.linear_solver = cg(
            self.operator,
            self.rhs,
            self.z,
            M=self.inverse,
            # This only approximates a search direction; the outer physical
            # force and ALM acceptance tests retain their original tolerances.
            tol=1e-2
            if getattr(coupler.interaction, "tunnel_alm", None) is not None
            and coupler.interaction.tunnel_alm.compliance_per_m > 0
            else 1e-3,
            atol=0.0,
            maxiter=80,
            check_every=0 if self._conditional else 10,
            use_cuda_graph=self._conditional,
            run=False,
        )
        self._graph = self._graph_key = None
        self._graph_inputs = ()
        self.graph_captures = self.graph_launches = 0
        self.begin_step()

    def begin_step(self):
        """Cache accepted engagement once before an unchanged-history trial sequence."""
        interaction = self.coupler.interaction
        self.active = bool(
            interaction.engaged_track_length > 0
            and (
                getattr(interaction, "tunnel_alm", None) is not None
                or interaction.shaft_stiffness_per_m > 0
                or interaction.shaft_damping_per_m > 0
            )
        )

    def mass_apply(self, x, out):
        # The projected operator's fixed rows are exactly the identity this
        # system needs, and its values stay fresh for the FEM solves; reuse it
        # for the tissue rows. The masked row loop remains the fallback.
        fixed_particles = self.coupler.fem_solver._fixed_particles
        projected = None if fixed_particles is None else fixed_particles._projected
        if projected is not None:
            count = self.n - 2
            fixed_particles.operator.matvec(x[:count], out[:count], out[:count], alpha=1.0, beta=0.0)
            wp.launch(rigid_mass_action, 2, [self.rigid_blocks, x, out], device=self.device)
            return
        matrix = self.coupler.fem_solver.A_bsr
        wp.launch(
            mass_action,
            self.n,
            [self.fixed, matrix.offsets, matrix.columns, matrix.values, self.rigid_blocks, x, out],
            device=self.device,
        )

    def matvec(self, x, y, z, alpha, beta):
        self.mass_apply(x, self.work)
        if self.count:
            interaction = self.coupler.interaction
            wp.launch(
                contact_action,
                self.count,
                [
                    self.fixed,
                    interaction._track_nodes,
                    interaction._track_bary,
                    self.matrices,
                    self.levers,
                    self.time_step,
                    x,
                    self.work,
                ],
                device=self.device,
            )
        wp.launch(combine, self.n, [self.work, y, z, alpha, beta], device=self.device)

    def precondition(self, x, y, z, alpha, beta):
        wp.launch(diagonal_action, self.n, [self.inverse_diagonal, x, y, z, alpha, beta], device=self.device)

    def _solve(self, residual, output):
        self.z.zero_()
        self.linear_solver(b=self.rhs, x=self.z, M=self.inverse)
        wp.copy(output, residual)
        interaction = self.coupler.interaction
        wp.launch(
            balanced_correction,
            self.count,
            [
                self.fixed,
                interaction._track_nodes,
                interaction._track_bary,
                self.matrices,
                self.levers,
                self.time_step,
                self.z,
                output,
            ],
            device=self.device,
        )

    def prepare(self, dt):
        """Stage current coefficients and allocate track scratch without a solve."""
        c, interaction = self.coupler, self.coupler.interaction
        if (
            not self.active
            or interaction.track_count == 0
            or (
                getattr(interaction, "tunnel_alm", None) is None
                and interaction.shaft_stiffness_per_m == 0
                and interaction.shaft_damping_per_m == 0
            )
        ):
            return False
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Preconditioner timestep must be finite and positive")
        damping = np.asarray(c.rigid_solver.mj_model.dof_damping, dtype=np.float32)
        if not np.isfinite(damping).all() or np.any(damping < 0) or active_tape() is not None:
            return False
        if self._last_dt != dt:
            self.time_step.assign(np.array([dt], dtype=np.float32))
            self._last_dt = dt
        if self._last_damping is None or not np.array_equal(damping, self._last_damping):
            self.native_damping.assign(damping)
            self._last_damping = damping.copy()
        if self.count != interaction.track_count:
            self.count = interaction.track_count
            self.matrices = wp.empty(self.count, dtype=wp.mat33, device=self.device)
            self.levers = wp.empty(self.count, dtype=wp.vec3, device=self.device)
        return True

    def apply(self, state, dt, residual, output):
        """Write a correction in the coupler's force/torque-scaled coordinates."""
        if not self.prepare(dt):
            wp.copy(output, residual)
            return
        c, interaction = self.coupler, self.coupler.interaction
        wp.launch(
            prepare_rigid,
            1,
            [
                state.body_q,
                self.mass,
                self.inertia,
                self.native_damping,
                self.time_step,
                c.moment_length,
                self.rigid_blocks,
            ],
            device=self.device,
        )
        alm = getattr(interaction, "tunnel_alm", None)
        if alm is None:
            wp.launch(
                prepare_anchors,
                self.count,
                [
                    interaction._track_nodes,
                    interaction._track_bary,
                    interaction._track_lengths,
                    interaction._track_engaged,
                    state.particle_q,
                    state.particle_qd,
                    interaction._samples,
                    state.body_q,
                    state.body_qd,
                    c.needle_body,
                    wp.vec3(*c.needle_com),
                    float(interaction.radii.max()),
                    interaction.shaft_stiffness_per_m,
                    interaction.shaft_damping_per_m,
                    interaction.friction,
                    interaction.speed_scale,
                    dt,
                    c.moment_length,
                    self.matrices,
                    self.levers,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                prepare_alm_anchors,
                self.count,
                [
                    interaction._track_nodes,
                    interaction._track_bary,
                    interaction._track_lengths,
                    interaction._track_engaged,
                    state.particle_q,
                    state.particle_qd,
                    interaction._samples,
                    state.body_q,
                    state.body_qd,
                    c.needle_body,
                    wp.vec3(*c.needle_com),
                    float(interaction.radii.max()),
                    alm.normal,
                    alm.tangent,
                    alm.penalty_per_m,
                    alm.compliance_per_m,
                    alm.clearance,
                    interaction.friction,
                    dt,
                    c.moment_length,
                    self.matrices,
                    self.levers,
                ],
                device=self.device,
            )
        matrix = c.fem_solver.A_bsr
        wp.launch(
            base_diagonal,
            self.n,
            [self.fixed, matrix.offsets, matrix.columns, matrix.values, self.rigid_blocks, self.diagonal],
            device=self.device,
        )
        wp.launch(
            contact_diagonal,
            self.count,
            [
                self.fixed,
                interaction._track_nodes,
                interaction._track_bary,
                self.matrices,
                self.levers,
                self.time_step,
                self.diagonal,
            ],
            device=self.device,
        )
        wp.launch(invert_diagonal, self.n, [self.diagonal, self.inverse_diagonal], device=self.device)
        wp.launch(unpack, self.n, [residual, self.rhs], device=self.device)
        fixed_particles = c.fem_solver._fixed_particles
        projected = None if fixed_particles is None else fixed_particles._projected
        arrays = (
            matrix.offsets,
            matrix.columns,
            matrix.values,
            interaction._track_nodes,
            interaction._track_bary,
            self.matrices,
            self.levers,
            residual,
            output,
            *(() if projected is None else (projected.values,)),
        )
        key = (self.count, tuple((array.ptr, array.shape, array.dtype) for array in arrays))
        if not self._conditional:
            self._solve(residual, output)
        elif key != self._graph_key:
            self._graph = None
            self._graph_key = key
            self._graph_inputs = arrays
            # A real first application compiles kernels without discarding a
            # solve or changing physical state. The next application captures.
            self._solve(residual, output)
        else:
            if self._graph is None:
                with wp.ScopedCapture(device=self.device) as capture:
                    self._solve(residual, output)
                self._graph = capture.graph
                self.graph_captures += 1
            wp.capture_launch(self._graph)
            self.graph_launches += 1
