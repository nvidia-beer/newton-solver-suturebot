# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen shaft tangent used only to accelerate the exact interface residual.

For residual r in (nodal force, rigid force, torque/L) coordinates, return
r-h J_full.T D J_free (M+h J_free.T D J_free)^-1 r.
Free rows equal M (M+h J_free.T D J_free)^-1 r for an exact inner solve;
the full transpose also returns fixed-support reactions and preserves internal
load balance when the inner solve is approximate. M is projected FEM A plus
native needle effective inertia; rigid velocity coordinates are (v, L*omega).
For articulated tools, a frozen native contact Hessian is condensed to this
needle twist, retaining translation/rotation cross blocks. For each engaged anchor,
D=ell*(h*k+c)*(I-t*t.T) + mu*|normal_force|*v_eps/max(v_eps,|slip|)^2*(t*t.T).
The positive tangential approximation decreases during saturated sliding.
It is not a quadratic majorizer or a full-contact energy model. Surface and
nonsymmetric cross derivatives are omitted; the unchanged outer physical
residual remains the acceptance test.

For the optional tunnel ALM, D uses the exact radial proximal derivative
(frozen shaft projection), plus the tangential derivative on the sticking
branch. On the sliding branch the drag is the Coulomb bound of the wall
reaction of the trial, sign*mu*|R|, so a change dR of that reaction changes
the anchor force by (I+S) dR with the shear S = mu*sign*t*d.T (d the unit
direction of R). The anchors then act as (I+S) D, the operator is not
symmetric, and its solve is BiCGSTAB instead of CG while the track has
samples. Dual values are read without modification; the physical ALM
residual remains responsible for constraint and complementarity convergence.

The anchors of the track (tangent, lever, shear) have one row for every row of
the interaction's track reserve. The kernels over them are launched over that
reserve and read the number of track samples from the interaction's live counts
on the device: a row beyond it is neither read nor written. A graph that holds
these launches therefore serves every later number of samples.
"""

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator, bicgstab, cg

from ....utils.capture import PauseGC, ScopedCapture
from ...fem.differentiation import active_tape
from ...fem.kernels import inverse_diagonal_block
from ...fem.solver import SolverFEM
from ...mujoco.mujoco_free_body import MuJoCoFreeBody
from ..interface.insertion import LIVE_TRACK, _shaft_projection


@wp.kernel(enable_backward=False)
def prepare_anchors(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    lengths: wp.array[float],
    engaged: wp.array[int],
    live: wp.array[int],
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
    h: wp.array[float],
    moment_length: float,
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
):
    i = wp.tid()
    if i >= live[LIVE_TRACK]:
        return
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
        matrix = lengths[i] * (h[0] * stiffness + damping) * projection
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
    live: wp.array[int],
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
    h: wp.array[float],
    moment_length: float,
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
    shears: wp.array[wp.mat33],
):
    i = wp.tid()
    if i >= live[LIVE_TRACK]:
        return
    matrix = wp.mat33(0.0)
    lever = wp.vec3(0.0)
    shear = wp.mat33(0.0)
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
            matrix = lengths[i] * h[0] * derivative
            normal_magnitude = factor * (magnitude - threshold)
            bound = friction * normal_magnitude
            tangent_trial = tangential[i] + penalty * h[0] * slip
            if wp.abs(tangent_trial) < bound:
                matrix += lengths[i] * penalty * h[0] * wp.outer(tangent, tangent)
            else:
                # Sliding: the drag is the Coulomb bound of this reaction, sign * friction * |reaction|.
                # A change dR of the reaction changes it by friction * sign * (direction . dR) along the shaft.
                shear = friction * wp.sign(tangent_trial) * wp.outer(tangent, direction)
    matrices[i] = matrix
    levers[i] = lever
    shears[i] = shear


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
        result = rigid_blocks[0] * x[i] + rigid_blocks[2] * x[count + 1]
    else:
        result = rigid_blocks[1] * x[i] + rigid_blocks[3] * x[count]
    out[i] = result


@wp.kernel(enable_backward=False)
def rigid_mass_action(rigid_blocks: wp.array[wp.mat33], x: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    i = wp.tid()
    count = x.shape[0] - 2
    out[count + i] = rigid_blocks[i] * x[count + i] + rigid_blocks[2 + i] * x[count + 1 - i]


@wp.kernel(enable_backward=False)
def contact_action(
    fixed: wp.array[int],
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    live: wp.array[int],
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
    shears: wp.array[wp.mat33],
    h: wp.array[float],
    x: wp.array[wp.vec3],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    if i >= live[LIVE_TRACK]:
        return
    count = fixed.shape[0]
    relative = x[count] + wp.cross(x[count + 1], levers[i])
    for j in range(4):
        node = nodes[i][j]
        if fixed[node] == 0:
            relative -= bary[i][j] * x[node]
    force = h[0] * matrices[i] * relative
    # A sliding tunnel anchor adds the change of its drag with the wall reaction; the shear is zero otherwise.
    force += shears[i] * force
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
    live: wp.array[int],
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
    shears: wp.array[wp.mat33],
    h: wp.array[float],
    diagonal: wp.array[wp.mat33],
):
    i = wp.tid()
    if i >= live[LIVE_TRACK]:
        return
    matrix = h[0] * matrices[i]
    matrix += shears[i] * matrix
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
    live: wp.array[int],
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
    shears: wp.array[wp.mat33],
    h: wp.array[float],
    x: wp.array[wp.vec3],
    output: wp.array[wp.float64],
):
    i = wp.tid()
    if i >= live[LIVE_TRACK]:
        return
    count = fixed.shape[0]
    relative = x[count] + wp.cross(x[count + 1], levers[i])
    for j in range(4):
        node = nodes[i][j]
        if fixed[node] == 0:
            relative -= bary[i][j] * x[node]
    force = h[0] * matrices[i] * relative
    force += shears[i] * force
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


def _native_needle_mass(model, data, body, moment_length, dt, scratch=None):
    """Condense a frozen native contact Hessian to the needle COM twist.

    H = M + h*diag(damping) + J.T*D*J is an SPD search-direction metric.
    D is the native piecewise constraint Hessian (including elliptic cones),
    evaluated on private data. B maps joint rates to (v_COM, L*omega), and
    (B H^-1 B.T)^-1 is the corresponding six-dimensional effective mass.
    Contacts, native integration and the outer physical residual are unchanged.
    This freezes the acceleration-space tangent; it is not the exact derivative
    of the complete subcycled transition or of changing contact geometry.
    """
    import mujoco

    if not np.isfinite([moment_length, dt]).all() or moment_length <= 0 or dt <= 0:
        raise ValueError("Positive finite moment length and timestep are required")
    if not np.isfinite(model.dof_damping).all() or np.any(model.dof_damping < 0):
        raise ValueError("All native joint damping must be finite and nonnegative")
    scratch = mujoco.MjData(model) if scratch is None else scratch
    mujoco.mj_copyData(scratch, model, data)
    mass = np.empty((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, scratch, mass)
    mass.flat[:: model.nv + 1] += dt * model.dof_damping
    count = scratch.nefc
    if count:
        jacobian = np.zeros((count, model.nv))
        if mujoco.mj_isSparse(model):
            for row in range(count):
                begin = scratch.efc_J_rowadr[row]
                end = begin + scratch.efc_J_rownnz[row]
                jacobian[row, scratch.efc_J_colind[begin:end]] = scratch.efc_J[begin:end]
        else:
            jacobian[:] = scratch.efc_J.reshape(count, model.nv)
        jar = jacobian @ scratch.qacc - scratch.efc_aref
        mujoco.mj_constraintUpdate(model, scratch, jar, None, 1)
        quadratic = scratch.efc_state == int(mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC)
        mass += jacobian[quadratic].T @ (scratch.efc_D[quadratic, None] * jacobian[quadratic])
        # Elliptic contact rows share one cone Hessian, not independent scalar
        # efc_D entries. Reading it after constraintUpdate avoids stale storage.
        for contact in scratch.contact:
            row, dimension = contact.efc_address, contact.dim
            if row < 0 or scratch.efc_state[row] != int(mujoco.mjtConstraintState.mjCNSTRSTATE_CONE):
                continue
            tangent = contact.H.reshape(6, 6)[:dimension, :dimension]
            block = jacobian[row : row + dimension]
            mass += block.T @ tangent @ block
    body_jacobian = np.zeros((6, model.nv))
    mujoco.mj_jacBodyCom(model, scratch, body_jacobian[:3], body_jacobian[3:], body)
    body_jacobian[3:] *= moment_length
    response = body_jacobian @ np.linalg.solve(mass, body_jacobian.T)
    response = 0.5 * (response + response.T)
    result = np.linalg.solve(response, np.eye(6))
    result = 0.5 * (result + result.T)
    if not np.isfinite(result).all() or np.linalg.eigvalsh(result).min() <= 0:
        raise ValueError("Native needle metric must be finite and positive definite")
    return result


def _rigid_blocks_from_mass(matrix):
    """Validate the matrix actually used by float32 CG, including cast rounding."""
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.asarray(matrix, dtype=np.float32)
    if value.shape != (6, 6) or not np.isfinite(value).all():
        raise ValueError("The float32 native needle metric must be finite and 6 by 6")
    # Positive double precision eigenvalues can round to zero or negative in
    # an ill-conditioned constrained metric. Never feed that matrix to CG.
    np.linalg.cholesky(value.astype(float))
    return np.asarray([value[:3, :3], value[3:, 3:], value[:3, 3:], value[3:, :3]])


def _supported_rigid_needle(coupler):
    """Accept the legacy body or a known subcycled world-free needle at body 0."""
    from ...mujoco.experimental_rigid_backend import supported_subcycled

    model, solver = coupler.model, coupler.rigid_solver
    if MuJoCoFreeBody.supports(model, solver):
        return True
    if not supported_subcycled(solver) or solver.model is not model or model.world_count != 1:
        return False
    import mujoco
    import newton

    native = solver.mj_model
    identity = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    return bool(
        model.joint_type.numpy()[0] == int(newton.JointType.FREE)
        and model.joint_parent.numpy()[0] == -1
        and model.joint_child.numpy()[0] == 0
        and model.body_flags.numpy()[0] == int(newton.BodyFlags.DYNAMIC)
        and np.array_equal(model.joint_X_p.numpy()[0], identity)
        and np.array_equal(model.joint_X_c.numpy()[0], identity)
        and not np.any(model.body_com.numpy()[0])
        and native.jnt_type[0] == int(mujoco.mjtJoint.mjJNT_FREE)
        and native.jnt_bodyid[0] == 1
        and native.body_parentid[1] == 0
        and native.jnt_dofadr[0] == 0
        and native.jnt_qposadr[0] == 0
    )


# Devices on which load_kernels() has created the kernels of the two solvers.
_KERNEL_DEVICES = set()


def load_kernels(device):
    """Create and load for ``device`` the kernels of a correction's solves, without a correction.

    A correction creates the kernels of its two Krylov solvers and loads their modules where
    it is made (``_ShaftPreconditioner._make_linear_solver``), with the coupled step. A scene
    that makes its coupled step after its start-up, when a path is planned, calls this at
    start-up instead. It loads the kernel module of this file by name and issues, for CG and
    for BiCGSTAB over three-vectors, the same solve of an identity system into a capture that
    is never launched: Warp creates the kernels of a solver for the type of the unknowns where
    they are first launched, and that solve records every launch of the solver. Nothing runs
    and nothing is kept. The kernels are created once for a device: a later call loads the
    module of this file, which is loaded already, and nothing else.

    Without conditional CUDA graphs a correction creates its kernels at its first solve, as
    before, and this loads the module of this file alone. Nothing is done on a CPU device.
    """
    device = wp.get_device(device)
    if not device.is_cuda:
        return
    wp.load_module(__name__, device=device, block_dim=256)
    if device.alias in _KERNEL_DEVICES or not wp.is_conditional_graph_supported():
        return
    rhs = wp.zeros(1, dtype=wp.vec3, device=device)
    z = wp.zeros_like(rhs)

    def identity(x, y, out, alpha, beta):
        wp.launch(combine, rhs.shape[0], [x, y, out, alpha, beta], device=device)

    operator = LinearOperator((rhs.shape[0], rhs.shape[0]), wp.vec3, device, identity)
    for method in (cg, bicgstab):
        # The arguments of _make_linear_solver on a device with conditional graphs.
        solver = method(
            operator, rhs, z, M=operator, tol=1e-3, atol=0.0, maxiter=80, check_every=0, use_cuda_graph=True, run=False
        )
        with ScopedCapture(device=device, force_module_load=False):
            solver(A=operator, b=rhs, x=z, M=operator)
    _KERNEL_DEVICES.add(device.alias)


class _ShaftPreconditioner:
    """Approximate the interface inverse without changing its physical residual.

    The model topology, particle constraints and rigid mass properties stay
    fixed for this object's lifetime. Current pose, native damping, timestep,
    material tangent and accepted track values are read on every application.
    Only its own Krylov buffers, private native scratch and cached CUDA graph are mutated.
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
            or not _supported_rigid_needle(coupler)
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
        damping = np.asarray(coupler.rigid_solver.mj_model.dof_damping[:6])
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
            raise ValueError("Shaft preconditioning requires projected FEM and a supported zero-COM world-free needle")
        self.coupler = coupler
        self.device = coupler.model.device
        self.n = coupler.model.particle_count + 2
        self.fixed = wp.array(coupler.interaction.fixed, dtype=int, device=self.device)
        self.rhs = wp.zeros(self.n, dtype=wp.vec3, device=self.device)
        self.z = wp.zeros_like(self.rhs)
        self.work = wp.zeros_like(self.rhs)
        self.diagonal = wp.zeros(self.n, dtype=wp.mat33, device=self.device)
        self.inverse_diagonal = wp.zeros_like(self.diagonal)
        self.rigid_blocks = wp.zeros(4, dtype=wp.mat33, device=self.device)
        self._articulated = coupler.model.body_count > 1
        self._native_mass_ready = False
        self._native_scratch = None
        self.native_mass_fallbacks = 0
        if self._articulated:
            import mujoco

            self._native_scratch = mujoco.MjData(coupler.rigid_solver.mj_model)
        # The time step of the coupled step as device data: the array that the FEM
        # step, the interface forces and the rigid contact metric of that step read.
        step_time = getattr(coupler, "step_time", None)
        self.step_time = coupler.fem_solver.step_time if step_time is None else step_time
        self.time_step = self.step_time.seconds
        self.native_damping = wp.zeros(6, dtype=float, device=self.device)
        self._last_dt = None
        self._last_damping = None
        # Anchors of the track, one row per row of the track reserve (capacity, the launch size),
        # and the host's number of track samples (count); the kernels read theirs on the device.
        self.matrices = self.levers = self.shears = self._has_track = None
        self.capacity = self.count = 0
        self.storage_growths = 0
        self._allocate_track_storage()
        self.mass = float(coupler.model.body_mass.numpy()[0])
        self.inertia = wp.mat33(coupler.model.body_inertia.numpy()[0])
        self.operator = LinearOperator((self.n, self.n), wp.vec3, self.device, self._solver_matvec)
        self.inverse = LinearOperator((self.n, self.n), wp.vec3, self.device, self.precondition)
        self._conditional = bool(self.device.is_cuda and wp.is_conditional_graph_supported())
        # This only approximates a search direction; the outer physical
        # force and ALM acceptance tests retain their original tolerances.
        self._solver_tolerance = (
            1e-2
            if getattr(coupler.interaction, "tunnel_alm", None) is not None
            and coupler.interaction.tunnel_alm.compliance_per_m > 0
            else 1e-3
        )
        # The solver follows the anchor law, not the friction coefficient, which may change
        # between corrections: a friction tunnel shears its sliding anchors (see the module text).
        # Both solvers are made here, with their kernels, for every correction that holds track
        # anchors, whatever law the track follows now: no step and no correction makes one.
        self.nonsymmetric = self._friction_tunnel() is not None
        self._symmetric_solver = self._make_linear_solver(cg)
        self._sheared_solver = None if self.shears is None else self._make_linear_solver(bicgstab)
        self.linear_solver = self._anchor_solver()
        # The solve graph in use, and the graph kept for each of the two forms of the solve
        # (without and with an attached start guess): {form: (key, graph, arrays)}.
        self._graph = self._graph_key = None
        self._graph_inputs = ()
        self._graphs = {}
        self.graph_captures = self.graph_launches = 0
        # The start guess of the solve. ``_guess_storage`` is the one object of this correction,
        # made when a response-basis core first exists (``start_guess``); ``_woodbury_guess`` is
        # that object once a solve is to start from it, and None before.
        self._guess_storage = None
        self._woodbury_guess = None
        self._input_preparation = None
        self.capture_preparation = True
        # Value of ``active`` for the current step: None until its first reader in the step.
        self._active = None
        # True while a caller captures :meth:`apply` behind a device condition of its own (the flag
        # of the track in the free-needle trial graph): the host gate of :meth:`prepare` is then
        # that condition.
        self._device_gated = False
        self.begin_step()

    def _friction_tunnel(self):
        """The tunnel model of the track, whose sliding anchors carry a shear; None for every other anchor law."""
        return getattr(self.coupler.interaction, "tunnel_alm", None)

    def _allocate_track_storage(self):
        """Make the anchors of the track at the track reserve of the interaction, zero in every row.

        The anchor kernels write the rows of the track samples in every correction, and no kernel
        reads or writes a row beyond them. A metric without track anchors overrides this with nothing.
        """
        interaction = self.coupler.interaction
        self.capacity = interaction.track_capacity
        self.count = interaction.track_count
        self.matrices = wp.zeros(self.capacity, dtype=wp.mat33, device=self.device)
        self.levers = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        # Written by the tunnel anchors alone: zero for every other anchor law.
        self.shears = wp.zeros(self.capacity, dtype=wp.mat33, device=self.device)
        # The live track count as the condition of a device branch: nonzero while the track has samples.
        self._has_track = interaction._live[LIVE_TRACK : LIVE_TRACK + 1]

    def _sync_track_storage(self):
        """Take the host's number of track samples and follow a growth of the track storage.

        The interaction doubles its track storage when its reserve was too small. The anchors are
        then made anew at the new size, outside a capture, and the growth is counted. No row is
        copied: the anchor kernels write every row of a track sample in each correction.
        """
        interaction = self.coupler.interaction
        self.count = interaction.track_count
        if interaction.track_capacity != self.capacity and not (self.device.is_cuda and self.device.is_capturing):
            self._allocate_track_storage()
            self.storage_growths += 1

    def storage_key(self):
        """Launch size and storage of what a captured correction addresses on this object.

        The anchors, the time step, the native damping and, when the tissue solver projects its
        operator, the values of that operator. It does not change with the number of track samples.
        """
        fixed_particles = self.coupler.fem_solver._fixed_particles
        projected = None if fixed_particles is None else fixed_particles._projected
        arrays = (self.matrices, self.levers, self.shears, self.time_step, self.native_damping)
        if projected is not None:
            arrays += (projected.values,)
        return (self.capacity, tuple((array.ptr, array.shape, array.dtype) for array in arrays))

    def _anchor_solver(self):
        """Krylov solver of the current anchors.

        BiCGSTAB while a friction tunnel has track samples: their shear makes the operator
        nonsymmetric. CG for every other anchor law, and for a track without samples, where the
        operator holds no track anchor and is symmetric (the system before the first puncture).
        """
        return self._sheared_solver if self.nonsymmetric and self.count else self._symmetric_solver

    def _make_linear_solver(self, method):
        """Warp's stock solver ``method`` (``cg`` or ``bicgstab``) over this operator, with its kernels loaded."""
        solver = method(
            self.operator,
            self.rhs,
            self.z,
            M=self.inverse,
            tol=self._solver_tolerance,
            atol=0.0,
            maxiter=80,
            check_every=0 if self._conditional else 10,
            use_cuda_graph=self._conditional,
            run=False,
        )
        if self._conditional:
            # Warp creates a solver's kernels for the type of the unknowns where they are first
            # launched, which would be the first correction of a run, and every kernel it creates
            # gives the module of all its Krylov kernels a new hash. One solve of an identity
            # system, issued into a capture that is never launched, records every launch of the
            # solver: its kernels are created and their modules loaded here, and nothing runs.
            identity = LinearOperator((self.n, self.n), wp.vec3, self.device, self._identity)
            with ScopedCapture(device=self.device, force_module_load=False):
                solver(A=identity, b=self.rhs, x=self.z, M=identity)
        return solver

    def _identity(self, x, y, z, alpha, beta):
        wp.launch(combine, self.n, [x, y, z, alpha, beta], device=self.device)

    def invalidate_preparation(self):
        """Retire preparation owners between physical steps or before reset."""
        for helper in (
            self._input_preparation,
            getattr(self._woodbury_guess, "_packet_preparation", None),
        ):
            if helper is not None:
                helper.replay.invalidate()

    def start_guess(self, manager):
        """The start guess of this correction for the response-basis ``manager``; None while it has no core.

        One object for the lifetime of the correction: it is made the first time a core
        exists, with its scratch and kernels, and every later call returns it.
        """
        if self._guess_storage is None and manager is not None and manager.core is not None:
            from .woodbury_guess import WoodburyInitialGuess

            self._guess_storage = WoodburyInitialGuess(self)
        return self._guess_storage

    @property
    def active(self):
        """Whether the accepted track transmits a shaft interaction in the current step.

        The first reader after :meth:`begin_step` evaluates it, and the value is held for the
        trials of the step, which do not change the history: trials are implicit, and the
        engagement is written where a step is accepted. A step in which nothing reads the flag
        evaluates nothing; that is a step of a captured trial loop whose correction is the same
        graph for both values. A track without a shaft law is inactive, which the host knows
        without the engaged length.

        The evaluation also asks the interaction to read the engaged length where this step is
        accepted (``request_engaged_track_length``). A path that reads the flag in every step
        is then answered from that read and reads the device itself only in the first such
        step. A reader that no reader follows in the next step evaluates through
        :meth:`read_active` without that request. Evaluate the flag outside a CUDA graph
        capture, because it can read the device.

        An assigned value holds until the next :meth:`begin_step`.
        """
        return self.read_active(request=True)

    def read_active(self, *, request):
        """:attr:`active`, evaluated here if the step holds no value yet, with or without the request.

        ``request`` says whether that evaluation asks the acceptance of this step for the
        engaged length of the step after it. Without the request the evaluation reads the
        engaged length from the device, once, and the acceptance of this step reads nothing.
        That is the reading of host-controlled trials that take over from device trials
        which do not evaluate the flag: the step after them does not ask, so a read in the
        acceptance would serve no one. The interaction keeps the length that was read until
        the engagement is written, which answers a second attempt of a step whose first
        attempt was rejected. A value that the step already holds is returned as it is, with
        no read and no request.
        """
        if self._active is None:
            interaction = self.coupler.interaction
            law = (
                getattr(interaction, "tunnel_alm", None) is not None
                or interaction.shaft_stiffness_per_m > 0
                or interaction.shaft_damping_per_m > 0
            )
            if not law:
                self._active = False
            elif request:
                interaction.request_engaged_track_length()
                self._active = bool(interaction.engaged_track_length > 0)
            else:
                self._active = bool(interaction.keep_engaged_track_length() > 0)
        return self._active

    @active.setter
    def active(self, value):
        self._active = bool(value)

    def begin_step(self):
        """Begin an unchanged-history trial sequence; the engagement is read by the first reader of ``active``."""
        self._native_mass_ready = False
        nonsymmetric = self._friction_tunnel() is not None
        if nonsymmetric != self.nonsymmetric:
            # A scene creates its tunnel model with the interaction, so this branch is not
            # entered. A model attached or removed afterwards changes the solver of the track,
            # and both solvers exist since construction: nothing is made here. A removed model
            # leaves shears that the other anchor kernels do not write. Neither solver and no
            # anchor array is released, so a graph captured before still addresses live storage,
            # and the trial loops capture anew: the identity of the model is an entry of their
            # signature.
            self.nonsymmetric = nonsymmetric
            if not nonsymmetric and self.shears is not None:
                self.shears.zero_()
            self.linear_solver = self._anchor_solver()
            self._graph = self._graph_key = None
            self._graphs.clear()
        # Nothing is read here: ``active`` is evaluated by its first reader in the step.
        self._active = None

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
        interaction = self.coupler.interaction
        # Over the track reserve: the threads beyond the live track count add nothing.
        wp.launch(
            contact_action,
            self.capacity,
            [
                self.fixed,
                interaction._track_nodes,
                interaction._track_bary,
                interaction._live,
                self.matrices,
                self.levers,
                self.shears,
                self.time_step,
                x,
                self.work,
            ],
            device=self.device,
        )
        self._augment_contact_action(x)
        wp.launch(combine, self.n, [self.work, y, z, alpha, beta], device=self.device)

    def _solver_matvec(self, x, y, z, alpha, beta):
        """Operator of the Krylov solve; a certified start already owns its first residual."""
        guess = self._woodbury_guess
        if guess is None or not guess.first_residual(x, y, z, alpha, beta):
            self.matvec(x, y, z, alpha, beta)

    def _augment_contact_action(self, x):
        """Optional extra contact action added to self.work."""

    def _augment_correction(self, velocity, output):
        """Optional full-transpose correction after the shaft correction."""

    def precondition(self, x, y, z, alpha, beta):
        wp.launch(diagonal_action, self.n, [self.inverse_diagonal, x, y, z, alpha, beta], device=self.device)

    def _krylov(self, solver):
        """One Krylov solve of the prepared system by ``solver``; a start guess then counts its iterations."""
        self.linear_solver = solver
        with PauseGC():
            self.linear_solver(b=self.rhs, x=self.z, M=self.inverse)
        if self._woodbury_guess is not None:
            self._woodbury_guess.after_solve()

    def _solve(self, residual, output):
        if self._woodbury_guess is None:
            self.z.zero_()
        else:
            self._woodbury_guess.initialize()
        if self.nonsymmetric and self.device.is_capturing:
            # A captured solve of a friction tunnel is launched at every later number of track
            # samples, so the device takes the solver by the live count of each launch: CG for
            # a track without samples, BiCGSTAB from the first sample on.
            wp.capture_if(
                self._has_track,
                on_true=lambda: self._krylov(self._sheared_solver),
                on_false=lambda: self._krylov(self._symmetric_solver),
            )
        else:
            self._krylov(self._anchor_solver())
        # The solver of the anchors that the host counts: a track without samples is solved by CG.
        self.linear_solver = self._anchor_solver()
        wp.copy(output, residual)
        interaction = self.coupler.interaction
        wp.launch(
            balanced_correction,
            self.capacity,
            [
                self.fixed,
                interaction._track_nodes,
                interaction._track_bary,
                interaction._live,
                self.matrices,
                self.levers,
                self.shears,
                self.time_step,
                self.z,
                output,
            ],
            device=self.device,
        )

        self._augment_correction(self.z, output)

    def prepare(self, dt):
        """Stage current coefficients without a solve; False keeps the residual as the correction.

        The host gate (a track sample, a shaft law, an engaged track) is skipped while
        ``_device_gated`` is set: the caller then captures the correction behind a device
        condition that stands for the gate, for launches at every later state of the track.
        """
        interaction = self.coupler.interaction
        # What the host knows decides first: ``active`` can read the engaged length from the device.
        if not self._device_gated and (
            interaction.track_count == 0
            or (
                getattr(interaction, "tunnel_alm", None) is None
                and interaction.shaft_stiffness_per_m == 0
                and interaction.shaft_damping_per_m == 0
            )
            or not self.active
        ):
            return False
        return self.stage(dt, tape=True)

    def stage(self, dt, *, tape):
        """Write the time step and the native damping where they changed, and follow the track storage.

        False for an invalid native damping and, with ``tape``, under an active tape: nothing is
        written then. Nothing is allocated unless the track storage of the interaction grew.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Preconditioner timestep must be finite and positive")
        damping = np.asarray(self.coupler.rigid_solver.mj_model.dof_damping[:6], dtype=np.float32)
        if not np.isfinite(damping).all() or np.any(damping < 0) or (tape and active_tape() is not None):
            return False
        self.step_time.set(dt)
        self._last_dt = dt
        if self._last_damping is None or not np.array_equal(damping, self._last_damping):
            self.native_damping.assign(damping)
            self._last_damping = damping.copy()
        self._sync_track_storage()
        return True

    def _augment_diagonal(self):
        """Optional extra positive contact contributions before inversion."""

    def _extra_graph_arrays(self):
        return ()

    def _prepare_inputs(self, state, dt, residual):
        from .contact_preparation import ContactPreparation

        if self._input_preparation is None:
            self._input_preparation = ContactPreparation(self)
        self._input_preparation.run(state, dt, residual)

    def _enqueue_inputs(self, state, dt, residual):
        """Enqueue the unchanged current-state contact/diagonal preparation.

        The launches over the track have the size of the track reserve and read the live track
        count on the device, so a capture of them does not follow the number of track samples.
        """
        c, interaction = self.coupler, self.coupler.interaction
        # Every launch below reads the time step from self.time_step. This is a
        # comparison after prepare(dt); a different value inside a capture raises.
        self.step_time.set(dt)
        if not self._articulated:
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
        from ..interface.shaft_projection import prepare_projection

        projection = prepare_projection(
            interaction,
            state,
            c.needle_body,
            float(interaction.radii.max()),
            operands=(residual, self.matrices, self.levers),
        )
        anchor_kernel, alm_kernel, projection_extra = prepare_anchors, prepare_alm_anchors, []
        if projection is not None:
            from ..interface.shaft_projected_kernels import prepare_alm_anchors_projected, prepare_anchors_projected

            anchor_kernel, alm_kernel = prepare_anchors_projected, prepare_alm_anchors_projected
            projection_extra = [projection.result]
        alm = getattr(interaction, "tunnel_alm", None)
        if alm is None:
            wp.launch(
                anchor_kernel,
                self.capacity,
                [
                    interaction._track_nodes,
                    interaction._track_bary,
                    interaction._track_lengths,
                    interaction._track_engaged,
                    interaction._live,
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
                    self.time_step,
                    c.moment_length,
                    self.matrices,
                    self.levers,
                ]
                + projection_extra,
                device=self.device,
            )
        else:
            wp.launch(
                alm_kernel,
                self.capacity,
                [
                    interaction._track_nodes,
                    interaction._track_bary,
                    interaction._track_lengths,
                    interaction._track_engaged,
                    interaction._live,
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
                    self.time_step,
                    c.moment_length,
                    self.matrices,
                    self.levers,
                    self.shears,
                ]
                + projection_extra,
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
            self.capacity,
            [
                self.fixed,
                interaction._track_nodes,
                interaction._track_bary,
                interaction._live,
                self.matrices,
                self.levers,
                self.shears,
                self.time_step,
                self.diagonal,
            ],
            device=self.device,
        )
        self._augment_diagonal()
        wp.launch(invert_diagonal, self.n, [self.diagonal, self.inverse_diagonal], device=self.device)
        wp.launch(unpack, self.n, [residual, self.rhs], device=self.device)

    def prepare_rigid_metric(self, dt):
        """Prepare the same frozen metric before eager or captured trials."""
        c = self.coupler
        if self._articulated and not self._native_mass_ready:
            try:
                matrix = _native_needle_mass(
                    c.rigid_solver.mj_model, c.rigid_solver.mj_data, 1, c.moment_length, dt, self._native_scratch
                )
                blocks = _rigid_blocks_from_mass(matrix)
                self.rigid_blocks.assign(blocks)
                self._native_mass_ready = True
            except (ValueError, np.linalg.LinAlgError):
                # A singular/nonfinite approximate metric cannot modify or
                # bypass the unchanged physical acceptance test.
                self.native_mass_fallbacks += 1
                return False
        return True

    def apply(self, state, dt, residual, output):
        """Write a correction in the coupler's force/torque-scaled coordinates."""
        if not self.prepare(dt):
            wp.copy(output, residual)
            return
        if not self.prepare_rigid_metric(dt):
            wp.copy(output, residual)
            return
        self._prepare_inputs(state, dt, residual)
        manager = getattr(self.coupler.fem_solver, "_affine_predictor", None)
        if self._woodbury_guess is None and manager is not None and manager.active and manager.core is not None:
            self._woodbury_guess = self.start_guess(manager)
        if self._woodbury_guess is not None:
            self._woodbury_guess.prepare(manager)
        self._solve_graph(residual, output)

    def capture_host_graphs(self, state, dt, residual, output):
        """Capture and launch, ahead of the first step, the graphs of a host-controlled correction.

        For start-up, inside an attempt of the coupler that ends like a rejected step, with a
        zero ``residual`` and the arrays that the host-controlled trials pass to :meth:`apply`.
        It does not go through :meth:`apply`, which returns without a track sample. Issued
        here, each twice (as written for a new key, then captured and launched, by the rule
        of its owner):

        - the contact preparation (``_prepare_inputs``);
        - the solve (``_solve_graph``) without a start guess;
        - with a start guess (a response-basis core exists): the preparation of its host
          packet, and the solve that starts from the guess. The guess is bound to the core
          for this, and is attached only for these passes.

        With no track sample and a zero residual the passes write scratch of this correction
        and ``output`` only: the anchor kernels return at the live track count, and a zero
        right-hand side ends the solve before its first iteration. ``graph_captures`` and the
        ``captures`` of the two preparations count the captures. What the passes counted as
        work is given back: ``graph_launches``, ``native_mass_fallbacks``, the ``eager`` and
        ``launches`` of the two preparations and the counters of the guess, on the host and on
        the device. The native metric is prepared again by the first correction of a step.

        Returns:
            False, with nothing issued, where this correction captures no graph (no
            conditional CUDA graphs) and for a native damping that is refused.
        """
        if not self._conditional:
            return False
        last_dt = self._last_dt
        if not self.stage(dt, tape=False):
            return False
        manager = getattr(self.coupler.fem_solver, "_affine_predictor", None)
        guess = self.start_guess(manager) if manager is not None and manager.core is not None else None
        attached = self._woodbury_guess
        launches, fallbacks = self.graph_launches, self.native_mass_fallbacks
        kept = None
        if guess is not None:
            kept = (
                {name: getattr(guess, name) for name in ("preparations", "ready", "fallbacks", "last_reason")},
                dict(guess.fallback_reasons),
                dict(guess.timings),
                wp.clone(guess.counters),
            )
        replays = {}

        def counted(replay):
            # The work counters of a preparation before its first pass here.
            replays.setdefault(id(replay), (replay, replay.eager, replay.launches))

        try:
            # The result is not used: a refused metric leaves the blocks of the last one.
            self.prepare_rigid_metric(dt)
            from .contact_preparation import ContactPreparation

            if self._input_preparation is None:
                self._input_preparation = ContactPreparation(self)
            counted(self._input_preparation.replay)
            for _ in range(2):
                self._prepare_inputs(state, dt, residual)
            self._woodbury_guess = None
            for _ in range(2):
                self._solve_graph(residual, output)
            if guess is not None:
                from .contact_preparation import PacketPreparation

                # The free nodes of the core are read here, once per FEM structure.
                guess._bind_structure(manager)
                if guess._packet_preparation is None:
                    guess._packet_preparation = PacketPreparation(guess)
                counted(guess._packet_preparation.replay)
                self._woodbury_guess = guess
                # The flags of a start: a basis that may not be used gives the zero start.
                guess.prepare(manager)
                surface = getattr(self, "surface", None)
                count = guess.no_surface if surface is None else surface.anchors.count
                valid = guess.no_surface if surface is None else surface.valid
                for _ in range(2):
                    guess._packet_preparation.run(count, valid)
                for _ in range(2):
                    self._solve_graph(residual, output)
        finally:
            self._woodbury_guess = attached
            self.graph_launches, self.native_mass_fallbacks, self._last_dt = launches, fallbacks, last_dt
            self._native_mass_ready = False
            for replay, eager, replayed in replays.values():
                replay.eager, replay.launches = eager, replayed
            if kept is not None:
                host, reasons, timings, counters = kept
                for name, value in host.items():
                    setattr(guess, name, value)
                guess.fallback_reasons, guess.timings = reasons, timings
                wp.copy(guess.counters, counters)
        return True

    def _solve_graph(self, residual, output):
        """Issue the solve of the prepared inputs: as written at the first use of a key, then captured, then launched.

        The key is the storage key of this object, whether a start guess is attached, and the
        storage of the other arrays that the solve addresses. It does not hold the number of
        track samples: the kernels read that on the device. One graph is kept for each of the
        two forms of the solve, without and with a start guess, so going from one form to the
        other and back launches the graph that was captured for it.
        """
        c, interaction = self.coupler, self.coupler.interaction
        if not self._conditional:
            self._solve(residual, output)
            return
        matrix = c.fem_solver.A_bsr
        fixed_particles = c.fem_solver._fixed_particles
        projected = None if fixed_particles is None else fixed_particles._projected
        guess = self._woodbury_guess
        arrays = (
            matrix.offsets,
            matrix.columns,
            matrix.values,
            interaction._track_nodes,
            interaction._track_bary,
            interaction._live,
            self.matrices,
            self.levers,
            self.shears,
            self.time_step,
            residual,
            output,
            *self._extra_graph_arrays(),
            *(() if guess is None else guess.owners),
            *(() if projected is None else (projected.values,)),
        )
        form = guess is not None
        key = (self.storage_key(), form, tuple((array.ptr, array.shape, array.dtype) for array in arrays))
        if key != self._graph_key:
            kept = None if self._graph_key is None else self._graphs.get(form)
            if kept is not None and kept[0] == key:
                # The other form was in use last: this one's graph is still the one for this key.
                self._graph_key, self._graph, self._graph_inputs = kept
            else:
                self._graph = None
                self._graph_key = key
                self._graph_inputs = arrays
                self._graphs[form] = (key, None, arrays)
                # A real first application compiles kernels without discarding a
                # solve or changing physical state. The next application captures.
                self._solve(residual, output)
                return
        if self._graph is None:
            with ScopedCapture(device=self.device) as capture:
                self._solve(residual, output)
            self._graph = capture.graph
            self._graphs[form] = (key, self._graph, arrays)
            self.graph_captures += 1
        wp.capture_launch(self._graph)
        self.graph_launches += 1
