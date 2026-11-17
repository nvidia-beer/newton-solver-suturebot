# SPDX-License-Identifier: Apache-2.0
"""Device trial integration for the supported MuJoCo 3.17.0 free-body case.

This is a specialization of the native implicitfast equations, not a generic
MuJoCo replacement. Trials do not advance native time or integration history.
The coupling transaction must audit/advance native state once on acceptance.
"""

import mujoco
import numpy as np
import warp as wp

from ...utils.step_time import StepTime
from .mujoco_free_body import MuJoCoFreeBody, synchronizes_every_step


@wp.kernel(enable_backward=False)
def integrate(
    q: wp.array[float],
    qd: wp.array[float],
    body_f: wp.array[wp.spatial_vector],
    joint_f: wp.array[float],
    mass: wp.array[wp.float64],
    inertia: wp.array[wp.mat33d],
    damping: wp.array[wp.float64],
    timestep: wp.array[wp.float64],
    q_out: wp.array[float],
    qd_out: wp.array[float],
    body_q_out: wp.array[wp.transform],
    body_qd_out: wp.array[wp.spatial_vector],
):
    h = timestep[0]
    rot32 = wp.quat(q[3], q[4], q[5], q[6])
    # Match the native bridge's public float32 world-to-local conversion.
    local32 = wp.quat_rotate_inv(rot32, wp.vec3(qd[3], qd[4], qd[5]))
    w = wp.vec3d(local32)
    rot = wp.normalize(wp.quatd(rot32))
    force32 = body_f[0] + wp.spatial_vector(joint_f[0], joint_f[1], joint_f[2], joint_f[3], joint_f[4], joint_f[5])
    force = wp.vec3d(wp.spatial_top(force32))
    torque = wp.quat_rotate_inv(rot, wp.vec3d(wp.spatial_bottom(force32)))
    v = wp.vec3d(wp.float64(qd[0]), wp.float64(qd[1]), wp.float64(qd[2]))
    v_new = wp.vec3d()
    for axis in range(3):
        a = (force[axis] - damping[axis] * v[axis]) / (mass[0] + h * damping[axis])
        v_new[axis] = v[axis] + h * a
    local_inertia = inertia[0]
    momentum = local_inertia @ w
    D = wp.diag(wp.vec3d(damping[3], damping[4], damping[5]))
    gyro_jacobian = wp.skew(w) @ local_inertia - wp.skew(momentum)
    A = local_inertia + h * (D + gyro_jacobian)
    rhs = torque - D @ w - wp.cross(w, momentum)
    # Normalize before inversion to avoid reciprocal of a tiny determinant.
    scale = wp.max(wp.max(wp.abs(A[0, 0]), wp.abs(A[1, 1])), wp.abs(A[2, 2]))
    acceleration = wp.inverse(A / scale) @ (rhs / scale)
    w_new = w + h * acceleration
    angle = h * wp.length(w_new)
    rotation_axis = wp.normalize(w_new)
    half = wp.float64(0.5) * angle
    delta = wp.quatd(
        rotation_axis[0] * wp.sin(half), rotation_axis[1] * wp.sin(half), rotation_axis[2] * wp.sin(half), wp.cos(half)
    )
    next_rot = wp.quat(wp.normalize(rot * delta))
    next_pos = wp.vec3(wp.vec3d(wp.float64(q[0]), wp.float64(q[1]), wp.float64(q[2])) + h * v_new)
    next_v = wp.vec3(v_new)
    # Native bridge first rounds local velocity/quaternion, then converts.
    next_w = wp.quat_rotate(next_rot, wp.vec3(w_new))
    for axis_i in range(3):
        q_out[axis_i] = next_pos[axis_i]
        q_out[3 + axis_i] = next_rot[axis_i]
        qd_out[axis_i] = next_v[axis_i]
        qd_out[3 + axis_i] = next_w[axis_i]
    q_out[6] = next_rot[3]
    body_q_out[0] = wp.transform(next_pos, next_rot)
    body_qd_out[0] = wp.spatial_vector(next_v, next_w)


class MuJoCoFreeBodyGPU:
    """Graphable trials for one zero-COM world FREE body, without native forces.

    ``supports`` checks Newton frames once. Model topology, frame arrays and
    inertia storage must remain fixed for this object's lifetime; reconstruct
    after model changes. ``prepare`` checks native settings before every graph
    execution and updates damping/timestep buffers without reading Newton
    device arrays. Unsupported or changed settings raise ``ValueError`` so the
    caller can retain its native integration path. The timestep buffer is the
    ``float64`` array of ``step_time``, the ``StepTime`` of the step these trials
    belong to; without one this object owns its own.

    Native contacts, gravity, actuators, constraints, callbacks, fluid forces,
    armature, springs and friction loss are excluded. The caller owns physical
    acceptance, native-history synchronization and rollback. This specialization
    is non-differentiable and qualified only against MuJoCo 3.17.0.
    """

    @staticmethod
    def _native_supported(solver):
        if mujoco.__version__ != "3.17.0" or mujoco.mj_version() != 3017000:
            return False
        if any(
            callback() is not None
            for callback in (
                mujoco.get_mjcb_passive,
                mujoco.get_mjcb_control,
                mujoco.get_mjcb_sensor,
                mujoco.get_mjcb_contactfilter,
            )
        ):
            return False
        native = solver.mj_model
        return bool(
            solver.use_mujoco_cpu
            and synchronizes_every_step(solver)
            and native.nbody == 2
            and native.nq == 7
            and native.nv == 6
            and native.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            and not any(
                getattr(native, name)
                for name in ("nu", "na", "neq", "ntendon", "nmocap", "nplugin", "nhistory", "nsensor", "nflex", "npair")
            )
            and np.all(native.geom_bodyid == 1)
            and not any(
                np.any(getattr(native, name))
                for name in ("body_ipos", "dof_armature", "dof_frictionloss", "dof_dampingpoly", "jnt_stiffness")
            )
            and not np.any(native.opt.gravity)
            and native.opt.density == 0
            and native.opt.viscosity == 0
            and native.opt.enableflags == 0
            and native.opt.disableflags == int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
            and np.isfinite(native.body_mass[1])
            and native.body_mass[1] > 0
            and np.all(np.isfinite(native.body_inertia[1]))
            and np.all(native.body_inertia[1] > 0)
            and np.all(np.isfinite(native.body_iquat[1]))
            # Newton imports this frame from float32 model storage.
            and np.isclose(np.linalg.norm(native.body_iquat[1]), 1.0, rtol=0, atol=1e-6)
            and np.all(np.isfinite(native.dof_damping))
            and np.all(native.dof_damping >= 0)
        )

    @classmethod
    def supports(cls, model, solver):
        return bool(
            not getattr(model, "requires_grad", False)
            and MuJoCoFreeBody.supports(model, solver)
            and cls._native_supported(solver)
        )

    def __init__(self, model, solver, *, step_time=None):
        if not self.supports(model, solver):
            raise ValueError("Device free-body trials require the supported MuJoCo 3.17.0 implicitfast model")
        native = solver.mj_model
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, native.body_iquat[1])
        rotation = rotation.reshape(3, 3)
        inertia = rotation @ np.diag(native.body_inertia[1]) @ rotation.T
        self.model, self.solver = model, solver
        self.mass = wp.array([native.body_mass[1]], dtype=wp.float64, device=model.device)
        self.inertia = wp.array(inertia[None], dtype=wp.mat33d, device=model.device)
        self.damping = wp.array(native.dof_damping, dtype=wp.float64, device=model.device)
        self.step_time = StepTime(model.device) if step_time is None else step_time
        self.timestep = self.step_time.seconds_f64
        self.zero_joint_force = wp.zeros(6, dtype=float, device=model.device)
        self._native_model = native
        self._solver_step = solver.step
        self._damping = native.dof_damping.copy()
        self._mass = float(native.body_mass[1])
        self._inertia = native.body_inertia[1].copy()
        self._inertia_rotation = native.body_iquat[1].copy()
        self._dt = None

    def prepare(self, dt):
        """Validate and stage coefficients before capture/replay; no integration."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Free-body timestep must be finite and positive")
        if (
            self.solver.model is not self.model
            or self.solver.mj_model is not self._native_model
            or self.solver.step != self._solver_step
            or not self._native_supported(self.solver)
        ):
            raise ValueError("Native settings no longer support device free-body trials")
        native = self.solver.mj_model
        if (
            native.body_mass[1] != self._mass
            or not np.array_equal(native.body_inertia[1], self._inertia)
            or not np.array_equal(native.body_iquat[1], self._inertia_rotation)
        ):
            raise ValueError("Rebuild device free-body trials after inertia changes")
        if not np.array_equal(native.dof_damping, self._damping):
            self.damping.assign(native.dof_damping)
            self._damping[:] = native.dof_damping
        # Outside capture: a changed timestep is written here, before the graph runs.
        self.step_time.set(dt)
        self._dt = dt

    def step(self, source, out, control, dt=None):
        """Pure trial; call ``prepare`` before every enclosing graph execution."""
        if self._dt is None or (dt is not None and dt != self._dt) or self.step_time.value != self._dt:
            raise ValueError("Call prepare(dt) before the graphable free-body step")
        if (
            any(getattr(out, name, None) is not None for name in ("body_qdd", "body_parent_f"))
            or getattr(getattr(out, "mujoco", None), "qfrc_actuator", None) is not None
        ):
            raise ValueError("Extended rigid observables require the native solver")
        joint_f = self.zero_joint_force if control is None or control.joint_f is None else control.joint_f
        wp.launch(
            integrate,
            1,
            [
                source.joint_q,
                source.joint_qd,
                source.body_f,
                joint_f,
                self.mass,
                self.inertia,
                self.damping,
                self.timestep,
                out.joint_q,
                out.joint_qd,
                out.body_q,
                out.body_qd,
            ],
            device=self.model.device,
        )
