# SPDX-License-Identifier: Apache-2.0
"""Finite force/torque drive for the free MuJoCo needle, with no pose playback."""

import newton
import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _drive(
    body: int,
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    target: wp.transform,
    target_velocity: wp.spatial_vector,
    stiffness: float,
    damping: float,
    angular_stiffness: float,
    angular_damping: float,
    implicit_damping: bool,
    forces: wp.array[wp.spatial_vector],
    drive_force: wp.array[wp.spatial_vector],
):
    current = q[body]
    error = wp.transform_point(target, com[body]) - wp.transform_point(current, com[body])
    rotation = wp.transform_get_rotation(target) * wp.quat_inverse(wp.transform_get_rotation(current))
    if rotation[3] < 0.0:
        rotation = -rotation
    imaginary = wp.vec3(rotation[0], rotation[1], rotation[2])
    sine = wp.length(imaginary)
    angular_error = 2.0 * wp.atan2(sine, rotation[3]) * imaginary / wp.max(sine, 1.0e-8)
    velocity = qd[body]
    if implicit_damping:
        velocity = wp.spatial_vector()
    force = stiffness * error + damping * (wp.spatial_top(target_velocity) - wp.spatial_top(velocity))
    torque = angular_stiffness * angular_error
    torque += angular_damping * (wp.spatial_bottom(target_velocity) - wp.spatial_bottom(velocity))
    wrench = wp.spatial_vector(force, torque)
    wp.atomic_add(forces, body, wrench)
    drive_force[0] = wrench


def pose_velocity(previous, following, com, dt):
    """World COM linear/angular velocity between nearby desired body poses."""
    before, after = wp.transform(*previous), wp.transform(*following)
    linear = (
        np.asarray(wp.transform_point(after, wp.vec3(*com))) - np.asarray(wp.transform_point(before, wp.vec3(*com)))
    ) / dt
    rotation = wp.quat(*following[3:]) * wp.quat_inverse(wp.quat(*previous[3:]))
    values = np.asarray(rotation, dtype=float)
    if values[3] < 0:
        values = -values
    sine = np.linalg.norm(values[:3])
    angular = values[:3] * (2 * np.arctan2(sine, values[3]) / max(sine, 1.0e-12) / dt)
    return np.r_[linear, angular]


class NeedleDrive:
    """A PD wrench at COM; finite gains let tissue loads change actual motion."""

    def __init__(self, model, body, *, stiffness=40.0, angular_stiffness=0.001):
        if not np.isfinite([stiffness, angular_stiffness]).all() or min(stiffness, angular_stiffness) <= 0:
            raise ValueError("Needle drive stiffnesses must be finite and positive")
        self.model, self.body = model, body
        self.stiffness, self.angular_stiffness = stiffness, angular_stiffness
        self.com = model.body_com.numpy()[body]
        mass = float(model.body_mass.numpy()[body])
        inertia = np.linalg.eigvalsh(model.body_inertia.numpy()[body]).max()
        self.damping = 2 * np.sqrt(stiffness * mass)
        self.angular_damping = 2 * np.sqrt(angular_stiffness * inertia)
        self.implicit_damping = False
        self.spring_linearization_dt = None
        self._mujoco_solver = None
        self._configured_damping = np.zeros(6, dtype=np.float32)
        self.wrench = wp.zeros(1, dtype=wp.spatial_vector, device=model.device)

    def configure_mujoco(self, solver, dt=None):
        """Integrate this drive's velocity damping through native MuJoCo.

        The supported body has one world FREE joint and its origin at COM.
        Scalar linear/angular damping is then identical in Newton's world-COM
        twist and MuJoCo's world-linear/body-angular free-joint coordinates.
        Offset COMs require coupled damping terms and must retain explicit mode.
        Existing passive damping is preserved; repeated calls do not add twice.
        Requires ``implicitfast``. With ``dt=None`` spring feedback stays
        explicit. A positive ``dt`` adds ``dt * stiffness`` to native damping:
        translation is then exactly backward Euler for the held target. The
        same rotational term linearizes the quaternion error with an identity
        Jacobian: exact for collinear principal-axis motion, approximate for
        general rotations. This changes integration, not the continuous gains.
        Call between physical steps and update whenever the timestep changes.
        """
        import mujoco  # noqa: PLC0415

        if dt is not None and (not np.isfinite(dt) or dt <= 0):
            raise ValueError("Implicit spring timestep must be finite and positive")
        if solver.model is not self.model or not solver.use_mujoco_cpu:
            raise ValueError("Implicit drive damping requires the same model's CPU MuJoCo solver")
        if solver.mj_model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
            raise ValueError("Implicit drive damping requires the implicitfast integrator")
        if np.any(self.com != 0):
            raise ValueError("Implicit drive damping requires the body origin at COM")
        joints = np.flatnonzero(self.model.joint_child.numpy() == self.body)
        if len(joints) != 1:
            raise ValueError("Implicit drive damping requires one world FREE joint")
        joint = int(joints[0])
        starts = self.model.joint_qd_start.numpy()
        start, end = int(starts[joint]), int(starts[joint + 1])
        if (
            self.model.joint_type.numpy()[joint] != int(newton.JointType.FREE)
            or self.model.joint_parent.numpy()[joint] != -1
            or end - start != 6
        ):
            raise ValueError("Implicit drive damping requires one world FREE joint with six DOFs")
        damping = np.array([self.damping] * 3 + [self.angular_damping] * 3, dtype=np.float32)
        if dt is not None:
            damping += np.array([dt * self.stiffness] * 3 + [dt * self.angular_stiffness] * 3, dtype=np.float32)
        if self._mujoco_solver is solver and np.array_equal(damping, self._configured_damping):
            self.spring_linearization_dt = dt
            return
        values = self.model.joint_damping.numpy().copy()
        values[start:end] += damping - self._configured_damping
        self.model.joint_damping.assign(values)
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
        self._configured_damping = damping
        self.implicit_damping = True
        self._mujoco_solver = solver
        self.spring_linearization_dt = dt

    def prepare_timestep(self, dt):
        """Enable/update the implicit spring linearization before a physical trial.

        Unchanged timesteps do not notify or copy model properties. External
        target-velocity feedforward always uses the original damping gain.
        """
        if self._mujoco_solver is None:
            raise RuntimeError("Configure the MuJoCo drive before preparing its timestep")
        if dt is None or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Implicit spring timestep must be finite and positive")
        if dt != self.spring_linearization_dt:
            self.configure_mujoco(self._mujoco_solver, dt)

    def add_forces(self, state, target_pose, target_velocity):
        wp.launch(
            _drive,
            dim=1,
            inputs=[
                self.body,
                state.body_q,
                state.body_qd,
                self.model.body_com,
                wp.transform(*target_pose),
                wp.spatial_vector(*target_velocity),
                self.stiffness,
                self.damping,
                self.angular_stiffness,
                self.angular_damping,
                self.implicit_damping,
                state.body_f,
                self.wrench,
            ],
            device=self.model.device,
        )


def ramp_distance(time, speed, ramp_time=0.5):
    """Integrated smoothstep speed ramp: zero initial speed and acceleration."""
    u = np.clip(time / ramp_time, 0.0, 1.0)
    return speed * (ramp_time * (u**3 - 0.5 * u**4) + np.maximum(0.0, time - ramp_time))
