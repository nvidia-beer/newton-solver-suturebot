# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-way coupling of thread chains to rigid bodies without host readbacks.

Each chain's node 0 is tied to a point of one rigid body: its position follows
the point and its slope follows a body-fixed direction. A rigid solver owns the
body; the thread acts on it through an external wrench.

A light body on a thread cannot take the thread's force one step late: a taut
thread's stiffness times dt squared is far larger than the body's mass, and the
thread switches between slack and taut within a step. The body's response is
therefore solved with the thread. The rigid solver first steps the body
without the thread's wrench; that predicted state includes contact and every
other load. The thread solver then carries node 0 on a rigid head with the
body's mass and inertia that moves by a translation u and a rotation theta
about the predicted pose, as unknowns of its own Newton solve. A wrench W moves
a free body by dt² M^-1 W within a semi-implicit Euler step, so the head's
inertia is M / dt² and the wrench that produces the solved motion is

    W = (M / dt²) (u, theta).

The rigid solver then repeats the step from the same state with W added.

The head's solve starts where the thread's own motion carries node 0, at
q + dt qd like every free node, not at the predicted pose. When the body's
predicted velocity jumps, on a contact or when a hold lets go, that first
iterate strains no element; the offset from the predicted pose is a quadratic
cost on the head's inertia, which one Newton update removes. Started at the
predicted pose, the jump crushes a stiff thread's first element, and a solve
that stops after a few updates leaves that energy in the thread and in the
wrench. Lagged mode keeps the head's last motion as its start.

The head moves as a free body would. Where the rigid solver constrains the
body, by contact or a grasp, the body yields less than the head did: the thread
then sees a compliance of dt² / m in series with the attachment, and its pull
on the body is underestimated. Predicting with the previous wrench removes
that compliance but lets a taut thread and a stiff contact fight through the
one-step lag between the two solvers, which does not settle. A body that a
gripper holds should prescribe node 0 instead.

Lagged mode serves a body whose solver cannot repeat its step, when the body
is held by a support with a finite stiffness k and damping c, such as a drive.
The body then steps once per time step, with the wrench W_n of the thread's
last step. Within a step a wrench moves it by Z^-1 W with the impedance

    Z = M / dt² + c / dt + k,

so the pose it would have reached without the thread is its new pose moved
back by Z^-1 W_n. The head is placed there with Z as its inertia, and the
solved motion gives the wrench Z (u, theta); the next wrench W_{n+1} is the
mean of that and W_n, which keeps successive wrenches from overshooting in
turn. When the wrench has settled the head is at the body's pose. Until then node 0 is off the
attachment point by Z^-1 (W_{n+1} - W_n). A support stiffer than stated, such
as tissue around the body, only slows the wrench's approach to its value.
"""

from collections.abc import Sequence

import newton
import numpy as np
import warp as wp

from .kernels import Mat66d, Vec6d
from .solver import SolverThread


@wp.kernel(enable_backward=False)
def _attach(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    bodies: wp.array[int],
    point: wp.vec3,
    slope: wp.vec3,
    dt: float,
    dofs: int,
    support: wp.array[wp.vec4],
    lagged: bool,
    wrench: wp.array[wp.spatial_vector],
    boundary_q: wp.array[wp.float64],
    head_center: wp.array[wp.vec3d],
    head_arm: wp.array[wp.vec3d],
    head_slope: wp.array[wp.vec3d],
    head_inertia: wp.array[Mat66d],
):
    """Give each chain a rigid head at its body's predicted pose; one thread per chain.

    ``support`` holds the linear and angular damping and the linear and
    angular stiffness of whatever holds the body. In lagged mode the given
    pose includes the last wrench, which is taken out again.
    """
    chain = wp.tid()
    body = bodies[chain]
    pose = body_q[body]
    rotation = wp.transform_get_rotation(pose)
    center = wp.transform_point(pose, body_com[body])
    turn = wp.quat_to_matrix(rotation)
    inertia = turn * body_inertia[body] * wp.transpose(turn)
    held = support[chain]
    linear = body_mass[body] / (dt * dt) + held[0] / dt + held[2]
    angular = inertia / (dt * dt) + (held[1] / dt + held[3]) * wp.identity(n=3, dtype=float)
    if lagged:
        applied = wrench[chain]
        center -= wp.spatial_top(applied) / linear
        back = -(wp.inverse(angular) * wp.spatial_bottom(applied))
        angle = wp.length(back)
        if angle > 0.0:
            rotation = wp.quat_from_axis_angle(back / angle, angle) * rotation
    arm = wp.quat_rotate(rotation, point - body_com[body])
    tangent = wp.quat_rotate(rotation, slope)
    position = center + arm
    for axis in range(3):
        boundary_q[chain * dofs + axis] = wp.float64(position[axis])
        boundary_q[chain * dofs + 3 + axis] = wp.float64(tangent[axis])
    head_center[chain] = wp.vec3d(center)
    head_arm[chain] = wp.vec3d(arm)
    head_slope[chain] = wp.vec3d(tangent)
    rate = wp.float64(1.0) / (wp.float64(dt) * wp.float64(dt))
    inertial = Mat66d()
    for i in range(3):
        inertial[i, i] = rate * wp.float64(body_mass[body]) + wp.float64(held[0] / dt + held[2])
        for j in range(3):
            inertial[i + 3, j + 3] = rate * wp.float64(inertia[i, j])
        inertial[i + 3, i + 3] += wp.float64(held[1] / dt + held[3])
    head_inertia[chain] = inertial


@wp.kernel(enable_backward=False)
def _start_head(
    q: wp.array[wp.float64],
    qd: wp.array[wp.float64],
    dofs: int,
    dt: wp.float64,
    head_center: wp.array[wp.vec3d],
    head_arm: wp.array[wp.vec3d],
    head_slope: wp.array[wp.vec3d],
    head_motion: wp.array[Vec6d],
):
    """Start each head at the motion that puts node 0 at q + dt qd; one thread per chain.

    The rotation is the smallest one that turns the predicted slope onto the
    extrapolated one; the translation then places node 0.
    """
    chain = wp.tid()
    first = chain * dofs
    position = wp.vec3d(q[first] + dt * qd[first], q[first + 1] + dt * qd[first + 1], q[first + 2] + dt * qd[first + 2])
    slope = wp.normalize(
        wp.vec3d(
            q[first + 3] + dt * qd[first + 3], q[first + 4] + dt * qd[first + 4], q[first + 5] + dt * qd[first + 5]
        )
    )
    predicted = wp.normalize(head_slope[chain])
    axis = wp.cross(predicted, slope)
    sine = wp.length(axis)
    turn = wp.vec3d(wp.float64(0.0))
    lever = head_arm[chain]
    # Parallel slopes need no rotation, and their cross product gives no axis.
    if sine > wp.float64(0.0):
        angle = wp.atan2(sine, wp.dot(predicted, slope))
        turn = axis * (angle / sine)
        lever = wp.quat_rotate(wp.quat_from_axis_angle(axis / sine, angle), lever)
    shift = position - head_center[chain] - lever
    head_motion[chain] = Vec6d(shift[0], shift[1], shift[2], turn[0], turn[1], turn[2])


@wp.kernel(enable_backward=False)
def _apply_wrench(
    bodies: wp.array[int],
    head_inertia: wp.array[Mat66d],
    head_motion: wp.array[Vec6d],
    body_f: wp.array[wp.spatial_vector],
    wrench: wp.array[wp.spatial_vector],
):
    """Add to each body the wrench that moves it as its head moved; one thread per chain.

    The wrench is a world force and a world torque about the centre of mass.
    """
    chain = wp.tid()
    result = head_inertia[chain] * head_motion[chain]
    applied = wp.spatial_vector(
        float(result[0]), float(result[1]), float(result[2]), float(result[3]), float(result[4]), float(result[5])
    )
    wrench[chain] = applied
    wp.atomic_add(body_f, bodies[chain], applied)


@wp.kernel(enable_backward=False)
def _wrench(head_inertia: wp.array[Mat66d], head_motion: wp.array[Vec6d], wrench: wp.array[wp.spatial_vector]):
    """Move each stored wrench halfway to the one that moves its body as its head moved; one thread per chain.

    The wrench of one step answers the body's pose after the step before. A
    body that yields less than its stated impedance, or a thread much stiffer
    than that impedance, makes successive wrenches overshoot in turn, high,
    low, high. The mean of two successive values cancels that alternation and
    leaves a settled wrench as it is.
    """
    chain = wp.tid()
    result = head_inertia[chain] * head_motion[chain]
    solved = wp.spatial_vector(
        float(result[0]), float(result[1]), float(result[2]), float(result[3]), float(result[4]), float(result[5])
    )
    wrench[chain] = 0.5 * (wrench[chain] + solved)


@wp.kernel(enable_backward=False)
def _add_wrench(bodies: wp.array[int], wrench: wp.array[wp.spatial_vector], body_f: wp.array[wp.spatial_vector]):
    """Add each chain's stored wrench to its body; one thread per chain."""
    wp.atomic_add(body_f, bodies[wp.tid()], wrench[wp.tid()])


class ThreadAttachment:
    """Ties node 0 of every chain of a thread solver to a rigid body.

    Per step: fill ``state.body_f`` with every other load on the bodies and
    step the rigid solver into a scratch state; call :meth:`attach` with that
    predicted state; step the thread; call :meth:`apply` on the original
    state; step the rigid solver again from it. Both calls only launch
    kernels, so the sequence can be captured in a CUDA graph. The thread
    solver is switched to rigid heads.

    In lagged mode, per step: call :meth:`add_forces` on the body forces and
    step the rigid solver once; call :meth:`attach` with the new state; step
    the thread; call :meth:`update`.

    Args:
        thread: Thread solver; its ``n_envs`` chains are attached in order.
        model: Newton model that owns the bodies.
        bodies: One body index per chain.
        point: Attachment point in the body frame [m].
        tangent: Direction in which the thread leaves the body, body frame.
        lagged: The rigid solver steps once per time step with the wrench of
            the thread's previous step.

    Attributes:
        wrench: World force and torque about the centre of mass that each
            chain applied to its body in the last step [N, N m].
    """

    def __init__(
        self, thread: SolverThread, model: newton.Model, bodies: Sequence[int], point, tangent, *, lagged: bool = False
    ):
        if model.device != thread.device:
            raise ValueError("The thread and the rigid model must share a device")
        indices = np.asarray(bodies, dtype=np.int32)
        if indices.shape != (thread.n_envs,) or np.any(indices < 0) or np.any(indices >= model.body_count):
            raise ValueError("bodies must hold one valid body index per chain")
        tangent = np.asarray(tangent, dtype=np.float64)
        if np.linalg.norm(tangent) < 1.0e-12:
            raise ValueError("tangent must be nonzero")
        rest = thread.thread.rest_q.numpy()
        self.thread, self.model = thread, model
        thread.rigid_head = True
        self._bodies = wp.array(indices, dtype=int, device=model.device)
        self._point = wp.vec3(*np.asarray(point, dtype=np.float64))
        # The slope keeps the reference length of the chain's node-0 slope, so the attachment adds no axial strain.
        self._slope = wp.vec3(*(tangent / np.linalg.norm(tangent) * np.linalg.norm(rest[3:6])))
        self.wrench = wp.zeros(thread.n_envs, dtype=wp.spatial_vector, device=model.device)
        self.lagged = bool(lagged)
        self._support = wp.zeros(thread.n_envs, dtype=wp.vec4, device=model.device)

    def set_support(self, damping=(0.0, 0.0), stiffness=(0.0, 0.0)) -> None:
        """State what holds the bodies besides their inertia, such as a drive.

        Args:
            damping: Linear [N s/m] and angular [N m s] damping of the support.
            stiffness: Linear [N/m] and angular [N m] stiffness of the support.
        """
        values = np.array([damping[0], damping[1], stiffness[0], stiffness[1]], dtype=np.float32)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Support damping and stiffness must be finite and nonnegative")
        self._support.assign(np.tile(values, (self.thread.n_envs, 1)))

    def attach(self, predicted, dt: float) -> None:
        """Place every chain's head at its body's pose in ``predicted``, with the body's inertia.

        ``predicted`` is a state or its ``body_q`` array: the rigid solver's
        step of size ``dt`` without the thread's wrench or, in lagged mode,
        with the wrench of the thread's previous step.
        """
        model, thread = self.model, self.thread
        body_q = predicted.body_q if hasattr(predicted, "body_q") else predicted
        wp.launch(
            _attach,
            dim=thread.n_envs,
            inputs=[
                body_q,
                model.body_com,
                model.body_mass,
                model.body_inertia,
                self._bodies,
                self._point,
                self._slope,
                dt,
                thread.thread.dof_count,
                self._support,
                self.lagged,
                self.wrench,
            ],
            outputs=[
                thread.boundary_q,
                thread.head_center,
                thread.head_arm,
                thread.head_slope,
                thread.head_inertia,
            ],
            device=model.device,
        )
        if not self.lagged:
            # The thread solver takes head_motion as the head's first iterate.
            wp.launch(
                _start_head,
                dim=thread.n_envs,
                inputs=[
                    thread.q,
                    thread.qd,
                    thread.thread.dof_count,
                    dt,
                    thread.head_center,
                    thread.head_arm,
                    thread.head_slope,
                ],
                outputs=[thread.head_motion],
                device=model.device,
            )

    def apply(self, state: newton.State) -> None:
        """Add every chain's wrench to ``state.body_f``; call after the thread step."""
        thread = self.thread
        wp.launch(
            _apply_wrench,
            dim=thread.n_envs,
            inputs=[self._bodies, thread.head_inertia, thread.head_motion],
            outputs=[state.body_f, self.wrench],
            device=self.model.device,
        )

    def update(self) -> None:
        """Store every chain's wrench of the thread step just made, without applying it; for lagged mode."""
        thread = self.thread
        wp.launch(
            _wrench,
            dim=thread.n_envs,
            inputs=[thread.head_inertia, thread.head_motion],
            outputs=[self.wrench],
            device=self.model.device,
        )

    def add_forces(self, body_f) -> None:
        """Add every chain's stored wrench to a body force array; for lagged mode, before the rigid step."""
        wp.launch(
            _add_wrench,
            dim=self.thread.n_envs,
            inputs=[self._bodies, self.wrench],
            outputs=[body_f],
            device=self.model.device,
        )
