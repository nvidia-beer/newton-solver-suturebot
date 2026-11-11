# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact of one thread with a chain of capsules on a rigid body, without host readbacks.

The body's capsules share one radius and lie along a polyline in the body
frame, as the collision capsules of a curved needle do. Before a thread step
every element of the thread takes the nearest point of that polyline, at the
body's current pose, as its obstacle: the thread solver keeps the element's
closest point one capsule radius plus one thread radius away from it. The
element is taken as its chord. The first elements of the thread are left out,
because the thread is attached to the body there.

The thread solver treats the body as fixed during its step, and the body
receives the opposite forces as a wrench in its next step. As with the
tissue, that exchange is relaxed. Within a step the body resists a force at
least with K = m / dt² + c / dt, its mass and the damping of what holds it,
and a torque at least with K_r = I / dt² + c_r / dt, its smallest moment of
inertia and that support's angular damping. With the contact stiffness k and
the largest distance a of the capsule axis from the centre of mass, each
step moves the applied wrench towards the current one by the smaller of the
fractions K / (K + k) and K_r / (K_r + k a²).
"""

import numpy as np
import warp as wp

from .kernels import closest_on_segments
from .solver import SolverThread


@wp.kernel(enable_backward=False)
def _find_capsules(
    q: wp.array[wp.float64],
    body_q: wp.array[wp.transform],
    body: int,
    axis: wp.array[wp.vec3],
    reach: wp.float64,
    skip: int,
    xi: wp.array[wp.float64],
    point: wp.array[wp.vec3d],
    normal: wp.array[wp.vec3d],
):
    """Give every thread element the nearest point of the body's capsule axis as its obstacle; launch over elements."""
    element = wp.tid()
    normal[element] = wp.vec3d()
    if element < skip:
        return
    a0 = wp.vec3d(q[6 * element], q[6 * element + 1], q[6 * element + 2])
    a1 = wp.vec3d(q[6 * element + 6], q[6 * element + 7], q[6 * element + 8])
    pose = body_q[body]
    nearest = reach
    for segment in range(axis.shape[0] - 1):
        b0 = wp.vec3d(wp.transform_point(pose, axis[segment]))
        b1 = wp.vec3d(wp.transform_point(pose, axis[segment + 1]))
        fractions = closest_on_segments(a0, a1, b0, b1)
        here = a0 + fractions[0] * (a1 - a0)
        there = b0 + fractions[1] * (b1 - b0)
        distance = wp.length(here - there)
        if distance < nearest and distance > wp.float64(0.0):
            nearest = distance
            xi[element] = fractions[0]
            point[element] = there
            normal[element] = (here - there) / distance


@wp.kernel(enable_backward=False)
def _reaction(
    force: wp.array[wp.vec3d],
    point: wp.array[wp.vec3d],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body: int,
    fresh: wp.array[wp.spatial_vector],
):
    """Add the opposite of each element's obstacle force to the body's wrench about its centre of mass."""
    element = wp.tid()
    push = -wp.vec3(float(force[element][0]), float(force[element][1]), float(force[element][2]))
    at = wp.vec3(float(point[element][0]), float(point[element][1]), float(point[element][2]))
    lever = at - wp.transform_point(body_q[body], body_com[body])
    wp.atomic_add(fresh, 0, wp.spatial_vector(push, wp.cross(lever, push)))


@wp.kernel(enable_backward=False)
def _relax(fresh: wp.array[wp.spatial_vector], relaxation: float, wrench: wp.array[wp.spatial_vector]):
    wrench[0] += relaxation * (fresh[0] - wrench[0])
    fresh[0] = wp.spatial_vector()


@wp.kernel(enable_backward=False)
def _add_wrench(body: int, wrench: wp.array[wp.spatial_vector], body_f: wp.array[wp.spatial_vector]):
    wp.atomic_add(body_f, body, wrench[0])


class ThreadCapsuleContact:
    """Keeps a thread out of the capsules of one rigid body and returns the contact forces to the body.

    Per step: call :meth:`add_forces` on the body forces and step the rigid
    solver; call :meth:`update` with the body's new poses; step the thread;
    call :meth:`react`. All calls only launch kernels.

    Args:
        thread: Thread solver of one chain, made with an obstacle distance
            of the capsule radius plus the thread radius.
        model: Newton model that owns the body.
        body: Index of the body.
        axis: Polyline of the capsules' axis in the body frame [m], shape
            (count, 3).
        dt: Step size [s].
        damping: Linear [N s/m] and angular [N m s] damping of whatever
            holds the body, a lower bound over the run.
        skip: Number of elements at the thread's start that have no contact
            with the body.

    Attributes:
        wrench: The relaxed wrench of the thread's contact on the body, a
            world force and a torque about the centre of mass [N, N m]; part
            of the state between steps.
    """

    def __init__(self, thread: SolverThread, model, body: int, axis, *, dt: float, damping=(0.0, 0.0), skip: int = 2):
        if thread.n_envs != 1 or thread.obstacle_distance is None:
            raise ValueError("The capsule contact needs a thread solver of one chain with an obstacle distance")
        if model.device != thread.device:
            raise ValueError("The thread and the rigid model must share a device")
        axis = np.asarray(axis, dtype=np.float32).reshape(-1, 3)
        if len(axis) < 2 or not 0 <= body < model.body_count:
            raise ValueError("The capsule axis needs two points and the body must exist")
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(damping).all() or min(damping) < 0:
            raise ValueError("dt must be positive and the damping finite and nonnegative")
        self.thread, self.model, self.body, self.skip = thread, model, int(body), int(skip)
        device = model.device
        self._axis = wp.array(axis, dtype=wp.vec3, device=device)
        # Ten smoothing widths beyond touching, the smooth onset of the penalty has died away.
        self._reach = thread.obstacle_distance + 10.0 * thread.contact_smoothing
        mass = float(model.body_mass.numpy()[body])
        inertia = float(np.linalg.eigvalsh(model.body_inertia.numpy()[body]).min())
        reach = float(np.linalg.norm(axis - model.body_com.numpy()[body], axis=1).max())
        force_stiffness = mass / dt**2 + damping[0] / dt
        torque_stiffness = inertia / dt**2 + damping[1] / dt
        contact = thread.contact_stiffness
        self._relaxation = min(
            force_stiffness / (force_stiffness + contact),
            torque_stiffness / (torque_stiffness + contact * reach**2),
        )
        self.wrench = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        self._fresh = wp.zeros(1, dtype=wp.spatial_vector, device=device)

    @property
    def state(self) -> tuple:
        """The arrays that carry over from one step to the next; see :attr:`SolverThread.state`."""
        return (self.wrench,)

    def update(self, body_q) -> None:
        """Give the thread its obstacles for the body at the poses ``body_q``; call before the thread step."""
        thread = self.thread
        wp.launch(
            _find_capsules,
            dim=thread.thread.element_count,
            inputs=[thread.q, body_q, self.body, self._axis, self._reach, self.skip],
            outputs=[thread.obstacle_xi, thread.obstacle_point, thread.obstacle_normal],
            device=self.model.device,
        )

    def react(self, body_q) -> None:
        """Move the relaxed wrench towards that of the thread step just made; ``body_q`` as given to :meth:`update`."""
        thread, device = self.thread, self.model.device
        wp.launch(
            _reaction,
            dim=thread.thread.element_count,
            inputs=[thread.obstacle_force, thread.obstacle_point, body_q, self.model.body_com, self.body],
            outputs=[self._fresh],
            device=device,
        )
        wp.launch(_relax, dim=1, inputs=[self._fresh, self._relaxation], outputs=[self.wrench], device=device)

    def add_forces(self, body_f) -> None:
        """Add the relaxed wrench to a body force array; call before the rigid step."""
        wp.launch(_add_wrench, dim=1, inputs=[self.body, self.wrench], outputs=[body_f], device=self.model.device)
