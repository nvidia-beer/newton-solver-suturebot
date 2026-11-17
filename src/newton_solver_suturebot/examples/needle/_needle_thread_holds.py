# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Attachments for the needle and thread example: spring holds on the needle and pins on thread nodes.

The class is the backend of :class:`ScreenPickTool`. A right-button press
picks the needle or the thread node nearest to the cursor, the drag moves that
attachment's target, and releasing the button leaves it in place.

A hold on the needle is a spring and damper between the grabbed body point and
its target. While any hold is active the needle's weight is also carried, once
and at the centre of mass: a weight carried at the grabbed points has a lever,
which lifts the grabbed end of a needle that the floor still carries and turns
a needle with several holds away from them. A held needle therefore neither
sags nor swings under its own weight.

The gains follow from the step. A force at a point of a free body moves that
point as a mass m_eff would, with

    1 / m_eff = 1 / m + r² / I,

largest for the longest lever r and the smallest principal inertia I. Over one
semi-implicit Euler step dt, a spring K and damper C on that mass multiply the
state by a matrix with characteristic polynomial

    z² - (2 - a - b) z + (1 - b),    a = K dt² / m_eff,  b = C dt / m_eff,

which is stable for 0 < b < 2 and a + 2 b < 4. With b = 1 one root is zero:
the point's velocity is removed in one step and the position error is
multiplied by 1 - a per step. The holds use a = 1/4. Several holds on the
needle share these gains equally, so their sum stays within the same bound.

A pin on the thread is a prescribed node position of the thread solver, kept
by :class:`ThreadPins`.
"""

import numpy as np
import warp as wp

from newton_solver_suturebot.examples.needle._thread_pins import ThreadPins, nearest_to_ray


@wp.kernel(enable_backward=False)
def _hold_forces(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body: int,
    active: wp.array[int],
    local: wp.array[wp.vec3],
    target: wp.array[wp.vec3],
    stiffness: float,
    damping: float,
    spin_damping: float,
    support: wp.vec3,
    body_f: wp.array[wp.spatial_vector],
):
    """Add the needle holds' world force and torque about the centre of mass; one thread.

    ``support`` carries the body's weight once, at the centre of mass, where it has no lever.
    """
    count = int(0)
    for slot in range(active.shape[0]):
        if active[slot] != 0:
            count += 1
    if count == 0:
        return
    share = 1.0 / float(count)
    pose = body_q[body]
    center = wp.transform_point(pose, body_com[body])
    spin = wp.spatial_bottom(body_qd[body])
    linear = wp.spatial_top(body_qd[body])
    force_sum, torque_sum = wp.vec3(0.0), wp.vec3(0.0)
    for slot in range(active.shape[0]):
        if active[slot] != 0:
            point = wp.transform_point(pose, local[slot])
            lever = point - center
            velocity = linear + wp.cross(spin, lever)
            force = share * (stiffness * (target[slot] - point) - damping * velocity)
            force_sum += force
            torque_sum += wp.cross(lever, force)
    wp.atomic_add(body_f, body, wp.spatial_vector(force_sum + support, torque_sum - spin_damping * spin))


def hold_gains(mass, inertia, lever, dt, *, stiffness_fraction=0.25, damping_fraction=1.0, spin_fraction=0.1):
    """Spring [N/m], damper [N s/m] and spin damper [N m s/rad] of a hold that is stable for steps of ``dt``.

    Args:
        mass: Body mass [kg].
        inertia: Smallest principal inertia about the centre of mass [kg m²].
        lever: Largest distance from the centre of mass to a point that can be grabbed [m].
        dt: Step of the rigid solver [s].
        stiffness_fraction: a = K dt² / m_eff; the position error is multiplied by 1 - a per step.
        damping_fraction: b = C dt / m_eff; 1 removes the grabbed point's velocity in one step.
        spin_fraction: Share of a spin about the centre of mass, on the axis of the smallest inertia, that is
            removed per step. A spin about a grabbed point has the larger inertia I + m r² and loses less.
    """
    effective = 1.0 / (1.0 / mass + lever * lever / inertia)
    return (
        stiffness_fraction * effective / (dt * dt),
        damping_fraction * effective / dt,
        spin_fraction * inertia / dt,
    )


class NeedleThreadHolds:
    """Spring holds on one free rigid body and pins on the nodes of a one-chain thread solver.

    Attachments are numbered in the order they were made; removing one moves
    the last into its place, as :class:`ScreenPickTool` expects.

    Args:
        viewer: Newton viewer; its camera gives the size of a pixel for picking.
        model: Newton model that owns the body.
        body: Index of the held body.
        centerline: Points along the body's centerline in its frame [m]; a
            hold grabs the one nearest to the cursor.
        thread: Thread solver with one chain. Nodes that it already prescribes,
            such as node 0 and a fixed end, cannot be pinned.
        current_state: Returns the rigid state the viewer currently shows.
        dt: Step of the rigid solver [s].
        mass: Body mass [kg].
        inertia: Smallest principal inertia about the centre of mass [kg m²].
        lever: Largest distance from the centre of mass to a point of the
            centerline [m].
        gravity: Gravity vector [m/s²]; the holds carry the body's weight.
        capacity: Largest number of holds on the body.
    """

    # A press within this many pixels of the needle or of a thread node grabs it.
    click_radius = 12.0

    def __init__(
        self, viewer, model, body, centerline, thread, current_state, dt, mass, inertia, lever, gravity, capacity=8
    ):
        self.viewer, self.model, self.body, self.thread = viewer, model, int(body), thread
        self._centerline = np.asarray(centerline, dtype=np.float64)
        self._current_state = current_state
        device = model.device
        # Spring of one hold on the needle [N/m]; several holds share it.
        self.stiffness, self._damping, self._spin_damping = hold_gains(mass, inertia, lever, dt)
        self._support = wp.vec3(*(-mass * np.asarray(gravity, dtype=np.float64)))
        # Host copies of the device arrays; the device is only ever written from them.
        self._hold_active = np.zeros(capacity, dtype=np.int32)
        self._hold_local = np.zeros((capacity, 3), dtype=np.float32)
        self._hold_target = np.zeros((capacity, 3), dtype=np.float32)
        self._active_device = wp.array(self._hold_active, dtype=int, device=device)
        self._local_device = wp.array(self._hold_local, dtype=wp.vec3, device=device)
        self._target_device = wp.array(self._hold_target, dtype=wp.vec3, device=device)
        # The thread's pins; this class keeps the list of attachments, so the pins' own list stays empty.
        self._pins = ThreadPins(viewer, thread)
        # Committed attachments, ("needle", slot) or ("thread", node), and the one being dragged.
        self.attachments = []
        self._dragged = None

    # Attachments, for the pick tool and for scripted use.

    def hold_needle(self, point, pose=None):
        """Grab the needle at the world position ``point``; returns the attachment or None when all holds are used.

        ``point`` must lie on the needle: the gains are stable only within the lever given at construction.
        ``pose`` is the needle's current transform if the caller has it; it is read back otherwise.
        """
        free = np.flatnonzero(self._hold_active == 0)
        if not len(free):
            return None
        slot = int(free[0])
        if pose is None:
            pose = self._current_state().body_q.numpy()[self.body]
        local = wp.transform_point(wp.transform_inverse(wp.transform(pose[:3], pose[3:])), wp.vec3(*point))
        self._hold_active[slot], self._hold_local[slot], self._hold_target[slot] = 1, local, point
        self._upload_holds()
        return ("needle", slot)

    def pin_node(self, node, position=None):
        """Pin thread node ``node`` where it is; returns the attachment or None if the node cannot be pinned.

        ``position`` is the node's current position if the caller has it; it is read back otherwise.
        """
        node = self._pins.pin(node, position)
        return None if node is None else ("thread", node)

    def set_target(self, attachment, target):
        """Move the target of ``attachment`` to the world position ``target`` [m]."""
        kind, key = attachment
        if kind == "needle":
            self._hold_target[key] = target
            self._target_device.assign(self._hold_target)
        else:
            self._pins.set_target(key, target)

    def target(self, attachment):
        """World target of ``attachment`` [m]."""
        kind, key = attachment
        return np.array(self._hold_target[key], dtype=np.float64) if kind == "needle" else self._pins.target(key)

    def grabbed_point(self, attachment):
        """Point of the needle that a needle hold grabbed, in the body frame [m]."""
        kind, key = attachment
        if kind != "needle":
            raise ValueError("Only a hold on the needle has a grabbed body point")
        return np.array(self._hold_local[key], dtype=np.float64)

    def commit(self, attachment):
        """Keep ``attachment``; returns its index."""
        self.attachments.append(attachment)
        return len(self.attachments) - 1

    def release(self, attachment):
        kind, key = attachment
        if kind == "needle":
            self._hold_active[key] = 0
            self._upload_holds()
        else:
            self._pins.release(key)

    def clear(self):
        """Release every attachment, including one that is being dragged."""
        self.cancel()
        for attachment in self.attachments:
            self.release(attachment)
        self.attachments.clear()

    def _upload_holds(self):
        self._active_device.assign(self._hold_active)
        self._local_device.assign(self._hold_local)
        self._target_device.assign(self._hold_target)

    # ScreenPickTool backend.

    def begin(self, origin, direction):
        """Pick the needle or a thread node under the ray; returns the grabbed world point or None."""
        self.cancel()
        # Two small readbacks per click, for the initial pick only: the needle's pose and the thread's nodes.
        pose = self._current_state().body_q.numpy()[self.body]
        rotation = np.array(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
        needle = self._centerline @ rotation.T + pose[:3]
        nodes, free = self._pins.candidates()
        points = np.concatenate((needle, nodes))
        allowed = np.concatenate((np.ones(len(needle), dtype=bool), free))
        nearest = nearest_to_ray(self.viewer, points, allowed, origin, direction, self.click_radius)
        if nearest is None:
            return None
        if nearest < len(needle):
            self._dragged = self.hold_needle(points[nearest], pose)
        else:
            self._dragged = self.pin_node(nearest - len(needle), points[nearest])
        return None if self._dragged is None else points[nearest]

    def move(self, target):
        if self._dragged is not None:
            self.set_target(self._dragged, target)

    def finish(self):
        if self._dragged is not None:
            self.commit(self._dragged)
            self._dragged = None

    def cancel(self):
        if self._dragged is not None:
            self.release(self._dragged)
            self._dragged = None

    def remove(self, index):
        """Release committed attachment ``index`` and move the last one into its place."""
        if not 0 <= index < len(self.attachments):
            return False
        self.release(self.attachments[index])
        self.attachments[index] = self.attachments[-1]
        self.attachments.pop()
        return True

    # Simulation; both only launch kernels, so a CUDA graph can capture them.

    def apply(self, state):
        """Add the needle holds' force to ``state.body_f``; call every substep after clearing forces."""
        wp.launch(
            _hold_forces,
            dim=1,
            inputs=[
                state.body_q,
                state.body_qd,
                self.model.body_com,
                self.body,
                self._active_device,
                self._local_device,
                self._target_device,
                self.stiffness,
                self._damping,
                self._spin_damping,
                self._support,
            ],
            outputs=[state.body_f],
            device=self.model.device,
        )

    def follow(self, fraction):
        """Move the pins' prescribed positions ``fraction`` of the way to their targets; call before every thread step.

        With ``fraction = 1 / (substeps left in the frame)`` the positions move
        in equal parts and arrive with the frame's last substep.
        """
        self._pins.follow(fraction)
