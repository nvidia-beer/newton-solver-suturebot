# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compact public-API bridge for one world FREE body with its origin at COM.

The native MuJoCo model, integrator and integration history are unchanged.
Unsupported models continue through Newton's general SolverMuJoCo adapter.
"""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo


@wp.kernel(enable_backward=False)
def _pack(
    q: wp.array[float],
    qd: wp.array[float],
    body_f: wp.array[wp.spatial_vector],
    joint_f: wp.array[float],
    packet: wp.array[float],
):
    rotation = wp.quat(q[3], q[4], q[5], q[6])
    angular = wp.quat_rotate_inv(rotation, wp.vec3(qd[3], qd[4], qd[5]))
    for axis in range(3):
        packet[axis] = q[axis]
        packet[4 + axis] = q[3 + axis]
        packet[7 + axis] = qd[axis]
        packet[10 + axis] = angular[axis]
    packet[3] = q[6]
    for axis in range(6):
        packet[13 + axis] = body_f[0][axis] + joint_f[axis]


@wp.kernel(enable_backward=False)
def _unpack(
    packet: wp.array[float],
    q: wp.array[float],
    qd: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    position = wp.vec3(packet[0], packet[1], packet[2])
    rotation = wp.quat(packet[4], packet[5], packet[6], packet[3])
    linear = wp.vec3(packet[7], packet[8], packet[9])
    angular = wp.quat_rotate(rotation, wp.vec3(packet[10], packet[11], packet[12]))
    for axis in range(3):
        q[axis] = position[axis]
        q[3 + axis] = rotation[axis]
        qd[axis] = linear[axis]
        qd[3 + axis] = angular[axis]
    q[6] = rotation[3]
    body_q[0] = wp.transform(position, rotation)
    body_qd[0] = wp.spatial_vector(linear, angular)


class MuJoCoFreeBody:
    """Reuse an existing native solver, batching its coordinate/force transfers.

    The supported coordinate frames are deliberately narrow: one dynamic world
    FREE joint, identity anchors, zero COM offset and zero gravity. Actuators,
    extra joints/worlds, custom step methods and rigid observables use the
    general adapter. Forces remain world COM wrenches, including joint loads;
    MuJoCo's local angular velocity is converted in float32 just as in Newton.
    """

    @staticmethod
    def supports(model, solver):
        if (
            type(solver) is not SolverMuJoCo
            or getattr(solver.step, "__func__", None) is not SolverMuJoCo.step
            or solver.model is not model
            or not solver.use_mujoco_cpu
            or solver.update_data_interval != 1
            or model.body_count != 1
            or model.joint_count != 1
            or model.joint_coord_count != 7
            or model.joint_dof_count != 6
            or model.world_count != 1
        ):
            return False
        native = solver.mj_model
        identity = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        return bool(
            native.nbody == 2
            and native.nq == 7
            and native.nv == 6
            and native.nu == 0
            and model.joint_type.numpy()[0] == int(newton.JointType.FREE)
            and model.joint_parent.numpy()[0] == -1
            and model.joint_child.numpy()[0] == 0
            and model.body_flags.numpy()[0] == int(newton.BodyFlags.DYNAMIC)
            and np.array_equal(model.joint_X_p.numpy()[0], identity)
            and np.array_equal(model.joint_X_c.numpy()[0], identity)
            and not np.any(model.body_com.numpy())
            and not np.any(model.gravity.numpy())
            and not np.any(native.opt.gravity)
        )

    def __init__(self, model, solver):
        if not self.supports(model, solver):
            raise ValueError("Compact MuJoCo bridge requires one zero-COM world FREE body")
        self.model, self.solver = model, solver
        self._solver_step = solver.step
        self.input = wp.empty(19, dtype=float, device=model.device)
        self.output = wp.empty(13, dtype=float, device=model.device)
        self.host_output = np.empty(13, dtype=np.float32)
        self.zero_joint_force = wp.zeros(6, dtype=float, device=model.device)

    def step(self, state_in, state_out, control, dt):
        # Extended observables require Newton's general conversion. Never leave
        # a requested acceleration or transmitted force silently stale.
        if (
            self.solver.step != self._solver_step
            or any(getattr(state_out, name, None) is not None for name in ("body_qdd", "body_parent_f"))
            or getattr(getattr(state_out, "mujoco", None), "qfrc_actuator", None) is not None
        ):
            self.solver.step(state_in, state_out, control, None, dt)
            return
        joint_f = self.zero_joint_force if control is None or control.joint_f is None else control.joint_f
        wp.launch(
            _pack,
            1,
            [state_in.joint_q, state_in.joint_qd, state_in.body_f, joint_f, self.input],
            device=self.model.device,
        )
        packet = self.input.numpy()
        native, data = self.solver.mj_model, self.solver.mj_data
        data.qpos[:] = packet[:7]
        data.qvel[:] = packet[7:13]
        data.xfrc_applied.fill(0)
        data.xfrc_applied[1] = packet[13:]
        data.qfrc_applied.fill(0)
        native.opt.timestep = dt
        import mujoco  # noqa: PLC0415

        mujoco.mj_step(native, data)
        self.host_output[:7], self.host_output[7:] = data.qpos, data.qvel
        self.output.assign(self.host_output)
        wp.launch(
            _unpack,
            1,
            [self.output, state_out.joint_q, state_out.joint_qd, state_out.body_q, state_out.body_qd],
            device=self.model.device,
        )
