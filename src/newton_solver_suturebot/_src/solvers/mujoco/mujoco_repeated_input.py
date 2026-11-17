# SPDX-License-Identifier: Apache-2.0
"""Reuse an accepted native input while downloading only its trial needle load."""

import numpy as np
import warp as wp
from newton._src.solvers.mujoco.kernels import (
    apply_mjc_body_f_kernel,
    apply_mjc_free_joint_f_to_body_f_kernel,
)


class RepeatedNativeInput:
    """An explicit, single-transaction cache; never infer reuse from addresses.

    The first trial runs the complete upstream float32 conversion. Later trials
    restore that same converted packet on the CPU and replace only the varying
    body's wrench. The two original Newton force kernels preserve gravity and
    free-joint-force additions, including their float32 evaluation order.
    Native integration, rollback, controls and contact solves are unchanged.
    """

    def __init__(self, transfer, body):
        self.transfer, self.body = transfer, body
        solver = transfer.solver
        matches = np.flatnonzero(solver.mjc_body_to_newton.numpy()[0] == body)
        if len(matches) != 1:
            raise ValueError("Repeated native input requires one mapped rigid body")
        self.native_body = int(matches[0])
        self.mapping = solver.mjc_body_to_newton[:, self.native_body : self.native_body + 1]
        self.gravcomp = solver.mjw_model.body_gravcomp[:, self.native_body : self.native_body + 1]
        self.wrench = wp.empty((1, 1), dtype=wp.spatial_vector, device=transfer.device)
        self.host_wrench = wp.empty((1, 1), dtype=wp.spatial_vector, device="cpu", pinned=transfer.device.is_cuda)
        self.host_view = self.host_wrench.numpy()[0, 0]
        self.host_transaction = None
        self.host_token = None
        self.host_reuses = 0
        self.end()

    def begin(self, state, control):
        self.end()
        self.state, self.control = state, control
        self.ready = False

    def end(self):
        if self.host_transaction is not None:
            self.host_transaction.close()
        self.host_transaction = self.host_token = None
        self.state = self.control = None
        self.ready = False

    def enable_host_wrench(self, external, load, moment_length):
        from ..coupled.host_wrench import HostWrenchTransaction, supported_kernels
        from ..fem.differentiation import active_tape

        solver = self.transfer.solver
        arrays = (
            external,
            load,
            solver.model.body_flags,
            solver.model.body_mass,
            solver.model.body_world,
            solver.model.gravity,
            self.gravcomp,
            solver.body_free_qd_start,
            None if self.control is None else self.control.joint_f,
        )
        if (
            not self.transfer.device.is_cuda
            or not supported_kernels()
            or self.ready
            or active_tape() is not None
            or any(a is not None and a.requires_grad for a in arrays)
        ):
            return None
        self.host_transaction = HostWrenchTransaction(self, external, load, moment_length)
        return self.host_transaction

    def push(self, state, control):
        if state is not self.state or control is not self.control:
            raise RuntimeError("Repeated native input must use its original accepted state and held control")
        transfer, solver = self.transfer, self.transfer.solver
        token, self.host_token = self.host_token, None
        if not self.ready:
            transfer._push_full(state, control)
            self.ready = True
            return
        host = None if self.host_transaction is None else self.host_transaction.consume(token)
        if host is not None:
            # The Anderson packet read already fenced the preceding endpoint
            # upload. Restore every original converted input before this trial.
            for name, values in transfer.host_fields.items():
                getattr(solver.mj_data, name)[:] = values
            solver.mj_data.xfrc_applied[self.native_body] = host
            self.host_reuses += 1
            return
        joint_f = None if control is None else control.joint_f

        def convert_wrench():
            wp.launch(
                apply_mjc_body_f_kernel,
                (1, 1),
                [
                    self.mapping,
                    solver.model.body_flags,
                    state.body_f,
                    solver.model.body_mass,
                    solver.model.body_world,
                    solver.model.gravity,
                    self.gravcomp,
                ],
                [self.wrench],
                device=transfer.device,
            )
            if joint_f is not None:
                wp.launch(
                    apply_mjc_free_joint_f_to_body_f_kernel,
                    (1, 1),
                    [self.mapping, solver.model.body_flags, solver.body_free_qd_start, joint_f],
                    [self.wrench],
                    device=transfer.device,
                )
            wp.copy(self.host_wrench, self.wrench)

        arrays = (state.body_f, joint_f)
        key = (self.body, *(None if array is None else (array.ptr, array.shape, array.dtype) for array in arrays))
        transfer._execute("wrench", key, convert_wrench, (self, arrays))
        wp.synchronize_stream(transfer.device)
        # host_fields retain the first full import; the wrench-only readback
        # uses separate storage. Re-importing these float32 values is required:
        # the native rollback checkpoint instead holds double coordinates.
        for name, host_view in transfer.host_fields.items():
            getattr(solver.mj_data, name)[:] = host_view
        solver.mj_data.xfrc_applied[self.native_body] = self.host_view
        transfer.readbacks += 1
        transfer.repeated_readbacks += 1
