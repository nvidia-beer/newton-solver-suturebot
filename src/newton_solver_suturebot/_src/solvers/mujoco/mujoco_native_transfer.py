# SPDX-License-Identifier: Apache-2.0
"""Persistent, packed transfers around native articulated MuJoCo integration.

Newton's pinned-version coordinate/control conversion kernels still define
the mapping. They write views of one packet instead of allocating five device
arrays and synchronizing five independent readbacks for each rigid trial.
Native contact integration remains on the CPU; only its push/pull is captured.
"""

from types import SimpleNamespace

import warp as wp

from ...utils.capture import ScopedCapture
from ..fem.differentiation import active_tape


def _storage(array):
    return None if array is None else (array.ptr, array.shape, array.dtype)


class NativeMuJoCoTransfer:
    """Batch unchanged Newton conversion kernels with reusable host storage.

    The caller must import before exporting every trial. The import waits for
    the device's current stream, including the preceding export from this
    instance; this keeps the pinned output packet alive until its copy ends.
    A stream change explicitly waits for that previous upload before reusing
    the host packet. Model notifications must invalidate this object.
    """

    @staticmethod
    def supports(solver, state_in, state_out, control, observables):
        eligible = (
            solver.use_mujoco_cpu
            and solver.model.world_count == 1
            and solver.mj_model.nmocap == 0
            and not solver.enable_sleeping
            and observables is None
            and state_out.body_qdd is None
            and state_out.body_parent_f is None
            and not getattr(solver.model, "requires_grad", False)
            and active_tape() is None
        )
        if not eligible:
            return False
        # Native general conversion owns all optional native force exports.
        # Refuse any populated MuJoCo state namespace, including fields added
        # by later compatible Newton versions, rather than leaving one stale.
        namespace = getattr(state_out, "mujoco", None)
        if namespace is not None and any(value is not None for value in vars(namespace).values()):
            return False
        arrays = tuple(
            getattr(state, name, None)
            for state in (state_in, state_out)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f")
        )
        if control is not None:
            arrays += (
                control.joint_f,
                control.joint_target_q,
                control.joint_target_qd,
                getattr(getattr(control, "mujoco", None), "ctrl", None),
            )
        return not any(array is not None and array.requires_grad for array in arrays)

    def __init__(self, solver):
        self.solver, self.device = solver, solver.model.device
        native = solver.mj_model
        sizes = (native.nq, native.nv, native.nu, native.nv, native.nbody * 6)
        self.input = wp.empty(sum(sizes), dtype=float, device=self.device)
        self.output = wp.empty(native.nq + native.nv, dtype=float, device=self.device)
        self.host_input = wp.empty(self.input.size, dtype=float, device="cpu", pinned=self.device.is_cuda)
        self.host_output = wp.empty(self.output.size, dtype=float, device="cpu", pinned=self.device.is_cuda)
        self.host_input_view, self.host_output_view = self.host_input.numpy(), self.host_output.numpy()
        self.inputs, self.outputs = SimpleNamespace(nworld=1), SimpleNamespace(nworld=1)
        self.host_fields = {}
        offset = 0
        for name, count in zip(("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied"), sizes, strict=True):
            dtype = wp.spatial_vector if name == "xfrc_applied" else float
            width = 6 if name == "xfrc_applied" else 1
            if count:
                view = self.input[offset : offset + count]
                if width > 1:
                    view = view.reshape((count // width, width)).view(dtype)
                view = view.reshape((1, count // width))
            else:
                view = wp.empty((1, 0), dtype=dtype, device=self.device)
            setattr(self.inputs, name, view)
            self.host_fields[name] = (
                self.host_input_view[offset : offset + count].reshape((-1, width))
                if width > 1
                else self.host_input_view[offset : offset + count]
            )
            offset += count
        self.outputs.qpos = self.output[: native.nq].reshape((1, native.nq))
        self.outputs.qvel = self.output[native.nq :].reshape((1, native.nv))
        self.graphs = {"push": {}, "pull": {}, "wrench": {}}
        self.warmed = {"push": False, "pull": False, "wrench": False}
        self.captures = self.launches = self.uploads = self.readbacks = 0
        self._last_stream = None
        self._repeated_input = None
        self._repeated_active = False
        self.repeated_readbacks = 0

    def _execute(self, phase, key, operation, keepalive):
        if not self.device.is_cuda:
            operation()
            return
        if not self.warmed[phase]:
            operation()
            self.warmed[phase] = True
            return
        cache = self.graphs[phase]
        entry = cache.get(key)
        if entry is None:
            with ScopedCapture(device=self.device) as capture:
                operation()
            if len(cache) >= 2:
                cache.clear()
            entry = (capture.graph, keepalive)
            cache[key] = entry
            self.captures += 1
        wp.capture_launch(entry[0])
        self.launches += 1

    def begin_repeated_input(self, state, control, body):
        """Reuse immutable accepted coordinates/controls within one transaction.

        Only ``state.body_f[body]`` may change until ``end_repeated_input``.
        The caller still restores native integration history before each trial.
        No value cache survives the transaction, including failed/retried steps.
        """
        from .mujoco_repeated_input import RepeatedNativeInput

        if self._repeated_active:
            raise RuntimeError("Native repeated-input transactions cannot be nested")
        if self._repeated_input is None or self._repeated_input.body != body:
            self._repeated_input = RepeatedNativeInput(self, body)
        self._repeated_input.begin(state, control)
        self._repeated_active = True
        return self

    def end_repeated_input(self):
        self._repeated_active = False
        if self._repeated_input is not None:
            self._repeated_input.end()

    def push(self, state, control):
        current_stream = self.device.stream if self.device.is_cuda else None
        if self._last_stream is not None and self._last_stream != current_stream:
            wp.synchronize_stream(self._last_stream)
        self._last_stream = current_stream
        if self._repeated_active:
            self._repeated_input.push(state, control)
        else:
            self._push_full(state, control)

    def _push_full(self, state, control):
        solver = self.solver
        controls = (
            ()
            if control is None
            else (
                control.joint_f,
                control.joint_target_q,
                control.joint_target_qd,
                getattr(getattr(control, "mujoco", None), "ctrl", None),
            )
        )
        arrays = (state.joint_q, state.joint_qd, state.body_f, *controls)

        def convert():
            # The general adapter creates fresh zero arrays on each import.
            # Clear the packet to preserve its unmapped/absent-control values.
            self.input.zero_()
            solver._apply_mjc_control(solver.model, state, control, self.inputs)
            solver._update_mjc_data(self.inputs, solver.model, state)
            wp.copy(self.host_input, self.input)

        self._execute("push", tuple(_storage(array) for array in arrays), convert, (state, control, arrays))
        wp.synchronize_stream(self.device)
        for name, host_view in self.host_fields.items():
            getattr(solver.mj_data, name)[:] = host_view
        self.readbacks += 1

    def pull(self, state_in, state_out):
        solver, native = self.solver, self.solver.mj_model
        self.host_output_view[: native.nq] = solver.mj_data.qpos
        self.host_output_view[native.nq :] = solver.mj_data.qvel
        arrays = (
            state_in.joint_q,
            state_in.joint_qd,
            state_out.joint_q,
            state_out.joint_qd,
            state_out.body_q,
            state_out.body_qd,
        )

        def convert():
            wp.copy(self.output, self.host_output)
            solver._update_newton_state(solver.model, state_out, self.outputs, state_prev=state_in)

        self._execute("pull", tuple(_storage(array) for array in arrays), convert, (state_in, state_out, arrays))
        self.uploads += 1
