# SPDX-License-Identifier: Apache-2.0
"""Experimental, one-use host witness for an already uploaded interface load.

This is an import optimization, not a force approximation. The CUDA kernels
remain the fallback. Their pinned-version PTX uses FP64 add/FMA followed by
FP32 conversion, FP32 gravity FMA, then a separate FP32 free-joint addition.
The standard C math library reproduces those roundings without a GPU wait.
"""

import ctypes
import threading
from dataclasses import dataclass

import numpy as np
import warp as wp

try:
    _LIBM = ctypes.CDLL("libm.so.6")
    _FMA = _LIBM.fma
    _FMA.argtypes = (ctypes.c_double,) * 3
    _FMA.restype = ctypes.c_double
    _FMAF = _LIBM.fmaf
    _FMAF.argtypes = (ctypes.c_float,) * 3
    _FMAF.restype = ctypes.c_float
    _ROUND = _LIBM.fegetround
    _ROUND.argtypes = ()
    _ROUND.restype = ctypes.c_int
except (OSError, AttributeError):
    _LIBM = _FMA = _FMAF = _ROUND = None


def supported_math():
    """The tested Linux contract uses round-to-nearest, no fast-math/FTZ."""
    return _ROUND is not None and _ROUND() == 0


def supported_kernels():
    from . import coupling, host_trial_prepare

    full, rigid = coupling._HOST_WRENCH_INTERFACE_KERNELS
    unchanged = (
        coupling._apply_interface_load is full
        and coupling._apply_rigid_interface_load is rigid
        and host_trial_prepare._apply_interface_load is full
    )
    return (
        unchanged
        and supported_math()
        and all(
            not options.get("fast_math", False) and options.get("fuse_fp", True)
            for options in (
                wp.get_module_options("newton_solver_suturebot._src.solvers.coupled.coupling"),
                wp.get_module_options("newton._src.solvers.mujoco.kernels"),
            )
        )
    )


def stream_id(device):
    return int(device.stream.cuda_stream) if device.is_cuda else None


@dataclass
class HostLoadWitness:
    """Six exact FP64 rows, with the stream fence preceding their upload.

    Only DeviceSurfaceAnderson creates these after its existing packet.numpy
    read, and only the immediately following unmodified trial can use one.
    A witness is consumed even if a contract check fails.
    """

    load: object
    values: np.ndarray
    stream: int | None
    thread: object
    scope: object
    generation: int
    used: bool = False

    @classmethod
    def from_sparse(cls, load, indices, proposal, scope):
        values = np.zeros(6, dtype=np.float64)
        selected = indices >= load.size - 6
        values[indices[selected] - (load.size - 6)] = proposal[selected]
        values.flags.writeable = False
        return cls(load, values, stream_id(load.device), threading.current_thread(), scope, scope.generation)


@wp.kernel(enable_backward=False)
def snapshot_constants(
    body: int,
    external: wp.array[wp.spatial_vector],
    body_flags: wp.array[int],
    body_mass: wp.array[float],
    body_world: wp.array[int],
    gravity: wp.array[wp.vec3],
    gravcomp: wp.array2d[float],
    free_start: wp.array[int],
    joint_f: wp.array[float],
    packet: wp.array[float],
):
    # A separate snapshot precedes the FIRST full import. Its existing stream
    # wait also completes this copy; no additional readback wait is introduced.
    for axis in range(6):
        packet[axis] = external[body][axis]
        packet[14 + axis] = 0.0
    packet[6] = body_mass[body]
    packet[7] = gravcomp[0, 0]
    world = body_world[body]
    gravity_valid = world >= -1 and world < gravity.shape[0]
    for axis in range(3):
        packet[8 + axis] = 0.0
        packet[11 + axis] = 0.0
        if gravity_valid:
            packet[8 + axis] = gravity[world][axis]
            packet[11 + axis] = gravity[0][axis]
    packet[20] = float(gravity_valid)
    start = free_start[body]
    has_free = start >= 0 and joint_f.shape[0] > 0
    packet[21] = float(has_free)
    if has_free:
        for axis in range(6):
            packet[14 + axis] = joint_f[start + axis]
    # Newton BodyFlags.KINEMATIC == 2 in the supported version.
    packet[22] = float((body_flags[body] & 2) != 0)


def convert_wrench(constants, load, moment_length):
    """Return the original kernel chain's FP32 result, or None for fallback."""
    if (
        not supported_math()
        or constants.shape != (23,)
        or constants.dtype != np.float32
        or load.shape != (6,)
        or load.dtype != np.float64
        or not np.isfinite(constants).all()
        or not np.isfinite(load).all()
        or not np.isfinite(moment_length)
    ):
        return None
    if constants[22]:
        return np.zeros(6, dtype=np.float32)
    result = np.empty(6, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for axis in range(3):
            result[axis] = float(constants[axis]) + float(load[axis])
        for axis in range(3, 6):
            result[axis] = _FMA(float(moment_length), float(load[axis]), float(constants[axis]))
        if constants[20]:
            scale = np.float32(constants[6] * np.float32(np.float32(1.0) - constants[7]))
            for axis in range(3):
                delta = np.float32(constants[8 + axis] - constants[11 + axis])
                result[axis] = _FMAF(float(scale), float(delta), float(result[axis]))
        if constants[21]:
            result += constants[14:20]
    return result if np.isfinite(result).all() else None


class HostWrenchTransaction:
    """Owned constants and one-consume, next-push-only transaction tokens.

    Coordinates/controls still come from the first full FP32 native import.
    The caller must publish only immediately after an unchanged Anderson
    proposal and offer only immediately before its next native trial. No
    witness may survive another load writer, rollback or stream transition.
    """

    def __init__(self, repeated, external, load, moment_length):
        self.repeated, self.load = repeated, load
        self.moment_length = float(moment_length)
        self.thread = threading.current_thread()
        self.stream = stream_id(repeated.transfer.device)
        self.pending = None
        self.closed = False
        self.generation = 0
        self.constants = wp.empty(23, dtype=float, device=repeated.transfer.device)
        self.host = wp.empty(23, dtype=float, device="cpu", pinned=repeated.transfer.device.is_cuda)
        self.host_view = self.host.numpy()
        self.empty = wp.empty(0, dtype=float, device=repeated.transfer.device)
        solver, control = repeated.transfer.solver, repeated.control
        self.owners = (external, load, control, self.empty)
        self.bindings = self._bindings()
        wp.launch(
            snapshot_constants,
            1,
            [
                repeated.body,
                external,
                solver.model.body_flags,
                solver.model.body_mass,
                solver.model.body_world,
                solver.model.gravity,
                repeated.gravcomp,
                solver.body_free_qd_start,
                self.empty if control is None or control.joint_f is None else control.joint_f,
                self.constants,
            ],
            device=repeated.transfer.device,
        )
        wp.copy(self.host, self.constants)

    def _bindings(self):
        repeated = self.repeated
        solver, state, control = repeated.transfer.solver, repeated.state, repeated.control
        arrays = (
            self.load,
            state.body_f,
            state.joint_q,
            state.joint_qd,
            solver.model.body_flags,
            solver.model.body_mass,
            solver.model.body_world,
            solver.model.gravity,
            repeated.gravcomp,
            solver.body_free_qd_start,
            None if control is None else control.joint_f,
        )
        return tuple(None if a is None else (id(a), a.ptr, a.shape, a.dtype) for a in arrays)

    def invalidate(self):
        self.generation += 1
        if self.pending is not None:
            self.pending[0].used = True
        self.pending = None

    def publish(self, witness, next_iteration):
        """The witness's existing packet-read fence covers all prior exports."""
        if self.pending is not None:
            self.pending[0].used = True
        self.pending = None
        if witness is None:
            return
        transfer = self.repeated.transfer
        if (
            self.closed
            or witness.used
            or witness.scope is not self
            or witness.generation != self.generation
            or witness.load is not self.load
            or witness.thread is not self.thread
            or threading.current_thread() is not self.thread
            or witness.stream != self.stream
            or stream_id(transfer.device) != self.stream
        ):
            witness.used = True
            return
        self.pending = (witness, int(next_iteration), self.generation, transfer.uploads)

    def offer(self, iteration):
        """Arm only the immediately following push; the token is consumed there."""
        if self.pending is None:
            return None
        witness, expected, generation, uploads = self.pending
        self.pending = None
        if expected != iteration or generation != self.generation:
            witness.used = True
            return None
        return (self, witness, generation, uploads)

    def consume(self, token):
        if token is None:
            return None
        owner, witness, generation, uploads = token
        was_used, witness.used = witness.used, True
        transfer = self.repeated.transfer
        if (
            owner is not self
            or self.closed
            or was_used
            or not self.repeated.ready
            or not supported_kernels()
            or generation != self.generation
            or uploads != transfer.uploads
            or self.bindings != self._bindings()
            or threading.current_thread() is not self.thread
            or stream_id(transfer.device) != self.stream
            or transfer._last_stream != (transfer.device.stream if transfer.device.is_cuda else None)
        ):
            return None
        return convert_wrench(self.host_view, witness.values, self.moment_length)

    def close(self):
        self.invalidate()
        self.closed = True
