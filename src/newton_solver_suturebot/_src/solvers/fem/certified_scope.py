# SPDX-License-Identifier: Apache-2.0
"""Guarded, attempt-local activation of an explicit current-A certificate."""

import warp as wp

from .differentiation import active_tape


@wp.kernel(enable_backward=False)
def append_certificate(packet: wp.array[wp.float64], status: wp.array[int], output: wp.array[wp.float64]):
    for i in range(packet.shape[0]):
        output[i] = packet[i]
    output[packet.shape[0]] = wp.float64(status[0])


@wp.kernel(enable_backward=False)
def write_certificate(status: wp.array[int], output: wp.array[wp.float64]):
    output[output.shape[0] - 1] = wp.float64(status[0])


@wp.kernel(enable_backward=False)
def require_certificate(status: wp.array[int], valid: wp.array[int]):
    if status[0] == 0:
        valid[0] = 0


class CertifiedFEMScope:
    """Only the supported native, compact, forward-only transaction opts in.

    Unconfigured/gradient/custom solver uses the original code. A scope owns a
    stable status array; retained FEM/force graphs retain this same owner. The
    status is reset before every attempt and overwritten by every certified
    solve. Failed attempts do not retain an enabled scope or physical changes.
    """

    def __init__(self, solver):
        self.solver = solver
        self.device = solver.model.device
        self.status = wp.ones(1, dtype=int, device=self.device)
        self.active = False
        self.rejections = 0
        self.attempts = 0
        self._packets = {}

    def forward_supported(self, coupler, state, control):
        from ..coupled.coupling import NeedleTissueCoupler
        from ..mujoco.experimental_rigid_backend import supported_subcycled
        from .solver import SolverFEM

        s, model, native = self.solver, self.solver.model, coupler.rigid_solver
        if (
            type(s) is not SolverFEM
            or coupler.fem_solver is not s
            or not supported_subcycled(native)
            or native.model is not model
            or getattr(s.step, "__func__", None) is not SolverFEM.step
            or getattr(s.implicit_integration, "__func__", None) is not SolverFEM.implicit_integration
            or any(
                getattr(getattr(coupler, name), "__func__", None) is not getattr(NeedleTissueCoupler, name)
                for name in ("_step_fem", "_evaluate")
            )
            or s.differentiable
            or getattr(model, "requires_grad", False)
            or active_tape() is not None
            or s.solver_type != "cg"
            or s.preconditioner_type != "block_diag"
            or s.linear_solver_maxiter != 160
            or s.linear_solver_check_every != 0
            or not s.linear_solver_compact
            or not s.linear_solver_use_cuda_graph
            or s._streaming is not None
            or s._component_scaling is not None
            or model.world_count != 1
            or s._fixed_particles is None
            or s._fixed_particles.free.size == 0
            or s.A_bsr.row_counts is not None
            or any(
                getattr(a, "requires_grad", False)
                for a in (
                    s.A_bsr.values,
                    s._pc_inv_diag,
                    s._normalized_rhs,
                    s.dv,
                    s._fixed_particles.reference,
                    getattr(s._fixed_particles, "compact_rhs", None),
                    getattr(s._fixed_particles, "compact_increment", None),
                    getattr(s._fixed_particles, "compact_diagonal", None),
                    getattr(s._fixed_particles.compact_matrix, "values", None),
                )
            )
            or control is None
        ):
            return False
        for value in (state, coupler._candidate):
            for name in (
                "particle_q",
                "particle_qd",
                "particle_f",
                "body_q",
                "body_qd",
                "body_f",
                "joint_q",
                "joint_qd",
            ):
                if getattr(getattr(value, name, None), "requires_grad", False):
                    return False
        for name in ("joint_f", "joint_target_q", "joint_target_qd", "tet_activations"):
            if getattr(getattr(control, name, None), "requires_grad", False):
                return False
        return not getattr(getattr(getattr(control, "mujoco", None), "ctrl", None), "requires_grad", False)

    def begin_attempt(self, coupler, state, control):
        self.active = False
        if self.device.is_capturing:
            raise RuntimeError("Prepare certificate scope before capture")
        self.active = (
            self.device.is_cuda
            and wp.is_conditional_graph_supported()
            and self.forward_supported(coupler, state, control)
        )
        if self.active:
            fixed = self.solver._fixed_particles
            if fixed.prepare(self.solver.A_bsr, values_version=self.solver._projection_values_version()):
                self.solver._lin_solver = None
                self.solver._lin_solver_key = None
            self.active = fixed.compact_operator is not None
        # A supported attempt must actually execute a certified solve before it
        # can accept. Inactive scopes preserve the original solver path.
        self.status.fill_(0 if self.active else 1)
        self.attempts += int(self.active)

    def end_attempt(self):
        self.active = False

    def key(self):
        return self.active, self.status.ptr

    def reserve(self, size):
        """Allocate, ahead of its first use, the output of :meth:`pack` for a packet of ``size`` entries."""
        if size + 1 not in self._packets:
            self._packets[size + 1] = wp.empty(size + 1, dtype=wp.float64, device=self.device)

    def pack(self, packet):
        size = len(packet) + 1
        output = self._packets.get(size)
        if output is None:
            output = wp.empty(size, dtype=wp.float64, device=self.device)
            self._packets[size] = output
        wp.launch(append_certificate, 1, [packet, self.status, output], device=self.device)
        return output

    def check(self, value):
        if value != 1:
            from ..coupled.coupling import NeedleCouplingNotConverged

            self.rejections += 1
            raise NeedleCouplingNotConverged("FEM current-matrix residual failed its configured linear tolerance")
