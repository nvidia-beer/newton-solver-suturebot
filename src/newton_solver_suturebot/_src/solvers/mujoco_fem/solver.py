# SPDX-License-Identifier: Apache-2.0
"""MuJoCo/FEM integration with simultaneous GPU momentum/friction correction."""

from ...coupling import NeedleTissueCoupler


class SolverMuJoCoFEM(NeedleTissueCoupler):
    """Simultaneous GPU solve with the existing physical-step transaction.

    The initial backend supports the validated single MuJoCo implicitfast free
    body and an assembled, non-differentiable FEM with a compliant material
    tunnel. It uses a nonsymmetric GMRES correction for momentum and Coulomb
    projection residuals together. Shaft geometry derivatives are frozen in the
    search matrix; actual endpoint geometry is reevaluated for acceptance.

    Inherits physical event staging, native endpoint audit and rollback from
    the existing integration driver. Unsupported configurations and puncture
    events retain that driver's established path. This is not a general GPU
    implementation of arbitrary articulated MuJoCo models or differentiation.
    """

    # Scheduling can change atomic summation order; keep this policy local to
    # the opt-in backend and preserve the established driver's launch sizes.
    _needle_query_block_dim = 1

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("capture_fem", True)
        kwargs.setdefault("capture_coupling", True)
        kwargs.setdefault("shaft_preconditioner", True)
        super().__init__(*args, **kwargs)
        self._simultaneous_trials = None
        self.simultaneous_steps = 0
        self.simultaneous_iterations = 0
        self._simultaneous_converged = False

    def step(self, state_in, state_out, control, dt, *, external_forces=None):
        self._simultaneous_converged = False
        super().step(state_in, state_out, control, dt, external_forces=external_forces)
        if self._simultaneous_converged:
            self.simultaneous_steps += 1

    def _device_trials(self, state_in, control, dt):
        # Reuse the existing eligibility contract for force geometry, rigid
        # settings and supported state layouts. No legacy trials are executed.
        eligible = super()._device_trials(state_in, control, dt)
        alm = getattr(self.interaction, "tunnel_alm", None)
        if eligible is None or alm is None or alm.compliance_per_m <= 0:
            return eligible
        from .trials import _SimultaneousTrials

        if self._simultaneous_trials is None:
            self._simultaneous_trials = _SimultaneousTrials(self)
        self._simultaneous_trials.rigid.prepare(dt)
        return self._simultaneous_trials

    def _solve_device(self, graph, state_in, control, dt, tunnel_alm):
        value = super()._solve_device(graph, state_in, control, dt, tunnel_alm)
        if graph is self._simultaneous_trials:
            self.simultaneous_iterations += int(graph.last_result[1])
            if value:
                self._simultaneous_converged = True
                self.coupling_graph_status = "simultaneous momentum/friction"
        return value

    def invalidate_fem_graph(self):
        super().invalidate_fem_graph()
        if self._simultaneous_trials is not None:
            self._simultaneous_trials.invalidate()
