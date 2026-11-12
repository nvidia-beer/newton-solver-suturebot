# SPDX-License-Identifier: Apache-2.0
"""Captured force evaluations between native articulated MuJoCo trials.

MuJoCo remains on the CPU. Established-tunnel geometry, physical force laws and
residual reductions run together on CUDA. Conservative boundary certificates
request a complete transactional host fallback; acceptance still uses exact
native endpoints and the original swept-event checks.
"""

import warp as wp

from .coupling import _interface_residual
from .coupling_graph import finite_candidate
from .needle_trial_forces import _NeedleTrialForces


class TrialForceFallback(Exception):
    """Restart the unchanged physical transaction through the host oracle."""


@wp.kernel(enable_backward=False)
def _pack_statistics(
    statistics: wp.array[wp.float64],
    fallback: wp.array[int],
    finite: wp.array[int],
    result: wp.array[wp.float64],
):
    for i in range(6):
        result[i] = statistics[i]
    result[6] = wp.float64(fallback[0])
    result[7] = wp.float64(finite[0])


@wp.kernel(enable_backward=False)
def _pack_surface_result(surface: wp.array[wp.float64], result: wp.array[wp.float64]):
    result[8] = surface[6]
    result[9] = surface[7]


class CapturedHostTrialForces:
    def __init__(self, coupler):
        self.c = coupler
        self.device = coupler.model.device
        self.permissions = _NeedleTrialForces(
            coupler.interaction, query_block_dim=getattr(coupler, "_needle_query_block_dim", None)
        )
        self.finite = wp.ones(1, dtype=int, device=self.device)
        result_size = 10 if getattr(coupler, "_surface_preconditioner", None) is not None else 8
        self.result = wp.zeros(result_size, dtype=wp.float64, device=self.device)
        self.forces_base = wp.empty_like(coupler._forces)
        self.body_f_base = wp.empty_like(coupler._candidate.body_f)
        self.diagnostics_base = wp.empty_like(coupler.interaction._diagnostics)
        self.contact_flags_base = wp.empty_like(coupler.interaction._contact_flags)
        self.primal_reuses = 0
        self.graphs = {}
        self.signature = None
        self.captures = self.launches = self.fallbacks = 0

    def prepare(self, dt):
        c = self.c
        self.permissions.prepare(c.interaction)
        arrays = tuple(
            getattr(c._candidate, name) for name in ("particle_q", "particle_qd", "body_q", "body_qd", "body_f")
        ) + (c._forces, c._trial_load, c._previous_residual, c._value, c._residual, c._statistics)
        surface_pc = getattr(c, "_surface_preconditioner", None)
        surface_active = surface_pc is not None and surface_pc.available
        if surface_active:
            arrays += (surface_pc.valid, surface_pc.result)
        signature = (
            surface_active,
            dt,
            c.needle_body,
            tuple(c.needle_com),
            c.moment_length,
            self.permissions.signature(),
            tuple((a.ptr, a.shape, a.dtype) for a in arrays),
        )
        if signature != self.signature:
            self.graphs = {}
            self.signature = signature
            self._arrays = arrays

    def evaluate(self, dt, has_previous, *, fresh=True):
        c, state = self.c, self.c._candidate
        surface_pc = getattr(c, "_surface_preconditioner", None)
        surface_pc = surface_pc if surface_pc is not None and surface_pc.available else None

        def forces():
            if fresh:
                c._forces.zero_()
                state.body_f.zero_()
                self.permissions.add_surface_forces(state, c.needle_body, c.needle_com, c._forces, dt)
                wp.copy(self.forces_base, c._forces)
                wp.copy(self.body_f_base, state.body_f)
                wp.copy(self.diagnostics_base, c.interaction._diagnostics)
                wp.copy(self.contact_flags_base, c.interaction._contact_flags)
            else:
                # A dual update changes only tunnel forces. Its candidate,
                # geometry and surface response have already been evaluated.
                wp.copy(c._forces, self.forces_base)
                wp.copy(state.body_f, self.body_f_base)
                wp.copy(c.interaction._diagnostics, self.diagnostics_base)
                wp.copy(c.interaction._contact_flags, self.contact_flags_base)
            self.permissions.add_history_forces(state, c.needle_body, c.needle_com, c._forces, dt)
            c._statistics.zero_()
            # These two Aitken sums are ignored at a fresh mixing start, and
            # by the preconditioned branch. Computing them is otherwise exact.
            wp.launch(
                _interface_residual,
                c._load.size,
                [
                    c._forces,
                    state.body_f,
                    c.needle_body,
                    c.moment_length,
                    c._trial_load,
                    c._previous_residual,
                    int(has_previous),
                    c._value,
                    c._residual,
                    c._statistics,
                ],
                device=self.device,
            )

        key = (bool(has_previous), bool(fresh))
        graph = self.graphs.get(key)
        if graph is None:
            with wp.ScopedCapture(device=self.device) as capture:
                self.finite.fill_(1)
                wp.launch(
                    finite_candidate,
                    max(c.model.particle_count, c.model.body_count),
                    [state.particle_q, state.particle_qd, state.body_q, state.body_qd, self.finite],
                    device=self.device,
                )
                wp.capture_if(self.finite, forces)
                wp.launch(
                    _pack_statistics,
                    1,
                    [c._statistics, self.permissions.fallback, self.finite, self.result],
                    device=self.device,
                )
                if surface_pc is not None:
                    packet = surface_pc.pack_statistics(c._statistics)
                    wp.launch(_pack_surface_result, 1, [packet, self.result], device=self.device)
            graph = capture.graph
            self.graphs[key] = graph
            self.captures += 1
        wp.capture_launch(graph)
        self.launches += 1
        self.primal_reuses += int(not fresh)
        result = self.result.numpy()
        c._invalidate_query_cache()
        if not result[7]:
            raise RuntimeError("Nonfinite needle/tissue interface state or force")
        if result[6]:
            self.fallbacks += 1
            raise TrialForceFallback()
        if surface_pc is not None:
            # Preserve the coupler's six original scalars followed by the two
            # recorder scalars, using the one completed-graph host transfer.
            return result[[0, 1, 2, 3, 4, 5, 8, 9]]
        return result[:6]

    def stage(self):
        c = self.c
        if not c.interaction._stage_certified_trial(c._candidate, c.needle_body, c.needle_com, self.permissions.mask):
            self.fallbacks += 1
            raise TrialForceFallback()
