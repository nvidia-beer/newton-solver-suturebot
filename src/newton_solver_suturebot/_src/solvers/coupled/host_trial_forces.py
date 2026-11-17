# SPDX-License-Identifier: Apache-2.0
"""Captured force evaluations between native articulated MuJoCo trials.

MuJoCo remains on the CPU. Established-tunnel geometry, physical force laws and
residual reductions run together on CUDA. Conservative boundary certificates
request a complete transactional host fallback; acceptance still uses exact
native endpoints and the original swept-event checks.

One graph is kept for each variant of an evaluation: whether the surface recorder
is bound, whether the residual of the trial before enters the Aitken sums, and
whether the candidate is new. A variant is captured at its first use;
``capture_graphs`` captures and launches all of them ahead of the first step.
"""

import warp as wp

from ...utils.capture import ScopedCapture, note_release
from .coupling import _interface_residual
from .coupling_graph import finite_candidate
from .interface.needle_trial_forces import _NeedleTrialForces


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
        self.certificate = getattr(coupler.fem_solver, "_certified_linear", None)
        result_size += int(self.certificate is not None)
        self.result = wp.zeros(result_size, dtype=wp.float64, device=self.device)
        self.forces_base = wp.empty_like(coupler._forces)
        self.body_f_base = wp.empty_like(coupler._candidate.body_f)
        self.diagnostics_base = wp.empty_like(coupler.interaction._diagnostics)
        self.contact_flags_base = wp.empty_like(coupler.interaction._contact_flags)
        self.primal_reuses = 0
        # {(surface recorder bound, residual of the trial before, new candidate): graph}, and for
        # each of the two recorder states the signature its graphs were captured for and the
        # arrays they address. ``signature`` is the one of the state prepared last.
        self.graphs = {}
        self.signatures = {}
        self.signature = None
        self._arrays = {}
        self.captures = self.launches = self.fallbacks = 0

    def prepare(self, dt):
        c = self.c
        # Host preparation, outside capture. The time step is not part of the
        # signature below: the force kernels of every graph read it from these arrays.
        c.interaction.step_time.set(dt)
        self.permissions.prepare(c.interaction)
        from .interface.shaft_projection import projection_binding

        projection = projection_binding(
            c.interaction,
            c._candidate,
            c.needle_body,
            float(c.interaction.radii.max()),
            operands=(c._forces, c._residual, c._value, c._trial_load),
        )
        self._projection_binding = projection
        arrays = tuple(
            getattr(c._candidate, name) for name in ("particle_q", "particle_qd", "body_q", "body_qd", "body_f")
        ) + (c._forces, c._trial_load, c._previous_residual, c._value, c._residual, c._statistics)
        arrays += () if projection is None else projection.owners
        surface_pc = getattr(c, "_surface_preconditioner", None)
        surface_active = surface_pc is not None and surface_pc.available
        if surface_active:
            arrays += (surface_pc.valid, surface_pc.result)
        signature = (
            None if projection is None else projection.key,
            None if self.certificate is None else self.certificate.key(),
            surface_active,
            c.needle_body,
            tuple(c.needle_com),
            c.moment_length,
            self.permissions.signature(),
            tuple((a.ptr, a.shape, a.dtype) for a in arrays),
        )
        if signature != self.signatures.get(surface_active):
            # Only the graphs of this recorder state were captured for the signature that
            # changed. Going from one recorder state to the other drops nothing.
            dropped = [key for key in self.graphs if key[0] == surface_active]
            for key in dropped:
                del self.graphs[key]
            note_release(len(dropped))
            self.signatures[surface_active] = signature
            self._arrays[surface_active] = arrays
        self.signature = signature

    def _graph(self, dt, has_previous, fresh):
        """The graph of one variant for the prepared signature; captured here at its first use."""
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

        key = (surface_pc is not None, bool(has_previous), bool(fresh))
        graph = self.graphs.get(key)
        if graph is None:
            with ScopedCapture(device=self.device) as capture:
                self.finite.fill_(1)
                wp.launch(
                    finite_candidate,
                    max(c.model.particle_count, c.model.body_count),
                    [state.particle_q, state.particle_qd, state.body_q, state.body_qd, self.finite],
                    device=self.device,
                )
                if self.certificate is not None and self.certificate.active:
                    from ..fem.certified_scope import require_certificate

                    wp.launch(require_certificate, 1, [self.certificate.status, self.finite], device=self.device)
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
                if self.certificate is not None:
                    from ..fem.certified_scope import write_certificate

                    wp.launch(write_certificate, 1, [self.certificate.status, self.result], device=self.device)
            graph = capture.graph
            self.graphs[key] = graph
            self.captures += 1
        return graph

    def evaluate(self, dt, has_previous, *, fresh=True):
        c = self.c
        surface_pc = getattr(c, "_surface_preconditioner", None)
        surface_pc = surface_pc if surface_pc is not None and surface_pc.available else None
        # The force kernels read the time step from device data. Write it before the
        # capture and before every replay; this is a comparison within one coupled step.
        c.interaction.step_time.set(dt)
        graph = self._graph(dt, has_previous, fresh)
        wp.capture_launch(graph)
        self.launches += 1
        self.primal_reuses += int(not fresh)
        result = self.result.numpy()
        c._invalidate_query_cache()
        if self.certificate is not None and self.certificate.active:
            self.certificate.check(result[-1])
        if not result[7]:
            from .coupling import NeedleCouplingNotConverged

            raise NeedleCouplingNotConverged("Nonfinite needle/tissue interface state or force")
        if result[6]:
            self.fallbacks += 1
            raise TrialForceFallback()
        if surface_pc is not None:
            # Preserve the coupler's six original scalars followed by the two
            # recorder scalars, using the one completed-graph host transfer.
            return result[[0, 1, 2, 3, 4, 5, 8, 9]]
        return result[:6]

    def capture_graphs(self, state_in, control, dt):
        """Capture and launch once, ahead of the first step, every variant that has no graph.

        For start-up, inside an attempt of the coupler that ends like a rejected step: the
        multipliers are open and the certificate is as in a host-controlled trial, and the
        candidate holds a finite state. For the recorder bound (with a surface
        preconditioner whose recording is supported) and not bound, the four variants are
        captured as :meth:`evaluate` captures them, the ones with a new candidate first, so
        that the others copy forces that were written. Each graph is launched once, which
        builds its executable. Nothing is read back and nothing is decided from a launch:
        ``launches``, ``primal_reuses`` and ``fallbacks`` keep their values. What the launches
        write is trial scratch, which every trial writes again, and what the close of the
        attempt discards. The recorder is left as the begin phase of an attempt leaves it.

        Args:
            state_in: Accepted state of the attempt.
            control: Control of the attempt.
            dt: Time step of the attempt [s].
        """
        c = self.c
        surface_pc = getattr(c, "_surface_preconditioner", None)
        c._previous_residual.zero_()
        for surface in (False,) if surface_pc is None else (True, False):
            if surface_pc is not None:
                surface_pc.begin_step()
                if surface and not surface_pc.begin_trials(state_in, control, include_history=True):
                    # No evaluation records for this configuration: it has no such variant.
                    continue
            self.prepare(dt)
            for has_previous, fresh in ((False, True), (True, True), (False, False), (True, False)):
                if (surface, has_previous, fresh) not in self.graphs:
                    wp.capture_launch(self._graph(dt, has_previous, fresh))
        if surface_pc is not None:
            surface_pc.begin_step()
        c._invalidate_query_cache()

    def stage(self):
        c = self.c
        if not c.interaction._stage_certified_trial(c._candidate, c.needle_body, c.needle_com, self.permissions.mask):
            self.fallbacks += 1
            raise TrialForceFallback()
