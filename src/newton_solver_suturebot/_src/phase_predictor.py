# SPDX-License-Identifier: Apache-2.0
"""Optional initial interface guesses aligned with explicit control frames."""

import math

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _scales(
    current: wp.array[wp.float64],
    recent: wp.array[wp.float64],
    last: wp.array[wp.float64],
    older: wp.array[wp.float64],
    extrapolate: bool,
    valid: wp.array[int],
    scales: wp.array[wp.float64],
):
    i = wp.tid()
    a, b = last[i], older[i]
    if not wp.isfinite(a) or not wp.isfinite(current[i]) or current[i] != recent[i]:
        wp.atomic_min(valid, 0, 0)
        wp.atomic_min(valid, 1, 0)
    else:
        wp.atomic_max(scales, 0, wp.max(wp.abs(current[i]), wp.abs(a)))
        if extrapolate:
            if not wp.isfinite(b):
                wp.atomic_min(valid, 1, 0)
            else:
                wp.atomic_max(scales, 0, wp.abs(b))
                wp.atomic_max(scales, 1, wp.abs(a - b))
                wp.atomic_max(scales, 2, wp.abs(wp.float64(2.0) * a - b))


@wp.kernel(enable_backward=False)
def _engagement(
    current: wp.array[int],
    last: wp.array[int],
    older: wp.array[int],
    extrapolate: bool,
    valid: wp.array[int],
):
    i = wp.tid()
    if current[i] != last[i]:
        wp.atomic_min(valid, 0, 0)
        wp.atomic_min(valid, 1, 0)
    if extrapolate and current[i] != older[i]:
        wp.atomic_min(valid, 1, 0)


@wp.kernel(enable_backward=False)
def _guess(
    current: wp.array[wp.float64],
    last: wp.array[wp.float64],
    older: wp.array[wp.float64],
    extrapolate: bool,
    valid: wp.array[int],
    scales: wp.array[wp.float64],
    trial: wp.array[wp.float64],
    counts: wp.array[int],
):
    i = wp.tid()
    kind = 0
    value = current[i]
    if valid[0] != 0:
        value = last[i]
        kind = 1
        # The same coefficient applies to the entire nodal/rigid force vector,
        # retaining its balance. Abrupt growth retains the accepted phase load.
        if extrapolate and valid[1] != 0 and scales[1] <= scales[0] and scales[2] <= wp.float64(2.0) * scales[0]:
            value = wp.float64(2.0) * last[i] - older[i]
            kind = 2
    trial[i] = value
    if i == 0:
        counts[kind] += 1


class _Frame:
    def __init__(self, load, phases):
        self.loads = [wp.empty_like(load) for _ in range(phases)]
        self.engagement = [None] * phases
        self.signatures = [None] * phases


class PhaseLoadPredictor:
    """Completed accepted control frames; no failed trial enters history."""

    def __init__(self, load):
        self.load = load
        self.recent = wp.empty_like(load)
        self.valid = wp.zeros(2, dtype=int, device=load.device)
        self.scales = wp.zeros(3, dtype=wp.float64, device=load.device)
        self.counts = wp.zeros(3, dtype=int, device=load.device)
        self.frames = None
        self.phase_count = 0
        self.depth = 0
        self.open = self.frame_open = False
        self.pending_valid = False
        self.metadata = self.frame_signature = None
        self.last_time = self.last_command = None
        self.attempts = self.commits = self.invalidations = 0
        self.phase = 0

    def invalidate(self):
        # An internal force-graph fallback may invalidate while the enclosing
        # physical transaction remains open. Leave closing to commit/discard.
        self.depth = 0
        self.pending_valid = False
        self.metadata = None
        self.invalidations += 1

    def begin_frame(self, *, frame_dt, step_dt, elapsed, command, reset_token):
        if self.frame_open or self.open:
            raise RuntimeError("A predictor control frame is already open")
        if not np.isfinite([frame_dt, step_dt, elapsed]).all() or min(frame_dt, step_dt) <= 0:
            raise ValueError("Control-frame times must be finite and positive")
        phases = round(frame_dt / step_dt)
        command = np.asarray(command, dtype=float)
        regular = 0 < phases <= 64 and math.isclose(phases * step_dt, frame_dt, rel_tol=1e-12)
        signature = (float(frame_dt), float(step_dt), phases, reset_token)
        discontinuity = self.last_time is not None and not math.isclose(elapsed, self.last_time, abs_tol=1e-10)
        if self.last_command is not None:
            discontinuity |= (
                not np.isfinite(command).all()
                or np.linalg.norm(command[:3] - self.last_command[:3]) > 0.001
                or abs(float(command[3:7] @ self.last_command[3:7])) < 0.995
                or abs(float(command[7] - self.last_command[7])) > 0.05
            )
        if signature != self.frame_signature or discontinuity or not regular:
            self.invalidate()
        if regular and (self.frames is None or self.phase_count != phases):
            self.frames = [_Frame(self.load, phases) for _ in range(3)]
            self.phase_count = phases
        self.frame_signature = signature
        self.last_command = command.copy()
        self.frame_start = elapsed
        self.phase = 0
        self.frame_open = True
        self.pending_valid = regular

    def begin(self, current, trial, dt, signature, engagement):
        if self.open:
            raise RuntimeError("Predictor physical transaction already open")
        self.open = True
        wp.copy(trial, current)
        if (
            not self.frame_open
            or not self.pending_valid
            or self.phase >= self.phase_count
            or dt != self.frame_signature[1]
            or signature is None
        ):
            self.invalidate()
            return
        if self.metadata is not None and self.metadata != signature:
            self.invalidate()
            self.pending_valid = self.phase == 0
        self.metadata = signature
        pending = self.frames[2]
        pending.signatures[self.phase] = signature
        if pending.engagement[self.phase] is None or pending.engagement[self.phase].shape != engagement.shape:
            pending.engagement[self.phase] = wp.empty_like(engagement)
        wp.copy(pending.engagement[self.phase], engagement)
        if not self.pending_valid or self.depth == 0:
            return
        last, older = self.frames[:2]
        if last.signatures[self.phase] != signature or last.engagement[self.phase].shape != engagement.shape:
            return
        extrapolate = self.depth == 2 and older.signatures[self.phase] == signature
        old_load = older.loads[self.phase] if extrapolate else last.loads[self.phase]
        old_engagement = older.engagement[self.phase] if extrapolate else last.engagement[self.phase]
        self.attempts += 1
        self.valid.fill_(1)
        self.scales.zero_()
        wp.launch(
            _scales,
            current.size,
            [
                current,
                self.recent,
                last.loads[self.phase],
                old_load,
                extrapolate,
                self.valid,
                self.scales,
            ],
            device=current.device,
        )
        if engagement.size:
            wp.launch(
                _engagement,
                engagement.size,
                [
                    engagement,
                    last.engagement[self.phase],
                    old_engagement,
                    extrapolate,
                    self.valid,
                ],
                device=current.device,
            )
        wp.launch(
            _guess,
            current.size,
            [
                current,
                last.loads[self.phase],
                old_load,
                extrapolate,
                self.valid,
                self.scales,
                trial,
                self.counts,
            ],
            device=current.device,
        )

    def commit(self, accepted):
        if not self.open:
            raise RuntimeError("No predictor physical transaction is open")
        wp.copy(self.recent, accepted)
        if self.pending_valid and self.phase < self.phase_count:
            wp.copy(self.frames[2].loads[self.phase], accepted)
        self.phase += 1
        self.commits += 1
        self.open = False

    def cancel(self):
        self.invalidate()

    def discard(self):
        self.invalidate()
        self.open = False

    def end_frame(self, elapsed, success):
        complete = (
            success
            and self.pending_valid
            and self.phase == self.phase_count
            and not self.open
            and math.isclose(
                elapsed - self.frame_start,
                self.frame_signature[0],
                rel_tol=1e-9,
                abs_tol=1e-10,
            )
        )
        if complete:
            last, older, pending = self.frames
            self.frames = [pending, last, older]
            self.depth = min(2, self.depth + 1)
        else:
            self.invalidate()
        self.frame_open = False
        self.last_time = elapsed


def coupler_signature(coupler):
    """Stable, pre-accept force metadata, or None for unsupported force maps.

    Dynamic positions, velocities and ALM multipliers are deliberately absent:
    they are the smoothly varying state being extrapolated. Engagement is
    compared separately on the device, without a new host synchronization.
    Storage/topology replacement or parameter changes invalidate the history.
    """
    from .coupling import NeedleTissueCoupler
    from .insertion import NeedleTissueInteraction
    from .solvers.fem import SolverFEM
    from .solvers.fem.differentiation import active_tape
    from .subcycled_mujoco import SubcycledMuJoCo
    from .tunnel_alm import _TunnelALM

    c, x = coupler, coupler.interaction
    if (
        type(x) is not NeedleTissueInteraction
        or type(c.fem_solver) is not SolverFEM
        or getattr(c.fem_solver.step, "__func__", None) is not SolverFEM.step
        or type(c.rigid_solver) is not SubcycledMuJoCo
        or getattr(c.rigid_solver.step, "__func__", None) is not SubcycledMuJoCo.step
        or getattr(c.model, "requires_grad", False)
        or getattr(c.fem_solver, "differentiable", False)
        or active_tape() is not None
        or x.track_count == 0
        or x._previous_tip is None
        or not x.punctured
        or not x._in_material
        or x.status == "failed"
        or x._pending_opening is not None
    ):
        return None
    if any(
        getattr(getattr(c, name), "__func__", None) is not getattr(NeedleTissueCoupler, name)
        for name in ("_step_fem", "_evaluate", "_solve_host")
    ) or any(
        getattr(getattr(x, name), "__func__", None) is not getattr(NeedleTissueInteraction, name)
        for name in (
            "add_forces",
            "_add_forces",
            "_visited_sample_mask",
            "_tip_boundary_permission",
        )
    ):
        return None
    alm = x.tunnel_alm
    if alm is not None and (
        type(alm) is not _TunnelALM
        or any(
            getattr(getattr(alm, name), "__func__", None) is not getattr(_TunnelALM, name)
            for name in (
                "add_forces",
                "begin_step",
                "commit",
                "converged",
                "update_duals",
            )
        )
    ):
        return None
    arrays = tuple(
        getattr(x, name)
        for name in (
            "_samples",
            "_radii",
            "_weights",
            "_fixed",
            "_mapping",
            "_triangles",
            "_all_opening_faces",
            "_track_nodes",
            "_track_bary",
            "_track_lengths",
            "_track_engaged",
            "_particle_inverse_mass",
        )
    )
    return (
        id(x),
        tuple(sorted(x._visited_tets)),
        x.track_count,
        tuple(sorted(x._track_edges)),
        tuple((ids.tobytes(), bary.tobytes()) for ids, bary in x._openings),
        id(x._opening),
        x.punctured,
        x._in_material,
        x.stiffness,
        x.damping,
        x.friction,
        x.speed_scale,
        x.puncture_force,
        x.surface_sample_stiffness,
        x.shaft_stiffness_per_m,
        x.shaft_damping_per_m,
        x.track_spacing,
        float(x.radii.max()),
        x._rigid_inverse_mass,
        tuple(np.asarray(x._rigid_inverse_inertia).reshape(-1)),
        None
        if alm is None
        else (
            id(alm),
            alm.penalty_per_m,
            alm.compliance_per_m,
            alm.clearance,
            alm.tolerance,
        ),
        c.needle_body,
        tuple(c.needle_com),
        c.moment_length,
        tuple((array.ptr, array.shape, array.dtype) for array in arrays),
    )
