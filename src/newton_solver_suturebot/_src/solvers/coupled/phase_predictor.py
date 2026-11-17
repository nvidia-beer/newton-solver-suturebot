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


@wp.kernel(enable_backward=False)
def _increment(
    accepted: wp.array[wp.float64],
    start: wp.array[wp.float64],
    increment: wp.array[wp.float64],
):
    i = wp.tid()
    increment[i] = accepted[i] - start[i]


@wp.kernel(enable_backward=False)
def _increment_scales(
    current: wp.array[wp.float64],
    recent: wp.array[wp.float64],
    last: wp.array[wp.float64],
    older: wp.array[wp.float64],
    valid: wp.array[int],
    scales: wp.array[wp.float64],
):
    i = wp.tid()
    a, b = last[i], older[i]
    if not wp.isfinite(a) or not wp.isfinite(b) or not wp.isfinite(current[i]) or current[i] != recent[i]:
        wp.atomic_min(valid, 0, 0)
    else:
        # Interface loads are sparse; a zero row cannot raise any maximum.
        if a != wp.float64(0.0):
            wp.atomic_max(scales, 0, wp.abs(a))
        if b != wp.float64(0.0):
            wp.atomic_max(scales, 1, wp.abs(b))
        if a != b:
            wp.atomic_max(scales, 2, wp.abs(a - b))


@wp.kernel(enable_backward=False)
def _increment_guess(
    current: wp.array[wp.float64],
    increment: wp.array[wp.float64],
    valid: wp.array[int],
    scales: wp.array[wp.float64],
    trial: wp.array[wp.float64],
    counts: wp.array[int],
):
    i = wp.tid()
    kind = 0
    value = current[i]
    # Accepted loads are balanced, so adding the difference of two of them to a
    # third retains the balance. The guess is measured on accepted history: the
    # changes of this phase in the last two frames must differ by less than
    # either of them. Then repeating the older change would have started the
    # last frame closer to its accepted load than the plain start did, and a
    # single jump in either frame is never repeated.
    if valid[0] != 0 and valid[1] != 0 and scales[2] < wp.min(scales[0], scales[1]):
        value += increment[i]
        kind = 3
    trial[i] = value
    if i == 0:
        counts[kind] += 1


class _Frame:
    def __init__(self, load, phases, increments=False):
        self.loads = [wp.empty_like(load) for _ in range(phases)]
        self.increments = [wp.empty_like(load) for _ in range(phases)] if increments else None
        self.engagement = [None] * phases
        # Number of leading rows of each engagement snapshot that were copied.
        self.engaged_rows = [0] * phases
        self.signatures = [None] * phases


class PhaseLoadPredictor:
    """Completed accepted control frames; no failed trial enters history.

    The default guess is the load accepted at the same control phase of the
    last frame, extrapolated over two frames. With ``increments`` the guess is
    instead the last accepted load plus the change accepted over the same
    control phase of the last frame: x(f, p) = x(f, p-1) + x(f-1, p) - x(f-1, p-1).
    It is exact for a frame-periodic load on a linear drift and follows the most
    recent accepted load. It needs two completed frames, because it is used only
    where the last two changes of the phase agree. Either guess only chooses the
    first trial load; acceptance belongs to the coupler.

    ``begin`` takes the engagement as a device array and the number of its leading rows
    that are in use (all of them when it is omitted). Storage of a fixed size then keeps
    its snapshots: only the rows in use are copied and compared, and nothing is issued
    for none.

    A rejected transaction is never repeated from another start: the trial
    budget of a step is spent once, and the caller shortens the step as it does
    without a predictor. ``rejections`` counts, on the host, the rejected
    transactions whose trials started from an offered increment guess.
    """

    def __init__(self, load, *, increments=False):
        self.load = load
        self.increments = bool(increments)
        self.recent = wp.empty_like(load)
        self.start = wp.empty_like(load) if self.increments else None
        self.valid = wp.zeros(2, dtype=int, device=load.device)
        self.scales = wp.zeros(3, dtype=wp.float64, device=load.device)
        # No guess, same-phase load, two-frame extrapolation, same-phase increment.
        self.counts = wp.zeros(4, dtype=int, device=load.device)
        self.frames = None
        self.phase_count = 0
        self.depth = 0
        self.open = self.frame_open = False
        self.pending_valid = False
        self.metadata = self.frame_signature = None
        self.last_time = self.last_command = None
        self.attempts = self.commits = self.invalidations = self.rejections = 0
        self.phase = 0
        # Whether the trials of the open transaction started from an offered increment guess.
        self.attempted = False

    def invalidate(self):
        # An internal force-graph fallback may invalidate while the enclosing
        # physical transaction remains open. Leave closing to commit/discard.
        self.depth = 0
        self.pending_valid = False
        self.metadata = None
        self.invalidations += 1

    @staticmethod
    def _phases(frame_dt, step_dt):
        """Number of control phases of a frame, and whether the frame is a whole number of at most 64 steps."""
        phases = round(frame_dt / step_dt)
        return phases, 0 < phases <= 64 and math.isclose(phases * step_dt, frame_dt, rel_tol=1e-12)

    def reserve(self, *, frame_dt, step_dt, engagement=None):
        """Make the storage of the control frames ahead of the first frame; for start-up.

        :meth:`begin_frame` makes the three frames of a phase count where it first sees that
        count, and :meth:`begin` makes the snapshot of the engagement of a frame and phase where
        that phase first has a signature, which is inside a step of the run. This makes both
        now, for the frames of ``frame_dt`` and ``step_dt``: the loads (and increments) of every
        phase and, with ``engagement``, one snapshot per frame and phase with the storage shape
        of that array. Storage that exists with these sizes is kept. Nothing is written and no
        history changes: a snapshot is compared only where the signature of its phase was
        stored together with its rows.

        Returns:
            False, with nothing made, for a frame that is not a whole number of at most 64 steps.
        """
        if not np.isfinite([frame_dt, step_dt]).all() or min(frame_dt, step_dt) <= 0:
            raise ValueError("Control-frame times must be finite and positive")
        phases, regular = self._phases(frame_dt, step_dt)
        if not regular:
            return False
        if self.frames is None or self.phase_count != phases:
            self.frames = [_Frame(self.load, phases, self.increments) for _ in range(3)]
            self.phase_count = phases
        if engagement is not None:
            for frame in self.frames:
                for phase in range(phases):
                    snapshot = frame.engagement[phase]
                    if snapshot is None or snapshot.shape != engagement.shape:
                        frame.engagement[phase] = wp.empty_like(engagement)
        return True

    def begin_frame(self, *, frame_dt, step_dt, elapsed, command, reset_token):
        if self.frame_open or self.open:
            raise RuntimeError("A predictor control frame is already open")
        if not np.isfinite([frame_dt, step_dt, elapsed]).all() or min(frame_dt, step_dt) <= 0:
            raise ValueError("Control-frame times must be finite and positive")
        phases, regular = self._phases(frame_dt, step_dt)
        command = np.asarray(command, dtype=float)
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
            self.frames = [_Frame(self.load, phases, self.increments) for _ in range(3)]
            self.phase_count = phases
        self.frame_signature = signature
        self.last_command = command.copy()
        self.frame_start = elapsed
        self.phase = 0
        self.frame_open = True
        self.pending_valid = regular

    def begin(self, current, trial, dt, signature, engagement, count=None):
        if self.open:
            raise RuntimeError("Predictor physical transaction already open")
        rows = engagement.size if count is None else int(count)
        if not 0 <= rows <= engagement.size:
            raise ValueError("The engagement count must lie within the engagement array")
        self.open = True
        self.attempted = False
        wp.copy(trial, current)
        if self.increments:
            # The change accepted by this transaction is measured from its own start.
            wp.copy(self.start, current)
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
        pending.engaged_rows[self.phase] = rows
        if rows:
            # A count of zero means the whole array to wp.copy.
            wp.copy(pending.engagement[self.phase], engagement, count=rows)
        if not self.pending_valid or self.depth == 0:
            return
        last, older = self.frames[:2]

        def comparable(frame):
            # The same storage shape and the same rows in use as this transaction.
            snapshot = frame.engagement[self.phase]
            return (
                snapshot is not None and snapshot.shape == engagement.shape and frame.engaged_rows[self.phase] == rows
            )

        if self.increments:
            if (
                self.depth == 2
                and last.signatures[self.phase] == older.signatures[self.phase] == signature
                and comparable(last)
                and comparable(older)
            ):
                self._guess_increment(current, trial, last, older, engagement, rows)
            return
        if last.signatures[self.phase] != signature or not comparable(last):
            return
        extrapolate = self.depth == 2 and older.signatures[self.phase] == signature and comparable(older)
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
        if rows:
            wp.launch(
                _engagement,
                rows,
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

    def _guess_increment(self, current, trial, last, older, engagement, rows):
        self.attempts += 1
        self.attempted = True
        self.valid.fill_(1)
        self.scales.zero_()
        wp.launch(
            _increment_scales,
            current.size,
            [
                current,
                self.recent,
                last.increments[self.phase],
                older.increments[self.phase],
                self.valid,
                self.scales,
            ],
            device=current.device,
        )
        if rows:
            wp.launch(
                _engagement,
                rows,
                [engagement, last.engagement[self.phase], older.engagement[self.phase], True, self.valid],
                device=current.device,
            )
        wp.launch(
            _increment_guess,
            current.size,
            [current, last.increments[self.phase], self.valid, self.scales, trial, self.counts],
            device=current.device,
        )

    def commit(self, accepted):
        if not self.open:
            raise RuntimeError("No predictor physical transaction is open")
        wp.copy(self.recent, accepted)
        if self.pending_valid and self.phase < self.phase_count:
            wp.copy(self.frames[2].loads[self.phase], accepted)
            if self.increments:
                wp.launch(
                    _increment,
                    accepted.size,
                    [accepted, self.start, self.frames[2].increments[self.phase]],
                    device=accepted.device,
                )
        self.phase += 1
        self.commits += 1
        self.open = False
        self.attempted = False

    def cancel(self):
        # The exact fallback restarts its trials from the accepted load.
        self.invalidate()
        self.attempted = False

    def discard(self):
        self.invalidate()
        self.rejections += int(self.attempted)
        self.attempted = False
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


def coupler_signature(coupler, *, increments=False):
    """Stable, pre-accept force metadata, or None for unsupported force maps.

    Dynamic positions, velocities and ALM multipliers are deliberately absent:
    they are the smoothly varying state being extrapolated. Engagement is
    compared separately on the device, without a new host synchronization.
    Storage/topology replacement or parameter changes invalidate the history.

    ``increments`` is the signature of the frame-periodic increment guess. That
    guess continues the last accepted load, so an engaged track whose tip is
    outside the material is supported as well. Everything else is common to both
    guesses: track growth (an appended anchor, visited tetrahedron, track edge
    or opening), a puncture or material state change, a parameter change and
    replaced storage are each another force map, and no change recorded before
    one of them is used after it.
    """
    from ..fem import SolverFEM
    from ..fem.differentiation import active_tape
    from ..mujoco.experimental_rigid_backend import supported_subcycled
    from .coupling import NeedleTissueCoupler
    from .interface.insertion import NeedleTissueInteraction
    from .interface.tunnel_alm import _TunnelALM

    c, x = coupler, coupler.interaction
    if (
        type(x) is not NeedleTissueInteraction
        or type(c.fem_solver) is not SolverFEM
        or getattr(c.fem_solver.step, "__func__", None) is not SolverFEM.step
        or not supported_subcycled(c.rigid_solver)
        or getattr(c.model, "requires_grad", False)
        or getattr(c.fem_solver, "differentiable", False)
        or active_tape() is not None
        or x.track_count == 0
        or x._previous_tip is None
        or not (increments or (x.punctured and x._in_material))
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
        (type(c.rigid_solver), id(c.rigid_solver)),
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
