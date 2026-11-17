# SPDX-License-Identifier: Apache-2.0
"""Press and slide the jaws of an articulated dVRK tool on FEM tissue, then pinch a ridge of it and lift it, in eight phases: the rig of the tool/tissue contact.

Only the tools of ``fem_needle_grasp`` (MuJoCo) and its tissue block (FEM) are simulated: there
is no needle, no thread and no needle/tissue coupling, so a frame is the rigid step of the tools
and the tissue step. The first tool follows a scripted path through the joint-target interface
of the grasp scene, in eight phases: approach, press, press deeper, slide, lift, press at the held
border of the block and one fast approach with closed jaws, and a pinch: the jaws open over a
ridge of the tissue, go down beside it, close on it, lift it (``--pinch-lift``), lower and open
(``--start-phase pinch`` begins there). ``--contact`` chooses what tools and tissue exchange:
nothing (``off``, the default: the tools pass through the tissue), the first-version law of the
grasp scene (``first``) or the contact on the sweep of the capsules (``swept``: the tissue under a
capsule is placed on it and held through the next interval, and tissue that is held or cannot
follow stops the tool; tissue that it holds between the two jaws of a tool is gripped: carried by
them while the pull stays below the friction limit of the squeeze). Every frame is measured, and the
measurements are summed up by phase at the end; ``--measure-intervals`` also measures the end of
every physics interval. With ``--test`` and the swept contact the pinch is checked row by row: what
is clamped, the squeeze, the tissue at its places on the jaws at rest and while they move, the pull
on the tool, the tent, the release, the work through the carried nodes and the grip against the pull.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import newton
import newton.examples
import numpy as np
import warp as wp
from newton.viewer import ViewerNull

from newton_solver_suturebot.examples.grasp._tool_tissue_contact import CONTACTS
from newton_solver_suturebot.examples.grasp._tool_tissue_measure import Measurements
from newton_solver_suturebot.examples.grasp._tool_tissue_path import PHASES, PINCH, RESTART_PHASES, plan_path
from newton_solver_suturebot.examples.grasp._tool_tissue_scene import ToolTissueScene
from newton_solver_suturebot.examples.grasp.example_fem_needle_grasp import Example as GraspExample

# Speed of the motions that touch the tissue [m/s] and length of a hold [s]: the arc speed and the
# settling time of fem_needle_grasp, whose tool/tissue contact this example is the rig for.
SPEED = GraspExample.create_parser().get_default("speed")
HOLD_SECONDS = GraspExample.create_parser().get_default("settle_seconds")
# Commanded depths of the jaws below the tissue surface at rest [m]: the two presses, and the hold
# after the fast approach. In jaw contact radii (0.65 mm, the capsule that carries the jaw tip) they
# are 2.3, 4.6 and 3.1, in mesh spacings (1.31 mm) 1.1, 2.3 and 1.5: each press is deeper than one
# contact sphere and than one element.
PRESS_DEPTH = 0.0015
DEEP_DEPTH = 0.003
FAST_DEPTH = 0.002
# Length of the slide along the surface [m]: more than seven mesh spacings.
SLIDE_LENGTH = 0.010
# Gap between the jaw tips that the pinch commands when it closes on the ridge, in widths of the ridge at
# the height of the tips: a choice of this script, like the press depths, made on measurements of this scene
# (friction 0.83, the phase on the CPU). At 0.42 the squeeze at the end of the squeeze hold is 275 mN and the
# tissue pulls with 108 N/m of its rise: a lift of 5 mm is held with every row of the self-check met. A firmer
# close holds with more to spare and squeezes the elements at the faces thinner: at 0.36 (squeeze 364 mN) the
# friction limit is 1.33 times the pull at the end of a 5 mm lift, against 1.16, but the smallest volume ratio
# is 0.04 against 0.11, and after the lowering one carried node is 49 um from its place on its jaw and the work
# through the carried nodes is outside its bound (rows 3a and 8). A lighter close slips earlier (measured with
# the earlier carry rule: at 1.7 mm of tip rise at 0.5, at 1.2 mm at 0.6).
PINCH_GAP_FRACTION = 0.42
# How far the closed jaws lift the pinched ridge [m] (--pinch-lift): a choice of this script. At 5 mm, with the
# holds of this script (0.5 s), the carried tissue rises 4.5 mm; the friction limit is 1.16 times the pull at the
# end of the lift and 1.27 times at the end of its hold, every carried node is at its place on its jaw at the two
# holds (the farthest 35 um of the 39.5 um allowed, after the lowering) and the release leaves the tissue at rest.
# That margin does not cover a longer squeeze hold: after one of 2 s (--hold-seconds 2.0) the pair slips for two
# intervals in this lift and lets go of a part of the grip (the tissue then rises 3.5 mm). A lift of 8 mm is
# held and released as well (the limit 1.06 times the pull), but after its lowering one node is 44 um from its
# place: the self-check (--test) is met at 5 mm with the script's holds, and not at 8 mm.
PINCH_LIFT = 0.005
# A frame slower than this is below 15 frames per second [ms].
FRAME_LIMIT_MS = 1000.0 / 15.0


class Example:
    """A normal Newton viewer over the tool/tissue scene, its scripted path and its measurements."""

    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        self.frame = self.completed = 0
        self.sim_time = 0.0
        self.halted = None
        self.finished = False
        self.view = None
        # Time of every step of the run and of the loop period after it [ms], with the frame of the path that the
        # step ran: after a failed phase or a shortened fast approach the two numberings differ.
        self.step_ms, self.period_ms, self.step_frames = [], [], []
        self._entry = None
        # Row of the arm targets at which the target of the tool stands: the end of the frame that was run last.
        self._row = 0
        self._last = None
        # Half-opening of the jaws that was commanded last [rad]: a frame writes it only when it changes.
        self._opening = 0.0
        device = wp.get_device(args.device)
        wp.set_device(device)
        started = time.perf_counter()
        scene_args = ToolTissueScene.create_parser().parse_args(["--substeps", str(args.substeps)])
        self.scene = ToolTissueScene(
            ViewerNull(), scene_args, rigid_backend=self._backend(args, device), contact=args.contact
        )
        self.path, self.arm_targets = plan_path(
            self.scene,
            speed=args.speed,
            hold_seconds=args.hold_seconds,
            press_depth=args.press_depth,
            deep_depth=args.deep_depth,
            slide_length=args.slide_length,
            fast_depth=args.fast_depth,
            pinch_gap_fraction=PINCH_GAP_FRACTION,
            pinch_lift=getattr(args, "pinch_lift", PINCH_LIFT),
            jaw_tilt=math.radians(args.jaw_tilt_deg),
        )
        self.measurements = Measurements(self.scene, self.path)
        if getattr(getattr(self.scene.contact, "exchange", None), "pinch", None) is not None:
            # What the self-check of the pinch needs from the tool model: how far the palm gives under a force on
            # a jaw, at the two holds between which the pinch lifts (host arithmetic of start-up).
            for segment in self.path.segments:
                if segment.phase == PINCH and segment.label in ("squeeze hold", "lift hold"):
                    last = segment.last_frame
                    self.measurements.pinch_compliance[segment.label] = self.scene.actuator_compliance(
                        self.arm_targets[last + 1], float(self.path.opening[last])
                    )
        # Where the run begins: the start of the path, or the frame of a later phase at whose start the tool is
        # clear of the tissue (--start-phase), by the placement of a restart.
        start_phase = getattr(args, "start_phase", None)
        self.first_frame = 0 if start_phase is None else self.path.restarts[start_phase]
        # Frames of the run: to the end of the path, or as many as were asked for.
        self.frame_limit = self.path.frames if args.num_frames <= 0 else self.first_frame + args.num_frames
        if getattr(viewer, "num_frames", None) is not None and args.num_frames <= 0:
            viewer.num_frames = self.frame_limit - self.first_frame
        self._place(self.first_frame)
        self.scene.start()
        if not args.measure:
            # Without the measurements the frames of the fast approach are still observed (ToolPath.next_frame):
            # the launches of an observation belong to the start-up either way.
            self.scene.observe()
        # The first observation launches the kernels of every later one: it belongs to the start-up.
        self._last = self.scene.observe() if args.measure else None
        if args.measure and args.measure_intervals:
            self.scene.interval_observer = self._observe_interval
        if args.viewer != "null":
            self._build_view()
            # The launches of a draw belong to the start-up. A viewer that counts its frames up to a
            # limit would spend one of them on the draw itself.
            self.view.update(self.scene.state_0)
            if not GraspExample._frame_limited(viewer):
                self.view.draw(0.0)
        self._print_plan()
        # Before start-up ends, as in the grasp example: the garbage of start-up is collected once and what
        # survives leaves the reach of the cyclic collector, so that no full collection walks it in a frame.
        GraspExample._freeze_startup_objects()
        # Start-up ends here: the scene, its graphs, the observation and the view of the start state.
        self.scene.finish_startup()
        self.startup_seconds = time.perf_counter() - started
        self._run_started = None
        self.run_seconds = None

    @staticmethod
    def _backend(args, device):
        """The rigid backend: MuJoCo Warp on a CUDA device, native MuJoCo on the CPU, unless one is asked for."""
        backend = getattr(args, "rigid_backend", "auto")
        if backend == "auto":
            return "mujoco-warp" if device.is_cuda else "native"
        if backend == "mujoco-warp" and not device.is_cuda:
            raise ValueError("The MuJoCo Warp rigid backend needs a CUDA device")
        return backend

    def _build_view(self):
        from newton_solver_suturebot.examples.grasp._tool_tissue_view import ToolTissueView

        self.view = ToolTissueView(self.scene, self.viewer)
        self.model, self.state_0 = self.view.model, self.view.state
        if self.args.overview:
            tools = [np.asarray(tool.trocar_pose[:3]) for tool in self.scene.graspers]
            points = np.concatenate((np.asarray(self.scene.selection.vertices), tools))
        else:
            # Close enough to see the jaws on the tissue: every commanded tip position and the palm. The targets
            # of the fast approach after its first frame lie beyond the surface, where the tool does not follow.
            margin = 4.0 * self.scene.jaw_contact_radius
            ramp = self.path.fast_approach
            shown = np.ones(self.path.frames, dtype=bool)
            shown[ramp.first_frame + 1 : ramp.last_frame + 1] = False
            tip = self.path.tip[shown]
            points = np.concatenate((tip - margin, tip + margin, self.path.poses[shown, :3]))
        self.view.frame(points)

    def _place(self, frame):
        """Place the tool at rest where ``frame`` starts, with the jaw opening of that moment (a reset-time placement)."""
        self._opening = self.path.opening_before(frame)
        self.scene.place(self.arm_targets[frame], self._opening)
        self.frame = self._row = frame

    def _print_plan(self):
        numbers = self.path.derivation
        radius, spacing = numbers["jaw_contact_radius_m"], numbers["mesh_spacing_m"]
        print(
            f"Tool/tissue contact: {self.args.contact} | rigid backend: {self.scene.describe()['rigid_backend']} | "
            f"{self.path.frames} frames, {numbers['seconds']:.2f} s of simulated time",
            flush=True,
        )
        print(
            f"Jaw contact radius {1e3 * radius:.3f} mm, mesh spacing {1e3 * spacing:.3f} mm, physics interval "
            f"{1e3 * numbers['interval_s']:.3f} ms; speed {1e3 * numbers['speed_m_s']:.2f} mm/s "
            f"({numbers['speed_travel_per_interval_jaw_radii']:.4f} jaw radii per interval), transit "
            f"{1e3 * numbers['transit_speed_m_s']:.1f} mm/s, fast approach {1e3 * numbers['fast_speed_m_s']:.0f} mm/s "
            f"({numbers['fast_travel_per_interval_jaw_radii']:g} jaw radii per interval); jaws "
            f"{math.degrees(numbers['jaw_angle_below_horizontal_rad']):.1f} degrees below the horizontal",
            flush=True,
        )
        ramp = self.path.fast_approach
        for phase in PHASES:
            first, last = self.path.phase_frames(phase)
            parts = "; ".join(
                f"{segment.label} {'up to ' if segment is ramp and segment.frames > 1 else ''}{segment.frames}"
                for segment in self.path.segments
                if segment.phase == phase
            )
            print(f"  {PHASES.index(phase) + 1} {phase:10s} frames {first:4d} to {last:4d}: {parts}", flush=True)
        print(
            "  The target of the fast approach passes the commanded depth and goes on until the jaw tip itself has "
            "crossed the tissue surface at rest; the frames of it that are left are not run.",
            flush=True,
        )
        print(
            f"  Pinch: ridge {1e3 * numbers['pinch_ridge_width_m']:.2f} mm wide at the height of the jaw tips, top "
            f"{1e3 * numbers['pinch_ridge_top_m']:.2f} mm, middle {1e3 * numbers['pinch_ridge_middle_offset_m']:+.2f} mm "
            f"from the middle site; jaws open to a tip gap of {1e3 * numbers['pinch_open_tip_gap_m']:.2f} mm "
            f"({numbers['pinch_open_half_opening_rad']:.3f} rad each), close to "
            f"{1e3 * numbers['pinch_closed_tip_gap_m']:.2f} mm ({numbers['pinch_closed_half_opening_rad']:.3f} rad, "
            f"{numbers['pinch_gap_fraction']:g} of the width), tips {1e3 * numbers['pinch_depth_m']:.1f} mm below the "
            f"top, lift {1e3 * numbers['pinch_lift_m']:.1f} mm",
            flush=True,
        )
        if self.first_frame:
            print(f"  The run begins at frame {self.first_frame} (--start-phase {self.args.start_phase}).", flush=True)
        print("Space: pause/resume | .: single step | Esc: close", flush=True)

    def _observe_interval(self, dt):
        """Measure the end of one accepted physics interval (``--measure-intervals``)."""
        self.measurements.add_interval(self.frame, self.scene.observe(keep_travel=True), dt)

    def _halt(self, reason):
        """Stop stepping after a failure: raise without a view, freeze the last state with one."""
        if self.view is None and not self.args.test:
            raise RuntimeError(reason)
        self.halted = reason
        print(f"Simulation halted at {self.sim_time:.2f} s (frame {self.frame}): {reason}", flush=True)

    def _fail(self, reason):
        """The scene could not go on in this frame.

        Without a contact, and for a failure of the rigid solver, the run halts. With a contact,
        a tissue step that cannot be resolved is what this rig is there to show: the phase is
        recorded as failed, the tissue is put back at rest, the contact is cleared and the tool
        is placed at the start of the next phase that begins clear of the tissue (a reset-time
        placement, not a motion). After the last such phase the run ends.
        """
        frame = self.frame
        segment = self.path.segment_of(min(frame, self.path.frames - 1))
        if self.args.contact == "off" or self.scene.failure_kind != "tissue":
            self.measurements.add_failure(frame, reason, None)
            self._halt(f"{reason} (phase {segment.phase}, {segment.label})")
            return
        restart = self.path.restart_after(frame) if frame < self.path.frames else None
        self.measurements.add_failure(frame, reason, restart)
        if restart is None or restart >= self.frame_limit:
            print(f"Phase {segment.phase} failed in frame {frame}: {reason}. No later phase to go on with.", flush=True)
            self.finished = True
            self.scene.reset_tissue()
            return
        print(
            f"Phase {segment.phase} failed in frame {frame}: {reason}. Tissue back at rest, contact cleared, tool "
            f"placed at the start of phase {self.path.segment_of(restart).phase} (frame {restart}).",
            flush=True,
        )
        self.scene.reset_tissue()
        self._place(restart)

    def step(self):
        now = time.perf_counter()
        if self._entry is not None and len(self.period_ms) < len(self.step_ms):
            # Period of the loop: from the entry of the last step to this one, with observation and drawing.
            self.period_ms.append(1e3 * (now - self._entry))
            if self.measurements.records and self.measurements.records[-1]["period_ms"] is None:
                self.measurements.records[-1]["period_ms"] = self.period_ms[-1]
        self._entry = None
        if self.halted is not None or self.finished:
            return
        if self._run_started is None:
            self._run_started = now
        frame = self.frame
        index = min(frame, self.path.frames - 1)
        # Targets of the frame: from where the target stands (the end of the frame that was run before this one)
        # to the pose at the end of this frame; after the path, held.
        self.scene.set_frame_targets(self.arm_targets[self._row], self.arm_targets[index + 1])
        # The jaw opening of the frame, written when it changes: two fills of the actuator targets.
        opening = float(self.path.opening[index])
        if opening != self._opening:
            self.scene.set_jaw_opening(opening)
            self._opening = opening
        self._entry = now
        self.scene.step()
        if self.args.measure:
            # The time of the step alone: wait for what it issued before the observation reads.
            wp.synchronize_device(self.scene.device)
        stepped = time.perf_counter()
        self.step_ms.append(1e3 * (stepped - now))
        self.step_frames.append(frame)
        if self.scene.failure_reason is not None:
            self._fail(self.scene.failure_reason)
        else:
            # The path reads the tool after the frames of its fast approach, with or without the measurements.
            reads = frame < self.path.frames and self.path.reads_tool(frame)
            observation = self.scene.observe() if self.args.measure or reads else None
            if self.args.measure:
                self._last = observation
                self.measurements.add(
                    frame,
                    observation,
                    step_s=stepped - now,
                    period_s=None,
                    observe_s=time.perf_counter() - stepped,
                    clock_s=now,
                    started_at=self._row - 1,
                )
                segment = self.path.segment_of(index)
                if segment.phase == PINCH and segment.label == "squeeze hold" and frame == segment.last_frame:
                    # From here on the closed jaws move together: the two reports of the carry start anew, for
                    # the rows that hold the carry to the travel of the jaws (a host call between two steps).
                    pinch = getattr(getattr(self.scene.contact, "exchange", None), "pinch", None)
                    if pinch is not None:
                        pinch.clear_reports()
            self._row = index + 1
            self.frame = self.path.next_frame(frame, observation["tip_gap_m"]) if reads else frame + 1
            # Frames of the fast approach that the tool did not need.
            self.measurements.add_unused(frame + 1, self.frame)
            self.completed += 1
        self.sim_time = self.scene.elapsed
        if self.frame >= self.frame_limit and not self.finished:
            self.finished = True
        if self.finished or self.halted is not None:
            self.run_seconds = time.perf_counter() - self._run_started
            self._finish()

    def _finish(self):
        """The run has ended: print the summary once; a window with a frame limit closes."""
        if self.args.measure:
            print("\n".join(self.measurements.lines()), flush=True)
            travel = self.fast_travel()
            if travel is not None:
                crossing = "the tip did not cross it"
                if travel["in_crossing_interval"] is not None:
                    crossing = (
                        f"{travel['in_crossing_interval']:.2f} (frame {travel['crossing_frame']}, gap of the tip "
                        f"sphere {1e3 * travel['crossing_gap_before_m']:.2f} mm before and "
                        f"{1e3 * travel['crossing_gap_after_m']:.2f} mm after)"
                    )
                print(
                    f"  fast approach, travel of the jaw tip per physics interval in jaw contact radii: commanded "
                    f"{travel['commanded']:.2f}, largest of the tool {travel['largest']:.2f}, in the interval in "
                    f"which the tool crossed the tissue surface at rest {crossing}; {travel['approach_frames']} of "
                    f"{travel['approach_frames_planned']} planned frames of the approach were run",
                    flush=True,
                )
        timing = self.timing()
        for name, label in (("step", "step"), ("period", "loop period (step, observation and drawing)")):
            if f"{name}_ms_median" not in timing:
                continue
            slow = timing[f"{name}s_over_limit"]
            print(
                f"  {label}: {timing[f'{name}_ms_median']:.2f} ms median, {timing[f'{name}_ms_p99']:.2f} ms 99th "
                f"percentile, {timing[f'{name}_ms_max']:.2f} ms largest, first {timing[f'{name}_ms_first']:.2f} ms; "
                f"{len(slow)} over {FRAME_LIMIT_MS:.1f} ms"
                + (f" [frame of the path, ms]: {slow[:12]}{' ...' if len(slow) > 12 else ''}" if slow else ""),
                flush=True,
            )
        print(
            f"Run: {timing['frames']} frames in {self.run_seconds:.2f} s after a start-up of {self.startup_seconds:.1f} s",
            flush=True,
        )
        self._print_pinch()
        if self.args.report and not self.args.test:
            self.write_report()
        if self.args.num_frames > 0 and self.view is not None and not GraspExample._frame_limited(self.viewer):
            # A window ignores the viewer's frame limit: end the loop as closing the window does.
            import pyglet

            pyglet.app.event_loop.has_exit = True

    def pinch_rows(self):
        """The rows of the pinch's self-check (``Measurements.pinch_rows``); None unless the pinch was measured."""
        records = self.measurements.records
        if not self.args.measure or not any("pinch_min_volume_ratio" in record for record in records):
            return None
        return self.measurements.pinch_rows(self._tolerance())

    def _print_pinch(self):
        """Print the rows of the pinch's self-check and its slips, where the pinch was run and measured."""
        rows = self.pinch_rows()
        if rows is None:
            return
        print("  Pinch (the tissue between the jaws):", flush=True)
        for row in rows:
            verdict = "no verdict" if row["passed"] is None else ("ok" if row["passed"] else "NOT MET")
            print(f"    {row['row']} {row['name']}: {verdict}. {row['numbers']}", flush=True)
        slips = self.measurements.pinch_slips()
        if slips is not None:
            first = slips["first"]
            if first is None:
                text = "no slip while the jaws were closed and still"
            else:
                rise = first["commanded_rise_m"]
                text = (
                    f"the pair slipped while the jaws were closed and still, in {slips['closed_frames']} frames "
                    f"({slips['closed_intervals']} intervals) from frame {first['frame']} ({first['segment']}"
                    + (f", commanded lift {1e3 * rise:.2f} mm" if rise is not None else "")
                    + f") on: pull {1e3 * first['pull_N']:.1f} mN against a limit of {1e3 * first['limit_N']:.1f} mN"
                )
            print(
                f"    slip: {text}; {slips['moving_intervals']} slipping intervals while the jaws closed or opened; "
                f"slip length {1e3 * slips['slip_length_m']:.2f} mm",
                flush=True,
            )

    def timing(self):
        """Frame times of the run [ms]: the step alone and the period of the loop, with the frames over 66.7 ms.

        A slow step or period is listed with the frame of the path that the step ran, as the records
        number their frames, not with its count in the run.
        """
        steps, periods = np.asarray(self.step_ms), np.asarray(self.period_ms)
        result = {"frames": len(steps), "limit_ms": FRAME_LIMIT_MS, "run_seconds": self.run_seconds}
        result["startup_seconds"] = self.startup_seconds
        for name, values in (("step", steps), ("period", periods)):
            if len(values):
                result.update(
                    {
                        f"{name}_ms_first": float(values[0]),
                        f"{name}_ms_median": float(np.median(values)),
                        f"{name}_ms_p99": float(np.percentile(values, 99)),
                        f"{name}_ms_max": float(values.max()),
                        f"{name}s_over_limit": [
                            [self.step_frames[index], round(float(value), 2)]
                            for index, value in enumerate(values)
                            if value > FRAME_LIMIT_MS
                        ],
                    }
                )
        return result

    def fast_travel(self):
        """Travel of the jaw tip per physics interval in the fast phase, in jaw contact radii.

        What the path commands; the largest travel of the tool itself; its travel in the one
        interval in which the tip sphere crossed the tissue surface at rest (its gap to that
        surface changed sign), with the frame of that interval and the gap at its start and at
        its end [m]; the largest travel in the frame in which the jaws first reach the deformed
        surface; and how many frames of the fast approach were run, of how many planned. None
        before that phase has been run.
        """
        radius = self.scene.jaw_contact_radius
        records = [record for record in self.measurements.records if record["phase"] == "fast"]
        if not records:
            return None
        touched = [record for record in records if record["jaw_depth_m"] is not None and record["jaw_depth_m"] > 0.0]
        crossed = [record for record in records if record["tip_crossing_travel_m"] is not None]
        crossing = crossed[0] if crossed else {}
        ramp = self.path.fast_approach
        approach = [record for record in records if record["segment"] == ramp.label]
        return {
            # The command of the approach: the first frame of the hold, which brings the target back, is not it.
            "commanded": max((record["commanded_tip_travel_per_interval_m"] for record in approach), default=0.0)
            / radius,
            "largest": max(record["tip_travel_per_interval_m"] for record in records) / radius,
            "in_crossing_interval": crossing["tip_crossing_travel_m"] / radius if crossed else None,
            "crossing_frame": crossing.get("frame"),
            "crossing_gap_before_m": crossing.get("tip_crossing_gap_before_m"),
            "crossing_gap_after_m": crossing.get("tip_crossing_gap_after_m"),
            "at_first_touch": touched[0]["tip_travel_per_interval_m"] / radius if touched else None,
            "approach_frames": len(approach),
            "approach_frames_planned": ramp.frames,
        }

    def write_report(self, **extra):
        self.measurements.write(
            self.args.report,
            timing=self.timing(),
            census=self.scene.census(),
            fast_tip_travel_per_interval_jaw_radii=self.fast_travel(),
            pinch_rows=self.pinch_rows(),
            pinch_slips=self.measurements.pinch_slips(),
            **extra,
        )
        print(f"Report written to {self.args.report}", flush=True)

    def render(self):
        if self.view is None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.end_frame()
            return
        # After a failure the view keeps the last state that was drawn.
        if self.halted is None:
            self.view.update(self.scene.state_0)
        self.view.draw(self.sim_time)

    def gui(self, ui):
        ui.text("Tool/tissue contact rig | MuJoCo tools + FEM tissue")
        ui.text(f"Contact: {self.args.contact}")
        index = min(self.frame, self.path.frames - 1)
        segment = self.path.segment_of(index)
        ui.text(f"Phase {PHASES.index(segment.phase) + 1}/{len(PHASES)}: {segment.phase} ({segment.label})")
        ui.text(f"Frame {self.frame} of {self.path.frames} | time {self.sim_time:.2f} s")
        ui.text(f"Commanded depth: {1e3 * self.path.depth[index]:.2f} mm")
        record = self.measurements.records[-1] if self.measurements.records else None
        if record is not None:
            if segment.phase == PINCH:
                first, second = record["jaw_half_opening_rad"]
                ui.text(
                    f"Jaws open: {first:.3f} / {second:.3f} rad, commanded {record['commanded_half_opening_rad']:.3f}"
                )
            if record.get("pinch_state") is not None:
                carried, clamped = record["pinch_carried_nodes"], record["pinch_clamped_nodes"]
                ui.text(
                    f"Pinch: {record['pinch_state']} | carried {carried[0]} + {carried[1]} nodes, "
                    f"clamped {clamped[0]} + {clamped[1]}"
                )
                ui.text(
                    f"Squeeze: {1e3 * record['pinch_squeeze_N']:.1f} mN | pull: {1e3 * record['pinch_pull_N']:.1f} mN "
                    f"| limit 2 mu S: {1e3 * record['pinch_limit_N']:.1f} mN"
                )
                if "pinch_carried_rise_m" in record:
                    ui.text(
                        f"Lift: tissue {1e3 * record['pinch_carried_rise_m']:.2f} mm, palm "
                        f"{1e3 * record['pinch_palm_rise_m']:.2f} mm of {1e3 * record['pinch_commanded_rise_m']:.2f} mm "
                        f"commanded | slipped {1e3 * record['pinch_slip_length_m']:.2f} mm"
                    )
                    # The carried tissue against its places on the jaws (the self-check holds every node to its
                    # place at the two holds, and the carry to one interval's travel while the jaws move).
                    behind = record["pinch_jaw_point_rise_m"] - record["pinch_carried_rise_m"]
                    ui.text(
                        f"Tissue behind its places on the jaws: {1e3 * behind:.3f} mm in the mean, "
                        f"{1e3 * record['pinch_farthest_behind_m']:.2f} mm the farthest node"
                    )
                if "pinch_kept_nodes" in record:
                    ui.text(
                        f"Kept by the pinch: {record['pinch_kept_nodes']} nodes | walked down: "
                        f"{record['pinch_walking_nodes']}"
                    )
            if record["jaw_depth_m"] is not None:
                ui.text(f"Jaws inside the deformed surface: {1e3 * record['jaw_depth_m']:.2f} mm")
            ui.text(f"Dimple: {1e3 * record['dimple_depth_m']:.2f} mm")
            ui.text(
                f"Force on tissue: {1e3 * record['tissue_force_N']:.1f} mN | on tool: {1e3 * record['tool_force_N']:.1f} mN"
            )
            ui.text(f"Step: {record['step_ms']:.1f} ms")
        ui.text(
            f"Rejected intervals: {self.scene.rejected_intervals} | failed phases: {len(self.measurements.failures)}"
        )
        if self.halted is not None:
            ui.text_wrapped(f"Halted: {self.halted}")
        elif self.finished:
            ui.text("The scripted path has ended.")

    # ------------------------------------------------------------------ self-check

    def _tolerance(self):
        """What the inverse kinematics of the tool leaves between a resting tool and its target [m], twice.

        Its position tolerance, and its orientation tolerance over the reach of the jaws from the
        palm; with the contact off nothing else acts on a tool that has settled.
        """
        ik = self.scene.pose_ik[0]
        reach = float(np.linalg.norm(self.scene.jaw_points, axis=1).max()) + self.scene.jaw_contact_radius
        return 2.0 * (ik.position_tolerance_m + ik.orientation_tolerance_rad * reach)

    def _check_off(self, phases):
        """With the contact off: the tool is on its path and the jaws are as deep in the resting tissue as commanded.

        In the pinch the jaws open and close: there the depth, which the observation measures on
        closed and open jaws alike, is not compared, and both jaw hinges must be on their command
        at the end of every hold, within twice the orientation tolerance of the inverse kinematics.
        """
        tolerance = self._tolerance()
        hinge_tolerance = 2.0 * self.scene.pose_ik[0].orientation_tolerance_rad
        constant = self.path.derivation["actuator_time_constant_s"]
        problems = []
        fastest = 0.0
        for record in self.measurements.records:
            if record["frame"] >= self.path.frames:
                break
            segment = self.path.segment_of(record["frame"])
            # While it moves the palm lags a ramp of its target by speed times kd / kp of the slowest
            # actuator: the bound is that of the fastest commanded speed since the last hold ended.
            fastest = max(fastest, segment.speed)
            bound = fastest * constant + tolerance
            last_of_hold = segment.hold and record["frame"] == segment.last_frame
            if last_of_hold and segment.phase == PINCH:
                bound, fastest = tolerance, 0.0
                command = record["commanded_half_opening_rad"]
                if max(abs(angle - command) for angle in record["jaw_half_opening_rad"]) > hinge_tolerance:
                    first, second = record["jaw_half_opening_rad"]
                    problems.append(
                        f"frame {record['frame']} ({segment.phase}, end of {segment.label}): the jaws are open by "
                        f"{first:.5f} and {second:.5f} rad, commanded {command:.5f} rad"
                    )
            elif last_of_hold:
                bound, fastest = tolerance, 0.0
                difference = record["jaw_depth_m"] - record["commanded_depth_m"]
                if abs(difference) > tolerance:
                    problems.append(
                        f"frame {record['frame']} ({segment.phase}, end of {segment.label}): the jaws are "
                        f"{1e3 * record['jaw_depth_m']:.4f} mm deep, commanded {1e3 * record['commanded_depth_m']:.4f} mm"
                    )
            if record["palm_position_error_m"] > bound:
                problems.append(
                    f"frame {record['frame']} ({segment.phase}, {segment.label}): the palm is "
                    f"{1e3 * record['palm_position_error_m']:.4f} mm from its target, allowed {1e3 * bound:.4f} mm"
                )
        rejected = sum(item.get("rejected_intervals", 0) for item in phases.values())
        if rejected or self.measurements.failures:
            problems.append(
                f"{rejected} rejected intervals and {len(self.measurements.failures)} failures without a contact"
            )
        moved = max(record["surface_displacement_m"] for record in self.measurements.records)
        if moved > tolerance or any(record["tissue_force_N"] != 0.0 for record in self.measurements.records):
            problems.append(
                f"the tissue surface moved {1e3 * moved:.4f} mm or carried a contact force without a contact"
            )
        if problems:
            raise AssertionError(
                "; ".join(problems[:8]) + (f"; and {len(problems) - 8} more" if len(problems) > 8 else "")
            )
        return tolerance

    def test_final(self):
        """Self-check of a run; prints one line of JSON and writes the report.

        With the contact off the scenario must reach its end with finite states, the tool on its
        commanded path and the measured depth of the jaws equal to the commanded one (both within
        :meth:`_tolerance` at the end of every hold; in the pinch the jaw hinges on their command
        instead of the depth, :meth:`_check_off`), and in the fast approach the tool itself, not
        its command, must travel more than the contact radius of the jaw tip in the physics
        interval in which the tip crosses the tissue surface. With a contact, what happens is reported:
        failed phases do not fail the process, since they are what the rig is there to show. In
        both cases a failure of the rigid solver, a nonfinite state, a graph capture or a kernel
        module load after start-up do. Where the pinch was run with its contact (the swept one),
        the rows of its self-check (``Measurements.pinch_rows``) fail the process too; a slip is
        reported with them. A row without a verdict (frames that were not run, or the work
        balance of a pair that slipped) does not fail the process.
        """
        if not self.args.measure:
            raise AssertionError("The self-check needs the measurements: run it without --no-measure")
        if not self.completed:
            raise AssertionError("No frame completed")
        phases = self.measurements.phases()
        integrity = self.scene.tissue_integrity()
        bodies = self._last["body_q"] if self._last is not None else self.scene.state_0.body_q.numpy()
        problems = []
        if self.halted is not None:
            problems.append(f"the run halted: {self.halted}")
        if not integrity["finite"] or not np.isfinite(bodies).all():
            problems.append("nonfinite state")
        if not integrity["minimum_volume_ratio"] > 0:
            problems.append("a tissue tetrahedron is inverted")
        if not integrity["held_nodes_at_rest"]:
            problems.append("held tissue nodes moved")
        census = self.scene.census()
        if census["capture_census_status"] == "checked":
            late = {
                name: census[name]
                for name in ("captures_after_startup", "solver_captures_after_startup", "module_loads_after_startup")
            }
            late["startup_unlaunched_graphs"] = census["startup_unlaunched_graphs"]
            if any(late.values()):
                problems.append(
                    f"graph captures or kernel module loads after start-up: {late}, late sites "
                    f"{census['late_sites']}, late kernel modules {census['late_modules']}"
                )
        else:
            print(f"Captures and kernel module loads after start-up are not checked: {census['capture_census_status']}")
        # The whole scenario: from the start of the path to its end.
        whole = self.frame_limit >= self.path.frames and self.first_frame == 0
        tolerance = None
        travel = self.fast_travel()
        if not problems and self.args.contact == "off":
            if whole and self.frame < self.path.frames:
                problems.append(f"the scenario ended at frame {self.frame} of {self.path.frames}")
            tolerance = self._check_off(phases)
            # What the tool did, not what it was commanded: its travel in the interval of the crossing.
            crossing = travel["in_crossing_interval"] if travel else None
            if whole and crossing is None:
                problems.append("the jaw tip did not cross the tissue surface in the fast approach")
            elif whole and not crossing > 1.0:
                problems.append(
                    f"the jaw tip travelled {crossing:.3f} of its contact radius in the interval in which it crossed "
                    "the tissue surface, not more than one"
                )
        rows = self.pinch_rows()
        for row in rows or ():
            if row["passed"] is False:
                problems.append(f"pinch, row {row['row']} ({row['name']}) is not met: {row['numbers']}")
        result = {
            "physical_checks_passed": not problems,
            "pinch_rows": None if rows is None else {str(row["row"]): row["passed"] for row in rows},
            "contact": self.args.contact,
            "first_frame": self.first_frame,
            "frames": self.completed,
            "physical_time_s": self.sim_time,
            "scenario_completed": bool(whole and self.frame >= self.path.frames and not self.measurements.failures),
            "failed_phases": [failure["phase"] for failure in self.measurements.failures],
            "rejected_intervals": self.scene.rejected_intervals,
            "tolerance_m": tolerance,
            "run_seconds": self.run_seconds,
            "fast_tip_travel_per_interval_jaw_radii": travel,
            "step_ms_median": self.timing().get("step_ms_median"),
            "steps_over_66_7_ms": len(self.timing().get("steps_over_limit", [])),
            **{
                name: census[name]
                for name in census
                if name.endswith("_after_startup") or name == "capture_census_status"
            },
        }
        if self.args.report:
            self.write_report(self_check=result, problems=problems)
        print(json.dumps(result))
        if problems:
            raise AssertionError("; ".join(problems))

    def close(self):
        # Gives the objects of start-up back to the collector, then releases the scene.
        GraspExample._release(self.scene.close)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.description = __doc__
        # 0: the whole scripted path.
        parser.set_defaults(num_frames=0)
        parser.add_argument("--substeps", type=int, default=4, help="Physics intervals per 60 Hz frame.")
        parser.add_argument(
            "--rigid-backend",
            choices=("auto", "native", "mujoco-warp"),
            default="auto",
            help="Rigid solver of the tools: MuJoCo Warp (needs a CUDA device) or native CPU MuJoCo; auto takes "
            "MuJoCo Warp on a CUDA device.",
        )
        parser.add_argument(
            "--contact",
            choices=CONTACTS,
            default="off",
            help="What tools and tissue exchange: off (nothing: the tools pass through the tissue), first (the "
            "first-version law of the grasp scene: nearest-point forces one interval late, carries light touches "
            "only) or swept (the contact on the sweep of the capsules: surface nodes under a capsule are placed on "
            "it and held through the next interval; tissue that is held or cannot follow stops the tool).",
        )
        parser.add_argument(
            "--speed",
            type=float,
            default=SPEED,
            help="Speed of the motions that touch the tissue [m/s] (default: the arc speed of fem_needle_grasp).",
        )
        parser.add_argument(
            "--hold-seconds",
            type=float,
            default=HOLD_SECONDS,
            help="Length of every hold [s] (default: the settling time of fem_needle_grasp).",
        )
        parser.add_argument(
            "--press-depth",
            type=float,
            default=PRESS_DEPTH,
            help="Commanded depth of the first press and of the press at the held border [m].",
        )
        parser.add_argument(
            "--deep-depth",
            type=float,
            default=DEEP_DEPTH,
            help="Commanded depth of the second press and of the slide [m].",
        )
        parser.add_argument("--slide-length", type=float, default=SLIDE_LENGTH, help="Length of the slide [m].")
        parser.add_argument(
            "--fast-depth",
            type=float,
            default=FAST_DEPTH,
            help="Depth of the jaws at the end of the first frame of the fast approach and in the hold after it "
            "[m]; between them the target goes on until the tool has crossed the surface.",
        )
        parser.add_argument(
            "--pinch-lift",
            type=float,
            default=PINCH_LIFT,
            help="How far the closed jaws lift the ridge that they pinch [m].",
        )
        parser.add_argument(
            "--jaw-tilt-deg",
            type=float,
            default=0.0,
            help="Turn of the jaws about their closing direction towards the horizontal [degrees]; 0 keeps them on "
            "the line from the tool's port, as with a straight wrist.",
        )
        parser.add_argument(
            "--measure",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Observe every frame after its step (depths of the tool parts, dimple, contact forces, tracking) "
            "and print the summary by phase; --no-measure runs the steps alone and observes only the frames of "
            "the fast approach, whose end depends on where the tool is.",
        )
        parser.add_argument(
            "--measure-intervals",
            action="store_true",
            help="Also observe the end of every physics interval (how far the drawn capsules are inside the tissue, "
            "the contact forces, the actuator demand): a measurement run, slower, with reads in every interval.",
        )
        parser.add_argument(
            "--start-phase",
            choices=RESTART_PHASES,
            default=None,
            help="Begin the run at a later phase: the tool is placed at rest where that phase is clear of the "
            "tissue, as after a failed phase (for the pinch: after its lift-off from the hold of the fast "
            "approach). --num-frames then counts from there.",
        )
        parser.add_argument("--report", help="Write the measurements, the path and the frame times as JSON.")
        parser.add_argument("--overview", action="store_true", help="Frame the whole block and the trocar ports.")
        return parser


def main():
    parser = Example.create_parser()
    options = parser.parse_args()
    if options.substeps < 1 or options.num_frames < 0:
        parser.error("Substeps must be positive and the frame limit nonnegative")
    values = (
        options.speed,
        options.hold_seconds,
        options.press_depth,
        options.deep_depth,
        options.slide_length,
        options.fast_depth,
        options.pinch_lift,
    )
    if not np.isfinite(values).all() or min(values) <= 0:
        parser.error("Speed, hold time, depths, slide length and the lift of the pinch must be finite and positive")
    if options.deep_depth < options.press_depth:
        parser.error("The second press must be at least as deep as the first")
    if not np.isfinite(options.jaw_tilt_deg) or abs(options.jaw_tilt_deg) >= 90:
        parser.error("The jaw tilt must be finite and below 90 degrees")
    viewer, args = newton.examples.init(parser)
    example = None
    try:
        example = Example(viewer, args)
        newton.examples.run(example, args)
    except BaseException:
        viewer.close()
        raise
    finally:
        if example is not None:
            example.close()


if __name__ == "__main__":
    main()
