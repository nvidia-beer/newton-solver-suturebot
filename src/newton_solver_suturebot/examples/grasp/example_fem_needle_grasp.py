# SPDX-License-Identifier: Apache-2.0
"""Insert a physically grasped needle with articulated dVRK tools, FEM tissue and thread.

The tools pivot at fixed trocar ports and use bounded joint IK with finite
MuJoCo actuators. Jaw friction holds the free rigid needle. The procedural
controller drives the needle through the tissue until it is centred, lifts it to
show the tissue deformation, lets go, takes the needle by its tip and pulls it
on along its own curve until it is out of the tissue; the thread on its tail
follows it through the track it has cut. Then the thread is tied at its entry
(a stand-in for a knot), and the tool moves the needle on until the thread is
as tight as the tissue allows, and holds: the wound narrows at its top. The
thread is then tied where it comes out of the tissue as well, and the tool
takes the needle back until the thread to it is slack: the stitch keeps a part
of that narrowing by itself. No trained policy or Isaac Lab is used. One scene
runs in this process; several scenes each run in a worker process.

By default the tools push the tissue where they touch it (the swept tool/tissue
contact, --tool-tissue-contact), and with one scene in the window a step plans
for a picture no later than 66.7 ms after the one before (--frame-budget).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import weakref

import newton
import newton.examples
import numpy as np
import warp as wp
from newton.viewer import ViewerGL

from newton_solver_suturebot.examples._tissue_markers import TissueMarkers
from newton_solver_suturebot.examples.grasp._grasp_procedure import ProceduralGraspController
from newton_solver_suturebot.examples.grasp.parallel import LocalSutureScene, ParallelSutureScenes, SutureSceneConfig
from newton_solver_suturebot.examples.needle.example_fem_needle_insertion import (
    COUPLING_TRIAL_LIMIT,
    trial_budget_argument,
)

# The settings of this example that are choices: values that runs selected, not values that
# follow from the model. Each is named here once, with what it selects and what measured it;
# the numbers of the runs are in docs/usage.md, under the heading given. The scene (the parser
# of GraspScene), the insertion example and SutureSceneConfig keep their own values, given in
# brackets, so a caller of SutureSceneConfig.example_arguments gets what it asks for.
#
# Relative interface force tolerance of the needle/tissue coupling (scene: 1e-3) and trials of
# one coupling loop after which its best evaluated trial is kept (scene: no budget). Of the
# tolerances 1e-3 and 1e-2 and the budgets 6, 12 and none, run over 870 frames, these kept the
# needle path within the difference between two runs ("Coupling tolerance and trial budget").
# The budget does not follow from the correction of the loop: a captured loop without a
# multiplier update tightens the tissue-solver tolerance at trial 9 and starts a new Anderson
# history there, so the load of trial 12 is proposed from trials 9 to 11.
COUPLING_RELATIVE_TOLERANCE = 1e-2
TRIAL_BUDGET = 12
# Surface events between the device trials and the host oracle, as arguments of the scene
# (scene: tip-path, restart): a step that touches the puncture threshold is searched on the
# device, and the oracle evaluates the device trial it is handed instead of repeating it
# ("Surface events on the device and response columns per step").
SURFACE_EVENTS = ("--event-search", "puncture", "--event-oracle", "device-trial")
# Support nodes whose response columns the response-basis guess solves per coupled step after
# the track has grown (scene: 0, every missing column at once); measured with the surface
# events, under the same heading.
RESPONSE_BASIS_NODES = 1

# Defaults of this example that are not numbers of a run.
#
# Contact between the tool capsules and the tissue (--tool-tissue-contact): "swept", the contact
# on the sweep of the capsules, or False / "off" (the scene and SutureSceneConfig keep it off).
# With it the tools push the tissue instead of passing through it ("Tool/tissue contact").
TOOL_TISSUE_CONTACT = "swept"
# Frame time budget of the window [ms] (--frame-budget): the time of one picture at 15 pictures
# per second, 1000 / 15 ms written to 0.1 ms; 0 runs whole frames. Where a budget is used is
# decided by Example._frame_budget, and the smallest part of a frame after which a step may
# return by Example._frame_unit. The scene has no budget of its own: a deadline is an argument
# of LocalSutureScene.step ("Frame time budget").
FRAME_BUDGET_MS = 66.7
# Whether the thread follows the needle's track through the tissue (--thread-tracks): on, as in
# the scene and in SutureSceneConfig. The pull then takes the needle out of the tissue.
THREAD_TRACKS = True
# Needle arc speed [m/s] and closed-jaw settling time [s] of the procedure (--speed,
# --settle-seconds): choices of the demonstration. The self-test asks for the complete
# procedure only at these values (Example.test_final).
ARC_SPEED = 0.003
SETTLE_SECONDS = 0.5


def _pull_through(args):
    """Whether the tool takes the needle by its tip and pulls it on along its curve after the lift.

    With ``--lift-height 0`` the example stays the insertion alone, as before, whatever the pull says.
    """
    return bool(getattr(args, "pull_through", False)) and getattr(args, "lift_height", 0.0) > 0


def _tip_lift_height(args):
    """Height of the lift by the tip [m]: the option; without it none with the pull or without a first lift, else 5 mm."""
    height = getattr(args, "tip_lift_height", 0.0)
    if height is None:
        return 0.0 if _pull_through(args) or not getattr(args, "lift_height", 0.0) else 0.005
    return height


def threaded_in_order(passages, threaded):
    """Of ``threaded`` eyelets, how many each passage has when the thread takes the eyelets in the order they were made.

    A scene reports how many eyelets its thread goes through, not which ones. The thread hangs on
    the needle's tail, and the tail goes through the track behind the tip, so it passes the
    eyelets in the order in which the tip made them: the first ``threaded`` eyelets are taken for
    the threaded ones. An eyelet that the tail went past without threading it would be counted
    here as threaded, and the last one made as waiting; the number itself stays right.

    Args:
        passages: Number of eyelets of every passage, in the order the passages were made.
        threaded: Number of threaded eyelets of the scene.
    """
    counts, left = [], int(threaded)
    for size in passages:
        counts.append(min(left, int(size)))
        left -= counts[-1]
    return counts


def thread_line(thread):
    """The panel line of one scene's thread report: where the thread goes through the tissue.

    The eyelets that the thread goes through of those that the needle's track has made, the
    same per passage (:func:`threaded_in_order`), the rest length of thread inside the tissue,
    and the eyelets that found no room in the reserve, if any. None without a report and for
    a scene whose thread has no eyelet slots (it stays on the surface).
    """
    if not thread or not thread.get("eyelet_reserve", (0, 0))[0]:
        return None
    made, threaded, passages = thread["eyelets"], thread["eyelets_threaded"], thread["eyelet_passages"]
    if not made:
        line = "Thread: no eyelets yet, the needle has made no track"
    else:
        split = ", ".join(
            f"passage {index + 1}: {done} of {size}"
            for index, (done, size) in enumerate(zip(threaded_in_order(passages, threaded), passages, strict=True))
        )
        inside = 1.0e3 * thread["thread_inside_tissue_m"]
        line = f"Thread: {threaded} of {made} eyelets threaded ({split}), {inside:.1f} mm inside the tissue"
    refused = thread.get("eyelets_refused", 0)
    return line + (f"; {refused} eyelets found no room in the reserve" if refused else "")


def stitch_line(item):
    """The panel line of one scene's stitch: the wound's width, where the thread is tied, and its pull.

    The width of every gap that the planned needle path crosses, at the top and at the track,
    with the widths of the reference shape (``wound_widths`` of the observation); where the
    thread is tied to the tissue: nowhere, at its entry (one tie), or at its entry and at its
    exit (two); the thread's force on the needle and its largest force on an eyelet
    (``thread_tension``). None for a scene without such a gap or without a thread.
    """
    wound, tension = item.get("wound_widths"), item.get("thread_tension")
    if not wound or not tension:
        return None
    widths = "; ".join(
        f"{1.0e3 * gap['top_m']:.2f} mm wide at the top ({1.0e3 * gap['rest_top_m']:.2f} at rest), "
        f"{1.0e3 * gap['track_m']:.2f} mm at the track ({1.0e3 * gap['rest_track_m']:.2f})"
        for gap in wound
    )
    tied = ("not tied", "tied at its entry", "tied at its entry and at its exit")[min(int(tension["ties"]), 2)]
    pull = (
        f"pulls the needle with {1.0e3 * tension['needle_force_N']:.1f} mN, "
        f"an eyelet with {1.0e3 * tension['eyelet_force_N']:.1f} mN at most"
    )
    return f"Stitch: wound {widths}; thread {tied}, {pull}"


class Example:
    """A normal Newton viewer over one physical scene in its own process, or over several in workers."""

    _instances = weakref.WeakSet()
    # Radius of the entry and exit markers [m]: twice the needle wire radius of the dVRK scene.
    marker_radius = 0.0006
    marker_colors = {"entry": (0.1, 0.8, 0.2), "exit": (0.15, 0.35, 0.95)}
    # Colour of a marker of a scene while its thread is tied there (a stand-in for a knot): the first entry
    # marker with one tie, and the last exit marker as well with two.
    tie_color = (1.0, 0.8, 0.1)
    # Markers one scene can show at a time, per passage of the planned arc through the tissue:
    # the entry and the exit of the passage, four times over for crossings that the plan does not
    # have (a tip that goes through the surface again under a lift or a new grasp). The reserve is
    # allocated at start-up; beyond it the oldest marker of the scene makes room for the new one.
    marker_reserve_per_passage = 8
    # Frame time budget of this run [s]: the longest time from the start of one step() to the
    # start of the next that the example plans for. None runs whole frames. The constructor
    # takes it from --frame-budget (FRAME_BUDGET_MS) where a budget is used (_frame_budget);
    # None is what an example has before that.
    frame_budget = None
    # Unit of the frame budget of this run: the smallest part of a frame after which a step()
    # may return. "attempt": one attempt of a coupled step. "host-trial": also one
    # host-controlled trial inside an attempt. The constructor takes "host-trial" where that
    # unit applies (_frame_unit); "attempt" is what an example has before that and wherever
    # no budget is in use.
    frame_unit = "attempt"
    # Calls of step() that returned with their frame unfinished.
    unfinished_calls = 0
    # Time [s] the last render() took and the longest a render() of this example has taken, and
    # the physical time [s] of every scene at the start of the frame that the last step() left
    # unfinished.
    _draw_seconds = 0.0
    _draw_longest = 0.0
    _frame_start = None
    # Per scene, what the stitch left where the controller passed (_follow_stitch): None before a tightening.
    _stitch_record = None

    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        self.sim_time, self.completed = 0.0, 0
        self.last_phases = None
        self.halted = None
        self.frame_budget = self._frame_budget(viewer, args)
        self.frame_unit = self._frame_unit(args, self.frame_budget)
        self.config = SutureSceneConfig(
            device=str(wp.get_device(args.device)),
            control_mode="physical_grasp",
            rigid_backend=getattr(args, "rigid_backend", "mujoco-warp"),
            substeps=args.substeps,
            example_arguments=self._coupling(args),
            **self._bite(getattr(args, "bite_length", 0.0)),
        )
        # One scene runs in this process and CUDA context; several run in a worker each.
        self.pool = (
            LocalSutureScene(self.config)
            if args.num_envs == 1
            else ParallelSutureScenes(self.config, num_envs=args.num_envs)
        )
        # One entry once this example has frozen the objects of start-up: its release gives back
        # only what it froze.
        self._froze = []
        self._cleanup = weakref.finalize(self, self._release, self.pool.close, self._froze)
        self._instances.add(self)
        try:
            self.observations, self.geometry = self.pool.observe(), self.pool.geometry()
            self.controller = (
                ProceduralGraspController(
                    self.geometry,
                    self.config,
                    args.speed,
                    args.settle_seconds,
                    lift_height=getattr(args, "lift_height", 0.0),
                    tip_lift_height=self._tip_lift_height(args),
                    pull_through=self._pull_through(args),
                )
                if args.controller == "procedural"
                else None
            )
            self.render_scene = None
            if args.viewer != "null":
                self._build_view()
                # A viewer that counts its frames up to a limit would spend one of them on this draw.
                if not self._frame_limited(viewer):
                    self._warm_view()
            print("Space: pause/resume | .: single step | Reset: restart | Esc: close", flush=True)
            if self._freezes_startup(viewer, args):
                # Before start-up ends: what the collection frees here is freed during start-up.
                self._freeze_startup_objects()
                self._froze.append(True)
            # Start-up ends here: with one scene the viewer's kernels load in the scene's process.
            self.pool.finish_startup()
        except BaseException:
            self.close()
            raise

    _pull_through = staticmethod(_pull_through)
    _tip_lift_height = staticmethod(_tip_lift_height)

    @classmethod
    def _frame_budget(cls, viewer, args):
        """The frame time budget of this run [s], or None for whole frames.

        The budget is that of ``--frame-budget``; arguments without the option have the
        default of the example, :data:`FRAME_BUDGET_MS`, and zero is no budget. It is for one
        scene in the window of the OpenGL viewer on this machine
        (:meth:`_gl_window`): the time between two calls of :meth:`step` is there the time
        between two pictures. Whole frames are kept, whatever ``--frame-budget`` says, where a
        call that returns with its frame unfinished would be counted as a frame (the self-test
        and a viewer that ends the run after a number of frames, :meth:`_frame_limited`) or
        recorded or sent as one (the Rerun and Viser viewers, which get one sample per call),
        and in the RTX window, which shows the pictures of a renderer that runs beside the
        calls and with which the budget was never run. Several scenes step in their worker
        processes, always by whole frames.

        A budget in use is named in one line, with the option that turns it off. Where the
        budget is not used, the reason is printed for a budget other than the default: the
        default one is there without having been asked for, and a run that keeps whole frames
        says nothing about it.
        """
        milliseconds = getattr(args, "frame_budget", FRAME_BUDGET_MS)
        if not milliseconds:
            return None
        if args.num_envs != 1:
            reason = "several scenes step by whole frames in their worker processes"
        elif getattr(args, "test", False):
            reason = "the self-test runs whole frames"
        elif cls._frame_limited(viewer):
            reason = "this viewer ends the run after a number of calls, which are whole frames"
        elif not cls._gl_window(viewer):
            reason = (
                f"it is for the window of the OpenGL viewer; this viewer ({type(viewer).__name__}) keeps whole frames"
            )
        else:
            print(
                f"Frame budget {milliseconds:g} ms: a step may return with its frame unfinished and go on with it "
                "in the next call (--frame-budget 0 runs whole frames)",
                flush=True,
            )
            return 1.0e-3 * milliseconds
        if milliseconds != FRAME_BUDGET_MS:
            print(f"Frame budget not used: {reason}", flush=True)
        return None

    @classmethod
    def _freezes_startup(cls, viewer, args):
        """Whether this run freezes the objects of start-up (:meth:`_freeze_startup_objects`): only with a view.

        A run with a view draws between its frames, and the first full collection after
        start-up would fall into one of them. Without a view (the null viewer) and in a
        self-test without a window (``--test`` with a viewer that records or draws off screen)
        nothing is frozen: no full collection falls into the frames of such a run (none in the
        2400 frames of the headless demo, with and without the freeze), the collector keeps its
        usual reach, and the release of the example does not undo a freeze of the host program.
        """
        if args.viewer == "null":
            return False
        return not getattr(args, "test", False) or cls._window(viewer)

    @staticmethod
    def _frame_unit(args, budget):
        """The unit of the frame budget of this run: ``"host-trial"`` where it applies, else ``"attempt"``.

        With a frame budget in use (:meth:`_frame_budget`) on the GPU rigid backend the unit is
        the host trial: a step may return between two host-controlled trials of an attempt as
        well as between two attempts. The swept tool/tissue contact allows that: it changes its
        state only in the step after an acceptance, and an attempt reads its forces and its
        holds as that step left them, in one call or in several. On the native rigid backend
        the rigid data hold a trial while an attempt is open, and the observation of a call
        that left one open would read the jaw contact force and the native clock of that
        trial: the unit there is the attempt, and the reason is printed.

        Where no budget is in use (the self-test, a viewer without the window, several scenes,
        ``--frame-budget 0``) there is no unit either, and nothing is printed.
        """
        if budget is None:
            return "attempt"
        if getattr(args, "rigid_backend", "mujoco-warp") != "mujoco-warp":
            print(
                "Frame budget unit host-trial not used, the unit stays the attempt: on the native rigid backend an "
                "observation inside an attempt would read the contacts and the clock of a trial",
                flush=True,
            )
            return "attempt"
        print(
            "Frame budget unit host-trial: a step may also return inside an attempt, between two "
            "host-controlled trials",
            flush=True,
        )
        return "host-trial"

    @staticmethod
    def _freeze_startup_objects():
        """Take the objects of start-up out of the reach of Python's cyclic garbage collector.

        Start-up leaves more than a million long-lived objects: kernel modules, the scene, the
        view. A full collection walks every object the collector tracks, and the first one
        after start-up otherwise falls into a frame of the run. Here the garbage of start-up is
        collected once and what survives is moved to the collector's permanent generation, which
        no collection walks. Reference counting frees a frozen object as before; only a
        reference cycle among frozen objects stays until :meth:`_release` gives them back.

        The example is frozen with them. An example that is part of a reference cycle is therefore
        not released when it is dropped: no collection reaches a frozen cycle, so :meth:`_release`
        does not run and the scenes stay open until the interpreter exits. A caller calls
        :meth:`close`, or holds the example in no cycle, as the run loop and the viewer's reset do.

        The second collection finds nothing to walk. It is there for the collector's count of
        long-lived objects, which a freeze leaves at the count of start-up: the collector puts
        a full collection off until a quarter of that count has gathered, and would then walk
        that quarter in one frame. With the count at zero, full collections keep their usual
        spacing and walk only what the run itself has kept.
        """
        gc.collect()
        gc.freeze()
        gc.collect()

    @staticmethod
    def _release(close_pool, froze=None):
        """Give the frozen objects back to the collector, then release the scenes.

        Runs once, when the example is closed or dropped. A viewer's reset drops the example
        and collects before it builds the next one; that collection must reach this one's
        objects.

        ``froze`` is the example's record of its own freeze: empty for an example that froze
        nothing (a run without a view, :meth:`_freezes_startup`, or a start-up that failed
        before the freeze), and nothing is given back then. A caller that froze the objects
        itself passes no record.

        ``gc.unfreeze()`` acts on the whole process: it also gives back what another example or
        the host program froze, and the next full collection walks those objects again. The
        freeze is made for one example per process, which is what the run loop keeps.
        """
        if froze is None or froze:
            gc.unfreeze()
        close_pool()

    @staticmethod
    def _coupling(args):
        """Arguments for the scene's parser: coupling tolerance, trial budget, surface events and tool/tissue contact."""
        tolerance = getattr(args, "coupling_relative_tolerance", COUPLING_RELATIVE_TOLERANCE)
        budget = getattr(args, "trial_budget", TRIAL_BUDGET)
        arguments = ("--coupling-relative-tolerance", repr(float(tolerance)))
        if budget is not None:
            arguments += ("--trial-budget", str(int(budget)))
        arguments += SURFACE_EVENTS
        # A value that is the scene's own default is not forwarded.
        basis_nodes = int(getattr(args, "response_basis_nodes", RESPONSE_BASIS_NODES))
        if basis_nodes:
            arguments += ("--response-basis-nodes", str(basis_nodes))
        contact = getattr(args, "tool_tissue_contact", TOOL_TISSUE_CONTACT)
        if contact == "swept":
            arguments += ("--tool-tissue-contact", "swept")
        elif contact not in (False, None, "off"):
            raise ValueError("The contact between the tools and the tissue of this example is 'swept' or 'off'")
        if not getattr(args, "thread_tracks", THREAD_TRACKS):
            arguments += ("--no-thread-tracks",)
        return arguments

    # Reset grasp for the 10 mm bite: fraction of the arc from the tip, twist about the needle
    # tangent and bite angle [rad]. A scan of the tool's inverse kinematics along the whole arc
    # of that bite, for the dVRK port layout, finds twists of -105 to -90 degrees with bite
    # angles of -30 to -15 degrees reachable; this is the middle of that range, -97.5 and -22.5
    # degrees, written in radians to four digits. The fraction is the scene's own (0.8).
    deep_bite_grasp = (0.8, -1.7017, -0.3927)

    @classmethod
    def _bite(cls, length):
        """Scene settings for an entry-to-exit distance; zero keeps the recorded stitch.

        The recorded entry and exit labels are 5.4 mm apart. A needle of 9.45 mm radius through
        them passes 0.39 mm under the surface: it scrapes the tissue. The points are moved apart
        along their own line, about their midpoint; 10 mm gives a bite 1.4 mm deep that holds
        the needle when the jaws let go.
        """
        if not length:
            return {}
        from newton_solver_suturebot.examples.grasp._dvrk_layout import load_source_stitch

        stitch = load_source_stitch()
        entry, exit_point = np.asarray(stitch["entry"][:2]), np.asarray(stitch["exit"][:2])
        middle, direction = (entry + exit_point) / 2, (exit_point - entry) / np.linalg.norm(exit_point - entry)
        return {
            "entry_xy": tuple(middle - 0.5 * length * direction),
            "exit_xy": tuple(middle + 0.5 * length * direction),
            "initial_grasp": cls.deep_bite_grasp,
        }

    def _halt(self, reason, diagnostics=None):
        """Stop stepping after a failure: raise without a viewer, freeze the last state with one."""
        if self.render_scene is None:
            raise RuntimeError(reason if diagnostics is None else f"{reason}: {diagnostics}")
        self.halted = reason
        print(f"Simulation halted at {self.sim_time:.2f} s: {reason}", flush=True)

    def _build_view(self):
        from newton_solver_suturebot.examples.grasp._grasp_render import PhysicalSurgicalRenderScene
        from newton_solver_suturebot.examples.grasp._grasp_view import (
            fit_inspection_camera,
            framing_points,
            update_scene,
        )

        builder = newton.ModelBuilder(up_axis="Z")
        self.render_scene = PhysicalSurgicalRenderScene(builder, self.geometry, self.args.num_envs, self.config.device)
        self.model = builder.finalize(device=self.config.device)
        self.state_0 = self.model.state()
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((self.args.spacing, self.args.spacing, 0.0))
        self.viewer.picking_enabled = False
        self.viewer.show_particles = False
        world_offsets = self.viewer.world_offsets.numpy()
        self.world_offsets = world_offsets
        self.particle_offsets = world_offsets[self.model.particle_world.numpy()]
        update_scene(self.render_scene, self.state_0, self.observations, particle_offsets=self.particle_offsets)
        # The entry and exit markers ride on the drawn tissue surface of their scene: the surface
        # triangles of the first scene, and the particles each scene has in the drawn state.
        self.markers = TissueMarkers(
            self.viewer,
            "needle/entry_exit",
            radius=self.marker_radius,
            device=self.config.device,
            per_world=self.marker_reserve_per_passage * self.geometry["insertion_plan"]["expected_passages"],
            worlds=self.args.num_envs,
            triangles=self.render_scene.surface.triangles,
            world_nodes=self.model.particle_count // self.args.num_envs,
        )
        # Events of every scene that the markers have been asked about (render()), the markers attached
        # for them, which of those are the first entry and the last exit, and whether each shows a tie.
        self._events_drawn = [0] * self.args.num_envs
        self._markers_attached = [0] * self.args.num_envs
        self._entry_marker = [None] * self.args.num_envs
        self._exit_marker = [None] * self.args.num_envs
        self._tie_shown = [False] * self.args.num_envs
        self._lock_shown = [False] * self.args.num_envs
        if hasattr(self.viewer, "camera"):
            if self.args.overview:
                points = framing_points(self.render_scene, self.geometry, self.observations, world_offsets)
            else:
                # Start close enough to inspect jaw contact and insertion.
                # The overview option includes the full shafts and trocar ports.
                points = np.concatenate(
                    [
                        np.concatenate((self.render_scene.surface.rest_positions, item["thread_positions"])) + offset
                        for item, offset in zip(self.observations, world_offsets, strict=True)
                    ]
                )
            # The field of view that fit_inspection_camera frames the points for: its vertical
            # aperture of 15.2908 mm at its focal length of 12 mm.
            self.viewer.camera.fov = math.degrees(2 * math.atan(15.2908 / 24.0))
            eye, target = fit_inspection_camera(points, view_direction=(0.0, -1.0, 0.65))
            self.viewer.set_camera(wp.vec3(eye))
            self.viewer.camera.look_at(target)

    @staticmethod
    def _frame_limited(viewer):
        """Whether the viewer ends the run by itself after a number of the frames it is given.

        The USD viewer counts the frames it records up to ``num_frames``; the OpenGL and RTX
        viewers do so only without a window (a window runs until it is closed and ignores the
        limit). A viewer without ``num_frames`` has no limit.
        """
        if getattr(viewer, "num_frames", None) is None:
            return False
        return bool(getattr(viewer, "_headless", True))

    @staticmethod
    def _window(viewer):
        """Whether the viewer shows a window on this machine: the OpenGL and RTX viewers do unless they are headless."""
        return not getattr(viewer, "_headless", True)

    @staticmethod
    def _gl_window(viewer):
        """Whether the viewer is Newton's OpenGL viewer with a window on this machine."""
        return isinstance(viewer, ViewerGL) and not getattr(viewer, "_headless", True)

    def _single_step(self):
        """Whether this call is the single step of a paused viewer (its "." key or its Step button).

        The run loop calls :meth:`step` only when the viewer says that the loop should step. A
        paused viewer says so once per request, and stays paused.
        """
        paused = getattr(self.viewer, "is_paused", None)
        return paused is not None and bool(paused())

    def _warm_view(self):
        """Draw the start state once, so that the first draw is not part of the first frame.

        These are the launches of ``render()``: the state in every frame, and those of the entry
        and exit markers (``TissueMarkers.warm``): the search that binds a new marker to the
        tissue surface, the positions of the bound markers, and the batch itself, logged hidden
        with its whole reserve; ``render()`` shows it when the needle has gone in. The
        constructor does this only for a viewer without a frame limit (:meth:`_frame_limited`):
        a limited viewer counts this draw as a frame and would simulate one frame less than it
        was asked for.
        """
        self.viewer.begin_frame(0.0)
        self.viewer.log_state(self.state_0)
        self.markers.warm(self.state_0.particle_q)
        self.viewer.end_frame()

    def step(self):
        """Advance the scenes by one frame, or with a frame budget by as many attempts of one as fit.

        With :attr:`frame_budget` the scene gets a deadline: the start of this call, before the
        controller acts, plus the budget, less the longest time a drawing of this example has
        taken. The scene runs one attempt of a coupled step and starts another one, the next
        substep or the retry of a rejected attempt, only if a substep that uses up its trials
        still ends by then, in the worst case that the run has measured (``simulate`` of the
        scene); otherwise this call returns with the frame unfinished and the next call goes on
        with the same frame. The controller acts once per frame, on the state after the frame
        before; the frame counter and the check of the frame's length belong to the call that
        finishes the frame. ``observations`` is the accepted state after every call, which is
        what the viewer draws: a frame that takes several calls is shown as it advances, and
        simulated time falls behind by those calls. The single step of a paused viewer has no
        deadline: it finishes the frame in progress, or runs a whole one.

        With the unit ``host-trial`` (:attr:`frame_unit`) the scene may also return inside an
        attempt, between two host-controlled trials; the observation of such a call is the
        accepted state of before that attempt, and the next call goes on with the attempt.
        """
        if self.halted is not None:
            # A viewer keeps showing the last physical state so that it can be inspected.
            return
        started = time.perf_counter()
        if self.frame_budget is not None and self.pool.frame_unfinished:
            # The frame in progress goes on under the action that started it.
            actions, previous = None, self._frame_start
        else:
            actions = (
                self.controller(self.observations)
                if self.controller is not None
                else np.zeros((self.args.num_envs, self.config.action_dimension), dtype=np.float32)
            )
            if self.controller is not None:
                progress = self.controller.report()
                # A scene whose procedure has ended is named "done": the controller's own phase is
                # then "holding", which reads like a tool that waits for something.
                labels = ["done" if item.get("stage") == "done" else item["phase"] for item in progress]
                phases = tuple((label, item["reason"]) for label, item in zip(labels, progress, strict=True))
                if phases != self.last_phases:
                    print(
                        "Procedural insertion: "
                        + "; ".join(f"{item['env_id']}: {label}" for label, item in zip(labels, progress, strict=True)),
                        flush=True,
                    )
                    self.last_phases = phases
                self._follow_stitch(progress)
                if any(item["phase"] == "stopped" for item in progress):
                    self._halt(f"Procedural insertion stopped: {progress}")
                    return
            previous = self._frame_start = [item["diagnostics"]["physical_time_s"] for item in self.observations]
        # Only the render scene reads the tissue arrays and the thread report of a frame. Without
        # one the workers keep them on the device; test_final asks for the complete state.
        rendered = self.render_scene is not None
        options = {"include_tissue": rendered, "include_thread_diagnostics": rendered}
        # What the controller asks of the scenes beside the actions of a new frame (the tie of the thread).
        # A controller that has no commands asks for none.
        commands = getattr(self.controller, "commands", None)
        asked = commands() if actions is not None and commands is not None else None
        if asked is not None:
            options["commands"] = asked
        if self.frame_budget is not None and not self._single_step():
            # The drawing that follows this call is taken to cost what the longest one has cost.
            options["deadline"] = started + self.frame_budget - self._draw_longest
            if self.frame_unit == "host-trial":
                # Only with a deadline: a call without one finishes its frame, whatever the unit.
                options["unit"] = "host-trial"
        self.observations = self.pool.step(actions, **options)
        unfinished = self.frame_budget is not None and self.pool.frame_unfinished
        for before, item in zip(previous, self.observations, strict=True):
            diagnostics = item["diagnostics"]
            if diagnostics.get("failure_reason") or diagnostics.get("task_status") == "failed":
                self._halt(f"Physical scene failed: {diagnostics.get('failure_reason')}", diagnostics)
                return
            if not unfinished and not np.isclose(
                diagnostics["physical_time_s"] - before, self.config.frame_dt, rtol=0, atol=1e-9
            ):
                raise RuntimeError("A physical scene did not advance the complete requested frame")
        self.sim_time = float(self.observations[0]["diagnostics"]["physical_time_s"])
        if unfinished:
            self.unfinished_calls += 1
        else:
            self.completed += 1

    def _follow_stitch(self, progress):
        """Keep, per scene, the wound's widths and the least tissue volume at two calls of the controller.

        ``slack``: the first call of the stage ``tighten``, with the thread tied at its entry and
        the tool not yet moved. ``tight``: the first call of the stage ``lock``, which ends the
        hold of the tightened thread. Each is ``(top, track)`` [m] of every gap of
        ``wound_widths`` and the ``least_volume_ratio`` of the observation that the controller
        acted on: host values that the observation has anyway. The closure that the stitch
        keeps is measured against them (:meth:`_check_stitched`). A scene that was reset starts
        again without a record.

        Args:
            progress: The controller's report of this call, one entry per scene.
        """
        if self._stitch_record is None:
            self._stitch_record = [None] * len(self.observations)
        for index, (item, observation) in enumerate(zip(progress, self.observations, strict=True)):
            record, reset = self._stitch_record[index], observation.get("reset_count")
            if record is not None and record["reset_count"] != reset:
                record = self._stitch_record[index] = None
            name = {"tighten": "slack", "lock": "tight"}.get(item.get("stage"))
            wound = observation.get("wound_widths")
            if name is None or not wound:
                continue
            if record is None:
                record = self._stitch_record[index] = {"reset_count": reset}
            if name not in record:
                record[name] = {
                    "widths": [(gap["top_m"], gap["track_m"]) for gap in wound],
                    "least_volume_ratio": observation["diagnostics"].get("least_volume_ratio"),
                }

    def render(self):
        started = time.perf_counter()
        # After a failure the scene keeps the last state that was drawn.
        if self.render_scene is not None and self.halted is None:
            from newton_solver_suturebot.examples.grasp._grasp_view import update_scene

            update_scene(self.render_scene, self.state_0, self.observations, particle_offsets=self.particle_offsets)
        self.viewer.begin_frame(self.sim_time)
        if self.render_scene is not None:
            self.viewer.log_state(self.state_0)
            # Where the needle has actually gone into the tissue (green) and come out of it
            # (blue): a marker is bound to the drawn tissue surface when its event is first
            # reported, at the surface point closest to the event, and drawn where that
            # material point is in every frame after it.
            for world, (item, offset) in enumerate(zip(self.observations, self.world_offsets, strict=True)):
                events = item["diagnostics"].get("interaction_events", ())
                if len(events) < self._events_drawn[world]:
                    # The scene was reset: its markers go with its events.
                    self.markers.clear(world)
                    self._events_drawn[world] = self._markers_attached[world] = 0
                    self._entry_marker[world], self._tie_shown[world] = None, False
                    self._exit_marker[world], self._lock_shown[world] = None, False
                for event in events[self._events_drawn[world] :]:
                    if event["kind"] in self.marker_colors:
                        self.markers.attach_point(
                            self.state_0.particle_q,
                            np.asarray(event["position_m"]) + offset,
                            self.marker_colors[event["kind"]],
                            world,
                        )
                        if event["kind"] == "entry" and self._entry_marker[world] is None:
                            self._entry_marker[world] = self._markers_attached[world]
                        if event["kind"] == "exit":
                            # The newest exit is where the thread leaves the tissue: an older one that showed
                            # the second tie is an exit marker again.
                            if self._lock_shown[world]:
                                self.markers.recolor(world, self._exit_marker[world], self.marker_colors["exit"])
                            self._exit_marker[world], self._lock_shown[world] = self._markers_attached[world], False
                        self._markers_attached[world] += 1
                self._events_drawn[world] = len(events)
                # The first entry marker shows the first tie of the thread, which is tied where it goes in;
                # the last exit marker shows the second, which is tied where it comes out.
                ties = int((item.get("thread_tension") or {}).get("ties") or 0)
                tied = ties >= 1
                if tied != self._tie_shown[world] and self._entry_marker[world] is not None:
                    color = self.tie_color if tied else self.marker_colors["entry"]
                    if self.markers.recolor(world, self._entry_marker[world], color):
                        self._tie_shown[world] = tied
                locked = ties >= 2
                if locked != self._lock_shown[world] and self._exit_marker[world] is not None:
                    color = self.tie_color if locked else self.marker_colors["exit"]
                    if self.markers.recolor(world, self._exit_marker[world], color):
                        self._lock_shown[world] = locked
            self.markers.draw(self.state_0.particle_q)
        self.viewer.end_frame()
        self._draw_seconds = time.perf_counter() - started
        self._draw_longest = max(self._draw_longest, self._draw_seconds)

    def gui(self, ui):
        """The example's lines of the viewer's side panel: per scene, where the thread goes through the tissue and the stitch."""
        for item in self.observations:
            for line in (thread_line(item.get("thread_diagnostics")), stitch_line(item)):
                if line is not None:
                    ui.text_wrapped(line if self.args.num_envs == 1 else f"Scene {item['env_id']}: {line}")

    # What must be zero after start-up in a scene whose start-up call ran (the scene's report()).
    census_numbers = ("captures_after_startup", "solver_captures_after_startup", "module_loads_after_startup")

    def _check_census(self, observations):
        """Fail if a scene captured a CUDA graph or loaded a physics kernel module after its start-up.

        A scene is checked when the status of its ``capture_census`` is "checked": its start-up
        call ran and its start-up was marked. Each of :attr:`census_numbers` must then be zero,
        and every graph captured at start-up must have been launched there. A scene whose
        start-up call declined (native rigid backend, no conditional CUDA graphs, CPU device) is
        not checked, and its reason is printed. Loads of the viewer's kernel modules are reported
        and never fail. Diagnostics without a census are not checked.

        Args:
            observations: One explicit observation per scene: an explicit observation brings the
                numbers of late kernel module loads up to date; that of a step does not.

        Returns:
            The entries that the line of :meth:`test_final` gains: the status of every scene, and
            each number summed over the checked scenes (None if no scene is checked).
        """
        names = (*self.census_numbers, "render_module_loads_after_startup")
        statuses, checked, failures = [], [], []
        for item in observations:
            diagnostics = item["diagnostics"]
            census = diagnostics.get("capture_census")
            status = None if census is None else census["status"]
            statuses.append(status)
            if census is None:
                continue
            scene = item.get("env_id", len(statuses) - 1)
            if status != "checked":
                print(f"Scene {scene}: captures and kernel module loads after start-up are not checked: {status}")
                continue
            checked.append(diagnostics)
            numbers = {name: diagnostics.get(name) for name in self.census_numbers}
            numbers["startup_unlaunched_graphs"] = census.get("startup_unlaunched_graphs")
            if any(value != 0 for value in numbers.values()):
                failures.append(
                    f"scene {scene}: {numbers}, late sites {census.get('late_sites')}, "
                    f"late kernel modules {census.get('late_modules')}"
                )
        if failures:
            raise AssertionError(
                "Graph captures or kernel module loads after start-up, or a start-up graph that was never "
                "launched; " + "; ".join(failures)
            )
        totals = {
            name: sum(diagnostics.get(name) or 0 for diagnostics in checked) if checked else None for name in names
        }
        return {"capture_census_status": statuses, **totals}

    def test_final(self):
        """Check the physical worker states, including tetrahedra and boundaries, and the capture census."""
        if self.halted is not None:
            raise AssertionError(self.halted)
        if not self.completed:
            raise AssertionError("No physical steps completed")
        rest = self.geometry["tissue_rest_positions"]
        tets = self.geometry["tissue_tetrahedra"]
        fixed = self.geometry["fixed_particle_indices"]

        def determinants(points):
            vertices = points[tets]
            return np.linalg.det(vertices[:, 1:] - vertices[:, :1])

        original = determinants(rest)
        # One explicit observation: the complete state (a step without a render scene leaves the
        # tissue on the device), with the numbers of late kernel module loads up to date.
        observations = self.pool.observe()
        for item in observations:
            for name in ("tissue_positions", "rigid_body_poses", "thread_positions"):
                if not np.isfinite(item[name]).all():
                    raise AssertionError(f"Nonfinite {name}")
            if np.any(determinants(item["tissue_positions"]) / original <= 0):
                raise AssertionError("Tissue tetrahedron inverted")
            np.testing.assert_allclose(item["tissue_positions"][fixed], rest[fixed], rtol=0, atol=1e-8)
            if not item["diagnostics"]["independent_free_needle"] or item["diagnostics"]["needle_pose_drive"]:
                raise AssertionError("The needle must be an independently simulated rigid body")
        default_procedure = self.args.speed == ARC_SPEED and self.args.settle_seconds == SETTLE_SECONDS
        lift_height = getattr(self.args, "lift_height", 0.0)
        # What the line of this test gains from a stitch that was checked (_check_stitched).
        stitch = {}
        # Without a lift the procedure is the arc, complete after 6 s. With a lift it has no fixed
        # duration and ends with the stage "done": the self-test asks for that at its last frame,
        # however long the run is, so a run that ends while the procedure is under way fails.
        minimum_time = 6.0 if not lift_height else 0.0
        if self.controller is not None and default_procedure and self.sim_time >= minimum_time - 1e-8:
            for item in self.controller.report():
                if not item["commanded_arc_complete"] or min(item["entry_count"], item["exit_count"]) < 1:
                    raise AssertionError(f"Insertion did not complete an entry and exit: {item}")
                expected = 1 + int(self._tip_lift_height(self.args) > 0)
                if lift_height and item["lifts_complete"] < expected:
                    raise AssertionError(f"The lifts after the insertion and at the tip did not complete: {item}")
                # With the pull the procedure ends when the pull has ended by its rule: the needle out of
                # the tissue, or without thread tracks its tail at its entry.
                if self._pull_through(self.args) and not item["pull_complete"]:
                    raise AssertionError(f"The pull of the needle by its tip did not complete: {item}")
                if lift_height and item["stage"] != "done":
                    raise AssertionError(f"The procedure had not ended at the last frame: {item}")
            if self._pull_through(self.args) and self.geometry.get("thread_tracks"):
                self._check_pulled_through(observations)
                stitch = self._check_stitched(observations)
        for item in observations:
            # The reserve of eyelets is fixed at start-up: an eyelet that found no room in it is a failure.
            refused = item.get("thread_diagnostics", {}).get("eyelets_refused", 0)
            if refused:
                raise AssertionError(f"{refused} eyelets of the thread found no room in the reserve")
        census = self._check_census(observations)
        print(
            json.dumps(
                {
                    "physical_checks_passed": True,
                    "num_envs": self.args.num_envs,
                    "physical_time_s": self.sim_time,
                    **census,
                    **stitch,
                }
            )
        )

    def _check_pulled_through(self, observations):
        """Fail unless the needle is out of the tissue and the thread runs through the track it cut.

        For a procedure that is done, in a scene whose thread follows the needle's track. Per
        scene: the tail of the needle is past the exit marker and the tissue holds no part of
        the track; every planned passage has eyelets, every eyelet that was made is threaded,
        and thread is inside the tissue; the draw of the thread ended at its length, and while
        it ran the loose thread pulled the needle with less than the weight that the scene
        applies to the needle (``needle_weight_N`` of the geometry): it ran through the track
        and was not stopped at a mouth of it.
        """
        passages = int(self.geometry["insertion_plan"]["expected_passages"])
        weight = self.geometry["needle_weight_N"]
        for item, observation in zip(self.controller.report(), observations, strict=True):
            engaged = observation["diagnostics"]["engaged_track_length_m"]
            if not item["tail_past_exit_m"] > 0.0 or engaged != 0.0:
                raise AssertionError(
                    f"The needle is not out of the tissue: its tail is {item['tail_past_exit_m']} m past the exit "
                    f"marker and {engaged} m of its track are held"
                )
            thread = observation["thread_diagnostics"]
            made, threaded = thread["eyelets"], thread["eyelets_threaded"]
            if len(thread["eyelet_passages"]) < passages or not made or threaded != made:
                raise AssertionError(
                    f"The thread does not run through the needle's track: {threaded} of {made} eyelets threaded, "
                    f"passages {thread['eyelet_passages']} of {passages} planned"
                )
            if not thread["thread_inside_tissue_m"] > 0.0:
                raise AssertionError("No thread is inside the tissue after the pull")
            drawn, force = item["draw_travel_m"], item["draw_thread_force_N"]
            if not item["draw_complete"] or not 0.0 < item["draw_length_m"] == drawn:
                raise AssertionError(f"The thread was not drawn through the tissue by the length of the draw: {item}")
            if force is None or not force < weight:
                raise AssertionError(
                    f"The loose thread did not run through its track in the draw: it pulled the needle with "
                    f"{force} N, not with less than the needle's weight of {weight} N: {item}"
                )

    # Time that the stitch must have held the wound without the tool's pull when the self-test ends [s].
    stitch_rest_seconds = 1.0

    def _check_stitched(self, observations):
        """Fail unless the stitch was made and holds by itself; returns the numbers of the stitch.

        For a procedure that is done, in a scene whose thread follows the needle's track. Per scene:

        - two ties were asked for and two hold, in the scene's thread report and in the controller's;
        - the tightening ended by its rule (the controller's report has what the rule predicted;
          a tightening that ends at its travel bound stops the procedure), inside the travel
          bound, and the least tissue volume at the end of its hold (:meth:`_follow_stitch`) was
          within the fall that the rule predicted for its way to rest (slope times way) of the
          rule's value;
        - the slackening ended by its rule, the thread's force on the needle below the weight that
          the scene applies to the needle (``needle_weight_N`` of the geometry), within the
          travel of the tightening, and that force is below the weight now;
        - both jaws were on the needle in every call from the first tie on;
        - the slackening ended at least :attr:`stitch_rest_seconds` ago, and the wound is narrower
          at its top and at its track than in the reference shape and than it was when the
          tightening began: the stitch keeps a closure without the tool's pull.

        Returns:
            Per scene, in lists, for the line of :meth:`test_final`: the tool's travel of the
            tightening and back; the least tissue volume now and at the end of the tightening's
            hold; the thread's force on the needle and its largest force on an eyelet; the weight
            of the needle; the wound's widths; per gap the closure that the stitch keeps (the
            width when the tightening began less the width now) and its share of the closure at
            the end of the tightening's hold, at the top and at the track; the time since the
            slackening ended; and the largest travel and turn of the needle in the jaws since the
            tie at the exit.
        """
        numbers = {
            name: []
            for name in (
                "tighten_travel_mm",
                "slacken_travel_mm",
                "least_volume_ratio",
                "least_volume_ratio_tight",
                "thread_needle_force_mN",
                "thread_eyelet_force_mN",
                "needle_weight_mN",
                "wound_top_mm",
                "wound_track_mm",
                "wound_top_kept_mm",
                "wound_top_kept_share",
                "wound_track_kept_mm",
                "wound_track_kept_share",
                "stitch_rest_s",
                "lock_slip_mm",
                "lock_slip_mrad",
            )
        }
        weight = self.geometry["needle_weight_N"]
        records = self._stitch_record or [None] * len(observations)
        for item, observation, record in zip(self.controller.report(), observations, records, strict=True):
            thread, tension = observation["thread_diagnostics"], observation["thread_tension"]
            if not item["stitch_complete"] or (thread["ties"], thread["ties_asked"], tension["ties"]) != (2, 2, 2):
                raise AssertionError(
                    f"The thread is not tied at its entry and at its exit by two ties: {thread['ties']} ties of "
                    f"{thread['ties_asked']} asked for; the stitch of the procedure: {item}"
                )
            if record is None or "slack" not in record or "tight" not in record:
                raise AssertionError(f"The example did not see the tightening begin and its hold end: {record}")
            slope, way = item["tighten_slope_per_m"], item["tighten_way_to_rest_m"]
            travel, bound = item["tighten_travel_m"], item["tighten_travel_bound_m"]
            if item["tighten_predicted_volume"] is None or slope is None or not 0.0 < travel <= bound:
                raise AssertionError(f"The tightening did not end by its rule inside its travel bound: {item}")
            tight, rule = record["tight"]["least_volume_ratio"], item["tighten_stop_volume"]
            if tight is None or not abs(tight - rule) <= -slope * way:
                raise AssertionError(
                    f"The least tissue volume {tight} at the end of the tightening's hold is not within "
                    f"{-slope * way} of the rule's {rule}, the fall that the rule predicted for its way to rest: {item}"
                )
            back, pull = item["slacken_travel_m"], tension["needle_force_N"]
            if item["slacken_stop_force_N"] is None or not 0.0 <= back <= travel:
                raise AssertionError(
                    f"The slackening did not end by its rule within the travel of the tightening: the thread's "
                    f"force on the needle did not fall below the needle's weight of {weight} N: {item}"
                )
            if not pull < weight:
                raise AssertionError(
                    f"The thread pulls the needle with {pull} N, not with less than the needle's weight of "
                    f"{weight} N: it is not slack: {item}"
                )
            if item["stitch_calls_without_grasp"]:
                raise AssertionError(f"A jaw was off the needle while the stitch was made: {item}")
            rested = self.sim_time - item["slacken_end_time_s"]
            if not rested >= self.stitch_rest_seconds - 1e-9:
                raise AssertionError(
                    f"The run ended {rested} s after the slackening: the stitch must have held for "
                    f"{self.stitch_rest_seconds} s without the tool's pull"
                )
            wound = observation["wound_widths"]
            slack, held = record["slack"]["widths"], record["tight"]["widths"]
            if not wound or len(wound) != len(slack):
                raise AssertionError(f"The wound's widths are not those of the gaps that were tightened: {wound}")
            kept = [
                (before[0] - gap["top_m"], before[1] - gap["track_m"]) for gap, before in zip(wound, slack, strict=True)
            ]
            closed = [(before[0] - then[0], before[1] - then[1]) for before, then in zip(slack, held, strict=True)]
            narrower = all(
                gap["top_m"] < gap["rest_top_m"] and gap["track_m"] < gap["rest_track_m"] and min(left) > 0.0
                for gap, left in zip(wound, kept, strict=True)
            )
            if not narrower:
                raise AssertionError(
                    "The stitch does not hold the wound narrower without the tool's pull, at its top and at its "
                    f"track, than in the reference shape and than when the tightening began {slack}: {wound}"
                )
            for name, value in zip(
                numbers,
                (
                    1.0e3 * travel,
                    1.0e3 * back,
                    observation["diagnostics"]["least_volume_ratio"],
                    tight,
                    1.0e3 * pull,
                    1.0e3 * tension["eyelet_force_N"],
                    1.0e3 * weight,
                    [1.0e3 * gap["top_m"] for gap in wound],
                    [1.0e3 * gap["track_m"] for gap in wound],
                    [1.0e3 * left[0] for left in kept],
                    [left[0] / whole[0] if whole[0] > 0.0 else None for left, whole in zip(kept, closed, strict=True)],
                    [1.0e3 * left[1] for left in kept],
                    [left[1] / whole[1] if whole[1] > 0.0 else None for left, whole in zip(kept, closed, strict=True)],
                    rested,
                    1.0e3 * item["lock_slip_m"],
                    1.0e3 * item["lock_slip_rad"],
                ),
                strict=True,
            ):
                numbers[name].append(value)
        return numbers

    def close(self):
        self._cleanup()

    @staticmethod
    def create_parser():
        # The parser's formatter prints the default of every option after its help text.
        parser = newton.examples.create_parser()
        parser.description = __doc__
        parser.set_defaults(num_frames=2880)
        parser.add_argument("--num-envs", type=int, default=1, help="Independent scenes; start with one, then scale.")
        parser.add_argument("--substeps", type=int, default=4, help="Coupled physics intervals per 60 Hz frame.")
        parser.add_argument(
            "--rigid-backend",
            choices=("native", "mujoco-warp"),
            default="mujoco-warp",
            help="FP32 MuJoCo Warp GPU backend (tolerance 1e-6, needs a CUDA device) or native CPU contacts.",
        )
        parser.add_argument("--controller", choices=("procedural", "hold"), default="procedural")
        parser.add_argument("--speed", type=float, default=ARC_SPEED, help="Needle arc speed [m/s].")
        parser.add_argument(
            "--settle-seconds", type=float, default=SETTLE_SECONDS, help="Closed-jaw settling time [s]."
        )
        parser.add_argument(
            "--lift-height",
            type=float,
            default=0.01,
            help="Inspection lift along the tissue entry normal [m], after the insertion arc; 0 disables it.",
        )
        parser.add_argument(
            "--tip-lift-height",
            type=float,
            default=None,
            help="Lift by the needle tip after the tool has let go and taken the tip [m], with --no-pull-through "
            "only (default there: 0.005; 0 ends after the first lift). The pull replaces this lift.",
        )
        parser.add_argument(
            "--pull-through",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="After the lift the tool lets go, takes the needle by its tip and pulls it on along its own "
            "curve until it is out of the tissue; the thread follows it through the track, is then tied at its "
            "entry and tightened until the tissue allows no more, tied at its exit as well, and let go slack. "
            "--no-pull-through takes the tip with the earlier grasp and lifts by it instead (--tip-lift-height). "
            "With --lift-height 0 the example is the insertion alone.",
        )
        parser.add_argument(
            "--thread-tracks",
            action=argparse.BooleanOptionalAction,
            default=THREAD_TRACKS,
            help="The thread follows the needle through the tissue (default): the track the needle cuts becomes "
            "eyelets of the thread, which are threaded where the needle's tail has passed. With "
            "--no-thread-tracks the thread stays on the tissue surface and the pull stops at the entry.",
        )
        parser.add_argument(
            "--bite-length",
            type=float,
            default=0.010,
            help="Distance from entry to exit [m]; 0 keeps the recorded stitch, 5.4 mm apart and 0.39 mm deep.",
        )
        parser.add_argument(
            "--coupling-relative-tolerance",
            type=float,
            default=COUPLING_RELATIVE_TOLERANCE,
            help="Relative interface force tolerance of the needle/tissue coupling. The default is a measured "
            "choice (docs/usage.md); the insertion example keeps 1e-3.",
        )
        parser.add_argument(
            "--trial-budget",
            type=trial_budget_argument,
            default=TRIAL_BUDGET,
            help="Trials of one coupling loop after which its best evaluated trial is kept instead of iterating "
            f"on to the limit of {COUPLING_TRIAL_LIMIT} trials: 2 to {COUPLING_TRIAL_LIMIT - 1}, or 0 or none for "
            "no budget. The default is a measured choice (docs/usage.md). A kept step is not converged to the "
            "tolerance.",
        )
        parser.add_argument(
            "--tool-tissue-contact",
            default=TOOL_TISSUE_CONTACT,
            choices=("off", "swept"),
            metavar="LAW",
            help="Contact between the tools and the tissue, without friction: 'swept' or 'off' (the scene and "
            "SutureSceneConfig keep the contact off). swept: the tissue surface nodes that a tool capsule reaches "
            "in the next physics interval are held on it, so the tool pushes the tissue, and tissue that cannot "
            "follow stops the tool. off: the tools pass through the tissue (docs/usage.md).",
        )
        parser.add_argument(
            "--no-tool-tissue-contact",
            dest="tool_tissue_contact",
            action="store_false",
            help="No contact between the tools and the tissue (the same as --tool-tissue-contact off).",
        )
        parser.add_argument(
            "--response-basis-nodes",
            type=int,
            default=RESPONSE_BASIS_NODES,
            help="Support nodes whose response columns the response-basis guess solves in one coupled step when "
            "the track has grown; the first sample of an entry adds four nodes (twelve columns). 0: every missing "
            "column in the step that follows the growth.",
        )
        parser.add_argument(
            "--frame-budget",
            type=float,
            default=FRAME_BUDGET_MS,
            metavar="MS",
            help="Frame time budget [ms], with one scene in the window of the OpenGL viewer: the longest time from "
            "the start of one step to the start of the next that the example plans for; the default is the time "
            "of a picture at 15 frames/s. 0 runs whole frames. A step that would not fit returns with its frame "
            "unfinished and the next step goes on with it. Not a guarantee; not used with --test, another viewer "
            "or several scenes (docs/usage.md).",
        )
        parser.add_argument("--spacing", type=float, default=0.45, help="Viewer spacing [m]; does not change physics.")
        parser.add_argument("--overview", action="store_true", help="Frame the complete shafts and trocar ports.")
        return parser


def main():
    parser = Example.create_parser()
    options = parser.parse_args()
    if options.num_envs < 1 or options.substeps < 1:
        parser.error("Scene count and substeps must be positive")
    if not np.isfinite([options.speed, options.settle_seconds, options.spacing]).all():
        parser.error("Speed, settling time and spacing must be finite")
    if options.speed <= 0 or options.settle_seconds < 0 or options.spacing < 0.4:
        parser.error("Speed must be positive, settling time nonnegative, and spacing at least 0.4 m")
    if not np.isfinite(options.lift_height) or not 0 <= options.lift_height <= 0.01:
        parser.error("Lift height must be finite and between 0 and 0.01 metres")
    tip_lift_height = _tip_lift_height(options)
    if not np.isfinite(tip_lift_height) or not 0 <= tip_lift_height <= options.lift_height:
        parser.error("The lift at the tip must be between 0 and the lift after the insertion")
    if tip_lift_height and _pull_through(options):
        parser.error("The pull replaces the lift at the tip: use --no-pull-through with --tip-lift-height")
    if not np.isfinite(options.bite_length) or options.bite_length < 0:
        parser.error("The bite length must be finite and nonnegative")
    if not np.isfinite(options.coupling_relative_tolerance) or options.coupling_relative_tolerance <= 0:
        parser.error("The relative coupling tolerance must be finite and positive")
    if not np.isfinite(options.frame_budget) or options.frame_budget < 0:
        parser.error("The frame budget must be finite and nonnegative")
    viewer, args = newton.examples.init(parser)
    try:
        newton.examples.run(Example(viewer, args), args)
    except BaseException:
        viewer.close()
        raise
    finally:
        for example in tuple(Example._instances):
            example.close()


if __name__ == "__main__":
    main()
