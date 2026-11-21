# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cut the Baymax body, 135 mm across and on its back: slice, drill, carve; right-drag to lift cut material.

The body is the bundled tetrahedral Baymax mesh (assets/baymax), scaled and
placed at load (_baymax_body.py): 135 mm from hand to hand, lying on its back
on the floor, belly up, arm span along x. All of it is active; nothing is
held and nothing is cut by itself. It rests on the floor under gravity, and
the floor holds it and its cut pieces by Coulomb friction.

Line is the default cutter: left-drag and release to slice through the depth
of the view. Cylinder drills on left-click; Sphere carves while the left
button is held. Right-drag material and release to leave a spring hold.
Middle-click its green marker, or use Clear holds, to release it. Reset
restores the uncut body.

The cutting pipeline partitions the existing tetrahedra into material
fragments, duplicates disconnected basis nodes and creates new contact faces.
These are user-directed geometric edits, not a force-triggered fracture law.

Run: python -m newton_solver_suturebot.examples fem_surgical_incision
"""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import newton
import newton.examples
import numpy as np
import warp as wp

from newton_solver_suturebot import CutConnectivityDebug, CutTissue, TetrahedralTunnelGuide
from newton_solver_suturebot.examples._startup import Script, Startup, census_failure, first_draw, owns_startup
from newton_solver_suturebot.examples.incision import _baymax_body as body
from newton_solver_suturebot.examples.incision._baymax_body import load_body
from newton_solver_suturebot.examples.phantom._cut_debug_view import CutDebugView
from newton_solver_suturebot.examples.phantom._cut_diagnostics import rejection_message
from newton_solver_suturebot.examples.phantom._cylinder_extent import CylinderExtent
from newton_solver_suturebot.examples.phantom._hold_debug import HoldDebug
from newton_solver_suturebot.examples.phantom._screen_cut import ScreenCutTool
from newton_solver_suturebot.examples.phantom._screen_cylinder_cut import ScreenCylinderCutTool
from newton_solver_suturebot.examples.phantom._screen_pick import ScreenPickTool
from newton_solver_suturebot.examples.phantom._screen_sphere_cut import ScreenSphereCutTool
from newton_solver_suturebot.examples.phantom._surface_pick import SurfacePicker

# The scripted user of the self-test works within this fraction of the bounding box's size on either
# side of its centre: on the Baymax body that rectangle is belly (tests/test_baymax_asset.py).
SCRIPT_REACH = 0.1

# The radius of a cylinder or sphere tool [m] that the reserve of ``--cut_reserve`` is derived for.
# The tool radius is still the pad's design input (1 mm); the reserve is sized on this body in a later step.
RESERVE_TOOL_RADIUS = 0.001


def derive_reserve(positions, elements, scale, polygons=None):
    """Fixed storage for ``scale`` design loads of cutting with a tool of ``RESERVE_TOOL_RADIUS``.

    The derivation is that of the cut simulation (``CutTissue.derive_reserve``); ``polygons`` are the
    polygon vertices and faces per fragment of ``--cut_reserve_polygons``, or None for its default.
    """
    return CutTissue.derive_reserve(positions, elements, scale, polygons, tool_radius=RESERVE_TOOL_RADIUS)


class Example(CutTissue):
    """The cut simulation (``CutTissue``) with the Baymax body, the screen tools, the panel and the self-test."""

    # Start-up check of this example and the scripted user of its self-test (examples/_startup.py).
    # Both stay None in a scene built on this class: such a scene has more to prepare than this
    # class knows of, and opens, marks and checks its own start-up.
    startup = None
    script = None
    # The scripted user does nothing in a run that somebody else drives (a recorder's scenario).
    _script_idle = False

    def __init__(self, viewer, args=None):
        args = newton.examples.default_args(self.create_parser()) if args is None else args
        if min(args.substeps, args.solver_iterations, args.node_copies) < 1:
            raise ValueError("Substeps, solver iterations and node copies must be positive")
        if not np.isfinite([args.young_modulus, args.density, args.drop_height, args.gravity]).all():
            raise ValueError("Material and drop parameters must be finite")
        if args.young_modulus <= 0 or args.density <= 0 or args.drop_height < 0 or args.gravity < 0:
            raise ValueError("Material values must be positive and drop height/gravity nonnegative")
        frame_dt = 1.0 / 60.0
        # The grip numbers that were not given follow from the body, the material and the time step
        # (_baymax_body.py has the derivations).
        if args.grip_stiffness is None:
            args.grip_stiffness = body.grip_stiffness(args.young_modulus, args.poisson_ratio, args.grip_radius)
        if args.grip_damping is None:
            args.grip_damping = body.GRIP_RELAXATION_TIME * args.grip_stiffness
        if args.grip_speed is None:
            args.grip_speed = body.grip_speed(args.substeps / frame_dt)
        grips = (args.grip_stiffness, args.grip_radius, args.grip_speed)
        damping = (args.grip_damping, args.material_damping, args.linear_damping)
        if not np.isfinite(grips).all() or min(grips) <= 0:
            raise ValueError("Grip stiffness, radius and speed must be finite and positive")
        if not 0 < args.poisson_ratio < 0.5 or not np.isfinite(damping).all() or min(damping) < 0:
            raise ValueError("Poisson ratio must lie in (0, 0.5); damping must be finite and nonnegative")
        if not np.isfinite(args.cut_reserve) or args.cut_reserve < 0:
            raise ValueError("The cut reserve must be finite and nonnegative")
        if args.cut_reserve_polygons is not None and (args.cut_reserve == 0 or min(args.cut_reserve_polygons) < 4):
            raise ValueError("--cut_reserve_polygons needs --cut_reserve and at least four vertices and faces")
        self.viewer, self.args = viewer, args
        device = wp.get_device()
        if owns_startup(self, Example):
            # Everything from here to the end of the constructor is start-up.
            self.startup = Startup(device)
            self.startup.begin()
        positions, elements = load_body(Path(args.asset).expanduser().resolve())
        lower, upper = positions.min(axis=0), positions.max(axis=0)
        # The floor is where the body's lowest point is as it is loaded: the body lies on it.
        floor_height = float(lower[2])
        size = ", ".join(f"{1000.0 * value:.1f}" for value in upper - lower)
        print(
            f"[surgical incision] {len(positions):,} vertices, "
            f"{len(elements):,} tetrahedra, {size} mm; all material active; "
            f"floor at {1000.0 * floor_height:.1f} mm",
            flush=True,
        )
        # Fixed storage of --cut_reserve (derive_reserve), or None: storage grows inside an edit.
        reserve = None
        if args.cut_reserve > 0:
            reserve = derive_reserve(positions, elements, args.cut_reserve, args.cut_reserve_polygons)
            print(f"[surgical incision] Fixed cut reserve: {json.dumps(reserve)}", flush=True)
        # The simulation: the pool of node copies and everything a start-up builds on it (reset()).
        # Its one set of grips is that of the mouse (holds).
        super().__init__(
            positions,
            elements,
            frame_dt=frame_dt,
            substeps=args.substeps,
            solver_iterations=args.solver_iterations,
            density=args.density,
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
            material_damping=args.material_damping,
            linear_damping=args.linear_damping,
            min_stretch=body.MIN_STRETCH,
            max_stretch=body.MAX_STRETCH,
            gravity=args.gravity,
            floor_height=floor_height,
            ground_stiffness=args.ground_stiffness,
            piece_stiffness=args.piece_contact_stiffness,
            search_radius=body.CONTACT_SEARCH_RADIUS,
            ground_friction=args.ground_friction,
            grips=(
                {
                    "stiffness": args.grip_stiffness,
                    "damping": args.grip_damping,
                    "maximum_speed": args.grip_speed,
                    "radius": args.grip_radius,
                },
            ),
            node_copies=args.node_copies,
            drop_height=args.drop_height,
            streaming_topology=args.streaming_topology,
            fragment_capacity=args.fragment_capacity,
            surface_capacity=args.surface_capacity,
            reserve=reserve,
            rejection_message=rejection_message,
            device=device,
        )
        # The viewer and the screen tools are bound to the pool of node copies with the floor.
        self._display_model = self.pool_model
        self._display_state = self._display_model.state()
        self.viewer.set_model(self._display_model)
        if hasattr(viewer, "show_particles"):
            viewer.show_particles = False
        self.reset_camera()
        self.cutter_mode = args.cutter_mode
        self.screen_cut = ScreenCutTool(
            viewer,
            self._display_model,
            can_start=lambda: self.cutter_mode == "line" and not self.screen_pick.active,
        )
        self.surface_picker = SurfacePicker(self.device)
        self.cylinder_extent = CylinderExtent(self.device)
        self.tunnel_guide = TetrahedralTunnelGuide(self.device)
        self.screen_pick = ScreenPickTool(
            viewer,
            self._display_model,
            begin=self._begin_pick,
            move=self._move_pick,
            finish=self._finish_pick,
            cancel=self._cancel_pick,
            remove=self._remove_pick,
            can_start=lambda: (
                self.screen_cut.start is None and not self.screen_sphere.active and not self.screen_cylinder.active
            ),
        )
        self.screen_sphere = ScreenSphereCutTool(
            viewer,
            self._display_model,
            begin=self._pick_cut_surface,
            finish=self._finish_sphere,
            can_start=lambda: not self.screen_pick.active and self.screen_cut.start is None,
            speed=args.sphere_speed,
            enabled=self.cutter_mode == "sphere",
        )
        self.screen_cylinder = ScreenCylinderCutTool(
            viewer,
            self._display_model,
            begin=self._pick_cut_surface,
            can_start=lambda: not self.screen_pick.active and self.screen_cut.start is None,
            enabled=self.cutter_mode == "cylinder",
            diameter_pixels=args.cylinder_diameter_pixels,
        )
        self.connection_debug = CutDebugView(viewer, self.device, self._debug_topology, kind="linear")
        self._hold_debug = None
        self._hold_debug_revision = None
        self._hold_revision = 0
        self._built = False
        self.reset()
        self._built = True
        if self.startup is not None:
            if getattr(args, "test", False):
                self.script = self._test_script()
                self.script.reserve(viewer)
            self._finish_startup(frame=True)

    def _finish_startup(self, *, frame):
        """End a start-up: the kernel modules of the run loaded, the scene drawn once, then the mark.

        The constructor draws a whole frame where the viewer has no frame limit. A reset can
        come from a panel button, inside the viewer's frame: it issues the log calls alone.
        """
        self._load_run_modules()
        first_draw(self.viewer, self.render_pad, self.sim_time, frame=frame)
        self.startup.finish()

    def _load_run_modules(self):
        """Load the kernel modules whose first launch would otherwise fall into a frame or a user action.

        The module of the cut simulation (the transfer of cloned nodes at the first accepted edit) and
        the two modules of the debug view, which launch when its box is first ticked. The cutter, the
        picker, the cylinder extent and the tunnel guide load theirs when they are made.
        """
        self.load_kernel_modules()
        for owner in (CutConnectivityDebug, HoldDebug):
            wp.load_module(owner.__module__, device=self.device, block_dim=256 if self.device.is_cuda else 1)

    def reset_camera(self):
        """The start view: from the feet (-y) and above, the arm span across the picture (_baymax_body.py)."""
        lower, upper = self.reference_positions.min(axis=0), self.reference_positions.max(axis=0)
        center, span = (lower + upper) / 2, float((upper - lower).max())
        elevation = math.radians(body.VIEW_ELEVATION_DEGREES)
        offset = body.VIEW_DISTANCE * span * np.array((0.0, -math.cos(elevation), math.sin(elevation)))
        direction = -offset / np.linalg.norm(offset)
        right = np.cross(direction, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        target = center - body.VIEW_PANEL_SHIFT * span * right
        # Where the start view looks from and along: a screen tool takes its size from this depth.
        self._start_view = target + offset, direction
        self.viewer.camera_speed = 0.5 * span
        self.viewer.set_camera(
            pos=tuple(target + offset),
            pitch=float(np.degrees(np.arcsin(direction[2]))),
            yaw=float(np.degrees(np.arctan2(direction[1], direction[0]))),
        )
        camera = getattr(self.viewer, "camera", None)
        if camera is not None:
            camera.near = max(0.0001, 0.001 * span)
            camera.far = max(2.0, 10.0 * span)
            if hasattr(camera, "set_pivot"):
                camera.set_pivot(target)
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None:
            renderer.spotlight_enabled = False

    def _debug_topology(self):
        return self.cutter.geometry.arrays, self.cutter.material_components, self.model.tet_indices

    @property
    def holds(self):
        """The grips of the mouse: the one set of grips of this scene, made anew by every start-up."""
        return self.grip_sets[0]

    def reset(self):
        # A rebuilt scene starts up again. The constructor has its start-up open already and ends it itself.
        rebuild = self.startup is not None and self._built
        if rebuild:
            self.startup.begin()
        self.screen_cut.cancel()
        self.screen_sphere.cancel()
        self.screen_cylinder.cancel()
        self._sphere_clock = perf_counter()
        self._sphere_segment_start = None
        self._sphere_gesture = -1
        self.screen_pick.reset()
        self.connection_debug.invalidate()
        if self._hold_debug is not None:
            self._hold_debug.hide(self.viewer)
        self._hold_debug, self._hold_debug_revision = None, None
        # The simulation anew: the cutter, the models, the surface, the contact, the grips, the guard,
        # the solver and every graph (CutTissue.reset).
        super().reset()
        # Keep the display model bound: rebinding would clear cutting overlays
        # and UI callbacks. The viewer resizes the drawing storage as needed.
        if rebuild:
            self._finish_startup(frame=False)
        print("[surgical incision] Ready: left-drag to slice; right-drag to lift.", flush=True)

    def cut_cylinder_ray(self, origin, direction, radius):
        """Drill through the current material along a screen ray in one transaction."""
        self.contact.update_surface(self.state_0.particle_q)
        try:
            extent = self.cylinder_extent.query(self.contact.points, self._surface_count, origin, direction, radius)
        except ValueError as error:
            self.cut_error = f"Cut rejected: {error}"
            return False
        if extent is None:
            self.cut_error = ""
            return False
        return self.cut_cylinder(*extent, radius)

    def _consume_cylinder(self):
        request = self.screen_cylinder.consume()
        if request is not None:
            self.cut_cylinder_ray(*request)

    def _pick_cut_surface(self, origin, direction):
        self.contact.update_surface(self.state_0.particle_q)
        hit = self.surface_picker.pick(
            origin, direction, [(self.contact.points, self._render_indices, self._surface_count)]
        )
        return None if hit is None else hit[3]

    def _advance_sphere(self):
        now = perf_counter()
        # Avoid a long catch-up sweep after a paused debugger or blocked window.
        dt = min(now - self._sphere_clock, 0.1)
        self._sphere_clock = now
        if not self.screen_sphere.active:
            self._sphere_segment_start = None
            return
        if self._sphere_gesture != self.screen_sphere.gesture_id:
            self._sphere_gesture = self.screen_sphere.gesture_id
            self._sphere_segment_start = self.screen_sphere.center.copy()
            self._next_sphere_boundary()
            dt = 0.0
        segment = self.screen_sphere.advance(dt)
        if segment is not None:
            _, end, radius = segment
            boundary = self.tunnel_guide.distance(
                self.state_0.particle_q, self._sphere_segment_start, self.screen_sphere.direction
            )
            if boundary is None:
                boundary = self._next_sphere_boundary()
            # Existing tetrahedral faces set the update frequency, independent
            # of rendering FPS. Geometry still clips the current swept volume;
            # the guide neither deletes whole cells nor redirects the tunnel.
            if boundary is None or np.linalg.norm(end - self._sphere_segment_start) < boundary:
                return
            self.cut_sphere(self._sphere_segment_start, end, radius)
            self._sphere_segment_start = end.copy()
            if self.cut_error:
                self.screen_sphere.cancel()
            else:
                self._next_sphere_boundary()

    def _next_sphere_boundary(self):
        return self.tunnel_guide.next_boundary(
            self.cutter.geometry,
            self.state_0.particle_q,
            self._sphere_segment_start,
            self.screen_sphere.direction,
        )

    def _finish_sphere(self):
        """Publish the last partial cell at the displayed tip, without advancing."""
        start, end = self._sphere_segment_start, self.screen_sphere.center
        if start is not None and end is not None and np.linalg.norm(end - start) > self.cutter.tolerance:
            self.cut_sphere(start, end, self.screen_sphere.radius)
        self._sphere_segment_start = None

    def set_cutter_mode(self, mode):
        if mode not in ("line", "cylinder", "sphere"):
            raise ValueError("Cutter mode must be line, cylinder or sphere")
        self.screen_cut.cancel()
        self.screen_sphere.cancel()
        self.screen_cylinder.cancel()
        self.cutter_mode = mode
        self.screen_sphere.enabled = mode == "sphere"
        self.screen_cylinder.enabled = mode == "cylinder"
        self._sphere_clock = perf_counter()
        self._sphere_segment_start = None

    def _begin_pick(self, origin, direction):
        self.contact.update_surface(self.state_0.particle_q)
        hit = self.surface_picker.pick(
            origin, direction, [(self.contact.points, self._render_indices, self._surface_count)]
        )
        if hit is None:
            return None
        _, triangle, barycentric, point = hit
        if self.holds.begin(self.surface, self.cutter.geometry, triangle, barycentric, point, self.state_0.particle_q):
            self._hold_revision += 1
            return point
        return None

    def _move_pick(self, point):
        self.holds.move(point)

    def _finish_pick(self):
        self.holds.finish()

    def _remove_pick(self, index):
        removed = self.holds.remove(index)
        if removed:
            self._hold_revision += 1
        return removed

    def _cancel_pick(self):
        if hasattr(self, "holds"):
            self.holds.cancel()

    def clear_holds(self):
        self.screen_pick.reset()
        self.holds.clear()
        self._hold_revision += 1

    def step(self):
        # The user acts inside the frame: the frame is counted, the edits are applied, then its physics runs.
        self.begin_frame()
        if self.script is not None:
            self.script.play()
        self._consume_cylinder()
        self._advance_sphere()
        triangles = self.screen_cut.consume()
        if len(triangles):
            self.cut(triangles)
        self.advance()

    def render(self):
        if self.viewer.is_paused():
            self._consume_cylinder()
            self._advance_sphere()
            if self.screen_cut.pending:
                self.cut(self.screen_cut.consume())
        self.viewer.begin_frame(self.sim_time)
        self.render_pad()
        self.viewer.end_frame()

    def render_pad(self):
        """Draw the body and its debug view inside a viewer frame."""
        self.contact.update_surface(self.state_0.particle_q)
        self.viewer.log_state(self._display_state)
        self.viewer.log_mesh(
            "incision_body",
            self.contact.points,
            self._render_indices,
            color=(0.86, 0.48, 0.40),
            hidden=self.connection_debug.hide_skin,
        )
        self.connection_debug.render(self.state_0.particle_q, self.cut_revision)
        if self.connection_debug.enabled:
            if self._hold_debug is None:
                self._hold_debug = HoldDebug(self.holds, self.device)
            revision = (self.cut_revision, self._hold_revision, self.holds.count)
            if revision != self._hold_debug_revision:
                self._hold_debug.rebuild(self.model.tet_indices, self.cutter.material_components, self.fragment_count)
                self._hold_debug_revision = revision
            self._hold_debug.render(self.viewer, self.state_0.particle_q)
        elif self._hold_debug is not None:
            self._hold_debug.hide(self.viewer)

    def gui(self, ui):
        ui.text("Baymax body: slicing")
        ui.text(f"Whole body: {len(self.reference_elements):,} original tetrahedra")
        ui.text_wrapped("Floor support; nothing is held and nothing is cut by itself.")
        self.connection_debug.gui(ui)
        if self.connection_debug.enabled and self._hold_debug is not None:
            ui.text_wrapped("Green lines: individual grip anchors to their world targets.")
            if self._hold_debug.spanning_grips:
                grips = ", ".join(str(index + 1) for index in self._hold_debug.spanning_grips)
                ui.text_wrapped(
                    f"Holds spanning multiple pieces: {grips}. Release those green markers to free both sides."
                )
        modes = ("line", "cylinder", "sphere")
        changed, mode = ui.combo("Cutter", modes.index(self.cutter_mode), ["Line", "Cylinder", "Sphere"])
        if changed:
            self.set_cutter_mode(modes[mode])
        if self.cutter_mode == "cylinder":
            self.screen_cylinder.gui(ui)
        elif self.cutter_mode == "sphere":
            self.screen_sphere.gui(ui)
        else:
            self.screen_cut.gui(ui)
        self.screen_pick.gui(ui)
        if ui.button("Clear holds"):
            self.clear_holds()
        ui.text(f"Holds: {self.holds.count}/{self.holds.capacity}")
        if ui.button("Reset body and cuts"):
            self.reset()
        ui.text(f"Accepted cuts: {self.cut_revision}")
        ui.text(f"Physics: {self.args.substeps} substeps, {self.args.solver_iterations} CG iterations each")
        if self.cut_revision:
            ui.text(f"Last cut: {self.last_cut_ms:.1f} ms")
        if self.cut_error:
            ui.text_wrapped(self.cut_error)

    # ------------------------------------------------------------------ self-test

    def _scripted(self, action):
        """A scripted action that does nothing in a run that somebody else drives."""

        def run():
            if not self._script_idle:
                action()

        return run

    def _scripted_begin(self):
        """Self-test: the first action. A scene that somebody else has cut or gripped is theirs."""
        if self.cut_revision or self.holds.count or self.screen_cut.pending or self.cut_error:
            # The scenario of a recorder: the scripted user stands down for the whole run.
            self._script_idle = True

    def _script_lengths(self):
        """Lengths of the scripted user, all from the bounding box of the body [m].

        ``margin``: how far a scripted blade reaches beyond the body, a tenth of its smallest
        size. ``above``: the height at which a scripted ray starts, half the body's thickness
        above its top. ``lift``: how far a scripted drag lifts, a tenth of the thickness.
        """
        low, high = self.reference_positions.min(axis=0), self.reference_positions.max(axis=0)
        size = high - low
        top = float(high[2]) + self.args.drop_height
        return {"margin": 0.1 * float(size.min()), "above": top + 0.5 * float(size[2]), "lift": 0.1 * float(size[2])}

    def _view_tool_radius(self, point, diameter_pixels):
        """Radius [m] that a screen tool takes when ``point`` is clicked in the start view.

        The tool is ``diameter_pixels`` across at the depth of the clicked point, in the
        viewer's picture (``VIEW_HEIGHT_PIXELS`` rows over ``VIEW_FIELD_DEGREES``).
        """
        position, direction = self._start_view
        depth = float(np.dot(np.asarray(point, dtype=np.float64) - position, direction))
        pixel = 2.0 * depth * math.tan(math.radians(0.5 * body.VIEW_FIELD_DEGREES)) / body.VIEW_HEIGHT_PIXELS
        return 0.5 * diameter_pixels * pixel

    def _scripted_blade(self, start, end):
        """World triangles of a released line from ``start`` to ``end`` [m, floor plane].

        The blade stands from a margin below the floor to a margin above the top of the body as it
        is loaded: through the whole depth of a body that lies on the floor. Material that has sunk
        deeper than the margin into the floor is not reached.
        """
        lengths = self._script_lengths()
        low = self.floor_height - lengths["margin"]
        high = float(self.reference_positions[:, 2].max()) + self.args.drop_height + lengths["margin"]
        (ax, ay), (bx, by) = start, end
        return np.asarray(((ax, ay, low), (bx, by, low), (bx, by, high), (ax, ay, high)))

    def _scripted_line(self, label, start, end, corner=None):
        """Self-test: a released line, as the screen tool queues it; the next frame cuts.

        With ``corner`` the two triangles of the stroke lie in two planes that meet in a vertical
        edge there: they stay two knife patches.
        """
        if corner is None:
            triangles = self._scripted_blade(start, end)[[[0, 1, 2], [0, 2, 3]]]
        else:
            first, second = self._scripted_blade(start, corner), self._scripted_blade(corner, end)
            triangles = np.asarray((first[[0, 1, 2]], second[[0, 1, 2]]))
        self.screen_cut._pending.append(triangles)
        self._scripted_edits.append([label, self.cut_revision, None])

    def _scripted_cylinder(self, point):
        """Self-test: a left click of the cylinder tool straight above ``point`` [m]; the next frame drills.

        The hole has the radius that a click on the body's top takes in the start view.
        """
        top = float(self.reference_positions[:, 2].max()) + self.args.drop_height
        radius = self._view_tool_radius((point[0], point[1], top), self.screen_cylinder.diameter_pixels)
        origin = np.asarray((point[0], point[1], self._script_lengths()["above"]))
        self.screen_cylinder._pending.append((origin, np.asarray((0.0, 0.0, -1.0)), radius))
        self._scripted_edits.append(["cylinder", self.cut_revision, None])

    def _scripted_sphere(self, point):
        """Self-test: press and hold the sphere tool straight above ``point`` [m].

        The sphere has the radius that a press on the picked point takes in the start view.
        """
        tool = self.screen_sphere
        direction = np.asarray((0.0, 0.0, -1.0))
        origin = np.asarray((point[0], point[1], self._script_lengths()["above"]))
        hit = self._pick_cut_surface(origin, direction)
        self._presses.append(hit is not None)
        if hit is None:
            return
        tool.cancel()
        tool._pressed = True
        tool.radius = self._view_tool_radius(hit, tool.diameter_pixels)
        tool.center = np.asarray(hit, dtype=np.float64) - tool.radius * direction
        tool.direction = direction
        tool.gesture_id += 1
        self._scripted_edits.append(["sphere", self.cut_revision, None])

    def _scripted_sphere_hold(self, progress):
        """Self-test: the button stays down for a frame of a tenth of a second.

        The sphere advances with the time between frames, at most a tenth of a second for one
        frame: the scripted hold takes that step in every frame, however long the frame took.
        """
        if not self._script_idle:
            self._sphere_clock = perf_counter() - 1.0

    def _scripted_sphere_release(self):
        tool = self.screen_sphere
        tool._pressed = False
        if tool.active:
            self._finish_sphere()
        tool.cancel()

    def _scripted_settle(self):
        """Self-test: note whether the edit that the script queued last was accepted."""
        if self._scripted_edits and self._scripted_edits[-1][2] is None:
            edit = self._scripted_edits[-1]
            edit[2] = self.cut_revision > edit[1] and not self.cut_error
            edit.append(self.cut_error)

    def _scripted_press(self, point):
        """Self-test: a right button press straight above ``point`` [m] on the body."""
        origin = np.asarray((point[0], point[1], self._script_lengths()["above"]))
        hit = self._begin_pick(origin, np.asarray((0.0, 0.0, -1.0)))
        self._pressed = None if hit is None else np.asarray(hit, dtype=np.float64)
        self._presses.append(hit is not None)

    def _scripted_drag(self, travel, progress):
        if self._pressed is not None and not self._script_idle:
            self._move_pick(self._pressed + progress * np.asarray(travel))

    def _scripted_release(self):
        if self._pressed is not None:
            self._finish_pick()
            self._pressed = None

    def _test_script(self):
        """The scripted user of ``--test``, in 210 frames.

        A line cut across the body, a right-drag that leaves a grip, a line cut of two knife
        patches, a cylinder click, the debug view on and off, the grip removed, Reset, a line cut
        and a grip on the new body. With ``--cut_reserve`` also a held sphere: without the reserve
        a held sphere makes the polygon storage of every fragment grow inside a frame. The first
        action waits until frame 40 and stands down for the whole run if somebody else has cut or
        gripped by then (a recorder's scenario).

        Where the user acts follows from the bounding box of the body: the two lines cross it
        whole through its centre, one along y and after Reset one along x; everything else lies
        within ``SCRIPT_REACH`` of the box's size around its centre, ``(rx, ry)`` below.
        """
        self._presses, self._pressed, self._scripted_edits = [], None, []
        low, high = self.reference_positions.min(axis=0), self.reference_positions.max(axis=0)
        (cx, cy), (sx, sy) = 0.5 * (low[:2] + high[:2]), high[:2] - low[:2]
        rx, ry = SCRIPT_REACH * sx, SCRIPT_REACH * sy
        lengths = self._script_lengths()
        margin, lift = lengths["margin"], (0.0, 0.0, lengths["lift"])
        act = self._scripted
        script = Script()
        script.at(40, "first look", self._scripted_begin)
        script.at(
            40,
            "line across the body",
            act(lambda: self._scripted_line("line", (cx, low[1] - margin), (cx, high[1] + margin))),
        )
        script.at(43, "line settled", act(self._scripted_settle))
        script.at(46, "right press", act(lambda: self._scripted_press((cx + 0.3 * rx, cy))))
        script.drag(47, 66, "right drag", lambda p: self._scripted_drag(lift, p))
        script.at(66, "right release", act(self._scripted_release))
        script.at(
            72,
            "line of two patches",
            act(
                lambda: self._scripted_line(
                    "line of two patches",
                    (cx - rx, cy - ry),
                    (cx - 0.2 * rx, cy - 0.6 * ry),
                    corner=(cx - 0.6 * rx, cy - ry),
                )
            ),
        )
        script.at(75, "line settled", act(self._scripted_settle))
        script.at(80, "Cutter: Cylinder", act(lambda: self.set_cutter_mode("cylinder")))
        script.at(82, "cylinder click", act(lambda: self._scripted_cylinder((cx + 0.5 * rx, cy + 0.5 * ry))))
        script.at(85, "cylinder settled", act(self._scripted_settle))
        if self.reserve is not None:
            script.at(90, "Cutter: Sphere", act(lambda: self.set_cutter_mode("sphere")))
            script.at(92, "sphere press", act(lambda: self._scripted_sphere((cx + 0.8 * rx, cy - 0.8 * ry))))
            script.drag(93, 121, "sphere held", self._scripted_sphere_hold)
            script.at(122, "sphere release", act(self._scripted_sphere_release))
            script.at(123, "sphere settled", act(self._scripted_settle))
        script.at(128, "debug view on", act(lambda: setattr(self.connection_debug.connections, "enabled", True)))
        script.at(134, "debug view off", act(lambda: setattr(self.connection_debug.connections, "enabled", False)))
        script.at(138, "Cutter: Line", act(lambda: self.set_cutter_mode("line")))
        script.at(140, "middle click on the grip marker", act(lambda: self._remove_pick(0)))
        script.at(150, "Reset body and cuts", act(self.reset))
        script.at(
            156,
            "line across the body",
            act(lambda: self._scripted_line("line", (low[0] - margin, cy), (high[0] + margin, cy))),
        )
        script.at(159, "line settled", act(self._scripted_settle))
        script.at(162, "right press", act(lambda: self._scripted_press((cx, cy + 0.3 * ry))))
        script.drag(163, 178, "right drag", lambda p: self._scripted_drag(lift, p))
        script.at(178, "right release", act(self._scripted_release))
        script.rest(31)
        return script

    def startup_report(self):
        """The check of a self-test: nothing captured or loaded after a start-up that storage growth did not force.

        The start-ups are this example's (the construction and every Reset); the growths of storage
        and the captures inside them are the simulation's (``CutTissue.check_startups``). Without
        ``--cut_reserve`` an edit may grow storage and captures in that frame; with it nothing
        grows and every capture after a start-up mark fails.

        Returns:
            The numbers of the check, for the line that ``test_final`` prints.
        """
        return self.check_startups(self.startup, census_failure)

    def _check_script(self):
        """Fail if a scripted press missed the body or a scripted edit was not accepted."""
        if self._script_idle or not self.script.complete:
            return
        if not all(self._presses):
            raise AssertionError(f"A scripted press missed the body: {self._presses}")
        refused = [edit for edit in self._scripted_edits if not edit[2]]
        if refused:
            raise AssertionError(f"Scripted edits were not accepted: {refused}")

    def test_final(self):
        q, qd = self.state_0.particle_q.numpy(), self.state_0.particle_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise AssertionError("FEM body contains non-finite state")
        indices = self.model.tet_indices.numpy()[: self.fragment_count]
        corners = q[indices]
        gradients = (corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1) @ self.model.tet_poses.numpy()[
            : self.fragment_count
        ]
        if self.fragment_count and np.min(np.linalg.det(gradients)) <= 0.0:
            raise AssertionError("FEM body contains inverted material")
        if self.script is None:
            return
        # The self-test: the scripted edits were accepted, and nothing was captured or loaded after a
        # start-up (the construction and each Reset) except inside a growth of storage.
        numbers = self.startup_report()
        self._check_script()
        print(
            json.dumps(
                {
                    "cuts": self.cut_revision,
                    "scripted_edits": [edit[0] for edit in self._scripted_edits],
                    "presses": len(self._presses),
                    "script_idle": self._script_idle,
                    **self.script.summary(),
                    **numbers,
                }
            )
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1200)
        parser.add_argument("--asset", default=str(body.ASSET), help="Tetrahedral body; the bundled Baymax mesh.")
        parser.add_argument("--poisson_ratio", type=float, default=0.35)
        parser.add_argument(
            "--grip_stiffness",
            type=float,
            default=None,
            help="Grip stiffness [N/m]. Default: ten times the stiffness of the held patch of tissue, "
            "20 E a / (1 - nu^2) with a the grip radius (2.06 kN/m).",
        )
        parser.add_argument(
            "--grip_damping",
            type=float,
            default=None,
            help="Grip damping [kg/s]. Default: 1.5 ms times the grip stiffness (3.1 kg/s).",
        )
        parser.add_argument(
            "--grip_radius",
            type=float,
            default=body.GRIP_RADIUS,
            help="Grip patch radius [m]. Default: 1.6 surface edges of the body (3.0 mm).",
        )
        parser.add_argument(
            "--grip_speed",
            type=float,
            default=None,
            help="Grip target speed limit [m/s]. Default: an eighth of what the body's smallest element has "
            "above the stretch limit, per substep (3.0 mm/s).",
        )
        parser.add_argument("--cutter_mode", choices=("line", "cylinder", "sphere"), default="line")
        parser.add_argument(
            "--cylinder_diameter_pixels", type=float, default=8.0, help="Cylinder entry diameter [framebuffer pixels]."
        )
        parser.add_argument("--sphere_speed", type=float, default=1.0, help="Held sphere advance [diameters/s].")
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument("--solver_iterations", type=int, default=80)
        parser.add_argument(
            "--streaming_topology",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Update cut matrix entries in persistent GPU storage; disable to compare with sparse rebuilding.",
        )
        parser.add_argument("--young_modulus", type=float, default=30000.0, help="Elastic modulus [Pa].")
        parser.add_argument("--density", type=float, default=1000.0, help="Material density [kg/m^3].")
        parser.add_argument("--material_damping", type=float, default=0.01, help="Viscous material damping [Pa s].")
        parser.add_argument("--linear_damping", type=float, default=1.0, help="Mass-proportional damping [1/s].")
        parser.add_argument("--drop_height", type=float, default=0.0, help="Initial clearance above the floor [m].")
        parser.add_argument("--gravity", type=float, default=9.81, help="Gravity magnitude [m/s^2].")
        parser.add_argument("--ground_stiffness", type=float, default=2.0e7, help="Surface floor penalty [N/m^3].")
        parser.add_argument(
            "--ground_friction",
            type=float,
            default=body.FLOOR_FRICTION,
            help="Coulomb friction coefficient between the body and the floor; zero is a floor without friction.",
        )
        parser.add_argument(
            "--piece_contact_stiffness", type=float, default=2.0e7, help="Cut-piece contact penalty [N/m^3]."
        )
        parser.add_argument("--node_copies", type=int, default=4)
        parser.add_argument(
            "--fragment_capacity", type=int, default=0, help="Reserved material fragments; zero selects automatically."
        )
        parser.add_argument(
            "--surface_capacity",
            type=int,
            default=0,
            help="Initial boundary triangle reserve; grows automatically. Zero selects the initial size.",
        )
        parser.add_argument(
            "--cut_reserve",
            type=float,
            default=0.0,
            help="Fixed storage for this many design loads of cutting (a line cut across the body in each direction, "
            "a cylinder and a sphere tunnel through its depth), allocated at start and at Reset: no edit grows "
            "storage or captures a graph, and an edit beyond the reserve is refused. Zero: storage grows inside an edit.",
        )
        parser.add_argument(
            "--cut_reserve_polygons",
            type=int,
            nargs=2,
            default=None,
            metavar=("VERTICES", "FACES"),
            help="Polygon vertices and faces reserved per fragment with --cut_reserve; default 48 and 32, the first "
            "step of the cutter's growth. One pass of the sphere can need 64 and 34; growth ends at 64 and 48.",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
