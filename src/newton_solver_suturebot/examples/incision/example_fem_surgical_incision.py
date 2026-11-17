# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cut the Baymax body, 135 mm across and on its back: slice, drill, carve; right-drag to lift cut material.

The body is the bundled tetrahedral Baymax mesh (assets/baymax), scaled and
placed at load (_baymax_body.py): 135 mm from hand to hand, lying on its back
on the floor, belly up, arm span along x. All of it is active; nothing is
held and nothing is cut by itself. It rests on the floor under gravity.

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
import gc
import json
import math
from pathlib import Path
from time import perf_counter

import newton
import newton.examples
import numpy as np
import warp as wp

from newton_solver_suturebot import (
    CutConnectivityDebug,
    CutModel,
    CutSurfaceContact,
    CutVolumeGuard,
    LinearTetCut,
    MaterialHolds,
    SolverFEM,
    TetrahedralTunnelGuide,
)
from newton_solver_suturebot._src.utils import capture
from newton_solver_suturebot.examples._startup import (
    Script,
    Startup,
    census_failure,
    first_draw,
    launch_startup_graph,
    load_kernel_modules,
    owns_startup,
)
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


@wp.kernel
def _clone_nodes(
    sources: wp.array2d[int],
    count: wp.array[int],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    copied_q: wp.array[wp.vec3],
    copied_qd: wp.array[wp.vec3],
):
    i = wp.tid()
    if i < count[0]:
        target, source = sources[i, 0], sources[i, 1]
        copied_q[target] = q[source]
        copied_qd[target] = qd[source]


@wp.kernel(enable_backward=False)
def _node_components(
    elements: wp.array2d[int],
    fractions: wp.array[float],
    components: wp.array[int],
    node_components: wp.array[int],
):
    element = wp.tid()
    if fractions[element] > 0.0:
        for corner in range(4):
            wp.atomic_max(node_components, elements[element, corner], components[element])


# The scripted user of the self-test works within this fraction of the bounding box's size on either
# side of its centre: on the Baymax body that rectangle is belly (tests/test_baymax_asset.py).
SCRIPT_REACH = 0.1

# What the reserve of ``--cut_reserve`` is derived from: the radius of a cylinder or sphere tool [m], the
# planes of a cylinder (16 sides, two ends) and of a swept sphere (20 faces of the icosahedron and at most
# 10 sides of its sweep), the triangles of a section through a tetrahedron (a quadrilateral on each lip)
# and of the wall that one tool plane leaves in a tetrahedron (a triangular tool face cut by four faces
# has at most seven corners).
# The tool radius is still the pad's design input (1 mm); the reserve is sized on this body in a later step.
RESERVE_TOOL_RADIUS = 0.001
CYLINDER_PLANES, SPHERE_PLANES = 18, 30
SECTION_TRIANGLES, WALL_TRIANGLES = 4, 5
# Polygon vertices and faces that the reserve holds for every fragment: the sizes that the cutter's own
# growth reaches first (from 12 and 12). They are not a bound. A tetrahedron that every plane of a swept
# sphere crosses can leave pieces of 4 + 2 * 30 = 64 vertices and 4 + 30 = 34 faces; the cutter's limits
# are 64 and 48. The polygon storage is most of the reserve, so a larger one is asked for by name
# (``--cut_reserve_polygons``).
RESERVE_POLYGONS = (48, 32)


def derive_reserve(positions, elements, scale, polygons=None):
    """Fixed storage for ``scale`` design loads of cutting, derived from the mesh and the tools.

    One design load is a line cut through the whole depth across the body in each horizontal
    direction, one cylinder and one sphere tunnel through the whole depth. The counts are taken on
    this mesh: the tetrahedra that the two sections through its centre cross, and the tetrahedra
    whose bounding box a vertical tool of ``RESERVE_TOOL_RADIUS`` can touch at the fullest place
    of the body (the cutter selects the fragments of an edit by their bounding boxes).

    - fragments: one more for every tetrahedron a section crosses (its second half), and one less
      than the planes of the tool for every tetrahedron of a tool column (a convex tool leaves at
      most one piece per plane).
    - subtractor work: the tetrahedra of a tool column.
    - boundary triangles, beyond those of the uncut body: ``SECTION_TRIANGLES`` per crossed
      tetrahedron, ``WALL_TRIANGLES`` per tool plane and tetrahedron of a tool column.
    - polygon vertices and faces per fragment: ``RESERVE_POLYGONS``, or ``polygons``. This part is
      not derived from the mesh; an edit that needs more is refused.
    - subtractor scratch: the cutter's limits.

    Args:
        positions: Reference positions of the mesh [m].
        elements: Its tetrahedra.
        scale: Design loads; a fraction reserves less.
        polygons: Polygon vertices and faces per fragment, or None for ``RESERVE_POLYGONS``.

    Returns:
        The reserve as a dict of capacities, with the counts they were derived from.
    """
    corners = positions[elements]
    lower, upper = corners.min(axis=1), corners.max(axis=1)
    low, high = positions.min(axis=0), positions.max(axis=0)
    center = 0.5 * (low + high)
    crossed = sum(
        int(np.count_nonzero((lower[:, axis] < center[axis]) & (upper[:, axis] > center[axis]))) for axis in (0, 1)
    )
    # Tool centres on a grid of one radius: a tool whose centre lies in a cell touches only boxes
    # within a radius and half a cell of the cell's centre. Counted with a two-dimensional prefix sum.
    spacing, margin = RESERVE_TOOL_RADIUS, 1.5 * RESERVE_TOOL_RADIUS
    shape = np.ceil((high[:2] - low[:2]) / spacing).astype(int) + 2
    first = np.clip(np.ceil((lower[:, :2] - margin - low[:2]) / spacing).astype(int), 0, shape - 1)
    last = np.clip(np.floor((upper[:, :2] + margin - low[:2]) / spacing).astype(int), 0, shape - 1)
    grid = np.zeros(shape + 1, dtype=np.int64)
    np.add.at(grid, (first[:, 0], first[:, 1]), 1)
    np.add.at(grid, (last[:, 0] + 1, first[:, 1]), -1)
    np.add.at(grid, (first[:, 0], last[:, 1] + 1), -1)
    np.add.at(grid, (last[:, 0] + 1, last[:, 1] + 1), 1)
    column = int(grid.cumsum(axis=0).cumsum(axis=1).max())
    tool_planes = CYLINDER_PLANES + SPHERE_PLANES
    if polygons is None:
        polygons = RESERVE_POLYGONS
    return {
        "scale": float(scale),
        "crossed_tetrahedra": crossed,
        "column_tetrahedra": column,
        "fragment_capacity": len(elements) + math.ceil(scale * (crossed + (tool_planes - 2) * column)),
        "work_capacity": max(256, math.ceil(scale * column)),
        "surface_triangles": math.ceil(scale * (SECTION_TRIANGLES * crossed + WALL_TRIANGLES * tool_planes * column)),
        "vertex_capacity": int(polygons[0]),
        "face_capacity": int(polygons[1]),
        "scratch_vertex_capacity": LinearTetCut.SCRATCH_LIMIT,
        "scratch_face_capacity": LinearTetCut.SCRATCH_LIMIT,
    }


class Example:
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
        self.frame_dt = 1.0 / 60.0
        # The grip numbers that were not given follow from the body, the material and the time step
        # (_baymax_body.py has the derivations).
        if args.grip_stiffness is None:
            args.grip_stiffness = body.grip_stiffness(args.young_modulus, args.poisson_ratio, args.grip_radius)
        if args.grip_damping is None:
            args.grip_damping = body.GRIP_RELAXATION_TIME * args.grip_stiffness
        if args.grip_speed is None:
            args.grip_speed = body.grip_speed(args.substeps / self.frame_dt)
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
        self.device = wp.get_device()
        if owns_startup(self, Example):
            # Everything from here to the end of the constructor is start-up.
            self.startup = Startup(self.device)
            self.startup.begin()
        self.sim_time = 0.0
        self.sim_dt = self.frame_dt / args.substeps
        self.reference_positions, self.reference_elements = load_body(Path(args.asset).expanduser().resolve())
        lower, upper = self.reference_positions.min(axis=0), self.reference_positions.max(axis=0)
        # The floor is where the body's lowest point is as it is loaded: the body lies on it.
        self.floor_height = float(lower[2])
        size = ", ".join(f"{1000.0 * value:.1f}" for value in upper - lower)
        print(
            f"[surgical incision] {len(self.reference_positions):,} vertices, "
            f"{len(self.reference_elements):,} tetrahedra, {size} mm; all material active; "
            f"floor at {1000.0 * self.floor_height:.1f} mm",
            flush=True,
        )
        self._pool_size = len(self.reference_positions) * args.node_copies
        builder = newton.ModelBuilder()
        pool = np.tile(self.reference_positions, (args.node_copies, 1))
        builder.add_particles(
            pool.tolist(), np.zeros_like(pool).tolist(), np.ones(len(pool)).tolist(), radius=[0.0] * len(pool)
        )
        builder.add_ground_plane(height=self.floor_height)
        self._display_model = builder.finalize(device=self.device)
        self._display_model.set_gravity((0.0, 0.0, -args.gravity))
        self._display_state = self._display_model.state()
        self._model_factory = CutModel(
            self._display_model,
            self.reference_positions,
            self.reference_elements,
            density=args.density,
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
            damping=args.material_damping,
        )
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
        # Frames stepped so far, less one; and every growth of storage inside an edit, with its frame.
        self._frame = -1
        self.storage_growths = []
        # Fixed storage of --cut_reserve (derive_reserve), or None: storage grows inside an edit.
        self.reserve = None
        self.cutter = None
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

        The module of this file (the transfer of cloned nodes at the first accepted edit) and the
        two modules of the debug view, which launch when its box is first ticked. The cutter, the
        picker, the cylinder extent and the tunnel guide load theirs when they are made.
        """
        load_kernel_modules(self.device, _clone_nodes)
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

    def _make_solver(self, model):
        solver = SolverFEM(
            model,
            tet_volume_weights=model.cut_volume_fractions,
            dt=self.sim_dt,
            preconditioner_type="block_diag",
            solver_type="cg",
            linear_solver_maxiter=self.args.solver_iterations,
            ground_plane=None,
            linear_damping=self.args.linear_damping,
            min_stretch=body.MIN_STRETCH,
            max_stretch=body.MAX_STRETCH,
        )
        if self.args.streaming_topology:
            solver.prepare_tetrahedral_updates(len(self.reference_positions))
        return solver

    def _install_physics(self, model, state, surface):
        self.model, self.state_0, self.state_1 = model, state, model.state()
        self.state_1.assign(state)
        self.control = model.control()
        self.solver = self._make_solver(model)
        # The grips, the contact and the volume guard act in the solver's step and read its
        # time step from the same device arrays as the solver.
        self.holds.step_time = self.solver.step_time
        self._backup_velocity_increment = wp.empty_like(self.solver.velocity_increment)
        self._mechanical_components = wp.clone(self.cutter.material_components)
        self._node_components = wp.empty(model.particle_count, dtype=int, device=self.device)
        self._update_mechanical_components(self.cutter.material_components)
        self.solver.prepare_component_scaling(self._node_components, model.tet_count)
        self.volume_guard = CutVolumeGuard(
            model, components=self._mechanical_components, step_time=self.solver.step_time
        )
        self.contact = CutSurfaceContact(
            surface,
            state.particle_q,
            model.particle_inv_mass,
            ground_height=self.floor_height,
            ground_stiffness=self.args.ground_stiffness,
            piece_stiffness=self.args.piece_contact_stiffness,
            search_radius=body.CONTACT_SEARCH_RADIUS,
            step_time=self.solver.step_time,
        )
        self.surface = surface
        self._surface_count = int(surface.count.numpy()[0])
        # GL uploads indices once. Reserved triangles stay in the draw list;
        # inactive point triples collapse to zero until a cut activates them.
        self._render_indices = wp.array(np.arange(surface.indices.size, dtype=np.int32), device=self.device)
        self.holds.remap(self.cutter.geometry)
        self.solver.refresh_tangent(self.state_0.particle_q, self.sim_dt)
        self.graph = self._capture_physics_graph()
        self._launch_frame_graph()

    def _update_mechanical_components(self, components):
        # A growing surface warms the new model before the geometry transaction
        # commits. Its component labels must already match those new elements.
        wp.copy(self._mechanical_components, components)
        self._node_components.fill_(-1)
        wp.launch(
            _node_components,
            dim=self.model.tet_count,
            inputs=[
                self.model.tet_indices,
                self.model.cut_volume_fractions,
                self._mechanical_components,
                self._node_components,
            ],
            device=self.device,
        )

    def _frame_snapshot(self):
        """Keep what a frame changes; the returned call puts it back.

        The two states, the solver's velocity increment, the driven grip targets and the count
        of the volume guard: a frame that runs outside the run, eager or as a graph, leaves the
        scene as it found it.
        """
        states = (self.state_0, self.state_1)
        saved_states = tuple(self.model.state() for _ in states)
        for saved, state in zip(saved_states, states, strict=True):
            saved.assign(state)
        persistent = (self.solver.velocity_increment, self.holds.driven, self.volume_guard.limited_steps)
        saved_persistent = tuple(wp.clone(array) for array in persistent)

        def restore():
            for state, saved in zip(states, saved_states, strict=True):
                state.assign(saved)
            for array, saved in zip(persistent, saved_persistent, strict=True):
                wp.copy(array, saved)

        return restore

    def _capture_physics_graph(self):
        """Warm and capture the current buffers without advancing physical state."""
        if not self.device.is_cuda:
            return None
        restore = self._frame_snapshot()
        try:
            self.simulate()
            restore()
            with capture.ScopedCapture(device=self.device) as scope:
                self.simulate()
            return scope.graph
        finally:
            restore()

    def _launch_frame_graph(self):
        """Launch the captured frame once and put the scene back.

        Warp builds the executable of a graph at its first launch: that launch belongs to the
        start-up, or to the growth of the surface that captured the graph, not to the next frame.
        """
        if self.graph is None:
            return
        restore = self._frame_snapshot()
        try:
            launch_startup_graph(self.graph, self.device)
        finally:
            restore()

    def _list_growths(self, growths=()):
        """Add growths of storage to the list of this example, each with the frame it fell into.

        The growths that the cutter has listed since the last call come first: they precede a
        growth of the surface in the same edit.
        """
        cutter = self.cutter.storage_growths
        for growth in (*cutter[self._cutter_growths_listed :], *growths):
            self.storage_growths.append({"frame": self._frame, **growth})
        self._cutter_growths_listed = len(cutter)

    def _grow_surface(self, triangle_count):
        """Replace contact and drawing storage only when the boundary outgrows it."""
        self._list_growths()
        start = perf_counter()
        captured = capture.census.captures, capture.census.solver_captures
        growth = {"what": "boundary surface triangles", "from": self.contact.capacity, "completed": False}
        try:
            self._replace_surface(triangle_count)
            growth["completed"] = True
        finally:
            if self.device.is_cuda:
                wp.synchronize_device(self.device)
            growth["to"] = self.contact.capacity
            growth["seconds"] = perf_counter() - start
            growth["captures"] = capture.census.captures - captured[0]
            growth["solver_captures"] = capture.census.solver_captures - captured[1]
            self._list_growths([growth])

    def _replace_surface(self, triangle_count):
        capacity = max(1, self.contact.capacity)
        while capacity < triangle_count:
            capacity *= 2
        capacity = min(capacity, self.cutter.surface.components.size)
        surface = CutSurfaceContact.reserve_surface(self.cutter.surface, triangle_count, capacity)
        previous = self.contact
        contact = CutSurfaceContact(
            surface,
            self.state_0.particle_q,
            self.model.particle_inv_mass,
            ground_height=previous.ground_height,
            ground_stiffness=previous.ground_stiffness,
            piece_stiffness=previous.piece_stiffness,
            damping_ratio=previous.damping_ratio,
            search_radius=previous.search_radius,
            step_time=previous.step_time,
        )
        indices = wp.array(np.arange(surface.indices.size, dtype=np.int32), device=self.device)
        self.contact, self.surface, self._render_indices = contact, surface, indices
        # The FEM model, states, solver and sparse storage retain their addresses.
        # Only the graph's references to surface/contact buffers need replacing.
        self.graph = self._capture_physics_graph()
        self._launch_frame_graph()

    def simulate(self):
        self.solver.refresh_tangent(self.state_0.particle_q, self.sim_dt)
        self.holds.add_matrix(self.solver, self.sim_dt)
        for substep in range(self.args.substeps):
            current, following = (self.state_0, self.state_1) if substep % 2 == 0 else (self.state_1, self.state_0)
            forces = self.contact.evaluate(current.particle_q, current.particle_qd, self.sim_dt, piece_contact=True)
            self.holds.add_forces(current.particle_q, current.particle_qd, forces, self.sim_dt)
            self.solver.step(current, following, self.control, None, self.sim_dt, external_forces=forces)
            self.volume_guard.apply(
                following.particle_q, following.particle_qd, self.sim_dt, previous_q=current.particle_q
            )
        if self.args.substeps % 2:
            wp.copy(self.state_0.particle_q, self.state_1.particle_q)
            wp.copy(self.state_0.particle_qd, self.state_1.particle_qd)

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
        self.holds = MaterialHolds(
            self.device,
            stiffness=self.args.grip_stiffness,
            damping=self.args.grip_damping,
            maximum_speed=self.args.grip_speed,
            radius=self.args.grip_radius,
        )
        self.cut_revision = 0
        source_count = len(self.reference_elements)
        fragment_capacity = self.args.fragment_capacity or source_count + 16384
        # The subtractor's scratch starts at the cutter's limit. From its default (48 vertices and the
        # polygon faces of the geometry, 12) the first cylinder grew it, 12 faces to 24, and captured
        # the subtractor's graph again in that frame. It is storage of the work list (256 entries),
        # not of the mesh: 0.58 GiB more than the default, on any body.
        polygons = (12, 12)
        subtractor = {
            "scratch_vertex_capacity": LinearTetCut.SCRATCH_LIMIT,
            "scratch_face_capacity": LinearTetCut.SCRATCH_LIMIT,
        }
        if self.args.cut_reserve > 0:
            if self.reserve is None:
                self.reserve = derive_reserve(
                    self.reference_positions,
                    self.reference_elements,
                    self.args.cut_reserve,
                    self.args.cut_reserve_polygons,
                )
                print(f"[surgical incision] Fixed cut reserve: {json.dumps(self.reserve)}", flush=True)
            reserve = self.reserve
            fragment_capacity = self.args.fragment_capacity or reserve["fragment_capacity"]
            polygons = reserve["vertex_capacity"], reserve["face_capacity"]
            subtractor = {
                "work_capacity": reserve["work_capacity"],
                "scratch_vertex_capacity": reserve["scratch_vertex_capacity"],
                "scratch_face_capacity": reserve["scratch_face_capacity"],
            }
            if self.cutter is not None:
                # The reserve is as large as the storage it replaces: give that back first.
                self.cutter = None
                gc.collect()
        self.cutter = LinearTetCut(
            self.reference_positions,
            self.reference_elements,
            fragment_capacity=fragment_capacity,
            node_copies=self.args.node_copies,
            vertex_capacity=polygons[0],
            face_capacity=polygons[1],
            fixed_storage=self.reserve is not None,
            device=self.device,
        )
        self._cutter_growths_listed = 0
        # Every graph of the cutter and its subtractor: captured and launched here, not in the first
        # line cut of a patch count or in the first cylinder or sphere.
        self.cutter.prepare(patch_counts=tuple(range(1, self.cutter.max_patches + 1)))
        self.cutter.prepare_sphere(**subtractor)
        pool = np.tile(self.reference_positions, (self.args.node_copies, 1))
        pool[:, 2] += self.args.drop_height
        q = wp.array(pool, dtype=wp.vec3, device=self.device)
        qd = wp.zeros_like(q)
        self._cut_positions = wp.empty_like(q)
        self.fragment_count = len(self.reference_elements)
        model, state = self._build_model(self.cutter.geometry, self.cutter.geometry.nodeindices, q, qd)
        self._staged_model, self._staged_state = self._build_model(
            self.cutter.geometry, self.cutter.geometry.nodeindices, q, qd
        )
        self._backup_model, self._backup_state = self._build_model(
            self.cutter.geometry, self.cutter.geometry.nodeindices, q, qd
        )
        surface_count = int(self.cutter.surface.count.numpy()[0])
        surface_capacity = max(2 * surface_count, surface_count + 256)
        if self.reserve is not None:
            surface_capacity = min(
                surface_count + self.reserve["surface_triangles"], self.cutter.surface.components.size
            )
        surface = CutSurfaceContact.reserve_surface(
            self.cutter.surface, surface_count, self.args.surface_capacity or surface_capacity
        )
        self._install_physics(model, state, surface)
        # Keep the display model bound: rebinding would clear cutting overlays
        # and UI callbacks. The viewer resizes the drawing storage as needed.
        self.sim_time, self.cut_revision = 0.0, 0
        self.last_cut_ms, self.last_geometry_ms, self.last_rebuild_ms = 0.0, 0.0, 0.0
        self.cut_error = ""
        if rebuild:
            self._finish_startup(frame=False)
        print("[surgical incision] Ready: left-drag to slice; right-drag to lift.", flush=True)

    def _build_model(self, geometry, nodes, q, qd):
        return self._model_factory.build(
            geometry.parent,
            nodes,
            self.cutter.volume_fractions,
            self.cutter.mass_integrals,
            self.fragment_count,
            q,
            qd,
            capacity=self.cutter.capacity,
        )

    def cut(self, triangles):
        """Apply a finite screen-projected incision; leave a rejected edit unpublished."""
        return self._apply_edit(lambda positions: self.cutter.propose_triangles(positions, triangles))

    def cut_sphere(self, start, end, radius):
        """Remove the spherical tool's swept material; distances are in metres."""
        return self._apply_edit(lambda positions: self.cutter.propose_sphere(positions, start, end, radius))

    def cut_cylinder(self, start, end, radius):
        """Remove one straight, constant-radius cylinder; distances are in metres."""
        return self._apply_edit(lambda positions: self.cutter.propose_cylinder(positions, start, end, radius))

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

    def _apply_edit(self, propose):
        """Apply one edit; list the storage that grew inside it."""
        try:
            return self._publish_edit(propose)
        finally:
            self._list_growths()

    def _publish_edit(self, propose):
        start = perf_counter()
        published = False
        holds_staged = False
        previous_count, previous_surface_count = self.fragment_count, self._surface_count
        previous_surface_buffers = self.contact, self.surface, self._render_indices, self.graph
        try:
            wp.copy(self._cut_positions, self.state_0.particle_q)
            diagnostics = propose(self._cut_positions)
            if not diagnostics["changed_parents"]:
                self.cutter.discard()
                self.cut_error = ""
                return False
            surface_count = diagnostics["triangle_count"]
            self.last_geometry_ms = 1000.0 * (perf_counter() - start)
            self._staged_state.assign(self.state_0)
            wp.launch(
                _clone_nodes,
                dim=self._pool_size,
                inputs=[
                    self.cutter.candidate.clone_sources,
                    self.cutter.candidate.clone_count,
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self._staged_state.particle_q,
                    self._staged_state.particle_qd,
                ],
                device=self.device,
            )
            self._model_factory.update(
                self._staged_model,
                self.cutter.pending_geometry.parent,
                self.cutter.candidate.nodes,
                self.cutter.volume_fractions,
                self.cutter.mass_integrals,
                diagnostics["fragment_count"],
            )
            holds_staged = True
            self.holds.stage_remap(self.cutter.pending_geometry, self.cutter.candidate.nodes, self.cutter.geometry)
            self._model_factory.copy_into(self._backup_model, self.model)
            self._backup_state.assign(self.state_0)
            wp.copy(self._backup_velocity_increment, self.solver.velocity_increment)
            published = True
            self._model_factory.copy_into(self.model, self._staged_model)
            self.state_0.assign(self._staged_state)
            self.state_1.assign(self._staged_state)
            self._update_mechanical_components(self.cutter.candidate.material_components)
            self.solver.refresh_component_scaling()
            self.solver.refresh_topology(
                self.state_0.particle_q,
                self.sim_dt,
                element_sources=self.cutter.pending_geometry.source if self.args.streaming_topology else None,
            )
            if surface_count > self.contact.capacity:
                if self.reserve is not None:
                    raise ValueError(
                        f"boundary surface reserve exhausted: the cut needs {surface_count} triangles, "
                        f"{self.contact.capacity} are reserved"
                    )
                self._grow_surface(surface_count)
            else:
                self.contact.update_topology(self.cutter.surface, surface_count, self.state_0.particle_q)
            if self.device.is_cuda:
                wp.synchronize_device(self.device)
            self.cutter.commit()
            self.fragment_count, self._surface_count = diagnostics["fragment_count"], surface_count
            self.cut_revision += 1
        except (MemoryError, ValueError, RuntimeError) as error:
            self.cutter.discard()
            if holds_staged:
                self.holds.restore()
            self.contact, self.surface, self._render_indices, self.graph = previous_surface_buffers
            if published:
                self._model_factory.copy_into(self.model, self._backup_model)
                self.state_0.assign(self._backup_state)
                self.state_1.assign(self._backup_state)
                self._update_mechanical_components(self.cutter.material_components)
                self.solver.refresh_component_scaling()
                self.solver.refresh_topology(self.state_0.particle_q, self.sim_dt)
                wp.copy(self.solver.velocity_increment, self._backup_velocity_increment)
                self.contact.update_topology(self.cutter.surface, previous_surface_count, self.state_0.particle_q)
            self.fragment_count, self._surface_count = previous_count, previous_surface_count
            self.cut_error = rejection_message(error, self.cutter.geometry.capacity, fixed=self.reserve is not None)
            return False
        self.last_cut_ms = 1000.0 * (perf_counter() - start)
        self.last_rebuild_ms = self.last_cut_ms - self.last_geometry_ms
        self.cut_error = ""
        return True

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
        self._frame += 1
        if self.script is not None:
            self.script.play()
        self._consume_cylinder()
        self._advance_sphere()
        triangles = self.screen_cut.consume()
        if len(triangles):
            self.cut(triangles)
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

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

        Raises AssertionError on a CUDA graph capture or a Krylov solve outside a graph after a
        start-up mark (the construction and every Reset) that did not fall into a growth of
        storage, on a kernel module load after a mark and on a start-up graph that was never
        launched. The captures inside a growth are counted apart and listed with their growth:
        without ``--cut_reserve`` an edit may grow storage, and a growth binds new buffers, so the
        graphs on them are captured in that frame. With ``--cut_reserve`` nothing grows and every
        capture after a mark fails.

        Returns:
            The numbers of the check, for the line that ``test_final`` prints.
        """
        growths = [
            {
                "what": growth["what"],
                "from": growth["from"],
                "to": growth["to"],
                "frame": growth["frame"],
                "ms": round(1000.0 * growth["seconds"], 1),
                "captures": growth["captures"] + growth.get("solver_captures", 0),
            }
            for growth in self.storage_growths
        ]
        forced = sum(growth["captures"] for growth in self.storage_growths)
        forced_solver = sum(growth.get("solver_captures", 0) for growth in self.storage_growths)
        if self.reserve is not None and growths:
            raise AssertionError(f"Storage grew although the cut reserve is fixed: {growths}")
        if not growths:
            numbers = self.startup.check()
        else:
            census = self.startup.census()
            if census["capture_census_status"] != "checked":
                numbers = self.startup.check()
            else:
                beyond = dict(census)
                beyond["captures_after_startup"] -= forced
                beyond["solver_captures_after_startup"] -= forced_solver
                failure = census_failure(beyond)
                if failure is not None:
                    raise AssertionError(
                        "Graph captures or kernel module loads after start-up that no storage growth forced, or a "
                        f"start-up graph that was never launched: {failure}; storage growths {growths}"
                    )
                numbers = {name: value for name, value in census.items() if not isinstance(value, dict)}
        return {
            **numbers,
            "storage_growths": growths,
            "captures_forced_by_storage_growth": forced + forced_solver,
            "cut_reserve": None if self.reserve is None else self.reserve["scale"],
            # Scratch vertices, scratch faces and work entries of the subtractor as it stands.
            "subtractor_storage": list(self.cutter.subtractor_storage),
        }

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
