# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Suture a wound: the needle makes a row of stitches through FEM tissue, its thread follows and closes the wound.

The regional pad, the free MuJoCo needle, its finite force/torque drive and
the needle-tissue interaction are inherited from fem_needle_insertion. The
run has three parts.

Stitching. The needle makes the stitches one after the other, without a
reset. Each stitch is one turn of the needle about its circle's centre: in at
the entry, through the tissue, out at the exit and on until the tail has left
the tissue. The needle then turns on through the air to the next stitch. The
stitches are either planned as a row, the first entry and exit and copies
moved along the wound, or picked on the map as pairs of entry and exit; only
stitches the needle can make without touching fixed or inactive pad material
are accepted.

Closing. When the last stitch is made the thread is tied to the tissue at the
first entry, and the needle pulls on across the wound, towards where the next
entry would be. The thread tightens and draws the lips of the wound together.
The pull ends when the lips at the stitches are closer than the thread is
thick, when a tetrahedron of the tissue is squeezed below --least-volume of
its volume, or at the edge of the pad.

Drop. The drive's spring lets go of the needle and it sinks under its weight
onto the pad. This tests the inherited contact between the MuJoCo needle and
the FEM tissue, and the run ends with its outcome. That contact shares one
stiffness over the length of the shaft: it carries a needle that lies along
the surface, and gives way under one that stands on its end. The drive's
damping stays on, because the needle-tissue coupling does not converge for
the light needle in contact without it; the needle therefore sinks at the
speed where that damping balances its weight instead of falling freely.

A thread trails from the needle's tail. It lies on the deforming pad and is
held there by friction. Where the tail has passed through the tissue the
thread stays in the track: it must pass through material points on the
stitch's planned circle and slides through them against friction, and it
pulls the tissue with the opposite of every force the tissue puts on it. Spheres mark where the thread goes
into the tissue and where it comes out. The thread's far end is free and its
spare length lies in loose rows on the pad.

The thread pulls on the needle at its tail. The needle feels that pull one
step late and yields to it as far as its drive's spring and damper allow, so
a taut thread tilts and shifts the needle. After the pull the thread is also
tied at the last exit: the stitches stay tight when the needle is let go, and
the released needle is drawn back by the stretched end of the thread.

The thread also keeps out of itself and out of the needle: two stretches of
it, or a stretch and the needle's shaft, rest against each other without
friction.

Scope. The thread's forces reach the tissue and the needle relaxed and one
step late, and its contacts with itself and the needle are found on the
straight chords between its nodes at the start of each step. The tissue has
no contact with itself, so closed lips can overlap.

Run: python -m newton_solver_suturebot.examples fem_needle_suture
"""

import math
from argparse import BooleanOptionalAction
from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp

from newton_solver_suturebot import (
    NeedleCouplingNotConverged,
    NeedleStepTooLarge,
    SolverThread,
    ThreadAttachment,
    ThreadCapsuleContact,
    ThreadModel,
    ThreadTissueCoupling,
)
from newton_solver_suturebot._src.utils import capture
from newton_solver_suturebot.examples._startup import Script, launch_startup_graph, load_kernel_modules, owns_startup
from newton_solver_suturebot.examples._tissue_markers import TissueMarkers
from newton_solver_suturebot.examples.needle._needle_insertion_drive import pose_velocity, ramp_distance
from newton_solver_suturebot.examples.needle._needle_insertion_geometry import pick_material_point
from newton_solver_suturebot.examples.needle._suture_stitches import NeedleSchedule, plan_stitch, plan_stitches
from newton_solver_suturebot.examples.needle._thread_pins import bridge, lay_along_path, serpentine
from newton_solver_suturebot.examples.needle._thread_view import ThreadView
from newton_solver_suturebot.examples.needle.example_fem_needle_insertion import Example as InsertionExample

GRAVITY = 9.81  # [m/s²]
THREAD_RADIUS = 1.5e-4  # [m]
THREAD_DENSITY = 1140.0  # [kg/m³]
# The dropped needle is at rest once it has moved slower than this for the settling time.
REST_SPEED = 1.0e-3  # [m/s]
REST_TIME = 0.5  # [s]
# The look of the dVRK grippers' jaws.
GRIPPER_LOOK = Path(__file__).parents[1] / "assets/dvrk/models/replay_appearance.usda"
GRIPPER_SHADER = "/SutureBotReplay/IllustrativeVersiusMetal/BulkMetal/Surface"
ENTRY_COLOR, EXIT_COLOR = (0.2, 1.0, 0.3), (1.0, 0.25, 0.7)
THREAD_IN_COLOR, THREAD_OUT_COLOR = (0.2, 0.5, 1.0), (1.0, 0.55, 0.1)


def gripper_metal():
    """Colour, roughness and metallic value of the material the dVRK grippers' jaws are drawn with."""
    from pxr import Usd, UsdShade  # noqa: PLC0415

    stage = Usd.Stage.Open(str(GRIPPER_LOOK))  # the prim is only valid while its stage is held
    shader = UsdShade.Shader(stage.GetPrimAtPath(GRIPPER_SHADER))
    color = tuple(float(value) for value in shader.GetInput("diffuseColor").Get())
    return color, float(shader.GetInput("roughness").Get()), float(shader.GetInput("metallic").Get())


@wp.kernel(enable_backward=False)
def _gather_points(source: wp.array[wp.vec3d], slots: wp.array[int], points: wp.array[wp.vec3]):
    """Copy the thread's points at the given eyelets for drawing; launch over the points."""
    point = source[slots[wp.tid()]]
    points[wp.tid()] = wp.vec3(float(point[0]), float(point[1]), float(point[2]))


@wp.kernel(enable_backward=False)
def _wound_gaps(target: wp.array[wp.vec3d], near: wp.array[int], far: wp.array[int], closure: wp.array[float]):
    """Add up the distances between the two lips of the wound at the stitches; launch over the pairs."""
    pair = wp.tid()
    wp.atomic_add(closure, 0, float(wp.length(target[far[pair]] - target[near[pair]])))


@wp.kernel(enable_backward=False)
def _largest_pull(force: wp.array[wp.vec3d], closure: wp.array[float]):
    """Largest force of the thread on a track point; launch over the eyelets."""
    wp.atomic_max(closure, 1, float(wp.length(force[wp.tid()])))


@wp.kernel(enable_backward=False)
def _add_weight(body: int, weight: wp.vec3, body_f: wp.array[wp.spatial_vector]):
    """Add a body's weight at its centre of mass."""
    body_f[body] += wp.spatial_vector(weight, wp.vec3())


class Example(InsertionExample):
    def __init__(self, viewer, args=None, *, startup=False):
        """Build the scene and the row of stitches.

        Args:
            viewer: Newton viewer.
            args: Parsed arguments of :meth:`create_parser`; None for its defaults.
            startup: Run the start-up call of this example (the inherited ``capture_graphs``)
                whenever the row is built, at construction and at every rebuild, between the
                marks of a start-up (``examples/_startup.py``): the run then captures no CUDA
                graph and loads no kernel module in a frame, and ``--test`` fails if it does.
                The example's entry point asks for it.
        """
        args = newton.examples.default_args(self.create_parser()) if args is None else args
        if args.stitches < 1 or args.thread_elements < 2 or args.updates < 1:
            raise ValueError("At least 1 stitch, 2 thread elements and 1 Newton update are required")
        values = (args.stitch_pitch, args.eyelet_spacing, args.air_speed, args.thread_length, args.thread_modulus)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Stitch, speed and thread parameters must be finite and positive")
        if not 0.0 < args.least_volume < 1.0:
            raise ValueError("The least tissue volume must be a fraction between 0 and 1")
        self.stitches, self.schedule, self.stitch_index, self.phase = [], None, 0, "stitch"
        self.thread = self.coupling = self.thread_view = None
        self.show_nodes, self.show_crossings = False, True
        # None plans the row of the arguments; a list holds the entry and exit pairs picked on the map [m].
        self._stitch_points, self._pending_entry = None, None
        self._metal = gripper_metal()
        if startup and owns_startup(self, Example):
            self._open_startup()
        super().__init__(viewer, args)
        self.frame_row()
        if args.autostart and self.stitches:
            self.start()
        if self.startup is not None and owns_startup(self, Example):
            self._end_startup(frame=True)

    def configure_builder(self, builder):
        """Add the needle and draw it with the grippers' metal."""
        super().configure_builder(builder)
        color, roughness, metallic = self._metal
        for source in builder.shape_source:
            if isinstance(source, newton.Mesh):
                source.color, source.roughness, source.metallic = color, roughness, metallic

    # ------------------------------------------------------------------ scene

    def select_regions(self, region_ids):
        """Rebuild the scene, plan the stitches, lay the thread and place the needle before the first entry."""
        self.thread = self.coupling = self.thread_view = self.attachment = self.needle_contact = None
        self.stitches, self.schedule, self.stitch_index, self.phase = [], None, 0, "stitch"
        previous, pick_mode = tuple(self.selected_region_ids), getattr(self, "pick_mode", "region")
        super().select_regions(region_ids)
        if tuple(self.selected_region_ids) == previous:
            self.pick_mode = pick_mode
        else:
            # Picked stitches belong to the region they were picked on.
            self._stitch_points, self._pending_entry = None, None
        if self.solver is None:
            if self.startup is not None and self.screen_pick is not None and owns_startup(self, Example):
                # A rebuild without tissue: nothing to prepare, the start-up ends here.
                self._end_startup(frame=False)
            return
        pick = "Choose Add stitch, then click an entry and an exit on the map."
        try:
            self.stitches = self._plan()
            if self.stitches:
                self._build_row()
                self.outcome = f"{len(self.stitches)} stitches ready. Press Start."
            else:
                self.outcome = f"No stitches. {pick}"
        except ValueError as error:
            self.stitches, self.schedule = [], None
            self.thread = self.coupling = self.thread_view = None
            self.entry = self.exit = self.plan = None
            self.outcome = f"No stitches here: {error}. {pick}"
        if self.startup is not None and owns_startup(self, Example):
            if self.stitches:
                # The row is known: its coupled step is built and the start-up call runs here, so
                # that Start and the frames after it capture and load nothing.
                self._build_coupling()
                self._startup_status = "start-up not marked"
                self._startup_ready = self.capture_graphs()
            if self.screen_pick is not None:
                # A rebuild (a reset, a region switch, a changed row): the base class opened its start-up.
                self._end_startup(frame=False)
        self.frame_row()

    def _plan(self):
        """The stitches of the row: planned from the arguments, or one per picked pair."""
        args = self.args
        scene = (self.geometry, self.region_map, self.reference, self.selection, self.fixed_vertex_indices)
        if self._stitch_points is None:
            return plan_stitches(
                *scene, args.first_entry_xy, args.first_exit_xy, args.stitches, args.stitch_pitch, args.eyelet_spacing
            )
        return [
            plan_stitch(*scene, entry, exit_point, args.eyelet_spacing) for entry, exit_point in self._stitch_points
        ]

    def _build_row(self):
        """Schedule, thread, coupling and markers for the planned stitches."""
        args, device, geometry = self.args, self.device, self.geometry
        # Picked stitches may point anywhere: between them the needle travels above the top of the pad.
        clear = None
        if self._stitch_points is not None:
            clear = float(self.reference.vertices[:, 2].max()) + geometry.thickness
        self.schedule = NeedleSchedule(self.stitches, args.insertion_speed, args.air_speed, clear)
        self._arm_stitch(0)
        start = self.stitches[0].plan.pose(self.stitches[0].plan.initial_angle)
        self.desired_pose = start
        self._initialize_pose(start)

        # The thread leaves the tail along the axis of the needle's last collision capsule.
        outward = geometry.centerline[-1] - geometry.centerline[-2]
        outward /= np.linalg.norm(outward)
        element_length = args.thread_length / args.thread_elements
        straight = np.linspace(0.0, args.thread_length, args.thread_elements + 1)[:, None] * np.array([1.0, 0.0, 0.0])
        self.thread_model = ThreadModel.create(
            straight, young_modulus=args.thread_modulus, radius=THREAD_RADIUS, density=THREAD_DENSITY, device=device
        )
        # A tissue node resists a point force with a stiffness of about E h. The same stiffness stands for the
        # give of the tissue around the thread that the mesh does not resolve: it is the contact and the
        # eyelet spring. A quarter of it estimates the stiffness used to relax
        # the thread's force on the tissue; it does not bound every deformed configuration.
        edges = (
            self.selection.vertices[self.selection.indices[:, 1:]]
            - self.selection.vertices[self.selection.indices[:, :1]]
        )
        tissue_edge = float(np.linalg.norm(edges, axis=2).mean())
        tissue_stiffness = args.young_modulus * tissue_edge
        node_weight = THREAD_DENSITY * math.pi * THREAD_RADIUS**2 * element_length * GRAVITY
        self.thread = SolverThread(
            self.model,
            self.thread_model,
            gravity=(0.0, 0.0, -GRAVITY),
            mass_damping=2.0,
            tolerance=0.1 * node_weight,
            max_iterations=args.updates,
            contact_distance=THREAD_RADIUS,
            contact_stiffness=tissue_stiffness,
            contact_smoothing=0.1 * THREAD_RADIUS,
            friction=args.shaft_friction,
            max_eyelets=sum(len(stitch.angles) for stitch in self.stitches),
            eyelet_stiffness=tissue_stiffness,
            self_contact=True,
            obstacle_distance=THREAD_RADIUS + geometry.thickness,
        )
        self.thread.q.assign(self._initial_thread(start, outward, element_length).reshape(-1))
        self.thread.boundary_q.assign(self.thread.q)
        self.coupling = ThreadTissueCoupling(
            self.thread,
            self.reference.vertices,
            self.reference.exterior_surface_triangles,
            self.selection.indices,
            np.searchsorted(self.reference.global_vertex_indices, self.selection.global_vertex_indices),
            tissue_stiffness=0.25 * tissue_stiffness,
            contact_margin=args.contact_margin,
            max_tracks=len(self.stitches),
        )
        self._slots = [self.coupling.add_track(stitch.tetrahedra, stitch.weights) for stitch in self.stitches]
        self._threaded = [0] * len(self.stitches)
        self._tail = wp.vec3(*geometry.tail)
        # The thread's first node sits on the needle's tail and pulls on the needle, one step late. The drive's
        # spring and damper hold the needle against that pull.
        self.attachment = ThreadAttachment(
            self.thread, self.model, [self.needle_body], geometry.tail, outward, lagged=True
        )
        self._support_needle(spring=True)
        # The thread stays out of the needle's capsules. Within a step the needle resists it at least with
        # its inertia and the drive's damper.
        self.needle_contact = ThreadCapsuleContact(
            self.thread,
            self.model,
            self.needle_body,
            geometry.centerline,
            dt=self.sim_dt,
            damping=(self.drive.damping, self.drive.angular_damping),
        )
        weight = float(self.model.body_mass.numpy()[self.needle_body]) * GRAVITY
        self._weight = wp.vec3(0.0, 0.0, -weight)
        # The drop may last twice the time the needle takes to sink by its own diameter. It sinks at the speed
        # at which the drive's damper, which stays in MuJoCo, balances its weight.
        self._drop_limit = 2.0 * (2.0 * geometry.radius) * self.drive.damping / weight
        self._thread_particle_q = wp.clone(self.state_0.particle_q)
        self._thread_body_q = wp.clone(self.state_0.body_q)
        self._thread_graph = None
        self._lower_thread(tissue_edge)
        if device.is_cuda:
            self._capture_thread()
        self.thread_view = ThreadView(self.viewer, self.thread, THREAD_RADIUS)

        def vectors(values):
            return wp.array(np.asarray(values, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3, device=device)

        # The planned entry and exit of every stitch ride on their surface triangles: the entries, then the exits.
        self._stitch_markers = TissueMarkers(
            self.viewer, "/suture/stitches", radius=0.0004, device=device, per_world=2 * len(self.stitches)
        )
        for end, color in (("entry", ENTRY_COLOR), ("exit", EXIT_COLOR)):
            for stitch in self.stitches:
                point = getattr(stitch.plan, end)
                self._stitch_markers.attach(point.node_indices, point.weights, color)
        # Where the thread crosses the tissue surface: the first and the last track point of every passage.
        # They are listed in the order the thread reaches them, so the threaded ones are always the first.
        self._crossings = {}
        for name, end, color in (("in", 0, THREAD_IN_COLOR), ("out", 1, THREAD_OUT_COLOR)):
            order = [(index, passage[end]) for index, stitch in enumerate(self.stitches) for passage in stitch.passages]
            slots = [self._slots[index][local] for index, local in order]
            self._crossings[name] = (
                order,
                wp.array(slots, dtype=int, device=device),
                wp.zeros(len(slots), dtype=wp.vec3, device=device),
                vectors([color] * len(slots)),
            )
        # The wound's width at a stitch: from the last track point in one lip to the first in the other.
        near, far, rest = [], [], []
        for index, stitch in enumerate(self.stitches):
            for before, after in zip(stitch.passages[:-1], stitch.passages[1:], strict=True):
                near.append(self._slots[index][before[1]])
                far.append(self._slots[index][after[0]])
                rest.append(np.linalg.norm(stitch.points[after[0]] - stitch.points[before[1]]))
        self._gap_near = wp.array(near, dtype=int, device=device)
        self._gap_far = wp.array(far, dtype=int, device=device)
        self._closure = wp.zeros(2, dtype=float, device=device)
        self.rest_gap = float(np.mean(rest)) if rest else None
        self.wound_gap, self.largest_pull, self.least_volume = self.rest_gap, 0.0, 1.0
        self._pull, self._pull_reach, self._pulled = None, 0.0, 0.0
        self._drop_start, self._still, self.closing, self.drop_result = None, 0.0, "", ""

    def _arm_stitch(self, index):
        """Make stitch ``index`` the one the inherited markers, camera and failure report refer to."""
        self.stitch_index = index
        plan = self.stitches[index].plan
        self.entry, self.exit, self.plan = plan.entry, plan.exit, plan
        self.last_exit_error = self.exit_error = None
        self.actual_tip_history = []

    def _pad_height(self, points, cell):
        """Height of the pad's top under each of ``points``, taken over the cells of a grid around them [m]."""
        vertices = self.reference.vertices
        low = vertices[:, :2].min(axis=0)
        index = ((vertices[:, :2] - low) / cell).astype(int)
        grid = np.full(index.max(axis=0) + 1, -np.inf)
        np.maximum.at(grid, (index[:, 0], index[:, 1]), vertices[:, 2])
        # The highest of a cell and its eight neighbours, so that a point beside a step counts as on it.
        padded = np.pad(grid, 1, constant_values=-np.inf)
        rows, columns = grid.shape
        grid = np.max([padded[i : i + rows, j : j + columns] for i in range(3) for j in range(3)], axis=0)
        grid[np.isinf(grid)] = vertices[:, 2].max()
        cells = np.clip(((np.asarray(points)[:, :2] - low) / cell).astype(int), 0, np.array(grid.shape) - 1)
        return grid[cells[:, 0], cells[:, 1]]

    def _initial_thread(self, pose, outward, element_length):
        """Nodal coordinates of the unstrained thread before it is lowered: from the tail over to loose rows.

        The spare length lies level with the top of the pad. Its rows begin
        beside the first entry, on the side away from the exit, and are joined
        by turns that the thread's weight can hold flat. The rows may run in any direction
        and may be shortened, and the thread may come down from the tail in
        a tighter or a wider curve. Of the layouts that keep the whole
        thread on the pad, the one that stays clear of the stitches is taken,
        then the one that bends the thread least on its way down from the
        tail, then the one with the longest rows, then the one farthest from
        the stitches.
        """
        args, vertices = self.args, self.reference.vertices
        matrix = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
        tail, leave = matrix @ self.geometry.tail + pose[:3], matrix @ outward
        first = self.stitches[0].plan
        aside = (first.entry.reference_position - first.exit.reference_position)[:2]
        aside /= np.linalg.norm(aside)
        # A thread lying on the pad holds bends down to its elasto-gravity length (E I / rho A g)^(1/3);
        # tighter turns spring open.
        turn_radius = (args.thread_modulus * THREAD_RADIUS**2 / (4.0 * THREAD_DENSITY * GRAVITY)) ** (1.0 / 3.0)
        corner = first.entry.reference_position[:2] + turn_radius * aside
        top = float(vertices[:, 2].max()) + 2.0 * THREAD_RADIUS
        start = tail + element_length * leave
        step = 0.5 * element_length
        # The thread keeps one element away from the pad's edge.
        low = vertices[:, :2].min(axis=0) + element_length
        high = vertices[:, :2].max(axis=0) - element_length
        marks = np.array(
            [
                point.reference_position[:2]
                for stitch in self.stitches
                for point in (stitch.plan.entry, stitch.plan.exit)
            ]
        )
        best, best_rank = None, None
        for row_length in (args.row_length, 0.75 * args.row_length, 0.5 * args.row_length, 0.25 * args.row_length):
            for angle in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
                along = np.array([math.cos(angle), math.sin(angle), 0.0])
                for side in (1.0, -1.0):
                    across = side * np.array([-along[1], along[0], 0.0])
                    rows = serpentine(
                        np.array([*corner, top]), along, across, row_length, turn_radius, args.thread_length, step
                    )
                    for reach in (1.0 / 3.0, 2.0 / 3.0, 1.0):
                        lead = np.vstack((tail, bridge(start, leave, rows[0], along, step, reach)))
                        nodes = lay_along_path(np.vstack((lead, rows[1:])), args.thread_elements + 1, element_length)
                        if np.any(nodes[:, :2] < low) or np.any(nodes[:, :2] > high):
                            continue
                        # The rows are the nodes past the lead from the tail down to the pad.
                        lead_nodes = int(np.ceil(np.linalg.norm(np.diff(lead, axis=0), axis=1).sum() / element_length))
                        distance = np.linalg.norm(nodes[lead_nodes:, None, :2] - marks[None], axis=2).min(axis=1)
                        # Clear of the stitches means as far from them as the rows begin: one turn radius.
                        clear = round(min(float(distance.min()), turn_radius) / turn_radius, 2)
                        # A bent element's chord is shorter than the element. A kink on the way down from the
                        # tail would start the run with the thread springing open.
                        chords = np.linalg.norm(np.diff(nodes[:, :3], axis=0), axis=1) / element_length
                        rank = (clear, round(float(chords.min()), 2), row_length, float(distance.mean()))
                        if best_rank is None or rank > best_rank:
                            best, best_rank = nodes, rank
        if best is None:
            raise ValueError("The loose thread does not fit on the pad beside these stitches; shorten the thread")
        return best

    def _lower_thread(self, cell):
        """Lower the unstrained thread onto the pad before the run, so that the run starts with it at rest.

        The thread's mass damping is raised until its weight moves it by its
        radius per step, and it is stepped for one and a half times the way
        to the lowest point of the pad under it. Friction holds it where it
        lands.
        """
        thread = self.thread
        nodes = thread.q.numpy().reshape(-1, 6)
        way = float((nodes[:, 2] - self._pad_height(nodes, cell)).max())
        damping, thread.mass_damping = thread.mass_damping, GRAVITY * self.sim_dt / THREAD_RADIUS
        wp.copy(self._thread_particle_q, self.state_0.particle_q)
        wp.copy(self._thread_body_q, self.state_0.body_q)
        for _ in range(int(np.ceil(1.5 * way / THREAD_RADIUS))):
            self._advance_thread(self.sim_dt)
        thread.mass_damping = damping
        thread.qd.zero_()

    # --------------------------------------------------------- picked stitches

    def _picked(self):
        """The row as entry and exit pairs on the map [m], whether it was planned or picked."""
        if self._stitch_points is not None:
            return list(self._stitch_points)
        return [
            (stitch.plan.entry.reference_position[:2].copy(), stitch.plan.exit.reference_position[:2].copy())
            for stitch in self.stitches
        ]

    def _rebuild(self, points):
        self._stitch_points, self._pending_entry = points, None
        self.select_regions(self.selected_region_ids)

    def pick(self, pixel_x, pixel_y):
        """Take a click on the map as the entry of a new stitch, then the next one as its exit."""
        point = pick_material_point(self.region_map, self.reference, self.selection, pixel_x, pixel_y)
        position = point.reference_position[:2].copy()
        if self._pending_entry is None:
            self._pending_entry = position
            self.outcome = "Entry picked. Click the exit of this stitch on the map."
        else:
            entry, self._pending_entry = self._pending_entry, None
            self.add_stitch(entry, position)

    def add_stitch(self, entry_xy, exit_xy):
        """Append a stitch between two map positions [m] to the row; raises ValueError if the needle cannot make it."""
        scene = (self.geometry, self.region_map, self.reference, self.selection, self.fixed_vertex_indices)
        plan_stitch(*scene, entry_xy, exit_xy, self.args.eyelet_spacing)
        previous = self._stitch_points
        points = [*self._picked(), (np.asarray(entry_xy, dtype=float), np.asarray(exit_xy, dtype=float))]
        self._rebuild(points)
        if len(self.stitches) != len(points):
            # The row could not be built with this stitch: say why and keep the row as it was.
            reason = self.outcome
            self._rebuild(previous)
            raise ValueError(reason)
        self.outcome = f"{len(self.stitches)} stitches. Add more, or press Start."

    def remove_stitch(self):
        """Take the last stitch off the row."""
        self._rebuild(self._picked()[:-1])

    def clear_stitches(self):
        self._rebuild([])

    def plan_row(self):
        """Go back to the row planned from the arguments."""
        self._rebuild(None)

    def _planned_passages(self):
        """Every stitch is a passage of the needle; across the wound it has a section in each lip."""
        return len(self.stitches), max(1, *(len(stitch.passages) for stitch in self.stitches))

    def start(self):
        if not self.stitches:
            raise ValueError("No stitches. Choose Add stitch, then click an entry and an exit on the map.")
        super().start()
        self.outcome = f"Stitch {self.stitch_index + 1} of {len(self.stitches)}"

    def reset(self):
        super().reset()
        if self.args.autostart and self.stitches:
            self.start()

    # ----------------------------------------------------------------- camera

    def frame_row(self):
        """Point the camera at the row of stitches, from above the loose thread on the entry side."""
        if not self.stitches:
            self.reset_camera()
            return
        turn = np.linspace(0.0, 2.0 * np.pi, 13)
        points = np.vstack([stitch.plan.tip(turn) for stitch in self.stitches])
        center = 0.5 * (points.min(axis=0) + points.max(axis=0))
        radius = float(np.linalg.norm(points - center, axis=1).max())
        first = self.stitches[0].plan
        aside = first.entry.reference_position - first.exit.reference_position
        aside[2] = 0.0
        # Across the wound and down at 40 degrees, with the row running a little into the picture.
        offset = aside / np.linalg.norm(aside) + 0.4 * first.axis / np.linalg.norm(first.axis)
        offset = np.array([*offset[:2], math.tan(math.radians(40.0)) * np.linalg.norm(offset[:2])])
        distance = 3.4 * radius
        offset *= distance / np.linalg.norm(offset)
        direction = -offset / distance
        right = np.cross(direction, [0.0, 0.0, 1.0])
        # Leave the side panel clear: it covers the left third of the window, a quarter of the distance here.
        target = center - 0.25 * distance * right / np.linalg.norm(right)
        self.viewer.set_camera(
            pos=tuple(target + offset),
            pitch=float(np.degrees(np.arcsin(direction[2]))),
            yaw=float(np.degrees(np.arctan2(direction[1], direction[0]))),
        )
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "set_pivot"):
            camera.set_pivot(target)

    # ------------------------------------------------------------------- run

    def _observe_interaction(self):
        """Log the needle's punctures, entries and exits; a row of stitches does not stop at the first exit."""
        for event in self.interaction.events:
            self.interaction_events.append(
                {
                    "kind": event["kind"],
                    "time_s": self.elapsed,
                    "position_m": np.asarray(event["position"], dtype=float).tolist(),
                    "reused": bool(event.get("reused", False)),
                    "stitch": self.stitch_index,
                    "phase": self.phase,
                }
            )

    def _thread_tracks(self, index, angle):
        """Pass the thread through the track points of stitch ``index`` that the tail has gone by."""
        stitch = self.stitches[index]
        passed = int(np.searchsorted(stitch.angles, angle - self.geometry.span, side="right"))
        for slot in self._slots[index][self._threaded[index] : passed]:
            self.coupling.thread_eyelet(slot)
        self._threaded[index] = max(self._threaded[index], passed)

    def _support_needle(self, spring):
        """Tell the thread what holds the needle: the drive's damper and, while it drives, its spring."""
        drive = self.drive
        stiffness = (drive.stiffness, drive.angular_stiffness) if spring else (0.0, 0.0)
        self.attachment.set_support((drive.damping, drive.angular_damping), stiffness)

    def _advance_thread(self, dt):
        """One thread step against the needle and tissue state in the thread's input buffers."""
        self.attachment.attach(self._thread_body_q, dt)
        self.needle_contact.update(self._thread_body_q)
        self.coupling.update(self._thread_particle_q)
        self.thread.step(None, None, None, None, dt)
        # The thread's wrenches on the needle and its forces on the tissue, all for the next step.
        self.attachment.update()
        self.needle_contact.react(self._thread_body_q)
        self._external_forces.zero_()
        self.coupling.add_forces(self._external_forces)

    def _capture_thread(self):
        """Capture the thread step in a CUDA graph; it reads the step size from device data and serves every one."""
        state = (
            *self.thread.state,
            *self.coupling.state,
            *self.needle_contact.state,
            self._external_forces,
            self.attachment.wrench,
        )
        saved = [wp.clone(array) for array in state]
        # The first step loads the kernels and makes the step's allocations; neither may happen in a capture.
        self._advance_thread(self.sim_dt)
        with capture.ScopedCapture(device=self.device) as scope:
            self._advance_thread(self.sim_dt)
        self._thread_graph = scope.graph
        if self.startup is not None:
            # Warp builds the executable of a graph at its first launch: that launch belongs to the
            # start-up, not to the first frame. The thread is put back below.
            launch_startup_graph(scope.graph, self.device)
        for array, values in zip(state, saved, strict=True):
            wp.copy(array, values)

    def _step_thread(self, dt):
        """Advance the thread over an accepted needle and tissue step, and collect its forces on the tissue."""
        # The two states swap every step; the captured graph reads fixed buffers.
        wp.copy(self._thread_particle_q, self.state_0.particle_q)
        wp.copy(self._thread_body_q, self.state_0.body_q)
        if self._thread_graph is not None:
            # The graph reads the step size from the thread's step time, so one graph serves the
            # nominal step, a step halved after a rejection and the remainder of a frame. The
            # needle contact's relaxation is the one number formed from the nominal step at
            # construction, on this path as on the eager one.
            self.thread.step_time.set(dt)
            wp.capture_launch(self._thread_graph)
        else:
            self._advance_thread(dt)

    def _target(self, time):
        """The needle's target at ``time``: its pose, the stitch in progress and the tip angle on that stitch."""
        if self.phase == "stitch":
            return self.schedule.at(time)
        start, pose, direction = self._pull
        if self.phase == "pull":
            self._pulled = min(float(ramp_distance(time - start, self.args.insertion_speed)), self._pull_reach)
        target = pose.copy()
        target[:3] += self._pulled * direction
        return target, len(self.stitches) - 1, self.stitches[-1].leave_angle

    def _begin_pull(self, pose):
        """Tie the thread at the first entry and set the needle's pull across the wound."""
        self.phase = "pull"
        self.coupling.hold_eyelet(self._slots[0][0])
        last = self.stitches[-1].plan
        across = last.entry.reference_position - last.exit.reference_position
        across[2] = 0.0
        chord = float(np.linalg.norm(across))
        # The thread climbs from the pad at the slope it left the tissue with, the track's slope at the exit.
        slope = math.asin(min(1.0, chord / (2.0 * self.geometry.radius)))
        # The needle stays over the pad: its circle may reach the pad's edge.
        vertices = self.reference.vertices
        low = vertices[:, :2].min(axis=0) + self.geometry.radius
        high = vertices[:, :2].max(axis=0) - self.geometry.radius

        def reach(level):
            limits = [
                ((high[axis] if level[axis] > 0 else low[axis]) - pose[axis]) / level[axis]
                for axis in range(2)
                if abs(level[axis]) > 1.0e-9
            ]
            return max(0.0, min(limits)) / math.cos(slope)

        # Towards where the next entry of the row would be: back across the wound and one stitch on. Where the
        # pad's edge leaves the needle less than its own diameter of room that way, the pull goes straight back
        # across the wound or back along the row instead, whichever has the most room.
        ways = [across / chord]
        if len(self.stitches) > 1:
            ahead = last.center - self.stitches[-2].plan.center
            ahead[2] = 0.0
            ways = [way / np.linalg.norm(way) for way in (across + ahead, across, across - ahead)]
        level = ways[0]
        if reach(level) < 2.0 * self.geometry.radius:
            level = max(ways, key=reach)
        direction = math.cos(slope) * level + math.sin(slope) * np.array([0.0, 0.0, 1.0])
        self._pull_reach = reach(level)
        self._pull, self._pulled = (self.elapsed, np.asarray(pose, dtype=float).copy(), direction), 0.0

    def _observe_closing(self):
        """Once per frame of the pull: the wound's width, the thread's largest pull and the tissue's least volume."""
        self._closure.zero_()
        if self.rest_gap is not None:
            wp.launch(
                _wound_gaps,
                dim=len(self._gap_near),
                inputs=[self.thread.eyelet_target, self._gap_near, self._gap_far],
                outputs=[self._closure],
                device=self.device,
            )
        wp.launch(
            _largest_pull,
            dim=self.thread.max_eyelets,
            inputs=[self.thread.eyelet_force],
            outputs=[self._closure],
            device=self.device,
        )
        total, self.largest_pull = (float(value) for value in self._closure.numpy())
        # The integrity check of this frame left the smallest volume ratio in its result.
        self.least_volume = float(self._integrity_result.numpy()[2])
        if self.rest_gap is None:
            # Stitches through unbroken tissue have nothing to draw together; pulling would only squeeze it.
            self.closing = "there is no open wound to close"
        else:
            self.wound_gap = total / len(self._gap_near)
            if self.wound_gap <= 2.0 * THREAD_RADIUS:
                self.closing = "the wound is closed"
        if not self.closing and self.least_volume < self.args.least_volume:
            self.closing = "the thread is as tight as the tissue allows"
        if not self.closing and self._pulled >= self._pull_reach:
            self.closing = "the needle is at the edge of the pad"
        if self.closing:
            self.phase = "drop"
            # A knot at the last exit keeps the stitches tight when the needle lets the thread go.
            self.coupling.hold_eyelet(self._slots[-1][-1])
            # The drive's spring lets go. Its damping stays in MuJoCo: the needle-tissue coupling does not
            # converge for a 0.3 g needle in contact without it, so the needle sinks rather than falls.
            self.drive.configure_mujoco(self.rigid_solver, None)
            self._support_needle(spring=False)
            self._drop_start, self._still = self.elapsed, 0.0
            self.outcome = f"Needle dropped: {self.closing}"
        else:
            self.outcome = f"Pulling the thread: {self._gap_text()}"

    def _observe_drop(self):
        """Once per frame of the drop: the run ends when the needle lies still."""
        speed = float(np.linalg.norm(self.state_0.body_qd.numpy()[self.needle_body][:3]))
        self._still = self._still + self.frame_dt if speed < REST_SPEED else 0.0
        if self._still >= REST_TIME:
            self._end_drop("the pad carries the dropped needle, it lies still")
        elif self.elapsed - self._drop_start >= self._drop_limit:
            self._end_drop(f"the dropped needle still moves at {1.0e3 * speed:.1f} mm/s")

    def _end_drop(self, result):
        """End the run with the outcome of the drop, which tests the needle's contact with the pad."""
        self.running = False
        self.phase = "done"
        self.task_status = "success"
        self.drop_result = result
        self.outcome = f"Done: {len(self.stitches)} stitches, {self.closing}; {self._gap_text()}. Drop: {result}"

    def _fail(self, reason):
        if self.phase == "drop":
            # The stitches are made and the wound is drawn together; what gave way is the contact under test.
            self._end_drop(f"the needle's contact with the pad gave way ({reason})")
        else:
            super()._fail(reason)

    def _gap_text(self):
        if self.rest_gap is None:
            return "no wound width to measure"
        return f"wound {1.0e3 * self.wound_gap:.2f} mm of {1.0e3 * self.rest_gap:.2f} mm wide"

    def simulate(self):
        """One frame of the inherited needle and tissue stepping, with the stitch schedule and the thread."""
        if not self.running:
            return
        wp.copy(self._frame_particle_q, self.state_0.particle_q)
        step_dt, remaining = self.sim_dt, self.frame_dt
        minimum_retry_dt = min(self.sim_dt / 256, 5.0e-6)
        quick_steps = 0
        # A start-up call may have built this frame's tangent (see the inherited simulate()): its
        # token serves one frame.
        prepared, self._startup_tangent = self._startup_tangent, None
        if prepared is None or prepared != (self.solver._tangent_version, step_dt, self.accepted_physics_steps):
            self.solver.refresh_tangent(self._frame_particle_q, step_dt)
        while remaining > self.frame_dt * 1.0e-10 and self.running:
            self.state_0.clear_forces()
            # The thread's pull on the needle is that of the last accepted step.
            self.attachment.add_forces(self.state_0.body_f)
            self.needle_contact.add_forces(self.state_0.body_f)
            desired, index, angle = self._target(self.elapsed + step_dt)
            if self.phase == "drop":
                wp.launch(
                    _add_weight,
                    dim=1,
                    inputs=[self.needle_body, self._weight],
                    outputs=[self.state_0.body_f],
                    device=self.device,
                )
            else:
                previous, _, _ = self._target(self.elapsed)
                self.drive.prepare_timestep(step_dt)
                self.drive.add_forces(self.state_0, desired, pose_velocity(previous, desired, self.drive.com, step_dt))
            try:
                # The thread's forces on the tissue are those of the last accepted step.
                self.coupler.step(
                    self.state_0, self.state_1, self.control, step_dt, external_forces=self._external_forces
                )
            except (NeedleStepTooLarge, NeedleCouplingNotConverged):
                # The coupler restores the unaccepted trial; the thread has not moved.
                if step_dt <= minimum_retry_dt:
                    raise
                step_dt *= 0.5
                quick_steps = 0
                self.step_refinements += 1
                self.solver.refresh_tangent(self._frame_particle_q, step_dt)
                continue
            self.maximum_force_balance_error = max(
                self.maximum_force_balance_error, self.interaction.force_balance_error
            )
            self.maximum_torque_balance_error = max(
                self.maximum_torque_balance_error, self.interaction.torque_balance_error
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.elapsed += step_dt
            remaining -= step_dt
            self.accepted_physics_steps += 1
            self.minimum_step_dt = min(self.minimum_step_dt, step_dt)
            self.sim_time = self.elapsed
            self.desired_pose = desired
            if self.phase == "stitch":
                if index != self.stitch_index:
                    self._arm_stitch(index)
                self._thread_tracks(index, angle)
            self._step_thread(step_dt)
            pose_error = self._observe_state(desired)
            if pose_error is None:
                break
            if self.interaction.status == "failed":
                self._fail(self.interaction.reason)
                break
            self.maximum_interface_force = max(
                self.maximum_interface_force, float(np.linalg.norm(self.interaction.interface_force))
            )
            self._observe_interaction()
            if self.phase == "stitch":
                self.outcome = f"Stitch {index + 1} of {len(self.stitches)}: {self.interaction.status}"
            if self.phase != "drop" and pose_error > 0.005:
                self._fail("needle deviated more than 5 mm from the finite-force drive")
                break
            if self.phase == "stitch" and self.elapsed >= self.schedule.duration:
                self._begin_pull(desired)
            quick_steps = quick_steps + 1 if self.coupler.iterations <= 4 else 0
            if remaining > self.frame_dt * 1.0e-10:
                next_dt = min(step_dt, remaining)
                if quick_steps >= 2:
                    next_dt = min(2 * step_dt, self.sim_dt, remaining)
                    quick_steps = 0
                if next_dt != step_dt:
                    step_dt = next_dt
                    self.solver.refresh_tangent(self._frame_particle_q, step_dt)
        self.actual_tip_history.append(self.actual_tip())
        try:
            self._check_tissue_integrity()
        except AssertionError as error:
            self._fail(str(error))
        if self.running and self.phase == "pull":
            self._observe_closing()
        elif self.running and self.phase == "drop":
            self._observe_drop()

    # ---------------------------------------------------------------- display

    def render_pad(self):
        super().render_pad()
        if self.thread_view is None:
            for name in ("/suture/stitches", "/suture/thread_in", "/suture/thread_out"):
                self.viewer.log_points(name, None, hidden=True)
            return
        self.thread_view.render(show_nodes=self.show_nodes)
        self._stitch_markers.draw(self.state_0.particle_q)
        for name, (order, slots, points, colors) in self._crossings.items():
            # A crossing exists once the thread is through its track point.
            count = sum(local < self._threaded[index] for index, local in order) if self.show_crossings else 0
            if count:
                wp.launch(
                    _gather_points,
                    dim=count,
                    inputs=[self.thread.eyelet_point, slots],
                    outputs=[points],
                    device=self.device,
                )
            self.viewer.log_points(
                f"/suture/thread_{name}",
                points[:count] if count else None,
                radii=3.0 * THREAD_RADIUS,
                colors=colors[:count] if count else None,
                hidden=count == 0,
            )

    def _draw_map(self, ui):
        """The top-view map: choose the region, or click entry and exit of a new stitch; the row is drawn on it."""
        for mode, label in (("region", "Region"), ("stitch", "Add stitch")):
            if mode != "region":
                ui.same_line()
            if ui.radio_button(label, self.pick_mode == mode):
                self.pick_mode = mode
        clicked = self.region_map.draw(ui, self.selected_region_ids, max_width=240.0)
        if self.region_map.clicked_pixel is not None:
            if self.pick_mode == "region":
                if clicked is not None and self.selected_region_ids != (clicked,):
                    self._action(self.select_regions, [clicked])
            else:
                self._action(self.pick, *self.region_map.clicked_pixel)
        if self.region_map.screen_rect is None:
            return
        left, top, width, height = self.region_map.screen_rect
        rows, columns = self.region_map.region_ids.shape[:2]
        draw = ui.get_window_draw_list()

        def mark(position, color, label):
            pixel_x, pixel_y = self.region_map.world_to_pixel(float(position[0]), float(position[1]))
            point = ui.ImVec2(left + pixel_x * width / columns, top + pixel_y * height / rows)
            draw.add_circle_filled(point, 3.0, color)
            draw.add_text(ui.ImVec2(point.x + 4, point.y - 6), color, label)

        for number, (entry, exit_point) in enumerate(self._picked(), start=1):
            mark(entry, 0xFF4DFF33, str(number))
            mark(exit_point, 0xFFB340FF, "")
        if self._pending_entry is not None:
            mark(self._pending_entry, 0xFF4DFF33, "entry")

    def gui(self, ui):
        ui.text("Suturing | MuJoCo needle + FEM tissue + thread")
        ui.text_wrapped(self.outcome)
        if self.schedule is not None:
            ui.text(f"Time {self.elapsed:.1f} s; the stitches take {self.schedule.duration:.1f} s")
        if ui.button("Pause" if self.running else "Start"):
            self._action(self.toggle_running)
        ui.same_line()
        if ui.button("Reset"):
            self._action(self.reset)
        if ui.button("Frame row"):
            self.frame_row()
        ui.same_line()
        if ui.button("Frame stitch"):
            self.frame_insertion()
        ui.same_line()
        if ui.button("Frame pad"):
            self.reset_camera()
        ui.separator()
        ui.text_wrapped(
            "Region: click a region on the map to make it the tissue. Add stitch: click an entry, then an exit; "
            "every pair is one stitch, made in this order. A change rebuilds the scene; press Start to run it."
        )
        self._draw_map(ui)
        if ui.button("Remove last"):
            self._action(self.remove_stitch)
        ui.same_line()
        if ui.button("Clear"):
            self._action(self.clear_stitches)
        ui.same_line()
        if ui.button("Planned row"):
            self._action(self.plan_row)
        ui.separator()
        _, self.show_crossings = ui.checkbox("Spheres where the thread crosses the surface", self.show_crossings)
        _, self.show_nodes = ui.checkbox("Thread nodes and slopes", self.show_nodes)
        ui.text_wrapped(
            "Green: entries. Pink: exits. Blue spheres: the thread goes into the tissue. "
            "Orange spheres: the thread comes out. Cyan: planned tip path of the stitch in progress."
        )
        if self.thread is not None:
            ui.text(f"Stitches: {len(self.stitches)}, track points threaded: {sum(self._threaded)}")
            ui.text(
                f"Thread: {self.args.thread_elements} elements, {1.0e3 * self.args.thread_length:.0f} mm, "
                f"{1.0e-6 * self.args.thread_modulus:.0f} MPa"
            )
            if self.phase != "stitch":
                ui.text(f"Closing: {self._gap_text()}")
                ui.text(
                    f"Largest pull on a track point {1.0e3 * self.largest_pull:.1f} mN, "
                    f"least tissue volume {100.0 * self.least_volume:.0f} %"
                )
        ui.text(f"Accepted needle steps: {self.accepted_physics_steps}, refinements: {self.step_refinements}")
        ui.text_wrapped("The thread pulls on the needle one step late and keeps out of itself and of the needle.")
        ui.text("SI units: metres, kilograms, seconds; no mass rescaling")

    def _load_run_modules(self):
        """Also the kernel module of this example: the crossing spheres, the wound's width, the dropped needle's weight."""
        super()._load_run_modules()
        load_kernel_modules(self.device, _gather_points)

    def _test_script(self):
        """``--test`` has no scripted user here: the row starts by itself and runs for the frames it is given."""
        self._script_runs = []
        return Script()

    def test_final(self):
        super().test_final()
        if self.task_status == "failed":
            raise AssertionError(f"Suturing failed: {self.failure_reason}")
        if self.thread is None:
            return
        nodes = self.thread.q.numpy().reshape(-1, 6)
        if not np.isfinite(nodes).all() or not np.isfinite(self.thread.qd.numpy()).all():
            raise AssertionError("Thread state is not finite")
        pose = self.state_0.body_q.numpy()[self.needle_body]
        tail = np.asarray(wp.transform_point(wp.transform(*pose), self._tail), dtype=float)
        if np.linalg.norm(nodes[0, :3] - tail) > THREAD_RADIUS:
            raise AssertionError("The thread left the needle's tail")
        if self.phase != "stitch" and sum(self._threaded) != self.thread.max_eyelets:
            raise AssertionError("The row is finished but the thread is not through every track point")
        pulls = np.linalg.norm(self.thread.eyelet_force.numpy(), axis=1)
        summary = (
            f"Suturing: {self.task_status}, {self.phase}, stitch {self.stitch_index + 1} of {len(self.stitches)}, "
            f"{sum(self._threaded)} of {self.thread.max_eyelets} track points threaded, "
            f"largest pull on a track point {1.0e3 * pulls.max():.3f} mN, "
            f"largest tissue displacement {1.0e3 * self.maximum_deformation:.3f} mm, "
            f"{self.accepted_physics_steps} accepted steps, {self.step_refinements} refinements"
        )
        if self.phase != "stitch":
            summary += f"; {self._gap_text()}, least tissue volume {100.0 * self.least_volume:.0f} %"
        if self.phase == "done":
            summary += f"; {self.closing}; drop: {self.drop_result}"
        print(summary)

    @staticmethod
    def create_parser():
        parser = InsertionExample.create_parser()
        # Thread/tissue reactions are exchanged one step late. The insertion
        # example's single 1/60 s step destabilizes that extra coupling during
        # the second stitch; resolve it at 240 Hz without changing the forces,
        # material, mesh or commanded needle speed.
        parser.set_defaults(insertion_speed=0.004, substeps=4)
        parser.add_argument("--stitches", type=int, default=3, help="Number of stitches in the planned row")
        parser.add_argument(
            "--first-entry-xy", type=float, nargs=2, default=(0.030, -0.040), help="First entry on the map [m]"
        )
        parser.add_argument(
            "--first-exit-xy", type=float, nargs=2, default=(0.030, -0.048), help="First exit on the map [m]"
        )
        parser.add_argument("--stitch-pitch", type=float, default=0.006, help="Distance between stitches [m]")
        parser.add_argument(
            "--eyelet-spacing", type=float, default=0.0015, help="Spacing of the thread's track points [m]"
        )
        parser.add_argument("--air-speed", type=float, default=0.012, help="Tip speed between stitches [m/s]")
        parser.add_argument("--thread-length", type=float, default=0.18, help="Thread length [m]")
        parser.add_argument("--thread-elements", type=int, default=90, help="Thread elements")
        parser.add_argument("--thread-modulus", type=float, default=2.0e6, help="Thread Young's modulus [Pa]")
        parser.add_argument("--updates", type=int, default=2, help="Most Newton updates of the thread per step")
        parser.add_argument("--row-length", type=float, default=0.06, help="Length of the loose thread's rows [m]")
        parser.add_argument(
            "--contact-margin", type=float, default=0.005, help="Largest surface distance with thread contact [m]"
        )
        parser.add_argument(
            "--least-volume",
            type=float,
            default=0.5,
            help="Stop pulling when the end-of-frame check finds a tissue tetrahedron below this fraction "
            "of its rest volume; the coupled solver separately rejects steps that would invert tissue",
        )
        parser.add_argument(
            "--autostart", action=BooleanOptionalAction, default=True, help="Start the row at once (default: on)"
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    # The example as it is run: its graphs captured and its kernel modules loaded before the first frame.
    newton.examples.run(Example(viewer, args, startup=True), args)
