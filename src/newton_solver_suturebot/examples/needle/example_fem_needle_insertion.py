# SPDX-License-Identifier: Apache-2.0
"""Select entry and exit material points, then drive one MuJoCo needle into FEM tissue.

The regional pad, inactive tissue, and fixed interfaces/underside are inherited
from fem_phantom_regions. Needle insertion uses an illustrative sliding track,
not resolved fracture. A finite force/torque drive moves the actual free body;
the desired curve is a reference, never an overwritten simulated trajectory.

Run: python -m newton_solver_suturebot.examples fem_needle_insertion
"""

import gc
import json
import time
from argparse import ArgumentTypeError, BooleanOptionalAction
from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo
from newton.viewer import ViewerGL

from newton_solver_suturebot import (
    NeedleCouplingNotConverged,
    NeedleGeometry,
    NeedleStepTooLarge,
    NeedleTissueCoupler,
    NeedleTissueInteraction,
)
from newton_solver_suturebot._src.utils import capture
from newton_solver_suturebot._src.utils.frame_budget import HostTrialBudget, SubstepBound
from newton_solver_suturebot.examples._startup import Script, Startup, first_draw, load_kernel_modules, owns_startup
from newton_solver_suturebot.examples._tissue_markers import TissueMarkers, marker_points
from newton_solver_suturebot.examples.needle._needle_insertion_drive import (
    NeedleDrive,
    _drive,
    pose_velocity,
    ramp_distance,
)
from newton_solver_suturebot.examples.needle._needle_insertion_geometry import InsertionPlan, pick_material_point
from newton_solver_suturebot.examples.phantom.example_fem_phantom_regions import PAD_COLOR, _scatter_selected
from newton_solver_suturebot.examples.phantom.example_fem_phantom_regions import Example as PhantomExample


@wp.kernel(enable_backward=False)
def _gather_region_points(points: wp.array[wp.vec3], indices: wp.array[int], compact: wp.array[wp.vec3]):
    i = wp.tid()
    compact[i] = points[indices[i]]


@wp.kernel(enable_backward=False)
def _marker_positions(
    nodes: wp.array[wp.vec3i], weights: wp.array[wp.vec3], q: wp.array[wp.vec3], points: wp.array[wp.vec3]
):
    i = wp.tid()
    ids, weight = nodes[i], weights[i]
    points[i] = weight[0] * q[ids[0]] + weight[1] * q[ids[1]] + weight[2] * q[ids[2]]


@wp.kernel(enable_backward=False)
def _tissue_integrity(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    reference: wp.array[wp.vec3d],
    tetrahedra: wp.array[wp.vec4i],
    fixed: wp.array[int],
    result: wp.array[wp.float64],
):
    i = wp.tid()
    if i < q.shape[0]:
        for axis in range(3):
            if not wp.isfinite(q[i][axis]) or not wp.isfinite(qd[i][axis]):
                wp.atomic_max(result, 0, wp.float64(1.0))
    if i < fixed.shape[0]:
        node = fixed[i]
        for axis in range(3):
            if q[node][axis] != float(reference[node][axis]):
                wp.atomic_max(result, 1, wp.float64(1.0))
    if i < tetrahedra.shape[0]:
        nodes = tetrahedra[i]
        origin = wp.vec3d(q[nodes[0]])
        a = wp.vec3d(q[nodes[1]]) - origin
        b = wp.vec3d(q[nodes[2]]) - origin
        c = wp.vec3d(q[nodes[3]]) - origin
        rest = reference[nodes[0]]
        ra = reference[nodes[1]] - rest
        rb = reference[nodes[2]] - rest
        rc = reference[nodes[3]] - rest
        jacobian = wp.dot(a, wp.cross(b, c)) / wp.dot(ra, wp.cross(rb, rc))
        wp.atomic_min(result, 2, jacobian)


@wp.kernel(enable_backward=False)
def _state_diagnostics(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    reference: wp.array[wp.vec3d],
    body: int,
    desired_position: wp.vec3d,
    result: wp.array[wp.float64],
):
    i = wp.tid()
    if i < particle_q.shape[0]:
        position, velocity = particle_q[i], particle_qd[i]
        for axis in range(3):
            if not wp.isfinite(position[axis]) or not wp.isfinite(velocity[axis]):
                wp.atomic_max(result, 0, wp.float64(1.0))
        displacement = wp.length(wp.vec3d(position) - reference[i])
        wp.atomic_max(result, 1, displacement)
    if i < body_q.shape[0]:
        body_position = wp.transform_get_translation(body_q[i])
        rotation = wp.transform_get_rotation(body_q[i])
        body_velocity = body_qd[i]
        for axis in range(3):
            if not wp.isfinite(body_position[axis]):
                wp.atomic_max(result, 0, wp.float64(1.0))
        for axis in range(4):
            if not wp.isfinite(rotation[axis]):
                wp.atomic_max(result, 0, wp.float64(1.0))
        for axis in range(6):
            if not wp.isfinite(body_velocity[axis]):
                wp.atomic_max(result, 0, wp.float64(1.0))
        if i == body:
            result[2] = wp.length(wp.vec3d(body_position) - desired_position)


# Trial limit of one coupling loop of this example (NeedleTissueCoupler.max_iterations).
COUPLING_TRIAL_LIMIT = 160


def trial_budget_argument(text):
    """Command-line value of the coupling trial budget: 2 up to below the trial limit; 0 or none for no budget."""
    message = f"the trial budget is an integer from 2 to {COUPLING_TRIAL_LIMIT - 1}, or 0 or none for no budget"
    text = str(text).strip().lower()
    if text == "none":
        return None
    try:
        value = int(text)
    except ValueError:
        raise ArgumentTypeError(message) from None
    if value == 0:
        return None
    if value < 2 or value >= COUPLING_TRIAL_LIMIT:
        raise ArgumentTypeError(message)
    return value


# Kernel modules of the viewer: their loads after start-up are reported apart from those of the physics.
RENDER_MODULE_PREFIXES = ("newton._src.viewer.",)
# What a scene without a start-up mark reports for the capture counters and for the start-up record.
_NO_CAPTURE_COUNTS = dict.fromkeys(("captures_after_startup", "solver_captures_after_startup"))
_NO_STARTUP_RECORD = dict.fromkeys(capture.STARTUP_RECORD)


class Example(PhantomExample):
    surface_preconditioner = False
    preconditioned_anderson = False
    # Census of graph captures and kernel module loads after start-up (utils/capture.py). The driver
    # of a scene opens the start-up, ends it with a mark and says why when it takes none
    # (examples/grasp/parallel.py); a scene that nothing drives that way reports no numbers.
    _startup_begun = None
    _startup_mark = None
    _startup_status = "start-up not marked"
    _capture_census = None
    # (tangent version, time step, accepted steps) of a tangent that a start-up call built for the
    # first frame; simulate() uses it once in place of its own refresh.
    _startup_tangent = None
    # A frame that a call of simulate() with a deadline left unfinished: (time step of its next
    # attempt, rest of the frame, count of quick steps, what the calls so far measured of a
    # substep whose rejected attempt they ran and whose retry is still to come, or None between
    # two substeps). None between frames.
    _pending_frame = None
    # Substeps measured by the calls with a deadline since the last reset (SubstepBound), or None.
    _substep_bound = None
    # (desired pose, manual position) of an attempt that a call left open between two
    # host-controlled trials; its preparation has run. None otherwise.
    _pending_attempt = None
    # Admits the host-controlled trials of a call (HostTrialBudget), from the first call that
    # asks for it to the next reset. None before.
    _host_budget = None
    # The report entries that describe the last attempt, as they were before the open attempt began.
    _report_before_attempt = None
    # Whether the last start-up call of this scene ran (capture_graphs).
    _startup_ready = False
    # True while the constructor runs: the start-up of the construction is open, and a Preview in
    # it opens no start-up of its own.
    _constructing = False
    # The scripted user of ``--test`` leaves a run alone that somebody else drives.
    _script_idle = False

    def __init__(self, viewer, args=None, *, startup=False):
        """Build the scene.

        Args:
            viewer: Newton viewer.
            args: Parsed arguments of :meth:`create_parser`; None for its defaults.
            startup: Run the start-up call of this example (:meth:`capture_graphs`) as soon as the
                planned path is known, at construction where both points are given and otherwise
                at Preview, and at every rebuild after it, between the marks of a start-up
                (``examples/_startup.py``): the run then captures no CUDA graph and loads no
                kernel module in a frame, and ``--test`` fails if it does. The example's entry
                point asks for it. Without it, and in every scene built on this class, the
                graphs are captured where the run first needs them.
        """
        args = newton.examples.default_args(self.create_parser()) if args is None else args
        values = (
            args.insertion_speed,
            args.exit_tolerance,
            args.puncture_force,
            args.contact_stiffness,
            args.track_spacing,
            args.shaft_spacing,
            args.drive_stiffness,
            args.angular_stiffness,
            args.shaft_stiffness,
            args.coupling_absolute_tolerance,
            args.coupling_relative_tolerance,
            args.fem_tolerance,
        )
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Insertion, interaction and drive parameters must be finite and positive")
        self.geometry = NeedleGeometry.load(args.needle_asset)
        if max(args.shaft_spacing, args.track_spacing) > self.geometry.thickness:
            raise ValueError("Shaft and track spacing must not exceed the needle collision radius")
        self.shaft_samples = self.geometry.sample_centerline(args.shaft_spacing)
        self.rigid_solver = None
        self.entry = self.exit = self.plan = self.interaction = None
        self.coupler = None
        self.running = False
        self.pick_mode = "region"
        self.outcome = "Select an entry and exit on the active region."
        self.exit_error = None
        self.actual_tip_history = []
        self.desired_pose = None
        self.elapsed = 0.0
        self.maximum_interface_force = 0.0
        self.maximum_deformation = 0.0
        self._selection_pixels = {}
        self._constructing = True
        if startup and owns_startup(self, Example):
            self._open_startup()
        super().__init__(viewer, args)
        if (args.entry_xy is None) != (args.exit_xy is None):
            raise ValueError("Supply both --entry-xy and --exit-xy")
        if args.entry_xy is not None:
            self.select_point_xy("entry", *args.entry_xy)
            self.select_point_xy("exit", *args.exit_xy)
            self.preview()
            if args.start:
                self.start()
        elif args.start:
            raise ValueError("--start requires --entry-xy and --exit-xy")
        self._constructing = False
        if self.startup is not None and owns_startup(self, Example):
            self._end_startup(frame=True)

    # ---------------------------------------------------------------- start-up

    def _open_startup(self):
        """Take the start-up of this scene: everything from here to the end of the constructor belongs to it.

        Called by the most-derived class, before the scene is built. The base class then opens a
        start-up at every rebuild (``select_regions``) and makes the script of ``--test``.
        """
        self.startup = Startup(wp.get_device())
        self.startup.begin()

    def _finish_startup(self, *, frame):
        """Nothing here: the base class ends its start-up where its own scene is built.

        This scene has more to build after that (the rigid solver, with a planned path the
        coupled step) and ends its start-ups itself, in :meth:`_end_startup`.
        """

    def _load_run_modules(self):
        """Load the kernel modules that only eager code of a frame launches, and those of the tissue solver.

        By kernel, with no launch: the state check of every accepted step, the tissue check of
        every frame, the markers and the compact pad of the drawing, and the needle drive.
        """
        if self.solver is not None:
            self.solver.load_kernel_modules()
        load_kernel_modules(self.device, _state_diagnostics, _scatter_selected, _drive, marker_points)

    def _load_coupling_modules(self):
        """Load the kernel module builds of the coupled step, which do not depend on the planned path.

        The coupled step is made when the path is planned (:meth:`_prepare_run`), because its
        history reserve follows the plan. Its kernel modules do not: a start-up that ends
        without a planned path (the construction without points, a reset, a region switch)
        loads them here, so that the start-up of a Preview, or of a Start without one, captures
        graphs and neither loads nor compiles a kernel module. By name and with no launch:

        - the eager code of a native free-needle step and its step guard
          (``NeedleTissueCoupler._prepare_native_trials``), and the implicit integration of
          the first tissue step of a start-up;
        - the device trials, their free-body interval and the force graphs of the
          host-controlled trials, which a capture otherwise loads with its first launch;
        - Warp's sparse module, which the copy of the projected operator launches;
        - the trial-force helper of the device trials and of the host-controlled force graphs
          (``needle_trial_forces.load_kernel_modules``).

        The correction of the shaft also creates the kernels of its two Krylov solvers, once
        for the device, with two captures that are never launched
        (``shaft_preconditioner.load_kernels``). Nothing is done on a CPU device or without
        tissue.
        """
        if not self.device.is_cuda or self.solver is None:
            return
        from newton_solver_suturebot._src.solvers.coupled import coupling, coupling_graph, host_trial_forces, tet_step
        from newton_solver_suturebot._src.solvers.coupled.acceleration import shaft_preconditioner
        from newton_solver_suturebot._src.solvers.coupled.interface import needle_trial_forces
        from newton_solver_suturebot._src.solvers.fem import implicit_predictor
        from newton_solver_suturebot._src.solvers.mujoco import mujoco_free_body, mujoco_free_body_gpu

        modules = (
            coupling,
            mujoco_free_body,
            tet_step,
            implicit_predictor,
            coupling_graph,
            mujoco_free_body_gpu,
            host_trial_forces,
            "warp.sparse",
        )
        for module in modules:
            wp.load_module(module, device=self.device, block_dim=256)
        needle_trial_forces.load_kernel_modules(self.device)
        shaft_preconditioner.load_kernels(self.device)

    def _end_startup(self, *, frame):
        """End the open start-up: the eager kernel modules loaded, the scene drawn once, then the mark.

        A start-up that ends without a planned path also loads the kernel modules of the
        coupled step (:meth:`_load_coupling_modules`); with a path its start-up call has
        loaded them.

        Args:
            frame: Draw a whole viewer frame where the viewer has no frame limit (the
                constructor). A rebuild or a Preview can come from a panel button, inside the
                viewer's frame: it issues the log calls alone.
        """
        self._load_run_modules()
        if self.interaction is None:
            self._load_coupling_modules()
        first_draw(self.viewer, self.render_pad, self.sim_time, frame=frame)
        self.startup.finish()

    def _prepare_run(self):
        """Build the coupled step of the planned path and run the start-up call, where this scene owns a start-up.

        At the first moment at which the path is known: the history reserve of the interaction
        follows the plan, and the graphs bind that storage. During the construction the open
        start-up covers it; afterwards (Preview, or a Start without one) it is a start-up of
        its own, like a rebuild. A path that already has its coupled step is left alone.
        """
        if self.interaction is not None:
            return
        rebuild = not self._constructing
        if rebuild:
            self.startup.begin()
        self._build_coupling()
        self._startup_status = "start-up not marked"
        self._startup_ready = self.capture_graphs()
        if rebuild:
            self._end_startup(frame=False)

    def configure_builder(self, builder):
        builder.add_usd(str(self.geometry.asset_path), collapse_fixed_joints=False, hide_collision_shapes=True)

    def _capture_physics(self):
        # Puncture and track history must only advance on accepted physical steps.
        self.graph = None

    def begin_grab(self, origin, direction):
        """This task uses map material selections, without spring attachments."""
        return

    def select_regions(self, region_ids):
        super().select_regions(region_ids)
        if not hasattr(self, "_compact_region_meshes"):
            self._compact_region_meshes = []
            for region in self.regions:
                triangles = self.reference.exterior_surface_triangles[
                    self.reference.exterior_surface_region_ids == region.id
                ]
                nodes, indices = np.unique(triangles, return_inverse=True)
                self._compact_region_meshes.append(
                    (
                        region,
                        nodes,
                        wp.array(nodes, dtype=int, device=self.device),
                        wp.array(indices.reshape(-1), dtype=int, device=self.device),
                        wp.empty(len(nodes), dtype=wp.vec3, device=self.device),
                    )
                )
        # Cut-interface nodes are fixed, so their neighboring regions stay at rest.
        free = np.setdiff1d(np.arange(len(self.selection.vertices)), self.fixed_vertex_indices)
        active_nodes = np.searchsorted(self.reference.global_vertex_indices, self.selection.global_vertex_indices[free])
        self._dynamic_region_ids = {
            region.id for region, nodes, _, _, _ in self._compact_region_meshes if np.any(np.isin(nodes, active_nodes))
        }
        # set_model may clear viewer objects; a former active region must also
        # return to its reference positions after changing the active material.
        self._pad_upload_key = None
        # The picked entry and exit ride on their surface triangles: one marker each, bound when it is picked.
        self._pick_markers = [
            (
                which,
                color,
                TissueMarkers(self.viewer, f"/insertion/{which}", radius=0.0005, device=self.device, per_world=1),
            )
            for which, color in (("entry", (0.2, 1.0, 0.3)), ("exit", (1.0, 0.25, 0.7)))
        ]
        self._marker_key = None
        if self.solver is not None:
            # Contact iteration requires a repeatable structural response below
            # the interface residual tolerance, including small nodal masses.
            self.solver.linear_solver_tolerance = self.args.fem_tolerance
            # Chebyshev-accelerated block-Jacobi preconditioning: identical
            # matrix, right-hand side and stopping criterion with fewer, denser
            # Krylov iterations on the conditional CUDA path.
            self.solver.linear_solver_chebyshev_degree = self.args.fem_chebyshev_degree
        self.control = self.model.control()
        self._external_forces = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.device)
        self._frame_particle_q = wp.empty_like(self.model.particle_q)
        self._reference_positions = wp.array(self.selection.vertices, dtype=wp.vec3d, device=self.device)
        self._state_diagnostics = wp.zeros(3, dtype=wp.float64, device=self.device)
        self._integrity_tetrahedra = wp.array(self.selection.indices, dtype=wp.vec4i, device=self.device)
        self._integrity_fixed = wp.array(self.fixed_vertex_indices, dtype=int, device=self.device)
        self._integrity_result = wp.zeros(3, dtype=wp.float64, device=self.device)
        self.needle_body = list(self.model.body_label).index(self.geometry.body_path)
        self._configure_rigid_dynamics()
        self._reset_interaction_state()
        if self.startup is not None and self.screen_pick is not None and owns_startup(self, Example):
            # A rebuild (a region switch, a reset): the base class opened its start-up.
            self._end_startup(frame=False)

    def _configure_rigid_dynamics(self):
        """Build the default free-needle drive; physical grippers override this."""
        if self.model.body_count != 1:
            raise ValueError("Needle insertion needs exactly one free rigid body")
        self.rigid_solver = SolverMuJoCo(
            self.model,
            use_mujoco_cpu=True,
            iterations=30,
            ls_iterations=10,
            integrator="implicitfast",
            # The needle is never made kinematic; the captured trials need a model without the weld for that.
            kinematic_switchable_roots=False,
        )
        self.drive = NeedleDrive(
            self.model,
            self.needle_body,
            stiffness=self.args.drive_stiffness,
            angular_stiffness=self.args.angular_stiffness,
        )
        self.drive.configure_mujoco(self.rigid_solver, self.sim_dt)

    def _reset_interaction_state(self):
        """Reset task bookkeeping after rebuilding the physical model."""
        # An attempt that a call left open ends here, as a rejected attempt ends.
        self.abandon_attempt()
        if self.coupler is not None and self.coupler._shaft_preconditioner is not None:
            self.coupler._shaft_preconditioner.invalidate_preparation()
        self.entry = self.exit = self.plan = self.interaction = None
        self.coupler = None
        # A rebuilt model starts up again: the mark of the previous one no longer applies.
        self._startup_mark = self._capture_census = None
        self._startup_tangent = None
        # A reset in the middle of an unfinished frame drops that frame, and the measurements of
        # the replaced coupler with it.
        self._pending_frame = self._substep_bound = None
        self._host_budget = self._report_before_attempt = None
        self.running, self.elapsed, self.sim_time = False, 0.0, 0.0
        self.manual_control = False
        self.manual_position = np.zeros(2)  # Arc travel from preview, world-Z lift [m].
        self.manual_target = np.zeros(2)
        self.failure_reason = None
        self.step_refinements = self.accepted_physics_steps = 0
        self.minimum_step_dt = self.sim_dt
        self.task_status = "ready"
        self.interaction_events = []
        self.last_exit_error = None
        self.desired_pose, self.exit_error = None, None
        self.actual_tip_history = []
        self.maximum_interface_force = self.maximum_deformation = 0.0
        self.current_deformation = 0.0
        self.maximum_deformation_time = 0.0
        self.maximum_force_balance_error = self.maximum_torque_balance_error = 0.0
        self._selection_pixels = {}
        self.outcome = "Select an entry and exit on the active region."
        self.pick_mode = "region"
        self._initialize_pose(np.array([0.0, 0.0, self.reference.vertices[:, 2].max() + 0.025, 0, 0, 0, 1]))

    def _initialize_pose(self, pose):
        """Place the needle only during reset/preview, before simulation begins."""
        if self.elapsed > 0:
            raise RuntimeError("A running needle cannot be repositioned; reset the example")
        self.state_0.body_q.assign(np.asarray([pose], dtype=np.float32))
        self.state_0.body_qd.zero_()
        newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)
        self.rigid_solver.reset(self.state_0, flags=0)
        self.state_1.assign(self.state_0)

    def select_point(self, which, pixel_x, pixel_y):
        if which not in ("entry", "exit"):
            raise ValueError("Choose entry or exit selection mode")
        if self.elapsed > 0 or self.running:
            raise ValueError("Reset before selecting points on the undeformed pad")
        if not self.selected_region_ids:
            raise ValueError("Select one active region first")
        point = pick_material_point(self.region_map, self.reference, self.selection, pixel_x, pixel_y)
        other = self.exit if which == "entry" else self.entry
        if other is not None and np.linalg.norm(point.reference_position - other.reference_position) < 1.0e-5:
            raise ValueError("Entry and exit coincide; select two different surface points")
        setattr(self, which, point)
        self._selection_pixels[which] = (float(pixel_x), float(pixel_y))
        self.plan = self.interaction = None
        self.desired_pose = None
        self.exit_error = None
        self.outcome = f"Selected {which}. Preview the path when both points are set."
        self.pick_mode = "exit" if which == "entry" else "region"
        return point

    def select_point_xy(self, which, x, y):
        return self.select_point(which, *self.region_map.world_to_pixel(x, y))

    def preview(self):
        if self.elapsed > 0 or self.running or self.manual_control:
            raise ValueError("Reset before planning another insertion")
        if self.entry is None or self.exit is None:
            raise ValueError("Select both entry and exit points")
        plan = InsertionPlan.create(
            self.geometry, self.entry, self.exit, self.selection, self.reference, self.fixed_vertex_indices
        )
        self.plan = plan
        self.manual_control = False
        self.desired_pose = plan.pose(plan.initial_angle)
        self._initialize_pose(self.desired_pose)
        self.outcome = "Path ready. Cyan: desired tip path; yellow: desired needle; silver: actual needle."
        if self.startup is not None and owns_startup(self, Example):
            self._prepare_run()

    def _planned_passages(self):
        """Planned passages of the needle through the tissue, and the material sections of one passage.

        One insertion is one passage. Its plan crosses one section of intact tissue, and one in each
        lip across a wound.
        """
        return 1, self.plan.expected_passages

    def _history_reserve(self):
        """Storage that the interaction reserves for the history of the planned passages.

        Nothing bounds the number of re-entries of a tip, so this is a reserve and not a limit: the
        interaction doubles its storage, and counts that, when a run needs more. Sections: eight for
        every planned one, twice the four that the default insertion with one planned section opens.
        Openings: an entry and an exit for every reserved section.
        Track samples: one needle length of samples at the track spacing for every passage (a
        passage cannot lay more while the tip follows the needle's own arc), and one more for every
        reserved section.
        """
        passages, sections = self._planned_passages()
        reserved_sections = 8 * sections * passages
        needle_length = float(np.linalg.norm(np.diff(self.shaft_samples, axis=0), axis=1).sum())
        samples_per_passage = int(np.floor(needle_length / self.args.track_spacing))
        return {
            "opening_capacity": 2 * reserved_sections,
            "track_capacity": passages * samples_per_passage + reserved_sections,
        }

    def start(self):
        if self.plan is None:
            self.preview()
        if self.task_status == "failed" or (not self.manual_control and self.task_status in ("success", "missed")):
            raise ValueError("Insertion has finished; reset before starting again")
        if self.interaction is None:
            self._build_coupling()
        self.running = True
        self.task_status = "running"
        self.outcome = "Following slider targets" if self.manual_control else "Approaching entry"

    def _build_coupling(self):
        """Make the needle/tissue interaction of the planned path and the coupler of its steps."""
        active_to_reference = np.searchsorted(
            self.reference.global_vertex_indices, self.selection.global_vertex_indices
        )
        reverse = np.full(len(self.reference.vertices), -1, dtype=np.int32)
        reverse[active_to_reference] = np.arange(len(active_to_reference))
        self.interaction = NeedleTissueInteraction(
            self.selection.vertices,
            self.selection.indices,
            self.selection.exterior_surface_triangles,
            self.fixed_vertex_indices,
            self.shaft_samples,
            np.full(len(self.shaft_samples), self.geometry.thickness),
            self.geometry.tip,
            device=self.device,
            stiffness=self.args.contact_stiffness,
            surface_sample_stiffness=self.args.contact_sample_stiffness,
            puncture_force=self.args.puncture_force,
            friction=self.args.shaft_friction,
            shaft_stiffness_per_m=self.args.shaft_stiffness,
            tunnel_model=self.args.tunnel_model,
            tunnel_clearance=self.args.tunnel_clearance,
            tunnel_compliance_per_m=self.args.tunnel_compliance,
            tunnel_penalty_per_m=self.args.tunnel_penalty,
            tunnel_tolerance=self.args.tunnel_tolerance,
            track_spacing=self.args.track_spacing,
            reference_vertices=self.reference.vertices,
            reference_triangles=self.reference.exterior_surface_triangles,
            reference_to_active=reverse,
            particle_inverse_mass=self.model.particle_inv_mass.numpy(),
            needle_mass=float(self.model.body_mass.numpy()[self.needle_body]),
            needle_inertia=self.model.body_inertia.numpy()[self.needle_body],
            **self._history_reserve(),
        )
        self.coupler = NeedleTissueCoupler(
            self.model,
            self.solver,
            self.rigid_solver,
            self.interaction,
            self.needle_body,
            self.model.body_com.numpy()[self.needle_body],
            moment_length=self.geometry.radius,
            capture_fem=True,
            capture_coupling=self.args.capture_coupling,
            shaft_preconditioner=True,
            surface_preconditioner=self.surface_preconditioner,
            preconditioned_anderson=self.preconditioned_anderson,
            relaxation=1.0,
            max_iterations=COUPLING_TRIAL_LIMIT,
            absolute_tolerance=self.args.coupling_absolute_tolerance,
            relative_tolerance=self.args.coupling_relative_tolerance,
            trial_budget=getattr(self.args, "trial_budget", None),
            search_past_puncture=getattr(self.args, "event_search", "tip-path") == "puncture",
            oracle_at_device_trial=getattr(self.args, "event_oracle", "restart") == "device-trial",
            step_trial_budget=getattr(self.args, "event_budget", "loop") == "step",
        )
        self.coupler._host_wrench_enabled = getattr(self, "host_wrench_reuse", False)

    def toggle_running(self):
        if self.running:
            self.running = False
            self.outcome = "Paused"
        else:
            self.start()

    def use_manual_control(self):
        """Take over the desired path without changing the simulated needle pose."""
        if self.task_status == "failed":
            raise ValueError("Simulation failed; reset before moving the needle")
        if self.plan is None:
            self.preview()
        if not self.manual_control:
            distance = ramp_distance(self.elapsed, self.args.insertion_speed)
            maximum = (self.plan.finish_angle - self.plan.initial_angle) * self.geometry.radius
            self.manual_position[:] = min(distance, maximum), 0.0
            self.manual_target[:] = self.manual_position
            self.manual_control = True
            self.exit_error = None
        if not self.running:
            self.start()

    def set_manual_target(self, *, insertion=None, lift=None):
        """Set the drive's arc/lift target directly; never overwrite physical state."""
        if self.plan is None:
            raise ValueError("Preview an entry/exit path before using the sliders")
        maximum = (self.plan.finish_angle - self.plan.initial_angle) * self.geometry.radius
        if insertion is not None and (not np.isfinite(insertion) or not 0 <= insertion <= maximum):
            raise ValueError("Insertion target must be finite and within the previewed arc")
        if lift is not None and (not np.isfinite(lift) or abs(lift) > 0.005):
            raise ValueError("Lift target must be finite and within +/-5 mm")
        self.use_manual_control()
        if insertion is not None:
            self.manual_target[0] = insertion
        if lift is not None:
            self.manual_target[1] = lift

    def _manual_target_pose(self):
        # The slider is the drive target, with no extra slew-rate controller.
        # MuJoCo advances the actual body under the finite PD and tissue forces.
        position = self.manual_target.copy()
        angle = self.plan.initial_angle + position[0] / self.geometry.radius
        desired = self.plan.pose(angle)
        desired[2] += position[1]
        return desired, position

    def _desired_at(self, time):
        distance = ramp_distance(time, self.args.insertion_speed)
        angle = min(self.plan.initial_angle + distance / self.geometry.radius, self.plan.finish_angle)
        return self.plan.pose(angle)

    def actual_tip(self):
        pose = self.state_0.body_q.numpy()[self.needle_body]
        return np.asarray(wp.transform_point(wp.transform(*pose), wp.vec3(*self.geometry.tip)), dtype=float)

    def frame_insertion(self):
        if self.entry is None or self.exit is None:
            return
        points = [self.entry.reference_position, self.exit.reference_position]
        if self.plan is not None:
            for angle in np.linspace(self.plan.initial_angle, self.plan.finish_angle, 9):
                matrix = self.plan.matrix(angle)
                points.extend(self.geometry.centerline @ matrix[:3, :3].T + matrix[:3, 3])
            axis = self.plan.axis.copy()
        else:
            axis = np.cross(self.exit.reference_position - self.entry.reference_position, (0, 0, 1))
        if np.linalg.norm(axis) < 1.0e-8:
            axis = np.array((0.0, 1.0, 0.0))
        axis /= np.linalg.norm(axis)
        if np.dot(axis, (0.0, 1.0, 0.2)) < 0:
            axis = -axis
        view_offset = axis + np.array((0.0, 0.0, 0.65))
        view_offset[2] = max(view_offset[2], 0.1)
        points = np.asarray(points)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        radius = max(float(np.linalg.norm(points - center, axis=1).max()), self.geometry.radius)
        distance = 3.4 * radius
        offset = distance * view_offset / np.linalg.norm(view_offset)
        direction = -offset / np.linalg.norm(offset)
        right = np.cross(direction, [0, 0, 1])
        right /= np.linalg.norm(right)
        target = center - 0.12 * distance * right
        self.viewer.set_camera(
            pos=tuple(target + offset),
            pitch=float(np.degrees(np.arcsin(direction[2]))),
            yaw=float(np.degrees(np.arctan2(direction[1], direction[0]))),
        )
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "set_pivot"):
            camera.set_pivot(target)

    def _fail(self, reason):
        self.running = False
        self.failure_reason = reason
        self.task_status = "failed"
        self.outcome = f"Failed: {reason}"
        if self.exit is not None:
            self.exit_error = float(
                np.linalg.norm(self.actual_tip() - self.exit.position(self.state_0.particle_q.numpy()))
            )

    def _observe_interaction(self):
        """Interpret generic physical events for this entry/exit task only."""
        for event in self.interaction.events:
            self.interaction_events.append(
                {
                    "kind": event["kind"],
                    "time_s": self.elapsed,
                    "position_m": np.asarray(event["position"], dtype=float).tolist(),
                    "reused": bool(event.get("reused", False)),
                }
            )
            if event["kind"] == "exit":
                target = self.exit.position(self.state_0.particle_q)
                self.last_exit_error = float(np.linalg.norm(self.actual_tip() - target))
                if not self.manual_control and self.last_exit_error <= self.args.exit_tolerance:
                    self.exit_error = self.last_exit_error
                    self.running = False
                    self.task_status = "success"
                    self.outcome = f"Success: tip emerged, exit error {self.exit_error * 1000:.3f} mm"

    def _commanded_body(self):
        """Rigid body whose pose is commanded by this scene."""
        return self.needle_body

    def _observe_state(self, desired):
        """Check every accepted state while transferring only compact diagnostics."""
        self._state_diagnostics.zero_()
        wp.launch(
            _state_diagnostics,
            max(self.model.particle_count, self.model.body_count),
            [
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self.state_0.body_q,
                self.state_0.body_qd,
                self._reference_positions,
                self._commanded_body(),
                wp.vec3d(*desired[:3]),
                self._state_diagnostics,
            ],
            device=self.device,
        )
        invalid, displacement, pose_error = self._state_diagnostics.numpy()
        if invalid:
            self._fail("nonfinite state; reduce timestep or interaction stiffness")
            return None
        self.current_deformation = float(displacement)
        if displacement > self.maximum_deformation:
            self.maximum_deformation = float(displacement)
            self.maximum_deformation_time = self.elapsed
        return float(pose_error)

    def _prepare_step_loads(self, desired, velocity, dt):
        """External loads held throughout a transactional needle/FEM trial."""
        self._external_forces.zero_()
        self.drive.prepare_timestep(dt)
        self.drive.add_forces(self.state_0, desired, velocity)

    def _accepted_step(self, dt):
        """Advance optional physical subsystems only after needle/FEM acceptance."""

    def capture_graphs(self):
        """Start-up call of a scene: capture, before the first frame, graphs that the run would capture.

        The driver of a scene calls it at the end of a reset (examples/grasp/parallel.py) and
        checks the captures and module loads after start-up only for a scene that returns True.
        A scene built on this class has no start-up call here, and neither has this example
        unless it was asked for one (``startup=True``): their graphs are captured where the run
        first needs them.

        The example that owns its start-up calls it when the planned path has its coupled step
        (:meth:`_prepare_run`), with the scene at its start state. The call

        - builds the tangent of the first frame (:meth:`simulate` then uses it instead of its
          own refresh, once) and the projected operator of that tangent;
        - loads the kernel modules of the FEM solver and those that only eager code of a frame
          launches (:meth:`_load_run_modules`);
        - for both accepted-state buffers (``NeedleTissueCoupler.capture_graphs`` with
          ``native=True``) captures and launches the trial preparation of the host-controlled
          trials, their FEM step at every inner tolerance, the trial graph of the device trials
          (for an empty track, with the correction of both values of the shaft flag), the force
          graphs of the host-controlled trials and the graphs of their correction.

        The first two run on every device; the captures need a CUDA device with conditional
        graphs. They run real steps and discard them: the native integration state and every
        accepted array get back the bits they had, no time passes, no step is accepted and no
        history changes. Work counters keep their values; capture counters include these captures.

        Returns:
            True if the call ran. False if the scene has no start-up call or the coupled step is
            not captured; ``_startup_status`` then holds the reason (on a CPU device, where the
            tangent is built and nothing is captured: "CPU device").
        """
        if self.startup is None:
            self._startup_status = "native rigid backend"
            return False
        from newton_solver_suturebot._src.solvers.fem.differentiation import active_tape

        reason = None
        if self.coupler is None:
            reason = "no coupler"
        elif (
            getattr(self.model, "requires_grad", False)
            or getattr(self.solver, "differentiable", False)
            or active_tape() is not None
        ):
            reason = "differentiable configuration"
        elif self.elapsed > 0 or self.accepted_physics_steps:
            reason = "the scene has advanced since its reset"
        if reason is not None:
            self._startup_status = reason
            return False
        if self.device.is_cuda:
            # A rebuilt scene leaves the cycles of the one before it, which own device memory and
            # graphs: collected here, not in a frame.
            gc.collect()
        # The positions, the time step and the code path of the first frame's own refresh.
        wp.copy(self._frame_particle_q, self.state_0.particle_q)
        self.solver.refresh_tangent(self._frame_particle_q, self.sim_dt)
        self._startup_tangent = (self.solver._tangent_version, self.sim_dt, self.accepted_physics_steps)
        self.coupler._refresh_projection()
        self._load_run_modules()
        # Both buffers hold the start state: every captured step below starts from an accepted state.
        self.state_1.assign(self.state_0)
        buffers = (self.state_0, self.state_1)
        if not self.coupler.capture_graphs(buffers, self.control, self.sim_dt, native=True):
            self._startup_status = (
                f"coupled step not captured: {self.coupler.fem_graph_status}" if self.device.is_cuda else "CPU device"
            )
            return False
        self._startup_status = "start-up call ran"
        return True

    @property
    def frame_pending(self):
        """Whether a call with a deadline left the current frame unfinished (see :meth:`simulate`)."""
        return self._pending_frame is not None

    def _trial_limit(self):
        """Largest number of trials one loop of the coupler evaluates."""
        budget = self.coupler.trial_budget
        # A loop that the budget ends applies its best load once more unless that was its last trial.
        return self.coupler.max_iterations if budget is None else budget + 1

    def _substep_mark(self, spent=None):
        """What the host knows where a substep starts: the clock and the coupler's trials, loops and loop time.

        Args:
            spent: What earlier calls measured of the substep that this call goes on with (its
                rejected attempts), or None at the start of a substep. The mark is set back by
                it: the substep is then measured as one, without the time between the calls.
        """
        coupler = self.coupler
        mark = (time.perf_counter(), coupler.evaluations, coupler.trial_loops, coupler.loop_seconds)
        return mark if spent is None else tuple(now - before for now, before in zip(mark, spent, strict=True))

    def _substep_measured(self, mark):
        """Count the substep that started at ``mark`` and ends now; returns the mark of the next one."""
        if self._substep_bound is None:
            self._substep_bound = SubstepBound()
        now = self._substep_mark()
        self._substep_bound.add(now[1] - mark[1], now[2] - mark[2], now[3] - mark[3], now[0] - mark[0])
        return now

    def _wait_for_device(self):
        """Wait until the device has finished what was issued; nothing crosses to the host.

        A call with a deadline measures a substep on the host clock, and what a substep issues
        without waiting for it is part of that substep: the hook of an accepted step (the thread
        step of a scene with a thread), the tangent of a retry and the start of an attempt that
        the call leaves open. Waited for here, that time is
        in the substep, in the bound of the next one (:class:`SubstepBound`) and in the tail of
        its attempt (:class:`HostTrialBudget`). Left for the caller, the first read of the
        observation that follows the call waits for it, and the caller plans for the longest
        observation in every deadline: the same time would be planned for twice.
        """
        wp.synchronize_device(self.device)

    def _attempt_fits(self, now, deadline):
        """The rule of a call with a deadline: whether one more attempt may start at the host time ``now`` [s].

        It may if a substep that uses up its trials, in as many trial loops as the substep
        measured last ran, ends by the deadline. Nothing measured yet is no bound: it may not.
        """
        bound = self._substep_bound
        return bound is not None and now + bound.bound(self._trial_limit()) <= deadline

    def abandon_attempt(self):
        """End an attempt that a call left open, as a rejected attempt ends; nothing without one.

        If the coupler's abandon raises, the error is the caller's and the scene still forgets
        the attempt: the coupler has none open then, and a later call must not go on with it.
        """
        coupler = getattr(self, "coupler", None)
        try:
            if coupler is not None and getattr(coupler, "attempt_open", False):
                coupler.abandon_step()
        finally:
            self._pending_attempt = None

    def simulate(self, deadline=None, *, inside=False):
        """Advance the frame: all of it, or with a ``deadline`` as many of its attempts as are sure to end by then.

        An attempt is one coupled step, accepted or rejected as a whole; a rejected attempt is
        tried again at half the time step. A substep is an accepted step with the rejected
        attempts before it.

        Args:
            deadline: A time of the host clock ``time.perf_counter()`` [s], or None (default) for
                the whole frame. With a deadline the call runs one attempt in any case and
                starts another one, the next substep or the retry of a rejected attempt, only
                if a substep that uses up its trials, in as many trial loops as the last
                measured substep ran (the bound of :class:`SubstepBound`, from the substeps
                this scene has measured on the host clock), ends before the deadline.
                Otherwise it returns with the frame unfinished (:attr:`frame_pending`), and the
                next call, with or without a deadline, goes on with the same frame at its next
                attempt: the attempted steps are those of one call, and so is every accepted
                state. What starts a frame (the linearization positions, the tangent) is done
                by the call that starts it, and what ends it (the tip sample, the tissue check)
                by the call that finishes it. A rejected attempt halves the time step and
                builds the tangent for it before the call returns, as it does before a retry in
                the same call: an unfinished frame always holds the tangent of its next
                attempt. A substep ends when the device has finished what it issued: before
                a substep is measured, before the question for a retry and before it leaves
                an attempt open, the call waits for the accepted-step hook, for the tangent
                of the retry and for what the open attempt has issued
                (:meth:`_wait_for_device`), so a call that returns leaves nothing for the
                caller's observation to wait for. Without ``inside`` a call never leaves an
                attempt that it started with
                nothing open: one that takes longer than the bound (a trial loop handed over to
                the host-controlled loop) ends after the deadline. A call that goes on with an
                attempt that the call before left open is the exception (see ``inside``).
            inside: Whether the call may also return inside an attempt, between two
                host-controlled trials (the unit ``host-trial``). Before every such trial the
                coupler asks the scene's :class:`HostTrialBudget` whether the longest
                host-controlled trial and the longest end of an attempt that this scene has
                measured still end by the deadline; if not, the call returns with the attempt
                open (the frame is pending) and the next call, with or without ``inside`` or a
                deadline, goes on with that trial as its first piece. The preparation of the
                attempt has run and is not run again; the accepted state, and the report
                entries that describe the last attempt, are those of before the attempt until
                it ends. A device trial loop is not interrupted. A call that goes on with an
                open attempt asks the same questions for the rest of the call, with or without
                ``inside``: given a deadline it may leave that attempt again, or a later
                attempt of the call, between two host-controlled trials. Without a deadline it
                finishes the frame.
        """
        if not self.running:
            return
        pending, self._pending_frame = self._pending_frame, None
        resume, self._pending_attempt = self._pending_attempt, None
        spent = None
        if pending is None:
            # Keep the frame's material linearization when retrying a shorter
            # physical interval. A slider jump changes the target, not body poses.
            wp.copy(self._frame_particle_q, self.state_0.particle_q)
            step_dt, remaining = self.sim_dt, self.frame_dt
            quick_steps = 0
            # A start-up call may have built this frame's tangent: at these positions (no step was
            # accepted since), at this time step, and nothing has written the tangent since. Its
            # token serves one frame.
            prepared, self._startup_tangent = self._startup_tangent, None
            if prepared is None or prepared != (self.solver._tangent_version, step_dt, self.accepted_physics_steps):
                self.solver.refresh_tangent(self._frame_particle_q, step_dt)
        else:
            # The unfinished frame goes on where it stopped, with its linearization and its tangent.
            step_dt, remaining, quick_steps, spent = pending
        # A coarse normal step must retain the small physical intervals needed
        # by abrupt slider changes. Keep the previous fine-step retry range.
        minimum_retry_dt = min(self.sim_dt / 256, 5.0e-6)
        substep = None if deadline is None else self._substep_mark(spent)
        budget = None
        if inside or resume is not None:
            # Host-controlled trials may be left, or the call before left one: this call goes on with it.
            if self._host_budget is None:
                self._host_budget = HostTrialBudget()
            budget = self._host_budget
            budget.begin_call(deadline, resumed=resume is not None)
        while remaining > self.frame_dt * 1.0e-10:
            if not self.running:
                break
            if resume is None:
                self.state_0.clear_forces()
                if self.manual_control:
                    desired, position = self._manual_target_pose()
                    # A dragged position is a held servo target. Differencing a
                    # mouse jump by one physics dt would invent a velocity impulse.
                    velocity = np.zeros(6)
                else:
                    position = None
                    desired = self._desired_at(self.elapsed + step_dt)
                    previous_desired = self._desired_at(self.elapsed)
                    velocity = pose_velocity(
                        previous_desired, desired, self.model.body_com.numpy()[self._commanded_body()], step_dt
                    )
                self._prepare_step_loads(desired, velocity, step_dt)
                if budget is not None:
                    self._report_before_attempt = self._attempt_report()
            else:
                # The attempt that the call before left open: its preparation has run.
                desired, position = resume
            try:
                if budget is None:
                    self.coupler.step(
                        self.state_0, self.state_1, self.control, step_dt, external_forces=self._external_forces
                    )
                else:
                    if resume is None:
                        ended = self.coupler.begin_step(
                            self.state_0,
                            self.state_1,
                            self.control,
                            step_dt,
                            external_forces=self._external_forces,
                            pieces=budget,
                        )
                    else:
                        resume = None
                        ended = self.coupler.resume_step()
                    if not ended:
                        # Inside an attempt, before a host-controlled trial that does not fit: the
                        # next call goes on with that trial and measures the substep on from what
                        # this call spent on it (with what the attempt has issued so far).
                        if substep is not None:
                            self._wait_for_device()
                            now = self._substep_mark()
                            spent = tuple(after - before for after, before in zip(now, substep, strict=True))
                        self._pending_frame = (step_dt, remaining, quick_steps, spent)
                        self._pending_attempt = (desired, position)
                        return
            except (NeedleStepTooLarge, NeedleCouplingNotConverged):
                if budget is not None:
                    # The host-controlled trial that raised, with the close of its attempt.
                    budget.close()
                # The coupler restores the unaccepted trial. Resolve motion
                # without weakening the guard or slowing the slider target.
                if step_dt <= minimum_retry_dt:
                    raise
                step_dt *= 0.5
                quick_steps = 0
                self.step_refinements += 1
                self.solver.refresh_tangent(self._frame_particle_q, step_dt)
                if deadline is not None:
                    # Between a rejected attempt and its retry: the retry starts in this call
                    # only if a substep that uses up its trials ends by the deadline. The
                    # coupler has restored the accepted state, the time step is halved and
                    # its tangent is built (waited for: it is part of what this call spent);
                    # the next call starts with the retry and measures the substep on from
                    # what this call spent on it.
                    self._wait_for_device()
                    now = self._substep_mark()
                    if not self._attempt_fits(now[0], deadline):
                        spent = tuple(after - before for after, before in zip(now, substep, strict=True))
                        self._pending_frame = (step_dt, remaining, quick_steps, spent)
                        return
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
            if self.manual_control:
                self.manual_position = position
            # The state check reads a few bytes back, which waits for everything issued
            # before it. Issue it first: the device work of the accepted-step hook then
            # runs while the host prepares the next interval instead of being waited for
            # (a call with a deadline waits for it where the substep ends, below).
            pose_error = self._observe_state(desired)
            self._accepted_step(step_dt)
            if pose_error is None:
                break
            if self.interaction.status == "failed":
                self._fail(self.interaction.reason)
                break
            self.outcome = (
                "Tip outside tissue" if self.interaction.status == "outside" else self.interaction.status.capitalize()
            )
            self.maximum_interface_force = max(
                self.maximum_interface_force, float(np.linalg.norm(self.interaction.interface_force))
            )
            self._observe_interaction()
            max_time = (self.plan.finish_angle - self.plan.initial_angle) * self.geometry.radius
            max_time = max_time / self.args.insertion_speed + 1.0
            if not self.manual_control and self.elapsed > max_time and self.running:
                if self.last_exit_error is None:
                    self._fail("tip did not emerge before the drive stopped; possible stalled insertion")
                else:
                    self.running = False
                    self.task_status = "missed"
                    self.exit_error = self.last_exit_error
                    self.outcome = f"Missed target: last emergence error {self.exit_error * 1000:.3f} mm"
            # A distant manual target is an intentional command, not evidence
            # that automatic insertion has lost its planned trajectory.
            if not self.manual_control and pose_error > 0.005 and self.running:
                self._fail("needle deviated more than 5 mm from the finite-force drive")
            # Recover after a resolved transient instead of retaining its tiny
            # interval for the whole frame. Larger attempts still pass both
            # the unchanged force convergence and swept-motion checks.
            quick_steps = quick_steps + 1 if self.coupler.iterations <= 4 else 0
            if self.running and remaining > self.frame_dt * 1.0e-10:
                next_dt = min(step_dt, remaining)
                if quick_steps >= 2:
                    next_dt = min(2 * step_dt, self.sim_dt, remaining)
                    quick_steps = 0
                if next_dt != step_dt:
                    step_dt = next_dt
                    self.solver.refresh_tangent(self._frame_particle_q, step_dt)
                if deadline is not None:
                    # The substep ends when the device has finished what it issued above (the
                    # accepted-step hook): before its attempt's tail and the substep are measured.
                    self._wait_for_device()
                if budget is not None:
                    # The tail of the attempt (checks, commit, the work after the acceptance) ends here.
                    budget.close()
                if deadline is not None:
                    # Between two substeps: the one that ended is measured, and the next one is
                    # started only if a substep that uses up its trials ends by the deadline.
                    substep = self._substep_measured(substep)
                    if not self._attempt_fits(substep[0], deadline):
                        self._pending_frame = (step_dt, remaining, quick_steps, None)
                        return
        self.actual_tip_history.append(self.actual_tip())
        try:
            self._check_tissue_integrity()
        except AssertionError as error:
            self._fail(str(error))
        if budget is not None:
            # The tail of the frame's last attempt, with what ends the frame.
            budget.close()
        if deadline is not None:
            # The last substep of the frame, with what ends the frame.
            self._substep_measured(substep)

    def step(self, deadline=None, *, inside=False):
        """Advance one frame; with a ``deadline`` the frame may stay unfinished (see :meth:`simulate`).

        ``inside``: the call may also return inside an attempt, between two host-controlled trials.
        """
        if self.script is not None and self._pending_frame is None:
            # The scripted user of --test acts between two frames.
            self.script.play()
        if self.running:
            try:
                if inside:
                    self.simulate(deadline, inside=True)
                elif deadline is None:
                    self.simulate()
                else:
                    self.simulate(deadline)
            except (RuntimeError, ValueError, FloatingPointError) as error:
                self._fail(str(error))
        else:
            # A scene that has stopped does not go on with a frame it left unfinished, and an
            # attempt it left open ends as a rejected attempt ends.
            self._pending_frame = None
            self.abandon_attempt()

    def _lines(self, name, points, color):
        if points is None or len(points) < 2:
            self.viewer.log_lines(name, None, None, color, hidden=True)
            return
        self.viewer.log_lines(
            name,
            wp.array(np.asarray(points[:-1]), dtype=wp.vec3, device=self.device),
            wp.array(np.asarray(points[1:]), dtype=wp.vec3, device=self.device),
            color,
        )

    def render_pad(self):
        """Draw the pad, the needle and the insertion markers inside a viewer frame."""
        if self.solver is not None:
            wp.launch(
                _scatter_selected,
                dim=self.model.particle_count,
                inputs=[self.state_0.particle_q, self._active_to_display, self.display_points],
                device=self.device,
            )
        self.viewer.log_state(self.state_0)
        opacity = 0.55 if self.args.transparent_pad else 1.0
        upload_key = (self.selection_revision, opacity)
        refresh_all = self._pad_upload_key != upload_key
        for region, _, source_nodes, indices, points in self._compact_region_meshes:
            dynamic = region.id in self._dynamic_region_ids
            if refresh_all or dynamic:
                wp.launch(
                    _gather_region_points, len(points), [self.display_points, source_nodes, points], device=self.device
                )
            if not refresh_all and not dynamic and isinstance(self.viewer, ViewerGL):
                continue
            self.viewer.log_mesh(
                f"/phantom/region_{region.id:02d}",
                points,
                indices,
                color=PAD_COLOR,
                roughness=0.75,
                backface_culling=False,
                opacity=opacity,
            )
        self._pad_upload_key = upload_key
        # Follow the deforming material without downloading nodal positions: the picked entry and
        # exit are bound to their surface triangles when the picks change and drawn at those
        # material points on the device each frame (examples/_tissue_markers.py).
        key = (id(self.entry), id(self.exit))
        if self._marker_key != key:
            self._marker_key = key
            for which, color, markers in self._pick_markers:
                point = getattr(self, which)
                markers.clear()
                if point is not None:
                    markers.attach(point.node_indices, point.weights, color)
        for _, _, markers in self._pick_markers:
            markers.draw(self.state_0.particle_q)
        self._lines("/insertion/desired_tip", None if self.plan is None else self.plan.path_points, (0.1, 0.9, 1.0))
        desired_points = None
        if self.desired_pose is not None:
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*self.desired_pose[3:]))).reshape(3, 3)
            desired_points = self.geometry.centerline @ rotation.T + self.desired_pose[:3]
        self._lines("/insertion/desired_needle", desired_points, (1.0, 0.8, 0.1))
        self._lines("/insertion/actual_tip", self.actual_tip_history, (1.0, 0.45, 0.1))

    def _action(self, function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except (ValueError, RuntimeError) as error:
            self.outcome = f"Failed: {self.failure_reason}. Reset to try again." if self.failure_reason else str(error)

    def gui(self, ui):
        ui.text("Needle insertion | MuJoCo + FEM")
        ui.text_wrapped(self.outcome)
        ui.text(f"Time: {self.elapsed:.3f} s | tolerance: {1000 * self.args.exit_tolerance:.2f} mm")
        if self.interaction is not None:
            ui.text(f"Tissue displacement: {1000 * self.current_deformation:.3f} mm")
            ui.text(f"Shaft engagement: {1000 * self.interaction.engaged_track_length:.2f} mm")
        if self.exit_error is not None:
            ui.text(f"Exit error: {1000 * self.exit_error:.3f} mm")
        for mode, label in (("region", "Region"), ("entry", "Entry"), ("exit", "Exit")):
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
                self._action(self.select_point, self.pick_mode, *self.region_map.clicked_pixel)
        if self.region_map.screen_rect is not None:
            ox, oy, width, height = self.region_map.screen_rect
            draw = ui.get_window_draw_list()
            for name, (px, py) in self._selection_pixels.items():
                color = 0xFF4DFF33 if name == "entry" else 0xFFB340FF
                position = ui.ImVec2(
                    ox + px * width / self.region_map.region_ids.shape[1],
                    oy + py * height / self.region_map.region_ids.shape[0],
                )
                draw.add_circle_filled(position, 4.0, color)
                draw.add_text(ui.ImVec2(position.x + 6, position.y - 6), color, "E" if name == "entry" else "X")
        if ui.button("Preview path"):
            self._action(self.preview)
        ui.same_line()
        label = "Resume sliders" if self.manual_control else "Start automatic"
        if ui.button("Pause" if self.running else label):
            self._action(self.toggle_running)
        if self.plan is not None:
            maximum = (self.plan.finish_angle - self.plan.initial_angle) * self.geometry.radius
            target = (
                self.manual_target
                if self.manual_control
                else np.array((min(ramp_distance(self.elapsed, self.args.insertion_speed), maximum), 0.0))
            )
            changed, insertion = ui.slider_float("Insert/retract", 1000 * target[0], 0.0, 1000 * maximum, "%.2f mm")
            if changed:
                self._action(self.set_manual_target, insertion=float(np.clip(insertion / 1000, 0, maximum)))
            changed, lift = ui.slider_float("Lift/lower", 1000 * target[1], -5.0, 5.0, "%.2f mm")
            if changed:
                self._action(self.set_manual_target, lift=lift / 1000)
            if self.manual_control:
                ui.text(
                    f"Drive: {1000 * self.manual_position[0]:.2f} mm in, {1000 * self.manual_position[1]:+.2f} mm lift"
                )
            ui.text_wrapped(
                "Drag sliders to move the needle. Tissue forces affect actual motion. Reset restores automatic insertion."
            )
        ui.text_wrapped("Space toggles the viewer's global pause.")
        if ui.button("Reset"):
            self.reset()
        ui.same_line()
        if ui.button("Frame pad"):
            self.reset_camera()
        ui.same_line()
        if ui.button("Frame insertion"):
            self.frame_insertion()
        _, self.args.transparent_pad = ui.checkbox("See through pad", self.args.transparent_pad)
        ui.text(f"FEM: {len(self.selection.indices):,} tetrahedra")
        ui.text(f"Fixed nodes: {len(self.fixed_vertex_indices):,}")
        if self.interaction is not None:
            ui.text(f"Punctures: {self.interaction.puncture_count} | track samples: {self.interaction.track_count}")
            ui.text(f"Entries: {self.interaction.entry_count} | exits: {self.interaction.exit_count}")
            ui.text(f"Interface force: {1000 * np.linalg.norm(self.interaction.interface_force):.2f} mN")
            ui.text(f"Interface load on fixed nodes: {1000 * np.linalg.norm(self.interaction.support_force):.2f} mN")
        ui.text_wrapped(
            "Green entry / pink exit follow tissue. Cyan path and yellow needle are desired motion; silver needle is simulated."
        )
        ui.separator()
        ui.text_wrapped(
            "Illustrative puncture, sliding friction and drive gains. No resolved fracture, cavity, thread or physiological calibration."
        )
        ui.text_wrapped("Cutting resistance omitted.")
        ui.text(f"Needle: R={1000 * self.geometry.radius:.2f} mm, mass={1000 * self.geometry.mass:.2f} g")

    def refresh_capture_census(self):
        """Walk the loaded kernel modules once and keep what the start-up mark does not hold.

        ``report()`` returns these values until the next call; it never walks the modules itself.
        Without a mark nothing is kept. An example that owns its start-up keeps the census of all
        its start-ups (``Startup.census``).
        """
        startup = getattr(self, "startup", None)
        if startup is not None:
            self._capture_census = startup.census()
            return
        mark = self._startup_mark
        self._capture_census = None if mark is None else capture.modules(mark, RENDER_MODULE_PREFIXES)

    def _census_report(self):
        """Graph captures and kernel module loads after the start-up mark; None for each without a mark.

        The two capture numbers are differences of host counters, taken at every call. The module
        numbers and the tables are those of the last ``refresh_capture_census()``. For an example
        that owns its start-up every number is of that last refresh, added up over its start-ups.
        """
        startup = getattr(self, "startup", None)
        if startup is not None:
            census = self._capture_census or {}
            names = (*capture.STARTUP_RECORD, "graphs_released_after_startup", "late_sites", "late_modules")
            return {
                "captures_after_startup": census.get("captures_after_startup"),
                "solver_captures_after_startup": census.get("solver_captures_after_startup"),
                "module_loads_after_startup": census.get("module_loads_after_startup"),
                "render_module_loads_after_startup": census.get("render_module_loads_after_startup"),
                "capture_census": {
                    "status": census.get("capture_census_status", startup.status),
                    "startups": startup.startups,
                    **{name: census.get(name) for name in names},
                    "late_render_modules": census.get("late_render_modules"),
                },
            }
        mark, late = self._startup_mark, self._capture_census or {}
        if mark is None:
            numbers, startup, status = _NO_CAPTURE_COUNTS, _NO_STARTUP_RECORD, self._startup_status
        else:
            numbers, startup, status = capture.counts(mark), mark.startup, "checked"
        return {
            **numbers,
            "module_loads_after_startup": late.get("module_loads_after_startup"),
            "render_module_loads_after_startup": late.get("render_module_loads_after_startup"),
            "capture_census": {
                "status": status,
                **startup,
                "graphs_released_after_startup": late.get("graphs_released_after_startup"),
                "late_sites": late.get("late_sites"),
                "late_modules": late.get("late_modules"),
                "late_render_modules": late.get("late_render_modules"),
            },
        }

    # Report entries that describe the last attempt of the coupler (_attempt_report builds them).
    _ATTEMPT_REPORT = (
        "last_coupling_attempt_iterations",
        "last_coupling_attempt_residual_N",
        "fem_graph_status",
        "coupling_graph_status",
        "shaft_preconditioner_status",
        "last_interface_load_on_fixed_nodes_N",
        "tunnel_alm_last_normal_error_m",
        "tunnel_alm_last_dual_residual_N",
        "status",
    )

    def _attempt_report(self):
        """The report entries that describe the last attempt, as :meth:`_needle_report` gives them.

        Taken before every attempt of a call that may leave it open, so only these entries are
        built, each a host value. The whole report is not: once the needle is in the tissue its
        engaged track length is read from the device.
        """
        coupler, interaction = self.coupler, self.interaction
        multipliers = None if interaction is None else interaction.tunnel_alm
        return {
            "last_coupling_attempt_iterations": 0 if coupler is None else coupler.iterations,
            "last_coupling_attempt_residual_N": 0.0 if coupler is None else coupler.residual,
            "fem_graph_status": "not prepared" if coupler is None else coupler.fem_graph_status,
            "coupling_graph_status": "not prepared" if coupler is None else coupler.coupling_graph_status,
            "shaft_preconditioner_status": "not prepared" if coupler is None else coupler.shaft_preconditioner_status,
            "last_interface_load_on_fixed_nodes_N": None if interaction is None else interaction.support_force.tolist(),
            "tunnel_alm_last_normal_error_m": None if multipliers is None else multipliers.normal_error,
            "tunnel_alm_last_dual_residual_N": None if multipliers is None else multipliers.dual_residual,
            "status": None if interaction is None else interaction.status,
        }

    def report(self):
        result = self._needle_report()
        if self._pending_attempt is not None and self._report_before_attempt is not None:
            # An attempt is open: these entries are those of the state the observation shows.
            result.update(self._report_before_attempt)
        return result

    def _needle_report(self):
        # The nine entries that describe the last attempt have one source (_ATTEMPT_REPORT).
        attempt = self._attempt_report()
        force_graph = getattr(self.coupler, "_host_trial_forces", None)
        preparation_graph = getattr(self.coupler, "_host_trial_preparation", None)
        phase_predictor = getattr(self.coupler, "_phase_load_predictor", None)
        surface_pc = getattr(self.coupler, "_surface_preconditioner", None)
        return {
            "outcome": self.outcome,
            "success": self.task_status == "success",
            "task_status": self.task_status,
            "control_mode": "sliders" if self.manual_control else "automatic",
            "manual_position_m": self.manual_position.tolist(),
            "manual_target_m": self.manual_target.tolist(),
            "step_refinements": self.step_refinements,
            "accepted_physics_steps": self.accepted_physics_steps,
            "minimum_step_dt_s": self.minimum_step_dt,
            "current_deformation_m": self.current_deformation,
            "engaged_track_length_m": 0.0 if self.interaction is None else self.interaction.engaged_track_length,
            "failure_reason": self.failure_reason,
            "interaction_events": self.interaction_events,
            "entry_count": 0 if self.interaction is None else self.interaction.entry_count,
            "exit_count": 0 if self.interaction is None else self.interaction.exit_count,
            "physical_time_s": self.elapsed,
            "exit_error_m": self.exit_error,
            "exit_tolerance_m": self.args.exit_tolerance,
            "puncture_count": 0 if self.interaction is None else self.interaction.puncture_count,
            "track_samples": 0 if self.interaction is None else self.interaction.track_count,
            "track_capacity": 0 if self.interaction is None else self.interaction.track_capacity,
            "track_storage_growths": 0 if self.interaction is None else self.interaction.track_storage_growths,
            "cooperative_shaft_rebinds": getattr(self.interaction, "_cooperative_shaft_rebinds", 0),
            "openings": 0 if self.interaction is None else self.interaction.opening_count,
            "opening_capacity": 0 if self.interaction is None else self.interaction.opening_capacity,
            "opening_storage_growths": 0 if self.interaction is None else self.interaction.opening_storage_growths,
            "track_section_count": 0 if self.interaction is None else self.interaction.track_section_count,
            "track_length_m": 0.0 if self.interaction is None else self.interaction.track_length,
            "maximum_interface_force_N": self.maximum_interface_force,
            "maximum_deformation_m": self.maximum_deformation,
            "maximum_deformation_time_s": self.maximum_deformation_time,
            "maximum_force_balance_error_N": self.maximum_force_balance_error,
            "maximum_torque_balance_error_Nm": self.maximum_torque_balance_error,
            "maximum_coupling_iterations": 0 if self.coupler is None else self.coupler.maximum_iterations,
            "maximum_coupling_residual_N": 0.0 if self.coupler is None else self.coupler.maximum_residual,
            "last_coupling_attempt_iterations": attempt["last_coupling_attempt_iterations"],
            "last_coupling_attempt_residual_N": attempt["last_coupling_attempt_residual_N"],
            "surface_pc_applications": getattr(surface_pc, "applications", 0),
            "surface_pc_selection_changes": getattr(surface_pc, "selection_changes", 0),
            "surface_pc_graph_captures": getattr(surface_pc, "graph_captures", 0),
            "surface_pc_graph_launches": getattr(surface_pc, "graph_launches", 0),
            "surface_pc_native_mass_fallbacks": getattr(surface_pc, "native_mass_fallbacks", 0),
            "fem_graph_captures": 0 if self.coupler is None else self.coupler.fem_graph_captures,
            "fem_graph_launches": 0 if self.coupler is None else self.coupler.fem_graph_launches,
            "fem_graph_status": attempt["fem_graph_status"],
            "host_force_graph_captures": getattr(force_graph, "captures", 0),
            "host_force_graph_launches": getattr(force_graph, "launches", 0),
            "host_force_graph_fallbacks": getattr(force_graph, "fallbacks", 0),
            "host_force_primal_reuses": getattr(force_graph, "primal_reuses", 0),
            "trial_prepare_graph_captures": getattr(preparation_graph, "captures", 0),
            "trial_prepare_graph_launches": getattr(preparation_graph, "launches", 0),
            "phase_predictor_attempts": getattr(phase_predictor, "attempts", 0),
            "phase_predictor_commits": getattr(phase_predictor, "commits", 0),
            "phase_predictor_invalidations": getattr(phase_predictor, "invalidations", 0),
            "phase_predictor_rejections": getattr(phase_predictor, "rejections", 0),
            "coupling_graph_status": attempt["coupling_graph_status"],
            "coupling_graph_captures": 0 if self.coupler is None else self.coupler.coupling_graph_captures,
            "coupling_graph_launches": 0 if self.coupler is None else self.coupler.coupling_graph_launches,
            "coupling_graph_fallbacks": 0 if self.coupler is None else self.coupler.coupling_graph_fallbacks,
            "coupling_graph_event_searches": 0
            if self.coupler is None
            else getattr(self.coupler, "coupling_graph_event_searches", 0),
            "coupling_graph_event_restarts": 0
            if self.coupler is None
            else getattr(self.coupler, "coupling_graph_event_restarts", 0),
            "coupling_divergence_stops": 0
            if self.coupler is None
            else getattr(self.coupler, "coupling_divergence_stops", 0),
            "coupling_budget_keeps": 0 if self.coupler is None else getattr(self.coupler, "coupling_budget_keeps", 0),
            "coupling_budget_searches": 0
            if self.coupler is None
            else getattr(self.coupler, "coupling_budget_searches", 0),
            "coupling_budget_residual": 0.0
            if self.coupler is None
            else getattr(self.coupler, "coupling_budget_residual", 0.0),
            "coupling_budget_events": {}
            if self.coupler is None
            else dict(getattr(self.coupler, "coupling_budget_events", {})),
            "coupling_graph_fallback_reasons": {}
            if self.coupler is None
            else dict(getattr(self.coupler, "coupling_graph_fallback_reasons", {})),
            "coupling_graph_verifications": 0 if self.coupler is None else self.coupler.coupling_graph_verifications,
            "shaft_preconditioner_status": attempt["shaft_preconditioner_status"],
            "coupling_absolute_tolerance_N": self.args.coupling_absolute_tolerance,
            "coupling_relative_tolerance": self.args.coupling_relative_tolerance,
            "coupling_iteration_cap": None if self.coupler is None else self.coupler.max_iterations,
            "coupling_trial_budget": None if self.coupler is None else getattr(self.coupler, "trial_budget", None),
            "fem_iteration_cap": self.args.solver_iterations,
            "fem_relative_tolerance": self.args.fem_tolerance,
            "young_modulus_Pa": self.args.young_modulus,
            "poisson_ratio": self.args.poisson_ratio,
            "density_kg_per_m3": self.args.density,
            "cutting_resistance_omitted": True,
            "last_interface_load_on_fixed_nodes_N": attempt["last_interface_load_on_fixed_nodes_N"],
            "dt_s": self.sim_dt,
            "track_spacing_m": self.args.track_spacing,
            "shaft_spacing_m": self.args.shaft_spacing,
            "shaft_stiffness_N_per_m2": self.args.shaft_stiffness,
            "tunnel_model": self.args.tunnel_model,
            "tunnel_clearance_m": self.args.tunnel_clearance,
            "tunnel_compliance_m2_per_N": self.args.tunnel_compliance
            if self.args.tunnel_compliance is not None
            else 1.0 / self.args.shaft_stiffness,
            "tunnel_penalty_N_per_m2": self.args.tunnel_penalty
            if self.args.tunnel_penalty is not None
            else self.args.shaft_stiffness,
            "tunnel_tolerance_m": self.args.tunnel_tolerance,
            "tunnel_alm_dual_updates": 0
            if self.interaction is None or self.interaction.tunnel_alm is None
            else self.interaction.tunnel_alm.total_dual_updates,
            "tunnel_alm_last_normal_error_m": attempt["tunnel_alm_last_normal_error_m"],
            "tunnel_alm_last_dual_residual_N": attempt["tunnel_alm_last_dual_residual_N"],
            "drive_stiffness_N_per_m": self.args.drive_stiffness,
            "angular_drive_stiffness_Nm_per_rad": self.args.angular_stiffness,
            "implicit_drive_spring": True,
            "status": attempt["status"],
            "calibrated": False,
            **self._census_report(),
        }

    def _check_tissue_integrity(self):
        """Preserve regional checks without copying the entire mesh each frame."""
        if self.solver is None:
            return
        self._integrity_result.assign(np.array([0.0, 0.0, np.inf]))
        wp.launch(
            _tissue_integrity,
            max(self.model.particle_count, len(self.selection.indices), len(self.fixed_vertex_indices)),
            [
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self._reference_positions,
                self._integrity_tetrahedra,
                self._integrity_fixed,
                self._integrity_result,
            ],
            device=self.device,
        )
        invalid, fixed_moved, minimum_j = self._integrity_result.numpy()
        if invalid:
            raise AssertionError("Regional FEM produced nonfinite state")
        if minimum_j <= 0:
            raise AssertionError(f"Regional FEM inverted an element: minimum J={minimum_j:.6g}")
        if fixed_moved:
            raise AssertionError("Fixed cut/underside vertices moved")

    # Entry and exit [m] of the two insertions of the self-test: the documented pair, whose path stays
    # in the tissue, and the pair whose path crosses a gap.
    _SCRIPT_POINTS = (((0.030, -0.040), (0.038, -0.040)), ((0.030, -0.040), (0.030, -0.048)))

    def _scripted(self, action):
        """A scripted action that does nothing in a run that somebody else drives."""

        def run():
            if not self._script_idle:
                action()

        return run

    def _scripted_points(self, entry, exit_point):
        """Self-test: the two map clicks of an insertion, on a scene that nobody else has started to use."""
        if self.entry is not None or self.exit is not None or self.running or self.elapsed > 0:
            # Points of the command line, or the scenario of a recorder: this run is theirs.
            self._script_idle = True
            return
        self.select_point_xy("entry", *entry)
        self.select_point_xy("exit", *exit_point)

    def _scripted_reset(self):
        """Self-test: keep how the insertion ended, then Reset."""
        self._script_runs.append(
            {
                "task_status": self.task_status,
                "entries": 0 if self.interaction is None else self.interaction.entry_count,
                "exits": 0 if self.interaction is None else self.interaction.exit_count,
                "exit_error_m": self.exit_error,
            }
        )
        self.reset()

    def _test_script(self):
        """The scripted user of ``--test``, in 365 frames.

        Entry and exit on the map, Preview, Start, the whole automatic insertion with its exit,
        Reset, another entry and exit, Preview and a second Start, whose needle goes into the
        tissue. With points on the command line the script is empty: the run is the one that
        was asked for.
        """
        self._script_runs = []
        script = Script()
        if self.args.entry_xy is not None:
            return script
        first, second = self._SCRIPT_POINTS
        script.at(10, "entry and exit on the map", lambda: self._scripted_points(*first))
        script.at(12, "Preview path", self._scripted(self.preview))
        script.at(15, "Start automatic", self._scripted(self.toggle_running))
        script.at(265, "Reset", self._scripted(self._scripted_reset))
        script.at(268, "entry and exit on the map", self._scripted(lambda: self._scripted_points(*second)))
        script.at(270, "Preview path", self._scripted(self.preview))
        script.at(273, "Start automatic", self._scripted(self.toggle_running))
        script.rest(91)
        return script

    def _check_script(self):
        """Fail if the scripted insertions of a complete self-test did not go as scripted."""
        if self._script_idle or not self.script.complete or not self._script_runs:
            return
        first = self._script_runs[0]
        if first["task_status"] != "success" or first["entries"] < 1 or first["exits"] < 1:
            raise AssertionError(f"The scripted insertion did not end with its exit before the Reset: {first}")
        entered = 0 if self.interaction is None else self.interaction.entry_count
        if self.task_status == "failed" or entered < 1:
            raise AssertionError(
                f"The second scripted insertion did not enter the tissue: {self.task_status}, {entered} entries"
            )

    def _check_startup(self):
        """The check of a self-test: nothing captured or loaded after a start-up, and the reserves held.

        Raises AssertionError on a CUDA graph capture, a Krylov solve outside a graph or a kernel
        module load after a start-up mark, on a start-up graph that was never launched, and on a
        growth of the track or opening storage (a growth captures the graphs of the step anew).
        Prints the numbers of the check.
        """
        self.refresh_capture_census()
        numbers = self.startup.check()
        growths = {}
        if self.interaction is not None:
            growths = {
                "track_storage_growths": self.interaction.track_storage_growths,
                "opening_storage_growths": self.interaction.opening_storage_growths,
            }
            if any(growths.values()):
                raise AssertionError(f"The history reserve of the interaction was too small: {growths}")
        self._check_script()
        print(
            json.dumps(
                {
                    **self.script.summary(),
                    "scripted_insertions": self._script_runs,
                    "startup_ready": self._startup_ready,
                    "startup_status": self._startup_status,
                    **growths,
                    **numbers,
                }
            )
        )

    def test_final(self):
        self._check_tissue_integrity()
        for array in (self.state_0.body_q, self.state_0.body_qd):
            if not np.isfinite(array.numpy()).all():
                raise AssertionError("MuJoCo needle produced nonfinite state")
        if self.startup is not None and self.script is not None:
            # The self-test of an example that owns its start-up.
            self._check_startup()
        if self.args.report:
            self.refresh_capture_census()
            Path(self.args.report).write_text(json.dumps(self.report(), indent=2) + "\n")

    @staticmethod
    def create_parser():
        parser = PhantomExample.create_parser()
        parser.description = __doc__
        parser.set_defaults(substeps=1, solver_iterations=160, regions=[1], young_modulus=25000.0)
        parser.add_argument(
            "--needle-asset", default=None, help="Prepared standalone needle USD; default bundled needle"
        )
        parser.add_argument("--entry-xy", type=float, nargs=2, help="Undeformed entry world XY [m]")
        parser.add_argument("--exit-xy", type=float, nargs=2, help="Undeformed exit world XY [m]")
        parser.add_argument("--start", action="store_true", help="Start a specified valid entry/exit pair immediately")
        parser.add_argument(
            "--insertion-speed", type=float, default=0.003, help="Desired arc speed [m/s], illustrative"
        )
        parser.add_argument(
            "--exit-tolerance", type=float, default=0.001, help="Success tolerance [m], declared before running"
        )
        parser.add_argument("--puncture-force", type=float, default=0.003, help="Illustrative puncture threshold [N]")
        parser.add_argument(
            "--contact-stiffness", type=float, default=20.0, help="Illustrative surface stiffness [N/m]"
        )
        parser.add_argument(
            "--contact-sample-stiffness",
            type=float,
            default=None,
            help="Least surface stiffness of one shaft sample [N/m]; default shares --contact-stiffness over the shaft",
        )
        parser.add_argument("--shaft-friction", type=float, default=0.2, help="Illustrative shaft friction coefficient")
        parser.add_argument(
            "--shaft-stiffness", type=float, default=60000.0, help="Illustrative transverse shaft bearing [N/m^2]"
        )
        parser.add_argument(
            "--track-spacing", type=float, default=0.00025, help="Material track quadrature spacing [m]"
        )
        parser.add_argument("--tunnel-model", choices=("penalty", "alm"), default="alm")
        parser.add_argument("--tunnel-clearance", type=float, default=2e-5, help="ALM radial clearance [m]")
        parser.add_argument(
            "--tunnel-compliance",
            type=float,
            default=1.0 / 300000.0,
            help="ALM physical radial compliance [m^2/N], independent of numerical penalty; zero requests a hard wall",
        )
        parser.add_argument(
            "--tunnel-penalty", type=float, default=None, help="ALM numerical penalty [N/m^2]; default shaft stiffness"
        )
        parser.add_argument("--tunnel-tolerance", type=float, default=2e-6, help="ALM normal residual tolerance [m]")
        parser.add_argument(
            "--shaft-spacing", type=float, default=0.00014, help="Maximum shaft contact sample spacing [m]"
        )
        parser.add_argument("--drive-stiffness", type=float, default=400.0, help="COM drive stiffness [N/m]")
        parser.add_argument("--angular-stiffness", type=float, default=0.05, help="Rotational drive stiffness [Nm/rad]")
        parser.add_argument(
            "--coupling-absolute-tolerance", type=float, default=2e-6, help="Absolute interface force tolerance [N]"
        )
        parser.add_argument(
            "--coupling-relative-tolerance", type=float, default=1e-3, help="Relative interface force tolerance"
        )
        parser.add_argument(
            "--trial-budget",
            type=trial_budget_argument,
            default=None,
            help=f"Trial budget of the needle/tissue coupling, from 2 to {COUPLING_TRIAL_LIMIT - 1} (below the limit "
            f"of {COUPLING_TRIAL_LIMIT} trials): a trial loop that has evaluated this many trials without converging "
            "keeps its best evaluated trial instead of going on to the limit (NeedleTissueCoupler.trial_budget). "
            "0 or none (default): no budget, a step converges to the coupling tolerances or is rejected",
        )
        parser.add_argument(
            "--event-search",
            choices=("tip-path", "puncture"),
            default="tip-path",
            help="Surface-event requests that the device trials search past before the host oracle decides the "
            "event: tip-path (default), the requests about the path of the tip; puncture, also a contact force at "
            "the puncture threshold (NeedleTissueCoupler.search_past_puncture)",
        )
        parser.add_argument(
            "--event-oracle",
            choices=("restart", "device-trial"),
            default="restart",
            help="First trial of the host oracle after a device search: restart (default), a complete host trial "
            "from the load of the search; device-trial, the host evaluation of the last device trial as it stands "
            "(NeedleTissueCoupler.oracle_at_device_trial)",
        )
        parser.add_argument(
            "--event-budget",
            choices=("loop", "step"),
            default="loop",
            help="Trial budget of a step whose device search hands a surface event to the host oracle: loop "
            "(default), the oracle loop has a budget of its own; step, the trials of the search count in the "
            "oracle loop, one budget for the step (NeedleTissueCoupler.step_trial_budget)",
        )
        parser.add_argument(
            "--response-basis-nodes",
            type=int,
            default=0,
            help="Support nodes whose response columns the response-basis guess solves in one coupled step when "
            "the track has grown (AffineGuessManager.nodes_per_attempt); the remaining columns follow in the "
            "next steps and the guess is withheld until they are solved. 0 (default): every missing column in "
            "the step that follows the growth. Read by scenes that configure a response-basis guess",
        )
        parser.add_argument("--fem-tolerance", type=float, default=3e-3, help="Relative FEM linear residual tolerance")
        parser.add_argument(
            "--fem-chebyshev-degree",
            type=int,
            default=4,
            help="Chebyshev acceleration degree of the FEM block-Jacobi preconditioner; 0 disables",
        )
        parser.add_argument(
            "--capture-coupling",
            action=BooleanOptionalAction,
            default=True,
            help="Capture eligible repeated needle/FEM trials on CUDA; surface events retain exact fallback",
        )
        parser.add_argument("--transparent-pad", action="store_true", help="Render pad translucent for inspection")
        parser.add_argument("--report", help="Write final diagnostic JSON, with --test")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    # The example as it is run: its graphs captured and its kernel modules loaded before the first frame.
    newton.examples.run(Example(viewer, args, startup=True), args)
