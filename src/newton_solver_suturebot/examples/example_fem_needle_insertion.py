# SPDX-License-Identifier: Apache-2.0
"""Select entry and exit material points, then drive one MuJoCo needle into FEM tissue.

The regional pad, inactive tissue, and fixed interfaces/underside are inherited
from fem_phantom_regions. Needle insertion uses an illustrative sliding track,
not resolved fracture. A finite force/torque drive moves the actual free body;
the desired curve is a reference, never an overwritten simulated trajectory.

Run: python -m newton_solver_suturebot.examples fem_needle_insertion
"""

import json
from argparse import BooleanOptionalAction
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
    SolverMuJoCoFEM,
)
from newton_solver_suturebot.examples._needle_insertion_drive import NeedleDrive, pose_velocity, ramp_distance
from newton_solver_suturebot.examples._needle_insertion_geometry import InsertionPlan, pick_material_point
from newton_solver_suturebot.examples.example_fem_phantom_regions import PAD_COLOR, _scatter_selected
from newton_solver_suturebot.examples.example_fem_phantom_regions import Example as PhantomExample


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


class Example(PhantomExample):
    surface_preconditioner = False

    def __init__(self, viewer, args=None):
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
        self.entry = self.exit = self.plan = self.interaction = None
        self.coupler = None
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

    def start(self):
        if self.plan is None:
            self.preview()
        if self.task_status == "failed" or (not self.manual_control and self.task_status in ("success", "missed")):
            raise ValueError("Insertion has finished; reset before starting again")
        if self.interaction is None:
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
            )
            solver_class = SolverMuJoCoFEM if self.args.needle_solver == "mujoco-fem" else NeedleTissueCoupler
            self.coupler = solver_class(
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
                relaxation=1.0,
                max_iterations=160,
                absolute_tolerance=self.args.coupling_absolute_tolerance,
                relative_tolerance=self.args.coupling_relative_tolerance,
            )
        self.running = True
        self.task_status = "running"
        self.outcome = "Following slider targets" if self.manual_control else "Approaching entry"

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

    def simulate(self):
        if not self.running:
            return
        # Keep the frame's material linearization when retrying a shorter
        # physical interval. A slider jump changes the target, not body poses.
        wp.copy(self._frame_particle_q, self.state_0.particle_q)
        step_dt, remaining = self.sim_dt, self.frame_dt
        # A coarse normal step must retain the small physical intervals needed
        # by abrupt slider changes. Keep the previous fine-step retry range.
        minimum_retry_dt = min(self.sim_dt / 256, 5.0e-6)
        quick_steps = 0
        self.solver.refresh_tangent(self._frame_particle_q, step_dt)
        while remaining > self.frame_dt * 1.0e-10:
            if not self.running:
                break
            self.state_0.clear_forces()
            if self.manual_control:
                desired, position = self._manual_target_pose()
                # A dragged position is a held servo target. Differencing a
                # mouse jump by one physics dt would invent a velocity impulse.
                velocity = np.zeros(6)
            else:
                desired = self._desired_at(self.elapsed + step_dt)
                previous_desired = self._desired_at(self.elapsed)
                velocity = pose_velocity(
                    previous_desired, desired, self.model.body_com.numpy()[self._commanded_body()], step_dt
                )
            self._prepare_step_loads(desired, velocity, step_dt)
            try:
                self.coupler.step(
                    self.state_0, self.state_1, self.control, step_dt, external_forces=self._external_forces
                )
            except (NeedleStepTooLarge, NeedleCouplingNotConverged):
                # The coupler restores the unaccepted trial. Resolve motion
                # without weakening the guard or slowing the slider target.
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
            self._accepted_step(step_dt)
            if self.manual_control:
                self.manual_position = position
            pose_error = self._observe_state(desired)
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
        self.actual_tip_history.append(self.actual_tip())
        try:
            self._check_tissue_integrity()
        except AssertionError as error:
            self._fail(str(error))

    def step(self):
        if self.running:
            try:
                self.simulate()
            except (RuntimeError, ValueError, FloatingPointError) as error:
                self._fail(str(error))

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
        # Follow the deforming material without downloading nodal positions:
        # gather the two barycentric marker points on the device each frame.
        key = (id(self.entry), id(self.exit))
        if getattr(self, "_marker_key", None) != key:
            self._marker_key = key
            nodes, weights = np.zeros((2, 3), np.int32), np.zeros((2, 3), np.float32)
            for row, marker in enumerate((self.entry, self.exit)):
                if marker is not None:
                    nodes[row] = marker.node_indices
                    weights[row] = marker.weights
            self._marker_nodes = wp.array(nodes, dtype=wp.vec3i, device=self.device)
            self._marker_weights = wp.array(weights, dtype=wp.vec3, device=self.device)
            self._marker_points = wp.zeros(2, dtype=wp.vec3, device=self.device)
            self._marker_colors = wp.array([(0.2, 1.0, 0.3), (1.0, 0.25, 0.7)], dtype=wp.vec3, device=self.device)
        if self.state_0.particle_q is not None and (self.entry is not None or self.exit is not None):
            wp.launch(
                _marker_positions,
                2,
                [self._marker_nodes, self._marker_weights, self.state_0.particle_q, self._marker_points],
                device=self.device,
            )
        for row, which in enumerate(("entry", "exit")):
            marker = getattr(self, which)
            self.viewer.log_points(
                f"/insertion/{which}",
                None if marker is None else self._marker_points[row : row + 1],
                radii=0.0005,
                colors=self._marker_colors[row : row + 1],
                hidden=marker is None,
            )
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

    def report(self):
        force_graph = getattr(self.coupler, "_host_trial_forces", None)
        preparation_graph = getattr(self.coupler, "_host_trial_preparation", None)
        phase_predictor = getattr(self.coupler, "_phase_load_predictor", None)
        surface_pc = getattr(self.coupler, "_surface_preconditioner", None)
        return {
            "outcome": self.outcome,
            "success": self.task_status == "success",
            "task_status": self.task_status,
            "needle_solver_requested": self.args.needle_solver,
            "needle_solver_class": None if self.coupler is None else type(self.coupler).__name__,
            "simultaneous_steps": 0 if self.coupler is None else getattr(self.coupler, "simultaneous_steps", 0),
            "simultaneous_iterations": 0
            if self.coupler is None
            else getattr(self.coupler, "simultaneous_iterations", 0),
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
            "track_section_count": 0 if self.interaction is None else self.interaction.track_section_count,
            "track_length_m": 0.0 if self.interaction is None else self.interaction.track_length,
            "maximum_interface_force_N": self.maximum_interface_force,
            "maximum_deformation_m": self.maximum_deformation,
            "maximum_deformation_time_s": self.maximum_deformation_time,
            "maximum_force_balance_error_N": self.maximum_force_balance_error,
            "maximum_torque_balance_error_Nm": self.maximum_torque_balance_error,
            "maximum_coupling_iterations": 0 if self.coupler is None else self.coupler.maximum_iterations,
            "maximum_coupling_residual_N": 0.0 if self.coupler is None else self.coupler.maximum_residual,
            "last_coupling_attempt_iterations": 0 if self.coupler is None else self.coupler.iterations,
            "last_coupling_attempt_residual_N": 0.0 if self.coupler is None else self.coupler.residual,
            "surface_pc_applications": getattr(surface_pc, "applications", 0),
            "surface_pc_selection_changes": getattr(surface_pc, "selection_changes", 0),
            "surface_pc_graph_captures": getattr(surface_pc, "graph_captures", 0),
            "surface_pc_graph_launches": getattr(surface_pc, "graph_launches", 0),
            "surface_pc_native_mass_fallbacks": getattr(surface_pc, "native_mass_fallbacks", 0),
            "fem_graph_captures": 0 if self.coupler is None else self.coupler.fem_graph_captures,
            "fem_graph_launches": 0 if self.coupler is None else self.coupler.fem_graph_launches,
            "fem_graph_status": "not prepared" if self.coupler is None else self.coupler.fem_graph_status,
            "host_force_graph_captures": getattr(force_graph, "captures", 0),
            "host_force_graph_launches": getattr(force_graph, "launches", 0),
            "host_force_graph_fallbacks": getattr(force_graph, "fallbacks", 0),
            "host_force_primal_reuses": getattr(force_graph, "primal_reuses", 0),
            "trial_prepare_graph_captures": getattr(preparation_graph, "captures", 0),
            "trial_prepare_graph_launches": getattr(preparation_graph, "launches", 0),
            "phase_predictor_attempts": getattr(phase_predictor, "attempts", 0),
            "phase_predictor_commits": getattr(phase_predictor, "commits", 0),
            "phase_predictor_invalidations": getattr(phase_predictor, "invalidations", 0),
            "coupling_graph_status": "not prepared" if self.coupler is None else self.coupler.coupling_graph_status,
            "coupling_graph_captures": 0 if self.coupler is None else self.coupler.coupling_graph_captures,
            "coupling_graph_launches": 0 if self.coupler is None else self.coupler.coupling_graph_launches,
            "coupling_graph_fallbacks": 0 if self.coupler is None else self.coupler.coupling_graph_fallbacks,
            "coupling_graph_verifications": 0 if self.coupler is None else self.coupler.coupling_graph_verifications,
            "shaft_preconditioner_status": "not prepared"
            if self.coupler is None
            else self.coupler.shaft_preconditioner_status,
            "coupling_absolute_tolerance_N": self.args.coupling_absolute_tolerance,
            "coupling_relative_tolerance": self.args.coupling_relative_tolerance,
            "coupling_iteration_cap": None if self.coupler is None else self.coupler.max_iterations,
            "fem_iteration_cap": self.args.solver_iterations,
            "fem_relative_tolerance": self.args.fem_tolerance,
            "young_modulus_Pa": self.args.young_modulus,
            "poisson_ratio": self.args.poisson_ratio,
            "density_kg_per_m3": self.args.density,
            "cutting_resistance_omitted": True,
            "last_interface_load_on_fixed_nodes_N": None
            if self.interaction is None
            else self.interaction.support_force.tolist(),
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
            "tunnel_alm_last_normal_error_m": None
            if self.interaction is None or self.interaction.tunnel_alm is None
            else self.interaction.tunnel_alm.normal_error,
            "tunnel_alm_last_dual_residual_N": None
            if self.interaction is None or self.interaction.tunnel_alm is None
            else self.interaction.tunnel_alm.dual_residual,
            "drive_stiffness_N_per_m": self.args.drive_stiffness,
            "angular_drive_stiffness_Nm_per_rad": self.args.angular_stiffness,
            "implicit_drive_spring": True,
            "status": None if self.interaction is None else self.interaction.status,
            "calibrated": False,
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

    def test_final(self):
        self._check_tissue_integrity()
        for array in (self.state_0.body_q, self.state_0.body_qd):
            if not np.isfinite(array.numpy()).all():
                raise AssertionError("MuJoCo needle produced nonfinite state")
        if self.args.report:
            Path(self.args.report).write_text(json.dumps(self.report(), indent=2) + "\n")

    @staticmethod
    def create_parser():
        parser = PhantomExample.create_parser()
        parser.description = __doc__
        parser.set_defaults(substeps=1, solver_iterations=160, regions=[1], young_modulus=25000.0)
        parser.add_argument(
            "--needle-solver",
            choices=("partitioned", "mujoco-fem"),
            default="partitioned",
            help="Needle/tissue integration: existing force iteration or opt-in simultaneous MuJoCo/FEM GPU solve",
        )
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
    newton.examples.run(Example(viewer, args), args)
