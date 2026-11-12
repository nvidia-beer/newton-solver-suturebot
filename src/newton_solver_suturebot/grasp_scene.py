# SPDX-License-Identifier: Apache-2.0
"""Independent rigid needle, actuated hinged tools, FEM tissue and physical thread.

Only reset places the needle in the bite and locates the fixed trocar frames.
Bounded inverse kinematics supplies finite MuJoCo joint-actuator targets for
shaft pivot/insertion/roll and wrist articulation. Frictional jaw contact holds
the independently free needle. Thread reactions are lagged by one accepted
rigid/FEM interval; rendering never advances these physical subsystems.
"""

import newton
import numpy as np
import warp as wp

from ._src.grasp import GrasperConfig, needle_to_gripper_pose
from ._src.pose_ik import NativePoseIK
from ._src.rcm_grasp import add_rcm_grasper
from ._src.subcycled_mujoco import SubcycledMuJoCo
from ._src.suturebot_layout import load_source_layout, load_source_stitch
from .examples.example_fem_needle_insertion import Example


@wp.kernel(enable_backward=False)
def _needle_weight(needle: int, mass: wp.array[float], force: wp.array[wp.spatial_vector]):
    # The actuated tool links are gravity compensated. The independent needle
    # carries its full weight and receives no compensating support force.
    wp.atomic_add(force, needle, wp.spatial_vector(0.0, 0.0, -9.81 * mass[needle], 0.0, 0.0, 0.0))


def _initial_grasp_parameters(plan):
    """Choose a reset bite for the recorded stitch without changing joint limits.

    Its small oblique bite leaves wrist travel for the complete source arc.
    These are explicit demonstration grasp settings, not measured needle pose
    data. Other material selections retain the original transverse setup.
    """
    result = {"arc_fraction": 0.8, "tangent_roll_rad": -np.pi / 2, "bite_angle_rad": 0.0}
    if plan is not None:
        stitch = load_source_stitch()
        if all(
            point.region_id == stitch["region_id"]
            and np.allclose(point.reference_position, stitch[name], atol=1e-7, rtol=0)
            for point, name in ((plan.entry, "entry"), (plan.exit, "exit"))
        ):
            result.update(tangent_roll_rad=-5 * np.pi / 12, bite_angle_rad=-np.pi / 12)
    return result


class GraspScene(Example):
    """Worker scene; source examples retain their explicit free-needle mode."""

    surface_preconditioner = True

    def configure_builder(self, builder):
        super().configure_builder(builder)
        self.source_layout = load_source_layout()
        self.graspers = [
            add_rcm_grasper(
                builder,
                label=f"PSM{i}",
                config=GrasperConfig(shaft_length_m=0.24),
                nominal_insertion_m=0.18,
                insertion_limits_m=(-0.05, 0.05),
            )
            for i in (1, 2)
        ]

    def _commanded_body(self):
        return self.graspers[0].palm_body

    def _configure_rigid_dynamics(self):
        # The insertion preview supplies the nominal grasp pose after the FEM
        # model is built. Author fixed world anchors in _initialize_pose, then
        # construct native MuJoCo from those frames. They never move in a run.
        self.worker_thread = None
        self.jaw_opening_target = 0.0

    def _build_rigid_solver(self):
        import mujoco

        self.rigid_solver = SubcycledMuJoCo(
            self.model,
            use_mujoco_cpu=True,
            rigid_substeps=4,
            iterations=200,
            ls_iterations=50,
            tolerance=1e-12,
            cone="elliptic",
            integrator="implicitfast",
            impratio=10.0,
            kinematic_switchable_roots=False,
        )
        self.rigid_solver.reset(self.state_0, flags=0)
        self.pose_ik = []
        self.arm_targets = []
        native = self.rigid_solver.mj_model
        for tool in self.graspers:
            joint_ids = [
                mujoco.mj_name2id(
                    native,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    self.model.joint_label[joint].replace("/", "_"),
                )
                for joint in tool.arm_joints
            ]
            body_id = mujoco.mj_name2id(
                native,
                mujoco.mjtObj.mjOBJ_BODY,
                self.model.body_label[tool.palm_body].replace("/", "_"),
            )
            ik = NativePoseIK(native, joint_ids, body_id)
            ik.reset(self.rigid_solver.mj_data.qpos)
            self.pose_ik.append(ik)
            self.arm_targets.append(self.state_0.joint_q.numpy()[list(tool.arm_q_indices)].copy())

    def _initialize_pose(self, pose):
        if self.elapsed > 0:
            raise RuntimeError("A running needle cannot be repositioned; reset the scene")
        pose = np.asarray(pose, dtype=float)
        needle_joints = np.flatnonzero(self.model.joint_child.numpy() == self.needle_body)
        if len(needle_joints) != 1:
            raise ValueError("The needle must have one independent free joint")
        joint = int(needle_joints[0])
        if self.model.joint_type.numpy()[joint] != int(newton.JointType.FREE):
            raise ValueError("The needle must remain a free rigid body")
        coordinates = self.state_0.joint_q.numpy().copy()
        start = int(self.model.joint_q_start.numpy()[joint])
        coordinates[start : start + 7] = pose
        self.state_0.joint_q.assign(coordinates)
        self.state_0.joint_qd.zero_()
        # These world frames are independent of the needle and its reset noise.
        # The registered source ports stay fixed for every insertion and reset.
        for index, tool in enumerate(self.graspers):
            tool.set_trocar_pose(self.model, self.source_layout["trocar_poses"][index], reset=True)
            tool.initialize(self.model, self.state_0, opening_rad=0.7 if index else None)
        self._build_rigid_solver()
        self.parked_target = self.source_layout["parked_pose"].copy()
        targets = [None, self.parked_target]
        self.initial_grasp_parameters = _initial_grasp_parameters(self.plan)
        if self.plan is not None:
            # Reset-only bite placement; physical jaw contact and the measured
            # settled grasp determine all subsequent needle motion.
            relative = needle_to_gripper_pose(self.geometry, **self.initial_grasp_parameters)
            targets[0] = np.asarray(wp.transform_multiply(wp.transform(*pose), wp.transform(*relative)))
        # Reset-only IK places the ready arms. During stepping, the same IK
        # writes actuator targets only; it never writes physical coordinates.
        for index, (tool, ik, target) in enumerate(zip(self.graspers, self.pose_ik, targets, strict=True)):
            if target is not None:
                arm = ik.solve(target, self.rigid_solver.mj_data.qpos)
                tool.initialize(self.model, self.state_0, arm_positions=arm, opening_rad=0.7 if index else None)
        self.rigid_solver.reset(self.state_0, flags=0)
        self.arm_targets = []
        for tool, ik in zip(self.graspers, self.pose_ik, strict=True):
            ik.reset(self.rigid_solver.mj_data.qpos)
            target = self.state_0.joint_q.numpy()[list(tool.arm_q_indices)].copy()
            self.arm_targets.append(target)
            tool.set_arm_targets(self.control, target)
        self.jaw_opening_target = 0.0
        self.graspers[0].set_aperture(self.control, 0.0)
        self.graspers[1].set_aperture(self.control, 0.7 / self.graspers[1].config.maximum_opening_rad)
        self.state_1.assign(self.state_0)
        self.rigid_rest_poses = self.state_0.body_q.numpy().copy()

    def initialize_worker_thread(self, config):
        from ._src.worker_thread import WorkerThread

        self.worker_thread = WorkerThread(self, config)
        self._preload_insertion_modules()

    def _preload_insertion_modules(self):
        """Load known CUDA modules at startup without advancing physical state."""
        from ._src.solvers.fem.differentiation import active_tape

        if (
            not self.device.is_cuda
            or self.coupler is None
            or getattr(self.model, "requires_grad", False)
            or getattr(self.solver, "differentiable", False)
            or active_tape() is not None
        ):
            return
        from ._src import phase_predictor, shaft_preconditioner, tunnel_alm

        modules = []
        if self.coupler._shaft_preconditioner is not None:
            modules.append(shaft_preconditioner)
        if self.interaction.tunnel_alm is not None:
            modules.append(tunnel_alm)
        modules.append(phase_predictor)
        for module in modules:
            # No warmup integration or dummy kernel launch: module loading
            # only moves first-use compilation/loading out of needle entry.
            wp.load_module(module, device=self.device)

    def _prepare_step_loads(self, desired, velocity, dt):
        self._external_forces.zero_()
        wp.launch(
            _needle_weight,
            1,
            [self.needle_body, self.model.body_mass, self.state_0.body_f],
            device=self.device,
        )
        targets = [
            ik.solve(
                desired if index == 0 else self.parked_target,
                self.rigid_solver.mj_data.qpos,
            )
            for index, ik in enumerate(self.pose_ik)
        ]
        if not np.array_equal(targets, self.arm_targets):
            coordinates = self.control.joint_target_q.numpy()
            for assembly, target in zip(self.graspers, targets, strict=True):
                coordinates[list(assembly.arm_q_indices)] = target
            self.control.joint_target_q.assign(coordinates)
            self.arm_targets = targets
        self.graspers[0].set_aperture(
            self.control,
            self.jaw_opening_target / self.graspers[0].config.maximum_opening_rad,
        )
        if self.worker_thread is not None:
            self.worker_thread.add_forces(self.state_0.body_f, self._external_forces)

    def _accepted_step(self, dt):
        if self.worker_thread is not None:
            self.worker_thread.advance(dt)

    def step(self):
        """Align optional force guesses with complete worker control frames."""
        worker = getattr(self, "worker_config", None)
        if (
            not self.running
            or self.coupler is None
            or getattr(worker, "control_mode", None) != "physical_grasp"
            or not all(hasattr(self, name) for name in ("cartesian_target", "jaw_opening_target", "reset_count"))
        ):
            return super().step()
        from ._src.phase_predictor import PhaseLoadPredictor

        predictor = self.coupler._phase_load_predictor
        if predictor is None:
            predictor = PhaseLoadPredictor(self.coupler._load)
            self.coupler._phase_load_predictor = predictor
        predictor.begin_frame(
            frame_dt=self.frame_dt,
            step_dt=self.sim_dt,
            elapsed=self.elapsed,
            command=np.r_[self.cartesian_target, self.jaw_opening_target],
            reset_token=self.reset_count,
        )
        success = False
        try:
            result = super().step()
            success = self.running
            return result
        finally:
            predictor.end_frame(self.elapsed, success)

    def report(self):
        """Describe the actual actuators, rather than inherited needle-drive args."""
        result = super().report()
        transfer = getattr(self.rigid_solver, "_native_transfer", None)
        for key in (
            "drive_stiffness_N_per_m",
            "angular_drive_stiffness_Nm_per_rad",
            "implicit_drive_spring",
        ):
            result.pop(key, None)
        result.update(
            control_mode="physical_grasp",
            drive_body="RCM shaft, insertion, roll and wrist joints",
            shaft_constraint="fixed trocar remote center with bounded axial insertion",
            pose_control="bounded native-model IK to finite joint position actuators",
            ik_diagnostics=[ik.last_diagnostics.copy() for ik in self.pose_ik],
            independent_free_needle=True,
            needle_pose_drive=False,
            rigid_gravity_m_per_s2=9.81,
            tool_gravity_compensation="each tool link weight compensated; needle weight excluded",
            gripper_count=len(self.graspers),
            native_equality_constraint_count=int(self.rigid_solver.mj_model.neq),
            native_rigid_substeps=self.rigid_solver.rigid_substeps,
            native_batches=self.rigid_solver.native_batches,
            native_general_intervals=self.rigid_solver.general_intervals,
            native_transfer_graph_captures=getattr(transfer, "captures", 0),
            native_transfer_graph_launches=getattr(transfer, "launches", 0),
            native_transfer_readbacks=getattr(transfer, "readbacks", 0),
            native_transfer_repeated_readbacks=getattr(transfer, "repeated_readbacks", 0),
            native_transfer_uploads=getattr(transfer, "uploads", 0),
            native_contact_cone="elliptic",
            native_solver_tolerance=float(self.rigid_solver.mj_model.opt.tolerance),
            native_solver_iterations=int(self.rigid_solver.mj_model.opt.iterations),
            native_solver_line_search_iterations=int(self.rigid_solver.mj_model.opt.ls_iterations),
            native_rigid_dt_s=self.sim_dt / self.rigid_solver.rigid_substeps,
            rigid_load_interpolation="external wrenches and actuator targets held over each FEM interval",
            jaw_actuation="independent native revolute position actuators with finite torque limits",
            jaw_stiffness_Nm_per_rad=self.graspers[0].config.jaw_stiffness_nm_rad,
            jaw_damping_Nms_per_rad=self.graspers[0].config.jaw_damping_nms_rad,
            jaw_torque_limit_Nm=self.graspers[0].config.jaw_torque_limit_nm,
            thread_coupling="lagged reactions from accepted physical thread steps; no scheduled eyelets",
            fixture_layout=self.source_layout["provenance"],
            initial_grasp_parameters=self.initial_grasp_parameters.copy(),
            calibrated=False,
        )
        return result

    def observe_grasp(self, *, host_state=None):
        """Share one rigid readback across tool, RCM and thread diagnostics."""
        if host_state is None:
            host_state = {
                name: getattr(self.state_0, name).numpy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")
            }
        result = self.graspers[0].observe(
            self.model, self.state_0, self.rigid_solver, self.needle_body, host_state=host_state
        )
        result.update(
            gripper_target_pose=self.cartesian_target.copy(),
            jaw_target_rad=np.asarray([self.jaw_opening_target / 2, -self.jaw_opening_target / 2]),
            rigid_body_poses=host_state["body_q"].copy(),
            rigid_body_velocities=host_state["body_qd"].copy(),
            arm_joint_positions=np.asarray([host_state["joint_q"][list(tool.arm_q_indices)] for tool in self.graspers]),
            arm_joint_targets=np.asarray(self.arm_targets),
            trocar_poses=np.asarray([tool.trocar_pose for tool in self.graspers]),
            rcm_diagnostics=self._rcm_diagnostics(host_state["body_q"]),
            jaw_contact_count=result["jaw_needle_contact_count"].copy(),
        )
        if self.worker_thread is not None:
            result.update(
                thread_positions=self.worker_thread.positions(),
                thread_radius=self.worker_config.thread_radius_m,
                thread_diagnostics=self.worker_thread.diagnostics(needle_pose=host_state["body_q"][self.needle_body]),
            )
        return result

    def _rcm_diagnostics(self, poses=None):
        poses = self.state_0.body_q.numpy() if poses is None else poses
        result = []
        for tool in self.graspers:
            roll = poses[tool.part_body_indices["roll"]]
            axis = np.asarray(wp.quat_rotate(wp.quat(*roll[3:]), wp.vec3(0, 0, 1)), dtype=float)
            axis /= np.linalg.norm(axis)
            relative = np.asarray(tool.trocar_pose[:3]) - roll[:3]
            axial = float(relative @ axis)
            result.append(
                {
                    "shaft_to_trocar_distance_m": float(np.linalg.norm(np.cross(relative, axis))),
                    "trocar_axial_coordinate_m": axial,
                    "trocar_on_shaft": bool(-0.003 - tool.config.shaft_length_m <= axial <= -0.003),
                }
            )
        return result

    def grasp_geometry(self):
        return {
            "scope": "accepted FEM tissue, independent free MuJoCo needle, two fixed-trocar articulated tools and four actuated jaws; worker ANCF thread with lagged rigid/tissue reactions",
            "fixture_layout": self.source_layout["provenance"],
            "needle_body_index": self.needle_body,
            "needle_visual_asset": str(self.geometry.asset_path),
            "gripper_body_index": self.graspers[0].palm_body,
            "jaw_body_indices": list(self.graspers[0].jaw_bodies),
            "grippers": [
                {
                    "instrument": f"PSM{index + 1}",
                    "instrument_type": ("Large Needle Driver", "DeBakey forceps")[index],
                    "arm_side": ("right", "left")[index],
                    "nominal_insertion_m": tool.nominal_insertion_m,
                    "insertion_limits_m": tool.insertion_limits_m,
                    "palm_body_index": tool.palm_body,
                    "jaw_body_indices": list(tool.jaw_bodies),
                    "part_body_indices": dict(tool.part_body_indices),
                    "trocar_pose": np.asarray(tool.trocar_pose).copy(),
                    "arm_joint_names": [self.model.joint_label[joint] for joint in tool.arm_joints],
                    "capsule_bodies": list(tool.capsule_bodies),
                    "capsule_endpoints": tool.capsule_endpoints.copy(),
                    "capsule_radii": tool.capsule_radii.copy(),
                    "parameters": vars(tool.config),
                }
                for index, tool in enumerate(self.graspers)
            ],
            "rigid_body_labels": list(self.model.body_label),
            "rigid_body_rest_poses": self.rigid_rest_poses.copy(),
            "thread_point_count": len(self.worker_thread.positions()),
            "thread_radius": self.worker_config.thread_radius_m,
            "grasp_physics": "independent free needle; native MuJoCo finite hinge actuation and frictional capsule contacts; fixed-trocar shaft pivot/insertion/roll and wrist joint actuators with bounded IK; no weld or needle pose drive",
        }
