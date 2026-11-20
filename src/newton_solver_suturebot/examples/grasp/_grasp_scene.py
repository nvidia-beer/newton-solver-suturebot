# SPDX-License-Identifier: Apache-2.0
"""Independent rigid needle, actuated hinged tools, FEM tissue and physical thread.

Only reset places the needle in the bite and locates the fixed trocar frames.
Bounded inverse kinematics supplies finite MuJoCo joint-actuator targets for
shaft pivot/insertion/roll and wrist articulation. Frictional jaw contact holds
the independently free needle. Thread reactions are lagged by one accepted
rigid/FEM interval; rendering never advances these physical subsystems.
With ``--tool-tissue-contact`` the tool capsules press on the tissue surface
and are pushed back by it, also one accepted interval late; without it (the
default) tools and tissue exchange no force. ``--tool-tissue-contact swept``
selects the contact on the sweep of the capsules instead of the first-version
law: the tissue nodes that a capsule reaches in the next interval are placed on
it and held there, and tissue that is held or cannot follow stops the tool.
The thread follows the needle through the tissue: the track that the needle
cuts becomes eyelets of the thread, threaded where the needle's tail has passed
(``--no-thread-tracks`` leaves the thread on the surface).
"""

import argparse
import gc
import os

import newton
import numpy as np
import warp as wp

from ..._src.sim.ik.pose_ik import NativePoseIK
from ..._src.solvers.mujoco.experimental_rigid_backend import is_gpu_backend, supported_subcycled
from ..._src.solvers.mujoco.subcycled_mujoco import SubcycledMuJoCo
from ..._src.solvers.mujoco.subcycled_mujoco_warp import SubcycledMuJoCoWarp
from ..needle.example_fem_needle_insertion import Example
from ._dvrk_grasper import GrasperConfig, jaw_opening_on_needle, needle_to_gripper_pose
from ._dvrk_layout import load_source_layout, load_source_stitch
from ._dvrk_rcm_grasper import add_rcm_grasper

# Gravity of the needle's weight [m/s²]: the factor that _needle_weight applies to the needle's mass in every
# step. The scene publishes the weight itself from it (grasp_geometry, needle_weight_N).
NEEDLE_GRAVITY = 9.81


@wp.kernel(enable_backward=False)
def _needle_weight(needle: int, mass: wp.array[float], force: wp.array[wp.spatial_vector]):
    # The actuated tool links are gravity compensated. The independent needle
    # carries its full weight and receives no compensating support force.
    wp.atomic_add(force, needle, wp.spatial_vector(0.0, 0.0, -9.81 * mass[needle], 0.0, 0.0, 0.0))


@wp.kernel(enable_backward=False)
def _ramp_arm_targets(
    previous: wp.array[float],
    target: wp.array[float],
    fraction: float,
    indices: wp.array[int],
    joint_target: wp.array[float],
):
    # Actuator target of one arm joint on the straight line between the targets of two frames.
    joint = wp.tid()
    joint_target[indices[joint]] = previous[joint] + fraction * (target[joint] - previous[joint])


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


def add_dvrk_tools(builder):
    """Add the two fixed-trocar dVRK tools, PSM1 and PSM2, to ``builder``; returns their assemblies in that order."""
    return [
        add_rcm_grasper(
            builder,
            label=f"PSM{i}",
            config=GrasperConfig(shaft_length_m=0.24),
            nominal_insertion_m=0.18,
            insertion_limits_m=(-0.05, 0.05),
        )
        for i in (1, 2)
    ]


def tool_constraint_rows(model):
    """Constraint rows that the rigid model of the tools can need at once: the capacity of MuJoCo Warp (``njmax``).

    A wire between the jaws of one tool lies across the capsules that stand side by side
    along each blade (``JAW_BLADE_ROWS`` per jaw). Where two pieces of a blade's chain meet,
    it can touch both, and where two of its own segments meet, both can touch a capsule: 2 x
    2 contacts per row and jaw. A contact of the tool capsules has six rows (elliptic cone
    with torsional and rolling friction, condim 6). Every joint coordinate can add the row of
    a limit. With the first capsule set of the tools (one capsule per jaw, 12 rows in the
    grasp) the default of MuJoCo Warp, 64 rows, was enough.
    """
    from ._dvrk_capsules import JAW_BLADE_ROWS

    return 6 * (2 * JAW_BLADE_ROWS * 2 * 2) + int(model.joint_dof_count)


def tool_rigid_solver(model, rigid_backend, **options):
    """Rigid solver of the dVRK tools: MuJoCo Warp on the device ("mujoco-warp") or native MuJoCo on the CPU.

    Four microsteps per coupled interval with held loads and actuator targets, elliptic friction
    cones and the implicitfast integrator. The grasp scene and the tool/tissue example build
    their solver here, so both step the same tools. On the device the constraint rows are
    those of :func:`tool_constraint_rows`; native MuJoCo sizes its own. ``options`` are further
    arguments of the solver class of the backend.
    """
    experimental_gpu = rigid_backend == "mujoco-warp"
    rigid_class = SubcycledMuJoCoWarp if experimental_gpu else SubcycledMuJoCo
    if experimental_gpu:
        options.setdefault("njmax", tool_constraint_rows(model))
    return rigid_class(
        model,
        use_mujoco_cpu=not experimental_gpu,
        rigid_substeps=4,
        iterations=200,
        ls_iterations=50,
        tolerance=1e-12,
        cone="elliptic",
        integrator="implicitfast",
        impratio=10.0,
        kinematic_switchable_roots=False,
        **options,
    )


def tool_pose_ik(model, rigid_solver, graspers):
    """Bounded pose inverse kinematics of every tool's palm on the native model of ``rigid_solver``, reset to its state."""
    import mujoco

    native = rigid_solver.mj_model
    result = []
    for tool in graspers:
        joint_ids = [
            mujoco.mj_name2id(
                native,
                mujoco.mjtObj.mjOBJ_JOINT,
                model.joint_label[joint].replace("/", "_"),
            )
            for joint in tool.arm_joints
        ]
        body_id = mujoco.mj_name2id(
            native,
            mujoco.mjtObj.mjOBJ_BODY,
            model.body_label[tool.palm_body].replace("/", "_"),
        )
        ik = NativePoseIK(native, joint_ids, body_id)
        ik.reset(rigid_solver.mj_data.qpos)
        result.append(ik)
    return result


def tool_effective_masses(rigid_solver, model, state, bodies, endpoints, radii, contact_radii, capsule_tools):
    """Per tool: its smallest effective mass [kg] at an end of one of its capsules, every actuator free.

    ``1 / lambda_max(J M^-1 J^T)`` with the translational Jacobian ``J`` of the point and the
    joint-space inertia ``M`` of the native MuJoCo model: the lightest direction of the
    lightest point of the tool, so that a stiffness derived from it is below the bound of
    every point and direction of that tool. The points are the first and the last probe of
    every capsule (``capsule_pieces``). One rule for every scene with tools
    (:func:`placed_tool_effective_masses`). A host computation on native data of its own, which
    get the joint coordinates of ``state`` by the rigid solver's own conversion (the solver's
    data hold a placed state only after its first interval, or after its import on the device
    backend). The body positions of those data are compared with ``state.body_q``; a
    difference of more than 10 um is refused.

    Args:
        rigid_solver: Rigid solver of ``model`` with the native model (``mj_model``).
        model: Newton model of the tools.
        state: State with the joint coordinates and body poses of the configuration.
        bodies: Body of each capsule.
        endpoints: End points of each capsule axis in the body frame [m], shape (capsules, 2, 3).
        radii: Radii of the capsules [m].
        contact_radii: Radii with which the capsules touch the tissue [m]: the same radii, as the
            scenes pass them (the tools have one capsule set).
        capsule_tools: Tool of each capsule, numbered from zero.

    Returns:
        One mass per tool [kg], shape (tools,).
    """
    import mujoco

    from ..._src.solvers.coupled.interface.swept_tool_tissue_contact import capsule_pieces

    native_model = rigid_solver.mj_model
    data = mujoco.MjData(native_model)
    # Newton's own conversion of joint coordinates, into these private data.
    rigid_solver._update_mjc_data(data, model, state)
    mujoco.mj_forward(native_model, data)
    native = rigid_solver.mjc_body_to_newton.numpy()[0]
    piece_bodies, probes, _, _, _, _, firsts, counts = capsule_pieces(bodies, endpoints, radii, contact_radii)
    body_q = state.body_q.numpy().astype(float)
    jacobian, solved = np.zeros((3, native_model.nv)), np.zeros((3, native_model.nv))
    capsule_tools = np.asarray(capsule_tools, dtype=np.int64)
    lightest = np.full(int(capsule_tools.max()) + 1, np.inf)
    for tool, first, count in zip(capsule_tools, firsts, counts, strict=True):
        body = int(piece_bodies[first])
        index = int(np.flatnonzero(native == body)[0])
        if np.linalg.norm(data.xpos[index] - body_q[body, :3]) > 1.0e-5:
            raise RuntimeError("The native MuJoCo data do not reproduce the body poses of the given state")
        rotation = data.xmat[index].reshape(3, 3)
        for local in (probes[first], probes[first + count - 1]):
            mujoco.mj_jac(native_model, data, jacobian, None, data.xpos[index] + rotation @ local, index)
            mujoco.mj_solveM(native_model, data, solved, jacobian)
            lightest[tool] = min(lightest[tool], 1.0 / float(np.linalg.eigvalsh(jacobian @ solved.T).max()))
    return lightest


def placed_tool_effective_masses(scene):
    """Effective mass of each tool of ``scene`` [kg], at the configuration that ``scene.state_0`` holds.

    Per tool, its smallest effective mass at an end of one of its capsules, in any direction
    (:func:`tool_effective_masses`), with the capsule set that the tool models publish. The
    swept tool/tissue contact takes its held-tissue stiffness from it. A needle is a body of
    its own and adds nothing. A scene asks for it when its tools have been placed: the grasp
    scene at its reset (a reset builds a new worker thread, so every run uses the masses of
    its own placed start: the first tool on the needle with its jaws closed, the second parked
    with its jaws open), the tool/tissue scene after every placement of its tools.
    """
    from ..._src.solvers.coupled.interface.tool_tissue_exchange import tool_capsules

    bodies, endpoints, radii, tools, _ = tool_capsules(scene.graspers)
    return tool_effective_masses(scene.rigid_solver, scene.model, scene.state_0, bodies, endpoints, radii, radii, tools)


def tool_tissue_setup(scene, law):
    """What a scene with tools gives its tool/tissue exchange, or ``None`` for a setting without a contact.

    The one function that turns a scene into a ``ToolTissueSetup``; the grasp scene and the
    tool/tissue scene both call it, and each builds its one ``ToolTissueExchange`` from the
    result. The capsules and the jaw bodies are those of ``scene.graspers``; reach, node
    stiffness, shear frequency, fixed particles and sweep horizon follow from the scene by
    the rules of ``ToolTissueSetup.of_scene``; with the swept law the effective masses are
    those of the tools as ``scene.state_0`` has them (:func:`placed_tool_effective_masses`).
    No other radii than those of the capsule set are passed.

    Args:
        scene: A scene with ``graspers``, ``model``, ``state_0``, ``rigid_solver``, ``reference``,
            ``selection``, ``args``, and for the swept law ``fixed_vertex_indices`` and ``sim_dt``.
        law: A setting of the tool/tissue contact: ``None``, ``False`` or "off"; ``True`` or
            "first"; "swept" (``tool_tissue_law``).
    """
    from ..._src.solvers.coupled.interface.tool_tissue_exchange import ToolTissueSetup, tool_tissue_law

    law = tool_tissue_law(law)
    if law is None:
        return None
    masses = placed_tool_effective_masses(scene) if law == "swept" else None
    return ToolTissueSetup.of_scene(scene, law, effective_mass=masses)


class GraspScene(Example):
    """Worker scene; source examples retain their explicit free-needle mode."""

    cooperative_shaft_projection = True
    rigid_backend = "native"
    surface_preconditioner = True
    certified_linear_residual = True
    preconditioned_anderson = True
    held_fem_start = True
    response_basis_guesses = True
    host_wrench_reuse = True

    def configure_builder(self, builder):
        super().configure_builder(builder)
        self.source_layout = load_source_layout()
        self.graspers = add_dvrk_tools(builder)

    @staticmethod
    def create_parser():
        parser = Example.create_parser()
        parser.add_argument(
            "--tool-tissue-contact",
            nargs="?",
            const=True,
            default=False,
            choices=("off", "first", "swept"),
            help="Frictionless contact between the tool capsules (shaft, wrist, jaws of both tools) and the tissue "
            "surface, with the size of the drawn tool. Off by default: tools and tissue then exchange no force. "
            "Without a value, or with 'first': the tissue yields to a tool and pushes it back, one accepted interval "
            "late (ToolTissueContact); it carries light touches only: a tool that is driven into the tissue, or onto "
            "its held border, ends in rejected tetrahedra (docs/method.md). With 'swept': the contact on the sweep "
            "of the capsules (SweptToolTissueContact): every capsule is swept to its pose at the end of the next "
            "interval, the tissue nodes it reaches are placed on it and held there through that interval, and "
            "tissue that is held or cannot follow stops the tool. 'off' is the default.",
        )
        parser.add_argument(
            "--thread-tracks",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="The thread follows the needle through the tissue (default): the track that the needle actually "
            "cuts becomes eyelets of the thread, one group per passage and one eyelet every 1.5 mm, and an eyelet "
            "is threaded when the needle's tail has gone past it (WorkerThread, NeedleTrackEyelets). With "
            "--no-thread-tracks the thread has no eyelets and stays on the tissue surface: a tail that is taken "
            "into the tissue then drags a thread that the surface presses back out.",
        )
        return parser

    def _commanded_body(self):
        return self.graspers[0].palm_body

    def _configure_rigid_dynamics(self):
        # The insertion preview supplies the nominal grasp pose after the FEM
        # model is built. Author fixed world anchors in _initialize_pose, then
        # construct native MuJoCo from those frames. They never move in a run.
        self.worker_thread = None
        self.jaw_opening_target = 0.0
        # The needle's mass [kg] as a host value, read once here, at construction: the weight that the
        # scene publishes (grasp_geometry, needle_weight_N) is formed from it without a read.
        self.needle_mass_kg = float(self.model.body_mass.numpy()[self.needle_body])

    def _build_rigid_solver(self):
        self.rigid_solver = tool_rigid_solver(self.model, self.rigid_backend)
        self.rigid_solver.reset(self.state_0, flags=0)
        self.pose_ik = tool_pose_ik(self.model, self.rigid_solver, self.graspers)
        self.arm_targets = [self.state_0.joint_q.numpy()[list(tool.arm_q_indices)].copy() for tool in self.graspers]
        self._configure_gpu_pose_ik()
        self._configure_held_fem_start()
        self._configure_response_basis_guesses()
        self._configure_certified_linear_residual()

    def _configure_gpu_pose_ik(self):
        """Choose both private fixed-base device IK workspaces, or retain both native solvers."""
        self._gpu_pose_ik_active = False
        self._gpu_pose_ik_fallback = None
        if not is_gpu_backend(self.rigid_solver):
            return
        from ..._src.sim.ik.gpu_pose_ik import GPUPoseIK
        from ..._src.solvers.fem.differentiation import active_tape

        if getattr(self.model, "requires_grad", False) or active_tape() is not None:
            self._gpu_pose_ik_fallback = "gradient model or active tape"
            return
        try:
            candidates = [
                GPUPoseIK(
                    ik.model,
                    ik.joint_ids,
                    ik.body_id,
                    device=self.device,
                    position_tolerance_m=ik.position_tolerance_m,
                    orientation_tolerance_rad=ik.orientation_tolerance_rad,
                    rotation_length_m=ik.rotation_length_m,
                    max_iterations=ik.max_iterations,
                )
                for ik in self.pose_ik
            ]
        except ValueError as error:
            # Unsupported authored chains retain the original complete native path.
            self._gpu_pose_ik_fallback = str(error)
            return
        for ik in candidates:
            ik.reset(self.rigid_solver.mj_data.qpos)
        self.pose_ik = candidates
        self._gpu_pose_ik_active = True

    def start(self):
        super().start()
        if is_gpu_backend(self.rigid_solver):
            policy = os.environ.get("SUTUREBOT_EXPERIMENTAL_MJW_COUPLING", "device-anderson")
            if policy not in ("current", "device-aitken", "device-anderson"):
                raise ValueError("Experimental GPU coupling policy must be current, device-aitken or device-anderson")
            self.coupler._gpu_trial_policy = policy
            guess = os.environ.get("SUTUREBOT_EXPERIMENTAL_MJW_CONTACT_GUESS", "device")
            if guess not in ("device", "zero"):
                raise ValueError("Experimental GPU contact-preconditioner guess must be device or zero")
            self.coupler._gpu_contact_guess = guess == "device"
        self.interaction.cooperative_shaft_projection = bool(
            self.cooperative_shaft_projection
            and not getattr(self.model, "requires_grad", False)
            and not getattr(self.solver, "differentiable", False)
        )

    def _configure_certified_linear_residual(self):
        from ..._src.solvers.fem.certified_scope import CertifiedFEMScope

        previous = getattr(self.solver, "_certified_linear", None)
        if previous is not None:
            previous.end_attempt()
        self.solver._certified_linear = None
        if self.certified_linear_residual and self.model.device.is_cuda:
            self.solver._certified_linear = CertifiedFEMScope(self.solver)

    def _configure_held_fem_start(self):
        """Reuse only the immutable accepted start of native coupling trials."""
        from ..._src.solvers.fem.held_start import HeldFEMStart

        old = getattr(self.solver, "_held_start", None)
        if old is not None:
            old.invalidate()
        self.solver._held_start = None
        if not (
            self.held_fem_start
            and supported_subcycled(self.rigid_solver)
            and self.rigid_solver.rigid_substeps == 4
            and self.model.world_count == 1
        ):
            return
        held = HeldFEMStart(self.solver)
        if held.supported(self.state_0, self.control, None):
            self.solver._held_start = held

    def _configure_response_basis_guesses(self):
        """Reuse approximate responses only as guesses for the original solves."""
        from ..._src.solvers.coupled.acceleration.woodbury_guess import _CAPACITY as woodbury_capacity
        from ..._src.solvers.fem.affine_guess import AffineGuessManager
        from ..._src.solvers.fem.warp_basis_provider import WarpBasisProvider

        self.close()
        if not (
            self.response_basis_guesses
            and self.model.device.is_cuda
            and supported_subcycled(self.rigid_solver)
            and self.rigid_solver.rigid_substeps == 4
            and self.model.world_count == 1
            and getattr(self.solver, "_held_start", None) is not None
        ):
            return
        # One response column per support column of the start guess (four per track sample).
        capacity = woodbury_capacity
        provider = WarpBasisProvider(tolerance=3e-4, maxiter=160, capacity=capacity, degree=4)
        self._response_basis_provider = provider
        self.solver._affine_predictor = AffineGuessManager(
            self.solver,
            provider,
            capacity=capacity,
            refresh_generations=0,
            # 0 (the default of the option): every missing column at once.
            nodes_per_attempt=getattr(getattr(self, "args", None), "response_basis_nodes", 0) or None,
        )

    def close(self):
        """Release this scene's response graphs outside any physical capture."""
        # An attempt that a call left open ends first, as a rejected attempt ends: its close
        # addresses the response graphs that are released below.
        self.abandon_attempt()
        # A scene closed in the middle of an unfinished frame keeps nothing of that frame.
        self._pending_frame = self._ramp_frame = None
        coupler = getattr(self, "coupler", None)
        preconditioner = getattr(coupler, "_shaft_preconditioner", None)
        if preconditioner is not None:
            preconditioner.invalidate_preparation()
        solver = getattr(self, "solver", None)
        manager = getattr(solver, "_affine_predictor", None)
        if manager is not None:
            manager.invalidate()
            solver._affine_predictor = None
        provider = getattr(self, "_response_basis_provider", None)
        self._response_basis_provider = None
        if provider is not None:
            provider.close()

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
        self.initial_grasp_parameters = _initial_grasp_parameters(self.plan)
        configured = getattr(getattr(self, "worker_config", None), "initial_grasp", None)
        if configured is not None:
            fraction, roll, bite = configured
            self.initial_grasp_parameters = {
                "arc_fraction": float(fraction),
                "tangent_roll_rad": float(roll),
                "bite_angle_rad": float(bite),
            }
        # Reset opening of the jaws: the parked tool is open; the first tool starts with its
        # configured opening, or about a needle in its bite with each jaw at the angle at which
        # its capsules touch that needle.
        self.reset_jaw_openings = [None, 0.7]
        relative = None
        if self.plan is not None:
            # Reset-only bite placement; physical jaw contact and the measured
            # settled grasp determine all subsequent needle motion.
            relative = needle_to_gripper_pose(self.geometry, **self.initial_grasp_parameters)
            needle_in_palm = wp.transform_inverse(wp.transform(*relative))
            centerline = [
                np.asarray(wp.transform_point(needle_in_palm, wp.vec3(*point)), dtype=float)
                for point in np.asarray(self.geometry.centerline, dtype=float)
            ]
            self.reset_jaw_openings[0] = jaw_opening_on_needle(self.graspers[0], centerline, self.geometry.thickness)
        # These world frames are independent of the needle and its reset noise.
        # The registered source ports stay fixed for every insertion and reset.
        for tool, trocar, opening in zip(
            self.graspers, self.source_layout["trocar_poses"], self.reset_jaw_openings, strict=True
        ):
            tool.set_trocar_pose(self.model, trocar, reset=True)
            tool.initialize(self.model, self.state_0, opening_rad=opening)
        self._build_rigid_solver()
        self.parked_target = self.source_layout["parked_pose"].copy()
        targets = [None, self.parked_target]
        if relative is not None:
            targets[0] = np.asarray(wp.transform_multiply(wp.transform(*pose), wp.transform(*relative)))
        # Reset-only IK places the ready arms. During stepping, the same IK
        # writes actuator targets only; it never writes physical coordinates.
        for tool, ik, target, opening in zip(
            self.graspers, self.pose_ik, targets, self.reset_jaw_openings, strict=True
        ):
            if target is not None:
                arm = ik.solve(target, self.rigid_solver.mj_data.qpos)
                tool.initialize(self.model, self.state_0, arm_positions=arm, opening_rad=opening)
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
        if is_gpu_backend(self.rigid_solver):
            self.rigid_solver.initialize_from_newton(self.state_0, self.control)
            self.rigid_solver.bind_grasp_observer(
                self.model.body_label[self.needle_body],
                [self.model.body_label[body] for body in self.graspers[0].jaw_bodies],
            )
        self.state_1.assign(self.state_0)
        self.rigid_rest_poses = self.state_0.body_q.numpy().copy()

    def initialize_worker_thread(self, config):
        from ..._src.solvers.thread.worker_thread import WorkerThread

        # What tools and tissue exchange (None without a contact): the worker thread builds the one exchange
        # object from it, with the surface that its tissue coupling refits.
        setup = tool_tissue_setup(self, getattr(self.args, "tool_tissue_contact", False))
        self.worker_thread = WorkerThread(self, config, tool_tissue=setup)
        self._preload_insertion_modules()

    def _preload_insertion_modules(self):
        """Load known CUDA modules at startup without advancing physical state."""
        from ..._src.solvers.fem.differentiation import active_tape

        if (
            not self.device.is_cuda
            or self.coupler is None
            or getattr(self.model, "requires_grad", False)
            or getattr(self.solver, "differentiable", False)
            or active_tape() is not None
        ):
            return
        from ..._src.solvers.coupled import phase_predictor
        from ..._src.solvers.coupled.acceleration import shaft_preconditioner
        from ..._src.solvers.coupled.interface import tunnel_alm

        modules = []
        start_guess = None
        if self.coupler._shaft_preconditioner is not None:
            modules.append(shaft_preconditioner)
            if getattr(self.solver, "_affine_predictor", None) is not None:
                # The start guess of the correction is built with the first track sample.
                from ..._src.solvers.coupled.acceleration import woodbury_device, woodbury_guess

                modules.extend((woodbury_guess, woodbury_device))
                start_guess = woodbury_device
        if self.interaction.tunnel_alm is not None:
            modules.append(tunnel_alm)
        modules.append(phase_predictor)
        if getattr(self, "cooperative_shaft_projection", False):
            from ..._src.solvers.coupled.interface import (
                shaft_projected_kernels,
                shaft_projection,
                shaft_projection_tables,
            )

            modules.extend((shaft_projection_tables, shaft_projection, shaft_projected_kernels))
        for module in modules:
            # Nothing is launched and nothing advances here: loading by name only moves
            # first-use compilation/loading out of needle entry. The start-up work that
            # launches and discards is in capture_graphs().
            wp.load_module(module, device=self.device)
        if start_guess is not None:
            # The factorization of each padded tile of the start guess is a module of its own,
            # which a track would otherwise build when it first reaches the size of that tile.
            start_guess.load_kernels(self.device)

    def _startup_refusal(self):
        """Why this scene has no start-up call, or None if it has one."""
        from ..._src.solvers.fem.differentiation import active_tape

        if not is_gpu_backend(self.rigid_solver):
            return "native rigid backend"
        if not self.device.is_cuda:
            return "CPU device"
        if not wp.is_conditional_graph_supported():
            return "conditional CUDA graphs unavailable"
        if self.coupler is None:
            return "no coupler"
        if (
            getattr(self.model, "requires_grad", False)
            or getattr(self.solver, "differentiable", False)
            or active_tape() is not None
        ):
            return "differentiable configuration"
        if self.elapsed > 0 or self.accepted_physics_steps:
            return "the scene has advanced since its reset"
        return None

    def _load_run_modules(self):
        """Load the kernel modules that only eager code of a frame launches, at the block sizes it launches them with.

        By name, with no launch: a graph capture loads the modules of what it records, these
        are launched outside every graph.
        """
        from ..._src.solvers.thread import worker_thread
        from ..needle import example_fem_needle_insertion

        modules = [
            # Needle weight and the ramp of the arm targets, every interval.
            (__name__, (256,)),
            # State check of every accepted interval, tissue integrity of every frame, markers.
            (example_fem_needle_insertion, (256,)),
            # Thread reactions of every interval, centreline samples and reaction norm of every observation.
            (worker_thread, (256,)),
        ]
        if getattr(self, "_gpu_pose_ik_active", False):
            from ..._src.sim.ik import gpu_pose_ik

            # The solve of every frame (one thread per block) and the write of its targets.
            modules.append((gpu_pose_ik, (256, 1)))
        if self.coupler._tet_step_guard is not None:
            from ..._src.solvers.coupled import tet_step

            # Step guard of every interval that converged.
            modules.append((tet_step, (256,)))
        exchange = getattr(self.worker_thread, "exchange", None)
        if exchange is not None:
            # Reaction of the tool/tissue contact on the tools, every interval; with the swept law also the
            # balancing forces of the held nodes and the holds of every tangent refresh. The update itself is
            # in the thread's graph.
            modules.extend(exchange.modules())
        for module, block_dims in modules:
            for block_dim in block_dims:
                wp.load_module(module, device=self.device, block_dim=block_dim)

    def capture_graphs(self):
        """Start-up call: capture, before the first frame, graphs that the run would capture in its frames.

        Called at the end of a reset, with the scene at its start state. On the MuJoCo Warp rigid
        backend with conditional CUDA graphs it

        - builds the tangent of the first frame (:meth:`simulate` then uses it instead of its own
          refresh, once) and the projected operator of that tangent;
        - loads the kernel modules of the FEM solver (``SolverFEM.load_kernel_modules``) and those
          that only eager code of a frame launches;
        - captures and launches the thread's step graph for both accepted-state buffers
          (``WorkerThread.capture_graphs``), with the update of the tool/tissue contact in it
          where the scene has that contact (either law; the swept law's placement of tissue nodes
          and its holds in the tissue matrix are put back as well);
        - for both buffers (``NeedleTissueCoupler.capture_graphs``) makes the storage of the
          response-basis predictor and captures and launches the held start of the FEM solver, the
          trial preparation and the rigid interval of the host-controlled trials, their FEM step at
          every inner tolerance, the trial graph of the device trials, the force graphs of the
          host-controlled trials and the graphs of their correction;
        - for a scene that steps in control frames of a worker, makes the phase load predictor of
          those frames with the storage of its three frames and of the engagement snapshot of
          every phase (``PhaseLoadPredictor.reserve``). The predictor holds no history and takes
          no part in the captures above.

        The captures run real steps and discard them: the thread state, the integration state and
        every accepted array get back the bits they had, no time passes, no step is accepted and no
        history changes. Work counters keep their values; capture counters include these captures.
        A correction that starts from a response basis is captured there as well where the
        predictor is configured; without it that solve is captured where the run first needs it.

        Returns:
            True if the call ran. False if the scene has no start-up call or the coupled step is
            not captured; ``_startup_status`` then holds the reason, and nothing is checked after
            start-up.
        """
        reason = self._startup_refusal()
        if reason is not None:
            self._startup_status = reason
            return False
        # A rebuilt scene leaves the cycles of the one before it, which own device memory and
        # graphs: collected here, not in a frame.
        gc.collect()
        # The positions, the time step and the code path of the first frame's own refresh.
        wp.copy(self._frame_particle_q, self.state_0.particle_q)
        self.solver.refresh_tangent(self._frame_particle_q, self.sim_dt)
        self._startup_tangent = (self.solver._tangent_version, self.sim_dt, self.accepted_physics_steps)
        self.coupler._refresh_projection()
        self.solver.load_kernel_modules()
        self._load_run_modules()
        # Both buffers hold the start state: every captured step below starts from an accepted state.
        self.state_1.assign(self.state_0)
        buffers = (self.state_0, self.state_1)
        if self.worker_thread is not None:
            self.worker_thread.capture_graphs(buffers, self.sim_dt)
        if not self.coupler.capture_graphs(buffers, self.control, self.sim_dt):
            self._startup_status = f"coupled step not captured: {self.coupler.fem_graph_status}"
            return False
        if self._control_frames():
            # After the coupler's call, whose attempts take no predictor. Without this the first
            # frame allocates the loads of the predictor's frames, and each step that is the first
            # of its frame and phase with a force signature allocates its engagement snapshot.
            self._phase_predictor().reserve(
                frame_dt=self.frame_dt, step_dt=self.sim_dt, engagement=self.interaction._track_engaged
            )
        return True

    def _ramp_first_arm(self, dt):
        """Spread the frame's change of the first tool's arm targets over the frame.

        An action sets one pose target per frame, and its inverse kinematics gives one set of
        arm actuator targets. Held for the whole frame, those targets are a step at the frame's
        first interval: the tool speeds up and slows down again inside every frame, and the
        needle load changes by tens of percent from one interval to the next. The targets of an
        interval therefore lie on the straight line between the targets of the previous frame
        and of this frame, at the END of that interval. The last interval of a frame ends at
        the targets of the commanded pose, as it did with held targets. The inverse kinematics
        still runs once per frame; the ramp is one device kernel per interval, with no read.

        Args:
            dt: Duration of the interval about to be attempted [s].
        """
        frame = getattr(self, "_ramp_frame", None)
        if frame is None:
            return
        solved = np.asarray(self.arm_targets[0], dtype=np.float32)
        ramp = getattr(self, "_arm_ramp", None)
        device = self.control.joint_target_q.device
        if ramp is None or ramp["reset"] != self.reset_count or ramp["previous"].device != device:
            # No frame of this reset precedes this one: its targets are held.
            ramp = self._arm_ramp = {
                "reset": self.reset_count,
                "frame": frame,
                "solved": solved.copy(),
                "previous": wp.array(solved, dtype=float, device=device),
                "target": wp.array(solved, dtype=float, device=device),
                "indices": wp.array(list(self.graspers[0].arm_q_indices), dtype=int, device=device),
            }
        elif ramp["frame"] != frame:
            # First interval of a new frame: the last frame's targets are where the ramp starts.
            ramp["previous"].assign(ramp["solved"])
            ramp["target"].assign(solved)
            ramp["solved"], ramp["frame"] = solved.copy(), frame
        fraction = (self.elapsed + dt - frame[1]) / self.frame_dt
        wp.launch(
            _ramp_arm_targets,
            solved.size,
            [ramp["previous"], ramp["target"], float(fraction), ramp["indices"], self.control.joint_target_q],
            device=device,
        )

    def _weighs_needle(self):
        """Whether every step applies the needle's weight (the worker's ``needle_weight``; without a worker it does)."""
        return bool(getattr(getattr(self, "worker_config", None), "needle_weight", True))

    def _prepare_step_loads(self, desired, velocity, dt):
        self._external_forces.zero_()
        if self._weighs_needle():
            wp.launch(
                _needle_weight,
                1,
                [self.needle_body, self.model.body_mass, self.state_0.body_f],
                device=self.device,
            )
        device_ik = getattr(self, "_gpu_pose_ik_active", False)
        if device_ik:
            from ..._src.solvers.fem.differentiation import active_tape

            if (
                getattr(self.model, "requires_grad", False)
                or active_tape() is not None
                or self.control.joint_target_q.requires_grad
                or self.control.joint_target_q.grad is not None
            ):
                raise ValueError("Device pose IK requires non-differentiable actuator targets")
        targets = [
            ik.solve(
                desired if index == 0 else self.parked_target,
                None if device_ik else self.rigid_solver.mj_data.qpos,
            )
            for index, ik in enumerate(self.pose_ik)
        ]
        if not np.array_equal(targets, self.arm_targets):
            if device_ik:
                # Both convergence checks finish before any actuator buffer changes.
                for index, (assembly, ik) in enumerate(zip(self.graspers, self.pose_ik, strict=True)):
                    ik.solve_into(
                        desired if index == 0 else self.parked_target,
                        self.control.joint_target_q,
                        assembly.arm_q_indices,
                    )
            else:
                coordinates = self.control.joint_target_q.numpy()
                for assembly, target in zip(self.graspers, targets, strict=True):
                    coordinates[list(assembly.arm_q_indices)] = target
                self.control.joint_target_q.assign(coordinates)
            self.arm_targets = targets
        self._ramp_first_arm(dt)
        self.graspers[0].set_aperture(
            self.control,
            self.jaw_opening_target / self.graspers[0].config.maximum_opening_rad,
        )
        if self.worker_thread is not None:
            self.worker_thread.add_forces(self.state_0.body_f, self._external_forces)

    def _accepted_step(self, dt):
        if self.worker_thread is not None:
            self.worker_thread.advance(dt)

    def command(self, name, **arguments):
        """Carry out a command that came with an action (``commands`` of the pool's ``step``), before the frame.

        ``"tie_thread"`` is the one command (:meth:`tie_thread`).
        """
        if name != "tie_thread":
            raise ValueError(f"Unknown scene command: {name!r}")
        self.tie_thread(**arguments)

    def tie_thread(self, passage=0, last=False):
        """Tie the thread to the tissue at an end of a passage of the needle's track (``WorkerThread.tie``).

        A stand-in for a knot at the hole through which the thread goes into the tissue (or, with
        ``last``, comes out of it). Between two frames only: the accepted step of the thread reads
        the tie from device data. The observation reports the ties that hold (``thread_tension``);
        a tie that could not be made, such as one at an eyelet that the thread does not pass
        through, is missing there against ``ties_asked``.
        """
        if self.worker_thread is None:
            raise ValueError("This scene has no thread to tie")
        if self.frame_pending:
            raise RuntimeError("A tie is made between two frames, not inside an unfinished one")
        self.worker_thread.tie(passage, last)

    def _reset_interaction_state(self):
        super()._reset_interaction_state()
        # A reset in the middle of an unfinished frame ends the frame of the arm-target ramp too.
        self._ramp_frame = None

    def _control_frames(self):
        """Whether :meth:`step` advances control frames of a worker, which the phase load predictor follows."""
        worker = getattr(self, "worker_config", None)
        return (
            self.coupler is not None
            and getattr(worker, "control_mode", None) == "physical_grasp"
            and all(hasattr(self, name) for name in ("cartesian_target", "jaw_opening_target", "reset_count"))
        )

    def _phase_predictor(self):
        """The coupler's phase load predictor; made by the first caller (the start-up call, or the first frame)."""
        from ..._src.solvers.coupled.phase_predictor import PhaseLoadPredictor

        predictor = self.coupler._phase_load_predictor
        if predictor is None:
            # The device-Anderson GPU coupling follows the last accepted load by its frame-periodic
            # change. The other policies and the native backend keep the same-phase guess.
            predictor = PhaseLoadPredictor(
                self.coupler._load,
                increments=is_gpu_backend(self.rigid_solver)
                and getattr(self.coupler, "_gpu_trial_policy", None) == "device-anderson",
            )
            self.coupler._phase_load_predictor = predictor
        return predictor

    def step(self, deadline=None, *, inside=False):
        """Align optional force guesses with complete worker control frames.

        With a ``deadline`` (see the inherited ``simulate``) a call may leave the frame
        unfinished, between two substeps or between a rejected attempt and its retry, and with
        ``inside`` also inside an attempt, between two host-controlled trials. The call
        that goes on with it runs nothing of the frame's start again: the predictor's control
        frame and the frame of the arm-target ramp stay open from the call that starts a frame
        to the call that finishes it.
        """
        if self.frame_pending:
            return self._step_in_frame(self.coupler._phase_load_predictor, deadline, inside)
        rigid_solver = getattr(self, "rigid_solver", None)
        if is_gpu_backend(rigid_solver) and not getattr(self, "_gpu_pose_ik_active", False):
            rigid_solver.sync_native_shadow(forward=True, token=self.elapsed)
        if not self.running or not self._control_frames():
            return super().step()
        predictor = self._phase_predictor()
        predictor.begin_frame(
            frame_dt=self.frame_dt,
            step_dt=self.sim_dt,
            elapsed=self.elapsed,
            command=np.r_[self.cartesian_target, self.jaw_opening_target],
            reset_token=self.reset_count,
        )
        # A frame is known to the arm-target ramp by its reset and the time it starts at.
        self._ramp_frame = (self.reset_count, self.elapsed)
        return self._step_in_frame(predictor, deadline, inside)

    def _step_in_frame(self, predictor, deadline, inside=False):
        """Step inside the open control frame, and close that frame unless the step left it unfinished."""
        success = False
        try:
            if inside:
                result = super().step(deadline, inside=True)
            else:
                result = super().step() if deadline is None else super().step(deadline)
            success = self.running
            return result
        finally:
            if not self.frame_pending:
                predictor.end_frame(self.elapsed, success)
                self._ramp_frame = None

    def report(self):
        """Describe the actual actuators, rather than inherited needle-drive args."""
        result = super().report()
        transfer = getattr(self.rigid_solver, "_native_transfer", None)
        anderson = getattr(self.coupler, "_device_surface_anderson_workspace", None)
        pc = self.coupler._shaft_preconditioner
        contact_prepare = getattr(getattr(pc, "_input_preparation", None), "replay", None)
        packet_prepare = getattr(
            getattr(getattr(pc, "_woodbury_guess", None), "_packet_preparation", None), "replay", None
        )
        for key in (
            "drive_stiffness_N_per_m",
            "angular_drive_stiffness_Nm_per_rad",
            "implicit_drive_spring",
        ):
            result.pop(key, None)
        captured = getattr(self.coupler, "_captured_trials", None)
        graph_anderson = getattr(getattr(captured, "rigid", None), "anderson", None)
        result.update(
            gpu_coupling_anderson_counters=None if graph_anderson is None else graph_anderson.counters.numpy().tolist(),
            cooperative_shaft_projection=bool(getattr(self.interaction, "cooperative_shaft_projection", False)),
            authorized_outside_certified_trials=getattr(self.interaction, "_authorized_outside_certified_trials", 0),
            control_mode="physical_grasp",
            drive_body="RCM shaft, insertion, roll and wrist joints",
            shaft_constraint="fixed trocar remote center with bounded axial insertion",
            pose_control=(
                "bounded private GPU kinematic IK to finite joint position actuators"
                if getattr(self, "_gpu_pose_ik_active", False)
                else "bounded native-model IK to finite joint position actuators"
            ),
            gpu_pose_ik_active=getattr(self, "_gpu_pose_ik_active", False),
            gpu_pose_ik_fallback=getattr(self, "_gpu_pose_ik_fallback", None),
            ik_diagnostics=[ik.last_diagnostics.copy() for ik in self.pose_ik],
            independent_free_needle=True,
            needle_pose_drive=False,
            rigid_gravity_m_per_s2=9.81,
            tool_gravity_compensation="each tool link weight compensated; needle weight excluded",
            gripper_count=len(self.graspers),
            native_equality_constraint_count=int(self.rigid_solver.mj_model.neq),
            native_rigid_substeps=self.rigid_solver.rigid_substeps,
            native_batches=getattr(self.rigid_solver, "native_batches", 0),
            rigid_backend="mujoco-warp-experimental" if is_gpu_backend(self.rigid_solver) else "mujoco-native",
            gpu_coupling_policy=getattr(self.coupler, "_gpu_trial_policy", None),
            gpu_rigid_diagnostics=self.rigid_solver.diagnostics() if is_gpu_backend(self.rigid_solver) else None,
            native_reset_guard_enabled=not is_gpu_backend(self.rigid_solver),
            native_clock_s=None if is_gpu_backend(self.rigid_solver) else float(self.rigid_solver.mj_data.time),
            native_clock_error_s=None
            if is_gpu_backend(self.rigid_solver)
            else float(self.rigid_solver.mj_data.time) - self.elapsed,
            native_guard_rejections=getattr(self.rigid_solver, "native_guard_rejections", 0),
            native_guard_microsteps=getattr(self.rigid_solver, "native_guard_microsteps", 0),
            native_guard_last_failure=getattr(self.rigid_solver, "native_guard_last_failure", None),
            certified_linear_residual_enabled=getattr(self.solver, "_certified_linear", None) is not None,
            certified_linear_attempts=getattr(getattr(self.solver, "_certified_linear", None), "attempts", 0),
            certified_linear_rejections=getattr(getattr(self.solver, "_certified_linear", None), "rejections", 0),
            held_fem_start_enabled=getattr(self.solver, "_held_start", None) is not None,
            held_fem_start_preparations=getattr(getattr(self.solver, "_held_start", None), "preparations", 0),
            response_basis_enabled=getattr(self.solver, "_affine_predictor", None) is not None,
            response_basis_builds=getattr(getattr(self.solver, "_affine_predictor", None), "basis_builds", 0),
            response_reference_builds=getattr(getattr(self, "_response_basis_provider", None), "reference_builds", 0),
            device_anderson_transport_calls=getattr(anderson, "transport_calls", 0),
            device_anderson_transport_bytes=getattr(anderson, "transport_bytes", 0),
            device_anderson_overflow_packets=getattr(anderson, "overflows", 0),
            device_anderson_max_active_rows=getattr(anderson, "max_active_rows", 0),
            device_anderson_packet_capacity=getattr(anderson, "_capacity", 0),
            shaft_pc_graph_captures=getattr(pc, "graph_captures", 0),
            shaft_pc_storage_growths=getattr(pc, "storage_growths", 0),
            contact_preparation_captures=getattr(contact_prepare, "captures", 0),
            contact_preparation_launches=getattr(contact_prepare, "launches", 0),
            contact_preparation_eager=getattr(contact_prepare, "eager", 0),
            woodbury_packet_preparation_captures=getattr(packet_prepare, "captures", 0),
            woodbury_packet_preparation_launches=getattr(packet_prepare, "launches", 0),
            woodbury_packet_preparation_eager=getattr(packet_prepare, "eager", 0),
            native_general_intervals=getattr(self.rigid_solver, "general_intervals", 0),
            native_transfer_graph_captures=getattr(transfer, "captures", 0),
            native_transfer_graph_launches=getattr(transfer, "launches", 0),
            native_transfer_readbacks=getattr(transfer, "readbacks", 0),
            native_transfer_repeated_readbacks=getattr(transfer, "repeated_readbacks", 0),
            native_transfer_uploads=getattr(transfer, "uploads", 0),
            native_host_wrench_reuses=getattr(getattr(transfer, "_repeated_input", None), "host_reuses", 0),
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
            thread_coupling="lagged reactions from accepted physical thread steps; "
            + (
                "eyelets along the needle's actual track, threaded where the needle's tail has passed"
                if getattr(self.worker_thread, "thread_tracks", False)
                else "no scheduled eyelets"
            ),
            fixture_layout=self.source_layout["provenance"],
            initial_grasp_parameters=self.initial_grasp_parameters.copy(),
            reset_jaw_opening_rad=getattr(self, "reset_jaw_openings", [None])[0],
            reset_jaw_opening_rule="the configured initial opening, or with a needle in the bite the angle of each "
            "jaw at which its capsules touch it (jaw_opening_on_needle)",
            calibrated=False,
        )
        contact = getattr(self.worker_thread, "tool_tissue", None)
        if contact is not None:
            # Host values only; the forces are read by ToolTissueContact.diagnostics().
            result["tool_tissue_contact"] = {
                **contact.parameters(),
                "capsule_contact_radii_m": list(self.worker_thread.tool_tissue_radii),
                "contact_radius_source": "the capsule set of the tools, fitted to the drawn tool (dvrk_tool_capsules)",
            }
        swept = getattr(self.worker_thread, "swept_tool_tissue", None)
        if swept is not None:
            # Host values only; the state is read by SweptToolTissueContact.diagnostics().
            result["tool_tissue_contact"] = {
                **swept.parameters(),
                "capsule_contact_radii_m": list(self.worker_thread.tool_tissue_radii),
                "contact_radius_source": "the capsule set of the tools, fitted to the drawn tool (dvrk_tool_capsules)",
                "effective_tool_mass_kg": list(self.worker_thread.tool_effective_mass),
                "effective_tool_mass_rule": "per tool, at the reset state: smallest 1 / lambda_max(J M^-1 J^T) over "
                "the ends of its capsules",
            }
        return result

    def observe_grasp(self, *, host_state=None, include_thread_diagnostics=True):
        """Share one rigid readback across tool, RCM and thread diagnostics.

        ``include_thread_diagnostics=False`` leaves ``thread_diagnostics`` out. That report reads
        14 thread and thread-contact arrays from the device and checks the thread coordinates and
        velocities for nonfinite values; ``thread_positions`` keeps the check of the sampled
        centreline.
        """
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
            )
            # What the read of the positions brought with it: the ties and the thread's pull.
            result["thread_tension"] = self.worker_thread.tension
            if include_thread_diagnostics:
                result["thread_diagnostics"] = self.worker_thread.diagnostics(
                    needle_pose=host_state["body_q"][self.needle_body]
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
        contact = getattr(self.worker_thread, "tool_tissue", None)
        swept = getattr(self.worker_thread, "swept_tool_tissue", None)
        return {
            "scope": "accepted FEM tissue, independent free MuJoCo needle, two fixed-trocar articulated tools and four actuated jaws; worker ANCF thread with lagged rigid/tissue reactions"
            + ("" if contact is None else "; lagged frictionless contact of the tool capsules with the tissue surface")
            + ("" if swept is None else "; swept contact of the tool capsules with the tissue surface (held nodes)"),
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
                    # Lower and upper limits of the arm joints in that order [rad; m for the slide].
                    "arm_joint_limits": np.stack((tool.arm_lower, tool.arm_upper)),
                    # Tolerances within which the bounded pose IK of this tool meets a target [m, rad].
                    "ik_position_tolerance_m": ik.position_tolerance_m,
                    "ik_orientation_tolerance_rad": ik.orientation_tolerance_rad,
                    "capsule_bodies": list(tool.capsule_bodies),
                    "capsule_endpoints": tool.capsule_endpoints.copy(),
                    "capsule_radii": tool.capsule_radii.copy(),
                    "capsule_parts": list(tool.capsule_parts),
                    "capsule_bounds": tool.capsule_bounds.copy(),
                    "parameters": vars(tool.config),
                }
                for index, (tool, ik) in enumerate(zip(self.graspers, self.pose_ik, strict=True))
            ],
            "rigid_body_labels": list(self.model.body_label),
            "rigid_body_rest_poses": self.rigid_rest_poses.copy(),
            "thread_point_count": len(self.worker_thread.positions()),
            "thread_radius": self.worker_config.thread_radius_m,
            # Whether the thread follows the needle's track through the tissue (--thread-tracks), and the
            # length of one thread element [m]: a tail that is taken through the tissue is clear of it
            # when the element that hangs on it is.
            "thread_tracks": bool(self.worker_thread.thread_tracks),
            "thread_element_length": self.worker_config.thread_length_m / self.worker_config.thread_elements,
            # Radius of the hole that the needle makes in the surface [m]: the needle/tissue interaction
            # takes a point of the surface within it of the needle's track for the opening of that track.
            "needle_capture_radius": self.interaction.capture_radius,
            # Weight that every step of this scene applies to the needle [N] (_needle_weight): its mass, read
            # once at construction, times NEEDLE_GRAVITY; zero for a scene that applies none
            # (SutureSceneConfig.needle_weight). A thread that pulls the needle with less is slack for the
            # tool that holds the needle.
            "needle_weight_N": NEEDLE_GRAVITY * self.needle_mass_kg if self._weighs_needle() else 0.0,
            "grasp_physics": "independent free needle; native MuJoCo finite hinge actuation and frictional capsule contacts; fixed-trocar shaft pivot/insertion/roll and wrist joint actuators with bounded IK; no weld or needle pose drive",
        }
