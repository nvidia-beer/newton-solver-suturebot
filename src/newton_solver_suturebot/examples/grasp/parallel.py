# SPDX-License-Identifier: Apache-2.0
"""Action-controlled physical needle/FEM scenes: one in the caller's process, or several in workers.

:class:`LocalSutureScene` runs one scene in the calling process and CUDA context.
:class:`ParallelSutureScenes` runs independent scenes in spawned workers: each process
owns its MuJoCo model, Warp arrays, nonlinear solver workspaces, and puncture history.
Commands for every scene are sent before any result is received, so CPU native
integration and CUDA submissions can overlap. This is process parallelism, not a
fused GPU multi-world solver. Both control modes
use the insertion example's accepted-step transaction. ``free_needle`` drives
the needle directly; ``physical_grasp`` drives articulated distal jaws/palms
and advances a worker-owned thread. Neither mode includes full robot arms.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
import traceback
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SutureSceneConfig:
    """Physical workload shared by workers; distances are metres, angles radians.

    Actions are bounded normalized Cartesian target increments in world axes,
    ``[dx, dy, dz, rx, ry, rz]``. They change the needle target in
    ``free_needle`` or the active palm target in ``physical_grasp``; the latter
    appends a seventh jaw-opening increment. These finite-gain commands never
    overwrite simulated poses. A zero action holds all previous actuator
    targets. The task layer interprets accepted states and puncture events.
    """

    device: str = "cuda:0"
    asset: str | None = None
    region: int = 1
    entry_xy: tuple[float, float] | None = None
    exit_xy: tuple[float, float] | None = None
    substeps: int = 4
    frame_dt: float = 1.0 / 60.0
    translation_step: float = 0.0001
    rotation_step: float = 0.01
    reset_position_noise_m: float = 0.0
    seed: int = 0
    control_mode: str = "free_needle"
    rigid_backend: str = "native"
    # The released needle rests where it is: its weight is carried as that of the tool links is.
    needle_weight: bool = True
    # Reset grasp of the needle: fraction of the arc from the tip, twist about the needle
    # tangent [rad] and bite angle [rad]. None keeps the settings of the recorded stitch.
    initial_grasp: tuple[float, float, float] | None = None
    jaw_step_rad: float = 0.02
    thread_elements: int = 90
    thread_length_m: float = 0.18
    thread_radius_m: float = 0.00015
    thread_modulus_pa: float = 2.0e6
    # Keep source material, contact, convergence and sampling defaults unless
    # explicitly configured. Every override is included in geometry metadata.
    example_arguments: tuple[str, ...] = ()

    def __post_init__(self):
        if self.rigid_backend not in ("native", "mujoco-warp"):
            raise ValueError("rigid_backend must be native or mujoco-warp")
        if self.rigid_backend == "mujoco-warp" and (
            self.control_mode != "physical_grasp" or not str(self.device).startswith("cuda")
        ):
            raise ValueError("mujoco-warp currently requires CUDA physical_grasp scenes")
        # The physical tools use the recorded trocar layout. Its annotated
        # stitch belongs on the same side of the parked forceps; the older
        # free-needle demo at x=0 would drive PSM1 across that tool's workspace.
        defaults = ((0.0, -0.046), (0.0, -0.054))
        if (
            self.control_mode == "physical_grasp"
            and self.asset is None
            and self.region == 1
            and (self.entry_xy is None or self.exit_xy is None)
        ):
            from ._dvrk_layout import load_source_stitch

            stitch = load_source_stitch()
            defaults = (tuple(stitch["entry"][:2]), tuple(stitch["exit"][:2]))
        for name, default in zip(("entry_xy", "exit_xy"), defaults, strict=True):
            if getattr(self, name) is None:
                object.__setattr__(self, name, default)
        if self.substeps < 1 or self.region < 1:
            raise ValueError("substeps and region must be positive")
        values = (self.frame_dt, self.translation_step, self.rotation_step)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Frame timestep and action scales must be finite and positive")
        for point in (self.entry_xy, self.exit_xy):
            if np.asarray(point).shape != (2,) or not np.isfinite(point).all():
                raise ValueError("Entry and exit coordinates must each contain two finite values")
        if not np.isfinite(self.reset_position_noise_m) or self.reset_position_noise_m < 0:
            raise ValueError("Reset position noise must be finite and nonnegative")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.control_mode not in ("free_needle", "physical_grasp"):
            raise ValueError("control_mode must be free_needle or physical_grasp")
        values = (self.jaw_step_rad, self.thread_length_m, self.thread_radius_m, self.thread_modulus_pa)
        if not np.isfinite(values).all() or min(values) <= 0 or self.thread_elements < 2:
            raise ValueError("Physical grasp jaw/thread dimensions and material must be positive")

    @property
    def action_dimension(self):
        return 7 if self.control_mode == "physical_grasp" else 6


def _make_scene(config, env_id):
    # Imports stay in children: parent does not own a CUDA context, and spawn
    # never inherits an initialized native solver/CUDA context through fork.
    import warp as wp
    from newton.viewer import ViewerNull

    from newton_solver_suturebot.examples.needle.example_fem_needle_insertion import Example

    if config.control_mode == "physical_grasp":
        from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene

        Example = GraspScene

    class CartesianScene(Example):
        rigid_backend = config.rigid_backend
        tip_poses = None

        def _manual_target_pose(self):
            return self.cartesian_target.copy(), self.manual_target.copy()

        def actual_tip(self):
            # A frame ends with a tip sample, and the worker observes the same accepted poses
            # right after it. Keep what this sample read, with the array and the accepted-step
            # count it belongs to, so that _step can hand it to that observation.
            poses = self.state_0.body_q.numpy()
            self.tip_poses = (self.state_0.body_q, self.accepted_physics_steps, poses)
            return _tip_position(self, poses[self.needle_body])

        def _observe_interaction(self):
            previous = len(self.interaction_events)
            super()._observe_interaction()
            if len(self.interaction_events) > previous:
                # Material targets move with FEM. Capture their error at the
                # accepted event, rather than comparing an old world-space
                # event with a subsequently deformed target during training.
                positions = self.state_0.particle_q.numpy()
                entry = self.entry.position(positions)
                exit_point = self.exit.position(positions)
                for event in self.interaction_events[previous:]:
                    position = np.asarray(event["position_m"])
                    event["entry_error_m"] = float(np.linalg.norm(position - entry))
                    event["exit_error_m"] = float(np.linalg.norm(position - exit_point))

    wp.init()
    wp.set_device(config.device)
    arguments = [
        "--regions",
        str(config.region),
        "--substeps",
        str(config.substeps),
    ]
    if config.asset is not None:
        arguments.extend(("--asset", config.asset))
    arguments.extend(config.example_arguments)
    args = CartesianScene.create_parser().parse_args(arguments)
    scene = CartesianScene(ViewerNull(), args)
    scene.worker_config = config
    scene.frame_dt = config.frame_dt
    scene.sim_dt = config.frame_dt / args.substeps
    scene.reset_rng = np.random.default_rng(np.random.SeedSequence([config.seed, env_id]))
    scene.reset_count = -1
    _reset_scene(scene, config)
    return scene


def _reset_scene(scene, config, *, rebuild=False):
    import warp as wp

    from newton_solver_suturebot._src.utils import capture

    # Everything from here to the mark is start-up: graph captures and kernel module loads after
    # the mark are reported as late ones (the scene's report()).
    scene._startup_begun = capture.begin_startup()
    if rebuild:
        scene.reset()
    scene.select_point_xy("entry", *config.entry_xy)
    scene.select_point_xy("exit", *config.exit_xy)
    scene.preview()
    if config.reset_position_noise_m:
        offset = scene.reset_rng.uniform(-config.reset_position_noise_m, config.reset_position_noise_m, 3)
        # Initialization may move the needle only before time advances. Move
        # upward along Z so randomized starts remain above the tissue.
        offset[2] = abs(offset[2])
        pose = scene.desired_pose.copy()
        pose[:3] += offset
        scene._initialize_pose(pose)
        scene.desired_pose = pose
    scene.cartesian_target = (
        scene.state_0.body_q.numpy()[scene.graspers[0].palm_body].astype(float)
        if config.control_mode == "physical_grasp"
        else scene.desired_pose.copy()
    )
    scene.use_manual_control()
    if config.control_mode == "physical_grasp":
        scene.initialize_worker_thread(config)
    scene.actual_tip_history.clear()
    scene.reset_count += 1
    scene.observation_nodes = wp.array(
        [scene.entry.node_indices, scene.exit.node_indices], dtype=wp.vec3i, device=scene.device
    )
    scene.observation_weights = wp.array([scene.entry.weights, scene.exit.weights], dtype=wp.vec3, device=scene.device)
    scene.observation_landmarks = wp.empty(2, dtype=wp.vec3, device=scene.device)
    # The scene's start-up call captures, before the first frame, graphs that the run would capture.
    # Captures and loads after start-up are checked for a scene whose call ran; a scene that
    # declines says why in its status.
    scene._startup_status = "start-up not marked"
    scene._startup_ready = scene.capture_graphs()
    _mark_startup(scene)


def _mark_startup(scene):
    """End the start-up of a scene here; a later call moves the mark and keeps the start-up's beginning.

    The reset marks at its end. The driver of a scene marks again after its first observation,
    which launches kernels of its own, and an example with a view once more after it has drawn
    the start state. A scene that no reset of this module prepared is left alone.
    """
    begun = getattr(scene, "_startup_begun", None)
    if begun is None:
        return
    from newton_solver_suturebot._src.utils import capture

    mark = capture.finish_startup(begun, scene.device) if scene._startup_ready else None
    if scene._startup_ready and mark is None:
        scene._startup_status = "CPU device"
    scene._startup_mark = mark
    scene.refresh_capture_census()


def _refresh_census(scene):
    """Bring the scene's numbers of late kernel module loads up to date: one walk of the loaded modules."""
    refresh = getattr(scene, "refresh_capture_census", None)
    if refresh is not None:
        refresh()


def _apply_action(scene, config, action):
    import warp as wp

    action = np.asarray(action, dtype=float)
    if action.shape != (config.action_dimension,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise ValueError(f"Each action must be {config.action_dimension} finite normalized values in [-1, 1]")
    target = scene.cartesian_target.copy()
    target[:3] += config.translation_step * action[:3]
    rotation = config.rotation_step * action[3:6]
    angle = float(np.linalg.norm(rotation))
    if angle:
        delta = wp.quat_from_axis_angle(wp.vec3(*(rotation / angle)), angle)
        quat = np.asarray(delta * wp.quat(*target[3:]), dtype=float)
        target[3:] = quat / np.linalg.norm(quat)
    scene.cartesian_target = target
    if config.control_mode == "physical_grasp":
        scene.jaw_opening_target = float(
            np.clip(
                scene.jaw_opening_target + config.jaw_step_rad * action[6],
                0,
                scene.graspers[0].config.maximum_opening_rad,
            )
        )


def _tip_position(scene, needle_pose):
    import warp as wp

    return np.asarray(wp.transform_point(wp.transform(*needle_pose), wp.vec3(*scene.geometry.tip)), dtype=float)


def _step(scene, deadline=None, inside=False):
    """Advance one frame and return the accepted rigid poses its last tip sample read, or None.

    The poses are returned only when that sample read the array that holds the accepted state
    after the frame and no interval was accepted after it; the scene writes the accepted poses
    nowhere else during a frame. The caller observes before anything else touches the scene.

    With a ``deadline`` (a time of ``time.perf_counter()`` [s]) the scene may leave the frame
    unfinished (``scene.frame_pending``); the next call goes on with it. A frame that is not
    finished has taken no tip sample, so nothing is returned for it. With ``inside`` the scene
    may also return inside an attempt, between two host-controlled trials.
    """
    scene.tip_poses = None
    if inside:
        scene.step(deadline, inside=True)
    elif deadline is None:
        scene.step()
    else:
        scene.step(deadline)
    sample, scene.tip_poses = scene.tip_poses, None
    if sample is not None:
        array, accepted_steps, poses = sample
        if array is scene.state_0.body_q and accepted_steps == scene.accepted_physics_steps:
            return poses
    return None


def _observe(scene, env_id, *, include_tissue=True, include_thread_diagnostics=True, body_q=None):
    """Read the accepted state; ``body_q`` is a host copy of the accepted poses that the caller holds."""
    import warp as wp

    from newton_solver_suturebot.examples.needle.example_fem_needle_insertion import _marker_positions

    state = scene.state_0
    if include_tissue:
        positions = state.particle_q.numpy().copy()
        entry, exit_point = scene.entry.position(positions), scene.exit.position(positions)
    else:
        wp.launch(
            _marker_positions,
            2,
            [scene.observation_nodes, scene.observation_weights, state.particle_q, scene.observation_landmarks],
            device=scene.device,
        )
        entry, exit_point = scene.observation_landmarks.numpy()
    # Each accepted rigid array crosses to the host once. All observation
    # fields below describe this same snapshot; none requests another readback.
    host_state = {"body_q": state.body_q.numpy() if body_q is None else body_q, "body_qd": state.body_qd.numpy()}
    needle_pose = host_state["body_q"][scene.needle_body]
    result = {
        "env_id": env_id,
        "pid": os.getpid(),
        "reset_count": scene.reset_count,
        "needle_pose": needle_pose.copy(),
        "needle_velocity": host_state["body_qd"][scene.needle_body].copy(),
        "tip_position": _tip_position(scene, needle_pose),
        "target_pose": scene.cartesian_target.copy(),
        "entry_position": entry.copy(),
        "exit_position": exit_point.copy(),
        "diagnostics": scene.report(),
    }
    if include_tissue:
        result["tissue_positions"] = positions
        result["tissue_velocities"] = state.particle_qd.numpy().copy()
    if hasattr(scene, "graspers"):
        host_state.update({name: getattr(state, name).numpy() for name in ("joint_q", "joint_qd")})
        result.update(scene.observe_grasp(host_state=host_state, include_thread_diagnostics=include_thread_diagnostics))
    return result


def _geometry(scene, config):
    result = {
        "tissue_rest_positions": scene.selection.vertices.copy(),
        "tissue_tetrahedra": scene.selection.indices.copy(),
        "tissue_triangles": scene.selection.exterior_surface_triangles.copy(),
        "fixed_particle_indices": scene.fixed_vertex_indices.copy(),
        "needle_centerline": scene.geometry.centerline.copy(),
        "needle_radius": scene.geometry.thickness,
        "needle_span": scene.geometry.span,
        "needle_tip_local": scene.geometry.tip.copy(),
        "ready_pose": scene.plan.pose(scene.plan.initial_angle),
        "entry_position": scene.entry.reference_position.copy(),
        "exit_position": scene.exit.reference_position.copy(),
        "entry_normal": scene.entry.normal.copy(),
        "exit_normal": scene.exit.normal.copy(),
        "insertion_plan": {
            "center": scene.plan.center.copy(),
            "axis": scene.plan.axis.copy(),
            "radial": scene.plan.radial.copy(),
            "tangent": scene.plan.tangent.copy(),
            "local_basis": scene.plan.local_basis.copy(),
            "arc_center_local": scene.geometry.arc_center.copy(),
            "radius": scene.geometry.radius,
            "initial_angle": scene.plan.initial_angle,
            "exit_angle": scene.plan.exit_angle,
            "finish_angle": scene.plan.finish_angle,
            "expected_passages": scene.plan.expected_passages,
        },
        "reference_positions": scene.reference.vertices.copy(),
        "reference_triangles": scene.reference.exterior_surface_triangles.copy(),
        # Preserve node identity across coincident region boundaries. Renderers
        # must deform the full reference surface without guessing by position.
        "active_reference_indices": np.searchsorted(
            scene.reference.global_vertex_indices, scene.selection.global_vertex_indices
        ),
        "config": config,
        "diagnostics": scene.report(),
        "scope": "FEM tissue + freely simulated MuJoCo needle; no thread or articulated robot",
        "control_mode": config.control_mode,
    }
    if config.control_mode == "physical_grasp":
        result.update(scene.grasp_geometry())
    return result


@contextmanager
def _worker_thread_limits():
    """Limit loaded BLAS pools and later imports only inside this child."""
    # NumPy is already imported when spawn resolves this module, so changing
    # its environment alone cannot resize the BLAS pool it already loaded.
    from threadpoolctl import threadpool_limits

    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        with ExitStack() as limits:

            def refresh():
                # A new controller also finds BLAS libraries loaded by scene
                # setup. Keep earlier limits alive until all cleanup finishes.
                limits.enter_context(threadpool_limits(limits=1, user_api="blas"))

            refresh()
            yield refresh
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _worker(connection, config, env_id, log_path):
    scene = None
    log = None
    limits = ExitStack()
    try:
        refresh_threads = limits.enter_context(_worker_thread_limits())
        if log_path is not None:
            log = open(log_path, "a", buffering=1)
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
        scene = _make_scene(config, env_id)
        refresh_threads()
        # Start-up ends after the first observation, which launches kernels that no reset launches.
        result = _observe(scene, env_id)
        _mark_startup(scene)
        connection.send((True, result))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            started = time.perf_counter_ns()
            if command == "step":
                action, include_tissue, include_thread_diagnostics = payload
                _apply_action(scene, config, action)
                body_q = _step(scene)
                result = _observe(
                    scene,
                    env_id,
                    include_tissue=include_tissue,
                    include_thread_diagnostics=include_thread_diagnostics,
                    body_q=body_q,
                )
                # Histories used by interaction events remain intact; this
                # viewer-only trail otherwise grows without bound in training.
                scene.actual_tip_history.clear()
                result["worker_started_ns"] = started
                result["worker_finished_ns"] = time.perf_counter_ns()
            elif command == "observe":
                include_tissue, include_thread_diagnostics = payload
                # An explicit observation reports the kernel module loads up to now; a step does not.
                _refresh_census(scene)
                result = _observe(
                    scene, env_id, include_tissue=include_tissue, include_thread_diagnostics=include_thread_diagnostics
                )
            elif command == "reset":
                _reset_scene(scene, config, rebuild=True)
                result = _observe(scene, env_id)
                _mark_startup(scene)
            elif command == "geometry":
                result = _geometry(scene, config)
            else:
                raise ValueError(f"Unknown worker command: {command}")
            connection.send((True, result))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send((False, traceback.format_exc()))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        try:
            if scene is not None:
                try:
                    if hasattr(scene, "close"):
                        scene.close()
                finally:
                    scene.viewer.close()
        finally:
            try:
                connection.close()
            finally:
                try:
                    if log is not None:
                        log.close()
                finally:
                    limits.close()


def _snapshot(value):
    """The copy a worker's pipe would deliver: later steps of the scene cannot change it."""
    return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


class LocalSutureScene:
    """One physical scene in the calling process, with the interface of :class:`ParallelSutureScenes`.

    No worker is spawned: the scene shares the caller's process and CUDA context, and no
    observation crosses a pipe. Results are lists of one scene, as for ``num_envs=1``.
    Observations are snapshots; they do not alias the scene's arrays. A failure of the
    scene raises its own exception here. The single-thread BLAS limit a worker sets for
    itself applies to the calling process until :meth:`close`.
    """

    num_envs = 1

    def __init__(self, config=None):
        self.config = SutureSceneConfig() if config is None else config
        self._closed = False
        self._scene = None
        # Host-clock time [s] that the last step took after its scene had stepped (observation and
        # copy), and the longest such time since the scene was made or reset: a step with a
        # deadline leaves the longest free, the worst case measured, not the last.
        self._observe_seconds = 0.0
        self._observe_longest = 0.0
        self._limits = ExitStack()
        try:
            refresh_threads = self._limits.enter_context(_worker_thread_limits())
            self._scene = _make_scene(self.config, 0)
            refresh_threads()
            # As a worker: start-up ends after the first observation.
            result = _observe(self._scene, 0)
            _mark_startup(self._scene)
            self._last = [_snapshot(result)]
        except BaseException:
            self.close()
            raise

    def _open_scene(self, env_ids=None):
        if self._closed:
            raise RuntimeError("The scene is closed")
        if env_ids is not None and list(env_ids) != [0]:
            raise ValueError("env_ids must be [0]: there is one scene")
        return self._scene

    @property
    def frame_unfinished(self):
        """Whether the last :meth:`step` had a deadline and returned before its frame was complete."""
        return self._scene is not None and getattr(self._scene, "frame_pending", False)

    def step(
        self,
        actions,
        *,
        include_tissue=True,
        include_thread_diagnostics=True,
        deadline=None,
        unit="attempt",
        observe_native_trials=False,
    ):
        """Advance the scene by one frame; see :meth:`ParallelSutureScenes.step`.

        ``deadline`` (default None: the whole frame, always) is a time of the host clock
        ``time.perf_counter()`` [s] by which the call should have returned. The scene then runs
        one attempt of a coupled step in any case and a further one, the next substep or the
        retry of a rejected attempt, only if a substep that uses up its trials, and the
        observation after it, end before the deadline (the scene's ``simulate``): each as long
        as the longest that this scene has measured since it was made or reset. If the frame
        is not complete then, :attr:`frame_unfinished` is true and the result is the accepted
        state after the substeps that ran. The next call must pass ``actions=None``: it goes
        on with the same frame, under the action that started it, with or without a deadline
        of its own. The accepted states of the frame, and everything that follows from them,
        are those of one call without a deadline; only the calls they fall into differ. An
        attempt that alone takes longer than the time left is not interrupted. A reset drops
        an unfinished frame.

        ``unit`` is the smallest part of a frame after which a call with a deadline may return:
        ``"attempt"`` (default; everything above) or ``"host-trial"``. With ``"host-trial"`` the
        call may also return inside an attempt, between two host-controlled trials (the trials
        of a loop that the host controls at a needle surface event), when the longest such
        trial and the longest end of an attempt that the scene has measured do not end by the
        deadline. The attempt stays open, :attr:`frame_unfinished` is true, and the next call
        (``actions=None``) goes on with that trial. The result of such a call is the accepted
        state of before the attempt, with the report entries that describe the last attempt as
        they were before it began; the commands of the attempt in progress (the arm joint
        targets) are the new ones. A device trial loop is not interrupted. A reset and
        :meth:`close` end an open attempt as a rejected attempt ends. On the native rigid
        backend the rigid data hold a trial while an attempt is open, and the observation reads
        the jaw contact force and the native clock from them: ``"host-trial"`` is refused there
        with a ``ValueError`` unless ``observe_native_trials`` is true (tests).
        """
        scene = self._open_scene()
        if unit not in ("attempt", "host-trial"):
            raise ValueError("unit must be 'attempt' or 'host-trial'")
        if unit == "host-trial" and not observe_native_trials:
            from ..._src.solvers.mujoco.experimental_rigid_backend import is_gpu_backend

            if not is_gpu_backend(getattr(scene, "rigid_solver", None)):
                raise ValueError(
                    "unit 'host-trial' on the native rigid backend: an observation inside an attempt would read "
                    "the contacts and the clock of a trial"
                )
        if self.frame_unfinished:
            if actions is not None:
                raise ValueError("A frame is unfinished: go on with it by passing actions=None")
        else:
            actions = np.asarray(actions, dtype=float)
            shape = (1, self.config.action_dimension)
            if actions.shape != shape or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
                raise ValueError(f"actions must have shape {shape} and finite values in [-1, 1]")
        started = time.perf_counter_ns()
        if actions is not None:
            _apply_action(scene, self.config, actions[0])
        if unit == "attempt":
            body_q = _step(scene) if deadline is None else _step(scene, deadline - self._observe_longest)
        else:
            body_q = _step(scene, None if deadline is None else deadline - self._observe_longest, inside=True)
        stepped = time.perf_counter()
        result = _observe(
            scene,
            0,
            include_tissue=include_tissue,
            include_thread_diagnostics=include_thread_diagnostics,
            body_q=body_q,
        )
        # As in a worker: this viewer-only trail otherwise grows without bound.
        scene.actual_tip_history.clear()
        result["worker_started_ns"] = started
        result["worker_finished_ns"] = time.perf_counter_ns()
        self._last = [_snapshot(result)]
        self._observe_seconds = time.perf_counter() - stepped
        self._observe_longest = max(self._observe_longest, self._observe_seconds)
        return self._last

    def reset(self, env_ids=None):
        """Rebuild the scene completely, including solver warm starts and puncture history."""
        scene = self._open_scene(env_ids)
        abandon = getattr(scene, "abandon_attempt", None)
        if abandon is not None:
            # An attempt that a call left open ends before the rebuild, as a rejected attempt ends.
            abandon()
        _reset_scene(scene, self.config, rebuild=True)
        # The rebuilt scene measures its substeps anew; so does the observation after its steps.
        self._observe_longest = 0.0
        result = _observe(scene, 0)
        _mark_startup(scene)
        self._last = [_snapshot(result)]
        return self._last

    def finish_startup(self):
        """End the scene's start-up now: what the caller ran in this process since then belongs to it.

        The scene shares the caller's process, so kernels that the caller launches (a viewer's)
        load in it. Graph captures and kernel module loads after this call are reported as late
        ones in ``diagnostics``.
        """
        _mark_startup(self._open_scene())

    def observe(self, env_ids=None, *, include_tissue=True, include_thread_diagnostics=True):
        """Read the accepted state; compact mode gathers only material landmarks on device.

        The numbers of late kernel module loads in ``diagnostics`` are brought up to date here;
        :meth:`step` reports those of the last observation.
        """
        scene = self._open_scene(env_ids)
        _refresh_census(scene)
        self._last = [
            _snapshot(
                _observe(scene, 0, include_tissue=include_tissue, include_thread_diagnostics=include_thread_diagnostics)
            )
        ]
        return self._last

    def geometry(self, env_id=0):
        """Return static render/validation geometry and exact physical configuration."""
        return _snapshot(_geometry(self._open_scene([env_id]), self.config))

    def close(self):
        if self._closed:
            return
        self._closed = True
        scene, self._scene = self._scene, None
        try:
            if scene is not None:
                try:
                    try:
                        abandon = getattr(scene, "abandon_attempt", None)
                        if abandon is not None:
                            # An attempt that a call left open ends first, as a rejected attempt ends.
                            abandon()
                    finally:
                        # Also when that abandon raised: the scene still releases what it holds.
                        if hasattr(scene, "close"):
                            scene.close()
                finally:
                    scene.viewer.close()
        finally:
            self._limits.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ParallelSutureScenes:
    """Synchronous vector interface over concurrently executing physical scenes.

    Creation must be under a ``if __name__ == '__main__'`` guard in scripts.
    Use as a context manager or call :meth:`close`. Startup first initializes
    one scene to populate compiler caches, then initializes the remainder
    concurrently. Resets rebuild only selected scenes, including solver warm
    starts and material puncture history; unselected state is untouched.
    """

    def __init__(self, config=None, *, num_envs=8, timeout=600.0, log_dir=None):
        if not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.config = SutureSceneConfig() if config is None else config
        self.num_envs, self.timeout = num_envs, float(timeout)
        self._connections, self._processes = [], []
        self._closed = False
        self._last = [None] * num_envs
        directory = None if log_dir is None else Path(log_dir)
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        context = mp.get_context("spawn")
        try:
            for env_id in range(num_envs):
                parent, child = context.Pipe()
                log_path = None if directory is None else str(directory / f"worker_{env_id:03d}.log")
                process = context.Process(target=_worker, args=(child, self.config, env_id, log_path), daemon=True)
                process.start()
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
                if env_id == 0:
                    self._last[0] = self._receive(0)
            for env_id in range(1, num_envs):
                self._last[env_id] = self._receive(env_id)
        except BaseException:
            self.close()
            raise

    def _receive(self, env_id):
        connection = self._connections[env_id]
        if not connection.poll(self.timeout):
            raise TimeoutError(f"Scene {env_id} did not respond within {self.timeout:g} seconds")
        try:
            success, result = connection.recv()
        except EOFError as error:
            raise RuntimeError(f"Scene {env_id} process exited unexpectedly") from error
        if not success:
            raise RuntimeError(f"Scene {env_id} failed:\n{result}")
        return result

    def _ids(self, env_ids):
        values = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        if any(not isinstance(value, (int, np.integer)) or not 0 <= value < self.num_envs for value in values):
            raise ValueError("env_ids must be valid integer scene indices")
        if len(set(values)) != len(values):
            raise ValueError("env_ids must not contain duplicates")
        return values

    def _request(self, command, payloads, env_ids):
        if self._closed:
            raise RuntimeError("Parallel scenes are closed")
        for env_id, payload in zip(env_ids, payloads, strict=True):
            self._connections[env_id].send((command, payload))
        results = [self._receive(env_id) for env_id in env_ids]
        if command != "geometry":
            for env_id, result in zip(env_ids, results, strict=True):
                self._last[env_id] = result
        return results

    def step(self, actions, *, include_tissue=True, include_thread_diagnostics=True):
        """Advance every scene; omit full tissue arrays for compact teacher observations.

        ``include_thread_diagnostics=False`` omits ``thread_diagnostics`` of physical-grasp
        scenes, a report that reads 14 device arrays and that only reporting consumes.
        """
        actions = np.asarray(actions, dtype=float)
        shape = (self.num_envs, self.config.action_dimension)
        if actions.shape != shape or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
            raise ValueError(f"actions must have shape {shape} and finite values in [-1, 1]")
        payloads = [(action, include_tissue, include_thread_diagnostics) for action in actions]
        return self._request("step", payloads, list(range(self.num_envs)))

    def reset(self, env_ids=None):
        """Reset selected scenes completely and return their states in requested order."""
        ids = self._ids(env_ids)
        return self._request("reset", [None] * len(ids), ids)

    def finish_startup(self):
        """Nothing to end: each worker ended its own start-up, and a caller's kernels load in another process."""

    def observe(self, env_ids=None, *, include_tissue=True, include_thread_diagnostics=True):
        """Read accepted state; compact mode gathers only material landmarks on device.

        The numbers of late kernel module loads in ``diagnostics`` are brought up to date here;
        :meth:`step` reports those of the last observation.
        """
        ids = self._ids(env_ids)
        return self._request("observe", [(include_tissue, include_thread_diagnostics)] * len(ids), ids)

    def geometry(self, env_id=0):
        """Return static render/validation geometry and exact physical configuration."""
        ids = self._ids([env_id])
        return self._request("geometry", [None], ids)[0]

    def close(self):
        if self._closed:
            return
        self._closed = True
        for connection in self._connections:
            try:
                connection.send(("close", None))
            except (EOFError, BrokenPipeError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for connection in self._connections:
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
