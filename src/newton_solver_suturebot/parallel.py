# SPDX-License-Identifier: Apache-2.0
"""Independent, action-controlled physical needle/FEM scenes in spawned workers.

Each process owns its MuJoCo model, Warp arrays, nonlinear solver workspaces,
and puncture history. Commands for every scene are sent before any result is
received, so CPU native integration and CUDA submissions can overlap. This is
process parallelism, not a fused GPU multi-world solver. Both control modes
use the insertion example's accepted-step transaction. ``free_needle`` drives
the needle directly; ``physical_grasp`` drives articulated distal jaws/palms
and advances a worker-owned thread. Neither mode includes full robot arms.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
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
    needle_solver: str = "partitioned"
    control_mode: str = "free_needle"
    jaw_step_rad: float = 0.02
    thread_elements: int = 90
    thread_length_m: float = 0.18
    thread_radius_m: float = 0.00015
    thread_modulus_pa: float = 2.0e6
    # Keep source material, contact, convergence and sampling defaults unless
    # explicitly configured. Every override is included in geometry metadata.
    example_arguments: tuple[str, ...] = ()

    def __post_init__(self):
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
            from ._src.suturebot_layout import load_source_stitch

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

    from newton_solver_suturebot.examples.example_fem_needle_insertion import Example

    if config.control_mode == "physical_grasp":
        from newton_solver_suturebot.grasp_scene import GraspScene

        Example = GraspScene

    class CartesianScene(Example):
        def _manual_target_pose(self):
            return self.cartesian_target.copy(), self.manual_target.copy()

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
        "--needle-solver",
        config.needle_solver,
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


def _observe(scene, env_id, *, include_tissue=True):
    import warp as wp

    from newton_solver_suturebot.examples.example_fem_needle_insertion import _marker_positions

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
    host_state = {name: getattr(state, name).numpy() for name in ("body_q", "body_qd")}
    needle_pose = host_state["body_q"][scene.needle_body]
    result = {
        "env_id": env_id,
        "pid": os.getpid(),
        "reset_count": scene.reset_count,
        "needle_pose": needle_pose.copy(),
        "needle_velocity": host_state["body_qd"][scene.needle_body].copy(),
        "tip_position": np.asarray(
            wp.transform_point(wp.transform(*needle_pose), wp.vec3(*scene.geometry.tip)), dtype=float
        ),
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
        result.update(scene.observe_grasp(host_state=host_state))
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


def _worker(connection, config, env_id, log_path):
    # Bound BLAS pools before importing native physics modules in this worker.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    scene = None
    log = None
    try:
        if log_path is not None:
            log = open(log_path, "a", buffering=1)
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
        scene = _make_scene(config, env_id)
        connection.send((True, _observe(scene, env_id)))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            started = time.perf_counter_ns()
            if command == "step":
                action, include_tissue = payload
                _apply_action(scene, config, action)
                scene.step()
                result = _observe(scene, env_id, include_tissue=include_tissue)
                # Histories used by interaction events remain intact; this
                # viewer-only trail otherwise grows without bound in training.
                scene.actual_tip_history.clear()
                result["worker_started_ns"] = started
                result["worker_finished_ns"] = time.perf_counter_ns()
            elif command == "observe":
                result = _observe(scene, env_id, include_tissue=payload)
            elif command == "reset":
                _reset_scene(scene, config, rebuild=True)
                result = _observe(scene, env_id)
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
        if scene is not None:
            scene.viewer.close()
        connection.close()
        if log is not None:
            log.close()


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

    def step(self, actions, *, include_tissue=True):
        """Advance every scene; omit full tissue arrays for compact teacher observations."""
        actions = np.asarray(actions, dtype=float)
        shape = (self.num_envs, self.config.action_dimension)
        if actions.shape != shape or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
            raise ValueError(f"actions must have shape {shape} and finite values in [-1, 1]")
        payloads = [(action, include_tissue) for action in actions]
        return self._request("step", payloads, list(range(self.num_envs)))

    def reset(self, env_ids=None):
        """Reset selected scenes completely and return their states in requested order."""
        ids = self._ids(env_ids)
        return self._request("reset", [None] * len(ids), ids)

    def observe(self, env_ids=None, *, include_tissue=True):
        """Read accepted state; compact mode gathers only material landmarks on device."""
        ids = self._ids(env_ids)
        return self._request("observe", [include_tissue] * len(ids), ids)

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
