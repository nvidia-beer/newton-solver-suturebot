# SPDX-License-Identifier: Apache-2.0
"""Insert a physically grasped needle with articulated dVRK tools, FEM tissue and thread.

The tools pivot at fixed trocar ports and use bounded joint IK with finite
MuJoCo actuators. Jaw friction holds the free rigid needle. The procedural
controller follows one entry/exit arc; no trained policy or Isaac Lab is used.
"""

from __future__ import annotations

import json
import math
import weakref

import newton
import newton.examples
import numpy as np
import warp as wp

from newton_solver_suturebot.examples._grasp_procedure import ProceduralGraspController
from newton_solver_suturebot.parallel import ParallelSutureScenes, SutureSceneConfig


class Example:
    """A normal Newton viewer over independently owned physical scenes."""

    _instances = weakref.WeakSet()

    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        self.sim_time, self.completed = 0.0, 0
        self.last_phases = None
        self.config = SutureSceneConfig(
            device=str(wp.get_device(args.device)), control_mode="physical_grasp", substeps=args.substeps
        )
        self.pool = ParallelSutureScenes(self.config, num_envs=args.num_envs)
        self._cleanup = weakref.finalize(self, self.pool.close)
        self._instances.add(self)
        try:
            self.observations, self.geometry = self.pool.observe(), self.pool.geometry()
            self.controller = (
                ProceduralGraspController(self.geometry, self.config, args.speed, args.settle_seconds)
                if args.controller == "procedural"
                else None
            )
            self.render_scene = None
            if args.viewer != "null":
                self._build_view()
            print("Space: pause/resume | .: single step | Reset: restart | Esc: close", flush=True)
        except BaseException:
            self.close()
            raise

    def _build_view(self):
        from newton_solver_suturebot.examples._grasp_render import PhysicalSurgicalRenderScene
        from newton_solver_suturebot.examples._grasp_view import fit_inspection_camera, framing_points, update_scene

        builder = newton.ModelBuilder(up_axis="Z")
        self.render_scene = PhysicalSurgicalRenderScene(builder, self.geometry, self.args.num_envs, self.config.device)
        self.model = builder.finalize(device=self.config.device)
        self.state_0 = self.model.state()
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((self.args.spacing, self.args.spacing, 0.0))
        self.viewer.picking_enabled = False
        self.viewer.show_particles = False
        world_offsets = self.viewer.world_offsets.numpy()
        self.particle_offsets = world_offsets[self.model.particle_world.numpy()]
        update_scene(self.render_scene, self.state_0, self.observations, particle_offsets=self.particle_offsets)
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
            self.viewer.camera.fov = math.degrees(2 * math.atan(15.2908 / 24.0))
            eye, target = fit_inspection_camera(points, view_direction=(0.0, -1.0, 0.65))
            self.viewer.set_camera(wp.vec3(eye))
            self.viewer.camera.look_at(target)

    def step(self):
        actions = (
            self.controller(self.observations)
            if self.controller is not None
            else np.zeros((self.args.num_envs, self.config.action_dimension), dtype=np.float32)
        )
        if self.controller is not None:
            progress = self.controller.report()
            phases = tuple((item["phase"], item["reason"]) for item in progress)
            if phases != self.last_phases:
                print(
                    "Procedural insertion: " + "; ".join(f"{item['env_id']}: {item['phase']}" for item in progress),
                    flush=True,
                )
                self.last_phases = phases
            if any(item["phase"] == "stopped" for item in progress):
                raise RuntimeError(f"Procedural insertion stopped: {progress}")
        previous = [item["diagnostics"]["physical_time_s"] for item in self.observations]
        self.observations = self.pool.step(actions)
        for before, item in zip(previous, self.observations, strict=True):
            diagnostics = item["diagnostics"]
            if diagnostics.get("failure_reason") or diagnostics.get("task_status") == "failed":
                raise RuntimeError(f"Physical scene failed: {diagnostics}")
            if not np.isclose(diagnostics["physical_time_s"] - before, self.config.frame_dt, rtol=0, atol=1e-9):
                raise RuntimeError("A physical scene did not advance the complete requested frame")
        self.sim_time = float(self.observations[0]["diagnostics"]["physical_time_s"])
        self.completed += 1

    def render(self):
        if self.render_scene is not None:
            from newton_solver_suturebot.examples._grasp_view import update_scene

            update_scene(self.render_scene, self.state_0, self.observations, particle_offsets=self.particle_offsets)
        self.viewer.begin_frame(self.sim_time)
        if self.render_scene is not None:
            self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Check the physical worker states, including tetrahedra and boundaries."""
        if not self.completed:
            raise AssertionError("No physical steps completed")
        rest = self.geometry["tissue_rest_positions"]
        tets = self.geometry["tissue_tetrahedra"]
        fixed = self.geometry["fixed_particle_indices"]

        def determinants(points):
            vertices = points[tets]
            return np.linalg.det(vertices[:, 1:] - vertices[:, :1])

        original = determinants(rest)
        for item in self.observations:
            for name in ("tissue_positions", "rigid_body_poses", "thread_positions"):
                if not np.isfinite(item[name]).all():
                    raise AssertionError(f"Nonfinite {name}")
            if np.any(determinants(item["tissue_positions"]) / original <= 0):
                raise AssertionError("Tissue tetrahedron inverted")
            np.testing.assert_allclose(item["tissue_positions"][fixed], rest[fixed], rtol=0, atol=1e-8)
            if not item["diagnostics"]["independent_free_needle"] or item["diagnostics"]["needle_pose_drive"]:
                raise AssertionError("The needle must be an independently simulated rigid body")
        default_procedure = self.args.speed == 0.003 and self.args.settle_seconds == 0.5
        if self.controller is not None and default_procedure and self.sim_time >= 6.0 - 1e-8:
            for item in self.controller.report():
                if not item["commanded_arc_complete"] or min(item["entry_count"], item["exit_count"]) < 1:
                    raise AssertionError(f"Insertion did not complete an entry and exit: {item}")
        print(
            json.dumps(
                {"physical_checks_passed": True, "num_envs": self.args.num_envs, "physical_time_s": self.sim_time}
            )
        )

    def close(self):
        self._cleanup()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.description = __doc__
        parser.set_defaults(num_frames=360)
        parser.add_argument("--num-envs", type=int, default=1, help="Independent scenes; start with one, then scale.")
        parser.add_argument("--substeps", type=int, default=4, help="Coupled physics intervals per 60 Hz frame.")
        parser.add_argument("--controller", choices=("procedural", "hold"), default="procedural")
        parser.add_argument("--speed", type=float, default=0.003, help="Needle arc speed [m/s].")
        parser.add_argument("--settle-seconds", type=float, default=0.5, help="Closed-jaw settling time [s].")
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
