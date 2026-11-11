# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lift a suture needle on its thread with the mouse: a free MuJoCo needle pulls a cable-element thread.

The bundled 0.3 g needle is a free rigid body in MuJoCo, resting on a floor on
its authored collision capsules. A 120 mm thread of 32 cable elements, 0.3 mm
thick, leaves its tail; the thread's far end is pinned to the floor. Needle
and thread are at real size and coupled both ways: the thread solve carries
the needle's mass and inertia on its first node, and the needle receives the
wrench that moves it as that node moved. There is no tissue in this scene.

Right-drag the needle to lift it. Releasing the button leaves a spring hold at
the green marker; a middle click on the marker releases it. The thread works
the same way: a right-drag pins the nearest thread node to the cursor. Clear
holds releases everything.

The stiffness slider changes the thread's Young's modulus between the default
soft thread and solid nylon; it scales the axial and the bending stiffness
together and does not restart the scene. The debug view shows the thread's
nodes with their slope vectors and the needle's collision capsules.

The thread is integrated with backward Euler and at most two Newton updates
per substep, which bounds the cost of a frame; see --updates. A stiff thread
on a moving needle needs more updates to converge; its step is then cut short
and damps the motion more. The thread rests on the floor without friction and
does not collide with itself or with the needle.

Run: python -m newton_solver_suturebot.examples needle_thread
"""

import math

import newton
import newton.examples
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo

from newton_solver_suturebot import NeedleGeometry, SolverThread, ThreadAttachment, ThreadModel
from newton_solver_suturebot.examples._needle_thread_holds import NeedleThreadHolds
from newton_solver_suturebot.examples._screen_pick import ScreenPickTool
from newton_solver_suturebot.examples._thread_view import ThreadView

GRAVITY = 9.81  # [m/s²]


class Example:
    """One needle on one thread; the needle and the thread are held with the mouse."""

    # Young's modulus the slider spans [Pa]: a soft thread to solid nylon.
    modulus_range = (2.0e6, 2.0e9)

    def __init__(self, viewer, args=None):
        self.args = args = newton.examples.default_args(self.create_parser()) if args is None else args
        if args.substeps < 1 or args.elements < 2 or args.updates < 1:
            raise ValueError("substeps and updates must be at least 1, and the thread needs at least 2 elements")
        if not self.modulus_range[0] <= args.young_modulus <= self.modulus_range[1]:
            raise ValueError(
                f"young-modulus must be between {self.modulus_range[0]:g} and {self.modulus_range[1]:g} Pa"
            )
        self.viewer, self.device = viewer, wp.get_device()
        self.frame_dt, self.sim_time = 1.0 / 60.0, 0.0
        self.sim_dt = dt = self.frame_dt / args.substeps
        self.geometry = geometry = NeedleGeometry.load()
        capsules = len(geometry.centerline) - 1

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -GRAVITY))
        SolverMuJoCo.register_custom_attributes(builder)
        # Contact is critically damped with MuJoCo's shortest stable time constant, two steps:
        # time constant 2 / kd and damping ratio kd / (2 sqrt(ke)).
        builder.default_shape_cfg.kd = 2.0 / (2.0 * dt)
        builder.default_shape_cfg.ke = (0.5 * builder.default_shape_cfg.kd) ** 2
        builder.add_ground_plane()
        # The needle lies flat on its collision capsules.
        builder.add_usd(
            str(geometry.asset_path),
            xform=wp.transform(wp.vec3(0.0, 0.0, geometry.thickness), wp.quat_identity()),
            collapse_fixed_joints=False,
            hide_collision_shapes=True,
        )
        self.model = builder.finalize(device=self.device)
        self.needle = list(self.model.body_label).index(geometry.body_path)
        if self.model.body_count != 1:
            raise ValueError("The scene needs exactly one free rigid body, the needle")
        self.mujoco = SolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            use_mujoco_contacts=True,
            solver="newton",
            integrator="implicitfast",
            cone="pyramidal",
            iterations=50,
            ls_iterations=50,
            # A capsule touches the floor at two points; a pyramidal friction contact has four rows.
            njmax=8 * capsules,
            nconmax=2 * capsules,
        )
        self.state_0, self.state_1, self._predicted = self.model.state(), self.model.state(), self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # A suture of real size, straight behind the needle's tail along the axis of its last
        # collision capsule; the far end is pinned in position.
        self.thread_length, self.thread_radius = 0.12, 1.5e-4
        density = 1140.0
        self._tail = geometry.tail
        tangent = geometry.centerline[-1] - geometry.centerline[-2]
        tangent /= np.linalg.norm(tangent)
        start = self._tail + np.array([0.0, 0.0, geometry.thickness])
        line = start + np.linspace(0.0, self.thread_length, args.elements + 1)[:, None] * tangent
        self.thread_model = ThreadModel.create(
            line, young_modulus=args.young_modulus, radius=self.thread_radius, density=density, device=self.device
        )
        self.young_modulus = args.young_modulus
        # Section stiffnesses per unit modulus: E A and E I are both proportional to Young's modulus.
        self._unit_axial = self.thread_model.axial_stiffness.numpy() / args.young_modulus
        self._unit_bending = self.thread_model.bending_stiffness.numpy() / args.young_modulus
        node_weight = density * math.pi * self.thread_radius**2 * self.thread_length / args.elements * GRAVITY
        self.thread = SolverThread(
            self.model,
            self.thread_model,
            fixed_dofs=[6 * args.elements + axis for axis in range(3)],
            mass_damping=2.0,
            # Converged when the force imbalance is a tenth of one node's weight.
            tolerance=0.1 * node_weight,
            # A backward Euler step may stop before it has converged; the cap bounds the frame time.
            max_iterations=args.updates,
            # The floor carries the needle's weight on one node at a tenth of the thread radius.
            contact_distance=self.thread_radius,
            contact_stiffness=geometry.mass * GRAVITY / (0.1 * self.thread_radius),
            contact_smoothing=0.1 * self.thread_radius,
        )
        self.attachment = ThreadAttachment(self.thread, self.model, [self.needle], self._tail, tangent)
        self._pin = line[-1]

        viewer.set_model(self.model)
        if hasattr(viewer, "picking_enabled"):
            # The pick tool below owns the right mouse button.
            viewer.picking_enabled = False
        self.reset_camera()
        center = self.model.body_com.numpy()[self.needle]
        self.holds = NeedleThreadHolds(
            viewer,
            self.model,
            self.needle,
            geometry.sample_centerline(geometry.thickness),
            self.thread,
            lambda: self.state_0,
            dt,
            geometry.mass,
            float(geometry.inertia.min()),
            # No point of the capsule arc is farther from the centre of mass than this.
            float(np.linalg.norm(geometry.arc_center - center) + geometry.radius + geometry.thickness),
            (0.0, 0.0, -GRAVITY),
        )
        self.screen_pick = ScreenPickTool(
            viewer,
            self.model,
            begin=self.holds.begin,
            move=self.holds.move,
            finish=self.holds.finish,
            cancel=self.holds.cancel,
            remove=self.holds.remove,
        )
        self.thread_view = ThreadView(viewer, self.thread, self.thread_radius)
        self.show_nodes, self.show_collision = False, False
        viewer.log_points(
            "/needle_thread/pinned_end",
            wp.array(self._pin[None].astype(np.float32), dtype=wp.vec3, device=self.device),
            radii=4.0 * self.thread_radius,
            colors=wp.full(1, wp.vec3(1.0, 0.45, 0.05), dtype=wp.vec3, device=self.device),
        )

        # In test mode the needle is grabbed at the middle of its arc after a third of a second and lifted
        # at 0.1 m/s, a slow hand, towards the pin: the thread stays slack and the hold reaches its target.
        # Ten frames later the hold lets go, and the needle falls and comes to rest. In those frames the needle
        # swings in behind its hold: it was 0.69 mm from the hold when the lift ended and 0.04 mm at the
        # release with four substeps, and 6.0 mm and 4.0 mm with one.
        self._frame, self._lift_frame, self._scripted, self._held = 0, 20, None, None
        self._grab = geometry.centerline[capsules // 2] + np.array([0.0, 0.0, geometry.thickness])
        self._lift = np.array([0.0, 0.0, 0.04]) + 0.03 * tangent
        self._lift_frames = round(float(np.linalg.norm(self._lift)) / (0.1 * self.frame_dt))
        self._release_frame = self._lift_frame + self._lift_frames + 10
        # The fall takes sqrt(2 h / g) = 0.09 s. Over the slider's range the needle was at rest, slower than
        # 1 mm/s and 0.1 rad/s, within 0.48 s of the release; the landing is tested from half a second on.
        self._rest_frame = self._release_frame + round(0.5 / self.frame_dt)
        self._capture_physics()

    def reset_camera(self):
        self.viewer.set_camera(pos=wp.vec3(0.17, -0.055, 0.1), pitch=-28.0, yaw=180.0)

    def set_young_modulus(self, modulus):
        """Change the thread's Young's modulus [Pa] in the running scene.

        The solver reads the section stiffnesses every step, so a captured
        graph uses the new values.
        """
        if not self.modulus_range[0] <= modulus <= self.modulus_range[1]:
            raise ValueError(
                f"young-modulus must be between {self.modulus_range[0]:g} and {self.modulus_range[1]:g} Pa"
            )
        self.thread_model.axial_stiffness.assign(self._unit_axial * modulus)
        self.thread_model.bending_stiffness.assign(self._unit_bending * modulus)
        self.young_modulus = modulus

    def clear_holds(self):
        self.screen_pick.reset()
        self.holds.clear()
        if self._scripted is not None:
            self.holds.release(self._scripted)
        self._scripted = None

    def simulate(self):
        """One frame of substeps on a stable pair of state buffers, for eager and captured stepping alike."""
        substeps = self.args.substeps
        for substep in range(substeps):
            current, following = (self.state_0, self.state_1) if substep % 2 == 0 else (self.state_1, self.state_0)
            current.clear_forces()
            self.holds.apply(current)
            # Predict the needle without the thread, solve the thread with the needle on its first node,
            # then repeat the needle's step with the thread's wrench.
            self.mujoco.step(current, self._predicted, None, None, self.sim_dt)
            self.attachment.attach(self._predicted, self.sim_dt)
            # Thread pins move in equal parts: each substep covers its share of what is left of the frame.
            self.holds.follow(1.0 / (substeps - substep))
            self.thread.step(None, None, None, None, self.sim_dt)
            self.attachment.apply(current)
            self.mujoco.step(current, following, None, None, self.sim_dt)
        if substeps % 2:
            self.state_0.assign(self.state_1)

    def _capture_physics(self):
        self.graph = None
        if not self.device.is_cuda:
            return
        snapshots = (self.model.state(), self.model.state())
        for target, source in zip(snapshots, (self.state_0, self.state_1), strict=True):
            target.assign(source)
        persistent = (self.thread.q, self.thread.qd, self.thread.boundary_q, self.thread.head_motion)
        saved = tuple(wp.clone(array) for array in persistent)

        def restore():
            self.state_0.assign(snapshots[0])
            self.state_1.assign(snapshots[1])
            for target, source in zip(persistent, saved, strict=True):
                wp.copy(target, source)

        try:
            # Both solvers allocate and load kernels in their first steps.
            self.simulate()
            wp.synchronize_device(self.device)
            restore()
            with wp.ScopedCapture(device=self.device) as capture:
                self.simulate()
            self.graph = capture.graph
        except (RuntimeError, ValueError) as error:
            print(f"[needle thread] CUDA capture unavailable; stepping without a graph: {error}", flush=True)
        finally:
            restore()

    def _hold_state(self):
        """Distance of the scripted hold's grabbed point from its target and from where it was grabbed [m].

        While the hold lasts this is the current state, afterwards the one kept at the release. It reads the
        needle's pose back, so only ``test_final`` calls it.
        """
        if self._scripted is None:
            body_q, local, target = self._held
        else:
            body_q = self.state_0.body_q
            local, target = self.holds.grabbed_point(self._scripted), self.holds.target(self._scripted)
        pose = body_q.numpy()[self.needle]
        grabbed = np.array(wp.transform_point(wp.transform(pose[:3], pose[3:]), wp.vec3(*local)))
        return float(np.linalg.norm(grabbed - target)), float(np.linalg.norm(grabbed - self._grab))

    def _run_script(self):
        """Test mode: grab the needle, lift it, hold it and let it go."""
        if self._frame == self._lift_frame:
            # The needle still rests where it started, so this world point is on its arc. The hold is not
            # committed: it has no marker, so it stays out of the list that the pick tool's markers index.
            self._scripted = self.holds.hold_needle(self._grab)
        if self._scripted is None:
            return
        if self._frame == self._release_frame:
            # Stepping reads nothing back: the pose is copied on the device, and test_final measures the hold.
            self._held = (
                wp.clone(self.state_0.body_q),
                self.holds.grabbed_point(self._scripted),
                self.holds.target(self._scripted),
            )
            self.holds.release(self._scripted)
            self._scripted = None
            return
        progress = min(1.0, (self._frame - self._lift_frame + 1) / self._lift_frames)
        self.holds.set_target(self._scripted, self._grab + progress * self._lift)

    def step(self):
        if self.args.test:
            self._run_script()
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self._frame += 1
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.thread_view.render(show_nodes=self.show_nodes)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Needle on a thread")
        ui.text_wrapped("A free needle in MuJoCo and a cable-element thread, coupled both ways. No tissue.")
        ScreenPickTool.gui(ui)
        ui.text_wrapped("Right-drag the thread to pin its nearest node in the same way.")
        ui.text(f"Holds: {len(self.holds.attachments) + (self._scripted is not None)}")
        if ui.button("Clear holds"):
            self.clear_holds()
        ui.same_line()
        if ui.button("Frame scene"):
            self.reset_camera()
        ui.separator()
        ui.text("Thread stiffness: Young's modulus [MPa]")
        changed, value = ui.slider_float(
            "##thread_modulus",
            1.0e-6 * self.young_modulus,
            1.0e-6 * self.modulus_range[0],
            1.0e-6 * self.modulus_range[1],
            "%.0f",
            ui.SliderFlags_.logarithmic.value,
        )
        if changed:
            self.set_young_modulus(min(self.modulus_range[1], max(self.modulus_range[0], 1.0e6 * value)))
        ui.text_wrapped("Left is softer. 2000 MPa is solid nylon.")
        ui.separator()
        ui.text("Debug view")
        _, self.show_nodes = ui.checkbox("Thread nodes and slopes", self.show_nodes)
        changed, self.show_collision = ui.checkbox("Needle collision capsules", self.show_collision)
        if changed:
            self.viewer.show_collision = self.show_collision
        ui.text_wrapped("Yellow: thread nodes, the only points where the thread touches the floor. Red: nodal slopes.")
        ui.separator()
        ui.text(f"Needle: {1.0e3 * self.geometry.mass:.1f} g, {len(self.geometry.centerline) - 1} capsules")
        ui.text(
            f"Thread: {self.args.elements} elements, {1.0e3 * self.thread_length:.0f} mm long, "
            f"{2.0e3 * self.thread_radius:.1f} mm thick"
        )
        ui.text(f"{self.args.substeps} substeps per 60 Hz frame, at most {self.args.updates} Newton updates each")
        ui.text_wrapped("The orange end is pinned. The thread does not collide with itself or with the needle.")
        ui.text("SI units: metres, kilograms, seconds")

    def test_final(self):
        pose = self.state_0.body_q.numpy()[self.needle]
        rate = self.state_0.body_qd.numpy()[self.needle]
        nodes = self.thread.q.numpy().reshape(-1, 6)
        if not (np.isfinite(pose).all() and np.isfinite(rate).all() and np.isfinite(nodes).all()):
            raise AssertionError("Needle or thread state is not finite")
        transform = wp.transform(pose[:3], pose[3:])
        tail = np.array(wp.transform_point(transform, wp.vec3(*self._tail)))
        gap = float(np.linalg.norm(nodes[0, :3] - tail))
        if gap > self.thread_radius:
            raise AssertionError(f"Thread left the needle's tail by {1.0e3 * gap:.3f} mm")
        if np.linalg.norm(nodes[-1, :3] - self._pin) > 1.0e-9:
            raise AssertionError("The thread's pinned end moved")
        chords = float(np.linalg.norm(np.diff(nodes[:, :3], axis=0), axis=1).sum())
        if chords > 1.001 * self.thread_length:
            raise AssertionError(f"Thread stretched to {1.0e3 * chords:.2f} mm")
        lowest = float(nodes[:, 2].min())
        if lowest < 0.5 * self.thread_radius:
            raise AssertionError(f"Thread sank into the floor: lowest centre {1.0e3 * lowest:.3f} mm")
        message = f"Needle and thread: tail gap {1.0e6 * gap:.1f} um, thread chords {1.0e3 * chords:.2f} mm"
        if self._scripted is not None or self._held is not None:
            error, travelled = self._hold_state()
            # The slack thread pulls on the held needle with less than the needle's weight.
            if error * self.holds.stiffness > self.geometry.mass * GRAVITY:
                raise AssertionError(f"Held needle is {1.0e3 * error:.2f} mm from its hold")
            # The hold's spring softens with the step, so the force alone passes a needle that never left the
            # floor: the needle must also be nearer to its hold than to where it was grabbed. A hold that still
            # moves lags its target, in the lift's first frames by more than it has travelled, so this is asked
            # of the hold that has rested, the one kept at the release.
            if self._held is not None and error > travelled:
                raise AssertionError(
                    f"Held needle is {1.0e3 * error:.2f} mm from its hold after {1.0e3 * travelled:.2f} mm"
                )
            message += f", hold error {1.0e3 * error:.3f} mm after a {1.0e3 * travelled:.1f} mm lift"
        if self._held is not None and self._frame >= self._rest_frame:
            # At rest the needle lies on its capsules, its axis one capsule radius above the floor, and its
            # kinetic energy could not lift it by that radius.
            mass, radius = self.geometry.mass, self.geometry.thickness
            rotation = np.array(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
            highest = float((self.geometry.centerline @ rotation.T + pose[:3])[:, 2].max())
            if highest > 2.0 * radius:
                raise AssertionError(f"Released needle is not on the floor: its axis reaches {1.0e3 * highest:.2f} mm")
            spin = rotation.T @ rate[3:]
            inertia = self.model.body_inertia.numpy()[self.needle]
            energy = 0.5 * mass * float(rate[:3] @ rate[:3]) + 0.5 * float(spin @ inertia @ spin)
            if energy > mass * GRAVITY * radius:
                raise AssertionError(f"Released needle is not at rest: kinetic energy {energy:.2e} J")
            message += f", released needle at rest with its axis below {1.0e3 * highest:.2f} mm"
        print(message)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--substeps", type=int, default=4, help="Substeps per 60 Hz frame (default: 4).")
        parser.add_argument("--elements", type=int, default=32, help="Thread elements (default: 32).")
        parser.add_argument(
            "--updates",
            type=int,
            default=2,
            help="Most Newton updates of the thread per substep (default: 2).",
        )
        parser.add_argument(
            "--young-modulus",
            type=float,
            default=2.0e6,
            help="Thread Young's modulus [Pa], 2e6 to 2e9; solid nylon is 2e9 (default: 2e6).",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
