# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared viewer for prepared USD scenes, pose arrays and camera image sequences."""

import argparse
import json

import newton
import newton.examples
import numpy as np
import warp as wp

from newton_solver_suturebot.examples._startup import Script, Startup, first_draw
from newton_solver_suturebot.examples.recording._suturebot_recording import Recording

# Dataset labels are 9x9-pixel clicks in the 1091.9 px-focal left camera about
# 67 mm from the pad: 0.30-0.32 mm on the surface. A sphere centred on the
# surface point leaves that footprint visible.
POINT_LABEL_RADIUS = 3.1e-4


def add_point_label_spheres(builder, recording):
    """Add the dataset's insertion/exit labels as visual-only spheres; return shape indices by label."""
    labels = recording.manifest.get("point_labels")
    if labels is None:
        return {}
    report = json.loads(recording.asset(labels["report"]).read_text())
    color = tuple(channel / 255.0 for channel in labels["display_color_rgb"])
    visual = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False)
    return {
        point["label"]: builder.add_shape_sphere(
            -1,
            xform=wp.transform(wp.vec3(*point["position_world_m"]), wp.quat_identity()),
            radius=POINT_LABEL_RADIUS,
            cfg=visual,
            color=color,
            label=f"/PointLabels/{point['label']}",
        )
        for point in report["points"]
    }


class RecordingExample:
    """Replay a prepared scene; subclasses supply its catalog, default and labels."""

    def __init__(self, viewer, args=None):
        self.args = newton.examples.default_args(self.create_parser()) if args is None else args
        self.viewer, self.device = viewer, wp.get_device()
        if not np.isfinite(self.args.playback_speed) or self.args.playback_speed <= 0:
            raise ValueError("Playback speed must be positive and finite")
        # Everything from here to the end of the constructor is start-up (examples/_startup.py).
        self.startup = Startup(self.device)
        self.startup.begin()
        self.samples = self.recording_catalog(include_aliases=False)
        self.pending_sample = None
        self.pending_frame = None
        self.recording = None
        self.overlay = None
        self.show_overlays = not self.args.no_camera_overlays
        # Whether this run can show the camera overlays at all: it starts with them, or its viewer has the
        # panel with the box that shows them later. Then every recording is loaded with its overlay.
        self.overlays_possible = self.show_overlays or hasattr(self.viewer, "register_ui_callback")
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.load_recording(self.args.sample or self.default_recording)
        self.reset_camera()
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.camera_gui, position="free")
        self.script = None
        if getattr(self.args, "test", False):
            self.script = self._test_script()
            self.script.reserve(self.viewer)
        self._finish_startup(frame=True)

    def _finish_startup(self, *, frame):
        """End a start-up: the scene drawn once, then the mark.

        The constructor draws a whole frame where the viewer has no frame limit; a recording
        switch issues the log calls alone.
        """
        first_draw(self.viewer, lambda: self.viewer.log_state(self.state_0), self.sim_time, frame=frame)
        self.startup.finish()

    def load_recording(self, path):
        self.pending_frame = None
        recording = Recording(path, catalog=self.recording_catalog())
        # A recording switch rebuilds the scene: it starts up again. The constructor has its
        # start-up open already and ends it itself.
        rebuild = self.recording is not None
        if rebuild:
            self.startup.begin()
        self.recording = recording
        self._build_scene()
        self.elapsed = 0.0
        self.playing = True
        self.seek(0)
        self._prepare_overlay()
        if rebuild:
            self._finish_startup(frame=False)

    def _build_overlay(self):
        from newton_solver_suturebot.examples.recording._suturebot_cameras import CameraOverlay

        return CameraOverlay(
            self.model, self.recording.cameras, self.recording.root, camera_order=self.recording.camera_order
        )

    def _render_overlay(self):
        poses, valid = self.recording.camera_frame(self.frame_index)
        self.overlay.render(self.state_0, self.frame_index, poses, valid)

    def _prepare_overlay(self):
        """Build the camera overlay of the loaded recording and render it once, while the recording loads.

        The first render of an overlay loads the ray-tracing kernel of Newton's camera sensor.
        Here that falls into the start-up, also while the overlays are hidden: ticking the box
        later then builds and loads nothing. A run that was started with --no-camera-overlays
        on a viewer without the panel has no box to tick and builds no overlay.
        """
        if self.overlays_possible:
            self.overlay = self._build_overlay()
            self._render_overlay()

    def _build_scene(self):
        """Import the prepared geometry when switching recordings."""
        if self.overlay is not None:
            self.overlay.close()
            self.overlay = None
        builder = newton.ModelBuilder()
        builder.add_usd(str(self.recording.scene_path), collapse_fixed_joints=False, hide_collision_shapes=True)
        add_point_label_spheres(builder, self.recording)
        self.model = builder.finalize(device=self.device)
        self.model.set_gravity((0, 0, 0))
        self.state_0 = self.model.state()
        labels = {label: index for index, label in enumerate(self.model.body_label)}
        missing = set(self.recording.body_paths) - set(labels)
        if missing:
            raise ValueError(f"Prepared motion references bodies absent from the scene USD: {sorted(missing)}")
        self.body_indices = np.asarray([labels[path] for path in self.recording.body_paths])
        self.body_poses = self.state_0.body_q.numpy().copy()
        self.viewer.set_model(self.model)
        self.frame_index = -1

    def seek(self, frame):
        frame = int(np.clip(frame, self.recording.first_frame, self.recording.last_frame))
        self.elapsed = float(self.recording.time_s[frame])
        self._show_frame(frame)

    def _show_frame(self, frame):
        if frame == self.frame_index:
            return
        self.frame_index = frame
        self.body_poses[self.body_indices] = self.recording.body_poses[frame]
        self.state_0.body_q.assign(self.body_poses)
        self.state_0.body_qd.zero_()

    def step(self):
        if self.script is not None:
            self.script.play()
        if self.pending_sample is not None:
            self.load_recording(self.pending_sample)
            self.pending_sample = None
        if self.pending_frame is not None:
            self.seek(self.pending_frame)
            self.pending_frame = None
        if self.playing:
            self.elapsed += self.frame_dt * self.args.playback_speed
            self._show_frame(self.recording.frame_at(self.elapsed, loop=self.args.loop))
            if not self.args.loop and self.elapsed >= self.recording.end_time:
                self.playing = False
        self.sim_time += self.frame_dt

    def reset_camera(self):
        self.viewer.camera_speed = 0.12
        view = self.recording.manifest.get("viewer", {}).get("scene", {})
        self.viewer.set_camera(
            pos=view.get("position", (-0.568, 0.551, 0.426)),
            pitch=view.get("pitch", -24.0),
            yaw=view.get("yaw", -55.0),
        )
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "set_pivot"):
            camera.set_pivot(view.get("pivot", (0.0, 0.0, 0.08)))

    def frame_phantom(self):
        view = self.recording.manifest.get("viewer", {}).get("pad", {})
        self.viewer.set_camera(
            pos=view.get("position", (-0.22, 0.21, 0.20)),
            pitch=view.get("pitch", -32.0),
            yaw=view.get("yaw", -44.0),
        )
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "set_pivot"):
            camera.set_pivot(view.get("pivot", (0.0, 0.0, 0.02)))

    def render(self):
        if self.show_overlays:
            if self.overlay is None:
                # Only where the overlays were shown although the run cannot show them (_prepare_overlay).
                self.overlay = self._build_overlay()
            self._render_overlay()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text_wrapped(self.title)
        ui.text_wrapped(self.recording.manifest["sample_id"])
        names = list(self.samples)
        if len(names) > 1:
            current = next(
                (index for index, name in enumerate(names) if self.samples[name].resolve() == self.recording.path), -1
            )
            changed, selected = ui.combo("Recording", current, names)
            if changed:
                self.pending_sample = self.samples[names[selected]]
        region = self.recording.manifest.get("region_label", f"Region {self.recording.manifest.get('region_id', '—')}")
        ui.text(f"{region} | {self.recording.duration:.2f} s")
        if "playback" in self.recording.manifest:
            ui.text_wrapped(self.recording.manifest["playback"].get("description", "Selected source phase"))
        changed, self.playing = ui.checkbox("Play recording", self.playing)
        changed, self.args.loop = ui.checkbox("Loop", self.args.loop)
        changed, frame = ui.slider_int("Frame", self.frame_index, self.recording.first_frame, self.recording.last_frame)
        if changed:
            self.playing = False
            self.pending_frame = frame
        time_label = self.recording.manifest.get("time_label", "Source time")
        ui.text(f"{time_label}: {self.recording.time_s[self.frame_index]:.3f} s")
        if ui.button("Restart recording"):
            self.pending_frame = self.recording.first_frame
            self.playing = True
        changed, self.show_overlays = ui.checkbox(f"{self.recording.camera_count} camera overlays", self.show_overlays)
        if ui.button("Frame scene"):
            self.reset_camera()
        if ui.button("Frame phantom"):
            self.frame_phantom()
        ui.text_wrapped(self.motion_note)
        alignment = self.recording.manifest.get("alignment", {})
        if alignment.get("failure_reason"):
            ui.text_wrapped(f"Alignment check failed: {alignment['failure_reason']}")
        elif not alignment.get("accepted", False):
            ui.text_wrapped("Full tool/pad geometry alignment is not validated.")
        if "point_labels" in self.recording.manifest:
            if self.recording.manifest["point_labels"].get("binding_valid") is False:
                ui.text_wrapped("Green marker bindings are superseded and need refitting.")
            else:
                ui.text("Dataset insertion/exit labels: spheres on the phantom")

    def camera_gui(self, ui):
        if self.show_overlays and self.overlay is not None:
            self.overlay.draw(ui)

    def _show_overlays(self, shown):
        """As the box of the camera overlays."""
        self.show_overlays = shown

    def _seek_slider(self, frame):
        """As the frame slider."""
        self.playing = False
        self.pending_frame = frame

    def _restart(self):
        """As the Restart recording button."""
        self.pending_frame = self.recording.first_frame
        self.playing = True

    def _next_recording(self):
        """As the Recording box: the recording after the current one in the list."""
        paths = [path.resolve() for path in self.samples.values()]
        current = paths.index(self.recording.path) if self.recording.path in paths else -1
        self.pending_sample = paths[(current + 1) % len(paths)]

    def _test_script(self):
        """The scripted user of ``--test``, in 100 frames.

        The replay, the frame slider, Restart recording, the camera overlays off and on, every
        bundled recording in turn (one of them loaded while the overlays are hidden), Frame
        phantom, Frame scene and the slider at the last frame. The overlays are switched only
        in a run that can show them; the recordings only where several are bundled.
        """
        script = Script()
        script.at(10, "Frame slider", lambda: self._seek_slider(self.recording.first_frame + 10))
        script.at(16, "Restart recording", self._restart)
        if self.overlays_possible:
            script.at(24, "camera overlays off", lambda: self._show_overlays(False))
            script.at(30, "camera overlays on", lambda: self._show_overlays(True))
        switches = len(self.samples) if len(self.samples) > 1 else 0
        hidden = self.overlays_possible and switches > 1
        frame = 36
        for index in range(switches):
            if hidden and index == 1:
                script.at(frame - 4, "camera overlays off", lambda: self._show_overlays(False))
                script.at(frame + 4, "camera overlays on", lambda: self._show_overlays(True))
            script.at(frame, "Recording", self._next_recording)
            frame += 10
        script.at(frame, "Frame phantom", self.frame_phantom)
        script.at(frame + 4, "Frame scene", self.reset_camera)
        script.at(frame + 8, "Frame slider", lambda: self._seek_slider(self.recording.last_frame))
        script.rest(max(0, 100 - script.frames))
        return script

    def test_final(self):
        actual = self.state_0.body_q.numpy()[self.body_indices]
        np.testing.assert_allclose(actual, self.recording.body_poses[self.frame_index], atol=2.0e-7, rtol=0)
        if not np.isfinite(self.state_0.body_q.numpy()).all():
            raise AssertionError("Recording replay produced nonfinite body poses")
        if self.script is not None:
            # The self-test: nothing was captured or loaded after a start-up (the construction and each
            # recording switch).
            print(json.dumps({**self.script.summary(), **self.startup.check()}))

    def close(self):
        if self.overlay is not None:
            self.overlay.close()

    @classmethod
    def create_parser(cls):
        parser = newton.examples.create_parser()
        parser.description = cls.__doc__
        parser.add_argument(
            "--sample", help="Bundled sample name or recording.json path; omit for a terminal recording menu"
        )
        parser.add_argument("--list-samples", action="store_true", help="List bundled recordings and exit")
        parser.add_argument(
            "--choose-sample", action="store_true", help="Choose a recording before starting the viewer"
        )
        parser.add_argument("--playback-speed", type=float, default=1.0)
        parser.add_argument("--loop", action="store_true")
        parser.add_argument("--no-camera-overlays", action="store_true")
        parser.add_argument("--demo", action="store_true", help=argparse.SUPPRESS)
        return parser
