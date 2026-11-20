"""Record one representative clip per discovered example, using its real simulation."""

import argparse
import json
import math
import os
import runpy
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from build_readme import ROOT, project

# What one camera of a recipe holds: where it stands and what it looks at [m, world frame of the
# example], the vertical field of view of the picture [degrees] and, optionally, the clip distances [m].
CAMERA_REQUIRED = ("position", "target", "fov")
CAMERA_OPTIONAL = ("near", "far")
# Newton's viewer keeps the pitch of its camera inside this bound [degrees].
STEEPEST_PITCH = 89.0


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def camera_keys(entry):
    """The timed cameras of a recipe's ``camera`` entry, checked and in frame order.

    One camera is a dictionary: ``position`` and ``target`` [m] in the world frame of the example,
    ``fov`` [degrees], the vertical field of view of the picture, and optionally ``near`` and ``far``
    [m], the clip distances (without them the example's stay). A timed camera is a list of such
    dictionaries, each with ``frame``, the index of a captured frame counted from 0, in rising order:
    the camera moves linearly from one to the next, and stands at the first before it and at the last
    after it. Newton's viewer has no roll, so there is no up vector: the up of the picture is the up
    axis of the example's model.
    """
    timed = isinstance(entry, list)
    cameras = entry if timed else [entry]
    if not cameras or not all(isinstance(camera, dict) for camera in cameras):
        raise ValueError("camera must be one camera (a dictionary) or a list of timed cameras")
    keys, previous = [], -1
    for camera in cameras:
        allowed = (*(("frame",) if timed else ()), *CAMERA_REQUIRED, *CAMERA_OPTIONAL)
        unknown = sorted(set(camera) - set(allowed))
        if unknown:
            # The one entry a camera of another program would have and this one cannot.
            no_roll = " (the viewer has no roll: the up of the picture is the up axis of the model)"
            raise ValueError(
                f"camera has unknown entries {unknown}; a camera holds {', '.join(allowed)}"
                + (no_roll if "up" in unknown else "")
            )
        missing = [name for name in allowed if name not in camera and name not in CAMERA_OPTIONAL]
        if missing:
            raise ValueError(f"camera lacks {', '.join(missing)}")
        key = {"frame": 0}
        if timed:
            frame = camera["frame"]
            if not isinstance(frame, int) or isinstance(frame, bool) or frame <= previous:
                raise ValueError("camera frames must be whole numbers from 0 in rising order")
            key["frame"] = previous = frame
        for name in ("position", "target"):
            point = camera[name]
            if not isinstance(point, (list, tuple)) or len(point) != 3 or not all(_number(value) for value in point):
                raise ValueError(f"camera {name} must be three numbers [m]")
            key[name] = tuple(float(value) for value in point)
        if key["position"] == key["target"]:
            raise ValueError("camera position and target must differ")
        for name in ("fov", *(name for name in CAMERA_OPTIONAL if name in camera)):
            if not _number(camera[name]) or camera[name] <= 0.0:
                raise ValueError(f"camera {name} must be a positive number")
            key[name] = float(camera[name])
        if key["fov"] >= 180.0:
            raise ValueError("camera fov is the vertical field of view in degrees, below 180")
        if key.get("near", 0.0) >= key.get("far", math.inf):
            raise ValueError("camera near must be below far")
        if keys and set(key) != set(keys[0]):
            raise ValueError("timed cameras must all hold the same entries, so that the move between them is defined")
        keys.append(key)
    return keys


def camera_at(keys, frame):
    """The camera of a captured frame: linear between the keys around it, the nearest key outside them."""
    if frame <= keys[0]["frame"]:
        return keys[0]
    for before, after in zip(keys, keys[1:], strict=False):
        if frame < after["frame"]:
            weight = (frame - before["frame"]) / (after["frame"] - before["frame"])
            camera = {"frame": frame}
            for name in before:
                if name in ("position", "target"):
                    camera[name] = tuple(a + weight * (b - a) for a, b in zip(before[name], after[name], strict=True))
                elif name != "frame":
                    camera[name] = before[name] + weight * (after[name] - before[name])
            return camera
    return keys[-1]


def view_angles(position, target, up_axis):
    """Pitch and yaw [degrees] with which Newton's viewer camera at ``position`` looks at ``target``.

    ``up_axis`` is the up axis of the model, 0, 1 or 2 for X, Y or Z. The pitch is the angle of the view
    direction above the horizontal plane, negative when the camera looks down. The yaw is the angle of
    its horizontal part from the first horizontal axis toward the second, which are X and Y with Z up:
    yaw 0 looks along +X and yaw 90 along +Y. The viewer keeps the pitch within 89 degrees, where the
    camera would no longer look at the target, so a steeper camera is refused here.
    """
    direction = [b - a for a, b in zip(position, target, strict=True)]
    length = math.sqrt(sum(value * value for value in direction))
    if length == 0.0:
        raise ValueError("camera position and target must differ")
    first, second = [axis for axis in range(3) if axis != up_axis]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, direction[up_axis] / length))))
    if abs(pitch) > STEEPEST_PITCH:
        raise ValueError(
            f"the camera looks {abs(pitch):.2f} degrees from the horizontal; the viewer allows {STEEPEST_PITCH:g}: "
            "move the position sideways (the horizontal part of the view direction is the up of the picture)"
        )
    return pitch, math.degrees(math.atan2(direction[second], direction[first]))


def place_camera(viewer, camera):
    """Give Newton's OpenGL viewer one camera of a recipe, with the viewer's own calls."""
    view = viewer.camera
    # Not Camera.look_at: it leaves the direction as it is when the target is nearer than 50 mm.
    pitch, yaw = view_angles(camera["position"], camera["target"], view.up_axis)
    view.fov = camera["fov"]
    for name in CAMERA_OPTIONAL:
        if name in camera:
            setattr(view, name, camera[name])
    viewer.set_camera(pos=camera["position"], pitch=pitch, yaw=yaw)
    # The point the viewer orbits about; its distance is also the depth reference of transparent surfaces.
    view.set_pivot(camera["target"])


def camera_state(viewer):
    """The camera the viewer has, for the sidecar: read back, not copied from the recipe."""
    view = viewer.camera
    return {
        "position": [float(value) for value in view.pos],
        "view_direction": [float(value) for value in view.get_front()],
        "pitch": float(view.pitch),
        "yaw": float(view.yaw),
        "fov": float(view.fov),
        "near": float(view.near),
        "far": float(view.far),
    }


def still_frames(text):
    """The frame indices of ``--stills 0,300,900``, in rising order."""
    try:
        frames = sorted({int(value) for value in text.split(",")})
    except ValueError:
        raise argparse.ArgumentTypeError("give the frames as whole numbers separated by commas") from None
    if frames[0] < 0:
        raise argparse.ArgumentTypeError("a frame is counted from 0")
    return frames


def _whole(value):
    return isinstance(value, int) and not isinstance(value, bool)


def edge_intervals(entry, frames):
    """The frame intervals of a recipe's ``edges`` entry, checked against a clip of ``frames`` frames.

    The entry is a list of ``[first_frame, last_frame]`` pairs: indices of captured frames counted from
    0, both ends included, in rising order without overlap. In the frames of an interval "Show Edges"
    of Newton's OpenGL viewer is on (``viewer.renderer.draw_edges``: the drawn meshes are drawn again
    as dark lines), in all other frames of the clip it is off.
    """
    if not isinstance(entry, list):
        raise ValueError("edges must be a list of [first_frame, last_frame] intervals")
    intervals, previous = [], -1
    for interval in entry:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2 or not all(_whole(v) for v in interval):
            raise ValueError(
                f"edges interval {json.dumps(interval)} must be [first_frame, last_frame], two whole numbers"
            )
        first, last = interval
        if first < 0 or last < first:
            raise ValueError(f"edges interval [{first}, {last}] must hold 0 <= first_frame <= last_frame")
        if last >= frames:
            raise ValueError(
                f"edges interval [{first}, {last}] is outside the clip of {frames} frames (0 to {frames - 1})"
            )
        if first <= previous:
            raise ValueError(
                f"edges interval [{first}, {last}] begins at or before frame {previous}, the end of the interval "
                "before it; the intervals must be in rising order without overlap"
            )
        intervals.append((first, last))
        previous = last
    return intervals


def edges_at(intervals, frame):
    """Whether the edges are drawn in a captured frame: inside an interval, its two ends included."""
    return any(first <= frame <= last for first, last in intervals)


def show_edges(viewer, shown):
    """Set "Show Edges" of Newton's OpenGL viewer: the switch that the checkbox of its panel sets."""
    renderer = getattr(viewer, "renderer", None)
    if not hasattr(renderer, "draw_edges"):
        raise ValueError(
            "the recipe holds edges, but the viewer has no renderer.draw_edges "
            '("Show Edges" of Newton\'s OpenGL viewer); nothing else draws the edges'
        )
    renderer.draw_edges = bool(shown)


def crop_box(entry, width, height):
    """The rectangle of a recipe's ``crop`` entry, checked against a captured frame of ``width`` x ``height``.

    The entry is ``[x, y, width, height]`` in pixels of the captured frame, with the origin at its
    top-left corner: ``x`` to the right and ``y`` down to the top-left corner of the part that is kept.
    The part lies inside the captured frame and its width and height are even, as the MP4 needs.
    """
    if not isinstance(entry, (list, tuple)) or len(entry) != 4 or not all(_whole(value) for value in entry):
        raise ValueError("crop must be [x, y, width, height], four whole numbers of pixels")
    x, y, crop_width, crop_height = entry
    if x < 0 or y < 0 or crop_width < 1 or crop_height < 1:
        raise ValueError("crop must hold x >= 0 and y >= 0 from the top-left corner and a positive width and height")
    if crop_width % 2 or crop_height % 2:
        raise ValueError(f"crop width and height must be even for the MP4; they are {crop_width} and {crop_height}")
    if x + crop_width > width or y + crop_height > height:
        raise ValueError(
            f"crop [{x}, {y}, {crop_width}, {crop_height}] reaches to pixel ({x + crop_width}, {y + crop_height}), "
            f"outside the captured frame of {width}x{height} (--width, --height)"
        )
    return x, y, crop_width, crop_height


def crop_frame(pixels, box):
    """The part of a frame (an array of rows, columns and colours) inside a checked ``crop_box``."""
    x, y, width, height = box
    if y + height > pixels.shape[0] or x + width > pixels.shape[1]:
        raise ValueError(
            f"crop [{x}, {y}, {width}, {height}] reaches outside the frame of {pixels.shape[1]}x{pixels.shape[0]}"
        )
    return pixels[y : y + height, x : x + width]


def recipe(name, options):
    """The recipe of an example and what a recording takes from it: recipe, frame count, timed cameras or None.

    Without ``--profiles`` the recipes are those of ``docs/videos.json``, where an example without an
    entry is recorded with the defaults. A file named with ``--profiles`` must hold the example. The
    entries ``edges`` and ``crop`` are checked here, against the frame count and the picture size of
    this recording; ``record`` reads them with ``edge_intervals`` and ``crop_box``.
    """
    if options.profiles is None:
        profiles = ROOT / "docs" / "videos.json"
        profile = json.loads(profiles.read_text()).get(name, {}) if profiles.exists() else {}
    else:
        profiles = json.loads(options.profiles.read_text())
        if name not in profiles:
            raise ValueError(f"{options.profiles} has no recipe for {name}")
        profile = profiles[name]
    frames = options.frames or profile.get("frames", 240)
    try:
        camera = camera_keys(profile["camera"]) if "camera" in profile else None
        if "edges" in profile:
            edge_intervals(profile["edges"], frames)
        if "crop" in profile:
            crop_box(profile["crop"], options.width, options.height)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from None
    if options.stills and options.stills[-1] >= frames:
        raise ValueError(
            f"{name}: still {options.stills[-1]} is outside the clip of {frames} frames (0 to {frames - 1})"
        )
    return profile, frames, camera


class Clip:
    def __init__(self, path, fps):
        self.path, self.fps = path, fps
        self.process = None
        self.count = 0
        self.last_image = None
        self.shape = None

    def write(self, pixels):
        import numpy as np

        pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("Expected an RGB frame")
        if self.shape is not None and pixels.shape != self.shape:
            raise ValueError(f"Capture dimensions changed from {self.shape} to {pixels.shape}")
        self.shape = pixels.shape
        if self.process is None:
            height, width, _ = pixels.shape
            self.process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "2",
                    "-crf",
                    "22",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(self.path),
                ],
                stdin=subprocess.PIPE,
            )
        self.process.stdin.write(pixels.tobytes())
        self.count += 1
        self.last_image = pixels

    def close(self):
        if self.process is None:
            raise RuntimeError("The example produced no frames")
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("ffmpeg could not encode the clip")


def record(name, options):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    _, package, examples = project()
    description, headless, relative_module = examples[name]
    profile, frames, camera = recipe(name, options)
    # The frames in which the viewer draws its edges, and the part of the picture that is kept: None without the entry.
    edges = edge_intervals(profile["edges"], frames) if "edges" in profile else None
    crop = crop_box(profile["crop"], options.width, options.height) if "crop" in profile else None
    fps = options.fps or profile.get("fps", 60)
    output = options.output_dir
    output.mkdir(parents=True, exist_ok=True)
    final = output / f"{name}.mp4"
    temporary = output / f".{name}.tmp.mp4"
    hooks_path = ROOT / "tools" / "video_scenarios.py"
    hooks = runpy.run_path(str(hooks_path)) if hooks_path.exists() else {}
    arguments = profile.get("args", []) + options.arguments
    caption = profile.get("caption", description)
    writer = Clip(temporary, fps)
    start = time.perf_counter()
    simulation_times = []
    scenario_metadata = None
    # The recipe's camera of the frame that is being captured, and what the viewer had when it was drawn.
    camera_now, camera_views = None, []
    # Whether the viewer draws its edges in the frame that is being captured; None leaves its switch alone.
    edges_now = None

    def write_frame(pixels, label=""):
        picture = Image.fromarray(pixels)
        draw = ImageDraw.Draw(picture)
        draw.rectangle((0, 0, picture.width, 29), fill=(25, 28, 34))
        draw.text(
            (12, 8),
            name.replace("_", " ") + ("  |  " + label if label else ""),
            fill="white",
            font=ImageFont.load_default(size=16),
        )
        if crop is not None:
            # The crop counts pixels of the captured frame, title bar included: cut after the bar is
            # drawn and before anything is written, so the clip, the poster, the stills and an
            # animated preview all hold the cropped picture.
            if picture.size != (options.width, options.height):
                raise ValueError(
                    f"The captured frame is {picture.width}x{picture.height}, not the "
                    f"{options.width}x{options.height} that the crop was checked against"
                )
            picture = Image.fromarray(crop_frame(np.asarray(picture), crop))
        if writer.count in options.stills:
            # The frame as it goes to the encoder, complete on disk while the recording goes on.
            still = output / f"{name}.still-{writer.count:05d}.jpg"
            partial = output / f".{name}.still.tmp.jpg"
            picture.save(partial, quality=92)
            partial.replace(still)
            print(f"Still {still}", flush=True)
        writer.write(np.asarray(picture))

    if headless:
        callback = hooks.get("record_terminal")
        if callback is None:
            raise ValueError(f"{name} is terminal-only; provide record_terminal in tools/video_scenarios.py")
        if camera:
            raise ValueError(f"{name} is terminal-only; its recipe cannot hold a camera")
        if edges is not None:
            raise ValueError(f"{name} is terminal-only; its recipe cannot hold edges")
        for pixels in callback(name, options.device, frames, options.width, options.height):
            write_frame(pixels, "optimization progress")
    else:
        import newton.examples
        import newton.viewer

        original_init, original_run = newton.examples.init, newton.examples.run
        original_gl = newton.viewer.ViewerGL

        class CaptureViewer(original_gl):
            def __init__(self, **kwargs):
                kwargs.update(width=options.width, height=options.height, vsync=False)
                # Newton initializes ImGui only with a window. Interactive
                # demonstrations need the same controls and grab markers as
                # an ordinary viewer run; other captures remain headless.
                if profile.get("show_ui", False):
                    kwargs["headless"] = False
                super().__init__(**kwargs)
                if profile.get("show_ui", False):
                    # Initialize the real UI, then keep capture independent of
                    # desktop tiling and interactive window resizing.
                    window = self.renderer.window
                    window.set_visible(False)
                    window.dispatch_events()
                    window.set_size(options.width, options.height)
                    window.dispatch_events()
                # The clip is read back with a Warp kernel of Newton's viewer. Launch it once
                # here, before the example is built: an example's self-test counts a kernel
                # module that loads after its start-up as a failure, and this one is the
                # recorder's, not the example's.
                self.get_frame(render_ui=profile.get("show_ui", False))

            def end_frame(self):
                # An example sets its own camera where it frames its scene: at a reset, in its
                # framing helpers, in its scripted self-test. A frame is drawn here, so the
                # recipe's camera is given last. Without one nothing is touched.
                if camera_now is not None:
                    place_camera(self, camera_now)
                # The same for "Show Edges": the switch of the frame is given where the frame is
                # drawn, after anything the example did with it. Without the entry it is not touched.
                if edges_now is not None:
                    show_edges(self, edges_now)
                super().end_frame()

        def capture_run(example, args):
            nonlocal scenario_metadata, camera_now, edges_now
            configure = hooks.get("configure")
            if configure:
                configure(name, example)
            before_frame = hooks.get("before_frame")
            label = profile.get("label", "simulation playback")
            viewer = example.viewer
            if hasattr(viewer, "hide_loading_splash"):
                viewer.hide_loading_splash()
            if hasattr(example, "gui") and hasattr(viewer, "register_ui_callback"):
                viewer.register_ui_callback(lambda ui: example.gui(ui), position="side")
            try:
                if camera:
                    # Every camera of the clip is one the viewer can take: known before the first frame.
                    for frame in range(frames):
                        view = camera_at(camera, frame)
                        view_angles(view["position"], view["target"], viewer.camera.up_axis)
                if edges is not None:
                    # A viewer without the switch is refused here, before the first frame.
                    show_edges(viewer, edges_at(edges, 0))
                for frame in range(frames):
                    if edges is not None:
                        # Before the frame is drawn: the first frame of an interval shows the edges
                        # and the frame after its last does not.
                        edges_now = edges_at(edges, frame)
                        show_edges(viewer, edges_now)
                    if camera:
                        # Before the scripted action and the step of this frame, which may read the
                        # camera (a pick through a pixel), and again where the frame is drawn.
                        camera_now = camera_at(camera, frame)
                        place_camera(viewer, camera_now)
                    if before_frame:
                        before_frame(name, example, frame)
                    example.step()
                    example.render()
                    if camera and frame in (0, frames - 1):
                        camera_views.append({"frame": frame, **camera_state(viewer)})
                        print(f"{name}: camera of frame {frame}: {json.dumps(camera_views[-1])}", flush=True)
                    frame_label = hooks.get("frame_label")
                    write_frame(
                        viewer.get_frame(render_ui=profile.get("show_ui", False)).numpy(),
                        frame_label(name, example) if frame_label else label,
                    )
                    if frame % max(1, frames // 8) == 0:
                        print(f"{name}: frame {frame + 1}/{frames}", flush=True)
                if args.test and hasattr(example, "test_final"):
                    example.test_final()
                metadata_hook = hooks.get("recording_metadata")
                if metadata_hook:
                    scenario_metadata = metadata_hook(name, example)
                for attribute in ("sim_time", "time", "t"):
                    value = getattr(example, attribute, None)
                    if isinstance(value, (int, float)):
                        simulation_times.append({"attribute": attribute, "end": value})
                        break
            finally:
                viewer.close()

        newton.viewer.ViewerGL = CaptureViewer
        newton.examples.run = capture_run
        module = f"{package}.examples.{relative_module}"
        sys.argv = [
            module,
            *arguments,
            "--device",
            options.device,
            "--viewer",
            "gl",
            "--headless",
            "--num-frames",
            str(frames),
        ]
        try:
            runpy.run_module(module, run_name="__main__")
        finally:
            newton.examples.init, newton.examples.run = original_init, original_run
            newton.viewer.ViewerGL = original_gl
    writer.close()
    temporary.replace(final)
    poster = output / f"{name}.jpg"
    if "poster_frame" in profile:
        # A released object can hide the interaction in the final frame.
        # Extract the configured representative frame from the actual clip.
        frame = min(max(0, int(profile["poster_frame"])), writer.count - 1)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                str(frame / fps),
                "-i",
                str(final),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(poster),
            ],
            check=True,
        )
    else:
        Image.fromarray(writer.last_image).save(poster, quality=88)
    if options.gif:
        # An animated preview, only when asked for: the README shows the poster and links the MP4.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-threads",
                "2",
                "-filter_complex_threads",
                "1",
                "-i",
                str(final),
                "-filter_complex",
                "fps=12,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
                str(output / f"{name}.gif"),
            ],
            check=True,
        )
    metadata = {
        "example": name,
        "module": f"{package}.examples.{relative_module}",
        "caption": caption,
        "arguments": arguments,
        "device": options.device,
        "frames": writer.count,
        "fps": fps,
        # The size of the written clip: that of the captured frame, or of the crop.
        "width": crop[2] if crop is not None else options.width,
        "height": crop[3] if crop is not None else options.height,
        "show_ui": profile.get("show_ui", False),
        "recording_wall_seconds": time.perf_counter() - start,
        "playback_is_performance_benchmark": False,
        "simulation_time": simulation_times,
        "dependencies": {package: version(package), "newton": version("newton"), "warp-lang": version("warp-lang")},
    }
    if camera:
        # The recipe's entry as it was written, and the camera the viewer had in the first and the last frame.
        metadata["camera"] = profile["camera"]
        metadata["camera_view"] = camera_views
    if edges is not None:
        # The recipe's entry as it was written: the intervals of frames, both ends included, with the edges drawn.
        metadata["edges"] = profile["edges"]
    if crop is not None:
        # The recipe's entry as it was written, [x, y, width, height] from the top-left corner of the
        # captured frame, and the size of that frame.
        metadata["crop"] = profile["crop"]
        metadata["captured_width"], metadata["captured_height"] = options.width, options.height
    if scenario_metadata is not None:
        metadata["scenario"] = scenario_metadata
    (output / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Recorded {final}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", nargs="?")
    parser.add_argument("--all", action="store_true", help="Capture every automatically discovered example.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "media")
    parser.add_argument(
        "--profiles",
        type=Path,
        help="Read the recipes from this file instead of docs/videos.json; it must hold every example that is recorded.",
    )
    parser.add_argument(
        "--stills",
        type=still_frames,
        default=[],
        metavar="FRAME,FRAME,...",
        help="Also write these frames of the clip, counted from 0, as <example>.still-<frame>.jpg beside it.",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Also write <example>.gif, an animated preview 480 pixels wide at 12 frames per second. "
        "Without it a recording writes the MP4, the JPG poster and the JSON sidecar; no document shows a GIF.",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    options, arguments = parser.parse_known_args()
    options.arguments = arguments[1:] if arguments[:1] == ["--"] else arguments
    _, _, examples = project()
    if options.list:
        print("\n".join(examples))
        return
    if (not options.all and options.example not in examples) or (options.all and options.example):
        parser.error("Choose --all or one discovered example; use --list to see names")
    if any(value is not None and value < 1 for value in (options.frames, options.fps, options.width, options.height)):
        parser.error("Capture dimensions, frame count and fps must be positive")
    if options.width % 2 or options.height % 2:
        parser.error("MP4 dimensions must be even")
    if not shutil.which("ffmpeg"):
        parser.error("Install ffmpeg to encode example videos")
    if options.child:
        record(options.example, options)
        return
    # A recording runs in this repository's folder: the two paths are those of the caller's folder.
    options.output_dir = options.output_dir.resolve()
    if options.profiles is not None:
        options.profiles = options.profiles.resolve()
    selected = list(examples) if options.all else [options.example]
    try:
        # A recipe that cannot be recorded is refused here, before a simulation is started for it.
        for name in selected:
            recipe(name, options)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    failures = []
    for name in selected:
        if options.skip_existing and (options.output_dir / f"{name}.mp4").is_file():
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            name,
            "--child",
            "--device",
            options.device,
            "--width",
            str(options.width),
            "--height",
            str(options.height),
            "--output-dir",
            str(options.output_dir),
        ]
        for argument, value in (
            ("--frames", options.frames),
            ("--fps", options.fps),
            ("--profiles", options.profiles),
            ("--stills", ",".join(str(frame) for frame in options.stills) or None),
        ):
            if value is not None:
                command.extend([argument, str(value)])
        command += ["--gif"] if options.gif else []
        command += ["--", *options.arguments] if options.arguments else []
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(command, cwd=ROOT, env=environment)
        if result.returncode:
            failures.append(name)
    if failures:
        parser.exit(1, f"Failed recordings: {', '.join(failures)}\n")
    if options.output_dir == (ROOT / "docs" / "media").resolve():
        # The README shows the clips of docs/media; a recording to another folder does not change it.
        subprocess.run([sys.executable, str(ROOT / "tools" / "build_readme.py")], check=True)


if __name__ == "__main__":
    main()
