# Development and recording

Work from this repository's root with Python 3.12. All solver implementation,
example code and prepared assets are local. Install the development/viewer
environment with `uv sync --locked --extra examples`; use the public package
APIs and preserve third-party source headers. See [repository layout](layout.md)
and [method](method.md) before changing mechanics or numerical contracts.

## Generate the README

```bash
python3 tools/build_readme.py
python3 tools/build_readme.py --check --require-videos
```

The standard-library generator reads project metadata and discovers runnable
`example_*.py` modules without importing their simulation code. Descriptions
come from the first line of each module's docstring. Edit detailed explanations
in the corresponding guides; the README links to them instead of copying them.
`--check` detects stale generated output. `--require-videos` additionally
requires an MP4 and its preview for each discovered example. The preview is
the JPG poster that a recording writes; a GIF is taken only where there is no
JPG.

## Record demonstrations

```bash
uv run --locked --extra examples python tools/record_examples.py fem_needle_grasp
uv run --locked --extra examples python tools/record_examples.py fem_surgical_incision --width 1280 --height 720
python3 tools/build_readme.py --require-videos
```

Recording needs the `examples` extra, `ffmpeg` and OpenGL, and performs real
simulation or recording replay. `docs/videos.json` stores capture arguments;
`tools/video_scenarios.py` scripts interactions. Neither defines the example
list. Each clip's JSON sidecar records arguments, frame count and capture
conditions. Keep historical sidecars unchanged; new captures receive new
metadata. Playback rate is not live simulation performance. See
[provenance](provenance.md) for imported media and asset restrictions.

A recording writes three files: the clip `<example>.mp4`, its poster
`<example>.jpg` (the recipe's `poster_frame` of the finished clip, or its last
frame) and the sidecar `<example>.json`. The README shows the poster and links
the clip. With `--gif` the recorder also writes `<example>.gif`, an animated
preview 480 pixels wide at 12 frames per second. No document shows a GIF and
the clips in `docs/media` have none: for four of the close-up clips it would
be 3.5 to 6.6 MB. The sidecar names no file.

The picture size is an option of the recorder (`--width`, `--height`, 960×540
without them), not an entry of a recipe, and `--all` records every example at
the one size it is given. The clips with the example's panel were recorded
larger, because the panel and the overlay windows have fixed pixel sizes and
the cameras of their recipes are set for that picture:
[provenance](provenance.md#current-clips-2026-11-20) lists the size of each
clip. A recipe with a `crop` is refused at a picture size that does not hold
its part: `fem_surgical_incision` is recorded with `--width 1280 --height
720`. At the default size the recorder stops with a message before any
simulation starts, also with `--all`.

A recording with the panel (`show_ui`) reads and writes the viewer's window
layout, `imgui.ini` in the repository's root (git-ignored). The clips with the
panel show the layout of a first use at their picture size: the panel at the
upper left, 300 pixels wide and 20 pixels less high than the picture. A layout
saved by an interactive run or by a recording of another size moves or
shortens the panel, so move a present `imgui.ini` aside before recording such
a clip. The `crop` of `fem_surgical_incision` is set for that layout: it
keeps the columns between the panel and the counters.

### Camera of a clip

A recipe in `docs/videos.json` may hold a `camera`. Without one the clip shows
the example's own view. With one, the recorder gives that camera to Newton's
OpenGL viewer before the scripted action of every frame and again where the
frame is drawn, so the example's own framing (in its constructor, at a reset,
in its framing helpers, in its `--test` script, in a view change of
`tools/video_scenarios.py`) does not show.

```json
"camera": {"position": [0.11, -0.057, 0.06], "target": [0.004, -0.022, 0.02], "fov": 30.0}
```

| Entry | Meaning |
| --- | --- |
| `position` | Where the camera stands [m], in the world frame of the example's model: right-handed, with Z up in all examples. |
| `target` | The point at the middle of the picture [m]. |
| `fov` | Vertical field of view [degrees]. At a distance `d` the picture is `2 d tan(fov / 2)` high and `width / height` times as wide. |
| `near`, `far` | Optional clip distances [m]. Without them the example's stay; the viewer starts with 0.01 and 1000, so a camera nearer than 10 mm to a surface needs `near`. |

Newton's viewer camera is a position, a pitch and a yaw; it has no roll, so
there is no up vector: the up of the picture is the up axis of the model. The
recorder computes the two angles from `position` and `target` and calls
`viewer.set_camera(pos, pitch, yaw)`, sets `viewer.camera.fov` (and `near`,
`far`) and makes the target the camera's pivot. With Z up the view direction
is `(cos yaw cos pitch, sin yaw cos pitch, sin pitch)`: yaw 0 looks along +X
and yaw 90 along +Y, a negative pitch looks down, and the right of the picture
is the view direction crossed with +Z (from +X toward the origin, +Y is at the
right). The viewer keeps the pitch within 89°, so a camera within 1° of
straight down or up is refused; the horizontal part of the view direction is
the up of the picture, which decides how a view from nearly above is turned.
The recorder's title bar covers the upper 30 pixels of the picture, and with
`show_ui` the example's panel covers part of its left side.

A timed camera is a list of such cameras, each with `frame`, the index of a
captured frame counted from 0, in rising order. The camera moves linearly from
one to the next (position, target, field of view and clip distances), stands
at the first before it and at the last after it; two entries with the same
camera hold a view between them.

```json
"camera": [
  {"frame": 0, "position": [0.075, -0.03, 0.035], "target": [0.005, 0.0015, 0.0003], "fov": 15.0},
  {"frame": 20, "position": [0.075, -0.03, 0.035], "target": [0.005, 0.0015, 0.0003], "fov": 15.0},
  {"frame": 50, "position": [0.11, -0.057, 0.06], "target": [0.004, -0.022, 0.02], "fov": 30.0}
]
```

The sidecar records the entry as `camera` and, as `camera_view`, the camera
the viewer had in the first and the last frame. `nsys/profile_examples.py`
refuses a clip whose sidecar camera differs from its recipe's.

### Mesh edges and crop of a clip

```json
"edges": [[300, 419]],
"crop": [322, 50, 800, 660]
```

A recipe may hold `edges`, a list of `[first_frame, last_frame]` intervals of
captured frames (counted from 0, both ends included, in rising order without
overlap): in the frames of an interval the recorder switches on "Show Edges"
of Newton's OpenGL viewer (`viewer.renderer.draw_edges`), which draws the
meshes again as dark lines over the shaded picture, in every other frame it
switches it off, and without the entry it leaves the switch alone. A recipe
may hold `crop`, `[x, y, width, height]` in pixels of the captured frame
counted from its top-left corner, inside the `--width`×`--height` picture and
with an even width and height: every frame is cut to that part after the
title bar is drawn and before it is written, so the clip, the poster, the
stills and a GIF have the size of the part, and a part with `y` of 30 or more
has no title bar. The sidecar records both entries as they are written, gives
the size of the written clip as `width` and `height` and that of the captured
frame as `captured_width` and `captured_height`, and
`nsys/profile_examples.py` refuses a clip whose sidecar differs from its
recipe in either entry.

Three clips in `docs/media` hold an `edges` interval: `fem_needle_insertion`,
`fem_needle_suture` and `fem_surgical_incision`, the last also the `crop` of
the example above ([provenance](provenance.md#current-clips-2026-11-20)). For
`fem_needle_insertion` the scenario file draws the pad opaque in the frames
of the interval (`INSERTION_OPAQUE_FRAMES` in `tools/video_scenarios.py`),
because through the see-through pad the lines of its underside lie over those
of its top surface. The interval therefore stands in two files;
`tests/test_record_examples.py` fails when they differ.

### Recipes of another file and stills

```bash
uv run --locked --extra examples python tools/record_examples.py needle_thread \
    --profiles /tmp/clips/recipes.json --output-dir /tmp/clips --stills 0,60,119
```

`--profiles` reads the recipes from another file of the same form, which must
hold every example that is recorded; `docs/videos.json` is not read. `--stills`
writes the named frames (counted from 0) as `<example>.still-<frame>.jpg`
beside the clip, each as soon as it is captured: the frame as it goes to the
encoder, with the title bar and after a crop. A recipe that cannot be
recorded (an unknown camera entry, an `edges` interval or a still outside the
clip, a `crop` outside the picture) is refused before the simulation
starts. Relative paths of `--profiles` and `--output-dir` are taken from the
caller's folder. The README is generated again only after a recording into
`docs/media`.

## Checks, builds and profiling

Run the lint, CPU/CUDA test and build commands on the [main page](../README.md#check-and-build).
The [validation guide](validation.md) distinguishes completed workloads,
numerical comparison gates and performance measurements. Do not infer one
from another.

Docker mounts this repository's source for ordinary development runs; rebuild
the image after changing dependencies. `./docker/run.sh -- COMMAND ...` runs an
arbitrary container command. For opt-in Nsight capture, use
`./docker/run.sh --nsys <example>` and follow the [profiling guide](../nsys/README.md).
Captured timings include profiler overhead; raw outputs stay under ignored
`nsys/results/`. Preserve workload settings when comparing changes.

### Start-up of an example

An example captures its CUDA graphs and loads its kernel modules while it
builds its scene, not in a frame of the run. `examples/_startup.py` holds the
pieces: `Startup` opens a start-up (`begin`) at construction and at every
rebuild of the scene (a reset, a region switch, a recording switch), ends it
with a mark (`finish`) and fails in `check` if a graph was captured or a
kernel module was loaded after a mark, or if a graph captured in a start-up
was not launched there; `load_kernel_modules` loads the module builds that
first launches would load; `first_draw` issues the launches of a drawn frame
without using a frame of a frame-limited viewer, and draws nothing to a
viewer that records what is logged (the USD and file viewers, where the
start state would become a sample of the recording): there it loads the
kernel module of the viewers' log calls by name; `Script` is the scripted
user of `--test`. `fem_phantom_regions`, `needle_thread`, `fem_dvrk` and
`fem_versius` use it for every scene of their class. So does
`fem_surgical_incision`, whose check sets the captures inside a listed growth of
storage apart
([start-up, storage growth and the fixed reserve](phantom-cut.md#start-up-storage-growth-and-the-fixed-reserve)).
`fem_needle_insertion` and `fem_needle_suture` use it where the example is
built with `startup=True`, which their entry points do: the classes are
also the base of worker scenes and of test fixtures that capture where a
run first needs a graph, and those are left as they were. A scene built on
an example class keeps its own
start-up: only the most-derived class takes the mark (`owns_startup`). The
counters are those of `_src/utils/capture.py` and belong to the process, so
`test_final` checks them only in a `--test` run. On a CPU device no mark is
taken.

```bash
PYTHONPATH=tests uv run --locked --extra examples python -m unittest \
    test_example_startup test_phantom_regions_example test_needle_thread_example \
    test_recording_startup test_needle_startup test_suture_startup test_cut_startup -v
```

### Rigid step on MuJoCo Warp

MuJoCo Warp is an unmodified dependency: `pyproject.toml` pins
`mujoco-warp==3.17.0` in the `examples` extra and the lockfile takes it from
the package index. The GPU rigid backend does not call `mujoco_warp.step`. It
issues the same step from the solver's own package
[`_src/solvers/mujoco/warp_step/`](../src/newton_solver_suturebot/_src/solvers/mujoco/warp_step/),
in a form that a conditional CUDA graph can hold:

| Module | Contents |
| --- | --- |
| `step.py` | `RigidStep`: the stages of `mujoco_warp.step` in their order for the supported configuration (implicitfast, Newton solver, elliptic cones, dense Jacobian, at most 32 degrees of freedom) |
| `tree.py` | `TreePasses`: `com_pos`, `crb` and `rne` with all levels of the body tree in one launch each |
| `constraint_solver.py` | `NewtonSolve`: the Newton constraint solve on a solver context kept between steps, with fused launches |
| `scratch.py` | `StepScratch`: a Warp allocator that serves the temporary arrays of the other stages from kept buffers |
| `block.py` | Block size and block barrier of the fused kernels |

The package launches kernels and calls functions of the private modules
`mujoco_warp._src` and repeats the arithmetic of some of their kernels, so it
is tied to exactly version 3.17.0. Before changing that pin, compare the stage
sequence of `step.py` and the fused kernels with the modules `forward`,
`smooth` and `solver` of `mujoco_warp._src` in the new release.

These tests compare the rigid step and its parts with the stock step. Run
them on the CPU, where the results must be equal bit for bit, and with
`NEWTON_TEST_DEVICE=cuda:0`:

```bash
PYTHONPATH=tests uv run --locked --extra examples python -m unittest \
    test_warp_step test_warp_step_tree test_warp_step_constraint_solver \
    test_warp_step_scratch test_mjw_fused_launches -v
```

`test_mjw_fused_launches` counts the launches of both steps and needs CUDA.
`python tests/_mjw_step_parity.py` rolls the grasp model and MuJoCo Warp's own
test models out with either step and compares the trajectories.

### Profiling a conditional graph

Kernels inside `wp.capture_while` / `wp.capture_if` bodies appear in an Nsight
trace only with `--trace=cuda-sw,nvtx --cuda-graph-trace=node` and Nsight
Systems 2026.5 or newer. A node trace names only the kernels of graphs that
were built while it was recording, so recapture the graph at the start of the
capture range. Node tracing slows the run by about half; use it for the
breakdown and untraced runs for frame times.
