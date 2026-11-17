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
requires an MP4 and JPG/GIF preview for each discovered example.

## Record demonstrations

```bash
uv run --locked --extra examples python tools/record_examples.py fem_surgical_incision
uv run --locked --extra examples python tools/record_examples.py --all
python3 tools/build_readme.py --require-videos
```

Recording needs the `examples` extra, `ffmpeg` and OpenGL, and performs real
simulation or recording replay. `docs/videos.json` stores capture arguments;
`tools/video_scenarios.py` scripts interactions. Neither defines the example
list. Each clip's JSON sidecar records arguments, frame count and capture
conditions. Keep historical sidecars unchanged; new captures receive new
metadata. Playback rate is not live simulation performance. See
[provenance](provenance.md) for imported media and asset restrictions.

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
