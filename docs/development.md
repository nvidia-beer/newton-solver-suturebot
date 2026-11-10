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
uv run --locked --extra examples python tools/record_examples.py fem_phantom_cut
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
