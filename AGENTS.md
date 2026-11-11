# SutureBot solver development

This repository is self-contained: keep required FEM, cutting, MuJoCo needle interaction,
geometry, assets and examples here. Do not import a sibling solver package or
modify installed Newton, Warp or MuJoCo code. Preserve upstream licenses.

FEM mechanics live in `src/newton_solver_suturebot/_src/solvers/fem/`, the
simultaneous needle/FEM solver in `_src/solvers/mujoco_fem/` and the suture
thread solver in `_src/solvers/thread/`. Preserve physical
laws, sparse operators, fixed boundaries, gradient contracts and convergence
checks during refactoring. Recorded-motion replay is not physical insertion.

Examples use the public package API; shared viewer helpers belong in `examples`.
Discover runnable `example_*.py` files automatically. Each needs a module
docstring and a `__main__` guard. Keep the seven examples and their
required data bundled; external dataset preparation is optional.

Keep shared platform models under `assets/dvrk/models` and `assets/versius/models`;
recordings hold motion, images, calibration, small placement layers and point-label reports.
Their `asset_root: "../../../"` permits references within the common examples
assets tree. Portable exports must include that tree, not just a recording folder.
External preparation lives in the source project's `preparation/newton_shared_assets/`,
after the `newton_versius` and `newton_dvrk_materials` authoring stages.
Runtime must not import that source project or require its preparation environment.
Keep generated motion distinct from measured recordings and label dVRK placeholder
videos explicitly. Camera order comes from metadata, not a fixed view count.

Use uv, Warp kernels and unittest. Keep ordinary modules and direct calls.
Document equations/gradient assumptions in docs/method.md, paper mappings in
docs/papers.md and prerequisites/controls in docs/usage.md. Generate README.md
with `python3 tools/build_readme.py`; keep one real demonstration video per
example and distinguish playback rate from measured live performance.

Run `uv run --locked ruff check .`, `uv run --locked ruff format --check .` and
`NEWTON_TEST_DEVICE=cuda:0 uv run --locked python -m unittest discover -s tests -v`.
Use nsys/profile_examples.py for opt-in profiling; keep results under ignored
nsys/results/. Preserve workload resolution and numerical budgets in comparisons.
