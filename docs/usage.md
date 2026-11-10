# Example guide

This self-contained repository has four runnable examples, discovered
from their modules. FEM, cutting, needle coupling, the optional joint MuJoCo/FEM
solver and their geometry helpers are included in this package. Launch commands and available recorded previews are in the
[main page's example gallery](../README.md#examples).
The [repository layout](layout.md) identifies the local solver implementations,
example modules and bundled assets.

| Example | What it demonstrates | Details |
| --- | --- | --- |
| `fem_phantom_cut` | Simulate the full tissue pad while slicing, drilling, carving and moving material with spring grips. There is no region selector. | [Cutters, grips and numerical scope](phantom-cut.md) |
| `fem_phantom_regions` | Simulate one selected Tet4 region with fixed interfaces to inactive material; use mouse spring forces to deform it. | [Controls, region mapping and geometry](phantom-regions.md) |
| `fem_needle_insertion` | Select entry and exit material points, then drive one free MuJoCo needle automatically or use sliders to insert, retract and lift it in regional FEM tissue. | [Controls, assumptions and numerical checks](needle-insertion.md) |
| `fem_dvrk` | Replay measured da Vinci Research Kit (dVRK) tool poses and inspect four camera overlays against source videos. This static-scene kinematic replay does not advance FEM or rigid-body dynamics. | [Recording setup and calibration scope](suturebot-recording.md) |

## Setup and bundled assets

Use Python 3.12 on Linux ARM64 or x86-64. Install the `examples` extra for
MuJoCo, the viewer and USD dependencies. CUDA requires a compatible NVIDIA
driver; headless CPU runs are also supported but may be much slower.
The lockfile pins the runtime and viewer dependencies.

For Docker, run `./docker/build.sh`, then `./docker/run.sh --list` or
`./docker/run.sh <example>`. Visual runs need an X11 display and readable
`XAUTHORITY`; headless runs do not. For local development:

```bash
uv sync --locked --extra examples
uv run --locked python -m newton_solver_suturebot.examples
```

For a particular example, run `./docker/run.sh <example>` or
`uv run --locked python -m newton_solver_suturebot.examples <example>`.
Append `--help` to see its available options. The phantom and recording examples
include their prepared assets; no source-dataset checkout is needed for playback.
USD references, motion arrays, camera rays, indexed images and point labels
resolve within `src/newton_solver_suturebot/examples/assets/`. The wheel and
Docker image package these data. See [asset and evidence provenance](provenance.md).
Large recording payloads are intentional; they avoid runtime downloads.

## Full-pad cutting

Run `./docker/run.sh fem_phantom_cut` or
`uv run --locked python -m newton_solver_suturebot.examples fem_phantom_cut`.
This example activates the whole bundled pad and changes its material topology
with the copied Tet4 cutting implementation. The **Line**, **Cylinder** and
**Sphere** tools make incisions, through-holes and carved cavities; spring grips
deform the resulting material. The [cutting guide](phantom-cut.md) describes the
controls and support settings. It is a physical FEM example, separate from the
recorded-motion replay and the needle's virtual passage model.

## Physical needle handling

The needle-insertion example also bundles its standalone needle; its rigid
dynamics use the MuJoCo CPU backend and its FEM device is selected separately.
On supported CUDA configurations, repeated established-tunnel trials run inside
a captured GPU loop; native MuJoCo history and exact surface-event acceptance
remain on the host. `--no-capture-coupling` restores the earlier coupling path
for comparison. Unsupported configurations retain that path automatically.
It includes puncture, transverse shaft contact and sliding friction. Cutting
resistance and fracture work are omitted.
The needle demo uses a 25 kPa Young's modulus, while the regional-pad example
retains 30 kPa. Historical numerical comparisons used 30 kPa. The softer
setting uses an analytic compliant tunnel reaction and ALM Coulomb friction
by default. With the earlier $1/60000$ m²/N wall compliance, all 24 public
scenarios at nominal 1/4/8 steps and six additional depth/handling cases passed
their completion and integrity checks. Abrupt
2 mm clicks can still require substantial adaptive work. These checks do not
establish timestep-independent forces or performance for every selected path.

For manual needle handling, preview the selected entry/exit path, then adjust
**Insert / retract** and **Lift / lower**. Each setting directly specifies the
drive target; finite forces and torques move the simulated needle. Releasing
the mouse holds that target while the body settles. The lift control is limited
to ±2 mm in this demo; this is not a guarantee for every region or selected path.
**Pause** freezes time;
**Resume sliders** continues and **Reset** restores automatic mode. **Space**
retains the viewer's separate global pause. The interactive preset uses one
nominal physical step per 60 Hz frame; unresolved needle motion or force
nonconvergence triggers smaller physical-step retries without changing the
slider target. Accepted easy steps can grow back toward the nominal duration.
The drive integrates damping and a spring linearization through native MuJoCo;
the full slider goal remains active while physical forces determine motion.
Retries rebuild timestep-dependent FEM terms and reset only the numerical
linear-solver guess. They do not reset tissue or needle positions.
Use `--substeps 12 --coupling-absolute-tolerance 2e-7 --coupling-relative-tolerance 2e-4 --fem-tolerance 1e-5` for a finer, stricter
reference. Successful insertion does not establish timestep-independent
force histories: the [needle guide](needle-insertion.md#current-interactive-preset-and-finer-reference)
records the observed differences and qualification scope.
All imported timing/test counts are historical originating-project evidence,
not measurements of this standalone extraction. Recorded 10-substep FPS
results describe the earlier automatic workload. Loaded handling has not met
a sustained 60 FPS target; playback speed is not live simulation speed. The
[needle guide](needle-insertion.md#selection-and-controls) includes a Docker
command with a prepared path.

For repeatable needle scenarios, use
`tools/validate_needle_insertion.py --suite robustness --substep-counts 1 4 8`.
The [needle guide](needle-insertion.md#repeatable-regional-and-handling-checks)
shows the full command and distinguishes nominal counts from adaptive physical
steps. The example defaults to `--tunnel-model alm`, with an analytic compliant
radial reaction and ALM Coulomb sticking/sliding. `--tunnel-model penalty`
selects the earlier transverse spring and regularized velocity friction;
the generic public interaction still defaults to that penalty mode.
The example explicitly defaults to `--tunnel-compliance 3.3333333333333333e-6`
($1/300000$ m²/N), independently of its 60000 N/m² numerical penalty.
This fivefold stiffer wall removed the observed shaft escape away from the
opening in the reproduced region-1 2 mm lift/return/withdrawal scenario.
It remains an illustrative compliant wall, not a calibrated material law or a
hard confinement guarantee. Earlier scenario counts and FPS measurements used
the more compliant wall and do not establish performance for this new default.
`--tunnel-compliance 0` retains normal ALM for a hard local wall, whose handling
is not generally qualified. The finite-compliance ALM preconditioner uses
relative tolerance $10^{-2}$; penalty and hard-wall branches retain $10^{-3}$.

The public `SolverMuJoCoFEM` API accepts the same construction and stepping
arguments as `NeedleTissueCoupler`. It provides the simultaneous GPU
momentum/friction solver described in [the method](method.md#simultaneous-mujocofem-momentum-and-friction).
Its initial CUDA backend supports one validated MuJoCo `implicitfast` free
body, assembled FEM without differentiation, and a positive-compliance ALM
tunnel. Other configurations and boundary events use the established
integration path. The needle example retains `--needle-solver partitioned`
as its default while trajectory equivalence and interactive performance are
being qualified. To opt in:

```bash
uv run --locked python -m newton_solver_suturebot.examples fem_needle_insertion --needle-solver mujoco-fem
```

The report records the requested solver and constructed class; its
`coupling_graph_status` and `simultaneous_steps` distinguish GPU use from
fallback. CPU and unsupported configurations retain the established path.
The joint solver uses one final affine FEM correction per certification,
warm-started from its simultaneous motion solution. Corrected endpoints must
pass the configured applied-load FEM accuracy and all force, friction and
motion checks; a failed certificate can resume the joint solve and require
another correction within the same budget. This is not one FEM solve per
Newton trial, nor a guarantee of only one correction per accepted step.

The opt-in solver also uses GPU query launch sizes tuned for this single
needle workload. The default partitioned solver keeps its original launch
sizes; neither choice changes the force laws or acceptance tolerances.

Selecting this solver does not establish trajectory equivalence, 60 FPS or
general articulated-body support. The loaded-checkpoint comparison passed
physical checks but did not meet the 2 µm nodal trajectory-equivalence target;
discrete engagement histories can differ. The partitioned solver remains the
default. Materials, tolerances and slider controls are shared by both choices.
