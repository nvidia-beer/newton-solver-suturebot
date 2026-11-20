# Numerical validation

## Physical grasp: retained outside authorization (2026-11-13)

The [compact validation report](validation/standalone/grasp-authorized-outside-2026-11-13.json)
records the captured-force extension for an already authorized needle tip that
remains outside tissue and strictly within the existing opening's clearance.
Exact current geometry and swept checks certify the branch; exact host staging
retains the same opening and events. Original force kernels, puncture criteria,
full selected region, solver tolerances and rollback are unchanged.

The full CUDA suite completed **1,021 tests in 206.077 seconds**, with three
skips, five expected failures and no unexpected failures. All 63 focused CUDA
tests passed. These counts describe the archived candidate before removing one
source-mirroring AST test and its copied-source fixture for maintainability;
the installed suite subsequently passed **1,020 tests in 201.815 seconds**,
with the same three skips and five expected failures. All 62 installed focused
cases passed in 23.877 seconds. Source and asset hashes stayed unchanged during
the full rerun. Functional
force, endpoint, threshold, ownership and rollback tests remain, and all runtime
bytes match that validated candidate. Three unprofiled candidate runs completed the 360-frame physical
procedure, including the native-clock and current-matrix residual checks.
The selected region remains 7,205 nodes and 29,160 tetrahedra, with four native
rigid substeps per FEM interval; no moving crop or prescribed needle pose is used.

An ABBA comparison used two full runs per variant. Pooled loaded-window physics
rate increased from **11.976 to 12.547 FPS**, approximately **4.8%**. Baseline
runs ranged from 11.427 to 12.581 FPS; candidate runs ranged from 12.075 to
13.057 FPS. This small sample is not a confidence interval or evidence of
sustained 60 FPS. Whole-interval force-graph fallbacks fell from 127/116 in the
baseline repeats to 40/38 in the candidate repeats. Numerical trajectories and
iteration counts vary, so individual timings are not interchangeable.

The eight-scene private-MPS validation also passed all 2,880 environment frames
and every native-clock check. It measured 35.426 loaded environment steps/s in
aggregate (4.428 physical updates/s per scene), versus 35.590 for the installed
baseline. Whole-sequence rates were 51.901 versus 52.608 environment steps/s.
This comparison does not establish an aggregate throughput improvement for
eight scenes.

A separate Nsight Systems capture of frames 219–260 covered every requested
worker command with no reported record loss. It recorded 6.072 seconds in worker
physics ranges and 3.020 seconds of GPU activity within those ranges; activity
union and host wall time are different measures. Host CUDA API elapsed time
includes waiting for dependencies and is not actual device-copy duration. The
report retains source/log hashes and the unchanged workload/budget metadata.


## Suturing inversion regression (2026-11-12)

The [original failure](validation/suture-inversion-before.json) reproduces
`fem_needle_suture` with its former one-substep default on CUDA: the second
stitch inverted tetrahedron 6910 at frame 1372, 22.867 s, with
$J=-0.00347857$. Reconstructing that interval shows that the linear volume
prediction was positive while the actual cubic nodal motion crossed zero.
Interface force convergence and the compression force alone did not certify
the geometry of the accepted step.

The coupled solver now rejects unresolved tetrahedral motion before committing
the state, then the example retries a shorter physical interval. The suturing
default is four substeps per frame to better resolve the lagged thread loads.
The material, full 7,205-node / 29,160-tetrahedron mesh, fixed boundaries,
90-element thread and commanded speed are preserved. A smaller timestep also
shortens the delay of the existing per-step force relaxation.

The [full-sequence report](validation/suture-inversion-after.json) records the
final CUDA regression, including every-frame volume and boundary checks,
accepted swept-volume minima, all 22 thread eyelets, and actual settled-drop
criteria. It records the transition into the closing phase even when the
tissue is already tight and that phase ends in the same display frame.
A timeout or a drop failure relabeled as success cannot pass this test.

The final run passed 2,008 frames / 8,032 physical steps (33.467 s simulated),
with minimum frame $J=0.15628$ and minimum accepted swept $J=0.14002$.
The wound narrowed from 2.363 mm to 1.570 mm; it was already at the tissue
limit when the closing phase began, so almost no additional pull occurred.
The needle then settled below 1 mm/s for more than 0.5 s. The complete CUDA
suite ran 623 tests with no unexpected failures, three opt-in skips and five
existing expected failures; the final full-sequence module separately passed
all seven tests. Ruff and generated-README checks passed.

`test_tet_step_guard.py` additionally checks positive endpoints with interior
inversion, extreme scales, inactive material slots, and rollback/retry through
both coupling backends. The full-row test is opt-in; its command is documented
in [usage](usage.md#suturing-a-wound). These checks cover the default row and
stated cases, not arbitrary picked stitches or calibrated tissue behavior.

## Full-pad cutting extension (2026-11-10)

The [standalone cutting report](validation/standalone/phantom-cut.json) records
checks for `fem_phantom_cut`, separately from the earlier extraction below.
That example cut the whole pad. On 2026-11-17 it was renamed
`fem_surgical_incision` and the Baymax body replaced the pad; the records of
this section are of the pad example and are kept as history.
The complete mesh has 97,489 vertices and 413,405 original tetrahedra; every
region is welded into the active pad, with no fixed nodes.

The complete standalone CUDA suite passed **494 tests in 415.280 seconds**,
with no skips. Its log is `results/phantom-cut/cuda-tests.log`. Ruff checks and
formatting passed for all 189 Python files. These are the post-extension
results; the 473-test extraction result below remains historical.

The 360-frame recorded demonstration of the pad example advanced six
simulated seconds. Its files `docs/media/fem_phantom_cut.*`, which the report
names, left the tree on 2026-11-20, the day after `fem_surgical_incision` got
its own clip ([provenance](provenance.md#earlier-clips)); they remain in the
repository's history. One accepted finite incision creates two components, then
a physical spring grip lifts and releases the separated strip. Before/after
mass sums are exactly equal in float32 at 0.1343255192041397 kg. Finite-state
and positive-volume checks passed every 60 frames and at the final state.
Frames 0, 240 and 354 were visually inspected. These sampled checks do not
establish integrity at every substep or for arbitrary cuts.

Four focused CPU cutting suites passed 19 tests in 21.023 seconds. Two
independent full-mesh tests passed in 2.663 seconds, and two warm-start rollback
regressions passed in 0.553 seconds. A full-pad two-frame headless CUDA smoke
also passed; its 82.201-second wall time includes setup. Rebuilt packaging and
an isolated Docker CPU check passed four-example discovery and full-pad
loading without network access or host source mounts.
The final rebuilt image's isolated full-pad load check passed in 6.971 seconds;
its image digest is recorded in the report.

The movie's 199.447-second wall time includes initialization, rendering and
encoding. It is not a live benchmark, and its 60 FPS encoding is playback
speed. The cutting preset has no demonstrated sustained 60 FPS result.

## Evidence scope for the standalone extraction

The extraction checks in this section cover the original three examples,
before `fem_phantom_cut` was added. They remain historical evidence for that
source snapshot; they do not qualify the new full-pad cutting workflow.

The historical solver results below describe earlier source snapshots. They
establish the scope and limitations of those implementations; they are
**not fresh passes of the current package**. Original report bytes, hashes and old namespaces
are preserved in `docs/validation/`. Fresh standalone import, build and test
results belong under `docs/validation/standalone/` and are assessed separately.
The fresh standalone CUDA suite passed **473 tests in 112.555 seconds**.
Ruff checks and formatting of 161 Python files passed; the 15 Docker launcher
tests passed in 1.678 seconds. All 8,301 audited originating files were unchanged.
Historical test counts below describe earlier source repositories and must not
be substituted for this result.

The local package/source checks are recorded in the
[standalone validation report](validation/standalone/validation.json),
[standalone audit](validation/standalone/audit.json) and
[runtime manifest](validation/standalone/runtime-manifest.json).
They verify the extracted implementation and packaged data separately from
the original solver's physical and performance measurements. The first full
standalone CUDA run found one missing test mesh, `tet_cantilever.usda`, and no
numerical test failures. Copying that fixture resolved the packaging issue;
the unchanged runtime passed the complete rerun. The raw log is preserved at
`results/migration/cuda-tests.log`.

Three fresh hidden-window OpenGL captures at 1280 × 720 completed with
`--test`, and their posters were visually inspected:

| Standalone capture | Confirmed result |
| --- | --- |
| `fem_needle_insertion`, 420 frames | Gap insertion succeeded with two punctures, entries and exits; 46.90 µm exit error, 229 accepted physical steps and zero refinements |
| `fem_phantom_regions`, 540 frames | Selection changed 4 → 6 → none → 4 with scripted 3 mm spring-grab targets |
| dVRK replay, 1,202 frames (captured as `fem_suturebot`, now `fem_dvrk`) | Four-camera recording replay covered 597 supported frames from the 601-frame source |

Capture sidecars are retained at `results/migration/rendered/`. These are
fresh execution/render checks, not live-FPS measurements or new material
calibration. The separate [540-frame phantom integrity check](validation/standalone/phantom-regions.json)
passed finite-state, positive-volume and exactly fixed-boundary checks on every
frame; its minimum tetrahedral Jacobian ratio was 0.9249783.

The rebuilt wheel passed isolated installation checks, with all 8,126 package
files byte-identical to the local package, including 8,068 asset files. Its
SHA-256 is `bd1bf20fb300df836aa2c256d3827c93964aa17146addaf939824c0359177420`.
All three isolated Docker runs passed without source-directory mounts: needle
insertion for 250 frames, phantom regions for 30 frames and recording replay
for 60 frames. Only a disposable Warp cache was mounted. The image makes its
bundled source/assets readable for the unprivileged launch user; this resolved
an inherited asset-directory permission issue without changing runtime code.
Imported gallery videos retain their original provenance separately.

The original three bundled demonstration movies are historical captures. The needle
and regional-pad movies render physical simulations; the SutureBot recording
movie is measured-motion replay over a static phantom. None supplies live FPS.
[Provenance](provenance.md) records original and copied artifact hashes.

To rerun relevant checks from this repository:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m unittest discover -s tests -v
NEWTON_TEST_DEVICE=cuda:0 uv run --locked python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 uv run --locked --extra examples python tools/validate_needle_insertion.py --suite robustness --substep-counts 1 4 8 --device cuda:0 --output /tmp/needle-robustness.json
uv run --locked --extra examples python tools/validate_phantom_regions.py --output /tmp/phantom-regions.json
```

These can take substantial time. A completed physical workload, a numerical
trajectory comparison, exact fixed boundaries, constitutive residuals and
performance are separate checks. Do not turn a failed comparison into a pass
because a movie looks plausible.

## Loaded needle frame cost and state-copy cleanup (2026-11-10)

Direct device iteration counters were added only in an ignored diagnostic
harness for the current fivefold-stiffer finite-compliance tunnel preset.
In 30 lifting frames, the partitioned solver made 313 FEM calls and 14,067 FEM
CG iterations; 30 lowering frames made 434 calls and 19,593 iterations.
Respectively 73 and 111 calls already needed zero CG iterations. Shaft
preconditioning added 4,715 and 5,766 CG iterations. There were no iteration-cap
hits, recursive-residual failures or rejected physical steps. CUDA graph replay
was active. Lowering required 110 friction dual updates versus 72 while lifting;
the remaining bottleneck is repeated global active-region linear work.

A small cleanup now initializes complete candidate state once per captured
attempt and retains the per-trial particle-force scratch copy. A 600-frame
comparison preserved the event/step schedule, finite states, exact fixed nodes
and positive saved-frame tetrahedral Jacobians. Maximum nodal difference was
1.424 micrometres and force-vector relative RMS difference was 0.0524%, within
the existing 2-micrometre/1% comparison gates. An unchanged-source repeat
differed by 1.104 micrometres. Reusing prepared FEM forces was rejected: its
3.326-micrometre nodal difference exceeded the same gate.

Two unprofiled hidden-GUI runs of the copy cleanup measured 52.99–53.14 active
overall FPS, 26.28–27.20 lifting FPS and 20.85–21.30 lowering FPS. The unchanged
baseline runs measured 51.49–51.72, 26.35–26.68 and 19.45–20.35 FPS respectively.
These are modest workload-specific gains; **60 FPS during loaded handling
remains unmet**. The workload, tissue parameters, iteration budgets and
physical acceptance criteria were unchanged. Diagnostic counter runs are not
throughput measurements, and the fresh Nsight graph trace failed kernel-name
attribution checks, so it is not used to assign kernel-category percentages.
The original ignored review was `nsys/results/needle_frame_budget_20261110/index.html`.
A portable subset of its evidence is retained under
[validation/frame-budget/](validation/frame-budget/decision.json); raw profile
databases and complete frame histories are not bundled.

All 492 CUDA-enabled tests passed in 111.667 seconds, including three new
state-publication/replay/rollback regressions; Ruff and formatting checks passed.
A separate region-1 2 mm lift/lower/withdrawal probe completed with no step
refinements and no sampled shaft escape away from openings across 2,158 engaged
anchor observations. Maximum tissue displacement was 1.757 mm. This sampled
check does not establish hard containment for arbitrary handling.

## Needle insertion and handling with ALM (2026-11-10)

At its earlier $1/60000$ m²/N wall compliance, the originating
[needle example](needle-insertion.md) passed all **30 physical
completion and integrity cases**: eight public scenarios at nominal 1, 4 and 8
substeps, plus six shallow/middle/deep handling cases. The full CPU suite ran
395 tests with 13 CUDA-only skips; all 395 CUDA tests passed. Source manifests
match the physical checks, clean UI measurements and reviewed movies. The
[final evidence report](validation/needle-insertion-final-20261110.json)
retains parameters, hashes, failed criteria and historical configurations.

The preset uses 25 kPa tissue, an analytic finite-compliance normal tunnel
reaction and ALM Coulomb friction. These are illustrative parameters. The
checked region-1 gap, signed handling, direct ±2 mm clicks, and separate
region-2/3 insertions do not establish arbitrary-pair or hard-wall feasibility.

Clean NVIDIA GB10 measurements used the normal GUI, event polling and buffer
presentation in a hidden 1280×720 window. They exclude initialization and the
visible desktop compositor. The identical smooth handling sequence gave:

| Nominal substeps | Active overall FPS | Lifting FPS | Lowering FPS | Accepted steps / retries |
| --- | ---: | ---: | ---: | ---: |
| 1, two runs | 56.13–56.93 | 29.87–34.54 | 21.52–21.59 | 364 / 0 each |
| 4 | 26.46 | 14.08 | 11.11 | 1456 / 0 |
| 8 | 18.15 | 10.25 | 8.54 | 2912 / 0 |

**Sustained 60 FPS during loaded handling remains unmet.** Nominal-1 p95
active-frame latency was about 44 ms; abrupt clicks can require additional
iterations and smaller steps. The [canonical gap movie](media/history/fem_needle_insertion/fem_needle_insertion.mp4)
and separate handling movie are 60 FPS simulation-time exports, not live-rate
measurements. The handling capture shows local lifting and relaxation without
deformation magnification; its 1 mm lift reached 0.757 mm maximum tissue
displacement in the reported simulation.

**Force-history convergence also remains unmet.** Nominal 1 versus 8 automatic
cases differed by 87–228 µN force-vector RMS (20–65%) on a common 60 Hz grid.
Some differences persist away from entry/exit events. Four steps generally
agree more closely with eight, which is itself not established as converged.
Separate analytic-normal and faster-preconditioner comparisons failed their
declared 2 µm nodal-history gates (16.40 and 12.46 µm); completion and integrity
passes do not erase those failures. See the guide's
[accuracy discussion](needle-insertion.md#current-alm-scenario-checks-and-accuracy-limits)
for sampling, event-window and matching-step-schedule distinctions.

## SutureBot wound regions (2026-11-03)

The final catalog uses one public numbering scheme, 1–12. Only the original
connected volume containing regions 5, 8 and 12 is subdivided. Nine other
original payloads remain byte-identical, and the full mesh retains its original
coordinates and connectivity. The lower-left wrap remains in region 8; only
the thin upper rim joins region 5. The prepared domains contain 18,790
tetrahedra in region 5 and 9,801 in region 12.

The complete 140-test suite passed with `NEWTON_TEST_DEVICE=cuda:0`. The CPU
run passed 135 tests and skipped five CUDA-only checks. Additional 30-frame
mouse-grab checks of regions 5 and 12 produced measurable deformation with
finite states, positive Jacobians and exactly fixed cut/underside anchors.

The retained [regional replay report](validation/phantom-regions.json) covers
all 540 frames of the video workload on NVIDIA GB10: wound 4, wound 6,
no active regions, then wound 4 again. Their actual FEM sizes were
11,795, 13,909, zero and 11,795 tetrahedra. The null viewer replays the same
physical ray picks and spring targets as the visible mouse controls.

Every replay frame had finite state, positive tetrahedron Jacobians and exactly
stationary cut/underside anchors. The minimum Jacobian was 0.924977. The 3 mm
grab-target motions produced maximum nodal displacements of approximately
0.811 mm and 1.258 mm in wounds 4 and 6. These are checks of this scripted
demonstration, not fitted tissue parameters or guarantees for arbitrary pulls.

The nine-second [actual GPU recording](media/history/fem_phantom_regions/fem_phantom_regions.mp4) has 540
decoded frames at 1280 × 720 and 60 fps. It shows clicks on the 2D region map, active
tetrahedron counts and mouse-grab markers; its [capture metadata](media/history/fem_phantom_regions/fem_phantom_regions.json)
retains the selection and deformation measurements. The movie's playback rate
is not a live performance result. It is the historical recording, archived on
2026-11-20. The [current clip](media/fem_phantom_regions.mp4) replays the same
script at the same size from a close camera, without `--test`; its
[capture metadata](media/fem_phantom_regions.json) records maximum nodal
displacements of 0.885 mm and 1.299 mm in wounds 4 and 6 and 0.883 mm in the
second drag of wound 4: its grabs are picked through the pixels of the close
camera, not along the example's own view as in the null-viewer replay. The
validation command below, which takes its arguments from the clip's recipe,
reproduced the replay's numbers above on 2026-11-19 (minimum Jacobian
0.924977, 0.811 mm and 1.258 mm).

```bash
uv run --locked --extra examples python tools/validate_phantom_regions.py --output /tmp/phantom-regions.json
uv run --locked --extra examples python tools/record_examples.py fem_phantom_regions --width 1280 --height 720
```

## Legacy SutureBot rigid mechanism and tissue contact (2026-11-03)

These retained results describe `_suturebot_contact_demo.py`, the earlier
approximate physics experiment. Menu item `fem_dvrk` now runs the separate
recording replay described below; its current video does not show this press.

All four [rig/contact tests](../tests/test_suturebot_example.py) passed with
Newton device `cpu` and again with `cuda:0`. These tests use MuJoCo's CPU backend
for rigid dynamics. They check force and torque balance for a transformed body,
driven RCM and jaw motion with a fixed fixture, needle settling on the pad,
rejection of invalid control targets without partial updates, and an active
MuJoCo contact constraint between a jaw and the free needle. The jaw/needle
test deliberately places the needle at a jaw to verify contact response; it
does not establish a successful pickup or sustained grasp.

A [540-frame press-and-release report](validation/suturebot-press.json)
on `cuda:0` exercises the earlier scripted `--demo` trajectory,
with a driven jaw pressing the training-wound surface and retracting.
The maximum sampled PSM1 jaw force delivered specifically to free FEM nodes
was 0.02701 N, with up to eight active contact samples. Jaw/active-tissue contact
was observed in 147 press/hold frames. The largest nodal displacement relative
to the settled pre-press state was 0.5634 mm; after retraction and recovery it
was 0.0807 mm (about 86% recovery). The maximum RCM line-to-pivot error over the
sequence was approximately `1.15e-8 m`. Every frame was checked for finite
states, positive tetrahedral volumes and exactly fixed boundary nodes.
These measurements demonstrate local contact deformation and
recovery with the chosen demonstration parameters. They do not validate tissue
material properties, needle puncture, suture thread, knot tying or a suturing
policy, and they contain no live performance measurement.

The [runtime press tests](../tests/test_suturebot_press.py) check the ordinary
demo path, motion of the three free vertices on its target triangle, unloading
after retraction, elastic recovery, and plan reset on region changes. A separate
empty-selection case lets the needle contact fixed tissue and requires zero
active-tissue force/counts, preventing fixed-pad contacts from passing as FEM
interaction. The contact kernel tests also cover partially and fully fixed
triangles without changing physical force/torque conservation.
The complete suite, including these regressions, passed all 142 tests with
`NEWTON_TEST_DEVICE=cuda:0`. The earlier nine-second recording independently
checks the same press/release criteria and measured zero final jaw load on
active tissue.
The additional full 540-frame CPU press run was terminated with exit 143 before
completion, without a test traceback; it is not counted as a CPU pass. The
CPU contact/rig tests and the empty-selection contact regression passed.

```bash
uv run --locked --extra examples python -m unittest discover -s tests -p test_suturebot_example.py -v
NEWTON_TEST_DEVICE=cuda:0 uv run --locked --extra examples python -m unittest discover -s tests -p test_suturebot_example.py -v
uv run --locked --extra examples python tools/validate_suturebot_press.py --device cuda:0 --output /tmp/suturebot-press.json
```

## SutureBot example scope update (2026-11-09)

Needle and thread reconstruction have been removed from the four prepared
recording bundles, their compatibility alias and the example's runtime controls.
The recordings retain their tools, cameras, fixture, table, phantom and available
UV point labels. Source recordings and reconstruction research remain in the
MSc workbench. Preparation now rebuilds caches containing these extensions and
publication rejects them, retaining the four-camera image-validation requirement.
This removal does not change the existing camera-alignment estimates or promote
any failed candidate. The selector lists four distinct recordings; the former
fifth choice remains a direct-name compatibility alias of the default.

Validation of the scoped check-in: the complete staged FEM suite passes
**175 tests on CUDA** (172.354 s). The built wheel passes 21 replay and
selector tests on CPU. Ruff checks and formatting pass. The separate
preparation/publication checks pass 46 tests and 16 subtests. Bundle checks resolve all 27 USD layers locally
and verify that all 8,024 retained nonmetadata payloads are byte-identical to
the pre-removal copies. The refreshed 1,202-frame, 1280 × 720 demo renders all
597 supported source frames with coverage in each of the four camera overlays.
These are replay and packaging checks; camera alignment remains unvalidated.

The dated reports below describe earlier snapshots. Their needle/thread assets,
shape counts and runtime behavior are historical evidence, not the current
example's contents. Original payloads removed from the FEM bundles are archived
under `MSc/SutureBot-workbench/outputs/simulation/example_feature_removal_20261109/`.

## SutureBot recorded placement replay (2026-11-03)

The [prepared replay report](validation/suturebot-recording.json) covers
`tissue_3/2_needle_throw/20250428-215300-489212`, region 1. Runtime and packaging
checks pass; the camera, pad and needle alignment remains an unaccepted visual
estimate. The package retains
all 601 source timestamps exactly. The fitted image-to-robot offset is
`-0.1069642378 s`; frames 0–3 lack pose support and are excluded from playback
and timed tool fitting. The 597 supported frames are 4–600. Wrist optical
poses equal the registered roll-link poses multiplied by their constant
mount transforms. The render bodies and camera trajectories share one clock.

The current default translates only the pad by **(+0.2584, +0.0775, +1.0083) mm**
in world coordinates, retaining its orientation, regional geometry and material
labels. Tools, cameras, fixture, table and the original needle grasp stay fixed.
At source frame 127, the original needle's rendered sharp tip is approximately
0.075 mm from the translated surface and 0.461 mm from the insertion marker.
This is a conditional placement adjustment, not confirmed recorded contact.
The two fixed material labels project 9.21 and 8.63 pixels from their original
source clicks; they were not rebound to conceal that discrepancy. Aggregate
pad fitting/reserved residuals change from 3.61/8.13 to 6.38/8.80 pixels.
The original calibration tables and clearance audits below are retained as
**pre-translation evidence**; current placement and preservation checks are in
the bundled `pad_placement.json` and the replay report.

The core replay USD imports 14 kinematic bodies and 27 visual shapes. With the
separate rigid needle and its collision support, the runtime imports 15 bodies
and 61 shapes. All twelve
phantom reference-position arrays and exterior triangle indices match the
prepared native source, with no repartition or coordinate change. Original
dataset and phantom input hashes remain unchanged. The phantom remains static,
with no tissue puncture or force inference. The added needle has separate
rigid-body dynamics, described below.

The constrained estimate includes visible wound crests, the unchanged negative-Y
base side, tool landmarks, observed shaft edges and sparse tool masks. Each arm
has one fitted rigid registration tied exactly to its RCM port; no per-frame
pose correction is allowed. Wrist optical axes are parallel to their shafts.
The pad normal is parallel to the fixture support, with a free displacement
relative to its feet. Camera focal lengths and principal points are fitted;
distortion retains its earlier estimated prior. The packaged
[constrained report](../src/newton_solver_suturebot/examples/assets/dvrk/recordings/tissue3_215300/constrained_fit_metrics.json)
records each feature's fitting and reserved residuals in original video pixels,
plus the optical parameters and mechanical checks.

The local optimizer stopped on `xtol` after 30 evaluations, with no active
bounds. Its reported first-order optimality was approximately 357; this is not
a stationary or globally optimal solution. Wrist optical centers are fitted
47.59/50.24 mm behind their wrist-pitch pivot origins, with radial offsets
6.69/6.08 mm and zero axis tilt (PSM1/PSM2). The pad underside is 17.92 mm above
the fixture feet/table plane. These are conditional estimates. The estimated
stereo baseline is 7.60 mm with an 8.73° relative frame rotation (8.64° between
forward axes); displaced
principal points and coupled intrinsics/extrinsics prevent interpreting them
as uniquely measured optical parameters.

| Projected feature | Fitting RMS (px) | Reserved RMS (px) |
| --- | ---: | ---: |
| Shaft, PSM1 wrist | 5.23 | 2.39 |
| Shaft, PSM2 wrist | 2.09 | 2.61 |
| Pad side, PSM1 wrist | 1.31 | 9.09 |
| Pad side, PSM2 wrist | 1.74 | 1.99 |
| Wound, left endoscope | 5.78 | 9.04 |
| Wound, right endoscope | 5.98 | 8.34 |
| Wound, PSM1 wrist | 4.58 | 13.88 |
| Wound, PSM2 wrist | 5.16 | 11.55 |

Only one physical pad side is observable; opposite edges and corners are not
validated. Wound annotations have 7–10 pixel uncertainty, boundary annotations
5–6 pixels, and shaft annotations 6–9 pixels. Frames 300 and 525 are excluded
from all image objectives of this constrained stage. Some earlier initialization
and diagnostic stages exposed those images, so the split is a reserved-feature
check rather than a blind whole-pipeline benchmark.

The historical lip-only estimate had reserved pad-edge RMS of 58.51/88.05 pixels
for PSM1/PSM2. A later soft fixture fit reduced those values to 9.79/2.68 pixels,
but allowed wrist-axis tilt and required coincident pad/fixture support heights.
Those results and its near-contact clearance values belong to the preserved
baseline; they do not describe the current hard-parallel, free-offset fit.

Independent native-resolution rendering of the actual shaft triangles gives
reserved RMS of 2.71/2.31 px, versus 24.39/38.12 px for the published baseline
(PSM1/PSM2). This audit preserves edge identity and uses 3/12 original reserved
points; PSM1 has only a short frame-525 fragment. The analytic table above uses
resampled ideal-cylinder curves. The retained polygonal tube's finite endpoint
projection and approximately 1 px raster quantization explain why these are
separate measurements. Actual phantom-only raster checks give reserved
frame-300 pad-side RMS of 8.21/3.48 px and frame-525 RMS of 11.78/2.24 px.
The 240-wide pad render has about 2.7 native pixels per wrist pixel; residual
quantization and the larger PSM1 mismatch remain visible.

The fixture feet define the table plane. Exact RCM-port closure and parallelism
are enforced constraints, not measured calibration precision: the CAD port
centers have approximately 3 mm uncertainty. The transparent endoscope follows
the stereo optics and uses the recorded 154.98 mm ECM tip-to-pivot distance for
length. This does not determine its optical angle. The straight-scope convention,
diameter, optical hand-eye and third fixture opening remain unmeasured.

Geometric clearance audits sample distal jaw surfaces across all 597 supported
frames. Minima below distinguish Euclidean nearest-surface distance from signed
clearance to the topmost phantom surface along world Z or its base normal:

| Jaw | Nearest surface (mm) | World-Z clearance (mm) | Base-normal clearance (mm) |
| --- | ---: | ---: | ---: |
| PSM1 jaw 1 | 0.724 | 1.073 | 1.068 |
| PSM1 jaw 2 | 0.891 | 0.984 | 0.979 |
| PSM2 jaw 1 | 9.364 | 12.351 | 12.350 |
| PSM2 jaw 2 | 6.228 | 10.396 | 10.287 |

No negative sampled surface or table clearances were found. These measurements
do not certify exact mesh contact or grasp success.
A held needle can span a legitimate tool-to-pad gap; the rigid-needle
visualization does not independently resolve metric depth. Exact PSM2 instrument
CAD and the physical jaw geometry under needle load are unavailable; the static pad also cannot
reproduce local recorded deformation. These limitations remain visible in the
[actual frame-300 mesh overlay](media/fem_suturebot_alignment.jpg).

The additional source-only shaft review contains 21 short wrist-image segments
(63 original-pixel points), with 6–9 pixel uncertainty. It reserves frames
300 and 525; PSM1 frame 300 has no reliable observed edge and is excluded.
No stereo long-cylinder tangent is inferred from a dark vignette or a visible
milled connector. The observed edges belong to the reflective outer shaft,
not its narrow central reflection. Since each wrist camera moves with that
shaft, these repeated observations test mount/lens consistency rather than
independently measuring global tool height.

The partial-mask review adds 35 records over fitting frames 150/450 and reserved
frames 300/525: 213 points on 51 external contour fragments, 70 known-foreground
points and 60 known-background points. Native-resolution masks label only small
reviewed point disks; all other pixels remain ignored. Jaw observations are
audited separately rather than fitted until their geometry/opening is verified.
PSM2 silhouette residuals include the documented surrogate-CAD discrepancy.
The independent full-scene USD render gives reserved head-contour RMS of
8.27/5.66 px in PSM1/PSM2 wrist views and 10.31/7.53 px in left/right stereo
views. Reviewed head foreground points lie inside the rendered arms, but some
stereo background points are incorrectly covered; those remaining mask errors
are reported separately. Jaw checks remain diagnostic, including a PSM2
reserved contour RMS of 8.59 px and missed PSM1 jaw foreground points. These
actual-mesh checks use full-arm visibility and original points, so they differ
from the fit's head-only, mixed contour/foreground/background objective.

The two existing dataset point labels are rendered as an ordinary USD UV
texture on the original tetrahedral exterior triangles. The nearest visible
surface hit determines each triangle and barycentric binding; face-varying UVs
preserve the source mask coverage without changing reference vertices or triangle
indices. The rendered markers use bright green RGB `(0, 255, 0)` for visibility;
the original mask remains unchanged, with source and display policies recorded
separately. Before the pad translation, an actual native-resolution camera render
recovered all 162 labelled source pixels. After translation, the same material
labels move with their triangles: 159 green pixels are visible in the left image,
with source-mask IoU 0.0355. That remaining image mismatch is reported explicitly.
Strict material-point color checks still pass on CPU and CUDA. The replay does not evolve
the FEM vertices; the material bindings can follow the same vertices if they
are deformed.

The needle uses recorded jaw hysteresis for grasp/release. A fresh open-to-closed
transition translates a free needle's nearest centerline point to the grasp
center without rotating it in the world. Each later release and fresh closure
can repeat this assistance. The resulting tool-to-needle transform stays fixed
while held, and the public Newton MuJoCo solver advances it after release.
Nearby closed-jaw capture and handoff remain available. Closure events are
consumed while another tool holds the needle, preventing a stale closure from
causing delayed capture. This is imposed replay assistance, not a physical
contact-grasp or image-alignment result. Needle dynamics use its CPU backend by default, including during CUDA
rendering. No recorded needle trajectory is stored or consumed. Tests cover
release velocity and gravity, collision support, nearby closed-jaw capture,
handoff, repeatable backward seeking and clock mismatch rejection. The physical
CAD end labels are corrected: the tapered endpoint is the sharp tip. Its image
placement remains explicitly unaccepted. The only static environment collider
is the flat rectangular backing beneath the raised tissue, approximately
133.19 × 132.48 × 4.45 mm. Raised tissue, tools, fixture, endoscope and table
have no needle collision shapes. A held needle follows its imposed grasp and
can pass through the raised tissue; after release, MuJoCo lets it fall onto
the backing. No tissue puncture forces, deformation or suture dynamics are
modeled. Passing these tests does not establish the recorded grasp,
needle-to-pad contact or suturing success.

Rendering checks cover asymmetric intrinsics, rotated/translated optical
frames, nearest-surface occlusion, moving geometry, unavailable frames and
the absence of artificial edges at coplanar region seams. Before the later
fresh-closure assistance change, all 21 then-existing replay,
camera, UV and needle tests pass against the extracted wheel on CPU (6.313 s)
and CUDA (7.007 s), including both actual-texture and material-binding tests. The
portable-package test copies the entire sample before loading it, checks that
USD dependencies stay inside the copy, and rejects workstation paths.
Seven preparation camera tests additionally verify inverse-lens rays, source-image
dimensions, fixed camera mounts, reserved curve splits and physical constraints.
They independently resample the source Cartesian poses and verify that one
constant arm transform preserves every supported relative motion increment.
Together with six projection/constraint tests, five actual-mesh mask tests and
two input-preservation regressions, all 20 preparation tests pass. Three actual
source frames (150, 300 and 525) were rendered through all four cameras and
reviewed separately; the independent boundary audit covers seven frames. The
fresh wheel includes 2,478 bundle files: all 2,404 source JPEGs, 35 reviewed mask
rasters, the UV texture and the rigid needle assets. Every bundled asset is
byte-identical to the published sample. It passes all 597 supported replay
frames from an extracted copy outside the repository, with every USD and
texture dependency resolving inside the bundle.
The later green-marker update changes the atlas colors and display metadata;
source images, source masks, USD bytes, material bindings, motion and cameras
remain unchanged. Both focused UV tests pass again on CPU and CUDA, and five
preparation tests pass, including exact preservation of atlas coverage during
recoloring. The refreshed wheel also passes the dependency and full supported-frame
replay checks. Later pad-placement checks verify unchanged local geometry,
UVs, tool/camera arrays and needle grasp. The new backing regression checks
one actual USD box, free fall through raised tissue, and a stopped needle on
that backing after 400 ms of MuJoCo dynamics. All 597 frames replay with the
initial grasp held through frame 303, release at 304, and deterministic backward
seeking in that historical run. Its proximity-only rule did not produce a later
capture. The current fresh-closure assistance and its new audit are described
separately below.

Four additional source episodes (tissue3_212958, tissue4_211216, tissue5_102359
and tissue7_200743) were prepared and rendered in the MSc workbench. All were
reviewed as region 1; this does not validate automatic classification of all six
dataset regions. Their remaining tool and pad discrepancies fail image
alignment acceptance, so none is included in the example package. Successful
export, coordinate consistency and UV rendering are separate from calibration
acceptance.

The full CUDA suite passed all 162 then-existing tests in 286.227 seconds after
the pad translation, including the retained FEM contact regressions. The later
backing-only asset change and added regression are covered by the 21 focused
tests above; no full 163-test run is claimed. Repository-wide
Ruff checks and formatting also pass. Earlier runs with CUDA-context memory
errors and the interrupted full CPU run remain recorded as historical checks;
they are not substituted for this completed run.

The [historical full viewer recording](media/history/fem_dvrk/fem_dvrk.mp4) contains 1,202 frames at
1280 × 720 and 60 fps, with four simultaneous overlays. Its
[capture metadata](media/history/fem_dvrk/fem_dvrk.json) confirms all 597 supported source
frames were displayed; the regenerated capture took 50.880 seconds with CUDA rendering and the
MuJoCo CPU needle backend. The close-up from 3 to 14.17 seconds makes both green
surface marks and the frame-127 approach visible in the main OpenGL viewport.
Encoded frames at 4.1 and 10 seconds were inspected separately from
the camera overlays. Source frames 444/448/464/494 additionally show the free-to-held
transition and later tool following. Hashes of the final MP4, GIF, poster and metadata are
recorded in the validation JSON. Playback rate is not a dynamics or live-performance
measurement. The self-contained sample is documented in its
[README](../src/newton_solver_suturebot/examples/assets/dvrk/recordings/tissue3_215300/README.md).
Preparation commands and full audit arrays remain separately under the MSc
workbench in `docs/recording-replay.md`; the example does not require it.

```bash
PYTHONPATH=tests uv run --locked --extra examples python -m unittest test_suturebot_recording test_suturebot_cameras test_suturebot_point_labels test_suturebot_needle -v
NEWTON_TEST_DEVICE=cuda:0 uv run --locked --extra examples python -m unittest discover -s tests -v
uv run --locked --extra examples python tools/record_examples.py fem_dvrk --width 1920 --height 1080
```


## SutureBot bounded approach replays (2026-11-03)

Two additional portable samples are included in the example selector:
`tissue3_222119_approach` and `tissue3_224616_approach`. The
[structured approach report](validation/suturebot-approaches.json) records their
current bundle hashes, scoped image residuals and portable checks separately
from the default needle example above.

| Sample | Source frames, inclusive | Duration | Reserved image frames | Historical incision RMS in left / right / PSM1 / PSM2, px |
| --- | --- | ---: | --- | --- |
| `tissue3_222119_approach` | 5–141 | 4.527 s | 33, 76 | 2.05 / 1.72 / 1.75 / 12.64 |
| `tissue3_224616_approach` | 4–106 | 3.401 s | 27, 61 | 4.03 / 4.59 / 9.45 / 5.66 |

The user now confirms the entire `224616` recording is public region 1,
including both visible incision fragments. Nearby feature A is tentatively
region 2. This supersedes the earlier region-3 interpretation. The bundled
placement and green label bindings at that identity-review stage still came
from the historical region-3 fit and were marked invalid for the corrected
identity. The current bundle subsequently received a region-1 pose and rebound
labels; its alignment remains unaccepted. The table's
`224616` residuals describe that old fit; they do not validate a region-1
placement. All twelve native meshes and their canonical IDs remain unchanged.
The correction and exact prior metadata are retained in `audit/region_identity/`.

Source-only inspection defined these initial approaches before visible tissue
deformation, independently of the fit residuals. The full 575/461 frame arrays,
original timestamps and reserved-frame roles remain intact. Normal playback and
the frame slider stop at the declared bounds. Later frames remain available for
diagnostics; the complete recovery is not validated against a static phantom.

Both historical fits passed the original incision/cap phase screen: at least two reserved incision frames per
camera, incision RMS at most 25 px and median at most 12 px, plus separate cap
and source-motion checks. These are engineering thresholds, not pixel-exact
agreement. The superseded region-3 fit for 224616 has a 20.38 px maximum PSM1
incision residual and 12.27 px maximum right-image residual. Full-clip reserved stereo incision RMS
remains above the declared approach errors; those full-clip values remain in
the bundled reports. Own PSM1 cap centers are not observable and are not invented.

The original four-camera reviews identify corresponding tool neighborhoods and
incision direction while retaining visible jaw, head and groove-bank differences.
PSM2 uses surrogate CAD. Both `alignment.accepted` and
`absolute_metric_accuracy_validated` remain false. The pad bottom is approximately
20.2 mm above the estimated fixture-foot table plane for 222119 and 17.8 mm for
224616; these dimensions are not independently measured or a physical mounting
validation. No exact needle contact, complete suturing or tissue dynamics is claimed.

**Recording 4 boundary correction (2026-11-04):** independent native-resolution
USD silhouette checks found reserved outer-edge RMS of **193.94 px in PSM1 and
159.86 px in PSM2** at source frames 27 and 61. These fail the 12 px RMS / 30 px
maximum thresholds and the 80% pad/background-side recall checks. The earlier
incision-only publication result is superseded: the bundle now records a failed
alignment check and is retained for inspection. Geometry, recorded motion,
cameras and region numbering were not changed by this status correction.
The bundle's `audit/pad_boundary/` contains the native observations, rendered
review and numerical report. New publication requires this separate boundary
check; a passing incision trace cannot qualify the pad rectangle as aligned.

The two dataset labels are bound to the mesh and rendered green in all four
views. Recording `224616` reproduces all 162 source-mask pixels after rebinding
to region 3. Recording `222119` preserves its original material bindings through
the new pad translation: at annotation frame 574, outside the replay interval,
the two points move about 14.8/15.2 px from their source pixels and mask IoU is
0.0. Rendering success is separate from this source-label mismatch. The
original mask files, clicks and mesh regions are preserved. The final-frame
annotation is a static material target, not a measured contact event during
the approach.

Recording `222119` now contains an estimated needle after jointly fitting its
shape and fixed grasp, PSM1 optics/mount and a constant pad translation. Original
reserved frames 33/76 remain excluded from fitting. Needle reserved RMS is
4.66 px and maximum error 11.48 px, within the original thresholds. All 137 held
poses replay deterministically with at least 5.60 mm clearance from the backing.
Native renders still show a thicker needle and approximately 10 px endpoint
error at frame 139. The tip also has shallow 0.025–0.077 mm raised-surface
occlusion at frames 138–140; complete surface-contact alignment is unaccepted.

The PSM1 first-frame head RMS improves from 8.54 to 7.75 px and shaft RMS from
15.47 to 11.61 px. Held-out head/shaft checks improve too, while opposite-tool
cap RMS worsens from 8.88 to 11.41 px and PSM2 incision RMS from 7.27 to 12.64 px.
These tradeoffs and visible CAD differences remain explicit; passing the existing
thresholds does not establish full camera alignment. The earlier needle fit is
retained as predecessor evidence. `224616` still has no needle because its fit
retained roughly 25–30 px early sharp-tip discrepancies. Penetration through
raised tissue is permitted; only the flat backing is a collision surface.

Before the region correction, the same batch preparation command was repeated with both cache hits and all
nine recorded geometry, motion, calibration, clock and screening artifacts
byte-identical per sample. Source-input preservation, coordinate checks and
actual camera rendering pass. Portable publication retains relative USD/texture
paths, original image clocks and bounded playback metadata. The full FEM CUDA suite passes all 164 tests in 216.751 seconds with the
default backing sample and both approach bundles. The structured report records
this run separately from the earlier default-only checks and the isolated-wheel
validation. All 105 final workbench recording tests pass in 10.646 seconds,
including source-label preservation, pad-report regeneration, stale-feature
guards and phase/publication policy regressions.

Preparation remains in `MSc/SutureBot-workbench`, using
`config/recording_batch/newton_approach_examples.json`. The Newton runtime loads
prepared data without refitting cameras or importing the preparation checkout.


### Full supported pickup replay

`tissue3_215248_pickup` is also bundled. It retains all 282 original source
entries and replays all supported frames 4–281 (9.226 seconds) from
`tissue_3/1_needle_pickup/20250428-215248-974340`. The entire supported motion
precedes visible tissue deformation; there is no excluded later tail. Source
frames 0–3 remain unavailable under the recorded robot clock.

Independent actual-mesh reviews cover reserved frames 39, 94, 149, 204 and 259
in all four cameras. Reserved incision RMS is 6.48/8.27/10.04/10.27 px in
left/right/PSM1/PSM2. The source-defined phase and original reservations remain
unchanged. Tool neighborhoods and incision direction follow the images, while
jaw shape/opening, near-shaft contours and groove-bank widths retain visible
mismatches. The estimated pad bottom is 18.15 mm above the fixture-foot table
plane; metric mounting accuracy is not validated.

This recording has no source clicked-point CSV or mask. Its manifest records
`source_labels.available: false`; there are no invented UV markers and no needle
assets. It replays tools used during pickup, not a validated needle grasp.
Calibration remains `accepted: false`. Source preservation, metadata repair and
unchanged-input cache proofs are retained alongside the actual render review.
The structured report includes its current bundle hashes and final isolated-wheel
checks; the full 164-test CUDA run above predates this third asset addition.


Before the fresh-closure needle change, the validated wheel contained the
default sample and all three new pose replays. It was 126,689,059 bytes with SHA256
`f2b7ceb0c7b07eb0c15e13e359974afc02e178f298b8a7e6b18d4e554ab9b4b6`.
For that wheel, an isolated `python -I` process loaded the extracted package
outside the checkout, checked all 5,448 files across the three new bundles byte-for-byte, and rendered
all four cameras on CUDA at each replay's first, last and original reserved
frames inside its scope. All passed without the source dataset or preparation
checkout. The post-pickup CUDA recording suite also passes all 11 tests in
11.685 seconds across all four bundles, covering source clocks, all-frame
replay, sample switching and copied-package independence. Final Ruff checks,
formatting of 122 Python files and `git diff --check` pass. These runtime and
portability checks do not change any sample's estimated calibration status.


### Publication image gate

The three new scoped bundles each pass all 28 required publication image checks.
The gate rechecks the declared numerical limits and reserved observations in
all four views, visual-review readiness, exact reviewed overlay hashes and the
unchanged geometry, camera and motion inputs. Runtime or rendering success alone
cannot substitute for a missing or failed image review.
`--publish-review-required` only permits a clearly labeled absolute-calibration
estimate after those checks; it cannot waive image failures or include a failed
needle initialization. Only the original `tissue3_215300` reference without automatic-fit metadata
retains its legacy estimate status; every other new recording requires the
four-camera image scope. All three new samples still have `alignment.accepted: false`.


### Repeated needle capture on fresh jaw closure

The default recording now releases the needle at source frame 304 and captures
it again at frame 446. The new closure translates its nearest centerline point
8.436046 mm to the PSM1 grasp center while preserving world orientation. The
measured grasp-point error is 1.31e−9 m. Across all 597 supported frames, the
needle is held for 455 frames and free for 142; backward replay reproduces the
poses exactly. All 12 audited geometry, initial-needle, camera and motion assets
remain byte-identical.

Eight focused needle tests cover multiple release/closure cycles across both
arms, preserved capture orientation, fixed held transforms, consumed stale
closures, nearby handoff, free dynamics/backing collision and the actual
frame-446 replay. The initial needle image placement remains unaccepted.
Repeated closure translation is deliberate replay assistance, not evidence of
physical grasp contact, calibrated needle position or suturing success.


After the repeated-capture change, the complete FEM CUDA suite passes all
167 tests in 240.473 seconds. All 113 preparation tests pass in 11.701 seconds,
including strict four-camera publication checks and the explicit original-reference
exception. The current 1,202-frame viewer video covers all 597 supported source
frames and records the 4/304/446 initial-grasp, release and recapture events.
Current runtime, media and unchanged-input hashes are in
[the default recording validation](validation/suturebot-recording.json); these
supersede the older proximity-only needle audits retained as history.


The wheel built after the capture change, before the region-3 correction, is
126,693,990 bytes, SHA256
`0832189164641f07babdbce0840d19f338af941731ba9b8333567a0466b4a782`.
Isolated CPU checks import that needle runtime from its extracted copy,
verify all 7,928 files across the four bundles byte-for-byte, pass eight needle
tests including actual capture at frame 446, and render 20 source frames through
all four cameras across the three pose-only samples. Runtime needs no source
checkout. The final-wheel CUDA attempt was incomplete because its context
allocation ran out of memory; that failure is retained separately. The successful
167-test repository CUDA suite is separate evidence, not a claim that the final
extracted wheel completed CUDA checks. No other GPU process was changed.
Those wheel checks do not validate the current region-1 identity correction,
the superseded region-3 placement, or a replacement calibration.

### Current portable examples (2026-11-04)

The [current pre-push report](validation/suturebot-prepush.json) records 33 passing
CPU replay, camera, needle, CLI and documentation tests, plus two material-label
tests for each corrected recording. The new wheel imports independently and
matches all 7,958 bundled SutureBot files byte-for-byte. Ruff and formatting pass.
CUDA context creation failed because of GPU memory pressure. The full CPU run
was terminated during the legacy active-tissue press test after 156 passes and
five skips; the remaining recording tests pass in the focused run. A separate
legacy press rerun had not finished at commit review, so a completed full-suite
run is not claimed for this revision. These checks do not certify camera alignment.

### Initial oriented thread estimate (2026-11-04)

This subsection records the estimate before the hard-attachment update below.

The default recording has an optional source-frame-4 thread reference. The
[structured report](validation/suturebot-thread.json) separates geometry and
runtime checks from unresolved image alignment. This is a partial reconstruction
of two visible sections, not a complete suture or tracked trajectory.

| Section | Geometric length | Left RMS | Right RMS | PSM1 RMS | PSM2 RMS |
| --- | --- | --- | --- | --- | --- |
| Needle-side section | 20.48 mm | 2.30 px | 3.28 px | 3.36 px | Unobserved |
| Diagonal section | 35.47 mm | 1.14 px | 2.17 px | 2.29 px | 4.12 px |

These native-pixel residuals cover annotated spans used in fitting. Worst
point residuals are 11.20 and 12.27 px, respectively. Source PNG bytes and their
hashes are preserved. The underlying cameras, motion, rig, phantom and needle
payloads are unchanged.

The [actual four-camera mesh render](media/fem_suturebot_thread.jpg) uses the
imported USD tubes, complete-scene occlusion and native inverse-distortion rays.
The needle-side section has a 4.35 mm gap from the modeled needle attachment.
Its tail also remains visible below the tool in PSM1, outside its annotated
visible span: whole-render surface-to-guide error reaches approximately 53 px
at the 95th percentile and 66 px maximum. This discrepancy is retained. No
camera-specific visibility mask, invented connector or calibration correction
conceals it. Image alignment remains unvalidated.

Offline checks verify analytic spline evaluation, nonzero tangents, integrated
component arclengths, unit sign-continuous quaternions, local +Z/tangent agreement
and matching USD triangle tubes. The imported thread adds two visual shapes and
zero bodies, joints or collision shapes. Runtime tests verify exact source
timestamps, malformed orientation rejection, and removal when playback leaves
the reference frame. Physical twist, stress-free geometry, forces and later
motion are outside this validation scope.

### Additional initial thread references (2026-11-06)

The metrics here precede the hard-attachment update below.

Recordings 2, 3 and 4 now include portable oriented splines and visual-only USD
tubes. The five CLI choices cover four distinct recordings; choice 5 uses the
default recording's existing thread. Preparation now supports a recording
without a needle and constrains the complete spline against its primary source
guide, preventing unsupported detours from escaping that coverage residual.

| Recording | Source frame | Left RMS | Right RMS | PSM1 RMS | PSM2 RMS |
| --- | --- | --- | --- | --- | --- |
| `tissue3_215248_pickup` | 4 | 1.64 px | 3.14 px | **14.18 px** | 2.93 px |
| `tissue3_222119_approach` | 5 | 0.92 px | 1.94 px | 2.28 px | Unmeasured |
| `tissue3_224616_approach` | 4 | 2.05 px | 2.94 px | 3.77 px | Unmeasured |

These are centreline residuals on fitting guides, not independent four-view
validation. Actual imported-mesh renders retain important discrepancies:
recording 2 has a clear PSM1 offset; recording 3 has 100% guide-to-surface
coverage within 5 px in its three observed views, but its PSM1 surface-to-guide
95th percentile is 8.65 px; recording 4 has PSM1 guide-to-surface and
surface-to-guide 95th percentiles of 7.74 and 12.63 px. PSM2's clipped fragments
in recordings 3 and 4 have no quantitative image-fit claim. A lower-residual
alternative for recording 2 introduced unsupported endpoint curls and was not
installed. No cameras or source annotations were moved to conceal these errors.

Each installation verifies that the existing tool, pad, rig, camera, needle and
motion payloads remain unchanged. The actual replay was constructed for all
four recordings on CPU: each reference paused at its declared frame, disappeared
on seeking away, and could be enabled again. The new references add one visible
shape each, with zero bodies, joints or collision shapes. The original default
adds two shapes. These checks do not validate image alignment, hidden thread
connectivity, physical twist or subsequent thread motion.

The [structured additions report](validation/suturebot-thread-additions.json)
links to each bundle's native four-view render and source evidence. Detailed
known discrepancies also appear in the workbench's `add_examples_20261106/issues.html`.

### Hard needle attachment (2026-11-06)

The needle-side references in choices 1, 2, 3 and 5 were refitted with their
endpoint fixed at the needle CAD's blunt end cap. The anchor is stored in
needle-local coordinates and transformed by the unchanged source-frame pose.
Tangent and orientation remain free. Choice 4 has no needle asset and retains
its unanchored partial fragment. The separate diagonal fragment in choice 1
also retains unknown connectivity.

| Recording | Previous endpoint gap | Constrained gap | Left RMS | Right RMS | PSM1 RMS | PSM2 RMS |
| --- | --- | --- | --- | --- | --- | --- |
| `tissue3_215300`, needle-side section | 4.35 mm | 0 mm | 2.64 px | 2.78 px | 3.29 px | Unobserved |
| `tissue3_215248_pickup` | 1.09 mm | 0 mm | 1.73 px | 2.89 px | 14.19 px | 3.07 px |
| `tissue3_222119_approach` | 1.60 mm | 0 mm | 1.17 px | 2.10 px | 2.39 px | Unmeasured |

The [attachment report](validation/suturebot-thread-attachment.json) includes
the actual four-camera mesh renders and replay checks. All five choices load,
pause, remove the reference on seeking away, and restore it when enabled again.
The attached endpoints differ from the actual replay needle anchor by less
than 0.000002 mm after float32 conversion. Export, installation and runtime
reject inconsistent endpoints, stale needle poses/assets and detached USD tube
caps. Refitting against the current manifest is required if the scene has
changed since an older thread installation.

Attachment does not resolve the remaining image errors. The default PSM1
unsupported tail remains visible (surface-to-guide 95th percentile 51.69 px).
Pickup PSM1 remains offset (guide-to-visible-surface 95th percentile 22.80 px).
The approach's three observed guides stay within 5 px, while its PSM1
surface-to-guide 95th percentile is 11.85 px; its clipped PSM2 fragment has no
quantitative claim. These are fitting and visibility checks, without independent
four-view calibration acceptance. The 2,483, 1,206 and 2,384 existing core files
in the three updated bundles were preserved; only thread payloads and their
manifest/provenance/validation metadata changed, plus the default bundle's
README describing the corrected attachment and matching thread metadata in its
choice-5 alias. Thread motion, forces and
physical material twist remain unmeasured.

## Physical grasp and worker thread (2026-11-12)

The explicit `physical_grasp` worker has an independent native MuJoCo needle,
two finite-drive distal tools and a worker-owned ANCF thread. This is a new
control contract; earlier six-action free-needle policy results and scripted
stitch demonstrations do not validate a learned physical grasp policy.

The [focused thread report](validation/physical-thread-tests.json) records ten
passing CUDA tests in 3.696 s: seven in
[`test_thread_rigid_contact.py`](../tests/test_thread_rigid_contact.py) and three
in [`test_worker_thread.py`](../tests/test_worker_thread.py). They check
simultaneous opposing jaw contacts, finite-difference energy/gradient
consistency, recovery of a specified penetrated contact fixture, moving and
rotating friction anchors, the lagged Coulomb bound, fresh force/world-torque
balance, body-specific attachment exclusions, complete contact-state rollback,
worker load feedback and independent worker state. Position sampling and
repeatedly reading stored loads do not advance physical time or friction state.
A separate CPU subset ran 96 tests in 48.755 s with no unexpected failures,
two existing skips and five existing expected failures. It included the
unchanged legacy thread-contact API, solver/tissue tests and stitch geometry.
These counts describe those specific runs, not a new full-project test total.

The historical [five native grasp tests](validation/physical-grasp-focused-cuda.log)
passed with CUDA Newton state and the native MuJoCo CPU backend in 1.747 s,
before the elliptic-cone/tight-solve correction below. This earlier test
version did not include the later angular-transport assertion. Its pass is
not a validation of the final contact configuration.
They cover independent needle topology, holding
against gravity, contact-driven transport, release on opening, and slip under
sufficient load. The test explicitly measures physical state; it does not
infer retention from tool/needle proximity or impose a weld.
The transport fixture commands 5 mm and 0.1 rad at 1/240 s intervals and checks
the 0.02 N·m jaw-effort bound. The overload fixture applies a 1.5 N tangential
load and requires more than 5 mm of slip; opening requires more than 0.5 m of
free fall over 0.5 s. These are test conditions and assertions, not calibrated
holding-force limits. Three additional CPU lifecycle tests in
[`test_grasp_scene.py`](../tests/test_grasp_scene.py) passed: actions preserve
independent state, rejected multi-body trials restore native/output state, and
adaptive retries advance thread history only over accepted intervals.

A historical [full-worker initial hold](validation/physical-grasp-initial-hold.json),
also before that contact correction and using pyramidal contacts,
advanced 30 control frames, 120 accepted physical steps and 0.5 s of simulated
time without a solver failure or timestep refinement. At the final snapshot,
the two jaw normal forces were 0.39931 and 0.39945 N, the thread head/tail
discrepancy was 3.82 nm, and its coordinates and velocities were finite. The
report records zero needle entry/exit events and zero active thread/capsule
candidates at that final snapshot. Thus it checks coupled initialization and
a short physical hold; thread/tool contact under load is exercised by the
separate focused fixtures, not established by this quiet final state. The
6.57 s measured run time is not a sustained training-throughput benchmark.
Inherited `drive_*` fields in this early report came from the free-needle
example and should not be read as identified physical gripper gains.

The [method](method.md#physical-distal-grasp-and-worker-owned-thread) specifies
compliant discrete chord contacts, simultaneous body/radius groups, lagged
friction and relaxed reactions. Fresh opposing forces and torques balance;
loads applied one step later need not balance instantaneously. The two-update
thread budget is not a full-convergence guarantee. There are no worker eyelets
or puncture-track constraints, and the free end is not tied. No completed
suture, wound closure, knot, autonomous pickup, calibrated dVRK dynamics or
learned physical-grasp policy is certified by these results.

### Physical grasp contact correction

The [preserved initial eight-world gate](validation/physical-grasp-failed-gate.json)
failed in every world after 0.342–1.004 simulated seconds. Each reported
`Needle center penetrated fixed or inactive tissue; reduce timestep`.
The [single-world action replay](validation/physical-grasp-failed-actions.json)
reproduced world zero at 0.341667 s. At the first commanded frame, native jaw
normals fell from about 0.399 N to 0.00134 N and zero, and needle speed rose
to 39.6 mm/s while the stored thread attachment force remained about 22 µN.
This locates the observed transient in the rigid grasp response; it does not
establish a unique initiating numerical cause. The quiet hold above did not
exercise this motion.

The corrected native settings are elliptic cones, `impratio=10`, tolerance
`1e-12`, 200 Newton iterations and 50 line-search iterations. The model retains
impedance 0.9999, rolling friction length 20 µm, torsional length 0.28 mm,
geometry, masses, gains and friction coefficients. Cone selection changes
the friction model; convergence settings do not calibrate the instrument.
The [comparison and hash manifest](validation/physical-grasp-contact-comparison.json)
preserve the measured configurations and distinguish successful fixtures from
failed stronger commands.

| Measured workload | Result |
| --- | --- |
| Smooth 5 mm / 0.1 rad native transport, 1/240 s | 18.13 µm relative translation drift, 5.605 mrad rotation drift; no lost jaw contact, minimum normal 0.37490 N |
| Five focused native tests under corrected settings, CPU, including angular transport | All five passed in 1.378 s; [raw log](validation/physical-grasp-elliptic-tests-cpu.log) |
| Full-worker 100 µm incremental X stress, 4 intervals/frame | Failed at 0.525 s |
| Same X commands plus 0.01 rad incremental X rotations, 4 intervals/frame | Failed at 0.304167 s |
| Full-worker X stress, 16 intervals/frame | 128 frames / 2.13333 s / 2,048 accepted intervals, no failure or refinement; relative drift 21.95 µm / 8.644 mrad |
| Full-worker X plus rotation, 16 intervals/frame | Same duration/count, no failure or refinement; relative drift 21.33 µm / 7.953 mrad, minimum normal 0.31817 N |

The 128-frame stresses begin with eight zero-command frames, then reverse
command signs every ten frames; jaw opening increments are ±0.001 rad per
control frame. Relative drifts in the last two rows are measured from the end
of the eight-frame warmup. The [archived stress trace](validation/physical-grasp-elliptic-stress-16.json)
contains actual needle/palm poses, relative poses, native forces and final
diagnostics. These trials used CUDA FEM/thread state and native MuJoCo CPU
integration, and 16 complete coupled intervals, not four outer intervals
with rigid-only subcycling. No completed suture, action-space-wide stability
guarantee or eight-world gate is established by these single-world runs; the
separate final gate is reported below.

The final [native-subcycling probe](validation/physical-grasp-subcycling-stress.json)
repeats both stress commands with four outer FEM/thread intervals and four
native rigid steps inside each. Each completed 128 frames, 512 accepted outer
intervals and 2,048 native microsteps without refinement, failure or needle
entry/exit events. The pure-X and combined probes respectively measured
21.56/21.18 µm relative translation drift, 8.659/7.975 mrad rotation drift and
minimum jaw normals 0.31873/0.31823 N. Their measured loops took 4.38/4.60 s,
versus 12.71/12.86 s for 16 complete intervals. This roughly 2.8–2.9× single-world
probe difference excludes initialization and is not an eight-world training
throughput result.

Ten focused tests passed on [CPU in 2.505 s](validation/physical-grasp-subcycling-cpu.log)
and [CUDA Newton state in 3.173 s](validation/physical-grasp-subcycling-cuda.log),
both with native CPU MuJoCo. The set includes native grasp, multi-body
transaction lifecycle, exact comparison with four explicit native steps, and
an output-particle sentinel proving that rigid subcycling leaves FEM output
untouched. These are distinct from the earlier ten thread/contact tests and
are not a new full-suite count.

### Final eight-world gate and native integration

The [final eight-world report](validation/physical-grasp-eight-scenes.json)
passed all 17 checks at solver fingerprint
`1ae8d9eee6c5bbb9d2668c17398acb8965f277754cc4748c552f5f0468fa64cb`.
Each scene retained 7,205 active FEM nodes, 29,160 tetrahedra, seven independent
rigid bodies and a 90-element thread. After eight warmup and 120 measured
control frames, all eight worlds reached 2.13333 simulated seconds with
512 accepted outer intervals, no refinements or solver failures, and no entry
or exit events. Minimum tetrahedral volume ratios exceeded 0.9999851; fixed
boundaries, distinct action response and selective reset checks passed.
Final jaw normal forces were approximately 0.381 N per jaw.

The measured 960 environment transitions took 7.5143 s: **127.757 environment
steps/s**, or 0.266× real time per scene at a 60 Hz control rate. Worker startup
took another 13.405 s. The workload is oscillatory gripper/jaw motion above
tissue, not insertion or a sustained loaded-grasp test. Positive contact
diagnostics refer to the last solved native interval, not an independently
reconstructed endpoint contact certificate.

The separate [Isaac Lab native check](validation/physical-grasp-native-isaaclab.json)
completed eight steps in eight environments with time-limit resets. It
verified 72-value teacher state, 20-value command observations, stereo RGB
history shaped `[8,12,144,192]`, and 56 separately mirrored/rendered rigid
bodies under surgical appearance version 2. The
[artifact manifest](validation/physical-grasp-native-artifacts.json) binds
runtime, source, commands and files; wrapper wall time was 34.201 s. This is
an environment lifecycle/render integration check, not policy training or a
completed physical-grasp suture.

The final [full source CUDA-configured suite](validation/physical-grasp-full-cuda.log)
ran **646 tests in 147.523 s**, with three skips, five expected failures and
no unexpected failures or errors. Individual CPU-only fixtures still run on
CPU. An [earlier attempt](validation/physical-grasp-full-cuda-initial.log)
reported six test-wrapper argument errors; updating those test-only wrappers
for the external-force keyword did not change the frozen physics sources.
The full-suite count is separate from downstream Isaac Lab project tests.

### Short scripted needle/FEM entry with physical grasp

The [entry probe](validation/physical-grasp-fem-entry.json) commands the finite
gripper servo along a reference path while the independent needle moves
through contact. It does not assign simulated needle poses. The
[exact probe](validation/physical-grasp-fem-entry-probe.py.txt) and
[console](validation/physical-grasp-fem-entry.log) preserve this diagnostic's
scope and sampling. One scene completed 91 frames / 1.51667 simulated seconds
with 364 accepted outer intervals and no refinement or solver failure.
Puncture and entry occurred by frame 71; accepted entry error was 0.46212 mm.
The probe continued for another 20 frames and stopped deliberately, with one
entry and zero exits.

An independent control-frame volume audit found minimum ratio 0.894073
relative to the initial state. Fixed nodes remained bitwise unchanged and
sampled FEM, rigid-body and thread states remained finite. The separate
solver integrity checks still operate at accepted outer intervals. Peak
needle/tissue interface force was 3.130 mN and peak nodal deformation
183.4 µm. Final track engagement was 1.024 mm, jaw normals were about
0.381 N each and thread head-to-needle-tail discrepancy was 5.25 nm.
The 9.889 s wall measurement includes scene creation. This establishes a
short scripted needle/FEM entry under contact grasp; it is not a trained
policy, full entry/exit, complete suture or direct grasper–FEM collision test.


### Embedded-needle inspection lift (13 November 2026)

The 600-frame single-scene procedure passed with a 0.5 mm commanded tool lift,
0.489 mm actual needle lift and 0.187 mm upward tissue displacement from the
pre-lift state. All observed tetrahedra stayed positive, fixed boundaries stayed
within 1e-8 m, and native accepted clocks stayed within 1e-8 s. The original
frictional grasp and joint controls remained active. The arc paused for the
inspection motion, returned and completed entry/exit. The complete CUDA suite
ran 1,096 cases in 302.400 seconds (five skips, five expected failures).
[Machine-readable evidence](validation/standalone/grasp-inspection-lift-2026-11-13.json).
Use `--lift-height 0` to retain the historical insertion-only benchmark.
