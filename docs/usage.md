# Example guide

This self-contained repository has seven runnable examples, discovered
from their modules. FEM, cutting, needle coupling, the optional joint MuJoCo/FEM
solver, the thread solver and their geometry helpers are included in this package. Launch commands and available recorded previews are in the
[main page's example gallery](../README.md#examples).
The [repository layout](layout.md) identifies the local solver implementations,
example modules and bundled assets.

| Example | What it demonstrates | Details |
| --- | --- | --- |
| `fem_phantom_cut` | Simulate the full tissue pad while slicing, drilling, carving and moving material with spring grips. There is no region selector. | [Cutters, grips and numerical scope](phantom-cut.md) |
| `fem_phantom_regions` | Simulate one selected Tet4 region with fixed interfaces to inactive material; use mouse spring forces to deform it. | [Controls, region mapping and geometry](phantom-regions.md) |
| `fem_needle_insertion` | Select entry and exit material points, then drive one free MuJoCo needle automatically or use sliders to insert, retract and lift it in regional FEM tissue. | [Controls, assumptions and numerical checks](needle-insertion.md) |
| `fem_dvrk` | Replay measured da Vinci Research Kit (dVRK) tool poses and inspect four camera overlays against source videos. This static-scene kinematic replay does not advance FEM or rigid-body dynamics. | [Recording setup and calibration scope](suturebot-recording.md) |
| `fem_versius` | Inspect two supplied Versius distal needle holders over the full pad with generated motion and explicitly labeled dVRK stereo placeholder video. | [Preview scope and preparation](versius-recording.md) |
| `needle_thread` | Lift the free MuJoCo needle on a cable-element thread whose far end is pinned, with mouse holds on the needle and pins on the thread. There is no tissue. | [Controls and scope](#needle-on-a-thread) |
| `fem_needle_suture` | Suture a wound: the free MuJoCo needle makes a row of planned stitches through regional FEM tissue, and its thread follows through every stitch and lies on the pad. | [Controls and scope](#suturing-a-wound) |

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
to ±5 mm in this demo; validated handling sequences cover ±2 mm, and this is
not a guarantee for every region or selected path. In the default region a
+5 mm lift completes. Lowering compresses tissue against a volumetric
compression barrier (elements stiffen instead of inverting), and the run stops
with a guarded failure near a −3.6 mm target when the needle reaches the
region's fixed cut boundary; deeper targets need a deeper active region.
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
`--fem-chebyshev-degree` (default 4; 0 disables) accelerates the FEM
block-Jacobi preconditioner without changing the matrix, right-hand side or
stopping rule; the
[latency round](needle-insertion.md#november-11-single-scene-latency-round)
records its measured effects and limits.
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

`--contact-sample-stiffness` [N/m] is off by default. Without it the 192
shaft samples behind the tip share `--contact-stiffness` (20 N/m) by length,
about 0.1 N/m each, which cannot carry a needle that rests on a few of them. With it
every shaft sample is at least that stiff, with its damping raised in the same
ratio; the tip keeps `--contact-stiffness`, so the puncture criterion is
unchanged. For a needle of weight $mg$ and shaft radius $r$, $2mg/r$ keeps one
sample within one radius when the needle is set down on it without speed:
21.0 N/m for the bundled 0.3 g needle. `NeedleTissueInteraction` takes the
same value as `surface_sample_stiffness`, and its
`set_surface_sample_stiffness` changes it between steps.
The [method](method.md#needle-insertion-interaction) derives the values.

Measured on region 1 by releasing the needle 3 mm above the pad with its
weight (2.94 mN) as the only load and the drive's damping left in MuJoCo, in
twelve drops: on its tail, on its tip, lying flat and tilted, over the
tissue, over the region's fixed rim and over inactive material. Without the
option the needle sank in every drop until the interaction reported that a
sample's centre penetrated fixed or inactive tissue. With 21.0 N/m it was
carried in all twelve: the deepest sample dipped 0.12 to 0.56 radii into the
surface at most, no step was refined and a step took at most 37 interface trials.
Eleven needles lay still (slower than 0.05 mm/s for one second) within 23 s
and one was still creeping at 0.05 mm/s after 25 s.

Scope. The option also stiffens the shaft's contact with the surface during
an insertion: on the region-1 path from (0.030, −0.040) to (0.030, −0.048) the
insertion succeeded either way, with a peak interface force of 4.4 mN instead
of 3.3 mN and a largest tissue displacement of 0.21 mm instead of 0.17 mm.
The stiffness of a shaft lying along the surface grows with the number of
samples in contact. A needle without the drive's damping, falling freely, is
not carried by 21.0 N/m (eight of twelve drops failed); with the value for a
3 mm fall, $2mg(h+r)/r^2=246$ N/m, all twelve were carried with the deepest
sample at most 0.55 radii down, and the needle that fell on its tip over
tissue punctured it. The coupling then rejected 57 to 329 steps per run,
took steps down to 0.03 ms and up to 160 trials each, and needed 9 to 386 s
of computation for at most one simulated second, on a GPU shared with other
jobs. These runs started the coupler's relaxation at 0.05; with the
example's 1.0, two such drops onto inactive material did not reach 1.5
simulated seconds in 240 s.

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

## Needle on a thread

`needle_thread` has the bundled needle as a free rigid body in MuJoCo, resting
on a floor, and a 120 mm thread, 0.3 mm thick, from the needle's tail to a
pinned point on the floor. The needle and the thread act on each other; there
is no tissue, robot or knot. It runs on CPU or CUDA; on CUDA a frame of
substeps is one captured graph that includes the MuJoCo steps.

| Control | Effect |
| --- | --- |
| Right-drag the needle | Grab the nearest point of the needle and move it; releasing leaves a spring hold at the green marker. Several holds share one spring. While any hold is active the needle's weight is carried at its centre of mass. |
| Right-drag the thread | Pin the nearest thread node and move it; releasing leaves the pin. A pin prescribes the node's position. |
| Middle-click a marker | Release that hold or pin. A marker under the side panel cannot be clicked; use Clear holds. |
| Clear holds | Release everything; the needle falls back to the floor. |
| Thread stiffness slider | Young's modulus from 2 MPa to 2000 MPa, applied to the running scene. It scales the axial and the bending stiffness together. |
| Debug view | Show the thread's nodes and slope vectors, and the needle's collision capsules. |

Options: `--substeps` (default 4), `--elements` (32), `--updates`, the most
Newton updates of the thread per substep (2), and `--young-modulus` [Pa]
(2e6). `--test` grabs the needle after a third of a second, lifts it 50 mm at
0.1 m/s, lets it go a sixth of a second later, and checks the hold, the
landing and the thread. That hold has no marker; Clear holds releases it.

Two updates hold over the slider's range. In scripted runs at 2, 20, 200,
500, 1000 and 2000 MPa on CPU, and at 2, 200 and 2000 MPa on CUDA, the needle
was let go after a lift on a slack thread, after pulls of up to 300 mm beyond
the thread's reach, and by thread pins that had lifted or stretched the
thread. Every run stayed finite, the needle and thread never had more energy
than at the release, and the needle came to rest. A stiff thread on a moving
needle needs four or more updates to converge a step; with two its motion is
more damped. When every substep used all its updates, the physics of a frame
took 6.9 ms with two updates, 11.0 ms with four and 19.3 ms with eight on a
GB10. These are the best means over 60 frames from five runs on a GPU shared
with other jobs; the slowest runs took 9.5, 17.7 and 23.4 ms. Two updates
leave room in a 60 Hz frame and eight do not. With the needle at rest the
physics took about 5 ms. Rendering is not included, and no frame rate is
claimed.

Scope. The needle collides with the floor through its authored capsules of
radius 0.28 mm; the drawn mesh is thicker and reaches about 0.2 mm below the
floor at rest. The thread touches the floor at its nodes, without friction,
and does not collide with itself or with the needle. A hold on the needle is a
soft spring: a taut thread pulls the needle away from its marker. Because a
hold carries the needle's weight, a held needle does not swing under its own
weight, and one that still touches the floor lies on it without weight or
friction. A pin on the thread is rigid, so dragging pins apart stretches the
thread. The needle mass and inertia are the asset's illustrative values.

Two limits come from solving the thread on the needle as on a free body,
after the floor contact. A hold dragged beyond the thread's reach settles,
to less than 0.02 mm/s and 0.04 rad/s, when it is 5 mm or more above the
floor. At floor level the needle keeps chattering on the floor, more the
further the hold is pulled: 100 mm beyond the reach it moved at up to
0.03 m/s and 7 rad/s at 2 MPa and at 0.11 m/s and 16 rad/s at 2000 MPa; more
updates or substeps did not change that. And on a needle that lies on the
floor the thread's first node sits off the tail by the motion its pull would
give a free needle in one substep: 0.02 mm at 2000 MPa with four substeps.
With two substeps it was 0.3 mm and with one 1.2 mm while the thread still
moved after a landing, more than the thread's radius, and the scripted test
fails there.

## Suturing a wound

`fem_needle_suture` combines `fem_needle_insertion` with the thread. The run
has three parts: the needle makes a row of stitches, the thread is tied at
the first entry and pulled to close the wound, and the needle is dropped onto
the pad.

Stitches. A stitch is an entry and an exit on the active region. Without
picking, the row is planned from the first entry and exit
(`--first-entry-xy`, `--first-exit-xy`): copies moved along the wound by
multiples of `--stitch-pitch`, keeping the `--stitches` nearest ones that
pass the insertion example's feasibility checks and whose needle clears the
pad while its tail leaves. Stitches can also be picked on the map, in any
number: choose Add stitch, click an entry and then an exit; the pair becomes
a stitch if the needle can make it. Picked stitches are made in the order
they were picked and may point in any direction: between them the needle
rises until its whole circle is above the top of the pad, moves across while
it turns into the next stitch, and comes down. The stitches of the planned
row are copies of each other and are joined by turning on instead. Entries
are drawn green, exits pink.

A stitch that passes the plan's checks can still fail while the needle is in
the tissue, when the path runs close to fixed or inactive material; the run
then stops with the interaction's message, as `fem_needle_insertion` does for
the same entry and exit. Remove that stitch and pick another.

The needle makes the stitches one after the other without a reset. A stitch
is one turn about its circle's centre: in at the entry, through the tissue,
out at the exit and on until the tail has left the tissue; the rest of the
turn is made through the air while the circle moves to the next stitch. The
needle is driven by the insertion example's finite force/torque drive and
coupled to the tissue as there. It is drawn with the metal of the dVRK
grippers' jaws.

The thread's first node sits on the needle's tail. Its spare length is laid
in rows beside the first entry and lowered onto the pad before the run, where
friction holds it. The rows are laid in the direction
that keeps the whole thread on the pad and clear of the stitches; near the
pad's edge they are shortened. A stitch is refused, with the reason shown,
if the thread cannot be laid out beside the row. Where the tail has passed through
the tissue the thread stays in the stitch's track; between stitches it lies
across the wound. The thread and the tissue act on each other: the pad
carries the thread, and a taut thread pulls the track and dents the surface.
Blue spheres mark where the thread goes into the tissue and orange spheres
where it comes out, at the thread's own position.

Closing. After the last stitch the thread is tied to the tissue at the first
entry and the needle pulls on, back across the wound towards where the next
entry would be. Where the pad's edge leaves the needle less than its own
diameter of room that way, it pulls straight back across the wound or back
along the row instead. The pull ends when the wound at the stitches is
narrower than the thread is thick, when a tetrahedron of the tissue is
squeezed below `--least-volume` of its volume (0.5), or at the edge of the
pad. With the defaults the tissue limit ends it: the wound at the three
stitches narrows from 2.36 mm to 1.28 mm. A single stitch (`--stitches 1`)
closes its wound, from 2.30 mm to 0.30 mm. A row whose stitches all pass
through unbroken tissue has no wound to close and is not pulled.

Drop. The thread is then tied at the last exit as well, so the stitches stay
tight, and the drive's spring lets go of the needle. The stretched end of the
thread draws the needle back and it sinks under its weight onto the pad; the
run ends with what happened. This tests the contact between the MuJoCo needle
and the FEM tissue. That contact shares one stiffness of 20 N/m over the
length of the shaft: a needle that comes to lie along the surface is carried,
while one that stands on its end on fixed or inactive material sinks in and
the interaction reports that its centre penetrated the tissue. With one
stitch the needle comes to rest; in the default run it was still creeping at
2 mm/s after five seconds. Of four runs of two picked stitches on other
regions, two ended at rest, one sank in, and in one a tetrahedron of the
tissue inverted during the drop.

| Control | Effect |
| --- | --- |
| Pause / Start | Pause or continue the run. |
| Reset | Rebuild the scene and start again. |
| Frame row, Frame stitch, Frame pad | Camera on the whole row, on the stitch in progress, or on the pad. |
| Region | Click a region on the map to make it the tissue. The planned row exists on region 1 only; on the others choose Add stitch and pick the stitches, then press Start. |
| Add stitch | Click an entry on the map, then an exit. Every change rebuilds the scene and waits for Start. |
| Remove last, Clear, Planned row | Take the last stitch off, remove all, or go back to the planned row. |
| Spheres where the thread crosses the surface | Show or hide the blue and orange spheres. |
| Thread nodes and slopes | Show the thread's nodes and slope vectors. |

Options: `--stitches` (3), `--stitch-pitch` [m] (0.006), `--eyelet-spacing`
[m], the spacing of the thread's track points (0.0015), `--insertion-speed`
[m/s] in tissue and for the pull (0.004), `--air-speed` [m/s] between
stitches (0.012), `--thread-length` [m] (0.18), `--thread-elements` (90),
`--thread-modulus` [Pa] (2e6), `--updates` (2), `--row-length` [m] (0.06),
`--contact-margin` [m] (0.005), `--least-volume` (0.5) and `--no-autostart`.
The options of `fem_needle_insertion` apply as well; its `--shaft-friction`
(0.2) is also the thread's friction coefficient on the pad and in its tracks.

Scope. The needle follows a fixed schedule; no robot holds it. The two knots
are ties of the thread to one material point each, not knotted thread. The
thread pulls on the needle one step late, and the needle yields to it only
as far as its drive's spring and damper allow; while the needle is in tissue
the tissue holds it as well, which the thread's solve does not know, so the
pull then builds up over a few more steps. In the drop the drive's
damping stays on, because the needle-tissue coupling does not converge for
the 0.3 g needle in contact without it: the needle sinks at about 4 mm/s,
where the drive's damper balances its weight, instead of falling freely. The
drop ends when the needle has been still for half a second, or after twice
the time it takes to sink by its own diameter. The thread keeps out of itself and of the needle's
shaft, without friction between them; these contacts are found on the
straight chords between its nodes at the start of each step. Its friction on
the pad uses each node's normal force of the previous step, and its contact
with the pad acts at the nodes only, two millimetres apart, so the thread
can cut a corner of the pad between two nodes. The tissue has no contact with
itself, so closed lips can overlap.
The thread's forces reach the tissue relaxed and one step late. The thread
is stepped once per accepted needle step; on CUDA that step is a captured
graph, while the needle and the tissue step as in `fem_needle_insertion`.
The stitch geometry, thread stiffness and contact parameters are not
calibrated against measurements, and no frame rate is claimed here.

## Parallel action-controlled insertion scenes

`newton_solver_suturebot.parallel` provides `SutureSceneConfig` and
`ParallelSutureScenes` for training integrations. Each spawned worker owns an
independent native MuJoCo needle, FEM region, convergence workspaces and accepted
puncture history. Every worker receives its action before the parent waits for
results. This is process parallelism; it is not a fused GPU multi-world solver.

```python
import numpy as np
from newton_solver_suturebot.parallel import ParallelSutureScenes, SutureSceneConfig

if __name__ == "__main__":
    config = SutureSceneConfig(device="cuda:0", reset_position_noise_m=0.0007, seed=42)
    with ParallelSutureScenes(config, num_envs=8, log_dir="results/parallel") as scenes:
        geometry = scenes.geometry()
        states = scenes.step(np.zeros((8, 6)))
        states_to_reset = scenes.reset([0, 3])
```

Actions are finite values in `[-1, 1]`, ordered world-space translation and
rotation vector `[dx, dy, dz, rx, ry, rz]`. Each control step increments the
held servo target by up to 0.1 mm per translation axis and 0.01 rad per rotation
axis. A zero action holds the previous target. The finite wrench drive advances
the actual needle; policy steps never overwrite its simulated pose. The default
control interval is 1/60 s with four nominal physical substeps, and retains the
source example's material, interaction and convergence settings.

`step()` and `observe()` return lists of state dictionaries with actual needle
pose (XYZ plus quaternion XYZW), velocity, tip, deformed material entry/exit
targets, FEM positions and velocities, and the source numerical diagnostics.
Pass `include_tissue=False` to `step()` or `observe()` for a compact privileged
teacher observation: material landmarks are gathered on device and full tissue
positions/velocities are omitted from IPC. The default includes them for camera
rendering and physical validation.
Accepted events record `entry_error_m` and `exit_error_m` against the material
targets at the event time. Evaluate task success from accepted chronological
entry/exit events and physics failure status, not target pose proximity.
`geometry()` returns fixed topology, canonical ready pose, needle centerline,
material normals and reference insertion plan. Reference-plan information is
privileged controller/validation data, not a camera observation.

Resets rebuild only selected scenes, clearing numerical warm starts and
puncture history. Optional initial translation noise uses independent seeded
streams per scene; its Z offset is upward. It changes actual and target pose
only during reset before physical time advances. The default noise is zero.

The scene currently includes tissue and a freely simulated needle. It has no
articulated robot, gripper, thread, knot or camera renderer. Training projects
can render the returned geometry from separate calibrated left/right cameras.
The insertion/contact laws remain illustrative and uncalibrated. Benchmark
loaded contact separately from motion above resting tissue; eight active
processes alone do not imply an eightfold throughput increase.

The optional authored-mesh isolation and seeded-reset tests run with:

```bash
NEWTON_TEST_PARALLEL=1 NEWTON_TEST_DEVICE=cuda:0 uv run --locked python -m unittest discover -s tests -p test_parallel_scenes.py -v
```
