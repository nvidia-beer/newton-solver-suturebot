# Example guide

This self-contained repository has nine runnable examples, discovered
from their modules. FEM, cutting, needle coupling, the thread solver and their geometry helpers are included in this package. Launch commands and available recorded previews are in the
[main page's example gallery](../README.md#examples).
The [repository layout](layout.md) identifies the local solver implementations,
example modules and bundled assets.

| Example | What it demonstrates | Details |
| --- | --- | --- |
| `fem_surgical_incision` | Cut the bundled Baymax body, 135 mm across and on its back on the floor: slice, drill and carve it and move material with spring grips. Nothing is held and nothing is cut by itself yet. | [Cutters, grips and numerical scope](phantom-cut.md) |
| `fem_phantom_regions` | Simulate one selected Tet4 region with fixed interfaces to inactive material; use mouse spring forces to deform it. | [Controls, region mapping and geometry](phantom-regions.md) |
| `fem_needle_insertion` | Select entry and exit material points, then drive one free MuJoCo needle automatically or use sliders to insert, retract and lift it in regional FEM tissue. | [Controls, assumptions and numerical checks](needle-insertion.md) |
| `fem_dvrk` | Replay measured da Vinci Research Kit (dVRK) tool poses and inspect four camera overlays against source videos. This static-scene kinematic replay does not advance FEM or rigid-body dynamics. | [Recording setup and calibration scope](suturebot-recording.md) |
| `fem_versius` | Inspect two supplied Versius distal needle holders over the full pad with generated motion and explicitly labeled dVRK stereo placeholder video. | [Preview scope and preparation](versius-recording.md) |
| `needle_thread` | Lift the free MuJoCo needle on a cable-element thread whose far end is pinned, with mouse holds on the needle and pins on the thread. There is no tissue. | [Controls and scope](#needle-on-a-thread) |
| `fem_needle_suture` | Suture a wound: the free MuJoCo needle makes a row of planned stitches through regional FEM tissue, and its thread follows through every stitch and lies on the pad. | [Controls and scope](#suturing-a-wound) |
| `fem_needle_grasp` | Insert a needle that an articulated dVRK tool holds by jaw friction into regional FEM tissue, with the thread, under procedural joint control. | [Controls and scope](#standalone-articulated-needle-grasp) |
| `fem_tool_tissue` | Press and slide the jaws of an articulated dVRK tool on regional FEM tissue along a scripted path of eight phases, with the tool/tissue contact off, on its first version or on the sweep of the capsules, and measure every frame; in the eighth phase the jaws pinch a ridge of the tissue and, with the swept contact, grip it and lift it. No needle and no thread: the test rig of that contact. | [Phases, options and scope](#tool-and-tissue-contact-rig) |

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

## Surgical incision (was full-pad cutting)

Run `./docker/run.sh fem_surgical_incision` or
`uv run --locked python -m newton_solver_suturebot.examples fem_surgical_incision`.
Until 2026-11-17 this example was `fem_phantom_cut` and cut the whole pad. It
now loads the bundled Baymax body, 135 mm from hand to hand, lying on its back
on the floor, and changes its material topology
with the copied Tet4 cutting implementation. The **Line**, **Cylinder** and
**Sphere** tools make incisions, through-holes and carved cavities; spring grips
deform the resulting material. The [cutting guide](phantom-cut.md) describes the
controls and support settings. It is a physical FEM example, separate from the
recorded-motion replay and the needle's virtual passage model.

On CUDA this example and `fem_phantom_regions` capture their frame in one
graph of their own: tangent refresh, grips and FEM step, here also surface
contact and the volume guard.
The kernels of that frame read the time step from the solver's `step_time`,
not from a launch argument. Code that captures `SolverFEM.step` in its own
graph in the same way must have the time step written when the graph is
launched: call `solver.step_time.set(dt)` before `wp.capture_launch` if
anything may have written another time step since the capture, and call
`solver.refresh_tangent(q, dt)` outside capture first when the time step
differs from the assembled one. A captured step uses the time step the arrays
hold at the launch, not the one of its capture.

Start-up. Run as an example on CUDA, `fem_surgical_incision` captures and launches
its graphs (the line cut for one and for two knife patches, the convex
subtraction of the Cylinder and Sphere tools, and the frame) and loads its
kernel modules before the first frame, and again inside **Reset body and
cuts**. After a start-up an edit captures a graph only where it makes storage
grow, and each growth is listed. `--cut_reserve 1` reserves fixed storage
instead: nothing grows and no edit captures, and an edit beyond the reserve is
refused. On the whole pad that reserve was tens of GiB; growing storage is
still the default, and the reserve is not yet sized for the Baymax body.
`--test` plays a scripted user and fails on a capture outside a listed growth
or on a kernel module load after start-up. Details and measured numbers:
[start-up, storage growth and the fixed reserve](phantom-cut.md#start-up-storage-growth-and-the-fixed-reserve).

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

Both solvers default to one sample per GPU block for the small shaft and
surface query batches. Explicit captured-query launch overrides remain
available; these scheduling choices leave force laws and acceptance
tolerances unchanged.

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
substeps is one captured graph that includes the MuJoCo steps. The graph is
captured before the run for the example's fixed substep, which stock Newton's
MuJoCo step keeps inside it. It is also launched once before the run, so
that the first frame does not build its executable; the state that this
launch and the eager frame before the capture changed is put back, including
MuJoCo's time and warm start. The kernel modules of the thread view load with
the scene, and the scene is drawn once before the first frame; where the
debug view can be switched on, by the panel or by the script of `--test`, it
is drawn once with it as well.

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
The script then goes on as a user would: it pins the middle node of the
thread and lifts it 20 mm, moves the stiffness slider to the middle of its
logarithmic range, lifts the needle 30 mm by a second hold, clears the holds,
moves the slider back and switches the debug view on and off. The run takes
251 frames; a viewer with a shorter frame limit, such as the default of 100
frames, is given that many. The landing is checked on the state at the end,
a second after the last release. On CUDA the self-test also fails if a graph
was captured or a kernel module was loaded after start-up, or if the frame
graph was not launched there, and prints these counts.

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

The default uses four physics substeps per frame (`--substeps 4`, 240 Hz).
The additional lagged thread forces need these smaller intervals: the former
one-step default inverted a tissue element during the second stitch at
22.867 s. Coupled steps now check tetrahedral volume before acceptance and
retry unresolved motion with smaller intervals. If retries are exhausted,
the example stops at its last accepted state. Mesh resolution, material and
needle speed are unchanged.

The opt-in full-sequence regression checks every frame, all three stitches,
threading, wound narrowing and a settled drop:

```bash
NEWTON_TEST_FULL_SUTURE=1 NEWTON_TEST_DEVICE=cuda:0 \
  uv run --locked python -m unittest discover -s tests -p test_suture_full_sequence.py -v
```

See the [inversion regression evidence](validation.md#suturing-inversion-regression-2026-11-12).

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
pad. The archived demonstration narrowed the wound at the three stitches
from 2.36 mm to 1.28 mm, and a single stitch (`--stitches 1`) from 2.30 mm
to 0.30 mm; those measurements predate the four-substep default. A row whose stitches all pass
through unbroken tissue has no wound to close and is not pulled.

Drop. The thread is then tied at the last exit as well, so the stitches stay
tight, and the drive's spring lets go of the needle. The stretched end of the
thread draws the needle back and it sinks under its weight onto the pad; the
run ends with what happened. This tests the contact between the MuJoCo needle
and the FEM tissue. That contact shares one stiffness of 20 N/m over the
length of the shaft: a needle that comes to lie along the surface is carried,
while one that stands on its end on fixed or inactive material sinks in and
the interaction reports that its centre penetrated the tissue. With one
stitch the needle came to rest in the archived demonstration; its default run was still creeping at
2 mm/s after five seconds. Of four runs of two picked stitches on other
regions, two ended at rest, one sank in, and in one a tetrahedron of the
tissue inverted during the drop. These are historical outcomes, before the
pre-acceptance tetrahedral motion check.

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
graph, which reads the step size from device data and so also serves the
shorter steps of a retry, while the needle and the tissue step as in
`fem_needle_insertion`.

Start-up. Run as an example, the scene captures its CUDA graphs and loads its
kernel modules where the row is built: while the scene is constructed and at
every rebuild (Reset, a region switch, a changed row). There the thread's
step graph is captured and launched once, and for both state buffers the
graphs of the coupled needle and tissue step are captured and launched as in
`fem_needle_insertion`
([start-up and self-test](needle-insertion.md#start-up-and-self-test)); the
thread, the needle and the tissue are then put back as they were. Start and
the frames after it capture no graph, build no graph executable and load no
kernel module, through every entry and exit of the needle, the pull and the
drop. `--test` fails otherwise and prints the counts. It runs the frames it
is given (100 by default): the whole demo is checked with
`--num-frames 2100 --test`, and by the opt-in test
`tests/test_suture_full_sequence.py`, which builds the example with its
start-up and ends with the same check.
Measured on cuda:0 with the null viewer on 2026-11-17, in runs of 2113 frames
(the whole demo, done at frame 2007, then a Reset and 82 frames of the second
run; the shaft flag flips seven times): no graph is captured, no graph
executable is built and no kernel module is loaded in any frame, with a warm
and with an empty kernel cache. Before, 51 graphs were captured in 23 frames
(31 in the demo, 12 of them at the six flips of the flag) and 29 kernel module
builds were loaded after construction. The first frame takes 33 ms (224 ms
before; 36 ms with an empty kernel cache, where the 67 module builds compile
during construction). Construction takes 12.0 s instead of 8.8 s with a warm
kernel cache and a Reset 5.1 s instead of 4.7 s. One frame of the third stitch
(frame 1580, the last exit of the tip) took 580 to 600 ms in two of three runs,
before as after: its four coupled steps evaluate 181 and 193 trials there. It
holds no capture and no load.
The stitch geometry, thread stiffness and contact parameters are not
calibrated against measurements, and no frame rate is claimed here.

## Parallel action-controlled insertion scenes

`newton_solver_suturebot.examples.grasp.parallel` provides `SutureSceneConfig` and
`ParallelSutureScenes` for training integrations. Each spawned worker owns an
independent native MuJoCo needle, FEM region, convergence workspaces and accepted
puncture history. Every worker receives its action before the parent waits for
results. This is process parallelism; it is not a fused GPU multi-world solver.

Each worker limits its BLAS libraries to one CPU thread with `threadpoolctl`.
NumPy may load BLAS before the worker starts, so setting environment variables
alone is too late. This prevents each independent scene from creating another
large CPU thread pool. Limits cover scene setup, execution and cleanup; the
parent process keeps its existing settings.

```python
import numpy as np
from newton_solver_suturebot.examples.grasp.parallel import ParallelSutureScenes, SutureSceneConfig

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

The default `free_needle` scene includes tissue and a freely simulated needle.
It has no gripper, thread, knot or camera renderer. The explicit
[`physical_grasp` mode](#physical-distal-grasper-mode) adds distal tools and a
worker-owned physical thread. Training projects can render the returned
geometry from separate calibrated left/right cameras. The insertion/contact
laws remain illustrative and uncalibrated. Benchmark
loaded contact separately from motion above resting tissue; eight active
processes alone do not imply an eightfold throughput increase.

The optional authored-mesh isolation and seeded-reset tests run with:

```bash
NEWTON_TEST_PARALLEL=1 NEWTON_TEST_DEVICE=cuda:0 uv run --locked python -m unittest discover -s tests -p test_parallel_scenes.py -v
```

### Physical distal-grasper mode

The source API keeps `free_needle` as its default for compatibility with the
existing examples and six-action policies. Select the physical tool task
explicitly when constructing workers:

```python
import numpy as np
from newton_solver_suturebot.examples.grasp.parallel import ParallelSutureScenes, SutureSceneConfig

if __name__ == "__main__":
    config = SutureSceneConfig(device="cuda:0", control_mode="physical_grasp")
    with ParallelSutureScenes(config, num_envs=1, log_dir="results/grasp") as scenes:
        states = scenes.step(np.zeros((1, config.action_dimension)))
```

Save this as a file and run it normally: spawned workers cannot import a Python
stdin entry point. The seventh action increments total jaw opening by up to
0.02 rad per control step; the first six increment the first gripper's Cartesian
wrist target. Bounded IK on a separate native-model scratch state converts that
pose to six finite joint-position actuator targets. It never overwrites live
body poses or needle coordinates. An unreachable target fails instead of moving
the fixed trocar. Zero holds all current targets. The second tool
stays at a separately held parked target. The independent needle moves under
actual contact, gravity, thread and tissue loads. Reset places it in an initial
bite; this is not a pickup controller or a complete dVRK hardware model.

Reported jaw forces and contact counts come from the final native rigid
microstep. Exported body poses are at that microstep's endpoint, so these
samples can differ in time by one native microstep. A brief missing contact
sample should be distinguished from a sustained dropped needle.

Each tool's shaft passes through a fixed trocar (remote center of motion).
Two hinges pivot the shaft, an axial slide provides insertion/retraction,
and a roll joint and two wrist joints orient the jaws. `GraspScene` uses fixed
port frames from the registered `tissue3_215300` recording: approximately 172 mm
apart, with shafts converging toward the workspace in the reference pose.
This registration is an estimate, not measured robot-to-phantom calibration.
Reset solves the initial grasp's joint coordinates against these ports, and
the ports remain fixed while physical time advances. The scene uses 180 mm
nominal trocar-to-roll distance and ±50 mm insertion travel. Shaft yaw/pitch
are limited to ±0.6 rad, roll to ±π, and wrist pitch/yaw to ±1.1 rad. Viewer
markers only display the trocar; the actual joint chain enforces the constraint.

The right tool, PSM1, is the Large Needle Driver holding the needle. PSM2/left
uses a photo-estimated DeBakey forceps visual. Each tool collides with the
capsule set of its own instrument, fitted to its drawn CAD
([method](method.md#the-collision-capsules-of-a-tool)). The scene starts with an
initialized grasp and the second tool parked; it does not perform the recorded
pickup/handoff or release/regrasp sequence.

For the bundled region 1, omitting `entry_xy` and `exit_xy` in a
`physical_grasp` configuration selects the original insertion/exit material
labels from `tissue3_215300/point_labels.json`. Their registered world positions
are transformed into native phantom coordinates before surface selection. This
keeps the stitch and fixed port layout in the same recorded workspace; the
earlier generic `x=0` stitch made PSM1 reach across the parked PSM2. The labels
inherit the recording's estimated camera/phantom alignment.

The initial source-stitch grasp is at 80% of the needle arc from its tip, with
a −75° rotation about the needle tangent followed by a −15° rotation about the
resulting jaw-closing X axis. These demonstration angles leave wrist travel
for the insertion; they do not reconstruct a measured recorded needle grasp.
The needle subsequently moves through physical jaw contacts. Explicit
`entry_xy`/`exit_xy` values are supported in native phantom metres. Other
stitches retain the transverse reset grasp (−90° tangent rotation, zero oblique
angle); custom asset/region and `free_needle` point defaults remain unchanged.

`GraspScene` explicitly uses 240 mm shaft capsule centrelines, shortened from
the source model's 547 mm at the proximal end. Standalone `add_rcm_grasper`
assemblies retain 120 mm shafts, 80 mm nominal insertion and ±30 mm travel;
these compact low-level defaults do not describe the registered scene layout.
Shaft radius, distal endpoint, wrists and jaws keep their source dimensions.
`GrasperConfig.shaft_length_m` supports custom assemblies, with shaft coverage
checked against the configured insertion range. Each tool has one set of
collision capsules, fitted to the CAD that is drawn on its bodies (18 per
tool: one for the shaft, one for the wrist, eight for each jaw). The exported
gripper parameters carry these body-local capsules, with the part and the body
of each, to the rigid model, the tool/tissue contact, the thread contacts and
the renderers; the viewer's "Show Collision" option draws them. Link mass, inertia, gains
and effort limits are demonstration values. Tool-link weight is assumed
gravity compensated; the independent needle retains its full weight. This is
not a calibrated model of a shortened physical instrument.

FEM forces couple to the needle and thread. The jaw/shaft capsules have native
rigid-body and thread contacts. In the scene (`GraspScene`,
`SutureSceneConfig`) they have no contact with the tissue unless it is asked
for: the tools pass through it and exchange no force with it, so a rendered
tool/tissue overlap is not a physical interaction. A contact of the tool
capsules with the tissue surface exists as an option of the scene
(`--tool-tissue-contact` among `SutureSceneConfig.example_arguments`), with
two laws. The first-version law is frictionless, acts one physics interval
late and carries light touches only. The law on the sweep of the capsules
(`--tool-tissue-contact swept`) places the tissue on the capsules before they
arrive and stops a tool on tissue that is held or cannot follow. With the
first the default procedure of `fem_needle_grasp` does not complete. With the
second it completes, and the example `fem_needle_grasp` runs that law by
default; it has rejected steps and, with whole frames, frames over 66.7 ms
that a run without the contact does not have ([below](#tooltissue-contact)).
There is no full-arm collision model.

The current mode retains four outer FEM/thread intervals per 60 Hz control
step and resolves four native rigid steps inside each interval. Earlier
[contact comparisons](validation.md#physical-grasp-contact-correction) used
the free-palm model and do not validate the new RCM chain. First inspect one
scene with actual commanded motion, grasp contacts and fixed-trocar diagnostics;
then repeat the scaling check for eight or more scenes. A quiet hold alone
does not validate insertion, nor does a one-scene run establish parallel scale.

The CUDA path batches native rigid-state transfers, reuses warmed FEM
workspaces/graphs across the existing tolerance levels, and captures trial
load/state preparation and accepted thread updates. Articulated native
MuJoCo remains on the CPU: partial force
graphs accelerate certified free-air, external subthreshold contact and
established-track trials, with the exact host path as a fallback. A per-face swept-bound and cubic Bernstein
certificate rules out clear tip trajectories; uncertain geometry keeps the
original CPU crossing checks, including float32 tip-position uncertainty.
The certificate does not authorize puncture or publish events.
Puncture-threshold proximity and possible reuse of an existing opening retain
the original host authorization. A conservative distance certificate permits
external contact when the tip is separated from every actual deformed
opening. In all material phases, the exact CUDA host path queries the tip
first; it prepares full shaft containment queries only when the device shaft
mask needs the exact CPU fallback. Custom query/mask methods retain the full
query batch. Rank-filtered Anderson proposals with adaptive regularization
can accelerate the unpunctured force solve once inner FEM accuracy is sufficient.
The following actual residual checks each proposal; insufficient reduction
recovers the saved Aitken guess. Every proposal still passes the unchanged
physical residual, motion and material acceptance checks within the existing
iteration budget. Gradient-enabled state/control arrays and Tape execution
retain the original iteration.
A dual-only ALM update reuses unchanged candidate
mechanics and surface forces while rechecking tunnel forces and convergence.
Supported unmodified solvers submit GPU FEM after native-state import so it
can overlap CPU MuJoCo integration; final exports remain stream-ordered.
A frozen articulated contact metric preconditions the numerical
interface correction; it does not replace the physical residual or relax its
tolerances. `GraspScene` also enables the surface-contact preconditioner.
It acts alone before the first accepted track and joins the shaft metric in
one solve when both contacts are active. Its positive friction approximation
reduces
repeated slip reversals during the numerical solve while retaining the actual
surface-force law. Empty or unsupported contact recordings use the original
Anderson/Aitken path. The public `NeedleTissueCoupler` and free-needle examples
keep this option disabled by default; custom native scenes can request
`surface_preconditioner=True`, with `surface_tangent_capacity=4096` facet pairs.
Capacity overflow disables the surface approximation for that trial and never
drops physical contact forces; an active shaft metric remains available.
`GraspScene` explicitly enables `preconditioned_anderson=True` for its native
shaft/contact iteration. Custom `NeedleTissueCoupler` callers retain the original
preconditioned iteration unless they opt in with the same constructor keyword.
Its secants and safeguards use raw physical force residuals, even though its
proposals use preconditioned directions. Existing precontact Anderson behavior
is unchanged. Gradient-enabled and unsupported paths retain their fallbacks.
The [method](method.md#needle-insertion-interaction)
describes the frozen-load assumption and unchanged acceptance checks.
Eligible CUDA Anderson iterations keep their histories on the GPU and transfer
only the exact active rows to the existing CPU SVD. Packet storage grows when
needed; no contact or residual component is dropped. Reports expose
`device_anderson_transport_calls`, `device_anderson_transport_bytes`,
`device_anderson_overflow_packets`, `device_anderson_max_active_rows` and
`device_anderson_packet_capacity`. CPU and differentiable paths remain unchanged.

Reports expose `surface_pc_applications`, `surface_pc_selection_changes`,
`surface_pc_graph_captures`, `surface_pc_graph_launches` and
`surface_pc_native_mass_fallbacks` as host counters; they do not add device
readbacks. Reports also expose native bridge, FEM graph, trial-preparation graph,
partial force-graph and thread-graph counts separately. The
`trial_prepare_graph_captures` / `trial_prepare_graph_launches` counters
identify load/state preparation; `host_force_primal_reuses` counts dual-only
updates that reuse the unchanged mechanics. Within one coupling transaction,
accepted rigid coordinates and controls are converted once; later trials
restore that same input and transfer only the changed needle wrench. The
`native_transfer_repeated_readbacks` counter is a subset of total native
readbacks: each still waits for its required GPU-produced wrench. Failed or
completed transactions invalidate cached input values. Throughput remains
measured from completed physical steps and elapsed simulation time, separately
from initialization, profiling overhead and drawing.
The procedural worker also reuses accepted interface loads from matching phases of completed control frames as
an initial guess. With the GPU rigid backend and its `device-anderson` policy the guess is the last
accepted load plus the change accepted over the same phase of the previous frame, offered only where
the last two such changes agree. `phase_predictor_attempts`, `phase_predictor_commits`,
`phase_predictor_invalidations` and `phase_predictor_rejections` are host counters; they do not count unchecked
physical steps. Device guards can retain the ordinary accepted-load guess,
and all existing convergence and acceptance checks still run. Resets, abrupt
commands, topology changes, retries and incomplete frames clear this history.
`coupling_graph_status` can
still report that
the complete conditional coupling graph is unsupported for this articulated
model. See the [method](method.md#physical-distal-grasp-and-worker-owned-thread)
for the execution and fallback contracts. Measure loaded insertion after
startup compilation, with the same mesh, substeps and solver budgets; viewer
drawing FPS and headless simulation throughput are different measurements.

During non-differentiable CUDA worker setup, the enabled shaft-preconditioner,
tunnel-ALM and phase-predictor modules are explicitly compiled/loaded with
`wp.load_module`. This moves their first-use cost into initialization without
running a hidden physics step. Keep initialization and complete-run wall times
when comparing benchmarks; relocating module loading is not a steady-state
solver speedup. CPU and Tape paths retain ordinary lazy loading.

Each worker owns its 90-element physical thread. `thread_positions` contains
samples of its accepted cubic centreline; `thread_diagnostics` reports the
attachment gap, actual loads, contact candidates and numerical budget. Actual
rigid-body poses, jaw coordinates, contact forces and needle-relative pose are
also reported. The renderer does not integrate these physical states. The
thread follows the needle through the tissue (`--no-thread-tracks` switches
that off): the track that the needle/tissue interaction records becomes
eyelets of the thread, one every 1.5 mm and one group per passage, and an
eyelet is threaded in the accepted step in which the needle's tail has gone
past it ([the pull](#the-way-to-the-tip-and-the-pull)). The eyelets live in a
reserve that is fixed at start-up, 36 eyelets in 4 tracks for the default
stitch; `thread_diagnostics` reports the eyelets made (`eyelets`,
`eyelet_passages`), threaded (`eyelets_threaded`) and refused for want of room
(`eyelets_refused`), and the thread inside the tissue
(`thread_inside_tissue_m`). The thread can be tied to the tissue at an end of
a passage, a stand-in for a knot (`GraspScene.tie_thread`, asked for with the
optional `commands` of the pool's `step`); every observation has the held
eyelets and the thread's pull (`thread_tension`) and the width of a wound
that the planned stitch crosses (`wound_widths`)
([the tie and the tightening](#the-tie-and-the-tightening)), and the scene's
geometry has the weight that it applies to the needle (`needle_weight_N`),
below which a thread's pull on the needle counts as slack
([the lock and the slackening](#the-lock-and-the-slackening)). No eyelet is
planned and no passage is learned, so
this mode's validation is separate from the completed scripted stitches in
`fem_needle_suture`.

Run the focused physical checks with:

```bash
PYTHONPATH=tests NEWTON_TEST_DEVICE=cuda:0 uv run --locked python -m unittest \
  test_rcm_grasp test_pose_ik test_grasp_scene test_thread_rigid_contact test_worker_thread -v
```

The [method](method.md#physical-distal-grasp-and-worker-owned-thread) documents
lagged force exchange, compliant contact and uncalibrated parameters;
[validation](validation.md#physical-grasp-and-worker-thread-2026-11-12) distinguishes
unit contact checks, short holds and any later multi-world runs. The separate
IsaacLabSuturebot integration now chooses this physical mode by default; its
seven-action policies must be trained and evaluated with the articulated
configuration. Procedural or manual actuator commands require no training.
Existing six-action free-needle results and free-palm scaling reports do not
validate this model.


## Standalone articulated needle grasp

Run the physical tools, needle, tissue and thread directly in Newton:

```bash
./docker/run.sh fem_needle_grasp
# Or, after uv sync --locked --extra examples:
uv run --locked python -m newton_solver_suturebot.examples fem_needle_grasp
```

Two defaults of this example are switched on since 2026-11-17: the tools push
the tissue where they touch it ([tool/tissue contact](#tooltissue-contact),
the law on the sweep of the capsules; `--no-tool-tissue-contact` switches it
off), and with one scene in the window a step plans for a picture no later
than 66.7 ms after the one before
([frame time budget](#frame-time-budget-default); `--frame-budget 0` runs
whole frames).

What in this example is a choice that runs selected, and not a value that
follows from the model, is named in one place, the head of
`examples/grasp/example_fem_needle_grasp.py`: the coupling tolerance and the
trial budget, the two surface-event arguments it hands to its scene, and the
response columns per step; each with the section here that has its runs. The
options that selected a way the example no longer uses were taken out on
2026-11-19: `--tool-tissue-contact first` (and the option without a value),
`--event-search`, `--event-oracle` and
`--frame-budget-unit`. A default run is the same as before: the example hands
its scene the same arguments. The scene keeps its own arguments of those
names for callers of `SutureSceneConfig.example_arguments`.

No Isaac Lab, Gym, RL policy, training checkpoint or external dataset is required.
The two dVRK tools use the bundled source trocar layout. The active needle holder
pivots and slides through its fixed trocar; bounded IK commands its finite joint
actuators and articulated wrist. Friction between the jaws holds the independently
free rigid needle. The other tool stays parked. The procedural controller settles
the grasp and then runs one sequence:

1. It drives the needle along the entry/exit arc and on past the planned end
   until the needle is centred in the tissue, with both ends outside. The jaws
   stop 3 mm of arc short of the entry point.
2. Once the arc is complete and the needle has entered the tissue, it lifts the
   tool 10 mm along the tissue entry normal, holds for 0.5 s and lowers it.
3. It opens the jaws. The tissue holds the needle.
4. It backs the open jaws away along the axis of one of them, rises along its
   shaft and, with the needle at rest, plans where to take the needle by its
   tip ([below](#the-grasp-at-the-needle-tip)). It goes straight over to a
   point above that grasp while it turns, comes down on the needle along the
   jaws and closes ([the way](#the-way-to-the-tip-and-the-pull)).
5. It pulls the needle on along its curve: the tool turns the needle about the
   needle's own centre of curvature, the tail goes into its entry, through
   the tissue and out of its exit, the thread follows it through the track,
   and the tool holds the needle clear of the tissue
   ([the pull](#the-way-to-the-tip-and-the-pull)). With `--no-thread-tracks`
   the example ends after the lift: the pull needs a thread that follows the
   needle's track.
6. After the pull through the tissue it ties the thread where it goes into
   the tissue (a stand-in for a knot), moves the needle on along its chord
   until the thread is as tight as the tissue allows, and holds: the wound
   narrows at its top while the tool holds
   ([the tie and the tightening](#the-tie-and-the-tightening)).
7. It ties the thread where it comes out of the tissue as well, takes the
   needle back along the same line until the thread to the needle is slack,
   and holds: the thread between the two ties keeps a part of the narrowing
   without the tool's pull
   ([the lock and the slackening](#the-lock-and-the-slackening)).

Every motion uses the same bounded joint IK and finite actuators. Nothing welds
or directly repositions the needle. The physical thread collides with tissue and
with the tool jaw/shaft capsules. The tools themselves push the tissue where
they touch it: the [tool/tissue contact](#tooltissue-contact) on the sweep of
the capsules is on by default (`--no-tool-tissue-contact` lets them pass
through the tissue). The entire selected wound region remains active.

The recorded entry and exit labels are 5.4 mm apart. A needle of 9.45 mm radius
through them passes only 0.39 mm under the surface and does not stay in the
tissue when the jaws open. The default therefore moves the two points apart
along their own line to 10 mm (`--bite-length 0.010`), which gives a bite 1.4 mm
deep with about 9.6 mm of needle in the tissue. The pad surface has a fold
there, so the tip leaves the tissue and enters it again in the middle of the
bite. `--bite-length 0` restores the recorded stitch. The grasp at reset and the
grasp at the tip are orientations that a scan of the tool's inverse kinematics
found reachable from its port for this layout; they are not general.

Which numbers of the procedure are derived and which are choices is written in
one place: the table at the top of
[_grasp_procedure.py](../src/newton_solver_suturebot/examples/grasp/_grasp_procedure.py)
names every number that is not derived, with what chose it, and
`tests/test_grasp_lift.py` holds each name to its value and its row.

- Measured: the two grasps at the tip (twist, bite and whether the jaws are
  exchanged, from scans of the tool's joints for this port layout and stitch;
  the code holds the angles in radians rounded to four digits, -1.0472 and
  0.5236 for -60 and 30 degrees) and the number of jaw capsules that one
  controller call of the plan samples (one; it changes the number of calls,
  not the plan); the furthest the tool moves the needle on to tighten the tied
  thread (7.5 mm, the travel at which the tissue under the tie collapsed when
  it was pulled on without a stop, rounded down).
- Set by hand (no run on record compared them with another value): the arc
  speed (3 mm/s) and its ramp (0.5 s), the least rise time of a lift (0.75 s)
  and its hold (0.5 s), the part of the action rate limits that a lift or the
  pull may use (80 %), the times after which the procedure stops (0.25 s
  without both jaws on the needle or with the pull at its bound, 2 s without
  a first grasp, 3 s for the jaws to close on the tip or the needle to come
  to rest), the 3 mm of arc kept before the entry, the 6 mm from the palm
  origin at which the jaws hold the needle, the 12 mm of the descent and the
  0.6 rad opening of the regrasp, the resolution of the clearances (a
  quarter radius, four times finer where it decides; stations half a wire
  radius apart on the wire; six halvings at the end of a search), the
  one thread element of slack beyond the end of the pull and the length of
  the draw of the thread (one length of the needle).
- Assumed, not measured: the creep of the needle between the plan of the
  grasp and the closed jaws, twice the rest tolerance of the plan (it is part
  of the reach margin below).
- Derived where they are used: the time of a lift, the angle to which the arc
  is driven on, the arc speed, the end and the bound of the pull, the reach
  margin of the joints, the rest tolerance of the needle, the turning radius
  of the tool (from the capsules and the wrist length of the tool model),
  every length of the way from the one grasp to the other, the direction,
  the speed and the stop of the tightening, and the stop and the bound of the
  slackening (its threshold is the weight that the scene applies to the
  needle).

#### The way to the tip and the pull

From the pose at which the jaws let go to the grasp at the tip, the commanded
way keeps every capsule of the tool (shaft, wrist, the open jaws) clear of the
tissue at rest and of the needle wire, by the clearance of the plan of the
grasp (`ProceduralGraspController._transfer` in
[_grasp_procedure.py](../src/newton_solver_suturebot/examples/grasp/_grasp_procedure.py)):
back along the axis of one open jaw until the tool can rise (5.7 mm), up along
the shaft until tissue and wire are further from the way of its palm than the
tool's turning radius of 13.5 mm (12.6 to 16 mm of rise), straight over to a
point above the grasp while the tool turns, and down along the jaws (12 mm
twice: to the pregrasp, where the needle is compared with the pose the grasp
was planned for, and on to the grasp). Only the first part of the way back
starts inside the tissue: the jaws let go where the arc ended, the lower one
commanded 3.2 mm through the rest surface. Measured against the rest surface
with a signed distance of its own, for the needle poses of 17 device runs:
at least 5.9 mm clear on the way over, 0.6 mm at the end of the way down,
0.05 mm when the jaws have closed (the grasp point is the nearest one to the
tissue at which they stay out of it).

A limit: after the pregrasp the needle is not looked at again. Through the
descent and the closing (1.3 s and 0.5 s in the runs) the jaws go to the grasp
as it was planned, and a needle that moves in that time is closed on where the
plan had it (in a replay without physics a needle moved by 0.5 mm there went
unnoticed). The reach margin below allows for the creep of a resting needle
only. Once the jaws have closed, the pull is tested again from the grasp as
they measure it.

The tool then pulls the needle on along its curve
(`ProceduralGraspController._pull_target`): it turns the needle about the
needle's own centre of curvature, in the needle's own plane, in the sense of
the insertion, so the tip rises out of its exit and the tail moves towards its
entry. The speed is the arc speed of the insertion (`--speed`, 3 mm/s, with the
palm within 80 % of the action rate limits), and the pull starts and stops with
the insertion's ramp: it stops with that ramp run backwards over the arc the
tail has left. The tail goes into its entry, through both lips of the wound
and out of its exit. The pull ends when both hold: the tissue holds no part of
the needle's track any more (the needle/tissue interaction lets a track sample
go when the tail has passed it; `engaged_track_length_m` is then zero), and
the tail is one thread element, 2 mm, of arc past the exit marker, so that the
element of thread that hangs on the tail is outside the tissue. While a part
of the track is still held the tail is inside, and at least that clearance is
left to go. The commanded angle is bounded by the arc that takes the tail two
elements past the exit marker from where the needle rests; a needle that is
not free of the tissue, or has not followed, 0.25 s after the command has
reached that bound stops the procedure with that reason. The tool then holds
for 0.5 s: the needle is held by its tip, clear of the tissue, with the thread
running from its tail into the exit.
[The tie and the tightening](#the-tie-and-the-tightening) and
[the lock and the slackening](#the-lock-and-the-slackening) follow.

The thread follows through the track that the needle has actually cut
(`WorkerThread`, `NeedleTrackEyelets`, `ThreadTissueCoupling.follow_tail`).
While the needle goes in, the track rows that the needle/tissue interaction
records (one every 0.25 mm) become eyelets of the thread: the first row of a
passage, then every row 1.5 mm of rest length from the last eyelet, and at
the exit of the tip the last row unless an eyelet is within 0.75 mm of it.
In the pull an eyelet is threaded in the accepted step in which the tail has
gone past it: when it lies behind the tail by at most one eyelet spacing along
the direction in which the thread leaves the tail, and at most two wire radii
(0.56 mm) beside that line. From then on the thread slides through it against
the thread's friction, and the thread between the eyelets of a passage has no
contact with the tissue surface. The default stitch makes 4 eyelets in each of
its two passages. A scene run with `--no-thread-tracks` has no eyelets, and
the example does not pull in it: a thread without a track is pressed back out
through the surface behind a tail that goes in.

A run with a window shows how far the thread is through the tissue: the side
panel has one line per scene, for example
`Thread: 5 of 8 eyelets threaded (passage 1: 4 of 4, passage 2: 1 of 4), 4.6 mm inside the tissue`:
the eyelets that the thread goes through of those that the needle's track has
made, the same for every passage, and the rest length of thread inside the
tissue. All of it comes from `thread_diagnostics` of the observation that a
drawn frame has anyway: no array is read from the device for it. That report
counts the threaded eyelets and does not say which they are. The line counts
them to the passages in the order in which the needle made the eyelets, which
is the order in which the tail passes them; an eyelet that the tail went past
without threading it would be counted to the wrong passage, while the count
of all stays right.

For the pull the jaws take the tip with another grasp than the lift by the tip
did (a procedure that has been removed): twist -60 degrees about the needle tangent, bite 30 degrees, and the tool
turned half a turn about its jaw axis (`needle_to_gripper_pose(...,
jaws_exchanged=True)`: the same grasp with the two jaws exchanged, which a
fixed-trocar tool reaches with its roll joint half a turn on). The earlier
grasp (twist -150 degrees, bite 45 degrees) cannot pull: its roll joint stands
0.002 to 0.03 rad from its limit and the pull turns it further. These are
measured choices for this port layout and stitch, bounded on a grid of 15
degrees by the reach test below: no point of the needle is reached at a twist
of -90 degrees, none or no way down with a bite of 15 degrees at twists of
-60, -30 and -15 degrees; over the arc that takes the needle out of the tissue
the wrist joints stay 0.39 rad inside their limits with this grasp, 0.33 rad
with a bite of 45 degrees and 0.20 to 0.23 rad with twists of -75 and -45
degrees. Without the exchange of the jaws the roll joint would have to pass
its limit on the way over, and the procedure stops there with that reason.

Reach is a condition of the plan. A point of the needle is taken only if the
tool reaches the descent to it, the closing and the whole arc that takes the
needle out of the tissue from it (the tail to two thread elements, 4 mm, past
the exit marker: the bound of the pull) with every joint inside its limits by a margin; the
way back from the needle is tested when the jaws have opened (all 12 mm of
it; the tool leaves it as soon as it can rise), each step of the rise before
it is commanded (the rise that is open to the tool ends at the limit of its
slide, where no margin is left), the way down to the pregrasp
and the way over when they are known, and the arc of the pull once
more from the grasp as the jaws have measured it. The joints of a tool pose
are those of the fixed-trocar tool in closed form (`arm_joint_positions`; they
differ from those of the scene's bounded pose IK by at most 0.0008 rad). The
margin is 0.024 rad for the rotary joints and 0.31 mm for the slide: the
tolerance of the pose IK (0.001 rad, 0.01 mm, read from what the scene
publishes for the tool) and the creep of the needle
between plan and closing (twice the rest tolerance of the plan, 0.04 mm, at
each end of the wire), taken through the wrist (`_read_joints`). If no point
of the wire is both clear and reached the procedure stops with "No point of
the needle tip is both clear of the tissue for the jaws and within the reach
of the tool" and the joint; a run does not end in an unreachable tool pose.

Measured on the device (one DGX Spark, MuJoCo Warp backend, null viewer,
2026-11-17) with the pull through the tissue, in eleven self-tests
(seven without the tool/tissue contact, today `--no-tool-tissue-contact`, four
with `--tool-tissue-contact swept`, today the default; three runs shared the GPU at a time, so none is
a timing run), with records per frame: every run completes the procedure (done at 30.6 to 30.7 s of simulated
time without the tool/tissue contact and 30.6 to 30.8 s with it), with 2
entries and 2 exits as before and none during the pull, no CUDA graph captured
and no kernel module loaded after start-up while the 8 eyelets appear, no jaw
off the needle while it is held or pulled (smallest jaw force 0.67 N),
no rejected tissue step and no rejected coupled attempt in the pull. The pull
turns the needle by 111.4 to 112.7 degrees (18.36 to 18.58 mm of arc) without
the contact and 113.7 to 114.9 degrees (18.76 to 18.94 mm) with it, in
385 to 397 frames. The tissue lets go of the last track sample at
101.9 to 102.2 and 104.3 to 105.2 degrees, with the tail level with
the exit marker; the pull ends with the tail 2.00 to 2.05 mm past
that marker, and the needle is then held with its tail 1.81 to
2.01 mm above the deformed tissue. All 8 eyelets (4 and 4) are
threaded, in the order of the track, none is refused; 7.1 to 8.6 mm of thread
are inside the deformed tissue in two stretches, one in each lip. A coupled
step of the pull takes 1.71 to 1.83 trials in the mean (7 at most; in
a run with records per physics interval and the contact: 1.79 per attempt,
7 at most, over 1576 attempts, no node held and no force stored by the tool,
the tool 11.1 mm clear of the deformed tissue in the hold). The tissue holds
the needle with 24.9 to 26.1 mN at most without the contact (in the first
frames of the pull: the load that the closing of the jaws left, 23 mN) and
15.3 to 17.2 mN with it (at 82 degrees, the tail in the second lip); the thread
pulls on the needle with 12 to 19 mN at most and on a tissue node
with 5 to 9 mN (the scale of the tissue's own give, E h^2, is 43 mN). In
9 of the 11 runs the thread at the end lies within 0.14 mm of
the needle's track, passes within 0.06 mm of every eyelet and pulls on
the needle with 0.3 to 1.5 mN. In two of the seven runs without the
contact the thread stayed taut after the pull, stretched by 3 to 7 % between the needle
and the edge of the pad: 4.2 to 7.0 mN on the needle in the hold, the thread
0.20 to 0.27 mm off its first eyelet and up to 1.4 mm off the track where it
leaves the entry; those runs completed as the others.

The spare thread. In those runs the spare thread lay along the nearest side of
the pad's bounding box: 15.4 mm above the plate, which is 2.3 to 3.9 mm inside
the plate's edge (the plate is turned by 0.8 degrees in its box), in a 5 mm
turn, with its lead round the exit side. Three more such runs with the thread
and its contact planes read in every frame (two of them taut) show what
followed: the thread fell and its turn sprang open, 11 to 16 nodes lay beyond
the plate's edge from the fifth frame on, and when the pull began the lead,
which the insertion had folded back on itself at the tail, swung over and
pushed a loop of 60 to 98 mm over the edge. Drawn back, the first node beyond
the edge was held by the contact plane of the plate's side face with the
whole pull (4.3 to 5.1 mN): the thread's contact acts at its nodes only, each
on the plane of the face of its nearest surface point, so the element to the
node on the plate passes through the plate's corner and the side face stops
the node. That is unchanged and is an expected failure in
`tests/test_thread_tissue.py` (4.2 mN on a thread whose hanging part weighs
0.0095 mN).

`loose_thread_coordinates` now lays the spare thread on the pad, behind the
stitch (how often it has stayed there, why not always, and what was changed
on 2026-11-19: the end of this paragraph). Every node lies a
thread radius above the pad's surface under it. The
thread comes down two radii of the needle's circle (18.9 mm) straight behind
the entry, on the line from the exit to the entry: that is the way it leaves
its entry hole in the pull, so it is drawn along its own length, and no point
of the needle on its circle through the entry comes that far. From there it
takes one elasto-gravity length of the thread, `(E I / (rho A g))^(1/3)` =
10.0 mm, towards the nearest side of the pad's border that is not before that
point, turns behind the stitch again and runs straight on to the border,
which it follows away from the stitch, that length inside the pad's box and
round its corners with that radius: a thread on the pad holds a bend of that
radius and is bent aside by a push over about that length. (Until 2026-11-19
it went from that point straight on to the nearest side; the runs of
2026-11-17 below are on that layout.) The lead from the
tail is the one of three smooth curves that stays a thread radius clear of
the needle's and the tools' capsules and bends the thread least. Three
self-tests without the contact on that layout (2026-11-17, 19:25 to 19:28,
`--device cuda:0 --viewer null --test --no-tool-tissue-contact`, the three
together on the GPU, the thread read in every frame, so not timing runs) pass
with 2 entries, 2 exits and 8 of 8 eyelets threaded, done at 30.7 to 30.8 s:
no node is beyond the plate's edge at any frame (the nearest comes within
4.2 mm of it, the lowest lies on the plate), no element is stretched at the
end (1.0001), the thread passes within 0.002 mm of the entry's eyelet and
0.03 mm of every eyelet and lies within 0.11 mm of the track. On the needle it
pulls with 11.8 to 11.9 mN at most, near the end of the pull as before (13.3
to 13.9 mN in the three runs above), and in
the hold it pushes with 2.5 mN falling to 1.5 mN: the stop leaves its first
element 2 to 3 % short (a relaxed run of before: 1.5 falling to 0.5 mN). What
is new to see: the lead hangs from the tail on the exit side at reset and
comes down in the first half second, and because the tail's tangent turns
round in the insertion the thread ends in a loop of about 7 mm beside the
entry, from which the pull draws its thread.

With the tool/tissue contact, the example's default, the layout was run in
the round that brought it in (2026-11-17, 20:38 to 20:42 and 21:05 to 21:09:
each time two self-tests with no option and one with
`--no-tool-tissue-contact`, the three together on the GPU, the thread read
in every frame; `h1_def_1`, `h1_def_2`, `h1_noc_1`, `h3_def_1`, `h3_def_2`,
`h3_noc_1` under `/tmp/suturebot-gpu-work/grasp_fix/integ/gpu/`, a work
folder outside this repository). All six pass with 2 entries, 2 exits and 8
of 8 eyelets threaded, done at 30.7 to 30.8 s, and in none is the thread
taut after the pull: no element is stretched at the end (1.0005 at most),
the thread is within 0.004 mm of the entry's eyelet, and it pushes on the
needle with 2.1 to 2.7 mN falling to 0.9 to 1.8 mN. **With the contact the
thread does not always stay on the pad.** In one of the four default runs
(`h1_def_1`) the
lead came down at the end of the insertion as an open arc west and north of
the tail, not as the loop beside the entry, and pushed the row at the border
to 1.9 mm from the plate's edge (4.3 mm in the other five runs); in the pull
its nodes crossed the edge from frame 1528 on, and at the end 44 mm of thread
hang over it, the lowest node at a height of -19.9 mm where the plate's top
is at -4.2 to -5.0 mm (near the end of the pull the thread was a straight
line from the entry to the edge, with 9 mN on the needle; it relaxed after
the stop like the others). In the other three
default runs and in the two runs without the contact no node is beyond the
edge at any frame.

Why, and the change of 2026-11-19. Three more default self-tests on that
layout (15:21 to 15:25, the three together on the GPU, the thread and the
tools' capsule forces on it read in every frame; `h1b_base_1` to `_3` under
`/tmp/suturebot-gpu-work/grasp_small/thread/out/`, a work folder outside
this repository): two stayed on the pad, in the third the lead again came
down as the open arc and 46 mm of thread hung over the edge at the end
(lowest node -20.7 mm). That is 2 of 7 default runs. The records rule out
the tools and the tissue: up to frame 417 the rigid bodies of the three runs
are equal to under 1 µm, up to frame 376 the tissue nodes too, and in the
run that failed no tool capsule touched the thread at any frame. The thread
alone differs, from the first frame, and by 0.2 to 1.3 mm from frame 10 on.
What happened in it: the lead did not come down at reset. From the sixth
frame to frame 350 it stood on the pad as an arch (nodes 14 to 20 about 8 mm
above the plate), its foot held in its direction by the leg that ran on from
it to the border. It came down when the tail's tangent had turned through
the vertical (frames 350 to 420). In the runs that made the loop the turning
tail lifted the arch once more, to 16 mm at frames 353 to 367, and carried
it over the top; in the run that failed the arch was not lifted, it sank
from frame 250 on and was laid down as the open arc, which the pull then
unbent against the leg and so pushed the row at the border over the edge.
Which of the two is decided by where the arch's foot rests, to about a
millimetre, and that is rounding on the device grown in the first twelve
frames. The
same fork appears on the CPU with the tool/tissue contact absent: the
scene's own thread alone under the recorded body poses, the tissue at rest,
its start moved by noise of 1 µm (13 replays) or 0.1 mm (8): 2 of the 21 end
with 26 and 27 nodes over the edge. So the contact is not the cause, and the
five runs without it that stayed on the pad do not show otherwise (a chance
of 0.19 at 2 in 7).

`border_walk` now cuts the leg to one turn radius and turns the thread
behind the stitch from there (the line reaches the border 27 mm further
north; the lead at reset is the same to 0.14 mm). Two other rules were
tried on the replay and dropped: a turn radius from the thread's friction
instead of its weight, `(E I / (mu rho A g))^(1/3)` = 17.1 mm, and a walk
straight behind the stitch with no leg; both put the lead where the tools
work, and a tool lifts it and throws it over the edge (2 of 2 replays
each). With the short
leg the foot of the lead gives way: the lead lies down within 13 frames of
reset, in the same shape every time, and the loop forms from it as the tail
turns. Measured: 58 of 58 replays end on the pad, 44 with noise of 1 µm, 8
with 0.1 mm and 6 under the body poses of five other recorded runs (the loop
encloses 126 to 135 mm² at the end of the insertion in every one; the old
layout 103 to 150 mm², and the two open arcs), and three default
self-tests (15:56 to 16:00, as above; `h2_tree_1` to `_3`) pass with 2
entries, 2 exits and 8 of 8 eyelets threaded, done at frames 1839 to 1857,
no capture and no module load after start-up: no node is beyond the plate's
edge at any frame (the nearest 4.4, 5.2 and 4.5 mm inside, the lowest on the
plate), the three threads differ by 4.6 mm at most in the whole run and by
2.6 mm from frame 30 to frame 500 (before: 23 mm by frame 451 and 29 mm in
the pull), no element is stretched at the end (1.0007 at most),
the thread is within 0.002 mm of the entry's eyelet and 0.033 mm of every
eyelet, and it loads the needle with 11.4 to 12.4 mN at most in the pull
(11.4 to 11.8 mN in the three runs of the same day before the change) and
0.5 to 1.8 mN at the last frame. New with it: the lead now lies under the
first tool's jaw when the lift begins, and the jaw presses on elements 16 to
19 with 1.7 to 2.3 mN at frames 435 to 440 (0.2 mN at most before); the
thread stays where it is. Unchanged: in the first 30 frames the row at the
border slides towards the edge where it lies on the foot of a mound, from
7.7 mm inside to 4.7 to 5.4 mm, with either layout; and a thread that does
get over an edge is still held there (above). Noise of 1 mm, which
stretches the elements, throws thread over the edge in the first 20 frames
with either layout (it ends there in 2 of 8 replays, and in 1 of 8 with the
old one). Three runs and 58 replays are no rate either.

Since the same day the swept tool/tissue contact is the example's default
([below](#tooltissue-contact)). Two self-tests with no option for it
(`--device cuda:0 --viewer null --test`, the two together on the GPU, 17:21 to
17:24, on the tree of that change) complete the same way: the swept law in the
scene, 2 entries and 2 exits, all 8 eyelets threaded and none refused, the
pull 114.2 and 115.4 degrees (18.83 and 19.03 mm) in 394 and 399 frames, done
at 30.4 and 30.8 s, the tail 2.00 mm past the exit marker when the pull ends
and 1.97 and 1.88 mm above the deformed tissue, the thread at the end within
0.07 mm of the track and of every eyelet and 0.7 and 1.0 mN on the needle, no
jaw off the needle (0.66 N at least), nothing captured or loaded after
start-up.

Before the thread tracks, and until its removal with `--no-thread-tracks`, the
pull ended at the entry (one self-test: it
turned the needle by 30.5 degrees, 5.04 mm of arc, in 125 frames and ended with
the tail 0.55 mm before the entry marker; done at 26.3 s; no eyelet). That
pull was measured in nine runs (three self-tests without the tool/tissue
contact, three with `--tool-tissue-contact swept`, and three more runs with
that contact and records per physics interval), from which the rest of this
paragraph is:
every run completes the procedure (done at 26.3 to 27.1 s of simulated time),
with 2 entries and 2 exits, no CUDA graph captured and no kernel module loaded
after start-up, and no jaw off the needle while it is held or pulled. The
grasp point is 2.38 mm behind the tip without the tool/tissue
contact and 1.81 to 2.13 mm with it (the needle rests lower then). The
pull turns the needle by 30.5 degrees
(5.03 to 5.04 mm of arc) without the contact and 32.6 to 33.7 degrees
(5.37 to 5.57 mm) with it, in 125 to 136 frames; the jaws hold it with
0.57 N at least; the tissue pulls on the needle with
12 mN at most; the needle shifts 0.32 to 0.38 mm in the jaws as
the load comes on; a coupled step of the pull takes 1.7 to 1.9 trials in the
mean. On the way from the end of the way back to the closing, with the
contact: no tissue node held, no force stored by the tool, no rejected step
(three runs with records per physics interval). The joint
targets stay at least 0.145 rad from a limit on the way over
(the wrist yaw where the tool stands after the arc), 0.30 rad
in the descent and 0.36 rad in the closing and the pull; the
slide 38.6 mm. The controller's calls of the regrasp take at
most 5.9 ms.

#### The draw of the thread

After the pull the needle is out of the tissue and its tail stands one thread
element past the exit: the thread has only just followed through the track.
Before the stitch the tool therefore draws the thread through
(`ProceduralGraspController._begin_draw`, `_draw_target`): the palm moves
straight, without turning, along the chord of the needle from its tail to the
point between the jaws, the line on which the tightening goes on afterwards,
by one length of the needle (`draw_needle_lengths`; the chord from its tail to
its tip, 17.53 mm). In the runs below that is upwards at 47 to 49 degrees,
back over the wound. Speed and ramp are those of the insertion, 3 mm/s after
0.5 s, and the stop is the same ramp run backwards: 6.3 s. The needle then
hangs clear of the tissue by its own length, as much of the loose thread has
run through both lips behind it, and the tool holds for 0.5 s (stage
`drawn`) before the tie. Before the tool moves, the whole line to the end of
the draw and on to the travel bound of the tightening is asked for reach
(25 mm; from where the pull of a recorded run ended the tool reaches 88 mm
along that line before a joint comes to its limit).

The thread is not tied yet and its far end is loose, so it slides through the
eyelets of its track and takes no load. A thread that does not slide pulls the
tissue along instead: the draw stops the procedure with the reason when the
least tissue volume reaches the tightness value of the stitch.

The mouths of the track. Where the track meets the surface the needle has left
a hole, and a thread node over that hole has no surface under it. Until this
change it had the contact plane of the face around the hole all the same
(`ThreadTissueCoupling`: every node that is not between the threaded eyelets
of a track takes the nearest surface point as its plane), so a node that was
drawn into a track was stopped on the surface over its mouth until the pull
pressed it through the contact penalty, and a node that had just left a track
was thrown out by it. The coupling now leaves a node without a plane where its
nearest surface point lies over the eyelet at an end of a threaded track,
within two wire radii of it across the surface normal, the radius within
which the needle/tissue interaction takes a point for the needle's track, and
with the eyelet at most one eyelet spacing under it
(`ThreadTissueCoupling.open_mouths`, called by `WorkerThread`;
[method](method.md)). Measured on the device, the same draw without and with
the open mouths:

| | mouths closed (before) | mouths open |
|---|---|---|
| Thread's force on the needle during the draw | sawtooth, 0.11 to 12.2 mN | 0.13 to 0.40 mN |
| Least tissue volume during the draw | 0.79 to 1.00 | 0.992 to 0.998 |
| Wound at its top during the draw | 2.50 to 2.88 mm, with the sawtooth | 2.87 to 2.89 mm |

The stop of the tightening. With the mouths closed the tightening of the tied
thread seemed to reach the rule of the thread/tissue coupling, a least tissue
volume of 0.5, after 3.4 mm of travel. That was the stopped nodes pressing on
the tissue at the mouths. With the mouths open, pulled on without a stop (one
device run with the rule at 0.5 and the travel bound widened to 25 mm), the
least volume falls by 0.05 to 0.07 per millimetre to 0.637 at 7.65 mm of
travel and 24.6 mN on the needle, and then collapses to 0.306 within the next
0.44 mm, while the thread's force falls from 24.95 to 21.7 mN: the tetrahedra under
the first tie, which sits in the first row of the track at the surface and is
pulled into the tissue, give way (five tetrahedra under 0.7, the least one
shares a node with the tied eyelet). The rule value 0.5 lies inside that
collapse, so no tightening comes to rest at it. The grasp scene therefore
publishes its own value for its stitch, `tight_volume_ratio` = 0.75
(`STITCH_TIGHT_VOLUME_RATIO` in `_grasp_scene.py`), which the controller takes
in place of the coupling's 0.5 (a scene may ask for an earlier stop, never for
a later one): there the thread pulls with 15.5 mN, 63 % of the pull at the
collapse. The travel bound of the tightening is now 7.5 mm, the travel at
which the tissue collapsed, rounded down.

Three device runs of the self-test with these settings (2026-11-20, whole
frames, null viewer; all pass):

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Draw begins / procedure done [s] | 30.9 / 42.48 | 30.8 / 42.43 | 31.0 / 42.55 |
| Thread's largest force on the needle in the draw [mN] | 0.40 | 0.39 | 0.40 |
| Travel of the tightening [mm] (bound 7.5) | 5.10 | 5.20 | 5.10 |
| Thread's force on the needle at the end of its hold [mN] | 15.3 | 15.6 | 15.5 |
| Least tissue volume: tightening begins / its hold ends | 0.994 / 0.780 | 0.994 / 0.779 | 0.993 / 0.778 |
| Fall that the rule predicted for its way to rest | 0.067 | 0.069 | 0.070 |
| Wound at its top: begins / held tight / kept [mm] | 2.874 / 2.302 / 2.579 | 2.873 / 2.286 / 2.573 | 2.872 / 2.296 / 2.576 |
| Wound at its track: begins / held tight / kept [mm] | 1.746 / 1.496 / 1.548 | 1.746 / 1.490 / 1.545 | 1.746 / 1.494 / 1.546 |
| Travel back of the slackening [mm] | 3.16 | 3.24 | 3.18 |
| Thread's force on the needle at the end [mN] (needle's weight 2.94) | 0.51 | 0.51 | 0.53 |
| Needle in the jaws since the tie at the exit [mm] | 0.08 | 0.12 | 0.07 |

The fall of the least volume is not even. In every run with the mouths open
it drops once by 0.02 to 0.03 within a frame or two, at 4.3 to 4.5 mm of
travel, where it otherwise falls by 0.002 per frame; the thread's force
changes by less than 0.7 mN there. The cause is not established (a thread
node that passes a mouth under load is a candidate). In the three runs the
rule began its stop in the call that first saw that drop, with a slope of -71
to -74 per metre where the slope before it was -47 per metre: the travel of
5.1 to 5.2 mm, the least volume of 0.78 and the 15.5 mN are the numbers of
that trigger. Without the drop the curve of the run without a stop puts the
stop near 5.75 mm, at 0.75 and 17.0 mN (read from that curve, not run). Both
are inside the travel bound and before the collapse.

The stitch keeps the wound 0.30 mm narrower at its top and 0.20 mm at its
track than when the tightening began. Before this change it kept 0.39 to
0.40 mm and 0.22 mm, with the tissue at the mouths pressed in by the stopped
nodes.

The two sections below describe the tie, the tightening, the lock and the
slackening as they were measured before the draw and the open mouths. Their
mechanism is unchanged. These numbers of theirs no longer hold and are
replaced by the table above: the rule value 0.5 (now 0.75 for this scene), the
travel of the tightening (3.4 to 5.6 mm; now 5.1 to 5.2 mm), its bound
(6.75 mm; now 7.5 mm), the forces of the thread (17 to 26 mN on the needle
while the tool holds; now 15.5 mN), the least volumes, the widths of the
wound, the times (done at 35.2 to 35.6 s; now 42.4 to 42.6 s) and the curves
measured on the CPU with the mouths closed.

#### The tie and the tightening

After the pull and the draw ([above](#the-draw-of-the-thread)) the thread lies
through both lips of the wound and its far end
lies loose on the pad: pulled on, it slides through, so it takes no load and
the wound stays as open as it was (in the reference shape 2.88 mm at its top
and 1.75 mm between its walls at the depth of the track). With thread tracks
the procedure therefore goes on with the first step of a stitch
(`ProceduralGraspController._stitch_target`): it ties the thread where it goes
into the tissue, moves the needle on until the thread is as tight as the
tissue allows, and holds; then it ties the thread at its exit and lets it go
slack to the needle ([below](#the-lock-and-the-slackening)). No option
switches this off by itself: without thread tracks (`--no-thread-tracks`) and
without the lift (`--lift-height 0`) the example does not pull at all.

The tie. In the first call after the hold that ends the pull, the controller
asks the scene to tie the thread at the first eyelet of its first passage,
the hole through which the thread goes into the tissue. An action has seven
values and cannot say that, so the request travels beside the actions:
`commands()` of the controller gives, for that one call, an entry per scene
(`("tie_thread", {})`), and the example passes it on as the optional
`commands` of the pool's `step` (`LocalSutureScene.step`,
`ParallelSutureScenes.step`; a worker receives it with its action). The scene
carries it out before it steps the frame (`GraspScene.command`,
`GraspScene.tie_thread`, `WorkerThread.tie`): one kernel launch that holds the
eyelet (`ThreadTissueCoupling.hold_eyelet`), the stand-in for a knot that
`fem_needle_suture` ties with. The thread no longer slides through a held
eyelet and pulls its material point in every direction. The kernel refuses an
eyelet that the thread does not pass through, on the device, and nothing is
read back for it; the observation after the frame must report one held eyelet
(`thread_tension["ties"]`), or the procedure stops with what the scene
reports.

The tightening (`_begin_tighten`, `_tighten_target`). The palm moves straight,
without turning, along the chord of the needle from its tail to the point
between the jaws: in the runs below upwards at 48 to 51 degrees, back over the
wound. The thread hangs on the tail and comes to lie along that line when it
is taut, so its pull points at the grasp and turns the needle in the jaws with
no lever. Speed and ramp are those of the insertion, 3 mm/s after 0.5 s.
Before the tool moves, the whole line to the travel bound is asked for reach,
as every other way of the procedure is.

The stop is the tightness rule of `fem_needle_suture`: a tetrahedron of the
tissue at half its rest volume (`TIGHT_VOLUME_RATIO` = 0.5 in
`_src/solvers/thread/tissue.py`, the default of that example's
`--least-volume`; one number for both examples). The scene finds the least
volume ratio `J` of its tetrahedra once per finished frame anyway, in the
check that no element is inverted, and now keeps it
(`diagnostics["least_volume_ratio"]`): no array is read for the rule. A stop on
a measured number needs its way to rest. From the commanded speed
`v u^2 (3 - 2 u)` (`u` is the part of the ramp time `T` that has passed, 1 at
full speed) the ramp to rest covers

    d(u) = v T (u^3 - u^4 / 2),      d(1) = v T / 2 = 0.75 mm  (v = 3 mm/s, T = 0.5 s),

and `J` goes on falling over it: by 0.09 to 0.11 over those 0.75 mm in the
measured pull below, so a rule "stop when `J` is 0.5" would come to rest near
0.4. The
rule therefore predicts where a stop that begins now would end. With `s` the
commanded travel, `a` the travel of the palm itself along the direction (the
observed palm is behind its target) and `k` the slope of `J` against `a` that
this tightening has measured itself over the last `v T / 2` of `a`, the length
it then extrapolates over,

    J_rest = J + k (s + d - a),      k = (J - J_w) / (a - a_w),

and the stop ramp begins in the first frame with `J_rest <= 0.5`, to end at
`s + d`. `J` at or below 0.5 stops as well, whatever the slope. The self-test
asks that the least volume at the end of the tightening's hold is within
`|k| (s + d - a)` of 0.5: the fall that the rule predicted for its way to
rest.

The travel is bounded by `tighten_travel_bound`, 6.75 mm: twice the travel at
which the measured pull reaches the rule (3.37 mm, below). It is a measured
number of this stitch and is named so in the table at the top of
`_grasp_procedure.py`. A tightening that has gone twice as far as the
measured one without its rule does not behave as measured: the command comes
to rest at the bound by the same ramp and the procedure stops there with that
reason. At 6.75 mm the measured tissue still holds (least volume 0.12; the
measured pull fails at 13.0 mm). When the stop of the rule has ended the tool
holds for 0.5 s, and [the lock and the slackening](#the-lock-and-the-slackening)
follow.

What a run shows. The first entry marker of the scene turns from green to
yellow while the thread is tied there. The side panel has a second line per
scene, for example
`Stitch: wound 2.18 mm wide at the top (2.88 at rest), 1.43 mm at the track (1.75); thread tied at its entry, pulls the needle with 25.8 mN, an eyelet with 18.1 mN at most`.
None of it costs a read from the device. The widths are measured between
four material points of the surface that the plan gives
(`wound_gaps` in `examples/needle/_needle_insertion_geometry.py`): the two
points where the planned circle crosses the walls of the gap between its two
passages ("at the track": their distance), and the rim of each wall, found by
following the surface upwards from the crossing, in the plane of the circle,
while it is steeper than 45 degrees ("at the top": their distance along the
level direction from entry to exit). In a compact observation they are four
more rows of the one array of landmarks that is read for the entry and exit
markers; with the tissue in the observation they come from the positions
that are read anyway (`wound_widths`). The ties and the thread's pull are one
more row of the array of thread samples that every observation reads
(`thread_tension`: `ties`, `ties_asked`, `eyelet_force_N`, `needle_force_N`).

Measured on the CPU before the stages were written, with the scene's own
tissue, thread and coupling code (2026-11-19, runs `chord_lv02` and
`chord_lv05` of the design review's tool `pull_cpu.py`: `fem_needle_suture`
with this stitch as its one stitch, without the stitching: the needle is put
where its stitch ends, the thread is laid unstrained through the planned track
and tied at the first entry, and the driven needle pulls on along its chord).
Against the travel of the needle:

| Travel | Least volume | Thread on the needle | On the knot (the largest eyelet force) | On the exit eyelet | Wound at its top | At the track |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.998 | 0.1 mN | 0.1 mN | 0.1 mN | 2.880 mm | 1.848 mm |
| 1.38 mm | 0.808 | 6.3 mN | 4.3 mN | 3.6 mN | 2.711 mm | 1.777 mm |
| 2.72 mm | 0.591 | 13.6 mN | 9.3 mN | 7.0 mN | 2.488 mm | 1.679 mm |
| 3.37 mm | 0.500 | 17.4 mN | 12.0 mN | 8.3 mN | 2.377 mm | 1.627 mm |
| 5.04 mm | 0.300 | 27.1 mN | 19.2 mN | 11.0 mN | 2.083 mm | 1.478 mm |
| 6.72 mm | 0.119 | 39.3 mN | 26.7 mN | 14.0 mN | 1.813 mm | 1.336 mm |
| 13.0 mm | the step fails | 88.9 mN | 56.0 mN | 20.5 mN | 0.637 mm | 0.624 mm |

`J` falls nearly linearly down to 0.3, by 0.148 per mm (0.11 to 0.18 locally);
the thread's force on the needle grows by 5.1 to 5.4 mN per mm. From this
curve come the fall over the way to rest (0.11), the travel bound, and what
the stop was
expected to show: 3.4 mm of travel, 17 mN on the needle, 12 mN on the knot,
the top from 2.88 to 2.38 mm, the track from 1.85 to 1.63 mm. That tool has a
driven needle, the planned eyelets (5 and 5) and tissue that no insertion has
loaded; "at the track" is there the distance between the last eyelet in one
lip and the first in the other.

Measured on the device (one DGX Spark, MuJoCo Warp backend, null viewer,
2026-11-19; four default self-tests with a record per frame: `s1_first` alone
at 17:58, `s1_a`, `s1_b` and `s1_c` together at 22:52, so none is a timing
run; the procedure then ended with the hold of the tightening). Every run
completes that procedure, done at 33.4 to 33.8 s of simulated
time (frames 2005 to 2028), with 2 entries and 2 exits, 8 of 8 eyelets
threaded and none refused, one tie asked for and one eyelet held on the
device, no CUDA graph captured and no kernel module loaded after start-up, no
rejected tissue step and no rejected coupled attempt from the tie to the last
frame, and both jaws on the needle in every frame (0.667 N at least). The tie
is asked for at 30.6 to 31.0 s; the tightening takes 2.32 to 2.33 s:

| Commanded travel | Least volume | Thread on the needle | Largest eyelet force | Wound at its top | Between the walls at the track |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.876 to 0.886 | 2.1 to 2.7 mN | 1.1 to 1.3 mN | 2.841 to 2.851 mm | 1.725 to 1.731 mm |
| 0.5 mm | 0.927 to 0.950 | 0.2 to 0.5 mN | 0.8 mN | 2.849 to 2.855 mm | 1.729 to 1.732 mm |
| 1.0 mm | 0.963 to 0.965 | 2.4 to 2.5 mN | 1.8 to 1.9 mN | 2.845 to 2.850 mm | 1.729 to 1.731 mm |
| 2.0 mm | 0.896 to 0.900 | 6.4 to 6.7 mN | 4.7 to 4.9 mN | 2.740 to 2.750 mm | 1.682 to 1.686 mm |
| 3.0 mm | 0.818 to 0.826 | 11.3 to 11.8 mN | 7.9 to 8.3 mN | 2.606 to 2.619 mm | 1.623 to 1.628 mm |
| 3.5 mm | 0.791 to 0.802 | 15.1 to 15.6 mN | 8.9 to 9.3 mN | 2.541 to 2.554 mm | 1.635 to 1.640 mm |
| 4.0 mm | 0.703 to 0.713 | 12.0 to 12.8 mN | 12.5 to 13.0 mN | 2.421 to 2.435 mm | 1.519 to 1.524 mm |
| 4.5 mm | 0.674 to 0.685 | 18.0 to 18.9 mN | 13.6 to 14.2 mN | 2.360 to 2.372 mm | 1.502 to 1.507 mm |
| 5.05 mm | 0.586 to 0.601 | 21.6 to 22.3 mN | 16.6 to 17.4 mN | 2.317 to 2.333 mm | 1.538 to 1.548 mm |
| at rest, 5.55 to 5.60 mm | 0.543 to 0.554 | 25.8 to 26.2 mN | 18.0 to 18.8 mN | 2.174 to 2.185 mm | 1.426 to 1.432 mm |

- The stop. The rule began the stop with a measured slope of -0.120 to -0.128
  per mm and 0.94 mm of way left for the palm, predicting 0.490 to 0.500 at
  rest. At rest the least volume is 0.543 to 0.554 (0.547 to 0.557 in the last
  frame, 6 to 7 s later): 0.043 to 0.054 above the rule's 0.5, inside the fall the
  rule predicted for its way (0.112 to 0.120). The commanded travel is 5.55 to
  5.60 mm of the 6.75 mm bound (the palm 5.48 to 5.55 mm).
- The closure. The wound narrows at its top by 0.66 to 0.68 mm, from 2.84 to
  2.85 mm to 2.17 to 2.19 mm (23 % of its width), and between its walls at the
  track by 0.29 to 0.30 mm, to 1.43 mm; the two eyelets across the wound come
  0.30 to 0.31 mm nearer (1.91 to 1.60 to 1.61 mm). The thread pulls the needle
  with 25.8 to 26.2 mN, the knot with 18.0 to 18.8 mN and the exit eyelet with
  15.1 to 15.4 mN (up to 3.5 mm of travel the two carry about the same; from
  there on the knot's is the largest eyelet force in every frame); its most
  stretched element is 14 % longer than at rest (13 % in the last frame).
- Against the expected numbers: 1.63 to 1.65 times the travel, 1.52 to 1.54
  times the force on the needle, 1.50 to 1.57 times the force on the knot,
  1.31 to 1.35 times the closure at the top and 1.33 to 1.38 times at the
  track. All are inside a factor of two, and all on the side of more travel,
  force and closure, at a least volume that stays above the rule's. Nothing
  was tuned towards them.
- Why the curve of the CPU does not carry over (measured in these runs). The
  first 0.9 mm of travel take up slack (0.2 to 0.5 mN on the needle at
  0.5 mm). The tetrahedron with the least volume is another one: before the
  tightening that of the exit eyelet (0.88, left by the pull-through; it
  recovers to 0.96 while the thread is slack), at the stop one beside the
  knot that shares nodes with the first two eyelets; 3 tetrahedra are under
  0.7 and 13 under 0.9. Twice `J` drops by 0.07 within a quarter of a
  millimetre, at 3.5 and at 4.7 mm of travel in every run; at the first the
  thread's pull on the needle falls from 15 to 9 mN within two frames while
  its pull on the knot rises from 9 to 12.6 mN. The cause is not established
  by a record; a thread node that passes an eyelet is the known event of that
  kind. Between the drops `J` falls by 0.06 to 0.08 per mm, half of the CPU's
  slope. The slope over the rule's window takes a drop for slope, which is why
  the stop begins early and ends at 0.54 to 0.55 instead of 0.50.
- The needle in the jaws, by the controller's own numbers (`tighten_slip_m`,
  `tighten_slip_rad` of its report, largest since the tightening began): it
  moves 0.04 to 0.12 mm and turns 3.6 to 11.7 mrad. They are reported, with no
  bound on them. The line of the tightening keeps the joints 0.137 rad and
  38.7 mm inside their limits.
- Two scenes, each in its own worker process (run `s1_two_plain`, with
  `--num-envs 2 --viewer null --test`): the tie reaches each worker with its
  action, and the self-test passes for both scenes (travel 5.60 and 5.55 mm,
  the top 2.18 mm wide in both at the last frame, nothing captured or loaded
  after start-up).
- The older procedures, both removed since, ended where they ended (one
  self-test each, runs `s1_notracks` and `s1_nopull`): with
  `--no-thread-tracks` the pull ended at the entry and the procedure was done
  at 26.2 s, with `--no-pull-through` the lift by the tip ended it at 28.1 s;
  no call of either asked the scene for anything, and nothing was tied.
- Without the tool/tissue contact (`--no-tool-tissue-contact`, one self-test,
  run `s1_nocontact`) the stitch is the same: travel 5.55 mm, least volume
  0.543 at rest, the top from 2.84 to 2.19 mm, 26.2 mN on the needle, no
  rejected coupled attempt in the whole run. The tool is clear of the tissue
  in these stages.
- Two more default self-tests, of the text as it landed (runs `s2_a` and
  `s2_b` of the check before the landing, together, 2026-11-20, with a
  recorder that reads nothing from the device): both pass with the same
  counts. Travel 5.65 and 5.55 mm (84 % of the bound at most), least volume
  0.551 and 0.546 at rest, the top from 2.86 and 2.84 mm to 2.18 and 2.17 mm,
  between the walls from 1.73 to 1.42 and 1.43 mm, 26.3 and 26.1 mN on the
  needle, 17.3 and 18.6 mN as the largest eyelet force. The six runs differ
  by two frames of travel (0.1 mm).

What one tied stitch does not do:

- It does not close the wound. The top stays 2.17 to 2.19 mm wide, and the
  deep part of the gap stays: 1.43 mm between the walls at the track. More
  travel closes more only with tissue squeezed further than the rule allows
  (CPU: 0.80 mm at the top with a tetrahedron at 0.3, 1.02 mm at 0.14).
- There is no knot in the thread. The tie is a held eyelet, as in
  `fem_needle_suture`.
- Nothing holds yet when its hold ends. The thread is tied at its entry
  only, and the tension is the tool's. The lock and the slackening, next,
  change that.
- The tissue has no contact with itself, and the thread is elastic (2 MPa).

#### The lock and the slackening

When the hold of the tightened thread ends, the thread is tight only because
the tool pulls. The procedure goes on (`_stitch_target`, stages `lock`,
`slacken`, `hold`): it ties the thread a second time, where it comes out of
the tissue, takes the needle back until the thread between that tie and the
needle is slack, and holds. The thread between the two ties then holds the
tissue by itself. The tool still holds the needle.

The lock. In the first call after the hold the controller asks for the second
tie beside its action, as for the first:
`("tie_thread", {"passage": n - 1, "last": True})`, the last eyelet of the
last of the `n` planned passages (`expected_passages` of the plan, 2 here),
the hole through which the thread leaves the tissue. The observation after
the scene's next frame must report two held eyelets
(`thread_tension["ties"]`), or the procedure stops with what the scene
reports. Nothing new runs on the device: it is the tie of the first knot at
another eyelet (`WorkerThread.tie(passage, last)`), one launch of a kernel
that start-up has loaded.

The slackening (`_slacken_target`). The palm goes back along the line of the
tightening, towards where the tightening began, at its speed and ramp
(3 mm/s after 0.5 s), and no further than where it began: the tool came along
that line, so it is not asked for reach again. The thread to the needle is
slack when its force on the needle (`thread_tension["needle_force_N"]`) is
below the weight that the scene itself applies to the needle. The scene
publishes that weight as one more key of its geometry, `needle_weight_N`
(`GraspScene.grasp_geometry`): the needle's mass, read once when the scene is
constructed, times the 9.81 m/s² of the kernel `_needle_weight` that adds the
weight in every step; 2.943 mN here, and zero for a scene that applies none
(`SutureSceneConfig.needle_weight`). No number about the needle is written
into the controller, which needs the key and the number of planned passages
wherever it needs the length of a thread element.

The stop. The stop ramp begins in the first frame in which the force is below
the weight, from the speed that the command then has (`ramp_way`, the inverse
of `ramp_speed`). From the full speed that stop takes `v T / 2` = 0.75 mm,
and here those 0.75 mm matter: the force on the needle is not monotone in the
travel back (the measurement below). It falls to a least value where the
thread is slack and rises again beyond it, because the needle then pushes the
slack thread together against the tied eyelet; the number that is read is the
size of the thread's force and does not tell a pull from a push. The command
therefore comes to that frame slowly. With `b` the commanded travel back and
`k` the slope of the force `F` against `b` that this slackening has measured
itself over the last `v T / 2` of `b` (as the tightening measures its own;
only over at least the travel of one frame at full speed, and only while `F`
falls), the thread is predicted to carry nothing at

    b_slack = b + F / (-k),

and until the stop begins the command goes no faster than the stop ramp
allows for the travel that is left to `b_slack` (`ramp_speed(F / (-k))`; the
pull slows down in the same way over the arc that its tail has left). The
stop that begins under the weight then ends within the travel of one frame of
`b_slack`. No number is new: speed, ramp and window are the tightening's and
the threshold is the scene's. The slope is taken over the commanded travel,
not over the palm's own: the palm springs forward as the load falls, and a
prediction in the palm's travel could ask the command to wait for a palm that
never arrives. Without a slope the command is not slowed, and the stop takes
the way of the speed that it has. A slackening that arrives where the
tightening began without the force below the weight ends there as well: the
controller's report then has no force at which the stop began, and the
self-test fails.

Then the tool holds for 0.5 s and the procedure is done. `report()` of the
controller has the travel back (`slacken_travel_m`), the weight that it
stopped by (`needle_weight_N`), the force and the slope when the stop began
(`slacken_stop_force_N`, `slacken_slope_N_per_m`), the time at which the
slackening ended (`slacken_end_time_s`), the thread's force on the needle as
last seen, also after the procedure is done (`thread_needle_force_N`), and
the largest travel and turn of the needle in the jaws since the second tie
(`lock_slip_m`, `lock_slip_rad`; reported, with no bound on them).

What a run shows. The last exit marker of the scene turns from blue to yellow
while two ties hold, as the first entry marker does with the first; a reset
of the scene clears both. The panel line says where the thread is tied, at
the end for example
`Stitch: wound 2.46 mm wide at the top (2.88 at rest), 1.52 mm at the track (1.75); thread tied at its entry and at its exit, pulls the needle with 0.3 mN, an eyelet with 17.4 mN at most`.
The example keeps, per scene, the wound's widths and the least tissue volume
of two observations that the controller acted on: the one in which the
tightening began and the one in which its hold ended (`Example._follow_stitch`;
host values that every observation has). The closure that the stitch keeps
is measured against them.

The self-test (`Example._check_stitched`) asks, per scene: two ties asked for
and two held; the tightening ended by its rule inside its travel bound, with
the least volume at the end of its hold within the rule's allowance (the
least volume of the last frame is reported and not asked for: the thread is
slack to the needle then); the slackening ended by its rule within the travel
of the tightening, and the thread's force on the needle is below
`needle_weight_N` in the last frame; both jaws on the needle in every call
from the first tie on; the slackening ended at least 1 s before the last
frame; and the wound is narrower at its top and at its track than in the
reference shape and than it was when the tightening began. A stitch that
keeps nothing of its closure fails the self-test. Its line has the travel
back, the kept closure at the top and at the track and each as a share of the
closure at the end of the tightening's hold.

Measured on the CPU before the stages were written (the night of 2026-11-19
to 20; tool `slack_cpu.py`, a copy of the design review's `pull_cpu.py` with
the act added: `fem_needle_suture` with this stitch as its one stitch, without
the stitching, the thread tied at the first entry, then the motion law of
this controller: tighten to a given travel, hold 0.5 s, tie at the last
eyelet of the last passage, back along the same line, rest). The needle of
that scene weighs 2.943 mN as well. Tightened to 3.37 mm (the travel at which
that pull is at the tightness rule: 17.1 mN on the needle, least volume
0.50), all the way back to where the tightening began (run `curve_337`):

| Travel back | Thread on the needle | Least volume | On the exit eyelet (tied) | Wound at its top | At the track |
| --- | --- | --- | --- | --- | --- |
| 0 | 16.7 mN | 0.504 | 8.0 mN | 2.385 mm | 1.525 mm |
| 0.20 mm | 13.5 mN | 0.549 | 7.2 mN | 2.412 mm | 1.524 mm |
| 0.51 mm | 8.9 mN | 0.620 | 6.5 mN | 2.459 mm | 1.525 mm |
| 0.90 mm | 4.3 mN | 0.682 | 7.0 mN | 2.537 mm | 1.560 mm |
| 1.30 mm | 0.7 mN | 0.664 | 8.6 mN | 2.596 mm | 1.580 mm |
| 1.70 mm | 2.6 mN | 0.569 | 10.0 mN | 2.635 mm | 1.590 mm |
| 2.50 mm | 4.5 mN | 0.432 | 11.6 mN | 2.652 mm | 1.591 mm |
| 3.35 mm | 6.2 mN | 0.274 | 13.5 mN | 2.656 mm | 1.585 mm |

The force is below the weight only from 1.05 to about 1.85 mm. Beyond its
least value it is a push, which loads the tied eyelet and squeezes the tissue
under it further than the tightening did. With a stop that begins under the
weight at full speed (run `stop_337`) the command ends at 1.80 mm, 0.46 mm
past the least force, with 2.84 mN on the needle: 0.1 mN under the weight.
The same stop from a tighter state (run `stop_485`: tightened to 4.85 mm,
24 mN, least volume 0.32, which is beyond the tightness rule) drives the
least volume to 0.20, and the tissue step fails 0.25 s later. With the rule
as it is built (run `built_337`) the stop begins at 1.05 mm with the command
at 2.1 mm/s and ends at 1.32 mm, 0.02 mm from the least force of the curve,
with 0.64 mN on the needle (22 % of the weight), steady over the next 1.5 s;
the least volume is 0.656. Kept 1 s after the slackening ended: 0.279 mm of
the 0.501 mm that the tightening closed at the top (56 %) and 0.166 mm of
0.224 mm at the track (74 %). (An earlier form of the rule, run `valley_485`
from the tighter state: 0.43 mN at rest, 56 % and 70 % kept.) From these runs
the device was expected to show, before it ran: about 2.2 mm of travel back,
under 1 mN on the needle at rest, 56 % of the closure kept at the top and 70
to 75 % at the track.

Measured on the device (one DGX Spark, MuJoCo Warp backend, null viewer,
2026-11-20; four default self-tests of this text with a record per frame that
copies host values only: `s2_first` alone at 00:38, `s2_a`, `s2_b` and `s2_c`
together at 00:47; other jobs loaded the machine, so none is a timing run).
Every run completes the procedure through `tie`, `tighten`, `hold`, `lock`,
`slacken` and `hold`, done at 35.22 to 35.45 s of simulated time (frames 2112
to 2126), with 2 entries and 2 exits, 8 of 8 eyelets threaded and none refused,
two ties asked for and two eyelets held on the device (the first of the first
passage and the last of the second), no CUDA graph captured and no kernel
module loaded after start-up, no rejected coupled attempt and no rejected
tissue step from the first tie to the last frame, and both jaws on the needle
in every frame from the first tie on (0.668 N at least). The second tie is
asked for at 33.5 to 33.7 s; the slackening takes 1.18 to 1.20 s:

| | Thread on the needle | Least volume | Wound at its top | Between the walls at the track |
| --- | --- | --- | --- | --- |
| The tightening begins | 1.98 to 2.58 mN | 0.880 to 0.891 | 2.843 to 2.853 mm | 1.726 to 1.732 mm |
| End of the tightening's hold, travel 5.55 to 5.60 mm | 25.35 to 26.13 mN | 0.544 to 0.559 | 2.176 to 2.191 mm | 1.429 to 1.437 mm |
| The slackening has ended, travel back 2.15 to 2.17 mm | 0.20 to 0.22 mN | 0.561 to 0.575 | 2.447 to 2.458 mm | 1.509 to 1.516 mm |
| 1 s later | 0.20 to 0.24 mN | 0.561 to 0.575 | 2.448 to 2.459 mm | 1.510 to 1.516 mm |
| Last frame, 5.1 to 5.3 s after the slackening | 0.29 to 0.52 mN | 0.561 to 0.575 | 2.447 to 2.457 mm | 1.509 to 1.516 mm |

- The stop. It began with 2.64 to 2.78 mN on the needle, under the weight of
  2.943 mN, with a measured slope of -11.2 to -10.8 mN per mm and the command
  slowed down from 3.0 mm/s to between 2.14 and 2.18 mm/s (the commanded speed
  steps up by 0.15 mm/s between two frames at most). The command ended 2.15 to
  2.17 mm back, of the 5.55 to 5.60 mm of the tightening (the palm itself 2.10
  to 2.11 mm). In 4 of the 4 runs the force in the frame in which the
  slackening ended is the least of the whole way back: the command came to rest
  where the thread is slack, not beyond.
- The thread to the needle is slack. In the last frame it pulls the needle with
  0.29 to 0.52 mN, 10 to 18 % of the needle's weight (0.41, 0.33, 0.29, 0.52 mN
  in the four runs), and never with more than 0.52 mN after the slackening. It
  is not constant while the tool holds: in `s2_first` it rises from 0.22 to
  0.41 mN in the 5 s after the slackening. The cause is not established.
- The kept closure. The tightening had narrowed the top by 0.66 to 0.67 mm and
  the width between the walls at the track by 0.29 to 0.30 mm. One second after
  the slackening the stitch keeps 0.39 to 0.40 mm of that at the top (59 %) and
  0.21 to 0.22 mm at the track (73 to 74 %); in the last frame 0.39 to 0.40 mm
  and 0.22 mm. The wound is then 2.45 to 2.46 mm wide at its top, where it was
  2.84 to 2.85 mm before the stitch and is 2.88 mm in the reference shape.
- Against the expected numbers. The CPU runs above gave about 2.2 mm of travel
  back, under 1 mN at rest, 56 % kept at the top and 70 to 75 % at the track;
  the design review's measurement of the same act, with the needle dropped
  instead of taken back, gave 0.25 to 0.32 mm (38 to 48 %) at the top and 0.19
  to 0.22 mm (61 to 71 %) at the track. The four runs are inside a factor of
  two of each (0.98 to 0.99 times the travel, 1.23 to 1.24 times the review's
  band at the top and 1.00 times at the track), and the force at rest is under
  1 mN up to one second after the slackening. Nothing was tuned towards them.
- The tissue after the lock. The least volume is 0.544 to 0.559 at the end of
  the tightening's hold, never under 0.541 from the lock on, and 0.561 to 0.575
  in the last frame: it is reported and not bounded, and it stays near the
  tightness rule because the thread between the two ties stays as tight as the
  tightening made it: in the last frame the knot carries 17.4 to 18.2 mN (the
  largest eyelet force, 17.7 to 18.7 mN at the end of the tightening's hold)
  and the eyelet that is tied at the exit 13.9 to 14.3 mN. The tetrahedron with
  the least volume is still the one beside the knot.
- The needle in the jaws since the second tie (`lock_slip_m`, `lock_slip_rad`):
  it moves 0.06 to 0.17 mm and turns 5.2 to 16.3 mrad while the load of the
  thread comes off it. Reported, with no bound.
- The controller by replay without physics: its calls of the four runs (2400
  each), replayed from the recorded observations on the CPU, give back every
  action bit for bit, every command and every stage. The controller of this
  text on the records of the earlier procedure (`s1_a`, `s1_b`, `s1_c`) gives
  the same actions in every call up to the one in which the hold of the
  tightening ends; on the records `s1_nopull` and `s1_notracks` it differs in
  no call.
- Two scenes, each in its own worker process (run `s2_two`, with `--num-envs 2
  --viewer null --test`): both ties reach each worker with its action, and the
  self-test passes for both scenes: travel back 2.16, 2.15 mm, 0.44, 0.24 mN on
  the needle in the last frame, kept at the top 0.40, 0.40 mm (60, 59 %) and at
  the track 0.22, 0.22 mm (74, 73 %), nothing captured or loaded after
  start-up.
- Two more default self-tests with the same record per frame, run by the
  independent check of this step (`c_a` and `c_b` together at 02:01): both
  pass, with two ties held of two asked for, nothing captured or loaded after
  start-up, no rejected coupled attempt from the first tie on and both jaws
  on the needle in every frame from it. Done at 35.62 and 35.40 s (frames 2136
  and 2123); travel back 2.15 mm; 0.23 and 0.21 mN on the needle when the
  slackening ended, the least of the way back in both, and 0.26 and 0.29 mN in
  the last frame; kept one second after the slackening 0.40 and 0.40 mm at the
  top (59 %) and 0.22 mm at the track (73 %). `c_a` lies outside the ranges of
  the four runs above in three numbers: it is done 10 frames later, its
  tightening narrows the top by 0.68 mm, and its top is 2.44 mm wide at the
  end.
- The older procedures, both removed since, asked the scene for nothing and
  tied nothing. With `--no-thread-tracks` (one self-test, run `s2_notracks`)
  the pull ended at the entry and the procedure was done at 26.2 s, as before.
  With `--no-pull-through` the lift by the tip ended the procedure at 28.1 s in three
  of four self-tests: `s2_nopull_2` of this text, and `b2_nopull_1` and
  `b2_nopull_2` of the text before it. The fourth, `s2_nopull` of this text,
  stopped at 19.0 s on the way to the tip, in the stage `approach`, with the
  scene's "RCM GPU tool pose is unreachable within unchanged joint
  limits/tolerances". The controller of the text before this step, replayed on
  that run's recorded observations, gives the same commands in every one of its
  1142 calls: the stop does not come from the controller of this step. Its
  cause is the grasp of that older procedure, which has no reach test and takes
  the roll joint to its limit: in the four runs of it that pass (with
  `s1_nopull`) the commanded targets keep that joint 0.017 to 0.031 rad inside
  the limit; in `s2_nopull` the last target is 0.004 rad inside and the next
  one beyond. The needle that the tool lets go of comes to rest up to 0.39 mm
  and 3.2 degrees apart between device runs (17 recorded runs), and `s2_nopull`
  is at one end of that spread (four other recorded runs, among them `s1_a` of
  the text before this step, have the needle within 0.05 mm and 0.4 degree of
  it; they used the pull, whose plan asks for reach). By this
  `--no-pull-through` could stop in the same way before this step as after it; no
  run of the text before it did (three of three passed).
- Not seen in a window by this step: the second marker, the panel line and the
  pace of the new stages are untested on screen.

What a locked stitch does, and what it does not do:

- It holds the wound narrower without the tool's pull: 0.39 to 0.40 mm at the
  top and 0.22 mm at the track, 5.1 to 5.3 s after the slackening, with 0.29 to
  0.52 mN of thread force on the needle. The tool is free of the thread; it
  still holds the needle.
- It does not close the wound. The top stays 2.45 to 2.46 mm wide (2.84 to
  2.85 mm before the stitch), and 1.51 to 1.52 mm stay between the walls at the
  track.
- It keeps 59 to 60 % of what the tightening closed at the top and 73 to 74 %
  at the track. The rest was held by the thread between the exit and the
  needle, which pulled the exit lip up and back over the wound while the tool
  held it tight; that pull is gone when the thread is slack (an explanation
  from the direction of the tightening, not a separate measurement).
- There is no knot in the thread. Both ties are held eyelets, as in
  `fem_needle_suture`: they cannot slip and need no thread.
- The tool does not put the needle down, cut the thread or take a second bite.
  The thread behind the first tie stays on the pad.
- The tissue has no contact with itself, and the thread is elastic (2 MPa).

#### The grasp at the needle tip

Where on the tip the jaws close is not a setting. It is the point of the needle
nearest to where it leaves the tissue at which the jaws stay out of the tissue
while they come down on it and close
(`ProceduralGraspController._plan_tip_grasp`, `jaw_clearance` in
[_grasp_procedure.py](../src/newton_solver_suturebot/examples/grasp/_grasp_procedure.py)):

```
s* = max { s in [r_w, s_free] :  c(s) >= 0 }
c(s) = min over the jaw capsules j of  [ min over x on S of dist(x, K_j(s)) - R_j ]
```

- `s`: length of needle between its tip and the grasp point. `r_w`: radius of
  the wire. `s_free`: length from the tip over which the wire is clear of the
  tissue.
- `S`: the rest surface of the tissue (`reference_positions`,
  `reference_triangles` of the scene's geometry). With the tool away the tissue
  around the resting needle is within 0.03 mm of it, and a run without a view
  does not hand the tissue to the controller.
- `K_j(s)`: what the axis of jaw capsule `j` sweeps for a grasp at `s`: the axis
  at every jaw angle from closed to the opening of the transfer (0.3 rad per
  jaw) at the grasp pose, and the open axis over the descent (12 mm along the
  jaws). The grasp pose follows from the pose of the resting needle, with the
  twist and bite above.
- `R_j` and the axes are read from the tool model: every capsule on a jaw body
  of the scene's `grippers` entry, eight on each jaw of the registered tools
  ([method](method.md#the-collision-capsules-of-a-tool)). The tools touch the
  tissue with the radii of that one set; an entry `capsule_contact_radii` is
  read only from a tool model that publishes other radii for the tissue (the
  axis is then shortened so that its caps reach as far as those of the model
  capsule, as the tool/tissue contact has it).
- The open jaws must also be clear of the wire itself when they come onto it:
  `s*` is taken among the points at which the centre line of the wire is
  further from the axis of every jaw capsule, turned open, than the radius of
  the capsule, its bound and the radius of the wire (`wire_clearance`). With
  the twist and bite of the earlier grasp (the lift by the tip, removed since) the needle's
  tip lies towards the roots of the jaws: the open jaws of the needle driver
  are clear of the wire up to 2.10 mm of needle from the tip and at no point
  further from it.

The clearance is evaluated on host arrays, on surface samples a quarter radius
apart and then a sixteenth of a radius apart where the nearest point can lie;
what the sampling can miss is subtracted, so a jaw that is reported clear is
clear of that surface, and the grasp point lies up to 0.1 mm of needle further
from the tissue than the exact one. The plan is made while the tool rises
along its shaft, a step of work per frame: one capsule of a jaw against the
tissue, or the reach of a point (about 60 steps, the largest controller call
of the regrasp 5.9 ms in the runs of 2026-11-17; before that
day it was made in 14 frames of 5 to 35 ms while the tool stood still).
It stands only if the needle has not
moved by more than 0.04 mm while it was made, and is made again otherwise,
also when the needle is found to have moved while the tool still rises or
when it stands at the pregrasp:
tissue that the jaws pressed down (with the
[tool/tissue contact](#tooltissue-contact) on) lets the needle come up by
about 2 mm when the tool backs away. The procedure
stops if no point of the wire that sticks out has clear jaws ("The needle tip
does not stick out of the tissue far enough ..."), also if none
of them is within the reach of the tool
([above](#the-way-to-the-tip-and-the-pull)), or if the needle has not come to
rest 3 s after the plan was begun.

With the fitted capsule set and the earlier grasp (the lift by the tip, removed since),
measured on the device on 2026-11-17 in seven
whole runs: the grasp point is 1.84 to 1.86 mm from the tip in four runs
without the tool/tissue contact and 1.78, 1.87 and 1.95 mm in three runs with the
[tool/tissue contact](#tooltissue-contact) on. The jaws close on the wire at
that point and carry the needle through the lift at the tip with 0.87 N per
jaw in the median and 0.83 N at least; no jaw leaves the needle. The roll
joint of the needle driver stands 0.008 to 0.009 rad (0.45 to 0.49 degrees)
from its limit at that grasp. The reset of the first grasp seats each jaw on
the wire with two capsules side by side
([method](method.md#the-collision-capsules-of-a-tool)).

The demo ends with 2 entries and 2 exits of the needle tip (frames 83 and 211,
176 and 312), with the tool/tissue contact off and on. With the four capsules
per tool of before it ended with 3 and 3: the third pair, at 11.0 and 12.1 s
in the lowering of the first lift, does not occur with the set
([method](method.md#the-collision-capsules-of-a-tool) has the frames).
Passages of this page that report runs made with the capsules of before say
3 entries and 3 exits; they are records of those runs.

The numbers of the next paragraph were measured with the four capsules per
tool of before (jaw radius 0.98 mm).
Without the tool/tissue contact (`--no-tool-tissue-contact`; the default of
then) the tip stands 5.0 mm above the rest surface, the wire
is free over 4.8 mm, and the grasp point is 1.7 mm from the tip (1.85 to
1.94 mm with the tool/tissue contact on, where the needle rests 0.1 to 0.2 mm
higher). Before 2026-11-17 it was fixed at 3.5 mm, where the lower jaw closed 1.5 mm
inside the tissue; now the drawn lower jaw is 0.1 mm clear of the deformed
tissue while it closes. A jaw capsule of 2 mm radius with the same reach would
give 1.4 mm; the same radius on today's axis leaves no point, and the procedure
would stop. Not covered by the rule: the first grasp, where the jaws stand
up to 2.75 mm inside the tissue at the release, and the first part of the way
back out of it; with the contact off these overlaps exchange no force. The way
over to the tip, where the upper jaw passed up to 1.9 mm through the tissue,
is clear of it since 2026-11-17
([above](#the-way-to-the-tip-and-the-pull)).

Green spheres mark where the needle tip entered the tissue and blue spheres
where it came out. Each sphere is bound to the tissue when its event is
reported: to the point of the drawn tissue surface closest to the event, as
weights of one surface triangle. From then on it is drawn at that material
point, so it moves with the tissue when the needle lifts it or a tool presses
on it. A scene shows at most eight spheres for each passage of the planned
arc through the tissue (16 with the default bite); a further one takes the
place of the oldest. If the physics fails, a
run with a viewer stops stepping and keeps the last state on screen; a run
without one raises.

Space pauses/resumes; period advances one step; Reset rebuilds the scene; Esc
closes it. The initial camera focuses on the needle and jaws; `--overview` frames
the complete shafts and trocar ports. Use `--controller hold` to inspect the settled setup without executing
the insertion arc. `--speed` specifies arc speed in metres/second and
`--settle-seconds` sets the initial grasp-settling interval. `--lift-height`
sets the lift after the insertion in metres (0 to 0.01, default 0.01);
`--lift-height 0` restores the insertion-only procedure. After the lift the
tool takes the needle by its tip and pulls it on along
its curve, out of the tissue, with the thread following through the needle's
track; with `--no-thread-tracks` the thread stays on the surface and the
example ends after the lift, because the pull needs a thread that follows the
track. A
lift rises with smooth endpoint speeds in
the time that keeps its peak speed at 80 % of the action rate limit (3.1 s for
10 mm, at least 0.75 s), holds for 0.5 s and returns in the same time. Grasp
loss pauses its clock under the existing contact-loss checks. The viewer
defaults to 2880 frames (48 s of simulated time); the sequence ends before
45.0 s (the needle is clear of the tissue at 30.3 to 31.0 s; the draw of the
thread takes 6.35 s and its hold 0.5 s; the stitch ends at
42.4 to 42.6 s in the runs, and at 45.0 s at the latest: a tightening to its
travel bound takes 3.0 s, and the slackening at most as long again). With a
lift, `--test` fails a run of the default speed and settling time that ends
while the procedure is under way, or less than 1 s after the slackening has
ended: a shorter run (`--num-frames`) has to reach that. The clip recipe
(`docs/videos.json`) records 2640 frames with `--test`: 44 s. That length is
measured: the latest frame at which the three runs of
[the draw of the thread](#the-draw-of-the-thread) were done (2552), 60 frames
more, rounded up to a multiple of 60; done is half
a second after the slackening has ended, so the second that the self-test asks
for is inside. The stages end on what the simulation reaches, not at fixed
times. The clip in `docs/media` was recorded with this recipe on 2026-11-20
with the code of the commit that brought the draw of the thread, from a
close camera that changes its view three times (the recipe's `camera`); the
run passed the self-test. See
[provenance](provenance.md#current-clips-2026-11-20) for all nine clips.

Validate one scene, then display more independent scenes:

```bash
./docker/run.sh --headless fem_needle_grasp --test
./docker/run.sh fem_needle_grasp --num-envs 8
```

One scene runs in the viewer's own process and CUDA context
(`LocalSutureScene`). With `--num-envs` above one, each scene owns its physics
and puncture history in a spawned worker (`ParallelSutureScenes`). Display
spacing does not alter physical coordinates. This example needs no scaling report
from the RL project. The default has four coupled intervals per 60 Hz simulation
frame and four native MuJoCo microsteps per interval; simulated time and video
playback rate are distinct from measured live FPS. This remains a procedural
insertion demonstration, not a complete suturing or knot-tying controller.

The physical-grasp scene reuses the unchanged starting-state FEM force assembly
within each coupled attempt. It still simulates the complete selected region
and solves each changed interface load to the configured tolerances or, with
the trial budget that this example sets by default, to the best trial of the
budget ([below](#coupling-tolerance-and-trial-budget-default)). Unsupported
solver or differentiation modes keep the original assembly path.

Every native MuJoCo microstep checks for an integration reset before accepting
its endpoint. An unstable trial is rejected and retried with the existing
adaptive timestep logic. Scene diagnostics report `native_clock_s`,
`native_clock_error_s` and `native_guard_rejections`; the accepted native clock
must follow the accepted simulation time. A rejected trial can emit a MuJoCo
warning even when a smaller retry succeeds. These checks and the held-start
cache are described in the [method](method.md#physical-distal-grasp-and-worker-owned-thread).


### Tool/tissue contact

```bash
# The default, written out: the contact on the sweep of the capsules (the procedure completes; what it costs is below):
uv run --locked python -m newton_solver_suturebot.examples fem_needle_grasp --tool-tissue-contact swept
# Without the contact, also `--tool-tissue-contact off`: the tools pass through the tissue
uv run --locked python -m newton_solver_suturebot.examples fem_needle_grasp --no-tool-tissue-contact
```

The option of the example takes `swept` (its default) or `off` (also
`--no-tool-tissue-contact`). `GraspScene`, `SutureSceneConfig` and
`WorkerThread` keep the contact off; only this example turns it on, by handing
`--tool-tissue-contact swept` to its scene (`TOOL_TISSUE_CONTACT` in the
example). A Python caller that builds scenes through `SutureSceneConfig`
therefore has no contact unless it asks for one, with
`"--tool-tissue-contact"` (first-version law) or `"--tool-tissue-contact",
"swept"` among `SutureSceneConfig.example_arguments`. The scene has two laws.
The first-version law is a law of the scene only: it was an option of this
example (`--tool-tissue-contact first`, or the option without a value) until
2026-11-19 and was taken out because the default procedure stops with it at
6.4 s (below). The example `fem_tool_tissue` runs both laws with the tools and
the tissue alone, on a scripted path
([Tool and tissue contact rig](#tool-and-tissue-contact-rig)). The
first-version law is described first, the law on the sweep of the capsules
[after it](#on-the-sweep-of-the-capsules).

Why the swept law is the default, and what it costs. With the swept law every
run of the two groups below reaches the end of the procedure (GPU, one DGX
Spark, 2026-11-17), each with 2 entries and 2 exits of the needle; the
self-tests and the window runs capture no CUDA graph and load no kernel
module after start-up. The procedure changed on that day, so the runs are
of two kinds:

- The procedure as it is now, with the pull through the tissue and the thread
  tracks (done at 30.4 to 30.9 s of simulated time): **8 of 8 runs**. Five
  with `--tool-tissue-contact swept` written out (four self-tests and one run
  with records per physics interval, 16:28 to 16:45;
  [the pull](#the-way-to-the-tip-and-the-pull)), and three of the default,
  with no option for the contact, the budget or the thread, on the tree that
  made the contact the default (`/tmp/suturebot-gpu-work/contact_default2/`,
  a work folder outside this repository, `gpu/logs` there): the self-tests
  `h1_def_1` and `h1_def_2` (17:21 to 17:24, the two together; all 8 eyelets
  threaded) and the window run `w_def_1950` (17:24 to 17:26, alone on the
  GPU, 1950 frames). What the window run shows of the
  [frame time budget](#frame-time-budget-default) is in that section: one
  interval between two pictures of 67.1 ms, 0.4 ms over the 66.7 ms, and
  100.7 s on the screen for the 30.9 s of the procedure. No window run of
  this procedure was made without the contact or without the budget.
- The procedure of before the thread tracks, in which the pull ended at the
  needle's entry (removed since; done at 26.2 to 26.7 s): **11 of
  11 runs** in one shared round (15:25 to 15:57; its notes and logs are
  `/tmp/suturebot-gpu-work/round2/notes.md` and `r2_1117/logs` there, a work
  folder outside this repository): five with the window, three self-tests,
  three with records per physics interval. In the window (1750 frames, each
  run alone on the GPU) the contact costs time: with whole frames the 26 s of
  that procedure take 76 s on the screen instead of 58 s, and 111 of the 1750
  frames take longer than 66.7 ms (the longest 470 ms) instead of 9 (the
  longest 121 ms). With the frame time budget, which is a default of the
  example as well, no interval between two pictures is over 66.7 ms in three
  of three runs (the longest 65.4, 60.9 and 60.4 ms), and that procedure
  takes 100 to 111 s on the screen (70 s without the contact).

The other measurements of this section are earlier runs of the same day, made
with the option written out and the procedure of their hour (the lift by the
tip, later the pull to the entry; both removed since).

What it does. The shaft, the wrist and the two jaws of both tools are covered
with spheres along the axes of their collision capsules, 274 in all, with the
radii of those capsules: the contact uses the one capsule set of the tools,
which is fitted to the drawn tool
([method](method.md#the-collision-capsules-of-a-tool)). The numbers of this
section were measured with the four capsules per tool of before 2026-11-17
(162 spheres; 0.98 mm for the jaws of the needle driver, 3.0 mm for the wrist
and 4.18 mm for the shaft tube). After every
accepted physics interval each sphere finds the nearest point of the tissue
surface, raises its contact force while it is inside the tissue and lowers it
while there is a gap. During the next interval that force pushes the tissue at
that point and, with the opposite sign, the tool link. Nothing is added to the
trial loop of the needle/tissue coupling. There is no friction.

What it carries, measured with a scripted jaw-sized chain of nine spheres on
free tissue (no tool behind the chain, CPU, native rigid backend):

- A light touch. Pressed 0.3 mm into the tissue at 3 mm/s, the chain rides
  0.06 mm (at most 0.11 mm) inside the surface while it moves, is 0.03 mm
  inside 0.1 s after it stops and lies on the surface after 1 s, with 16 mN
  on six spheres and a dimple of 0.37 mm. When it leaves, the force is gone
  after 0.11 s. No step is rejected.
- Not a tool that is driven into the tissue. Commanded 1.5 mm in, the dimple
  starts to alternate from one interval to the next at 0.4 mm; at 0.86 mm
  (67 mN on nine spheres) the first step is rejected for an inverted
  tetrahedron, and at 1.27 mm the step control gives up. Tip first, with one
  sphere in contact, the first rejection comes at 0.80 mm with 20 mN.
- Not the held border of the simulated region. There the tissue cannot give
  way, the force grows until the tool does, and the free nodes beside the
  fixed ones are pushed through their tetrahedra.

What happens in the default procedure with it (four runs on `cuda:0`: two
measured ones, the self-test and a census; all the same). Up to 6.1 s nothing
touches, and trials, rejected steps, stitch events and needle path are those
of a run without the contact (the needle position differs by less than
0.0001 mm, as between two runs without it). At 6.12 s the tip of the second
jaw of the needle driver meets the pad at the border of the active region, on
a triangle whose vertices are fixed or outside the region. While the jaw moves
on, 0.1 to 0.3 mm inside the surface, the force on that one sphere rises by 1
to 2.5 mN per interval to 43 mN at 6.27 s, with no tissue motion. Then the
nearest point moves on to a triangle with one free node. At 90 to 100 mN that
node is 0.8 to 1.5 mm from its rest position, beside fixed nodes 1 mm away.
The step control rejects the attempts for a nonpositive tetrahedron volume and
halves the interval, 21 to 66 times, and the scene fails at 6.38 to 6.42 s
with "Unresolved FEM tetrahedron motion". A run with a viewer keeps the last
state on screen; a run without one raises, so `--test` fails. The stitch then
has two entries and two exits (without the contact the third pair came at
11.0 and 12.1 s with the capsules of before). The tool has been pushed back by 0.13 to 0.15 mm.

For comparison, without the contact (`--no-tool-tissue-contact`) the drawn
tool is inside the
deformed tissue in 157 of the first 826 frames: the second jaw by up to
1.33 mm (0.83 mm typically), the first jaw by up to 0.45 mm, the wrist by up
to 0.50 mm and the yaw pin by up to 0.61 mm; the shaft stays 5.4 mm clear.

Cost while it is on and nothing touches: 0.06 ms more for the thread's step
graph of an accepted interval (1.19 ms against 1.13 ms), and stage medians
0.5 ms per frame above those of four runs without it (19.0 against 18.5 ms
while settling, 28.8 against 27.9 to 28.4 ms while inserting). No frame over
66.7 ms is added before the failure, and no CUDA graph is captured and no
kernel module is loaded or compiled after start-up.

The law, its parameters and the reasons for these limits are in the
[method](method.md#tooltissue-contact).

#### On the sweep of the capsules

`--tool-tissue-contact swept` selects the contact that the rig
`fem_tool_tissue --contact swept` runs
([what it does](#tool-and-tissue-contact-rig)): after every accepted physics
interval each capsule of the tools is swept to the pose it will have at
the end of the next interval; the free surface nodes it reaches are placed on
it, at most by their allowed move, and held there through that interval by a
block in the tissue matrix; their elastic force along the push direction is
the reaction on the tool, one interval late; tissue that is held (fixed nodes,
the pad outside the simulated region) or cannot follow at that rate stops the
tool through one stored force per axis piece. It is the default of
`fem_needle_grasp`; the scene keeps it off. Since the
regrasp plans where the jaws take the needle tip
([the grasp at the needle tip](#the-grasp-at-the-needle-tip), 2026-11-17) the
procedure completes with it in every run that was made (nine of
nine in the runs of this subsection: five with records per physics interval,
the self-test, three with the window; 11 of 11 in the later round and 8 of 8
with the pull through the tissue, [above](#tooltissue-contact)); before, with the grasp fixed 3.5 mm behind the
tip, it completed in four runs of eight. What still fails with the contact on is
marked below: frames over 66.7 ms with whole frames, steps rejected at the end
of the approach, and how far a jaw stands inside tissue that stops it.

Where it sits. `WorkerThread(scene, config, tool_tissue="swept",
tool_effective_mass=masses)` builds the law. Its update is the last launch of
the thread's accepted step, in the thread's step graph of each state buffer,
which the start-up call captures and launches: at the accepted state, on the
pad surface that the thread's tissue coupling has just refitted. With the
thread's loads of every attempt the worker adds the stored reaction to the
body forces and, after the thread's forces on the tissue, the forces that
balance the elastic force on the held nodes. The holds are refreshed with
every tangent of the tissue solver. A scene reset builds a new tissue solver
and a new worker, so no stored force, no hold and no placed node is left
behind; the start-up call puts back the tissue, the contact state and the
holds that its discarded steps changed.

Constants, all from the scene (none tuned): sweep horizon 1/240 s, the full
physics interval, also when an attempt is shorter; `K_c = m / h^2` per tool,
with `m` the smallest effective mass of the tool at an end of one of its drawn
capsules in any direction, every actuator free, on the rigid model at the
reset state: 23.12 g and 1,331.6 N/m for the needle driver (jaws closed on the
needle), 20.46 g and 1,178.7 N/m for the parked forceps; integral gain
`K_c / 16`; reach of the surface search 20 mm, the thickness of the pad;
stretch limit 0.2, the tissue solver's; move fraction 1/8; 162 axis pieces and
2,554 free surface nodes.

Measured in the default procedure on `cuda:0` (one DGX Spark), 2026-11-17,
with the repaired law (`swept_tool_tissue_contact.py`, sha256 `28f29a5c...`)
and the planned grasp at the tip: nine runs with the contact (five with
records per physics interval, the example's self-test, three with the window:
two with `--frame-budget 66.7`, one without), and three self-tests without
the contact. "Inside" is how far the deepest
point of a drawn capsule (the drawn radius about the axis from the first to the
last probe) is inside the deformed tissue surface at the end of a physics
interval; times are those of the runs with records per interval.

- **Completion: nine of nine runs** reach the end of the procedure, with both
  lifts. With the grasp fixed 3.5 mm behind the tip (before the regrasp change
  of 2026-11-17) four of eight runs stopped in the lift by the
  tip or the lowering after it: the jaws closed on the hole where the needle
  leaves the tissue, one of them commanded 1.5 mm into it, and the nodes held
  under them were squeezed between the holds and the needle ("Unresolved FEM
  tetrahedron motion"), or the stopped jaw was pushed off the needle by 2 N.
- Speed. A capsule travels at most 0.11 of its radius per interval (the second
  jaw as it comes free at the end of the approach, 17.8 s), and at most 0.065
  while it is inside the tissue; without the contact 0.055. The procedure never
  comes near one radius per interval, the speed above which this law does not
  stop a tool at the surface.
- 6.12 s. The tip of the second jaw of the needle driver meets the pad on the
  held border of the simulated region. The tissue there does not move; the jaw
  is stopped, at most 0.065 mm inside (up to 1.70 mm in this part of the
  procedure without the contact). From 6.5 s it also reaches nodes that can
  move: one to six are placed and held, the dimple reaches 0.51 mm. The force
  on the tool rises to 0.79 N at 7.0 s and presses the jaw onto the needle (jaw
  normal force 0.80 and 0.88 N against 0.38 N without the contact).
- 7.0 to 8.2 s, the first lift. The force runs out while the held nodes follow
  the jaw back; no force is left once no node is held.
- 12.8 to 14.3 s, lowering and release. The jaw meets the border again (0.80 N
  at 13.8 s, at most 0.061 mm inside); the opening jaws press on it with up to
  1.43 N, at most 0.165 mm inside (limit of the acceptance at the release
  0.25 mm; 2.75 mm without the contact). The tissue under the stopped jaw is
  held down and the needle with it: its tip stands 2.9 to 3.7 mm above the
  rest surface when the jaws let go.
- 14.3 to 14.7 s, the retreat. For 0.28 s the tool has a force on it
  (0.98 N falling to 0) while no node is held: it stands just clear of the
  held border (13 um at the largest force), where no node can be held, and
  follows its command away.
  The tissue comes up and the needle with it (its tip moves at up to
  22 mm/s); from 15.3 s it rests with the tip 5.1 to 5.2 mm above the rest
  surface.
- From 14.3 s on with the way of 2026-11-17 and the pull
  ([the way to the tip and the pull](#the-way-to-the-tip-and-the-pull);
  three runs with records per physics interval, the four items
  that follow this one are records of the regrasp of before): nodes are held
  for 33 frames and force is stored for
  41 frames of the way back (1.36 N),
  while the lower jaw backs out of the tissue it was stopped on. From there to
  the end of the closing **no node is held, no force is stored and no step is
  rejected**; the tetrahedron guard rejects nothing in the whole run. In the
  pull and the hold after it no force is stored, and in these three runs no
  node is held either. Of the eight recorded runs of this pull (these, three
  of a later round and two of an independent check) one held one node for 7
  frames at the start of the pull (its first 0.15 degrees; 0.06 mm deep, no
  force stored). The tool moves away from the tissue and is 4.6 to 4.7 mm
  clear of the deformed tissue in the hold.
- 16.2 to 16.4 s, the end of the retreat. The tool stands still for 12 frames
  while the grasp at the tip is planned for the needle where it now rests:
  1.85 to 1.94 mm from the tip in these runs.
- 16.9 to 17.8 s, the approach to the needle tip. The open second jaw slides
  over tissue that can yield: up to nine nodes are placed and held and are
  carried up to 2.2 mm sideways (0.36 mm down); 0.71 N; **up to 0.155 mm
  inside** while the tool is being stopped (limit 0.14 mm; 3 to 4 frames above
  it). As the jaw comes free (17.75 s) **6 or 7 attempts of one frame are
  rejected by the tetrahedron guard** and repeated shorter (3 with the old
  regrasp): held nodes that the jaw has left behind are released one by one
  and spring back past their neighbours.
- 19.3 to 21.5 s, descend and close on the tip. **No node is held and no force
  acts on the tool**: the first jaw comes within 0.10 mm of the deformed
  tissue and stays clear of it (0.32 mm inside and 2.0 N with the old
  regrasp; 1.54 mm inside without the contact then).
- 21.5 s on, the lift by the tip and the lowering: no node held, no force on
  the tool, no rejected step; the jaws stay 0.75 mm clear. The jaws hold the
  needle with 0.38 N at least in the lift and 0.33 N at least in the
  lowering, as without the contact (with the old regrasp the stored force of
  2 N on the first jaw took the grip down to 0.00 to 0.06 N and the needle was
  lost in one run of three or four).
- Needle and stitch. Up to 6.1 s the stitch events are at the same times and
  the needle is within 0.00005 mm of a run without the contact (as two runs
  without it). From 6.1 s on the tool is pushed and the needle with it: up to
  1.3 to 1.4 mm from its path without the contact before the release. The
  stitch has 3 entries and 3 exits in eight of the nine runs (2 and 2 since
  the capsule set, measured in three whole runs with the contact) and 4 and 4 in
  one (a window run with the frame budget: the tip leaves the tissue at
  12.02 s, enters it again and leaves at 12.03 s; this extra pair is known
  from runs without the contact as well). The third entry comes at 10.99 s
  instead of 11.03 to 11.04 s.
- Rejected attempts. 6 or 7 attempts rejected by the tetrahedron guard per
  run, all in the one frame at the end of the approach (0 without the
  contact); 13 to 15 attempts rejected for the needle (5 to 6 without), in the
  first lowering, the release and the retreat, where the stopped jaw is pushed
  by its stored force; 19 to 22 refinements per run (6 without).
- **Frame time.** With the window, each run alone on the GPU, the monitor awake.
  Whole frames (no frame budget, one run): 75 intervals from one step to the
  next are over 66.7 ms, the longest 249 ms, the median 36.6 ms; without the
  contact 21 and 31 are over (two runs before the regrasp change), with the
  same median. 57 of the 75 are in the release
  and the retreat (frames 831 to 893), where the second jaw is being stopped
  on the held border and the coupling takes 12 to 13 device trials per
  interval; 8 in the first lowering; 6 at the end of the approach, among them
  the 249 ms of the frame with the rejected attempts; 4 elsewhere. With
  `--frame-budget 66.7` ([below](#frame-time-budget-default); the unit was
  the attempt), two
  runs: **0 and 2 intervals over 66.7 ms**, the longest 65.2 and 106.0 ms,
  the median 30.8 and 30.9 ms (without the contact 1 and 0, the longest 95.6
  and 62.7 ms). The 106 ms is one interval that takes that long alone (the
  host-controlled loop at an exit of the needle and a rejected attempt), the
  other is 66.7 ms with its draw. 1,084 and 1,105 of about 2,820 calls return
  with their frame unfinished: the 28.8 s of the procedure take 79.5 and
  80.3 s on the screen (70.9 s with whole frames). The plan of the grasp at
  the tip costs 1.5 to 5.0 ms in each of 12 frames. Before the regrasp change
  and before the needle interaction kept its device candidates after a
  placement, 466 and 688 intervals of two window runs were over 66.7 ms.
  The window runs with the contact, the budget and the unit `host-trial`
  (the pull instead of the lift at the tip) are in the
  [frame time budget](#frame-time-budget-default) section: three with the
  pull to the entry, one with the pull through the tissue.
- Stored force. Only the second jaw is ever stopped: 0.74 N in the arc, 0.77 N
  in the first lowering, 1.29 N at the release, 1.34 N in the retreat, 0.56 N
  in the approach; nothing on the first jaw, the wrist or the shaft.
- Holds. A held node is within 0.06 um of where it was placed at the end of
  the coupled step (0.3 um with the old regrasp).
- Captures. No CUDA graph is captured and no kernel module is loaded after
  start-up in any run (with an empty kernel cache, measured before the regrasp
  change: start-up 176 s, then none).

Compared with the first-version law (stop at 6.4 s): the jaws no longer pass
through the pad, the tissue under them is pushed and deforms where it can, and
the tool is stopped where it cannot; the procedure runs to its end.


### Coupling tolerance and trial budget (default)

This example runs the needle/tissue coupling with two settings of its own. The
insertion example, `SutureSceneConfig` and `GraspScene` keep relative tolerance
$10^{-3}$ and no trial budget, and `NeedleTissueCoupler` keeps its own defaults
(relative tolerance $2\cdot 10^{-4}$, 40 trials, no budget), so Python callers
are unaffected.

```bash
# The example's defaults, written out:
uv run --locked python -m newton_solver_suturebot.examples fem_needle_grasp \
    --coupling-relative-tolerance 1e-2 --trial-budget 12
# Both off, as in the insertion example:
uv run --locked python -m newton_solver_suturebot.examples fem_needle_grasp \
    --coupling-relative-tolerance 1e-3 --trial-budget none
```

Measured on one DGX Spark (GB10) over 870 frames (insertion, first lift,
lowering), GPU rigid backend, one scene. The path difference is that of the
needle position against the first run at $10^{-3}$, at equal simulated time:

| Relative tolerance, trial budget | Frames slower than 66.7 ms | Slower than 200 ms | Worst frame | Run time | Needle path, median / worst |
|---|---|---|---|---|---|
| $10^{-3}$, none (two runs) | 347 and 318 of about 860 | 28 and 27 | 1.5 s and 1.2 s | 72 s and 66 s | 0.24 / 0.50 mm between the two runs |
| $10^{-2}$, none | 253 | 16 | 0.8 s | 55 s | 0.06 / 0.23 mm |
| $10^{-2}$, 12 on the GPU trial loop only (prototype) | 229 | 5 | 2.0 s | 51 s | 0.12 / 0.28 mm |
| $10^{-2}$, 6 on the GPU trial loop only (prototype) | 218 | 10 | 1.9 s | 52 s | 0.47 / 1.26 mm |

`--coupling-relative-tolerance` is the relative part of the interface force
tolerance (the absolute part stays 2e-6 N). At $10^{-2}$ the needle path differs
from the path at $10^{-3}$ by less than two runs at $10^{-3}$ differ from each
other.

`--trial-budget` (2 to 159, or 0 or `none`) bounds the trials of one coupling
loop (`NeedleTissueCoupler(trial_budget=...)`). A loop that has evaluated that
many trials without converging, and has not failed, keeps its best evaluated
trial instead of iterating on to the limit of 160 trials: the trial with the
smallest force residual since the last update of the tunnel multipliers. If
that is not the last trial, its load is applied once more and that trial is
kept.

The default of 12 is a measured choice, not a derived value: of the two budgets
that were run over the 870 frames, it is the one that left the needle path
within the difference between two runs. It does not follow from the correction
of the trial loop. A GPU trial loop without a multiplier update tightens the
tissue-solver tolerance at trial 9 and starts a new Anderson history there, so
the load of trial 12 is proposed from trials 9 to 11; the best trial is carried
over that change. In the prototype run 139 of about 3600 loops were kept at the
budget and the sequence reached the release stage as the other runs did; it
recorded 4 entries and 4 exits of the tip where the runs without a budget
recorded 3 and 3 (all with the capsules of before). Whether that difference comes from the budget or from the
difference between runs is not established. A budget of 6 kept 561 loops and
moved the path to twice the difference between two runs. The two frames of
2.0 s of the prototype run held trial loops on the host that used all 160
trials; the budget applies to the host loop as well.

With the budget in both loops, the first 300 frames of the sequence (approach,
entry and most of the arc; one run): 87 of 299 frames slower than 66.7 ms, none
slower than 200 ms, worst frame 197 ms, no trial loop longer than 13 trials on
the GPU or on the host. The same frames of the runs in the table: 132 and 131
slower than 66.7 ms, 9 and 12 slower than 200 ms and worst frames of 0.55 s and
0.69 s at $10^{-3}$ without a budget; 100, 5 and 0.58 s at $10^{-2}$ without a
budget. The budget does not by itself keep every frame under 66.7 ms: the six
slowest frames that remain hold rejected steps with their retries at half the
step, steps that go to the host oracle at surface events, and trial loops of
10 to 13 trials.

The whole sequence with the budget in both loops was measured later the same
night, after the scene had got a ramp of its pose target within a frame. The
sequence then ends after 826 frames and is not comparable with the table
above. One run each at tolerance $10^{-2}$, on that state of the scene:

| Trial budget | Frames slower than 66.7 ms | Slower than 200 ms | Worst frame | Run time | Trial loops longer than 13 trials | Rejected steps |
|---|---|---|---|---|---|---|
| none | 142 of 826 | 16 | 1.64 s | 44.8 s | 44 (up to 160 trials) | 25 (2.6 s) |
| 12 | 131 of 826 | 8 | 0.35 s | 40.9 s | none | 24 (0.4 s) |

With the budget 66 of 3359 accepted steps were kept (61 GPU loops, 1 host loop,
4 loops of the host oracle), 64 of them while the tool lowers the needle after
the lift. Their residual was 1.0 mN in the median and 38 mN at most, which is
0.02 of the needle force of that frame in the median and 0.45 at most. The
needle path differs from the run without a budget by 0.0001 mm in the median
and 0.06 mm at most. Both runs record the same surface events (2 punctures,
3 entries, 3 exits with the capsules of before; the last exit one frame
earlier with the budget). The
eight frames slower than 200 ms that remain are all in the lowering: they hold
steps rejected for a tip moved beyond the step resolution with their retries
at half and a quarter of the step, loops of the host oracle, and runs of loops
of 12 and 13 trials.

A kept step is an accepted step whose interface force is the best of the
evaluated trials, not converged to the tolerances: the force that the contact
laws give at the end state of the step differs from the load the step was
computed with by the residual of that trial. In a second run of those 300
frames 66 of 1285 accepted steps were kept (39 as their last trial, 27 after
one more trial at the best load). Their residual was 0.18 mN in the median and
7.5 mN at most, which is 0.11 of the largest load entry of the step in the
median and more than that entry in 7 of the 66 steps; a converged step with a
load above 0.2 mN is within 0.004 of it in the median and 0.014 at most.
Tissue and needle are both advanced with that one load, so the momentum
exchanged between them is unaffected: the net force of the kept load on tissue
and needle together stayed below 1e-8 N, as at converged steps (1.4e-9 N),
against a needle force of 2 mN in the median. The rule only chooses which
evaluated trial is kept; it bounds no force, load or state. A nonfinite or
invalid trial, a tip moved beyond the step resolution, a diverging load and a
failed tissue-solver certificate are handled as without a budget and are never
kept: such a step is rejected and retried with a smaller step.

Surface events (puncture, reuse of an opening, entry, exit) with the budget:

- A GPU trial that asks for the host oracle is never kept. A request that ends
  the GPU trials ends them as without a budget.
- A GPU loop that searches past a surface event and uses up the budget ends
  there. If its best trial still carries a request, the load of that trial
  starts the host oracle, as the load of a converged search does, and the
  oracle decides the step; no trial of the search is accepted. Without a budget
  a search that does not converge hands over nothing.
- The host oracle is the host trial loop and has the budget too. A trial it
  keeps is accepted with the surface event of that trial, so the event is
  decided at a state that is not converged. On a small CPU scene with the
  needle pressed on the pad, a budget of 2 and the puncture threshold placed
  between the tip force of the kept trial (2.009 mN) and that of the converged
  step (1.997 mN), the kept step commits the puncture that the converged step
  does not. In the 826-frame run above 4 oracle loops were kept, 2 of them
  after a GPU search that the budget had ended; none of these kept trials
  carried a surface event, so a kept trial that commits a surface event has not
  occurred in a measured run of this example.
- A host trial in which the interaction reports a failure (for example a tip
  that crossed the surface before a puncture was authorized) is never kept;
  that loop goes on as without a budget.

The scene diagnostics report, next to `coupling_divergence_stops`:

- `coupling_trial_budget`: the setting.
- `coupling_budget_keeps`: trial loops the budget ended with a kept trial, GPU
  and host loops. The checks that follow every loop (endpoint certificate, step
  resolution, tetrahedron guard) can still refuse a kept trial, as they can a
  converged one; that loop is counted all the same.
- `coupling_budget_searches`: GPU searches past a surface event that the budget
  ended and whose load started the host oracle. They are not in
  `coupling_budget_keeps` and not in `coupling_graph_event_searches`, which
  counts the converged searches.
- `coupling_budget_residual`: largest force residual [N] of a kept trial that
  became an accepted step.
- `coupling_budget_events`: surface events committed by those steps, by kind.

`--trial-budget 0` or `none` switches the budget off and
`--coupling-relative-tolerance 1e-3` restores the tolerance of the insertion
example. Python callers opt in with
`SutureSceneConfig(example_arguments=("--coupling-relative-tolerance", "1e-2", "--trial-budget", "12"))`.


### Surface events on the device and response columns per step (default)

Three arguments of the scene decide what a coupled step of the GPU rigid backend
does at a surface event (a puncture, an entry or an exit of the needle tip) and
how many response columns it solves after the track has grown. The insertion
example, `SutureSceneConfig`, `GraspScene`, `NeedleTissueCoupler` and
`AffineGuessManager` keep all three off, so Python callers are unaffected. This
example switches all three on: the two surface-event arguments are fixed
in the example (`SURFACE_EVENTS`), and the response columns per step are its
option `--response-basis-nodes`. Until 2026-11-19 the example had the two
event arguments as options of its own; they selected the scene's way, which
the example does not use, and were taken out. The scene keeps them.

| Argument of the scene | Scene | This example | The value that is not the scene's |
|---|---|---|---|
| `--event-search` | `tip-path` | `puncture` | `puncture`: the GPU trials also continue past a contact force at the puncture threshold (`NeedleTissueCoupler(search_past_puncture=True)`) |
| `--event-oracle` | `restart` | `device-trial` | `device-trial`: the host oracle evaluates the last GPU trial as it stands instead of solving it again (`oracle_at_device_trial=True`) |
| `--response-basis-nodes` | `0` | `1` | `n`: a coupled step solves the response columns of at most `n` support nodes (`AffineGuessManager(nodes_per_attempt=n)`); `0`: every missing column in the step after the track has grown |

```bash
# The example's default, written out (the surface events are not an option of the example):
uv run --locked python -m newton_solver_suturebot.examples fem_needle_grasp --response-basis-nodes 1
# The response columns as the scene has them, every missing column in one step:
uv run --locked python -m newton_solver_suturebot.examples fem_needle_grasp --response-basis-nodes 0
```

A caller of the scene gets the scene's way of all three by naming them after
the example's arguments (a later argument replaces an earlier one), as
`tests/test_needle_event_search.py` does for its comparison on the device:
`("--event-search", "tip-path", "--event-oracle", "restart", "--response-basis-nodes", "0")`.

What they are for. With the scene's settings a GPU trial whose contact force
touches the puncture threshold ends the GPU trials, and the host oracle repeats
the search from the accepted load (4 trials at the two punctures, 8 and 12 in
the two steps after the second exit; a host trial takes 3.6 to 4.1 ms, a GPU
trial 2.1 to 2.6 ms). A GPU search that converged past a request about the tip
path is solved once more by the oracle (4.1 ms; 1.7 to 1.9 ms when the oracle
evaluates the GPU trial as it stands). The step after an entry solves the
twelve response columns of the first track sample (18 to 20 ms). The
[method](method.md#gpu-trial-pipeline) describes the two coupler options and
the bound on the columns, and what was compared for each.

Measured on one DGX Spark (GB10), GPU rigid backend, one scene, no window, 826
frames (insertion, first lift, lowering), three runs with the example's
settings alternated with three runs with the scene's settings. A frame is
action, step and observation:

| Frame [ms] | Scene's settings, three runs | Example's settings, three runs |
|---|---|---|
| 81 (first puncture) | 68.6, 67.8, 69.1 | 55.5, 55.3, 53.9 |
| 83 (first entry) | 72.1, 68.5, 71.6 | 56.4, 57.0, 54.3 |
| 84 (remaining columns of the first track sample) | 30.3, 30.5, 31.7 | 41.6, 42.5, 40.6 |
| 212 (second puncture) | 57.7, 60.1, 57.1 | 37.2, 36.1, 36.3 |
| 213 (second entry) | 57.0, 57.4, 56.9 | 37.7, 38.4, 38.2 |
| 214 (remaining columns) | 31.6, 30.3, 28.7 | 41.8, 43.3, 42.1 |
| 316 (second exit) | 118.3, 126.4, 127.3 | 74.2, 71.9, 74.4 |
| Frames slower than 66.7 ms, frames 1 to 420 | 3, 3, 3 (81, 83, 316) | 1, 1, 1 (316) |
| Frames slower than 66.7 ms, frames 421 to 825 | 8, 12, 7 | 9, 10, 9 |
| Host loops of frames 1 to 420 | 18 loops, 48 host trials, 185 to 195 ms | 15 loops, 17 oracle evaluations (2 of them host trials), 34 to 35 ms |
| Median frame | 29.6, 30.2, 30.0 | 30.0, 29.8, 30.4 |

All six runs record the same surface events in the same coupled steps through
the stitch (puncture 81.1, entry 83.1, exit 178.2, puncture 212.1, entry
213.2, exit 316.1; frame.step) and end with 3 entries and 3 exits; the third
exit falls on frames 725 and 726 in both groups (runs with the capsules of
before; with the capsule set the demo ends with 2 and 2). The needle position of the
runs with the example's settings stays within 0.08 micrometres of a run with
the scene's settings through frame 420 (two runs with the scene's settings:
0.05 micrometres). In no stage of the procedure is the median frame time
higher than with the scene's settings by more than the three runs with the
scene's settings differ among themselves (the largest increase is 0.07 ms,
while inserting). No CUDA graph is captured and no kernel module is loaded
after start-up in any of the runs, in the self-test of the whole procedure or
in a census of it (1688 frames). With `--response-basis-nodes 1` the same 20
bases are loaded with the same column counts, a basis of $k$ new nodes $k-1$
steps later (three steps after an entry). The 15 steps that only solve
columns read nothing from the device and make no array; a step that loads a
basis reads the residual records of its new columns once, as without the
option.

The mask of an opening is computed from the faces around the punctured face
in every configuration (0.3 to 0.5 ms per puncture or exit; 8 to 11 ms
before). The scene's settings in the table include that: before it, frames 81
and 212 took 71 to 87 ms and 64 to 71 ms.

**Not fixed by these options.**

- Frame 316 stays over 66.7 ms (71.9 to 74.4 ms without a window). No host
  trial is left in it. In the step of the exit the tip force is 0; in the two
  steps after it the tip, now outside, carries 2.48 and 2.61 mN, and these
  steps take 8 GPU trials (converged) and 12 (kept at the trial budget, as
  the same step is with the scene's settings): 20 trials, 42 to 45 ms. That
  the tip is pressing on the rim of its own exit opening there is inferred
  from that force; the contact was not located.
- The lowering keeps 7 to 12 frames over 66.7 ms in every run of both groups
  (frames 711 to 781; the slowest: 244, 315 and 200 ms with the scene's
  settings, 129, 168 and 255 ms with the example's). They hold steps rejected
  for a tip moved beyond the step resolution with their retries at half the
  step, trial loops that end at the trial budget, and the steps at the third
  exit in which the host oracle does not converge: 54 to 81 ms for such a
  step with the scene's settings (two of them in one run), 27, 82 and 92 ms
  with the example's. The slowest of these (92 ms; frame 726 at 168 ms) used
  the trial budget twice, 13 GPU trials and then 13 evaluations of the
  oracle, where the scene's settings spend 2 GPU trials and 13 host trials.
- Frames 76 to 80 (first contact, before the puncture) take 52 to 65 ms
  without a window in both groups (frame 77: 60 to 68 ms in eight later
  runs). These options do not touch them. With the window (one run with the
  example's settings) their intervals were 60.1, 67.6, 65.9, 60.5 and
  60.0 ms, so frame 77 is over 66.7 ms there. In that run frames 81 and 83
  took 60.8 and 61.5 ms, frame 316 81.1 ms and frame 317 67.1 ms, and 15
  intervals of the lowering were over 66.7 ms.

`NeedleTissueCoupler.coupling_oracle_device_trials` counts the oracle loops
that started at a GPU trial and `coupling_oracle_device_accepts` those the
oracle accepted in that evaluation (31 to 33 and 29 to 32 in the runs above);
`AffineGuessManager.deferred_builds` counts the steps that solved columns
without loading a basis (15). Python callers opt in with
`SutureSceneConfig(example_arguments=("--event-search", "puncture", "--event-oracle", "device-trial", "--response-basis-nodes", "1"))`.


### Frame time budget (default)

```bash
# The example's default, written out: plan for a picture of the window no later than 66.7 ms
# after the one before (15 frames/s); a step may also return between two host-controlled trials:
uv run --locked python -m newton_solver_suturebot.examples fem_needle_grasp --frame-budget 66.7
# Whole frames, as the example ran before the budget was its default:
uv run --locked python -m newton_solver_suturebot.examples fem_needle_grasp --frame-budget 0
```

The budget is a default of this example since 2026-11-17 (`FRAME_BUDGET_MS`
in the example): no picture of the window is to come
later than 66.7 ms after the one before. It is used where it was used as an
option, with one scene in the window of the OpenGL viewer; every other run
keeps whole frames (the last paragraphs of this section). `LocalSutureScene`,
`ParallelSutureScenes` and `SutureSceneConfig` have no budget of their own.

A frame of this example is four coupled substeps, and where the needle enters
or leaves the tissue a frame takes several times as long as its neighbours:
the window stands still for that frame. `--frame-budget MS` gives the
example's `step()` a time budget: the longest time, in milliseconds, from the
start of one step to the start of the next that the example plans for. A
substep is an accepted coupled step with the attempts that were rejected
before it; a rejected attempt is tried again at half the time step. A step
runs one attempt in any case. It starts another one, the next substep or the
retry of a rejected attempt, only if a substep that uses up its trials still
fits, together with the drawing that follows. Otherwise it returns with the
frame unfinished, the window shows the state reached, and the next step goes
on with the same frame at its next attempt: simulated time falls behind by
that call instead of the window standing still. The accepted states are those
of whole frames, the same attempts in the same order, and the controller acts
once per frame on the state after the frame before. On the CPU they are the
same bit for bit: with rejected attempts, with a retry in a later step than
its rejected attempt, and with a reset or a close between the two. The default
is `66.7`; `0` runs whole frames.

**The budget is a plan, not a guarantee.** A trial loop on the device is never
interrupted, and with the unit `attempt` no attempt is, so one piece can take
longer than the whole budget. Measured before 2026-11-17, when a step could
return only between two attempts (at first, two substeps), with the capsules
of then: where the needle left the tissue during the
lowering (frames 724 to 726 of the demo of then), the trial loop on
the device handed over to the host-controlled loop, which then ran 7 to 13
trials: that one attempt took 65 to 77 ms. It did so in 10 of 29 runs, with
or without the option. From the start of the step that ran it to the start
of the next one it was then 67 to 96 ms, with the observation and the drawing
after it and whatever the step ran before it. In about one run of three, with
or without a window, one frame there stayed over 66.7 ms. The unit
`host-trial` (below), the default, lets a step return between two trials of
such a host-controlled loop. With the capsule set of today the procedure has
2 entries and 2 exits and that exit is gone; the longest pieces of the window
runs with the contact and the budget (below) are device loops of 13 trials
where the open jaw stands on the tissue (the longest of a run 43 to 49 ms),
and no unit splits those. The window run of the procedure as it is now has
one interval over 66.7 ms (67.1 ms, below): a step whose second attempt ran
such a loop within the bound, and whose observation then took 4.7 ms longer
than the last one, which is what a step planned with in that run. The one
window run made since with the worst-case plan (below) has no interval over
60 ms and takes 27 % longer to the end.

What a step plans with is measured by the run itself on the host clock, with
no read from the device. Every term is the worst case that the run has
measured since the last reset. (In every window run of this section the
first two were the last value and the third was made of two means,
`outside + loops * (trial budget + 1) * time per trial`: of the device loops
that run 12 or 13 trials, up to five per run took longer than 13 mean trials,
by up to 11 ms.)

- the drawing: as long as the longest `render()`;
- the observation after the scene has stepped: as long as the longest one. It
  waits for nothing that the step issued on the device (since 2026-11-19,
  "The two times of the plan that were not explained" below);
- one more attempt:
  `outside + loops * max((trial budget + 1) * time per trial, loop)`.
  `outside` is the longest time a substep has taken outside its trial loops,
  with the wait for what it issued on the device (the step of the thread),
  `loop` the longest time of a trial loop (of a substep with several loops,
  their mean). The time per trial is the time spent in the coupling's trial
  loops divided by the trials evaluated, a mean over the run: a loop is taken
  as no less than that many mean trials, which stands for a loop that uses up
  its trials before one has been measured. `loops` is the number of trial
  loops the substep measured last ran (one; one more for a handover to the
  host-controlled loop and for every rejected attempt). `SubstepBound` in
  `_src/utils/frame_budget.py`.
- with the unit `host-trial`, one more host-controlled trial: the longest such
  trial plus the longest end of an attempt (its checks, its commit, the work
  after an acceptance and, in the last substep, the end of the frame) that the
  run has measured since the last reset (`HostTrialBudget` in the same file).
- the controller: nothing. The budget counts from the start of the step,
  before the controller acts, so its time is inside the step it acts in.

So a step that is `q` into its time starts one more attempt only if `q`, that
attempt, the observation and the drawing, each as long as the longest
measured, end within the budget. The picture after such a step is late only
if one of these parts takes longer than it ever has in the run (a trial loop:
longer than `loop` above, which is the mean of the loops where a substep ran
several, so not always the longest loop measured), or if the attempt runs
more loops than the substep before it (and by what no term covers, the
return from the scene to its observation and the run loop around the
drawing: 0.06 to 0.09 ms in the median and at most 0.6 ms in the recorded
runs). The first attempt of a step is not asked (a step makes progress): a
single attempt that is too long for the budget is late under any plan.

A maximum does not come down. One part that took longer than ever before, for
whatever reason, stays in the plan until the scene is reset (the Reset of the
window starts every measurement again). It enters with its own excess: a loop
that stalls raises the plan by what it took longer than the longest loop
before it, not by the trial limit times its time per trial (a loop of one
trial takes 3.6 to 4.3 ms and up to 11 ms, a loop of 13 trials 2.2 to 3.3 ms
per trial). What it costs is steps, not time between two pictures: fewer
attempts start as a second attempt of a step, at worst none, and every step
still runs one. Nothing is forgotten after a time: a worst case that the plan
forgets can come back as a late picture.

What in this plan is a choice, and not a maximum that the run has measured, is
listed in one place, the module text of `_src/utils/frame_budget.py`: the mean
trial that stands for a loop of the trial limit until one has been measured,
the number of loops taken for the next substep, and that no maximum is
forgotten before a reset.

**What the worst case costs: replayed here, measured in one window run
below.** The replay was made before any window run with this plan. Replayed
on the attempts that five window runs recorded
(the run of the procedure as it is now and four of the runs of before the
thread tracks, both below; the same attempts in the same order, packed into
steps by the plan; where a step returns at a place at which the recorded run
went on, the observation and the drawing of the nearest recorded return),
the worst case leaves no room for a second attempt in any step of any run:
the bound of one more attempt (59 to 69 ms at the end of a run: 18 to 20 ms
outside the loops, which is the run's first attempt, whose end takes 16 to
17 ms in every run, and 39 to 49 ms for a loop), the longest observation
(3 to 12 ms) and the longest drawing (7 to 13 ms) are 69 to 94 ms together.
Every step then runs one attempt:

| recorded window run | steps, recorded / replayed | intervals over 66.7 ms | intervals over 60 ms | to the end, recorded / predicted |
|---|---|---|---|---|
| the procedure as it is now (`w_def_1950`) | 4272 / 7827 | 1 / **0** | 1 / 0 | 100.7 / 126.5 s |
| before the thread tracks, swept contact, run 1 | 5073 / 7020 | 0 / 0 | 15 / 15 | 111.0 / 124.3 s |
| the same, run 2 | 4642 / 7025 | 0 / 0 | 4 / 4 | 103.5 / 121.2 s |
| the same, run 3 | 4636 / 7021 | 0 / 0 | 2 / 2 | 100.4 / 117.2 s |
| before the thread tracks, `--no-tool-tissue-contact` | 3309 / 7001 | 0 / 0 | 0 / 0 | 70.0 / 92.5 s |

The one late picture is gone (its second attempt is refused, by 27 ms), and
the price is predicted at 12 to 32 % more time on the screen. With the
longest observation and drawing recorded so far at every such place instead
of the nearest ones, the prediction is 133 to 152 s with the contact and
100 s without. The intervals over 60 ms that stay are steps of one attempt
whose device loop ran 12 or 13 trials: no plan changes those. The same
replay gives the recorded steps back under the plan of the recorded runs
(the same returns but one in two of the five runs). Tools and logs:
`/tmp/suturebot-gpu-work/grasp_fix/budget/` (`notes.md`, `tools/prove.py`),
a work folder outside this repository. The window run made with this plan
afterwards (`w_fix_1950`, below) came out as the first row predicts: 7820
steps, no interval over 60 ms, 128.3 s to the end.

The budget needs the
[trial budget](#coupling-tolerance-and-trial-budget-default): without one a
loop may run to its limit of 160 trials and every step runs a single attempt.

**The unit: `host-trial`, or `attempt` where that cannot be used.** The unit
is the smallest part of a frame after which a step with a budget may return.
The example takes it by itself (`Example._frame_unit`); until 2026-11-19 it
was an option, `--frame-budget-unit`, whose other value `attempt` selected the
earlier unit. That option was taken out: `host-trial` returns between two
attempts as well, and the one window run with `attempt` (the table below)
showed no interval that it avoids. With `attempt` the part is one attempt of
a coupled step, as described above. With
`host-trial` a step may also return inside an attempt, at one kind of place:
between two trials of a host-controlled loop (the loop that the host controls
where the needle meets, enters or leaves the tissue surface). Before every
such trial the coupler asks whether the bound above still ends by the deadline
of the step. If not, the step returns with the attempt open, the window shows
the state accepted before that attempt, and the next step starts with that
trial, whatever its deadline says, so every step makes progress. An attempt
still starts only by the rule of the unit `attempt`, and the first
host-controlled trial of a run is always left for the next step: nothing is
measured yet. A trial loop on the device is one piece under either unit.

- The accepted states are those of whole frames. On the CPU they are the same
  bit for bit with an attempt left before every one of its host-controlled
  trials, without the tool/tissue contact and with the swept one
  (`tests/test_host_trial_budget.py`).
- Where `host-trial` is not used: it needs a frame budget in use (the last
  paragraphs of this section) and the GPU rigid backend (on the native backend
  the rigid data hold a trial while an attempt is open, and the observation
  would read the jaw contact force and the native clock of that trial). The
  unit is then the attempt, and with a budget in use the example prints why. In a paused window a single step
  finishes an open attempt and its frame.

**Measured in the window, the procedure as it is now** (the pull through the
tissue with the thread tracks; done at 30.9 s of simulated time). One run of
the default, with no option for the contact, the budget or the thread, on
2026-11-17, 17:24:53 to 17:26:47 (one DGX Spark, alone on the GPU, the
monitor awake, machine load 1.9 at its start and 1.2 at its end; run
`w_def_1950` of 1950 frames on the tree that made the budget the default,
`/tmp/suturebot-gpu-work/contact_default2/`, a work folder outside this
repository: `gpu/logs`, `gpu/runs` and `notes.md` there). It is the only
window run of this procedure with the plan of then (the last observation,
the last drawing, two means); the run with the worst-case plan follows it.
Nothing was run without the contact, without
the budget or with the unit `attempt`, and one run is not a rate. It printed
"Frame budget 66.7 ms: ..." and "Frame budget unit host-trial: ...". The
interval is the time from the start of one step to the start of the next:

| stage of the procedure | frames | steps | time on the screen | interval median / 99 % / longest | intervals over 66.7 ms |
|---|---|---|---|---|---|
| the needle goes in (arc) | 421 | 957 | 22.1 s | 23.7 / 35.6 / 38.6 ms | 0 |
| lift, hold, lowering | 406 | 1082 | 23.9 s | 21.3 / 35.9 / 40.1 ms | 0 |
| release (the jaws open) | 31 | 140 | 5.5 s | 42.0 / 54.9 / 56.3 ms | 0 |
| retreat | 206 | 512 | 14.3 s | 26.3 / 53.5 / **67.1 ms** | **1** |
| the way over, the descent, the closing | 356 | 700 | 16.2 s | 25.4 / 33.6 / 41.3 ms | 0 |
| **the pull through the tissue** | 400 | 751 | 17.9 s | 27.5 / 34.7 / 36.0 ms | 0 |
| hold after the pull, done | 129 | 129 | 3.7 s | 28.8 / 30.0 / 30.2 ms | 0 |
| the whole run | 1950 | 4272 | 103.6 s | 23.8 / 48.6 / **67.1 ms** | **1** |

(A stage row counts the steps that began in the stage and the frames they
completed; the last step of the run has no interval after it.)

- One picture came later than 66.7 ms after the one before: 67.1 ms, 0.4 ms
  over, at frame 903 in the retreat, where the open jaw stands on the tissue.
  Its step ran two attempts, neither rejected: a trial loop on the device of
  5 trials (10.1 ms) and then, because one more attempt still fitted by the
  plan (by 0.7 ms), a second one whose loop ran 13 trials in 40.8 ms. That
  attempt took 42.3 ms, less than the 46.1 ms the plan had for it. The
  observation after the step took 5.8 ms, where the plan had the 1.2 ms of
  the step before; with the drawing (4.7 ms) the interval came to 67.1 ms.
  In this run the observation takes 1.2 ms after a step that completes its
  frame (2.0 ms at most) and 3.5 ms in the median, 8.4 ms at most, after one
  that does not; the plan of that run took it as long as the last one, and
  in 348 of the 4,271 steps it was over 4 ms longer than that. Why it differs
  was not looked into then (it is the wait for the step of the thread: "The
  two times of the plan that were not explained" below). It
  is the only interval over 60 ms of the run (the next are 59.7, 59.0 and
  57.9 ms, in the retreat, each with an observation 4.5 to 6 ms longer than
  the last one; 34 are over 50 ms, all in the release and the retreat). So
  the aim, no picture later than 66.7 ms, is missed once in the 4,271
  intervals of this run.
- The pull through the tissue, never timed before, keeps the budget with a
  margin: 400 frames in 751 steps and 17.9 s on the screen, no interval over
  36.0 ms. After it a frame takes one step (28.8 ms).
- The procedure is done at frame 1851, 100.7 s on the screen after the first
  step for its 30.9 s; 2,322 steps return with their frame unfinished. It
  ends with 2 entries and 2 exits, the pull complete and the tail 2.0 mm past
  the exit marker; 7 attempts are rejected and taken again at a shorter
  step, the tetrahedron guard rejects no tissue step, nothing is captured or
  loaded after start-up. The 33 device loops of 13 trials take 37.6 ms in
  the median and 42.9 ms at most.
- The unit: one step returned inside an attempt, at the run's first
  host-controlled trial (21 such trials admitted, 1 refused), as in the runs
  below.

**Measured in the window with the worst-case plan** (the plan described
above; the same procedure, with the spare thread laid on the pad). One run
of the default, with no option, on 2026-11-17, 20:50:17 to 20:52:43 (the
same DGX Spark, alone on the GPU, the monitor woken for it, machine load 7.2
at its start and 4.8 at its end from other work on the CPU, where the run
above had 1.9 and 1.2; run `w_fix_1950` of 1950 frames on the tree that
brought the plan in, `/tmp/suturebot-gpu-work/grasp_fix/integ/`, a work
folder outside this repository: `gpu/logs`, `gpu/runs` and `notes.md`
there). One run is not a rate. In brackets, the run above (`w_def_1950`):

| stage of the procedure | frames | steps | time on the screen | interval median / 99 % / longest | intervals over 66.7 ms |
|---|---|---|---|---|---|
| the needle goes in (arc) | 421 | 1688 (957) | 28.3 s (22.1) | 16.1 / 26.5 / 29.5 ms | 0 |
| lift, hold, lowering | 406 | 1624 (1082) | 28.7 s (23.9) | 17.2 / 25.3 / 39.9 ms | 0 |
| release (the jaws open) | 31 | 140 (140) | 5.4 s (5.5) | 44.0 / 57.6 / 57.9 ms | 0 |
| retreat | 202 (206) | 811 (512) | 16.8 s (14.3) | 17.3 / 51.4 / 56.4 ms | 0 (1) |
| the way over, the descent, the closing | 352 (356) | 1408 (700) | 23.0 s (16.2) | 16.6 / 22.7 / 46.8 ms | 0 |
| **the pull through the tissue** | 391 (400) | 1564 (751) | 24.4 s (17.9) | 15.7 / 20.5 / 25.2 ms | 0 |
| hold after the pull, done | 146 (129) | 584 (129) | 7.6 s (3.7) | 13.0 / 14.4 / 15.1 ms | 0 |
| the whole run | 1950 | 7820 (4272) | 134.3 s (103.6) | 16.1 / 46.4 / **57.9 ms** (23.8 / 48.6 / 67.1) | **0** (1) |

- No picture came later than 66.7 ms after the one before, and none later
  than 60 ms: the longest interval is 57.9 ms, in the release, a step of one
  attempt whose trial loop on the device ran 13 trials in 41.9 ms. 51
  intervals are over 50 ms, all in the release and the retreat (34 in the run
  above); the ten over 55 ms are each one attempt with a device loop of 12 or
  13 trials (35.3 to 42.7 ms).
- The price is the one the replay predicted. **No step of the run started a
  second attempt**: after the run's first attempt the bound of one more
  attempt never was under 60.6 ms (at the end 17.9 ms outside the loops,
  which is the end of that first attempt and stays the longest of the run,
  and 42.7 ms for the longest loop), and with the longest observation
  (11.2 ms) and drawing (8.6 ms) the plan stood at 70.5 ms at the least
  after any step (80.4 ms at the end), over the budget. So every step ran
  one attempt, four steps for
  a frame without a rejected attempt: 5,870 of the 7,820 steps return with
  their frame unfinished, and a frame of the hold after the pull takes four
  steps of 13 ms where it took one step of 28.8 ms.
- The procedure is done at frame 1834 (30.6 s of simulated time), 128.3 s on
  the screen after the first step: 70.0 ms per frame against 54.4 ms in the
  run above, 27 % more time to the end (predicted 126.5 s; per frame 2.3 %
  more than predicted, at the higher machine load). Pictures come every
  16.1 ms in the median while simulated time advances at 0.24 of the clock
  (0.31 above).
- The observation after a step that leaves its frame unfinished takes 2.8 ms
  in the median and 11.2 ms at most (1.2 and 3.1 ms after a complete frame);
  three steps in four now pay it.
- The unit: one step returned inside an attempt; 21 host-controlled trials
  admitted, 1 refused, each admitted one inside its deadline (by 19 ms at
  the least). The run ends with 2 entries and 2 exits, the pull complete and
  the tail 2.1 mm past the exit marker; 5 attempts are rejected, the
  tetrahedron guard rejects no tissue step, nothing is captured or loaded
  after start-up. The window run does not read the thread.

**The two times of the plan that were not explained (2026-11-19).** Two
window runs of the default, 1950 frames each, alone on the GPU, the monitor
woken, machine load 0.3 to 1.1: `w_base_1` (17:19:30 to 17:21:53, commit
`640000d`) and `w_tree_1` (17:46:46 to 17:49:09, with the two changes below);
records and tools in `/tmp/suturebot-gpu-work/pace_cheap/build/`, a work
folder outside this repository. One run each is not a rate.

- The observation after a step that leaves its frame unfinished waited for
  the step of the thread, which an accepted substep issues on the device
  without waiting for it (1.5 ms in the median, 7.0 ms at most, timed as a
  wait placed right after that step in a third run, 400 frames without a
  window; a step that completes its frame waits for it in the tip sample).
  The work is real, but the plan left it free twice: in `outside` and in the
  longest observation. A step with a budget now waits for the device before
  it measures a substep (`_wait_for_device` in `simulate` of the insertion
  example; no data crosses and nothing is launched: the wait moves from the
  observation into the step), and the observation takes 1.2 ms in the median
  and 2.2 ms at most after either kind of step (before: 2.7 and 8.1 ms after
  a step that leaves its frame unfinished, 1.2 and 2.3 ms after one that
  completes it).
- The end of the run's first attempt was the swept test of the needle tip:
  the first step after a reset has no committed copy of the tissue, and its
  complete host test gathered all 110,956 faces of the reference surface
  (13.4 ms), of which only those with an active vertex move. The bounds of
  the others are now taken once, when the interaction is built
  (`_swept_surface_bounds`; the same faces selected, the same crossings):
  2.3 ms, and the first substep is no longer the longest of the run.

| | `w_base_1` | `w_tree_1` |
|---|---|---|
| steps for the 1950 frames; with a second attempt | 7829; 0 | 7722; 103, all in frames 20 to 87 |
| intervals over 66.7 ms; longest; median | 0; 57.8 ms; 15.5 ms | 0; 56.8 ms; 15.8 ms |
| done at frame; after; the 1950 frames | 1838; 124.1 s; 129.7 s | 1852; 125.2 s; 130.3 s |
| at the end: `outside` + `loop`; longest observation; longest drawing | 17.5 + 43.0; 8.1; 7.4 ms | 12.6 + 43.5; 2.2; 6.4 ms |
| the plan after a step, least / at the end | 70.5 / 76.0 ms | 56.8 / 64.7 ms |

The pace did not change: the plan stands 11 to 14 ms lower, but a step's
first attempt takes 10 ms in the median and the loop term alone is 39 to
43.5 ms. A replay of the plan on the record of `w_base_1` had predicted that
(7,829 steps, 124.1 s). The pace needs a device loop that a step can leave.

**Measured in the window, the procedure of before the thread tracks** (the
pull ended at the needle's entry; that procedure has been removed). The shared round on the GPU of 2026-11-17, 15:25
to 15:57, with the options that are the defaults now written out (one DGX
Spark; seven window runs of 1750 frames, each alone on the
GPU, the monitor awake, machine load 1.4 to 7.7 from other work on the CPU;
notes and logs: `/tmp/suturebot-gpu-work/round2/notes.md` and `r2_1117/logs`
there, a work folder outside this repository; the plan of then, as in the
run above). "To the end" is the time on
the screen until that procedure is done (26.2 to 26.7 s of simulated time):

| window run, with the pull to the entry | runs | intervals over 66.7 ms, per run | longest interval, per run | median interval | steps for the 1750 frames | to the end | the 1750 frames |
|---|---|---|---|---|---|---|---|
| swept contact, 66.7 ms, `host-trial` (the defaults of today) | 3 | **0, 0, 0** | 65.4, 60.9, 60.4 ms | 22.9, 23.3, 22.8 ms | 5073, 4642, 4636 | 111.0, 103.5, 100.4 s | 119.3, 109.9, 107.0 s |
| `--frame-budget-unit attempt` (an option until 2026-11-19) | 1 | 0 | 59.1 ms | 22.4 ms | 4805 | 103.7 s | 109.9 s |
| `--frame-budget 0` (whole frames) | 1 | 111 | 470.3 ms | 41.0 ms | 1750 | 75.6 s | 81.3 s |
| `--no-tool-tissue-contact` | 1 | 0 | 51.5 ms | 22.9 ms | 3309 | 70.0 s | 75.4 s |
| `--no-tool-tissue-contact --frame-budget 0` (the example of before) | 1 | 9 | 120.8 ms | 35.6 ms | 1750 | 57.9 s | 63.2 s |

In all seven runs that procedure completes inside the 1750 frames, with 2
entries and 2 exits; nothing is captured or loaded after start-up, and the
tetrahedron guard rejects no tissue step. What the table shows:

- With the contact, the budget and the unit `host-trial` no picture came later
  than 66.7 ms after the one before,
  in three of three runs. The margin is small: the longest interval, 65.4 ms,
  is 1.3 ms under the limit, in the run with the highest machine load. The
  intervals over 60 ms (15, 4 and 2 per run) are all in frames 834 to 889, the
  release and the first part of the retreat, where the open jaw stands on the
  tissue: each is a step that ran one attempt whose trial loop on the device
  took 13 trials (once 12) and 42.3 to 49.1 ms. Neither unit splits such a
  loop.
- The price is time on the screen: the 26 s of that procedure take 100 to 111 s
  instead of 76 s with the contact (70 s instead of 58 s without it), because
  2,886 to 3,323 steps return with their frame unfinished. The release, 30
  frames, takes 5.8 to 6.0 s on the screen.
- Without a budget the contact brings 111 frames over 66.7 ms (9 without the
  contact): 28 of the 30 frames of the release (the longest 470 ms), 44 in the
  retreat, 17 while the needle goes in, 18 in the lift and its lowering, 4
  elsewhere; each is a whole frame of four or more attempts.
- The unit made no measurable difference in these runs: with the unit
  `attempt` the one run has no interval over 66.7 ms either. The long
  host-controlled loop that the unit is for did not occur; every run with the
  unit `host-trial` returned inside an attempt once, at its first
  host-controlled trial (21 such trials admitted and 1 refused per run).

The measurements that follow are older: the budget as an option, the unit
`attempt` (and, before it, the substep), no tool/tissue contact, the capsules
and the procedure of before.

Measured on one DGX Spark (GB10), whole demo of 1700 frames; the interval is
the time from the start of one step to the start of the next. "Substep as the
unit" is the option as it first landed, when a step could return only between
two substeps: twelve runs of two independent sets. "Attempt as the unit" is
the unit `attempt` of today: three runs. The runs without the option are those
of all three sets. Some runs are from before the
[surface-event defaults](#surface-events-on-the-device-and-response-columns-per-step-default)
above, most from after them:

| | runs | intervals over 66.7 ms, per run | longest interval, per run | steps with the frame unfinished | time of the 1700 frames |
|---|---|---|---|---|---|
| window, off | 7 | 21 to 31 | 166 to 320 ms | 0 | 63.3 to 67.2 s |
| window, 66.7, substep as the unit | 6 | 1, 0, 1, 0, 1, 0 | 95.6, 62.7, 81.0, 57.5, 84.2, 62.0 ms | 1062 to 1309 | 73.8 to 79.1 s |
| window, 66.7, attempt as the unit | 3 | 0, 0, 1 | 57.8, 57.7, 82.3 ms | 997 to 1145 | 72.0 to 76.3 s |
| no window, off | 7 | 14 to 23 | 145 to 292 ms | 0 | 49.4 to 53.1 s |
| no window, 66.7 forced on, substep as the unit | 6 | 0, 0, 0, 1, 0, 0 | 62.0, 56.5, 63.1, 66.8, 57.6, 61.1 ms | 93 to 113 | 49.5 to 51.9 s |

The four intervals over 66.7 ms of the twelve runs with the substep as the
unit are all at frames 724 to 726. Three are the single attempt with the
host-controlled loop: 66.8 ms (an attempt of 64.6 ms alone in its step), 84.2 ms
(74.6 ms alone in its step) and 95.6 ms (74.4 ms after a substep of 11 ms). The
fourth, 81.0 ms, was a rejected attempt and its retry of 13 trials (52 ms
together) that started 18 ms into the step. A step now asks before a retry as
it asks before a substep. In the three runs with the attempt as the unit, 14
attempts were rejected: after 2 of them the step returned, after 12 the
retry started in the same step because the bound still fit, and the longest
interval with a rejected attempt and its retry was 57.3 ms. The one interval
over 66.7 ms of these runs (82.3 ms, frame 725) is the single attempt with
the host-controlled loop (72.3 ms, alone in its step). A retry that starts
because the bound fits can still take longer than the bound.

With the window the budget costs time (these older runs; today's numbers are
in the two tables above): a drawing takes 5 ms, the bound for one
more attempt is 39 ms, and three ordinary substeps take 22 ms, so the fourth
substep of most frames waits for the next step. Between 953 and 1257 of the
1700 frames then take two steps or more and the demo takes 17 % longer (76.9
against 65.7 s and 76.0 against 65.0 s with the substep as the unit, 73.9
against 63.3 s with the attempt as the unit; means of three runs with the
option against three, three and one without). A larger budget costs less
(replayed from the recorded substeps of one run, not measured):
72 ms about 7 % more time, 80 ms about 1 %. Without a window the same budget
splits one frame in twenty and costs nothing measurable. Stitch (3 entries,
3 exits with the capsules of before) and needle path are within the spread
between two runs without the option; nothing is captured or loaded after start-up.

With the option on, fewer trial loops are kept at the trial budget (21.2
against 25.3 per run over 24 runs, one-sided permutation p = 0.0007); the
cause is open.

The budget is for one scene in the window of the OpenGL viewer, where the
time between two steps is the time between two pictures. Everything else
keeps whole frames: the self-test
(`--test`), a viewer that ends the run after a number of frames (`null`,
`usd`, `gl --headless`), the Rerun and Viser viewers (they would record or
send a step that returns with its frame unfinished as a frame), the RTX
window (its pictures come from a renderer that runs beside the steps; the
budget was never run with it) and several scenes (`--num-envs` above 1). In a
paused window a single step (the `.` key or the Step button) has no budget:
it finishes the frame in progress, or runs a whole frame.

What a run prints about it. In the window one line names the budget in use
and the option that turns it off ("Frame budget 66.7 ms: ...
(--frame-budget 0 runs whole frames)"), and a second line the unit, or why
`host-trial` is not used there (the native rigid backend). A run that keeps whole frames prints nothing about the default
budget and nothing about the unit: it did not ask for them. Only a budget
other than the default, for example `--frame-budget 50`, says why it is not
used ("Frame budget not used: ...").

`LocalSutureScene.step` takes the deadline as an optional argument
(`deadline=`, a `time.perf_counter()` time; `frame_unfinished` tells whether
the frame is complete, and the step that goes on with it passes
`actions=None`) and the unit as another (`unit="host-trial"`; the default
there is `"attempt"`, and on the native rigid backend `"host-trial"` is
refused unless the caller passes `observe_native_trials=True`). Without a
deadline, and in `ParallelSutureScenes`, a step is always a whole frame.
`SutureSceneConfig` has no setting for either.


### GPU rigid backend (default)

```bash
uv run --locked python -m newton_solver_suturebot.examples fem_needle_grasp
# Native MuJoCo on the CPU instead, for a machine without CUDA:
uv run --locked python -m newton_solver_suturebot.examples fem_needle_grasp --rigid-backend native
```

`--rigid-backend mujoco-warp`, the example's default, integrates the tools and
the needle with MuJoCo Warp on the GPU instead of native MuJoCo on the CPU.
`SutureSceneConfig` and `GraspScene` keep `native` as their own default, so
Python callers are unaffected. The physical model, the four rigid microsteps per coupled interval,
the coupling tolerances, the iteration budgets and every acceptance check are
the same; the rigid solve runs in single precision with tolerance 1e-6 where
the native backend uses double precision and 1e-12. It needs CUDA with
conditional graphs and MuJoCo Warp 3.17.0, which
`uv sync --locked --extra examples` installs. Python callers set
`GraspScene.rigid_backend = "mujoco-warp"`.

With this backend all coupling trials of one interval run as one conditional
CUDA graph: tissue solve, rigid microsteps, contact forces and the correction
of the interface load, with no host round trip between trials. The
[method](method.md#gpu-trial-pipeline) describes it. Two environment variables
select search variants for comparisons; neither changes what is accepted:

- `SUTUREBOT_EXPERIMENTAL_MJW_COUPLING`: `device-anderson` (default in the
  benchmarks), `device-aitken` or `current`.
- `SUTUREBOT_EXPERIMENTAL_MJW_CONTACT_GUESS`: `device` (default) or `zero`
  for the start of the contact-preconditioner solve.

Report keys for this backend: `rigid_backend`, `gpu_coupling_policy`,
`gpu_rigid_diagnostics` (accepted clock, effective solver options,
`gpu_guard_rejections`), `coupling_graph_captures`, `coupling_graph_launches`,
`coupling_graph_fallbacks` with `coupling_graph_fallback_reasons`,
`coupling_graph_event_searches` and `coupling_graph_event_restarts`.

A surface event (puncture, entry, exit, reuse of an opening) is still decided
by the host oracle. Such an interval is counted as a fallback and is slower
than the others. A rigid trial whose coordinates leave MuJoCo's valid range is
rejected and the interval is retried with a smaller step, as the native
backend does after a MuJoCo instability warning. Around surface events single
frames take several hundred milliseconds, and without the trial budget the
host oracle occasionally needs several seconds for one interval.

Measured on one DGX Spark (GB10) with the 360-frame insertion, `--lift-height 0`,
one scene, relative tolerance $10^{-3}$ and no trial budget (this example's
settings before the [defaults above](#coupling-tolerance-and-trial-budget-default);
not measured again with them): about 10 simulated frames per second while the
needle is in the tissue (median frame 81 ms) and about 42 before contact. The
native backend
reaches about 13 (65 ms) and 47. Neither is real time during insertion, and
the GPU backend is slower than the native one there. Results differ from run
to run, because MuJoCo Warp sums contact rows in an order that is not fixed
and the surface events then fall into different intervals.

The supported CUDA physical-grasp configuration also reuses a small approximate
FEM response basis to propose better starting guesses for its ordinary tissue
and contact solves. Each guess must reduce the residual of the current full
operator; the original CG solves and physical acceptance checks still run.
This is enabled by `GraspScene.response_basis_guesses`, alongside
`held_fem_start`, and needs only the existing Warp dependency. Custom or
unsupported configurations retain the original paths. Host diagnostics report
`response_basis_enabled`, `response_basis_builds` and
`response_reference_builds`; the latter counts retained matrix preparations,
not simulated steps. These settings do not reduce the active tissue region or
change the 60 Hz simulated timestep into a live-FPS guarantee.


Supported CUDA physical-grasp scenes also verify the current compact FEM
linear residual with float64 arithmetic before accepting a physical interval.
The optional bounded correction keeps the original tolerance and shared
160-action primary-matrix budget. `GraspScene.certified_linear_residual`
controls this supported-path check; unsupported or differentiable solvers
keep their original behavior. Preparation graphs reuse only the unchanged
contact-input and Woodbury-packet kernel sequences. Neither setting changes
the tissue resolution, native contact budgets or simulated 60 Hz timestep.


## Tool and tissue contact rig

`fem_tool_tissue` is the test rig of the contact between the tools and the
tissue. It holds the two dVRK tools of `fem_needle_grasp` on the same rigid
solver and the same block of tissue (region 1: 7,205 nodes and 29,160
tetrahedra, 1,841 nodes held on the cut faces and the underside, the material
and solver settings of the grasp scene), and nothing else: no needle, no
thread and no needle/tissue coupling. A frame is four physics intervals, and
an interval is the rigid interval of the tools followed by the tissue step.

```bash
uv run --locked python -m newton_solver_suturebot.examples fem_tool_tissue
# The first-version contact law of the grasp scene instead of no contact:
uv run --locked python -m newton_solver_suturebot.examples fem_tool_tissue --contact first
# Without a window, with the self-check and every measurement as JSON:
uv run --locked python -m newton_solver_suturebot.examples fem_tool_tissue --viewer null --test --report rig.json
```

The needle driver (PSM1) follows a scripted path of eight phases through the
joint-target interface of the grasp scene (its jaws are closed in the first
seven and open and close in the eighth, the pinch): one palm pose per frame, its bounded
inverse kinematics solved for the whole path before the first frame, and the
arm targets of an interval on the straight line between the targets of two
frames. The forceps (PSM2) is parked as in the grasp scene; it is kept so that
the rigid model is the grasp scene's without the needle and the contact sees
the same capsules. The orientation of the palm never changes: the jaws
lie on the line from the tool's port to the middle of the block, as with a
straight wrist (65 degrees below the horizontal), and close horizontally.

| # | Phase | What the tool is commanded to do | Frames |
| --- | --- | --- | --- |
| 1 | `approach` | Come down from one jaw diameter above the middle of the block to its surface. | 26 |
| 2 | `press` | Press the jaws 1.5 mm into the tissue, hold. | 61 |
| 3 | `press_deep` | Press on to 3.0 mm, hold. | 77 |
| 4 | `slide` | Slide 10 mm along the long axis of the block at that depth, hold. | 230 |
| 5 | `lift` | Lift off to one jaw diameter above the surface, hold. | 161 |
| 6 | `border` | Move to the nearest held node of the block's upper surface, press 1.5 mm, hold, lift off. | 229 |
| 7 | `fast` | Move back above the middle and wait; then the fast approach: the target of the jaw tip runs along the tool's insertion axis, passes 2.0 mm inside the tissue at the end of its first frame and goes on until the tool itself has crossed the tissue surface; hold at 2.0 mm. | 127 of 131 |
| 8 | `pinch` | Lift off, move above the ridge of the tissue at the middle site, open the jaws, go down beside the ridge, close on it to 0.42 of its width, hold, lift it 5 mm (`--pinch-lift`), hold, lower, hold, open, hold, retreat. `--start-phase pinch` begins here. | 753 |

The eight phases are 1668 planned frames, 27.80 s of simulated time, of which
1664 are run: the fast approach needs 2 of its 6. The first seven are 911 frames
of 915, 15.2 s. The path follows the radius of the jaws (the
clearance, the speed in the air and the fast approach are multiples of it):
with the four capsules per tool of before 2026-11-17, whose jaw radius for the
tissue was 0.976 mm, it had 938 frames of 942, and the measurements further
down that name 938 frames were taken on that path.

Where the numbers come from:

- "Depth" is the depth of the deepest point of the closed jaws inside the
  tissue: the largest distance by which a point of their contact spheres lies
  inside the surface, to the nearest surface point. The spheres are those of
  the first tool/tissue contact (`capsule_samples` on the capsule set of the
  tools; the capsule that carries the jaw tip has a radius of 0.646 mm, which
  is the jaw radius of this path). On a flat surface this is the vertical
  depth below it. Every position of the path is found by moving the palm until
  the jaws have the commanded depth in the tissue at rest, so a tool exactly on
  its path, over tissue at rest, measures exactly the commanded depth. The
  block is a ridge about 4 mm wide beside the wound, so the palm travels
  further than the depth: 3.8 mm from the first touch for 3.0 mm at
  the middle and 1.8 mm for 1.5 mm at the border, which lies in the
  valley beside the ridge.
- The depths are options. Their defaults are 2.3, 4.6 and 3.1 jaw contact
  radii and 1.1, 2.3 and 1.5 mesh spacings (mean tetrahedron edge 1.31 mm):
  every press is deeper than one contact sphere and than one element.
- `--speed` (3 mm/s, the arc speed of `fem_needle_grasp`) is the speed of
  every motion that touches the tissue: 0.0125 mm, or 0.019 jaw radii, per
  physics interval. `--hold-seconds` (0.5 s, the settling time of
  `fem_needle_grasp`) is the length of every hold: six time constants of the
  slowest arm actuator.
- A position actuator lags a ramp of its target by the speed times kd / kp,
  at most 1/12 s for this arm. The tool waits one jaw diameter (1.29 mm) above
  the surface, and moves through the air at the speed whose lag is that
  clearance: 15.5 mm/s.
- The fast approach commands the jaw tip its own diameter per physics interval
  (310 mm/s) along the tool's insertion axis: two consecutive placements of the
  tip sphere do not overlap. The tool follows through its position actuators,
  and a position actuator slows down as it nears a target that stands still: a
  target that stopped 2.0 mm inside the tissue is reached at less than one jaw
  radius per interval. The target therefore passes that depth at the end of its
  first frame and goes on at the same speed, frame after frame, for as long as
  the tip sphere of the tool itself is clear of the tissue surface at rest; the
  example reads that gap in its observation after every frame of the approach,
  never in the step. The first frame after which the tip has crossed is
  followed by the hold, whose first frame brings the target back to 2.0 mm; the
  frames of the approach that are left are not run. They are planned up to the
  one at whose end a tool that lagged its target by the whole lag of the
  slowest actuator (310 mm/s times 1/12 s, 25.9 mm) would have brought the jaws
  to the surface, which the commanded jaws touch 3.0 mm from the start of the
  ramp: 6 frames of 5.2 mm, and no further than the tool can reach (all 6 are
  within reach, also with `--jaw-tilt-deg 30`; on the longer ramp of the path
  of before 2026-11-17 the range of the insertion joint ended it after 5).

Options: `--contact off|first|swept`, `--speed`, `--hold-seconds`,
`--press-depth`, `--deep-depth`, `--slide-length`, `--fast-depth` (the depth at
the end of the first frame of the fast approach and in the hold after it; it
also moves the start of the ramp, see the measurements), `--jaw-tilt-deg`
(turns the jaws about their closing direction towards the horizontal),
`--substeps`, `--rigid-backend auto|native|mujoco-warp`, `--num-frames` (0, the
default, is the whole path; with a number the run ends when the path has
reached that frame, also in a window), `--no-measure`, `--measure-intervals`
(also observe the end of every physics interval: a measurement run with reads
in every interval), `--report <path>` and `--overview`.

`--contact` chooses the one object that the scene calls at four places of an
interval (`examples/grasp/_tool_tissue_contact.py`): before the rigid interval
for its reaction on the tools, before the tissue step with the poses of the
tools at the start and at the end of the interval, after the tissue step, and
after the interval is accepted. `off` exchanges nothing: the tools pass
through the tissue. `first` is the law of the grasp scene
([Tool/tissue contact](#tooltissue-contact)) with the parameters that scene
derives. `swept` is the contact on the sweep of the capsules
([swept_tool_tissue_contact.py](../src/newton_solver_suturebot/_src/solvers/coupled/interface/swept_tool_tissue_contact.py),
whose module text states the law). In this rig the default is `off`
(`fem_needle_grasp` runs the swept law by default): see what is
measured and what is missing below.

What `swept` does. After every accepted interval each of the eight drawn
capsules is swept to the pose it will have at the end of the next interval
(its accepted pose plus the full interval times its accepted velocity). The
axis of a capsule is cut into pieces of at most one radius (162 in the scene)
and every surface triangle into the three parts that belong to its vertices; a
piece against a part gives a depth and a push direction, from the exact
closest pair, or, where the axis went through the surface, along the direction
out of the tissue against the capsule over its whole sweep.

- Tissue that can follow is moved, not loaded. A free surface node takes the
  deepest of its own parts, is moved onto the predicted capsule with zero
  velocity and is held there as a point through the next interval by a block
  on its diagonal of the tissue matrix
  ([node_holds.py](../src/newton_solver_suturebot/_src/solvers/fem/node_holds.py):
  4096 times the mean of its own block); the tissue step moves everything else
  around it. The elastic force on a held node along its push direction is the
  reaction on the tool, one interval late; a node that pulls is released.
- A node moves into the tissue by at most one eighth of what the smallest
  principal stretch of its tetrahedra has left above the stretch limit of the
  material law (0.2), times its rest altitude, per interval, so that four
  moving nodes of a tetrahedron use at most half of that margin and no
  placement can invert an element.
- Tissue that is held (fixed nodes, the pad outside the block) or cannot
  follow at that rate stops the tool: every axis piece stores one force, a
  vector, that grows with the depth that the tissue under it cannot take
  (`K_c = m / h^2` with the interval `h` and the smallest effective mass `m`
  of that tool at a capsule end, taken at the joint coordinates of its last
  placement: 23.9 g and 1,376 N/m for the driven tool, 20.5 g and 1,179 N/m
  for the parked one with its jaws open, on either backend; integral gain
  `K_c / 16`).
- The stored force does not outlive its cause. Once no tissue under a piece
  is limited, its stored force runs out at `K_c` times the slack of that
  tissue (or the clearance of the piece) per interval, and what a piece loses
  goes to the pieces of the same tool that still stop it, in proportion to
  their depths. Only when no piece of the tool is stopped is it lost: a tool
  is not carried over tissue that can yield and not pushed when nothing
  touches it.

The object sits in three of the four calls: its stored reaction is added to the
tools before the rigid interval of every attempt, the forces that balance the
elastic force on the held nodes are added before the tissue step of every
attempt (so that the right-hand side of the tissue solve holds only their
change), and the update runs after an interval is accepted, in the end-of-step
graph of each state buffer. In `fem_needle_grasp` the same three calls belong
in the worker thread: the two additions with its stored loads, the update in
its graph after an accepted step.
Outside the step the scene calls the object once more after every placement
of the tools (start-up and a reset): the swept contact then takes `K_c` of
every tool from its placed joint coordinates, a host computation.

Measurements. After every step the example observes the accepted state, in a
call of its own (with `--measure-intervals` also after every accepted interval): the step reads one array of three numbers per physics
interval (the check that decides whether the interval is accepted) and nothing
else. A frame's record holds the commanded depth, the depth of the jaws, the
wrist and the shaft inside the deformed tissue surface, the largest
displacement of a surface node and its part into the tissue (the dimple), the
contact force on the tissue and on the tools, the distance of the palm from
its target, the largest travel of the jaw tip in one interval, the gap of the
tip sphere to the tissue surface at rest, the rejected intervals and the time
of the step. The check of an interval also notes, on the device, the interval
in which that gap changes sign from clear to inside: the observation reads the
travel of the tip in it and the gap on both sides of it with the same array as
the largest travel. The summary by phase is printed at the end; `--report`
writes it with every record, the derived numbers of the path, the frame times
and the census of graph captures and kernel module loads. A step or a loop
period over 66.7 ms is listed with the frame of the path that it ran, as the
records number their frames; after a failed phase that is not its count in the
run. `--no-measure` leaves the observation out, except after the frames of the
fast approach, whose end depends on where the tool is.

An interval that the tetrahedron guard rejects is repeated from its unchanged
start with half the interval, as in the coupled scenes. With a contact, a
tissue step that cannot be resolved ends the phase: it is recorded as failed,
the tissue is put back at rest, the contact is cleared and the tool is placed
at the start of the next phase that begins clear of the tissue (`border`,
then `fast`), so one run shows every phase group. Without a contact, and for
a failure of the rigid solver, the run halts.

`--test` checks a run. With the contact off: the path is run to its end, all
states are finite, no interval is rejected, and at the end of every hold the
palm is within 0.039 mm of its target and the measured depth of the jaws
within 0.039 mm of the commanded one (twice what the tool's inverse kinematics
leaves: 0.01 mm in position and 0.001 rad over the 9.7 mm reach of the jaws);
while the tool moves, the palm lags by no more than the fastest commanded speed
since the last hold times 1/12 s; and in the fast approach the tool itself, not
its command, travels more than the contact radius of the jaw tip in the physics
interval in which the tip sphere crosses the tissue surface at rest. That
travel, the frame and the gap on both sides of the interval are in the line of
JSON that the check prints and in the report
(`fast_tip_travel_per_interval_jaw_radii`). With a contact the failed phases
are reported and do not fail the process, and the travel of the crossing is
reported and not checked: an interval that the guard halves is half as long. In
both cases a halted run, a nonfinite state, a graph capture or a kernel module
load after start-up do.

Measured with the capsule set fitted to the drawn tool (one DGX Spark, GB10,
MuJoCo Warp backend, 2026-11-17, `--viewer null --test`; "before" is the same
tree with the four capsules per tool of before, whose path has 938 frames).
The example also measures how far the drawn tool itself is inside the tissue
as drawn: the deepest vertex of the CAD of each part (`drawn_part_depth_m` of
an observation, the last column of the summary, `cad_jaw_depth_max_m` in the
report).

| `--contact swept --measure-intervals` | Before | With the fitted set |
| --- | --- | --- |
| Self-check, phases completed, rejected intervals | passes, 7 of 7, 0 | passes, 7 of 7, 0 |
| Drawn jaws (CAD) inside the tissue, largest over the intervals: `press`, `press_deep`, `slide`, `lift`, `border`, `fast` | 355, 452, 249, 135, 0, 2318 micrometres | 87, 93, 0, 0, 0, 2210 micrometres |
| The same at the end of the holds of `press`, `press_deep`, `slide`, `fast` | 231, 222, 100, 157 micrometres | 0, 0, 0, 58 micrometres |
| Jaw capsules inside the tissue, largest over the intervals, same phases | 128, 226, 74, 6, 53, 2469 micrometres | 127, 181, 72, 11, 61, 2301 micrometres |
| Dimple at the end of the holds of `press`, `press_deep`, `slide` | 1.06, 1.19, 1.47 mm | 1.07, 1.10, 1.29 mm |
| Largest force on the tissue in `press`, `press_deep`, `slide` | 36, 43, 56 mN | 34, 36, 54 mN |
| Axis pieces of the contact, contact spheres of the measurement | 162, 162 | 274, 274 |
| Step, median and longest | 18.2 and 50.9 ms, none over 66.7 ms | 21.9 and 86.8 ms, one over 66.7 ms (frame 885, in the hold after the fast approach) |
| Graph captures and kernel module loads after start-up | 0 and 0 | 0 and 0 |

Without a contact (`--contact off`) the self-check passes before and with the
set; a step takes 7.25 ms before and 7.44 ms with it (median), none over
66.7 ms. The fast approach is still not stopped at the surface (the known
failure of the swept contact); its cost in the frame after the crossing grows
with the number of axis pieces.

Measured on one DGX Spark (GB10), MuJoCo Warp backend, 2026-11-17, with the
four capsules per tool of before (path of 938 frames):

- Without a contact and without a window (`--viewer null --test`) the 938
  frames take 7.7 s, after a start-up of 4.8 s with a warm kernel cache. A step
  takes 7.4 ms (median), 8.6 ms (99th percentile) and 11.8 ms at most; the
  observation of a frame takes 0.47 ms. No step is slower than 66.7 ms.
- At the end of every hold the palm is within 1.6 micrometres of its target and
  the jaws are as deep as commanded within 0.1 micrometres. While it moves at 3
  mm/s the palm lags by 0.12 mm, in the transits by up to 1.72 mm and in the
  fast approach by 9.8 mm. The tissue does not move (largest surface
  displacement 2.6 nm, the rounding of the solve). The jaws pass through it to
  the commanded depth; the wrist stays 2.5 mm and the shaft 9.4 mm clear of the
  surface in every phase.
- With the window (`--viewer gl --num-frames 942`, the whole path: 938 frames)
  the loop of step, observation and drawing takes 14.1 ms (median) and 16.6 ms
  (99th percentile), 70 frames per second over the run. Slower than 66.7 ms:
  frame 586 (159 ms: step 11 ms, observation 1 ms; the X server's log has its
  probe of the sleeping monitor 100 ms after the frame began). The screen was
  blanked during the run. The X server then detects the sleeping monitor again
  every 10.24 s and serves no window for about 0.14 s; the draw of the frame
  that falls there waits for it. No other frame is over 21.9 ms.
- Start-up captures 3 CUDA graphs (the tangent refresh of a frame and the
  interval of each of the two state buffers; 5 with `first`, whose end-of-step
  call is a graph of each buffer) and loads 77 kernel module builds (83 with
  the window). After start-up no graph is captured, no Krylov solve is issued
  outside a graph and no kernel module is loaded in any of the 5 runs of
  2026-11-17, also through the failed phases of `first` and with
  `--no-measure`. A step reads one array of three numbers per physics interval
  and nothing else.
- Fast approach: the path commands the jaw tip 2.00 contact radii per physics
  interval. The tool starts 5.1 mm above the surface and is still gaining speed
  when its tip sphere crosses the surface at rest, in frame 907, the second of
  the approach: in the interval of the crossing it travels 1.43 contact radii
  (the gap of the tip sphere is 1.11 mm before that interval and -0.18 mm after
  it), and 1.44 radii at most in any interval. 2 of the 6 planned frames of the
  approach are run. The hold then brings the target back; the jaws go 3.21 mm
  deep before they settle at the commanded 2.0 mm. With a target that stops at
  2.0 mm after one frame, as before this rule, the tool crosses the surface at
  0.63 radii per interval and never exceeds 0.85 (CPU, native MuJoCo). With
  `--jaw-tilt-deg 30` the crossing is at 1.43 radii, 2 of 5 frames (CPU, native
  MuJoCo).
- `--fast-depth` does not make the crossing faster. The ramp starts one frame
  of the fast speed (7.8 mm) before that depth, so a deeper target starts
  nearer to the surface and the tool crosses it earlier in its acceleration.
  With 1, 2, 3 and 4 mm the jaws start 6.1, 5.1, 3.4 and 0.6 mm above the
  surface, the tip crosses it at 1.58, 1.43, 1.26 and 0.64 contact radii per
  interval after 3, 2, 2 and 1 frames of the approach, and the jaws go 4.5,
  3.2, 3.8 and 4.1 mm deep (CPU, native MuJoCo, the fast phase alone). On the
  GPU `--fast-depth 0.004 --test` crosses at 0.64 radii and fails the
  self-check on that number.
- `--contact first` reproduces the failures of the first version in one run
  (283 steps, 4.0 s; a step takes 8.5 ms (median); 9 steps are slower than 66.7
  ms, 87 to 406 ms, all with rejected intervals: the frames 54, 56, 673, 678,
  679, 681, 682, 683 and 684 of the path):
  - `press`: the two tip spheres carry the load. The first intervals are
    rejected in frame 54, at a commanded depth of 0.73 mm (30 mN, dimple 0.72
    mm), and the tissue step cannot be resolved in frame 57, at 0.87 mm (34 mN
    at most, 22 rejected intervals). The phases up to the lift are skipped.
  - `border`: one sphere presses on the held node. The force climbs to 227 mN
    at a commanded depth of 1.16 mm while the jaws stay 0.50 mm inside the
    surface and the palm is pushed 0.63 mm off its target; first rejected
    intervals in frame 673 (0.74 mm), failure in frame 684 (1.21 mm).
  - `fast`: the tool crosses the surface at rest as without a contact (1.43
    radii in the interval of the crossing, frame 907): the law acts one
    interval late. The jaws are 0.18 mm inside the deformed surface in the
    frame of the first force, 2.96 mm at most and 2.12 mm in the last frame
    that is accepted (31 mN on 12 spheres, dimple 1.11 mm); the tissue step
    cannot be resolved in frame 914, after 2 accepted frames of the hold.
  - Three runs (this one and two of 2026-11-16, whose path was the same up to
    the fast approach) fail `press` in frames 57, 57 and 57 and `border` in
    frames 684, 684 and 681, with 227, 230 and 206 mN at the border: the frame
    and the force of a failure change by a few frames and a few tens of
    millinewtons from run to run.

- `--contact swept` (2026-11-17, `--viewer null --test --measure-intervals`,
  938 frames, 3,752 intervals): every phase is completed, with no rejected
  interval and no inverted element. "Inside" is how far the deepest point of
  the exact drawn capsules of the driven tool is inside the tissue surface at
  the end of an interval: the radius of the capsule less the distance of its
  axis from the surface (stations 1/32 of a radius apart along the axis; this
  is exact around the axis and on the caps), and for a capsule whose axis is
  under the surface the deepest of the points on its own surface; the sign is
  that of ray parity. The chain of spheres of the other measurements lies up
  to 0.13 radii inside the capsule between two centres.
  - Tissue that follows: 6.7 micrometres inside at most in the presses, the
    slide and at the border. In the lift the dimple rises with the jaws; over
    four runs the capsules are at most 5.8, 5.9, 7.3 and 32 micrometres inside
    there, the 32 in a single interval (12 as drawn, after the update of
    that interval has placed the nodes).
  - `press`: the dimple follows the jaws to about 0.8 mm; then the tissue
    under the jaw tips is at its stretch limit and stops the tool, 0.65 mm
    short of its target. The dimple reaches 1.10 mm (1.06 mm at the end of
    the hold) with 425 mN on the tool; the capsules are 128 micrometres
    inside while the tool is still driven, below 20 micrometres 0.23 s after
    the command stops and 2.2 at the end of the hold.
  - `press_deep`: the same rule stops the tool at a dimple of 1.06 to
    1.31 mm, 2.7 mm short of its target, with 1.96 N. 226
    micrometres inside while driven, below 20 micrometres 0.24 s after the
    command stops, 0.6 at the end of the hold.
  - `slide`: the dimple travels with the jaws, 0.76 to 1.24 mm deep,
    with 1.1 to 2.4 N on the tool (0.24 of the largest actuator
    force limit); the capsules are at most 75 micrometres inside.
  - `lift`: the tool stays on the tissue while its actuators unload
    (0.98 N at the start of the lift), and the tissue follows it up. No
    force acts on a tool that no node touches.
  - `border`: the tool is stopped on the held node (dimple 0.06 mm, 596 mN):
    53 micrometres inside while driven, below 20 micrometres 0.06 s after
    the command stops, none at the end of the hold.
  - `fast`: the tool arrives at 1.43 contact radii per interval (335 mm/s) and
    is not stopped at the surface. The capsule goes 2.47 mm into the tissue,
    its centre line 1.82 mm under the surface, and is then pushed back out:
    below 20 micrometres 18 intervals after the first touch, under 2
    micrometres at the end of the 0.5 s hold. The stored force stops a tool
    that is driven at the pressing speed; it does not stop one that arrives at
    this speed within one radius.
  - Above 50 mN the contact force changes by at most 16.5 % in one interval,
    except in the impact of the fast approach (82.3 %).
  - Without the interval measurement a step takes 10.5 ms (median; 7.4 ms
    without a contact), 12.9 ms (99th percentile) and 40.0 ms at most, in the
    frames after the fast approach, while every probe is inside the tissue; no
    step is slower than 66.7 ms. Start-up captures 5 CUDA graphs and loads 80
    kernel module builds; after start-up none, also with an empty kernel cache
    (start-up 128 s).
  - With the window (`--viewer gl --num-frames 942`, five runs, the display
    woken before each and the monitor awake throughout) the loop of step,
    observation and drawing takes 17.7 to 18.0 ms (median) and 20.2 to 21.0 ms
    (99th percentile), 55 to 56 frames per second; a step takes 13.1 ms
    (median) and 43 to 47 ms at most, after the fast approach (loop period 48
    to 52 ms).
    One draw per run takes 32 to 52 ms in three of the five runs (frames 303,
    424 and 437 of the path, steps of 13 to 14 ms), and 34 to 46 ms in each of
    three runs without a contact (frames 516 to 556); the X server's log has
    no line at those moments. With the swept contact that draw makes a loop
    period of 67.1 ms in one of the five runs, 0.4 ms over 66.7 ms; no other
    period is over 58.6 ms.
  - Why it is not the default of this rig (`fem_needle_grasp` runs it by
    default since 2026-11-17, with these shortfalls open): the deep press
    exceeds 0.14 mm while the tool is
    driven, the fast approach is not stopped within one contact radius, and
    its impact changes the force by more than a quarter per interval. The
    remainder of a stopped tool closes 0.23 to 0.24 s after the command stops,
    not within 0.1 s. On the recorded path of the second jaw of
    `fem_needle_grasp`, replayed on the CPU, the tissue under the stopped tool
    is compressed to 0.19 of its rest volume when the jaw withdraws, below the
    0.2 that the review of the design asked for. In one of four runs one
    interval of the lift has more than 20 micrometres in tissue that follows,
    and with the window one loop period in five runs is over 66.7 ms (see
    above).

What it does not model: friction between tool and tissue other than between the
two jaws of a tool (the pinch, below), the needle, the
thread, cutting or puncture, and the tools' gravity (compensated, as in the
grasp scene). The tool's dimensions, inertias and actuator gains are the
illustrative ones of the grasp scene.

### The pinch (phase 8)

In the eighth phase the jaws take tissue between them. The planner finds the ridge of the
resting surface at the middle site along the closing direction of the jaws (top
2.30 mm, 4.80 mm wide at the height of the tip centres), opens the
jaws to the width of the ridge plus a clearance of 0.25 mm on each side
(5.30 mm between the tips), lowers them until the tips are 3.0 mm below the top
(`--deep-depth`), closes them to a tip gap of 0.42 of the width
(`PINCH_GAP_FRACTION`, a choice of the script: 2.02 mm), lifts by
`--pinch-lift` (5 mm by default, `PINCH_LIFT`), lowers, and opens. Every length
comes from the capsule set of the tool and from the surface at rest; the phase has
753 frames and the whole path 1668 (27.80 s).

```bash
# The pinch alone, with the contact that carries it:
uv run --locked python -m newton_solver_suturebot.examples fem_tool_tissue --contact swept --start-phase pinch
# Another lift [m] (what 3 and 8 mm give: below):
uv run --locked python -m newton_solver_suturebot.examples fem_tool_tissue --contact swept --start-phase pinch --pinch-lift 0.003
```

In the container, with the window:

```bash
./docker/run.sh fem_tool_tissue --device cuda:0 --viewer gl --contact swept --start-phase pinch
./docker/run.sh fem_tool_tissue --device cuda:0 --viewer gl --contact swept --start-phase pinch --pinch-lift 0.003
```

What holds the tissue is part of the swept contact in every scene, not of this example
(`JawTissuePinch` in the tool/tissue exchange, [method](method.md#tooltissue-contact)):
tissue that the contact holds on the inner faces of the two jaws of a tool is carried by
them while the pull across the closing direction stays below `2 mu S`, with `S` the
contact's force on the jaws along the closing direction. `mu` is
`GrasperConfig.tissue_friction` = 0.83, from Heijnsdijk, de Visser, Dankelman and Gouma,
"Slip and damage properties of jaws of laparoscopic graspers", Surg Endosc 18(6):974-979,
2004: bowel pulled with 5 N slips out of a grasper between a pinch of 3 N and one of 22 N,
by the jaw profile, which is 0.11 to 0.83 for two faces. The upper end (a profiled jaw) is
the default. It is an effective number: a profile holds by interlock. With
`--contact off` or `first` there is no pinch and the jaws slide off as before.

What the default run does (GPU, 2026-11-17, the whole path, run `g1_default`): at the end of the squeeze hold the pair
sticks with 14 + 13 carried and 12 + 12 clamped nodes and a squeeze of
275 mN; at the end of the lift hold the carried tissue is 4.49 mm
up with the points of the jaws that held it 4.49 mm up and the palm
4.52 mm of 5 mm commanded (the arm gives under what it carries); the
pull is 484 mN against a limit of 612 mN (1.26 times the pull);
the upper surface is 1.86 mm up at 3 mm and 0.36 mm at 5 mm from the
pinch. Before the pinch existed the tissue rose 0.3 mm and
the jaws slid off; until 2026-11-17 the default lift was 3 mm.

`--test` with `--contact swept` checks the phase row by row and fails the process for a
row that is not met; a row without a verdict (frames that were not run, or the work row of
a pair that slipped) does not fail it:

| # | Row | Rule |
| --- | --- | --- |
| 1 | Something is clamped | At the end of the squeeze hold each jaw has a clamped node and the pair sticks. |
| 2 | The squeeze is the jaws' | At the end of both holds the contact's force on each jaw along its closing direction is positive and no larger than the torque limit of the jaw actuator over the distance of its nearest carried node from the hinge. |
| 3a | The tissue is at its places on the jaws at rest | At the end of the lift hold and of the lower hold every node that was carried at the end of the squeeze hold is as high above where it was then as the point of its jaw body that coincided with it (its place), within the tracking tolerance of the tool (39.5 um), and the slip length has not grown. |
| 3b | The tissue stays at its places on moving jaws | From the end of the squeeze hold to the end of the last closed segment, in every interval in which the pair stuck: no carried node was further from its place when the interval ended, across the closing direction, than (1 + 3/4096) times the largest travel of a place in one interval; and at the end of the lift the mean rise of the carried nodes is that of their places within the tolerance plus that travel. Where the bound comes from: [method](method.md#tooltissue-contact). The same distance while the jaws close, where the contact itself places the nodes, is printed and not judged. |
| 4 | The tool carries the pull | The palm stays behind its commanded lift by what the compliance of the tool (its actuator gains, at the planned configuration of the two holds) makes of the change of the tissue's wrench on palm and jaws between the two holds; that wrench is taken from the tissue (the contact's wrench plus the elastic forces of the carried nodes at those nodes), not from what the pinch applies. |
| 5 | A tent forms | The rise of the upper surface falls from the carried nodes over 3 mm to 5 mm from the pinch and is positive there. |
| 6 | Nothing breaks | The smallest volume ratio of a tetrahedron is positive in every frame and no interval of the phase is rejected. |
| 7 | Release | At the end of the hold after the release no node is carried, kept or let down by the pinch and the upper surface within 3 mm of the pinch is within 20 um of rest. |
| 8 | The work through the carried nodes balances | From the end of the squeeze hold to the end of the lower hold, in the lift with its hold and in the lowering with its hold: what the tool paid through the carried nodes (their force on the jaws along the travel of the jaws) is what the tissue received through them (the elastic force on the nodes along their moves), within half of `T_max d_max + 2 T_max r_max` in each (largest pull, largest travel of a place in one interval, largest distance of a node from its place): the one-interval lag of the reaction and the work of making up that distance. Not judged for a pair that slipped. Not in it: the closing and the opening, and the contact's own force on the tool, which are printed. |
| 9 | The grip holds the lift | No slip while the jaws are closed; the friction limit `2 mu S` at least the pull at the end of every frame from the squeeze hold to the lower hold; at the end of the lift hold the smallest principal stretch of the tetrahedra without a carried node above the stretch limit of the material (0.2). Printed with it: the squeeze at the squeeze hold `S_0`, the pull per rise `k_T`, the rise `2 mu S_0 / k_T` at which that pull would reach the friction limit of `S_0`, the limit over the pull at the end of the lift and of its hold, and the smallest stretch of the tent. |

With the default lift every row is met on the CPU (the phase from its restart, run `f_default`: 687
frames, 0 rejected intervals; the tissue at rest 0.2 s before the jaws have opened). On the GPU the whole path gives
the same verdicts (run `g1_default`: 1664 frames, 0 rejected intervals, 0 graph captures and 0 kernel
module loads after start-up; a step takes 15.0 ms in the median, 19.4 ms in the pinch, and one step
is over 66.7 ms, in the fast approach of phase 7), and the runs with `--pinch-lift 0.003` and `0.008`
end as on the CPU: row 8 is not met at 3 mm and row 3a at 8 mm (runs `g1_l3`, `g1_l8`; the table below).

The carried tissue is at its places on the jaws, at rest and while they move. A carried node is given
the velocity that takes it to its place on its jaw at the pose predicted for the next interval, and
the tissue step moves it there with its neighbours (rule P3,
[method](method.md#tooltissue-contact)). Measured at the default (CPU): the largest distance of a
carried node from its place at the end of an interval is 4.55 um while the jaws travel 13.70 um per
interval; at the end of the lift the 27 carried nodes are 0.2 um above their places in the mean
(farthest 18.3 um); at the end of the lift hold the farthest is 19.2 um and at the end of the lower
hold 34.7 um from its place (allowed 39.5 um). The last of these is the thinnest number of the
self-check: through a lift and a lowering one jaw ends 0.03 rad further open than it began, and nodes
that only the pinch holds do not follow a jaw that opens alone. The panel shows the same comparison
while the phase runs ("Tissue behind its places on the jaws").

How high it lifts, and what limits it (CPU, the phase from its restart, friction 0.83, tips 3 mm
below the top of the ridge, 2026-11-17; "limit / pull" is `2 mu S` over the pull at the end of the
lift; run names in brackets):

| Commanded close | Lift commanded | Carried tissue / its places at the lift hold | Limit / pull | Smallest volume ratio | Outcome |
| --- | --- | --- | --- | --- | --- |
| 0.42 (default) | 5 mm (default) | 4.49 / 4.49 mm | 1.16 (1.27 after the hold) | 0.11 | Completes, 0 rejected; every row met [`f_default`]. |
| 0.42 | 5 mm, holds of 2 s (`--hold-seconds 2.0`) | 3.50 / 4.27 mm | 1.11 | 0.09 | Completes, 0 rejected, the tissue at rest; the pair slips for two intervals in the lift and lets go of 10 of 27 nodes: rows 3a, 3b and 9 not met [`f10_hold2`]. |
| 0.42 | 3 mm, holds of 2 s | 2.33 / 2.33 mm | 1.32 (1.13 after the hold) | 0.05 | Completes, 0 rejected, no slip; every row met but 8 [`x_hold2_l3`]. |
| 0.42 | 3 mm | 2.75 / 2.74 mm | 1.61 | 0.11 | Completes, 0 rejected; every row met but 8: in the lowering the tool gets 0.0038 mJ less back than the tissue gives, 0.0034 mJ allowed [`s7_l3`]. |
| 0.42 | 8 mm | 7.13 / 7.13 mm | 1.06 | 0.08 | Held, lowered and released, 0 rejected, the tissue at rest; every row met but 3a: after the lowering one of 27 nodes is 44 um from its place [`f_l8`]. |
| 0.42 | 12 mm | 10.79 / 10.79 mm | 1.09 | 0.03 | Held; the tissue around the grip is at the stretch limit of the material from 9.4 mm of rise; the phase FAILS while lowering [`x_l12`]. |
| 0.36 | 5 mm | 4.62 / 4.62 mm | 1.33 | 0.04 | Completes, 0 rejected; rows 3a and 8 not met: after the lowering one of 30 nodes is 49 um from its place [`s8c_l5`]. |
| 0.36 | 8 mm | 7.33 / 7.32 mm | 1.07 | 0.02 | Held, lowered and released, 0 rejected; rows 3a and 8 not met: after the lowering two of 30 nodes are up to 67 um from their places [`s8c_l8`]. |
| contact alone (no pinch) | 3 mm | 0.24 / 2.79 mm | - | - | The jaws slide off (measured before 2026-11-17). |

- At the default close the friction limit follows the pull. The squeeze grows with the lift
  (275 mN at the squeeze hold, 331 mN at the end of a 5 mm lift, 478 mN at 8 mm) and the pull
  with 108 N/m of rise: the limit over the pull falls from 1.61 at 3 mm to 1.16 at 5 mm and
  stays between 1.04 and 1.09 from 6 mm of rise to 12 mm commanded. With the script's holds
  of 0.5 s the pair did not slip in a lift. With the squeeze of the squeeze hold alone the
  pull would reach the friction limit at `2 mu S_0 / k_T` = 4.2 mm of rise (row 9 prints it),
  below the 4.4 mm of the default lift (4.5 mm after its hold): the default lift is held by
  the growth of the squeeze, which is measured and not derived.
- The tissue elements are what ends a lift: around the grip they reach the stretch limit of
  the material at 9.4 mm of rise (10.6 mm commanded), twice the rise of the default lift.
- A lift of 12 mm fails while lowering: the squeeze falls, the pair slips for one interval and
  lets go of a part of the grip, the rest holds again, and the tissue between the two is
  squeezed until a step is rejected to failure. A slip under load that lets the tissue down to
  rest is not built; the example reports the phase as failed.
- No largest lift of this grip is derived. The default of 5 mm is a choice of the script, beyond
  the 4.2 mm of rise of the friction estimate above; the self-check is met there with the
  script's holds, and it is not met at 3 mm (row 8) and at 8 mm (row 3a); 12 mm fails (the table).
- The hold before the lift matters. After a squeeze hold of 2 s the squeeze falls through the
  lift and a 5 mm lift slips for two intervals: a part of the grip lets go and the tissue rises
  3.5 mm instead of 4.5 mm (it is released to rest all the same). A 3 mm lift holds after the
  same hold. The self-check is met with the script's holds of 0.5 s.
- A firmer close holds with more to spare (1.33 at 5 mm at a close of 0.36) and squeezes the
  elements at the faces thinner; the self-check is not met there (rows 3a and 8), and after a
  squeeze hold of 2 s it slips in a 5 mm lift as well [`x_hold2_c36`].
- The arm reaches and sags by what it carries (0.48 mm at 5 mm, 0.83 mm at 8 mm).
- `--pinch-lift` sets another lift: `0.003` is the default of before (row 8 of the self-check
  is not met there, by 0.0004 mJ), `0.008` is held and released with one node 44 um from its
  place after the lowering (row 3a is not met: 39.5 um allowed).

Two limits of the mechanism: clamped tissue goes with the jaws where both jaws go (up, down,
along the jaws); where they close, open, or only one of them moves, it is the contact that places it.
And whether the lift holds depends on the friction (measured before 2026-11-17 at the default
close: held at 0.83 and at 0.30, where the squeeze is larger; at 0.11 no face is inside the friction cone, the pair never clamps and the tissue slides out with 0.25 mm of rise).
