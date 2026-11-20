# Full-pad cutting

> On 2026-11-17 `fem_phantom_cut` was renamed `fem_surgical_incision`, and the
> bundled Baymax body (135 mm across, on its back on the floor) replaced the
> pad. The controls below apply to it as they stand. The description of the
> pad, the measured numbers and the validation report in this guide
> are those of the pad example and have not been measured again on the body.
> The recorded clip is of the body ([below](#recorded-demonstration-and-checks)).
> The grip defaults and the `--asset` option named below are the pad's as well:
> on the body a grip is 3.0 mm in radius, 2.06 kN/m stiff and follows the mouse
> at no more than 3.0 mm/s by default, and `--asset` takes one tetrahedral body.
> The body rests on the floor. At rest its back is 0.27 mm under the floor plane
> at its deepest point and 0.10 mm on average over the 4.4 cm² that touch: the
> pressure there over the floor's penalty of 2e7 N/m³ (`--ground_stiffness`), at
> any number of substeps. The top of the belly comes down 2.2 mm as the body
> settles, because a body of 30 kPa flattens on its round back and its arms come
> to lie on the floor (measured on 2026-11-20, on `cuda:0` and on the CPU alike).
> Until 2026-11-20 the floor's force was divided by an implicit denominator that
> was not part of the solve, and the body lay 13 mm deep; the recorded clip
> still shows that.

`fem_phantom_cut` simulated the entire bundled tissue pad using the local FEM
and embedded cutting implementation. All 97,489 original vertices and 413,405
tetrahedra are loaded; adjoining stored partitions are welded. There is no
region selector, inactive tissue or fixed internal/underside boundary. The pad
is translated vertically so its lowest point starts at the floor, then rests
under gravity with surface contact. The mesh is not downsampled.

```bash
./docker/build.sh
./docker/run.sh fem_surgical_incision
```

For local development:

```bash
uv sync --locked --extra examples
uv run --locked python -m newton_solver_suturebot.examples fem_surgical_incision
```

## Controls

Choose the cutter in the example panel. **Line** is the default.

| Action | Control and result |
| --- | --- |
| Slice | With **Line**, left-drag a finite line across the pad and release. The incision extends through its depth along the camera view. |
| Drill | With **Cylinder**, left-click the visible surface. One straight cylindrical hole passes through the current material along that view ray; holding does not repeat it. |
| Carve | With **Sphere**, hold the left button on the surface. The spherical tool advances along its initial view direction; release ends the stroke. |
| Lift and separate material | Right-drag a surface patch. Releasing leaves a spring hold at the target; the tissue moves under finite forces. |
| Release a grip | Middle-click its green marker, or use **Clear holds** to release all grips. |
| Inspect separation | Enable the connectivity debug view to inspect pieces and grip connections. A grip can span both sides of a cut; release it if it is holding them together. |
| Start again | **Reset body and cuts** restores the original uncut body and clears holds. |

The default grip radius is 2 mm, stiffness 20 N/m and target speed limit
15 mm/s. These are spring controls, not exact node-position constraints.
Cutter dimensions are set from the picked surface depth and screen footprint;
the tool panel shows the resulting dimensions.

## Material and numerical scope

The preset uses Young's modulus 30 kPa, Poisson ratio 0.35 and density
1000 kg/m³, with gravity 9.81 m/s². It advances four substeps per 60 Hz
simulation frame with an 80-iteration CG budget per substep. These settings
describe the workload, not a live 60 FPS result. The complete pad requires
substantially more work and memory than a single active region.

Run with `--help` for options. Example-specific CLI names use underscores,
including `--cutter_mode cylinder`, `--substeps 8`, `--solver_iterations 120`,
`--grip_stiffness 20` and `--young_modulus 30000`. `--asset` selects a compatible
prepared regional pad asset; every listed region is loaded and welded.

The tools prescribe geometric cuts: they clip material fragments, duplicate
disconnected basis nodes, update mass/stiffness and create new contact faces.
They do not simulate fracture toughness or a force-triggered blade. Rejected
edits leave the previous material state in place. Finite capacity, volume and
contact guards do not guarantee arbitrary repeated cuts, freedom from
self-intersection or calibrated tissue response. Cut topology is not
differentiable. See [method](method.md#embedded-cuts-and-material-grips) and
[paper mappings](papers.md#embedded-cutting-and-streaming-assembly).

## Start-up, storage growth and the fixed reserve

On CUDA the example prepares its graphs and kernel modules before the first
frame, and again inside **Reset body and cuts**, which builds the scene anew.
Each of these start-ups captures and launches once:

- the graph of a line cut, for one and for two knife patches;
- the graph of the convex subtraction that **Cylinder** and **Sphere** share;
- the graph of the frame (tangent refresh, grips, FEM step, surface contact
  and volume guard).

It also loads the kernel modules whose first launch would otherwise fall into
a frame (the transfer of cloned nodes at the first accepted edit and the two
debug views) and draws the start state once. After a start-up an edit captures
no graph, builds no graph executable and loads no kernel module, unless it
makes storage grow.

By default storage grows inside the edit that needs it: polygon vertices and
faces per fragment (from 12 and 12 to 48 and 32, at most 64 and 48), the
scratch and the work list of the subtractor, and the boundary triangles. A
growth replaces buffers, so the graphs bound to them are captured again in
that frame: the subtractor's graph for its scratch or work list, the
subtractor's graph and both line graphs for the polygon storage, and the frame
graph, with the Krylov solves of the frame that warms it, for the boundary
triangles. Every growth is listed with its frame, sizes, time and captures
(`LinearTetCut.storage_growths`, `Example.storage_growths`).

`--test` plays a scripted user for 210 frames: a line across the pad, a grip,
a line of two knife patches, a cylinder click, the debug view on and off, the
grip removed, Reset, a line and a grip on the new pad. It fails if a graph was
captured after a start-up outside a listed growth, if a kernel module was
loaded after a start-up or if a start-up graph was never launched, and prints
the list of growths. The example's default length is 1200 frames; the runs
below pass `--num-frames 210`.

`--cut_reserve N` allocates fixed storage for `N` design loads at start and at
Reset. Nothing grows then, so no edit captures a graph; an edit that needs
more than the reserve is refused, the pad stays as it was and the panel names
the option. One design load is a line cut through the whole depth across the
pad in each horizontal direction and one cylinder and one sphere tunnel
through the depth with a tool of 1 mm radius, counted on the mesh
(`derive_reserve`). The tool radius is a chosen design input, not a bound: the
screen tools size the tool from the view (about 0.7 mm at the default camera
in a 1080-pixel-high window), and a tool drawn from farther away is larger.
`--cut_reserve_polygons VERTICES FACES` sets the polygon vertices and faces
reserved per fragment, 48 and 32 by default. These are not a bound either: one
pass of the sphere can need 64 and 34.

The polygon storage is most of the reserve. On the whole pad one design load
reserves 470,763 fragments; with 32 and 20 the cutter's storage is 29.3 GiB
by its allocation sizes (34.4 GiB measured for the whole process), with the
default 48 and 32 it is 57.3 GiB and at the cutter's limits of 64 and 48 it is
102.6 GiB, each plus about 5 GiB for the solver, the models and the contact.
Only the reserve of 32 and 20 was run. The whole pad therefore keeps growing
storage as its default.

Measured on 2026-11-17 on an NVIDIA GB10 with the null viewer, `--test
--num-frames 210` on `cuda:0` (runs under
`/tmp/suturebot-gpu-work/incision/step1/build/runs/gpu/`, a work folder
outside this repository):

| Run | Options | Kernel cache | Captures: construction / Reset / frames | Module loads after start-up | Frame of the first cylinder | Device memory: after construction, peak |
| --- | --- | --- | --- | --- | --- | --- |
| `hA_test_warm` | default | warm | 8 / 8 / 1 | 0 | 495 ms | 12.9 GiB, 22.3 GiB |
| `hB_test_cold` | default | empty at start | 8 / 8 / 1 | 0 | 494 ms | 12.9 GiB, 22.3 GiB |
| `hC_test_r1p32` | `--cut_reserve 1 --cut_reserve_polygons 32 20` | warm | 8 / 8 / 0 | 0 | 449 ms | 34.4 GiB, 38.0 GiB |

All three runs end with exit code 0, with no Krylov solve outside a graph
after a start-up and no start-up graph left unlaunched. The memory is that of
the process on the device as NVML reports it; the peak falls into Reset. The
host process holds 1.4 to 1.7 GiB.

In the default runs the one capture after start-up is the subtractor's graph
in the frame of the cylinder click, inside the listed growth of its scratch
faces from 12 to 24 (39 ms in `hA_test_warm`). That scratch starts at 12
faces, the size of the polygon storage, at start and again after Reset (12 at
the end of `hA_test_warm`). The first Cylinder click made it grow in every run
so far, at two places of the pad, so a default run captures this one graph in
the frame of its first Cylinder click. A first Cylinder click after a Reset
was not run. A frame without an edit takes
279 ms (median); a line cut adds 48 ms and a line of two patches 141 ms.
With the empty cache all 32 kernel module builds are compiled at construction
(173 s instead of 49 s) and none in a frame. Before this preparation the first
cylinder compiled and captured in its frame: with an empty kernel cache that
frame took 10.1 s (`h02_b_cold` under
`/tmp/suturebot-gpu-work/examples_census/step3/cut/runs/gpu/`; 8.4 s with 7.9 s
of compilation in the census of the examples), with a warm one 0.52 s
(`h01_b_warm`). Reset is a start-up and takes 46 to 47 s on the whole pad.
Ticking the connectivity debug view rebuilds its links on the host: 4.2 to
4.9 s in that frame on the whole pad, with no capture and no module load.

A default run whose edits make the polygon storage grow was not repeated here.
The earlier work measured it on the whole pad with a held sphere
(`h05_a_warm`): the growth of the polygon vertices from 12 to 48 took 1.2 s
and that of the faces from 12 to 32 took 2.4 s, with three captures each, in
frames of 1.5 and 2.8 s, and the device memory rose to 77 GiB.

With the reserve (`hC_test_r1p32`) the script also holds the sphere for 29
frames. Every scripted edit is accepted (13 cuts before Reset), nothing grows
and nothing is captured in a frame. The longest frame with an edit is 468 ms
(the line of two patches) against a median of 306 ms; the cylinder frame takes
449 ms and the frames under the held sphere at most 455 ms. The edits used
polygons of at most 23 vertices and 14 faces of the 32 and 20 reserved.

The eight cut and start-up test modules pass on `cuda:0` (71 tests, three
skipped that compare two CUDA runs bit for bit; `gpu_tests_tree.log` under
`/tmp/suturebot-gpu-work/incision/step1/check/logs/`). On the CPU
the scripted whole-pad run of 54 frames is bit-identical with and without
this preparation in state, topology and grips (`id_base`, `id_tree` under
`runs/cpu/`). The earlier work measured the same for the reserve of 32 and 20
(`id_f6_r1p32`, 32.9 GiB of host memory on the CPU).

## Recorded demonstration and checks

The [recorded clip](media/fem_surgical_incision.mp4) of 2026-11-19 shows the
Baymax body from straight above. The picture is the middle 800×660 of the
1280×720 recording: the `crop` of the clip's recipe cuts off the side panel,
the counters and the recorder's title bar, and the two grip markers, which
the viewer's interface draws, stay. A planar blade cuts the
belly along the midline, 28.5 mm long and 13.5 mm deep where the body is 42 mm
thick: the scripted cut goes through the call that a stroke of the Line tool
ends in, because the Line tool itself cuts through the whole depth of the
view. Two right-drags, one on each side of the wound, are pulled 4 mm and 8 mm
outward and released as spring holds. Nothing holds the body, so it follows
the first grip and the second pull opens the wound, to about 7.5 mm at the
skin (7.57 mm in this recording). From 3.5 s to 6.5 s of the clip, while the
second grip pulls, the viewer's Show Edges option is on (the `edges` of the
recipe): the dark lines are the edges of the triangles that the example
draws, which are the surface of the simulated tetrahedral mesh itself, not a
finer drawing mesh. Before the cut these are the 9,882 outer faces of the
tetrahedra, with a median edge of 1.9 mm; after it about 10,800 triangles, of
which the 860 or so on the two walls of the wound are seen at a grazing angle
and merge into dark hatching, as do those at the body's outline. The
body stays one piece of 38,730 original tetrahedra with no fixed nodes, and
the mass is 0.0924512 kg before and after the cut: the two float32 sums of
the capture metadata differ by one unit in the last place, 7.5e-9 kg. The
capture has 480 frames over eight simulated
seconds; the volume guard shortened no step, the finite-state and
positive-volume checks passed every 60 frames and at the end, and the
self-test of `--test` found no graph capture after start-up. The clip was
recorded before the floor contact was corrected on 2026-11-20: in it the body
sinks about 13 mm into the floor in the first half second, which the view from
above shows only as the body getting a tenth smaller. The example itself now
rests on the floor (the top of this guide), and the same scenario run on it
opens the wound to 7.6 mm. The [capture metadata](media/fem_surgical_incision.json)
holds the events; the 15.1-second recording wall time includes initialization,
rendering and encoding and is not a live-performance benchmark.

The rest of this section describes the full-pad movie of the pad example,
whose files left `docs/media/` on 2026-11-20 and remain in the repository's
history. It showed one finite slice separating
an edge strip, followed by a spring grip lifting and releasing it. All 413,405
original tetrahedra start active and there are no fixed nodes. The cut creates
two material components; the total mass before and after the incision is the
same float32 sum, 0.1343255192041397 kg.

That capture contains 360 frames over six simulated seconds. Finite-state and
positive-volume checks passed every 60 frames and at the end; selected frames
were also visually inspected. The 199.447-second recording wall time includes
initialization, rendering and encoding, so neither it nor the encoded 60 FPS
is a live-performance benchmark.

Focused cutting, full-mesh and rejected-edit rollback tests also passed, along
with a two-frame headless GPU smoke. The complete standalone CUDA suite passed
494 tests with no skips; Ruff checks and formatting passed for 189 Python files.
The rebuilt Docker image also passed isolated full-pad loading without network
access or host source mounts. The [standalone cutting report](validation/standalone/phantom-cut.json)
records their scope and raw evidence. Earlier three-example extraction results
in [validation](validation.md) remain separate historical checks.
