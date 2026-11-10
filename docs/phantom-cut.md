# Full-pad cutting

`fem_phantom_cut` simulates the entire bundled tissue pad using the local FEM
and embedded cutting implementation. All 97,489 original vertices and 413,405
tetrahedra are loaded; adjoining stored partitions are welded. There is no
region selector, inactive tissue or fixed internal/underside boundary. The pad
is translated vertically so its lowest point starts at the floor, then rests
under gravity with surface contact. The mesh is not downsampled.

```bash
./docker/build.sh
./docker/run.sh fem_phantom_cut
```

For local development:

```bash
uv sync --locked --extra examples
uv run --locked python -m newton_solver_suturebot.examples fem_phantom_cut
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
| Start again | **Reset pad and cuts** restores the original uncut pad and clears holds. |

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

## Recorded demonstration and checks

The [full-pad movie](media/fem_phantom_cut.mp4) shows one finite slice separating
an edge strip, followed by a spring grip lifting and releasing it. All 413,405
original tetrahedra start active and there are no fixed nodes. The cut creates
two material components; the total mass before and after the incision is the
same float32 sum, 0.1343255192041397 kg.

The capture contains 360 frames over six simulated seconds. Finite-state and
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
