# Standalone package and evidence provenance

This repository contains the required implementation and data for four
examples: `fem_phantom_cut`, `fem_needle_insertion`, `fem_phantom_regions` and
`fem_dvrk`. The FEM core, embedded cutting, prepared mechanics, partitioned needle coupling, optional
`SolverMuJoCoFEM`, geometry queries and viewer helpers live under
`src/newton_solver_suturebot/`. Newton, Warp, MuJoCo and the other locked
packages are external dependencies installed normally by `uv`.

The full-pad cutting workflow uses the local cutting and material-grip
implementation. Its Line, Cylinder and Sphere tools use embedded Tet4
material fragments. The original
three-example extraction reports below retain their original scope and hashes.

## Bundled data

| Data | Local location and scope |
| --- | --- |
| Regional tissue geometry | [phantom_pad](../src/newton_solver_suturebot/examples/assets/phantom_pad/README.md): original tetrahedral coordinates, surface display and prepared twelve-region assignment |
| Curved needle | [needle_insertion](../src/newton_solver_suturebot/examples/assets/needle_insertion/README.md): standalone USD and provenance for the physical insertion example |
| Recording scene and samples | [suturebot](../src/newton_solver_suturebot/examples/assets/suturebot/README.md): prepared tools, fixture, cameras, phantom, motion, indexed source images and available point labels |

The four prepared recording choices resolve their USD and array/image references
inside the packaged assets. Replay needs no original dataset checkout, download
or calibration pass. Source dataset identifiers and historical preparation
paths in JSON/README files are provenance records, not dependencies. Preparing a
new recording is a separate optional workflow requiring its source data; the
runtime only loads a prepared `recording.json` package.

The `fem_dvrk` entry point assigns measured da Vinci Research Kit (dVRK) tool
poses from the SutureBot recordings kinematically over a static phantom.
It does not simulate tissue contact or needle insertion. Legacy
physical tool-contact helpers remain for validation and are not additional
launcher examples. The physical needle example instead advances a free rigid
body and regional FEM with two-way loads; it has no robot grasp, thread or knot.

## Imported demonstration media

The original three MP4s, GIF/JPG previews and JSON sidecars in `docs/media/`
retain their historical capture bytes. Each sidecar retains its original
module name, dependency names, parameters, source identifiers and capture wall
time. These fields have deliberately **not** been rewritten to imply a fresh
capture by `newton_solver_suturebot`.

The live replay entry and its primary gallery files were renamed from
`fem_suturebot` to `fem_dvrk`. The MP4, JPG, GIF and JSON bytes are unchanged;
the sidecar still records the original example/module name. The import manifest
uses the new copied-file paths while retaining each original path and hash.
Supplemental alignment/thread images and historical reports retain their names.

| Movie | Meaning of the recorded result |
| --- | --- |
| [Needle insertion](media/fem_needle_insertion.mp4) | Physical automatic insertion through two tissue sections, rendered in simulation time. Its earlier wall compliance was `1/60000` m²/N; the current example uses `1/300000` m²/N. |
| [Phantom regions](media/fem_phantom_regions.mp4) | Physical regional FEM and spring-grab response while changing the active region. |
| [dVRK recording](media/fem_dvrk.mp4) | Kinematic measured-motion replay and fitted camera/image overlays for the da Vinci Research Kit, not a physical suturing experiment. |

All three are encoded at 60 frames per second. Encoding/playback rate and capture
wall time do not establish live simulation FPS. The latest imported loaded
needle measurements remain below sustained 60 FPS. Camera/phantom alignment and
tissue/needle parameters remain estimates, with limitations in the individual
guides.

The [full-pad cutting movie](media/fem_phantom_cut.mp4) is a new standalone
capture, rather than an imported clip. Its [sidecar](media/fem_phantom_cut.json)
records this package's module, 360 frames, six simulated seconds and the
slice/grip/release sequence. The encoding rate is 60 FPS; the 199.447-second
capture wall time includes setup, rendering and encoding and is not live FPS.
The new [cutting validation report](validation/standalone/phantom-cut.json)
keeps its evidence separate from the original imported artifact hashes.

`docs/videos.json` stores the capture recipes. To replace an imported
clip with a fresh standalone capture, use the recorder and regenerate the README:

```bash
uv run --locked --extra examples python tools/record_examples.py fem_needle_insertion
python3 tools/build_readme.py --require-videos
```

Record only when a display/GPU workload is intended; the recorder performs real
work and is not a lightweight documentation build. A new capture writes its own
sidecar rather than relabeling old metadata.

## Validation and attribution

[import-provenance.json](import-provenance.json) records originating paths and
SHA-256 hashes for copied media, retained validation JSON and adapted document
sources. Original validation JSON remains byte-identical. Its old source hashes,
absolute paths and namespaces describe the original run; they do not certify the
new package. Compact evidence under `docs/validation/frame-budget/` includes the
latest imported copy-lifetime optimization, rejected alternatives and confinement
check. Large raw Nsight databases, frame histories and exploratory recordings are
retained locally under ignored `nsys/results/`. Their original run settings and
provenance are unchanged; they are not fresh measurements or runtime
dependencies. The original 41 imported artifacts remain byte-identical.
Exact earlier guides and preservation hashes are retained in the
[historical document archive](validation/original-fem-documents/README.md).

The [validation guide](validation.md) labels the imported numerical evidence.
Fresh standalone import, packaging and test checks are recorded in the
[standalone validation report](validation/standalone/validation.json) and its
adjacent audit files. The larger source manifest is kept at ignored
`results/extraction/source-manifest.json`. These reports describe the code
already present under `src/`; they are not runtime loaders or dependencies.
Full-trajectory comparisons, local equation tests, physical
completion and throughput measure different properties; their failures and
scopes must remain distinct.

The root [LICENSE](../LICENSE) and
[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES), source copyright headers and asset
attribution accompany the extracted code and data. Bundling does not change
third-party rights or establish clinical calibration or a validated surgical
model.
