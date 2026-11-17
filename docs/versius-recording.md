# Versius instrument preview

`fem_versius` displays two supplied CMR Versius distal needle holders with
generated motion and a static phantom. It is a kinematic preview, not a measured
Versius recording or a physical suturing simulation. Its two stereo-view panels
use explicitly labeled **dVRK placeholder videos**. Those images are unrelated
to the generated Versius motion; overlay alignment has no calibration meaning.

The scene preserves all 12 regions of the same full pad as the tissue examples:
97,489 original unique global nodes and 413,405 tetrahedra. The volume payloads
remain in `assets/phantom_pad/`, and interior partition faces are excluded from rendering. It does
not substitute the preparation source's reconstructed pad. The pad is static
in this viewing example; retaining its mesh does not imply a running FEM solve.

```bash
./docker/build.sh
./docker/run.sh fem_versius
```

For local development:

```bash
uv sync --locked --extra examples
uv run --locked python -m newton_solver_suturebot.examples fem_versius
```

Append `--help` for the current playback options. Playback advances prepared
poses; it does not run the needle/FEM interaction or deform the tissue.

Use **Play recording**, **Loop** and **Restart recording** to control playback.
Dragging **Frame** pauses at the selected pose. **2 camera overlays** toggles
the initial left/right panels; **Frame scene** and **Frame phantom** adjust the
viewer framing. CLI options include `--playback-speed`, `--loop` and
`--no-camera-overlays`. The camera toggle's count follows the loaded bundle.
The overlay is built and rendered once while the scene loads, as in
[`fem_dvrk`](suturebot-recording.md); `--test` plays the same scripted user
without a recording switch.

## What the scene represents

| Part | Evidence and limits |
| --- | --- |
| Instruments | Supplied CMR NeedleHolderPlatform shaft, yoke and jaw meshes, with their authored transforms and material groups. These are distal instruments, not the complete BSU arms. |
| Motion | Generated articulation and placement for inspecting the scene. It is not measured Versius telemetry, a controller reconstruction or successful task execution. |
| Phantom | Packaged pad geometry displayed as a static scene prop. Its placement relative to the tools is illustrative. |
| Stereo views | Prepared left/right camera views with illustrative placement. They are not validated Versius camera-to-pad calibration. |
| Video panels | dVRK images used as explicit placeholders. They are not real Versius footage and do not provide synchronized reference motion or pixel-alignment evidence. |

The initial bundle uses recording schema 2 with an explicit `camera_order`
listing left and right. The viewer reads the ordered camera metadata rather
than assuming every recording has two or four cameras. Additional camera
payloads can use that representation; their presence still requires their own
source provenance and calibration checks.

## Preparation and provenance

Initial asset conversion, generated motion and placeholder-video preparation
live in the source project's `preparation/newton_versius/`; the final shared
layout is prepared by `preparation/newton_shared_assets/`. Geometry and materials
are authored once under `assets/versius/models/`. The recording under
`assets/versius/recordings/versius_stereo_preview/` supplies motion, images,
camera metadata and a small scene referencing those models and the existing pad.
Its `asset_root: "../../../"` resolves to the common examples assets directory.
Portable exports need the complete assets tree. Playback needs no source-project
import, data download or runtime calibration pass.

The source instrument meshes already use metres. Their GLB node transforms,
URDF visual origins and built-in mounting transforms must not be applied twice.
The supplied CAD does not provide a calibrated rigid-body dynamics model, and
its source rights are not replaced by the dVRK assets' license.
The bundle retains its [instrument source hashes](../src/newton_solver_suturebot/examples/assets/versius/recordings/versius_stereo_preview/instrument-provenance.json)
and [third-party notice](../src/newton_solver_suturebot/examples/assets/versius/recordings/versius_stereo_preview/THIRD_PARTY.txt).

Historical Versius research in the preparation source describes sampled motion
and videos that are not present in the local source checkout used for this
preview. Its retained optical registration attempts do not establish accepted
camera calibration. Genuine recordings therefore need an independently
prepared bundle with correct units, arm mapping, clocks and camera models;
the placeholder images must not silently be relabeled as Versius evidence.

Scene loading, valid transforms, playback and camera-panel checks assess this
preview's implementation. They do not establish physical interaction, measured
motion fidelity or calibration. A recorded demonstration shows the generated
preview in simulation time; encoded FPS is not live simulation performance.
See [provenance](provenance.md) for the distinction from the existing dVRK
recordings and historical validation reports.

The [recorded preview](media/fem_versius.mp4) shows 240 viewer frames at
1920×1080 with both labeled camera panels. The capture passed `--test`, which
checks displayed body poses against the prepared motion and checks finite
transforms. Its [capture metadata](media/fem_versius.json) records four seconds
of generated playback and 120 observed source-image indices. This is a preview
loading/rendering check, not physical or calibration validation.
