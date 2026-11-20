# Standalone package and evidence provenance

This repository contains the required implementation and data for nine
examples: `fem_surgical_incision`, `fem_needle_insertion`, `fem_phantom_regions` and
`fem_dvrk`, plus `fem_versius`, `needle_thread`, `fem_needle_suture`,
`fem_needle_grasp` and `fem_tool_tissue`. The FEM core, embedded cutting, prepared mechanics, partitioned needle coupling,
geometry queries and viewer helpers live under
`src/newton_solver_suturebot/`. Newton, Warp, MuJoCo and the other locked
packages are external dependencies installed normally by `uv`.

The thread solver under `_src/solvers/thread/` and the `needle_thread` example
were added on 2026-11-11. The solver was developed in the sibling
`newton-solver-ancf` working tree and is included here as self-contained code:
its element, mass, integration and floor-contact functions are local, and it
imports nothing from that package. On that tree's reference scenarios the two
implementations gave identical step histories on CPU. The example reuses the
bundled needle and adds no data. The original extraction reports below do not
cover it.

The thread's contact planes, friction, eyelets, self-contact and capsule
contact, the lagged mode of `ThreadAttachment`, `ThreadTissueCoupling`, the
cyclic reduction of the chain solve on CUDA and the `fem_needle_suture`
example were written in this repository on the same day. They reuse the bundled pad and needle, add no data and are not covered by the
extraction reports either. With these features left off and the line search's
multiples set to the reference's `(4, 2, 1, 0.5, 0.25, 0)`, the CPU solver
still gives the reference scenarios' step histories bit for bit; its default
line search has two smaller multiples and does not.

The `fem_tool_tissue` example, the rig of the contact between tools and
tissue, was written in this repository on 2026-11-16. It reuses the dVRK tools
and the tissue block of `fem_needle_grasp`, adds no data and is not covered by
the extraction reports.

The cutting workflow (`fem_surgical_incision` on the Baymax body; until
2026-11-17 `fem_phantom_cut` on the whole pad) uses the local cutting and material-grip
implementation. Its Line, Cylinder and Sphere tools use embedded Tet4
material fragments. The original
three-example extraction reports below retain their original scope and hashes.

## Bundled data

| Data | Local location and scope |
| --- | --- |
| Regional tissue geometry | [phantom_pad](../src/newton_solver_suturebot/examples/assets/phantom_pad/README.md): original tetrahedral coordinates, surface display and prepared twelve-region assignment |
| Curved needle | [needle_insertion](../src/newton_solver_suturebot/examples/assets/needle_insertion/README.md): standalone USD and provenance for the physical insertion example |
| Baymax tetrahedral body | [baymax](../src/newton_solver_suturebot/examples/assets/baymax/README.md): the body that `fem_surgical_incision` cuts, a byte-for-byte copy of `tet_baymax.usda` of the local `newton-solver-fem` repository at revision `6968f27`. The file carries no licence line; that repository attributes it to The Newton Developers under Apache-2.0; no upstream location is recorded; the shape is a film character |
| Recording scene and samples | [dVRK recordings](../src/newton_solver_suturebot/examples/assets/dvrk/README.md): prepared tools, fixture, cameras, phantom, motion, indexed source images and available point labels |
| Versius preview | [Preview guide](versius-recording.md): two supplied CMR distal needle holders, generated motion, illustrative stereo views and unrelated dVRK placeholder images |

The four prepared dVRK recording choices resolve their USD and array/image references
inside the packaged assets. Replay needs no original dataset checkout, download
or calibration pass. Source dataset identifiers and historical preparation
paths in JSON/README files are provenance records, not dependencies. Preparing a
new recording is a separate optional workflow requiring its source data; the
runtime loads a prepared `recording.json` and its references within the packaged
examples assets tree. `dvrk/models/` and `versius/models/` hold the shared platform
libraries. Recording folders retain motion, images, calibration and placement/UV
opinions; portable exports must include the complete tree. Fitted fixture and
legacy-tool variants remain explicit, and Versius reuses `phantom_pad/` volume
data rather than duplicating it inside the recording.

The `fem_dvrk` entry point assigns measured da Vinci Research Kit (dVRK) tool
poses from the SutureBot recordings kinematically over a static phantom.
It does not simulate tissue contact or needle insertion. Legacy
physical tool-contact helpers remain for validation and are not additional
launcher examples. The physical needle example instead advances a free rigid
body and regional FEM with two-way loads; it has no robot grasp, thread or knot.

`fem_versius` is a separate generated kinematic preview. The local source
checkout supplies CMR instrument CAD but not the numerical episodes or video
payloads described in its historical research. Its initial preparation lives in
`preparation/newton_versius/`; `preparation/newton_shared_assets/` produces the
shared-library layout after the instrument/material authoring stages. The runtime loads the
exported `assets/versius/recordings/versius_stereo_preview/recording.json` bundle.
It does not import the preparation code. The source CAD rights and transforms
are preserved; its absent proximal-arm meshes are not fabricated. Generated
motion, illustrative camera poses and dVRK placeholder videos do not establish
Versius calibration, image correspondence or task success. The original
extraction and dVRK replay reports do not validate this new preview.

## Demonstration media and history

### Current clips (2026-11-20)

The README shows one clip per example. Five were recorded on 2026-11-19 with
the example code of commit `640000d`, which commit `4b9fe78` left as it was.
Three were recorded on 2026-11-20 with the code of commit `fe9306e`, the
parent of the commit that brought the clips: `fem_needle_grasp`, whose
procedure had gained its stitch since, and `fem_needle_suture` and
`fem_needle_insertion`, recorded again with shorter texts in the recorder's
title bar, which now fit the 960 pixels of the picture. Of the five,
`fem_tool_tissue` and `needle_thread` were recorded once more from `fe9306e`
as a check and not replaced: the sidecars came out equal but for the wall
time and the pictures within the noise of two runs. The recorder's `camera`,
`edges` and `crop` entries, the scenario file `tools/video_scenarios.py` and
the recipes in `docs/videos.json` were committed together with the clips.
The ninth, `fem_surgical_incision`, was recorded again on 2026-11-20 from
commit `a67249d`, after the floor contact of that example was corrected,
and replaced the clip of 2026-11-19 in which the body sank into the floor.
Each clip is taken from a camera close on its action, the `camera` of its
recipe ([camera of a clip](development.md#camera-of-a-clip)), and shows the
example's default behaviour unless its arguments below or its caption name
an option.
The caption in the recipe and in the sidecar says what is seen and names the
known limits. A clip is three files: the MP4, its JPG poster, which the README
shows, and its JSON sidecar.

| Clip, sidecar | Picture | Frames | Simulated time | Example arguments | MP4 |
| --- | --- | ---: | ---: | --- | ---: |
| [fem_needle_grasp](media/fem_needle_grasp.mp4), [JSON](media/fem_needle_grasp.json) | 960×540 | 2,220 | 37.0 s | `--test` | 2.98 MB |
| [fem_tool_tissue](media/fem_tool_tissue.mp4), [JSON](media/fem_tool_tissue.json) | 960×540 | 1,668 | 27.7 s | `--contact swept --test` | 2.37 MB |
| [fem_needle_suture](media/fem_needle_suture.mp4), [JSON](media/fem_needle_suture.json) | 960×540 | 2,220 | 33.5 s | `--test` | 3.43 MB |
| [fem_needle_insertion](media/fem_needle_insertion.mp4), [JSON](media/fem_needle_insertion.json) | 960×540 | 420 | 3.8 s | `--regions 1 --transparent-pad --test` | 0.76 MB |
| [fem_surgical_incision](media/fem_surgical_incision.mp4), [JSON](media/fem_surgical_incision.json) | 800×660, cut from 1280×720 with the panel | 480 | 8.0 s | `--test` | 2.11 MB |
| [fem_dvrk](media/fem_dvrk.mp4), [JSON](media/fem_dvrk.json) | 1920×1080, with the panel | 1,202 | 20.0 s | none | 6.85 MB |
| [fem_versius](media/fem_versius.mp4), [JSON](media/fem_versius.json) | 1920×1080, with the panel | 240 | 4.0 s | none | 0.68 MB |
| [needle_thread](media/needle_thread.mp4), [JSON](media/needle_thread.json) | 1280×720 | 300 | 5.0 s | `--test` | 0.37 MB |
| [fem_phantom_regions](media/fem_phantom_regions.mp4), [JSON](media/fem_phantom_regions.json) | 1280×720, with the panel | 540 | 9.0 s | `--regions 4` | 1.08 MB |

- The picture size is an option of the recorder (`--width`, `--height`;
  960×540 without them), not an entry of the recipe. The panel and the overlay
  windows have fixed pixel sizes, so a clip of the table is recorded again
  with its size (commands below). The `crop` of the `fem_surgical_incision`
  recipe counts pixels of the 1280×720 picture: at a size that does not hold
  it, such as the default one, the recorder refuses the recipe before a
  simulation starts.
- The six recordings with `--test` ended with the example's self-test passed.
  `fem_dvrk`, `fem_versius` and `fem_phantom_regions` are recorded without it:
  with `--test` each plays its own scripted user, which ends the two replays
  paused at their last frame and presses and switches regions during the
  recorded drag of the third. The scenario file checks the state during those
  three recordings (`test_final` every 60 frames of a replay; at each
  selection and release of the regions clip); the start-up census of `--test`
  is not part of them.
- `fem_needle_grasp` has 2,220 frames, 37 s (1,700 before, of the procedure
  that ended with a lift by the needle tip): the procedure with the pull
  through the tissue and the stitch is done at about 35.3 s in this
  recording (at 35.2 to 35.6 s in the runs of
  [usage](usage.md#the-lock-and-the-slackening)). The marker of the first
  tie is yellow from frame 1,843 and that of the second from frame 2,015;
  the poster is frame 2,150, after the procedure is done, with both markers
  yellow and the slack thread from the exit to the needle.
- `fem_needle_suture` has 2,220 frames (2,760 before): the default run now
  ends in frame 2,007, with the closing pull ending in the frame it begins,
  and the clip holds the finished stitches for 3.5 s.
- The clips are encoded at 60 frames per second. Encoding/playback rate and
  the capture wall times of the sidecars (7 s to 106 s, with start-up,
  rendering and encoding) do not establish live simulation FPS.
- The 27 files of the nine clips are 21.6 MB together, 20.6 MB of them the
  MP4s. Two files are over 3 MB. `fem_dvrk.mp4` (6.85 MB; 5.12 MB before) is
  1920×1080 with the four video views, and a close view changes more of the
  picture from frame to frame than the wide view did. `fem_needle_suture.mp4`
  is 3.43 MB: its 180 frames with the mesh edges take 1.60 MB of it, because
  the fine lines over the whole picture compress badly. For the same reason
  the frames with the edges take 1.87 MB of the 2.11 MB of
  `fem_surgical_incision.mp4` and 0.58 MB of the 0.76 MB of
  `fem_needle_insertion.mp4`. `fem_needle_grasp.mp4` is 2.98 MB.
- The clips have no GIF preview. The README shows the JPG poster and links
  the MP4, no document showed a GIF, and the GIFs of the close views would be
  0.5 to 6.6 MB each, 28.5 MB together. The recorder writes one only with
  `--gif`. The GIF previews of the earlier clips left the tree with this
  change; the one GIF that remains is that of the archived replay quartet
  ([earlier clips](#earlier-clips)).

Three clips hold one moment in which the mesh is seen, the `edges` interval of
their recipes
([mesh edges and crop of a clip](development.md#mesh-edges-and-crop-of-a-clip)).
In the frames of the interval "Show Edges" of Newton's OpenGL viewer is on and
draws the meshes again as dark lines; in all other frames it is off. On the
tissue the lines are the edges of the outer faces of the tetrahedral mesh
itself, not of a finer drawing mesh: of the simulated tetrahedra on the Baymax
body and in region 1 of the pad, and of the same pad mesh around that region,
which these two examples do not simulate. The needle, the thread and the
marker spheres show the lines of their own drawing meshes. The three posters
are frames without the lines.

| Clip | Frames with the edges | Time in the clip | What is seen then |
| --- | --- | --- | --- |
| `fem_needle_suture` | 1,230 to 1,409 | 20.5 s to 23.5 s | The thread tightens through the second stitch and the tissue moves most, 2.3 mm (2.27 mm in this recording; 2.23 mm at frame 1,302 in a run of `fe9306e` without drawing). The edges on the pad are about 1 mm long; the thread is black under its lines. |
| `fem_surgical_incision` | 210 to 389 | 3.5 s to 6.5 s | The second grip pulls the wound open. The median edge is 1.9 mm; on the walls of the wound and at the body's outline the lines merge into dark hatching. |
| `fem_needle_insertion` | 120 to 251 | 2.0 s to 4.2 s | The needle goes through the first lip. The edges on the pad are about 1 mm long. The tissue moves 0.17 mm at most, so this moment shows how fine the mesh is around the needle more than a motion: at the puncture (2.2 s to 2.4 s) the lines of the first lip beside the entry shift by one to three pixels and go back. |

In the moment of `fem_needle_insertion` the pad is drawn opaque: the scripted
user of `tools/video_scenarios.py` switches the panel's "See through pad" off
for these frames and on again after them (`INSERTION_OPAQUE_FRAMES`, which a
test keeps equal to the recipe's `edges`). Through the translucent pad the
lines of its underside and of the slit walls lie over those of the top
surface, and single triangles cannot be told apart. The part of the needle
inside the tissue is hidden for that time.

The [Baymax cutting clip](media/fem_surgical_incision.mp4) is seen from
straight above. Its [sidecar](media/fem_surgical_incision.json) records 480
frames, eight simulated seconds: a planar blade cuts the belly along the
midline, 28.5 mm long and 13.5 mm deep, and two right-drag grips, one on each
side, are pulled 4 mm and 8 mm outward and stay as spring holds (green
markers) that keep the wound about 7.6 mm open (7.59 mm in this recording).
The clip is the recorder's own output: the blade and the two grips are
scripted in `tools/video_scenarios.py`, and the camera, the `edges` interval
and the `crop` are entries of the recipe in `docs/videos.json`. The picture is
the middle 800×660 of the 1280×720 recording, the part that the `crop` keeps.
The grip markers are drawn by the viewer's interface, so the recording has the
interface on (`show_ui`), and the crop cuts off its side panel, the counters
and the recorder's title bar. Nothing holds the body: it follows the first
grip. It rests on the floor, 0.27 mm under its plane at the deepest point;
the top of the belly comes down 2.4 mm before the cut as the soft body
settles on its round back. The clip was recorded on 2026-11-20 from commit
`a67249d`, after the floor contact was corrected; the clip before it showed
the body sinking 13 mm into the floor.
The encoding rate is 60 frames per second: simulation-time playback, not live
FPS. The clip of commit `4b9fe78` showed the same scene with the same crop,
cut with ffmpeg from an earlier recording of this scenario and camera, without
the moment with the mesh edges.

### Earlier clips

The three original clips (needle insertion, phantom regions and the dVRK
replay) were imported on 2026-11-10 with their GIF/JPG previews and JSON
sidecars. They retain their historical capture bytes and are archived under
`docs/media/history/<example>/`: the replay quartet since the standalone dVRK
capture of 2026-11-11; the MP4, JPG poster and JSON sidecar of the
needle-insertion and of the phantom-regions clip since 2026-11-20, when the
captures above took their places in `docs/media/`. The GIF previews of these
two were not archived: they left the tree with the other GIF previews, and
the import manifest lists them with their hashes under `removed`. Each
sidecar retains its original
module name, dependency names, parameters, source identifiers and capture wall
time. These fields have deliberately **not** been rewritten to imply a fresh
capture by `newton_solver_suturebot`.

The replay entry was renamed from `fem_suturebot` to `fem_dvrk`. The archived
MP4, JPG, GIF and JSON bytes are unchanged; their sidecar still records the original
example/module name. The import manifest uses the archived copied-file paths
while retaining each original path and hash.
Supplemental alignment/thread images and historical reports retain their names.

| Movie | Meaning of the recorded result |
| --- | --- |
| [Historical needle insertion](media/history/fem_needle_insertion/fem_needle_insertion.mp4) | Physical automatic insertion through two tissue sections, rendered in simulation time. Its earlier wall compliance was `1/60000` m²/N; the current example uses `1/300000` m²/N. |
| [Historical phantom regions](media/history/fem_phantom_regions/fem_phantom_regions.mp4) | Physical regional FEM and spring-grab response while changing the active region. |
| [Historical dVRK recording](media/history/fem_dvrk/fem_dvrk.mp4) | Original measured-motion replay and fitted camera/image overlays, not a physical suturing experiment. |

All three are encoded at 60 frames per second. Encoding/playback rate and capture
wall time do not establish live simulation FPS. The latest imported loaded
needle measurements remain below sustained 60 FPS. Camera/phantom alignment and
tissue/needle parameters remain estimates, with limitations in the individual
guides.

The other clips that the captures above replaced on 2026-11-20 were
standalone captures of this package. They are not archived in the tree and
remain in the repository's history up to commit `fe9306e`:

- the dVRK gallery movie of 2026-11-11 (metal-appearance preset, 1,202 frames
  at 1920×1080, the whole rig with four changes of view) and the Versius
  preview of the same day;
- the `needle_thread` clip of 2026-11-11 and the `fem_needle_suture` clip of
  2026-11-12 (2,760 frames; its caption described a closing pull that
  narrowed the wound from 2.4 mm to about 1.2 mm);
- the grasp and tool/tissue contact movies of 2026-11-17: 1,700 frames of the
  procedure that ended with a lift by the needle tip, without contact between
  tools and tissue, and 942 frames of the contact rig before it had the pinch;
- the full-pad cutting movie `fem_phantom_cut` of 2026-11-10, of the pad
  example that `fem_surgical_incision` replaced on 2026-11-17. Its four files
  left `docs/media/` on 2026-11-20 with the captures above, after the
  incision clip had been added in commit `4b9fe78`. The
  [cutting validation report](validation/standalone/phantom-cut.json) of the
  pad example still names the movie and its sidecar by their former paths and
  keeps its evidence separate from the original imported artifact hashes.

`docs/videos.json` stores the capture recipes. To record a clip again, use the
recorder with the picture size of the table; a recording into `docs/media`
regenerates the README:

```bash
uv run --locked --extra examples python tools/record_examples.py fem_needle_insertion
uv run --locked --extra examples python tools/record_examples.py fem_phantom_regions --width 1280 --height 720
uv run --locked --extra examples python tools/record_examples.py fem_surgical_incision --width 1280 --height 720
uv run --locked --extra examples python tools/record_examples.py fem_dvrk --width 1920 --height 1080
python3 tools/build_readme.py --check --require-videos
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
dependencies. Of the original 41 imported artifacts, 39 are in the tree and
remain byte-identical; the two GIF previews that left on 2026-11-20 are
listed under `removed` in the manifest with their paths and hashes.
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
