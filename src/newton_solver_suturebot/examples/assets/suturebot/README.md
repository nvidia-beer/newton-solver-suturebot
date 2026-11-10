# Prepared SutureBot rig

`suturebot_rig.usda` contains the two instrument articulations, a free needle,
a fixed trocar cage, table, wrist camera proxies and endoscope. Newton imports
the authored geometry, transforms, limits, drives, masses and collision shapes.
No CAD conversion or geometry generation happens in the example.

The endoscope, both wrist borescopes and their holders use 25% opacity,
authored in their USD materials and display primvars and imported directly by
Newton. The endoscope and borescope visuals are flat-ended cylinders. Their
visual geometry is separate from the hidden collision proxies; other shapes
remain opaque.

The USD uses metres and Z up, in the unchanged phantom frame. The runtime
camera looks from the native +Y lettering edge; the assets are not rotated.
There are 18
rigid bodies, 16 driven instrument coordinates, one world-fixed fixture joint
and a free needle joint added by the importer. Each instrument has RCM yaw and
pitch, insertion, roll, wrist pitch and yaw, and two symmetric jaw hinges.

The original dVRK LND meshes supply one instrument. The second uses an explicitly
documented LND-based DeBakey surrogate. The needle uses the source mesh with
32 capsule segments for collision, preserving its open curve. The cage uses
its original STL as a visual mesh and 80 capsule segments along CAD-derived
strut paths as collision proxies. Its ports remain open. The RCM positions
are fitted to the source cage's port centres under the stated illustrative
scale, not measured robot registration.

`manifest.json` records source hashes, transforms, joint limits and assumptions;
`validation.json` records USD import, joint closure and displaced-shaft RCM
checks. See [THIRD_PARTY.md](THIRD_PARTY.md) and [the dVRK license](recordings/tissue3_215300/DVRK_MODEL_LICENSE.txt)
for source attribution and retained notices.

## Run the bundled scene

From the `newton-solver-suturebot` repository root:

```bash
uv run --locked --extra examples python -m newton_solver_suturebot.examples fem_dvrk
```

The packaged recordings need no preparation checkout or external dataset.
Their `audit/**/history/` directories preserve original documentation and
commands verbatim; those historical commands describe the predecessor package,
not the current launcher.

## Historical preparation provenance

The original preparation modules were located at
`/home/beer/Dev/MSc/SutureBot-workbench/suturebot/simulation`. This is the
recorded source location, not a runtime requirement. Optional regeneration
uses that separate workbench's `suturebot.simulation.prepare_rig` and
`suturebot.simulation.validate_rig` tools and their preparation dependencies;
follow the workbench's environment instructions rather than a sibling solver's
virtual environment.

The validator checks the input hashes, USD transforms, positive masses and
inertias, original visual mesh separation from collision geometry, actual
Newton import, collision filtering, and RCM invariance after moving all the
instrument joints. Copy the validated outputs into this directory to update
the runtime bundle. Keep the original CAD inputs and previous preparation
outputs in MSc.

This rig is the approximate arrangement retained for the legacy contact
experiment. The source STL has no declared units; its scale is an explicit
preparation assumption. That experiment models contact without needle puncture
or thread mechanics.

The current `fem_dvrk` example loads the self-contained samples under
`recordings/`, including their camera rays and source images. Each sample
documents its estimated calibration and replay scope separately.
