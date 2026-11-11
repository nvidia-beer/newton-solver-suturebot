# Repository layout

The implementation, examples and required assets are included in this repository.

Platform geometry and materials are authored once in `examples/assets/dvrk/models/`
and `examples/assets/versius/models/`. Recordings keep motion, images, calibration
and small scene layers for placement and UV labels. Relative USD references join
them; fitted fixture and legacy-tool variants stay explicit. The Versius preview
references the existing `examples/assets/phantom_pad/` volume data. Copy the full
examples assets tree for a portable export, preserving these relative paths.

| Location | Contents |
| --- | --- |
| [FEM solver](../src/newton_solver_suturebot/_src/solvers/fem/) | Sparse tissue mechanics, material kernels, fixed boundaries, prepared steps and the retained differentiation API |
| [Cutting](../src/newton_solver_suturebot/_src/cutting/) | Embedded Tet4 fragments, cut topology, streaming assembly inputs, contact and material grips |
| [MuJoCo/FEM solver](../src/newton_solver_suturebot/_src/solvers/mujoco_fem/) | Optional simultaneous needle momentum and friction solve |
| [Needle coupling](../src/newton_solver_suturebot/_src/coupling.py) | Default partitioned coupling, with local interaction, tunnel and geometry modules alongside it |
| [Examples](../src/newton_solver_suturebot/examples/) | Five runnable examples: full-pad cutting, regional tissue FEM, physical needle insertion, da Vinci Research Kit (dVRK) replay and Versius instrument preview |
| [Example assets](../src/newton_solver_suturebot/examples/assets/) | Needle and phantom geometry, prepared recording scenes, motion arrays, camera data and images |
| [Tests](../tests/) | Solver, contact, rollback, asset and example checks |
| [Documentation](usage.md) | Setup, controls, implemented equations, paper mappings and scoped validation evidence |

The default needle path advances tissue and rigid-body physics. The dVRK
recording example replays measured tool motion over a static phantom. The
[Versius preview](versius-recording.md) uses generated instrument motion and
explicitly labeled dVRK placeholder images; it is not measured Versius replay.
Their videos are playback artifacts, not live-speed measurements. Fresh
standalone audit reports are kept under `docs/validation/standalone/`.
