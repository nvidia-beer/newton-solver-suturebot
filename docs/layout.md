# Repository layout

The implementation, examples and required assets are included in this repository.

| Location | Contents |
| --- | --- |
| [FEM solver](../src/newton_solver_suturebot/_src/solvers/fem/) | Sparse tissue mechanics, material kernels, fixed boundaries, prepared steps and the retained differentiation API |
| [Cutting](../src/newton_solver_suturebot/_src/cutting/) | Embedded Tet4 fragments, cut topology, streaming assembly inputs, contact and material grips |
| [MuJoCo/FEM solver](../src/newton_solver_suturebot/_src/solvers/mujoco_fem/) | Optional simultaneous needle momentum and friction solve |
| [Needle coupling](../src/newton_solver_suturebot/_src/coupling.py) | Default partitioned coupling, with local interaction, tunnel and geometry modules alongside it |
| [Examples](../src/newton_solver_suturebot/examples/) | Four runnable examples: full-pad cutting, regional tissue FEM, physical needle insertion and da Vinci Research Kit (dVRK) replay |
| [Example assets](../src/newton_solver_suturebot/examples/assets/) | Needle and phantom geometry, prepared recording scenes, motion arrays, camera data and images |
| [Tests](../tests/) | Solver, contact, rollback, asset and example checks |
| [Documentation](usage.md) | Setup, controls, implemented equations, paper mappings and scoped validation evidence |

The default needle path advances tissue and rigid-body physics. The dVRK
recording example instead replays measured tool motion over a static phantom.
Their videos are playback artifacts, not live-speed measurements. Fresh
standalone audit reports are kept under `docs/validation/standalone/`.
