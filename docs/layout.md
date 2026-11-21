# Repository layout

The implementation, examples and required assets are included in this repository.

Platform geometry and materials are authored once in `examples/assets/dvrk/models/`
and `examples/assets/versius/models/`. Recordings keep motion, images, calibration,
small placement layers and point-label reports. Relative USD references join
them; fitted fixture and legacy-tool variants stay explicit. The Versius preview
references the existing `examples/assets/phantom_pad/` volume data. Copy the full
examples assets tree for a portable export, preserving these relative paths.

| Location | Contents |
| --- | --- |
| [FEM solver](../src/newton_solver_suturebot/_src/solvers/fem/) | Sparse tissue mechanics, material kernels, fixed boundaries, prepared steps and the retained differentiation API |
| [Cutting](../src/newton_solver_suturebot/_src/cutting/) | Embedded Tet4 fragments, cut topology, streaming assembly inputs, contact and material grips; in `tissue.py` the cut simulation as one object, `CutTissue`, that a cutting scene is built on or owns: the frame and its graph, the start-up, the edit transaction with its roll-back, storage growth and the fixed reserve |
| [Rigid MuJoCo](../src/newton_solver_suturebot/_src/solvers/mujoco/) | Subcycled rigid integration inside a coupled interval: native MuJoCo, and MuJoCo Warp, an unmodified dependency, with the solver's own rigid step in [`warp_step/`](../src/newton_solver_suturebot/_src/solvers/mujoco/warp_step/) |
| [Thread solver](../src/newton_solver_suturebot/_src/solvers/thread/) | Backward Euler chain of cable elements for a suture thread with friction and self-contact, its attachment to a MuJoCo rigid body, its contact with a body's capsules and its two-way coupling to FEM tissue |
| [Needle coupling](../src/newton_solver_suturebot/_src/solvers/coupled/) | Partitioned needle/tissue coupling in `coupling.py`, its captured trial graph in `coupling_graph.py`, interface forces in `interface/`, search directions and start guesses in `acceleration/` |
| [Geometry](../src/newton_solver_suturebot/_src/geometry/) | USD mesh and region loading, needle geometry and its queries, mesh refit |
| [Pose control](../src/newton_solver_suturebot/_src/sim/ik/) and [graph helpers](../src/newton_solver_suturebot/_src/utils/) | Pose inverse kinematics of the tool models; CUDA graph capture helpers |
| [Examples](../src/newton_solver_suturebot/examples/) | Nine runnable examples in five folders: `phantom/` (regional tissue FEM, and the screen tools of the cutting example), `incision/` (cutting the Baymax body, with its scene content `_baymax_body.py`), `needle/` (physical needle insertion, a needle on a thread, a sutured row of stitches), `grasp/` (a needle held and inserted by articulated dVRK tools, the rig of the tool/tissue contact with tools and tissue alone, and the parallel scene pool `parallel.py`) and `recording/` (da Vinci Research Kit (dVRK) replay, Versius instrument preview); `_startup.py` is the start-up check and `_tissue_markers.py` the sphere markers bound to the tissue surface that examples of several folders share |
| [Example assets](../src/newton_solver_suturebot/examples/assets/) | Needle and phantom geometry, prepared recording scenes, motion arrays, camera data and images |
| [Tests](../tests/) | Solver, contact, rollback, asset and example checks |
| [Documentation](usage.md) | Setup, controls, implemented equations, paper mappings and scoped validation evidence |

The default needle path advances tissue and rigid-body physics. The dVRK
recording example replays measured tool motion over a static phantom. The
[Versius preview](versius-recording.md) uses generated instrument motion and
explicitly labeled dVRK placeholder images; it is not measured Versius replay.
Their videos are playback artifacts, not live-speed measurements. Fresh
standalone audit reports are kept under `docs/validation/standalone/`.
