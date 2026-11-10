# Papers mapped to this implementation

| Paper | Implemented portion and code |
| --- | --- |
| Smith, de Goes and Kim, [Stable Neo-Hookean Flesh Simulation (2018)](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf) | The regularized isotropic energy and rest correction are used by the Tet4 force kernels in [kernels.py](../src/newton_solver_suturebot/_src/solvers/fem/kernels.py) and its Q1 `eval_hexahedra` kernel. Stretch/determinant clamps and the approximate PSD tangent are implementation choices; this is not the paper's full nonlinear solve. |

## Embedded cutting and streaming assembly

The full-pad example uses the local embedded cutting implementation. These mappings describe the implemented portions, not a
reproduction of either paper's complete solver or performance results.

| Paper | Implemented portion and code |
| --- | --- |
| Sifakis, Der and Fedkiw, [Arbitrary Cutting of Deformable Tetrahedralized Objects (2007)](https://pages.cs.wisc.edu/~sifakis/papers/cutting.pdf), §3 | [LinearTetCut](../src/newton_solver_suturebot/_src/cutting/tetrahedral_cut.py) clips embedded material regions. [CutModel](../src/newton_solver_suturebot/_src/cutting/model.py) retains parent affine basis functions with duplicated nodes, fragment volume weights and barycentric mass integrals. Cuts are prescribed geometric edits, not a calibrated fracture criterion. |
| Acharya and Hyde, [STA-FEM (2026)](https://arxiv.org/abs/2605.11673), §3.4 and Algorithm 1 | [StreamingTetAssembly](../src/newton_solver_suturebot/_src/solvers/fem/streaming.py) maintains additive element blocks under local topology changes. It adapts the element-assembly idea to reserved copies of basis nodes; it does not implement the paper's graph-Laplacian proxy or benchmark. |

`SolverFEM.prepare_tetrahedral_updates` reserves sparse connections before
capture. A proposed cut stages fragments and connectivity; capacity checks
precede publication. `StreamingTetAssembly.update` subtracts cached old element
blocks, adds changed contributions and compacts active rows in persistent
storage. The existing sparse Krylov solve advances the result; no dense inverse
or Woodbury solve is introduced. Deformation and material changes also affect
element tangents, so cached blocks can be reused only when those inputs agree.
The [method](method.md#embedded-cuts-and-material-grips) describes the physical
and differentiation limits.

## Needle insertion literature

These references inform the implemented needle model.
The source PDFs are not redistributed here; citations identify their provenance. They inform the
[insertion approximation](needle-insertion.md); none supplies calibrated
properties for the bundled pad or needle.

In the needle interaction, cutting resistance and the work of forming a passage
are omitted; these papers' needle cutting models are not implemented. The
separate full-pad cutting example above edits Tet4 material geometry.

| Source | Paper content and connection to code | Boundary of the claim |
| --- | --- | --- |
| Russell C. Jackson and M. Cenk Çavuşoğlu, **Modeling of Needle-Tissue Interaction Forces During Surgical Suturing**, ICRA 2012, pp. 4675–4680, DOI `10.1109/ICRA.2012.6224756`; supplied `nihms495965.pdf` | §III motivates rotation about a curved needle's center. §IV, Eq. 10 separates friction, compression and cutting; Eqs. 14–16 include their moments. `InsertionPlan` in [_needle_insertion_geometry.py](../src/newton_solver_suturebot/examples/_needle_insertion_geometry.py) preserves authored curvature; `_surface_forces`, `_shaft_forces` and `_give_force` in [insertion.py](../src/newton_solver_suturebot/_src/insertion.py) apply contact/friction and transfer moments. | The paper's swept-area spring and angular stiffness/cutting Eq. 13 are not reproduced; cutting resistance is omitted. FEM supplies tissue elasticity. The compact facet-pressure blend, tip-force threshold, shaft tangent regularization and regularized friction are implementation choices. The paper fitted slow planar CT-1 needle experiments on a Simulab SCS-10 phantom; those fitted parameters are not this pad's parameters. This PDF is Jackson 2012, not Okamura 2004. |
| Julien Lenoir, Philippe Meseure, Laurent Grisoni and Christophe Chaillou, **A Suture Model for Surgical Simulation**, ISMS 2004, pp. 105–113, DOI `10.1007/978-3-540-25968-8_12`; supplied `ISMS04-Lenoir.pdf` | §4 describes free-parameter sliding constraints, friction and opposite forces on the organ. `_shaft_forces` follows the same qualitative requirement that transverse resistance must permit tangential sliding. | This is a thread model. Its spline mechanics, multiplier system, Baumgarte stabilization and direction constraint are not reproduced. The needle example's ALM tunnel is a separate local-plane construction with projected Coulomb friction; there is no thread. |
| Tleukhan Mussin, Yafei Ou and Mahdi Tavakoli, **Simulation of Surgical Suturing Using Position-Based Dynamics and the Material Point Method for Robot Reinforcement Learning**, arXiv `2607.27494v1`, July 29, 2026; supplied `2607.27494v1.pdf` | §§III-C/D discuss a 1D needle manifold and exchanging contact momentum between different solvers. `NeedleTissueCoupler.step` in [coupling.py](../src/newton_solver_suturebot/_src/coupling.py) exchanges interface loads while advancing rigid and deformable states over a shared physical interval. | Neither MPM nor PBD is used here, and the Aitken interface iteration is an implementation choice. The paper defers its detailed rigid-needle contact model to CRESSim–MPM; that model is not claimed as implemented. Its Unity-unit thresholds, Lamé constants, timings and RL success rates do not validate this FEM example. |

The optional shaft preconditioner in
[shaft_preconditioner.py](../src/newton_solver_suturebot/_src/shaft_preconditioner.py)
is also an implementation choice. It approximates transverse bearing and
regularized friction to accelerate the unchanged endpoint force equation;
its full reaction scatter includes constrained nodes. The implicit drive in
[_needle_insertion_drive.py](../src/newton_solver_suturebot/examples/_needle_insertion_drive.py)
uses backward Euler for held translation targets and an approximate rotational
linearization. Neither mechanism implements a paper's contact solver or
establishes material calibration. Their equations and limitations are recorded
in [method](method.md).

The [ALM virtual tunnel](../src/newton_solver_suturebot/_src/tunnel_alm.py) is the
needle example's default; the generic public interaction retains penalty
contact by default. It evaluates the finite-compliance radial
reaction analytically and uses ALM for axial Coulomb friction at accepted
barycentric track anchors. Zero compliance retains the transverse
clearance-disk normal proximal update. Physical wall compliance is separate
from the numerical penalty. The analytic branch matches the local normal
constitutive fixed point; it does not establish identical discrete engagement
histories or global trajectory equivalence.
This is not the spline/thread solver of Lenoir or a reproduction of the cited
papers' contact algorithms. Its local residual and transactional tests do not
imply global energy stability, a resolved cavity or calibrated tissue behavior.

The brief also names Okamura et al. (2004), Duriez et al. (2009) and a
curved-needle planning reference. Those full papers were not among the three
supplied PDFs; no claim of reproducing their formulations is made. In
particular, the embedded track is not a resolved cavity, fracture or validated
damage model, and the coupling is not the cited complementarity formulation.

## Simultaneous momentum/friction solve

`SolverMuJoCoFEM` combines prepared linearly implicit tissue mechanics,
the supported MuJoCo free-body update and axial tunnel friction unknowns.
The finite-compliance normal reaction is eliminated analytically. The
projection equation and its sticking/sliding derivatives are inspired by
primal-dual contact methods; the present solver is not a reproduction of
either paper below.

| Primary source | Connection to this implementation | Boundary of the claim |
| --- | --- | --- |
| Stefan Hüeber, Georg Stadler and Barbara I. Wohlmuth, [A Primal-Dual Active Set Algorithm for Three-Dimensional Contact Problems with Coulomb Friction](https://math.nyu.edu/~stadler/papers/friction3d.pdf), SIAM Journal on Scientific Computing 30(2), 572–596 (2008), DOI `10.1137/060671061` | The paper derives contact/friction active sets from nonsmooth equations and treats the solution-dependent Coulomb bound. `linearize_anchors` and `contact_action` in [kernels.py](../src/newton_solver_suturebot/_src/solvers/mujoco_fem/kernels.py) likewise distinguish sticking/sliding and retain the normal-force derivative in the sliding bound. | The paper's mortar discretization, two-component surface friction, multigrid strategy and convergence results are not implemented or transferred to this moving one-dimensional virtual tunnel. Its exact Newton conclusions do not apply to the frozen geometric derivatives or search regularization here. |
| Stéphane Abide, Mikaël Barboteu, Soufiane Cherkaoui and Serge Dumont, [Unified primal-dual active set method for dynamic frictional contact problems](https://link.springer.com/article/10.1186/s13663-022-00729-4), Fixed Point Theory and Algorithms for Sciences and Engineering 2022, 19, DOI `10.1186/s13663-022-00729-4` | The dynamic formulation relates contact reactions, motion and a semismooth/active-set treatment of Coulomb conditions. [trials.py](../src/newton_solver_suturebot/_src/solvers/mujoco_fem/trials.py) solves velocity increments and friction density together rather than nesting a converged motion solve inside each friction update. | The paper's unilateral impact/sweeping formulation and its particular PDAS algorithms are not reproduced. The scalar projection, compliant radial wall, prepared FEM equation, GMRES preconditioner and transactional event model are this implementation's choices. |
| [MuJoCo numerical integration documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html#numerical-integration) | `rigid_system` and `reconstruct_rigid` in [kernels.py](../src/newton_solver_suturebot/_src/solvers/mujoco_fem/kernels.py) assemble/reconstruct the supported `implicitfast` free-body equations; [mujoco_free_body_gpu.py](../src/newton_solver_suturebot/_src/mujoco_free_body_gpu.py) supplies the canonical device step and [coupling.py](../src/newton_solver_suturebot/_src/coupling.py) audits its endpoint against native MuJoCo. | Native parity is checked for the guarded single zero-COM free-body configuration, not arbitrary MuJoCo contacts, articulated systems, actuators or callbacks. The needle/tissue contact law remains external to MuJoCo. |

The implementation-specific equations and acceptance procedure are in
[method](method.md#simultaneous-mujocofem-momentum-and-friction):

| Operation | Code and verification scope |
| --- | --- |
| Prepare the affine tissue equation and preserve fixed boundaries | `SolverFEM.prepare_step` and [PreparedFEMStep](../src/newton_solver_suturebot/_src/solvers/fem/prepared_step.py) expose RHS assembly, sparse action and endpoint reconstruction without a second physical step. |
| Evaluate motion and projected Coulomb residuals | `momentum_residual`, `friction_residual` and `predict_tangent` in [kernels.py](../src/newton_solver_suturebot/_src/solvers/mujoco_fem/kernels.py); actual forces still use [tunnel_alm.py](../src/newton_solver_suturebot/_src/tunnel_alm.py) and the existing surface law. The sliding predictor only initializes numerical unknowns. |
| Build the joint search direction | `linearize_anchors`, `contact_action` and `precondition_friction`, invoked by [trials.py](../src/newton_solver_suturebot/_src/solvers/mujoco_fem/trials.py). The search freezes geometric derivatives, omits the surface-force derivative and adds a sticking-only diagonal $\epsilon=0.01$; none changes the physical residual. |
| Globalize and certify | [control.py](../src/newton_solver_suturebot/_src/solvers/mujoco_fem/control.py) uses frozen-per-search block-weighted residual backtracking. A proposal passing the physical force/ALM gates receives one prepared affine FEM correction under its balanced applied load, warm-started from the joint increment. The corrected endpoint must pass true applied-load FEM relative L2 accuracy, both applied/fresh momentum force bounds, and fresh interface/ALM gates. Fresh-law relative FEM L2 is diagnostic, not an extra acceptance condition. A failed certificate may resume within the shared budget, so an accepted step can use multiple final corrections. Native endpoint audit, swept-motion checks and rollback precede publication. |
| Check local equations independently | [test_mujoco_fem_equations.py](../tests/test_mujoco_fem_equations.py) compares dense references, finite differences and Warp actions, including force/moment transfer, native frame scaling, search regularization, predictor history and block preconditioning. These local checks do not establish full trajectory equivalence or performance. |

The use of projection equations does not confer a general semismooth-Newton
convergence guarantee on this quasi-Newton implementation. Prepared FEM still
uses its selected material linearization; contact topology is committed only
after acceptance. Differentiation through the joint solve is unsupported.
