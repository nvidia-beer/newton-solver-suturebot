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
| Russell C. Jackson and M. Cenk Çavuşoğlu, **Modeling of Needle-Tissue Interaction Forces During Surgical Suturing**, ICRA 2012, pp. 4675–4680, DOI `10.1109/ICRA.2012.6224756`; supplied `nihms495965.pdf` | §III motivates rotation about a curved needle's center. §IV, Eq. 10 separates friction, compression and cutting; Eqs. 14–16 include their moments. `InsertionPlan` in [_needle_insertion_geometry.py](../src/newton_solver_suturebot/examples/needle/_needle_insertion_geometry.py) preserves authored curvature; `_surface_forces`, `_shaft_forces` and `_give_force` in [insertion.py](../src/newton_solver_suturebot/_src/solvers/coupled/interface/insertion.py) apply contact/friction and transfer moments. | The paper's swept-area spring and angular stiffness/cutting Eq. 13 are not reproduced; cutting resistance is omitted. FEM supplies tissue elasticity. The compact facet-pressure blend, tip-force threshold, shaft tangent regularization and regularized friction are implementation choices. The paper fitted slow planar CT-1 needle experiments on a Simulab SCS-10 phantom; those fitted parameters are not this pad's parameters. This PDF is Jackson 2012, not Okamura 2004. |
| Julien Lenoir, Philippe Meseure, Laurent Grisoni and Christophe Chaillou, **A Suture Model for Surgical Simulation**, ISMS 2004, pp. 105–113, DOI `10.1007/978-3-540-25968-8_12`; supplied `ISMS04-Lenoir.pdf` | §4 describes free-parameter sliding constraints, friction and opposite forces on the organ. `_shaft_forces` follows the same qualitative requirement that transverse resistance must permit tangential sliding. | This is a thread model. Its spline mechanics, multiplier system, Baumgarte stabilization and direction constraint are not reproduced. The needle example's ALM tunnel is a separate local-plane construction with projected Coulomb friction; there is no thread. |
| Tleukhan Mussin, Yafei Ou and Mahdi Tavakoli, **Simulation of Surgical Suturing Using Position-Based Dynamics and the Material Point Method for Robot Reinforcement Learning**, arXiv `2607.27494v1`, July 29, 2026; supplied `2607.27494v1.pdf` | §§III-C/D discuss a 1D needle manifold and exchanging contact momentum between different solvers. `NeedleTissueCoupler.step` in [coupling.py](../src/newton_solver_suturebot/_src/solvers/coupled/coupling.py) exchanges interface loads while advancing rigid and deformable states over a shared physical interval. | Neither MPM nor PBD is used here, and the Aitken interface iteration is an implementation choice. The paper defers its detailed rigid-needle contact model to CRESSim–MPM; that model is not claimed as implemented. Its Unity-unit thresholds, Lamé constants, timings and RL success rates do not validate this FEM example. |

The optional shaft preconditioner in
[shaft_preconditioner.py](../src/newton_solver_suturebot/_src/solvers/coupled/acceleration/shaft_preconditioner.py)
is also an implementation choice. It approximates transverse bearing and
regularized friction to accelerate the unchanged endpoint force equation;
its full reaction scatter includes constrained nodes. The combined surface/shaft
metric adds their curvatures in a single auxiliary system; the opt-in
preconditioned Anderson extension minimizes raw-residual secants while mixing
preconditioned search images. These are numerical implementation choices,
not a replacement physical model or an additional paper claim. The implicit drive in
[_needle_insertion_drive.py](../src/newton_solver_suturebot/examples/needle/_needle_insertion_drive.py)
uses backward Euler for held translation targets and an approximate rotational
linearization. Neither mechanism implements a paper's contact solver or
establishes material calibration. Their equations and limitations are recorded
in [method](method.md).

The [ALM virtual tunnel](../src/newton_solver_suturebot/_src/solvers/coupled/interface/tunnel_alm.py) is the
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

## Suture thread

The thread solver uses the element below; its integration, iteration matrix
and needle attachment are implementation choices described in
[method.md](method.md#suture-thread-and-its-needle-attachment).

| Paper | Implemented portion and code |
| --- | --- |
| Gerstmayr and Shabana, [Analysis of Thin Beams and Cables Using the Absolute Nodal Co-ordinate Formulation (2006)](https://doi.org/10.1007/s11071-006-1856-1) | The two-node cable element with position and slope coordinates, cubic Hermite interpolation, axial Green strain and curvature bending is evaluated by `evaluate_elements` in [kernels.py](../src/newton_solver_suturebot/_src/solvers/thread/kernels.py). The paper's time integration and examples are not reproduced; the tangent here is a positive semidefinite approximation. |
| Heller, [Some Aspects of the Cyclic Reduction Algorithm for Block Tridiagonal Linear Systems (1976)](https://doi.org/10.1137/0713042); the parallel variant, in which every level eliminates both neighbours of every node, follows Hockney and Jesshope, Parallel Computers (1981) | The solve of a chain's block-tridiagonal Newton system on CUDA devices: `begin_reduction`, `condense_nodes`, `reduce_nodes` and `solve_reduced` in [thread/kernels.py](../src/newton_solver_suturebot/_src/solvers/thread/kernels.py). CPU devices eliminate block by block. |

This is not the spline thread of Lenoir et al. listed above, and no paper is
claimed for the backward Euler step with a fixed update budget or for the
rigid head that carries the needle. None is claimed either for the contact
damping, the eyelets, the relaxed exchange of forces with the tissue or the
closing pull.


## Numerical response reuse

The physical-grasp response basis and Woodbury initial guess implement standard
linear algebra for low-rank contact updates. They do not introduce a new
physical contact model or claim a paper's convergence result. Approximate Warp
CG columns of a retained matrix initialize the existing current-matrix solves;
current-operator residual checks and the original physical acceptance remain
authoritative. The exact formulas, fallback conditions and cache scope are
specified in [method](method.md#physical-distal-grasp-and-worker-owned-thread).
