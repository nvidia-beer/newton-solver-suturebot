# Sparse FEM method

`SolverFEM` advances position-only Tet4 and Q1 Hex8 solids in SI units.
Tet4 uses constant reference gradients; Hex8 uses $2\times2\times2$ Gauss
quadrature. Both assemble $3\times3$ blocks into a sparse BSR matrix. “Linear”
refers to the shape functions; the constitutive force is nonlinear.

The [paper-to-code mapping](papers.md) identifies the implemented portions of
Stable Neo-Hookean elasticity, embedded cutting and the implemented
needle-contact formulations. FEM, cutting and needle-specific solvers are
copied locally under this package; there is no import from another solver
checkout.

## Elasticity and stepping

For deformation gradient $F$, $I_C=\mathrm{tr}(F^T F)$, $J=\det F$, and
Lamé coefficients $\mu,\lambda>0$, the smooth-branch energy is

$$
\psi(F)=\frac\mu2(I_C-3)-\frac\mu2\log(I_C+1)
 +\frac\lambda2(J-\alpha)^2,\qquad
\alpha=1+\frac{3\mu}{4\lambda}.
$$

Its stress is $P=\mu(1-1/(I_C+1))F+\lambda(J-\alpha)\mathrm{cof}\,F$.
The force evaluates clamped principal stretches (default $[0.05,20]$) and
replaces $J$ by $\max(J,0.01)$ in the volumetric stress. Thus the displayed
energy derivative applies on the unclamped branch. Viscous forces are explicit.
These regularizations do not guarantee positive integrated element volumes.

With lumped mass $M$, step $h$, approximate stiffness $\widetilde K$, and
assembled spring damping $C_s$, each substep solves

$$
A=M+h^2\widetilde K+hC_s,\qquad
A\Delta v=h(f_{\rm elastic}+f_{\rm viscous}+f_{\rm contact}+f_{\rm ext}+Mg)
-h^2\widetilde K v,
$$

$$
v^+=v+\Delta v,\qquad x^+=x+h v^+.
$$

The omitted geometric Hessian terms make this a linearly implicit update,
not a converged nonlinear equilibrium. Block-Jacobi Krylov solves never form
a dense global matrix. The examples choose their own material, tangent-refresh
and substep settings; those settings are not interchangeable accuracy presets.

The fast-path Krylov relative residual tolerance is the public
`linear_solver_tolerance` setting, defaulting to $10^{-2}$. The needle coupling
example initially sets $3\cdot10^{-3}$. Historical penalty-branch full-state
comparisons checked $10^{-4}$ against $10^{-5}$, $10^{-3}$ against $10^{-4}$,
and $3\cdot10^{-3}$ against $10^{-3}$; these are not an ALM accuracy theorem.
`--fem-tolerance 1e-5` retains the stricter reference.
This does not alter the differentiable branch's fixed-matrix solve contract.
Within a coupled needle solve, eight unresolved primal iterations trigger a
tenfold tightening of that tolerance, down to $10^{-6}$ or the caller's tighter
setting. Each tightening restarts Aitken history; the configured tolerance is
restored after acceptance or rejection. This resolves force-map noise without
relaxing the interface acceptance criterion or changing a slider target.

The non-differentiable CPU CG path exposes the same operator as one public
Warp reduction batch spanning all scalar degrees of freedom. This avoids
redundant lane work in the installed Warp CPU tiled reduction; the matrix,
preconditioner, right-hand side, iteration cap and tolerance stay unchanged.
Existing batched operators, other Krylov methods and CUDA retain their original
paths. Reduction order can differ, so parity is checked against the physical
solution and true residual rather than bitwise equality.

Ground contact is an analytic penalty with inward normal damping and velocity
friction capped by both Coulomb force and particle weight. Surface contact
uses evolving boundary geometry. External forces enter through
`step(..., external_forces=forces)`; incoming `particle_f` is workspace.

Stationary boundaries use `SolverFEM(..., fixed_particle_indices=indices)`.
With $P$ the projector onto free nodes, the solve is
$(PAP+I-P)\Delta v=Pb$. Boundary positions remain at the model's construction
positions and boundary velocities are zero. Physical masses and the original
matrix remain intact, including the constrained columns needed by the elastic
velocity predictor. Boundary state entries are restored before force evaluation
and after integration; tangent/topology reassembly also restores fixed positions.
This option currently excludes differentiable stepping,
streaming topology and component scaling; no gradient support is claimed for it.

On CUDA, a separate sparse value buffer stores $PAP+I-P$ for the
guarded sparse multiply described below. Its values refresh once per solve, including captured solves;
changed sparse storage rebuilds the cached operator. The original $A$ remains
available to the predictor. The region-1 needle example adds about 3.2 MB of
matrix values while sharing index buffers. CPU and unsupported sparse layouts
retain the masked multiply.

The CUDA needle path uses a fixed-degree Chebyshev polynomial of the block
Jacobi inverse as a CG preconditioner. Let $D$ contain the $3\times3$ diagonal
blocks of the constrained free-free matrix and $B=D^{-1}A$. Its interval upper
endpoint is

$$
b=\min\left(\max_i\sum_j\|B_{ij}\|_F,
             \max_{i,a}\sum_{j,c}|(B_{ij})_{ac}|\right).
$$

Both terms bound the spectral radius: the first bounds the induced block
infinity/Euclidean norm; the second is the scalar infinity norm. Taking their
global minimum preserves that bound. Both exclude fixed rows and columns and
refresh with the current tangent in graph-stable storage. The lower endpoint
is $b/256$; the needle example retains degree four. This changes only the
preconditioner coefficients. The sparse physical operator, load, warm start,
CG relative tolerance, maximum iterations and coupling acceptance checks stay
unchanged. No unconstrained power/Ritz estimate substitutes for an upper bound.

For supported non-differentiable CUDA `mat33`/`vec3` fixed-particle
operators, [packed_bsr.py](../src/newton_solver_suturebot/_src/solvers/fem/packed_bsr.py)
uses 16 cooperating threads per block row, with several rows packed into a
256-thread block. Each thread accumulates its assigned sparse blocks; subgroup
shuffles reduce the three result components. Padded threads participate in all
shuffles. Matrix values, row counts, constrained-node handling and the affine
`alpha*A*x + beta*y` contract are unchanged. Persistent snapshots preserve
supported input/output aliasing during capture. The reduction order differs
from Warp's original implementation, so numerical equality of operators does
not establish bitwise trajectory equivalence. CPU, unsupported layouts,
gradient-enabled operands and active Tape execution use the original operator.

On the exact free-node compact operator only,
[fused_chebyshev.py](../src/newton_solver_suturebot/_src/solvers/fem/fused_chebyshev.py)
fuses each sparse multiply with the following Jacobi recurrence. The original
first step, degree, coefficients and full-indexed spectral-bound calculation
remain unchanged. Two disjoint iterate buffers alternate; their initial parity
places the final result in the requested output without an extra copy. No
thread overwrites a neighbor's input during a launch. Matrix and scratch owners
remain alive with the cached preconditioner. Full/projected or unknown indexing
mappings, alias conflicts and differentiable execution retain the original
Chebyshev implementation. This reduces launches without changing the Krylov
residual checks, iteration limits, physical tangent or accepted boundary state.

### Prepared FEM mechanics

`SolverFEM.prepare_step(state, control, contacts, h)` exposes one attempt's
affine mechanics through
[PreparedFEMStep](../src/newton_solver_suturebot/_src/solvers/fem/prepared_step.py).
Preparation copies the accepted state, evaluates the ordinary forces once,
and retains the elastic velocity predictor. The prepared equation is
$A_D\Delta v=b(f_{\rm ext})$, with the same constrained sparse operator,
load-before-predictor arithmetic, fixed boundaries and integration as `step`.
`base_rhs` is the zero-additional-load impulse in N s; `rhs(forces)` accepts
additional nodal forces in N. `apply_operator`, `apply_preconditioner`, and
`reconstruct` are GPU operations that do not advance physical history.

The caller refreshes the material tangent separately, as for ordinary stepping,
and prepares again after any accepted state, timestep, material or support
change. One owned workspace is reused per FEM solver. Optional `prepare_solve`
and `solve` supply a reference Krylov solution; a simultaneous integrator may
instead solve the prepared equation directly. These operations currently
exclude differentiation, streaming topology and component scaling. They do
not change the existing `SolverFEM.step` or fixed-matrix adjoint contract.

## Embedded cuts and material grips

The full-pad cutting example uses [CutModel](../src/newton_solver_suturebot/_src/cutting/model.py)
and [LinearTetCut](../src/newton_solver_suturebot/_src/cutting/tetrahedral_cut.py).
Cuts clip material into convex fragments while preserving each parent Tet4's
affine basis. Disconnected material regions receive copied basis nodes.
Fragment volumes weight elastic energy; barycentric first moments determine
mass. Thus a cut changes the simulated connectivity and remaining material,
not just the displayed surface. This prescribed geometric cutting does not
model a blade's fracture energy or a calibrated tissue failure criterion.

[StreamingTetAssembly](../src/newton_solver_suturebot/_src/solvers/fem/streaming.py)
subtracts cached element blocks and adds changed contributions in reserved
sparse adjacency. Active rows are compacted without moving captured storage.
Capacity failures roll back. Small components use mass scaling and a bounded
volume guard limits distortion; neither the guard nor penalty contact
guarantees freedom from inversion or self-intersection. The prepared-step and
fixed-particle projection APIs described above do not support streaming cut
topology.

The floor and the contact between cut pieces
([CutSurfaceContact](../src/newton_solver_suturebot/_src/cutting/contact.py))
are area-weighted penalties on the corners of the boundary triangles: a row
with stiffness $k$ [N/m] (the penalty per area times a third of its triangle)
and damper $c$ pushes with $-k\,(g + h\,\dot g) - c\,\dot g$ for its gap $g$ and
time step $h$. Like a grip, the row is linearly implicit in the step: its
tangent $h^2 k + h c$ along the contact normal stands on the diagonal of the
system matrix for the solve, as a bound by row sums
([SolveDiagonal](../src/newton_solver_suturebot/_src/solvers/fem/solve_diagonal.py)),
and multiplies the velocity change there. A body at rest is therefore carried
at the depth of its contact pressure over the penalty per area, whatever the
time step. A row is in a step while its point is under the surface at the
start of the step or would be at its end; within the step it is linear and is
not clamped, so it pulls for the rest of the one step in which its point
leaves. Dividing the row's force by $1 + h^2 k/m + h c/m$ instead (the form
until 2026-11-20) leaves the balance of a resting body wrong by that factor,
about 500 for the Baymax body.

The floor has Coulomb friction with the coefficient $\mu$ of the scene
(`ground_friction`; zero is a floor without friction and the contact as it was,
bit for bit). The law is that of the thread on its contact planes
([Friction on a contact plane](#suture-thread-and-its-needle-attachment)). A
floor row whose push $N=-k\,(g+h\,\dot g)-c\,\dot g$ is positive is tied to an
anchor $\mathbf a$ on the floor by a spring of the row's own stiffness $k$, whose
force stops at $\mu N$. The row sticks within the reach $\rho=\mu N/k$ of its
anchor, which at rest is $\mu$ times its depth, and slides beyond it. With the
row's point $\mathbf p$, its velocity $\mathbf u$ and the offset at the end of
the step if that velocity stayed,
$\mathbf d=(\mathbf I-\mathbf n\mathbf n^T)(\mathbf p+h\,\mathbf u-\mathbf a)$,

$$
\mathbf f=-s\,\mathbf d,\qquad s=k\,\min\!\big(1,\;\rho/|\mathbf d|\big),
$$

and $h^2 s\,(\mathbf I-\mathbf n\mathbf n^T)$ stands in the system matrix beside
the push's $(h^2k+hc)\,\mathbf n\mathbf n^T$, with the same bound by row sums. It
is the tangent of the spring that has the force $\mathbf f$ at the offset
$\mathbf d$, as in the thread's Newton matrix, and positive semidefinite, so the
system matrix stays positive definite. Before the force, the anchor of a row
that the last step left further from it than that step's reach is moved to that
reach behind the row, and a row that had no grip is anchored at its own point.

Three things differ from the thread. The thread iterates to the minimum of its
potential with the limit of the previous step; here a step is one solve, the
limit is that of the step's own push, and the tangent is part of the result. A
sliding row is held back by exactly $\mu N$ for the velocity the step starts
with, and for the velocity the step adds it still has its spring. A block that
gains speed at $v$ therefore gains it as if every sliding row added the mass
$\mu N h^2/(\rho+hv)$: its acceleration is $(F-\mu W)/m$ over
$1+\mu g h/(v+\rho/h)$, below Coulomb's by less than the fraction $\mu g h/v$.
The cube of the tests (1 kg, $\mu=0.5$) pulled with $1.2\,\mu W$ gains
0.9755 of $(F-\mu W)/m$ in the second in which it goes from 0.42 to 1.38 m/s at
four substeps, and 0.9904 at eight (from 0.64 to 1.61 m/s); this formula gives
0.9754 and 0.9903. Pulled with half of $\mu W$ it
moves its floor rows by 6 µm, no anchor moves, and the floor holds the load to
0.02 %; at rest no anchor moves between 3 s and 10 s. Second, the limit is kept
as the distance $\rho$ and the anchor as a point of the floor, per corner of a
surface triangle. A cut renumbers those triangles, so the offset from the anchor
and the reach are carried as fields on the surface: every row of the new surface
takes them from the triangle of the old one on which it lies (`update_topology`;
`take_over` for a contact that replaces a grown one). Third, friction holds a
body in the shape in which it came to rest, and how it got there depends on the
time step: the cube's depths differ by 5 % between one substep and four when its
weight comes on at once, by 0.3 % when it comes on over a second, and not at all
without friction; the floor's force is the weight in every case. A sticking
row's force is its stiffness times a difference of single-precision
coordinates: on a node of that cube, whose rows add up to 2e5 N/m, one
spacing of a coordinate at 0.05 m (3.7 nm) is 0.74 mN along the floor, so two
runs whose positions agree to a few spacings differ by that much there. The
contact between cut pieces has no friction.

Material grips are finite spring loads attached to material coordinates. They
follow the cut fragments rather than prescribing every node's position.
The [full-pad guide](phantom-cut.md) records the example's controls and support
choices. The separate needle interaction below leaves Tet4 connectivity intact;
its virtual passage is not this cutting model.

## Differentiation contract

The copied FEM core retains its fixed-matrix differentiation API. The physical
needle interaction, fixed-boundary option and thread
solver are not differentiable. Material-hold helpers support the tissue viewers. Cut topology
changes are discrete and are not differentiated.

For a fixed assembled $A$, differentiable mode records the force/predictor and
state update while using the sparse linear-solve adjoint

$$
\Delta v=A^{-1}b(x,v,u),\qquad
A^Tz=\overline{\Delta v},\qquad \bar b\mathrel{+}=z.
$$

Supported derivatives are initial position/velocity, external nodal force, and
Tet activation. Geometry, mass, material, $h$, contact coefficients, and matrix
assembly are fixed; cut topology and contact creation are discrete. Clamps and
contact switches have branch-local derivatives. Checked forward/transpose
solves require $\|Ax-b\|\le2\cdot10^{-5}\|b\|+10^{-10}$; failures raise errors.
This differentiates a converged fixed-matrix solve, not finite CG iterations.
Use unique taped states; rebuild graphs after model changes. Ordinary untaped
stepping allocates no adjoint/trajectory storage. `SolverLinearFEM` is an alias.

## Needle insertion interaction

The public `NeedleTissueInteraction` class in
[insertion.py](../src/newton_solver_suturebot/_src/solvers/coupled/interface/insertion.py) adds an illustrative,
non-differentiable contact/track model to unchanged Tet4 material. Its force kernels
use Warp; host queries locate accepted tip positions in current tetrahedra.
The [example guide](needle-insertion.md) records geometry, defaults and limits.

For surface sample $i$ at $z_i$, weight $w_i$ and proxy radius $r_i$,
`_surface_forces` blends nearby blocking facets rather than selecting one
facet's force at a concave corner. For each facet $f$, let $p_f$ be its closest
barycentric point, $n_f$ its outward unit normal, $A_f$ its area,
$d_f=\|z_i-p_f\|$ and $g_f=n_f\cdot(z_i-p_f)$. Let $P_i$ indicate that the
geometrically nearest facet is already permitted by a local opening or material
association. If $P_i=1$, the nearest remaining blocking facet must satisfy
$d_f<r_i$ and $g_f\geq0$; it sets distance $d_0$ and side $s=+1$. Thus another
wall can contribute only within the physical radius, without treating an
authorized interior point as penetration through an entire distant component.
Otherwise the nearest blocking facet sets $d_0$ and
$s=\operatorname{sign}(g_0)$, with zero assigned the positive side. A sample
classified outside the full surface again considers only $g_f\geq0$ when
choosing this distance. An unauthorized interior sample retains signed
penetration recovery. The search includes the full physical radius even if
the nearest facet is permitted, so another intact wall is still considered.

With fixed physical support width $h_i=r_i/2$, `_facet_weight` forms

$$
u_f=\frac{\max(0,d_f-d_0)}{h_i},\qquad
a_f=A_f[\max(0,1-u_f^2)]^2
\left[\max\left(0,\frac{s g_f}{d_f}\right)\right]^2,\qquad
c_f=\begin{cases}
[\max(0,1-d_f^2/r_i^2)]^2,&P_i=1,\\
1,&P_i=0,
\end{cases}\qquad
\beta_f=\frac{a_fc_f}{\sum_{j\,\mathrm{blocking}}a_jc_j}.
$$

The radial factor $c_f$ has compact support and a continuous first derivative
at the proxy radius; this is a statement about the weight, not the entire
contact law. At distance at most $10^{-8}$ m, the orientation factor is set to
one as a numerical zero-distance convention. There is
no force when no blocking facet has positive weight or the sample is outside
contact. Penetration is capped by the nearest blocking distance:

$$
\delta_0=\max(0,r_i-s d_0),\qquad
\delta_f=\min\big(\max(0,r_i-g_f),\delta_0\big).
$$

For relative velocity $v_f=v_{n,f}n_f+v_{t,f}$ at $p_f$, the facet force is

$$
F_{n,if}=w_i\beta_f
\left[k_s\delta_f-c_s\frac{\delta_f}{r_i}\min(v_{n,f},0)\right],\qquad
f_{if}=F_{n,if}n_f-\mu F_{n,if}
\frac{v_{t,f}}{\max(\|v_{t,f}\|,v_\epsilon)}.
$$

Permissions are checked independently for each facet. On a flat patch with
uniform relative velocity, this recovers the specified normal stiffness and
damping. The damping acts only during compression and vanishes with
penetration; $c_s$ is its effective coefficient at $\delta_f=r_i$. There is no
tensile facet force. The compact area/orientation blend is a finite-radius
pressure-force approximation, not a force derived by differentiating a contact
potential. No elastic contact-energy conservation is claimed.
The localized tip has $w_0=1$; the
remaining shaft weights are normalized by their represented lengths to sum to
one. Thus $k_s$ controls localized tip stiffness and total shaft surface
stiffness separately. Before puncture, the tip normal resultant
$\|\sum_f F_{n,0f}n_f\|$, including compression damping but excluding friction,
must reach $F_p$ to authorize an opening.
This is an illustrative force criterion, not a measured fracture threshold.
Authorization is committed by `accept`
after an accepted physical step; it never creates the future planned passage.

The optional `surface_sample_stiffness` $k_1$ puts a floor under the shaft
weights. With $L_i$ the length represented by shaft sample $i\geq1$,

$$
w_i=\max\left(\frac{L_i}{\sum_{j\geq1}L_j},\frac{k_1}{k_s}\right),\qquad w_0=1 .
$$

One shaft sample then has stiffness $w_ik_s\geq k_1$ and damping coefficient
$w_ic_s\geq k_1c_s/k_s$; the tip and the puncture criterion are unchanged. The
shaft weights no longer sum to one: a shaft lying along the surface is as
stiff as the sum over its samples in contact, which grows with their number
and with the sampling density. Without $k_1$ (the default) the weights are the
shares alone. $k_1$ follows from the load one sample must carry within one
radius, the depth at which `_surface_forces` flags a sample's centre inside
fixed or inactive material. A needle of weight $mg$ set down on one sample
without speed dips to twice its static depth $mg/k_1$. If it arrives with
normal speed $v$, or falls freely through a clearance $h$ so that $v^2=2gh$,
and the contact dissipates nothing, $k_1r^2/2=mv^2/2+mgr$. Hence

$$
k_1=\frac{2mg}{r}\quad(v=0),\qquad
k_1=\frac{2mg}{r}+\frac{mv^2}{r^2}=\frac{2mg(h+r)}{r^2}\quad(\text{free fall}).
$$

For the bundled needle ($m=0.3$ g, $r=0.28$ mm) these are 21 N/m and, for
$h=3$ mm, 246 N/m.

`NeedleTissueCoupler` applies the interface load to the needle as an explicit
wrench within a trial and evaluates the contact at the trial's end state. For
a free body with native linear damping $d$ and step $\Delta t$, a change of the
load moves a contact point by $\Delta t^2/(m+d\,\Delta t)$ per newton, more
with rotation. The normal stiffness of $n$ samples in contact, and the
regularized friction of a contact that carries $F_n$ and slips slower than
$v_\epsilon$, then give the trial map gains of about

$$
\gamma_n=\frac{n\,k_1\,\Delta t^2}{m+d\,\Delta t},\qquad
\gamma_t=\frac{\mu F_n}{v_\epsilon}\,\frac{\Delta t}{m+d\,\Delta t}.
$$

Aitken relaxation has to undo these gains; its factor is bounded below by
0.001. At $\Delta t=1/60$ s, with $k_1=21$ N/m, $\mu=0.2$,
$v_\epsilon=0.2$ mm/s and $F_n=mg$, the needle drive's damping (0.69 N·s/m)
gives $\gamma_n=0.49$ per sample and $\gamma_t=4.1$; without native damping
they are 19.5 per sample and 163. This is why a needle with only its weight
is carried at the nominal step while the drive's damping stays in MuJoCo, and
needs many trials and much smaller steps without it; the
[usage notes](usage.md#physical-needle-handling) give the measurements. The
shaft preconditioner does not contain the surface contact's normal or
friction tangent.

Accepted actual tip samples are embedded as $x=\sum_a N_a q_a$ using current
tetrahedral barycentrics. Their material-coordinate lengths provide track
quadrature weights $\ell$. For closest shaft point $y$, shaft unit tangent $t$,
$P_\perp=I-tt^T$ and relative velocity $u$, the penalty tunnel mode's
`_shaft_forces` applies to the needle

$$
f_\perp=-\ell[k_\ell P_\perp(y-x)+c_\ell P_\perp u],\qquad
f=f_\perp-\mu\|f_\perp\|
\frac{u\cdot t}{\max(|u\cdot t|,v_\epsilon)}t.
$$

Let $p(s)$ interpolate the unchanged authored polyline by arclength and $s$
locate the closest shaft point. The tangent is the normalized secant from
$p(\max(0,s-r))$ to $p(\min(L,s+r))$, where $L$ is total shaft length and $r$
is the largest proxy radius. This gives a continuous sliding direction across
authored segment joints, independent of contact-sample subdivision. It is a
regularized force direction; the needle positions and closest-point geometry
remain the original polyline.

Material track sections are kept separate across empty gaps. Withdrawal and
re-entry can reuse existing local tracks; repeated visits do not create a
second copy of the same material spring. No samples or quadrature intervals
are created in empty space.

An engaged material point continues resisting lateral separation while the
finite shaft covers it; the spring does not disappear at a distance cutoff.
When the tip or tail withdraws past the point, it disengages and can re-engage
only on local return. No endpoint attachment is created beyond the tip or
tail. For implicit stepping, engagement is accepted history held fixed during
the force solve; `accept` updates finite-shaft coverage and local return for the
next step. A newly traversed material sample begins engaged. These discrete
transitions are resolved at substep boundaries, avoiding a finite spring-force
jump inside the endpoint equation. Tangential position is unconstrained. Cutting
resistance and the work of forming a passage are omitted.

The example's implicit endpoint evaluation uses the friction laws above
without an impulse cap. For the low-level **explicit** `add_forces` mode only,
supplying needle and particle masses caps each local tangential friction
magnitude by $s/(hW)$, where $s$ is relative slip speed in the friction direction
and the inverse relative effective mass is

$$
W=1/m_r+b^T I_r^{-1}b+\sum_{a\,\mathrm{free}}N_a^2/m_a,
$$

and $b$ is the body-local representation of the moment arm crossed with the
unit friction direction. The example supplies these masses and inertia. This
limits an explicit friction impulse using local inertial response and depends
on $h$. It is not a passivity guarantee for all simultaneous FEM, drive and
contact loads, and is excluded from the converged implicit force law.

The example's default `tunnel_model="alm"` constrains each material anchor
relative to a virtual lumen in the shaft's local transverse plane. It reuses
the same barycentric anchors, quadrature lengths and authored needle geometry;
it does not excavate a mesh cavity. Let $d=P_\perp(y-x)$, clearance $c$, physical
compliance $\alpha$ in $\mathrm{m^2/N}$, numerical penalty $\rho$ in
$\mathrm{N/m^2}$, and normal/tangential reaction densities $\lambda_n,\lambda_t$
in $\mathrm{N/m}$. The generic public interaction retains `"penalty"` as its
default. For finite compliance $\alpha>0$, `tunnel_alm.py`
evaluates the converged normal reaction directly:

$$
\widehat\lambda_n=\frac{(1-c/\|d\|)_+}{\alpha}\,d,
$$

with zero reaction inside the clearance. This is the exact fixed point of the
compliant normal proximal update, so no normal multiplier iterations are needed
in this branch. For a hard wall ($\alpha=0$), normal ALM remains:

$$
z=P_\perp\lambda_n+\rho d,\qquad
\widehat\lambda_n=(1-\rho c/\|z\|)_+z.
$$

Both branches use the axial Coulomb projection

$$
\widehat\lambda_t=
\operatorname{clip}(\lambda_t+\rho h(u\cdot t),
 -\mu\|\widehat\lambda_n\|,\mu\|\widehat\lambda_n\|),\qquad
f=-\ell(\widehat\lambda_n+\widehat\lambda_t t).
$$

At a converged positive normal reaction, $\|d\|-c=\alpha\|\lambda_n\|$;
the numerical penalty therefore does not define physical wall compliance.
The generic interaction uses $\alpha=1/k_\ell$ when no compliance is supplied.
The needle example explicitly uses $\alpha=1/300000$ m²/N, with a 20 µm
clearance, while its numerical penalty remains 60000 N/m². This gives a
fivefold stiffer radial wall than the earlier example setting without changing
the penalty-mode shaft stiffness. Positive compliance permits radial
separation under load; convergence does not impose hard geometric confinement.
Setting $\alpha=0$ requests a hard local wall, requiring the normal
cone residual to converge. The tangential projection permits static traction
at zero slip and saturates at the Coulomb bound during sliding; it replaces
the penalty mode's regularized velocity friction. No extra shaft normal
damping is included in this ALM mode.

`NeedleTissueCoupler` first converges the primal force solve for fixed trial friction
multipliers, then updates them and resets its Aitken history. Hard-wall normal
multipliers follow the same cycle. All evaluations share the same iteration
budget. Acceptance requires the normal constitutive/geometric residual and
the aggregate force-unit bound $\sum_i\ell_i|\Delta\lambda_{t,i}|$, in addition
to the original interface-force criterion. The hard-wall branch also includes
$\ell_i\|\Delta\lambda_{n,i}\|$ in that bound. Multipliers are committed only with
an accepted physical state; failure discards them. New anchors start at zero,
and withdrawal clears released anchors. These are discrete local-plane
constraint and force-balance checks, not a global collision, energy,
unconditional stability or exact no-slip guarantee for the curved mesh.
Surface puncture, gap handling and general rigid/FEM coupling are unchanged.
The finite-compliance preconditioner uses the frozen-plane derivative
$[(1-c/r)P_\perp+(c/r)uu^T]/\alpha$ outside clearance, where $r=\|d\|$
and $u=d/r$, and zero inside. This is not the derivative of the full moving
shaft geometry. Pointwise constitutive equivalence does not imply identical
discrete engagement histories: the stricter matched-step comparison failed
its 2 µm maximum nodal-difference criterion, as recorded in the
[needle guide](needle-insertion.md#interaction-and-outcomes).
Surface blocking is
exempted only in the local opening's connected, similarly oriented face
neighborhood or through an accepted material-track association. For an
associated sample, each facet is exempted only when the segment from the sample
to its closest barycentric point meets no active-material boundary before the
endpoint. The ray excludes the final $10^{-8}$ m for float32 endpoint roundoff
and forms its direction from local barycentric displacements. This tests actual
material visibility rather than the sign of a triangle's extrapolated plane,
which can extend through a concave tissue interior. A separate wall across
empty or inactive material remains blocking. Samples in
visited tetrahedra near the track retain that association. A retained engaged
anchor also permits samples within $2r$ arclength of its closest finite-shaft
projection, provided each sample lies in active material and the anchor-to-sample
segment crosses no exterior surface. Thus large lateral deformation does not
replace an existing spring with a new bulk-surface penalty. The $2r$ Euclidean
capture radius still limits new engagement; it does not switch off retained
springs. These distances are independent of track quadrature spacing. Exterior
crossing checks preserve contact with a separate material component. The full
boundary derived from active tetrahedra also protects interfaces to inactive
material that do not appear on the reference phantom's outer surface. Local
association cannot cross those interfaces, and accepted tip steps crossing them
fail before publishing a track through inactive material.
The punctured leading tip also has trial-local permission to continue from its
accepted interior position into another active cell when its swept segment
crosses no exterior surface. Internal tetrahedral faces are not tissue
boundaries. This permission creates no accepted track samples, applies only to
the tip, and cannot bridge an exterior gap or bypass supported/inactive faces.
The first candidate entry through an already registered puncture opening also
receives leading-tip permission before a track exists. It requires one inward
crossing that agrees between active and reference boundaries and an unfixed
active endpoint. The same active-material visibility test limits its facet
permissions; a separate intact wall still contributes contact.
A single outward candidate crossing can provisionally exempt the leading tip
at the crossed face's connected, similarly oriented patch, within twice the
proxy radius of the swept crossing point. This avoids artificial exit blocking
without exempting a separate gap wall. The exit event and persistent opening
are still published only by `accept`.

Each force is applied at its material interaction point $x$ as needle wrench
$(f,(x-c_{\rm COM})\times f)$ and opposite nodal forces $-N_a f$. Using the same
point for both transfers preserves force and moment balance algebraically.
Every active nodal share, including constrained nodes, remains in the interface
load vector. The FEM projector removes constrained motion, not its reaction
load; inactive, unmapped shares are recorded as support loads.
Diagnostics split free-tissue reactions from fixed/inactive support shares;
the latter are interface loads assigned to supports, not the complete FEM
boundary reaction. Drive loads are separate external loads.
These diagnostics describe the endpoint force-law scatter. The force iterate
actually applied by the implicit solvers can differ by the declared coupling
tolerance; its balance is a separate quantity, checked in
[focused regression tests](../tests/test_needle_tissue_coupler.py) rather than
recorded for every full-pad substep.

The public `NeedleTissueCoupler` in
[coupling.py](../src/newton_solver_suturebot/_src/solvers/coupled/coupling.py) solves an implicit
partitioned interface problem. Let $L$ contain every nodal interface force,
the needle force and needle torque divided by a length $R$ (the example uses
the needle's curve radius). Let $S_h(y_n,L)$ advance both solvers from the same
accepted state under $L$ plus separately held external loads, and let $G$
evaluate interface forces at that candidate endpoint. The force solve requires

$$
\|G(S_h(y_n,L))-L\|_\infty
\leq \epsilon_a+\epsilon_r\max(\|G(S_h(y_n,L))\|_\infty,\|L\|_\infty).
$$

The public coupler defaults are $\epsilon_a=2\cdot10^{-7}$ N,
$\epsilon_r=2\cdot10^{-4}$ and at most 40 trials. The interactive needle
example uses $\epsilon_a=2\cdot10^{-6}$ N and $\epsilon_r=10^{-3}$,
and a 160-trial cap; convergence still ends a solve immediately. It enables
the shaft preconditioner described below and starts relaxation at 1.0.
The previous accepted load initializes each solve. The generic unpreconditioned
default starts vector Aitken relaxation at $\omega_0=0.3$;
for interface residual $r_k=G(S_h(y_n,L_k))-L_k$ and
$\Delta r_k=r_k-r_{k-1}$, subsequent updates use

$$
\omega_k=\operatorname{clip}\!\left(
-\omega_{k-1}\frac{r_{k-1}^T\Delta r_k}{\|\Delta r_k\|^2},\,0.001,\,\omega_{\max}\right),
\qquad L_{k+1}=L_k+\omega_k r_k.
$$

Here $\omega_{\max}=1$ for the unpreconditioned solve and 2 while the shaft
preconditioner below is active. An insignificant residual difference retains
the previous relaxation. Interface loads and residual vectors stay in persistent
float64 arrays on the model device. Six reduced interface statistics return to
the host for the unchanged convergence and nonfinite-force checks.

Preconditioned, non-differentiable CUDA host trials also retain the relaxation
scalar on the device. The unchanged direction reduction feeds a float64 update
with the same denominator guard (`> 1e-30`) and `[0.001, 2]` clamp. Two scalar
slots separate old and new values so concurrent load-update threads cannot
observe partially updated history. Tighter FEM tolerance, ALM updates, a new
coupled interval and transactional fallback restart from the configured scalar
without reading old slots. NaNs propagate as in NumPy's host formula. This
removes the second statistics readback used only for mixing; it changes neither
the physical force equation nor stopping tolerances. CPU, unpreconditioned,
gradient and Tape paths retain host scalar updates.

For eligible CUDA host-loop trials, [DeviceSurfaceAnderson](../src/newton_solver_suturebot/_src/solvers/coupled/acceleration/surface_anderson_device.py)
retains the load, raw residual and search-direction histories in device memory.
One packet contains the exact union of rows that are nonzero in any retained
history, current input or pending recovery vector; subnormal and nonfinite
values are never discarded by a threshold. Original row order is restored on
the CPU before the same thin-SVD calculation. Cached full-vector residual norms
preserve the existing safeguard reductions. The default packet holds 128 rows;
overflow grows its retained capacity to cover the complete union and repeats
transport before committing a proposal or history. Dense data can grow to the
full interface size. History resets retain this allocation but discard all
secants and recovery metadata. CPU and gradient/Tape configurations retain the
ordinary host implementation. This changes transport and storage only: each
proposal still needs the original physical evaluation and acceptance tests.

For unpunctured surface contact in the non-differentiable, unpreconditioned
subcycled-MuJoCo path, a bounded Anderson proposal supplements Aitken once the
adaptive FEM tolerance reaches `min(configured_tolerance, 1e-6)`. Gradient-enabled
input/candidate state arrays or controls, differentiable solvers and an active
Tape retain the original iteration. At most six load/residual secants enter a
thin SVD solve, scaled by the largest secant-column norm. Directions are retained
only for $\sigma>\max(10^{-4}\sigma_{\max},10^{-14})$. Rows exactly zero in every
stored load and residual are omitted; every nonzero value is retained.
The regularization is $\lambda=\mu\|r_{\rm best}\|_2^2$ (scaled internally),
with $\mu$ initially $10^{-8}$ and bounded to $[10^{-12},10^8]$.

The residual reference weights the best recent iterate by 55% and the mean of
the others by 45%. The next physical evaluation measures actual reduction
against the predicted reduction, using a 0.95 prediction factor in the ratio's
denominator. A ratio below 0.1, or an infinity residual above 1.05 times its
corresponding reference, rejects the proposal: the history is cleared, $\mu$
is multiplied by ten, and the saved best Aitken proposal is returned. A ratio
above 0.75 reduces $\mu$ by ten. Coefficient, predicted-residual and step-size
bounds reject unreliable proposals before evaluation. This adaptive
regularization idea follows [Ouyang et al. (2023)](https://link.springer.com/article/10.1007/s10915-023-02231-4);
the SVD filter, infinity guard and Aitken recovery are application-specific,
and the paper's convergence assumptions are not established for contact switches.

Every proposed load is evaluated from the same physical start and consumes
the existing iteration budget; only the original physical force residual can
accept a state. Uniform coefficients act on complete nodal/rigid vectors to
preserve load balance. New intervals, tighter inner tolerances, dual updates
and host-force fallback start fresh numerical histories. This changes numerical
force guesses, not contact laws, accepted material history or physical tolerances.

The separate `preconditioned_anderson=True` constructor option extends this
mixing to supported native shaft/contact directions. It defaults off in
`NeedleTissueCoupler`; the physical `GraspScene` explicitly enables it. Let
$d_i$ be the preconditioned search direction and let a star mark the best raw
residual in the current history. With $\Delta L_i=L_i-L_*$,
$\Delta r_i=r_i-r_*$ and $\Delta d_i=d_i-d_*$, the same filtered, regularized
least-squares problem minimizes the **raw** force residual
$\|r_*+\Delta R\gamma\|_2$. Its proposal is

$$
L^+=L_*+d_*+(\Delta L+\Delta D)\gamma.
$$

Thus the residual chooses the affine coefficients while the search directions
supply the iterate images. Every trust ratio and infinity guard still uses the
raw force residual; a small preconditioned direction cannot certify physical
convergence. Recovery uses $L_*+\omega_0 d_*$, saved with the proposal. Without
an accelerated proposal, this opt-in branch uses the configured relaxed
direction. It begins at the configured inner FEM accuracy and clears history
whenever that accuracy tightens; it does not reuse secants from a different
approximate force map. Surface-metric validity switches, ALM dual updates,
new intervals and transactional fallback also clear history. Invalid/nonfinite
vectors discard pending recovery, and an interface-size change starts a new
history. Direction-only nonzero rows remain in the affine reconstruction.
This is a numerical search extension, not a derivative or a convergence
guarantee for changing contact sets; the original force and ALM acceptance,
adaptive retries and shared trial budget remain authoritative.

The optional `shaft_preconditioner=True` accelerates this iteration for
supported projected FEM configurations with a free needle, including the
subcycled articulated grasper described below. With generalized
velocity $(v_a,v_b,R\omega_b)$, let $J_f$ map free motion to each engaged
anchor's relative velocity, and let $J_{\rm all}^T$ scatter reactions to all
nodes, including supports. In penalty mode, for frozen tangent $t_i$ and
normal load $f_{n,i}$, the positive semidefinite anchor approximation is

$$
D_i=\ell_i(hk_s+c_s)(I-t_it_i^T)
+\frac{\mu\|f_{n,i}\|v_\epsilon}{\max(v_\epsilon,|u_i\cdot t_i|)^2}t_it_i^T.
$$

Let $H$ contain the full projected FEM matrix and the native rigid effective
inertia, including joint damping. The correction uses

$$
(H+hJ_f^TDJ_f)z=r,\qquad
d=r-hJ_{\rm all}^TDJ_fz,\qquad L^+=L+\omega d.
$$

The ALM branch uses the radial derivative described above and, on the
sticking branch, a positive approximation of its projected tangential update.
On the sliding branch the drag is the Coulomb bound of the wall reaction of
the trial, $\pm\mu\|R_i\|$ along $t_i$, so a change of that reaction changes
the anchor force through the shear $S_i=\pm\mu\,t_i d_i^T$, with $d_i$ the
unit direction of $R_i$: the anchor acts as $(I+S_i)D_i$ in the operator and
in the correction, and the operator is not symmetric. An auxiliary
block-Jacobi Krylov solve (CG; BiCGSTAB while a track that follows the tunnel
ALM has samples) uses an 80-iteration cap, with relative tolerance $10^{-2}$ for
finite-compliance ALM and $10^{-3}$ for penalty or hard-wall modes. Aitken
relaxation then operates on $d$. The full transpose
preserves reaction balance even for an approximate auxiliary solve. This is
a frozen shaft/friction approximation, not the full contact Jacobian:
apart from that shear, nonsymmetric derivatives are omitted. Surface curvature is added only when
the independent surface option below supplies a valid recording. In the penalty expression
above, the tangential term equals
the frozen-normal sticking slope and continues as a positive approximation
decaying with squared slip speed in saturated sliding. It is not a quadratic
majorizer or the full friction Jacobian. The selected physical force law
is unchanged, and the original physical residual remains the acceptance test.
The looser finite-ALM auxiliary solve passed task/integrity checks but failed
a declared 2 µm fast-handling trajectory-equivalence gate (12.46 µm maximum
nodal difference); it is not an identical-history guarantee. Unsupported configurations
retain unpreconditioned Aitken; no coupled derivatives are provided.

BiCGSTAB can break down: the stock solver divides by a product that may
vanish, and the correction is then nonfinite. The solve does not test for
this and the correction is applied as it is. The trial that applies it is
rejected by a test the trial loop already has. In the captured device loop
of the free needle that is the finiteness test of the candidate, and the
step ends as a nonfinite interface state. In the captured device loop of
the articulated tools (the grasp scene) the tissue solve of that trial also
fails its linear-residual certificate, and the step ends with that reason:
the host asks the certificate before it names a nonfinite state. In the host
loop it is the acceleration warning of the guarded native rigid adapter. The
host loop of the free-needle example with the plain MuJoCo solver tests the
evaluated interface force, not the load it applies; there such a step ends at
the trial cap of the loop or at the step-resolution test. In every case the
step is rejected as a whole and the accepted state stands.

The anchors of this correction (tangent $D_i$, lever, shear $S_i$) are
stored with one row for every row of the interaction's track reserve. The
kernels over them are launched over the reserve and read the number of track
samples from the interaction's live counts on the device; a row beyond it is
neither read nor written. Where a solve of the tunnel ALM is captured, the
device takes its solver by that count in every launch: CG while the track has
no sample, BiCGSTAB from the first sample on. Outside a capture the host
makes the same choice from its own count. Both solvers exist from the
construction of the correction. The number of track samples is therefore no
part of a graph key of the correction; the storage of its anchors is, and it
changes only when a reserve that was too small has grown.

Whether the track transmits a shaft interaction in a step, which is a shaft
law together with a nonzero engaged length, decides if host-controlled trials
and the captured loop of the free needle apply this correction or relax the
residual themselves. The value is evaluated by its first reader in a step and
held for the trials of that step, whose history does not change. The
evaluation reads the engaged length from the device, unless the acceptance
before it has read it on request. Trials that evaluate the value in every
step, which are the host-controlled trials of a coupler without device trials
and the captured loop of the free needle, make that request for the step
after them: each of their steps is answered from the read of the acceptance
before it. The captured loop of the articulated tools with the surface
metric applies the combined correction in every trial, and its launches over
the track read the live count, so its graph is the same for both values.
There the value is no part of the graph key and is not evaluated: such a step
reads no engaged length, and a track that starts or stops transmitting
captures nothing. When a step of that loop is handed to host-controlled
trials, they evaluate the value where they first run, without the request:
that step reads the engaged length once, its acceptance reads nothing, and a
restart of its trials keeps the value. The interaction keeps the length that
was read until the engagement is written, so a step that is rejected and
tried again at a shorter interval reads once as well. A query of the engaged
length between steps, as the scene report makes once per frame, reads the
device while no acceptance has kept it.

The independent `surface_preconditioner=True` option accelerates native
subcycled-MuJoCo contact. Before the first accepted track it acts alone; when
both options are enabled and shaft support is active, the two curvatures share
one solve. It defaults off in
the public coupler and free-needle examples; the physical-grasper worker enables
it. The optional [surface recorder](../src/newton_solver_suturebot/_src/solvers/coupled/interface/surface_contact_forces.py)
retains the original force evaluation and records every contributing
needle-sample/surface-facet pair during its final pass. It does not repeat the
nearest-surface query. The original default kernel remains available, and a
source-parity regression guards the physical statements shared by both paths.

For a recorded pair, let $a$ be its sample weight times normalized facet
weight, $p$ its capped penetration, $n$ its normal, $v_n=u\cdot n$, and
$u_t=(I-nn^T)u$. With the original normal force $f_n$, normal stiffness $k$,
damping $c$, sample radius $r_s$, friction $\mu$ and regularization speed
$v_\epsilon$, the positive search matrix is

$$
D=a\left[h\left(k-\frac{c}{r_s}\min(v_n,0)\right)
  +\frac{cp}{r_s}\mathbf{1}_{v_n<0}\right]nn^T
  +\frac{\mu f_n}{\max(\|u_t\|,v_\epsilon)}(I-nn^T).
$$

The tangential term retains radial sliding curvature. For frozen $f_n$, the
unchanged regularized-friction potential is
$\phi(s)=\mu f_n s^2/(2v_\epsilon)$ below $v_\epsilon$ and
$\phi(s)=\mu f_n(s-v_\epsilon/2)$ above it. At frozen speed $s_k$,

$$
\phi(s)\leq\phi(s_k)+
\frac{\mu f_n}{2\max(s_k,v_\epsilon)}(s^2-s_k^2).
$$

This lagged quadratic majorizer prevents the search matrix from losing its
radial friction slope at sliding onset. The approach follows the contact
iteration diagonal in
[the ANCF solver](https://github.com/nvidia-beer/newton-solver-ancf/blob/333f8edae7b47fa2e34dc32b493eddf64d68cfe6/src/newton_solver_ancf/_src/solvers/ancf/kernels_contact.py#L22),
adapted from its tanh regularization to this solver's clipped-linear law.
No ANCF package is imported. The bound assumes frozen normal load and geometry;
it does not establish global convergence of changing contact configurations.

The [surface preconditioner](../src/newton_solver_suturebot/_src/solvers/coupled/acceleration/surface_preconditioner.py)
uses the same $H$, auxiliary CG budgets and balanced correction as the shaft
preconditioner above, replacing its anchor map by triangle barycentric
interpolation. The inner $h$ in $D$ approximates endpoint displacement from
velocity; the outer $h$ in $H+hJ_f^TDJ_f$ produces the expected stiffness
$h^2$ and damping $h$ factors. Unmapped surface vertices are static supports;
mapped fixed vertices receive reactions through $J_{\rm all}^T$ but contribute
no free motion through $J_f$. Neither case renormalizes the remaining weights.

Geometry, penetration-cap choice, weights and normal/friction cross derivatives
are frozen in this search approximation. Actual forces, residual tolerances,
iteration caps, tetrahedral-volume checks and swept material-event checks
remain unchanged. Empty, nonfinite, overflow or impulse-capped recordings
restore the original search path; every change between preconditioned and raw
directions clears Anderson and Aitken history. Recorder count/validity are
packed into the existing statistics readback. Fixed recording storage and
auxiliary CG workspaces support CUDA capture without a new per-trial fence.
The path requires unmodified supported native/FEM/contact methods and disables
itself for differentiable arrays or active Tape execution.

With both contact options enabled, accepted shaft tracks and valid surface
facets contribute to one operator,

$$
H+hJ_{s,f}^TD_sJ_{s,f}+hJ_{c,f}^TD_cJ_{c,f}.
$$

The block diagonal and final balanced correction include the same two terms;
applying two sequential inverse corrections would be a different operator.
[CombinedContactPreconditioner](../src/newton_solver_suturebot/_src/solvers/coupled/acceleration/combined_contact_preconditioner.py)
retains the existing shaft mass metric, auxiliary solve budgets and reaction
scatter. The surface recorder covers postpuncture contact as well, without
changing the physical surface force. Invalid or empty surface records reduce
the augmented action exactly to the shaft-only search. Recorder storage,
validity and count are part of capture bindings; replacing those arrays
invalidates cached graphs while retained owners keep older graphs valid.

Each trial restores MuJoCo's full `mjSTATE_INTEGRATION` state and restarts FEM
from the same physical state $y_n$. Only the numerical CG velocity-increment
guess carries forward from the preceding interface trial. Rejection restores
the guess saved before the physical step, along with its physical state.
Trial interaction
evaluations preserve accepted engagement and restore authorization state.
Failed force iterations therefore do not advance accepted material
history or physical time. A converged candidate must also resolve the needle's
sampled motion and boundary crossings before it can be committed. An unresolved
candidate raises `NeedleStepTooLarge` and restores the rigid integration state
and FEM warm start. Once the last trial converges and passes resolution checks,
`retain_trial_forces` retains that exact endpoint evaluation instead of
evaluating identical forces again. Its candidate state must remain unchanged.
The actual time-$n$ state still supplies swept checks, and `accept` commits
events once; rejected solves discard the uncommitted trial context.
The resulting contact active-set changes apply to the next physical substep.
The example halves unresolved intervals and intervals whose interface force
iteration exhausts its budget, retaining the complete slider target
and the frame's material linearization. `refresh_tangent(q, h)` rebuilds all
timestep-dependent matrix terms when the non-differentiable timestep changes;
it also clears the numerical velocity-increment guess to avoid cancellation
from a guess for another timestep. Physical positions and velocities are
unchanged by this reset. Captured step graphs are not rebuilt for that
duration: they read $h$ from device data and replay unchanged (see the device
time step below).
After two consecutive accepted intervals needing at most four interface trials,
the example may double the interval, capped by the nominal duration and the
frame's remaining time. Rejection clears this growth counter. Every enlarged
attempt must pass the same force and swept-motion checks; each new frame starts
from the nominal duration.
Only accepted intervals advance time and material history. Differentiable
execution retains its fixed-timestep contract.

Before committing a coupled Tet4 step, the solver also checks the signed
volume along the straight segment from each node's accepted position to its
candidate position. The determinant is cubic in the segment parameter. Its
endpoints and every interior stationary point must have positive volume,
allowing for a float64 evaluation margin. Positive endpoints alone are not
sufficient: a tetrahedron can flatten and reverse twice within one interval.
An unresolved segment raises `NeedleStepTooLarge` inside the same transaction,
before interaction history, output state or the thread advances. Retrying
recomputes the physical step; no positions are projected and no material or
force tolerances change. This guard covers active tetrahedra, not arbitrary
element types or tissue self-contact. If refinement is exhausted, the run
stops with its last accepted state intact. The end-frame integrity check
remains a separate check for nonfinite state, moved supports and inversion.

The nominal substep count sets $h_0=(1/60)/N$; it is separate from the adaptive
retry limit. A rejected interval is halved while it exceeds
$\min(h_0/256,5\cdot10^{-6}\,\mathrm{s})$, then failure remains visible if the
force solve or motion guard still rejects it. This retains small physical
intervals for abrupt slider goals even when the ordinary frame uses few
substeps. Halving never changes the target, contact law or convergence tolerance.

The FEM material tangent is still refreshed once per display frame; convergence
of this interface solve is not convergence of a full nonlinear FEM equilibrium.
The coupler requires MuJoCo's CPU backend with its data synchronized from the
input state on every step, so that each trial exposes its endpoint state. This
is Newton's default; a solver created with the deprecated `update_data_interval`
set to another value is rejected. Input and output `body_f` retain only
the caller's external wrench; interface reactions are kept separately and
are not carried into the next step as an extra external force.

For one eligible dynamic free body on CUDA, the compact MuJoCo bridge batches
coordinates, velocities and the applied wrench into 19 input floats and returns
13 coordinate/velocity floats. It calls the same native `mj_step`, with the
same model, integrator, damping, contacts and full integration-state rollback.
Newton's float32 coordinate and angular-velocity conversions are preserved.
Eligibility requires a world free joint with identity anchors, zero body COM,
zero gravity and no actuators; frames and topology remain fixed for the
coupler's lifetime. Unsupported configurations, extra output observables or a
custom solver step use the general adapter. This changes transfers, not rigid
dynamics.

Optional FEM CUDA graph replay changes submission of $S_h$, retaining the
complete non-differentiable Tet4 CG step, block-diagonal preconditioning,
triangle terms, iteration budget, tolerance and numerical warm-start behavior. Rigid
integration, Aitken updates, endpoint forces and accepted events stay outside
the graph. Unsupported configurations, gradient-enabled execution and host
residual checks use eager FEM execution. The host-controlled path retains
bounded caches of Krylov workspaces and FEM graphs for recurring solver
tolerances and state buffers; the time step is no part of a graph's key (see
the device time step below). The device trials fetch the workspaces of their
tolerance levels from the same registry and pin them while a trial graph
binds them; the registry keeps at most eight workspaces that no graph owner
has pinned, and the oldest of the others leave where a step registers a
workspace or an owner releases one. The existing tightening schedule
still runs; returning to an earlier tolerance reuses its storage instead of
allocating a new solver and recapturing every interval. Rebuilt projections and
changed matrix storage invalidate the associated resources. Active
differentiation retains its original execution path. A step uses no discarded
physical integration to warm these caches: a workspace or a graph is made
where a trial first needs it. A scene may ask for them ahead of its first step
instead. `NeedleTissueCoupler.capture_graphs`, which the grasp scene on the
MuJoCo Warp rigid backend calls at the end of a reset, captures and launches
the FEM step of the host-controlled trials at every inner tolerance, the
held-start preparation and the trial graph of the device trials for both
accepted-state buffers, inside attempts that end like a rejected step, and makes
the storage of the response-basis predictor and its column engine there; in the
attempt of the first buffer it also builds the device helper of the host sample
mask and captures and launches the force graphs of the host-controlled trials,
one for each of their eight variants (`capture_host_force_graphs`), and in the
attempt of the last buffer it captures and launches the contact preparation and
the solves of the shaft and the surface correction and makes the device
workspace of the host Anderson mixing (`capture_contact_graphs`), whose owners
give back the work they counted; these
launches are discarded work, and every accepted array and work counter is
given back. With a trial budget the call also allocates the best load of a
host-controlled loop, so that the first such loop of a run does not. The grasp
scene then makes the phase load predictor of its control frames with the
storage of its three frames and of the engagement snapshot of every phase
(`PhaseLoadPredictor.reserve`), so that neither the first frame nor a coupled
step of the run allocates them. Geometry
caching and batched
broad-phase queries preserve the original bounds arithmetic and ordering.
On CUDA, conservative Warp BVHs return sorted tetrahedron/facet candidates;
the original float64 NumPy containment and cubic swept-crossing tests still
decide material membership and crossings. Candidate overflow falls back to
the complete CPU query. The swept facet query pads the box of the tip's path
by the largest motion of the tissue, on each axis, from the accepted positions
that the interaction committed to the query. A step may start from other
positions: a contact that places nodes moves them in the accepted state after
the acceptance. The box is then grown further by the largest difference of a
coordinate between the start positions and the committed copy, measured for
that step, and the exact test reads the actual start positions. Every facet
whose swept bounds reach the tip's box has end bounds within that growth and
motion of it, so the candidates still hold all facets the exact test needs.
The complete CPU query is kept without a committed copy, when the query
measured its motion from an earlier accepted state, for another particle
count, for a difference that is not finite, and on overflow. The interaction
counts both per test of one surface (`shifted_start_candidate_tests`,
`shifted_start_host_tests` by reason) and keeps the largest growth
(`shifted_start_largest_growth`). Accepted material history remains
host-owned, and cached endpoint geometry is reused only for matching nodal
positions.
The allocation-free CUDA mesh refit retains Warp's triangle-bound arithmetic
and bottom-up BVH updates. Its float32 per-triangle edge sums are accumulated
in float64 by one complete warp per 256-face chunk, followed by a complete-warp
float64 total and the existing float32 mean conversion. This changes only the
deterministic reduction order. General wide-range inputs have the usual
float64 summation-roundoff bound; equality to the earlier serial double sum is
not promised for every range, nor is bitwise equality to native float32 CUB
scan. The original serial reduction remains on CPU. No bounds or edge values
are retained across a move unless the existing static-vertex mask permits it.
Contact checks distance/side, footprint weight and force contribution before
material visibility, keeping the original facet order and positive-weight
nonpenetrating blocking facets in normalization. For verified local passages,
other blocking facets contribute only at $d<r$, so an AABB half-extent $1.5r$
retains the full contact support and blending margin. Unauthorized interior
recovery keeps the wider nearest-surface query, and active-material visibility
rays are unchanged. Surface launches use one sample per GPU block; this
changes scheduling, not the contact equations. These optimizations add no
gradient, force-history convergence or material-accuracy guarantees.

`capture_coupling=True` additionally captures the repeated established-tunnel
solve in [coupling_graph.py](../src/newton_solver_suturebot/_src/solvers/coupled/coupling_graph.py).
FEM, rigid integration, geometry updates, force evaluation, Aitken relaxation,
adaptive inner tolerances and ALM dual decisions run inside a conditional CUDA
loop. Their budgets and acceptance criteria are unchanged. A finite-state
test runs before any candidate BVH refit. A compact result is read only after
the loop ends. Host-owned material history is unchanged during these trials.
Candidate tunnel points are rebuilt from their material barycentric weights.
Anchor/segment projections run in parallel with the original double-precision
arithmetic, followed by the same ascending first-minimum selection. Candidate
material masks are recomputed for every trial; only allocations and accepted
topology are reused. Krylov workspaces for the existing tolerance levels are
allocated once and retained across track recaptures.

Whether the accepted track transmits a shaft interaction selects the
correction of a trial: with an engaged track the residual is corrected by the
shaft metric, without one it is relaxed as it is. For the free needle on
native MuJoCo this flag is device data as well. The trial graph issues both
corrections behind one device flag, each with the launches, and in the
order, of a graph that holds that correction alone, and the host writes the
flag before a launch when its value has changed. An entry or an exit of the
needle therefore neither drops nor captures a graph; before, the flag was an
entry of the trial signature, and the graphs of both state buffers were
captured again at every flip. The corrected branch is the device part of the
correction, the anchor and diagonal preparation and the Krylov solve: its
launches over the track have the size of the track reserve and read the live
count, so the branch can be captured before the track has a sample. With a
response-basis start guess, whose binding is host work of every correction,
the flag stays an entry of the signature. So it does in the captured loop of
the articulated tools on MuJoCo Warp without a surface metric, whose rigid
adapter applies the correction itself: there a flip captures the graphs of
the other value. With a surface metric the flag is not evaluated. All three
are as before.

`NeedleTissueCoupler.capture_graphs(native=True)` is the start-up entry
described above for that backend. `fem_needle_insertion` and
`fem_needle_suture` call it when they are run as examples; a caller that
does not ask is declined, as before. There is no rigid interval to capture:
it is the native step behind the compact bridge. The trial graph is captured
and launched with an empty track, a state that the device trials of a run
never start from: the one trial of that launch asks for the host path and
the loop ends. The contact preparation and the solve of the host-controlled
correction are captured and launched with a zero residual
(`capture_contact_graphs(native=True)`), and the kernel modules and the
buffers that only eager code of such a step uses are loaded and allocated
first. The attempts step neither MuJoCo nor an accepted state, and end like
a rejected step.
Measured on cuda:0 on 2026-11-17: the flag flips six times in the whole demo
of `fem_needle_suture` and six times in the self-test of
`fem_needle_insertion` followed by a slider session, with no capture (before,
both buffer graphs were captured again at every flip).
`tests/test_needle_coupling_graph.py` compares the steps on both sides of a
flip with host-controlled trials and holds the rule of where the graph
branches, `tests/test_mjw_device_trials.py` pins the signature entry and the
captures of the MuJoCo Warp loop without a surface metric and a response
basis, and `tests/test_needle_startup.py` checks
that the start-up call leaves the accepted states, the native integration
state and the work counters as they were and that every graph it captured has
been launched.

The time step is device data for these trials. One `StepTime`
([step_time.py](../src/newton_solver_suturebot/_src/utils/step_time.py)),
owned by the FEM solver and shared by the coupler with the interaction, the
contact preconditioner and the rigid interval, holds $h$ as one `float32`
and one `float64` and the rigid microstep $h/n$ ($n=4$ microsteps in the
grasp scene) as one `float32`. The host
forms each value in double precision, as it did when the value was a launch
argument, and writes it only when $h$ changes and never during a capture; the
kernels read element 0 and their arithmetic is unchanged. A retried shorter
interval therefore replays the captured trial graph: $h$ is not part of its
signature. The free needle's native damping, which its drive scales with $h$,
is read from device arrays in the same way. The tangent refresh runs outside
the graph in `refresh_tangent`. Its tetrahedral and hexahedral kernels read
$h$ from the same arrays; the mass, spring and triangle blocks, which take
$h$ as a launch argument, are assembled only at construction, on a changed
$h$ and on a topology refresh, and that assembly raises during a capture.

The other graphs of a coupled step read the same arrays, and none is keyed on
$h$: the FEM step of the host-controlled trials, the rigid solver's own
interval, the force graphs of the host-controlled trials, the preparation of
the contact preconditioner and the held FEM start. The code that launches one
of them writes $h$ first, outside the capture. A graph is captured for a
state buffer, a solver level or array storage, and replays at every $h$.
The worker thread owns a second `StepTime` for its accepted interval, with
$1/h^2+c/h$ of its consistent mass held as one `float64` that the host forms
and refills when $h$ or the mass damping $c$ changes.
A write of another $h$ while a graph is being captured raises. If a value
that the host derives from $h$ cannot be formed, as for $h=0$, nothing is
written and the arrays keep the previous $h$.

A graph that a caller captures itself reads the same arrays. `SolverFEM.step`,
the tangent refresh of `refresh_tangent`, the thread step and the cutting
helpers (`MaterialHolds`, `CutSurfaceContact`, `CutVolumeGuard`, which take
the solver's `step_time`) hold no $h$ by value, so such a graph uses the $h$
that the arrays hold when it is launched: call `solver.step_time.set(h)`
before the launch, after `refresh_tangent(q, h)` outside capture when $h$
differs from the assembled one. A constant $h$ remains in three places, none
of which changes during a run: the frame graph of `needle_thread`, captured
before the run, in which stock Newton's `SolverMuJoCo.step` fills MuJoCo's
time step and the hold gains are formed from the step at construction; the
relaxation factor of `ThreadCapsuleContact`, formed once from the nominal
step and used by eager and captured steps of every size; and the force sum of
a captured differentiable FEM step, whose solver accepts its construction
step only.

The captured attempt copies the complete incoming state once at graph entry.
FEM and rigid integration overwrite the evolving particle/body/joint coordinates
and velocities in every trial. The per-trial particle-force scratch copy is
retained because the FEM step publishes its RHS through the incoming force
array. Trial force evaluation clears the rigid force buffer before writing its
reaction. Graph replay, swapped input buffers and retries therefore refresh
the same state fields without repeating the other full-state copies.

For the supported single COM free body, the device integrator reproduces
MuJoCo 3.17.0's `implicitfast` update in double precision, with Newton's original
float32 coordinate conversions. The MuJoCo model must have no equality
constraint: Newton adds an inactive weld to every free root unless the solver
is created with `kinematic_switchable_roots=False`, as the needle example does. With local inertia $I$, local angular velocity
$\omega$, damping $D$, and torque $\tau$, its angular increment solves

$$
\left[I+h\left(D+[\omega]_\times I-[I\omega]_\times\right)\right]\Delta\omega
=h\left(\tau-D\omega-\omega\times I\omega\right).
$$

This specialization excludes native contacts, actuators, constraints, fluids,
gravity and callbacks. Supported coefficients are checked before reuse.
Native `mj_step` still advances integration history once using the final
applied load. Every joint/body coordinate and velocity must match the device
endpoint bit for bit before retaining the converged device forces and ALM
proposal. The host stages only the accepted-step metadata; it does not repeat
shaft association or force evaluation. The original swept-motion checks and
history acceptance still follow. Conservative GPU permission
certificates cover unchanged material passages; ambiguous tip boundary events
restore the initial trial state and use the exact existing event path. Thus
this is a device-resident repeated solve, not fully device-resident history
management. CPU execution and unsupported configurations retain the original
solver. Differentiation through this history-dependent path is unsupported.

### Needle query launch sizes

The coupling launches the small, branch-heavy shaft and surface query batches
with one sample per GPU block, including captured fallback paths. Explicit
query launch overrides are honored by the graph constructor and included
in the capture signature. This changes scheduling only; the query kernels,
force laws and acceptance checks are unchanged. Different atomic accumulation order can still
change later engagement or retry decisions, so stationary force agreement
does not establish trajectory equivalence. This tuning targets the single
needle example; throughput for batches of independent scenes is unqualified.

### Shared stepping and example control

On supported CUDA devices, the persistent Krylov solver uses conditional graph
iterations to stop at the existing residual tolerance, with the same maximum
iteration budget. `linear_solver_use_cuda_graph=False` retains the fixed-budget
path. Unsupported devices retain the existing execution path.

The example reduces finiteness of all particle positions/velocities and rigid
poses/velocities, maximum material displacement and rigid target error on the
device after every accepted substep. It reads three float64 diagnostics rather
than full state arrays; displacement is computed against float64 reference
positions. End-frame element-inversion and exact fixed-node checks remain.
Those checks also run on the device: float64 determinant ratios, a nonfinite
flag and an exact constrained-position comparison reduce to three scalars.
Rendering uses compact per-region vertex buffers with unchanged triangles and
normal generation. Regions containing active free vertices are updated;
neighbors sharing only fixed interface vertices remain at rest. Unchanged
OpenGL meshes persist until a selection or opacity change invalidates them.
Other viewers retain per-frame logging. These changes do not scale or amplify
the displayed deformation.

Swept tip/boundary queries solve the cubic plane-crossing equation under linear
per-step tip and triangle-vertex interpolation and check triangle barycentrics.
Multiple boundary crossings or an unresolved tip displacement reject a coupled
candidate before track publication, allowing a shorter physical retry. The
explicit interaction interface retains its visible failure on unresolved motion.
These are discrete-event guards, not a
continuous collision solver for the entire moving needle.
The displacement bound applies to every sampled shaft point, including when
the tip stays still during rotation; swept triangle-crossing roots test the tip.
Each entry through an untouched surface must cross a physically authorized
opening neighborhood. Existing local entry/exit openings permit withdrawal
and re-entry. The interaction exposes actual entry and exit events without
knowing a planned path, target, expected passage count or task termination rule.
The example controller alone chooses the drive and decides whether an actual
exit reaches its selected material target.
The finite-force COM drive is separate from the selected tunnel law:
$f_d=k_d(c^*-c)+b_d(v^*-v)$; its rotational
counterpart uses the shortest quaternion rotation error. Damping is estimated
from mass and the largest inertia eigenvalue. Desired pose is never substituted
for actual state during stepping. Timestep and track-spacing refinement remain
necessary for any particular parameter/geometry choice.

An entry or an exit is a passage of the tip, so `accept` also looks at the
tip's own motion. An accepted step that is not converged beside an opening (a
trial loop kept at the trial budget, or a step within the relative tolerance
of a large load) can leave the tissue surface on the other side of the tip for
that one step, while the tip keeps moving the same way; the surface falls back
in the next step. The point-in-tetrahedron test alone counts that as an entry
and an exit of a needle that never came back. For a change of side `accept`
therefore takes the face of the event, its outward unit normal $n$ at the
accepted state, and the tip's displacement over the step,
$d=(x_{\rm tip}-x_{\rm start})\cdot n$. The face of the event is the face of
the swept crossing in the direction of the change; else the face at which the
last accepted step held this change, while that face stands; else the face
nearest the tip. A kept face stands while the tip is still past its plane and,
for an exit, while the exit branch would take it: the tip not more than two
needle radii $r$ from that triangle, and the triangle's point nearest the tip
not in supported or inactive tissue. These are the exit branch's own tests of
its face, read by the same lines, so that branch never refuses a kept face. In
the steps after a held one no surface crosses the tip, and the nearest face
can be another one: the far face of tissue thinner than twice the tip's depth,
or another wall across a gap. Its normal reads the tip's motion with the other
sign, so the face of the hold is kept until the next accepted step. An entry
with $d>\epsilon$ and an exit with $d<-\epsilon$ are held: no event, no
counter, no opening, no track section and no failure check of that branch. The
tip keeps its side, a staged reuse of an opening is committed as before, and
the next accepted step asks again. $\epsilon$ is 16 units in the last place of
the tip's largest `float32` coordinate (about 60 nm at 0.04 m), the resolution
of the stored pose, so tissue that closes over a needle at rest still commits
its entry. A change of side is held only while the tip is at most one resolved
step (`step_resolution`) past that face. A change of side that the tissue
makes against a moving needle is therefore committed once the tip is more than
one resolved step past the face, or once the tip no longer moves the other way
by more than $\epsilon$. It is committed at the face it was held at, or at the
nearest face when that one no longer stands: a tip that slid along the surface
during the hold, more than $2r$ from the triangle it was held at or to where
that triangle's nearest point is supported, leaves by the triangle it is then
over. Otherwise the exit and its opening stay at the triangle of the hold, at
its point nearest the tip: in the fixture 0.46 mm from a tip that slid that
far onto the next triangle of the same plane. If the tip is back on its side
first, nothing is committed, however many steps that takes: a hold has no
limit in time. In the one-tetrahedron fixture, tissue that comes 290 µm over a
tip that withdraws 2 µm per step (one resolved step there: 303 µm) is held for
142 accepted steps, until the tip is out again, with no entry and no exit.
Back on its side need not be back through the same face: after a held entry, a
needle that reverses and leaves by another face within the next accepted step
commits no entry and no exit. That needs tissue thinner than the tip's travel
in one step, about 11 µm in the demo. While an exit is held the tip counts as
in the material, so a second wall that closes over it is ordinary travel: a
track sample and a visited tetrahedron in that wall, no entry, and the exit is
later committed at the wall's face. A commit after a hold runs the checks of
its branch at the positions of that later step, and can end in status `failed`
where a commit in the first step passed: the exit is refused when the tip is
by then more than $2r$ from the surface or over supported or inactive tissue,
the entry when it is more than $2r$ from its opening. For the exit of a tip
that is over the triangle of the hold, or over a neighbour in its plane, the
tip was at most one resolved step from the surface when held, so the first
needs the tip and the surface to move apart by more than
$2r-{\tt step\_resolution}$ in the step of the commit: 0.3075 mm in the needle
scenes ($r$ = 0.28 mm, resolved step 0.2525 mm). The needle's own motion in
that step is part of it. With a needle at the resolved step the surface need
only move 55 µm; at the demo's 11 µm per step it must move about 296 µm, where
the largest single-step move measured is 0.17 mm. For the entry the hold gives
no such bound: the tip can be anywhere within $2r$ of its opening when the
surface passes over it. In the fixture ($r$ = 0.5 mm) an exit held 8 µm past
the face is committed after the tissue retracts 0.9 mm more in one step, and
fails after 1.2 mm. The entry is also refused, as it was before this rule,
when the opening that the held step's trial staged is not the one of the face
of the event: in tissue so thin that the held tip is nearer the far face. That
is the authorization of entries, which this rule leaves as it is. This is host
arithmetic in `accept` on the steps with a change of side: no kernel, no
device read, nothing in a captured graph. It does not make the unconverged
step converge.

For the example's zero-COM-offset free joint, the velocity-dependent damping
is supplied as MuJoCo joint damping and integrated by `implicitfast`.
`NeedleDrive.prepare_timestep(h)` additionally contributes $hk_d$ to linear
damping and $hk_\theta$ to angular damping, updating these terms on retries.
The external spring and target-velocity loads retain the original gains.
For a held translation target this gives the backward-Euler drive equation

$$
(m+hb_d+h^2k_d)v^+
=mv+h[k_d(c^*-c)+b_dv^*+f_{\rm other}].
$$

The rotational term uses an identity Jacobian approximation of quaternion
error: it is exact for collinear principal-axis motion, approximate for general
rotations. This changes integration, not the continuous PD law or its gains.
`configure_mujoco(solver, dt=None)` retains damping-only integration; the example
opts into the spring linearization. Existing passive joint damping is preserved.
Tissue contact and changing rotations still require resolved physical steps.

Slider control changes only the example's desired arc angle and world-Z
offset: $x^*=x_{\mathrm{plan}}(\theta)+z_h e_z$, with the planned orientation
$R_{\mathrm{plan}}(\theta)$. Each slider value is the full drive target for the
next physical interval; there is no target slew rate or ramp. Manual target
velocity is zero, so differencing a mouse jump does not create a velocity
feedforward impulse. Releasing the mouse holds the target while physical forces
move the needle and tissue. Arc travel is bounded by the preview and finish,
and world-Z lift by ±2 mm in this example. Accepted-target reporting commits only after a
coupled step succeeds. The existing COM force/torque drive acts on this target,
without overwriting the actual body pose. Manual handling
retains physical failure checks and event logging but has no automatic exit
termination or insertion deadline. Reset restores automatic control.

These event and
contact kernels have backward recording disabled; no coupled gradient is
claimed. The track does not change topology, stiffness, mass or damage state.

## Suture thread and its needle attachment

`SolverThread` integrates chains of two-node cubic Hermite cable elements. A
node has a position $\mathbf r$ and a slope $\mathbf r'$ along the reference
length. With the Green strain $\varepsilon=(|\mathbf r'|^2/|\mathbf r_0'|^2-1)/2$
and the curvature $\kappa=|\mathbf r'\times\mathbf r''|/|\mathbf r'|^3$, the
elastic energy of a straight reference is

$$
U=\int_0^L\Big(\tfrac12EA\,\varepsilon^2+\tfrac12EI\,\kappa^2\Big)\,ds ,
$$

integrated with five Gauss points per element; a curved reference uses
$(\kappa-\kappa_0)^2$. The mass is the consistent Hermite mass. There is no
twist and no shear.

Time integration is backward Euler. A slack thread buckles, Newton's method is
then slow, and a second-order step that stops before it has converged passes
an acceleration error to its next predictor: the thread vibrates without end.
A backward Euler step is the minimum of a step potential,

$$
\Phi(\mathbf q)=\frac1{2h^2}(\mathbf q-\widetilde{\mathbf q})^T\mathbf M(\mathbf q-\widetilde{\mathbf q})
+\frac{c}{2h}(\mathbf q-\mathbf q_n)^T\mathbf M(\mathbf q-\mathbf q_n)
+U(\mathbf q)+V_{\rm floor}(\mathbf q)-\mathbf q^T\mathbf M\mathbf g,
\qquad \widetilde{\mathbf q}=\mathbf q_n+h\mathbf v_n,\quad \mathbf v=(\mathbf q-\mathbf q_n)/h,
$$

with the mass damping rate $c$. The Newton loop stops at a tolerance or at a
fixed number of updates, and its line search never raises $\Phi$. A step cut
short therefore ends no higher than its first iterate, and it is a damped,
stable step only if that iterate is one. The free nodes start at
$\widetilde{\mathbf q}$, where $\Phi$ has no inertial part; whatever else the
first iterate strains stays in the thread when the updates run out. The
attachment below starts node 0 accordingly.
The forces are exact. The iteration matrix is positive semidefinite: it keeps the axial
geometric stiffness only in tension and the Gauss-Newton part of the bending
tangent. Each chain takes the multiple of its Newton update, from the fixed set
$\{4,2,1,\tfrac12,\tfrac14,\tfrac1{16},\tfrac1{64},0\}$, with the lowest $\Phi$.
The two smallest serve a straight slack thread held at one end: it has no
stiffness across itself, the update of a load across it stretches it, and
with a quarter as the smallest multiple it did not move at all. The tangent of a chain
is block tridiagonal with one 6-by-6 block per node and is eliminated directly:
on a CPU block by block from the far end, one kernel thread per chain, and on a
GPU by cyclic reduction, in which every level eliminates the two neighbours of
all nodes at once at a stride that doubles, so a chain of $n$ nodes takes
$\lceil\log_2 n\rceil$ levels. Contact acts on the nodes, each against its own
plane with point $\mathbf p$ and unit normal $\mathbf n$; a level floor is the
same plane for every node. With the thread radius $r$,
$d=r-\mathbf n\cdot(\mathbf r-\mathbf p)$ and $p=(d+\sqrt{d^2+s^2})/2$,
$V_{\rm floor}=kp^2/2$. A step reads nothing back to the host and can be
captured in a CUDA graph; its Newton loop is a conditional graph node.

Friction and eyelets serve a thread on tissue. Their tangents are positive
semidefinite, so the iteration matrix stays so.

Friction on a contact plane. A node in contact is tied to an anchor
$\mathbf a$ on its plane by a spring whose force stops at the Coulomb limit
$f_c=\mu N$, with the node's normal force $N$ at the end of the previous
step. With the offset $\mathbf d=(\mathbf I-\mathbf n\mathbf n^T)(\mathbf r-\mathbf a)$
and the contact stiffness $k$,

$$
V_{\rm f}(\mathbf d)=
\begin{cases}
\tfrac12k|\mathbf d|^2, & k|\mathbf d|\le f_c,\\
f_c\big(|\mathbf d|-f_c/2k\big), & \text{otherwise.}
\end{cases}
$$

A node that sticks holds any force up to $f_c$ and stays; a node that slides
feels $f_c$ against its way. After the step the anchor of a sliding node is
moved to the point $f_c/k$ behind the node, and a node without grip has its
anchor at its own position.

The curvature of $V_{\rm f}$ along a sliding direction is zero. A Newton
update with it overshoots, and the fixed set of multiples then finds no
descent: with that tangent, and with a friction law in the step's velocity
before it, the capped loop stalled, contact forces rose to a thousand times a
node's weight and returned through the next step's limit. The tangent is
therefore that of the spring that has the same force at the current offset,

$$
\mathbf K_{\rm f}=\min\!\big(k,\;f_c/|\mathbf d|\big)\,(\mathbf I-\mathbf n\mathbf n^T).
$$

That spring's energy touches $V_{\rm f}$ at the current offset and lies above
it everywhere else, so an update with it cannot raise $V_{\rm f}$. On a level
plane a thread of 20 elements dragged along its axis is held back by
$\mu\sum N$ to within 4 % with two updates per step and 0.4 % with eight, and
it is at rest, with its tension locked in, after the drag stops.

Eyelets. An eyelet is a point $\mathbf x$ the thread must pass through but
may slide through. It keeps a chain coordinate, an element and a position
$\xi$ on it, which at the start of every step moves to the chain point
nearest to $\mathbf x$. With the centreline point $\mathbf r(\xi)$ there and
its unit tangent $\mathbf t$ at the start of the step,

$$
V_{\rm eyelet}=\tfrac12k_e\,\big|(\mathbf I-\mathbf t\mathbf t^T)(\mathbf r(\xi)-\mathbf x)\big|^2 ,
$$

a spring across the thread. Along the thread the eyelet has the friction of
the contact planes in one dimension: it keeps an anchored chain coordinate
$\xi_a$, the offset is $d=\mathbf t\cdot(\mathbf r(\xi_a)-\mathbf x)$, the
stiffness is $k_e$ and the limit is $\mu$ times the force across the thread
at the end of the previous step. The anchored point stays while the thread
sticks and otherwise trails the tracked point by $f_c/k_e$. A taut thread
drawn over a quarter circle of seven eyelets with $\mu=0.2$ needs 1.38 times
the tension behind them; Coulomb friction at an eyelet that turns the thread
by $\alpha$ gives the factor $(1+\mu\sin\tfrac\alpha2)/(1-\mu\sin\tfrac\alpha2)$,
1.37 for these seven. When the draw stops, a ratio of 1.2 stays locked in.

Obstacles on elements. Besides the planes on its nodes, an element can have
a plane on one of its own points $\mathbf r(\xi)$, with the contact potential
of the node planes in the height $\mathbf n\cdot(\mathbf r(\xi)-\mathbf p)$ and
without friction. Two uses share it, and both find $\xi$ on the element's
chord at the start of the step.

With self-contact every element looks for the nearest other stretch of its
chain; elements closer than three apart along the chain are left out. The
plane lies midway between the two closest points, with the normal from the
other stretch, and the element is kept one thread radius from it. Both
stretches do the same, so their centres keep one diameter, and an overlap of
two like stretches is taken back by half from each side instead of in full
from both. A thread that springs open across itself on a floor keeps
0.26 mm or more between the centres of two stretches of diameter 0.30 mm;
without self-contact they pass through each other.

`ThreadCapsuleContact` gives every element the nearest point of a rigid
body's capsule axis, a polyline at the body's current pose, as its obstacle,
at the capsule radius plus the thread radius. The opposite forces go to the
body as a wrench in its next step, relaxed like the loads on the tissue with
$K/(K+k)$, where $K=m/h^2+c/h$ is the least stiffness with which the body
resists a force within a step. A thread of 20 elements laid over a bar above
a floor stays 0.44 mm from the bar's axis for radii of 0.43 mm, and its
weight of 31.62 µN is carried by the floor, 5.66 µN, the bar, 19.35 µN, and
its held end, 6.61 µN.

`ThreadAttachment` ties node 0 of a chain to a rigid body that another solver
owns. A light body cannot take the thread's force one step late, so the body's
response is an unknown of the thread's solve. The rigid solver first steps the
body without the thread; that predicted pose has centre $\mathbf c$ and
includes contact and every other load. Node 0 then sits on a rigid head that
moves by a translation $\mathbf u$ and a rotation vector $\boldsymbol\theta$
from that pose,

$$
\mathbf r_0=\mathbf c+\mathbf u+\mathbf R(\boldsymbol\theta)\mathbf a,\qquad
\mathbf r_0'=\mathbf R(\boldsymbol\theta)\mathbf s,\qquad
\Phi_{\rm head}=\frac1{2h^2}\mathbf m^T\mathbf M_b\mathbf m,\quad
\mathbf m=(\mathbf u,\boldsymbol\theta),
$$

with the lever $\mathbf a$ and slope $\mathbf s$ at the predicted pose and the
body's mass and inertia $\mathbf M_b$. $\Phi_{\rm head}$ is added to $\Phi$:
a wrench $\mathbf W$ moves a free body by $h^2\mathbf M_b^{-1}\mathbf W$ within
one semi-implicit Euler step. The head's rows of the residual use the exact
Jacobian, $d\mathbf r_0=d\mathbf u+(\mathbf T\,d\boldsymbol\theta)\times\mathbf R\mathbf a$
and $d\mathbf r_0'=(\mathbf T\,d\boldsymbol\theta)\times\mathbf R\mathbf s$, with
the tangent operator of the rotation vector,

$$
\mathbf T=\mathbf I+\frac{1-\cos\theta}{\theta^2}[\boldsymbol\theta]_\times
+\frac{\theta-\sin\theta}{\theta^3}[\boldsymbol\theta]_\times^2 ,
$$

so they are the gradient of $\Phi$ that the line search needs. The rigid
solver then repeats its step from the same state with
$\mathbf W=\mathbf M_b\mathbf m/h^2$ added.

The head's first iterate puts node 0 where the thread's own motion carries
it, like every free node: $\boldsymbol\theta^0$ is the smallest rotation that
turns $\mathbf s$ onto the extrapolated slope
$\widetilde{\mathbf r}_0'$, and
$\mathbf u^0=\widetilde{\mathbf r}_0-\mathbf c-\mathbf R(\boldsymbol\theta^0)\mathbf a$.
That iterate strains no element. Its offset from the predicted pose costs
$\Phi_{\rm head}(\mathbf m^0)$, which is quadratic, so one update removes it,
and which is of the order of the kinetic energy $\tfrac12m|\Delta\mathbf v|^2$
of the change $\Delta\mathbf v$ that the prediction made to the body's
velocity. A first iterate at the predicted pose, with the head's last motion,
instead moves node 0 by $h\,\Delta\mathbf v$ against its neighbour: an
elastic energy of the order of $EA\,(h\,\Delta v)^2/2\ell$ for elements of
length $\ell$, which grows with the modulus and has no bound in the body's
energy. In `needle_thread` at 2e9 Pa a needle that landed at 0.78 m/s then
left 2.2e-3 J in the thread after two updates, nineteen times the energy of
the whole scene, and the wrench threw the needle back up; from 5e8 Pa
released needles gained energy and some runs became non-finite. With node 0
at $\widetilde{\mathbf r}_0$ the same runs, and releases from pulled and
pinned threads, did not gain energy with two updates from 2e6 to 2e9 Pa.
Lagged mode keeps the head's last motion as its first iterate.

The head moves as a free body would. Where the rigid solver constrains the
body, by contact or a grasp, the body yields less than the head did and the
thread's pull on it is underestimated; node 0 is then off the attachment
point by the head's motion $\mathbf m$. A body held by a gripper should
prescribe node 0 instead.

`ThreadTissueCoupling` couples one thread to a tetrahedral tissue. Before the
thread's step every node takes the nearest point of the pad's triangle surface
within a margin, and the outward normal of that triangle, as its contact
plane. The surface vertices that are tissue particles move with them; the
bounding volume hierarchy is refitted in place. A stitch's track is a group
of eyelets at material points of the tissue, ordered from entry to exit. A
node between the first and the last threaded eyelet of a group is inside the
tissue and takes no contact plane; while a group is only partly threaded, so
is the thread from its first node to the group's last threaded eyelet.

Where a threaded track meets the surface the needle has left a hole, the
mouth of the track, and the surface mesh has none. A node over that hole
would be stopped by the plane of the face around it until the pull presses it
through the contact penalty, with up to $k\,(r+d_1)$ for the thread radius
$r$ and the depth $d_1$ of the track's first eyelet, and a node that has just
left a track would be thrown out by the same plane. A scene can therefore
open the mouths (`ThreadTissueCoupling.open_mouths`): with $\mathbf p$ the
nearest surface point of a node, $\mathbf n$ the outward normal there and
$\mathbf e$ the eyelet at an end of a track's threaded span, the node takes
no plane when

$$
0\le(\mathbf p-\mathbf e)\cdot\mathbf n\le d_{\rm mouth},\qquad
\big|(\mathbf I-\mathbf n\mathbf n^T)(\mathbf p-\mathbf e)\big|\le r_{\rm mouth}.
$$

The grasp scene gives $r_{\rm mouth}$ two wire radii, the radius within which
the needle/tissue interaction takes a surface point for the needle's track,
and $d_{\rm mouth}$ one eyelet spacing, under which the eyelet at the end of
a track lies. The end of the span at the thread's start is a mouth only when
the whole track is threaded. The test is made with the nearest surface point:
where the surface around a mouth is drawn in by more than $d_{\rm mouth}$,
the node has its plane again. On the device the thread's force on the needle
while it is drawn through both lips falls from a sawtooth of up to 12 mN to
0.4 mN ([usage](usage.md#the-draw-of-the-thread)).

The friction anchor of a node that has grip is kept as a point of the surface
triangle under it and moves with the surface, so a thread that sticks rides
on the tissue.

Small CUDA thread batches evaluate every ordered chord pair in parallel before
selecting the nearest self-contact in the original ascending element order.
The original segment arithmetic, reach, adjacent-element exclusion, strict
positive-distance test and first equal-distance tie are preserved. Scratch is
bounded to 131072 ordered pairs with at most 128 elements per chain. CPU,
differentiable/taped, custom-query and larger cases retain the serial query.
This changes scheduling only: the contact planes, all tissue/tool contacts,
Newton stopping rule, update budget and potential-based line search are
unchanged. Thread state advances only after accepted rigid/FEM intervals.

After the thread's step the contact force of a node, normal and friction,
with the opposite sign, is spread over the tissue particles among the three
vertices of its triangle, and the force on an eyelet over the four vertices
of its tetrahedron. The tissue receives these loads in its next step. Passed on
at once, this staggered exchange diverges when the thread's spring $k$ is
stiffer than the tissue node $K$ it acts on, because each side overcorrects
the other, so the applied load $\mathbf f_a$ is relaxed towards the thread's
current force $\mathbf f$:

$$
\mathbf f_a\leftarrow\mathbf f_a+\frac{K}{K+k}\,(\mathbf f-\mathbf f_a) .
$$

The examples take $k=Eh$ for the contact and the eyelet spring, with the
tissue's Young's modulus $E$ and mean edge length $h$, as the give of the
tissue around the thread that the mesh does not resolve, and $K=Eh/4$ as the
reference estimate of a single tissue node's stiffness. This estimate is not
a proven lower bound for deformed tissue with multiple loaded eyelets. The
tissue then feels a change of the thread's pull after about five steps.

In `fem_needle_suture` the stitches are planned before the run. A stitch is
an insertion plan between an entry and an exit; it is kept when that plan's
checks pass and the needle's sweep from the plan's finish angle until the
tail has left the tissue touches no pad material. Its track points are the
plan's circle sampled inside the tissue. During the run the track point at
angle $\varphi$ becomes an eyelet once the tip angle exceeds $\varphi$ plus the
needle's angular span, that is when the tail has passed it. The thread is
stepped once per accepted step of the needle and the tissue, with that step's
size, and the relaxed loads enter the next step as external forces.

In the grasp scenes nothing is planned: the thread follows the track that the
needle has cut (`WorkerThread` with `thread_tracks`, the default of the grasp
scene). The needle/tissue interaction records that track as material points a
quarter millimetre apart, in sections, one per entry of the tip into material
(a passage). `NeedleTrackEyelets` reads those rows from the interaction's host
lists after every accepted step, nothing from the device, and makes eyelets of
them while the needle goes in: the first row of a section, every later row
whose rest position is at least the spacing $s_e$ from the last eyelet of its
section, and at an exit of the tip the last row of the section unless it is
nearer than $s_e/2$ to the last eyelet. The eyelets of a section are one
group. An eyelet is written in place into a reserve that is fixed at
construction (`ThreadTissueCoupling.add_track_point`): with $p$ planned
passages per bite, a needle of length $L$ and a track no longer than the
needle, a bite makes at most $\lfloor L/s_e\rfloor+2p$ eyelets in $p$ groups;
the reserve holds two bites, 36 eyelets and 4 groups for the default stitch
($L=22.4$ mm, $s_e=1.5$ mm, $p=2$). An eyelet beyond the reserve is not made
and is counted. The spacing is the suture example's, a choice between the
mean tissue edge (1.31 mm; closer eyelets load the same nodes) and
$\sqrt{8Rr}=3.4$ mm, beyond which the chord between two eyelets leaves the
needle's circle of radius $R$ by more than the thread's radius $r$.

The tail threads the eyelets. In the tissue update of the thread's accepted
step, after the eyelet positions and before the groups' spans, one kernel over
the reserve takes the tail $\mathbf x_t$ of the needle and the unit direction
$\mathbf o$ in which the thread leaves it, from the accepted pose, and threads
an unthreaded eyelet at $\mathbf x$ at the thread's first node when

$$
0\le a\le s_e ,\qquad |\mathbf d-a\,\mathbf o|\le 2\,r_w ,\qquad
\mathbf d=\mathbf x-\mathbf x_t,\quad a=\mathbf d\cdot\mathbf o ,
$$

that is when the eyelet lies behind the tail by at most one spacing, so that
the eyelet behind the one being passed is out of reach, and beside the
thread's line by at most two wire radii $r_w$, the radius within which the
needle/tissue interaction takes a point for the needle's track. A tail that
passes a millimetre beside a track threads nothing. The eyelet spring is the
contact spring $k=Eh$, so the relaxation above is unchanged, and a scene whose
tail never enters the tissue steps bit for bit as without the reserve on the
CPU (the thread alone, and the coupled grasp scene over its first entry and
exit, with and without the swept tool/tissue contact). The step graph of the
thread is captured at start-up with these kernels in it, each launched over
the whole reserve; an eyelet that appears later is four rows written from the
host, and no graph, array or kernel module is made after start-up.

The suturing example defaults to four physical substeps per 60 Hz frame
(240 Hz). A default three-stitch CUDA run at the insertion example's single
substep inverted tissue during the second stitch at 22.867 s. Four substeps
completed the same row, closure and drop with the same material, mesh, forces
and commanded speed. The force relaxation fraction is per physical step, so
reducing the timestep also shortens its delay in seconds; it is not a
timestep-independent constitutive law. The coupled Tet4 motion check above
remains active at every substep.

Between two stitches of the planned row, which are copies of each other, the
needle makes the rest of its full turn while its circle's centre moves on.
Picked stitches may point anywhere. Between them the centre rises to
$z_{\rm top}+t+R$, with the pad's highest point $z_{\rm top}$, the needle's
thickness $t$ and its radius $R$, moves level while the orientation is
interpolated along the shorter arc, and comes down. Every point of the needle
stays within $R$ of the centre, so it cannot touch the pad on the way.

An eyelet can be held: it then keeps its point of the chain and its spring
acts in every direction, $V=\tfrac12k_e|\mathbf r(\xi)-\mathbf x|^2$, a knot
at a material point. `fem_needle_suture` holds the first track point of the
first stitch once the row is made, and moves the needle's target on along
$\mathbf d=\cos\beta\,\mathbf l+\sin\beta\,\mathbf e_z$. Here $\mathbf l$ is
the horizontal direction from the last exit back across the wound and one
stitch on, towards where the next entry would be, and
$\beta=\arcsin(c/2R)$ is the slope of the track at the exit for the stitch's
chord $c$ and the needle's radius $R$. A thread that turns back over the
wound at an exit pulls that lip towards the other one. The wound's width is
measured as the mean distance between the last track point in one lip and the
first in the other. The pull ends when that width is below the thread's
diameter, when the smallest volume ratio of the tissue's tetrahedra falls
below a chosen fraction, one half by default, or when the needle's circle
reaches the pad's edge.

A tie needs a point of the chain. `ThreadTissueCoupling.hold_eyelet` launches
one kernel, which sets the held flag of an eyelet only if the thread passes
through it and writes nothing otherwise: the threading is known on the
device, and nothing is read back for the refusal. Held without the thread,
an eyelet would tie the chain's first node to itself in the step that threads
it. The fraction of its rest volume at which a tetrahedron ends a pull is one
number of the coupling (`TIGHT_VOLUME_RATIO`, one half): the default of
`fem_needle_suture --least-volume` and the value at which `fem_needle_grasp`
stops tightening its tied thread
([below](#physical-distal-grasp-and-worker-owned-thread)). It is a choice of
how far an element may be squeezed, not a strength of tissue. Both examples
take the smallest ratio from the check that ends every frame, which reads it
once and keeps it (`least_volume_ratio`).

The needle in `fem_needle_suture` is stepped with the tissue once per time
step and that step cannot be repeated, so `ThreadAttachment` runs in its
lagged mode. The needle's step takes the wrench $\mathbf W_n$ of the thread's
last step. A drive with stiffness $k$ and damping $c$ holds the needle, so
within a step a wrench moves it by $\mathbf Z^{-1}\mathbf W$ with

$$
\mathbf Z=\frac{\mathbf M_b}{h^2}+\frac{\mathbf c}{h}+\mathbf k .
$$

The pose the needle would have reached without the thread is therefore its
new pose moved back by $\mathbf Z^{-1}\mathbf W_n$. The head is placed there
with $\mathbf Z$ in place of $\mathbf M_b/h^2$, and the solved motion gives
$\mathbf W_{n+1}=\mathbf Z\,\mathbf m$. For a thread of stiffness
$\mathbf K$ at the head this is the relaxation
$\mathbf W_{n+1}=\mathbf W_n+\mathbf Z(\mathbf Z+\mathbf K)^{-1}(\mathbf W-\mathbf W_n)$
towards the thread's wrench $\mathbf W$ at the needle's pose: a slack thread
is felt at once, a taut one along its own direction over a few steps. Node 0
is off the tail by $\mathbf Z^{-1}(\mathbf W_{n+1}-\mathbf W_n)$ until the
wrench has settled; in the default run that stays below 0.01 mm. Tissue
around the needle makes it stiffer than $\mathbf Z$, which only slows the
approach.

When the pull ends the last track point of the last stitch is held as well,
so the thread between the two knots keeps its tension. The drop removes the
drive's spring, also from $\mathbf Z$, and adds the needle's weight
$m\mathbf g$ at its centre of mass; the model's gravity stays zero, as the
needle-tissue coupling requires. The drive's damping stays in MuJoCo, so
the needle sinks at $mg/c$ for the drive's damping $c$, about 4 mm/s in the
example, instead of falling.

Before a run the thread is laid unstrained, level with the top of the pad,
and lowered. For that its mass damping is raised to $g h/r$, at which its
weight moves a node by the thread's radius $r$ per step, and it is stepped
for one and a half times the way to the lowest point of the pad under it;
then the damping is set back and the velocities are cleared. Laid straight
onto the pad's steps instead, the thread starts strained, and dropped freely
it lands at a speed at which the capped loop does not resolve the contact.

There is no friction between the thread and itself or the needle, and an
element keeps only its nearest contact of each kind. Backward recording is disabled in its kernels;
no gradient is claimed. These equations and the examples' checks do not
establish calibrated suture behaviour or a frame rate.

## Independent action-controlled scenes

The `parallel` module executes the existing accepted-step needle/FEM algorithm
in spawned processes, one scene per process. It changes scheduling and the
source of the desired rigid pose, while preserving the implicit drive, FEM
operator, contact laws, convergence criteria, adaptive timestep retry and
accepted-event transaction. Every process owns its native rigid state, tissue
arrays, solver warm starts and material history. There is no shared mechanical
state and no cross-scene contact. A selected reset rebuilds that scene only.

A normalized six-vector increments the held world-space position and rotation
target. Its rotation vector is converted to a quaternion and left-multiplied
onto the target quaternion. Target jumps carry zero velocity feedforward, as
in manual insertion, so discontinuous actions do not invent a target-velocity
impulse. The drive's finite wrench determines actual motion. The reference
insertion plan remains optional privileged information; it is never applied
as a kinematic state trajectory.

Only reset may initialize the body pose. Seeded translation perturbations are
independent across scene IDs and advance only when the corresponding scene
resets. Event-to-target errors are frozen against the accepted state's deformed
material targets when an entry or exit is committed. A later tissue deformation
therefore cannot change the accuracy of an earlier event.

The process interface is not differentiable across IPC and makes no new
gradient claim. In Warp 1.17, `ScopedStream` changes the shared device stream
and host array conversions temporarily select the null stream; using these
existing solver paths from independent Python threads would race stream
selection. Process isolation avoids that hazard. Its performance must be
measured with loaded contact and identical numerical budgets; inactive tissue
and additional CUDA contexts can substantially change throughput.

## Physical distal grasp and worker-owned thread

The optional `SutureSceneConfig(control_mode="physical_grasp")` selects
[GraspScene](../src/newton_solver_suturebot/examples/grasp/_grasp_scene.py). The needle remains
an independent free rigid body in native MuJoCo. Each tool has two fixed-trocar
RCM hinges, an axial insertion slide, shaft roll, two wrist joints and two
independently actuated revolute jaws. Finite native joint-position actuators
move the tool, and frictional jaw/needle contacts transmit the load. Jaw opening
and sufficient external loading can release the needle. Reset initially places
the needle in the bite. This demonstrates retention, transport and release of
an initialized grasp, rather than autonomous pickup. The second tool currently
holds a parked pose with its own actuators and open jaws. Full robot arm
hardware, transmissions and calibrated actuator dynamics are outside this
reduced articulated model.

[_dvrk_rcm_grasper.py](../src/newton_solver_suturebot/examples/grasp/_dvrk_rcm_grasper.py) attaches the first
hinge to world. Its intersecting yaw/pitch axes and downstream axial slide
enforce a shaft line through a fixed trocar. `GraspScene` uses the registered
port frames in the bundled
[`tissue3_215300` metadata](../src/newton_solver_suturebot/examples/assets/dvrk/recordings/tissue3_215300/recording.json),
with approximately 172 mm between ports. The reference shafts converge by about
44° toward the workspace. This is an estimated recording-to-phantom alignment,
not independently measured robot calibration; the shaft angle varies with
joint coordinates and the dataset contains other fixture/phantom placements.
The shaft reference directions use registered frame 4 roll-body transforms.
PSM1's roll zero is shifted by π about its shaft to keep the insertion within
its roll limits, preserving the port position and shaft direction. PSM2's
parked pose is nearby bounded FK with wrist pitch −1.0 rad; its recorded wrist
pitch exceeds the surrogate's range. These references initialize the layout
and do not replay recorded motion.

The default physical-grasp stitch in the bundled region 1 uses the insertion
and exit material labels in the same recording's
[`point_labels.json`](../src/newton_solver_suturebot/examples/assets/dvrk/recordings/tissue3_215300/point_labels.json).
`load_source_stitch` validates the semantic labels, common region and triangle
bindings, then maps each registered position through the inverse phantom
transform into native FEM coordinates. The retained reference vertex indices
and barycentric weights identify the original material points. Configuration
uses their XY coordinates for surface selection; the selected points match
those bindings on the bundled reference geometry. Combining this layout with
the earlier generic `x=0` stitch made PSM1 cross the parked PSM2 workspace.
The source labels and ports share the same estimated registration; neither is
an independent metric calibration. Explicit point selections remain supported,
and other assets/regions and `free_needle` retain their previous defaults.

For this source stitch, reset selects a bite at 80% of the needle arc length
from the tip. Starting from the transverse bite frame, it applies a −75° twist
about the local needle tangent, followed by a −15° rotation about the resulting
jaw-closing X axis. The second rotation inclines the needle through the jaw
plane and preserves the bite position. This is an illustrative grasp choice
that leaves wrist travel within the existing limits, not a needle pose fitted
to the recorded images. Other stitches retain the prior −90° tangent twist and
zero oblique angle. This selection only initializes the free needle and tool
at reset; subsequent motion uses finite actuators and frictional contact.

The physical scene explicitly uses 240 mm shaft centrelines and a nominal
180 mm trocar-to-roll-origin distance, with ±50 mm axial travel. Shaft yaw/pitch
have ±0.6 rad limits, roll ±π, and wrist pitch/yaw ±1.1 rad. Reset solves initial
joint coordinates for the ready grasp against the registered port frames; it
does not move a port to satisfy that grasp. The trocar-to-shaft perpendicular
distance and axial coverage are measured from accepted body states. Visual
trocar markers impose no contact or constraint.

Standalone `add_rcm_grasper` assemblies retain their compact defaults: 120 mm
shaft centreline, 80 mm nominal insertion, ±30 mm travel, and −0.7 rad nominal
wrist pitch. Their nominal-wrist anchor helper computes
$T_{W,R}=T_{W,G}^{\rm ready}(T_{R,G}(q_{\rm nominal}))^{-1}$ using exact chain FK.
Those low-level defaults and reset helper are distinct from `GraspScene`'s
registered fixed-port configuration. Both retain the source shaft radius and
distal joint dimensions.

[NativePoseIK](../src/newton_solver_suturebot/_src/sim/ik/pose_ik.py) fits the commanded
wrist pose within those six joint limits using independent native scratch
kinematics. Its solution supplies actuator targets only; it never copies a
kinematic solution into live joint/body state. Failure to meet pose tolerances
rejects an unreachable command rather than translating the fixed base. Tool
links assume ideal gravity compensation, while the independent needle retains
its full gravitational load.

The right-hand PSM1 uses the Large Needle Driver visual, and left-hand PSM2
uses the photo-estimated DeBakey forceps visual. The LND retains source CAD;
the DeBakey is not manufacturer CAD. Each physical assembly carries the
collision capsules of its own instrument, fitted to that CAD; they are not
separately calibrated instrument contact geometry.

#### The collision capsules of a tool

[_dvrk_grasper.py](../src/newton_solver_suturebot/examples/grasp/_dvrk_grasper.py) retains the bundled
PSM jaw hinge frames: hinge axes are local Z, jaw length is local +Y, and the
two jaws close across X. The collision capsules are not typed in. A tool has
one capsule set, fitted to the CAD mesh that is drawn on the same body
([_dvrk_capsules.py](../src/newton_solver_suturebot/examples/grasp/_dvrk_capsules.py),
`dvrk_tool_capsules`), and its model publishes it (`capsule_bodies`,
`capsule_endpoints`, `capsule_radii`, `capsule_parts`, `capsule_bounds`). The
rigid collision shapes of the MuJoCo model, the tool/tissue contact (either
law), the thread's contact with the tools, the measurements of
`fem_tool_tissue` and the collision shapes that the viewer's "Show Collision"
option draws all read these rows, in `fem_needle_grasp`, in `fem_tool_tissue`
and in every scene built with `add_rcm_grasper` or `add_grasper`. No reader
has radii of its own.

The rule. The surface of a drawn part is divided into cells: along the part by
planes across its axis, and across it into rows side by side. Each cell gets
the smallest capsule about its centre line that contains the surface of the
cell: seen along the centre line, the axis goes through the centre of the
smallest circle around the cell, and along it the capsule is as short as its
round caps allow. A cell reaches one sampling distance (1/64 of the smallest
extent of the part) beyond the planes that bound it, so every point of the
surface lies between three sampled points of its own cell and nothing is left
out between two cells. A round capsule cannot follow a right-angled edge: with
the edge on its surface it reaches $(1 - 1/\sqrt 2)\,r$ beyond both faces of
the edge. The capsule of a part with edges is therefore drawn in until the
edge is as far outside it as it is outside the faces; both are then

$$b = \frac{\sqrt 2 - 1}{\sqrt 2 + 1}\, r = 0.172\, r .$$

That is the bound of the set: no point of a drawn part is further outside the
set than 0.172 times the radius of the capsule of its cell, and nothing of the
round shaft is outside its capsule. A test measures it on both instruments on
surface points twice as close as those of the fit.

The layout follows the drawn shape of each part (18 capsules per tool):

| Part | Drawn shape | Capsules | Radii [mm] (needle driver; forceps) | Bound [mm] |
| --- | --- | --- | --- | --- |
| Shaft | tube of 4.18 mm radius with a ring and a fork at its distal end | 1 | 4.37 | 0 |
| Wrist | square in section, 6.3 by 6.0 mm, corners at 3.8 to 4.3 mm | 1 | 3.63 | 0.62 |
| Jaw root | the hub about the hinge (2.41 mm radius, 2 mm thick along the hinge axis) and the block between hub and blade, as far as the jaw is thicker across its closing direction than the hub's radius | 2 side by side across the closing direction | 1.26, 1.28; 1.27, 1.27 | 0.22 |
| Jaw blade | flat (needle driver 1.5 by 2.0 mm at its root, a D in section; forceps 1.3 by 2.1 mm, a rectangle) and tapered to 1.0 by 1.3 mm and 0.6 by 1.0 mm at the tip | a chain of 3 pieces, each 2 side by side across the hinge axis | 0.85/0.80, 0.75/0.74, 0.65/0.65; 0.77/0.82, 0.71/0.71, 0.54/0.54 | 0.15, 0.13, 0.11; 0.14, 0.12, 0.09 |

Measured against the CAD (`capsule_cover`): no point of a drawn jaw is more
than 0.22 mm outside the set (its root; 0.10 to 0.15 mm along the blade),
where the four capsules that the tools had before left the hub 3.3 mm, the
block behind the blade 1.5 mm, the blade 0.14 to 0.43 mm and the last 3 mm of
the forceps 2.1 mm outside the radius that the tissue contact used. The set
reaches at most 0.21 to 0.37 mm beyond a drawn blade (0.63 to 0.72 mm before),
0.7 to 1.1 mm beyond a root where the fork of the wrist is, and 1.4 mm beyond
the wrist at its ends.

The capsules of a blade reach across the plane in which the closed jaws meet by
0.05 to 0.17 mm, the share of the edges of its gripping face; the drawn teeth
reach across it by 0.06 mm. At the 6 mm bite the capsules stand 0.05 mm across
that plane: a needle (collision diameter 0.56 mm) that the jaws hold lies on
the tips of the drawn teeth to within 0.05 mm, where the drawn jaws were
0.15 to 0.20 mm inside it with the capsules of before. The capsules of the two
jaws of a tool therefore overlap when the jaws are closed, their roots lie
side by side along the hinge axis and overlap, and both overlap the wrist
capsule in its fork. The hinge limits keep each jaw on its side of the closing
plane, so the pairs among the wrist and the two jaws of one tool are excluded
from the collision; needle, thread, shaft/jaw and inter-tool contacts are
retained. The rigid model on the device gets the constraint rows that the set
can need (`tool_constraint_rows`: 118, where 64 were enough for one capsule
per jaw).

Two numbers of the grasp follow from the set and are not typed in. A
reset seats each jaw on the needle in its bite (`jaw_opening_on_needle`). For
every capsule of the jaw, the hinge angle at which it touches the wire follows
from its row and the radius of the wire (bisection between the closed jaw and
the largest opening). The capsule with the largest angle meets the wire first
as the jaw closes. The jaw is placed where the capsule beside that one, the
other of the two that stand side by side on that piece of the blade, touches
the wire too: 0.0539 and 0.0511 rad in the default scene, 1.7 and 2.1 mrad
further closed than the first touch, with the first capsule 10 and 12
micrometres inside the wire (with the capsules of before the configured
0.03 rad was the angle of the first touch). The reset is seated, not at the
first touch, because at the first touch each jaw has exactly one of its two
capsules on the wire, at zero gap. Measured 2026-11-17: on the device the
jaws at first touch close by 7 and 9 mrad in the first frame and the jaw
forces go from 0.64 and 0.83 N to 0.88 and 1.14 N in the second frame before
they settle; seated jaws move by 0.3 and 0.5 mrad and their forces go from
0.73 and 0.88 N to 0.72 and 0.74 N at frame 30 without a peak. On the host,
the trial loop of the coupled step without the surface metric alternates
between the contact sets of jaws at first touch and uses up its 160 trials on
a needle pressed on the tissue; on seated jaws it converges in 3. From the
settled grasp on, the two resets give the same demo (jaw angles 0.052 rad,
smallest jaw force of a holding frame 0.602 N, self-test passed in 3 of 3 and
2 of 2 runs).
The regrasp at the needle's tip is planned with the same capsules
([usage](usage.md#the-grasp-at-the-needle-tip)): they stay out of the tissue,
and the open jaws are clear of the wire, by the bound of every capsule, when
they come onto it (`wire_clearance`; 3.5 mm from the tip, where that grasp was
fixed before, the tip is 0.28 mm inside the root capsule of one jaw, 0.26 mm
from the drawn root). Since 2026-11-17 the way from the release to that grasp
keeps every capsule of the set clear of the tissue at rest and of the wire,
and the default procedure takes the tip with a grasp from which the tool
pulls the needle on along its curve
([usage](usage.md#the-way-to-the-tip-and-the-pull)): the plan takes only
points whose descent, closing and pull-out arc the tool reaches with every
joint 0.024 rad (the slide 0.31 mm) inside its limits, by the joints of the
fixed-trocar tool in closed form (`arm_joint_positions`); the way back from
the needle, each step of the rise, the way over and the way down are asked
the same. A limit: after the pregrasp the needle is not looked at again, so
the jaws close on the needle where the plan had it. The pull turns
the needle about its own centre of curvature until the tissue holds no part of
its track and its tail is one thread element past the exit, with the thread
following through the track (above); in a scene without thread tracks until
its tail arrives at the hole of its entry. In eleven self-tests of the pull
through the tissue (seven without the tool/tissue contact, four with
`--tool-tissue-contact swept`) every run completes with 2 entries and 2 exits, all 8 eyelets threaded, the
tail 1.81 to 2.01 mm above the deformed tissue, no jaw off the needle while
it is held or pulled (smallest jaw force in the pull 0.67 N), and
the joint targets 0.39 rad from a limit from the closing to
the end of the pull. In the nine runs of the pull to the entry (three
self-tests without the tool/tissue contact, three with the swept contact,
three more with that contact and records per physics interval) the smallest jaw force in the pull was 0.57 N
and the joint targets stayed 0.36 rad from a limit.
Link inertias, contact friction, force limits
and servo gains are explicit demonstration assumptions. They are not
identified dVRK hardware parameters. No equality constraint fixes the needle
to a tool.

After the pull through the tissue the procedure makes the first step of a
stitch ([usage](usage.md#the-tie-and-the-tightening)). The thread is tied at
the first eyelet of its first passage (`WorkerThread.tie`: the held eyelet of
the [thread](#suture-thread-and-its-needle-attachment)). The request travels
beside the action of one controller call (the `commands` of the scenes'
`step`) and the scene makes the tie between two frames, so it never falls
into an open coupled attempt; the captured step of the thread reads the held
eyelets from device data. Then the palm moves straight along the chord of the
needle, from its tail to the point between the jaws, by the commanded travel
$s$ of the insertion's ramp: speed $v\,u^2(3-2u)$ with $u=t/T$ in the ramp
time $T$, then $v$. A stop from the speed $v\,u^2(3-2u)$ by the same ramp
run backwards covers

$$
d(u)=v\,T\left(u^3-\tfrac12u^4\right),\qquad d(1)=\tfrac12 vT=0.75\ \text{mm}
$$

for $v=3$ mm/s and $T=0.5$ s. Before the tie the palm has gone the same line by
one length of the needle, which draws that much of the loose thread through
the track ([usage](usage.md#the-draw-of-the-thread)). The tightening ends by
the tightness rule of the
thread's section: the smallest volume ratio $J$ of the tissue's tetrahedra at
$J_{\rm tight}$ = `TIGHT_VOLUME_RATIO`, or at the larger value that a scene
publishes for its own tissue under the tie (`tight_volume_ratio`; the grasp
scene: 0.75, because its tissue collapses from $J$ = 0.64). $J$ goes on
falling over the way to rest (by 0.05 over 0.75 mm on this tissue), so the stop
begins where a stop begun now is predicted to end at the rule. With $a$ the
travel of the observed palm along the direction (it is behind its target) and
$k$ the slope of $J$ against $a$ over the last $\tfrac12vT$ of $a$,

$$
J_{\rm rest}=J+k\,(s+d-a),\qquad k=\frac{J-J_w}{a-a_w},
$$

and the stop ramp begins in the first frame with $J_{\rm rest}\le J_{\rm tight}$,
to end at $s+d$. The prediction is linear over the length its slope was
measured on. Its error is what $J$ does that is not linear in the travel. With
the mouths of the track open $J$ still drops once by 0.02 to 0.03 within a
frame or two, at 4.3 to 4.5 mm of travel (cause not established); the slope
over the window takes that drop for slope, the stop begins early, and the
three runs end at $J$ = 0.778 to 0.780 for the rule 0.75, inside the predicted
fall $|k|(s+d-a)$ = 0.067 to 0.070 that the self-test allows. (With the
mouths closed $J$ dropped twice by 0.07 within a quarter of a millimetre.)
The commanded
travel is bounded by 7.5 mm, the travel at which the tissue under the tie
collapsed when it was pulled on without a stop, rounded down; a tightening
that arrives there without its rule
stops the procedure. One tied stitch narrows the top of the wound by 0.57 to
0.59 mm of 2.87 mm at 15.5 mN on the needle while the tool holds; it does not
close the wound, it is not a knot, and nothing holds the thread yet when the
hold of the tightening ends.

Then the thread is tied a second time, at the last eyelet of its last planned
passage, where it leaves the tissue, by the same request beside an action, and
the palm goes back along the line of the tightening by a commanded travel $b$
at the same ramp ([usage](usage.md#the-lock-and-the-slackening)). The thread
to the needle is slack when the size of its force on the needle, $F$, is below
the weight $W$ that the scene applies to the needle (`needle_weight_N` of the
scene's geometry: the needle's mass times the 9.81 m/s² of the scene's weight
kernel). The stop ramp begins in the first frame with $F<W$, from the speed
that the command then has. Its way matters: beyond the point where the thread
is slack the needle pushes the thread together against the tied eyelet, and
$F$ rises again (on the CPU, tightened to the rule: 0.7 mN at its least, 2.8
mN 0.46 mm further, 6 mN where the tightening began). So the command comes to
that frame slowly. With $k$ the slope of $F$ against $b$ over the last
$\tfrac12vT$ of $b$, taken while $F$ falls, the thread is predicted to carry
nothing at

$$
b_{\rm slack}=b+\frac{F}{-k},
$$

and until the stop begins the commanded speed is at most that of the stop
ramp with the travel $F/(-k)$ left, the speed $v\,u^2(3-2u)$ at which
$d(u)=F/(-k)$. A stop that begins at $F=W$ then has the way $W/(-k)$ and ends
at $b_{\rm slack}$, to within the travel of a frame. The slope is taken
against the commanded travel, not the palm's own, because the palm springs
forward as the load falls. The travel back is bounded by the travel of the
tightening. The thread between the two ties then holds the tissue without the
tool: in four device runs the command ends 2.15 to 2.17 mm back with 0.20 to
0.22 mN on the needle, and one second later the wound is 0.39 to 0.40 mm
narrower at its top and 0.21 to 0.22 mm at the track than before the stitch,
59 % and 73 to 74 % of what the tightening closed. It does not close the wound,
and the ties are held eyelets, not knots.

Measured on one DGX Spark (GB10), MuJoCo Warp backend, 2026-11-17,
self-test of `fem_needle_grasp` without the tool/tissue contact (2400 frames;
the contact is switched off today with `--no-tool-tissue-contact`), three
groups of three runs: with the four capsules per tool of before (regrasp fixed
3.5 mm from the needle's tip); with the set in its first form (jaws at first
touch at the reset, regrasp 2.1 mm from the tip, taken before the regrasp was
planned against the tissue); and with the set as it is now (jaws seated at the
reset, regrasp planned). The three runs of the last group shared the GPU, so
its frame times are those of a fourth run, which had the GPU alone while other
work used the CPU, and which passed as well.

| | Before | With the fitted set, first form | With the fitted set as it is now |
| --- | --- | --- | --- |
| Self-test | passes, 3 of 3 | passes, 3 of 3 | passes, 3 of 3 |
| Entries and exits of the needle | 3 and 3: frames 83, 213, 662 and 178, 316, 725 | 2 and 2: frames 83, 212 and 176, 313 | 2 and 2: frames 83, 211 and 176, 312 |
| Lifts completed | 2 | 2 | 2 |
| Regrasp, from the needle's tip | 3.5 mm (fixed) | 2.1 mm (fixed) | 1.84, 1.85, 1.86 mm (planned) |
| Normal force of a jaw on the needle, settled grasp (smaller of the two) | 0.377 N | 0.690 N | 0.691 N |
| The same after the regrasp at the tip | 0.373 N | 0.90 N | 0.87 N |
| Smallest force of a jaw in a frame in which the needle is held | 0.33 N, and 0 in one frame of one run | 0.602 N | 0.602 N; no frame with a jaw off the needle |
| Contacts of a jaw with the needle | 1 (2 in the lifts) | 3 (2 to 4) | 3 (2 to 5) |
| Jaw angle on the needle | 0.029 rad | 0.052 rad | 0.052 rad |
| Roll joint of the needle driver at the regrasp, from its limit | not recorded | not recorded | 0.008 to 0.009 rad (0.45 to 0.49 degrees) |
| Drawn jaws inside the needle wire (frames 60 and 300) | 0.15 to 0.20 mm | 0.03 to 0.05 mm (the tips of the teeth) | not measured again |
| Median frame | 25.2, 25.2, 25.0 ms | 24.6, 24.7, 24.7 ms | 27.8 ms (one run alone) |
| Frames over 66.7 ms | 17, 16, 14 (8 to 9 in the first lift, 7 in the release, longest 227 to 255 ms) | 6, 5, 5 (3 in the release, 2 as the jaws close on the tip, longest 98 to 104 ms) | 7 (3 in the release, 2 while the regrasp is planned, 1 as the jaws close on the tip, 1 in the last lowering, longest 115 ms; one run alone) |
| Graph captures and kernel module loads after start-up | 0 and 0 | 0 and 0 | 0 and 0 |

With the tool/tissue contact on (`--tool-tissue-contact swept`) three whole
runs of the set as it is now complete the procedure (2 entries, 2 exits, both
lifts; regrasp 1.78, 1.87 and 1.95 mm from the tip; 11, 7 and 10 rejected
steps, each taken again at a shorter step; no jaw leaves the needle while it
is held, smallest jaw force 0.66 N; nothing captured or loaded after
start-up). That contact is the default of `fem_needle_grasp` since the same
day: 11 of 11 runs with it completed the procedure with the pull to the
entry in one round, and 8 of 8 the procedure with the pull through the
tissue (five with the option written out, three of the default)
([usage](usage.md#tooltissue-contact)).

The demo therefore ends with 2 entries and 2 exits of the needle, where it
had 3 and 3 with the capsules of before. The third pair of before came in the
lowering of the first lift (frames 662 and 725). Measured frame by frame in
one run of each kind: in the bite the wire lies across the jaws, along their
hinge axis. While the tissue is lifted it pulls on the wire, and the needle
rocks in the jaws: the axis of its turn lies in the plane in which the jaws
meet (within 3 degrees), 27 to 31 degrees from the long axis of the jaws and
80 to 86 degrees from the wire at the bite. On one capsule per jaw it rocked
by 35.2 degrees up to the top
of the lift and on to 38.6 degrees in the lowering; on two capsules side by
side, which carry the wire at two places across each jaw, it rocks by 26.4
degrees. The wire at the bite moves by 0.1 to 0.2 mm in the jaws in both; the
tip, 15.5 mm from the bite, moves by 9.4 mm and by 7.2 mm. So the tip of
before stood 1.4 mm above the level of the tissue surface at rest at the top
of the lift, entered the lifted tissue at frame 662, was below the rest level
from frame 683 to 723 (0.52 mm at frame 710) and came out at frame 725. With
the set the tip stands 3.7 mm above the rest level at the top of the lift and
comes no nearer than 2.19 mm to it (frame 720) and 2.60 mm to the deformed
surface (frame 713). The frames of that lowering no longer run over 66.7 ms.
The needle's tip is within
0.39 mm of its path of before during the insertion (median 0.15 mm), 2.3 mm
from it during the first lift, 0.6 mm during the lift at the tip and 0.3 mm
at the end (medians).

The FEM coupling acts on the needle and thread. Tool/shaft capsules
participate in native rigid-body collisions and in the thread's capsule
contacts. In a scene that does not ask for it there is no grasper–FEM force
of any kind: tool geometry
intersects the tissue without a tool/tissue response, and visible CAD does not
extend the physical interaction scope. A contact of the tool capsules with the
tissue surface is an option of the scene ([below](#tooltissue-contact)), with
a first-version law that carries light touches only and a law on the sweep of
the capsules. The scene (`GraspScene`), `SutureSceneConfig` and the rig
`fem_tool_tissue` keep it off; the example `fem_needle_grasp` runs the law on
the sweep of the capsules by default
([usage](usage.md#tooltissue-contact)). Full-arm collision
geometry and dynamics are absent in every case.

### Tool/tissue contact

The scene's argument `--tool-tissue-contact` selects one of two laws; both are
outside the trial loop of the needle/tissue coupling: the iterated load, the
residual, the corrections and the trial graph are those of a scene without
them. This section states the first-version law,
[ToolTissueContact](../src/newton_solver_suturebot/_src/solvers/coupled/interface/tool_tissue_contact.py),
built by `WorkerThread(scene, config, tool_tissue=True)` (the argument without
a value, or `first`).

The law on the sweep of the capsules (`--tool-tissue-contact swept`,
[SweptToolTissueContact](../src/newton_solver_suturebot/_src/solvers/coupled/interface/swept_tool_tissue_contact.py),
whose module text states it) is built by `WorkerThread(scene, config,
tool_tissue="swept", tool_effective_mass=m)` with the constants of the scene:
the full interval as its sweep horizon, $K_c = m/h^2$ with the smallest
effective mass $m = 1/\lambda_{\max}(J M^{-1} J^\top)$ of each tool at an end
of one of its drawn capsules on the rigid model at the reset state (23.12 g
and 1,331.6 N/m for the needle driver, 20.46 g and 1,178.7 N/m for the parked
forceps), the thickness of the pad as the reach of its surface search and the
stretch limit of the tissue solver. Its calls sit where
the first-version law's do. The update is the last launch of the thread's
accepted step, in the thread's step graph: it reads the accepted state and the
pad surface that the thread's tissue coupling has refitted, places the nodes
that are held through the next interval (positions and velocities of the
accepted state), writes their holds into the tissue matrix
([node_holds.py](../src/newton_solver_suturebot/_src/solvers/fem/node_holds.py))
and stores the reaction on the tools. Before every attempt the worker adds
that reaction to the body forces and the balancing forces of the held nodes to
the nodal forces of the tissue solve. A graph of the thread is therefore bound
to the particle velocities of its buffer as well, and the start-up capture
puts back the tissue of both buffers and the holds.

**Tissue between the jaws of a tool: the pinch.** The swept law has no friction: it
holds a node along its push direction and lets tissue slide across it. With that law the
exchange also builds
[JawTissuePinch](../src/newton_solver_suturebot/_src/solvers/coupled/interface/jaw_tissue_pinch.py)
for every tool whose model names two jaws and a friction coefficient with the tissue
(`GrasperConfig.tissue_friction`, 0.83). It is part of `ToolTissueExchange.update` in every
scene, after the law's update and on what that update published; no scene asks for it and
`fem_needle_grasp` has no command for it. Its rules, per pair of jaws A and B:

- P1 Catch. The closing direction `c` of a node is the unit vector from its foot on the
  axis of the nearest capsule of its own jaw to its foot on the nearest capsule of the
  other jaw, at the poses the law predicted. A free surface node that the law holds on a
  capsule of A with `push . c > 0` is *carried* by A until it leaves the pinch. A carried
  node that the law holds is *clamped* from the first interval in which it pushed back,
  `push . c >= 1 / sqrt(1 + mu^2)` (a face tilted more than the friction angle pushes the
  tissue out), the foot on the other jaw is not at a free end of its capsule set, and the
  two faces are apart there. Closed jaws, whose capsules reach across the plane in which
  they meet, therefore never clamp, and tissue in the groove under jaws that hold a needle
  is carried at most.
- P2 Squeeze and pull. `N_A = max(0, -F_A . c_A)` with `F_A` the force of the law's wrench
  on A; `S = min(N_A, N_B)`; `T` is the size of the sum of the carried nodes' elastic
  forces across the closing direction of their jaws. The pair clamps when both jaws have a
  clamped node and the squeeze is above what rounding can make of none,
  `S > u (K_c + n K_node)`: `u` is the float32 spacing of a tissue coordinate (at the
  largest coordinate of the tissue at rest), `K_c` the held-tissue stiffness of the tool
  (all its touching pieces together turn a depth into force with it), `n` the larger
  number of carried nodes of the two jaws and `K_node` the largest stiffness of a free
  surface node against a move of its own (over those nodes, the sum over its tetrahedra of
  `(lambda + 2 mu_L) V |grad N|^2`). `S` is a sum of forces that are stiffnesses times
  differences of stored positions, and a position is stored to one spacing: a squeeze
  below that sum cannot be told from none. It is a resolution, not a proven bound of the
  rounding error. In `fem_tool_tissue` it is 7.45 nm x (1273 N/m + 14 x 289 N/m) = 40 uN.
  A clamping pair sticks while `T <= 2 mu S` and slips otherwise.
- P3 Stick. Tissue between two faces goes where the faces go. A carried node gets the
  velocity that takes it, in one full interval, to its place in the frame of its jaw at
  the jaw's predicted pose: across the closing direction `c_J` of the jaw, and along `c_J`
  by what both jaws travel alike, `s = minmod(u_J . c_J, u_O . c_J)` with `u_J`, `u_O` the
  travel of the node's own jaw and of the other jaw at the node over the next interval
  (zero where the two differ in sign, else the smaller in size). The node is not moved
  here. It is held (by the law, or by the pinch for a node that the law has let go: its
  flag in the law's hold set and the balancing force), and a held node keeps the velocity
  it is given (the hold leaves 1/4096 of what a free node would change of it), so the
  tissue step takes it there together with its free neighbours and the guard of the
  interval checks that motion like any other: an interval that an element cannot follow
  is rejected and repeated at half its length. A motion that both jaws make alike moves
  the clamped tissue with them whatever the closing direction is, and no depth of the
  contact changes. Where the jaws close, open, or only one of them moves, `s = 0` and
  along `c_J` the node is the law's, with the law's bound on how far tissue can be pushed.
- P4 Slip. No node is carried; nodes that the law holds stay marked, the others leave; the
  travel of the jaws over the tissue is added to the slip length.
- P5 Not clamping. Nothing is written.
- P6 Reaction. A carried node gives its jaw its elastic force less what the law returned
  of it, across `c_J` (times `2 mu S / T` in a slip). The part along `c_J` goes to that
  jaw if the law holds the node, and half to each of the two jaws if only the pinch holds
  it: the two jaws together are then paid all of that part's work in a travel that they
  share and none in a closing in which they travel alike in opposite senses. It acts at
  the node through the next interval, in a wrench array of the pinch that the exchange
  adds beside the law's.
- P7 Letting go. A node that leaves while the law holds it is the law's. A node of a pair
  that clamped while it was carried, that the law does not hold and that carries a force
  is let down. Per interval its own step is taken, that of one block-Jacobi iteration of
  the tissue step for this node alone (`h^2 A_ii^-1 f`, with its block of the tissue
  matrix without a hold), and the largest share `q` of the volume of one of its
  tetrahedra that this step takes (exact: the volume is linear in the place of one node).
  While `q` is above `MOVE_FRACTION` = 1/8 the node is held and gets the velocity of the
  step shortened to that share; with `q` within the share it is free, as it is when the
  law takes it or it is carried again. A node of a pair that never clamped is free as the
  law leaves it.
- P8 Reports and holds. Per tool, two running maxima over the intervals in which the pair
  stuck (since a reset or `clear_reports`): how far a carried node was from its place on
  its jaw when an interval ended, across `c_J`, and how far such a place travels in one
  interval. Nothing in a step reads them. The law's holds are applied once more after the
  pinch.

Invariants, each a test of `tests/test_jaw_tissue_pinch.py`: a scene in which no pair has
clamped keeps every bit (no particle, flag, balance force or body force is written, no node
is kept or let down); storage is fixed and the update reads nothing back; the pinch writes
velocities of tissue nodes and never a position, and two arrays of the law only, `held`
and `balance`, for nodes that the law has let go. The gate for this was measured: with the
cone of P1 the seven closed-jaw phases of `fem_tool_tissue` and a whole run of
`fem_needle_grasp --tool-tissue-contact swept` have no interval in which a pair clamps,
and their accepted states are bit-identical on the CPU with and without the pinch, and
before and after the rules P3, P6 and P7 were changed on 2026-11-17 (784 + 127 frames of
the rig, 40 + 40 of the grasp scene).

The grip is the law's stored force: `S` is what the law presses on the jaws, mostly the
stand-off force of tissue that cannot follow (rule 5 of the law), so a change of that rule
changes the limit `2 mu S`. Limits: at the lower end of the cited friction range (0.11) no
face is inside the friction cone and nothing clamps; a lift beyond the friction limit of a
grip ends in a slip that the rules above do not bring to rest (below).

**How far carried tissue is from moving jaws.** The carry (P3) is a step at constant
velocity to the place that the law predicts for the next interval. When that interval has
ended a carried node is off its place by the error of the prediction (the change of the
jaw's travel from one interval to the next: at most one interval's travel while the jaws
do not turn back within an interval) and by what the hold left of the move (at most
3/4096 of it: the hold adds 4096 times the mean diagonal entry of the node's block of the
tissue matrix, and the largest eigenvalue of that block is at most its trace). So, while
both jaws travel alike,

    distance from the place at the end of an interval  <=  (1 + 3/4096) x travel of the place in one interval.

A carry that is slow by a share of the travel adds that share in every interval and is
outside after as many intervals as the inverse of the share. While the jaws close or open
the law itself moves the nodes on the faces, across `c_J` too, and the distance holds that
move: 23.9 um against a travel of 15.4 um while the jaws of `fem_tool_tissue` close
(printed with the row, not judged).

Measured in the default pinch of `fem_tool_tissue` (close 0.42, lift 5 mm; CPU, the phase
from its restart, run `f_default`, 2026-11-17): from the end of the squeeze hold to the
end of the lower hold the largest distance is 4.55 um against a travel of 13.70 um; at the
end of the lift the 27 carried nodes are 0.2 um above their places in the mean, the
farthest 18.3 um; at the end of the lift hold none is more than the tracking tolerance of
the tool (39.5 um) from its place (farthest 19.2 um), at the end of the lower hold none
(farthest 34.7 um). With the rule before (the node moved by a step that the stretch of
its elements bounded) the same rows read, at a lift of 3 mm (run `s3_pinch_test`): 1644 um
at the end of an interval, one node 472 um behind at the lift hold, three nodes up to
223 um off at the lower hold.

What the same grip does beyond that (CPU, same day):

- The squeeze grows with the lift (275 mN at the squeeze hold, 331 mN at the end of a 5 mm
  lift, 478 mN at 8 mm, 697 mN at 12 mm) and the pull with it (108 N/m of rise). The
  friction limit over the pull is 1.61 at the end of a 3 mm lift, 1.16 at 5 mm (1.27 at the
  end of its hold), 1.06 at 8 mm and between 1.04 and 1.09 from there to 12 mm commanded
  (10.6 mm of rise): with the script's holds of 0.5 s the pair does not slip in a lift up
  to 12 mm (runs `s7_l3`, `f_default`, `f_l8`, `x_l12`). With the squeeze of the squeeze
  hold alone the pull would reach the friction limit at `2 mu S_0 / k_T` = 4.2 mm of rise,
  below the 4.4 mm of rise of the default lift (4.5 mm after its hold): the default lift
  is held by this growth of the squeeze, which is measured and not derived.
- The elements are what ends a lift. The smallest stretch of the tetrahedra without a
  carried node (the tent) is 0.34 at the end of a 5 mm lift and 0.29 at 8 mm, and reaches
  the stretch limit 0.2 of the material at 9.4 mm of rise (10.6 mm commanded); the smallest
  volume ratio at the carried nodes is 0.11, 0.08 and, at 12 mm, 0.03.
- A lift of 8 mm is held, lowered and released without a rejected interval; after the
  lowering one node is 44 um from its place (row 3a of the self-check is not met). A lift
  of 12 mm is held and fails while lowering: nine frames in, the squeeze has fallen by a
  tenth and the pair slips for one interval, which lets go of the 18 nodes that only the
  pinch held; it sticks again on the 9 that the law holds, the tissue hangs on them while
  the others are let down, and three frames later a tissue step is rejected to failure
  (smallest volume ratio 0.017). A slip under load that returns the tissue to rest is not
  built.
- No largest lift of the grip is derived. The default lift of 5 mm is a choice of the
  script, beyond `2 mu S_0 / k_T`; the self-check is met there with the script's holds and
  not at 3 mm (row 8, below), not at 8 mm (row 3a) and not at 12 mm.
- The margin of the default lift does not cover a longer hold before it. With holds of 2 s
  (`--hold-seconds 2.0`) the squeeze at the end of the squeeze hold is larger (324 mN: the
  law's stored force has grown and one jaw stands 0.025 rad further open) and falls
  through the lift, to 256 mN at the end of a 5 mm lift: the pair slips for two intervals,
  the 10 nodes that only the pinch held leave, and the carried tissue ends 0.77 mm behind
  its places in the mean (3.5 mm up; no interval is rejected and the release leaves the
  tissue at rest; rows 3a, 3b and 9 of the self-check are not met; run `f10_hold2`). A
  3 mm lift is held after the same hold (limit over the pull 1.32 at its end, 1.13 after
  2 s of lift hold, in which the squeeze falls from 230 to 197 mN; run `x_hold2_l3`); a
  close of 0.36 slips in a 5 mm lift as well (run `x_hold2_c36`).
- Without the halves of P6 (the force along `c_J` of a node that only the pinch holds
  reaching no body) the same grip slipped at 6.84 mm of rise in an 8 mm lift and failed in
  the same way (run `s6_l8`): the halves pull the jaws onto the tissue that pulls on them.
- A firmer close holds with more to spare and squeezes the elements at the faces thinner.
  At a close of 0.36 of the ridge width (squeeze 364 mN, 30 carried nodes) the limit over
  the pull is 1.33 at the end of a 5 mm lift, but the smallest volume ratio is 0.04 and, in
  an 8 mm lift, 0.02, and after the lowering one node (5 mm) or two (8 mm) are 49 and 67 um
  from their places (runs `s8c_l5`, `s8c_l8`).
- Through a lift and a lowering the law's own force on the tool along the lift grows and
  does not come back (0.38 N at the squeeze hold, 0.65 N at the lift hold, 0.69 N at the
  lower hold of a 5 mm lift), and one jaw ends 0.03 rad further open than it began. Nodes
  that only the pinch holds do not follow a jaw that opens alone, so at the lower hold the
  farthest node is 35 um from its place, of 39.5 um allowed. That number is the height
  alone, as row 3a takes it. Of the 27 carried nodes the pinch alone holds 3 at the squeeze
  hold, 13 at the lift hold and 18 at the lower hold, and the points of the jaw bodies that
  coincided with the carried nodes at the squeeze hold are then up to 0.23 mm and 0.36 mm
  from them along the closing direction (across it at most 6 um): rows 3a and 3b do not
  judge that direction (an independent record of every interval, 2026-11-17).

**The work through the carried nodes.** What the tool pays is the force that the carried
nodes give the jaws (the law's part and the pinch's) along the travel of the jaws at the
nodes; what the tissue receives is the elastic force on the nodes along their moves. The
two differ by the one-interval lag of the reaction (through an interval the tool feels the
force of the state before it: half of the change of the force times the travel, which adds
up to `T d / 2` over a lift and again over a lowering) and by the work of making up the
distance `r` of a node from its place (`T r` per stroke):

    | W_tool - W_tissue |  <=  T_max d_max + 2 T_max r_max      over a lift and a lowering, half of it in each,

with `T_max` the largest pull, `d_max` the largest travel of a place in one interval and
`r_max` the largest distance (the two reports of P8; `r` counts as far as the carry rule
allows it). Measured from the frame records (`Measurements.work_row`, forces at the two
ends of each frame), default pinch, run `f_default`: the tool pays 1.1242 mJ and the
tissue receives 1.1250 mJ in the lift and its hold; the tool gets 1.0702 mJ back for
1.0734 mJ in the lowering and its hold (differences 0.0008 and 0.0032 mJ, 0.0055 mJ allowed
in each). The sums over every interval, with the tool's work taken from the wrenches as
they were applied (run `iw_default`), are 1.1209 mJ paid for 1.1249 mJ and 1.0727 mJ back
for 1.0732 mJ: the tissue's sums agree with the frame records to 0.0002 mJ, the tool's
differ from them by the lag that the frame records leave out (0.0033 and 0.0025 mJ; the
lag term of those intervals is 0.0031 and 0.0024 mJ). With the rule before, the frame
records read 0.0068 mJ against 0.0063 mJ in the lift of 3 mm, and an independent sum over
every interval 0.040 mJ over the cycle. At a lift of 3 mm the lowering is 0.0038 mJ off
against 0.0034 mJ allowed (run `s7_l3`): the row is not met there, by 0.0004 mJ of a
stroke of 0.40 mJ; no single cause of that remainder was separated. Summed over every
interval (an independent record, 2026-11-17) the lift with its hold is 0.0041 mJ off at
3 mm (0.4250 mJ paid for 0.4290 mJ) as it is at 5 mm (1.1209 mJ for 1.1250 mJ), outside
the 0.0034 mJ allowed at 3 mm: that part does not grow with the pull, as the bound does.
So the bound is derived from the two terms above and is not a bound of all that is
measured: it is met at the default lift and not at 3 mm. Not in this balance: the work
while the jaws close and open (the law's placements, slips, the release) and the law's
own force on the tool.

The self-check of the pinch in `fem_tool_tissue` holds a run to this
(`Measurements.pinch_rows`): every carried node at its place at the end of the lift hold
and of the lower hold (row 3a); the distance bound above from the end of the squeeze hold
to the end of the last closed segment, and the tissue risen with its places at the end of
the lift (row 3b); the work bound (row 8); and the grip: no slip while the jaws are closed,
the friction limit at least the pull at the end of every frame, the tetrahedra around the
grip above the stretch limit (row 9).

One consequence of placing nodes of the accepted state after the coupled step
has accepted it: the needle interaction keeps a host copy of the accepted
tissue positions and answers the swept test of the needle tip from its device
candidates only while the start positions of the next step equal that copy.
After a placement they do not, and the test takes bounds over every surface
triangle of the pad on the host. Its result is that of the actual start
positions; its cost is 10 ms per interval in which a node was placed
([usage](usage.md#on-the-sweep-of-the-capsules)).

Samples. Each capsule of the tools carries a chain of spheres on its axis,
fixed on its body, at most one sphere radius apart and reaching, cap to cap,
as far as the capsule does. The spheres have the radius of their capsule: the
contact uses the capsule set of the tools
([above](#the-collision-capsules-of-a-tool)) and no radii of its own. For the
registered tools that is 58 spheres on a shaft, 3 on a wrist and 34 (needle
driver) or 42 (forceps) on each jaw: 274 in the scene.

Law. At the accepted state of an interval every sphere $s$ takes the nearest
point of the pad surface (one signed query of the bounding volume hierarchy
that the thread's tissue coupling has just refitted, within the 20 mm
thickness of the pad), its gap $g_s$ (negative inside the tissue) and the
outward direction $\mathbf n_s$ there: the normal of the triangle, or over an
edge or a vertex the direction between the surface point and the centre. A
sphere is active while it carries a force or is inside the tissue; $n$ is the
number of active spheres of the scene, counted on the device. Then

$$
f_s \leftarrow \max\!\left(0,\; f_s - \varphi\,\frac{K}{n}\,g_s\right),\qquad
\varphi = \min\!\left(1,\tfrac12(\omega h)^2\right).
$$

$-f_s\mathbf n_s$ is spread over the tissue particles of the triangle by the
barycentric weights of the surface point (the share of fixed nodes and of
vertices outside the active region goes to the ground), and $+f_s\mathbf n_s$
acts on the body of the sphere at the surface point. Both are stored and
applied through every trial of the next interval, the nodal forces with the
thread's forces on the tissue and the wrench with the thread's wrenches, so
the two sides are equal and opposite as applied. $\max(0,\cdot)$ is the
contact condition (a contact cannot pull); the force has no upper limit, and
the law has no stiffness, damping or speed constant.

Parameters, none of them tuned:

- $K = \tfrac14 E\bar h$, the stiffness of one tissue node that already relaxes
  the thread's forces on the tissue: 8.2 N/m for $E = 25$ kPa and the mean
  tetrahedron edge $\bar h = 1.31$ mm.
- $\omega = \tfrac{\pi}{2}\sqrt{G/\rho}/H$, the lowest shear frequency of a
  layer of depth $H$ on a fixed base, with $G = E/(2(1+\nu))$ and $H$ the
  vertical extent of the active region (11.9 mm): 401 rad/s, so $\varphi = 1$
  at 1/240 s, 0.35 at 1/480 s and 0.087 at 1/960 s. $h$ is read from the step
  time on the device; a refined interval needs no other graph.
- $K$ and $\omega$ are device data.

The step. One update is one projected-gradient iteration of the frictionless
contact problem of the spheres against the tissue, with the step $K/n$. With
$W$ the compliance of the tissue between the spheres, it does not overshoot
while $(K/n)\lambda_{\max}(W)\le 1$ and is stable below 2. $W$ is symmetric
positive semi-definite, so $\lambda_{\max}(W)\le\operatorname{tr}W\le n/K$ as
long as no loaded surface point is softer than $K$, for any number and
placement of contacts. $\varphi$ keeps that margin when a rejected interval is
halved. The fixed point has zero gap at every loaded sphere, whatever the
stiffness of the tissue; the price is a delay, with a time constant of the
stiffness of the touched patch over $K$, in intervals.

Where that argument ends. It takes $W$ as given, that is, loads whose points
of application stay where they are. Each sphere applies one point load at its
nearest surface point, and two cases are outside the argument. Both end in
tetrahedra that the step control rejects (measured, see
[usage](usage.md#tooltissue-contact)):

- Held tissue. A point on a triangle with fixed vertices, or with vertices
  outside the active region, hardly yields: a triangle with one free vertex of
  weight $w$ and nodal stiffness $k$ has the compliance $w^2/k$, and the free
  node moves by $1/w$ of what the point moves. The force then rises by
  $K|g|$ per interval until the tool gives way, which takes the stiffness of
  the tool times the overlap (by an estimate of the tool's static stiffness,
  not measured, 0.3 to 1 N for 1.5 mm), and the free node takes the share
  $w f$. In the default procedure the tip of a jaw meets the pad on the border
  of the active region; the free node beside it is pushed through its
  tetrahedra at 0.1 N. A scripted chain that does not yield reaches 1.9 N
  there in 1.2 s.
- The pit of a point load. Under one load the surface forms a pit, and the
  nearest point of the sphere moves from the pit to its flank; the load moves
  with it, the pit recovers and the flank sinks. On the 1.3 mm tetrahedra of
  the demo a dent of 0.3 mm is carried. From 0.4 mm (nine spheres) or 0.7 mm
  (one sphere, 11 mN) the nearest triangle alternates between two neighbours
  from one interval to the next, with a growing amplitude at an almost
  constant force, until a tetrahedron inverts (first rejection at 0.80 to
  0.86 mm, 10 to 20 mN on a sphere).

A contact that is to carry a jaw driven into the tissue therefore needs what
this one leaves out: a load that is spread over the patch under a sphere and
moves smoothly with the surface, and a treatment of tissue that is held.

Where it runs. The three kernels of the update (clear, nearest points with
the count, forces) are part of the worker thread's accepted step and of its
two captured graphs; the sample forces and the stored wrenches are part of
`WorkerThread.state`, which the start-up capture copies back after its
discarded steps. The kernel that adds the wrench runs with the thread's loads
before every attempt. A rejected attempt does not advance the thread, so the
stored forces are applied again at the shorter interval. Storage is fixed by
the capsules: one force per sphere, no contact list, nothing that can
overflow. The step reads nothing back; `ToolTissueContact.diagnostics()` is an
explicit read for reports and tests. A reset builds a new worker and with it
a contact without forces.

Limits beyond the two above. No friction. One nearest point per sphere, so in
a fold narrower than the tool only one side pushes. Spheres on capsule axes,
not the CAD surface. The contact needs the worker thread and its surface mesh,
so it exists only in the physical-grasp scene.

The corrected native contact configuration uses `implicitfast`, elliptic
friction cones, `impratio=10`, tolerance $10^{-12}$, at most 200 Newton
iterations and 50 line-search iterations. Elliptic cones change the contact
model, while the tighter tolerance changes numerical solution accuracy. The
comparison retains the same geometry, masses, gains and friction coefficients:
sliding coefficient 1, torsional length 0.28 mm, rolling length 20 µm and
normal impedance 0.9999. This follows MuJoCo's recommendation to combine
elliptic cones, high impedance ratio and a tight Newton solve when slip is a
problem. Soft contact can still slip under sustained load; it does not impose
a no-slip constraint. See [MuJoCo's solver guidance](https://mujoco.readthedocs.io/en/3.17.0/modeling.html#solver-settings).

Contact integration resolution matters independently of the iteration cap.
The earlier free-palm [contact comparison](validation.md#physical-grasp-contact-correction)
separates the smooth transport fixture at 1/240 s from stronger incremental
commands that passed at 1/960 s. Those stress measurements used 16 complete
rigid/FEM/thread intervals per frame. A subsequent independently measured
schedule uses [SubcycledMuJoCo](../src/newton_solver_suturebot/_src/solvers/mujoco/subcycled_mujoco.py):
four native steps inside each of four outer FEM/thread intervals per frame.
Each native step resolves contacts and finite joint actuators; external world
wrenches and target commands remain held for the complete outer trial. The
current RCM model uses native finite joint actuators rather than that earlier
free-palm Cartesian drive. Historical free-palm measurements do not establish
RCM insertion success or parallel performance.
Only the final rigid/joint state is exported, without writing FEM particles.
The enclosing partitioned transaction restores the entire native integration
state on rejection. Thread and material history advance only after an accepted
outer interval. This finer rigid resolution does not claim a fully implicit
thread coupling or continuous collision detection.

Each native microstep is checked before its endpoint is exported.
[NativeMicrostepGuard](../src/newton_solver_suturebot/_src/solvers/mujoco/native_step_guard.py)
rejects fresh MuJoCo bad-position, bad-velocity or bad-acceleration warnings and
an unexpected native clock advance. MuJoCo can reset an unstable trial to a
finite state; endpoint finiteness alone would therefore miss that failure.
Fresh warning sentinels also detect a reset at time zero or after an earlier
warning whose counter was already one. Warning records retain their cumulative
meaning after successful microsteps and roll back with the accepted native
integration checkpoint after rejection. A separate monotonic rejection counter
records failed trials. The existing adaptive interval retry handles these
failures; physical tolerances, contact settings and solver budgets are unchanged.
This guard detects native integration resets, rather than every possible
MuJoCo warning such as an invalid control or exhausted contact capacity.

The physical-grasp scene also enables
[HeldFEMStart](../src/newton_solver_suturebot/_src/solvers/fem/held_start.py) for
the supported native four-microstep configuration. Within one attempted coupled
interval, the accepted particle positions, velocities, control and assembled
FEM tangent are held constant. Their original internal/gravity/damping force
impulse and elastic predictor are evaluated once into owned arrays. Each trial
then preserves the original order: restore that impulse, add the current
interface and external load, subtract the cached predictor, and restore the
spring damping predictor term. The ordinary full-region compact solve, current
linear tolerance and physical residual checks remain authoritative.

Cache preparation happens before capture and is repeated for every accepted or
retried interval, including a reduced timestep. Storage identity, control
identity, tangent generation and differentiation revision guard reuse; graphs
retain the arrays they reference. A `finally` block clears the active scope on
success or failure, and reset also invalidates it. The accepted state and control
must remain held throughout that scope. Custom FEM force methods, gradient/Tape
execution, streaming or component-scaled operators, captured tangent refresh,
and unsupported contact modes use the original force assembly. Other examples
do not enable this cache by default.


For this same supported CUDA native-grasp path, `response_basis_guesses` enables
[AffineGuessManager](../src/newton_solver_suturebot/_src/solvers/fem/affine_guess.py)
and [WarpBasisProvider](../src/newton_solver_suturebot/_src/solvers/fem/warp_basis_provider.py).
They build a numerical response basis on the first compact free-node matrix
$A_0$: $Z \approx A_0^{-1}E$, where $E$ selects the committed track's free
scalar degrees of freedom. Warp CG constructs only missing columns when this
support grows, retaining the original $A_0$. Its setup tolerance is $3\times10^{-4}$
and its cap is 160 iterations; these control an approximate initial guess and
never replace or loosen the physical solve's tolerance or budget. Each column
records its true double-precision residual, including capped solves. No cuDSS,
SciPy or external solver dependency is needed at runtime.

The provider keeps its columns in fixed storage that is made once with its
column engine, at the start-up of the scene: one slot per column for up to the
capacity of the guess, and a second array that receives the columns of a
request in the order of its support. A request allocates nothing. It reads the
device once: the residual records of the columns solved since the last
request, by which the host refuses a nonfinite column before the basis is
loaded.

**Response columns per step.** The columns are solved in the attempt that
follows a growth of the track, one per degree of freedom of the nodes that
carry track samples. A sample in the next tetrahedron adds one node (three
columns); the first sample of a new section adds the four nodes of its
tetrahedron (twelve columns, 17 to 20 ms on the GB10 in the step after an
entry). With `AffineGuessManager(nodes_per_attempt=n)`
(`--response-basis-nodes n`, off by default) an attempt solves the columns of
at most `n` nodes into the provider's storage. The others follow in the next
attempts, and the basis is loaded when the last column is solved. Until then
the guess is withheld, as before the first track sample. An attempt that only
solves columns allocates nothing and reads nothing from the device; the
attempt that loads the basis makes the one read of an unbounded build. Every
column is the solve of the same unit load on the same reference matrix $A_0$
as in an unbounded build. That the columns are then the same was compared bit
for bit for the provider on a small matrix, on the CPU and on CUDA
(`tests/test_response_basis_bound.py`). In the grasp scene the bounded and the
unbounded build loaded the same number of bases with the same column counts;
their column values were not compared there.

For a later trial, the candidate is
$x_{\rm guess}=x_{\rm previous}+Z E^T(b-b_{\rm previous})$.
The RHS difference comes from the original assembled physical RHS, preserving
float32 assembly order. Unknown loads outside $E$ are omitted from this guess.
The helper checks that the original warm start still matches its stored
history, then compares both candidates with the **current** full free-node
operator using double-precision squared residual sums. Nonfinite, differently
normalized, history-mismatched or non-improving proposals keep the original
guess. Original current-matrix CG and endpoint reconstruction always follow.

The same columns initialize the shaft contact preconditioner's solve through
[WoodburyInitialGuess](../src/newton_solver_suturebot/_src/solvers/coupled/acceleration/woodbury_guess.py).
A small host contact system uses the current rigid metric and shaft blocks,
with the shear of the sliding anchors of a friction tunnel;
a device matrix-vector product expands its coefficients into a candidate.
The candidate omits surface terms and unsupported RHS components, so its
residual is tested against the complete current operator, including simultaneous
surface contact. A proposal is retained only when that residual is finite and
smaller than the zero-start residual. Otherwise the original zero start is
used. In either case the original preconditioner CG, balanced correction and
physical coupled acceptance remain unchanged.

Numeric tangent or timestep changes may retain $A_0$ because it is only a
residual-checked guess. Topology, fixed-node mapping, matrix storage and reset
invalidate the binding. Capacity is 96 scalar response columns and 32 shaft
anchors; unsupported cases retain the existing solve. Preparation occurs
outside capture, and graph caches strongly own referenced basis and scratch
arrays. Trial history resets on internal retries; a `finally` block disables
the helper after each attempt. Scene rebuild and worker shutdown close owned
response graphs. Gradient/Tape, custom solver methods and unsupported solver
configurations use the original paths. An exception during basis setup, including
nonfinite column diagnostics, propagates through the ordinary transactional
rollback rather than silently hiding a CUDA or numerical failure. Track-count
metadata is committed only after a usable support binds successfully, so an
oversized or failed support cannot reactivate an older smaller binding on retry.
No physical position, contact law,
friction history, material constant or selected FEM region is replaced.

CUDA physical-grasp worker initialization explicitly loads the enabled shaft
preconditioner, tunnel-ALM and phase-predictor Warp modules. Loading compiles or
loads kernel code only: no integration, force evaluation or history update is
performed. Gradient/Tape and CPU paths keep their existing loading behavior;
benchmark initialization cost remains separate from physical stepping.

The native bridge imports the incoming joint state, held controls and body
wrenches once, executes the four native microsteps, and exports only the final
rigid/joint state. Supported CUDA layouts pack these transfers into persistent
host/device buffers and capture their conversion kernels. Native coordinates
retain MuJoCo precision between microsteps instead of passing through Newton's
float32 state after each microstep; endpoint conversion remains unchanged.
Unsupported layouts or requested observables retain the general adapter.
Contacts, actuator updates, integrator settings and native iteration budgets
are unchanged, and rejection restores the complete native integration state.
Jaw normal forces and contact counts are the solved native outputs from the
last rigid microstep, while the exported poses are its endpoint coordinates.
Their sample times can differ by one microstep; a single missing bilateral
contact sample does not establish a sustained dropped needle.

Within a supported coupling transaction, accepted joint coordinates, held
controls and all body loads except the needle wrench remain constant.
[RepeatedNativeInput](../src/newton_solver_suturebot/_src/solvers/mujoco/mujoco_repeated_input.py)
performs the complete Newton float32 conversion on the first trial. Subsequent
trials restore that converted input packet on the CPU and replace only the
needle's six-component wrench. The original Newton force kernels perform
body-gravity correction and free-joint-force addition in their original
float32 order; the wrench packet contains 24 bytes. The caller still restores
the complete native integration checkpoint before each trial. Reusing the
checkpoint's double-precision coordinates instead of the first converted
input would change the trial and is not done here.
Only buffer allocation survives transaction completion or failure; input
values are invalidated in `finally`. A new timestep, accepted state or control
therefore starts with a full conversion. Unsupported layouts, custom rigid
steps and differentiation retain ordinary imports. This removes redundant
coordinate/control conversion work, but keeps the required GPU-to-CPU stream
synchronization for each trial's new wrench. It neither runs native MuJoCo
asynchronously from its required force nor removes that dependency.
`native_transfer_repeated_readbacks` counts these wrench-only imports as a
subset of `native_transfer_readbacks`, not as avoided synchronizations.
For the unmodified supported FEM/subcycled-MuJoCo pair, native inputs are
imported before the independent FEM solve is submitted on the current CUDA
stream. CPU MuJoCo can then integrate while that GPU solve runs; final rigid
export is ordered after FEM on the same stream. The callback must not change
rigid inputs, controls or native state. Custom solver steps retain sequential
execution, and exceptions still return to the enclosing rollback transaction.

The host-controlled CUDA path also captures trial preparation: staging the
current interface load and copying the complete incoming Newton state into
the candidate, including custom namespaces and history arrays. A bounded
cache binds graphs to array storage; replay reads the current device values.
The first actual preparation warms resources without an extra physical
step. Unsupported layouts and gradient-enabled execution use the original
operations. Native rollback, acceptance and history commits remain outside
this graph. `trial_prepare_graph_captures` and `trial_prepare_graph_launches`
report its use separately from force and integration graphs.

Package-owned physical coupling, native transfer and thread graph captures use
[ScopedCapture](../src/newton_solver_suturebot/_src/utils/capture.py) to defer automatic
Python cyclic garbage collection until capture has ended. An unreachable old
solver can own pinned host memory; freeing that memory during a later CUDA
capture invalidates the graph even though the new solver's arrays remain valid.
The scope restores the original GC setting on success or exception and accounts
for nested or overlapping scopes. It performs no collection or synchronization
by default, and graph replay is unchanged. Explicit resource destruction or an
external `gc.collect()` still must occur outside capture. Tests retain both the
general BiCGStab path and an explicitly configured compact-CG capture path.

The GC-only `PauseGC` scope also surrounds project-owned Warp Krylov calls,
including response-basis setup. Warp can create a conditional graph internally
during an ordinary CG, BiCGStab or GMRES call, before an application's explicit
capture. This scope starts no graph and changes no linear stopping rule.
[NVIDIA's capture guidance](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/capture-failures.html)
identifies pinned-memory release through `cudaFreeHost` as illegal during capture.

Articulated tools still require native CPU MuJoCo, so the complete conditional
coupling loop cannot run as one CUDA graph. A separate partial graph captures
certified geometry queries, needle/tissue force evaluation and residual
reductions while the host retains native dynamics and convergence decisions.
Eligibility includes certified free-air motion, external subthreshold
contact and established-track cases once the previous accepted tip is
available. External contact requires no swept crossing and the original
surface kernel's localized tip force strictly below the puncture threshold,
with a relative `1e-6` guard margin. For an unpunctured tip, existing openings
are allowed only when a separate double-precision certificate proves that the tip is farther than
`2 * max(needle_radii)` from every opening's current barycentric material
position, including tip-conversion and arithmetic uncertainty. This checks
actual deformed openings rather than treating proximity to any tissue face
as possible opening reuse. With no openings, this condition is vacuous.
Threshold proximity, possible opening reuse, inconsistent material state or
uncertain geometry restores the complete original host transaction. The
puncture threshold and opening metadata are part of the capture signature.
Final metadata retains the exact CPU tip embedding and publishes `contact`
when the unchanged kernel reports a positive subthreshold force; it cannot
stage an opening. These certificates neither change puncture criteria nor
authorize a new or reused opening.

A separate stable branch preserves an existing puncture authorization after the
tip has left the tissue. It requires the previous accepted state to be outside
and already punctured, the candidate tip to remain outside every material cell,
no swept boundary crossing, and a strict double-precision proof that the tip is
within `2 * max(needle_radii)` of the **current** opening's deformed barycentric
position. The distance margin includes FP32 tip conversion and arithmetic
uncertainty. The eager interaction retains that same opening on this branch and
does not perform opening reuse while already punctured. The captured outside
tip mask is explicitly zero, independent of previously visited tetrahedra.
A fresh retained-opening proof is cleared and recomputed for every candidate;
its arrays belong to the graph binding, and opening data are refreshed even
when the opening count is unchanged. Exact host staging independently repeats
the branch and distance checks before publishing metadata. Near-threshold,
missing, ambiguous or inconsistent geometry retains the original whole-interval
fallback. No physical force kernel, puncture criterion, accepted event or solver
budget is changed by this extension.

Finite-state and conservative swept/material-permission checks precede use of
the result. Ambiguous permissions restart the transaction through the original
exact host path; unsupported or gradient-enabled configurations select that
path directly. Accepted surface events and puncture history remain host-owned.
Diagnostics distinguish these partial force-graph launches from complete
coupling-graph launches and record fallbacks.

The swept boundary certificate first tests each candidate face against the
union of its old and new triangle bounds. For overlapping bounds, the signed
coplanarity polynomial

$$
p(s)=(x(s)-a(s))\cdot[(b(s)-a(s))\times(c(s)-a(s))]
$$

is cubic because the tip and vertices move linearly over the interval. Its four
Bernstein controls enclose all polynomial values on $[0,1+10^{-7}]$, including
the endpoint allowance of the original CPU root oracle. A face is certified
clear only when every control lies strictly above a positive margin, or every
control lies below its negative. The margin is
$10^{-8}REF+2\delta EF+10^{-30}$, where $R,E,F$ are endpoint $L_1$ bounds on
the relative point and two edges, and $\delta$ bounds tip-transform uncertainty.
It includes the oracle's coefficient truncation/near-real-root conventions and
the difference between float32 point transforms and the graph's double sample
coordinates. The permitted 10 nm local-tip discrepancy is componentwise;
its bound includes all three components and rotation amplification. Swept and
free-air proximity boxes include that uncertainty.

Mapped faces use actual previous/current active-node coordinates; unmapped
reference vertices remain in helper-owned float64 storage included in the graph
signature. Nonfinite values, degenerate endpoint triangles, materially nonunit
quaternions, uncertain controls and near-boundary cases retain the original
host moving-triangle oracle. This certificate only removes proven-empty queries;
it does not produce event roots, alter a force law, accept a physical step or
publish history. Final native endpoint, swept-event and material checks remain.

The exact CUDA host force path initially queries only the tip's containing
tetrahedron in every material phase. The standard device shaft-permission
mask consumes current geometry directly and needs no host shaft embeddings.
If that mask cannot certify the candidate, the complete shaft query batch is
prepared lazily before the original exact CPU mask requests those embeddings.
This skips unused candidate downloads and CPU containment solves while
retaining the original float64 tip predicate, boundary permission, fallback
mask and surface forces. Customized query/mask methods and subclasses retain
the complete original query batch. The optimization changes which unused
queries are omitted, not the geometry used by a queried predicate.

On a CUDA host fallback with an established accepted track, the lateral shaft
permissions can still use the existing device geometry query. A private cached
permission helper evaluates samples against visited tetrahedra and continuous
material segments; it does not certify the tip. The original CPU tip-boundary
rule supplies sample zero and its outgoing-face location under exactly the same
conditions as the full CPU mask. Every visited tetrahedron is checked for degeneracy, including cells outside
all sample AABBs, so the original CPU singular-cell behavior is preserved.
Degenerate geometry requests that original complete mask. Empty history, custom interaction methods, changed/stale query
geometry, Tape and gradient-enabled states retain the CPU path. Replacing
sample/topology owners or query storage rebuilds the helper; strong owner
references prevent identity reuse while GPU copies remain cached. Accepted
track growth refreshes its metadata. Only the small permission mask and fallback flag return to the
host. No entry, exit, track or multiplier history is published by this query,
and the same surface/tunnel kernels evaluate forces afterward.

After a converged primal evaluation, an ALM dual-only update leaves the applied
load, candidate endpoint, geometry and surface response unchanged. The partial
path reuses those values and reevaluates the changed tunnel forces and full
physical residual. This evaluation still consumes the shared iteration budget;
if the residual requires a load correction, the next trial restores native
state and solves both mechanics again. Accepted history is committed only after
the unchanged primal and dual convergence checks.

The physical-grasp viewer optionally pauses its insertion arc for a small
inspection lift. The trigger requires a measured entry, no exit, an inserted
tip, at least 1 mm of currently engaged shaft and bilateral jaw contact. The
gripper target moves along the normalized entry-surface normal using
`h * (3*u**2 - 2*u**3)` over 0.75 s, holds for 0.5 s, then reverses the same
profile. Its original arc progress is frozen throughout the two-second motion;
contact loss pauses its motion clock. All targets pass through the existing
rate limits, trocar-constrained joint IK and finite actuators. No needle or
tissue state is imposed, and accepted physical steps retain the same residual
and rollback rules. The generic controller defaults to zero lift so existing
insertion-only benchmarks keep their trajectory.

The procedural worker can seed the interface load from completed control
frames at the same substep phase. For phase $p$ of frame $k$, one stored frame
supplies $f_{k-1,p}$; two supply the candidate $2f_{k-1,p}-f_{k-2,p}$. A device
guard retains the current accepted load if engagement, finiteness or accepted
load continuity fails. Excessive extrapolated growth falls back to the previous
phase load. The same coefficients apply to the complete tissue-force/rigid-wrench
vector, preserving its balance. This is only an initial numerical guess:
every subsequent force evaluation, convergence criterion, physical acceptance
check and iteration budget remains unchanged.

With the GPU rigid backend and its device-Anderson policy the guess follows the
change of the load instead of the load: $f_{k,p-1}+f_{k-1,p}-f_{k-1,p-1}$, the
last accepted load plus the change accepted over the same phase one frame
earlier. With $d_1$ and $d_2$ the last two such changes, it is offered only if
$\lVert d_1-d_2\rVert_\infty<\min(\lVert d_1\rVert_\infty,\lVert d_2\rVert_\infty)$,
so a single jump is never repeated. Track growth clears this history like any
other change of the force map.

History becomes reusable only after the whole control frame completes. A
rejected attempt, incomplete frame, timestep/phase-count change, reset, abrupt
target change, force-map/topology change or exact-path fallback invalidates it.
Smooth commands can reuse history. Unsupported or gradient-enabled paths
retain the accepted-load initialization. The optional predictor is worker-owned;
the general coupler does not infer control frames. Reports expose host counters
for prediction attempts, accepted transaction commits and invalidations without
reading device counters or introducing a new synchronization. An attempt can
still be gated by the device checks, and a commit does not itself mean the whole
frame became reusable.

For supported articulated graspers, the shaft preconditioner uses a frozen
native contact/constraint Hessian and joint mass/damping matrix, evaluated on
private MuJoCo scratch data. Condensing that matrix through the needle COM
Jacobian gives a six-dimensional effective mass for the numerical correction.
This approximate metric includes elliptic contact-cone coupling; it is not an
exact derivative of the subcycled transition. Singular or nonfinite metrics
fall back without accepting a different physical residual. The force laws,
acceptance tolerances and iteration budgets remain authoritative.

[WorkerThread](../src/newton_solver_suturebot/_src/solvers/thread/worker_thread.py) owns the
physical ANCF thread in each worker. Its default is 90 elements over 180 mm,
radius 0.15 mm, Young's modulus 2 MPa, density 1140 kg/m³, gravity 9.81 m/s²
and mass damping 2 s⁻¹. Two Newton updates per accepted interval match the
source stitch example's budget; this cap does not assert full nonlinear
convergence. The normal contact stiffness is $k=E_{\rm tissue}\bar h$, where
$\bar h$ is the mean sampled tetrahedral edge length; contact smoothing is
$0.1r$. Surface/tool friction uses `shaft_friction` (default 0.2). These are
uncalibrated constitutive and contact choices.

On supported untaped CUDA execution, the accepted thread update is captured
as a graph, including its conditional Newton loop, attachment, capsule contact,
tissue contact and lagged reaction updates. The first real accepted update
warms resources; subsequent updates replay the graph of the same state
storage at every accepted interval, which the kernels read from device data.
Python counters advance once per accepted interval. Rejected
rigid/FEM trials never advance the thread or its friction history. CPU and
gradient-enabled execution retain the eager path.

For CUDA thread systems with at most 1024 total elements, the five Gauss
points of each element are evaluated independently, then summed in the
original order. Persistent scratch storage holds the partial integrands.
Quadrature points, weights, forces, tangent terms and iteration budgets
are unchanged. CPU execution, active tapes and gradient-enabled arrays
retain the original element evaluation.

For CUDA batches of at most 128 query items, tissue contact planes and stored
friction-anchor projections use one item per GPU block. These queries write
independent node outputs with unchanged arithmetic and search order. The
original self-contact fallback also uses one item per block up to 128 total
elements; the bounded pair search above handles eligible small chains. CPU
and larger fallback batches keep their original launch width. Force-scatter
kernels and the nonlinear solve are unchanged.

The thread head attaches to the actual needle tail through `ThreadAttachment`
in lagged mode, using the needle's mass and inertia. It assumes zero support
from a needle drive: the needle has no Cartesian servo in this mode. Tool and
tissue constraints can make the true compliance smaller than the attachment's
free-body estimate. The solved thread head may consequently differ from the
needle tail while reactions converge; `head_to_needle_tail_error_m` reports
that difference. Initial loose rows are geometry initialization, not a settled
state or a certificate that every contact is initially clear.

The multi-body extension
[ThreadRigidCapsuleContact](../src/newton_solver_suturebot/_src/solvers/thread/thread_rigid_contact.py)
uses the same body IDs, local endpoints and radii that create the MuJoCo
capsules. Needle capsules and both tools' shaft, wrist and jaw capsules are
included. Each `(body, radius)` group supplies its nearest capsule to each
thread element. Different jaw bodies retain simultaneous contact planes.
Only the first two elements are excluded from needle contact at the attachment;
that exclusion does not suppress their contacts with another body.

Detection uses the element's straight chord at the accepted rigid pose. The
chosen material coordinate $\xi$, axis point and normal stay fixed during the
thread solve. At the cubic centreline point $\mathbf r(\xi)$, the normal
potential above uses separation radius $r_{\rm thread}+r_{\rm capsule}$.
Tangential contact uses the same $V_{\rm f}$ and positive spring majorizer as
node friction, with the previous accepted step's bound $f_c=\mu N_n$.
Body-local anchors move with the actual tool pose; when the nearest material
coordinate changes, the existing elastic slip is transferred over that
element. Both normal and friction contributions enter the gradient, tangent
blocks and line-search energy of the ordinary thread solve. There is no
post-step displacement correction.

Fresh forces have exactly opposite rigid reactions at the same ANCF application
points. A force $\mathbf f$ on the thread contributes
$(-\mathbf f,(\mathbf r(\xi)-\mathbf c)\times(-\mathbf f))$ to its body's
world wrench about the actual centre of mass $\mathbf c$. Applying both forces
at this centreline point preserves their resultant and world torque; the model
does not resolve a separate thread cross-section friction couple or twist.
The applied next-step wrench is relaxed:

$$
\mathbf W_{n+1}=\mathbf W_n+\alpha(\mathbf W_{\rm fresh}-\mathbf W_n),\qquad
\alpha=\min\left(\frac{K}{K+n_ck},\frac{K_r}{K_r+n_ck a^2}\right),
$$

where $K=m/h^2+c/h$, $K_r=I_{\min}/h^2+c_r/h$, $n_c$ is at least one and
otherwise the number of active candidate contacts, and $a$ bounds capsule
reach about the body's centre of mass. Default rigid-contact support damping
$c,c_r$ is zero; only a known real support can supply it. Fresh opposing
wrenches are balanced; the relaxed loads applied asynchronously to the next
rigid step are not claimed to balance instantaneously. Tissue reactions use
the existing `ThreadTissueCoupling` relaxation, with support estimate
$0.25E_{\rm tissue}\bar h$.

Each rigid/FEM trial reads stored thread loads. Only an accepted interval
advances the thread, friction anchors and reaction history, then stores the
loads for the next interval. Rejected trials leave those histories unchanged.
The contact state is included in `SolverThread.state` for explicit rollback;
rendering only samples accepted thread coordinates. The original
`ThreadCapsuleContact` API and legacy example defaults remain unchanged.

This is discrete, compliant contact, not continuous collision detection or an
exact nonpenetration constraint. It retains one contact per element and
body/radius group, can miss unresolved curvature or a fast sweep, and uses a
lagged friction bound. Self-contact remains the existing frictionless chord
approximation. The worker schedules no eyelets: with `thread_tracks` its
eyelets are the track that the needle has cut, threaded where the needle's
tail has passed (above), and without it the thread has none. The far end
stays free. The spare thread is laid on the pad's surface behind the planned
stitch and along the pad's border away from it (`loose_thread_coordinates`,
usage). The thread's contact with the pad acts at its nodes only: a thread
that hangs over an edge of the pad is not drawn back over it, the face
beside the edge holds its first node there
(`tests/test_thread_tissue.py`, an expected failure). Initial grasp/approach and contact
checks do not establish completed suturing, loop retention, wound closure,
knots, or a learned grasp policy. The implementation has no new backward-mode
gradient claim.

Contact-preconditioner input assembly and Woodbury packet construction can
reuse separate CUDA graphs after their first real eager application. These
graphs launch the original kernels in the original order and read current
geometry, residuals, matrix coefficients and ALM multipliers from device
arrays. Host validity checks, native metric construction, small dense solve,
packet readback, coefficient upload and the existing PCG solve keep their
previous order. Scalar, storage, gradient or stream changes choose the eager
path and invalidate the corresponding cached preparation; retired storage
is retained until its launch stream has completed. Reset, scene close and
explicit coupler invalidation also retire these owners. An enclosing CUDA
capture receives the original eager kernel sequence while the helper retains
its older cache; explicit retirement is rejected during capture, before any
synchronization or owner release.


The supported native physical-grasp configuration additionally checks the
normalized compact FEM solve against its current raw float32 matrix and RHS,
using float64 products and reductions. The certificate uses the configured
relative tolerance on the rounded float32 compact increment, before the usual
unscale and state update. It does not certify a separately recomputed final
physical-state linear residual. The original force, ALM, tetrahedral and
swept-event acceptance tests remain authoritative for physical stepping.

The first CG has a 120-iteration allowance. If its explicit residual does not
meet the original tolerance, one float32 correction solve has 36 iterations
and relative target 0.1. That target is only for the numerical correction: the
result must again satisfy the original-system certificate. Initialization and
certification bring the maximum primary current-matrix actions to 160; the
unchanged Chebyshev preconditioner actions are separate. Nonfinite or unmet
certificates reject the coupled attempt and preserve its original rollback.
No extra 160-iteration solve or tolerance relaxation is permitted.

`GraspScene.certified_linear_residual` enables this check only for the supported
untaped native compact-CG path. Each eligible attempt starts uncertified,
every fresh solve overwrites its status, and both existing force-statistics
readbacks reject a failed status before physical acceptance. The scope and
status storage participate in graph/workspace keys; direct, gradient or custom
solver paths retain the original behavior. Diagnostics report
`certified_linear_residual_enabled`, `certified_linear_attempts` and
`certified_linear_rejections`.


### One-trial ordered surface facets

For eligible captured CUDA force evaluations, bounded per-sample scratch keeps
one ordered traversal of the current surface AABB query. Its exact current
facet barycentrics, closest point, normal, distance, gap and area are reused by
the original minimum, normalization and force passes. Material permissions are
computed lazily at the first original use and reused only within that same
unchanged trial. The nearest signed query, full/active mesh refits, openings,
track eligibility and force/tangent laws remain current and unchanged.

Collection completes before any physical force or tangent record is written.
If any sample exceeds its capacity, a device conditional executes the entire
original surface kernel. Every fresh candidate refills the scratch; an existing
dual-only primal reuse still follows its original path. CPU, taped/gradient,
custom-kernel and unsupported block-width calls retain the original kernel.
Scratch arrays are held by the force-graph owner and included in its binding
signature. Saving FP32 facet values can affect compiler contraction; parity
checks cover branch boundaries and the physical outputs rather than assuming
universal bitwise equality from the source transformation alone.

## GPU trial pipeline

`--rigid-backend mujoco-warp` replaces the native rigid integration of the
physical grasp by MuJoCo Warp and runs all coupling trials of one interval on
the device. The equations, tolerances, budgets and acceptance checks of the
preceding sections are unchanged. This section describes only how the trials
are executed and how their search is shortened.

**One graph per interval.** The trials of an interval are one conditional CUDA
graph. A `while` node repeats the trial; `if` nodes select the tissue tolerance
level, the correction and the dual update. A trial restores the rigid
checkpoint, applies the interface load, integrates tissue and rigid bodies,
evaluates the interface forces at the candidate state and compares them with
the applied load. The host launches the graph once, reads one small result and
then runs the acceptance checks of the native path. MuJoCo Warp allocates
scratch arrays in every step, which the body of a graph conditional cannot do;
the solver's rigid step (`_src/solvers/mujoco/warp_step/`) issues the stages of
the MuJoCo Warp step itself and serves those arrays from a pool that it keeps
between steps.

**Concurrent stages.** Work captured on one stream forms a chain, so stages
that do not depend on each other would still run in sequence. The capture
replaces the dependencies of the stream instead: the tissue stage (linear
solve, then the refit of the surface meshes and tetrahedron queries) and the
rigid stage (four microsteps) both start from the applied load and join before
the interface forces. Only dependency edges change; a stage must not write
what the other reads or writes.

**Rigid warm start between trials.** Each microstep starts its constraint solve
from the solution the same microstep found in the previous trial of the
interval. The accepted rigid velocities then depend on the trial history
within the solver tolerance. With stopping scale $1/\operatorname{tr}M$ and
tolerance $\varepsilon$ the difference of a velocity component $i$ after an
interval $h$ is bounded by $2h\sqrt{2\varepsilon\operatorname{tr}(M)\,(M^{-1})_{ii}}$.

**First microstep shared between trials.** Every trial of an interval starts
from the same rigid position, velocity, activation and controls; trials differ
in the applied wrench. The stages of the first microstep that read only that
state (frames, inertia, contacts, constraint rows, bias, passive and actuator
forces) are therefore issued once per launch, before the trial loop, on a
second MuJoCo Warp `Data` that shares the integration state with the first.
Each trial issues the rest of that microstep (smooth acceleration, constraint
solve, integration) and the following microsteps whole. Overflow flags that
the shared stages raise are carried into every trial.

**Fewer launches per microstep.** A node of a replayed graph has a fixed cost
and a microstep of a small articulated model is mostly nodes. The solver's
rigid step runs the tree passes over all levels in one launch and fuses the
constraint update with the gradient, and the Cholesky step with the product
$M\,s$ and the termination test, on a solver context that it keeps between
steps. The arithmetic of each kernel is that of the MuJoCo Warp kernels.

**Contact metric and correction.** After a trial that has not converged, the
load is corrected along a preconditioned residual. The rigid part of the
preconditioner is the response of the needle to a wrench with the tool
contacts active: the $6\times6$ block of $(H+hD)^{-1}$ taken from the kept
solver context by a tile Cholesky. The Krylov solve of the contact
preconditioner starts from the response-basis (Woodbury) guess. Its small
Schur system $(I+hF^\top CF)\,y=F^\top(\dots)$, with $D=FF^\top$ the factored
anchor tangents, is assembled and solved on the device in every correction. A
tangent of rank below three leaves zero columns in $F$. Their unknowns are
zero and couple to no other, so beyond 38 track samples the system is
factorized on its nonzero columns alone whenever they fit the smaller tile. A
sliding anchor of a friction tunnel acts as $(I+S_i)D_i$, with the shear of
the shaft correction. The system of a track that follows such a tunnel is
$(I+hF^\top C(I+S)F)\,y=F^\top(\dots)$ and its contact weights are $(I+S)Fy$.
It is not symmetric, so the same Cholesky tile factorizes its normal
equations, formed with double-precision sums, of whichever system is taken:
the padded one or the one on the nonzero columns. The host path packs
$(I+S)D$ and takes the same system with its general dense solve. Every other
anchor law keeps the symmetric solve. A guess is used only if it lowers the
residual of the current operator; otherwise the solve starts from zero as
before.

The guess is one object from the first response basis on, and a captured
trial loop holds the start of this solve once. Whether a launch may use the
basis (an active basis of the current FEM structure whose support columns and
track samples fit the scratch of the guess) and which padded tile factorizes
its track samples are device flags, which the host writes where they change;
the map from compact degrees of freedom to basis columns is rewritten in
place when the support grows. A launch that may not use the basis starts from
zero inside the same graph, and the first residual of its solve is the
right-hand side itself, so one operator product is saved and an exact zero
may differ in sign from the product-based residual. A new track sample or
support column therefore captures nothing. The factorization kernels of all
six padded tiles (12, 24, 48, 114, 120 and 144 rows) are created and loaded
when the scene starts, not when a track first reaches their size.

**Surface events.** Certificates decide in every trial whether the device
force evaluation is valid: for example, that the swept path of the tip stays
clear of the moving surface triangles. A request that concerns only the path
of the tip does not end the trials. They continue to convergence and the host
oracle of the native path then decides the event, starting from the load they
reached instead of the last accepted one. If the oracle cannot use that load,
the interval starts again from the accepted load. Every other request ends
the device trials at once. By default that includes a contact force at the
puncture threshold. The accepted state is always one the oracle or a clean
final device trial has confirmed.

Two options of `NeedleTissueCoupler`, each off by default, change how such
a step passes from the device trials to the oracle. Every puncture, entry and
exit is still decided by the host evaluation and committed with the accepted
step. What was measured with them is in the
[usage guide](usage.md#surface-events-on-the-device-and-response-columns-per-step-default).
They do not bring every event step under a frame of 66.7 ms: in the grasp
example the frame of the second exit (frame 316) and the exit of the lowering
stay over it, as that section states under "Not fixed by these options".

- `search_past_puncture`: the trials continue past a contact force at the
  puncture threshold as well. Outside the tissue the trial forces are the
  same below and above the threshold: the host evaluation only notes the
  pending opening, the puncture is committed with the accepted step and
  changes the forces of the step after it. On the CPU the device kernels and
  the host evaluation give the same nodal forces, wrench, flags and
  diagnostics bit for bit on both sides of the threshold
  (`tests/test_needle_event_search.py`); on CUDA the same test passes with
  a relative tolerance of 2e-6. A trial that only overshoots the threshold
  then costs no host loop, and a step that converges at or above it is
  handed to the oracle, which decides the puncture.
- `oracle_at_device_trial`: the oracle does not repeat the trial it is
  handed. The candidate state, the rigid backend state and the tunnel
  multipliers of the last device trial stand, and the first trial of the
  host-controlled loop is the host evaluation of that state. If the oracle
  accepts it, the step is that device trial with the oracle's event. If not,
  the loop corrects the load and continues with complete host trials. If the
  oracle cannot use the trial, the interval starts again from the accepted
  load, as without the option. Handing a converged host trial to a second
  host loop on the CPU gives bit-identical accepted states. On the device
  the standing trial is the last trial of the device loop where the default
  path solves that trial a second time on the host: the same load and start
  state, not the same bits.

The mask of an opening (the surface faces that share a vertex with the
punctured face and face its way) is computed from the faces connected to the
punctured face instead of from the normals of every surface face. It is the
same mask (`tests/test_needle_event_search.py`).

**Unstable rigid trials.** A trial whose rigid coordinates, velocities or
accelerations are not finite or exceed MuJoCo's bound is rejected and the
interval is retried with a smaller step. This is the counterpart of the
native check for a MuJoCo instability warning. It does not alter a trial.
