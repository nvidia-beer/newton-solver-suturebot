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

On CUDA, a separate sparse value buffer stores $PAP+I-P$ for Warp's native
sparse multiply. Its values refresh once per solve, including captured solves;
changed sparse storage rebuilds the cached operator. The original $A$ remains
available to the predictor. The region-1 needle example adds about 3.2 MB of
matrix values while sharing index buffers. CPU and unsupported sparse layouts
retain the masked multiply.

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

Material grips are finite spring loads attached to material coordinates. They
follow the cut fragments rather than prescribing every node's position.
The [full-pad guide](phantom-cut.md) records the example's controls and support
choices. The separate needle interaction below leaves Tet4 connectivity intact;
its virtual passage is not this cutting model.

## Differentiation contract

The copied FEM core retains its fixed-matrix differentiation API. The physical
needle interaction, fixed-boundary option, simultaneous solver and thread
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
[insertion.py](../src/newton_solver_suturebot/_src/insertion.py) adds an illustrative,
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
[coupling.py](../src/newton_solver_suturebot/_src/coupling.py) solves an implicit
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
the previous relaxation. Interface
loads, residual vectors and Aitken updates stay in persistent float64 arrays on
the model device. Only six reduced interface statistics return to the host per trial for
the unchanged convergence test and relaxation update. This changes data movement
and reduction order, not the force equation or stopping tolerances.

The optional `shaft_preconditioner=True` accelerates this iteration for the
supported projected FEM/single-free-body configuration. With generalized
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

The ALM branch uses the radial derivative described above and a positive
approximation of its projected tangential update. An auxiliary block-Jacobi
CG solve uses an 80-iteration cap, with relative tolerance $10^{-2}$ for
finite-compliance ALM and $10^{-3}$ for penalty or hard-wall modes. Aitken
relaxation then operates on $d$. The full transpose
preserves reaction balance even for an approximate auxiliary solve. This is
a frozen shaft/friction approximation, not the full contact Jacobian: surface
contact and nonsymmetric derivatives are omitted. In the penalty expression
above, the tangential term equals
the frozen-normal sticking slope and continues as a positive approximation
decaying with squared slip speed in saturated sliding. It is not a quadratic
majorizer or the full friction Jacobian. The selected physical force law
is unchanged, and the original physical residual remains the acceptance test.
The looser finite-ALM auxiliary solve passed task/integrity checks but failed
a declared 2 µm fast-handling trajectory-equivalence gate (12.46 µm maximum
nodal difference); it is not an identical-history guarantee. Unsupported configurations
retain unpreconditioned Aitken; no coupled derivatives are provided.

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
unchanged by this reset. Captured step graphs are rebuilt for that duration.
After two consecutive accepted intervals needing at most four interface trials,
the example may double the interval, capped by the nominal duration and the
frame's remaining time. Rejection clears this growth counter. Every enlarged
attempt must pass the same force and swept-motion checks; each new frame starts
from the nominal duration.
Only accepted intervals advance time and material history. Differentiable
execution retains its fixed-timestep contract.

The example's end-frame integrity check is separate from these transactional
rejections. If an element has inverted, it stops further simulation with the
failed state and reason visible; the already committed frame is not rolled
back. Reset is required. A converged interface solve therefore does not by
itself guarantee valid element geometry under an arbitrarily large target.

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
residual checks use eager FEM execution. Geometry caching and batched
broad-phase queries preserve the original bounds arithmetic and ordering.
On CUDA, conservative Warp BVHs return sorted tetrahedron/facet candidates;
the original float64 NumPy containment and cubic swept-crossing tests still
decide material membership and crossings. Candidate overflow falls back to
the complete CPU query. Accepted material history remains host-owned, and
cached endpoint geometry is reused only for matching nodal positions.
Contact checks distance/side, footprint weight and force contribution before
material visibility, keeping the original facet order and positive-weight
nonpenetrating blocking facets in normalization. For verified local passages,
other blocking facets contribute only at $d<r$, so an AABB half-extent $1.5r$
retains the full contact support and blending margin. Unauthorized interior
recovery keeps the wider nearest-surface query, and active-material visibility
rays are unchanged. Surface launches use 32-thread blocks; this changes GPU
scheduling, not the contact equations. These optimizations add no
gradient, force-history convergence or material-accuracy guarantees.

`capture_coupling=True` additionally captures the repeated established-tunnel
solve in [coupling_graph.py](../src/newton_solver_suturebot/_src/coupling_graph.py).
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
allocated once and retained across timestep/track recaptures.

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

### Simultaneous MuJoCo/FEM momentum and friction

The public `SolverMuJoCoFEM` in
[solvers/mujoco_fem/solver.py](../src/newton_solver_suturebot/_src/solvers/mujoco_fem/solver.py)
uses a joint solve for eligible finite-compliance tunnel steps. It retains the
prepared affine FEM mechanics, native rigid integrator and physical contact law
above. It does not solve a fully nonlinear FEM equilibrium or replace the
material, passage or event model.

For $N$ tissue nodes and $K$ accepted track anchors, the unknowns are

$$
U=(\Delta v_1,\ldots,\Delta v_N,\Delta v_r,\,
 L R_n\Delta\omega_{\rm local},\,\zeta_1,\ldots,\zeta_K),
\qquad \zeta_i=\frac{\tau_i}{\rho h}.
$$

$L$ is the torque scaling length, $R_n$ is the accepted rigid orientation,
and $\tau_i$ is axial friction density in N/m. Each scalar $\zeta_i$ occupies
one lane of a three-component device row; the other lanes are identity
equations. Motion increments and $\zeta$ have velocity units. Fixed tissue
increments are prescribed as zero. At the reconstructed endpoint, material
point $x_i$, closest shaft point $y_i$, tangent $t_i$ and quadrature length
$\ell_i$ define

$$
d_i=(I-t_it_i^T)(y_i-x_i),\quad
n_i=\frac{(1-c/\|d_i\|)_+}{\alpha}d_i,\quad
s_i=t_i^T\!\left(v_r+\omega\times(x_i-c_r)-\sum_j w_{ij}v_j\right),
$$

$$
p_i=\operatorname{clip}(\tau_i+\rho h s_i,-\mu\|n_i\|,\mu\|n_i\|),
\qquad R_{\tau,i}=\ell_i(\tau_i-p_i).
$$

Here $c_r$ is the rigid COM, $w_{ij}$ are material barycentric weights,
and $\alpha>0$ is physical normal compliance; $n_i=0$ inside clearance.
The normal reaction is eliminated analytically, while friction density and
motion are solved together. The needle receives
$-\ell_i(n_i+p_i t_i)$, with the opposite barycentric tissue force and the
moment at $x_i$. The free tissue momentum equation is
$A_D\Delta v=b(f_{\rm ext}+f_{\rm law})$; rigid momentum uses the
`implicitfast` blocks above, including its gyroscopic angular term.
Momentum residuals are divided by $h$, and rigid torque rows by $L$.
Together with $R_\tau$, active rows therefore have force units.
Using the projected $p_i$ in momentum has the same roots as using $\tau_i$:
their momentum rows differ by the contact scatter of $R_{\tau,i}$.
Inactive or zero-length anchor rows reduce to identity equations.

[kernels.py](../src/newton_solver_suturebot/_src/solvers/mujoco_fem/kernels.py)
constructs a nonsymmetric quasi-Newton action. Outside clearance its normal
derivative is

$$
K_{n,i}=\alpha^{-1}\left[(1-c/r_i)(I-t_it_i^T)
 +(c/r_i)e_ie_i^T\right],\qquad r_i=\|d_i\|,\quad e_i=d_i/r_i.
$$

Let $\chi_i=1$ when $|\tau_i+\rho h s_i|<\mu\|n_i\|$ with positive capacity,
and zero otherwise, and let $\sigma_i$ be the sign of that projection
argument. With $b_i=\ell_i\rho h$, the search action uses

$$
\delta f_{n,i}=\ell_i h K_{n,i}J_{g,i}\delta u,\qquad
\delta f_{t,i}=\chi_i b_i(t_i^TJ_i\delta u+\delta\zeta_i)
 +(1-\chi_i)\mu\sigma_i e_i^T\delta f_{n,i}.
$$

$J_i$ gathers relative motion at the material force point; $J_{g,i}$ uses
the closest shaft point for the geometric angular lever. The force correction
scatters through $J_i^T$, and the friction row is
$b_i\delta\zeta_i-\delta f_{t,i}$.
Closest features, tangent and rotation derivatives are frozen for this search
action; actual geometry is reevaluated for every residual. Surface forces are
included in that residual, but their derivative is currently omitted.
The normal-dependent sliding bound and rigid gyroscopic block make the full
action nonsymmetric. Frozen geometry and the omitted derivative prevent a
claim of an exact full Newton Jacobian or general superlinear convergence.

[trials.py](../src/newton_solver_suturebot/_src/solvers/mujoco_fem/trials.py)
uses restarted GMRES (restart 16, maximum 128 iterations, relative search
tolerance 0.02) with an approximate block motion/constraint preconditioner.
Its local friction Schur denominator is
$S_i=(1-\chi_i+\epsilon\chi_i)b_i+\chi_i b_i^2m_i$,
where $m_i=t_i^TJ_iH_0^{-1}J_i^Tt_i$ uses the block motion inverse.
Invalid or unsupported sticking mobility fails the search.
The $\epsilon=0.01$ term also adds
$\epsilon\chi_i b_i\delta\zeta_i$ to the search friction row, limiting
underdetermined sticking corrections. It changes neither the evaluated
friction residual nor physical compliance. The free-motion initializer puts
nonquiet sliding guesses on the current Coulomb bound; its
$10^{-6}$ m/s cutoff is only a predictor deadband. Quiet history is clipped
to current capacity. Accepted multipliers are not changed by initialization.

[control.py](../src/newton_solver_suturebot/_src/solvers/mujoco_fem/control.py)
backtracks on a block-weighted squared residual norm. At the start of each
Newton line search, it freezes the scales

$$
\sigma_F=\eta\|b_0\|_2/h,\qquad
\sigma_R=\epsilon_a+\epsilon_r F_0,\qquad
\sigma_T=\epsilon_a+\epsilon_r S_0,
$$

where $F_0$ is the current maximum scaled interface-force component and $S_0$
is the ALM force-density/length scale used by its dual criterion. It then
compares the base and every backtracking proposal using the same merit

$$
\phi=\|r_F\|_2^2/\sigma_F^2+
\|r_R\|_2^2/\sigma_R^2+\|R_\tau\|_2^2/\sigma_T^2.
$$

The tissue and rigid residuals have force units, with rigid torque divided by
$L$. This weights tissue momentum progress against the configured FEM accuracy
instead of allowing larger rigid or friction blocks to dominate the search. Growing a proposal's RHS or
force scale cannot reduce the merit by changing its denominator. Zero RHS
uses $\sigma_F=\epsilon_a$ only to define a finite search objective; it does
not change the exact-zero FEM acceptance rule below. Friction L2 is a search
surrogate, while the aggregate L1 dual criterion remains authoritative.
The Armijo coefficient is $10^{-4}$, with at most 24 search evaluations and
the same shared force-evaluation budget. Passing all physical residual gates
also permits a search point to request final certification. The inexact quasi-Newton direction
is not guaranteed to descend in this merit; failure still rejects the step.
Neither search weighting, regularization nor a successful linear solve
replaces the physical or FEM accuracy gates.

The Newton proposal checks the original force, normal and aggregate friction
criteria. It also assembles the exact prepared RHS for its current loads and
records the true FEM residual, preserving the ordinary load-before-predictor
floating-point order. Relative FEM accuracy is enforced at the corrected
endpoint, rather than requiring every proposed fresh law to meet that
additional relative criterion before correction.

Final certification copies the balanced current contact law into applied load
$F_a$. It warm-starts one prepared affine FEM Krylov solve from the joint tissue
increment under $f_{\rm ext}+F_a$, using the configured FEM tolerance and budget.
This solve reconstructs tissue from the same accepted incoming state; it does
not integrate forward from the Newton proposal. The canonical device rigid
integrator advances under the same applied load, then the contact law is
reevaluated as $F_b$. No FEM solve runs inside ordinary Newton or line-search
residual evaluations.

The directly recomputed applied-load FEM certificate requires

$$
\|A_D\Delta v-b(f_{\rm ext}+F_a)\|_2^2
\leq\eta^2\|b(f_{\rm ext}+F_a)\|_2^2,
\qquad \eta=\texttt{linear\_solver\_tolerance},
$$

over free tissue rows. Both sides have squared impulse units
$(\mathrm{N\,s})^2$. A componentwise force tolerance alone cannot enforce this
relative accuracy when the RHS is small or many nodal defects accumulate.
No guessed absolute or roundoff floor relaxes this check: zero RHS requires
zero measured residual. Very tight requested tolerances can be unattainable
with the float32 sparse action, in which case the step rejects within its
budget. This true residual check can be stricter than the original CG solver's
recursive stopping estimate; a normally returning CG solve cannot override it.

The certificate also retains both free-node momentum force defects

$$
r_a=\frac{A_D\Delta v-b(f_{\rm ext}+F_a)}h,\qquad
r_b=\frac{A_D\Delta v-b(f_{\rm ext}+F_b)}h.
$$

The maximum of their component magnitudes and the original interface-load
defect must meet the force criterion, alongside the normal and aggregate
friction consistency criteria. Fresh-law relative FEM L2 accuracy is recorded
but is not a separate acceptance gate. The partitioned contract solves under
$F_a$ and permits a bounded difference $F_b-F_a$; since the impulse residuals
satisfy $h r_b=h r_a-[b(f_{\rm ext}+F_b)-b(f_{\rm ext}+F_a)]$, requiring the same
relative L2 tolerance for both loads can impose a much smaller contact-force
accuracy than the configured interface criterion. The retained true applied
L2 and two momentum force checks are additional safeguards, not a claim of
bitwise equivalence to partitioned iterations.

Subtraction, division and reductions use float64; sparse action and RHS
storage remain float32. These are directly recomputed discrete residuals,
not exact arithmetic or a forward-error bound. Residual defects are not
relabelled as physical applied forces. Fixed contact reactions remain
separate from prescribed-motion rows and total structural support reactions.
Fresh-law scatter balance and finite-tolerance applied-load world-moment
balance remain distinct diagnostics.

A failed finite certificate can continue from its corrected tissue increment,
canonical rigid increment and unchanged working friction unknown. Every new
force evaluation still consumes the shared budget. Thus there is one affine
FEM solve **per certification**, potentially several certifications per accepted
step. The device `final_corrections` counter resets per physical attempt and
records these solves, including rejected attempts; it is diagnostic only.

The host then advances native MuJoCo once under the same applied wrench and
requires the canonical joint/body endpoint to match bit for bit. Motion and
event checks precede the single history commit; rejection restores native
state, FEM numerical guess and multipliers. A failed certificate may resume
the joint solve within its remaining budget. Conservative permission failure
returns to the established event path. This specialization requires the
validated single zero-COM `implicitfast` free body, supported CUDA capture,
assembled non-differentiable FEM and positive tunnel compliance. Penalty,
hard-wall and unsupported configurations retain the established integration
path; there is no articulated-body, cable or coupled-gradient implementation
in this solver. The equations and local reference tests do not establish trajectory
equivalence, unconditional stability, physiological accuracy or a frame rate.
Primary-method inspiration and implementation boundaries are mapped in
[papers](papers.md#simultaneous-momentumfriction-solve).

The opt-in solver launches the small, branch-heavy shaft and surface query
batches with one sample per GPU block, including its captured fallback. The
default partitioned solver retains 64-thread shaft and 32-thread surface
blocks. This changes scheduling only; the query kernels, force laws and
acceptance checks are shared. Different atomic accumulation order can still
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

The friction anchor of a node that has grip is kept as a point of the surface
triangle under it and moves with the surface, so a thread that sticks rides
on the tissue.

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
lower bound of the measured stiffness of a single tissue node. The tissue
then feels a change of the thread's pull after about five steps.

In `fem_needle_suture` the stitches are planned before the run. A stitch is
an insertion plan between an entry and an exit; it is kept when that plan's
checks pass and the needle's sweep from the plan's finish angle until the
tail has left the tissue touches no pad material. Its track points are the
plan's circle sampled inside the tissue. During the run the track point at
angle $\varphi$ becomes an eyelet once the tip angle exceeds $\varphi$ plus the
needle's angular span, that is when the tail has passed it. The thread is
stepped once per accepted step of the needle and the tissue, with that step's
size, and the relaxed loads enter the next step as external forces.

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
