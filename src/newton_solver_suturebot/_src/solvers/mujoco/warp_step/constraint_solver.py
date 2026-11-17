# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Newton constraint solve of a MuJoCo Warp step on a kept context, with fused launches.

``NewtonSolve.solve`` leaves in ``Data`` what ``mujoco_warp._src.solver.solve`` leaves
there from the same inputs. Every kernel launch and every memory operation is a node
with a fixed cost in a replayed CUDA graph, and the solve of a small articulated model
is mostly such nodes, so this solve issues fewer of them:

- The solver context is created once. MuJoCo Warp creates one per solve, which is a set
  of allocation, fill and free nodes. A solve reads no context entry that it has not
  written, so the kept context is not zeroed and carries nothing from one solve to the next.
- Launches that always follow one another are one launch. A solve starts with eight
  launches and takes five per Newton iteration: the linesearch, the constraint rows
  with ``qfrc_constraint`` and the gradient, the two terms of the Hessian, and the
  Cholesky solve with ``mv = M @ search`` and the termination test.

Kernels that MuJoCo Warp has are launched from there with its own arguments. The kernels
defined here repeat, operation for operation, the arithmetic of the MuJoCo Warp 3.17.0
kernels they fuse (``mujoco_warp/_src/solver.py``, Apache-2.0). On the CPU the results
equal MuJoCo Warp's bit for bit. On CUDA two sums take their terms in an unspecified
order, in this solve as in MuJoCo Warp's: the cone curvature that several contacts add
to one entry of the Hessian, and the squared gradient norm of a world.

A fused kernel gives every world one CUDA block and orders its stages with a block
barrier (``block``).
"""

import functools
from math import ceil

import warp as wp
from mujoco_warp._src import support, types
from mujoco_warp._src.block_cholesky import solve_search_sums
from mujoco_warp._src.solver import (
    _BLOCK_CHOLESKY_DIM,
    _create_solver_context,
    _eval_constraint,
    _linesearch_iterative,
    _solve_init_dof,
    _solve_init_jaref_kernel,
    _update_gradient_JTCJ_dense,
    _update_gradient_JTDAJ_dense_tiled,
)

from .block import BLOCK_DIM_MAX, block_barrier, block_dim


@wp.kernel(module="unique", enable_backward=False)
def _solve_init_scale_efc(
    # Model:
    nv: int,
    M_rownnz: wp.array[int],
    M_rowadr: wp.array[int],
    # Data in:
    M_in: wp.array2d[float],
    nworld_in: int,
    # Data out:
    solver_niter_out: wp.array[int],
    # Out:
    inertia_scale_out: wp.array[float],
    ctx_search_dot_out: wp.array[float],
    ctx_done_out: wp.array[bool],
    nsolving_out: wp.array[int],
):
    """Start of a solve for one world: _inertia_scale, _solve_init_efc and every world unconverged."""
    worldid = wp.tid()

    # inverse trace of the inertia matrix, which scales the termination criteria
    trace = float(0.0)
    for i in range(nv):
        trace += M_in[worldid, M_rowadr[i] + M_rownnz[i] - 1]
    inertia_scale_out[worldid] = 1.0 / wp.max(types.MJ_MINVAL, trace)

    solver_niter_out[worldid] = 0
    ctx_done_out[worldid] = False
    ctx_search_dot_out[worldid] = 0.0
    if worldid == 0:
        nsolving_out[0] = nworld_in


@wp.func
def _update_constraint_row(
    # Model:
    opt_impratio_invsqrt: wp.array[float],
    # Data in:
    ne_in: wp.array[int],
    nf_in: wp.array[int],
    nefc_in: wp.array[int],
    contact_friction_in: wp.array[types.vec5],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_D_in: wp.array2d[float],
    efc_frictionloss_in: wp.array2d[float],
    nacon_in: wp.array[int],
    # In:
    ctx_Jaref_in: wp.array2d[float],
    ctx_done_in: wp.array[bool],
    worldid: int,
    efcid: int,
    # Data out:
    efc_force_out: wp.array2d[float],
    efc_state_out: wp.array2d[int],
):
    """Force and state of one constraint row: the _update_constraint_efc kernel that tracks no changes."""
    if ctx_done_in[worldid]:
        return

    if efcid >= nefc_in[worldid]:
        return

    efc_D = efc_D_in[worldid, efcid]
    Jaref = ctx_Jaref_in[worldid, efcid]

    ne = ne_in[worldid]
    nf = nf_in[worldid]

    is_equality = efcid < ne
    is_friction = (not is_equality) and (efcid < ne + nf)
    is_elliptic = efc_type_in[worldid, efcid] == types.ConstraintType.CONTACT_ELLIPTIC

    frictionloss = efc_frictionloss_in[worldid, efcid] if is_friction else 0.0

    efcid0 = -1
    jaref0 = float(0.0)
    D0 = float(0.0)
    mu = float(0.0)
    ufrictionj = float(0.0)
    TT = float(0.0)

    if is_elliptic:
        conid = efc_id_in[worldid, efcid]
        if conid >= nacon_in[0]:
            return
        efcid0 = contact_efc_address_in[conid, 0]
        if efcid0 < 0:
            return

        dim = contact_dim_in[conid]
        friction = contact_friction_in[conid]
        mu = friction[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]
        jaref0 = ctx_Jaref_in[worldid, efcid0]
        D0 = efc_D_in[worldid, efcid0]

        for j in range(1, dim):
            efcidj = contact_efc_address_in[conid, j]
            if efcidj < 0:
                return
            frictionj = friction[j - 1]
            uj = ctx_Jaref_in[worldid, efcidj] * frictionj
            TT += uj * uj
            if efcid == efcidj:
                ufrictionj = uj * frictionj

    res = _eval_constraint(
        is_equality,
        is_friction,
        is_elliptic,
        Jaref,
        efc_D,
        frictionloss,
        efcid,
        efcid0,
        jaref0,
        D0,
        mu,
        ufrictionj,
        TT,
    )

    new_state = int(res[1])
    efc_force_out[worldid, efcid] = res[0]
    efc_state_out[worldid, efcid] = new_state


@wp.kernel(module="unique", enable_backward=False)
def _update_constraint_gradient_fused(
    # Model:
    nv: int,
    opt_impratio_invsqrt: wp.array[float],
    # Data in:
    ne_in: wp.array[int],
    nf_in: wp.array[int],
    nefc_in: wp.array[int],
    qfrc_smooth_in: wp.array2d[float],
    contact_friction_in: wp.array[types.vec5],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_frictionloss_in: wp.array2d[float],
    efc_Ma_in: wp.array2d[float],
    njmax_in: int,
    nacon_in: wp.array[int],
    # In:
    ctx_Jaref_in: wp.array2d[float],
    ctx_done_in: wp.array[bool],
    # Data out:
    qfrc_constraint_out: wp.array2d[float],
    efc_force_out: wp.array2d[float],
    efc_state_out: wp.array2d[int],
    # Out:
    ctx_grad_out: wp.array2d[float],
    ctx_grad_dot_out: wp.array[float],
    ctx_newton_decrement_out: wp.array[float],
    ctx_grad_scale_out: wp.array[float],
    ctx_search_unchanged_out: wp.array[bool],
):
    """Constraint rows, dense qfrc_constraint and gradient of one world in one block.

    Runs _update_constraint_efc, _update_constraint_init_qfrc_constraint_dense,
    _update_gradient_zero_grad_dot and _update_gradient_grad of MuJoCo Warp, in the forms
    that track no constraint state changes, with their arithmetic. The block barrier stands
    for the launch boundaries after which the degrees of freedom read the row forces and
    add to the zeroed grad_dot.
    """
    worldid, tid = wp.tid()

    if tid == 0:
        ctx_search_unchanged_out[worldid] = False

    if ctx_done_in[worldid]:
        return

    if tid == 0:
        ctx_grad_dot_out[worldid] = 0.0
        ctx_newton_decrement_out[worldid] = 0.0
        ctx_grad_scale_out[worldid] = 1.0

    nefc = min(njmax_in, nefc_in[worldid])
    for efcid in range(tid, nefc, wp.block_dim()):
        _update_constraint_row(
            opt_impratio_invsqrt,
            ne_in,
            nf_in,
            nefc_in,
            contact_friction_in,
            contact_dim_in,
            contact_efc_address_in,
            efc_type_in,
            efc_id_in,
            efc_D_in,
            efc_frictionloss_in,
            nacon_in,
            ctx_Jaref_in,
            ctx_done_in,
            worldid,
            efcid,
            efc_force_out,
            efc_state_out,
        )
    block_barrier()

    for dofid in range(tid, nv, wp.block_dim()):
        sum_qfrc = float(0.0)
        for efcid in range(nefc):
            efc_J = efc_J_in[worldid, efcid, dofid]
            force = efc_force_out[worldid, efcid]
            sum_qfrc += efc_J * force

        qfrc_constraint_out[worldid, dofid] = sum_qfrc

        grad = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - sum_qfrc
        ctx_grad_out[worldid, dofid] = grad
        wp.atomic_add(ctx_grad_dot_out, worldid, grad * grad)


@functools.cache
def _update_gradient_cholesky_fused(tile_size: int, terminate: bool, warn_overflow: int):
    """Kernel for the Newton search of one world, its product with M and the termination test.

    Runs _update_gradient_cholesky, support.mul_m_kernel on the new search and, if
    ``terminate``, _solve_done of MuJoCo Warp with their arithmetic.

    Args:
        tile_size: Number of degrees of freedom.
        terminate: Run the termination test of an iteration.
        warn_overflow: Overflow warning bitmask.
    """
    TERMINATE = terminate

    @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
    def kernel(
        # Model:
        opt_tolerance: wp.array[float],
        opt_iterations: int,
        M_mulm_rowadr: wp.array[int],
        M_mulm_col: wp.array[int],
        M_mulm_madr: wp.array[int],
        # Data in:
        M_in: wp.array2d[float],
        # In:
        ctx_grad_in: wp.array2d[float],
        h_in: wp.array3d[float],
        ctx_inertia_scale_in: wp.array[float],
        ctx_grad_dot_in: wp.array[float],
        ctx_improvement_in: wp.array[float],
        ctx_alpha_in: wp.array[float],
        ctx_done_in: wp.array[bool],
        # Data out:
        solver_niter_out: wp.array[int],
        overflow_out: wp.array[int],
        # Out:
        ctx_search_out: wp.array2d[float],
        ctx_search_dot_out: wp.array[float],
        ctx_newton_decrement_out: wp.array[float],
        ctx_mv_out: wp.array2d[float],
        nsolving_out: wp.array[int],
        ctx_done_out: wp.array[bool],
    ):
        worldid, tid = wp.tid()
        TILE_SIZE = wp.static(tile_size)

        if ctx_done_in[worldid]:
            return

        mat_tile = wp.tile_load(h_in[worldid], shape=(TILE_SIZE, TILE_SIZE))
        wp.tile_cholesky_inplace(mat_tile, fill_mode="upper")
        input_tile = wp.tile_load(ctx_grad_in[worldid], shape=TILE_SIZE)
        output_tile = wp.tile_cholesky_solve(mat_tile, input_tile, fill_mode="upper")
        sums = wp.tile_reduce(wp.add, wp.tile_map(solve_search_sums, input_tile, output_tile))[0]
        ctx_search_dot_out[worldid] = sums[0]
        ctx_newton_decrement_out[worldid] = sums[1]
        wp.tile_store(ctx_search_out[worldid], wp.tile_map(wp.mul, output_tile, -1.0))

        # the block has stored the whole search; mv = M @ search is the first step of the next linesearch
        block_barrier()
        for dofid in range(tid, TILE_SIZE, wp.block_dim()):
            acc = float(0.0)
            start = M_mulm_rowadr[dofid]
            end = M_mulm_rowadr[dofid + 1]
            for k in range(start, end):
                col = M_mulm_col[k]
                madr = M_mulm_madr[k]
                acc += M_in[worldid, madr] * ctx_search_out[worldid, col]

            ctx_mv_out[worldid, dofid] = acc

        if wp.static(TERMINATE):
            # one thread of the block decides termination, from the decrement it has just stored
            if tid == 0:
                solver_niter_out[worldid] += 1
                tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
                scale = ctx_inertia_scale_in[worldid]

                alpha = ctx_alpha_in[worldid]
                improvement = scale * ctx_improvement_in[worldid]
                gradient = scale * wp.sqrt(ctx_grad_dot_in[worldid])
                model_improvement = scale * 0.5 * sums[1]
                done = (
                    (alpha == 0.0)
                    or (improvement > 0.0 and improvement < tolerance)
                    or (gradient < tolerance)
                    or (model_improvement < tolerance)
                )
                if done or solver_niter_out[worldid] == opt_iterations:
                    # converged or out of iterations: the world leaves the count of unconverged worlds
                    if not done and solver_niter_out[worldid] == opt_iterations:
                        if wp.static(bool(warn_overflow & types.OverflowType.ITERATIONS)):
                            wp.printf(
                                "solver iterations limit reached - please increase iterations beyond %u\n"
                                "To disable the print warning: "
                                "m.opt.warn_overflow &= ~mjw.OverflowType.ITERATIONS (or = 0 for all)\n",
                                opt_iterations,
                            )
                        overflow_out[worldid] = overflow_out[worldid] | types.OverflowType.ITERATIONS
                    ctx_done_out[worldid] = True
                    wp.atomic_add(nsolving_out, 0, -1)

    return kernel


def _check_supported(m, d):
    """Raise ``ValueError`` unless the fused launches of this module solve the model."""
    opt = m.opt
    required = (
        (opt.solver == types.SolverType.NEWTON, "the Newton solver"),
        (opt.cone == types.ConeType.ELLIPTIC, "elliptic friction cones"),
        (opt.integrator != types.IntegratorType.DISCRETE, "an integrator other than DISCRETE"),
        (not opt.enableflags & types.EnableBit.SLEEP, "sleeping disabled"),
        (opt.noslip_iterations == 0, "no noslip iterations"),
        (not opt.deterministic & types.DeterminismType.ATOMICS, "atomics that are not deterministic"),
        (not m.is_sparse, "a dense constraint Jacobian"),
        (d.M.ndim == 2, "an inertia matrix that is not compacted"),
        (m.nv <= _BLOCK_CHOLESKY_DIM, f"at most {_BLOCK_CHOLESKY_DIM} degrees of freedom"),
        (max(d.njmax, m.nv) <= BLOCK_DIM_MAX, f"at most {BLOCK_DIM_MAX} constraint rows per world"),
    )
    missing = [name for met, name in required if not met]
    if missing:
        raise ValueError("NewtonSolve requires " + ", ".join(missing))


class NewtonSolve:
    """Constraint solve of one MuJoCo Warp ``Model`` and ``Data`` on a context that it keeps.

    Supported, and checked when the object is created: the Newton solver with elliptic
    friction cones, a dense constraint Jacobian, at most 32 degrees of freedom and 256
    constraint rows per world, no deterministic atomics, sleeping, noslip iterations or
    DISCRETE integrator. The model options that this check reads must stay as they are.

    Create it outside any graph capture. A captured graph addresses the context by pointer:
    keep the object alive while such a graph is launched. The context and every launch are
    on the device of ``data``, whatever the default device of the caller is.

    Args:
        model: MuJoCo Warp ``Model``.
        data: MuJoCo Warp ``Data`` of that model; ``solve`` reads and writes its arrays.

    Attributes:
        context: The ``SolverContext`` of every solve. After a solve ``context.h`` holds the
            Hessian of every world at the qacc that the solve ended at, and ``context.mv`` the
            product of the inertia matrix with ``context.search``.

    Raises:
        ValueError: If the model or data is outside the supported configuration.
        RuntimeError: If the device of ``data`` is capturing a graph.
    """

    def __init__(self, model, data):
        _check_supported(model, data)
        self.model, self.data = model, data
        self.device = data.qacc.device
        if self.device.is_capturing:
            raise RuntimeError("Create NewtonSolve outside graph capture")
        with wp.ScopedDevice(self.device):
            self.context = _create_solver_context(model, data)
        # number of unconverged worlds, the condition of the iteration loop
        self._nsolving = wp.empty(1, dtype=int, device=self.device)

    def solve(self, *, after_loop=None):
        """Write the arrays of ``data`` that ``mujoco_warp._src.solver.solve`` writes, with its results.

        These are qacc, qfrc_constraint, efc.force, efc.state, efc.Ma, solver_niter and overflow.
        A model without constraint rows or degrees of freedom has nothing to solve: qacc is
        qacc_smooth, solver_niter is zero and the context is not touched.

        Args:
            after_loop: Called once per solve, right after the iteration loop has been issued:
                after ``wp.capture_while`` where the model uses a graph conditional, else after
                the last iteration. An owner that arranges the dependencies of a CUDA graph
                capture itself restores them here, because Warp resumes the capture after a
                conditional node from every leaf of the graph.
        """
        m, d = self.model, self.data
        with wp.ScopedDevice(self.device):
            if d.njmax == 0 or m.nv == 0:
                wp.copy(d.qacc, d.qacc_smooth)
                d.solver_niter.fill_(0)
            else:
                self._start()
                if m.opt.iterations != 0 and m.opt.graph_conditional:
                    # The body is launched for as long as a world is unconverged. A world that reaches
                    # opt.iterations counts as converged, which ends the loop.
                    wp.capture_while(self._nsolving, while_body=self._iteration)
                else:
                    for _ in range(m.opt.iterations):
                        self._iteration()
            if after_loop is not None:
                after_loop()

    def _start(self):
        """Constraint state, gradient and first search direction at the initial qacc."""
        m, d, ctx = self.model, self.data, self.context
        warmstart = not (m.opt.disableflags & types.DisableBit.WARMSTART)
        wp.launch(
            _solve_init_dof(warmstart, False),  # dense constraint Jacobian
            dim=(d.nworld, m.nv),
            inputs=[d.nefc, d.qacc_warmstart, d.qacc_smooth],
            outputs=[d.qacc, d.qfrc_constraint],
        )
        wp.launch(
            _solve_init_scale_efc,
            dim=d.nworld,
            inputs=[m.nv, m.M_rownnz, m.M_rowadr, d.M, d.nworld],
            outputs=[d.solver_niter, ctx.inertia_scale, ctx.search_dot, ctx.done, self._nsolving],
        )
        # Jaref = efc_J @ qacc - efc_aref. The kernel is the one for a dense Jacobian, 50 degrees of
        # freedom per thread, full coordinates and atomics that are not deterministic. A row of at
        # most 32 degrees of freedom is one thread, which writes the entry instead of adding to it.
        wp.launch(
            _solve_init_jaref_kernel(False, m.nv, 50, False, False),
            dim=(d.nworld, d.njmax, 1),
            inputs=[d.nefc, d.qacc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J, d.efc.aref, d.dof_cdof],
            outputs=[ctx.Jaref],
        )
        # Ma = M @ qacc
        support.mul_m(m, d, d.efc.Ma, d.qacc, skip=ctx.done)
        self._update(terminate=False)

    def _iteration(self):
        """One Newton iteration of every unconverged world."""
        # the linesearch kernel computes jv = J @ search itself; mv comes from the kernel that formed the search
        _linesearch_iterative(self.model, self.data, self.context, fuse_jv=True)
        self._update(terminate=True)

    def _update(self, *, terminate):
        """Constraint rows, gradient, Hessian and search direction at the current qacc.

        Args:
            terminate: Run the termination test of an iteration after the search.
        """
        m, d, ctx = self.model, self.data, self.context
        wp.launch_tiled(
            _update_constraint_gradient_fused,
            dim=d.nworld,
            inputs=[
                m.nv,
                m.opt.impratio_invsqrt,
                d.ne,
                d.nf,
                d.nefc,
                d.qfrc_smooth,
                d.contact.friction,
                d.contact.dim,
                d.contact.efc_address,
                d.efc.type,
                d.efc.id,
                d.efc.J,
                d.efc.D,
                d.efc.frictionloss,
                d.efc.Ma,
                d.njmax,
                d.nacon,
                ctx.Jaref,
                ctx.done,
            ],
            outputs=[
                d.qfrc_constraint,
                d.efc.force,
                d.efc.state,
                ctx.grad,
                ctx.grad_dot,
                ctx.newton_decrement,
                ctx.grad_scale,
                ctx.search_unchanged,
            ],
            block_dim=block_dim(max(d.njmax, m.nv)),
        )
        # h = M + (efc_J.T * efc_D * active) @ efc_J
        wp.launch_tiled(
            _update_gradient_JTDAJ_dense_tiled(m.nv_pad, types.TILE_SIZE_JTDAJ_DENSE, d.njmax, m.M_colind.shape[0]),
            dim=d.nworld,
            inputs=[m.M_colind, m.M_hinit_i, d.nefc, d.M, d.efc.J, d.efc.D, d.efc.state, ctx.done],
            outputs=[ctx.h],
            block_dim=m.block_dim.update_gradient_JTDAJ_dense,
        )
        # h += curvature of the contacts in the cone state. On CUDA the launch takes a number of thread
        # rows in proportion to the multiprocessors of the device and each row strides over the contacts.
        if self.device.is_cuda:
            dim_block = ceil((self.device.sm_count * 6 * 256) / m.dof_tri_row.size)
            nblocks_perblock = int((d.naconmax + dim_block - 1) / dim_block)
        else:
            dim_block = 1
            nblocks_perblock = d.naconmax
        wp.launch(
            # atomics that are not deterministic; only deterministic ones read the first argument
            _update_gradient_JTCJ_dense(0, False),
            dim=(dim_block, m.dof_tri_row.size),
            inputs=[
                m.opt.impratio_invsqrt,
                m.dof_tri_row,
                m.dof_tri_col,
                d.contact.dist,
                d.contact.includemargin,
                d.contact.friction,
                d.contact.dim,
                d.contact.efc_address,
                d.contact.worldid,
                d.efc.J,
                d.efc.D,
                d.efc.state,
                d.naconmax,
                d.nacon,
                ctx.Jaref,
                ctx.done,
                nblocks_perblock,
                dim_block,
            ],
            outputs=[ctx.h],
        )
        wp.launch_tiled(
            _update_gradient_cholesky_fused(m.nv, terminate, int(m.opt.warn_overflow)),
            dim=d.nworld,
            inputs=[
                m.opt.tolerance,
                m.opt.iterations,
                m.M_mulm_rowadr,
                m.M_mulm_col,
                m.M_mulm_madr,
                d.M,
                ctx.grad,
                ctx.h,
                ctx.inertia_scale,
                ctx.grad_dot,
                ctx.improvement,
                ctx.alpha,
                ctx.done,
            ],
            outputs=[
                d.solver_niter,
                d.overflow,
                ctx.search,
                ctx.search_dot,
                ctx.newton_decrement,
                ctx.mv,
                self._nsolving,
                ctx.done,
            ],
            block_dim=m.block_dim.update_gradient_cholesky,
        )
