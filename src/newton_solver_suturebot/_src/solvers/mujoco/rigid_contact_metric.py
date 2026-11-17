"""GPU condensation of a frozen articulated contact search metric.

The metric is (B (H + dt*diag(damping))^-1 B.T)^-1, where H is the
MuJoCo Warp Newton solver's last contact Hessian and B maps joint rates to
(needle COM velocity, moment_length * angular velocity). This is a search
preconditioner, not an exact derivative of the subcycled physical transition.
Fresh force, ALM, volume and swept-contact acceptance checks remain mandatory.
"""

import functools

import warp as wp
from mujoco_warp._src.support import jac_dof

from ...utils.step_time import StepTime


@wp.kernel(enable_backward=False)
def _body_jacobian(
    body_parent: wp.array[int],
    body_root: wp.array[int],
    dof_body: wp.array[int],
    ancestor: wp.array2d[int],
    subtree_com: wp.array2d[wp.vec3],
    cdof: wp.array2d[wp.spatial_vector],
    body_com: wp.array2d[wp.vec3],
    body: int,
    length: float,
    jacobian: wp.array2d[wp.float64],
):
    dof = wp.tid()
    linear, angular = jac_dof(
        body_parent,
        body_root,
        dof_body,
        ancestor,
        subtree_com,
        cdof,
        body_com[0, body],
        body,
        dof,
        0,
    )
    for component in range(3):
        jacobian[component, dof] = wp.float64(linear[component])
        jacobian[component + 3, dof] = wp.float64(length) * wp.float64(angular[component])


@wp.kernel(enable_backward=False)
def _prepare_factor(
    hessian: wp.array3d[float],
    damping: wp.array2d[float],
    dt: wp.array[float],
    factor: wp.array2d[wp.float64],
):
    i, j = wp.tid()
    # MuJoCo only promises the upper triangle of its Newton Hessian.
    value = wp.float64(hessian[0, wp.min(i, j), wp.max(i, j)])
    if i == j:
        # dt holds the interval [s] in its one element (StepTime.seconds).
        value += wp.float64(dt[0]) * wp.float64(damping[0, i])
    factor[i, j] = value


_TILE_BLOCK_DIM = 64


@functools.cache
def _condense_tiles(n: int):
    """Cooperative FP64 condensation for an ``n`` x ``n`` Hessian.

    One block of threads factorizes H, solves the six Jacobian columns, forms
    the 6 x 6 response and inverts it. Tile shapes are compile-time constants,
    so each rigid model size has its own kernel. A matrix that is not positive
    definite leaves non-finite pivots, which ``_finish`` rejects.
    """

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        jacobian: wp.array2d[wp.float64],
        factor: wp.array2d[wp.float64],
        small_factor: wp.array2d[wp.float64],
        inverse: wp.array2d[wp.float64],
    ):
        lower = wp.tile_cholesky(wp.tile_load(factor, shape=(n, n)))
        rates = wp.tile_load(jacobian, shape=(6, n))
        solution = wp.tile_cholesky_solve(lower, wp.tile_transpose(rates))
        response = wp.tile_matmul(rates, solution)
        response = (response + wp.tile_transpose(response)) * wp.float64(0.5)
        small = wp.tile_cholesky(response)
        identity = wp.tile_diag_add(
            wp.tile_zeros(shape=(6, 6), dtype=wp.float64), wp.tile_ones(shape=6, dtype=wp.float64)
        )
        wp.tile_store(inverse, wp.tile_cholesky_solve(small, identity))
        wp.tile_store(factor, lower)
        wp.tile_store(small_factor, small)

    return kernel


@wp.kernel(enable_backward=False)
def _finish(
    factor: wp.array2d[wp.float64],
    inverse: wp.array2d[wp.float64],
    response: wp.array2d[wp.float64],
    small_factor: wp.array2d[wp.float64],
    blocks: wp.array[wp.mat33],
    valid: wp.array[int],
):
    valid[0] = 0
    for i in range(factor.shape[0]):
        pivot = factor[i, i]
        if not wp.isfinite(pivot) or pivot <= wp.float64(0.0):
            return
    for i in range(6):
        pivot = small_factor[i, i]
        if not wp.isfinite(pivot) or pivot <= wp.float64(0.0):
            return
    # Validate the symmetrized FP32 matrix actually consumed by CG. A positive
    # double-precision metric can become indefinite when rounded to float32.
    for i in range(6):
        for j in range(i + 1):
            rounded = float(wp.float64(0.5) * (inverse[i, j] + inverse[j, i]))
            value = wp.float64(rounded)
            response[i, j] = value
            response[j, i] = value
            for k in range(j):
                value -= small_factor[i, k] * small_factor[j, k]
            if not wp.isfinite(value):
                return
            if i == j:
                if value <= wp.float64(0.0):
                    return
                small_factor[i, j] = wp.sqrt(value)
            else:
                small_factor[i, j] = value / small_factor[j, j]
    for block in range(4):
        row = int(0)
        column = int(0)
        if block == 1 or block == 3:
            row = 3
        if block == 1 or block == 2:
            column = 3
        matrix = wp.mat33(0.0)
        for i in range(3):
            for j in range(3):
                matrix[i, j] = float(response[row + i, column + j])
        blocks[block] = matrix
    valid[0] = 1


def condense(jacobian, factor, response, small_factor, inverse, blocks, valid):
    """Replace ``factor`` (H + dt D) by its Cholesky factor and write the FP32 metric blocks."""
    device = factor.device
    wp.launch_tiled(
        _condense_tiles(factor.shape[0]),
        dim=[1],
        inputs=[jacobian, factor, small_factor, inverse],
        block_dim=_TILE_BLOCK_DIM,
        device=device,
    )
    wp.launch(_finish, 1, [factor, inverse, response, small_factor, blocks, valid], device=device)


class CapturedRigidContactMetric:
    """Single-scene, primal-only search metric with entirely device-side updates.

    ``rigid_step`` is the ``RigidStep`` of the rigid solver, which its first eager
    interval creates. The metric reads the Hessian that the constraint solve of
    that step ended with (``rigid_step.constraint_solve.context.h``), so prepare it
    after a step. ``valid`` is a device flag; callers must use the unpreconditioned
    residual when it is zero. No physical state or contact history is modified.

    ``step_time`` is the ``StepTime`` of the step whose interval the metric is condensed
    for; the factor kernel reads the interval there. Without one the metric owns its own.
    """

    def __init__(self, rigid_step, moment_length, rigid_blocks, *, body=1, step_time=None):
        self.model, self.data = rigid_step.model, rigid_step.data
        if self.data.nworld != 1 or self.model.nv < 6:
            raise ValueError("Needle metric requires one articulated GPU scene")
        if not moment_length > 0 or not 0 < body < self.model.nbody:
            raise ValueError("Invalid needle body or moment scale")
        if any(array.requires_grad for array in (self.data.qpos, self.data.qvel, rigid_blocks)):
            raise ValueError("Contact metric is a primal search direction only")
        self.rigid_step, self.context = rigid_step, rigid_step.constraint_solve.context
        self.body, self.length, self.blocks = body, float(moment_length), rigid_blocks
        self.device = self.data.qpos.device
        self.step_time = StepTime(self.device) if step_time is None else step_time
        n = self.model.nv
        self.jacobian = wp.zeros((6, n), dtype=wp.float64, device=self.device)
        self.factor = wp.zeros((n, n), dtype=wp.float64, device=self.device)
        self.response = wp.zeros((6, 6), dtype=wp.float64, device=self.device)
        self.small_factor = wp.zeros_like(self.response)
        self.inverse = wp.zeros_like(self.response)
        self.valid = wp.zeros(1, dtype=int, device=self.device)

    def prepare(self, dt):
        m, d = self.model, self.data
        self.rigid_step.validate(m, d)
        # Written by the host before a capture; inside one this is a comparison.
        self.step_time.set(dt)
        wp.launch(
            _body_jacobian,
            m.nv,
            [
                m.body_parentid,
                m.body_rootid,
                m.dof_bodyid,
                m.body_isdofancestor,
                d.subtree_com,
                d.cdof,
                d.xipos,
                self.body,
                self.length,
                self.jacobian,
            ],
            device=self.device,
        )
        wp.launch(
            _prepare_factor,
            (m.nv, m.nv),
            [self.context.h, m.dof_damping, self.step_time.seconds, self.factor],
            device=self.device,
        )
        condense(self.jacobian, self.factor, self.response, self.small_factor, self.inverse, self.blocks, self.valid)
