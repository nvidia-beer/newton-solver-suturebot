# SPDX-License-Identifier: Apache-2.0
"""Bounded correction with a raw-current-matrix FP64 residual certificate."""

from typing import Any

import warp as wp
from warp._src.optim.linear import TiledDot
from warp.optim.linear import cg

from ...utils.capture import PauseGC


@wp.kernel(enable_backward=False)
def finish_solve(
    first: wp.array[int],
    second: wp.array[int],
    used: wp.array[int],
    status: wp.array[int],
    iterations: wp.array[int],
    rr: wp.array[wp.float64],
    bb: wp.array[wp.float64],
    tolerance: float,
    residuals: wp.array2d[wp.float64],
):
    iterations[0] = first[0] + second[0]
    residuals[0, 0] = rr[0]
    residuals[1, 0] = wp.float64(tolerance) * wp.float64(tolerance) * bb[0]
    if first[0] + 2 + used[0] * (second[0] + 2) > 160:
        status[0] = 0
    iterations[1] = 1 - status[0]


@wp.kernel(enable_backward=False)
def copy_double(source: wp.array(dtype=Any), target: wp.array[wp.vec3d]):
    i = wp.tid()
    a = source[i]
    target[i] = wp.vec3d(wp.float64(a[0]), wp.float64(a[1]), wp.float64(a[2]))


# The argument type that the certified solve passes (its right-hand side), declared so
# that the module has its final hash when it is imported: see ``raw_residual`` below.
wp.overload(copy_double, {"source": wp.array(dtype=wp.vec3)})


@wp.kernel(enable_backward=False)
def raw_residual(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    b: wp.array[wp.vec3],
    x: wp.array(dtype=Any),
    residual: wp.array[wp.vec3d],
    correction_rhs: wp.array[wp.vec3],
):
    row = wp.tid()
    product = wp.vec3d(0.0)
    for slot in range(offsets[row], offsets[row + 1]):
        value = values[slot]
        v = x[columns[slot]]
        for i in range(3):
            for j in range(3):
                product[i] += wp.float64(value[i, j]) * wp.float64(v[j])
    rhs = b[row]
    r = wp.vec3d(wp.float64(rhs[0]), wp.float64(rhs[1]), wp.float64(rhs[2])) - product
    residual[row] = r
    correction_rhs[row] = wp.vec3(float(r[0]), float(r[1]), float(r[2]))


# The certified solve passes its solution vector. With both overloads declared the first
# solve adds none, so the build loaded by name at start-up is the one every solve uses.
wp.overload(raw_residual, {"x": wp.array(dtype=wp.vec3)})


@wp.kernel(enable_backward=False)
def assess(
    rr: wp.array[wp.float64],
    bb: wp.array[wp.float64],
    tolerance: float,
    need_correction: wp.array[int],
    status: wp.array[int],
):
    bound = wp.float64(tolerance) * wp.float64(tolerance) * bb[0]
    finite = wp.isfinite(rr[0]) and wp.isfinite(bb[0])
    passed = finite and rr[0] <= bound
    need_correction[0] = int(finite and not passed)
    status[0] = int(passed)


@wp.kernel(enable_backward=False)
def add_correction(x: wp.array[wp.vec3], delta: wp.array[wp.vec3]):
    i = wp.tid()
    x[i] += delta[i]


class CertifiedCorrection:
    """At most160 primary A actions including initialization/certification.

    First CG gets120 iterations. Its initialization plus an explicit FP64
    certificate cost two additional A actions. If needed, a correction gets36
    iterations plus initialization and a second certificate. The correction
    has relative target0.1 only as a numerical proposal; acceptance exclusively
    uses the unchanged requested original-system tolerance on FP32 output.

    A failed certificate remains a failure. No implicit second160-iteration
    budget, tolerance increase, or current-matrix substitution is performed.
    Preconditioner actions are unchanged and reported separately by the runner.
    """

    def __init__(self, matrix, operator, b, x, preconditioner, tolerance, *, status=None):
        if b.requires_grad or x.requires_grad or matrix.values.requires_grad:
            raise ValueError("Certified correction is forward-only")
        self.matrix, self.operator, self.b, self.x = matrix, operator, b, x
        self.device, self.tolerance = b.device, float(tolerance)
        self.r64 = wp.empty(len(b), dtype=wp.vec3d, device=self.device)
        self.b64 = wp.empty_like(self.r64)
        self.r32 = wp.empty_like(b)
        self.delta = wp.empty_like(b)
        self.bb = TiledDot(max_length=3 * len(b), scalar_type=wp.float64, device=self.device)
        self.rr = TiledDot(max_length=3 * len(b), scalar_type=wp.float64, device=self.device)
        self.condition = wp.zeros(1, dtype=int, device=self.device)
        self.status = wp.zeros(1, dtype=int, device=self.device) if status is None else status
        self.used_correction = wp.zeros(1, dtype=int, device=self.device)
        self.first = cg(
            operator,
            b,
            x,
            tol=tolerance,
            atol=0.0,
            maxiter=120,
            M=preconditioner,
            check_every=0,
            run=False,
        )
        self.second = cg(
            operator,
            self.r32,
            self.delta,
            tol=0.1,
            atol=0.0,
            maxiter=36,
            M=preconditioner,
            check_every=0,
            run=False,
        )
        # Keep the FEM/affine workspace protocol. These diagnostics describe
        # total work and the actual FP64 residual, not the first recursive one.
        self._x, self._b, self._maxiter = x, b, 160
        self._cur_iter_and_condition = wp.zeros(2, dtype=int, device=self.device)
        self._residuals = wp.zeros((2, 1), dtype=wp.float64, device=self.device)
        self._tiled_dot = self.rr
        self._preconditioner = preconditioner

    def certificate(self):
        m = self.matrix
        wp.launch(
            raw_residual,
            len(self.b),
            [m.offsets, m.columns, m.values, self.b, self.x, self.r64, self.r32],
            device=self.device,
        )
        self.rr.compute(self.r64, self.r64)
        wp.launch(
            assess,
            1,
            [
                self.rr.col(0),
                self.bb.col(0),
                self.tolerance,
                self.condition,
                self.status,
            ],
            device=self.device,
        )

    def __call__(self, *, b=None, M=None):
        if b is not None and b is not self.b:
            raise ValueError("Certified CG workspace requires its bound RHS storage")
        if M is not None and M is not self._preconditioner:
            raise ValueError("Certified CG workspace requires its bound preconditioner")
        wp.launch(copy_double, len(self.b), [self.b, self.b64], device=self.device)
        self.bb.compute(self.b64, self.b64)
        self.used_correction.zero_()
        self.second._cur_iter_and_condition.zero_()
        with PauseGC():
            self.first()
        self.certificate()

        def correct():
            self.used_correction.fill_(1)
            self.delta.zero_()
            with PauseGC():
                self.second()
            wp.launch(add_correction, len(self.x), [self.x, self.delta], device=self.device)
            self.certificate()

        if self.device.is_capturing:
            wp.capture_if(self.condition, correct)
        elif int(self.condition.numpy()[0]):
            correct()
        wp.launch(
            finish_solve,
            1,
            [
                self.first._cur_iter_and_condition,
                self.second._cur_iter_and_condition,
                self.used_correction,
                self.status,
                self._cur_iter_and_condition,
                self.rr.col(0),
                self.bb.col(0),
                self.tolerance,
                self._residuals,
            ],
            device=self.device,
        )
        return self._cur_iter_and_condition[0:1], self.rr.col(0), self._residuals[1]

    def diagnostics(self):
        first = int(self.first._cur_iter_and_condition.numpy()[0])
        second = int(self.second._cur_iter_and_condition.numpy()[0])
        used = int(self.used_correction.numpy()[0])
        return {
            "certified": bool(self.status.numpy()[0]),
            "first_iterations": first,
            "correction_iterations": second,
            "correction_used": bool(used),
            "primary_A_actions": first + 2 + (second + 2 if used else 0),
            "true_residual_norm": float(self.rr.col(0).numpy()[0]) ** 0.5,
            "rhs_norm": float(self.bb.col(0).numpy()[0]) ** 0.5,
        }
