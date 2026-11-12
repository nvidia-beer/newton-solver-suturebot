# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Chebyshev acceleration of the 3x3 block-Jacobi inverse for use inside CG.

One application performs a fixed-degree Chebyshev iteration for ``A z = r``
with the block-Jacobi splitting and zero initial guess, so it equals
``z = p(M⁻¹A) M⁻¹ r`` for a fixed polynomial ``p`` that is positive on the
estimated spectrum interval. With fixed coefficients the operator is symmetric
positive definite in the ``A``-compatible inner product, so it is a valid CG
preconditioner: the matrix, right-hand side and stopping rule are unchanged
and only the preconditioner quality differs from plain block Jacobi.

The interval is ``[λmax/ratio, λmax]``. For ``B = M⁻¹A``, both
``max_i Σ_j ||B_ij||_F`` and ``max_(i,a) Σ_(j,b) |B_ij[a,b]|`` bound its
spectral radius: the former bounds the induced block infinity/Euclidean norm,
and the latter is the scalar infinity norm. Their global minimum is still
an upper bound and avoids an unnecessary square-root-of-three penalty for
near-isotropic blocks. Both bounds are recomputed from the current free-free
values together (one kernel; in-place, graph-stable storage). No approximate
eigenvalue estimate is used as an upper bound. Eigenvalues below the interval are
damped less but remain positive, which CG tolerates. Each application costs
``degree−1`` operator actions plus ``degree`` fused vector kernels, replacing
full CG iterations whose dot products and updates dominate submission latency
on small systems.
"""

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _block_gershgorin(
    offsets: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    values: wp.array(dtype=wp.mat33f),
    inverse_diagonal: wp.array(dtype=wp.mat33f),
    fixed: wp.array(dtype=int),
    bound: wp.array(dtype=float),
):
    # Preconditioned residuals vanish on constrained rows for the projected
    # operator, so the relevant spectrum is that of the free-free subblock.
    row = wp.tid()
    if fixed.shape[0] > 0 and fixed[row] != 0:
        return
    inverse = inverse_diagonal[row]
    total = float(0.0)
    scalar_rows = wp.vec3(0.0)
    for block in range(offsets[row], offsets[row + 1]):
        if fixed.shape[0] > 0 and fixed[columns[block]] != 0:
            continue
        product = inverse * values[block]
        frobenius = float(0.0)
        for i in range(3):
            for j in range(3):
                frobenius += product[i, j] * product[i, j]
                scalar_rows[i] += wp.abs(product[i, j])
        total += wp.sqrt(frobenius)
    wp.atomic_max(bound, 0, total)
    # Two independently valid spectral upper bounds. Their GLOBAL minima
    # remain bounds; taking minima per block row would not be justified.
    wp.atomic_max(bound, 1, wp.max(scalar_rows[0], wp.max(scalar_rows[1], scalar_rows[2])))


@wp.kernel(enable_backward=False)
def _chebyshev_coefficients(
    bound: wp.array(dtype=float),
    ratio: float,
    degree: int,
    coefficients: wp.array(dtype=float),
):
    # One thread: tiny fixed-degree recurrence. interval [b/ratio, b].
    b = wp.max(wp.min(bound[0], bound[1]), 1e-30)
    bound[0] = b
    a = b / ratio
    theta = 0.5 * (b + a)
    delta = 0.5 * (b - a)
    sigma = theta / delta
    rho = 1.0 / sigma
    coefficients[0] = 1.0 / theta
    for k in range(1, degree):
        previous = rho
        rho = 1.0 / (2.0 * sigma - previous)
        coefficients[2 * k] = rho * previous
        coefficients[2 * k + 1] = 2.0 * rho / delta


@wp.kernel(enable_backward=False)
def _first_step(
    inverse_diagonal: wp.array(dtype=wp.mat33f),
    r: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=float),
    direction: wp.array(dtype=wp.vec3),
    z: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    step = coefficients[0] * (inverse_diagonal[i] * r[i])
    direction[i] = step
    z[i] = step


@wp.kernel(enable_backward=False)
def _next_step(
    inverse_diagonal: wp.array(dtype=wp.mat33f),
    r: wp.array(dtype=wp.vec3),
    Az: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=float),
    k: int,
    direction: wp.array(dtype=wp.vec3),
    z: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    preconditioned = inverse_diagonal[i] * (r[i] - Az[i])
    step = coefficients[2 * k] * direction[i] + coefficients[2 * k + 1] * preconditioned
    direction[i] = step
    z[i] = z[i] + step


class ChebyshevJacobi:
    """SPD Chebyshev/block-Jacobi preconditioner with graph-stable storage."""

    def __init__(self, operator, inverse_diagonal, *, degree, ratio, device):
        if degree < 2:
            raise ValueError("Chebyshev degree must be at least 2; use plain block Jacobi otherwise")
        if not np.isfinite(ratio) or ratio <= 1:
            raise ValueError("Chebyshev interval ratio must be finite and greater than one")
        count = inverse_diagonal.shape[0]
        self.operator = operator
        self.inverse_diagonal = inverse_diagonal
        self.degree, self.ratio = int(degree), float(ratio)
        self.device, self.count = device, count
        # Element 0 exposes the selected upper bound after refresh. Element 1
        # is persistent reduction scratch for the independent scalar bound.
        self.bound = wp.zeros(2, dtype=float, device=device)
        self.coefficients = wp.zeros(2 * self.degree, dtype=float, device=device)
        self.Az = wp.empty(count, dtype=wp.vec3, device=device)
        self.direction = wp.empty(count, dtype=wp.vec3, device=device)
        self._no_fixed = wp.zeros(0, dtype=int, device=device)
        from warp.optim.linear import LinearOperator  # noqa: PLC0415

        #: The Warp operator handed to ``cg(..., M=...)``.
        self.preconditioner = LinearOperator((count, count), wp.vec3, device, self.matvec)

    def refresh(self, matrix, *, fixed=None, bound_diagonal=None):
        """Recompute the spectral bound and coefficients from current values.

        ``matrix`` supplies the assembled values; with ``fixed`` given, fixed
        rows and columns are excluded so the bound describes the free-free
        subblock that the projected operator exposes to the Krylov iteration.
        ``bound_diagonal`` optionally supplies a matrix-indexed inverse
        diagonal when the application-side diagonal uses different indexing
        (the compact free-node formulation); the retained blocks are equal, so
        the bound is identical. In-place and graph-safe.
        """
        diagonal = self.inverse_diagonal if bound_diagonal is None else bound_diagonal
        if diagonal.shape[0] != matrix.nrow:
            raise ValueError("The spectral bound needs a matrix-indexed inverse diagonal")
        self.bound.zero_()
        wp.launch(
            _block_gershgorin,
            matrix.nrow,
            [
                matrix.offsets,
                matrix.columns,
                matrix.values,
                diagonal,
                self._no_fixed if fixed is None else fixed,
                self.bound,
            ],
            device=self.device,
        )
        wp.launch(
            _chebyshev_coefficients,
            1,
            [self.bound, self.ratio, self.degree, self.coefficients],
            device=self.device,
        )

    def matvec(self, x, y, z, alpha, beta):
        if alpha != 1.0 or beta != 0.0 or z.ptr == x.ptr:
            raise ValueError("ChebyshevJacobi supports the CG preconditioner form z = M(r) only")
        wp.launch(
            _first_step,
            self.count,
            [self.inverse_diagonal, x, self.coefficients, self.direction, z],
            device=self.device,
        )
        for k in range(1, self.degree):
            self.operator.matvec(z, self.Az, self.Az, alpha=1.0, beta=0.0)
            wp.launch(
                _next_step,
                self.count,
                [self.inverse_diagonal, x, self.Az, self.coefficients, k, self.direction, z],
                device=self.device,
            )
