# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compact packed SpMV/Chebyshev recurrence with guarded fallback.

Bound/coefficient generation and the first step use the original implementation.
Input/output iterate buffers alternate so
no row overwrites a value another row still needs within one CUDA launch.
"""

import warp as wp

from .chebyshev import ChebyshevJacobi, _first_step
from .packed_bsr import BLOCK_DIM, _overlap, _shuffle_down, packed_supported

GROUP_SIZE = 16


@wp.kernel(enable_backward=False, grid_stride=False)
def _packed_next_step(
    nrow: int,
    offsets: wp.array(dtype=int),
    row_counts: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    values: wp.array(dtype=wp.mat33),
    inverse_diagonal: wp.array(dtype=wp.mat33),
    r: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=float),
    k: int,
    direction: wp.array(dtype=wp.vec3),
    current: wp.array(dtype=wp.vec3),
    following: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    row = tid // GROUP_SIZE
    lane = tid % GROUP_SIZE
    total = wp.vec3(0.0)
    if row < nrow:
        begin = offsets[row]
        end = offsets[row + 1]
        if row_counts:
            end = begin + row_counts[row]
        for block in range(begin + lane, end, GROUP_SIZE):
            total += values[block] * current[columns[block]]
    for level in range(4):
        delta = GROUP_SIZE >> (level + 1)
        total += wp.vec3(
            _shuffle_down(total[0], delta, GROUP_SIZE),
            _shuffle_down(total[1], delta, GROUP_SIZE),
            _shuffle_down(total[2], delta, GROUP_SIZE),
        )
    if row < nrow and lane == 0:
        # Identical polynomial update to the original _next_step. The SpMV
        # result is float32 here, as in its original temporary array.
        preconditioned = inverse_diagonal[row] * (r[row] - total)
        step = coefficients[2 * k] * direction[row] + coefficients[2 * k + 1] * preconditioned
        direction[row] = step
        following[row] = current[row] + step


class FusedPackedChebyshev(ChebyshevJacobi):
    """Optional compact adapter; false/unknown eligibility uses original matvec."""

    def __init__(self, operator, inverse_diagonal, *, matrix, compact=False, degree, ratio, device, bound_source=None):
        super().__init__(operator, inverse_diagonal, degree=degree, ratio=ratio, device=device)
        self.matrix = matrix
        self.compact = compact is True
        # Optional identities from SolverFEM's proven free-node construction.
        # They permit the unchanged full-indexed bound refresh while applying
        # the identical free-free blocks. Arbitrary full/projected operators
        # never receive this compact adapter from SolverFEM.
        self._bound_source = bound_source
        self._refresh_is_compact = False
        self.capture_owners = (
            matrix,
            matrix.offsets,
            matrix.columns,
            matrix.values,
            matrix.row_counts,
            self.Az,
            self.direction,
        )

    def refresh(self, matrix, *, fixed=None, bound_diagonal=None):
        super().refresh(matrix, fixed=fixed, bound_diagonal=bound_diagonal)
        direct = matrix is self.matrix and fixed is None and bound_diagonal is None
        source = self._bound_source
        mapped = (
            source is not None
            and matrix is source[0]
            and fixed is source[1]
            and bound_diagonal is source[2]
            and not matrix.values.requires_grad
            and not bound_diagonal.requires_grad
        )
        self._refresh_is_compact = direct or mapped

    def can_fuse(self, x, z):
        if (
            not self.compact
            or not self._refresh_is_compact
            or self.count == 0
            or self.matrix.nrow != self.count
            or self.matrix.ncol != self.count
            or not packed_supported(self.matrix, x, z)
        ):
            return False
        arrays = (self.inverse_diagonal, self.coefficients, self.direction, self.Az)
        if any(a.requires_grad or not a.is_contiguous or a.device != self.matrix.values.device for a in arrays):
            return False
        return (
            self.inverse_diagonal.shape == (self.count,)
            and self.inverse_diagonal.dtype == wp.mat33
            and not _overlap(x, z)
            and not _overlap(self.Az, z)
            and not _overlap(self.Az, x)
            and not _overlap(self.direction, z)
            and not _overlap(self.direction, x)
        )

    def matvec(self, x, y, z, alpha, beta):
        # Preserve the exact original preconditioner call contract.
        if alpha != 1.0 or beta != 0.0 or z.ptr == x.ptr:
            raise ValueError("ChebyshevJacobi supports the CG preconditioner form z = M(r) only")
        if not self.can_fuse(x, z):
            return super().matvec(x, y, z, alpha, beta)
        # degree-1 swaps: choose first destination so the final one is z.
        current = z if self.degree % 2 else self.Az
        wp.launch(
            _first_step,
            self.count,
            [self.inverse_diagonal, x, self.coefficients, self.direction, current],
            device=self.device,
        )
        dim = ((self.count * GROUP_SIZE + BLOCK_DIM - 1) // BLOCK_DIM) * BLOCK_DIM
        matrix = self.matrix
        for k in range(1, self.degree):
            following = self.Az if current is z else z
            wp.launch(
                _packed_next_step,
                dim,
                block_dim=BLOCK_DIM,
                inputs=[
                    matrix.nrow,
                    matrix.offsets,
                    matrix.row_counts,
                    matrix.columns,
                    matrix.values,
                    self.inverse_diagonal,
                    x,
                    self.coefficients,
                    k,
                    self.direction,
                    current,
                    following,
                ],
                device=self.device,
            )
            current = following
        assert current is z
