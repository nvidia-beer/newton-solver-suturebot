# SPDX-License-Identifier: Apache-2.0

"""Stationary Dirichlet boundaries without changing the physical stiffness."""

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator, aslinearoperator
from warp.sparse import bsr_copy, bsr_set_from_triplets, bsr_zeros


@wp.kernel(enable_backward=False)
def _restore(indices: wp.array[int], reference: wp.array[wp.vec3], q: wp.array[wp.vec3], v: wp.array[wp.vec3]):
    node = indices[wp.tid()]
    q[node] = reference[node]
    v[node] = wp.vec3(0.0)


@wp.kernel(enable_backward=False)
def _gather_compact_values(
    slots: wp.array[int],
    values: wp.array[wp.mat33f],
    compact: wp.array[wp.mat33f],
):
    i = wp.tid()
    compact[i] = values[slots[i]]


@wp.kernel(enable_backward=False)
def _gather_free(free: wp.array[int], source: wp.array[wp.vec3], compact: wp.array[wp.vec3]):
    i = wp.tid()
    compact[i] = source[free[i]]


@wp.kernel(enable_backward=False)
def _gather_free_blocks(free: wp.array[int], source: wp.array[wp.mat33f], compact: wp.array[wp.mat33f]):
    i = wp.tid()
    compact[i] = source[free[i]]


@wp.kernel(enable_backward=False)
def _scatter_free(free: wp.array[int], compact: wp.array[wp.vec3], destination: wp.array[wp.vec3]):
    i = wp.tid()
    destination[free[i]] = compact[i]


@wp.kernel(enable_backward=False)
def _restore_positions(indices: wp.array[int], reference: wp.array[wp.vec3], q: wp.array[wp.vec3]):
    node = indices[wp.tid()]
    q[node] = reference[node]


@wp.kernel(enable_backward=False)
def _zero_fixed(indices: wp.array[int], values: wp.array[wp.vec3]):
    values[indices[wp.tid()]] = wp.vec3(0.0)


@wp.kernel(enable_backward=False)
def _projected_matvec(
    fixed: wp.array[int],
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    z: wp.array[wp.vec3],
    alpha: float,
    beta: float,
):
    row = wp.tid()
    result = wp.vec3(0.0)
    if fixed[row] != 0:
        result = x[row]
    else:
        for block in range(offsets[row], offsets[row + 1]):
            column = columns[block]
            if fixed[column] == 0:
                result += values[block] * x[column]
    result *= alpha
    if beta != 0.0:
        result += beta * y[row]
    z[row] = result


@wp.kernel(enable_backward=False)
def _project_values(
    fixed: wp.array[int],
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    projected: wp.array[wp.mat33],
):
    row = wp.tid()
    for block in range(offsets[row], offsets[row + 1]):
        column = columns[block]
        value = wp.mat33(0.0)
        if fixed[row] != 0:
            if row == column:
                value = wp.identity(n=3, dtype=float)
        elif fixed[column] == 0:
            value = values[block]
        projected[block] = value


class FixedParticles:
    """Solve (P A P + I-P) dv = P b, retaining A for the elastic predictor.

    P is the diagonal projector onto free nodes. The original matrix must keep
    its constrained columns: the predictor's block-row differences need them
    even when the prescribed boundary velocity is zero.
    """

    def __init__(self, model, matrix, indices):
        self.device = model.device
        self.indices = wp.array(indices, dtype=int, device=self.device)
        mask = np.zeros(model.particle_count, dtype=np.int32)
        mask[indices] = 1
        self.mask = wp.array(mask, dtype=int, device=self.device)
        self.reference = wp.clone(model.particle_q)

        def matvec(x, y, z, alpha, beta):
            wp.launch(
                _projected_matvec,
                model.particle_count,
                inputs=[self.mask, matrix.offsets, matrix.columns, matrix.values, x, y, z, alpha, beta],
                device=self.device,
            )

        self.operator = LinearOperator((model.particle_count, model.particle_count), wp.vec3, self.device, matvec)
        self._masked_operator = self.operator
        self._projected = None
        self._projected_key = None
        self._projected_version = None
        # Optional exact reduction to the free-free subblock. The Krylov
        # iteration then runs on free nodes only, with gathers/scatters at the
        # solve boundaries; fixed rows carry exact zeros in both formulations.
        self.free = wp.array(np.flatnonzero(mask == 0).astype(np.int32), dtype=int, device=self.device)
        self.compact_matrix = None
        self.compact_operator = None
        self._compact_slots = None
        self._compact_version = None

    def _prepare_compact(self, matrix, offsets, columns):
        """Build the free-free block structure and its source-slot gather map."""
        mask = self.mask.numpy().astype(bool)
        free = np.flatnonzero(~mask)
        renumber = np.full(mask.size, -1, dtype=np.int64)
        renumber[free] = np.arange(free.size)
        # Storage has reserve capacity; only the first nrow+1 offsets and the
        # blocks they reference are live.
        offsets = np.asarray(offsets[: matrix.nrow + 1], dtype=np.int64)
        columns = np.asarray(columns[: offsets[-1]], dtype=np.int64)
        rows = np.repeat(np.arange(mask.size), np.diff(offsets))
        keep = ~mask[rows] & ~mask[columns]
        if free.size == 0 or not np.any(keep):
            self.compact_matrix = self.compact_operator = None
            return
        compact_rows = renumber[rows[keep]].astype(np.int32)
        compact_columns = renumber[columns[keep]].astype(np.int32)
        compact = bsr_zeros(free.size, free.size, wp.mat33f, device=self.device)
        bsr_set_from_triplets(
            compact,
            wp.array(compact_rows, dtype=wp.int32, device=self.device),
            wp.array(compact_columns, dtype=wp.int32, device=self.device),
            # Placeholder values define the retained structure; pruning would
            # discard it before the first numerical gather.
            wp.array(np.zeros((compact_rows.size, 3, 3), np.float32), dtype=wp.mat33f, device=self.device),
            prune_numerical_zeros=False,
        )
        # Map every stored compact block back to its source slot explicitly;
        # column lists are sorted within each row of the canonical structure.
        compact_offsets = compact.offsets.numpy()[: compact.nrow + 1].astype(np.int64)
        stored_columns = compact.columns.numpy()[: compact_offsets[-1]].astype(np.int64)
        source_slots = np.empty(compact_offsets[-1], dtype=np.int64)
        for compact_row in range(compact.nrow):
            row = free[compact_row]
            row_columns = columns[offsets[row] : offsets[row + 1]]
            for slot in range(compact_offsets[compact_row], compact_offsets[compact_row + 1]):
                column = free[stored_columns[slot]]
                position = np.searchsorted(row_columns, column)
                if position >= row_columns.size or row_columns[position] != column:
                    raise RuntimeError("Compact structure does not match the source matrix")
                source_slots[slot] = offsets[row] + position
        self.compact_matrix = compact
        self._compact_slots = wp.array(source_slots, dtype=int, device=self.device)
        self.compact_operator = aslinearoperator(compact)
        self._compact_version = None
        self.compact_rhs = wp.zeros(free.size, dtype=wp.vec3, device=self.device)
        self.compact_increment = wp.zeros(free.size, dtype=wp.vec3, device=self.device)
        self.compact_diagonal = wp.zeros(free.size, dtype=wp.mat33f, device=self.device)

    def gather_solve_operands(self, rhs, increment):
        wp.launch(_gather_free, self.free.size, [self.free, rhs, self.compact_rhs], device=self.device)
        wp.launch(_gather_free, self.free.size, [self.free, increment, self.compact_increment], device=self.device)

    def scatter_increment(self, increment):
        wp.launch(_scatter_free, self.free.size, [self.free, self.compact_increment, increment], device=self.device)

    def gather_diagonal(self, inverse_diagonal):
        if self.compact_matrix is not None:
            wp.launch(
                _gather_free_blocks,
                self.free.size,
                [self.free, inverse_diagonal, self.compact_diagonal],
                device=self.device,
            )

    def prepare(self, matrix, *, values_version=None):
        """Refresh the derived operator once per solve; retain the predictor's A.

        ``values_version`` is an optional caller-maintained counter of the
        source matrix's assembled numerical values. When it matches the version
        already projected into the derived operator, the (pure, elementwise)
        value projection launch is skipped; the result is identical. ``None``
        always refreshes, preserving the original behavior.
        """
        if not self.device.is_cuda:
            return False
        key = (
            matrix.nrow,
            matrix.ncol,
            matrix.nnz,
            matrix.offsets.ptr,
            matrix.columns.ptr,
            matrix.values.ptr,
            matrix.values.shape,
            None if matrix.row_counts is None else matrix.row_counts.ptr,
        )
        changed = key != self._projected_key
        if changed:
            if self.device.is_capturing:
                raise RuntimeError("Prepare the fixed-particle operator before CUDA graph capture")
            # A diagonal is needed to represent the constrained identity using
            # the same sparse structure. Unusual empty rows retain the old path.
            offsets, columns = matrix.offsets.numpy(), matrix.columns.numpy()
            indices = self.indices.numpy()
            supported = matrix.row_counts is None and all(
                np.any(columns[offsets[row] : offsets[row + 1]] == row) for row in indices
            )
            self._projected = bsr_copy(matrix, structure_only=True) if supported else None
            if self._projected is not None:
                # Retain source capacity for in-place topology updates.
                self._projected.values = wp.empty_like(matrix.values)
                # Share topology so in-place connectivity changes remain visible.
                self._projected.offsets = matrix.offsets
                self._projected.columns = matrix.columns
                self._projected.nnz_sync()
                self.operator = aslinearoperator(self._projected)
                self._prepare_compact(matrix, offsets, columns)
            else:
                self.operator = self._masked_operator
                self.compact_matrix = self.compact_operator = None
            self._projected_key = key
            self._projected_version = None
        if self._projected is not None and (
            changed or values_version is None or values_version != self._projected_version
        ):
            wp.launch(
                _project_values,
                matrix.nrow,
                [self.mask, matrix.offsets, matrix.columns, matrix.values, self._projected.values],
                device=self.device,
            )
            if self.compact_matrix is not None:
                wp.launch(
                    _gather_compact_values,
                    self._compact_slots.size,
                    [self._compact_slots, matrix.values, self.compact_matrix.values],
                    device=self.device,
                )
            self._projected_version = values_version
        return changed

    def restore(self, state):
        wp.launch(
            _restore,
            self.indices.size,
            inputs=[self.indices, self.reference, state.particle_q, state.particle_qd],
            device=self.device,
        )

    def project_rhs(self, rhs):
        wp.launch(_zero_fixed, self.indices.size, inputs=[self.indices, rhs], device=self.device)

    def restore_positions(self, positions):
        wp.launch(
            _restore_positions,
            self.indices.size,
            inputs=[self.indices, self.reference, positions],
            device=self.device,
        )
