# SPDX-License-Identifier: Apache-2.0

"""Blocks on the diagonal of the system matrix for the length of one solve.

A force law that is linearly implicit in a step and whose tangent is one 3×3 block per
particle (a penalty against the world, such as a floor) adds that block to the diagonal
of the system matrix before the solve and takes it out after it. The elastic predictor
reads differences of velocities and so never sees a diagonal block; the law's own share
of the predictor is part of its force, as for the material grips.

The block is written to the copies of the matrix that a solve reads, and to nothing else:
the assembled matrix or, with prepared tetrahedral updates, its compact copy, and the
inverse block of the block-Jacobi preconditioner. Both are saved before and put back
after, so the solver is afterwards what it was, to the bit; a refresh of the tangent or
of the preconditioner in between is not supported. Only values are written, at slots that
the kernel finds in the row, so a captured graph follows a change of the topology.
"""

import warp as wp

from .kernels import inverse_diagonal_block


@wp.kernel(enable_backward=False)
def _add_blocks(
    blocks: wp.array[wp.mat33],
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    inverse: wp.array[wp.mat33],
    slots: wp.array[int],
    saved_values: wp.array[wp.mat33],
    saved_inverse: wp.array[wp.mat33],
):
    node = wp.tid()
    slots[node] = -1
    block = blocks[node]
    if wp.ddot(block, block) > 0.0:
        first, end = int(offsets[node]), int(offsets[node + 1])
        while first < end:
            middle = (first + end) // 2
            if columns[middle] < node:
                first = middle + 1
            else:
                end = middle
        # Every row of a system matrix has its diagonal block (the mass).
        if first < offsets[node + 1] and columns[first] == node:
            slots[node] = first
            saved_values[node] = values[first]
            saved_inverse[node] = inverse[node]
            shifted = values[first] + block
            values[first] = shifted
            inverse[node] = inverse_diagonal_block(shifted)


@wp.kernel(enable_backward=False)
def _remove_blocks(
    slots: wp.array[int],
    saved_values: wp.array[wp.mat33],
    saved_inverse: wp.array[wp.mat33],
    values: wp.array[wp.mat33],
    inverse: wp.array[wp.mat33],
):
    node = wp.tid()
    slot = slots[node]
    if slot >= 0:
        values[slot] = saved_values[node]
        inverse[node] = saved_inverse[node]
        slots[node] = -1


class SolveDiagonal:
    """Diagonal blocks that stand in a :class:`SolverFEM`'s system for one solve.

    ``add`` before the solver's step, ``remove`` after it; each is one launch over the
    particles and neither allocates, synchronizes or reads back. Create it before a CUDA
    graph capture. The solver needs the ``block_diag`` preconditioner without Chebyshev
    acceleration and no fixed particles (their solve reads projected copies of the matrix).

    Args:
        particle_count: Particles of the solver's model.
        device: Device of the solver's model.
    """

    def __init__(self, particle_count, device):
        self.device = wp.get_device(device)
        self._slots = wp.full(particle_count, -1, dtype=int, device=self.device)
        self._saved_values = wp.zeros(particle_count, dtype=wp.mat33, device=self.device)
        self._saved_inverse = wp.zeros(particle_count, dtype=wp.mat33, device=self.device)

    @staticmethod
    def _solve_arrays(solver):
        """The matrix that the solver's Krylov products read, and the inverse blocks of its preconditioner."""
        if solver.preconditioner_type != "block_diag" or solver.linear_solver_chebyshev_degree >= 2:
            raise ValueError("Diagonal blocks for a solve need the plain block_diag preconditioner")
        if solver._fixed_particles is not None:
            raise ValueError("Diagonal blocks for a solve are not supported with fixed_particle_indices")
        matrix = solver.A_bsr if solver._streaming is None else solver._streaming.compact_matrix
        return matrix, solver._pc_inv_diag

    def add(self, solver, blocks):
        """Add ``blocks`` [kg], one ``wp.mat33`` per particle, to the diagonal that the next solve reads.

        A zero block leaves its particle untouched. Call after everything else that changes
        the matrix in this step, and :meth:`remove` after the solve.
        """
        matrix, inverse = self._solve_arrays(solver)
        if blocks.shape != self._slots.shape or blocks.dtype != wp.mat33 or blocks.device != self.device:
            raise ValueError("blocks must hold one wp.mat33 per particle on the solver's device")
        wp.launch(
            _add_blocks,
            dim=self._slots.shape[0],
            inputs=[
                blocks,
                matrix.offsets,
                matrix.columns,
                matrix.values,
                inverse,
                self._slots,
                self._saved_values,
                self._saved_inverse,
            ],
            device=self.device,
        )

    def remove(self, solver):
        """Put back what :meth:`add` replaced; without a preceding ``add`` nothing is written."""
        matrix, inverse = self._solve_arrays(solver)
        wp.launch(
            _remove_blocks,
            dim=self._slots.shape[0],
            inputs=[self._slots, self._saved_values, self._saved_inverse, matrix.values, inverse],
            device=self.device,
        )
