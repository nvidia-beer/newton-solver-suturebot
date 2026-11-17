# SPDX-License-Identifier: Apache-2.0

"""Point holds of single nodes in the assembled tissue matrix.

A held node keeps the place it was put at through one implicit step. Its diagonal block of
the system matrix gets

    A_ii  <-  A_ii + HOLD_FACTOR * trace(A_ii) / 3 * I

and its velocity is set to zero by the owner, so the step leaves it within
``force / (HOLD_FACTOR * K_ii)`` of that place while every other node finds its equilibrium
around it. Nothing is added to the nodal forces: what the hold carries is read afterwards as
the elastic force on the node.

``HOLD_FACTOR`` = 4096 = 2^12 = 1 / sqrt(float32 epsilon): the remainder of the hold (1 / 4096
of what a free node would move) balances the round-off of the node's own block inside the sum
(epsilon * 4096).

The block is written to every copy of the matrix that a solve reads: the assembled matrix
(which the elastic predictor and the CPU solve read), the projected copy and the compact
free-free copy of the fixed-particle operator (CUDA), and its inverse to the block-Jacobi
diagonals. Only values are written, at fixed slots, so no CUDA graph is invalidated. The
Chebyshev bound is taken from the matrix without holds (a hold only lowers the row sum of
its own row), so :meth:`NodeHolds.refresh` runs after the solver has refreshed its
preconditioner on the freshly assembled matrix.
"""

import numpy as np
import warp as wp

from .kernels import inverse_diagonal_block

# 2^12 = 1 / sqrt(float32 epsilon = 2^-24).
HOLD_FACTOR = 4096.0


@wp.kernel(enable_backward=False)
def _capture_pristine(
    nodes: wp.array[int],
    slots: wp.array[int],
    values: wp.array[wp.mat33f],
    pristine: wp.array[wp.mat33f],
):
    """Keep the freshly assembled diagonal block of every listed node."""
    index = wp.tid()
    pristine[index] = values[slots[index]]


@wp.kernel(enable_backward=False)
def _apply_holds(
    nodes: wp.array[int],
    held: wp.array[int],
    pristine: wp.array[wp.mat33f],
    factor: float,
    slots: wp.array[int],
    values: wp.array[wp.mat33f],
    inverse: wp.array[wp.mat33f],
    projected_values: wp.array[wp.mat33f],
    compact_slots: wp.array[int],
    compact_values: wp.array[wp.mat33f],
    compact_rows: wp.array[int],
    compact_inverse: wp.array[wp.mat33f],
):
    """Write the diagonal block of every listed node, with its hold if it is held, and the inverse block.

    ``projected_values`` shares the slots of ``values``. Without a projected or a compact copy
    the caller passes the assembled arrays again: the same block is then written twice.
    """
    index = wp.tid()
    node = nodes[index]
    block = pristine[index]
    if held[node] != 0:
        scale = factor * (block[0, 0] + block[1, 1] + block[2, 2]) / 3.0
        block = block + scale * wp.identity(n=3, dtype=wp.float32)
    inverted = inverse_diagonal_block(block)
    values[slots[index]] = block
    inverse[node] = inverted
    projected_values[slots[index]] = block
    compact_values[compact_slots[index]] = block
    compact_inverse[compact_rows[index]] = inverted


class NodeHolds:
    """Diagonal hold blocks of a fixed set of nodes of a :class:`SolverFEM`.

    Args:
        solver: The tissue solver, after its first :meth:`SolverFEM.refresh_tangent` (the
            sparse structure is then final), with the ``block_diag`` preconditioner and
            without streaming or component scaling.
        nodes: Particles that can be held, shape (count,); none of them may be a fixed
            particle of the solver.
        held: One ``int`` per particle of the model on its device; nonzero holds the node.
            The owner writes it (on the device) and then calls :meth:`apply`.

    The holds join the solver: :meth:`SolverFEM.refresh_tangent` calls :meth:`refresh` after
    it has rewritten the matrix. :meth:`apply` after a change of ``held`` is the owner's call;
    both only launch kernels. A solver takes one set of holds: a second registration is refused.
    """

    def __init__(self, solver, nodes, held):
        model = solver.model
        device = model.device
        nodes = np.asarray(nodes, dtype=np.int32)
        if nodes.ndim != 1 or not len(nodes) or len(np.unique(nodes)) != len(nodes):
            raise ValueError("nodes must be a nonempty vector of distinct particles")
        if nodes.min() < 0 or nodes.max() >= model.particle_count:
            raise ValueError("Held nodes must be particles of the model")
        if held.shape != (model.particle_count,) or held.dtype != wp.int32 or held.device != device:
            raise ValueError("held must be one int per particle on the device of the model")
        if solver.preconditioner_type != "block_diag" or solver._streaming is not None:
            raise ValueError("Node holds need the assembled matrix with the block_diag preconditioner")
        if solver._component_scaling is not None:
            raise ValueError("Node holds do not support component scaling")
        if solver._tangent_triplet_slots is None:
            raise RuntimeError("Call refresh_tangent once before registering node holds")
        if getattr(solver, "_node_holds", None) is not None:
            # The solver refreshes one set of holds after it has rewritten its matrix; a second set would
            # replace the first, whose next apply() would write blocks captured before that refresh.
            raise ValueError("The solver already has node holds: one NodeHolds per solver")
        fixed = solver._fixed_particles
        if fixed is not None and np.any(fixed.mask.numpy()[nodes] != 0):
            raise ValueError("A fixed particle cannot be held")
        matrix = solver.A_bsr
        offsets = matrix.offsets.numpy()[: matrix.nrow + 1]
        columns = matrix.columns.numpy()[: offsets[-1]]
        slots = np.empty(len(nodes), dtype=np.int32)
        for index, node in enumerate(nodes):
            hit = np.flatnonzero(columns[offsets[node] : offsets[node + 1]] == node)
            if len(hit) != 1:
                raise RuntimeError("Every held node needs one diagonal block in the assembled matrix")
            slots[index] = offsets[node] + hit[0]
        self.solver, self.device, self.held = solver, device, held
        self.count = len(nodes)
        self.factor = HOLD_FACTOR
        self._nodes_host = nodes
        self._nodes = wp.array(nodes, dtype=int, device=device)
        self._slots = wp.array(slots, dtype=int, device=device)
        self.pristine = wp.zeros(self.count, dtype=wp.mat33f, device=device)
        self._bound_key = None
        self._projected_values = self._compact_values = self._compact_inverse = None
        self._compact_slots = self._compact_rows = None
        self._bind()
        self.capture_pristine()
        solver._node_holds = self

    def _key(self):
        fixed = self.solver._fixed_particles
        projected = None if fixed is None else fixed._projected
        compact = None if fixed is None else fixed.compact_matrix
        return (
            self.solver.A_bsr.values.ptr,
            self.solver._pc_inv_diag.ptr,
            None if projected is None else projected.values.ptr,
            None if compact is None else (compact.values.ptr, fixed.compact_diagonal.ptr),
        )

    def _bind(self):
        """Find the copies of the matrix that the solves read; a host step, outside CUDA graph capture."""
        key = self._key()
        if key == self._bound_key:
            return
        if self.device.is_capturing:
            raise RuntimeError("Bind the node holds before CUDA graph capture: step the solver once first")
        solver = self.solver
        fixed = solver._fixed_particles
        projected = None if fixed is None else fixed._projected
        compact = None if fixed is None else fixed.compact_matrix
        self._projected_values = solver.A_bsr.values if projected is None else projected.values
        if compact is None:
            self._compact_values, self._compact_slots = solver.A_bsr.values, self._slots
            self._compact_inverse, self._compact_rows = solver._pc_inv_diag, self._nodes
        else:
            # Row of each node among the free nodes, and the slot of its diagonal block there.
            rows = np.searchsorted(fixed.free_host, self._nodes_host).astype(np.int32)
            if not np.array_equal(fixed.free_host[rows], self._nodes_host):
                raise RuntimeError("A held node is not a free node of the compact matrix")
            offsets = compact.offsets.numpy()[: compact.nrow + 1]
            columns = compact.columns.numpy()[: offsets[-1]]
            slots = np.empty(self.count, dtype=np.int32)
            for index, row in enumerate(rows):
                hit = np.flatnonzero(columns[offsets[row] : offsets[row + 1]] == row)
                if len(hit) != 1:
                    raise RuntimeError("Every held node needs one diagonal block in the compact matrix")
                slots[index] = offsets[row] + hit[0]
            self._compact_values, self._compact_inverse = compact.values, fixed.compact_diagonal
            self._compact_slots = wp.array(slots, dtype=int, device=self.device)
            self._compact_rows = wp.array(rows, dtype=int, device=self.device)
        self._bound_key = key

    def capture_pristine(self):
        """Keep the diagonal blocks of the matrix as assembled; call when no hold is written in it."""
        wp.launch(
            _capture_pristine,
            self.count,
            inputs=[self._nodes, self._slots, self.solver.A_bsr.values],
            outputs=[self.pristine],
            device=self.device,
        )

    def apply(self):
        """Write the blocks for the current ``held`` flags into every copy of the matrix and its inverse diagonals."""
        self._bind()
        solver = self.solver
        wp.launch(
            _apply_holds,
            self.count,
            inputs=[self._nodes, self.held, self.pristine, self.factor, self._slots],
            outputs=[
                solver.A_bsr.values,
                solver._pc_inv_diag,
                self._projected_values,
                self._compact_slots,
                self._compact_values,
                self._compact_rows,
                self._compact_inverse,
            ],
            device=self.device,
        )

    def refresh(self):
        """After the solver has rewritten its matrix and refreshed its preconditioner: capture and apply."""
        self.capture_pristine()
        self.apply()
