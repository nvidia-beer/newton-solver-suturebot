# SPDX-License-Identifier: Apache-2.0
"""Retained approximate Warp-CG columns, used only as guarded numerical guesses."""

import math
import time
from typing import Any

import numpy as np
import warp as wp
import warp._src.context as warp_context
from warp.optim.linear import LinearOperator, cg
from warp.sparse import bsr_assign, bsr_copy, bsr_get_diag

from ...utils.capture import PauseGC, ScopedCapture
from .fused_chebyshev import FusedPackedChebyshev
from .packed_bsr import packed_aslinearoperator

_TILE = 256


@wp.kernel(enable_backward=False)
def _inverse_blocks(diagonal: wp.array[wp.mat33], inverse: wp.array[wp.mat33]):
    i = wp.tid()
    inverse[i] = wp.inverse(diagonal[i])


@wp.kernel(enable_backward=False)
def _unit_column(index: wp.array[int], b: wp.array[float], x: wp.array[float]):
    i = wp.tid()
    b[i] = float(i == index[0])
    x[i] = 0.0


@wp.kernel(enable_backward=False)
def _true_residual(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    x: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    squared: wp.array[wp.float64],
):
    row = wp.tid()
    total = wp.float64(0.0)
    for axis in range(3):
        ax = wp.float64(0.0)
        for entry in range(offsets[row], offsets[row + 1]):
            value = values[entry]
            vector = x[columns[entry]]
            for component in range(3):
                ax += wp.float64(value[axis, component]) * wp.float64(vector[component])
        r = wp.float64(b[row][axis]) - ax
        total += r * r
    squared[row] = total


@wp.kernel(enable_backward=False)
def _partial_sums(values: wp.array[wp.float64], partials: wp.array[wp.float64]):
    block, _lane = wp.tid()
    tile = wp.tile_load(values, shape=_TILE, offset=block * _TILE)
    wp.tile_store(partials, wp.tile_sum(tile), offset=block)


@wp.kernel(enable_backward=False)
def _quality_packet(
    sums: wp.array[wp.float64],
    recursive: wp.array(dtype=Any),
    threshold: wp.array(dtype=Any),
    iterations: wp.array[int],
    slot: wp.array[int],
    quality: wp.array2d[wp.float64],
):
    total = wp.float64(0.0)
    for i in range(sums.shape[0]):
        total += sums[i]
    row = slot[0]
    quality[row, 0] = total
    quality[row, 1] = wp.float64(recursive[0])
    quality[row, 2] = wp.float64(threshold[0])
    quality[row, 3] = wp.float64(iterations[0])


# The argument type that the engine's CG passes, declared so that the module has its final
# hash when it is imported: the first column solve adds no overload and loads no second build.
wp.overload(_quality_packet, {"recursive": wp.array(dtype=wp.float32), "threshold": wp.array(dtype=wp.float32)})


def matrix_key(matrix):
    return (id(matrix), matrix.nrow, matrix.ncol, matrix.offsets.ptr, matrix.columns.ptr, matrix.values.ptr)


class WarpColumns:
    """Own an immutable A0 and one captured CG setup solve, outside physical trials.

    This tolerance controls a guess only. Every physical trial still solves its
    current A with the original tolerance and budget after residual selection.

    The constructor issues the column solve once as written, captures it and launches
    the graph once, so the first requested column neither loads a kernel nor builds the
    executable graph. :meth:`rebase` replaces A0 by the current values of a matrix of
    the same structure, in the storage of this engine: an engine made ahead of the first
    request then gives the columns of one constructed at that request.

    The columns live in fixed storage that the constructor makes once: ``held`` keeps
    every solved column in the slot it was solved into (``capacity`` slots of ``n``
    values), ``ordered`` receives the columns of a request in the order of its support,
    and row ``slot`` of ``quality`` holds the residual packet of the column in that
    slot. :meth:`solve` writes a column and its packet and reads nothing back;
    :meth:`diagnose` is the one read of the packets.
    """

    def __init__(self, matrix, *, tolerance=3e-4, maxiter=160, capacity=96, degree=4):
        if (
            matrix.nrow < 1
            or matrix.nrow != matrix.ncol
            or matrix.block_shape != (3, 3)
            or matrix.values.dtype != wp.mat33
            or matrix.values.requires_grad
            or matrix.row_counts is not None
            or warp_context.runtime.tape is not None
        ):
            raise ValueError("Warp response columns require a non-gradient compact BSR3x3 matrix")
        if not math.isfinite(tolerance) or tolerance <= 0 or maxiter < 1 or capacity < 1:
            raise ValueError("Invalid approximate-basis settings")
        self.device = matrix.device
        self.matrix = bsr_copy(matrix)
        self.n = 3 * matrix.nrow
        self.tolerance, self.maxiter, self.capacity = tolerance, maxiter, capacity
        self.closed = False
        self.operator = packed_aslinearoperator(self.matrix)
        # Match SolverFEM's upstream CPU tiled-dot workaround without changing
        # installed Warp or the ordinary unbatched CUDA algorithm.
        if self.device.is_cpu:
            self.operator = LinearOperator(
                self.operator.shape,
                self.operator.dtype,
                self.device,
                self.operator.matvec,
                batch_offsets=wp.array([0, self.n], dtype=int, device=self.device),
                max_batch_length=self.n,
            )
        self.diagonal = bsr_get_diag(self.matrix)
        self.inverse = wp.empty_like(self.diagonal)
        wp.launch(_inverse_blocks, matrix.nrow, [self.diagonal, self.inverse], device=self.device)
        self.chebyshev = FusedPackedChebyshev(
            self.operator,
            self.inverse,
            matrix=self.matrix,
            compact=True,
            degree=degree,
            ratio=256.0,
            device=self.device,
        )
        self.chebyshev.refresh(self.matrix)
        self.b = wp.zeros(matrix.nrow, dtype=wp.vec3, device=self.device)
        self.x = wp.zeros_like(self.b)
        self.index = wp.zeros(1, dtype=int, device=self.device)
        self.slot = wp.zeros(1, dtype=int, device=self.device)
        self.squared = wp.zeros(matrix.nrow, dtype=wp.float64, device=self.device)
        self.partials = wp.zeros((matrix.nrow + _TILE - 1) // _TILE, dtype=wp.float64, device=self.device)
        self.quality = wp.zeros((capacity, 4), dtype=wp.float64, device=self.device)
        # Fixed storage of the columns (see the class text). Nothing replaces these arrays.
        self.held = wp.zeros(self.n * capacity, dtype=float, device=self.device)
        self.ordered = wp.zeros(self.n * capacity, dtype=float, device=self.device)
        # The solved column, as the scalars that a slot of ``held`` takes.
        self.x_scalar = self.x.view(dtype=float).flatten()
        self.solver = cg(
            self.operator,
            self.b,
            self.x,
            M=self.chebyshev.preconditioner,
            tol=tolerance,
            atol=0.0,
            maxiter=maxiter,
            check_every=0,
            use_cuda_graph=self.device.is_cuda,
            run=False,
        )
        self.graph = None
        # Prepare all kernels/scratch before capture. No physical state is bound
        # to this engine; its immutable matrix and all scratch remain owned here.
        self._solve_column()
        if self.device.is_cuda:
            with ScopedCapture(device=self.device) as captured:
                self._solve_column()
            self.graph = captured.graph
            # The executable graph is built at the first launch: here, not at the first column.
            wp.capture_launch(self.graph)
        self.records = []

    def rebase(self, matrix):
        """Take the current values of ``matrix`` as A0; no storage of the engine is replaced.

        ``matrix`` has the structure of the matrix this engine was constructed on. The
        copy, its block diagonal, the inverse blocks and the Chebyshev bound are written
        in place by the calls that the constructor makes, so the captured column solve
        addresses the same arrays before and after.
        """
        if self.closed:
            raise RuntimeError("Response engine is closed")
        if self.device.is_capturing:
            raise RuntimeError("Rebase the response engine outside CUDA graph capture")
        if matrix.nrow != self.matrix.nrow or matrix.ncol != self.matrix.ncol:
            raise ValueError("The response engine is rebased on a matrix of its own structure")
        bsr_assign(dest=self.matrix, src=matrix, topology="masked")
        bsr_get_diag(self.matrix, out=self.diagonal)
        wp.launch(_inverse_blocks, self.matrix.nrow, [self.diagonal, self.inverse], device=self.device)
        self.chebyshev.refresh(self.matrix)

    def _solve_column(self):
        wp.launch(
            _unit_column,
            self.n,
            [self.index, self.b.view(dtype=float).flatten(), self.x.view(dtype=float).flatten()],
            device=self.device,
        )
        with PauseGC():
            self.solver(b=self.b, M=self.chebyshev.preconditioner)
        wp.launch(
            _true_residual,
            self.matrix.nrow,
            [self.matrix.offsets, self.matrix.columns, self.matrix.values, self.x, self.b, self.squared],
            device=self.device,
        )
        sums = self.squared
        if self.device.is_cuda:
            wp.launch_tiled(
                _partial_sums,
                dim=self.partials.size,
                inputs=[self.squared, self.partials],
                block_dim=_TILE,
                device=self.device,
            )
            sums = self.partials
        wp.launch(
            _quality_packet,
            1,
            [
                sums,
                self.solver._tiled_dot.col(0),
                self.solver._residuals[1],
                self.solver._cur_iter_and_condition,
                self.slot,
                self.quality,
            ],
            device=self.device,
        )

    def solve(self, support, slots):
        """Solve the column of every degree of freedom of ``support`` into its slot of ``held``.

        ``slots`` names one slot of the fixed storage per column. The column and its residual
        packet (row ``slot`` of ``quality``) are written on the device. Nothing is allocated
        and nothing is read back; :meth:`diagnose` reads the packets.
        """
        if self.closed:
            raise RuntimeError("Response engine is closed")
        if len(support) > self.capacity:
            raise ValueError("Basis request exceeds diagnostic capacity")
        if len(slots) != len(support) or any(slot < 0 or slot >= self.capacity for slot in slots):
            raise ValueError("One slot of the fixed column storage per response column")
        start = time.perf_counter()
        for dof, slot in zip(support, slots, strict=True):
            self.index.assign(np.asarray([dof], dtype=np.int32))
            self.slot.assign(np.asarray([slot], dtype=np.int32))
            if self.graph is None:
                self._solve_column()
            else:
                wp.capture_launch(self.graph)
            wp.copy(self.held, self.x_scalar, dest_offset=slot * self.n, count=self.n)
        self.last_seconds = time.perf_counter() - start

    def diagnose(self, support, slots):
        """Read the residual packets of ``slots`` (one device read) and refuse a nonfinite column.

        Returns the record of every column, in the order of ``support``, and appends them
        to ``records``.
        """
        packet = self.quality.numpy()[np.asarray(slots, dtype=np.intp)]
        if not np.isfinite(packet).all() or np.any(packet[:, :3] < 0):
            raise RuntimeError("Nonfinite approximate response column")
        records = [
            {
                "dof": int(dof),
                "iterations": int(row[3]),
                "true_residual": math.sqrt(float(row[0])),
                "recursive_residual": math.sqrt(float(row[1])),
                "threshold": math.sqrt(float(row[2])),
                "hit_budget": int(row[3]) >= self.maxiter,
            }
            for dof, row in zip(support, packet, strict=True)
        ]
        self.records.extend(records)
        return records

    def close(self):
        if not self.closed and self.device.is_cuda:
            wp.synchronize_stream(wp.get_stream(self.device))
        self.closed = True
        self.graph = None


class WarpBasisProvider:
    """Append missing approximate columns while preserving the original A0.

    The columns are kept in the fixed storage of the engine (``WarpColumns.held``), which
    is made once with the engine: at :meth:`prepare`, or at the first request of a
    provider that was not prepared. A request allocates nothing. The array a request
    returns is a view of the engine's ``ordered`` storage and holds the columns of that
    request until the next one; a caller that keeps a basis copies it, as
    ``DeviceBasisGuess.load_basis`` does.

    Device reads: a request (:meth:`__call__`) reads the residual packets of the columns
    solved since the last request, once, and refuses a nonfinite column before it
    returns it. :meth:`extend` solves columns ahead of the request that returns them and
    reads nothing.
    """

    def __init__(self, *, tolerance=3e-4, maxiter=160, capacity=96, degree=4):
        self.settings = dict(tolerance=tolerance, maxiter=maxiter, capacity=capacity, degree=degree)
        self.engine = None
        self.key = None
        # Support of the last request or extension: the degrees of freedom whose columns are held.
        self.support = ()
        # The columns returned by the last request (a view of the engine's ordered storage).
        self.columns = None
        # Slot of every held column in the engine's storage, and the held columns whose
        # residual packet has not been read.
        self._slots = {}
        self._unread = []
        self.reference_builds = self.solves = 0
        # An engine made by prepare() holds the values of that moment; the first support
        # request takes the values of its own matrix as the reference.
        self._reference_pending = False

    def prepare(self, matrix):
        """Make the column engine for ``matrix`` ahead of the first support request.

        For start-up, outside capture: the engine's scratch is allocated, its column
        solve captured and launched, and its kernels loaded. The reference values are
        taken at the first support request (``WarpColumns.rebase``), which is also where
        ``reference_builds`` counts the reference, as it does for an engine made there.
        Nothing is done when the engine of this matrix exists.
        """
        key = matrix_key(matrix)
        if key != self.key:
            self.invalidate()
        if self.engine is None:
            self.engine = WarpColumns(matrix, **self.settings)
            self.key = key
            self._reference_pending = True

    def _hold(self, matrix, support):
        """Solve the columns of ``support`` that are not held into free slots of the fixed storage.

        The storage holds the columns of the support it was last asked for: the slots of
        other columns are free again. Returns the support as a tuple of integers and the
        degrees of freedom that were solved. Nothing is allocated or read once the engine
        of ``matrix`` exists.
        """
        support = tuple(int(v) for v in support)
        if not support or len(set(support)) != len(support) or min(support) < 0 or max(support) >= 3 * matrix.nrow:
            raise ValueError("Invalid response support")
        if len(support) > self.settings["capacity"]:
            raise ValueError("Response support exceeds capacity")
        key = matrix_key(matrix)
        if key != self.key:
            self.invalidate()
        if self.engine is None:
            self.engine = WarpColumns(matrix, **self.settings)
            self.key = key
            self.reference_builds += 1
        elif self._reference_pending:
            # The engine was made ahead: the matrix of this request is its reference.
            self.engine.rebase(matrix)
            self._reference_pending = False
            self.reference_builds += 1
        requested = set(support)
        self._slots = {dof: slot for dof, slot in self._slots.items() if dof in requested}
        self._unread = [dof for dof in self._unread if dof in requested]
        missing = tuple(dof for dof in support if dof not in self._slots)
        if missing:
            used = set(self._slots.values())
            free = (slot for slot in range(self.settings["capacity"]) if slot not in used)
            slots = [next(free) for _ in missing]
            self.engine.solve(missing, slots)
            self._slots.update(zip(missing, slots, strict=True))
            self._unread.extend(missing)
            self.solves += 1
        self.support = support
        return support, missing

    def extend(self, matrix, support):
        """Solve the columns of ``support`` that are not held, ahead of the request that returns them.

        For a caller that builds one support over several calls
        (``AffineGuessManager.nodes_per_attempt``). The columns go into the fixed storage
        of the engine. Nothing is allocated, nothing is read from the device and the
        stream is not awaited: the request that returns these columns reads their residual
        packets and refuses a nonfinite one.

        Returns:
            The number of columns solved.
        """
        return len(self._hold(matrix, support)[1])

    def __call__(self, matrix, support):
        start = time.perf_counter()
        support, _ = self._hold(matrix, support)
        engine, n = self.engine, 3 * matrix.nrow
        # The one device read of a request: the residual packets of the columns solved since
        # the last request, here and in extend(). A nonfinite column is refused before it is returned.
        diagnostics = engine.diagnose(self._unread, [self._slots[dof] for dof in self._unread]) if self._unread else []
        self._unread = []
        for column, dof in enumerate(support):
            wp.copy(engine.ordered, engine.held, dest_offset=column * n, src_offset=self._slots[dof] * n, count=n)
        if matrix.device.is_cuda:
            wp.synchronize_stream(wp.get_stream(matrix.device))
        self.columns = engine.ordered[: n * len(support)]
        return self.columns, {
            "provider": "retained raw A0 approximate Warp CG, numerical guess only",
            "eager_setup_seconds": time.perf_counter() - start,
            "columns": len(support),
            "new_columns": len(diagnostics),
            "reference_builds": self.reference_builds,
            "solve_calls": self.solves,
            "approximate_basis_tolerance": self.settings["tolerance"],
            "basis_maxiter": self.settings["maxiter"],
            "basis_chebyshev_degree": self.settings["degree"],
            "column_diagnostics": diagnostics,
        }

    def invalidate(self):
        if self.engine is not None:
            self.engine.close()
        self.engine = self.key = self.columns = None
        self.support = ()
        self._slots, self._unread = {}, []
        self._reference_pending = False

    def close(self):
        self.invalidate()
