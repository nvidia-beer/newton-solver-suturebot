# SPDX-License-Identifier: Apache-2.0
"""CUDA-only packed BSR3x3 matvec and guarded operator adapter.

The native snippet is a CUDA shuffle intrinsic only. Matrix access and all
arithmetic remain Warp code. Unsupported devices/layouts/gradient paths use
Warp's original bsr_mv. Reduction order differs from the original scalar-row
kernel, so equality of mathematical operators does not imply bitwise equality.
"""

from functools import cache

import warp as wp
from warp._src import context as warp_context
from warp.optim.linear import LinearOperator, aslinearoperator
from warp.sparse import bsr_mv

GROUP_SIZES = (4, 8, 16, 32)
BLOCK_DIM = 256


@wp.func_native(
    """
    #if defined(__CUDA_ARCH__)
        return __shfl_down_sync(0xffffffffu, value, delta, width);
    #else
        // Compilation-only CPU stub. The public dispatcher never launches
        // the packed kernel on CPU; CPU execution uses Warp's bsr_mv.
        return value;
    #endif
    """
)
def _shuffle_down(value: float, delta: int, width: int) -> float: ...


@cache
def packed_kernel(group_size: int):
    """Return one specialization; raw launches require padded complete warps."""
    if type(group_size) is not int or group_size not in GROUP_SIZES:
        raise ValueError(f"group_size must be one of {GROUP_SIZES}")

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def matvec(
        nrow: int,
        offsets: wp.array(dtype=int),
        row_counts: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=wp.mat33),
        x: wp.array(dtype=wp.vec3),
        y: wp.array(dtype=wp.vec3),
        alpha: float,
        beta: float,
    ):
        tid = wp.tid()
        row = tid // wp.static(group_size)
        lane = tid % wp.static(group_size)
        total = wp.vec3(0.0)
        if row < nrow and alpha != 0.0:
            begin = offsets[row]
            end = offsets[row + 1]
            if row_counts:
                end = begin + row_counts[row]
            for block in range(begin + lane, end, wp.static(group_size)):
                total += values[block] * x[columns[block]]

        # All physical threads, including padding and empty rows, participate.
        # The group divides a hardware warp, so shuffles never cross rows.
        for level in range(wp.static(group_size.bit_length() - 1)):
            delta = wp.static(group_size) >> (level + 1)
            total += wp.vec3(
                _shuffle_down(total[0], delta, wp.static(group_size)),
                _shuffle_down(total[1], delta, wp.static(group_size)),
                _shuffle_down(total[2], delta, wp.static(group_size)),
            )
        if row < nrow and lane == 0:
            value = alpha * total
            if beta != 0.0:
                value += beta * y[row]
            y[row] = value

    return matvec


def _overlap(a, b):
    """Byte overlap for contiguous arrays, including offset views."""
    return a.ptr < b.ptr + b.size * wp.types.type_size_in_bytes(
        b.dtype
    ) and b.ptr < a.ptr + a.size * wp.types.type_size_in_bytes(a.dtype)


def packed_supported(A, x, y):
    """Conservative eligibility; no device synchronization or device reads."""
    if y is None or not hasattr(A, "values") or A.block_shape != (3, 3):
        return False
    if not A.values.device.is_cuda or A.values.dtype != wp.mat33:
        return False
    arrays = (A.values, A.offsets, A.columns, x, y)
    if A.row_counts is not None:
        arrays += (A.row_counts,)
    if any(not a.is_contiguous or a.device != A.values.device or a.requires_grad for a in arrays):
        return False
    if warp_context.runtime.tape is not None:
        return False
    return x.dtype == wp.vec3 and y.dtype == wp.vec3 and x.shape == (A.ncol,) and y.shape == (A.nrow,)


def packed_bsr_mv(A, x, y=None, alpha=1.0, beta=0.0, *, group_size=8, work_buffer=None):
    """Compute y = alpha*A*x + beta*y, preserving original fallback semantics.

    On the custom path, overlapping x/y require a snapshot of x. Pass a
    non-overlapping work_buffer with x's shape/dtype/device to avoid allocating
    during graph capture. A zero alpha never reads x; a zero beta never reads y.
    No backward implementation is provided: active tapes or gradient arrays
    dispatch to the unchanged Warp implementation.
    """
    if type(group_size) is not int or group_size not in GROUP_SIZES:
        raise ValueError(f"group_size must be one of {GROUP_SIZES}")
    if not packed_supported(A, x, y):
        return bsr_mv(A, x, y, alpha=alpha, beta=beta, work_buffer=work_buffer, tile_size=-1)

    if alpha != 0.0 and _overlap(x, y):
        if work_buffer is None:
            work_buffer = wp.empty_like(x)
        if (
            work_buffer.shape != x.shape
            or work_buffer.dtype != x.dtype
            or work_buffer.device != x.device
            or not work_buffer.is_contiguous
            or _overlap(work_buffer, x)
            or _overlap(work_buffer, y)
        ):
            raise ValueError("work_buffer must match x and must not overlap x or y")
        wp.copy(work_buffer, x)
        x = work_buffer

    if A.nrow:
        dim = ((A.nrow * group_size + BLOCK_DIM - 1) // BLOCK_DIM) * BLOCK_DIM
        wp.launch(
            packed_kernel(group_size),
            dim=dim,
            block_dim=BLOCK_DIM,
            device=A.values.device,
            inputs=[A.nrow, A.offsets, A.row_counts, A.columns, A.values, x, y, alpha, beta],
        )
    return y


def packed_aslinearoperator(matrix):
    """Use packed16 for supported non-gradient BSR3x3 operators only.

    Closure-owned snapshots make x/z and partial y/z aliasing capture-safe.
    Strong array owners also survive matrix storage replacement while an old
    graph/operator is still retained. Existing FEM workspace/graph signatures
    continue to invalidate on matrix storage and operator identity changes.
    """
    original = aslinearoperator(matrix)
    if (
        matrix.block_shape != (3, 3)
        or matrix.values.dtype != wp.mat33
        or not matrix.values.device.is_cuda
        or matrix.values.requires_grad
        or warp_context.runtime.tape is not None
    ):
        return original
    # Allocation occurs when FixedParticles prepares its derived matrices,
    # outside CUDA capture. The ordinary Krylov path needs neither snapshot.
    x_work = wp.empty(matrix.ncol, dtype=wp.vec3, device=matrix.values.device)
    y_work = wp.empty(matrix.nrow, dtype=wp.vec3, device=matrix.values.device)

    def matvec(x, y, z, alpha, beta):
        if not packed_supported(matrix, x, z) or (
            y.requires_grad
            or y.device != z.device
            or y.dtype != wp.vec3
            or not y.is_contiguous
            or y.shape != (matrix.nrow,)
        ):
            original.matvec(x, y, z, alpha, beta)
            return
        # Preserve x before initializing z from a different affine operand.
        if matrix.nrow == 0:
            return
        if alpha != 0.0 and _overlap(x, z):
            wp.copy(x_work, x)
            x = x_work
        if beta != 0.0 and y.ptr != z.ptr:
            if _overlap(y, z):
                wp.copy(y_work, y)
                y = y_work
            wp.copy(z, y)
        packed_bsr_mv(matrix, x, z, alpha, beta, group_size=16)

    operator = LinearOperator((matrix.nrow, matrix.ncol), wp.vec3, matrix.values.device, matvec)
    operator._packed_bsr_group_size = 16
    operator._packed_bsr_owners = (
        matrix,
        matrix.offsets,
        matrix.row_counts,
        matrix.columns,
        matrix.values,
        x_work,
        y_work,
        original,
    )
    return operator
