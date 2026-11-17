# SPDX-License-Identifier: Apache-2.0
"""Captured FP64 small-Gram Anderson variant with fresh physical safeguards.

The existing CPU thin SVD remains the oracle, not an arithmetic-equivalence
claim. Squared singular cutoff and finite/conditioning guards decline uncertain
normal-equation solves. Caller physical convergence and rollback are unchanged.
"""

import warp as wp
import warp._src.context as warp_context

from . import surface_anderson_decide as _decision


@wp.kernel(enable_backward=False)
def _reset(
    state: wp.array[int],
    scalars: wp.array[wp.float64],
    applied: wp.array[int],
    skipped: wp.array[int],
):
    for index in range(10):
        state[index] = 0
    scalars[0] = wp.float64(1e-8)
    applied[0] = 0
    skipped[0] = 1


@wp.kernel(enable_backward=False)
def _prepare(
    state: wp.array[int],
    scalars: wp.array[wp.float64],
    order: wp.array[int],
    iteration: wp.array[int],
    values: wp.array[wp.float64],
    total: wp.array[int],
    invalid: wp.array[int],
    applied: wp.array[int],
    skipped: wp.array[int],
    counters: wp.array[int],
):
    # [count,pending,current,action,epoch,tolerance,last_iteration,initialized,best,ncolumns]
    if state[7] == 0 or state[4] != iteration[1] or state[5] != iteration[2] or iteration[0] <= state[6]:
        state[0] = 0
        state[1] = 0
        scalars[0] = wp.float64(1e-8)
        counters[5] += 1
    state[4] = iteration[1]
    state[5] = iteration[2]
    state[6] = iteration[0]
    state[7] = 1
    state[3] = 0
    free = int(0)  # noqa: UP018, RUF046 -- Warp mutable scalar
    for candidate in range(8):
        used = bool(False)  # noqa: UP018 -- Warp mutable scalar
        for index in range(state[0]):
            if order[index] == candidate:
                used = True
        if not used:
            free = candidate
            break
    state[2] = free
    total[0] = 0
    invalid[0] = 0
    if not wp.isfinite(values[0]):
        invalid[0] = 1
    applied[0] = 0
    skipped[0] = 1
    counters[0] += 1


@wp.kernel(enable_backward=False)
def _pack(
    load: wp.array[wp.float64],
    residual: wp.array[wp.float64],
    direction: wp.array[wp.float64],
    history: wp.array3d[wp.float64],
    state: wp.array[int],
    order: wp.array[int],
    total: wp.array[int],
    invalid: wp.array[int],
    rows: wp.array[int],
    capacity: int,
):
    row = wp.tid()
    a, b, c = load[row], residual[row], direction[row]
    current = state[2]
    history[current, 0, row] = a
    history[current, 1, row] = b
    history[current, 2, row] = c
    if not wp.isfinite(a) or not wp.isfinite(b) or not wp.isfinite(c):
        wp.atomic_max(invalid, 0, 1)
    active = a != wp.float64(0.0) or b != wp.float64(0.0) or c != wp.float64(0.0)
    for index in range(state[0]):
        past = order[index]
        for field in range(3):
            active = active or history[past, field, row] != wp.float64(0.0)
    if active:
        destination = wp.atomic_add(total, 0, 1)
        if destination < capacity:
            rows[destination] = row


@wp.kernel(enable_backward=False)
def _write(
    load: wp.array[wp.float64],
    fallback: wp.array[wp.float64],
    history: wp.array3d[wp.float64],
    state: wp.array[int],
    scalars: wp.array[wp.float64],
    columns: wp.array[int],
    coefficients: wp.array[wp.float64],
):
    row = wp.tid()
    if state[3] == 2:
        load[row] = fallback[row]
    elif state[3] == 1:
        best = state[8]
        proposal = history[best, 0, row] + history[best, 2, row]
        for j in range(state[9]):
            past = columns[j]
            proposal += (
                (history[past, 0, row] - history[best, 0, row]) + (history[past, 2, row] - history[best, 2, row])
            ) * coefficients[j]
        fallback[row] = history[best, 0, row] + scalars[5] * history[best, 2, row]
        load[row] = proposal


class CapturedSurfaceAnderson:
    """Owned GPU history and scalar state; every public method only enqueues work.

    ``applied`` means only load was replaced. The caller owns previous residuals,
    relaxation and physical state. Exact active-row overflow returns not_applied,
    clears numerical history and never truncates a physical load. All arrays must
    be retained with the graph; no allocation occurs after construction.
    """

    def __init__(self, template, depth=6, *, capacity=256):
        if template.dtype != wp.float64 or template.ndim != 1 or not template.is_contiguous or template.requires_grad:
            raise ValueError("Captured Anderson requires a contiguous non-gradient FP64 vector")
        if not 1 <= depth <= 6 or capacity < 1:
            raise ValueError("Captured Anderson requires depth1..6 and positive capacity")
        if warp_context.runtime.tape is not None:
            raise ValueError("Captured Anderson is not differentiable")
        if template.device.is_capturing:
            raise ValueError("Prepare captured Anderson outside capture")
        self.device, self.size, self.depth = template.device, template.size, int(depth)
        self.capacity = min(int(capacity), max(1, self.size))

        def zeros(shape, dtype=wp.float64):
            return wp.zeros(shape, dtype=dtype, device=self.device)

        self.history = zeros((8, 3, self.size))
        self.fallback = zeros(self.size)
        self.state = zeros(10, int)
        self.scalars = zeros(6)
        self.order = zeros(7, int)
        self.rows = zeros(self.capacity, int)
        self.total = zeros(1, int)
        self.invalid = zeros(1, int)
        self.norms, self.maxima = zeros(8), zeros(8)
        self.columns, self.coefficients = zeros(6, int), zeros(6)
        self.applied, self.not_applied = zeros(1, int), zeros(1, int)
        self.counters = zeros(8, int)
        # Private scratch of the staged decision kernels.
        self.ordered = zeros(self.capacity, int)
        self.packed = zeros((self.history.shape[0], self.capacity))
        self.peaks = zeros(_decision.PEAKS, wp.int64)
        self.distances = zeros((self.history.shape[0], self.history.shape[0]))
        self.terms = zeros((_decision.TERMS, self.capacity))
        self.stage = zeros(_decision.STAGES, int)
        self.sums = zeros(_decision.SUMS)
        self.predictions = zeros(self.capacity)
        self.owners = tuple(value for value in vars(self).values() if isinstance(value, wp.array))
        self.reset_device()

    def reset_device(self):
        wp.launch(
            _reset,
            1,
            [self.state, self.scalars, self.applied, self.not_applied],
            device=self.device,
        )

    def propose_device(self, load, residual, values, direction, iteration_state):
        if warp_context.runtime.tape is not None:
            raise RuntimeError("Captured Anderson cannot run under a tape")
        for array in (load, residual, direction):
            if (
                array.dtype != wp.float64
                or array.shape != (self.size,)
                or array.device != self.device
                or array.requires_grad
                or not array.is_contiguous
            ):
                raise ValueError("Captured Anderson vector binding changed")
        if values.dtype != wp.float64 or values.size < 1 or values.device != self.device or values.requires_grad:
            raise ValueError("Captured Anderson values must be non-gradient device FP64")
        if iteration_state.dtype != wp.int32 or iteration_state.size < 3 or iteration_state.device != self.device:
            raise ValueError("Captured Anderson iteration state must be device int32")
        wp.launch(
            _prepare,
            1,
            [
                self.state,
                self.scalars,
                self.order,
                iteration_state,
                values,
                self.total,
                self.invalid,
                self.applied,
                self.not_applied,
                self.counters,
            ],
            device=self.device,
        )
        wp.launch(
            _pack,
            self.size,
            [
                load,
                residual,
                direction,
                self.history,
                self.state,
                self.order,
                self.total,
                self.invalid,
                self.rows,
                self.capacity,
            ],
            device=self.device,
        )
        _decision.enqueue_decision(self, values)
        wp.launch(
            _write,
            self.size,
            [
                load,
                self.fallback,
                self.history,
                self.state,
                self.scalars,
                self.columns,
                self.coefficients,
            ],
            device=self.device,
        )
        return self.applied, self.not_applied
