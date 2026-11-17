# SPDX-License-Identifier: Apache-2.0
"""Frozen single-thread Anderson decision kernel and a generator of decision states.

``decide_reference`` is the ``_decide`` kernel of ``surface_anderson_graph`` that the staged
kernels of ``surface_anderson_decide`` replaced, kept verbatim. This is its only copy: the
oracle of ``test_surface_anderson_decide`` and the old side of the cost benchmark
``nsys/results/gpu-pipeline-2026-11-13/bench/anderson_decide_cost.py``.

No Warp module option is set here on purpose. The kernel is compiled with the default
floating-point contraction, as it was in production, so on CUDA its results depend on which
products the compiler and the PTX assembler of the driver fuse. The staged kernels pin the
fusion observed with Warp 1.17.0 (CUDA 13 toolkit) and driver 580.126. If the bitwise tests
start to fail on CUDA after a toolchain change while they pass on CPU, this oracle moved, not
the production arithmetic.
"""

import numpy as np
import warp as wp

_M6 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
_V6 = wp.types.vector(length=6, dtype=wp.float64)


@wp.kernel(enable_backward=False)
def decide_reference(
    history: wp.array3d[wp.float64],
    state: wp.array[int],
    order: wp.array[int],
    rows: wp.array[int],
    total: wp.array[int],
    invalid: wp.array[int],
    capacity: int,
    depth: int,
    scalars: wp.array[wp.float64],
    norms: wp.array[wp.float64],
    maxima: wp.array[wp.float64],
    columns: wp.array[int],
    coefficients: wp.array[wp.float64],
    values: wp.array[wp.float64],
    applied: wp.array[int],
    skipped: wp.array[int],
    counters: wp.array[int],
):
    count_rows = total[0]
    counters[6] = wp.max(counters[6], count_rows)
    if invalid[0] != 0 or count_rows > capacity:
        state[0] = 0
        state[1] = 0
        scalars[0] = wp.float64(1e-8)
        if count_rows > capacity:
            counters[3] += 1
        else:
            counters[4] += 1
        return
    # Stable ascending physical row order; no numerical zero threshold.
    for i in range(1, count_rows):
        sorted_row = rows[i]
        j = i - 1
        while j >= 0:
            if rows[j] <= sorted_row:
                break
            rows[j + 1] = rows[j]
            j -= 1
        rows[j + 1] = sorted_row
    current = state[2]
    square = wp.float64(0.0)
    maximum = wp.float64(0.0)
    for i in range(count_rows):
        value = history[current, 1, rows[i]]
        square += value * value
        maximum = wp.max(maximum, wp.abs(value))
    norm = wp.sqrt(square)
    if not wp.isfinite(norm):
        state[0] = 0
        state[1] = 0
        scalars[0] = wp.float64(1e-8)
        counters[4] += 1
        return
    norms[current] = norm
    maxima[current] = maximum
    if state[1] != 0:
        state[1] = 0
        ratio = (scalars[1] - norm) / wp.max(scalars[1] - wp.float64(0.95) * scalars[2], wp.float64(1e-30))
        scalars[4] = ratio
        if ratio < wp.float64(0.1) or maximum > wp.float64(1.05) * scalars[3]:
            state[0] = 0
            state[3] = 2
            scalars[0] = wp.min(scalars[0] * wp.float64(10.0), wp.float64(1e8))
            applied[0] = 1
            skipped[0] = 0
            counters[2] += 1
            return
        if ratio > wp.float64(0.75):
            scalars[0] = wp.max(scalars[0] * wp.float64(0.1), wp.float64(1e-12))
    count = state[0]
    if count == depth + 1:
        for i in range(depth):
            order[i] = order[i + 1]
        count = depth
    order[count] = current
    count += 1
    state[0] = count
    if count < 2 or norm <= wp.float64(1e-30):
        return
    best = order[0]
    for i in range(1, count):
        candidate = order[i]
        if norms[candidate] <= norms[best]:
            best = candidate
    state[8] = best
    ncol = int(0)  # noqa: UP018, RUF046 -- Warp mutable scalar
    for i in range(count):
        if order[i] != best:
            columns[ncol] = order[i]
            ncol += 1
    state[9] = ncol
    scale_squared = wp.float64(0.0)
    for col in range(ncol):
        accumulated = wp.float64(0.0)
        for i in range(count_rows):
            row = rows[i]
            delta = history[columns[col], 1, row] - history[best, 1, row]
            accumulated += delta * delta
        scale_squared = wp.max(scale_squared, accumulated)
    scale = wp.sqrt(scale_squared)
    if not wp.isfinite(scale) or scale <= wp.max(wp.float64(1e-30), norms[best] * wp.float64(1e-12)):
        return
    gram = _M6(wp.float64(0.0))
    eigvec = _M6(wp.float64(0.0))
    rhs = _V6(wp.float64(0.0))
    for j in range(ncol):
        eigvec[j, j] = wp.float64(1.0)
        for i in range(count_rows):
            row = rows[i]
            dj = (history[columns[j], 1, row] - history[best, 1, row]) / scale
            rhs[j] -= dj * (history[best, 1, row] / scale)
            for k in range(j, ncol):
                dk = (history[columns[k], 1, row] - history[best, 1, row]) / scale
                gram[j, k] += dj * dk
        for k in range(j + 1, ncol):
            gram[k, j] = gram[j, k]
    magnitude = wp.float64(0.0)
    for j in range(ncol):
        magnitude = wp.max(magnitude, wp.abs(gram[j, j]))
    # Serial cyclic Jacobi on the small FP64 Gram, distinct from thin SVD.
    for sweep in range(24):  # noqa: B007 - fixed Jacobi sweep budget
        for p in range(ncol):
            for q in range(p + 1, ncol):
                off = gram[p, q]
                if wp.abs(off) <= wp.float64(1e-16) * magnitude:
                    continue
                tau = (gram[q, q] - gram[p, p]) / (wp.float64(2.0) * off)
                tangent = wp.float64(1.0) / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
                if tau < wp.float64(0.0):
                    tangent = -tangent
                cosine = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + tangent * tangent)
                sine = tangent * cosine
                app, aqq = gram[p, p], gram[q, q]
                for k in range(ncol):
                    if k != p and k != q:
                        apk, aqk = gram[p, k], gram[q, k]
                        gram[p, k] = cosine * apk - sine * aqk
                        gram[k, p] = gram[p, k]
                        gram[q, k] = sine * apk + cosine * aqk
                        gram[k, q] = gram[q, k]
                    vkp, vkq = eigvec[k, p], eigvec[k, q]
                    eigvec[k, p] = cosine * vkp - sine * vkq
                    eigvec[k, q] = sine * vkp + cosine * vkq
                gram[p, p] = app - tangent * off
                gram[q, q] = aqq + tangent * off
                gram[p, q] = wp.float64(0.0)
                gram[q, p] = wp.float64(0.0)
    largest = wp.float64(0.0)
    for j in range(ncol):
        if not wp.isfinite(gram[j, j]):
            return
        largest = wp.max(largest, gram[j, j])
    cutoff = wp.max(largest * wp.float64(1e-8), wp.float64(1e-28))
    ridge = scalars[0] * (norms[best] / scale) * (norms[best] / scale)
    solution = _V6(wp.float64(0.0))
    rank = int(0)  # noqa: UP018, RUF046 -- Warp mutable scalar
    for j in range(ncol):
        # A materially indefinite Gram indicates unresolved arithmetic; decline.
        if gram[j, j] < -wp.float64(1e-12) * wp.max(largest, wp.float64(1e-30)):
            return
        if gram[j, j] > cutoff:
            rank += 1
            projection = wp.float64(0.0)
            for k in range(ncol):
                projection += eigvec[k, j] * rhs[k]
            projection /= gram[j, j] + ridge
            for k in range(ncol):
                solution[k] += eigvec[k, j] * projection
    counters[7] += ncol - rank
    if rank == 0:
        return
    l1 = wp.float64(0.0)
    for j in range(ncol):
        if not wp.isfinite(solution[j]):
            return
        l1 += wp.abs(solution[j])
    if l1 > wp.float64(64.0):
        return
    predicted_square = wp.float64(0.0)
    step_max, load_max = wp.float64(0.0), wp.float64(0.0)
    for i in range(count_rows):
        row = rows[i]
        predicted = history[best, 1, row]
        proposal = history[best, 0, row] + history[best, 2, row]
        for j in range(ncol):
            past = columns[j]
            predicted += (history[past, 1, row] - history[best, 1, row]) * solution[j]
            proposal += (
                (history[past, 0, row] - history[best, 0, row]) + (history[past, 2, row] - history[best, 2, row])
            ) * solution[j]
        if not wp.isfinite(predicted) or not wp.isfinite(proposal):
            return
        predicted_square += predicted * predicted
        step_max = wp.max(step_max, wp.abs(proposal - history[best, 0, row]))
        load_max = wp.max(load_max, wp.abs(history[best, 0, row]))
    predicted_norm = wp.sqrt(predicted_square)
    step_scale = wp.max(wp.max(load_max, maxima[best]), wp.float64(1e-30))
    if (
        not wp.isfinite(predicted_norm)
        or predicted_norm >= wp.float64(0.95) * norms[best]
        or step_max > wp.float64(2.0) * step_scale
    ):
        return
    sum_norm, sum_max = wp.float64(0.0), wp.float64(0.0)
    for j in range(ncol):
        coefficients[j] = solution[j]
        sum_norm += norms[columns[j]]
        sum_max += maxima[columns[j]]
    scalars[1] = wp.float64(0.55) * norms[best] + wp.float64(0.45) * sum_norm / wp.float64(ncol)
    scalars[2] = predicted_norm
    scalars[3] = wp.float64(0.55) * maxima[best] + wp.float64(0.45) * sum_max / wp.float64(ncol)
    scalars[5] = values[0]
    state[1] = 1
    state[3] = 1
    applied[0] = 1
    skipped[0] = 0
    counters[1] += 1


def enqueue_reference(aa, values):
    """Launch the frozen kernel on the arrays of a ``CapturedSurfaceAnderson``."""
    wp.launch(
        decide_reference,
        1,
        [
            aa.history,
            aa.state,
            aa.order,
            aa.rows,
            aa.total,
            aa.invalid,
            aa.capacity,
            aa.depth,
            aa.scalars,
            aa.norms,
            aa.maxima,
            aa.columns,
            aa.coefficients,
            values,
            aa.applied,
            aa.not_applied,
            aa.counters,
        ],
        device=aa.device,
    )


def populate(
    aa,
    *,
    count,
    rows,
    rank,
    seed,
    pending=0,
    magnitude=1e-3,
    ridge=1e-8,
    stale=1.0,
    invalid=0,
    ascending=False,
):
    """Write the state that ``_prepare`` and ``_pack`` leave before a decision.

    ``count`` iterates of a relaxed affine map of numerical rank ``rank`` are retained in
    randomly chosen slots, one more is the current slot, and ``rows`` active rows arrive in
    random order (``rows`` above the capacity is the overflow case). ``pending`` selects a
    measured-rejection regime of a previous proposal: 0 none, 1 residual rejection,
    2 neutral, 3 ridge decrease, 4 component rejection. Unused slots hold ``stale`` values.
    """
    rng = np.random.default_rng(seed)
    size, slots = aa.size, aa.history.shape[0]
    active = rng.choice(size, rows, replace=False)
    if ascending:
        active = np.sort(active)
    used = min(rank, rows)
    basis = np.linalg.qr(rng.normal(size=(rows, max(used, 1))))[0][:, :used]
    gains = rng.uniform(0.4, 1.8, used)
    target = rng.normal(size=used)
    history = np.full((slots, 3, size), stale, dtype=np.float64)
    history[:, :, active] = stale * rng.normal(size=(slots, 3, rows))
    placement = rng.permutation(slots)
    visits = rng.permutation(count + 1)
    coordinates = np.zeros(used)
    norms, maxima = np.zeros(slots), np.zeros(slots)
    for visit in range(count + 1):
        slot = placement[visits[visit]]
        residual = magnitude * (basis @ (gains * (target - coordinates)))
        history[slot] = 0.0
        history[slot, 0, active] = magnitude * (basis @ coordinates)
        history[slot, 1, active] = residual
        history[slot, 2, active] = 0.4 * residual
        with np.errstate(over="ignore"):
            norms[slot] = np.linalg.norm(residual)
        maxima[slot] = np.max(np.abs(residual), initial=0.0)
        coordinates = coordinates + 0.12 * gains * (target - coordinates) + 0.02 * rng.normal(size=used)
    current = placement[count]
    measured = norms[current]
    # Reference norm for the wanted reduction ratio with a predicted norm of half the measured one.
    ratio = (0.5, 0.05, 0.5, 0.9, 0.5)[pending]
    reference = measured * (1.0 - 0.475 * ratio) / (1.0 - ratio)
    component = maxima[current] * (0.5 if pending == 4 else 4.0)
    norms[current], maxima[current] = 0.0, 0.0
    aa.history.assign(history)
    aa.state.assign(np.array([count, int(pending != 0), current, 0, 0, 0, 1, 1, 0, 0], dtype=np.int32))
    aa.order.assign(np.r_[placement[:count], np.zeros(aa.order.size - count)].astype(np.int32))
    stored = active[: aa.capacity]
    aa.rows.assign(np.r_[stored, np.zeros(aa.capacity - stored.size)].astype(np.int32))
    aa.total.assign(np.array([rows], dtype=np.int32))
    aa.invalid.assign(np.array([invalid], dtype=np.int32))
    aa.norms.assign(norms)
    aa.maxima.assign(maxima)
    aa.scalars.assign(np.array([ridge, reference, 0.5 * measured, component, 0.0, 0.3], dtype=np.float64))
    aa.columns.zero_()
    aa.coefficients.zero_()
    aa.applied.zero_()
    aa.not_applied.fill_(1)
    aa.counters.zero_()
