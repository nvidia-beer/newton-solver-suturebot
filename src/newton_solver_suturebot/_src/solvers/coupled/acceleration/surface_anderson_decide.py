# SPDX-License-Identifier: Apache-2.0
"""Staged decision kernels of the captured Anderson variant.

The decision used to be one thread that walked every active row for every sum.
These kernels compute the same quantities with the same arithmetic: each
accumulated sum keeps its ascending row order in a thread of its own, and all
work that is independent between rows (ordering, scaling, prediction) runs one
thread per row. Policy, thresholds, rank cutoff, regularisation and measured
rejection are unchanged; ``CapturedSurfaceAnderson`` owns every array.

Launch order: ``_decide_rows``, ``_decide_history``, ``_decide_scale``,
``_decide_gram``, ``_decide_solve``, ``_decide_accept``. Each kernel reads only
what an earlier kernel wrote, or what no thread of its own launch writes.
"""

import warp as wp

# The single-thread kernel was compiled with floating-point contraction. On CUDA
# the compiler and the PTX assembler fused a product into the following sum or
# difference wherever the shape of the code allowed, so restructured loops would
# be contracted differently. Contraction is off for this module and every
# multiply-add the CUDA build of that kernel executed (read from its machine
# code) is written as `_fused`; its CPU build fused nothing. The staged kernels
# therefore reproduce its results bit for bit on both devices.
wp.set_module_options({"fuse_fp": False})

_M6 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
_V6 = wp.types.vector(length=6, dtype=wp.float64)

# Offsets into the scalar scratch `sums`: Gram entry (j, k) at j * 6 + k.
GRAM = 0
RHS = GRAM + 36
SCALE = RHS + 6
SOLUTION = SCALE + 1
SUMS = SOLUTION + 6
# Cells of `peaks`: exact maxima over the active rows, gathered as bit patterns.
RESIDUAL = 0
STEP = 1
LOAD = 2
PEAKS = 3
# Flags of `stage`: history ready, Gram ready, solution ready, nonfinite prediction, retained iterates on entry.
RETAINED = 4
STAGES = 5
# Rows of the per-row scratch `terms`: six scaled differences, then the negated scaled best residual.
REFERENCE = 6
TERMS = REFERENCE + 1


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return fma(a, b, c);
#else
    double product = a * b;
    return product + c;
#endif
""")
def _fused(a: wp.float64, b: wp.float64, c: wp.float64) -> wp.float64: ...


# A maximum of magnitudes needs no order, and for finite values it is the
# maximum of their bit patterns, which is a single native atomic instead of a
# compare-and-swap loop per contending thread.
@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __double_as_longlong(fabs(value));
#else
    union { double value; long long bits; } pattern;
    pattern.value = fabs(value);
    return pattern.bits;
#endif
""")
def _magnitude_bits(value: wp.float64) -> wp.int64: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(bits);
#else
    union { double value; long long bits; } pattern;
    pattern.bits = bits;
    return pattern.value;
#endif
""")
def _magnitude(bits: wp.int64) -> wp.float64: ...


@wp.kernel(enable_backward=False)
def _decide_rows(
    history: wp.array3d[wp.float64],
    state: wp.array[int],
    order: wp.array[int],
    rows: wp.array[int],
    total: wp.array[int],
    invalid: wp.array[int],
    capacity: int,
    ordered: wp.array[int],
    packed: wp.array2d[wp.float64],
    peaks: wp.array[wp.int64],
    stage: wp.array[int],
):
    position = wp.tid()
    count_rows = total[0]
    retained = state[0]
    if position == 0:
        # `_decide_history` updates the count while its slot-pair threads run.
        stage[RETAINED] = retained
    if invalid[0] != 0 or count_rows > capacity or position >= count_rows:
        return
    # Stable ascending physical row order; no numerical zero threshold.
    row = rows[position]
    rank = int(0)  # noqa: UP018, RUF046 -- Warp mutable scalar
    for i in range(count_rows):
        other = rows[i]
        if other < row or (other == row and i < position):
            rank += 1
    ordered[rank] = row
    # Residuals of the current and the retained iterates, contiguous in row order.
    for index in range(retained):
        slot = order[index]
        packed[slot, rank] = history[slot, 1, row]
    current = state[2]
    value = history[current, 1, row]
    packed[current, rank] = value
    # `_decide_history` takes this maximum and clears the cell.
    wp.atomic_max(peaks, RESIDUAL, _magnitude_bits(value))


@wp.kernel(enable_backward=False)
def _decide_history(
    packed: wp.array2d[wp.float64],
    state: wp.array[int],
    order: wp.array[int],
    total: wp.array[int],
    invalid: wp.array[int],
    capacity: int,
    depth: int,
    scalars: wp.array[wp.float64],
    norms: wp.array[wp.float64],
    maxima: wp.array[wp.float64],
    columns: wp.array[int],
    applied: wp.array[int],
    skipped: wp.array[int],
    counters: wp.array[int],
    stage: wp.array[int],
    distances: wp.array2d[wp.float64],
    peaks: wp.array[wp.int64],
):
    first, second = wp.tid()
    count_rows = total[0]
    slots = packed.shape[0]
    if first < slots:
        # Every ordered slot pair: the best iterate is only known once the
        # serial thread below has measured the current residual. A first
        # iterate cannot propose and needs no distance.
        if first == second or stage[RETAINED] == 0 or invalid[0] != 0 or count_rows > capacity:
            return
        accumulated = wp.float64(0.0)
        for i in range(count_rows):
            delta = packed[first, i] - packed[second, i]
            accumulated = _fused(delta, delta, accumulated)
        distances[first, second] = accumulated
        return
    # First thread of the row after the slot pairs. With eight slots it starts a
    # warp of its own and does not wait for the pair loops.
    if second != 0:
        return
    counters[6] = wp.max(counters[6], count_rows)
    for index in range(RETAINED):
        stage[index] = 0
    peaks[STEP] = wp.int64(0)
    peaks[LOAD] = wp.int64(0)
    if invalid[0] != 0 or count_rows > capacity:
        state[0] = 0
        state[1] = 0
        scalars[0] = wp.float64(1e-8)
        if count_rows > capacity:
            counters[3] += 1
        else:
            counters[4] += 1
        return
    current = state[2]
    maximum = _magnitude(peaks[RESIDUAL])
    peaks[RESIDUAL] = wp.int64(0)
    square = wp.float64(0.0)
    for i in range(count_rows):
        value = packed[current, i]
        square = _fused(value, value, square)
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
        ratio = (scalars[1] - norm) / wp.max(_fused(scalars[2], wp.float64(-0.95), scalars[1]), wp.float64(1e-30))
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
    stage[0] = 1


@wp.kernel(enable_backward=False)
def _decide_scale(
    packed: wp.array2d[wp.float64],
    state: wp.array[int],
    total: wp.array[int],
    columns: wp.array[int],
    norms: wp.array[wp.float64],
    stage: wp.array[int],
    distances: wp.array2d[wp.float64],
    sums: wp.array[wp.float64],
    terms: wp.array2d[wp.float64],
):
    # One thread per term and row: the divisions by the scale do not wait for each other.
    term, i = wp.tid()
    if stage[0] == 0:
        return
    best = state[8]
    ncol = state[9]
    scale_squared = wp.float64(0.0)
    for j in range(6):
        if j < ncol:
            scale_squared = wp.max(scale_squared, distances[columns[j], best])
    scale = wp.sqrt(scale_squared)
    if not wp.isfinite(scale) or scale <= wp.max(wp.float64(1e-30), norms[best] * wp.float64(1e-12)):
        return
    if term == 0 and i == 0:
        sums[SCALE] = scale
        stage[1] = 1
    if i >= total[0]:
        return
    reference = packed[best, i]
    if term == REFERENCE:
        terms[term, i] = -(reference / scale)
    elif term < ncol:
        terms[term, i] = (packed[columns[term], i] - reference) / scale


@wp.kernel(enable_backward=False)
def _decide_gram(
    state: wp.array[int],
    total: wp.array[int],
    stage: wp.array[int],
    terms: wp.array2d[wp.float64],
    sums: wp.array[wp.float64],
):
    # One thread per accumulated sum, each in the ascending row order: Gram
    # entry (j, k), and with the negated reference as row 6 the right-hand side
    # (6, k). Every thread runs the same loop, so none waits for another.
    j, k = wp.tid()
    if stage[1] == 0:
        return
    if k >= state[9] or (j < REFERENCE and k < j):
        return
    accumulated = wp.float64(0.0)
    for i in range(total[0]):
        accumulated = _fused(terms[j, i], terms[k, i], accumulated)
    sums[j * 6 + k] = accumulated


@wp.kernel(enable_backward=False)
def _decide_solve(
    history: wp.array3d[wp.float64],
    packed: wp.array2d[wp.float64],
    state: wp.array[int],
    ordered: wp.array[int],
    total: wp.array[int],
    columns: wp.array[int],
    scalars: wp.array[wp.float64],
    norms: wp.array[wp.float64],
    counters: wp.array[int],
    stage: wp.array[int],
    sums: wp.array[wp.float64],
    predictions: wp.array[wp.float64],
    peaks: wp.array[wp.int64],
):
    position = wp.tid()
    count_rows = total[0]
    if stage[1] == 0 or (position > 0 and position >= count_rows):
        return
    best = state[8]
    ncol = state[9]
    scale = sums[SCALE]
    # Constant six-column loops keep the 6x6 matrices in registers. Every row
    # thread repeats the same deterministic solve; thread 0 owns the outputs.
    gram = _M6(wp.float64(0.0))
    eigvec = _M6(wp.float64(0.0))
    rhs = _V6(wp.float64(0.0))
    for j in range(6):
        if j < ncol:
            eigvec[j, j] = wp.float64(1.0)
            rhs[j] = sums[RHS + j]
            for k in range(6):
                if k >= j and k < ncol:
                    gram[j, k] = sums[j * 6 + k]
                    gram[k, j] = gram[j, k]
    magnitude = wp.float64(0.0)
    for j in range(6):
        if j < ncol:
            magnitude = wp.max(magnitude, wp.abs(gram[j, j]))
    # Serial cyclic Jacobi on the small FP64 Gram, distinct from thin SVD. A
    # sweep without a rotation leaves the matrix as it is, and so would every
    # later sweep of the fixed budget.
    sweep = int(0)  # noqa: UP018, RUF046 -- Warp mutable scalar
    rotated = int(0)  # noqa: UP018, RUF046 -- Warp mutable scalar
    if ncol > 1:
        rotated = 1
    while sweep < 24 and rotated != 0:
        sweep += 1
        rotated = 0
        for p in range(6):
            for q in range(6):
                if q > p and q < ncol:
                    off = gram[p, q]
                    if not (wp.abs(off) <= wp.float64(1e-16) * magnitude):
                        rotated = 1
                        tau = (gram[q, q] - gram[p, p]) / (wp.float64(2.0) * off)
                        tangent = wp.float64(1.0) / (wp.abs(tau) + wp.sqrt(_fused(tau, tau, wp.float64(1.0))))
                        if tau < wp.float64(0.0):
                            tangent = -tangent
                        cosine = wp.float64(1.0) / wp.sqrt(_fused(tangent, tangent, wp.float64(1.0)))
                        sine = tangent * cosine
                        app, aqq = gram[p, p], gram[q, q]
                        for k in range(6):
                            if k < ncol:
                                if k != p and k != q:
                                    apk, aqk = gram[p, k], gram[q, k]
                                    gram[p, k] = _fused(cosine, apk, -(sine * aqk))
                                    gram[k, p] = gram[p, k]
                                    gram[q, k] = _fused(sine, apk, cosine * aqk)
                                    gram[k, q] = gram[q, k]
                                vkp, vkq = eigvec[k, p], eigvec[k, q]
                                eigvec[k, p] = _fused(cosine, vkp, -(sine * vkq))
                                eigvec[k, q] = _fused(sine, vkp, cosine * vkq)
                        gram[p, p] = _fused(-off, tangent, app)
                        gram[q, q] = _fused(off, tangent, aqq)
                        gram[p, q] = wp.float64(0.0)
                        gram[q, p] = wp.float64(0.0)
    largest = wp.float64(0.0)
    for j in range(6):
        if j < ncol:
            if not wp.isfinite(gram[j, j]):
                return
            largest = wp.max(largest, gram[j, j])
    cutoff = wp.max(largest * wp.float64(1e-8), wp.float64(1e-28))
    ratio = norms[best] / scale
    ridge = scalars[0] * ratio * ratio
    solution = _V6(wp.float64(0.0))
    rank = int(0)  # noqa: UP018, RUF046 -- Warp mutable scalar
    for j in range(6):
        if j < ncol:
            # A materially indefinite Gram indicates unresolved arithmetic; decline.
            if gram[j, j] < -wp.float64(1e-12) * wp.max(largest, wp.float64(1e-30)):
                return
            if gram[j, j] > cutoff:
                rank += 1
                projection = wp.float64(0.0)
                for k in range(6):
                    if k < ncol:
                        projection = _fused(eigvec[k, j], rhs[k], projection)
                projection /= gram[j, j] + ridge
                for k in range(6):
                    if k < ncol:
                        solution[k] = _fused(eigvec[k, j], projection, solution[k])
    if position == 0:
        counters[7] += ncol - rank
    if rank == 0:
        return
    l1 = wp.float64(0.0)
    for j in range(6):
        if j < ncol:
            if not wp.isfinite(solution[j]):
                return
            l1 += wp.abs(solution[j])
    if l1 > wp.float64(64.0):
        return
    if position == 0:
        for j in range(6):
            if j < ncol:
                sums[SOLUTION + j] = solution[j]
        stage[2] = 1
    if position >= count_rows:
        return
    row = ordered[position]
    reference = packed[best, position]
    load = history[best, 0, row]
    direction = history[best, 2, row]
    predicted = reference
    proposal = load + direction
    for j in range(6):
        if j < ncol:
            past = columns[j]
            predicted = _fused(packed[past, position] - reference, solution[j], predicted)
            proposal = _fused(
                (history[past, 0, row] - load) + (history[past, 2, row] - direction), solution[j], proposal
            )
    if not wp.isfinite(predicted) or not wp.isfinite(proposal):
        stage[3] = 1
        return
    predictions[position] = predicted
    wp.atomic_max(peaks, STEP, _magnitude_bits(proposal - load))
    wp.atomic_max(peaks, LOAD, _magnitude_bits(load))


@wp.kernel(enable_backward=False)
def _decide_accept(
    state: wp.array[int],
    total: wp.array[int],
    columns: wp.array[int],
    scalars: wp.array[wp.float64],
    norms: wp.array[wp.float64],
    maxima: wp.array[wp.float64],
    coefficients: wp.array[wp.float64],
    values: wp.array[wp.float64],
    applied: wp.array[int],
    skipped: wp.array[int],
    counters: wp.array[int],
    stage: wp.array[int],
    sums: wp.array[wp.float64],
    predictions: wp.array[wp.float64],
    peaks: wp.array[wp.int64],
):
    if stage[2] == 0 or stage[3] != 0:
        return
    count_rows = total[0]
    best = state[8]
    ncol = state[9]
    predicted_square = wp.float64(0.0)
    for i in range(count_rows):
        predicted = predictions[i]
        predicted_square = _fused(predicted, predicted, predicted_square)
    predicted_norm = wp.sqrt(predicted_square)
    step_scale = wp.max(wp.max(_magnitude(peaks[LOAD]), maxima[best]), wp.float64(1e-30))
    if (
        not wp.isfinite(predicted_norm)
        or predicted_norm >= wp.float64(0.95) * norms[best]
        or _magnitude(peaks[STEP]) > wp.float64(2.0) * step_scale
    ):
        return
    sum_norm, sum_max = wp.float64(0.0), wp.float64(0.0)
    for j in range(ncol):
        coefficients[j] = sums[SOLUTION + j]
        sum_norm += norms[columns[j]]
        sum_max += maxima[columns[j]]
    scalars[1] = _fused(norms[best], wp.float64(0.55), wp.float64(0.45) * sum_norm / wp.float64(ncol))
    scalars[2] = predicted_norm
    scalars[3] = _fused(maxima[best], wp.float64(0.55), wp.float64(0.45) * sum_max / wp.float64(ncol))
    scalars[5] = values[0]
    state[1] = 1
    state[3] = 1
    applied[0] = 1
    skipped[0] = 0
    counters[1] += 1


def enqueue_decision(aa, values):
    """Enqueue the decision on the arrays of a ``CapturedSurfaceAnderson``; no readback, no allocation."""
    slots = aa.history.shape[0]
    wp.launch(
        _decide_rows,
        aa.capacity,
        [
            aa.history,
            aa.state,
            aa.order,
            aa.rows,
            aa.total,
            aa.invalid,
            aa.capacity,
            aa.ordered,
            aa.packed,
            aa.peaks,
            aa.stage,
        ],
        device=aa.device,
    )
    wp.launch(
        _decide_history,
        (slots + 1, slots),
        [
            aa.packed,
            aa.state,
            aa.order,
            aa.total,
            aa.invalid,
            aa.capacity,
            aa.depth,
            aa.scalars,
            aa.norms,
            aa.maxima,
            aa.columns,
            aa.applied,
            aa.not_applied,
            aa.counters,
            aa.stage,
            aa.distances,
            aa.peaks,
        ],
        device=aa.device,
    )
    wp.launch(
        _decide_scale,
        (TERMS, aa.capacity),
        [aa.packed, aa.state, aa.total, aa.columns, aa.norms, aa.stage, aa.distances, aa.sums, aa.terms],
        device=aa.device,
    )
    wp.launch(_decide_gram, (TERMS, 6), [aa.state, aa.total, aa.stage, aa.terms, aa.sums], device=aa.device)
    wp.launch(
        _decide_solve,
        aa.capacity,
        [
            aa.history,
            aa.packed,
            aa.state,
            aa.ordered,
            aa.total,
            aa.columns,
            aa.scalars,
            aa.norms,
            aa.counters,
            aa.stage,
            aa.sums,
            aa.predictions,
            aa.peaks,
        ],
        device=aa.device,
    )
    wp.launch(
        _decide_accept,
        1,
        [
            aa.state,
            aa.total,
            aa.columns,
            aa.scalars,
            aa.norms,
            aa.maxima,
            aa.coefficients,
            values,
            aa.applied,
            aa.not_applied,
            aa.counters,
            aa.stage,
            aa.sums,
            aa.predictions,
            aa.peaks,
        ],
        device=aa.device,
    )
