# SPDX-License-Identifier: Apache-2.0
"""Device convergence and safeguarded nonlinear steps; no physical history."""

import warp as wp


@wp.func
def _fem_valid(statistics: wp.array[wp.float64]):
    return (
        statistics[2] == wp.float64(0.0)
        and wp.isfinite(statistics[0])
        and wp.isfinite(statistics[1])
        and statistics[0] >= wp.float64(0.0)
        and statistics[1] >= wp.float64(0.0)
    )


@wp.func
def _fem_converged(statistics: wp.array[wp.float64], relative: wp.float64):
    # True impulse residual, not the recursively updated Krylov residual.
    # No roundoff allowance silently relaxes the requested FEM tolerance.
    # A zero RHS therefore requires an exactly zero measured residual.
    return statistics[0] <= relative * relative * statistics[1]


@wp.func
def _search_merit(
    statistics: wp.array[wp.float64],
    fem_statistics: wp.array[wp.float64],
    weights: wp.array[wp.float64],
    dt: float,
):
    h = wp.float64(dt)
    return weights[0] * fem_statistics[0] / (h * h) + weights[1] * statistics[3] + weights[2] * statistics[4]


@wp.kernel(enable_backward=False)
def finite_vectors(values: wp.array[wp.vec3], valid: wp.array[int]):
    i = wp.tid()
    for j in range(3):
        if not wp.isfinite(values[i][j]):
            wp.atomic_min(valid, 0, 0)


@wp.kernel(enable_backward=False)
def finite_scalars(values: wp.array[float], valid: wp.array[int]):
    if not wp.isfinite(values[wp.tid()]):
        wp.atomic_min(valid, 0, 0)


@wp.kernel(enable_backward=False)
def invalid_trial(valid: wp.array[int], state: wp.array[int], merit: wp.array[wp.float64]):
    if valid[0] == 0:
        state[0] += 1
        state[3] = 0
        merit[0] = wp.float64(1.0e30)
        merit[5] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def initialize(
    state: wp.array[int], merit: wp.array[wp.float64], running: wp.array[int], weights: wp.array[wp.float64]
):
    for i in range(6):
        state[i] = 0
        merit[i] = wp.float64(0.0)
    running[0] = 1
    for i in range(3):
        weights[i] = wp.float64(1.0)


@wp.kernel(enable_backward=False)
def measure(
    stats: wp.array[wp.float64],
    alm: wp.array[float],
    fem_statistics: wp.array[wp.float64],
    fallback: wp.array[int],
    absolute: wp.float64,
    relative: wp.float64,
    normal_tolerance: float,
    fem_tolerance: wp.float64,
    dt: float,
    weights: wp.array[wp.float64],
    state: wp.array[int],
    merit: wp.array[wp.float64],
):
    state[0] += 1
    valid = stats[5] == wp.float64(0.0) and fallback[0] == 0 and _fem_valid(fem_statistics)
    for i in range(5):
        if not wp.isfinite(alm[i]):
            valid = False
    valid = valid and alm[4] == 0.0
    momentum = stats[0] / (absolute + relative * stats[1])
    dual = wp.float64(alm[2]) / (absolute + relative * wp.float64(alm[3]))
    normal = wp.float64(wp.max(alm[0], alm[1]) / normal_tolerance)
    value = wp.max(momentum, wp.max(dual, normal))
    if not wp.isfinite(value):
        valid = False
    # Weights are frozen at begin_search, never recomputed from a proposal's
    # RHS or force scale. Otherwise a growing denominator can fake descent.
    merit[0] = _search_merit(stats, fem_statistics, weights, dt)
    # This proposes certification, not final acceptance. The final affine
    # correction enforces configured FEM accuracy under its applied load.
    state[5] = int(value <= wp.float64(1.0))
    merit[2] = stats[0]
    merit[3] = wp.float64(wp.max(alm[0], alm[1]))
    merit[4] = wp.float64(alm[2])
    merit[5] = wp.float64(int(valid))


@wp.kernel(enable_backward=False)
def decide(
    state: wp.array[int], merit: wp.array[wp.float64], fallback: wp.array[int], limit: int, running: wp.array[int]
):
    state[3] = int(state[4] == 0 and state[0] <= limit and merit[5] != wp.float64(0.0) and state[5] != 0)
    running[0] = int(state[3] == 0 and merit[5] != wp.float64(0.0) and state[0] < limit and state[4] == 0)


@wp.kernel(enable_backward=False)
def begin_search(
    state: wp.array[int],
    merit: wp.array[wp.float64],
    step: wp.array[float],
    searching: wp.array[int],
    statistics: wp.array[wp.float64],
    alm: wp.array[float],
    fem_statistics: wp.array[wp.float64],
    absolute: wp.float64,
    relative: wp.float64,
    fem_tolerance: wp.float64,
    dt: float,
    weights: wp.array[wp.float64],
):
    state[1] += 1
    state[2] = 0
    fem_scale = fem_tolerance * wp.sqrt(fem_statistics[1]) / wp.float64(dt)
    if fem_scale == wp.float64(0.0):
        # A zero RHS still requires exactly zero residual for acceptance.
        # This finite scale only defines the search objective in that case.
        fem_scale = absolute
    rigid_scale = absolute + relative * statistics[1]
    friction_scale = absolute + relative * wp.float64(alm[3])
    weights[0] = wp.float64(1.0) / (fem_scale * fem_scale)
    weights[1] = wp.float64(1.0) / (rigid_scale * rigid_scale)
    weights[2] = wp.float64(1.0) / (friction_scale * friction_scale)
    merit[0] = _search_merit(statistics, fem_statistics, weights, dt)
    merit[1] = merit[0]
    if not wp.isfinite(merit[1]):
        state[4] = 1
    step[0] = 1.0
    searching[0] = int(state[4] == 0)


@wp.kernel(enable_backward=False)
def assess_search(
    state: wp.array[int],
    merit: wp.array[wp.float64],
    step: wp.array[float],
    searching: wp.array[int],
    limit: int,
):
    state[2] += 1
    accepted = merit[5] != wp.float64(0.0) and (
        state[5] != 0 or merit[0] <= (wp.float64(1.0) - wp.float64(0.0001) * wp.float64(step[0])) * merit[1]
    )
    searching[0] = 0
    if not accepted:
        if state[2] < 24 and state[0] < limit:
            step[0] *= 0.5
            searching[0] = 1
        else:
            state[4] = 1


@wp.kernel(enable_backward=False)
def record_correction(counter: wp.array[int]):
    counter[0] += 1


@wp.kernel(enable_backward=False)
def certification(
    statistics: wp.array[wp.float64],
    alm: wp.array[float],
    applied_fem_statistics: wp.array[wp.float64],
    fresh_fem_statistics: wp.array[wp.float64],
    fallback: wp.array[int],
    absolute: wp.float64,
    relative: wp.float64,
    normal_tolerance: float,
    fem_tolerance: wp.float64,
    state: wp.array[int],
    merit: wp.array[wp.float64],
):
    state[0] += 1
    valid = (
        statistics[5] == wp.float64(0.0)
        and fallback[0] == 0
        and alm[4] == 0.0
        and _fem_valid(applied_fem_statistics)
        and _fem_valid(fresh_fem_statistics)
    )
    for i in range(5):
        valid = valid and wp.isfinite(alm[i])
    merit[5] = wp.float64(int(valid))
    valid = valid and statistics[0] <= absolute + relative * wp.max(statistics[1], statistics[2])
    valid = valid and wp.max(alm[0], alm[1]) <= normal_tolerance
    valid = valid and wp.float64(alm[2]) <= absolute + relative * wp.float64(alm[3])
    valid = valid and _fem_converged(applied_fem_statistics, fem_tolerance)
    # Fresh-law L2 accuracy is not the partitioned solver's contract. Its
    # finite and force-unit momentum checks remain authoritative above.
    state[3] = int(valid)
    merit[2] = statistics[0]
    merit[3] = wp.float64(wp.max(alm[0], alm[1]))
    merit[4] = wp.float64(alm[2])


@wp.kernel(enable_backward=False)
def continue_after_certificate(
    state: wp.array[int],
    merit: wp.array[wp.float64],
    fallback: wp.array[int],
    limit: int,
    continuing: wp.array[int],
):
    # limit reserves the final certificate. Resumption itself consumes one
    # force evaluation, so it is allowed only strictly before that limit.
    continuing[0] = int(
        state[3] == 0 and state[4] == 0 and state[0] < limit and merit[5] != wp.float64(0.0) and fallback[0] == 0
    )


@wp.kernel(enable_backward=False)
def result(state: wp.array[int], merit: wp.array[wp.float64], fallback: wp.array[int], out: wp.array[wp.float64]):
    for i in range(11):
        out[i] = wp.float64(0.0)
    out[0] = wp.float64(state[0])
    out[1] = wp.float64(state[1])
    out[3] = wp.float64(state[3])
    out[7] = merit[2]
    out[8] = merit[3]
    out[9] = merit[4]
    out[10] = wp.float64(fallback[0])
