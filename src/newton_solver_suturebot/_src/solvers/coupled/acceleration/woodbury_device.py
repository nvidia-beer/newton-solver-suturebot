# SPDX-License-Identifier: Apache-2.0
"""Device Schur coefficients for the guarded shaft-PC guess; no host packet.

The proposal is the one ``host_coefficients`` solves. With the stored FEM
response Z on track support (tissue compliance P Z P.T), the rigid response
G R^-1 G.T and the frozen anchor tangents D, the contact weights w solve
(I + h D C) w = D (G R^-1 r_rigid - P Z r_tissue), C = P Z P.T + G R^-1 G.T.
Every D block is symmetric positive semidefinite, D = F F.T, so w = F y with
(I + h F.T C F) y = F.T (G R^-1 r_rigid - P Z r_tissue): a symmetric system
not below the identity, factorized by one tiled Cholesky. P Z P.T is
symmetrized; the exact compliance is symmetric, the stored response columns
only to their solve tolerance.

A sliding anchor of a friction tunnel acts as (I + S) D, S its shear (see
``shaft_preconditioner``). Then w = (I + S) F y with
(I + h F.T C (I + S) F) y = F.T (G R^-1 r_rigid - P Z r_tissue): the same
system with the right factor (I + S) F in place of F. It is not symmetric. For
a track that follows a friction tunnel (``nonsymmetric`` of the
preconditioner) the system K y = b is therefore solved through its normal
equations K.T K y = K.T b, which the same tiled Cholesky factorizes; every
other anchor law has no shear and keeps the symmetric solve unchanged. The
normal equations are positive definite for every nonsingular K, so their
factorization fails for a singular or nonfinite system only. A system whose
symmetric part is indefinite is solved as the host's general solve solves it,
and the residual certificate decides.

The normal equations are formed in every correction of a friction tunnel, also
when no anchor slides: what selects them is the tunnel model of the track, not
the state of its anchors. The system of a tunnel whose anchors all stick is
symmetric and is still solved through its square, with the square of its
condition. The proposal is accurate while cond(K) squared stays well below
1 / FP32 roundoff (1.7e7). Measured on 2026-11-16: in the whole grasp demo
cond(K) is at most 286 (2316 systems on cuda:0; cond(K.T K) at most 8.2e4,
error of the proposal at most 2.2e-3). On CPU fixtures the residual of the
guess passes the solve tolerance of 1e-3 from cond(K) of about 1100 to 2100,
and from about 2e4 no guess is given (``enabled`` is cleared); the same
tangents without a tunnel give a guess up to cond(K) of 1.3e6. While nothing
slides the price is 0.2 to 0.3 Krylov iterations per correction in
three stages of the demo (0.87, 0.98 and 0.95 against 0.67, 0.65 and 0.62
with the symmetric solve; one run each) and none in a fourth (0.93 against
1.01). No result is wrong beyond the range: the residual certificate and the
Krylov solve remain. Deciding per correction on the device whether a live
anchor has a shear, and factorizing K itself when none has, is not built.

A tangent of rank below three leaves zero columns in its factor F, and with
them zero columns in (I + S) F. With finite data a zero column makes its row
and column of F.T C (I + S) F and its right-hand side zero: that row of the
system is a row of the identity with a zero right-hand side, so its unknown is
zero and couples to no other. The system on the nonzero columns alone
therefore has the same solution. Where the padded system has more rows than
the compacted tile, a captured correction factorizes that smaller system
whenever it fits the tile; otherwise the padded one is, as before. The normal
equations are those of whichever system is factorized.

Nonfinite data stays visible on that path. The unknown of a dropped column is
returned as the right-hand side of its identity row, F.T (...) of the zero
column: zero for finite data, nonfinite when the tissue residual, the response
row or the lever of its sample is, and ``enabled`` is then cleared as after
the padded solve. Not seen is a nonfinite compliance entry of a dropped sample
beside finite response rows. ``schur_topology`` forms both from the same
entries of the basis, so the kernels do not produce that; the proposal is then
finite and the residual certificate decides.

Precision follows the consumer. The guess is stored and certified in FP32 and
polished by an FP32 Krylov solve, so the well-conditioned Schur system is
assembled and solved in FP32. Only the 6 x 6 rigid metric, whose inverse is
formed explicitly, the rank decisions of the tangent factors and the sums of
the normal equations use FP64. The normal equations have the square of the
condition of the system, so their entries are FP64 sums of the FP32 system,
rounded once.

Kernels that mix unlike work give each kind whole warps: threads of one warp
share control flow, so unlike branches inside it would run one after another.

Only the binding of a response basis stays on the host. Whether the bound
basis may be used by a launch, and which padded tile factorizes its track
samples, is device data that the host writes where it changes (``available``
of the guess, ``size_flags`` here): a captured correction holds the launches
of every tile behind conditional nodes and serves every later number of track
samples. ``enabled`` is the device validity flag of one correction; zero
selects the zero start. The current-operator residual certificate and the
unchanged PCG remain authoritative.
"""

import functools

import numpy as np
import warp as wp

from ..interface.insertion import LIVE_TRACK
from .woodbury_guess import _CAPACITY, _MAX_TRACKS

# CUDA threads that share one control flow.
_WARP = 32


def _warp_start(threads):
    """First thread index of the first whole warp after ``threads`` threads."""
    return -(-threads // _WARP) * _WARP


# ``schur_inputs``: response rows, then tangent factors, then the rigid solve.
_FACTORS = _warp_start(3 * _MAX_TRACKS)
_RIGID = _FACTORS + _warp_start(_MAX_TRACKS)
# ``schur_coefficients``: support columns, then the rigid twist and validity.
_TWIST = _warp_start(_CAPACITY)
# Smallest thread block of the cooperative factorization, as for the rigid metric.
_TILE_BLOCK_DIM = 64
# Unit roundoff of the FP32 anchor tangents: the data precision of D.
_ROUNDOFF = 2.0**-24
# Thread block of every launch of this module but the factorizations. The launches name it, so
# the module loaded with the scratch is the one they run. 256 is also the default of ``wp.launch``.
_LAUNCH_BLOCK_DIM = 256
# Rows of the compacted Schur system and the thread block of its factorization: the largest
# tile found whose factorization is cheap on this device. Measured on the GB10 (sm_121, Warp
# 1.17.0, cuSolverDx in one thread block): with 256 block threads the solve takes 47 us at 113
# and 114 rows, 289 us at 115 and 290 to 570 us at the nine sizes measured from 116 to 144; the
# factorization alone 22 us at 114 rows and 295 us at 120. A size is slow exactly when its
# loaded kernel has a stack frame of 320 bytes or more per thread instead of at most 72
# (cuFuncGetAttribute); the shared memory of the tile does not move it. That is a property of
# the code generated for a size, not a limit in rows: with 64 threads 77, 78 and 80 rows are
# cheap and 79 and 81 are not. So measure again after a change of Warp, of CUDA or of the
# kernel. With 512 block threads the kernels of 120 and 144 rows do not launch (CUDA error 701,
# too many resources requested for launch; they are built with 181 and 198 registers per
# thread). The tile holds 38 whole track samples.
_COMPACT_ROWS = 114
_COMPACT_BLOCK_DIM = 256


@wp.func
def _double(m: wp.mat33):
    return wp.mat33d(
        wp.float64(m[0, 0]),
        wp.float64(m[0, 1]),
        wp.float64(m[0, 2]),
        wp.float64(m[1, 0]),
        wp.float64(m[1, 1]),
        wp.float64(m[1, 2]),
        wp.float64(m[2, 0]),
        wp.float64(m[2, 1]),
        wp.float64(m[2, 2]),
    )


@wp.func
def _single(m: wp.mat33d):
    return wp.mat33(
        float(m[0, 0]),
        float(m[0, 1]),
        float(m[0, 2]),
        float(m[1, 0]),
        float(m[1, 1]),
        float(m[1, 2]),
        float(m[2, 0]),
        float(m[2, 1]),
        float(m[2, 2]),
    )


@wp.func
def _semidefinite_factor(source: wp.mat33):
    """Diagonally pivoted Cholesky F F.T = D of one FP32 tangent in FP64.

    LAPACK xPSTRF rule: pivot on the largest remaining diagonal and stop once
    none exceeds n * roundoff * max(diagonal). D is positive semidefinite by
    construction but assembled in FP32, so the dropped remainder is rounding
    noise of either sign. A remainder below -sqrt(roundoff) * max(diagonal)
    cannot be rounding: the block is indefinite and the factor is not valid.
    """
    a = _double(source)
    a = wp.float64(0.5) * (a + wp.transpose(a))
    scale = wp.max(a[0, 0], wp.max(a[1, 1], a[2, 2]))
    stop = wp.float64(3.0 * _ROUNDOFF) * scale
    factor = wp.mat33d(wp.float64(0.0))
    used = wp.vec3i(0, 0, 0)
    for k in range(3):
        pivot = int(-1)
        value = stop
        for i in range(3):
            if used[i] == 0 and a[i, i] > value:
                pivot = i
                value = a[i, i]
        if pivot >= 0:
            root = wp.sqrt(value)
            column = wp.vec3d(wp.float64(0.0))
            for i in range(3):
                if used[i] == 0:
                    column[i] = a[i, pivot] / root
            column[pivot] = root
            for i in range(3):
                factor[i, k] = column[i]
            a -= wp.outer(column, column)
            used[pivot] = 1
    valid = int(wp.isfinite(wp.ddot(factor, factor)) and wp.isfinite(scale))
    for i in range(3):
        if used[i] == 0 and a[i, i] < -wp.float64(_ROUNDOFF**0.5) * scale:
            valid = 0
    return _single(factor), valid


@wp.kernel(enable_backward=False)
def schur_topology(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    inverse_free: wp.array[int],
    support_column: wp.array[int],
    basis: wp.array[float],
    free: wp.array[int],
    support: wp.array[int],
    support_count: wp.array[int],
    live: wp.array[int],
    stride: int,
    tissue: wp.array2d[float],
    compliance: wp.array2d[float],
    supported: wp.array[int],
    gather: wp.array[int],
):
    """Rows of P Z and of sym(P Z P.T); they change only with tracks or basis."""
    # Launched over the whole scratch: support columns below _CAPACITY, compliance columns from
    # there on. A thread beyond the live track samples or support columns returns before any read.
    row, column = wp.tid()
    track, axis = row // 3, row % 3
    if track >= live[LIVE_TRACK]:
        return
    if column < _CAPACITY:
        if column >= support_count[0]:
            return
        value = wp.float64(0.0)
        covered = int(1)
        for j in range(4):
            compact = inverse_free[nodes[track][j]]
            if compact >= 0 and bary[track][j] != 0.0:
                value += wp.float64(bary[track][j]) * wp.float64(basis[column * stride + 3 * compact + axis])
                for component in range(3):
                    if support_column[3 * compact + component] < 0:
                        covered = 0
        tissue[row, column] = float(value)
        if column == 0 and axis == 0:
            supported[track] = covered
        if row == 0:
            # Scalar position of this support DOF in the full residual.
            dof = support[column]
            gather[column] = 3 * free[dof // 3] + dof % 3
    else:
        other = column - _CAPACITY
        other_track, other_axis = other // 3, other % 3
        if other_track >= live[LIVE_TRACK]:
            return
        value = wp.float64(0.0)
        for j in range(4):
            compact = inverse_free[nodes[track][j]]
            if compact >= 0 and bary[track][j] != 0.0:
                for k in range(4):
                    other_compact = inverse_free[nodes[other_track][k]]
                    if other_compact >= 0 and bary[other_track][k] != 0.0:
                        forward = support_column[3 * other_compact + other_axis]
                        backward = support_column[3 * compact + axis]
                        if forward >= 0 and backward >= 0:
                            value += (
                                wp.float64(0.5)
                                * wp.float64(bary[track][j])
                                * wp.float64(bary[other_track][k])
                                * (
                                    wp.float64(basis[forward * stride + 3 * compact + axis])
                                    + wp.float64(basis[backward * stride + 3 * other_compact + other_axis])
                                )
                            )
        compliance[row, other] = float(value)


@wp.kernel(enable_backward=False)
def schur_inputs(
    matrices: wp.array[wp.mat33],
    shears: wp.array[wp.mat33],
    rigid_blocks: wp.array[wp.mat33],
    rhs: wp.array[float],
    gather: wp.array[int],
    support_count: wp.array[int],
    live: wp.array[int],
    tissue: wp.array2d[float],
    response: wp.array[float],
    factors: wp.array[wp.mat33],
    rights: wp.array[wp.mat33],
    factor_valid: wp.array[int],
    rigid_inverse: wp.array[wp.mat33d],
    rigid_response: wp.array[wp.mat33],
    rigid_solution: wp.array[wp.vec3d],
    rigid_valid: wp.array[int],
):
    """P Z r_tissue per row, D = F F.T and (I + S) F per track, and the 6 x 6 rigid solve."""
    i = wp.tid()
    count = support_count[0]
    tracks = live[LIVE_TRACK]
    if i < 3 * tracks:
        value = float(0.0)
        for column in range(count):
            value += tissue[i, column] * rhs[gather[column]]
        response[i] = value
    elif i >= _FACTORS and i < _FACTORS + tracks:
        track = i - _FACTORS
        factor, valid = _semidefinite_factor(matrices[track])
        factors[track] = factor
        # The anchor acts as (I + S) F F.T. Without a shear the right factor is F itself.
        rights[track] = factor + shears[track] * factor
        factor_valid[track] = valid
    elif i == _RIGID:
        # Blocks follow ``rigid_blocks``: translation, rotation, then the two
        # cross blocks. Eliminate translation first; both pivots of a positive
        # definite metric have positive determinants.
        a = _double(rigid_blocks[0])
        d = _double(rigid_blocks[1])
        b = _double(rigid_blocks[2])
        c = _double(rigid_blocks[3])
        a_inverse = wp.inverse(a)
        left = c * a_inverse
        schur = d - left * b
        rotation = wp.inverse(schur)
        lower = -(rotation * left)
        upper = -(a_inverse * b * rotation)
        translation = a_inverse - upper * left
        rigid_inverse[0] = translation
        rigid_inverse[1] = rotation
        rigid_inverse[2] = upper
        rigid_inverse[3] = lower
        rigid_response[0] = _single(translation)
        rigid_response[1] = _single(rotation)
        rigid_response[2] = _single(upper)
        rigid_response[3] = _single(lower)
        last = rhs.shape[0] - 6
        force = wp.vec3d(wp.float64(rhs[last]), wp.float64(rhs[last + 1]), wp.float64(rhs[last + 2]))
        torque = wp.vec3d(wp.float64(rhs[last + 3]), wp.float64(rhs[last + 4]), wp.float64(rhs[last + 5]))
        linear = translation * force + upper * torque
        angular = lower * force + rotation * torque
        rigid_solution[0] = linear
        rigid_solution[1] = angular
        # Every entry of the inverse multiplies into one of these components,
        # so a nonfinite inverse or residual leaves a nonfinite solution.
        rigid_valid[0] = int(
            wp.determinant(a) > wp.float64(0.0)
            and wp.determinant(schur) > wp.float64(0.0)
            and wp.isfinite(wp.dot(linear, linear) + wp.dot(angular, angular))
        )


@wp.kernel(enable_backward=False)
def schur_assemble(
    live: wp.array[int],
    levers: wp.array[wp.vec3],
    h: wp.array[float],
    compliance: wp.array2d[float],
    response: wp.array[float],
    factors: wp.array[wp.mat33],
    rights: wp.array[wp.mat33],
    rigid_response: wp.array[wp.mat33],
    rigid_solution: wp.array[wp.vec3d],
    matrix: wp.array2d[float],
    rhs: wp.array[float],
):
    """I + h F.T C (I + S) F and F.T (G R^-1 r_rigid - P Z r_tissue); padding is identity."""
    i, j = wp.tid()
    tracks = live[LIVE_TRACK]
    block = wp.mat33(0.0)
    if i < tracks and j < tracks:
        skew_i = wp.skew(levers[i])
        skew_j = wp.skew(levers[j])
        c = (
            rigid_response[0]
            + rigid_response[2] * skew_j
            - skew_i * rigid_response[3]
            - skew_i * rigid_response[1] * skew_j
        )
        for a in range(3):
            for b in range(3):
                c[a, b] += compliance[3 * i + a, 3 * j + b]
        block = h[0] * wp.transpose(factors[i]) * c * rights[j]
    if i == j:
        value = wp.vec3(0.0)
        if i < tracks:
            relative = (
                wp.vec3(rigid_solution[0])
                + wp.cross(wp.vec3(rigid_solution[1]), levers[i])
                - wp.vec3(response[3 * i], response[3 * i + 1], response[3 * i + 2])
            )
            value = wp.transpose(factors[i]) * relative
        block += wp.identity(n=3, dtype=float)
        for a in range(3):
            rhs[3 * i + a] = value[a]
    for a in range(3):
        for b in range(3):
            matrix[3 * i + a, 3 * j + b] = block[a, b]


@wp.kernel(enable_backward=False)
def schur_compact_index(
    live: wp.array[int],
    factors: wp.array[wp.mat33],
    index: wp.array[int],
    position: wp.array[int],
    count: wp.array[int],
    fits: wp.array[int],
):
    """Padded rows of the nonzero tangent-factor columns in their order, and its inverse map."""
    # One thread per padded row and one more for the count. Every thread counts the nonzero
    # columns in front of its row over all live tracks, so the threads share one control flow.
    row = wp.tid()
    before = int(0)
    own = int(0)
    for track in range(live[LIVE_TRACK]):
        factor = factors[track]
        for column in range(3):
            # A nonfinite entry differs from zero: such a column is kept, and ``factor_valid`` rejects it.
            if factor[0, column] != 0.0 or factor[1, column] != 0.0 or factor[2, column] != 0.0:
                other = 3 * track + column
                if other < row:
                    before += 1
                elif other == row:
                    own = 1
    if row == 3 * _MAX_TRACKS:
        count[0] = before
        fits[0] = int(before <= _COMPACT_ROWS)
    elif own != 0:
        position[row] = before
        if before < _COMPACT_ROWS:
            index[before] = row
    else:
        position[row] = -1


@wp.kernel(enable_backward=False)
def schur_compact_assemble(
    levers: wp.array[wp.vec3],
    h: wp.array[float],
    compliance: wp.array2d[float],
    response: wp.array[float],
    factors: wp.array[wp.mat33],
    rights: wp.array[wp.mat33],
    rigid_response: wp.array[wp.mat33],
    rigid_solution: wp.array[wp.vec3d],
    index: wp.array[int],
    count: wp.array[int],
    matrix: wp.array2d[float],
    rhs: wp.array[float],
):
    """The entries of ``schur_assemble`` at the nonzero columns; padding is identity."""
    # Each entry is taken from the same 3 x 3 block product as in ``schur_assemble``, so the
    # compacted system holds the values of the padded one.
    a, b = wp.tid()
    rows = count[0]
    value = float(0.0)
    if a < rows and b < rows:
        i, j = index[a] // 3, index[b] // 3
        skew_i = wp.skew(levers[i])
        skew_j = wp.skew(levers[j])
        c = (
            rigid_response[0]
            + rigid_response[2] * skew_j
            - skew_i * rigid_response[3]
            - skew_i * rigid_response[1] * skew_j
        )
        for r in range(3):
            for s in range(3):
                c[r, s] += compliance[3 * i + r, 3 * j + s]
        block = h[0] * wp.transpose(factors[i]) * c * rights[j]
        value = block[index[a] % 3, index[b] % 3]
    if a == b:
        right = float(0.0)
        if a < rows:
            track = index[a] // 3
            relative = (
                wp.vec3(rigid_solution[0])
                + wp.cross(wp.vec3(rigid_solution[1]), levers[track])
                - wp.vec3(response[3 * track], response[3 * track + 1], response[3 * track + 2])
            )
            right = (wp.transpose(factors[track]) * relative)[index[a] % 3]
        value += 1.0
        rhs[a] = right
    matrix[a, b] = value


@wp.kernel(enable_backward=False)
def schur_compact_scatter(
    live: wp.array[int],
    levers: wp.array[wp.vec3],
    response: wp.array[float],
    factors: wp.array[wp.mat33],
    rigid_solution: wp.array[wp.vec3d],
    position: wp.array[int],
    compact: wp.array[float],
    solution: wp.array[float],
):
    """The compacted solution in its padded rows; a dropped column gets its padded right-hand side."""
    # The padded system solves a dropped column as a row of the identity, so its unknown is its
    # right-hand side: the entry of ``schur_assemble`` for a zero column of the factor. That is
    # zero for finite data and nonfinite otherwise, which ``schur_coefficients`` then rejects.
    row = wp.tid()
    track, column = row // 3, row % 3
    value = float(0.0)
    if position[row] >= 0:
        value = compact[position[row]]
    elif track < live[LIVE_TRACK]:
        relative = (
            wp.vec3(rigid_solution[0])
            + wp.cross(wp.vec3(rigid_solution[1]), levers[track])
            - wp.vec3(response[3 * track], response[3 * track + 1], response[3 * track + 2])
        )
        value = (wp.transpose(factors[track]) * relative)[column]
    solution[row] = value


@wp.kernel(enable_backward=False)
def schur_normal(
    count: wp.array[int],
    entry: int,
    per: int,
    matrix: wp.array2d[float],
    rhs: wp.array[float],
    gram: wp.array2d[float],
    normal_rhs: wp.array[float],
):
    """K.T K and K.T b of a system K y = b that is an identity beyond its rows in use."""
    # One thread per entry of K.T K and one more row of threads for K.T b. The rows in use are
    # ``per * count[entry]``: three per track sample of the padded system, the nonzero columns of
    # the compacted one. Beyond them K is an identity, and so are its normal equations. The
    # normal equations square the condition of K, so the products are summed in FP64 and
    # rounded to FP32 once. Both triangles hold the same sums in the same order.
    row, column = wp.tid()
    used = per * count[entry]
    width = gram.shape[0]
    value = wp.float64(0.0)
    if column < used:
        if row < used:
            for k in range(used):
                value += wp.float64(matrix[k, row]) * wp.float64(matrix[k, column])
        elif row == width:
            for k in range(used):
                value += wp.float64(matrix[k, column]) * wp.float64(rhs[k])
    elif row == width:
        value = wp.float64(rhs[column])
    elif row == column:
        value = wp.float64(1.0)
    if row == width:
        normal_rhs[column] = float(value)
    else:
        gram[row, column] = float(value)


@functools.cache
def _solve_tiles(n: int):
    """Cooperative Cholesky solve of the leading ``n`` x ``n`` system.

    Tile shapes are compile-time constants, so each track capacity has its own
    kernel. A failed factorization leaves a nonpositive or nonfinite pivot.
    """

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        matrix: wp.array2d[float],
        rhs: wp.array[float],
        solution: wp.array[float],
        valid: wp.array[int],
    ):
        _block, lane = wp.tid()
        lower = wp.tile_load(matrix, shape=(n, n))
        wp.tile_cholesky_inplace(lower)
        y = wp.tile_load(rhs, shape=n)
        wp.tile_cholesky_solve_inplace(lower, y)
        wp.tile_store(solution, y)
        if lane == 0:
            ok = int(1)
            for i in range(n):
                pivot = lower[i, i]
                if not wp.isfinite(pivot) or pivot <= 0.0:
                    ok = 0
            valid[0] = ok

    return kernel


def _padded_samples(count: int):
    """Track samples of the padded tile that factorizes a track of ``count`` samples."""
    # Powers of two bound the padded factorization by eight times the
    # exact one up to 16 samples, with three compiled kernels.
    capacity = 4
    while capacity < count and capacity < 16:
        capacity *= 2
    # From 17 samples on the padded tile is the compacted tile, which holds 38 samples. Its
    # 114 rows cost what the 96 rows of 32 samples cost in a tile of their own (measured:
    # 47.8 against 49.5 us), so no kernel is built for those.
    if count > 16:
        capacity = _COMPACT_ROWS // 3
    # A padded tile of 64 samples (192 rows) does not fit the shared memory of the device
    # (147 kB against 101 kB). Beyond the compacted tile the padded size grows in steps of
    # 8 samples, so 39 and 40 samples factorize 120 rows and 41 to 48 samples 144 rows
    # whenever their nonzero columns do not fit the compacted tile.
    if count > _COMPACT_ROWS // 3:
        capacity = 8 * (-(-count // 8))
    return capacity


# Every padded tile that a track within the scratch of the start guess can need, smallest
# first: 4, 8, 16, 38, 40 and 48 samples (12, 24, 48, 114, 120 and 144 rows).
_TILES = tuple(sorted({_padded_samples(count) for count in range(1, _MAX_TRACKS + 1)}))


def _tile(capacity: int):
    """Factorization kernel and thread block of the padded tile of ``capacity`` track samples."""
    if 3 * capacity == _COMPACT_ROWS:
        return _solve_tiles(_COMPACT_ROWS), _COMPACT_BLOCK_DIM
    # One thread per row at least: cuSolverDx is several times slower
    # on this system once a tile has more rows than its block threads.
    block_dim = _TILE_BLOCK_DIM
    while block_dim < 3 * capacity:
        block_dim *= 2
    return _solve_tiles(3 * capacity), block_dim


def load_kernels(device):
    """Create and load for ``device`` every kernel that a correction of the start guess can launch.

    A capture preloads named modules only, and the factorization of each padded tile is a
    module of its own, which exists once its kernel is created. No correction of a run may
    build one: a scene calls this at start-up, and the scratch of a guess calls it where it
    is made. The tiles are those of every track count that the scratch holds, not of the
    counts that one scene reaches. A module that is loaded already is not loaded again.
    """
    for capacity in _TILES:
        kernel, block_dim = _tile(capacity)
        wp.load_module(kernel.module, device=device, block_dim=block_dim if device.is_cuda else 1)
    # The module of this file, at the thread block that its launches name.
    wp.load_module(schur_compact_index.module, device=device, block_dim=_LAUNCH_BLOCK_DIM if device.is_cuda else 1)


@wp.kernel(enable_backward=False)
def schur_coefficients(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    inverse_free: wp.array[int],
    levers: wp.array[wp.vec3],
    h: wp.array[float],
    rhs: wp.array[float],
    gather: wp.array[int],
    support: wp.array[int],
    support_count: wp.array[int],
    live: wp.array[int],
    supported: wp.array[int],
    rights: wp.array[wp.mat33],
    factor_valid: wp.array[int],
    rigid_inverse: wp.array[wp.mat33d],
    rigid_solution: wp.array[wp.vec3d],
    rigid_valid: wp.array[int],
    solution: wp.array[float],
    solve_valid: wp.array[int],
    surface_count: wp.array[int],
    surface_valid: wp.array[int],
    coefficients: wp.array[float],
    surface_active: wp.array[int],
    enabled: wp.array[int],
):
    """r_tissue + h P.T w per support column, w = (I + S) F y, then the rigid twist and validity."""
    column = wp.tid()
    count = support_count[0]
    tracks = live[LIVE_TRACK]
    if column < count:
        dof = support[column]
        compact, axis = dof // 3, dof % 3
        value = rhs[gather[column]]
        for track in range(tracks):
            for j in range(4):
                if inverse_free[nodes[track][j]] == compact:
                    weight = rights[track] * wp.vec3(
                        solution[3 * track], solution[3 * track + 1], solution[3 * track + 2]
                    )
                    value += h[0] * bary[track][j] * weight[axis]
        coefficients[column] = value
    elif column == _TWIST:
        ok = rigid_valid[0] != 0 and solve_valid[0] != 0
        force = wp.vec3(0.0)
        torque = wp.vec3(0.0)
        for track in range(tracks):
            ok = ok and supported[track] != 0 and factor_valid[track] != 0
            weight = rights[track] * wp.vec3(solution[3 * track], solution[3 * track + 1], solution[3 * track + 2])
            force += weight
            torque += wp.cross(levers[track], weight)
        step = wp.float64(h[0])
        linear = rigid_solution[0] - step * (rigid_inverse[0] * wp.vec3d(force) + rigid_inverse[2] * wp.vec3d(torque))
        angular = rigid_solution[1] - step * (rigid_inverse[3] * wp.vec3d(force) + rigid_inverse[1] * wp.vec3d(torque))
        for axis in range(3):
            coefficients[_CAPACITY + axis] = float(linear[axis])
            coefficients[_CAPACITY + 3 + axis] = float(angular[axis])
        # Nonfinite residuals or weights reach every tissue coefficient through
        # the same sums, so these two vectors certify the whole proposal. The
        # weight of a dropped column is its zero column of the right factor
        # times the padded right-hand side that ``schur_compact_scatter``
        # returns for it; a nonfinite shear leaves a nonfinite right factor.
        ok = ok and wp.isfinite(wp.dot(linear, linear) + wp.dot(angular, angular))
        surface_active[0] = int(surface_count[0] > 0 and surface_valid[0] != 0)
        enabled[0] = int(ok)


class DeviceSchurCoefficients:
    """Persistent scratch and launches for one ``WoodburyInitialGuess``.

    ``bind_structure``, ``bind_support`` and ``select`` run on the host outside capture.
    ``enqueue_topology`` belongs once in front of a trial loop, ``enqueue`` in every
    correction after the contact inputs. Both read what the host bound last.

    The kernels read the number of support columns from ``core.count`` and the number of
    track samples from the live counts of the interaction, and every launch but the
    factorization has the size of its scratch. The host decides whether they run and
    writes that decision to the device (``available`` of the guess): it is set only with
    counts that the scratch holds.

    The padded tile follows the number of track samples (``_padded_samples``). Its
    kernels are all created and loaded here, with the scratch, and ``size_flags`` names
    the one of the current count on the device: one entry per tile, one of them nonzero,
    and a last entry that is nonzero while the padded tile is larger than the compacted
    one. The host writes them where the tile changes. A captured correction holds the
    factorization of every tile behind a conditional node on its entry, so one graph
    serves every track count; outside a capture the host launches the tile it selected.

    A track that follows a friction tunnel (``nonsymmetric`` of the preconditioner) has a
    nonsymmetric system: the normal equations of the system that is factorized, padded or
    compacted, are formed in front of its factorization. That is a property of the anchor
    law and an entry of the key; whether an anchor slides is device data in ``pre.shears``.

    The compacted system has one tile of ``_COMPACT_ROWS`` rows for every track count. How
    many of its rows are in use, and whether the nonzero columns fit at all, is device data
    of each correction, so neither changes a captured graph. Its kernels are created and
    loaded here, with the scratch. The same tile factorizes the padded system of 17 to 38
    track samples, which it holds whole.

    The choice between the compacted and the padded system is a conditional node of the
    captured graph. A correction issued outside a capture factorizes the padded system:
    choosing there would be a read of the flag on the host.

    The map from compact degrees of freedom to basis columns (``support_column``) keeps
    its storage for one FEM structure and is rewritten in place when the support of the
    basis changes, so a graph that addresses it follows every later support.
    """

    def __init__(self, guess):
        self.guess = guess
        device = guess.pre.device
        rows = 3 * _MAX_TRACKS
        self.tissue = wp.zeros((rows, _CAPACITY), dtype=float, device=device)
        self.compliance = wp.zeros((rows, rows), dtype=float, device=device)
        self.supported = wp.zeros(_MAX_TRACKS, dtype=int, device=device)
        self.gather = wp.zeros(_CAPACITY, dtype=int, device=device)
        self.response = wp.zeros(rows, dtype=float, device=device)
        self.factors = wp.zeros(_MAX_TRACKS, dtype=wp.mat33, device=device)
        # (I + S) F of each anchor: the factor itself wherever the anchor has no shear.
        self.rights = wp.zeros(_MAX_TRACKS, dtype=wp.mat33, device=device)
        self.factor_valid = wp.zeros(_MAX_TRACKS, dtype=int, device=device)
        self.rigid_inverse = wp.zeros(4, dtype=wp.mat33d, device=device)
        self.rigid_response = wp.zeros(4, dtype=wp.mat33, device=device)
        self.rigid_solution = wp.zeros(2, dtype=wp.vec3d, device=device)
        self.rigid_valid = wp.zeros(1, dtype=int, device=device)
        self.matrix = wp.zeros((rows, rows), dtype=float, device=device)
        self.rhs = wp.zeros(rows, dtype=float, device=device)
        self.solution = wp.zeros(rows, dtype=float, device=device)
        # Normal equations of the padded system of a friction tunnel.
        self.gram = wp.zeros((rows, rows), dtype=float, device=device)
        self.normal_rhs = wp.zeros(rows, dtype=float, device=device)
        self.solve_valid = wp.zeros(1, dtype=int, device=device)
        # Compacted system: padded row of each compacted row, its inverse (-1 for a dropped
        # column), the number of nonzero columns and whether they fit the compacted tile.
        self.compact_index = wp.zeros(_COMPACT_ROWS, dtype=int, device=device)
        self.compact_position = wp.zeros(rows, dtype=int, device=device)
        self.compact_count = wp.zeros(1, dtype=int, device=device)
        self.compact_fits = wp.zeros(1, dtype=int, device=device)
        self.compact_matrix = wp.zeros((_COMPACT_ROWS, _COMPACT_ROWS), dtype=float, device=device)
        self.compact_rhs = wp.zeros(_COMPACT_ROWS, dtype=float, device=device)
        self.compact_solution = wp.zeros(_COMPACT_ROWS, dtype=float, device=device)
        # Normal equations of the compacted system of a friction tunnel.
        self.compact_gram = wp.zeros((_COMPACT_ROWS, _COMPACT_ROWS), dtype=float, device=device)
        self.compact_normal_rhs = wp.zeros(_COMPACT_ROWS, dtype=float, device=device)
        # A capture preloads named modules only, and no correction of a run may load a kernel:
        # the factorizations of every padded tile and this module are loaded now.
        load_kernels(device)
        self.compact_kernel = _solve_tiles(_COMPACT_ROWS)
        self.tiles = tuple(_tile(capacity) for capacity in _TILES)
        # One entry per padded tile, nonzero for the tile of the current track count, and a
        # last entry that is nonzero while that tile is larger than the compacted one.
        self.size_flags = wp.zeros(len(_TILES) + 1, dtype=int, device=device)
        self.size_views = [self.size_flags[index : index + 1] for index in range(len(_TILES) + 1)]
        # Basis column of every compact degree of freedom of the bound structure, -1 outside the support.
        self.support_column = None
        self.support = None
        # The padded tile that the host selected last: its track samples, thread block and kernel.
        self.capacity = 0
        self.block_dim = 0
        self.solve_kernel = None

    @property
    def owners(self):
        return tuple(value for value in vars(self).values() if isinstance(value, wp.array))

    def bind_structure(self):
        """Size the column map for the free nodes of the structure that the guess bound; no support yet."""
        size = 3 * len(self.guess.free_host)
        if self.support_column is None or self.support_column.shape[0] != size:
            self.support_column = wp.full(size, -1, dtype=int, device=self.guess.pre.device)
        self.support = None

    def bind_support(self, support):
        """Rewrite the column map in place for the support of the basis; nothing is read from the device."""
        if support == self.support:
            return
        columns = np.full(self.support_column.shape[0], -1, dtype=np.int32)
        columns[list(support)] = np.arange(len(support), dtype=np.int32)
        self.support_column.assign(columns)
        self.support = support

    def select(self, count):
        """Select the padded tile of ``count`` track samples and name it on the device where it changed."""
        capacity = _padded_samples(count)
        if capacity != self.capacity:
            index = _TILES.index(capacity)
            self.capacity = capacity
            self.solve_kernel, self.block_dim = self.tiles[index]
            flags = np.zeros(len(_TILES) + 1, dtype=np.int32)
            flags[index] = 1
            flags[-1] = int(3 * capacity > _COMPACT_ROWS)
            self.size_flags.assign(flags)

    def key(self):
        """Storage of what a captured correction addresses on this object beside its scratch."""
        return tuple((a.ptr, a.shape, a.dtype) for a in (self.support_column, self.size_flags))

    def enqueue_topology(self):
        """Rows of P Z and of the track compliance, once per launch of a trial loop, while the guess is available."""
        g = self.guess
        if g.pre.device.is_capturing:
            # The bound basis may be unusable for a later launch of this graph; its maps are then stale.
            wp.capture_if(g.available, self._enqueue_topology)
        elif g.usable:
            # Outside a capture the host decides: a conditional there would read the flag back.
            self._enqueue_topology()

    def _enqueue_topology(self):
        g, p = self.guess, self.guess.pre
        interaction = p.coupler.interaction
        wp.launch(
            schur_topology,
            (3 * _MAX_TRACKS, _CAPACITY + 3 * _MAX_TRACKS),
            [
                interaction._track_nodes,
                interaction._track_bary,
                g.inverse_free,
                self.support_column,
                g.core.basis,
                g.core.free,
                g.core.support,
                g.core.count,
                interaction._live,
                3 * len(g.free_host),
                self.tissue,
                self.compliance,
                self.supported,
                self.gather,
            ],
            device=p.device,
            block_dim=_LAUNCH_BLOCK_DIM,
        )

    def enqueue(self):
        self._enqueue_inputs()
        if self.guess.pre.device.is_capturing:
            # One graph for every track count: the device follows the tile that the host named.
            small = [index for index, capacity in enumerate(_TILES) if 3 * capacity <= _COMPACT_ROWS]
            large = [index for index, capacity in enumerate(_TILES) if 3 * capacity > _COMPACT_ROWS]
            wp.capture_if(
                self.size_views[-1],
                on_true=lambda: self._enqueue_counted(large),
                on_false=lambda: self._enqueue_padded(small),
            )
        else:
            # The tile that the host selected. Warp would read a flag on the host to choose here.
            self._enqueue_padded()
        self._enqueue_coefficients()

    def _enqueue_counted(self, tiles):
        """A padded tile larger than the compacted one: factorize the nonzero columns alone when they fit."""
        # The device counts the nonzero columns of this correction.
        self._enqueue_compact_index()
        wp.capture_if(self.compact_fits, self._enqueue_compacted, lambda: self._enqueue_padded(tiles))

    def _enqueue_inputs(self):
        """Response rows, tangent factors and the rigid solve of this correction."""
        g, p = self.guess, self.guess.pre
        interaction = p.coupler.interaction
        wp.launch(
            schur_inputs,
            _RIGID + 1,
            [
                p.matrices,
                p.shears,
                p.rigid_blocks,
                g.rhs_scalar,
                self.gather,
                g.core.count,
                interaction._live,
                self.tissue,
                self.response,
                self.factors,
                self.rights,
                self.factor_valid,
                self.rigid_inverse,
                self.rigid_response,
                self.rigid_solution,
                self.rigid_valid,
            ],
            device=p.device,
            block_dim=_LAUNCH_BLOCK_DIM,
        )

    def _enqueue_coefficients(self):
        """The proposal and its validity flag from the solution of either system."""
        g, p = self.guess, self.guess.pre
        interaction = p.coupler.interaction
        surface = getattr(p, "surface", None)
        wp.launch(
            schur_coefficients,
            _TWIST + 1,
            [
                interaction._track_nodes,
                interaction._track_bary,
                g.inverse_free,
                p.levers,
                p.time_step,
                g.rhs_scalar,
                self.gather,
                g.core.support,
                g.core.count,
                interaction._live,
                self.supported,
                self.rights,
                self.factor_valid,
                self.rigid_inverse,
                self.rigid_solution,
                self.rigid_valid,
                self.solution,
                self.solve_valid,
                g.no_surface if surface is None else surface.anchors.count,
                g.no_surface if surface is None else surface.valid,
                g.coefficients,
                g.surface_active,
                g.enabled,
            ],
            device=p.device,
            block_dim=_LAUNCH_BLOCK_DIM,
        )

    def _enqueue_padded(self, tiles=None):
        """Assemble and factorize the padded system.

        By the tile that the host selected, or, with the indices ``tiles`` inside a capture,
        by the one of those tiles that the size flags name on the device.
        """
        p = self.guess.pre
        wp.launch(
            schur_assemble,
            (_MAX_TRACKS, _MAX_TRACKS),
            [
                p.coupler.interaction._live,
                p.levers,
                p.time_step,
                self.compliance,
                self.response,
                self.factors,
                self.rights,
                self.rigid_response,
                self.rigid_solution,
                self.matrix,
                self.rhs,
            ],
            device=p.device,
            block_dim=_LAUNCH_BLOCK_DIM,
        )
        matrix, rhs = self.matrix, self.rhs
        if p.nonsymmetric:
            # Three rows in use per track sample; the padding beyond them is an identity.
            live = p.coupler.interaction._live
            matrix, rhs = self._enqueue_normal(live, LIVE_TRACK, 3, matrix, rhs, self.gram, self.normal_rhs)
        if tiles is None:
            self._enqueue_factorization(self.solve_kernel, self.block_dim, matrix, rhs)
            return
        for index in tiles:
            kernel, block_dim = self.tiles[index]
            wp.capture_if(
                self.size_views[index],
                lambda kernel=kernel, block_dim=block_dim: self._enqueue_factorization(kernel, block_dim, matrix, rhs),
            )

    def _enqueue_factorization(self, kernel, block_dim, matrix, rhs):
        """Cooperative Cholesky solve of the leading block of the padded system by one tile."""
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[matrix, rhs, self.solution, self.solve_valid],
            block_dim=block_dim,
            device=self.guess.pre.device,
        )

    def _enqueue_normal(self, count, entry, per, matrix, rhs, gram, normal_rhs):
        """Normal equations of the system that is factorized next, with ``per * count[entry]`` rows in use."""
        wp.launch(
            schur_normal,
            (gram.shape[0] + 1, gram.shape[0]),
            [count, entry, per, matrix, rhs, gram, normal_rhs],
            device=self.guess.pre.device,
            block_dim=_LAUNCH_BLOCK_DIM,
        )
        return gram, normal_rhs

    def _enqueue_compact_index(self):
        """Count and order the nonzero tangent-factor columns of this correction."""
        p = self.guess.pre
        wp.launch(
            schur_compact_index,
            3 * _MAX_TRACKS + 1,
            [
                p.coupler.interaction._live,
                self.factors,
                self.compact_index,
                self.compact_position,
                self.compact_count,
                self.compact_fits,
            ],
            device=p.device,
            block_dim=_LAUNCH_BLOCK_DIM,
        )

    def _enqueue_compacted(self):
        """Assemble and factorize the system on its nonzero columns; they fit the compacted tile."""
        p = self.guess.pre
        wp.launch(
            schur_compact_assemble,
            (_COMPACT_ROWS, _COMPACT_ROWS),
            [
                p.levers,
                p.time_step,
                self.compliance,
                self.response,
                self.factors,
                self.rights,
                self.rigid_response,
                self.rigid_solution,
                self.compact_index,
                self.compact_count,
                self.compact_matrix,
                self.compact_rhs,
            ],
            device=p.device,
            block_dim=_LAUNCH_BLOCK_DIM,
        )
        matrix, rhs = self.compact_matrix, self.compact_rhs
        if p.nonsymmetric:
            # The rows in use are the nonzero columns that the index kernel counted.
            matrix, rhs = self._enqueue_normal(
                self.compact_count, 0, 1, matrix, rhs, self.compact_gram, self.compact_normal_rhs
            )
        wp.launch_tiled(
            self.compact_kernel,
            dim=[1],
            inputs=[matrix, rhs, self.compact_solution, self.solve_valid],
            block_dim=_COMPACT_BLOCK_DIM,
            device=p.device,
        )
        wp.launch(
            schur_compact_scatter,
            3 * _MAX_TRACKS,
            [
                p.coupler.interaction._live,
                p.levers,
                self.response,
                self.factors,
                self.rigid_solution,
                self.compact_position,
                self.compact_solution,
                self.solution,
            ],
            device=p.device,
            block_dim=_LAUNCH_BLOCK_DIM,
        )
