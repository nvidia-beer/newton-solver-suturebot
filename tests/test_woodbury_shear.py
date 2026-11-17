# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""The start guess of a track that follows a friction tunnel: the Schur system with the shear.

A sliding anchor of a friction tunnel acts as (I + S) D, S = friction * sign * t d.T
(``shaft_preconditioner``). The start guess of the correction solves its small system with that
tangent. The host packs (I + S) D and its general dense solve takes the system as it is. The device
solves (I + h F.T C (I + S) F) y = F.T (...) through its normal equations, on the padded system or on
its nonzero columns, and returns w = (I + S) F y.

Three things are shown. With a shear, the device coefficients are those of the host solve of the
nonsymmetric system, on both device paths and up to the condition of the system in the demo, and
they are far from the solve without the shear. With no shear in a friction tunnel, the normal
equations give the coefficients of the symmetric solve to rounding. For every other anchor law the
kernels write the bits of their text before the shear: that text is rebuilt here from each
kernel's source by undoing the edit, as ``test_woodbury_live_counts`` does for the counts. The bits
are those of tangents without an entry of -0.0, which the fixtures do not hold: with a zero shear
``tangent + shear * tangent`` and ``factor + shear * factor`` turn such an entry into +0.0, so the
packet and the right factor then equal the tangent and the factor in value, not bit for bit. No
coefficient changes with it.
"""

import inspect
import linecache
import re
import unittest
from unittest.mock import patch

import numpy as np
import test_woodbury_device as fixture
import warp as wp
from _woodbury_reference import sheared

from newton_solver_suturebot._src.solvers.coupled.acceleration import woodbury_device, woodbury_guess
from newton_solver_suturebot._src.solvers.coupled.acceleration.woodbury_device import (
    _COMPACT_ROWS,
    _LAUNCH_BLOCK_DIM,
    schur_normal,
)
from newton_solver_suturebot._src.solvers.coupled.acceleration.woodbury_guess import _CAPACITY, _HEADER, _MAX_TRACKS
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import LIVE_TRACK
from newton_solver_suturebot._src.utils.capture import ScopedCapture

DEVICE = fixture.DEVICE
EPS = float(np.finfo(np.float32).eps)
# Friction coefficient of the demo's tunnel.
FRICTION = 0.2
# Condition number of the Schur system K = I + h F.T C (I + S) F in the grasp demo, sampled once per frame
# over the whole demo on cuda:0 (2026-11-16, 2316 systems): median 18, 90th percentile 92, at most 286,
# reached while the tool lets go of the needle (frames 827 to 857: median 90, 90th percentile 129). In
# the first 826 frames it is at most 85 in one run and 90 in another: median 6 while the needle is
# inserted, 21 during the lift and 20 to 45 while it is lowered. The fixtures below go beyond it by a
# more compliant tissue and needle.
DEMO_CONDITION = 286.0
# Compliance of tissue and needle against the fixture's default: 1, 10 and 40 times.
SCALES = (1.0, 0.1, 0.025)
# Relative tolerance of the solve that the guess starts (``_ShaftPreconditioner``): the tighter of its
# two, for a hard wall or a penalty tunnel. A compliant tunnel is solved to ten times that.
SOLVE_TOLERANCE = 1.0e-3


def tunnel(seed, tracks, *, scale=1.0, friction=FRICTION, ranks=(2, 3, 3, 0, 2)):
    """A packet of the start-guess tests; ``friction`` None is an anchor law without a shear."""
    return fixture.Packet(
        seed,
        tracks,
        ranks=ranks,
        friction=friction,
        tissue=(0.003 * scale, 0.03 * scale),
        rigid=(0.002 * scale, 0.02 * scale),
    )


def used(helper):
    """The coefficients that the guess reads: the support columns and the rigid twist."""
    return np.r_[np.arange(len(helper.support)), np.arange(_CAPACITY, _CAPACITY + 6)]


def distance(actual, expected, helper):
    """Largest difference of the coefficients in use, relative to the largest of ``expected``."""
    columns = used(helper)
    return float(np.abs(actual[columns] - expected[columns]).max() / np.abs(expected[columns]).max())


def systems(helper, compacted):
    """K, and the normal equations the device factorized, on the rows in use of the padded or compacted system."""
    solve = helper.device_solve
    if compacted:
        rows = int(solve.compact_count.numpy()[0])
        return solve.compact_matrix.numpy()[:rows, :rows], solve.compact_gram.numpy()[:rows, :rows]
    rows = 3 * helper.pre.count
    return solve.matrix.numpy()[:rows, :rows], solve.gram.numpy()[:rows, :rows]


def condition(matrix):
    return float(np.linalg.cond(np.asarray(matrix, dtype=float))) if matrix.size else 1.0


def correction(packet, *, compact=None):
    """Coefficients and flag of one device correction with the branch of a captured graph, or the forced one."""
    helper = packet.helper
    if helper.prepare_device(packet.manager) is None:
        raise AssertionError("No response basis is bound")
    helper.coefficients.fill_(np.nan)
    helper.enabled.fill_(-1)
    helper.device_solve.enqueue_topology()
    fixture._correction(helper.device_solve, compact=compact)
    return helper.coefficients.numpy().copy(), int(helper.enabled.numpy()[0])


def supported_load(packet):
    """Leave a load on the track support and the needle only: the guess then solves the whole frozen system."""
    support = {dof // 3 for dof in packet.manager.support}
    free = np.flatnonzero(packet.fixed == 0)
    for index, node in enumerate(free):
        if index not in support:
            packet.rhs[node] = 0
    packet.pre.rhs.assign(packet.rhs)


def launched_correction(packet, compact):
    """Kernels launched by one device correction, its coefficients and its flag."""
    result = []
    names = fixture._launched(lambda: result.append(correction(packet, compact=compact)))
    return names, *result[0]


def guess_residual(packet, operator):
    """|b - A z| / |b| of the certified start in the FP64 dense operator, and whether the certificate took it."""
    packet.helper.initialize()
    z = packet.pre.z.numpy().astype(float).reshape(-1)
    b = packet.pre.rhs.numpy().astype(float).reshape(-1)
    return float(np.linalg.norm(operator @ z - b) / np.linalg.norm(b)), int(packet.helper.selected.numpy()[0])


class ShearedCoefficientTests(unittest.TestCase):
    def test_device_coefficients_are_the_host_solve_of_the_nonsymmetric_system_on_both_paths(self):
        # Condition of the system of every correction: the compacted ones, and the padded ones above 38 samples.
        conditions = {True: [], False: []}
        sliding_cases = 0
        cases = [(scale, name) for scale in SCALES for name in ("mixed", "sliding")]
        # No sliding anchor in the first; anchors without a tangent at both ends in the second.
        cases += [(0.1, "sticking"), (0.1, "disengaged ends")]
        for tracks in range(1, _MAX_TRACKS + 1):
            for scale, name in cases:
                with self.subTest(tracks=tracks, scale=scale, pattern=name):
                    ranks = fixture._patterns(tracks)[name]
                    packet = tunnel(900 + tracks, tracks, scale=scale, ranks=ranks)
                    helper, pre = packet.helper, packet.pre
                    self.assertTrue(pre.nonsymmetric)
                    sliding = int(np.any(packet.shears, axis=(1, 2)).sum())
                    self.assertEqual(sliding, sum(np.linalg.matrix_rank(m, tol=1e-5) == 2 for m in packet.matrices))
                    # The host solve of the nonsymmetric system, and of the same system without the shear.
                    expected = packet.host()
                    symmetric = tunnel(900 + tracks, tracks, scale=scale, ranks=ranks, friction=None).host()
                    effect = distance(symmetric, expected, helper)
                    sliding_cases += int(sliding > 0)
                    self.assertEqual(effect > 0.0, sliding > 0)
                    # The branch of a captured graph, and the padded system where that was the compacted one.
                    for compact in (None, False):
                        if compact is False and not fixture._compacted(helper.device_solve):
                            continue
                        actual, enabled = correction(packet, compact=compact)
                        solve = helper.device_solve
                        above = 3 * solve.capacity > _COMPACT_ROWS
                        compacted = compact is None and fixture._compacted(solve)
                        self.assertEqual(enabled, 1)
                        self.assertEqual(solve.solve_valid.numpy()[0], 1)
                        system, normal = systems(helper, compacted)
                        if above:
                            conditions[compacted].append(condition(system))
                        # The right factor is (I + S) F to FP32 rounding, and F itself without a shear.
                        factors = solve.factors.numpy()[:tracks]
                        rights = np.asarray(sheared(factors.astype(float), packet.shears))
                        bound = 4 * EPS * np.abs(rights).max(initial=1.0)
                        np.testing.assert_allclose(solve.rights.numpy()[:tracks], rights, rtol=0, atol=bound)
                        still = ~np.any(packet.shears, axis=(1, 2))
                        np.testing.assert_array_equal(solve.rights.numpy()[:tracks][still], factors[still])
                        # Forward error bound of an FP32 Cholesky solve of the normal equations.
                        tolerance = fixture._bound(normal, 3 * tracks)
                        error = distance(actual, expected, helper)
                        self.assertLessEqual(error, tolerance)
                        if sliding:
                            # Ten times nearer to the solve with the shear than the solve without it is.
                            self.assertLess(10 * error, effect)
                        # The proposal passes the residual certificate of the current operator.
                        helper.initialize()
                        self.assertEqual(helper.selected.numpy()[0], 1)
        # Both device paths were taken above 38 samples, each up to the condition of the demo's system and
        # beyond: seven of the eight packets of a track count fit the compacted tile (not the sticking one).
        above = _MAX_TRACKS - _COMPACT_ROWS // 3
        self.assertEqual(len(conditions[True]), 7 * above)
        self.assertEqual(len(conditions[False]), 8 * above)
        self.assertGreater(max(conditions[True]), DEMO_CONDITION)
        self.assertGreater(max(conditions[False]), DEMO_CONDITION)
        # The mixed and the sliding pattern have sliding anchors at every track count and scale.
        self.assertGreaterEqual(sliding_cases, 6 * _MAX_TRACKS)

    def test_guess_of_a_load_on_the_track_support_solves_the_sheared_operator(self):
        for scale in SCALES:
            for tracks in (1, 5, 8, 16, 24, _COMPACT_ROWS // 3, 40, _MAX_TRACKS):
                for name in ("mixed", "sliding"):
                    with self.subTest(tracks=tracks, scale=scale, pattern=name):
                        ranks = fixture._patterns(tracks)[name]
                        packet = tunnel(900 + tracks, tracks, scale=scale, ranks=ranks)
                        plain = tunnel(900 + tracks, tracks, scale=scale, ranks=ranks, friction=None)
                        supported_load(packet)
                        supported_load(plain)
                        operator = packet.matrix
                        self.assertGreater(np.abs(operator - operator.T).max(), 0.0)
                        # Host and device guess of the nonsymmetric system: below the tolerance of the
                        # solve they start, which then takes no iteration.
                        packet.host()
                        residual, selected = guess_residual(packet, operator)
                        self.assertEqual(selected, 1)
                        self.assertLess(residual, SOLVE_TOLERANCE)
                        for compact in (None, False):
                            self.assertEqual(correction(packet, compact=compact)[1], 1)
                            residual, selected = guess_residual(packet, operator)
                            self.assertEqual(selected, 1)
                            self.assertLess(residual, SOLVE_TOLERANCE)
                        # The guess of the system without the shear, on the same sheared operator: taken
                        # by the certificate and above that tolerance, so the solve has to iterate.
                        plain.matrix_device.assign(operator)
                        self.assertEqual(correction(plain)[1], 1)
                        residual, selected = guess_residual(plain, operator)
                        self.assertEqual(selected, 1)
                        self.assertGreater(residual, SOLVE_TOLERANCE)

    def test_host_packet_holds_the_sheared_tangent(self):
        packet = tunnel(77, 9)
        packet.host()
        records = packet.helper.packet.numpy()[_HEADER : _HEADER + 20 * 9].reshape(9, 20)
        expected = np.asarray(sheared(packet.matrices.astype(float), packet.shears))
        bound = 4 * EPS * np.abs(expected).max()
        np.testing.assert_allclose(records[:, 8:17].reshape(9, 3, 3), expected, rtol=0, atol=bound)
        sliding = np.flatnonzero(np.any(packet.shears, axis=(1, 2)))
        self.assertEqual(len(sliding), 3)
        for index in sliding:
            # S D = friction * sign * a * t d.T for the largest eigenvalue a of D; the largest entry of the
            # outer product of two unit vectors is at least a third.
            change = np.abs(expected[index] - packet.matrices[index]).max()
            largest = np.linalg.eigvalsh(packet.matrices[index].astype(float)).max()
            self.assertGreater(change, 0.999 * FRICTION * largest / 3)
        # Without a shear the packet holds the tangent itself.
        plain = tunnel(77, 9, friction=None)
        plain.host()
        records = plain.helper.packet.numpy()[_HEADER : _HEADER + 20 * 9].reshape(9, 20)
        np.testing.assert_array_equal(records[:, 8:17].reshape(9, 3, 3), plain.matrices)

    def test_key_of_the_device_solve_holds_the_anchor_law(self):
        packet = tunnel(31, 6)
        key = packet.helper.prepare_device(packet.manager)
        self.assertIsNotNone(key)
        packet.pre.nonsymmetric = False
        other = packet.helper.prepare_device(packet.manager)
        self.assertNotEqual(other, key)
        packet.pre.nonsymmetric = True
        self.assertEqual(packet.helper.prepare_device(packet.manager), key)


class ZeroShearTests(unittest.TestCase):
    def test_friction_tunnel_without_a_shear_gives_the_symmetric_coefficients_to_rounding(self):
        above = 0
        for tracks in range(1, _MAX_TRACKS + 1):
            with self.subTest(tracks=tracks):
                # A friction tunnel whose anchors do not slide (here: no friction), and no friction tunnel.
                zero = tunnel(600 + tracks, tracks, scale=0.1, friction=0.0)
                plain = tunnel(600 + tracks, tracks, scale=0.1, friction=None)
                self.assertTrue(zero.pre.nonsymmetric)
                self.assertFalse(plain.pre.nonsymmetric)
                np.testing.assert_array_equal(zero.pre.shears.numpy(), 0)
                expected = plain.host()
                np.testing.assert_array_equal(zero.host(), expected)
                for compact in (None, False):
                    if compact is False and not fixture._compacted(zero.helper.device_solve):
                        continue
                    names, actual, enabled = launched_correction(zero, compact)
                    solve = zero.helper.device_solve
                    above += int(compact is False)
                    compacted = compact is None and fixture._compacted(solve)
                    self.assertEqual(names.count("schur_normal"), 1)
                    plain_names, reference, flag = launched_correction(plain, compact)
                    self.assertNotIn("schur_normal", plain_names)
                    self.assertEqual((enabled, flag), (1, 1))
                    other = plain.helper.device_solve
                    # The same factors, the right factor equal to them, and the same system bit for bit.
                    self.assertEqual(solve.rights.numpy().tobytes(), solve.factors.numpy().tobytes())
                    self.assertEqual(solve.factors.numpy().tobytes(), other.factors.numpy().tobytes())
                    system, normal = systems(zero.helper, compacted)
                    name = "compact_matrix" if compacted else "matrix"
                    self.assertEqual(getattr(solve, name).numpy().tobytes(), getattr(other, name).numpy().tobytes())
                    # Only the solve differs: the normal equations against the system itself.
                    bound = fixture._bound(normal, 3 * tracks) + fixture._bound(system, 3 * tracks)
                    self.assertLessEqual(distance(actual, reference, zero.helper), bound)
                    self.assertLessEqual(distance(actual, expected, zero.helper), fixture._bound(normal, 3 * tracks))
        self.assertEqual(above, _MAX_TRACKS - _COMPACT_ROWS // 3)


# ---------------------------------------------------------------------------------------------
# The text of the kernels before the shear, rebuilt from their source by undoing the edit.

_SHEARS = (r"    shears: wp\.array\[wp\.mat33\],\n", "", 1)
_RIGHTS = (r"    rights: wp\.array\[wp\.mat33\],\n", "", 1)
_EARLIER_TEXT = {
    "schur_inputs": (
        woodbury_device,
        [
            _SHEARS,
            _RIGHTS,
            (r"D = F F\.T and \(I \+ S\) F per track", "D = F F.T per track", 1),
            (
                r"        # The anchor acts as[^\n]*\n        rights\[track\] = factor \+ shears\[track\] \* factor\n",
                "",
                1,
            ),
        ],
    ),
    "schur_assemble": (
        woodbury_device,
        [
            _RIGHTS,
            (r"I \+ h F\.T C \(I \+ S\) F and", "I + h F.T C F and", 1),
            (r"\* c \* rights\[j\]", "* c * factors[j]", 1),
        ],
    ),
    "schur_compact_assemble": (woodbury_device, [_RIGHTS, (r"\* c \* rights\[j\]", "* c * factors[j]", 1)]),
    "schur_coefficients": (
        woodbury_device,
        [
            (r"    rights: wp\.array\[wp\.mat33\],\n", "    factors: wp.array[wp.mat33],\n", 1),
            (r"per support column, w = \(I \+ S\) F y, then", "per support column, then", 1),
            (r"weight = rights\[track\]", "weight = factors[track]", 2),
            (
                r"its zero column of the right factor\n        # times the padded right-hand side that "
                r"``schur_compact_scatter``\n        # returns for it; a nonfinite shear leaves a nonfinite right factor\.",
                "its zero factor column times the padded\n"
                "        # right-hand side that ``schur_compact_scatter`` returns for it.",
                1,
            ),
        ],
    ),
    "pack_tracks": (
        woodbury_guess,
        [
            _SHEARS,
            (
                r"    # The whole anchor tangent[^\n]*\n    tangent = matrices\[i\] \+ shears\[i\] \* matrices\[i\]\n",
                "",
                1,
            ),
            (r"= tangent\[row, col\]", "= matrices[i][row, col]", 1),
        ],
    ),
}


def earlier_text(name):
    """Source of a kernel as it was before the shear."""
    module, rules = _EARLIER_TEXT[name]
    text = inspect.getsource(getattr(module, name).func)
    for pattern, replacement, places in rules:
        text, found = re.subn(pattern, replacement, text)
        if found != places:
            raise AssertionError(f"{name}: {pattern!r} matches {found} places, expected {places}")
    if re.search(r"shear|rights|tangent\[|\(I \+ S\)", text):
        raise AssertionError(f"{name}: the earlier text still holds a shear or a right factor")
    return text


_compiled = {}


def earlier_kernels():
    """Compile the earlier text of every changed kernel next to the globals of its own module."""
    if not _compiled:
        namespaces = {}
        for name, (module, _) in _EARLIER_TEXT.items():
            namespace = namespaces.get(module)
            if namespace is None:
                short = module.__name__.rsplit(".", 1)[-1]
                namespace = namespaces[module] = dict(vars(module), __name__=f"{__name__}.{short}")
            text = earlier_text(name)
            filename = f"<{namespace['__name__']}.{name}>"
            # Warp reads a kernel's source through inspect, which serves this text from the line cache.
            linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
            exec(compile(text, filename, "exec"), namespace)
            _compiled[name] = namespace[name]
    return _compiled


def with_earlier_kernels(solve, call):
    """Run ``call`` with every changed kernel replaced by its earlier text; returns the kernels replaced."""
    kernels = earlier_kernels()
    replaced = []
    real = wp.launch

    def launch(kernel, dim, inputs, *args, **kwargs):
        earlier = kernels.get(kernel.key)
        if earlier is None:
            return real(kernel, dim, inputs, *args, **kwargs)
        parameters = list(inspect.signature(kernel.func).parameters)
        if len(parameters) != len(inputs):
            raise AssertionError(f"{kernel.key}: {len(inputs)} arguments for {len(parameters)} parameters")
        values = dict(zip(parameters, inputs, strict=True))
        # Before, the coefficients took the tangent factors where they now take the right factors.
        values["factors"] = solve.factors
        replaced.append(kernel.key)
        earlier_inputs = [values[parameter] for parameter in inspect.signature(earlier.func).parameters]
        return real(earlier, dim, earlier_inputs, *args, **kwargs)

    with patch.object(wp, "launch", side_effect=launch):
        call()
    return replaced


def written(packet):
    """Every array the host packet and the device coefficients write, but the right factors."""
    arrays = fixture._snapshot(packet.helper)
    del arrays["rights"]
    arrays["packet"] = packet.helper.packet.numpy().copy()
    return arrays


class EarlierTextTests(unittest.TestCase):
    def test_symmetric_solve_holds_the_bits_of_the_kernels_before_the_shear(self):
        earlier_kernels()
        # Above 38 samples a second packet of full-rank tangents, whose columns do not fit the compacted tile.
        cases = [(tracks, ranks) for tracks in range(1, _MAX_TRACKS + 1) for ranks in ((2, 3, 3, 0, 2), (3,))]
        cases = [(tracks, ranks) for tracks, ranks in cases if 3 * tracks > _COMPACT_ROWS or len(ranks) > 1]
        compacted = usable = 0
        for tracks, ranks in cases:
            with self.subTest(tracks=tracks, ranks=ranks):
                packet = fixture.Packet(200 + tracks, tracks, surface=bool(tracks % 2), ranks=ranks)
                helper = packet.helper
                self.assertFalse(packet.pre.nonsymmetric)
                self.assertIsNotNone(helper.prepare_device(packet.manager))
                solve = helper.device_solve

                def launches(packet=packet, solve=solve):
                    packet.host()
                    solve.enqueue_topology()
                    fixture._correction(solve)

                fixture._reset(helper)
                helper.packet.fill_(-7.25)
                names = fixture._launched(launches)
                self.assertNotIn("schur_normal", names)
                actual = written(packet)
                self.assertEqual(actual["enabled"].tolist(), [1])
                compacted += int(fixture._compacted(solve))
                # Without a shear the right factors are the tangent factors.
                self.assertEqual(solve.rights.numpy()[:tracks].tobytes(), solve.factors.numpy()[:tracks].tobytes())
                fixture._reset(helper)
                helper.packet.fill_(-7.25)
                replaced = with_earlier_kernels(solve, launches)
                expected_names = [name for name in names if name in _EARLIER_TEXT]
                self.assertEqual(replaced, expected_names)
                self.assertEqual(len(replaced), 4)
                earlier = written(packet)
                self.assertEqual(sorted(actual), sorted(earlier))
                for name in actual:
                    self.assertEqual(actual[name].tobytes(), earlier[name].tobytes(), f"{tracks} samples: {name}")
                columns = used(helper)
                usable += int(np.isfinite(actual["coefficients"][columns]).all())
                # The earlier text writes no right factor, and neither solve the normal equations.
                np.testing.assert_array_equal(solve.rights.numpy(), -7.25)
                for name in ("gram", "normal_rhs", "compact_gram", "compact_normal_rhs"):
                    np.testing.assert_array_equal(actual[name], -7.25, name)
        self.assertEqual(usable, len(cases))
        self.assertEqual(compacted, _MAX_TRACKS - _COMPACT_ROWS // 3)


# ---------------------------------------------------------------------------------------------
# The normal equations.


def normal_equations(matrix, rhs, rows):
    """K.T K and K.T b of the leading ``rows`` rows as the kernel forms them: FP64 sums in row order, one rounding."""
    k = matrix[:rows, :rows].astype(np.float64)
    gram = np.zeros((rows, rows))
    right = np.zeros(rows)
    for row in range(rows):
        gram += np.outer(k[row], k[row])
        right += k[row] * float(rhs[row])
    return gram.astype(np.float32), right.astype(np.float32)


class NormalEquationTests(unittest.TestCase):
    def launch(self, width, count, entry, per, matrix, rhs):
        device = wp.get_device(DEVICE)
        gram = wp.full((width, width), -7.25, dtype=float, device=device)
        normal_rhs = wp.full(width, -7.25, dtype=float, device=device)
        wp.launch(
            schur_normal,
            (width + 1, width),
            [
                wp.array(count, dtype=int, device=device),
                entry,
                per,
                wp.array(matrix, dtype=float, device=device),
                wp.array(rhs, dtype=float, device=device),
                gram,
                normal_rhs,
            ],
            device=device,
            block_dim=_LAUNCH_BLOCK_DIM,
        )
        return gram.numpy(), normal_rhs.numpy()

    def test_normal_equations_are_exact_sums_on_the_rows_in_use_and_an_identity_beyond(self):
        rng = np.random.default_rng(3)
        live = np.zeros(3, dtype=np.int32)
        cases = [
            (3 * _MAX_TRACKS, tracks, LIVE_TRACK, 3) for tracks in (0, 1, 5, 17, _COMPACT_ROWS // 3, 40, _MAX_TRACKS)
        ]
        cases += [(_COMPACT_ROWS, columns, 0, 1) for columns in (0, 1, 2, 77, _COMPACT_ROWS - 1, _COMPACT_ROWS)]
        for width, number, entry, per in cases:
            with self.subTest(width=width, number=number):
                rows = per * number
                # A nonsymmetric system on the rows in use. Beyond them the kernel reads the right-hand
                # side only: the matrix there is poisoned.
                matrix = np.full((width, width), np.nan, dtype=np.float32)
                matrix[:rows, :rows] = np.eye(rows) + rng.normal(0, 1.0, (rows, rows))
                rhs = rng.normal(0, 1.0, width).astype(np.float32)
                count = live.copy() if per == 3 else np.zeros(1, dtype=np.int32)
                count[entry] = number
                gram, right = self.launch(width, count, entry, per, matrix, rhs)
                expected, expected_right = normal_equations(matrix, rhs, rows)
                self.assertEqual(gram[:rows, :rows].tobytes(), expected.tobytes())
                self.assertEqual(right[:rows].tobytes(), expected_right.tobytes())
                # Both triangles hold the same bits; beyond the rows in use an identity and the right-hand side.
                self.assertEqual(gram.tobytes(), gram.T.copy().tobytes())
                np.testing.assert_array_equal(gram[rows:, rows:], np.eye(width - rows))
                np.testing.assert_array_equal(gram[rows:, :rows], 0)
                self.assertEqual(right[rows:].tobytes(), rhs[rows:].tobytes())
                if rows > 1:
                    self.assertGreater(np.abs(matrix[:rows, :rows] - matrix[:rows, :rows].T).max(), 0.1)
                if rows:
                    # The device count decides, not the launch: one sample or column fewer.
                    count[entry] = number - 1
                    lower = per * (number - 1)
                    gram, right = self.launch(width, count, entry, per, matrix, rhs)
                    expected, expected_right = normal_equations(matrix, rhs, lower)
                    self.assertEqual(gram[:lower, :lower].tobytes(), expected.tobytes())
                    np.testing.assert_array_equal(gram[lower:, lower:], np.eye(width - lower))
                    self.assertEqual(right[:lower].tobytes(), expected_right.tobytes())
                    self.assertEqual(right[lower:].tobytes(), rhs[lower:].tobytes())

    def test_scratch_of_the_normal_equations_is_built_and_its_kernel_loaded_with_the_helper(self):
        # Fitting and not fitting columns at 40 samples: both branches of the conditional.
        for ranks, compacted in (((2, 3, 3, 0, 2), True), ((3,), False)):
            with self.subTest(compacted=compacted):
                packet = tunnel(651, 40, ranks=ranks)
                helper = packet.helper
                self.assertIsNotNone(helper.prepare_device(packet.manager))
                solve = helper.device_solve
                self.assertIs(schur_normal.module, woodbury_device.schur_compact_index.module)
                self.assertEqual(solve.gram.shape, (3 * _MAX_TRACKS, 3 * _MAX_TRACKS))
                self.assertEqual(solve.compact_gram.shape, (_COMPACT_ROWS, _COMPACT_ROWS))
                for name in ("rights", "gram", "normal_rhs", "compact_gram", "compact_normal_rhs"):
                    self.assertTrue(any(owner is getattr(solve, name) for owner in solve.owners), name)
                modules = (schur_normal.module, solve.compact_kernel.module, solve.solve_kernel.module)
                self.assertEqual(len({id(module) for module in modules}), 3)
                device = packet.pre.device
                self.assertIn((device.context, _LAUNCH_BLOCK_DIM if device.is_cuda else 1), schur_normal.module.execs)
                loaded = [set(module.execs) for module in modules]
                blocks, sizes = {}, {}
                real = wp.launch

                def launch(kernel, dim, *args, _blocks=blocks, _sizes=sizes, _real=real, **kwargs):
                    _blocks[kernel.key] = kwargs.get("block_dim")
                    _sizes[kernel.key] = dim
                    return _real(kernel, dim, *args, **kwargs)

                with patch.object(wp, "launch", side_effect=launch):
                    solve.enqueue_topology()
                    fixture._correction(solve)
                self.assertEqual(fixture._compacted(solve), compacted)
                self.assertEqual(helper.enabled.numpy()[0], 1)
                # Every launch names the thread block that the module was loaded for with the scratch, the
                # normal equations have the size of the scratch of their system, and no launch loads a module.
                names = {"schur_topology", "schur_inputs", "schur_compact_index", "schur_coefficients", "schur_normal"}
                names |= {"schur_compact_assemble", "schur_compact_scatter"} if compacted else {"schur_assemble"}
                self.assertEqual(blocks, dict.fromkeys(names, _LAUNCH_BLOCK_DIM))
                width = _COMPACT_ROWS if compacted else 3 * _MAX_TRACKS
                self.assertEqual(sizes["schur_normal"], (width + 1, width))
                self.assertEqual([set(module.execs) for module in modules], loaded)


class ValidityTests(unittest.TestCase):
    def test_nonfinite_shear_clears_the_flag_on_both_paths(self):
        # Sample 0 slides (rank two); sample 3 is disengaged: its tangent and its factor are zero.
        for tracks in (6, 40):
            for sample in (0, 3):
                for value in (np.inf, np.nan):
                    with self.subTest(tracks=tracks, sample=sample, value=value):
                        packet = tunnel(40 + tracks, tracks)
                        self.assertEqual(
                            np.linalg.matrix_rank(packet.matrices[sample], tol=1e-5), 2 if sample == 0 else 0
                        )
                        shears = packet.shears.copy()
                        shears[sample, 1, 2] = value
                        packet.pre.shears.assign(shears)
                        for compact in (None, False):
                            self.assertEqual(correction(packet, compact=compact)[1], 0)
                            packet.pre.z.fill_(wp.vec3(1.0))
                            packet.helper.initialize()
                            self.assertEqual(packet.helper.selected.numpy()[0], 0)
                            np.testing.assert_array_equal(packet.pre.z.numpy(), 0)
                        self.assertEqual(fixture._compacted(packet.helper.device_solve), tracks == 40)
                        # Restored inputs are accepted again by the same persistent scratch.
                        packet.pre.shears.assign(packet.shears)
                        self.assertEqual(correction(packet)[1], 1)

    def test_indefinite_system_of_a_friction_tunnel_is_solved_and_left_to_the_certificate(self):
        def flipped(friction):
            # A sign-flipped response makes the system indefinite while every input stays finite.
            packet = tunnel(28, 6, friction=friction)
            flat = packet.manager.core.basis.numpy()
            flat[: packet.response.size] *= -1e3
            packet.manager.core.basis.assign(flat)
            packet.manager.basis_builds += 1
            return packet

        # No friction tunnel: the Cholesky factorization of the symmetric system fails and clears the flag.
        packet = flipped(None)
        self.assertEqual(correction(packet)[1], 0)
        np.testing.assert_array_equal(packet.helper.device_solve.solve_valid.numpy(), [0])
        # Friction tunnel: the normal equations of a nonsingular system are positive definite. The proposal
        # is the one of the host's general solve, and the certificate of the current operator rejects it.
        packet = flipped(FRICTION)
        expected = packet.host()
        actual, enabled = correction(packet)
        solve = packet.helper.device_solve
        self.assertEqual(enabled, 1)
        np.testing.assert_array_equal(solve.solve_valid.numpy(), [1])
        system, normal = systems(packet.helper, False)
        self.assertLess(np.linalg.eigvalsh(0.5 * (system + system.T).astype(float)).min(), -1.0)
        self.assertGreater(np.linalg.eigvalsh(normal.astype(float)).min(), 0.0)
        self.assertLessEqual(distance(actual, expected, packet.helper), fixture._bound(normal, 18))
        packet.pre.z.fill_(wp.vec3(1.0))
        packet.helper.initialize()
        self.assertEqual(packet.helper.selected.numpy()[0], 0)
        np.testing.assert_array_equal(packet.pre.z.numpy(), 0)

    def test_finite_system_beyond_the_range_of_the_normal_equations_clears_the_flag_and_keeps_the_zero_start(self):
        # Tissue and needle 1e5 times as compliant as the fixture's default, every anchor sliding:
        # cond(K) is about 1e6, and its square is far beyond what FP32 holds (1 / roundoff is 1.7e7).
        for tracks in (38, 40):
            with self.subTest(tracks=tracks):
                packet = tunnel(900 + tracks, tracks, scale=1.0e-5, ranks=(2,))
                # Every input is finite, and the host's FP64 general solve takes the system as it is.
                self.assertTrue(np.isfinite(packet.host()).all())
                for compact in (None, False):
                    self.assertEqual(correction(packet, compact=compact)[1], 0)
                    solve = packet.helper.device_solve
                    compacted = compact is None and fixture._compacted(solve)
                    self.assertEqual(compacted, compact is None and tracks == 40)
                    np.testing.assert_array_equal(solve.solve_valid.numpy(), [0])
                    np.testing.assert_array_equal(solve.factor_valid.numpy()[:tracks], 1)
                    system, _ = systems(packet.helper, compacted)
                    self.assertTrue(np.isfinite(system).all())
                    self.assertGreater(condition(system) ** 2, 100.0 / EPS)
                    # The zero start, marked as a discarded guess: the solve starts from zero.
                    packet.pre.z.fill_(wp.vec3(1.0))
                    packet.helper.initialize()
                    self.assertEqual(packet.helper.selected.numpy()[0], 0)
                    np.testing.assert_array_equal(packet.pre.z.numpy(), 0)
                # The loss is that of the normal equations: the same tangents without a friction tunnel
                # are factorized as the system itself and give a guess that the certificate takes.
                plain = tunnel(900 + tracks, tracks, scale=1.0e-5, ranks=(2,), friction=None)
                self.assertEqual(correction(plain)[1], 1)
                plain.helper.initialize()
                self.assertEqual(plain.helper.selected.numpy()[0], 1)


@unittest.skipUnless(DEVICE.startswith("cuda"), "Conditional CUDA graph replay contract")
class CapturedShearTests(unittest.TestCase):
    def test_captured_correction_of_a_friction_tunnel_follows_the_device_shears_counts_and_columns(self):
        # One graph per padded tile: 5 to 8 samples, 17 to 38 (no compaction), 39 and 40, 41 to 48.
        for tracks, fewest in ((8, 5), (_COMPACT_ROWS // 3, 17), (40, 39), (48, 41)):
            with self.subTest(tracks=tracks):
                packet = tunnel(81 + tracks, tracks, scale=0.1)
                helper, pre = packet.helper, packet.pre
                key = helper.prepare_device(packet.manager)
                self.assertIsNotNone(key)
                solve, live = helper.device_solve, pre.coupler.interaction._live
                self.assertEqual(solve.capacity, tracks)
                remaining = wp.zeros(1, dtype=int, device=pre.device)
                applied = wp.ones(1, dtype=int, device=pre.device)

                def set_tracks(count, live=live, pre=pre):
                    counts = live.numpy()
                    counts[LIVE_TRACK] = count
                    live.assign(counts)
                    pre.count = count

                def trial(solve=solve, applied=applied, remaining=remaining, pre=pre):
                    # As in the trial loop: the correction is itself a branch of a loop body.
                    wp.capture_if(applied, solve.enqueue)
                    wp.launch(fixture.countdown, 1, [remaining], device=pre.device)

                # Captured with the fewest samples of this factorization.
                set_tracks(fewest)
                with ScopedCapture(device=pre.device) as capture:
                    solve.enqueue_topology()
                    wp.capture_while(remaining, trial)
                sticking = np.asarray([fixture._tangent(np.random.default_rng(7), 3) for _ in range(tracks)])
                indexed = 3 * solve.capacity > _COMPACT_ROWS
                still = np.zeros_like(packet.shears)
                seen = []
                for count, matrices, shears, compacted in (
                    (fewest, packet.matrices, packet.shears, indexed),
                    (tracks, packet.matrices, packet.shears, indexed),
                    # The anchors stop sliding: the same graph, the same tangents, no shear.
                    (tracks, packet.matrices, still, indexed),
                    # Sticking anchors have no shear and more columns than the compacted tile holds.
                    (tracks, sticking, still, False),
                    (tracks, packet.matrices, packet.shears, indexed),
                ):
                    with self.subTest(count=count, compacted=compacted, sliding=bool(shears.any())):
                        set_tracks(count)
                        pre.matrices.assign(matrices)
                        pre.shears.assign(shears)
                        fixture._reset(helper)
                        remaining.fill_(2)
                        wp.capture_launch(capture.graph)
                        replayed = fixture._snapshot(helper)
                        self.assertEqual(replayed["enabled"].tolist(), [1])
                        self.assertEqual(replayed["solve_valid"].tolist(), [1])
                        self.assertEqual(replayed["compact_fits"].tolist(), [int(compacted) if indexed else -5])
                        self.assertEqual(fixture._compacted(solve), compacted)
                        # The normal equations of the system that was factorized, and of no other.
                        normal, other = ("compact_gram", "gram") if compacted else ("gram", "compact_gram")
                        self.assertFalse((replayed[normal] == -7.25).any())
                        self.assertTrue((replayed[other] == -7.25).all())
                        # The replay is the proposal of the host solve at this count, these tangents and shears.
                        system, gram = systems(helper, compacted)
                        expected = packet.host()
                        error = distance(replayed["coefficients"], expected, helper)
                        self.assertLessEqual(error, fixture._bound(gram, 3 * count))
                        if shears.any():
                            # And not the proposal of the same anchors without their shear.
                            pre.shears.assign(still)
                            effect = distance(packet.host(), expected, helper)
                            pre.shears.assign(shears)
                            self.assertLess(10 * error, effect)
                            self.assertGreater(np.abs(system - system.T).max(), 0.0)
                        # And it holds the bits of the same launches issued one by one, with its branch.
                        fixture._reset(helper)
                        solve.enqueue_topology()
                        fixture._correction(solve)
                        direct = fixture._snapshot(helper)
                        self.assertEqual(sorted(replayed), sorted(direct))
                        for name in replayed:
                            self.assertEqual(replayed[name].tobytes(), direct[name].tobytes(), name)
                        seen.append(replayed["coefficients"].tobytes())
                # The replays differ with the count, the shears and the tangents, and repeat with them.
                self.assertEqual(len(set(seen)), 4)
                self.assertEqual(seen[1], seen[4])
                set_tracks(tracks)
                self.assertEqual(helper.prepare_device(packet.manager), key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
