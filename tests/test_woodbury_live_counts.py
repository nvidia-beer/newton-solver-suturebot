# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""The start-guess kernels read their counts from device arrays and are launched at the size of their scratch.

Before, each kernel took the number of support columns and of track samples as host integers,
and the launches had exactly those sizes. That text is rebuilt here from each kernel's source by
undoing the edit. Both texts are launched from the same start values, the earlier one at its
earlier size: every array must hold the same bits afterwards, also with a count lowered by one.

Above 38 samples a captured correction factorizes the Schur system on its nonzero tangent-factor
columns when they fit the compacted tile. That step has no earlier text: the comparisons issue the
branch that a captured graph takes, and ``schur_assemble`` is compared where that branch launches
it, on the padded system.

From 17 to 38 samples the padded system is factorized in the tile of the compacted system (114
rows, 256 block threads), where it was a tile of 96 rows up to 32 samples and of 120 rows above,
both with 128 block threads. Another tile rounds differently, so that step is compared with the
earlier tiles within the bound of the solve, not bit for bit.

The kernels take the shear of a sliding anchor of a friction tunnel since: ``schur_inputs`` also
writes the right factors (I + S) F, an output more than before, and for a track that follows a
friction tunnel the normal equations of the system are formed in front of its factorization. That
step has no earlier text either. The references here follow it: the earlier texts are those of the
kernels with the shear, the normal equations are issued as the package issues them, and every
comparison also runs on packets of a friction tunnel with sliding anchors. That the kernels write
the bits they wrote before the shear, where no anchor law has one, is ``test_woodbury_shear``.
"""

import inspect
import linecache
import os
import re
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_woodbury_device as fixture
import warp as wp

from newton_solver_suturebot._src.solvers.coupled.acceleration import woodbury_device, woodbury_guess
from newton_solver_suturebot._src.solvers.coupled.acceleration.woodbury_device import _COMPACT_ROWS, _RIGID, _TWIST
from newton_solver_suturebot._src.solvers.coupled.acceleration.woodbury_guess import _CAPACITY, _MAX_TRACKS
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import LIVE_TRACK
from newton_solver_suturebot._src.utils.capture import ScopedCapture

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
# Friction coefficient of the packets that follow a friction tunnel.
FRICTION = 0.2

# How each kernel got its counts before: (pattern, earlier text, number of places).
_BOTH_COUNTS = r"    support_count: wp\.array\[int\],\n    live: wp\.array\[int\],\n"
_READ_BOTH = (r"    count = support_count\[0\]\n    tracks = live\[LIVE_TRACK\]\n", "", 1)
_SUPPORT_PARAMETER = (r"    support_count: wp\.array\[int\],\n", "    count: int,\n", 1)
_SUPPORT_LOOP = (r"range\(support_count\[0\]\)", "range(count)", 1)

_EARLIER_TEXT = {
    "schur_topology": (
        woodbury_device,
        [
            (_BOTH_COUNTS, "    count: int,\n", 1),
            # The comment on the launch size and the guards.
            (r"    # Launched over the whole scratch:[^\n]*\n    #[^\n]*\n", "", 1),
            (r"    if track >= live\[LIVE_TRACK\]:\n        return\n", "", 1),
            (
                r"    if column < _CAPACITY:\n        if column >= support_count\[0\]:\n            return\n",
                "    if column < count:\n",
                1,
            ),
            (r"column - _CAPACITY", "column - count", 1),
            (r"        if other_track >= live\[LIVE_TRACK\]:\n            return\n", "", 1),
        ],
    ),
    "schur_inputs": (woodbury_device, [(_BOTH_COUNTS, "    count: int,\n    tracks: int,\n", 1), _READ_BOTH]),
    "schur_assemble": (
        woodbury_device,
        [
            (r"    live: wp\.array\[int\],\n", "    tracks: int,\n", 1),
            (r"    tracks = live\[LIVE_TRACK\]\n", "", 1),
        ],
    ),
    "schur_coefficients": (woodbury_device, [(_BOTH_COUNTS, "    count: int,\n    tracks: int,\n", 1), _READ_BOTH]),
    "pack_head": (woodbury_guess, [_SUPPORT_PARAMETER, _SUPPORT_LOOP]),
    "pack_tracks": (
        woodbury_guess,
        [
            (r"    live: wp\.array\[int\],\n", "", 1),
            (r"    if i >= live\[LIVE_TRACK\]:\n        return\n", "", 1),
        ],
    ),
    "build_guess": (woodbury_guess, [_SUPPORT_PARAMETER, _SUPPORT_LOOP]),
    "collect_iterations": (
        woodbury_guess,
        [
            (r"    active: wp\.array\[int\],\n", "", 1),
            (r"    if active\[0\] == 0:\n        return\n", "", 1),
        ],
    ),
}

# Launch sizes of the package: the scratch, whatever the counts are.
_LAUNCH_SIZE = {
    "schur_topology": (3 * _MAX_TRACKS, _CAPACITY + 3 * _MAX_TRACKS),
    "schur_inputs": _RIGID + 1,
    "schur_assemble": (_MAX_TRACKS, _MAX_TRACKS),
    "schur_coefficients": _TWIST + 1,
    "pack_head": 1,
    "pack_tracks": _MAX_TRACKS,
    "collect_iterations": 1,
}

# Launch sizes before: (track samples, support columns, samples of the factorization) -> size.
_EARLIER_SIZE = {
    "schur_topology": lambda tracks, support, capacity: (3 * tracks, support + 3 * tracks),
    "schur_assemble": lambda tracks, support, capacity: (capacity, capacity),
    "pack_tracks": lambda tracks, support, capacity: tracks,
}

# What each kernel writes; these start from values that no kernel writes.
_OUTPUTS = {
    "schur_topology": ("tissue", "compliance", "supported", "gather"),
    "schur_inputs": (
        "response",
        "factors",
        "rights",
        "factor_valid",
        "rigid_inverse",
        "rigid_response",
        "rigid_solution",
        "rigid_valid",
    ),
    "schur_assemble": ("matrix", "rhs"),
    "schur_coefficients": ("coefficients", "surface_active", "enabled"),
    "pack_head": ("packet",),
    "pack_tracks": ("packet",),
    "build_guess": ("guess",),
    "collect_iterations": ("counters",),
}


def earlier_text(name):
    """Source of a kernel with its counts as host integers again."""
    module, rules = _EARLIER_TEXT[name]
    text = inspect.getsource(getattr(module, name).func)
    for pattern, replacement, places in rules:
        text, found = re.subn(pattern, replacement, text)
        if found != places:
            raise AssertionError(f"{name}: {pattern!r} matches {found} places, expected {places}")
    if re.search(r"\blive\b|LIVE_|support_count|\bactive\b", text):
        raise AssertionError(f"{name}: the earlier text still reads a count or a flag from an array")
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


def record(call):
    """Size and arguments of every launch of a changed kernel that ``call`` issues, in order."""
    launches = []
    real = wp.launch

    def launch(kernel, dim, inputs, *args, **kwargs):
        if kernel.key in _EARLIER_TEXT:
            launches.append((kernel, dim, list(inputs)))
        return real(kernel, dim, inputs, *args, **kwargs)

    with patch.object(wp, "launch", side_effect=launch):
        call()
    return launches


def named(kernel, inputs):
    parameters = list(inspect.signature(kernel.func).parameters)
    if len(parameters) != len(inputs):
        raise AssertionError(f"{kernel.key}: {len(inputs)} arguments for {len(parameters)} parameters")
    return dict(zip(parameters, inputs, strict=True))


def earlier_launch(name, dim, values, tracks, support, capacity):
    """Launch the earlier text of a kernel as the package launched it before: host counts, earlier size."""
    kernel = earlier_kernels()[name]
    host = {"count": support, "tracks": tracks}
    inputs = [host[p] if p in host else values[p] for p in inspect.signature(kernel.func).parameters]
    size = _EARLIER_SIZE.get(name)
    wp.launch(kernel, dim if size is None else size(tracks, support, capacity), inputs, device=DEVICE)


def untouched(array):
    """Start values that no kernel writes."""
    values = array.numpy()
    return np.full_like(values, -5 if np.issubdtype(values.dtype, np.integer) else -7.25)


def lowered(array, entry):
    values = array.numpy().copy()
    values[entry] -= 1
    return wp.array(values, dtype=int, device=array.device)


def padded_identity(matrix, rhs, rows):
    """Beyond the leading block that the factorization reads, the system is an identity with a zero right-hand side."""
    return (
        np.array_equal(matrix[rows:, rows:], np.eye(len(matrix) - rows, dtype=matrix.dtype))
        and not matrix[rows:, :rows].any()
        and not matrix[:rows, rows:].any()
        and not rhs[rows:].any()
    )


def compare(name, kernel, dim, inputs, tracks, support, capacity, *, lower=()):
    """Launch a kernel and its earlier text from the same start values; every array must hold the same bits.

    ``tracks`` and ``support`` are the host counts the earlier launch took, ``capacity`` the samples of its
    factorization. ``lower`` names the count parameters lowered by one for the kernel; the earlier text then
    gets host counts lowered by one. The arrays are left as they were. Returns the parameters whose arrays
    the kernel changed.
    """
    values = named(kernel, inputs)
    if "live" in lower:
        values["live"], tracks = lowered(values["live"], LIVE_TRACK), tracks - 1
    if "support_count" in lower:
        values["support_count"], support = lowered(values["support_count"], 0), support - 1
    if "live" in values and int(values["live"].numpy()[LIVE_TRACK]) != tracks:
        raise AssertionError(f"{name}: the live counts hold another number of track samples than the host")
    if "support_count" in values and int(values["support_count"].numpy()[0]) != support:
        raise AssertionError(f"{name}: the device holds another number of support columns than the host")
    arrays = {
        parameter: value
        for parameter, value in values.items()
        if isinstance(value, wp.array) and value.size and parameter not in ("live", "support_count", "active")
    }
    original = {parameter: value.numpy().copy() for parameter, value in arrays.items()}
    for parameter in _OUTPUTS[name]:
        arrays[parameter].assign(untouched(arrays[parameter]))
    start = {parameter: value.numpy().copy() for parameter, value in arrays.items()}
    wp.launch(kernel, dim, list(values.values()), device=DEVICE)
    written = {parameter: value.numpy().copy() for parameter, value in arrays.items()}
    for parameter, value in arrays.items():
        value.assign(start[parameter])
    earlier_launch(name, dim, values, tracks, support, capacity)
    for parameter, value in arrays.items():
        before, after = value.numpy(), written[parameter]
        if name == "schur_assemble" and parameter in ("matrix", "rhs"):
            # The earlier launch wrote the leading block that the factorization reads; the package pads the rest.
            rows = 3 * capacity
            if not padded_identity(written["matrix"], written["rhs"], rows):
                raise AssertionError(f"{name}: no identity padding beyond {rows} rows")
            before = before[:rows, :rows] if parameter == "matrix" else before[:rows]
            after = after[:rows, :rows] if parameter == "matrix" else after[:rows]
        if before.tobytes() != after.tobytes():
            raise AssertionError(
                f"{name}: {parameter} differs from the earlier text (tracks {tracks}, support {support})"
            )
        value.assign(original[parameter])
    return {parameter for parameter in arrays if written[parameter].tobytes() != start[parameter].tobytes()}


def packet_launches(packet):
    """Every changed kernel as the package launches it: host packet, device coefficients, guess, counters."""
    helper = packet.helper
    packet.pre.linear_solver = SimpleNamespace(
        _cur_iter_and_condition=wp.array([4, 0], dtype=int, device=packet.pre.device), _maxiter=80
    )

    def call():
        packet.host()
        coefficients, enabled = packet.device()
        if enabled != 1 or not np.isfinite(coefficients[: len(helper.support)]).all():
            raise AssertionError("The fixture has no usable device guess")
        helper.initialize()
        helper.after_solve()

    return record(call)


def compare_packet(packet, changed=None):
    """All changed kernels of one packet against their earlier text, at the live counts and with each lowered."""
    # The texts are checked before anything is launched: a kernel that lost a guard would read out of bounds.
    earlier_kernels()
    launches = packet_launches(packet)
    names = [kernel.key for kernel, _, _ in launches]
    helper = packet.helper
    expected = set(_EARLIER_TEXT)
    if fixture._compacted(helper.device_solve):
        # The nonzero columns were factorized alone: the padded system is not assembled.
        expected.discard("schur_assemble")
    if sorted(names) != sorted(expected):
        raise AssertionError(f"Launches of the changed kernels: {names}")
    tracks, support, capacity = packet.pre.count, len(helper.support), helper.device_solve.capacity
    interaction = packet.pre.coupler.interaction
    # A graph that holds these launches owns the two arrays the counts are read from.
    for array in (helper.core.count, interaction._live):
        if not any(owner is array for owner in helper.owners):
            raise AssertionError("The guess does not own an array that its kernels read a count from")
    for kernel, dim, inputs in launches:
        name = kernel.key
        values = named(kernel, inputs)
        # The package passes the two device arrays and launches at the size of the scratch.
        if "live" in values and values["live"] is not interaction._live:
            raise AssertionError(f"{name}: not the live counts of the interaction")
        if "support_count" in values and values["support_count"] is not helper.core.count:
            raise AssertionError(f"{name}: not the support count of the response basis")
        if "active" in values and (values["active"] is not helper.one or values["active"].numpy().tolist() != [1]):
            raise AssertionError(f"{name}: not the constant flag of the guess")
        if name in _LAUNCH_SIZE and dim != _LAUNCH_SIZE[name]:
            raise AssertionError(f"{name}: launched at {dim}, not at {_LAUNCH_SIZE[name]}")
        written = compare(name, kernel, dim, inputs, tracks, support, capacity)
        if changed is not None:
            changed.setdefault(name, set()).update(written)
        # Every count that the kernel reads, lowered by one: the kernel follows the array, not the launch.
        if "live" in values and tracks > 1:
            compare(name, kernel, dim, inputs, tracks, support, capacity, lower=("live",))
        if "support_count" in values and support > 1:
            compare(name, kernel, dim, inputs, tracks, support, capacity, lower=("support_count",))
    return len(launches)


def bounds_probe():
    """Run by a child process whose kernels are built with Warp's debug bounds checks (CPU)."""
    if wp.config.mode != "debug" or wp.get_device(DEVICE).is_cuda:
        raise AssertionError("The probe needs debug kernels on the CPU")
    launches = packets = compacted = 0
    for tracks in range(1, _MAX_TRACKS + 1):
        # Track arrays, tangents and levers have exactly ``tracks`` rows and the support list exactly its
        # columns, so a thread that passes a guard it should not pass reads out of bounds and stops the process.
        # Above 38 samples a second packet of full-rank tangents: its columns do not fit the compacted
        # tile, so the padded system is assembled there as well. Each also as a track of a friction
        # tunnel, whose sliding anchors carry a shear and whose normal equations are formed.
        for ranks in ((2, 3, 3, 0, 2), (3,))[: 1 + int(3 * tracks > _COMPACT_ROWS)]:
            for friction in (None, FRICTION):
                packet = fixture.Packet(300 + tracks, tracks, surface=bool(tracks % 2), ranks=ranks, friction=friction)
                launches += compare_packet(packet)
                packets += 1
                compacted += int(fixture._compacted(packet.helper.device_solve))
    print(f"bounds probe ok: {launches} launches of {packets} packets, {compacted} compacted")


def scratch_arrays(solve):
    # Not the size flags: the host writes them, as it writes the live counts.
    return {name: value for name, value in vars(solve).items() if isinstance(value, wp.array) and name != "size_flags"}


def reset_scratch(helper):
    """Every output of the device coefficients starts from values that no kernel writes."""
    for name, value in scratch_arrays(helper.device_solve).items():
        if name != "support_column":
            value.assign(untouched(value))
    helper.coefficients.fill_(np.nan)
    helper.enabled.fill_(-1)
    helper.surface_active.fill_(-1)


def results(helper):
    """What the device coefficients left: the proposal, its flags and the scratch behind them."""
    solve = helper.device_solve
    rows = 3 * solve.capacity
    values = {name: value.numpy().copy() for name, value in scratch_arrays(solve).items()}
    # The factorization reads the leading block; beyond it the package pads and the earlier launch wrote nothing.
    values["matrix"], values["rhs"] = values["matrix"][:rows, :rows], values["rhs"][:rows]
    # The normal equations hand on the right-hand side beyond the rows in use, so likewise.
    values["gram"], values["normal_rhs"] = values["gram"][:rows, :rows], values["normal_rhs"][:rows]
    values["coefficients"] = helper.coefficients.numpy().copy()
    values["enabled"] = helper.enabled.numpy().copy()
    values["surface_active"] = helper.surface_active.numpy().copy()
    return values


def earlier_coefficients(helper, launches, tracks, support, *, tile=None):
    """The device coefficients as the package issued them before: earlier texts, earlier sizes, host counts.

    ``tile`` is (samples, block threads) of an earlier factorization; without it the bound one is launched.
    """
    solve = helper.device_solve
    capacity, block_dim = (solve.capacity, solve.block_dim) if tile is None else tile
    solve_kernel = solve.solve_kernel if tile is None else woodbury_device._solve_tiles(3 * capacity)
    kernels = {kernel.key: (kernel, dim, inputs) for kernel, dim, inputs in launches}
    for name in ("schur_topology", "schur_inputs"):
        kernel, dim, inputs = kernels[name]
        earlier_launch(name, dim, named(kernel, inputs), tracks, support, capacity)
    if 3 * solve.capacity > _COMPACT_ROWS:
        # The nonzero columns are counted before either system is assembled.
        solve._enqueue_compact_index()
    if "schur_assemble" in kernels:
        kernel, dim, inputs = kernels["schur_assemble"]
        earlier_launch("schur_assemble", dim, named(kernel, inputs), tracks, support, capacity)
        matrix, rhs = solve.matrix, solve.rhs
        if helper.pre.nonsymmetric:
            # A track of a friction tunnel: the normal equations of the padded system. No earlier text.
            live = helper.pre.coupler.interaction._live
            matrix, rhs = solve._enqueue_normal(live, LIVE_TRACK, 3, matrix, rhs, solve.gram, solve.normal_rhs)
        wp.launch_tiled(
            solve_kernel,
            dim=[1],
            inputs=[matrix, rhs, solve.solution, solve.solve_valid],
            block_dim=block_dim,
            device=DEVICE,
        )
    else:
        # The package factorized the nonzero columns alone; that step has no earlier text.
        solve._enqueue_compacted()
    kernel, dim, inputs = kernels["schur_coefficients"]
    earlier_launch("schur_coefficients", dim, named(kernel, inputs), tracks, support, capacity)


def assert_same_bits(test, actual, expected, label):
    test.assertEqual(sorted(actual), sorted(expected))
    for name in actual:
        test.assertEqual(actual[name].tobytes(), expected[name].tobytes(), f"{label}: {name}")


class EarlierTextTests(unittest.TestCase):
    def test_kernels_write_the_bits_of_their_earlier_text(self):
        changed = {}
        packets = [fixture.Packet(100 + tracks, tracks, surface=bool(tracks % 2)) for tracks in (1, 2, 5, 9, 17, 33)]
        packets.append(fixture.Packet(148, _MAX_TRACKS))
        # Full-rank tangents over the whole scratch do not fit the compacted tile: the padded system of 144 rows.
        packets.append(fixture.Packet(149, _MAX_TRACKS, ranks=(3,)))
        # Tracks of a friction tunnel with sliding anchors: the padded and the compacted system.
        packets.append(fixture.Packet(109, 9, surface=True, friction=FRICTION))
        packets.append(fixture.Packet(148, _MAX_TRACKS, friction=FRICTION))
        self.assertTrue(all(packet.shears.any() for packet in packets[-2:]))
        compacted = []
        for packet in packets:
            with self.subTest(tracks=packet.pre.count):
                # The support stays below the capacity: unused response columns are poisoned in the fixture.
                self.assertLess(len(packet.manager.support), _CAPACITY)
                launches = compare_packet(packet, changed)
                compacted.append(fixture._compacted(packet.helper.device_solve))
                # Every changed kernel but ``schur_assemble`` where the nonzero columns were factorized alone.
                self.assertEqual(launches, len(_EARLIER_TEXT) - int(compacted[-1]))
        self.assertEqual(compacted, [False] * 6 + [True, False] + [False, True])
        # No comparison is empty: every kernel changed each of its outputs on these inputs.
        self.assertEqual(
            {name: sorted(written) for name, written in changed.items()},
            {name: sorted(outputs) for name, outputs in _OUTPUTS.items()},
        )

    def test_recorded_packet_writes_the_bits_of_the_earlier_text(self):
        recorded = fixture._recorded()
        packet = SimpleNamespace(pre=recorded.pre, helper=recorded.helper, manager=recorded.manager)

        def host():
            recorded.helper.prepare(recorded.manager)

        def device():
            helper = recorded.helper
            self.assertIsNotNone(helper.prepare_device(recorded.manager))
            helper.device_solve.enqueue_topology()
            helper.device_solve.enqueue()
            return helper.coefficients.numpy().copy(), int(helper.enabled.numpy()[0])

        packet.host, packet.device = host, device
        self.assertEqual(compare_packet(packet), len(_EARLIER_TEXT))

    def test_fixed_launches_stay_inside_their_arrays_under_debug_bounds_checks(self):
        # An out-of-bounds read stops a debug kernel with an assertion, so the kernels run in a child process.
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NEWTON_TEST_DEVICE="cpu")
        command = (
            "import warp as wp; wp.config.mode = 'debug'; "
            "import test_woodbury_live_counts as probe; probe.bounds_probe()"
        )
        result = subprocess.run([sys.executable, "-c", command], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-2000:])
        # 1 to 48 samples, and a second packet above 38 samples; each without and with a friction tunnel.
        # Compacted: the packets of mixed ranks above 38 samples, each without ``schur_assemble``.
        compacted = 2 * (_MAX_TRACKS - _COMPACT_ROWS // 3)
        packets = 2 * _MAX_TRACKS + compacted
        launches = len(_EARLIER_TEXT) * packets - compacted
        self.assertIn(
            f"bounds probe ok: {launches} launches of {packets} packets, {compacted} compacted", result.stdout
        )


class DeviceCoefficientBitsTests(unittest.TestCase):
    def test_coefficients_are_the_bits_of_the_earlier_kernels_for_every_track_count(self):
        usable = padded = 0
        earlier_kernels()
        # Above 38 samples a second packet of full-rank tangents, whose columns do not fit the compacted tile.
        cases = [(tracks, ranks) for tracks in range(1, _MAX_TRACKS + 1) for ranks in ((2, 3, 3, 0, 2), (3,))]
        cases = [(tracks, ranks) for tracks, ranks in cases if 3 * tracks > _COMPACT_ROWS or len(ranks) > 1]
        # Each without and with a friction tunnel: the second with sheared anchors and normal equations.
        cases = [(tracks, ranks, friction) for tracks, ranks in cases for friction in (None, FRICTION)]
        for tracks, ranks, friction in cases:
            with self.subTest(tracks=tracks, ranks=ranks, friction=friction):
                packet = fixture.Packet(200 + tracks, tracks, surface=bool(tracks % 2), ranks=ranks, friction=friction)
                helper = packet.helper
                self.assertIsNotNone(helper.prepare_device(packet.manager))
                solve = helper.device_solve

                def coefficients(solve=solve):
                    # The branch that a captured graph takes; outside a capture ``enqueue`` pads.
                    solve.enqueue_topology()
                    fixture._correction(solve)

                reset_scratch(helper)
                launches = record(coefficients)
                actual = results(helper)
                if fixture._compacted(solve):
                    # The nonzero columns were factorized alone: the padded system is not assembled.
                    self.assertNotIn("schur_assemble", [kernel.key for kernel, _, _ in launches])
                    self.assertGreater(3 * tracks, _COMPACT_ROWS)
                else:
                    self.assertTrue(padded_identity(solve.matrix.numpy(), solve.rhs.numpy(), 3 * solve.capacity))
                    padded += 1
                reset_scratch(helper)
                earlier_coefficients(helper, launches, tracks, len(helper.support))
                assert_same_bits(self, actual, results(helper), f"{tracks} samples")
                # Columns beyond the support are never written.
                used = np.r_[np.arange(len(helper.support)), np.arange(_CAPACITY, _CAPACITY + 6)]
                usable += int(actual["enabled"][0] == 1 and np.isfinite(actual["coefficients"][used]).all())
        # The comparison is one of usable proposals, not of two rejected ones.
        self.assertEqual(usable, len(cases))
        # The padded system: every count up to 38 samples, and the full-rank packets above.
        self.assertEqual(padded, 2 * (_COMPACT_ROWS // 3 + (_MAX_TRACKS - _COMPACT_ROWS // 3)))

    def test_tile_of_17_to_38_samples_solves_as_the_earlier_tiles_did(self):
        earlier_kernels()
        for tracks in (17, 24, 32, 33, 36, _COMPACT_ROWS // 3):
            with self.subTest(tracks=tracks):
                packet = fixture.Packet(250 + tracks, tracks, surface=bool(tracks % 2))
                helper = packet.helper
                self.assertIsNotNone(helper.prepare_device(packet.manager))
                solve = helper.device_solve
                self.assertIs(solve.solve_kernel, solve.compact_kernel)
                # The tile before: 32 samples (96 rows) up to 32, 40 samples (120 rows) above; 128 block threads.
                tile = (32 if tracks <= 32 else 40, 128)
                self.assertNotEqual(tile, (solve.capacity, solve.block_dim))

                def coefficients(solve=solve):
                    solve.enqueue_topology()
                    solve.enqueue()

                reset_scratch(helper)
                launches = record(coefficients)
                actual = results(helper)
                self.assertEqual(actual["enabled"].tolist(), [1])
                self.assertEqual(actual["solve_valid"].tolist(), [1])
                reset_scratch(helper)
                earlier_coefficients(helper, launches, tracks, len(helper.support), tile=tile)
                expected = results(helper)
                n = 3 * tracks
                # Both tiles are given the same system: its engaged block holds the same bits.
                system = actual["matrix"][:n, :n]
                self.assertEqual(system.tobytes(), expected["matrix"][:n, :n].tobytes())
                self.assertEqual(actual["rhs"][:n].tobytes(), expected["rhs"][:n].tobytes())
                # Everything in front of the factorization, and both flags, bit for bit.
                different = {"matrix", "rhs", "solution", "coefficients"}
                assert_same_bits(
                    self,
                    {name: value for name, value in actual.items() if name not in different},
                    {name: value for name, value in expected.items() if name not in different},
                    f"{tracks} samples",
                )
                # Each FP32 Cholesky solve is within n * roundoff * condition of the solution of the system,
                # so the two tiles differ by at most twice that.
                bound = 2 * fixture._bound(system, n)
                reference = expected["solution"][:n]
                np.testing.assert_allclose(
                    actual["solution"][:n], reference, rtol=0, atol=bound * np.abs(reference).max()
                )
                used = np.r_[np.arange(len(helper.support)), np.arange(_CAPACITY, _CAPACITY + 6)]
                reference = expected["coefficients"][used]
                self.assertTrue(np.isfinite(reference).all())
                np.testing.assert_allclose(
                    actual["coefficients"][used], reference, rtol=0, atol=bound * np.abs(reference).max()
                )

    @unittest.skipUnless(DEVICE.startswith("cuda"), "CUDA graph replay contract")
    def test_captured_coefficients_follow_the_device_counts(self):
        # Five to eight samples share one factorization, so one graph serves them.
        packet = fixture.Packet(61, 8)
        helper = packet.helper
        key = helper.prepare_device(packet.manager)
        self.assertIsNotNone(key)
        solve, live = helper.device_solve, packet.pre.coupler.interaction._live
        self.assertEqual(solve.capacity, 8)

        def coefficients():
            solve.enqueue_topology()
            solve.enqueue()

        def set_tracks(tracks):
            counts = live.numpy()
            counts[LIVE_TRACK] = tracks
            live.assign(counts)

        # Captured with the fewest samples: a launch sized by that count would miss the later ones.
        set_tracks(5)
        with ScopedCapture(device=packet.pre.device) as capture:
            launches = record(coefficients)
        seen = []
        for tracks in (5, 8, 6, 7, 8):
            with self.subTest(tracks=tracks):
                set_tracks(tracks)
                reset_scratch(helper)
                wp.capture_launch(capture.graph)
                replayed = results(helper)
                self.assertEqual(replayed["enabled"].tolist(), [1])
                reset_scratch(helper)
                coefficients()
                assert_same_bits(self, replayed, results(helper), f"replay at {tracks} samples")
                reset_scratch(helper)
                earlier_coefficients(helper, launches, tracks, len(helper.support))
                assert_same_bits(self, replayed, results(helper), f"earlier text at {tracks} samples")
                seen.append(replayed["coefficients"].tobytes())
        # The replays differ with the count and repeat with it.
        self.assertEqual(len(set(seen)), 4)
        self.assertEqual(seen[1], seen[4])
        self.assertEqual(helper.prepare_device(packet.manager), key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
