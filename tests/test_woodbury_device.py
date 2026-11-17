"""Device FP32 Schur coefficients against the host FP64 dense Woodbury solve."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_woodbury_guess as recorded
import warp as wp
from _woodbury_reference import (
    block_diagonal,
    compacted_solution,
    interpolation,
    nonzero_columns,
    rigid_interpolation,
    sheared,
    sliding_shears,
)

from newton_solver_suturebot._src.solvers.coupled.acceleration import woodbury_device
from newton_solver_suturebot._src.solvers.coupled.acceleration.woodbury_device import (
    _COMPACT_BLOCK_DIM,
    _COMPACT_ROWS,
    _LAUNCH_BLOCK_DIM,
)
from newton_solver_suturebot._src.solvers.coupled.acceleration.woodbury_guess import (
    _CAPACITY,
    _MAX_TRACKS,
    HostSchurPlan,
    WoodburyInitialGuess,
    host_coefficients,
)
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import LIVE_TRACK
from newton_solver_suturebot._src.utils.capture import ScopedCapture

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
H = 1 / 240


@wp.kernel(enable_backward=False)
def countdown(remaining: wp.array[int]):
    remaining[0] -= 1


def _spd(rng, size, low, high):
    q, _ = np.linalg.qr(rng.normal(size=(size, size)))
    return q @ np.diag(rng.uniform(low, high, size)) @ q.T


def _tangent(rng, rank):
    """One FP32 anchor tangent: zero, one direction, a transverse projection, or with a stick term."""
    if rank == 0:
        return np.zeros((3, 3), np.float32)
    t = rng.normal(size=3)
    t /= np.linalg.norm(t)
    d = rng.normal(size=3)
    d -= d.dot(t) * t
    d /= np.linalg.norm(d)
    if rank == 1:
        return (rng.uniform(0.05, 0.6) * np.outer(d, d)).astype(np.float32)
    fraction = rng.uniform(0.1, 0.9)
    matrix = rng.uniform(0.05, 0.6) * ((1 - fraction) * (np.eye(3) - np.outer(t, t)) + fraction * np.outer(d, d))
    if rank == 3:
        matrix += rng.uniform(0.01, 0.3) * np.outer(t, t)
    return matrix.astype(np.float32)


class Packet:
    """A synthetic frozen shaft system with the layout of the recorded fixture."""

    def __init__(
        self,
        seed,
        tracks,
        *,
        nodes=40,
        pool=30,
        surface=False,
        tissue=(0.003, 0.03),
        rigid=(0.002, 0.02),
        ranks=(2, 3, 3, 0, 2),
        friction=None,
    ):
        rng = np.random.default_rng(seed)
        self.fixed = np.zeros(nodes, np.int32)
        self.fixed[rng.choice(nodes, nodes // 5, replace=False)] = 1
        # Track support stays within the fixed response capacity of 32 free nodes.
        candidates = np.r_[np.flatnonzero(self.fixed == 0)[:pool], np.flatnonzero(self.fixed)]
        self.nodes = np.asarray([rng.choice(candidates, 4, replace=False) for _ in range(tracks)], np.int32)
        self.bary = rng.dirichlet(np.ones(4), tracks).astype(np.float32)
        self.bary[::3, 0] = 0
        # ``ranks`` are the tangent ranks along the track, repeated.
        self.matrices = np.asarray([_tangent(rng, ranks[i % len(ranks)]) for i in range(tracks)])
        # ``friction`` is the coefficient of a friction tunnel, whose sliding anchors (rank two) carry a
        # shear and whose system is not symmetric; None is every other anchor law: no shear.
        self.friction = friction
        self.shears = sliding_shears(self.matrices, 0.0 if friction is None else friction)
        self.levers = rng.normal(0, 0.5, (tracks, 3)).astype(np.float32)
        free_dofs = np.flatnonzero(np.repeat(self.fixed == 0, 3))
        # Magnitudes of the recorded packet: h D C reaches tens, far from identity.
        self.mass = np.eye(3 * nodes + 6)
        self.mass[np.ix_(free_dofs, free_dofs)] = _spd(rng, len(free_dofs), *tissue)
        self.mass[-6:, -6:] = _spd(rng, 6, *rigid).astype(np.float32)
        self.rhs = rng.normal(0, 1e-3, (nodes + 2, 3)).astype(np.float32)
        self.surface = surface
        self.build()

    def operator(self):
        """Dense frozen shaft system of the current tangents and levers."""
        full, _, _ = interpolation(self.fixed, self.nodes, self.bary)
        full[:, np.flatnonzero(np.repeat(self.fixed, 3))] = 0
        j = np.column_stack((-full, rigid_interpolation(self.levers)))
        return self.mass + float(np.float32(H)) * j.T @ block_diagonal(sheared(self.matrices, self.shears)) @ j

    def build(self):
        device = wp.get_device(DEVICE)
        _, b, free = interpolation(self.fixed, self.nodes, self.bary)
        self.matrix = self.operator()
        self.matrix_device = wp.array(self.matrix, dtype=float, device=device)
        p = self.pre = SimpleNamespace(device=device, n=len(self.fixed) + 2, count=len(self.nodes), _last_dt=H)
        p.time_step = wp.array([H], dtype=float, device=device)
        p.rhs = wp.array(self.rhs, dtype=wp.vec3, device=device)
        p.z = wp.zeros_like(p.rhs)
        rigid = self.mass[-6:, -6:]
        p.rigid_blocks = wp.array(
            [rigid[:3, :3], rigid[3:, 3:], rigid[:3, 3:], rigid[3:, :3]], dtype=wp.mat33, device=device
        )
        p.matrices = wp.array(self.matrices, dtype=wp.mat33, device=device)
        p.levers = wp.array(self.levers, dtype=wp.vec3, device=device)
        p.shears = wp.array(self.shears, dtype=wp.mat33, device=device)
        p.nonsymmetric = self.friction is not None
        # The kernels read the number of track samples from the live counts of the interaction.
        live = np.zeros(3, dtype=np.int32)
        live[LIVE_TRACK] = len(self.nodes)
        p.coupler = SimpleNamespace(
            interaction=SimpleNamespace(
                _track_nodes=wp.array(self.nodes, dtype=wp.vec4i, device=device),
                _track_bary=wp.array(self.bary, dtype=wp.vec4, device=device),
                _live=wp.array(live, dtype=int, device=device),
            )
        )
        p.operator = SimpleNamespace(matvec=self.matvec)
        if self.surface:
            p.surface = SimpleNamespace(
                anchors=SimpleNamespace(count=wp.array([3], dtype=int, device=device)),
                valid=wp.array([1], dtype=int, device=device),
            )
        support = tuple(int(i) for i in np.flatnonzero(np.any(b != 0, axis=0)))
        a = self.mass[np.ix_(free, free)]
        self.response = np.linalg.solve(a, np.eye(len(a))[:, support]).astype(np.float32)
        # Unused response capacity is poisoned: reading it invalidates a guess.
        flat = np.full(len(a) * _CAPACITY, np.nan, np.float32)
        flat[: self.response.size] = self.response.ravel(order="F")
        core = SimpleNamespace(
            basis=wp.array(flat, dtype=float, device=device),
            free=wp.array(np.flatnonzero(self.fixed == 0), dtype=int, device=device),
            support=wp.array(support, dtype=int, device=device),
            # The number of support columns as device data, as ``load_basis`` writes it, and the free nodes.
            count=wp.array([len(support)], dtype=int, device=device),
            n=len(a) // 3,
        )
        self.manager = SimpleNamespace(
            active=True, core=core, support=support, basis_builds=1, structural_key=("synthetic",)
        )
        self.manager._structure = lambda: self.manager.structural_key
        self.helper = WoodburyInitialGuess(p)

    def matvec(self, x, y, z, alpha, beta):
        wp.launch(recorded.dense_matvec, self.pre.n, [self.matrix_device, x, y, z, alpha, beta], device=DEVICE)

    def host(self):
        """Coefficients of the unchanged host packet and dense solve."""
        self.helper.prepare(self.manager)
        if self.helper.enabled.numpy()[0] != 1:
            raise AssertionError(self.helper.last_reason)
        return self.helper.coefficients.numpy().copy()

    def device(self):
        """Coefficients and validity flag of the device kernels alone, with the branch of a captured graph."""
        helper = self.helper
        if helper.prepare_device(self.manager) is None:
            raise AssertionError("No response basis is bound")
        helper.coefficients.fill_(np.nan)
        helper.enabled.fill_(-1)
        helper.device_solve.enqueue_topology()
        _correction(helper.device_solve)
        return helper.coefficients.numpy().copy(), int(helper.enabled.numpy()[0])


def _recorded():
    fixture = recorded.Tests()
    fixture.setUp()
    fixture.pre.time_step = wp.array([float(fixture.f["h"])], dtype=float, device=DEVICE)
    return fixture


class SymmetricPlan(HostSchurPlan):
    """Host plan with the symmetrized track compliance the device solve factorizes."""

    def interpolation(self, records):
        p, pc, pcp = super().interpolation(records)
        return p, pc, 0.5 * (pcp + pcp.T)


def _correction(solve, *, compact=None):
    """One correction, launch by launch, with the branch that a captured graph takes.

    ``enqueue`` makes the choice between the compacted and the padded system a conditional node of
    the graph, and factorizes the padded system outside a capture. Here the flag is read on the host
    instead; ``compact`` overrides it. The captured tests compare a replayed graph with these launches.
    """
    solve._enqueue_inputs()
    if 3 * solve.capacity > _COMPACT_ROWS:
        solve._enqueue_compact_index()
        if compact is None:
            compact = int(solve.compact_fits.numpy()[0]) == 1
    if compact:
        solve._enqueue_compacted()
    else:
        solve._enqueue_padded()
    solve._enqueue_coefficients()


def _compacted(solve):
    """Whether the last correction factorized the nonzero tangent-factor columns alone."""
    return 3 * solve.capacity > _COMPACT_ROWS and int(solve.compact_fits.numpy()[0]) == 1


def _bound(system, n):
    """n * roundoff * condition of a factorized FP32 system; an empty system has nothing to solve."""
    system = np.asarray(system, dtype=float)
    return n * np.finfo(np.float32).eps * (np.linalg.cond(system) if system.size else 1.0)


def _factorized(helper):
    """The system that the last correction factorized, on its rows in use.

    The padded system or the compacted one; for a track that follows a friction tunnel the
    normal equations of that system.
    """
    n = 3 * helper.pre.count
    solve = helper.device_solve
    normal = helper.pre.nonsymmetric
    if _compacted(solve):
        rows = int(solve.compact_count.numpy()[0])
        return (solve.compact_gram if normal else solve.compact_matrix).numpy()[:rows, :rows]
    return (solve.gram if normal else solve.matrix).numpy()[:n, :n]


def _tolerance(helper):
    """Forward error bound of the device FP32 assembly and Cholesky solve.

    A backward stable solve of n equations in FP32 is accurate to the order of
    n * roundoff * condition; the host solves the same FP32 data in FP64. The
    compacted system has the condition of the padded one, whose other rows are
    rows of the identity while no eigenvalue of I + h F.T C F is below one. The
    normal equations of a nonsymmetric system have the square of its condition.
    """
    return _bound(_factorized(helper), 3 * helper.pre.count)


def _close(actual, expected, helper):
    # Columns beyond the support are never read by the guess.
    used = np.r_[np.arange(len(helper.support)), np.arange(_CAPACITY, _CAPACITY + 6)]
    bound = _tolerance(helper) * np.abs(expected[used]).max()
    np.testing.assert_allclose(actual[used], expected[used], rtol=0, atol=bound)


class DeviceCoefficientTests(unittest.TestCase):
    def test_recorded_packet_matches_host_solve_and_solves_the_frozen_system(self):
        fixture = _recorded()
        helper = fixture.helper
        helper.prepare(fixture.manager)
        expected = helper.coefficients.numpy().copy()
        self.assertIsNotNone(helper.prepare_device(fixture.manager))
        helper.coefficients.zero_()
        helper.enabled.zero_()
        helper.device_solve.enqueue_topology()
        helper.device_solve.enqueue()
        self.assertEqual(helper.enabled.numpy()[0], 1)
        _close(helper.coefficients.numpy(), expected, helper)
        # The recorded track ends in a disengaged anchor with a zero tangent.
        np.testing.assert_array_equal(helper.device_solve.factors.numpy()[14], 0)
        helper.initialize()
        self.assertEqual(helper.selected.numpy()[0], 1)
        answer = fixture.pre.z.numpy().reshape(-1).astype(float)
        rhs = fixture.pre.rhs.numpy().reshape(-1).astype(float)
        self.assertLess(np.linalg.norm(fixture.matrix @ answer - rhs) / np.linalg.norm(rhs), 2e-6)

    def test_track_counts_tangent_ranks_and_surface_flag_match_host(self):
        # Counts on both sides of every padded factorization size, up to the tile that holds 38 samples.
        for tracks in (1, 3, 4, 5, 8, 9, 16, 17, 32, 33, _COMPACT_ROWS // 3):
            for surface in (False, True):
                with self.subTest(tracks=tracks, surface=surface):
                    packet = Packet(100 + tracks, tracks, surface=surface)
                    expected = packet.host()
                    actual, enabled = packet.device()
                    self.assertEqual(enabled, 1)
                    _close(actual, expected, packet.helper)
                    self.assertEqual(packet.helper.surface_active.numpy()[0], int(surface))
                    solve = packet.helper.device_solve
                    capacity = solve.capacity
                    self.assertLessEqual(tracks, capacity)
                    # Powers of two up to 16 samples, then the tile of the compacted system.
                    powers = max(4, 1 << (tracks - 1).bit_length())
                    self.assertEqual(capacity, powers if tracks <= 16 else _COMPACT_ROWS // 3)
                    # The cooperative factorization has a thread for every row.
                    self.assertGreaterEqual(solve.block_dim, 3 * capacity)
                    # Padding beyond the engaged tracks is an exact identity over the whole scratch,
                    # whatever leading block the factorization of this track count reads.
                    matrix = solve.matrix.numpy()
                    self.assertEqual(matrix.shape, (3 * _MAX_TRACKS, 3 * _MAX_TRACKS))
                    padding = np.eye(3 * (_MAX_TRACKS - tracks))
                    np.testing.assert_array_equal(matrix[3 * tracks :, 3 * tracks :], padding)
                    np.testing.assert_array_equal(matrix[3 * tracks :, : 3 * tracks], 0)
                    np.testing.assert_array_equal(matrix[: 3 * tracks, 3 * tracks :], 0)
                    np.testing.assert_array_equal(solve.rhs.numpy()[3 * tracks :], 0)
                    np.testing.assert_array_equal(solve.solution.numpy()[3 * tracks : 3 * capacity], 0)
                    packet.helper.initialize()
                    self.assertEqual(packet.helper.selected.numpy()[0], 1)

    def test_tangent_factors_reproduce_semidefinite_blocks(self):
        packet = Packet(7, 10)
        packet.device()
        factors = packet.helper.device_solve.factors.numpy()[:10]
        np.testing.assert_array_equal(packet.helper.device_solve.factor_valid.numpy()[:10], 1)
        for factor, matrix in zip(factors, packet.matrices.astype(float), strict=True):
            # Dropped pivots are FP32 rounding of the assembled block.
            bound = 8 * np.finfo(np.float32).eps * np.abs(matrix).max()
            np.testing.assert_allclose(factor.astype(float) @ factor.T.astype(float), matrix, rtol=0, atol=bound)
            if not matrix.any():
                np.testing.assert_array_equal(factor, 0)

    def test_symmetrized_track_compliance_is_the_only_difference_from_the_host_solve(self):
        # A light tissue under a heavy needle makes the track compliance dominate.
        packet = Packet(11, 12, tissue=(1e-4, 1e-3), rigid=(0.05, 0.5))
        rng = np.random.default_rng(5)
        # Stored response columns are symmetric only to their solve tolerance.
        noisy = packet.response * (1 + 1e-2 * rng.normal(size=packet.response.shape)).astype(np.float32)
        flat = packet.manager.core.basis.numpy()
        flat[: noisy.size] = noisy.ravel(order="F")
        packet.manager.core.basis.assign(flat)
        packet.manager.basis_builds += 1
        unsymmetric = packet.host()
        helper = packet.helper
        symmetric = host_coefficients(
            helper.packet.numpy(),
            packet.pre.count,
            helper.support,
            helper.free_host,
            helper.compliance,
            float(np.float32(H)),
            plan=SymmetricPlan(helper.support, helper.free_host, helper.compliance),
        )
        actual, enabled = packet.device()
        self.assertEqual(enabled, 1)
        _close(actual, symmetric, helper)
        # The two host solves differ by the injected asymmetry, far above the
        # comparison tolerance, so the previous line is a discriminating check.
        scale = np.abs(symmetric).max()
        difference = np.abs(unsymmetric - symmetric).max()
        self.assertGreater(difference, 10 * _tolerance(helper) * scale)
        self.assertLess(difference, 1e-1 * scale)

    def test_untrusted_inputs_clear_the_device_flag_and_keep_the_zero_start(self):
        def zero_start(packet):
            _, enabled = packet.device()
            self.assertEqual(enabled, 0)
            packet.pre.z.fill_(wp.vec3(1.0))
            packet.helper.initialize()
            self.assertEqual(packet.helper.selected.numpy()[0], 0)
            np.testing.assert_array_equal(packet.pre.z.numpy(), 0)

        with self.subTest("singular rigid metric"):
            packet = Packet(21, 6)
            blocks = packet.pre.rigid_blocks.numpy()
            blocks[1] = 0
            packet.pre.rigid_blocks.assign(blocks)
            zero_start(packet)
            self.assertEqual(packet.helper.device_solve.rigid_valid.numpy()[0], 0)
        with self.subTest("indefinite rigid metric"):
            packet = Packet(22, 6)
            blocks = packet.pre.rigid_blocks.numpy()
            blocks[0] = -blocks[0]
            packet.pre.rigid_blocks.assign(blocks)
            zero_start(packet)
        with self.subTest("indefinite anchor tangent"):
            packet = Packet(23, 6)
            matrices = packet.matrices.copy()
            matrices[2] = np.diag([0.2, -0.01, 0.1])
            packet.pre.matrices.assign(matrices)
            zero_start(packet)
            np.testing.assert_array_equal(packet.helper.device_solve.factor_valid.numpy()[:6], [1, 1, 0, 1, 1, 1])
            # Restored inputs are accepted again by the same persistent scratch.
            packet.pre.matrices.assign(packet.matrices)
            self.assertEqual(packet.device()[1], 1)
        with self.subTest("nonfinite residual on track support"):
            packet = Packet(24, 6)
            free = np.flatnonzero(packet.fixed == 0)
            rhs = packet.rhs.copy()
            rhs[free[packet.manager.support[4] // 3], 1] = np.nan
            packet.pre.rhs.assign(rhs)
            zero_start(packet)
        with self.subTest("nonfinite rigid residual"):
            packet = Packet(24, 6)
            rhs = packet.rhs.copy()
            rhs[-1, 2] = np.inf
            packet.pre.rhs.assign(rhs)
            zero_start(packet)
            self.assertEqual(packet.helper.device_solve.rigid_valid.numpy()[0], 0)
        with self.subTest("nonfinite residual outside the packet"):
            # The coefficients never read it; the current-operator certificate does.
            packet = Packet(24, 6)
            support = {dof // 3 for dof in packet.manager.support}
            free = np.flatnonzero(packet.fixed == 0)
            outside = next(int(node) for index, node in enumerate(free) if index not in support)
            rhs = packet.rhs.copy()
            rhs[outside, 0] = np.nan
            packet.pre.rhs.assign(rhs)
            self.assertEqual(packet.device()[1], 1)
            packet.helper.initialize()
            self.assertEqual(packet.helper.selected.numpy()[0], 0)
            np.testing.assert_array_equal(packet.pre.z.numpy(), 0)
        with self.subTest("nonfinite anchor tangent"):
            packet = Packet(25, 6)
            matrices = packet.matrices.copy()
            matrices[0, 1, 1] = np.inf
            packet.pre.matrices.assign(matrices)
            zero_start(packet)
        with self.subTest("track outside the response support"):
            packet = Packet(26, 6)
            packet.host()
            packet.manager.support = packet.manager.support[:-3]
            packet.manager.core.count.fill_(len(packet.manager.support))
            packet.manager.basis_builds += 1
            packet.helper.prepare(packet.manager)
            self.assertIn("exceeds basis support", packet.helper.last_reason)
            zero_start(packet)
            self.assertIn(0, packet.helper.device_solve.supported.numpy()[:6])
        with self.subTest("indefinite Schur system"):
            # A sign-flipped response makes I + h F.T C F indefinite while every
            # input stays finite: only the factorization can clear the flag.
            packet = Packet(28, 6)
            flat = packet.manager.core.basis.numpy()
            flat[: packet.response.size] *= -1e3
            packet.manager.core.basis.assign(flat)
            packet.manager.basis_builds += 1
            zero_start(packet)
            solve = packet.helper.device_solve
            np.testing.assert_array_equal(solve.solve_valid.numpy(), [0])
            np.testing.assert_array_equal(solve.rigid_valid.numpy(), [1])
            np.testing.assert_array_equal(solve.factor_valid.numpy()[:6], 1)
            np.testing.assert_array_equal(solve.supported.numpy()[:6], 1)
            self.assertLess(np.linalg.eigvalsh(solve.matrix.numpy()[:18, :18]).min(), -1.0)

    def test_host_rebinding_between_launches_rebinds_the_device_columns(self):
        packet = Packet(51, 7)
        self.assertEqual(packet.device()[1], 1)
        # A rebuilt response basis may order its columns differently.
        support = packet.manager.support[::-1]
        flat = packet.manager.core.basis.numpy()
        flat[: packet.response.size] = packet.response[:, ::-1].ravel(order="F")
        packet.manager.core.basis.assign(flat)
        packet.manager.core.support.assign(np.asarray(support, dtype=np.int32))
        packet.manager.support = support
        packet.manager.basis_builds += 1
        # A host-controlled application binds the new basis first; the next
        # captured launch must not keep the previous column map.
        expected = packet.host()
        self.assertEqual(packet.helper.support, support)
        actual, enabled = packet.device()
        self.assertEqual(enabled, 1)
        _close(actual, expected, packet.helper)

    def test_key_is_constant_from_the_first_binding_and_the_device_flag_says_whether_the_basis_is_usable(self):
        packet = Packet(31, 4)
        helper, manager = packet.helper, packet.manager

        def available():
            return helper.available.numpy().tolist()

        # Nothing was ever bound: no guess can be captured. A manager without a core, without a
        # recorded structure or with another structure than the solver's binds nothing.
        self.assertIsNone(helper.prepare_device(None))
        core, structure, current = manager.core, manager.structural_key, manager._structure
        manager.active = False
        for attribute, value in (("core", None), ("structural_key", None), ("_structure", lambda: ("another",))):
            setattr(manager, attribute, value)
            self.assertIsNone(helper.prepare_device(manager), attribute)
            manager.core, manager.structural_key, manager._structure = core, structure, current
        self.assertIsNone(helper.device_solve)
        self.assertEqual((available(), helper.usable), ([0], False))
        # A core of the current structure is bound as soon as it exists, before any basis is
        # applied (the storage that start-up makes ahead of the first track support): the key
        # of every later launch, with the device flag saying that the basis may not be used.
        key = helper.prepare_device(manager)
        self.assertIsNotNone(key)
        self.assertIsNotNone(helper.device_solve)
        self.assertEqual((available(), helper.usable, helper.last_reason), ([0], False, "inactive FEM basis manager"))
        manager.active = True
        self.assertEqual(helper.prepare_device(manager), key)
        self.assertEqual((available(), helper.usable), ([1], True))
        self.assertEqual(helper.prepare_device(manager), key)
        solve = helper.device_solve
        pointers = [array.ptr for array in (helper.inverse_free, solve.support_column, solve.size_flags)]

        def unusable(reason):
            # A bound basis that one launch may not use: the same key, and the flag on the device.
            self.assertEqual(helper.prepare_device(manager), key, reason)
            self.assertEqual((available(), helper.usable, helper.last_reason), ([0], False, reason))

        def usable():
            self.assertEqual(helper.prepare_device(manager), key)
            self.assertEqual((available(), helper.usable), ([1], True))
            self.assertEqual(
                [array.ptr for array in (helper.inverse_free, solve.support_column, solve.size_flags)], pointers
            )

        manager.active = False
        unusable("inactive FEM basis manager")
        manager.active = True
        usable()
        self.assertEqual(packet.device()[1], 1)
        # A rebuilt basis of the same structure is the same binding of a captured start.
        manager.basis_builds += 1
        usable()
        # More support columns than the scratch holds.
        support = manager.support
        manager.support = support + tuple(range(900, 900 + _CAPACITY))
        unusable("capacity fallback")
        manager.support = support
        usable()
        # One sample more than the scratch of the start guess holds, and no sample.
        for count in (_MAX_TRACKS + 1, 0):
            packet.pre.count = count
            unusable("capacity fallback")
        packet.pre.count = 4
        usable()
        # A structure that the manager has not followed yet.
        manager._structure = lambda: ("changed",)
        unusable("changed compact/free binding")
        manager._structure = lambda: manager.structural_key
        usable()
        # No manager at all, as after a scene has closed its response basis.
        self.assertEqual(helper.prepare_device(None), key)
        self.assertEqual(available(), [0])
        usable()
        # Every track count that the scratch holds, and its padded tile on the device.
        tiles = woodbury_device._TILES
        self.assertEqual(tiles, (4, 8, 16, _COMPACT_ROWS // 3, 40, 48))
        for count in range(1, _MAX_TRACKS + 1):
            packet.pre.count = count
            usable()
            capacity = solve.capacity
            self.assertEqual(capacity, woodbury_device._padded_samples(count))
            self.assertGreaterEqual(capacity, count)
            flags = solve.size_flags.numpy().tolist()
            expected = [int(tile == capacity) for tile in tiles] + [int(3 * capacity > _COMPACT_ROWS)]
            self.assertEqual(flags, expected, count)
            self.assertEqual((solve.solve_kernel, solve.block_dim), solve.tiles[tiles.index(capacity)])
        packet.pre.count = 4
        # A new support of the basis rewrites the column map in place.
        manager.support = support[::-1]
        usable()
        columns = solve.support_column.numpy()
        np.testing.assert_array_equal(columns[list(support[::-1])], np.arange(len(support)))
        self.assertEqual(int((columns >= 0).sum()), len(support))
        manager.support = support
        usable()
        # New storage that a captured start addresses is another key.
        kept = manager.core.basis
        manager.core.basis = wp.clone(kept)
        other = helper.prepare_device(manager)
        self.assertNotEqual(other, key)
        self.assertEqual(
            [array.ptr for array in (helper.inverse_free, solve.support_column, solve.size_flags)], pointers
        )
        manager.core.basis = kept
        self.assertEqual(helper.prepare_device(manager), key)

    def test_host_and_device_bindings_follow_the_basis_independently(self):
        packet = Packet(33, 6)
        helper, manager = packet.helper, packet.manager
        expected = packet.host()
        compliance, plan = helper.compliance, helper.host_plan
        self.assertIsNone(helper.device_solve)
        # The device binding of a bound structure reads nothing from the device.
        with patch.object(wp.array, "numpy", side_effect=AssertionError("a read on the host")):
            key = helper.prepare_device(manager)
        solve = helper.device_solve
        columns = solve.support_column
        self.assertIs(helper.compliance, compliance)
        # A rebuilt basis: the device binding has nothing to do, the host binding gathers again.
        manager.basis_builds += 1
        with (
            patch.object(wp.array, "numpy", side_effect=AssertionError("a read on the host")),
            patch.object(wp.array, "assign", side_effect=AssertionError("a write of the column map")),
        ):
            self.assertEqual(helper.prepare_device(manager), key)
        self.assertIs(helper.compliance, compliance)
        self.assertIs(helper.host_plan, plan)
        actual, enabled = packet.device()
        self.assertEqual(enabled, 1)
        _close(actual, expected, helper)
        np.testing.assert_array_equal(packet.host(), expected)
        self.assertIsNot(helper.compliance, compliance)
        self.assertIsNot(helper.host_plan, plan)
        compliance, plan = helper.compliance, helper.host_plan
        # Another order of the support: the device binding rewrites its map, the host binding
        # stays as it is until a host-controlled application.
        support = manager.support[::-1]
        flat = manager.core.basis.numpy()
        flat[: packet.response.size] = packet.response[:, ::-1].ravel(order="F")
        manager.core.basis.assign(flat)
        manager.core.support.assign(np.asarray(support, dtype=np.int32))
        manager.support = support
        manager.basis_builds += 1
        with patch.object(wp.array, "numpy", side_effect=AssertionError("a read on the host")):
            self.assertEqual(helper.prepare_device(manager), key)
        self.assertIs(solve.support_column, columns)
        self.assertIs(helper.compliance, compliance)
        self.assertIs(helper.host_plan, plan)
        self.assertEqual(plan.support, support[::-1])
        actual, enabled = packet.device()
        self.assertEqual(enabled, 1)
        expected = packet.host()
        self.assertEqual(helper.host_plan.support, support)
        _close(actual, expected, helper)
        self.assertEqual(helper.prepare_device(manager), key)

    def test_core_made_ahead_of_the_first_track_sample_gives_the_key_that_the_first_sample_keeps(self):
        # The state in which a start-up leaves a response basis whose storage it makes ahead of the
        # first track support: a core of the current FEM structure without a column, an inactive
        # manager, no track sample. The first trial preparation binds that core: it reads the free
        # nodes, once, makes the device solve and returns the key of every later launch. The first
        # track sample, the support of its basis and the samples after it then read nothing and
        # replace no storage; they write the device flag, the column map and the size flags.
        packet = Packet(37, 5)
        helper, pre, manager = packet.helper, packet.pre, packet.manager
        support, columns = manager.support, manager.core.count
        live = pre.coupler.interaction._live

        def tracks(count):
            counts = np.zeros(3, dtype=np.int32)
            counts[LIVE_TRACK] = count
            live.assign(counts)
            pre.count = count

        def basis(active, held):
            manager.active, manager.support = active, held
            columns.fill_(len(held))

        def flag():
            return helper.available.numpy().tolist(), helper.usable

        read, reads = wp.array.numpy, []

        def counted(array, *args, **kwargs):
            reads.append(array.ptr)
            return read(array, *args, **kwargs)

        basis(False, ())
        tracks(0)
        with patch.object(wp.array, "numpy", counted):
            key = helper.prepare_device(manager)
        self.assertIsNotNone(key)
        # One read on the host: the free nodes of the structure.
        self.assertEqual(reads, [manager.core.free.ptr])
        solve = helper.device_solve
        self.assertIsNotNone(solve)
        self.assertEqual((flag(), helper.last_reason), (([0], False), "inactive FEM basis manager"))
        # No padded tile is selected and no column is mapped while no launch may use the basis.
        self.assertEqual((solve.capacity, solve.solve_kernel, solve.support), (0, None, None))
        np.testing.assert_array_equal(solve.size_flags.numpy(), 0)
        np.testing.assert_array_equal(solve.support_column.numpy(), -1)
        storage = [array.ptr for array in helper.owners + solve.owners]

        def unusable(reason):
            # Nothing is read from the device and nothing is written to it: the flag is 0 already.
            with (
                patch.object(wp.array, "numpy", side_effect=AssertionError("a read on the host")),
                patch.object(wp.array, "assign", side_effect=AssertionError("a write of a map or of the size flags")),
                patch.object(wp.array, "fill_", side_effect=AssertionError("a write of the device flag")),
            ):
                self.assertEqual(helper.prepare_device(manager), key, reason)
            self.assertEqual((flag(), helper.last_reason), (([0], False), reason))
            self.assertIs(helper.device_solve, solve)

        # A later launch ahead of the track, then every other state without a usable basis: an
        # active manager without a support column, a support without a track sample, a track
        # sample without a support column.
        unusable("inactive FEM basis manager")
        basis(True, ())
        unusable("capacity fallback")
        basis(True, support)
        unusable("capacity fallback")
        basis(True, ())
        tracks(1)
        unusable("capacity fallback")
        # The first track sample with the support of its basis: the same key and the same storage,
        # no read of the device, and the launch may use the basis.
        basis(True, support)
        with patch.object(wp.array, "numpy", side_effect=AssertionError("a read on the host")):
            self.assertEqual(helper.prepare_device(manager), key)
        self.assertEqual(flag(), ([1], True))
        self.assertIs(helper.device_solve, solve)
        self.assertEqual([array.ptr for array in helper.owners + solve.owners], storage)
        self.assertEqual(solve.capacity, woodbury_device._padded_samples(1))
        self.assertEqual(int((solve.support_column.numpy() >= 0).sum()), len(support))
        # Its coefficients, and those of the samples after it, are the host's.
        for count in (1, 2, 5):
            tracks(count)
            expected = packet.host()
            actual, enabled = packet.device()
            self.assertEqual(enabled, 1, count)
            _close(actual, expected, helper)
            self.assertEqual(helper.prepare_device(manager), key, count)
        self.assertEqual([array.ptr for array in helper.owners + solve.owners], storage)
        # Back ahead of the track, as after a reset of the scene: the same key, the flag cleared.
        basis(False, ())
        tracks(0)
        self.assertEqual(helper.prepare_device(manager), key)
        self.assertEqual(flag(), ([0], False))

    @unittest.skipUnless(DEVICE.startswith("cuda"), "Conditional CUDA graph replay contract")
    def test_conditional_graph_replays_current_packets_without_host_coefficients(self):
        packet = Packet(41, 9, surface=True)
        helper, pre = packet.helper, packet.pre
        expected = packet.host()
        key = helper.prepare_device(packet.manager)
        self.assertIsNotNone(key)
        # As in the coupled loop, nothing runs eagerly first: binding alone must
        # have loaded the factorization kernel, which a capture does not preload.
        variants = helper.device_solve.solve_kernel.module.execs
        self.assertIn((pre.device.context, helper.device_solve.block_dim), variants)
        owners = helper.owners + helper.device_solve.owners
        remaining = wp.zeros(1, dtype=int, device=pre.device)
        applied = wp.zeros(1, dtype=int, device=pre.device)

        def correction():
            helper.device_solve.enqueue()
            helper.initialize()

        def trial():
            wp.capture_if(applied, correction)
            wp.launch(countdown, 1, [remaining], device=pre.device)

        with ScopedCapture(device=pre.device) as capture:
            helper.device_solve.enqueue_topology()
            wp.capture_while(remaining, trial)

        def replay(trials, apply):
            helper.coefficients.fill_(np.nan)
            helper.enabled.fill_(-1)
            helper.selected.fill_(-1)
            pre.z.fill_(wp.vec3(7.0))
            remaining.fill_(trials)
            applied.fill_(apply)
            wp.capture_launch(capture.graph)

        replay(3, 1)
        self.assertEqual(helper.enabled.numpy()[0], 1)
        self.assertEqual(helper.selected.numpy()[0], 1)
        _close(helper.coefficients.numpy(), expected, helper)
        first = pre.z.numpy().copy()
        # The untaken branch leaves every output of the correction untouched.
        replay(2, 0)
        self.assertEqual(helper.enabled.numpy()[0], -1)
        np.testing.assert_array_equal(pre.z.numpy(), 7)
        # New residual, tangents and levers are read by the same graph.
        packet.rhs *= -0.75
        packet.matrices[[0, 4]] = packet.matrices[[4, 0]]
        packet.levers += 0.05
        pre.rhs.assign(packet.rhs)
        pre.matrices.assign(packet.matrices)
        pre.levers.assign(packet.levers)
        packet.matrix_device.assign(packet.operator())
        expected = packet.host()
        self.assertEqual(helper.prepare_device(packet.manager), key)
        self.assertEqual([a.ptr for a in owners], [a.ptr for a in helper.owners + helper.device_solve.owners])
        replay(1, 1)
        self.assertEqual(helper.enabled.numpy()[0], 1)
        self.assertEqual(helper.selected.numpy()[0], 1)
        _close(helper.coefficients.numpy(), expected, helper)
        self.assertFalse(np.allclose(pre.z.numpy(), -0.75 * first, rtol=1e-3, atol=0))
        # An untrusted packet falls back inside the graph and recovers afterwards.
        packet.matrices[1] = np.diag([0.2, -0.01, 0.1])
        pre.matrices.assign(packet.matrices)
        replay(2, 1)
        self.assertEqual(helper.enabled.numpy()[0], 0)
        self.assertEqual(helper.selected.numpy()[0], 0)
        np.testing.assert_array_equal(pre.z.numpy(), 0)
        packet.matrices[1] = packet.matrices[2]
        pre.matrices.assign(packet.matrices)
        expected = packet.host()
        replay(2, 1)
        self.assertEqual(helper.enabled.numpy()[0], 1)
        _close(helper.coefficients.numpy(), expected, helper)


def _patterns(tracks):
    """Tangent ranks along a track of ``tracks`` samples, by name.

    Both shaft laws write a zero tangent for a disengaged or unloaded anchor, the transverse
    projection (rank 2) for a sliding one and rank 3 with the stick term; a track that entered
    and left the tissue is disengaged at both ends. Rank 1 is produced by neither law.
    """
    third = tracks // 3
    return {
        "mixed": (2, 3, 3, 0, 2),
        "sliding": (2,),
        "sticking": (3,),
        "disengaged": (0,),
        "disengaged ends": tuple(0 if i < third or i >= tracks - third else 2 + i % 2 for i in range(tracks)),
        "with rank one": (1, 2, 0, 3),
    }


def _launched(call):
    """Names of the kernels that ``call`` launches, in order (the tile factorizations are not listed)."""
    names = []
    real = wp.launch

    def launch(kernel, *args, **kwargs):
        names.append(kernel.key)
        return real(kernel, *args, **kwargs)

    with patch.object(wp, "launch", side_effect=launch):
        call()
    return names


def _scratch(helper):
    """Every array the device coefficients write, by name."""
    # Not the column map and the size flags: the host writes those, as it writes the live counts.
    arrays = {
        name: value
        for name, value in vars(helper.device_solve).items()
        if isinstance(value, wp.array) and name not in ("support_column", "size_flags")
    }
    arrays.update(coefficients=helper.coefficients, enabled=helper.enabled, surface_active=helper.surface_active)
    return arrays


def _reset(helper):
    """Start every output from a value that no kernel writes."""
    for value in _scratch(helper).values():
        value.fill_(-5 if value.dtype is wp.int32 else -7.25)


def _snapshot(helper):
    return {name: value.numpy().copy() for name, value in _scratch(helper).items()}


class CompactedSolveTests(unittest.TestCase):
    def test_compacted_solve_equals_the_padded_solve_for_every_track_count_and_column_pattern(self):
        compacted = too_many = dropped_columns = 0
        for tracks in range(1, _MAX_TRACKS + 1):
            for name, ranks in _patterns(tracks).items():
                with self.subTest(tracks=tracks, pattern=name):
                    packet = Packet(400 + tracks, tracks, ranks=ranks)
                    helper = packet.helper
                    expected = packet.host()
                    # The branch a captured graph takes for this track count against the host solve.
                    actual, enabled = packet.device()
                    self.assertEqual(enabled, 1)
                    _close(actual, expected, helper)
                    solve = helper.device_solve
                    capacity = solve.capacity
                    # Both solves of the same inputs, whichever of them the package chose.
                    solve.solution.fill_(np.nan)
                    solve.solve_valid.fill_(-1)
                    solve._enqueue_padded()
                    self.assertEqual(solve.solve_valid.numpy()[0], 1)
                    padded = solve.solution.numpy()[: 3 * capacity].copy()
                    matrix, rhs = solve.matrix.numpy(), solve.rhs.numpy()
                    rows = nonzero_columns(solve.factors.numpy()[:tracks])
                    if name == "disengaged":
                        self.assertEqual(rows, [])
                    if name == "sticking":
                        self.assertEqual(rows, list(range(3 * tracks)))
                    dropped = np.setdiff1d(np.arange(3 * _MAX_TRACKS), rows)
                    # A dropped column leaves a row of the identity and a zero right-hand side,
                    # so the padded solve returns exactly zero for it.
                    np.testing.assert_array_equal(matrix[dropped][:, dropped], np.eye(len(dropped)))
                    np.testing.assert_array_equal(matrix[dropped][:, rows], 0)
                    np.testing.assert_array_equal(matrix[rows][:, dropped], 0)
                    np.testing.assert_array_equal(rhs[dropped], 0)
                    np.testing.assert_array_equal(padded[dropped[dropped < 3 * capacity]], 0)
                    # The device orders the nonzero columns as the reference does.
                    solve.compact_index.fill_(-5)
                    solve._enqueue_compact_index()
                    self.assertEqual(solve.compact_count.numpy()[0], len(rows))
                    self.assertEqual(solve.compact_fits.numpy()[0], int(len(rows) <= _COMPACT_ROWS))
                    held = min(len(rows), _COMPACT_ROWS)
                    index = solve.compact_index.numpy()
                    np.testing.assert_array_equal(index[:held], rows[:held])
                    np.testing.assert_array_equal(index[held:], -5)
                    position = np.full(3 * _MAX_TRACKS, -1)
                    position[rows] = np.arange(len(rows))
                    np.testing.assert_array_equal(solve.compact_position.numpy(), position)
                    dropped_columns += 3 * tracks - len(rows)
                    if len(rows) > _COMPACT_ROWS:
                        too_many += 1
                        continue
                    solve.solution.fill_(np.nan)
                    solve.solve_valid.fill_(-1)
                    solve._enqueue_compacted()
                    self.assertEqual(solve.solve_valid.numpy()[0], 1)
                    # The compacted system is the padded one on its nonzero columns, value by value,
                    # and an identity with a zero right-hand side beyond them.
                    count = len(rows)
                    system, right = solve.compact_matrix.numpy(), solve.compact_rhs.numpy()
                    np.testing.assert_array_equal(system[:count, :count], matrix[rows][:, rows])
                    np.testing.assert_array_equal(right[:count], rhs[rows])
                    np.testing.assert_array_equal(system[count:, count:], np.eye(_COMPACT_ROWS - count))
                    np.testing.assert_array_equal(system[count:, :count], 0)
                    np.testing.assert_array_equal(system[:count, count:], 0)
                    np.testing.assert_array_equal(right[count:], 0)
                    # Every padded row is written; a dropped unknown is its right-hand side, exactly zero.
                    solution = solve.solution.numpy()
                    np.testing.assert_array_equal(solution[dropped], 0)
                    # Equal to rounding: both are FP32 Cholesky solves of this system, compared
                    # with its FP64 solution within the bound of a backward stable solve.
                    reference = compacted_solution(matrix, rhs, rows)
                    bound = _bound(matrix[rows][:, rows], 3 * tracks) * np.abs(reference).max()
                    np.testing.assert_allclose(solution, reference, rtol=0, atol=bound)
                    np.testing.assert_allclose(padded, reference[: 3 * capacity], rtol=0, atol=bound)
                    compacted += 1
        # Zero columns were dropped, and the sticking tracks beyond 38 samples do not fit the tile.
        self.assertGreater(dropped_columns, 0)
        self.assertGreaterEqual(too_many, _MAX_TRACKS - _COMPACT_ROWS // 3)
        self.assertEqual(compacted + too_many, 6 * _MAX_TRACKS)

    def test_compacted_tile_is_used_once_the_padded_tile_is_larger_and_the_columns_fit(self):
        def exact_rank_two(packet, tracks):
            # A tangent whose third row and column are zero has exactly two nonzero factor columns.
            packet.matrices[list(tracks)] = np.diag([0.3, 0.1, 0.0]).astype(np.float32)
            packet.pre.matrices.assign(packet.matrices)
            packet.matrix_device.assign(packet.operator())

        cases = (
            # samples, ranks, exact rank-2 samples, nonzero columns, compacted, compacted kernels launched
            (17, (3,), (), 51, False, False),  # up to 38 samples the padded system fits the same tile
            (38, (3,), (), _COMPACT_ROWS, False, False),
            (39, (3,), (36, 37, 38), _COMPACT_ROWS, True, True),  # the compacted tile is full
            (39, (3,), (37, 38), _COMPACT_ROWS + 1, False, True),  # one column more than it holds
            (39, (2,), (), 78, True, True),
            (40, (3,), (), 120, False, True),
            (40, (3,), range(34, 40), _COMPACT_ROWS, True, True),
            (41, (3,), range(32, 41), _COMPACT_ROWS, True, True),
            (48, (3,), (), 144, False, True),
            (48, (0,), (), 0, True, True),
            (48, (3,), range(18, 48), _COMPACT_ROWS, True, True),
            (48, (3,), range(19, 48), _COMPACT_ROWS + 1, False, True),
        )
        for tracks, ranks, rank_two, columns, compacted, indexed in cases:
            with self.subTest(tracks=tracks, columns=columns):
                packet = Packet(500 + tracks, tracks, ranks=ranks)
                exact_rank_two(packet, rank_two)
                helper = packet.helper
                expected = packet.host()
                self.assertIsNotNone(helper.prepare_device(packet.manager))
                solve = helper.device_solve
                _reset(helper)
                names = _launched(lambda: (solve.enqueue_topology(), _correction(solve)))  # noqa: B023
                self.assertEqual(helper.enabled.numpy()[0], 1)
                self.assertEqual(solve.solve_valid.numpy()[0], 1)
                self.assertEqual(_compacted(solve), compacted)
                _close(helper.coefficients.numpy(), expected, helper)
                self.assertEqual("schur_compact_index" in names, indexed)
                self.assertEqual("schur_compact_assemble" in names, compacted)
                self.assertEqual("schur_compact_scatter" in names, compacted)
                self.assertEqual("schur_assemble" in names, not compacted)
                if indexed:
                    self.assertEqual(solve.compact_count.numpy()[0], columns)
                    self.assertEqual(solve.compact_fits.numpy()[0], int(compacted))
                else:
                    self.assertEqual(len(nonzero_columns(solve.factors.numpy()[:tracks])), columns)
                    np.testing.assert_array_equal(solve.compact_count.numpy(), -5)
                    np.testing.assert_array_equal(solve.compact_fits.numpy(), -5)
                # Only the system that was factorized is assembled.
                self.assertEqual(bool((solve.compact_matrix.numpy() == -7.25).all()), not compacted)
                self.assertEqual(bool((solve.matrix.numpy() == -7.25).all()), compacted)
                # The proposal passes the residual certificate of the current operator.
                helper.initialize()
                self.assertEqual(helper.selected.numpy()[0], 1)

    def test_invalid_inputs_clear_the_flag_on_the_compacted_path(self):
        with self.subTest("indefinite Schur system"):
            packet = Packet(628, 40)
            flat = packet.manager.core.basis.numpy()
            flat[: packet.response.size] *= -1e3
            packet.manager.core.basis.assign(flat)
            packet.manager.basis_builds += 1
            self.assertEqual(packet.device()[1], 0)
            solve = packet.helper.device_solve
            self.assertTrue(_compacted(solve))
            np.testing.assert_array_equal(solve.solve_valid.numpy(), [0])
            np.testing.assert_array_equal(solve.factor_valid.numpy()[:40], 1)
        with self.subTest("nonfinite anchor tangent"):
            packet = Packet(625, 40)
            matrices = packet.matrices.copy()
            matrices[5, 1, 1] = np.inf
            packet.pre.matrices.assign(matrices)
            self.assertEqual(packet.device()[1], 0)
            packet.pre.z.fill_(wp.vec3(1.0))
            packet.helper.initialize()
            self.assertEqual(packet.helper.selected.numpy()[0], 0)
            np.testing.assert_array_equal(packet.pre.z.numpy(), 0)
            # Restored inputs are accepted again by the same persistent scratch.
            packet.pre.matrices.assign(packet.matrices)
            self.assertEqual(packet.device()[1], 1)

    def test_nonfinite_data_of_dropped_columns_clears_the_flag_as_the_padded_solve_does(self):
        def flags(packet, poison=None):
            """Validity flag of the compacted and of the padded solve of the same inputs."""
            helper = packet.helper
            self.assertIsNotNone(helper.prepare_device(packet.manager))
            solve = helper.device_solve
            result = []
            for compact in (None, False):
                _reset(helper)
                solve.enqueue_topology()
                if poison is not None:
                    poison(solve)
                _correction(solve, compact=compact)
                result.append(int(helper.enabled.numpy()[0]))
                # The columns fit, so the first of the two solves is the compacted one.
                self.assertEqual(int(solve.compact_fits.numpy()[0]), 1)
                self.assertEqual(bool((solve.compact_matrix.numpy() == -7.25).all()), compact is False)
            return result

        def zero_start(packet):
            packet.pre.z.fill_(wp.vec3(1.0))
            packet.helper.initialize()
            self.assertEqual(packet.helper.selected.numpy()[0], 0)
            np.testing.assert_array_equal(packet.pre.z.numpy(), 0)

        def nan_rows(name, track):
            def poison(solve):
                values = getattr(solve, name).numpy()
                values[3 * track : 3 * track + 3] = np.nan
                if name == "compliance":
                    values[:, 3 * track : 3 * track + 3] = np.nan
                getattr(solve, name).assign(values)

            return poison

        # Sample 3 of the mixed pattern is disengaged: its tangent, and with it its factor, is zero.
        dropped = 3
        with self.subTest("finite data"):
            packet = Packet(631, 40)
            self.assertFalse(packet.matrices[dropped].any())
            self.assertEqual(flags(packet), [1, 1])
        with self.subTest("tissue residual, every column dropped"):
            packet = Packet(632, 40, ranks=(0,))
            free = np.flatnonzero(packet.fixed == 0)
            rhs = packet.rhs.copy()
            rhs[free[packet.manager.support[4] // 3], packet.manager.support[4] % 3] = np.nan
            packet.pre.rhs.assign(rhs)
            self.assertEqual(flags(packet), [0, 0])
            self.assertEqual(packet.helper.device_solve.compact_count.numpy()[0], 0)
            zero_start(packet)
        with self.subTest("response row of a dropped sample"):
            packet = Packet(633, 40)
            self.assertEqual(flags(packet, nan_rows("tissue", dropped)), [0, 0])
            # The rows of every kept column are finite: only the dropped unknowns carry it.
            solve = packet.helper.device_solve
            _reset(packet.helper)
            solve.enqueue_topology()
            nan_rows("tissue", dropped)(solve)
            _correction(solve, compact=True)
            solution = solve.solution.numpy()[:120]
            self.assertTrue(np.isnan(solution[3 * dropped : 3 * dropped + 3]).all())
            self.assertTrue(np.isfinite(np.delete(solution, range(3 * dropped, 3 * dropped + 3))).all())
            self.assertEqual(solve.solve_valid.numpy()[0], 1)
        with self.subTest("lever of a dropped sample"):
            packet = Packet(634, 40)
            levers = packet.levers.copy()
            levers[dropped, 1] = np.inf
            packet.pre.levers.assign(levers)
            self.assertEqual(flags(packet), [0, 0])
        with self.subTest("basis entry under a dropped sample"):
            # As the kernels produce it: the response basis is nonfinite at a node of a disengaged track.
            packet = Packet(635, 40, ranks=(0,))
            free = np.flatnonzero(packet.fixed == 0)
            node = next(
                int(n) for n, w in zip(packet.nodes[dropped], packet.bary[dropped], strict=True) if w and n in free
            )
            flat = packet.manager.core.basis.numpy()
            flat[3 * int(np.searchsorted(free, node)) + 1] = np.inf
            packet.manager.core.basis.assign(flat)
            packet.manager.basis_builds += 1
            self.assertEqual(flags(packet), [0, 0])
            zero_start(packet)
        with self.subTest("compliance entry alone"):
            # Not what ``schur_topology`` writes: it forms the compliance from the entries of the basis
            # that the response rows hold, and those are finite here. The compacted solve never reads the
            # compliance of a dropped sample, so the proposal is the host's and the certificate decides.
            packet = Packet(636, 40)
            expected = packet.host()
            self.assertEqual(flags(packet, nan_rows("compliance", dropped)), [1, 0])
            helper = packet.helper
            _reset(helper)
            helper.device_solve.enqueue_topology()
            nan_rows("compliance", dropped)(helper.device_solve)
            _correction(helper.device_solve)
            self.assertTrue(_compacted(helper.device_solve))
            _close(helper.coefficients.numpy(), expected, helper)
            helper.initialize()
            self.assertEqual(helper.selected.numpy()[0], 1)

    def test_correction_outside_a_capture_factorizes_the_padded_system_without_reading_the_flag(self):
        # 40 samples of mixed ranks: the nonzero columns fit the compacted tile.
        packet = Packet(641, 40)
        helper = packet.helper
        expected = packet.host()
        self.assertIsNotNone(helper.prepare_device(packet.manager))
        solve = helper.device_solve
        self.assertGreater(3 * solve.capacity, _COMPACT_ROWS)
        _reset(helper)
        solve.enqueue_topology()
        # Outside a capture ``wp.capture_if`` copies its condition to the host and waits for it.
        with patch.object(wp, "capture_if", side_effect=AssertionError("a read of the flag on the host")):
            names = _launched(solve.enqueue)
        self.assertEqual(names, ["schur_inputs", "schur_assemble", "schur_coefficients"])
        self.assertEqual(helper.enabled.numpy()[0], 1)
        self.assertEqual(solve.solve_valid.numpy()[0], 1)
        # Nothing of the compacted system was written, the flag included.
        for name in ("compact_index", "compact_position", "compact_count", "compact_fits"):
            np.testing.assert_array_equal(getattr(solve, name).numpy(), -5, name)
        for name in ("compact_matrix", "compact_rhs", "compact_solution"):
            np.testing.assert_array_equal(getattr(solve, name).numpy(), -7.25, name)
        _close(helper.coefficients.numpy(), expected, helper)
        helper.initialize()
        self.assertEqual(helper.selected.numpy()[0], 1)
        # The same inputs with the branch of a captured graph: the compacted system, the same proposal.
        padded = helper.coefficients.numpy().copy()
        _reset(helper)
        solve.enqueue_topology()
        _correction(solve)
        self.assertTrue(_compacted(solve))
        _close(helper.coefficients.numpy(), expected, helper)
        _close(helper.coefficients.numpy(), padded, helper)

    def test_compacted_kernels_are_created_and_loaded_with_the_scratch(self):
        # Four samples are far below the track counts that use the compacted tile.
        packet = Packet(71, 4)
        device = packet.pre.device
        with patch.object(wp, "load_module", wraps=wp.load_module) as load:
            self.assertIsNotNone(packet.helper.prepare_device(packet.manager))
        solve = packet.helper.device_solve
        self.assertEqual(solve.capacity, 4)
        self.assertIs(solve.compact_kernel, woodbury_device._solve_tiles(_COMPACT_ROWS))
        self.assertEqual(solve.compact_matrix.shape, (_COMPACT_ROWS, _COMPACT_ROWS))
        loaded = {(call.args[0], call.kwargs["block_dim"]) for call in load.call_args_list}
        tile = (solve.compact_kernel.module, _COMPACT_BLOCK_DIM if device.is_cuda else 1)
        ordinary = (woodbury_device.schur_compact_index.module, _LAUNCH_BLOCK_DIM if device.is_cuda else 1)
        self.assertIn(tile, loaded)
        self.assertIn(ordinary, loaded)
        # Loaded for the thread blocks they are launched with, before any launch.
        self.assertIn((device.context, tile[1]), tile[0].execs)
        self.assertIn((device.context, ordinary[1]), ordinary[0].execs)
        for kernel in ("schur_compact_index", "schur_compact_assemble", "schur_compact_scatter"):
            self.assertIs(getattr(woodbury_device, kernel).module, ordinary[0])
        # The padded system of 17 to 38 samples is factorized by that kernel: binding them loads nothing.
        for count in (17, _COMPACT_ROWS // 3):
            packet.pre.count = count
            with patch.object(wp, "load_module", wraps=wp.load_module) as load:
                self.assertIsNotNone(packet.helper.prepare_device(packet.manager))
            load.assert_not_called()
            self.assertEqual(solve.capacity, _COMPACT_ROWS // 3)
            self.assertIs(solve.solve_kernel, solve.compact_kernel)
            self.assertEqual(solve.block_dim, _COMPACT_BLOCK_DIM)
        # More samples bind a larger padded tile for the columns that do not fit, and no other compacted one.
        packet.pre.count = 40
        with patch.object(wp, "load_module", wraps=wp.load_module) as load:
            self.assertIsNotNone(packet.helper.prepare_device(packet.manager))
        load.assert_not_called()
        self.assertEqual(solve.capacity, 40)
        self.assertIs(solve.compact_kernel, woodbury_device._solve_tiles(_COMPACT_ROWS))
        self.assertIsNot(solve.solve_kernel, solve.compact_kernel)

    def test_factorization_of_every_padded_tile_is_created_and_loaded_with_the_scratch(self):
        packet = Packet(72, 1)
        device = packet.pre.device
        with patch.object(wp, "load_module", wraps=wp.load_module) as load:
            self.assertIsNotNone(packet.helper.prepare_device(packet.manager))
        solve = packet.helper.device_solve
        loaded = {(call.args[0], call.kwargs["block_dim"]) for call in load.call_args_list}
        # 12, 24 and 48 rows in the smallest thread block, the compacted tile, then 120 and 144 rows.
        rows = [3 * capacity for capacity in woodbury_device._TILES]
        self.assertEqual(rows, [12, 24, 48, _COMPACT_ROWS, 120, 144])
        blocks = [block_dim for _, block_dim in solve.tiles]
        self.assertEqual(blocks, [64, 64, 64, _COMPACT_BLOCK_DIM, 128, 256])
        modules = []
        for size, (kernel, block_dim) in zip(rows, solve.tiles, strict=True):
            self.assertIs(kernel, woodbury_device._solve_tiles(size))
            # One thread per row at least.
            self.assertGreaterEqual(block_dim, size)
            launched = block_dim if device.is_cuda else 1
            self.assertIn((kernel.module, launched), loaded)
            self.assertIn((device.context, launched), kernel.module.execs)
            modules.append(kernel.module)
        self.assertEqual(len({id(module) for module in modules}), len(rows))
        # No later track count loads or builds anything, and neither does a scene that loads them at start-up.
        built = [set(module.execs) for module in modules]
        with patch.object(wp, "load_module", wraps=wp.load_module) as load:
            for count in range(1, _MAX_TRACKS + 1):
                packet.pre.count = count
                self.assertIsNotNone(packet.helper.prepare_device(packet.manager))
        load.assert_not_called()
        woodbury_device.load_kernels(device)
        self.assertEqual([set(module.execs) for module in modules], built)

    def test_no_launch_of_a_correction_loads_a_module(self):
        # Fitting and not fitting columns at 40 samples: both branches of the conditional.
        for ranks, compacted in (((2, 3, 3, 0, 2), True), ((3,), False)):
            with self.subTest(compacted=compacted):
                packet = Packet(651, 40, ranks=ranks)
                helper = packet.helper
                self.assertIsNotNone(helper.prepare_device(packet.manager))
                solve = helper.device_solve
                modules = (
                    woodbury_device.schur_compact_index.module,
                    solve.compact_kernel.module,
                    solve.solve_kernel.module,
                )
                self.assertEqual(len({id(module) for module in modules}), 3)
                loaded = [set(module.execs) for module in modules]
                blocks = {}
                real = wp.launch

                def launch(kernel, *args, _blocks=blocks, _real=real, **kwargs):
                    _blocks[kernel.key] = kwargs.get("block_dim")
                    return _real(kernel, *args, **kwargs)

                with patch.object(wp, "launch", side_effect=launch):
                    solve.enqueue_topology()
                    _correction(solve)
                self.assertEqual(_compacted(solve), compacted)
                self.assertEqual(helper.enabled.numpy()[0], 1)
                # Every launch names the thread block that the module was loaded for with the scratch.
                names = {"schur_topology", "schur_inputs", "schur_compact_index", "schur_coefficients"}
                names |= {"schur_compact_assemble", "schur_compact_scatter"} if compacted else {"schur_assemble"}
                self.assertEqual(blocks, dict.fromkeys(names, _LAUNCH_BLOCK_DIM))
                self.assertEqual([set(module.execs) for module in modules], loaded)

    # On every device: Warp captures on the CPU device as well, so a CPU-only run takes the branches of ``enqueue``.
    def test_captured_compacted_solve_follows_the_device_counts_and_columns(self):
        # One graph per padded tile: 17 to 38 samples (no compaction), 39 and 40, 41 to 48.
        for tracks, fewest in ((_COMPACT_ROWS // 3, 17), (40, 39), (48, 41)):
            with self.subTest(tracks=tracks):
                packet = Packet(81 + tracks, tracks)
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
                    wp.launch(countdown, 1, [remaining], device=pre.device)

                # Captured with the fewest samples of this factorization.
                set_tracks(fewest)
                with ScopedCapture(device=pre.device) as capture:
                    solve.enqueue_topology()
                    wp.capture_while(remaining, trial)
                sticking = np.asarray([_tangent(np.random.default_rng(7), 3) for _ in range(tracks)])
                indexed = 3 * solve.capacity > _COMPACT_ROWS
                seen = []
                for count, matrices, compacted in (
                    (fewest, packet.matrices, indexed),
                    (tracks, packet.matrices, indexed),
                    # More nonzero columns than the compacted tile holds: the same graph falls back.
                    (tracks, sticking, False),
                    (fewest, sticking, False),
                    (tracks, packet.matrices, indexed),
                ):
                    with self.subTest(count=count, compacted=compacted):
                        set_tracks(count)
                        pre.matrices.assign(matrices)
                        _reset(helper)
                        remaining.fill_(2)
                        wp.capture_launch(capture.graph)
                        replayed = _snapshot(helper)
                        self.assertEqual(replayed["enabled"].tolist(), [1])
                        self.assertEqual(replayed["compact_fits"].tolist(), [int(compacted) if indexed else -5])
                        self.assertEqual(_compacted(solve), compacted)
                        # The replay is the proposal of the host solve at this count and these tangents.
                        tolerance = _tolerance(helper)
                        expected = packet.host()
                        used = np.r_[np.arange(len(helper.support)), np.arange(_CAPACITY, _CAPACITY + 6)]
                        np.testing.assert_allclose(
                            replayed["coefficients"][used],
                            expected[used],
                            rtol=0,
                            atol=tolerance * np.abs(expected[used]).max(),
                        )
                        # And it holds the bits of the same launches issued one by one, with its branch.
                        _reset(helper)
                        solve.enqueue_topology()
                        _correction(solve)
                        direct = _snapshot(helper)
                        self.assertEqual(sorted(replayed), sorted(direct))
                        for name in replayed:
                            self.assertEqual(replayed[name].tobytes(), direct[name].tobytes(), name)
                        seen.append(replayed["coefficients"].tobytes())
                # The replays differ with the count and with the tangents, and repeat with them.
                self.assertEqual(len(set(seen)), 4)
                self.assertEqual(seen[1], seen[4])
                set_tracks(tracks)
                self.assertEqual(helper.prepare_device(packet.manager), key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
