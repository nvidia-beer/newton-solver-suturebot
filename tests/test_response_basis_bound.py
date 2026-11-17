# SPDX-License-Identifier: Apache-2.0
"""The bound on the response columns that one attempt solves (``AffineGuessManager.nodes_per_attempt``).

The manager serves CUDA scenes only. These checks run its support logic on the CPU with a
provider, a core and a solver that record what they are asked for: which columns are solved
in which attempt, when the basis is loaded, and that the guess is withheld until then. The
Warp provider is then checked by itself, on the test device: the columns of a bounded build
are those of an unbounded one, and an extension allocates and reads nothing.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from test_affine_guess import DEVICE, sparse

from newton_solver_suturebot._src.solvers.fem.affine_guess import AffineGuessManager
from newton_solver_suturebot._src.solvers.fem.warp_basis_provider import WarpBasisProvider


class _Requests:
    """Keeps the columns of the support it was last asked for; every solve is a request."""

    def __init__(self):
        self.support = ()
        self.requests, self.solved = [], []
        self.invalidations = 0

    def _hold(self, support):
        support = tuple(int(dof) for dof in support)
        self.solved.append(tuple(dof for dof in support if dof not in self.support))
        self.support = support
        return support

    def __call__(self, matrix, support):
        support = self._hold(support)
        self.requests.append(support)
        return ("columns", support), {"columns": len(support), "new_columns": len(self.solved[-1])}

    def invalidate(self):
        self.support = ()
        self.invalidations += 1


class _Provider(_Requests):
    """As the Warp provider: columns can be solved ahead of the request that returns them."""

    def __init__(self):
        super().__init__()
        self.extensions = []

    def extend(self, matrix, support):
        self.extensions.append(self._hold(support))
        return len(self.solved[-1])


class _Core:
    def __init__(self):
        self.loaded, self.enabled, self.resets = [], [], 0

    def load_basis(self, columns, support):
        self.loaded.append((columns, tuple(support)))

    def set_enabled(self, value):
        self.enabled.append(bool(value))

    def reset(self):
        self.resets += 1


def _manager(nodes_per_attempt, *, free_nodes=12, provider=None):
    array = SimpleNamespace(ptr=1, size=free_nodes)
    matrix = SimpleNamespace(nrow=free_nodes, ncol=free_nodes, nnz=1, offsets=array, columns=array, values=array)
    fixed = SimpleNamespace(
        compact_matrix=matrix, compact_operator=object(), free=array, free_host=np.arange(100, 100 + free_nodes)
    )
    solver = SimpleNamespace(_fixed_particles=fixed, _tangent_version=0)
    provider = _Provider() if provider is None else provider
    manager = AffineGuessManager(solver, provider, capacity=3 * free_nodes, nodes_per_attempt=nodes_per_attempt)
    manager.core = _Core()
    manager.structural_key = manager._structure()
    interaction = SimpleNamespace(track_count=0, _track_nodes_host=[])
    coupler = SimpleNamespace(interaction=interaction, _refresh_projection=lambda: None)
    return manager, provider, coupler


def _attempt(manager, coupler):
    with patch.object(AffineGuessManager, "_supported", return_value=True):
        manager.begin_attempt(coupler, None, None)
    return manager.active


def _dofs(*nodes):
    return tuple(3 * node + axis for node in nodes for axis in range(3))


class ResponseBasisBoundTests(unittest.TestCase):
    def _track(self, coupler, *tetrahedra):
        # Model nodes; free node k is model node 100 + k.
        coupler.interaction._track_nodes_host = [np.asarray(nodes) + 100 for nodes in tetrahedra]
        coupler.interaction.track_count = len(tetrahedra)

    def test_without_the_bound_every_missing_column_is_solved_in_the_attempt_after_the_growth(self):
        manager, provider, coupler = _manager(None)
        self._track(coupler, (0, 1, 2, 3))
        self.assertTrue(_attempt(manager, coupler))
        self.assertEqual((provider.requests, provider.extensions), ([_dofs(0, 1, 2, 3)], []))
        self.assertEqual(manager.core.loaded, [(("columns", _dofs(0, 1, 2, 3)), _dofs(0, 1, 2, 3))])
        self.assertEqual((manager.basis_builds, manager.deferred_builds), (1, 0))
        # An unchanged track asks for nothing.
        self.assertTrue(_attempt(manager, coupler))
        self.assertEqual(len(provider.requests), 1)

    def test_first_sample_of_a_section_is_solved_one_node_per_attempt_and_loaded_when_complete(self):
        manager, provider, coupler = _manager(1)
        self._track(coupler, (4, 1, 7, 2))
        active = [_attempt(manager, coupler) for _ in range(4)]
        # Three attempts solve the columns of one node each and withhold the guess; the fourth
        # solves the last node, loads the basis and applies it.
        self.assertEqual(active, [False, False, False, True])
        self.assertEqual(provider.solved, [_dofs(1), _dofs(2), _dofs(4), _dofs(7)])
        # The three attempts that do not load extend what the provider holds; the one request
        # is that of the attempt that loads, as in an unbounded build.
        self.assertEqual(provider.extensions, [_dofs(1), _dofs(1, 2), _dofs(1, 2, 4)])
        self.assertEqual(provider.requests, [_dofs(1, 2, 4, 7)])
        self.assertEqual(manager.core.loaded, [(("columns", _dofs(1, 2, 4, 7)), _dofs(1, 2, 4, 7))])
        self.assertEqual((manager.basis_builds, manager.deferred_builds), (1, 3))
        self.assertEqual(manager.support, _dofs(1, 2, 4, 7))
        self.assertEqual(manager.status, "guarded stale-basis guess")
        # The reports say which requests were loaded.
        self.assertEqual([report.get("loaded", True) for report in manager.preparation_reports], [False] * 3 + [True])
        self.assertEqual([report["new_columns"] for report in manager.preparation_reports], [3, 3, 3, 3])
        self.assertEqual([report["support_dofs"] for report in manager.preparation_reports], [3, 6, 9, 12])
        # The loaded basis is the one an unbounded build loads.
        unbounded, _, other = _manager(None)
        self._track(other, (4, 1, 7, 2))
        _attempt(unbounded, other)
        self.assertEqual(unbounded.core.loaded, manager.core.loaded)
        # Nothing more is asked for while the track stands.
        self.assertTrue(_attempt(manager, coupler))
        self.assertEqual((len(provider.extensions), len(provider.requests)), (3, 1))

    def test_provider_without_extension_is_asked_for_the_partial_support(self):
        manager, provider, coupler = _manager(1, provider=_Requests())
        self._track(coupler, (4, 1, 7, 2))
        active = [_attempt(manager, coupler) for _ in range(4)]
        self.assertEqual(active, [False, False, False, True])
        self.assertEqual(provider.solved, [_dofs(1), _dofs(2), _dofs(4), _dofs(7)])
        self.assertEqual(provider.requests, [_dofs(1), _dofs(1, 2), _dofs(1, 2, 4), _dofs(1, 2, 4, 7)])
        # Only the last request is loaded.
        self.assertEqual(manager.core.loaded, [(("columns", _dofs(1, 2, 4, 7)), _dofs(1, 2, 4, 7))])
        self.assertEqual((manager.basis_builds, manager.deferred_builds), (1, 3))

    def test_withheld_guess_reports_its_state_and_keeps_the_old_basis_out(self):
        manager, provider, coupler = _manager(1)
        self._track(coupler, (0, 1, 2, 3))
        for _ in range(4):
            _attempt(manager, coupler)
        loaded = len(manager.core.loaded)
        # A second section: four new nodes. The basis of the first section is not applied meanwhile.
        self._track(coupler, (0, 1, 2, 3), (8, 9, 10, 11))
        self.assertFalse(_attempt(manager, coupler))
        self.assertEqual(manager.status, "solving response columns")
        self.assertFalse(manager.core.enabled[-1])
        self.assertEqual(len(manager.core.loaded), loaded)
        # The provider keeps the columns it has: only the new node is solved.
        self.assertEqual(provider.solved[-1], _dofs(8))
        self.assertEqual(provider.extensions[-1], _dofs(0, 1, 2, 3, 8))
        self.assertEqual(manager.support, _dofs(0, 1, 2, 3))
        active = [_attempt(manager, coupler) for _ in range(3)]
        self.assertEqual(active, [False, False, True])
        self.assertEqual(manager.core.loaded[-1][1], _dofs(0, 1, 2, 3, 8, 9, 10, 11))
        self.assertEqual(provider.solved[-3:], [_dofs(9), _dofs(10), _dofs(11)])

    def test_growth_by_one_node_is_one_attempt_as_without_the_bound(self):
        manager, provider, coupler = _manager(1)
        self._track(coupler, (0, 1, 2, 3))
        for _ in range(4):
            _attempt(manager, coupler)
        requests = len(provider.requests)
        # The next tetrahedron shares a face with the last one: one new node.
        self._track(coupler, (0, 1, 2, 3), (1, 2, 3, 5))
        extensions = len(provider.extensions)
        self.assertTrue(_attempt(manager, coupler))
        self.assertEqual((len(provider.requests), len(provider.extensions)), (requests + 1, extensions))
        self.assertEqual(provider.solved[-1], _dofs(5))
        self.assertEqual(manager.deferred_builds, 3)

    def test_track_that_grows_during_a_build_is_followed(self):
        manager, provider, coupler = _manager(2)
        self._track(coupler, (0, 1, 2, 3))
        self.assertFalse(_attempt(manager, coupler))
        self.assertEqual(provider.solved[-1], _dofs(0, 1))
        self._track(coupler, (0, 1, 2, 3), (1, 2, 3, 6))
        # Three nodes are missing now (2, 3, 6): two in this attempt, the last in the next.
        self.assertFalse(_attempt(manager, coupler))
        self.assertEqual(provider.solved[-1], _dofs(2, 3))
        self.assertTrue(_attempt(manager, coupler))
        self.assertEqual(provider.solved[-1], _dofs(6))
        self.assertEqual(manager.support, _dofs(0, 1, 2, 3, 6))

    def test_structure_change_and_first_core_are_built_at_once(self):
        manager, provider, coupler = _manager(1)
        self._track(coupler, (0, 1, 2, 3))
        manager.structural_key = None  # as after invalidate(): another structure
        with patch("newton_solver_suturebot._src.solvers.fem.affine_guess.DeviceBasisGuess", lambda *a, **k: _Core()):
            coupler.invalidate_fem_graph = lambda: None
            self.assertTrue(_attempt(manager, coupler))
        self.assertEqual((provider.requests, provider.extensions), ([_dofs(0, 1, 2, 3)], []))
        self.assertEqual((manager.basis_builds, manager.deferred_builds), (1, 0))
        self.assertEqual(provider.invalidations, 1)

    def test_storage_prepared_again_during_a_build_starts_the_build_over(self):
        manager, provider, coupler = _manager(1)
        # What prepare() asks of the solver and the coupler beyond the fixture.
        manager.solver.model = SimpleNamespace(device=SimpleNamespace(is_capturing=False))
        coupler.invalidate_fem_graph = lambda: None
        self._track(coupler, (4, 1, 7, 2))
        self.assertEqual([_attempt(manager, coupler) for _ in range(2)], [False, False])
        self.assertEqual(provider.solved, [_dofs(1), _dofs(2)])
        # Another structure, prepared as at a start-up: the provider drops the columns it holds.
        manager.solver._fixed_particles.compact_matrix.nnz = 2
        with (
            patch.object(AffineGuessManager, "_supported", return_value=True),
            patch("newton_solver_suturebot._src.solvers.fem.affine_guess.DeviceBasisGuess", lambda *a, **k: _Core()),
        ):
            self.assertTrue(manager.prepare(coupler, None, None))
        self.assertEqual((provider.invalidations, provider.support), (1, ()))
        solved = len(provider.solved)
        self.assertEqual([_attempt(manager, coupler) for _ in range(4)], [False, False, False, True])
        # One node per attempt again: no column of the build before the preparation counts as held.
        self.assertEqual(provider.solved[solved:], [_dofs(1), _dofs(2), _dofs(4), _dofs(7)])
        self.assertEqual(manager.core.loaded[-1][1], _dofs(1, 2, 4, 7))

    def test_bound_is_validated(self):
        solver, provider = SimpleNamespace(), _Provider()
        for value in (0, -1, 1.5, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AffineGuessManager(solver, provider, nodes_per_attempt=value)
        self.assertIsNone(AffineGuessManager(solver, provider).nodes_per_attempt)
        self.assertEqual(AffineGuessManager(solver, provider, nodes_per_attempt=np.int64(2)).nodes_per_attempt, 2)


class WarpProviderStorageTests(unittest.TestCase):
    """The Warp provider under a bounded build: fixed storage, the same columns, one read per request."""

    @staticmethod
    def matrix():
        n = 36
        A = np.diag(np.linspace(2.0, 5.0, n).astype(np.float32))
        A += np.diag(np.full(n - 1, -0.2, np.float32), 1)
        A += np.diag(np.full(n - 1, -0.2, np.float32), -1)
        return sparse(A)

    @staticmethod
    def storage(engine):
        return [array.ptr for array in (engine.held, engine.ordered, engine.quality, engine.x, engine.b)]

    def test_columns_of_a_bounded_build_are_those_of_an_unbounded_one_bit_for_bit(self):
        matrix = self.matrix()
        support = _dofs(4, 1, 7, 2)
        bounded, unbounded = WarpBasisProvider(), WarpBasisProvider()
        try:
            # The reference of both is the matrix of their first support request.
            bounded.prepare(matrix)
            unbounded.prepare(matrix)
            self.assertEqual(bounded.extend(matrix, support[:3]), 3)
            self.assertEqual(bounded.extend(matrix, support[:6]), 3)
            # A node that sorts before the held ones, as a track that grows backwards adds.
            self.assertEqual(bounded.extend(matrix, _dofs(4, 1, 7)), 3)
            self.assertEqual(bounded.reference_builds, 1)
            columns, meta = bounded(matrix, _dofs(1, 2, 4, 7))
            expected, expected_meta = unbounded(matrix, _dofs(1, 2, 4, 7))
            np.testing.assert_array_equal(columns.numpy().view(np.uint32), expected.numpy().view(np.uint32))
            # The request reports every column solved since the last one, in the order of the solves.
            self.assertEqual((meta["new_columns"], expected_meta["new_columns"]), (12, 12))
            self.assertEqual([row["dof"] for row in meta["column_diagnostics"]], list(_dofs(4, 1, 7, 2)))
            by_dof = {row["dof"]: row for row in expected_meta["column_diagnostics"]}
            for row in meta["column_diagnostics"]:
                self.assertEqual(row, by_dof[row["dof"]])
            self.assertEqual((bounded.solves, unbounded.solves), (4, 1))
            # A support that drops a node frees its slots; the columns that stay are not solved again.
            fewer, meta = bounded(matrix, _dofs(7, 1))
            self.assertEqual(meta["new_columns"], 0)
            np.testing.assert_array_equal(
                fewer.numpy().reshape(6, -1), expected.numpy().reshape(12, -1)[[9, 10, 11, 0, 1, 2]]
            )
            again, meta = bounded(matrix, _dofs(1, 2, 4, 7))
            self.assertEqual(meta["new_columns"], 6)
            np.testing.assert_array_equal(again.numpy().view(np.uint32), expected.numpy().view(np.uint32))
        finally:
            bounded.close()
            unbounded.close()

    def test_extension_and_request_use_the_storage_made_with_the_engine(self):
        matrix = self.matrix()
        device = wp.get_device(DEVICE)
        provider = WarpBasisProvider(capacity=12)
        try:
            provider.prepare(matrix)
            engine = provider.engine
            self.assertEqual((engine.held.size, engine.ordered.size), (12 * 36, 12 * 36))
            pointers = self.storage(engine)
            pooled = device.is_cuda and wp.is_mempool_enabled(device)
            # The first support request takes its reference; it is not what is measured here.
            provider.extend(matrix, _dofs(0))
            used = wp.get_mempool_used_mem_current(device) if pooled else None
            reads, numpy = [], wp.array.numpy

            def counted(array, *arguments, **options):
                reads.append(tuple(array.shape))
                return numpy(array, *arguments, **options)

            refuse = AssertionError("an array made after start-up")
            with (
                patch.object(wp.array, "numpy", counted),
                patch.object(wp, "empty", side_effect=refuse),
                patch.object(wp, "zeros", side_effect=refuse),
                patch.object(wp, "synchronize_stream", side_effect=AssertionError("an extension awaits nothing")),
            ):
                self.assertEqual(provider.extend(matrix, _dofs(0, 5)), 3)
                self.assertEqual(provider.extend(matrix, _dofs(0, 5, 9)), 3)
                # No device read and no array: the columns are in the engine's storage.
                self.assertEqual(reads, [])
            with (
                patch.object(wp.array, "numpy", counted),
                patch.object(wp, "empty", side_effect=refuse),
                patch.object(wp, "zeros", side_effect=refuse),
            ):
                columns, meta = provider(matrix, _dofs(0, 5, 9, 11))
                # One read: the residual packets, for the columns of the extensions and of this request.
                self.assertEqual(reads, [(12, 4)])
                self.assertEqual(meta["new_columns"], 12)
                reads.clear()
                same, meta = provider(matrix, _dofs(0, 5, 9, 11))
                # A request with nothing new reads nothing.
                self.assertEqual((reads, meta["new_columns"]), ([], 0))
            if pooled:
                self.assertEqual(wp.get_mempool_used_mem_current(device), used)
            self.assertEqual(self.storage(engine), pointers)
            # The returned columns are the engine's ordered storage, valid until the next request.
            self.assertEqual((columns.ptr, columns.size), (engine.ordered.ptr, 12 * 36))
            self.assertEqual(same.ptr, columns.ptr)
            self.assertIs(provider.columns, same)
            self.assertEqual(provider.support, _dofs(0, 5, 9, 11))
            self.assertEqual(provider.solves, 4)
        finally:
            provider.close()

    def test_nonfinite_column_of_an_extension_is_refused_by_the_request_that_returns_it(self):
        matrix = self.matrix()
        provider = WarpBasisProvider()
        try:
            provider.extend(matrix, _dofs(0))
            engine = provider.engine
            packet = engine.quality.numpy()
            packet[provider._slots[1], 0] = np.nan
            engine.quality.assign(packet)
            with self.assertRaisesRegex(RuntimeError, "Nonfinite approximate response column"):
                provider(matrix, _dofs(0, 1))
            # The refused packets stay unread: a second request refuses them again.
            with self.assertRaisesRegex(RuntimeError, "Nonfinite approximate response column"):
                provider(matrix, _dofs(0, 1))
        finally:
            provider.close()


if __name__ == "__main__":
    unittest.main()
