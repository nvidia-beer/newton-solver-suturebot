"""CPU kernel/ownership contracts on the frozen actual shaft system."""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp
from _woodbury_reference import block_diagonal, interpolation, rigid_interpolation

from newton_solver_suturebot._src.solvers.coupled.acceleration.shaft_preconditioner import _ShaftPreconditioner
from newton_solver_suturebot._src.solvers.coupled.acceleration.woodbury_guess import (
    _HEADER,
    WoodburyInitialGuess,
    host_coefficients,
)
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import LIVE_TRACK
from newton_solver_suturebot._src.utils.capture import ScopedCapture

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


@wp.kernel(enable_backward=False)
def dense_matvec(
    matrix: wp.array2d[float],
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    out: wp.array[wp.vec3],
    alpha: float,
    beta: float,
):
    node = wp.tid()
    result = wp.vec3(0.0)
    for axis in range(3):
        for column in range(matrix.shape[1]):
            result[axis] += matrix[3 * node + axis, column] * x[column // 3][column % 3]
    if beta != 0.0:
        result = alpha * result + beta * y[node]
    else:
        result = alpha * result
    out[node] = result


class Tests(unittest.TestCase):
    def setUp(self):
        with np.load(ROOT / "tests/data/woodbury_contact.npz") as source:
            self.f = {key: source[key].copy() for key in source.files}
        full, b, free = interpolation(self.f["fixed"], self.f["nodes"], self.f["bary"])
        g = rigid_interpolation(self.f["levers"])
        masked = full.copy()
        masked[:, np.flatnonzero(np.repeat(self.f["fixed"], 3))] = 0
        j = np.column_stack((-masked, g))
        self.matrix = self.f["mass"] + float(np.float32(self.f["h"])) * j.T @ block_diagonal(self.f["matrices"]) @ j
        self.matrix_device = wp.array(self.matrix, dtype=float, device=DEVICE)
        self.free = free
        self.pre = SimpleNamespace(
            device=wp.get_device(DEVICE),
            n=len(self.f["fixed"]) + 2,
            count=len(self.f["nodes"]),
            _last_dt=float(self.f["h"]),
        )
        p = self.pre
        p.rhs = wp.array(self.f["r"].astype(np.float32).reshape(-1, 3), dtype=wp.vec3, device=DEVICE)
        p.z = wp.zeros_like(p.rhs)
        mass = self.f["mass"][-6:, -6:]
        blocks = [mass[:3, :3], mass[3:, 3:], mass[:3, 3:], mass[3:, :3]]
        p.rigid_blocks = wp.array(blocks, dtype=wp.mat33, device=DEVICE)
        p.matrices = wp.array(self.f["matrices"], dtype=wp.mat33, device=DEVICE)
        p.levers = wp.array(self.f["levers"], dtype=wp.vec3, device=DEVICE)
        # The recorded track follows no friction tunnel: no anchor has a shear and the system is symmetric.
        p.shears = wp.zeros(len(self.f["nodes"]), dtype=wp.mat33, device=DEVICE)
        p.nonsymmetric = False
        # The kernels read the number of track samples from the live counts of the interaction.
        live = np.zeros(3, dtype=np.int32)
        live[LIVE_TRACK] = len(self.f["nodes"])
        interaction = SimpleNamespace(
            _track_nodes=wp.array(self.f["nodes"], dtype=wp.vec4i, device=DEVICE),
            _track_bary=wp.array(self.f["bary"], dtype=wp.vec4, device=DEVICE),
            _live=wp.array(live, dtype=int, device=DEVICE),
        )
        p.coupler = SimpleNamespace(interaction=interaction)
        p.operator = SimpleNamespace(matvec=self.matvec)
        support = tuple(int(i) for i in np.flatnonzero(np.any(b != 0, axis=0)))
        a = self.f["mass"][np.ix_(free, free)]
        z = np.linalg.solve(a, np.eye(len(a))[:, support]).astype(np.float32)
        flat = np.full(len(a) * 96, np.nan, np.float32)
        flat[: z.size] = z.ravel(order="F")
        core = SimpleNamespace(
            basis=wp.array(flat, dtype=float, device=DEVICE),
            free=wp.array(np.flatnonzero(self.f["fixed"] == 0), dtype=int, device=DEVICE),
            support=wp.array(support, dtype=int, device=DEVICE),
            # The number of support columns as device data, as ``load_basis`` writes it.
            count=wp.array([len(support)], dtype=int, device=DEVICE),
        )
        self.manager = SimpleNamespace(
            active=True,
            core=core,
            support=support,
            basis_builds=1,
            structural_key=("frozen-actual",),
        )
        self.manager._structure = lambda: self.manager.structural_key
        self.helper = WoodburyInitialGuess(p)

    def matvec(self, x, y, z, alpha, beta):
        wp.launch(
            dense_matvec,
            self.pre.n,
            [self.matrix_device, x, y, z, alpha, beta],
            device=DEVICE,
        )

    def test_actual_packet_kernel_dense_guess_and_current_residual_certificate(self):
        original = self.pre.rhs.numpy().copy()
        self.helper.prepare(self.manager)
        self.assertEqual(self.helper.ready, 1)
        self.helper.initialize()
        self.assertEqual(self.helper.selected.numpy()[0], 1)
        answer = self.pre.z.numpy().reshape(-1).astype(float)
        rhs = self.pre.rhs.numpy().reshape(-1).astype(float)
        self.assertLess(np.linalg.norm(self.matrix @ answer - rhs) / np.linalg.norm(rhs), 2e-6)
        # The fixed rows of the original system remain identities and the
        # existing correction continues to scatter their full reactions.
        fixed_dofs = np.flatnonzero(np.repeat(self.f["fixed"], 3))
        np.testing.assert_array_equal(answer[fixed_dofs], rhs[fixed_dofs])
        np.testing.assert_array_equal(self.pre.rhs.numpy(), original)

    def test_cached_mapping_preserves_current_rhs_and_geometry(self):
        self.helper.prepare(self.manager)
        plan = self.helper.host_plan
        interpolation = plan.p
        packet = self.helper.packet.numpy()
        arguments = (
            self.pre.count,
            self.helper.support,
            self.helper.free_host,
            self.helper.compliance,
            float(np.float32(self.pre._last_dt)),
        )
        for scale in (1.0, -0.4, 2.0):
            changed = packet.copy()
            changed[38:_HEADER] *= scale
            changed[_HEADER + 17 : _HEADER + 20] += 0.03 * scale
            cached = host_coefficients(changed, *arguments, plan=plan)
            fresh = host_coefficients(changed, *arguments)
            np.testing.assert_array_equal(cached, fresh)
            self.assertIs(plan.p, interpolation)
        changed = packet.copy()
        changed[_HEADER + 4 : _HEADER + 8] *= 0.9
        cached = host_coefficients(changed, *arguments, plan=plan)
        fresh = host_coefficients(changed, *arguments)
        np.testing.assert_array_equal(cached, fresh)
        self.assertIsNot(plan.p, interpolation)
        self.helper.prepare(self.manager)
        self.assertIs(self.helper.host_plan, plan)
        self.manager.basis_builds += 1
        self.helper.prepare(self.manager)
        self.assertIsNot(self.helper.host_plan, plan)

    def test_inactive_manager_and_nonfinite_coefficients_fall_back(self):
        self.manager.active = False
        self.helper.prepare(self.manager)
        self.helper.initialize()
        np.testing.assert_array_equal(self.pre.z.numpy(), 0)
        self.manager.active = True
        self.helper.prepare(self.manager)
        self.helper.coefficients.fill_(np.nan)
        self.helper.initialize()
        np.testing.assert_array_equal(self.pre.z.numpy(), 0)

    def test_surface_approximation_uses_complete_current_operator(self):
        self.pre.surface = SimpleNamespace(
            anchors=SimpleNamespace(count=wp.array([1], dtype=int, device=DEVICE)),
            valid=wp.array([1], dtype=int, device=DEVICE),
        )
        surface_row = np.zeros(len(self.matrix))
        surface_row[self.free[0]], surface_row[-6] = -1, 1
        surface_operator = np.outer(surface_row, surface_row)
        rhs = self.pre.rhs.numpy().reshape(-1).astype(float)
        self.pre.linear_solver = SimpleNamespace(
            _cur_iter_and_condition=wp.array([4, 0], dtype=int, device=DEVICE),
            _maxiter=80,
        )
        for stiffness, expected in ((0.001, 1), (1e6, 0)):
            current = self.matrix + stiffness * surface_operator
            self.matrix_device.assign(current)
            self.helper.prepare(self.manager)
            self.assertEqual(self.helper.enabled.numpy()[0], 1)
            self.helper.initialize()
            self.assertEqual(self.helper.selected.numpy()[0], expected)
            candidate = self.pre.z.numpy().reshape(-1).astype(float)
            if expected:
                self.assertLess(np.linalg.norm(rhs - current @ candidate), np.linalg.norm(rhs))
            else:
                np.testing.assert_array_equal(candidate, 0)
            self.helper.after_solve()
        counters = self.helper.diagnostics()["counters"]
        self.assertEqual(counters["surface_applications"], 2)
        self.assertEqual(counters["surface_selected"], 1)
        self.assertEqual(counters["cg_iterations"], 8)
        self.assertGreater(self.helper.timings["host_dense_seconds"], 0)
        self.assertGreaterEqual(
            self.helper.timings["prepare_wall_seconds"],
            self.helper.timings["host_dense_seconds"],
        )

    def test_stale_operator_rejection_and_binding_refresh_keep_owners(self):
        self.helper.prepare(self.manager)
        old_owner = self.helper.inverse_free
        pointer = old_owner.ptr
        mapping = old_owner.numpy().copy()
        old_packet = self.helper.packet
        changed = self.matrix.copy()
        changed[np.ix_(self.free, self.free)] *= 10
        self.matrix_device.assign(changed)
        self.helper.initialize()
        self.assertEqual(self.helper.selected.numpy()[0], 0)
        np.testing.assert_array_equal(self.pre.z.numpy(), 0)
        # Numerical basis refresh can occur in place; build counter changes
        # force a new small compliance snapshot. The free mapping is bound to
        # the structure and keeps its storage, which graphs of the guess address.
        compliance, structure = self.helper.compliance, self.helper.structure
        self.manager.basis_builds += 1
        self.helper.prepare(self.manager)
        self.assertIsNot(self.helper.compliance, compliance)
        self.assertEqual(self.helper.structure, structure)
        self.assertIs(self.helper.inverse_free, old_owner)
        self.assertIs(self.helper.packet, old_packet)
        self.assertIn(self.manager.core.basis, self.helper.owners)
        self.assertIn(self.helper.inverse_free, self.helper.owners)
        self.assertIn(self.helper.available, self.helper.owners)
        # New storage of the basis is a new binding of the structure: the map is
        # written again, in the same storage.
        self.helper.inverse_free.fill_(-7)
        self.manager.core.basis = wp.clone(self.manager.core.basis)
        self.helper.prepare(self.manager)
        self.assertNotEqual(self.helper.structure, structure)
        self.assertIs(self.helper.inverse_free, old_owner)
        self.assertEqual(self.helper.inverse_free.ptr, pointer)
        np.testing.assert_array_equal(self.helper.inverse_free.numpy(), mapping)
        # Other free nodes of the same model: rewritten in place as well.
        free = self.manager.core.free.numpy()
        self.manager.core.free = wp.array(free[::-1].copy(), dtype=int, device=DEVICE)
        self.helper.prepare(self.manager)
        self.assertEqual(self.helper.inverse_free.ptr, pointer)
        np.testing.assert_array_equal(self.helper.inverse_free.numpy()[free], np.arange(len(free))[::-1])
        self.manager.core.free = wp.array(free, dtype=int, device=DEVICE)
        self.helper.prepare(self.manager)
        np.testing.assert_array_equal(self.helper.inverse_free.numpy(), mapping)
        self.manager.support = self.manager.support[:-3]
        self.manager.core.count.fill_(len(self.manager.support))
        self.helper.prepare(self.manager)
        self.assertIn("exceeds basis support", self.helper.last_reason)
        self.helper.initialize()
        np.testing.assert_array_equal(self.pre.z.numpy(), 0)

    def test_first_residual_is_the_certificate_product_once_per_solve(self):
        p, helper = self.pre, self.helper
        direct = wp.zeros_like(p.rhs)
        served = wp.zeros_like(p.rhs)
        other = wp.zeros_like(p.rhs)

        def serve(expected_selected):
            served.fill_(wp.vec3(7.0))
            self.assertEqual(helper.selected.numpy()[0], expected_selected)
            # Not the solution vector or not a residual: the operator must apply.
            self.assertFalse(helper.first_residual(other, p.rhs, served, -1.0, 1.0))
            self.assertFalse(helper.first_residual(p.z, p.rhs, served, 1.0, 0.0))
            np.testing.assert_array_equal(served.numpy(), 7)
            self.assertTrue(helper.first_residual(p.z, p.rhs, served, -1.0, 1.0))
            self.matvec(p.z, p.rhs, direct, -1.0, 1.0)
            np.testing.assert_array_equal(served.numpy(), direct.numpy())
            # The product belongs to one certified start only.
            self.assertFalse(helper.first_residual(p.z, p.rhs, served, -1.0, 1.0))

        self.assertFalse(helper.first_residual(p.z, p.rhs, served, -1.0, 1.0))
        helper.prepare(self.manager)
        helper.initialize()
        serve(1)
        self.assertTrue(np.any(served.numpy() != p.rhs.numpy()))
        # A rejected guess is the zero start: its residual is the right-hand side.
        changed = self.matrix.copy()
        changed[np.ix_(self.free, self.free)] *= 10
        self.matrix_device.assign(changed)
        helper.initialize()
        serve(0)
        np.testing.assert_array_equal(served.numpy(), p.rhs.numpy())
        # A finished solve retires a product nobody asked for.
        p.linear_solver = SimpleNamespace(
            _cur_iter_and_condition=wp.array([0, 0], dtype=int, device=DEVICE), _maxiter=80
        )
        helper.initialize()
        helper.after_solve()
        self.assertFalse(helper.first_residual(p.z, p.rhs, served, -1.0, 1.0))

    def test_solver_operator_asks_the_guess_before_applying(self):
        # No operator storage exists on this bare object: a full application would raise.
        pre = object.__new__(_ShaftPreconditioner)
        calls = []
        pre._woodbury_guess = SimpleNamespace(first_residual=lambda *operands: calls.append(operands) or True)
        pre._solver_matvec("x", "y", "z", -1.0, 1.0)
        self.assertEqual(calls, [("x", "y", "z", -1.0, 1.0)])
        for guess in (SimpleNamespace(first_residual=lambda *operands: False), None):
            pre._woodbury_guess = guess
            with self.assertRaises(AttributeError):
                pre._solver_matvec("x", "y", "z", -1.0, 1.0)

    def test_nonzero_rhs_outside_track_support_remains_in_current_residual(self):
        # Add one free tissue node that has RHS but no contact interpolation.
        old_rhs = self.pre.rhs.numpy()
        old_matrix = self.matrix.copy()
        mapping = np.r_[np.arange(24), np.arange(27, 33)]
        self.matrix = np.eye(33) * 0.3
        self.matrix[np.ix_(mapping, mapping)] = old_matrix
        self.matrix_device = wp.array(self.matrix, dtype=float, device=DEVICE)
        self.pre.n = 11
        self.pre.rhs = wp.array(
            np.vstack((old_rhs[:-2], [0.001, -0.002, 0.003], old_rhs[-2:])),
            dtype=wp.vec3,
            device=DEVICE,
        )
        self.pre.z = wp.zeros_like(self.pre.rhs)
        core = self.manager.core
        old_basis = core.basis.numpy().reshape((-1, 96), order="F")
        basis = np.vstack((old_basis, np.zeros((3, 96), np.float32)))
        core.basis = wp.array(basis.ravel(order="F"), dtype=float, device=DEVICE)
        core.free = wp.array(np.r_[core.free.numpy(), 8], dtype=int, device=DEVICE)
        self.manager.basis_builds += 1
        self.helper = WoodburyInitialGuess(self.pre)
        self.helper.prepare(self.manager)
        self.helper.initialize()
        candidate = self.pre.z.numpy().reshape(-1)
        rhs = self.pre.rhs.numpy().reshape(-1)
        np.testing.assert_array_equal(candidate[24:27], 0)
        residual = rhs.astype(float) - self.matrix @ candidate
        np.testing.assert_allclose(residual[24:27], rhs[24:27], rtol=0, atol=0)
        self.assertLess(np.linalg.norm(residual), np.linalg.norm(rhs.astype(float)))

    @unittest.skipUnless(DEVICE.startswith("cuda"), "CUDA graph replay contract")
    def test_captured_guess_reads_new_rhs_and_keeps_owned_storage(self):
        self.helper.prepare(self.manager)
        self.helper.initialize()
        owners = self.helper.owners
        with ScopedCapture(device=self.pre.device) as capture:
            self.helper.initialize()
        wp.capture_launch(capture.graph)
        original = self.pre.z.numpy()
        self.pre.rhs.assign(-0.75 * self.pre.rhs.numpy())
        self.helper.prepare(self.manager)
        self.assertEqual([a.ptr for a in owners], [a.ptr for a in self.helper.owners])
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(self.pre.z.numpy(), -0.75 * original, rtol=5e-6, atol=2e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
