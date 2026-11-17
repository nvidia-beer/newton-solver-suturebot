"""Composition preserves FEM loads and resets both independent numerical caches."""

import os
import unittest

import numpy as np
import test_fem_prepared_step as prepared_fixture
import warp as wp

from newton_solver_suturebot._src.solvers.fem.affine_guess import AffineGuessManager, DeviceBasisGuess
from newton_solver_suturebot._src.solvers.fem.held_start import HeldFEMStart
from newton_solver_suturebot._src.solvers.fem.warp_basis_provider import WarpBasisProvider
from newton_solver_suturebot._src.utils.capture import ScopedCapture


class CombinedHelpersTests(unittest.TestCase):
    device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")

    def fixture(self):
        source = prepared_fixture.TestPreparedFEMStep()
        source.device = self.device
        model, state, make = source.fixture(mixed=True)
        actual, reference = make(), make()
        for solver in (actual, reference):
            solver.refresh_tangent(state.particle_q)
        control = model.control()
        held = HeldFEMStart(actual)
        actual._held_start = held
        provider = WarpBasisProvider()
        manager = AffineGuessManager(actual, provider, capacity=96)
        actual._affine_predictor = manager
        return (
            source.dt,
            model,
            state,
            control,
            actual,
            reference,
            held,
            manager,
            provider,
        )

    def test_held_cache_and_inactive_predictor_preserve_force_and_endpoint(self):
        dt, model, state, control, actual, reference, held, manager, provider = self.fixture()
        incoming = model.state()
        incoming.assign(state)
        out, expected = model.state(), model.state()
        try:
            self.assertTrue(held.prepare(state, control, None, dt))
            for scale in (0.0, 0.2, -0.3):
                load = wp.array(
                    scale * np.linspace(-0.01, 0.02, 24).reshape(8, 3),
                    dtype=wp.vec3,
                    device=self.device,
                )
                actual.velocity_increment.zero_()
                reference.velocity_increment.zero_()
                actual.step(state, out, control, None, dt, external_forces=load)
                reference.step(incoming, expected, control, None, dt, external_forces=load)
                np.testing.assert_array_equal(state.particle_f.numpy(), incoming.particle_f.numpy())
                np.testing.assert_array_equal(out.particle_q.numpy(), expected.particle_q.numpy())
                np.testing.assert_array_equal(out.particle_qd.numpy(), expected.particle_qd.numpy())
            actual.reset(state)
            self.assertFalse(held.active)
            self.assertFalse(manager.active)
            self.assertIsNone(held.capture_key())
            self.assertIsNone(manager.capture_key())
        finally:
            provider.close()

    def test_captured_held_assembly_and_retained_guess_keep_original_compact_cg(self):
        if not wp.get_device(self.device).is_cuda:
            self.skipTest("Combined compact CG capture requires CUDA")
        dt, model, state, control, actual, reference, held, manager, provider = self.fixture()
        incoming = model.state()
        incoming.assign(state)
        out, expected = model.state(), model.state()
        load = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        # The shared prepared-step fixture defaults to BiCGStab. This test
        # explicitly exercises the production compact-CG path on both sides.
        actual.solver_type = reference.solver_type = "cg"
        try:
            # One ordinary solve prepares the genuine free-node operator. The
            # small fixture uses the actual Warp-only approximate provider;
            # runtime native/articulation eligibility is unchanged.
            actual.step(state, out, control, None, dt, external_forces=load)
            fixed = actual._fixed_particles
            self.assertTrue(actual._lin_solver.compact_solve)
            support = tuple(range(3 * fixed.free.size))
            columns, _ = provider(fixed.compact_matrix, support)
            manager.core = DeviceBasisGuess(fixed.compact_operator, fixed.free, capacity=96)
            manager.core.load_basis(columns, support)
            manager.support, manager.structural_key = support, manager._structure()
            manager.active = True
            held.prepare(state, control, None, dt)
            actual.step(state, out, control, None, dt, external_forces=load)
            with ScopedCapture(device=self.device) as captured:
                actual.step(state, out, control, None, dt, external_forces=load)
            for scale in (0.1, -0.2, 0.05):
                load.fill_(wp.vec3(scale * 0.01, -scale * 0.02, scale * 0.03))
                wp.capture_launch(captured.graph)
                reference.step(incoming, expected, control, None, dt, external_forces=load)
                np.testing.assert_array_equal(state.particle_f.numpy(), incoming.particle_f.numpy())
                np.testing.assert_allclose(
                    out.particle_q.numpy(),
                    expected.particle_q.numpy(),
                    rtol=2e-6,
                    atol=2e-8,
                )
                np.testing.assert_allclose(
                    out.particle_qd.numpy(),
                    expected.particle_qd.numpy(),
                    rtol=3e-5,
                    atol=2e-7,
                )
                np.testing.assert_array_equal(out.particle_q.numpy()[:2], model.particle_q.numpy()[:2])
                self.assertEqual(actual._lin_solver._maxiter, reference._lin_solver._maxiter)
                self.assertEqual(actual._lin_solver._tol, reference._lin_solver._tol)
            self.assertGreater(manager.diagnostics()["counters"]["selected"], 0)
            self.assertGreater(held.diagnostics()["cached_trial_assemblies"], 0)
            actual.reset(state)
            self.assertFalse(held.active)
            self.assertFalse(manager.active)
            self.assertEqual(int(manager.core.history.numpy()[0]), 0)
            self.assertEqual(provider.reference_builds, 1)
        finally:
            provider.close()


if __name__ == "__main__":
    unittest.main()
