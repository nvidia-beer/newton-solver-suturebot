# SPDX-License-Identifier: Apache-2.0
"""Current-matrix certification participates in real FEM/coupling transactions."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_fixed_particles as fixed_fixture
import warp as wp
from test_combined_contact_preconditioner import native_embedded
from test_fixed_particles import dense
from test_needle_insertion_example import _example
from test_needle_tissue_coupler import _native_state

from newton_solver_suturebot._src.solvers.coupled.coupling import NeedleCouplingNotConverged
from newton_solver_suturebot._src.solvers.fem.certified_cg import CertifiedCorrection
from newton_solver_suturebot._src.solvers.fem.certified_scope import (
    CertifiedFEMScope,
    require_certificate,
    write_certificate,
)
from newton_solver_suturebot._src.utils.capture import ScopedCapture


def configure(solver):
    solver.solver_type = "cg"
    solver.linear_solver_maxiter = 160
    solver.linear_solver_check_every = 0
    solver.linear_solver_use_cuda_graph = True
    solver.linear_solver_compact = True
    scope = CertifiedFEMScope(solver)
    solver._certified_linear = scope
    return scope


class CertifiedScopeTests(unittest.TestCase):
    def test_forward_contract_rejects_gradient_custom_and_unsupported_workspaces(self):
        with _example() as ex:
            _, _, c = native_embedded(ex)
            scope = configure(ex.solver)
            self.assertTrue(scope.forward_supported(c, ex.state_0, ex.control))
            for owner, name in (
                (ex.state_0, "particle_q"),
                (ex.state_0, "body_f"),
                (ex.state_0, "joint_qd"),
                (c._candidate, "particle_qd"),
                (c._candidate, "joint_q"),
                (ex.control, "joint_f"),
                (ex.control, "tet_activations"),
                (ex.solver.A_bsr, "values"),
            ):
                with (
                    self.subTest(field=name),
                    patch.object(owner, name, wp.clone(getattr(owner, name), requires_grad=True)),
                ):
                    self.assertFalse(scope.forward_supported(c, ex.state_0, ex.control))
            for name, value in (
                ("linear_solver_maxiter", 80),
                ("linear_solver_check_every", 1),
                ("linear_solver_compact", False),
                ("_fixed_particles", None),
            ):
                with self.subTest(setting=name), patch.object(ex.solver, name, value):
                    self.assertFalse(scope.forward_supported(c, ex.state_0, ex.control))
            with patch.object(c, "_evaluate", wraps=c._evaluate):
                self.assertFalse(scope.forward_supported(c, ex.state_0, ex.control))
            with wp.Tape():
                self.assertFalse(scope.forward_supported(c, ex.state_0, ex.control))
            with patch.object(
                ex.control, "mujoco", SimpleNamespace(ctrl=SimpleNamespace(requires_grad=True)), create=True
            ):
                self.assertFalse(scope.forward_supported(c, ex.state_0, ex.control))

    def test_packet_preserves_existing_statistics_and_rejects_nonfinite_certificate(self):
        with _example() as ex:
            scope = configure(ex.solver)
            for count in (6, 8):
                values = np.linspace(-3, 7, count)
                packet = wp.array(values, dtype=wp.float64, device=ex.device)
                scope.status.zero_()
                result = scope.pack(packet).numpy()
                np.testing.assert_array_equal(result[:-1], values)
                self.assertEqual(result[-1], 0)
                scope.status.fill_(1)
                scope.check(scope.pack(packet).numpy()[-1])
            for value in (0, np.nan, np.inf, -1):
                with self.assertRaises(NeedleCouplingNotConverged):
                    scope.check(value)
            self.assertEqual(scope.rejections, 4)

    def test_injected_certificate_rejection_rolls_back_real_native_and_tissue_transaction(self):
        with _example() as ex:
            h, native, c = native_embedded(ex)
            scope = configure(ex.solver)
            before_native = _native_state(native)
            before_input = {
                name: getattr(ex.state_0, name).numpy().copy()
                for name in ("particle_q", "particle_qd", "particle_f", "body_q", "body_qd")
            }
            before_output = ex.state_1.particle_q.numpy().copy()
            before_load = c._load.numpy().copy()
            tolerance = ex.solver.linear_solver_tolerance

            # Inject the same certificate failure at the shared statistics
            # check. CPU remains on its original solver; CUDA exercises the
            # certified solve and whichever real force path is eligible.
            original_pack = scope.pack

            def fail_packet(packet):
                scope.status.zero_()
                return original_pack(packet)

            original_check = scope.check

            def fail_check(_value):
                original_check(0)

            original_begin = scope.begin_attempt

            def enable_check(*args):
                original_begin(*args)
                if not ex.device.is_cuda:
                    # Enable the check only after the normal CPU integration.
                    scope.active = False

            with patch.object(scope, "begin_attempt", side_effect=enable_check):
                if ex.device.is_cuda:
                    with patch.object(scope, "check", side_effect=fail_check):
                        with self.assertRaises(NeedleCouplingNotConverged):
                            c.step(ex.state_0, ex.state_1, ex.control, h)
                else:
                    original_evaluate = c._evaluate

                    def evaluate(*args, **kwargs):
                        result = original_evaluate(*args, **kwargs)
                        scope.active = True
                        return result

                    with (
                        patch.object(c, "_evaluate", side_effect=evaluate),
                        patch.object(scope, "pack", side_effect=fail_packet),
                    ):
                        with self.assertRaises(NeedleCouplingNotConverged):
                            c.step(ex.state_0, ex.state_1, ex.control, h)
            self.assertFalse(scope.active)
            self.assertEqual(scope.rejections, 1)
            self.assertEqual(ex.solver.linear_solver_tolerance, tolerance)
            np.testing.assert_array_equal(_native_state(native), before_native)
            for name, values in before_input.items():
                np.testing.assert_array_equal(getattr(ex.state_0, name).numpy(), values)
            np.testing.assert_array_equal(ex.state_1.particle_q.numpy(), before_output)
            np.testing.assert_array_equal(c._load.numpy(), before_load)

    def test_compact_fem_workspace_certificate_tracks_current_A_and_fresh_rhs(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("Actual compact conditional CUDA solve")
        fixture = fixed_fixture.TestFixedParticles()
        fixture.device = str(device)
        model = fixture.model()
        solver = fixture.solver(model)
        solver.linear_solver_chebyshev_degree = 4
        scope = configure(solver)
        scope.active = True
        solver.linear_solver_tolerance = 1e-6
        current, following = model.state(), model.state()
        forces = wp.array([[0, 0, 0], [0, 0, 0], [0.01, 0.02, 0.03], [-0.02, 0, 0.01]], dtype=wp.vec3, device=device)
        solver.step(current, following, None, None, fixture.dt, external_forces=forces)
        self.assertIsInstance(solver._lin_solver, CertifiedCorrection)
        with ScopedCapture(device=device) as capture:
            solver.step(current, following, None, None, fixture.dt, external_forces=forces)
        for multiplier in (1.0, -0.3):
            forces.fill_(wp.vec3(multiplier * 0.01, 0, 0))
            wp.capture_launch(capture.graph)
            report = solver._lin_solver.diagnostics()
            self.assertTrue(report["certified"])
            self.assertLessEqual(report["primary_A_actions"], 160)
            fixed = solver._fixed_particles
            raw = dense(fixed.compact_matrix).astype(float)
            b = fixed.compact_rhs.numpy().astype(float).ravel()
            x = fixed.compact_increment.numpy().astype(float).ravel()
            self.assertLessEqual(np.linalg.norm(b - raw @ x), 1e-6 * np.linalg.norm(b))
            np.testing.assert_array_equal(following.particle_q.numpy()[:2], model.particle_q.numpy()[:2])
            np.testing.assert_array_equal(following.particle_qd.numpy()[:2], 0)
        old = solver._lin_solver
        scope.end_attempt()
        solver.step(current, following, None, None, fixture.dt, external_forces=forces)
        self.assertNotIsInstance(solver._lin_solver, CertifiedCorrection)
        self.assertIs(old.status, scope.status, "Retained graphs retain their certificate owner")

    def test_real_captured_force_packet_and_eager_oracle_both_certify(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("Actual captured native coupling")
        from contextlib import ExitStack

        with ExitStack() as stack:
            examples = [stack.enter_context(_example()) for _ in range(2)]
            scopes, couplers = [], []
            for index, ex in enumerate(examples):
                h, _, c = native_embedded(ex)
                scopes.append(configure(ex.solver))
                c.capture_fem = True
                c.capture_coupling = bool(index)
                couplers.append(c)
            for c, ex in zip(couplers, examples, strict=True):
                c.step(ex.state_0, ex.state_1, ex.control, h)
                self.assertIsInstance(ex.solver._lin_solver, CertifiedCorrection)
                self.assertTrue(ex.solver._lin_solver.diagnostics()["certified"])
            for field in ("particle_q", "particle_qd", "body_q", "body_qd"):
                np.testing.assert_allclose(
                    getattr(examples[1].state_1, field).numpy(),
                    getattr(examples[0].state_1, field).numpy(),
                    rtol=4e-6,
                    atol=1e-8,
                )
            graph = couplers[1]._host_trial_forces
            self.assertIsNotNone(graph)
            self.assertGreater(graph.launches, 0)
            self.assertIs(graph.certificate, scopes[1])
            self.assertEqual(graph.result.shape, (11,))
            self.assertEqual(graph.result.numpy()[-1], 1)
            self.assertFalse(scopes[1].active)
            # A live, failed certificate blocks force evaluation before any
            # surface/tunnel metadata can be committed by the outer coupler.
            scopes[1].active = True
            scopes[1].status.zero_()
            graph.prepare(h)
            with self.assertRaises(NeedleCouplingNotConverged):
                graph.evaluate(h, False)
            scopes[1].end_attempt()

    def test_capture_packet_gates_forces_and_reads_live_certificate(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not device.is_cuda:
            self.skipTest("Captured packet")
        status = wp.ones(1, dtype=int, device=device)
        finite = wp.ones(1, dtype=int, device=device)
        packet = wp.zeros(11, dtype=wp.float64, device=device)
        wp.launch(require_certificate, 1, [status, finite], device=device)
        wp.launch(write_certificate, 1, [status, packet], device=device)
        with ScopedCapture(device=device) as capture:
            finite.fill_(1)
            wp.launch(require_certificate, 1, [status, finite], device=device)
            wp.launch(write_certificate, 1, [status, packet], device=device)
        for valid in (0, 1, 0):
            status.fill_(valid)
            wp.capture_launch(capture.graph)
            self.assertEqual(finite.numpy()[0], valid)
            self.assertEqual(packet.numpy()[-1], valid)


if __name__ == "__main__":
    unittest.main()
