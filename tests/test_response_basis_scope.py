# SPDX-License-Identifier: Apache-2.0
"""Response guesses belong to one supported coupling attempt, never physical state."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import test_fem_prepared_step as prepared_fixture
import warp as wp
from test_combined_contact_preconditioner import native_embedded
from test_grasp_scene import _scene
from test_needle_insertion_example import _example
from test_needle_tissue_coupler import _native_state

from newton_solver_suturebot._src.solvers.fem.affine_guess import COUNTERS, AffineGuessManager, DeviceBasisGuess
from newton_solver_suturebot._src.solvers.fem.warp_basis_provider import WarpBasisProvider
from newton_solver_suturebot._src.utils.capture import ScopedCapture


class ResponseBasisScopeTests(unittest.TestCase):
    def test_scene_option_and_close_release_owned_provider_without_device_reads(self):
        with _scene() as (scene, _):
            scene.close()
            provider = Mock()
            manager = Mock()
            scene._response_basis_provider = provider
            scene.solver._affine_predictor = manager
            scene.response_basis_guesses = False
            scene._configure_response_basis_guesses()
            manager.invalidate.assert_called_once_with()
            provider.close.assert_called_once_with()
            self.assertIsNone(scene.solver._affine_predictor)
            self.assertIsNone(scene._response_basis_provider)
            scene.close()
            provider.close.assert_called_once_with()

    def test_end_attempt_drops_history_and_disables_direct_fem_hooks(self):
        source = prepared_fixture.TestPreparedFEMStep()
        model, state, make = source.fixture(mixed=True)
        actual, reference = make(), make()
        control = model.control()
        incoming = model.state()
        incoming.assign(state)
        actual.refresh_tangent(state.particle_q)
        reference.refresh_tangent(state.particle_q)
        provider = WarpBasisProvider()
        manager = AffineGuessManager(actual, provider)
        core = Mock()
        core.owners = ()
        manager.core, manager.active = core, True
        core.set_enabled.assert_called_once_with(True)
        actual._affine_predictor = manager
        manager.reset()
        self.assertTrue(manager.active, "In-attempt retries may continue using the basis")
        manager.end_attempt()
        self.assertFalse(manager.active)
        # The same decision on the device, for the solves that a graph holds.
        core.set_enabled.assert_called_with(False)
        self.assertIs(manager.core, core, "The owned basis survives an attempt boundary")
        if manager._gated:
            # With conditional graphs the key is the storage of the core, applied or not: a graph
            # holds the launches of the basis behind the device flag, so the key does not follow it.
            self.assertEqual(manager.capture_key(), (id(manager), ()))
        else:
            self.assertIsNone(manager.capture_key())
        out, expected = model.state(), model.state()
        actual.step(state, out, control, None, source.dt)
        reference.step(incoming, expected, control, None, source.dt)
        core.before.assert_not_called()
        core.after.assert_not_called()
        np.testing.assert_array_equal(out.particle_q.numpy(), expected.particle_q.numpy())
        np.testing.assert_array_equal(out.particle_qd.numpy(), expected.particle_qd.numpy())
        self.assertEqual(core.reset.call_count, 2)
        provider.close()

    def test_storage_made_ahead_is_kept_by_the_first_track_support(self):
        with _scene() as (scene, _):
            scene.close()
            provider = Mock()
            manager = AffineGuessManager(scene.solver, provider, capacity=9)
            device = scene.state_0.particle_q.device
            free = wp.array([0, 1], dtype=int, device=device)
            fixed = SimpleNamespace(
                free=free,
                free_host=np.array([0, 1], dtype=np.int32),
                compact_matrix=object(),
                compact_operator=object(),
            )
            interaction = SimpleNamespace(track_count=0, _track_nodes_host=[])
            structure = ["first structure"]
            with (
                patch.object(scene.model, "device", SimpleNamespace(is_cuda=True, is_capturing=False)),
                patch.object(scene.solver, "_fixed_particles", fixed),
                patch.object(scene.coupler, "interaction", interaction),
                patch.object(scene.coupler, "_refresh_projection") as projection,
                patch.object(scene.coupler, "invalidate_fem_graph") as invalidate,
                patch.object(manager, "_structure", side_effect=lambda: tuple(structure)),
            ):
                # Start-up: the storage and the engine exist, no basis is built, nothing is applied.
                self.assertTrue(manager.prepare(scene.coupler, scene.state_0, scene.control))
                core = manager.core
                self.assertIsInstance(core, DeviceBasisGuess)
                self.assertEqual(core.capacity, 9)
                projection.assert_called_once_with()
                provider.prepare.assert_called_once_with(fixed.compact_matrix)
                provider.assert_not_called()
                self.assertEqual(
                    (manager.structural_key, manager.support, manager.active, manager.basis_builds),
                    (("first structure",), (), False, 0),
                )
                np.testing.assert_array_equal(core.enabled.numpy(), [0])
                self.assertEqual(manager.diagnostics()["counters"], dict.fromkeys(COUNTERS, 0))
                pointers = [array.ptr for array in core.owners]
                # A second start-up attempt of the same structure makes nothing.
                self.assertTrue(manager.prepare(scene.coupler, scene.state_0, scene.control))
                self.assertIs(manager.core, core)
                self.assertEqual(provider.prepare.call_count, 1)
                manager.begin_attempt(scene.coupler, scene.state_0, scene.control)
                self.assertEqual((manager.status, manager.active), ("waiting for committed track support", False))
                manager.end_attempt()
                # The first track support loads its columns into that storage. It replaces no array,
                # drops no graph and reads nothing from the device.
                interaction.track_count, interaction._track_nodes_host = 1, [[0, 0, 0, 0]]
                provider.return_value = (wp.full(18, 0.5, dtype=float, device=device), {})
                provider.invalidate.reset_mock()
                with patch.object(wp.array, "numpy", side_effect=AssertionError("device read")):
                    manager.begin_attempt(scene.coupler, scene.state_0, scene.control)
                provider.assert_called_once_with(fixed.compact_matrix, (0, 1, 2))
                self.assertTrue(manager.active)
                self.assertIs(manager.core, core)
                self.assertEqual([array.ptr for array in core.owners], pointers)
                self.assertEqual((manager.support, manager.basis_builds), ((0, 1, 2), 1))
                invalidate.assert_not_called()
                provider.invalidate.assert_not_called()
                np.testing.assert_array_equal(core.enabled.numpy(), [1])
                np.testing.assert_array_equal(core.count.numpy(), [3])
                manager.end_attempt()
                np.testing.assert_array_equal(core.enabled.numpy(), [0])
                # Another structure is other storage: the graphs of the old one are dropped.
                structure[0] = "second structure"
                self.assertTrue(manager.prepare(scene.coupler, scene.state_0, scene.control))
                self.assertIsNot(manager.core, core)
                invalidate.assert_called_once_with()
                provider.invalidate.assert_called_once_with()
                self.assertEqual(provider.prepare.call_count, 2)
                self.assertEqual((manager.structural_key, manager.support), (("second structure",), ()))
                # Without a compact operator, and for a solver this guess does not serve, nothing is made.
                other = AffineGuessManager(scene.solver, Mock())
                with patch.object(other, "_structure", return_value=None):
                    self.assertFalse(other.prepare(scene.coupler, scene.state_0, scene.control))
                with patch.object(scene.solver, "solver_type", "bicgstab"):
                    self.assertFalse(other.prepare(scene.coupler, scene.state_0, scene.control))
                self.assertIsNone(other.core)
                other.provider.prepare.assert_not_called()

    def test_captured_step_takes_the_basis_by_the_device_flag_of_its_core(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("The device flag is a conditional node of a CUDA graph")
        source = prepared_fixture.TestPreparedFEMStep()
        source.device = str(device)
        model, start, make = source.fixture(mixed=False)
        control, dt = model.control(), source.dt
        loads = [
            wp.array(scale * np.linspace(-0.01, 0.02, 24).reshape(8, 3), dtype=wp.vec3, device=device)
            for scale in (1.0, -0.5, 0.25)
        ]
        load = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)

        def prepared():
            solver = make()
            solver.solver_type = "cg"
            solver.linear_solver_check_every = 0
            solver.linear_solver_use_cuda_graph = True
            solver.linear_solver_compact = True
            solver.linear_solver_chebyshev_degree = 4
            solver.refresh_tangent(start.particle_q)
            state, out = model.state(), model.state()
            state.assign(start)
            # One step as written: the workspace and every scratch array exist before a capture.
            solver.step(state, out, control, None, dt, external_forces=load)
            self.assertTrue(solver._lin_solver.compact_solve)
            solver.velocity_increment.zero_()
            state.assign(start)
            return solver, state, out

        def with_core(solver, provider):
            fixed = solver._fixed_particles
            manager = AffineGuessManager(solver, provider, capacity=9)
            manager.core = DeviceBasisGuess(fixed.compact_operator, fixed.free, capacity=9)
            solver._affine_predictor = manager
            return manager

        def bits(state):
            return state.particle_q.numpy().tobytes(), state.particle_qd.numpy().tobytes()

        def counters(manager):
            return manager.core.counters.numpy().tolist()

        provider = WarpBasisProvider(capacity=9)
        try:
            flagged, flagged_in, flagged_out = prepared()
            written, written_in, written_out = prepared()
            plain, plain_in, plain_out = prepared()
            manager, reference = with_core(flagged, provider), with_core(written, provider)
            key = manager.capture_key()
            self.assertIsNotNone(key)
            np.testing.assert_array_equal(manager.core.enabled.numpy(), [0])
            # One graph of the step, captured while the basis is withheld.
            with ScopedCapture(device=device) as capture:
                flagged.step(flagged_in, flagged_out, control, None, dt, external_forces=load)
            # Flag 0: the step of a solver without a predictor, and nothing of the basis runs.
            for values in loads[:2]:
                wp.copy(load, values)
                wp.capture_launch(capture.graph)
                plain.step(plain_in, plain_out, control, None, dt, external_forces=load)
                self.assertEqual(bits(flagged_out), bits(plain_out))
                np.testing.assert_array_equal(
                    flagged.velocity_increment.numpy().view(np.uint32), plain.velocity_increment.numpy().view(np.uint32)
                )
            self.assertEqual(counters(manager), [0] * len(COUNTERS))
            np.testing.assert_array_equal(manager.core.history.numpy(), [0])
            self.assertEqual(manager.capture_key(), key)
            # A basis in both cores. The reference graph is captured as on a device without
            # conditional graphs: the launches of an active manager, with no flag in front.
            support = (0, 1, 2, 9, 10, 11)
            columns, _ = provider(flagged._fixed_particles.compact_matrix, support)
            for each in (manager, reference):
                each.core.load_basis(columns, support)
                each.active = True
            np.testing.assert_array_equal(manager.core.enabled.numpy(), [1])
            self.assertEqual(manager.capture_key(), key)
            with patch.object(AffineGuessManager, "_gated", new_callable=PropertyMock, return_value=False):
                with ScopedCapture(device=device) as unconditional:
                    written.step(written_in, written_out, control, None, dt, external_forces=load)
            # Both solvers continue from the same numerical guess of the increment.
            written.velocity_increment.assign(flagged.velocity_increment.numpy())
            # Flag 1: the same kernels in the same order on the same kind of arrays.
            for values in (*loads, loads[0]):
                wp.copy(load, values)
                wp.capture_launch(capture.graph)
                wp.capture_launch(unconditional.graph)
                self.assertEqual(bits(flagged_out), bits(written_out))
                self.assertEqual(counters(manager), counters(reference))
            applied = counters(manager)
            self.assertEqual(applied[0], 4)
            self.assertGreater(applied[0] - applied[2], 0, "a trial with history")
            self.assertEqual(manager.capture_key(), key)
            # Flag 0 again, as after the end of the attempt: the same graph, the step without a predictor.
            manager.end_attempt()
            np.testing.assert_array_equal(manager.core.enabled.numpy(), [0])
            self.assertEqual(manager.capture_key(), key)
            plain.velocity_increment.assign(flagged.velocity_increment.numpy())
            wp.copy(load, loads[1])
            wp.capture_launch(capture.graph)
            plain.step(plain_in, plain_out, control, None, dt, external_forces=load)
            self.assertEqual(bits(flagged_out), bits(plain_out))
            self.assertEqual(counters(manager), applied)
        finally:
            provider.close()

    def test_gradient_and_custom_solver_paths_do_not_prepare_a_basis(self):
        with _scene() as (scene, _):
            scene.close()
            provider = Mock(side_effect=AssertionError("Unsupported attempt requested a basis"))
            manager = AffineGuessManager(scene.solver, provider)
            # The real small native scene has no committed tracks. Expose only
            # CUDA eligibility during these host checks; no device work follows.
            model = scene.model
            with patch.object(model, "device", SimpleNamespace(is_cuda=True)):
                manager.begin_attempt(scene.coupler, scene.state_0, scene.control)
                self.assertEqual(manager.status, "waiting for committed track support")
                for owner, name in (
                    (scene.state_0, "particle_q"),
                    (scene.state_0, "particle_qd"),
                    (scene.state_0, "particle_f"),
                    (scene.state_0, "body_q"),
                    (scene.state_0, "body_qd"),
                    (scene.state_0, "body_f"),
                    (scene.state_0, "joint_q"),
                    (scene.state_0, "joint_qd"),
                    *(
                        (scene.coupler._candidate, field)
                        for field in (
                            "particle_q",
                            "particle_qd",
                            "particle_f",
                            "joint_q",
                            "joint_qd",
                            "body_q",
                            "body_qd",
                            "body_f",
                        )
                    ),
                    (scene.control, "joint_f"),
                    (scene.control, "joint_target_q"),
                    (scene.control, "joint_target_qd"),
                    (scene.control, "tet_activations"),
                ):
                    with self.subTest(field=name), patch.object(owner, name, SimpleNamespace(requires_grad=True)):
                        manager.begin_attempt(scene.coupler, scene.state_0, scene.control)
                        self.assertEqual(manager.status, "unsupported solver configuration")
                        self.assertFalse(manager.active)
                with patch.object(
                    scene.control, "mujoco", SimpleNamespace(ctrl=SimpleNamespace(requires_grad=True)), create=True
                ):
                    manager.begin_attempt(scene.coupler, scene.state_0, scene.control)
                    self.assertEqual(manager.status, "unsupported solver configuration")
                with patch.object(scene.solver, "step", wraps=scene.solver.step):
                    manager.begin_attempt(scene.coupler, scene.state_0, scene.control)
                    self.assertEqual(manager.status, "unsupported solver configuration")
                with patch.object(scene.rigid_solver, "step", wraps=scene.rigid_solver.step):
                    manager.begin_attempt(scene.coupler, scene.state_0, scene.control)
                    self.assertEqual(manager.status, "unsupported solver configuration")
                with wp.Tape():
                    manager.begin_attempt(scene.coupler, scene.state_0, scene.control)
                    self.assertEqual(manager.status, "unsupported solver configuration")
            provider.assert_not_called()

    def test_coupler_preparation_failure_clears_scope_and_restores_native_state(self):
        with _example() as example:
            h, native, coupler = native_embedded(example)
            manager = AffineGuessManager(example.solver, Mock())
            example.solver._affine_predictor = manager
            before = _native_state(native)
            positions = example.state_0.particle_q.numpy().copy()

            def fail(*args):
                manager.active = True
                raise RuntimeError("basis preparation failed")

            with patch.object(manager, "begin_attempt", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "basis preparation failed"):
                    coupler.step(example.state_0, example.state_1, example.control, h)
            self.assertFalse(manager.active)
            self.assertIsNone(manager.capture_key())
            np.testing.assert_array_equal(_native_state(native), before)
            np.testing.assert_array_equal(example.state_0.particle_q.numpy(), positions)

    def test_repeated_support_overflow_never_reactivates_stale_smaller_binding(self):
        self._failed_support_attempts(capacity=3, fails_provider=False)

    def test_provider_failure_retries_support_instead_of_committing_track_count(self):
        self._failed_support_attempts(capacity=9, fails_provider=True)

    def _failed_support_attempts(self, *, capacity, fails_provider):
        with _scene() as (scene, _):
            scene.close()
            provider = Mock(side_effect=ValueError("basis setup failed"))
            manager = AffineGuessManager(scene.solver, provider, capacity=capacity)
            manager.core = Mock()
            manager.structural_key = ("same owned binding",)
            manager.support = (0, 1, 2)
            manager._track_count = 1
            free = wp.array([0, 1], dtype=int, device=scene.state_0.particle_q.device)
            fixed = SimpleNamespace(free=free, compact_matrix=object())
            interaction = SimpleNamespace(track_count=2, _track_nodes_host=[[0, 0, 0, 0], [1, 1, 1, 1]])
            with (
                patch.object(scene.model, "device", SimpleNamespace(is_cuda=True)),
                patch.object(scene.solver, "_fixed_particles", fixed),
                patch.object(scene.coupler, "interaction", interaction),
                patch.object(scene.coupler, "_refresh_projection"),
                patch.object(manager, "_structure", return_value=manager.structural_key),
            ):
                for _ in range(2):
                    if fails_provider:
                        with self.assertRaisesRegex(ValueError, "basis setup failed"):
                            manager.begin_attempt(scene.coupler, scene.state_0, scene.control)
                    else:
                        manager.begin_attempt(scene.coupler, scene.state_0, scene.control)
                        self.assertEqual(manager.status, "track support outside response capacity")
                    self.assertFalse(manager.active)
                    self.assertEqual(manager._track_count, 1)
                    self.assertEqual(manager.support, (0, 1, 2))
                self.assertEqual(provider.call_count, 2 if fails_provider else 0)
                # A later usable support may reuse the original owned basis.
                # Only that successful binding commits the new track count.
                interaction._track_nodes_host[1] = [0, 0, 0, 0]
                manager.begin_attempt(scene.coupler, scene.state_0, scene.control)
                self.assertTrue(manager.active)
                self.assertEqual(manager._track_count, 2)
                manager.end_attempt()
                manager.begin_attempt(scene.coupler, scene.state_0, scene.control)
                self.assertTrue(manager.active)
                self.assertEqual(provider.call_count, 2 if fails_provider else 0)

    def test_numeric_refresh_retains_reference_but_reset_invalidates_binding(self):
        source = prepared_fixture.TestPreparedFEMStep()
        model, state, make = source.fixture(mixed=True)
        solver = make()
        manager = AffineGuessManager(solver, Mock())
        solver._affine_predictor = manager
        manager.structural_key = ("retained reference",)
        solver.refresh_tangent(state.particle_q, source.dt)
        self.assertEqual(manager.structural_key, ("retained reference",))
        solver.reset(state)
        self.assertIsNone(manager.structural_key)
        self.assertFalse(manager.active)


if __name__ == "__main__":
    unittest.main()
