# SPDX-License-Identifier: Apache-2.0
"""Both Anderson transports retain raw acceptance, history epochs and rollback."""

import inspect
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from test_combined_contact_preconditioner import native_embedded
from test_needle_insertion_example import _example
from test_needle_shaft_preconditioner import history
from test_needle_tissue_coupler import _native_state
from test_surface_anderson_coupling import _snapshot

from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_anderson import SurfaceAnderson
from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_anderson_device import DeviceSurfaceAnderson
from newton_solver_suturebot._src.solvers.coupled.coupling import NeedleCouplingNotConverged, NeedleTissueCoupler
from newton_solver_suturebot._src.solvers.coupled.interface.tunnel_alm import _TunnelALM
from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene
from newton_solver_suturebot.examples.needle.example_fem_needle_insertion import Example


def observe(test, ex, coupler, *, device):
    """Observe the selected real transport without replacing either physical solver."""
    calls, instances = [], []

    def record(instance, load, residual, direction):
        alm = ex.interaction.tunnel_alm
        epoch = (
            ex.solver.linear_solver_tolerance,
            None if alm is None else alm.dual_updates,
            coupler._surface_preconditioner.selection_changes,
        )
        test.assertIsNotNone(direction)
        if device:
            # Read only for this diagnostic test; the real transport receives
            # the actual device arrays and executes its ordinary pack/SVD/scatter.
            test.assertIs(load, coupler._trial_load)
            test.assertIs(residual, coupler._residual)
            test.assertIs(direction, coupler._direction)
            length, pending = len(instance._slots), instance._pending_metadata
            raw, search = residual.numpy(), direction.numpy()
        else:
            length, pending = len(instance.history), instance.pending
            raw, search = residual, direction
        if epoch != instance.epoch:
            test.assertEqual(length, 0, "A changed force map retained old secants")
            test.assertIsNone(pending, "A changed force map retained a recovery guess")
        np.testing.assert_array_equal(raw, coupler._residual.numpy())
        np.testing.assert_array_equal(search, coupler._direction.numpy())
        calls.append((epoch, float(np.max(np.abs(search - raw))), length, pending is not None))
        instance.epoch = epoch

    class ObserveCPU(SurfaceAnderson):
        def __init__(self):
            self.epoch = None
            super().__init__()
            instances.append(self)

        def propose(self, load, residual, relaxation, direction=None):
            record(self, load, residual, direction)
            return super().propose(load, residual, relaxation, direction=direction)

    class ObserveDevice(DeviceSurfaceAnderson):
        def __init__(self, template, **kwargs):
            self.epoch = None
            super().__init__(template, **kwargs)
            instances.append(self)

        def propose_device(self, load, residual, relaxation, direction=None):
            record(self, load, residual, direction)
            return super().propose_device(load, residual, relaxation, direction=direction)

    name = "surface_anderson_device.DeviceSurfaceAnderson" if device else "surface_anderson.SurfaceAnderson"
    cls = ObserveDevice if device else ObserveCPU
    return patch("newton_solver_suturebot._src.solvers.coupled.acceleration." + name, cls), calls, instances


class PreconditionedAndersonCouplingTests(unittest.TestCase):
    def transports(self):
        # CUDA runs both the explicit CPU fallback and the actual device path.
        # CPU remains an ordinary supported fallback, not an emulated GPU loop.
        return (False, True) if wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda else (False,)

    def assert_transport(self, coupler, instances, device):
        self.assertTrue(instances, "The test did not exercise the selected Anderson path")
        workspace = getattr(coupler, "_device_surface_anderson_workspace", None)
        if device:
            self.assertIs(workspace, instances[0])
            self.assertGreater(workspace.transport_calls, 0)
            self.assertFalse(workspace._slots, "Completed attempt retained device history")
            self.assertIsNone(workspace._pending_metadata)
        else:
            self.assertIsNone(workspace)

    def test_public_default_is_off_and_only_physical_grasp_example_opts_in(self):
        default = inspect.signature(NeedleTissueCoupler).parameters["preconditioned_anderson"].default
        self.assertIs(default, False)
        self.assertIs(Example.preconditioned_anderson, False)
        self.assertIs(GraspScene.preconditioned_anderson, True)
        with _example() as ex:
            h, _native, coupler = native_embedded(ex)
            coupler.preconditioned_anderson = False
            with patch.object(SurfaceAnderson, "propose", side_effect=AssertionError("unexpected opt-in")):
                coupler.step(ex.state_0, ex.state_1, ex.control, h)
            self.assertIsNone(getattr(coupler, "_device_surface_anderson_workspace", None))
            self.assertLessEqual(
                coupler.residual,
                coupler.absolute_tolerance
                + coupler.relative_tolerance
                * max(np.max(np.abs(coupler._value.numpy())), np.max(np.abs(coupler._trial_load.numpy()))),
            )

    def test_real_contact_uses_raw_residual_and_resets_on_inner_tolerance_change(self):
        for device in self.transports():
            with self.subTest(device_transport=device), _example() as ex:
                h, native, coupler = native_embedded(ex)
                coupler._device_surface_anderson_enabled = device
                ex.solver.linear_solver_tolerance = 3e-3
                observer, calls, instances = observe(self, ex, coupler, device=device)
                with observer:
                    coupler.step(ex.state_0, ex.state_1, ex.control, h)
                self.assert_transport(coupler, instances, device)
                self.assertGreater(len({call[0][0] for call in calls}), 1)
                self.assertGreater(max(call[1] for call in calls), 1e-6)
                self.assertGreater(instances[0].proposals, 0)
                self.assertEqual(ex.solver.linear_solver_tolerance, 3e-3)
                self.assertAlmostEqual(native.mj_data.time, 3.25 + h, places=12)

    def test_gradient_state_and_control_keep_original_preconditioned_iteration(self):
        for owner in ("input", "candidate", "control"):
            with self.subTest(owner=owner), _example() as ex:
                h, _native, coupler = native_embedded(ex)
                array = {
                    "input": ex.state_0.body_qd,
                    "candidate": coupler._candidate.particle_qd,
                    "control": ex.control.joint_target_qd,
                }[owner]
                array.requires_grad = True
                with patch.object(SurfaceAnderson, "propose", side_effect=AssertionError("detached gradient")):
                    coupler.step(ex.state_0, ex.state_1, ex.control, h)
                self.assertEqual(coupler._surface_anderson_proposals, 0)
                self.assertIsNone(getattr(coupler, "_device_surface_anderson_workspace", None))

    def test_real_tunnel_dual_updates_invalidate_numerical_history(self):
        for device in self.transports():
            with self.subTest(device_transport=device), _example() as ex:
                h, _native, coupler = native_embedded(ex)
                coupler._device_surface_anderson_enabled = device
                coupler.absolute_tolerance, coupler.relative_tolerance = 2e-7, 1e-4
                ex.interaction.friction = 0.02
                alm = ex.interaction.tunnel_alm = _TunnelALM(
                    ex.interaction,
                    penalty_per_m=6000,
                    compliance_per_m=1 / 6000,
                    clearance=2e-5,
                    tolerance=2e-6,
                )
                observer, calls, instances = observe(self, ex, coupler, device=device)
                with observer:
                    coupler.step(ex.state_0, ex.state_1, ex.control, h)
                self.assert_transport(coupler, instances, device)
                self.assertGreater(alm.dual_updates, 1)
                self.assertGreater(len({call[0][1] for call in calls}), 1)
                self.assertTrue(alm.converged(coupler.absolute_tolerance, coupler.relative_tolerance))
                self.assertFalse(alm._open)

    def test_surface_metric_validity_switch_clears_history_without_changing_forces(self):
        for device in self.transports():
            with self.subTest(device_transport=device), _example() as ex:
                h, _native, coupler = native_embedded(ex)
                coupler._device_surface_anderson_enabled = device
                pc = coupler._surface_preconditioner
                original, evaluations = pc.validate, []

                def optional_metric_unavailable(original=original, evaluations=evaluations, pc=pc):
                    original()
                    evaluations.append(None)
                    # Disable only the search approximation, never the forces.
                    if len(evaluations) in (2, 3, 7):
                        pc.valid.zero_()

                observer, calls, instances = observe(self, ex, coupler, device=device)
                with patch.object(pc, "validate", side_effect=optional_metric_unavailable), observer:
                    coupler.step(ex.state_0, ex.state_1, ex.control, h)
                self.assert_transport(coupler, instances, device)
                self.assertGreaterEqual(len({call[0][2] for call in calls}), 3)
                self.assertLess(ex.interaction.force_balance_error, 2e-8)
                self.assertLess(ex.interaction.torque_balance_error, 1e-9)

    def test_failed_preconditioned_proposals_restore_physics_history_and_inner_tolerance(self):
        for device in self.transports():
            with self.subTest(device_transport=device), _example() as ex:
                h, native, coupler = native_embedded(ex, iterations=4)
                coupler._device_surface_anderson_enabled = device
                ex.solver.linear_solver_tolerance = 3e-3
                before_in, before_out = _snapshot(ex.state_0), _snapshot(ex.state_1)
                before_native, before_history = _native_state(native), history(ex.interaction)
                before_guess = ex.solver.velocity_increment.numpy().copy()
                before_load = coupler._load.numpy().copy()
                observer, calls, instances = observe(self, ex, coupler, device=device)
                with observer, patch.object(ex.interaction, "accept", wraps=ex.interaction.accept) as accept:
                    with self.assertRaises(NeedleCouplingNotConverged):
                        coupler.step(ex.state_0, ex.state_1, ex.control, h)
                    self.assert_transport(coupler, instances, device)
                    self.assertGreater(instances[0].proposals, 0)
                    self.assertEqual(accept.call_count, 0)
                    for state, before in ((ex.state_0, before_in), (ex.state_1, before_out)):
                        for name, expected in before.items():
                            np.testing.assert_array_equal(getattr(state, name).numpy(), expected, err_msg=name)
                    np.testing.assert_array_equal(_native_state(native), before_native)
                    np.testing.assert_array_equal(ex.solver.velocity_increment.numpy(), before_guess)
                    np.testing.assert_array_equal(coupler._load.numpy(), before_load)
                    self.assertEqual(history(ex.interaction), before_history)
                    self.assertEqual(ex.solver.linear_solver_tolerance, 3e-3)
                    # A retry gets fresh secants, while the device path keeps
                    # its owned allocation. The same physical start must pass.
                    previous_calls = len(calls)
                    coupler.max_iterations = 160
                    coupler.step(ex.state_0, ex.state_1, ex.control, h)
                    self.assertGreater(len(calls), previous_calls)
                    self.assertEqual(calls[previous_calls][2:], (0, False))
                    self.assertEqual(len(instances), 1 if device else 2)
                    self.assertEqual(accept.call_count, 1)
                self.assert_transport(coupler, instances, device)
                self.assertAlmostEqual(native.mj_data.time, 3.25 + h, places=12)
                self.assertEqual(ex.solver.linear_solver_tolerance, 3e-3)


if __name__ == "__main__":
    unittest.main()
