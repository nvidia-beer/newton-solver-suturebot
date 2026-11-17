"""Backend routing and capture epochs retain the current native contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp
from test_certified_fem_scope import configure
from test_combined_contact_preconditioner import native_embedded
from test_needle_insertion_example import _example

from newton_solver_suturebot._src.solvers.coupled.coupling import NeedleTissueCoupler
from newton_solver_suturebot._src.solvers.coupled.mujoco_warp_trials import _surface_map_epoch
from newton_solver_suturebot._src.solvers.fem.certified_scope import require_certificate
from newton_solver_suturebot._src.solvers.mujoco.experimental_rigid_backend import is_gpu_backend, supported_subcycled
from newton_solver_suturebot._src.solvers.mujoco.subcycled_mujoco_warp import SubcycledMuJoCoWarp


class RebasedBackendContracts(unittest.TestCase):
    def test_gpu_checkpoint_restore_never_invokes_native_integration_or_shadow_restore(self):
        owner = object.__new__(NeedleTissueCoupler)
        owner._gpu_checkpoint = SimpleNamespace(restore=Mock())
        owner._mujoco = Mock(side_effect=AssertionError("Native integration is not the GPU transaction"))
        owner._restore_native()
        owner._gpu_checkpoint.restore.assert_called_once_with()
        self.assertEqual(owner._mujoco.mock_calls, [])

    def test_exact_gpu_type_extends_certificate_without_relaxing_gradient_or_custom_guards(self):
        with _example() as ex:
            _, native, coupler = native_embedded(ex)
            scope = configure(ex.solver)
            gpu = object.__new__(SubcycledMuJoCoWarp)
            gpu.model = ex.model
            gpu.use_mujoco_cpu = False
            self.assertTrue(supported_subcycled(native))
            self.assertTrue(is_gpu_backend(gpu))
            with patch.object(coupler, "rigid_solver", gpu):
                self.assertTrue(scope.forward_supported(coupler, ex.state_0, ex.control))
                with patch.object(gpu, "step", Mock()):
                    self.assertFalse(scope.forward_supported(coupler, ex.state_0, ex.control))
                with patch.object(
                    ex.control, "joint_target_q", wp.clone(ex.control.joint_target_q, requires_grad=True)
                ):
                    self.assertFalse(scope.forward_supported(coupler, ex.state_0, ex.control))
                with wp.Tape():
                    self.assertFalse(scope.forward_supported(coupler, ex.state_0, ex.control))

    def test_surface_selection_changes_reset_only_numerical_history_epoch(self):
        state = wp.array([11, 3, 2, 0, 0, 4], dtype=int, device="cpu")
        values = wp.array([1.7, 0.03, 0.001, 0.04], dtype=wp.float64, device="cpu")
        valid = wp.ones(1, dtype=int, device="cpu")
        previous = wp.zeros(1, dtype=int, device="cpu")
        wp.launch(_surface_map_epoch, 1, [valid, previous, state, values, 0.3], device="cpu")
        np.testing.assert_array_equal(state.numpy(), [11, 11, 2, 0, 0, 4])
        np.testing.assert_array_equal(values.numpy(), [0.3, 0.03, 0.001, 0.04])
        # Repeating the same surface map must preserve adapted relaxation.
        values.assign(np.asarray([0.7, 0.02, 0.001, 0.04], dtype=np.float64))
        wp.launch(_surface_map_epoch, 1, [valid, previous, state, values, 0.3], device="cpu")
        self.assertEqual(values.numpy()[0], 0.7)
        valid.zero_()
        wp.launch(_surface_map_epoch, 1, [valid, previous, state, values, 0.3], device="cpu")
        self.assertEqual(values.numpy()[0], 0.3)
        self.assertEqual(previous.numpy()[0], 0)

    def test_failed_current_A_certificate_cannot_leave_device_candidate_valid(self):
        valid = wp.ones(1, dtype=int, device="cpu")
        certificate = wp.zeros(1, dtype=int, device="cpu")
        wp.launch(require_certificate, 1, [certificate, valid], device="cpu")
        self.assertEqual(valid.numpy()[0], 0)
        # A successful linear solve cannot erase an independent rigid failure.
        certificate.fill_(1)
        wp.launch(require_certificate, 1, [certificate, valid], device="cpu")
        self.assertEqual(valid.numpy()[0], 0)


if __name__ == "__main__":
    unittest.main()
