# SPDX-License-Identifier: Apache-2.0
"""Parallel quadrature preserves elastic forces, PSD tangents and inactive chains."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot import ThreadModel
from newton_solver_suturebot._src.solvers.thread import kernels
from newton_solver_suturebot._src.solvers.thread.gauss import ParallelGaussElements


class ThreadGaussTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.device.__enter__()

    def tearDown(self):
        self.device.__exit__(None, None, None)

    def test_active_tape_retains_original_element_path(self):
        from test_thread_solver import DT, _line, _solver

        solver = _solver(_line(4), max_iterations=1)
        helper = ParallelGaussElements(solver.thread, solver.thread.element_count)
        solver._gauss_elements = helper
        with patch.object(helper, "evaluate", wraps=helper.evaluate) as evaluate:
            solver.step(None, None, None, None, DT)
            self.assertGreater(evaluate.call_count, 0)
            evaluate.reset_mock()
            with wp.Tape():
                solver.step(None, None, None, None, DT)
            evaluate.assert_not_called()

    def test_contact_friction_and_moving_head_preserve_complete_step_state(self):
        from test_thread_solver import DT, ELEMENT, _dragged_thread, _rounding

        reference, reference_step = _dragged_thread()
        candidate, candidate_step = _dragged_thread()
        reference._gauss_elements = None
        candidate._gauss_elements = ParallelGaussElements(candidate.thread, candidate.thread.element_count)
        for index in range(30):
            reference_step(index)
            candidate_step(index)
        expected = (*reference.state, reference.contact_force, reference.eyelet_force)
        actual = (*candidate.state, candidate.contact_force, candidate.eyelet_force)
        scale = np.tile(np.repeat((1.0, ELEMENT), 3), candidate.thread.node_count)
        position_bound = _rounding(30, 2)
        np.testing.assert_allclose(
            candidate.q.numpy() * scale, reference.q.numpy() * scale, rtol=0, atol=position_bound
        )
        np.testing.assert_allclose(
            candidate.qd.numpy() * scale, reference.qd.numpy() * scale, rtol=0, atol=position_bound / DT
        )
        for index, (first, second) in enumerate(zip(expected, actual, strict=True)):
            with self.subTest(state_array=index):
                a, b = first.numpy(), second.numpy()
                if a.dtype.kind in "iu":
                    np.testing.assert_array_equal(a, b)
                else:
                    np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-10)
        np.testing.assert_array_equal(candidate.iterations.numpy(), reference.iterations.numpy())
        self.assertGreater(candidate.eyelet_xi.numpy()[0], 0.55)
        self.assertGreater(np.linalg.norm(candidate.head_motion.numpy()[0, :3]), 2e-4)

    def test_compressed_stretched_and_curved_batched_gradients_and_tangents(self):
        rng = np.random.default_rng(573)
        for curved in (False, True):
            with self.subTest(curved=curved):
                t = np.linspace(0, 1, 13)
                points = np.column_stack((0.03 * t, 0.004 * np.sin(np.pi * t) if curved else 0 * t, 0 * t))
                thread = ThreadModel.create(points, young_modulus=2e6, radius=0.00015, density=1100)
                rest = thread.rest_q.numpy().reshape(-1, 6)
                states = []
                for stretch in (0.8, 1.0, 1.2):
                    q = rest.copy()
                    q[:, :3] *= stretch
                    q[:, 3:] *= stretch
                    q[:, :3] += rng.normal(scale=1e-4, size=q[:, :3].shape)
                    q[:, 3:] += rng.normal(scale=0.03, size=q[:, 3:].shape)
                    states.append(q)
                q = wp.array(np.concatenate(states).flatten(), dtype=wp.float64)
                count = 3 * thread.element_count
                status = wp.array([0, 2, 0], dtype=int)
                helper = ParallelGaussElements(thread, count)
                for tangent in (False, True):
                    gradient = wp.full((count, 4), wp.vec3d(123), dtype=wp.vec3d)
                    hessian = wp.full((count, 4, 4), wp.mat33d(321), dtype=wp.mat33d)
                    expected_gradient, expected_hessian = wp.clone(gradient), wp.clone(hessian)
                    wp.launch(
                        kernels.evaluate_elements,
                        count,
                        [
                            q,
                            thread.lengths,
                            thread.axial_stiffness,
                            thread.bending_stiffness,
                            thread.rest_q,
                            tangent,
                            status,
                            expected_gradient,
                            expected_hessian,
                        ],
                    )
                    helper.evaluate(q, tangent, status, gradient, hessian)
                    actual_g, reference_g = gradient.numpy(), expected_gradient.numpy()
                    actual_h, reference_h = hessian.numpy(), expected_hessian.numpy()
                    # Separate kernel stores prevent some fused multiply-adds.
                    # Fixed quadrature order limits the difference to float64
                    # roundoff; the model and every force/tangent term agree.
                    np.testing.assert_allclose(actual_g, reference_g, rtol=3e-13, atol=2e-13)
                    np.testing.assert_allclose(actual_h, reference_h, rtol=3e-13, atol=2e-11)
                    inactive = slice(thread.element_count, 2 * thread.element_count)
                    np.testing.assert_array_equal(actual_g[inactive], reference_g[inactive])
                    np.testing.assert_array_equal(actual_h[inactive], reference_h[inactive])
                    if not tangent:
                        np.testing.assert_array_equal(actual_h, reference_h)
                    else:
                        active = np.r_[np.arange(thread.element_count), np.arange(2 * thread.element_count, count)]
                        dense = actual_h[active].transpose(0, 1, 3, 2, 4).reshape(-1, 12, 12)
                        np.testing.assert_allclose(dense, dense.transpose(0, 2, 1), rtol=1e-14, atol=1e-13)
                        self.assertGreater(np.linalg.eigvalsh(dense).min(), -1e-9)


if __name__ == "__main__":
    unittest.main()
