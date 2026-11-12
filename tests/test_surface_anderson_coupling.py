# SPDX-License-Identifier: Apache-2.0
"""Anderson proposals keep native MuJoCo/FEM contact acceptance transactional."""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
from test_needle_insertion_example import _example, _prepare
from test_needle_tissue_coupler import _native_state

from newton_solver_suturebot._src.coupling import (
    NeedleCouplingNotConverged,
    NeedleTissueCoupler,
)
from newton_solver_suturebot._src.subcycled_mujoco import SubcycledMuJoCo
from newton_solver_suturebot._src.surface_anderson import SurfaceAnderson

_STATE_FIELDS = (
    "particle_q",
    "particle_qd",
    "particle_f",
    "body_q",
    "body_qd",
    "body_f",
    "joint_q",
    "joint_qd",
)


@contextmanager
def _contact(*, accelerated=True, iterations=160, absolute_tolerance=2e-8, relative_tolerance=1e-5):
    """Real FEM block and native subcycled free needle, contacting before puncture.

    The fixture uses the example's real contact/friction kernels and fixed FEM
    boundaries. No analytic or mocked mechanics replace either physical solve.
    """
    with _example() as example:
        _prepare(example)
        example.start()
        pose = example.plan.pose(0).copy()
        pose[2] += 0.6 * example.geometry.thickness
        example._initialize_pose(pose)
        example.interaction.puncture_force = 1.0
        native = SubcycledMuJoCo(
            example.model,
            use_mujoco_cpu=True,
            rigid_substeps=4,
            iterations=200,
            ls_iterations=50,
            tolerance=1e-12,
            integrator="implicitfast",
            cone="elliptic",
        )
        native.reset(example.state_0, flags=0)
        native.mj_data.time = 3.25
        example.solver.linear_solver_tolerance = 1e-6
        example.solver.refresh_tangent(example.state_0.particle_q, example.sim_dt)
        external = np.zeros((example.model.body_count, 6), dtype=np.float32)
        external[example.needle_body] = [0.00001, 0, -0.0002, 0, 0, 1e-8]
        example.state_0.body_f.assign(external)
        coupler = NeedleTissueCoupler(
            example.model,
            example.solver,
            native,
            example.interaction,
            example.needle_body,
            example.drive.com,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            max_iterations=iterations,
            relaxation=1.0,
            compact_rigid=False,
        )
        coupler._surface_anderson_enabled = accelerated
        yield example, coupler, native, external


def _snapshot(state):
    return {name: getattr(state, name).numpy().copy() for name in _STATE_FIELDS}


class SurfaceAndersonCouplingTests(unittest.TestCase):
    def test_real_fem_contact_matches_aitken_at_identical_acceptance_tolerances(self):
        records = []
        for accelerated in (False, True):
            with (
                self.subTest(accelerated=accelerated),
                # Tighten both reference and accelerated test fixtures to resolve
                # their shared fixed point well below the state comparison bound.
                _contact(accelerated=accelerated, absolute_tolerance=2e-9, relative_tolerance=1e-6) as (
                    ex,
                    coupler,
                    native,
                    external,
                ),
            ):
                fixed = ex.state_0.particle_q.numpy()[ex.fixed_vertex_indices].copy()
                with patch.object(ex.interaction, "accept", wraps=ex.interaction.accept) as accept:
                    coupler.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
                    self.assertEqual(accept.call_count, 1)
                self.assertAlmostEqual(native.mj_data.time, 3.25 + ex.sim_dt, places=12)
                self.assertEqual(ex.interaction.status, "contact")
                self.assertEqual(ex.interaction.events, ())
                self.assertEqual((ex.interaction.puncture_count, ex.interaction.track_count), (0, 0))
                self.assertGreater(ex.interaction.last_tip_force, 0)
                self.assertLess(ex.interaction.force_balance_error, 1e-8)
                self.assertLess(ex.interaction.torque_balance_error, 1e-10)
                self.assertLessEqual(
                    coupler.residual,
                    coupler.absolute_tolerance
                    + coupler.relative_tolerance
                    * max(
                        np.max(np.abs(coupler._value.numpy())),
                        np.max(np.abs(coupler._trial_load.numpy())),
                    ),
                )
                self.assertEqual(ex.solver.linear_solver_tolerance, 1e-6)
                self.assertEqual(coupler.max_iterations, 160)
                np.testing.assert_array_equal(ex.state_1.particle_q.numpy()[ex.fixed_vertex_indices], fixed)
                np.testing.assert_array_equal(ex.state_0.body_f.numpy(), external)
                np.testing.assert_array_equal(ex.state_1.body_f.numpy(), external)
                if accelerated:
                    self.assertGreater(coupler._surface_anderson_proposals, 0)
                records.append(
                    (
                        _snapshot(ex.state_1),
                        ex.interaction.interface_force.copy(),
                        coupler.iterations,
                    )
                )
        for field in (
            "particle_q",
            "particle_qd",
            "body_q",
            "body_qd",
            "joint_q",
            "joint_qd",
        ):
            np.testing.assert_allclose(
                records[1][0][field],
                records[0][0][field],
                atol=3e-7,
                rtol=2e-5,
                err_msg=field,
            )
        np.testing.assert_allclose(records[1][1], records[0][1], atol=5e-8, rtol=2e-5)
        self.assertLess(records[1][2], records[0][2])

    def test_gradient_enabled_input_and_candidate_arrays_keep_original_iteration(self):
        for which in ("input", "candidate"):
            for field in _STATE_FIELDS:
                with self.subTest(state=which, field=field), _contact() as (ex, coupler, _native, _external):
                    state = ex.state_0 if which == "input" else coupler._candidate
                    array = getattr(state, field)
                    array.requires_grad = True
                    self.assertFalse(ex.model.requires_grad)
                    with patch.object(
                        SurfaceAnderson, "propose", side_effect=AssertionError("host AA detached gradients")
                    ):
                        coupler.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
                    self.assertEqual(coupler._surface_anderson_proposals, 0)
                    self.assertEqual(ex.interaction.status, "contact")

    def test_gradient_enabled_controls_keep_original_iteration(self):
        for field in ("joint_f", "joint_target_q", "joint_target_qd", "tet_activations"):
            with self.subTest(field=field), _contact() as (ex, coupler, _native, _external):
                array = getattr(ex.control, field)
                array.requires_grad = True
                with patch.object(SurfaceAnderson, "propose", side_effect=AssertionError("host AA detached controls")):
                    coupler.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
                self.assertEqual(coupler._surface_anderson_proposals, 0)
                self.assertEqual(ex.interaction.status, "contact")

    def test_exhausted_budget_after_acceleration_restores_native_fem_and_output(self):
        with _contact(iterations=6) as (ex, coupler, native, _external):
            self._assert_rollback(ex, coupler, native, NeedleCouplingNotConverged, "did not converge")
            self.assertGreater(coupler._surface_anderson_proposals, 0)

    def test_exception_after_an_accelerated_trial_restores_transaction(self):
        with _contact() as (ex, coupler, native, _external):
            original = SurfaceAnderson.propose

            def fail_after_a_real_proposal(acceleration, *args):
                if acceleration.proposals:
                    raise RuntimeError("injected acceleration exception after native/FEM trial")
                return original(acceleration, *args)

            with patch.object(SurfaceAnderson, "propose", fail_after_a_real_proposal):
                self._assert_rollback(ex, coupler, native, RuntimeError, "injected acceleration exception")
            self.assertGreater(coupler._surface_anderson_proposals, 0)

    def _assert_rollback(self, ex, coupler, native, exception, message):
        before, output = _snapshot(ex.state_0), _snapshot(ex.state_1)
        native_before = _native_state(native)
        increment = ex.solver.velocity_increment.numpy().copy()
        status = ex.interaction.status
        with patch.object(ex.interaction, "accept", wraps=ex.interaction.accept) as accept:
            with self.assertRaisesRegex(exception, message):
                coupler.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
            self.assertEqual(accept.call_count, 0)
        np.testing.assert_array_equal(_native_state(native), native_before)
        np.testing.assert_array_equal(ex.solver.velocity_increment.numpy(), increment)
        for field, values in before.items():
            np.testing.assert_array_equal(getattr(ex.state_0, field).numpy(), values, err_msg=field)
        for field, values in output.items():
            np.testing.assert_array_equal(getattr(ex.state_1, field).numpy(), values, err_msg=field)
        self.assertEqual(ex.interaction.status, status)
        self.assertEqual(ex.interaction.events, ())
        self.assertEqual((ex.interaction.puncture_count, ex.interaction.track_count), (0, 0))
        self.assertIsNone(ex.interaction._trial_fields)
        self.assertIsNone(ex.interaction._pending_opening)
        self.assertEqual(ex.solver.linear_solver_tolerance, 1e-6)


if __name__ == "__main__":
    unittest.main()
