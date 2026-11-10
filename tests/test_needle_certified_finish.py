# SPDX-License-Identifier: Apache-2.0
"""Certified endpoint metadata reuses forces only after an exact native audit."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import test_needle_coupling_graph as graph_tests
import warp as wp
from test_needle_coupling_graph import _DT, _pair, _snapshot

from newton_solver_suturebot import NeedleStepTooLarge, NeedleTissueCoupler, NeedleTissueInteraction
from newton_solver_suturebot._src.coupling import _compare_rigid_endpoint, _snapshot_rigid_endpoint


class EndpointAuditTests(unittest.TestCase):
    def test_all_joint_and_body_fields_are_compared_bitwise(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        parts = [wp.zeros(n, dtype=wp.uint32, device=device) for n in (7, 6, 7, 6)]
        snapshot = wp.empty(26, dtype=wp.uint32, device=device)
        equal = wp.ones(1, dtype=int, device=device)
        wp.launch(_snapshot_rigid_endpoint, 26, [*parts, snapshot], device=device)
        for part in parts:
            for index in range(part.size):
                # The float sign bit distinguishes +0 from -0, so a numeric
                # float equality implementation would miss this discrepancy.
                values = np.zeros(part.size, np.uint32)
                values[index] = 0x80000000
                part.assign(values)
                equal.fill_(1)
                wp.launch(_compare_rigid_endpoint, 26, [*parts, snapshot, equal], device=device)
                self.assertEqual(equal.numpy()[0], 0)
                part.zero_()
        equal.fill_(1)
        wp.launch(_compare_rigid_endpoint, 26, [*parts, snapshot, equal], device=device)
        self.assertEqual(equal.numpy()[0], 1)


class CertifiedFinishTests(unittest.TestCase):
    assert_equivalent = graph_tests.NeedleCouplingGraphTests.assert_equivalent
    assert_rollback = graph_tests.NeedleCouplingGraphTests.assert_rollback

    def setUp(self):
        if not wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda:
            self.skipTest("Captured endpoint finish requires CUDA")

    def test_certified_finish_reuses_forces_and_publishes_identical_metadata(self):
        for mode in ("penalty", "alm"):
            with self.subTest(mode=mode), _pair(mode) as (reference, captured):
                reference.coupler.step(reference.state_0, reference.state_1, reference.control, _DT)
                point_counts = []
                original_query = NeedleTissueInteraction._prepare_device_queries

                def query(interaction, state, q, points, point_counts=point_counts, original_query=original_query):
                    point_counts.append(len(points))
                    return original_query(interaction, state, q, points)

                def forbidden(*args, **kwargs):
                    raise AssertionError("Certified finish repeated the original force evaluation")

                # Patch the class so the eligibility guard still sees its
                # canonical method, then prove the successful path never calls it.
                with (
                    patch.object(NeedleTissueCoupler, "_evaluate", forbidden),
                    patch.object(NeedleTissueInteraction, "_prepare_device_queries", query),
                ):
                    captured.coupler.step(captured.state_0, captured.state_1, captured.control, _DT)
                self.assert_equivalent(reference, captured)
                self.assertEqual(point_counts, [1])
                self.assertEqual(captured.coupler.coupling_graph_fallbacks, 0)
                self.assertEqual(captured.coupler.coupling_graph_verifications, 1)
                self.assertEqual(captured.coupler.evaluations, captured.coupler.iterations)
                self.assertIsNone(captured.interaction._trial_fields)
                self.assertIsNone(captured.interaction._pending_opening)
                for name in (
                    "interface_force",
                    "interface_torque",
                    "interface_torque_origin",
                    "tissue_force",
                    "support_force",
                    "support_torque",
                    "last_tip_force",
                    "force_balance_error",
                    "torque_balance_error",
                    "tip_position",
                ):
                    np.testing.assert_allclose(
                        getattr(captured.interaction, name),
                        getattr(reference.interaction, name),
                        rtol=3e-6,
                        atol=1e-12,
                        err_msg=name,
                    )

    def test_native_mismatch_in_any_endpoint_field_restores_then_uses_host(self):
        for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
            with self.subTest(field=field), _pair("alm") as (reference, captured):
                original_step = captured.coupler._rigid_bridge.step
                calls = 0

                def mismatch_once(source, destination, control, dt, original_step=original_step, field=field):
                    nonlocal calls
                    original_step(source, destination, control, dt)
                    calls += 1
                    if calls == 1:
                        values = getattr(destination, field).numpy().copy()
                        values.view(np.uint32).flat[0] ^= np.uint32(1)
                        getattr(destination, field).assign(values)

                reference.coupler.step(reference.state_0, reference.state_1, reference.control, _DT)
                with patch.object(captured.coupler._rigid_bridge, "step", mismatch_once):
                    captured.coupler.step(captured.state_0, captured.state_1, captured.control, _DT)
                self.assert_equivalent(reference, captured)
                self.assertGreater(calls, 1)
                self.assertEqual(captured.coupler.coupling_graph_fallbacks, 1)
                self.assertEqual(captured.coupler.coupling_graph_status, "exact endpoint fallback")
                self.assertEqual(captured.interaction.entry_count, reference.interaction.entry_count)

    def test_late_rejection_discards_staged_metadata_and_identical_retry_succeeds(self):
        with _pair("alm") as (reference, captured):
            saved = _snapshot(captured)

            def reject(*args):
                self.assertIsNotNone(captured.interaction._trial_fields)
                raise NeedleStepTooLarge("after certified metadata")

            with patch.object(captured.interaction, "check_step_resolution", reject):
                with self.assertRaisesRegex(NeedleStepTooLarge, "after certified metadata"):
                    captured.coupler.step(captured.state_0, captured.state_1, captured.control, _DT)
            self.assert_rollback(captured, saved)
            self.assertIsNone(captured.interaction._trial_fields)
            reference.coupler.step(reference.state_0, reference.state_1, reference.control, _DT)
            captured.coupler.step(captured.state_0, captured.state_1, captured.control, _DT)
            self.assert_equivalent(reference, captured)
            self.assertEqual(captured.coupler.coupling_graph_captures, 1)


if __name__ == "__main__":
    unittest.main()
