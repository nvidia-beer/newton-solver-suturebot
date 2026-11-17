"""Retained outside authorization keeps exact forces, events and fallback guards."""

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import warp as wp
from test_needle_insertion_example import _example
from test_needle_shaft_preconditioner import setup
from test_needle_trial_forces import prepare
from test_surface_anderson_coupling import _contact, _snapshot
from test_surface_pc_capture import seed_contact

from newton_solver_suturebot import NeedleStepTooLarge
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import _closest_triangle
from newton_solver_suturebot._src.solvers.coupled.interface.needle_trial_forces import (
    FALLBACK_REASONS,
    _NeedleTrialForces,
    retained_opening_certificate,
    tip_material_certificate,
)
from newton_solver_suturebot._src.solvers.coupled.interface.tunnel_alm import _TunnelALM
from newton_solver_suturebot._src.utils.capture import ScopedCapture


def authorize(ex, body):
    """Seed a geometric fixture; this does not claim a dynamic entry/exit."""
    x = ex.interaction
    q, _ = prepare(x, ex.state_0, body)
    tip = x._body_tip(ex.state_0, body)
    face, bary, _, _ = _closest_triangle(tip, x.surface_vertices, x.surface_triangles)
    ids = x.mapping[x.surface_triangles[face]]
    assert np.all(ids >= 0) and x._locate(tip, q) is None
    x._opening = (ids.copy(), bary.copy())
    x._opening_faces = x._register_opening(ids, bary, face)
    x._in_material, x.punctured, x.status = False, True, "outside"
    x._tip_embedding = None
    x._previous_tip, x._previous_q = tip.copy(), q.copy()
    return x


@contextmanager
def contact():
    with _contact(accelerated=False) as record:
        ex, c, _, _ = record
        seed_contact(ex, c)
        authorize(ex, c.needle_body)
        yield record


class AuthorizedOutsideTests(unittest.TestCase):
    def test_strict_distance_margin_and_invalid_opening(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        nodes = wp.array([[0, 1, 2]], dtype=wp.vec3i, device=device)
        bary = wp.array([[0.2, 0.3, 0.5]], dtype=wp.vec3d, device=device)
        valid = wp.ones(1, dtype=int, device=device)
        q = wp.zeros(3, dtype=wp.vec3, device=device)
        samples = wp.empty(1, dtype=wp.vec3d, device=device)
        pose = wp.array([[0, 0, 0, 0, 0, 0, 1]], dtype=wp.transform, device=device)
        flags = wp.array([0, 1, 1], dtype=int, device=device)
        retained = wp.empty(1, dtype=int, device=device)
        for distance, expected in ((0, 1), (0.0004, 1), (0.00056 - 1e-13, 0), (0.00056, 0), (0.001, 0), (np.nan, 0)):
            with self.subTest(distance=distance):
                samples.assign([[distance, 0, 0]])
                wp.launch(
                    retained_opening_certificate,
                    1,
                    [nodes, bary, valid, q, samples, pose, 0, wp.vec3d(), flags, 0.00056, retained],
                    device=device,
                )
                self.assertEqual(int(retained.numpy()[0]), expected)
        valid.zero_()
        samples.zero_()
        wp.launch(
            retained_opening_certificate,
            1,
            [nodes, bary, valid, q, samples, pose, 0, wp.vec3d(), flags, 0.00056, retained],
            device=device,
        )
        self.assertEqual(int(retained.numpy()[0]), 0)

    def test_material_requires_proven_retention_and_clears_outside_tip_mask(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        cell = wp.array([-1], dtype=int, device=device)
        bary = wp.array([[0.25] * 4], dtype=wp.vec4d, device=device)
        tets = wp.array([[0, 1, 2, 3]], dtype=wp.vec4i, device=device)
        fixed = wp.zeros(4, dtype=int, device=device)
        flags = wp.array([0, 1, 1], dtype=int, device=device)
        retained = wp.zeros(1, dtype=int, device=device)
        mask = wp.ones(1, dtype=int, device=device)
        # The certificate records the reason of a request beside the request itself.
        fallback = wp.zeros(len(FALLBACK_REASONS), dtype=int, device=device)
        for proof, location, expected in ((0, -1, 1), (1, -1, 0), (1, 0, 1)):
            retained.fill_(proof)
            cell.fill_(location)
            mask.fill_(4)
            fallback.zero_()
            wp.launch(
                tip_material_certificate,
                1,
                [cell, bary, tets, fixed, flags, retained, 0.0002, 0.00056, mask, fallback],
                device=device,
            )
            self.assertEqual(int(fallback.numpy()[0]), expected)
            self.assertEqual(int(mask.numpy()[0]), 0)

    def test_nonzero_penalty_and_alm_force_metadata_parity(self):
        with _example() as ex:
            setup(ex, 0.0005, 1500)
            ex._initialize_pose(ex.plan.pose(ex.plan.exit_angle + 0.03))
            x = authorize(ex, ex.needle_body)
            helper = _NeedleTrialForces(x)
            saved = (x.track_count, x.puncture_count, x.entry_count, x.exit_count, repr(x.events))
            for alm in (False, True):
                with self.subTest(alm=alm):
                    if alm:
                        x.tunnel_alm = _TunnelALM(
                            x, penalty_per_m=60000, compliance_per_m=1 / 60000, clearance=0.00002, tolerance=0.000002
                        )
                        x.tunnel_alm.begin_step()
                    rows, fields = [], []
                    for captured in (False, True):
                        ex.state_0.body_f.zero_()
                        ex._external_forces.zero_()
                        if captured:
                            helper.add_forces(ex.state_0, ex.needle_body, ex.drive.com, ex._external_forces, 0.0005)
                            self.assertEqual(int(helper.fallback.numpy()[0]), 0)
                            self.assertTrue(
                                x._stage_certified_trial(ex.state_0, ex.needle_body, ex.drive.com, helper.mask)
                            )
                        else:
                            x.add_forces(
                                ex.state_0,
                                ex.needle_body,
                                ex.drive.com,
                                ex._external_forces,
                                0.0005,
                                implicit=True,
                                trial=True,
                            )
                        rows.append(
                            [
                                ex._external_forces.numpy().copy(),
                                ex.state_0.body_f.numpy().copy(),
                                x._contact_flags.numpy().copy(),
                                x._diagnostics.numpy().copy(),
                            ]
                        )
                        if alm:
                            rows[-1] += [
                                x.tunnel_alm.proposed_normal.numpy().copy(),
                                x.tunnel_alm.proposed_tangent.numpy().copy(),
                                x.tunnel_alm._statistics.numpy().copy(),
                            ]
                        fields.append(x._trial_fields.copy())
                        x.discard_trial_forces()
                    self.assertGreater(np.linalg.norm(rows[0][0]), 1e-5)
                    for a, b in zip(*rows, strict=True):
                        np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-10)
                    for key, value in fields[0].items():
                        if key == "_opening":
                            for a, b in zip(value, fields[1][key], strict=True):
                                np.testing.assert_array_equal(a, b)
                        else:
                            np.testing.assert_array_equal(value, fields[1][key], err_msg=key)
                    self.assertEqual(
                        saved, (x.track_count, x.puncture_count, x.entry_count, x.exit_count, repr(x.events))
                    )
                    self.assertTrue(x.punctured)
                    self.assertFalse(x._in_material)
                    if alm:
                        x.tunnel_alm.discard()

    def test_current_opening_updates_without_count_change_and_storage_rebind_invalidates(self):
        with contact() as (ex, c, _, _):
            x = ex.interaction
            helper = _NeedleTrialForces(x)
            signature = helper.signature()
            original = (x._opening[0].copy(), x._opening[1].copy())
            helper.evaluate(ex.state_0, c.needle_body)
            self.assertEqual(int(helper.fallback.numpy()[0]), 0)
            # Same registered history, different current opening: retained capture
            # must follow mutable accepted metadata, not stale copied values.
            x._opening = (original[0], np.array([1.0, 0.0, 0.0]))
            self.assertFalse(helper.prepare(x))
            self.assertEqual(signature, helper.signature())
            helper.evaluate(ex.state_0, c.needle_body)
            self.assertEqual(int(helper.fallback.numpy()[0]), 1)
            x._opening = original
            helper.prepare(x)
            helper.evaluate(ex.state_0, c.needle_body)
            self.assertEqual(int(helper.fallback.numpy()[0]), 0)
            helper.current_opening_bary = wp.clone(helper.current_opening_bary)
            self.assertNotEqual(signature, helper.signature())
            x._opening = None
            helper.prepare(x)
            helper.evaluate(ex.state_0, c.needle_body)
            self.assertEqual(int(helper.fallback.numpy()[0]), 1)
            self.assertFalse(x._stage_certified_trial(ex.state_0, c.needle_body, c.needle_com, helper.mask))

    def test_candidate_clearance_entry_sweep_and_virgin_reuse_still_fall_back(self):
        with contact() as (ex, c, _, _):
            x = ex.interaction
            helper = _NeedleTrialForces(x)
            pose = ex.state_0.body_q.numpy().copy()
            oldtip = x._previous_tip.copy()
            for label, dz, previous_dz, punctured in (
                ("clearance", 0.001, 0.0, True),
                ("entry", -0.0004, 0.0, True),
                ("swept_crossing", 0.0, -0.0004, True),
                ("virgin_reuse", 0.0, 0.0, False),
            ):
                with self.subTest(label=label):
                    moved = pose.copy()
                    moved[c.needle_body, 2] += dz
                    ex.state_0.body_q.assign(moved)
                    x._previous_tip = oldtip + np.array([0.0, 0.0, previous_dz])
                    x.punctured = punctured
                    helper.prepare(x)
                    helper.evaluate(ex.state_0, c.needle_body)
                    self.assertEqual(int(helper.fallback.numpy()[0]), 1)
            ex.state_0.body_q.assign(pose)

    def test_cuda_capture_retains_fresh_opening_buffers(self):
        if not wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda:
            self.skipTest("CUDA capture")
        with contact() as (ex, c, _, _):
            x = ex.interaction
            helper = _NeedleTrialForces(x)
            helper.evaluate(ex.state_0, c.needle_body)
            with ScopedCapture(device=x.device) as capture:
                helper.evaluate(ex.state_0, c.needle_body)
            original = x._opening
            for value, expected in ((original, 0), (None, 1), (original, 0)):
                x._opening = value
                helper.prepare(x)
                wp.capture_launch(capture.graph)
                self.assertEqual(int(helper.fallback.numpy()[0]), expected)

    def test_cuda_native_endpoint_and_late_rejection_preserve_history(self):
        if not wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda:
            self.skipTest("CUDA captured physical solve")
        results = []
        for captured in (False, True):
            with contact() as (ex, c, native, _):
                c.capture_fem = captured
                c.capture_coupling = captured
                before = _snapshot(ex.state_0)
                history = repr(ex.interaction.events)
                if captured:
                    with patch.object(
                        ex.interaction, "check_step_resolution", side_effect=NeedleStepTooLarge("late guard")
                    ):
                        with self.assertRaisesRegex(NeedleStepTooLarge, "late guard"):
                            c.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
                    self.assertIsNone(ex.interaction._trial_fields)
                    self.assertEqual(history, repr(ex.interaction.events))
                    self.assertTrue(ex.interaction.punctured)
                    self.assertAlmostEqual(native.mj_data.time, 3.25, places=12)
                    for name, value in before.items():
                        np.testing.assert_array_equal(getattr(ex.state_0, name).numpy(), value)
                c.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
                self.assertAlmostEqual(native.mj_data.time, 3.25 + ex.sim_dt, places=12)
                self.assertTrue(ex.interaction.punctured)
                self.assertFalse(ex.interaction._in_material)
                if captured:
                    self.assertGreater(c._host_trial_forces.launches, 0)
                    self.assertEqual(c._host_trial_forces.fallbacks, 0)
                results.append((_snapshot(ex.state_1), repr(ex.interaction.events), ex.interaction.status))
        self.assertEqual(results[0][1:], results[1][1:])
        for name, value in results[0][0].items():
            np.testing.assert_allclose(results[1][0][name], value, rtol=3e-6, atol=5e-8, err_msg=name)


if __name__ == "__main__":
    unittest.main()
