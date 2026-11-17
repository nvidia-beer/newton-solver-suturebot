# SPDX-License-Identifier: Apache-2.0
"""Virgin contact certificates must match exact forces and preserve events."""

import unittest

import numpy as np
import test_needle_trial_commit as fixtures

from newton_solver_suturebot._src.solvers.coupled.interface.needle_trial_forces import _NeedleTrialForces


class VirginContactGraphTests(unittest.TestCase):
    def build(self):
        fixture = fixtures.NeedleTrialCommitTests()
        fixture.setUp()
        x, state, force = fixture.build()
        state.body_q.assign([[0.002, 0.002, -0.0007, 0, 0, 0, 1]])
        x.add_forces(state, 0, np.zeros(3), force, 0.0005, implicit=True)
        x.accept(state, 0)
        self.assertFalse(x._openings)
        return x, state, force

    def test_subthreshold_contact_matches_exact_force_and_staged_history(self):
        for velocity in (0.0, 0.001):
            with self.subTest(velocity=velocity):
                reference, candidate = self.build(), self.build()
                for use_device, (x, state, force) in zip((False, True), (reference, candidate), strict=True):
                    state.body_q.assign([[0.002, 0.002, -0.00049, 0, 0, 0, 1]])
                    state.body_qd.assign([[0, 0, velocity, 0, 0, 0]])
                    state.body_f.zero_()
                    force.zero_()
                    if use_device:
                        device = _NeedleTrialForces(x)
                        device.add_forces(state, 0, np.zeros(3), force, 0.0005)
                        self.assertEqual(int(device.fallback.numpy()[0]), 0)
                        self.assertTrue(x._stage_certified_trial(state, 0, np.zeros(3), device.mask))
                    else:
                        x.add_forces(
                            state,
                            0,
                            np.zeros(3),
                            force,
                            0.0005,
                            implicit=True,
                            trial=True,
                        )
                    self.assertEqual(x._trial_fields["status"], "contact")
                    self.assertIsNone(x._trial_fields["_pending_opening"])
                    self.assertEqual((x.puncture_count, x.entry_count, x.track_count), (0, 0, 0))
                np.testing.assert_allclose(reference[2].numpy(), candidate[2].numpy(), rtol=1e-6, atol=1e-10)
                np.testing.assert_allclose(
                    reference[1].body_f.numpy(),
                    candidate[1].body_f.numpy(),
                    rtol=1e-6,
                    atol=1e-10,
                )
                np.testing.assert_allclose(
                    reference[0]._diagnostics.numpy(),
                    candidate[0]._diagnostics.numpy(),
                    rtol=1e-6,
                    atol=1e-10,
                )

    def test_threshold_crossing_and_prior_openings_require_host_oracle(self):
        for scenario in ("threshold", "crossing", "reuse"):
            with self.subTest(scenario=scenario):
                x, state, force = self.build()
                z = -0.00049
                if scenario == "threshold":
                    z = -0.0004
                elif scenario == "crossing":
                    z = 0.00001
                else:
                    face = next(i for i, ids in enumerate(x.surface_triangles) if set(ids) == {0, 1, 2})
                    ids = x.surface_triangles[face]
                    weights = np.asarray([0.6 if v == 0 else 0.2 for v in ids])
                    x._register_opening(ids, weights, face)
                state.body_q.assign([[0.002, 0.002, z, 0, 0, 0, 1]])
                state.body_f.zero_()
                force.zero_()
                before = (x.puncture_count, x.entry_count, x.track_count)
                device = _NeedleTrialForces(x)
                device.add_forces(state, 0, np.zeros(3), force, 0.0005)
                self.assertEqual(int(device.fallback.numpy()[0]), 1)
                self.assertEqual((x.puncture_count, x.entry_count, x.track_count), before)

    def test_threshold_change_invalidates_capture_signature(self):
        x, _, _ = self.build()
        device = _NeedleTrialForces(x)
        previous = device.signature()
        x.puncture_force *= 2
        self.assertNotEqual(previous, device.signature())

    def test_contact_far_from_existing_opening_matches_exact_force(self):
        reference, candidate = self.build(), self.build()
        for use_device, (x, state, force) in zip((False, True), (reference, candidate), strict=True):
            face = next(i for i, ids in enumerate(x.surface_triangles) if set(ids) == {0, 1, 2})
            ids = x.surface_triangles[face]
            # The old opening is at (8, 1, 0) mm; the current contact is (2, 2, 0) mm.
            weights = np.asarray([{0: 0.1, 1: 0.8, 2: 0.1}[v] for v in ids])
            x._register_opening(ids, weights, face)
            state.body_q.assign([[0.002, 0.002, -0.00049, 0, 0, 0, 1]])
            state.body_f.zero_()
            force.zero_()
            if use_device:
                device = _NeedleTrialForces(x)
                device.add_forces(state, 0, np.zeros(3), force, 0.0005)
                self.assertEqual(int(device.fallback.numpy()[0]), 0)
                self.assertTrue(x._stage_certified_trial(state, 0, np.zeros(3), device.mask))
            else:
                x.add_forces(state, 0, np.zeros(3), force, 0.0005, implicit=True, trial=True)
            self.assertIsNone(x._trial_fields["_pending_opening"])
            self.assertEqual(x._trial_fields["status"], "contact")
            self.assertEqual((x.entry_count, x.track_count, len(x._openings)), (0, 0, 1))
        np.testing.assert_allclose(reference[2].numpy(), candidate[2].numpy(), rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(reference[1].body_f.numpy(), candidate[1].body_f.numpy(), rtol=1e-6, atol=1e-10)

    def test_reuse_certificate_tracks_deformed_opening_and_rejects_uncertainty(self):
        import warp as wp

        from newton_solver_suturebot._src.solvers.coupled.interface.needle_trial_forces import (
            FALLBACK_REASONS,
            opening_reuse_certificate,
        )

        x, state, _ = self.build()
        nodes = wp.array([[0, 1, 2]], dtype=wp.vec3i, device=x.device)
        bary = wp.array([[1, 0, 0]], dtype=wp.vec3d, device=x.device)
        live = wp.array([0, 0, 1], dtype=int, device=x.device)  # one opening
        samples = wp.array([[0, 0, 0]], dtype=wp.vec3d, device=x.device)
        flags = wp.zeros(3, dtype=int, device=x.device)
        fallback = wp.zeros(len(FALLBACK_REASONS), dtype=int, device=x.device)
        capture = 2 * float(x.radii.max())
        for distance, expected in ((0.01, 0), (0.5 * capture, 1), (capture, 1), (float("nan"), 1)):
            with self.subTest(distance=distance):
                positions = state.particle_q.numpy().copy()
                positions[0] = [distance, 0, 0]
                state.particle_q.assign(positions)
                fallback.zero_()
                wp.launch(
                    opening_reuse_certificate,
                    1,
                    [
                        nodes,
                        bary,
                        live,
                        state.particle_q,
                        samples,
                        state.body_q,
                        0,
                        wp.vec3d(0),
                        flags,
                        capture,
                        fallback,
                    ],
                    device=x.device,
                )
                self.assertEqual(int(fallback.numpy()[0]), expected)


if __name__ == "__main__":
    unittest.main()
