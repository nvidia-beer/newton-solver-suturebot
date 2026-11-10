# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""The ALM accelerator differentiates the proximal map without updating duals."""

import os
import unittest

import numpy as np
import warp as wp

from newton_solver_suturebot._src.shaft_preconditioner import balanced_correction, prepare_alm_anchors


def _prox(d, normal, penalty, compliance, clearance):
    tangent = np.array([1.0, 0.0, 0.0])
    projection = np.eye(3) - np.outer(tangent, tangent)
    if compliance > 0:
        offset = projection @ d
        distance = np.linalg.norm(offset)
        return np.zeros(3) if distance <= clearance else (distance - clearance) / compliance * offset / distance
    z = projection @ (normal + penalty * d)
    magnitude = np.linalg.norm(z)
    if magnitude <= penalty * clearance:
        return np.zeros(3)
    return (1.0 - penalty * clearance / magnitude) * z / (1.0 + penalty * compliance)


class NeedleALMPreconditionerTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
        self.nodes = wp.array([[0, 1, 2, 3]], dtype=wp.vec4i, device=self.device)
        self.bary = wp.array([[0.1, 0.2, 0.3, 0.4]], dtype=wp.vec4, device=self.device)
        self.q = wp.zeros(4, dtype=wp.vec3, device=self.device)
        self.qd = wp.zeros_like(self.q)
        self.pose = wp.array([wp.transform_identity()], dtype=wp.transform, device=self.device)
        self.matrices = wp.zeros(1, dtype=wp.mat33, device=self.device)
        self.levers = wp.zeros(1, dtype=wp.vec3, device=self.device)

    def prepare(
        self,
        d=(0, 0.0001, 0.00004),
        normal=(0, 0.2, 0.1),
        tangential=0.0,
        slip=0.0,
        length=0.00025,
        engaged=1,
        h=0.001,
        penalty=60000.0,
        compliance=1 / 60000.0,
        clearance=0.00002,
        friction=0.2,
    ):
        samples = wp.array([[-0.01, d[1], d[2]], [0.01, d[1], d[2]]], dtype=wp.vec3, device=self.device)
        normal_array = wp.array([normal], dtype=wp.vec3, device=self.device)
        tangent_array = wp.array([tangential], dtype=float, device=self.device)
        lengths = wp.array([length], dtype=float, device=self.device)
        engagement = wp.array([engaged], dtype=int, device=self.device)
        velocity = wp.array([[slip, 0, 0, 0, 0, 0]], dtype=wp.spatial_vector, device=self.device)
        before = (normal_array.numpy().copy(), tangent_array.numpy().copy(), engagement.numpy().copy())
        wp.launch(
            prepare_alm_anchors,
            1,
            [
                self.nodes,
                self.bary,
                lengths,
                engagement,
                self.q,
                self.qd,
                samples,
                self.pose,
                velocity,
                0,
                wp.vec3(0.0),
                0.00028,
                normal_array,
                tangent_array,
                penalty,
                compliance,
                clearance,
                friction,
                h,
                0.01,
                self.matrices,
                self.levers,
            ],
            device=self.device,
        )
        for expected, actual in zip(
            before, [normal_array.numpy(), tangent_array.numpy(), engagement.numpy()], strict=True
        ):
            np.testing.assert_array_equal(expected, actual)
        return self.matrices.numpy()[0].astype(float)

    def test_normal_derivative_matches_independent_proximal_finite_difference(self):
        d = np.array([0, 0.0001, 0.00004])
        normal = np.array([8.0, 0.2, 0.1])
        for compliance in [0.0, 1 / 60000.0, 4 / 60000.0]:
            matrix = self.prepare(d=d, normal=normal, compliance=compliance, friction=0)
            columns = []
            for axis in np.eye(3):
                delta = axis * 1e-9
                columns.append(
                    (
                        _prox(d + delta, normal, 60000, compliance, 2e-5)
                        - _prox(d - delta, normal, 60000, compliance, 2e-5)
                    )
                    / 2e-9
                )
            derivative = np.array(columns).T
            np.testing.assert_allclose(matrix, 0.00025 * 0.001 * derivative, rtol=3e-6, atol=1e-9)
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)
            self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-10)

    def test_clearance_and_disengaged_or_zero_measure_have_no_contact_tangent(self):
        for kwargs in [dict(d=(0, 1e-5, 0), normal=(0, 0, 0)), dict(engaged=0), dict(length=0)]:
            np.testing.assert_array_equal(self.prepare(**kwargs), np.zeros((3, 3)))

    def test_sticking_and_sliding_tangents_use_current_multiplier_bound(self):
        normal_only = self.prepare(friction=0)
        sticking = self.prepare(slip=1e-5)
        self.assertAlmostEqual(sticking[0, 0], 0.00025 * 60000 * 0.001, places=8)
        np.testing.assert_allclose(sticking[1:, 1:], normal_only[1:, 1:], atol=1e-10)
        np.testing.assert_allclose(self.prepare(tangential=100.0), normal_only, atol=1e-10)
        np.testing.assert_allclose(self.prepare(slip=-1.0), normal_only, atol=1e-10)
        # A pre-existing normal dual can engage a contact whose displacement is
        # inside the clearance; its actual proximal normal load sets friction.
        inside = self.prepare(d=(0, 1e-5, 0), normal=(0, 3.0, 0), compliance=0)
        np.testing.assert_array_equal(self.prepare(d=(0, 1e-5, 0), normal=(0, 3.0, 0)), 0)
        self.assertGreater(inside[0, 0], 0)
        np.testing.assert_allclose(self.prepare(length=0.0005), 2 * self.prepare(), rtol=1e-6, atol=1e-10)

    def test_fixed_support_correction_preserves_force_and_moment(self):
        self.prepare()
        # The anchor remains at the origin; the rigid COM is offset. The same
        # physical application point must be used in both transpose scatters.
        center = np.array([0.003, -0.002, 0.001])
        self.levers.assign(np.array([-center / 0.01], dtype=np.float32))
        fixed = wp.array([1, 0, 0, 0], dtype=int, device=self.device)
        velocity = wp.array(np.random.default_rng(91).normal(0, 0.1, (6, 3)), dtype=wp.vec3, device=self.device)
        output = wp.zeros(18, dtype=wp.float64, device=self.device)
        h = wp.array([0.001], dtype=float, device=self.device)
        wp.launch(
            balanced_correction,
            1,
            [fixed, self.nodes, self.bary, self.matrices, self.levers, h, velocity, output],
            device=self.device,
        )
        load = output.numpy().reshape(-1, 3)
        self.assertGreater(np.linalg.norm(load[0]), 0)
        np.testing.assert_allclose(load[:4].sum(axis=0) + load[4], 0, atol=2e-11)
        np.testing.assert_allclose(np.cross(center, load[4]) + 0.01 * load[5], 0, atol=2e-12)


if __name__ == "__main__":
    unittest.main()
