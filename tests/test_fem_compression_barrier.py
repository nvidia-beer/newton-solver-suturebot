# SPDX-License-Identifier: Apache-2.0
"""Compression-barrier constitutive checks for the tet and hex force kernels.

The volumetric response must measure the RAW volume ratio det(F) and gain the
C² barrier ψ_b'(J) below ``J_BARRIER``: with the principal-stretch clamp
enabled (the needle example uses ``min_stretch=0.2``), the clamped det(F)
freezes near ``min_stretch³``·(lateral stretches), which historically
saturated compression resistance and let a finite drive force push elements
through inversion (region-1 needle lowering inverted an element at a −2.63 mm
target; see ``nsys/results/lift_range_5mm_20261111/``).
"""

import os
import unittest

import numpy as np
import warp as wp

from newton_solver_suturebot._src.solvers.fem import kernels

MU = 8.6e3
LAM = 77.6e3
ALPHA = 1.0 + MU / LAM - MU / (4.0 * LAM)


def volumetric_scalar(J):
    """dψ_vol/dJ per unit reference volume: stable Neo-Hookean + barrier."""
    J = max(J, kernels.J_MIN)
    value = LAM * (J - ALPHA)
    if J < kernels.J_BARRIER:
        gap = kernels.J_BARRIER - J
        value -= 0.5 * LAM * gap * gap * (2.0 * J + kernels.J_BARRIER) / (J * J)
    return value


def clamped_diagonal(h, min_stretch, max_stretch):
    if min_stretch < 0.0 or max_stretch < 0.0:
        return h
    return min(max(h, min_stretch), max_stretch)


class CompressionBarrierTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))

    def tet_apex_force(self, h, min_stretch=-1.0, max_stretch=-1.0):
        """z-force on the apex of a unit right tet squashed to F = diag(1, 1, h)."""
        x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, h]], dtype=float)
        xq = wp.array(x, dtype=wp.vec3, device=self.device)
        vq = wp.zeros(4, dtype=wp.vec3, device=self.device)
        f = wp.zeros(4, dtype=wp.vec3, device=self.device)
        indices = wp.array(np.arange(4, dtype=np.int32).reshape(1, 4), dtype=wp.int32, device=self.device)
        pose = wp.array(np.eye(3, dtype=np.float32).reshape(1, 3, 3), dtype=wp.mat33, device=self.device)
        materials = wp.array(np.array([[MU, LAM, 0.0]], dtype=np.float32), dtype=wp.float32, device=self.device)
        wp.launch(
            kernels.eval_tetrahedra,
            1,
            [xq, vq, indices, pose, materials, min_stretch, max_stretch, f],
            device=self.device,
        )
        return float(f.numpy()[3, 2])

    def tet_apex_expected(self, h, min_stretch=-1.0, max_stretch=-1.0):
        """Kernel formula mirrored in numpy: clamped F deviatoric, raw-J volumetric."""
        hc = clamped_diagonal(h, min_stretch, max_stretch)
        F = np.diag([1.0, 1.0, hc])
        Ic = np.trace(F.T @ F)
        P = F * (MU / 6.0) * (1.0 - 1.0 / (Ic + 1.0))  # k_mu folds rest volume 1/6
        # dJdx3 = cross(x10, x20) * inv_rest_volume / 6 = e_z for this tet.
        return -(P[2, 2] + volumetric_scalar(h) / 6.0)

    def hex_bottom_force(self, h, min_stretch=-1.0, max_stretch=-1.0):
        """Total z-force on the bottom face of a unit cube squashed to height h."""
        corners = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
            dtype=float,
        )
        x = corners.copy()
        x[:, 2] *= h
        xq = wp.array(x, dtype=wp.vec3, device=self.device)
        vq = wp.zeros(8, dtype=wp.vec3, device=self.device)
        f = wp.zeros(8, dtype=wp.vec3, device=self.device)
        indices = wp.array(np.arange(8, dtype=np.int32).reshape(1, 8), dtype=wp.int32, device=self.device)
        inv = wp.array(np.tile(2.0 * np.eye(3, dtype=np.float32), (1, 8, 1, 1)), dtype=wp.mat33f, device=self.device)
        det = wp.array(np.full((1, 8), 0.125, dtype=np.float32), dtype=wp.float32, device=self.device)
        materials = wp.array(np.array([[MU, LAM, 0.0]], dtype=np.float32), dtype=wp.float32, device=self.device)
        wp.launch(
            kernels.eval_hexahedra,
            1,
            [xq, vq, indices, inv, det, materials, min_stretch, max_stretch],
            [f],
            device=self.device,
        )
        return float(f.numpy()[:4, 2].sum())

    def hex_bottom_expected(self, h, min_stretch=-1.0, max_stretch=-1.0):
        """Constant-F state: f_bottom_z = +P_zz (∑_bottom ∫∂N/∂z dV = −1)."""
        hc = clamped_diagonal(h, min_stretch, max_stretch)
        F = np.diag([1.0, 1.0, hc])
        Ic = np.trace(F.T @ F)
        P = F * MU * (1.0 - 1.0 / (Ic + 1.0))
        P = P + np.diag([hc, hc, 1.0]) * volumetric_scalar(h)  # cof(F_clamped)
        return float(P[2, 2])

    def test_tet_force_matches_formula_across_the_barrier(self):
        for h in (1.0, 0.5, 0.25, 0.19, 0.1, 0.05, 0.02, 0.005):
            with self.subTest(h=h):
                measured = self.tet_apex_force(h)
                expected = self.tet_apex_expected(h)
                self.assertAlmostEqual(measured, expected, delta=2e-4 * max(1.0, abs(expected)))

    def test_hex_force_matches_formula_across_the_barrier(self):
        for h in (1.0, 0.5, 0.25, 0.19, 0.1, 0.05, 0.02, 0.005):
            with self.subTest(h=h):
                measured = self.hex_bottom_force(h)
                expected = self.hex_bottom_expected(h)
                self.assertAlmostEqual(measured, expected, delta=2e-4 * max(1.0, abs(expected)))

    def test_stretch_clamp_cannot_saturate_compression_resistance(self):
        # The needle example's configuration. Below min_stretch the clamped F
        # freezes, so the pre-fix response was constant with h; the raw-J
        # volumetric term must keep the resistance growing toward divergence.
        # The pre-fix response was CONSTANT below min_stretch (ratio 1.0 per
        # step); the barrier must keep growing (>1.25× per halving) and
        # accelerate toward divergence (>8× over the sweep).
        forces = []
        for h in (0.19, 0.1, 0.05, 0.02):
            tet = self.tet_apex_force(h, min_stretch=0.2, max_stretch=5.0)
            hexa = self.hex_bottom_force(h, min_stretch=0.2, max_stretch=5.0)
            self.assertAlmostEqual(tet, self.tet_apex_expected(h, 0.2, 5.0), delta=2e-4 * abs(tet))
            self.assertAlmostEqual(hexa, self.hex_bottom_expected(h, 0.2, 5.0), delta=2e-4 * abs(hexa))
            forces.append((tet, hexa))
        for component in (0, 1):
            # The tet apex is pushed +z and the hex bottom face −z (both away
            # from the crushed element); compare restoring magnitudes.
            values = [abs(force[component]) for force in forces]
            for shallow, deep in zip(values, values[1:], strict=False):
                self.assertGreater(deep, 1.25 * shallow)
            self.assertGreater(values[-1], 8.0 * values[0])

    def test_barrier_is_inactive_at_and_above_the_threshold(self):
        for h in (kernels.J_BARRIER, 0.3, 1.0, 2.0):
            self.assertEqual(volumetric_scalar(h), LAM * (h - ALPHA))


if __name__ == "__main__":
    unittest.main()
