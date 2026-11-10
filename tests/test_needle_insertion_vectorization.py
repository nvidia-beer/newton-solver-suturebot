# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Parity against the original scalar material-permission query, including guards."""

import os
import unittest
from itertools import permutations
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleTissueInteraction, boundary_faces
from newton_solver_suturebot._src.insertion import _surface_crossings


# Keep the original sample-by-sample contraction and norms as the reference.
# Boundary tests must catch changes in rounding, anchor order, or permissions.
def _scalar_visited_sample_mask(self, state, body, q):
    mask = np.zeros(len(self.samples), dtype=np.int32)
    self._tip_exit_point = np.zeros(3)
    samples = self._body_samples(state, body)
    if not self._visited_tets:
        mask[0] = self._tip_boundary_permission(samples[0], q)
        return mask
    tets = self._tet_geometry[np.asarray(sorted(self._visited_tets))]
    matrices = np.transpose(tets[:, 1:] - tets[:, :1], (0, 2, 1))
    inverse = np.linalg.inv(matrices)
    low, high = tets.min(axis=1), tets.max(axis=1)
    for i, sample in enumerate(samples):
        candidate = np.flatnonzero(np.all((sample >= low - 1e-8) & (sample <= high + 1e-8), axis=1))
        if len(candidate):
            uvw = np.einsum("ijk,ik->ij", inverse[candidate], sample - tets[candidate, 0])
            mask[i] = np.any((uvw.min(axis=1) >= -1e-5) & (uvw.sum(axis=1) <= 1 + 1e-5))
    # Lateral shaft motion must not encounter an artificial wall at an
    # internal tetrahedral face either. Extend permission only through a
    # continuous, local material segment from an actual traversed anchor.
    # A nearby disconnected wall or lip has intervening exterior crossings
    # and receives ordinary surface contact even within the capture radius.
    track = np.einsum("ij,ijk->ik", np.asarray(self._track_bary_host), q[np.asarray(self._track_nodes_host)])
    capture = 2 * float(self.radii.max())
    directions = np.diff(samples, axis=0)
    lengths = np.diff(self._sample_arc)
    fractions = np.einsum("ijk,jk->ij", track[:, None] - samples[None, :-1], directions) / lengths**2
    projections = samples[None, :-1] + np.clip(fractions, 0, 1)[..., None] * directions
    segments = np.argmin(np.sum((projections - track[:, None]) ** 2, axis=2), axis=1)
    fractions = fractions[np.arange(self.track_count), segments]
    arc = self._sample_arc[segments] + np.clip(fractions, 0, 1) * lengths[segments]
    covered = ~(((segments == 0) & (fractions < 0)) | ((segments == len(lengths) - 1) & (fractions > 1)))
    engaged = (self._track_engaged.numpy() != 0) & covered
    for i in range(1, len(samples)):
        distances = np.linalg.norm(track - samples[i], axis=1)
        near = distances <= capture
        if mask[i] and np.any(near):
            continue
        # Once engaged, the material anchor follows its finite-shaft arc
        # neighborhood during lateral motion. Its transverse spring keeps
        # resisting even beyond the radius for acquiring a new engagement.
        associated = engaged & (np.abs(arc - self._sample_arc[i]) <= capture)
        anchors = np.flatnonzero(near | associated)
        if not len(anchors):
            continue
        embedding = self._locate(samples[i], q)
        if embedding is None:
            continue
        ids, bary = embedding
        if np.any(self.fixed[ids] & (bary > 0.05)):
            continue
        for anchor in anchors[np.argsort(distances[anchors])]:
            if not _surface_crossings(
                track[anchor],
                samples[i],
                q,
                q,
                self._active_triangles,
                bounds=self._active_bounds,
            ):
                mask[i] = 4 if associated[anchor] else 1
                break
    if mask[0] == 0 or self.track_spacing > capture or not self._in_material:
        mask[0] = self._tip_boundary_permission(samples[0], q)
    return mask


def _block(xmin, xmax):
    vertices = np.array([(x, y, z) for x in (xmin, xmax) for y in (-0.001, 0.001) for z in (-0.001, 0)])
    tets = []
    for first, second, _ in permutations(range(3)):
        tet = np.array([0, 1 << first, (1 << first) | (1 << second), 7])
        if np.linalg.det((vertices[tet[1:]] - vertices[tet[:1]]).T) < 0:
            tet[1], tet[2] = tet[2], tet[1]
        tets.append(tet)
    return vertices, np.asarray(tets, dtype=np.int32)


class NeedleVectorizationParityTests(unittest.TestCase):
    def compare(self, interaction, state, q):
        reference = _scalar_visited_sample_mask(interaction, state, 0, q)
        exit_point = interaction._tip_exit_point.copy()
        candidate = NeedleTissueInteraction._visited_sample_mask(interaction, state, 0, q)
        np.testing.assert_array_equal(candidate, reference)
        np.testing.assert_array_equal(interaction._tip_exit_point, exit_point)
        return candidate

    def test_random_and_boundary_membership_matches_scalar_contractions(self):
        rng = np.random.default_rng(481)
        q = 0.001 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
        q += [0.03, -0.04, 0.001]
        tets = np.array([[0, 1, 2, 3], [4, 1, 3, 2]])
        samples = [q[:4].mean(axis=0)]
        samples.extend(q[0] + 0.001 * rng.uniform(-0.2, 1.2, (160, 3)))
        # Shared face, exact vertices, and both sides of the accepted barycentric
        # threshold exercise ambiguous tetrahedral membership and comparison ties.
        samples.extend(q)
        for boundary in (-1e-5, 0.0, 1e-5):
            for value in (boundary - 1e-10, boundary, boundary + 1e-10):
                samples.append(q[0] + 0.001 * np.array([value, 0.25, 0.25]))
                samples.append(q[0] + 0.001 * rng.uniform(-0.2, 1.2, 3))
                samples.append(q[0] + 0.001 * np.array([0.25, 0.25, 0.5 + value]))
                samples.append(q[0] + 0.001 * rng.uniform(-0.2, 1.2, 3))
        samples = np.asarray(samples)
        calls = []
        fixture = SimpleNamespace(
            samples=samples,
            tetrahedra=tets,
            _visited_tets={1, 0},
            _tet_geometry=q[tets],
            _body_samples=lambda *_: samples,
            _track_bary_host=[np.full(4, 0.25), np.full(4, 0.25)],
            _track_nodes_host=list(tets),
            radii=np.full(len(samples), 0.00028),
            _sample_arc=np.r_[0, np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1))],
            track_count=2,
            _track_engaged=SimpleNamespace(numpy=lambda: np.array([0, 1])),
            _locate=lambda point, _: calls.append(tuple(point)),
            _tip_boundary_permission=lambda *_: 0,
            track_spacing=0.00025,
            _in_material=True,
        )
        reference = _scalar_visited_sample_mask(fixture, None, 0, q)
        reference_calls = calls.copy()
        calls.clear()
        candidate = NeedleTissueInteraction._visited_sample_mask(fixture, None, 0, q)
        np.testing.assert_array_equal(candidate, reference)
        self.assertEqual(calls, reference_calls)
        self.assertGreater(np.count_nonzero(candidate), 10)
        self.assertGreater(np.count_nonzero(candidate == 0), 10)
        shifted = samples + [1, 1, 1]
        fixture.samples = shifted
        fixture._body_samples = lambda *_: shifted
        # A nonempty visited set can still yield zero broad-phase candidates.
        np.testing.assert_array_equal(self.compare(fixture, None, q), 0)

    def test_batched_distance_norms_preserve_capture_comparisons_and_anchor_order(self):
        rng = np.random.default_rng(482)
        track = rng.normal(size=(26, 3)) * 0.001 + [0.03, -0.04, 0.001]
        track[0] = 0
        samples = rng.normal(size=(193, 3)) * 0.001 + [0.03, -0.04, 0.001]
        capture = 0.00056
        for i, distance in enumerate((np.nextafter(capture, 0), capture, np.nextafter(capture, np.inf))):
            samples[i] = track[0] + [distance, 0, 0]
        track[1] = track[0]  # Equal distances retain the existing argsort order.
        scalar = np.array([np.linalg.norm(track - sample, axis=1) for sample in samples])
        batched = np.linalg.norm(track[None] - samples[:, None], axis=2)
        np.testing.assert_array_equal(batched, scalar)
        np.testing.assert_array_equal(batched <= capture, scalar <= capture)
        np.testing.assert_array_equal(batched[:3, 0] <= capture, [True, True, False])
        np.testing.assert_array_equal(np.argsort(batched, axis=1), np.argsort(scalar, axis=1))

    def test_empty_visited_history_retains_provisional_tip_permission_only(self):
        samples = np.array([[0, 0, 0], [0, 0, 0.001]])
        for permission in (0, 2, -8):
            with self.subTest(permission=permission):
                fixture = SimpleNamespace(
                    samples=samples,
                    _visited_tets=set(),
                    _body_samples=lambda *_: samples,
                    _tip_boundary_permission=lambda *_, value=permission: value,
                )
                actual = self.compare(fixture, None, np.empty((0, 3)))
                np.testing.assert_array_equal(actual, [permission, 0])

    def test_actual_track_lateral_motion_withdrawal_and_disconnected_wall(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        first, tets = _block(-0.001, 0.001)
        second, other_tets = _block(0.0012, 0.0032)
        vertices = np.vstack([first, second])
        tets = np.vstack([tets, other_tets + len(first)])
        samples = np.array([[0, 0, 0], [0.00035, 0, 0.0001], [0.0006, 0, 0.0002], [0.00085, 0, 0.0003]])
        interaction = NeedleTissueInteraction(
            vertices,
            tets,
            boundary_faces(tets)[0],
            [],
            samples,
            0.0004,
            samples[0],
            device=device,
            stiffness=20,
            damping=0,
            friction=0,
            puncture_force=0.003,
        )
        state = SimpleNamespace(
            particle_q=wp.array(vertices, dtype=wp.vec3, device=device),
            particle_qd=wp.zeros(len(vertices), dtype=wp.vec3, device=device),
            body_q=wp.array([[0.0009, 0, 0.0001, 0, 0, 0, 1]], dtype=wp.transform, device=device),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=device),
        )
        forces = wp.zeros(len(vertices), dtype=wp.vec3, device=device)
        interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True)
        interaction.accept(state, 0)
        self.assertEqual(interaction.puncture_count, 1)
        for z in (-0.0001, -0.0003, -0.0005):
            state.body_q.assign(np.array([[0.0009, 0, z, 0, 0, 0, 1]], np.float32))
            interaction.accept(state, 0)
            self.assertNotEqual(interaction.status, "failed", interaction.reason)
        self.assertGreater(interaction.track_count, 1)
        q = state.particle_q.numpy().astype(float)
        history = (interaction.entry_count, interaction.exit_count, interaction.track_count)
        for x, z in ((0.0009, -0.0005), (0.0008, -0.0005), (0.0009, -0.0001)):
            with self.subTest(position=(x, z)):
                state.body_q.assign(np.array([[x, 0, z, 0, 0, 0, 1]], np.float32))
                tip = interaction._body_tip(state, 0)
                interaction._tip_embedding = interaction._locate(tip, q)
                mask = self.compare(interaction, state, q)
                self.assertGreater(mask[0], 0)
                # Shaft samples in the nearby disconnected block cannot inherit
                # permission from anchors across the genuine empty interval.
                np.testing.assert_array_equal(mask[1:], 0)
        self.assertEqual((interaction.entry_count, interaction.exit_count, interaction.track_count), history)
        # Accept the return inside the first layer, then query an outward trial.
        for z in (-0.0003, -0.0001):
            state.body_q.assign(np.array([[0.0009, 0, z, 0, 0, 0, 1]], np.float32))
            interaction.accept(state, 0)
            self.assertNotEqual(interaction.status, "failed", interaction.reason)
        state.body_q.assign(np.array([[0.0009, 0, 0.00005, 0, 0, 0, 1]], np.float32))
        interaction._tip_embedding = interaction._locate(interaction._body_tip(state, 0), q)
        mask = self.compare(interaction, state, q)
        self.assertLess(mask[0], 0)


if __name__ == "__main__":
    unittest.main()
