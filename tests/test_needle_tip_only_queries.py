# SPDX-License-Identifier: Apache-2.0
"""Unused shaft containment queries must not change forces or puncture events."""

import os
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleTissueInteraction, boundary_faces


class TipOnlyQueryGuardTests(unittest.TestCase):
    def test_standard_queries_start_with_only_tip_before_and_after_material_history(self):
        interaction = object.__new__(NeedleTissueInteraction)
        interaction._visited_tets = set()
        tip = np.array([0.001, 0.002, 0.003])
        samples = np.array([tip, tip + [0, 0, -0.01]])
        np.testing.assert_array_equal(interaction._force_query_points(tip, samples), [tip])
        interaction._visited_tets.add(0)
        np.testing.assert_array_equal(interaction._force_query_points(tip, samples), [tip])

    def test_custom_methods_and_subclasses_keep_original_complete_query_batch(self):
        interaction = object.__new__(NeedleTissueInteraction)
        interaction._visited_tets = set()
        tip = np.array([0.001, 0.002, 0.003])
        samples = np.array([tip, tip + [0, 0, -0.01]])
        for name in (
            "_visited_sample_mask",
            "_device_visited_sample_mask",
            "_tip_boundary_permission",
            "_prepare_device_queries",
            "_locate",
        ):
            with (
                self.subTest(method=name),
                patch.object(interaction, name, lambda *args: None),
            ):
                np.testing.assert_array_equal(
                    interaction._force_query_points(tip, samples),
                    np.vstack((tip, samples)),
                )

        class CustomInteraction(NeedleTissueInteraction):
            pass

        custom = object.__new__(CustomInteraction)
        custom._visited_tets = set()
        np.testing.assert_array_equal(custom._force_query_points(tip, samples), np.vstack((tip, samples)))


class TipOnlyQueryParityTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], np.float32)
        self.tets = np.array([[0, 1, 2, 3]], np.int32)

    def build(self, full_queries=False):
        samples = np.zeros((11, 3))
        samples[:, 2] = np.linspace(0, -0.01, len(samples))
        interaction = NeedleTissueInteraction(
            self.vertices,
            self.tets,
            boundary_faces(self.tets)[0],
            [],
            samples,
            0.0005,
            samples[0],
            device=self.device,
        )
        if full_queries:
            interaction._force_query_points = MethodType(
                lambda self, tip, samples: np.vstack((tip, samples)), interaction
            )
        state = SimpleNamespace(
            particle_q=wp.array(self.vertices, dtype=wp.vec3, device=self.device),
            particle_qd=wp.zeros(4, dtype=wp.vec3, device=self.device),
            body_q=wp.array(
                [[0.002, 0.002, -0.0001, 0, 0, 0, 1]],
                dtype=wp.transform,
                device=self.device,
            ),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
        )
        return interaction, state, wp.zeros(4, dtype=wp.vec3, device=self.device)

    def test_tip_embedding_and_empty_history_mask_match_full_queries_under_deformation(
        self,
    ):
        candidate, reference = self.build(), self.build(full_queries=True)
        for offset in (0, 0.0003, 0):
            for tip in (
                [0.002, 0.002, -0.0001],
                [0.002, 0.002, 0],
                [0.002, 0.002, 0.001],
            ):
                results = []
                for interaction, state, _ in (candidate, reference):
                    q = self.vertices.copy()
                    q[3, 2] += offset
                    state.particle_q.assign(q)
                    state.body_q.assign([[*tip, 0, 0, 0, 1]])
                    q = state.particle_q.numpy().astype(float)
                    samples = interaction._body_samples(state, 0)
                    points = interaction._force_query_points(np.asarray(tip), samples)
                    interaction._prepare_device_queries(state, q, points)
                    interaction._tip_embedding = interaction._locate(np.asarray(tip), q)
                    mask = interaction._visited_sample_mask(state, 0, q, samples=samples)
                    results.append((interaction._tip_embedding, mask))
                first, second = results
                if first[0] is None:
                    self.assertIsNone(second[0])
                else:
                    np.testing.assert_array_equal(first[0][0], second[0][0])
                    np.testing.assert_array_equal(first[0][1], second[0][1])
                np.testing.assert_array_equal(first[1], second[1])
                self.assertEqual(len(candidate[0]._device_query_points), 1)
                self.assertEqual(
                    len(reference[0]._device_query_points),
                    len(reference[0].samples) + 1,
                )

    def test_full_force_and_event_path_matches_complete_queries(self):
        if not self.device.is_cuda:
            self.skipTest("Force evaluation uses the device query path on CUDA")
        candidate, reference = self.build(), self.build(full_queries=True)
        path = np.r_[
            -0.0001,
            np.linspace(0, 0.0012, 9),
            np.linspace(0.00105, -0.0012, 16),
            np.linspace(-0.00105, 0.0012, 16),
        ]
        saw_tip_only = saw_history_tip_only = False
        for step, z in enumerate(path):
            for interaction, state, forces in (candidate, reference):
                start = SimpleNamespace(particle_q=wp.clone(state.particle_q), body_q=wp.clone(state.body_q))
                moved = self.vertices.copy()
                moved[3, 2] += 1e-5 * np.sin(step * 0.1)
                state.particle_q.assign(moved)
                state.body_q.assign([[0.002, 0.002, z, 0, 0, 0, 1]])
                state.body_qd.assign([[0, 0, 0.001 if step < 10 or step >= 26 else -0.001, 0, 0, 0]])
                state.body_f.zero_()
                forces.zero_()
                interaction.add_forces(
                    state,
                    0,
                    np.zeros(3),
                    forces,
                    0.0005,
                    implicit=True,
                    start_state=start,
                )
                interaction.accept(state, 0)
                self.assertNotEqual(interaction.status, "failed", interaction.reason)
            a, b = candidate[0], reference[0]
            saw_tip_only |= len(a._device_query_points) == 1
            saw_history_tip_only |= bool(a._visited_tets) and len(a._device_query_points) == 1
            for actual, expected in (
                (candidate[2], reference[2]),
                (candidate[1].body_f, reference[1].body_f),
                (a._diagnostics, b._diagnostics),
            ):
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=2e-10, rtol=2e-6)
            np.testing.assert_array_equal(a._contact_flags.numpy(), b._contact_flags.numpy())
            self.assertEqual(
                (
                    a.status,
                    a.puncture_count,
                    a.entry_count,
                    a.exit_count,
                    a.track_count,
                ),
                (
                    b.status,
                    b.puncture_count,
                    b.entry_count,
                    b.exit_count,
                    b.track_count,
                ),
            )
            self.assertEqual(
                [event["kind"] for event in a.events],
                [event["kind"] for event in b.events],
            )
        self.assertTrue(saw_tip_only)
        self.assertTrue(saw_history_tip_only)
        self.assertEqual(
            (
                candidate[0].puncture_count,
                candidate[0].entry_count,
                candidate[0].exit_count,
            ),
            (1, 2, 1),
        )

    def test_device_mask_fallback_lazily_prepares_full_sample_batch_before_exact_cpu_mask(self):
        if not self.device.is_cuda:
            self.skipTest("Lazy force query batching is CUDA-specific")
        from newton_solver_suturebot._src.needle_trial_forces import _NeedleTrialForces

        candidate, reference = self.build(), self.build(full_queries=True)
        for interaction, state, forces in (candidate, reference):
            for z in np.r_[-0.0001, np.linspace(0, 0.0012, 9)]:
                start = SimpleNamespace(particle_q=wp.clone(state.particle_q), body_q=wp.clone(state.body_q))
                state.body_q.assign([[0.002, 0.002, z, 0, 0, 0, 1]])
                state.body_f.zero_()
                forces.zero_()
                interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True, start_state=start)
                interaction.accept(state, 0)
                self.assertNotEqual(interaction.status, "failed", interaction.reason)
            self.assertTrue(interaction._visited_tets)
            self.assertGreater(interaction.track_count, 0)
        original = _NeedleTrialForces.evaluate

        def fallback(helper, *args, **kwargs):
            original(helper, *args, **kwargs)
            helper.fallback.fill_(1)

        requests = []
        with patch.object(_NeedleTrialForces, "evaluate", fallback):
            for interaction, state, forces in (candidate, reference):
                points = []
                query = interaction._device_queries
                point_candidates = query.point_candidates

                def record_points(values, points=points, point_candidates=point_candidates):
                    points.append(len(values))
                    return point_candidates(values)

                state.body_f.zero_()
                forces.zero_()
                interaction._device_query_points = None
                with patch.object(query, "point_candidates", side_effect=record_points):
                    interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True, trial=True)
                requests.append(points)
        self.assertEqual(requests[0], [1, len(candidate[0].samples) + 1])
        self.assertEqual(requests[1], [len(reference[0].samples) + 1])
        for actual, expected in (
            (candidate[2], reference[2]),
            (candidate[1].body_f, reference[1].body_f),
            (candidate[0]._sample_in_visited_material, reference[0]._sample_in_visited_material),
        ):
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=2e-10, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
