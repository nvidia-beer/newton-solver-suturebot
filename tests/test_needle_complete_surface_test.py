# SPDX-License-Identifier: Apache-2.0
"""The complete host test of the swept tip takes its bounds from the faces that move.

A step whose start is not the committed copy of the accepted tissue gets the complete host
test of the reference surface: the first step after a reset always, since nothing is committed
yet. Without bounds that test gathers both position sets of every face of the reference
surface. Only the faces with an active vertex move; the others keep the bounds they were built
with. The interaction hands the test the same bounds, taken once for the faces that never move
and per step for those that do: the faces the test selects, and its crossings, are the same.
"""

import unittest
from unittest.mock import patch

import numpy as np
import test_needle_shifted_start_crossings as shifted
from test_needle_shifted_start_crossings import CELLS, SPACING

from newton_solver_suturebot._src.solvers.coupled.interface import insertion
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import _surface_crossings, _vertex_bounds


class CompleteSurfaceTestBounds(unittest.TestCase):
    # The slab of the shifted-start tests, with its sheet of inactive material above the top.
    setUp = shifted.ShiftedStartCrossingTests.setUp
    build = shifted.ShiftedStartCrossingTests.build

    def positions(self, interaction, seed, size=0.0002):
        """Start and end positions of the tissue of a step, and the reference surface at each."""
        rng = np.random.default_rng(seed)
        count = len(interaction.vertices)
        old_q = self.rest.astype(float) + rng.uniform(-size, size, (count, 3))
        q = old_q + rng.uniform(-size, size, (count, 3))
        live = interaction._live_surface_vertices
        points = []
        for positions in (old_q, q):
            surface = interaction.surface_vertices.copy()
            surface[live] = positions[interaction.mapping[live]]
            points.append(surface)
        return old_q, q, points[0], points[1]

    def segments(self, seed, count=40):
        """Tip paths through the top of the slab and through the sheet above it, and some that miss both."""
        rng = np.random.default_rng(seed)
        top = float(self.top)
        xy = rng.uniform(0.0, CELLS[0] * SPACING, (count, 2))
        starts = np.c_[xy, np.full(count, top + 0.0008)]
        ends = np.c_[xy + rng.uniform(-SPACING, SPACING, (count, 2)), rng.uniform(top - 0.001, top + 0.0006, count)]
        return starts, ends

    def test_the_bounds_are_those_the_test_takes_for_itself(self):
        interaction, _ = self.build(lid=True)
        triangles = interaction.surface_triangles
        for seed in range(4):
            _, _, old_points, points = self.positions(interaction, seed)
            old_low, old_high = _vertex_bounds(old_points[triangles])
            new_low, new_high = _vertex_bounds(points[triangles])
            low, high = interaction._swept_surface_bounds(old_points, points)
            # Bit for bit, for every face of the reference surface.
            self.assertEqual(low.tobytes(), np.minimum(old_low, new_low).tobytes())
            self.assertEqual(high.tobytes(), np.maximum(old_high, new_high).tobytes())

    def test_a_face_without_an_active_vertex_keeps_the_bounds_it_was_built_with(self):
        interaction, _ = self.build(lid=True)
        faces = interaction._dynamic_surface_faces
        still = np.setdiff1d(np.arange(len(interaction.surface_triangles)), faces)
        # The two triangles of the sheet: no particle moves their vertices.
        self.assertEqual(len(still), 2)
        built = tuple(bound.copy() for bound in interaction._reference_surface_bounds)
        _, _, old_points, points = self.positions(interaction, 7)
        low, high = interaction._swept_surface_bounds(old_points, points)
        self.assertEqual(low[still].tobytes(), built[0][still].tobytes())
        self.assertEqual(high[still].tobytes(), built[1][still].tobytes())
        # The faces that move were taken from the positions, and the stored bounds are not written.
        self.assertFalse(np.array_equal(low[faces], built[0][faces]))
        self.assertEqual(interaction._reference_surface_bounds[0].tobytes(), built[0].tobytes())
        self.assertEqual(interaction._reference_surface_bounds[1].tobytes(), built[1].tobytes())

    def test_the_test_with_these_bounds_returns_the_crossings_of_the_test_without(self):
        interaction, _ = self.build(lid=True)
        triangles = interaction.surface_triangles
        found = 0
        for seed in range(3):
            _, _, old_points, points = self.positions(interaction, seed)
            bounds = interaction._swept_surface_bounds(old_points, points)
            for start, end in zip(*self.segments(seed), strict=True):
                complete = _surface_crossings(start, end, old_points, points, triangles)
                bounded = _surface_crossings(start, end, old_points, points, triangles, bounds=bounds)
                # Whether there is a crossing, its time, its sign and its face.
                self.assertEqual(complete, bounded)
                found += len(complete)
        # The paths cross the moving top and the sheet that never moves.
        self.assertGreater(found, 20)

    def first_step(self, interaction, state, *, bounds):
        """The resolution check of a first step (nothing committed); returns what it kept and the tests it ran."""
        self.assertIsNone(interaction._previous_q)
        calls, test = [], insertion._surface_crossings

        def recorded(start, end, old_points, new_points, triangles, **options):
            calls.append((len(triangles), options.get("bounds") is not None, options.get("candidates") is not None))
            return test(start, end, old_points, new_points, triangles, **options)

        moved = self.rest.copy()
        moved[:, 2] -= np.float32(0.0001)
        candidate = type(state)(
            particle_q=type(state.particle_q)(moved, dtype=state.particle_q.dtype, device=self.device),
            # The tip comes down through the sheet (0.2 mm above the top) and stops above the tissue.
            body_q=type(state.body_q)(
                [[0.0045, 0.0045, float(self.top) + 0.00015, 0, 0, 0, 1]], dtype=state.body_q.dtype, device=self.device
            ),
        )
        state.body_q.assign(np.array([[0.0045, 0.0045, float(self.top) + 0.00025, 0, 0, 0, 1]], dtype=np.float32))
        with patch.object(insertion, "_surface_crossings", recorded):
            if bounds:
                interaction.check_step_resolution(state, candidate, 0)
            else:
                # The test as it was: no bounds for the complete test.
                with patch.object(type(interaction), "_swept_surface_bounds", lambda self, old, new: None):
                    interaction.check_step_resolution(state, candidate, 0)
        return interaction._checked_crossings, calls, dict(interaction.shifted_start_host_tests)

    def test_the_first_step_after_a_reset_gets_the_bounds_and_the_same_crossings(self):
        results = []
        for bounds in (True, False):
            interaction, state = self.build(lid=True)
            results.append(self.first_step(interaction, state, bounds=bounds))
        (kept, calls, reasons), (expected, complete_calls, expected_reasons) = results
        surface = len(interaction.surface_triangles)
        # The test of the reference surface ran with bounds; before, it ran without.
        self.assertIn((surface, True, False), calls)
        self.assertIn((surface, False, False), complete_calls)
        self.assertNotIn((surface, False, False), calls)
        # The same crossings (the tip passes the sheet), and the same count of complete tests by reason.
        self.assertEqual(kept[4], expected[4])
        self.assertEqual(kept[5], expected[5])
        self.assertEqual(len(kept[4]), 1)
        self.assertEqual(reasons, expected_reasons)
        self.assertEqual(sum(reasons.values()), 2)

    # --- added by the check of 2026-11-19: both places that hand the test its bounds, and where none are taken.

    describe = shifted.ShiftedStartCrossingTests.describe
    commit = shifted.ShiftedStartCrossingTests.commit

    def step(self, seed, size=2.0e-5):
        """Start and end positions of the tissue of a step [float32 values], and a tip path through the sheet and the top."""
        rng = np.random.default_rng(seed)
        old_q = self.rest + rng.uniform(-size, size, self.rest.shape).astype(np.float32)
        q = old_q + rng.uniform(-size, size, self.rest.shape).astype(np.float32)
        top = float(self.top)
        return old_q, q, np.array([0.0045, 0.0052, top + 0.0006]), np.array([0.0046, 0.0051, top - 0.0004])

    def reference_tests(self, interaction, run):
        """Run ``run``; returns its result and one entry per test of the reference surface in it.

        An entry is (whether the bounds the test was handed are, bit for bit, those it takes for itself from the
        positions it was handed, or None without bounds; whether it was handed candidates). Every such test is
        repeated without bounds and without candidates: the crossings must be those of the complete test.
        """
        calls, test = [], insertion._surface_crossings

        def recorded(start, end, old_points, new_points, triangles, *, bounds=None, candidates=None):
            found = test(start, end, old_points, new_points, triangles, bounds=bounds, candidates=candidates)
            if triangles is interaction.surface_triangles:
                exact = None
                if bounds is not None:
                    old_low, old_high = _vertex_bounds(old_points[triangles])
                    new_low, new_high = _vertex_bounds(new_points[triangles])
                    exact = (
                        bounds[0].tobytes() == np.minimum(old_low, new_low).tobytes()
                        and bounds[1].tobytes() == np.maximum(old_high, new_high).tobytes()
                    )
                calls.append((exact, candidates is not None))
                self.assertEqual(found, test(start, end, old_points, new_points, triangles))
            return found

        with patch.object(insertion, "_surface_crossings", recorded):
            result = run()
        return result, calls

    def test_without_device_geometry_the_test_gets_the_bounds_of_its_start_and_end_positions(self):
        # The place of a step whose end positions the device query does not describe (_resolution_crossings).
        interaction, _ = self.build(lid=True)
        old_q, q, start, tip = self.step(3)
        old_q, q = old_q.astype(float), q.astype(float)
        self.assertFalse(interaction._device_geometry_matches(q))
        (crossings, active), calls = self.reference_tests(
            interaction, lambda: interaction._resolution_crossings(start, tip, old_q, q)
        )
        # One test of the reference surface, with the bounds of the positions it was handed: the start
        # positions differ from the end positions in every moving face, so bounds of one set alone are not these.
        self.assertEqual(calls, [(True, False)])
        self.assertEqual(interaction.shifted_start_host_tests, {"no device geometry of the end positions": 2})
        # The path passes the sheet, which never moves, and the top, which does.
        self.assertEqual(len(crossings), 2)
        self.assertEqual(len(active), 1)
        self.assertGreaterEqual(max(face for _, _, face in crossings), len(self.exterior))

    def test_with_device_geometry_and_no_committed_copy_the_test_gets_the_same_bounds(self):
        # The place of the first step after a reset on the device path (_material_crossings): the device
        # describes the end positions, and nothing is committed yet.
        interaction, state = self.build(lid=True)
        old_q, end, start, tip = self.step(4)
        q = self.describe(interaction, state, end)
        old_q = old_q.astype(float)
        self.assertTrue(interaction._device_geometry_matches(q))
        self.assertIsNone(interaction._previous_q)
        crossings, calls = self.reference_tests(
            interaction, lambda: interaction._material_crossings(start, tip, old_q, q)
        )
        self.assertEqual(calls, [(True, False)])
        self.assertEqual(interaction.shifted_start_host_tests, {"no committed copy": 1})
        self.assertEqual(len(crossings), 2)
        self.assertGreaterEqual(max(face for _, _, face in crossings), len(self.exterior))

    def test_bounds_are_taken_only_for_the_complete_test_of_the_reference_surface(self):
        taken, swept = [], insertion.NeedleTissueInteraction._swept_surface_bounds

        def counted(interaction, old_points, points):
            taken.append(len(points))
            return swept(interaction, old_points, points)

        with patch.object(insertion.NeedleTissueInteraction, "_swept_surface_bounds", counted):
            # A step from the committed copy: the device gives the candidates, and no bounds are gathered.
            interaction, state = self.build(lid=True)
            committed, end, start, tip = self.step(5)
            self.commit(interaction, state, committed)
            q = self.describe(interaction, state, end)
            old_q = committed.astype(float)
            crossings, calls = self.reference_tests(
                interaction, lambda: interaction._material_crossings(start, tip, old_q, q)
            )
            self.assertEqual(calls, [(None, True)])
            self.assertEqual(len(crossings), 2)
            self.assertEqual(taken, [])
            self.assertEqual(interaction.shifted_start_host_tests, {})
            # The active surface of a step with nothing committed: its complete test takes its own bounds,
            # of the active triangles; the bounds of the reference surface are not for it.
            interaction, state = self.build(lid=True)
            old_q, end, start, tip = self.step(6)
            q = self.describe(interaction, state, end)
            old_q = old_q.astype(float)
            active, calls = self.reference_tests(
                interaction, lambda: interaction._material_crossings(start, tip, old_q, q, active=True)
            )
            self.assertEqual(calls, [])
            self.assertEqual(taken, [])
            self.assertEqual(active, _surface_crossings(start, tip, old_q, q, interaction._active_triangles))
            self.assertEqual(len(active), 1)
            self.assertEqual(interaction.shifted_start_host_tests, {"no committed copy": 1})


if __name__ == "__main__":
    unittest.main()
