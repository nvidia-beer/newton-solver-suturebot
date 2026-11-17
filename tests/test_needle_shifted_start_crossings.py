# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Swept tip tests of a step that does not start from the committed copy of the accepted tissue.

A caller may move nodes of the accepted state after ``accept()`` (a contact that places nodes).
The next step then starts from positions that are not the copy the interaction kept and committed
to its device query. The test still takes the device candidates, for a query box grown by the
measured difference, and must give what the complete host test gives from the actual start
positions, in every field: whether there is a crossing, its triangle, its time and its sign.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleTissueInteraction, boundary_faces
from newton_solver_suturebot._src.geometry.needle_queries import NeedleQueryGeometry
from newton_solver_suturebot._src.solvers.coupled.interface import insertion
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import _surface_crossings

SPACING = 0.001  # edge of a cell of the slab [m]
CELLS = (10, 10, 3)
OVERFLOW = "candidate overflow"
STALE = "device motion not measured from the committed copy"


def _slab():
    """A slab of cubes of edge SPACING, six positively oriented tetrahedra each; float32 vertices."""
    nx, ny, nz = CELLS
    grid = np.stack(np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij"), axis=-1)
    vertices = (grid.reshape(-1, 3) * SPACING).astype(np.float32)

    def node(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                base = np.array([i, j, k])
                for order in ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)):
                    corners = [base.copy()]
                    for axis in order:
                        corners.append(corners[-1].copy())
                        corners[-1][axis] += 1
                    ids = [node(*corner) for corner in corners]
                    # The three even orders give a positive volume; the odd ones are mirrored.
                    if order in ((0, 2, 1), (2, 1, 0), (1, 0, 2)):
                        ids[1], ids[2] = ids[2], ids[1]
                    tets.append(ids)
    return vertices, np.asarray(tets, dtype=np.int32)


class ShiftedStartCrossingTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.rest, self.tets = _slab()
        self.exterior = boundary_faces(self.tets)[0]
        self.top = np.float32(CELLS[2] * SPACING)

    def build(self, *, lid=False, capacity=None):
        """An interaction on the slab and a state at rest; the needle is far from the tissue.

        ``lid``: the full surface is a reference surface with more vertices than the tissue has
        particles: a sheet of inactive material 0.2 mm above a part of the top (vertices that no
        particle moves). ``capacity``: candidate slots of the device query, instead of its default.
        """
        samples = np.zeros((3, 3))
        samples[:, 2] = (0, -0.001, -0.002)
        exterior = self.exterior
        options = {}
        if lid:
            corners = np.array([[2, 2], [7, 2], [7, 7], [2, 7]]) * SPACING + 0.0003
            sheet = np.c_[corners, np.full(4, float(self.top) + 0.0002)]
            count = len(self.rest)
            options = dict(
                reference_vertices=np.vstack((self.rest.astype(float), sheet)),
                reference_triangles=np.vstack(
                    (exterior, [[count, count + 1, count + 2], [count, count + 2, count + 3]])
                ),
                reference_to_active=np.r_[np.arange(count), -np.ones(4, int)],
            )
        interaction = NeedleTissueInteraction(
            self.rest, self.tets, exterior, [], samples, 0.0005, samples[0], device=self.device, **options
        )
        if capacity is not None:
            interaction._device_queries = NeedleQueryGeometry(
                interaction.tetrahedra, len(interaction.vertices), device=self.device, capacity=capacity
            )
        state = SimpleNamespace(
            particle_q=wp.array(self.rest, dtype=wp.vec3, device=self.device),
            body_q=wp.array([[0.05, 0.05, 0.05, 0, 0, 0, 1]], dtype=wp.transform, device=self.device),
        )
        return interaction, state

    def describe(self, interaction, state, positions, *, measure=True):
        """Bring the device surfaces and the device query to ``positions``, as a force evaluation does.

        The cached certificate of the query is dropped first, as the coupler drops it after device
        work (``_invalidate_query_cache``), so that the query measures its motion again;
        ``measure=False`` keeps the certificate, and a query that describes these very positions
        then keeps the motion it measured before.
        """
        state.particle_q.assign(np.asarray(positions, dtype=np.float32))
        wp.launch(
            insertion._surface_positions,
            len(interaction.mapping),
            [state.particle_q, interaction._mapping, interaction._surface],
            device=self.device,
        )
        interaction.mesh.refit()
        wp.copy(interaction._active_surface, state.particle_q, count=len(interaction.vertices))
        interaction._active_mesh.refit()
        q = state.particle_q.numpy().astype(float)
        if measure:
            interaction._device_query_positions = None
        interaction._prepare_device_queries(state, q, np.array([[0.05, 0.05, 0.05]]))
        return q

    def commit(self, interaction, state, positions):
        """Accept ``positions``: the interaction keeps its copy and commits it to the device query."""
        self.describe(interaction, state, positions)
        interaction.accept(state, 0)
        self.assertNotEqual(interaction.status, "failed", interaction.reason)
        np.testing.assert_array_equal(interaction._previous_q, np.asarray(positions, dtype=np.float32))

    @staticmethod
    def host_test(interaction, start, end, old_q, q, active):
        """The complete host test from the actual start positions: bounds of every triangle of the surface."""
        if active:
            return _surface_crossings(start, end, old_q, q, interaction._active_triangles)
        live = interaction._live_surface_vertices
        points = interaction.surface_vertices.copy()
        points[live] = q[interaction.mapping[live]]
        old_points = points.copy()
        old_points[live] = old_q[interaction.mapping[live]]
        return _surface_crossings(start, end, old_points, points, interaction.surface_triangles)

    def crossings(self, interaction, start, end, old_q, q):
        """Both surfaces through the interaction, each checked against the complete host test."""
        results = []
        for active in (False, True):
            result = interaction._material_crossings(start, end, old_q, q, active=active)
            self.assertEqual(result, self.host_test(interaction, start, end, old_q, q, active))
            results.append(result)
        return results

    def top_triangle(self, cell):
        """Index of a boundary triangle on the top of ``cell`` (i, j), and its three particles.

        The interaction numbers the triangles of its active boundary and, without a reference
        surface, of its full surface in this order.
        """
        i, j = cell
        triangles = self.exterior
        points = self.rest[triangles]
        on_top = np.all(points[:, :, 2] == self.top, axis=1)
        centre = points[:, :, :2].mean(axis=1) / SPACING
        inside = (centre[:, 0] > i) & (centre[:, 0] < i + 1) & (centre[:, 1] > j) & (centre[:, 1] < j + 1)
        index = int(np.flatnonzero(on_top & inside)[0])
        return index, triangles[index]

    def float32(self, positions):
        return np.asarray(positions, dtype=np.float32).astype(float)

    def placed_case(self, depth, *, lift=0.0, cell=(4, 5)):
        """A top triangle whose nodes were placed ``depth`` into the tissue after the accepted step.

        Returns (committed, start, end positions, tip, the triangle). The committed copy has the
        triangle ``lift`` below the top; the step starts with its three nodes ``depth`` lower and
        ends with them on the top. The tip stands still under the triangle's centre, 0.7 ``depth``
        (or, with a lift, half the lift) below the top: only the motion from the actual start
        carries the triangle across it.
        """
        triangle, nodes = self.top_triangle(cell)
        committed = self.rest.copy()
        committed[nodes, 2] -= np.float32(lift)
        start = committed.copy()
        start[nodes, 2] -= np.float32(depth)
        end = self.rest.copy()
        tip = self.rest[nodes].astype(float).mean(axis=0)
        tip[2] -= 0.5 * lift if lift else 0.7 * depth
        return committed, start, end, tip, triangle

    def test_equal_start_takes_the_unchanged_path(self):
        interaction, state = self.build()
        committed = self.rest.copy()
        end = committed.copy()
        end[:, 2] += np.float32(2e-5)
        self.commit(interaction, state, committed)
        q = self.describe(interaction, state, end)
        start, tip = np.array([0.0042, 0.0053, 0.00305]), np.array([0.0043, 0.0052, 0.00295])
        calls = []
        query = interaction._device_queries
        original = query.surface_candidates

        def recorded(mesh, low, high, *, swept=False):
            calls.append((mesh, np.array(low), np.array(high), swept))
            return original(mesh, low, high, swept=swept)

        with patch.object(query, "surface_candidates", recorded):
            # The same values in another array, as check_step_resolution reads them from the state.
            results = self.crossings(interaction, start, tip, self.float32(committed), q)
            self.crossings(interaction, start, tip, q, q)
        self.assertEqual([len(result) for result in results], [1, 1])
        self.assertEqual([call[0] for call in calls], [interaction.mesh, interaction._active_mesh] * 2)
        self.assertEqual([call[3] for call in calls], [True, True, False, False])
        for _, low, high, _ in calls:
            np.testing.assert_array_equal(low, np.minimum(start, tip) - 1e-9)
            np.testing.assert_array_equal(high, np.maximum(start, tip) + 1e-9)
        self.assertEqual(interaction.shifted_start_candidate_tests, 0)
        self.assertEqual(interaction.shifted_start_host_tests, {})
        self.assertEqual(interaction.shifted_start_largest_growth, 0.0)

    def test_generated_cases_match_the_complete_host_test(self):
        # Differences between the start positions and the committed copy: a few micrometres, the
        # 15.7 um measured in the grasp demo, the mesh spacing, three spacings.
        for shift, lid in ((3e-6, False), (1.57e-5, True), (SPACING, False), (SPACING, True), (3 * SPACING, False)):
            with self.subTest(shift=shift, lid=lid):
                interaction, state = self.build(lid=lid)
                rng = np.random.default_rng(int(shift * 1e9) + lid)
                found = tests = sheet = 0
                largest = 0.0
                for case in range(24):
                    committed = self.rest + rng.uniform(-2e-5, 2e-5, self.rest.shape).astype(np.float32)
                    end = committed + rng.uniform(-3e-5, 3e-5, self.rest.shape).astype(np.float32)
                    start = committed.copy()
                    # Placed nodes: a patch of the top around the cell the tip aims at, or every node.
                    cell = rng.integers(1, 9, 2)
                    _, nodes = self.top_triangle(cell)
                    centre = self.rest[nodes].astype(float).mean(axis=0)
                    near = np.linalg.norm(self.rest[:, :2] - centre[:2], axis=1) < rng.uniform(0.5, 2.5) * SPACING
                    moved = near if case % 3 else np.ones(len(self.rest), bool)
                    start[moved] += rng.uniform(-shift, shift, (int(moved.sum()), 3)).astype(np.float32)
                    self.commit(interaction, state, committed)
                    q = self.describe(interaction, state, end)
                    old_q = self.float32(start)
                    largest = max(largest, float(np.abs(old_q - self.float32(committed)).max()))
                    reach = max(shift, 1e-4)
                    for _ in range(4):
                        above = centre + np.r_[rng.uniform(-3e-4, 3e-4, 2), rng.uniform(0.2, 1.2) * reach]
                        below = centre + np.r_[rng.uniform(-3e-4, 3e-4, 2), -rng.uniform(0.2, 1.2) * reach]
                        for a, b in ((above, below), (below, above), (above, above)):
                            results = self.crossings(interaction, a, b, old_q, q)
                            found += sum(bool(result) for result in results)
                            sheet += any(face >= len(self.exterior) for _, _, face in results[0])
                            tests += 2
                taken = interaction.shifted_start_candidate_tests
                self.assertEqual(taken + sum(interaction.shifted_start_host_tests.values()), tests)
                self.assertLessEqual(set(interaction.shifted_start_host_tests), {OVERFLOW})
                self.assertGreater(found, tests // 6)
                # Crossings of the inactive sheet, whose vertices no particle moves, where the tip reaches it.
                self.assertEqual(sheet > 0, lid and shift == SPACING)
                if shift <= SPACING:
                    self.assertGreater(taken, tests // 2)
                self.assertEqual(interaction.shifted_start_largest_growth > 0, taken > 0)
                self.assertLessEqual(interaction.shifted_start_largest_growth, largest)
                self.assertGreater(interaction.shifted_start_largest_growth, 0.5 * shift)

    def test_placed_nodes_on_the_crossed_triangle(self):
        for depth in (3e-6, 1.57e-5, SPACING):
            with self.subTest(depth=depth):
                committed, start, end, tip, triangle = self.placed_case(depth)
                interaction, state = self.build()
                self.commit(interaction, state, committed)
                q = self.describe(interaction, state, end)
                # The tissue ends where the committed copy has it: the device measured no motion.
                np.testing.assert_array_equal(interaction._device_queries.motion.numpy(), 0.0)
                old_q = self.float32(start)
                surface, active = self.crossings(interaction, tip, tip, old_q, q)
                self.assertEqual(interaction.shifted_start_candidate_tests, 2)
                self.assertEqual(interaction.shifted_start_host_tests, {})
                self.assertAlmostEqual(interaction.shifted_start_largest_growth / depth, 1.0, places=3)
                # One crossing, of that triangle, as the surface rises past the tip: the tip ends inside.
                self.assertEqual(len(active), 1)
                time, sign, face = active[0]
                self.assertEqual((sign, face), (-1, triangle))
                self.assertAlmostEqual(time, 0.3, places=3)
                self.assertEqual(len(surface), 1)
                self.assertEqual(surface[0][:2], (time, sign))
                np.testing.assert_array_equal(
                    np.sort(interaction.mapping[interaction.surface_triangles[surface[0][2]]]),
                    np.sort(interaction._active_triangles[triangle]),
                )
                # The point the callers derive from the time lies on the triangle at that time.
                moving = old_q[interaction._active_triangles[triangle]]
                moving = moving + time * (q[interaction._active_triangles[triangle]] - moving)
                self.assertAlmostEqual(float(moving[:, 2].mean()), tip[2], delta=1e-3 * depth)
                # From the committed copy the triangle never moves: the crossing exists only for the actual start.
                self.assertEqual(self.host_test(interaction, tip, tip, self.float32(committed), q, True), [])

    def test_shift_and_motion_of_the_step_add_up(self):
        # The committed triangle lies 40 um under the top, its nodes were placed 5 um lower, and the
        # step carries it to the top: the tip 20 um under the top is reached by the motion and the shift.
        committed, start, end, tip, triangle = self.placed_case(5e-6, lift=4e-5)
        interaction, state = self.build()
        self.commit(interaction, state, committed)
        q = self.describe(interaction, state, end)
        surface, active = self.crossings(interaction, tip, tip, self.float32(start), q)
        self.assertEqual([len(surface), len(active)], [1, 1])
        self.assertEqual(active[0][1:], (-1, triangle))
        self.assertAlmostEqual(active[0][0], 25 / 45, places=3)
        self.assertEqual(interaction.shifted_start_candidate_tests, 2)

    def test_tip_grazing_an_edge_or_touching_the_surface(self):
        interaction, state = self.build()
        _, nodes = self.top_triangle((4, 5))
        committed = self.rest.copy()
        start = committed.copy()
        start[nodes, 2] -= np.float32(1.57e-5)
        end = committed.copy()
        end[nodes, 2] += np.float32(4e-6)
        self.commit(interaction, state, committed)
        q = self.describe(interaction, state, end)
        old_q = self.float32(start)
        # The diagonal of the cell is an edge of two top triangles; a cell corner is a vertex of six.
        corner = np.array([4, 5, 3]) * SPACING
        diagonal = corner + np.array([0.5, 0.5, 0]) * SPACING
        found = 0
        for point in (diagonal, corner, corner + np.array([0.5, 0, 0]) * SPACING):
            for offset in (0.0, 1e-9, -1e-9, 1e-7, -1e-7):
                for lateral in (np.zeros(3), np.array([offset, -offset, 0])):
                    above = point + lateral + [0, 0, 5e-5]
                    below = point + lateral - [0, 0, 5e-5]
                    # Through the edge, ending on the surface, starting on it, and standing on it.
                    on = point + lateral + [0, 0, offset]
                    for a, b in ((above, below), (below, above), (above, on), (on, below), (on, on)):
                        results = self.crossings(interaction, a, b, old_q, q)
                        found += sum(bool(result) for result in results)
        self.assertGreater(found, 60)
        self.assertEqual(interaction.shifted_start_host_tests, {})
        self.assertEqual(interaction.shifted_start_candidate_tests, 3 * 5 * 2 * 5 * 2)

    def test_the_limit_where_the_complete_host_test_takes_over(self):
        committed, start, end, tip, triangle = self.placed_case(SPACING)
        old_q = self.float32(start)

        def run(capacity):
            interaction, state = self.build(capacity=capacity)
            self.commit(interaction, state, committed)
            q = self.describe(interaction, state, end)
            result = interaction._material_crossings(tip, tip, old_q, q, active=True)
            self.assertEqual(result, self.host_test(interaction, tip, tip, old_q, q, True))
            self.assertEqual(result[0][2], triangle)
            return interaction, int(interaction._device_queries.surface_result.numpy()[0])

        # The number of triangles in the grown box, read with slots to spare.
        _, count = run(None)
        self.assertGreater(count, 8)
        # Exactly as many slots as candidates: the packet is complete and is taken.
        full, same = run(count)
        self.assertEqual(same, count)
        self.assertEqual((full.shifted_start_candidate_tests, full.shifted_start_host_tests), (1, {}))
        # One slot fewer: the packet overflowed, and the complete host test gives the same crossing.
        short, same = run(count - 1)
        self.assertEqual(same, count)
        self.assertEqual((short.shifted_start_candidate_tests, short.shifted_start_host_tests), (0, {OVERFLOW: 1}))
        self.assertEqual(short.shifted_start_largest_growth, 0.0)

    def test_candidate_overflow_keeps_the_complete_host_test(self):
        tests = 0
        for capacity in (1, 3):
            for cell in ((1, 1), (4, 5), (8, 2), (2, 8), (6, 6), (5, 3)):
                # A growth of 1.2 spacings: the box holds the triangles of nine cells and more.
                committed, start, end, tip, triangle = self.placed_case(1.2 * SPACING, cell=cell)
                interaction, state = self.build(capacity=capacity)
                self.commit(interaction, state, committed)
                q = self.describe(interaction, state, end)
                surface, active = self.crossings(interaction, tip, tip, self.float32(start), q)
                self.assertEqual((len(surface), len(active)), (1, 1))
                self.assertEqual(active[0][2], triangle)
                self.assertEqual(interaction.shifted_start_candidate_tests, 0)
                self.assertEqual(interaction.shifted_start_host_tests, {OVERFLOW: 2})
                self.assertGreater(int(interaction._device_queries.surface_result.numpy()[0]), capacity)
                tests += 2
        self.assertEqual(tests, 24)

    def test_no_committed_copy_keeps_the_complete_host_test(self):
        _, start, end, tip, triangle = self.placed_case(1.57e-5)
        interaction, state = self.build()
        q = self.describe(interaction, state, end)
        self.assertIsNone(interaction._previous_q)
        surface, active = self.crossings(interaction, tip, tip, self.float32(start), q)
        self.assertEqual((len(surface), active[0][2]), (1, triangle))
        self.assertEqual(interaction.shifted_start_candidate_tests, 0)
        self.assertEqual(interaction.shifted_start_host_tests, {"no committed copy": 2})

    def test_motion_measured_before_the_last_accept_is_not_used(self):
        # The query describes the end positions and measured their motion from an earlier accepted
        # state; then another state is accepted, without the query being brought to it. Its motion
        # (zero) says nothing about the way from the new committed copy to the end positions.
        committed, start, end, tip, triangle = self.placed_case(3e-6, lift=4e-5)
        interaction, state = self.build()
        self.commit(interaction, state, end)
        self.describe(interaction, state, end)
        state.particle_q.assign(committed)
        interaction.accept(state, 0)
        np.testing.assert_array_equal(interaction._previous_q, committed)
        # The end positions are evaluated again; the query describes them already and measures nothing.
        q = self.describe(interaction, state, end, measure=False)
        self.assertTrue(interaction._device_geometry_matches(q))
        np.testing.assert_array_equal(interaction._device_queries.motion.numpy(), 0.0)
        surface, active = self.crossings(interaction, tip, tip, self.float32(start), q)
        self.assertEqual((len(surface), active[0][2]), (1, triangle))
        self.assertEqual(interaction.shifted_start_candidate_tests, 0)
        self.assertEqual(interaction.shifted_start_host_tests, {STALE: 2})
        # Once the query has measured the motion from the committed copy, its candidates are taken.
        q = self.describe(interaction, state, end)
        self.crossings(interaction, tip, tip, self.float32(start), q)
        self.assertEqual(interaction.shifted_start_candidate_tests, 2)

    def test_query_made_after_the_accepted_step_measured_no_motion(self):
        # The accepted state was committed before the interaction had a device query. The query made
        # afterwards holds no committed positions and measures no motion of the end positions.
        committed, start, end, tip, triangle = self.placed_case(5e-6, lift=4e-5)
        interaction, state = self.build()
        state.particle_q.assign(committed)
        interaction.accept(state, 0)
        self.assertIsNone(interaction._device_queries)
        q = self.describe(interaction, state, end)
        self.assertFalse(interaction._device_queries.has_previous)
        np.testing.assert_array_equal(interaction._device_queries.motion.numpy(), 0.0)
        surface, active = self.crossings(interaction, tip, tip, self.float32(start), q)
        self.assertEqual((len(surface), active[0][2]), (1, triangle))
        self.assertEqual(interaction.shifted_start_candidate_tests, 0)
        self.assertEqual(interaction.shifted_start_host_tests, {STALE: 2})

    def test_certified_device_trial_measured_the_motion(self):
        # A device trial refits the surfaces and the query to its candidate without the host's
        # query preparation; the coupler drops the host's certificate, and the stage of a certified
        # trial publishes it again for positions whose motion the query has measured.
        committed, start, end, tip, triangle = self.placed_case(5e-6, lift=4e-5)
        interaction, state = self.build()
        self.commit(interaction, state, committed)
        state.particle_q.assign(end)
        wp.launch(
            insertion._surface_positions,
            len(interaction.mapping),
            [state.particle_q, interaction._mapping, interaction._surface],
            device=self.device,
        )
        interaction.mesh.refit()
        wp.copy(interaction._active_surface, state.particle_q, count=len(interaction.vertices))
        interaction._active_mesh.refit()
        interaction._device_queries.update(state.particle_q)
        interaction._query_positions = interaction._device_query_positions = None
        interaction._device_query_points, interaction._device_locations = None, {}
        mask = wp.zeros(len(interaction.samples), dtype=int, device=self.device)
        self.assertTrue(interaction._stage_certified_trial(state, 0, np.zeros(3), mask))
        q = state.particle_q.numpy().astype(float)
        self.assertTrue(interaction._device_geometry_matches(q))
        np.testing.assert_allclose(interaction._device_queries.motion.numpy(), [0, 0, 4e-5], rtol=1e-4)
        surface, active = self.crossings(interaction, tip, tip, self.float32(start), q)
        self.assertEqual((len(surface), active[0][2]), (1, triangle))
        self.assertEqual(interaction.shifted_start_candidate_tests, 2)
        self.assertEqual(interaction.shifted_start_host_tests, {})

    def test_other_particle_count_and_non_finite_start_keep_the_complete_host_test(self):
        committed, start, end, tip, triangle = self.placed_case(1.57e-5)
        interaction, state = self.build()
        self.commit(interaction, state, committed)
        q = self.describe(interaction, state, end)
        longer = np.vstack((self.float32(start), [[1.0, 1.0, 1.0]]))
        surface, active = self.crossings(interaction, tip, tip, longer, q)
        self.assertEqual((len(surface), active[0][2]), (1, triangle))
        self.assertEqual(interaction.shifted_start_host_tests, {"changed particle count": 2})
        broken = self.float32(start)
        broken[0, 1] = np.nan
        surface, active = self.crossings(interaction, tip, tip, broken, q)
        self.assertEqual((len(surface), active[0][2]), (1, triangle))
        self.assertEqual(
            interaction.shifted_start_host_tests, {"changed particle count": 2, "start difference not finite": 2}
        )
        self.assertEqual(interaction.shifted_start_candidate_tests, 0)
        self.assertEqual(interaction.shifted_start_largest_growth, 0.0)

    def test_growth_is_measured_over_the_particles_of_the_tissue(self):
        # A particle that is not tissue (a thread node after the tissue's particles) moved a long
        # way after the accepted step. The test reads no such particle, and neither does the device.
        committed, start, end, tip, triangle = self.placed_case(1.57e-5)
        interaction, state = self.build()
        extra = np.array([[0.02, 0.02, 0.02]], np.float32)
        state.particle_q = wp.array(np.vstack((self.rest, extra)), dtype=wp.vec3, device=self.device)
        self.commit(interaction, state, np.vstack((committed, extra)))
        q = self.describe(interaction, state, np.vstack((end, extra)))
        old_q = self.float32(np.vstack((start, extra + np.float32(0.5))))
        surface, active = self.crossings(interaction, tip, tip, old_q, q)
        self.assertEqual((len(surface), active[0][2]), (1, triangle))
        self.assertEqual(interaction.shifted_start_candidate_tests, 2)
        self.assertAlmostEqual(interaction.shifted_start_largest_growth / 1.57e-5, 1.0, places=3)


class ShiftedStartStepTests(unittest.TestCase):
    """The calls of a coupled step, with nodes of the accepted state moved after every acceptance."""

    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], np.float32)

    def run_steps(self):
        tets = np.array([[0, 1, 2, 3]], np.int32)
        samples = np.zeros((11, 3))
        samples[:, 2] = np.linspace(0, -0.01, 11)
        interaction = NeedleTissueInteraction(
            self.vertices, tets, boundary_faces(tets)[0], [], samples, 0.0005, samples[0], device=self.device
        )
        state = SimpleNamespace(
            particle_q=wp.array(self.vertices, dtype=wp.vec3, device=self.device),
            particle_qd=wp.zeros(4, dtype=wp.vec3, device=self.device),
            body_q=wp.array([[0.002, 0.002, -0.0001, 0, 0, 0, 1]], dtype=wp.transform, device=self.device),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
        )
        forces = wp.zeros(4, dtype=wp.vec3, device=self.device)
        # Tip heights that enter the material through the face z = 0, withdraw and enter again [m].
        path = np.r_[
            -0.0001, np.linspace(0, 0.0012, 9), np.linspace(0.00105, -0.0012, 16), np.linspace(-0.00105, 0.0012, 16)
        ]
        record, loads = [], []
        for iteration, z in enumerate(path):
            start = SimpleNamespace(particle_q=wp.clone(state.particle_q), body_q=wp.clone(state.body_q))
            moved = self.vertices.copy()
            moved[3, 2] += 1e-5 * np.sin(iteration * 0.1)
            state.particle_q.assign(moved)
            state.body_q.assign([[0.002, 0.002, z, 0, 0, 0, 1]])
            state.body_qd.assign([[0, 0, 0.001 if iteration < 10 or iteration >= 26 else -0.001, 0, 0, 0]])
            state.body_f.zero_()
            forces.zero_()
            interaction.add_forces(state, 0, np.zeros(3), forces, 0.0005, trial=True, implicit=True)
            interaction.check_step_resolution(start, state, 0)
            checked = interaction._checked_crossings[4:]
            interaction.retain_trial_forces(start, 0)
            interaction.accept(state, 0)
            self.assertNotEqual(interaction.status, "failed", interaction.reason)
            record.append(
                (
                    checked,
                    interaction.status,
                    [(event["kind"], event["position"].tobytes(), event["reused"]) for event in interaction.events],
                    (interaction.entry_count, interaction.exit_count, interaction.puncture_count),
                    interaction.track_count,
                )
            )
            loads.append(np.r_[forces.numpy().ravel(), state.body_f.numpy().ravel()])
            # A caller places nodes of the accepted state: two nodes of the face the tip crosses, and the apex.
            placed = moved.copy()
            placed[0, 2] += 4e-6 * np.cos(iteration)
            placed[1, 2] -= 3e-6 * np.sin(2.0 * iteration)
            placed[3, 0] += 5e-6 * np.cos(0.5 * iteration)
            state.particle_q.assign(placed)
        return interaction, record, np.asarray(loads)

    def test_steps_after_a_placement_match_steps_with_the_complete_host_test(self):
        taken, record, loads = self.run_steps()
        with patch.object(NeedleQueryGeometry, "surface_candidates", return_value=None):
            complete, reference, reference_loads = self.run_steps()
        steps = len(record)
        # Crossings, status, events and counts are host values; the loads are sums of a device kernel.
        self.assertEqual(record, reference)
        np.testing.assert_allclose(loads, reference_loads, rtol=2e-6, atol=2e-10)
        # The tip enters through the face z = 0, leaves through it and enters again.
        self.assertGreaterEqual(sum(bool(step[0][0]) for step in record), 3)
        entries, exits, _ = record[-1][3]
        self.assertGreaterEqual(min(entries, exits), 1)
        # Every step but the first starts from placed nodes: two tests each, one per surface.
        if self.device.is_cuda:
            self.assertEqual(taken.shifted_start_candidate_tests, 2 * (steps - 1))
            self.assertEqual(taken.shifted_start_host_tests, {"no committed copy": 2})
            self.assertAlmostEqual(taken.shifted_start_largest_growth, 5e-6, delta=1e-8)
            self.assertEqual(complete.shifted_start_candidate_tests, 0)
            self.assertEqual(complete.shifted_start_host_tests, {"no committed copy": 2, OVERFLOW: 2 * (steps - 1)})
        else:
            # The interaction makes no device query on the CPU: the step keeps its host test.
            for interaction in (taken, complete):
                self.assertIsNone(interaction._device_queries)
                self.assertEqual(interaction.shifted_start_candidate_tests, 0)
                self.assertEqual(
                    interaction.shifted_start_host_tests, {"no device geometry of the end positions": 2 * steps}
                )


if __name__ == "__main__":
    unittest.main()
