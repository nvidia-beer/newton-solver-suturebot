# SPDX-License-Identifier: Apache-2.0
"""The sphere markers that ride on the tissue: binding, following a deforming block, the reserve and the start-up."""

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import warp as wp
from warp._src import context as warp_context

from newton_solver_suturebot._src.utils import capture
from newton_solver_suturebot.examples import _tissue_markers
from newton_solver_suturebot.examples._tissue_markers import TissueMarkers

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
RADIUS = 0.0006
GREEN, BLUE = (0.1, 0.8, 0.2), (0.15, 0.35, 0.95)
# Edge of a cube of the block [m] and the corner the block starts at: the size and the place of the pad's elements.
EDGE = 0.002
ORIGIN = np.array([0.012, -0.048, 0.006])
CELLS = (4, 3, 2)
EPSILON = float(np.finfo(np.float32).eps)


def block():
    """A block of cubes cut into six tetrahedra each: nodes [V,3] (float64), tetrahedra [E,4] and surface triangles [T,3]."""
    nx, ny, nz = CELLS
    grid = np.stack(np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij"), axis=-1)
    nodes = ORIGIN + EDGE * grid.reshape(-1, 3)

    def node(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    tetrahedra = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                corner = [node(i + a, j + b, k + c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
                # Six tetrahedra around the diagonal from corner 0 to corner 7.
                for a, b in ((1, 3), (3, 2), (2, 6), (6, 4), (4, 5), (5, 1)):
                    tetrahedra.append((corner[0], corner[a], corner[b], corner[7]))
    tetrahedra = np.asarray(tetrahedra)
    faces = np.concatenate([tetrahedra[:, columns] for columns in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))])
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    return nodes, tetrahedra, faces[first[counts == 1]]


def closest_point(point, nodes, triangles):
    """The point of the surface closest to ``point`` (float64, every triangle): position, triangle, weights, distance."""
    a, b, c = (nodes[triangles[:, corner]] for corner in range(3))
    ab, ac = b - a, c - a
    best = (np.inf, None, None, None)
    for index in range(len(triangles)):
        # In the plane where the foot of the perpendicular is inside; else on the closest edge.
        matrix = np.array(
            [[ab[index] @ ab[index], ab[index] @ ac[index]], [ab[index] @ ac[index], ac[index] @ ac[index]]]
        )
        v, w = np.linalg.solve(matrix, [ab[index] @ (point - a[index]), ac[index] @ (point - a[index])])
        candidates = []
        if v >= 0 and w >= 0 and v + w <= 1:
            candidates.append(np.array([1 - v - w, v, w]))
        for start, end in ((0, 1), (1, 2), (2, 0)):
            corners = (a[index], b[index], c[index])
            edge = corners[end] - corners[start]
            along = np.clip(edge @ (point - corners[start]) / (edge @ edge), 0.0, 1.0)
            weights = np.zeros(3)
            weights[start], weights[end] = 1 - along, along
            candidates.append(weights)
        for weights in candidates:
            position = weights @ np.array([a[index], b[index], c[index]])
            distance = np.linalg.norm(position - point)
            if distance < best[0]:
                best = (distance, position, index, weights)
    return best[1], best[2], best[3], best[0]


class _Viewer:
    """Keeps the arguments of every log_points call, as host copies."""

    def __init__(self):
        self.calls = []

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        self.calls.append(
            (
                name,
                None if points is None else points.numpy().copy(),
                None if radii is None else radii.numpy().copy(),
                None if colors is None else colors.numpy().copy(),
                hidden,
            )
        )


class _QuietViewer:
    """Counts the log_points calls and reads nothing."""

    calls = 0

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        self.calls += 1


ALLOCATORS = ("wp_alloc_host", "wp_alloc_pinned", "wp_alloc_device_default", "wp_alloc_device_async")


@contextmanager
def counted_allocations():
    """Count the calls of Warp's native allocators, host and device, while the block runs."""
    core, calls, originals = warp_context.runtime.core, [], {}
    for name in ALLOCATORS:
        if hasattr(core, name):
            originals[name] = getattr(core, name)

            def counted(*arguments, _name=name, _function=originals[name]):
                calls.append(_name)
                return _function(*arguments)

            setattr(core, name, counted)
    try:
        yield calls
    finally:
        for name, function in originals.items():
            setattr(core, name, function)


def tolerance(*positions):
    """Sixteen float32 steps at the largest coordinate of the positions [m]."""
    return 16 * EPSILON * max(float(np.abs(values).max()) for values in positions)


class TissueMarkersTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(DEVICE)
        self.nodes, self.tetrahedra, self.triangles = block()
        self.state = wp.array(self.nodes.astype(np.float32), dtype=wp.vec3, device=self.device)
        self.viewer = _Viewer()

    def markers(self, **options):
        options = {"radius": RADIUS, "device": self.device, "per_world": 16, "triangles": self.triangles} | options
        return TissueMarkers(self.viewer, "needle/entry_exit", **options)

    def drawn(self, markers, state=None):
        markers.draw(self.state if state is None else state)
        return self.viewer.calls[-1]

    def positions(self):
        """Positions near the surface of the block, each with one closest surface point [m]."""
        low, high = self.nodes.min(0), self.nodes.max(0)
        size = high - low
        return np.array(
            [
                low + size * (0.37, 0.41, 1.0) + (0, 0, 5.0e-6),  # above the top, inside a triangle
                low + size * (0.62, 0.23, 1.0) - (0, 0, 5.0e-6),  # just under the top: an entry
                low + size * (0.50, 1.0 / 3.0, 1.0) + (0, 0, 2.0e-5),  # above a node of the top
                low + size * (0.25, 0.40, 1.0) + (0, 0, 1.0e-5),  # above an edge of the top
                low + size * (1.0, 0.31, 0.57) + (3.0e-5, 0, 0),  # beside a side face
                low + size * (0.13, 0.0, 0.29) - (0, 1.0e-5, 0),  # beside another side face
                low + size * (0.81, 0.66, 0.0) - (0, 0, 4.0e-5),  # under the bottom
                high + (2.0e-4, 3.0e-4, 1.0e-4),  # beyond a corner
                low + size * (0.44, 1.0, 1.0) + (0, 2.0e-4, 1.0e-4),  # beyond an outer edge
            ]
        )

    def test_a_position_is_bound_to_the_closest_point_of_the_drawn_surface(self):
        markers = self.markers()
        positions = self.positions()
        for index, position in enumerate(positions):
            self.assertEqual(markers.attach_point(self.state, position, GREEN if index % 2 else BLUE), index)
        name, points, radii, colors, hidden = self.drawn(markers)
        self.assertEqual((name, hidden, markers.count), ("needle/entry_exit", False, len(positions)))
        np.testing.assert_array_equal(radii, np.full(len(positions), RADIUS, dtype=np.float32))
        np.testing.assert_array_equal(
            colors, np.array([GREEN if index % 2 else BLUE for index in range(len(positions))], dtype=np.float32)
        )
        nodes, weights = markers.nodes.numpy()[: len(positions)], markers.weights.numpy()[: len(positions)]
        surface = {tuple(sorted(triangle)) for triangle in self.triangles.tolist()}
        for index, position in enumerate(positions):
            expected, _, _, distance = closest_point(position, self.nodes, self.triangles)
            with self.subTest(position=index):
                # On the drawn surface: a triangle of it, with weights of a point inside it.
                self.assertIn(tuple(sorted(nodes[index].tolist())), surface)
                self.assertTrue(np.all(weights[index] >= 0))
                self.assertAlmostEqual(float(weights[index].sum()), 1.0, delta=4 * EPSILON)
                # At the closest point of the whole surface, and as far from the position as that point.
                np.testing.assert_allclose(points[index], expected, rtol=0, atol=tolerance(self.nodes))
                self.assertAlmostEqual(
                    float(np.linalg.norm(points[index].astype(float) - position)), distance, delta=tolerance(self.nodes)
                )

    def test_a_known_triangle_is_bound_as_given(self):
        markers = self.markers(triangles=None)
        random = np.random.default_rng(7)
        chosen = self.triangles[random.choice(len(self.triangles), 6, replace=False)]
        weights = random.dirichlet((1.0, 1.0, 1.0), 6)
        for nodes, weight in zip(chosen, weights, strict=True):
            markers.attach(nodes, weight, GREEN)
        _, points, _, colors, hidden = self.drawn(markers)
        self.assertFalse(hidden)
        np.testing.assert_array_equal(markers.nodes.numpy()[:6], chosen)
        np.testing.assert_array_equal(markers.weights.numpy()[:6], weights.astype(np.float32))
        np.testing.assert_array_equal(colors, np.tile(np.float32(GREEN), (6, 1)))
        expected = np.einsum("mk,mkd->md", weights, self.nodes[chosen])
        np.testing.assert_allclose(points, expected, rtol=0, atol=tolerance(self.nodes))
        with self.assertRaisesRegex(ValueError, "surface triangles"):
            markers.attach_point(self.state, self.nodes[0], GREEN)

    def test_markers_stay_on_their_material_points_when_the_block_deforms(self):
        markers = self.markers()
        positions = self.positions()
        material = []
        for position in positions:
            markers.attach_point(self.state, position, GREEN)
            material.append(closest_point(position, self.nodes, self.triangles))
        random = np.random.default_rng(11)
        picked = self.triangles[random.choice(len(self.triangles), 5, replace=False)]
        for nodes, weight in zip(picked, random.dirichlet((1.0, 1.0, 1.0), 5), strict=True):
            markers.attach(nodes, weight, BLUE)
            material.append((weight @ self.nodes[nodes], None, weight, 0.0))
        rest = np.array([item[0] for item in material])
        at_rest = tolerance(self.nodes)
        np.testing.assert_allclose(self.drawn(markers)[1], rest, rtol=0, atol=at_rest)

        axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + np.sin(0.7) * skew + (1 - np.cos(0.7)) * skew @ skew
        centre = self.nodes.mean(0)
        maps = {
            "stretch": (np.diag([1.5, 0.8, 1.2]), np.zeros(3)),
            "shear": (np.array([[1.0, 0.0, 0.4], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.zeros(3)),
            "rigid motion": (rotation, np.array([0.010, -0.020, 0.030])),
            "all three": (
                rotation @ np.array([[1.3, 0.0, 0.3], [0.1, 0.9, 0.0], [0.0, 0.2, 1.1]]),
                np.array([-0.004, 0.007, 0.002]),
            ),
        }
        for label, (gradient, shift) in maps.items():
            with self.subTest(deformation=label):
                # About the centre of the block: every material point x goes to centre + F (x - centre) + t.
                moved = centre + (self.nodes - centre) @ gradient.T + shift
                self.state.assign(moved.astype(np.float32))
                expected = centre + (rest - centre) @ gradient.T + shift
                # The binding error at rest grows with the map; the drawn state and its sum are float32.
                bound = np.linalg.norm(gradient, 2) * at_rest + tolerance(moved)
                points = self.drawn(markers)[1]
                np.testing.assert_allclose(points, expected, rtol=0, atol=bound)
                # The markers moved with the block: more than a hundred times the bound.
                self.assertGreater(float(np.linalg.norm(points - rest, axis=1).min()), 100 * bound)

        # A deformation that is not affine: the marker is at the same weights of its triangle's moved nodes.
        wave = self.nodes + 0.0004 * np.sin(900.0 * self.nodes[:, [1, 2, 0]])
        self.state.assign(wave.astype(np.float32))
        nodes, weights = markers.nodes.numpy()[: markers.count], markers.weights.numpy()[: markers.count]
        expected = np.einsum("mk,mkd->md", weights.astype(float), wave.astype(np.float32).astype(float)[nodes])
        np.testing.assert_allclose(self.drawn(markers)[1], expected, rtol=0, atol=tolerance(wave))
        # The same from the independent closest points, where that point has one triangle (inside it).
        for index, (_, triangle, weight, _) in enumerate(material[: len(positions)]):
            if weight.min() > 1.0e-3:
                np.testing.assert_allclose(
                    self.viewer.calls[-1][1][index],
                    weight @ wave[self.triangles[triangle]],
                    rtol=0,
                    atol=3 * at_rest + tolerance(wave),
                )

    def test_a_marker_beyond_the_reserve_takes_the_row_of_the_oldest_of_its_world(self):
        nodes = len(self.nodes)
        offset = np.array([0.45, 0.0, 0.0])
        state = wp.array(
            np.concatenate([self.nodes, self.nodes + offset]).astype(np.float32), dtype=wp.vec3, device=self.device
        )
        markers = self.markers(per_world=3, worlds=2, world_nodes=nodes)
        self.assertEqual(markers.capacity, 6)
        positions = self.positions()
        expected = [closest_point(position, self.nodes, self.triangles)[0] for position in positions]
        rows = [markers.attach_point(state, positions[index], GREEN) for index in range(3)]
        rows.append(markers.attach_point(state, positions[3] + offset, BLUE, world=1))
        self.assertEqual((rows, markers.count, markers.replaced), ([0, 1, 2, 3], 4, 0))
        # The reserve of world 0 is full: its oldest markers make room, one after the other; world 1 keeps its own.
        rows = [markers.attach_point(state, positions[index], BLUE) for index in (4, 5, 6, 7)]
        self.assertEqual((rows, markers.count, markers.replaced), ([0, 1, 2, 0], 4, 4))
        _, points, _, colors, hidden = self.drawn(markers, state)
        self.assertFalse(hidden)
        shown = np.array([expected[7], expected[5], expected[6], expected[3] + offset])
        np.testing.assert_allclose(points, shown, rtol=0, atol=tolerance(shown))
        np.testing.assert_array_equal(colors, np.tile(np.float32(BLUE), (4, 1)))
        # Two more markers of world 1 are new rows, the next one replaces its first.
        rows = [markers.attach_point(state, positions[index] + offset, GREEN, world=1) for index in (0, 1, 2)]
        self.assertEqual((rows, markers.count, markers.replaced), ([4, 5, 3], 6, 5))
        self.assertEqual(len(self.drawn(markers, state)[1]), markers.capacity)

    def test_a_marker_is_recoloured_by_its_number_in_its_world_while_it_is_shown(self):
        """Fails if a colour goes to another marker than the one named, to a row that a newer marker has taken,
        or moves a marker."""
        nodes = len(self.nodes)
        offset = np.array([0.45, 0.0, 0.0])
        state = wp.array(
            np.concatenate([self.nodes, self.nodes + offset]).astype(np.float32), dtype=wp.vec3, device=self.device
        )
        markers = self.markers(per_world=3, worlds=2, world_nodes=nodes)
        positions = self.positions()
        yellow = (1.0, 0.8, 0.1)
        # Without a marker there is nothing to colour.
        self.assertFalse(markers.recolor(0, 0, yellow))
        for index in range(2):
            markers.attach_point(state, positions[index], GREEN)
        markers.attach_point(state, positions[3] + offset, BLUE, world=1)
        before = self.drawn(markers, state)
        # The second marker of world 0 and the first of world 1: each world counts its own markers.
        self.assertTrue(markers.recolor(0, 1, yellow))
        self.assertTrue(markers.recolor(1, 0, yellow))
        _, points, _, colors, _ = self.drawn(markers, state)
        np.testing.assert_array_equal(points, before[1])
        np.testing.assert_array_equal(colors, np.float32([GREEN, yellow, yellow]))
        # A number that no marker of the world has changes nothing.
        for world, index in ((0, 2), (0, -1), (1, 1)):
            self.assertFalse(markers.recolor(world, index, BLUE))
        np.testing.assert_array_equal(self.drawn(markers, state)[3], colors)
        # Three more markers of world 0, the last two beyond its reserve of three: they take the rows of
        # its first two markers, which are not shown any more and cannot be coloured. Marker 3 is in row 0.
        for index in (4, 5, 6):
            markers.attach_point(state, positions[index], GREEN)
        self.assertEqual(markers.replaced, 2)
        self.assertFalse(markers.recolor(0, 0, BLUE))
        self.assertFalse(markers.recolor(0, 1, BLUE))
        self.assertTrue(markers.recolor(0, 3, BLUE))
        np.testing.assert_array_equal(self.drawn(markers, state)[3], np.float32([BLUE, GREEN, yellow, GREEN]))
        # A reset of world 0 starts its numbers again; the marker of world 1 keeps its colour.
        markers.clear(0)
        self.assertFalse(markers.recolor(0, 0, yellow))
        markers.attach_point(state, positions[0], GREEN)
        self.assertTrue(markers.recolor(0, 0, BLUE))
        np.testing.assert_array_equal(self.drawn(markers, state)[3], np.float32([yellow, BLUE]))

    def test_each_world_binds_to_its_own_surface(self):
        nodes = len(self.nodes)
        offset = np.array([0.45, 0.45, 0.0])
        state = wp.array(
            np.concatenate([self.nodes, self.nodes + offset]).astype(np.float32), dtype=wp.vec3, device=self.device
        )
        markers = self.markers(worlds=2, world_nodes=nodes)
        position = self.positions()[1]
        expected = closest_point(position, self.nodes, self.triangles)[0]
        markers.attach_point(state, position, GREEN, world=0)
        markers.attach_point(state, position + offset, GREEN, world=1)
        # A position of world 1 that lies at the block of world 0 is still bound to the surface of world 1.
        markers.attach_point(state, position, BLUE, world=1)
        bound = markers.nodes.numpy()[:3]
        self.assertTrue(np.all(bound[0] < nodes))
        self.assertTrue(np.all(bound[1:] >= nodes))
        points = self.drawn(markers, state)[1]
        np.testing.assert_allclose(points[0], expected, rtol=0, atol=tolerance(self.nodes))
        np.testing.assert_allclose(points[1], expected + offset, rtol=0, atol=tolerance(self.nodes + offset))
        far = closest_point(position, self.nodes + offset, self.triangles)[0]
        np.testing.assert_allclose(points[2], far, rtol=0, atol=tolerance(self.nodes + offset))
        with self.assertRaisesRegex(ValueError, "World 2"):
            markers.attach_point(state, position, GREEN, world=2)
        self.assertEqual(markers.count, 3)
        with self.assertRaisesRegex(ValueError, "world_nodes"):
            self.markers(worlds=2)

    def test_a_reset_clears_the_markers(self):
        nodes = len(self.nodes)
        offset = np.array([0.45, 0.0, 0.0])
        state = wp.array(
            np.concatenate([self.nodes, self.nodes + offset]).astype(np.float32), dtype=wp.vec3, device=self.device
        )
        markers = self.markers(per_world=4, worlds=2, world_nodes=nodes)
        positions = self.positions()
        for index, world in enumerate((0, 1, 0, 1, 1)):
            markers.attach_point(state, positions[index] + world * offset, GREEN if world else BLUE, world=world)
        before = self.drawn(markers, state)
        # A reset of world 0: its two markers go, those of world 1 stay where they are bound.
        markers.clear(0)
        self.assertEqual(markers.count, 3)
        _, points, _, colors, hidden = self.drawn(markers, state)
        self.assertFalse(hidden)
        np.testing.assert_array_equal(points, before[1][[1, 3, 4]])
        np.testing.assert_array_equal(colors, np.tile(np.float32(GREEN), (3, 1)))
        # They still ride: move the nodes of world 1.
        moved = np.concatenate([self.nodes, self.nodes + offset + (0.0, 0.001, 0.002)]).astype(np.float32)
        state.assign(moved)
        np.testing.assert_allclose(
            self.drawn(markers, state)[1], before[1][[1, 3, 4]] + (0.0, 0.001, 0.002), rtol=0, atol=tolerance(moved)
        )
        # A new marker of the cleared world is a new row.
        self.assertEqual(markers.attach_point(state, positions[5], BLUE), 3)
        # A reset of everything: nothing is shown, and the rows start again.
        markers.clear()
        self.assertEqual(markers.count, 0)
        self.assertEqual(self.drawn(markers, state), ("needle/entry_exit", None, None, None, True))
        self.assertEqual(markers.attach_point(state, positions[6], GREEN, world=1), 0)
        self.assertEqual(len(self.drawn(markers, state)[1]), 1)

    def test_the_start_up_draw_logs_the_whole_reserve_hidden_and_leaves_no_marker(self):
        for triangles in (self.triangles, None):
            with self.subTest(search=triangles is not None):
                markers = self.markers(per_world=5, triangles=triangles)
                markers.warm(self.state)
                name, points, radii, colors, hidden = self.viewer.calls[-1]
                self.assertEqual((name, hidden, len(points), len(colors)), ("needle/entry_exit", True, 5, 5))
                np.testing.assert_array_equal(radii, np.full(5, RADIUS, dtype=np.float32))
                self.assertEqual((markers.count, markers.replaced), (0, 0))
                # Without a marker a frame hides the batch and launches nothing.
                self.assertEqual(self.drawn(markers), ("needle/entry_exit", None, None, None, True))
                self.assertEqual(markers.attach(self.triangles[0], (0.2, 0.3, 0.5), GREEN), 0)
                with self.assertRaisesRegex(RuntimeError, "before the first marker"):
                    markers.warm(self.state)

    def test_frames_and_new_markers_allocate_nothing_and_load_no_kernel_module(self):
        viewer = _QuietViewer()
        size = 256 if self.device.is_cuda else 1
        with counted_allocations() as calls:
            wp.zeros(4, dtype=float, device=self.device)
        # The counter sees an array of this device.
        self.assertEqual(len(calls), 1, calls)
        nodes = len(self.nodes)
        state = wp.array(
            np.concatenate([self.nodes, self.nodes + (0.45, 0.0, 0.0)]).astype(np.float32),
            dtype=wp.vec3,
            device=self.device,
        )
        markers = TissueMarkers(
            viewer,
            "needle/entry_exit",
            radius=RADIUS,
            device=self.device,
            per_world=2,
            worlds=2,
            triangles=self.triangles,
            world_nodes=nodes,
        )
        # The kernel module is loaded when the markers are made, with no launch.
        loaded = capture.loaded_modules()
        self.assertIn((_tissue_markers.marker_points.module.name, self.device.alias, size), loaded)
        # Every kernel that the markers launch, by name, in the start-up draw and then in the frames.
        launched, launch = [], wp.launch

        def recorded(kernel, *arguments, **options):
            launched.append(kernel.key)
            return launch(kernel, *arguments, **options)

        with patch.object(wp, "launch", new=recorded):
            markers.warm(state)
        warmed = set(launched)
        del launched[:]
        positions = self.positions()
        with counted_allocations() as calls, patch.object(wp, "launch", new=recorded):
            for frame in range(12):
                if frame in (2, 5, 6, 7, 9):
                    # A new marker, the last three beyond the reserve of their world.
                    markers.attach_point(state, positions[frame % len(positions)], GREEN, world=0)
                if frame == 3:
                    # The marker of frame 2 gets another colour (the tie of a thread at an entry).
                    self.assertTrue(markers.recolor(0, 0, BLUE))
                if frame == 8:
                    markers.attach(self.triangles[3] + nodes, (0.2, 0.3, 0.5), BLUE, world=1)
                if frame == 10:
                    markers.clear(0)
                markers.draw(state)
        self.assertEqual(calls, [])
        self.assertEqual(viewer.calls, 1 + 12)
        self.assertEqual(markers.replaced, 3)
        self.assertEqual(capture.loaded_modules(), loaded)
        # The start-up draw issued every launch of the frames: none of them is a first launch.
        self.assertIn(_tissue_markers._color_marker.key, launched)
        self.assertEqual(set(launched) - warmed, set())

    def test_a_reserve_and_a_surface_are_checked_when_the_markers_are_made(self):
        with self.assertRaisesRegex(ValueError, "at least one marker"):
            self.markers(per_world=0)
        with self.assertRaisesRegex(ValueError, r"\[T, 3\]"):
            self.markers(triangles=np.zeros((0, 3), dtype=int))
        markers = self.markers(per_world=1)
        with self.assertRaisesRegex(ValueError, "World -1"):
            markers.attach(self.triangles[0], (1.0, 0.0, 0.0), GREEN, world=-1)
        self.assertEqual(markers.count, 0)


if __name__ == "__main__":
    unittest.main()
