# SPDX-License-Identifier: Apache-2.0
"""The capsule set of the dVRK tools: the fit to a drawn part, its bound, and that every reader uses the one set.

The fit is checked on hand-made meshes (a flat bar, a tapered bar, a curved jaw, a round rod)
and on the CAD of both instruments. The readers are the tool models, the rigid collision
shapes of the MuJoCo model, the tool/tissue contact, the measurements and the collision shapes
that the viewer draws; all of them are checked against ``dvrk_tool_capsules``.
"""

import math
import os
import unittest

import newton
import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from test_grasp import add_grasper

from newton_solver_suturebot._src.solvers.coupled.interface.swept_tool_tissue_contact import capsule_pieces
from newton_solver_suturebot._src.solvers.coupled.interface.tool_tissue_contact import capsule_samples
from newton_solver_suturebot.examples.grasp import _dvrk_capsules as fit
from newton_solver_suturebot.examples.grasp import _grasp_scene, _grasp_visual_assets
from newton_solver_suturebot.examples.grasp._dvrk_grasper import (
    CAPSULE_PART_NAMES,
    GrasperConfig,
    jaw_opening_on_needle,
)
from newton_solver_suturebot.examples.grasp._dvrk_rcm_grasper import add_rcm_grasper
from newton_solver_suturebot.examples.grasp._grasp_render import collision_capsules
from newton_solver_suturebot.examples.grasp._grasp_scene import add_dvrk_tools
from newton_solver_suturebot.examples.grasp._grasp_visual_assets import load_dvrk_tool

MM = 1.0e-3
X, Y, Z = np.eye(3)
DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


def _loft(sections):
    """A closed triangle surface through rectangular sections: each (centre (3,), half thickness along x, half width along z)."""
    vertices, triangles = [], []
    for centre, thick, wide in sections:
        centre = np.asarray(centre, dtype=float)
        for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            vertices.append(centre + (sx * thick, 0.0, sz * wide))
    for ring in range(len(sections) - 1):
        for corner in range(4):
            a, b = 4 * ring + corner, 4 * ring + (corner + 1) % 4
            triangles.extend(((a, b, b + 4), (a, b + 4, a + 4)))
    last = 4 * (len(sections) - 1)
    triangles.extend(((0, 2, 1), (0, 3, 2), (last, last + 1, last + 2), (last, last + 2, last + 3)))
    return np.asarray(vertices), np.asarray(triangles)


def _rod(radius, length, sides=96):
    """A round rod along y from 0 to ``length``."""
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.stack((radius * np.cos(angles), np.zeros(sides), radius * np.sin(angles)), axis=1)
    vertices = np.concatenate((ring, ring + (0.0, length, 0.0), [[0.0, 0.0, 0.0], [0.0, length, 0.0]]))
    triangles = []
    for i in range(sides):
        j = (i + 1) % sides
        triangles.extend(
            ((i, j, j + sides), (i, j + sides, i + sides), (2 * sides, j, i), (2 * sides + 1, i + sides, j + sides))
        )
    return vertices, np.asarray(triangles)


class CapsuleFitTests(unittest.TestCase):
    """The rule of the fit on meshes whose answer is known."""

    spacing = 0.05 * MM

    def _cover(self, vertices, triangles, endpoints, radii):
        return fit.capsule_cover(vertices, triangles, endpoints, radii, spacing=self.spacing)

    def test_edge_share_puts_a_right_angled_edge_as_far_outside_as_the_capsule_is_outside_its_faces(self):
        self.assertAlmostEqual(fit.EDGE_SHARE, (math.sqrt(2.0) - 1.0) / (math.sqrt(2.0) + 1.0), places=15)
        # A long square bar in one capsule: the edges are EDGE_SHARE r outside, the faces EDGE_SHARE r inside.
        half = 1.0 * MM
        vertices, triangles = _loft([((0, 0, 0), half, half), ((0, 40 * MM, 0), half, half)])
        points = fit.surface_points(vertices, triangles, spacing=self.spacing)
        endpoints, radii, bounds = fit.fit_capsules(points, Y, Z, edges=True)
        self.assertEqual(len(radii), 1)
        self.assertAlmostEqual(radii[0], math.sqrt(2.0) * half / (1.0 + fit.EDGE_SHARE), delta=1e-12)
        self.assertAlmostEqual(bounds[0], fit.EDGE_SHARE * radii[0], delta=1e-15)
        self.assertAlmostEqual(math.sqrt(2.0) * half - radii[0], radii[0] - half, delta=1e-12)

    def test_round_rod_gets_the_capsule_that_contains_it_and_nothing_is_drawn_in(self):
        radius, length = 4.0 * MM, 60.0 * MM
        vertices, triangles = _rod(radius, length)
        endpoints, radii, bounds = fit.fit_capsules(vertices, Y, Z)
        self.assertEqual(len(radii), 1)
        self.assertAlmostEqual(radii[0], radius, delta=1e-9)
        self.assertEqual(bounds[0], 0.0)
        # The rims of both ends lie on the capsule: its axis is as long as the rod.
        np.testing.assert_allclose(endpoints[0], [[0, 0, 0], [0, length, 0]], rtol=0, atol=1e-9)
        cover = self._cover(vertices, triangles, endpoints, radii)
        self.assertLessEqual(cover["drawn_outside_m"], 1e-9)
        # Beyond the flat ends the caps reach one radius; the pole of a cap is sqrt(2) r from the rim.
        self.assertAlmostEqual(cover["capsules_outside_m"][0], radius, delta=2.0 * self.spacing)

    def test_flat_bar_gets_capsules_side_by_side_within_the_bound(self):
        thick, wide, length = 0.75 * MM, 1.0 * MM, 6.0 * MM
        vertices, triangles = _loft([((0, 0, 0), thick, wide), ((0, length, 0), thick, wide)])
        points = fit.surface_points(vertices, triangles, spacing=self.spacing)
        endpoints, radii, bounds = fit.fit_capsules(points, Y, Z, rows_wide=2, edges=True, margin=self.spacing)
        self.assertEqual(len(radii), 2)
        # Each capsule contains half of the bar and up to one sampling distance of the other half, drawn in by
        # the share of its edges.
        low, high = (math.hypot(thick, 0.5 * (wide + reach)) / (1.0 + fit.EDGE_SHARE) for reach in (0.0, self.spacing))
        self.assertTrue(np.all((radii > low) & (radii <= high + 1e-12)), radii)
        self.assertAlmostEqual(radii[0], radii[1], delta=1e-9)
        np.testing.assert_allclose(bounds, fit.EDGE_SHARE * radii, rtol=0, atol=1e-15)
        np.testing.assert_allclose(endpoints[:, :, 0], 0.0, atol=1e-9)
        offsets = np.sort(endpoints[:, 0, 2])
        np.testing.assert_allclose(offsets, [-0.5 * wide, 0.5 * wide], rtol=0, atol=0.5 * self.spacing)
        # No point of the bar is further outside the set than the bound, on a sampling twice as fine as that of
        # the fit; its edges are that far outside.
        cover = fit.capsule_cover(vertices, triangles, endpoints, radii, spacing=0.5 * self.spacing)
        self.assertLessEqual(cover["drawn_outside_m"], bounds.max() + 1e-9)
        self.assertGreater(cover["drawn_outside_m"], bounds.max() - 1e-9)
        # Along the bar the set is furthest outside it where a capsule passes a side face: by its radius less
        # the distance of its axis from that face. Beyond the flat ends of the bar its caps reach one radius.
        surface = fit.capsule_surface_points(endpoints, radii, spacing=0.5 * self.spacing)
        surface = surface[fit.distance_to_capsules(surface, endpoints, radii) > -1e-9]
        middle = surface[np.abs(surface[:, 1] - 0.5 * length) < length / 6.0]
        self.assertAlmostEqual(
            fit._distance_to_mesh(middle, vertices, triangles).max(),
            radii[0] - (wide - abs(offsets[0])),
            delta=0.5 * self.spacing,
        )
        self.assertAlmostEqual(cover["capsules_outside_m"].max(), radii[0], delta=self.spacing)
        # Without the margin a capsule holds the points of its own half only.
        _, bare, _ = fit.fit_capsules(points, Y, Z, rows_wide=2, edges=True)
        np.testing.assert_allclose(bare, low, rtol=0, atol=2e-6)
        # One capsule for the whole bar needs a larger radius and reaches further out of the thin faces.
        _, single, _ = fit.fit_capsules(points, Y, Z, edges=True)
        self.assertGreater(single[0] - thick, 2.0 * (radii[0] - thick))

    def test_tapered_bar_gets_a_chain_of_capsules_of_falling_radius(self):
        length = 6.0 * MM
        vertices, triangles = _loft([((0, 0, 0), 0.8 * MM, 1.0 * MM), ((0, length, 0), 0.5 * MM, 0.65 * MM)])
        points = fit.surface_points(vertices, triangles, spacing=self.spacing)
        pieces = (length / 3.0, 2.0 * length / 3.0)
        endpoints, radii, bounds = fit.fit_capsules(
            points, Y, Z, boundaries=pieces, rows_wide=2, edges=True, margin=self.spacing
        )
        self.assertEqual(len(radii), 6)
        # Piece by piece from the thick end, two rows each: the radius falls along the chain.
        chain = radii.reshape(3, 2)
        np.testing.assert_allclose(chain[:, 0], chain[:, 1], rtol=0, atol=1e-9)
        self.assertTrue(np.all(np.diff(chain[:, 0]) < 0.0))
        # The two rows run towards each other as the bar gets narrower.
        gap = np.abs(endpoints[:, :, 2])
        self.assertTrue(np.all(gap[:, 1] < gap[:, 0]))
        cover = fit.capsule_cover(vertices, triangles, endpoints, radii, spacing=0.5 * self.spacing)
        # Every point of the bar is within the bound of the capsule of its cell (their bounds differ).
        finer = fit.surface_points(vertices, triangles, spacing=0.5 * self.spacing)
        self.assertLessEqual(fit.distance_to_capsules(finer, endpoints, radii + bounds).max(), 1e-9)
        self.assertLessEqual(cover["drawn_outside_m"], bounds.max() + 1e-9)
        # A chain follows the taper: one piece for the whole bar is further outside it at its thin end.
        whole = fit.fit_capsules(points, Y, Z, rows_wide=2, edges=True, margin=self.spacing)
        self.assertLess(
            cover["capsules_outside_m"].max(), self._cover(vertices, triangles, *whole[:2])["capsules_outside_m"].max()
        )

    def test_curved_jaw_gets_a_chain_that_follows_its_centre_line(self):
        # A bar of constant section across y whose centre line is an eighth of a circle of 8 mm radius in the
        # x-y plane.
        bend, thick, wide = 8.0 * MM, 0.6 * MM, 1.0 * MM
        stations = bend * np.sin(np.linspace(0.0, 0.25 * np.pi, 25))
        vertices, triangles = _loft(
            [((bend - math.sqrt(bend * bend - station * station), station, 0.0), thick, wide) for station in stations]
        )
        points = fit.surface_points(vertices, triangles, spacing=self.spacing)
        pieces = tuple(bend * math.sin(angle) for angle in (np.pi / 12.0, np.pi / 6.0))
        endpoints, radii, bounds = fit.fit_capsules(
            points, Y, Z, boundaries=pieces, rows_wide=2, edges=True, margin=self.spacing
        )
        self.assertEqual(len(radii), 6)
        # The axes of the pieces turn with the centre line: by a twelfth of pi from piece to piece.
        axes = endpoints[::2, 1] - endpoints[::2, 0]
        headings = np.arctan2(axes[:, 0], axes[:, 1])
        np.testing.assert_allclose(headings, np.pi / 24.0 + np.arange(3) * np.pi / 12.0, rtol=0, atol=0.04)
        finer = fit.surface_points(vertices, triangles, spacing=0.5 * self.spacing)
        self.assertLessEqual(fit.distance_to_capsules(finer, endpoints, radii + bounds).max(), 1e-9)
        # A single straight piece cannot follow the curve: its capsules are thicker by more than a fifth.
        _, straight, _ = fit.fit_capsules(points, Y, Z, rows_wide=2, edges=True, margin=self.spacing)
        self.assertGreater(straight.max(), 1.2 * radii.max())

    def test_part_that_a_ball_contains_gets_an_axis_of_the_smallest_length(self):
        # The corners of an octahedron: the ball about its centre contains it.
        vertices = 2.0 * MM * np.concatenate((np.eye(3), -np.eye(3)))
        endpoints, radii, _ = fit.fit_capsules(vertices, Y, Z)
        self.assertAlmostEqual(radii[0], 2.0 * MM, delta=1e-12)
        self.assertAlmostEqual(np.linalg.norm(endpoints[0, 1] - endpoints[0, 0]), fit.MINIMUM_AXIS_LENGTH, delta=1e-12)
        np.testing.assert_allclose(endpoints[0].mean(axis=0), 0.0, atol=1e-12)
        self.assertLessEqual(fit.distance_to_capsules(vertices, endpoints, radii).max(), 1e-12)
        # A short rod is no ball: the rims of its ends lie on the capsule, whose axis is as long as the rod.
        endpoints, _, _ = fit.fit_capsules(_rod(2.0 * MM, 1.0 * MM)[0], Y, Z)
        self.assertAlmostEqual(np.linalg.norm(endpoints[0, 1] - endpoints[0, 0]), 1.0 * MM, delta=1e-9)

    def test_smallest_circle_contains_every_point_and_touches_some(self):
        points = np.random.default_rng(3).normal(size=(5000, 2))
        centre, radius = fit.smallest_circle(points)
        distance = np.linalg.norm(points - centre, axis=1)
        self.assertLessEqual(distance.max(), radius + 1e-12)
        self.assertGreaterEqual(int((distance > radius - 1e-9).sum()), 2)


class ToolCapsuleSetTests(unittest.TestCase):
    """The set of both instruments, against the CAD that is drawn."""

    def test_no_drawn_point_is_further_outside_the_set_than_its_bound(self):
        for instrument in ("PSM1", "PSM2"):
            asset = load_dvrk_tool(instrument=instrument, jaw_opening=0, shaft_length_m=0.24)
            capsules = fit.dvrk_tool_capsules(instrument, 0.24)
            # Shaft and wrist one capsule each; a jaw two at its root and a chain of three pairs along its blade.
            self.assertEqual(
                [capsules.parts.count(name) for name in ("roll", "pitch", "jaw1", "jaw2")],
                [1, 1, *2 * [fit.JAW_ROOT_ROWS + fit.JAW_BLADE_ROWS * fit.JAW_BLADE_PIECES]],
            )
            np.testing.assert_allclose(
                capsules.bounds,
                np.where([name == "roll" for name in capsules.parts], 0.0, fit.EDGE_SHARE * capsules.radii),
            )
            for mesh in asset.meshes:
                if mesh.name == "yaw":
                    continue
                with self.subTest(instrument=instrument, part=mesh.name):
                    frame = capsules.frames[mesh.name]
                    vertices = (np.asarray(mesh.vertices, dtype=float) - frame[:3, 3]) @ frame[:3, :3]
                    # Points of the drawn surface twice as close as those of the fit.
                    spacing = 0.5 * fit.SAMPLING * float(np.ptp(vertices, axis=0).min())
                    points = fit.surface_points(vertices, mesh.triangles, spacing=spacing)
                    chosen = np.asarray([name == mesh.name for name in capsules.parts])
                    # Every point of the drawn surface lies within the bound of one of the capsules of its part.
                    outside = fit.distance_to_capsules(
                        points, capsules.endpoints[chosen], capsules.radii[chosen] + capsules.bounds[chosen]
                    )
                    self.assertLessEqual(outside.max(), 1.0e-9)

    def test_jaw_capsules_are_thinner_towards_the_tip_and_the_blade_begins_beyond_the_hub(self):
        for instrument in ("PSM1", "PSM2"):
            capsules = fit.dvrk_tool_capsules(instrument, 0.24)
            for jaw in ("jaw1", "jaw2"):
                endpoints, radii = capsules.of(jaw)
                blade = radii[fit.JAW_ROOT_ROWS :].reshape(fit.JAW_BLADE_PIECES, fit.JAW_BLADE_ROWS)
                self.assertTrue(np.all(np.diff(blade.max(axis=1)) < 0.0), (instrument, jaw))
                self.assertTrue(np.all(radii[: fit.JAW_ROOT_ROWS] > blade.max()))
                # The root reaches behind the hinge, the blade lies beyond the root.
                self.assertLess(endpoints[: fit.JAW_ROOT_ROWS, 0, 1].max(), 0.0)
                self.assertGreater(capsules.root_ends[jaw], 2.0 * MM)
                self.assertGreater(endpoints[fit.JAW_ROOT_ROWS :, :, 1].min() + blade.max(), capsules.root_ends[jaw])

    def test_set_does_not_depend_on_the_shaft_length_except_for_the_shaft(self):
        short, long = fit.dvrk_tool_capsules("PSM1", 0.12), fit.dvrk_tool_capsules("PSM1", 0.24)
        np.testing.assert_array_equal(short.endpoints[1:], long.endpoints[1:])
        np.testing.assert_array_equal(short.radii[1:], long.radii[1:])
        self.assertAlmostEqual(short.radii[0], long.radii[0], delta=1e-12)
        np.testing.assert_allclose(short.endpoints[0, 1], long.endpoints[0, 1], rtol=0, atol=1e-7)
        lengths = [np.linalg.norm(capsules.endpoints[0, 1] - capsules.endpoints[0, 0]) for capsules in (short, long)]
        self.assertAlmostEqual(lengths[1] - lengths[0], 0.12, delta=1e-6)


def _expected_rows(tool, instrument, *, palm=False):
    """Bodies' capsule rows of a tool model as ``dvrk_tool_capsules`` gives them: end points, radii, part names."""
    capsules = fit.dvrk_tool_capsules(instrument, float(tool.config.shaft_length_m))
    endpoints = capsules.endpoints.copy()
    if palm:
        for row, part in enumerate(capsules.parts):
            if part in ("roll", "pitch"):
                frame = capsules.frames[part]
                endpoints[row] = endpoints[row] @ frame[:3, :3].T + frame[:3, 3]
    return endpoints, capsules.radii, tuple(CAPSULE_PART_NAMES[part] for part in capsules.parts)


class OneCapsuleSetTests(unittest.TestCase):
    """Every reader of the tools' capsules reads the set of ``dvrk_tool_capsules``: none has radii of its own."""

    def test_second_definitions_are_gone(self):
        self.assertFalse(hasattr(_grasp_visual_assets, "drawn_capsule_radii"))
        self.assertFalse(hasattr(_grasp_scene, "drawn_tool_capsule_radii"))
        self.assertFalse(hasattr(_grasp_scene.GraspScene, "_drawn_capsule_radii"))

    def test_both_tool_models_publish_the_fitted_set_and_build_their_collision_shapes_from_it(self):
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            articulated = add_rcm_grasper(builder, label="PSM2", config=GrasperConfig(shaft_length_m=0.24))
            free = add_grasper(builder, label="PSM1")
            other = add_rcm_grasper(builder, label="tool", instrument="PSM2")
            model = builder.finalize()
            transforms, scales, bodies = (
                model.shape_transform.numpy(),
                model.shape_scale.numpy(),
                model.shape_body.numpy(),
            )
            for tool, instrument, palm in ((articulated, "PSM2", False), (free, "PSM1", True), (other, "PSM2", False)):
                endpoints, radii, parts = _expected_rows(tool, instrument, palm=palm)
                np.testing.assert_allclose(tool.capsule_endpoints, endpoints, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(tool.capsule_radii, radii)
                self.assertEqual(tool.capsule_parts, parts)
                np.testing.assert_allclose(
                    tool.capsule_bounds, np.where(np.asarray(parts) == "shaft", 0.0, fit.EDGE_SHARE * radii)
                )
                # The collision shapes of the model are these rows.
                for row, shape in enumerate(tool.capsule_shapes):
                    self.assertEqual(int(bodies[shape]), tool.capsule_bodies[row])
                    self.assertEqual(int(model.shape_type.numpy()[shape]), int(newton.GeoType.CAPSULE))
                    transform = wp.transform(*transforms[shape])
                    ends = [
                        np.asarray(wp.transform_point(transform, wp.vec3(0, 0, sign * scales[shape, 1])))
                        for sign in (-1, 1)
                    ]
                    np.testing.assert_allclose(ends, tool.capsule_endpoints[row], rtol=0, atol=2e-8)
                    self.assertAlmostEqual(scales[shape, 0], radii[row], delta=1e-9)
                # A tool has no other collision shape.
                on_tool = {
                    int(shape)
                    for shape in np.flatnonzero(
                        np.isin(bodies, [tool.palm_body, *tool.jaw_bodies, *tool.capsule_bodies])
                    )
                }
                self.assertEqual(on_tool, set(tool.capsule_shapes))

    def test_jaws_of_one_tool_and_its_wrist_do_not_collide_with_each_other(self):
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            tools = add_dvrk_tools(builder)
            excluded = {frozenset(pair) for pair in builder.shape_collision_filter_pairs}
            for tool in tools:
                rows = list(zip(tool.capsule_shapes, tool.capsule_parts, strict=True))
                for first, first_part in rows:
                    for second, second_part in rows:
                        if "shaft" in (first_part, second_part) or first_part == second_part:
                            continue
                        self.assertIn(frozenset((first, second)), excluded)
            # Capsules of different tools are not excluded.
            self.assertNotIn(frozenset((tools[0].capsule_shapes[-1], tools[1].capsule_shapes[-1])), excluded)

    def test_reset_opening_seats_each_jaw_on_the_needle(self):
        from newton_solver_suturebot import NeedleGeometry
        from newton_solver_suturebot.examples.grasp._dvrk_grasper import _segment_distance, needle_to_gripper_pose

        geometry = NeedleGeometry.load()
        with wp.ScopedDevice("cpu"):
            tool = add_dvrk_tools(newton.ModelBuilder())[0]
        relative = wp.transform_inverse(wp.transform(*needle_to_gripper_pose(geometry)))
        centerline = np.asarray(
            [np.asarray(wp.transform_point(relative, wp.vec3(*point))) for point in geometry.centerline]
        )
        angles = jaw_opening_on_needle(tool, centerline, geometry.thickness)
        self.assertEqual(len(angles), 2)
        self.assertTrue(all(0.0 < angle < 0.5 * tool.config.maximum_opening_rad for angle in angles))
        segments = np.stack((centerline[:-1], centerline[1:]), axis=1)
        interval = 1.0e-7
        for body, sign, seated in zip(tool.jaw_bodies, (1.0, -1.0), angles, strict=True):
            rows = [row for row in range(len(tool.capsule_bodies)) if tool.capsule_bodies[row] == body]

            def gap(row, opened, sign=sign):
                """Gap between the capsule ``row`` of the jaw opened by ``opened`` and the wire (negative: inside)."""
                angle = sign * opened
                turn = np.array(
                    [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
                )
                ends = tool.capsule_endpoints[row] @ turn.T
                return (
                    min(_segment_distance(ends, segment) for segment in segments)
                    - tool.capsule_radii[row]
                    - geometry.thickness
                )

            at = {row: gap(row, seated) for row in rows}
            on_wire = sorted((row for row in rows if at[row] < 0.0), key=at.get)
            # Seated: two capsules of the jaw lie on the wire, the one that met it first and the one beside it.
            self.assertEqual(len(on_wire), 2)
            first, last = on_wire
            # Side by side: the two axes cover the same stretch of the jaw (the middle of each lies within the
            # ends of the other along the jaw) and stand apart across it, along the hinge axis.
            along = tool.capsule_endpoints[:, :, 1]
            across = tool.capsule_endpoints[:, :, 2].mean(axis=1)
            for one, other in ((first, last), (last, first)):
                self.assertTrue(along[other].min() < along[one].mean() < along[other].max())
            self.assertGreater(
                abs(across[first] - across[last]), max(tool.capsule_radii[first], tool.capsule_radii[last])
            )
            # The last of them touches: it lies on the wire by less than a bisection interval, and one interval
            # further open it is clear of it. No smaller opening is needed and no larger one seats the jaw.
            self.assertGreater(at[last], -1.0e-8)
            self.assertGreaterEqual(gap(last, seated + interval), 0.0)
            # The capsule that met the wire first is inside it by the difference, which is less than its own
            # bound on the drawn jaw (20.3 and 20.0 micrometres against 128 here, measured): the jaw closed
            # further than where it first touches, by the angle between the two touches.
            self.assertLess(at[first], at[last])
            self.assertLess(-at[first], tool.capsule_bounds[first])
            low, high = seated, 0.5 * tool.config.maximum_opening_rad
            while high - low > interval:
                middle = 0.5 * (low + high)
                low, high = (middle, high) if gap(first, middle) < 0.0 else (low, middle)
            self.assertGreater(low, seated + interval)
            # Every other capsule of the jaw is clear of the wire there.
            self.assertTrue(all(at[row] >= 0.0 for row in rows if row not in on_wire))
        # A thinner wire is touched with the jaws further closed.
        thinner = jaw_opening_on_needle(tool, centerline, 0.5 * geometry.thickness)
        self.assertTrue(all(first < second for first, second in zip(thinner, angles, strict=True)))
        # A reset takes the two angles as they are, or a total opening for both jaws alike.
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            free = add_grasper(builder, label="PSM1")
            model = builder.finalize()
            state = model.state()
            free.initialize(model, state, (0, 0, 0, 0, 0, 0, 1), opening_rad=angles)
            np.testing.assert_allclose(
                state.joint_q.numpy()[list(free.jaw_q_indices)], (angles[0], -angles[1]), rtol=1e-6
            )
            free.initialize(model, state, (0, 0, 0, 0, 0, 0, 1), opening_rad=0.4)
            np.testing.assert_allclose(state.joint_q.numpy()[list(free.jaw_q_indices)], (0.2, -0.2), rtol=1e-6)
            for wrong in (-0.1, (0.1, -0.1), (0.1, 0.2, 0.3), 2.0 * free.config.maximum_opening_rad):
                with self.assertRaises(ValueError):
                    free.initialize(model, state, (0, 0, 0, 0, 0, 0, 1), opening_rad=wrong)

    def test_regrasp_is_planned_with_the_open_jaws_clear_of_the_wire(self):
        from test_grasp_tip_clearance import (
            TIP_GRASP,
            controller_for,
            flat_tissue,
            planned,
            straight_needle,
            straight_pose,
        )

        from newton_solver_suturebot import NeedleGeometry
        from newton_solver_suturebot.examples.grasp._dvrk_grasper import _segment_distance, needle_to_gripper_pose
        from newton_solver_suturebot.examples.grasp._grasp_procedure import ProceduralGraspController as Controller
        from newton_solver_suturebot.examples.grasp._grasp_procedure import jaw_cover, wire_clearance

        geometry = NeedleGeometry.load()
        with wp.ScopedDevice("cpu"):
            tool = add_dvrk_tools(newton.ModelBuilder())[0]
        described = {
            "jaw_body_indices": list(tool.jaw_bodies),
            "capsule_bodies": list(tool.capsule_bodies),
            "capsule_endpoints": tool.capsule_endpoints,
            "capsule_radii": tool.capsule_radii,
            "capsule_bounds": tool.capsule_bounds,
        }
        radius = geometry.thickness
        half = 0.5 * Controller.release_opening_rad
        cover = jaw_cover(described)
        # Every capsule of the two jaws, with its radius and its bound; a tool without bounds has no such rows.
        self.assertEqual(len(cover), sum(body in tool.jaw_bodies for body in tool.capsule_bodies))
        np.testing.assert_allclose(
            sorted(row[2] for row in cover),
            sorted((tool.capsule_radii + tool.capsule_bounds)[np.isin(tool.capsule_bodies, tool.jaw_bodies)]),
            rtol=1e-12,
        )
        self.assertIsNone(jaw_cover({name: value for name, value in described.items() if name != "capsule_bounds"}))

        def wire(grasp_distance):
            """The needle's centre line in the frame of the tool that holds it this far from its tip."""
            length = float(np.linalg.norm(np.diff(geometry.centerline, axis=0), axis=1).sum())
            relative = needle_to_gripper_pose(
                geometry,
                arc_fraction=grasp_distance / length,
                tangent_roll_rad=TIP_GRASP["pull_grasp_roll_rad"],
                bite_angle_rad=TIP_GRASP["pull_grasp_bite_rad"],
            )
            to_palm = wp.transform_inverse(wp.transform(*relative))
            return np.asarray(
                [np.asarray(wp.transform_point(to_palm, wp.vec3(*point))) for point in geometry.centerline]
            )

        def by_hand(grasp_distance, rows):
            """Smallest distance of the wire from a capsule of the open jaws, less its bound, pair by pair [m]."""
            points = wire(grasp_distance)
            result = np.inf
            for body, sign in zip(tool.jaw_bodies, (1.0, -1.0), strict=True):
                angle = sign * half
                turn = np.array(
                    [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
                )
                for row in rows:
                    if tool.capsule_bodies[row] != body:
                        continue
                    ends = tool.capsule_endpoints[row] @ turn.T
                    gap = min(
                        _segment_distance(ends, np.stack(pair)) for pair in zip(points[:-1], points[1:], strict=True)
                    )
                    result = min(result, gap - tool.capsule_radii[row] - radius - tool.capsule_bounds[row])
            return result

        every = range(len(tool.capsule_bodies))
        for distance in (0.0021, 0.00238, 0.0035, 0.006):
            with self.subTest(distance=distance):
                self.assertAlmostEqual(
                    wire_clearance(cover, wire(distance), radius, half), by_hand(distance, every), delta=1e-12
                )
        # With the twist and bite of the regrasp the needle's tip lies towards the roots of the jaws: 3.5 mm
        # from the tip (where the regrasp was fixed before) the root of one jaw is in the wire, and the open
        # jaws are clear of it, by the bound of every capsule, 2.1 mm from the tip and not one wire radius further.
        self.assertGreaterEqual(wire_clearance(cover, wire(0.0021), radius, half), 0.0)
        self.assertLess(wire_clearance(cover, wire(0.0021 + radius), radius, half), 0.0)
        self.assertLess(wire_clearance(cover, wire(0.0035), radius, half), -0.0002)
        # Blades without roots are clear of the wire there.
        blades = [row for row in every if tool.capsule_bodies[row] in tool.jaw_bodies and tool.capsule_radii[row] < MM]
        thin = {
            **described,
            "capsule_bodies": [tool.capsule_bodies[row] for row in blades],
            "capsule_endpoints": tool.capsule_endpoints[blades],
            "capsule_radii": tool.capsule_radii[blades],
            "capsule_bounds": tool.capsule_bounds[blades],
        }
        self.assertGreaterEqual(wire_clearance(jaw_cover(thin), wire(0.0035), radius, half), 0.0)
        self.assertAlmostEqual(
            wire_clearance(jaw_cover(thin), wire(0.0035), radius, half), by_hand(0.0035, blades), delta=1e-12
        )

        # The plan of the regrasp. A straight wire that stands on its tail far above the tissue: nothing of the
        # tissue is in the way, and a tool without bounds takes the wire by its far end. The tool with its
        # fitted set takes it where its open jaws are clear of the wire, and no point further from the tip
        # that the plan looked at is clear.
        far = flat_tissue(size=0.004, centre=(0.3, 0.3))
        needle_pose = straight_pose(np.pi / 2, 0.03)
        unbounded = {name: value for name, value in described.items() if name != "capsule_bounds"}
        result, _steps = planned(controller_for(straight_needle(), unbounded, tissue=far), needle_pose)
        self.assertAlmostEqual(result[1], 0.02, delta=1e-9)
        controller = controller_for(straight_needle(), described, tissue=far)
        self.assertEqual(len(controller.jaw_cover), len(cover))
        result, _steps = planned(controller, needle_pose)
        self.assertIsNotNone(result)
        pose, offset = result
        self.assertLess(offset, 0.02 - radius)
        self.assertGreater(offset, radius)

        def planned_clearance(distance):
            grasp = controller._tip_grasp(needle_pose, distance)
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*needle_pose[3:])), dtype=float).reshape(3, 3)
            turn = np.asarray(wp.quat_to_matrix(wp.quat(*grasp[3:])), dtype=float).reshape(3, 3)
            points = (controller.centerline @ rotation.T + needle_pose[:3] - grasp[:3]) @ turn
            return wire_clearance(controller.jaw_cover, points, controller.needle_radius, half)

        np.testing.assert_allclose(pose, controller._tip_grasp(needle_pose, offset), atol=1e-12)
        self.assertGreaterEqual(planned_clearance(offset), 0.0)
        # The search ends within 1/64 of the smallest jaw radius of a point that is not clear.
        finest = min(tool.capsule_radii[np.isin(tool.capsule_bodies, tool.jaw_bodies)]) / 64
        self.assertLess(planned_clearance(offset + 1.01 * finest), 0.0)

    def test_tool_tissue_scene_contact_measurements_and_view_read_the_set_of_the_tool_models(self):
        from newton_solver_suturebot.examples.grasp._tool_tissue_scene import ToolTissueScene

        wp.set_device(DEVICE)
        backend = "mujoco-warp" if wp.get_device(DEVICE).is_cuda else "native"
        args = ToolTissueScene.create_parser().parse_args([])
        scene = ToolTissueScene(ViewerNull(), args, rigid_backend=backend, contact="swept")
        tools = scene.graspers
        rows = [_expected_rows(tool, f"PSM{index + 1}") for index, tool in enumerate(tools)]
        endpoints = np.concatenate([row[0] for row in rows])
        radii = np.concatenate([row[1] for row in rows])
        bodies = np.concatenate([tool.capsule_bodies for tool in tools])
        # The scene's capsules are the rows of the tool models, which are the fitted set.
        np.testing.assert_allclose(scene.capsule_endpoints, endpoints, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(scene.capsule_radii, radii)
        np.testing.assert_array_equal(scene.capsule_bodies, bodies)
        # No other radii: what touches the tissue has the radii of the set.
        self.assertIs(scene.capsule_contact_radii, scene.capsule_radii)
        self.assertEqual(scene.jaw_contact_radius, scene.tip_radius)
        self.assertIn(scene.tip_radius, set(tools[0].capsule_radii))
        # The swept contact: its axis pieces are those of the set.
        pieces = capsule_pieces(bodies, endpoints, radii)
        law = scene.contact.law
        np.testing.assert_array_equal(law._sample_body.numpy(), pieces[0])
        np.testing.assert_allclose(law._probe_local.numpy(), pieces[1], rtol=0, atol=1e-7)
        np.testing.assert_allclose(law._sample_radius.numpy(), pieces[4], rtol=0, atol=1e-10)
        # The measurements: the contact spheres and the points of the capsules.
        sample_bodies, centres, sample_radii = capsule_samples(bodies, endpoints, radii)
        np.testing.assert_array_equal(scene.sample_bodies, sample_bodies)
        np.testing.assert_allclose(scene.sample_points, centres, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(scene.sample_radii, sample_radii)
        measured = np.unique(scene._capsule_point_radius.numpy())
        measured = measured[measured > 0.0]
        self.assertEqual(len(measured), len(np.unique(radii.astype(np.float32))))
        self.assertTrue(all(np.isclose(radii, value, rtol=1e-6, atol=0).any() for value in measured))
        # The view: the collision shapes that the viewer draws.
        drawn = collision_capsules(scene.geometry())
        self.assertEqual(len(drawn), len(radii))
        for (body, start, end, radius), expected_body, expected_ends, expected_radius in zip(
            drawn, bodies, endpoints, radii, strict=True
        ):
            self.assertEqual(body, expected_body)
            np.testing.assert_allclose((start, end), expected_ends, rtol=0, atol=1e-12)
            self.assertEqual(radius, expected_radius)
        # The rigid model: one MuJoCo capsule per row, with its radius.
        native = scene.rigid_solver.mj_model
        mapping = scene.rigid_solver.mjc_geom_to_newton_shape.numpy()[0]
        shapes = [shape for tool in tools for shape in tool.capsule_shapes]
        self.assertEqual(native.ngeom, len(shapes))
        for shape, radius in zip(shapes, radii, strict=True):
            geom = int(np.flatnonzero(mapping == shape)[0])
            self.assertAlmostEqual(native.geom_size[geom, 0], radius, delta=1e-9)
        # Counts of the scene: capsules, contact spheres and axis pieces follow from the set.
        self.assertEqual(len(radii), 2 * (2 + 2 * (fit.JAW_ROOT_ROWS + fit.JAW_BLADE_ROWS * fit.JAW_BLADE_PIECES)))
        self.assertEqual(scene.describe()["contact_spheres"], len(sample_bodies))
        self.assertEqual(law.sample_count, len(pieces[0]))
        scene.close()


if __name__ == "__main__":
    unittest.main()
