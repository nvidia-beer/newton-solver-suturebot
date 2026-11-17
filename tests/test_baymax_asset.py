"""The Baymax body of the incision example: the bundled file, its facts, its size and place, its parts."""

import hashlib
import json
import math
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from test_cut_pipeline import _cube

from newton_solver_suturebot import load_mesh
from newton_solver_suturebot.examples.grasp._dvrk_layout import load_source_stitch
from newton_solver_suturebot.examples.incision import _baymax_body as body
from newton_solver_suturebot.examples.incision import example_fem_surgical_incision as incision


def _edge_lengths(triangles):
    return np.concatenate([np.linalg.norm(triangles[:, k] - triangles[:, (k + 1) % 3], axis=-1) for k in range(3)])


class BaymaxAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.mesh,) = load_mesh(body.ASSET)
        cls.points = np.asarray(cls.mesh.vertices, dtype=np.float64)
        cls.cells = np.asarray(cls.mesh.indices)
        cls.surface = np.asarray(cls.mesh.surface_triangles)
        cls.positions, cls.elements = body.load_body()

    def test_file_is_the_recorded_copy(self):
        data = body.ASSET.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), body.SHA256)
        record = json.loads((body.ASSET.parent / "provenance.json").read_text())
        self.assertEqual((record["sha256"], record["bytes"], record["changes"]), (body.SHA256, len(data), []))
        self.assertEqual(
            (record["counts"]["points"], record["counts"]["tetrahedra"], record["counts"]["surface_triangles"]),
            (body.POINTS, body.TETRAHEDRA, body.SURFACE_TRIANGLES),
        )
        # The file has no licence line of its own; the record and the README say so.
        self.assertNotIn(b"SPDX", data)
        self.assertNotIn(b"icense", data)
        readme = (body.ASSET.parent / "README.md").read_text()
        for statement in ("no licence line", "The Newton Developers under Apache-2.0", "No upstream location", "film"):
            self.assertIn(statement, readme)
        self.assertIn(body.SHA256, readme)

    def test_counts_volumes_and_closed_surface(self):
        self.assertEqual(self.points.shape, (body.POINTS, 3))
        self.assertEqual(self.cells.shape, (body.TETRAHEDRA, 4))
        self.assertEqual(self.surface.shape, (body.SURFACE_TRIANGLES, 3))
        corners = self.points[self.cells]
        volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
        self.assertTrue(np.all(volumes > 0.0))
        self.assertAlmostEqual(volumes.sum(), 0.037576, delta=1.0e-6)
        # The authored surface is the boundary of the volume, closed: every edge is used twice.
        faces = self.cells[:, ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))].reshape(-1, 3)
        keys, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
        np.testing.assert_array_equal(keys[counts == 1], np.unique(np.sort(self.surface, axis=1), axis=0))
        edges = np.sort(self.surface[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
        self.assertTrue(np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2))

    def test_facts_of_the_asset_are_what_the_scene_module_states(self):
        low, high = self.points.min(axis=0), self.points.max(axis=0)
        np.testing.assert_allclose(high - low, body.EXTENT, atol=1.0e-6)
        np.testing.assert_allclose(low, -0.5 * np.asarray(body.EXTENT), atol=1.0e-6)
        corners = self.points[self.cells]
        self.assertAlmostEqual(
            np.linalg.norm(corners[:, 1:] - corners[:, :1], axis=-1).mean(), body.MEAN_EDGE, delta=1.0e-6
        )
        self.assertAlmostEqual(np.median(_edge_lengths(self.points[self.surface])), body.SURFACE_EDGE, delta=1.0e-6)
        volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
        areas = np.stack(
            [
                0.5 * np.linalg.norm(np.cross(corners[:, b] - corners[:, a], corners[:, c] - corners[:, a]), axis=-1)
                for a, b, c in ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
            ],
            axis=1,
        )
        self.assertAlmostEqual((3.0 * volumes[:, None] / areas).min(), body.SMALLEST_ALTITUDE, delta=1.0e-8)
        apex = int(np.argmax(self.points[:, 2]))
        self.assertEqual(apex, 2459)
        np.testing.assert_allclose(self.points[apex], body.BELLY_APEX, atol=1.0e-6)

    def test_scale_holds_its_bounds(self):
        # Upper, thread: mean tissue edge at most sqrt(8 R r); lower: the level belly holds needle and stitch row.
        chord = math.sqrt(8.0 * 0.009449 * 0.00015)
        self.assertLess(body.SCALE * body.MEAN_EDGE, chord)
        self.assertLess(body.SCALE * 0.02140, chord)
        self.assertGreaterEqual(body.SCALE * 2.0 * body.WOUND_HALF_LENGTH, 0.020)
        self.assertGreaterEqual(body.SCALE * 2.0 * body.WOUND_HALF_LENGTH, 2.0 * 0.009449)

    def test_body_in_the_scene_is_135_mm_across_on_its_back_at_the_pad_stitch(self):
        np.testing.assert_array_equal(self.elements, self.cells)
        low, high = self.positions.min(axis=0), self.positions.max(axis=0)
        np.testing.assert_allclose(1000.0 * (high - low), (135.0, 85.4, 44.1), atol=0.05)
        # Belly up: the apex is the highest point; arm span along x; head at +y.
        self.assertEqual(int(np.argmax(self.positions[:, 2])), 2459)
        sets = body.node_sets(self.positions)
        self.assertGreater(self.positions[sets["head"], 1].min(), self.positions[sets["leg_plus_x"], 1].max())
        self.assertAlmostEqual(low[2], body.FLOOR_HEIGHT, delta=1.0e-9)
        np.testing.assert_allclose(body.asset_coordinates(self.positions), self.points, atol=1.0e-12)
        # The middle of the stitch is where the middle of the recorded stitch is on the pad.
        stitch = load_source_stitch()
        np.testing.assert_allclose(0.5 * (stitch["entry"] + stitch["exit"]), body.PAD_STITCH_MIDPOINT, atol=1.0e-6)
        frame = body.stitch_frame()
        np.testing.assert_allclose(frame["midpoint"], body.PAD_STITCH_MIDPOINT, atol=1.0e-12)
        # It lies on the skin: a vertical line through it meets the belly within a hundredth of a millimetre.
        triangles = self.positions[self.surface]
        a, b, c = (triangles[:, k, :2] for k in range(3))

        def cross(u, v):
            return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        area = cross(b - a, c - a)
        second, third = cross(frame["midpoint"][:2] - a, c - a) / area, cross(b - a, frame["midpoint"][:2] - a) / area
        weights = np.stack((1.0 - second - third, second, third), axis=1)
        heights = (weights * triangles[:, :, 2]).sum(axis=1)[(weights >= 0.0).all(axis=1)]
        self.assertAlmostEqual(heights.max(), frame["midpoint"][2], delta=1.0e-5)
        # The body's back is its thickness below: 43.7 mm at the stitch.
        self.assertAlmostEqual(1000.0 * (heights.max() - heights.min()), 43.7, delta=0.05)
        # The stitch runs towards the feet (-y), across a wound that runs along the arm span (x); up is +z.
        np.testing.assert_array_equal(frame["direction"], (0.0, -1.0, 0.0))
        np.testing.assert_array_equal(frame["wound_axis"], (1.0, 0.0, 0.0))
        np.testing.assert_array_equal(frame["up"], (0.0, 0.0, 1.0))

    def test_node_sets_follow_their_rules(self):
        sets = body.node_sets(self.positions)
        counts = {name: len(nodes) for name, nodes in sets.items()}
        self.assertEqual(
            counts,
            {
                "arm_plus_x": 1044,
                "arm_minus_x": 1027,
                "hand_plus_x": 127,
                "hand_minus_x": 138,
                "leg_plus_x": 269,
                "leg_minus_x": 269,
                "foot_plus_x": 93,
                "foot_minus_x": 92,
                "head": 310,
                "belly": 866,
            },
        )
        parts = ("arm_plus_x", "arm_minus_x", "leg_plus_x", "leg_minus_x", "head", "belly")
        joined = np.concatenate([sets[name] for name in parts])
        self.assertEqual(len(np.unique(joined)), len(joined))
        for limb, tip in (("arm_plus_x", "hand_plus_x"), ("leg_minus_x", "foot_minus_x")):
            self.assertTrue(np.isin(sets[tip], sets[limb]).all())
        # Each leg below the rule's height is a piece of its own: no tetrahedron joins the two.
        legs = np.zeros(len(self.points), dtype=int)
        legs[sets["leg_plus_x"]], legs[sets["leg_minus_x"]] = 1, 2
        both = legs[self.cells]
        self.assertFalse(np.any((both == 1).any(axis=1) & (both == 2).any(axis=1)))

    def test_wound_blade_crosses_the_belly_and_stops_inside(self):
        blade = body.asset_coordinates(body.wound_blade())
        lower, upper = body.BELLY_BOX
        np.testing.assert_allclose(blade[:, 1], body.WOUND_Y, atol=1.0e-12)
        self.assertTrue(np.all((blade[:, 0] >= lower[0]) & (blade[:, 0] <= upper[0])))
        np.testing.assert_allclose(np.ptp(blade[:, 0]), 2.0 * body.WOUND_HALF_LENGTH)
        # Lower edge inside the body, upper edge above every node.
        self.assertGreater(blade[:2, 2].min(), self.points[:, 2].min() + 0.5 * body.EXTENT[2])
        self.assertGreater(blade[2:, 2].min(), self.points[:, 2].max())
        scene = body.wound_blade()
        self.assertAlmostEqual(1000.0 * np.ptp(scene[:, 0]), 27.0, delta=0.01)
        self.assertAlmostEqual(1000.0 * (body.stitch_frame()["midpoint"][2] - scene[0, 2]), 13.5, delta=0.01)

    def test_lengths_of_the_cut_scene_are_derived_from_the_body(self):
        self.assertAlmostEqual(1000.0 * body.CONTACT_SEARCH_RADIUS, 4.04, delta=0.005)
        self.assertGreater(body.CONTACT_SEARCH_RADIUS, 30000.0 / 2.0e7)
        self.assertAlmostEqual(1000.0 * body.GRIP_RADIUS, 3.01, delta=0.005)
        parser = incision.Example.create_parser()
        args = parser.parse_args([])
        self.assertEqual(args.grip_radius, body.GRIP_RADIUS)
        stiffness = body.grip_stiffness(args.young_modulus, args.poisson_ratio)
        self.assertAlmostEqual(stiffness, 2061.0, delta=1.0)
        self.assertAlmostEqual(1000.0 * body.grip_speed(args.substeps * 60.0), 2.96, delta=0.005)
        # The start view: the far hand at nine tenths of half the picture's width.
        half_width = body.VIEW_DISTANCE * body.VIEW_ASPECT * math.tan(math.radians(0.5 * body.VIEW_FIELD_DEGREES))
        self.assertAlmostEqual((body.VIEW_PANEL_SHIFT + 0.5) / half_width, 0.9)

    def test_example_takes_its_scene_numbers_from_the_scene_module(self):
        # What reaches the simulation: the bundled file as the default asset, the floor at the lowest point
        # of the body as loaded, the contact reach and the grip numbers of the scene module. A cube whose
        # lowest point is not at zero stands in for the body: these numbers do not depend on the mesh.
        parser = incision.Example.create_parser()
        self.assertEqual(parser.parse_args([]).asset, str(body.ASSET))
        args = parser.parse_args(["--fragment_capacity", "64"])
        points, elements = _cube()
        points = points - np.asarray((0.0, 0.0, 0.02))
        device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
        with wp.ScopedDevice(device), patch.object(incision, "load_body", side_effect=lambda *_: (points, elements)):
            viewer = ViewerNull()
            try:
                example = incision.Example(viewer, args)
                self.assertEqual(example.floor_height, -0.02)
                self.assertEqual(example.contact.ground_height, example.floor_height)
                self.assertEqual(example.contact.search_radius, body.CONTACT_SEARCH_RADIUS)
                stiffness = body.grip_stiffness(args.young_modulus, args.poisson_ratio)
                holds = example.holds
                self.assertEqual(holds.radius, body.GRIP_RADIUS)
                self.assertAlmostEqual(holds.stiffness, stiffness)
                self.assertAlmostEqual(holds.damping, body.GRIP_RELAXATION_TIME * stiffness)
                self.assertAlmostEqual(holds.maximum_speed, body.grip_speed(example.args.substeps * 60.0))
            finally:
                viewer.close()

    def test_scripted_user_of_the_self_test_works_on_the_belly(self):
        # Every scripted press and edit lies within SCRIPT_REACH of the bounding box's size
        # around its centre: on this body that rectangle is belly.
        low, high = self.positions.min(axis=0), self.positions.max(axis=0)
        centre, reach = 0.5 * (low + high), incision.SCRIPT_REACH * (high - low)
        corners = np.asarray(
            [(centre[0] + sx * reach[0], centre[1] + sy * reach[1], 0.0) for sx in (-1, 1) for sy in (-1, 1)]
        )
        own = body.asset_coordinates(corners)
        lower, upper = body.BELLY_BOX
        self.assertTrue(np.all((own[:, 0] > lower[0]) & (own[:, 0] < upper[0])))
        self.assertTrue(np.all((own[:, 1] > lower[1]) & (own[:, 1] < upper[1])))


if __name__ == "__main__":
    unittest.main()
