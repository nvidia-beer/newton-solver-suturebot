# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Material correspondence and fixed authored-needle planning failure cases."""

import json
import unittest
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import numpy as np

from newton_solver_suturebot import MeshRegion, RegionMesh, boundary_faces, load_mesh_regions
from newton_solver_suturebot.examples._needle_insertion_geometry import (
    InsertionPlan,
    MaterialPoint,
    NeedleGeometry,
    TetLocator,
    pick_material_point,
)
from newton_solver_suturebot.examples._region_map import RegionMap
from newton_solver_suturebot.examples.example_fem_phantom_regions import underside_vertices


def _boxes(bounds, labels=None):
    labels = [1] * len(bounds) if labels is None else labels
    vertices, tetrahedra, owners = [], [], []
    for index, ((lower, upper), label) in enumerate(zip(bounds, labels, strict=True)):
        points = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
        points = np.asarray(lower) + points * (np.asarray(upper) - lower)
        for first, second, _ in permutations(range(3)):
            nodes = np.array([0, 1 << first, (1 << first) | (1 << second), 7])
            if np.linalg.det((points[nodes[1:]] - points[nodes[:1]]).T) < 0:
                nodes[1], nodes[2] = nodes[2], nodes[1]
            tetrahedra.append(nodes + 8 * index)
            owners.append(label)
        vertices.extend(points)
    vertices, tetrahedra, owners = np.asarray(vertices), np.asarray(tetrahedra, dtype=np.int32), np.asarray(owners)
    triangles, parents = boundary_faces(tetrahedra)
    regions = tuple(
        MeshRegion(i, f"region_{i}", f"/R{i}", (1, 1, 1), int(np.sum(owners == i)), 8) for i in sorted(set(labels))
    )
    return RegionMesh(
        vertices,
        tetrahedra,
        triangles,
        owners,
        owners[parents],
        triangles,
        owners[parents],
        np.empty(0, dtype=np.int32),
        np.arange(len(vertices)),
        np.arange(len(tetrahedra)),
        regions,
    )


def _needle():
    angles = np.linspace(0, -2.4, 33)
    points = np.c_[0.01 * np.cos(angles), 0.01 * np.sin(angles), np.zeros(len(angles))]
    return NeedleGeometry(
        Path("synthetic-test-only"),
        "/Needle",
        points,
        np.zeros(3),
        np.array((0, 0, 1)),
        0.01,
        2.4,
        0.00025,
        0.0003,
        np.array((1e-8, 1e-8, 2e-8)),
    )


def _point(x, y=0.0, z=0.0):
    return MaterialPoint(
        1, np.array((0, 1, 2)), np.array((1.0, 0.0, 0.0)), np.array((x, y, z)), np.array((0.0, 0.0, 1.0))
    )


class NeedleInsertionGeometryTests(unittest.TestCase):
    def test_shaft_sampling_preserves_capsule_segments_and_tip(self):
        needle = _needle()
        samples = needle.sample_centerline(0.00014)
        np.testing.assert_array_equal(samples[0], needle.tip)
        np.testing.assert_array_equal(samples[-1], needle.tail)
        self.assertLessEqual(np.linalg.norm(np.diff(samples, axis=0), axis=1).max(), 0.00014)
        for endpoint in needle.centerline:
            self.assertLess(np.linalg.norm(samples - endpoint, axis=1).min(), 1e-15)
        for point in samples:
            start, direction = needle.centerline[:-1], np.diff(needle.centerline, axis=0)
            t = np.clip(np.einsum("ij,ij->i", point - start, direction) / np.sum(direction**2, axis=1), 0, 1)
            self.assertLess(np.linalg.norm(point - start - t[:, None] * direction, axis=1).min(), 1e-15)
        with self.assertRaisesRegex(ValueError, "spacing"):
            needle.sample_centerline(0)

    def test_authored_geometry_preserves_proxy_shape_and_physical_values(self):
        from pxr import Usd

        needle = NeedleGeometry.load()
        self.assertEqual(needle.centerline.shape, (33, 3))
        self.assertAlmostEqual(needle.radius, 0.009449042716420122, places=9)
        self.assertAlmostEqual(needle.span, 2.3748037767486117, places=7)
        self.assertAlmostEqual(needle.thickness, 0.00028, places=9)
        self.assertAlmostEqual(needle.mass, 0.0003, places=9)
        np.testing.assert_allclose(needle.inertia, (1e-8, 1e-8, 2e-8), rtol=1e-6)
        np.testing.assert_allclose(
            np.linalg.norm(needle.centerline - needle.arc_center, axis=1), needle.radius, atol=2e-9
        )
        stage = Usd.Stage.Open(str(needle.asset_path))
        for prim in stage.Traverse():
            for relation in prim.GetRelationships():
                for target in relation.GetTargets():
                    self.assertTrue(stage.GetObjectAtPath(target), str(target))
            for attribute in prim.GetAttributes():
                for target in attribute.GetConnections():
                    self.assertTrue(stage.GetObjectAtPath(target), str(target))

    def test_map_inverse_rotation_exact_depth_and_material_deformation(self):
        mesh = _boxes([((0, 0, -0.01), (0.02, 0.01, 0))])
        # A sloping visible face demonstrates that selection is not fixed-height.
        mesh.vertices[:, 2] += mesh.vertices[:, 0] * 0.1
        region_map = RegionMap(
            mesh.vertices,
            mesh.exterior_surface_triangles,
            mesh.exterior_surface_region_ids,
            mesh.regions,
            resolution=64,
        )
        pixel = region_map.world_to_pixel(0.004, 0.003)
        np.testing.assert_allclose(region_map.pixel_to_world_xy(*pixel), (0.004, 0.003), atol=1e-15)
        self.assertGreater(pixel[0], region_map.region_ids.shape[1] / 2)
        point = pick_material_point(region_map, mesh, mesh, *pixel)
        np.testing.assert_allclose(point.reference_position, (0.004, 0.003, 0.0004), atol=1e-12)
        deformed = mesh.vertices + np.array((0.001, -0.002, 0.003))
        np.testing.assert_allclose(point.position(deformed), point.reference_position + (0.001, -0.002, 0.003))
        self.assertGreater(point.normal[2], 0.99)

    def test_map_rejects_background_and_frontmost_other_region(self):
        ref = _boxes([((0, 0, -0.01), (0.02, 0.02, 0)), ((0.004, 0.004, 0.001), (0.016, 0.016, 0.003))], [1, 2])
        selection = _boxes([((0, 0, -0.01), (0.02, 0.02, 0))])
        region_map = RegionMap(
            ref.vertices, ref.exterior_surface_triangles, ref.exterior_surface_region_ids, ref.regions, resolution=64
        )
        with self.assertRaisesRegex(ValueError, "selected region"):
            pick_material_point(region_map, ref, selection, *region_map.world_to_pixel(0.01, 0.01))
        with self.assertRaisesRegex(ValueError, "background"):
            pick_material_point(region_map, ref, selection, 0, 0)

    def test_plan_keeps_rigid_shape_follows_minor_arc_and_starts_above_pad(self):
        needle = _needle()
        plan = InsertionPlan.create(needle, _point(-0.005), _point(0.005))
        np.testing.assert_allclose(plan.tip(0), plan.entry.reference_position, atol=1e-15)
        np.testing.assert_allclose(plan.tip(plan.exit_angle), plan.exit.reference_position, atol=1e-15)
        self.assertLess(plan.tip(plan.exit_angle / 2)[2], 0)
        self.assertGreater(plan.tip(plan.initial_angle)[2], needle.thickness)
        for angle in (plan.initial_angle, 0, plan.exit_angle / 2, plan.exit_angle):
            matrix = plan.matrix(angle)
            world = needle.centerline @ matrix[:3, :3].T + matrix[:3, 3]
            np.testing.assert_allclose(world[0], plan.tip(angle), atol=1e-15)
            np.testing.assert_allclose(
                np.linalg.norm(np.diff(world, axis=0), axis=1),
                np.linalg.norm(np.diff(needle.centerline, axis=0), axis=1),
                atol=1e-15,
            )
            self.assertAlmostEqual(np.linalg.det(matrix[:3, :3]), 1.0, places=12)

    def test_impossible_chord_arc_and_coincident_selection_rejected(self):
        needle = _needle()
        with self.assertRaisesRegex(ValueError, "coincident"):
            InsertionPlan.create(needle, _point(0), _point(0))
        with self.assertRaisesRegex(ValueError, "diameter"):
            InsertionPlan.create(needle, _point(-0.011), _point(0.011))
        with self.assertRaisesRegex(ValueError, "arc length"):
            InsertionPlan.create(replace(needle, span=0.5), _point(-0.005), _point(0.005))

    def test_solid_path_passes_and_fixed_surface_rejected(self):
        mesh = _boxes([((-0.03, -0.02, -0.02), (0.03, 0.02, 0))])
        plan = InsertionPlan.create(_needle(), _point(-0.005), _point(0.005), mesh, mesh)
        self.assertGreater(plan.exit_angle, 0)
        self.assertEqual(plan.expected_passages, 1)
        self.assertEqual(plan.tissue_intervals, ((0.0, plan.exit_angle),))
        fixed_top = np.flatnonzero(mesh.vertices[:, 2] == 0)
        with self.assertRaisesRegex(ValueError, "fixed pad boundary"):
            plan.validate_material(mesh, mesh, fixed_top)

    def test_tet_occupancy_does_not_fill_wound_gap(self):
        mesh = _boxes([((-0.03, -0.02, -0.02), (-0.001, 0.02, 0)), ((0.001, -0.02, -0.02), (0.03, 0.02, 0))])
        owners, weights = TetLocator(mesh.vertices, mesh.indices).locate([(-0.005, 0, -0.001), (0, 0, -0.001)])
        self.assertGreaterEqual(owners[0], 0)
        self.assertEqual(owners[1], -1)
        self.assertAlmostEqual(weights[0].sum(), 1)
        plan = InsertionPlan.create(_needle(), _point(-0.005), _point(0.005), mesh, mesh)
        self.assertEqual(plan.expected_passages, 2)
        first, second = plan.tissue_intervals
        self.assertEqual(first[0], 0)
        self.assertEqual(second[1], plan.exit_angle)
        self.assertLess(first[1], second[0])
        np.testing.assert_allclose(plan.tip(first[1])[0], -0.001, atol=1e-12)
        np.testing.assert_allclose(plan.tip(second[0])[0], 0.001, atol=1e-12)
        self.assertIs(plan.passage_boundaries[0][0], plan.entry)
        self.assertIs(plan.passage_boundaries[-1][1], plan.exit)
        for interval, boundaries in zip(plan.tissue_intervals, plan.passage_boundaries, strict=True):
            for angle, point in zip(interval, boundaries, strict=True):
                if point is plan.entry or point is plan.exit:
                    continue  # Synthetic endpoint fixtures have no mesh binding.
                np.testing.assert_allclose(point.position(mesh.vertices), plan.tip(angle), atol=1e-12)
                np.testing.assert_allclose(
                    point.position(mesh.vertices + (0, 0, 0.001)), plan.tip(angle) + (0, 0, 0.001), atol=1e-12
                )

    def test_analytic_crossings_retain_subsample_gaps_and_thin_material(self):
        cases = (
            [((-0.03, -0.02, -0.02), (-0.00001, 0.02, 0)), ((0.00001, -0.02, -0.02), (0.03, 0.02, 0))],
            [
                ((-0.03, -0.02, -0.02), (-0.002, 0.02, 0)),
                ((-0.00001, -0.02, -0.02), (0.00001, 0.02, 0)),
                ((0.002, -0.02, -0.02), (0.03, 0.02, 0)),
            ],
        )
        for bounds, expected in zip(cases, (2, 3), strict=True):
            with self.subTest(expected_passages=expected):
                mesh = _boxes(bounds)
                plans = [
                    InsertionPlan.create(_needle(), _point(-0.005), _point(0.005), mesh, mesh, sample_spacing=spacing)
                    for spacing in (0.00005, 0.001)
                ]
                for plan in plans:
                    self.assertEqual(plan.expected_passages, expected)
                np.testing.assert_allclose(plans[0].tissue_intervals, plans[1].tissue_intervals, atol=1e-12)

    def test_bundled_wound_one_plan_has_two_passages_with_true_empty_gap(self):
        needle = NeedleGeometry.load()
        asset = needle.asset_path.parents[1] / "phantom_pad/phantom_regions.usd"
        numbering = json.loads(asset.with_name("suturebot_region_map.json").read_text())["regions"]
        stored_region = next(item["source_region_id"] for item in numbering if item["id"] == 1)
        reference = load_mesh_regions(asset)
        selected = load_mesh_regions(asset, [stored_region])
        region_map = RegionMap(
            reference.vertices,
            reference.exterior_surface_triangles,
            reference.exterior_surface_region_ids,
            reference.regions,
        )
        entry, exit_point = [
            pick_material_point(region_map, reference, selected, *region_map.world_to_pixel(x, y))
            for x, y in ((0.030, -0.040), (0.030, -0.048))
        ]
        underside = reference.global_vertex_indices[
            underside_vertices(reference.vertices, reference.exterior_surface_triangles)
        ]
        fixed = np.union1d(
            selected.fixed_vertex_indices, np.flatnonzero(np.isin(selected.global_vertex_indices, underside))
        )
        plan = InsertionPlan.create(needle, entry, exit_point, selected, reference, fixed)
        self.assertEqual(plan.expected_passages, 2)
        gap_start, gap_end = plan.tissue_intervals[0][1], plan.tissue_intervals[1][0]
        self.assertGreater((gap_end - gap_start) * needle.radius, 0.002)
        owners, _ = TetLocator(reference.vertices, reference.indices).locate(plan.tip([(gap_start + gap_end) / 2]))
        self.assertEqual(owners[0], -1, "The selected wound gap must be actual empty geometry, not inactive material")
        for boundaries in plan.passage_boundaries:
            for point in boundaries:
                np.testing.assert_allclose(point.position(selected.vertices), point.reference_position, atol=1e-12)

    def test_full_shaft_radius_rejects_nearby_inactive_surface(self):
        selection = _boxes([((-0.03, -0.02, -0.02), (0.03, 0.0001, 0))])
        reference = _boxes(
            [((-0.03, -0.02, -0.02), (0.03, 0.0001, 0)), ((-0.03, 0.0001, -0.02), (0.03, 0.02, 0))], [1, 2]
        )
        with self.assertRaisesRegex(ValueError, "inactive material"):
            InsertionPlan.create(_needle(), _point(-0.005), _point(0.005), selection, reference)


if __name__ == "__main__":
    unittest.main()
