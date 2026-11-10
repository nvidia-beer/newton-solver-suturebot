"""The cutting example must use the complete welded pad, not one USD payload."""

import unittest
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

from newton_solver_suturebot.examples.example_fem_phantom_cut import load_pad

ASSET = Path(__file__).parents[1] / "src/newton_solver_suturebot/examples/assets/phantom_pad/phantom_regions.usd"


class FullPadCutGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points, cls.cells = load_pad(ASSET)
        # Independent source oracle: read all payloads and their original IDs,
        # without using the regional union loader under test.
        stage = Usd.Stage.Open(str(ASSET))
        transforms = UsdGeom.XformCache()
        units = float(UsdGeom.GetStageMetersPerUnit(stage))
        if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
            raise AssertionError("The bundled reference pad must retain its native Z-up frame")
        positions, node_ids, cells, cell_ids, exterior, interfaces = [], [], [], [], [], []
        cls.region_count = 0
        for prim in stage.Traverse():
            if prim.GetTypeName() != "TetMesh":
                continue
            cls.region_count += 1
            q = np.asarray(prim.GetAttribute("newton:referencePoints").Get(), dtype=np.float64)
            transform = np.asarray(transforms.GetLocalToWorldTransform(prim), dtype=np.float64)
            positions.append((q @ transform[:3, :3] + transform[3, :3]) * units)
            ids = np.asarray(prim.GetAttribute("suturebot:globalVertexIndices").Get(), dtype=np.int64)
            node_ids.append(ids)
            cells.append(ids[np.asarray(prim.GetAttribute("tetVertexIndices").Get(), dtype=np.int64)])
            cell_ids.append(np.asarray(prim.GetAttribute("suturebot:globalTetrahedronIndices").Get()))
            faces = ids[np.asarray(prim.GetAttribute("surfaceFaceVertexIndices").Get(), dtype=np.int64)]
            cut = np.asarray(prim.GetAttribute("suturebot:boundaryIsCutInterface").Get(), dtype=bool)
            exterior.append(faces[~cut])
            interfaces.append(faces[cut])
        cls.source_ids, first, inverse = np.unique(np.concatenate(node_ids), return_index=True, return_inverse=True)
        all_positions = np.concatenate(positions)
        cls.reference = all_positions[first]
        np.testing.assert_array_equal(all_positions, cls.reference[inverse])
        cls.source_cell_ids = np.concatenate(cell_ids)
        cls.source_cells = np.concatenate(cells)[np.argsort(cls.source_cell_ids)]
        cls.source_exterior = np.concatenate(exterior)
        cls.source_interfaces = np.concatenate(interfaces)

    def test_complete_positive_volume_and_rigid_floor_translation(self):
        self.assertEqual(self.region_count, 12)
        self.assertEqual(self.points.shape, (97489, 3))
        self.assertEqual(self.cells.shape, (413405, 4))
        np.testing.assert_array_equal(self.source_ids, np.arange(97489))
        np.testing.assert_array_equal(np.sort(self.source_cell_ids), np.arange(413405))
        np.testing.assert_array_equal(self.source_ids[self.cells], self.source_cells)
        np.testing.assert_array_equal(self.points[:, :2], self.reference[:, :2])
        shift = self.points - self.reference
        np.testing.assert_allclose(shift, np.broadcast_to([0, 0, -self.reference[:, 2].min()], shift.shape), atol=4e-18)
        self.assertEqual(float(self.points[:, 2].min()), 0.0)
        edges = self.points[self.cells[:, 1:]] - self.points[self.cells[:, :1]]
        original = self.reference[self.cells[:, 1:]] - self.reference[self.cells[:, :1]]
        determinants = np.linalg.det(edges)
        self.assertTrue(np.all(determinants > 0))
        np.testing.assert_allclose(edges, original, rtol=0, atol=4e-18)
        np.testing.assert_allclose(determinants, np.linalg.det(original), rtol=2e-12, atol=0)

    def test_only_true_exterior_faces_remain_after_welding(self):
        local_faces = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
        faces = self.source_ids[self.cells[:, local_faces].reshape(-1, 3)]
        keys, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
        self.assertTrue(np.all((counts == 1) | (counts == 2)))
        boundary = keys[counts == 1]
        self.assertEqual(len(boundary), 110956)
        expected = np.unique(np.sort(self.source_exterior, axis=1), axis=0)
        np.testing.assert_array_equal(boundary, expected)
        interfaces, interface_counts = np.unique(np.sort(self.source_interfaces, axis=1), axis=0, return_counts=True)
        self.assertEqual(len(interfaces), 11033)
        np.testing.assert_array_equal(interface_counts, 2)
        # Every artificial partition face has two volume owners. None can be
        # mistaken for a fixed or contact boundary in the full-pad topology.
        both = np.concatenate((boundary, interfaces))
        self.assertEqual(len(np.unique(both, axis=0)), len(both))


if __name__ == "__main__":
    unittest.main()
