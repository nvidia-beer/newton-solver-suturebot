# SPDX-License-Identifier: Apache-2.0
"""The storage of the FEM system matrix after construction: no block of it is left unwritten."""

import unittest
from unittest import mock

import numpy as np
import warp as wp
from test_solver import cube, make_solver


class MatrixStorageTests(unittest.TestCase):
    def test_the_storage_of_the_system_matrix_is_zero_after_construction_beyond_the_blocks_in_use(self):
        """Fails if the reserve of ``A_bsr`` keeps what stock Warp's ``bsr_zeros`` hands out (``wp.empty``)."""
        empty = wp.empty

        def unwritten(*args, **kwargs):
            # wp.empty hands out whatever its memory held: here NaN in every float array, so that a block
            # which nothing writes afterwards shows.
            array = empty(*args, **kwargs)
            if array.size and np.issubdtype(array.numpy().dtype, np.floating):
                array.fill_(float("nan"))
            return array

        for kind in ("tet", "hex"):
            with self.subTest(kind=kind):
                model = cube(kind)
                with mock.patch.object(wp, "empty", unwritten):
                    solver = make_solver(model)
                values = solver.A_bsr.values.numpy()
                used = int(solver.A_bsr.offsets.numpy()[-1])
                # The storage has one block per triplet. The triplets of one node pair are summed into one
                # block, so most of it is a reserve that the build of the matrix does not write.
                self.assertEqual(len(values), solver.bsr_rows.size)
                self.assertGreater(used, 0)
                self.assertLess(used, len(values))
                np.testing.assert_array_equal(values[used:], np.zeros_like(values[used:]))
                # The blocks in use are the matrix: finite, and not all zero.
                self.assertTrue(np.isfinite(values[:used]).all())
                self.assertTrue(np.any(values[:used] != 0.0))


if __name__ == "__main__":
    unittest.main()
