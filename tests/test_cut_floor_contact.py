# SPDX-License-Identifier: Apache-2.0
"""The floor of the cut scene carries a body with the plain penalty of the surface under it, at every time step.

The six-tetrahedron cube of the cut tests, 0.1 m on a side and 1 kg, stands on the floor with
the example's defaults. A floor row is a corner of a surface triangle with a third of that
triangle's area, so the floor's reaction is k times the sum of area times depth over the corners
under the floor (on this cube the lower corners of the side walls count with the area of their
coarse triangles, beside the underside). At rest that sum times k is the weight, 9.81 N: the
corners lie 12 to 24 um deep, whatever the time step.

A contact force that is divided by its own implicit denominator, instead of standing in the
solve with that denominator, balances the same weight only where dt2 g M equals the sum of mass
times depth of the nodes under the floor: 0.4 to 0.9 mm deep at four substeps and 1.1 to 3.1 mm
at two (measured on the code as it was until 2026-11-20).
"""

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from test_cut_pipeline import _cube

from newton_solver_suturebot._src.solvers.fem.solve_diagonal import SolveDiagonal
from newton_solver_suturebot.examples.incision import example_fem_surgical_incision as cutting

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
GRAVITY, STIFFNESS = 9.81, 2.0e7
# Frames until the standing cube is at rest (three seconds), at any of the time steps below.
FRAMES = 180


@contextmanager
def _standing_cube(*arguments):
    args = cutting.Example.create_parser().parse_args(["--fragment_capacity", "64", *arguments])
    with wp.ScopedDevice(DEVICE), patch.object(cutting, "load_body", side_effect=lambda *_: _cube()):
        viewer = ViewerNull()
        try:
            yield cutting.Example(viewer, args)
        finally:
            viewer.close()


def _rest(*arguments):
    """The standing cube after ``FRAMES`` frames: what the floor rows weigh and how deep they lie."""
    with _standing_cube(*arguments) as example:
        for _ in range(FRAMES):
            example.step()
        count = int(example.surface.count.numpy()[0])
        points = example.contact.update_surface(example.state_0.particle_q).numpy()[: 3 * count].astype(np.float64)
        # One row per corner of a surface triangle: a third of the triangle's area [m2], its depth [m].
        area = np.repeat(example.contact.areas.numpy()[:count].astype(np.float64), 3) / 3.0
        depth = np.clip(example.floor_height - points[:, 2], 0.0, None)
        mass = float(example.model.cut_physical_mass.numpy().sum())
        return {
            "weight": mass * GRAVITY,
            "area": area,
            "depth": depth,
            # The floor's reaction as the contact scattered it in the last substep [N].
            "reaction": float(example.contact.forces.numpy()[:, 2].sum()),
            "rows": int(example.contact.active_rows.numpy()[0]),
            "vertical_speed": float(np.abs(example.state_0.particle_qd.numpy()[:, 2]).max()),
        }


class TestFloorCarriesTheWeight(unittest.TestCase):
    def test_a_standing_block_is_carried_by_the_plain_penalty_at_every_time_step(self):
        """Fails if the floor's reaction is not k times area times depth of the rows under it, or the depth follows dt."""
        depths = {}
        for substeps in (4, 2, 8):
            with self.subTest(substeps=substeps):
                rest = _rest("--substeps", str(substeps))
                weight = rest["weight"]
                self.assertAlmostEqual(weight, GRAVITY, delta=1.0e-3)
                # At rest: no node moves up or down by a micrometre in a frame, and the reaction is the weight.
                self.assertLess(rest["vertical_speed"], 6.0e-5)
                self.assertAlmostEqual(rest["reaction"], weight, delta=0.002 * weight)
                self.assertEqual(rest["rows"], int(np.count_nonzero(rest["depth"])))
                # The reaction is the plain penalty of the rows under the floor: nothing divides it.
                penalty = STIFFNESS * float((rest["area"] * rest["depth"]).sum())
                self.assertAlmostEqual(penalty, rest["reaction"], delta=0.002 * weight)
                depths[substeps] = rest["depth"]
        for substeps in (2, 8):
            np.testing.assert_allclose(depths[substeps], depths[4], rtol=0.01, atol=1.0e-8)

    def test_a_floor_ten_times_as_stiff_carries_the_block_a_tenth_as_deep(self):
        """Fails if the depth at rest does not follow the stiffness the floor was given."""
        soft, stiff = _rest(), _rest("--ground_stiffness", str(10.0 * STIFFNESS))
        self.assertAlmostEqual(stiff["reaction"], stiff["weight"], delta=0.002 * stiff["weight"])
        np.testing.assert_allclose(10.0 * stiff["depth"], soft["depth"], rtol=0.02, atol=1.0e-8)


class TestSolveDiagonal(unittest.TestCase):
    def solve_arrays(self, solver):
        matrix = solver.A_bsr if solver._streaming is None else solver._streaming.compact_matrix
        return matrix, solver._pc_inv_diag

    def check_blocks_stand_for_one_solve(self, *arguments):
        with _standing_cube(*arguments) as example:
            solver = example.solver
            matrix, inverse = self.solve_arrays(solver)
            count = example.model.particle_count
            offsets, columns = matrix.offsets.numpy(), matrix.columns.numpy()
            diagonal = np.array(
                [
                    offsets[node] + np.flatnonzero(columns[offsets[node] : offsets[node + 1]] == node)[0]
                    for node in range(count)
                ]
            )
            values_before, inverse_before = matrix.values.numpy().copy(), inverse.numpy().copy()
            candidate_before = solver.A_bsr.values.numpy().copy()
            blocks = np.zeros((count, 3, 3), dtype=np.float32)
            normal = np.array((0.6, 0.0, 0.8), dtype=np.float32)
            blocks[2] = 0.25 * np.outer(normal, normal)
            blocks[5] = np.diag((0.0, 0.0, 3.0e-3)).astype(np.float32)
            device_blocks = wp.array(blocks, dtype=wp.mat33, device=solver.model.device)
            diagonal_blocks = SolveDiagonal(count, solver.model.device)

            diagonal_blocks.remove(solver)
            np.testing.assert_array_equal(matrix.values.numpy(), values_before)

            diagonal_blocks.add(solver, device_blocks)
            values, inverted = matrix.values.numpy(), inverse.numpy()
            expected = values_before.copy()
            expected[diagonal] = values_before[diagonal] + blocks
            np.testing.assert_array_equal(values, expected)
            for node in (2, 5):
                product = inverted[node].astype(np.float64) @ expected[diagonal[node]].astype(np.float64)
                np.testing.assert_allclose(product, np.eye(3), atol=1.0e-5)
            untouched = np.setdiff1d(np.arange(count), (2, 5))
            np.testing.assert_array_equal(inverted[untouched], inverse_before[untouched])

            diagonal_blocks.remove(solver)
            np.testing.assert_array_equal(matrix.values.numpy(), values_before)
            np.testing.assert_array_equal(inverse.numpy(), inverse_before)
            np.testing.assert_array_equal(solver.A_bsr.values.numpy(), candidate_before)

    def test_blocks_stand_in_the_compact_matrix_of_prepared_tetrahedral_updates_and_leave_it_to_the_bit(self):
        """Fails if a block misses the matrix or the inverse block that a streaming solve reads, or is left in them."""
        self.check_blocks_stand_for_one_solve()

    def test_blocks_stand_in_the_assembled_matrix_and_leave_it_to_the_bit(self):
        """Fails if a block misses the assembled matrix or its inverse block, or is left in them."""
        self.check_blocks_stand_for_one_solve("--no-streaming_topology")

    def test_a_frame_leaves_no_contact_block_in_what_the_solve_reads(self):
        """Fails if the floor rows of a frame's last substep stay in the solver's matrix after the frame."""
        with _standing_cube() as example:
            for _ in range(30):
                example.step()
            self.assertGreater(int(example.contact.active_rows.numpy()[0]), 0)
            self.assertGreater(float(np.abs(example.contact.node_tangent.numpy()).max()), 0.0)
            solver = example.solver
            matrix, inverse = self.solve_arrays(solver)
            # What a tangent refresh at the state after the frame writes, with no contact in it.
            solver.refresh_tangent(example.state_0.particle_q, example.sim_dt)
            example.holds.add_matrix(solver, example.sim_dt)
            clean_values, clean_inverse = matrix.values.numpy().copy(), inverse.numpy().copy()
            forces = example.contact.evaluate(example.state_0.particle_q, example.state_0.particle_qd, example.sim_dt)
            example.contact.add_matrix(solver)
            # The blocks of the matrix in use (the reserve behind them is never written).
            used = int(matrix.offsets.numpy()[matrix.nrow])
            self.assertGreater(float(np.abs(matrix.values.numpy()[:used] - clean_values[:used]).max()), 0.0)
            self.assertGreater(float(np.abs(inverse.numpy() - clean_inverse).max()), 0.0)
            solver.step(example.state_0, example.state_1, example.control, None, example.sim_dt, external_forces=forces)
            example.contact.remove_matrix(solver)
            np.testing.assert_array_equal(matrix.values.numpy(), clean_values)
            np.testing.assert_array_equal(inverse.numpy(), clean_inverse)

    def test_a_solver_whose_solve_reads_other_copies_is_refused(self):
        """Fails if blocks are accepted for a solver whose preconditioner or matrix copies they would not reach."""
        with _standing_cube() as example:
            solver = example.solver
            count = example.model.particle_count
            blocks = wp.zeros(count, dtype=wp.mat33, device=solver.model.device)
            diagonal_blocks = SolveDiagonal(count, solver.model.device)
            with patch.object(solver, "preconditioner_type", "diag"), self.assertRaises(ValueError):
                diagonal_blocks.add(solver, blocks)
            with patch.object(solver, "linear_solver_chebyshev_degree", 2), self.assertRaises(ValueError):
                diagonal_blocks.add(solver, blocks)
            with patch.object(solver, "_fixed_particles", object()), self.assertRaises(ValueError):
                diagonal_blocks.remove(solver)
            with self.assertRaises(ValueError):
                diagonal_blocks.add(solver, wp.zeros(count + 1, dtype=wp.mat33, device=solver.model.device))


if __name__ == "__main__":
    unittest.main()
