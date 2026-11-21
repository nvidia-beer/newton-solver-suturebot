# SPDX-License-Identifier: Apache-2.0
"""The incision example is the shared cut simulation with a body, tools and a script: same frames, grips and reserve.

The example derives from :class:`CutTissue`. Here it is built on the six-tetrahedron cube of the
cut tests and compared with a simulation of its own that gets the example's numbers (its
parser's defaults and the scene module's lengths): bit for bit on the CPU, where two runs of one
scene are the same run. The simulation without any example is in tests/test_cut_tissue.py.
"""

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from test_cut_pipeline import _cube
from test_cut_tissue import _bits, _blade, _failing_publication, _grip_top

from newton_solver_suturebot import CutTissue
from newton_solver_suturebot.examples.incision import example_fem_surgical_incision as cutting

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


@contextmanager
def _example(*arguments):
    args = cutting.Example.create_parser().parse_args(["--fragment_capacity", "64", *arguments])
    with wp.ScopedDevice(DEVICE), patch.object(cutting, "load_body", side_effect=lambda *_: _cube()):
        viewer = ViewerNull()
        try:
            yield cutting.Example(viewer, args)
        finally:
            viewer.close()


def _simulation_of_its_own(example):
    """A :class:`CutTissue` that no example is built on, with the numbers of the example's, started up."""
    args = example.args
    grips = {
        "stiffness": args.grip_stiffness,
        "damping": args.grip_damping,
        "maximum_speed": args.grip_speed,
        "radius": args.grip_radius,
    }
    tissue = CutTissue(
        example.reference_positions,
        example.reference_elements,
        frame_dt=example.frame_dt,
        substeps=example.substeps,
        solver_iterations=example.solver_iterations,
        density=args.density,
        young_modulus=args.young_modulus,
        poisson_ratio=args.poisson_ratio,
        material_damping=args.material_damping,
        linear_damping=example.linear_damping,
        min_stretch=example.min_stretch,
        max_stretch=example.max_stretch,
        gravity=args.gravity,
        floor_height=example.floor_height,
        ground_stiffness=example.ground_stiffness,
        piece_stiffness=example.piece_stiffness,
        search_radius=example.search_radius,
        ground_friction=example.ground_friction,
        grips=(grips,),
        node_copies=example.node_copies,
        drop_height=example.drop_height,
        streaming_topology=example.streaming_topology,
        fragment_capacity=example.fragment_capacity,
        surface_capacity=example.surface_capacity,
        reserve=example.reserve,
        device=example.device,
    )
    tissue.reset()
    return tissue


class CutTissueExampleTests(unittest.TestCase):
    @unittest.skipIf(DEVICE != "cpu", "Float32 assembly on CUDA uses atomics: two runs of one scene differ in rounding")
    def test_example_runs_the_bits_of_a_simulation_of_its_own_with_its_numbers(self):
        """Fails if the example's frames, edits, refused edit or Reset differ from those of the simulation it is built on."""
        with _example() as example:
            alone = _simulation_of_its_own(example)
            self.assertIs(type(alone), CutTissue)
            for frame in range(40):
                for tissue in (example, alone):
                    if frame == 10:
                        self.assertTrue(tissue.cut(_blade()), tissue.cut_error)
                        _grip_top(tissue, tissue.grip_sets[0], 1.0)
                    if frame == 20:
                        with _failing_publication(tissue) as calls:
                            self.assertFalse(tissue.cut(_blade(axis=1)))
                        self.assertEqual(len(calls), 2)
                    if frame == 30:
                        self.assertTrue(tissue.cut(_blade(axis=1)), tissue.cut_error)
                    tissue.step()
                for name, values in _bits(example).items():
                    np.testing.assert_array_equal(_bits(alone)[name], values, err_msg=f"frame {frame}: {name}")
            self.assertEqual((alone.cut_revision, alone.fragment_count), (example.cut_revision, example.fragment_count))
            self.assertEqual((example.cut_revision, example.cut_error, alone.cut_error), (2, "", ""))
            for tissue in (example, alone):
                tissue.reset()
                tissue.step()
            for name, values in _bits(example).items():
                np.testing.assert_array_equal(_bits(alone)[name], values, err_msg=f"after Reset: {name}")

    def test_mouse_grips_of_the_example_are_its_one_set_and_a_reset_makes_them_anew(self):
        """Fails if ``holds`` is not the example's one set of grips, or a Reset leaves a grip of the scene before it."""
        with _example() as example:
            self.assertIsInstance(example, CutTissue)
            self.assertEqual(len(example.grip_sets), 1)
            holds = example.holds
            self.assertIs(holds, example.grip_sets[0])
            self.assertIs(holds.step_time, example.solver.step_time)
            _grip_top(example, holds, 1.0)
            example.step()
            self.assertEqual(holds.count, 1)
            example.reset()
            self.assertIsNot(example.holds, holds)
            self.assertIs(example.holds, example.grip_sets[0])
            self.assertEqual(example.holds.count, 0)
            self.assertFalse(example.holds.enabled.numpy().any())
            example.step()
            example.test_final()

    def test_reserve_of_the_example_is_the_simulations_at_the_examples_tool_radius(self):
        """Fails if the example's ``derive_reserve`` is not the simulation's for ``RESERVE_TOOL_RADIUS``."""
        points, elements = _cube()
        radius = cutting.RESERVE_TOOL_RADIUS
        self.assertEqual(
            cutting.derive_reserve(points, elements, 1.0, (64, 34)),
            CutTissue.derive_reserve(points, elements, 1.0, (64, 34), tool_radius=radius),
        )
        # A slab of 5 mm cells: a tool five times as wide touches more tetrahedra at its fullest place.
        cells = [(x, y) for x in range(6) for y in range(5)]
        unit = (points - points.min(axis=0)) / 0.1
        slab = np.concatenate([(unit + (*cell, 0.0)) * 0.005 for cell in cells])
        slab_elements = np.concatenate([elements + 8 * index for index in range(len(cells))])
        thin = CutTissue.derive_reserve(slab, slab_elements, 1.0, tool_radius=radius)
        wide = CutTissue.derive_reserve(slab, slab_elements, 1.0, tool_radius=5.0 * radius)
        self.assertGreater(wide["column_tetrahedra"], thin["column_tetrahedra"])
        self.assertEqual(wide["crossed_tetrahedra"], thin["crossed_tetrahedra"])


if __name__ == "__main__":
    unittest.main()
