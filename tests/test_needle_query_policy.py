# SPDX-License-Identifier: Apache-2.0
"""Needle queries retain explicit launch overrides and graph invalidation."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from test_needle_insertion_example import _example
from test_needle_shaft_preconditioner import setup

from newton_solver_suturebot import NeedleTissueCoupler, SolverMuJoCoFEM
from newton_solver_suturebot._src import coupling_graph
from newton_solver_suturebot._src.needle_trial_forces import _NeedleTrialForces
from newton_solver_suturebot._src.solvers.mujoco_fem import trials


class NeedleQueryPolicyTests(unittest.TestCase):
    def test_actual_force_launches_use_small_defaults_and_preserve_explicit_overrides(self):
        with _example(needle_solver="partitioned") as example:
            setup(example, 0.0005, 1500)
            interaction = example.interaction
            outputs = []
            for override, expected in ((None, (1, 1)), (64, (64, 64))):
                with self.subTest(query_block_dim=override):
                    helper = _NeedleTrialForces(interaction, query_block_dim=override)
                    example.state_0.body_f.zero_()
                    example._external_forces.zero_()
                    with patch.object(wp, "launch", wraps=wp.launch) as launch:
                        helper.add_forces(
                            example.state_0, example.needle_body, example.drive.com, example._external_forces, 0.0005
                        )
                    observed = {
                        call.args[0].key: call.kwargs.get("block_dim")
                        for call in launch.call_args_list
                        if call.args and call.args[0].key in ("shaft_mask", "_surface_forces")
                    }
                    self.assertEqual(observed, dict(zip(("shaft_mask", "_surface_forces"), expected, strict=True)))
                    outputs.append(
                        (
                            helper.mask.numpy(),
                            helper.fallback.numpy(),
                            example._external_forces.numpy(),
                            example.state_0.body_f.numpy(),
                            interaction._contact_flags.numpy(),
                        )
                    )
            for original, changed in zip(*outputs, strict=True):
                # Scheduling may reorder atomic force additions on CUDA; the
                # saved actual-pad GPU audit supplies exact stationary parity.
                np.testing.assert_allclose(changed, original, rtol=1e-6, atol=1e-12)

    def test_both_joint_graph_paths_inherit_defaults_and_explicit_overrides(self):
        for name, cls in (
            ("partitioned", NeedleTissueCoupler),
            ("mujoco-fem", SolverMuJoCoFEM),
        ):
            with self.subTest(solver=name), _example(needle_solver=name) as example:
                setup(example, 0.0005, 1500)
                self.assertIsInstance(example.coupler, cls)

                def check_graph_policy(expected, solver_name=name):
                    # Only the CUDA-specific rigid backend is replaced. Construct
                    # real permission buffers through each production constructor.
                    with patch.object(coupling_graph, "MuJoCoFreeBodyGPU"):
                        fallback = coupling_graph._CapturedNeedleTrials(example.coupler)
                    self.assertEqual(
                        (fallback.permissions.shaft_block_dim, fallback.permissions.surface_block_dim), expected
                    )
                    if solver_name == "mujoco-fem":
                        with patch.object(trials, "MuJoCoFreeBodyGPU"):
                            joint = trials._SimultaneousTrials(example.coupler)
                        self.assertEqual(
                            (joint.permissions.shaft_block_dim, joint.permissions.surface_block_dim), expected
                        )

                check_graph_policy((1, 1))
                with patch.object(example.coupler, "_needle_query_block_dim", 64, create=True):
                    check_graph_policy((64, 64))
                if name == "partitioned":
                    self.assertFalse(hasattr(example.coupler, "_needle_query_block_dim"))

    def test_each_actual_launch_size_invalidates_the_capture_signature(self):
        with _example() as example:
            setup(example, 0.0005, 1500)
            helper = _NeedleTrialForces(example.interaction)
            original = helper.signature()
            for attribute in ("shaft_block_dim", "surface_block_dim"):
                with self.subTest(attribute=attribute):
                    previous = getattr(helper, attribute)
                    setattr(helper, attribute, 64)
                    self.assertNotEqual(helper.signature(), original)
                    setattr(helper, attribute, previous)
                    self.assertEqual(helper.signature(), original)


if __name__ == "__main__":
    unittest.main()
