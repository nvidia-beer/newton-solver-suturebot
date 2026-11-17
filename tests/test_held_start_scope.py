# SPDX-License-Identifier: Apache-2.0
"""Held-start caching is an explicit, guarded physical-grasp transaction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from test_combined_contact_preconditioner import native_embedded
from test_needle_insertion_example import _example
from test_needle_tissue_coupler import _native_state

from newton_solver_suturebot._src.solvers.fem.held_start import HeldFEMStart
from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene


class HeldStartScopeTests(unittest.TestCase):
    def test_scene_opt_in_checks_supported_solver_and_invalidates_previous_cache(self):
        with _example() as example:
            h, native, _ = native_embedded(example)
            scene = SimpleNamespace(
                model=example.model,
                solver=example.solver,
                rigid_solver=native,
                state_0=example.state_0,
                control=example.control,
                held_fem_start=GraspScene.held_fem_start,
            )
            GraspScene._configure_held_fem_start(scene)
            held = scene.solver._held_start
            self.assertIsInstance(held, HeldFEMStart)
            self.assertTrue(held.prepare(scene.state_0, scene.control, None, h))
            scene.held_fem_start = False
            GraspScene._configure_held_fem_start(scene)
            self.assertIsNone(scene.solver._held_start)
            self.assertFalse(held.active)
            scene.held_fem_start = True
            native.rigid_substeps = 2
            GraspScene._configure_held_fem_start(scene)
            self.assertIsNone(scene.solver._held_start)
            native.rigid_substeps = 4
            with wp.Tape():
                GraspScene._configure_held_fem_start(scene)
            self.assertIsNone(scene.solver._held_start)
            with patch.object(scene.solver, "_accumulate_forces", wraps=scene.solver._accumulate_forces):
                GraspScene._configure_held_fem_start(scene)
            self.assertIsNone(scene.solver._held_start)
            GraspScene._configure_held_fem_start(scene)
            self.assertIsInstance(scene.solver._held_start, HeldFEMStart)

    def test_real_coupler_clears_cache_after_commit_and_late_rejection(self):
        with _example() as example:
            h, native, coupler = native_embedded(example)
            held = HeldFEMStart(example.solver)
            example.solver._held_start = held
            coupler.step(example.state_0, example.state_1, example.control, h)
            self.assertEqual(held.preparations, 1)
            self.assertFalse(held.active)
            self.assertIsNone(held.capture_key())
            self.assertGreater(held.diagnostics()["cached_trial_assemblies"], 0)
            example.state_0.assign(example.state_1)
            example.solver.refresh_tangent(example.state_0.particle_q, h)
            before = _native_state(native)
            positions = example.state_0.particle_q.numpy().copy()
            with patch.object(example.interaction, "check_step_resolution", side_effect=RuntimeError("late guard")):
                with self.assertRaisesRegex(RuntimeError, "late guard"):
                    coupler.step(example.state_0, example.state_1, example.control, h)
            self.assertEqual(held.preparations, 2)
            self.assertFalse(held.active)
            self.assertIsNone(held.capture_key())
            np.testing.assert_array_equal(_native_state(native), before)
            np.testing.assert_array_equal(example.state_0.particle_q.numpy(), positions)
            self.assertTrue(held.prepare(example.state_0, example.control, None, h))
            example.solver.reset(example.state_0)
            self.assertFalse(held.active)
            self.assertIsNone(held.capture_key())

    def test_gradient_state_and_control_arrays_keep_uncached_path(self):
        with _example() as example:
            h, _, _ = native_embedded(example)
            held = HeldFEMStart(example.solver)
            self.assertTrue(held.prepare(example.state_0, example.control, None, h))
            for owner, name in (
                (example.state_0, "particle_q"),
                (example.state_0, "particle_qd"),
                (example.state_0, "particle_f"),
                (example.control, "tet_activations"),
            ):
                original = getattr(owner, name)
                setattr(owner, name, wp.clone(original, requires_grad=True))
                try:
                    with self.subTest(field=name):
                        self.assertFalse(held.prepare(example.state_0, example.control, None, h))
                        self.assertFalse(held.active)
                        self.assertIsNone(held.capture_key())
                finally:
                    setattr(owner, name, original)
            self.assertTrue(held.prepare(example.state_0, example.control, None, h))

    def test_preparation_failure_invalidates_scope_and_restores_accepted_state(self):
        with _example() as example:
            h, native, coupler = native_embedded(example)
            held = HeldFEMStart(example.solver)
            example.solver._held_start = held
            prepare = held.prepare
            before = _native_state(native)
            state = {
                name: getattr(example.state_0, name).numpy().copy()
                for name in ("particle_q", "particle_qd", "particle_f")
            }

            def fail_after_preparation(*args):
                self.assertTrue(prepare(*args))
                raise RuntimeError("preparation failed")

            with patch.object(held, "prepare", side_effect=fail_after_preparation):
                with self.assertRaisesRegex(RuntimeError, "preparation failed"):
                    coupler.step(example.state_0, example.state_1, example.control, h)
            self.assertFalse(held.active)
            self.assertIsNone(held.capture_key())
            np.testing.assert_array_equal(_native_state(native), before)
            for name, value in state.items():
                np.testing.assert_array_equal(getattr(example.state_0, name).numpy(), value)


if __name__ == "__main__":
    unittest.main()
