"""Known insertion modules load at setup without hidden physical advancement."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from test_grasp_scene import _native, _scene

from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene


class GraspModulePreloadTests(unittest.TestCase):
    def setUp(self):
        wp.init()

    def fixture(self):
        return SimpleNamespace(
            device=SimpleNamespace(is_cuda=True),
            coupler=SimpleNamespace(_shaft_preconditioner=object()),
            model=SimpleNamespace(requires_grad=False),
            solver=SimpleNamespace(differentiable=False),
            interaction=SimpleNamespace(tunnel_alm=object()),
        )

    def test_only_enabled_known_modules_load_without_any_kernel_launch(self):
        scene = self.fixture()
        with patch.object(wp, "load_module") as load, patch.object(wp, "launch", side_effect=self.fail):
            GraspScene._preload_insertion_modules(scene)
        self.assertEqual(
            [call.args[0].__name__.split(".")[-1] for call in load.call_args_list],
            ["shaft_preconditioner", "tunnel_alm", "phase_predictor"],
        )
        for call in load.call_args_list:
            self.assertEqual(call.kwargs, {"device": scene.device})
        # With response-basis guesses the correction starts from the Woodbury guess, which is built with the
        # first track sample: its two modules load here, in front of the first frame.
        scene.solver._affine_predictor = object()
        with patch.object(wp, "load_module") as load, patch.object(wp, "launch", side_effect=self.fail):
            GraspScene._preload_insertion_modules(scene)
        named = [call for call in load.call_args_list if hasattr(call.args[0], "__name__")]
        self.assertEqual(
            [call.args[0].__name__.split(".")[-1] for call in named],
            ["shaft_preconditioner", "woodbury_guess", "woodbury_device", "tunnel_alm", "phase_predictor"],
        )
        for call in named:
            self.assertEqual(call.kwargs, {"device": scene.device})
        # So do the factorizations of its padded tiles, each a module of its own that a track would
        # otherwise build when it first reaches the size of that tile: after the named modules, for
        # the thread block that the tile is launched with.
        from newton_solver_suturebot._src.solvers.coupled.acceleration import woodbury_device

        kernels = load.call_args_list[len(named) :]
        self.assertEqual(load.call_args_list[: len(named)], named)
        tiles = [woodbury_device._tile(capacity) for capacity in woodbury_device._TILES]
        self.assertEqual([3 * capacity for capacity in woodbury_device._TILES], [12, 24, 48, 114, 120, 144])
        expected = [(kernel.module, block_dim) for kernel, block_dim in tiles]
        expected.append((woodbury_device.schur_compact_index.module, woodbury_device._LAUNCH_BLOCK_DIM))
        self.assertEqual([(call.args[0], call.kwargs["block_dim"]) for call in kernels], expected)
        for call in kernels:
            self.assertEqual(call.kwargs["device"], scene.device)
        scene.coupler._shaft_preconditioner = None
        scene.interaction.tunnel_alm = None
        with patch.object(wp, "load_module") as load:
            GraspScene._preload_insertion_modules(scene)
        self.assertEqual([call.args[0].__name__.split(".")[-1] for call in load.call_args_list], ["phase_predictor"])

    def test_cpu_gradient_tape_and_nonpartitioned_paths_do_not_preload(self):
        scene = self.fixture()
        with patch.object(wp, "load_module", side_effect=AssertionError("unrelated preload")):
            scene.device.is_cuda = False
            GraspScene._preload_insertion_modules(scene)
            scene.device.is_cuda = True
            scene.model.requires_grad = True
            GraspScene._preload_insertion_modules(scene)
            scene.model.requires_grad = False
            scene.solver.differentiable = True
            GraspScene._preload_insertion_modules(scene)
            scene.solver.differentiable = False
            with wp.Tape():
                GraspScene._preload_insertion_modules(scene)
            scene.coupler = None
            GraspScene._preload_insertion_modules(scene)

    def test_real_initialized_state_is_unchanged_by_repeated_module_loading(self):
        with _scene() as (scene, _):
            self.assertEqual(scene.elapsed, 0)
            self.assertEqual(scene.accepted_physics_steps, 0)
            self.assertEqual(scene.worker_thread.accepted_steps, 0)
            self.assertEqual(scene.worker_thread.physical_time_s, 0)
            native = _native(scene)
            arrays = {
                name: getattr(scene.state_0, name).numpy().copy()
                for name in ("particle_q", "particle_qd", "joint_q", "joint_qd", "body_q", "body_qd")
            }
            thread = scene.worker_thread.positions().copy()
            events = list(scene.interaction_events)
            with (
                patch.object(scene.rigid_solver, "step", side_effect=AssertionError("hidden native step")),
                patch.object(scene.solver, "step", side_effect=AssertionError("hidden FEM step")),
                patch.object(scene.worker_thread, "advance", side_effect=AssertionError("hidden thread step")),
            ):
                scene._preload_insertion_modules()
            self.assertEqual(scene.elapsed, 0)
            self.assertEqual(scene.accepted_physics_steps, 0)
            self.assertEqual(scene.interaction_events, events)
            np.testing.assert_array_equal(_native(scene), native)
            np.testing.assert_array_equal(scene.worker_thread.positions(), thread)
            for name, value in arrays.items():
                np.testing.assert_array_equal(getattr(scene.state_0, name).numpy(), value)


if __name__ == "__main__":
    unittest.main()
