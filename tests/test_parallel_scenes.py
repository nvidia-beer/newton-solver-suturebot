# SPDX-License-Identifier: Apache-2.0
"""Input contracts and opt-in physical worker isolation/reset checks.

The integration check runs the authored Wound 1 mesh, native MuJoCo needle,
and FEM solver. Enable with NEWTON_TEST_PARALLEL=1; NEWTON_TEST_DEVICE selects
the device. It does not substitute recorded motion or a small test mesh.
"""

import os
import unittest
from contextlib import contextmanager
from dataclasses import replace
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from newton_solver_suturebot._src.utils import capture
from newton_solver_suturebot.examples.grasp import parallel
from newton_solver_suturebot.examples.grasp.parallel import (
    LocalSutureScene,
    ParallelSutureScenes,
    SutureSceneConfig,
    _geometry,
)


class LocalSceneTests(unittest.TestCase):
    """One scene in the calling process: same calls as a worker, no process and no pipe."""

    def setUp(self):
        self.calls = []
        self.pose = np.arange(7.0)
        self.scene = SimpleNamespace(
            actual_tip_history=[1, 2],
            close=lambda: self.calls.append("scene.close"),
            viewer=SimpleNamespace(close=lambda: self.calls.append("viewer.close")),
        )

        @contextmanager
        def limits():
            self.calls.append("limits.enter")
            try:
                yield lambda: self.calls.append("limits.refresh")
            finally:
                self.calls.append("limits.exit")

        def observe(scene, env_id, *, include_tissue=True, include_thread_diagnostics=True, body_q=None):
            self.calls.append(("observe", env_id, include_tissue, include_thread_diagnostics, body_q))
            # A live array and a live list of the scene, as a real observation can hold.
            return {
                "env_id": env_id,
                "needle_pose": self.pose,
                "diagnostics": {"events": self.scene.actual_tip_history},
            }

        patches = (
            patch.object(parallel, "_worker_thread_limits", limits),
            patch.object(parallel, "_make_scene", lambda config, env_id: self.calls.append("make") or self.scene),
            patch.object(parallel, "_observe", observe),
            patch.object(parallel, "_apply_action", lambda scene, config, action: self.calls.append(("apply", action))),
            patch.object(parallel, "_step", lambda scene: self.calls.append("step") or "poses"),
            patch.object(
                parallel, "_reset_scene", lambda scene, config, *, rebuild=False: self.calls.append(("reset", rebuild))
            ),
            patch.object(parallel, "_geometry", lambda scene, config: {"needle_radius": 0.00028, "pose": self.pose}),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_construction_builds_and_observes_under_the_worker_thread_limits(self):
        scenes = LocalSutureScene()
        self.assertEqual(self.calls, ["limits.enter", "make", "limits.refresh", ("observe", 0, True, True, None)])
        self.assertEqual(scenes.num_envs, 1)
        self.assertEqual(scenes.config, SutureSceneConfig())
        scenes.close()

    def test_step_applies_the_action_steps_observes_and_returns_a_snapshot(self):
        scenes = LocalSutureScene()
        del self.calls[:]
        action = np.array([[0.5, 0, 0, 0, 0, -1.0]])
        result = scenes.step(action, include_tissue=False, include_thread_diagnostics=False)
        self.assertEqual(self.calls[0][0], "apply")
        np.testing.assert_array_equal(self.calls[0][1], action[0])
        # The accepted poses the frame already read are handed to the observation.
        self.assertEqual(self.calls[1:], ["step", ("observe", 0, False, False, "poses")])
        self.assertEqual(self.scene.actual_tip_history, [])
        self.assertEqual(len(result), 1)
        self.assertLessEqual(result[0]["worker_started_ns"], result[0]["worker_finished_ns"])
        # A later change of the scene's own arrays and lists does not reach the observation.
        np.testing.assert_array_equal(result[0]["needle_pose"], np.arange(7.0))
        self.pose[:] = -1
        self.scene.actual_tip_history.append(3)
        np.testing.assert_array_equal(result[0]["needle_pose"], np.arange(7.0))
        self.assertEqual(result[0]["diagnostics"]["events"], [])
        scenes.close()

    def test_invalid_actions_and_scene_ids_are_rejected_before_the_scene_is_touched(self):
        scenes = LocalSutureScene()
        del self.calls[:]
        for actions in (np.zeros(6), np.zeros((2, 6)), np.zeros((1, 7)), [[2, 0, 0, 0, 0, 0]], [[np.nan] * 6]):
            with self.subTest(actions=actions), self.assertRaisesRegex(ValueError, r"shape \(1, 6\)"):
                scenes.step(actions)
        for call in (lambda: scenes.observe([1]), lambda: scenes.reset([0, 0]), lambda: scenes.geometry(2)):
            with self.assertRaisesRegex(ValueError, "one scene"):
                call()
        self.assertEqual(self.calls, [])
        scenes.close()

    def test_reset_rebuilds_observe_passes_its_options_and_geometry_is_a_snapshot(self):
        scenes = LocalSutureScene()
        del self.calls[:]
        self.assertEqual(scenes.reset([0])[0]["env_id"], 0)
        scenes.observe(include_tissue=False)
        self.assertEqual(
            self.calls, [("reset", True), ("observe", 0, True, True, None), ("observe", 0, False, True, None)]
        )
        geometry = scenes.geometry()
        self.pose[:] = -1
        np.testing.assert_array_equal(geometry["pose"], np.arange(7.0))
        scenes.close()

    def test_close_releases_scene_viewer_and_thread_limits_once_and_ends_use(self):
        with LocalSutureScene() as scenes:
            del self.calls[:]
        self.assertEqual(self.calls, ["scene.close", "viewer.close", "limits.exit"])
        scenes.close()
        self.assertEqual(len(self.calls), 3)
        for call in (lambda: scenes.step(np.zeros((1, 6))), scenes.observe, scenes.reset, scenes.geometry):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                call()

    def test_a_failed_start_closes_what_was_built_and_releases_the_limits(self):
        def failing(scene, env_id, **options):
            raise RuntimeError("first observation failed")

        with patch.object(parallel, "_observe", failing), self.assertRaisesRegex(RuntimeError, "first observation"):
            LocalSutureScene()
        self.assertEqual(
            self.calls, ["limits.enter", "make", "limits.refresh", "scene.close", "viewer.close", "limits.exit"]
        )

    def open_startup(self, *, ready=True):
        """Leave the scene double as a reset of this module leaves a scene: with an open start-up.

        Returns the marks that the double of ``capture.finish_startup`` hands out.
        """
        self.scene._startup_begun = "begun"
        self.scene._startup_ready = ready
        self.scene._startup_status = "start-up not marked" if ready else "native rigid backend"
        self.scene.device = "device"
        self.scene.refresh_capture_census = lambda: self.calls.append("refresh")
        marks = []

        def finish(begun, device):
            self.calls.append(("mark", begun, device))
            marks.append(object())
            return marks[-1]

        item = patch.object(capture, "finish_startup", finish)
        item.start()
        self.addCleanup(item.stop)
        return marks

    def test_start_up_is_marked_after_the_first_observation_after_a_reset_and_on_request(self):
        marks = self.open_startup()
        scenes = LocalSutureScene()
        observed = ("observe", 0, True, True, None)
        marked = [("mark", "begun", "device"), "refresh"]
        self.assertEqual(self.calls, ["limits.enter", "make", "limits.refresh", observed, *marked])
        self.assertIs(self.scene._startup_mark, marks[0])
        del self.calls[:]
        scenes.reset()
        self.assertEqual(self.calls, [("reset", True), observed, *marked])
        self.assertIs(self.scene._startup_mark, marks[1])
        del self.calls[:]
        # The caller ends the start-up once more, after what it ran itself in this process.
        scenes.finish_startup()
        self.assertEqual(self.calls, marked)
        self.assertIs(self.scene._startup_mark, marks[2])
        self.assertEqual(len(marks), 3)
        scenes.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            scenes.finish_startup()

    def test_scene_that_is_not_checked_gets_no_mark_and_keeps_its_reason(self):
        marks = self.open_startup(ready=False)
        scenes = LocalSutureScene()
        scenes.reset()
        scenes.finish_startup()
        self.assertEqual(marks, [])
        self.assertEqual(self.calls.count("refresh"), 3)
        self.assertIsNone(self.scene._startup_mark)
        self.assertEqual(self.scene._startup_status, "native rigid backend")
        # A checked scene on a CPU device: the census hands out no mark, and the reason says so.
        self.scene._startup_ready = True
        with patch.object(capture, "finish_startup", lambda begun, device: None):
            scenes.finish_startup()
        self.assertIsNone(self.scene._startup_mark)
        self.assertEqual(self.scene._startup_status, "CPU device")
        scenes.close()

    def test_scene_without_a_start_up_record_is_left_alone(self):
        def finish(begun, device):
            raise AssertionError("A scene that no reset prepared has no start-up to end")

        with patch.object(capture, "finish_startup", finish):
            scenes = LocalSutureScene()
            scenes.reset()
            scenes.observe()
            scenes.finish_startup()
            scenes.close()
        self.assertEqual(
            sorted(vars(self.scene)), ["actual_tip_history", "close", "viewer"], "the double gained no census state"
        )
        # Workers end their own start-up: the pool of worker processes has nothing to do.
        self.assertIsNone(ParallelSutureScenes.finish_startup(SimpleNamespace()))

    def test_step_does_not_walk_the_module_map_and_observe_walks_it_once(self):
        from newton_solver_suturebot.examples.needle.example_fem_needle_insertion import Example

        walks = []

        def modules(mark, render_prefixes):
            self.calls.append("walk")
            walks.append((mark, render_prefixes))
            return {"module_loads_after_startup": len(walks)}

        # The scene's own refresh, on the double: it walks through capture.modules.
        self.scene._startup_mark = "mark"
        self.scene.refresh_capture_census = lambda: Example.refresh_capture_census(self.scene)
        with patch.object(capture, "modules", modules):
            scenes = LocalSutureScene()
            scenes.step(np.zeros((1, 6)))
            scenes.geometry()
            self.assertEqual(walks, [])
            del self.calls[:]
            scenes.observe(include_tissue=False)
            # The walk precedes the observation that reports it.
            self.assertEqual(self.calls, ["walk", ("observe", 0, False, True, None)])
            self.assertEqual(walks, [("mark", ("newton._src.viewer.",))])
            self.assertEqual(self.scene._capture_census, {"module_loads_after_startup": 1})
            scenes.step(np.zeros((1, 6)))
            self.assertEqual(len(walks), 1)
            scenes.observe()
            self.assertEqual(len(walks), 2)
            # Without a mark nothing is walked and nothing is kept.
            self.scene._startup_mark = None
            scenes.observe()
            self.assertEqual(len(walks), 2)
            self.assertIsNone(self.scene._capture_census)
            scenes.close()

    def test_worker_marks_and_refreshes_where_the_local_scene_does(self):
        marks = self.open_startup()
        commands = [("observe", (True, True)), ("step", (np.zeros(6), False, True)), ("reset", None), ("close", None)]
        sent = []

        def send(message):
            self.calls.append("send")
            sent.append(message)

        connection = SimpleNamespace(
            recv=lambda: commands.pop(0), send=send, close=lambda: self.calls.append("connection.close")
        )
        parallel._worker(connection, SutureSceneConfig(), 0, None)
        self.assertEqual([success for success, _ in sent], [True] * 4)
        calls = [call for call in self.calls if not (isinstance(call, tuple) and call[0] == "apply")]
        observed = ("observe", 0, True, True, None)
        marked = [("mark", "begun", "device"), "refresh"]
        self.assertEqual(
            calls,
            [
                "limits.enter",
                "make",
                "limits.refresh",
                # Start-up ends after the first observation and before it is sent.
                observed,
                *marked,
                "send",
                # An explicit observation refreshes the module numbers first; a step does not.
                "refresh",
                observed,
                "send",
                "step",
                ("observe", 0, False, True, "poses"),
                "send",
                ("reset", True),
                observed,
                *marked,
                "send",
                "scene.close",
                "viewer.close",
                "connection.close",
                "limits.exit",
            ],
        )
        self.assertEqual(len(marks), 2)
        self.assertIs(self.scene._startup_mark, marks[1])


class ParallelSceneConfigurationTests(unittest.TestCase):
    def test_physical_default_stitch_matches_recording_and_free_needle_keeps_demo(self):
        from newton_solver_suturebot.examples.grasp._dvrk_layout import load_source_stitch

        stitch = load_source_stitch()
        physical = SutureSceneConfig(control_mode="physical_grasp")
        self.assertEqual(physical.region, stitch["region_id"])
        np.testing.assert_array_equal(physical.entry_xy, stitch["entry"][:2])
        np.testing.assert_array_equal(physical.exit_xy, stitch["exit"][:2])
        legacy = SutureSceneConfig()
        self.assertEqual(legacy.entry_xy, (0.0, -0.046))
        self.assertEqual(legacy.exit_xy, (0.0, -0.054))

    def test_explicit_and_custom_mesh_points_are_not_replaced_by_recording(self):
        from newton_solver_suturebot.examples.grasp._dvrk_layout import load_source_stitch

        explicit = SutureSceneConfig(control_mode="physical_grasp", entry_xy=(0.01, 0.02), exit_xy=(0.03, 0.04))
        self.assertEqual(explicit.entry_xy, (0.01, 0.02))
        self.assertEqual(explicit.exit_xy, (0.03, 0.04))
        partial = SutureSceneConfig(control_mode="physical_grasp", entry_xy=(0.01, 0.02))
        self.assertEqual(partial.entry_xy, (0.01, 0.02))
        np.testing.assert_array_equal(partial.exit_xy, load_source_stitch()["exit"][:2])
        custom = SutureSceneConfig(control_mode="physical_grasp", asset="custom.usd")
        self.assertEqual(custom.entry_xy, (0.0, -0.046))
        self.assertEqual(custom.exit_xy, (0.0, -0.054))

    def test_geometry_exports_full_pad_binding_by_vertex_identity_not_coordinates(self):
        from test_needle_insertion_example import _example

        for region, center in ((1, 0.0), (2, 0.1)):
            with self.subTest(region=region), _example() as scene:
                scene.select_regions([region])
                entry, exit_point = (center - 0.005, 0), (center + 0.005, 0)
                scene.select_point_xy("entry", *entry)
                scene.select_point_xy("exit", *exit_point)
                scene.preview()
                reference = scene.reference
                # A distinct full-pad vertex may share coordinates with an
                # active node. Only global IDs can identify the right one.
                duplicate = scene.selection.vertices[0]
                scene.reference = replace(
                    reference,
                    vertices=np.vstack((reference.vertices, duplicate)),
                    global_vertex_indices=np.r_[
                        reference.global_vertex_indices, reference.global_vertex_indices[-1] + 17
                    ],
                )
                lookup = {
                    int(global_id): index for index, global_id in enumerate(scene.reference.global_vertex_indices)
                }
                expected = np.array([lookup[int(global_id)] for global_id in scene.selection.global_vertex_indices])
                config = SutureSceneConfig(
                    region=region, entry_xy=entry, exit_xy=exit_point, substeps=scene.args.substeps
                )
                exported = _geometry(scene, config)
                mapping = exported["active_reference_indices"]
                self.assertTrue(np.issubdtype(mapping.dtype, np.integer))
                np.testing.assert_array_equal(mapping, expected)
                np.testing.assert_array_equal(
                    exported["reference_positions"][mapping], exported["tissue_rest_positions"]
                )
                self.assertNotEqual(mapping[0], len(scene.reference.vertices) - 1)
                # Exported arrays must not expose mutable scene identities.
                mapping[:] = -1
                np.testing.assert_array_equal(_geometry(scene, config)["active_reference_indices"], expected)

    def test_invalid_time_scales_coordinates_and_counts_fail_before_starting_workers(self):
        for arguments in (
            {"substeps": 0},
            {"region": 0},
            {"frame_dt": 0},
            {"translation_step": np.inf},
            {"rotation_step": -1},
            {"entry_xy": (np.nan, 0)},
            {"entry_xy": (0,)},
            {"exit_xy": (0, 0, 0)},
            {"reset_position_noise_m": -1},
            {"reset_position_noise_m": np.inf},
            {"seed": -1},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                SutureSceneConfig(**arguments)
        for arguments in ({"num_envs": 0}, {"num_envs": 1.5}, {"timeout": 0}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ParallelSutureScenes(**arguments)


@unittest.skipUnless(os.environ.get("NEWTON_TEST_PARALLEL") == "1", "Opt-in authored multi-process physics workload")
class ParallelScenePhysicsTests(unittest.TestCase):
    def test_seeded_resets_are_reproducible_independent_and_reset_only(self):
        config = SutureSceneConfig(
            device=os.environ.get("NEWTON_TEST_DEVICE", "cpu"), reset_position_noise_m=0.0007, seed=42
        )
        records = []
        for _ in range(2):
            with TemporaryDirectory() as logs, ParallelSutureScenes(config, num_envs=2, log_dir=logs) as scenes:
                initial = scenes.observe()
                canonical = scenes.geometry()["ready_pose"]
                self.assertFalse(np.array_equal(initial[0]["needle_pose"], initial[1]["needle_pose"]))
                for observation in initial:
                    offset = observation["needle_pose"][:3] - canonical[:3]
                    self.assertLessEqual(np.max(np.abs(offset)), config.reset_position_noise_m + 1e-8)
                    self.assertGreaterEqual(offset[2], -1e-8)
                    np.testing.assert_allclose(observation["target_pose"], observation["needle_pose"], atol=1e-8)
                    self.assertEqual(observation["diagnostics"]["accepted_physics_steps"], 0)
                scenes.step(np.zeros((2, 6)))
                unselected = scenes.observe([1])[0]
                reset = scenes.reset([0])[0]
                after = scenes.observe([1])[0]
                np.testing.assert_array_equal(after["needle_pose"], unselected["needle_pose"])
                self.assertEqual(after["reset_count"], 0)
                self.assertEqual(reset["reset_count"], 1)
                self.assertEqual(reset["diagnostics"]["physical_time_s"], 0)
                self.assertEqual(reset["diagnostics"]["entry_count"], 0)
                self.assertFalse(np.array_equal(reset["needle_pose"], initial[0]["needle_pose"]))
                records.append((initial[0]["needle_pose"], initial[1]["needle_pose"], reset["needle_pose"]))
        for first, second in zip(*records, strict=True):
            np.testing.assert_array_equal(first, second)

    def test_actions_affect_only_their_scene_and_selective_reset_preserves_other_history(self):
        config = SutureSceneConfig(device=os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        with TemporaryDirectory() as logs, ParallelSutureScenes(config, num_envs=2, log_dir=logs) as scenes:
            initial = scenes.observe()
            compact = scenes.observe(include_tissue=False)
            for full, small in zip(initial, compact, strict=True):
                self.assertNotIn("tissue_positions", small)
                self.assertNotIn("tissue_velocities", small)
                np.testing.assert_array_equal(small["needle_pose"], full["needle_pose"])
                np.testing.assert_allclose(small["entry_position"], full["entry_position"], rtol=0, atol=1e-8)
                np.testing.assert_allclose(small["exit_position"], full["exit_position"], rtol=0, atol=1e-8)
            geometry = scenes.geometry()
            self.assertNotEqual(initial[0]["pid"], initial[1]["pid"])
            self.assertGreater(len(geometry["tissue_tetrahedra"]), 1000)
            actions = np.zeros((2, 6))
            actions[0, 0] = 0.2
            result = scenes.step(actions)
            for _ in range(2):
                result = scenes.step(np.zeros((2, 6)))
            for value in result:
                self.assertIsNone(value["diagnostics"]["failure_reason"])
                self.assertGreater(value["diagnostics"]["accepted_physics_steps"], 0)
                self.assertTrue(np.isfinite(value["tissue_positions"]).all())
                fixed = geometry["fixed_particle_indices"]
                np.testing.assert_array_equal(value["tissue_positions"][fixed], initial[0]["tissue_positions"][fixed])
            delta0 = result[0]["needle_pose"][0] - initial[0]["needle_pose"][0]
            delta1 = result[1]["needle_pose"][0] - initial[1]["needle_pose"][0]
            self.assertGreater(delta0, 0.5 * config.translation_step * actions[0, 0])
            self.assertLess(abs(delta1), 1.0e-7)
            before_reset = scenes.observe([1])[0]
            reset = scenes.reset([0])[0]
            after_reset = scenes.observe([1])[0]
            self.assertEqual(reset["diagnostics"]["physical_time_s"], 0.0)
            self.assertEqual(reset["diagnostics"]["entry_count"], 0)
            np.testing.assert_array_equal(reset["needle_pose"], initial[0]["needle_pose"])
            np.testing.assert_array_equal(reset["tissue_positions"], initial[0]["tissue_positions"])
            for name in ("needle_pose", "needle_velocity", "tissue_positions", "tissue_velocities", "target_pose"):
                np.testing.assert_array_equal(after_reset[name], before_reset[name], err_msg=name)
            self.assertEqual(after_reset["diagnostics"], before_reset["diagnostics"])
            with self.assertRaises(ValueError):
                scenes.step(np.full((2, 6), np.nan))
            with self.assertRaises(ValueError):
                scenes.reset([0, 0])


if __name__ == "__main__":
    unittest.main()
