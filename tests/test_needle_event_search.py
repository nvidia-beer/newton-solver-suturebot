# SPDX-License-Identifier: Apache-2.0
"""Surface events between the device trials and the host oracle.

Coupler options, both off by default: the device trials search past a contact force at the
puncture threshold, and the host oracle evaluates the last trial of a device search as it
stands. Most checks here run on the CPU, where every trial loop is the host-controlled one:
the forces the first option relies on, and the hand-over of a standing trial. On CUDA the
scene of the default demo is driven through its first puncture and entry with the options
off and on.
"""

import contextlib
import io
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from test_grasp_host_trial_budget import _press
from test_grasp_scene import _NOTHING_LATE, _counted_captures, _late, _scene
from test_needle_insertion_example import _REFERENCE, _example, _prepare, _selection

from newton_solver_suturebot import NeedleTissueCoupler
from newton_solver_suturebot._src.solvers.coupled.interface.needle_trial_forces import (
    FALLBACK_REASONS,
    PUNCTURE_REASON,
    SEARCHABLE_REASONS,
    _NeedleTrialForces,
)
from newton_solver_suturebot.examples.grasp._grasp_procedure import ProceduralGraspController
from newton_solver_suturebot.examples.grasp.example_fem_needle_grasp import Example as GraspExample
from newton_solver_suturebot.examples.grasp.parallel import (
    SutureSceneConfig,
    _apply_action,
    _geometry,
    _make_scene,
    _mark_startup,
    _observe,
    _snapshot,
    _step,
)
from newton_solver_suturebot.examples.needle import example_fem_needle_insertion as insertion
from newton_solver_suturebot.examples.phantom import example_fem_phantom_regions as phantom

_FIELDS = ("particle_q", "particle_qd", "joint_q", "joint_qd", "body_q", "body_qd")
_CUDA = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda


def _bits(state):
    return [getattr(state, name).numpy().copy() for name in _FIELDS]


class PunctureThresholdForceTests(unittest.TestCase):
    """What ``search_past_puncture`` relies on: the threshold changes no force of the trial."""

    def test_device_trial_forces_are_the_host_forces_below_and_above_the_threshold(self):
        with _example(tunnel_model=None) as example:
            _prepare(example)
            example.start()
            x = example.interaction
            for _ in range(80):
                if x.status == "contact":
                    break
                example.step()
            for _ in range(3):
                example.step()
            self.assertEqual((x.status, x.punctured, x.track_count), ("contact", False, 0))
            tip_force = x.last_tip_force
            self.assertGreater(tip_force, 0.0)
            state, com = example.state_0, example.coupler.needle_com
            for threshold, requests, pending in (
                (2.0 * tip_force, [], False),
                (0.5 * tip_force, [PUNCTURE_REASON], True),
            ):
                with self.subTest(threshold=threshold):
                    x.puncture_force = threshold
                    device = _NeedleTrialForces(x)
                    x.tunnel_alm.begin_step()
                    outputs = []
                    for host in (True, False):
                        state.body_f.zero_()
                        example._external_forces.zero_()
                        if host:
                            x.add_forces(
                                state,
                                example.needle_body,
                                com,
                                example._external_forces,
                                example.sim_dt,
                                trial=True,
                                implicit=True,
                            )
                            # The host evaluation notes the opening; it changes no force of this trial.
                            self.assertEqual(x._trial_fields["_pending_opening"] is not None, pending)
                            self.assertEqual(x._trial_fields["status"], "contact")
                        else:
                            x._query_positions = x._device_query_positions = x._device_query_points = None
                            device.add_forces(state, example.needle_body, com, example._external_forces, example.sim_dt)
                        outputs.append(
                            [
                                example._external_forces.numpy().copy(),
                                state.body_f.numpy().copy(),
                                x._contact_flags.numpy().copy(),
                                x._diagnostics.numpy().copy(),
                            ]
                        )
                    x.tunnel_alm.discard()
                    raised = device.fallback.numpy()
                    self.assertEqual([FALLBACK_REASONS[i] for i in np.flatnonzero(raised[1:]) + 1], requests)
                    self.assertEqual(int(raised[0]), int(bool(requests)))
                    for host_value, device_value in zip(*outputs, strict=True):
                        if _CUDA:
                            np.testing.assert_allclose(host_value, device_value, rtol=2e-6, atol=1e-10)
                        else:
                            np.testing.assert_array_equal(host_value, device_value)
                    self.assertEqual(float(outputs[0][3][6][0]), float(outputs[1][3][6][0]))
            self.assertEqual((x.puncture_count, x.punctured, x.track_count), (0, False, 0))

    def test_the_threshold_is_a_request_of_its_own_and_not_one_about_the_tip_path(self):
        self.assertIn(PUNCTURE_REASON, FALLBACK_REASONS)
        self.assertNotIn(PUNCTURE_REASON, SEARCHABLE_REASONS)


class OpeningMaskTests(unittest.TestCase):
    """The mask of an opening is taken from the faces connected to the punctured face alone."""

    def test_mask_equals_the_mask_from_the_normals_of_every_surface_face(self):
        with _example(tunnel_model=None) as example:
            _prepare(example)
            example.start()
            x = example.interaction
            triangles, vertices = x.surface_triangles, x.surface_vertices
            geometry = vertices[triangles]
            normals = np.cross(geometry[:, 1] - geometry[:, 0], geometry[:, 2] - geometry[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
            faces = np.flatnonzero(np.all(x.mapping[triangles] >= 0, axis=1))[: x.opening_capacity]
            self.assertGreater(len(faces), 1)
            sizes = []
            for face in faces:
                connected = np.any(np.isin(triangles, triangles[face]), axis=1)
                expected = (connected & (normals @ normals[face] > 0.95)).astype(np.int32)
                ids = x.mapping[triangles[face]]
                mask = x._register_opening(ids, np.full(3, 1.0 / 3.0), int(face))
                self.assertEqual(mask.dtype, np.int32)
                np.testing.assert_array_equal(mask, expected)
                self.assertEqual(mask[face], 1)
                sizes.append(int(mask.sum()))
                # The row on the device is the mask that was returned.
                row = len(x._openings) - 1
                np.testing.assert_array_equal(x._all_opening_faces.numpy()[row], expected)
            # Not every face of the surface: the masks select neighbours.
            self.assertLess(max(sizes), len(triangles))


class EventOptionTests(unittest.TestCase):
    def test_options_are_off_by_default_and_reach_the_coupler_from_the_command_line(self):
        with _example(tunnel_model=None) as example:
            _prepare(example)
            example.start()
            coupler = example.coupler
            self.assertIs(type(coupler), NeedleTissueCoupler)
            self.assertEqual((coupler.search_past_puncture, coupler.oracle_at_device_trial), (False, False))
            self.assertEqual((coupler.coupling_oracle_device_trials, coupler.coupling_oracle_device_accepts), (0, 0))
        parser = insertion.Example.create_parser()
        defaults = parser.parse_args([])
        self.assertEqual((defaults.event_search, defaults.event_oracle), ("tip-path", "restart"))
        chosen = parser.parse_args(["--event-search", "puncture", "--event-oracle", "device-trial"])
        self.assertEqual((chosen.event_search, chosen.event_oracle), ("puncture", "device-trial"))

    def test_command_line_options_set_the_coupler(self):
        with (
            TemporaryDirectory() as directory,
            wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu")),
            patch.object(phantom, "list_mesh_regions", return_value=_REFERENCE.regions),
            patch.object(phantom, "load_mesh_regions", side_effect=_selection),
        ):
            arguments = ["--asset", str(Path(directory) / "pad.usd"), "--regions", "1", "--substeps", "32"]
            arguments += ["--event-search", "puncture", "--event-oracle", "device-trial"]
            viewer = ViewerNull()
            try:
                example = insertion.Example(viewer, insertion.Example.create_parser().parse_args(arguments))
                _prepare(example)
                example.start()
                coupler = example.coupler
                self.assertEqual((coupler.search_past_puncture, coupler.oracle_at_device_trial), (True, True))
                # On the CPU every loop is the host loop: the options select nothing there.
                example.step()
                self.assertTrue(example.running, example.failure_reason)
                self.assertEqual(coupler.coupling_oracle_device_trials, 0)
            finally:
                viewer.close()

    def test_grasp_example_hands_its_surface_events_to_the_scene_and_has_no_option_for_them(self):
        from newton_solver_suturebot.examples.grasp import example_fem_needle_grasp as grasp_example
        from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene

        parser = GraspExample.create_parser()
        defaults = GraspExample._coupling(parser.parse_args([]))
        # The search past the puncture threshold and the oracle at the device trial: one way,
        # named once in the example.
        self.assertEqual(grasp_example.SURFACE_EVENTS, ("--event-search", "puncture", "--event-oracle", "device-trial"))
        self.assertEqual(defaults[4:8], grasp_example.SURFACE_EVENTS)
        # The parser of the example's scene: the insertion example's, and the tool/tissue contact
        # that the example forwards by default.
        scene_parser = GraspScene.create_parser()
        parsed = scene_parser.parse_args(["--asset", "pad.usd", *defaults])
        self.assertEqual((parsed.event_search, parsed.event_oracle), ("puncture", "device-trial"))
        # The alternatives are options of the scene, not of the example.
        names = vars(parser.parse_args([]))
        for name in ("event_search", "event_oracle"):
            self.assertNotIn(name, names)
        for option, value in (("--event-search", "tip-path"), ("--event-oracle", "restart")):
            with (
                self.subTest(option=option),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args([option, value])
        # A caller of the scene replaces them after the example's arguments: the later argument stands.
        replaced = scene_parser.parse_args(
            ["--asset", "pad.usd", *defaults, "--event-search", "tip-path", "--event-oracle", "restart"]
        )
        self.assertEqual((replaced.event_search, replaced.event_oracle), ("tip-path", "restart"))


class StandingTrialTests(unittest.TestCase):
    """The hand-over of ``oracle_at_device_trial`` on the host loop of the grasp scene.

    On the CPU there is no device search. A host loop that has converged leaves what a device
    search leaves: a candidate state computed with the trial load, no correction after it, and
    the tunnel multipliers of that trial. A second loop is handed that standing trial.
    """

    def _run(self, *, hand_over, steps=12, puncture_force=0.0019):
        with _scene() as (scene, _):
            # The tip force of the first step is about 2 mN: the pressed needle punctures.
            _press(scene, puncture_force)
            coupler, interaction = scene.coupler, scene.interaction
            solve_host, step_fem = coupler._solve_host, coupler._step_fem
            record = {"handed": 0, "first": 0, "solves": 0, "inside": False, "states": [], "events": [], "loads": []}

            def fem(*args, **kwargs):
                record["solves"] += int(record["inside"])
                return step_fem(*args, **kwargs)

            def twice(*args, **kwargs):
                converged = solve_host(*args, **kwargs)
                if not (hand_over and converged):
                    return converged
                iterations = coupler.iterations
                record["handed"] += 1
                record["inside"] = True
                try:
                    again = solve_host(*args, **{**kwargs, "device_trial": True})
                finally:
                    record["inside"] = False
                record["first"] += int(bool(again) and coupler.iterations == 1)
                coupler.iterations = iterations
                return again

            # The loop stays the scene's own: the coupler compares _step_fem with its class, so the
            # count is taken on the class, and _solve_host is not a method that selects a path.
            coupler._solve_host = twice
            original = type(coupler)._step_fem
            type(coupler)._step_fem = lambda self, *arguments, **options: fem(*arguments, **options)
            try:
                for _ in range(steps):
                    coupler.step(scene.state_0, scene.state_1, scene.control, scene.sim_dt)
                    record["states"].append(_bits(scene.state_1))
                    record["loads"].append(coupler._load.numpy().copy())
                    record["events"].append([event["kind"] for event in interaction.events])
                    scene.state_0, scene.state_1 = scene.state_1, scene.state_0
            finally:
                type(coupler)._step_fem = original
            record["punctures"] = interaction.puncture_count
            return record

    def _kept(self, *, hand_over, budget=3):
        """First step of the pressed scene, the needle free and no surface metric: the loop does not settle.

        The fixture of ``test_grasp_host_trial_budget`` whose loop only the trial budget ends
        (both jaws at the limit of their hinges, so no number here depends on how the jaws rest
        on the wire; with the needle in the seated grasp of the reset that loop converges in 3
        trials, before any of the budgets below).
        ``hand_over``: the loop that the budget ended is followed by a second one that is handed
        its kept trial.
        """
        with _scene() as (scene, _):
            _press(scene, 1.0, free=True)
            coupler = scene.coupler
            coupler.trial_budget = budget
            evaluate, solve_host = coupler._evaluate, coupler._solve_host
            # A coupler with its own _evaluate gets no surface metric (test_grasp_host_trial_budget).
            coupler._evaluate = lambda *arguments, **options: evaluate(*arguments, **options)
            record = {"second": None}

            def twice(*args, **kwargs):
                kept = solve_host(*args, **kwargs)
                if not (hand_over and kept and coupler._budget_kept):
                    return kept
                record["first"] = coupler.iterations
                again = solve_host(*args, **{**kwargs, "device_trial": True})
                record["second"] = coupler.iterations
                return again

            coupler._solve_host = twice
            coupler.step(scene.state_0, scene.state_1, scene.control, scene.sim_dt)
            record.update(keeps=coupler.coupling_budget_keeps, kept=coupler._budget_kept)
            return record

    @unittest.skipIf(_CUDA, "the host loop stands in for a device search: CPU only")
    def test_oracle_loop_that_is_handed_the_kept_trial_of_a_search_has_a_budget_of_its_own(self):
        reference = self._kept(hand_over=False)
        self.assertEqual((reference["keeps"], reference["kept"]), (1, True))
        # The oracle loop goes on from the kept trial, and its own budget ends it with a kept trial.
        own = self._kept(hand_over=True)
        self.assertGreater(own["second"], 1)
        self.assertEqual((own["keeps"], own["kept"]), (2, True))

    @unittest.skipIf(_CUDA, "bit-for-bit comparison of two runs: CPU only")
    def test_oracle_accepts_a_standing_converged_trial_in_its_first_trial_bit_for_bit(self):
        reference = self._run(hand_over=False)
        handed = self._run(hand_over=True)
        self.assertEqual(reference["punctures"], 1)
        self.assertIn(["puncture"], reference["events"])
        # Every step was handed over once, ended there in its first trial and solved no tissue step.
        self.assertEqual(
            (handed["handed"], handed["first"], handed["solves"]), (len(handed["states"]), handed["handed"], 0)
        )
        # The same surface events at the same steps, and the same accepted states and carried loads.
        self.assertEqual(handed["events"], reference["events"])
        for step, (expected, actual) in enumerate(zip(reference["states"], handed["states"], strict=True)):
            for name, a, b in zip(_FIELDS, expected, actual, strict=True):
                np.testing.assert_array_equal(a, b, err_msg=f"step {step} {name}")
        for expected, actual in zip(reference["loads"], handed["loads"], strict=True):
            np.testing.assert_array_equal(expected, actual)


class DeviceSearchSceneTests(unittest.TestCase):
    """The scene of the default demo on the device, through its first puncture and entry.

    Driven as the example drives it, once with the scene's own defaults and once with the
    device search past the puncture threshold, the oracle at the device trial it is handed
    and the response columns of one node per step. Both configurations are named in full, so
    the comparison does not follow the defaults of the example: the tool/tissue contact, which
    the example turns on, is off in both, as it is in the scene. The puncture and the entry are
    decided by the host oracle in both; with the options it evaluates device trials.
    """

    # The first puncture is accepted in frame 81 and the entry in frame 83; the response
    # columns of the first track sample are solved over the four steps that follow it.
    frames = 90
    scene_defaults = ("--event-search", "tip-path", "--event-oracle", "restart", "--response-basis-nodes", "0")
    options = ("--event-search", "puncture", "--event-oracle", "device-trial", "--response-basis-nodes", "1")
    # So does the contact between the tools and the tissue: the scene's own default, not the example's.
    contact = ("--no-tool-tissue-contact",)

    def setUp(self):
        if not _CUDA or not wp.is_conditional_graph_supported():
            self.skipTest("Device trials and their captured graphs: CUDA only")

    def _drive(self, arguments):
        # The example's own settings without its tool/tissue contact, then the scene arguments of
        # this drive: an argument that comes later replaces the one of the example.
        defaults = GraspExample.create_parser().parse_args(list(self.contact))
        config = SutureSceneConfig(
            device=os.environ["NEWTON_TEST_DEVICE"],
            control_mode="physical_grasp",
            rigid_backend=defaults.rigid_backend,
            substeps=defaults.substeps,
            example_arguments=(*GraspExample._coupling(defaults), *arguments),
            **GraspExample._bite(defaults.bite_length),
        )
        self.assertEqual(config.rigid_backend, "mujoco-warp")
        scene = _make_scene(config, 0)
        try:
            # As a worker: the first observation, then the mark that ends start-up.
            observation = _snapshot(_observe(scene, 0))
            controller = ProceduralGraspController(
                _snapshot(_geometry(scene, config)),
                config,
                defaults.speed,
                defaults.settle_seconds,
                lift_height=defaults.lift_height,
                pull_through=GraspExample._pull_through(defaults),
            )
            _mark_startup(scene)
            coupler, interaction = scene.coupler, scene.interaction
            manager, provider = scene.solver._affine_predictor, scene._response_basis_provider
            engine = provider.engine
            storage = [array.ptr for array in (engine.held, engine.ordered, engine.quality)]
            events, position = [], {"frame": 0, "substep": 0}
            step = type(coupler).step

            def recorded(self, *arguments, **options):
                # Coupled steps of the frame in the order they are attempted; a refused one has no events.
                try:
                    result = step(self, *arguments, **options)
                    events.extend(
                        (position["frame"], position["substep"], event["kind"]) for event in interaction.events
                    )
                    return result
                finally:
                    position["substep"] += 1

            # The class, not the instance: the coupler's own methods select its trial loops.
            with _counted_captures() as begun, patch.object(type(coupler), "step", recorded):
                for frame in range(self.frames):
                    position.update(frame=frame, substep=0)
                    action = controller([observation])
                    self.assertNotEqual(controller.report()[0]["phase"], "stopped", controller.report())
                    _apply_action(scene, config, action[0])
                    body_q = _step(scene)
                    self.assertTrue(scene.running, scene.failure_reason)
                    observation = _snapshot(
                        _observe(scene, 0, include_tissue=False, include_thread_diagnostics=False, body_q=body_q)
                    )
                    scene.actual_tip_history.clear()
                scene.refresh_capture_census()
                report = scene.report()
            census = report["capture_census"]
            self.assertEqual(census["status"], "checked")
            return {
                "events": events,
                "late": _late(report),
                "late_names": (census["late_sites"], census["late_modules"], census["graphs_released_after_startup"]),
                "begun": begun,
                "options": (coupler.search_past_puncture, coupler.oracle_at_device_trial),
                "oracle": (coupler.coupling_oracle_device_trials, coupler.coupling_oracle_device_accepts),
                "fallbacks": coupler.coupling_graph_fallbacks,
                "basis": (manager.nodes_per_attempt, manager.basis_builds, manager.deferred_builds),
                "storage": (
                    provider.engine is engine,
                    [array.ptr for array in (engine.held, engine.ordered, engine.quality)] == storage,
                ),
                "entries": report["entry_count"],
                "track": report["track_samples"],
            }
        finally:
            scene.close()
            scene.viewer.close()

    def test_oracle_evaluates_device_trials_and_decides_the_events_of_the_default_scene_at_their_substeps(self):
        off = self._drive(self.scene_defaults)
        on = self._drive(self.options)
        self.assertEqual((off["options"], on["options"]), ((False, False), (True, True)))
        # Both went through the first puncture and the entry behind it.
        self.assertEqual([kind for _, _, kind in off["events"]][:2], ["puncture", "entry"])
        self.assertEqual((off["entries"], on["entries"]), (1, 1))
        # The same events in the same coupled steps of the same frames.
        self.assertEqual(on["events"], off["events"])
        # Without the options every oracle loop starts with a complete host trial.
        self.assertEqual(off["oracle"], (0, 0))
        # With them the oracle evaluated device trials as they stood, and accepted some at once.
        trials, accepts = on["oracle"]
        self.assertGreater(trials, 0)
        self.assertLessEqual(accepts, trials)
        self.assertLessEqual(trials, on["fallbacks"])
        # The columns of the first track sample: at once, or over the steps after the entry.
        self.assertEqual(off["basis"][0], None)
        self.assertEqual(off["basis"][2], 0)
        self.assertEqual(on["basis"][0], 1)
        self.assertGreater(on["basis"][2], 0)
        self.assertGreater(on["basis"][1], 0)
        self.assertGreater(on["track"], 0)
        for run in (off, on):
            # Nothing was captured, loaded or released after start-up, and the provider's
            # column storage is the one start-up made.
            self.assertEqual(run["late"], _NOTHING_LATE)
            self.assertEqual(run["late_names"], ({}, {}, 0))
            self.assertEqual(run["begun"], [])
            self.assertEqual(run["storage"], (True, True))


if __name__ == "__main__":
    unittest.main()
