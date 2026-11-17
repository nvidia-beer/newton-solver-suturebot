# SPDX-License-Identifier: Apache-2.0
"""Start-up of fem_needle_insertion, and of the coupler on the native free-needle backend, on the two test blocks."""

import io
import os
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import warp as wp
from _startup_support import final_line, launch_late_kernel, marked_on_every_device, run_script
from newton.viewer import ViewerNull
from test_needle_insertion_example import _REFERENCE, _selection
from warp._src import context as warp_context

from newton_solver_suturebot import NeedleTissueCoupler
from newton_solver_suturebot._src.solvers.coupled.acceleration import shaft_preconditioner
from newton_solver_suturebot._src.solvers.coupled.interface import needle_trial_forces
from newton_solver_suturebot._src.utils import capture
from newton_solver_suturebot.examples.needle import example_fem_needle_insertion as insertion
from newton_solver_suturebot.examples.phantom import example_fem_phantom_regions as phantom

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
CUDA = wp.get_device(DEVICE).is_cuda
# Entry and exit on the first test block [m], as the other tests of the example use them.
ENTRY, EXIT = (-0.005, 0.0), (0.005, 0.0)
POINTS = ("--entry-xy", str(ENTRY[0]), str(ENTRY[1]), "--exit-xy", str(EXIT[0]), str(EXIT[1]))
STATE_FIELDS = ("particle_q", "particle_qd", "particle_f", "joint_q", "joint_qd", "body_q", "body_qd", "body_f")


@contextmanager
def _scene(*arguments, cls=insertion.Example, startup=True):
    """The example on the two test blocks with one step per frame; ``startup=None`` builds it with no argument."""
    with (
        TemporaryDirectory() as directory,
        wp.ScopedDevice(DEVICE),
        patch.object(phantom, "list_mesh_regions", return_value=_REFERENCE.regions),
        patch.object(phantom, "load_mesh_regions", side_effect=_selection),
    ):
        pad = str(Path(directory) / "pad.usd")
        args = cls.create_parser().parse_args(["--asset", pad, "--regions", "1", "--substeps", "1", *arguments])
        viewer = ViewerNull()
        try:
            with redirect_stdout(io.StringIO()):
                example = cls(viewer, args) if startup is None else cls(viewer, args, startup=startup)
            yield example
        finally:
            viewer.close()


def _points(example):
    example.select_point_xy("entry", *ENTRY)
    example.select_point_xy("exit", *EXIT)


def _frames(example, count):
    with redirect_stdout(io.StringIO()):
        for _ in range(count):
            example.step()
            example.render()


def _loaded(load):
    """(module, thread block) of every call of a patched ``wp.load_module``; a module of the package by its short name."""
    result = []
    for call in load.call_args_list:
        module = call.args[0]
        name = module if isinstance(module, str) else module.__name__
        name = name.removeprefix("newton_solver_suturebot._src.").removeprefix("solvers.")
        result.append((name, call.kwargs["block_dim"]))
    return result


@contextmanager
def _requested_builds(device):
    """Record the kernel module builds that Warp is asked for on ``device``: (module name, block size, hash).

    At ``Module.load``, which every launch and every load by name goes through, whether the
    build is loaded already or not: what a scene asks for does not depend on what another
    scene of the process has loaded.
    """
    builds = set()
    load = warp_context.Module.load

    def recorded(module, target, *arguments, **options):
        executable = load(module, target, *arguments, **options)
        if executable is not None and executable.device == device:
            builds.add((module.name, executable.block_dim, executable.module_hash))
        return executable

    with patch.object(warp_context.Module, "load", recorded):
        yield builds


def _bits(example):
    """The accepted state, the native integration state and the task's counts, for bit comparisons."""
    solver = example.rigid_solver
    native = np.empty(mujoco.mj_stateSize(solver.mj_model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(solver.mj_model, solver.mj_data, native, mujoco.mjtState.mjSTATE_INTEGRATION)
    result = {name: getattr(example.state_0, name).numpy().copy() for name in STATE_FIELDS}
    result["native"] = native
    result["increment"] = example.solver.velocity_increment.numpy().copy()
    interaction = example.interaction
    result["counts"] = np.array(
        [
            example.accepted_physics_steps,
            example.step_refinements,
            0 if interaction is None else interaction.puncture_count,
            0 if interaction is None else interaction.entry_count,
            0 if interaction is None else interaction.exit_count,
            0 if interaction is None else interaction.track_count,
        ]
    )
    return result


def _sliders(example, frames=260):
    """Automatic entry, then the sliders out of the tissue and in again; returns the flips of the shaft flag."""
    reach, flips, engaged = None, 0, False
    with redirect_stdout(io.StringIO()):
        for frame in range(frames):
            if frame == 70:
                example.use_manual_control()
                reach = float(example.manual_position[0])
            if 70 <= frame < 130:
                example.set_manual_target(insertion=reach * (1.0 - (frame - 69) / 60))
            if 150 <= frame < 210:
                example.set_manual_target(insertion=reach * ((frame - 149) / 60))
            example.step()
            example.render()
            now = example.interaction.engaged_track_length > 0
            flips += now != engaged
            engaged = now
    return flips


class StartupRuleTests(unittest.TestCase):
    def test_example_that_was_not_asked_keeps_its_graphs_for_the_run(self):
        with _scene(startup=None) as example:
            self.assertIsNone(example.startup)
            self.assertIsNone(example.script)
            _points(example)
            example.preview()
            self.assertIsNone(example.interaction)
            self.assertIsNone(example.coupler)
            with patch.object(example.solver, "refresh_tangent", side_effect=AssertionError("hidden tangent refresh")):
                self.assertFalse(example.capture_graphs())
            self.assertEqual(example._startup_status, "native rigid backend")
            self.assertIsNone(example._startup_tangent)
            example.start()
            self.assertIs(type(example.coupler), NeedleTissueCoupler)
            with patch.object(example.coupler, "capture_graphs", side_effect=AssertionError("hidden attempt")):
                self.assertFalse(example.capture_graphs())
            _frames(example, 2)
            self.assertTrue(example.running, example.failure_reason)
            self.assertEqual(example.report()["capture_census"]["status"], "native rigid backend")
            example.test_final()

    def test_scene_built_on_the_example_never_owns_a_start_up(self):
        class Scene(insertion.Example):
            pass

        with _scene(startup=True, cls=Scene) as scene:
            self.assertIsNone(scene.startup)
            self.assertIsNone(scene.script)
            _points(scene)
            scene.preview()
            self.assertIsNone(scene.interaction)
            scene.start()
            with patch.object(scene.coupler, "capture_graphs", side_effect=AssertionError("hidden attempt")):
                self.assertFalse(scene.capture_graphs())
            self.assertEqual(scene._startup_status, "native rigid backend")
            _frames(scene, 2)
            self.assertTrue(scene.running, scene.failure_reason)

    def test_start_up_runs_at_preview_and_start_builds_nothing(self):
        with _scene() as example:
            startup = example.startup
            self.assertEqual(startup.startups, 0 if not CUDA else 1)
            self.assertIsNone(example.interaction)
            _points(example)
            self.assertIsNone(example.interaction)
            begins = [0]
            begin = startup.begin

            def counted():
                begins[0] += 1
                return begin()

            with patch.object(startup, "begin", side_effect=counted):
                example.preview()
                # The coupled step of the planned path exists, and the tangent of the first frame.
                coupler, interaction = example.coupler, example.interaction
                self.assertIs(type(coupler), NeedleTissueCoupler)
                self.assertEqual(begins[0], 1)
                self.assertEqual(example._startup_tangent, (example.solver._tangent_version, example.sim_dt, 0))
                if CUDA:
                    self.assertTrue(example._startup_ready, example._startup_status)
                    self.assertEqual(startup.status, "checked")
                else:
                    self.assertFalse(example._startup_ready)
                    self.assertEqual(example._startup_status, "CPU device")
                    self.assertEqual(startup.status, "CPU device")
                # Start builds nothing and opens no start-up.
                captures = capture.census.captures, capture.census.solver_captures
                with patch.object(example, "_build_coupling", side_effect=AssertionError("built at Start")):
                    example.toggle_running()
                self.assertTrue(example.running)
                self.assertIs(example.coupler, coupler)
                self.assertEqual((capture.census.captures, capture.census.solver_captures), captures)
                self.assertEqual(begins[0], 1)
                _frames(example, 2)
                self.assertTrue(example.running, example.failure_reason)
                self.assertIsNone(example._startup_tangent)
                # A reset is a start-up; the path and its coupled step are gone until the next Preview.
                example.reset()
                self.assertEqual(begins[0], 2)
                self.assertIsNone(example.interaction)
                self.assertIsNone(example.coupler)
                _points(example)
                example.preview()
                self.assertEqual(begins[0], 3)
                self.assertIsNot(example.coupler, coupler)
                self.assertIsNot(example.interaction, interaction)
                # The same path once more: its coupled step stands, and no start-up is opened.
                coupler = example.coupler
                example.preview()
                self.assertEqual(begins[0], 3)
                self.assertIs(example.coupler, coupler)
                # Another point: the next Preview builds the step of the new path.
                example.select_point_xy("exit", EXIT[0], 0.002)
                self.assertIsNone(example.interaction)
                example.preview()
                self.assertEqual(begins[0], 4)
                self.assertIsNot(example.coupler, coupler)

    def test_start_up_without_a_path_names_the_kernel_modules_of_the_coupled_step(self):
        # By name and with no launch, each at the thread block of its launches; the trial-force
        # helper and the correction of the shaft name their own.
        scene = SimpleNamespace(device=SimpleNamespace(is_cuda=True), solver=object())
        with (
            patch.object(wp, "load_module") as load,
            patch.object(wp, "launch", side_effect=self.fail),
            patch.object(needle_trial_forces, "load_kernel_modules") as helper,
            patch.object(shaft_preconditioner, "load_kernels") as correction,
        ):
            insertion.Example._load_coupling_modules(scene)
        self.assertEqual(
            _loaded(load),
            [
                ("coupled.coupling", 256),
                ("mujoco.mujoco_free_body", 256),
                ("coupled.tet_step", 256),
                ("fem.implicit_predictor", 256),
                ("coupled.coupling_graph", 256),
                ("mujoco.mujoco_free_body_gpu", 256),
                ("coupled.host_trial_forces", 256),
                ("warp.sparse", 256),
            ],
        )
        for call in load.call_args_list:
            self.assertIs(call.kwargs["device"], scene.device)
        helper.assert_called_once_with(scene.device)
        correction.assert_called_once_with(scene.device)
        # Warp registers the kernels of its sparse file under that name: none of them is created per type.
        sparse = warp_context.user_modules["warp.sparse"]
        self.assertGreater(len(sparse.kernels), 0)
        self.assertFalse(any(kernel.is_generic for kernel in sparse.kernels.values()))
        # The builds of the helper, in the order of its launches; one sample per block for the queries.
        with patch.object(wp, "load_module") as load, patch.object(wp, "get_device", lambda device: device):
            needle_trial_forces.load_kernel_modules(scene.device)
        self.assertEqual(
            _loaded(load),
            [
                ("coupled.interface.needle_trial_forces", 256),
                ("coupled.interface.needle_trial_forces", 1),
                ("coupled.interface.insertion", 256),
                ("coupled.interface.insertion", 1),
                ("coupled.interface.surface_contact_forces", 1),
                ("coupled.interface.surface_facet_scratch", 1),
                ("geometry.needle_queries", 256),
                ("geometry.mesh_refit", 256),
                ("geometry.mesh_refit", 32),
                ("coupled.interface.tunnel_alm", 256),
                ("coupled.interface.shaft_projected_kernels", 256),
                ("coupled.interface.shaft_projection", 256),
                ("coupled.interface.shaft_projection_tables", 256),
            ],
        )
        # Nothing on a CPU device, and nothing without tissue.
        with patch.object(wp, "load_module", side_effect=AssertionError("module loaded")):
            needle_trial_forces.load_kernel_modules("cpu")
            shaft_preconditioner.load_kernels("cpu")
            with (
                patch.object(needle_trial_forces, "load_kernel_modules", side_effect=AssertionError("helper")),
                patch.object(shaft_preconditioner, "load_kernels", side_effect=AssertionError("correction")),
            ):
                scene.solver = None
                insertion.Example._load_coupling_modules(scene)
                scene.solver, scene.device.is_cuda = object(), False
                insertion.Example._load_coupling_modules(scene)

    def test_start_up_with_a_path_leaves_the_kernel_modules_of_the_coupled_step_to_its_start_up_call(self):
        # With the points on the command line the coupled step is built in the constructor: its
        # start-up call loads what it launches, as before, and a reset, which ends without a
        # path, names the modules.
        with patch.object(insertion.Example, "_load_coupling_modules", autospec=True) as named:
            with _scene(*POINTS) as example:
                self.assertIsNotNone(example.interaction)
                named.assert_not_called()
                example.reset()
                self.assertIsNone(example.interaction)
                named.assert_called_once_with(example)
            with _scene() as example:
                self.assertIsNone(example.interaction)
                self.assertEqual(named.call_count, 2)
                _points(example)
                example.preview()
                self.assertIsNotNone(example.interaction)
                self.assertEqual(named.call_count, 2)

    def test_points_on_the_command_line_start_up_in_the_constructor(self):
        with _scene(*POINTS, "--start") as example:
            self.assertIs(type(example.coupler), NeedleTissueCoupler)
            self.assertTrue(example.running)
            self.assertEqual(example.startup.closed, 0)
            self.assertEqual(example._startup_tangent, (example.solver._tangent_version, example.sim_dt, 0))
            self.assertEqual(example._startup_ready, CUDA)
            _frames(example, 2)
            self.assertTrue(example.running, example.failure_reason)

    def test_start_without_a_preview_runs_the_start_up_first(self):
        with _scene() as example:
            _points(example)
            self.assertIsNone(example.plan)
            example.toggle_running()
            self.assertTrue(example.running)
            self.assertIsNotNone(example.plan)
            self.assertIs(type(example.coupler), NeedleTissueCoupler)
            self.assertEqual(example._startup_ready, CUDA)
            self.assertEqual(example.startup.closed, 1 if CUDA else 0)

    def test_first_frames_are_bit_identical_with_and_without_the_start_up_call(self):
        # Through the puncture, the entry and the first step with an engaged shaft (frame 58).
        if CUDA:
            self.skipTest("Two runs of one scene are not the same bits on a CUDA device: see the test below")
        with _scene(startup=None) as plain, _scene() as started:
            for example in (plain, started):
                _points(example)
                example.preview()
            self.assertIsNone(plain.interaction)
            self.assertIsNotNone(started.interaction)
            before = _bits(started)
            for name, values in _bits(plain).items():
                if name != "counts":
                    np.testing.assert_array_equal(before[name], values, err_msg=f"after Preview: {name}")
            for example in (plain, started):
                example.start()
            for frame in range(64):
                _frames(plain, 1)
                _frames(started, 1)
                expected, actual = _bits(plain), _bits(started)
                for name, values in expected.items():
                    np.testing.assert_array_equal(actual[name], values, err_msg=f"frame {frame}: {name}")
            self.assertEqual(started.interaction.entry_count, 1)
            self.assertGreater(started.interaction.engaged_track_length, 0)

    @unittest.skipUnless(CUDA, "The start-up runs real steps and discards them only on a CUDA device")
    def test_start_up_changes_the_first_frames_no_more_than_a_second_run_does(self):
        # On a CUDA device two runs of the same scene differ in their last bits: the test block is
        # not exactly at rest in float32, its free nodes creep by about 1.4e-9 m in the first
        # frames, and the tissue forces are summed by atomic adds whose order is not fixed. The
        # start-up runs real steps and gives the state back; what it leaves must be within that
        # difference: a scene with the start-up call is compared with one without it, and that
        # one with a second run of itself, over the frames before the needle touches the tissue.
        def difference(a, b):
            return max(
                float(np.abs(a[name].astype(np.float64) - b[name].astype(np.float64)).max())
                for name in a
                if name != "counts"
            )

        with _scene(startup=None) as plain, _scene(startup=None) as again, _scene() as started:
            scenes = (plain, again, started)
            for example in scenes:
                _points(example)
                example.preview()
            # After the start-up call, before any frame: the same bits as a scene without it.
            before = _bits(plain)
            for name, values in _bits(started).items():
                if name != "counts":
                    np.testing.assert_array_equal(values, before[name], err_msg=f"after Preview: {name}")
            for example in scenes:
                example.start()
            runs = started_runs = 0.0
            for frame in range(40):
                for example in scenes:
                    _frames(example, 1)
                reference, second, with_startup = (_bits(example) for example in scenes)
                np.testing.assert_array_equal(with_startup["counts"], reference["counts"], err_msg=f"frame {frame}")
                np.testing.assert_array_equal(with_startup["native"], reference["native"], err_msg=f"frame {frame}")
                runs = max(runs, difference(reference, second))
                started_runs = max(started_runs, difference(reference, with_startup))
            print(
                f"largest difference of the accepted state and the velocity increment over 40 frames [SI]: two runs "
                f"without the start-up call {runs:.3e}, without and with it {started_runs:.3e}",
                flush=True,
            )
            # Far below the float32 resolution of a coordinate of the block (1.9e-9 m at 20 mm).
            self.assertLess(started_runs, 1.0e-11)
            self.assertLess(runs, 1.0e-11)

    def test_report_carries_the_census_of_the_start_ups(self):
        with _scene(*POINTS) as example:
            example.refresh_capture_census()
            report = example.report()
            census = report["capture_census"]
            self.assertEqual(census["status"], "checked" if CUDA else "CPU device")
            self.assertEqual(census["startups"], 1 if CUDA else 0)
            if CUDA:
                self.assertEqual(report["captures_after_startup"], 0)
                self.assertEqual(census["startup_unlaunched_graphs"], 0)
                self.assertGreater(census["startup_captures"], 0)


class SelfTestScriptTests(unittest.TestCase):
    def test_script_is_empty_with_points_on_the_command_line(self):
        with _scene("--test", *POINTS, "--start") as example:
            self.assertEqual(example.script.frames, 0)
            _frames(example, 3)
            line = final_line(example)
            self.assertEqual(line["script_frames"], [3, 0])
            self.assertEqual(line["scripted_insertions"], [])
            self.assertEqual(line["capture_census_status"], "checked" if CUDA else "CPU device")

    def test_script_leaves_a_run_alone_that_somebody_else_drives(self):
        with _scene("--test") as example:
            self.assertEqual(example.script.frames, 365)
            _frames(example, 5)
            example.select_point_xy("entry", *ENTRY)
            _frames(example, 12)
            self.assertTrue(example._script_idle)
            self.assertIsNone(example.plan)
            self.assertIsNone(example.exit)
            self.assertFalse(example.running)
            run_script(example)
            self.assertEqual(example._script_runs, [])
            final_line(example)

    def test_script_previews_starts_resets_and_starts_again(self):
        # On the test block the automatic insertion ends in the supported material under it, so
        # the scripted run does not end with an exit: the self-test says so.
        with patch.object(insertion.Example, "_SCRIPT_POINTS", ((ENTRY, EXIT), (ENTRY, (EXIT[0], 0.002)))):
            with _scene("--test") as example:
                _frames(example, 13)
                self.assertIs(type(example.coupler), NeedleTissueCoupler)
                self.assertFalse(example.running)
                first = example.coupler
                _frames(example, 3)
                self.assertTrue(example.running)
                run_script(example)
                self.assertEqual(
                    [label for _, label in example.script.done],
                    [
                        "entry and exit on the map",
                        "Preview path",
                        "Start automatic",
                        "Reset",
                        "entry and exit on the map",
                        "Preview path",
                        "Start automatic",
                    ],
                )
                self.assertIsNot(example.coupler, first)
                self.assertEqual(len(example._script_runs), 1)
                self.assertEqual(example._script_runs[0]["task_status"], "failed")
                self.assertGreaterEqual(example.interaction.entry_count, 1)
                with (
                    self.assertRaisesRegex(AssertionError, "did not end with its exit"),
                    redirect_stdout(io.StringIO()),
                ):
                    example.test_final()
                # The same run with an insertion that ended with its exit passes.
                example._script_runs[0].update(task_status="success", entries=1, exits=1)
                line = final_line(example)
                self.assertEqual(line["script_frames"], [365, 365])
                self.assertEqual(line["script_actions"], 7)
                self.assertEqual(line["track_storage_growths"], 0)
                self.assertEqual(line["opening_storage_growths"], 0)

    def test_self_test_fails_on_a_kernel_module_loaded_after_start_up(self):
        with _scene(*POINTS, "--start", startup=None) as other:
            # The kernel modules of a frame are loaded by another scene first: on the CPU no
            # start-up loads them.
            _frames(other, 3)
            with marked_on_every_device(), _scene("--test", *POINTS, "--start") as example:
                _frames(example, 3)
                line = final_line(example)
                self.assertEqual(line["capture_census_status"], "checked")
                self.assertEqual(line["module_loads_after_startup"], 0)
                launch_late_kernel(example.device)
                with self.assertRaisesRegex(AssertionError, "module_loads_after_startup': 1"):
                    final_line(example)

    def test_self_test_fails_on_a_grown_history_reserve(self):
        with _scene("--test", *POINTS, "--start") as example:
            _frames(example, 2)
            example.interaction.track_storage_growths = 1
            with self.assertRaisesRegex(AssertionError, "history reserve"), redirect_stdout(io.StringIO()):
                example.test_final()


class CouplerEntryTests(unittest.TestCase):
    def test_entries_decline_for_a_caller_that_did_not_ask(self):
        with _scene(startup=None) as example:
            _points(example)
            example.preview()
            example.start()
            coupler, buffers = example.coupler, (example.state_0, example.state_1)
            with patch.object(coupler, "_open_attempt", side_effect=AssertionError("hidden attempt")):
                self.assertFalse(coupler.capture_graphs(buffers, example.control, example.sim_dt))
                self.assertFalse(coupler.capture_contact_graphs(example.sim_dt))
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    coupler.capture_graphs(buffers, example.control, 0.0, native=True)
                if not CUDA:
                    self.assertFalse(coupler.capture_graphs(buffers, example.control, example.sim_dt, native=True))
                    self.assertFalse(coupler.capture_contact_graphs(example.sim_dt, native=True))
            self.assertIsNone(coupler._host_trial_preparation)
            self.assertIsNone(coupler._captured_trials)


@unittest.skipUnless(CUDA, "The start-up captures CUDA graphs")
class NativeStartupTests(unittest.TestCase):
    def test_start_up_captures_and_launches_every_graph_of_the_native_step_and_gives_the_state_back(self):
        with _scene(startup=None) as example:
            _points(example)
            example.preview()
            example.start()
            coupler, solver, interaction = example.coupler, example.solver, example.interaction
            buffers = (example.state_0, example.state_1)
            # What the first frame does before its first step.
            wp.copy(example._frame_particle_q, example.state_0.particle_q)
            solver.refresh_tangent(example._frame_particle_q, example.sim_dt)
            before = _bits(example)
            other = {name: getattr(example.state_1, name).numpy().copy() for name in STATE_FIELDS}
            kept = {name: getattr(coupler, name) for name in coupler._STARTUP_KEPT}
            requested, interval = interaction._engaged_length_requested, coupler.step_time.value
            self.assertTrue(coupler.capture_graphs(buffers, example.control, example.sim_dt, native=True))
            wp.synchronize_device(example.device)
            # Every graph of the step exists and has been launched.
            levels = solver.tolerance_levels()
            self.assertEqual(len(levels), 5)
            self.assertEqual(len(coupler._fem_graphs), 2 * len(levels))
            self.assertEqual(coupler.fem_graph_captures, 2 * len(levels))
            preparation = coupler._host_trial_preparation
            self.assertEqual((len(preparation.graphs), preparation.captures, preparation.launches), (2, 2, 0))
            trials = coupler._captured_trials
            self.assertEqual((len(trials.graphs), trials.captures, trials.launches), (2, 2, 0))
            self.assertTrue(trials.branches_on_shaft_flag)
            np.testing.assert_array_equal(trials.shaft_active.numpy(), [0])
            forces = coupler._host_trial_forces
            self.assertEqual(sorted(forces.graphs), sorted((False, p, f) for p in (False, True) for f in (False, True)))
            self.assertEqual((forces.captures, forces.launches, forces.fallbacks), (4, 0, 0))
            correction = coupler._shaft_preconditioner
            self.assertIsNotNone(correction._input_preparation.replay.graph)
            self.assertEqual(correction._input_preparation.replay.launches, 0)
            self.assertIsNotNone(correction._graphs[False][1])
            self.assertEqual((correction.graph_captures, correction.graph_launches), (1, 0))
            graphs = [entry[0] for entry in coupler._fem_graphs.values()]
            graphs += [entry[0] for entry in preparation.graphs.values()]
            graphs += [entry[0] for entry in trials.graphs.values()]
            graphs += list(forces.graphs.values())
            graphs += [correction._input_preparation.replay.graph, correction._graphs[False][1]]
            self.assertEqual(len(graphs), 20)
            for graph in graphs:
                self.assertIsNotNone(graph.graph_exec)
            self.assertIsNotNone(coupler._endpoint_snapshot)
            # No state is advanced, no history is accepted and no work is counted.
            after = _bits(example)
            for name, values in before.items():
                np.testing.assert_array_equal(after[name], values, err_msg=name)
            for name, values in other.items():
                np.testing.assert_array_equal(getattr(example.state_1, name).numpy(), values, err_msg=name)
            for name, value in kept.items():
                self.assertEqual(getattr(coupler, name), value, name)
            self.assertEqual(coupler.coupling_graph_captures, 2)
            self.assertEqual(interaction._engaged_length_requested, requested)
            self.assertFalse(interaction._device_queries.has_previous)
            self.assertIsNone(interaction._previous_tip)
            self.assertEqual((interaction.track_count, interaction.puncture_count), (0, 0))
            self.assertEqual(coupler.step_time.value, interval)
            self.assertEqual((example.elapsed, example.accepted_physics_steps), (0, 0))
            _frames(example, 2)
            self.assertTrue(example.running, example.failure_reason)

    def test_start_up_without_a_path_asks_for_every_kernel_module_build_that_the_preview_uses(self):
        # The kernel module builds of the coupled step do not depend on the planned path: the
        # start-up of the construction asks for every build that the start-up of the Preview
        # launches, at the block size and with the hash it has there. In a process that has
        # loaded nothing the Preview therefore captures graphs and neither loads nor compiles a
        # kernel module. The solver kernels of the correction count as in such a process.
        device = wp.get_device(DEVICE)
        with patch.object(shaft_preconditioner, "_KERNEL_DEVICES", set()), _requested_builds(device) as builds:
            with _scene() as example:
                self.assertIsNone(example.interaction)
                constructed = set(builds)
                builds.clear()
                captures = capture.census.captures
                _points(example)
                example.preview()
                self.assertTrue(example._startup_ready, example._startup_status)
                previewed = set(builds)
                self.assertGreater(capture.census.captures, captures)
                # The coupled step, the tissue solver, the correction and Warp's Krylov kernels.
                self.assertGreater(len(previewed), 30)
                self.assertEqual(sorted((name, block_dim) for name, block_dim, _ in previewed - constructed), [])
                # The solver kernels are created once for a device: a second call captures nothing.
                self.assertIn(device.alias, shaft_preconditioner._KERNEL_DEVICES)
                captures = capture.census.captures
                shaft_preconditioner.load_kernels(device)
                self.assertEqual(capture.census.captures, captures)
                _frames(example, 1)
                example.toggle_running()
                _frames(example, 2)
                self.assertTrue(example.running, example.failure_reason)

    def test_run_after_the_start_up_captures_nothing_through_entry_exit_and_the_flips_of_the_flag(self):
        begun = [0]
        begin = wp.capture_begin

        def counted(device=None, stream=None, *arguments, **options):
            # At Warp's entry: also a capture that no scope of the package encloses.
            begun[0] += 1
            return begin(device, stream, *arguments, **options)

        with _scene("--test", *POINTS, "--start") as example:
            self.assertTrue(example._startup_ready, example._startup_status)
            coupler = example.coupler
            trials = coupler._captured_trials
            counters = (
                capture.census.captures,
                capture.census.solver_captures,
                trials.captures,
                coupler.fem_graph_captures,
                coupler._host_trial_preparation.captures,
                coupler._host_trial_forces.captures,
                coupler._shaft_preconditioner.graph_captures,
            )
            graphs = [entry[0] for entry in trials.graphs.values()]
            with patch.object(wp, "capture_begin", counted):
                flips = _sliders(example)
            self.assertTrue(example.running, example.failure_reason)
            interaction = example.interaction
            self.assertGreaterEqual(interaction.entry_count, 2)
            self.assertGreaterEqual(interaction.exit_count, 1)
            self.assertGreaterEqual(flips, 3)
            # Not one capture, of the package or of stock Warp, and the graphs are the start-up's.
            self.assertEqual(begun[0], 0)
            self.assertEqual(
                (
                    capture.census.captures,
                    capture.census.solver_captures,
                    trials.captures,
                    coupler.fem_graph_captures,
                    coupler._host_trial_preparation.captures,
                    coupler._host_trial_forces.captures,
                    coupler._shaft_preconditioner.graph_captures,
                ),
                counters,
            )
            self.assertIs(coupler._captured_trials, trials)
            for kept, graph in zip(graphs, (entry[0] for entry in trials.graphs.values()), strict=True):
                self.assertIs(graph, kept)
            self.assertGreater(trials.launches, 0)
            self.assertGreater(coupler.fem_graph_launches, 0)
            line = final_line(example)
            self.assertEqual(line["capture_census_status"], "checked")
            for name in example.startup.numbers:
                self.assertEqual(line[name], 0, name)
            # A capture after the start-up fails the self-test.
            with capture.ScopedCapture(device=example.device):
                pass
            with self.assertRaisesRegex(AssertionError, "captures_after_startup': 1"):
                final_line(example)


class SliderScenarioTests(unittest.TestCase):
    def test_sliders_take_the_needle_out_and_in_again_on_the_test_block(self):
        # The scenario of the capture test above, on every device: two entries, an exit, three flips.
        with _scene(*POINTS, "--start") as example:
            flips = _sliders(example)
            self.assertTrue(example.running, example.failure_reason)
            self.assertGreaterEqual(example.interaction.entry_count, 2)
            self.assertGreaterEqual(example.interaction.exit_count, 1)
            self.assertGreaterEqual(flips, 3)
            self.assertEqual(example.interaction.track_storage_growths, 0)
            self.assertEqual(example.interaction.opening_storage_growths, 0)


if __name__ == "__main__":
    unittest.main()
