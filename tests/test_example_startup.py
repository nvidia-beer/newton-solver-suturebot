# SPDX-License-Identifier: Apache-2.0
"""The start-up helper of the examples: the mark, the first draw, the module loads, the check and the script."""

import gc
import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import warp as wp
from newton.viewer import ViewerNull

from newton_solver_suturebot._src.utils import capture
from newton_solver_suturebot.examples import _startup
from newton_solver_suturebot.examples._startup import Script, Startup


class _Graph:
    """Stands for wp.Graph: Warp builds ``graph_exec`` at the first launch."""

    def __init__(self, launched=False):
        self.graph_exec = object() if launched else None


class _Scope:
    """Stands for wp.ScopedCapture: it holds the device it was given and, once ended, a graph."""

    def __init__(self, device=None, stream=None, graph=None):
        self.device, self.stream, self.graph, self._graph = device, stream, None, graph

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.graph = self._graph


def _cuda():
    """Stands for a CUDA device of Warp."""
    return SimpleNamespace(is_cuda=True, has_context=True, is_capturing=False)


class _Viewer:
    """Records the frame calls of a viewer; ``num_frames`` and ``_headless`` as the real viewers have them."""

    def __init__(self, **attributes):
        self.calls = []
        vars(self).update(attributes)

    def begin_frame(self, time):
        self.calls.append(("begin_frame", time))

    def end_frame(self):
        self.calls.append(("end_frame",))


@wp.kernel(enable_backward=False, module=wp.get_module("test_example_startup.first"))
def _first(values: wp.array[float]):
    values[wp.tid()] = 1.0


@wp.kernel(enable_backward=False, module=wp.get_module("test_example_startup.first"))
def _first_too(values: wp.array[float]):
    values[wp.tid()] = 2.0


@wp.kernel(enable_backward=False, module=wp.get_module("test_example_startup.second"))
def _second(values: wp.array[float]):
    values[wp.tid()] = 3.0


class StartupRuleTests(unittest.TestCase):
    def test_only_the_most_derived_class_owns_the_start_up(self):
        class Base:
            pass

        class Scene(Base):
            pass

        self.assertTrue(_startup.owns_startup(Base(), Base))
        self.assertTrue(_startup.owns_startup(Scene(), Scene))
        # A scene built on the class: the class leaves it alone.
        self.assertFalse(_startup.owns_startup(Scene(), Base))

    def test_frame_limit_is_the_rule_of_the_grasp_example(self):
        from newton_solver_suturebot.examples.grasp.example_fem_needle_grasp import Example as GraspExample

        viewers = (
            SimpleNamespace(),  # no limit at all
            SimpleNamespace(num_frames=None),
            SimpleNamespace(num_frames=100),  # null and USD viewers
            SimpleNamespace(num_frames=100, _headless=True),  # OpenGL without a window
            SimpleNamespace(num_frames=100, _headless=False),  # a window ignores the limit
            SimpleNamespace(num_frames=None, _headless=True),
            ViewerNull(num_frames=3),
        )
        expected = (False, False, True, True, False, False, True)
        for viewer, limited in zip(viewers, expected, strict=True):
            self.assertEqual(_startup.frame_limited(viewer), limited)
            self.assertEqual(GraspExample._frame_limited(viewer), limited)

    def test_first_draw_is_a_whole_frame_only_where_no_frame_would_be_lost(self):
        def calls(viewer, **options):
            viewer.calls.clear()
            _startup.first_draw(viewer, lambda: viewer.calls.append(("draw",)), 0.25, **options)
            return viewer.calls

        window = _Viewer(num_frames=100, _headless=False)
        self.assertEqual(calls(window), [("begin_frame", 0.25), ("draw",), ("end_frame",)])
        self.assertEqual(calls(_Viewer()), [("begin_frame", 0.25), ("draw",), ("end_frame",)])
        # A frame-limited viewer counts frames in begin_frame or end_frame: the log calls alone.
        for limited in (_Viewer(num_frames=100), _Viewer(num_frames=100, _headless=True)):
            self.assertEqual(calls(limited), [("draw",)])
        # A rebuild from a panel button runs inside the viewer's frame: no frame for any viewer.
        self.assertEqual(calls(window, frame=False), [("draw",)])

    def test_first_draw_leaves_a_recording_viewer_alone_and_loads_its_kernel_module(self):
        # The USD viewer writes every log call as a time sample: a start-up draw would add the
        # start state to the recording. Nothing is drawn; the kernel module of the viewer's log
        # calls is loaded by name, with the block size of a first launch.
        from newton.viewer import ViewerFile, ViewerUSD

        self.assertFalse(_startup.records_logs(_Viewer()))
        null = ViewerNull(num_frames=2)
        self.addCleanup(null.close)
        self.assertFalse(_startup.records_logs(null))
        cpu = wp.get_device("cpu")
        drawn = []

        def draw():
            drawn.append("draw")

        for cls in (ViewerUSD, ViewerFile):
            # An instance without its file: the rule looks at the class alone.
            viewer = cls.__new__(cls)
            viewer.device = cpu
            self.assertTrue(_startup.records_logs(viewer), cls.__name__)
            for frame in (True, False):
                with patch.object(_startup.wp, "load_module") as load:
                    _startup.first_draw(viewer, draw, 0.25, frame=frame)
                self.assertEqual(drawn, [])
                self.assertEqual(load.call_args.args, (_startup.VIEWER_KERNELS,))
                self.assertEqual(load.call_args.kwargs, {"device": cpu, "block_dim": 1})
        # The module is the one a log call of the viewers launches: loading it by name loads that build.
        from newton._src.viewer import kernels

        self.assertEqual(kernels.__name__, _startup.VIEWER_KERNELS)
        _startup.first_draw(viewer, lambda: self.fail("drawn"))
        loaded = capture.loaded_modules()
        self.assertIn((_startup.VIEWER_KERNELS, "cpu", 1), loaded)

    def test_first_draw_leaves_a_null_viewer_every_frame_it_was_asked_for(self):
        viewer = ViewerNull(num_frames=2)
        self.addCleanup(viewer.close)
        drawn = []
        _startup.first_draw(viewer, lambda: drawn.append(viewer.frame_count))
        self.assertEqual(drawn, [0])
        frames = 0
        while viewer.is_running():
            viewer.begin_frame(0.0)
            viewer.end_frame()
            frames += 1
        self.assertEqual(frames, 2)


class LoadKernelModulesTests(unittest.TestCase):
    def test_each_module_is_loaded_once_with_the_block_size_of_a_first_launch(self):
        cpu = wp.get_device("cpu")
        with patch.object(_startup.wp, "load_module") as load:
            _startup.load_kernel_modules(cpu, _first, _first_too, _second)
        self.assertEqual([call.args[0] for call in load.call_args_list], [_first.module, _second.module])
        for call in load.call_args_list:
            self.assertEqual(call.kwargs, {"device": cpu, "block_dim": 1})
        cuda = SimpleNamespace(is_cuda=True)
        with patch.object(_startup.wp, "load_module") as load, patch.object(_startup.wp, "get_device", lambda d: d):
            _startup.load_kernel_modules(cuda, _first)
            _startup.load_kernel_modules(cuda, _second, block_dim=32)
        self.assertEqual(
            [call.kwargs for call in load.call_args_list],
            [
                {"device": cuda, "block_dim": 256},
                {"device": cuda, "block_dim": 32},
            ],
        )

    def test_a_launch_after_the_load_loads_nothing(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        _startup.load_kernel_modules(device, _first, _second)
        loaded = capture.loaded_modules()
        size = 256 if device.is_cuda else 1
        for kernel in (_first, _second):
            self.assertIn((kernel.module.name, device.alias, size), loaded)
        values = wp.zeros(3, dtype=float, device=device)
        for kernel in (_first, _first_too, _second):
            wp.launch(kernel, dim=3, inputs=[values], device=device)
        self.assertEqual(capture.loaded_modules(), loaded)
        self.assertEqual(values.numpy().tolist(), [3.0, 3.0, 3.0])


class StartupTests(unittest.TestCase):
    """The mark and the check, on doubles of Warp's capture scope and of the loaded kernel modules."""

    def setUp(self):
        self.cuda = _cuda()
        double = patch.object(capture.wp, "ScopedCapture", _Scope)
        double.start()
        self.addCleanup(double.stop)
        # Kernel module builds loaded in the process, as capture.loaded_modules() reports them.
        self.loaded = {("physics.a", "cuda:0", 256): b"1"}
        modules = patch.object(capture, "loaded_modules", side_effect=lambda: dict(self.loaded))
        modules.start()
        self.addCleanup(modules.stop)
        gc.collect()

    def capture_graph(self, launched=False):
        graph = _Graph(launched)
        with capture.ScopedCapture(device=self.cuda, graph=graph):
            pass
        return graph

    def check(self, startup):
        output = io.StringIO()
        with redirect_stdout(output):
            result = startup.check()
        return result, output.getvalue()

    def test_scene_without_a_mark_is_not_checked_and_says_why(self):
        startup = Startup(self.cuda)
        self.assertEqual(startup.census(), {"capture_census_status": "start-up not marked", "startups": 0})
        with self.assertRaisesRegex(RuntimeError, "No start-up is open"):
            startup.finish()
        startup.begin()
        self.capture_graph()
        self.loaded[("physics.late", "cuda:0", 256)] = b"1"
        result, printed = self.check(startup)
        self.assertEqual(result, {"capture_census_status": "start-up not marked", "startups": 0})
        self.assertIn("not checked: start-up not marked", printed)

    def test_cpu_device_takes_no_mark_and_is_not_checked(self):
        for device in ("cpu", wp.get_device("cpu")):
            startup = Startup(device)
            startup.begin()
            self.assertIsNone(startup.finish())
            self.loaded[("physics.late", "cpu", 1)] = b"1"
            result, printed = self.check(startup)
            self.assertEqual(result, {"capture_census_status": "CPU device", "startups": 0})
            self.assertIn("not checked: CPU device", printed)

    def test_start_up_work_is_recorded_and_a_quiet_run_passes(self):
        startup = Startup(self.cuda)
        startup.begin()
        graph = self.capture_graph()
        graph.graph_exec = object()  # launched at start-up
        # A Krylov solve of the eager start-up frame: stock Warp captures a single-use graph for it.
        with patch.object(capture.wp, "get_cuda_devices", return_value=[self.cuda]), capture.PauseGC():
            pass
        self.loaded[("physics.b", "cuda:0", 256)] = b"1"
        mark = startup.finish()
        self.assertIs(startup.mark, mark)
        result, printed = self.check(startup)
        self.assertEqual(printed, "")
        self.assertEqual(result["capture_census_status"], "checked")
        self.assertEqual(result["startups"], 1)
        for name in (*Startup.numbers, "render_module_loads_after_startup"):
            self.assertEqual(result[name], 0, name)
        self.assertEqual(
            (result["startup_captures"], result["startup_solver_captures"], result["startup_module_loads"]), (1, 1, 1)
        )
        self.assertGreaterEqual(result["startup_seconds"], 0.0)
        # The line that test_final prints holds numbers, no tables.
        self.assertFalse(set(result) & {"late_sites", "late_modules", "late_render_modules"})
        self.assertEqual(startup.census()["late_modules"], {})

    def test_capture_after_the_mark_fails(self):
        startup = Startup(self.cuda)
        startup.begin()
        startup.finish()
        graph = self.capture_graph(launched=True)
        self.assertEqual(startup.census()["captures_after_startup"], 1)
        with self.assertRaisesRegex(
            AssertionError, r"'captures_after_startup': 1.*late sites \{'[^']*test_example_startup"
        ):
            startup.check()
        del graph

    def test_solver_capture_after_the_mark_fails(self):
        startup = Startup(self.cuda)
        startup.begin()
        startup.finish()
        with patch.object(capture.wp, "get_cuda_devices", return_value=[self.cuda]), capture.PauseGC():
            pass
        with self.assertRaisesRegex(AssertionError, "'solver_captures_after_startup': 1"):
            startup.check()

    def test_kernel_module_load_after_the_mark_fails_and_names_the_module(self):
        startup = Startup(self.cuda)
        startup.begin()
        startup.finish()
        self.loaded[("examples.needle._thread_view", "cuda:0", 256)] = b"1"
        self.loaded[("newton._src.viewer.kernels", "cuda:0", 256)] = b"1"
        census = startup.census()
        self.assertEqual(census["module_loads_after_startup"], 2)
        self.assertEqual(census["late_modules"], {"examples.needle._thread_view": 1, "newton._src.viewer.kernels": 1})
        with self.assertRaisesRegex(AssertionError, r"'module_loads_after_startup': 2.*examples\.needle\._thread_view"):
            startup.check()

    def test_another_build_of_a_loaded_module_is_a_load(self):
        startup = Startup(self.cuda)
        startup.begin()
        startup.finish()
        # A kernel that gains an overload gives its module another hash; a launch with another block size, another build.
        self.loaded[("physics.a", "cuda:0", 256)] = b"2"
        self.loaded[("physics.a", "cuda:0", 1)] = b"2"
        with self.assertRaisesRegex(AssertionError, "'module_loads_after_startup': 2"):
            startup.check()

    def test_viewer_modules_fail_unless_the_scene_names_their_prefix(self):
        strict, lenient = Startup(self.cuda), Startup(self.cuda, render_prefixes=("newton._src.viewer.",))
        for startup in (strict, lenient):
            startup.begin()
            startup.finish()
        self.loaded[("newton._src.viewer.kernels", "cuda:0", 256)] = b"1"
        with self.assertRaisesRegex(AssertionError, "'module_loads_after_startup': 1"):
            strict.check()
        result, _ = self.check(lenient)
        self.assertEqual((result["module_loads_after_startup"], result["render_module_loads_after_startup"]), (0, 1))
        self.assertEqual(lenient.census()["late_render_modules"], {"newton._src.viewer.kernels": 1})

    def test_start_up_graph_that_was_never_launched_fails(self):
        startup = Startup(self.cuda)
        startup.begin()
        graph = self.capture_graph()
        startup.finish()
        self.assertEqual(startup.census()["startup_unlaunched_graphs"], 1)
        with self.assertRaisesRegex(AssertionError, "'startup_unlaunched_graphs': 1"):
            startup.check()
        # Launched before the mark is taken again: the later mark replaces the earlier one.
        graph.graph_exec = object()
        startup.finish()
        result, _ = self.check(startup)
        self.assertEqual(
            (result["startup_unlaunched_graphs"], result["startup_captures"], result["startups"]), (0, 1, 1)
        )

    def test_unlaunched_graph_of_another_scene_is_not_counted(self):
        foreign = self.capture_graph()
        startup = Startup(self.cuda)
        startup.begin()
        own = self.capture_graph(launched=True)
        mark = startup.finish()
        # The record of the package counts every live graph of the process; the helper those of this start-up.
        self.assertEqual(mark.startup["startup_unlaunched_graphs"], 1)
        result, _ = self.check(startup)
        self.assertEqual(result["startup_unlaunched_graphs"], 0)
        del foreign, own

    def test_mark_can_move_and_what_came_before_the_later_mark_is_start_up(self):
        startup = Startup(self.cuda)
        startup.begin()
        startup.finish()
        self.loaded[("physics.view", "cuda:0", 256)] = b"1"
        graph = self.capture_graph(launched=True)
        self.assertEqual(startup.census()["module_loads_after_startup"], 1)
        startup.finish()
        result, _ = self.check(startup)
        self.assertEqual(result["startups"], 1)
        self.assertEqual((result["module_loads_after_startup"], result["captures_after_startup"]), (0, 0))
        self.assertEqual((result["startup_module_loads"], result["startup_captures"]), (1, 1))
        del graph

    def test_rebuild_is_a_start_up_and_keeps_what_came_after_the_earlier_mark(self):
        startup = Startup(self.cuda)
        startup.begin()
        first = self.capture_graph(launched=True)
        self.loaded[("physics.b", "cuda:0", 256)] = b"1"
        startup.finish()
        # A frame of the run loads a module and captures; then the scene is rebuilt.
        self.loaded[("physics.late", "cuda:0", 256)] = b"1"
        late = self.capture_graph(launched=True)
        startup.begin()
        self.assertEqual(startup.census()["capture_census_status"], "start-up not marked")
        # The rebuild captures, launches and loads: start-up work.
        second = self.capture_graph(launched=True)
        self.loaded[("physics.c", "cuda:0", 256)] = b"1"
        startup.finish()
        census = startup.census()
        self.assertEqual(census["startups"], 2)
        self.assertEqual((census["captures_after_startup"], census["module_loads_after_startup"]), (1, 1))
        self.assertEqual(census["late_modules"], {"physics.late": 1})
        self.assertEqual((census["startup_captures"], census["startup_module_loads"]), (2, 2))
        with self.assertRaisesRegex(AssertionError, r"physics\.late.*in 2 start-ups"):
            startup.check()
        del first, late, second

    def test_rebuilds_without_late_work_pass_and_add_their_records(self):
        startup = Startup(self.cuda)
        graphs = []
        for index in range(3):
            startup.begin()
            graphs.append(self.capture_graph(launched=True))
            self.loaded[(f"physics.scene{index}", "cuda:0", 256)] = b"1"
            startup.finish()
        result, _ = self.check(startup)
        self.assertEqual((result["startups"], result["startup_captures"], result["startup_module_loads"]), (3, 3, 3))
        for name in Startup.numbers:
            self.assertEqual(result[name], 0, name)

    def test_census_failure_reads_a_census_of_any_scene(self):
        quiet = {"capture_census_status": "checked", **dict.fromkeys(Startup.numbers, 0)}
        self.assertIsNone(_startup.census_failure(quiet))
        self.assertIsNone(_startup.census_failure({"capture_census_status": "CPU device"}))
        for name in Startup.numbers:
            self.assertIn(f"'{name}': 1", _startup.census_failure({**quiet, name: 1}))
        # A checked scene without a number is not a quiet one.
        self.assertIsNotNone(_startup.census_failure({"capture_census_status": "checked"}))


class StartupGraphTests(unittest.TestCase):
    def test_real_graph_is_counted_until_it_is_launched(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not device.is_cuda:
            self.skipTest("A graph needs CUDA")
        values = wp.zeros(4, dtype=float, device=device)
        wp.launch(_first, dim=4, inputs=[values], device=device)
        startup = Startup(device)
        startup.begin()
        with capture.ScopedCapture(device=device) as scope:
            wp.launch(_second, dim=4, inputs=[values], device=device)
        startup.finish()
        self.assertEqual(startup.census()["startup_unlaunched_graphs"], 1)
        _startup.launch_startup_graph(scope.graph, device)
        self.assertEqual(values.numpy().tolist(), [3.0] * 4)
        startup.finish()
        census = startup.census()
        self.assertEqual((census["startup_unlaunched_graphs"], census["startup_captures"]), (0, 1))
        # Launches of the start-up graph are what a frame does: nothing is captured or loaded.
        for _ in range(3):
            wp.capture_launch(scope.graph)
        self.assertEqual(startup.check()["captures_after_startup"], 0)
        with capture.ScopedCapture(device=device):
            wp.launch(_first, dim=4, inputs=[values], device=device)
        with self.assertRaisesRegex(AssertionError, "'captures_after_startup': 1"):
            startup.check()


class ScriptTests(unittest.TestCase):
    def test_actions_run_before_their_frame_in_the_order_they_were_added(self):
        script, log = Script(), []
        script.at(2, "press", lambda: log.append("press"))
        script.drag(2, 5, "drag", lambda progress: log.append(progress))
        script.at(5, "release", lambda: log.append("release"))
        script.at(0, "first", lambda: log.append("first"))
        self.assertEqual(script.frames, 6)
        script.rest(4)
        self.assertEqual(script.frames, 10)
        played = []
        while not script.complete:
            before = len(log)
            script.play()
            played.append(log[before:])
        self.assertEqual(played, [["first"], [], ["press", 0.25], [0.5], [0.75], [1.0, "release"], [], [], [], []])
        self.assertEqual(script.frame, 10)
        # A drag is listed once, at its first frame.
        self.assertEqual(script.done, [(0, "first"), (2, "press"), (2, "drag"), (5, "release")])
        self.assertEqual(script.summary(), {"script_frames": [10, 10], "script_actions": 4})
        # Frames beyond the script have no action.
        script.play()
        self.assertEqual(script.frame, 11)
        with self.assertRaisesRegex(ValueError, "ends at or after"):
            script.drag(4, 3, "backwards", lambda progress: None)

    def test_frame_limited_viewer_runs_the_whole_script(self):
        script = Script()
        script.at(149, "last", lambda: None)
        limited, long_run = ViewerNull(num_frames=100), ViewerNull(num_frames=400)
        window = SimpleNamespace(num_frames=100, _headless=False)
        unlimited = SimpleNamespace()
        for viewer in (limited, long_run, window, unlimited):
            script.reserve(viewer)
        self.assertEqual(limited.num_frames, 150)
        self.assertEqual(long_run.num_frames, 400)
        # A window runs until it is closed; its limit is not used.
        self.assertEqual(window.num_frames, 100)
        self.assertFalse(hasattr(unlimited, "num_frames"))
        for viewer in (limited, long_run):
            viewer.close()


if __name__ == "__main__":
    unittest.main()
