# SPDX-License-Identifier: Apache-2.0
"""Capture-owned cyclic-GC suspension restores state, including exceptional exits."""

import ast
import gc
import os
import sys
import threading
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot._src.utils import capture
from newton_solver_suturebot._src.utils.capture import PauseGC, ScopedCapture


class FakeCapture:
    def __init__(self, *, fail_enter=False, fail_exit=False):
        self.fail_enter, self.fail_exit = fail_enter, fail_exit

    def __enter__(self):
        if gc.isenabled():
            raise AssertionError("Cyclic GC must already be disabled")
        if self.fail_enter:
            raise RuntimeError("entry failed")
        return self

    def __exit__(self, *args):
        if gc.isenabled():
            raise AssertionError("GC must remain disabled until capture ends")
        if self.fail_exit:
            raise RuntimeError("exit failed")


@wp.kernel(enable_backward=False)
def copy_value(source: wp.array[float], target: wp.array[float]):
    target[wp.tid()] = source[wp.tid()]


# A generic kernel in a Warp module of its own: every overload gives that module another hash.
@wp.kernel(enable_backward=False, module=wp.get_module("test_capture_scope.census_generic"))
def fill_any(target: wp.array[Any], value: Any):
    target[wp.tid()] = value


class _Graph:
    """Stands for wp.Graph: Warp builds ``graph_exec`` at the first launch."""

    def __init__(self, launched):
        self.graph_exec = object() if launched else None


class _Scope:
    """Stands for wp.ScopedCapture: it holds the device or stream it was given and, once ended, a graph."""

    def __init__(self, device=None, stream=None, graph=None):
        self.device, self.stream, self.graph, self._graph = device, stream, None, graph

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.graph = self._graph


def _cuda(*, capturing=False, context=True):
    """Stands for a CUDA device of Warp."""
    return SimpleNamespace(is_cuda=True, has_context=context, is_capturing=capturing)


class CaptureScopeTests(unittest.TestCase):
    def setUp(self):
        self.enabled = gc.isenabled()
        self.threshold = gc.get_threshold()

    def tearDown(self):
        gc.set_threshold(*self.threshold)
        (gc.enable if self.enabled else gc.disable)()

    def test_nested_scopes_and_initially_disabled_gc_preserve_state(self):
        for enabled in (True, False):
            (gc.enable if enabled else gc.disable)()
            with patch("newton_solver_suturebot._src.utils.capture.wp.ScopedCapture", FakeCapture):
                with ScopedCapture():
                    with ScopedCapture():
                        self.assertFalse(gc.isenabled())
                    self.assertFalse(gc.isenabled())
            self.assertEqual(gc.isenabled(), enabled)

    def test_capture_entry_body_and_exit_failures_restore_gc(self):
        gc.enable()
        with patch("newton_solver_suturebot._src.utils.capture.wp.ScopedCapture", FakeCapture):
            with self.assertRaisesRegex(RuntimeError, "entry failed"):
                with ScopedCapture(fail_enter=True):
                    self.fail("Failed capture cannot enter its body")
            self.assertTrue(gc.isenabled())
            with self.assertRaisesRegex(RuntimeError, "body failed"):
                with ScopedCapture():
                    raise RuntimeError("body failed")
            self.assertTrue(gc.isenabled())
            with self.assertRaisesRegex(RuntimeError, "exit failed"):
                with ScopedCapture(fail_exit=True):
                    pass
            self.assertTrue(gc.isenabled())

    def test_optional_collection_only_precedes_first_scope(self):
        gc.enable()
        with (
            patch("newton_solver_suturebot._src.utils.capture.wp.ScopedCapture", FakeCapture),
            patch("newton_solver_suturebot._src.utils.capture.gc.collect") as collect,
        ):
            with ScopedCapture(collect=True):
                collect.assert_called_once_with()
                with self.assertRaisesRegex(RuntimeError, "outside capture"):
                    with ScopedCapture(collect=True):
                        self.fail("Collection inside another capture is forbidden")
                self.assertFalse(gc.isenabled())
            collect.assert_called_once_with()
        self.assertTrue(gc.isenabled())

    def test_overlapping_threads_restore_gc_only_after_last_capture(self):
        gc.enable()
        first_entered, second_entered, first_done, release_second = (threading.Event() for _ in range(4))
        failures = []

        def first():
            try:
                with ScopedCapture():
                    first_entered.set()
                    if not second_entered.wait(5):
                        raise RuntimeError("Second capture did not start")
                self.assertFalse(gc.isenabled())
            except BaseException as error:
                failures.append(error)
            finally:
                first_done.set()

        def second():
            try:
                if not first_entered.wait(5):
                    raise RuntimeError("First capture did not start")
                with ScopedCapture():
                    second_entered.set()
                    if not release_second.wait(5):
                        raise RuntimeError("Second capture was not released")
            except BaseException as error:
                failures.append(error)

        with patch("newton_solver_suturebot._src.utils.capture.wp.ScopedCapture", FakeCapture):
            threads = [threading.Thread(target=first), threading.Thread(target=second)]
            for thread in threads:
                thread.start()
            try:
                self.assertTrue(first_done.wait(5))
                self.assertFalse(gc.isenabled())
            finally:
                release_second.set()
                for thread in threads:
                    thread.join(5)
            self.assertEqual(failures, [])
        self.assertTrue(gc.isenabled())

    def test_pause_gc_starts_no_capture_and_restores_after_failure(self):
        for enabled in (True, False):
            (gc.enable if enabled else gc.disable)()
            with patch("newton_solver_suturebot._src.utils.capture.wp.ScopedCapture") as capture:
                with self.assertRaisesRegex(RuntimeError, "solver failed"):
                    with PauseGC():
                        self.assertFalse(gc.isenabled())
                        with PauseGC():
                            self.assertFalse(gc.isenabled())
                        raise RuntimeError("solver failed")
                capture.assert_not_called()
            self.assertEqual(gc.isenabled(), enabled)
        gc.enable()
        with patch("newton_solver_suturebot._src.utils.capture.wp.ScopedCapture", FakeCapture):
            with PauseGC():
                with ScopedCapture():
                    with PauseGC():
                        self.assertFalse(gc.isenabled())
                self.assertFalse(gc.isenabled())
        self.assertTrue(gc.isenabled())

    def test_all_project_krylov_invocations_are_guarded(self):
        # All six callsites are needed: warmup creates internal Warp captures
        # before the first application-owned outer graph exists.
        root = Path(__file__).resolve().parents[1] / "src/newton_solver_suturebot/_src"
        sites = {
            "solvers/fem/solver.py": "self._lin_solver",
            "solvers/fem/prepared_step.py": "self._solvers[key]",
            "solvers/fem/differentiation.py": "self.backward_solver if transpose else self.forward_solver",
            "solvers/coupled/acceleration/shaft_preconditioner.py": "self.linear_solver",
            "solvers/coupled/acceleration/surface_preconditioner.py": "self.linear_solver",
            "solvers/fem/warp_basis_provider.py": "self.solver",
        }
        for name, target in sites.items():
            module = ast.parse((root / name).read_text())
            parents = {child: parent for parent in ast.walk(module) for child in ast.iter_child_nodes(parent)}
            calls = [n for n in ast.walk(module) if isinstance(n, ast.Call) and ast.unparse(n.func) == target]
            self.assertEqual(len(calls), 1, name)
            node, guarded = calls[0], False
            while node in parents:
                node = parents[node]
                if isinstance(node, ast.With):
                    guarded |= any(ast.unparse(item.context_expr) == "PauseGC()" for item in node.items)
            self.assertTrue(guarded, name)

    def test_pending_pinned_cycle_is_released_only_after_cuda_capture(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not device.is_cuda:
            self.skipTest("CUDA pinned-resource lifetime")
        source = wp.array([1.5, -2.0], dtype=float, device=device)
        target = wp.empty_like(source)
        wp.launch(copy_value, 2, [source, target], device=device)

        class PinnedCycle:
            def __init__(self):
                self.buffer = wp.empty(32, dtype=float, device="cpu", pinned=True)
                self.cycle = self

        gc.set_threshold(1000000, 1000000, 1000000)
        pending = PinnedCycle()
        reference = weakref.ref(pending)
        del pending
        gc.enable()
        with ScopedCapture(device=device) as capture:
            gc.set_threshold(1, 1, 1)
            cycles = []
            for _ in range(64):
                value = []
                value.append(value)
                cycles.append(value)
            del cycles
            self.assertIsNotNone(reference())
            wp.launch(copy_value, 2, [source, target], device=device)
        gc.collect()
        self.assertIsNone(reference())
        source.assign(np.asarray([3.0, 4.0], np.float32))
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(target.numpy(), [3.0, 4.0])


class CaptureCensusTests(unittest.TestCase):
    """Counters of captures, solver captures and module loads, and the start-up mark."""

    def setUp(self):
        self.enabled = gc.isenabled()
        self.cuda = _cuda()
        # Warp's scope is a double unless a test ends it: no test here needs a CUDA device for a count.
        self.double = patch.object(capture.wp, "ScopedCapture", _Scope)
        self.double.start()
        self.addCleanup(self.double.stop)

    def tearDown(self):
        (gc.enable if self.enabled else gc.disable)()

    def devices(self, *devices):
        return patch.object(capture.wp, "get_cuda_devices", return_value=list(devices))

    def mark(self):
        return capture.finish_startup(capture.begin_startup(), self.cuda)

    def test_package_capture_is_counted_on_a_cuda_scope_and_not_on_the_cpu_device(self):
        mark = self.mark()
        with ScopedCapture(device="cpu"):
            pass
        self.assertEqual(capture.counts(mark), {"captures_after_startup": 0, "solver_captures_after_startup": 0})
        with ScopedCapture(device=self.cuda):
            self.assertEqual(capture.counts(mark)["captures_after_startup"], 1)
        # A stream decides the device, as in wp.capture_begin.
        with ScopedCapture(device="cpu", stream=SimpleNamespace(device=self.cuda)):
            pass
        self.assertEqual(capture.counts(mark), {"captures_after_startup": 2, "solver_captures_after_startup": 0})
        # A scope that Warp refuses to enter is no capture.
        with patch.object(_Scope, "__enter__", side_effect=RuntimeError("already capturing")):
            with self.assertRaisesRegex(RuntimeError, "already capturing"):
                with ScopedCapture(device=self.cuda):
                    self.fail("Failed capture cannot enter its body")
        self.assertEqual(capture.counts(mark)["captures_after_startup"], 2)
        self.assertEqual(gc.isenabled(), self.enabled)

    def test_real_cpu_capture_is_not_counted_and_its_graph_is_not_kept(self):
        self.double.stop()
        mark = self.mark()
        graphs = len(capture.census.graphs)
        source = wp.array([1.5, -2.0], dtype=float, device="cpu")
        target = wp.empty_like(source)
        with ScopedCapture(device="cpu") as scope:
            wp.launch(copy_value, 2, [source, target], device="cpu")
        self.assertIsNotNone(scope.graph)
        self.assertEqual(capture.counts(mark), {"captures_after_startup": 0, "solver_captures_after_startup": 0})
        self.assertEqual(len(capture.census.graphs), graphs)
        self.assertEqual(capture.modules(mark)["late_sites"], {})

    def test_outer_pause_with_two_inner_pauses_counts_two_solver_captures(self):
        mark = self.mark()
        with self.devices(self.cuda):
            with PauseGC():
                with PauseGC():
                    pass
                with PauseGC():
                    pass
            self.assertEqual(capture.counts(mark), {"captures_after_startup": 0, "solver_captures_after_startup": 2})
            with PauseGC():
                pass
            self.assertEqual(capture.counts(mark)["solver_captures_after_startup"], 3)
            # A call that raised is not counted; the pause around it then holds no completed call either.
            with self.assertRaisesRegex(RuntimeError, "solver failed"):
                with PauseGC():
                    with PauseGC():
                        raise RuntimeError("solver failed")
            self.assertEqual(capture.counts(mark)["solver_captures_after_startup"], 3)
        self.assertEqual(gc.isenabled(), self.enabled)

    def test_pause_inside_a_package_capture_counts_nothing(self):
        mark = self.mark()
        with self.devices(self.cuda):
            with ScopedCapture(device=self.cuda):
                with PauseGC():
                    pass
                with PauseGC():
                    with PauseGC():
                        pass
            self.assertEqual(capture.counts(mark), {"captures_after_startup": 1, "solver_captures_after_startup": 0})
            # The capture has ended: the next call is outside again.
            with PauseGC():
                pass
            self.assertEqual(capture.counts(mark)["solver_captures_after_startup"], 1)

    def test_nothing_is_counted_without_a_cuda_device_in_use_or_while_one_is_capturing(self):
        mark = self.mark()
        for devices in (
            (),
            (_cuda(context=False),),
            (_cuda(capturing=True),),
            (self.cuda, _cuda(capturing=True)),
        ):
            with self.subTest(devices=devices), self.devices(*devices):
                with PauseGC():
                    with PauseGC():
                        pass
        self.assertEqual(capture.counts(mark)["solver_captures_after_startup"], 0)
        # Only a device with a context is asked whether it captures: a CPU run acquires no context.
        unused = SimpleNamespace(is_cuda=True, has_context=False)
        with self.devices(unused, self.cuda):
            with PauseGC():
                pass
        self.assertEqual(capture.counts(mark)["solver_captures_after_startup"], 1)

    def test_pause_nesting_and_open_captures_are_kept_per_thread(self):
        mark = self.mark()
        inside, done = threading.Event(), threading.Event()
        failures = []

        def other():
            try:
                if not inside.wait(5):
                    raise RuntimeError("The first thread did not enter its scopes")
                with PauseGC():
                    pass
            except BaseException as error:
                failures.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=other)
        thread.start()
        try:
            with self.devices(self.cuda):
                with PauseGC():
                    inside.set()
                    self.assertTrue(done.wait(5))
                # The other thread's call does not turn this pause into an enclosing one.
                self.assertEqual(capture.counts(mark)["solver_captures_after_startup"], 2)
                inside.clear()
                done.clear()
                thread.join(5)
                thread = threading.Thread(target=other)
                thread.start()
                with ScopedCapture(device=self.cuda):
                    inside.set()
                    self.assertTrue(done.wait(5))
                # A capture open on this thread is not one of the other thread.
                self.assertEqual(capture.counts(mark)["solver_captures_after_startup"], 3)
        finally:
            inside.set()
            thread.join(5)
        self.assertEqual(failures, [])

    def test_second_overload_of_a_generic_kernel_is_one_module_load(self):
        name = "test_capture_scope.census_generic"
        single = wp.zeros(3, dtype=wp.float32, device="cpu")
        wp.launch(fill_any, 3, [single, wp.float32(2.0)], device="cpu")
        builds = {key: digest for key, digest in capture.loaded_modules().items() if key[0] == name}
        self.assertEqual(len(builds), 1)
        (((_, alias, block_size), digest),) = builds.items()
        self.assertEqual(alias, "cpu")
        self.assertIsInstance(block_size, int)
        self.assertIsInstance(digest, bytes)
        mark = self.mark()
        self.assertEqual(mark.startup["startup_module_loads"], 0)
        wp.launch(fill_any, 3, [single, wp.float32(3.0)], device="cpu")
        self.assertEqual(capture.modules(mark)["module_loads_after_startup"], 0)
        double = wp.zeros(3, dtype=wp.float64, device="cpu")
        wp.launch(fill_any, 3, [double, wp.float64(4.0)], device="cpu")
        np.testing.assert_array_equal(double.numpy(), [4.0, 4.0, 4.0])
        late = capture.modules(mark)
        self.assertEqual(late["module_loads_after_startup"], 1)
        self.assertEqual(late["render_module_loads_after_startup"], 0)
        self.assertEqual(late["late_modules"], {name: 1})
        after = {key: value for key, value in capture.loaded_modules().items() if key[0] == name}
        self.assertEqual(list(after), list(builds))
        self.assertNotEqual(after, builds)
        # The same prefix test puts the module on the render side, and only there.
        late = capture.modules(mark, ("test_capture_scope.census_",))
        self.assertEqual((late["module_loads_after_startup"], late["render_module_loads_after_startup"]), (0, 1))
        self.assertEqual((late["late_modules"], late["late_render_modules"]), ({}, {name: 1}))

    def test_render_modules_are_counted_under_render_only_and_tables_hold_sixteen_names(self):
        render = ("newton._src.viewer.",)
        before = {("physics.a", "cuda:0", 256): b"1", ("newton._src.viewer.kernels", "cuda:0", 256): b"1"}
        after = dict(before)
        after[("physics.a", "cuda:0", 256)] = b"2"  # another hash: a kernel gained an overload
        after[("physics.a", "cuda:0", 1)] = b"2"  # another block size
        after[("physics.a", "cpu", 256)] = b"2"  # another device
        after[("physics.b", "cuda:0", 256)] = b"7"
        after[("newton._src.viewer.kernels", "cuda:0", 64)] = b"1"
        after[("newton._src.viewer.gl.kernels", "cuda:0", 256)] = b"3"
        with patch.object(capture, "loaded_modules", return_value=dict(before)):
            begun = capture.begin_startup()
            mark = capture.finish_startup(begun, self.cuda)
        self.assertEqual(mark.startup["startup_module_loads"], 0)
        with patch.object(capture, "loaded_modules", return_value=dict(after)):
            late = capture.modules(mark, render)
            everything = capture.modules(mark)
            # A mark taken now holds these builds: they belong to its start-up.
            later = capture.finish_startup(begun, self.cuda)
            self.assertEqual(later.startup["startup_module_loads"], 6)
            self.assertEqual(capture.modules(later, render)["module_loads_after_startup"], 0)
            self.assertEqual(capture.modules(later, render)["render_module_loads_after_startup"], 0)
        self.assertEqual((late["module_loads_after_startup"], late["render_module_loads_after_startup"]), (4, 2))
        self.assertEqual(late["late_modules"], {"physics.a": 3, "physics.b": 1})
        self.assertEqual(
            late["late_render_modules"], {"newton._src.viewer.gl.kernels": 1, "newton._src.viewer.kernels": 1}
        )
        self.assertEqual(
            (everything["module_loads_after_startup"], everything["render_module_loads_after_startup"]), (6, 0)
        )
        many = dict(before)
        many.update({(f"physics.m{index:02d}", "cuda:0", 256): b"1" for index in range(20)})
        many[("physics.m07", "cuda:0", 1)] = b"1"
        with patch.object(capture, "loaded_modules", return_value=many):
            late = capture.modules(mark, render)
        self.assertEqual(late["module_loads_after_startup"], 21)
        self.assertEqual(capture.TABLE_NAMES, 16)
        self.assertEqual(len(late["late_modules"]), capture.TABLE_NAMES + 1)
        # The largest entries are named, the others are counted.
        self.assertEqual(next(iter(late["late_modules"].items())), ("physics.m07", 2))
        self.assertEqual(late["late_modules"]["other"], 4)
        self.assertNotIn("physics.m16", late["late_modules"])
        self.assertEqual(sum(late["late_modules"].values()), 21)

    def test_late_sites_name_the_caller_and_counts_walk_nothing(self):
        mark = self.mark()
        here = sys._getframe().f_code.co_filename
        with self.devices(self.cuda):
            line = sys._getframe().f_lineno + 1
            with ScopedCapture(device=self.cuda):
                pass
            for _ in range(2):
                with PauseGC():
                    pass
        with patch.object(capture, "loaded_modules", side_effect=AssertionError("module map walked")):
            self.assertEqual(capture.counts(mark), {"captures_after_startup": 1, "solver_captures_after_startup": 2})
        self.assertEqual(capture.modules(mark)["late_sites"], {f"{here}:{line + 3}": 2, f"{here}:{line}": 1})
        # Inside the package a site is named relative to it, in an installed library relative to site-packages.
        frame = SimpleNamespace(
            f_code=SimpleNamespace(co_filename=capture._PACKAGE + "_src/solvers/fem/solver.py"), f_lineno=7
        )
        self.assertEqual(capture._site(frame), "_src/solvers/fem/solver.py:7")
        frame.f_code.co_filename = "/venv/lib/python3.12/site-packages/warp/_src/optim/linear.py"
        self.assertEqual(capture._site(frame), "warp/_src/optim/linear.py:7")
        self.assertTrue(capture.__file__.startswith(capture._PACKAGE + "_src"))

    def test_finish_startup_twice_moves_the_mark_and_keeps_the_record_from_begun(self):
        begun = capture.begin_startup()
        with self.devices(self.cuda):
            with ScopedCapture(device=self.cuda):
                pass
            first = capture.finish_startup(begun, self.cuda)
            self.assertEqual((first.startup["startup_captures"], first.startup["startup_solver_captures"]), (1, 0))
            # What a driver does between the end of the reset and its own mark: the first observation.
            with ScopedCapture(device=self.cuda):
                pass
            with PauseGC():
                pass
            self.assertEqual(capture.counts(first), {"captures_after_startup": 1, "solver_captures_after_startup": 1})
            second = capture.finish_startup(begun, self.cuda)
        self.assertEqual(capture.counts(second), {"captures_after_startup": 0, "solver_captures_after_startup": 0})
        self.assertEqual((second.startup["startup_captures"], second.startup["startup_solver_captures"]), (2, 1))
        self.assertGreaterEqual(second.startup["startup_seconds"], first.startup["startup_seconds"])
        self.assertGreater(first.startup["startup_seconds"], 0.0)
        self.assertEqual(capture.modules(second)["late_sites"], {})
        self.assertEqual(tuple(second.startup), capture.STARTUP_RECORD)
        self.assertEqual(
            capture.STARTUP_RECORD,
            (
                "startup_seconds",
                "startup_captures",
                "startup_solver_captures",
                "startup_module_loads",
                "startup_unlaunched_graphs",
            ),
        )
        # No mark on a CPU device: no CUDA graph is captured there.
        self.assertIsNone(capture.finish_startup(begun, "cpu"))
        self.assertIsNone(capture.finish_startup(begun, wp.get_device("cpu")))

    def test_unlaunched_live_graphs_and_released_graphs_are_reported(self):
        begun = capture.begin_startup()
        base = capture.finish_startup(begun, self.cuda).startup["startup_unlaunched_graphs"]
        launched, waiting = _Graph(True), _Graph(False)
        for graph in (launched, waiting):
            with ScopedCapture(device=self.cuda, graph=graph):
                pass
        # A graph of a CPU scope is not followed.
        cpu_graph = _Graph(False)
        with ScopedCapture(device="cpu", graph=cpu_graph):
            pass
        mark = capture.finish_startup(begun, self.cuda)
        self.assertEqual(mark.startup["startup_unlaunched_graphs"], base + 1)
        waiting.graph_exec = object()
        self.assertEqual(capture.finish_startup(begun, self.cuda).startup["startup_unlaunched_graphs"], base)
        # A graph that its owner dropped is no longer waiting for a launch.
        waiting.graph_exec = None
        del waiting, graph
        gc.collect()
        self.assertEqual(capture.finish_startup(begun, self.cuda).startup["startup_unlaunched_graphs"], base)
        capture.note_release()
        capture.note_release(3)
        self.assertEqual(capture.modules(mark)["graphs_released_after_startup"], 4)
        self.assertEqual(capture.modules(capture.finish_startup(begun, self.cuda))["graphs_released_after_startup"], 0)

    def test_census_equals_warp_captures_of_real_scopes_and_krylov_solves_on_cuda(self):
        self.double.stop()
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("conditional CUDA graphs are unavailable on the selected device")
        import newton

        from newton_solver_suturebot import SolverFEM

        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for point in ((0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1)):
            builder.add_particle(pos=point, vel=(0, 0, 0), mass=0.001)
        model = builder.finalize(device=device)
        solver = SolverFEM(
            model,
            dt=0.002,
            fixed_particle_indices=(0, 1),
            preconditioner_type="block_diag",
            solver_type="cg",
            linear_solver_maxiter=80,
            linear_solver_tolerance=1e-5,
        )
        current, following = model.state(), model.state()
        solver.step(current, following, None, None, 0.002)
        # What Warp itself captures, counted at its entry as the external census tool counts it.
        begun = []
        begin = wp.capture_begin

        def counting(*args, **kwargs):
            begun.append(sys._getframe(1).f_code.co_filename)
            return begin(*args, **kwargs)

        start = capture.begin_startup()
        mark = capture.finish_startup(start, device)
        self.assertEqual(mark.startup["startup_captures"], 0)
        with patch.object(wp, "capture_begin", counting):
            for _ in range(3):
                solver.step(current, following, None, None, 0.002)
            self.assertEqual(capture.counts(mark), {"captures_after_startup": 0, "solver_captures_after_startup": 3})
            self.assertEqual(len(begun), 3)
            with ScopedCapture(device=device) as scope:
                solver.step(current, following, None, None, 0.002)
            self.assertEqual(capture.counts(mark), {"captures_after_startup": 1, "solver_captures_after_startup": 3})
            self.assertEqual(len(begun), 4)
        late = capture.modules(mark)
        self.assertEqual(sum(late["late_sites"].values()), 4)
        # The three solves are named where the solver issues them, the capture where this test opened it.
        solver_site = next(site for site in late["late_sites"] if site.startswith("_src/solvers/fem/solver.py:"))
        self.assertEqual(late["late_sites"][solver_site], 3)
        self.assertEqual(capture.census.graphs[scope.graph].rsplit(":", 1)[0], sys._getframe().f_code.co_filename)
        # The package graph is alive and was never launched; one launch builds its executable graph.
        self.assertIsNone(scope.graph.graph_exec)
        waiting = capture.finish_startup(start, device).startup["startup_unlaunched_graphs"]
        self.assertGreaterEqual(waiting, 1)
        wp.capture_launch(scope.graph)
        self.assertIsNotNone(scope.graph.graph_exec)
        self.assertLess(capture.finish_startup(start, device).startup["startup_unlaunched_graphs"], waiting)
        self.assertEqual(capture.finish_startup(start, device).startup["startup_captures"], 1)
        self.assertEqual(capture.finish_startup(start, device).startup["startup_solver_captures"], 3)


if __name__ == "__main__":
    unittest.main()
