"""CPU lifecycle and CLI regression checks for the standalone grasp viewer."""

import contextlib
import gc
import io
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot.examples._tissue_markers import TissueMarkers
from newton_solver_suturebot.examples.grasp import example_fem_needle_grasp as grasp_example

# A drawn tissue surface of two triangles in the plane z = 0 [m], for the view doubles.
_SURFACE_NODES = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.01, 0.01, 0.0]])
_SURFACE_TRIANGLES = np.array([[0, 1, 2], [1, 3, 2]])


def _small_view(example, offsets):
    """What ``_build_view`` leaves, for a tissue surface of two triangles per world: state, offsets and markers."""
    offsets = np.asarray(offsets, dtype=float)
    particles = np.concatenate([_SURFACE_NODES + offset for offset in offsets]).astype(np.float32)
    example.render_scene = object()
    example.state_0 = SimpleNamespace(particle_q=wp.array(particles, dtype=wp.vec3, device="cpu"))
    example.world_offsets, example.particle_offsets = offsets, None
    example.markers = TissueMarkers(
        example.viewer,
        "needle/entry_exit",
        radius=grasp_example.Example.marker_radius,
        device="cpu",
        per_world=2,
        worlds=len(offsets),
        triangles=_SURFACE_TRIANGLES,
        world_nodes=len(_SURFACE_NODES),
    )
    example._events_drawn = [0] * len(offsets)


class _PoolResource:
    """External worker resource; constructor ownership must survive later errors."""

    def __init__(self, *, fail_observe=False):
        self.fail_observe = fail_observe
        self.close_calls = 0
        self.finish_startup_calls = 0

    def finish_startup(self):
        self.finish_startup_calls += 1

    def observe(self):
        if self.fail_observe:
            raise RuntimeError("worker observation failed")
        return []

    def geometry(self):
        return {}

    def close(self):
        self.close_calls += 1


class _SingleCloseViewer:
    """USD-style close contract: closing a finalized stage again is an error."""

    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        if self.close_calls > 1:
            raise RuntimeError("viewer already finalized")


class GraspExampleLifecycleTests(unittest.TestCase):
    @staticmethod
    def args(**updates):
        defaults = dict(
            device="cpu",
            num_envs=1,
            substeps=4,
            controller="hold",
            # These are CPU checks; the example's own default is the GPU rigid backend.
            rigid_backend="native",
            viewer="null",
            speed=0.003,
            settle_seconds=0.5,
            spacing=0.45,
        )
        return SimpleNamespace(**(defaults | updates))

    def test_constructor_failure_releases_worker_resource_once(self):
        pool = _PoolResource(fail_observe=True)
        with patch.object(grasp_example, "LocalSutureScene", return_value=pool):
            with self.assertRaisesRegex(RuntimeError, "worker observation failed"):
                grasp_example.Example(_SingleCloseViewer(), self.args())
        gc.collect()
        self.assertEqual(pool.close_calls, 1)
        # A constructor that failed did not end the start-up.
        self.assertEqual(pool.finish_startup_calls, 0)

    def test_dropped_example_releases_workers_before_browser_recreation(self):
        pool = _PoolResource()
        with (
            patch.object(grasp_example, "LocalSutureScene", return_value=pool),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            example = grasp_example.Example(_SingleCloseViewer(), self.args())
        self.assertEqual(pool.finish_startup_calls, 1)
        reference = weakref.ref(example)
        del example
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(pool.close_calls, 1)

    def _frozen_example(self, seen, viewer="gl", test=False, **limits):
        """An example whose pool leaves garbage and a long-lived object at start-up, and looks at both when it ends.

        With a view by default (a viewer double that records, ``viewer="gl"``): start-up freezes
        its objects only for a run with a view. ``limits`` are attributes of the viewer double;
        without ``_headless=False`` it has no window and is not drawn before the run.
        """

        class Node:
            pass

        class Viewer(_SingleCloseViewer):
            # A viewer that ends the run by itself: the view is built and nothing is drawn before the run.
            num_frames = 4

            def begin_frame(self, time):
                pass

            def log_state(self, state):
                pass

            def log_points(self, name, points, radii=None, colors=None, hidden=False):
                pass

            def end_frame(self):
                pass

        def build_view(example):
            _small_view(example, np.zeros((1, 3)))

        def pool(config):
            resource = _PoolResource()
            # A long-lived object of start-up that the collector tracks.
            seen["kept"] = resource.kept = [Node()]

            def observe():
                # Garbage of start-up: a reference cycle, which only a collection frees.
                first, second = Node(), Node()
                first.other, second.other = second, first
                seen["garbage"] = weakref.ref(first)
                return [{"entry_position": np.zeros(3)}]

            def finish_startup():
                seen["garbage_at_the_end_of_start_up"] = seen["garbage"]()
                seen["frozen_at_the_end_of_start_up"] = gc.get_freeze_count()

            def close():
                seen.setdefault("frozen_at_release", []).append(gc.get_freeze_count())

            resource.observe, resource.finish_startup, resource.close = observe, finish_startup, close
            return resource

        double = Viewer()
        for name, value in limits.items():
            setattr(double, name, value)
        with (
            patch.object(grasp_example, "LocalSutureScene", side_effect=pool),
            patch.object(grasp_example.Example, "_build_view", build_view),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return grasp_example.Example(double, self.args(viewer=viewer, test=test))

    @staticmethod
    def _tracked(item):
        """Whether a collection would walk the object: frozen objects are in no generation."""
        return any(other is item for other in gc.get_objects())

    @staticmethod
    def _permanent():
        """Objects that the interpreter itself keeps in the permanent generation, with nothing frozen.

        A full collection moves the interpreter's immortal objects there (CPython 3.12: 375 in a
        fresh interpreter), so the freeze count is not zero after one although nothing was frozen.
        The second collection is for an example that the first one frees: its release unfreezes.
        """
        gc.unfreeze()
        gc.collect()
        gc.collect()
        return gc.get_freeze_count()

    def test_start_up_objects_are_collected_and_frozen_before_start_up_ends_and_given_back_at_close(self):
        permanent = self._permanent()
        seen = {}
        example = self._frozen_example(seen)
        # When the scene is told that start-up has ended, its garbage is gone and what survived
        # is out of the collector's reach: no collection in a frame walks the objects of start-up.
        self.assertIsNone(seen["garbage_at_the_end_of_start_up"])
        self.assertGreater(seen["frozen_at_the_end_of_start_up"], permanent)
        self.assertFalse(self._tracked(seen["kept"]))
        self.assertFalse(self._tracked(example))
        # Closing gives every frozen object back, before the scenes are released, once.
        example.close()
        self.assertEqual(len(seen["frozen_at_release"]), 1)
        self.assertLessEqual(seen["frozen_at_release"][0], permanent)
        self.assertLessEqual(gc.get_freeze_count(), permanent)
        self.assertTrue(self._tracked(seen["kept"]))

    def test_dropped_example_gives_its_frozen_objects_back_without_a_collection(self):
        # The viewer's reset drops the example and then collects, before it builds the next one.
        # The drop alone must give the objects back, or that collection could not free them.
        permanent = self._permanent()
        seen = {}
        example = self._frozen_example(seen)
        self.assertGreater(gc.get_freeze_count(), permanent)
        kept = weakref.ref(seen["kept"][0])
        # A cycle through an object of start-up: frozen, it is out of every collection's reach.
        seen["kept"].append(seen["kept"])
        del example
        self.assertLessEqual(gc.get_freeze_count(), permanent)
        self.assertEqual(len(seen["frozen_at_release"]), 1)
        self.assertLessEqual(seen["frozen_at_release"][0], permanent)
        seen.clear()
        self.assertIsNotNone(kept())
        gc.collect()
        self.assertIsNone(kept())

    def test_start_up_that_fails_after_the_freeze_leaves_nothing_frozen(self):
        permanent = self._permanent()
        pool = _PoolResource()
        frozen = []

        def finish_startup():
            frozen.append(gc.get_freeze_count())
            raise RuntimeError("the end of start-up failed")

        pool.finish_startup = finish_startup
        with (
            patch.object(grasp_example, "LocalSutureScene", return_value=pool),
            patch.object(grasp_example.Example, "_build_view", lambda example: None),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "the end of start-up failed"):
                # A viewer that records: a view, which is not drawn before the run.
                grasp_example.Example(SimpleNamespace(num_frames=4), self.args(viewer="usd"))
        self.assertGreater(frozen[0], permanent)
        self.assertLessEqual(gc.get_freeze_count(), permanent)
        self.assertEqual(pool.close_calls, 1)

    def test_run_with_a_view_freezes_and_a_run_without_one_or_a_self_test_without_a_window_does_not(self):
        freezes = grasp_example.Example._freezes_startup
        window, off_screen, recorder = (
            SimpleNamespace(_headless=False),
            SimpleNamespace(_headless=True),
            SimpleNamespace(num_frames=4),
        )
        for viewer, double, test, expected in (
            ("null", _SingleCloseViewer(), False, False),
            ("null", _SingleCloseViewer(), True, False),
            ("gl", window, False, True),
            ("gl", window, True, True),
            ("rtx", window, True, True),
            ("gl", off_screen, False, True),
            ("gl", off_screen, True, False),
            ("usd", recorder, False, True),
            ("usd", recorder, True, False),
            ("rerun", SimpleNamespace(), True, False),
        ):
            with self.subTest(viewer=viewer, double=double, test=test):
                self.assertEqual(freezes(double, self.args(viewer=viewer, test=test)), expected)
        # Arguments without the entry of the self-test: no self-test.
        self.assertTrue(freezes(off_screen, self.args(viewer="gl")))
        self.assertFalse(freezes(window, self.args()))

    def test_example_without_a_view_freezes_nothing_and_leaves_a_freeze_of_the_host_program(self):
        # The null viewer, and a self-test whose viewer has no window (off screen, or a recording).
        for viewer, test in (("null", False), ("null", True), ("gl", True), ("usd", True)):
            with self.subTest(viewer=viewer, test=test):
                permanent = self._permanent()
                host = [[] for _ in range(1000)]
                try:
                    # A freeze of the host program, before the example exists.
                    gc.freeze()
                    frozen = gc.get_freeze_count()
                    self.assertGreater(frozen, permanent + len(host))
                    seen = {}
                    example = self._frozen_example(seen, viewer=viewer, test=test)
                    # Nothing more is frozen at the end of start-up: the objects of start-up and the
                    # example stay within the collector's reach.
                    self.assertLessEqual(seen["frozen_at_the_end_of_start_up"], frozen)
                    self.assertTrue(self._tracked(seen["kept"]))
                    self.assertTrue(self._tracked(example))
                    self.assertFalse(self._tracked(host[0]))
                    # The release gives nothing back: what the host froze stays frozen.
                    example.close()
                    self.assertEqual(len(seen["frozen_at_release"]), 1)
                    self.assertGreater(seen["frozen_at_release"][0], permanent + len(host))
                    self.assertGreater(gc.get_freeze_count(), permanent + len(host))
                    self.assertFalse(self._tracked(host[0]))
                    # A dropped example does not either.
                    seen = {}
                    example = self._frozen_example(seen, viewer=viewer, test=test)
                    del example
                    gc.collect()
                    self.assertEqual(len(seen["frozen_at_release"]), 1)
                    self.assertGreater(gc.get_freeze_count(), permanent + len(host))
                finally:
                    gc.unfreeze()

    def test_self_test_with_a_window_freezes_as_a_run_with_a_view(self):
        permanent = self._permanent()
        seen = {}
        # A window is drawn once before the run and runs until it is closed.
        example = self._frozen_example(seen, test=True, _headless=False)
        self.assertGreater(seen["frozen_at_the_end_of_start_up"], permanent)
        self.assertFalse(self._tracked(seen["kept"]))
        example.close()
        self.assertLessEqual(gc.get_freeze_count(), permanent)

    def test_start_up_that_fails_before_the_freeze_gives_nothing_back(self):
        # The release of an example that froze nothing leaves the collector as it is, also where
        # the run has a view: here the first observation fails, before the view is built.
        permanent = self._permanent()
        host = [[] for _ in range(1000)]
        pool = _PoolResource(fail_observe=True)
        try:
            gc.freeze()
            with patch.object(grasp_example, "LocalSutureScene", return_value=pool):
                with self.assertRaisesRegex(RuntimeError, "worker observation failed"):
                    grasp_example.Example(SimpleNamespace(num_frames=4), self.args(viewer="usd"))
            self.assertEqual(pool.close_calls, 1)
            self.assertGreater(gc.get_freeze_count(), permanent + len(host))
            self.assertFalse(self._tracked(host[0]))
        finally:
            gc.unfreeze()

    def test_full_collections_keep_their_spacing_after_start_up(self):
        # A freeze leaves the collector's count of long-lived objects at the count of start-up,
        # and the collector puts a full collection off until a quarter of that count has gathered
        # again: it would then walk that quarter in one frame. Start-up sets the count to zero,
        # so the next full collection comes after the usual number of young ones, over the few
        # objects of the run, however many objects start-up left.
        if not gc.isenabled() or 0 in gc.get_threshold():
            self.skipTest("the automatic collector is off")
        young, middle, old = gc.get_threshold()
        # Objects that must gather until a full collection is due. A collection is tried for
        # every `young + 1` new objects; the middle generation is collected at every
        # `middle + 2`-th try, and the full collection is due at the try after its
        # `old + 1`-th collection.
        due = (young + 1) * ((middle + 2) * (old + 1) + 1)
        # Start-up objects: a quarter of them is twice what gathers below.
        ballast = [[] for _ in range(8 * due)]
        run = None
        example = self._frozen_example({})
        try:
            full = gc.get_stats()[2]["collections"]
            # One more try than is due, for the objects made since start-up ended.
            run = [[] for _ in range(due + young + 1)]
            self.assertGreater(gc.get_stats()[2]["collections"], full)
            # That collection did not walk the objects of start-up: they are still frozen.
            self.assertFalse(self._tracked(ballast[0]))
        finally:
            example.close()
            del ballast, run

    def _drawn_example(self, calls, offsets=((0.5, 0.0, 0.0),), **limits):
        """An example over a view double whose viewer records its draw calls; ``limits`` are viewer attributes.

        The view double has the drawn state and the markers of a tissue surface of two triangles
        per world, at ``offsets``.
        """

        class Viewer(_SingleCloseViewer):
            def begin_frame(self, time):
                calls.append(("begin_frame", time))

            def log_state(self, state):
                calls.append(("log_state", state))

            def log_points(self, name, points, radii=None, colors=None, hidden=False):
                host = [None if array is None else array.numpy().copy() for array in (points, radii, colors)]
                calls.append(("log_points", name, *host, hidden))

            def end_frame(self):
                calls.append(("end_frame",))

        def build_view(example):
            calls.append(("build_view",))
            _small_view(example, offsets)

        def pool(config):
            resource = _PoolResource()
            resource.observe = lambda: [{"entry_position": np.array([0.01, 0.02, 0.03])}]
            resource.finish_startup = lambda: calls.append(("finish_startup",))
            return resource

        viewer = Viewer()
        for name, value in limits.items():
            setattr(viewer, name, value)
        with (
            patch.object(grasp_example, "LocalSutureScene", side_effect=pool),
            patch.object(grasp_example.Example, "_build_view", build_view),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return grasp_example.Example(viewer, self.args(viewer="null" if limits.get("null") else "gl"))

    def test_view_is_drawn_once_before_start_up_ends_and_nothing_is_drawn_without_a_viewer(self):
        drawn = ["build_view", "begin_frame", "log_state", "log_points", "end_frame", "finish_startup"]
        # A viewer without a frame limit: one that has no such attribute, one whose limit is None,
        # and a window, which runs until it is closed whatever number of frames it was given.
        for limits in ({}, {"num_frames": None}, {"num_frames": 4, "_headless": False}):
            calls = []
            with self.subTest(limits=limits):
                example = self._drawn_example(calls, **limits)
                # One draw of the start state, with the marker batch hidden, and then the end of start-up.
                self.assertEqual([call[0] for call in calls], drawn)
                self.assertEqual(calls[1], ("begin_frame", 0.0))
                self.assertIs(calls[2][1], example.state_0)
                # The batch is logged with its whole reserve, so that a viewer makes room for it
                # here, and no marker is left attached.
                _, name, points, radii, colors, hidden = calls[3]
                reserve = example.markers.capacity
                self.assertEqual(
                    (name, hidden, len(points), len(colors)), ("needle/entry_exit", True, reserve, reserve)
                )
                np.testing.assert_array_equal(
                    radii, np.full(reserve, grasp_example.Example.marker_radius, dtype=np.float32)
                )
                self.assertEqual(example.markers.count, 0)
                example.close()
        # Without a viewer nothing is built and nothing is drawn; the start-up still ends.
        calls = []
        example = self._drawn_example(calls, null=True)
        self.assertEqual(calls, [("finish_startup",)])
        example.close()

    def test_markers_are_bound_to_the_tissue_where_the_needle_went_through_and_ride_on_it(self):
        from newton_solver_suturebot.examples.grasp import _grasp_view

        green, blue = (grasp_example.Example.marker_colors[kind] for kind in ("entry", "exit"))
        offsets = np.array([[0.5, 0.0, 0.0], [0.5, 0.45, 0.0]])
        calls = []
        example = self._drawn_example(calls, offsets=offsets)
        events = [[], []]

        def draw(shift=None):
            """One frame: the scenes report their events so far; the drawn tissue is where ``shift`` puts it."""
            example.observations = [{"diagnostics": {"interaction_events": list(items)}} for items in events]
            if shift is not None:
                moved = np.concatenate([_SURFACE_NODES + offset + shift for offset in offsets])
                example.state_0.particle_q.assign(moved.astype(np.float32))
            del calls[:]
            with patch.object(_grasp_view, "update_scene") as update:
                example.render()
            self.assertEqual(update.call_count, 1)
            self.assertEqual([call[0] for call in calls], ["begin_frame", "log_state", "log_points", "end_frame"])
            return calls[2][1:]

        def event(kind, *position):
            return {"kind": kind, "time_s": 0.0, "position_m": list(position), "reused": False}

        # No event yet: the batch stays hidden.
        self.assertEqual(draw(), ("needle/entry_exit", None, None, None, True))
        # The needle goes in in the first scene, 0.1 mm under the drawn surface: one green marker
        # on the surface above the event, at the scene's offset. A puncture has no marker.
        events[0] += [event("puncture", 0.002, 0.003, 0.0001), event("entry", 0.002, 0.003, -0.0001)]
        name, points, radii, colors, hidden = draw()
        self.assertEqual((name, hidden), ("needle/entry_exit", False))
        np.testing.assert_allclose(points, [[0.502, 0.003, 0.0]], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(radii, np.float32([grasp_example.Example.marker_radius]))
        np.testing.assert_array_equal(colors, np.float32([green]))
        # The tissue moves and no event is new: the marker is drawn where its material point is now.
        lift = np.array([0.0, 0.001, 0.004])
        np.testing.assert_allclose(draw(lift)[1], [[0.502, 0.004, 0.004]], rtol=0, atol=1e-7)
        self.assertEqual(example.markers.count, 1)
        # The needle comes out, above the lifted surface, and goes in in the second scene: each
        # new marker is bound to the surface of its own scene as it is drawn now.
        events[0].append(event("exit", 0.008, 0.008, 0.0042))
        events[1].append(event("entry", 0.001, 0.001, 0.0039))
        _, points, _, colors, _ = draw()
        np.testing.assert_allclose(
            points, [[0.502, 0.004, 0.004], [0.508, 0.008, 0.004], [0.501, 0.451, 0.004]], rtol=0, atol=1e-7
        )
        np.testing.assert_array_equal(colors, np.float32([green, blue, green]))
        # Back at rest the three markers are at the material points they were bound to.
        at_rest = [[0.502, 0.003, 0.0], [0.508, 0.007, 0.0], [0.501, 0.450, 0.0]]
        np.testing.assert_allclose(draw(np.zeros(3))[1], at_rest, rtol=0, atol=1e-7)
        # The first scene is reset: its events start again, and its markers go with them.
        events[0] = [event("entry", 0.009, 0.001, 0.0)]
        _, points, _, colors, _ = draw()
        np.testing.assert_allclose(points, [[0.501, 0.450, 0.0], [0.509, 0.001, 0.0]], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(colors, np.float32([green, green]))
        # After a failure the last drawn state stays, with its markers.
        example.halted = "Physical scene failed"
        del calls[:]
        with patch.object(_grasp_view, "update_scene") as update:
            example.render()
        self.assertEqual(update.call_count, 0)
        np.testing.assert_allclose(calls[2][2], points, rtol=0, atol=0)
        example.close()

    def test_frame_limited_viewer_is_not_drawn_before_the_run_and_simulates_every_frame_it_was_asked_for(self):
        # A viewer that stops by itself counts the frames it is given: the recording viewer in
        # begin_frame, the viewers without a window in end_frame. A draw before the run would be
        # one of them. The view is built and the start-up ends; nothing is drawn.
        for limits in ({"num_frames": 4}, {"num_frames": 4, "_headless": True}, {"num_frames": 0}):
            calls = []
            with self.subTest(limits=limits):
                example = self._drawn_example(calls, **limits)
                self.assertEqual(calls, [("build_view",), ("finish_startup",)])
                example.close()

        class Recorder(_SingleCloseViewer):
            """Counts the frames it is given and stops at its limit, as the USD viewer does."""

            def __init__(self, num_frames):
                super().__init__()
                self.num_frames, self.frames = num_frames, 0

            def begin_frame(self, time):
                self.frames += 1

            def log_state(self, state):
                pass

            def log_points(self, *arguments, **options):
                pass

            def end_frame(self):
                pass

            def is_running(self):
                return self.frames < self.num_frames

        def build_view(example):
            example.render_scene, example.state_0 = None, "start state"
            example.world_offsets = np.array([[0.0, 0.0, 0.0]])

        def pool(config):
            resource = _PoolResource()
            resource.observe = lambda: [{"entry_position": np.zeros(3)}]
            return resource

        viewer = Recorder(4)
        with (
            patch.object(grasp_example, "LocalSutureScene", side_effect=pool),
            patch.object(grasp_example.Example, "_build_view", build_view),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            example = grasp_example.Example(viewer, self.args(viewer="usd"))
        steps = 0
        # The loop of newton.examples.run: a step and a draw while the viewer runs.
        while viewer.is_running():
            steps += 1
            example.render()
        self.assertEqual((steps, viewer.frames), (4, 4))
        example.close()

    def test_frame_limit_is_read_as_the_viewers_of_newton_apply_it(self):
        import newton.viewer

        limited = grasp_example.Example._frame_limited
        # The null viewer is never drawn; it carries a limit as the recording viewers do.
        self.assertTrue(limited(newton.viewer.ViewerNull(num_frames=4)))
        # The viewers with a window option stop at the limit only without a window. Their own
        # is_running is asked on an object that has no window and no renderer of its own.
        for kind, state in (
            (newton.viewer.ViewerGL, {"renderer": SimpleNamespace(has_exit=lambda: False)}),
            (newton.viewer.ViewerRTX, {"_should_close": False}),
        ):
            for headless, num_frames in ((True, 4), (False, 4), (True, None), (False, None)):
                with self.subTest(kind=kind.__name__, headless=headless, num_frames=num_frames):
                    viewer = kind.__new__(kind)
                    vars(viewer).update(state, _headless=headless, num_frames=num_frames, _frame_count=4)
                    # Four frames were counted: a viewer with a limit of four has stopped.
                    self.assertEqual(limited(viewer), not viewer.is_running())
                    self.assertEqual(limited(viewer), headless and num_frames is not None)
        # The recording viewer always applies its limit; without one it records until it is closed.
        for num_frames in (4, None):
            with self.subTest(kind="ViewerUSD", num_frames=num_frames):
                viewer = newton.viewer.ViewerUSD.__new__(newton.viewer.ViewerUSD)
                vars(viewer).update(num_frames=num_frames, _frame_count=4)
                self.assertEqual(limited(viewer), not viewer.is_running())
                self.assertEqual(limited(viewer), num_frames is not None)

    def test_one_scene_runs_in_this_process_and_several_in_workers(self):
        for num_envs, expected in ((1, "LocalSutureScene"), (3, "ParallelSutureScenes")):
            made = []

            def local(config, made=made):
                made.append(("LocalSutureScene", config, None))
                return _PoolResource()

            def workers(config, *, num_envs, made=made):
                made.append(("ParallelSutureScenes", config, num_envs))
                return _PoolResource()

            with (
                self.subTest(num_envs=num_envs),
                patch.object(grasp_example, "LocalSutureScene", side_effect=local),
                patch.object(grasp_example, "ParallelSutureScenes", side_effect=workers),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                example = grasp_example.Example(_SingleCloseViewer(), self.args(num_envs=num_envs))
                self.assertEqual([name for name, _, _ in made], [expected])
                self.assertIs(made[0][1], example.config)
                self.assertEqual(made[0][2], None if num_envs == 1 else num_envs)
                example.close()

    def _run_main(self, *, fail=False):
        viewer = _SingleCloseViewer()
        resources = []
        real_parser = grasp_example.Example.create_parser

        class LightweightExample:
            _instances = weakref.WeakSet()
            create_parser = staticmethod(real_parser)

            def __init__(self, viewer, args):
                self.viewer, self.args = viewer, args
                self.pool = _PoolResource()
                resources.append(self.pool)
                self._instances.add(self)

            def close(self):
                self.pool.close()

        def runner(example, args):
            if fail:
                raise RuntimeError("physical worker failed")
            # Both newton.examples.run and the existing record_examples runner
            # own ordinary successful viewer shutdown.
            example.viewer.close()

        with (
            patch.object(grasp_example, "Example", LightweightExample),
            patch.object(
                grasp_example.newton.examples,
                "init",
                return_value=(viewer, self.args()),
            ),
            patch.object(grasp_example.newton.examples, "run", side_effect=runner),
            patch("sys.argv", ["grasp-example", "--device", "cpu", "--viewer", "null"]),
        ):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "physical worker failed"):
                    grasp_example.main()
            else:
                grasp_example.main()
        return viewer, resources

    def test_successful_standard_or_recording_runner_is_not_closed_twice(self):
        viewer, resources = self._run_main()
        self.assertEqual(viewer.close_calls, 1)
        # The real Example additionally owns weakref finalization; this check
        # focuses on preserving the runner's single-close viewer contract.
        self.assertEqual(len(resources), 1)

    def test_runner_error_closes_viewer_and_releases_workers(self):
        viewer, resources = self._run_main(fail=True)
        self.assertEqual(viewer.close_calls, 1)
        self.assertEqual(resources[0].close_calls, 1)

    def final_state(self, **options):
        example = grasp_example.Example.__new__(grasp_example.Example)
        example.args = self.args(**options)
        example.completed, example.sim_time = 360, 6.0
        example.halted = None
        rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        example.geometry = {
            "tissue_rest_positions": rest,
            "tissue_tetrahedra": np.array([[0, 1, 2, 3]]),
            "fixed_particle_indices": np.array([0]),
        }
        example.observations = [
            {
                "tissue_positions": rest.copy(),
                "rigid_body_poses": np.zeros((1, 7)),
                "thread_positions": np.zeros((2, 3)),
                "diagnostics": {
                    "independent_free_needle": True,
                    "needle_pose_drive": False,
                },
            }
        ]
        example.controller = SimpleNamespace(
            report=lambda: [{"commanded_arc_complete": False, "entry_count": 0, "exit_count": 0}]
        )
        # test_final asks the scenes for one explicit observation; here it is the state above.
        example.observe_calls = 0

        def observe():
            example.observe_calls += 1
            return example.observations

        example.pool = SimpleNamespace(observe=observe)
        return example

    @staticmethod
    def census(status="checked", **values):
        """Diagnostics entries of a scene's capture census: zeros for a checked scene, None for the others."""
        numbers = (
            "captures_after_startup",
            "solver_captures_after_startup",
            "module_loads_after_startup",
            "render_module_loads_after_startup",
        )
        if status == "checked":
            record = {
                "status": status,
                "startup_seconds": 2.0,
                "startup_captures": 37,
                "startup_solver_captures": 4,
                "startup_module_loads": 98,
                "startup_unlaunched_graphs": 0,
                "graphs_released_after_startup": 0,
                "late_sites": {},
                "late_modules": {},
                "late_render_modules": {},
            }
            result = {**dict.fromkeys(numbers, 0), "capture_census": record}
        else:
            # A scene without a start-up mark: its reason, and no number.
            names = (
                "startup_seconds",
                "startup_captures",
                "startup_solver_captures",
                "startup_module_loads",
                "startup_unlaunched_graphs",
                "graphs_released_after_startup",
                "late_sites",
                "late_modules",
                "late_render_modules",
            )
            record = {"status": status, **dict.fromkeys(names)}
            result = {**dict.fromkeys(numbers), "capture_census": record}
        for name, value in values.items():
            (result if name in numbers else record)[name] = value
        return result

    def run_final(self, example):
        """The lines that test_final prints; the last one is its JSON line."""
        import json

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            example.test_final()
        lines = output.getvalue().splitlines()
        return lines[:-1], json.loads(lines[-1])

    def test_final_passes_a_scene_that_captured_and_loaded_nothing_after_start_up(self):
        example = self.final_state(speed=0.001)
        example.observations[0]["diagnostics"].update(self.census())
        printed, line = self.run_final(example)
        self.assertEqual(printed, [])
        self.assertEqual(
            line,
            {
                "physical_checks_passed": True,
                "num_envs": 1,
                "physical_time_s": 6.0,
                "capture_census_status": ["checked"],
                "captures_after_startup": 0,
                "solver_captures_after_startup": 0,
                "module_loads_after_startup": 0,
                "render_module_loads_after_startup": 0,
            },
        )
        # The numbers are those of one explicit observation, which brings the module loads up to
        # date, not those of the last step: the stepped state here says nothing was checked.
        self.assertEqual(example.observe_calls, 1)
        stepped = self.final_state(speed=0.001).observations
        del stepped[0]["tissue_positions"]
        stepped[0]["diagnostics"].update(self.census("start-up not marked"))
        observed, example.observations = example.observations, stepped
        example.pool = SimpleNamespace(observe=lambda: observed)
        printed, line = self.run_final(example)
        self.assertEqual((printed, line["capture_census_status"]), ([], ["checked"]))

    def test_final_raises_for_a_capture_or_a_module_load_after_start_up_and_for_an_unlaunched_graph(self):
        tables = {
            "late_sites": {"_src/solvers/coupled/coupling_graph.py:785": 4, "_src/solvers/fem/certified_cg.py:174": 2},
            "late_modules": {"newton_solver_suturebot._src.solvers.fem.held_start": 1},
        }
        for name in (
            "captures_after_startup",
            "solver_captures_after_startup",
            "module_loads_after_startup",
            "startup_unlaunched_graphs",
        ):
            # A number that was never brought up to date is not a zero.
            for value in (1, 4, None):
                with self.subTest(name=name, value=value):
                    example = self.final_state(speed=0.001)
                    example.observations[0]["diagnostics"].update(self.census(**{name: value}, **tables))
                    with (
                        contextlib.redirect_stdout(io.StringIO()) as output,
                        self.assertRaises(AssertionError) as raised,
                    ):
                        example.test_final()
                    message = str(raised.exception)
                    self.assertIn("after start-up", message)
                    self.assertIn(f"'{name}': {value}", message)
                    # Where the captures were made and which modules were loaded.
                    self.assertIn("coupling_graph.py:785': 4", message)
                    self.assertIn("certified_cg.py:174': 2", message)
                    self.assertIn("fem.held_start': 1", message)
                    # No line that says the checks passed.
                    self.assertNotIn("physical_checks_passed", output.getvalue())
        # The physical checks keep their own failures: an inverted tetrahedron is reported as one.
        example = self.final_state(speed=0.001)
        example.observations[0]["diagnostics"].update(self.census(captures_after_startup=3))
        example.observations[0]["tissue_positions"][3, 2] = -1.0
        with self.assertRaisesRegex(AssertionError, "inverted"):
            example.test_final()

    def test_final_prints_why_a_scene_is_not_checked_and_does_not_raise(self):
        for status in ("native rigid backend", "CPU device", "start-up not marked", "coupled step not captured: eager"):
            with self.subTest(status=status):
                example = self.final_state(speed=0.001)
                example.observations[0]["diagnostics"].update(self.census(status))
                example.observations[0]["env_id"] = 0
                printed, line = self.run_final(example)
                self.assertEqual(len(printed), 1)
                self.assertIn("Scene 0", printed[0])
                self.assertIn(f"not checked: {status}", printed[0])
                self.assertEqual(line["capture_census_status"], [status])
                self.assertEqual(
                    [
                        line[name]
                        for name in (*grasp_example.Example.census_numbers, "render_module_loads_after_startup")
                    ],
                    [None] * 4,
                )
                self.assertTrue(line["physical_checks_passed"])

    def test_final_reports_viewer_module_loads_without_failing_and_skips_diagnostics_without_a_census(self):
        # Kernel modules of the viewer that loaded after start-up are counted apart and reported.
        example = self.final_state(speed=0.001)
        example.observations[0]["diagnostics"].update(
            self.census(render_module_loads_after_startup=2, late_render_modules={"newton._src.viewer.kernels": 2})
        )
        printed, line = self.run_final(example)
        self.assertEqual(printed, [])
        self.assertEqual(line["render_module_loads_after_startup"], 2)
        self.assertEqual([line[name] for name in grasp_example.Example.census_numbers], [0, 0, 0])
        # Diagnostics without a census, as the other tests of this class use them: nothing is
        # checked and nothing is said about it.
        example = self.final_state(speed=0.001)
        printed, line = self.run_final(example)
        self.assertEqual(printed, [])
        self.assertEqual(line["capture_census_status"], [None])
        self.assertIsNone(line["captures_after_startup"])

    def test_final_checks_every_scene_and_names_the_one_that_fails(self):
        def scenes(*entries):
            example = self.final_state(speed=0.001, num_envs=len(entries))
            first = example.observations[0]
            example.observations = []
            for env_id, entry in enumerate(entries):
                item = {name: np.copy(value) for name, value in first.items() if name != "diagnostics"}
                item["diagnostics"] = {**first["diagnostics"], **entry}
                item["env_id"] = env_id
                example.observations.append(item)
            example.controller.report = lambda: [
                {"commanded_arc_complete": False, "entry_count": 0, "exit_count": 0} for _ in entries
            ]
            return example

        # Numbers are summed over the checked scenes; a scene that declined is named with its reason.
        example = scenes(
            self.census(render_module_loads_after_startup=1),
            self.census("native rigid backend"),
            self.census(render_module_loads_after_startup=2),
        )
        printed, line = self.run_final(example)
        self.assertEqual(len(printed), 1)
        self.assertIn("Scene 1", printed[0])
        self.assertEqual(line["capture_census_status"], ["checked", "native rigid backend", "checked"])
        self.assertEqual((line["captures_after_startup"], line["render_module_loads_after_startup"]), (0, 3))
        # One failing scene fails the run, and the message names it and not the others.
        example = scenes(self.census(), self.census(), self.census(module_loads_after_startup=7))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(AssertionError) as raised:
            example.test_final()
        self.assertIn("scene 2", str(raised.exception))
        self.assertNotIn("scene 0", str(raised.exception))
        self.assertNotIn("scene 1", str(raised.exception))

    def test_custom_speed_or_settling_does_not_imply_six_second_completion(self):
        for options in ({"speed": 0.001}, {"settle_seconds": 7.0}):
            with (
                self.subTest(options=options),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.final_state(**options).test_final()

    def test_default_six_second_procedure_still_requires_entry_and_exit(self):
        with self.assertRaisesRegex(AssertionError, "entry and exit"):
            self.final_state().test_final()

    def test_custom_settings_do_not_bypass_physical_integrity(self):
        example = self.final_state(speed=0.001)
        example.observations[0]["tissue_positions"][3, 2] = -1.0
        with self.assertRaisesRegex(AssertionError, "inverted"):
            example.test_final()

    def test_lifts_move_the_completion_check_to_the_end_of_the_sequence(self):
        done = {"commanded_arc_complete": True, "entry_count": 1, "exit_count": 1, "stage": "done"}
        # With a lift the procedure has no fixed duration: it must have ended at the last frame of the
        # self-test, however short or long the run is. A run that ends while the arc is under way fails ...
        example = self.final_state(lift_height=0.01)
        example.controller.report = lambda: [
            {"commanded_arc_complete": False, "entry_count": 0, "exit_count": 0, "stage": "arc", "lifts_complete": 0}
        ]
        for sim_time in (0.5, 6.0, 30.0):
            example.sim_time = sim_time
            with self.subTest(sim_time=sim_time), self.assertRaisesRegex(AssertionError, "entry and exit"):
                example.test_final()
        # ... and so does one that ends in the lift.
        example.controller.report = lambda: [done | {"stage": "lift", "lifts_complete": 0}]
        with self.assertRaisesRegex(AssertionError, "did not complete"):
            example.test_final()
        # At the end the insertion must have happened ...
        example.controller.report = lambda: [done | {"entry_count": 0, "lifts_complete": 1}]
        with self.assertRaisesRegex(AssertionError, "entry and exit"):
            example.test_final()
        # ... and the lift must have completed, ...
        example.controller.report = lambda: [done | {"lifts_complete": 0}]
        with self.assertRaisesRegex(AssertionError, "did not complete"):
            example.test_final()
        example.controller.report = lambda: [done | {"lifts_complete": 1}]
        with contextlib.redirect_stdout(io.StringIO()):
            example.test_final()
        # ... both of them when the tool also lifts by the needle tip.
        example = self.final_state(lift_height=0.01, tip_lift_height=0.005)
        example.controller.report = lambda: [done | {"lifts_complete": 1}]
        with self.assertRaisesRegex(AssertionError, "did not complete"):
            example.test_final()
        example.controller.report = lambda: [done | {"lifts_complete": 2}]
        with contextlib.redirect_stdout(io.StringIO()):
            example.test_final()
        # A run that ends between the two lifts (the regrasp) or in the lift by the tip has not completed them.
        for stage in ("release", "retreat", "approach", "descend", "close", "tip_lift"):
            example.controller.report = lambda stage=stage: [done | {"stage": stage, "lifts_complete": 1}]
            with self.subTest(stage=stage), self.assertRaisesRegex(AssertionError, "did not complete"):
                example.test_final()
        # With the pull the procedure ends with the tail of the needle at its entry, after the one lift.
        example = self.final_state(lift_height=0.01, pull_through=True)
        pulled = done | {"lifts_complete": 1, "pull_complete": True}
        with contextlib.redirect_stdout(io.StringIO()):
            example.controller.report = lambda: [pulled]
            example.test_final()
        # A run that ends on the way to the tip or in the pull fails: its pull has not ended ...
        for stage in ("release", "retreat", "approach", "descend", "close", "pull"):
            example.controller.report = lambda stage=stage: [pulled | {"stage": stage, "pull_complete": False}]
            with self.subTest(stage=stage), self.assertRaisesRegex(AssertionError, "pull of the needle by its tip"):
                example.test_final()
        # ... and one that ends while the tool holds the needle after the pull has not ended either.
        example.controller.report = lambda: [pulled | {"stage": "clear"}]
        with self.assertRaisesRegex(AssertionError, "had not ended at the last frame"):
            example.test_final()
        # Another speed or settling time has another duration: nothing is asked of that procedure.
        for options in ({"speed": 0.001}, {"settle_seconds": 7.0}):
            other = self.final_state(lift_height=0.01, pull_through=True, **options)
            other.controller.report = lambda: [pulled | {"stage": "pull", "pull_complete": False}]
            with self.subTest(options=options), contextlib.redirect_stdout(io.StringIO()):
                other.test_final()
        example.controller.report = lambda: [pulled]
        # The physical checks apply at every stage.
        example.observations[0]["tissue_positions"][3, 2] = -1.0
        with self.assertRaisesRegex(AssertionError, "inverted"):
            example.test_final()

    def test_with_thread_tracks_the_self_test_asks_for_the_needle_out_and_the_thread_through_its_track(self):
        done = {
            "commanded_arc_complete": True,
            "entry_count": 2,
            "exit_count": 2,
            "stage": "done",
            "lifts_complete": 1,
            "pull_complete": True,
            "tail_past_exit_m": 0.002,
        }
        threaded = {
            "eyelets": 8,
            "eyelet_passages": [4, 4],
            "eyelets_threaded": 8,
            "eyelets_refused": 0,
            "thread_inside_tissue_m": 0.009,
        }

        def example_with(report=None, thread=None, engaged=0.0, tracks=True, **options):
            example = self.final_state(**({"lift_height": 0.01, "pull_through": True} | options))
            example.geometry["insertion_plan"] = {"expected_passages": 2}
            if tracks:
                example.geometry["thread_tracks"] = True
            example.controller.report = lambda: [done | (report or {})]
            example.observations[0]["thread_diagnostics"] = threaded | (thread or {})
            example.observations[0]["diagnostics"]["engaged_track_length_m"] = engaged
            return example

        with contextlib.redirect_stdout(io.StringIO()):
            example_with().test_final()
        # The needle must be out: its tail past the exit marker and nothing of its track held by the tissue.
        for change in (
            {"report": {"tail_past_exit_m": -0.0001}},
            {"report": {"tail_past_exit_m": 0.0}},
            {"engaged": 0.0005},
        ):
            with self.subTest(change=change), self.assertRaisesRegex(AssertionError, "not out of the tissue"):
                example_with(**change).test_final()
        # The thread must run through the track: every planned passage has eyelets and each one is threaded.
        for thread in (
            {"eyelets_threaded": 7},
            {"eyelet_passages": [8]},
            {"eyelets": 0, "eyelets_threaded": 0, "eyelet_passages": []},
        ):
            with self.subTest(thread=thread), self.assertRaisesRegex(AssertionError, "does not run through"):
                example_with(thread=thread).test_final()
        with self.assertRaisesRegex(AssertionError, "No thread is inside the tissue"):
            example_with(thread={"thread_inside_tissue_m": 0.0}).test_final()
        # An eyelet that found no room in the reserve fails the self-test, pull or not, also where nothing
        # is asked of the procedure (another speed).
        for options in (
            {},
            {"pull_through": False},
            {"speed": 0.001, "report": {"stage": "lift", "pull_complete": False}},
        ):
            with self.subTest(options=options), self.assertRaisesRegex(AssertionError, "no room in the reserve"):
                example_with(thread={"eyelets_refused": 1}, **options).test_final()
        # A scene whose thread does not follow the track (--no-thread-tracks) ends its pull at the entry:
        # none of the conditions on the exit and the eyelets apply.
        with contextlib.redirect_stdout(io.StringIO()):
            example_with(
                report={"tail_past_exit_m": -0.0115},
                thread={"eyelets": 0, "eyelets_threaded": 0, "eyelet_passages": [], "thread_inside_tissue_m": 0.0},
                engaged=0.009,
                tracks=False,
            ).test_final()
        # A run that ends in the pull through the tissue fails: its pull has not ended.
        with self.assertRaisesRegex(AssertionError, "pull of the needle by its tip"):
            example_with(
                report={"stage": "pull", "pull_complete": False, "tail_past_exit_m": -0.004},
                thread={"eyelets_threaded": 3},
                engaged=0.003,
            ).test_final()

    def test_thread_tracks_are_on_by_default_and_the_switch_reaches_the_scene(self):
        from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene

        parser = grasp_example.Example.create_parser()
        self.assertTrue(grasp_example.THREAD_TRACKS)
        self.assertTrue(parser.parse_args([]).thread_tracks)
        self.assertFalse(parser.parse_args(["--no-thread-tracks"]).thread_tracks)
        scene_parser = GraspScene.create_parser()
        self.assertTrue(scene_parser.parse_args([]).thread_tracks)
        # On: nothing is forwarded, the scene's own default applies (also for an argument namespace without it).
        default = grasp_example.Example._coupling(parser.parse_args([]))
        self.assertNotIn("--no-thread-tracks", default)
        self.assertEqual(grasp_example.Example._coupling(SimpleNamespace()), default)
        self.assertTrue(scene_parser.parse_args(list(default)).thread_tracks)
        # Off: the scene is told, after the other settings.
        off = grasp_example.Example._coupling(parser.parse_args(["--no-thread-tracks"]))
        self.assertEqual(off, (*default, "--no-thread-tracks"))
        self.assertFalse(scene_parser.parse_args(list(off)).thread_tracks)

    def test_failure_raises_without_a_viewer_and_freezes_the_last_state_with_one(self):
        example = self.final_state()
        example.render_scene = None
        with self.assertRaisesRegex(RuntimeError, "Physical scene failed: tip left the pad: {'frame': 3}"):
            example._halt("Physical scene failed: tip left the pad", {"frame": 3})
        self.assertIsNone(example.halted)
        # With a render scene the run stops stepping and keeps what was drawn last.
        example.render_scene = object()
        example.controller = None  # a step would fail on the missing pool and observations
        del example.observations
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            example._halt("Physical scene failed: tip left the pad")
        self.assertEqual(example.halted, "Physical scene failed: tip left the pad")
        self.assertIn("halted at 6.00 s", output.getvalue())
        example.step()
        self.assertEqual(example.completed, 360)
        with self.assertRaisesRegex(AssertionError, "tip left the pad"):
            example.test_final()

    def test_bite_length_moves_entry_and_exit_apart_about_the_recorded_stitch(self):
        from newton_solver_suturebot.examples.grasp._dvrk_layout import load_source_stitch

        self.assertEqual(grasp_example.Example._bite(0.0), {})
        stitch = load_source_stitch()
        entry, exit_point = np.asarray(stitch["entry"][:2]), np.asarray(stitch["exit"][:2])
        recorded = np.linalg.norm(exit_point - entry)
        self.assertAlmostEqual(recorded, 0.0054, delta=1e-4)
        bite = grasp_example.Example._bite(0.010)
        new_entry, new_exit = np.asarray(bite["entry_xy"]), np.asarray(bite["exit_xy"])
        self.assertAlmostEqual(np.linalg.norm(new_exit - new_entry), 0.010, delta=1e-12)
        np.testing.assert_allclose((new_entry + new_exit) / 2, (entry + exit_point) / 2, atol=1e-12)
        # Same line and same direction as the recorded stitch.
        np.testing.assert_allclose((new_exit - new_entry) / 0.010, (exit_point - entry) / recorded, atol=1e-9)
        self.assertEqual(bite["initial_grasp"], grasp_example.Example.deep_bite_grasp)

    def test_viewer_defaults_run_the_whole_sequence_on_the_gpu_backend(self):
        parser = grasp_example.Example.create_parser()
        args = parser.parse_args([])
        self.assertEqual(args.rigid_backend, "mujoco-warp")
        self.assertEqual((args.lift_height, args.bite_length), (0.01, 0.010))
        # After the lift the tool takes the needle by its tip and pulls it on; the pull replaces the lift by the tip.
        example = grasp_example.Example
        self.assertTrue(args.pull_through)
        self.assertIsNone(args.tip_lift_height)
        self.assertEqual((example._pull_through(args), example._tip_lift_height(args)), (True, 0.0))
        # Without the pull the procedure is the earlier one, with its lift of 5 mm by the tip.
        earlier = parser.parse_args(["--no-pull-through"])
        self.assertEqual((example._pull_through(earlier), example._tip_lift_height(earlier)), (False, 0.005))
        shorter = parser.parse_args(["--no-pull-through", "--tip-lift-height", "0.002"])
        self.assertEqual((example._pull_through(shorter), example._tip_lift_height(shorter)), (False, 0.002))
        # Without the first lift the example is the insertion alone: no pull and no lift by the tip.
        alone = parser.parse_args(["--lift-height", "0"])
        self.assertEqual((example._pull_through(alone), example._tip_lift_height(alone)), (False, 0.0))
        # An argument set without the options (a caller of the class) has neither.
        self.assertEqual((example._pull_through(self.args()), example._tip_lift_height(self.args())), (False, 0.0))
        # 40 s at 60 Hz: the measured sequence ends before 30 s of simulated time.
        self.assertEqual(args.num_frames, 2400)
        self.assertEqual(parser.parse_args(["--lift-height", "0"]).lift_height, 0)
        self.assertEqual(parser.parse_args(["--rigid-backend", "native"]).rigid_backend, "native")

    def test_main_rejects_heights_and_bites_outside_their_ranges(self):
        for arguments in (
            ["--lift-height", "0.011"],
            ["--lift-height", "0.002", "--tip-lift-height", "0.003"],
            ["--tip-lift-height", "-0.001"],
            # The earlier procedure's lift of 5 mm by the tip does not fit under a first lift of 2 mm.
            ["--lift-height", "0.002", "--no-pull-through"],
            # The pull replaces the lift by the tip.
            ["--tip-lift-height", "0.004"],
            ["--bite-length", "-0.001"],
            ["--bite-length", "nan"],
        ):
            with (
                self.subTest(arguments=arguments),
                patch("sys.argv", ["grasp-example", *arguments]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                grasp_example.main()

    def test_coupling_defaults_are_this_examples_own_and_each_can_be_overridden(self):
        from newton_solver_suturebot.examples.grasp.parallel import SutureSceneConfig
        from newton_solver_suturebot.examples.needle import example_fem_needle_insertion as insertion

        parser = grasp_example.Example.create_parser()
        args = parser.parse_args([])
        # Measured choices (870 frames, docs/usage.md), not values derived from the trial loop.
        self.assertEqual((grasp_example.COUPLING_RELATIVE_TOLERANCE, grasp_example.TRIAL_BUDGET), (1e-2, 12))
        self.assertEqual((args.coupling_relative_tolerance, args.trial_budget), (1e-2, 12))
        for off in ("0", "00", "none"):
            self.assertIsNone(parser.parse_args(["--trial-budget", off]).trial_budget)
        self.assertEqual(parser.parse_args(["--trial-budget", "20"]).trial_budget, 20)
        # Up to below the trial limit of the scene's coupler.
        self.assertEqual(insertion.COUPLING_TRIAL_LIMIT, 160)
        self.assertEqual(parser.parse_args(["--trial-budget", "159"]).trial_budget, 159)
        self.assertEqual(parser.parse_args(["--coupling-relative-tolerance", "1e-3"]).coupling_relative_tolerance, 1e-3)
        for arguments in (
            ["--trial-budget", "1"],
            ["--trial-budget", "-12"],
            ["--trial-budget", "12.5"],
            ["--trial-budget", "160"],
            ["--trial-budget", "1000"],
            ["--coupling-relative-tolerance", "0"],
            ["--coupling-relative-tolerance", "-1e-2"],
            ["--coupling-relative-tolerance", "nan"],
        ):
            with (
                self.subTest(arguments=arguments),
                patch("sys.argv", ["grasp-example", *arguments]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                grasp_example.main()
        # The insertion example and the scene configuration of Python callers keep theirs.
        defaults = insertion.Example.create_parser().parse_args([])
        self.assertEqual((defaults.coupling_relative_tolerance, defaults.trial_budget), (1e-3, None))
        self.assertEqual(SutureSceneConfig(device="cpu").example_arguments, ())

    def test_example_passes_its_coupling_settings_to_the_scene(self):
        from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene

        parser = grasp_example.Example.create_parser()

        def parsed(*arguments):
            # The two coupling settings of a command line; the other options stay those of these CPU checks.
            options = parser.parse_args(list(arguments))
            return {name: getattr(options, name) for name in ("coupling_relative_tolerance", "trial_budget")}

        # The surface-event options and the response columns per step that the example forwards
        # by default (docs/usage.md), then its default tool/tissue contact; the coupling settings
        # come before them.
        events = ("--event-search", "puncture", "--event-oracle", "device-trial", "--response-basis-nodes", "1")
        events += ("--tool-tissue-contact", "swept")
        cases = (
            # An argument namespace without the two settings, as the lifecycle tests use it.
            ({}, ("--coupling-relative-tolerance", "0.01", "--trial-budget", "12", *events), (1e-2, 12)),
            (parsed(), ("--coupling-relative-tolerance", "0.01", "--trial-budget", "12", *events), (1e-2, 12)),
            (
                parsed("--trial-budget", "none", "--coupling-relative-tolerance", "1e-3"),
                ("--coupling-relative-tolerance", "0.001", *events),
                (1e-3, None),
            ),
            (
                parsed("--trial-budget", "24", "--coupling-relative-tolerance", "0.005"),
                ("--coupling-relative-tolerance", "0.005", "--trial-budget", "24", *events),
                (5e-3, 24),
            ),
        )
        for options, arguments, expected in cases:
            made = []

            def local(config, made=made):
                made.append(config)
                return _PoolResource()

            with (
                self.subTest(arguments=arguments),
                patch.object(grasp_example, "LocalSutureScene", side_effect=local),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                example = grasp_example.Example(_SingleCloseViewer(), self.args(**options))
                example.close()
            self.assertEqual(made[0].example_arguments, arguments)
            # The scene parses them with its own parser: the insertion example's and the tool/tissue contact.
            scene_args = GraspScene.create_parser().parse_args(list(made[0].example_arguments))
            self.assertEqual((scene_args.coupling_relative_tolerance, scene_args.trial_budget), expected)
            self.assertEqual(scene_args.tool_tissue_contact, "swept")

    def test_scene_built_from_the_example_defaults_has_the_tolerance_and_the_budget_on_its_coupler(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from test_needle_insertion_example import _REFERENCE, _selection

        from newton_solver_suturebot.examples.grasp.parallel import SutureSceneConfig, _make_scene
        from newton_solver_suturebot.examples.phantom import example_fem_phantom_regions as phantom

        for arguments, expected in (
            (grasp_example.Example._coupling(grasp_example.Example.create_parser().parse_args([])), (1e-2, 12)),
            # The scene configuration of Python callers: tolerance 1e-3, no budget.
            ((), (1e-3, None)),
        ):
            with (
                self.subTest(arguments=arguments),
                TemporaryDirectory() as directory,
                patch.object(phantom, "list_mesh_regions", return_value=_REFERENCE.regions),
                patch.object(phantom, "load_mesh_regions", side_effect=_selection),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                config = SutureSceneConfig(
                    device="cpu",
                    asset=str(Path(directory) / "pad.usd"),
                    entry_xy=(0, 0.005),
                    exit_xy=(0, -0.005),
                    control_mode="physical_grasp",
                    example_arguments=arguments,
                )
                scene = _make_scene(config, 0)
                try:
                    coupler = scene.coupler
                    self.assertEqual((coupler.relative_tolerance, coupler.trial_budget), expected)
                    self.assertEqual(coupler.max_iterations, 160)
                    report = scene.report()
                    self.assertEqual(report["coupling_relative_tolerance"], expected[0])
                    self.assertEqual(report["coupling_trial_budget"], expected[1])
                    self.assertEqual((report["coupling_budget_keeps"], report["coupling_budget_residual"]), (0, 0.0))
                    self.assertEqual((report["coupling_budget_searches"], report["coupling_budget_events"]), (0, {}))
                    self._check_census_of_a_native_scene(scene, config, report)
                finally:
                    scene.close()
                    scene.viewer.close()

    def _check_census_of_a_native_scene(self, scene, config, report):
        """Census keys of a real scene: no numbers on the native backend, counts from a mark once it has one."""
        import json
        import pickle

        from newton_solver_suturebot._src.utils import capture
        from newton_solver_suturebot.examples.grasp import parallel

        numbers = (
            "captures_after_startup",
            "solver_captures_after_startup",
            "module_loads_after_startup",
            "render_module_loads_after_startup",
        )
        tables = ("graphs_released_after_startup", "late_sites", "late_modules", "late_render_modules")
        # The reset opened a start-up; the scene runs the native rigid backend, so it took no mark.
        self.assertIsNotNone(scene._startup_begun)
        self.assertIsNone(scene._startup_mark)
        self.assertEqual([report[name] for name in numbers], [None] * 4)
        census = report["capture_census"]
        self.assertEqual(list(census), ["status", *capture.STARTUP_RECORD, *tables])
        self.assertEqual(census["status"], "native rigid backend")
        self.assertEqual([value for name, value in census.items() if name != "status"], [None] * 9)
        # The existing keys keep their order: the census is appended.
        self.assertEqual(list(report)[0], "outcome")
        self.assertLess(list(report).index("calibrated"), list(report).index("captures_after_startup"))
        # With a mark the report counts from it and walks no module map; a refresh walks it once.
        scene._startup_mark = capture.finish_startup(scene._startup_begun, SimpleNamespace(is_cuda=True))
        with patch.object(capture, "loaded_modules", side_effect=AssertionError("module map walked")):
            report = scene.report()
        self.assertEqual([report[name] for name in numbers], [0, 0, None, None])
        self.assertEqual(report["capture_census"]["status"], "checked")
        self.assertGreater(report["capture_census"]["startup_seconds"], 0.0)
        self.assertEqual(report["capture_census"]["startup_captures"], 0)
        scene.refresh_capture_census()
        report = scene.report()
        self.assertEqual([report[name] for name in numbers], [0, 0, 0, 0])
        self.assertEqual([report["capture_census"][name] for name in tables], [0, {}, {}, {}])
        # What a worker sends and what the insertion example writes with --report.
        self.assertEqual(json.loads(json.dumps(report["capture_census"])), report["capture_census"])
        self.assertEqual(pickle.loads(pickle.dumps(report))["capture_census"], report["capture_census"])
        # A rebuilt model has no mark: the one of the model before it no longer applies.
        scene.reset()
        self.assertIsNone(scene._startup_mark)
        self.assertIsNone(scene._capture_census)
        # A reset starts up again: the mark and the kept numbers are gone, the reason is back.
        begun = scene._startup_begun
        scene._startup_mark = capture.finish_startup(begun, SimpleNamespace(is_cuda=True))
        parallel._reset_scene(scene, config, rebuild=True)
        self.assertIsNot(scene._startup_begun, begun)
        self.assertIsNone(scene._startup_mark)
        self.assertIsNone(scene._capture_census)
        report = scene.report()
        self.assertEqual([report[name] for name in numbers], [None] * 4)
        self.assertEqual(report["capture_census"]["status"], "native rigid backend")
        # A scene whose start-up call ran is marked by the reset itself, at its end, from the
        # start-up that this reset opened; its first report is a checked one with refreshed numbers.
        finish, marks = capture.finish_startup, []

        def marking(begun, device):
            marks.append((begun, device))
            return finish(begun, SimpleNamespace(is_cuda=True))

        def started(scene):
            # The call comes last in the reset, when the scene is complete.
            self.assertIsNotNone(scene.worker_thread)
            self.assertEqual(scene._startup_status, "start-up not marked")
            return True

        with patch.object(type(scene), "capture_graphs", started), patch.object(capture, "finish_startup", marking):
            parallel._reset_scene(scene, config, rebuild=True)
        self.assertEqual(len(marks), 1)
        self.assertIs(marks[0][0], scene._startup_begun)
        self.assertIs(marks[0][1], scene.device)
        self.assertIsNotNone(scene._startup_mark)
        report = scene.report()
        self.assertEqual(report["capture_census"]["status"], "checked")
        self.assertEqual([report[name] for name in numbers], [0, 0, 0, 0])
        self.assertLess(report["capture_census"]["startup_seconds"], 60.0)


if __name__ == "__main__":
    unittest.main()
