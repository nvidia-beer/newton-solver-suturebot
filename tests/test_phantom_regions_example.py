# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical selected-region interaction checks on two adjoining small boxes."""

import io
import json
import os
import unittest
from contextlib import contextmanager, redirect_stdout
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from _startup_support import final_line, launch_late_kernel, marked_on_every_device, prime, run_script
from newton.viewer import ViewerNull

from newton_solver_suturebot import MeshRegion, RegionMesh, boundary_faces
from newton_solver_suturebot._src.utils import capture
from newton_solver_suturebot.examples.phantom import _surface_pick
from newton_solver_suturebot.examples.phantom import example_fem_phantom_regions as phantom


def _regions():
    points = np.array([(x, y, z) for x in range(3) for y in range(2) for z in range(2)], dtype=float) * 0.01
    tets, labels = [], []
    for region in range(2):
        for first, second, _ in permutations(range(3)):
            nodes = np.asarray([0, 1 << first, (1 << first) | (1 << second), 7]) + 4 * region
            corners = points[nodes]
            if np.linalg.det((corners[1:] - corners[0]).T) < 0:
                nodes[1], nodes[2] = nodes[2], nodes[1]
            tets.append(nodes)
            labels.append(region + 1)
    return points, np.asarray(tets, dtype=np.int32), np.asarray(labels)


_POINTS, _TETS, _LABELS = _regions()
_CATALOG = tuple(MeshRegion(i, f"feature_{i}", f"/Pad/region_{i}", (0.5, 0.7, 0.2), 6, 8) for i in (1, 2))
_FULL_FACES, _ = boundary_faces(_TETS)
_EXTERIOR_KEYS = {tuple(sorted(face)) for face in _FULL_FACES}


def _selection(_path, region_ids):
    selected = np.flatnonzero(np.isin(_LABELS, region_ids))
    tets = _TETS[selected]
    gids = np.unique(tets)
    local = np.searchsorted(gids, tets).astype(np.int32)
    if len(selected):
        faces, owners = boundary_faces(local)
    else:
        faces, owners = np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.int32)
    is_cut = np.asarray([tuple(sorted(gids[face])) not in _EXTERIOR_KEYS for face in faces], dtype=bool)
    return RegionMesh(
        _POINTS[gids],
        local,
        faces,
        _LABELS[selected],
        _LABELS[selected][owners],
        faces[~is_cut],
        _LABELS[selected][owners][~is_cut],
        np.unique(faces[is_cut]),
        gids,
        selected,
        tuple(region for region in _CATALOG if region.id in region_ids),
    )


@contextmanager
def _example(substeps=2, viewer_cls=ViewerNull, numbering=None, arguments=(), example_cls=phantom.Example):
    with (
        TemporaryDirectory() as directory,
        wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu")),
        patch.object(phantom, "list_mesh_regions", return_value=_CATALOG),
        patch.object(phantom, "load_mesh_regions", side_effect=_selection),
    ):
        asset = Path(directory) / "phantom_regions.usd"
        if numbering is not None:
            asset.with_name("suturebot_region_map.json").write_text(json.dumps({"regions": numbering}))
        args = phantom.Example.create_parser().parse_args(
            ["--asset", str(asset), "--regions", "1", "--substeps", str(substeps), *arguments]
        )
        viewer = viewer_cls()
        try:
            yield example_cls(viewer, args)
        finally:
            viewer.close()


@contextmanager
def _self_test(**options):
    """The example as ``--test`` starts it, with its start-up marked on the CPU device as well."""
    with _example() as other:
        prime(other)
    options.setdefault("arguments", ("--test",))
    with marked_on_every_device(), _example(**options) as example:
        yield example


class PhantomRegionsExampleTests(unittest.TestCase):
    def test_one_numbering_selects_original_geometry_without_changing_membership(self):
        numbering = [
            {"id": 1, "source_region_id": 2, "name": "first_wound"},
            {"id": 2, "source_region_id": 1, "name": "remaining_pad"},
        ]
        with _example(numbering=numbering) as example:
            self.assertEqual(example.selected_region_ids, (1,))
            self.assertEqual(example.selection.regions[0].name, "first_wound")
            np.testing.assert_array_equal(example.reference.vertices, _POINTS)
            np.testing.assert_array_equal(example.reference.indices, _TETS)
            np.testing.assert_array_equal(example.selection.global_tetrahedron_indices, np.arange(6, 12))
            self.assertTrue(np.all(example.selection.tetrahedron_region_ids == 1))
            self.assertTrue(np.all(example.selection.vertices[:, 0] >= 0.01))
            example.select_regions([2])
            np.testing.assert_array_equal(example.selection.global_tetrahedron_indices, np.arange(6))
            self.assertTrue(np.all(example.selection.tetrahedron_region_ids == 2))
            for region in example.regions:
                self.assertEqual(example.region_map.region_at(*example.region_map.centers[region.id]), region.id)

    def test_selection_compacts_physics_and_rejects_multiple_active_regions(self):
        with _example() as example:
            self.assertEqual(example.model.tet_count, 6)
            self.assertEqual(example.model.particle_count, 8)
            original_model = example.model
            with self.assertRaisesRegex(ValueError, "Unknown pad regions"):
                example.select_regions([99])
            self.assertIs(example.model, original_model)
            with self.assertRaisesRegex(ValueError, "at most one active FEM region"):
                example.select_regions([1, 2])
            self.assertIs(example.model, original_model)
            self.assertEqual(example.selected_region_ids, (1,))
            self.assertEqual(len(example.fixed_vertex_indices), 6)
            example.step()
            example.test_final()
            example.select_regions([])
            self.assertEqual(example.model.tet_count, 0)
            self.assertIsNone(example.solver)
            example.step()
            self.assertIsNone(example.begin_grab((0.005, 0.005, 0.1), (0, 0, -1)))
            example.select_regions([2])
            self.assertEqual(example.model.tet_count, 6)
            np.testing.assert_array_equal(
                example.state_0.particle_q.numpy(), example.selection.vertices.astype(np.float32)
            )

    def test_inactive_rejects_grab_and_active_spring_deforms_fem(self):
        with _example() as example:
            self.assertIsNone(example.begin_grab((0.015, 0.005, 0.1), (0, 0, -1)))
            hit = example.begin_grab((0.003, 0.004, 0.1), (0, 0, -1))
            self.assertIsNotNone(hit)
            example.move_grab(hit + (0, 0, 0.001))
            for _ in range(20):
                example.step()
            example.test_final()
            displacement = np.linalg.norm(example.state_0.particle_q.numpy() - example.selection.vertices, axis=1)
            self.assertGreater(displacement.max(), 1.0e-5)
            example.render()
            displayed = example.display_points.numpy()
            inactive = ~np.isin(example.reference.global_vertex_indices, example.selection.global_vertex_indices)
            np.testing.assert_array_equal(displayed[inactive], example.reference.vertices[inactive].astype(np.float32))
            example.finish_grab()
            self.assertEqual(example.holds.count, 1)
            example.clear_holds()
            for _ in range(40):
                example.step()
            example.test_final()
            after = np.linalg.norm(example.state_0.particle_q.numpy() - example.selection.vertices, axis=1)
            self.assertLess(after.max(), displacement.max())
            example.reset()
            self.assertEqual(example.holds.count, 0)
            np.testing.assert_array_equal(
                example.state_0.particle_q.numpy(), example.selection.vertices.astype(np.float32)
            )

    def test_model_contains_only_selected_cells_and_inactive_display_stays_at_rest(self):
        with _example() as example:
            for region in (1, 2):
                with self.subTest(region=region):
                    example.select_regions([region])
                    selected = example.selection
                    self.assertEqual(example.model.tet_count, len(selected.indices))
                    self.assertEqual(example.model.particle_count, len(selected.vertices))
                    np.testing.assert_array_equal(example.model.tet_indices.numpy(), selected.indices)
                    self.assertTrue(np.all(selected.tetrahedron_region_ids == region))
                    np.testing.assert_array_equal(
                        selected.global_vertex_indices[example.model.tet_indices.numpy()],
                        example.reference.indices[selected.global_tetrahedron_indices],
                    )
                    active = example._active_to_display.numpy()
                    inverse = np.full(len(example.reference.vertices), -1, dtype=np.int32)
                    inverse[active] = np.arange(len(active))
                    inactive_cells = example.reference.indices[example.reference.tetrahedron_region_ids != region]
                    shared = inverse[np.unique(inactive_cells)]
                    shared = shared[shared >= 0]
                    self.assertTrue(np.isin(shared, example.fixed_vertex_indices).all())
                    self.assertTrue(np.isin(selected.fixed_vertex_indices, example.fixed_vertex_indices).all())
                    underside = np.flatnonzero(np.isin(selected.global_vertex_indices, example._underside_global))
                    self.assertTrue(np.isin(underside, example.fixed_vertex_indices).all())
                    # Probe rendering ownership without asking the solver to
                    # produce a particular deformation: move only free nodes.
                    q = example.state_0.particle_q.numpy()
                    free = np.setdiff1d(np.arange(len(q)), example.fixed_vertex_indices)
                    q[free] += np.array([0.0001, 0.0002, 0.0003], dtype=np.float32)
                    example.state_0.particle_q.assign(q)
                    example.render()
                    displayed = example.display_points.numpy()
                    np.testing.assert_array_equal(displayed[active], q)
                    inactive_faces = example.reference.exterior_surface_triangles[
                        example.reference.exterior_surface_region_ids != region
                    ]
                    inactive_nodes = np.unique(inactive_faces)
                    np.testing.assert_array_equal(
                        displayed[inactive_nodes],
                        example.reference.vertices[inactive_nodes].astype(np.float32),
                    )
            example.select_regions([])
            self.assertEqual(example.model.particle_count, 0)
            self.assertEqual(example.model.tet_count, 0)
            self.assertIsNone(example.solver)
            np.testing.assert_array_equal(example.display_points.numpy(), example.reference.vertices.astype(np.float32))

    def test_odd_substeps_match_eager_after_live_grip_changes(self):
        with _example(substeps=3) as captured, _example(substeps=3) as eager:
            if captured.device.is_cuda:
                self.assertIsNotNone(captured.graph)
            eager.graph = None
            for example in (captured, eager):
                hit = example.begin_grab((0.003, 0.004, 0.1), (0, 0, -1))
                self.assertIsNotNone(hit)
                example.move_grab(hit + (0.0002, 0, 0.0007))
            for _ in range(3):
                captured.step()
                eager.step()
                np.testing.assert_allclose(
                    captured.state_0.particle_q.numpy(), eager.state_0.particle_q.numpy(), rtol=0, atol=2.0e-8
                )
                # Float32 assembly uses atomics: independent captured/captured
                # and eager/eager runs also differ by nanometres. Compare the
                # velocity-induced displacement over one substep (40 nm), not
                # a relative tolerance on velocity components near zero.
                np.testing.assert_allclose(
                    captured.state_0.particle_qd.numpy() * captured.sim_dt,
                    eager.state_0.particle_qd.numpy() * eager.sim_dt,
                    rtol=0,
                    atol=4.0e-8,
                )
            captured.clear_holds()
            eager.clear_holds()
            captured.step()
            eager.step()
            np.testing.assert_allclose(
                captured.state_0.particle_q.numpy(), eager.state_0.particle_q.numpy(), rtol=0, atol=2.0e-8
            )
            captured.test_final()
            eager.test_final()

    def test_model_switch_restores_each_ui_callback_once(self):
        class CallbackViewer(ViewerNull):
            def __init__(self):
                self.callbacks = {"side": [], "free": []}
                super().__init__()

            def set_model(self, model):
                super().set_model(model)
                # ViewerGL clears example-owned callbacks on every model swap.
                self.callbacks = {"side": [], "free": []}

            def register_ui_callback(self, callback, position="side"):
                self.callbacks[position].append(callback)

        with _example(viewer_cls=CallbackViewer) as example:
            # Match initial registration by newton.examples.run and the GL tool.
            example.viewer.register_ui_callback(example.gui, position="side")
            example.viewer.register_ui_callback(example.screen_pick.draw_overlay, position="free")
            for selection in ([2], [], [1], [2]):
                example.select_regions(selection)
                self.assertEqual(example.viewer.callbacks["side"], [example.gui])
                self.assertEqual(example.viewer.callbacks["free"], [example.screen_pick.draw_overlay])
                self.assertIs(example.screen_pick.model, example.model)

    def test_bad_damping_rejected_before_loading(self):
        args = phantom.Example.create_parser().parse_args(["--grip-damping", "nan"])
        with self.assertRaisesRegex(ValueError, "damping"):
            phantom.Example(None, args)

    def test_map_click_selects_one_region_and_rejected_load_keeps_previous_selection(self):
        class UI:
            button_clicked = None

            def button(self, label):
                return label == self.button_clicked

            def get_item_rect_min(self):
                return SimpleNamespace(x=0, y=0)

            def get_item_rect_max(self):
                return SimpleNamespace(x=10, y=10)

            def __getattr__(self, _name):
                return lambda *_args: False

        with _example() as example:
            ui = UI()
            original_model = example.model
            with patch.object(example.region_map, "draw", return_value=1):
                example.gui(ui)
            self.assertEqual(example.selected_region_ids, (1,))
            self.assertIs(example.model, original_model)
            with patch.object(example.region_map, "draw", return_value=2):
                example.gui(ui)
            self.assertEqual(example.selected_region_ids, (2,))
            previous_model = example.model
            with (
                patch.object(example.region_map, "draw", return_value=1),
                patch.object(phantom, "load_mesh_regions", side_effect=ValueError("Nonmanifold cut boundary")),
            ):
                example.gui(ui)
            self.assertEqual(example.selected_region_ids, (2,))
            self.assertIs(example.model, previous_model)
            self.assertIn("Nonmanifold", example.selection_error)
            ui.button_clicked = "None"
            with patch.object(example.region_map, "draw", return_value=None):
                example.gui(ui)
            self.assertEqual(example.selected_region_ids, ())
            self.assertIsNone(example.solver)


class PhantomRegionsStartupTests(unittest.TestCase):
    """Start-up of the example: graph launched, kernel modules loaded, scene drawn, and the check of ``--test``."""

    def test_scene_built_on_the_example_keeps_its_own_start_up(self):
        class Scene(phantom.Example):
            # As the insertion example and the grasp and tool/tissue scenes have them.
            _startup_begun = "opened by the scene"
            _startup_mark = None

        with (
            patch.object(phantom, "load_kernel_modules") as load,
            patch.object(phantom, "first_draw") as draw,
            _example(arguments=("--test",), example_cls=Scene) as scene,
        ):
            self.assertIsNone(scene.startup)
            self.assertIsNone(scene.script)
            for selection in ([2], [], [1]):
                scene.select_regions(selection)
                scene.step()
            scene.reset()
            output = io.StringIO()
            with redirect_stdout(output):
                scene.test_final()
            self.assertEqual(output.getvalue(), "")
            self.assertEqual((scene._startup_begun, scene._startup_mark), ("opened by the scene", None))
        # Nothing of the example's own start-up ran for the scene.
        load.assert_not_called()
        draw.assert_not_called()
        with _example() as example:
            self.assertIsNotNone(example.startup)
            # Without --test there is no script, and test_final checks the state alone.
            self.assertIsNone(example.script)
            example.step()
            output = io.StringIO()
            with redirect_stdout(output):
                example.test_final()
            self.assertEqual(output.getvalue(), "")

    def test_surface_picker_loads_its_kernel_module_when_it_is_built(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        with patch.object(_surface_pick, "load_kernel_modules") as load:
            picker = _surface_pick.SurfacePicker(device)
        load.assert_called_once_with(device, _surface_pick._intersect, _surface_pick._select, _surface_pick._read_hit)
        picker = _surface_pick.SurfacePicker(device)
        loaded = capture.loaded_modules()
        self.assertIn((_surface_pick._intersect.module.name, device.alias, 256 if device.is_cuda else 1), loaded)
        # The first press of a run then loads nothing.
        points = wp.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=wp.vec3, device=device)
        indices = wp.array([0, 1, 2], dtype=int, device=device)
        hit = picker.pick((0.2, 0.2, 1.0), (0, 0, -1), [(points, indices, 1)])
        self.assertEqual(hit[:2], (0, 0))
        self.assertEqual(capture.loaded_modules(), loaded)

    def test_start_state_is_drawn_without_using_a_frame_of_a_limited_viewer(self):
        class Window(ViewerNull):
            # As the OpenGL viewer with a window: it ignores its frame limit.
            _headless = False

        with _example() as example:
            self.assertEqual(example.viewer.frame_count, 0)
            example.select_regions([2])
            self.assertEqual(example.viewer.frame_count, 0)
        with _example(viewer_cls=Window) as example:
            # One whole frame at construction; a rebuild comes from inside a frame and begins none.
            self.assertEqual(example.viewer.frame_count, 1)
            example.select_regions([2])
            example.reset()
            self.assertEqual(example.viewer.frame_count, 1)
        calls = []
        with (
            patch.object(phantom, "load_kernel_modules", lambda device, *kernels: calls.append(kernels)),
            patch.object(phantom, "first_draw", lambda viewer, draw, time, frame: calls.append(frame)),
            _example() as example,
        ):
            example.select_regions([2])
            example.reset()
            example.select_regions([])
        # The kernel of the drawing is loaded and the scene drawn in the constructor and in every rebuild.
        self.assertEqual(calls, [(phantom._scatter_selected,), True] + 3 * [(phantom._scatter_selected,), False])

    def test_self_test_plays_its_script_and_finds_nothing_after_start_up(self):
        with _self_test() as example:
            self.assertEqual(example.script.frames, 100)
            run_script(example)
            line = final_line(example)
            self.assertEqual(
                [label for _, label in example.script.done],
                ["right press", "right drag", "right release"] * 2
                + ["middle click on the first marker", "Clear holds", "region map click"]
                + ["right press", "right drag", "right release", "Reset deformation", "right press", "right drag"]
                + ["Esc", "None", "region map click"],
            )
            self.assertEqual(example._presses, [True] * 4)
            self.assertEqual(example.selected_region_ids, (1,))
            self.assertEqual(example.holds.count, 0)
            self.assertEqual((line["presses"], line["script_frames"], line["script_actions"]), (4, [100, 100], 18))
            # The construction and four rebuilds: the other region, the reset, None, the first region.
            self.assertEqual((line["capture_census_status"], line["startups"]), ("checked", 5))
            for name in (*phantom.Startup.numbers, "render_module_loads_after_startup"):
                self.assertEqual(line[name], 0, name)
            if example.device.is_cuda:
                # One frame graph in every start-up with an active region; two Krylov solves in its eager frame.
                self.assertEqual((line["startup_captures"], line["startup_solver_captures"]), (4, 8))
            # A scripted press that took no hold fails the self-test: the script must do what it says.
            example._presses[1] = False
            with self.assertRaisesRegex(AssertionError, "missed the active region"):
                example.test_final()

    def test_self_test_that_starts_without_an_active_region(self):
        with _self_test(arguments=("--test", "--regions")) as example:
            self.assertEqual(example.selected_region_ids, ())
            run_script(example)
            line = final_line(example)
            # No press while nothing is active; the two presses on the region that the script selects.
            self.assertEqual(example._presses, [True, True])
            self.assertEqual(example.selected_region_ids, ())
            self.assertIsNone(example.solver)
            self.assertEqual((line["presses"], line["script_actions"], line["startups"]), (2, 18, 5))
            for name in phantom.Startup.numbers:
                self.assertEqual(line[name], 0, name)

    def test_late_kernel_module_load_fails_the_self_test(self):
        with _self_test() as example:
            draw = example.render_pad

            def wrong_render_pad():
                # A wrong version: the module of its drawing kernel loads with the fourth frame.
                if example.script.frame == 4:
                    launch_late_kernel(example.device)
                draw()

            example.render_pad = wrong_render_pad
            run_script(example, frames=6)
            with self.assertRaisesRegex(AssertionError, r"'module_loads_after_startup': 1.*late_startup_test_kernel"):
                example.test_final()

    def test_late_load_before_a_rebuild_still_fails_after_it(self):
        with _self_test() as example:
            run_script(example, frames=45)
            # A user action loads a kernel module; the region switch of frame 46 then starts up again.
            launch_late_kernel(example.device)
            run_script(example)
            self.assertEqual(example.startup.startups, 5)
            with self.assertRaisesRegex(AssertionError, "'module_loads_after_startup': 1"):
                example.test_final()

    def test_capture_in_a_frame_fails_the_self_test(self):
        with _self_test() as example:
            if not example.device.is_cuda:
                self.skipTest("A graph needs CUDA")
            run_script(example, frames=3)
            # A wrong version: the frame graph is captured at its first use, inside a frame.
            example._capture_physics()
            run_script(example, frames=3)
            with self.assertRaisesRegex(
                AssertionError, "'captures_after_startup': 1, 'solver_captures_after_startup': 2"
            ):
                example.test_final()
        with _self_test() as example:
            # A wrong version: no frame graph, so stock Warp captures a graph for every Krylov solve of a frame.
            example.graph = None
            run_script(example, frames=3)
            with self.assertRaisesRegex(AssertionError, "'solver_captures_after_startup': 6"):
                example.test_final()

    def test_frame_graph_is_launched_at_start_up_and_the_rest_state_is_put_back(self):
        with _self_test() as example:
            if not example.device.is_cuda:
                self.skipTest("A graph needs CUDA")
            self.assertIsNotNone(example.graph.graph_exec)
            census = example.startup.census()
            self.assertEqual((census["startup_captures"], census["startup_unlaunched_graphs"]), (1, 0))
            np.testing.assert_array_equal(
                example.state_0.particle_q.numpy(), example.selection.vertices.astype(np.float32)
            )
            for array in (example.state_0.particle_qd, example.state_1.particle_qd, example.solver.velocity_increment):
                np.testing.assert_array_equal(array.numpy(), 0.0)
        with patch.object(phantom, "launch_startup_graph", lambda graph, device: None), _self_test() as example:
            # A wrong version: the graphs are captured at start-up and first launched by a frame.
            self.assertIsNone(example.graph.graph_exec)
            run_script(example)
            with self.assertRaisesRegex(AssertionError, "'startup_unlaunched_graphs': 4"):
                example.test_final()

    def test_start_up_frames_leave_later_states_bit_identical(self):
        """A frame that runs at start-up and is put back does not change the run.

        On CUDA the start-up runs the frame twice, eager before the capture and as the launched
        graph. Here a scene that ran one such frame is compared with scenes that ran two and
        three. A scene that ran none differs from them in rounding: its first frame assembles
        the tangent while it builds the sparse structure.
        """
        with _example() as one, _example() as two, _example() as three:
            if one.device.is_cuda:
                self.skipTest("Float32 assembly on CUDA uses atomics: two runs of one scene differ in rounding")
            for scene, frames in ((one, 1), (two, 2), (three, 3)):
                for _ in range(frames):
                    restore = scene._frame_snapshot()
                    scene.simulate()
                    restore()
                np.testing.assert_array_equal(
                    scene.state_0.particle_q.numpy(), scene.selection.vertices.astype(np.float32)
                )
                hit = scene.begin_grab((0.003, 0.004, 0.1), (0, 0, -1))
                scene.move_grab(hit + (0.0002, 0, 0.0007))
            for _ in range(12):
                for scene in (one, two, three):
                    scene.step()
                for other in (two, three):
                    for name in ("particle_q", "particle_qd"):
                        np.testing.assert_array_equal(
                            getattr(one.state_0, name).numpy(), getattr(other.state_0, name).numpy()
                        )
            displacement = np.abs(one.state_0.particle_q.numpy() - one.selection.vertices).max()
            self.assertGreater(displacement, 1.0e-5)


if __name__ == "__main__":
    unittest.main()
