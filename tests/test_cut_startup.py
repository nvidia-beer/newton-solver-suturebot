# SPDX-License-Identifier: Apache-2.0
"""Start-up, self-test and fixed reserve of the cutting example on the six-tetrahedron cube."""

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import warp as wp
from _startup_support import final_line, launch_late_kernel, marked_on_every_device, run_script
from newton.viewer import ViewerNull
from test_cut_pipeline import _blade, _cube

from newton_solver_suturebot import LinearTetCut
from newton_solver_suturebot._src.utils import capture
from newton_solver_suturebot.examples.incision import example_fem_surgical_incision as cutting

_FREE_FALL = (
    "--drop_height",
    "1",
    "--linear_damping",
    "0",
    "--material_damping",
    "0",
    "--ground_stiffness",
    "0",
    "--piece_contact_stiffness",
    "0",
)


@contextmanager
def _example(*arguments, example_cls=None, fragment_capacity=512):
    capacity = () if fragment_capacity is None else ("--fragment_capacity", str(fragment_capacity))
    args = cutting.Example.create_parser().parse_args([*capacity, *arguments])
    device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
    with wp.ScopedDevice(device), patch.object(cutting, "load_body", side_effect=lambda *_: _cube()):
        viewer = ViewerNull()
        try:
            yield (example_cls or cutting.Example)(viewer, args)
        finally:
            viewer.close()


@contextmanager
def _self_test(*arguments):
    """The example as ``--test`` starts it, with its start-up marked on the CPU device as well.

    Another scene plays the script first: on the CPU the kernel modules of a frame and of an edit
    load with their first launch, which the eager start-up frame of a CUDA scene does at start-up.
    """
    if not _primed:
        with _example("--test", "--cut_reserve", "1") as other:
            run_script(other)
        _primed.append(True)
    with marked_on_every_device(), _example("--test", *arguments) as example:
        yield example


# Whether a scene of this process has played the script (see _self_test).
_primed = []


def _resting_blade(x=0.003):
    """A vertical blade through the cube that rests on the floor, across its whole width."""
    polygon = np.asarray(((x, -0.2, -0.1), (x, 0.2, -0.1), (x, 0.2, 0.3), (x, -0.2, 0.3)))
    return polygon[np.asarray(((0, 1, 2), (0, 2, 3)))]


def _counted_captures():
    """Captures of the package scope and Krylov solves outside a graph, so far in this process."""
    return capture.census.captures + capture.census.solver_captures


def _state(example):
    arrays = {
        "q": example.state_0.particle_q,
        "qd": example.state_0.particle_qd,
        "q1": example.state_1.particle_q,
        "qd1": example.state_1.particle_qd,
        "dv": example.solver.velocity_increment,
        "driven": example.holds.driven,
        "limited": example.volume_guard.limited_steps,
        "fractions": example.cutter.volume_fractions,
        "mass": example.model.cut_physical_mass,
    }
    return {name: array.numpy().copy() for name, array in arrays.items()}


def _storage_pointers(example):
    cutter = example.cutter
    arrays = {
        f"geometry.{name}": getattr(cutter.geometry, name) for name in ("vertices", "faces", "face_size", "parent")
    }
    arrays.update({f"scratch.{name}": getattr(cutter._sphere.scratch[0], name) for name in ("vertices", "faces")})
    arrays.update(work=cutter._sphere.work, adjacency=cutter.adjacency.points, surface=example.surface.components)
    arrays.update(contact=example.contact.points)
    return {name: array.ptr for name, array in arrays.items()}


class Scene(cutting.Example):
    """A scene built on the example: it does not own a start-up."""


class CutStartupTests(unittest.TestCase):
    def test_reset_prepares_and_launches_every_graph_and_puts_the_start_state_back(self):
        with _example() as example:
            cutter = example.cutter
            self.assertIsNotNone(cutter._sphere)
            self.assertEqual(len(cutter._sphere.work), 256)
            # The subtractor's scratch starts at the cutter's limit: no first cylinder grows it.
            self.assertEqual(cutter.subtractor_storage, (LinearTetCut.SCRATCH_LIMIT, LinearTetCut.SCRATCH_LIMIT, 256))
            pool = np.tile(example.reference_positions, (example.args.node_copies, 1)).astype(np.float32)
            state = _state(example)
            np.testing.assert_array_equal(state["q"], pool)
            np.testing.assert_array_equal(state["q1"], pool)
            for name in ("qd", "qd1", "dv", "driven", "limited"):
                np.testing.assert_array_equal(state[name], 0, err_msg=name)
            # Integrals and boundary of the uncut cube: a cutter that prepared nothing holds the same.
            plain = LinearTetCut(
                example.reference_positions, example.reference_elements, fragment_capacity=512, device=example.device
            )
            np.testing.assert_array_equal(cutter.volume_fractions.numpy(), plain.volume_fractions.numpy())
            np.testing.assert_array_equal(cutter.mass_integrals.numpy(), plain.mass_integrals.numpy())
            self.assertEqual(int(cutter.surface.count.numpy()[0]), int(plain.surface.count.numpy()[0]))
            self.assertEqual(int(cutter.geometry.count.numpy()[0]), 6)
            self.assertEqual((example.cut_revision, example.storage_growths, cutter.storage_growths), (0, [], []))
            if not example.device.is_cuda:
                self.assertEqual((cutter._graphs, cutter._sphere_graph, example.graph), ({}, None, None))
                self.assertEqual(example.startup.census()["capture_census_status"], "CPU device")
                return
            self.assertEqual(sorted(cutter._graphs), [1, 2])
            for graph in (*(graph for graph, _ in cutter._graphs.values()), cutter._sphere_graph, example.graph):
                self.assertIsNotNone(graph.graph_exec)
            census = example.startup.census()
            # Two line graphs, the subtractor's graph and the frame graph.
            self.assertEqual((census["startup_captures"], census["startup_unlaunched_graphs"]), (4, 0))

    def test_reset_starts_up_again(self):
        with _example() as example:
            self.assertTrue(example.cut(_resting_blade()), example.cut_error)
            example.step()
            before = _counted_captures()
            example.reset()
            self.assertEqual(example.startup.startups, 2 if example.device.is_cuda else 0)
            self.assertEqual((example.cut_revision, example.fragment_count), (0, 6))
            if example.device.is_cuda:
                census = example.startup.census()
                self.assertEqual(census["startup_unlaunched_graphs"], 0)
                self.assertGreater(_counted_captures(), before)
                self.assertEqual((census["captures_after_startup"], census["solver_captures_after_startup"]), (0, 0))
                self.assertEqual(sorted(example.cutter._graphs), [1, 2])

    def test_edits_within_the_storage_capture_build_and_load_nothing(self):
        with _example(*_FREE_FALL, "--surface_capacity", "4096") as example:
            example.step()
            before = _counted_captures()
            executables = (
                [graph.graph_exec for graph, _ in example.cutter._graphs.values()] if example.device.is_cuda else []
            )
            self.assertTrue(example.cut(_blade()), example.cut_error)
            example.step()
            two_planes = np.asarray(
                (
                    ((-0.02, -0.2, 0.8), (-0.02, -0.01, 0.8), (-0.02, -0.01, 1.3)),
                    ((0.01, 0.02, 0.8), (0.2, 0.02, 0.8), (0.2, 0.02, 1.3)),
                )
            )
            self.assertTrue(example.cut(two_planes), example.cut_error)
            example.step()
            self.assertEqual(_counted_captures(), before)
            self.assertEqual(example.storage_growths, [])
            if example.device.is_cuda:
                self.assertEqual([graph.graph_exec for graph, _ in example.cutter._graphs.values()], executables)

    def test_self_test_passes_and_lists_the_captures_of_storage_growth_apart(self):
        with _self_test() as example:
            run_script(example)
            line = final_line(example)
            self.assertEqual(line["scripted_edits"], ["line", "line of two patches", "cylinder", "line"])
            self.assertEqual((line["presses"], line["script_idle"], line["startups"]), (2, False, 2))
            self.assertEqual((line["module_loads_after_startup"], line["startup_unlaunched_graphs"]), (0, 0))
            self.assertIsNone(line["cut_reserve"])
            growths = line["storage_growths"]
            # The cylinder of the script: the boundary outgrows its reserve. The subtractor's scratch
            # does not grow: it starts at the cutter's limit, and the census shows it.
            names = [growth["what"] for growth in growths]
            self.assertIn("boundary surface triangles", names)
            self.assertFalse([name for name in names if name.startswith("subtractor scratch")])
            self.assertEqual(line["subtractor_storage"], [64, 64, 256])
            late = line["captures_after_startup"] + line["solver_captures_after_startup"]
            self.assertEqual(late, line["captures_forced_by_storage_growth"])
            self.assertEqual(late, sum(growth["captures"] for growth in growths))
            if example.device.is_cuda:
                self.assertGreater(late, 0)
            else:
                self.assertEqual(late, 0)

    def test_capture_that_no_growth_forced_fails_the_self_test(self):
        with _self_test() as example:
            if not example.device.is_cuda:
                self.skipTest("A graph needs CUDA")
            # A wrong version: the graph of the line cut is not prepared at start-up.
            example.cutter._graphs.clear()
            run_script(example, frames=45)
            with self.assertRaisesRegex(AssertionError, "'captures_after_startup': 1.*tetrahedral_cut.py"):
                example.test_final()
            # The cylinder of frame 82 grows storage: its captures are set apart, those of the two line cuts
            # (one patch and two patches) are not.
            run_script(example, frames=45)
            self.assertTrue(example.storage_growths)
            with self.assertRaisesRegex(
                AssertionError,
                "no storage growth forced.*'captures_after_startup': 2, 'solver_captures_after_startup': 0",
            ):
                example.test_final()

    def test_late_kernel_module_load_fails_the_self_test(self):
        with _self_test() as example:
            run_script(example, frames=4)
            launch_late_kernel(example.device)
            run_script(example, frames=2)
            with self.assertRaisesRegex(AssertionError, r"'module_loads_after_startup': 1.*late_startup_test_kernel"):
                example.test_final()

    def test_scripted_user_stands_down_for_somebody_else(self):
        with _self_test() as example:
            run_script(example, frames=30)
            self.assertTrue(example.cut(_resting_blade()), example.cut_error)
            run_script(example)
            line = final_line(example)
            self.assertEqual(
                (line["script_idle"], line["cuts"], line["scripted_edits"], line["presses"]), (True, 1, [], 0)
            )
            self.assertEqual(line["startups"], 1)

    def test_refused_scripted_edit_fails_the_self_test(self):
        with _self_test("--fragment_capacity", "8") as example:
            run_script(example)
            with self.assertRaisesRegex(AssertionError, "Scripted edits were not accepted.*reserve exhausted"):
                example.test_final()

    def test_start_up_frames_leave_later_states_bit_identical(self):
        """A frame that runs at start-up and is put back does not change the run.

        On CUDA the start-up runs the frame three times: eager before the capture, and as the
        launched graph. Here scenes that ran one, two and three such frames are compared.
        """
        with _example(*_FREE_FALL) as one, _example(*_FREE_FALL) as two, _example(*_FREE_FALL) as three:
            if one.device.is_cuda:
                self.skipTest("Float32 assembly on CUDA uses atomics: two runs of one scene differ in rounding")
            for scene, frames in ((one, 1), (two, 2), (three, 3)):
                for _ in range(frames):
                    restore = scene._frame_snapshot()
                    scene.simulate()
                    restore()
                self.assertTrue(scene.cut(_blade()), scene.cut_error)
            for _ in range(8):
                for scene in (one, two, three):
                    scene.step()
                for other in (two, three):
                    for name, values in _state(one).items():
                        np.testing.assert_array_equal(_state(other)[name], values, err_msg=name)

    def test_scene_with_and_without_its_own_start_up_runs_the_same_bits(self):
        with _example(*_FREE_FALL) as own, _example(*_FREE_FALL, example_cls=Scene) as scene:
            if own.device.is_cuda:
                self.skipTest("Float32 assembly on CUDA uses atomics: two runs of one scene differ in rounding")
            self.assertIsNotNone(own.startup)
            self.assertIsNone(scene.startup)
            center = (0.003, 0.004, 1.05)
            for example in (own, scene):
                example.step()
                self.assertTrue(example.cut(_blade(ymax=0.013)), example.cut_error)
                example.step()
                self.assertTrue(example.cut_sphere(center, center, 0.018), example.cut_error)
                for _ in range(4):
                    example.step()
                example.reset()
                self.assertTrue(example.cut(_blade()), example.cut_error)
                for _ in range(4):
                    example.step()
            for name, values in _state(own).items():
                np.testing.assert_array_equal(_state(scene)[name], values, err_msg=name)
            self.assertEqual(
                [growth["what"] for growth in own.storage_growths], [growth["what"] for growth in scene.storage_growths]
            )


class CutStorageGrowthTests(unittest.TestCase):
    def test_growth_is_listed_with_its_frame_and_its_captures(self):
        with _example(*_FREE_FALL, "--surface_capacity", "12") as example:
            example.step()
            example.step()
            before = _counted_captures()
            self.assertTrue(example.cut(_blade()), example.cut_error)
            (growth,) = example.storage_growths
            self.assertEqual(
                (growth["what"], growth["frame"], growth["from"], growth["completed"]),
                ("boundary surface triangles", 1, 12, True),
            )
            self.assertGreaterEqual(growth["to"], example._surface_count)
            self.assertGreater(growth["seconds"], 0.0)
            self.assertEqual(growth["captures"] + growth["solver_captures"], _counted_captures() - before)
            if example.device.is_cuda:
                # The frame graph, and the Krylov solves of the eager frame that precedes its capture.
                self.assertEqual(growth["captures"], 1)
                self.assertGreater(growth["solver_captures"], 0)
                self.assertIsNotNone(example.graph.graph_exec)

    def test_polygon_growth_captures_every_cutter_graph_inside_the_growth(self):
        with _example(*_FREE_FALL, "--surface_capacity", "8192") as example:
            example.step()
            center = (0.003, 0.004, 1.05)
            self.assertTrue(example.cut(_blade(ymax=0.013)), example.cut_error)
            self.assertEqual(example.storage_growths, [])
            before = _counted_captures()
            # The sphere through the slit leaves pieces of more than twelve corners.
            self.assertTrue(example.cut_sphere(center, center, 0.018), example.cut_error)
            names = [growth["what"] for growth in example.storage_growths]
            self.assertIn("polygon vertices per fragment", names)
            self.assertEqual(example.cutter.geometry.vertex_capacity, 48)
            self.assertTrue(all(growth["completed"] and growth["frame"] == 0 for growth in example.storage_growths))
            self.assertEqual(
                sum(growth["captures"] for growth in example.storage_growths), _counted_captures() - before
            )
            if not example.device.is_cuda:
                return
            # The subtractor's graph and both graphs of the line cut were bound to the replaced buffers.
            polygons = [growth for growth in example.storage_growths if growth["what"].startswith("polygon")]
            self.assertTrue(all(growth["captures"] == 3 for growth in polygons), polygons)
            self.assertEqual(sorted(example.cutter._graphs), [1, 2])
            for graph in (*(graph for graph, _ in example.cutter._graphs.values()), example.cutter._sphere_graph):
                self.assertIsNotNone(graph.graph_exec)
            after = _counted_captures()
            self.assertTrue(example.cut(_blade()), example.cut_error)
            self.assertEqual(_counted_captures(), after)


class CutReserveTests(unittest.TestCase):
    def test_reserve_is_derived_from_the_mesh_and_the_tools(self):
        points, elements = _cube()
        reserve = cutting.derive_reserve(points, elements, 1.0)
        # Every tetrahedron of the cube holds the body diagonal: both central sections cross all six,
        # and a tool at the centre touches all six bounding boxes.
        self.assertEqual((reserve["crossed_tetrahedra"], reserve["column_tetrahedra"]), (12, 6))
        self.assertEqual(reserve["fragment_capacity"], 6 + 12 + (18 + 30 - 2) * 6)
        self.assertEqual(reserve["work_capacity"], 256)
        self.assertEqual(reserve["surface_triangles"], 4 * 12 + 5 * (18 + 30) * 6)
        self.assertEqual((reserve["vertex_capacity"], reserve["face_capacity"]), (48, 32))
        self.assertEqual((reserve["scratch_vertex_capacity"], reserve["scratch_face_capacity"]), (64, 64))
        double = cutting.derive_reserve(points, elements, 2.0, (64, 34))
        self.assertEqual(double["fragment_capacity"] - 6, 2 * (reserve["fragment_capacity"] - 6))
        self.assertEqual(double["surface_triangles"], 2 * reserve["surface_triangles"])
        self.assertEqual((double["vertex_capacity"], double["face_capacity"]), (64, 34))

    def test_column_count_matches_a_search_over_tool_positions(self):
        rng = np.random.default_rng(3)
        # A slab of small tetrahedra: 5 mm cells, each split into six.
        cells = [(x, y, z) for x in range(6) for y in range(5) for z in range(2)]
        corner_points, cube_elements = _cube()
        unit = (corner_points - corner_points.min(axis=0)) / 0.1
        points = np.concatenate([(unit + cell) * 0.005 for cell in cells])
        points += rng.uniform(-2.0e-4, 2.0e-4, points.shape)
        elements = np.concatenate([cube_elements + 8 * index for index in range(len(cells))])
        reserve = cutting.derive_reserve(points, elements, 1.0)
        corners = points[elements]
        lower, upper = corners.min(axis=1)[:, :2], corners.max(axis=1)[:, :2]
        radius = cutting.RESERVE_TOOL_RADIUS
        most = 0
        for center in rng.uniform(points.min(axis=0)[:2], points.max(axis=0)[:2], (4000, 2)):
            touched = np.all((lower < center + radius) & (upper > center - radius), axis=1)
            most = max(most, int(np.count_nonzero(touched)))
        self.assertGreaterEqual(reserve["column_tetrahedra"], most)
        self.assertLessEqual(reserve["column_tetrahedra"], 4 * most)

    def test_reserved_storage_never_grows_and_captures_nothing(self):
        with _example(*_FREE_FALL, "--cut_reserve", "1", fragment_capacity=None) as example:
            cutter, reserve = example.cutter, example.reserve
            self.assertTrue(cutter.fixed_storage)
            self.assertEqual((cutter.geometry.vertex_capacity, cutter.geometry.face_capacity), (48, 32))
            self.assertEqual(cutter.capacity, reserve["fragment_capacity"])
            scratch = cutter._sphere.scratch[0]
            self.assertEqual((scratch.vertex_capacity, scratch.face_capacity, len(cutter._sphere.work)), (64, 64, 256))
            self.assertEqual(example.contact.capacity, example._surface_count + reserve["surface_triangles"])
            example.step()
            pointers, before = _storage_pointers(example), _counted_captures()
            center = (0.003, 0.004, 1.05)
            self.assertTrue(example.cut(_blade(ymax=0.013)), example.cut_error)
            self.assertTrue(example.cut_sphere(center, center, 0.018), example.cut_error)
            self.assertTrue(example.cut_cylinder((0.02, -0.02, 1.2), (0.02, -0.02, 0.9), 0.004), example.cut_error)
            example.step()
            self.assertEqual(_storage_pointers(example), pointers)
            self.assertEqual((example.storage_growths, cutter.storage_growths), ([], []))
            self.assertEqual(_counted_captures(), before)
            example.test_final()

    def test_edit_beyond_the_polygon_reserve_is_refused_and_leaves_the_pad_as_it_was(self):
        with _example(
            *_FREE_FALL, "--cut_reserve", "1", "--cut_reserve_polygons", "12", "12", fragment_capacity=None
        ) as example:
            example.step()
            self.assertTrue(example.cut(_blade(ymax=0.013)), example.cut_error)
            example.step()
            pointers, before = _storage_pointers(example), _counted_captures()
            state, counts = _state(example), (example.fragment_count, example._surface_count, example.cut_revision)
            center = (0.003, 0.004, 1.05)
            self.assertFalse(example.cut_sphere(center, center, 0.018))
            self.assertRegex(
                example.cut_error, "^Cut refused: the fixed cut reserve is exhausted; restart with a larger"
            )
            self.assertIn("'refused_growth': 'polygon vertices per fragment'", example.cut_error)
            self.assertEqual(example.cutter.refused_growth, "polygon vertices per fragment")
            self.assertEqual((example.fragment_count, example._surface_count, example.cut_revision), counts)
            self.assertEqual(_storage_pointers(example), pointers)
            self.assertEqual((example.storage_growths, _counted_captures()), ([], before))
            for name, values in _state(example).items():
                np.testing.assert_array_equal(values, state[name], err_msg=name)
            # The pad stays usable: the next edit within the reserve is accepted.
            self.assertTrue(example.cut(_blade()), example.cut_error)
            self.assertEqual(example.cut_error, "")
            self.assertIsNone(example.cutter.refused_growth)
            example.step()
            example.test_final()

    def test_edit_beyond_the_surface_reserve_is_refused(self):
        with _example(*_FREE_FALL, "--cut_reserve", "1", "--surface_capacity", "12", fragment_capacity=None) as example:
            example.step()
            state, counts = _state(example), (example.fragment_count, example._surface_count, example.cut_revision)
            contact, graph = example.contact, example.graph
            self.assertFalse(example.cut(_blade()))
            self.assertRegex(example.cut_error, "^Cut refused: boundary surface reserve exhausted: the cut needs")
            self.assertEqual((example.fragment_count, example._surface_count, example.cut_revision), counts)
            self.assertIs(example.contact, contact)
            self.assertIs(example.graph, graph)
            self.assertEqual(example.storage_growths, [])
            # The rollback of a published edit copies the accepted state into both buffers.
            np.testing.assert_array_equal(example.state_1.particle_q.numpy(), state["q"])
            for name, values in _state(example).items():
                if name not in ("q1", "qd1"):
                    np.testing.assert_array_equal(values, state[name], err_msg=name)
            example.step()
            example.test_final()

    def test_fragment_reserve_message_names_the_cut_reserve(self):
        with _example(*_FREE_FALL, "--cut_reserve", "0.005", fragment_capacity=None) as example:
            self.assertEqual(example.cutter.capacity, 6 + 2)
            # The blade splits all six tetrahedra: twelve fragments.
            self.assertFalse(example.cut(_blade()))
            self.assertRegex(
                example.cut_error, "Material fragment reserve exhausted; restart with a larger --cut_reserve"
            )

    def test_cutter_with_fixed_storage_refuses_scratch_and_work_growth(self):
        """The cutter alone: neither the subtractor's scratch nor its work list grows when the storage is fixed."""
        points, elements = _cube()
        device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
        positions = wp.array(np.tile(points, (4, 1)), dtype=wp.vec3, device=device)
        full_scratch = {"scratch_vertex_capacity": 64, "scratch_face_capacity": 64}
        for prepare, drill, reserve in (
            # A cylinder leaves pieces of more than twelve faces in the scratch.
            ({}, (0.02, -0.02), "subtractor scratch faces"),
            # All six tetrahedra meet in the body diagonal through the centre: six work entries, four reserved.
            ({"work_capacity": 4, **full_scratch}, (0.0, 0.0), "subtractor work"),
        ):
            start, end = (*drill, 0.2), (*drill, -0.1)
            cutter = LinearTetCut(points, elements, fragment_capacity=512, fixed_storage=True, device=device)
            cutter.prepare_sphere(**prepare)
            scratch = cutter._sphere.scratch[0]
            sizes = scratch.vertex_capacity, scratch.face_capacity, len(cutter._sphere.work)
            before = _counted_captures()
            with self.assertRaisesRegex(ValueError, f"'refused_growth': '{reserve}'"):
                cutter.propose_cylinder(positions, start, end, 0.004)
            self.assertEqual(cutter.refused_growth, reserve)
            scratch = cutter._sphere.scratch[0]
            self.assertEqual((scratch.vertex_capacity, scratch.face_capacity, len(cutter._sphere.work)), sizes)
            self.assertEqual((cutter.storage_growths, _counted_captures()), ([], before))
            # The same edit on storage that may grow is accepted, and that growth is listed.
            grown = LinearTetCut(points, elements, fragment_capacity=512, device=device)
            grown.prepare_sphere(**prepare)
            self.assertGreater(grown.propose_cylinder(positions, start, end, 0.004)["changed_parents"], 0)
            self.assertEqual([growth["what"] for growth in grown.storage_growths], [reserve])
            self.assertIsNone(grown.refused_growth)

    def test_reset_with_a_reserve_builds_the_same_storage_again(self):
        with _example(*_FREE_FALL, "--cut_reserve", "1", fragment_capacity=None) as example:
            first = example.cutter
            self.assertTrue(example.cut(_blade()), example.cut_error)
            example.reset()
            self.assertIsNot(example.cutter, first)
            self.assertEqual(
                (
                    example.cutter.capacity,
                    example.cutter.geometry.vertex_capacity,
                    example.cutter.geometry.face_capacity,
                ),
                (first.capacity, 48, 32),
            )
            self.assertEqual((example.fragment_count, example.cut_revision, example.storage_growths), (6, 0, []))
            self.assertTrue(example.cut(_blade()), example.cut_error)

    def test_self_test_with_a_reserve_holds_the_sphere_and_fails_on_any_capture(self):
        with _self_test("--cut_reserve", "1") as example:
            run_script(example)
            line = final_line(example)
            self.assertEqual(line["scripted_edits"], ["line", "line of two patches", "cylinder", "sphere", "line"])
            self.assertEqual((line["presses"], line["startups"], line["cut_reserve"]), (3, 2, 1.0))
            self.assertEqual((line["storage_growths"], line["captures_forced_by_storage_growth"]), ([], 0))
            for name in cutting.Startup.numbers:
                self.assertEqual(line[name], 0, name)
        with _self_test("--cut_reserve", "1") as example:
            if not example.device.is_cuda:
                self.skipTest("A graph needs CUDA")
            example.cutter._graphs.clear()
            run_script(example, frames=45)
            with self.assertRaisesRegex(AssertionError, "'captures_after_startup': 1"):
                example.test_final()

    def test_reserve_changes_the_storage_and_not_the_result(self):
        """The same edits with storage that grows and with the fixed reserve leave the same bits."""
        center = (0.003, 0.004, 1.05)
        with (
            _example(*_FREE_FALL, "--surface_capacity", "8192") as grown,
            _example(*_FREE_FALL, "--cut_reserve", "1", fragment_capacity=None) as fixed,
        ):
            if grown.device.is_cuda:
                self.skipTest("Float32 assembly on CUDA uses atomics: two runs of one scene differ in rounding")
            for example in (grown, fixed):
                example.step()
                self.assertTrue(example.cut(_blade(ymax=0.013)), example.cut_error)
                example.step()
                self.assertTrue(example.cut_sphere(center, center, 0.018), example.cut_error)
                for _ in range(4):
                    example.step()
                self.assertTrue(example.cut_cylinder((0.02, -0.02, 1.2), (0.02, -0.02, 0.9), 0.004), example.cut_error)
                for _ in range(4):
                    example.step()
            self.assertIn("polygon vertices per fragment", [growth["what"] for growth in grown.storage_growths])
            self.assertEqual(fixed.storage_growths, [])
            self.assertEqual((fixed.fragment_count, fixed._surface_count), (grown.fragment_count, grown._surface_count))
            for name, values in _state(grown).items():
                other = _state(fixed)[name]
                # The volume fractions are stored for every fragment slot: compare the slots that both have.
                slots = min(len(values), len(other))
                np.testing.assert_array_equal(other[:slots], values[:slots], err_msg=name)

    def test_reserve_arguments_are_checked(self):
        for arguments, message in (
            (("--cut_reserve", "-1"), "cut reserve must be finite and nonnegative"),
            (("--cut_reserve_polygons", "48", "32"), "needs --cut_reserve"),
            (("--cut_reserve", "1", "--cut_reserve_polygons", "3", "12"), "at least four vertices and faces"),
        ):
            with self.assertRaisesRegex(ValueError, message), _example(*arguments):
                pass


if __name__ == "__main__":
    unittest.main()
