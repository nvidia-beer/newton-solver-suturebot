# SPDX-License-Identifier: Apache-2.0
"""A scene that owns the shared cut simulation: it steps, cuts, refuses and resets with no, one and two sets of grips.

The scene of these tests holds a :class:`CutTissue` and is not one, and this module imports
nothing from the examples; ``test_the_simulation_and_this_scene_load_no_example`` runs it in a
fresh interpreter to show that. The body is the six-tetrahedron cube of the cut tests, 0.1 m on
a side and 1 kg, built here; the numbers of the scene are this module's own. What the incision
example makes of the same simulation is in tests/test_cut_tissue_example.py.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from itertools import permutations
from pathlib import Path
from unittest.mock import patch

import numpy as np

import newton_solver_suturebot
from newton_solver_suturebot import CutTissue

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
GRAVITY = 9.81
# One set of grips of the test scene: a spring of 2 kN/m with 3 kg/s on a patch of 3 mm radius, its
# target moving at most 3 mm/s.
GRIP = {"stiffness": 2000.0, "damping": 3.0, "maximum_speed": 0.003, "radius": 0.003}


def _cube():
    """The cube of tests/test_cut_pipeline.py (that module imports the example and is not imported here)."""
    points = np.asarray([(x, y, z) for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (0.0, 0.1)])
    elements = []
    for first, second, _ in permutations(range(3)):
        nodes = [0, 1 << first, (1 << first) | (1 << second), 7]
        corners = points[nodes]
        if np.linalg.det((corners[1:] - corners[0]).T) < 0:
            nodes[1], nodes[2] = nodes[2], nodes[1]
        elements.append(nodes)
    return points, np.asarray(elements, dtype=np.int32)


class Scene:
    """A scene that owns a cut simulation and is not one: the cube on a floor with friction, or in free fall.

    Args:
        sets: Sets of grips of the simulation.
        floor: The cube stands on the floor; without it the cube starts 1 m up and nothing carries it.
        changes: Other arguments of :class:`CutTissue` than those of this scene.
    """

    def __init__(self, sets=1, *, floor=True, **changes):
        points, elements = _cube()
        numbers = {
            "frame_dt": 1.0 / 60.0,
            "substeps": 4,
            "solver_iterations": 80,
            "density": 1000.0,
            "young_modulus": 30000.0,
            "poisson_ratio": 0.35,
            "material_damping": 0.01,
            "linear_damping": 1.0,
            "min_stretch": 0.2,
            "max_stretch": 5.0,
            "gravity": GRAVITY,
            "floor_height": 0.0,
            "ground_stiffness": 2.0e7 if floor else 0.0,
            "piece_stiffness": 2.0e7 if floor else 0.0,
            "search_radius": 0.004,
            "ground_friction": 0.5,
            "grips": (GRIP,) * sets,
            "drop_height": 0.0 if floor else 1.0,
            "fragment_capacity": 64,
            "device": DEVICE,
        }
        numbers.update(changes)
        self.tissue = CutTissue(points, elements, **numbers)
        # The start-up is the owner's call.
        self.tissue.reset()


def _blade(axis=0, low=-0.02):
    """A vertical blade through the whole cube, in the plane 3 mm from its middle across ``axis``."""
    polygon = np.asarray(((0.003, -0.2, low), (0.003, 0.2, low), (0.003, 0.2, low + 1.5), (0.003, -0.2, low + 1.5)))
    if axis:
        polygon = polygon[:, [1, 0, 2]]
    return polygon[np.asarray(((0, 1, 2), (0, 2, 3)))]


@contextmanager
def _failing_publication(tissue):
    """The publication of the next edit fails after the contact has taken its surface (as in test_cut_pipeline)."""
    update, calls = tissue.contact.update_topology, []

    def fail_after_update(*arguments, **keywords):
        calls.append(1)
        update(*arguments, **keywords)
        if len(calls) == 1:
            raise RuntimeError("injected publication failure after contact mutation")

    with patch.object(tissue.contact, "update_topology", side_effect=fail_after_update):
        yield calls


def _grip_top(tissue, grips, side, *, y=0.01, lift=0.0):
    """A grip of the set ``grips`` on the top of the cube, left as a hold.

    It takes the top 25 mm to the side ``side`` (-1, +1) of the plane x = 3 mm, at ``y`` [m]; its
    target is asked ``lift`` [m] above where it took the top.
    """
    count = tissue._surface_count
    points = tissue.contact.update_surface(tissue.state_0.particle_q).numpy()[: 3 * count].reshape(count, 3, 3)
    points = points.astype(np.float64)
    target = np.array((0.003 + side * 0.025, y))
    for triangle in np.flatnonzero(points[:, :, 2].min(axis=1) > points[:, :, 2].max() - 0.01):
        a, b, c = points[triangle, :, :2]
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if area == 0.0:
            continue
        second = ((target[0] - a[0]) * (c[1] - a[1]) - (target[1] - a[1]) * (c[0] - a[0])) / area
        third = ((b[0] - a[0]) * (target[1] - a[1]) - (b[1] - a[1]) * (target[0] - a[0])) / area
        weights = np.array((1.0 - second - third, second, third))
        if weights.min() > 0.0:
            point = weights @ points[triangle]
            q = tissue.state_0.particle_q
            if not grips.begin(tissue.surface, tissue.cutter.geometry, int(triangle), weights, point, q):
                raise AssertionError("The grip took no material")
            if lift:
                grips.move(point + (0.0, 0.0, lift))
            grips.finish()
            return
    raise AssertionError("No triangle of the top holds the grip's point")


def _piece_heights(tissue):
    """Heights [m] of the centres of mass of the pieces, ordered along x."""
    q = tissue.state_0.particle_q.numpy().astype(np.float64)
    mass = tissue.model.cut_physical_mass.numpy().astype(np.float64)
    labels = tissue._node_components.numpy()
    centres = []
    for label in np.unique(labels[labels >= 0]):
        nodes = (labels == label) & (mass > 0.0)
        centres.append(mass[nodes] @ q[nodes] / mass[nodes].sum())
    return [float(centre[2]) for centre in sorted(centres, key=lambda centre: centre[0])]


def _bits(tissue):
    arrays = {
        "q": tissue.state_0.particle_q,
        "qd": tissue.state_0.particle_qd,
        "q1": tissue.state_1.particle_q,
        "dv": tissue.solver.velocity_increment,
        "forces": tissue.contact.forces,
        "tangent": tissue.contact.node_tangent,
        "anchor": tissue.contact.anchor,
        "reach": tissue.contact.reach,
        "mass": tissue.model.cut_physical_mass,
    }
    return {name: array.numpy().copy() for name, array in arrays.items()}


def _solve_matrix(tissue):
    """The blocks of the matrix that the solve reads, as far as they are in use."""
    solver = tissue.solver
    matrix = solver.A_bsr if solver._streaming is None else solver._streaming.compact_matrix
    used = int(matrix.offsets.numpy()[matrix.nrow])
    return matrix.values.numpy()[:used].astype(np.float64)


# What a fresh interpreter runs: this module's scene, a frame, a cut, a frame; then the modules it loaded.
_FRESH = """
import json, sys
sys.path[:0] = {paths!r}
import test_cut_tissue as scenes
import newton_solver_suturebot as package
tissue = scenes.Scene(1).tissue
tissue.step()
accepted = tissue.cut(scenes._blade())
tissue.step()
examples = sorted(name for name in sys.modules if name.startswith(package.__name__ + ".examples") or name == "newton.examples")
print(json.dumps({{"examples": examples, "package": package.__file__, "cuts": tissue.cut_revision, "accepted": bool(accepted)}}))
"""


class CutTissueSceneTests(unittest.TestCase):
    def test_scene_steps_cuts_refuses_and_resets_with_no_one_and_two_sets_of_grips(self):
        """Fails if a simulation that a scene owns does not start at the body's reference state, count its frames and
        time, keep its mass through a cut, leave positions, velocities, the solver's velocity increment and the counts
        as they were after a refused cut, or come back to the start state with empty sets of grips at a Reset."""
        points, _ = _cube()
        for sets in (0, 1, 2):
            with self.subTest(sets=sets):
                tissue = Scene(sets).tissue
                self.assertEqual(len(tissue.grip_sets), sets)
                if tissue.device.is_cuda:
                    self.assertIsNotNone(tissue.graph.graph_exec)
                else:
                    self.assertIsNone(tissue.graph)
                pool = np.tile(points, (tissue.node_copies, 1)).astype(np.float32)
                np.testing.assert_array_equal(tissue.state_0.particle_q.numpy(), pool)
                mass = float(tissue.model.cut_physical_mass.numpy().sum())
                self.assertAlmostEqual(mass, 1.0, delta=1.0e-5)
                for _ in range(3):
                    tissue.step()
                for index, grips in enumerate(tissue.grip_sets):
                    _grip_top(tissue, grips, (-1.0, 1.0)[index])
                self.assertTrue(tissue.cut(_blade()), tissue.cut_error)
                for _ in range(3):
                    tissue.step()
                self.assertEqual((tissue.cut_revision, tissue.fragment_count, tissue._frame), (1, 12, 5))
                self.assertAlmostEqual(tissue.sim_time, 6 * tissue.frame_dt)
                self.assertAlmostEqual(float(tissue.model.cut_physical_mass.numpy().sum()), mass, delta=3.0e-6 * mass)
                self.assertEqual([grips.count for grips in tissue.grip_sets], [1] * sets)
                # A cut that the cutter refuses before anything is published: its error, without a scene's options.
                self.assertFalse(tissue.cut(np.zeros((2, 3, 3))))
                self.assertTrue(tissue.cut_error.startswith("Cut rejected: "))
                # A cut whose publication fails after the contact has taken its surface: rolled back.
                # Copies: on the CPU numpy() is a view of the live array.
                q, qd = tissue.state_0.particle_q.numpy().copy(), tissue.state_0.particle_qd.numpy().copy()
                increment = tissue.solver.velocity_increment.numpy().copy()
                self.assertGreater(float(np.abs(increment).max()), 0.0)
                with _failing_publication(tissue) as calls:
                    self.assertFalse(tissue.cut(_blade(axis=1)))
                self.assertEqual(len(calls), 2)
                self.assertIn("injected publication failure", tissue.cut_error)
                self.assertEqual((tissue.cut_revision, tissue.fragment_count), (1, 12))
                np.testing.assert_array_equal(tissue.state_0.particle_q.numpy(), q)
                np.testing.assert_array_equal(tissue.state_0.particle_qd.numpy(), qd)
                np.testing.assert_array_equal(tissue.solver.velocity_increment.numpy(), increment)
                tissue.step()
                # The same cut is accepted afterwards.
                self.assertTrue(tissue.cut(_blade(axis=1)), tissue.cut_error)
                self.assertEqual((tissue.cut_revision, tissue.cut_error), (2, ""))
                tissue.step()
                self.assertTrue(np.isfinite(tissue.state_0.particle_q.numpy()).all())
                tissue.reset()
                self.assertEqual((tissue.cut_revision, tissue.fragment_count, tissue.sim_time), (0, 6, 0.0))
                # The frames are counted on over a Reset.
                self.assertEqual((tissue._frame, tissue.storage_growths), (7, []))
                np.testing.assert_array_equal(tissue.state_0.particle_q.numpy(), pool)
                self.assertEqual([grips.count for grips in tissue.grip_sets], [0] * sets)
                tissue.step()
                self.assertTrue(np.isfinite(tissue.state_0.particle_q.numpy()).all())

    def test_the_simulation_and_this_scene_load_no_example(self):
        """Fails if the package root, the simulation or this module imports a module of the examples."""
        source = Path(newton_solver_suturebot.__file__).resolve().parents[1]
        script = _FRESH.format(paths=[str(source), str(Path(__file__).resolve().parent)])
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", NEWTON_TEST_DEVICE="cpu")
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", script],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        # The interpreter ran the package under test, and the scene worked in it.
        self.assertEqual(Path(report["package"]).resolve(), Path(newton_solver_suturebot.__file__).resolve())
        self.assertEqual((report["accepted"], report["cuts"]), (True, 1))
        self.assertEqual(report["examples"], [])

    def test_the_matrix_of_every_set_of_grips_stands_in_what_the_solve_reads(self):
        """Fails if a frame leaves the spring of a set of grips, dt² k + dt c on its nodes, out of the solver's matrix."""
        for streaming in (True, False):
            plain = Scene(2, streaming_topology=streaming).tissue
            plain.simulate()
            clean = _solve_matrix(plain)
            for index in (0, 1):
                with self.subTest(streaming=streaming, set=index):
                    tissue = Scene(2, streaming_topology=streaming).tissue
                    grips = tissue.grip_sets[index]
                    _grip_top(tissue, grips, 1.0)
                    tissue.simulate()
                    added = _solve_matrix(tissue) - clean
                    # The blocks of a grip between the nodes of its anchors: summed over all of them, each
                    # axis holds the spring once for every anchor, by its share of the grip.
                    held = grips.enabled.numpy() != 0
                    weights = grips.weights.numpy()[held].astype(np.float64).sum(axis=1)
                    shares = float((grips.fractions.numpy()[held].astype(np.float64) * weights**2).sum())
                    dt = tissue.sim_dt
                    expected = (dt * dt * grips.stiffness + dt * grips.damping) * shares
                    self.assertGreater(shares, 0.5)
                    for axis in range(3):
                        self.assertAlmostEqual(float(added[:, axis, axis].sum()), expected, delta=0.01 * expected)

    def test_a_grip_of_each_of_two_sets_holds_its_piece(self):
        """Fails if a set of grips other than the first adds no force, or does not follow an accepted cut."""
        held, half = Scene(2, floor=False).tissue, Scene(2, floor=False).tissue
        for tissue in (held, half):
            # One grip of the first set on one side of the blade's plane; in ``held`` one of the second set on the other.
            _grip_top(tissue, tissue.grip_sets[0], -1.0)
        _grip_top(held, held.grip_sets[1], 1.0)
        (start,) = _piece_heights(held)
        for tissue in (held, half):
            self.assertTrue(tissue.cut(_blade(low=0.8)), tissue.cut_error)
            for _ in range(30):
                tissue.step()
        self.assertEqual([grips.count for grips in held.grip_sets], [1, 1])
        fallen = 0.5 * GRAVITY * held.sim_time**2
        heights = _piece_heights(held)
        self.assertEqual(len(heights), 2)
        for height in heights:
            self.assertGreater(height, start - 0.1 * fallen)
        left, right = _piece_heights(half)
        self.assertGreater(left, start - 0.1 * fallen)
        # Half a second without a grip, slowed only by the mass damping.
        self.assertLess(right, start - 0.5 * fallen)

    def test_a_refused_edit_puts_the_grips_of_every_set_back(self):
        """Fails if a roll-back leaves a grip of any set, the first or a later one, on the nodes of the refused edit."""
        tissue = Scene(2, floor=False).tissue
        # Both on the side of the blade whose material the cut gives copied nodes, so that the refused
        # edit stages the grips of both sets on other nodes than they hold.
        _grip_top(tissue, tissue.grip_sets[0], 1.0, y=0.01)
        _grip_top(tissue, tissue.grip_sets[1], 1.0, y=-0.02)
        tissue.step()
        before = [(grips.nodes.numpy().copy(), grips.enabled.numpy().copy()) for grips in tissue.grip_sets]
        staged = []
        for grips in tissue.grip_sets:
            stage = grips.stage_remap

            def stage_remap(*arguments, stage=stage, grips=grips):
                stage(*arguments)
                staged.append(grips.nodes.numpy().copy())

            grips.stage_remap = stage_remap
        with _failing_publication(tissue) as calls:
            self.assertFalse(tissue.cut(_blade(low=0.8)))
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(staged), 2)
        for index, (grips, nodes, (kept, enabled)) in enumerate(zip(tissue.grip_sets, staged, before, strict=True)):
            with self.subTest(set=index):
                # The refused edit did move this set's grip to copied nodes; the roll-back put it back.
                self.assertFalse(np.array_equal(nodes, kept))
                np.testing.assert_array_equal(grips.nodes.numpy(), kept)
                np.testing.assert_array_equal(grips.enabled.numpy(), enabled)
        self.assertTrue(tissue.cut(_blade(low=0.8)), tissue.cut_error)
        tissue.step()

    def test_a_frame_that_is_run_and_put_back_leaves_the_targets_of_every_set(self):
        """Fails if the snapshot of a frame (warm-up, graph capture) does not hold the driven targets of every set."""
        tissue = Scene(2).tissue
        # The targets are asked 50 mm up: each frame drives them on by the speed limit.
        _grip_top(tissue, tissue.grip_sets[0], -1.0, lift=0.05)
        _grip_top(tissue, tissue.grip_sets[1], 1.0, lift=0.05)
        tissue.step()
        driven = [grips.driven.numpy().copy() for grips in tissue.grip_sets]
        requested = [grips.requested.numpy().copy() for grips in tissue.grip_sets]
        q = tissue.state_0.particle_q.numpy().copy()
        restore = tissue._frame_snapshot()
        tissue.simulate()
        for grips, kept in zip(tissue.grip_sets, driven, strict=True):
            self.assertGreater(float(np.abs(grips.driven.numpy() - kept).max()), 0.0)
        restore()
        for index, grips in enumerate(tissue.grip_sets):
            with self.subTest(set=index):
                np.testing.assert_array_equal(grips.driven.numpy(), driven[index])
                np.testing.assert_array_equal(grips.requested.numpy(), requested[index])
        np.testing.assert_array_equal(tissue.state_0.particle_q.numpy(), q)

    def test_reset_makes_every_set_of_grips_anew(self):
        """Fails if a Reset keeps a set of grips of the scene it replaces, or a new set reads another time step than the solver's."""
        tissue = Scene(2).tissue
        _grip_top(tissue, tissue.grip_sets[0], -1.0)
        _grip_top(tissue, tissue.grip_sets[1], 1.0)
        tissue.step()
        old = tissue.grip_sets
        self.assertEqual([grips.count for grips in old], [1, 1])
        tissue.reset()
        self.assertEqual(len(tissue.grip_sets), 2)
        for index, (grips, replaced) in enumerate(zip(tissue.grip_sets, old, strict=True)):
            with self.subTest(set=index):
                self.assertIsNot(grips, replaced)
                self.assertEqual((grips.count, grips.active), (0, None))
                self.assertFalse(grips.enabled.numpy().any())
                # The new sets read the time step of the new solver.
                self.assertIs(grips.step_time, tissue.solver.step_time)
        tissue.step()
        self.assertTrue(np.isfinite(tissue.state_0.particle_q.numpy()).all())


@unittest.skipIf(DEVICE != "cpu", "Float32 assembly on CUDA uses atomics: two runs of one scene differ in rounding")
class CutTissueSceneBitsTests(unittest.TestCase):
    def assert_same_bits(self, one, other):
        for name, values in _bits(one).items():
            np.testing.assert_array_equal(_bits(other)[name], values, err_msg=name)

    def test_a_set_of_grips_that_holds_nothing_changes_no_bit(self):
        """Fails if a second set of grips without a grip changes a frame, an edit or a refused edit."""
        one, two = Scene(1).tissue, Scene(2).tissue
        for frame in range(30):
            for tissue in (one, two):
                if frame == 8:
                    _grip_top(tissue, tissue.grip_sets[0], 1.0, lift=0.01)
                if frame == 12:
                    self.assertTrue(tissue.cut(_blade()), tissue.cut_error)
                if frame == 20:
                    with _failing_publication(tissue):
                        self.assertFalse(tissue.cut(_blade(axis=1)))
                tissue.step()
            self.assert_same_bits(one, two)

    def test_a_grip_acts_the_same_in_whichever_set_it_is(self):
        """Fails if a grip of the second set gets another matrix, force, staging or roll-back than one of the first."""
        first, second = Scene(2).tissue, Scene(2).tissue
        for frame in range(30):
            for index, tissue in enumerate((first, second)):
                if frame == 5:
                    # On the side whose material a cut gives copied nodes: a refused cut stages the grip on them.
                    _grip_top(tissue, tissue.grip_sets[index], 1.0, lift=0.01)
                if frame == 8:
                    with _failing_publication(tissue) as calls:
                        self.assertFalse(tissue.cut(_blade()))
                    self.assertEqual(len(calls), 2)
                if frame == 14:
                    self.assertTrue(tissue.cut(_blade()), tissue.cut_error)
                tissue.step()
            self.assert_same_bits(first, second)
        self.assertEqual(
            ([grips.count for grips in first.grip_sets], [grips.count for grips in second.grip_sets]), ([1, 0], [0, 1])
        )


if __name__ == "__main__":
    unittest.main()
