# SPDX-License-Identifier: Apache-2.0
"""The floor of the cut scene carries a body with the plain penalty of the surface under it, at every time step.

The six-tetrahedron cube of the cut tests, 0.1 m on a side and 1 kg, stands on the floor with
the example's defaults. A floor row is a corner of a surface triangle with a third of that
triangle's area, so the floor's reaction is k times the sum of area times depth over the corners
under the floor (on this cube the lower corners of the side walls count with the area of their
coarse triangles, beside the underside). At rest that sum times k is the weight, 9.81 N: the
corners lie 12 to 24 um deep, whatever the time step.

A contact force that is divided by its own implicit denominator, instead of standing in the
solve with that denominator, balances the same weight only where dt2 g M equals the sum of mass
times depth of the nodes under the floor: 0.4 to 0.9 mm deep at four substeps and 1.1 to 3.1 mm
at two (measured on the code as it was until 2026-11-20).

The floor's friction is that of an anchor: a floor row with push N and stiffness k sticks within
the reach mu N / k of its anchor, which at rest is mu times the row's depth (6 to 12 um on this
cube with mu = 0.5), and slides beyond it against mu N. A block pulled with less than mu W
therefore moves by less than the reaches of its rows and stays; one pulled with more slides. A
step holds a sliding row at mu N for the velocity it starts with, and for the velocity it adds
the row has the spring k min(1, reach / offset), whose offset after a step at speed v is
reach + dt v. A block that gains speed thus gains it as if its mass were larger by
mu N dt2 / (reach + dt v) per row: its acceleration is below (F - mu W) / m by less than the
fraction mu g dt / v.
"""

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from test_cut_pipeline import _cube

from newton_solver_suturebot._src.cutting import contact as contact_kernels
from newton_solver_suturebot._src.solvers.fem.solve_diagonal import SolveDiagonal
from newton_solver_suturebot.examples.incision import example_fem_surgical_incision as cutting

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
GRAVITY, STIFFNESS = 9.81, 2.0e7
# Frames until the standing cube is at rest (three seconds), at any of the time steps below.
FRAMES = 180
# The friction coefficient of the tests of this file: a number of these tests, not of a scene.
FRICTION = 0.5
# A node at rest moves less than a micrometre in a frame [m/s].
REST_SPEED = 6.0e-5


@contextmanager
def _standing_cube(*arguments):
    args = cutting.Example.create_parser().parse_args(["--fragment_capacity", "64", *arguments])
    with wp.ScopedDevice(DEVICE), patch.object(cutting, "load_body", side_effect=lambda *_: _cube()):
        viewer = ViewerNull()
        try:
            yield cutting.Example(viewer, args)
        finally:
            viewer.close()


def _weigh(example):
    """What the floor rows of the block weigh and how deep they lie, as it stands."""
    count = int(example.surface.count.numpy()[0])
    points = example.contact.update_surface(example.state_0.particle_q).numpy()[: 3 * count].astype(np.float64)
    # One row per corner of a surface triangle: a third of the triangle's area [m2], its depth [m].
    area = np.repeat(example.contact.areas.numpy()[:count].astype(np.float64), 3) / 3.0
    depth = np.clip(example.floor_height - points[:, 2], 0.0, None)
    mass = float(example.model.cut_physical_mass.numpy().sum())
    return {
        "weight": mass * GRAVITY,
        "area": area,
        "depth": depth,
        # The floor's reaction as the contact scattered it in the last substep [N].
        "reaction": float(example.contact.forces.numpy()[:, 2].sum()),
        # What the floor holds along itself, in that substep [N].
        "friction": example.contact.forces.numpy()[:, :2].astype(np.float64).sum(axis=0),
        "rows": int(example.contact.active_rows.numpy()[0]),
        "vertical_speed": float(np.abs(example.state_0.particle_qd.numpy()[:, 2]).max()),
    }


def _rest(*arguments):
    """The standing cube after ``FRAMES`` frames: what the floor rows weigh and how deep they lie.

    On a floor without friction, unless ``arguments`` name a coefficient: friction holds a block
    in the shape in which it came to rest, and that shape depends on how it got there.
    """
    with _standing_cube("--ground_friction", "0", *arguments) as example:
        for _ in range(FRAMES):
            example.step()
        return _weigh(example)


class TestFloorCarriesTheWeight(unittest.TestCase):
    def test_a_standing_block_is_carried_by_the_plain_penalty_at_every_time_step(self):
        """Fails if the floor's reaction is not k times area times depth of the rows under it, or the depth follows dt."""
        depths = {}
        for substeps in (4, 2, 8):
            with self.subTest(substeps=substeps):
                rest = _rest("--substeps", str(substeps))
                weight = rest["weight"]
                self.assertAlmostEqual(weight, GRAVITY, delta=1.0e-3)
                # At rest: no node moves up or down by a micrometre in a frame, and the reaction is the weight.
                self.assertLess(rest["vertical_speed"], 6.0e-5)
                self.assertAlmostEqual(rest["reaction"], weight, delta=0.002 * weight)
                self.assertEqual(rest["rows"], int(np.count_nonzero(rest["depth"])))
                # The reaction is the plain penalty of the rows under the floor: nothing divides it.
                penalty = STIFFNESS * float((rest["area"] * rest["depth"]).sum())
                self.assertAlmostEqual(penalty, rest["reaction"], delta=0.002 * weight)
                depths[substeps] = rest["depth"]
        for substeps in (2, 8):
            np.testing.assert_allclose(depths[substeps], depths[4], rtol=0.01, atol=1.0e-8)

    def test_a_floor_ten_times_as_stiff_carries_the_block_a_tenth_as_deep(self):
        """Fails if the depth at rest does not follow the stiffness the floor was given."""
        soft, stiff = _rest(), _rest("--ground_stiffness", str(10.0 * STIFFNESS))
        self.assertAlmostEqual(stiff["reaction"], stiff["weight"], delta=0.002 * stiff["weight"])
        np.testing.assert_allclose(10.0 * stiff["depth"], soft["depth"], rtol=0.02, atol=1.0e-8)


def _floor_rows(example):
    """The floor rows of the block, one per corner of a surface triangle: point, depth, anchor and reach [m]."""
    count = int(example.surface.count.numpy()[0])
    points = example.contact.update_surface(example.state_0.particle_q).numpy()[: 3 * count].astype(np.float64)
    return {
        "points": points,
        "depth": np.clip(example.floor_height - points[:, 2], 0.0, None),
        "anchor": example.contact.anchor.numpy()[: 3 * count].astype(np.float64),
        "reach": example.contact.reach.numpy()[: 3 * count].astype(np.float64),
    }


def _along_floor(vectors):
    """Lengths of the parts of ``vectors`` along the floor [m]."""
    return np.linalg.norm(vectors[:, :2], axis=1)


def _centre(example):
    """Velocity of the block's centre of mass [m/s] and its mass [kg]."""
    mass = example.model.cut_physical_mass.numpy().astype(np.float64)
    return mass @ example.state_0.particle_qd.numpy().astype(np.float64) / mass.sum(), float(mass.sum())


def _pull(example, acceleration):
    """A horizontal load on the block: gravity gets the component ``acceleration`` [m/s2] along x."""
    example.model.gravity.assign(np.array([[acceleration, 0.0, -GRAVITY]], dtype=np.float32))


def _blade(axis=0):
    """A vertical blade through the whole standing cube, in the plane 3 mm from its middle across ``axis``."""
    polygon = np.asarray(((0.003, -0.2, -0.02), (0.003, 0.2, -0.02), (0.003, 0.2, 0.2), (0.003, -0.2, 0.2)))
    if axis:
        polygon = polygon[:, [1, 0, 2]]
    return polygon[np.asarray(((0, 1, 2), (0, 2, 3)))]


class TestFloorFriction(unittest.TestCase):
    def test_the_floor_carries_a_block_that_it_holds_by_friction_at_every_time_step(self):
        """Fails if friction changes what the floor carries, pulls on a block at rest, or makes its depth follow dt."""
        depths = {}
        for substeps in (1, 2, 4, 8):
            with (
                self.subTest(substeps=substeps),
                _standing_cube("--substeps", str(substeps), "--ground_friction", str(FRICTION)) as example,
            ):
                # The weight comes on over one second. Put on at once, the block bounces into its rest
                # shape, the bounce is that of the time step, and friction keeps what it left: the
                # depths then differ by 5 % between one substep and four (measured), not by 0.3 %.
                for frame in range(60 + FRAMES):
                    scale = min(1.0, (frame + 1) / 60.0)
                    example.model.gravity.assign(np.array([[0.0, 0.0, -scale * GRAVITY]], dtype=np.float32))
                    example.step()
                rest = _weigh(example)
                weight = rest["weight"]
                self.assertLess(rest["vertical_speed"], REST_SPEED)
                self.assertAlmostEqual(rest["reaction"], weight, delta=0.002 * weight)
                penalty = STIFFNESS * float((rest["area"] * rest["depth"]).sum())
                self.assertAlmostEqual(penalty, rest["reaction"], delta=0.002 * weight)
                # Nothing pulls the block: what the rows hold along the floor adds up to nothing.
                self.assertLess(float(np.abs(rest["friction"]).max()), 0.002 * weight)
                depths[substeps] = rest["depth"]
        for substeps in (1, 2, 8):
            np.testing.assert_allclose(depths[substeps], depths[4], rtol=0.01, atol=1.0e-8)

    def test_a_block_pulled_with_less_than_the_friction_limit_stays(self):
        """Fails if a block under a horizontal load of half mu W slides, creeps or is not held by that load."""
        with _standing_cube("--ground_friction", str(FRICTION)) as example:
            for _ in range(FRAMES):
                example.step()
            before = _floor_rows(example)
            weight = _centre(example)[1] * GRAVITY
            load = 0.5 * FRICTION * weight
            # The load comes on over one second, slowly against the block's own sway, and stays for three.
            for frame in range(60 + FRAMES):
                _pull(example, min(1.0, (frame + 1) / 60.0) * load / weight * GRAVITY)
                example.step()
            after = _floor_rows(example)
            grip = after["reach"] > 0.0
            self.assertGreater(int(grip.sum()), 0)
            np.testing.assert_array_equal(grip, before["reach"] > 0.0)
            # No row has slid: every anchor is where it was before the load.
            np.testing.assert_array_equal(after["anchor"][grip], before["anchor"][grip])
            # A row that has not slid is within its reach of its anchor, before and after: it has
            # moved by less than the two reaches together, which are mu times its depths.
            self.assertTrue((_along_floor(after["points"] - after["anchor"])[grip] <= after["reach"][grip]).all())
            moved = _along_floor(after["points"] - before["points"])[grip]
            self.assertTrue((moved <= (before["reach"] + after["reach"])[grip]).all())
            self.assertLess(float(moved.max()), 2.0 * FRICTION * max(before["depth"].max(), after["depth"].max()))
            # The floor holds the load, and the block is at rest.
            force = example.contact.forces.numpy().astype(np.float64).sum(axis=0)
            self.assertAlmostEqual(-force[0], load, delta=0.002 * weight)
            self.assertAlmostEqual(force[1], 0.0, delta=0.002 * weight)
            self.assertAlmostEqual(force[2], weight, delta=0.002 * weight)
            self.assertLess(float(np.abs(example.state_0.particle_qd.numpy()).max()), REST_SPEED)

    def test_a_block_pulled_with_more_than_the_friction_limit_slides_against_mu_times_its_weight(self):
        """Fails if a sliding block is held back by more or less than mu N, beyond what the step's tangent adds."""
        with _standing_cube("--ground_friction", str(FRICTION), "--linear_damping", "0") as example:
            for _ in range(FRAMES):
                example.step()
            dt = example.sim_dt
            pull = 1.2 * FRICTION * GRAVITY
            _pull(example, pull)
            expected = pull - FRICTION * GRAVITY
            # The centre's velocity along the floor plus mu times its velocity up gains (F - mu W) / m per
            # second whatever the floor pushes with: m dvx/dt = F - mu N and m dvz/dt = N - W.
            marks = []
            for frame in range(100):
                example.step()
                if frame in (39, 99):
                    velocity = _centre(example)[0]
                    marks.append((velocity[0] + FRICTION * velocity[2], velocity[0], velocity[1]))
            (start, speed, _), (end, _, sideways) = marks
            measured = (end - start) / 1.0
            # It slides: after two thirds of a second it is faster than the tangent's bound needs.
            self.assertGreater(speed, 0.25)
            self.assertLess(measured, expected)
            self.assertGreater(measured, expected * (1.0 - FRICTION * GRAVITY * dt / speed))
            # Every row slides against mu times its push, along the block's way to within its speed sideways.
            rows = _floor_rows(example)
            self.assertTrue((_along_floor(rows["points"] - rows["anchor"])[rows["reach"] > 0.0] > 0.0).all())
            force = example.contact.forces.numpy().astype(np.float64).sum(axis=0)
            self.assertLess(abs(sideways), 0.01 * end)
            self.assertAlmostEqual(-force[0], FRICTION * force[2], delta=1.0e-4 * force[2])

    def test_a_block_at_rest_does_not_drift_in_ten_seconds(self):
        """Fails if a resting block's floor rows leave the reach of their anchors, or it still moves after ten seconds."""
        with _standing_cube("--ground_friction", str(FRICTION)) as example:
            for _ in range(FRAMES):
                example.step()
            before = _floor_rows(example)
            for _ in range(600 - FRAMES):
                example.step()
            after = _floor_rows(example)
            grip = after["reach"] > 0.0
            np.testing.assert_array_equal(grip, before["reach"] > 0.0)
            # The rows under the floor have grip: this is the floor with friction.
            self.assertGreater(int(grip.sum()), 0)
            np.testing.assert_array_equal(grip, after["depth"] > 0.0)
            # A row that stays within the reach of an anchor that stays has not slid.
            np.testing.assert_array_equal(after["anchor"][grip], before["anchor"][grip])
            moved = _along_floor(after["points"] - before["points"])[grip]
            self.assertTrue((moved <= (before["reach"] + after["reach"])[grip]).all())
            self.assertLess(float(np.abs(example.state_0.particle_qd.numpy()).max()), REST_SPEED)

    @contextmanager
    def sliding_cube(self, frames, *arguments):
        """The standing cube, pulled along x with 1.2 mu W for ``frames`` frames: it slides."""
        with _standing_cube("--ground_friction", str(FRICTION), "--linear_damping", "0", *arguments) as example:
            for _ in range(FRAMES):
                example.step()
            _pull(example, 1.2 * FRICTION * GRAVITY)
            for _ in range(frames):
                example.step()
            yield example

    def test_a_block_that_slid_and_is_let_go_stays_where_it_stopped(self):
        """Fails if the anchors of rows that slid do not follow them: the floor would then draw the block back."""
        with self.sliding_cube(40) as example:
            start = _floor_rows(example)
            self.assertGreater(_centre(example)[0][0], 0.25)
            # The load goes; friction stops the block (against mu g: within a tenth of a second) and it
            # comes to rest.
            _pull(example, 0.0)
            for _ in range(120):
                example.step()
            stopped = _floor_rows(example)
            # It slid on while it stopped: every row that has grip is further along than a row that sticks
            # can get, which stays within its reaches of where it was. And every anchor came along.
            grip = stopped["reach"] > 0.0
            self.assertGreater(int(grip.sum()), 0)
            travelled = (stopped["points"] - start["points"])[grip, 0]
            self.assertTrue((travelled > 100.0 * (start["reach"] + stopped["reach"])[grip]).all())
            offsets = _along_floor(stopped["points"] - stopped["anchor"])[grip]
            self.assertTrue((offsets <= 2.0 * FRICTION * stopped["depth"].max()).all())
            for _ in range(120):
                example.step()
            later = _floor_rows(example)
            # Two seconds on it is where it stopped: no row has moved by more than the reaches allow.
            moved = _along_floor(later["points"] - stopped["points"])[grip]
            self.assertTrue((moved <= (stopped["reach"] + later["reach"])[grip]).all())
            self.assertLess(float(np.abs(example.state_0.particle_qd.numpy()).max()), REST_SPEED)

    def test_a_row_that_leaves_the_floor_keeps_no_grip(self):
        """Fails if a row that is no longer in the step keeps the reach of the step in which it was."""
        with _standing_cube("--ground_friction", str(FRICTION)) as example:
            for _ in range(FRAMES):
                example.step()
            self.assertGreater(int((_floor_rows(example)["reach"] > 0.0).sum()), 0)
            # Gravity is turned upwards for a tenth of a second: the block leaves the floor.
            example.model.gravity.assign(np.array([[0.0, 0.0, GRAVITY]], dtype=np.float32))
            for _ in range(6):
                example.step()
            rows = _floor_rows(example)
            self.assertGreater(float(rows["points"][:, 2].min()), example.floor_height + 0.01)
            self.assertEqual(float(rows["reach"].max()), 0.0)
            # Back on the floor the rows that push have grip again.
            _pull(example, 0.0)
            for _ in range(FRAMES):
                example.step()
            landed = _floor_rows(example)
            np.testing.assert_array_equal(landed["reach"] > 0.0, landed["depth"] > 0.0)
            self.assertGreater(int((landed["reach"] > 0.0).sum()), 0)

    def test_a_frame_that_is_run_and_put_back_leaves_the_grip_as_it_was(self):
        """Fails if the snapshot of a frame (warm-up, graph capture) does not hold the anchors and reaches."""
        with self.sliding_cube(40) as example, self.sliding_cube(40) as twin:
            anchor, reach = example.contact.anchor.numpy().copy(), example.contact.reach.numpy().copy()
            restore = example._frame_snapshot()
            example.simulate()
            # The block slides: the frame has moved its anchors.
            self.assertFalse(np.array_equal(example.contact.anchor.numpy(), anchor))
            restore()
            np.testing.assert_array_equal(example.contact.anchor.numpy(), anchor)
            np.testing.assert_array_equal(example.contact.reach.numpy(), reach)
            # The frames after it are those of a block on which that frame was never run: to the bit on
            # the CPU, where two runs of one scene are the same run (the device sums in no fixed order).
            if DEVICE == "cpu":
                for _ in range(5):
                    example.step()
                    twin.step()
                np.testing.assert_array_equal(example.state_0.particle_q.numpy(), twin.state_0.particle_q.numpy())
                np.testing.assert_array_equal(example.contact.anchor.numpy(), twin.contact.anchor.numpy())

    @unittest.skipUnless(
        DEVICE == "cpu", "the rows before the cut are read from a twin, which is the same run only on the CPU"
    )
    def test_a_cut_of_a_sliding_block_keeps_the_grip_where_the_block_is(self):
        """Fails if the carry through a cut reads the surface where it was a step ago instead of where it is."""
        for arguments in ((), ("--surface_capacity", "12")):
            with (
                self.subTest(arguments=arguments),
                self.sliding_cube(20, *arguments) as example,
                self.sliding_cube(20, *arguments) as twin,
            ):
                speed = _centre(example)[0][0]
                # The rows as they are before the cut are read from a twin that came the same way: reading
                # them from the block itself would bring its surface to where the block is, which is what
                # the carry has to do by itself.
                np.testing.assert_array_equal(example.state_0.particle_q.numpy(), twin.state_0.particle_q.numpy())
                np.testing.assert_array_equal(example.contact.anchor.numpy(), twin.contact.anchor.numpy())
                before = _floor_rows(twin)
                # In a substep it moves by far more than the reaches of its rows: a carry that read the
                # surface where it was a substep ago would find every row away from its anchor.
                self.assertGreater(speed * example.sim_dt, 10.0 * before["reach"].max())
                capacity = example.contact.capacity
                self.assertTrue(example.cut(_blade(axis=1)), example.cut_error)
                after = _floor_rows(example)
                self.assertEqual(example.contact.capacity > capacity, bool(arguments))
                # The rows of the triangles that meet in a node have the same point. While the block
                # moves their reaches differ a little (measured: by up to 7 nm of 3.6 to 13.5 um), and
                # so do their offsets. The carry gives a row of the new surface the reach and the offset
                # of the old triangle that it finds at its point: those of one of the old rows there.
                spacing = float(np.spacing(np.float32(0.1)))
                old = np.flatnonzero(before["reach"] > 0.0)
                self.assertGreater(len(old), 0)
                for row in old:
                    here = np.flatnonzero(np.linalg.norm(before["points"] - before["points"][row], axis=1) < spacing)
                    there = np.flatnonzero(
                        np.linalg.norm(after["points"] - before["points"][row], axis=1) < 4.0 * spacing
                    )
                    self.assertGreater(len(there), 0)
                    offsets = (before["points"] - before["anchor"])[here, :2]
                    for new in there:
                        reach = after["reach"][new]
                        self.assertLess(float(np.abs(before["reach"][here] - reach).min()), 1.0e-6 * reach)
                        offset = (after["points"] - after["anchor"])[new, :2]
                        self.assertLess(float(np.linalg.norm(offsets - offset, axis=1).min()), 4.0 * spacing)

    def test_the_halves_of_a_block_cut_through_stay_on_a_floor_with_friction(self):
        """Fails if the halves of a block that is cut through slide apart on a floor that holds them."""
        apart = {}
        for friction in (0.0, FRICTION):
            with self.subTest(friction=friction), _standing_cube("--ground_friction", str(friction)) as example:
                for _ in range(FRAMES):
                    example.step()
                self.assertTrue(example.cut(_blade()), example.cut_error)
                before = _floor_rows(example)
                for _ in range(120):
                    example.step()
                after = _floor_rows(example)
                under = (before["depth"] > 0.0) & (after["depth"] > 0.0)
                left, right = under & (after["points"][:, 0] < 0.003), under & (after["points"][:, 0] > 0.003)
                moved = (after["points"] - before["points"])[:, 0]
                apart[friction] = float(moved[right].mean() - moved[left].mean())
                if friction:
                    # Held: no row under the floor has moved further than the reaches of an anchor allow.
                    limit = 2.0 * FRICTION * max(before["depth"].max(), after["depth"].max())
                    self.assertLess(float(_along_floor(after["points"] - before["points"])[under].max()), limit)
        # On the floor without friction the same halves are a hundred times further apart after two seconds.
        self.assertGreater(apart[0.0], 100.0 * abs(apart[FRICTION]))
        self.assertGreater(apart[0.0], 1.0e-3)

    @contextmanager
    def held_cube(self, *arguments):
        """The standing cube, held by the floor against half of mu W along x; with it, that load [N] and its weight [N]."""
        with _standing_cube("--ground_friction", str(FRICTION), *arguments) as example:
            for _ in range(FRAMES):
                example.step()
            weight = _centre(example)[1] * GRAVITY
            load = 0.5 * FRICTION * weight
            for frame in range(60 + 60):
                _pull(example, min(1.0, (frame + 1) / 60.0) * load / weight * GRAVITY)
                example.step()
            yield example, load, weight

    def assert_grip_carried(self, before, after):
        """Every row that had grip before is a row after, at the same point with the same reach and offset from its anchor."""
        old = np.flatnonzero(before["reach"] > 0.0)
        distance = np.linalg.norm(after["points"][:, None] - before["points"][None, old], axis=2)
        same = distance.argmin(axis=0)
        # Float32 at the size of the cube: a point, its anchor and their difference are each within a spacing.
        spacing = float(np.spacing(np.float32(0.1)))
        self.assertLess(float(distance.min(axis=0).max()), 4.0 * spacing)
        np.testing.assert_allclose(after["reach"][same], before["reach"][old], rtol=1.0e-6)
        np.testing.assert_allclose(
            (after["points"] - after["anchor"])[same, :2],
            (before["points"] - before["anchor"])[old, :2],
            atol=4.0 * spacing,
        )

    def check_a_cut_keeps_the_grip(self, *arguments):
        with self.held_cube(*arguments) as (example, load, weight):
            before = _floor_rows(example)
            capacity = example.contact.capacity
            # The blade stands along the load: each half keeps the length on which it stands against it.
            self.assertTrue(example.cut(_blade(axis=1)), example.cut_error)
            after = _floor_rows(example)
            self.assertEqual(example.contact.capacity > capacity, bool(arguments))
            self.assert_grip_carried(before, after)
            # The halves are still held against the load a second later.
            for _ in range(60):
                example.step()
            force = example.contact.forces.numpy().astype(np.float64).sum(axis=0)
            self.assertAlmostEqual(-force[0], load, delta=0.002 * weight)
            held = _floor_rows(example)
            under = (after["depth"] > 0.0) & (held["depth"] > 0.0)
            limit = 2.0 * FRICTION * max(after["depth"].max(), held["depth"].max())
            self.assertLess(float(_along_floor(held["points"] - after["points"])[under].max()), limit)

    def test_a_cut_keeps_the_grip_of_the_rows_it_renumbers(self):
        """Fails if a cut that fits the reserved surface lets go of what the floor held."""
        self.check_a_cut_keeps_the_grip()

    def test_a_cut_that_grows_the_surface_keeps_the_grip(self):
        """Fails if the contact that replaces a grown one does not take over its anchors."""
        self.check_a_cut_keeps_the_grip("--surface_capacity", "12")

    def test_a_refused_cut_leaves_the_grip_as_it_was(self):
        """Fails if a cut that is refused after the contact took its surface changes what the floor holds."""
        with self.held_cube() as (example, load, weight):
            before = _floor_rows(example)
            update, calls = example.contact.update_topology, []

            def fail_after_update(*arguments, **keywords):
                calls.append(1)
                update(*arguments, **keywords)
                if len(calls) == 1:
                    raise RuntimeError("injected publication failure after contact mutation")

            with patch.object(example.contact, "update_topology", side_effect=fail_after_update):
                self.assertFalse(example.cut(_blade(axis=1)))
            self.assertEqual(len(calls), 2)
            after = _floor_rows(example)
            np.testing.assert_array_equal(after["reach"] > 0.0, before["reach"] > 0.0)
            self.assert_grip_carried(before, after)
            example.step()
            force = example.contact.forces.numpy().astype(np.float64).sum(axis=0)
            self.assertAlmostEqual(-force[0], load, delta=0.002 * weight)

    def test_a_floor_without_friction_launches_what_it_launched_before_and_keeps_no_anchor(self):
        """Fails if coefficient zero runs a friction kernel or writes friction state: it must be the contact as it was."""
        launched = set()
        launch = wp.launch

        def record(kernel, *arguments, **keywords):
            if kernel.module is contact_kernels._floor_rows.module:
                launched.add(kernel.key)
            return launch(kernel, *arguments, **keywords)

        with _standing_cube("--ground_friction", "0", "--surface_capacity", "12") as example:
            with patch.object(wp, "launch", side_effect=record):
                for _ in range(30):
                    example.step()
                self.assertTrue(example.cut(_blade()), example.cut_error)
                for _ in range(30):
                    example.step()
            self.assertGreater(len(example.storage_growths), 0)
            self.assertGreater(int(example.contact.active_rows.numpy()[0]), 0)
            for array in example.contact.state:
                self.assertFalse(array.numpy().any())
        self.assertEqual(
            launched,
            {
                "_copy_surface_topology",
                "_surface_points",
                "_surface_bounds",
                "_floor_rows",
                "_piece_rows",
                "_contact_bounds",
                "_scatter_contacts",
            },
        )


class TestSolveDiagonal(unittest.TestCase):
    def solve_arrays(self, solver):
        matrix = solver.A_bsr if solver._streaming is None else solver._streaming.compact_matrix
        return matrix, solver._pc_inv_diag

    def check_blocks_stand_for_one_solve(self, *arguments):
        with _standing_cube(*arguments) as example:
            solver = example.solver
            matrix, inverse = self.solve_arrays(solver)
            count = example.model.particle_count
            offsets, columns = matrix.offsets.numpy(), matrix.columns.numpy()
            diagonal = np.array(
                [
                    offsets[node] + np.flatnonzero(columns[offsets[node] : offsets[node + 1]] == node)[0]
                    for node in range(count)
                ]
            )
            values_before, inverse_before = matrix.values.numpy().copy(), inverse.numpy().copy()
            candidate_before = solver.A_bsr.values.numpy().copy()
            blocks = np.zeros((count, 3, 3), dtype=np.float32)
            normal = np.array((0.6, 0.0, 0.8), dtype=np.float32)
            blocks[2] = 0.25 * np.outer(normal, normal)
            blocks[5] = np.diag((0.0, 0.0, 3.0e-3)).astype(np.float32)
            device_blocks = wp.array(blocks, dtype=wp.mat33, device=solver.model.device)
            diagonal_blocks = SolveDiagonal(count, solver.model.device)

            diagonal_blocks.remove(solver)
            np.testing.assert_array_equal(matrix.values.numpy(), values_before)

            diagonal_blocks.add(solver, device_blocks)
            values, inverted = matrix.values.numpy(), inverse.numpy()
            expected = values_before.copy()
            expected[diagonal] = values_before[diagonal] + blocks
            np.testing.assert_array_equal(values, expected)
            for node in (2, 5):
                product = inverted[node].astype(np.float64) @ expected[diagonal[node]].astype(np.float64)
                np.testing.assert_allclose(product, np.eye(3), atol=1.0e-5)
            untouched = np.setdiff1d(np.arange(count), (2, 5))
            np.testing.assert_array_equal(inverted[untouched], inverse_before[untouched])

            diagonal_blocks.remove(solver)
            np.testing.assert_array_equal(matrix.values.numpy(), values_before)
            np.testing.assert_array_equal(inverse.numpy(), inverse_before)
            np.testing.assert_array_equal(solver.A_bsr.values.numpy(), candidate_before)

    def test_blocks_stand_in_the_compact_matrix_of_prepared_tetrahedral_updates_and_leave_it_to_the_bit(self):
        """Fails if a block misses the matrix or the inverse block that a streaming solve reads, or is left in them."""
        self.check_blocks_stand_for_one_solve()

    def test_blocks_stand_in_the_assembled_matrix_and_leave_it_to_the_bit(self):
        """Fails if a block misses the assembled matrix or its inverse block, or is left in them."""
        self.check_blocks_stand_for_one_solve("--no-streaming_topology")

    def test_a_frame_leaves_no_contact_block_in_what_the_solve_reads(self):
        """Fails if the floor rows of a frame's last substep stay in the solver's matrix after the frame."""
        with _standing_cube() as example:
            for _ in range(30):
                example.step()
            self.assertGreater(int(example.contact.active_rows.numpy()[0]), 0)
            self.assertGreater(float(np.abs(example.contact.node_tangent.numpy()).max()), 0.0)
            solver = example.solver
            matrix, inverse = self.solve_arrays(solver)
            # What a tangent refresh at the state after the frame writes, with no contact in it.
            solver.refresh_tangent(example.state_0.particle_q, example.sim_dt)
            example.holds.add_matrix(solver, example.sim_dt)
            clean_values, clean_inverse = matrix.values.numpy().copy(), inverse.numpy().copy()
            forces = example.contact.evaluate(example.state_0.particle_q, example.state_0.particle_qd, example.sim_dt)
            example.contact.add_matrix(solver)
            # The blocks of the matrix in use (the reserve behind them is never written).
            used = int(matrix.offsets.numpy()[matrix.nrow])
            self.assertGreater(float(np.abs(matrix.values.numpy()[:used] - clean_values[:used]).max()), 0.0)
            self.assertGreater(float(np.abs(inverse.numpy() - clean_inverse).max()), 0.0)
            solver.step(example.state_0, example.state_1, example.control, None, example.sim_dt, external_forces=forces)
            example.contact.remove_matrix(solver)
            np.testing.assert_array_equal(matrix.values.numpy(), clean_values)
            np.testing.assert_array_equal(inverse.numpy(), clean_inverse)

    def test_a_solver_whose_solve_reads_other_copies_is_refused(self):
        """Fails if blocks are accepted for a solver whose preconditioner or matrix copies they would not reach."""
        with _standing_cube() as example:
            solver = example.solver
            count = example.model.particle_count
            blocks = wp.zeros(count, dtype=wp.mat33, device=solver.model.device)
            diagonal_blocks = SolveDiagonal(count, solver.model.device)
            with patch.object(solver, "preconditioner_type", "diag"), self.assertRaises(ValueError):
                diagonal_blocks.add(solver, blocks)
            with patch.object(solver, "linear_solver_chebyshev_degree", 2), self.assertRaises(ValueError):
                diagonal_blocks.add(solver, blocks)
            with patch.object(solver, "_fixed_particles", object()), self.assertRaises(ValueError):
                diagonal_blocks.remove(solver)
            with self.assertRaises(ValueError):
                diagonal_blocks.add(solver, wp.zeros(count + 1, dtype=wp.mat33, device=solver.model.device))


if __name__ == "__main__":
    unittest.main()
