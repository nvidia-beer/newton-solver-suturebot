# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""An entry or an exit is a passage of the needle tip, not of the tissue surface over it.

One tetrahedron, as in ``test_needle_insertion_physics``. The interaction is driven as the
coupler drives it: an implicit trial, the resolution check, the retained trial and its
acceptance. Each step moves the tip along the needle axis and the slanted face, through which
the tip leaves, along its outward normal by set amounts. No solver runs. From case K on the
mesh can be another one (thin, cut in two, folded into a valley or over a ridge, or with a
second tetrahedron as another wall), and a step can move the tip and the nodes by vectors.

"Gap" is the distance of the tip from the slanted face along that normal: positive outside the
tissue, negative inside. Lengths in the case descriptions are micrometres along the normal.

Cases X and Y hold the tip inside the tolerance of the point-in-tetrahedron test (barycentric
1e-6, about 6 nm at these faces): 2.5 nm outside a face and still counted as in the tissue.
Only there can a step that changes side have a swept crossing against the change (X), or one
in its direction right after a held step (Y). Anywhere else such a step needs a second
crossing, and a step with two crossings is rejected.

Mutations of the rule that no case here fails, and why they are accepted (2026-11-17):

- The entry branch or the exit branch taking the kept face before a crossing of its own
  direction. For the entry that changes only the face whose place in the opening's mask is
  asked. In case Y both faces are in the mask; where they are not, the commit is refused
  ("outside its authorized material opening"), and a case would pin that refusal, not the
  rule. For the exit no step has both: the tolerance only widens the tissue, so a tip that
  is held outside cannot cross a face outward without crossing one inward first.
- A tip exactly on the plane of a face counting as past it, in the entry half of the kept
  face's test or in the exit branch's sign test: an exact tie.
- The exit branch without its sign test, or with it also after an outward crossing. Both are
  older than the rule. A tip outside a closed surface is over its nearest face, so the sign
  test needs a tip that leaves the active tissue into tissue that is not active, which this
  fixture does not have. After an outward crossing the tip is over the crossed face unless
  it passes back under that plane beside the triangle within the step.

The lifetime of the kept face is read from the private field (case S), because no event shows
a stale face: a kept face is read only in a step that changes side without a crossing in the
direction of the change, and outside the tolerance such a step follows a held step, which has
just set the face.
"""

import os
import unittest
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleStepTooLarge, NeedleTissueInteraction, boundary_faces

UM = 1e-6
REST = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], dtype=float)
TETS = np.array([[0, 1, 2, 3]], dtype=np.int32)
NORMAL = np.ones(3) / np.sqrt(3.0)  # outward unit normal of the slanted face
AXIS = np.array([0.0, 0.0, 1.0])  # the needle axis; the tip advances along it
THIN = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.0001]], dtype=float)  # 60 um under the tip
SPLIT = np.vstack((REST, [[0.005, 0.005, 0.0]]))  # the tetrahedron cut in two by the plane x = y
SPLIT_TETS = np.array([[0, 1, 4, 3], [0, 4, 2, 3]], dtype=np.int32)
# The cut tetrahedron with node 4 moved 400 um inward: the slanted face is folded into a shallow valley.
VALLEY = SPLIT - 400 * UM * NORMAL * (np.arange(5) == 4)[:, None]
# The same with node 4 moved 400 um outward: the slanted face is folded over a shallow ridge.
RIDGE = SPLIT + 400 * UM * NORMAL * (np.arange(5) == 4)[:, None]
# A second tetrahedron whose face (4, 5, 6) lies 100 um over the slanted face: another wall.
WALL = np.vstack((REST, REST[1:] + 100 * UM * NORMAL, REST[1:].mean(axis=0) + (100 * UM + 0.005) * NORMAL))
WALL_TETS = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)

Row = namedtuple("Row", "events gap status in_material tip_inside")


def along_axis(travel):
    """Tip displacement along the needle axis [m] for ``travel`` [m] along the outward normal."""
    return travel / float(AXIS @ NORMAL)


class Drive:
    """The interaction on a small mesh and the two states of a step."""

    def __init__(self, case, device, rest=REST, tets=TETS, tip=(0.002, 0.002, -0.0001)):
        samples = np.zeros((11, 3))
        samples[:, 2] = np.linspace(0, -0.01, 11)
        self.case, self.device = case, device
        self.faces = boundary_faces(tets)[0]
        self.interaction = NeedleTissueInteraction(
            rest,
            tets,
            self.faces,
            (),
            samples,
            0.0005,
            samples[0],
            device=device,
            damping=0,
            shaft_damping_per_m=0,
            friction=0,
        )
        self.nodes = np.array(rest, dtype=float)
        self.tip = np.array(tip, dtype=float)
        self.start, self.candidate = self.state(), self.state()
        self.forces = wp.zeros(len(self.nodes), dtype=wp.vec3, device=device)
        self.rows = []

    def state(self):
        return SimpleNamespace(
            particle_q=wp.array(self.nodes, dtype=wp.vec3, device=self.device),
            particle_qd=wp.zeros(len(self.nodes), dtype=wp.vec3, device=self.device),
            body_q=wp.array([[*self.tip, 0, 0, 0, 1]], dtype=wp.transform, device=self.device),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
        )

    def gap(self):
        return float((self.tip - self.nodes[1]) @ NORMAL)

    def face(self, *nodes):
        """Index of the surface triangle with these three nodes."""
        return int(np.flatnonzero((np.sort(self.faces, axis=1) == sorted(nodes)).all(axis=1))[0])

    def outside(self, *nodes):
        """Distance [m] of the tip from the plane of that surface triangle: positive on its outer side."""
        a, b, c = self.nodes[self.faces[self.face(*nodes)]]
        normal = np.cross(b - a, c - a)
        return float((self.tip - a) @ normal / np.linalg.norm(normal))

    def to_height(self, nodes, target, step=1.2e-4):
        """Steps of at most ``step`` [m] along the axis until the tip is ``target`` [m] outside the face ``nodes``."""
        a, b, c = self.nodes[self.faces[self.face(*nodes)]]
        normal = np.cross(b - a, c - a)
        slope = float(AXIS @ normal / np.linalg.norm(normal))
        while self.interaction.status != "failed" and abs(self.outside(*nodes) - target) > 1e-12:
            self.step(float(np.clip((target - self.outside(*nodes)) / slope, -step, step)))

    def step(self, tip_axis=0.0, surface_normal=0.0, *, tip=None, tissue=None):
        """One accepted step: the tip ``tip_axis`` [m] along the axis, the face ``surface_normal`` [m] outward.

        ``tip`` [m] is a further displacement of the tip (a vector) and ``tissue`` [m] one of the
        nodes: a vector for all of them, or a row per node.
        """
        x, start, candidate = self.interaction, self.start, self.candidate
        self.tip = self.tip + tip_axis * AXIS
        self.nodes[1:] += surface_normal * NORMAL
        if tip is not None:
            self.tip = self.tip + np.asarray(tip, dtype=float)
        if tissue is not None:
            self.nodes += np.asarray(tissue, dtype=float)
        candidate.particle_q.assign(self.nodes)
        candidate.body_q.assign(np.array([[*self.tip, 0, 0, 0, 1]], np.float32))
        candidate.body_f.zero_()
        self.forces.zero_()
        x.add_forces(candidate, 0, np.zeros(3), self.forces, 0.0005, implicit=True, trial=True)
        x.check_step_resolution(start, candidate, 0)
        x.retain_trial_forces(start, 0)
        x.accept(candidate, 0)
        self.start, self.candidate = candidate, start
        events = [event["kind"] + (" (reused)" if event.get("reused") else "") for event in x.events]
        self.rows.append(Row(events, self.gap(), x.status, bool(x._in_material), bool(x.tip_inside)))
        return self.rows[-1]

    def refused(self, tip):
        """A candidate with the tip displaced by ``tip`` [m] that is not accepted.

        Returns the error that rejects it, or None when its trial passed and is discarded.
        """
        x, candidate = self.interaction, self.candidate
        candidate.particle_q.assign(self.nodes)
        candidate.body_q.assign(np.array([[*(self.tip + np.asarray(tip, dtype=float)), 0, 0, 0, 1]], np.float32))
        candidate.body_f.zero_()
        self.forces.zero_()
        try:
            x.add_forces(candidate, 0, np.zeros(3), self.forces, 0.0005, implicit=True, trial=True)
            x.check_step_resolution(self.start, candidate, 0)
        except NeedleStepTooLarge as error:
            return error
        finally:
            x.discard_trial_forces()
        return None

    def to_gap(self, target, step=1.2e-4):
        """Steps of at most ``step`` [m] along the axis until the gap is ``target`` [m]."""
        while self.interaction.status != "failed" and abs(self.gap() - target) > 1e-12:
            self.step(float(np.clip(along_axis(target - self.gap()), -step, step)))

    def move_to(self, target, step=1.2e-4):
        """Straight to ``target`` [m] in steps of at most ``step`` [m]."""
        target = np.asarray(target, dtype=float)
        while self.interaction.status != "failed" and np.linalg.norm(target - self.tip) > 1e-13:
            move = target - self.tip
            self.step(tip=move * min(1.0, step / float(np.linalg.norm(move))))

    def puncture(self):
        """The force evaluation at the start and its acceptance: the tip presses on the bottom face."""
        x = self.interaction
        x.add_forces(self.start, 0, np.zeros(3), self.forces, 0.0005)
        x.accept(self.start, 0)
        self.case.assertTrue(x.punctured, x.status)

    def insert_to_exit(self, gap):
        """Puncture, insert, and stop 11.5 um before the slanted face; then leave to ``gap`` [m] in one step."""
        x = self.interaction
        self.puncture()
        while self.tip[2] < 0.0003 - 1e-12:
            self.step(min(1.2e-4, 0.0003 - self.tip[2]))
        self.case.assertEqual(x.status, "inserted", x.reason)
        self.to_gap(-11.5 * UM)
        self.case.assertEqual((x.status, x.entry_count, x.exit_count), ("inserted", 1, 0), x.reason)
        self.rows.clear()
        if gap is not None:
            self.step(along_axis(gap + 11.5 * UM))

    def events(self):
        return [row.events for row in self.rows]


class NeedleSurfaceExcursionTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))

    def left(self, gap):
        """The interaction after the tip has left the tissue to ``gap`` [um]; None: still 11.5 um inside."""
        drive = Drive(self, self.device)
        drive.insert_to_exit(None if gap is None else gap * UM)
        return drive, drive.interaction

    def counts(self, x):
        return x.status, x.reason, x.entry_count, x.exit_count

    def test_a_surface_passing_over_a_leaving_tip_commits_no_entry_and_no_second_exit(self):
        # The measured sequence: exit at +0.85; the tip goes on outward by 12.4 while the surface
        # moves 19.4 outward over it, then by 13.3 while the surface falls back by 137.5.
        drive, x = self.left(0.85)
        history = (x._next_section, x.track_count, x.opening_count)
        row = drive.step(along_axis(12.4 * UM), 19.4 * UM)
        self.assertLess(row.gap, 0)
        # The tip is in the tetrahedron and it moved outward: it stays outside. The opening
        # beside it is reused as for any tip beside an opening; nothing else is committed.
        self.assertEqual((row.events, row.tip_inside, row.in_material), ([], True, False))
        self.assertTrue(x.punctured)
        self.assertIs(x._opening, x._openings[-1])  # the exit's opening, as the staged reuse left it
        self.assertEqual(self.counts(x), ("outside", "", 1, 1))
        self.assertEqual((x._next_section, x.track_count, x.opening_count), history)
        drive.step(along_axis(13.3 * UM), -137.5 * UM)
        for _ in range(3):
            drive.step(along_axis(13.5 * UM))
        self.assertEqual(drive.events(), [["exit"], [], [], [], [], []])
        self.assertEqual(self.counts(x), ("outside", "", 1, 1))
        self.assertEqual((x._next_section, x.track_count, x.opening_count), history)

    def test_b_tip_that_comes_back_commits_its_reentry(self):
        drive, x = self.left(0.85)
        drive.step(along_axis(-30 * UM))
        drive.step(along_axis(-30 * UM))
        drive.step(along_axis(80 * UM))
        self.assertEqual(drive.events(), [["exit"], ["entry (reused)"], [], ["exit"]])
        self.assertEqual(self.counts(x), ("outside", "", 2, 2))

    def test_c_tissue_closing_over_a_resting_tip_commits_the_entry(self):
        drive, x = self.left(5)
        drive.step(0.0, 20 * UM)
        drive.step()
        drive.step(along_axis(40 * UM))
        self.assertEqual(drive.events(), [["exit"], ["entry (reused)"], [], ["exit"]])
        self.assertEqual(self.counts(x), ("outside", "", 2, 2))

    def test_d_surface_dropping_below_an_entering_tip_commits_no_exit_and_no_second_entry(self):
        # The shape of the re-entry: the tip comes in by 2.5 per step and enters by 0.5; in the
        # next step the surface drops by 10 while the tip goes on inward, and stays there.
        drive, x = self.left(20)
        drive.to_gap(2.0 * UM, step=along_axis(2.5 * UM))
        self.assertEqual(self.counts(x), ("outside", "", 1, 1))
        drive.rows.clear()
        drive.step(along_axis(-2.5 * UM))
        history = (x._next_section, x.opening_count, x.emergence_position.copy())
        opening = x._opening
        self.assertIsNotNone(opening)
        row = drive.step(along_axis(-2.5 * UM), -10 * UM)
        self.assertGreater(row.gap, 0)
        # The tip is in no tetrahedron and it moved inward: it stays in the material.
        self.assertEqual((row.events, row.tip_inside, row.in_material), ([], False, True))
        self.assertTrue(x.punctured)
        self.assertIs(x._opening, opening)  # the opening of the entry stays
        self.assertEqual(self.counts(x), ("inserted", "", 2, 1))
        for _ in range(4):
            drive.step(along_axis(-2.5 * UM))
        self.assertEqual(drive.events(), [["entry (reused)"], [], [], [], [], []])
        self.assertEqual([row.tip_inside for row in drive.rows], [True, False, False, False, True, True])
        self.assertTrue(all(row.in_material for row in drive.rows))
        self.assertEqual(self.counts(x), ("inserted", "", 2, 1))
        self.assertEqual((x._next_section, x.opening_count), history[:2])
        np.testing.assert_array_equal(x.emergence_position, history[2])

    def test_e_plain_passage_out_commits_one_exit(self):
        drive, x = self.left(0.85)
        for _ in range(4):
            drive.step(along_axis(12.4 * UM))
        self.assertEqual(drive.events(), [["exit"], [], [], [], []])
        self.assertEqual(self.counts(x), ("outside", "", 1, 1))

    def drift(self, x):
        """5 nm, and that it is below the resolution of the stored tip: 16 float32 units in the last place."""
        tip = x.tip_position
        self.assertLess(5e-9, 16 * float(np.spacing(np.float32(np.abs(tip).max()))))
        return 5e-9

    def test_f_tissue_closing_over_a_tip_that_drifts_below_its_resolution_commits_the_entry_at_once(self):
        # A held needle is at rest to a few units in the last place of its pose, not to the bit.
        drive, x = self.left(5)
        drift = self.drift(x)
        drive.step(along_axis(drift), 20 * UM)
        drive.step(along_axis(drift))
        drive.step()
        drive.step(along_axis(40 * UM))
        self.assertEqual(drive.events(), [["exit"], ["entry (reused)"], [], [], ["exit"]])
        self.assertEqual(self.counts(x), ("outside", "", 2, 2))

    def test_g_tissue_rising_over_a_slowly_withdrawing_tip_commits_the_entry_within_one_resolved_step(self):
        # The tip withdraws by 2 per step and the tissue follows it by 20 per step: a change of
        # side made by the tissue. It is not held beyond one resolved step past the face.
        drive, x = self.left(5)
        for _ in range(60):
            drive.step(along_axis(2 * UM), 20 * UM)
        drive.step()
        drive.step()
        self.assertEqual(self.counts(x), ("inserted", "", 2, 1))
        self.assertEqual([events for events in drive.events() if events], [["exit"], ["entry (reused)"]])
        entry = drive.events().index(["entry (reused)"])
        beyond = next(i for i, row in enumerate(drive.rows) if -row.gap > x.step_resolution)
        self.assertEqual(entry, beyond)

    def test_h_tissue_retracting_from_a_tip_that_creeps_below_its_resolution_commits_the_exit_at_once(self):
        drive, x = self.left(None)
        drift = self.drift(x)
        for _ in range(60):
            drive.step(along_axis(-drift), -20 * UM)
        drive.step()
        drive.step()
        self.assertEqual(drive.events(), [["exit"]] + [[]] * 61)
        self.assertEqual(self.counts(x), ("outside", "", 1, 1))

    def test_i_reentry_under_a_drifting_tip_is_seen_before_the_tip_leaves(self):
        drive, x = self.left(5)
        drift = self.drift(x)
        drive.step(along_axis(drift), 20 * UM)
        for _ in range(4):
            drive.step(along_axis(drift))
        drive.step(along_axis(40 * UM))
        self.assertEqual(drive.events(), [["exit"], ["entry (reused)"], [], [], [], [], ["exit"]])
        self.assertEqual(self.counts(x), ("outside", "", 2, 2))

    def test_j_tissue_retracting_from_a_slowly_advancing_tip_commits_the_exit_within_one_resolved_step(self):
        # The mirror of case G: the tip advances by 0.1 per step, the tissue retracts by 20.
        drive, x = self.left(None)
        for _ in range(60):
            drive.step(along_axis(-0.1 * UM), -20 * UM)
        drive.step()
        drive.step()
        self.assertEqual(self.counts(x), ("outside", "", 1, 1))
        self.assertEqual([events for events in drive.events() if events], [["exit"]])
        beyond = next(i for i, row in enumerate(drive.rows) if row.gap > x.step_resolution)
        self.assertEqual(drive.events().index(["exit"]), beyond)

    def thin(self, y=0.002, above=30 * UM):
        """Thin tissue after the tip went in by the bottom face and out by the slanted one, ``above`` [m] over it.

        ``y`` [m] is the distance of the tip from the side face y = 0.
        """
        drive = Drive(self, self.device, rest=THIN, tip=(0.002, y, -0.0001))
        x = drive.interaction
        drive.puncture()
        drive.step(130 * UM)
        drive.step(60 * UM)
        self.assertEqual(drive.events(), [["entry"], ["exit"]])
        self.assertEqual(self.counts(x), ("outside", "", 1, 1))
        self.assertAlmostEqual(drive.outside(1, 2, 3), above, delta=0.01 * UM)
        drive.rows.clear()
        return drive, x

    def test_k_needle_that_reverses_into_thin_tissue_after_a_held_entry_commits_its_entry_and_its_exit(self):
        # 60 um of tissue under the tip. The tissue rises 45 over the leaving tip: held. Then the
        # needle really reverses and goes through: 35 down, to 15 above the bottom face, which is
        # then the face nearest the tip, and 30 down, out by the bottom face.
        drive, x = self.thin()
        row = drive.step(5 * UM, tissue=45 * UM * AXIS)
        self.assertEqual((row.events, row.tip_inside, row.in_material), ([], True, False))
        row = drive.step(-35 * UM)
        # The far face is the nearer one: 15 from the bottom face, 45 from the slanted one.
        self.assertLess(-drive.outside(0, 1, 2), -drive.outside(1, 2, 3))
        self.assertGreater(-drive.outside(0, 1, 2), 0)
        drive.step(-30 * UM)
        drive.step(-30 * UM)
        self.assertEqual(drive.events(), [[], ["entry (reused)"], ["exit"], []])
        self.assertEqual(self.counts(x), ("outside", "", 2, 2))

    def test_l_tip_that_goes_on_out_of_thin_tissue_after_a_held_entry_commits_nothing(self):
        # The tissue rises 75 over the leaving tip, which is then 20 above the bottom face and 40
        # under the slanted one, and stays; the tip goes on upward and leaves by the slanted face.
        drive, x = self.thin()
        drive.step(5 * UM, tissue=75 * UM * AXIS)
        self.assertLess(-drive.outside(0, 1, 2), -drive.outside(1, 2, 3))
        drive.step(5 * UM)
        drive.step(5 * UM)
        drive.step(40 * UM)
        self.assertEqual(drive.events(), [[], [], [], []])
        self.assertEqual([row.tip_inside for row in drive.rows], [True, True, True, False])
        self.assertFalse(any(row.in_material for row in drive.rows))
        self.assertEqual(self.counts(x), ("outside", "", 1, 1))

    def sliding(self, inward):
        """The face y = 0 passes 60 nm over a tip that slides 100 um along it and ``inward`` [m] toward the tissue."""
        drive = Drive(self, self.device, tip=(0.002, 0.0002, -0.0001))
        x = drive.interaction
        drive.puncture()
        drive.move_to((0.002, 0.0002, 0.0003))
        drive.move_to((0.002, 10e-9, 0.0003), step=5e-5)
        self.assertEqual((x.status, x.entry_count, x.exit_count), ("inserted", 1, 0), x.reason)
        drive.rows.clear()
        drive.step(tip=(100 * UM, inward, 0.0), tissue=(0.0, 60e-9, 0.0))
        self.assertGreater(drive.outside(0, 1, 3), 40e-9)
        drive.step()
        return drive, x

    def test_m_exit_against_the_tip_is_held_from_sixteen_units_of_its_largest_coordinate_on(self):
        # The tip moves across the face y = 0, where its coordinate is 10 nm and float32 keeps a
        # difference of a tenth of a unit of the largest coordinate (x = 0.0021 m). Half a unit
        # under the deadband the exit is the tissue's and is committed at once; half a unit over
        # it the exit is held for the one step in which the tip moves against it.
        unit = float(np.spacing(np.float32(0.0021)))
        for units, events in ((15.5, [["exit"], []]), (16.5, [[], ["exit"]])):
            with self.subTest(units=units):
                drive, x = self.sliding(units * unit)
                self.assertEqual(float(np.spacing(np.float32(np.abs(x.tip_position).max()))), unit)
                self.assertEqual(drive.events(), events)
                self.assertEqual(self.counts(x), ("outside", "", 1, 1))

    def test_n_held_entry_into_a_tetrahedron_the_tip_never_visited_records_nothing(self):
        # The tetrahedron cut in two. The tip goes in and out in the first half and moves 400 um
        # along the face to the second half, farther than the track spacing from every sample.
        # There the surface passes over it as in case A.
        drive = Drive(self, self.device, rest=SPLIT, tets=SPLIT_TETS, tip=(0.00205, 0.002, -0.0001))
        x = drive.interaction
        drive.insert_to_exit(0.85 * UM)
        drive.move_to(drive.tip + 400 * UM * np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0))
        self.assertEqual(self.counts(x), ("outside", "", 1, 1))
        self.assertLess(drive.tip[0], drive.tip[1])  # over the second half
        self.assertEqual(len(x._visited_tets), 1)
        samples = np.asarray(x._track_rest)
        self.assertGreater(np.linalg.norm(samples - drive.tip, axis=1).min(), x.track_spacing + 50 * UM)
        history = (x._next_section, x.track_count, x.opening_count, set(x._visited_tets))
        row = drive.step(along_axis(12.4 * UM), 19.4 * UM)
        self.assertEqual((row.events, row.tip_inside, row.in_material), ([], True, False))
        self.assertEqual((x._next_section, x.track_count, x.opening_count, set(x._visited_tets)), history)
        drive.step(along_axis(13.3 * UM), -137.5 * UM)
        self.assertEqual(self.counts(x), ("outside", "", 1, 1))
        self.assertEqual((x._next_section, x.track_count, x.opening_count, set(x._visited_tets)), history)

    def beside_a_wall(self):
        """Case D with another wall: the exit is held and the wall is 3 um over the tip, the face it left 7 under."""
        drive = Drive(self, self.device, rest=WALL, tets=WALL_TETS)
        x = drive.interaction
        drive.insert_to_exit(20 * UM)
        drive.to_gap(2.0 * UM, step=along_axis(2.5 * UM))
        self.assertEqual(self.counts(x), ("outside", "", 1, 1))
        drive.rows.clear()
        drive.step(along_axis(-2.5 * UM))
        move = np.zeros((8, 3))
        move[1:4], move[4:] = -10 * UM * NORMAL, -100 * UM * NORMAL
        row = drive.step(along_axis(-2.5 * UM), tissue=move)
        self.assertEqual((row.events, row.tip_inside, row.in_material), ([], False, True))
        self.assertAlmostEqual(drive.gap(), 7 * UM, delta=0.01 * UM)
        self.assertAlmostEqual(drive.outside(4, 5, 6), 3 * UM, delta=0.01 * UM)
        return drive, x

    def test_o_exit_held_beside_another_wall_stays_held_while_the_tip_goes_on_inward(self):
        # The wall follows the tip and stays the nearest face. The tip moves away from it, which
        # is inward at the face it left: no exit, and no opening in the wall.
        drive, x = self.beside_a_wall()
        history = (x._next_section, x.opening_count)
        move = np.zeros((8, 3))
        move[4:] = -2.5 * UM * NORMAL
        drive.step(along_axis(-2.5 * UM), tissue=move)
        self.assertLess(drive.outside(4, 5, 6), drive.gap())
        drive.step(along_axis(-2.5 * UM))
        for _ in range(3):
            drive.step(along_axis(-2.5 * UM))
        self.assertEqual(drive.events(), [["entry (reused)"], [], [], [], [], [], []])
        self.assertEqual([row.tip_inside for row in drive.rows], [True, False, False, False, True, True, True])
        self.assertEqual(self.counts(x), ("inserted", "", 2, 1))
        self.assertEqual((x._next_section, x.opening_count), history)

    def test_p_exit_committed_after_a_hold_beside_another_wall_is_at_the_face_the_tip_left(self):
        # The tip rests: the exit is the tissue's and is committed, at the slanted face and its
        # opening, not at the wall, which is the face nearest the tip.
        drive, x = self.beside_a_wall()
        openings = x.opening_count
        drive.step()
        self.assertEqual(drive.events(), [["entry (reused)"], [], ["exit"]])
        self.assertEqual(self.counts(x), ("outside", "", 2, 2))
        self.assertEqual(x.opening_count, openings)
        self.assertFalse(any(mask[drive.face(4, 5, 6)] for mask in x._opening_masks))

    def test_q_tip_that_goes_round_an_edge_after_a_held_exit_leaves_by_the_face_it_is_over(self):
        # Case D 2 um from the face y = 0. After the held exit the tip goes round the edge of
        # the two faces: it is then outside over the face y = 0 and behind the plane of the
        # slanted face, which no longer separates it from the tissue. The exit is committed
        # there, at the face y = 0.
        drive = Drive(self, self.device, tip=(0.002, 2 * UM, -0.0001))
        x = drive.interaction
        drive.insert_to_exit(20 * UM)
        drive.to_gap(2.0 * UM, step=along_axis(2.5 * UM))
        drive.rows.clear()
        drive.step(along_axis(-2.5 * UM))
        row = drive.step(along_axis(-2.5 * UM), -10 * UM)
        self.assertEqual((row.events, row.tip_inside, row.in_material), ([], False, True))
        openings = x.opening_count
        drive.step(tip=-40 * UM * np.array([0.0, 1.0, 0.0]) - 10 * UM * NORMAL)
        self.assertGreater(drive.outside(0, 1, 3), 30 * UM)
        self.assertLess(drive.gap(), -20 * UM)
        drive.step()
        self.assertEqual(drive.events(), [["entry (reused)"], [], ["exit"], []])
        self.assertEqual(self.counts(x), ("outside", "", 2, 2))
        self.assertEqual(x.opening_count, openings + 1)
        self.assertTrue(x._opening_masks[-1][drive.face(0, 1, 3)])

    def test_r_first_entry_that_is_held_leaves_the_needle_on_approach(self):
        # Before any entry: the tissue comes 110 um down over a tip that withdraws by 5, and goes
        # back. The needle was never in the tissue; away from the opening it is on approach.
        drive = Drive(self, self.device)
        x = drive.interaction
        drive.puncture()
        row = drive.step(-5 * UM, tissue=-110 * UM * AXIS)
        self.assertEqual((row.events, row.tip_inside, row.in_material), ([], True, False))
        drive.step(-5 * UM, tissue=110 * UM * AXIS)
        drive.move_to(drive.tip - 0.0015 * AXIS)
        self.assertTrue(all(events == [] for events in drive.events()))
        self.assertEqual(self.counts(x), ("approach", "", 0, 0))

    def test_s_face_of_a_held_change_of_side_is_kept_until_the_next_accepted_step(self):
        drive, x = self.thin()
        slanted = drive.face(1, 2, 3)
        self.assertIsNone(x._held_face)
        drive.step(5 * UM, tissue=45 * UM * AXIS)
        self.assertEqual(x._held_face, slanted)
        # A trial that is discarded and a candidate that is rejected are not accepted steps.
        self.assertIsNone(drive.refused(-35 * UM * AXIS))
        self.assertEqual(x._held_face, slanted)
        self.assertIsInstance(drive.refused(-400 * UM * AXIS), NeedleStepTooLarge)
        self.assertEqual(x._held_face, slanted)
        # Held again, at the kept face: the tip goes on outward.
        drive.step(2 * UM)
        self.assertEqual(x._held_face, slanted)
        # Committed, at the kept face: the needle reverses. The face is dropped.
        row = drive.step(-37 * UM)
        self.assertEqual(row.events, ["entry (reused)"])
        self.assertIsNone(x._held_face)
        # Dropped when the tip is back on its side.
        drive, x = self.thin()
        drive.step(5 * UM, tissue=45 * UM * AXIS)
        self.assertEqual(x._held_face, slanted)
        row = drive.step(5 * UM, tissue=-45 * UM * AXIS)
        self.assertEqual((row.events, row.tip_inside, row.in_material), ([], False, False))
        self.assertIsNone(x._held_face)

    def test_t_tip_that_is_no_longer_under_the_face_of_its_held_entry_enters_by_the_face_it_is_under(self):
        # The slanted face folded into a valley; the tip leaves 20 um beside its line. The surface
        # passes over the leaving tip (held, at the face on its side of the line). Then the tip
        # moves 220 um across the line, in the tissue: under the other face and over the plane
        # of the face its entry was held at. It moved away from that plane and toward the face
        # it is under, so the entry is the needle's own and is committed.
        across = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)
        start = (0.002 + 20 * UM * np.sqrt(2.0), 0.002, -0.0002)
        drive = Drive(self, self.device, rest=VALLEY, tets=SPLIT_TETS, tip=start)
        x = drive.interaction
        drive.puncture()
        drive.move_to((*drive.tip[:2], 0.0003))
        drive.to_height((1, 4, 3), -11.5 * UM)
        self.assertEqual((x.status, x.entry_count, x.exit_count), ("inserted", 1, 0), x.reason)
        drive.rows.clear()
        drive.to_height((1, 4, 3), 0.85 * UM)
        row = drive.step(along_axis(12.4 * UM), tissue=19.4 * UM * NORMAL)
        self.assertEqual((row.events, row.tip_inside, row.in_material), ([], True, False))
        self.assertLess(drive.outside(1, 4, 3), 0)
        row = drive.step(tip=220 * UM * across)
        self.assertGreater(drive.outside(1, 4, 3), 3 * UM)
        self.assertLess(drive.outside(4, 2, 3), -10 * UM)
        self.assertEqual((row.events, row.tip_inside, row.in_material), (["entry (reused)"], True, True))
        drive.to_height((4, 2, 3), 20 * UM)
        self.assertEqual([events for events in drive.events() if events], [["exit"], ["entry (reused)"], ["exit"]])
        self.assertEqual(self.counts(x), ("outside", "", 2, 2))

    def test_u_exit_held_while_the_tip_slides_onto_the_next_triangle_is_committed_where_the_tip_is(self):
        # The tetrahedron cut in two: the slanted face is two triangles of one plane. The face
        # retracts 20 um from a tip that goes on inward by 0.1 a step: the exit is held, at the
        # first triangle. The tip then slides 5 x 250 um along the surface, 8 um over it, onto
        # the second triangle, still going inward by 0.1 a step (held), and rests. It is then
        # more than two needle radii from the triangle its exit was held at, where the exit
        # branch refuses it, and 8 um over the second one: the exit is the tissue's and is
        # committed there, its opening under the tip. The surface is at rest after the first step.
        across = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)
        drive = Drive(self, self.device, rest=SPLIT, tets=SPLIT_TETS, tip=(0.00205, 0.002, -0.0001))
        x = drive.interaction
        drive.insert_to_exit(None)
        row = drive.step(along_axis(-0.1 * UM), tissue=-20 * UM * NORMAL)
        self.assertEqual((row.events, row.tip_inside, row.in_material), ([], False, True))
        self.assertLess(drive.tip[1], drive.tip[0])  # over the first triangle
        for _ in range(5):
            row = drive.step(along_axis(-0.1 * UM), tip=250 * UM * across)
            self.assertEqual((row.events, row.tip_inside, row.in_material), ([], False, True))
        beyond = float(drive.tip[1] - drive.tip[0]) / np.sqrt(2.0)  # of the line between the two triangles
        self.assertGreater(beyond, 2 * float(x.radii.max()))
        self.assertAlmostEqual(drive.gap(), 7.9 * UM, delta=0.01 * UM)
        drive.step()
        self.assertEqual(self.counts(x), ("outside", "", 1, 1))
        self.assertEqual(drive.events(), [[]] * 6 + [["exit"]])
        ids, bary = x._openings[-1]
        self.assertAlmostEqual(float(np.linalg.norm(bary @ drive.nodes[ids] - drive.tip)), drive.gap(), delta=0.01 * UM)

    def test_v_tip_that_leaves_thin_tissue_sideways_after_a_held_entry_commits_its_entry_and_its_exit(self):
        # 80 um of tissue under the tip, which is 40 um from the side face y = 0. The tissue rises
        # 25 over the leaving tip: held, 10 under the slanted face. Then the needle moves
        # sideways: 36 toward the side face, which is then the face nearest the tip (4 away), and
        # 10 more, out by the side face. Sideways is not outward at the slanted face: the entry
        # is the needle's own and is committed, and then its exit.
        drive, x = self.thin(y=40 * UM, above=10.4 * UM)
        row = drive.step(5 * UM, tissue=25 * UM * AXIS)
        self.assertEqual((row.events, row.tip_inside, row.in_material), ([], True, False))
        self.assertAlmostEqual(-drive.outside(1, 2, 3), 9.6 * UM, delta=0.01 * UM)
        drive.step(tip=(0.0, -36 * UM, 0.0))
        self.assertAlmostEqual(-drive.outside(0, 1, 3), 4 * UM, delta=0.01 * UM)
        self.assertLess(-drive.outside(0, 1, 3), -drive.outside(1, 2, 3))
        drive.step(tip=(0.0, -10 * UM, 0.0))
        drive.step(tip=(0.0, -10 * UM, 0.0))
        self.assertEqual(drive.events(), [[], ["entry (reused)"], ["exit"], []])
        self.assertEqual(self.counts(x), ("outside", "", 2, 2))

    def test_w_exit_committed_after_a_hold_reaches_two_needle_radii_from_its_face_and_no_farther(self):
        # The exit is held (the face retracts 20 um from a tip that goes on inward by 0.1). Then
        # the tissue retracts farther in one step, from a tip that still goes inward by 0.1: the
        # tip is more than one resolved step past the face and the exit is committed, if the
        # exit branch takes the face. 908 um from it, under two needle radii (1 mm), it does;
        # 1208 um from it, it refuses the tip, as it refuses an exit that was never held.
        refused = "Tip left active tissue without a resolved outward surface crossing"
        for retract, events, counts in (
            (900, [[], ["exit"]], ("outside", "", 1, 1)),
            (1200, [[], []], ("failed", refused, 1, 0)),
        ):
            with self.subTest(retract=retract):
                drive, x = self.left(None)
                row = drive.step(along_axis(-0.1 * UM), tissue=-20 * UM * NORMAL)
                self.assertEqual((row.events, row.tip_inside, row.in_material), ([], False, True))
                drive.step(along_axis(-0.1 * UM), tissue=-retract * UM * NORMAL)
                self.assertAlmostEqual(drive.gap(), (retract + 8.3) * UM, delta=0.01 * UM)
                self.assertEqual(drive.gap() > 2 * float(x.radii.max()), retract == 1200)
                self.assertEqual(drive.events(), events)
                self.assertEqual(self.counts(x), counts)

    def beside_the_ridge(self, lateral):
        """A held entry at the first triangle of the ridge, then ``lateral`` [m] across the ridge, in the tissue.

        The tip ends 2.5 nm outside the second triangle. That is inside the tolerance of the
        point-in-tetrahedron test (barycentric 1e-6, about 5 nm here), so the tip still counts as in
        the tissue although the step has an outward crossing of the second triangle. Returns the
        drive, the interaction, the row of that step and the outward unit normal of the second
        triangle.
        """
        across = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)
        start = (0.002 + 20 * UM * np.sqrt(2.0), 0.002, -0.0002)
        drive = Drive(self, self.device, rest=RIDGE, tets=SPLIT_TETS, tip=start)
        x = drive.interaction
        drive.puncture()
        drive.move_to((*drive.tip[:2], 0.0003))
        drive.to_height((1, 4, 3), -11.5 * UM)
        self.assertEqual((x.status, x.entry_count, x.exit_count), ("inserted", 1, 0), x.reason)
        drive.rows.clear()
        drive.to_height((1, 4, 3), 0.85 * UM)
        row = drive.step(along_axis(12.4 * UM), tissue=19.4 * UM * NORMAL)
        self.assertEqual((row.events, row.tip_inside, row.in_material), ([], True, False))
        # The second triangle as the interaction reads it: from float32 nodes.
        a, b, c = drive.nodes.astype(np.float32).astype(float)[drive.faces[drive.face(4, 2, 3)]]
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal)
        target = drive.tip + lateral * across
        row = drive.step(tip=target - ((target - a) @ normal - 2.5e-9) * normal - drive.tip)
        outside = float((drive.tip.astype(np.float32).astype(float) - a) @ normal)
        self.assertTrue(1e-9 < outside < 4e-9, outside)
        self.assertTrue(row.tip_inside)
        self.assertLess(drive.outside(1, 4, 3), -1 * UM)  # still under the plane of the first triangle
        return drive, x, row, normal

    def test_x_face_of_a_held_entry_is_not_replaced_by_a_crossing_against_the_entry(self):
        # 120 um across: the tip is deeper under the plane of the face its entry was held at than
        # before, so it moved inward there, and outward at the second triangle, which it crossed
        # outward. A crossing against the change is not the face of the event: the entry is read
        # at the kept face, is the needle's own and is committed. Then the tip leaves.
        drive, x, row, normal = self.beside_the_ridge(120 * UM)
        self.assertEqual(row.events, ["entry (reused)"])
        drive.step(tip=20 * UM * normal)
        self.assertEqual(drive.events(), [["exit"], [], ["entry (reused)"], ["exit"]])
        self.assertEqual(self.counts(x), ("outside", "", 2, 2))

    def test_y_crossing_in_the_direction_of_a_held_entry_is_the_face_of_the_event(self):
        # 50 um across: the tip is less deep under the plane of the kept face than before, so it
        # moved outward there: held again. Then it goes 20 um back toward the ridge and 0.5 um
        # in through the second triangle: outward at the kept face, inward at the triangle it
        # crosses. The crossing in the direction of the change is the face of the event, before
        # the kept face: the entry is the needle's own and is committed. Then the tip leaves.
        across = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)
        drive, x, row, normal = self.beside_the_ridge(50 * UM)
        self.assertEqual((row.events, row.in_material), ([], False))
        depth = -drive.outside(1, 4, 3)
        row = drive.step(tip=-20 * UM * across - 0.5 * UM * normal)
        self.assertLess(-drive.outside(1, 4, 3), depth - 0.5 * UM)  # outward at the kept face
        self.assertLess(drive.outside(4, 2, 3), -0.5 * UM)  # in the tissue under the second triangle
        self.assertEqual(row.events, ["entry (reused)"])
        drive.step(tip=20 * UM * normal)
        self.assertEqual(drive.events(), [["exit"], [], [], ["entry (reused)"], ["exit"]])
        self.assertEqual(self.counts(x), ("outside", "", 2, 2))

    def test_z_exit_held_at_a_triangle_whose_nearest_point_is_then_supported_is_committed_where_the_tip_is(self):
        # The tetrahedron cut in two. The exit is held at the first triangle, 35 um from the line
        # to the second. Node 4, on that line, is supported: set in the array accept() reads,
        # after the entry, because a face with a supported node cannot be punctured and the tip
        # entered by the bottom face. The tip slides 100 um across the line (held) and rests.
        # The first triangle is within reach and the tip is over it, but its point nearest the
        # tip, on the line, has more than 0.05 of the supported node, where the exit branch
        # refuses an exit. The tip's foot on the second triangle has less: the exit is the
        # tissue's and is committed there, its opening under the tip.
        across = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)
        drive = Drive(self, self.device, rest=SPLIT, tets=SPLIT_TETS, tip=(0.00029, 0.00024, -0.0001))
        x = drive.interaction
        drive.insert_to_exit(None)
        x.fixed[4] = 1
        row = drive.step(along_axis(-0.1 * UM), tissue=-20 * UM * NORMAL)
        self.assertEqual((row.events, row.tip_inside, row.in_material), ([], False, True))
        row = drive.step(along_axis(-0.1 * UM), tip=100 * UM * across)
        self.assertEqual((row.events, row.tip_inside, row.in_material), ([], False, True))
        # The line from node 4 to node 3 and the edge from node 4 to node 2 are perpendicular.
        line, edge = drive.nodes[3] - drive.nodes[4], drive.nodes[2] - drive.nodes[4]
        along = float((drive.tip - drive.nodes[4]) @ line / (line @ line))
        beyond = float((drive.tip - drive.nodes[4]) @ edge / (edge @ edge))
        self.assertGreater(beyond, 0)  # over the second triangle
        self.assertLess(beyond * float(np.linalg.norm(edge)), 2 * float(x.radii.max()))
        self.assertGreater(1 - along, 0.05)  # of node 4 at the first triangle's nearest point
        self.assertLess(1 - along - beyond, 0.05)  # of node 4 at the tip's foot on the second
        drive.step()
        self.assertEqual(self.counts(x), ("outside", "", 1, 1))
        self.assertEqual(drive.events(), [[], [], ["exit"]])
        ids, bary = x._openings[-1]
        self.assertAlmostEqual(float(np.linalg.norm(bary @ drive.nodes[ids] - drive.tip)), drive.gap(), delta=0.01 * UM)


if __name__ == "__main__":
    unittest.main()
