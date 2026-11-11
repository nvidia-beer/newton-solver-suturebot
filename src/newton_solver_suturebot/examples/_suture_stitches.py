# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stitches across a wound: feasible entry and exit pairs and the thread's track through the tissue.

A stitch is the circular path of the bundled needle from an entry point to an
exit point on the pad surface, as :class:`InsertionPlan` builds and validates
it. Further stitches are the first one moved along the wound. The part of a
stitch's circle that lies in tissue is the track the thread follows after the
needle; it is sampled as material points that become the thread's eyelets.
"""

from dataclasses import dataclass

import numpy as np

from newton_solver_suturebot.examples._needle_insertion_geometry import InsertionPlan, TetLocator, pick_material_point


@dataclass(frozen=True)
class Stitch:
    """One needle pass and the material points of its track, ordered from entry to exit.

    Attributes:
        plan: The needle's circular path.
        angles: Angle of each track point on the plan's circle [rad].
        tetrahedra: Tetrahedron of each track point in the active tissue.
        weights: Barycentric weights of each track point, shape (count, 4).
        leave_angle: Tip angle at which the needle's tail has left the tissue
            by the plan's own margin [rad].
        passages: For each stretch of the track inside tissue, the indices of
            its first and last track point: where the thread goes into the
            tissue and where it comes out. A stitch across an open wound has
            two passages, one through each lip.
    """

    plan: InsertionPlan
    angles: np.ndarray
    tetrahedra: np.ndarray
    weights: np.ndarray
    leave_angle: float
    passages: tuple

    @property
    def points(self):
        """Reference positions of the track points [m]."""
        return self.plan.tip(self.angles)


def wound_axis(selection):
    """Unit horizontal direction along which the active region's upper surface is longest."""
    triangles = selection.vertices[selection.exterior_surface_triangles]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    upward = normals[:, 2] > 0.7 * np.linalg.norm(normals, axis=1)
    centers = triangles[upward].mean(axis=1)[:, :2]
    _, _, directions = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)
    return directions[0] / np.linalg.norm(directions[0])


def plan_stitch(geometry, region_map, reference, selection, fixed_vertices, entry_xy, exit_xy, spacing):
    """Plan one stitch between two map positions [m] and sample its track every ``spacing`` metres.

    Raises ValueError when the needle cannot make the pass: the checks of
    :class:`InsertionPlan`, and that the needle clears the pad while its tail
    leaves on the exit side.
    """
    entry = pick_material_point(region_map, reference, selection, *region_map.world_to_pixel(*entry_xy))
    exit_point = pick_material_point(region_map, reference, selection, *region_map.world_to_pixel(*exit_xy))
    plan = InsertionPlan.create(geometry, entry, exit_point, selection, reference, fixed_vertices)
    # The plan validates the sweep up to its finish angle; the tail leaves one needle span later.
    leave_angle = plan.finish_angle + geometry.span
    padding = geometry.thickness / geometry.radius
    sweep = np.arange(plan.finish_angle, leave_angle + padding, plan.sample_spacing / geometry.radius)
    owners, _ = TetLocator(reference.vertices, reference.indices).locate(plan.tip(sweep))
    if np.any(owners >= 0):
        raise ValueError("The needle would re-enter the pad while its tail leaves the tissue")
    locator = TetLocator(selection.vertices, selection.indices)
    angles = []
    for start, end in plan.tissue_intervals:
        count = max(2, int(np.ceil((end - start) * geometry.radius / spacing)) + 1)
        # The two ends lie on the surface; step inside by a hundredth of the passage so they have a tetrahedron.
        angles.append(np.linspace(start, end, count) + 0.01 * (end - start) * np.r_[1.0, np.zeros(count - 2), -1.0])
    passage = np.concatenate([np.full(len(part), index) for index, part in enumerate(angles)])
    angles = np.concatenate(angles)
    tetrahedra, weights = locator.locate(plan.tip(angles), tolerance=1.0e-6)
    inside = tetrahedra >= 0
    if inside.sum() < 2:
        raise ValueError("The stitch has no track through the tissue")
    members = [np.flatnonzero(passage[inside] == index) for index in range(len(plan.tissue_intervals))]
    passages = tuple((int(found[0]), int(found[-1])) for found in members if len(found))
    return Stitch(plan, angles[inside], tetrahedra[inside], weights[inside], float(leave_angle), passages)


def plan_stitches(
    geometry, region_map, reference, selection, fixed_vertices, entry_xy, exit_xy, count, pitch, spacing, search=8
):
    """Plan ``count`` stitches: the given one and copies moved along the wound by multiples of ``pitch``.

    Copies that the needle cannot make are skipped. Positions on either side
    of the first stitch are tried, nearest first, up to ``search`` pitches away.

    Raises ValueError when fewer than ``count`` feasible stitches are found.
    """
    axis = wound_axis(selection)
    entry_xy, exit_xy = np.asarray(entry_xy, dtype=float), np.asarray(exit_xy, dtype=float)
    found = {}
    for step in sorted(range(-search, search + 1), key=abs):
        if len(found) == count:
            break
        offset = step * pitch * axis
        try:
            found[step] = plan_stitch(
                geometry, region_map, reference, selection, fixed_vertices, entry_xy + offset, exit_xy + offset, spacing
            )
        except ValueError as error:
            if step == 0:
                raise ValueError(f"The first stitch is not feasible: {error}") from error
    if len(found) < count:
        raise ValueError(f"Only {len(found)} of {count} stitches are feasible along this wound")
    return [found[step] for step in sorted(found)]


def _ramp(time, duration, blend=0.25):
    """Fraction of a move done at ``time``: constant speed with smooth starts and stops of ``blend`` seconds."""
    time = float(np.clip(time, 0.0, duration))
    cruise = 1.0 / (duration - blend)
    if time < blend:
        return cruise * blend * ((time / blend) ** 3 - 0.5 * (time / blend) ** 4)
    if time > duration - blend:
        left = (duration - time) / blend
        return 1.0 - cruise * blend * (left**3 - 0.5 * left**4)
    return cruise * (time - 0.5 * blend)


def _rotate(rotation, vector):
    """Rotate a vector by a unit quaternion (x, y, z, w)."""
    axis = rotation[:3]
    return vector + 2.0 * rotation[3] * np.cross(axis, vector) + 2.0 * np.cross(axis, np.cross(axis, vector))


def _slerp(start, end, fraction):
    """Unit quaternion a fraction of the way from one to another along the shorter arc; ``end`` is on that side."""
    angle = np.arccos(np.clip(start @ end, -1.0, 1.0))
    if angle < 1.0e-6:
        blend = (1.0 - fraction) * start + fraction * end
    else:
        blend = (np.sin((1.0 - fraction) * angle) * start + np.sin(fraction * angle) * end) / np.sin(angle)
    return blend / np.linalg.norm(blend)


class _Turn:
    """Travel between two stitches that are copies of each other along the wound.

    The needle keeps turning about its circle's centre, the rest of the full
    turn, while the circle moves from one stitch to the next.
    """

    def __init__(self, stitch, following):
        self._stitch, self._following = stitch, following
        self._turn = following.plan.initial_angle + 2.0 * np.pi - stitch.leave_angle
        self.length = self._turn * stitch.plan.geometry.radius

    def pose(self, fraction):
        here = self._stitch.plan.pose(self._stitch.leave_angle + fraction * self._turn)
        plan = self._following.plan
        there = plan.pose(plan.initial_angle - (1.0 - fraction) * self._turn)
        # Both poses have the same turn about parallel axes; blend the short way.
        rotation = (1.0 - fraction) * here[3:] + fraction * (there[3:] if here[3:] @ there[3:] >= 0.0 else -there[3:])
        position = (1.0 - fraction) * here[:3] + fraction * there[:3]
        return np.concatenate((position, rotation / np.linalg.norm(rotation)))


class _Lift:
    """Travel between any two stitches, clear of the pad.

    The needle rises until its whole circle is above ``clear_height``, moves
    across while it turns into the next stitch's first pose, and comes down.
    A needle on a circle above that height cannot touch the pad however it
    is turned.
    """

    def __init__(self, stitch, following, clear_height):
        radius = stitch.plan.geometry.radius
        self._start = stitch.plan.pose(stitch.leave_angle)
        self._end = following.plan.pose(following.plan.initial_angle).copy()
        if self._start[3:] @ self._end[3:] < 0.0:
            self._end[3:] = -self._end[3:]
        centers = (np.asarray(stitch.plan.center, dtype=float), np.asarray(following.plan.center, dtype=float))
        self._rise = [max(0.0, clear_height + radius - center[2]) for center in centers]
        self._high = [center + np.array([0.0, 0.0, rise]) for center, rise in zip(centers, self._rise, strict=True)]
        # The circle's centre in the needle's own frame, at either end.
        conjugate = np.array([-1.0, -1.0, -1.0, 1.0])
        self._local = [
            _rotate(conjugate * pose[3:], center - pose[:3])
            for pose, center in zip((self._start, self._end), centers, strict=True)
        ]
        turn = 2.0 * np.arccos(np.clip(self._start[3:] @ self._end[3:], -1.0, 1.0))
        # No point of the needle moves farther than the centre's way plus the arc of the turn.
        self._across = float(np.linalg.norm(self._high[1] - self._high[0])) + radius * turn
        self.length = self._rise[0] + self._across + self._rise[1]

    def pose(self, fraction):
        distance = fraction * self.length
        if distance <= self._rise[0]:
            pose = self._start.copy()
            pose[2] += distance
        elif distance >= self.length - self._rise[1]:
            pose = self._end.copy()
            pose[2] += self.length - distance
        else:
            across = (distance - self._rise[0]) / self._across
            rotation = _slerp(self._start[3:], self._end[3:], across)
            center = (1.0 - across) * self._high[0] + across * self._high[1]
            local = (1.0 - across) * self._local[0] + across * self._local[1]
            pose = np.concatenate((center - _rotate(rotation, local), rotation))
        return pose


class NeedleSchedule:
    """Target pose of the needle over time for a row of stitches.

    Every stitch is one turn about its circle's centre: the tip goes from
    the plan's approach angle through the tissue until the tail has left it.
    Between two stitches the needle travels through the air. Stitches that
    are copies of each other along the wound are joined by the rest of the
    full turn. Other stitches may point anywhere, so the needle is lifted
    clear of the pad between them.

    Args:
        stitches: The stitches in the order they are made.
        pass_speed: Speed of the tip along its circle during a pass [m/s].
        air_speed: Speed of the needle between stitches [m/s].
        clear_height: Height above which nothing of the pad lies [m]; the
            needle travels lifted above it. ``None`` for a row of copies,
            which is joined by turning on.

    Attributes:
        duration: Time of the whole row [s].
    """

    def __init__(self, stitches, pass_speed, air_speed, clear_height=None):
        self._moves = []
        radius = stitches[0].plan.geometry.radius
        for index, stitch in enumerate(stitches):
            sweep = stitch.leave_angle - stitch.plan.initial_angle
            # The smooth start and stop of a move cost a quarter of a second each way.
            self._moves.append((sweep * radius / pass_speed + 0.5, index, None))
            if index + 1 < len(stitches):
                following = stitches[index + 1]
                travel = _Turn(stitch, following) if clear_height is None else _Lift(stitch, following, clear_height)
                self._moves.append((travel.length / air_speed + 0.5, index, travel))
        self._stitches = stitches
        self._starts = np.concatenate(([0.0], np.cumsum([move[0] for move in self._moves])))
        self.duration = float(self._starts[-1])

    def at(self, time):
        """Target at ``time``: the body pose, the index of the stitch in progress and its tip angle.

        The tip angle is on the circle of that stitch; between two stitches it
        stays at the angle where the needle left the first one.
        """
        move = min(int(np.searchsorted(self._starts, time, side="right")) - 1, len(self._moves) - 1)
        duration, index, travel = self._moves[move]
        stitch = self._stitches[index]
        fraction = _ramp(time - self._starts[move], duration)
        if travel is None:
            angle = stitch.plan.initial_angle + fraction * (stitch.leave_angle - stitch.plan.initial_angle)
            return stitch.plan.pose(angle), index, angle
        return travel.pose(fraction), index, stitch.leave_angle
