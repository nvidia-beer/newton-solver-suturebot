# SPDX-License-Identifier: Apache-2.0
"""The scripted path of the tool/tissue example: eight named phases of the driven tool's palm and jaws.

The path is a palm pose and a jaw opening for the end of every frame. Its orientation never
changes; its positions follow from the scene, by the rules of :func:`plan_path`. "Depth" is
always the depth of the deepest point of the jaws as they are commanded, with the radius of
the drawn jaw, below the tissue surface at rest (negative: the gap above it); a position is
found by moving the palm until the jaws have the commanded depth, so a tool that is exactly
on its path, over tissue at rest, measures exactly the commanded depth. In the first seven
phases the jaws are closed. In the eighth, ``pinch``, they open over a ridge of the tissue, go
down beside it, close on it, lift, lower and open again (:func:`_plan_pinch`).

Lengths in metres, times in seconds, speeds in metres per second, angles in radians.
"""

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..._src.solvers.coupled.interface.swept_tool_tissue_contact import STATION_RADII

# Names of the eight phases, in their order.
PHASES = ("approach", "press", "press_deep", "slide", "lift", "border", "fast", "pinch")
# The phase in which the jaws take a ridge of the tissue between them.
PINCH = "pinch"
# Phases with a frame at whose start the tool is clear of the tissue: a run can be placed there.
RESTART_PHASES = ("approach", "border", "fast", PINCH)
# The fast approach commands the jaw tip to travel this many of its contact radii in one physics
# interval: its own diameter, so that two consecutive placements of the tip sphere do not overlap
# and a contact that tests placements sees nothing of the way between them.
FAST_RADII_PER_INTERVAL = 2.0
# Label of the segment of the fast approach: its target ramps through the tissue surface until
# the tool itself has crossed it (ToolPath.next_frame).
FAST_APPROACH = "fast approach"
# The tool waits one jaw diameter above the surface: nearer, the lag of the slowest actuator behind
# a transit at TRANSIT speed would be larger than the gap (see plan_path).
CLEARANCE_JAW_RADII = 2.0


@dataclass(frozen=True)
class Segment:
    """Frames of one motion or hold of a phase.

    Attributes:
        phase: Name of the phase (:data:`PHASES`).
        label: What the segment does.
        first_frame: Index of its first frame.
        frames: Number of frames.
        depth: Commanded depth at its end [m], negative above the surface; for the fast approach at the
            end of its first frame, the depth that the hold after it returns to.
        speed: Commanded speed of the palm [m/s]; zero for a hold.
        hold: Whether the target is held: the last frame of a hold is where the tool is compared
            with its command.
        in_tissue: Whether the commanded jaws are inside the tissue at rest somewhere in it.
    """

    phase: str
    label: str
    first_frame: int
    frames: int
    depth: float
    speed: float
    hold: bool
    in_tissue: bool

    @property
    def last_frame(self):
        return self.first_frame + self.frames - 1


@dataclass(frozen=True)
class ToolPath:
    """Palm targets of the driven tool for every frame, with the numbers they were derived from.

    Attributes:
        start_pose: Palm pose before the first frame (xyz, xyzw).
        poses: Palm pose at the end of every frame, shape (frames, 7).
        depth: Commanded depth of the jaws at the end of every frame [m], shape (frames,).
        tip: Commanded position of the jaw tip at the end of every frame [m], shape (frames, 3).
        opening: Commanded half-opening of each jaw of the driven tool at the end of every frame [rad],
            shape (frames,); zero (closed jaws) in the seven phases before the pinch.
        segments: The segments, in order.
        restarts: For each phase of :data:`RESTART_PHASES`, the first of its frames at whose start the
            tool is clear of the tissue: the first frame of the phase, and for the pinch, which begins
            with the lift-off from the hold of the fast approach, the first frame after that lift-off.
        derivation: Every derived number with the rule it follows from.
    """

    start_pose: np.ndarray
    poses: np.ndarray
    depth: np.ndarray
    tip: np.ndarray
    opening: np.ndarray
    segments: tuple
    restarts: dict
    derivation: dict

    @property
    def frames(self):
        return len(self.poses)

    def opening_before(self, frame):
        """Commanded half-opening of the jaws at the start of ``frame`` [rad]: that at the end of the frame before it."""
        return float(self.opening[frame - 1]) if frame > 0 else 0.0

    def segment_of(self, frame):
        for segment in self.segments:
            if segment.first_frame <= frame <= segment.last_frame:
                return segment
        raise IndexError(f"Frame {frame} is not on the path")

    def phase_frames(self, phase):
        frames = [segment for segment in self.segments if segment.phase == phase]
        return frames[0].first_frame, frames[-1].last_frame

    def restart_after(self, frame):
        """First frame of the next phase that starts clear of the tissue after ``frame``, or None."""
        later = [first for first in self.restarts.values() if first > frame]
        return min(later) if later else None

    @property
    def fast_approach(self):
        """The segment of the fast approach."""
        return next(segment for segment in self.segments if segment.label == FAST_APPROACH)

    def reads_tool(self, frame):
        """Whether the frame that follows ``frame`` depends on where the tool is after it (:meth:`next_frame`)."""
        ramp = self.fast_approach
        return ramp.first_frame <= frame < ramp.last_frame

    def next_frame(self, frame, tip_gap):
        """The frame that follows ``frame``, given the gap of the jaw tip to the surface at rest after it [m].

        Every frame is followed by the next one, except in the fast approach. A position actuator
        slows down as it nears a target that stands still, so a target that stops inside the
        tissue is reached more slowly than it was commanded. The target of the fast approach
        therefore goes on along the insertion axis, frame after frame, for as long as the tip
        sphere of the tool itself is clear of the surface at rest (``tip_gap`` positive, or None
        beyond the reach of the search). The first frame after which it is not is followed by
        the hold, whose first frame brings the target back to the commanded depth; the frames of
        the approach that are left are not run.
        """
        if self.reads_tool(frame) and tip_gap is not None and tip_gap <= 0.0:
            return self.fast_approach.last_frame + 1
        return frame + 1


def _rotation(x, y):
    """Quaternion (xyzw) of the frame with the given x and y axes."""
    x, y = x / np.linalg.norm(x), y / np.linalg.norm(y)
    matrix = np.column_stack((x, y, np.cross(x, y)))
    quaternion = np.asarray(wp.quat_from_matrix(wp.mat33(matrix)), dtype=float)
    return quaternion / np.linalg.norm(quaternion), matrix


class _Planner:
    """Positions of the palm at a commanded jaw depth, found on the surface at rest."""

    def __init__(self, scene, rotation, matrix):
        self.scene, self.rotation, self.matrix = scene, rotation, matrix
        self.radius = scene.jaw_contact_radius
        pad = np.asarray(scene.reference.vertices, dtype=float)
        self.top = float(pad[:, 2].max())
        self.thickness = float(np.ptp(pad[:, 2]))
        # The lowest point of the jaws in the palm frame: the spheres whose lowest point is lowest.
        centres = scene.jaw_points @ matrix.T
        low = centres[:, 2] - scene.jaw_radii
        lowest = low <= low.min() + 1e-9
        # Offset from the palm to the point of the jaws that leads into the tissue (middle of the lowest spheres).
        self.lead = np.r_[centres[lowest, :2].mean(axis=0), low.min()]

    def depth(self, positions):
        return self.scene.jaw_depth_at_rest(positions, self.rotation)

    def along(self, origins, direction, target):
        """Move each origin along ``direction`` (into the tissue) until the jaws have the depth ``target``.

        The origins are clear of the tissue. A march in steps of half a jaw radius finds the first
        step at which the depth passes the target; bisection between that step and the one before
        it finds the position to 1e-9 m.
        """
        origins = np.asarray(origins, dtype=float).reshape(-1, 3)
        direction = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        if np.any(self.depth(origins) >= target):
            raise ValueError("A path position that must be clear of the tissue is not")
        step = 0.5 * self.radius
        count = int(math.ceil((self.thickness + 4.0 * self.radius) / (step * abs(direction[2])))) + 1
        distances = step * np.arange(count)
        points = origins[:, None, :] + distances[None, :, None] * direction
        depths = self.depth(points.reshape(-1, 3)).reshape(len(origins), count)
        reached = depths >= target
        if not reached.any(axis=1).all():
            raise ValueError("The jaws do not reach the commanded depth along a path direction")
        first = reached.argmax(axis=1)
        low, high = distances[first - 1], distances[first]
        for _ in range(40):
            middle = 0.5 * (low + high)
            inside = self.depth(origins + middle[:, None] * direction) >= target
            low, high = np.where(inside, low, middle), np.where(inside, middle, high)
            if np.max(high - low) < 1e-9:
                break
        return origins + (0.5 * (low + high))[:, None] * direction

    def over(self, sites, target):
        """Palm positions whose jaws lead over the sites (xy) and have the depth ``target``, found from above."""
        sites = np.asarray(sites, dtype=float).reshape(-1, 2)
        origins = np.column_stack((sites - self.lead[:2], np.full(len(sites), self.top + 4.0 * self.radius)))
        origins[:, 2] -= self.lead[2]
        return self.along(origins, (0.0, 0.0, -1.0), target)


def _upper_surface_nodes(scene):
    """Nodes of the block's exterior triangles that face upward (normal within 60 degrees of +z)."""
    rest = np.asarray(scene.selection.vertices, dtype=float)
    triangles = np.asarray(scene.selection.exterior_surface_triangles)
    normals = np.cross(rest[triangles[:, 1]] - rest[triangles[:, 0]], rest[triangles[:, 2]] - rest[triangles[:, 0]])
    upward = normals[:, 2] > 0.5 * np.linalg.norm(normals, axis=1)
    return np.unique(triangles[upward])


def _move(start, end, speed, frame_dt):
    """Positions at the end of every frame of a straight move at no more than ``speed``."""
    length = float(np.linalg.norm(end - start))
    frames = max(1, math.ceil(length / (speed * frame_dt) - 1e-9))
    fractions = np.arange(1, frames + 1) / frames
    return start + fractions[:, None] * (end - start), length / (frames * frame_dt)


class _RestProfile:
    """Height of the upper tissue surface at rest along a horizontal line (host, for planning).

    The line goes through ``origin`` (xy) along the unit vector ``direction`` (xy). The height
    at an offset along it is that of the highest surface triangle above or below that point;
    NaN where the surface has no triangle there.
    """

    def __init__(self, vertices, triangles, origin, direction, reach):
        vertices = np.asarray(vertices, dtype=float)
        corners = vertices[np.asarray(triangles)]
        self.origin, self.direction = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
        # Triangles whose box meets the box of the line's span.
        ends = self.origin + np.outer((-reach, reach), self.direction)
        low, high = ends.min(axis=0), ends.max(axis=0)
        near = np.all((corners[:, :, :2].max(axis=1) >= low) & (corners[:, :, :2].min(axis=1) <= high), axis=1)
        self.corners = corners[near]

    def height(self, offsets):
        """Surface height [m] at each offset [m] along the line."""
        a, b, c = (self.corners[:, corner] for corner in range(3))
        first, second = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
        area = first[:, 0] * second[:, 1] - second[:, 0] * first[:, 1]
        flat = np.abs(area) > 1e-18
        result = np.full(np.shape(offsets), np.nan)
        for index, offset in enumerate(np.atleast_1d(offsets)):
            point = self.origin + offset * self.direction - a[:, :2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = (point[:, 0] * second[:, 1] - second[:, 0] * point[:, 1]) / area
                v = (first[:, 0] * point[:, 1] - point[:, 0] * first[:, 1]) / area
            inside = flat & (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1.0 + 1e-12)
            if inside.any():
                heights = (
                    a[inside, 2] + u[inside] * (b[inside, 2] - a[inside, 2]) + v[inside] * (c[inside, 2] - a[inside, 2])
                )
                result.reshape(-1)[index] = heights.max()
        return result


def _plan_pinch(
    scene,
    matrix,
    *,
    start,
    hover,
    raised,
    middle,
    speed,
    transit_speed,
    clearance,
    hold_frames,
    depth,
    lift,
    gap_fraction,
    frame_dt,
    time_constant,
):
    """Plan the pinch: the open jaws straddle the ridge at the middle site, close on it, lift, lower and open.

    Everything is found on the surface at rest and on the capsule set of the tool:

    - Tips: of each jaw, the sphere whose lowest point is lowest with the jaws closed. The tip
      gap is the distance between the surfaces of the two tip spheres along the closing
      direction, a function of the half-opening of the jaws (``scene.jaw_spheres``).
    - Profile: the height of the surface along the closing direction through the middle site,
      a quarter of the jaw contact radius apart (``STATION_RADII``, the sampling of the swept
      contact), as far to each side as the tips of the fully opened jaws are apart.
    - Ridge: its top is the local maximum of the profile reached by walking uphill from the
      site; the ridge is the interval around the top in which the surface is above the height
      of the tip centres once the lowest point of the tips is ``depth`` below the top (the
      ends of the interval to 1e-9 m by bisection); its width and its middle follow.
    - Opening: the tip gap of the open jaws is the width of the ridge plus a clearance on
      each side, ``speed`` times the largest actuator time constant (the lag of the slowest
      arm actuator behind a ramp at that speed). The tip gap of the closed command is
      ``gap_fraction`` times the width.
    - Positions: the palm is placed so that the middle between the tips is over the middle of
      the ridge. It waits with closed jaws ``clearance`` above the top (the clearance of the
      other phases), and the lowest point of the open tips goes down to ``depth`` below the
      top. While the jaws close and open the palm stays where the open jaws put the tips.
    - Speeds: ``speed`` for every motion in or beside the tissue and for the jaw tips while
      they open and close (the length of their arc about the hinge); the transit speed in the
      air; every hold lasts ``hold_frames``.

    Args:
        scene: The ``ToolTissueScene`` with its surface at rest.
        matrix: Rotation of the palm (columns: closing direction, jaws, their normal).
        start: Palm position at the end of the phase before (the hold of the fast approach) [m].
        hover: Palm position with the closed jaws at the clearance above the middle site [m].
        raised: Function of a palm position: the same position at the transit height.
        middle: The middle site (xy) [m].
        depth: Depth of the lowest point of the tips below the top of the ridge [m].
        lift: How far the closed jaws are lifted from there, straight up [m].
        gap_fraction: Tip gap of the closed command in widths of the ridge.

    Returns:
        The positions of the palm and the half-openings at the end of every frame, segment by
        segment, the segments as (label, speed of the palm, hold), the number of frames of the
        lift-off, and the derived numbers.
    """
    radius = scene.jaw_contact_radius
    closing = matrix[:, 0]
    widest_opening = scene.maximum_half_opening
    closed, radii = scene.jaw_spheres(0.0)
    lowest = (closed @ matrix.T)[:, 2] - radii
    tip = [
        int(np.flatnonzero(scene.jaw_of_sphere == jaw)[np.argmin(lowest[scene.jaw_of_sphere == jaw])]) for jaw in (0, 1)
    ]
    tip_radii = radii[tip]

    def tips(opening):
        """Centres of the two tip spheres from the palm, in the world, at a half-opening [m]."""
        return scene.jaw_spheres(opening)[0][tip] @ matrix.T

    def tip_gap(opening):
        centres = tips(opening)
        return abs(float((centres[1] - centres[0]) @ closing)) - float(tip_radii.sum())

    def opening_for(gap):
        """Half-opening at which the tip gap is ``gap``: the gap grows with the opening."""
        low, high = 0.0, widest_opening
        if tip_gap(high) < gap:
            raise ValueError(
                f"The fully opened jaws are {1e3 * tip_gap(high):.2f} mm apart at their tips, less than the "
                f"{1e3 * gap:.2f} mm that the pinch needs"
            )
        if tip_gap(low) >= gap:
            return 0.0
        while high - low > 1e-10:
            half = 0.5 * (low + high)
            low, high = (half, high) if tip_gap(half) < gap else (low, half)
        return 0.5 * (low + high)

    def palm_for(site, height, opening):
        """Palm position with the middle between the tips over ``site`` (xy) and their lowest point at ``height``."""
        centres = tips(opening)
        return np.r_[site - centres.mean(axis=0)[:2], height - float((centres[:, 2] - tip_radii).min())]

    def tip_travel(first, second):
        """Length of the arc of a tip about its hinge while the half-opening goes from ``first`` to ``second`` [m]."""
        turn = abs(second - first)
        if turn <= 0.0:
            return 0.0
        chords = np.linalg.norm(tips(second) - tips(first), axis=1)
        return float(chords.max()) * (0.5 * turn) / math.sin(0.5 * turn)

    # The profile of the surface at rest along the closing direction through the middle site.
    along = closing[:2] / np.linalg.norm(closing[:2])
    widest = tip_gap(widest_opening)
    spacing = STATION_RADII * radius
    count = int(math.ceil(widest / spacing))
    offsets = spacing * np.arange(-count, count + 1)
    reference = scene.reference
    profile = _RestProfile(
        reference.vertices, reference.exterior_surface_triangles, middle, along, offsets[-1] + spacing
    )
    heights = profile.height(offsets)
    # Uphill from the site to the top of the ridge.
    top = count
    if not np.isfinite(heights[top]):
        raise ValueError("The tissue surface has no point at the middle site")
    while True:
        higher = [index for index in (top - 1, top + 1) if 0 <= index < len(offsets) and heights[index] > heights[top]]
        if not higher:
            break
        top = max(higher, key=lambda index: heights[index])
    top_height = float(heights[top])
    # The height of the tip centres at the commanded depth, and the ridge: the surface above it around the top.
    level = top_height - depth + radius
    if level >= top_height:
        raise ValueError(
            "No ridge of the tissue at the middle site that the open jaws can straddle: the tips are commanded "
            f"{1e3 * depth:.2f} mm below its top, not deeper than their radius of {1e3 * radius:.2f} mm, so no tissue "
            "lies between them"
        )
    first = last = top
    while first > 0 and heights[first - 1] > level:
        first -= 1
    while last < len(offsets) - 1 and heights[last + 1] > level:
        last += 1
    pinch_clearance = speed * time_constant
    refusal = (
        "No ridge of the tissue at the middle site that the open jaws can straddle: {} "
        f"(top {1e3 * top_height:.2f} mm, tips {1e3 * depth:.2f} mm below it, widest tip gap {1e3 * widest:.2f} mm)"
    )
    if first == 0 or last == len(offsets) - 1:
        raise ValueError(refusal.format("the surface stays above the height of the tips to the end of their reach"))

    def edge(outside, inside):
        while abs(inside - outside) > 1e-9:
            half = 0.5 * (outside + inside)
            if profile.height(half) > level:
                inside = half
            else:
                outside = half
        return 0.5 * (outside + inside)

    left, right = edge(offsets[first - 1], offsets[first]), edge(offsets[last + 1], offsets[last])
    width, ridge_middle = float(right - left), float(0.5 * (left + right))
    open_gap, closed_gap = width + 2.0 * pinch_clearance, gap_fraction * width
    if open_gap > widest:
        raise ValueError(refusal.format(f"it is {1e3 * width:.2f} mm wide at the height of the tips"))
    open_opening, closed_opening = opening_for(open_gap), opening_for(closed_gap)
    site = middle + ridge_middle * along
    wait = palm_for(site, top_height + clearance, 0.0)
    low = palm_for(site, top_height - depth, open_opening)
    lifted = low + np.array([0.0, 0.0, lift])

    blocks, openings, segments = [], [], []

    def add(label, points, opening, moved_at, hold=False):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        blocks.append(points)
        openings.append(np.broadcast_to(np.asarray(opening, dtype=float), (len(points),)).copy())
        segments.append((label, moved_at, hold))

    def move(label, begin, end, at, opening):
        if np.linalg.norm(end - begin) > 1e-9:
            points, actual = _move(begin, end, at, frame_dt)
            add(label, points, opening, actual)
        return end

    def hold(label, at, opening):
        add(label, np.tile(at, (hold_frames, 1)), opening, 0.0, hold=True)

    def turn(label, at, begin, end):
        """The jaws from one half-opening to another, their tips at no more than ``speed``; the palm stays."""
        travel = tip_travel(begin, end)
        frames = max(1, math.ceil(travel / (speed * frame_dt) - 1e-9))
        add(label, np.tile(at, (frames, 1)), begin + (end - begin) * np.arange(1, frames + 1) / frames, 0.0)
        return travel / (frames * frame_dt)

    # Out of the tissue where the fast approach left the tool, then through the air to above the ridge.
    at = move("lift off", start, hover, speed, 0.0)
    lift_off_frames = sum(len(block) for block in blocks)
    at = move("rise to the transit height", at, raised(at), transit_speed, 0.0)
    at = move("transit to the ridge", at, raised(wait), transit_speed, 0.0)
    at = move("descend above the ridge", at, wait, transit_speed, 0.0)
    open_speed = turn("open", at, 0.0, open_opening)
    at = move("descend beside the ridge", at, low, speed, open_opening)
    hold("settle", at, open_opening)
    close_speed = turn("close", at, open_opening, closed_opening)
    hold("squeeze hold", at, closed_opening)
    at = move("lift", at, lifted, speed, closed_opening)
    hold("lift hold", at, closed_opening)
    at = move("lower", at, low, speed, closed_opening)
    hold("lower hold", at, closed_opening)
    release_speed = turn("release", at, closed_opening, open_opening)
    hold("release hold", at, open_opening)
    move("retreat", at, wait, speed, open_opening)
    numbers = {
        "pinch_rule": "the open jaws straddle the ridge of the tissue at the middle site, go down beside it, close "
        "to the commanded part of its width, lift, lower and open; the palm stays where the open jaws put the tips "
        "while the jaws close and open",
        "pinch_profile_spacing_m": spacing,
        "pinch_profile_span_m": float(offsets[-1]),
        "pinch_profile_rule": f"surface at rest along the closing direction through the middle site, {STATION_RADII:g} "
        "jaw contact radii apart, as far to each side as the tips of the fully opened jaws are apart",
        "pinch_widest_tip_gap_m": widest,
        "pinch_widest_half_opening_rad": widest_opening,
        "pinch_tip_radii_m": [float(value) for value in tip_radii],
        "pinch_tip_rule": "of each jaw, the sphere of its capsule set whose lowest point is lowest with the jaws "
        "closed; tip gap: distance of the surfaces of the two along the closing direction",
        "pinch_ridge_top_m": top_height,
        "pinch_ridge_top_offset_m": float(offsets[top]),
        "pinch_ridge_level_m": level,
        "pinch_ridge_width_m": width,
        "pinch_ridge_middle_offset_m": ridge_middle,
        "pinch_ridge_rule": "top: the local maximum of the profile uphill from the site; ridge: the interval around "
        "it in which the surface is above the height of the tip centres (top less the depth plus the jaw contact "
        "radius); offsets along the closing direction from the middle site",
        "pinch_site_m": site.tolist(),
        "pinch_depth_m": depth,
        "pinch_depth_rule": "lowest point of the open tips below the top of the ridge: the depth of the second press",
        "pinch_lift_m": lift,
        "pinch_lift_rule": "how far the closed jaws go straight up from where they closed (a number of the script; "
        "at the depth of the tips they come back to the height of the top of the ridge at rest)",
        "pinch_lifted_tip_height_m": top_height - depth + lift,
        "pinch_clearance_m": pinch_clearance,
        "pinch_clearance_rule": "speed times the largest actuator time constant, on each side of the ridge",
        "pinch_open_tip_gap_m": open_gap,
        "pinch_open_half_opening_rad": open_opening,
        "pinch_gap_fraction": gap_fraction,
        "pinch_closed_tip_gap_m": closed_gap,
        "pinch_closed_half_opening_rad": closed_opening,
        "pinch_jaw_tip_speeds_m_s": {"open": open_speed, "close": close_speed, "release": release_speed},
        "pinch_jaw_speed_rule": "the jaw tips travel their arc about the hinge at no more than the speed of the "
        "motions that touch the tissue",
        "pinch_wait_height_m": top_height + clearance,
    }
    return blocks, openings, segments, lift_off_frames, numbers


def plan_path(
    scene,
    *,
    speed,
    hold_seconds,
    press_depth,
    deep_depth,
    slide_length,
    fast_depth,
    pinch_gap_fraction,
    pinch_lift=None,
    jaw_tilt=0.0,
):
    """Plan the eight phases for ``scene`` (a ``ToolTissueScene`` with its surface at rest).

    Rules, all from the scene:

    - Orientation: the jaws point along the line from the tool's port to the middle of the
      block, as with a straight wrist, and close horizontally; ``jaw_tilt`` [rad] turns the jaws
      about their closing direction towards the horizontal. It is the same in every frame.
    - Middle site: the surface above the centre of the block's bounding box. Border site: the
      held node of the block's upper surface nearest to the end of the slide that the jaws can
      touch from above.
    - Slide: along the long axis of the block, to the side on which the surface under the path
      changes its height less.
    - Clearance: the tool waits ``CLEARANCE_JAW_RADII`` jaw contact radii above the surface.
    - Speeds: ``speed`` for every motion that touches the tissue; in the air the transit speed
      at which the lag of the slowest arm actuator behind a ramp of its target (speed times
      kd / kp) equals the clearance; the fast approach moves the jaw tip
      ``FAST_RADII_PER_INTERVAL`` contact radii per physics interval along the tool's insertion
      axis. Its first frame ends at ``fast_depth``; the target then goes on at the same speed
      until the tool itself has crossed the surface (:meth:`ToolPath.next_frame`), and the hold
      brings it back to ``fast_depth``. Frames of it are planned up to the one at whose end a
      tool that lagged its target by the whole lag of the slowest actuator behind that ramp would
      have brought the jaws to the surface, and no further than the tool can reach.
    - Holds last ``hold_seconds``.
    - Pinch: after the seven phases with closed jaws, planned for the orientation that they
      chose (:func:`_plan_pinch`); the lowest point of the open jaw tips goes ``deep_depth``
      below the top of the ridge at the middle site and the closed jaws lift it by ``pinch_lift``.

    Args:
        speed: Speed of the motions that touch the tissue [m/s].
        hold_seconds: Length of every hold [s].
        press_depth: Depth of the first press and of the press at the border [m].
        deep_depth: Depth of the second press and of the slide [m].
        slide_length: Length of the slide [m].
        fast_depth: Depth at the end of the first frame of the fast approach, and of the hold after it [m].
        pinch_gap_fraction: Gap between the jaw tips that the pinch commands when it closes, in widths
            of the ridge that it takes; between zero and one.
        pinch_lift: How far the closed jaws of the pinch are lifted [m]; ``None`` takes ``deep_depth``
            (the tips come back to the height of the top of the ridge at rest).
        jaw_tilt: Turn of the jaws towards the horizontal [rad].
    """
    values = (speed, hold_seconds, press_depth, deep_depth, slide_length, fast_depth)
    if not np.isfinite(values).all() or min(values) <= 0 or not np.isfinite(jaw_tilt):
        raise ValueError("Speed, hold time, depths and slide length must be finite and positive")
    if not np.isfinite(pinch_gap_fraction) or not 0.0 <= pinch_gap_fraction < 1.0:
        raise ValueError("The closed gap of the pinch must be a part of the ridge width, from zero to below one")
    if deep_depth < press_depth:
        raise ValueError("The second press must be at least as deep as the first")
    pinch_lift = deep_depth if pinch_lift is None else float(pinch_lift)
    if not np.isfinite(pinch_lift) or pinch_lift <= 0:
        raise ValueError("The lift of the pinch must be finite and positive")
    frame_dt, interval = scene.frame_dt, scene.sim_dt
    radius = scene.jaw_contact_radius
    rest = np.asarray(scene.selection.vertices, dtype=float)
    low, high = rest.min(axis=0), rest.max(axis=0)
    middle = 0.5 * (low[:2] + high[:2])
    upper = _upper_surface_nodes(scene)
    trocar = np.asarray(scene.graspers[0].trocar_pose[:3], dtype=float)
    # Orientation: jaws (palm y) from the port to the surface node nearest to the middle; closing (palm x) horizontal.
    aim = rest[upper[np.argmin(np.linalg.norm(rest[upper, :2] - middle, axis=1))]]
    jaws = (aim - trocar) / np.linalg.norm(aim - trocar)
    closing = np.cross(jaws, (0.0, 0.0, 1.0))
    closing /= np.linalg.norm(closing)
    if jaw_tilt:
        # About the horizontal closing direction, the sign that raises the jaw tip.
        across = np.cross(closing, jaws)
        raised = math.cos(jaw_tilt) * jaws + math.sin(jaw_tilt) * across
        lowered = math.cos(jaw_tilt) * jaws - math.sin(jaw_tilt) * across
        jaws = raised if raised[2] > lowered[2] else lowered
    candidates = [_rotation(sign * closing, jaws) for sign in (1.0, -1.0)]
    clearance = CLEARANCE_JAW_RADII * radius
    time_constant = float(np.max(scene.actuator_time_constants))
    transit_speed = clearance / time_constant
    fast_speed = FAST_RADII_PER_INTERVAL * radius / interval
    fast_length = fast_speed * frame_dt

    def build(rotation, matrix):
        planner = _Planner(scene, rotation, matrix)
        hover, touch = planner.over([middle], -clearance)[0], planner.over([middle], 0.0)[0]
        press, deep = planner.over([middle], press_depth)[0], planner.over([middle], deep_depth)[0]
        # Slide: along the long axis, to the side where the touched surface changes its height less.
        axis = np.zeros(2)
        axis[int(np.argmax(high[:2] - low[:2]))] = 1.0
        slide_frames = max(1, math.ceil(slide_length / (speed * frame_dt) - 1e-9))
        offsets = slide_length * np.arange(1, slide_frames + 1) / slide_frames
        ranges = []
        for sign in (1.0, -1.0):
            sites = middle + sign * offsets[:, None] * axis
            inside = np.all((sites > low[:2]) & (sites < high[:2]), axis=1)
            heights = planner.over(sites, 0.0)[:, 2] if inside.all() else np.array([np.inf, -np.inf])
            ranges.append(float(np.ptp(heights)) if inside.all() else np.inf)
        side = 1.0 if ranges[0] <= ranges[1] else -1.0
        slide_sites = middle + side * offsets[:, None] * axis
        slide = planner.over(slide_sites, deep_depth)
        end_site = slide_sites[-1]
        lifted = planner.over([end_site], -clearance)[0]
        # Border: the held node of the upper surface nearest to the end of the slide that the jaws touch from above.
        fixed = np.zeros(len(rest), dtype=bool)
        fixed[np.asarray(scene.fixed_vertex_indices)] = True
        held = upper[fixed[upper]]
        order = held[np.argsort(np.linalg.norm(rest[held, :2] - end_site, axis=1))]
        border = None
        for node in order[:64]:
            try:
                position = planner.over([rest[node, :2]], 0.0)[0]
            except ValueError:
                continue
            # The jaws meet the surface at the node, not on something above it.
            if abs(position[2] + planner.lead[2] - rest[node, 2]) <= radius:
                border = node
                break
        if border is None:
            raise ValueError("No held node of the block's upper surface can be touched from above")
        border_site = rest[border, :2]
        border_hover = planner.over([border_site], -clearance)[0]
        border_touch = planner.over([border_site], 0.0)[0]
        border_press = planner.over([border_site], press_depth)[0]
        # Fast approach: along the insertion axis (port to palm) to the fast depth at the middle site.
        fast_end = planner.over([middle], fast_depth)[0]
        insertion = (fast_end - trocar) / np.linalg.norm(fast_end - trocar)
        fast_start = fast_end - fast_length * insertion
        if planner.depth([fast_start])[0] >= 0.0:
            raise ValueError("The fast approach would start inside the tissue")
        # The target ramps on through the surface, one frame after the other, until the run finds the tool across
        # it. The jaws of a tool on its target touch the surface after fast_touch_travel; a tool behind its target
        # by the lag of the slowest actuator (speed times kd / kp) has touched it when the target is that much
        # further: frames are planned up to there (plan_path cuts them where the tool cannot reach).
        fast_touch_travel = float(np.linalg.norm(planner.along([fast_start], insertion, 0.0)[0] - fast_start))
        fast_lag = fast_speed * time_constant
        ramp_frames = max(1, math.ceil((fast_touch_travel + fast_lag) / fast_length - 1e-9))
        ramp = fast_start + fast_length * np.arange(1, ramp_frames + 1)[:, None] * insertion
        # Transits run at the height of the highest waiting position, over whatever lies between.
        level = max(lifted[2], border_hover[2], fast_start[2], hover[2])

        def raised(position):
            return np.r_[position[:2], level]

        hold_frames = max(1, round(hold_seconds / frame_dt))
        positions, segments = [], []

        def add(phase, label, points, depth, moved_at, hold=False):
            positions.append(np.asarray(points, dtype=float).reshape(-1, 3))
            segments.append((phase, label, depth, moved_at, hold))

        def move(phase, label, start, end, at, depth):
            if np.linalg.norm(end - start) > 1e-9:
                points, actual = _move(start, end, at, frame_dt)
                add(phase, label, points, depth, actual)
            return end

        def hold(phase, at, depth, label="hold"):
            add(phase, label, np.tile(at, (hold_frames, 1)), depth, 0.0, hold=True)

        at = move("approach", "descend to the surface", hover, touch, speed, 0.0)
        at = move("press", "press", at, press, speed, press_depth)
        hold("press", at, press_depth)
        at = move("press_deep", "press on", at, deep, speed, deep_depth)
        hold("press_deep", at, deep_depth)
        add("slide", "slide", slide, deep_depth, slide_length / (slide_frames * frame_dt))
        at = slide[-1]
        hold("slide", at, deep_depth)
        at = move("lift", "lift off", at, lifted, speed, -clearance)
        hold("lift", at, -clearance)
        at = move("border", "rise to the transit height", at, raised(at), transit_speed, -clearance)
        at = move("border", "transit to the border", at, raised(border_hover), transit_speed, -clearance)
        at = move("border", "descend above the border", at, border_hover, transit_speed, -clearance)
        at = move("border", "press", at, border_press, speed, press_depth)
        hold("border", at, press_depth)
        at = move("border", "lift off", at, border_hover, speed, -clearance)
        at = move("fast", "rise to the transit height", at, raised(at), transit_speed, -clearance)
        at = move("fast", "transit to the fast approach", at, raised(fast_start), transit_speed, -clearance)
        at = move("fast", "descend to the start", at, fast_start, transit_speed, -clearance)
        hold("fast", at, -clearance, label="wait")
        add("fast", FAST_APPROACH, ramp, fast_depth, fast_speed)
        hold("fast", fast_end, fast_depth)
        return (
            planner,
            hover,
            positions,
            segments,
            # What the pinch goes on from: where the fast approach holds the tool, and the transit height.
            (fast_end, raised),
            {
                "middle_site_m": middle.tolist(),
                "slide_direction": (side * axis).tolist(),
                "slide_surface_height_range_m": min(ranges),
                "border_site_m": border_site.tolist(),
                "border_node": int(border),
                "border_surface_height_m": float(rest[border, 2]),
                "middle_surface_height_m": float(touch[2] + planner.lead[2]),
                "transit_height_m": float(level + planner.lead[2]),
                "fast_start_height_above_surface_m": float(-planner.depth([fast_start])[0]),
                # Travel of the palm from the first touch to each commanded depth: more than the depth where the
                # surface is not flat, since the depth is measured to the nearest surface point.
                "press_palm_travel_m": float(np.linalg.norm(press - touch)),
                "deep_palm_travel_m": float(np.linalg.norm(deep - touch)),
                "border_palm_travel_m": float(np.linalg.norm(border_press - border_touch)),
                "fast_palm_travel_m": float(np.linalg.norm(fast_end - touch)),
                "fast_touch_travel_m": fast_touch_travel,
                "fast_lag_m": fast_lag,
                "fast_ramp_frames_for_lag": ramp_frames,
                "insertion_axis": insertion.tolist(),
            },
        )

    # Both closing directions give a straight wrist; the one whose path the tool can reach, with the smaller roll.
    best = None
    errors = []
    for rotation, matrix in candidates:

        def posed(points, rotation=rotation):
            return np.concatenate((points, np.tile(rotation, (len(points), 1))), axis=1)

        try:
            planner, start, blocks, segments, onward, sites = build(rotation, matrix)
            start_pose = np.r_[start, rotation]
            # The arm targets in the order of the path, each solved from the one before it.
            arm = [scene.solve_arm_targets(start_pose[None])]
            for index, (segment, block) in enumerate(zip(segments, blocks, strict=True)):
                if segment[1] != FAST_APPROACH:
                    arm.append(scene.solve_arm_targets(posed(block)))
                    continue
                # The ramp through the surface ends before the first of its frames that the tool cannot reach;
                # its first frame is a frame of the path like every other.
                arm.append(scene.solve_arm_targets(posed(block[:1])))
                reached = 1
                for pose in posed(block[1:]):
                    try:
                        arm.append(scene.solve_arm_targets(pose[None]))
                    except ValueError:
                        break
                    reached += 1
                blocks[index] = block[:reached]
            arm = np.concatenate(arm)
        except ValueError as error:
            errors.append(str(error))
            continue
        roll = float(np.abs(arm[:, 3]).max())
        if best is None or roll < best[0]:
            best = (roll, planner, start_pose, blocks, arm, segments, onward, sites, matrix, rotation)
    if best is None:
        raise ValueError("The tool cannot follow the scripted path: " + "; ".join(errors))
    _, planner, start_pose, blocks, arm, segments, (fast_end, raised), sites, matrix, rotation = best
    positions = np.concatenate(blocks)
    poses = np.concatenate((positions, np.tile(rotation, (len(positions), 1))), axis=1)
    depth = planner.depth(poses[:, :3])
    tip = poses[:, :3] + scene.tip_local @ matrix.T
    # The pinch goes on from the seven phases, with the orientation that they chose: their frames, and the
    # order in which their arm targets were solved, do not depend on it.
    pinch_first = len(poses)
    pinch_blocks, pinch_openings, pinch_segments, lift_off_frames, pinch_numbers = _plan_pinch(
        scene,
        matrix,
        start=fast_end,
        hover=start_pose[:3],
        raised=raised,
        middle=middle,
        speed=speed,
        transit_speed=transit_speed,
        clearance=clearance,
        hold_frames=max(1, round(hold_seconds / frame_dt)),
        depth=deep_depth,
        lift=pinch_lift,
        gap_fraction=pinch_gap_fraction,
        frame_dt=frame_dt,
        time_constant=time_constant,
    )
    pinch_positions = np.concatenate(pinch_blocks)
    pinch_poses = np.concatenate((pinch_positions, np.tile(rotation, (len(pinch_positions), 1))), axis=1)
    try:
        arm = np.concatenate((arm, scene.solve_arm_targets(pinch_poses, initial=arm[-1])))
    except ValueError as error:
        raise ValueError(f"The tool cannot follow the pinch: {error}") from error
    opening = np.concatenate((np.zeros(pinch_first), *pinch_openings))
    # The open jaws go down beside the ridge. The ridge was found on one line under the tips; how deep any capsule
    # of both jaws comes inside the tissue at rest on the whole descent is recorded (a tilted jaw can dip a blade
    # into the tissue beside the ridge, which the contact then resolves).
    open_descent_depth = -math.inf
    for (label, _, _), block, openings in zip(pinch_segments, pinch_blocks, pinch_openings, strict=True):
        if label == "descend beside the ridge":
            open_descent_depth = float(scene.jaw_depth_at_rest(block, rotation, half_opening=float(openings[0])).max())
    pinch_numbers["pinch_open_descent_depth_m"] = open_descent_depth
    pinch_numbers["pinch_open_descent_rule"] = (
        "deepest point of any capsule of the open jaws inside the tissue at rest while they go down beside the "
        "ridge; negative: the smallest gap, positive: a capsule is inside the tissue there"
    )
    # Depth and tip of the jaws as they are commanded: opened by the half-opening of the frame.
    pinch_depth, pinch_tip = np.empty(len(pinch_poses), dtype=depth.dtype), np.empty((len(pinch_poses), 3))
    tip_sphere = int(np.flatnonzero(np.all(scene.jaw_points == scene.tip_local, axis=1))[0])
    for value in np.unique(opening[pinch_first:]):
        rows = np.flatnonzero(opening[pinch_first:] == value)
        pinch_depth[rows] = scene.jaw_depth_at_rest(pinch_positions[rows], rotation, half_opening=float(value))
        pinch_tip[rows] = pinch_positions[rows] + scene.jaw_spheres(float(value))[0][tip_sphere] @ matrix.T
    poses, depth, tip = (
        np.concatenate((poses, pinch_poses)),
        np.concatenate((depth, pinch_depth)),
        np.concatenate((tip, pinch_tip)),
    )
    blocks = [*blocks, *pinch_blocks]
    segments = [*segments, *((PINCH, label, None, moved_at, held) for label, moved_at, held in pinch_segments)]
    built, restarts, first = [], {}, 0
    for (phase, label, _, moved_at, held), block in zip(segments, blocks, strict=True):
        frames = len(block)
        inside = bool(np.any(depth[first : first + frames] > 0.0))
        # The depth at the end of the segment; of the fast approach, at the end of its first frame.
        at = first if label == FAST_APPROACH else first + frames - 1
        built.append(Segment(phase, label, first, frames, float(depth[at]), float(moved_at), held, inside))
        first += frames
    for phase in RESTART_PHASES:
        restarts[phase] = next(segment.first_frame for segment in built if segment.phase == phase)
    # The pinch begins inside the tissue, where the fast approach holds the tool: it can be placed after its lift-off.
    restarts[PINCH] += lift_off_frames
    spacing = scene.mesh_spacing
    derivation = {
        "jaw_contact_radius_m": radius,
        "jaw_contact_radius_rule": "radius of the capsule of the first jaw that carries its tip (the capsule set "
        "of the tool, dvrk_tool_capsules)",
        "mesh_spacing_m": spacing,
        "mesh_spacing_rule": "mean tetrahedron edge of the tissue block",
        "interval_s": interval,
        "frame_s": frame_dt,
        "actuator_time_constant_s": time_constant,
        "actuator_time_constant_rule": "largest kd / kp of the arm actuators: lag of a joint behind a ramp of its "
        "target is speed times kd / kp",
        "clearance_m": clearance,
        "clearance_rule": f"{CLEARANCE_JAW_RADII:g} jaw contact radii (one jaw diameter)",
        "speed_m_s": speed,
        "speed_rule": "option --speed; default: the arc speed of fem_needle_grasp",
        "speed_travel_per_interval_m": speed * interval,
        "speed_travel_per_interval_jaw_radii": speed * interval / radius,
        "speed_lag_m": speed * time_constant,
        "transit_speed_m_s": transit_speed,
        "transit_speed_rule": "clearance / actuator time constant: in the air the slowest actuator lags its target "
        "by no more than the clearance",
        "fast_speed_m_s": fast_speed,
        "fast_speed_rule": f"{FAST_RADII_PER_INTERVAL:g} jaw contact radii per physics interval (the jaw's own "
        "diameter), commanded along the insertion axis",
        "fast_ramp_rule": "the first frame of the fast approach ends at the fast depth; the target goes on at the "
        "fast speed for as long as the tip sphere of the tool is clear of the surface at rest after a frame, and the "
        "hold brings it back to the fast depth. fast_ramp_frames are planned: to the frame at whose end the target "
        "is fast_touch_travel_m (start of the ramp to the first touch of the commanded jaws) plus fast_lag_m (fast "
        "speed times the actuator time constant) from the start (fast_ramp_frames_for_lag), and no further than the "
        "tool can reach",
        "fast_length_m": fast_length,
        "fast_travel_per_interval_jaw_radii": FAST_RADII_PER_INTERVAL,
        "fast_ramp_frames": next(segment.frames for segment in built if segment.label == FAST_APPROACH),
        "hold_s": hold_seconds,
        "hold_rule": "option --hold-seconds; default: the settling time of fem_needle_grasp, "
        f"{hold_seconds / time_constant:.3g} actuator time constants",
        "press_depth_m": press_depth,
        "deep_depth_m": deep_depth,
        "fast_depth_m": fast_depth,
        "slide_length_m": slide_length,
        "depths_in_jaw_radii": [press_depth / radius, deep_depth / radius, fast_depth / radius],
        "depths_in_mesh_spacings": [press_depth / spacing, deep_depth / spacing, fast_depth / spacing],
        "depth_rule": "deepest point of the jaws as they are commanded (closed, or opened in the pinch), with the "
        "radii of their capsules, below the tissue surface at rest",
        "jaw_tilt_rad": jaw_tilt,
        "jaw_axis": matrix[:, 1].tolist(),
        "jaw_angle_below_horizontal_rad": float(-math.asin(matrix[2, 1])),
        "closing_axis": matrix[:, 0].tolist(),
        "orientation_rule": "jaws along the line from the port of PSM1 to the middle of the block (a straight "
        "wrist), closing direction horizontal; the same in every frame",
        "frames": len(poses),
        "seconds": len(poses) * frame_dt,
        **sites,
        **pinch_numbers,
        "pinch_first_frame": pinch_first,
        "pinch_restart_frame": restarts[PINCH],
        "pinch_restart_rule": "the first frame after the lift-off from the hold of the fast approach: the tool is "
        "clear of the tissue there",
        "pinch_frames": len(poses) - pinch_first,
    }
    return ToolPath(start_pose, poses, depth, tip, opening, tuple(built), restarts, derivation), arm
