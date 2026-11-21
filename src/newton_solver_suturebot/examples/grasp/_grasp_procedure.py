"""Procedural needle-arc commands through the contact-driven distal gripper.

The source example supplies the desired needle arc. A settled, measured grasp
frame converts that arc to gripper targets. The scene solves bounded joint IK
for the fixed-trocar tools; finite actuators and contact determine actual motion.
No policy or needle pose drive is used.

The numbers of the procedure
----------------------------
A number that follows from the scene stands with its derivation where it is used: the time of a
lift and the angle to which the arc is driven on (``ProceduralGraspController.__init__``), the arc
speed, the end and the bound of the pull (``_begin_pull``, ``_pull_target``, ``_pull_out_span``), the
reach margin of the joints (``_read_joints``), the rest tolerance of the needle (``_rest_tolerance``),
the turning radius of the tool and every length of the way from the one grasp to the other
(``_read_tool``, ``_rise``, ``_strip``, ``_transfer``, ``_plan_route``), where on the needle the jaws
take it (``_plan_tip_grasp``), the direction and the speed of the draw of the loose thread through the
tissue (``_begin_draw``, ``_draw_target``), the direction, the speed and the stop of the tightening of
the tied thread (``_begin_tighten``, ``_tighten_target``), and the stop and the bound of its slackening
when it is tied at its exit as well (``_slacken_target``).

The numbers below are not derived. They are choices, each named once; this table is the one place
that says what chose them. "by hand" means: set once; neither this file nor docs/usage.md records a
scan or a run that compared it with another value. Angles are written in radians rounded to four
digits; the degrees are what was scanned.

======================================  ==========  ===============================================================
name                                    value       what chose it
======================================  ==========  ===============================================================
``speed`` (argument, ``--speed``)       3 mm/s      by hand: the arc speed of the insertion and of the pull
``settle_seconds`` (argument)           0.5 s       by hand: closed jaws before the grasp frame is measured
``RAMP_SECONDS``                        0.5 s       by hand: start and stop of an arc without a step in speed
``REACHED_STEPS``                       1e-3        by hand: above the rounding of a pose difference
``CLEARANCE_SPACING_RADII``             0.25        the spacing at which the tool/tissue contact samples its own
                                                    capsules (``STATION_RADII``); coarser moves the grasp away
``CLEARANCE_REFINEMENT``                4           by hand: the grasp point is then within 0.1 mm of needle of
                                                    the exact one (docs/usage.md, the grasp at the needle tip)
``PLAN_CAPSULES_PER_STEP``              1           measured: largest controller call of the regrasp 5.9 ms in
                                                    the runs of 2026-11-17 (docs/usage.md); it changes how
                                                    many calls a plan takes, not the plan
``ROUTE_CAPSULES_PER_STEP``             4           by hand, for the same call time; changes no answer either
``WIRE_STATION_RADII``                  0.5         by hand: stations on the wire half a wire radius apart
``SEARCH_HALVINGS``                     6           by hand: 1/64 of a search step
``ABOVE_SEARCH_RADII``                  2           by hand: jaws at thirty degrees or more to the tissue
``lift_rise_seconds``                   0.75 s      by hand: the least time of a rise (the rate limit can ask more)
``lift_hold_seconds``                   0.5 s       by hand: hold of a lift, of the needle after the pull, of the
                                                    tightened thread and of the slackened one
``lift_height_limit``                   10 mm       by hand: the highest lift that is accepted
``rate_fraction``                       0.8         by hand: a lift or pull that the gripper follows
``contact_force_n``                     1e-5 N      by hand: a jaw contact that counts
``contact_loss_seconds``                0.25 s      by hand; ``pull_wait_seconds`` is the same time
``grasp_wait_seconds``                  2 s         by hand
``stage_wait_seconds``                  3 s         by hand; the jaws close in 0.5 s in the runs (docs/usage.md)
``entry_clearance``                     3 mm        by hand
``transfer_clearance``                  12 mm       by hand; the scan of the grasps below asked for the points
                                                    6 and 12 mm behind the grasp to be reachable
``release_opening_rad``                 0.6 rad     by hand
``GRASP_BITE_DEPTH_M``                  6 mm        by hand (``_dvrk_grasper.needle_to_gripper_pose``): where
                                                    the jaws hold the needle, from the palm along the jaws
``pull_grasp_roll_rad``                 -1.0472     measured, the pull (default): -60 and 30 degrees with the jaws
``pull_grasp_bite_rad``                 0.5236      exchanged, from a grid of 15 degrees with the reach test of
``pull_grasp_jaws_exchanged``           True        the plan for the dVRK port layout and the default stitch
                                                    (docs/usage.md, the way to the tip and the pull, has the
                                                    numbers that bound twist and bite)
slack of the pull through the tissue    1 element   by hand (``_pull_out_span``): the bound is one thread element
                                                    beyond the end of the pull
``draw_needle_lengths``                 1           by hand: after the pull the tool draws the thread on through
                                                    the tissue by one length of the needle (the chord from its
                                                    tail to its tip, 17.5 mm), so that the needle is clear of
                                                    the tissue by its own length before the thread is tied
``tighten_travel_bound``                7.5 mm      measured: the travel at which the tissue under the tie
                                                    collapsed when the tied thread was pulled on along the
                                                    needle's chord without a stop (7.7 mm, device, docs/usage.md,
                                                    the tie and the tightening), rounded down; the tightness
                                                    rule of the grasp scene ends the tightening at 5.2 mm
least volume at which the thread is     0.5         the choice of the thread/tissue coupling, shared with the
tight (``TIGHT_VOLUME_RATIO``)                      suture example (``_src/solvers/thread/tissue.py``); a scene
                                                    may publish a larger one for its own tissue under the tie
                                                    (``tight_volume_ratio`` of the geometry; the grasp scene:
                                                    0.75, ``STITCH_TIGHT_VOLUME_RATIO``)
plans repeated at the pregrasp          1           by hand (``_approach``)
creep of the needle between the plan    2 rest      assumed, not measured (``_read_joints``): a plan is given up
of the grasp and the closed jaws        tolerances  beyond one; the descent and the closing get as much again
grasp at reset (arc fraction, twist,    the scene   demonstration settings of the scene, not of this file
bite of the driving grasp)                          (``_grasp_scene._initial_grasp_parameters``)
======================================  ==========  ===============================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


def pose_at(plan, angle):
    """Desired xyz/xyzw needle pose on the configured source insertion arc."""
    radial = np.cos(angle) * plan["radial"] + np.sin(angle) * plan["tangent"]
    tangent = -np.sin(angle) * plan["radial"] + np.cos(angle) * plan["tangent"]
    rotation = np.column_stack((radial, tangent, plan["axis"])) @ plan["local_basis"].T
    quaternion = np.asarray(wp.quat_from_matrix(wp.mat33(*rotation.astype(np.float32).reshape(-1))), dtype=float)
    return np.r_[plan["center"] - rotation @ plan["arc_center_local"], quaternion]


def _pose_step(old, target, config):
    """World-frame gripper pose increment from ``old`` to ``target``, in units of the configured rate limits."""
    rotation = np.asarray(wp.quat(*target[3:]) * wp.quat_inverse(wp.quat(*old[3:])), dtype=float)
    if rotation[3] < 0:
        rotation = -rotation
    sine = np.linalg.norm(rotation[:3])
    rotvec = rotation[:3] * 2 * np.arctan2(sine, rotation[3]) / max(sine, 1e-12)
    return np.r_[(target[:3] - old[:3]) / config.translation_step, rotvec / config.rotation_step]


def action_to(old, target, config, *, strict=False):
    """Bound a world-frame gripper pose increment by the configured rate limits."""
    raw = _pose_step(old, target, config)
    if strict and np.max(np.abs(raw)) > 1 + 1e-5:
        raise ValueError("Reference command exceeds the configured action-rate limits")
    return np.clip(raw, -1, 1)


def line_to(old, target, config):
    """Action that takes a gripper target one frame along the straight way to ``target``.

    :func:`action_to` bounds every component of the increment by itself, which bends the way of a
    target whenever the components are unequal: the small components end first. Here the translation
    and the rotation are each scaled as a whole. The largest component moves at its rate limit and the
    others in proportion, so the target stays on the straight segment to ``target`` and turns about one
    axis. Translation and rotation each take the frames they need and need not end together.
    """
    raw = _pose_step(old, target, config)
    for part in (slice(0, 3), slice(3, 6)):
        largest = np.max(np.abs(raw[part]))
        if largest > 1.0:
            raw[part] /= largest
    return raw


# Time over which an arc speed ramps up from rest and, run backwards, down to rest [s]: a choice (table at the top).
RAMP_SECONDS = 0.5
# A gripper target stands at a pose when less than this is left to it, in units of the rate limits of the
# actions: the quaternion arithmetic of a pose difference leaves a remainder above zero (a choice).
REACHED_STEPS = 1e-3


def ramp_distance(time, speed, ramp_time=RAMP_SECONDS):
    """Integrated smoothstep speed ramp: zero initial speed and acceleration."""
    u = np.clip(time / ramp_time, 0.0, 1.0)
    return speed * (ramp_time * (u**3 - 0.5 * u**4) + np.maximum(0.0, time - ramp_time))


def ramp_speed(distance, speed, ramp_time=RAMP_SECONDS):
    """Speed of the smoothstep ramp of :func:`ramp_distance`, run backwards, with ``distance`` left to its stop.

    The ramp covers ``speed * ramp_time / 2`` while its speed goes from zero to ``speed`` as
    ``speed * (3 u^2 - 2 u^3)`` with ``u = time / ramp_time``, and ``speed * ramp_time * (u^3 - u^4 / 2)``
    by then. Backwards it is a stop without a step in speed or acceleration: with that much left
    the speed is that of the ramp at the same ``u``; with more left it is ``speed``.
    """
    left = distance / (speed * ramp_time)
    if left >= 0.5:
        return speed
    if left <= 0.0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(50):
        u = 0.5 * (low + high)
        if u**3 - 0.5 * u**4 < left:
            low = u
        else:
            high = u
    u = 0.5 * (low + high)
    return speed * u * u * (3.0 - 2.0 * u)


def ramp_way(rate, speed, ramp_time=RAMP_SECONDS):
    """Distance that the stop of :func:`ramp_speed` covers from the speed ``rate``: the inverse of that function.

    ``speed * ramp_time * (u^3 - u^4 / 2)`` for the ``u`` at which the ramp runs at ``rate``,
    ``speed * u^2 (3 - 2 u) = rate``: nothing from rest, ``speed * ramp_time / 2`` from ``speed`` or more.
    """
    part = rate / speed
    if part >= 1.0:
        return 0.5 * speed * ramp_time
    if part <= 0.0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(50):
        u = 0.5 * (low + high)
        if u * u * (3.0 - 2.0 * u) < part:
            low = u
        else:
            high = u
    u = 0.5 * (low + high)
    return speed * ramp_time * (u**3 - 0.5 * u**4)


def _pose(value):
    result = np.asarray(value, dtype=float).copy()
    if result.shape != (7,) or not np.isfinite(result).all() or np.linalg.norm(result[3:]) < 1e-8:
        raise ValueError("Procedural control needs finite xyz/xyzw poses with nonzero quaternions")
    result[3:] /= np.linalg.norm(result[3:])
    return result


def gripper_target(needle_target, needle_in_gripper):
    """Solve T_world_gripper = T_world_needle * inverse(T_gripper_needle)."""
    target = wp.transform_multiply(
        wp.transform(*_pose(needle_target)), wp.transform_inverse(wp.transform(*_pose(needle_in_gripper)))
    )
    return _pose(np.asarray(target, dtype=float))


def _quat_product(first, second):
    """Product of xyzw quaternions (..., 4), the rotation ``second`` followed by ``first``; the leading axes broadcast."""
    v1, w1, v2, w2 = first[..., :3], first[..., 3:], second[..., :3], second[..., 3:]
    return np.concatenate(
        (w1 * v2 + w2 * v1 + _cross(v1, v2), w1 * w2 - np.sum(v1 * v2, axis=-1, keepdims=True)), axis=-1
    )


def _quat_turn(quaternion, vectors):
    """Vectors (..., 3) turned by unit xyzw quaternions (..., 4); the leading axes broadcast."""
    v, w = quaternion[..., :3], quaternion[..., 3:]
    twice = 2.0 * _cross(v, vectors)
    return vectors + w * twice + _cross(v, twice)


def pull_targets(needle_pose, needle_in_gripper, center, axis, angles):
    """Gripper poses that turn a held needle about its own centre of curvature, in its own plane.

        N(phi) = N * Rot(c, e, phi)          the needle turned by phi about the line through c along e
        T(phi) = N(phi) * inverse(G)         the gripper that holds it as it is held now

    ``N`` is the needle pose in the world, ``G`` the needle pose in the frame of the gripper (the
    measured grasp), ``c`` and ``e`` the centre of curvature and the arc axis of the needle in its own
    frame. Every point of the needle then moves along the needle's own circle: the tip leads for a
    positive angle about ``e``, the sense of the insertion.

    Args:
        needle_pose: ``N``, xyz/xyzw.
        needle_in_gripper: ``G``, xyz/xyzw.
        center: ``c`` [m], in the frame of the needle.
        axis: ``e``, in the frame of the needle; any length.
        angles: Angles ``phi`` [rad], a number or shape (count,).

    Returns:
        The gripper poses ``T(phi)``, xyz/xyzw, shape (count, 7).
    """
    needle, grasp = _pose(needle_pose), _pose(needle_in_gripper)
    center, axis = np.asarray(center, dtype=float), np.asarray(axis, dtype=float)
    half = 0.5 * np.atleast_1d(np.asarray(angles, dtype=float))[:, None]
    turn = np.concatenate((np.sin(half) * (axis / np.linalg.norm(axis)), np.cos(half)), axis=1)
    # The gripper in the frame of the needle: inverse(G).
    back = grasp[3:] * np.array([-1.0, -1.0, -1.0, 1.0])
    origin = -_quat_turn(back, grasp[:3])
    quaternions = _quat_product(needle[3:], _quat_product(turn, back))
    positions = needle[:3] + _quat_turn(needle[3:], center + _quat_turn(turn, origin - center))
    return np.concatenate((positions, quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)), axis=1)


def line_targets(start, goal, config):
    """The gripper targets that :func:`line_to` commands frame by frame on the way from ``start`` to ``goal``.

    The translation takes as many frames as its largest component needs at the rate limit and the
    rotation, about one axis, as many as the largest component of its rotation vector; each ends at
    ``goal`` and stays there while the other goes on.

    Returns:
        The targets from ``start`` (first row) to ``goal`` (last row), xyz/xyzw, shape (frames + 1, 7).
    """
    start, goal = _pose(start), _pose(goal)
    raw = _pose_step(start, goal, config)
    travel, turn = float(np.max(np.abs(raw[:3]))), float(np.max(np.abs(raw[3:])))
    # A remainder below REACHED_STEPS is no way (as in ProceduralGraspController._reached).
    steps = np.arange(int(np.ceil(max(travel, turn) - REACHED_STEPS)) + 1, dtype=float)
    moved = np.minimum(steps / travel, 1.0) if travel > 0 else np.ones_like(steps)
    turned = np.minimum(steps / turn, 1.0) if turn > 0 else np.ones_like(steps)
    vector = raw[3:] * config.rotation_step
    angle = float(np.linalg.norm(vector))
    half = 0.5 * angle * turned[:, None]
    rotation = np.concatenate((np.sin(half) * (vector / max(angle, 1e-300)), np.cos(half)), axis=1)
    quaternions = _quat_product(rotation, start[3:])
    positions = start[:3] + moved[:, None] * (goal[:3] - start[:3])
    return np.concatenate((positions, quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)), axis=1)


# Resolution of the jaw clearance (:func:`jaw_clearance`), in radii of the capsule that is tested: the
# tissue surface is sampled at most this far apart and the closing jaw is turned in steps that move no
# point of its axis further. A quarter radius is the resolution at which the tool/tissue contact samples
# its own capsules (STATION_RADII of swept_tool_tissue_contact.py). The triangles that can hold the
# nearest point of the surface are then sampled, and the jaw turned, CLEARANCE_REFINEMENT times finer.
# What the sampling can miss is subtracted, so coarser values only move the grasp further from the tissue.
CLEARANCE_SPACING_RADII = 0.25
CLEARANCE_REFINEMENT = 4
# Jaw capsules with tissue in reach that one step of the plan of the tip grasp samples before it hands
# back to its caller (ProceduralGraspController._plan_tip_grasp): a choice, for a controller call of a
# few milliseconds with the sixteen jaw capsules of the dVRK tools (measured in docs/usage.md). It
# changes how many calls a plan takes, not its result.
PLAN_CAPSULES_PER_STEP = 1
# Capsules of the tool that one call asks whether they are clear on the rise of the tool
# (ProceduralGraspController._rise_blocked): a choice of the same kind; the tool stands while capsules
# are left to ask at its target.
ROUTE_CAPSULES_PER_STEP = 4
# Stations on the centre line of the needle wire are at most this many wire radii apart, so a point of
# the wire is within half of that of a station (a choice). A sample of the tissue matters to a station
# if it is within the wire radius, that half spacing and its own cover of it: WIRE_REACH_RADII in all.
WIRE_STATION_RADII = 0.5
WIRE_REACH_RADII = 1.0 + 0.5 * WIRE_STATION_RADII + CLEARANCE_SPACING_RADII
# Halvings that end a search between a point that is clear and one that is not (the grasp point on the
# wire, the point above the pregrasp): to 1/64 of the step of the search, a quarter of the finest
# spacing of the clearance where the step is a radius (a choice).
SEARCH_HALVINGS = 6
# How far beyond the transfer clearance the point above the pregrasp is looked for, in turning radii of
# the tool: jaws that point at the tissue at thirty degrees or more gain one turning radius of height
# on two of length (2 sin 30 = 1). The thirty degrees are a choice.
ABOVE_SEARCH_RADII = 2.0


def _rotation(pose):
    """Rotation matrix of an xyz/xyzw pose."""
    return np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=float).reshape(3, 3)


def _cross(first, second):
    """Cross product of two vectors of three components (the last axis), as ``numpy.cross`` forms it, without its overhead."""
    return np.stack(
        (
            first[..., 1] * second[..., 2] - first[..., 2] * second[..., 1],
            first[..., 2] * second[..., 0] - first[..., 0] * second[..., 2],
            first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0],
        ),
        axis=-1,
    )


def _segment_distance(points, start, end):
    """Distance [m] of every point (..., 3) from the segment between ``start`` and ``end``; shape (...)."""
    edge = end - start
    offsets = points - start
    along = np.clip(offsets @ edge / max(edge @ edge, 1.0e-30), 0.0, 1.0)
    return np.linalg.norm(offsets - along[..., None] * edge, axis=-1)


def _parallelogram_distance(points, origin, first, second):
    """Distance [m] of every point (n, 3) from the parallelogram ``origin + a first + b second``, a and b in [0, 1]."""
    corners = (origin, origin + first, origin + first + second, origin + second)
    border = np.min([_segment_distance(points, corners[i], corners[(i + 1) % 4]) for i in range(4)], axis=0)
    normal = _cross(first, second)
    area = np.linalg.norm(normal)
    if area <= 1.0e-12 * max(np.linalg.norm(first), np.linalg.norm(second)) ** 2:
        # No area (no descent, or an axis along the descent): the border is the whole of it.
        return border
    offsets = points - origin
    # Coordinates of the foot of each point in the plane of the parallelogram.
    a = offsets @ _cross(second, normal) / (first @ _cross(second, normal))
    b = offsets @ _cross(first, normal) / (second @ _cross(first, normal))
    inside = (a >= 0.0) & (a <= 1.0) & (b >= 0.0) & (b <= 1.0)
    return np.where(inside, np.abs(offsets @ normal) / area, border)


def _triangle_distance(points, first, second, third):
    """Distance [m] of every point (n, 3) from the triangle with the corners ``first``, ``second``, ``third``."""
    corners = (first, second, third)
    border = np.min([_segment_distance(points, corners[i], corners[(i + 1) % 3]) for i in range(3)], axis=0)
    along, across = second - first, third - first
    normal = _cross(along, across)
    area = np.linalg.norm(normal)
    if area <= 1.0e-12 * max(np.linalg.norm(along), np.linalg.norm(across), 1.0e-150) ** 2:
        # No area: the border is the whole of it.
        return border
    offsets = points - first
    a = offsets @ _cross(across, normal) / (along @ _cross(across, normal))
    b = offsets @ _cross(along, normal) / (across @ _cross(along, normal))
    inside = (a >= 0.0) & (b >= 0.0) & (a + b <= 1.0)
    return np.where(inside, np.abs(offsets @ normal) / area, border)


def _strip_distance(points, strip):
    """Distance [m] of every point (n, 3) from the flat four-sided strip with the corners ``strip`` in order."""
    first, second, third, fourth = strip
    return np.minimum(
        _triangle_distance(points, first, second, third), _triangle_distance(points, first, third, fourth)
    )


def _jaw_sweep_distance(points, start, end, sense, half_opening, steps, descent):
    """Distance [m] of points (n, 3), in the frame of the tool, from what the axis of a jaw capsule sweeps.

    The axis from ``start`` to ``end`` (frame of the closed jaw) is turned about the Z axis by ``steps + 1``
    equally spaced angles from 0 to ``sense * half_opening``; at the last angle it is moved back along
    -Y by every length up to ``descent``. The points are turned the other way against the one closed axis.

    Returns:
        The distance of each point from the turned axes and the swept parallelogram of the descent, and
        how much nearer the sweep between two neighbouring angles can be than the turned axes [m].
    """
    arm = max(np.hypot(start[0], start[1]), np.hypot(end[0], end[1]))
    angles = sense * np.linspace(0.0, half_opening, steps + 1)
    cosine, sine = np.cos(angles), np.sin(angles)
    turned = np.stack(
        (
            points[:, 0:1] * cosine + points[:, 1:2] * sine,
            points[:, 1:2] * cosine - points[:, 0:1] * sine,
            np.broadcast_to(points[:, 2:3], (len(points), steps + 1)),
        ),
        axis=-1,
    )
    closing = _segment_distance(turned, start, end).min(axis=1)
    opened = np.array([[cosine[-1], -sine[-1], 0.0], [sine[-1], cosine[-1], 0.0], [0.0, 0.0, 1.0]])
    coming = _parallelogram_distance(points, opened @ start, opened @ (end - start), np.array([0.0, -descent, 0.0]))
    return closing, coming, 0.5 * arm * half_opening / steps


class TissueSurface:
    """Triangles of a tissue surface for the distance of a tool from it; host arrays, nothing on a device.

    The surface is given by ``vertices`` (n, 3) [m] and ``triangles`` (m, 3). It is tested through sample
    points (:meth:`samples`): the corners of each triangle and the points that cut its edges into equal
    parts. Every point of a triangle then lies within the triangle's *cover* of one of its samples: the
    largest distance of a point of the triangle from its nearest corner (the circumradius of an acute
    triangle, half the longest edge of any other), divided by the number of parts.
    """

    def __init__(self, vertices, triangles):
        vertices = np.asarray(vertices, dtype=float)
        triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(triangles) or not np.isfinite(vertices).all():
            raise ValueError("A tissue surface needs finite vertices (n, 3) and at least one triangle")
        self.corners = vertices[triangles]
        edges = np.linalg.norm(self.corners - np.roll(self.corners, 1, axis=1), axis=2)
        longest = edges.max(axis=1)
        twice_area = np.linalg.norm(
            np.cross(self.corners[:, 1] - self.corners[:, 0], self.corners[:, 2] - self.corners[:, 0]), axis=1
        )
        acute = 2.0 * longest**2 < (edges**2).sum(axis=1)
        circumradius = edges.prod(axis=1) / np.where(twice_area > 0.0, 2.0 * twice_area, 1.0)
        self.cover = np.where(acute & (twice_area > 0.0), circumradius, 0.5 * longest)
        self.centres = self.corners.mean(axis=1)
        self.radii = np.linalg.norm(self.corners - self.centres[:, None, :], axis=2).max(axis=1)

    def subset(self, indices):
        """The surface of the triangles ``indices`` alone: the same triangles, covers and bounds, nothing recomputed."""
        result = object.__new__(TissueSurface)
        for name in ("corners", "cover", "centres", "radii"):
            setattr(result, name, getattr(self, name)[indices])
        return result

    def near(self, centre, radius, among=None):
        """Indices of the triangles with a point within ``radius`` [m] of ``centre`` (by their bounding spheres).

        ``among`` restricts the search to those triangle indices.
        """
        centres = self.centres if among is None else self.centres[among]
        radii = self.radii if among is None else self.radii[among]
        offsets = centres - centre
        found = np.flatnonzero(np.einsum("tc,tc->t", offsets, offsets) <= (radius + radii) ** 2)
        return found if among is None else among[found]

    def encloses(self, point, direction):
        """Whether ``point`` is inside the surface: the ray from it along ``direction`` crosses an odd number of triangles.

        For a closed surface every direction gives the same answer; for an open sheet the inside is the
        side from which the ray crosses it. A ray that meets a triangle within 1e-9 of its border (in
        barycentric coordinates) is not counted on: up to three slightly turned directions are tried instead.
        """
        point, direction = np.asarray(point, dtype=float), np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        side = np.cross(direction, np.eye(3)[int(np.argmin(np.abs(direction)))])
        side /= np.linalg.norm(side)
        other = np.cross(direction, side)
        inside = False
        # The turned directions lie along no axis and no diagonal of a mesh that is laid out along the axes.
        turns = (0.0, 0.0), (np.cos(0.3), np.sin(0.3)), (np.cos(1.9), np.sin(1.9)), (np.cos(4.1), np.sin(4.1))
        for along_side, along_other in turns:
            turned = direction + 1.0e-3 * (along_side * side + along_other * other)
            ray = turned / np.linalg.norm(turned)
            offsets = self.centres - point
            along = offsets @ ray
            across = np.einsum("tc,tc->t", offsets, offsets) - along**2
            corners = self.corners[(across <= self.radii**2) & (along >= -self.radii)]
            first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
            normal = np.cross(ray, second)
            scale = np.einsum("tc,tc->t", first, normal)
            facing = np.abs(scale) > 1.0e-30
            scale = np.where(facing, scale, 1.0)
            start = point - corners[:, 0]
            u = np.einsum("tc,tc->t", start, normal) / scale
            turn = np.cross(start, first)
            v = turn @ ray / scale
            distance = np.einsum("tc,tc->t", second, turn) / scale
            hit = facing & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (distance > 0.0)
            inside = bool(np.count_nonzero(hit) % 2)
            if not np.any(hit & (np.minimum(np.minimum(u, v), 1.0 - u - v) < 1.0e-9)):
                break
        return inside

    def samples(self, indices, spacing):
        """Sample points of the triangles ``indices`` and, for each point, the cover of its triangle [m].

        Every edge is cut into the same number of parts, the smallest for which no cover exceeds ``spacing``.
        """
        if not len(indices):
            return np.zeros((0, 3)), np.zeros(0)
        parts = max(1, int(np.ceil(self.cover[indices].max() / spacing)))
        weights = np.asarray(
            [(i, j, parts - i - j) for i in range(parts + 1) for j in range(parts + 1 - i)], dtype=float
        )
        points = np.einsum("wc,tcx->twx", weights / parts, self.corners[indices])
        # Triangle by triangle: the samples of one triangle are consecutive, the same number for each.
        return points.reshape(-1, 3), np.repeat(self.cover[indices] / parts, len(weights))


def _search_margin(capsule, half_opening):
    """How far from the box of a jaw capsule's sweep a point of the surface can be and still matter [m].

    The contact radius, twice the coarse spacing (the cover of a sample and half a step of the jaw are
    less), and what the arcs of the turning axis stand out beyond the box of its closed and open ends.
    """
    start, end, radius, _sense = capsule
    arm = max(np.hypot(start[0], start[1]), np.hypot(end[0], end[1]))
    return radius + 2.0 * CLEARANCE_SPACING_RADII * radius + arm * (1.0 - np.cos(0.5 * half_opening))


def _contact_axis(start, end, radius, reach):
    """The two ends of the axis of a capsule of contact radius ``reach`` on a model capsule of radius ``radius`` [m].

    The axis from ``start`` to ``end`` is shortened by ``reach - radius`` at each end, so that the caps
    reach as far as those of the model capsule (``capsule_samples``, the rule of the tool/tissue
    contact); an axis shorter than that is its middle.
    """
    length = float(np.linalg.norm(end - start))
    inset = (reach - radius) / length
    if length - 2.0 * (reach - radius) > 0.0:
        return start + inset * (end - start), end - inset * (end - start)
    return 0.5 * (start + end), 0.5 * (start + end)


def jaw_capsules(tool):
    """The capsules with which the jaws of a tool touch the tissue, in the frame of the closed jaws.

    ``tool`` is one entry of the scene's ``grippers`` (``GraspScene.grasp_geometry``): ``capsule_bodies``,
    ``capsule_endpoints`` [m] and ``capsule_radii`` [m] of its capsule set, ``jaw_body_indices``, and
    ``capsule_contact_radii`` [m] where the tool touches the tissue with other radii than those of its
    capsule set (without that entry the capsule radii are the contact radii). The axis of a capsule of
    another contact radius is that of :func:`_contact_axis`.

    A jaw body is the palm turned about the palm's Z axis through its origin by the jaw angle
    (``add_rcm_grasper``): positive for the first jaw, negative for the second.

    Returns:
        One row per capsule of a jaw: the two ends of its axis [m] in the frame of its body, its contact
        radius [m] and the sign of the angle that opens its jaw.
    """
    bodies = [int(body) for body in tool["capsule_bodies"]]
    jaws = [int(body) for body in tool["jaw_body_indices"]]
    endpoints = np.asarray(tool["capsule_endpoints"], dtype=float)
    radii = np.asarray(tool["capsule_radii"], dtype=float)
    contact = np.asarray(tool.get("capsule_contact_radii", radii), dtype=float)
    if (
        len(jaws) != 2
        or endpoints.shape != (len(bodies), 2, 3)
        or radii.shape != (len(bodies),)
        or contact.shape != radii.shape
        or not np.isfinite(endpoints).all()
        or not np.isfinite([radii, contact]).all()
        or min(radii.min(), contact.min()) <= 0
    ):
        raise ValueError("The tool needs two jaw bodies and finite capsules with positive radii, one radius each")
    result = []
    for body, (start, end), radius, reach in zip(bodies, endpoints, radii, contact, strict=True):
        if body not in jaws:
            continue
        if float(np.linalg.norm(end - start)) <= 1.0e-12:
            raise ValueError("Capsule axes must have nonzero length")
        result.append((*_contact_axis(start, end, radius, reach), float(reach), 1.0 if body == jaws[0] else -1.0))
    if not result:
        raise ValueError("The tool has no capsule on a jaw body")
    return result


def jaw_cover(tool):
    """The capsules of the jaws of a tool with how far the drawn jaw can be outside each, or None.

    ``tool`` is one entry of the scene's ``grippers``. A tool whose capsule set is fitted to its drawn
    parts publishes ``capsule_bounds`` [m] beside ``capsule_radii`` (``dvrk_tool_capsules``): no point of
    a drawn part is further outside its capsule than the bound.

    Returns:
        One row per capsule of a jaw: the two ends of its axis [m] in the frame of its body, its radius
        plus its bound [m] and the sign of the angle that opens its jaw; None for a tool without bounds.
    """
    if "capsule_bounds" not in tool:
        return None
    jaws = [int(body) for body in tool["jaw_body_indices"]]
    endpoints = np.asarray(tool["capsule_endpoints"], dtype=float)
    reach = np.asarray(tool["capsule_radii"], dtype=float) + np.asarray(tool["capsule_bounds"], dtype=float)
    if reach.shape != (len(endpoints),) or not np.isfinite(reach).all():
        raise ValueError("The tool needs one finite bound for each of its capsules")
    return [
        (endpoints[row, 0], endpoints[row, 1], float(reach[row]), 1.0 if int(body) == jaws[0] else -1.0)
        for row, body in enumerate(tool["capsule_bodies"])
        if int(body) in jaws
    ]


def tool_capsules(tool):
    """Every capsule with which a tool can touch the tissue, with the link of the tool that carries it.

    ``tool`` is one entry of the scene's ``grippers``, as for :func:`jaw_capsules`, whose rule gives the
    axis and the contact radius of each capsule. The link is "jaw1" or "jaw2" for the two jaw bodies and
    otherwise the name under which ``part_body_indices`` lists the body ("roll": the shaft, "pitch": the
    wrist, "yaw": the palm); None for a body that is not listed, which the procedure cannot place.

    Returns:
        One row per capsule: the two ends of its axis [m] in the frame of its body, its contact radius
        [m], how far the drawn part can be outside it [m] (``capsule_bounds``; zero for a tool without
        bounds) and its link.
    """
    bodies = [int(body) for body in tool["capsule_bodies"]]
    jaws = [int(body) for body in tool["jaw_body_indices"]]
    endpoints = np.asarray(tool["capsule_endpoints"], dtype=float)
    radii = np.asarray(tool["capsule_radii"], dtype=float)
    contact = np.asarray(tool.get("capsule_contact_radii", radii), dtype=float)
    bounds = np.asarray(tool.get("capsule_bounds", np.zeros(len(radii))), dtype=float)
    if bounds.shape != radii.shape or not np.isfinite(bounds).all() or bounds.min(initial=0.0) < 0:
        raise ValueError("The tool needs one finite nonnegative bound for each of its capsules")
    links = {int(body): str(name) for name, body in dict(tool.get("part_body_indices", {})).items()}
    links.update({jaws[0]: "jaw1", jaws[1]: "jaw2"})
    rows = []
    for body, (start, end), radius, reach, bound in zip(bodies, endpoints, radii, contact, bounds, strict=True):
        rows.append((*_contact_axis(start, end, radius, reach), float(reach), float(bound), links.get(body)))
    return rows


def sweep_clearance(surface, start, end, radius, move, *, decide=False):
    """How far a capsule stays from a tissue surface while it moves straight [m], and the point that decides.

    The axis from ``start`` to ``end`` (world frame) sweeps the parallelogram ``P`` spanned by itself and
    the translation ``move`` (the axis alone for no move). The clearance is

        min over the points x of the surface of  dist(x, P)  -  radius

    negative when the capsule reaches into the tissue anywhere on the way; an axis that lies inside the
    tissue and comes out of it on the way passes through the surface and is not clear. It is evaluated
    as :func:`jaw_clearance`: on samples of the surface at most ``CLEARANCE_SPACING_RADII * radius`` apart,
    then on the triangles that can hold the nearest point ``CLEARANCE_REFINEMENT`` times finer, with the
    cover of each sample subtracted, so the result is a lower bound.

    Args:
        surface: The tissue surface (:class:`TissueSurface`).
        start: One end of the capsule axis [m].
        end: Its other end [m].
        radius: Radius of the capsule [m].
        move: Translation of the capsule [m], shape (3,).
        decide: Leave the finer samples out when the first ones already show the capsule clear; the
            value is then the lower bound of the first samples.

    Returns:
        The lower bound of the clearance [m] and the sample of the surface it is taken at; ``inf`` and
        None if no triangle has a point within about one and a half radii of the box of the sweep.
    """
    start, end, move = (np.asarray(value, dtype=float) for value in (start, end, move))
    spacing = CLEARANCE_SPACING_RADII * radius
    extent = np.stack((start, end, start + move, end + move))
    # Triangles with a point within the radius, and what the sampling can miss, of the box of the sweep.
    reach = (radius + 2.0 * spacing + surface.radii)[:, None]
    indices = np.flatnonzero(
        np.all(
            (surface.centres >= extent.min(axis=0) - reach) & (surface.centres <= extent.max(axis=0) + reach), axis=1
        )
    )
    if not len(indices):
        return np.inf, None
    points, cover = surface.samples(indices, spacing)
    found = _parallelogram_distance(points, start, end - start, move)
    lower = found - cover
    if found.min() < radius:
        # A point of the surface is nearer than the radius: the capsule reaches into the tissue.
        return float(lower.min() - radius), points[int(np.argmin(found))]
    if decide and lower.min() >= radius:
        return float(lower.min() - radius), points[int(np.argmin(lower))]
    nearest = lower.reshape(len(indices), -1).min(axis=1) <= found.min()
    points, cover = surface.samples(indices[nearest], spacing / CLEARANCE_REFINEMENT)
    lower = _parallelogram_distance(points, start, end - start, move) - cover
    return float(lower.min() - radius), points[int(np.argmin(lower))]


def turning_clearance(surface, wire, wire_margin, strip, radius):
    """How far the tissue and a wire stay from everything within ``radius`` of a flat strip [m], and what decides.

    A tool whose palm goes straight from one point to another, turning as it likes on the way, stays
    within its turning radius of the strip between the two points and the points part of the way from
    them to its trocar (``ProceduralGraspController._strip``). It can therefore touch nothing that is
    further than that radius from the strip:

        min( min over the points x of the surface of  dist(x, strip),
             min over the points w of the wire of  dist(w, strip) - wire_margin )  -  radius

    The surface is first taken by the centres of its triangles, each standing for its triangle within
    the radius of the triangle about it; where those do not decide, the triangles that can hold the
    nearest point are sampled ``CLEARANCE_SPACING_RADII * radius / CLEARANCE_REFINEMENT`` apart at most.
    What a point stands for is subtracted: the result is a lower bound.

    Args:
        surface: The tissue surface (:class:`TissueSurface`).
        wire: Stations on the centre line of the wire [m], shape (count, 3).
        wire_margin: Radius of the wire plus what a point between two stations can be nearer [m].
        strip: The four corners of the strip in order [m].
        radius: Turning radius of the tool [m].

    Returns:
        The lower bound [m], and what it is taken at: ("tissue", the sample of the surface) or ("wire",
        the index of the station); ``inf`` and None if neither is near.
    """
    spacing = CLEARANCE_SPACING_RADII * radius
    corners = np.stack(strip)
    low, high = corners.min(axis=0), corners.max(axis=0)
    margin = radius + 2.0 * spacing
    value, witness = np.inf, None
    indices = surface.near(0.5 * (low + high), 0.5 * np.linalg.norm(high - low) + margin)
    outside = np.maximum(np.maximum(low - surface.centres[indices], surface.centres[indices] - high), 0.0)
    indices = indices[np.linalg.norm(outside, axis=1) <= margin + surface.radii[indices]]
    if len(indices):
        points = surface.centres[indices]
        found = _strip_distance(points, strip)
        lower = found - surface.radii[indices]
        if found.min() >= radius > lower.min():
            # Not decided by the centres: the triangles that can hold the nearest point, by their samples.
            points, cover = surface.samples(indices[lower <= found.min()], spacing / CLEARANCE_REFINEMENT)
            found = _strip_distance(points, strip)
            lower = found - cover
        index = int(np.argmin(found if found.min() < radius else lower))
        value, witness = float(lower.min() - radius), ("tissue", points[index])
    if len(wire):
        gaps = _strip_distance(np.asarray(wire, dtype=float), strip) - wire_margin - radius
        index = int(np.argmin(gaps))
        if gaps[index] < value:
            value, witness = float(gaps[index]), ("wire", index)
    return value, witness


def _segments_distance(first, second):
    """Smallest distance [m] between any segment of ``first`` (m, 2, 3) and any of ``second`` (n, 2, 3); shape (m,).

    Two segments are nearest at points inside both, or at an end point of one of them. The first case is
    the common perpendicular of their lines where its feet lie inside both; the second is the distance of
    an end point from the other segment (:func:`_segment_distance`).
    """
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    # Every pair at once: rows are the segments of ``first``, columns those of ``second``.
    p, a, q, b = first[:, 0], first[:, 1], second[:, 0], second[:, 1]
    u, v = a - p, b - q
    w = p[:, None, :] - q[None, :, :]
    uu, vv = np.einsum("ij,ij->i", u, u)[:, None], np.einsum("ij,ij->i", v, v)[None, :]
    uv, uw, vw = u @ v.T, np.einsum("mnj,mj->mn", w, u), np.einsum("mnj,nj->mn", w, v)
    denominator = uu * vv - uv * uv
    crossing = denominator > 1.0e-18 * np.maximum(uu * vv, 1.0e-300)
    safe = np.where(crossing, denominator, 1.0)
    s, t = (uv * vw - vv * uw) / safe, (uu * vw - uv * uw) / safe
    crossing &= (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
    between = w + s[:, :, None] * u[:, None, :] - t[:, :, None] * v[None, :, :]
    nearest = np.where(crossing, np.linalg.norm(between, axis=2), np.inf)
    # The end points of the second segments against the first ones, and the end points of the first against them.
    for point in (q, b):
        offsets = point[None, :, :] - p[:, None, :]
        part = np.clip(np.einsum("mnj,mj->mn", offsets, u) / np.maximum(uu, 1.0e-30), 0.0, 1.0)
        nearest = np.minimum(nearest, np.linalg.norm(offsets - part[:, :, None] * u[:, None, :], axis=2))
    for point in (p, a):
        offsets = point[:, None, :] - q[None, :, :]
        part = np.clip(np.einsum("mnj,nj->mn", offsets, v) / np.maximum(vv, 1.0e-300), 0.0, 1.0)
        nearest = np.minimum(nearest, np.linalg.norm(offsets - part[:, :, None] * v[None, :, :], axis=2))
    return nearest.min(axis=1)


def wire_clearance(cover, wire, wire_radius, half_opening):
    """How far the open jaws of a tool and their drawn parts are from a wire [m].

    The jaws are turned open by ``half_opening`` each about the Z axis of the tool. For each capsule of
    ``cover`` the clearance is the distance of its axis from the centre line of the wire, less its radius,
    its bound and the radius of the wire: negative when the capsule, or the drawn jaw outside it, can touch
    the wire. The result is the smallest clearance.

    Args:
        cover: Rows of :func:`jaw_cover`.
        wire: Centre line of the wire in the frame of the tool [m], shape (count, 3).
        wire_radius: Radius of the wire [m].
        half_opening: Angle by which each jaw is turned open [rad].
    """
    wire = np.asarray(wire, dtype=float)
    segments = np.stack((wire[:-1], wire[1:]), axis=1)
    axes, reach = [], []
    for start, end, radius, sense in cover:
        cosine, sine = np.cos(sense * half_opening), np.sin(sense * half_opening)
        turn = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        axes.append(np.stack((start, end)) @ turn.T)
        reach.append(radius + wire_radius)
    return float(np.min(_segments_distance(np.asarray(axes), segments) - np.asarray(reach)))


def jaw_clearance(surface, capsules, pose, bite, half_opening, descent):
    """How far the jaws of a tool stay from a tissue surface while they come down on a grasp and close [m].

    The tool comes down on the grasp pose ``pose`` along its own Y axis, from ``descent`` further back,
    with each jaw turned open by ``half_opening``, and then closes its jaws there. The axis of a jaw
    capsule sweeps a set ``K`` in that motion: the axis at every jaw angle from closed to open at the
    grasp pose, and the open axis at every point of the descent. The clearance of the capsule is

        min over the points x of the surface of  dist(x, K)  -  R

    with ``R`` its contact radius: negative when the capsule reaches into the tissue anywhere on the
    way. The result is the smallest clearance of ``capsules``, and never larger than the distance of the
    surface from the straight lines between ``bite`` and the closed axes: a surface that passes between
    the grasp point and a jaw has that jaw on its far side.

    It is evaluated on sample points of the surface at most ``CLEARANCE_SPACING_RADII * R`` apart and on
    axes turned in steps that move no point of an axis further than that; the triangles that can hold
    the nearest point (those whose bound is not above the smallest distance found) are then sampled, and
    the axes turned, ``CLEARANCE_REFINEMENT`` times finer. The cover of each sample and half the largest
    move of a step are subtracted, so the result is a lower bound: a jaw that is reported clear is clear
    of the given surface.

    Args:
        surface: The tissue surface (:class:`TissueSurface`).
        capsules: Rows of :func:`jaw_capsules`.
        pose: Grasp pose of the tool, xyz/xyzw.
        bite: Grasp point [m] in the frame of the tool: the point of the needle between the jaws.
        half_opening: Angle of each open jaw [rad], nonnegative.
        descent: Length of the descent along the tool's Y axis [m], nonnegative.

    Returns:
        The lower bound of the clearance [m]; ``inf`` if no triangle has a point within about one and a
        half radii of the box of a sweep.
    """
    rotation, translation = _rotation(pose), np.asarray(pose[:3], dtype=float)
    bite = np.asarray(bite, dtype=float)
    return min(
        _capsule_clearance(surface, capsule, rotation, translation, bite, half_opening, descent) for capsule in capsules
    )


def _capsule_clearance(surface, capsule, rotation, translation, bite, half_opening, descent, decide=False):
    """The clearance of one jaw capsule (:func:`jaw_clearance`), for a tool at ``rotation`` and ``translation``.

    With ``decide`` the finer samples are left out when the first ones already show the capsule clear:
    the value is then the lower bound of the first samples, which is all that a caller who asks only
    whether the capsule is clear needs.
    """
    start, end, radius, sense = capsule
    spacing = CLEARANCE_SPACING_RADII * radius
    # The points of the axis move on circles about the Z axis of the tool: no further than the larger
    # distance of its two ends from that axis times the angle.
    arm = max(np.hypot(start[0], start[1]), np.hypot(end[0], end[1]))
    steps = max(1, int(np.ceil(arm * half_opening / spacing)))
    angle = sense * half_opening
    opened = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    back = np.array([0.0, -descent, 0.0])
    # The nearest point of the closed axis to the grasp point.
    axis = end - start
    closed = start + np.clip((bite - start) @ axis / max(axis @ axis, 1.0e-30), 0.0, 1.0) * axis
    # Triangles with a point within the radius, and what the sampling can miss, of the box of the sweep.
    extent = np.stack((start, end, opened @ start, opened @ end, opened @ start + back, opened @ end + back, bite))
    low, high = extent.min(axis=0), extent.max(axis=0)
    margin = _search_margin(capsule, half_opening)
    indices = surface.near(rotation @ (0.5 * (low + high)) + translation, 0.5 * np.linalg.norm(high - low) + margin)
    centres = (surface.centres[indices] - translation) @ rotation
    outside = np.maximum(np.maximum(low - centres, centres - high), 0.0)
    indices = indices[np.linalg.norm(outside, axis=1) <= margin + surface.radii[indices]]
    if not len(indices):
        return np.inf

    def bounds(triangles, spacing, steps):
        points, cover = surface.samples(triangles, spacing)
        points = (points - translation) @ rotation
        closing, coming, missed = _jaw_sweep_distance(points, start, end, sense, half_opening, steps, descent)
        between = _segment_distance(points, bite, closed) - cover
        return np.minimum(closing, coming), np.minimum(closing - missed, coming) - cover, between

    found, lower, between = bounds(indices, spacing, steps)
    if found.min() < radius:
        # A point of the surface is nearer than the radius: the capsule reaches into the tissue.
        return float(lower.min() - radius)
    if decide and lower.min() >= radius:
        return min(float(lower.min() - radius), float(between.min()))
    # A triangle whose bound is above the smallest distance found does not hold the nearest point.
    nearest = lower.reshape(len(indices), -1).min(axis=1) <= found.min()
    _, lower, _ = bounds(indices[nearest], spacing / CLEARANCE_REFINEMENT, steps * CLEARANCE_REFINEMENT)
    return min(float(lower.min() - radius), float(between.min()))


@dataclass
class _Scene:
    reset_count: int
    started: float
    last_time: float
    angle: float
    status: str = "settling"
    reason: str | None = None
    grasp: np.ndarray | None = None
    progress_time: float = 0.0
    lost_since: float | None = None
    had_contact: bool = False
    relative_translation_m: float = 0.0
    relative_rotation_rad: float = 0.0
    entries: int = 0
    exits: int = 0
    lift_origin: np.ndarray | None = None
    lift_time: float = 0.0
    lift_complete: bool = False
    lift_offset: float = 0.0
    # Stage of the procedure: arc, lift, done; with the pull: arc, lift, release, retreat, approach,
    # descend, close, pull, clear, and the draw and the stitch after the clear hold: draw, drawn, tie,
    # tighten, hold, lock, slacken, hold, done.
    stage: str = "arc"
    drive_angle: float = 0.0
    lifts_complete: int = 0
    # Way points of the regrasp (ProceduralGraspController._transfer): where the tool let go, where it
    # has left the needle and the tissue, where it can turn, the point above the pregrasp, the pregrasp
    # and the grasp at the tip.
    released: np.ndarray | None = None
    via: np.ndarray | None = None
    turn: np.ndarray | None = None
    above: np.ndarray | None = None
    tip_grasp: np.ndarray | None = None
    pregrasp: np.ndarray | None = None
    stage_time: float = 0.0
    # Part of the stage: "leave" and "rise" of the retreat; "traverse", "down" and "again" of the approach.
    leg: str | None = None
    # Direction in which the tool backs away from the needle, and the rise along its shaft that is
    # open to it from ``via`` [m].
    leave_axis: np.ndarray | None = None
    rise: np.ndarray | None = None
    # What was last found in the way of the rise and of the turn, and the tissue near the tool.
    rise_witness: tuple | None = None
    rise_rows: tuple | None = None
    turn_witness: tuple | None = None
    region: tuple | None = None
    region_begun: tuple | None = None
    turn_region: tuple | None = None
    turn_clear: np.ndarray | None = None
    # The plan of the grasp at the tip while it is being made (a generator), the needle pose it is
    # made for, the time of the stage at which planning began [s], and its result: the length of
    # needle between the tip and the grasp point [m]. Then the plan of the way to it (a generator),
    # whether both stand, and how often the grasp was planned again because the needle had moved.
    plan: object | None = None
    plan_pose: np.ndarray | None = None
    plan_since: float | None = None
    tip_grasp_offset: float | None = None
    route: object | None = None
    planned: bool = False
    plans_repeated: int = 0
    # What the plan found about the reach of the tool: why its last point was refused, and how far
    # the joints stay inside their limits on the way that stands [rad, m].
    plan_notes: dict | None = None
    reach_rad: float | None = None
    reach_m: float | None = None
    # The pull (ProceduralGraspController._pull_target): the needle pose it starts from, the time it
    # has run with both jaws on the needle [s], the commanded angle and its bound [rad], the arc
    # speed [m/s], how long the command has stood at the bound [s], the arc by which the tail of the
    # needle is past the entry and the exit marker [m], whether the pull has ended as it should, and
    # the time the tool has held the needle since [s].
    pull_needle: np.ndarray | None = None
    pull_time: float = 0.0
    pull_angle: float = 0.0
    pull_bound: float = 0.0
    pull_speed: float = 0.0
    pull_wait: float = 0.0
    tail_past_entry: float | None = None
    tail_past_exit: float | None = None
    pull_complete: bool = False
    hold_time: float = 0.0
    # The draw of the loose thread through the tissue after the pull (ProceduralGraspController._draw_target):
    # the palm target it starts from, its direction, its length [m], the time it has run with both jaws on the
    # needle [s], the commanded travel [m], the largest force of the thread on the needle that the scene
    # reported meanwhile [N], and whether it has ended.
    draw_origin: np.ndarray | None = None
    draw_direction: np.ndarray | None = None
    draw_length: float = 0.0
    draw_time: float = 0.0
    draw_travel: float = 0.0
    draw_pull: float | None = None
    draw_complete: bool = False
    # The tie and the tightening after the pull through the tissue (ProceduralGraspController._stitch_target):
    # whether the tie has been asked for, and the ties that the scene reports. Then, of the tightening: the palm
    # target and the palm position it starts from, its direction, the time it has run with both jaws on the
    # needle [s], the commanded travel [m], the palm's own travel and the least tissue volume of every frame,
    # the commanded travel at which the stop that the rule has begun ends [m], and what the rule found when it
    # fired: the least volume it predicted at rest, the slope it measured [1/m] and the way to rest [m]. The
    # largest travel and turn of the needle in the jaws since the tightening began [m, rad], the calls
    # since the tie without both jaws on the needle, and whether the stitch has ended as it should.
    tie_asked: bool = False
    ties: int | None = None
    least_volume: float | None = None
    tighten_origin: np.ndarray | None = None
    tighten_palm: np.ndarray | None = None
    tighten_direction: np.ndarray | None = None
    tighten_grasp: np.ndarray | None = None
    tighten_time: float = 0.0
    tighten_travel: float = 0.0
    tighten_samples: list | None = None
    tighten_end: float | None = None
    tighten_predicted: float | None = None
    tighten_slope: float | None = None
    tighten_way: float | None = None
    tighten_start_volume: float | None = None
    tighten_slip_m: float = 0.0
    tighten_slip_rad: float = 0.0
    stitch_lost_calls: int = 0
    stitch_complete: bool = False
    # The tie at the exit and the slackening after the tightening's hold (ProceduralGraspController._stitch_target):
    # whether that tie has been asked for, and the thread's force on the needle as the scene last reported it [N].
    # Then, of the slackening: the time it has run with both jaws on the needle [s], the commanded travel back
    # along the line of the tightening [m] and the speed of its last frame [m/s], that travel and the thread's
    # force of every frame, the commanded travel at which the stop ends that the rule has begun [m], and what
    # the rule found when it began it: the thread's force [N] and the slope it had measured [N/m]. Whether the
    # slackening has ended, the physical time at which it did [s], the needle pose in the tool frame at the tie
    # at the exit, and the largest travel and turn of the needle in the jaws since [m, rad].
    lock_asked: bool = False
    thread_pull: float | None = None
    slack_time: float = 0.0
    slack_travel: float = 0.0
    slack_rate: float = 0.0
    slack_samples: list | None = None
    slack_end: float | None = None
    slack_stop_pull: float | None = None
    slack_slope: float | None = None
    slack_complete: bool = False
    slack_ended_at: float | None = None
    lock_grasp: np.ndarray | None = None
    lock_slip_m: float = 0.0
    lock_slip_rad: float = 0.0


class ProceduralGraspController:
    """Follow one source entry/exit arc using a fixed measured grasp frame.

    Each world settles independently before starting. Loss of bilateral jaw
    contact pauses its commands; ``contact_loss_seconds`` of continuous loss stop it until
    reset. The grasp frame stays fixed even if the needle slips. Completion of
    the commanded arc is reported separately from actual entry and exit events.
    An optional inspection lift follows the arc, which then continues past the plan
    until the needle is centred in the tissue: the gripper moves along the entry
    normal, holds, then returns.

    With ``pull_through`` the tool then opens its jaws (the tissue holds the needle) and
    takes the tip that has come out of the tissue, with a grasp from which it can go on
    along the needle's curve (``pull_grasp_*``). On the way from the one grasp to the other
    every capsule of the tool stays clear of the tissue at rest and of the needle
    (``_transfer``). It then turns the
    needle about the needle's own centre of curvature, in its own plane, at the speed rule
    of the insertion (``_pull_target``): the tip rises out of the exit and the tail moves
    towards the entry. The pull needs a scene whose thread follows the needle's track through
    the tissue (``geometry["thread_tracks"]``; the constructor refuses another): the tail goes
    on into the entry, through the tissue and
    out of the exit: the pull ends when the tissue holds no part of the track any more and
    the tail is one thread element (``geometry["thread_element_length"]``) past the exit
    marker, and the tool then holds the needle clear of the tissue. The thread now lies through
    the track with its far end loose, so it can take no load. The procedure therefore goes on
    (``_stitch_target``): it moves the needle on straight along its own chord, from the tail to
    the jaws, by one length of the needle, which draws that much of the loose thread through the
    track (``_draw_target``), and holds. It asks the scene to tie the thread at its first entry
    (:meth:`commands`; a stand-in for a knot), moves the needle further on along the same line
    until the thread is as tight as the tissue allows (``_tighten_target``), and
    holds. Then it asks for a second tie where the thread comes out of the tissue, takes the
    needle back along the same line until the thread to the needle is slack
    (``_slacken_target``), and holds: the thread between the two ties stays tight without
    the tool. The
    pull needs no lift before it. The plan of that grasp takes only points whose descent, closing
    and pull-out arc the tool can reach with every joint inside its limits by the reach
    margin, and the procedure stops with a reason when the needle has none
    (``_grasp_reach``). Without the argument nothing of this is used.

    ``geometry`` contains ``control_mode="physical_grasp"`` and the scene's
    ``insertion_plan``. ``config`` supplies that control mode and positive
    ``frame_dt``, ``translation_step``, ``rotation_step`` and ``jaw_step_rad``.
    Calling the controller with observation dictionaries returns one float32
    seven-value action per observation: world translation, world rotation and
    jaw closure. Apply these to gripper actuator targets, never to needle state.
    What an action cannot say, the ties of the thread, is asked for beside the
    actions of a call (:meth:`commands`), for the ``commands`` of the scenes' ``step``.
    ``env_id`` and ``reset_count`` keep reset and contact history independent.
    """

    # The choices of the procedure: the table at the top of this file says what chose each of them.
    lift_rise_seconds = 0.75
    lift_hold_seconds = 0.5
    # The highest lift that is accepted [m].
    lift_height_limit = 0.01
    # Part of the rate limits of the actions that a commanded lift or pull may use, so that the gripper follows it.
    rate_fraction = 0.8
    # A jaw holds the needle while it reports a contact with more than this normal force [N].
    contact_force_n = 1e-5
    # Time without both jaws on the needle that stops the procedure [s], time after settling that the first
    # grasp is waited for [s], and time a stage waits for what it cannot command: the jaws to close on the
    # tip, the needle to come to rest for the plan of the grasp [s].
    contact_loss_seconds = 0.25
    grasp_wait_seconds = 2.0
    stage_wait_seconds = 3.0
    # Arc length kept between the jaws and the entry point when the drive continues past the plan [m].
    entry_clearance = 0.003
    # Regrasp at the tip: length of the descent of the open jaws along their own axis onto the wire
    # [m], which is also the furthest they back away along it from where they let go, and their
    # opening [rad]. Where on the needle the jaws take it is not a setting, and neither is the way
    # there: both follow from the capsules of the tool, its trocar, the tissue, the wire and where
    # the needle rests (_plan_tip_grasp, _transfer).
    transfer_clearance = 0.012
    release_opening_rad = 0.6
    # The grasp at the tip from which the tool pulls the needle on along its curve (pull_through):
    # twist of the tool about the needle tangent -60 degrees, bite angle 30 degrees [rad] (the
    # parameters of needle_to_gripper_pose), and the tool turned half a turn about its jaw axis (the
    # jaws exchanged): without the exchange the roll joint would have to pass its limit on the way
    # over to the tip.
    pull_grasp_roll_rad = -1.0472
    pull_grasp_bite_rad = 0.5236
    pull_grasp_jaws_exchanged = True
    # The pull through the tissue ends one thread element past the exit marker (the element that
    # hangs on the tail is then outside, and the eyelet at the exit holds a free one) and may go
    # on by one more element (_pull_out_span).
    # Time the pull waits at its bound for the tail to arrive [s]: as long as a lost contact.
    pull_wait_seconds = contact_loss_seconds
    # How far the tool draws the loose thread on through the tissue after the pull, before the first tie,
    # in lengths of the needle (the chord from its tail to its tip): the needle is then clear of the
    # tissue by its own length, and as much thread has run through the track (_begin_draw).
    draw_needle_lengths = 1.0
    # Furthest the tool moves the needle on to tighten the tied thread [m]: a measured number of this
    # scene's stitch (the table at the top). The tightening stops by its rule (_tighten_target), which
    # the measured pull reaches at 5.2 mm. Pulled on without a stop, the tissue under the tie collapsed
    # at 7.7 mm (least volume from 0.64 to 0.30): a tightening that arrives here without its rule does
    # not behave as measured, and the procedure stops with the reason before it gets there. The way is
    # asked for reach to this length before it begins.
    tighten_travel_bound = 0.0075

    def __init__(
        self,
        geometry,
        config,
        speed=0.003,
        settle_seconds=0.5,
        *,
        lift_height=0.0,
        pull_through=False,
    ):
        if geometry.get("control_mode") != "physical_grasp" or config.control_mode != "physical_grasp":
            raise ValueError("Procedural gripper control requires physical_grasp")
        values = [
            speed,
            settle_seconds,
            config.frame_dt,
            config.translation_step,
            config.rotation_step,
            config.jaw_step_rad,
        ]
        if not np.isfinite(values).all() or speed <= 0 or settle_seconds < 0 or min(values[2:]) <= 0:
            raise ValueError("Speed and action scales must be positive and finite; settling time must be nonnegative")
        self.config, self.speed, self.settle_seconds = config, float(speed), float(settle_seconds)
        if not np.isfinite(lift_height) or not 0 <= lift_height <= self.lift_height_limit:
            raise ValueError(f"Lift height must be finite and between 0 and {self.lift_height_limit:g} metres")
        self.lift_height = float(lift_height)
        self.pull_through = bool(pull_through)
        # Whether the thread of the scene follows the needle's track through the tissue, as the pull
        # needs it to, and the arc by which the tail is then taken past the exit [m]: one thread element.
        self.thread_tracks, self.pull_clearance = False, 0.0
        # What the last call asks of every scene beside its action (commands()).
        self._asked = []
        if self.pull_through:
            if not geometry.get("thread_tracks"):
                raise ValueError(
                    "The pull needs a scene whose thread follows the needle's track through the tissue (thread_tracks)"
                )
            clearance = geometry.get("thread_element_length")
            if clearance is None or not np.isfinite(clearance) or clearance <= 0:
                raise ValueError(
                    "The pull through the tissue needs the length of a thread element (thread_element_length)"
                )
            self.thread_tracks, self.pull_clearance = True, float(clearance)
            # The least tissue volume at which the tied thread is as tight as the tissue allows: the rule
            # of the thread/tissue coupling, or what the scene has found for its own tissue under the tie
            # (tight_volume_ratio), which may ask for an earlier stop and never for a later one.
            from ..._src.solvers.thread.tissue import TIGHT_VOLUME_RATIO

            volume = geometry.get("tight_volume_ratio", TIGHT_VOLUME_RATIO)
            if volume is None or not np.isfinite(volume) or not TIGHT_VOLUME_RATIO <= volume < 1.0:
                raise ValueError(
                    "The least tissue volume of a tight thread (tight_volume_ratio) must be at least the "
                    f"coupling's {TIGHT_VOLUME_RATIO} and below 1"
                )
            self.tight_volume = float(volume)
            # The stitch ends with the thread tied where it leaves the last planned passage and slack
            # between there and the needle: slack is a thread that pulls the needle with less than the
            # weight that the scene itself applies to the needle [N] (zero: the scene applies none).
            weight = geometry.get("needle_weight_N")
            passages = geometry["insertion_plan"].get("expected_passages")
            if weight is None or not np.isfinite(weight) or weight < 0:
                raise ValueError(
                    "The stitch after the pull through the tissue needs the weight that the scene applies to "
                    "the needle (needle_weight_N)"
                )
            if isinstance(passages, bool) or not isinstance(passages, (int, np.integer)) or passages < 1:
                raise ValueError(
                    "The stitch after the pull through the tissue needs the number of planned passages "
                    "(insertion_plan expected_passages)"
                )
            self.needle_weight, self.passages = float(weight), int(passages)
        self.lift_normal = np.zeros(3)
        if self.lift_height or self.pull_through:
            # The lift goes along the entry normal, and the plan of the regrasp asks along it whether the
            # needle tip is inside the tissue.
            normal = np.asarray(geometry.get("entry_normal"), dtype=float)
            if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-8:
                raise ValueError("An inspection lift requires a finite nonzero entry_normal")
            self.lift_normal = normal / np.linalg.norm(normal)
        self.plan = {
            key: np.asarray(value, dtype=float).copy() if isinstance(value, (list, tuple, np.ndarray)) else value
            for key, value in geometry["insertion_plan"].items()
        }
        self.initial_angle, self.finish_angle, self.radius = (
            float(self.plan[key]) for key in ("initial_angle", "finish_angle", "radius")
        )
        if (
            not np.isfinite([self.initial_angle, self.finish_angle, self.radius]).all()
            or self.radius <= 0
            or not 0 < self.finish_angle - self.initial_angle < 2 * np.pi
        ):
            raise ValueError("The insertion plan needs a positive radius and an increasing arc shorter than one turn")
        _pose(pose_at(self.plan, self.initial_angle))
        _pose(pose_at(self.plan, self.finish_angle))
        self.scenes = {}
        # A smoothstep rise of height h in time T peaks at 1.5 h / T. Keep that within rate_fraction
        # of the translation rate limit of the actions, so the gripper follows the commanded lift.
        limit = self.rate_fraction * config.translation_step / config.frame_dt
        self.rise_seconds = max(self.lift_rise_seconds, 1.5 * self.lift_height / limit)
        # A lift pulls on the needle, and a needle that has only just come out slips back out
        # of the tissue. Before a lift the drive therefore continues until the middle of the
        # needle is midway between entry (arc angle zero) and exit, with both ends outside.
        # The regrasp of the pull lets go of the needle and needs it there as well.
        self.drive_angle = self.finish_angle
        if self.lift_height or self.pull_through:
            centred = 0.5 * float(self.plan["exit_angle"]) + 0.5 * float(geometry["needle_span"])
            self.drive_angle = max(self.finish_angle, centred)
        basis = np.asarray(self.plan["local_basis"], dtype=float)
        self.arc_center_local = np.asarray(self.plan["arc_center_local"], dtype=float)
        self.arc_axes = basis[:, 0], basis[:, 1], np.cross(basis[:, 0], basis[:, 1])
        self.tip_local = np.asarray(geometry.get("needle_tip_local", np.zeros(3)), dtype=float)
        self.centerline = np.asarray(geometry.get("needle_centerline", np.zeros((2, 3))), dtype=float)
        self.jaws = self.tissue = self.jaw_cover = self._wire_pieces = None
        self.joint_limits = self.reach_margin = None
        if self.pull_through:
            # The regrasp at the tip is planned against the jaws of the tool that is commanded and the
            # rest surface of the tissue: both as the scene publishes them, read once here.
            try:
                self.jaws = jaw_capsules(geometry["grippers"][0])
                self.jaw_cover = jaw_cover(geometry["grippers"][0])
                self.tissue = TissueSurface(geometry["reference_positions"], geometry["reference_triangles"])
                self.needle_radius = float(geometry["needle_radius"])
            except (KeyError, IndexError, TypeError) as error:
                raise ValueError(
                    "The grasp at the tip needs the capsules of the tool (grippers), the rest surface of the tissue "
                    "(reference_positions, reference_triangles) and needle_radius"
                ) from error
            if not np.isfinite(self.needle_radius) or self.needle_radius <= 0 or len(self.centerline) < 2:
                raise ValueError("The grasp at the tip needs a positive needle radius and the needle centre line")
            self._read_tool(geometry["grippers"][0])
            self._read_joints(geometry["grippers"][0])

    def _read_tool(self, tool):
        """Read what the way of the regrasp is planned with: every capsule of the tool, its trocar and the wire.

        The way from the release to the grasp at the tip keeps every capsule of the tool clear of the
        tissue, so the capsules of the shaft and the wrist are placed as well as those of the jaws. That
        needs the trocar through which the shaft passes (``trocar_pose``) and the link of every capsule
        (``part_body_indices``). A tool without a trocar is refused when the controller is made
        (``_read_joints``); one without the links of its capsules can be planned a grasp for, and the
        regrasp stops with a reason when it would have to move such a tool (``routable``).
        """
        rows = tool_capsules(tool)
        self.tool_ends = np.asarray([[start, end] for start, end, _reach, _bound, _link in rows])
        self.tool_reach = np.asarray([reach for _start, _end, reach, _bound, _link in rows])
        # Against the wire a capsule counts with the bound of its drawn part, as in wire_clearance.
        self.tool_cover = self.tool_reach + np.asarray([bound for _start, _end, _reach, bound, _link in rows])
        self.tool_links = [link for _start, _end, _reach, _bound, link in rows]
        trocar = tool.get("trocar_pose")
        self.trocar = None if trocar is None else _pose(trocar)
        self.routable = self.trocar is not None and all(
            link in ("roll", "pitch", "yaw", "jaw1", "jaw2") for link in self.tool_links
        )
        # The shortest the shaft can be between the trocar and the wrist: the lower limit of its slide.
        self.shaft_shortest = 0.0
        if "nominal_insertion_m" in tool and "insertion_limits_m" in tool:
            self.shaft_shortest = max(0.0, float(tool["nominal_insertion_m"]) + float(tool["insertion_limits_m"][0]))
        # Stations on the centre line of the wire, WIRE_STATION_RADII wire radii apart at most: a point
        # between two of them is within half of that of one.
        lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.centerline, axis=0), axis=1))]
        count = max(1, int(np.ceil(lengths[-1] / (WIRE_STATION_RADII * self.needle_radius))))
        along = lengths[-1] * np.arange(count + 1) / count
        self.stations = np.stack([np.interp(along, lengths, self.centerline[:, axis]) for axis in range(3)], axis=1)
        self.station_slack = 0.5 * lengths[-1] / count
        if not self.routable:
            return
        from ._dvrk_rcm_grasper import WRIST_LENGTH_M

        # No point of the tissue is nearer the trocar than this: its corners less the largest cover.
        corners = self.tissue.corners.reshape(-1, 3)
        self.trocar_gap = float(np.linalg.norm(corners - self.trocar[:3], axis=1).min() - self.tissue.cover.max())
        # Turning radius of the tool: no point of it, drawn parts included, is further from the strip of
        # _strip, whatever the orientation of the palm and the opening of the jaws. A jaw turns about the
        # palm origin, the wrist link holds the palm origin at a fixed point of its own frame (the wrist
        # length along its X axis, arm_link_poses), and a point of the shaft the fraction t of the way
        # from the wrist joint W to the trocar C lies (1 - t) |W - palm| from the point that far from
        # the palm origin to C.
        self.wrist_length = WRIST_LENGTH_M
        palm = np.array([WRIST_LENGTH_M, 0.0, 0.0])
        self.turning_radius, self.shaft_cover = 0.0, None
        for ends, cover, link in zip(self.tool_ends, self.tool_cover, self.tool_links, strict=True):
            if link == "roll":
                # Beside the line from W to C, and beyond W towards the palm.
                far = self.wrist_length + max(np.hypot(end[0], end[1]) + max(end[2], 0.0) for end in ends)
                self.shaft_cover = cover if self.shaft_cover is None else max(self.shaft_cover, cover)
            elif link == "pitch":
                far = np.linalg.norm(ends - palm, axis=1).max()
            else:
                far = np.linalg.norm(ends, axis=1).max()
            self.turning_radius = max(self.turning_radius, float(far + cover))

    def _read_joints(self, tool):
        """Read what the reach of the tool is tested with: its joint limits; and derive the reach margin.

        The plan is made for the needle where it rests, and the tool is commanded later to poses
        that follow the needle as it then is. Between the two:

        - the scene's pose IK meets a target within its tolerances, which the scene publishes for
          the tool (``ik_position_tolerance_m``, ``ik_orientation_tolerance_rad``);
        - the needle creeps with the tissue. An end of the wire that has moved by more than the
          rest tolerance ``e`` of the plan (:meth:`_rest_tolerance`) has the plan made again, up to
          the pregrasp (:meth:`_needle_moved`); the descent and the closing that follow are given
          as much again (assumed). Both ends within ``2 e`` turn the needle by no more than
          ``4 e / chord`` about an axis across the chord between its ends; a turn about the chord
          itself is not seen by that test and is taken as none.

        A commanded pose is then within ``dx = tolerance + 2 e`` and ``dq = tolerance + 4 e / chord``
        of the planned one, and the joints answer with at most

            m   = (dq + dx / l) / cos(w)     for the rotary joints [rad],
            m_s = dx + L m                   for the slide [m],

        ``l`` the shortest shaft between trocar and wrist (a displacement ``dx`` of the palm turns
        the shaft about the trocar by ``dx / l``), ``w`` the largest wrist pitch inside the limits
        (the roll axis and the wrist-yaw axis come together as the wrist pitch goes to a quarter
        turn: a turn of the palm asks ``1 / cos(w)`` of it from them) and ``L`` the length of the
        wrist link. With the dVRK tools: ``e`` 0.040 mm, chord 17.5 mm, ``l`` 130 mm, ``w`` 1.1 rad:
        0.024 rad and 0.31 mm. Where the jaws have closed the pull is tested once more, from the
        measured grasp (:meth:`_begin_pull`).
        """
        from ._dvrk_rcm_grasper import WRIST_LENGTH_M

        try:
            limits = np.asarray(tool["arm_joint_limits"], dtype=float)
            self.nominal_insertion = float(tool["nominal_insertion_m"])
            self.ik_position_tolerance = float(tool["ik_position_tolerance_m"])
            self.ik_orientation_tolerance = float(tool["ik_orientation_tolerance_rad"])
        except KeyError as error:
            raise ValueError(
                "The pull needs the joint limits of the tool, its nominal insertion and the tolerances of its "
                "pose IK (arm_joint_limits, nominal_insertion_m, ik_position_tolerance_m, ik_orientation_tolerance_rad)"
            ) from error
        if (
            self.trocar is None
            or limits.shape != (2, 6)
            or not np.isfinite(limits).all()
            or np.any(limits[0] >= limits[1])
            or self.nominal_insertion + limits[0, 2] <= 0
            or np.max(np.abs(limits[:, 4])) >= 0.5 * np.pi
        ):
            raise ValueError(
                "The pull needs the trocar of the tool and six increasing finite joint limits, with the wrist "
                "behind the trocar and a wrist pitch of less than a quarter turn"
            )
        tolerances = self.ik_position_tolerance, self.ik_orientation_tolerance
        if not np.isfinite(tolerances).all() or min(tolerances) < 0:
            raise ValueError("The pull needs finite nonnegative tolerances of the pose IK of the tool")
        self.joint_limits = limits
        creep = 2.0 * self._rest_tolerance()
        chord = float(np.linalg.norm(self.centerline[-1] - self.centerline[0]))
        turn = self.ik_orientation_tolerance + 2.0 * creep / chord
        shift = self.ik_position_tolerance + creep
        rotary = (turn + shift / (self.nominal_insertion + limits[0, 2])) / np.cos(np.max(np.abs(limits[:, 4])))
        self.reach_margin = np.array([rotary, rotary, shift + WRIST_LENGTH_M * rotary, rotary, rotary, rotary])

    def _reach(self, poses):
        """Whether the tool reaches a row of palm poses, one after the other, with every joint inside its limits.

        The joints of each pose are those of the fixed-trocar tool in closed form
        (``arm_joint_positions``: a palm pose has one set of them). The row is reached when at every
        pose every joint is inside its limits by the reach margin (:meth:`_read_joints`) and by what
        that joint moves from the pose before and to the pose after: between two poses of the row it
        then stays inside its limits as well, and a roll joint that would have to pass its limit to
        get from one pose to the next is not inside.

        Args:
            poses: Palm poses in the order in which they are commanded, xyz/xyzw, shape (count, 7).

        Returns:
            None if the row is reached, else the joint that is not inside, how far it is inside its
            limit and how far it has to be (a text); then the smallest distance from a limit over
            the row of a rotary joint [rad] and of the slide [m].
        """
        from ._dvrk_rcm_grasper import ARM_JOINT_NAMES, arm_joint_positions

        joints = np.atleast_2d(arm_joint_positions(self.trocar, poses, self.nominal_insertion))
        inside = np.minimum(joints - self.joint_limits[0], self.joint_limits[1] - joints)
        moved = np.zeros_like(joints)
        if len(joints) > 1:
            steps = np.abs(np.diff(joints, axis=0))
            moved[1:] = steps
            moved[:-1] = np.maximum(moved[:-1], steps)
        short = (inside - self.reach_margin - moved) / self.reach_margin
        slide = np.arange(6) == 2
        least = float(inside[:, ~slide].min()), float(inside[:, slide].min())
        row, joint = np.unravel_index(int(np.argmin(short)), short.shape)
        if short[row, joint] >= 0.0:
            return None, *least
        unit = "m" if joint == 2 else "rad"
        return (
            f"{ARM_JOINT_NAMES[joint]} is {inside[row, joint]:.3g} {unit} inside its limit where "
            f"{self.reach_margin[joint] + moved[row, joint]:.3g} {unit} are needed",
            *least,
        )

    def _tail_past(self, needle_pose, marker):
        """Arc [m] by which the tail of the needle is past a marker on the tissue, along the needle's own circle.

        Tail and marker are taken by their angle about the axis of the needle through its centre of
        curvature, in the sense of the insertion: negative while the tail has still to come to the
        marker, zero when it is beside it.
        """
        needle = _pose(needle_pose)
        rotation = _rotation(needle)
        centre, axis = rotation @ self.arc_center_local + needle[:3], rotation @ self.arc_axes[2]
        tail = rotation @ self.centerline[-1] + needle[:3] - centre
        mark = np.asarray(marker, dtype=float) - centre
        tail, mark = tail - (tail @ axis) * axis, mark - (mark @ axis) * axis
        return self.radius * float(np.arctan2(_cross(mark, tail) @ axis, tail @ mark))

    def _follow_tail(self, state, needle_pose, item):
        """Keep the arcs by which the tail of the needle is past the entry and the exit marker of an observation [m]."""
        state.tail_past_entry = self._tail_past(needle_pose, item["entry_position"])
        state.tail_past_exit = self._tail_past(needle_pose, item["exit_position"])

    def _pull_row(self, needle_pose, held, span):
        """Palm poses of a pull over the angle ``span`` [rad], one every rotation step of the actions (:func:`pull_targets`).

        ``held`` is the needle pose in the frame of the tool, as the needle is held through the pull.
        """
        angles = np.linspace(0.0, span, int(np.ceil(span / self.config.rotation_step)) + 1)
        return pull_targets(needle_pose, held, self.arc_center_local, self.arc_axes[2], angles)

    def _pull_span(self, needle_pose, marker, short, slack):
        """Angle [rad] of the pull that takes the tail of the needle to ``short`` [m] of arc before a marker, with the slack beyond.

        ``slack`` [m] is the arc beyond; a negative ``short`` is an arc past the marker.
        """
        return (max(-self._tail_past(needle_pose, marker) - short, 0.0) + slack) / self.radius

    def _pull_out_span(self, needle_pose, exit_position):
        """Angle [rad] that bounds the pull out of the tissue from where the needle is: the reach that the tool needs.

        The arc that takes the tail one thread element past the exit marker, and one more element
        beyond, as far as the resting needle tells.
        """
        return self._pull_span(needle_pose, exit_position, -self.pull_clearance, self.pull_clearance)

    def _grasp_reach(self, needle_pose, grasp, exit_position):
        """Whether the tool reaches a grasp at the tip and what follows it, by its joints (:meth:`_reach`).

        Two rows of palm poses are asked:

        - the descent: straight along the jaws from the pregrasp, the transfer clearance back, to
          the grasp, where the jaws close; a pose every translation step of the actions;
        - the pull: the needle turned about its own centre of curvature from where it rests, held
          as at this grasp (:func:`pull_targets`), a pose every rotation step of the actions, over
          the arc that takes the tail of the needle out of the tissue (:meth:`_pull_out_span`): to
          the bound of the pull, two thread elements past the exit marker; as far as the resting
          needle tells.

        Returns:
            None or why the grasp is not reached (the first row that is not), and the smallest
            distance from a limit over the two rows of a rotary joint [rad] and of the slide [m].
        """
        count = int(np.ceil(self.transfer_clearance / self.config.translation_step))
        back = self._backed_off(grasp)[:3] - grasp[:3]
        descent = np.tile(grasp, (count + 1, 1))
        descent[:, :3] += np.linspace(1.0, 0.0, count + 1)[:, None] * back
        rows = [("the descent", descent)]
        held = wp.transform_multiply(wp.transform_inverse(wp.transform(*grasp)), wp.transform(*_pose(needle_pose)))
        span = self._pull_out_span(needle_pose, exit_position)
        rows.append(("the pull", self._pull_row(needle_pose, np.asarray(held, dtype=float), span)))
        refusal, rotary, slide = None, np.inf, np.inf
        for name, poses in rows:
            why, least_rad, least_m = self._reach(poses)
            if why is not None and refusal is None:
                refusal = f"in {name} {why}"
            rotary, slide = min(rotary, least_rad), min(slide, least_m)
        return refusal, rotary, slide

    @property
    def lift_duration(self):
        """Commanded time of one lift, hold and return; contact loss can extend it."""
        return 2 * self.rise_seconds + self.lift_hold_seconds if self.lift_height else 0.0

    def _arc_angle(self, point):
        """Angle of a needle-frame point about the arc axis of the needle."""
        relative = point - self.arc_center_local
        return float(np.arctan2(relative @ self.arc_axes[1], relative @ self.arc_axes[0]))

    def _tip_grasp(self, needle_pose, offset):
        """Gripper pose that holds the needle ``offset`` [m] of centre line behind its tip, where the needle rests."""
        from types import SimpleNamespace

        from ._dvrk_grasper import needle_to_gripper_pose

        shape = SimpleNamespace(centerline=self.centerline, arc_normal=self.arc_axes[2])
        lengths = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)
        relative = needle_to_gripper_pose(
            shape,
            arc_fraction=offset / float(lengths.sum()),
            tangent_roll_rad=self.pull_grasp_roll_rad,
            bite_angle_rad=self.pull_grasp_bite_rad,
            jaws_exchanged=self.pull_grasp_jaws_exchanged,
        )
        target = wp.transform_multiply(wp.transform(*_pose(needle_pose)), wp.transform(*relative))
        return _pose(np.asarray(target, dtype=float))

    def _wire_point(self, offset):
        """Point of the needle centre line ``offset`` [m] behind the tip, in the frame of the needle."""
        lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.centerline, axis=0), axis=1))]
        return np.asarray([np.interp(offset, lengths, self.centerline[:, axis]) for axis in range(3)])

    def _free_wire(self, needle_pose, among, inside=None):
        """Length of the needle from its tip [m] over which the wire is clear of the tissue surface.

        Stations half a wire radius apart along the centre line, from the tip: the length ends at the
        last station before the first one at which the surface, by its samples less their cover, is
        nearer than the wire radius plus the quarter radius that a point between two stations can be
        away from both. Zero if the tip itself is not clear, or is inside the tissue (by the parity of
        the surface crossings of a ray from it along the lift normal; ``inside`` is that answer where
        the caller has it already).
        """
        rotation, translation = _rotation(needle_pose), np.asarray(needle_pose[:3], dtype=float)
        lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.centerline, axis=0), axis=1))]
        radius = self.needle_radius
        free = 0.0
        if inside is None:
            inside = self.tissue.encloses(rotation @ self.centerline[0] + translation, self.lift_normal)
        if inside:
            return free
        # A triangle within reach of a station has a point in the box of the wire, grown by that reach.
        wire = self.centerline @ rotation.T + translation
        grown = WIRE_REACH_RADII * radius + 0.5 * np.diff(lengths).max()
        centres, radii = self.tissue.centres[among], self.tissue.radii[among][:, None]
        among = among[
            np.all(
                (centres + radii >= wire.min(axis=0) - grown) & (centres - radii <= wire.max(axis=0) + grown), axis=1
            )
        ]
        if self._wire_pieces is None:
            # Stations of each piece of the centre line, in the frame of the needle: made once.
            self._wire_pieces = []
            for start, end in zip(lengths[:-1], lengths[1:], strict=True):
                count = max(1, int(np.ceil((end - start) / (WIRE_STATION_RADII * radius))))
                offsets = start + (end - start) * np.arange(count + 1) / count
                self._wire_pieces.append((offsets, np.asarray([self._wire_point(offset) for offset in offsets])))
        for (offsets, local), end in zip(self._wire_pieces, lengths[1:], strict=True):
            stations = local @ rotation.T + translation
            centre = 0.5 * (stations[0] + stations[-1])
            reach = 0.5 * np.linalg.norm(stations[-1] - stations[0]) + WIRE_REACH_RADII * radius
            points, cover = self.tissue.samples(
                self.tissue.near(centre, reach, among), CLEARANCE_SPACING_RADII * radius
            )
            if len(points):
                gap = np.linalg.norm(stations[:, None, :] - points[None, :, :], axis=2) - cover[None, :]
                touching = np.flatnonzero(gap.min(axis=1) < radius + 0.5 * (offsets[1] - offsets[0]))
                if len(touching):
                    return float(offsets[touching[0] - 1]) if touching[0] > 0 else free
            free = float(end)
        return free

    def _plan_tip_grasp(self, needle_pose, exit_position=None, notes=None):
        """Choose where the jaws take the needle by its tip; a generator, one clearance evaluation per step.

        The grasp point is the point of the needle nearest to where it leaves the tissue at which the
        jaws stay out of the tissue while they come down on it and close:

            s* = max { s in [r_w, s_free] :  jaw_clearance(S, J, T(s), b(s), o / 2, D) >= 0
                                             and wire_clearance(C, W(s), r_w, o / 2) >= 0 }

        - ``s``: length of the needle centre line from the tip to the grasp point [m];
        - ``r_w``: radius of the wire (``needle_radius``): the grasp point lies on the wire, not on its
          rounded end;
        - ``s_free``: length from the tip over which the wire is clear of the tissue (:meth:`_free_wire`);
        - ``S``: rest surface of the tissue (``reference_positions``, ``reference_triangles``). With the
          tool away, the tissue around the resting needle is within 0.03 mm of it (measured), and the
          controller's observations do not hold the tissue in a run without a view;
        - ``J``: capsules of the jaws and their contact radii, from the tool model (:func:`jaw_capsules`);
        - ``T(s)``: tool pose of the grasp at ``s`` for the needle where it rests (``needle_pose`` of the
          observation after the tool has backed away), twist and bite of this class; ``b(s)``: the grasp
          point in the frame of the tool;
        - ``o``: opening of the jaws in the transfer (``release_opening_rad``); ``D``: length of the
          descent (``transfer_clearance``);
        - ``C``: the capsules of the jaws with the bound of each (:func:`jaw_cover`), where the tool model
          has bounds; ``W(s)``: the centre line of the wire in the frame of the tool at ``T(s)``. The open
          jaws come onto the wire along their own axis and the needle's tip then lies towards their roots:
          at ``s*`` neither a capsule of a jaw nor the drawn jaw (which is outside its capsule by at most
          its bound) touches the wire before the jaws close (:func:`wire_clearance`).

        Search: ``s_free`` first, then towards the tip in steps of the smallest jaw radius until the
        jaws are clear, then six halvings between that point and the last one that was not (to 1/64 of
        that radius). A window of clear points shorter than one radius between points that are not clear
        can be passed over; the point that is returned is clear.

        With ``exit_position`` (the pull) a point counts only if the tool also reaches it: the descent
        to the grasp, the closing and the pull-out arc from it, with every joint inside its limits by
        the reach margin (:meth:`_grasp_reach`). The orientation of the grasp is the same along one
        piece of the needle's centre line and steps from piece to piece, so the reach can change from
        one point of the search to the next; the point that is returned is clear and reached.

        A point of the search needs only the sign of the clearance. The wire is asked first, then the
        capsules one by one, the one that was last in the way first, against the triangles that any of
        them can reach at that grasp (taken once for all); a capsule that the first samples of the
        surface already show clear is not sampled finer, and the first capsule that is not clear ends
        the point.

        Yields:
            None after each step of work, so that a caller can spread the plan over its frames.

        Returns:
            The grasp pose of the tool (xyz/xyzw) and ``s*`` [m], or None if no point of the wire that
            sticks out is clear: the tip does not stick out far enough for these jaws. With
            ``exit_position``, also None if no point that was looked at is clear and reached;
            ``notes["unreachable"]`` then says why the last such point was not reached, and for a
            point that is returned ``notes["reach_rad"]`` and ``notes["reach_m"]`` are the smallest
            distances of a rotary joint and of the slide from a limit (:meth:`_grasp_reach`).
        """
        needle_pose = _pose(needle_pose)
        rotation, translation = _rotation(needle_pose), needle_pose[:3]
        length = float(np.linalg.norm(np.diff(self.centerline, axis=0), axis=1).sum())
        # The wire lies within its length of the tip; the triangles that can matter are taken once.
        tip = rotation @ self.centerline[0] + translation
        half, descent = 0.5 * self.release_opening_rad, self.transfer_clearance
        step = min(radius for _, _, radius, _ in self.jaws)
        # What a clearance looks at lies within this of the grasp point: the origin of the tool from it is
        # added below, then a jaw from the origin, the descent, and the reach of the search about the sweep.
        reach = max(
            max(np.linalg.norm(start), np.linalg.norm(end)) + _search_margin((start, end, radius, sense), half)
            for start, end, radius, sense in self.jaws
        )
        # Two looks at the whole surface, then a step: the triangles within the length of the wire of the
        # tip and what the search can look at from there, and whether the tip is inside the tissue.
        near = self.tissue.near(tip, length + reach + descent)
        inside = self.tissue.encloses(tip, self.lift_normal)
        yield
        free = self._free_wire(needle_pose, near, inside)
        order = list(range(len(self.jaws)))

        def grasp(offset):
            pose = self._tip_grasp(needle_pose, offset)
            return pose, (rotation @ self._wire_point(offset) + translation - pose[:3]) @ _rotation(pose)

        def clear(offset):
            pose, bite = grasp(offset)
            frame, origin = _rotation(pose), pose[:3]
            if self.jaw_cover is not None:
                # The wire in the frame of the tool at the grasp: the open jaws must be clear of it too.
                wire = (self.centerline @ rotation.T + translation - origin) @ frame
                if wire_clearance(self.jaw_cover, wire, self.needle_radius, half) < 0.0:
                    return False
            if exit_position is not None:
                # The joints of the tool on the descent to this grasp and on the pull from it: a step of its own.
                refusal = self._grasp_reach(needle_pose, pose, exit_position)[0]
                yield
                if refusal is not None:
                    if notes is not None:
                        notes["unreachable"] = refusal
                    return False
            # The sweeps of this grasp lie within the reach of a jaw of the line of the descent.
            local = field.subset(field.near(origin - 0.5 * descent * frame[:, 1], 0.5 * descent + reach))
            sampled = 0
            for index in tuple(order):
                value = _capsule_clearance(local, self.jaws[index], frame, origin, bite, half, descent, decide=True)
                if value < 0.0:
                    order.remove(index)
                    order.insert(0, index)
                    return False
                # A capsule with tissue in reach was sampled: after PLAN_CAPSULES_PER_STEP of them the
                # point goes on in the caller's next step, if capsules are left.
                sampled += value < np.inf
                if sampled == PLAN_CAPSULES_PER_STEP and index != order[-1]:
                    sampled = 0
                    yield
            return True

        lowest = self.needle_radius
        offset = min(free, length * (1.0 - 1.0e-9))
        yield
        if offset < lowest:
            return None
        # The triangles that can matter, taken once: within the free wire, the origin of the tool from the
        # grasp point and the reach of the tip. They are among those of the first look unless the grasp
        # point is further from the origin of the tool than the wire is long.
        looked = free + np.linalg.norm(grasp(offset)[1]) + reach + descent
        field = self.tissue.subset(self.tissue.near(tip, looked, near if looked <= length + reach + descent else None))
        # The first point is the end of the free wire itself; the search goes on from the last blocked point.
        blocked, found = None, None
        while found is None:
            passed = yield from clear(offset)
            yield
            if passed:
                found = offset
            elif offset <= lowest:
                return None
            else:
                blocked, offset = offset, max(offset - step, lowest)
        for _ in range(SEARCH_HALVINGS if blocked is not None else 0):
            offset = 0.5 * (found + blocked)
            passed = yield from clear(offset)
            yield
            if passed:
                found = offset
            else:
                blocked = offset
        pose = self._tip_grasp(needle_pose, found)
        if exit_position is not None and notes is not None:
            _, notes["reach_rad"], notes["reach_m"] = self._grasp_reach(needle_pose, pose, exit_position)
        return pose, found

    def _backed_off(self, pose, distance=None):
        """``pose`` moved back along the jaws (the gripper's local Y axis), by the transfer clearance or ``distance`` [m]."""
        result = pose.copy()
        along = wp.quat_rotate(wp.quat(*pose[3:]), wp.vec3(0.0, 1.0, 0.0))
        result[:3] -= (self.transfer_clearance if distance is None else distance) * np.asarray(along, dtype=float)
        return result

    @staticmethod
    def _shifted(pose, shift):
        """``pose`` moved by the translation ``shift`` [m], with its orientation."""
        result = pose.copy()
        result[:3] += shift
        return result

    @staticmethod
    def _stop(state, reason):
        """End the procedure of a scene, until its reset, with the reason why."""
        state.status, state.reason = "stopped", reason

    def _reaches(self, state, poses, refusal, keep=True):
        """Whether the tool reaches a row of palm poses (:meth:`_reach`); if not, the procedure stops.

        The reason is ``refusal`` and the joint that is not inside its limits. With ``keep`` the smallest
        distances from a joint limit over a row that is reached go into the joint margins of the scene,
        which the plan of the grasp has begun (:meth:`_tip_grasp_planned`).
        """
        why, least_rad, least_m = self._reach(poses)
        if why is not None:
            self._stop(state, f"{refusal}: {why}")
            return False
        if keep:
            state.reach_rad, state.reach_m = min(state.reach_rad, least_rad), min(state.reach_m, least_m)
        return True

    @staticmethod
    def _ask_rise(state, leg):
        """Start a leg in which the rise of the tool is asked afresh: what was found in its way is forgotten."""
        state.leg, state.rise_witness, state.rise_rows, state.region = leg, None, None, None

    def _begin_rise(self, state, palm_pose):
        """Start the rise along the shaft from ``palm_pose``, during which the grasp at the tip is planned afresh."""
        state.via, state.leg, state.rise = palm_pose.copy(), "rise", self._rise(palm_pose)
        state.plan, state.route, state.planned, state.plan_since = None, None, False, None
        state.turn_witness, state.turn_region, state.turn_clear = None, None, None

    def _closing(self, jaw_target):
        """Jaw action that takes the commanded opening of the jaws to zero, at their rate limit at most."""
        return np.clip(-(jaw_target[0] - jaw_target[1]) / self.config.jaw_step_rad, -1, 0)

    def _reached(self, old, target):
        """Whether a gripper target stands at ``target``.

        A remainder below ``REACHED_STEPS`` of the rate limits means the target is reached.
        """
        return bool(np.max(np.abs(_pose_step(old, target, self.config))) < REACHED_STEPS)

    def _rest_tolerance(self):
        """How far an end of the wire may move for the needle to count as at rest [m]: the finest spacing of the clearance."""
        return CLEARANCE_SPACING_RADII * min(radius for _, _, radius, _ in self.jaws) / CLEARANCE_REFINEMENT

    def _needle_moved(self, state, item):
        """Whether an end of the wire is further than the rest tolerance from where the plan of the tip grasp has it."""
        ends = self.centerline[[0, -1]]
        now = _pose(item["needle_pose"])
        moved = (ends @ _rotation(now).T + now[:3]) - (ends @ _rotation(state.plan_pose).T + state.plan_pose[:3])
        return bool(np.linalg.norm(moved, axis=1).max() > self._rest_tolerance())

    def _wire(self, needle_pose):
        """Stations on the centre line of the wire in the world [m], for a needle pose."""
        return self.stations @ _rotation(needle_pose).T + np.asarray(needle_pose[:3], dtype=float)

    def _safe_reach(self, wire):
        """Radius about the trocar within which there is neither tissue nor wire [m]."""
        nearest = float(np.linalg.norm(wire - self.trocar[:3], axis=1).min())
        return min(self.trocar_gap, nearest - self.needle_radius - self.station_slack)

    def _tool_axes(self, palm_pose, half_opening, safe):
        """Where the capsules of the tool are for a palm pose: their axes in the world [m].

        The jaws are the palm turned about its Z axis by ``half_opening`` each way; the shaft and the
        wrist follow from the palm pose and the trocar (``arm_link_poses``). A point within ``safe``
        of the trocar (:meth:`_safe_reach`), less the radius and bound of its capsule, touches
        nothing. The axis of a shaft capsule is therefore cut down to what lies further from the
        trocar, and a shaft capsule with nothing left is not live.

        Returns:
            The axes (count, 2, 3) with those of the shaft cut down, which capsules are live, and the
            axes as they are.
        """
        from ._dvrk_rcm_grasper import arm_link_poses

        rotation, origin = _rotation(palm_pose), np.asarray(palm_pose[:3], dtype=float)
        cosine, sine = np.cos(half_opening), np.sin(half_opening)
        frames = {
            "yaw": (rotation, origin),
            "jaw1": (rotation @ np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]), origin),
            "jaw2": (rotation @ np.array([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]]), origin),
        }
        if "roll" in self.tool_links or "pitch" in self.tool_links:
            roll, pitch = arm_link_poses(self.trocar, palm_pose)
            frames["roll"], frames["pitch"] = (_rotation(roll), roll[:3]), (_rotation(pitch), pitch[:3])
        whole = np.stack(
            [
                ends @ frames[link][0].T + frames[link][1]
                for ends, link in zip(self.tool_ends, self.tool_links, strict=True)
            ]
        )
        axes, live = whole.copy(), np.ones(len(whole), dtype=bool)
        for row, link in enumerate(self.tool_links):
            if link != "roll":
                continue
            limit = safe - self.tool_cover[row]
            outside = np.linalg.norm(whole[row] - self.trocar[:3], axis=1) > limit
            if not outside.any():
                live[row] = False
            elif not outside.all():
                # The point of the axis at the limit from the trocar, between its inner and its outer end.
                inner, outer = whole[row][~outside][0], whole[row][outside][0]
                direction, offset = outer - inner, inner - self.trocar[:3]
                a, b, c = direction @ direction, 2.0 * (offset @ direction), offset @ offset - limit * limit
                cut = inner + (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a) * direction
                axes[row] = (cut, outer) if outside[1] else (outer, cut)
        return axes, live, whole

    def _tissue_near(self, state, origin, first, second, begin=False):
        """The tissue that the jaws and the wrist of the tool can come near, for palm origins on a parallelogram.

        The palm origin lies on the parallelogram at ``origin`` spanned by ``first`` and ``second``
        [m]. The jaws and the wrist stay within the turning radius of it, and a clearance looks no
        further beyond their axes than one and a half of their largest radius. The set is kept in
        ``state.region`` while the parallelogram is the same. A capsule of the shaft, which reaches
        further, is asked against the whole surface instead. With ``begin`` only the look at the whole
        surface is made, and the next call for the same parallelogram finishes the set: two calls of
        a caller that has the time share the work.

        Returns:
            The surface near the jaws and the wrist; None after ``begin``.
        """
        key = np.r_[origin, first, second]
        if state.region is None or not np.array_equal(state.region[0], key):
            shaft = np.asarray([link == "roll" for link in self.tool_links])
            reach = self.turning_radius
            reach += (1.0 + 2.0 * CLEARANCE_SPACING_RADII) * self.tool_reach[~shaft].max(initial=0.0)
            begun = state.region_begun
            if begun is None or not np.array_equal(begun[0], key):
                corners = origin + np.array([np.zeros(3), first, second, first + second])
                centre = 0.5 * (corners.min(axis=0) + corners.max(axis=0))
                size = 0.5 * np.linalg.norm(corners.max(axis=0) - corners.min(axis=0))
                begun = state.region_begun = (key, self.tissue.near(centre, size + reach))
                if begin:
                    return None
            indices, state.region_begun = begun[1], None
            distance = _parallelogram_distance(self.tissue.centres[indices], origin, first, second)
            state.region = (key, self.tissue.subset(indices[distance <= reach + self.tissue.radii[indices]]))
        return state.region[1]

    def _rise(self, palm_pose):
        """The straight rise that is open to the tool at ``palm_pose`` [m]: along its shaft to the end of its slide.

        Along the line from the wrist joint to the trocar only the slide of the shaft moves: every
        link of the tool is carried along without turning.
        """
        from ._dvrk_rcm_grasper import arm_link_poses

        roll, _pitch = arm_link_poses(self.trocar, palm_pose)
        shaft = self.trocar[:3] - roll[:3]
        length = float(np.linalg.norm(shaft))
        return shaft / length * max(length - self.shaft_shortest, 0.0)

    def _leave_axis(self, palm_pose, half_opening):
        """Direction in which the tool backs away from the needle once its jaws have opened (a unit vector).

        The wire lies between the open jaws. When the tool backs away along the axis of one of its
        open jaws, that jaw moves along its own length and the other one moves away from it: the
        wire comes no nearer to either and leaves between them. Of the two axes the tool takes the
        one along which it rises faster towards its trocar, the side it came from (a choice: the
        jaw that lies towards the tissue then leaves it along its own length instead of across it).
        """
        rotation, rise = _rotation(palm_pose), self._rise(palm_pose)
        cosine, sine = np.cos(half_opening), np.sin(half_opening)
        back = [-(rotation @ np.array([-sense * sine, cosine, 0.0])) for sense in (1.0, -1.0)]
        return max(back, key=lambda direction: float(direction @ rise))

    def _rise_blocked(self, state, item, target, half_opening):
        """Whether the tissue or the wire is in the way of the tool when it rises from ``target`` along its shaft.

        On the rise that is open to the tool (:meth:`_rise`) the axis of each capsule, with the jaws
        at ``half_opening``, sweeps a parallelogram. The rise is clear when every capsule stays clear
        of the rest surface of the tissue on it (:func:`sweep_clearance`: its contact radius, a lower
        bound) and when no station of the wire is nearer to the sweep than the radius of the capsule,
        the bound of its drawn part, the wire radius and what a point between two stations can be
        nearer. A capsule that starts inside the tissue passes through its surface on the way and is
        in the way. What was last found in the way is asked first: a sample of the surface inside the
        sweep of its capsule, or a station of the wire within it, settles the question at once.
        Otherwise the capsules are asked in the order of their distance from it, ``ROUTE_CAPSULES_PER_STEP``
        of them in one call (a capsule of the shaft ends the call).

        Returns:
            True if something is in the way, False if the rise is clear, None if capsules are left to
            ask at this target: the next call with the same target goes on with them.
        """
        wire = self._wire(item["needle_pose"])
        safe = self._safe_reach(wire)
        axes, live, _whole = self._tool_axes(target, half_opening, safe)
        move = self._rise(target)
        if state.rise_rows is None or not np.array_equal(state.rise_rows[0], target):
            rows = [row for row in range(len(axes)) if live[row]]
            if state.rise_witness is not None:
                kind, row, what = state.rise_witness
                point = wire[what] if kind == "wire" else what
                if live[row]:
                    start, end = axes[row]
                    limit = self.tool_cover[row] + self.needle_radius if kind == "wire" else self.tool_reach[row]
                    if _parallelogram_distance(point[None], start, end - start, move)[0] < limit:
                        return True
                rows.sort(key=lambda row: float(np.linalg.norm(axes[row].mean(axis=0) - point)))
            state.rise_rows = (target.copy(), rows)
        rows = state.rise_rows[1]
        near = self._tissue_near(state, *self._rise_field(state, target, move))
        asked = 0
        while rows and asked < ROUTE_CAPSULES_PER_STEP:
            row = rows.pop(0)
            start, end = axes[row]
            gaps = _parallelogram_distance(wire, start, end - start, move) - self.tool_cover[row]
            if gaps.min() < self.needle_radius + self.station_slack:
                state.rise_witness, state.rise_rows = ("wire", row, int(np.argmin(gaps))), None
                return True
            # A capsule of the shaft is asked against the whole surface, which is the work of a call.
            shaft = self.tool_links[row] == "roll"
            field, asked = (self.tissue, ROUTE_CAPSULES_PER_STEP) if shaft else (near, asked + 1)
            value, point = sweep_clearance(field, start, end, self.tool_reach[row], move, decide=True)
            if value < 0.0:
                state.rise_witness, state.rise_rows = ("tissue", row, point), None
                return True
        if rows:
            return None
        state.rise_witness, state.rise_rows = None, None
        return False

    def _leave_field(self, state, target, begin=False):
        """Take the tissue near the way on which the tool will back away from ``target`` (:meth:`_tissue_near`)."""
        self._tissue_near(state, *self._rise_field(state, target, None), begin)

    def _rise_field(self, state, target, move):
        """The parallelogram of palm origins that the rises of a leg lie on: its origin and its two sides [m].

        While the tool backs away from where it let go, the rise is asked at every target on the way.
        The rises from all of them lie within the parallelogram of the way back and the rise from its
        start, grown along the way back by what the direction of the rise can turn on it (the way,
        times the length of the rise over the length of the shaft): the tissue near is taken once for
        the leg. Elsewhere the rise is asked at one target and is its own parallelogram.
        """
        if state.leg != "leave":
            return target[:3], np.zeros(3), move
        rise = self._rise(state.released)
        shaft = float(np.linalg.norm(rise)) + self.shaft_shortest
        grown = self.transfer_clearance * float(np.linalg.norm(rise)) / max(shaft, 1.0e-30)
        back = (self.transfer_clearance + 2.0 * grown) * state.leave_axis
        return state.released[:3] - grown * state.leave_axis, back, rise

    def _strip(self, first, second, safe):
        """Corners of the strip that the tool stays near while its palm goes straight from ``first`` to ``second``.

        Whatever the orientation of the palm and the opening of the jaws, the jaws and the wrist stay
        within their reach of the palm origin. A point of the shaft the fraction ``t`` of the way from
        the wrist joint to the trocar ``C`` is within the wrist length of the point that fraction of
        the way from the palm origin to ``C``, and from the fraction

            f = max over the two ends p of  1 - (safe - shaft cover) / (|p - C| + wrist length)

        on it is within ``safe`` of the trocar, less its capsule, where it touches nothing
        (:meth:`_safe_reach`). So no point of the tool is further than the turning radius from the
        four-sided strip with the corners ``first``, ``second``, ``second + f (C - second)`` and
        ``first + f (C - first)``; without a shaft capsule it is the segment.
        """
        fraction, trocar = 0.0, self.trocar[:3]
        if self.shaft_cover is not None:
            for point in (first, second):
                longest = float(np.linalg.norm(point - trocar)) + self.wrist_length
                fraction = max(fraction, 1.0 - (safe - self.shaft_cover) / longest)
            fraction = min(fraction, 1.0)
        return first, second, second + fraction * (trocar - second), first + fraction * (trocar - first)

    def _turn_blocked(self, state, item, first, second):
        """Whether tissue or wire is within the turning radius of the tool on the straight way of its palm.

        The palm goes straight from ``first`` to ``second`` [m] while the tool turns as it likes. It
        can touch nothing if the rest surface of the tissue and the wire, with its radius, are
        further than the turning radius from the strip of that way (:meth:`_strip`,
        :func:`turning_clearance`). What was last found within it is asked first, and a way that was
        found clear is not looked at again while the tool stands.

        Returns:
            Whether the way is blocked, and whether the tissue had to be looked at afresh for the answer.
        """
        way = np.r_[first, second]
        if state.turn_clear is not None and np.array_equal(state.turn_clear, way):
            return False, False
        wire = self._wire(item["needle_pose"])
        strip = self._strip(first, second, self._safe_reach(wire))
        radius = self.turning_radius
        if state.turn_witness is not None:
            kind, what = state.turn_witness
            point, limit = (wire[what], radius + self.needle_radius) if kind == "wire" else (what, radius)
            if _strip_distance(point[None], strip)[0] < limit:
                return True, False
        corners = np.stack(strip)
        centre = 0.5 * (corners.min(axis=0) + corners.max(axis=0))
        need = (
            0.5 * np.linalg.norm(corners.max(axis=0) - corners.min(axis=0))
            + radius
            + 2.0 * (CLEARANCE_SPACING_RADII * radius)
        )
        region = state.turn_region
        if region is None or np.linalg.norm(centre - region[0]) + need > region[1]:
            # Kept while the strip stays within half a transfer clearance of where it was.
            reach = need + 0.5 * self.transfer_clearance
            region = state.turn_region = (centre, reach, self.tissue.subset(self.tissue.near(centre, reach)))
        value, witness = turning_clearance(region[2], wire, self.needle_radius + self.station_slack, strip, radius)
        state.turn_witness, state.turn_clear = (witness, None) if value < 0.0 else (None, way)
        return value < 0.0, True

    def _transfer(self, state, item, old_target, jaw_target, contact, elapsed, relative):
        """Action of the regrasp: open, leave the needle and the tissue, rise, move over the tip, come down, close.

        From the pose at which the jaws have let go to the grasp at the tip, the commanded way keeps
        every capsule of the tool (shaft, wrist, the jaws at the opening they have) clear of the
        rest surface of the tissue and of the wire, by the clearance of the plan of the grasp
        (the contact radius of the capsule, a lower bound of the distance):

        - retreat, first part: straight back along the axis of an open jaw (:meth:`_leave_axis`),
          as far as it takes until the rise from there is clear (:meth:`_rise_blocked`), and no
          further than the transfer clearance. This part starts where the arc ended, which can be
          inside the tissue. The tool must reach the whole of it, the transfer
          clearance back, with every joint inside its limits (:meth:`_reach`): asked when the jaws
          have opened, before the first step back;
        - retreat, second part: straight up along the shaft, until the tool can turn where it is
          and on its way to the point above the pregrasp (:meth:`_turn_blocked`). The grasp at the
          tip and the way down to it are planned meanwhile (:meth:`_tip_grasp_planned`). The rise
          that is open to the tool ends at the limit of its slide (:meth:`_rise`), where no margin
          is left: each step of the rise is asked for reach before it is commanded;
        - approach: the palm goes straight to the point above the pregrasp while the tool turns to
          the orientation of the grasp, then down along the jaws to the pregrasp, where the needle
          is looked at again (:meth:`_approach`);
        - descent: straight along the jaws to the grasp, as the plan of the grasp has it.

        Every length follows from the capsules, the trocar, the tissue and the wire; the speeds
        are the rate limits of the actions (:func:`line_to`). A way that cannot be kept clear,
        or one that the tool does not reach, stops the procedure with its reason.
        """
        action = np.zeros(7, dtype=np.float32)
        opening = float(jaw_target[0] - jaw_target[1])
        state.stage_time += elapsed
        state.status = state.stage
        if state.stage == "release":
            action[6] = np.clip((self.release_opening_rad - opening) / self.config.jaw_step_rad, 0, 1)
            if not self.routable:
                self._stop(
                    state,
                    "The regrasp needs the trocar of the tool and the link of each of its capsules "
                    "(trocar_pose, part_body_indices) to keep the tool out of the tissue",
                )
                action[:] = 0
            elif state.released is None:
                # The tool stands while its jaws open. The open jaws will leave the wire along the axis of
                # one of them: the tissue near that way is taken now, in a call that has nothing else to do.
                state.released = old_target.copy()
                state.leave_axis = self._leave_axis(old_target, 0.5 * self.release_opening_rad)
                self._ask_rise(state, "leave")
                self._leave_field(state, old_target, begin=True)
            elif state.region is None:
                self._leave_field(state, old_target)
            elif opening >= self.release_opening_rad - 1e-9 and not contact:
                # The joints of the tool on the whole way back, frame by frame as it can be commanded.
                end = self._shifted(state.released, self.transfer_clearance * state.leave_axis)
                reached = self._reaches(
                    state,
                    line_targets(state.released, end, self.config),
                    "The open jaws cannot back away from the needle within the joint limits of the tool",
                    keep=False,
                )
                if reached:
                    state.stage, state.stage_time = "retreat", 0.0
                else:
                    action[:] = 0
            return action
        if state.stage == "close":
            action[6] = self._closing(jaw_target)
            if contact and opening <= 1e-9:
                # The measured grasp at the tip is the frame of everything that follows.
                state.grasp = relative.copy()
                state.had_contact, state.lost_since = True, None
                self._begin_pull(state, item, relative)
            elif state.stage_time > self.stage_wait_seconds:
                self._stop(state, "The jaws did not close on the needle tip")
            return action
        if state.stage == "retreat":
            target = self._retreat(state, item, old_target, 0.5 * opening)
        elif state.stage == "approach":
            target = self._approach(state, item, old_target, 0.5 * opening)
        else:
            target = state.tip_grasp
        if state.status == "stopped":
            return action
        action[:6] = line_to(old_target, target, self.config)
        if state.stage == "descend" and np.max(np.abs(action[:6])) < REACHED_STEPS:
            state.stage, state.stage_time = "close", 0.0
        return action

    def _retreat(self, state, item, old_target, half_opening):
        """Target of the retreat: back away from the needle and the tissue, then up to where the tool can turn."""
        if state.leg == "leave":
            blocked = self._rise_blocked(state, item, old_target, half_opening)
            if blocked:
                end = self._shifted(state.released, self.transfer_clearance * state.leave_axis)
                if self._reached(old_target, end):
                    self._stop(
                        state,
                        "The open jaws cannot leave the needle and the tissue by backing away along their own axis",
                    )
                return end
            if blocked is None:
                return old_target
            self._begin_rise(state, old_target)
            return old_target
        # The tool is clear of the tissue and rises: the needle comes to rest and the grasp is planned.
        # A plan that stands is given up if the needle has moved since: the tissue it was pressed into
        # can go on creeping for seconds. One piece of work per call: a step of the plan, unless the
        # turn had to be looked at afresh.
        if state.planned and self._needle_moved(state, item):
            state.planned, state.turn_clear = False, None
        goal = state.above if state.planned else old_target
        blocked, looked = self._turn_blocked(state, item, old_target[:3], goal[:3])
        if not state.planned and not looked:
            self._tip_grasp_planned(state, item)
            if state.status == "stopped":
                return old_target
        if blocked:
            top = self._shifted(state.via, state.rise)
            if self._reached(old_target, top):
                self._stop(state, "The tool cannot rise far enough to turn towards the needle tip clear of the tissue")
            else:
                # The rise ends at the limit of the slide: the joints of the tool on its next step.
                self._reaches(
                    state,
                    line_targets(old_target, top, self.config)[:2],
                    "The tool cannot rise far enough to turn towards the needle tip within its joint limits",
                    keep=False,
                )
            return top
        if not state.planned:
            # High enough to turn: the tool waits here for the plan.
            return old_target
        # The joints of the tool on the straight way over, frame by frame as it will be commanded.
        if not self._reaches(
            state,
            line_targets(old_target, state.above, self.config),
            "The tool cannot turn over to the needle tip within its joint limits",
        ):
            return old_target
        state.turn = old_target.copy()
        state.stage, state.stage_time, state.leg = "approach", 0.0, "traverse"
        state.status = state.stage
        return state.above

    def _approach(self, state, item, old_target, half_opening):
        """Target of the approach: over to the point above the pregrasp, down to the pregrasp, then the needle again.

        The jaws close some seconds after the grasp was planned. At the pregrasp the needle is
        therefore compared with the pose the plan was made for: if an end of the wire has moved by
        more than the rest tolerance of the plan, the tool rises from where it stands and the grasp
        and the way to it are planned again (the retreat, second part). A needle that has moved once
        more by the time the tool is back at the pregrasp stops the procedure (a choice: one
        repetition).

        A limit: from the pregrasp on the needle is not looked at again. The descent and the
        closing go to the grasp as it was planned, and a needle that moves in that time is closed
        on where the plan had it: the reach margin allows for the creep of a resting needle only
        (:meth:`_read_joints`). With the pull, the arc of the pull is then tested once more from
        the grasp as the jaws measure it (:meth:`_begin_pull`).
        """
        if state.leg == "traverse":
            if not self._reached(old_target, state.above):
                return state.above
            state.leg = "down"
        if state.leg == "down":
            if not self._reached(old_target, state.pregrasp):
                return state.pregrasp
            if not self._needle_moved(state, item):
                state.stage, state.stage_time = "descend", 0.0
                state.status = state.stage
                return state.tip_grasp
            if state.plans_repeated:
                self._stop(state, "The needle did not stay at rest for the grasp at its tip")
                return old_target
            self._ask_rise(state, "again")
        # The needle has moved: the tool rises from where it stands, if that is clear, and plans again.
        blocked = self._rise_blocked(state, item, old_target, half_opening)
        if blocked:
            self._stop(
                state,
                "The needle moved after the grasp at its tip was planned, and the tool cannot rise to plan it again",
            )
        elif blocked is not None:
            state.plans_repeated += 1
            self._begin_rise(state, old_target)
            state.stage, state.stage_time = "retreat", 0.0
            state.status = state.stage
        return old_target

    def _plan_route(self, state):
        """Plan the way from where the tool turns down to the grasp at the tip; a generator of steps of work.

        The palm comes to the line of the descent at the point above the pregrasp: the grasp backed
        off along its jaws by the transfer clearance, or by the least distance beyond it at which
        tissue and wire are further from the palm's strip than the turning radius
        (:func:`turning_clearance`), so that the tool can end its turn there. The distance is searched
        outwards in steps of a quarter turning radius and then by six halvings; it is looked for no
        further than two turning radii beyond the transfer clearance (a choice: jaws that point at
        the tissue at thirty degrees or more gain a turning radius of height on that length).

        From that point the tool comes down along its jaws with its orientation fixed. The jaws
        move straight: down to the pregrasp each of their capsules must stay clear of the tissue
        (:func:`sweep_clearance`) and of the wire, as on the rise (:meth:`_rise_blocked`); from the
        pregrasp on they are the plan of the grasp. The shaft and the wrist turn about the trocar as
        the palm comes down, so they are placed at points of the whole way down to the grasp, no
        further apart than a quarter of the smallest of their radii, and each must be clear of the
        tissue there by more than its axis moves from the point before.

        With the pull the tool must also reach the way down from the point above the pregrasp to the
        pregrasp with every joint inside its limits (:meth:`_reach`).

        Yields:
            None after each step of work.

        Returns:
            None when the way stands (``state.above`` is set), else the reason why there is none.
        """
        grasp = state.tip_grasp
        wire = self._wire(state.plan_pose)
        safe = self._safe_reach(wire)
        along = _rotation(grasp)[:, 1]
        radius, margin = self.turning_radius, self.needle_radius + self.station_slack
        nearest, furthest = self.transfer_clearance, self.transfer_clearance + ABOVE_SEARCH_RADII * radius
        # The tissue that a strip of this search can come near, taken once.
        strip = np.stack(self._strip(grasp[:3] - nearest * along, grasp[:3] - furthest * along, safe))
        centre = 0.5 * (strip.min(axis=0) + strip.max(axis=0))
        reach = (
            0.5 * np.linalg.norm(strip.max(axis=0) - strip.min(axis=0)) + (1.0 + 2.0 * CLEARANCE_SPACING_RADII) * radius
        )
        field = self.tissue.subset(self.tissue.near(centre, reach))
        yield

        def blocked(distance):
            palm = grasp[:3] - distance * along
            return turning_clearance(field, wire, margin, self._strip(palm, palm, safe), radius)[0] < 0.0

        distance, inner = nearest, None
        while blocked(distance):
            yield
            if distance >= furthest:
                return "There is no room above the needle tip for the tool to turn towards it"
            inner, distance = distance, min(distance + CLEARANCE_SPACING_RADII * radius, furthest)
        yield
        for _ in range(SEARCH_HALVINGS if inner is not None else 0):
            middle = 0.5 * (inner + distance)
            if blocked(middle):
                inner = middle
            else:
                distance = middle
            yield
        above = self._backed_off(grasp, distance)
        half = 0.5 * self.release_opening_rad
        axes, live, whole = self._tool_axes(above, half, safe)
        down = (distance - self.transfer_clearance) * along
        near = self._tissue_near(state, above[:3], np.zeros(3), distance * along)
        yield
        for row, link in enumerate(self.tool_links):
            if link in ("roll", "pitch") or not live[row]:
                continue
            start, end = axes[row]
            if distance > self.transfer_clearance:
                gaps = _parallelogram_distance(wire, start, end - start, down) - self.tool_cover[row]
                if gaps.min() < self.needle_radius + self.station_slack:
                    return "The open jaws would touch the needle on their way down to it"
            # A capsule on the palm goes on to the grasp; the jaws are planned from the pregrasp on.
            move = down if link in ("jaw1", "jaw2") else distance * along
            if np.any(move) and sweep_clearance(near, start, end, self.tool_reach[row], move, decide=True)[0] < 0.0:
                return "The jaws would reach into the tissue on their way down to the needle tip"
        yield
        links = [row for row, link in enumerate(self.tool_links) if link in ("roll", "pitch")]
        if links:
            count = max(1, int(np.ceil(distance / (CLEARANCE_SPACING_RADII * self.tool_reach[links].min()))))
            before, asked = whole, 0
            for index in range(count + 1):
                pose = self._backed_off(grasp, distance * (1.0 - index / count))
                axes, live, whole = self._tool_axes(pose, half, safe)
                for row in links:
                    if not live[row]:
                        continue
                    moved = float(np.linalg.norm(whole[row] - before[row], axis=1).max())
                    # As on the rise: a capsule of the shaft against the whole surface is the work of a call.
                    shaft = self.tool_links[row] == "roll"
                    field, asked = (self.tissue, ROUTE_CAPSULES_PER_STEP) if shaft else (near, asked + 1)
                    value = sweep_clearance(field, *axes[row], self.tool_reach[row], np.zeros(3), decide=True)[0]
                    if value < moved:
                        return "The wrist or the shaft of the tool would reach into the tissue on the way down to the needle tip"
                    if asked >= ROUTE_CAPSULES_PER_STEP:
                        asked = 0
                        yield
                before = whole
        # The joints of the tool on the way down to the pregrasp (the descent from there is in the plan).
        why, least_rad, least_m = self._reach(line_targets(above, state.pregrasp, self.config))
        if why is not None:
            return f"The tool cannot come down on the needle tip within its joint limits: {why}"
        state.reach_rad, state.reach_m = min(state.reach_rad, least_rad), min(state.reach_m, least_m)
        state.above = above
        return None

    def _tip_grasp_planned(self, state, item):
        """Plan the grasp at the tip and the way to it, one step per call; True once both stand.

        Where the tip grasp is depends on where the needle rests once the tool has left it and the tissue:
        tissue that the jaws pressed down holds the needle down with it and lets it come up when the
        tool backs away. The plan therefore starts when the tool has left the tissue, from the needle
        pose of that observation, and it stands only if no end of the wire has moved, while it was
        made, by more than the finest spacing of the clearance (a sixteenth of the smallest jaw
        radius); otherwise the needle was not at rest and the plan is made again. A plan that stands
        is given up in the same way when the needle is found further than that from its pose while the
        tool still rises (:meth:`_retreat`). The procedure stops when the wire has no point for the
        jaws (none that is clear and that the tool reaches), when the needle has not
        come to rest ``stage_wait_seconds`` after the first plan was begun (the time the closing jaws are given as well),
        or when there is no clear way down to the grasp (:meth:`_plan_route`, which follows the plan
        of the grasp).
        """
        if state.planned:
            return True
        if state.route is not None:
            try:
                next(state.route)
            except StopIteration as routed:
                state.route = None
                if routed.value is None:
                    state.planned = True
                else:
                    self._stop(state, routed.value)
            return state.planned
        if state.plan is None:
            state.plan_pose = _pose(item["needle_pose"])
            state.plan_notes = {}
            # The plan takes only points that the tool reaches (:meth:`_grasp_reach`).
            marker = np.asarray(item["exit_position"], dtype=float)
            state.plan = self._plan_tip_grasp(state.plan_pose, marker, state.plan_notes)
            if state.plan_since is None:
                state.plan_since = state.stage_time
        try:
            next(state.plan)
            return False
        except StopIteration as planned:
            state.plan = None
            result = planned.value
        if self._needle_moved(state, item):
            if state.stage_time - state.plan_since > self.stage_wait_seconds:
                self._stop(state, "The needle did not come to rest for the grasp at its tip")
            return False
        if result is None:
            unreachable = state.plan_notes.get("unreachable")
            if unreachable is None:
                reason = (
                    "The needle tip does not stick out of the tissue far enough for the jaws to take it "
                    "without reaching into the tissue"
                )
            else:
                reason = (
                    "No point of the needle tip is both clear of the tissue for the jaws and within the reach "
                    f"of the tool: {unreachable}"
                )
            self._stop(state, reason)
            return False
        state.tip_grasp, state.tip_grasp_offset = result
        state.reach_rad, state.reach_m = state.plan_notes.get("reach_rad"), state.plan_notes.get("reach_m")
        state.pregrasp = self._backed_off(state.tip_grasp)
        state.route = self._plan_route(state)
        return False

    def _begin_pull(self, state, item, relative):
        """Start the pull from the grasp the jaws have just closed: its frame, its speed, its bound and its reach.

        The needle pose ``N0`` and the needle pose in the tool frame ``G`` of this observation are kept
        for the whole pull, as the arc keeps its measured grasp. The arc speed is the speed of the
        insertion, with the palm kept within the part ``f`` of the rate limits of the actions that
        the lift uses (``rate_fraction``):

            v = min( speed,  f (translation_step / frame_dt) R / r_palm,  f (rotation_step / frame_dt) R )

        ``R`` the radius of the needle and ``r_palm`` the distance of the palm origin from the
        needle's axis, from ``G``. The commanded angle is bounded by the arc that takes the tail to
        the end of the pull from where the needle is now, and the slack beyond it: out of the
        tissue, one thread element past the exit marker (:meth:`_pull_out_span`).
        The tool must reach that arc from the grasp as it is measured (:meth:`_reach`); if it does
        not, the procedure stops here, before the first target of the pull.
        """
        state.grasp = relative.copy()
        state.pull_needle = _pose(item["needle_pose"])
        state.pull_time = state.pull_angle = state.pull_wait = 0.0
        self._follow_tail(state, state.pull_needle, item)
        state.pull_bound = self._pull_out_span(state.pull_needle, item["exit_position"])
        palm = -_quat_turn(state.grasp[3:] * np.array([-1.0, -1.0, -1.0, 1.0]), state.grasp[:3])
        offset = palm - self.arc_center_local
        arm = float(np.linalg.norm(offset - (offset @ self.arc_axes[2]) * self.arc_axes[2]))
        config = self.config
        state.pull_speed = min(
            self.speed,
            self.rate_fraction * config.translation_step / config.frame_dt * self.radius / max(arm, 1e-9),
            self.rate_fraction * config.rotation_step / config.frame_dt * self.radius,
        )
        if self._reaches(
            state,
            self._pull_row(state.pull_needle, state.grasp, state.pull_bound),
            "The tool cannot pull the needle on from where its jaws closed within its joint limits",
        ):
            state.stage, state.stage_time = "pull", 0.0

    def _pull_target(self, state, item, old_target, elapsed):
        """Target of the pull: the tool turns the needle about the needle's own centre of curvature.

        Where the needle is, is read from the observation: its pose gives its tail, centre and axis,
        and the entry and exit markers are those of the deformed tissue (:meth:`_tail_past`).

        The thread of the scene follows the needle's track (``thread_tracks``) and the tail is taken
        through the tissue. The pull ends when both hold: the tissue holds no part of the track
        any more (``engaged_track_length_m`` of the observation's diagnostics is zero: the
        needle/tissue interaction lets a track sample go when the tail has passed it), and the
        tail is one thread element past the exit marker (``pull_clearance``). While a part of the
        track is still held the tail is inside the tissue, so at least that clearance is left.

        While both jaws hold the needle the commanded angle advances at the speed rule of the
        insertion at both ends: it starts with the ramp of the insertion (:func:`ramp_distance`)
        and it stops with the same ramp run backwards over the arc that the tail has left to go
        (:func:`ramp_speed`), so the needle comes to rest without a step in speed.
        The angle is bounded (:meth:`_begin_pull`); a tail that has not arrived
        ``pull_wait_seconds`` after the command has reached its bound stops the procedure: the
        needle has not come free of the tissue, or has not followed. After the pull the tool
        holds its last target for the hold time of a lift (stage ``clear``); the draw of the
        thread and the stitch follow (:meth:`_begin_draw`, :meth:`_stitch_target`).
        """
        self._follow_tail(state, item["needle_pose"], item)
        if state.stage == "clear":
            state.hold_time += elapsed
            state.status = "holding"
            if state.hold_time >= self.lift_hold_seconds - 1e-9:
                self._begin_draw(state, item, old_target)
            return old_target
        state.status = "pulling"
        held = float(item["diagnostics"].get("engaged_track_length_m", 0.0)) > 0.0
        left = self.pull_clearance - state.tail_past_exit
        if held:
            left = max(left, self.pull_clearance)
        if left <= 0.0:
            state.stage, state.pull_complete, state.hold_time, state.status = "clear", True, 0.0, "holding"
            return old_target
        # The arc of this frame: what the ramp of the insertion adds, and no more than the speed of
        # the same ramp towards its stop allows with the arc that the tail has left.
        started = float(ramp_distance(state.pull_time, state.pull_speed))
        state.pull_time += elapsed
        arc = float(ramp_distance(state.pull_time, state.pull_speed)) - started
        arc = min(arc, ramp_speed(left, state.pull_speed) * elapsed)
        state.pull_angle = min(state.pull_angle + arc / self.radius, state.pull_bound)
        if state.pull_angle >= state.pull_bound:
            state.pull_wait += elapsed
            if state.pull_wait >= self.pull_wait_seconds - 1e-9:
                if held:
                    self._stop(state, "The needle did not come free of the tissue")
                else:
                    self._stop(state, "The needle did not follow the pull out of its track")
                return old_target
        center, axis = self.arc_center_local, self.arc_axes[2]
        return pull_targets(state.pull_needle, state.grasp, center, axis, state.pull_angle)[0]

    @property
    def tighten_speed(self):
        """Speed at which the tool moves the needle on to tighten the thread [m/s].

        The speed of the insertion, with the palm, which moves straight, kept within the part
        ``rate_fraction`` of the translation rate limit of the actions, as a lift is.
        """
        return min(self.speed, self.rate_fraction * self.config.translation_step / self.config.frame_dt)

    def _follow_grip(self, state, item):
        """Keep the largest travel and turn of the needle in the jaws [m, rad]: since the tightening began and since the lock."""
        relative = _pose(item["needle_relative_pose"])
        for name, held in (("tighten", state.tighten_grasp), ("lock", state.lock_grasp)):
            if held is None:
                continue
            travel = float(np.linalg.norm(relative[:3] - held[:3]))
            turn = float(2 * np.arccos(np.clip(abs(np.dot(relative[3:], held[3:])), 0, 1)))
            setattr(state, f"{name}_slip_m", max(getattr(state, f"{name}_slip_m"), travel))
            setattr(state, f"{name}_slip_rad", max(getattr(state, f"{name}_slip_rad"), turn))

    def _stitch_target(self, state, item, old_target, elapsed, index):
        """Target of the stitch that follows the pull through the tissue: draw, hold, tie, tighten, hold, lock, slacken, hold.

        After the pull the thread lies through both passages of the track with its far end loose
        on the pad: pulled on, it only slides through. Eight stages follow the clear hold.

        ``draw``: :meth:`_draw_target`.

        ``drawn``: the tool holds its last target for the hold time of a lift.

        ``tie``: the scene is asked, beside this call's action (:meth:`commands`), to tie the
        thread at the first eyelet of its first passage, the hole through which it goes into the
        tissue; a stand-in for a knot (``GraspScene.tie_thread``). The scene ties before its next
        frame, and the observation after that frame must report one tie (``thread_tension``):
        an eyelet that the thread does not pass through is refused on the device, and the
        procedure then stops here with what the scene reports.

        ``tighten``: :meth:`_tighten_target`.

        ``hold``: the tool holds its last target for the hold time of a lift. The thread is tight
        only while the tool holds the needle: nothing ties it at the exit yet.

        ``lock``: the scene is asked, in the same way, for a second tie: at the last eyelet of the
        last planned passage, the hole through which the thread comes out of the tissue
        (``("tie_thread", {"passage": passages - 1, "last": True})``). The observation after the
        scene's next frame must report two ties, else the procedure stops here with what the scene
        reports. The thread between the two ties now holds the tissue by itself.

        ``slacken``: :meth:`_slacken_target`.

        ``hold``, again: the tool holds its last target for the same time, and the procedure is
        done. The tool still holds the needle; the thread to it is slack.
        """
        self._follow_tail(state, item["needle_pose"], item)
        tension = item.get("thread_tension") or {}
        state.ties = tension.get("ties")
        pull = tension.get("needle_force_N")
        state.thread_pull = None if pull is None else float(pull)
        volume = item["diagnostics"].get("least_volume_ratio")
        state.least_volume = None if volume is None else float(volume)
        if state.stage == "draw":
            return self._draw_target(state, item, old_target, elapsed)
        if state.stage == "drawn":
            state.hold_time += elapsed
            state.status = "holding"
            if state.hold_time >= self.lift_hold_seconds - 1e-9:
                state.stage = "tie"
            return old_target
        if state.stage == "tie":
            state.status = "tying"
            if not state.tie_asked:
                state.tie_asked = True
                self._asked[index] = (("tie_thread", {}),)
            elif state.ties != 1:
                self._stop(
                    state,
                    "The thread was not tied at its entry: the scene reports "
                    f"{state.ties} ties of {tension.get('ties_asked')} asked for",
                )
            else:
                self._begin_tighten(state, item, old_target)
            return old_target
        if state.stage == "tighten":
            return self._tighten_target(state, item, old_target, elapsed)
        if state.stage == "slacken":
            return self._slacken_target(state, item, old_target, elapsed)
        self._follow_grip(state, item)
        if state.stage == "lock":
            state.status = "tying"
            if not state.lock_asked:
                state.lock_asked = True
                self._asked[index] = (("tie_thread", {"passage": self.passages - 1, "last": True}),)
            elif state.ties != 2:
                self._stop(
                    state,
                    "The thread was not tied at its exit: the scene reports "
                    f"{state.ties} ties of {tension.get('ties_asked')} asked for",
                )
            elif state.thread_pull is None:
                self._stop(
                    state, "The scene does not report the thread's pull on the needle that the slackening stops by"
                )
            else:
                # The slackening begins with the next call, from where the tool holds the needle now.
                state.lock_grasp = _pose(item["needle_relative_pose"])
                state.slack_time = state.slack_travel = state.slack_rate = 0.0
                state.slack_samples, state.stage = [], "slacken"
            return old_target
        state.hold_time += elapsed
        state.status = "holding"
        if state.hold_time >= self.lift_hold_seconds - 1e-9:
            if state.slack_complete:
                state.stage, state.stitch_complete = "done", True
            else:
                state.stage = "lock"
        return old_target

    def _chord(self, state, item):
        """Unit vector along the chord of the needle from its tail to the point between the jaws, as observed.

        The point between the jaws is the bite point of the tool (``GRASP_BITE_DEPTH_M`` along the jaws
        from the palm). The thread hangs on the tail and comes to lie straight behind it along that
        line when the tool moves the needle on along it, so its pull on the tail points at the grasp
        and turns the needle in the jaws with no lever. Jaws that hold the needle at its tail have no
        such line: the procedure stops, and None is returned.
        """
        from ._dvrk_grasper import GRASP_BITE_DEPTH_M

        needle, palm = _pose(item["needle_pose"]), _pose(item["gripper_pose"])
        tail = _rotation(needle) @ self.centerline[-1] + needle[:3]
        bite = _rotation(palm) @ np.array([0.0, GRASP_BITE_DEPTH_M, 0.0]) + palm[:3]
        chord = bite - tail
        length = float(np.linalg.norm(chord))
        if length <= self.needle_radius:
            self._stop(state, "The jaws hold the needle at its tail: there is no chord to move it on along")
            return None
        return chord / length

    def _begin_draw(self, state, item, old_target):
        """Start the draw of the loose thread through the tissue: its direction, its length and its reach.

        The palm moves straight, without turning, along the chord of the needle from its tail to
        the point between the jaws (:meth:`_chord`), as this observation has them: the line on
        which the tightening goes on afterwards. Its length is ``draw_needle_lengths`` lengths of
        the needle, the chord from its tail to its tip.

        The tool must reach that line from its target to the end of the draw and on to the travel
        bound of the tightening (``tighten_travel_bound``; :meth:`_reach`, a pose every translation
        step of the actions); if it does not, the procedure stops here.
        """
        direction = self._chord(state, item)
        if direction is None:
            return
        length = self.draw_needle_lengths * float(np.linalg.norm(self.centerline[0] - self.centerline[-1]))
        way = length + self.tighten_travel_bound
        count = int(np.ceil(way / self.config.translation_step))
        row = np.tile(old_target, (count + 1, 1))
        row[:, :3] += np.linspace(0.0, way, count + 1)[:, None] * direction
        if not self._reaches(
            state, row, "The tool cannot draw the thread on through the tissue within its joint limits"
        ):
            return
        state.draw_origin, state.draw_direction, state.draw_length = old_target.copy(), direction, length
        state.draw_time = state.draw_travel = 0.0
        state.stage = "draw"

    def _draw_target(self, state, item, old_target, elapsed):
        """Target of the draw: the tool moves the needle on and the loose thread runs through the tissue behind it.

        The commanded travel along the direction of :meth:`_begin_draw` advances at the speed rule
        of the insertion (:attr:`tighten_speed`, :func:`ramp_distance`) and comes to rest at the
        length of the draw by the same ramp run backwards (:func:`ramp_speed`). The tool then holds
        (stage ``drawn``), and the tie follows.

        The thread is not tied yet and its far end is loose, so it takes no load: it slides through
        the eyelets of its track. A thread that does not slide pulls the tissue along instead. That
        is seen by the tightness rule of the stitch (``least_volume_ratio`` of the observation's
        diagnostics at the value that :meth:`_tighten_target` stops by; a scene that does not report
        it is not checked here), and the procedure stops there with the reason. The largest force of
        the thread on the needle that the scene reports meanwhile goes into the report.
        """
        state.status = "drawing"
        if state.thread_pull is not None:
            state.draw_pull = max(state.draw_pull or 0.0, state.thread_pull)
        if state.least_volume is not None and state.least_volume <= self.tight_volume:
            self._stop(
                state,
                f"The thread did not run through its track: {1e3 * state.draw_travel:.2f} mm into the draw the "
                f"least tissue volume is {state.least_volume} where {self.tight_volume} is a tight thread",
            )
            return old_target
        speed = self.tighten_speed
        left = state.draw_length - state.draw_travel
        if left <= 0.0:
            state.stage, state.hold_time, state.status, state.draw_complete = "drawn", 0.0, "holding", True
            return old_target
        # The travel of this frame: what the ramp of the insertion adds, no more than the speed of the
        # same ramp towards its stop allows with the travel that is left, and no more than is left.
        started = float(ramp_distance(state.draw_time, speed))
        state.draw_time += elapsed
        added = float(ramp_distance(state.draw_time, speed)) - started
        state.draw_travel += min(added, ramp_speed(left, speed) * elapsed, left)
        target = state.draw_origin.copy()
        target[:3] += state.draw_travel * state.draw_direction
        return target

    def _begin_tighten(self, state, item, old_target):
        """Start the tightening of the tied thread: its direction, where it starts and its reach.

        The palm moves straight, without turning, along the chord of the needle from its tail to
        the point between the jaws (:meth:`_chord`), both as this observation has them: after the
        draw, which came along the same line, the thread lies straight behind the tail on it.

        The tool must reach that line from its target to the travel bound
        (``tighten_travel_bound``; :meth:`_reach`, a pose every translation step of the actions);
        if it does not, or if the scene does not report the least tissue volume that the
        tightening stops by, the procedure stops here.
        """
        if state.least_volume is None:
            self._stop(state, "The scene does not report the least tissue volume that the tightening stops by")
            return
        palm = _pose(item["gripper_pose"])
        direction = self._chord(state, item)
        if direction is None:
            return
        count = int(np.ceil(self.tighten_travel_bound / self.config.translation_step))
        row = np.tile(old_target, (count + 1, 1))
        row[:, :3] += np.linspace(0.0, self.tighten_travel_bound, count + 1)[:, None] * direction
        if not self._reaches(
            state, row, "The tool cannot move the needle on to tighten the thread within its joint limits"
        ):
            return
        state.tighten_origin, state.tighten_palm, state.tighten_direction = (
            old_target.copy(),
            palm[:3].copy(),
            direction,
        )
        state.tighten_grasp = _pose(item["needle_relative_pose"])
        state.tighten_time = state.tighten_travel = 0.0
        state.tighten_samples, state.tighten_start_volume = [], state.least_volume
        state.stage = "tighten"

    def _tighten_target(self, state, item, old_target, elapsed):
        """Target of the tightening: the tool moves the needle on until the thread is as tight as the tissue allows.

        The commanded travel ``s`` along the direction of :meth:`_begin_tighten` advances at the
        speed rule of the insertion (:attr:`tighten_speed`, :func:`ramp_distance`) and comes to
        rest by the same ramp run backwards (:func:`ramp_speed`).

        The stop is the tightness rule of the thread/tissue coupling, which the suture example
        stops its own pull by: a tetrahedron of the tissue at a fraction ``J_tight`` of its rest
        volume (``least_volume_ratio`` of the observation's diagnostics, ``J``; the scene finds it
        once per finished frame). ``J_tight`` is the coupling's ``TIGHT_VOLUME_RATIO``, or the larger
        value that the scene publishes for its own tissue under the tie (``tight_volume_ratio`` of
        the geometry; the grasp scene: 0.75). A stop takes its way: from the commanded speed
        ``v u^2 (3 - 2 u)`` the ramp to rest covers

            d(u) = v T (u^3 - u^4 / 2)        (v T / 2 = 0.75 mm from the full 3 mm/s, T = RAMP_SECONDS),

        and ``J`` goes on falling over it (measured on this scene's tissue: by 0.05 over 0.75 mm).
        The rule therefore predicts where a stop begun now would end. With ``a`` the palm's own
        travel along the direction (the observed palm, which is behind its target) and ``k`` the
        slope that this tightening has measured itself over the last ``v T / 2`` of ``a``, the
        length it then extrapolates over,

            J_rest = J + k (s + d - a),        k = (J - J_w) / (a - a_w),

        and the stop ramp begins in the first frame with ``J_rest <= J_tight``, to end
        at ``s + d``. A slope is taken only over at least the travel of one frame at full speed.
        ``J`` at or below the rule value stops as well, whatever the slope.

        The travel is bounded (``tighten_travel_bound``): the command comes to rest there by the
        same ramp, and a tightening that arrives there without its rule stops the procedure with
        the reason. When the stop of the rule has ended the tool holds (stage ``hold``).
        """
        state.status = "tightening"
        self._follow_grip(state, item)
        speed, direction, volume = self.tighten_speed, state.tighten_direction, state.least_volume
        moved = float((np.asarray(item["gripper_pose"], dtype=float)[:3] - state.tighten_palm) @ direction)
        if volume is not None:
            state.tighten_samples.append((moved, volume))
        if state.tighten_end is None and volume is not None:
            # Where a stop begun now ends: the ramp to rest from the commanded speed, which the start
            # ramp has reached after as much travel; the stop at the bound may already be nearer.
            way = min(
                float(ramp_distance(min(state.tighten_time, RAMP_SECONDS), speed)),
                self.tighten_travel_bound - state.tighten_travel,
            )
            window = 0.5 * speed * RAMP_SECONDS
            first = next(sample for sample in state.tighten_samples if sample[0] >= moved - window)
            predicted, slope = volume, None
            if moved - first[0] >= speed * self.config.frame_dt:
                slope = (volume - first[1]) / (moved - first[0])
                predicted = min(volume, volume + slope * (state.tighten_travel + way - moved))
            if predicted <= self.tight_volume:
                state.tighten_end = state.tighten_travel + way
                state.tighten_predicted, state.tighten_slope = predicted, slope
                state.tighten_way = state.tighten_end - moved
        left = (self.tighten_travel_bound if state.tighten_end is None else state.tighten_end) - state.tighten_travel
        if left <= 0.0:
            if state.tighten_end is None:
                self._stop(
                    state,
                    f"The thread did not come tight: the tool moved the needle on by {1e3 * state.tighten_travel:.2f} mm, "
                    f"the bound of the tightening, and the least tissue volume is {volume} where "
                    f"{self.tight_volume} ends it",
                )
            else:
                state.stage, state.hold_time, state.status = "hold", 0.0, "holding"
            return old_target
        # The travel of this frame: what the ramp of the insertion adds, no more than the speed of the
        # same ramp towards its stop allows with the travel that is left, and no more than is left.
        started = float(ramp_distance(state.tighten_time, speed))
        state.tighten_time += elapsed
        added = float(ramp_distance(state.tighten_time, speed)) - started
        state.tighten_travel += min(added, ramp_speed(left, speed) * elapsed, left)
        target = state.tighten_origin.copy()
        target[:3] += state.tighten_travel * direction
        return target

    def _slacken_target(self, state, item, old_target, elapsed):
        """Target of the slackening: the tool takes the needle back until the thread to it is slack.

        The thread is tied at both ends of its way through the tissue. The commanded travel back
        ``b`` runs along the line of the tightening (:meth:`_begin_tighten`), towards where the
        tightening began, at its speed rule (:attr:`tighten_speed`, :func:`ramp_distance`), and
        comes to rest by the same ramp run backwards (:func:`ramp_speed`). It ends at the latest
        where the tightening began (``b`` equal to the travel of the tightening): the tool has
        come along that line, so it is not asked for reach again.

        The thread is slack when its force on the needle (``needle_force_N`` of the observation's
        ``thread_tension``, ``F``) is below the weight that the scene applies to the needle
        (``needle_weight_N`` of the geometry): the stop ramp begins in the first frame in which
        it is, from the speed that the command then has (:func:`ramp_way`). A stop takes its
        way, ``v T / 2`` = 0.75 mm from the full speed, and the thread that the needle then
        moves into pushes back: measured on this scene's tissue, the force on the needle falls
        to a least value and rises again beyond it, above the weight in the end, and the pushed
        thread loads the tied eyelet (docs/usage.md, the lock and the slackening). The command
        therefore comes to that frame slowly. With ``k`` the slope of ``F`` over the last
        ``v T / 2`` of ``b`` that this slackening has measured itself (taken only over at least
        the travel of one frame at full speed, and only while ``F`` falls), the thread is
        predicted to carry nothing at

            b_slack = b + F / (-k),

        and until the stop begins the command goes no faster than the stop ramp allows for the
        travel left to ``b_slack`` (:func:`ramp_speed`). The stop that begins where ``F`` falls
        below the weight then ends there, within the travel of a frame. Without a slope the
        command is not slowed, and the stop takes the way of the speed that it has.

        When the command is at rest the tool holds (stage ``hold``); the report has the travel
        back and what the rule found. A slackening that reaches its bound without the thread
        having gone slack also ends there: the report then has no force at which the stop began.
        """
        state.status = "slackening"
        self._follow_grip(state, item)
        speed, pull, bound = self.tighten_speed, state.thread_pull, state.tighten_travel
        slope = slack = None
        if pull is not None:
            state.slack_samples.append((state.slack_travel, pull))
            window = 0.5 * speed * RAMP_SECONDS
            first = next(sample for sample in state.slack_samples if sample[0] >= state.slack_travel - window)
            if state.slack_travel - first[0] >= speed * self.config.frame_dt:
                slope = (pull - first[1]) / (state.slack_travel - first[0])
                if slope < 0.0:
                    slack = pull / -slope
        if state.slack_end is None and pull is not None and pull < self.needle_weight:
            # The stop ramp begins, from the speed that the command has.
            state.slack_end = min(state.slack_travel + ramp_way(state.slack_rate, speed), bound)
            state.slack_stop_pull, state.slack_slope = pull, slope
        left = (bound if state.slack_end is None else state.slack_end) - state.slack_travel
        if left <= 0.0:
            state.stage, state.hold_time, state.status = "hold", 0.0, "holding"
            state.slack_complete = True
            state.slack_ended_at = float(item["diagnostics"]["physical_time_s"])
            return old_target
        # The travel of this frame: what the ramp of the insertion adds, no more than the speed of the
        # same ramp towards its stop allows with the travel that is left (to the end of a stop that has
        # begun, else to the bound and to where the thread is predicted slack), and no more than is left.
        limit = ramp_speed(left, speed)
        if state.slack_end is None and slack is not None:
            limit = min(limit, ramp_speed(slack, speed))
        started = float(ramp_distance(state.slack_time, speed))
        state.slack_time += elapsed
        added = float(ramp_distance(state.slack_time, speed)) - started
        step = min(added, limit * elapsed, left)
        state.slack_travel += step
        if elapsed > 0.0:
            state.slack_rate = step / elapsed
        target = state.tighten_origin.copy()
        target[:3] += (state.tighten_travel - state.slack_travel) * state.tighten_direction
        return target

    def commands(self):
        """What the last call asks of the scenes beside its actions: the ``commands`` of their ``step``.

        Returns:
            None when the last call asks no scene for anything, which is every call but the two
            that ask for a tie. Else one entry per observation of that call, in its order:
            None, or a tuple of ``(name, arguments)`` pairs for the scene's ``command`` method
            (``("tie_thread", {})``: tie the thread at its first entry;
            ``("tie_thread", {"passage": n - 1, "last": True})``: at its exit from the last of
            ``n`` planned passages).
        """
        return list(self._asked) if any(asked is not None for asked in self._asked) else None

    def _lift_target(self, state, elapsed):
        """Smoothly lift, hold and return the gripper after its arc."""
        state.lift_time += elapsed
        rise, hold = self.rise_seconds, self.lift_hold_seconds
        if state.lift_time < rise:
            u = state.lift_time / rise
            weight, state.status = u * u * (3 - 2 * u), "lifting"
        elif state.lift_time < rise + hold:
            weight, state.status = 1.0, "lift_hold"
        elif state.lift_time < self.lift_duration:
            u = (state.lift_time - rise - hold) / rise
            weight, state.status = 1 - u * u * (3 - 2 * u), "lowering"
        else:
            weight, state.lift_complete = 0.0, True
            state.status = "holding"
        state.lift_offset = self.lift_height * weight
        target = state.lift_origin.copy()
        target[:3] += self.lift_normal * state.lift_offset
        return target

    def __call__(self, snapshots):
        actions = np.zeros((len(snapshots), 7), dtype=np.float32)
        self._asked = [None] * len(snapshots)
        seen = set()
        for index, item in enumerate(snapshots):
            env_id = int(item["env_id"])
            if env_id in seen:
                raise ValueError("Each procedural-control snapshot must have a distinct env_id")
            seen.add(env_id)
            diagnostics = item["diagnostics"]
            now, reset_count = float(diagnostics["physical_time_s"]), int(item["reset_count"])
            if not np.isfinite(now) or now < 0:
                raise ValueError("Procedural control needs nonnegative finite physical time")
            state = self.scenes.get(env_id)
            if state is None or state.reset_count != reset_count:
                state = self.scenes[env_id] = _Scene(reset_count, now, now, self.initial_angle)
            if now < state.last_time:
                raise ValueError("Physical time moved backward without a scene reset")
            elapsed = now - state.last_time
            state.last_time = now
            state.entries = int(diagnostics.get("entry_count", 0))
            state.exits = int(diagnostics.get("exit_count", 0))
            failure = diagnostics.get("failure_reason")
            if failure or diagnostics.get("status") == "failed" or diagnostics.get("task_status") == "failed":
                self._stop(state, str(failure or "Physics reported failure"))
            if state.status == "stopped":
                continue

            relative, old_target = _pose(item["needle_relative_pose"]), _pose(item["gripper_target_pose"])
            counts = np.asarray(item["jaw_contact_count"], dtype=float)
            forces = np.asarray(item["jaw_contact_normal_force_n"], dtype=float)
            jaw_target = np.asarray(item["jaw_target_rad"], dtype=float)
            if any(value.shape != (2,) or not np.isfinite(value).all() for value in (counts, forces, jaw_target)):
                raise ValueError("Procedural control needs two finite jaw counts, forces and targets")
            contact = bool(np.all(counts > 0) and np.all(forces > self.contact_force_n))
            if state.grasp is None:
                actions[index, 6] = self._closing(jaw_target)
                if now - state.started < self.settle_seconds:
                    continue
                if not contact:
                    state.status = "waiting_for_grasp"
                    if now - state.started >= self.settle_seconds + self.grasp_wait_seconds:
                        self._stop(state, "Bilateral needle grasp was not established")
                        actions[index] = 0
                    continue
                state.grasp = relative.copy()
                state.had_contact = True
                elapsed = 0.0
                # The jaws hold the needle this far behind its tip and must stay clear of the entry.
                held = wp.transform_get_translation(wp.transform_inverse(wp.transform(*state.grasp)))
                behind = (self._arc_angle(self.tip_local) - self._arc_angle(np.asarray(held, dtype=float))) % (
                    2 * np.pi
                )
                state.drive_angle = max(
                    self.finish_angle, min(self.drive_angle, behind - self.entry_clearance / self.radius)
                )

            if state.stage in ("release", "retreat", "approach", "descend", "close"):
                # The tool has let go on purpose: the tissue holds the needle while the tool
                # moves to the tip. Jaw contact is neither expected nor a failure here.
                actions[index] = self._transfer(state, item, old_target, jaw_target, contact, elapsed, relative)
                continue
            state.relative_translation_m = float(np.linalg.norm(relative[:3] - state.grasp[:3]))
            similarity = np.clip(abs(np.dot(relative[3:], state.grasp[3:])), 0, 1)
            state.relative_rotation_rad = float(2 * np.arccos(similarity))
            if not contact:
                if state.stage in ("tie", "tighten", "hold", "lock", "slacken"):
                    state.stitch_lost_calls += 1
                if state.lost_since is None:
                    state.lost_since = now
                state.status = "grasp_lost"
                state.had_contact = False
                if now - state.lost_since >= self.contact_loss_seconds - 1e-9:
                    self._stop(state, f"Bilateral needle contact lost for {self.contact_loss_seconds:g} seconds")
                continue
            contact_elapsed = elapsed if state.had_contact else 0.0
            state.had_contact, state.lost_since = True, None
            if state.stage == "arc":
                state.progress_time += contact_elapsed
                state.angle = min(
                    self.initial_angle + float(ramp_distance(state.progress_time, self.speed)) / self.radius,
                    state.drive_angle,
                )
                state.status = "inserting"
                target = gripper_target(pose_at(self.plan, state.angle), state.grasp)
                # The lift shows the tissue deformed by the inserted needle, so it starts
                # once the commanded arc is complete. An arc angle alone does not prove
                # that the needle is in the tissue: require an actual entry as well.
                if state.angle >= state.drive_angle:
                    state.status = "holding"
                    if self.lift_height > 0 and state.entries > 0:
                        state.stage, state.lift_origin, state.lift_time = "lift", old_target.copy(), 0.0
                        state.lift_complete = False
                    elif self.pull_through and state.entries > 0:
                        # No lift is asked for: the tool lets go of the centred needle at once.
                        state.stage, state.stage_time = "release", 0.0
            elif state.stage == "lift":
                target = self._lift_target(state, contact_elapsed)
                if state.lift_complete:
                    state.lifts_complete += 1
                    if self.pull_through:
                        state.stage, state.stage_time = "release", 0.0
                    else:
                        state.stage = "done"
            elif state.stage in ("pull", "clear"):
                target = self._pull_target(state, item, old_target, contact_elapsed)
            elif state.stage in ("draw", "drawn", "tie", "tighten", "hold", "lock", "slacken"):
                target = self._stitch_target(state, item, old_target, contact_elapsed, index)
            else:
                state.status = "holding"
                target = old_target
                if state.pull_complete:
                    # The procedure is done and the tool holds the needle: the report follows the tail,
                    # and after a stitch the thread's force on the needle and the needle in the jaws.
                    self._follow_tail(state, item["needle_pose"], item)
                if state.stitch_complete:
                    pull = (item.get("thread_tension") or {}).get("needle_force_N")
                    state.thread_pull = None if pull is None else float(pull)
                    self._follow_grip(state, item)
            actions[index, :6] = action_to(old_target, target, self.config)
            actions[index, 6] = self._closing(jaw_target)
        return actions

    @staticmethod
    def _length(first, second):
        """Distance between the origins of two poses [m], or None while one of them is not known."""
        return None if first is None or second is None else float(np.linalg.norm(second[:3] - first[:3]))

    def report(self):
        """Plain diagnostics; a completed target arc is not a success claim."""
        return [
            {
                "env_id": env_id,
                "phase": state.status,
                "reason": state.reason,
                "commanded_angle_rad": state.angle,
                "commanded_arc_complete": state.angle >= self.finish_angle,
                "stage": state.stage,
                "lifts_complete": state.lifts_complete,
                "entry_count": state.entries,
                "exit_count": state.exits,
                "relative_translation_m": state.relative_translation_m,
                "relative_rotation_rad": state.relative_rotation_rad,
                "lift_started": state.lift_origin is not None,
                "lift_complete": state.lift_complete,
                "commanded_lift_m": state.lift_offset,
                "tip_grasp_offset_m": state.tip_grasp_offset,
                # Lengths of the way of the regrasp [m]: back along the open jaw, up along the shaft,
                # and from the grasp back along the jaws to the point above the pregrasp.
                "retreat_m": self._length(state.released, state.via),
                "rise_m": self._length(state.via, state.turn),
                "above_m": self._length(state.above, state.tip_grasp),
                "tip_grasp_plans": int(state.tip_grasp is not None) + state.plans_repeated,
                # The pull: commanded angle [rad], the arc by which the tail of the needle is past the
                # entry and the exit marker [m] (negative: still to come), whether the pull has ended
                # by its rule (the needle free
                # of the tissue and its tail clear of the exit), and the smallest distance from a joint limit on the
                # planned way (approach, descent, pull) of a rotary joint [rad] and of the slide [m].
                "pull_angle_rad": state.pull_angle,
                "tail_past_entry_m": state.tail_past_entry,
                "tail_past_exit_m": state.tail_past_exit,
                "pull_complete": state.pull_complete,
                "joint_margin_rad": state.reach_rad,
                "joint_margin_m": state.reach_m,
                # The draw of the loose thread through the tissue after the pull: the commanded travel and
                # its length [m] (zero until it begins), whether it has ended, and the largest force of the
                # thread on the needle that the scene reported meanwhile [N] (None where it reported none).
                "draw_travel_m": state.draw_travel,
                "draw_length_m": state.draw_length,
                "draw_complete": state.draw_complete,
                "draw_thread_force_N": state.draw_pull,
                # The stitch after the pull through the tissue: the ties that the scene reported, the
                # least tissue volume when the tightening began and as last seen, the commanded travel
                # of the tightening and its bound [m], and what its rule found when it began the stop:
                # the least volume it predicted at rest, the slope it had measured [1/m] and the way the
                # palm had left to rest [m] (all None until then; a tightening that ends at its bound
                # stops the procedure). The largest travel and turn of the needle in the jaws since the
                # tightening began [m, rad], the calls since the tie without both jaws on the needle, and
                # whether the stitch has ended as it should: tied, tightened by its rule, held, tied at
                # its exit, slackened and held.
                "ties": state.ties,
                "tighten_start_volume": state.tighten_start_volume,
                "least_volume_ratio": state.least_volume,
                "tighten_travel_m": state.tighten_travel,
                "tighten_travel_bound_m": self.tighten_travel_bound if self.thread_tracks else None,
                "tighten_stop_volume": self.tight_volume if self.thread_tracks else None,
                "tighten_predicted_volume": state.tighten_predicted,
                "tighten_slope_per_m": state.tighten_slope,
                "tighten_way_to_rest_m": state.tighten_way,
                "tighten_slip_m": state.tighten_slip_m,
                "tighten_slip_rad": state.tighten_slip_rad,
                "stitch_calls_without_grasp": state.stitch_lost_calls,
                "stitch_complete": state.stitch_complete,
                # The slackening after the tie at the exit: the commanded travel back along the line of
                # the tightening [m] (it is bounded by the travel of the tightening), the weight of the
                # needle below which the thread's force on it ends the slackening [N], the force [N] and
                # the slope [N/m] that the rule found when it began the stop (None until then, and for a
                # slackening that ended at its bound; no slope where none was measured), the physical
                # time at which the slackening ended [s], the thread's force on the needle as last seen
                # [N], and the largest travel and turn of the needle in the jaws since the tie at the
                # exit [m, rad].
                "slacken_travel_m": state.slack_travel,
                "needle_weight_N": self.needle_weight if self.thread_tracks else None,
                "slacken_stop_force_N": state.slack_stop_pull,
                "slacken_slope_N_per_m": state.slack_slope,
                "slacken_end_time_s": state.slack_ended_at,
                "thread_needle_force_N": state.thread_pull,
                "lock_slip_m": state.lock_slip_m,
                "lock_slip_rad": state.lock_slip_rad,
            }
            for env_id, state in sorted(self.scenes.items())
        ]
